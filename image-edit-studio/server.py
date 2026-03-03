"""
server.py — Qwen Image Edit backend for Colab
Run: python server.py
"""

import io, os, sys, json, math, uuid, time, base64, subprocess
import threading, asyncio, queue
from pathlib import Path
from PIL import Image, ImageFilter
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

# ---- Log capture ----
log_lock = threading.Lock()
log_entries = []  # [{"id": int, "text": str, "ts": float}]
log_counter = 0

class LogCapture:
    """Tee stdout/stderr into log_entries while still printing to real console."""
    def __init__(self, real_stream, name="stdout"):
        self.real = real_stream
        self.name = name
    def write(self, text):
        self.real.write(text)
        if text.strip():
            global log_counter
            with log_lock:
                log_counter += 1
                log_entries.append({"id": log_counter, "text": text.rstrip(), "ts": time.time()})
                if len(log_entries) > 500:
                    log_entries[:] = log_entries[-500:]
    def flush(self):
        self.real.flush()

sys.stdout = LogCapture(sys.__stdout__, "stdout")
sys.stderr = LogCapture(sys.__stderr__, "stderr")

# ---- State ----
status = {"status": "loading", "detail": "Starting...", "step": 0, "total": 5, "ready": False, "gpu": "", "model": "Qwen-Image-Edit-2511"}
pipeline = None
history = []
gpu_sem = threading.Semaphore(1)
wait_list_lock = threading.Lock()
wait_list = []

# ---- LoRA state ----
lora_lock = threading.Lock()
lora_state = {"loaded": None, "loading": False, "error": None}

def get_queue_position(jid):
    with wait_list_lock:
        try: return wait_list.index(jid)
        except ValueError: return -1

def get_queue_length():
    with wait_list_lock: return len(wait_list)

def img_to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def b64_to_img(data):
    if data.startswith("data:"): _, data = data.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

# ---- Generation ----
def run_generation_blocking(job_id, body, images, event_q):
    with wait_list_lock: wait_list.append(job_id)
    try:
        pos = get_queue_position(job_id)
        if pos > 0: event_q.put({"type": "queue", "position": pos, "queue_length": get_queue_length()})
        while not gpu_sem.acquire(timeout=1.0):
            pos = get_queue_position(job_id)
            if pos > 0: event_q.put({"type": "queue", "position": pos, "queue_length": get_queue_length()})
        try:
            if not pipeline or not status["ready"]:
                event_q.put({"type": "error", "error": "Model still loading."}); return
            steps = body.get("num_inference_steps", 25)
            cfg = body.get("true_cfg_scale", 4.0)
            pk = {
                "image": images if len(images) > 1 else images[0],
                "prompt": body.get("prompt", ""),
                "num_inference_steps": steps,
                "true_cfg_scale": cfg,
                "num_images_per_prompt": body.get("num_images_per_prompt", 1),
            }
            if body.get("negative_prompt"): pk["negative_prompt"] = body["negative_prompt"]
            gs = body.get("guidance_scale")
            if gs is not None: pk["guidance_scale"] = gs
            msl = body.get("max_sequence_length")
            if msl: pk["max_sequence_length"] = msl
            w = body.get("width"); hv = body.get("height")
            if w and hv: pk["width"] = int(w); pk["height"] = int(hv)
            seed = body.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                pk["generator"] = torch.Generator("cpu").manual_seed(int(seed))

            total_steps = steps
            if cfg > 1.0 and body.get("negative_prompt"): total_steps = steps * 2

            gen_start = time.time()

            def step_cb(po, si, ts, ck):
                el = time.time() - gen_start
                ps = round(el / (si + 1), 2)
                rem = round(ps * (total_steps - si - 1), 1)
                prog = min(99, int((si + 1) / total_steps * 100))
                event_q.put({"type": "progress", "step": si + 1, "total": total_steps, "progress": prog, "per_step": ps, "remaining": rem})
                return ck

            pk["callback_on_step_end"] = step_cb
            event_q.put({"type": "progress", "step": 0, "total": total_steps, "progress": 0, "per_step": "?", "remaining": 0})

            print(f"🎨 Generating: steps={steps}, cfg={cfg}, batch={pk.get('num_images_per_prompt',1)}")
            result = pipeline(**pk)
            out_images = result.images

            mask_b64 = body.get("mask")
            if mask_b64:
                mi = b64_to_img(mask_b64).convert("L")
                mb = int(body.get("mask_blur", 0))
                if mb > 0: mi = mi.filter(ImageFilter.GaussianBlur(radius=mb))
                orig = images[0]; comp = []
                for oi in out_images:
                    or2 = orig.resize(oi.size, Image.LANCZOS) if orig.size != oi.size else orig
                    comp.append(Image.composite(oi, or2, mi.resize(oi.size, Image.LANCZOS)))
                out_images = comp

            ob = [img_to_b64(i) for i in out_images]
            entry = {"id": str(uuid.uuid4()), "ts": int(time.time()), "prompt": body.get("prompt", ""), "outputs": ob, "input_images": [img_to_b64(i) for i in images[:1]]}
            history.insert(0, entry)
            if len(history) > 50: history[:] = history[:50]
            et = round(time.time() - gen_start, 1)
            print(f"✅ Done in {et}s")
            event_q.put({"type": "done", "progress": 100, "results": ob, "history_entry": entry, "elapsed": et})

        except Exception as e:
            import traceback; traceback.print_exc()
            event_q.put({"type": "error", "error": str(e)})
        finally:
            gpu_sem.release()
    finally:
        with wait_list_lock:
            if job_id in wait_list: wait_list.remove(job_id)
        event_q.put(None)

# ---- Model loading ----
def load_model():
    global pipeline
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from nunchaku import NunchakuQwenImageTransformer2DModel
    from nunchaku.utils import get_precision

    t0 = time.time()
    cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    try:
        go = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], text=True).strip()
        status["gpu"] = go; print(f"🖥️  GPU: {go}")
        print(f"🖥️  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
    except Exception as e: print(f"⚠️  GPU info: {e}")

    try:
        status.update({"status": "loading", "step": 1, "detail": "Loading quantized transformer..."})
        RANK = 256; precision = get_precision()
        if "int4" in precision: precision = "int4"
        elif "fp4" in precision: precision = "fp4"
        vbr = {32: "ultimate_speed", 128: "balance", 256: "best_quality"}
        variant = vbr.get(RANK, "balance")
        hfp = f"QuantFunc/Nunchaku-Qwen-Image-EDIT-2511/nunchaku_qwen_image_edit_2511_{variant}_{precision}.safetensors"
        print(f"Loading transformer: {hfp}")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(hfp)

        status.update({"step": 2, "detail": "Building scheduler..."})
        scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
            "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
            "use_exponential_sigmas": False, "use_karras_sigmas": False,
        })

        status.update({"step": 3, "detail": "Loading text encoder + VAE..."})
        print("Loading pipeline: Qwen/Qwen-Image-Edit-2511")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511", transformer=transformer, scheduler=scheduler,
            torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        )

        status.update({"step": 4, "detail": "Enabling CPU offload..."})
        pipeline.enable_model_cpu_offload()

        status.update({"step": 5, "detail": "Finalizing..."})
        el = round(time.time() - t0, 1)
        print(f"✅ Pipeline ready ({el}s)")
        status.update({"status": "ready", "step": 5, "detail": f"Ready ({el}s startup)", "ready": True})
    except Exception as e:
        import traceback; traceback.print_exc()
        status.update({"status": "error", "detail": f"Load failed: {str(e)}"})

# ---- LoRA management ----
def _lora_load_thread(repo, scale):
    global pipeline
    with lora_lock:
        lora_state["loading"] = True
        lora_state["error"] = None
    try:
        if not pipeline or not status["ready"]:
            raise RuntimeError("Model not ready yet")

        # Unload existing LoRA first
        if lora_state["loaded"]:
            print(f"🔄 Unloading existing LoRA: {lora_state['loaded']}")
            try:
                pipeline.unload_lora_weights()
            except Exception as ue:
                print(f"⚠️  Unload warning: {ue}")

        print(f"📦 Loading LoRA: {repo} (scale={scale})")
        pipeline.load_lora_weights(repo)
        pipeline.fuse_lora(lora_scale=scale)
        print(f"✅ LoRA loaded and fused: {repo}")

        with lora_lock:
            lora_state["loaded"] = repo
            lora_state["loading"] = False
            lora_state["error"] = None
    except Exception as e:
        import traceback; traceback.print_exc()
        with lora_lock:
            lora_state["loading"] = False
            lora_state["error"] = str(e)

def _lora_unload_thread():
    global pipeline
    with lora_lock:
        lora_state["loading"] = True
        lora_state["error"] = None
    try:
        if not pipeline:
            raise RuntimeError("Model not ready")
        print(f"🔄 Unfusing and unloading LoRA: {lora_state['loaded']}")
        try:
            pipeline.unfuse_lora()
        except Exception:
            pass
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
        print("✅ LoRA unloaded")
        with lora_lock:
            lora_state["loaded"] = None
            lora_state["loading"] = False
            lora_state["error"] = None
    except Exception as e:
        import traceback; traceback.print_exc()
        with lora_lock:
            lora_state["loading"] = False
            lora_state["error"] = str(e)

# ---- FastAPI ----
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/health")
async def health():
    return {**status, "queue_length": get_queue_length(), "lora": {**lora_state}}

@app.post("/api/generate")
async def api_generate(request: Request):
    body = await request.json()
    imgs_d = body.get("images", {})
    images = [b64_to_img(imgs_d[k]) for k in sorted(imgs_d.keys(), key=lambda k: int(k))]
    if not images:
        async def e(): yield f"data: {json.dumps({'type':'error','error':'Upload at least one image.'})}\n\n"
        return StreamingResponse(e(), media_type="text/event-stream")
    if not pipeline or not status["ready"]:
        async def e(): yield f"data: {json.dumps({'type':'error','error':'Model still loading.'})}\n\n"
        return StreamingResponse(e(), media_type="text/event-stream")
    jid = str(uuid.uuid4()); eq = queue.Queue()
    threading.Thread(target=run_generation_blocking, args=(jid, body, images, eq), daemon=True).start()
    async def stream():
        lt = time.time()
        while True:
            try: ev = eq.get_nowait()
            except queue.Empty:
                if time.time() - lt >= 2: yield ": keepalive\n\n"; lt = time.time()
                await asyncio.sleep(0.15); continue
            if ev is None: break
            lt = time.time(); yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

@app.get("/api/queue")
async def api_queue():
    ql = get_queue_length(); return {"queue_length": ql, "busy": ql > 0 or not gpu_sem._value}

@app.get("/api/history")
async def api_history():
    return {"history": history}

@app.post("/api/history/delete")
async def api_history_delete(request: Request):
    body = await request.json(); history[:] = [h for h in history if h["id"] != body.get("id", "")]; return {"ok": True}

# ---- LoRA endpoints ----
@app.post("/api/lora/load")
async def api_lora_load(request: Request):
    body = await request.json()
    repo = body.get("repo", "").strip()
    scale = float(body.get("scale", 1.0))
    if not repo: return {"ok": False, "error": "Missing repo"}
    if lora_state["loading"]: return {"ok": False, "error": "Already loading a LoRA"}
    if not pipeline or not status["ready"]: return {"ok": False, "error": "Model not ready"}
    # Acquire GPU sem so we don't load during generation
    if not gpu_sem.acquire(timeout=0.1):
        return {"ok": False, "error": "GPU busy — wait for generation to finish"}
    try:
        _lora_load_thread(repo, scale)
    finally:
        gpu_sem.release()
    if lora_state["error"]:
        return {"ok": False, "error": lora_state["error"]}
    return {"ok": True, "loaded": repo}

@app.post("/api/lora/unload")
async def api_lora_unload():
    if not lora_state["loaded"]: return {"ok": True, "message": "No LoRA loaded"}
    if lora_state["loading"]: return {"ok": False, "error": "LoRA operation in progress"}
    if not gpu_sem.acquire(timeout=0.1):
        return {"ok": False, "error": "GPU busy"}
    try:
        _lora_unload_thread()
    finally:
        gpu_sem.release()
    if lora_state["error"]:
        return {"ok": False, "error": lora_state["error"]}
    return {"ok": True}

@app.get("/api/lora/status")
async def api_lora_status():
    return {**lora_state}

# ---- Log endpoints ----
@app.get("/api/logs")
async def api_logs(after: int = 0):
    with log_lock:
        filtered = [l for l in log_entries if l["id"] > after]
    return {"logs": filtered[-200:]}

@app.post("/api/logs/clear")
async def api_logs_clear():
    with log_lock:
        log_entries.clear()
    return {"ok": True}

# ---- Launch ----
PORT = int(os.environ.get("PORT", "8000"))

def launch():
    """Call this from a Colab cell: from server import launch; launch()"""
    print("=" * 60)
    print("🎨 AI Image Edit Studio — Missing Link")
    print("=" * 60)

    # Load model in background
    threading.Thread(target=load_model, daemon=True).start()

    # Keepalive thread
    def _keepalive():
        import urllib.request
        url = f"http://127.0.0.1:{PORT}/api/health"
        while True:
            time.sleep(30)
            try: urllib.request.urlopen(url, timeout=5)
            except: pass
    threading.Thread(target=_keepalive, daemon=True).start()
    print("💓 Keepalive thread started")

    # Start server in background thread (non-blocking)
    def _serve():
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(2)
    print(f"🚀 Server running on port {PORT}")

    # Build Colab proxy URL
    from google.colab import output as colab_output
    from IPython.display import display, HTML as IPHTML, Javascript

    # Use Javascript to resolve the proxy URL inside the notebook kernel
    # then write it into the output cell
    display(IPHTML(f'''
    <div id="ml-launch" style="margin:8px 0">
      <div style="font-family:monospace;font-size:13px;color:#888;margin-bottom:8px" id="ml-url-box">
        Resolving Colab proxy URL...
      </div>
    </div>
    <script>
    (async () => {{
      try {{
        const url = await google.colab.kernel.proxyPort({PORT}, {{"cache": true}});
        const box = document.getElementById('ml-url-box');
        box.innerHTML = '🌐 <a href="' + url + '" target="_blank" style="color:#d4a017;font-weight:bold">' + url + '</a>';
        // Insert the open button
        const btn = document.createElement('a');
        btn.href = url;
        btn.target = '_blank';
        btn.style.cssText = 'display:inline-block;padding:10px 24px;background:#d4a017;color:#000;font-family:sans-serif;font-weight:700;font-size:14px;border-radius:8px;text-decoration:none;margin:8px 0';
        btn.textContent = '🚀 Open AI Image Edit Studio';
        box.parentElement.insertBefore(btn, box);
        // Insert iframe
        const iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.width = '100%';
        iframe.height = '750';
        iframe.frameBorder = '0';
        iframe.style.cssText = 'border:1px solid #333;border-radius:8px;margin-top:8px';
        iframe.allow = 'clipboard-read; clipboard-write';
        box.parentElement.appendChild(iframe);
      }} catch(e) {{
        document.getElementById('ml-url-box').innerHTML = '⚠️ Could not resolve proxy URL: ' + e.message + '<br>Try opening: <a href="http://localhost:{PORT}" target="_blank">http://localhost:{PORT}</a>';
      }}
    }})();
    </script>
    '''))

    print("📡 Model loading in background — UI status bar shows progress")
    print("=" * 60)


if __name__ == "__main__":
    # Fallback: if run directly as a script (not from notebook), just block
    launch()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

