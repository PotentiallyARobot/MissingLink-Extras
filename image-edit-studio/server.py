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
    def isatty(self):
        return hasattr(self.real, 'isatty') and self.real.isatty()
    def fileno(self):
        return self.real.fileno()
    def readable(self):
        return False
    def writable(self):
        return True
    def seekable(self):
        return False
    @property
    def encoding(self):
        return getattr(self.real, 'encoding', 'utf-8')
    @property
    def errors(self):
        return getattr(self.real, 'errors', 'strict')
    def __getattr__(self, name):
        return getattr(self.real, name)

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
# ══════════════════════════════════════════════════════════════
# Robust 3-tier Colab launch (health-checked, with fallbacks)
#
#   Tier 1: proxyPort URL → verified new-tab link
#   Tier 2: serve_kernel_port_as_iframe → embedded in cell
#   Tier 3: serve_kernel_port_as_window → clickable Colab link
#
# Server is never advertised until localhost health-check passes.
# ══════════════════════════════════════════════════════════════
import socket as _socket

def _find_free_port(preferred=8000):
    """Return preferred if available, otherwise an OS-assigned free port."""
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]

PORT = _find_free_port(int(os.environ.get("PORT", "8000")))

IN_COLAB = False
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    eval_js = None

def launch():
    """Call this from a Colab cell: from server import launch; launch()"""
    import traceback as _tb
    import requests as _requests

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

    # Start server in background thread (non-blocking)
    _server_error = [None]

    def _serve():
        try:
            # uvicorn's log formatter can choke on the LogCapture tee
            # that replaces sys.stdout/stderr earlier in this file.
            # Passing log_config=None disables uvicorn's own log setup
            # entirely, which sidesteps the "Unable to configure
            # formatter 'default'" crash.
            uvicorn.run(
                app, host="0.0.0.0", port=PORT,
                log_level="warning", log_config=None,
            )
        except Exception as exc:
            _server_error[0] = exc
            sys.__stderr__.write(f"\n❌ Server crashed: {exc}\n")
            _tb.print_exc(file=sys.__stderr__)

    threading.Thread(target=_serve, daemon=True).start()

    # ── 1) Wait until server is actually responding on localhost ──
    _local_ready = False
    _last_err = None
    for _attempt in range(80):  # ~40s total
        if _server_error[0] is not None:
            break
        try:
            _r = _requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=0.75)
            if _r.status_code == 200:
                _local_ready = True
                break
        except Exception as _e:
            _last_err = _e
        time.sleep(0.5)

    if not _local_ready:
        raise RuntimeError(
            f"Server never became reachable on localhost:{PORT}.\n"
            f"  Server error: {_server_error[0]}\n"
            f"  Last health-check error: {_last_err}"
        )

    sys.__stdout__.write(f"✅ Server healthy on localhost:{PORT}\n")
    print("📡 Model loading in background — UI status bar shows progress")
    print("=" * 60)

    # ── 2) Display UI (Colab vs local) ──
    _launch_mode = None
    public_url = None

    if IN_COLAB:
        from IPython.display import display, HTML as _HTML

        # ── Shared pop-out button (resolves proxy URL client-side) ──
        _POPOUT_BTN_JS = """
        <script>
        (function() {
            var btn = document.getElementById('mle-popout-btn');
            if (!btn) return;
            btn.addEventListener('click', async function() {
                btn.textContent = 'Opening...';
                btn.style.opacity = '0.6';
                try {
                    var url = await google.colab.kernel.proxyPort(%d, {cache: false});
                    if (url && !url.startsWith('http')) url = 'https://' + url;
                    window.open(url, '_blank');
                    btn.textContent = '↗ Open in new tab';
                    btn.style.opacity = '1';
                } catch(e) {
                    btn.textContent = '⚠ Failed — try again';
                    btn.style.opacity = '1';
                    console.error('proxyPort error:', e);
                }
            });
        })();
        </script>
        """ % PORT

        _POPOUT_BTN_HTML = (
            '<button id="mle-popout-btn" style="'
            'margin-left:12px;padding:8px 18px;'
            'background:#d4a017;color:#000;border:none;border-radius:8px;'
            'font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;'
            '">↗ Open in new tab</button>'
        )

        # ── Tier 1: proxyPort URL with end-to-end verification ──
        _last_proxy_err = None
        for _proxy_attempt in range(20):  # ~10s
            try:
                _candidate = eval_js(
                    f"google.colab.kernel.proxyPort({PORT}, {{'cache': false}})"
                )
                if _candidate and not _candidate.startswith("http"):
                    _candidate = "https://" + _candidate
            except Exception as _pe:
                _last_proxy_err = _pe
                _candidate = None

            if _candidate:
                try:
                    _rr = _requests.get(
                        _candidate.rstrip("/") + "/api/health", timeout=4
                    )
                    if _rr.status_code == 200:
                        public_url = _candidate
                        _launch_mode = "proxy_url"
                        break
                except Exception:
                    pass
            time.sleep(0.5)

        if _launch_mode == "proxy_url":
            display(_HTML(f"""
            <div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #d4a017;border-radius:12px;font-family:monospace;">
                <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🎨 AI Image Edit Studio is live — click to open:</div>
                <a href="{public_url}" target="_blank" style="color:#d4a017;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
                <div style="color:#8A8A8A;font-size:12px;margin-top:10px;">
                    Health: <a href="{public_url.rstrip('/')}/api/health" target="_blank"
                       style="color:#8A8A8A;text-decoration:underline;">/api/health</a>
                </div>
            </div>
            """))
        else:
            # ── Tier 2: Embed as iframe ──
            sys.__stdout__.write(
                f"⚠ Proxy URL not reachable (last error: {_last_proxy_err}).\n"
                f"  Falling back to embedded iframe...\n"
            )
            _iframe_ok = False
            try:
                from google.colab import output as _colab_output
                _colab_output.serve_kernel_port_as_iframe(PORT, height='750')
                _launch_mode = "iframe"
                _iframe_ok = True
                display(_HTML(f"""
                <div style="margin:8px 0 4px;padding:10px 16px;background:#141414;border:2px solid #d4a017;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                    <span style="color:#d4a017;font-weight:bold;">🎨 Image Edit Studio</span>
                    <span style="color:#8A8A8A;font-size:13px;"> — embedded above ↑</span>
                    {_POPOUT_BTN_HTML}
                </div>
                {_POPOUT_BTN_JS}
                """))
            except Exception as _iframe_err:
                sys.__stdout__.write(f"  ⚠ iframe failed: {_iframe_err}\n")

            # ── Tier 3: serve_kernel_port_as_window ──
            if not _iframe_ok:
                try:
                    from google.colab import output as _colab_output
                    _colab_output.serve_kernel_port_as_window(PORT, anchor_text="🎨 Click to open Image Edit Studio")
                    _launch_mode = "window"
                    display(_HTML(f"""
                    <div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #d4a017;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                        <span style="color:#8A8A8A;font-size:13px;">Click the link above to open the UI, or:</span>
                        {_POPOUT_BTN_HTML}
                    </div>
                    {_POPOUT_BTN_JS}
                    """))
                except Exception as _window_err:
                    sys.__stdout__.write(f"  ⚠ window fallback also failed: {_window_err}\n")
                    # ── Absolute last resort: raw JS iframe injection ──
                    try:
                        from IPython.display import Javascript as _JS
                        display(_JS("""
                        (async () => {
                            const url = await google.colab.kernel.proxyPort(%d, {cache: false});
                            const iframe = document.createElement('iframe');
                            iframe.src = url;
                            iframe.width = '100%%';
                            iframe.height = '750';
                            iframe.style.border = '2px solid #d4a017';
                            iframe.style.borderRadius = '12px';
                            document.querySelector('#output-area').appendChild(iframe);

                            const btn = document.createElement('button');
                            btn.textContent = '↗ Open in new tab';
                            btn.style.cssText = 'margin:8px 0;padding:8px 16px;background:#d4a017;color:#000;border:none;border-radius:6px;font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;';
                            btn.onclick = async function() {
                                try {
                                    const u = await google.colab.kernel.proxyPort(%d, {cache: false});
                                    window.open(u.startsWith('http') ? u : 'https://' + u, '_blank');
                                } catch(e) { btn.textContent = '⚠ Failed'; }
                            };
                            document.querySelector('#output-area').appendChild(btn);
                        })();
                        """ % (PORT, PORT)))
                        _launch_mode = "js_iframe"
                    except Exception as _js_err:
                        raise RuntimeError(
                            f"All Colab display methods failed for port {PORT}.\n"
                            f"  proxyPort: {_last_proxy_err}\n"
                            f"  iframe: {_iframe_err}\n"
                            f"  window: {_window_err}\n"
                            f"  JS: {_js_err}\n"
                            f"  Server IS running on localhost:{PORT}.\n"
                            f"  Try: Runtime → Disconnect and delete runtime, then reconnect."
                        )

        sys.__stdout__.write(f"🚀 Launch mode: {_launch_mode}\n")

    else:
        _launch_mode = "local"
        print(f"\n🎨 Image Edit Studio running at http://localhost:{PORT}\n")

    print("Server running. Interrupt cell to stop.\n")

    try:
        _start_ts = time.time()
        while True:
            time.sleep(30)
            _uptime = int(time.time() - _start_ts)
            _h, _rem = divmod(_uptime, 3600)
            _m, _s = divmod(_rem, 60)
            sys.__stdout__.write(
                f"\r🎨 Uptime: {_h:02d}:{_m:02d}:{_s:02d} | Port: {PORT} | Mode: {_launch_mode}   "
            )
            sys.__stdout__.flush()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")


if __name__ == "__main__":
    launch()
