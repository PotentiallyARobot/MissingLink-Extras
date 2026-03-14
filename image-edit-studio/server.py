"""
server.py — Qwen Image Edit backend (GGUF / stable_diffusion_cpp)
Run: python server.py
"""

# Disable HuggingFace Xet storage — it stalls/hangs on large GGUF files.
# Must be set before importing huggingface_hub.
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import io, sys, json, math, uuid, time, base64, subprocess
import threading, asyncio, queue
from pathlib import Path
from PIL import Image, ImageFilter
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
    """Tee stdout/stderr into log_entries while also printing to the real console.
    
    In Colab, background threads writing to sys.__stdout__ may not appear in the
    cell output unless explicitly flushed. We flush after every write.
    """
    def __init__(self, real_stream, name="stdout"):
        self.real = real_stream
        self.name = name
    def write(self, text):
        # Write to real stream (Colab cell output)
        try:
            self.real.write(text)
            self.real.flush()
        except Exception:
            pass
        # Also capture for UI console
        if text and text.strip():
            global log_counter
            with log_lock:
                log_counter += 1
                log_entries.append({"id": log_counter, "text": text.rstrip(), "ts": time.time()})
                if len(log_entries) > 500:
                    log_entries[:] = log_entries[-500:]
    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass
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

# ── Capture C-level stdout/stderr (from stable_diffusion_cpp C++ code) ──
# The C++ library writes directly to fd 1/2, bypassing Python sys.stdout.
# We replace the fd with a pipe, read from it in a thread, and tee to both
# the original fd (Colab cell output) and log_entries (UI console).
import select as _select

def _start_fd_capture():
    """Redirect C-level stdout to a pipe, tee output to Colab + log_entries."""
    _orig_fd = os.dup(1)  # save original stdout fd
    _r, _w = os.pipe()
    os.dup2(_w, 1)         # replace stdout fd with write end of pipe
    os.close(_w)

    def _reader():
        global log_counter
        buf = b''
        while True:
            try:
                chunk = os.read(_r, 4096)
            except OSError:
                break
            if not chunk:
                break
            # Write to original fd (Colab cell output)
            try:
                os.write(_orig_fd, chunk)
            except OSError:
                pass
            # Decode and add to log_entries for UI console
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                text = line.decode('utf-8', errors='replace').rstrip()
                if text:
                    with log_lock:
                        log_counter += 1
                        log_entries.append({"id": log_counter, "text": text, "ts": time.time()})
                        if len(log_entries) > 500:
                            log_entries[:] = log_entries[-500:]
        # Flush remaining
        if buf:
            text = buf.decode('utf-8', errors='replace').rstrip()
            if text:
                with log_lock:
                    log_counter += 1
                    log_entries.append({"id": log_counter, "text": text, "ts": time.time()})
        os.close(_r)
        os.close(_orig_fd)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

try:
    _start_fd_capture()
except Exception as _e:
    print(f"⚠ fd capture setup failed (non-fatal): {_e}")

# ---- State ----
status = {"status": "loading", "detail": "Starting...", "step": 0, "total": 4, "ready": False, "gpu": "", "model": "Qwen-Image-Edit-2511-GGUF"}
sd_model = None  # StableDiffusion instance
history = []
gpu_sem = threading.Semaphore(1)
wait_list_lock = threading.Lock()
wait_list = []
_gen_progress = {}  # job_id -> latest progress event (polling fallback)

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

# ---- Config loading ----
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def _load_config():
    defaults = {
        "backend": "gguf",
        "diffusion_model": {
            "repo_id": "unsloth/Qwen-Image-Edit-2511-GGUF",
            "filename": "qwen-image-edit-2511-Q4_K_M.gguf",
        },
        "llm_model": {
            "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "filename": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        },
        "vae_model": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
        },
        "offload_params_to_cpu": True,
        "diffusion_flash_attn": True,
        "lora_model_dir": "/content/loras",
        "default_settings": {
            "sample_steps": 25,
            "cfg_scale": 4.0,
            "width": 512,
            "height": 512,
            "seed": -1,
            "batch_count": 1,
        },
    }
    cfg = dict(defaults)
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                file_cfg = json.load(f)
            # Deep merge for nested dicts
            for k, v in file_cfg.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            print(f"📄 Config loaded from {_CONFIG_PATH}")
        except Exception as e:
            print(f"⚠ Failed to read {_CONFIG_PATH}: {e} — using defaults")
    return cfg

MODEL_CONFIG = _load_config()

# ---- Generation ----
def _save_temp_image(img):
    """Save a PIL image to a temp file and return the path."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path, format="PNG")
    return path

def run_generation_blocking(job_id, body, images, event_q):
    with wait_list_lock: wait_list.append(job_id)
    try:
        pos = get_queue_position(job_id)
        if pos > 0: event_q.put({"type": "queue", "position": pos, "queue_length": get_queue_length()})
        while not gpu_sem.acquire(timeout=1.0):
            pos = get_queue_position(job_id)
            if pos > 0: event_q.put({"type": "queue", "position": pos, "queue_length": get_queue_length()})
        try:
            if not sd_model or not status["ready"]:
                event_q.put({"type": "error", "error": "Model still loading."}); return

            prompt = body.get("prompt", "")
            neg_prompt = body.get("negative_prompt", "")
            steps = body.get("num_inference_steps", 25)
            cfg_scale = body.get("true_cfg_scale", 4.0)
            batch = body.get("num_images_per_prompt", 1)
            w = body.get("width") or 512
            h = body.get("height") or 512
            seed = body.get("seed", -1)
            if seed is None or int(seed) < 0:
                import random
                seed = random.randint(0, 2**31 - 1)

            # stable_diffusion_cpp needs file paths for ref images
            ref_paths = []
            for img in images:
                ref_paths.append(_save_temp_image(img))

            total_steps = steps
            gen_start = time.time()

            # stable_diffusion_cpp runs all steps internally with no per-step callback.
            # Send indeterminate progress ticks from a timer thread so the UI stays alive.
            _gen_done = threading.Event()

            def _tick_progress():
                while not _gen_done.is_set():
                    el = round(time.time() - gen_start, 1)
                    event_q.put({
                        "type": "progress",
                        "indeterminate": True,
                        "elapsed": el,
                        "step": 0,
                        "total": total_steps,
                        "progress": -1,  # signals indeterminate to frontend
                        "per_step": "?",
                        "remaining": 0,
                        "detail": f"Generating... {el}s elapsed",
                    })
                    _gen_done.wait(timeout=1.5)

            tick_thread = threading.Thread(target=_tick_progress, daemon=True)
            tick_thread.start()

            print(f"🎨 Generating: steps={steps}, cfg={cfg_scale}, batch={batch}, seed={seed}, size={w}x{h}")

            all_outputs = []
            for bi in range(batch):
                cur_seed = seed + bi

                try:
                    gen_kwargs = dict(
                        prompt=prompt,
                        ref_images=ref_paths,
                        sample_steps=int(steps),
                        cfg_scale=float(cfg_scale),
                        width=int(w),
                        height=int(h),
                        seed=int(cur_seed),
                    )
                    # Only pass negative_prompt if non-empty
                    if neg_prompt:
                        gen_kwargs["negative_prompt"] = neg_prompt

                    result_images = sd_model.generate_image(**gen_kwargs)

                    if result_images:
                        for ri in result_images:
                            all_outputs.append(ri)
                except Exception as gen_err:
                    print(f"⚠ Batch {bi+1} error: {gen_err}")
                    import traceback; traceback.print_exc()

                # Update progress per batch item
                prog = min(99, int((bi + 1) / batch * 100))
                el = time.time() - gen_start
                ps = round(el / (bi + 1), 2)
                rem = round(ps * (batch - bi - 1), 1)
                event_q.put({"type": "progress", "step": bi + 1, "total": batch, "progress": prog, "per_step": ps, "remaining": rem})

            # Stop the progress ticker
            _gen_done.set()

            # Cleanup temp files
            for p in ref_paths:
                try: os.unlink(p)
                except: pass

            if not all_outputs:
                event_q.put({"type": "error", "error": "No images generated. Check console for details."})
                return

            # Apply mask compositing if provided
            mask_b64 = body.get("mask")
            if mask_b64:
                mi = b64_to_img(mask_b64).convert("L")
                mb = int(body.get("mask_blur", 0))
                if mb > 0: mi = mi.filter(ImageFilter.GaussianBlur(radius=mb))
                orig = images[0]; comp = []
                for oi in all_outputs:
                    or2 = orig.resize(oi.size, Image.LANCZOS) if orig.size != oi.size else orig
                    comp.append(Image.composite(oi, or2, mi.resize(oi.size, Image.LANCZOS)))
                all_outputs = comp

            ob = [img_to_b64(i) for i in all_outputs]
            entry = {"id": str(uuid.uuid4()), "ts": int(time.time()), "prompt": prompt, "outputs": ob, "input_images": [img_to_b64(i) for i in images[:1]]}
            history.insert(0, entry)
            if len(history) > 50: history[:] = history[:50]
            et = round(time.time() - gen_start, 1)
            print(f"✅ Done in {et}s — {len(all_outputs)} image(s)")
            event_q.put({"type": "done", "progress": 100, "results": ob, "history_entry": entry, "elapsed": et})

        except Exception as e:
            import traceback; traceback.print_exc()
            try: _gen_done.set()
            except: pass
            import gc
            try:
                gc.collect()
                print("🧹 Cleaned up after error")
            except Exception as ce:
                print(f"⚠ Cleanup warning: {ce}")
            event_q.put({"type": "error", "error": str(e)})
        finally:
            gpu_sem.release()
    finally:
        with wait_list_lock:
            if job_id in wait_list: wait_list.remove(job_id)
        event_q.put(None)

# ---- Model loading ----
def load_model():
    global sd_model
    cfg = MODEL_CONFIG

    t0 = time.time()
    try:
        go = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], text=True).strip()
        status["gpu"] = go; print(f"🖥️  GPU: {go}")
    except Exception as e: print(f"⚠️  GPU info: {e}")

    try:
        from huggingface_hub import hf_hub_download

        # ── Step 1: Download diffusion model ──
        diff_cfg = cfg["diffusion_model"]
        status.update({"status": "loading", "step": 1, "detail": f"Downloading {diff_cfg['filename']}..."})
        print(f"⬇ Downloading diffusion model: {diff_cfg['repo_id']}/{diff_cfg['filename']}")
        diff_path = hf_hub_download(repo_id=diff_cfg["repo_id"], filename=diff_cfg["filename"])
        print(f"  ✓ {diff_path}")

        # ── Step 2: Download LLM text encoder ──
        llm_cfg = cfg["llm_model"]
        status.update({"step": 2, "detail": f"Downloading {llm_cfg['filename']}..."})
        print(f"⬇ Downloading LLM: {llm_cfg['repo_id']}/{llm_cfg['filename']}")
        llm_path = hf_hub_download(repo_id=llm_cfg["repo_id"], filename=llm_cfg["filename"])
        print(f"  ✓ {llm_path}")

        # ── Step 3: Download VAE ──
        vae_cfg = cfg["vae_model"]
        status.update({"step": 3, "detail": f"Downloading VAE..."})
        print(f"⬇ Downloading VAE: {vae_cfg['repo_id']}/{vae_cfg['filename']}")
        vae_path = hf_hub_download(repo_id=vae_cfg["repo_id"], filename=vae_cfg["filename"])
        print(f"  ✓ {vae_path}")

        # ── Step 4: Initialize stable_diffusion_cpp ──
        status.update({"step": 4, "detail": "Loading model into GPU..."})

        # Ensure LoRA directory exists
        lora_dir = cfg.get("lora_model_dir", "/content/loras")
        os.makedirs(lora_dir, exist_ok=True)

        from stable_diffusion_cpp import StableDiffusion

        print(f"🔧 Initializing StableDiffusion...")
        print(f"   diffusion_model: {diff_path}")
        print(f"   llm_path:        {llm_path}")
        print(f"   vae_path:        {vae_path}")
        print(f"   lora_dir:        {lora_dir}")

        sd_model = StableDiffusion(
            diffusion_model_path=diff_path,
            llm_path=llm_path,
            vae_path=vae_path,
            lora_model_dir=lora_dir,
            offload_params_to_cpu=cfg.get("offload_params_to_cpu", True),
            diffusion_flash_attn=cfg.get("diffusion_flash_attn", True),
        )

        el = round(time.time() - t0, 1)
        print(f"✅ Model ready ({el}s)")
        status.update({"status": "ready", "step": 4, "detail": f"Ready ({el}s startup)", "ready": True})
    except Exception as e:
        import traceback; traceback.print_exc()
        status.update({"status": "error", "detail": f"Load failed: {str(e)}"})

# ---- LoRA management ----
# stable_diffusion_cpp handles LoRAs via the <lora:name:weight> tag in prompts
# AND by placing .safetensors files in the lora_model_dir.
# So loading = download the file into lora_dir, unloading = delete it.

def _resolve_lora_path(repo):
    """Download or locate a LoRA safetensors file and place it in lora_dir."""
    from huggingface_hub import hf_hub_download, list_repo_files

    lora_dir = MODEL_CONFIG.get("lora_model_dir", "/content/loras")
    os.makedirs(lora_dir, exist_ok=True)

    if os.path.isfile(repo):
        # Already a local file — symlink/copy into lora_dir
        fname = os.path.basename(repo)
        dest = os.path.join(lora_dir, fname)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(repo, dest)
        return dest, os.path.splitext(fname)[0]

    if repo.endswith(".safetensors") and "/" in repo:
        # "user/repo/path/to/file.safetensors"
        parts = repo.split("/")
        if len(parts) >= 3:
            hf_repo = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        else:
            hf_repo = repo
            filename = None
        path = hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=lora_dir)
        name = os.path.splitext(os.path.basename(path))[0]
        return path, name

    # Bare HF repo — find a .safetensors file
    files = list_repo_files(repo)
    st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        raise RuntimeError(f"No .safetensors file found in {repo}")
    pick = next((f for f in st_files if "lora" in f.lower()), st_files[0])
    print(f"  → Auto-selected LoRA file: {pick}")
    path = hf_hub_download(repo_id=repo, filename=pick, local_dir=lora_dir)
    name = os.path.splitext(os.path.basename(path))[0]
    return path, name


def _lora_load_thread(repo, scale):
    """Download LoRA into lora_dir. The actual application happens at generation
    time via <lora:name:scale> in the prompt."""
    with lora_lock:
        lora_state["loading"] = True
        lora_state["error"] = None
    try:
        if not sd_model or not status["ready"]:
            raise RuntimeError("Model not ready yet")

        # Unload existing LoRA first
        if lora_state["loaded"]:
            print(f"🔄 Previous LoRA state cleared: {lora_state['loaded']}")

        print(f"📦 Loading LoRA: {repo} (scale={scale})")
        path, name = _resolve_lora_path(repo)
        print(f"  ✓ LoRA ready: {path}")
        print(f"  ℹ To use: prompt will auto-inject <lora:{name}:{scale}>")
        print(f"  ⚠ Note: LoRAs may not work with quantized (GGUF) models.")
        print(f"    If results look wrong, try a full-precision .safetensors base model.")

        with lora_lock:
            lora_state["loaded"] = repo
            lora_state["loading"] = False
            lora_state["error"] = None
            lora_state["_lora_name"] = name
            lora_state["_lora_path"] = path
            lora_state["_lora_scale"] = scale

        print(f"✅ LoRA loaded: {repo}")
    except Exception as e:
        import traceback; traceback.print_exc()
        with lora_lock:
            lora_state["loading"] = False
            lora_state["error"] = str(e)

def _lora_unload_thread():
    with lora_lock:
        lora_state["loading"] = True
        lora_state["error"] = None
    try:
        old = lora_state.get("loaded")
        print(f"🔄 Unloading LoRA: {old}")
        with lora_lock:
            lora_state["loaded"] = None
            lora_state["loading"] = False
            lora_state["error"] = None
            lora_state.pop("_lora_name", None)
            lora_state.pop("_lora_path", None)
            lora_state.pop("_lora_scale", None)
        print("✅ LoRA unloaded")
    except Exception as e:
        import traceback; traceback.print_exc()
        with lora_lock:
            lora_state["loading"] = False
            lora_state["error"] = str(e)

def _inject_lora_tag(prompt):
    """If a LoRA is loaded, auto-append <lora:name:scale> to the prompt
    (unless the user already included a <lora:...> tag)."""
    with lora_lock:
        if not lora_state.get("loaded"):
            return prompt
        name = lora_state.get("_lora_name", "")
        scale = lora_state.get("_lora_scale", 1.0)
    if not name:
        return prompt
    # Don't double-inject if user already typed a lora tag
    if f"<lora:" in prompt:
        return prompt
    return f"{prompt} <lora:{name}:{scale}>"


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
    return {**status, "queue_length": get_queue_length(), "lora": {k: v for k, v in lora_state.items() if not k.startswith("_")}}

@app.post("/api/generate")
async def api_generate(request: Request):
    body = await request.json()
    imgs_d = body.get("images", {})
    images = [b64_to_img(imgs_d[k]) for k in sorted(imgs_d.keys(), key=lambda k: int(k))]
    if not images:
        async def e(): yield f"data: {json.dumps({'type':'error','error':'Upload at least one image.'})}\n\n"
        return StreamingResponse(e(), media_type="text/event-stream")
    if not sd_model or not status["ready"]:
        async def e(): yield f"data: {json.dumps({'type':'error','error':'Model still loading.'})}\n\n"
        return StreamingResponse(e(), media_type="text/event-stream")

    # Auto-inject LoRA tag into prompt
    body["prompt"] = _inject_lora_tag(body.get("prompt", ""))

    jid = str(uuid.uuid4()); eq = queue.Queue()
    _gen_progress[jid] = {"type": "progress", "step": 0, "total": 0, "progress": 0}

    threading.Thread(target=run_generation_blocking, args=(jid, body, images, eq), daemon=True).start()

    _SSE_PAD = ": " + "x" * 2048 + "\n"

    async def stream():
        lt = time.time()
        yield _SSE_PAD + _SSE_PAD + f"data: {json.dumps({'type': 'init', 'job_id': jid})}\n\n"
        while True:
            try: ev = eq.get_nowait()
            except queue.Empty:
                if time.time() - lt >= 1.5:
                    yield _SSE_PAD + ": keepalive\n\n"
                    lt = time.time()
                await asyncio.sleep(0.15); continue
            if ev is None:
                _gen_progress[jid] = {"type": "done"}
                break
            _gen_progress[jid] = ev
            lt = time.time()
            yield _SSE_PAD + f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "Content-Type": "text/event-stream"})

@app.get("/api/queue")
async def api_queue():
    ql = get_queue_length(); return {"queue_length": ql, "busy": ql > 0 or not gpu_sem._value}

@app.get("/api/gen_progress/{job_id}")
async def api_gen_progress(job_id: str):
    p = _gen_progress.get(job_id)
    if p is None:
        return {"type": "unknown"}
    return p

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
    if not sd_model or not status["ready"]: return {"ok": False, "error": "Model not ready"}
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
    return {k: v for k, v in lora_state.items() if not k.startswith("_")}

# ---- CivitAI LoRA download ----
import requests as _http_requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

CIVITAI_LORA_DIR = Path(MODEL_CONFIG.get("lora_model_dir", "/content/loras"))
CIVITAI_LORA_DIR.mkdir(parents=True, exist_ok=True)

def _validate_safetensors(path):
    """Check if a file looks like a valid safetensors file."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if len(header) < 8:
            return False, "File too small"
        import struct
        header_size = struct.unpack("<Q", header[:8])[0]
        if header_size > 50_000_000:
            return False, "Header too large — probably not safetensors"
        if header_size == 0:
            return False, "Header size is 0 — corrupt file"
        try:
            text_start = header.decode("utf-8", errors="ignore").lower()
            if "<html" in text_start or "<!doc" in text_start:
                return False, "File is HTML, not safetensors"
        except Exception:
            pass
        return True, "ok"
    except Exception as e:
        return False, str(e)

@app.post("/api/lora/download_civitai")
async def api_download_civitai(request: Request):
    body = await request.json()
    url = body.get("url", "")
    filename = body.get("filename", "")
    token = body.get("token", "")
    if not url:
        return {"error": "Missing download URL"}

    if not filename:
        filename = f"civitai_lora_{body.get('civitai_id', 'unknown')}.safetensors"
    filename = filename.replace("/", "_").replace("\\", "_")

    out_path = CIVITAI_LORA_DIR / filename

    if out_path.exists():
        if out_path.stat().st_size > 1024:
            valid, reason = _validate_safetensors(str(out_path))
            if valid:
                print(f"📦 CivitAI LoRA already cached: {out_path}")
                return {"ok": True, "path": str(out_path), "filename": filename, "cached": True}
            else:
                print(f"⚠ Cached file invalid ({reason}), re-downloading...")
                out_path.unlink()
        else:
            out_path.unlink()

    token = token.strip() if token else os.environ.get("CIVITAI_API_KEY", "")
    if token:
        parts = urlparse(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["token"] = token
        url = urlunparse((parts.scheme, parts.netloc, parts.path,
                          parts.params, urlencode(query), parts.fragment))

    try:
        print(f"⬇ Downloading CivitAI LoRA: {filename}")
        headers = {
            "User-Agent": "Mozilla/5.0 (CivitAI-Download)",
            "Referer": "https://civitai.com/",
            "Accept": "*/*",
        }
        with _http_requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "text/html" in ct.lower():
                return {"error": "CivitAI returned HTML — you likely need a CivitAI API token."}
            cd = r.headers.get("content-disposition", "")
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip().strip('"')
                if fname:
                    filename = fname.replace("/", "_").replace("\\", "_")
                    out_path = CIVITAI_LORA_DIR / filename
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(str(out_path), "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            if pct % 20 == 0:
                                print(f"  ⬇ {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)")

        valid, reason = _validate_safetensors(str(out_path))
        if not valid:
            print(f"⚠ Downloaded file is not valid safetensors: {reason}")
            if out_path.exists():
                out_path.unlink()
            return {"error": f"Downloaded file is not valid: {reason}. You may need a CivitAI API token."}

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"✅ Downloaded: {out_path} ({size_mb:.1f}MB)")
        return {"ok": True, "path": str(out_path), "filename": filename}
    except _http_requests.exceptions.HTTPError as he:
        if he.response is not None and he.response.status_code == 401:
            return {"error": "CivitAI returned 401 Unauthorized."}
        return {"error": f"Download failed: HTTP {he.response.status_code if he.response else '?'} — {str(he)}"}
    except Exception as e:
        import traceback; traceback.print_exc()
        if out_path.exists():
            try: out_path.unlink()
            except: pass
        return {"error": f"Download failed: {str(e)}"}

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
import socket as _socket

def _find_free_port(preferred=8000):
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
    print("🎨 AI Image Edit Studio — GGUF Backend")
    print("   Powered by stable_diffusion_cpp")
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

    # Start server
    _server_error = [None]

    def _serve():
        try:
            uvicorn.run(
                app, host="0.0.0.0", port=PORT,
                log_level="warning", log_config=None,
            )
        except Exception as exc:
            _server_error[0] = exc
            sys.__stderr__.write(f"\n❌ Server crashed: {exc}\n")
            _tb.print_exc(file=sys.__stderr__)

    threading.Thread(target=_serve, daemon=True).start()

    # Wait until server is responding
    _local_ready = False
    _last_err = None
    for _attempt in range(80):
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

    # Display UI (Colab vs local)
    _launch_mode = None
    public_url = None

    if IN_COLAB:
        from IPython.display import display, HTML as _HTML

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

        # Tier 1: proxyPort URL
        _last_proxy_err = None
        for _proxy_attempt in range(20):
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
                        )

        sys.__stdout__.write(f"🚀 Launch mode: {_launch_mode}\n")

    else:
        _launch_mode = "local"
        print(f"\n🎨 Image Edit Studio running at http://localhost:{PORT}\n")

    print("Server running in background. You can run other cells.\n")
    print("  ⚡ The server stays alive as long as this runtime is active.")
    print("  🛑 To stop: Runtime → Disconnect and delete runtime\n")

    # Return immediately — server and model-loading threads are daemonic
    # and will keep running in the background.
    return


if __name__ == "__main__":
    launch()
    # When run as a script (not from Colab), block so the process doesn't exit
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
