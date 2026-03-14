"""
server.py — Qwen Image Edit backend (diffusers + GGUF)
Run: python server.py
"""

import os
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import io, sys, json, math, uuid, time, base64, subprocess
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
log_entries = []
log_counter = 0

class LogCapture:
    def __init__(self, real_stream, name="stdout"):
        self.real = real_stream
        self.name = name
    def write(self, text):
        try:
            self.real.write(text)
            self.real.flush()
        except Exception:
            pass
        if text and text.strip():
            global log_counter
            with log_lock:
                log_counter += 1
                log_entries.append({"id": log_counter, "text": text.rstrip(), "ts": time.time()})
                if len(log_entries) > 500:
                    log_entries[:] = log_entries[-500:]
    def flush(self):
        try: self.real.flush()
        except: pass
    def isatty(self):
        return hasattr(self.real, 'isatty') and self.real.isatty()
    def fileno(self):
        return self.real.fileno()
    def readable(self): return False
    def writable(self): return True
    def seekable(self): return False
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
status = {"status": "loading", "detail": "Starting...", "step": 0, "total": 4, "ready": False, "gpu": "", "model": "Qwen-Image-Edit-2511-GGUF"}
pipeline = None
history = []
gpu_sem = threading.Semaphore(1)
wait_list_lock = threading.Lock()
wait_list = []
_gen_progress = {}

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

# ---- Config ----
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def _load_config():
    defaults = {
        "local_dir": "/content/models",
        "diffusion_model": {
            "repo_id": "unsloth/Qwen-Image-Edit-2511-GGUF",
            "filename": "qwen-image-edit-2511-Q4_0.gguf",
        },
        "pipeline_repo": "Qwen/Qwen-Image-Edit-2511",
        "lora_model_dir": "/content/loras",
        "default_settings": {
            "sample_steps": 25,
            "cfg_scale": 4.0,
            "width": 512,
            "height": 512,
        },
    }
    cfg = dict(defaults)
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                file_cfg = json.load(f)
            for k, v in file_cfg.items():
                if k.startswith("_"): continue
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            print(f"📄 Config loaded from {_CONFIG_PATH}")
        except Exception as e:
            print(f"⚠ Failed to read {_CONFIG_PATH}: {e}")
    return cfg

MODEL_CONFIG = _load_config()

# ---- Generation (diffusers pipeline with per-step callback) ----
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
            if body.get("negative_prompt"):
                pk["negative_prompt"] = body["negative_prompt"]
            gs = body.get("guidance_scale")
            if gs is not None: pk["guidance_scale"] = gs
            msl = body.get("max_sequence_length") or 512
            pk["max_sequence_length"] = int(msl)
            w = body.get("width"); hv = body.get("height")
            if w and hv: pk["width"] = int(w); pk["height"] = int(hv)
            seed = body.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                pk["generator"] = torch.Generator("cpu").manual_seed(int(seed))

            total_steps = steps
            if cfg > 1.0 and body.get("negative_prompt"):
                total_steps = steps * 2

            gen_start = time.time()

            def step_cb(po, si, ts, ck):
                el = time.time() - gen_start
                ps = round(el / (si + 1), 2)
                rem = round(ps * (total_steps - si - 1), 1)
                prog = min(99, int((si + 1) / total_steps * 100))
                event_q.put({"type": "progress", "step": si + 1, "total": total_steps,
                             "progress": prog, "per_step": ps, "remaining": rem})
                return ck

            pk["callback_on_step_end"] = step_cb
            event_q.put({"type": "progress", "step": 0, "total": total_steps,
                         "progress": 0, "per_step": "?", "remaining": 0})

            print(f"🎨 Generating: steps={steps}, cfg={cfg}, batch={pk.get('num_images_per_prompt',1)}, max_seq_len={msl}")
            result = pipeline(**pk)
            out_images = result.images

            # Mask compositing
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
            entry = {"id": str(uuid.uuid4()), "ts": int(time.time()),
                     "prompt": body.get("prompt", ""), "outputs": ob,
                     "input_images": [img_to_b64(i) for i in images[:1]]}
            history.insert(0, entry)
            if len(history) > 50: history[:] = history[:50]
            et = round(time.time() - gen_start, 1)
            print(f"✅ Done in {et}s")
            event_q.put({"type": "done", "progress": 100, "results": ob,
                         "history_entry": entry, "elapsed": et})

        except Exception as e:
            import traceback; traceback.print_exc()
            import gc
            try:
                if pipeline is not None and hasattr(pipeline, '_all_hooks'):
                    for hook in pipeline._all_hooks:
                        if hasattr(hook, 'offload'):
                            try: hook.offload()
                            except: pass
                torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()
                print("🧹 GPU memory cleaned up after error")
            except Exception as ce:
                print(f"⚠ Cleanup warning: {ce}")
            event_q.put({"type": "error", "error": str(e)})
        finally:
            gpu_sem.release()
    finally:
        with wait_list_lock:
            if job_id in wait_list: wait_list.remove(job_id)
        event_q.put(None)

# ---- Model downloading & loading ----
_model_paths = {}

def _download_model(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1024:
        sz_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  ✓ Already downloaded ({sz_mb:.0f} MB): {local_path}")
        return local_path
    full_path = os.path.join(local_dir, filename)
    if os.path.isfile(full_path) and os.path.getsize(full_path) > 1024:
        sz_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  ✓ Already downloaded ({sz_mb:.0f} MB): {full_path}")
        return full_path
    print(f"  Downloading...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    sz_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✓ Downloaded ({sz_mb:.0f} MB): {path}")
    return path


def download_models():
    """Download GGUF transformer. Runs BLOCKING before server starts."""
    cfg = MODEL_CONFIG
    local_dir = cfg.get("local_dir", "/content/models")
    os.makedirs(local_dir, exist_ok=True)
    t0 = time.time()

    try:
        go = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"], text=True).strip()
        status["gpu"] = go
        print(f"🖥️  GPU: {go}")
        print(f"🖥️  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
    except Exception as e:
        print(f"⚠️  GPU info: {e}")

    print("\n📦 Downloading GGUF transformer...\n")
    diff_cfg = cfg["diffusion_model"]
    status.update({"status": "loading", "step": 1, "total": 3, "detail": f"Downloading {diff_cfg['filename']}..."})
    print(f"[1/1] GGUF model: {diff_cfg['repo_id']}/{diff_cfg['filename']}")
    _model_paths["gguf"] = _download_model(diff_cfg["repo_id"], diff_cfg["filename"], local_dir)

    el = round(time.time() - t0, 1)
    print(f"\n✅ GGUF downloaded ({el}s)")
    print("   (Pipeline components like text encoder + VAE download automatically from HuggingFace)\n")


def init_model():
    """Load GGUF transformer into diffusers pipeline. Runs in background."""
    global pipeline
    cfg = MODEL_CONFIG
    t0 = time.time()

    try:
        status.update({"status": "loading", "step": 2, "total": 3, "detail": "Loading GGUF transformer..."})

        from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig, QwenImageEditPlusPipeline

        gguf_path = _model_paths["gguf"]
        pipeline_repo = cfg.get("pipeline_repo", "Qwen/Qwen-Image-Edit-2511")

        print(f"🔧 Loading GGUF transformer: {gguf_path}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ Transformer loaded")

        status.update({"step": 3, "detail": "Loading pipeline (text encoder + VAE)..."})
        print(f"🔧 Loading pipeline: {pipeline_repo}")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            pipeline_repo,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        print("  ✓ Pipeline loaded")

        print("🔧 Enabling CPU offload...")
        pipeline.enable_model_cpu_offload()

        el = round(time.time() - t0, 1)
        print(f"✅ Model ready ({el}s)")
        status.update({"status": "ready", "step": 3, "detail": f"Ready ({el}s)", "ready": True})
    except Exception as e:
        import traceback; traceback.print_exc()
        status.update({"status": "error", "detail": f"Load failed: {str(e)}"})

# ---- LoRA management (native diffusers) ----
def _resolve_lora_source(repo):
    """Resolve a LoRA repo/path to arguments for pipeline.load_lora_weights()."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if os.path.isfile(repo):
        return {"pretrained_model_name_or_path_or_dict": repo}

    if repo.endswith(".safetensors") and "/" in repo:
        parts = repo.split("/")
        if len(parts) >= 3:
            hf_repo = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            return {"pretrained_model_name_or_path_or_dict": hf_repo, "weight_name": filename}

    # Bare HF repo — find a .safetensors file
    try:
        files = list_repo_files(repo)
        st_files = [f for f in files if f.endswith(".safetensors")]
        if st_files:
            pick = next((f for f in st_files if "lora" in f.lower()), st_files[0])
            print(f"  → Auto-selected: {pick}")
            return {"pretrained_model_name_or_path_or_dict": repo, "weight_name": pick}
    except Exception:
        pass

    return {"pretrained_model_name_or_path_or_dict": repo}


def _lora_load_thread(repo, scale):
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
        lora_kwargs = _resolve_lora_source(repo)
        print(f"  → load_lora_weights({lora_kwargs})")
        pipeline.load_lora_weights(**lora_kwargs)
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
    with lora_lock:
        lora_state["loading"] = True
        lora_state["error"] = None
    try:
        if not pipeline:
            raise RuntimeError("Model not ready")
        print(f"🔄 Unfusing and unloading LoRA: {lora_state['loaded']}")
        try: pipeline.unfuse_lora()
        except: pass
        try: pipeline.unload_lora_weights()
        except: pass
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
    return {**status, "queue_length": get_queue_length(), "lora": {k: v for k, v in lora_state.items() if not k.startswith("_")}}

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
                _gen_progress[jid] = {"type": "done"}; break
            _gen_progress[jid] = ev; lt = time.time()
            yield _SSE_PAD + f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "Content-Type": "text/event-stream"})

@app.get("/api/queue")
async def api_queue():
    ql = get_queue_length(); return {"queue_length": ql, "busy": ql > 0 or not gpu_sem._value}

@app.get("/api/gen_progress/{job_id}")
async def api_gen_progress(job_id: str):
    return _gen_progress.get(job_id) or {"type": "unknown"}

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
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if len(header) < 8: return False, "File too small"
        import struct
        header_size = struct.unpack("<Q", header[:8])[0]
        if header_size > 50_000_000: return False, "Header too large"
        if header_size == 0: return False, "Header size 0"
        try:
            text_start = header.decode("utf-8", errors="ignore").lower()
            if "<html" in text_start or "<!doc" in text_start:
                return False, "File is HTML"
        except: pass
        return True, "ok"
    except Exception as e:
        return False, str(e)

@app.post("/api/lora/download_civitai")
async def api_download_civitai(request: Request):
    body = await request.json()
    url = body.get("url", "")
    filename = body.get("filename", "")
    token = body.get("token", "")
    if not url: return {"error": "Missing download URL"}
    if not filename:
        filename = f"civitai_lora_{body.get('civitai_id', 'unknown')}.safetensors"
    filename = filename.replace("/", "_").replace("\\", "_")
    out_path = CIVITAI_LORA_DIR / filename

    if out_path.exists() and out_path.stat().st_size > 1024:
        valid, reason = _validate_safetensors(str(out_path))
        if valid:
            return {"ok": True, "path": str(out_path), "filename": filename, "cached": True}
        out_path.unlink()

    token = token.strip() if token else os.environ.get("CIVITAI_API_KEY", "")
    if token:
        parts = urlparse(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["token"] = token
        url = urlunparse((parts.scheme, parts.netloc, parts.path,
                          parts.params, urlencode(query), parts.fragment))
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://civitai.com/", "Accept": "*/*"}
        with _http_requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "text/html" in ct.lower():
                return {"error": "CivitAI returned HTML — need API token."}
            cd = r.headers.get("content-disposition", "")
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip().strip('"')
                if fname: filename = fname.replace("/","_").replace("\\","_"); out_path = CIVITAI_LORA_DIR / filename
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(str(out_path), "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk); downloaded += len(chunk)
        valid, reason = _validate_safetensors(str(out_path))
        if not valid:
            if out_path.exists(): out_path.unlink()
            return {"error": f"Not valid safetensors: {reason}"}
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"✅ Downloaded: {out_path} ({size_mb:.1f}MB)")
        return {"ok": True, "path": str(out_path), "filename": filename}
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
    with log_lock: log_entries.clear()
    return {"ok": True}

# ---- Launch ----
import socket as _socket

def _find_free_port(preferred=8000):
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try: s.bind(("0.0.0.0", preferred)); return preferred
        except OSError: s.bind(("0.0.0.0", 0)); return s.getsockname()[1]

PORT = int(os.environ.get("PORT", "8000"))

IN_COLAB = False
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    eval_js = None

def launch():
    import traceback as _tb
    import requests as _requests

    print("=" * 60)
    print("🎨 AI Image Edit Studio — diffusers + GGUF")
    print("=" * 60)

    # Kill any previous server on this port
    try:
        import signal
        _old_resp = _requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=1)
        if _old_resp.status_code == 200:
            print(f"⚠ Old server found on port {PORT}, it will be replaced.")
    except Exception:
        pass

    # Download GGUF model BLOCKING (visible in cell)
    try:
        download_models()
    except Exception as dl_err:
        print(f"\n❌ Download failed: {dl_err}")
        _tb.print_exc()
        raise

    # Load into GPU in background
    threading.Thread(target=init_model, daemon=True).start()

    # Keepalive
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
            uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", log_config=None)
        except Exception as exc:
            _server_error[0] = exc
            sys.__stderr__.write(f"\n❌ Server crashed: {exc}\n")
            _tb.print_exc(file=sys.__stderr__)

    _server_thread = threading.Thread(target=_serve, daemon=True)
    _server_thread.start()

    # Wait for server — with visible progress
    print(f"⏳ Waiting for server on port {PORT}...")
    _local_ready = False
    for _attempt in range(80):
        if _server_error[0]:
            print(f"❌ Server thread crashed: {_server_error[0]}")
            break
        try:
            if _requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=0.75).status_code == 200:
                _local_ready = True; break
        except: pass
        time.sleep(0.5)

    if not _local_ready:
        err_detail = f"Server error: {_server_error[0]}" if _server_error[0] else "No response after 40s"
        print(f"❌ Server failed to start: {err_detail}")
        raise RuntimeError(f"Server never became reachable on localhost:{PORT}. {err_detail}")

    sys.__stdout__.write(f"✅ Server healthy on localhost:{PORT}\n")
    print("📡 Model loading in background — UI shows progress")
    print("=" * 60)

    # Display UI
    if IN_COLAB:
        from IPython.display import display, HTML as _HTML
        _POPOUT_BTN_JS = """<script>(function(){var btn=document.getElementById('mle-popout-btn');if(!btn)return;btn.addEventListener('click',async function(){btn.textContent='Opening...';try{var url=await google.colab.kernel.proxyPort(%d,{cache:false});if(url&&!url.startsWith('http'))url='https://'+url;window.open(url,'_blank');btn.textContent='↗ Open in new tab';}catch(e){btn.textContent='⚠ Failed';}});})();</script>""" % PORT
        _POPOUT_BTN = '<button id="mle-popout-btn" style="margin-left:12px;padding:8px 18px;background:#d4a017;color:#000;border:none;border-radius:8px;font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;">↗ Open in new tab</button>'

        # Try proxy URL
        public_url = None
        for _ in range(20):
            try:
                c = eval_js(f"google.colab.kernel.proxyPort({PORT}, {{'cache': false}})")
                if c and not c.startswith("http"): c = "https://" + c
                if c and _requests.get(c.rstrip("/")+"/api/health", timeout=4).status_code == 200:
                    public_url = c; break
            except: pass
            time.sleep(0.5)

        if public_url:
            display(_HTML(f'<div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #d4a017;border-radius:12px;font-family:monospace;"><div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🎨 AI Image Edit Studio is live:</div><a href="{public_url}" target="_blank" style="color:#d4a017;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a></div>'))
        else:
            try:
                from google.colab import output as _co
                _co.serve_kernel_port_as_iframe(PORT, height='750')
                display(_HTML(f'<div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #d4a017;border-radius:12px;font-family:monospace;display:flex;align-items:center;gap:6px;"><span style="color:#d4a017;font-weight:bold;">🎨 Image Edit Studio</span><span style="color:#8A8A8A;font-size:13px;"> — embedded above ↑</span>{_POPOUT_BTN}</div>{_POPOUT_BTN_JS}'))
            except:
                try:
                    from google.colab import output as _co
                    _co.serve_kernel_port_as_window(PORT, anchor_text="🎨 Click to open Image Edit Studio")
                except: pass
    else:
        print(f"\n🎨 Image Edit Studio running at http://localhost:{PORT}\n")

    print("Server running in background. You can run other cells.\n")
    return


if __name__ == "__main__":
    launch()
    try:
        while True: time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
