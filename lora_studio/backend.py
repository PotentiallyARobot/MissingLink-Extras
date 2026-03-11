# ============================================================
# Missing Link — AI Image Studio Backend (backend.py)
# Flask server for LoRA browsing + generation with
# Z-Image Turbo (GGUF) and Wan 2.2 I2V (GGUF)
# Google Drive history persistence
# https://www.missinglink.build/
# ============================================================

import os, sys, pathlib, subprocess, time, threading, traceback, json, uuid, gc, math, shutil, glob
from io import BytesIO
from datetime import datetime

# ── Ensure deps ──
def _ensure(pkg, pip_name=None):
    try: __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg],
                       capture_output=True, check=True)

_ensure("flask")
_ensure("PIL", "Pillow")
_ensure("huggingface_hub")
_ensure("numpy")
_ensure("tqdm")
_ensure("requests")

from flask import Flask, request, jsonify, send_file, Response
import torch, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import requests as http_requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from tqdm.auto import tqdm

# ── Colab detection ──
IN_COLAB = False
eval_js = None
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

LORA_BASE = pathlib.Path("/content/loras")
OUTPUTS   = pathlib.Path("/content/outputs")
UPLOADS   = pathlib.Path("/content/uploads")
for d in [LORA_BASE, OUTPUTS, UPLOADS]:
    d.mkdir(parents=True, exist_ok=True)

LORA_DIRS = {
    "z-image-turbo": LORA_BASE / "z_image_turbo",
    "wan-2.2":       LORA_BASE / "wan_2_2",
}
for d in LORA_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# GOOGLE DRIVE HISTORY
# ══════════════════════════════════════════════════════════════

GDRIVE_SAVE_PATH = os.environ.get("GDRIVE_SAVE_PATH", "/content/drive/MyDrive/MissingLink_Studio")
_gdrive_lock = threading.Lock()

def _ensure_gdrive_mounted():
    """Try to mount Google Drive if in Colab and not already mounted."""
    if not IN_COLAB:
        return False
    if os.path.exists("/content/drive/MyDrive"):
        return True
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return os.path.exists("/content/drive/MyDrive")
    except Exception as e:
        print(f"⚠ Google Drive mount failed: {e}")
        return False

def _get_save_dir():
    """Return the current save directory path, creating it if needed."""
    p = pathlib.Path(GDRIVE_SAVE_PATH)
    p.mkdir(parents=True, exist_ok=True)
    (p / "images").mkdir(exist_ok=True)
    return p

def _history_file():
    return _get_save_dir() / "history.json"

def load_history():
    with _gdrive_lock:
        hf = _history_file()
        if hf.exists():
            try:
                with open(hf, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Failed to load history: {e}")
        return []

def save_history(history):
    with _gdrive_lock:
        hf = _history_file()
        try:
            with open(hf, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠ Failed to save history: {e}")

def add_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    if len(history) > 500:
        history = history[:500]
    save_history(history)
    return history

def copy_image_to_gdrive(src_path, filename=None):
    try:
        save_dir = _get_save_dir() / "images"
        if not filename:
            filename = os.path.basename(src_path)
        dst = save_dir / filename
        shutil.copy2(str(src_path), str(dst))
        return str(dst)
    except Exception as e:
        print(f"⚠ Failed to copy image to Drive: {e}")
        return str(src_path)

def copy_input_image_to_gdrive(src_path):
    try:
        save_dir = _get_save_dir() / "images" / "inputs"
        save_dir.mkdir(exist_ok=True)
        filename = f"input_{uuid.uuid4().hex[:8]}_{os.path.basename(src_path)}"
        dst = save_dir / filename
        shutil.copy2(str(src_path), str(dst))
        return str(dst)
    except Exception as e:
        print(f"⚠ Failed to copy input image to Drive: {e}")
        return str(src_path)


# ══════════════════════════════════════════════════════════════
# CONSOLE CAPTURE
# ══════════════════════════════════════════════════════════════

import collections

class TeeWriter:
    def __init__(self, original, buf):
        self._original = original
        self._buf = buf
    def write(self, s):
        self._original.write(s)
        if s.strip():
            self._buf.append(s.rstrip('\n'))
        return len(s)
    def flush(self):
        self._original.flush()
    def fileno(self):
        return self._original.fileno()
    def __getattr__(self, name):
        return getattr(self._original, name)

console_lines = collections.deque(maxlen=500)
sys.stdout = TeeWriter(sys.__stdout__, console_lines)
sys.stderr = TeeWriter(sys.__stderr__, console_lines)


# ══════════════════════════════════════════════════════════════
# PRE-DOWNLOAD Z-IMAGE TURBO WEIGHTS
# ══════════════════════════════════════════════════════════════

print("⬇ Pre-downloading Z-Image Turbo model weights...")
try:
    _zit_diff = hf_hub_download(repo_id="unsloth/Z-Image-Turbo-GGUF", filename="z-image-turbo-Q4_K_M.gguf")
    print(f"  ✓ Diffusion: {_zit_diff}")
    _zit_llm = hf_hub_download(repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF", filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    print(f"  ✓ LLM: {_zit_llm}")
    _zit_vae = hf_hub_download(repo_id="black-forest-labs/FLUX.1-schnell", filename="ae.safetensors")
    print(f"  ✓ VAE: {_zit_vae}")
    print("✅ Z-Image Turbo weights ready")
except Exception as _e:
    print(f"⚠ Pre-download failed (will retry on first generate): {_e}")

if IN_COLAB:
    _ensure_gdrive_mounted()
    print(f"📂 History save path: {GDRIVE_SAVE_PATH}")

# ══════════════════════════════════════════════════════════════
# PIPELINE STATE
# ══════════════════════════════════════════════════════════════

pipeline_lock = threading.Lock()
current_pipeline = {
    "name": None,
    "sd": None,
    "lora_dir": None,
}

def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def unload_pipeline():
    if current_pipeline["sd"] is not None:
        print("📤 Unloading current pipeline...")
        del current_pipeline["sd"]
        current_pipeline["sd"] = None
        current_pipeline["name"] = None
        current_pipeline["lora_dir"] = None
        gpu_cleanup()
        if torch.cuda.is_available():
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            print(f"   VRAM free after unload: {vram_free:.1f} GB")

def load_z_image_turbo(lora_dir=None):
    from stable_diffusion_cpp import StableDiffusion
    DIFF_REPO = "unsloth/Z-Image-Turbo-GGUF"
    DIFF_FILE = "z-image-turbo-Q4_K_M.gguf"
    LLM_REPO  = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    LLM_FILE  = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    VAE_REPO  = "black-forest-labs/FLUX.1-schnell"
    VAE_FILE  = "ae.safetensors"
    print("⬇ Downloading Z-Image Turbo components...")
    diff_path = hf_hub_download(repo_id=DIFF_REPO, filename=DIFF_FILE)
    llm_path  = hf_hub_download(repo_id=LLM_REPO,  filename=LLM_FILE)
    vae_path  = hf_hub_download(repo_id=VAE_REPO,  filename=VAE_FILE)
    ldir = str(lora_dir) if lora_dir else None
    print(f"🔧 Loading Z-Image Turbo (lora_dir={ldir})...")
    sd = StableDiffusion(
        diffusion_model_path=diff_path, llm_path=llm_path, vae_path=vae_path,
        lora_model_dir=ldir, offload_params_to_cpu=True, diffusion_flash_attn=True,
    )
    print("✅ Z-Image Turbo loaded")
    return sd

def load_wan_22(lora_dir=None):
    from stable_diffusion_cpp import StableDiffusion
    I2V_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
    QUANT = "Q4_K_S"
    LOW_NOISE_FILE  = f"LowNoise/Wan2.2-I2V-A14B-LowNoise-{QUANT}.gguf"
    HIGH_NOISE_FILE = f"HighNoise/Wan2.2-I2V-A14B-HighNoise-{QUANT}.gguf"
    VAE_FILE        = "VAE/Wan2.1_VAE.safetensors"
    T5_REPO = "city96/umt5-xxl-encoder-gguf"
    T5_FILE = "umt5-xxl-encoder-Q8_0.gguf"
    print("⬇ Downloading Wan 2.2 I2V components...")
    low_path  = hf_hub_download(repo_id=I2V_REPO, filename=LOW_NOISE_FILE)
    high_path = hf_hub_download(repo_id=I2V_REPO, filename=HIGH_NOISE_FILE)
    vae_path  = hf_hub_download(repo_id=I2V_REPO, filename=VAE_FILE)
    t5_path   = hf_hub_download(repo_id=T5_REPO,  filename=T5_FILE)
    ldir = str(lora_dir) if lora_dir else None
    print(f"🔧 Loading Wan 2.2 I2V (lora_dir={ldir})...")
    sd = StableDiffusion(
        diffusion_model_path=low_path, high_noise_diffusion_model_path=high_path,
        t5xxl_path=t5_path, vae_path=vae_path, lora_model_dir=ldir,
        flow_shift=3.0, keep_clip_on_cpu=True, diffusion_flash_attn=True,
    )
    print("✅ Wan 2.2 I2V loaded")
    return sd

def ensure_pipeline(model_name, lora_dir=None):
    with pipeline_lock:
        need_reload = (
            current_pipeline["name"] != model_name or
            current_pipeline["sd"] is None or
            str(current_pipeline["lora_dir"]) != str(lora_dir)
        )
        if not need_reload:
            return current_pipeline["sd"]
        unload_pipeline()
        if model_name == "z-image-turbo":
            sd = load_z_image_turbo(lora_dir)
        elif model_name == "wan-2.2":
            sd = load_wan_22(lora_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        current_pipeline["name"] = model_name
        current_pipeline["sd"] = sd
        current_pipeline["lora_dir"] = str(lora_dir) if lora_dir else None
        return sd


# ══════════════════════════════════════════════════════════════
# CIVITAI DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════

def civitai_download(url, out_path, token=None, chunk_size=1024*1024):
    token = token or os.environ.get("CIVITAI_API_KEY", "")
    if token:
        parts = urlparse(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["token"] = token
        url = urlunparse((parts.scheme, parts.netloc, parts.path,
                          parts.params, urlencode(query), parts.fragment))
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://civitai.com/", "Accept": "*/*"}
    with http_requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        if "filename=" in cd and out_path.endswith("/"):
            fname = cd.split("filename=")[-1].strip().strip('"')
            out_path = out_path + fname
        total = int(r.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    return out_path


# ══════════════════════════════════════════════════════════════
# GENERATION JOBS
# ══════════════════════════════════════════════════════════════

jobs = {}

def run_z_image_job(job_id):
    job = jobs[job_id]
    try:
        s = job["settings"]
        lora_dir = LORA_DIRS["z-image-turbo"]
        job["progress"] = {"pct": 10, "phase": "Loading model..."}
        print(f"🎨 Starting Z-Image generation: {s.get('width',512)}x{s.get('height',512)}, {s.get('steps',25)} steps")
        sd = ensure_pipeline("z-image-turbo", lora_dir)
        input_path = s.get("input_image_path")
        has_input = input_path and os.path.exists(input_path)
        if has_input:
            job["progress"] = {"pct": 30, "phase": "Generating (img2img)..."}
            print(f"  Using input image: {input_path} (strength={s.get('strength', 0.75)})")
            images = sd.generate_image(
                init_image=input_path, prompt=s.get("prompt", ""),
                negative_prompt=s.get("negative_prompt", ""),
                width=s.get("width", 512), height=s.get("height", 512),
                cfg_scale=s.get("cfg_scale", 4.0), sample_steps=s.get("steps", 25),
                sample_method=s.get("sampler", "euler"), seed=s.get("seed", -1),
                strength=s.get("strength", 0.75),
            )
        else:
            job["progress"] = {"pct": 30, "phase": "Generating (txt2img)..."}
            print(f"  txt2img: prompt='{s.get('prompt','')[:60]}...'")
            images = sd.txt_to_img(
                prompt=s.get("prompt", ""), negative_prompt=s.get("negative_prompt", ""),
                width=s.get("width", 512), height=s.get("height", 512),
                cfg_scale=s.get("cfg_scale", 4.0), sample_steps=s.get("steps", 25),
                sample_method=s.get("sampler", "euler"), seed=s.get("seed", -1),
            )
        if not images:
            raise RuntimeError("No images returned")

        job["progress"] = {"pct": 90, "phase": "Saving..."}
        out_path = OUTPUTS / f"{job_id}.png"
        images[0].save(str(out_path))

        # Save to Google Drive + history
        gdrive_path = copy_image_to_gdrive(str(out_path), f"{job_id}.png")
        input_gdrive_path = None
        if has_input:
            input_gdrive_path = copy_input_image_to_gdrive(input_path)

        history_entry = {
            "id": job_id,
            "timestamp": datetime.now().isoformat(),
            "model": "z-image-turbo",
            "type": "image",
            "prompt": s.get("prompt", ""),
            "negative_prompt": s.get("negative_prompt", ""),
            "width": s.get("width", 512), "height": s.get("height", 512),
            "steps": s.get("steps", 25), "cfg_scale": s.get("cfg_scale", 4.0),
            "seed": s.get("seed", -1), "strength": s.get("strength", 0.75),
            "output_path": gdrive_path, "local_output_path": str(out_path),
            "input_image_path": input_gdrive_path,
            "local_input_path": input_path if has_input else None,
        }
        add_history_entry(history_entry)

        job["result"] = {"path": str(out_path), "type": "image", "history_entry": history_entry}
        job["status"] = "done"
        job["progress"] = {"pct": 100, "phase": "Complete!"}
        print(f"✅ Z-Image generation done: {out_path}")
    except Exception as e:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)
        job["progress"] = {"pct": 100, "phase": f"Error: {e}"}

def run_wan_job(job_id):
    job = jobs[job_id]
    try:
        s = job["settings"]
        lora_dir = LORA_DIRS["wan-2.2"]
        input_path = s.get("input_image_path")
        if not input_path or not os.path.exists(input_path):
            raise ValueError("Input image required for Wan 2.2 I2V")
        job["progress"] = {"pct": 5, "phase": "Preparing image..."}
        img = Image.open(input_path).convert("RGB")
        target_area = s.get("width", 832) * s.get("height", 480)
        aspect = img.width / img.height
        w = math.sqrt(target_area * aspect)
        h = target_area / w
        W = max(32, int(round(w / 32.0) * 32))
        H = max(32, int(round(h / 32.0) * 32))
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        resized_path = str(UPLOADS / f"{job_id}_resized.png")
        img.save(resized_path)
        job["progress"] = {"pct": 10, "phase": "Loading model..."}
        sd = ensure_pipeline("wan-2.2", lora_dir)
        job["progress"] = {"pct": 30, "phase": "Generating video..."}
        common = dict(
            prompt=s.get("prompt", ""), negative_prompt=s.get("negative_prompt", ""),
            width=W, height=H, cfg_scale=s.get("cfg_scale", 6.0),
            sample_method=s.get("sampler", "euler"), video_frames=s.get("video_frames", 49),
        )
        frames = None
        for extra in [
            {"init_image": resized_path, "strength": s.get("strength", 0.75)},
            {"image": resized_path, "strength": s.get("strength", 0.75)},
            {"ref_images": [resized_path], "strength": s.get("strength", 0.75)},
            {"init_image": resized_path},
            {"image": resized_path},
        ]:
            try:
                frames = sd.generate_video(**common, **extra)
                break
            except TypeError:
                continue
        if not frames:
            raise RuntimeError("No frames returned from Wan 2.2 I2V")
        job["progress"] = {"pct": 85, "phase": "Encoding video..."}
        out_path = OUTPUTS / f"{job_id}.mp4"
        _save_video_ffmpeg(frames, s.get("fps", 16), str(out_path))

        gdrive_path = copy_image_to_gdrive(str(out_path), f"{job_id}.mp4")
        input_gdrive_path = copy_input_image_to_gdrive(input_path)
        history_entry = {
            "id": job_id, "timestamp": datetime.now().isoformat(),
            "model": "wan-2.2", "type": "video",
            "prompt": s.get("prompt", ""), "negative_prompt": s.get("negative_prompt", ""),
            "width": W, "height": H, "steps": s.get("steps", 20),
            "cfg_scale": s.get("cfg_scale", 6.0), "seed": s.get("seed", -1),
            "strength": s.get("strength", 0.75), "video_frames": s.get("video_frames", 49),
            "output_path": gdrive_path, "local_output_path": str(out_path),
            "input_image_path": input_gdrive_path, "local_input_path": input_path,
        }
        add_history_entry(history_entry)

        job["result"] = {"path": str(out_path), "type": "video", "history_entry": history_entry}
        job["status"] = "done"
        job["progress"] = {"pct": 100, "phase": "Complete!"}
        print(f"✅ Wan 2.2 I2V generation done: {out_path}")
    except Exception as e:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)
        job["progress"] = {"pct": 100, "phase": f"Error: {e}"}

def _save_video_ffmpeg(frames, fps, out_path):
    import ffmpeg as ffmpeg_lib
    if not frames:
        raise ValueError("No frames")
    w, h = frames[0].size
    raw = b"".join(np.array(f.convert("RGB"), dtype=np.uint8).tobytes() for f in frames)
    (
        ffmpeg_lib
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}", r=fps)
        .output(out_path, vcodec="libx264", pix_fmt="yuv420p", r=fps, movflags="+faststart")
        .overwrite_output()
        .run(input=raw, quiet=True)
    )


# ══════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.route("/api/keepalive")
def api_keepalive():
    return jsonify({"ok": True})

@app.route("/")
def index():
    html_path = pathlib.Path(__file__).parent / "index.html"
    if html_path.exists():
        return send_file(str(html_path), mimetype="text/html")
    return Response("<h1>index.html not found</h1>", mimetype="text/html", status=404)

# ── LoRA Management ──
@app.route("/api/loras")
def api_list_loras():
    result = {}
    for model_key, ldir in LORA_DIRS.items():
        files = []
        for f in sorted(ldir.glob("*.safetensors")):
            files.append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
        for f in sorted(ldir.glob("*.gguf")):
            files.append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
        result[model_key] = files
    return jsonify(result)

@app.route("/api/lora/download", methods=["POST"])
def api_download_lora():
    data = request.json
    url = data.get("url")
    model_key = data.get("model")
    filename = data.get("filename", "")
    token = data.get("token", os.environ.get("CIVITAI_API_KEY", ""))
    if not url or not model_key:
        return jsonify({"error": "url and model required"}), 400
    ldir = LORA_DIRS.get(model_key)
    if not ldir:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400
    out_path = str(ldir / filename) if filename else str(ldir) + "/"
    try:
        saved = civitai_download(url, out_path, token=token)
        return jsonify({"ok": True, "path": saved, "filename": os.path.basename(saved)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/lora/delete", methods=["POST"])
def api_delete_lora():
    data = request.json
    path = data.get("path", "")
    if not path:
        return jsonify({"error": "path required"}), 400
    real = os.path.realpath(path)
    allowed = any(real.startswith(os.path.realpath(str(d))) for d in LORA_DIRS.values())
    if not allowed or not os.path.isfile(real):
        return jsonify({"error": "Not allowed"}), 403
    os.remove(real)
    return jsonify({"ok": True})

@app.route("/api/lora/clear", methods=["POST"])
def api_clear_loras():
    model_key = request.json.get("model")
    ldir = LORA_DIRS.get(model_key)
    if not ldir:
        return jsonify({"error": "Unknown model"}), 400
    for f in ldir.glob("*"):
        if f.is_file():
            f.unlink()
    with pipeline_lock:
        if current_pipeline["name"] == model_key:
            unload_pipeline()
    return jsonify({"ok": True})

# ── Upload ──
@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    fname = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    path = UPLOADS / fname
    f.save(str(path))
    return jsonify({"path": str(path), "filename": fname})

# ── Generation ──
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    model_key = data.get("model")
    settings = data.get("settings", {})
    if model_key not in LORA_DIRS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id, "model": model_key, "settings": settings,
        "status": "running", "progress": {"pct": 0, "phase": "Starting..."},
        "result": None, "error": None,
    }
    jobs[job_id] = job
    if model_key == "z-image-turbo":
        threading.Thread(target=run_z_image_job, args=(job_id,), daemon=True).start()
    elif model_key == "wan-2.2":
        threading.Thread(target=run_wan_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status": job["status"], "progress": job["progress"],
        "result": job["result"], "error": job["error"],
    })

@app.route("/api/file")
def api_file():
    p = request.args.get("p", "")
    if not p or not os.path.isfile(p):
        return "Not found", 404
    real = os.path.realpath(p)
    allowed_roots = [str(OUTPUTS), str(UPLOADS), os.path.realpath(GDRIVE_SAVE_PATH)]
    if not any(real.startswith(r) for r in allowed_roots):
        return "Forbidden", 403
    return send_file(real)

@app.route("/api/hw")
def api_hw():
    hw = {}
    try:
        hw["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        hw["gpu_total_mb"] = round(props.total_memory / 1e6)
        hw["gpu_alloc_mb"] = round(torch.cuda.memory_allocated(0) / 1e6)
        hw["gpu_free_mb"] = hw["gpu_total_mb"] - round(torch.cuda.memory_reserved(0) / 1e6)
    except Exception:
        pass
    hw["pipeline"] = current_pipeline["name"]
    hw["lora_dir"] = current_pipeline["lora_dir"]
    return jsonify(hw)

@app.route("/api/console")
def api_console():
    n = int(request.args.get("n", 80))
    return jsonify({"lines": list(console_lines)[-n:]})

# ── History API ──
@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 100))
    history = load_history()
    return jsonify({"history": history[:limit], "total": len(history)})

@app.route("/api/history/delete", methods=["POST"])
def api_history_delete():
    entry_id = request.json.get("id")
    if not entry_id:
        return jsonify({"error": "id required"}), 400
    history = load_history()
    history = [h for h in history if h.get("id") != entry_id]
    save_history(history)
    return jsonify({"ok": True})

@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    save_history([])
    return jsonify({"ok": True})

# ── Save Path API ──
@app.route("/api/save-path")
def api_get_save_path():
    return jsonify({"path": GDRIVE_SAVE_PATH, "gdrive_mounted": os.path.exists("/content/drive/MyDrive")})

@app.route("/api/save-path", methods=["POST"])
def api_set_save_path():
    global GDRIVE_SAVE_PATH
    new_path = request.json.get("path", "").strip()
    if not new_path:
        return jsonify({"error": "path required"}), 400
    GDRIVE_SAVE_PATH = new_path
    try:
        _get_save_dir()
        return jsonify({"ok": True, "path": GDRIVE_SAVE_PATH})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
