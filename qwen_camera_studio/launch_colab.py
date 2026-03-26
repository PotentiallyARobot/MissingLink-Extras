# ============================================================
# Qwen Camera Studio — Colab Launch Script
#
# Usage (Colab cell — just ONE line needed):
#   exec(open("/content/qwen_camera_studio/launch_colab.py").read())
#
# Or:
#   %cd /content/qwen_camera_studio
#   exec(open("launch_colab.py").read())
# ============================================================

import os, time, threading, traceback, sys
import socket as _socket
import requests as _requests

# ── Auto-import backend if not already imported ────────────────
if 'app' not in dir():
    # Find the project directory
    _project_dir = None
    for _p in [
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else None,
        "/content/qwen_camera_studio",
        os.path.join(os.getcwd(), "qwen_camera_studio"),
        os.getcwd(),
    ]:
        if _p and os.path.isfile(os.path.join(_p, "backend.py")):
            _project_dir = _p
            break

    if _project_dir is None:
        raise FileNotFoundError(
            "Cannot find backend.py. Make sure qwen_camera_studio/ is in /content/ "
            "or the current directory."
        )

    print(f"[Launch] Found project at: {_project_dir}")
    if _project_dir not in sys.path:
        sys.path.insert(0, _project_dir)
    os.chdir(_project_dir)

    from backend import *  # noqa: imports app, IN_COLAB, etc.

# ══════════════════════════════════════════════════════════════
# 0) Pre-download all model files (with progress visible in notebook)
# ══════════════════════════════════════════════════════════════

GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "Q4_K_M")

print()
print("=" * 60)
print("  📦 Pre-downloading model files (this only happens once)")
print("=" * 60)
print()

from huggingface_hub import hf_hub_download, snapshot_download

# 1) GGUF transformer
_gguf_file = f"qwen-image-edit-2511-{GGUF_VARIANT}.gguf"
print(f"[1/4] GGUF transformer: {_gguf_file}")
hf_hub_download(
    repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
    filename=_gguf_file,
)
print(f"  ✓ GGUF cached")

# 2) Base model (text encoder, VAE, config — NOT the full transformer)
print(f"[2/4] Base pipeline config + text encoder + VAE...")
# Download specific files, not the full 57GB repo
for _sub in [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
    "tokenizer/special_tokens_map.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
    "processor/preprocessor_config.json",
    "transformer/config.json",
]:
    try:
        hf_hub_download(repo_id="Qwen/Qwen-Image-Edit-2511", filename=_sub)
    except Exception:
        pass  # some files may not exist with these exact names
# Text encoder is large — download its shards
print("  Downloading text encoder (Qwen2.5-VL-7B)...")
for _shard in [
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
]:
    try:
        hf_hub_download(repo_id="Qwen/Qwen-Image-Edit-2511", filename=_shard)
    except Exception:
        pass
print(f"  ✓ Base pipeline cached")

# 3) Lightning LoRA
print(f"[3/4] Lightning 4-step LoRA...")
hf_hub_download(
    repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
    filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
)
print(f"  ✓ Lightning LoRA cached")

# 4) Multi-Angles LoRA
print(f"[4/4] Multi-Angles LoRA...")
hf_hub_download(
    repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    filename="qwen-image-edit-2511-multiple-angles-lora.safetensors",
)
print(f"  ✓ Multi-Angles LoRA cached")

print()
print("  ✅ All model files downloaded!")
print("=" * 60)
print()


def _find_free_port(preferred=5000):
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]


PORT = _find_free_port(5000)
_server_error = [None]


def _run_server():
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
    except Exception as exc:
        _server_error[0] = exc
        sys.__stderr__.write(f"\n❌ Flask server crashed: {exc}\n")
        traceback.print_exc(file=sys.__stderr__)


server_thread = threading.Thread(target=_run_server, daemon=True, name='flask-server')
server_thread.start()

# Wait for Flask
_local_ready = False
for _attempt in range(80):
    if _server_error[0] is not None:
        break
    try:
        _r = _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive", timeout=0.75)
        if _r.status_code == 200:
            _local_ready = True
            break
    except Exception:
        pass
    time.sleep(0.5)

if not _local_ready:
    raise RuntimeError(f"Flask server never became reachable on localhost:{PORT}.")

print(f"✅ Flask server healthy on localhost:{PORT}")

# ── Colab display ──────────────────────────────────────────────

_launch_mode = None
public_url = None

if IN_COLAB:
    from IPython.display import display, HTML as _HTML

    # Try proxyPort URL
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
                _rr = _requests.get(_candidate.rstrip("/") + "/api/keepalive", timeout=4)
                if _rr.status_code == 200:
                    public_url = _candidate
                    _launch_mode = "proxy_url"
                    break
            except Exception:
                pass
        time.sleep(0.5)

    if _launch_mode == "proxy_url":
        display(_HTML(f"""
        <div style="margin:16px 0;padding:16px 24px;background:#0c0c0f;border:2px solid #00e5a0;border-radius:12px;font-family:monospace;">
            <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🎬 Qwen Camera Studio is live:</div>
            <a href="{public_url}" target="_blank"
               style="color:#00e5a0;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
            <div style="color:#8A8A8A;font-size:12px;margin-top:10px;">
                Open in a new tab for the best experience.
            </div>
        </div>
        """))
    else:
        # Fallback: iframe
        try:
            from google.colab import output as _colab_output
            _colab_output.serve_kernel_port_as_iframe(PORT, height='850')
            _launch_mode = "iframe"

            display(_HTML(f"""
            <div style="margin:8px 0;padding:10px 16px;background:#0c0c0f;border:2px solid #00e5a0;border-radius:12px;font-family:monospace;display:flex;align-items:center;gap:8px;">
                <span style="color:#00e5a0;font-weight:bold;">🎬 Qwen Camera Studio</span>
                <span style="color:#8A8A8A;font-size:13px;">— embedded above ↑</span>
                <button onclick="(async()=>{{const u=await google.colab.kernel.proxyPort({PORT},{{cache:false}});window.open(u.startsWith('http')?u:'https://'+u,'_blank')}})()"
                        style="margin-left:auto;padding:6px 14px;background:#00e5a0;color:#0c0c0f;border:none;border-radius:6px;font-family:monospace;font-size:12px;font-weight:bold;cursor:pointer;">
                    ↗ Open in New Tab
                </button>
            </div>
            """))
        except Exception:
            # Last resort: JS iframe
            from IPython.display import Javascript as _JS
            display(_JS("""
            (async () => {
                const url = await google.colab.kernel.proxyPort(%d, {cache: false});
                const iframe = document.createElement('iframe');
                iframe.src = url;
                iframe.width = '100%%';
                iframe.height = '850';
                iframe.style.border = '2px solid #00e5a0';
                iframe.style.borderRadius = '12px';
                document.querySelector('#output-area').appendChild(iframe);
            })();
            """ % PORT))
            _launch_mode = "js_iframe"

    print(f"🚀 Launch mode: {_launch_mode}")
else:
    print(f"\n🎬 Qwen Camera Studio running at http://localhost:{PORT}\n")

print()
print("=" * 60)
print("  ✅ Server running. Load the model from the UI.")
print("=" * 60)
