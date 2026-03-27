# launch_colab.py — MissingLink Qwen Studio
import os,time,threading,traceback,sys
import socket as _socket
import requests as _requests

if 'app' not in dir():
    _pd=None
    for _c in ["/content/qwen_camera_studio",os.path.join(os.getcwd(),"qwen_camera_studio"),os.getcwd()]:
        if os.path.isfile(os.path.join(_c,"backend.py")): _pd=_c; break
    if not _pd: raise FileNotFoundError("Cannot find backend.py")
    if _pd not in sys.path: sys.path.insert(0,_pd)
    os.chdir(_pd); print(f"[launch] project: {_pd}")
    from backend import *

# ── Pre-download models ────────────────────────────────────────
VARIANT=os.environ.get("GGUF_VARIANT","Q4_K_M")
print("\n"+"="*60)
print("  📦 Pre-downloading model files")
print("="*60+"\n")

from huggingface_hub import hf_hub_download

print(f"[1/4] GGUF transformer (qwen-image-edit-2511-{VARIANT}.gguf)...")
hf_hub_download(repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",filename=f"qwen-image-edit-2511-{VARIANT}.gguf")
print("  ✓ cached")

print("[2/4] Base pipeline (text encoder + VAE)...")
# Download the full repo with allow_patterns to get only what we need
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Image-Edit-2511",
    allow_patterns=["model_index.json","scheduler/*","vae/*","tokenizer/*","processor/*",
                    "transformer/config.json","text_encoder/*"],
    ignore_patterns=["transformer/*.safetensors","transformer/**/*.safetensors","*.gguf"])
print("  ✓ cached")

print("[3/4] Lightning LoRA...")
hf_hub_download(repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
    filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
print("  ✓ cached")

print("[4/4] Multi-Angles LoRA...")
hf_hub_download(repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    filename="qwen-image-edit-2511-multiple-angles-lora.safetensors")
print("  ✓ cached")

print("\n  ✅ All model files ready!\n"+"="*60+"\n")

# ── Start Flask ────────────────────────────────────────────────
def _free_port(p=5000):
    with _socket.socket(_socket.AF_INET,_socket.SOCK_STREAM) as s:
        try: s.bind(("0.0.0.0",p)); return p
        except: s.bind(("0.0.0.0",0)); return s.getsockname()[1]

PORT=_free_port(5000)
_err=[None]
def _run():
    try: app.run(host="0.0.0.0",port=PORT,threaded=True,use_reloader=False)
    except Exception as e: _err[0]=e; print(f"❌ Flask crashed: {e}")

threading.Thread(target=_run,daemon=True).start()

ok=False
for _ in range(80):
    if _err[0]: break
    try:
        if _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive",timeout=.75).status_code==200: ok=True; break
    except: pass
    time.sleep(.5)
if not ok: raise RuntimeError(f"Flask unreachable on :{PORT}")
print(f"✅ Flask healthy on localhost:{PORT}")

# ── Display ────────────────────────────────────────────────────
if IN_COLAB:
    from IPython.display import display,HTML as _H
    url=None
    for _ in range(20):
        try:
            c=eval_js(f"google.colab.kernel.proxyPort({PORT},{{cache:false}})")
            if c and not c.startswith("http"): c="https://"+c
            if c and _requests.get(c.rstrip("/")+"/api/keepalive",timeout=4).status_code==200: url=c; break
        except: pass
        time.sleep(.5)
    if url:
        display(_H(f'''<div style="margin:12px 0;padding:14px 20px;background:#111113;border:1px solid #E8A917;border-radius:8px;font-family:monospace">
            <span style="color:#71717A;font-size:12px">🎬 Qwen Studio live:</span>
            <a href="{url}" target="_blank" style="color:#E8A917;font-size:16px;font-weight:bold;margin-left:8px">{url}</a></div>'''))
    else:
        try:
            from google.colab import output as _co
            _co.serve_kernel_port_as_iframe(PORT,height='850')
        except: pass
        display(_H(f'''<div style="margin:8px 0;padding:10px 16px;background:#111113;border:1px solid #E8A917;border-radius:8px;font-family:monospace;display:flex;align-items:center;gap:8px">
            <span style="color:#E8A917;font-weight:bold">🎬 Qwen Studio</span><span style="color:#71717A;font-size:12px">embedded above ↑</span>
            <button onclick="(async()=>{{const u=await google.colab.kernel.proxyPort({PORT},{{cache:false}});window.open(u.startsWith('http')?u:'https://'+u,'_blank')}})()"
                style="margin-left:auto;padding:5px 12px;background:#E8A917;color:#000;border:none;border-radius:5px;font-family:monospace;font-size:11px;font-weight:bold;cursor:pointer">↗ New Tab</button></div>'''))
else:
    print(f"\n🎬 Qwen Studio: http://localhost:{PORT}\n")

print("\n"+"="*60)
print("  ✅ Server running. Auto-loading model...")
print("="*60)

# Verify gguf is importable before trying to load
# Verify gguf is importable before trying to load
try:
    import gguf as _gguf_check
    print(f"  ✓ gguf package installed")
except ImportError:
    print("  ⚠️  WARNING: gguf package not found!")
    print("  Run this in a cell and restart runtime:")
    print('    !pip install "gguf>=0.10.0"')

# Auto-load the pipeline in background
import threading as _thr
_thr.Thread(target=load_pipeline, args=(VARIANT,), daemon=True).start()
print(f"  ⏳ Model loading in background (GGUF {VARIANT})...")
print("  📺 UI is usable now — model status shown in header.")
