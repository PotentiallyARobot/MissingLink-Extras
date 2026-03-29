# launch_colab.py — MissingLink Qwen Studio (hardened)
import os,time,threading,traceback,sys
import socket as _socket
import requests as _requests

# ── MissingLink DRM ────────────────────────────────────────────
try:
    import stable_diffusion_cpp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable,"-m","pip","install",
        "https://missinglink.build/wheel/stable_diffusion_cpp_python-0.4.5-cp312-cp312-linux_x86_64.whl",
        "-q","--break-system-packages"])
    import stable_diffusion_cpp

if 'app' not in dir():
    _pd=None
    for _c in ["/content/qwen_camera_studio",os.path.join(os.getcwd(),"qwen_camera_studio"),os.getcwd()]:
        if os.path.isfile(os.path.join(_c,"backend.py")): _pd=_c; break
    if not _pd: raise FileNotFoundError("Cannot find backend.py")
    if _pd not in sys.path: sys.path.insert(0,_pd)
    os.chdir(_pd); print(f"[launch] project: {_pd}")
    from backend import *

# ── Detect restart ─────────────────────────────────────────────
_state_dir=os.path.join(PROJECT_DIR,".state")
_restart_marker=os.path.join(_state_dir,"last_launch")
_is_restart=os.path.isfile(_restart_marker)
if _is_restart:
    try:
        with open(_restart_marker) as f: _last=float(f.read().strip())
        elapsed=time.time()-_last
        print(f"\n  ♻️  Restart detected (last launch was {elapsed:.0f}s ago)")
        print(f"  📂 Preserved state in .state/, uploads/, outputs/")
    except: pass
os.makedirs(_state_dir,exist_ok=True)
with open(_restart_marker,"w") as f: f.write(str(time.time()))

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

# ── Install waitress (production WSGI server) ──────────────────
try:
    import waitress
    print("  ✓ waitress installed")
except ImportError:
    import subprocess
    print("  📦 Installing waitress (production server)...")
    subprocess.check_call([sys.executable,"-m","pip","install","waitress","-q","--break-system-packages"])
    import waitress
    print("  ✓ waitress installed")

# ── Start server ───────────────────────────────────────────────
def _free_port(p=5000):
    with _socket.socket(_socket.AF_INET,_socket.SOCK_STREAM) as s:
        try: s.bind(("0.0.0.0",p)); return p
        except: s.bind(("0.0.0.0",0)); return s.getsockname()[1]

PORT=_free_port(5000)
_err=[None]

def _run():
    try:
        # waitress: true multi-threaded, no GIL-starved request blocking,
        # proper connection handling, no "development server" warnings
        waitress.serve(app,host="0.0.0.0",port=PORT,
                       threads=8,              # 8 worker threads
                       channel_timeout=120,    # keep connections alive
                       recv_bytes=65536,
                       send_bytes=65536,
                       connection_limit=200,
                       cleanup_interval=30,
                       _quiet=False)
    except Exception as e:
        _err[0]=e; print(f"❌ Server crashed: {e}")

threading.Thread(target=_run,daemon=True).start()

ok=False
for _ in range(80):
    if _err[0]: break
    try:
        if _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive",timeout=.75).status_code==200: ok=True; break
    except: pass
    time.sleep(.5)
if not ok: raise RuntimeError(f"Server unreachable on :{PORT}")
print(f"✅ Server healthy on localhost:{PORT} (waitress, 8 threads)")

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

    _restart_msg=' <span style="color:#22C55E;font-size:11px">♻️ Session restored</span>' if _is_restart else ''

    if url:
        # ── Self-healing iframe ──────────────────────────
        # Instead of Colab's broken serve_kernel_port_as_iframe,
        # we inject our own iframe with a health-check loop that
        # automatically reloads the iframe if the proxy 500s.
        display(_H(f'''
        <div id="qs-container" style="position:relative;width:100%;height:850px;border:1px solid #E8A917;border-radius:8px;overflow:hidden;background:#09090B">
            <iframe id="qs-frame" src="{url}" style="width:100%;height:100%;border:none" allow="clipboard-write"></iframe>
            <div id="qs-overlay" style="display:none;position:absolute;inset:0;background:#09090B;z-index:10;
                 display:none;flex-direction:column;align-items:center;justify-content:center;gap:12px;font-family:monospace">
                <div style="color:#E8A917;font-size:14px;font-weight:bold">⏳ Reconnecting to Qwen Studio...</div>
                <div id="qs-overlay-sub" style="color:#71717A;font-size:11px">Waiting for server</div>
            </div>
        </div>
        <div style="margin:8px 0;padding:10px 16px;background:#111113;border:1px solid #E8A917;border-radius:8px;font-family:monospace;display:flex;align-items:center;gap:8px">
            <span style="color:#E8A917;font-weight:bold">🎬 Qwen Studio</span>
            <span style="color:#71717A;font-size:12px">embedded above ↑</span>{_restart_msg}
            <span id="qs-conn-status" style="margin-left:8px;color:#22C55E;font-size:10px">●</span>
            <button onclick="document.getElementById('qs-frame').src='{url}'"
                style="margin-left:8px;padding:4px 10px;background:none;color:#71717A;border:1px solid #27272A;border-radius:4px;font-family:monospace;font-size:10px;cursor:pointer"
                title="Force reload the UI">↻ Reload</button>
            <button onclick="(async()=>{{const u=await google.colab.kernel.proxyPort({PORT},{{cache:false}});window.open(u.startsWith('http')?u:'https://'+u,'_blank')}})()"
                style="margin-left:auto;padding:5px 12px;background:#E8A917;color:#000;border:none;border-radius:5px;font-family:monospace;font-size:11px;font-weight:bold;cursor:pointer">↗ New Tab</button>
        </div>
        <script>
        (function(){{
            const frame=document.getElementById('qs-frame');
            const overlay=document.getElementById('qs-overlay');
            const dot=document.getElementById('qs-conn-status');
            const sub=document.getElementById('qs-overlay-sub');
            const url='{url}'.replace(/\\/$/,'')+'/api/keepalive';
            let fails=0, checking=false;

            async function check(){{
                if(checking) return;
                checking=true;
                try{{
                    const c=new AbortController();
                    const t=setTimeout(()=>c.abort(),5000);
                    const r=await fetch(url,{{signal:c.signal,cache:'no-store'}});
                    clearTimeout(t);
                    if(r.ok){{
                        if(fails>2){{
                            // Was disconnected, reload iframe
                            frame.src='{url}';
                            overlay.style.display='none';
                        }}
                        fails=0;
                        dot.style.color='#22C55E';
                        dot.textContent='●';
                    }} else {{
                        fails++;
                    }}
                }}catch(e){{
                    fails++;
                }}
                if(fails>=3){{
                    dot.style.color='#EF4444';
                    dot.textContent='○';
                    overlay.style.display='flex';
                    sub.textContent='Attempt '+fails+' — checking every 3s...';
                }}
                checking=false;
            }}
            setInterval(check,3000);
            check();
        }})();
        </script>
        '''))
    else:
        # Fallback: try Colab's iframe + manual link
        try:
            from google.colab import output as _co
            _co.serve_kernel_port_as_iframe(PORT,height='850')
        except: pass
        display(_H(f'''<div style="margin:8px 0;padding:10px 16px;background:#111113;border:1px solid #E8A917;border-radius:8px;font-family:monospace;display:flex;align-items:center;gap:8px">
            <span style="color:#E8A917;font-weight:bold">🎬 Qwen Studio</span><span style="color:#71717A;font-size:12px">embedded above ↑</span>{_restart_msg}
            <button onclick="(async()=>{{const u=await google.colab.kernel.proxyPort({PORT},{{cache:false}});window.open(u.startsWith('http')?u:'https://'+u,'_blank')}})()"
                style="margin-left:auto;padding:5px 12px;background:#E8A917;color:#000;border:none;border-radius:5px;font-family:monospace;font-size:11px;font-weight:bold;cursor:pointer">↗ New Tab</button></div>'''))
else:
    print(f"\n🎬 Qwen Studio: http://localhost:{PORT}\n")

print("\n"+"="*60)
if _is_restart:
    print("  ♻️  RESTARTED — previous session state preserved")
    print("  📂 Uploads, outputs, and UI state restored from disk")
else:
    print("  ✅ Fresh start")
print("  ⏳ Auto-loading model...")
print("="*60)

try:
    import gguf as _gguf_check
    print(f"  ✓ gguf package installed")
except ImportError:
    print("  ⚠️  WARNING: gguf package not found!")
    print("  Run this in a cell and restart runtime:")
    print('    !pip install "gguf>=0.10.0"')

import threading as _thr
_thr.Thread(target=load_pipeline, args=(VARIANT,), daemon=True).start()
print(f"  ⏳ Model loading in background (GGUF {VARIANT})...")
print("  📺 UI is usable now — model status shown in header.")
