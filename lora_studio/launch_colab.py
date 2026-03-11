# ============================================================
# LoRA Studio — Colab Launch Script (launch_colab.py)
#
# Usage (single Colab cell):
#   %cd /content/lora_studio
#   exec(open("launch_colab.py").read())
# ============================================================

import time, threading, traceback, sys, subprocess
import socket as _socket
import requests as _requests

# ── Install deps ──
print("📦 Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "flask", "Pillow", "huggingface_hub", "numpy", "tqdm",
    "requests", "ffmpeg-python",
    "stable-diffusion-cpp-python",
], stdout=sys.stdout, stderr=sys.stderr)

# ── Import backend ──
from backend import app, IN_COLAB, eval_js

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

# ── Wait for health ──
_local_ready = False
for _attempt in range(60):
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
    raise RuntimeError(f"Flask server never became reachable on localhost:{PORT}")

print(f"✅ Flask server healthy on localhost:{PORT}")

# ── Colab proxy URL ──
if IN_COLAB:
    from IPython.display import display, HTML

    public_url = None
    for _proxy_attempt in range(20):
        try:
            _candidate = eval_js(f"google.colab.kernel.proxyPort({PORT}, {{'cache': false}})")
            if _candidate and not _candidate.startswith("http"):
                _candidate = "https://" + _candidate
        except Exception:
            _candidate = None

        if _candidate:
            try:
                _rr = _requests.get(_candidate.rstrip("/") + "/api/keepalive", timeout=4)
                if _rr.status_code == 200:
                    public_url = _candidate
                    break
            except Exception:
                pass
        time.sleep(0.5)

    if public_url:
        display(HTML(f"""
        <div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #7b68ee;border-radius:12px;font-family:monospace;">
            <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">⚡ LoRA Studio is live — click to open:</div>
            <a href="{public_url}" target="_blank" style="color:#7b68ee;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
        </div>
        """))
    else:
        # Fallback: iframe
        try:
            from google.colab import output as _colab_output
            _colab_output.serve_kernel_port_as_iframe(PORT, height='850')
            display(HTML(f"""
            <div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #7b68ee;border-radius:12px;font-family:monospace;">
                <span style="color:#7b68ee;font-weight:bold;">⚡ LoRA Studio</span>
                <span style="color:#8A8A8A;font-size:13px;"> — embedded above ↑</span>
            </div>
            """))
        except Exception:
            print(f"⚠ Could not open proxy. Server running at http://localhost:{PORT}")
else:
    print(f"\n⚡ LoRA Studio running at http://localhost:{PORT}\n")

print()
print("=" * 60)
print("  ✅ Server running in background thread.")
print("=" * 60)
