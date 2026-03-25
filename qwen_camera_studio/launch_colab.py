# ============================================================
# Qwen Camera Studio — Colab Launch Script
#
# Usage (Colab cell):
#   %cd /content/qwen_camera_studio
#   from backend import *
#   exec(open("launch_colab.py").read())
# ============================================================

import time, threading, traceback, sys
import socket as _socket
import requests as _requests

# app, IN_COLAB, eval_js come from `from backend import *`


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
