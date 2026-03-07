# ============================================================
# 🔺 TRELLIS.2 — Colab Launch Script (launch_colab.py)
#
# NON-BLOCKING: Starts the Flask server in a daemon thread,
# verifies health, sets up the Colab proxy URL, then RETURNS.
# The cell finishes executing, freeing the kernel to run
# the keep_alive.py cell (which is what actually prevents
# Colab from disconnecting).
#
# Usage (Colab cell):
#   import os
#   os.chdir("/content/drive/MyDrive/MissingLink-Extras/model-generation-studio")
#   from backend import *
#   exec(open("launch_colab.py").read())
#
# Then in the NEXT cell, run:
#   exec(open("keep_alive.py").read())
# ============================================================

import time, threading, traceback, sys
import socket as _socket
import requests as _requests

# ── These come from backend.py (already imported via `from backend import *`) ──
# app, IN_COLAB, eval_js, jobs, console_lines


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

# ══════════════════════════════════════════════════════════════
# 1) Wait until Flask is responding on localhost
# ══════════════════════════════════════════════════════════════

_local_ready = False
_last_local_err = None
for _attempt in range(80):
    if _server_error[0] is not None:
        break
    try:
        _r = _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive", timeout=0.75)
        if _r.status_code == 200:
            _local_ready = True
            break
    except Exception as _e:
        _last_local_err = _e
    time.sleep(0.5)

if not _local_ready:
    raise RuntimeError(
        f"Flask server never became reachable on localhost:{PORT}.\n"
        f"  Server-thread error : {_server_error[0]}\n"
        f"  Last health-check error: {_last_local_err}\n"
    )

print(f"✅ Flask server healthy on localhost:{PORT}")

# ══════════════════════════════════════════════════════════════
# 2) Obtain & verify public access (Colab only)
# ══════════════════════════════════════════════════════════════

_launch_mode = None
public_url = None

if IN_COLAB:
    from IPython.display import display, HTML as _HTML

    # ── Tier 1: proxyPort URL ──
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
        <div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;">
            <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🔺 TRELLIS.2 Studio is live — click to open:</div>
            <a href="{public_url}" target="_blank" style="color:#E8A917;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
            <div style="color:#8A8A8A;font-size:12px;margin-top:10px;">
                Health check:
                <a href="{public_url.rstrip('/')}/api/keepalive" target="_blank"
                   style="color:#8A8A8A;text-decoration:underline;">/api/keepalive</a>
            </div>
        </div>
        """))
    else:
        sys.__stdout__.write(
            f"⚠ Proxy URL not reachable (last error: {_last_proxy_err}).\n"
            f"  Falling back to embedded iframe...\n"
        )

        _POPOUT_BTN_JS = """
        <script>
        (function() {
            var btn = document.getElementById('trellis-popout-btn');
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
                }
            });
        })();
        </script>
        """ % PORT

        _POPOUT_BTN_HTML = (
            '<button id="trellis-popout-btn" style="'
            'margin-left:12px;padding:6px 14px;'
            'background:#E8A917;color:#141414;border:none;border-radius:6px;'
            'font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;'
            '">↗ Open in new tab</button>'
        )

        # ── Tier 2: iframe ──
        _iframe_ok = False
        try:
            from google.colab import output as _colab_output
            _colab_output.serve_kernel_port_as_iframe(PORT, height='820')
            _launch_mode = "iframe"
            _iframe_ok = True
            display(_HTML(f"""
            <div style="margin:8px 0 4px;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                <span style="color:#E8A917;font-weight:bold;">🔺 TRELLIS.2 Studio</span>
                <span style="color:#8A8A8A;font-size:13px;"> — embedded above ↑</span>
                {_POPOUT_BTN_HTML}
            </div>
            {_POPOUT_BTN_JS}
            """))
        except Exception as _iframe_err:
            sys.__stdout__.write(f"  ⚠ iframe fallback failed: {_iframe_err}\n")

        # ── Tier 3: window ──
        if not _iframe_ok:
            try:
                from google.colab import output as _colab_output
                _colab_output.serve_kernel_port_as_window(PORT, anchor_text="🔺 Click to open TRELLIS.2 Studio")
                _launch_mode = "window"
                display(_HTML(f"""
                <div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
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
                        iframe.height = '820';
                        iframe.style.border = '2px solid #E8A917';
                        iframe.style.borderRadius = '12px';
                        document.querySelector('#output-area').appendChild(iframe);
                    })();
                    """ % PORT))
                    _launch_mode = "js_iframe"
                except Exception as _js_err:
                    raise RuntimeError(
                        f"All Colab display methods failed for port {PORT}.\n"
                        f"  The Flask server IS running on localhost:{PORT}.\n"
                    )

    print(f"🚀 Launch mode: {_launch_mode}")
else:
    _launch_mode = "local"
    print(f"\n🔺 TRELLIS.2 Studio running at http://localhost:{PORT}\n")

# ══════════════════════════════════════════════════════════════
# DONE — Cell finishes here. Server continues in background.
# Now run keep_alive.py in the NEXT cell.
# ══════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  ✅ Server is running in background thread.")
print("  👉 Run the NEXT cell (keep_alive.py) to prevent disconnect.")
print("=" * 60)