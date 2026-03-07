# ============================================================
# 🔄 Colab Keep-Alive — Paste this in a cell and run it
# ============================================================
# Simulates organic user activity via JavaScript injection
# so Colab doesn't disconnect when you tab away.
# ============================================================
def run_keep_alive():
    # ============================================================
    # 🔄 Colab Keep-Alive v7 — Kernel Exec + Extension-Style Click
    # ============================================================
    #
    # Three independent mechanisms, any one of which should suffice:
    #
    #   1. REAL KERNEL EXECUTION — get_ipython().run_cell() every 20s
    #      This is actual cell execution that Colab cannot dismiss.
    #
    #   2. EXTENSION-STYLE BODY CLICK — google.colab.output.eval_js()
    #      runs document.body.click() in Colab's privileged JS context,
    #      equivalent to what a Chrome extension content script does.
    #      This is NOT subject to background-tab throttling because
    #      eval_js() is a kernel→frontend RPC call, not a page timer.
    #
    #   3. JS MutationObserver + Web Worker for dialog dismissal
    #      Catches and auto-clicks disconnect dialogs instantly.
    #
    # PREREQUISITE: launch_colab.py must have run first (non-blocking).
    # TO STOP: Interrupt this cell (⬜ stop button).
    # ============================================================

    import time, sys, os, gc, threading
    import IPython
    from IPython.display import display, HTML

    # ── Check server ──
    try:
        import requests as _req
        _r = _req.get("http://127.0.0.1:5000/api/keepalive", timeout=2)
        assert _r.status_code == 200
        print("✅ Flask server alive")
    except Exception:
        print("⚠ Flask server not detected — did you run launch_colab.py?")

    # ── Mechanism 2: eval_js body click function ──
    # This uses Colab's kernel→frontend bridge. Unlike injected JS
    # that gets throttled in background tabs, eval_js() is triggered
    # by a kernel message, so it fires reliably regardless of tab state.
    _EVAL_JS_AVAILABLE = False
    try:
        from google.colab.output import eval_js as _eval_js
        _EVAL_JS_AVAILABLE = True
        print("✅ eval_js available — extension-style clicks enabled")
    except ImportError:
        print("⚠ Not in Colab — eval_js clicks disabled")

    def _extension_click():
        """Simulate what the Chrome extension does: document.body.click()
        plus connect button click, via Colab's privileged eval_js bridge."""
        if not _EVAL_JS_AVAILABLE:
            return
        try:
            _eval_js("""
            (function() {
                // Click body (same as extension)
                document.body.click();
                // Click connect button if present
                var cb = document.querySelector('colab-connect-button')
                      || document.querySelector('#connect');
                if (cb) cb.click();
                // Dismiss any dialogs
                var sels = ['paper-button#ok','mwc-button[slot="primaryAction"]',
                            'md-text-button[slot="primaryAction"]','md-filled-button[slot="primaryAction"]'];
                for (var i=0; i<sels.length; i++) {
                    var el = document.querySelector(sels[i]);
                    if (el && el.offsetParent !== null) { el.click(); break; }
                }
                // Mouse event for good measure
                document.dispatchEvent(new MouseEvent('mousemove', {
                    clientX: 100 + Math.floor(Math.random()*600),
                    clientY: 100 + Math.floor(Math.random()*400),
                    bubbles: true
                }));
            })();
            """, ignore_result=True)
        except Exception:
            pass

    # ── Mechanism 3: Persistent JS injection (runs once) ──
    display(HTML("""
    <script>
    (function(){
        function tryDismiss() {
            var sels = ['paper-button#ok','mwc-button[slot="primaryAction"]',
                'md-text-button[slot="primaryAction"]','md-filled-button[slot="primaryAction"]',
                'colab-dialog paper-button','colab-dialog mwc-button'];
            for (var i=0; i<sels.length; i++) {
                var el = document.querySelector(sels[i]);
                if (el && el.offsetParent !== null) { el.click(); return true; }
            }
            var btns = document.querySelectorAll('paper-button,mwc-button,md-text-button,md-filled-button,button');
            for (var j=0; j<btns.length; j++) {
                var t = (btns[j].textContent||'').trim().toLowerCase();
                if (['ok','yes','connect','reconnect','dismiss'].indexOf(t) >= 0 && btns[j].offsetParent !== null) {
                    btns[j].click(); return true;
                }
            }
            return false;
        }
        // MutationObserver — instant dialog catch
        new MutationObserver(function(muts) {
            for (var m=0; m<muts.length; m++) {
                for (var k=0; k<muts[m].addedNodes.length; k++) {
                    var n = muts[m].addedNodes[k];
                    if (n.nodeType===1) {
                        var tag = (n.tagName||'').toLowerCase();
                        if (tag.indexOf('dialog')>=0 || tag.indexOf('overlay')>=0 ||
                            (n.querySelector && (n.querySelector('[role="dialog"]') || n.querySelector('[role="alertdialog"]')))) {
                            setTimeout(tryDismiss, 200);
                            setTimeout(tryDismiss, 1000);
                        }
                    }
                }
            }
        }).observe(document.body, {childList:true, subtree:true});
        setInterval(tryDismiss, 6000);
        // Web Worker — background-tab proof
        try {
            var w = new Worker(URL.createObjectURL(new Blob(['function t(){postMessage(1);setTimeout(t,20000+Math.floor(Math.random()*15000))}t();'],{type:'application/javascript'})));
            w.onmessage = function(){
                tryDismiss();
                document.body.click();
                var cb = document.querySelector('colab-connect-button')||document.querySelector('#connect');
                if(cb) cb.click();
                document.dispatchEvent(new MouseEvent('mousemove',{clientX:300,clientY:300,bubbles:true}));
            };
        } catch(e) { setInterval(function(){ tryDismiss(); document.body.click(); }, 30000); }
        // Burst on tab hide
        document.addEventListener('visibilitychange', function(){
            if(document.hidden){
                for(var i=0;i<5;i++) setTimeout(function(){
                    document.body.click();
                    document.dispatchEvent(new MouseEvent('mousemove',{clientX:200+Math.floor(Math.random()*400),clientY:200,bubbles:true}));
                }, i*400);
            } else { tryDismiss(); }
        });
        console.log('[KA7] JS layers active');
    })();
    </script>
    <div style="padding:4px 8px;background:#111;border:1px solid #333;border-radius:5px;font:10px monospace;color:#4a4;display:inline-block;">
    ✅ JS dialog dismissal + Web Worker + body.click() active
    </div>
    """))

    # ══════════════════════════════════════════════════════════════
    # MECHANISM 1: Real kernel execution loop via run_cell()
    # + MECHANISM 2: eval_js body click on every cycle
    # ══════════════════════════════════════════════════════════════

    _CELL_CODE = r'''
    import time, os, gc, sys

    _ts = time.strftime("%H:%M:%S")
    _pid = os.getpid()

    # GPU stats
    try:
        import torch
        torch.cuda.synchronize()
        _galloc = torch.cuda.memory_allocated(0) / 1e9
        _gtotal = torch.cuda.get_device_properties(0).total_memory / 1e9
        _gpu = f"VRAM:{_galloc:.1f}/{_gtotal:.1f}GB"
    except Exception:
        _gpu = "GPU:n/a"

    try:
        import subprocess
        _nv = subprocess.run(
            ["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,power.draw","--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if _nv.returncode == 0:
            _p = [x.strip() for x in _nv.stdout.strip().split(",")]
            _gpu += f" util:{_p[0]}% {_p[1]}°C {_p[2]}W"
    except Exception:
        pass

    # Server health
    try:
        import requests
        _srv = "srv:✅" if requests.get("http://127.0.0.1:5000/api/keepalive",timeout=2).status_code==200 else "srv:❌"
    except Exception:
        _srv = "srv:—"

    # Jobs
    try:
        _jc = len(jobs)
        _ac = sum(1 for j in jobs.values() if j.get("status")=="running")
        _jstr = f"jobs:{_jc}({_ac}run)"
    except Exception:
        _jstr = ""

    gc.collect()
    print(f"🔄 {_ts} | {_gpu} | {_srv} | {_jstr} | pid={_pid}")
    '''

    print()
    print("=" * 64)
    print("  🔄 KEEP-ALIVE v7 — TRIPLE MECHANISM")
    print("=" * 64)
    print("  1. Real kernel execution (run_cell) every 20s")
    print("  2. Extension-style body.click() via eval_js")
    print("  3. JS Web Worker + MutationObserver + body.click()")
    print()
    print("  Interrupt (⬜) to stop. Server keeps running.")
    print("=" * 64)
    print()

    _shell = get_ipython()
    _cycle = 0
    _t0 = time.time()

    try:
        while True:
            _cycle += 1
            _elapsed = int(time.time() - _t0)
            _h, _rem = divmod(_elapsed, 3600)
            _m, _s = divmod(_rem, 60)

            # Mechanism 1: Real cell execution
            _shell.run_cell(
                f"# ka cycle {_cycle} | {_h:02d}:{_m:02d}:{_s:02d}\n" + _CELL_CODE,
                silent=False,
                store_history=False,
            )

            # Mechanism 2: Extension-style click via eval_js
            _extension_click()

            time.sleep(20)

    except KeyboardInterrupt:
        print(f"\n🛑 Keep-alive stopped after {_cycle} cycles.")
        print("  Server still running in background.")
        print("  Re-run this cell to restart.")
