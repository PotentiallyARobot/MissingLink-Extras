# ============================================================
# 🔄 Colab Keep-Alive — Paste this in a cell and run it
# ============================================================
# Simulates organic user activity via JavaScript injection
# so Colab doesn't disconnect when you tab away.
# ============================================================
def run_keep_alive():
    # ============================================================
    # 🔄 Colab Keep-Alive v5 — Real Kernel Execution Loop
    # ============================================================
    #
    # WHY EVERYTHING ELSE FAILED:
    #
    #   ❌ print() output does NOT count as user activity
    #   ❌ JS timers get throttled to 1/min in background tabs
    #   ❌ Web Workers help but Colab can still kill idle kernels
    #   ❌ Fake mouse/keyboard events don't fool Colab's backend
    #
    # THE ONE THING THAT CANNOT BE FAKED:
    #
    #   ✅ Real kernel execution — an actual cell running Python code
    #      and producing output. Colab MUST keep the runtime alive
    #      while a cell is actively executing.
    #
    # HOW IT WORKS:
    #
    #   This cell runs a blocking while-loop that:
    #   1. Sleeps 25 seconds
    #   2. Executes real Python (GPU health check, memory stats)
    #   3. Prints real output with newlines
    #   4. Pings the Flask server to verify it's still up
    #   5. Runs JS dialog dismissal via IPython.display
    #
    #   Because this cell is ACTIVELY EXECUTING, Colab's runtime
    #   monitor sees a busy kernel and won't recycle it.
    #
    # PREREQUISITE:
    #   launch_colab.py must have run first (non-blocking).
    #   The Flask server runs in a daemon thread.
    #   This cell keeps the kernel "busy" so Colab doesn't kill it.
    #
    # TO STOP: Interrupt this cell (⬜ stop button or Ctrl+C).
    #          The Flask server will keep running until runtime dies.
    # ============================================================

    import time
    import sys
    import os
    import gc
    import threading

    from IPython.display import display, HTML, clear_output

    # ── Check that the server is actually running ──
    try:
        import requests as _req
        _r = _req.get("http://127.0.0.1:5000/api/keepalive", timeout=2)
        assert _r.status_code == 200
        print("✅ Flask server is alive")
    except Exception:
        print("⚠ Flask server not detected on :5000 — did you run launch_colab.py first?")
        print("  This cell will still run as a keep-alive but the UI won't be accessible.")

    # ── JS injection: dialog dismissal + interaction sim ──
    # This runs once when the cell starts. The JS persists in the page.
    _JS_KEEPALIVE = """
    <script>
    (function(){
        // MutationObserver for instant dialog dismissal
        function tryDismiss() {
            const sels = [
                'paper-button#ok','mwc-button[slot="primaryAction"]',
                'md-text-button[slot="primaryAction"]','md-filled-button[slot="primaryAction"]',
                'colab-dialog paper-button','colab-dialog mwc-button',
            ];
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el && el.offsetParent !== null) { el.click(); return true; }
            }
            const btns = document.querySelectorAll('paper-button,mwc-button,md-text-button,md-filled-button,button');
            for (const b of btns) {
                const t = (b.textContent||'').trim().toLowerCase();
                if (['ok','yes','connect','reconnect','dismiss'].includes(t) && b.offsetParent !== null) {
                    b.click(); return true;
                }
            }
            return false;
        }

        const obs = new MutationObserver(function(muts) {
            for (const m of muts) for (const n of m.addedNodes) {
                if (n.nodeType===1) {
                    const tag = (n.tagName||'').toLowerCase();
                    if (tag.includes('dialog')||tag.includes('overlay')||
                        (n.querySelector && (n.querySelector('[role="dialog"]')||n.querySelector('[role="alertdialog"]')))) {
                        setTimeout(tryDismiss, 200);
                        setTimeout(tryDismiss, 1000);
                    }
                }
            }
        });
        obs.observe(document.body, {childList:true, subtree:true});
        setInterval(tryDismiss, 6000);

        // Web Worker for background-tab-proof interaction
        try {
            const blob = new Blob([`
                function tick() {
                    postMessage('t');
                    setTimeout(tick, 20000 + Math.floor(Math.random()*15000));
                }
                tick();
            `], {type:'application/javascript'});
            const w = new Worker(URL.createObjectURL(blob));
            w.onmessage = function() {
                tryDismiss();
                const cb = document.querySelector('colab-connect-button')||document.querySelector('#connect');
                if (cb) cb.click();
                const x = 100+Math.floor(Math.random()*600);
                const y = 100+Math.floor(Math.random()*400);
                document.dispatchEvent(new MouseEvent('mousemove',{clientX:x,clientY:y,bubbles:true}));
                window.dispatchEvent(new MouseEvent('mousemove',{clientX:x,clientY:y,bubbles:true}));
            };
        } catch(e) {
            setInterval(function(){
                tryDismiss();
                document.dispatchEvent(new MouseEvent('mousemove',{clientX:300,clientY:300,bubbles:true}));
            }, 30000);
        }

        // Burst on tab hide
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                for (let i=0;i<5;i++) setTimeout(function(){
                    document.dispatchEvent(new MouseEvent('mousemove',{
                        clientX:100+Math.floor(Math.random()*500),
                        clientY:100+Math.floor(Math.random()*400),bubbles:true}));
                    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Shift',code:'ShiftLeft',bubbles:true}));
                    document.dispatchEvent(new KeyboardEvent('keyup',{key:'Shift',code:'ShiftLeft',bubbles:true}));
                }, i*400);
            } else { tryDismiss(); }
        });

        console.log('[KA5] JS layers active: MutationObserver + Worker + VisibilityAPI');
    })();
    </script>
    """
    display(HTML(_JS_KEEPALIVE))

    # ══════════════════════════════════════════════════════════════
    # MAIN KEEP-ALIVE LOOP
    # ══════════════════════════════════════════════════════════════
    # This is the critical part. The cell stays in "executing" state.
    # Every 25s it does REAL WORK that Colab's backend can verify.

    print()
    print("=" * 60)
    print("  🔄 KEEP-ALIVE v5 — REAL KERNEL EXECUTION LOOP")
    print("=" * 60)
    print("  This cell will keep running. Do NOT stop it.")
    print("  Server continues in background thread.")
    print("  Interrupt (⬜) only when you're done.")
    print("=" * 60)
    print()

    _cycle = 0
    _start = time.time()

    try:
        while True:
            time.sleep(25)
            _cycle += 1
            _uptime = int(time.time() - _start)
            _h, _rem = divmod(_uptime, 3600)
            _m, _s = divmod(_rem, 60)

            # ── Real work: things Colab can verify are genuine ──

            # 1. Memory stats (actual computation)
            try:
                import torch
                _gpu_alloc = torch.cuda.memory_allocated() / 1e9
                _gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                _gpu_str = f"GPU: {_gpu_alloc:.1f}/{_gpu_total:.1f}GB"
            except Exception:
                _gpu_str = "GPU: n/a"

            # 2. Server health check (real network I/O)
            try:
                _sr = _req.get("http://127.0.0.1:5000/api/keepalive", timeout=2)
                _srv = "srv:✅" if _sr.status_code == 200 else "srv:❌"
            except Exception:
                _srv = "srv:💀"

            # 3. Job count (real state inspection)
            try:
                _jcount = len(jobs)
                _active = sum(1 for j in jobs.values() if j.get("status") == "running")
                _job_str = f"jobs:{_jcount}({_active} active)"
            except Exception:
                _job_str = "jobs:?"

            # 4. Garbage collection (real work)
            gc.collect()

            # 5. Print with REAL NEWLINE — proof of execution
            print(
                f"🔄 #{_cycle:>4d} | "
                f"{_h:02d}:{_m:02d}:{_s:02d} | "
                f"{_gpu_str} | "
                f"{_srv} | "
                f"{_job_str} | "
                f"pid={os.getpid()}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n🛑 Keep-alive stopped. Server still running in background.")
        print("  Runtime will disconnect after ~90 min of no cell execution.")
        print("  Re-run this cell to restart keep-alive.")