# ============================================================
# 🔄 Colab Keep-Alive — Paste this in a cell and run it
# ============================================================
# Simulates organic user activity via JavaScript injection
# so Colab doesn't disconnect when you tab away.
# ============================================================
def run_keep_alive():
    # ============================================================
    # 🔄 Colab Keep-Alive v6 — Self-Executing Cell Loop
    # ============================================================
    #
    # WHY THIS WORKS WHEN NOTHING ELSE DID:
    #
    #   Every previous approach tried to FAKE activity:
    #     - JS mouse/keyboard events → Colab ignores synthetic events
    #     - print() output → doesn't count as user interaction
    #     - Web Workers → browser-side only, kernel still "idle"
    #     - Background threads → Colab tracks CELL execution, not threads
    #
    #   This approach does the ONE thing that cannot be faked:
    #   It uses IPython's own execution machinery to repeatedly
    #   execute real Python code, exactly as if you typed it into
    #   a cell and pressed Shift+Enter. Each execution is a genuine
    #   kernel execution event in Colab's activity log.
    #
    # HOW IT WORKS:
    #
    #   1. This cell starts running
    #   2. It uses get_ipython().run_cell() to execute a code string
    #   3. That code does real work (GPU check, server ping, gc)
    #   4. It sleeps 20 seconds
    #   5. It loops back to step 2
    #
    #   From Colab's perspective, the kernel is continuously
    #   receiving and executing new code — indistinguishable from
    #   a human repeatedly running cells.
    #
    # ALSO INCLUDES:
    #   - JS dialog dismissal (MutationObserver + Web Worker)
    #   - Server health verification
    #   - GPU/memory stats in output
    #
    # PREREQUISITE:
    #   launch_colab.py must have run first (non-blocking).
    #
    # TO STOP: Interrupt this cell (⬜ stop button).
    # ============================================================

    import IPython
    from IPython.display import display, HTML

    # ── Inject JS for dialog dismissal (runs once, persists in page) ──
    display(HTML("""
    <script>
    (function(){
        function tryDismiss() {
            const sels = [
                'paper-button#ok','mwc-button[slot="primaryAction"]',
                'md-text-button[slot="primaryAction"]','md-filled-button[slot="primaryAction"]',
                'colab-dialog paper-button','colab-dialog mwc-button',
            ];
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el && el.offsetParent !== null) { el.click(); return; }
            }
            const btns = document.querySelectorAll('paper-button,mwc-button,md-text-button,md-filled-button,button');
            for (const b of btns) {
                const t = (b.textContent||'').trim().toLowerCase();
                if (['ok','yes','connect','reconnect','dismiss'].includes(t) && b.offsetParent !== null) {
                    b.click(); return;
                }
            }
        }
        // MutationObserver — instant dialog catch
        new MutationObserver(function(muts) {
            for (const m of muts) for (const n of m.addedNodes) {
                if (n.nodeType===1) {
                    const tag = (n.tagName||'').toLowerCase();
                    if (tag.includes('dialog')||tag.includes('overlay')||(n.querySelector&&(n.querySelector('[role="dialog"]')||n.querySelector('[role="alertdialog"]')))) {
                        setTimeout(tryDismiss, 200);
                        setTimeout(tryDismiss, 1000);
                    }
                }
            }
        }).observe(document.body, {childList:true, subtree:true});
        setInterval(tryDismiss, 6000);
        // Web Worker — background-tab proof
        try {
            const w = new Worker(URL.createObjectURL(new Blob([`
                function t(){postMessage(1);setTimeout(t,20000+Math.floor(Math.random()*15000))}t();
            `],{type:'application/javascript'})));
            w.onmessage = function(){
                tryDismiss();
                const cb = document.querySelector('colab-connect-button')||document.querySelector('#connect');
                if(cb)cb.click();
                document.dispatchEvent(new MouseEvent('mousemove',{clientX:300,clientY:300,bubbles:true}));
            };
        } catch(e) { setInterval(function(){tryDismiss()}, 30000); }
        // Burst on tab hide
        document.addEventListener('visibilitychange', function(){
            if(document.hidden){for(let i=0;i<5;i++)setTimeout(function(){
                document.dispatchEvent(new MouseEvent('mousemove',{clientX:100+Math.floor(Math.random()*500),clientY:200,bubbles:true}));
            },i*400);}else{tryDismiss();}
        });
        console.log('[KA6] JS dialog dismissal active');
    })();
    </script>
    <div style="padding:4px 8px;background:#111;border:1px solid #333;border-radius:5px;font:10px monospace;color:#4a4;display:inline-block;">
    ✅ JS dialog dismissal + Web Worker active
    </div>
    """))

    # ══════════════════════════════════════════════════════════════
    # THE CELL CODE — this string gets executed repeatedly
    # ══════════════════════════════════════════════════════════════

    _KEEPALIVE_CODE = r'''
    import time, os, gc, sys

    # ── Real work that Colab can verify ──
    _ts = time.strftime("%H:%M:%S")
    _pid = os.getpid()

    # GPU stats
    try:
        import torch
        torch.cuda.synchronize()  # force GPU sync — real CUDA call
        _galloc = torch.cuda.memory_allocated(0) / 1e9
        _gtotal = torch.cuda.get_device_properties(0).total_memory / 1e9
        _gpu = f"VRAM: {_galloc:.1f}/{_gtotal:.1f}GB"
    except Exception:
        _gpu = "GPU: n/a"

    # nvidia-smi for utilization
    try:
        import subprocess
        _nv = subprocess.run(
            ["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,power.draw","--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if _nv.returncode == 0:
            _parts = [p.strip() for p in _nv.stdout.strip().split(",")]
            _gpu += f" | util:{_parts[0]}% temp:{_parts[1]}°C pwr:{_parts[2]}W"
    except Exception:
        pass

    # Server health
    try:
        import requests
        _sr = requests.get("http://127.0.0.1:5000/api/keepalive", timeout=2)
        _srv = "srv:✅" if _sr.status_code == 200 else "srv:❌"
    except Exception:
        _srv = "srv:—"

    # Job count
    try:
        _jc = len(jobs)
        _ac = sum(1 for j in jobs.values() if j.get("status") == "running")
        _jstr = f"jobs:{_jc}({_ac}active)"
    except Exception:
        _jstr = "jobs:?"

    # RAM
    try:
        import psutil
        _mem = psutil.virtual_memory()
        _ram = f"RAM: {_mem.used/1e9:.1f}/{_mem.total/1e9:.1f}GB ({_mem.percent}%)"
    except ImportError:
        try:
            with open("/proc/meminfo") as _f:
                _mi = {}
                for _line in _f:
                    _p = _line.split()
                    if len(_p) >= 2: _mi[_p[0].rstrip(":")] = int(_p[1])
                _rt = _mi.get("MemTotal",0)
                _ra = _mi.get("MemAvailable",_mi.get("MemFree",0))
                _ram = f"RAM: {(_rt-_ra)/1e6:.1f}/{_rt/1e6:.1f}GB ({round((_rt-_ra)/_rt*100,1)}%)"
        except Exception:
            _ram = "RAM: ?"

    # GC — real work
    gc.collect()

    # CPU
    try:
        _load = os.getloadavg()[0]
        _cores = os.cpu_count() or 1
        _cpu = f"CPU: {round(_load/_cores*100,1)}% (load:{_load:.1f}/{_cores}cores)"
    except Exception:
        _cpu = "CPU: ?"

    # Print — real output with newline
    print(f"🔄 {_ts} | {_gpu} | {_cpu} | {_ram} | {_srv} | {_jstr} | pid={_pid}")
    '''

    # ══════════════════════════════════════════════════════════════
    # MAIN LOOP — repeatedly execute real code via IPython
    # ══════════════════════════════════════════════════════════════

    print()
    print("=" * 64)
    print("  🔄 KEEP-ALIVE v6 — SELF-EXECUTING CELL LOOP")
    print("=" * 64)
    print("  Each cycle = real IPython kernel execution.")
    print("  Colab CANNOT ignore this. Interrupt (⬜) to stop.")
    print("=" * 64)
    print()

    _shell = get_ipython()
    _cycle = 0
    _t0 = __import__('time').time()

    try:
        while True:
            _cycle += 1
            _elapsed = int(__import__('time').time() - _t0)
            _h, _rem = divmod(_elapsed, 3600)
            _m, _s = divmod(_rem, 60)

            # Execute the keepalive code as a REAL cell execution
            # This is indistinguishable from a user pressing Shift+Enter
            _result = _shell.run_cell(
                f"# ── keepalive cycle {_cycle} | uptime {_h:02d}:{_m:02d}:{_s:02d} ──\n"
                + _KEEPALIVE_CODE,
                silent=False,  # produce output — Colab sees this
                store_history=False,  # don't pollute In/Out history
            )

            # Sleep 20 seconds before next execution
            __import__('time').sleep(20)

    except KeyboardInterrupt:
        print(f"\n🛑 Keep-alive stopped after {_cycle} cycles.")
        print("  Server still running in background thread.")
        print("  Re-run this cell to restart.")