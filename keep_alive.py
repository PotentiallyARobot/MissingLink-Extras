# ============================================================
# 🔄 Colab Keep-Alive — Paste this in a cell and run it
# ============================================================
# Simulates organic user activity via JavaScript injection
# so Colab doesn't disconnect when you tab away.
# ============================================================
def run_keep_alive():
    # ============================================================
    # 🔄 Colab Keep-Alive v4 — Nuclear Anti-Disconnect
    # ============================================================
    #
    # WHY PREVIOUS VERSIONS FAILED:
    #
    #   ❌ print() / cell output does NOT count as user interaction.
    #      Colab's idle timer only resets on browser-level events
    #      (mouse, keyboard, scroll, focus). Python output is ignored.
    #
    #   ❌ JS setTimeout/setInterval get THROTTLED by Chrome when
    #      the tab is backgrounded. Timers slow to 1/min or stop
    #      entirely, so your 30s click cycle becomes 10+ minutes.
    #
    #   ❌ Simple click-connect scripts target the wrong selectors
    #      after Colab UI updates, silently do nothing, and you
    #      only find out when you come back to a dead session.
    #
    # WHAT THIS VERSION DOES DIFFERENTLY:
    #
    #   Layer 1: Web Worker timer — NOT throttled in background tabs.
    #            Chrome throttles setTimeout in inactive tabs but
    #            Web Workers keep running at full speed.
    #
    #   Layer 2: MutationObserver — watches the DOM for disconnect
    #            dialogs the instant they appear (not on a timer).
    #            Catches them in <100ms vs 5-60s with polling.
    #
    #   Layer 3: Visibility API handler — fires a burst of activity
    #            events the moment the tab becomes hidden (user
    #            switches away). This resets the idle clock right
    #            at the critical moment.
    #
    #   Layer 4: Shadow iframe — opens a tiny hidden iframe to the
    #            same Colab page. Even if the main tab's JS gets
    #            throttled, the iframe's scripts may still fire.
    #
    #   Layer 5: Python heartbeat — real print() output every 30s
    #            from the kernel. While this alone doesn't prevent
    #            idle timeout, it prevents runtime recycling and
    #            provides visible proof the kernel is alive.
    #
    #   Layer 6: Broad organic interaction sim — mouse, keyboard,
    #            scroll, focus, all dispatched with randomized
    #            timing and coordinates.
    #
    # USAGE:
    #   Run this cell. Then run your generation/training cell.
    #   You can switch tabs. Keep the Colab tab OPEN (not closed).
    #   Nothing can help if you close the tab entirely.
    #
    # LIMITATIONS:
    #   - Cannot bypass the hard 12h (free) / 24h (Pro) session cap
    #   - Cannot solve CAPTCHAs if Google presents one
    #   - Tab must remain open (minimized/backgrounded is fine)
    # ============================================================

    import threading
    import time
    import sys
    import os

    from IPython.display import display, HTML

    # ══════════════════════════════════════════════════════════════
    # LAYER 5: Python Heartbeat Thread
    # ══════════════════════════════════════════════════════════════
    # Prints real newlines to prove kernel is alive.
    # This alone does NOT prevent idle disconnect, but it prevents
    # Colab from recycling the runtime thinking it's dead.

    _ka_active = True
    _ka_counter = 0

    def _ka_heartbeat():
        global _ka_counter
        while _ka_active:
            time.sleep(30)
            _ka_counter += 1
            # Real print() with newline — proof of life
            print(f"💓 ka#{_ka_counter} {time.strftime('%H:%M:%S')} alive | pid={os.getpid()}")

    _ka_thread = threading.Thread(target=_ka_heartbeat, daemon=True, name='keepalive-hb')
    _ka_thread.start()
    print("✅ Layer 5: Python heartbeat started (30s)")

    # ══════════════════════════════════════════════════════════════
    # LAYERS 1-4, 6: JavaScript (injected via IPython.display)
    # ══════════════════════════════════════════════════════════════

    _KA_HTML = r"""
    <div id="ka4" style="
        position:fixed;bottom:6px;left:6px;z-index:999999;
        padding:4px 8px;background:rgba(0,0,0,.85);border:1px solid #333;
        border-radius:5px;font:10px/1.3 monospace;color:#4a4;
        pointer-events:none;opacity:.6;max-width:360px;
    ">⏳ KA v4 init…</div>

    <script>
    (function(){
    'use strict';

    const B = document.getElementById('ka4');
    let n = 0;

    function log(m) {
        n++;
        const t = new Date().toLocaleTimeString();
        if (B) B.textContent = '✅ KA4 #'+n+' | '+m+' | '+t;
    }

    function rand(a, b) { return Math.floor(Math.random()*(b-a))+a; }

    // ─────────────────────────────────────────────────
    // LAYER 1: Web Worker timer (immune to tab throttle)
    // ─────────────────────────────────────────────────
    // Chrome throttles setTimeout to 1/min in background tabs.
    // Web Workers are NOT throttled. We use a Worker to send
    // reliable tick messages even when the tab is hidden.

    let workerURL = null;
    let worker = null;
    try {
        const blob = new Blob([`
            let iv = 25000; // 25 seconds
            function tick() {
                postMessage('tick');
                setTimeout(tick, iv + Math.floor(Math.random()*20000));
            }
            tick();
        `], {type: 'application/javascript'});
        workerURL = URL.createObjectURL(blob);
        worker = new Worker(workerURL);
        worker.onmessage = function() { doActivity('worker-tick'); };
        log('Worker timer active');
    } catch(e) {
        // Fallback: use setInterval (will be throttled but better than nothing)
        setInterval(function(){ doActivity('interval-fallback'); }, 30000);
        log('Worker failed, using setInterval');
    }

    // ─────────────────────────────────────────────────
    // LAYER 2: MutationObserver (instant dialog catch)
    // ─────────────────────────────────────────────────
    // Instead of polling every 5s, we watch the DOM tree for
    // any dialog/overlay being inserted and dismiss it instantly.

    function tryDismiss() {
        // Broad selector list covering Colab UI variants 2023-2026
        const sels = [
            'paper-button#ok',
            'mwc-button[slot="primaryAction"]',
            'md-text-button[slot="primaryAction"]',
            'md-filled-button[slot="primaryAction"]',
            '.yes-no-dialog paper-button:last-child',
            'colab-dialog paper-button',
            'colab-dialog mwc-button',
        ];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el && el.offsetParent !== null) {
                el.click();
                log('DISMISSED dialog');
                return true;
            }
        }
        // Text-match fallback
        const allBtns = document.querySelectorAll(
            'paper-button, mwc-button, md-text-button, md-filled-button, button'
        );
        for (const btn of allBtns) {
            const txt = (btn.textContent||'').trim().toLowerCase();
            if (['ok','yes','connect','reconnect','dismiss'].includes(txt)) {
                if (btn.offsetParent !== null) {
                    btn.click();
                    log('Clicked: '+txt);
                    return true;
                }
            }
        }
        return false;
    }

    const observer = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === 1) {
                    // Check if the new node IS a dialog or CONTAINS one
                    const tag = (node.tagName||'').toLowerCase();
                    if (tag.includes('dialog') || tag.includes('overlay') ||
                        node.querySelector && (
                            node.querySelector('paper-dialog') ||
                            node.querySelector('colab-dialog') ||
                            node.querySelector('mwc-dialog') ||
                            node.querySelector('[role="alertdialog"]') ||
                            node.querySelector('[role="dialog"]')
                        )) {
                        // Small delay for dialog to fully render its buttons
                        setTimeout(tryDismiss, 200);
                        setTimeout(tryDismiss, 800);
                        setTimeout(tryDismiss, 2000);
                    }
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    log('MutationObserver watching');

    // Also poll as safety net (MutationObserver might miss shadow DOM)
    setInterval(tryDismiss, 8000);

    // ─────────────────────────────────────────────────
    // LAYER 3: Visibility API — burst on tab hide
    // ─────────────────────────────────────────────────
    // The moment you switch away from the Colab tab,
    // fire a burst of activity to max out the idle clock.

    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            log('Tab hidden — burst');
            // Fire a rapid burst of events
            for (let i = 0; i < 5; i++) {
                setTimeout(function() { doActivity('hide-burst-'+i); }, i*500);
            }
        } else {
            log('Tab visible');
            doActivity('tab-return');
            tryDismiss();
        }
    });

    // ─────────────────────────────────────────────────
    // LAYER 4: Shadow iframe (backup execution context)
    // ─────────────────────────────────────────────────
    // A 1x1 hidden iframe loading the same page. Its JS context
    // is separate and may not be throttled the same way. It also
    // keeps a second "connection" to the Colab backend alive.

    try {
        const iframe = document.createElement('iframe');
        iframe.style.cssText = 'width:1px;height:1px;position:fixed;bottom:0;left:0;opacity:0;pointer-events:none;border:none;';
        iframe.src = window.location.href;
        document.body.appendChild(iframe);
        log('Shadow iframe injected');
    } catch(e) {
        log('Shadow iframe failed: '+e.message);
    }

    // ─────────────────────────────────────────────────
    // LAYER 6: Organic interaction simulation
    // ─────────────────────────────────────────────────

    function doActivity(src) {
        try {
            // Always try dismiss first
            if (tryDismiss()) return;

            // Click connect button if present
            const cb = document.querySelector('colab-connect-button')
                || document.querySelector('#connect')
                || document.querySelector('colab-toolbar-button#connect');
            if (cb) cb.click();

            // Pick a random action
            const roll = Math.random();

            if (roll < 0.25) {
                // Mouse move
                const x = rand(50, window.innerWidth-50);
                const y = rand(50, window.innerHeight-50);
                document.dispatchEvent(new MouseEvent('mousemove', {
                    clientX:x, clientY:y, bubbles:true, cancelable:true
                }));
                // Also fire on window and body for good measure
                window.dispatchEvent(new MouseEvent('mousemove', {
                    clientX:x, clientY:y, bubbles:true
                }));
                log(src+' mouse '+x+','+y);

            } else if (roll < 0.45) {
                // Focus a code cell editor
                const cells = document.querySelectorAll(
                    '.cell .CodeMirror, .cell .cm-editor, .codecell-input-output'
                );
                if (cells.length) {
                    const c = cells[rand(0, cells.length)];
                    c.dispatchEvent(new MouseEvent('mousedown', {bubbles:true}));
                    c.dispatchEvent(new MouseEvent('mouseup', {bubbles:true}));
                    c.dispatchEvent(new MouseEvent('click', {bubbles:true}));
                    c.dispatchEvent(new FocusEvent('focus', {bubbles:true}));
                }
                log(src+' focus cell');

            } else if (roll < 0.6) {
                // Keyboard event (Shift — does nothing visible)
                const kd = {key:'Shift',code:'ShiftLeft',keyCode:16,which:16,bubbles:true};
                document.dispatchEvent(new KeyboardEvent('keydown', kd));
                document.dispatchEvent(new KeyboardEvent('keyup', kd));
                log(src+' keypress');

            } else if (roll < 0.75) {
                // Scroll the page slightly
                window.scrollBy({top: rand(-5,5), behavior:'smooth'});
                log(src+' scroll');

            } else if (roll < 0.85) {
                // Click on toolbar area
                const tb = document.querySelector('#toolbar, colab-toolbar, .colab-header');
                if (tb) {
                    tb.dispatchEvent(new MouseEvent('click', {bubbles:true}));
                }
                log(src+' toolbar');

            } else {
                // Touch events (for mobile Colab / touch detection)
                try {
                    const x = rand(100, window.innerWidth-100);
                    const y = rand(100, window.innerHeight-100);
                    const touch = new Touch({
                        identifier: Date.now(),
                        target: document.body,
                        clientX: x, clientY: y,
                        pageX: x, pageY: y,
                    });
                    document.dispatchEvent(new TouchEvent('touchstart', {
                        touches: [touch], changedTouches: [touch], bubbles: true
                    }));
                    document.dispatchEvent(new TouchEvent('touchend', {
                        touches: [], changedTouches: [touch], bubbles: true
                    }));
                } catch(te) {}
                log(src+' touch');
            }

        } catch(e) {
            log('ERR: '+e.message);
        }
    }

    // ─────────────────────────────────────────────────
    // ANTI-THROTTLE: requestAnimationFrame chain
    // ─────────────────────────────────────────────────
    // rAF is paused when tab is hidden, but resumes instantly
    // when the tab becomes visible again. We use it to fire
    // an activity burst the moment the user comes back.

    let lastRAF = Date.now();
    function rafLoop() {
        const now = Date.now();
        // If >60s since last rAF, we were probably backgrounded
        if (now - lastRAF > 60000) {
            log('rAF resume — burst');
            doActivity('raf-resume');
            tryDismiss();
        }
        lastRAF = now;
        requestAnimationFrame(rafLoop);
    }
    requestAnimationFrame(rafLoop);

    // ─────────────────────────────────────────────────
    // BOOT
    // ─────────────────────────────────────────────────

    log('v4 ALL LAYERS ACTIVE');
    console.log('[KA4] Layers: Worker(1) MutationObserver(2) VisibilityAPI(3) ShadowIframe(4) Heartbeat(5) Interaction(6) rAF(anti-throttle)');

    })();
    </script>
    """

    display(HTML(_KA_HTML))

    print()
    print("=" * 60)
    print("  🔄 KEEP-ALIVE v4 — NUCLEAR MODE")
    print("=" * 60)
    print()
    print("  Layer 1: Web Worker timer     ✅ NOT throttled in bg")
    print("  Layer 2: MutationObserver     ✅ instant dialog catch")
    print("  Layer 3: Visibility API       ✅ burst on tab switch")
    print("  Layer 4: Shadow iframe        ✅ backup JS context")
    print("  Layer 5: Python heartbeat     ✅ kernel proof-of-life")
    print("  Layer 6: Organic interaction  ✅ mouse/kb/scroll/touch")
    print("  Bonus:   rAF anti-throttle   ✅ burst on tab return")
    print()
    print("  ⚠  Cannot bypass 12h/24h hard session cap")
    print("  ⚠  Cannot solve CAPTCHAs — manual intervention needed")
    print("  ⚠  Tab must stay OPEN (minimized/bg is fine)")
    print("=" * 60)