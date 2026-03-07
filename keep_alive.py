# ============================================================
# 🔄 Colab Keep-Alive — Paste this in a cell and run it
# ============================================================
# Simulates organic user activity via JavaScript injection
# so Colab doesn't disconnect when you tab away.
# ============================================================
def run_keep_alive():
    from IPython.display import display, HTML
    import random

    keep_alive_html = """
    <div id="keep-alive-status" style="
        padding: 8px 14px;
        background: #1a1a2e;
        color: #0f0;
        font-family: monospace;
        font-size: 13px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px 0;
    ">⏳ Keep-Alive: Initializing...</div>
    
    <script>
    (function() {
        const STATUS = document.getElementById('keep-alive-status');
        let clickCount = 0;
        let lastAction = '';
    
        // Random interval between min and max ms
        function randInterval(minMs, maxMs) {
            return Math.floor(Math.random() * (maxMs - minMs)) + minMs;
        }
    
        // Update the status badge
        function updateStatus(action) {
            clickCount++;
            lastAction = action;
            const now = new Date().toLocaleTimeString();
            STATUS.textContent = `✅ Keep-Alive active | #${clickCount} | ${action} | ${now}`;
            STATUS.style.color = '#0f0';
        }
    
        // --- Action pool: each mimics a real user gesture ---
    
        function clickConnect() {
            // Click the "Connect" / "Reconnect" button if visible
            const btn = document.querySelector("colab-connect-button")
                || document.querySelector("#connect")
                || document.querySelector("colab-toolbar-button#connect");
            if (btn) {
                btn.click();
                updateStatus("Clicked connect button");
                return true;
            }
            return false;
        }
    
        function dismissDialog() {
            // Close any "runtime disconnected" or idle-timeout dialogs
            const okBtn = document.querySelector("paper-button#ok")
                || document.querySelector("mwc-button[slot='primaryAction']");
            if (okBtn) {
                okBtn.click();
                updateStatus("Dismissed dialog");
                return true;
            }
            return false;
        }
    
        function scrollCodeCell() {
            // Scroll a random code cell slightly — mimics reading
            const cells = document.querySelectorAll(".codecell-input-output");
            if (cells.length > 0) {
                const cell = cells[Math.floor(Math.random() * cells.length)];
                cell.scrollTop += Math.floor(Math.random() * 30) - 15;
                updateStatus("Scrolled code cell");
                return true;
            }
            return false;
        }
    
        function focusEditor() {
            // Click into a CodeMirror editor to simulate focus
            const editors = document.querySelectorAll(".CodeMirror");
            if (editors.length > 0) {
                const ed = editors[Math.floor(Math.random() * editors.length)];
                const cm = ed.CodeMirror;
                if (cm) {
                    cm.focus();
                    // Move cursor slightly
                    const pos = cm.getCursor();
                    cm.setCursor({line: pos.line, ch: pos.ch});
                    updateStatus("Focused editor");
                    return true;
                }
            }
            return false;
        }
    
        function clickToolbar() {
            // Click the main toolbar area (harmless, keeps session alive)
            const toolbar = document.querySelector("#toolbar")
                || document.querySelector("colab-toolbar");
            if (toolbar) {
                toolbar.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                updateStatus("Clicked toolbar");
                return true;
            }
            return false;
        }
    
        function simulateMouseMove() {
            // Fire a mousemove on the document body
            const x = Math.floor(Math.random() * window.innerWidth);
            const y = Math.floor(Math.random() * window.innerHeight);
            document.dispatchEvent(new MouseEvent('mousemove', {
                clientX: x, clientY: y, bubbles: true
            }));
            updateStatus(`Mouse move (${x}, ${y})`);
            return true;
        }
    
        function simulateKeypress() {
            // Send a harmless Shift keypress (no character inserted)
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Shift', code: 'ShiftLeft', bubbles: true
            }));
            document.dispatchEvent(new KeyboardEvent('keyup', {
                key: 'Shift', code: 'ShiftLeft', bubbles: true
            }));
            updateStatus("Keypress (Shift)");
            return true;
        }
    
        function scrollPage() {
            window.scrollBy(0, Math.floor(Math.random() * 10) - 5);
            updateStatus("Page scroll");
            return true;
        }
    
        // --- Scheduler ---
    
        const actions = [
            { fn: dismissDialog,    weight: 10 },  // highest priority
            { fn: clickConnect,     weight: 9  },
            { fn: focusEditor,      weight: 5  },
            { fn: simulateMouseMove, weight: 4 },
            { fn: scrollCodeCell,   weight: 3  },
            { fn: simulateKeypress, weight: 3  },
            { fn: clickToolbar,     weight: 2  },
            { fn: scrollPage,       weight: 2  },
        ];
    
        function pickAction() {
            // Weighted random selection
            const totalWeight = actions.reduce((s, a) => s + a.weight, 0);
            let r = Math.random() * totalWeight;
            for (const a of actions) {
                r -= a.weight;
                if (r <= 0) return a.fn;
            }
            return actions[0].fn;
        }
    
        function tick() {
            try {
                // Always check for dialogs first
                if (!dismissDialog()) {
                    if (!clickConnect() || Math.random() > 0.3) {
                        const action = pickAction();
                        action();
                    }
                }
            } catch (e) {
                STATUS.textContent = `⚠️ Keep-Alive error: ${e.message}`;
                STATUS.style.color = '#ff0';
            }
    
            // Schedule next tick at a human-ish random interval (30s–90s)
            const next = randInterval(30000, 90000);
            setTimeout(tick, next);
        }
    
        // Start after a short delay
        setTimeout(tick, 5000);
        STATUS.textContent = "✅ Keep-Alive active | Waiting for first tick...";
        STATUS.style.color = '#0f0';
    
        console.log("[Keep-Alive] Started. Actions will fire every 30–90s.");
    })();
    </script>
    """

    display(HTML(keep_alive_html))
    print("✅ Keep-Alive is running. Leave this cell's output visible.")
    print("   Actions fire every 30–90s with randomized intervals.")
    print("   You can safely switch tabs now.")