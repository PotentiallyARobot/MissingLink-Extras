"""
PromptFlow Interpreter v1.2
===========================
Parses and executes .promptflow YAML config files.

Step types:
  llm_call       — single LLM call, returns raw text (local) or parsed text/JSON (openai)
  llm_call_loop  — calls the LLM once per item in a list; collects raw text responses
                   into a list. Local LLM NEVER asked for JSON here.
  code           — arbitrary Python executed in the flow context
  code_loop      — iterates a list, runs actions (image_gen, log, display_image) per item

LLM routing:
  provider=local  → POST /llm/chat on model_server. Raw text only. No parse_json.
  provider=openai → OpenAI API. parse_json supported.

GPU lifecycle:
  Reads llm.gpu_lifecycle config and calls /llm/offload before the first code_loop
  (image generation) and optionally /llm/reload after.
"""

import gc
import os
import re
import json
import time
import yaml
import base64
import requests
import argparse
import textwrap
import traceback
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
        def log(self, *a, **kw): print(*a)
    console = _FallbackConsole()


# ─────────────────────────── Colab Keep-Alive (background) ────────────────────
# Runs as a daemon thread so it doesn't block the promptflow cell.
# Uses three mechanisms:
#   1. Real kernel execution via get_ipython().run_cell() every 25s
#   2. eval_js body.click() via Colab's kernel→frontend bridge
#   3. One-time JS injection for MutationObserver dialog dismissal
# The thread starts automatically when the interpreter runs and stops when done.

_KEEPALIVE_THREAD: Optional[threading.Thread] = None
_KEEPALIVE_STOP = threading.Event()


def _colab_keepalive_loop():
    """Background thread: keeps Colab alive while long GPU work runs."""
    # ── Check if we're actually in Colab ──
    try:
        ipython = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return  # Not in a notebook at all

    in_colab = False
    eval_js_fn = None
    try:
        from google.colab.output import eval_js as _ejs
        eval_js_fn = _ejs
        in_colab = True
    except ImportError:
        pass

    if not in_colab:
        return  # Not in Colab — no keep-alive needed

    # ── One-time JS injection for dialog dismissal ──
    try:
        from IPython.display import display, HTML
        display(HTML("""<script>
(function(){
    function tryDismiss() {
        var sels = ['paper-button#ok','mwc-button[slot="primaryAction"]',
            'md-text-button[slot="primaryAction"]','md-filled-button[slot="primaryAction"]',
            'colab-dialog paper-button','colab-dialog mwc-button'];
        for (var i=0; i<sels.length; i++) {
            var el = document.querySelector(sels[i]);
            if (el && el.offsetParent !== null) { el.click(); return; }
        }
        var btns = document.querySelectorAll('paper-button,mwc-button,md-text-button,md-filled-button,button');
        for (var j=0; j<btns.length; j++) {
            var t = (btns[j].textContent||'').trim().toLowerCase();
            if (['ok','yes','connect','reconnect','dismiss'].indexOf(t) >= 0 && btns[j].offsetParent !== null) {
                btns[j].click(); return;
            }
        }
    }
    new MutationObserver(function(muts) {
        for (var m=0; m<muts.length; m++)
            for (var k=0; k<muts[m].addedNodes.length; k++) {
                var n = muts[m].addedNodes[k];
                if (n.nodeType===1) {
                    var tag = (n.tagName||'').toLowerCase();
                    if (tag.indexOf('dialog')>=0 || tag.indexOf('overlay')>=0 ||
                        (n.querySelector && (n.querySelector('[role="dialog"]') || n.querySelector('[role="alertdialog"]')))) {
                        setTimeout(tryDismiss, 200); setTimeout(tryDismiss, 1000);
                    }
                }
            }
    }).observe(document.body, {childList:true, subtree:true});
    setInterval(tryDismiss, 6000);
    try {
        var w = new Worker(URL.createObjectURL(new Blob(
            ['function t(){postMessage(1);setTimeout(t,20000+Math.floor(Math.random()*15000))}t();'],
            {type:'application/javascript'})));
        w.onmessage = function(){ tryDismiss(); document.body.click(); };
    } catch(e) { setInterval(function(){ tryDismiss(); document.body.click(); }, 30000); }
    document.addEventListener('visibilitychange', function(){
        if(document.hidden) for(var i=0;i<5;i++) setTimeout(function(){ document.body.click(); }, i*400);
        else tryDismiss();
    });
    console.log('[PromptFlow KA] JS dialog-dismiss active');
})();
</script><div style="padding:2px 6px;background:#111;border:1px solid #333;border-radius:4px;font:9px monospace;color:#4a4;display:inline-block">🔄 keep-alive active</div>"""))
    except Exception:
        pass

    # ── Heartbeat loop ──
    _mini_cell = "import gc; gc.collect()  # promptflow heartbeat\n"
    cycle = 0
    while not _KEEPALIVE_STOP.wait(timeout=25):
        cycle += 1
        # Mechanism 1: real kernel execution
        try:
            ipython.run_cell(_mini_cell, silent=True, store_history=False)
        except Exception:
            pass
        # Mechanism 2: eval_js body click
        if eval_js_fn is not None:
            try:
                eval_js_fn("""(function(){
                    document.body.click();
                    var cb = document.querySelector('colab-connect-button') || document.querySelector('#connect');
                    if(cb) cb.click();
                    document.dispatchEvent(new MouseEvent('mousemove',{
                        clientX: 100+Math.floor(Math.random()*600),
                        clientY: 100+Math.floor(Math.random()*400), bubbles:true}));
                })();""", ignore_result=True)
            except Exception:
                pass


def start_keepalive():
    """Start the background keep-alive thread (idempotent)."""
    global _KEEPALIVE_THREAD
    _KEEPALIVE_STOP.clear()
    if _KEEPALIVE_THREAD is not None and _KEEPALIVE_THREAD.is_alive():
        return  # already running
    _KEEPALIVE_THREAD = threading.Thread(target=_colab_keepalive_loop, daemon=True,
                                         name="promptflow-keepalive")
    _KEEPALIVE_THREAD.start()


def stop_keepalive():
    """Signal the keep-alive thread to stop."""
    _KEEPALIVE_STOP.set()


# ─────────────────────────── Notebook image display ───────────────────────────
def display_image_in_notebook(path: str, caption: str = "") -> bool:
    try:
        from IPython.display import display, Image as IPyImage, HTML
        display(HTML(f"<p style='font-weight:bold;margin:4px 0'>{caption}</p>"))
        display(IPyImage(filename=path, width=512))
        return True
    except Exception:
        return False


# ─────────────────────────── Template engine ──────────────────────────────────
def slugify(text: str) -> str:
    text = text.lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]", "", text)


def resolve_path(path: str, context: Dict) -> Any:
    parts = path.split(".")
    val = context
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p, f"<{path}>")
        elif isinstance(val, list) and p.isdigit():
            val = val[int(p)]
        else:
            return f"<{path}>"
    if isinstance(val, (dict, list)):
        return json.dumps(val, indent=2)
    return val


def resolve_template(template: str, context: Dict) -> str:
    def replace_match(m):
        expr = m.group(1).strip()
        if "|" in expr:
            parts = [p.strip() for p in expr.split("|")]
            value = resolve_path(parts[0], context)
            for filt in parts[1:]:
                if filt == "slugify":
                    value = slugify(str(value))
            return str(value)
        return str(resolve_path(expr, context))
    return re.sub(r"\{\{(.+?)\}\}", replace_match, template)


# ─────────────────────────── LLM Clients ──────────────────────────────────────
class LocalLLMClient:
    """
    Routes LLM calls to model_server.py POST /llm/chat.
    Uses the local uncensored Llama-3.2-3B-Instruct model.
    """
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def chat(self, prompt: str, temperature: float = 0.85, max_tokens: int = 4096) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.server_url}/llm/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Local LLM error: {data['error']}")
        return data["content"]

    def status(self) -> Dict:
        try:
            r = requests.get(f"{self.server_url}/llm/status", timeout=5)
            return r.json()
        except Exception:
            return {"loaded": False, "on_gpu": False}

    def offload(self) -> Dict:
        r = requests.post(f"{self.server_url}/llm/offload", json={}, timeout=120)
        r.raise_for_status()
        return r.json()

    def reload(self) -> Dict:
        r = requests.post(f"{self.server_url}/llm/reload", json={}, timeout=120)
        r.raise_for_status()
        return r.json()


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ─────────────────────────── Image Gen Client ─────────────────────────────────
class ImageGenClient:
    """Calls model_server.py image endpoints (txt2img / img2img).
    
    Includes:
      - Retry with exponential backoff (3 attempts)
      - Server health check before each attempt
      - 600s timeout (Z-Image on T4 can be slow)
      - Automatic wait-and-retry if server returns 503
    """

    TIMEOUT = 600          # 10 minutes — Z-Image Turbo on T4 can take 2-4 min per image
    MAX_RETRIES = 3
    RETRY_DELAYS = [10, 30, 60]  # seconds between retries

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        # Use a Session with connection pooling + keep-alive
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=0,          # we handle retries ourselves
            pool_connections=1,
            pool_maxsize=1,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _wait_for_server(self, max_wait: int = 120) -> bool:
        """Block until the image server responds to /health or /image/status."""
        for attempt in range(max_wait // 5):
            try:
                r = self._session.get(
                    f"{self.server_url}/image/status", timeout=5
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("loaded") or data.get("status") == "ready":
                        return True
            except Exception:
                pass
            # Also try /health (model_server.py)
            try:
                r = self._session.get(f"{self.server_url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("image", {}).get("loaded"):
                        return True
            except Exception:
                pass
            time.sleep(5)
        return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        cfg_scale: float = 1.0,
        seed: int = -1,
        sample_image_b64: Optional[str] = None,
        strength: float = 0.75,
    ) -> bytes:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }

        if sample_image_b64:
            payload["init_images"] = [f"data:image/png;base64,{sample_image_b64}"]
            payload["denoising_strength"] = strength
            endpoint = f"{self.server_url}/image/img2img"
        else:
            endpoint = f"{self.server_url}/image/txt2img"

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                # Health check before each attempt
                if attempt > 0:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    print(f"    ⏳ Retry {attempt + 1}/{self.MAX_RETRIES} in {delay}s...",
                          flush=True)
                    time.sleep(delay)

                    print(f"    🔍 Checking server health...", flush=True)
                    if not self._wait_for_server(max_wait=120):
                        print(f"    ⚠️  Server not responding — retrying anyway", flush=True)

                # Force garbage collection before GPU-heavy work
                gc.collect()

                resp = self._session.post(
                    endpoint, json=payload, timeout=self.TIMEOUT
                )

                if resp.status_code == 503:
                    # Server busy or model not loaded — wait and retry
                    last_error = RuntimeError(f"Server returned 503: {resp.text[:200]}")
                    print(f"    ⚠️  Server busy (503) — will retry", flush=True)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    last_error = RuntimeError(f"Image gen error: {data['error']}")
                    print(f"    ⚠️  Server error: {data['error'][:100]} — will retry",
                          flush=True)
                    continue

                img_b64 = data["images"][0] if "images" in data else data["image"]
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                return base64.b64decode(img_b64)

            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"    ⚠️  Timeout after {self.TIMEOUT}s — will retry", flush=True)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                print(f"    ⚠️  Connection lost — will retry", flush=True)
            except Exception as e:
                last_error = e
                print(f"    ⚠️  Error: {e} — will retry", flush=True)

        raise RuntimeError(
            f"Image generation failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# ─────────────────────────── PromptFlow Interpreter ───────────────────────────
class PromptFlowInterpreter:
    def __init__(
        self,
        promptflow_path: str,
        cli_inputs: Dict,
        openai_api_key: str = "",
        model_server_url: str = "http://localhost:7860",
        sample_image_path: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.promptflow_path = promptflow_path
        self.cli_inputs = cli_inputs
        self.openai_api_key = openai_api_key
        self.model_server_url = model_server_url.rstrip("/")
        self.sample_image_path = sample_image_path
        self.dry_run = dry_run

        self.config: Dict = {}
        self.inputs: Dict = {}
        self.steps_output: Dict = {}
        self.context: Dict = {}

        # Clients (set after config load)
        self._llm_provider: str = "local"
        self._local_llm: Optional[LocalLLMClient] = None
        self._openai_llm: Optional[OpenAIClient] = None
        self._image_client: Optional[ImageGenClient] = None

        # GPU lifecycle state
        self._llm_offloaded: bool = False

        # Reference image
        self.sample_image_b64: Optional[str] = None
        if sample_image_path and os.path.exists(sample_image_path):
            with open(sample_image_path, "rb") as f:
                self.sample_image_b64 = base64.b64encode(f.read()).decode()
            self._log(f"📎 Sample image loaded: {sample_image_path}")

    # ── Logging helpers ────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "info"):
        if HAS_RICH:
            colors = {"info": "white", "success": "green", "warn": "yellow", "error": "red"}
            console.print(f"[{colors.get(level, 'white')}]{msg}[/{colors.get(level, 'white')}]")
        else:
            print(msg)

    def _header(self, title: str, subtitle: str = "", style: str = "blue"):
        if HAS_RICH:
            body = f"[bold {style}]{title}[/bold {style}]"
            if subtitle:
                body += f"\n[dim]{subtitle}[/dim]"
            console.print(Panel.fit(body, border_style=style))
        else:
            print(f"\n{'═'*60}\n{title}")
            if subtitle:
                print(f"  {subtitle}")

    def _step_header(self, step: Dict, idx: int, total: int):
        style = {"llm_call": "blue", "code_loop": "yellow", "code": "magenta"}.get(
            step.get("type", ""), "white")
        name = step.get("name", step.get("id", ""))
        desc = step.get("description", "")
        if HAS_RICH:
            console.rule(f"[bold {style}]Step {idx}/{total}: {name}[/bold {style}]")
            if desc:
                console.print(f"[dim]{desc}[/dim]")
        else:
            print(f"\n{'─'*60}\nStep {idx}/{total} [{step.get('type','').upper()}]: {name}")
            if desc:
                print(f"  {desc}")

    # ── Load & validate config ─────────────────────────────────────────────────
    def load(self):
        with open(self.promptflow_path) as f:
            self.config = yaml.safe_load(f)
        meta = self.config.get("meta", {})
        self._header(
            title=meta.get("name", "PromptFlow"),
            subtitle=f"{meta.get('description','')}  v{meta.get('version','')}",
            style="green",
        )

    # ── Resolve inputs ─────────────────────────────────────────────────────────
    def resolve_inputs(self):
        input_defs = self.config.get("inputs", {})
        resolved = {}
        for key, defn in input_defs.items():
            if key in self.cli_inputs and self.cli_inputs[key] is not None:
                val = self.cli_inputs[key]
                if defn.get("type") == "integer":
                    val = int(val)
                resolved[key] = val
            elif "default" in defn:
                resolved[key] = defn["default"]
            elif defn.get("required"):
                raise ValueError(f"Required input '{key}' not provided.")
        self.inputs = resolved
        self.context["inputs"] = resolved
        out_dir = resolved.get("out_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        self.context["out_dir"] = out_dir

        if HAS_RICH:
            t = Table(title="⚙️  Inputs", show_header=True)
            t.add_column("Key", style="cyan")
            t.add_column("Value")
            for k, v in resolved.items():
                t.add_row(k, str(v)[:100])
            console.print(t)
        else:
            print("⚙️  Inputs:")
            for k, v in resolved.items():
                print(f"   {k}: {str(v)[:100]}")

    # ── Initialize clients ─────────────────────────────────────────────────────
    def initialize_clients(self):
        llm_cfg = self.config.get("llm", {})
        img_cfg = self.config.get("image_gen", {})

        self._llm_provider = llm_cfg.get("provider", "local")

        if self._llm_provider == "local":
            self._local_llm = LocalLLMClient(server_url=self.model_server_url)
            # Check LLM is available
            status = self._local_llm.status()
            if status.get("loaded"):
                on_gpu = status.get("on_gpu", False)
                model  = status.get("model", "unknown")
                self._log(f"✅ Local LLM ready: {model}  (GPU={on_gpu})", "success")
            else:
                self._log("⚠️  Local LLM not yet loaded — will wait or fallback", "warn")
        else:
            if not self.openai_api_key:
                raise ValueError("openai_api_key required when llm.provider='openai'")
            openai_model = llm_cfg.get("openai_model", "gpt-4.1")
            self._openai_llm = OpenAIClient(api_key=self.openai_api_key, model=openai_model)
            self._log(f"✅ OpenAI client: {openai_model}", "success")

        server_url = img_cfg.get("server_url", self.model_server_url)
        self._image_client = ImageGenClient(server_url=server_url)
        self._log(f"✅ Image gen client → {server_url}", "success")

    # ── GPU lifecycle: offload before image gen ────────────────────────────────
    def _maybe_offload_llm(self):
        """Called before a code_loop step if gpu_lifecycle.offload_before_image_gen=true."""
        llm_cfg = self.config.get("llm", {})
        lifecycle = llm_cfg.get("gpu_lifecycle", {})

        if not lifecycle.get("offload_before_image_gen", False):
            return
        if self._llm_provider != "local":
            return
        if self._llm_offloaded:
            return
        if self.dry_run:
            self._log("DRY RUN: would offload LLM to CPU", "warn")
            return

        self._log("⬇️  GPU lifecycle: offloading local LLM to CPU before image gen...", "warn")
        try:
            result = self._local_llm.offload()
            self._llm_offloaded = True
            free_mb = result.get("gpu_free_mb", "?")
            self._log(f"✅ LLM offloaded to CPU. GPU now free: {free_mb} MB", "success")
        except Exception as e:
            self._log(f"⚠️  LLM offload failed (continuing): {e}", "warn")

    def _maybe_reload_llm(self):
        """Called after a code_loop step if gpu_lifecycle.reload_after_image_gen=true."""
        llm_cfg = self.config.get("llm", {})
        lifecycle = llm_cfg.get("gpu_lifecycle", {})

        if not lifecycle.get("reload_after_image_gen", False):
            return
        if self._llm_provider != "local":
            return
        if not self._llm_offloaded:
            return
        if self.dry_run:
            self._log("DRY RUN: would reload LLM to GPU", "warn")
            return

        self._log("⬆️  GPU lifecycle: reloading local LLM to GPU...", "warn")
        try:
            result = self._local_llm.reload()
            self._llm_offloaded = False
            self._log(f"✅ LLM restored to GPU. Result: {result.get('status')}", "success")
        except Exception as e:
            self._log(f"⚠️  LLM reload failed: {e}", "warn")

    # ── LLM call helper ────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Single LLM call. For local provider, always returns raw text — never
        attempts JSON parsing. parse_json is only honoured for openai provider.
        """
        llm_cfg = self.config.get("llm", {})
        mt = max_tokens or llm_cfg.get("max_tokens", 512)

        if self._llm_provider == "local":
            self._wait_for_local_llm()
            return self._local_llm.chat(
                prompt=prompt,
                temperature=llm_cfg.get("temperature", 0.85),
                max_tokens=mt,
            )
        else:
            return self._openai_llm.chat(
                prompt=prompt,
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=mt,
            )

    def _wait_for_local_llm(self):
        for attempt in range(30):
            if self._local_llm.status().get("loaded"):
                return
            self._log(f"  Waiting for local LLM... ({attempt+1}/30)", "warn")
            time.sleep(5)
        raise RuntimeError("Local LLM not ready after 150s")

    # ── Execute llm_call step ──────────────────────────────────────────────────
    def execute_llm_call(self, step: Dict) -> Any:
        llm_cfg = self.config.get("llm", {})
        prompt  = resolve_template(step["prompt"], self.context)

        if HAS_RICH:
            console.print(f"[dim]{textwrap.shorten(prompt, 200, placeholder='...')}[/dim]")

        label = (f"local LLM ({llm_cfg.get('model','?')})"
                 if self._llm_provider == "local"
                 else f"OpenAI ({llm_cfg.get('openai_model','gpt-4.1')})")
        self._log(f"🤖 {label}...")

        if self.dry_run:
            self._log("DRY RUN: skipping", "warn")
            return ""

        t0   = time.time()
        text = self._call_llm(prompt, step.get("max_tokens"))
        self._log(f"✅ {time.time()-t0:.1f}s — {len(text)} chars", "success")

        # parse_json only supported for openai; local model gets raw text
        if step.get("parse_json") and self._llm_provider == "openai":
            try:
                clean = re.sub(r"```json\s*|```\s*", "", text).strip()
                result = json.loads(clean)
                self._log("✅ JSON parsed", "success")
                return result
            except json.JSONDecodeError as e:
                self._log(f"⚠️  JSON parse failed ({e}) — returning raw text", "warn")

        return text

    # ── Execute llm_call_loop step ─────────────────────────────────────────────
    def execute_llm_call_loop(self, step: Dict) -> Dict:
        llm_cfg = self.config.get("llm", {})
        expr    = step["iterate_over"].replace("{{", "").replace("}}", "").strip()
        items   = resolve_path(expr, self.context)
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                raise ValueError(f"Cannot resolve iterate_over: {expr}")

        output_key  = step.get("output_key", "descriptions")
        step_max_tokens = step.get("max_tokens", llm_cfg.get("max_tokens", 120))
        total       = len(items)
        loop_var    = step.get("loop_var", "item")
        results     = []

        self._log(f"🔁 LLM loop: {total} calls")

        if self._llm_provider == "local":
            self._wait_for_local_llm()

        for idx, item in enumerate(items, 1):
            loop_ctx = {
                **self.context,
                loop_var: item,
                "loop": {"index": idx, "total": total},
            }

            prompt = resolve_template(step["prompt"], loop_ctx)
            cat    = item.get("category", "") if isinstance(item, dict) else ""
            vid    = item.get("variant_index", idx) if isinstance(item, dict) else idx

            if HAS_RICH:
                console.print(f"  [dim][{idx}/{total}] {cat} variant {vid}[/dim]", end=" ")
            else:
                print(f"  [{idx}/{total}] {cat} v{vid}", end=" ", flush=True)

            if self.dry_run:
                results.append(f"<dry run — {cat} variant {vid}>")
                print()
                continue

            t0   = time.time()
            text = self._call_llm(prompt, step_max_tokens).strip()

            # Defensive cleanup: if the model rambled, take only the first sentence
            first_sentence = re.split(r"(?<=[.!?])\s", text)[0] if text else text

            results.append(first_sentence)
            elapsed = time.time() - t0

            if HAS_RICH:
                console.print(f"[green]→ {first_sentence[:70]}...[/green]  [dim]{elapsed:.1f}s[/dim]")
            else:
                print(f"→ {first_sentence[:60]}...  ({elapsed:.1f}s)")

        self._log(f"✅ LLM loop complete — {len(results)} descriptions", "success")
        return {output_key: results}

    # ── Execute code_loop step ─────────────────────────────────────────────────
    def execute_code_loop(self, step: Dict) -> List:
        img_cfg = self.config.get("image_gen", {})
        out_dir = self.inputs.get("out_dir", "output")

        # Resolve iterable
        expr = step["iterate_over"].replace("{{", "").replace("}}", "").strip()
        items = resolve_path(expr, self.context)
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                raise ValueError(f"Cannot resolve iterate_over: {expr}")

        total = len(items)

        # ── GPU lifecycle: offload LLM before generating images ────────────────
        self._maybe_offload_llm()

        self._log(f"🔁 Loop: {total} items to process")
        results = []

        for idx, item in enumerate(items, 1):
            loop_ctx = {
                **self.context,
                step["loop_var"]: item,
                "loop": {"index": idx, "total": total},
            }
            item_name = item.get("name", f"item_{idx}")
            item_cat  = item.get("category", "")

            if HAS_RICH:
                console.print(f"\n[bold yellow]  [{idx}/{total}] {item_cat} — {item_name}[/bold yellow]")
            else:
                print(f"\n  [{idx}/{total}] {item_name}")

            for action in step.get("actions", []):
                atype = action["type"]

                if atype == "image_gen":
                    prompt_text = resolve_template(action["prompt"], loop_ctx)
                    neg_prompt  = resolve_template(action.get("negative_prompt", ""), loop_ctx)
                    fname_raw   = resolve_template(action["filename"], loop_ctx)
                    fname       = re.sub(r"[^a-zA-Z0-9_\-.]", "_", fname_raw)
                    out_path    = os.path.join(out_dir, fname)

                    # ── Resume support: skip already-generated images ──
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                        self._log(f"  ⏭️  Already exists → {fname} (skipping)", "success")
                        display_image_in_notebook(out_path, caption=f"[{idx}/{total}] {item_name} (cached)")
                        results.append({**item, "output_path": out_path, "status": "cached"})
                        continue

                    self._log(f"  🎨 → {fname}")
                    if HAS_RICH:
                        console.print(f"  [dim]{textwrap.shorten(prompt_text, 110)}[/dim]")
                    else:
                        print(f"    {prompt_text[:110]}...")

                    if not self.dry_run:
                        # GC before each image to keep VRAM clean
                        gc.collect()
                        t_img = time.time()
                        try:
                            img_bytes = self._image_client.generate(
                                prompt=prompt_text,
                                negative_prompt=neg_prompt,
                                width=img_cfg.get("width", 1024),
                                height=img_cfg.get("height", 1024),
                                steps=img_cfg.get("steps", 4),
                                cfg_scale=img_cfg.get("cfg_scale", 1.0),
                                sample_image_b64=(
                                    self.sample_image_b64
                                    if img_cfg.get("sample_image") else None
                                ),
                            )
                            with open(out_path, "wb") as f:
                                f.write(img_bytes)
                            img_elapsed = time.time() - t_img
                            self._log(f"  ✅ Saved → {out_path}  ({img_elapsed:.1f}s)", "success")
                            display_image_in_notebook(out_path, caption=f"[{idx}/{total}] {item_name}")
                            results.append({**item, "output_path": out_path, "status": "success"})
                        except Exception as e:
                            self._log(f"  ✗ Failed: {e}", "error")
                            results.append({**item, "status": "error", "error": str(e)})
                    else:
                        self._log(f"  DRY RUN: would save {fname}", "warn")
                        results.append({**item, "output_path": out_path, "status": "dry_run"})

                elif atype == "log":
                    msg = resolve_template(action["message"], loop_ctx)
                    self._log(msg, "success")

                elif atype == "display_image":
                    img_path = resolve_template(action["path"], loop_ctx)
                    if os.path.exists(img_path):
                        display_image_in_notebook(img_path, item_name)

        # ── GPU lifecycle: reload LLM after generating images ──────────────────
        self._maybe_reload_llm()

        return results

    # ── Execute code step ──────────────────────────────────────────────────────
    def execute_code(self, step: Dict) -> Any:
        code = textwrap.dedent(step.get("code", ""))
        fn_code = "def _pf_fn(inputs, steps, os, json, Path):\n"
        for line in code.splitlines():
            fn_code += f"    {line}\n"

        globs = {"inputs": self.inputs, "steps": self.steps_output,
                 "os": os, "json": json, "Path": Path}
        try:
            exec(fn_code, globs)
            return globs["_pf_fn"](self.inputs, self.steps_output, os, json, Path)
        except Exception as e:
            self._log(f"Code step error: {e}\n{traceback.format_exc()}", "error")
            return {"error": str(e)}

    # ── Main run loop ──────────────────────────────────────────────────────────
    def run(self) -> Dict:
        self.load()
        self.resolve_inputs()
        self.initialize_clients()

        # ── Start background keep-alive for Colab ──────────────────────────────
        start_keepalive()
        self._log("🔄 Background keep-alive started (Colab anti-disconnect)", "info")

        steps = self.config.get("steps", [])
        total = len(steps)
        self._log(f"\n🚀 Starting PromptFlow — {total} steps\n")
        t0 = time.time()

        for idx, step in enumerate(steps, 1):
            step_id   = step["id"]
            step_type = step.get("type", "")
            self._step_header(step, idx, total)
            t_step = time.time()

            try:
                if step_type == "llm_call":
                    result     = self.execute_llm_call(step)
                    output_key = step.get("output_key", step_id)
                    stored     = {output_key: result}
                    self.steps_output[step_id] = stored
                    self.context.setdefault("steps", {})[step_id] = stored

                elif step_type == "llm_call_loop":
                    result = self.execute_llm_call_loop(step)
                    self.steps_output[step_id] = result
                    self.context.setdefault("steps", {})[step_id] = result

                elif step_type == "code_loop":
                    result = self.execute_code_loop(step)
                    self.steps_output[step_id] = result
                    self.context.setdefault("steps", {})[step_id] = result

                elif step_type == "code":
                    result = self.execute_code(step)
                    self.steps_output[step_id] = result
                    self.context.setdefault("steps", {})[step_id] = result

                else:
                    self._log(f"Unknown step type: {step_type}", "warn")

                elapsed = time.time() - t_step
                self._log(f"✅ '{step_id}' done in {elapsed:.1f}s", "success")

            except Exception as e:
                self._log(f"✗ Step '{step_id}' FAILED: {e}\n{traceback.format_exc()}", "error")
                raise

        total_elapsed = time.time() - t0

        # ── Stop background keep-alive ─────────────────────────────────────────
        stop_keepalive()

        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold green]✅ Complete![/bold green]\n"
                f"Time: [cyan]{total_elapsed:.1f}s[/cyan]  |  "
                f"Output: [cyan]{self.inputs.get('out_dir','output')}[/cyan]",
                border_style="green",
            ))
        else:
            print(f"\n✅ Done in {total_elapsed:.1f}s — output: {self.inputs.get('out_dir','output')}")

        return self.steps_output


# ─────────────────────────── CLI Entry Point ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PromptFlow Interpreter — run .promptflow config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Using local LLM (default, as specified in .promptflow):
          python interpreter.py game_terrain_generator.promptflow \\
            --game_context 'Desert canyon sci-fi RTS' \\
            --terrain_assets_to_make 10 \\
            --model_server_url http://localhost:7860 \\
            --out_dir output/terrain

          # Override to use OpenAI instead of local LLM:
          python interpreter.py game_terrain_generator.promptflow \\
            --game_context 'Desert canyon sci-fi RTS' \\
            --terrain_assets_to_make 10 \\
            --openai_api_key sk-... \\
            --out_dir output/terrain
        """),
    )
    parser.add_argument("promptflow", help="Path to .promptflow file")
    parser.add_argument("--game_context", default="")
    parser.add_argument("--terrain_asset_instructions", default="")
    parser.add_argument("--terrain_assets_to_make", type=int, default=25)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--openai_api_key", default=None,
                        help="OpenAI key (only needed when llm.provider=openai)")
    parser.add_argument("--model_server_url", default="http://localhost:7860",
                        help="URL of model_server.py")
    parser.add_argument("--sample_image", default=None,
                        help="Reference image for img2img style consistency")
    parser.add_argument("--dry_run", action="store_true",
                        help="Plan only — skip all API calls")

    args = parser.parse_args()

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    interpreter = PromptFlowInterpreter(
        promptflow_path=args.promptflow,
        cli_inputs={
            "game_context":               args.game_context,
            "terrain_asset_instructions": args.terrain_asset_instructions,
            "terrain_assets_to_make":     args.terrain_assets_to_make,
            "out_dir":                    args.out_dir,
        },
        openai_api_key=api_key,
        model_server_url=args.model_server_url,
        sample_image_path=args.sample_image,
        dry_run=args.dry_run,
    )
    interpreter.run()


if __name__ == "__main__":
    main()
