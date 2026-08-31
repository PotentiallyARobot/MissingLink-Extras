# ============================================================================
#  MissingLink MiniMax Studio — one cell, pure Python.
#
#  ComfyUI is imported as a LIBRARY. No server, no graph JSON, no object_info,
#  no frontend. Node classes are called as ordinary Python functions:
#
#    UNETLoader -> MiniMaxH3SigmaShift -> MiniMaxH3ImageToVideo -> BasicGuider
#    -> SamplerCustomAdvanced -> VAEDecode + VAEDecodeAudio -> CreateVideo
#
#  Weights: Comfy-Org pruned int8 convrot (~44 GB).
#  Runtime: A100 40GB + High-RAM. A100 attention autotune: SageAttention 2.2 vs SDPA.
#
#  LICENSE: MiniMax H3 Community License, territory clause.
#           https://platform.minimax.io/h3-license
# ============================================================================

COMFY_DIR = "/content/ComfyUI"
UI_PORT   = 7860

# Prefer the real REDMIX Beta2 checkpoint. If publisher access blocks its bytes,
# explicitly notify the user and fall back to the verified public Comfy-Org pruned
# FL2VA transformer + NaughtyTimes v2 + LightX2V rather than crashing or pretending
# REDMIX loaded.
DIT_CHOICE = "redmix_beta2"
# RedCraft REDMIX H3 A2A Beta2, CivitAI version 3262321.
# This merged H3 checkpoint already incorporates NaughtyTimes-derived NSFW tuning
# and LightX2V/Turbo work, so those LoRAs are NOT stacked by default.
REDMIX_VERSION = 3262321
REDMIX_MODEL_ID = 958009
REDMIX_FILE = "REDMix-MiniMaxH3-A2Ab2-pruned-int8-convrot-ComfyMCP.safetensors"
REDMIX_FILE_ALIASES = [
    REDMIX_FILE,
    "redcraftREDMIXHybridA2A_h3A2AREDBeta2.safetensors",
]
FALLBACK_DIT_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REQUIRE_REDMIX = False
USING_REDMIX = True
REDMIX_SHA256 = "6a1e09871380982a96c0af058af35cd61b34e4a47a567b4704bdaa7d0f5fd60f"
REDMIX_PAGE_URL = "https://civitai.red/models/958009/redcraft-or-or-hybrid-h3-a2a-beta2-ltx25-2k"
REDMIX_DIRECT_URL_SECRET = "REDMIX_DIRECT_URL"  # optional: copy the authorized browser Download link into a Colab Secret
REDMIX_LOCAL_PATH_SECRET = "REDMIX_LOCAL_PATH"  # optional: exact checkpoint path already mounted in Colab/Drive
CIVITAI_COOKIE_SECRET = "CIVITAI_COOKIE"        # optional: full Cookie header for an account that has model access
DITS = {
  "redmix_beta2": (None, None, REDMIX_FILE),
}

# SexGod1979 NaughtyTimes v2.0 PRUNED, CivitAI version 3212436.
# The current release notes recommend v2 at strength 1.0. There is no special
# activation token; use ordinary descriptive H3 prompts. The author currently
# recommends I2V over T2V when judging whether the LoRA is having an effect.
NSFW_LORA_VERSION = 3212436
NSFW_LORA_FILE = "NaughtyTimes_pruned_r256_v2.safetensors"
NSFW_LORA_SHA256 = "947efec5a357505bb93bdc1b050d33786ec150aa1c85f24337f0d59f39aaf31a"
NSFW_LORA_STRENGTH = 1.0

# Optional LightX2V Lightning / Turbo LoRA. This is the current FL2V 4-step v1.1
# high-fidelity dynamic-rank ComfyUI resize. It can be stacked after NaughtyTimes
# for fast previews, but is OFF by default so the NSFW LoRA can be evaluated
# without an accelerator changing its behavior.
LIGHTNING_REPO = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
LIGHTNING_FILE = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_64_bf16.safetensors"
LIGHTNING_STRENGTH = 1.0
LIGHTNING_DEFAULT = False
LIGHTNING_STEPS = 4
LIGHTNING_SHIFT_VIDEO = 6.0
LIGHTNING_SHIFT_AUDIO = 3.0

# Optional action LoRA from CivitAI model 2011853. The URL supplied by the user
# points at version 2277119, whose page labels it as WAN 2.2. Do NOT apply that
# file to H3. At startup we query the model API and choose the newest version
# whose baseModel/name identifies MiniMax H3, then expose its trainedWords in UI.
ACTION_MODEL_ID = 2011853
ACTION_LINKED_VERSION = 2277119
ACTION_STRENGTH = 1.0
ACTION_DEFAULT = False
ACTION_META = {}
ACTION_PATH = None
ACTION_FILE = None
ACTION_AVAILABLE = False

# H3 video is modeled at 24 fps and the latent/VAE length must be 17*n+5.
# The node accepts up to 3600 frames; the largest legal grid value <=3600 is 3592.
MODEL_FPS = 24.0
MIN_FRAMES = 5
MAX_FRAMES = 3592

# cloudflared is off by default — Cloudflare's quick-tunnel API has been
# refusing registrations. Colab's own iframe/window transport is used instead.
TUNNEL_FALLBACK = False

# Memory. In library mode Comfy's manager is unconfigured and tries to keep the
# whole 33B DiT resident, which OOMs regardless of card size. reserve_vram is
# headroom left free for activations and the cast buffers; raise it if sampling
# OOMs. LOWVRAM forces block-by-block streaming — slower, but survives anything.
RESERVE_VRAM = 3.0
LOWVRAM      = False

# A100 attention/runtime optimization. We benchmark SageAttention 2.2 against
# PyTorch SDPA on the actual GPU before ComfyUI imports its attention aliases.
# The faster verified backend wins; failure falls back safely to SDPA.
AUTO_OPTIMIZE_ATTENTION = True
SAGEATTN_VERSION = "2.2.0"
ATTN_BENCH_SEQ = 4096
ATTN_BENCH_HEADS = 24
ATTN_BENCH_DIM = 128

# Smaller official H3 Qwen3-VL-32B conditioning encoder. Comfy-Org documents
# this NVFP4/AWQ file as H3-compatible and usable on non-Blackwell GPUs.
TEXT_ENCODER_FILE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TEXT_ENCODER_GIB  = 14.61   # binary GiB, published Comfy-Org size
REDMIX_GIB        = 19.53   # ~20.97 GB decimal on disk
VIDEO_VAE_GIB     = 4.85
AUDIO_VAE_GIB     = 0.57
# ───────────────────────────────────────────────────────────────────────────

import os, re, sys, gc, time, json, uuid, stat, shutil, socket, asyncio
import threading, subprocess, traceback, urllib.request

# Must be set BEFORE torch initialises CUDA. The DiT/VAE handoff fragments the
# allocator badly; expandable segments let freed blocks be reused across sizes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try: sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception: pass

# Capture the real Python stdout/stderr stream while still forwarding it to the
# Colab cell. The browser UI reads this bounded ring buffer, so ComfyUI/model
# loader/sampler messages remain visible during generation instead of being hidden.
from collections import deque
_CONSOLE_LOCK = threading.Lock()
_CONSOLE_LINES = deque(maxlen=6000)
_CONSOLE_SEQ = 0

def _console_push(text, stream="stdout"):
    global _CONSOLE_SEQ
    text = str(text).replace("\r", "\n")
    for line in text.splitlines():
        if not line:
            continue
        with _CONSOLE_LOCK:
            _CONSOLE_SEQ += 1
            _CONSOLE_LINES.append((_CONSOLE_SEQ, stream, line))

class _ConsoleTee:
    def __init__(self, original, stream):
        self.original = original
        self.stream = stream
        self.pending = ""
    def write(self, data):
        data = str(data)
        try:
            self.original.write(data)
            self.original.flush()
        except Exception:
            pass
        self.pending += data.replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line:
                _console_push(line, self.stream)
        return len(data)
    def flush(self):
        try: self.original.flush()
        except Exception: pass
    def isatty(self):
        try: return self.original.isatty()
        except Exception: return False
    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")
    def fileno(self):
        return self.original.fileno()

if not isinstance(sys.stdout, _ConsoleTee):
    sys.stdout = _ConsoleTee(sys.stdout, "stdout")
if not isinstance(sys.stderr, _ConsoleTee):
    sys.stderr = _ConsoleTee(sys.stderr, "stderr")

def log(m): print(m, flush=True)

log("="*74)
log("  MissingLink · MiniMax Studio   (MiniMax-H3, library mode)")
log("="*74)

# ── License gate ───────────────────────────────────────────────────────────
# A MissingLink token is required. Set it in Colab secrets as
# MISSING_LINK_TOKEN, or export it before running.
#   https://www.missinglink.build/pricing.html
MACHINE = os.environ.get("MACHINE", "a100")
ML_TOKEN = os.environ.get("MISSING_LINK_TOKEN", "").strip()

if not ML_TOKEN:
    try:
        from google.colab import userdata
        ML_TOKEN = (userdata.get("MISSING_LINK_TOKEN") or "").strip()
        os.environ["MISSING_LINK_TOKEN"] = ML_TOKEN
    except Exception:
        pass

BUY_URL = ("https://missinglink.build/buy/all_gpu_bundle"
           "?price=price_1T3lGSJTLzTiUThCUah5Sm7o")

if not ML_TOKEN:
    raise RuntimeError(
        "\n  No MISSING_LINK_TOKEN found.\n"
        "  Add it to Colab secrets (key icon, left sidebar) as MISSING_LINK_TOKEN,\n"
        "  or set os.environ['MISSING_LINK_TOKEN'] before running this cell.\n"
        f"  Get a token: {BUY_URL}\n")

def _validate_token(tok):
    import urllib.request, base64
    req = urllib.request.Request(f"https://missinglink.build/{MACHINE}.txt")
    req.add_header("Authorization", "Basic " +
                   base64.b64encode(f"{tok}:".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as e:
        code = getattr(e, "code", None)
        if code in (401, 403):
            return False
        log(f"  ⚠ could not reach missinglink.build ({e}) — continuing offline")
        return True          # network trouble shouldn't lock you out mid-session

if not _validate_token(ML_TOKEN):
    raise RuntimeError(
        "\n  MISSING_LINK_TOKEN was rejected (401/403).\n"
        "  Check for typos or an expired bundle.\n"
        f"  {BUY_URL}\n")
log("✓ MissingLink token accepted")
ML_OK = True

# ── ComfyUI source bootstrap ────────────────────────────────────────────────
# A fresh Colab can have /content/ComfyUI/models populated by this studio while
# the actual ComfyUI Python source is absent. Detect that case before importing
# torch/comfy. If needed, merge the official ComfyUI source into COMFY_DIR while
# preserving the huge models/ directory that may already contain downloaded H3
# weights. PyTorch itself is deliberately excluded from the requirements install
# so Colab's working CUDA build is not replaced.
def _is_comfy_source(root):
    return (os.path.isfile(os.path.join(root, "nodes.py")) and
            os.path.isdir(os.path.join(root, "comfy")) and
            os.path.isfile(os.path.join(root, "comfy_extras", "nodes_minimax_h3.py")))

def _ensure_comfyui_source():
    global COMFY_DIR

    candidates = []
    for c in (
        os.environ.get("COMFY_DIR", "").strip(),
        COMFY_DIR,
        "/content/MissingLink-Extras/ComfyUI",
        os.path.abspath(os.path.join(os.getcwd(), "..", "ComfyUI")),
    ):
        if c and c not in candidates:
            candidates.append(c)

    for c in candidates:
        if _is_comfy_source(c):
            COMFY_DIR = c
            log(f"✓ ComfyUI source: {COMFY_DIR}")
            return

    target = COMFY_DIR
    os.makedirs(target, exist_ok=True)
    tmp = "/content/_comfyui_source_bootstrap"
    shutil.rmtree(tmp, ignore_errors=True)
    log("  ↓ ComfyUI source missing — bootstrapping official Comfy-Org/ComfyUI")
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", tmp],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError("Could not clone ComfyUI source:\n" + r.stdout[-2000:])

    # Merge source into target but never overwrite/delete models/. This works
    # even when the downloader has already created /content/ComfyUI/models.
    for item in os.listdir(tmp):
        if item in (".git", "models"):
            continue
        s = os.path.join(tmp, item)
        d = os.path.join(target, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    shutil.rmtree(tmp, ignore_errors=True)

    if not _is_comfy_source(target):
        raise RuntimeError(
            f"ComfyUI bootstrap finished but required H3 source files are missing in {target}")

    # Install ComfyUI dependencies except torch/torchvision/torchaudio; replacing
    # Colab's CUDA PyTorch build here can break the A100 runtime.
    req = os.path.join(target, "requirements.txt")
    if os.path.isfile(req):
        filtered = "/tmp/comfy_requirements_no_torch.txt"
        keep = []
        with open(req, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().lower()
                pkg = re.split(r"[<>=!~\[ ;]", s, 1)[0]
                if pkg in {"torch", "torchvision", "torchaudio"}:
                    continue
                keep.append(line)
        with open(filtered, "w", encoding="utf-8") as f:
            f.writelines(keep)
        log("  ↓ installing ComfyUI Python dependencies (keeping existing CUDA PyTorch)")
        rr = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", filtered],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if rr.returncode != 0:
            raise RuntimeError("ComfyUI dependency install failed:\n" + rr.stdout[-2500:])

    log(f"✓ ComfyUI source bootstrapped: {target}")

_ensure_comfyui_source()

subprocess.run(["pkill","-9","-f","ComfyUI/main.py"], check=False)
subprocess.run(["pkill","-9","-f","cloudflared"], check=False)

# ComfyUI-Manager breaks library-mode import: it reaches for
# PromptServer.instance, which only exists when the server is running.
for cand in ("ComfyUI-Manager", "ComfyUI-Manager.off"):
    p = os.path.join(COMFY_DIR, "custom_nodes", cand)
    if os.path.isdir(p):
        shutil.move(p, "/content/_mgr_disabled_" + uuid.uuid4().hex[:4])

for pkg in ("flask","nest_asyncio","av","huggingface_hub"):
    subprocess.run([sys.executable,"-m","pip","install","-q",pkg], check=False)

import torch, numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, Response, send_file

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device. Runtime -> Change runtime type -> GPU.")
gpu  = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
disk = shutil.disk_usage("/content").free / 1e9
log(f"  {gpu} · {vram:.0f} GB VRAM · {disk:.0f} GB disk free")


# ── A100 runtime + attention autotune ────────────────────────────────────────
# These settings are lossless runtime optimizations. They do not change H3
# steps, scheduler, LoRA strengths, or model weights.
torch.set_grad_enabled(False)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass
try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

ATTN_BACKEND = "pytorch-sdpa"
ATTN_BENCH = {}

def _timed_cuda(fn, warmup=2, iters=4):
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn()
    end.record(); torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)

def _ensure_sageattention():
    """Install + validate an official SageAttention build for A100/SM80 without replacing torch."""
    import importlib, importlib.util

    def _probe():
        try:
            importlib.invalidate_caches()
            import sageattention  # noqa: F401
            from sageattention import sageattn  # noqa: F401
            sm80 = importlib.util.find_spec("sageattention._qattn_sm80")
            if sm80 is None:
                return False, "imported, but sageattention._qattn_sm80 is missing"
            return True, f"SM80 extension: {getattr(sm80, 'origin', 'found')}"
        except Exception as e:
            return False, repr(e)

    ok, note = _probe()
    if ok:
        log(f"  ✓ SageAttention {SAGEATTN_VERSION} present | {note}")
        return True, "already installed"

    log(f"  ↓ SageAttention {SAGEATTN_VERSION} missing/broken — building official SM80 extension")
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = "unavailable"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    log(f"    build env: Python {sys.version.split()[0]} | torch {torch.__version__} | torch CUDA {torch.version.cuda} | Triton {triton_ver}")
    log(f"    build env: nvcc={nvcc if os.path.exists(nvcc) else 'NOT FOUND'} | TORCH_CUDA_ARCH_LIST=8.0")

    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.0"
    env["MAX_JOBS"] = str(min(16, os.cpu_count() or 8))
    env["EXT_PARALLEL"] = "4"
    env["NVCC_APPEND_FLAGS"] = "--threads 8"
    env["CUDA_HOME"] = env.get("CUDA_HOME") or "/usr/local/cuda"

    # Build only dependencies; deliberately do not upgrade/replace torch.
    dep = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ninja", "packaging"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if dep.returncode != 0:
        log("  ⚠ Sage build dependency install failed:\n" + dep.stdout[-4000:])

    # First use the official PyPI source package. Keep the log so failures are visible.
    cmd = [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", "--no-cache-dir",
           f"sageattention=={SAGEATTN_VERSION}"]
    rr = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok, note = _probe()
    if rr.returncode == 0 and ok:
        log(f"  ✓ SageAttention official PyPI SM80 build installed | {note}")
        return True, "PyPI SM80 build"

    log("  ⚠ SageAttention PyPI build did not produce a usable SM80 kernel")
    if rr.stdout:
        log("    compiler tail:\n" + rr.stdout[-5000:])

    # Official source fallback only; no third-party wheels/mirrors.
    src_dir = "/content/SageAttention-sm80"
    try:
        shutil.rmtree(src_dir, ignore_errors=True)
        clones = [
            ["git", "clone", "--depth", "1", "--branch", f"v{SAGEATTN_VERSION}", "https://github.com/thu-ml/SageAttention.git", src_dir],
            ["git", "clone", "--depth", "1", "--branch", SAGEATTN_VERSION, "https://github.com/thu-ml/SageAttention.git", src_dir],
            ["git", "clone", "--depth", "1", "https://github.com/thu-ml/SageAttention.git", src_dir],
        ]
        clone_out = ""
        cloned = False
        for cc in clones:
            shutil.rmtree(src_dir, ignore_errors=True)
            cr = subprocess.run(cc, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            clone_out = cr.stdout
            if cr.returncode == 0:
                cloned = True
                break
        if not cloned:
            log("  ⚠ SageAttention official GitHub clone failed:\n" + clone_out[-3000:])
            return False, "official source clone failed"

        sr = subprocess.run([sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", "--no-cache-dir", "."],
                            env=env, cwd=src_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok, note = _probe()
        if sr.returncode == 0 and ok:
            log(f"  ✓ SageAttention official source SM80 build installed | {note}")
            return True, "official source SM80 build"
        log("  ⚠ SageAttention official source build failed")
        if sr.stdout:
            log("    compiler tail:\n" + sr.stdout[-6000:])
        return False, note
    except Exception as e:
        log(f"  ⚠ SageAttention source-build exception: {e!r}")
        return False, repr(e)

def _autotune_attention():
    global ATTN_BACKEND, ATTN_BENCH
    cap = torch.cuda.get_device_capability(0)
    log(f"  attention autotune -> GPU capability sm_{cap[0]}{cap[1]}")
    # H3 uses head_dim=128. 4096 tokens is long enough to expose kernel behavior
    # without burning startup time or allocating the huge production sequence.
    B,H,S,D = 1, ATTN_BENCH_HEADS, ATTN_BENCH_SEQ, ATTN_BENCH_DIM
    q = torch.randn((B,H,S,D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
    try:
        sdpa_ms = _timed_cuda(sdpa)
    except Exception as e:
        sdpa_ms = float("inf")
        log(f"  ⚠ SDPA benchmark failed: {e}")

    sage_ms = float("inf")
    sage_ok = False
    install_note = "disabled"
    if AUTO_OPTIMIZE_ATTENTION:
        sage_ok, install_note = _ensure_sageattention()
    if sage_ok:
        try:
            from sageattention import sageattn
            sage = lambda: sageattn(q,k,v,tensor_layout="HND",is_causal=False)
            # First call is also a real kernel-image/self-test for SM80.
            sage_ms = _timed_cuda(sage)
        except Exception as e:
            sage_ok = False
            log(f"  ⚠ SageAttention self-test failed on A100: {e}")

    ATTN_BENCH = {"sdpa_ms": sdpa_ms, "sage_ms": sage_ms,
                  "sage_ok": bool(sage_ok), "sage_install": install_note}
    # Require a real margin so benchmark noise never selects a slower backend.
    if sage_ok and sage_ms < sdpa_ms * 0.95:
        ATTN_BACKEND = "sageattention"
        speedup = sdpa_ms / sage_ms
        log(f"  ✓ attention backend: SageAttention {SAGEATTN_VERSION} | {sage_ms:.2f} ms vs SDPA {sdpa_ms:.2f} ms | {speedup:.2f}× kernel win")
    else:
        ATTN_BACKEND = "pytorch-sdpa"
        if sage_ok:
            log(f"  ✓ attention backend: PyTorch SDPA | {sdpa_ms:.2f} ms vs Sage {sage_ms:.2f} ms (SDPA faster on this runtime)")
        else:
            log(f"  ✓ attention backend: PyTorch SDPA | {sdpa_ms:.2f} ms | Sage unavailable")
    del q,k,v
    gc.collect(); torch.cuda.empty_cache()

_autotune_attention()

# ── 1. Weights ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
if DIT_CHOICE not in DITS:
    raise RuntimeError(f"DIT_CHOICE must be one of {list(DITS)}")
_repo, _sub, DIT_FILE = DITS[DIT_CHOICE]

if "nvfp4" in DIT_CHOICE and not any(k in gpu for k in ("B200","RTX 50","GB200","RTX 60")):
    log(f"  ⚠ {DIT_CHOICE} is a Blackwell profile and {gpu} is not Blackwell.\n"
        f"    Expect a slow emulated path or a load failure — use eros_int8 instead.")

MODELS = os.path.join(COMFY_DIR, "models")

# REDMIX Beta2 is currently distributed from CivitAI. Download the exact checkpoint
# with the user's CIVITAI_API_KEY and verify its published SHA256 before Comfy sees it.
def _early_civitai_token():
    tok = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    if not tok:
        from google.colab import userdata
        tok = (userdata.get("CIVITAI_API_KEY") or "").strip()
    if not tok:
        raise RuntimeError("CIVITAI_API_KEY is required to download REDMIX H3 Beta2")
    os.environ["CIVITAI_API_KEY"] = tok
    return tok

def _download_redmix_beta2():
    """Robust authenticated CivitAI downloader with metadata + URL fallbacks.

    CivitAI occasionally changes/rotates the concrete download URL returned by
    the API, and some protected/adult files accept Bearer auth where a query
    token alone can return HTTP 401/403.  Resolve the exact file by SHA/name,
    then try both auth styles and the canonical model-version endpoint.  The
    final SHA256 check is authoritative, so no fallback can silently install a
    different checkpoint.
    """
    import hashlib
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    from urllib.error import HTTPError, URLError

    global USING_REDMIX, DIT_FILE, LIGHTNING_DEFAULT

    def _optional_secret(name):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
        try:
            from google.colab import userdata
            value = (userdata.get(name) or "").strip()
        except Exception:
            value = ""
        if value:
            os.environ[name] = value
        return value

    ddir = os.path.join(MODELS, "diffusion_models")
    os.makedirs(ddir, exist_ok=True)
    dest = os.path.join(ddir, REDMIX_FILE)
    part = dest + ".part"

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(b)
        return h.hexdigest().lower()

    # Accept either published filename for the exact same hash.  If the user
    # already has the checkpoint in the Comfy folder, no network access is needed.
    for alias in REDMIX_FILE_ALIASES:
        ap = os.path.join(ddir, alias)
        if os.path.isfile(ap):
            try:
                if sha(ap) == REDMIX_SHA256:
                    if ap != dest:
                        try:
                            os.replace(ap, dest)
                        except Exception:
                            shutil.copy2(ap, dest)
                    log(f"  ✓ {REDMIX_FILE} (SHA256 verified; local checkpoint)")
                    USING_REDMIX = True
                    DIT_FILE = REDMIX_FILE
                    return True
            except Exception:
                pass
    if os.path.exists(dest):
        os.remove(dest)

    # If the user already downloaded the paid/access-gated checkpoint to Drive
    # or another mounted path, use it directly and verify the published hash.
    local_path = _optional_secret(REDMIX_LOCAL_PATH_SECRET)
    if local_path:
        local_path = os.path.expanduser(local_path)
        if not os.path.isfile(local_path):
            raise RuntimeError(f"{REDMIX_LOCAL_PATH_SECRET} points to a missing file: {local_path}")
        log(f"  ↳ REDMIX local path supplied via {REDMIX_LOCAL_PATH_SECRET}; verifying SHA256")
        got = sha(local_path)
        if got != REDMIX_SHA256:
            raise RuntimeError(
                f"REDMIX local-path SHA256 mismatch: expected {REDMIX_SHA256}, got {got}. "
                "Refusing to load a different checkpoint."
            )
        try:
            os.link(local_path, dest)
        except Exception:
            shutil.copy2(local_path, dest)
        log(f"  ✓ {REDMIX_FILE} (SHA256 verified; REDMIX_LOCAL_PATH)")
        USING_REDMIX = True
        DIT_FILE = REDMIX_FILE
        return True

    tok = _early_civitai_token()
    direct_url = _optional_secret(REDMIX_DIRECT_URL_SECRET)
    civitai_cookie = _optional_secret(CIVITAI_COOKIE_SECRET)
    common_headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 MissingLink-H3-REDMIX/2.0",
    }

    def api_json(url):
        req = urllib.request.Request(url, headers=common_headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except HTTPError as e:
            body = ""
            try:
                body = e.read(600).decode("utf-8", "replace")
            except Exception:
                pass
            raise RuntimeError(f"CivitAI metadata HTTP {e.code}: {url}\n{body}") from e
        except URLError as e:
            raise RuntimeError(f"CivitAI metadata request failed: {url}: {e}") from e

    # First ask for the known release.  If CivitAI has re-parented/reissued the
    # file, search the model's current versions and identify it by SHA/name.
    metas = []
    try:
        metas.append(api_json(f"https://civitai.com/api/v1/model-versions/{REDMIX_VERSION}"))
    except Exception as e:
        log(f"  ⚠ version metadata lookup failed: {e}")

    try:
        model_meta = api_json(f"https://civitai.com/api/v1/models/{REDMIX_MODEL_ID}")
        metas.extend(model_meta.get("modelVersions") or [])
    except Exception as e:
        log(f"  ⚠ model metadata fallback failed: {e}")

    def file_sha(x):
        hashes = x.get("hashes") or {}
        return str(hashes.get("SHA256") or hashes.get("sha256") or "").lower()

    chosen = None
    for meta in metas:
        for x in (meta.get("files") or []):
            if file_sha(x) == REDMIX_SHA256:
                chosen = x
                break
        if chosen:
            break
    if chosen is None:
        for meta in metas:
            for x in (meta.get("files") or []):
                name = (x.get("name") or "").lower()
                if name == REDMIX_FILE.lower():
                    chosen = x
                    break
            if chosen:
                break
    if chosen is None:
        for meta in metas:
            for x in (meta.get("files") or []):
                name = (x.get("name") or "").lower()
                if name.endswith(".safetensors") and "beta2" in name:
                    chosen = x
                    break
            if chosen:
                break

    if not chosen:
        raise RuntimeError(
            f"Could not locate REDMIX Beta2 in CivitAI model {REDMIX_MODEL_ID} / "
            f"version {REDMIX_VERSION}. The release may have been removed or made unavailable."
        )

    found_name = chosen.get("name") or REDMIX_FILE
    found_id = chosen.get("id")
    log(f"  ↳ CivitAI resolved file: {found_name}" + (f" (file id {found_id})" if found_id else ""))

    # Build several equivalent URLs. A browser-authorized direct URL is first
    # because publisher-gated CivitAI files may be downloadable in the signed-in
    # browser while the generic API token endpoint still returns the HTML app.
    raw_urls = []
    if direct_url:
        raw_urls.append(direct_url)
        log(f"  ↳ using authorized {REDMIX_DIRECT_URL_SECRET} before API fallbacks")
    if chosen.get("downloadUrl"):
        raw_urls.append(chosen["downloadUrl"])
    # Try both the canonical API and the civitai.red front-end domain supplied
    # by the user.  Hash verification prevents a front-end HTML response from
    # ever being mistaken for the checkpoint.
    raw_urls.append(f"https://civitai.com/api/download/models/{REDMIX_VERSION}")
    raw_urls.append(f"https://civitai.red/api/download/models/{REDMIX_VERSION}")

    # Deduplicate while preserving order.
    seen = set(); raw_urls = [u for u in raw_urls if not (u in seen or seen.add(u))]

    def with_token(u):
        sp = urlsplit(u)
        q = dict(parse_qsl(sp.query, keep_blank_values=True))
        q["token"] = tok
        return urlunsplit((sp.scheme, sp.netloc, sp.path, urlencode(q), sp.fragment))

    # Each URL is attempted first with Authorization: Bearer and then with the
    # legacy ?token= form.  Never print the token or a tokenized URL.
    attempts = []
    for u in raw_urls:
        # Signed browser URLs commonly need no extra auth and may break if a
        # token query parameter is appended, so try them verbatim first.
        if direct_url and u == direct_url:
            attempts.append((u, False, False, "authorized direct URL"))
            if civitai_cookie:
                attempts.append((u, False, True, "authorized direct URL + browser cookie"))
        attempts.append((u, True, False, "Bearer auth"))
        attempts.append((with_token(u), False, False, "query-token auth"))
        if civitai_cookie:
            attempts.append((u, False, True, "browser cookie auth"))

    log(f"  ↓ REDMIX H3 Beta2: {REDMIX_FILE} (~20.97 GB)")
    errors = []
    downloaded = False

    for idx, (url, bearer, use_cookie, label) in enumerate(attempts, 1):
        # A partial file from a prior 401/403/HTML response must never be
        # resumed.  Resume only a plausible multi-megabyte binary partial.
        resume = os.path.isfile(part) and os.path.getsize(part) > 8 * 1024 * 1024
        cmd = [
            "curl", "-L", "--fail-with-body", "--show-error", "--progress-bar",
            "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
            "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
            "-A", "Mozilla/5.0 MissingLink-H3-REDMIX/2.0",
        ]
        if bearer:
            cmd += ["-H", f"Authorization: Bearer {tok}"]
        if use_cookie and civitai_cookie:
            cmd += ["-H", f"Cookie: {civitai_cookie}"]
        if resume:
            cmd += ["--continue-at", "-"]
        else:
            try:
                os.remove(part)
            except FileNotFoundError:
                pass
        cmd += ["-o", part, url]

        log(f"    attempt {idx}/{len(attempts)}: {label}" + (" + resume" if resume else ""))
        rr = subprocess.run(cmd)
        if rr.returncode == 0 and os.path.isfile(part):
            # Catch JSON/HTML access-denied pages that somehow arrived as HTTP 200.
            size = os.path.getsize(part)
            if size < 1024 * 1024:
                try:
                    head = open(part, "rb").read(512).lower()
                except Exception:
                    head = b""
                errors.append(f"{label}: suspiciously small response ({size} bytes): {head[:120]!r}")
                try: os.remove(part)
                except Exception: pass
                continue
            downloaded = True
            break

        errors.append(f"{label}: curl exit {rr.returncode}")
        # Exit 22 is HTTP >=400.  Its body may now be in .part; discard it so a
        # subsequent authenticated attempt does not send an invalid Range.
        if rr.returncode == 22:
            try:
                if os.path.isfile(part) and os.path.getsize(part) < 8 * 1024 * 1024:
                    os.remove(part)
            except Exception:
                pass

    if not downloaded:
        # CivitAI can return the public website shell (HTTP 200 HTML) for a
        # publisher-gated file even when API metadata is readable.  If that
        # happens, give the user one in-run chance to paste the *actual*
        # authorized Download link from their signed-in browser.  This does not
        # bypass access controls; it simply uses the entitlement-bearing URL
        # CivitAI gave the user and still verifies the exact published SHA256.
        try:
            if os.path.exists(part): os.remove(part)
        except Exception:
            pass

        log("  ⚠ CivitAI returned website HTML instead of REDMIX checkpoint bytes.")
        log("  ↳ REDMIX is not available to this Colab session; no interactive download-link prompt will be shown.")
        pasted_url = ""

        if pasted_url:
            ps = urlsplit(pasted_url)
            if ps.scheme not in ("http", "https") or not ps.netloc:
                raise RuntimeError("The pasted REDMIX_DIRECT_URL is not a valid http(s) URL.")

            # Do not append tokens to an entitlement-bearing browser URL.
            cmd = [
                "curl", "-L", "--fail-with-body", "--show-error", "--progress-bar",
                "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
                "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
                "-A", "Mozilla/5.0", "-o", part, pasted_url,
            ]
            log("    attempt: pasted authorized browser URL")
            rr = subprocess.run(cmd)
            if rr.returncode == 0 and os.path.isfile(part):
                size = os.path.getsize(part)
                head = b""
                try:
                    head = open(part, "rb").read(512).lower()
                except Exception:
                    pass
                if size >= 8 * 1024 * 1024 and b"<html" not in head and b"<!doctype" not in head:
                    downloaded = True
                else:
                    errors.append(
                        f"pasted authorized URL: response was not checkpoint bytes ({size} bytes): {head[:120]!r}"
                    )
                    try: os.remove(part)
                    except Exception: pass
            else:
                errors.append(f"pasted authorized URL: curl exit {rr.returncode}")

        if not downloaded:
            raise RuntimeError(
                "The real REDMIX Beta2 checkpoint is still unavailable to this Colab session. "
                "The notebook will NOT silently substitute base H3.\n\n"
                f"Open: {REDMIX_PAGE_URL}\n"
                "Use one of these authorized paths:\n"
                f"  • Colab Secret {REDMIX_DIRECT_URL_SECRET}: the actual signed-in Download link\n"
                f"  • Colab Secret {REDMIX_LOCAL_PATH_SECRET}: a local/Drive path to the checkpoint\n"
                f"  • Place {REDMIX_FILE} directly in /content/ComfyUI/models/diffusion_models/\n\n"
                "The expected SHA256 is:\n"
                f"  {REDMIX_SHA256}\n\n"
                "Download attempts:\n  " + "\n  ".join(errors)
            )

    got = sha(part)
    if got != REDMIX_SHA256:
        bad_size = os.path.getsize(part) / (1024**3)
        os.remove(part)
        raise RuntimeError(
            f"REDMIX SHA256 mismatch after download ({bad_size:.2f} GiB): "
            f"expected {REDMIX_SHA256}, got {got}. "
            "CivitAI likely returned a different file/revision; refusing to install it."
        )

    os.replace(part, dest)
    log(f"  ✓ {REDMIX_FILE} (SHA256 verified)")
    USING_REDMIX = True
    DIT_FILE = REDMIX_FILE
    return True

def _activate_public_nsfw_fallback(reason=""):
    global USING_REDMIX, DIT_FILE, LIGHTNING_DEFAULT
    USING_REDMIX = False
    DIT_FILE = FALLBACK_DIT_FILE
    LIGHTNING_DEFAULT = True
    log("!")
    log("  ⚠⚠⚠ REDMIX BETA2 NOT LOADED ⚠⚠⚠")
    log("  ↳ publisher/access gate prevented the real REDMIX checkpoint from loading")
    log("  ↳ ACTIVATING PUBLIC NSFW FALLBACK: MiniMax-H3 INT8 + NaughtyTimes v2 @ 1.0 + LightX2V 4-step")
    if reason:
        msg = str(reason).split("\n", 1)[0]
        log(f"  ↳ reason: {msg[:320]}")
    log("!")

try:
    _download_redmix_beta2()
except Exception as _redmix_error:
    if REQUIRE_REDMIX:
        raise
    _activate_public_nsfw_fallback(_redmix_error)

FILES = [("text_encoders", TEXT_ENCODER_FILE,
          "Comfy-Org/MiniMax-H3", "text_encoders"),
         ("vae", "minimax_h3_video_vae_fp16.safetensors",
          "Comfy-Org/MiniMax-H3", "vae"),
         ("vae", "minimax_h3_audio_vae_fp32.safetensors",
          "Comfy-Org/MiniMax-H3", "vae")]
if not USING_REDMIX:
    FILES.append(("diffusion_models", FALLBACK_DIT_FILE,
                  "Comfy-Org/MiniMax-H3", "diffusion_models"))
# REDMIX + the smaller official NVFP4/AWQ text encoder + VAEs are downloaded
# concurrently. The text encoder is ~14.61 GiB instead of ~25.28 GiB INT8, so the
# first launch spent most of its time waiting on one stream at a time.
# hf_hub_download is thread-safe (per-file locks in the cache), so they
# now run concurrently: total wall time collapses toward the single
# largest file. Progress bars interleave; the byte counts still climb.
from concurrent.futures import ThreadPoolExecutor as _TPE

def _fetch_one(job):
    sub, fname, repo, remote_sub = job
    os.makedirs(os.path.join(MODELS, sub), exist_ok=True)
    dest = os.path.join(MODELS, sub, fname)
    if os.path.exists(dest):
        log(f"  ✓ {fname}"); return
    log(f"  ↓ {fname}  ({repo})")
    p = hf_hub_download(repo, filename=fname, subfolder=remote_sub)
    # The 10Eros repo nests under FL2VA/ while ComfyUI wants a flat folder.
    if os.path.abspath(p) != os.path.abspath(dest):
        shutil.copy(p, dest)

with _TPE(max_workers=4) as _pool:
    # list() propagates the first exception instead of swallowing it —
    # a missing weight must still be a loud failure, exactly as before.
    list(_pool.map(_fetch_one, FILES))
log(f"✓ transformer: {DIT_FILE}")

# ── NaughtyTimes v2 PRUNED LoRA ─────────────────────────────────────────────
# This LoRA is REQUIRED for this studio. The exact pruned r256 v2 build is
# matched to the stock Comfy-Org pruned FL2VA transformer above.
#
# Download priority:
#   1) already-present local file (SHA256 verified)
#   2) optional Hugging Face override via NSFW_LORA_HF_REPO / NSFW_LORA_HF_FILE
#   3) optional direct URL override via NSFW_LORA_DIRECT_URL
#   4) CivitAI version 3212436 using CIVITAI_API_KEY / *_TOKEN
#   5) Colab upload prompt for the exact .safetensors file
#
# The studio never silently starts without this LoRA.
ldir = os.path.join(MODELS, "loras")
os.makedirs(ldir, exist_ok=True)
NSFW_LORA_PATH = os.path.join(ldir, NSFW_LORA_FILE)

# Optional non-CivitAI mirror overrides. Leave blank unless you have an exact
# mirror of NaughtyTimes_pruned_r256_v2.safetensors. SHA256 is always checked.
NSFW_LORA_HF_REPO = os.environ.get("NSFW_LORA_HF_REPO", "").strip()
NSFW_LORA_HF_FILE = os.environ.get("NSFW_LORA_HF_FILE", NSFW_LORA_FILE).strip()
NSFW_LORA_DIRECT_URL = os.environ.get("NSFW_LORA_DIRECT_URL", "").strip()


def _civitai_token():
    """Return the required CivitAI token from the exact Colab Secret name.

    This studio expects a Colab Secret named CIVITAI_API_KEY with notebook
    access enabled. We intentionally do not silently probe alternate names:
    if the configured secret cannot be read, startup should explain exactly
    what needs fixing.
    """
    tok = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    if tok:
        return tok

    try:
        from google.colab import userdata
        tok = (userdata.get("CIVITAI_API_KEY") or "").strip()
    except Exception as e:
        raise RuntimeError(
            "Could not read the Colab Secret CIVITAI_API_KEY.\n"
            "Open Colab's Secrets panel (key icon), make sure a secret named "
            "CIVITAI_API_KEY exists, and enable notebook access for it.\n"
            f"Colab userdata error: {e}"
        ) from e

    if not tok:
        raise RuntimeError(
            "Colab Secret CIVITAI_API_KEY is missing or empty.\n"
            "Create/enable that exact secret in Colab's Secrets panel, then rerun."
        )

    os.environ["CIVITAI_API_KEY"] = tok
    return tok

def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _verify_nsfw_lora(path, label=None):
    label = label or os.path.basename(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Required NSFW LoRA is missing: {path}")
    got = _sha256(path)
    if got != NSFW_LORA_SHA256:
        raise RuntimeError(
            f"{label} failed SHA256 verification.\n"
            f"Expected: {NSFW_LORA_SHA256}\n"
            f"Got:      {got}\n"
            "This studio requires the exact pruned r256 v2 LoRA matched to the pruned H3 base.")
    log(f"  ✓ {NSFW_LORA_FILE} (SHA256 verified)")
    return True


def _civitai_file(version_id, wanted_name, token=""):
    api = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {
        "User-Agent": "MissingLink-MiniMax-H3-Studio/1.2",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(api, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    files = data.get("files") or []
    exact = [f for f in files if (f.get("name") or "").lower() == wanted_name.lower()]
    if not exact:
        exact = [f for f in files
                 if "pruned_r256_v2" in (f.get("name") or "").lower()]
    if not exact:
        names = ", ".join(f.get("name", "?") for f in files)
        raise RuntimeError(
            f"CivitAI version {version_id} no longer exposes {wanted_name}. "
            f"Files currently returned: {names or '(none)'}")
    f = exact[0]
    url = f.get("downloadUrl")
    if not url:
        raise RuntimeError(f"CivitAI returned no downloadUrl for {f.get('name')}")
    return f, url


def _curl_download(url, dest, token="", display_name=None, require_token=False):
    """Download through curl with resume/retry. For CivitAI, append ?token=.
    For direct mirrors, leave token blank and require_token=False."""
    display_name = display_name or os.path.basename(dest)
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    final_url = url
    if token:
        p = urlsplit(url)
        q = dict(parse_qsl(p.query, keep_blank_values=True))
        q["token"] = token
        final_url = urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))
    elif require_token:
        raise RuntimeError("CivitAI token required for this download path.")

    part = dest + ".part"
    cmd = [
        "curl", "-L", "--fail-with-body", "--show-error", "--progress-bar",
        "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
        "--connect-timeout", "30", "--continue-at", "-",
        "--header", "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "--output", part, final_url,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        try:
            os.remove(part)
        except OSError:
            pass
        raise RuntimeError(f"Download failed for {display_name} (curl exit {rc}).")
    os.replace(part, dest)


def _download_nsfw_from_hf():
    if not NSFW_LORA_HF_REPO:
        return False
    log(f"  ↳ trying Hugging Face override: {NSFW_LORA_HF_REPO}/{NSFW_LORA_HF_FILE}")
    try:
        src = hf_hub_download(NSFW_LORA_HF_REPO, filename=NSFW_LORA_HF_FILE)
        shutil.copy2(src, NSFW_LORA_PATH)
        _verify_nsfw_lora(NSFW_LORA_PATH, "Hugging Face NSFW LoRA")
        return True
    except Exception as e:
        log(f"  ⚠ Hugging Face override failed: {e}")
        if os.path.exists(NSFW_LORA_PATH):
            try: os.remove(NSFW_LORA_PATH)
            except OSError: pass
        return False


def _download_nsfw_from_direct_url():
    if not NSFW_LORA_DIRECT_URL:
        return False
    log("  ↳ trying NSFW_LORA_DIRECT_URL mirror")
    try:
        _curl_download(NSFW_LORA_DIRECT_URL, NSFW_LORA_PATH,
                       display_name=NSFW_LORA_FILE)
        _verify_nsfw_lora(NSFW_LORA_PATH, "direct-mirror NSFW LoRA")
        return True
    except Exception as e:
        log(f"  ⚠ direct mirror failed: {e}")
        if os.path.exists(NSFW_LORA_PATH):
            try: os.remove(NSFW_LORA_PATH)
            except OSError: pass
        return False


def _download_nsfw_from_civitai(tok):
    log(f"  ↳ resolving CivitAI v{NSFW_LORA_VERSION}: {NSFW_LORA_FILE}")
    meta, url = _civitai_file(NSFW_LORA_VERSION, NSFW_LORA_FILE, tok)
    api_sha = ((meta.get("hashes") or {}).get("SHA256") or "").lower()
    if api_sha and api_sha != NSFW_LORA_SHA256:
        raise RuntimeError(
            f"CivitAI's SHA256 for {meta.get('name')} changed: {api_sha}. "
            "Refusing to download an unexpected file.")
    log(f"  ↓ NSFW LoRA: {meta.get('name', NSFW_LORA_FILE)} "
        f"({float(meta.get('sizeKB') or 0)/1024/1024:.2f} GB)")
    _curl_download(url, NSFW_LORA_PATH, tok, NSFW_LORA_FILE, require_token=True)
    _verify_nsfw_lora(NSFW_LORA_PATH, "CivitAI NSFW LoRA")
    return True


def _upload_nsfw_in_colab():
    """Last-resort interactive fallback. The LoRA is still mandatory."""
    try:
        from google.colab import files as colab_files
    except Exception:
        return False

    log("\n  REQUIRED LoRA not found and no working authenticated/mirror download is available.")
    log(f"  Upload the exact file now: {NSFW_LORA_FILE}")
    log(f"  Expected SHA256: {NSFW_LORA_SHA256}")
    uploaded = colab_files.upload()
    if not uploaded:
        return False

    # Prefer exact filename, otherwise accept a single uploaded safetensors file
    # only if its SHA256 proves it is the exact required LoRA.
    names = list(uploaded.keys())
    candidate = NSFW_LORA_FILE if NSFW_LORA_FILE in names else None
    if candidate is None:
        safes = [n for n in names if n.lower().endswith(".safetensors")]
        if len(safes) == 1:
            candidate = safes[0]
    if candidate is None:
        raise RuntimeError(
            f"Upload must include {NSFW_LORA_FILE} (or exactly one .safetensors file).")

    src = os.path.abspath(candidate)
    if src != os.path.abspath(NSFW_LORA_PATH):
        shutil.move(src, NSFW_LORA_PATH)
    _verify_nsfw_lora(NSFW_LORA_PATH, "uploaded NSFW LoRA")
    return True


def _ensure_nsfw_lora():
    # Existing exact file: verify and reuse it.
    if os.path.exists(NSFW_LORA_PATH):
        try:
            return _verify_nsfw_lora(NSFW_LORA_PATH)
        except Exception as e:
            bad = NSFW_LORA_PATH + f".bad-{int(time.time())}"
            log(f"  ⚠ existing {NSFW_LORA_FILE} is invalid; moving to {bad}")
            os.replace(NSFW_LORA_PATH, bad)
            log(f"    {e}")

    # The required LoRA is fetched from its exact CivitAI model version using
    # the user's Colab Secret CIVITAI_API_KEY. This is non-interactive: no
    # browser upload widget and no optional fallback that can launch without it.
    tok = _civitai_token()
    log("  ↳ CivitAI auth: CIVITAI_API_KEY loaded")

    try:
        return _download_nsfw_from_civitai(tok)
    except Exception as e:
        raise RuntimeError(
            f"Could not download required {NSFW_LORA_FILE} from CivitAI v{NSFW_LORA_VERSION}.\n"
            "The CIVITAI_API_KEY secret was found, but the authenticated download failed.\n"
            "Check that the token is valid and has model-download access, then rerun.\n"
            f"Underlying error: {e}"
        ) from e


_ensure_nsfw_lora()
log(f"✓ REQUIRED NSFW LoRA: {NSFW_LORA_FILE} @ {NSFW_LORA_STRENGTH}")

# ── Optional LightX2V Lightning LoRA ─────────────────────────────────────────
LIGHTNING_PATH = os.path.join(ldir, LIGHTNING_FILE)
def _ensure_lightning_lora():
    if os.path.exists(LIGHTNING_PATH):
        log(f"  ✓ Lightning: {LIGHTNING_FILE}")
        return True
    try:
        log(f"  ↓ Lightning LoRA: {LIGHTNING_FILE} ({LIGHTNING_REPO})")
        src = hf_hub_download(LIGHTNING_REPO, filename=LIGHTNING_FILE)
        shutil.copy(src, LIGHTNING_PATH)
        log(f"  ✓ Lightning: {LIGHTNING_FILE}")
        return True
    except Exception as e:
        # Lightning is optional; the studio must remain usable with NaughtyTimes only.
        log(f"  ⚠ Lightning download failed (optional): {e}")
        return False

LIGHTNING_AVAILABLE = _ensure_lightning_lora()

# ── Optional CivitAI action LoRA (auto-select MiniMax H3 version) ────────────
def _civitai_json(url, token=""):
    headers = {
        "User-Agent": "MissingLink-MiniMax-H3-Studio/1.1",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)

def _is_h3_version(v):
    # Version descriptions often mention sibling WAN/LTX releases, so do not use
    # description text for compatibility. Match only the version name/baseModel.
    base = str(v.get("baseModel") or "").lower()
    name = str(v.get("name") or "").lower()
    hay = f"{base} {name}"
    if "wan" in base or "ltx" in base:
        return False
    return "mh3" in hay or "minimax h3" in hay or ("minimax" in hay and "h3" in hay)

def _pick_model_file(v):
    files = v.get("files") or []
    safe = [f for f in files if (f.get("name") or "").lower().endswith(".safetensors")]
    if not safe:
        return None
    prim = [f for f in safe if f.get("primary")]
    pool = prim or safe
    # Prefer normal model files over training data/config artifacts.
    pool.sort(key=lambda f: (str(f.get("type") or "").lower() != "model", -(float(f.get("sizeKB") or 0))))
    return pool[0]

def _ensure_action_lora():
    global ACTION_META, ACTION_PATH, ACTION_FILE, ACTION_AVAILABLE
    tok = _civitai_token()
    if not tok:
        log("  ⚠ action LoRA skipped: no CivitAI token available")
        return False
    try:
        model = _civitai_json(f"https://civitai.com/api/v1/models/{ACTION_MODEL_ID}", tok)
        versions = model.get("modelVersions") or []
        compatible = [v for v in versions if _is_h3_version(v) and _pick_model_file(v)]
        if not compatible:
            linked = next((v for v in versions if int(v.get("id") or 0)==ACTION_LINKED_VERSION), None)
            linked_desc = (f"; linked v{ACTION_LINKED_VERSION} is {linked.get('name')} / {linked.get('baseModel')}"
                           if linked else "")
            log(f"  ⚠ action LoRA model {ACTION_MODEL_ID}: no MiniMax-H3 version found{linked_desc}")
            return False
        compatible.sort(key=lambda v: (str(v.get("createdAt") or ""), int(v.get("id") or 0)), reverse=True)
        v = compatible[0]
        f = _pick_model_file(v)
        ACTION_FILE = f.get("name")
        ACTION_PATH = os.path.join(ldir, ACTION_FILE)
        ACTION_META = {
            "model_id": ACTION_MODEL_ID,
            "version_id": int(v.get("id") or 0),
            "version_name": v.get("name") or "",
            "base_model": v.get("baseModel") or "",
            "trained_words": v.get("trainedWords") or [],
            "file": ACTION_FILE,
            "size_kb": float(f.get("sizeKB") or 0),
            "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower(),
        }
        log(f"  ↳ action model {ACTION_MODEL_ID}: linked v{ACTION_LINKED_VERSION} is not H3; "
            f"using H3 v{ACTION_META['version_id']} {ACTION_META['version_name']} "
            f"({ACTION_META['base_model']})")
        if ACTION_META["trained_words"]:
            log("  ↳ action trigger words: " + ", ".join(ACTION_META["trained_words"]))

        expected = ACTION_META["sha256"]
        if os.path.exists(ACTION_PATH):
            if expected:
                got = _sha256(ACTION_PATH)
                if got == expected:
                    log(f"  ✓ action LoRA: {ACTION_FILE} (SHA256 verified)")
                    ACTION_AVAILABLE = True
                    return True
                bad = ACTION_PATH + f".bad-{int(time.time())}"
                log(f"  ⚠ existing action LoRA hash mismatch; moving to {bad}")
                os.replace(ACTION_PATH, bad)
            else:
                log(f"  ✓ action LoRA already present: {ACTION_FILE}")
                ACTION_AVAILABLE = True
                return True

        url = f.get("downloadUrl") or v.get("downloadUrl")
        if not url:
            raise RuntimeError("CivitAI API returned no download URL for the H3 action LoRA")
        log(f"  ↓ action LoRA: {ACTION_FILE} ({ACTION_META['size_kb']/1024/1024:.2f} GB)")
        _curl_download(url, ACTION_PATH, tok, ACTION_FILE)
        if expected:
            got = _sha256(ACTION_PATH)
            if got != expected:
                raise RuntimeError(f"Action LoRA SHA256 mismatch: expected {expected}, got {got}")
            log(f"  ✓ action LoRA: {ACTION_FILE} (SHA256 verified)")
        else:
            log(f"  ✓ action LoRA: {ACTION_FILE}")
        ACTION_AVAILABLE = True
        return True
    except Exception as e:
        ACTION_AVAILABLE = False
        log(f"  ⚠ action LoRA unavailable (optional): {e}")
        return False

ACTION_AVAILABLE = _ensure_action_lora()

# ── 2. Import ComfyUI as a library ─────────────────────────────────────────
sys.path.insert(0, COMFY_DIR)
import nest_asyncio; nest_asyncio.apply()

# ComfyUI's memory manager reads comfy.cli_args at import time. In library mode
# nothing populates it, so it defaults to keeping every weight resident and the
# DiT alone fills the card. Set the flags Comfy's own launcher would set BEFORE
# importing anything under comfy.*.
import comfy.options
comfy.options.enable_args_parsing(False)
from comfy.cli_args import args
args.reserve_vram = RESERVE_VRAM          # GB kept free for activations
args.preview_method = "none"              # no per-step preview encode
# A100 40GB policy: use NORMAL_VRAM so the 32B text encoder and H3 DiT each
# get the GPU when active, while Comfy's smart-memory manager swaps them between
# stages. LOW_VRAM forces block streaming/CPU work and leaves the A100 idle.
for _name, _value in {
    "lowvram": False,
    "novram": False,
    "highvram": False,
    "gpu_only": False,
    "normalvram": True,
}.items():
    if hasattr(args, _name):
        setattr(args, _name, _value)

# Attention aliases are chosen while comfy.ldm modules import, so set this BEFORE
# importing nodes/model_management. This is the critical part for MiniMax H3,
# which imports optimized_attention by value.
for _name in ("use_sage_attention", "use_flash_attention",
              "use_pytorch_cross_attention", "use_split_cross_attention",
              "use_quad_cross_attention"):
    if hasattr(args, _name):
        setattr(args, _name, False)
if ATTN_BACKEND == "sageattention" and hasattr(args, "use_sage_attention"):
    args.use_sage_attention = True
elif hasattr(args, "use_pytorch_cross_attention"):
    args.use_pytorch_cross_attention = True
if hasattr(args, "disable_xformers"):
    args.disable_xformers = True

import nodes, folder_paths, comfy.utils
import comfy.model_management as mm
log(f"  vram state: {mm.vram_state}  · reserve {RESERVE_VRAM} GB · LOWVRAM={LOWVRAM}")
log(f"  attention active -> {ATTN_BACKEND} (selected before H3 module import)")

# init_extra_nodes is a coroutine; unawaited it silently registers nothing.
_r = nodes.init_extra_nodes()
if asyncio.iscoroutine(_r):
    asyncio.get_event_loop().run_until_complete(_r)
N = nodes.NODE_CLASS_MAPPINGS
log(f"✓ {len(N)} nodes registered")
for req in ("MiniMaxH3ImageToVideo","MiniMaxH3SigmaShift","VAEDecodeAudio"):
    if req not in N: raise RuntimeError(f"node {req} missing — engine too old")

def call(name, **kw):
    cls = N[name]
    return getattr(cls(), cls.FUNCTION)(**kw)

def opts(node, field):
    spec = N[node].INPUT_TYPES()["required"][field]
    o = spec[1].get("options") if len(spec)>1 and isinstance(spec[1],dict) else None
    return o or (spec[0] if isinstance(spec[0],list) else [])

SAMPLERS   = opts("KSamplerSelect","sampler_name")
SCHEDULERS = opts("BasicScheduler","scheduler")
LORAS      = folder_paths.get_filename_list("loras")

# ── 3. Progress ────────────────────────────────────────────────────────────
PROG = {"cur":0,"total":0,"stage":"idle"}

# ── Continuous GPU telemetry ────────────────────────────────────────────────
# Sample nvidia-smi once per second. The latest sample is exposed to the UI,
# and while a render is active a compact line is also injected into RAW CONSOLE
# every two seconds so the generation transcript contains utilization history.
_GPU_LOCK = threading.Lock()
_GPU_TELEMETRY = {
    "ok": False, "ts": 0.0, "util": None, "mem_util": None,
    "mem_used_mb": None, "mem_total_mb": None, "temp_c": None,
    "power_w": None, "power_limit_w": None, "clock_mhz": None,
    "stage": "idle", "error": "starting",
}

def _gpu_sample():
    fields = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit,clocks.current.graphics"
    )
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "nvidia-smi failed").strip())
    vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    if len(vals) < 8:
        raise RuntimeError("unexpected nvidia-smi output: " + r.stdout.strip())
    def num(x, kind=float):
        try: return kind(float(x)) if kind is int else float(x)
        except Exception: return None
    return {
        "ok": True, "ts": time.time(),
        "util": num(vals[0], int), "mem_util": num(vals[1], int),
        "mem_used_mb": num(vals[2]), "mem_total_mb": num(vals[3]),
        "temp_c": num(vals[4]), "power_w": num(vals[5]),
        "power_limit_w": num(vals[6]), "clock_mhz": num(vals[7], int),
        "stage": PROG.get("stage", "idle"), "error": "",
    }

def _gpu_monitor_loop():
    last_console = 0.0
    while True:
        try:
            sample = _gpu_sample()
        except Exception as e:
            sample = dict(_GPU_TELEMETRY)
            sample.update(ok=False, ts=time.time(), stage=PROG.get("stage", "idle"), error=str(e))
        with _GPU_LOCK:
            _GPU_TELEMETRY.clear(); _GPU_TELEMETRY.update(sample)
        now = time.time()
        stage = PROG.get("stage", "idle")
        if sample.get("ok") and stage not in ("idle", "ready") and now - last_console >= 2.0:
            used = (sample.get("mem_used_mb") or 0) / 1024.0
            total = (sample.get("mem_total_mb") or 0) / 1024.0
            _console_push(
                "[GPU] "
                f"util={sample.get('util')}% | mem-util={sample.get('mem_util')}% | "
                f"VRAM={used:.1f}/{total:.1f} GiB | temp={sample.get('temp_c')}C | "
                f"power={sample.get('power_w'):.0f}/{sample.get('power_limit_w'):.0f} W | "
                f"clock={sample.get('clock_mhz')} MHz | stage={stage}",
                "stdout")
            last_console = now
        time.sleep(1.0)

threading.Thread(target=_gpu_monitor_loop, daemon=True, name="gpu-telemetry").start()
# ComfyUI calls this as hook(current, total, preview, node_id=...) — the extra
# keyword is not optional to accept, so swallow anything else it adds later.
comfy.utils.set_progress_bar_global_hook(
    lambda cur, total, preview=None, **kw: PROG.update(cur=cur, total=total))

# ── 4. Model cache ─────────────────────────────────────────────────────────
CACHE = {}

def _patch_count(model):
    """Count patch entries, not just patch keys. Two stacked LoRAs usually patch
    the same parameter names, so len(model.patches) alone cannot prove the second
    LoRA was actually added."""
    patches = getattr(model, "patches", {}) or {}
    total = 0
    for v in patches.values():
        if isinstance(v, (list, tuple)):
            total += len(v)
        else:
            total += 1
    return total

def _apply_lora_checked(model, name, strength, label):
    if not name or name == "none" or float(strength) == 0:
        return model, None
    before = _patch_count(model)
    model, = call("LoraLoaderModelOnly", model=model, lora_name=name,
                  strength_model=float(strength))
    after = _patch_count(model)
    if after <= before:
        raise RuntimeError(
            f"{label} LoRA '{name}' applied 0 new patch entries. "
            f"It is not compatible with {DIT_FILE}, or ComfyUI could not map its keys.")
    info = f"{label}: {name} @ {float(strength):g} (+{after-before} patches)"
    log(f"  ✓ LoRA ACTIVE: {info}")
    return model, info

def get_models(weight_dtype, lora, lora_strength, action=False,
               action_strength=ACTION_STRENGTH, lightning=False,
               lightning_strength=LIGHTNING_STRENGTH, unet=None):
    """Cache the base H3 weights and stack generation LoRAs on a clone per job.
    The primary LoRA (normally NaughtyTimes) is applied first; the optional
    compatible H3 action LoRA is applied second; LightX2V acceleration is last."""
    unet = unet or DIT_FILE
    if CACHE.get("key") != (unet, weight_dtype):
        CACHE.clear()
        PROG["stage"] = "loading unet"
        log(f"  loading transformer: {unet}")
        base, = call("UNETLoader", unet_name=unet, weight_dtype=weight_dtype)
        PROG["stage"] = "loading clip"
        clip, = call("CLIPLoader",
                     clip_name=TEXT_ENCODER_FILE,
                     type="minimax")
        PROG["stage"] = "loading vae"
        vae,  = call("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
        avae, = call("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
        CACHE.update(key=(unet, weight_dtype), base=base, clip=clip,
                     vae=vae, avae=avae)

    use_action = str(action).lower() in ("1", "true", "yes", "on")
    use_lightning = str(lightning).lower() in ("1", "true", "yes", "on")
    variant_key = (
        unet, weight_dtype, str(lora or "none"), float(lora_strength),
        bool(use_action), float(action_strength),
        bool(use_lightning), float(lightning_strength),
    )
    if CACHE.get("variant_key") == variant_key and CACHE.get("variant_model") is not None:
        return (CACHE["variant_model"], CACHE["clip"], CACHE["vae"], CACHE["avae"],
                CACHE.get("variant_info", "none"))

    model = CACHE["base"]
    infos = []
    model, info = _apply_lora_checked(model, lora, lora_strength, "primary")
    if info: infos.append(info)

    if use_action:
        if not ACTION_AVAILABLE or not ACTION_FILE or not ACTION_PATH or not os.path.exists(ACTION_PATH):
            raise RuntimeError(
                "The action LoRA is enabled, but no MiniMax-H3-compatible file from "
                f"CivitAI model {ACTION_MODEL_ID} is available. Check the startup log.")
        if lora == ACTION_FILE:
            log("  ↳ action LoRA is already selected as primary; not applying twice")
        else:
            model, info = _apply_lora_checked(
                model, ACTION_FILE, action_strength, "action")
            if info: infos.append(info)

    if use_lightning:
        if not os.path.exists(LIGHTNING_PATH):
            raise RuntimeError(
                f"Lightning is enabled but {LIGHTNING_FILE} is not present in {ldir}.")
        if lora == LIGHTNING_FILE:
            log("  ↳ Lightning file is already selected as the primary LoRA; not applying twice")
        else:
            model, info = _apply_lora_checked(
                model, LIGHTNING_FILE, lightning_strength, "Lightning")
            if info: infos.append(info)

    variant_info = " | ".join(infos) or "none"
    CACHE["variant_key"] = variant_key
    CACHE["variant_model"] = model
    CACHE["variant_info"] = variant_info
    return model, CACHE["clip"], CACHE["vae"], CACHE["avae"], variant_info

# Preload the default generation stack before the UI starts. The smaller official
# NVFP4/AWQ encoder cuts the conditioning model from ~25.28 GiB to ~14.61 GiB.
# REDMIX + TE is still deliberately not force-pinned together: on a 40 GB A100
# the remaining activation/workspace margin is too small at 1152x768. Keep the
# patched REDMIX DiT resident, then let Comfy smart-memory swap the TE onto the GPU
# for conditioning and restore REDMIX for sampling.
def _log_vram_budget():
    try:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        total = 39.5
    dit_gib = REDMIX_GIB if USING_REDMIX else 19.55
    dit_label = "REDMIX" if USING_REDMIX else "base H3 INT8"
    core = dit_gib + TEXT_ENCODER_GIB
    all_weights = core + VIDEO_VAE_GIB + AUDIO_VAE_GIB
    log(f"  VRAM budget -> {dit_label} ~{dit_gib:.2f} GiB + TE ~{TEXT_ENCODER_GIB:.2f} GiB = ~{core:.2f} GiB")
    log(f"  VRAM budget -> + video/audio VAE ~= {all_weights:.2f} GiB weights vs {total:.1f} GiB physical")
    log(f"  VRAM policy -> keep {'REDMIX' if USING_REDMIX else 'base H3 + LoRAs'} resident; GPU-swap {TEXT_ENCODER_FILE} for conditioning; reserve {RESERVE_VRAM:.1f} GiB")

def _preload_default_gpu_stack():
    PROG["stage"] = "startup preload"
    _log_vram_budget()
    log("  ↳ preloading " + ("REDMIX H3 Beta2" if USING_REDMIX else "public H3 + NaughtyTimes + LightX2V fallback") + " to GPU before UI launch")
    try:
        preload_lora = "none" if USING_REDMIX else NSFW_LORA_FILE
        preload_lora_strength = 0.0 if USING_REDMIX else NSFW_LORA_STRENGTH
        preload_lightning = False if USING_REDMIX else True
        model, clip, vae, avae, info = get_models(
            "default", preload_lora, preload_lora_strength,
            action=False, action_strength=ACTION_STRENGTH,
            lightning=preload_lightning,
            lightning_strength=LIGHTNING_STRENGTH,
            unet=DIT_FILE,
        )
        # ModelPatcher objects are what Comfy's memory manager expects here.
        # force_full_load asks NORMAL_VRAM to place the complete patched DiT on GPU.
        try:
            mm.load_models_gpu([model], force_full_load=True)
        except TypeError:
            mm.load_models_gpu([model])
        torch.cuda.synchronize()
        free_b, total_b = torch.cuda.mem_get_info()
        used = (total_b-free_b)/1024**3
        total = total_b/1024**3
        log(f"  ✓ startup GPU preload complete: {used:.1f}/{total:.1f} GiB VRAM used")
        log(f"  ✓ resident startup stack: {info}")
        PROG["stage"] = "ready"
        return True
    except Exception as e:
        PROG["stage"] = "ready"
        log(f"  ⚠ startup GPU preload warning: {e}")
        log("    UI will still start; Comfy will retry model loading on GENERATE.")
        return False

STARTUP_GPU_PRELOADED = _preload_default_gpu_stack()

def _prepare_frame(path, width, height, fit_mode="cover"):
    """Resize without accidental aspect distortion before H3 sees the frame.
    cover = preserve aspect + center crop; contain = preserve aspect + letterbox;
    stretch = old behavior."""
    img = Image.open(path).convert("RGB")
    size = (int(width), int(height))
    resample = Image.Resampling.LANCZOS
    mode = (fit_mode or "cover").lower()
    if mode == "contain":
        fitted = ImageOps.contain(img, size, method=resample)
        canvas = Image.new("RGB", size, (0, 0, 0))
        x = (size[0] - fitted.width) // 2
        y = (size[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        img = canvas
    elif mode == "stretch":
        img = img.resize(size, resample)
    else:
        img = ImageOps.fit(img, size, method=resample, centering=(0.5, 0.5))
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None,]

def _snap_frames(value):
    try:
        value = float(value)
    except Exception:
        value = MIN_FRAMES
    k = round((value - 5) / 17)
    frames = 17 * max(0, k) + 5
    return max(MIN_FRAMES, min(MAX_FRAMES, frames))

def resolve_length(p):
    """Resolve UI length to H3's legal 17*n+5 frame grid.

    In seconds mode, `duration` means FINAL OUTPUT duration. Because optional
    playback retiming happens after generation, generate enough model-time that
    model_seconds / playback_speed ~= requested final seconds. This makes the
    duration field actually honor what the user typed.

    In frames mode, `frames` is an explicit MODEL frame request.
    """
    mode = (p.get("length_mode") or "seconds").lower()
    speed = max(0.05, min(8.0, float(p.get("playback_speed") or 1.0)))
    if mode == "frames":
        requested_frames = float(p.get("frames") or 124)
        frames = _snap_frames(requested_frames)
        requested_final_seconds = (frames / MODEL_FPS) / speed
    else:
        requested_final_seconds = max(0.01, float(p.get("duration") or 20.0))
        frames = _snap_frames(requested_final_seconds * speed * MODEL_FPS)
    model_seconds = frames / MODEL_FPS
    return frames, model_seconds, requested_final_seconds

def _atempo_chain(speed):
    # ffmpeg atempo supports 0.5..2.0 per stage; chain stages outside that range.
    x = float(speed)
    parts = []
    while x > 2.0 + 1e-9:
        parts.append(2.0); x /= 2.0
    while x < 0.5 - 1e-9:
        parts.append(0.5); x /= 0.5
    parts.append(x)
    return ",".join(f"atempo={v:.8g}" for v in parts)

def _retime_video(src, dest, speed):
    speed = float(speed)
    if abs(speed - 1.0) < 1e-6:
        os.replace(src, dest)
        return True, ""
    if speed <= 0:
        raise ValueError("playback speed must be greater than 0")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        os.replace(src, dest)
        return False, "ffmpeg not found; playback speed was not applied"
    filt = f"[0:v]setpts=PTS/{speed:.10g}[v];[0:a]{_atempo_chain(speed)}[a]"
    cmd = [ffmpeg, "-y", "-i", src, "-filter_complex", filt,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
           "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-movflags", "+faststart", dest]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        log("  ⚠ ffmpeg retime failed; keeping native-speed output: " + r.stderr[-500:])
        if os.path.exists(dest):
            os.remove(dest)
        os.replace(src, dest)
        return False, "ffmpeg retime failed; native 1.0x output kept"
    try: os.remove(src)
    except OSError: pass
    return True, ""

# ── 5. Generate ────────────────────────────────────────────────────────────
OUT = "/content/h3_out"; os.makedirs(OUT, exist_ok=True)
JOBS = {}

# Only one generate may touch the GPU at a time. Without this, every failed or
# still-running job keeps its models resident and they compete for the card —
# which looks exactly like a memory leak but is really N pipelines at once.
GPU_LOCK = threading.Lock()

def generate(jid, p):
    j = JOBS[jid]
    if not GPU_LOCK.acquire(blocking=False):
        j.update(status="error",
                 msg="Another generation is still running on the GPU.\n"
                     "Wait for it to finish, or restart the runtime if it is stuck.")
        return
    try:
        # IMPORTANT: use no_grad(), NOT inference_mode().  ComfyUI smart-memory
        # may partially unload/reload ModelPatcher weights while swapping the
        # preloaded DiT for the 32B text encoder.  Tensors created/touched under
        # inference_mode cannot later be wrapped back into torch.nn.Parameter,
        # which causes: "Cannot set version_counter for inference tensor".
        # no_grad still prevents autograd graphs without changing tensor semantics.
        with torch.no_grad():
            _generate(jid, p)
    finally:
        GPU_LOCK.release()

def _generate(jid, p):
    j = JOBS[jid]
    try:
        j.update(status="running")

        # Keep the startup-preloaded DiT resident. Comfy smart memory will swap it
        # out only when the 32B text encoder needs the card for conditioning, then
        # bring the DiT back for sampling. Unloading everything here would defeat
        # the whole point of startup preloading.
        try:
            gc.collect(); mm.soft_empty_cache(); torch.cuda.empty_cache()
        except Exception:
            pass
        f0 = torch.cuda.mem_get_info()[0]/1e9
        log(f"  job {jid} start: {f0:.1f} GB free")
        j["memlog"] = f"VRAM at job start: {f0:.1f} GB free\n"

        model, clip, vae, avae, lora_info = get_models(
            p["weight_dtype"], p.get("lora"), p.get("lora_strength", NSFW_LORA_STRENGTH),
            action=p.get("action", "0"),
            action_strength=p.get("action_strength", ACTION_STRENGTH),
            lightning=p.get("lightning", "0"),
            lightning_strength=p.get("lightning_strength", LIGHTNING_STRENGTH),
            unet=p.get("unet") or DIT_FILE)
        j["lora_info"] = lora_info

        model, = call("MiniMaxH3SigmaShift", model=model,
                      shift_video=float(p["shift_video"]),
                      shift_audio=float(p["shift_audio"]))

        PROG["stage"] = "conditioning"
        n_frames, actual_sec, requested_sec = resolve_length(p)
        log(f"  length -> {n_frames} legal frames ({actual_sec:.2f}s model time; requested final {requested_sec:.2f}s)")
        log(f"  render config -> {p.get('width')}x{p.get('height')} | steps={p.get('steps')} | "
            f"Lightning={p.get('lightning','0')} | attention={ATTN_BACKEND} | LOWVRAM={LOWVRAM} | playback={p.get('playback_speed','1.0')}x")
        if n_frames > 362:
            log("  ⚠ long H3 clip: more than 362 model frames; sampling and VAE decode can be substantially slower/more memory-heavy")
        width, height = int(p["width"]), int(p["height"])
        if width % 32 or height % 32:
            raise ValueError("width and height must be multiples of 32")
        if width < 32 or height < 32:
            raise ValueError("width and height must be at least 32")
        kw = dict(clip=clip, vae=vae, prompt=p["prompt"],
                  width=width, height=height, length=n_frames)
        fit_mode = p.get("image_fit") or "cover"
        if p.get("first_frame"):
            kw["first_frame"] = _prepare_frame(p["first_frame"], width, height, fit_mode)
        if p.get("last_frame"):
            kw["last_frame"] = _prepare_frame(p["last_frame"], width, height, fit_mode)
        positive, latent = call("MiniMaxH3ImageToVideo", **kw)

        guider,  = call("BasicGuider", model=model, conditioning=positive)
        sampler, = call("KSamplerSelect", sampler_name=p["sampler_name"])
        sigmas,  = call("BasicScheduler", model=model, scheduler=p["scheduler"],
                        steps=int(p["steps"]), denoise=float(p["denoise"]))
        noise,   = call("RandomNoise", noise_seed=int(p["seed"]))

        PROG["stage"] = "sampling"
        samples = call("SamplerCustomAdvanced", noise=noise, guider=guider,
                       sampler=sampler, sigmas=sigmas, latent_image=latent)[0]

        # The 33B DiT is done but still resident; the VAE cannot fit beside it.
        # A real graph unloads between nodes — in library mode we must. Drop our
        # own references first or the patchers stay pinned. A bare `del` here
        # raises NameError if any name is unbound, which would skip the unload
        # entirely and OOM in the VAE.
        PROG["stage"] = "freeing dit"
        before = torch.cuda.mem_get_info()[0]/1e9
        guider = sampler = sigmas = noise = positive = latent = None
        model = clip = None
        gc.collect()
        unload_err = None
        try:
            mm.unload_all_models()
        except Exception as _e:
            unload_err = repr(_e)
        gc.collect()
        try: mm.soft_empty_cache()
        except Exception: pass
        torch.cuda.empty_cache()

        free_b, total_b = torch.cuda.mem_get_info()
        after = free_b/1e9
        MEMLOG = (f"VRAM  before unload: {before:.1f} GB free\n"
                  f"VRAM  after  unload: {after:.1f} GB free of {total_b/1e9:.1f} GB\n"
                  f"unload error: {unload_err}\n"
                  f"loaded models still held: {len(mm.current_loaded_models)}\n")
        log(MEMLOG)
        j["memlog"] = j.get("memlog","") + MEMLOG
        j["vram_free"] = round(after, 1)

        # NOTE: no tiled fallback. The H3 VAE's decode_tiled is a stub that
        # calls decode(), and the latent is a NestedTensor packing video and
        # audio rows, so tiling cannot split it. If this OOMs, cut `length`.
        PROG["stage"] = "decoding video"
        try:
            lat = samples["samples"]
            info = (f"latent type: {type(lat).__name__}\n"
                    f"latent dtype: {getattr(lat,'dtype',None)}\n"
                    f"latent shape: {getattr(lat,'shape','nested/unknown')}\n"
                    f"vae dtype: {getattr(vae,'vae_dtype',None)} "
                    f"device: {getattr(vae,'device',None)}\n"
                    f"vae offload device: {getattr(vae,'offload_device',None)}\n")
            log(info); j["memlog"] = j.get("memlog","") + info
        except Exception as _e:
            log(f"  latent introspection failed: {_e}")

        images, = call("VAEDecode", samples=samples, vae=vae)

        PROG["stage"] = "decoding audio"
        audio,  = call("VAEDecodeAudio", samples=samples, vae=avae)

        PROG["stage"] = "muxing"
        # Nothing here needs autograd, and Comfy's server normally runs under
        # inference mode. Without it the audio waveform arrives with a grad
        # graph attached and av's numpy conversion refuses it.
        if isinstance(audio, dict) and "waveform" in audio:
            audio = {**audio, "waveform": audio["waveform"].detach()}
        if torch.is_tensor(images):
            images = images.detach()

        video, = call("CreateVideo", images=images, fps=MODEL_FPS, audio=audio)

        PROG["stage"] = "saving"
        # Write the container directly. SaveVideo's format/codec are structured
        # V3 values that cannot be synthesised from the schema, and the video
        # object here is already complete.
        dest = os.path.join(OUT, f"{jid}.mp4")
        speed = max(0.05, min(8.0, float(p.get("playback_speed") or 1.35)))
        raw_dest = dest if abs(speed - 1.0) < 1e-6 else os.path.join(OUT, f"{jid}.native.mp4")
        video.save_to(raw_dest)
        if not os.path.exists(raw_dest):
            raise RuntimeError("save_to produced no file")
        speed_ok = True
        speed_note = ""
        if raw_dest != dest:
            PROG["stage"] = "retiming"
            speed_ok, speed_note = _retime_video(raw_dest, dest, speed)
        final_speed = speed if speed_ok else 1.0
        final_duration = actual_sec / final_speed

        j.update(status="done", file=os.path.basename(dest),
                 secs=round(time.time()-j["t0"],1), frames=n_frames,
                 model_duration=round(actual_sec, 2), duration=round(final_duration, 2),
                 playback_speed=round(final_speed, 3), note=speed_note)
    except Exception:
        traceback.print_exc()
        # Put the memory readings AT THE TOP of what the panel shows — the panel
        # truncates long tracebacks from the front, which hid this before.
        head = j.get("memlog", "(failed before the unload step)\n")
        j.update(status="error", msg=head + "\n" + traceback.format_exc()[-1200:])
    finally:
        PROG["stage"] = "ready"
        gc.collect(); torch.cuda.empty_cache()

# ── 6. UI ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.get("/api/meta")
def meta():
    # Read the folder fresh so LoRAs dropped in after startup appear.
    folder_paths.cache_helper.clear()
    return jsonify(samplers=SAMPLERS, schedulers=SCHEDULERS,
                   loras=["none"]+folder_paths.get_filename_list("loras"),
                   lora_default=("none" if USING_REDMIX else NSFW_LORA_FILE),
                   lora_strength_default=NSFW_LORA_STRENGTH,
                   lightning_file=LIGHTNING_FILE,
                   lightning_available=os.path.exists(LIGHTNING_PATH),
                   lightning_default=LIGHTNING_DEFAULT,
                   lightning_strength_default=LIGHTNING_STRENGTH,
                   action_model_id=ACTION_MODEL_ID,
                   action_linked_version=ACTION_LINKED_VERSION,
                   action_available=bool(ACTION_AVAILABLE and ACTION_FILE and ACTION_PATH and os.path.exists(ACTION_PATH)),
                   action_file=ACTION_FILE,
                   action_strength_default=ACTION_STRENGTH,
                   action_version_id=ACTION_META.get("version_id"),
                   action_version_name=ACTION_META.get("version_name", ""),
                   action_base_model=ACTION_META.get("base_model", ""),
                   action_trained_words=ACTION_META.get("trained_words", []),
                   unets=folder_paths.get_filename_list("diffusion_models"),
                   unet_default=DIT_FILE, using_redmix=USING_REDMIX,
                   model_mode=("REDMIX Beta2" if USING_REDMIX else "PUBLIC NSFW FALLBACK · H3 + NaughtyTimes + LightX2V"),
                   fallback_notice=("" if USING_REDMIX else "REDMIX unavailable/access-gated — using public H3 + NaughtyTimes v2 + LightX2V"),
                   recommended_steps=(6 if USING_REDMIX else 4),
                   recommended_sampler=("er_sde" if USING_REDMIX else "euler"),
                   recommended_scheduler=("beta" if USING_REDMIX else "simple"))

@app.get("/api/keepalive")
def keepalive(): return jsonify(ok=True)

@app.post("/api/generate")
def api_gen():
    if not ML_OK:
        return jsonify(error="MissingLink token not validated."), 402
    jid = uuid.uuid4().hex[:8]
    p = {k: request.form.get(k) for k in
         ("prompt","width","height","duration","frames","length_mode",
          "playback_speed","image_fit","steps","seed","denoise",
          "shift_video","shift_audio","sampler_name","scheduler",
          "weight_dtype","lora","lora_strength","action","action_strength","lightning",
          "lightning_strength","unet")}
    for k in ("first_frame","last_frame"):
        f = request.files.get(k)
        if f and f.filename:
            path = os.path.join(OUT, f"{jid}_{k}.png")
            Image.open(f.stream).convert("RGB").save(path)
            p[k] = path
    JOBS[jid] = {"status":"queued","t0":time.time()}
    threading.Thread(target=generate, args=(jid,p), daemon=True).start()
    return jsonify(id=jid)

@app.get("/api/job/<jid>")
def api_job(jid):
    j = dict(JOBS.get(jid, {"status":"error","msg":"unknown job"}))
    if j.get("status") in ("running","queued"):
        j["el"]=int(time.time()-j["t0"]); j["stage"]=PROG["stage"]
        j["cur"]=PROG["cur"]; j["total"]=PROG["total"]
    j.pop("t0", None)
    return jsonify(j)

@app.get("/api/gpu")
def api_gpu():
    with _GPU_LOCK:
        d = dict(_GPU_TELEMETRY)
    d["stage"] = PROG.get("stage", d.get("stage", "idle"))
    return jsonify(d)

@app.get("/api/console")
def api_console():
    try:
        since = int(request.args.get("since", "0") or 0)
    except Exception:
        since = 0
    with _CONSOLE_LOCK:
        rows = [(seq, stream, text) for seq, stream, text in _CONSOLE_LINES if seq > since]
        latest = _CONSOLE_SEQ
    # Bound each response as well; the browser will immediately ask for newer rows.
    rows = rows[-800:]
    return jsonify(latest=latest, lines=[{"seq":a,"stream":b,"text":c} for a,b,c in rows])

@app.get("/out/<path:f>")
def out(f): return send_file(os.path.join(OUT,f))

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MissingLink MiniMax Studio</title>
<link rel="icon" href="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2">
<style>
*{box-sizing:border-box}
:root{--bg:#09090b;--panel:#101013;--panel2:#151519;--line:#25252b;--muted:#777982;--text:#ededf0;--accent:#E8A917}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
.wrap{height:100vh;display:grid;grid-template-columns:minmax(350px,390px) minmax(0,1fr)}
.side{border-right:1px solid var(--line);padding:16px;overflow:auto;background:#0c0c0e}
.main{padding:16px;overflow:auto;display:grid;grid-template-rows:auto auto minmax(320px,1fr) auto auto;gap:12px;min-width:0}
.brand{display:flex;align-items:center;gap:10px;margin:0 0 14px;text-decoration:none;color:inherit;font-size:13px;font-weight:800;letter-spacing:1.5px}
.brand .ml{color:#8a8a8f}.brand .st{color:var(--accent)}#logo{height:26px;width:auto}
.card,details{border:1px solid var(--line);border-radius:10px;background:var(--panel);margin-bottom:10px;overflow:hidden}
.cardbody{padding:12px}
.cardtitle,summary{padding:9px 12px;color:#a0a1a9;font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;cursor:default}
summary{cursor:pointer;user-select:none;border-bottom:0}
details[open] summary{border-bottom:1px solid var(--line)}
details>div{padding:12px}
label{display:block;font-size:10px;color:#8d8f98;margin:8px 0 4px}
input,textarea,select{width:100%;background:var(--panel2);border:1px solid #2c2c33;color:var(--text);padding:8px 9px;border-radius:7px;font:inherit;font-size:12px;outline:none}
input:focus,textarea:focus,select:focus{border-color:var(--accent)}
textarea{min-height:126px;resize:vertical;line-height:1.5}
input[type=file]{font-size:10.5px;padding:6px}
input[type=file]::file-selector-button{background:#29292f;color:#ccc;border:0;padding:5px 8px;border-radius:5px;margin-right:8px;font:inherit;cursor:pointer}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.hint{font-size:9.5px;color:#686a73;margin-top:5px;line-height:1.4}.hint b{color:#a6a7ae}
.switchrow{display:flex;gap:8px;align-items:center;margin-top:8px}.switchrow input{width:auto;accent-color:var(--accent)}.switchrow label{margin:0;font-size:10.5px;color:#aaa}
button{border:0;border-radius:7px;background:var(--accent);color:#111;padding:10px 11px;font:inherit;font-weight:800;cursor:pointer}
button:disabled{background:#29292f;color:#666;cursor:not-allowed}.inlinebtn{background:#29292f;color:#ccc;padding:8px 9px;width:100%;margin-top:8px;font-size:10.5px}
#go{width:100%;font-size:13px;margin-top:4px;padding:12px}
.previewgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.preview{border:1px solid var(--line);border-radius:7px;overflow:hidden;background:#09090b}.preview img{width:100%;height:84px;object-fit:contain;display:none;background:#050506}.pcap{padding:5px 6px;color:#6f7078;font-size:8.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#status{padding:10px 12px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}.row{display:flex;align-items:center;gap:8px;font-size:12px}.dot{width:7px;height:7px;border-radius:50%;background:#5fd68a}.dot.live{background:var(--accent);animation:p 1.3s infinite}.dot.err{background:#ff6b6b}@keyframes p{50%{opacity:.25}}.bar{height:3px;background:#1f2024;border-radius:2px;margin-top:7px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .25s}
.gpugrid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px}.gpucard{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:7px 9px;min-width:0}.gpuk{font-size:8px;color:#656771;letter-spacing:.6px;text-transform:uppercase}.gpuv{font-size:12px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.gpuv.busy{color:var(--accent)}.gpuv.hot{color:#ffb36b}
#vwrap{border:1px solid var(--line);border-radius:10px;background:#070708;display:flex;align-items:center;justify-content:center;min-height:320px;overflow:hidden}#empty{color:#44464f;font-size:11px}video{width:100%;height:100%;max-height:60vh;object-fit:contain;background:#000}
*{scrollbar-width:none}*::-webkit-scrollbar{display:none;width:0;height:0}
.consolebox{border:1px solid var(--line);border-radius:9px;background:#060607;overflow:hidden}.consolehead{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:7px 9px;border-bottom:1px solid var(--line);color:#81838c;font-size:9.5px}.consoleactions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.consolehead button{width:auto;margin:0;background:#29292f;color:#bbb;padding:5px 9px;font-size:9px;border-radius:5px}.consolebox pre{margin:0;padding:9px 10px;height:160px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#c8c9ce;font:9.5px/1.4 ui-monospace,Menlo,monospace}.consolebox.collapsed pre{display:none}.consolebox.collapsed .consolehead{border-bottom:0}
#err{display:none;white-space:pre-wrap;color:#ff8a8a;font-size:10px;max-height:180px;overflow:auto;border:1px solid #3a2020;background:#160e0e;padding:10px;border-radius:8px}
.footerlink{display:block;text-align:center;color:#62646d;text-decoration:none;font-size:9px;margin:6px 0 2px}.footerlink:hover{color:var(--accent)}
@media(max-width:1250px){.gpugrid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:850px){body{overflow:auto}.wrap{height:auto;grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line);max-height:none}.main{min-height:900px}.gpugrid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class=wrap>
<div class=side>
<a class=brand href="https://missinglink.build" target="_blank" rel="noopener"><img id=logo src="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2" alt="" onerror="this.style.display='none'"><span><span class=ml>MISSINGLINK</span> <span class=st>MINIMAX H3</span></span></a>

<div class=card><div class=cardtitle>Prompt</div><div class=cardbody>
<textarea id=prompt placeholder="Describe the shot, motion, camera, environment, soundscape and music..."></textarea>
</div></div>

<div class=card><div class=cardtitle>Input + Output</div><div class=cardbody>
<div class=g2><div><label>First frame</label><input type=file id=first_frame accept="image/*"></div><div><label>Last frame</label><input type=file id=last_frame accept="image/*"></div></div>
<div class=previewgrid><div class=preview><img id=first_preview><div class=pcap id=first_cap>first frame: none</div></div><div class=preview><img id=last_preview><div class=pcap id=last_cap>last frame: none</div></div></div>
<div class=g2><div><label>Width</label><input id=width type=number value=1344 step=32 min=32 autocomplete=off></div><div><label>Height</label><input id=height type=number value=768 step=32 min=32 autocomplete=off></div></div>
<div class=g2><div><label>Duration</label><input id=duration type=number value=20 step=0.1 min=0.21 max=149.7 autocomplete=off></div><div><label>Playback</label><input id=playback_speed type=number value=1.0 step=0.05 min=0.05 max=8 autocomplete=off></div></div>
<input id=length_mode type=hidden value=seconds><input id=frames type=hidden value=481>
<div class=hint id=durhint></div>
<div class=switchrow><input id=auto_aspect type=checkbox checked><label for=auto_aspect>fit canvas to first-frame aspect</label></div>
<div class=g2><div><label>Short edge</label><input id=short_edge type=number value=768 step=32 min=256></div><div><label>Image fit</label><select id=image_fit><option value=cover selected>cover / crop</option><option value=contain>contain</option><option value=stretch>stretch</option></select></div></div>
<button id=fitframe class=inlinebtn>FIT TO FIRST FRAME</button>
</div></div>

<div class=card><div class=cardtitle>Preset</div><div class=cardbody>
<div class=g2><button id=lightpreset class=inlinebtn>FAST · 4 STEP</button><button id=nsfw class=inlinebtn>QUALITY · 30 STEP</button></div>
<div class=g2><button id=actionpreset class=inlinebtn>ACTION LORA</button><button id=fastmotion class=inlinebtn>FAST MOTION</button></div>
<div class=hint id=modelhint>Model mode will be shown here after startup.</div>
</div></div>

<details><summary>Sampling</summary><div>
<div class=g2><div><label>Steps</label><input id=steps type=number value=4 min=1></div><div><label>Denoise</label><input id=denoise type=number value=1 step=0.01 min=0 max=1></div></div>
<div class=g2><div><label>Sampler</label><select id=sampler_name></select></div><div><label>Scheduler</label><select id=scheduler></select></div></div>
<label>Seed</label><div style="display:grid;grid-template-columns:1fr 64px;gap:8px"><input id=seed type=number value=42><button id=rnd class=inlinebtn style="margin:0">RAND</button></div>
<div class=g2><div><label>Video shift</label><input id=shift_video type=number value=6 step=.01></div><div><label>Audio shift</label><input id=shift_audio type=number value=3 step=.01></div></div>
</div></details>

<details><summary>Model + LoRAs</summary><div>
<label>Transformer</label><select id=unet></select>
<div class=g2><div><label>Weight dtype</label><select id=weight_dtype><option>default</option><option>fp8_e4m3fn</option><option>fp8_e4m3fn_fast</option><option>fp8_e5m2</option></select></div><div><label>Primary strength</label><input id=lora_strength type=number value=1 step=.01></div></div>
<label>Primary LoRA</label><div style="display:grid;grid-template-columns:1fr 52px;gap:8px"><select id=lora></select><button id=refresh class=inlinebtn style="margin:0">↻</button></div>
<div class=switchrow><input id=lightning type=checkbox><label for=lightning>LightX2V Lightning</label></div><label>Lightning strength</label><input id=lightning_strength type=number value=1 step=.05 min=0 max=2>
<div class=switchrow><input id=action type=checkbox><label for=action>H3 action LoRA</label></div><label>Action strength</label><input id=action_strength type=number value=1 step=.05 min=0 max=2><div class=hint id=action_hint></div>
</div></details>

<button id=go>GENERATE 20s VIDEO</button>
<a class=footerlink href="https://missinglink.build/studio" target="_blank" rel="noopener">missinglink.build/studio</a>
</div>

<div class=main>
<div id=status><div class=row><span class=dot></span><span id=stxt>ready</span></div><div class=bar><i id=pb></i></div></div>
<div class=gpugrid>
<div class=gpucard><div class=gpuk>GPU</div><div class=gpuv id=gpu_util>--</div></div><div class=gpucard><div class=gpuk>VRAM</div><div class=gpuv id=gpu_vram>--</div></div><div class=gpucard><div class=gpuk>Mem</div><div class=gpuv id=gpu_memutil>--</div></div><div class=gpucard><div class=gpuk>Temp</div><div class=gpuv id=gpu_temp>--</div></div><div class=gpucard><div class=gpuk>Power</div><div class=gpuv id=gpu_power>--</div></div><div class=gpucard><div class=gpuk>Clock</div><div class=gpuv id=gpu_clock>--</div></div>
</div>
<div id=vwrap><div id=empty>generated video appears here</div></div>
<div id=err></div>
<div class=consolebox id=consolebox><div class=consolehead><span>LIVE CONSOLE</span><div class=consoleactions><button id=copyconsole type=button>COPY</button><button id=clearconsole type=button>CLEAR</button><button id=minconsole type=button>MINIMIZE</button></div></div><pre id=console></pre></div>
<div id=meta class=hint></div>
</div></div>
<script>
const $=i=>document.getElementById(i);
let job=null;
$('rnd').onclick=e=>{e.preventDefault();$('seed').value=Math.floor(Math.random()*1e9)};

const dot=k=>document.querySelector('.dot').className='dot '+(k||'');
const say=t=>$('stxt').textContent=t;

// Live raw stdout/stderr from the Python/ComfyUI process. This is global because
// the studio only allows one GPU generation at a time.
let consoleSeq=0;
let consolePaused=false;
$('clearconsole').onclick=e=>{e.preventDefault();$('console').textContent='';};
$('copyconsole').onclick=async e=>{
  e.preventDefault();
  const text=$('console').textContent||'';
  const btn=$('copyconsole');
  try{
    if(navigator.clipboard&&window.isSecureContext){
      await navigator.clipboard.writeText(text);
    }else{
      const ta=document.createElement('textarea'); ta.value=text; ta.style.position='absolute'; ta.style.left='-9999px';
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
    const old=btn.textContent; btn.textContent='COPIED'; setTimeout(()=>btn.textContent=old,900);
  }catch(err){
    const old=btn.textContent; btn.textContent='COPY FAILED'; setTimeout(()=>btn.textContent=old,1200);
  }
};
$('minconsole').onclick=e=>{
  e.preventDefault();
  const box=$('consolebox');
  const collapsed=box.classList.toggle('collapsed');
  $('minconsole').textContent=collapsed?'EXPAND':'MINIMIZE';
};
async function pollConsole(){
  try{
    const r=await fetch('/api/console?since='+consoleSeq,{cache:'no-store'});
    const d=await r.json();
    if(Array.isArray(d.lines)&&d.lines.length){
      const box=$('console');
      const nearBottom=(box.scrollHeight-box.scrollTop-box.clientHeight)<50;
      for(const row of d.lines){
        const text=(row.stream==='stderr'?'[stderr] ':'')+row.text+'\n';
        box.appendChild(document.createTextNode(text));
        consoleSeq=Math.max(consoleSeq,Number(row.seq||0));
      }
      // Avoid unbounded browser DOM growth while retaining a useful raw tail.
      if(box.textContent.length>240000) box.textContent=box.textContent.slice(-180000);
      if(nearBottom) box.scrollTop=box.scrollHeight;
    }else if(Number(d.latest||0)>consoleSeq){
      consoleSeq=Number(d.latest||0);
    }
  }catch(e){}
  setTimeout(pollConsole,700);
}
pollConsole();

// Continuous GPU telemetry. This is independent of job polling, so it remains
// visible during model load/unload, VAE decode, muxing, and after an error.
async function pollGPU(){
  try{
    const r=await fetch('/api/gpu',{cache:'no-store'});
    const d=await r.json();
    if(d.ok){
      const util=Number(d.util||0), mu=Number(d.mem_util||0);
      const used=Number(d.mem_used_mb||0)/1024, total=Number(d.mem_total_mb||0)/1024;
      const temp=Number(d.temp_c||0), pw=Number(d.power_w||0), pl=Number(d.power_limit_w||0);
      $('gpu_util').textContent=util.toFixed(0)+'%';
      $('gpu_vram').textContent=used.toFixed(1)+' / '+total.toFixed(1)+' GiB';
      $('gpu_memutil').textContent=mu.toFixed(0)+'%';
      $('gpu_temp').textContent=temp.toFixed(0)+' °C';
      $('gpu_power').textContent=pw.toFixed(0)+' / '+pl.toFixed(0)+' W';
      $('gpu_clock').textContent=Number(d.clock_mhz||0).toFixed(0)+' MHz';
      $('gpu_util').className='gpuv '+(util>=70?'busy':'');
      $('gpu_temp').className='gpuv '+(temp>=80?'hot':'');
    }else{
      $('gpu_util').textContent='nvidia-smi unavailable';
    }
  }catch(e){}
  setTimeout(pollGPU,1000);
}
pollGPU();

// H3 uses a fixed 24-fps latent clock and legal frame lengths 17*n+5.
function snapFrames(v){
  v=Number(v||5);
  let k=Math.round((v-5)/17);
  return Math.max(5,Math.min(3592,17*Math.max(0,k)+5));
}
function updateDur(){
  const mode=$('length_mode').value;
  const speed=Math.max(.05,Number($('playback_speed').value||1));
  let f;
  if(mode==='frames'){
    f=snapFrames(Number($('frames').value||124));
  }else{
    const finalSec=Math.max(.01,Number($('duration').value||20));
    f=snapFrames(finalSec*speed*24);
  }
  const modelSec=f/24, finalSec=modelSec/speed;
  const trained=f>=124&&f<=362;
  $('durhint').innerHTML=`→ H3 model: <b>${f} frames / ${modelSec.toFixed(2)}s</b> · saved MP4 ≈ <b>${finalSec.toFixed(2)}s</b> at ${speed.toFixed(2)}×`+
    (trained?'':`<br><span style="color:#c9a227">outside the best-tested 124–362 frame range (≈5.2–15.1s model time); H3 accepts it, but quality/memory are less predictable.</span>`);
}
$('duration').addEventListener('input',()=>{$('length_mode').value='seconds';updateDur();});
$('frames').addEventListener('input',()=>{$('length_mode').value='frames';updateDur();});
$('playback_speed').addEventListener('input',updateDur);
$('length_mode').addEventListener('change',updateDur);
updateDur();

let firstImageDims=null;
function snap32(v){return Math.max(32,Math.round(v/32)*32)}
function fitCanvas(){
  if(!firstImageDims){say('choose a first frame first');return}
  const short=Math.max(256,snap32(Number($('short_edge').value||768)));
  const ar=firstImageDims.w/firstImageDims.h;
  let w,h;
  if(ar>=1){h=short;w=Math.max(32,Math.floor((short*ar)/32)*32)}
  else{w=short;h=Math.max(32,Math.floor((short/ar)/32)*32)}
  $('width').value=w;$('height').value=h;
  say(`canvas fitted to ${w}×${h}`);
}
function bindPreview(fileInput,imgId,capId,isFirst){
  $(fileInput).addEventListener('change',()=>{
    const f=$(fileInput).files[0], img=$(imgId), cap=$(capId);
    if(!f){img.style.display='none';img.removeAttribute('src');cap.textContent=(isFirst?'first':'last')+' frame: none';if(isFirst)firstImageDims=null;return}
    const u=URL.createObjectURL(f);
    img.onload=()=>{
      cap.textContent=`${f.name} · ${img.naturalWidth}×${img.naturalHeight}`;
      if(isFirst){firstImageDims={w:img.naturalWidth,h:img.naturalHeight};if($('auto_aspect').checked)fitCanvas()}
      URL.revokeObjectURL(u);
    };
    img.src=u;img.style.display='block';
  });
}
bindPreview('first_frame','first_preview','first_cap',true);
bindPreview('last_frame','last_preview','last_cap',false);
$('fitframe').onclick=e=>{e.preventDefault();fitCanvas()};

let firstMeta=true;
function loadMeta(){
  return fetch('/api/meta').then(r=>r.json()).then(m=>{
    const keep=$('lora').value;
    $('sampler_name').innerHTML=m.samplers.map(s=>
      `<option${s==='euler'?' selected':''}>${s}</option>`).join('');
    $('scheduler').innerHTML=m.schedulers.map(s=>
      `<option${s==='simple'?' selected':''}>${s}</option>`).join('');
    $('lora').innerHTML=m.loras.map(s=>`<option>${s}</option>`).join('');
    if(keep&&m.loras.includes(keep)) $('lora').value=keep;
    else if(m.lora_default&&m.loras.includes(m.lora_default)) $('lora').value=m.lora_default;
    const ku=$('unet').value;
    $('unet').innerHTML=(m.unets||[]).map(s=>
      `<option${s===m.unet_default?' selected':''}>${s}</option>`).join('');
    if(ku&&(m.unets||[]).includes(ku))$('unet').value=ku;
    if(firstMeta){
      // Force the requested startup profile instead of accepting browser-restored
      // form values from an older 5-second session.
      $('length_mode').value='seconds';
      $('duration').value=20;
      $('frames').value=481;
      $('playback_speed').value=1.0;
      $('lora_strength').value=m.lora_strength_default||1.0;
      $('action_strength').value=m.action_strength_default||1.0;
      $('action').disabled=!m.action_available;
      if(!m.action_available) $('action').checked=false;
      const words=(m.action_trained_words||[]).join(', ');
      $('action_hint').innerHTML=m.action_available
        ? `<b>${m.action_version_name||'H3 version'}</b> · ${m.action_base_model||'MiniMax H3'} · file <code>${m.action_file||''}</code>`+
          (words?`<br>trained words: <code>${words}</code>`:'<br>no trainedWords published by the API')+
          `<br>linked v${m.action_linked_version} is WAN, so it is intentionally not loaded.`
        : `No MiniMax-H3-compatible version of CivitAI model ${m.action_model_id} was resolved. The linked v${m.action_linked_version} is WAN and cannot patch H3.`;
      $('lightning_strength').value=m.lightning_strength_default||1.0;
      $('lightning').disabled=!m.lightning_available;
      if(m.using_redmix){
        $('lora').value='none';
        $('lightning').checked=false;
        $('action').checked=false;
        $('steps').value=m.recommended_steps||6;
        if([...$('sampler_name').options].some(x=>x.value===(m.recommended_sampler||'er_sde'))) $('sampler_name').value=m.recommended_sampler||'er_sde';
        if([...$('scheduler').options].some(x=>x.value===(m.recommended_scheduler||'beta'))) $('scheduler').value=m.recommended_scheduler||'beta';
        $('denoise').value=1.0;
        $('playback_speed').value=1.0;
        updateDur();
        say('ready · REAL REDMIX Beta2 · 6-step H3 profile');
      }else if(m.lightning_available && m.lightning_default){
        $('lightning').checked=true;
        $('steps').value=4;
        $('sampler_name').value='euler';
        $('scheduler').value='simple';
        $('shift_video').value=6;
        $('shift_audio').value=3;
        $('denoise').value=1.0;
        $('playback_speed').value=1.0;
        updateDur();
        say('⚠ REDMIX unavailable · PUBLIC NSFW FALLBACK · NaughtyTimes + LightX2V 4-step');
      }else{
        $('lightning').checked=false;
      }
      firstMeta=false;
    }
    window.H3META=m;
    const mh=$('modelhint');
    if(mh){
      mh.innerHTML = m.using_redmix
        ? '<b>REAL REDMIX Beta2 ACTIVE</b> · integrated tuning; extra NaughtyTimes/Lightning are off by default.'
        : '<b>⚠ PUBLIC NSFW FALLBACK ACTIVE</b> · REDMIX was inaccessible, so this run uses stock H3 INT8 + NaughtyTimes v2 + LightX2V.';
    }
    return m;
  });
}
loadMeta();
$('refresh').onclick=e=>{e.preventDefault();loadMeta().then(m=>
  say(`${m.loras.length-1} lora(s) found`))};

$('nsfw').onclick=e=>{
  e.preventDefault();
  const m=window.H3META||{};
  if(m.unet_default && [...$('unet').options].some(x=>x.value===m.unet_default)) $('unet').value=m.unet_default;
  if(m.lora_default && [...$('lora').options].some(x=>x.value===m.lora_default)) $('lora').value=m.lora_default;
  $('lora_strength').value=m.lora_strength_default||1.0;
  $('lightning').checked=false;$('action').checked=false;
  $('steps').value=30;$('denoise').value=1.0;$('shift_video').value=12;$('shift_audio').value=3;
  $('playback_speed').value=1.0;
  updateDur();
  say('quality / NaughtyTimes preset applied');
};
$('lightpreset').onclick=e=>{
  e.preventDefault();
  const m=window.H3META||{};
  if(!m.lightning_available){alert('Lightning LoRA is not available. Check the startup log or place '+(m.lightning_file||'the Lightning file')+' in models/loras.');return}
  if(m.unet_default && [...$('unet').options].some(x=>x.value===m.unet_default)) $('unet').value=m.unet_default;
  if(m.lora_default && [...$('lora').options].some(x=>x.value===m.lora_default)) $('lora').value=m.lora_default;
  $('lora_strength').value=m.lora_strength_default||1.0;
  $('action').checked=false;
  $('lightning').checked=true;$('lightning_strength').value=m.lightning_strength_default||1.0;
  $('steps').value=4;$('sampler_name').value='euler';$('scheduler').value='simple';
  $('shift_video').value=6;$('shift_audio').value=3;$('denoise').value=1.0;
  $('playback_speed').value=1.0;
  updateDur();
  say('LightX2V 4-step preset applied');
};
$('actionpreset').onclick=e=>{
  e.preventDefault();
  const m=window.H3META||{};
  if(!m.action_available){alert('No H3-compatible action LoRA was resolved. Check the startup log.');return}
  if(m.unet_default && [...$('unet').options].some(x=>x.value===m.unet_default)) $('unet').value=m.unet_default;
  if(m.lora_default && [...$('lora').options].some(x=>x.value===m.lora_default)) $('lora').value=m.lora_default;
  $('lora_strength').value=m.lora_strength_default||1.0;
  $('action').checked=true;$('action_strength').value=m.action_strength_default||1.0;
  $('lightning').checked=false;
  $('steps').value=30;$('denoise').value=1.0;$('shift_video').value=12;$('shift_audio').value=3;
  $('playback_speed').value=1.0;
  updateDur();
  say('H3 action LoRA preset applied — use a first frame (I2V)');
};
$('fastmotion').onclick=e=>{
  e.preventDefault();
  const m=window.H3META||{};
  if(m.unet_default && [...$('unet').options].some(x=>x.value===m.unet_default)) $('unet').value=m.unet_default;
  if(m.lora_default && [...$('lora').options].some(x=>x.value===m.lora_default)) $('lora').value=m.lora_default;
  $('lora_strength').value=m.lora_strength_default||1.0;
  if(m.lightning_available){$('lightning').checked=true;$('lightning_strength').value=m.lightning_strength_default||1.0;}
  $('steps').value=4;$('sampler_name').value='euler';$('scheduler').value='simple';
  $('shift_video').value=6;$('shift_audio').value=3;$('denoise').value=1.0;
  $('playback_speed').value=1.75;
  updateDur();
  say('Fast Motion preset applied — requested length preserved');
};


function fail(m){dot('err');say('failed');$('err').style.display='block';
  $('err').textContent=m;$('go').disabled=false;$('pb').style.width='0'}

$('go').onclick=async()=>{
  $('err').style.display='none';
  const fd=new FormData();
  for(const k of ['prompt','width','height','duration','frames','length_mode',
    'playback_speed','image_fit','steps','seed','denoise','shift_video','shift_audio',
    'sampler_name','scheduler','weight_dtype','lora','lora_strength',
    'action_strength','lightning_strength','unet'])fd.append(k,$(k).value);
  fd.append('action',$('action').checked?'1':'0');
  fd.append('lightning',$('lightning').checked?'1':'0');
  for(const k of ['first_frame','last_frame'])
    if($(k).files[0])fd.append(k,$(k).files[0]);
  $('go').disabled=true;dot('live');say('submitting');
  const r=await(await fetch('/api/generate',{method:'POST',body:fd})).json();
  if(r.error){fail(r.error);return}
  job=r.id;poll();
};

async function poll(){
  const j=await(await fetch('/api/job/'+job)).json();
  if(j.status==='running'||j.status==='queued'){
    dot('live');
    const pct=j.total?Math.round(100*j.cur/j.total):0;
    say(`${j.stage||j.status}${j.total?` · ${j.cur}/${j.total}`:''} · ${j.el}s`);
    $('pb').style.width=pct+'%';setTimeout(poll,1500);return}
  if(j.status==='done'){
    dot('');say(`done in ${j.secs}s · output ${j.duration}s · model ${j.model_duration||j.duration}s / ${j.frames} frames · ${j.playback_speed||1}×`+
      (j.vram_free!==undefined?` · ${j.vram_free} GB free at decode`:''));
    $('pb').style.width='100%';
    $('vwrap').innerHTML=`<video controls autoplay src="/out/${j.file}"></video>`;
    $('meta').textContent=j.file+(j.lora_info?` · ${j.lora_info}`:'')+(j.note?` · ${j.note}`:'');
    $('go').disabled=false;return}
  fail(j.msg||'unknown error');
}
</script></body></html>"""

@app.get("/")
def index(): return Response(PAGE, mimetype="text/html")

# ── 7. Serve + launch ──────────────────────────────────────────────────────
with socket.socket() as s:
    try: s.bind(("0.0.0.0",UI_PORT))
    except OSError: s.bind(("0.0.0.0",0)); UI_PORT=s.getsockname()[1]

threading.Thread(target=lambda: app.run(host="0.0.0.0",port=UI_PORT,
                 threaded=True,use_reloader=False), daemon=True).start()
time.sleep(2)

import requests
IN_COLAB = "google.colab" in sys.modules
url = mode = None

if IN_COLAB:
    from google.colab import output as _co
    from IPython.display import display, HTML as _H

    # Colab's own transport first. This is reliable now that the page is our
    # Flask app — the origin-header middleware that broke it was ComfyUI's.
    # The iframe gives you a working UI immediately; the window link opens a
    # real tab. Both are shown so neither is a single point of failure.
    try:
        _co.serve_kernel_port_as_iframe(UI_PORT, height="900")
        mode = "iframe"
    except Exception as e:
        log(f"  iframe failed: {e}")
    try:
        _co.serve_kernel_port_as_window(
            UI_PORT, anchor_text="◤ Open MissingLink MiniMax Studio in a new tab")
        mode = (mode or "") + "+window"
    except Exception as e:
        log(f"  window failed: {e}")

    if not mode and TUNNEL_FALLBACK:
        BIN="/usr/local/bin/cloudflared"
        if not os.path.exists(BIN):
            urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/"
                "releases/latest/download/cloudflared-linux-amd64",BIN)
            os.chmod(BIN, os.stat(BIN).st_mode|stat.S_IEXEC)
        tun=subprocess.Popen([BIN,"tunnel","--url",f"http://127.0.0.1:{UI_PORT}",
            "--no-autoupdate"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1)
        t0=time.time()
        for line in tun.stdout:
            m=re.search(r"https://[-a-z0-9]+\.trycloudflare\.com",line)
            if m: url,mode=m.group(0),"tunnel"; break
            if time.time()-t0>90: break
        threading.Thread(target=lambda:[None for _ in tun.stdout],daemon=True).start()
        if url:
            display(_H(f'<a href="{url}" target="_blank" style="color:#E8A917;'
                       f'font-size:18px;font-weight:bold">{url}</a>'))

    log(f"🚀 mode: {mode or 'none'}  ·  port {UI_PORT}")
else:
    log(f"\n  http://localhost:{UI_PORT}\n")

log("="*74)
if STARTUP_GPU_PRELOADED:
    if USING_REDMIX:
        log(f"  ✓ REAL REDMIX H3 A2A Beta2 is preloaded; conditioning TE: {TEXT_ENCODER_FILE}.")
    else:
        log(f"  ⚠ PUBLIC NSFW FALLBACK is preloaded: base H3 INT8 + NaughtyTimes v2 + LightX2V; conditioning TE: {TEXT_ENCODER_FILE}.")
else:
    log("  Startup preload failed; first GENERATE will retry model loading.")
log("  Errors come back as a full traceback in the red panel.")
log("="*74)
