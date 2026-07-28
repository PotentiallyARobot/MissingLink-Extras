# ============================================================
#  VIDEO STUDIO  —  single-cell Colab launcher
#  image-to-video  +  runtime LoRA support
#  UI: MissingLink Video Studio (branded)
#
#  Model: Wan 2.2, Wan 2.1 and LTX 2.3 + Audio  (open-source video foundation model)
#
#  HOW TO USE:
#    1. Runtime -> Change runtime type -> GPU.  An A100 (or L4) is
#       strongly recommended — the Wan 2.1 i2v model is 14B.
#    2. MEMBERSHIP: this studio unlocks with a MissingLink API key +
#       ACTIVE membership (sign up at https://www.missinglink.build/).
#       Provide it via the MISSINGLINK_API_KEY Colab secret, the
#       MISSINGLINK_API_KEY_MANUAL line below, or the in-app login
#       screen. Civitai + Auto Prompt then run on MissingLink's keys.
#       (Optional secret: HF_TOKEN for gated HuggingFace repos.)
#    3. Paste this whole cell into Colab and run it.
#    4. Click the link it prints to open the studio UI.
#    5. Upload a starting frame, then Generate.
#    6. Add LoRAs at runtime: paste a HuggingFace or Civitai
#       .safetensors download URL into the "Add LoRA" box.
#
#  GPU RESIDENCY:
#    The pipeline is placed FULLY on the GPU when there is enough
#    VRAM for the chosen quantization (with the default GGUF Q4_K_M
#    that means ~21 GB+, i.e. an L4 / A100). On smaller cards it
#    automatically falls back to model CPU offload so it still runs,
#    just slower. See build_pipeline().
#
#  NOTE ON MODEL CHOICE:
#    Wan 2.1 image-to-video ships only as 14B repos:
#      Wan-AI/Wan2.1-I2V-14B-480P-Diffusers   (use 480P resolution)
#      Wan-AI/Wan2.1-I2V-14B-720P-Diffusers   (use 720P resolution)
#    Pick the resolution profile in the UI that matches the model.
#    With QUANTIZATION="gguf" the ~28 GB bf16 transformer inside those
#    repos is REPLACED at load time by city96's ~11.3 GB Q4_K_M GGUF
#    single-file quant (loaded via WanTransformer3DModel.from_single_file).
# ============================================================

import os, sys, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Early launch fetch of the LTX-2.3 weights ────────────────────────────
# The big model downloads start HERE — before the ~3 min of pip installs
# below — so the checkpoint is pulling from second zero. aria2c opens 16
# parallel connections (single-stream HTTP tops out far below what the
# runtime's NIC can do; this is usually a 3-8x speedup). _ltx_download()
# later waits on these processes, verifies sizes, and resumes anything
# unfinished, so this is purely a head start — never a second copy.
# NOTE: keep the paths/URLs in sync with the LTX section further down.
LTX_EARLY_FETCH = {}
try:
    import shutil as _esh
    import urllib.request as _eurl
    from pathlib import Path as _EPath
    _e_models = "/content/ltx/models"
    _e_files = [
        ("https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/"
         "main/ltx-2.3-22b-distilled-fp8.safetensors",
         f"{_e_models}/ltx-2.3-22b-distilled-fp8.safetensors"),
        ("https://huggingface.co/Lightricks/LTX-2.3/resolve/"
         "main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
         f"{_e_models}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    ]
    if _esh.which("aria2c") is None:
        subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"],
                       capture_output=True)
    if _esh.which("aria2c"):
        _EPath(_e_models).mkdir(parents=True, exist_ok=True)
        _etok = os.environ.get("HF_TOKEN", "").strip()
        for _u, _d in _e_files:
            _dp = _EPath(_d)
            _have = _dp.stat().st_size if _dp.exists() else 0
            _tot = 0
            try:
                _rq = _eurl.Request(_u, method="HEAD")
                if _etok:
                    _rq.add_header("Authorization", f"Bearer {_etok}")
                with _eurl.urlopen(_rq, timeout=20) as _rr:
                    _tot = int(_rr.headers.get("Content-Length") or 0)
            except Exception:
                pass
            if _have and _tot and _have >= _tot:
                continue   # finished on a previous run
            _cmd = ["aria2c", "-c", "-x16", "-s16", "-k1M",
                    "--file-allocation=none", "--console-log-level=warn",
                    "--summary-interval=0", "-d", _e_models,
                    "-o", _dp.name]
            if _etok:
                _cmd += ["--header", f"Authorization: Bearer {_etok}"]
            LTX_EARLY_FETCH[_d] = subprocess.Popen(
                _cmd + [_u],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if LTX_EARLY_FETCH:
            print(f"LTX-2.3 weights: {len(LTX_EARLY_FETCH)} download(s) "
                  "running in the background (aria2c, 16 parallel "
                  "connections) while dependencies install...")
except Exception as _efe:
    print(f"  (early LTX fetch skipped: {_efe} — the normal preload "
          "will download instead.)")

print("Installing dependencies (~3 min first run)...")
# Support libraries first. Wan needs ftfy (text cleanup) and the UMT5
# text encoder + CLIP vision encoder from transformers. "gguf" is the
# parser diffusers uses to read city96's Q4_K_M single-file quants.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "transformers", "accelerate", "safetensors", "sentencepiece",
                "ftfy", "peft", "imageio", "imageio-ffmpeg", "flask",
                "bitsandbytes", "torchao", "gguf"],
               check=True)
# diffusers: clean reinstall from a single git commit so the Wan
# pipeline + transformer + LoRA-conversion code all come from the same
# commit. A partial --upgrade can mix modules and break Wan LoRA loading.
#
# IMPORTANT: if you previously ran an older diffusers in this runtime,
# RESTART the runtime (Runtime -> Restart) after the first run of this
# cell — Python caches already-imported modules, so a reinstall alone
# does not swap out classes that are already loaded.
print("Force-reinstalling diffusers from git (clean, single commit)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "--force-reinstall", "--no-deps",
                "git+https://github.com/huggingface/diffusers.git"],
               check=True)

import io, time, json, uuid, base64, tempfile, threading, traceback
import collections, re, math, select
import socket as _socket
import requests as _requests
import torch
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, Response

# ---- GPU sanity check --------------------------------------------------
def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip()
    except Exception:
        return ""

_gpu = _gpu_name()
if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda") + 1
        print(f"  CUDA OK — {_gpu}, torch {torch.__version__}")
    except Exception as e:
        print(f"  WARNING: CUDA kernel launch failed ({e}).")
else:
    print("  WARNING: no CUDA GPU — set Runtime -> GPU.")

IN_COLAB = False
try:
    from google.colab.output import eval_js
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    eval_js = None
    userdata = None

def _get_secret(key, verbose=True):
    """Read a Colab secret, reporting clearly *why* it is unavailable.

    The common failure is not a missing key but the per-secret
    'Notebook access' toggle being off — userdata.get() raises in that
    case, which earlier silently looked identical to 'no key set'."""
    if userdata is None:
        return None
    try:
        val = userdata.get(key)
        if verbose:
            n = len(val) if val else 0
            print(f"  secret {key}: loaded ({n} chars).")
        return val
    except Exception as e:
        name = type(e).__name__
        if verbose:
            if "NotebookAccess" in name:
                print(f"  secret {key}: EXISTS but Notebook access is "
                      "OFF — open the Secrets panel and enable the "
                      "toggle next to it.")
            elif "SecretNotFound" in name:
                print(f"  secret {key}: not set — add it in the Secrets "
                      "panel (name must match exactly).")
            else:
                print(f"  secret {key}: unavailable ({name}: {e})")
        return None

# ---- MissingLink membership ---------------------------------------------
# This studio is a MissingLink member tool. It unlocks with your
# MissingLink API key + an ACTIVE membership — and it then uses
# MissingLink's own service keys (Civitai search/downloads, ✨ Auto
# Prompt vision) through missinglink.build, so you never paste those.
#
# Provide your MissingLink key one of three ways:
#   1. paste it into MISSINGLINK_API_KEY_MANUAL below, or
#   2. add a Colab secret named MISSINGLINK_API_KEY (enable Notebook
#      access), or
#   3. just run the cell and sign in on the login screen in the UI.
# No membership yet? Sign up at https://www.missinglink.build/
MISSINGLINK_API = "https://missinglink.build"
MISSINGLINK_SIGNUP_URL = "https://www.missinglink.build/"
# Opened in a NEW TAB by the login screen — the user signs in with Google
# there and copies the notebook session token back into the cell. (OAuth
# can't redirect inside the Colab output iframe, so this is the standard
# device-style hand-off.)
MISSINGLINK_LOGIN_URL = "https://missinglink.build/notebook-signin"
MISSINGLINK_API_KEY_MANUAL = ""   # <-- paste your MissingLink API key here

# HuggingFace token stays optional + personal (only needed for gated HF
# repos and LoRAs) — paste it or use the HF_TOKEN Colab secret.
HF_TOKEN_MANUAL = ""              # <-- paste your HuggingFace token here

print("Resolving keys...")
if HF_TOKEN_MANUAL.strip():
    HF_TOKEN = HF_TOKEN_MANUAL.strip()
    print(f"  HF_TOKEN: using token pasted in the cell "
          f"({len(HF_TOKEN)} chars).")
else:
    HF_TOKEN = _get_secret("HF_TOKEN")

# Optional user-supplied OpenAI key. If set (env var or Colab secret
# named OPENAI_API_KEY), image generation runs DIRECTLY on the user's own
# key — it does NOT go through MissingLink's server, costs no MissingLink
# tokens, and (since it isn't our server calling OpenAI) doesn't run our
# billing or moderation path. When absent, generation falls back to
# MissingLink's server (token-billed + screened).
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY")
                  or _get_secret("OPENAI_API_KEY", verbose=False) or "").strip()
if OPENAI_API_KEY:
    print(f"  OPENAI_API_KEY: found ({len(OPENAI_API_KEY)} chars) — image "
          "generation will use YOUR key directly (no MissingLink tokens).")

# Live auth/session state. The token comes from signing in with Google
# on missinglink.build (the login screen opens it in a new tab and gives
# you a short code / token to paste). Membership + free-trial balance are
# reported by the server; the studio unlocks when the user is signed in
# AND (a member OR still has free renders left).
FREE_RENDERS_HINT = 25   # display only; the server is authoritative
ML = {"key": None, "email": None, "member": False,
      "used": 0, "remaining": FREE_RENDERS_HINT, "free_limit": FREE_RENDERS_HINT,
      "tokens": None, "tokens_per_gen": 100,
      "authed": False, "reason": "no_session"}

# Session persistence: the ML dict resets on every CELL re-run, but the
# Colab VM (and /content) lives for the whole session. Cache the sign-in
# token on disk so one Google login survives cell re-runs and page
# reloads; it is removed only by explicit logout (or the VM ending).
_ML_SESSION_FILE = ("/content/.ml_session.json"
                    if os.path.isdir("/content")
                    else os.path.expanduser("~/.ml_session.json"))

def _ml_session_save():
    try:
        with open(_ML_SESSION_FILE, "w") as f:
            json.dump({"key": ML.get("key")}, f)
        os.chmod(_ML_SESSION_FILE, 0o600)
    except Exception:
        pass

def _ml_session_load():
    try:
        with open(_ML_SESSION_FILE) as f:
            k = (json.load(f) or {}).get("key")
        if k:
            ML["key"] = k          # validated lazily on the first status call
    except Exception:
        pass

def _ml_session_clear():
    ML.update(key=None, authed=False, member=False, email=None,
              reason="no_session")
    try:
        os.remove(_ML_SESSION_FILE)
    except Exception:
        pass

_ml_session_load()

def _ml_unlocked():
    """Studio is usable when signed in and either a member or with free
    renders left. The server enforces this too; this is the local mirror
    the request gate checks."""
    return ML.get("authed") and (
        ML.get("member") or ML.get("remaining", 0) > 0
        or ML.get("remaining", 0) == -1)

def _ml_validate(key):
    """Check a session token against missinglink.build /api/notebook/me.

    Returns (valid, data, error). The notebook accepts the Google
    sign-in session token as a Bearer credential."""
    try:
        r = _requests.get(f"{MISSINGLINK_API}/api/notebook/me",
                          headers={"Authorization": f"Bearer {key}"},
                          timeout=20)
        j = r.json()
    except Exception as e:
        return False, None, f"could not reach MissingLink — {e}"
    if not j.get("ok"):
        err = j.get("error") or "auth failed"
        return False, None, (
            "not signed in — sign in with Google"
            if err in ("login_required", "invalid_token") else err)
    return True, j, None

def _ml_apply(key, data):
    ML.update(
        key=key, authed=True,
        email=data.get("email"),
        member=bool(data.get("member")),
        used=int(data.get("used", 0) or 0),
        free_limit=int(data.get("free_limit", FREE_RENDERS_HINT) or FREE_RENDERS_HINT),
        remaining=(-1 if data.get("member")
                   else int(data.get("remaining", 0) or 0)),
        tokens=data.get("tokens"),
        tokens_per_gen=int(data.get("tokens_per_gen", 100) or 100),
        reason=("member" if data.get("member")
                else ("free_ok" if (data.get("remaining", 0) or 0) > 0
                      else "free_limit_reached")))

    _ml_session_save()

def _ml_login(key):
    """Validate + store a session token; return the auth state for the UI."""
    key = (key or "").strip()
    if not key:
        return {"authed": False, "reason": "no_session",
                "error": "sign in with Google to continue"}
    valid, data, err = _ml_validate(key)
    if not valid:
        return {"authed": False, "reason": "invalid", "error": err}
    _ml_apply(key, data)
    if ML["member"]:
        _log(f"  MissingLink: signed in as {ML['email'] or 'member'} — "
             "MEMBER (unlimited renders).")
    elif ML["remaining"] > 0:
        _log(f"  MissingLink: signed in as {ML['email'] or 'a user'} — "
             f"free trial: {ML['remaining']} of {ML['free_limit']} "
             "renders left.")
    else:
        _log(f"  MissingLink: {ML['email'] or 'a user'} has used all "
             f"{ML['free_limit']} free renders — sign up at "
             f"{MISSINGLINK_SIGNUP_URL}")
    return _ml_public()

def _ml_public():
    return {"authed": ML["authed"], "member": ML["member"],
            "email": ML["email"], "used": ML["used"],
            "remaining": ML["remaining"], "free_limit": ML["free_limit"],
            "tokens": ML.get("tokens"),
            "tokens_per_gen": ML.get("tokens_per_gen", 100),
            "reason": ML["reason"], "unlocked": _ml_unlocked(),
            "signup_url": MISSINGLINK_SIGNUP_URL,
            "login_url": MISSINGLINK_LOGIN_URL}

def _ml_refresh():
    """Re-fetch membership + free-trial balance for the stored token."""
    if not ML.get("key"):
        return _ml_public()
    valid, data, _ = _ml_validate(ML["key"])
    if valid:
        _ml_apply(ML["key"], data)
    return _ml_public()

_ml_boot_key = (MISSINGLINK_API_KEY_MANUAL.strip()
                or _get_secret("MISSINGLINK_API_KEY") or "")
if _ml_boot_key:
    print("  session token found — validating with MissingLink...")
    _ml_login(_ml_boot_key)
else:
    print("  not signed in — the studio UI will ask you to sign in with "
          "Google before anything runs.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Wan 2.1 model map (multiple tasks) --------------------------------
# Each generation MODE uses a different pipeline + repo:
#   i2v   -> WanImageToVideoPipeline  (animate one still; supports chaining)
#   flf2v -> WanImageToVideoPipeline  (first + last frame; 720P only)
#   vace  -> WanVACEPipeline          (reference-to-video + frame control)
MODE_REPOS = {
    "i2v": {
        "480P": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "720P": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    },
    "flf2v": {
        "720P": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    },
    "vace": {
        "1.3B": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "14B":  "Wan-AI/Wan2.1-VACE-14B-diffusers",
    },
}
# Back-compat alias used by older code paths / the startup default.
MODEL_OPTIONS = MODE_REPOS["i2v"]
MODEL_ID = MODEL_OPTIONS["480P"]   # default; switchable from the UI

# max_area pixel budgets per profile (the official Wan i2v values).
MAX_AREA = {"480P": 480 * 832, "720P": 720 * 1280}

# ---- quantization ------------------------------------------------------
# NOTE on the Civitai "fp8 / 8-bit pruned" Wan files: those are
# SINGLE-FILE checkpoints packaged for ComfyUI and do NOT load into the
# diffusers WanImageToVideoPipeline this studio uses. The diffusers-
# native ways to shrink the model are the load-time options below:
#   "none" - full bf16   (~42 GB of weights; wants an 80 GB card)
#   "gguf" - city96 GGUF Q4_K_M transformer  (~20 GB total)  <- current
#   "8bit" - bitsandbytes int8               (~24 GB total)
#   "fp8"  - torchao float8_weight_only      (~24 GB total; literal fp8)
#   "4bit" - bitsandbytes nf4                (~16 GB total; softest)
#
# "gguf": the ~28 GB bf16 transformer is replaced by city96's ~11.3 GB
# Q4_K_M single-file GGUF quant (loaded straight from HuggingFace with
# WanTransformer3DModel.from_single_file + GGUFQuantizationConfig), and
# the UMT5 text encoder is kept small with bitsandbytes int8. GGUF
# weights are DEQUANTIZED ON THE FLY each forward pass, so per-step
# speed is the same or a touch slower than bf16 — the win is VRAM:
# full GPU residency fits a 24 GB L4, and the first-run download drops
# by the ~28 GB bf16 transformer. Q4_K_M is the accepted quality sweet
# spot (Q3 and below get visibly mushy, especially stacked with the
# Lightning distill LoRA).
#
# GGUF covers the i2v 480P/720P and FLF2V pipelines. VACE uses a
# different transformer class (WanVACETransformer3DModel), so it — and
# any GGUF download/attach failure — automatically falls back to
# bitsandbytes 8-bit. If the Lightning LoRA ever fails to attach on the
# GGUF transformer (runtime LoRA-on-GGUF is newer in diffusers than the
# bnb path), set QUANTIZATION = "8bit" below and re-run.
QUANTIZATION = "gguf"

# city96's Q4_K_M GGUF single-file transformer quants, keyed by
# (mode, repo_key). Anything not mapped here — or any load failure —
# falls back to bitsandbytes 8-bit for that pipeline.
GGUF_Q4KM_URLS = {
    ("i2v", "480P"): ("https://huggingface.co/city96/"
                      "Wan2.1-I2V-14B-480P-gguf/resolve/main/"
                      "wan2.1-i2v-14b-480p-Q4_K_M.gguf"),
    ("i2v", "720P"): ("https://huggingface.co/city96/"
                      "Wan2.1-I2V-14B-720P-gguf/resolve/main/"
                      "wan2.1-i2v-14b-720p-Q4_K_M.gguf"),
    ("flf2v", "720P"): ("https://huggingface.co/city96/"
                        "Wan2.1-FLF2V-14B-720P-gguf/resolve/main/"
                        "wan2.1-flf2v-14b-720p-Q4_K_M.gguf"),
}

# ---- Wan 2.2 I2V-A14B (two-expert MoE) --------------------------------
# Wan 2.2's 14B image-to-video model is what MOST Civitai LoRAs target
# ("Wan Video 2.2 I2V-A14B"). It's a Mixture-of-Experts: a HIGH-noise
# transformer for early denoising + a LOW-noise transformer for late
# detail. In diffusers' WanImageToVideoPipeline these load as
# `transformer` (high) and `transformer_2` (low). We use city96/
# QuantStack Q4_K_M GGUF for each expert (~9.65 GB each -> ~19 GB both,
# fits a 24 GB L4), with the diffusers repo supplying the configs.
WAN22_I2V_CONFIG_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN22_GGUF = {
    "480P": {
        "high": ("https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/"
                 "resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"),
        "low":  ("https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/"
                 "resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"),
    },
}
WAN22_GGUF["720P"] = WAN22_GGUF["480P"]   # same GGUF serves both profiles

# lightx2v "Lightning" 4-step distill LoRAs — one per expert. Applied to
# their matching transformer; enables the fast 4-step / CFG~1 render.
# These are the community-standard rank-64 4step LoRAs.
WAN22_LIGHTNING = {
    "high": ("https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/"
             "main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022"
             ".safetensors"),
    "low":  ("https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/"
             "main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022"
             ".safetensors"),
}
# Wan 2.2 A14B Lightning render settings (both experts distilled to 4
# steps total, split across the high->low handoff; CFG off).
WAN22_LIGHTNING_STEPS = 4
WAN22_LIGHTNING_GUIDANCE = 1.0
# Civitai baseModel string for the LoRA search filter.
WAN22_CIVITAI_BASE = "Wan Video 2.2 I2V-A14B"

# VRAM (GB) at/above which the whole pipeline is kept resident on the
# GPU; below it, model CPU offload is used instead. The figure depends
# on the quantization mode above (quantized models are far smaller).
# gguf: ~11.3 GB transformer + ~6 GB int8 UMT5 + fp32 VAE/CLIP ≈ 20 GB
# of weights, so 21 GB lets a 24 GB L4 go fully resident (a failed
# .to() still falls back to CPU offload automatically).
RESIDENCY_THRESHOLD = {"none": 45.0, "8bit": 30.0,
                       "fp8": 30.0, "4bit": 22.0, "gguf": 21.0}
def _residency_threshold():
    return RESIDENCY_THRESHOLD.get(QUANTIZATION.lower(), 45.0)

# ---- LTX-2.3 engine (alternate base model) -------------------------------
# Second engine next to Wan 2.1: Lightricks LTX-2.3 22B DISTILLED, fp8.
# Why this exact variant (the "fast + easy download + optimized" pick):
#   * distilled  -> the lightning behavior is BAKED IN: fixed 8+4 step
#                   two-stage schedule at CFG 1 — no separate lightning
#                   LoRA to download, fuse, or babysit.
#   * fp8        -> 29.5 GB single file instead of the 46 GB bf16
#                   distilled ckpt, loaded with --quantization fp8-cast
#                   (works on any CUDA GPU; folds prequant scales at
#                   load time — not the Hopper-only fp8-scaled-mm path).
#   * bonus      -> LTX-2 generates synced AUDIO with the video, does a
#                   2x spatial-upscale second stage (1080p-class out),
#                   and one clip can run up to ~481 frames (~20 s @ 24).
# It runs via the official ltx-pipelines package installed into the MAIN
# Colab environment (same interpreter as Wan — no isolated venv), invoked
# as a subprocess per job. Everything is set up lazily on the FIRST LTX
# job (~55 GB once: checkpoint 29.5 + Gemma text encoder ~24 + upscaler ~1).
LTX_DIR = "/content/ltx"
LTX_MODELS = f"{LTX_DIR}/models"
LTX_REPO_GIT = "https://github.com/Lightricks/LTX-2.git"
LTX_CKPT_URL = ("https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/"
                "main/ltx-2.3-22b-distilled-fp8.safetensors")
# BF16 distilled v1.1 — required for VIDEO-TO-VIDEO. The docs are explicit
# that fp8-cast pairs with BF16 checkpoints; feeding it the fp8 checkpoint
# works for plain rendering (no merge) but breaks the IC-LoRA merge: the
# merged weights blow up to bf16 (~93 GB resident, observed) and lose their
# input_scale keys (the streaming KeyError). The current Union IC-LoRA is
# built against this v1.1 distilled model.
LTX_CKPT_BF16_URL = ("https://huggingface.co/Lightricks/LTX-2.3/resolve/"
                     "main/ltx-2.3-22b-distilled-1.1.safetensors")
LTX_UPSCALER_URL = ("https://huggingface.co/Lightricks/LTX-2.3/resolve/"
                    "main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
# Lightricks' own UNGATED mirror of google/gemma-3-12b-it-qat-q4_0-
# unquantized (the google repo needs a license click + HF token).
LTX_GEMMA_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
# ── IC-LoRA control models for VIDEO-TO-VIDEO (verified HF locations) ──
# Union Control = canny/depth structure control (used for both raw
# conditioning and canny/depth motion-transfer). Motion Track = spline
# trajectory control. Both are ref0.5 (reference downscaled 2x).
LTX_ICLORA = {
    "union": {
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    },
    "motion_track": {
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        "file": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
    },
}
# Stage-1 pixel budgets (output is 2x after the upscale stage), dims
# snapped to /32 as the model requires.
LTX_STAGE1_AREA = {"480P": 352 * 640,    # -> ~704x1280-class output
                   "720P": 544 * 960}    # -> ~1088x1920-class output
# The 22B distilled fp8 checkpoint is ~30 GB. If the GPU has headroom for
# it plus activations, keep the whole model RESIDENT ON THE GPU (offload
# "none") — fastest, and it stays on-device between clips. Below that we
# fall back to CPU offload (weights streamed from RAM); the persistent
# worker still keeps them warm in RAM so clips don't reload the model.
LTX_GPU_RESIDENT_MIN_GB = 38.0   # >= this VRAM -> keep model on the GPU
LTX_OFFLOAD_BELOW_GB = 38.0      # < this VRAM -> stream weights from CPU

# ---- preset LoRAs ------------------------------------------------------
# LoRAs auto-downloaded at startup so they appear in the UI menu ready to
# use. The entry below is lightx2v's Wan 2.1 i2v 4-step "lightning"
# distill LoRA — it lets you generate in ~4 steps instead of 30-40.
# It is step- AND cfg-distilled, so when it is active you must also run
# Steps=4 and Guidance=1.0. The "Lightning 4-step" button in the UI sets
# all of that at once; it loads at strength 0 so the normal 40-step
# workflow is unaffected until you switch it on.
PRESET_LORAS = [
    {
        "name": "lightning_4step",
        "url": ("https://huggingface.co/lightx2v/Wan2.1-Distill-Loras/"
                "resolve/main/"
                "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors"),
        "scale": 0.0,
    },
]

# ---- console capture ---------------------------------------------------
console_lines = collections.deque(maxlen=800)

class _Tee:
    def __init__(self, o): self._o = o
    def write(self, s):
        self._o.write(s)
        if s.strip(): console_lines.append(s.rstrip("\n"))
        return len(s)
    def flush(self): self._o.flush()
    def __getattr__(self, n): return getattr(self._o, n)

sys.stdout = _Tee(sys.__stdout__)
sys.stderr = _Tee(sys.__stderr__)
def _log(m): print(m)

# ---- shared state ------------------------------------------------------
STATE = {
    "pipe": None,                      # the active Wan pipeline
    "mode": "i2v",                     # i2v | flf2v | vace
    "model_id": MODEL_ID,              # which repo the live pipe holds
    "pipe_key": None,                  # (mode, model_id) of the live pipe
    "residency": "unknown",            # "gpu" or "cpu-offload"
    "loras": {},                       # name -> {path, scale, attached}
    "lock": threading.Lock(),          # one GPU job at a time
    "load_lock": threading.Lock(),
}
jobs = {}
job_queue = collections.deque()      # pending (job_id, params), FIFO
job_queue_evt = threading.Event()    # wakes the dispatcher thread

class _JobCancelled(Exception):
    """Raised inside the diffusion step callback (or between clips) to
    abort a running job cleanly when the user cancels it."""
    pass

# ---- LoRA helpers ------------------------------------------------------
def download_lora(url, dest_name, out_dir="/content/loras"):
    """Download a .safetensors LoRA.

    HuggingFace: fetched directly (HF_TOKEN as bearer for gated repos).
    Civitai: proxied through missinglink.build with YOUR MissingLink
    membership key — the worker attaches MissingLink's Civitai API key
    server-side and streams the file back, so no personal Civitai key
    is ever needed (or exposed)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    headers = {}
    full_url = url
    low = url.lower()
    hf_key = HF_TOKEN or _get_secret("HF_TOKEN", verbose=False)
    if "huggingface" in low and hf_key:
        headers["Authorization"] = f"Bearer {hf_key}"
    elif "civitai" in low:
        # A Civitai *download* URL is /api/download/models/<id>. A model
        # *page* URL (/models/<id>) is not downloadable and will 401.
        if "/api/download/" not in low:
            raise RuntimeError(
                "this looks like a Civitai model PAGE url — use the "
                "direct download link instead, e.g. "
                "https://civitai.com/api/download/models/<versionId> "
                "(right-click the download button -> Copy link)")
        if not (ML.get("key") and _ml_unlocked()):
            raise RuntimeError(
                "Civitai downloads need you signed in with free renders "
                f"left or a membership — sign in in the studio UI, or "
                f"sign up at {MISSINGLINK_SIGNUP_URL}")
        from urllib.parse import quote as _q
        full_url = (f"{MISSINGLINK_API}/api/notebook/civitai/download"
                    f"?url={_q(url, safe='')}")
        headers["Authorization"] = f"Bearer {ML['key']}"
    path = str(Path(out_dir) / f"{dest_name}.safetensors")
    with _requests.get(full_url, headers=headers, stream=True,
                       allow_redirects=True, timeout=300) as r:
        if r.status_code in (401, 402, 403):
            if "civitai" in low:
                if r.status_code == 402:
                    ML["remaining"] = 0
                    ML["reason"] = "free_limit_reached"
                    raise RuntimeError(
                        "you've used all your free renders — sign up at "
                        f"{MISSINGLINK_SIGNUP_URL}")
                ML["authed"] = False
                ML["reason"] = "no_session"
                raise RuntimeError(
                    "please sign in with Google again in the studio UI.")
            raise RuntimeError(
                f"{r.status_code} unauthorized — for gated HuggingFace "
                "repos set HF_TOKEN")
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    if Path(path).stat().st_size < 4096:
        # A tiny file is usually an HTML error/login page, not a LoRA.
        raise RuntimeError("downloaded file is too small — likely an "
                           "auth or bad-URL error (check your MissingLink "
                           "membership / HF_TOKEN and that the URL is a "
                           "direct download link)")
    return path

def register_lora(name, url, scale, engine="wan"):
    """Download + attach a LoRA to the live pipeline.

    Wan: attaches to the resident diffusers pipe immediately.
    LTX: the distilled pipeline takes LoRAs at construction, so we
    download + register here and restart the resident worker so the LoRA
    attaches on the next render (handled in _ltx_worker_start)."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "lora"
    if name in STATE["loras"]:
        return False, f"a LoRA named '{name}' already exists"
    if not url.lower().startswith("http"):
        return False, "URL must start with http"
    _log(f"  LoRA '{name}': downloading...")
    try:
        path = download_lora(url, dest_name=name)
    except Exception as e:
        return False, f"download failed — {e}"
    # LTX: register + restart the worker so it rebuilds with the LoRA.
    if engine == "ltx":
        STATE["loras"][name] = {"path": path, "scale": float(scale),
                                "attached": True, "url": url, "engine": "ltx"}
        try:
            _ltx_worker_stop()   # next render restarts with the LoRA attached
        except Exception:
            pass
        _log(f"  LoRA '{name}': ready for LTX — attaches on the next render "
             f"(strength {scale}).")
        return True, "ok"
    attached = False
    pipe = STATE["pipe"]
    if pipe is not None:
        with STATE["load_lock"]:
            try:
                # Wan LoRAs from Civitai (kohya / musubi-tuner format)
                # are converted on the fly by recent diffusers.
                pipe.load_lora_weights(path, adapter_name=name)
                attached = True
            except Exception as e:
                _log(f"  LoRA '{name}': could not attach — {e}")
                if QUANTIZATION.lower() == "gguf":
                    _log("  (note: runtime LoRA on a GGUF-quantized "
                         "transformer is newer in diffusers — if this "
                         "keeps failing, set QUANTIZATION='8bit' at the "
                         "top of the cell and re-run.)")
    STATE["loras"][name] = {"path": path, "scale": float(scale),
                            "attached": attached, "url": url}
    if not attached:
        return False, ("downloaded but could not attach — the file may "
                       "not be a Wan 2.1 LoRA, or its key format is "
                       "unsupported")
    _log(f"  LoRA '{name}': ready (strength {scale}).")
    return True, "ok"

def remove_lora(name):
    info = STATE["loras"].pop(name, None)
    if not info:
        return False
    pipe = STATE["pipe"]
    if pipe is not None and info.get("attached"):
        try: pipe.delete_adapters(name)
        except Exception: pass
    try: os.remove(info["path"])
    except Exception: pass
    return True

def apply_loras(pipe):
    """Activate every registered LoRA at its current strength."""
    names = [n for n, i in STATE["loras"].items() if i.get("attached")]
    if not names:
        try: pipe.disable_lora()
        except Exception: pass
        return
    weights = [STATE["loras"][n]["scale"] for n in names]
    try:
        pipe.set_adapters(names, adapter_weights=weights)
    except Exception as e:
        _log(f"  set_adapters warning: {e}")

# ---- pipeline ----------------------------------------------------------
def _attach_all_loras(pipe):
    """(Re)attach every registered LoRA to a freshly built pipeline so
    switching task/model keeps LoRAs live. A LoRA trained for one task
    (e.g. i2v) may not load on another (vace); incompatible ones are
    skipped and flagged not-attached."""
    for name, info in STATE["loras"].items():
        try:
            pipe.load_lora_weights(info["path"], adapter_name=name)
            info["attached"] = True
        except Exception as e:
            info["attached"] = False
            _log(f"  LoRA '{name}': not compatible with this model — {e}")

def _load_gguf_transformer(mode, model_key, model_id, common):
    """Try to load city96's Q4_K_M GGUF quant of the Wan transformer.

    Returns the transformer module, or None if this (mode, model_key)
    has no mapped GGUF or the load fails — the caller then falls back
    to bitsandbytes 8-bit. VACE is deliberately unmapped: it uses a
    different transformer class (WanVACETransformer3DModel), so the
    i2v GGUF files cannot be loaded into it."""
    gguf_url = GGUF_Q4KM_URLS.get((mode, model_key))
    if gguf_url is None:
        _log(f"  gguf: no Q4_K_M file mapped for {mode}:{model_key} — "
             "using bitsandbytes 8-bit for this pipeline instead.")
        return None
    try:
        from diffusers import WanTransformer3DModel, GGUFQuantizationConfig
        fname = gguf_url.rsplit("/", 1)[-1]
        _log(f"  quantization: GGUF Q4_K_M — loading {fname}")
        _log("  (~11.3 GB transformer; one-time download, cached after)")
        transformer = WanTransformer3DModel.from_single_file(
            gguf_url,
            quantization_config=GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16),
            config=model_id, subfolder="transformer",
            torch_dtype=torch.bfloat16, **common)
        _log("  GGUF transformer ready. Weights dequantize on the fly: "
             "this saves VRAM, not per-step time — pair with Lightning "
             "for speed.")
        return transformer
    except Exception as e:
        _log(f"  GGUF transformer load failed ({e}) — falling back to "
             "bitsandbytes 8-bit.")
        return None

def _load_wan22_expert(url, config_repo, subfolder, common):
    """Load one Wan 2.2 expert (high- or low-noise) from a QuantStack
    Q4_K_M GGUF single file, with configs from the diffusers repo."""
    from diffusers import WanTransformer3DModel, GGUFQuantizationConfig
    fname = url.rsplit("/", 1)[-1]
    _log(f"  [wan2.2] loading {subfolder} expert GGUF: {fname} (~9.7 GB)")
    return WanTransformer3DModel.from_single_file(
        url,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16),
        config=config_repo, subfolder=subfolder,
        torch_dtype=torch.bfloat16, **common)


def build_pipeline_wan22(profile="480P", lightning=True):
    """Build the Wan 2.2 I2V-A14B two-expert pipeline.

    Loads BOTH experts as GGUF Q4_K_M (high-noise -> transformer,
    low-noise -> transformer_2) and, when lightning is on, attaches the
    matching lightx2v 4-step distill LoRA to each expert for a fast
    4-step / CFG~1 render. This is the variant most Civitai LoRAs target
    ("Wan Video 2.2 I2V-A14B"). Kept resident like the other engines;
    switching engines frees it.
    """
    try:
        _ltx_worker_stop()
    except Exception:
        pass
    key = ("wan22", "I2V-A14B", bool(lightning))
    with STATE["load_lock"]:
        if STATE["pipe"] is not None and STATE["pipe_key"] == key:
            return STATE["pipe"]
        if STATE["pipe"] is not None:
            _log("Switching -> Wan 2.2 I2V-A14B; freeing old pipeline.")
            STATE["pipe"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        from transformers import CLIPVisionModel
        common = {}
        if HF_TOKEN:
            common["token"] = HF_TOKEN
        cfg = WAN22_I2V_CONFIG_REPO
        gg = WAN22_GGUF.get(profile, WAN22_GGUF["480P"])

        _log("Loading Wan 2.2 I2V-A14B (two-expert MoE, GGUF Q4_K_M)...")
        vae = AutoencoderKLWan.from_pretrained(
            cfg, subfolder="vae", torch_dtype=torch.float32, **common)
        # high-noise -> transformer, low-noise -> transformer_2
        t_high = _load_wan22_expert(gg["high"], cfg, "transformer", common)
        t_low = _load_wan22_expert(gg["low"], cfg, "transformer_2", common)
        image_encoder = CLIPVisionModel.from_pretrained(
            cfg, subfolder="image_encoder",
            torch_dtype=torch.float32, **common)
        # UMT5 text encoder -> bnb int8 to keep both experts + encoder in
        # ~24 GB.
        quant_cfg = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=["text_encoder"])
        pipe = WanImageToVideoPipeline.from_pretrained(
            cfg, vae=vae, transformer=t_high, transformer_2=t_low,
            image_encoder=image_encoder, quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16, **common)

        # ---- Lightning: one distill LoRA per expert ----------------------
        if lightning:
            try:
                Path(LTX_DIR).mkdir(parents=True, exist_ok=True)
                hi = f"/content/wan22_light_high.safetensors"
                lo = f"/content/wan22_light_low.safetensors"
                if not Path(hi).exists():
                    _log("  [wan2.2] downloading Lightning (high-noise) LoRA...")
                    _ltx_download(WAN22_LIGHTNING["high"], hi,
                                  "Lightning high-noise LoRA")
                if not Path(lo).exists():
                    _log("  [wan2.2] downloading Lightning (low-noise) LoRA...")
                    _ltx_download(WAN22_LIGHTNING["low"], lo,
                                  "Lightning low-noise LoRA")
                # Attach each LoRA to its expert. diffusers routes a LoRA
                # to transformer_2 via the load_into_transformer_2 kwarg.
                pipe.load_lora_weights(hi, adapter_name="light_high")
                try:
                    pipe.load_lora_weights(
                        lo, adapter_name="light_low",
                        load_into_transformer_2=True)
                except TypeError:
                    # older diffusers: set the component explicitly
                    pipe.load_lora_weights(
                        lo, adapter_name="light_low",
                        components=["transformer_2"])
                pipe.set_adapters(["light_high", "light_low"],
                                  adapter_weights=[1.0, 1.0])
                _log(f"  [wan2.2] Lightning ON — {WAN22_LIGHTNING_STEPS}-step "
                     f"render, CFG {WAN22_LIGHTNING_GUIDANCE} "
                     "(both experts distilled).")
            except Exception as e:
                _log(f"  [wan2.2] Lightning attach failed ({e}) — running "
                     "the base model at normal steps instead.")

        vram_gb = (torch.cuda.mem_get_info()[1] / 1e9
                   if torch.cuda.is_available() else 0)
        residency = "cpu-offload"
        # Two GGUF experts (~19 GB) + int8 UMT5 + fp32 VAE/CLIP: needs ~30 GB
        # to sit fully on the GPU; below that, CPU offload streams experts.
        if vram_gb >= 30.0:
            try:
                pipe.to(DEVICE); residency = "gpu"
                _log(f"  [wan2.2] FULL GPU RESIDENCY ({vram_gb:.0f} GB).")
            except Exception as e:
                torch.cuda.empty_cache()
                _log(f"  [wan2.2] residency move failed ({e}) — CPU offload.")
                pipe.enable_model_cpu_offload()
        else:
            _log(f"  [wan2.2] model CPU offload ({vram_gb:.0f} GB VRAM). "
                 "The two experts stream between high/low denoise stages.")
            pipe.enable_model_cpu_offload()
        for fn in ("enable_slicing", "enable_tiling"):
            try: getattr(pipe.vae, fn)()
            except Exception: pass

        STATE["pipe"] = pipe
        STATE["mode"] = "i2v"
        STATE["model_id"] = "Wan2.2-I2V-A14B"
        STATE["pipe_key"] = key
        STATE["residency"] = residency
        STATE["wan22_lightning"] = bool(lightning)
        _attach_all_loras(pipe)   # user's Civitai 2.2 LoRAs on top
        _log(f"  Wan 2.2 pipeline ready (residency: {residency}).")
        return pipe


def build_pipeline(mode=None, model_key=None):
    """Build/swap the shared Wan pipeline for a given task.

    mode      : 'i2v' | 'flf2v' | 'vace'
    model_key : repo key within MODE_REPOS[mode]
                (i2v/flf2v -> '480P'/'720P'; vace -> '1.3B'/'14B')

    Switching to Wan frees the resident LTX model — the two engines don't
    share the GPU, and the user asked that a model stay loaded only until
    a different one is picked.

    i2v and flf2v use WanImageToVideoPipeline (+ CLIP vision encoder);
    vace uses WanVACEPipeline (no vision encoder). Exactly one pipeline
    is kept resident at a time — switching task or repo rebuilds it.
    Placement + quantization follow the same residency rules throughout.
    """
    try:
        _ltx_worker_stop()
    except Exception:
        pass
    mode = mode or STATE.get("mode", "i2v")
    if mode not in MODE_REPOS:
        mode = "i2v"
    if model_key is None or model_key not in MODE_REPOS[mode]:
        model_key = next(iter(MODE_REPOS[mode]))
    model_id = MODE_REPOS[mode][model_key]
    key = (mode, model_id)
    with STATE["load_lock"]:
        if STATE["pipe"] is not None and STATE["pipe_key"] == key:
            return STATE["pipe"]

        # Switching task/model: drop the old pipe first to free VRAM.
        if STATE["pipe"] is not None:
            _log(f"Switching -> {mode}:{model_id}; freeing old pipeline.")
            STATE["pipe"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _log(f"Loading {mode} pipeline: {model_id} ...")
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        try:
            from diffusers import WanVACEPipeline
        except Exception:
            WanVACEPipeline = None

        common = {}
        if HF_TOKEN:
            common["token"] = HF_TOKEN

        # VAE (fp32, for quality) is shared by every task; never quantized.
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32, **common)

        # Shrink the heavy modules (transformer + UMT5 text encoder).
        qmode = QUANTIZATION.lower()
        quant_cfg = None
        gguf_transformer = None

        if qmode == "gguf":
            gguf_transformer = _load_gguf_transformer(
                mode, model_key, model_id, common)
            if gguf_transformer is None:
                qmode = "8bit"   # fallback path below

        if gguf_transformer is not None:
            # Transformer is already GGUF-quantized; only the UMT5 text
            # encoder still needs shrinking (bnb int8: ~6 GB vs ~11 GB).
            from diffusers.quantizers import PipelineQuantizationConfig
            quant_cfg = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["text_encoder"])
            _log("  text encoder: bitsandbytes int8.")
        elif qmode in ("8bit", "fp8", "4bit"):
            from diffusers.quantizers import PipelineQuantizationConfig
            heavy = ["transformer", "text_encoder"]
            if qmode == "8bit":
                quant_cfg = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_8bit",
                    quant_kwargs={"load_in_8bit": True},
                    components_to_quantize=heavy)
            elif qmode == "4bit":
                quant_cfg = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={"load_in_4bit": True,
                                  "bnb_4bit_quant_type": "nf4",
                                  "bnb_4bit_compute_dtype": torch.bfloat16},
                    components_to_quantize=heavy)
            else:   # fp8 via torchao (literal float8 weights)
                quant_cfg = PipelineQuantizationConfig(
                    quant_backend="torchao",
                    quant_kwargs={"quant_type": "float8_weight_only"},
                    components_to_quantize=heavy)
            _log(f"  quantization: {qmode} — transformer + text encoder.")
        else:
            _log("  quantization: none (full bf16).")

        pipe_kwargs = dict(vae=vae, torch_dtype=torch.bfloat16, **common)
        if quant_cfg is not None:
            pipe_kwargs["quantization_config"] = quant_cfg
        if gguf_transformer is not None:
            # Passing the module directly stops from_pretrained from
            # downloading the ~28 GB bf16 transformer shards at all.
            pipe_kwargs["transformer"] = gguf_transformer

        if mode == "vace":
            if WanVACEPipeline is None:
                raise RuntimeError(
                    "this diffusers build lacks WanVACEPipeline — update "
                    "diffusers (this cell installs from git).")
            pipe = WanVACEPipeline.from_pretrained(model_id, **pipe_kwargs)
        else:
            # i2v + flf2v both use the CLIP vision encoder (fp32).
            from transformers import CLIPVisionModel
            image_encoder = CLIPVisionModel.from_pretrained(
                model_id, subfolder="image_encoder",
                torch_dtype=torch.float32, **common)
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, image_encoder=image_encoder, **pipe_kwargs)

        vram_gb = (torch.cuda.mem_get_info()[1] / 1e9
                   if torch.cuda.is_available() else 0)
        # Residency threshold follows the quantization that actually got
        # used for THIS pipeline (gguf may have fallen back to 8bit).
        thr = RESIDENCY_THRESHOLD.get(
            "gguf" if gguf_transformer is not None else qmode, 45.0)

        residency = "cpu-offload"
        if vram_gb >= thr:
            try:
                pipe.to(DEVICE)
                residency = "gpu"
                _log(f"  FULL GPU RESIDENCY ({vram_gb:.0f} GB VRAM, "
                     f"{qmode} weights) — whole pipeline on the GPU.")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                _log(f"  full-residency move hit CUDA OOM despite "
                     f"{vram_gb:.0f} GB — falling back to CPU offload.")
                pipe.enable_model_cpu_offload()
            except Exception as e:
                torch.cuda.empty_cache()
                _log(f"  full-residency move failed ({e}) — "
                     "falling back to CPU offload.")
                pipe.enable_model_cpu_offload()
        else:
            _log(f"  model CPU offload ({vram_gb:.0f} GB VRAM — under "
                 f"the {thr:.0f} GB needed for full residency with "
                 f"{qmode} weights). Slower but it fits.")
            pipe.enable_model_cpu_offload()

        # VAE memory savers — safe in both residency modes.
        for fn in ("enable_slicing", "enable_tiling"):
            try: getattr(pipe.vae, fn)()
            except Exception: pass

        STATE["pipe"] = pipe
        STATE["mode"] = mode
        STATE["model_id"] = model_id
        STATE["pipe_key"] = key
        STATE["residency"] = residency
        _attach_all_loras(pipe)
        _log(f"  Wan {mode} pipeline ready (residency: {residency}).")
        return pipe

def _wan_dims(img, profile):
    """Derive output W/H for Wan i2v from the uploaded image.

    Wan i2v treats resolution as a pixel BUDGET (max_area) and keeps the
    source image's aspect ratio, then snaps to the model grid. This is
    the official diffusers recipe:
        height = round(sqrt(max_area * AR)) // mod * mod
        width  = round(sqrt(max_area / AR)) // mod * mod
    where AR = height/width of the source and mod is the VAE spatial
    scale factor times the transformer patch size."""
    pipe = STATE["pipe"]
    try:
        mod = (pipe.vae_scale_factor_spatial
               * pipe.transformer.config.patch_size[1])
    except Exception:
        mod = 16   # Wan 2.1: vae_scale_factor_spatial 8 * patch 2
    max_area = MAX_AREA.get(profile, MAX_AREA["480P"])
    ow, oh = img.size
    ar = oh / ow                                  # height / width
    H = int(round(math.sqrt(max_area * ar))) // mod * mod
    W = int(round(math.sqrt(max_area / ar))) // mod * mod
    H = max(mod, H)
    W = max(mod, W)
    return W, H

# ---- LTX engine ----------------------------------------------------------
LTX_SETUP_LOCK = threading.Lock()

def _ltx_paths():
    return {
        "ckpt": f"{LTX_MODELS}/ltx-2.3-22b-distilled-fp8.safetensors",
        "upscaler": f"{LTX_MODELS}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "gemma": f"{LTX_MODELS}/gemma",
    }

def _dl_progress(dest_p, total, label, job, p0, p1, extra=""):
    done = dest_p.stat().st_size if dest_p.exists() else 0
    if total:
        msg = (f"downloading {label}: "
               f"{done/1e9:.1f}/{total/1e9:.1f} GB{extra}")
        if job is not None:
            job.update(stage=msg, progress=int(
                p0 + done / total * (p1 - p0)))
        _log("  [ltx] " + msg)

def _aria2_wait(proc, dest_p, total, label, job, p0, p1):
    """Poll an aria2c child, streaming progress into the job/log."""
    last = 0.0
    while proc.poll() is None:
        if job is not None and job.get("cancel"):
            try: proc.terminate()
            except Exception: pass
            raise _JobCancelled()
        time.sleep(1)
        if time.time() - last > 4:
            last = time.time()
            _dl_progress(dest_p, total, label, job, p0, p1,
                         " (16 parallel connections)")
    return proc.returncode

def _ltx_download(url, dest, label, job=None, p0=10, p1=60):
    """Fetch a big file as fast as the pipe allows.

    Order of preference:
      1. an aria2c process already started by the launch-time early
         fetch (LTX_EARLY_FETCH) — just wait on it;
      2. a fresh aria2c with 16 parallel connections (resumes partials,
         including ones left by the old single-stream downloader);
      3. the original single-stream requests download (fallback)."""
    import shutil as _sh
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    total = 0
    try:
        head = _requests.head(url, headers=headers,
                              allow_redirects=True, timeout=30)
        total = int(head.headers.get("content-length") or 0)
    except Exception:
        pass

    # 1) the launch-time early fetch may already be pulling this file
    early = None
    try:
        early = LTX_EARLY_FETCH.pop(str(dest_p), None)
    except Exception:
        early = None
    if early is not None and early.poll() is None:
        _log(f"  [ltx] {label}: launch-time download already in flight — "
             "waiting for it to finish...")
        _aria2_wait(early, dest_p, total, label, job, p0, p1)
    have = dest_p.stat().st_size if dest_p.exists() else 0
    if total and have >= total:
        _log(f"  [ltx] {label}: already downloaded.")
        return

    # 2) fast path: aria2c, 16 connections, resume (-c) any partial
    if _sh.which("aria2c") is None:
        subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"],
                       capture_output=True)
    if _sh.which("aria2c"):
        if have:
            _log(f"  [ltx] {label}: resuming at {have/1e9:.1f} GB "
                 "(aria2c, 16 parallel connections).")
        else:
            _log(f"  [ltx] downloading {label} "
                 "(aria2c, 16 parallel connections)...")
        cmd = ["aria2c", "-c", "-x16", "-s16", "-k1M",
               "--file-allocation=none", "--console-log-level=warn",
               "--summary-interval=0", "-d", str(dest_p.parent),
               "-o", dest_p.name]
        if HF_TOKEN:
            cmd += ["--header", f"Authorization: Bearer {HF_TOKEN}"]
        proc = subprocess.Popen(cmd + [url], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        rc = _aria2_wait(proc, dest_p, total, label, job, p0, p1)
        have = dest_p.stat().st_size if dest_p.exists() else 0
        if rc == 0 and (not total or have >= total):
            _log(f"  [ltx] {label}: download complete.")
            return
        _log(f"  [ltx] {label}: aria2c exited {rc} "
             f"({have/1e9:.1f}/{total/1e9:.1f} GB) — falling back to "
             "single-stream download for the remainder.")

    # 3) fallback: original single-stream resume download
    mode = "wb"
    if 0 < have < (total or 0):
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        _log(f"  [ltx] {label}: resuming at {have/1e9:.1f} GB.")
    else:
        have = 0
    _log(f"  [ltx] downloading {label}...")
    with _requests.get(url, headers=headers, stream=True,
                       timeout=600) as r:
        r.raise_for_status()
        done, last = have, time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=16 << 20):
                if job is not None and job.get("cancel"):
                    raise _JobCancelled()
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 4:
                    last = time.time()
                    if total:
                        msg = (f"downloading {label}: "
                               f"{done/1e9:.1f}/{total/1e9:.1f} GB")
                        if job is not None:
                            job.update(stage=msg, progress=int(
                                p0 + done / total * (p1 - p0)))
                        _log("  [ltx] " + msg)
def _ensure_ltx_ready(job=None):
    """One-time lazy setup: install the official ltx-pipelines package
    into the MAIN Colab environment (same interpreter that runs Wan —
    no isolated venv) + model downloads (~55 GB total). Serialized by a
    lock so a queue of LTX jobs only sets up once; every step is
    resumable via the .pkgs / .ready markers."""
    def st(msg, pct=None):
        _log(f"  [ltx] {msg}")
        if job is not None:
            job.update(stage=msg,
                       progress=(pct if pct is not None
                                 else job.get("progress", 5)))
    with LTX_SETUP_LOCK:
        p = _ltx_paths()
        if (Path(LTX_DIR) / ".ready").exists():
            return p
        Path(LTX_MODELS).mkdir(parents=True, exist_ok=True)

        def _run(cmd, label):
            """Run a setup command, capturing output so a failure shows
            the REAL error (not a bare 'exit status 1')."""
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tail = ((r.stderr or "") + (r.stdout or "")).strip()
                tail = tail[-800:] if tail else "(no output)"
                _log(f"  [ltx] {label} FAILED (exit {r.returncode}):\n{tail}")
                raise RuntimeError(f"{label} failed — {tail[-300:]}")
            return r

        if not (Path(LTX_DIR) / ".pkgs").exists():
            if not Path(f"{LTX_DIR}/repo").exists():
                st("LTX setup: cloning Lightricks/LTX-2", 4)
                _run(["git", "clone", "--depth", "1",
                      LTX_REPO_GIT, f"{LTX_DIR}/repo"], "git clone LTX-2")
            # Install straight into the running Colab environment
            # (sys.executable) — Wan and LTX share it. PyPI only carries a
            # stale ltx-core 1.0.0, so install BOTH from the cloned repo:
            # core first, then pipelines with --no-deps so pip can't swap
            # in the old PyPI ltx-core.
            st("LTX setup: installing ltx-core "
               "(may adjust torch — takes a few minutes)", 5)
            _run([sys.executable, "-m", "pip", "install", "-q",
                  f"{LTX_DIR}/repo/packages/ltx-core"], "ltx-core install")
            st("LTX setup: installing ltx-pipelines", 8)
            _run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                  f"{LTX_DIR}/repo/packages/ltx-pipelines"],
                 "ltx-pipelines install")
            _run([sys.executable, "-m", "pip", "install", "-q",
                  "av", "tqdm", "pillow", "openimageio", "hf_transfer"],
                 "ltx deps install")
            (Path(LTX_DIR) / ".pkgs").touch()
        _ltx_download(LTX_CKPT_URL, p["ckpt"],
                      "LTX-2.3 distilled fp8 checkpoint", job, 10, 55)
        _ltx_download(LTX_UPSCALER_URL, p["upscaler"],
                      "spatial upscaler", job, 55, 57)
        if not Path(p["gemma"], "config.json").exists():
            st("LTX setup: downloading Gemma-3 12B text encoder "
               "(~24 GB, ungated Lightricks mirror)", 58)
            # hf_transfer = Rust multi-stream downloader; big speedup
            try:
                import hf_transfer  # noqa: F401
            except Exception:
                subprocess.run([sys.executable, "-m", "pip", "install",
                                "-q", "hf_transfer"], capture_output=True)
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            from huggingface_hub import snapshot_download
            snapshot_download(LTX_GEMMA_REPO, local_dir=p["gemma"],
                              max_workers=8, token=HF_TOKEN or None)
        (Path(LTX_DIR) / ".ready").touch()
        st("LTX engine ready.", 12)
        return p

def _ensure_ltx_bf16(job=None):
    """Lazily fetch the BF16 distilled v1.1 checkpoint (46 GB, one-time),
    needed only for video-to-video: the IC-LoRA merge is only correct
    against bf16 weights (fp8-cast then quantizes at load, so runtime
    memory matches the normal fp8 path)."""
    dest = Path(LTX_MODELS) / "ltx-2.3-22b-distilled-1.1.safetensors"
    if dest.exists():
        return str(dest)
    _log("  [ltx] v2v needs the BF16 distilled checkpoint (one-time 46 GB "
         "download; the fp8 file can't take the IC-LoRA merge)...")
    _ltx_download(LTX_CKPT_BF16_URL, str(dest),
                  "LTX-2.3 distilled v1.1 (bf16, for video-to-video)",
                  job, 5, 12)
    return str(dest)

def _ensure_iclora(kind, job=None):
    """Lazily download an IC-LoRA control model (~650 MB) on first v2v use.
    Returns the local .safetensors path. kind in LTX_ICLORA."""
    spec = LTX_ICLORA.get(kind)
    if not spec:
        raise RuntimeError(f"unknown IC-LoRA control '{kind}'")
    dest = Path(LTX_MODELS) / spec["file"]
    if dest.exists():
        return str(dest)
    Path(LTX_MODELS).mkdir(parents=True, exist_ok=True)
    if job is not None:
        job.update(stage=f"downloading {kind} control model (~650 MB)",
                   progress=job.get("progress", 8))
    _log(f"  [ltx] fetching IC-LoRA '{kind}' from {spec['repo']}")
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=spec["repo"], filename=spec["file"],
                          local_dir=LTX_MODELS, token=HF_TOKEN or None)
    # hf_hub_download may nest under the filename; normalise to dest.
    if Path(got) != dest and Path(got).exists():
        try:
            import shutil as _sh
            _sh.copyfile(got, dest)
        except Exception:
            dest = Path(got)
    _log(f"  [ltx] IC-LoRA '{kind}' ready: {dest}")
    return str(dest)

def _ltx_preprocess_control(video_path, mode, tmpd, job=None):
    """Turn a raw control video into the signal the Union Control IC-LoRA
    expects. mode: 'raw' (no change, loose conditioning), 'canny' (edge
    maps), or 'depth' (depth maps). Returns a path to the processed video.
    Canny uses OpenCV; depth uses a lightweight MiDaS model (downloaded on
    first use)."""
    if mode == "raw":
        return video_path
    import cv2, numpy as _np
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 768)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 432)
    out = str(Path(tmpd) / f"control_{mode}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    depth_model = None
    if mode == "depth":
        if job is not None:
            job.update(stage="loading depth estimator (first use)")
        try:
            import torch as _t
            depth_model = _t.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                      trust_repo=True)
            depth_model.to("cuda").eval()
            tfm = _t.hub.load("intel-isl/MiDaS", "transforms",
                              trust_repo=True).small_transform
        except Exception as e:
            _log(f"  [ltx] depth model load failed ({e}); "
                 "falling back to canny.")
            mode = "canny"; depth_model = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mode == "canny":
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(g, 100, 200)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == "depth" and depth_model is not None:
            import torch as _t
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = tfm(rgb).to("cuda")
            with _t.no_grad():
                d = depth_model(inp)
                d = _t.nn.functional.interpolate(
                    d.unsqueeze(1), size=(H, W), mode="bicubic",
                    align_corners=False).squeeze().cpu().numpy()
            d = (d - d.min()) / (max(1e-6, d.max() - d.min()))
            dm = (d * 255).astype(_np.uint8)
            frame = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)
        vw.write(frame); n += 1
    cap.release(); vw.release()
    if depth_model is not None:
        del depth_model
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
    _log(f"  [ltx] control preprocess '{mode}': {n} frames -> {out}")
    return out if n > 0 else video_path

def _ltx_copy_source_audio(control_video, output_video, tmpd):
    """Replace the (LTX-generated) audio on an IC-LoRA v2v output with the
    ORIGINAL audio from the control video, aligned to the output's
    duration. If the control clip has no audio, the output is returned
    unchanged. Returns the new path (or the original on any failure)."""
    import shutil as _sh
    try:
        # Does the control video even have an audio stream?
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", control_video],
            capture_output=True, text=True)
        if not (probe.stdout or "").strip():
            _log("  [ltx] source video has no audio track; keeping generated audio.")
            return output_video
        # Output duration to align to.
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", output_video],
            capture_output=True, text=True)
        out_dur = float((pr.stdout or "0").strip() or 0) or None
        aud = str(Path(tmpd) / "src_audio.m4a")
        subprocess.run(
            ["ffmpeg", "-y", "-i", control_video, "-vn", "-c:a", "aac", aud],
            capture_output=True, text=True)
        if not Path(aud).exists():
            return output_video
        out = str(Path(tmpd) / "v2v_srcaudio.mp4")
        cmd = ["ffmpeg", "-y", "-i", output_video, "-i", aud,
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac"]
        if out_dur:
            # Trim/pad audio to the output length so sound stays in sync.
            cmd += ["-af", f"apad", "-t", f"{out_dur:.3f}"]
        cmd += [out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if Path(out).exists():
            _log("  [ltx] copied the source video's original audio onto the output.")
            return out
        return output_video
    except Exception as e:
        _log(f"  [ltx] copy-source-audio failed ({e}); keeping generated audio.")
        return output_video

def _ltx_stage1_dims(img, profile):
    """Stage-1 W/H from the source aspect (final output is 2x).

    The two-stage LTX pipeline requires BOTH dimensions to be multiples
    of 64 (it hard-asserts this), so we snap to 64 — not 32."""
    area = LTX_STAGE1_AREA.get(profile, LTX_STAGE1_AREA["480P"])
    ow, oh = img.size
    ar = oh / ow
    H = max(64, int(round(math.sqrt(area * ar) / 64)) * 64)
    W = max(64, int(round(math.sqrt(area / ar) / 64)) * 64)
    return W, H

def _ltx_last_frame(video_path, out_png, back=1):
    """Grab a frame near the END of a rendered clip as a PNG, to seed the
    next clip (continuous chaining). `back` = seconds-fraction from the end
    to grab; LTX-2.3's final 6-8 frames can smear, so chaining reads a
    CLEAN frame slightly before the very end when back>1."""
    off = max(1, int(back))
    r = subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.%02d" % off, "-i", video_path,
         "-update", "1", "-q:v", "2", "-frames:v", "1", out_png],
        capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_png).exists():
        r1 = subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
             "-update", "1", "-q:v", "2", "-frames:v", "1", out_png],
            capture_output=True, text=True)
        if not Path(out_png).exists():
            r2 = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vf",
                 "select=eq(n\\,0)", "-vsync", "0", "-frames:v", "1", out_png],
                capture_output=True, text=True)
            if not Path(out_png).exists():
                raise RuntimeError(
                    "could not extract the last frame to chain the next clip — "
                    + (r.stderr or "")[-200:])


def _ltx_trim_tail(video_path, out_path, trim_frames, fps):
    """Trim the last `trim_frames` frames off a clip (LTX-2.3's tail can
    smear). Falls back to a copy if trimming fails or would empty the clip."""
    import shutil as _sh
    try:
        if trim_frames <= 0:
            _sh.copyfile(video_path, out_path); return
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", video_path],
            capture_output=True, text=True)
        dur = float((pr.stdout or "0").strip() or 0)
        keep = dur - (trim_frames / float(fps or 24))
        if keep <= 0.2:
            _sh.copyfile(video_path, out_path); return
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-t", "%.3f" % keep,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             out_path], capture_output=True, text=True)
        if r.returncode != 0 or not Path(out_path).exists():
            _sh.copyfile(video_path, out_path)
    except Exception:
        _sh.copyfile(video_path, out_path)


def _ltx_strip_music(video_path, tmpd):
    """Remove the MUSIC from a rendered clip while keeping dialogue and
    sound effects. Uses Demucs source separation. Standard music models
    split vocals/drums/bass/other, where SFX ride along with the
    instrumental stems — so a clean music-only removal isn't guaranteed.
    We use the DnR-style approach: separate, then rebuild the track from
    the non-music stems. Returns a new mp4 path, or None on failure (caller
    keeps the original audio)."""
    try:
        import shutil as _sh
        # Ensure Demucs is available (installed lazily on first use).
        try:
            import demucs  # noqa: F401
        except Exception:
            _log("  [ltx] strip-music: installing Demucs (first use, ~1 min)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "demucs"], capture_output=True, text=True)
            try:
                import demucs  # noqa: F401
            except Exception:
                _log("  [ltx] strip-music: Demucs unavailable; keeping original.")
                return None
        # 1) Extract the audio.
        wav = str(Path(tmpd) / "mix.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "2",
             "-ar", "44100", wav], capture_output=True, text=True)
        if not Path(wav).exists():
            _log("  [ltx] strip-music: could not extract audio; keeping original.")
            return None
        # 2) Run Demucs. Prefer a Dialog/SFX/Music model if present, else
        #    the 6-stem model (which separates 'other' where music lives).
        outdir = str(Path(tmpd) / "demucs")
        os.makedirs(outdir, exist_ok=True)
        model = "htdemucs_6s"
        r = subprocess.run(
            ["python", "-m", "demucs", "-n", model, "-o", outdir, wav],
            capture_output=True, text=True)
        base = Path(outdir) / model / "mix"
        if not base.exists():
            _log("  [ltx] strip-music: Demucs produced no stems "
                 + (r.stderr or "")[-160:] + "; keeping original.")
            return None
        # 3) Keep NON-music stems. In htdemucs_6s: vocals (dialogue),
        #    guitar/piano are melodic (music), drums/bass are rhythmic
        #    (music), 'other' is a mix. Best effort: keep vocals + 'other'
        #    (which holds most SFX), drop drums/bass/guitar/piano.
        keep = []
        for name in ("vocals", "other"):
            p = base / f"{name}.wav"
            if p.exists():
                keep.append(str(p))
        if not keep:
            return None
        mixed = str(Path(tmpd) / "nomusic.wav")
        if len(keep) == 1:
            _sh.copyfile(keep[0], mixed)
        else:
            inp = []
            for k in keep:
                inp += ["-i", k]
            subprocess.run(
                ["ffmpeg", "-y", *inp, "-filter_complex",
                 f"amix=inputs={len(keep)}:duration=longest:normalize=0",
                 mixed], capture_output=True, text=True)
        if not Path(mixed).exists():
            return None
        # 4) Mux the music-stripped audio back onto the video.
        out = str(Path(tmpd) / "final_nomusic.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", mixed,
             "-map", "0:v", "-map", "1:a", "-c:v", "copy",
             "-c:a", "aac", "-shortest", out],
            capture_output=True, text=True)
        if Path(out).exists():
            _log("  [ltx] strip-music: music removed (dialogue + SFX kept).")
            return out
        return None
    except Exception as e:
        _log(f"  [ltx] strip-music failed ({e}); keeping original audio.")
        return None


def _ltx_concat(clip_paths, out_path, tmpd):
    """Concatenate clips into one continuous video (re-encode so clips
    with slightly different params still join cleanly; audio preserved)."""
    if len(clip_paths) == 1:
        import shutil as _sh
        _sh.copyfile(clip_paths[0], out_path)
        return
    listf = str(Path(tmpd) / "concat.txt")
    with open(listf, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")
    # Try stream-copy first (fast, lossless); fall back to re-encode.
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
         "-c", "copy", out_path], capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_path).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", out_path],
            capture_output=True, text=True, check=True)


# ── Persistent LTX worker ────────────────────────────────────────────────
# The LTX DistilledPipeline loads the model in __init__ and renders per
# __call__. We run ONE long-lived worker process that constructs the
# pipeline once, then renders every clip against the resident model —
# reading job specs as JSON lines on stdin and printing status lines back.
# It stays warm until the engine changes (LTX -> Wan) or the model is
# switched, so a whole multi-clip movie pays the ~1-2 min load ONCE.
_LTX_WORKER_SRC = r'''
import sys, json, traceback, torch
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

# ── Ampere / pre-Hopper fp8 guard ───────────────────────────────────────
# The fp8 LoRA-merge kernel (fuse_cast_fp8_weight) uses a Triton path that
# emits the `fp8e4nv` dtype, which is HOPPER-ONLY (H100/H200, sm_90). On
# A100 and every pre-Hopper GPU that kernel raises "type fp8e4nv not
# supported in this architecture", killing any render that attaches a LoRA.
# The library already has a deterministic bf16 fallback gated on
# TRITON_AVAILABLE, so on pre-Hopper cards we force that flag off — LoRAs
# then merge via the plain bf16 add (no Triton), which works everywhere.
def _guard_fp8_for_gpu():
    try:
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability()
        if major >= 9:            # Hopper+ has native fp8e4nv — keep Triton
            return
        import ltx_core.loader.kernels as _kern
        _kern.TRITON_AVAILABLE = False
        # fp8_cast imported the name directly, so patch it there too.
        try:
            import ltx_core.quantization.fp8_cast as _fc
            _fc.TRITON_AVAILABLE = False
        except Exception:
            pass
        emit({"event":"fp8_fallback",
              "cc": f"{major}.{minor}"})
    except Exception as _e:
        emit({"event":"fp8_guard_warn","error":str(_e)[:200]})

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()

def main():
    _guard_fp8_for_gpu()
    cfg = json.loads(sys.argv[1])
    offload = {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU,
               "disk": OffloadMode.DISK}.get(cfg.get("offload","cpu"),
                                             OffloadMode.CPU)
    quant = QuantizationKind.FP8_CAST.to_policy(cfg["ckpt"])
    emit({"event":"loading"})
    # LoRAs: list of {path, scale} — attached at construction. The pipeline
    # expects LoraPathStrengthAndSDOps NamedTuples (path, strength, sd_ops),
    # NOT plain tuples. sd_ops uses the Comfy renaming map so Civitai/Comfy
    # LoRA key formats convert correctly.
    lora_cfg = cfg.get("loras") or []
    lora_arg = ()
    if lora_cfg:
        from ltx_core.loader import (LoraPathStrengthAndSDOps,
                                     LTXV_LORA_COMFY_RENAMING_MAP)
        lora_arg = tuple(
            LoraPathStrengthAndSDOps(L["path"], float(L.get("scale", 1.0)),
                                     LTXV_LORA_COMFY_RENAMING_MAP)
            for L in lora_cfg if L.get("path"))
    try:
        pipe = DistilledPipeline(
            distilled_checkpoint_path=cfg["ckpt"],
            gemma_root=cfg["gemma"],
            spatial_upsampler_path=cfg["upscaler"],
            loras=lora_arg,
            quantization=quant,
            offload_mode=offload,
        )
    except Exception as e:
        # If a LoRA fails to attach, retry without LoRAs so the render
        # still works, and report which ones were dropped.
        if lora_arg:
            emit({"event":"lora_warn","error":str(e)[:300]})
            pipe = DistilledPipeline(
                distilled_checkpoint_path=cfg["ckpt"],
                gemma_root=cfg["gemma"],
                spatial_upsampler_path=cfg["upscaler"],
                loras=(),
                quantization=quant,
                offload_mode=offload,
            )
        else:
            raise
    tiling = TilingConfig.default()
    emit({"event":"ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:
            continue
        if job.get("cmd") == "shutdown":
            emit({"event":"bye"}); return
        try:
            imgs = [ImageConditioningInput(path=job["image"], frame_idx=0,
                                           strength=float(job.get("strength",0.9)),
                                           crf=int(job.get("crf",33)))]
            n = int(job["frames"])
            # Optional END frame: condition the LAST frame so the clip lands
            # on a chosen image (frame_idx = n-1).
            if job.get("end_image"):
                imgs.append(ImageConditioningInput(
                    path=job["end_image"], frame_idx=int(n)-1,
                    strength=float(job.get("end_strength",
                                           job.get("strength",0.9))),
                    crf=int(job.get("crf",33))))
            chunks = get_video_chunks_number(n, tiling)
            emit({"event":"clip_start","id":job.get("id")})
            with torch.inference_mode():
                video, audio = pipe(
                    prompt=job["prompt"], seed=int(job["seed"]),
                    height=int(job["height"]), width=int(job["width"]),
                    num_frames=n, frame_rate=float(job["fps"]),
                    images=imgs, tiling_config=tiling,
                    enhance_prompt=bool(job.get("enhance", False)),
                )
                encode_video(video=video, fps=int(job["fps"]), audio=audio,
                             output_path=job["output"],
                             video_chunks_number=chunks)
            emit({"event":"clip_done","id":job.get("id"),
                  "output":job["output"]})
        except Exception as e:
            emit({"event":"clip_error","id":job.get("id"),
                  "error":str(e)[:400],
                  "trace":traceback.format_exc()[-600:]})

if __name__ == "__main__":
    main()
'''

LTX_WORKER = {"proc": None, "cfg_key": None, "lock": threading.Lock()}
# Persistent v2v worker — kept warm across clips so the 46 GB bf16 model
# loads once, not per clip. Keyed by ckpt/iclora/loras/offload.
V2V_WORKER = {"proc": None, "wpath": None, "key": None}
# A cell re-run replaces this dict but NOT child processes — a warm v2v
# worker from the previous run would linger as an orphan holding ~46 GB.
# Kill any stale ones at startup (best-effort).
try:
    subprocess.run(["pkill", "-f", "v2v_worker_persistent.py"],
                   capture_output=True)
except Exception:
    pass

def _v2v_worker_stop():
    """Shut down the persistent v2v worker and free its VRAM."""
    proc = V2V_WORKER.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.stdin.write("shutdown\n"); proc.stdin.flush()
            proc.wait(timeout=15)
        except Exception:
            try: proc.kill()
            except Exception: pass
    V2V_WORKER.update(proc=None, key=None)


def _ltx_worker_start(p, offload):
    """Launch (or reuse) the resident LTX worker. Keyed on the model +
    offload config; if that changes, the old worker is shut down and a
    new one started. Returns the live process."""
    _v2v_worker_stop()   # reclaim the v2v model's VRAM before a normal clip
    script = f"{LTX_DIR}/ltx_worker.py"
    with open(script, "w") as f:
        f.write(_LTX_WORKER_SRC)
    # Active LoRAs (downloaded + selected) attach at construction. A change
    # in the LoRA set is part of the cfg key, so the worker restarts to
    # pick them up (the distilled pipeline takes LoRAs at build time).
    # Only LTX-engine LoRAs with a positive strength attach to the LTX
    # worker. Wan LoRAs (incl. the scale-0 lightning preset) live in the
    # same STATE dict but must NOT be handed to the LTX pipeline — their
    # key format is incompatible and a 0-strength LoRA is inert anyway.
    lora_list = [{"path": i["path"], "scale": i.get("scale", 1.0)}
                 for i in STATE["loras"].values()
                 if i.get("path") and i.get("engine") == "ltx"
                 and float(i.get("scale", 1.0)) > 0]
    cfg = {"ckpt": p["ckpt"], "gemma": p["gemma"],
           "upscaler": p["upscaler"], "offload": offload,
           "loras": lora_list}
    key = json.dumps(cfg, sort_keys=True)
    w = LTX_WORKER
    if w["proc"] is not None and w["proc"].poll() is None and w["cfg_key"] == key:
        return w["proc"]                      # already warm with same config
    _ltx_worker_stop()                        # config changed -> restart
    if lora_list:
        _log(f"  [ltx] starting worker with {len(lora_list)} LoRA(s) "
             "attached (loads once, stays warm)...")
    else:
        _log("  [ltx] starting resident model worker (loads once, then stays "
             "warm for every clip)...")
    proc = subprocess.Popen([sys.executable, script, json.dumps(cfg)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    # Wait for the "ready" event (model loaded), streaming load logs.
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("LTX worker exited during model load — "
                               "check the console for the error.")
        line = line.strip()
        try:
            ev = json.loads(line)
        except Exception:
            if line:
                _log("  [ltx] " + line[:200])
            continue
        if ev.get("event") == "loading":
            _log("  [ltx] loading model into memory (first time is slow)...")
        elif ev.get("event") == "fp8_fallback":
            _log("  [ltx] pre-Hopper GPU (cc " + str(ev.get("cc", "?"))
                 + ") — using the bf16 LoRA-merge fallback (Triton fp8 kernel "
                 "is Hopper-only). LoRAs will attach correctly.")
        elif ev.get("event") == "fp8_guard_warn":
            _log("  [ltx] fp8 guard note: " + str(ev.get("error", ""))[:200])
        elif ev.get("event") == "lora_warn":
            _log("  [ltx] a LoRA could not attach — rendering without LoRAs. "
                 + str(ev.get("error", ""))[:200])
        elif ev.get("event") == "ready":
            _log("  [ltx] model resident — clips will now render without "
                 "reloading.")
            break
    w["proc"] = proc
    w["cfg_key"] = key
    return proc

def _ltx_worker_stop():
    """Shut the resident worker down (frees the model). Called when the
    engine switches away from LTX or the model config changes."""
    w = LTX_WORKER
    proc = w.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=8)
        except Exception:
            try: proc.kill()
            except Exception: pass
        _log("  [ltx] resident model worker stopped (memory freed).")
    w["proc"] = None
    w["cfg_key"] = None


# ── LTX-2.3 launch-time preload ──────────────────────────────────────────
# Kicked off in a background thread when the UI starts, so the ~55 GB
# one-time download AND the model load both happen while the user is
# still setting up their shot. By the time they click Generate, the
# resident worker is (usually) already warm and the clip starts at once.
LTX_PRELOAD = {"state": "idle", "detail": "", "started": None, "done": None}

def _ltx_background_preload():
    """Download + load the LTX-2.3 engine at UI launch (background).

    Safe to race with a real render:
      * _ensure_ltx_ready() is serialized by LTX_SETUP_LOCK — a render
        arriving mid-download just waits on the same lock, then finds
        the .ready marker and proceeds (no duplicate downloads).
      * the worker start is serialized by LTX_WORKER["lock"], and
        _ltx_worker_start() reuses the already-warm process when the
        config key matches, so the render and the preload can never
        spawn two workers."""
    try:
        LTX_PRELOAD.update(state="downloading", started=time.time(),
                           detail="fetching LTX-2.3 models (~55 GB "
                                  "one-time; instant if already on disk)")
        _log("  [ltx] launch preload: ensuring models are on disk...")
        p = _ensure_ltx_ready()          # no-op after the first run
        LTX_PRELOAD.update(state="loading",
                           detail="loading the model into memory")
        vg = (torch.cuda.mem_get_info()[1] / 1e9
              if torch.cuda.is_available() else 0)
        off = "none" if (vg and vg >= LTX_GPU_RESIDENT_MIN_GB) else "cpu"
        with LTX_WORKER["lock"]:
            _ltx_worker_start(p, off)
        LTX_PRELOAD.update(state="ready", detail="model resident",
                           done=time.time())
        _log("  [ltx] launch preload complete — LTX-2.3 is resident; "
             "Generate starts rendering immediately.")
    except Exception as e:
        LTX_PRELOAD.update(state="error", detail=str(e)[:300])
        _log(f"  [ltx] launch preload failed ({e}) — falling back to "
             "lazy setup on the first render.")

# ── IC-LoRA video-to-video one-shot worker ───────────────────────────────
# ICLoraPipeline uses a DIFFERENT model config than the persistent
# DistilledPipeline (it attaches an IC-LoRA), so we run it as a one-shot
# subprocess per v2v clip rather than in the warm worker. Reads a JSON
# spec on argv[1], writes status lines to stdout.
_V2V_WORKER_SRC = r'''
import sys, os, gc, json, traceback, subprocess, torch
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

def emit(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()

def vram(tag):
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            emit({"event":"vram","tag":tag,
                  "alloc_gb":round(torch.cuda.memory_allocated()/1e9,2),
                  "free_gb":round(free/1e9,2),"total_gb":round(total/1e9,2)})
    except Exception:
        pass

def _guard_fp8_for_gpu():
    try:
        if not torch.cuda.is_available(): return
        major,minor = torch.cuda.get_device_capability()
        if major >= 9: return
        import ltx_core.loader.kernels as _k; _k.TRITON_AVAILABLE=False
        try:
            import ltx_core.quantization.fp8_cast as _fc; _fc.TRITON_AVAILABLE=False
        except Exception: pass
        emit({"event":"fp8_fallback","cc":f"{major}.{minor}"})
    except Exception as e:
        emit({"event":"warn","msg":str(e)[:200]})

# One registry per process holds the loaded state dicts in memory so the
# weights are NOT re-read from disk on rebuilds — the model stays resident.
_REGISTRY = StateDictRegistry()
_REF_PATCHED = False

def build_pipe(cfg, warm):
    global _REF_PATCHED
    offload = {"none":OffloadMode.NONE,"cpu":OffloadMode.CPU,
               "disk":OffloadMode.DISK}.get(cfg.get("offload","cpu"),OffloadMode.CPU)
    quant = QuantizationKind.FP8_CAST.to_policy(cfg["ckpt"])
    loras = [LoraPathStrengthAndSDOps(cfg["iclora"], 1.0,
                                      LTXV_LORA_COMFY_RENAMING_MAP)]
    for L in cfg.get("user_loras") or []:
        try:
            loras.append(LoraPathStrengthAndSDOps(
                L["path"], float(L.get("scale",1.0)),
                LTXV_LORA_COMFY_RENAMING_MAP))
        except Exception as e:
            emit({"event":"lora_warn","error":str(e)[:200]})
    vram("before_load")
    emit({"event":"warm" if warm else "loading"})
    # Fresh pipeline object EVERY job (reusing the call object corrupts
    # native CUDA state -> silent crash on job 2). The shared _REGISTRY
    # keeps the weights cached in memory, so a rebuild is cheap wiring,
    # NOT a 46 GB reload.
    pipe = ICLoraPipeline(
        distilled_checkpoint_path=cfg["ckpt"],
        spatial_upsampler_path=cfg["upscaler"],
        gemma_root=cfg["gemma"],
        loras=loras,
        quantization=quant,
        offload_mode=offload,
        registry=_REGISTRY,
    )
    emit({"event":"ready"})
    vram("after_build")
    gc.collect(); torch.cuda.empty_cache()
    # Route the reference VAE encode through the tiled path (library
    # hardcodes it untiled). Patch once per process.
    if not _REF_PATCHED:
        import ltx_pipelines.ic_lora as _icl
        from ltx_pipelines.iclora_utils import (
            append_ic_lora_reference_video_conditionings as _orig_ref)
        _t = TilingConfig.default()
        def _tiled_ref(*a, **k):
            k["tiling_config"] = _t
            return _orig_ref(*a, **k)
        _icl.append_ic_lora_reference_video_conditionings = _tiled_ref
        _REF_PATCHED = True
    return pipe, TilingConfig.default()


def _load_key(cfg):
    # Reload only when something that affects the loaded weights changes.
    return json.dumps({"ckpt":cfg.get("ckpt"),"iclora":cfg.get("iclora"),
                       "offload":cfg.get("offload"),
                       "loras":cfg.get("user_loras")}, sort_keys=True)


def render_job(pipe, tiling, cfg):
    # Render each segment against the resident pipeline. Segment 0 may
    # carry a user first image; every later segment chains from the
    # PREVIOUS segment's near-end frame (skip the smeary last ~0.3s).
    segs = cfg["segments"]
    prev_out = None
    for si, seg in enumerate(segs):
        emit({"event":"segment","i":si+1,"n":len(segs)})
        vram("segment_%d_start"%(si+1))
        first = seg.get("first_image")
        if not first and prev_out:
            # Chain from the previous segment's tail — but prefer a frame
            # where a FACE is visible, so identity survives the boundary
            # (a chain frame without the subject makes the next segment
            # re-invent the character).
            cands = []
            for off in ("-0.30","-0.60","-0.90","-1.20"):
                p = prev_out + f".chain{off}.png"
                subprocess.run(["ffmpeg","-y","-sseof",off,"-i",prev_out,
                                "-frames:v","1","-q:v","2",p],
                               capture_output=True)
                if os.path.exists(p):
                    cands.append(p)
            pick = cands[0] if cands else None
            try:
                import cv2
                casc = cv2.CascadeClassifier(
                    cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
                for p in cands:   # nearest-to-end first
                    img = cv2.imread(p)
                    if img is None: continue
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(casc.detectMultiScale(g,1.1,4)) > 0:
                        pick = p
                        emit({"event":"chain_face","found":True})
                        break
            except Exception:
                pass
            first = pick
        seg_frames = int(seg["frames"])
        images = []
        if first:
            images.append(ImageConditioningInput(first, 0,
                              float(cfg.get("first_strength",0.9)), 8))
        # DRIFT RE-ANCHOR: if the chain frame has drifted too far from the
        # ORIGINAL start image, gently pull the scene back with a LOW-
        # strength anchor mid-segment (an abrupt frame-0 re-anchor would
        # jump-cut at the boundary; mid-segment at 0.35 corrects appearance
        # without dictating pose).
        anchor = cfg.get("anchor_image")
        if si > 0 and first and anchor and os.path.exists(anchor):
            try:
                import cv2
                a = cv2.imread(anchor); b = cv2.imread(first)
                if a is not None and b is not None:
                    a = cv2.resize(a,(128,128)); b = cv2.resize(b,(128,128))
                    ha = cv2.calcHist([cv2.cvtColor(a,cv2.COLOR_BGR2HSV)],
                                      [0,1],None,[32,32],[0,180,0,256])
                    hb = cv2.calcHist([cv2.cvtColor(b,cv2.COLOR_BGR2HSV)],
                                      [0,1],None,[32,32],[0,180,0,256])
                    cv2.normalize(ha,ha); cv2.normalize(hb,hb)
                    corr = float(cv2.compareHist(ha,hb,cv2.HISTCMP_CORREL))
                    if corr < 0.45:
                        mid = max(8, (seg_frames // 2) // 8 * 8)
                        images.append(ImageConditioningInput(
                            anchor, mid, 0.35, 8))
                        emit({"event":"reanchor","corr":round(corr,2),
                              "frame":mid})
            except Exception:
                pass
        # END-FRAME ANCHOR: a user-set end image pins the final frame of
        # the LAST segment (first+last-frame anchoring).
        if si == len(segs) - 1 and cfg.get("end_image")                 and os.path.exists(cfg["end_image"]):
            images.append(ImageConditioningInput(
                cfg["end_image"], seg_frames - 1, 0.8, 8))
        video_iter, audio = pipe(
            prompt=cfg["prompt"], seed=int(cfg.get("seed",42)),
            height=cfg["height"], width=cfg["width"],
            num_frames=seg_frames, frame_rate=cfg["fps"],
            images=images,
            video_conditioning=[(seg["control"], float(cfg.get("strength",1.0)))],
            conditioning_attention_strength=float(cfg.get("attn_strength",1.0)),
            enhance_prompt=bool(cfg.get("enhance",False)),
            tiling_config=tiling,
        )
        emit({"event":"encoding"})
        chunks = get_video_chunks_number(seg_frames, tiling)
        encode_video(video=video_iter, fps=int(round(cfg["fps"])),
                     audio=audio, output_path=seg["output"],
                     video_chunks_number=chunks,
                     crf=int(cfg.get("crf",20)))
        prev_out = seg["output"]
        del video_iter, audio
        gc.collect(); torch.cuda.empty_cache()
        vram("segment_%d_done"%(si+1))
    emit({"event":"done"})
    emit({"event":"done"})


def main():
    _guard_fp8_for_gpu()
    pipe = None; tiling = None; key = None
    # Persistent service: read one job per stdin line, keep the model warm
    # across clips, reload only when the load-key changes.
    if len(sys.argv) > 1 and sys.argv[1] not in ("-","serve"):
        # Back-compat: a single job passed as argv (one-shot).
        cfg = json.loads(sys.argv[1])
        pipe, tiling = build_pipe(cfg, warm=False)
        render_job(pipe, tiling, cfg)
        return
    built_once = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "shutdown":
            break
        try:
            cfg = json.loads(line)
        except Exception:
            continue
        try:
            # Rebuild a FRESH pipeline every job. After the first build the
            # registry has the weights cached, so subsequent builds reuse
            # the resident weights (fast) instead of reloading from disk.
            resident = (cfg.get("offload","cpu") == "none")
            pipe, tiling = build_pipe(cfg, warm=built_once and resident)
            built_once = True
            render_job(pipe, tiling, cfg)
            # Always release the per-job pipeline objects.
            try:
                pipe.stage_1 = None; pipe.stage_2 = None
            except Exception:
                pass
            del pipe
            # RESIDENT (big card): keep the registry's cached weights so the
            # next clip skips the reload. STREAMING (small card / L4): drop
            # the cache too so VRAM+RAM return to baseline between clips —
            # the model can't stay resident there anyway.
            if not resident:
                try:
                    _REGISTRY.clear()
                except Exception:
                    pass
            gc.collect(); torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize(); torch.cuda.ipc_collect()
            except Exception:
                pass
            vram("after_free")
        except Exception as e:
            emit({"event":"error","error":str(e),
                  "trace":traceback.format_exc()})


try:
    # Mirrors @torch.inference_mode() on the official ic_lora CLI main().
    # Without it, autograd retains every fp8->bf16 upcast weight copy and
    # all activations "for backward" — ~93 GB by the end of ONE forward
    # pass (the observed OOM on both 40 GB and 100 GB cards).
    with torch.inference_mode():
        main()
except Exception as e:
    emit({"event":"error","error":str(e),"trace":traceback.format_exc()[-1500:]})
    sys.exit(1)
'''

def _run_v2v_clip(clip, ci, out_path, tmpd, fps, seed, params, job):
    """Render one video-to-video clip via the IC-LoRA one-shot worker.
    control_video (data URL or path) supplies motion; first_image supplies
    appearance; control_type picks raw/canny/depth; original audio is
    copied on when requested."""
    p = _ensure_ltx_ready(job)
    # 1) Materialise the control video locally.
    cv = clip["control_video"]
    ctrl_raw = str(Path(tmpd) / f"ctrl_{ci}_src.mp4")
    if isinstance(cv, str) and cv.startswith("data:"):
        with open(ctrl_raw, "wb") as f:
            f.write(base64.b64decode(cv.split(",", 1)[1]))
    elif isinstance(cv, str) and os.path.exists(cv):
        import shutil as _sh; _sh.copyfile(cv, ctrl_raw)
    else:
        raise RuntimeError("v2v clip has no usable control video")
    # 1b) Auto-split slice: if this clip is one window of a longer control
    # video, cut just its time-slice so each chained clip drives from its
    # own portion (the frontend split a long control into N clips).
    split = clip.get("control_split")
    if split:
        try:
            idx = int(split.get("index", 0))
            per = float(split.get("per", 0)) or 0
            s0 = float(split.get("start", idx * per))
            ln = float(split.get("len", per)) or per
            if ln > 0:
                sliced = str(Path(tmpd) / f"ctrl_{ci}_slice.mp4")
                # LOSSLESS slice: the control video drives the generation,
                # so it must not be degraded. Output seeking (-ss after -i)
                # is frame-accurate; -qp 0 is lossless x264; audio is
                # stream-copied so per-clip original audio stays pristine.
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", ctrl_raw,
                     "-ss", f"{s0:.3f}", "-t", f"{ln:.3f}",
                     "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                     "-pix_fmt", "yuv444p", "-c:a", "copy", sliced],
                    capture_output=True, text=True)
                if not (Path(sliced).exists() and os.path.getsize(sliced) > 1000):
                    # Some players/filters dislike yuv444; fall back to
                    # near-lossless 4:2:0 rather than fail.
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-i", ctrl_raw,
                         "-ss", f"{s0:.3f}", "-t", f"{ln:.3f}",
                         "-c:v", "libx264", "-crf", "8", "-preset", "veryfast",
                         "-pix_fmt", "yuv420p", "-c:a", "copy", sliced],
                        capture_output=True, text=True)
                if Path(sliced).exists() and os.path.getsize(sliced) > 1000:
                    ctrl_raw = sliced
                    _log(f"  [ltx] v2v clip {ci+1}: control slice "
                         f"{idx+1}/{split.get('count')} "
                         f"({s0:.1f}\u2013{s0+ln:.1f}s).")
        except Exception as e:
            _log(f"  [ltx] v2v: slice failed ({e}); using full control.")
    # 2) Pick control model + preprocessing by control_type.
    ctype = clip.get("control_type", "raw")     # raw|canny|depth|motion_track
    iclora_kind = "motion_track" if ctype == "motion_track" else "union"
    iclora_path = _ensure_iclora(iclora_kind, job)
    pre_mode = {"raw": "raw", "canny": "canny", "depth": "depth",
                "motion_track": "raw"}.get(ctype, "raw")
    job.update(stage=f"v2v: preparing {ctype} control")
    control_video = _ltx_preprocess_control(ctrl_raw, pre_mode, tmpd, job)
    # 3) First-frame appearance image (optional; may be edited by the user).
    first_png = None
    if clip.get("start_image_path") and os.path.exists(clip["start_image_path"]):
        first_png = clip["start_image_path"]
    fi = clip.get("start_image")
    if fi and not first_png:
        try:
            raw = base64.b64decode(fi.split(",")[-1])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            first_png = str(Path(tmpd) / f"v2v_first_{ci}.png")
            im.save(first_png)
        except Exception as e:
            _log(f"  [ltx] v2v clip {ci+1}: bad first image ({e}); "
                 "motion-only.")
            first_png = None
    # End-frame anchor (optional): a user-set end image pins the LAST frame
    # of the clip — literal first+last-frame anchoring for v2v.
    end_png = None
    ei = clip.get("end_image")
    if ei:
        try:
            raw = base64.b64decode(ei.split(",")[-1])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            end_png = str(Path(tmpd) / f"v2v_end_{ci}.png")
            im.save(end_png)
        except Exception as e:
            _log(f"  [ltx] v2v clip {ci+1}: bad end image ({e}); ignoring.")
            end_png = None
    # 4) Dims from the first frame (or the control video's first frame).
    if first_png:
        dim_img = Image.open(first_png).convert("RGB")
    else:
        ff = str(Path(tmpd) / f"v2v_ctrlfirst_{ci}.png")
        _ltx_first_frame(control_video, ff)
        dim_img = Image.open(ff).convert("RGB")
    profile = params.get("ltx_profile", "480P")
    W1, H1 = _ltx_stage1_dims(dim_img, profile)
    # IC-LoRA dimension constraint — VERIFIED from the VAE source + crash:
    # the reference video is encoded at final/4 (stage1=final/2, ref0.5
    # halves again), and the VAE encoder is patchify(4) + THREE spatial
    # halvings, so the reference must be divisible by 4*2^3=32, i.e. the
    # FINAL dims divisible by 128. (The einops crash: 576 -> ref 144 ->
    # 36 -> 18 -> 9, dies at halving 3. 640 -> ref 160 -> 5, fine.)
    unit = 128
    W1 = max(unit, ((W1 + unit - 1) // unit) * unit)
    H1 = max(unit, ((H1 + unit - 1) // unit) * unit)
    _log(f"  [ltx] v2v dims: final {W1}x{H1}, stage-1 {W1//2}x{H1//2}, "
         f"reference {W1//4}x{H1//4}.")
    # v2v LENGTH: the IC-LoRA reference video is positionally aligned to the
    # output frame-by-frame, so the output should match the CONTROL VIDEO's
    # length. Probe it and use that count (snapped to LTX's 8k+1 rule),
    # unless the user explicitly overrode this clip's length.
    ctrl_frames = None
    ctrl_fps = None
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames,avg_frame_rate",
             "-of", "json", control_video],
            capture_output=True, text=True)
        st = (json.loads(pr.stdout or "{}").get("streams") or [{}])[0]
        ctrl_frames = int(st.get("nb_read_frames") or 0) or None
        fr = st.get("avg_frame_rate") or "0/1"
        num, _, den = fr.partition("/")
        v = (float(num) / float(den or 1)) if float(den or 1) else 0.0
        if 8.0 <= v <= 60.0:
            ctrl_fps = v
    except Exception:
        ctrl_frames = ctrl_fps = None
    # The IC-LoRA reference is paired frame-i -> frame-i, so the OUTPUT must
    # run at the control video's fps or motion plays fast/slow and any
    # copied original audio drifts. Adopt the control's fps when detected.
    if ctrl_fps and abs(ctrl_fps - fps) > 0.01:
        _log(f"  [ltx] v2v: adopting the control video's frame rate "
             f"({ctrl_fps:.3f} fps, UI was {fps}).")
        fps = ctrl_fps
    if clip.get("frames_overridden") or not ctrl_frames:
        n_req = int(clip.get("frames", 121))
    else:
        n_req = ctrl_frames
        _log(f"  [ltx] v2v: matching output length to the control video "
             f"({ctrl_frames} frames @ {fps:.3f} fps = "
             f"{ctrl_frames / fps:.1f}s).")
    n = max(9, ((n_req - 1) // 8) * 8 + 1)
    prompt = (clip.get("prompt") or params.get("prompt") or "").strip()
    if not prompt:
        _log("  [ltx] v2v: no prompt — following the video + first frame "
             "for appearance.")
    # 5) Run the one-shot worker.
    # CRITICAL (traced from the 'attn1.to_gate_logits.input_scale' crash):
    # the v2v pipeline must run the SAME offload mode the working distilled
    # path resolves to. Passing "auto" through unresolved dropped v2v into
    # CPU block-streaming, whose layout builder KeyErrors on the fp8
    # checkpoint's input_scale params after the bf16 LoRA merge. Resolve
    # auto exactly like the main path (resident when VRAM fits). And since
    # the resident distilled worker is already holding ~38 GB, stop it
    # first so the IC-LoRA pipeline can load resident without OOM — the
    # next non-v2v clip restarts it automatically (idempotent start).
    vram_gb = (torch.cuda.mem_get_info()[1] / 1e9
               if torch.cuda.is_available() else 0)
    # v2v offload — grounded in the loader source: builder.build() loads the
    # ENTIRE checkpoint to the GPU at native dtype BEFORE quantization, so
    # the 46 GB bf16 file needs ~46 GB (transformer) + Gemma at load when
    # resident. Resident therefore requires a ~80 GB+ card; below that we
    # use CPU block-streaming — the standard low-VRAM configuration, and
    # safe now: the earlier streaming KeyError was caused by the fp8
    # checkpoint's input_scale keys, which the bf16 file doesn't have.
    offload_pref = params.get("ltx_offload", "auto")
    if offload_pref in ("auto", "none"):
        # Resident needs to hold the full ~46 GB bf16 transformer + Gemma +
        # activations. 60 GB is a safe floor with real headroom; below it we
        # stream. (Was 80, which made 80 GB-class cards that enumerate as
        # ~79.x usable drop needlessly to slow CPU streaming.)
        v2v_offload = "none" if vram_gb >= 60 else "cpu"
        # Tiny cards (L4-class, ~24 GB) can't hold the model in VRAM *or*
        # reliably in Colab CPU RAM — fall back to DISK streaming there.
        if vram_gb < 30:
            v2v_offload = "disk"
            _log(f"  [ltx] v2v: {vram_gb:.0f} GB VRAM (L4-class) — using "
                 "DISK streaming (slowest, but fits). An 80 GB+ card runs "
                 "this resident and far faster.")
        if offload_pref == "none" and v2v_offload != "none":
            _log(f"  [ltx] v2v: overriding offload=none — {vram_gb:.0f} GB "
                 "VRAM can't hold the bf16 checkpoint resident (needs ~80 GB+); "
                 "using CPU streaming.")
    else:
        v2v_offload = offload_pref
    # Fetch the bf16 checkpoint FIRST (download can take a while — no
    # reason to kill the resident worker before it's on disk), then free
    # the distilled model's VRAM for the IC-LoRA run.
    ckpt_bf16 = _ensure_ltx_bf16(job)
    _ltx_worker_stop()   # free the distilled model's VRAM for the IC-LoRA run
    _log(f"  [ltx] v2v: running {'RESIDENT on GPU' if v2v_offload=='none' else 'with CPU streaming'} "
         f"({vram_gb:.0f} GB VRAM).")
    # ── SEGMENTATION (the OOM fix) ──────────────────────────────────────
    # IC-LoRA holds BOTH the output tokens AND the reference video's tokens
    # in context, so long control videos (this crash: 497 frames) cannot fit
    # one pass on 42 GB — the proven single-pass budget from your successful
    # renders is ~121-137 frames WITHOUT reference tokens. We render in
    # chained segments of <=121 frames: each segment's control is the
    # matching time-slice of the control video, and each segment after the
    # first starts from the previous segment's last frame (same continuation
    # mechanism as multi-clip movies). The model loads ONCE for all segments.
    # Segment size is VRAM-adaptive: on a 40 GB card the resident stack +
    # IC-LoRA leaves only ~0.5 GB headroom (a real run died 30 MiB short at
    # 121 frames), so drop to 89 frames there; 121 needs ~41 GB+.
    # Segment size = the model's attention context; bigger segments mean
    # fewer boundaries and less drift. Scaled to what the card can hold.
    # Segment size = attention context per pass. User-controllable (default
    # 20s); we still cap by what the card can hold so a big setting on a
    # small card can't OOM. frames = seconds * fps, snapped to 8k+1.
    seg_secs = float(params.get("v2v_seg_seconds", 20) or 20)
    vram_cap = 241 if vram_gb >= 60 else (121 if vram_gb >= 41 else 89)
    seg_req = max(9, int(round(seg_secs * fps / 8)) * 8 + 1)
    V2V_SEG = min(seg_req, vram_cap)
    if seg_req > vram_cap:
        _log(f"  [ltx] v2v: requested {seg_secs:.0f}s/segment exceeds this "
             f"GPU's safe budget — capping at {vram_cap} frames "
             f"(~{vram_cap/fps:.0f}s).")
    step = V2V_SEG - 1                # 1-frame overlap for the chained start
    n_segs = 1 if n <= V2V_SEG else ((n - 1 + step - 1) // step)
    if n_segs > 1:
        _log(f"  [ltx] v2v: {n} frames exceeds the single-pass VRAM budget "
             f"({V2V_SEG}) — rendering {n_segs} chained segments.")
    segments = []
    for si in range(n_segs):
        s0 = si * step
        seg_n = min(V2V_SEG, n - s0)
        seg_n = max(9, ((seg_n - 1) // 8) * 8 + 1)
        seg_ctrl = control_video
        if n_segs > 1:
            seg_ctrl = str(Path(tmpd) / f"ctrl_{ci}_s{si}.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", control_video,
                 "-ss", f"{s0 / fps:.4f}", "-frames:v", str(seg_n),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", seg_ctrl],
                capture_output=True, text=True)
            if not Path(seg_ctrl).exists():
                seg_ctrl = control_video
        segments.append({"control": seg_ctrl, "frames": seg_n,
                         "first_image": (first_png if si == 0 else None),
                         "output": str(Path(tmpd) / f"v2v_{ci}_s{si}.mp4")})
    # Same LoRA selection as the resident worker: LTX-engine LoRAs with a
    # positive strength ride along (identity/style LoRAs keep the character
    # consistent when the face leaves and re-enters the frame).
    user_loras = [{"path": i["path"], "scale": i.get("scale", 1.0)}
                  for i in STATE["loras"].values()
                  if i.get("path") and i.get("engine") == "ltx"
                  and float(i.get("scale", 1.0)) > 0]
    if user_loras:
        _log(f"  [ltx] v2v: attaching {len(user_loras)} user LoRA(s) "
             "alongside the IC-LoRA.")
    spec = {
        "ckpt": ckpt_bf16, "gemma": p["gemma"], "upscaler": p["upscaler"],
        "iclora": iclora_path, "user_loras": user_loras,
        "anchor_image": first_png, "end_image": end_png,
        "first_strength": 0.9,
        "prompt": prompt, "seed": seed, "height": H1, "width": W1,
        "fps": fps, "crf": int(params.get("ltx_crf", 20) or 20),
        "strength": float(clip.get("control_strength", 1.0)),
        "attn_strength": float(clip.get("attn_strength", 1.0)),
        "offload": v2v_offload,
        "enhance": bool(params.get("ltx_enhance", False)),
        "segments": segments,
    }
    wpath = f"{LTX_DIR}/v2v_worker_persistent.py"
    # ALWAYS rewrite (a redeploy must ship worker fixes; a stale on-disk
    # script once kept an old bug alive across cell updates).
    with open(wpath, "w") as f:
        f.write(_V2V_WORKER_SRC)
    _log(f"  [ltx] v2v clip {ci+1}: {ctype} control, {n} frames @ {fps}fps, "
         f"{W1*2}x{H1*2}, {n_segs} segment(s).")
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Load-key: reload the model only when the checkpoint / IC-LoRA / user
    # LoRAs / offload change. Same key -> reuse the warm process.
    import hashlib as _hl
    load_key = json.dumps({"ckpt": spec["ckpt"], "iclora": spec["iclora"],
                           "offload": spec["offload"],
                           "loras": spec["user_loras"],
                           "src": _hl.md5(_V2V_WORKER_SRC.encode()).hexdigest()},
                          sort_keys=True)
    proc = V2V_WORKER["proc"]
    alive = proc is not None and proc.poll() is None
    if not alive or V2V_WORKER["key"] != load_key:
        if alive:
            try:
                proc.stdin.write("shutdown\n"); proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                try: proc.kill()
                except Exception: pass
        proc = subprocess.Popen(
            [sys.executable, wpath, "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        V2V_WORKER.update(proc=proc, wpath=wpath, key=load_key)
        _log("  [ltx] v2v: starting a warm worker (model loads once, then "
             "stays resident across clips).")
    else:
        _log("  [ltx] v2v: reusing the resident model (fresh pipeline, "
             "cached weights \u2014 no reload).")
    # Send this clip's job.
    proc.stdin.write(json.dumps(spec) + "\n")
    proc.stdin.flush()
    err = None
    _v2v_seg_i, _v2v_seg_n = 1, len(segments)
    _last_step_line = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            # The pipeline prints tqdm bars as raw text (e.g. "40%|...| 2/5").
            # Show them in the CONSOLE and drive live per-step UI progress.
            m = re.search(r"(\d+)/(\d+)\s*\[", line)
            if m:
                cur_s, tot_s = int(m.group(1)), int(m.group(2))
                is_denoise = ("it/s" in line or "s/it" in line)
                phase = "denoising" if is_denoise else "encoding"
                seg_lbl = (f"segment {_v2v_seg_i}/{_v2v_seg_n} · "
                           if _v2v_seg_n > 1 else "")
                frac = (cur_s / tot_s) if tot_s else 0
                base = 15 + int(70 * ((_v2v_seg_i - 1) + frac)
                               / max(1, _v2v_seg_n))
                job.update(stage=f"v2v: {seg_lbl}{phase} {cur_s}/{tot_s}",
                           progress=min(88, base))
                # Keep it in the console too (throttled to changes).
                if line != _last_step_line:
                    _log(f"  [ltx:v2v] {phase} {cur_s}/{tot_s}")
                    _last_step_line = line
            else:
                _log(f"  [ltx:v2v] {line[:200]}")
            continue
        et = ev.get("event")
        if et == "loading":
            _log("  [ltx] v2v: loading IC-LoRA model (first time is slow)...")
        elif et == "warm":
            _log("  [ltx] v2v: warm start — model already resident.")
        elif et == "fp8_fallback":
            _log(f"  [ltx] v2v: pre-Hopper GPU (cc {ev.get('cc')}) — bf16 "
                 "LoRA-merge fallback.")
        elif et == "ready":
            _log("  [ltx] v2v: model ready, rendering...")
        elif et == "segment":
            si, sn = ev.get("i", 1), ev.get("n", 1)
            _v2v_seg_i, _v2v_seg_n = si, sn
            _log(f"  [ltx] v2v: segment {si}/{sn}...")
            job.update(stage=f"v2v: segment {si}/{sn} starting",
                       progress=15 + int(70 * (si - 1) / max(1, sn)))
        elif et == "encoding":
            _log("  [ltx] v2v: encoding segment...")
        elif et == "reanchor":
            _log(f"  [ltx] v2v: scene drifted (similarity "
                 f"{ev.get('corr')}), re-anchoring to the original start "
                 f"image at frame {ev.get('frame')}.")
        elif et == "vram":
            _log(f"  [ltx] v2v VRAM [{ev.get('tag')}]: "
                 f"{ev.get('alloc_gb')} GB allocated, "
                 f"{ev.get('free_gb')} GB free of {ev.get('total_gb')} GB.")
        elif et == "error":
            err = ev.get("error", "unknown")
            _log(f"  [ltx] v2v ERROR: {err}\n{ev.get('trace','')}")
            break
        elif et == "done":
            break        # this clip finished; leave the process warm
    # Silent death: the for-loop over proc.stdout ended (EOF) without a
    # 'done' or 'error' event, and nothing was produced. poll() can still
    # read None on a zombie, so we key off the missing output, not poll().
    if not err:
        got_any = any(Path(s["output"]).exists() for s in segments)
        if not got_any and not clip.get("_v2v_retried"):
            rc = proc.poll()
            _log(f"  [ltx] v2v worker ended with no output (exit {rc}) — "
                 "respawning a fresh worker and retrying this clip once.")
            try:
                if proc.poll() is None: proc.kill()
            except Exception:
                pass
            V2V_WORKER.update(proc=None, key=None)
            clip["_v2v_retried"] = True
            return _run_v2v_clip(clip, ci, out_path, tmpd, fps, seed,
                                 params, job)
        if not got_any:
            err = f"v2v worker crashed (exit {proc.poll()})"
            V2V_WORKER.update(proc=None, key=None)
    seg_outs = [s["output"] for s in segments if Path(s["output"]).exists()]
    if err or not seg_outs:
        raise RuntimeError(f"v2v render failed: {err or 'no output'}")
    if len(seg_outs) < len(segments):
        _log(f"  [ltx] v2v: only {len(seg_outs)}/{len(segments)} segments "
             "rendered — stitching what completed.")
    # Stitch segments into the clip output.
    _ltx_concat(seg_outs, out_path, tmpd)
    if not Path(out_path).exists():
        raise RuntimeError("v2v segment stitch failed")
    # 6) Copy the source video's original audio onto the output if asked
    #    (the FULL original track over the stitched clip — continuous audio).
    if clip.get("copy_source_audio"):
        job.update(stage="v2v: copying original audio")
        newp = _ltx_copy_source_audio(ctrl_raw, out_path, tmpd)
        if newp != out_path and Path(newp).exists():
            import shutil as _sh; _sh.copyfile(newp, out_path)


def _ltx_first_frame(video_path, out_png):
    """Grab the FIRST frame of a video as a PNG (for v2v dims / appearance)."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", "-q:v", "2",
         out_png], capture_output=True, text=True)
    if not Path(out_png).exists():
        raise RuntimeError("could not read the control video's first frame — "
                           + (r.stderr or "")[-200:])


def _run_ltx_job(job_id, params):
    """Render a CHAINED SEQUENCE of LTX-2.3 clips into one continuous
    video via the RESIDENT worker (model loaded once, kept warm). Clip 1
    starts from the uploaded image; each next clip starts from the LAST
    FRAME of the previous one. Each clip has its own prompt (its assigned
    dialogue) and length. A fixed seed is used across all clips for voice
    and character consistency. Clips are concatenated at the end."""
    job = jobs[job_id]
    job.update(status="running", stage="preparing LTX engine", progress=2)
    p = _ensure_ltx_ready(job)
    if job.get("cancel"):
        raise _JobCancelled()

    # LTX needs the GPU — drop a resident Wan pipeline first.
    with STATE["load_lock"]:
        if STATE["pipe"] is not None:
            _log("  [ltx] freeing the Wan pipeline to hand the GPU to LTX.")
            STATE["pipe"] = None
            STATE["pipe_key"] = None
            STATE["residency"] = "unknown"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clips_peek = params.get("clips") or []
    _img_src = (params.get("image")
                or (clips_peek[0].get("start_image") if clips_peek else None))
    _c1_v2v = bool(clips_peek and clips_peek[0].get("control_video") is not None)
    seed_img = None
    if _img_src:
        raw = base64.b64decode(_img_src.split(",")[-1])
        seed_img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif not _c1_v2v:
        raise ValueError("Clip 1 needs a start image — set one in its "
                         "timeline editor (or give it a control video).")
    profile = params.get("profile", "480P")

    # ---- clip plan --------------------------------------------------------
    clips = params.get("clips")
    if not clips:
        clips = [{"prompt": params.get("prompt", ""),
                  "frames": int(params.get("frames", 121))}]
    n_clips = len(clips)
    fps = max(1, int(params["fps"]))
    # FIXED seed across every clip — this is what keeps the synthesized
    # voices and characters consistent from clip to clip.
    seed = int(params["seed"])

    vram_gb = (torch.cuda.mem_get_info()[1] / 1e9
               if torch.cuda.is_available() else 0)
    strength = min(1.0, max(0.1, float(params.get("ltx_strength", 0.9))))
    try:
        crf = max(0, min(51, int(params.get("ltx_crf", 33))))
    except (TypeError, ValueError):
        crf = 33
    enhance = bool(params.get("ltx_enhance"))
    offload_pref = params.get("ltx_offload", "auto")
    if offload_pref == "auto":
        if vram_gb and vram_gb >= LTX_GPU_RESIDENT_MIN_GB:
            offload = "none"   # ~30 GB model fits -> stays on the GPU
        else:
            offload = "cpu"    # stream from RAM; worker keeps it warm
    else:
        offload = offload_pref
    if offload == "none":
        _log(f"  [ltx] model will stay RESIDENT ON THE GPU "
             f"({vram_gb:.0f} GB VRAM) — fastest, no per-clip streaming.")
    else:
        _log(f"  [ltx] weight streaming from CPU RAM "
             f"({'forced' if offload_pref=='cpu' else f'{vram_gb:.0f} GB VRAM'}) "
             "— the resident worker keeps the model loaded between clips, "
             "so clips don't reload it.")

    # ---- start / reuse the resident worker --------------------------------
    with LTX_WORKER["lock"]:
        # Lazy start: if the FIRST clip is v2v it would stop the resident
        # model immediately — normal clips (re)start it themselves.
        _first_v2v = bool(clips and clips[0].get("control_video") is not None)
        proc = None if _first_v2v else _ltx_worker_start(p, offload)

        tmpd = tempfile.mkdtemp(prefix="ltx_")
        cur_img = seed_img
        clip_paths = []
        _log(f"  [ltx] rendering a {n_clips}-clip movie on the resident "
             f"model (seed {seed} fixed across clips for consistency).")

        for ci, clip in enumerate(clips):
            if job.get("cancel"):
                raise _JobCancelled()
            # ── VIDEO-TO-VIDEO clip (IC-LoRA) ──────────────────────────
            # If this clip carries a control video, render it via the
            # ICLoraPipeline one-shot (motion/structure from the video,
            # appearance from the first-frame image, optional original
            # audio). It produces a finished clip we drop straight into the
            # sequence, then continue to the next clip.
            cvv = clip.get("control_video")
            if cvv is not None and isinstance(cvv, int):
                try:
                    clip["control_video"] = (params.get("control_sources")
                                             or [])[cvv]
                except Exception:
                    clip["control_video"] = None
            if clip.get("control_video"):
                # v2v -> v2v continuity: without a user start image, chain
                # from the PREVIOUS clip's near-last frame (same rule as
                # normal clips) so identity survives the clip boundary.
                if ci > 0 and not clip.get("start_image") and clip_paths:
                    try:
                        chain_png = str(Path(tmpd) / f"v2v_chain_{ci}.png")
                        _ltx_last_frame(clip_paths[-1], chain_png, back=6)
                        clip["start_image_path"] = chain_png
                    except Exception as e:
                        _log(f"  [ltx] v2v: chain frame failed ({e}); "
                             "starting unanchored.")
                clip_lbl = f"clip {ci+1}/{n_clips}"
                job.update(stage=f"LTX {clip_lbl}: video-to-video",
                           progress=15 + int(75 * ci / n_clips))
                out_path = str(Path(tmpd) / f"clip_{ci}.mp4")
                _run_v2v_clip(clip, ci, out_path, tmpd, fps, seed,
                              params, job)
                if not Path(out_path).exists():
                    raise RuntimeError(f"LTX {clip_lbl} (v2v) produced no output.")
                TAIL = 6 if n_clips > 1 else 0
                if TAIL:
                    trimmed = str(Path(tmpd) / f"clip_{ci}_t.mp4")
                    _ltx_trim_tail(out_path, trimmed, TAIL, fps)
                    clip_paths.append(trimmed)
                else:
                    clip_paths.append(out_path)
                # Post this finished v2v clip to the stage immediately.
                try:
                    with open(out_path, "rb") as _cf:
                        _cb64 = base64.b64encode(_cf.read()).decode()
                    job.update(stage=f"LTX clip {ci+1}/{n_clips} done",
                               partial=f"data:video/mp4;base64,{_cb64}",
                               partial_index=ci + 1, partial_total=n_clips)
                except Exception:
                    pass
                if ci < n_clips - 1:
                    nxt = str(Path(tmpd) / f"start_{ci+1}.png")
                    _ltx_last_frame(out_path, nxt, back=6)
                    cur_img = Image.open(nxt).convert("RGB")
                continue
            # If this clip has an explicit start image (set via Auto Next
            # Scene or a manual replace), use it; otherwise cur_img carries
            # the chained last frame from the previous clip.
            si = clip.get("start_image")
            is_chained = (ci > 0 and not si)   # start came from the prev clip
            if si:
                try:
                    sraw = base64.b64decode(si.split(",")[-1])
                    cur_img = Image.open(io.BytesIO(sraw)).convert("RGB")
                    _log(f"  [ltx] clip {ci+1}: using its own start image "
                         "(not the chained frame).")
                except Exception as e:
                    _log(f"  [ltx] clip {ci+1}: bad start image ({e}); "
                         "falling back to the chained frame.")
                    is_chained = (ci > 0)
            W1, H1 = _ltx_stage1_dims(cur_img, profile)
            n_req = int(clip.get("frames", 121))
            n = max(9, ((n_req - 1) // 8) * 8 + 1)
            prompt = (clip.get("prompt") or params.get("prompt") or "").strip()
            # Continuation guardrail: for a chained clip, prepend an explicit
            # "picks up exactly where the last ended" cue so the prompt and the
            # last-frame conditioning don't fight (a top cause of drift).
            if is_chained:
                prompt = ("This shot continues seamlessly from the previous "
                          "one, beginning on the exact same frame with the "
                          "same characters, wardrobe, lighting, colour and "
                          "framing; the motion carries on without any cut or "
                          "reset. " + prompt).strip()
            img_path = str(Path(tmpd) / f"start_{ci}.png")
            cur_img.save(img_path)
            out_path = str(Path(tmpd) / f"clip_{ci}.mp4")

            # Conditioning strengths. A CHAINED start frame is anchored hard
            # (~0.97) so the new clip truly begins where the last ended — a
            # weak anchor is the main reason continuations drift. A user-set
            # start image uses the normal strength.
            start_strength = 0.97 if is_chained else strength

            # Optional END frame for this clip — the render lands on it.
            # If this clip is NOT the last, its end frame also becomes the
            # next clip's start, so anchor it strongly (~0.85) for a clean
            # first-last-frame handoff (per LTX FLF guidance).
            end_path = None
            end_strength = strength
            ei = clip.get("end_image")
            if ei:
                try:
                    eraw = base64.b64decode(ei.split(",")[-1])
                    end_img = Image.open(io.BytesIO(eraw)).convert("RGB")
                    end_img = end_img.resize((W1, H1))
                    end_path = str(Path(tmpd) / f"end_{ci}.png")
                    end_img.save(end_path)
                    end_strength = 0.85 if ci < n_clips - 1 else strength
                    _log(f"  [ltx] clip {ci+1}: end frame set "
                         f"(clip lands on it, strength {end_strength}).")
                except Exception as e:
                    _log(f"  [ltx] clip {ci+1}: bad end image ({e}); ignoring.")
                    end_path = None

            clip_lbl = f"clip {ci+1}/{n_clips}"
            clip_lo = 15 + int(75 * ci / n_clips)
            clip_hi = 15 + int(75 * (ci + 1) / n_clips)
            job.update(stage=f"LTX {clip_lbl}: rendering", progress=clip_lo)
            _log(f"  [ltx] {clip_lbl}: ~{W1*2}x{H1*2} (stage-1 {W1}x{H1}), "
                 f"{n} frames @ {fps} fps (~{n/fps:.1f}s), seed {seed}"
                 + (", chained-start" if is_chained else "") + ".")
            _log(f"  [ltx] {clip_lbl} prompt: {prompt[:120]}")

            spec = {"id": ci, "prompt": prompt, "seed": seed,
                    "height": H1, "width": W1, "frames": n, "fps": fps,
                    "image": img_path, "strength": start_strength, "crf": crf,
                    "enhance": enhance, "output": out_path}
            if end_path:
                spec["end_image"] = end_path
                spec["end_strength"] = end_strength
            # Re-ensure the resident worker: a preceding v2v clip stops it
            # to free VRAM for the IC-LoRA pipeline. Idempotent — reuses the
            # live process when the config is unchanged.
            proc = _ltx_worker_start(p, offload)
            proc.stdin.write(json.dumps(spec) + "\n")
            proc.stdin.flush()

            # Read status lines until this clip finishes or errors.
            done = False
            while not done:
                if job.get("cancel"):
                    _ltx_worker_stop()
                    raise _JobCancelled()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(
                        "LTX worker died mid-render — check the console log.")
                line = line.strip()
                try:
                    ev = json.loads(line)
                except Exception:
                    m = re.search(r"(\d+)/(\d+)\s*\[", line)
                    if m:
                        cs, ts = int(m.group(1)), int(m.group(2))
                        frac = (cs / ts) if ts else 0
                        job.update(
                            stage=f"LTX {clip_lbl}: step {cs}/{ts}",
                            progress=int(clip_lo + (clip_hi - clip_lo)
                                         * (0.1 + 0.85 * frac)))
                    elif line:
                        _log("  [ltx] " + line[:200])
                    continue
                et = ev.get("event")
                if et == "clip_start":
                    job.update(stage=f"LTX {clip_lbl}: generating",
                               progress=int(clip_lo + (clip_hi-clip_lo)*0.1))
                elif et == "clip_done":
                    done = True
                elif et == "clip_error":
                    _log("  [ltx] " + (ev.get("trace") or ev.get("error","")))
                    raise RuntimeError(
                        f"LTX {clip_lbl} failed — {ev.get('error','see console')}")
            if not Path(out_path).exists():
                raise RuntimeError(f"LTX {clip_lbl} produced no output.")

            # LTX-2.3 tail-smear fix: the final few frames of a distilled
            # clip can smear. For a MULTI-clip movie, trim a small tail off
            # each clip before stitching so cuts are clean, and chain the
            # NEXT clip from a frame just before the (untrimmed) end.
            TAIL = 6 if n_clips > 1 else 0
            if TAIL:
                trimmed = str(Path(tmpd) / f"clip_{ci}_t.mp4")
                _ltx_trim_tail(out_path, trimmed, TAIL, fps)
                clip_paths.append(trimmed)
            else:
                clip_paths.append(out_path)

            # Post this finished clip to the stage immediately so the user
            # watches the movie build clip-by-clip (the latest plays live).
            try:
                with open(out_path, "rb") as _cf:
                    _cb64 = base64.b64encode(_cf.read()).decode()
                job.update(stage=f"LTX clip {ci+1}/{n_clips} done",
                           partial=f"data:video/mp4;base64,{_cb64}",
                           partial_index=ci + 1, partial_total=n_clips)
            except Exception:
                pass

            # Chain: a CLEAN near-last frame -> next clip's start image.
            if ci < n_clips - 1:
                nxt = str(Path(tmpd) / f"start_{ci+1}.png")
                # Grab ~0.06s before the end (skips the smeary tail).
                _ltx_last_frame(out_path, nxt, back=6)
                cur_img = Image.open(nxt).convert("RGB")

        # ---- stitch clips into one continuous video -----------------------
        job.update(stage="stitching clips", progress=92)
        final_path = str(Path(tmpd) / "final.mp4")
        _ltx_concat(clip_paths, final_path, tmpd)
        # Optional: strip the MUSIC stem, keeping dialogue + sound effects.
        if params.get("strip_music"):
            job.update(stage="removing music (keeping SFX & dialogue)",
                       progress=95)
            stripped = _ltx_strip_music(final_path, tmpd)
            if stripped:
                final_path = stripped
        total = sum(max(9, ((int(c.get("frames", 121)) - 1)//8)*8+1)
                    for c in clips)
        sz = os.path.getsize(final_path)
        _log(f"  [ltx] movie complete: {n_clips} clip(s), ~{total} frames, "
             f"{total/fps:.1f}s at {fps} fps ({sz/1024:.0f} KB, with audio). "
             "Model stays resident for the next render.")
        _v2v_worker_stop()   # release the v2v model once the movie is done
        with open(final_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try:
            import shutil as _sh
            _sh.rmtree(tmpd, ignore_errors=True)
        except Exception:
            pass
    job.update(status="done", stage="complete", progress=100,
               result=f"data:video/mp4;base64,{b64}")

# ---- generation worker -------------------------------------------------
def run_job(job_id, params):
    job = jobs[job_id]
    try:
        # ---- engine routing ---------------------------------------------
        # "wan" (default) renders in-process via diffusers below; "ltx"
        # hands the whole job to the LTX-2.3 distilled subprocess. Same
        # try/except so cancel + error handling behave identically.
        if params.get("engine", "wan") == "ltx":
            with STATE["lock"]:
                _run_ltx_job(job_id, params)
            return

        engine = params.get("engine", "wan")
        is_wan22 = (engine == "wan22")

        from diffusers.utils import export_to_video
        from diffusers import UniPCMultistepScheduler
        import numpy as _np
        job.update(status="running", stage="loading model", progress=5)
        if job.get("cancel"):
            raise _JobCancelled()

        # ---- resolve the task / mode -----------------------------------
        mode = params.get("mode", "i2v")
        if mode not in MODE_REPOS:
            mode = "i2v"

        def _load(key):
            if params.get(key):
                raw = base64.b64decode(params[key].split(",")[-1])
                return Image.open(io.BytesIO(raw)).convert("RGB")
            return None

        img = _load("image")            # i2v start / flf2v first / vace start
        last_img = _load("last_image")  # flf2v last
        ref_img = _load("reference_image")  # vace reference subject

        # Pick repo key + the pixel-budget profile per mode.
        if mode == "flf2v":
            model_key = "720P"; area_profile = "720P"
            if img is None or last_img is None:
                raise ValueError("FLF2V needs a first AND a last frame.")
        elif mode == "vace":
            size = params.get("vace_size", "1.3B")
            model_key = size if size in MODE_REPOS["vace"] else "1.3B"
            area_profile = "480P" if model_key == "1.3B" else "720P"
            if ref_img is None and img is None:
                raise ValueError("VACE needs a reference image or a "
                                 "start frame.")
        else:   # i2v
            mode = "i2v"
            area_profile = params.get("profile", "480P")
            if area_profile not in MODE_REPOS["i2v"]:
                area_profile = "480P"
            model_key = area_profile
            if img is None:
                raise ValueError("A starting image is required.")

        pipe = (build_pipeline_wan22(area_profile,
                                     lightning=params.get("wan22_lightning", True))
                if is_wan22 else build_pipeline(mode, model_key))

        # Wan frame rule: num_frames must be 4n+1.
        n_req = int(params["frames"])
        n = max(5, ((n_req - 1) // 4) * 4 + 1)
        steps = max(1, int(params["steps"]))
        shift = float(params.get("flow_shift", 3.0))
        # Wan 2.2 Lightning: the distill LoRAs are baked for a 4-step /
        # CFG~1 render — override whatever the UI sent so it actually runs
        # fast and correct.
        if is_wan22 and params.get("wan22_lightning", True):
            steps = WAN22_LIGHTNING_STEPS
            params = dict(params, guidance=WAN22_LIGHTNING_GUIDANCE)
        # Long video by chaining only applies to i2v. flf2v/vace are
        # single-clip (their conditioning is a fixed first/last/reference).
        segments = max(1, min(int(params.get("segments", 1)), 24))
        if mode != "i2v":
            segments = 1
        _log(f"  [job] mode={mode}; repo_key={model_key}; "
             f"area={area_profile}; frames/clip {n_req}->{n}; "
             f"clips={segments}; steps={steps}; "
             f"guidance={params['guidance']}; flow_shift={shift}; "
             f"fps={params['fps']}; seed={params['seed']}")
        _log(f"  [job] prompt: {params['prompt'][:120]}")

        def _as_pil(fr):
            # Coerce a returned frame (PIL or numpy) to a PIL.Image so it
            # can seed the next clip in a chained long render.
            if isinstance(fr, Image.Image):
                return fr
            a = _np.asarray(fr)
            if a.dtype != _np.uint8:
                a = ((a * 255.0) if a.max() <= 1.0 else a)
                a = a.clip(0, 255).astype("uint8")
            return Image.fromarray(a)

        def _vace_video_mask(first_im, last_im, W, H, n):
            # VACE conditioning: known frames get a BLACK mask (keep), the
            # frames the model should invent get a WHITE mask + grey filler.
            grey = Image.new("RGB", (W, H), (128, 128, 128))
            mb = Image.new("L", (W, H), 0)
            mw = Image.new("L", (W, H), 255)
            vframes, vmask = [], []
            for i in range(n):
                if i == 0 and first_im is not None:
                    vframes.append(first_im.resize((W, H))); vmask.append(mb)
                elif i == n - 1 and last_im is not None:
                    vframes.append(last_im.resize((W, H))); vmask.append(mb)
                else:
                    vframes.append(grey); vmask.append(mw)
            return vframes, vmask

        with STATE["lock"]:
            # flow_shift lives on the scheduler; set it per job.
            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=shift)
            except Exception as e:
                _log(f"  [job] scheduler flow_shift not applied: {e}")

            if job.get("cancel"):
                raise _JobCancelled()
            apply_loras(pipe)
            active = [k for k, i in STATE["loras"].items()
                      if i.get("attached")]
            _log(f"  [job] active LoRAs: {active or 'none'}")
            job.update(stage="generating", progress=15)

            # Output W/H: pixel budget from the area profile, aspect ratio
            # from whichever image we have.
            dim_src = img or ref_img or last_img
            W, H = _wan_dims(dim_src, area_profile)
            _log(f"  [job] -> generating {W}x{H}")

            # Pre-resize the mode's fixed conditioning inputs.
            cond_img = img.resize((W, H)) if img is not None else None
            last_resized = (last_img.resize((W, H))
                            if last_img is not None else None)
            ref_resized = (ref_img.resize((W, H))
                           if ref_img is not None else None)
            if mode == "vace":
                if img is not None or last_img is not None:
                    vace_video, vace_mask = _vace_video_mask(
                        img, last_img, W, H, n)
                else:
                    vace_video, vace_mask = None, None

            frames = []
            for seg in range(segments):
                if job.get("cancel"):
                    raise _JobCancelled()
                _step_count = {"n": 0}

                def _cb(pipe_self, step, t, kw, _seg=seg):
                    if job.get("cancel"):
                        raise _JobCancelled()
                    _step_count["n"] = step + 1
                    frac = (step + 1) / max(1, steps)
                    overall = (_seg + frac) / segments
                    lbl = (f"clip {_seg+1}/{segments} · " if segments > 1
                           else "")
                    job.update(
                        stage=f"{lbl}step {step+1}/{steps}",
                        progress=int(15 + overall * 73))
                    if step == 0 or (step + 1) % 5 == 0:
                        _log(f"  [job] {lbl}diffusion step "
                             f"{step+1}/{steps}")
                    return kw

                gen = torch.Generator().manual_seed(
                    int(params["seed"])
                    + (seg if params.get("vary_seed", True) else 0))
                if mode == "flf2v":
                    call = dict(
                        image=cond_img, last_image=last_resized,
                        prompt=params["prompt"],
                        negative_prompt=params["negative_prompt"],
                        height=H, width=W, num_frames=n,
                        num_inference_steps=steps,
                        guidance_scale=float(params["guidance"]),
                        generator=gen)
                elif mode == "vace":
                    call = dict(
                        prompt=params["prompt"],
                        negative_prompt=params["negative_prompt"],
                        height=H, width=W, num_frames=n,
                        num_inference_steps=steps,
                        guidance_scale=float(params["guidance"]),
                        generator=gen)
                    if ref_resized is not None:
                        call["reference_images"] = [ref_resized]
                    if vace_video is not None:
                        call["video"] = vace_video
                        call["mask"] = vace_mask
                else:   # i2v
                    call = dict(
                        image=cond_img,
                        prompt=params["prompt"],
                        negative_prompt=params["negative_prompt"],
                        height=H, width=W, num_frames=n,
                        num_inference_steps=steps,
                        guidance_scale=float(params["guidance"]),
                        generator=gen)

                if seg == 0:
                    _log(f"  [job] calling {mode} pipeline: "
                         f"{sorted(call.keys())}")
                try:
                    out = pipe(
                        callback_on_step_end=_cb,
                        callback_on_step_end_tensor_inputs=["latents"],
                        **call)
                except TypeError as te:
                    _log(f"  [job] callback signature rejected ({te}); "
                         "retrying without callbacks")
                    out = pipe(**call)    # older diffusers, no callbacks

                # WanPipelineOutput.frames[0] is this sample's frame list;
                # with return_dict=False it is a plain tuple instead.
                if isinstance(out, (tuple, list)):
                    seg_frames = out[0][0]
                else:
                    seg_frames = getattr(out, "frames", out)[0]

                # Stitch: keep every frame of the first clip; for later
                # clips drop frame 0 (it duplicates the conditioning frame)
                # so the seam doesn't stutter.
                if seg == 0:
                    frames.extend(seg_frames)
                else:
                    frames.extend(seg_frames[1:])

                # Chained long video (i2v only): next clip starts from the
                # final frame of this one.
                if mode == "i2v" and seg + 1 < segments:
                    cond_img = _as_pil(seg_frames[-1]).resize((W, H))
                    _log(f"  [job] clip {seg+1}/{segments} done; chaining "
                         "its last frame into the next clip.")

        # --- motion diagnostic -------------------------------------------
        # If the model produced a frozen clip, every frame is identical.
        # Compare first vs last and a mid frame so we can SEE whether the
        # problem is generation (no motion) or export (frames fine, video
        # frozen). This logs into the console panel.
        try:
            n_out = len(frames)
            a0 = _np.asarray(frames[0]).astype("float32")
            aL = _np.asarray(frames[-1]).astype("float32")
            diff_first_last = float(_np.abs(a0 - aL).mean())
            mid = frames[n_out // 2] if n_out > 2 else frames[-1]
            aM = _np.asarray(mid).astype("float32")
            diff_first_mid = float(_np.abs(a0 - aM).mean())
            _log(f"  [diag] frames produced: {n_out}")
            _log(f"  [diag] mean pixel diff  first vs last: "
                 f"{diff_first_last:.3f}  (0-255 scale)")
            _log(f"  [diag] mean pixel diff  first vs mid : "
                 f"{diff_first_mid:.3f}")
            if n_out <= 1:
                _log("  [diag] *** ONLY 1 FRAME PRODUCED — num_frames or "
                     "the Wan 4n+1 frame rule is wrong. ***")
            elif diff_first_last < 4.0:
                _log("  [diag] *** FRAMES ARE NEARLY STATIC (diff < 4) — "
                     "little motion. Use a motion-describing prompt, "
                     "raise Steps (>=30), and try flow_shift 3-5. ***")
            else:
                _log("  [diag] motion present — frames differ clearly. "
                     "If the video still looks frozen, the problem is "
                     "export/playback.")
        except Exception as _de:
            _log(f"  [diag] motion check skipped: {_de}")

        job.update(stage="encoding video", progress=92)
        fps = max(1, int(params["fps"]))
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        export_to_video(frames, tmp.name, fps=fps)
        _sz = os.path.getsize(tmp.name)
        _log(f"  [job] final file {_sz/1024:.0f} KB at {fps} fps")

        with open(tmp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try: os.remove(tmp.name)
        except Exception: pass
        _log(f"  done — {len(frames)} frames, {len(frames)/fps:.1f}s "
             f"at {fps} fps.")
        job.update(status="done", stage="complete", progress=100,
                   result=f"data:video/mp4;base64,{b64}")
    except _JobCancelled:
        _log("  [job] cancelled by user.")
        job.update(status="cancelled", stage="cancelled")
        try: _v2v_worker_stop()
        except Exception: pass
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        msg = str(e).strip() or e.__class__.__name__
        if "out of memory" in tb.lower():
            msg = ("CUDA OOM — try the 480P profile, fewer frames, or "
                   "fewer steps. " + msg)
        job.update(status="error", stage="failed", error=msg)
        try: _v2v_worker_stop()
        except Exception: pass

# ---- job dispatcher ------------------------------------------------------
def _job_dispatcher():
    """Single worker that drains the job queue one render at a time.

    /api/generate only ENQUEUES and returns immediately, so the UI can
    keep stacking jobs while a clip is rendering — nothing on the
    request path ever waits on the GPU. Running jobs serially in one
    thread (instead of one thread per job, which is what used to
    happen) also removes a real race: a second job with a different
    mode/resolution could previously call build_pipeline() and swap or
    free the live pipeline WHILE the first job was still generating on
    it. Now build_pipeline is only ever entered between renders."""
    while True:
        job_queue_evt.wait()
        while True:
            try:
                job_id, params = job_queue.popleft()
            except IndexError:
                job_queue_evt.clear()
                break
            job = jobs.get(job_id)
            if not job:
                continue
            if job.get("cancel"):
                job.update(status="cancelled", stage="cancelled")
                _log(f"  [queue] job {job_id} cancelled before it started.")
                continue
            # Count this render against the server (Google account). Members
            # always pass; non-members are refused once the free trial is
            # used up — the job fails with a sign-up message instead of
            # consuming GPU.
            ok, why = _ml_charge_render(params.get("engine", "wan"))
            if not ok:
                job.update(status="error", stage="limit reached", error=why)
                _log(f"  [queue] job {job_id} blocked — {why}")
                continue
            run_job(job_id, params)

def _ml_charge_render(engine):
    """Ping the worker to record + authorize one render. Returns
    (ok, reason). Updates the local free-trial balance from the reply."""
    if ML.get("member"):
        return True, ""
    if not ML.get("key"):
        return False, "sign in with Google to generate videos."
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/render",
                           headers={"Authorization": f"Bearer {ML['key']}"},
                           json={"engine": engine}, timeout=20)
        j = r.json()
    except Exception as e:
        # Fail open only for members (handled above); for free users a
        # server we can't reach means we can't authorize — fail closed.
        return False, f"could not reach MissingLink to authorize the render — {e}"
    if r.status_code == 200 and j.get("ok"):
        ML["used"] = int(j.get("used", ML["used"]))
        ML["remaining"] = (-1 if j.get("member")
                           else int(j.get("remaining", 0)))
        ML["member"] = bool(j.get("member"))
        left = ("unlimited" if ML["member"]
                else f"{ML['remaining']} of {ML['free_limit']} free renders left")
        _log(f"  [trial] render authorized — {left}.")
        return True, ""
    # Refused: free trial exhausted (or auth lost).
    ML["remaining"] = 0
    ML["reason"] = "free_limit_reached"
    return False, (f"You've used all {ML['free_limit']} free renders. "
                   f"Sign up for MissingLink to keep generating: "
                   f"{MISSINGLINK_SIGNUP_URL}")

threading.Thread(target=_job_dispatcher, daemon=True,
                 name="job-dispatcher").start()

# ---- Flask app ---------------------------------------------------------
app = Flask(__name__)

# ---- MissingLink login gate ----------------------------------------------
# Everything functional requires being signed in with Google AND either
# membership or free renders remaining. The UI, health pings, and the auth
# endpoints stay open so the login screen can render and report status.
_ML_OPEN_PATHS = ("/", "/api/keepalive", "/api/auth/status",
                  "/api/auth/login", "/api/console", "/api/hw")

@app.before_request
def _ml_gate():
    if request.path in _ML_OPEN_PATHS or not request.path.startswith("/api/"):
        return None
    if _ml_unlocked():
        return None
    # reason distinguishes "not signed in" from "free trial used up" so the
    # UI shows the Google-sign-in screen vs the sign-up prompt.
    reason = ("no_session" if not ML.get("authed")
              else "free_limit_reached")
    return jsonify(error="login_required", reason=reason,
                   **_ml_public()), 401

@app.route("/api/auth/status")
def _auth_status():
    # A token restored from disk (cell re-run / page reload) hasn't been
    # validated yet — do it on the first status check so the user lands
    # already signed in instead of seeing the gate.
    if ML.get("key") and not ML.get("authed"):
        _ml_refresh()
    # ?refresh=1 revalidates the stored token — used after signing up, or
    # to pull an updated free-render balance.
    elif request.args.get("refresh") and ML.get("key"):
        _ml_refresh()
    return jsonify(**_ml_public())

@app.route("/api/auth/logout", methods=["POST"])
def _auth_logout():
    _ml_session_clear()
    return jsonify(ok=True, **_ml_public())

@app.route("/api/auth/login", methods=["POST"])
def _auth_login():
    d = request.get_json(force=True) or {}
    # Accept either the pasted token field name.
    key = (d.get("token") or d.get("key") or "").strip()
    return jsonify(_ml_login(key))

def _ml_forward_error(r):
    """Map a worker 401/402/403 (not signed in, or free trial used up)
    back into the login gate so the UI reacts immediately."""
    if r.status_code in (401, 402, 403):
        try:
            j = r.json()
        except Exception:
            j = {}
        if r.status_code == 402 or j.get("error") == "free_limit_reached":
            ML["remaining"] = 0
            ML["reason"] = "free_limit_reached"
            reason = "free_limit_reached"
        else:
            ML["authed"] = False
            ML["reason"] = "no_session"
            reason = "no_session"
        return jsonify(error="login_required", reason=reason,
                       **_ml_public()), 401
    return None

@app.route("/")
def _index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/api/keepalive")
def _keepalive():
    return jsonify(ok=True, t=time.time())

@app.route("/api/preload")
def _preload_status():
    """Launch-time LTX-2.3 preload progress (idle/downloading/loading/
    ready/error) so the UI can show readiness before Generate."""
    return jsonify(**LTX_PRELOAD)

@app.route("/api/loras", methods=["GET"])
def _list_loras():
    return jsonify(loras=[{"name": n, "scale": i["scale"],
                           "attached": i["attached"],
                           "url": i.get("url", ""),
                           "engine": i.get("engine") or "wan"}
                          for n, i in STATE["loras"].items()])

@app.route("/api/loras/add", methods=["POST"])
def _add_lora():
    d = request.get_json(force=True)
    ok, reason = register_lora(d.get("name") or "lora",
                               (d.get("url") or "").strip(),
                               float(d.get("scale", 1.0)),
                               engine=(d.get("engine") or "wan"))
    return (jsonify(ok=True), 200) if ok else (jsonify(ok=False, error=reason), 400)

@app.route("/api/loras/update", methods=["POST"])
def _update_lora():
    d = request.get_json(force=True)
    name = d.get("name")
    if name in STATE["loras"]:
        STATE["loras"][name]["scale"] = float(d.get("scale", 1.0))
        return jsonify(ok=True)
    return jsonify(ok=False, error="No such LoRA."), 404

@app.route("/api/loras/remove", methods=["POST"])
def _remove_lora():
    d = request.get_json(force=True)
    return (jsonify(ok=True) if remove_lora(d.get("name"))
            else (jsonify(ok=False, error="No such LoRA."), 404))

def _job_thumb(params):
    """Tiny JPEG data-URI thumbnail of the job's conditioning image, so
    queue/history entries survive page reloads without the server
    shipping full-size uploads back on every poll."""
    for key in ("image", "reference_image", "last_image"):
        data = params.get(key)
        if not data:
            continue
        try:
            raw = base64.b64decode(data.split(",")[-1])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            im.thumbnail((128, 128))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
            return ("data:image/jpeg;base64,"
                    + base64.b64encode(buf.getvalue()).decode())
        except Exception:
            continue
    return ""

@app.route("/api/generate", methods=["POST"])
def _generate():
    params = request.get_json(force=True)
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": "queued",
                    "result": None, "error": None, "cancel": False,
                    "prompt": (params.get("prompt") or "")[:300],
                    "thumb": _job_thumb(params), "ts": time.time()}
    # Enqueue and return IMMEDIATELY — the dispatcher thread picks jobs
    # up one at a time, so submitting never waits on a running render.
    job_queue.append((job_id, params))
    job_queue_evt.set()
    _log(f"  [queue] job {job_id} enqueued "
         f"({len(job_queue)} waiting).")
    return jsonify(job_id=job_id)

@app.route("/api/partial/<job_id>")
def _partial(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("partial"):
        return jsonify(error="no partial"), 404
    return jsonify(partial=job.get("partial"),
                   index=job.get("partial_index", 0),
                   total=job.get("partial_total", 0))

@app.route("/api/facekeyframes", methods=["POST"])
def _face_keyframes():
    """Walk the video in 10s windows; in each window grab the FIRST frame
    that contains a face. Save those frames as PNGs + a timestamps JSON,
    bundled into a single downloadable zip."""
    d = request.get_json(force=True) or {}
    data_url = d.get("video") or ""
    if not data_url.startswith("data:"):
        return jsonify(ok=False, error="no video"), 400
    try:
        window = float(d.get("window", 10))
    except Exception:
        window = 10.0
    window = max(1.0, window)
    import tempfile, zipfile, io as _io
    work = tempfile.mkdtemp(prefix="facekf_")
    try:
        src = os.path.join(work, "src.mp4")
        with open(src, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        import cv2
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return jsonify(ok=False, error="could not open video"), 400
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if fps else 0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        # YuNet: OpenCV's built-in CNN face detector. Far better than Haar
        # (catches angled/profile/small faces, gives confidence), ~340 KB
        # model fetched once. Falls back to Haar if the model can't load.
        detector = _load_yunet(vw, vh, work)
        casc = None
        if detector is None:
            casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def _has_face(frame):
            if detector is not None:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
                if faces is None:
                    return 0
                # Keep confident detections only (col 14 = score).
                return int(sum(1 for f in faces if f[-1] >= 0.6))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return int(len(casc.detectMultiScale(gray, 1.1, 5,
                                                 minSize=(40, 40))))
        frames = []          # (index, timestamp, png_path)
        win_start = 0.0
        # Scan ~5 candidate frames per second (enough to catch a face early
        # without decoding every frame of a long clip).
        step = max(1, int(round(fps / 5)))
        fi = 0
        found_in_window = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps if fps else 0
            # New window boundary -> reset the "found" flag.
            if t >= win_start + window:
                win_start += window * ((t - win_start) // window)
                found_in_window = False
            if not found_in_window and (fi % step == 0):
                nfaces = _has_face(frame)
                if nfaces > 0:
                    idx = len(frames) + 1
                    png = os.path.join(work, f"face_{idx:03d}_{t:07.2f}s.png")
                    cv2.imwrite(png, frame)
                    frames.append({"index": idx,
                                   "timestamp_seconds": round(t, 3),
                                   "timestamp": _fmt_ts(t),
                                   "frame_number": fi,
                                   "faces_detected": int(nfaces),
                                   "file": os.path.basename(png)})
                    found_in_window = True
            fi += 1
        cap.release()
        if not frames:
            return jsonify(ok=False,
                           error="no faces found in any window"), 200
        # Bundle PNGs + timestamps.json into one zip.
        meta = {"source_duration_seconds": round(dur, 3),
                "window_seconds": window, "fps": round(fps, 3),
                "face_frames": frames}
        zpath = os.path.join(work, f"face_keyframes_{int(time.time())}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fr in frames:
                z.write(os.path.join(work, fr["file"]), fr["file"])
            z.writestr("timestamps.json", json.dumps(meta, indent=2))
        with open(zpath, "rb") as f:
            zb64 = base64.b64encode(f.read()).decode()
        return jsonify(ok=True,
                       zip="data:application/zip;base64," + zb64,
                       count=len(frames), timestamps=meta)
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:300]), 500
    finally:
        try:
            import shutil as _sh; _sh.rmtree(work, ignore_errors=True)
        except Exception:
            pass


def _fmt_ts(t):
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:06.3f}"


@app.route("/api/facesplit/start", methods=["POST"])
def _facesplit_start():
    """Kick off face detection on a control video as a BACKGROUND JOB so it
    shows in the queue with 'frame N/M' progress. Returns a job_id; the
    frontend polls /api/status and reads cut boundaries from the result."""
    d = request.get_json(force=True) or {}
    data_url = d.get("video") or ""
    if not data_url.startswith("data:"):
        return jsonify(ok=False, error="no video"), 400
    try:
        window = max(1.0, float(d.get("window", 10)))
    except Exception:
        window = 10.0
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "progress": 0, "stage": "starting",
                    "result": None, "error": None, "cancel": False,
                    "kind": "facesplit", "prompt": "Detecting faces",
                    "thumb": "", "ts": time.time()}
    threading.Thread(target=_facesplit_job,
                     args=(job_id, data_url, window), daemon=True).start()
    return jsonify(ok=True, job_id=job_id)

def _facesplit_job(job_id, data_url, window):
    import tempfile, cv2
    job = jobs[job_id]
    work = tempfile.mkdtemp(prefix="facesplit_")
    try:
        src = os.path.join(work, "src.mp4")
        with open(src, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            job.update(status="error", error="could not open video"); return
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if fps else 0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        detector = _load_yunet(vw, vh, work)
        casc = None
        if detector is None:
            casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        def _has_face(frame):
            if detector is not None:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
                return 0 if faces is None else int(sum(1 for f in faces
                                                       if f[-1] >= 0.6))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return int(len(casc.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))))
        cut_ts = []
        win_start = 0.0
        step = max(1, int(round(fps / 5)))
        fi = 0
        found = False
        while True:
            if job.get("cancel"):
                job.update(status="cancelled", stage="cancelled"); return
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps if fps else 0
            if t >= win_start + window:
                win_start += window * ((t - win_start) // window)
                found = False
            if not found and (fi % step == 0) and _has_face(frame) > 0:
                cut_ts.append(round(t, 3))
                found = True
            fi += 1
            if total and (fi % 15 == 0):
                job.update(progress=min(98, int(100 * fi / total)),
                           stage=f"frame {fi}/{total} \u00b7 {len(cut_ts)} shots")
        cap.release()
        if not cut_ts:
            job.update(status="error", error="no faces found to anchor clips")
            return
        bounds = []
        for i, a in enumerate(cut_ts):
            b = cut_ts[i + 1] if i + 1 < len(cut_ts) else dur
            if b - a > 0.05:
                bounds.append({"start": a, "len": round(b - a, 3)})
        job.update(status="done", stage="complete", progress=100,
                   result=json.dumps({"total": round(dur, 3),
                                      "count": len(bounds), "shots": bounds}))
    except Exception as e:
        job.update(status="error", error=str(e)[:200])
    finally:
        try:
            import shutil as _sh; _sh.rmtree(work, ignore_errors=True)
        except Exception:
            pass

@app.route("/api/faceautoclip", methods=["POST"])
def _face_autoclip():
    """Auto-cut the video into clips anchored on faces, then export, per
    clip: the clip video (lossless), its FIRST and LAST frame as PNGs, and
    a manifest JSON. The first/last frames are the reference points for
    reskinning a consistent character across cuts. One downloadable zip."""
    d = request.get_json(force=True) or {}
    data_url = d.get("video") or ""
    if not data_url.startswith("data:"):
        return jsonify(ok=False, error="no video"), 400
    try:
        window = max(1.0, float(d.get("window", 10)))
    except Exception:
        window = 10.0
    import tempfile, zipfile
    work = tempfile.mkdtemp(prefix="autoclip_")
    try:
        src = os.path.join(work, "src.mp4")
        with open(src, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        import cv2
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return jsonify(ok=False, error="could not open video"), 400
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if fps else 0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        detector = _load_yunet(vw, vh, work)
        casc = None
        if detector is None:
            casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def _has_face(frame):
            if detector is not None:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
                return 0 if faces is None else int(sum(1 for f in faces
                                                       if f[-1] >= 0.6))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return int(len(casc.detectMultiScale(gray, 1.1, 5,
                                                 minSize=(40, 40))))

        # Find the first face timestamp in each window -> cut points.
        cut_ts = []
        win_start = 0.0
        step = max(1, int(round(fps / 5)))
        fi = 0
        found = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps if fps else 0
            if t >= win_start + window:
                win_start += window * ((t - win_start) // window)
                found = False
            if not found and (fi % step == 0) and _has_face(frame) > 0:
                cut_ts.append(round(t, 3))
                found = True
            fi += 1
        cap.release()
        if not cut_ts:
            return jsonify(ok=False, error="no faces found to anchor clips"), 200

        # Boundaries: each clip spans from one face anchor to the next
        # (last clip runs to the end).
        bounds = []
        for i, a in enumerate(cut_ts):
            b = cut_ts[i + 1] if i + 1 < len(cut_ts) else dur
            if b - a > 0.05:
                bounds.append((a, b))

        clips_meta = []
        with zipfile.ZipFile(os.path.join(work, "out.zip"), "w",
                             zipfile.ZIP_DEFLATED) as z:
            for i, (a, b) in enumerate(bounds, 1):
                base = f"clip_{i:02d}_{a:07.2f}-{b:07.2f}s"
                clip_mp4 = os.path.join(work, base + ".mp4")
                # Lossless cut (frame-accurate output seeking).
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-ss", f"{a:.3f}",
                     "-to", f"{b:.3f}", "-c:v", "libx264", "-qp", "0",
                     "-preset", "veryfast", "-pix_fmt", "yuv444p",
                     "-c:a", "copy", clip_mp4],
                    capture_output=True, text=True)
                if not os.path.exists(clip_mp4):
                    continue
                # First + last frame PNGs.
                first_png = os.path.join(work, base + "_FIRST.png")
                last_png = os.path.join(work, base + "_LAST.png")
                subprocess.run(["ffmpeg", "-y", "-i", clip_mp4,
                                "-frames:v", "1", "-q:v", "2", first_png],
                               capture_output=True)
                subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i",
                                clip_mp4, "-frames:v", "1", "-q:v", "2",
                                "-update", "1", last_png],
                               capture_output=True)
                z.write(clip_mp4, "clips/" + os.path.basename(clip_mp4))
                if os.path.exists(first_png):
                    z.write(first_png, "frames/" + os.path.basename(first_png))
                if os.path.exists(last_png):
                    z.write(last_png, "frames/" + os.path.basename(last_png))
                clips_meta.append({
                    "clip": i, "start_seconds": a, "end_seconds": b,
                    "start": _fmt_ts(a), "end": _fmt_ts(b),
                    "duration_seconds": round(b - a, 3),
                    "video": "clips/" + os.path.basename(clip_mp4),
                    "first_frame": "frames/" + os.path.basename(first_png),
                    "last_frame": "frames/" + os.path.basename(last_png)})
            manifest = {"source_duration_seconds": round(dur, 3),
                        "fps": round(fps, 3), "window_seconds": window,
                        "clip_count": len(clips_meta), "clips": clips_meta}
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        with open(os.path.join(work, "out.zip"), "rb") as f:
            zb64 = base64.b64encode(f.read()).decode()
        return jsonify(ok=True, zip="data:application/zip;base64," + zb64,
                       count=len(clips_meta), manifest=manifest)
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:300]), 500
    finally:
        try:
            import shutil as _sh; _sh.rmtree(work, ignore_errors=True)
        except Exception:
            pass


# YuNet model cached under the LTX models dir (fetched once, ~340 KB).
_YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

def _load_yunet(w, h, work):
    """Build a cv2.FaceDetectorYN, fetching the small ONNX once. Returns
    None if unavailable (caller falls back to Haar)."""
    try:
        import cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        model = os.path.join(LTX_MODELS if 'LTX_MODELS' in globals()
                             else work, "face_detection_yunet.onnx")
        if not (os.path.exists(model) and os.path.getsize(model) > 10000):
            from huggingface_hub import hf_hub_download  # noqa
            got = None
            for url in _YUNET_URLS:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, model)
                    if os.path.getsize(model) > 10000:
                        got = model; break
                except Exception:
                    continue
            if not got:
                return None
        det = cv2.FaceDetectorYN.create(
            model, "", (w, h), score_threshold=0.6, nms_threshold=0.3)
        return det
    except Exception:
        return None


@app.route("/api/vr180", methods=["POST"])
def _vr180():
    """Convert a flat clip to MONO 180° VR (half-equirectangular, side-by-
    side, VR-tagged). Deterministic ffmpeg: upscale -> project flat->he
    -> duplicate to SBS -> inject spatial metadata. Mono = both eyes see
    the same image (immersive curved screen, no artifacts)."""
    d = request.get_json(force=True) or {}
    data_url = d.get("video") or ""
    if not data_url.startswith("data:"):
        return jsonify(ok=False, error="no video"), 400
    # How wide the flat footage sits inside the 180 hemisphere. Filling the
    # full 180 stretches normal footage badly; ~100 looks natural (the clip
    # occupies the center, curving away at the edges). Clamped 60..180.
    try:
        in_fov = float(d.get("fov", 100))
    except Exception:
        in_fov = 100.0
    in_fov = max(60.0, min(180.0, in_fov))
    import tempfile
    work = tempfile.mkdtemp(prefix="vr180_")
    try:
        src = os.path.join(work, "src.mp4")
        with open(src, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", src],
            capture_output=True, text=True)
        st = (json.loads(pr.stdout or "{}").get("streams") or [{}])[0]
        sw = int(st.get("width") or 1280); sh = int(st.get("height") or 720)
        # Upscale 2x (lanczos) so the reprojected result stays crisp; cap
        # the output for headset sanity. MONO = one full-resolution frame
        # (no SBS duplication — both eyes are identical, so a mono-tagged
        # file keeps full quality instead of halving it into a wasted pair).
        ow = min(sw * 2, 4096); oh = min(sh * 2, 4096)
        out = os.path.join(work, "vr180.mp4")
        # flat -> half-equirectangular. ih_fov/iv_fov say how much of the
        # sphere the SOURCE covers; h_fov/v_fov=180 defines the 180 output.
        # Going in at ~100 and out at 180 curves the footage naturally
        # instead of stretching it edge to edge.
        vf = (f"scale={ow}:{oh}:flags=lanczos,"
              f"v360=flat:hequirect:ih_fov={in_fov:.0f}:iv_fov={in_fov*oh/ow:.0f}"
              f":h_fov=180:v_fov=180:w={ow}:h={oh}:interp=lanczos")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vf", vf,
             "-c:v", "libx264", "-crf", "16", "-preset", "slow",
             "-pix_fmt", "yuv420p", "-c:a", "copy", out],
            capture_output=True, text=True)
        if not os.path.exists(out):
            return jsonify(ok=False,
                           error="ffmpeg failed: " + (r.stderr or "")[-400:]), 500
        # Tag as MONOSCOPIC 180 (no stereo mode) so players show one image
        # to both eyes on the curved 180 screen.
        tagged = _vr_inject_metadata(out, work, stereo=None)
        final = tagged if os.path.exists(tagged) else out
        with open(final, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return jsonify(ok=True, url="data:video/mp4;base64," + b64,
                       out_w=ow, out_h=oh, fov=in_fov)
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:300]), 500
    finally:
        try:
            import shutil as _sh; _sh.rmtree(work, ignore_errors=True)
        except Exception:
            pass


def _vr_inject_metadata(mp4_path, work, stereo="left-right"):
    """Tag an MP4 as spherical 180. stereo='left-right' for SBS stereo, or
    None for MONOSCOPIC (one image to both eyes). Prefers Google's
    spatial-media injector; falls back to the untagged file."""
    out = os.path.join(work, "vr180_tagged.mp4")
    try:
        try:
            from spatialmedia import metadata_utils  # noqa: F401
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "spatialmedia"], capture_output=True, text=True)
        from spatialmedia import metadata_utils
        md = metadata_utils.Metadata()
        # Mono => stereo=None (mono); tagging mono as left-right makes
        # headsets show only the left half. Spherical flag = 180 projection.
        md.video = metadata_utils.generate_spherical_xml(
            stereo=stereo, spherical=True)
        metadata_utils.inject_metadata(mp4_path, out, md, lambda *_: None)
        if os.path.exists(out):
            return out
    except Exception:
        pass
    return mp4_path


@app.route("/api/status/<job_id>")
def _status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    # Return everything EXCEPT the (potentially multi-MB base64) result,
    # so polling stays tiny and fast. The frontend fetches the result
    # once, separately, when status flips to "done". This is what stops
    # overlapping slow polls from duplicating finished jobs in history.
    slim = {k: v for k, v in job.items()
            if k not in ("result", "thumb", "partial")}
    slim["has_result"] = bool(job.get("result"))
    slim["partial_index"] = job.get("partial_index", 0)
    if job.get("status") == "queued":
        # Live position so the UI can show "queued · #2 in line".
        try:
            pos = [jid for jid, _ in list(job_queue)].index(job_id) + 1
            slim["stage"] = f"queued \u00b7 #{pos} in line"
        except ValueError:
            slim["stage"] = "starting"   # popped, about to run
    return jsonify(slim)

@app.route("/api/jobs")
def _jobs_list():
    """Every job the server knows about (oldest first). The frontend
    calls this once on page load to rebuild the queue and history —
    finished videos live in the server's jobs dict, so a browser
    reload loses nothing; results are re-fetched lazily on click."""
    out = []
    for jid, j in list(jobs.items()):
        out.append({"id": jid, "status": j.get("status"),
                    "progress": j.get("progress", 0),
                    "stage": j.get("stage", ""),
                    "prompt": j.get("prompt", ""),
                    "thumb": j.get("thumb", ""),
                    "kind": j.get("kind", "video"),
                    "ts": j.get("ts", 0),
                    "has_result": bool(j.get("result"))})
    out.sort(key=lambda x: x["ts"])
    return jsonify(jobs=out)

# ---- ✨ Auto Prompt (GPT vision) ------------------------------------------
def _ai_image(data_uri):
    """Downscale the upload to <=1024px JPEG so the vision request is
    fast and cheap regardless of the original size."""
    raw = base64.b64decode(data_uri.split(",")[-1])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode())

@app.route("/api/autoprompt", methods=["POST"])
def _autoprompt():
    """Forward to missinglink.build, which runs GPT vision with
    MissingLink's own OpenAI key — members never need their own."""
    d = request.get_json(force=True) or {}
    if not d.get("image"):
        return jsonify(ok=False, error="Upload a starting image first."), 400
    try:
        img_uri = _ai_image(d["image"])
    except Exception as e:
        return jsonify(ok=False, error=f"could not read the image — {e}"), 400
    end_uri = None
    if d.get("end_image"):
        try:
            end_uri = _ai_image(d["end_image"])
        except Exception:
            end_uri = None
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/autoprompt",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={"image": img_uri,
                                 "end_image": end_uri,
                                 "engine": d.get("engine", "wan"),
                                 "instructions": (d.get("instructions") or "")[:2000],
                                 "context": (d.get("context") or "")[:2000]},
                           timeout=120)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "auto prompt "
                       "failed"), 502
    _log("  [auto] MissingLink vision generated a "
         f"{'scene + dialogue' if d.get('engine') == 'ltx' else 'scene'} "
         "prompt.")
    return jsonify(ok=True, **{k: j.get(k) for k in
                               ("scene", "camera", "speakers", "lines")})

@app.route("/api/autoline", methods=["POST"])
def _autoline():
    """Forward: write ONE dialogue line for a speaker, continuing the
    story. Fast single vision call on MissingLink's key."""
    d = request.get_json(force=True) or {}
    payload = {"speaker": (d.get("speaker") or "SPEAKER")[:40],
               "scene": (d.get("scene") or "")[:1500],
               "lines_here": (d.get("lines_here") or "")[:1500],
               "story": (d.get("story") or "")[:2000],
               "context": (d.get("context") or "")[:1500],
               "instructions": (d.get("instructions") or "")[:1500]}
    if d.get("image"):
        try:
            payload["image"] = _ai_image(d["image"])
        except Exception:
            pass
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/autoline",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json=payload, timeout=90)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    if r.status_code == 404:
        return jsonify(ok=False, error="Auto-write line isn't on the server "
                       "yet \u2014 redeploy the MissingLink worker (index.ts)."), 502
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "autoline failed"), 502
    return jsonify(ok=True, line=j.get("line", ""))

@app.route("/api/extendscene", methods=["POST"])
def _extendscene():
    """Forward to MissingLink: write a continuation prompt (and optional
    dialogue) that continues the SAME shot's action smoothly, so a chained
    clip flows on rather than resetting."""
    d = request.get_json(force=True) or {}
    payload = {"prev_scene": (d.get("prev_scene") or "")[:1500],
               "story": (d.get("story") or "")[:2000],
               "instructions": (d.get("instructions") or "")[:1500],
               "want_dialogue": bool(d.get("want_dialogue", True))}
    # optional last-frame image for visual grounding
    if d.get("image"):
        try:
            payload["image"] = _ai_image(d["image"])
        except Exception:
            pass
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/extendscene",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json=payload, timeout=120)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    if r.status_code == 404:
        return jsonify(ok=False, error="Extend isn't on the server yet \u2014 "
                       "redeploy the MissingLink worker (index.ts)."), 502
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "extend failed"), 502
    return jsonify(ok=True, **{k: j.get(k) for k in
                   ("scene", "camera", "speakers", "lines")})

def _run_nextscene_job(job_id, d):
    """Run the agentic next-scene loop as a tracked job with progress.
    Runs on its own thread (not the video queue) so a ~45s image build
    never waits behind a long render. Progress is coarse — the worker
    does the multi-step loop server-side — but it shows the user it's
    working and roughly where it is."""
    job = jobs[job_id]
    try:
        job.update(status="running", stage="reading previous frame", progress=8)
        try:
            img_uri = _ai_image(d["image"])
        except Exception as e:
            job.update(status="error", stage="failed",
                       error=f"could not read the frame — {e}")
            return
        job.update(stage="designing the next scene", progress=25)
        try:
            r = _requests.post(f"{MISSINGLINK_API}/api/notebook/nextscene",
                               headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                               json={"image": img_uri,
                                     "context": (d.get("context") or "")[:2000],
                                     "instructions": (d.get("instructions") or "")[:2000],
                                     "quality": d.get("quality", "medium"),
                                     "shot": d.get("shot", "auto")},
                               timeout=240)
        except Exception as e:
            job.update(status="error", stage="failed",
                       error=f"MissingLink request failed — {e}")
            return
        job.update(stage="generating & self-checking image", progress=70)
        if r.status_code == 404:
            job.update(status="error", stage="failed",
                       error="Auto Next Scene isn't on the server yet \u2014 "
                       "redeploy the MissingLink worker (index.ts).")
            return
        # Subscription / free-trial gate: update ML state so the UI reacts,
        # and fail the job with a clear reason.
        if r.status_code in (401, 402, 403):
            _ml_forward_error(r)   # sets ML['remaining']/['reason']
            job.update(status="error", stage="failed",
                       error="Sign in / subscription needed to build images.")
            return
        # surface subscription/limit gates as an error on the job
        try:
            j = r.json()
        except Exception:
            job.update(status="error", stage="failed",
                       error=f"MissingLink {r.status_code}: {(r.text or '')[:200]}")
            return
        if not j.get("ok"):
            job.update(status="error", stage="failed",
                       error=j.get("error") or "next scene failed")
            return
        _log(f"  [next] built next-scene image "
             f"({'corrected, ' if j.get('corrected') else ''}"
             f"{'fits' if j.get('works') else 'may not fit'}).")
        # Store the whole payload as the job result (retrieved via /result).
        job.update(status="done", stage="complete", progress=100,
                   result=json.dumps({k: j.get(k) for k in
                       ("image", "image_prompt", "intent", "corrected",
                        "works", "verdict")}))
    except Exception as e:
        job.update(status="error", stage="failed", error=str(e)[:300])


@app.route("/api/nextscene", methods=["POST"])
def _nextscene():
    """Queue the agentic next-scene image build as a tracked job and
    return its id immediately, so the UI shows progress in the queue
    (~45s). The result (image + verdict) is fetched from /api/result."""
    d = request.get_json(force=True) or {}
    if not d.get("image"):
        return jsonify(ok=False, error="No previous frame to build from."), 400
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": "queued",
                    "result": None, "error": None, "cancel": False,
                    "kind": "image",
                    "prompt": "Next-scene image",
                    "thumb": "", "ts": time.time()}
    threading.Thread(target=_run_nextscene_job, args=(job_id, d),
                     daemon=True).start()
    return jsonify(ok=True, job_id=job_id)

@app.route("/api/nextscene/propose", methods=["POST"])
def _nextscene_propose():
    """Stage 1: PROPOSE an image-gen prompt from the previous frame (no
    generation, no token cost). Returns the prompt + intent so the user
    can edit before generating.

    With the user's own OPENAI_API_KEY set, runs directly on their key."""
    d = request.get_json(force=True) or {}
    if not d.get("image"):
        return jsonify(ok=False, error="No previous frame to build from."), 400
    shot = d.get("shot", "auto")
    shot_brief = {
        "angle": "This is the NEXT CAMERA ANGLE of the SAME moment: keep the "
                 "exact same character(s), wardrobe, location and lighting — "
                 "only the camera position/framing changes.",
        "location": "This is the SAME character(s) in a DIFFERENT location in "
                    "the same story/world: preserve identity, face, wardrobe "
                    "and style exactly, new setting.",
        "newchar": "This shot introduces a DIFFERENT character in the SAME "
                   "world/setting and style: keep location, lighting, era and "
                   "look consistent; returning characters stay identical.",
        "auto": "Choose the natural next beat, keeping the same character(s), "
                "wardrobe, setting and lighting so it reads as the same scene.",
    }.get(shot, "Choose the natural next beat, keeping continuity.")
    # ── Direct path: the user's own OpenAI key ──
    if OPENAI_API_KEY:
        ctxs = (d.get("context") or "")[:2000].strip()
        instr = (d.get("instructions") or "")[:2000].strip()
        text = ("This image is the LAST FRAME of the previous clip in a film. "
                "Propose the image-generation prompt for the NEXT clip's "
                f"starting frame. {shot_brief}\nThe previous frame will be "
                "attached to the image model as a visual reference, so refer "
                'back to it ("the same subject from the reference, same face '
                'and outfit") instead of re-describing returning subjects. '
                "Preserve identity and world exactly."
                + (f"\nStory context: {ctxs}" if ctxs else "")
                + (f"\nUser instructions: {instr}" if instr else "")
                + '\nRespond ONLY as JSON: {"image_prompt": "the full prompt '
                'to send to the image model", "intent": "one sentence on what '
                'happens in this next shot"}')
        try:
            rr = _requests.post("https://api.openai.com/v1/chat/completions",
                                headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                         "Content-Type": "application/json"},
                                json={"model": "gpt-4o-mini",
                                      "messages": [{"role": "user", "content": [
                                          {"type": "text", "text": text},
                                          {"type": "image_url",
                                           "image_url": {"url": d["image"]}}]}],
                                      "response_format": {"type": "json_object"},
                                      "max_completion_tokens": 700},
                                timeout=90)
            if rr.status_code >= 400:
                return jsonify(ok=False, error=f"OpenAI {rr.status_code}: "
                               f"{(rr.text or '')[:200]}"), 502
            jj = json.loads(rr.json()["choices"][0]["message"]["content"])
            return jsonify(ok=True, image_prompt=jj.get("image_prompt", ""),
                           intent=jj.get("intent", ""))
        except Exception as e:
            return jsonify(ok=False, error=f"propose failed — {e}"), 502
    # ── Server path: MissingLink's key ──
    try:
        img_uri = _ai_image(d["image"])
    except Exception as e:
        return jsonify(ok=False, error=f"could not read the frame — {e}"), 400
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/nextscene/propose",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={"image": img_uri,
                                 "context": (d.get("context") or "")[:2000],
                                 "instructions": (d.get("instructions") or "")[:2000],
                                 "shot": d.get("shot", "auto")},
                           timeout=90)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    if r.status_code == 404:
        return jsonify(ok=False, error="Two-stage next scene isn't on the "
                       "server yet \u2014 redeploy the MissingLink worker "
                       "(index.ts)."), 502
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "propose failed"), 502
    return jsonify(ok=True, image_prompt=j.get("image_prompt", ""),
                   intent=j.get("intent", ""))

def _openai_generate_direct(d):
    """Generate the next-scene image DIRECTLY on the user's own OpenAI key
    (OPENAI_API_KEY). Runs gpt-image-2 with the attached images as
    references, then a light self-check. Because this is the user calling
    OpenAI with their own key — not our server — it doesn't touch
    MissingLink tokens or our moderation path. Returns the same payload
    shape as the worker path. Raises on failure."""
    import base64 as _b64
    OA = "https://api.openai.com/v1"
    hdr = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    prompt = (d.get("image_prompt") or "")[:4000].strip()
    intent = (d.get("intent") or "")[:500].strip()
    quality = d.get("quality", "medium")
    if quality not in ("low", "medium", "high"):
        quality = "medium"
    refs = [im for im in (d.get("images") or [])
            if isinstance(im, str) and im.startswith("data:")][:6]

    def _b64_of(data_uri):
        return data_uri.split(",", 1)[1]

    # Generate: with references use the edits endpoint (multi-image),
    # otherwise a plain generation.
    if refs:
        files = []
        data = {"model": "gpt-image-2", "prompt": prompt,
                "size": "1024x1024", "quality": quality, "output_format": "png"}
        for i, u in enumerate(refs):
            try:
                raw = _b64.b64decode(_b64_of(u))
                files.append(("image[]", (f"ref{i}.png", raw, "image/png")))
            except Exception:
                pass
        rr = _requests.post(f"{OA}/images/edits", headers=hdr,
                            data=data, files=files, timeout=240)
    else:
        rr = _requests.post(f"{OA}/images/generations", headers=hdr,
                            json={"model": "gpt-image-2", "prompt": prompt,
                                  "size": "1024x1024", "quality": quality,
                                  "output_format": "png"}, timeout=240)
    if rr.status_code >= 400:
        raise RuntimeError(f"OpenAI image {rr.status_code}: {(rr.text or '')[:200]}")
    img_b64 = rr.json()["data"][0]["b64_json"]
    img_uri = "data:image/png;base64," + img_b64

    # Light self-check (best-effort; skip silently if it fails).
    works, verdict = True, ""
    try:
        vr = _requests.post(f"{OA}/chat/completions",
                            headers={**hdr, "Content-Type": "application/json"},
                            json={"model": "gpt-4o-mini",
                                  "messages": [{"role": "user", "content": [
                                      {"type": "text", "text":
                                       f'This is the proposed next shot. Intended: "{intent}". '
                                       "Will it work as a continuation (same characters/"
                                       "setting/lighting, sensible next moment)? Respond ONLY "
                                       'as JSON: {"works": true|false, "verdict": "one or two '
                                       'sentences for the user"}'},
                                      {"type": "image_url",
                                       "image_url": {"url": img_uri}}]}],
                                  "response_format": {"type": "json_object"},
                                  "max_completion_tokens": 300},
                            timeout=90)
        if vr.status_code < 400:
            vj = json.loads(vr.json()["choices"][0]["message"]["content"])
            works = vj.get("works", True) is not False
            verdict = (vj.get("verdict") or "").strip()
    except Exception:
        pass
    return {"image": img_uri, "image_prompt": prompt, "intent": intent,
            "corrected": False, "works": works, "verdict": verdict,
            "tokens": None, "tokens_per_gen": None, "own_key": True}


def _run_generate_job(job_id, d):
    """Stage 2 (tracked job): generate the next-scene image from the
    user's edited prompt + attached images.

    If the user set their own OPENAI_API_KEY, generate DIRECTLY on their
    key (no MissingLink tokens, no server-side path). Otherwise forward to
    MissingLink's worker (token-billed + moderated)."""
    job = jobs[job_id]
    # ── Direct path: the user's own OpenAI key ──
    if OPENAI_API_KEY:
        try:
            job.update(status="running", stage="generating on your OpenAI key",
                       progress=40)
            payload = _openai_generate_direct(d)
            job.update(status="done", stage="complete", progress=100,
                       result=json.dumps(payload))
        except Exception as e:
            job.update(status="error", stage="failed", error=str(e)[:300])
        return
    try:
        job.update(status="running", stage="preparing images", progress=10)
        # Convert every attached image to a hosted URI the worker accepts.
        imgs = []
        for im in (d.get("images") or [])[:6]:
            try:
                imgs.append(_ai_image(im))
            except Exception:
                pass
        job.update(stage="generating & self-checking image", progress=45)
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/nextscene/generate",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={"image_prompt": (d.get("image_prompt") or "")[:4000],
                                 "intent": (d.get("intent") or "")[:500],
                                 "context": (d.get("context") or "")[:2000],
                                 "quality": d.get("quality", "medium"),
                                 "images": imgs},
                           timeout=240)
        if r.status_code == 404:
            job.update(status="error", stage="failed",
                       error="Image generate isn't on the server yet \u2014 "
                       "redeploy the MissingLink worker (index.ts).")
            return
        if r.status_code in (401, 403):
            _ml_forward_error(r)
            job.update(status="error", stage="failed",
                       error="Sign in to generate images.")
            return
        try:
            j = r.json()
        except Exception:
            job.update(status="error", stage="failed",
                       error=f"MissingLink {r.status_code}: {(r.text or '')[:200]}")
            return
        # Out of tokens: surface a buy prompt via a structured error.
        if r.status_code == 402 or j.get("error") == "insufficient_tokens":
            job.update(status="error", stage="out of tokens",
                       error="insufficient_tokens",
                       tokens=j.get("tokens", 0),
                       tokens_per_gen=j.get("tokens_per_gen", 100))
            return
        # Blocked by the content policy (tokens were refunded server-side).
        if r.status_code == 422 or j.get("error") == "content_blocked":
            job.update(status="error", stage="blocked",
                       error=j.get("message")
                       or "This request was blocked by our content policy.")
            return
        # Safety screen couldn't run (fail-closed; refunded server-side).
        if r.status_code == 503 or j.get("error") == "moderation_unavailable":
            job.update(status="error", stage="safety check unavailable",
                       error=j.get("message")
                       or "Safety check temporarily unavailable \u2014 try again.")
            return
        if not j.get("ok"):
            job.update(status="error", stage="failed",
                       error=j.get("error") or "generate failed")
            return
        # Refresh the cached token balance for the UI.
        if j.get("tokens") is not None:
            ML["tokens"] = j.get("tokens")
        job.update(status="done", stage="complete", progress=100,
                   result=json.dumps({k: j.get(k) for k in
                       ("image", "image_prompt", "intent", "corrected",
                        "works", "verdict", "tokens", "tokens_per_gen")}))
    except Exception as e:
        job.update(status="error", stage="failed", error=str(e)[:300])

@app.route("/api/agent", methods=["POST"])
def _agent():
    """The storyboard agent. Sees the full script + conversation history +
    the user's request, returns {reply, actions[]}. Runs on the user's own
    OpenAI key if set, else forwards to MissingLink's worker.

    Images in the script are stripped to booleans before sending to the LLM
    (it doesn't need the base64; it edits by clip number)."""
    d = request.get_json(force=True) or {}
    msg = (d.get("message") or "").strip()
    if not msg:
        return jsonify(ok=False, error="empty message"), 400
    script = d.get("script") or {}
    history = d.get("history") or []

    # Strip heavy image data — replace with presence flags so the model
    # knows an image exists without paying for the base64.
    def _lighten(sc):
        s = json.loads(json.dumps(sc))   # deep copy
        if s.get("first_image"):
            s["first_image"] = "<present>"
        for cl in s.get("clips", []):
            if cl.get("start_image"):
                cl["start_image"] = "<present>"
        return s
    light = _lighten(script)

    sys_prompt = (
        "You are the storyboard agent for a video studio. You see the full "
        "SCRIPT (base scene, engine, fps, speakers with voices, and each clip's "
        "scene text, length, and dialogue lines) and the conversation so far. "
        "The user asks you to change things. Reply conversationally AND return "
        "a list of ACTIONS that make the changes.\n\n"
        "Clips are numbered from 1. Dialogue lines within a clip are numbered "
        "from 1 (in order). Speakers are named (uppercase). Length can be set "
        "in seconds or frames; prefer seconds. To make dialogue longer/shorter "
        "edit the lines; length auto-derives unless you set it explicitly.\n\n"
        "Available action types (use exact field names):\n"
        '- {"type":"set_base_scene","scene":"..."}\n'
        '- {"type":"set_engine","engine":"wan|wan22|ltx"}\n'
        '- {"type":"set_fps","fps":24}\n'
        '- {"type":"set_clip_scene","clip":2,"scene":"..."}\n'
        '- {"type":"reset_clip_scene","clip":2}\n'
        '- {"type":"set_clip_length","clip":2,"seconds":6}  (or "frames":49)\n'
        '- {"type":"auto_clip_length","clip":2}\n'
        '- {"type":"add_speaker","name":"VILLAIN","voice":"gravelly, low, menacing"}\n'
        '- {"type":"rename_speaker","from":"SPEAKER 1","to":"HERO"}\n'
        '- {"type":"set_voice","name":"HERO","voice":"..."}\n'
        '- {"type":"remove_speaker","name":"EXTRA"}\n'
        '- {"type":"add_line","clip":2,"speaker":"HERO","text":"..."}\n'
        '- {"type":"set_line","clip":2,"index":1,"text":"...","speaker":"HERO"}\n'
        '- {"type":"remove_line","clip":2,"index":1}\n'
        '- {"type":"clear_clip_dialogue","clip":2}\n'
        '- {"type":"add_clip","scene":"...","lines":[{"speaker":"HERO","text":"..."}]}\n'
        '- {"type":"delete_clip","clip":3}\n\n'
        "Only include actions that are needed. If the user just asks a question, "
        "return an empty actions list. Respond ONLY as JSON: "
        '{"reply":"what you did / your answer","actions":[...]}')

    user_content = ("SCRIPT:\n" + json.dumps(light)
                    + "\n\nUSER REQUEST:\n" + msg)
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-12:]:
        role = "user" if h.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": (h.get("content") or "")[:2000]})
    messages.append({"role": "user", "content": user_content})

    # ── Direct path: user's own OpenAI key ──
    if OPENAI_API_KEY:
        try:
            rr = _requests.post("https://api.openai.com/v1/chat/completions",
                                headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                         "Content-Type": "application/json"},
                                json={"model": "gpt-4o",
                                      "messages": messages,
                                      "response_format": {"type": "json_object"},
                                      "max_completion_tokens": 1500},
                                timeout=90)
            if rr.status_code >= 400:
                return jsonify(ok=False, error=f"OpenAI {rr.status_code}: "
                               f"{(rr.text or '')[:200]}"), 502
            j = json.loads(rr.json()["choices"][0]["message"]["content"])
            return jsonify(ok=True, reply=j.get("reply", ""),
                           actions=j.get("actions", []))
        except Exception as e:
            return jsonify(ok=False, error=f"agent failed — {e}"), 502

    # ── Server path: MissingLink worker ──
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/agent",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={"messages": messages}, timeout=90)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    if r.status_code == 404:
        return jsonify(ok=False, error="Agent isn't on the server yet \u2014 "
                       "redeploy the MissingLink worker (index.ts), or set your "
                       "own OPENAI_API_KEY to run it directly."), 502
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "agent failed"), 502
    return jsonify(ok=True, reply=j.get("reply", ""), actions=j.get("actions", []))

@app.route("/api/editor/export", methods=["POST"])
def _editor_export():
    """Stitch a timeline EDL into one MP4 with ffmpeg. Handles per-clip
    trim (in/out), crossfades between clips, per-clip fade-in, and extra
    audio tracks mixed over the video. Video clips are referenced by the
    URLs the app already serves; audio may be data URLs (user uploads)."""
    import tempfile, base64 as _b64, uuid as _uuid
    d = request.get_json(force=True) or {}
    vids = d.get("video") or []
    if not vids:
        return jsonify(ok=False, error="no clips to export"), 400
    work = tempfile.mkdtemp(prefix="edl_")
    try:
        # 1) Materialise each video clip locally, trimmed to [in,out].
        #    Clips are data: URLs (that's how the app stores rendered video).
        seg_paths = []
        for i, c in enumerate(vids):
            url = c.get("url", "")
            src = os.path.join(work, f"src_{i}.mp4")
            if url.startswith("data:"):
                with open(src, "wb") as f:
                    f.write(_b64.b64decode(url.split(",", 1)[1]))
            elif url.startswith("http"):
                try:
                    rr = _requests.get(url, timeout=60)
                    with open(src, "wb") as f:
                        f.write(rr.content)
                except Exception as e:
                    return jsonify(ok=False,
                                   error=f"could not read clip {i+1}: {e}"), 502
            else:
                return jsonify(ok=False,
                               error=f"clip {i+1} has no usable source"), 400
            ss = float(c.get("in", 0) or 0)
            oo = float(c.get("out", 0) or 0)
            seg = os.path.join(work, f"seg_{i}.mp4")
            dur = max(0.1, oo - ss)
            cmd = ["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", src,
                   "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-r", "24", seg]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(seg):
                return jsonify(ok=False,
                               error=f"trim failed on clip {i+1}: "
                               f"{(r.stderr or '')[-200:]}"), 500
            seg_paths.append((seg, c))

        # 2) Concatenate (with crossfades where requested).
        out_v = os.path.join(work, "video.mp4")
        any_xfade = any(float(c.get("xfade", 0) or 0) > 0 for _, c in seg_paths[1:])
        if not any_xfade or len(seg_paths) == 1:
            # Simple concat via demuxer.
            listf = os.path.join(work, "list.txt")
            with open(listf, "w") as f:
                for seg, _ in seg_paths:
                    f.write(f"file '{seg}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", listf, "-c:v", "libx264", "-pix_fmt",
                            "yuv420p", "-c:a", "aac", out_v],
                           capture_output=True, text=True)
        else:
            # Chain xfade filters across all segments.
            inputs = []
            for seg, _ in seg_paths:
                inputs += ["-i", seg]
            fc = []
            prev = "0:v"
            prev_a = "0:a"
            offset = 0.0
            # running duration of the accumulated left side
            durs = [max(0.1, float(c.get("out", 0)) - float(c.get("in", 0)))
                    for _, c in seg_paths]
            acc = durs[0]
            for i in range(1, len(seg_paths)):
                xf = float(seg_paths[i][1].get("xfade", 0) or 0)
                xf = min(xf, durs[i] - 0.05, acc - 0.05) if xf > 0 else 0
                if xf <= 0:
                    # hard cut via concat of two labels
                    fc.append(f"[{prev}][{i}:v]concat=n=2:v=1:a=0[v{i}]")
                    fc.append(f"[{prev_a}][{i}:a]concat=n=2:v=0:a=1[a{i}]")
                    acc += durs[i]
                else:
                    off = acc - xf
                    fc.append(f"[{prev}][{i}:v]xfade=transition=fade:"
                              f"duration={xf:.3f}:offset={off:.3f}[v{i}]")
                    fc.append(f"[{prev_a}][{i}:a]acrossfade=d={xf:.3f}[a{i}]")
                    acc += durs[i] - xf
                prev = f"v{i}"; prev_a = f"a{i}"
            cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(fc),
                   "-map", f"[{prev}]", "-map", f"[{prev_a}]",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_v]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(out_v):
                # fall back to simple concat
                listf = os.path.join(work, "list.txt")
                with open(listf, "w") as f:
                    for seg, _ in seg_paths:
                        f.write(f"file '{seg}'\n")
                subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                "-i", listf, "-c:v", "libx264", "-pix_fmt",
                                "yuv420p", "-c:a", "aac", out_v],
                               capture_output=True, text=True)

        # 3) Mix in extra audio tracks (each may hold placed clips).
        audio_tracks = d.get("audio") or []
        placed = []
        for ti, tr in enumerate(audio_tracks):
            if tr.get("muted"):
                continue
            for ci, ac in enumerate(tr.get("clips") or []):
                data = ac.get("data", "")
                ap = os.path.join(work, f"aud_{ti}_{ci}.m4a")
                try:
                    if data.startswith("data:"):
                        with open(ap, "wb") as f:
                            f.write(_b64.b64decode(data.split(",", 1)[1]))
                    elif data.startswith("http"):
                        rr = _requests.get(data, timeout=60)
                        with open(ap, "wb") as f:
                            f.write(rr.content)
                    else:
                        continue
                    placed.append((ap, float(ac.get("start", 0) or 0),
                                   float(ac.get("in", 0) or 0),
                                   float(ac.get("out", 0) or 0)))
                except Exception:
                    continue

        final = os.path.join(work, "final.mp4")
        if placed:
            inputs = ["-i", out_v]
            fc = []
            amix_labels = ["0:a"]
            for k, (ap, start, ain, aout) in enumerate(placed, start=1):
                inputs += ["-i", ap]
                trim = ""
                if aout > ain:
                    trim = f"atrim={ain:.3f}:{aout:.3f},asetpts=PTS-STARTPTS,"
                fc.append(f"[{k}:a]{trim}adelay={int(start*1000)}|{int(start*1000)}[a{k}]")
                amix_labels.append(f"a{k}")
            fc.append("".join(f"[{l}]" for l in amix_labels)
                      + f"amix=inputs={len(amix_labels)}:duration=longest[aout]")
            cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(fc),
                   "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                   "-c:a", "aac", final]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(final):
                final = out_v
        else:
            final = out_v

        # 4) Return the stitched video as a data URL (how the app serves video).
        with open(final, "rb") as f:
            out_b64 = _b64.b64encode(f.read()).decode()
        return jsonify(ok=True, url="data:video/mp4;base64," + out_b64)
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:300]), 500
    finally:
        try:
            import shutil as _sh; _sh.rmtree(work, ignore_errors=True)
        except Exception:
            pass


@app.route("/api/editimage", methods=["POST"])
def _editimage():
    """Edit an existing image with a text instruction (gpt-image-2 edit).
    Own key = direct + free; otherwise forwards to the worker (token-billed
    + moderated, same as generation)."""
    d = request.get_json(force=True) or {}
    img = d.get("image") or ""
    instr = (d.get("instruction") or "").strip()
    if not img or not instr:
        return jsonify(ok=False, error="image and instruction required"), 400
    quality = d.get("quality", "medium")
    if quality not in ("low", "medium", "high"):
        quality = "medium"
    # ── Direct path: user's own OpenAI key ──
    if OPENAI_API_KEY:
        import base64 as _b64
        try:
            b64 = img.split(",", 1)[1]
            raw = _b64.b64decode(b64)
            files = [("image", ("img.png", raw, "image/png"))]
            # Optional reference images guide the edit (added as extra
            # image[] inputs, which gpt-image-2 edits supports).
            for i, ref in enumerate(d.get("references") or []):
                try:
                    rraw = _b64.b64decode(ref.split(",", 1)[1])
                    files.append(("image", (f"ref{i}.png", rraw, "image/png")))
                except Exception:
                    pass
            data = {"model": "gpt-image-2", "prompt": instr,
                    "size": "1024x1024", "quality": quality,
                    "output_format": "png"}
            rr = _requests.post("https://api.openai.com/v1/images/edits",
                                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                                data=data, files=files, timeout=240)
            if rr.status_code >= 400:
                return jsonify(ok=False, error=f"OpenAI {rr.status_code}: "
                               f"{(rr.text or '')[:200]}"), 502
            out = "data:image/png;base64," + rr.json()["data"][0]["b64_json"]
            return jsonify(ok=True, image=out, own_key=True)
        except Exception as e:
            return jsonify(ok=False, error=f"edit failed — {e}"), 502
    # ── Server path: MissingLink worker ──
    try:
        img_uri = _ai_image(img)
    except Exception as e:
        return jsonify(ok=False, error=f"could not read the image — {e}"), 400
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/editimage",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={"image": img_uri, "instruction": instr[:2000],
                                 "quality": quality}, timeout=240)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    if r.status_code == 404:
        return jsonify(ok=False, error="Image edit isn't on the server yet "
                       "\u2014 redeploy the MissingLink worker (index.ts)."), 502
    if r.status_code in (401, 403):
        gate = _ml_forward_error(r)
        if gate:
            return gate
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if r.status_code == 402 or j.get("error") == "insufficient_tokens":
        return jsonify(ok=False, error="insufficient_tokens",
                       message=j.get("message") or "Not enough tokens.",
                       tokens=j.get("tokens", 0),
                       tokens_per_gen=j.get("tokens_per_gen", 100)), 402
    if r.status_code == 422 or j.get("error") == "content_blocked":
        return jsonify(ok=False, error=j.get("message")
                       or "Blocked by our content policy."), 422
    if not j.get("ok"):
        return jsonify(ok=False, error=j.get("error") or "edit failed"), 502
    if j.get("tokens") is not None:
        ML["tokens"] = j.get("tokens")
    return jsonify(ok=True, image=j.get("image"),
                   tokens=j.get("tokens"), tokens_per_gen=j.get("tokens_per_gen", 100))

# ── Character-consistency face swap (Qwen edit + BFS faceswap LoRA) ──────
# Runs on the deployed MissingLink Qwen worker via /api/studio/*, billed to
# the signed-in user's tokens. Per clip: upload the START FRAME + the
# character REFERENCE (BFS's inverted order: Image1=frame/target,
# Image2=face source), submit a mode:'edit' job with the faceswap +
# lightning LoRAs and the trained "head_swap:" prompt, then the frontend
# polls and writes the result back as that clip's start image.
_FACESWAP_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its "
    "lighting, environment, and background. remove the head from Picture 1 "
    "completely and replace it with the head from Picture 2, strictly "
    "preserving the hair, eye color, and nose structure of Picture 2. copy "
    "the eye direction, head rotation, and micro-expressions from Picture 1."
)

def _studio_headers():
    return {"Authorization": f"Bearer {ML.get('key') or ''}"}

def _studio_upload(data_uri):
    """Upload a data-URI image to /api/studio/upload; return its id."""
    import base64 as _b64
    raw = _b64.b64decode(_ai_image(data_uri).split(",", 1)[1])
    files = {"image": ("frame.png", raw, "image/png")}
    r = _requests.post(f"{MISSINGLINK_API}/api/studio/upload",
                       headers=_studio_headers(), files=files, timeout=120)
    j = r.json()
    if not j.get("id"):
        raise RuntimeError(j.get("error") or f"upload failed ({r.status_code})")
    return j["id"]

@app.route("/api/qwenedit/submit", methods=["POST"])
def _qwenedit_submit():
    """Qwen image-edit with user-set LoRA strengths, on the studio worker.
    Image order: [target image, ...references]. If faceswap>0 the first
    reference is the face source (BFS inverted order handled by prompt)."""
    d = request.get_json(force=True) or {}
    img = d.get("image") or ""
    refs = d.get("references") or []
    instr = (d.get("instruction") or "").strip()
    if not img:
        return jsonify(ok=False, error="no image"), 400
    def _f(k):
        try: return max(0.0, min(1.5, float(d.get(k, 0))))
        except Exception: return 0.0
    fs, ang, skin, ups = _f("lora_faceswap"), _f("lora_angles"), _f("lora_skin"), _f("lora_upscale")
    # Prompt: faceswap uses the trained head_swap trigger; else the user's
    # instruction (or a neutral edit).
    if fs > 0:
        prompt = _FACESWAP_PROMPT + (" " + instr if instr else "")
    else:
        prompt = instr or "edit the image as instructed."
    try:
        # image_ids[0] = target; for faceswap, ref[0] must be the face source.
        ids = [_studio_upload(img)]
        for rimg in refs[:3]:
            ids.append(_studio_upload(rimg))
    except Exception as e:
        return jsonify(ok=False, error=f"upload failed \u2014 {e}"), 502
    payload = {"mode": "edit", "image_ids": ids, "prompt": prompt,
               "negative_prompt": "", "randomize_seed": True,
               "guidance_scale": 1.0, "inference_steps": 4,
               "lora_lightning": 1.0, "lora_faceswap": fs,
               "lora_angles": ang, "lora_skin": skin, "lora_upscale": ups}
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/studio/generate",
                           headers={**_studio_headers(),
                                    "Content-Type": "application/json"},
                           json=payload, timeout=60)
        j = r.json()
    except Exception as e:
        return jsonify(ok=False, error=f"submit failed \u2014 {e}"), 502
    if r.status_code == 402 or j.get("error") == "insufficient_tokens":
        return jsonify(ok=False, error="insufficient_tokens",
                       tokens=j.get("tokens", 0)), 402
    jid = j.get("job_id") or j.get("id")
    if not jid:
        return jsonify(ok=False, error=j.get("error") or "no job id"), 502
    if j.get("tokens_remaining") is not None:
        ML["tokens"] = j.get("tokens_remaining")
    local_id = uuid.uuid4().hex[:12]
    jobs[local_id] = {"status": "running", "progress": 5,
                      "stage": "GPU warming up", "result": None,
                      "error": None, "cancel": False, "kind": "qwenedit",
                      "prompt": "Qwen edit", "thumb": "", "ts": time.time(),
                      "_remote": jid}
    threading.Thread(target=_faceswap_track, args=(local_id, jid),
                     daemon=True).start()
    return jsonify(ok=True, job_id=local_id, tokens=j.get("tokens_remaining"))

@app.route("/api/faceswap/submit", methods=["POST"])
def _faceswap_submit():
    """Upload frame+reference and submit ONE faceswap edit job. Returns the
    studio job_id for the frontend to poll. Token-billed server-side."""
    d = request.get_json(force=True) or {}
    frame = d.get("frame") or ""       # the clip's start frame (target)
    ref = d.get("reference") or ""     # the character reference (face source)
    if not frame or not ref:
        return jsonify(ok=False, error="frame and reference required"), 400
    try:
        strength = float(d.get("faceswap", 1.0))
    except Exception:
        strength = 1.0
    strength = max(0.0, min(1.5, strength))
    try:
        # BFS inverted order: image_ids[0] = frame/target, [1] = face source.
        frame_id = _studio_upload(frame)
        ref_id = _studio_upload(ref)
    except Exception as e:
        return jsonify(ok=False, error=f"upload failed \u2014 {e}"), 502
    payload = {
        "mode": "edit",
        "image_ids": [frame_id, ref_id],
        "prompt": _FACESWAP_PROMPT,
        "negative_prompt": "",
        "randomize_seed": True,
        "guidance_scale": 1.0,
        "inference_steps": 4,          # Lightning 4-step
        "lora_lightning": 1.0,         # MUST be on (distilled sampler)
        "lora_faceswap": strength,     # BFS Head V5
        "lora_angles": 0.0,
        "width": 0, "height": 0,
    }
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/studio/generate",
                           headers={**_studio_headers(),
                                    "Content-Type": "application/json"},
                           json=payload, timeout=60)
    except Exception as e:
        return jsonify(ok=False, error=f"submit failed \u2014 {e}"), 502
    if r.status_code in (401, 403):
        gate = _ml_forward_error(r)
        if gate:
            return gate
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"studio {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if r.status_code == 402 or j.get("error") == "insufficient_tokens":
        return jsonify(ok=False, error="insufficient_tokens",
                       message=j.get("message") or "Not enough tokens.",
                       tokens=j.get("tokens", 0)), 402
    if j.get("error") == "content_blocked":
        return jsonify(ok=False, error="Blocked by content policy."), 422
    jid = j.get("job_id") or j.get("id")
    if not jid:
        return jsonify(ok=False, error=j.get("error") or "no job id"), 502
    if j.get("tokens_remaining") is not None:
        ML["tokens"] = j.get("tokens_remaining")
    # Create a LOCAL job so this shows up in the Queue/History panel with
    # live progress, then background-poll the remote studio job into it.
    local_id = uuid.uuid4().hex[:12]
    jobs[local_id] = {"status": "running", "progress": 5,
                      "stage": "GPU warming up", "result": None,
                      "error": None, "cancel": False, "kind": "faceswap",
                      "prompt": f"Face swap \u2014 clip {d.get('clip','?')}",
                      "thumb": frame[:0] or "", "ts": time.time(),
                      "_remote": jid, "_clip": d.get("clip")}
    threading.Thread(target=_faceswap_track, args=(local_id, jid),
                     daemon=True).start()
    return jsonify(ok=True, job_id=local_id, remote_id=jid,
                   tokens=j.get("tokens_remaining"))

_FS_STAGE = {"fetching": ("loading inputs", 20),
             "preparing": ("preparing", 30),
             "encoding_prompt": ("encoding prompt", 40),
             "encoding_image": ("encoding image", 50),
             "generating": ("generating", 70),
             "decoding": ("finishing", 90),
             "upscaling": ("upscaling", 95)}

def _faceswap_track(local_id, remote_id):
    """Background: poll the remote studio job, mirror its state into the
    local job so the standard queue poller/UI shows it."""
    import time as _t
    for _ in range(180):                 # ~6 min cap
        job = jobs.get(local_id)
        if not job or job.get("cancel"):
            return
        try:
            r = _requests.get(f"{MISSINGLINK_API}/api/studio/job/{remote_id}",
                              headers=_studio_headers(), timeout=30)
            j = r.json()
        except Exception:
            _t.sleep(2); continue
        stt = j.get("status")
        if stt == "queued":
            pos = j.get("position")
            job.update(stage=(f"#{pos} in queue" if pos else "queued"),
                       progress=3)
        elif stt == "running":
            stg = j.get("stage")
            if not stg:
                job.update(stage="GPU warming up", progress=10)
            else:
                label, pct = _FS_STAGE.get(stg, (stg, 60))
                if stg == "generating" and j.get("total_steps"):
                    label += f" {j.get('step',0)}/{j.get('total_steps')}"
                job.update(stage=label, progress=pct)
        elif stt == "done":
            res = j.get("result") or {}
            url = res.get("url") if isinstance(res, dict) else res
            if url and url.startswith("/"):
                url = MISSINGLINK_API + url
            # Fetch the result image as a data URL for history + write-back.
            data_url = url
            try:
                ir = _requests.get(url, timeout=60)
                if ir.status_code == 200:
                    import base64 as _b64
                    ct = ir.headers.get("Content-Type", "image/png")
                    data_url = (f"data:{ct};base64,"
                                + _b64.b64encode(ir.content).decode())
            except Exception:
                pass
            job.update(status="done", stage="complete", progress=100,
                       result=data_url, _result_url=url)
            return
        elif stt == "error":
            job.update(status="error", stage="failed",
                       error=j.get("error") or "job failed")
            return
        _t.sleep(2)
    job = jobs.get(local_id)
    if job and job.get("status") == "running":
        job.update(status="error", stage="timed out",
                   error="face swap timed out")

@app.route("/api/faceswap/poll/<job_id>", methods=["GET"])
def _faceswap_poll(job_id):
    """Poll a studio job; when done, return the result image URL."""
    try:
        r = _requests.get(f"{MISSINGLINK_API}/api/studio/job/{job_id}",
                          headers=_studio_headers(), timeout=30)
        j = r.json()
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    st = j.get("status")
    out = {"ok": True, "status": st, "position": j.get("position"),
           "stage": j.get("stage"), "step": j.get("step") or 0,
           "total_steps": j.get("total_steps") or 0}
    if st == "done":
        res = j.get("result") or {}
        url = res.get("url") if isinstance(res, dict) else res
        if url and url.startswith("/"):
            url = MISSINGLINK_API + url
        out["url"] = url
    elif st == "error":
        out["ok"] = False
        out["error"] = j.get("error") or "job failed"
    return jsonify(out)

@app.route("/api/nextscene/generate", methods=["POST"])
def _nextscene_generate():
    """Stage 2: queue the token-billed image generation as a tracked job
    (progress in the queue). Result fetched from /api/result."""
    d = request.get_json(force=True) or {}
    if not (d.get("image_prompt") or "").strip():
        return jsonify(ok=False, error="Write a prompt first."), 400
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": "queued",
                    "result": None, "error": None, "cancel": False,
                    "kind": "image", "prompt": "Next-scene image",
                    "thumb": "", "ts": time.time()}
    threading.Thread(target=_run_generate_job, args=(job_id, d),
                     daemon=True).start()
    return jsonify(ok=True, job_id=job_id)

@app.route("/api/checkout", methods=["POST"])
def _checkout():
    """Buy tokens: forward to the worker's Stripe checkout, return {url}
    for the browser to redirect to."""
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/checkout",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json={}, timeout=20)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    if r.status_code == 404:
        return jsonify(ok=False, error="Buy-tokens isn't on the server yet "
                       "\u2014 redeploy the MissingLink worker (index.ts)."), 502
    try:
        j = r.json()
    except Exception:
        return jsonify(ok=False, error=f"MissingLink {r.status_code}: "
                       f"{(r.text or '')[:200]}"), 502
    if not j.get("ok") or not j.get("url"):
        return jsonify(ok=False, error=j.get("message") or j.get("error")
                       or "checkout failed"), 502
    return jsonify(ok=True, url=j.get("url"))

@app.route("/api/tokens")
def _tokens():
    """Current token balance for the signed-in user (re-fetched live).
    If the user supplied their own OpenAI key, image gen is on their key —
    no MissingLink tokens apply."""
    if OPENAI_API_KEY:
        return jsonify(ok=True, tokens=None, tokens_per_gen=None, own_key=True)
    try:
        ok, data, err = _ml_validate(ML.get("key") or "")
        if ok and data is not None:
            ML["tokens"] = data.get("tokens")
            ML["tokens_per_gen"] = data.get("tokens_per_gen", 100)
    except Exception:
        pass
    return jsonify(ok=True, tokens=ML.get("tokens"),
                   tokens_per_gen=ML.get("tokens_per_gen", 100), own_key=False)

@app.route("/api/result/<job_id>")
def _result(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="no result"), 404
    return jsonify(result=job["result"])

@app.route("/api/cancel/<job_id>", methods=["POST"])
def _cancel(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(ok=False, error="unknown job"), 404
    if job.get("status") in ("done", "error", "cancelled"):
        return jsonify(ok=True, already=True)
    # A running job checks this flag on every diffusion step (via the
    # callback) and between clips, so it stops at the next step
    # boundary; a still-queued job is skipped by the dispatcher before
    # it ever starts.
    job["cancel"] = True
    job.update(stage="cancelling")
    return jsonify(ok=True)

# ---- Civitai LoRA search (proxied through missinglink.build) -------------
# Search/tags/thumb/download all run on the MissingLink worker with ITS
# Civitai API key, authenticated by the member\'s MissingLink key. The
# response shape is identical to the old in-cell implementation, so the
# frontend is unchanged.
def _ml_get(path, **kw):
    kw.setdefault("timeout", 30)
    headers = kw.pop("headers", {})
    headers["Authorization"] = f"Bearer {ML.get('key') or ''}"
    return _requests.get(f"{MISSINGLINK_API}{path}", headers=headers, **kw)

@app.route("/api/civitai/search", methods=["POST"])
def _civitai_search():
    try:
        r = _requests.post(f"{MISSINGLINK_API}/api/notebook/civitai/search",
                           headers={"Authorization": f"Bearer {ML.get('key') or ''}"},
                           json=request.get_json(force=True) or {},
                           timeout=30)
    except Exception as e:
        return jsonify(ok=False, error=f"MissingLink request failed — {e}"), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    return Response(r.content, status=r.status_code,
                    mimetype="application/json")

@app.route("/api/civitai/model")
def _civitai_model():
    from urllib.parse import quote as _q
    mid = request.args.get("id", "")
    nsfw = request.args.get("nsfw", "false")
    if not mid:
        return jsonify(ok=False, error="id required"), 400
    try:
        r = _ml_get(f"/api/notebook/civitai/model?id={_q(mid, safe='')}"
                    f"&nsfw={_q(nsfw, safe='')}", timeout=25)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    return Response(r.content, status=r.status_code,
                    mimetype="application/json")

@app.route("/api/civitai/tags")
def _civitai_tags():
    try:
        r = _ml_get("/api/notebook/civitai/tags", timeout=20)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    gate = _ml_forward_error(r)
    if gate:
        return gate
    return Response(r.content, status=r.status_code,
                    mimetype="application/json")

@app.route("/api/civitai/thumb")
def _civitai_thumb():
    url = request.args.get("url", "")
    low = url.lower()
    if not low.startswith("https://") or "civitai" not in low:
        return jsonify(error="bad url"), 400
    try:
        from urllib.parse import quote as _q
        r = _ml_get("/api/notebook/civitai/thumb?url=" + _q(url, safe=""),
                    timeout=25)
        gate = _ml_forward_error(r)
        if gate:
            return gate
        r.raise_for_status()
        return Response(r.content,
                        mimetype=r.headers.get("Content-Type", "image/jpeg"))
    except Exception as e:
        return jsonify(error=str(e)), 502

@app.route("/api/console")
def _console():
    return jsonify(lines=list(console_lines)[-200:])

@app.route("/api/hw")
def _hw():
    info = {"gpu": None, "vram_used": 0, "vram_total": 0,
            "residency": STATE["residency"]}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_used"] = round((total - free) / 1e9, 2)
        info["vram_total"] = round(total / 1e9, 2)
    return jsonify(info)

# ---- frontend ----------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MissingLink Video Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --gold:#E8A917;--gold-light:#F5C842;--gold-dark:#C48E0E;--gold-dim:rgba(232,169,23,.08);
  --black:#09090B;--surface:#111113;--surface-2:#18181B;--surface-3:#1F1F23;
  --border:#27272A;--border-light:#3F3F46;--border-strong:#3F3F46;
  --white:#FAFAF9;--text:#E4E4E7;--text-dim:#A1A1AA;--text-muted:#71717A;
  --red:#EF4444;--green:#22C55E;--blue:#3B82F6;
  --font-display:'Space Mono','JetBrains Mono',monospace;
  --font-mono:'JetBrains Mono','Fira Code',monospace;
  --font-body:'DM Sans','Segoe UI',sans-serif;
  --sidebar-w:320px;--header-h:50px;
}
html,body{height:100%;overflow:hidden}
body{background:var(--black);color:var(--text);font-family:var(--font-body);
  font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased}

.app{display:grid;
  grid-template-columns:var(--sidebar-w) 1fr;
  grid-template-rows:var(--header-h) minmax(0,1fr) auto;
  grid-template-areas:"header header" "sidebar main" "dock dock";
  height:100vh;width:100vw;overflow:hidden}

/* ── header ── */
.app-header{grid-area:header;display:flex;align-items:center;padding:0 16px;
  background:var(--surface);border-bottom:1px solid var(--border);z-index:100;gap:11px}
.app-header img{height:30px;width:30px;border-radius:50%;object-fit:cover}
.app-header h1{font-family:var(--font-display);font-size:12px;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;color:var(--text-dim);white-space:nowrap}
.app-header h1 span{color:var(--gold)}
.stage-tabs{display:inline-flex;gap:2px;align-items:center;margin-left:16px}
.stage-tab{display:inline-flex;align-items:center;gap:6px;background:transparent;
  border:none;color:var(--text-muted);cursor:pointer;padding:7px 14px;border-radius:8px;
  font-family:var(--font-mono);font-size:10px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;transition:all .15s;line-height:1}
.stage-tab:hover{color:var(--text);background:var(--surface-2)}
.stage-tab.active{background:var(--gold);color:var(--black)}
.stage-tab-icon{font-size:13px;line-height:1}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.hdr-badge{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
  padding:4px 9px;border:1px solid var(--border);border-radius:5px;display:flex;
  align-items:center;gap:6px;white-space:nowrap}
.hdr-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--text-muted);
  transition:background .3s}
.hdr-badge .dot.on{background:var(--green);animation:pulse 2s infinite}
.hdr-badge .dot.warm{background:var(--gold);animation:pulse 1.2s infinite}
.hdr-badge .dot.cold{background:var(--text-muted)}
.hdr-badge .dot.off{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.user-menu-trigger{display:flex;align-items:center;gap:6px;padding:3px 11px 3px 3px;
  border-radius:14px;background:var(--surface-2);border:1px solid var(--border);
  color:var(--text-muted);font-family:var(--font-mono);font-size:10px}
.user-menu-fallback{width:22px;height:22px;border-radius:50%;background:var(--surface-3);
  color:var(--text-muted);display:flex;align-items:center;justify-content:center;
  border:1px solid var(--border);font-size:11px}
.user-menu-trigger-email{max-width:160px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;line-height:1}

/* ── sidebar ── */
.sidebar{grid-area:sidebar;background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;min-height:0;overflow:hidden}
.sidebar-scroll{flex:1;overflow-y:auto;min-height:0;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.mode-sec{padding:12px 16px;border-bottom:1px solid var(--border)}
.mode-cap{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted);margin-bottom:8px}
.mode-toggle{display:flex;gap:4px;background:var(--surface-3);border-radius:6px;padding:3px}
.mode-btn{flex:1;padding:9px 0;background:none;border:none;border-radius:4px;
  color:var(--text-muted);font-family:var(--font-mono);font-size:10px;font-weight:700;
  letter-spacing:.3px;cursor:pointer;transition:all .15s;text-align:center}
.mode-btn:hover{color:var(--text)}
.mode-btn.active{background:var(--gold);color:var(--black)}
.mode-hint{font-family:var(--font-mono);font-size:9px;color:var(--text-muted);
  letter-spacing:.3px;margin-top:8px;line-height:1.5}
.sec{padding:14px 16px;border-bottom:1px solid var(--border)}
.sec-label{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted);margin-bottom:10px;display:flex;
  align-items:center;gap:6px}
.sec-label .icon{color:var(--gold)}
.sec-label .c{margin-left:auto;color:var(--text-muted);letter-spacing:.5px;
  text-transform:none;font-weight:500}
.ta{width:100%;background:var(--surface-3);border:1px solid var(--border);border-radius:4px;
  padding:9px 10px;color:var(--text);font-family:var(--font-mono);font-size:12px;
  line-height:1.5;outline:none;resize:vertical;min-height:68px;transition:border-color .15s}
.ta:focus{border-color:var(--gold)}
.ta.neg{color:var(--text-dim);min-height:52px}
.dropzone{border:1.5px dashed var(--border);border-radius:8px;padding:18px 12px;
  text-align:center;cursor:pointer;transition:all .15s;color:var(--text-muted);
  font-family:var(--font-mono);font-size:10px;letter-spacing:.5px}
.dropzone:hover{border-color:var(--gold);color:var(--text)}
.dropzone.has{padding:10px;border-style:solid;border-color:var(--border-strong)}
.dropzone img{max-width:100%;max-height:140px;border-radius:5px;display:block;margin:0 auto}
.model-select{width:100%;padding:9px 10px;background:var(--surface-2);
  border:1px solid var(--border);border-radius:6px;color:var(--text);
  font-family:var(--font-body);font-size:13px;font-weight:500;cursor:pointer;
  transition:border-color .12s;-webkit-appearance:none;-moz-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path fill='%236c6c78' d='M6 8L0 0h12z'/></svg>");
  background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
.model-select:hover{border-color:var(--border-strong)}
.model-select:focus{outline:none;border-color:var(--gold)}
.model-select option{background:var(--surface-2);color:var(--text)}
.hintline{font-family:var(--font-mono);font-size:9px;color:var(--text-muted);
  letter-spacing:.4px;margin-top:8px;line-height:1.5}
.hintline.gold{color:var(--gold-dark)}
.slider-row{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.slider-row .sl{font-family:var(--font-mono);font-size:9px;font-weight:600;
  color:var(--text-muted);text-transform:uppercase;letter-spacing:.3px;min-width:66px}
.slider-row input[type=range]{flex:1;height:4px;accent-color:var(--gold);cursor:pointer;
  -webkit-appearance:none;background:var(--surface-3);border-radius:2px}
.slider-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;
  height:13px;border-radius:50%;background:var(--gold);cursor:pointer}
.slider-row .sv{font-family:var(--font-mono);font-size:10px;font-weight:700;color:var(--text);
  min-width:52px;text-align:right;background:transparent;border:1px solid transparent;
  border-radius:3px;padding:3px 5px;outline:none;-moz-appearance:textfield}
.slider-row .sv::-webkit-outer-spin-button,.slider-row .sv::-webkit-inner-spin-button{
  -webkit-appearance:none;margin:0}
.slider-row .sv:hover{border-color:var(--border);background:var(--surface-3)}
.slider-row .sv:focus{border-color:var(--gold);background:var(--surface-3);color:var(--gold)}
.lenbox{background:var(--gold-dim);border:1px solid rgba(232,169,23,.3);border-radius:6px;
  padding:6px 9px;margin-top:2px;font-family:var(--font-mono);font-size:10px;
  color:var(--gold);letter-spacing:.3px;display:flex;align-items:center;gap:6px}
.adv-toggle{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted);cursor:pointer;display:flex;
  align-items:center;gap:6px;padding:2px 0}
.adv-toggle:hover{color:var(--text)}
.adv-toggle .icon{color:var(--gold)}
.adv-body{display:none;margin-top:12px}
.adv-body.open{display:block}
.seed-row{display:flex;gap:8px;align-items:center;margin-top:4px}
.seed-row input[type=number]{flex:1;padding:8px 10px;background:var(--surface-2);
  border:1px solid var(--border);border-radius:4px;color:var(--text);
  font-family:var(--font-mono);font-size:12px;outline:none}
.seed-row input[type=number]:focus{border-color:var(--gold)}
.gen-btn{width:100%;padding:13px;background:var(--gold);color:var(--black);
  font-family:var(--font-mono);font-weight:700;font-size:12px;letter-spacing:.5px;
  text-transform:uppercase;border:none;border-radius:6px;cursor:pointer;transition:all .15s;
  display:flex;align-items:center;justify-content:center;gap:6px}
.gen-btn:hover:not(:disabled){background:var(--gold-light)}
.gen-btn:disabled{opacity:.5;cursor:not-allowed}
.gen-btn-secondary{background:transparent;border:1px solid var(--gold);color:var(--gold);
  padding:9px;font-size:11px}
.gen-btn-secondary:hover:not(:disabled){background:var(--gold-dim);color:var(--gold)}
.sidebar-foot{padding:12px 16px;border-top:1px solid var(--border);background:var(--surface)}
.lora-url{width:100%;padding:8px 10px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:4px;color:var(--text);font-family:var(--font-mono);font-size:11px;outline:none}
.lora-url:focus{border-color:var(--gold)}
.lora-row{display:flex;gap:6px;margin-top:6px}
.lora-row .lora-url{flex:1}
.lora-row .lora-scale{flex:0 0 64px}
.lora-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:9px 10px;margin-top:8px}
.lora-card-top{display:flex;align-items:center;gap:8px}
.lora-card-name{font-family:var(--font-mono);font-size:10px;color:var(--gold);font-weight:700;
  flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lora-x{background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;
  line-height:1;flex:0 0 auto}
.lora-x:hover{color:var(--red)}

/* ── main stage (the video is the hero) ── */
.stage{grid-area:main;position:relative;background:#000;overflow:hidden}
.viewer{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  padding:18px}
.viewer video{max-width:100%;max-height:100%;border-radius:8px;background:#000;
  box-shadow:0 8px 40px rgba(0,0,0,.6)}
.ph{text-align:center;color:var(--text-muted);font-family:var(--font-mono);font-size:12px;
  letter-spacing:.5px;line-height:2;padding:24px}
.ph b{color:var(--gold)}
.ph .big{font-size:32px;display:block;margin-bottom:8px;opacity:.5}
/* progress bar across the bottom of the stage */
.progrow{position:absolute;left:0;right:0;bottom:0;padding:0 0 0 0;z-index:15}
.prog{height:4px;background:rgba(255,255,255,.06);overflow:hidden}
.fill{height:100%;width:0%;background:var(--gold);transition:width .3s}
.stage-status{position:absolute;left:50%;top:14px;transform:translateX(-50%);
  font-family:var(--font-mono);font-size:10px;color:var(--text-dim);letter-spacing:.5px;
  background:rgba(9,9,11,.7);backdrop-filter:blur(6px);padding:5px 12px;border-radius:14px;
  border:1px solid var(--border);white-space:nowrap;display:none}
.stage-status.show{display:block}
/* download chip top-left when a video is shown */
.stage-tools{position:absolute;top:14px;left:14px;z-index:16;display:none;gap:8px}
.stage-tools.show{display:flex}
.chip{background:rgba(9,9,11,.72);backdrop-filter:blur(8px);border:1px solid var(--border);
  color:var(--text);font-family:var(--font-mono);font-size:10px;font-weight:600;
  letter-spacing:.5px;padding:7px 12px;border-radius:8px;cursor:pointer;transition:all .12s;
  text-decoration:none;display:inline-flex;align-items:center;gap:6px}
.chip:hover{border-color:var(--gold);color:var(--gold)}

/* ── free-floating Jobs panel (queue + history) ── */
.float{position:absolute;top:14px;right:14px;width:262px;z-index:20;
  background:rgba(17,17,20,.9);backdrop-filter:blur(14px);
  border:1px solid var(--border);border-radius:11px;
  box-shadow:0 14px 40px rgba(0,0,0,.55);display:flex;flex-direction:column;
  overflow:hidden;max-height:calc(100% - 28px)}
.float-head{display:flex;align-items:center;gap:7px;padding:10px 12px;cursor:grab;
  user-select:none;border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(255,255,255,.03),transparent)}
.float-head:active{cursor:grabbing}
.float-title{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text);display:flex;align-items:center;gap:7px}
.float-title .icon{color:var(--gold);font-size:11px}
.float-grip{color:var(--text-muted);font-size:11px;letter-spacing:-1px}
.float-btns{margin-left:auto;display:flex;gap:5px}
.float-min{width:22px;height:22px;background:var(--surface-3);border:1px solid var(--border);
  color:var(--text-muted);border-radius:5px;cursor:pointer;font-size:13px;line-height:1;
  display:flex;align-items:center;justify-content:center}
.float-min:hover{border-color:var(--gold);color:var(--gold)}
.float-inner{display:flex;flex-direction:column;min-height:0;overflow:hidden}
.float.min .float-inner{display:none}
.float-sec{display:flex;flex-direction:column;min-height:0;border-bottom:1px solid var(--border)}
.float-sec:last-child{border-bottom:none}
.float-sec.q{flex:0 0 auto;max-height:230px}
.float-sec.h{flex:1;min-height:0}
.float-sec-hd{display:flex;align-items:center;gap:6px;padding:9px 12px 7px;cursor:pointer;
  font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted);user-select:none}
.float-sec-hd:hover{color:var(--text)}
.float-sec-hd .icon{color:var(--gold)}
.float-sec-hd .q-count{margin-left:auto;background:var(--surface-3);border:1px solid var(--border);
  border-radius:10px;padding:0 7px;color:var(--text-dim);letter-spacing:0;font-weight:600}
.float-sec-hd .caret{margin-left:6px;font-size:9px;transition:transform .15s}
.float-sec.closed .caret{transform:rotate(-90deg)}
.float-sec-body{overflow-y:auto;min-height:0;padding:0 10px 10px;
  scrollbar-width:thin;scrollbar-color:var(--surface-3) transparent}
.float-sec.closed .float-sec-body{display:none}
.empty{color:var(--text-muted);font-family:var(--font-mono);font-size:9px;
  letter-spacing:.4px;padding:8px 2px;line-height:1.6}
.q-item{display:flex;gap:8px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:6px;padding:7px;margin-bottom:7px;align-items:center}
.q-cancel{flex:0 0 auto;width:22px;height:22px;background:none;border:1px solid var(--border);
  color:var(--text-muted);border-radius:5px;cursor:pointer;font-size:11px;line-height:1;
  display:flex;align-items:center;justify-content:center;transition:all .12s}
.q-cancel:hover{border-color:var(--red);color:var(--red);background:rgba(239,68,68,.08)}
.q-item img{width:40px;height:40px;border-radius:4px;object-fit:cover;flex:0 0 40px;
  background:var(--surface-3)}
.q-imgicon{width:40px;height:40px;border-radius:4px;flex:0 0 40px;background:var(--surface-3);
  display:flex;align-items:center;justify-content:center;font-size:18px}
.q-meta{flex:1;min-width:0;display:flex;flex-direction:column;gap:4px;justify-content:center}
.q-prompt{font-family:var(--font-mono);font-size:9px;color:var(--text);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.q-stat{display:flex;align-items:center;gap:5px;font-family:var(--font-mono);font-size:8px;
  color:var(--text-muted);letter-spacing:.3px}
.q-bar{height:3px;background:var(--surface-3);border-radius:2px;overflow:hidden}
.q-bar i{display:block;height:100%;background:var(--gold);width:0%;transition:width .3s}
.badge{font-family:var(--font-mono);font-size:7px;letter-spacing:1px;text-transform:uppercase;
  font-weight:700;padding:1px 5px;border-radius:3px}
.b-run{background:var(--gold-dim);color:var(--gold)}
.b-queue{background:var(--surface-3);color:var(--text-muted)}
.b-err{background:rgba(239,68,68,.12);color:var(--red)}
.h-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.history-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  overflow:hidden;position:relative;cursor:pointer;transition:all .15s}
.history-card:hover{border-color:var(--gold);transform:translateY(-2px)}
.history-card-thumb{width:100%;aspect-ratio:1/1;background:var(--surface-3);overflow:hidden;
  position:relative}
.history-card-thumb img{width:100%;height:100%;object-fit:cover;display:block}
.history-card-thumb .play{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;font-size:16px;color:#fff;text-shadow:0 1px 4px #000;
  background:rgba(0,0,0,.18)}
.history-card-thumb .h-imgtag{position:absolute;top:4px;left:4px;background:rgba(0,0,0,.65);
  color:var(--gold);font-family:var(--font-mono);font-size:8px;font-weight:700;
  padding:2px 5px;border-radius:3px;letter-spacing:.5px}

/* ── console dock (bottom, starts minimized) ── */
.dock{grid-area:dock;background:#0a0a0c;border-top:1px solid var(--border);
  display:flex;flex-direction:column}
.dock-head{display:flex;align-items:center;gap:8px;padding:9px 14px;cursor:pointer;
  user-select:none}
.dock-title{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  color:var(--text-muted);text-transform:uppercase}
.dock-title .icon{color:var(--gold)}
.dock-sp{flex:1}
.q-clear{background:var(--surface-3);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:10px;font-weight:600;cursor:pointer;
  letter-spacing:.5px;padding:5px 11px;border-radius:5px;line-height:1;transition:all .12s}
.q-clear:hover{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}
.console{height:180px;overflow-y:auto;padding:2px 16px 12px;font-family:var(--font-mono);
  font-size:11px;line-height:1.55;color:var(--text-dim);
  scrollbar-width:thin;scrollbar-color:var(--surface-3) transparent}
.dock.collapsed .console{display:none}
.console .ln{white-space:pre-wrap;word-break:break-word}
.console .diag{color:var(--gold)}
.console .warn{color:var(--gold-light)}

/* ── gallery modal ── */
.modal-scrim{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(3px);
  z-index:8000;display:none;align-items:center;justify-content:center;padding:24px}
.modal-scrim.open{display:flex}
.history-modal-panel{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;width:min(1200px,95vw);height:min(85vh,800px);display:flex;
  flex-direction:column;overflow:hidden}
.history-modal-head{display:flex;align-items:center;padding:14px 18px;
  border-bottom:1px solid var(--border);gap:12px}
.history-modal-title{font-family:var(--font-mono);font-size:13px;font-weight:700;
  color:var(--text);letter-spacing:1px;flex:1;display:flex;align-items:center;gap:8px}
.history-modal-title .icon{color:var(--gold)}
.modal-close{width:28px;height:28px;background:var(--surface-3);border:1px solid var(--border);
  color:var(--text);border-radius:5px;cursor:pointer;font-size:14px;line-height:1}
.modal-close:hover{border-color:var(--gold);color:var(--gold)}
.history-modal-body{flex:1;overflow-y:auto;padding:20px}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:14px;align-content:start}
.ov-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  overflow:hidden;cursor:pointer;transition:all .15s}
.ov-card:hover{border-color:var(--gold);transform:translateY(-2px)}
.ov-card video{width:100%;display:block;background:#000;aspect-ratio:1/1;object-fit:cover}
.ov-card .cap{padding:8px 10px;font-family:var(--font-mono);font-size:9px;
  color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ── Civitai LoRA search modal (ported from MissingLink SDXL studio) ── */
.civ-search-modal{position:fixed;inset:0;z-index:9600;background:rgba(0,0,0,.86);
  display:flex;align-items:center;justify-content:center;padding:2vh 2vw}
.civ-search-box{width:95vw;max-width:1400px;height:min(92vh,900px);max-height:92vh;
  background:var(--surface);border:1px solid var(--border-strong);border-radius:12px;
  display:flex;flex-direction:column;overflow:hidden}
.civ-search-head{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;border-bottom:1px solid var(--border);font-family:var(--font-mono);
  font-size:14px;letter-spacing:.5px;color:var(--text)}
.civ-search-close{background:none;border:none;color:var(--text-muted);font-size:20px;cursor:pointer}
.civ-search-close:hover{color:var(--text)}
.civ-search-controls{flex:0 0 auto;display:flex;flex-wrap:wrap;gap:10px;padding:12px 20px;
  border-bottom:1px solid var(--border);align-items:center}
.civ-search-controls input[type=text]{flex:1;min-width:180px;background:var(--surface-2);
  color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:13px}
.civ-search-controls input[type=text]:focus{outline:none;border-color:var(--gold)}
.civ-search-controls select{background:var(--surface-2);color:var(--text);
  border:1px solid var(--border);border-radius:8px;padding:9px 10px;font-size:12px}
.civ-nsfw-toggle{font-size:11px;color:var(--text-muted);display:flex;align-items:center;gap:5px;
  font-family:var(--font-mono)}
.civ-search-tags{flex:0 0 auto;display:flex;flex-wrap:wrap;gap:6px;padding:10px 20px;
  border-bottom:1px solid var(--border);max-height:88px;overflow:auto;
  scrollbar-width:none;-ms-overflow-style:none}
.civ-search-tags::-webkit-scrollbar{display:none}
.civ-tag-chip{background:var(--surface-2);border:1px solid var(--border);color:var(--text-muted);
  border-radius:20px;padding:4px 12px;font-size:11px;cursor:pointer;font-family:var(--font-mono)}
.civ-tag-chip:hover,.civ-tag-chip.active{border-color:var(--gold);color:var(--text)}
.civ-search-grid{flex:1;overflow:auto;padding:16px 20px;display:grid;gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
  grid-auto-rows:max-content;align-content:start;align-items:start;
  scrollbar-width:none;-ms-overflow-style:none}
.civ-search-grid::-webkit-scrollbar{display:none}
.civ-result{border:1px solid var(--border);border-radius:9px;overflow:hidden;
  background:var(--surface-2);display:flex;flex-direction:column;
  transition:border-color .15s,transform .15s;align-self:start}
.civ-result:hover{border-color:var(--gold);transform:translateY(-2px)}
/* Applied state — grayed + desaturated so you can see at a glance which
   LoRAs are already loaded (matches the SDXL studio's .applied). */
.civ-result.applied{opacity:.55;filter:grayscale(.55)}
.civ-result.applied:hover{border-color:var(--border);transform:none}
.civ-result.applied .add{background:var(--surface-3);color:var(--text-muted);cursor:default}
.civ-result .thumbwrap{width:100%;aspect-ratio:1/1;min-height:0;flex:0 0 auto;
  background:#111;overflow:hidden;position:relative}
.civ-result .thumbwrap img,.civ-result .thumbwrap video{width:100%;height:100%;
  object-fit:cover;display:block}
.civ-result .thumbwrap .vid-badge{position:absolute;top:6px;right:6px;
  background:rgba(0,0,0,.6);color:#fff;font-size:9px;line-height:1;
  padding:3px 5px;border-radius:3px;pointer-events:none}
.civ-result .thumbwrap .info-badge{position:absolute;bottom:6px;left:6px;
  background:rgba(0,0,0,.6);color:#fff;font-size:9px;line-height:1;
  padding:3px 6px;border-radius:3px;cursor:pointer;opacity:0;transition:opacity .12s}
.civ-result:hover .thumbwrap .info-badge{opacity:1}
.civ-result .thumbwrap,.civ-result .nm{cursor:pointer}
/* LoRA detail modal */
.ld-sec{margin-bottom:22px}
.ld-lbl{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;color:var(--text-muted);font-weight:700;margin-bottom:10px}
.ld-lbl .c{text-transform:none;letter-spacing:0;font-weight:400;opacity:.7}
.ld-samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.ld-samples img,.ld-samples video{width:100%;aspect-ratio:1/1;object-fit:cover;
  border-radius:8px;background:#111;display:block}
.ld-trigs{display:flex;flex-wrap:wrap;gap:6px}
.ld-trig{background:var(--gold-dim);color:var(--gold);border:1px solid var(--border);
  border-radius:6px;font-family:var(--font-mono);font-size:11px;padding:5px 10px;cursor:pointer}
.ld-trig:hover{border-color:var(--gold)}
.ld-desc{font-size:13px;line-height:1.6;color:var(--text-dim);max-height:280px;overflow-y:auto;
  border:1px solid var(--border);border-radius:8px;padding:12px 14px;background:var(--surface-2)}
.ld-desc img{max-width:100%;border-radius:6px}
.ld-desc a{color:var(--gold)}
.civ-result .noimg{width:100%;aspect-ratio:1/1;display:flex;align-items:center;
  justify-content:center;color:var(--text-muted);font-size:11px;
  font-family:var(--font-mono);background:#161618}
.civ-result .meta{padding:9px 11px;display:flex;flex-direction:column;gap:4px}
.civ-result .nm{font-size:12px;color:var(--text);font-weight:600;line-height:1.35;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.civ-result .bm{font-size:10px;color:var(--text-muted);font-family:var(--font-mono)}
.civ-result .add{margin:0 11px 11px;background:var(--gold);color:var(--black);border:none;
  border-radius:7px;padding:8px;font-size:11px;font-weight:700;cursor:pointer;
  font-family:var(--font-mono)}
.civ-search-status{flex:0 0 auto;padding:8px 16px;font-size:11px;color:var(--text-muted);
  font-family:var(--font-mono);text-align:center}

/* ── dialogue builder (LTX) ── */
.dlg-row{display:flex;gap:6px;margin-bottom:6px;align-items:center}
.dlg-spk{flex:0 0 104px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:4px;color:var(--gold);font-family:var(--font-mono);font-size:10px;
  font-weight:700;padding:7px 5px;cursor:pointer;min-width:0}
.dlg-spk:focus{outline:none;border-color:var(--gold)}
.dlg-row .lora-url{flex:1;min-width:0}
.dlg-clip{flex:0 0 62px;background:var(--surface-3);border:1px solid var(--border);
  border-radius:4px;color:var(--text-dim);font-family:var(--font-mono);font-size:9px;
  font-weight:700;padding:7px 3px;cursor:pointer;min-width:0}
.dlg-clip:focus{outline:none;border-color:var(--gold)}
/* clip overview cards (the compact list that replaces the cramped rows) */
/* Visual storyboard clip cards: thumbnail of the starting frame + scene */
.storyclip{margin-bottom:8px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:10px;cursor:pointer;transition:border-color .12s;overflow:hidden}
.storyclip:hover{border-color:var(--gold)}
.storyclip .sc-head{display:flex;align-items:center;gap:10px;padding:9px 10px;
  border-bottom:1px solid var(--border)}
.storyclip .sc-thumb{flex:0 0 46px;width:46px;height:46px;position:relative;
  background:#0d0d0f;border-radius:7px;overflow:hidden;display:flex;
  align-items:center;justify-content:center}
.storyclip .sc-thumb img{width:100%;height:100%;object-fit:cover;display:block}
.storyclip .sc-chain{color:var(--text-muted);font-size:16px;text-align:center;
  font-family:var(--font-mono);line-height:1}
.storyclip .sc-chain span{font-size:7px;letter-spacing:.3px;display:block}
.storyclip .sc-num{position:absolute;top:2px;left:2px;background:var(--gold);
  color:#0d0d0f;font-family:var(--font-mono);font-size:8px;font-weight:800;
  padding:1px 4px;border-radius:3px}
.storyclip .sc-headmeta{flex:1;min-width:0;display:flex;align-items:center;gap:8px;
  flex-wrap:wrap;font-family:var(--font-mono);font-size:10px}
.storyclip .sc-time{color:var(--gold);font-weight:700}
.storyclip .sc-lc{color:var(--text-muted)}
.storyclip .sc-tag{background:var(--gold-dim);color:var(--gold);font-size:8px;
  font-weight:700;padding:1px 6px;border-radius:3px}
.storyclip .sc-edit{flex:0 0 auto;background:var(--surface-3);border:1px solid var(--border);
  border-radius:5px;color:var(--gold);font-family:var(--font-mono);font-size:10px;
  padding:5px 10px;cursor:pointer}
.storyclip .sc-edit:hover{border-color:var(--gold)}
.storyclip .sc-del{flex:0 0 auto;background:var(--surface-3);border:1px solid var(--border);
  border-radius:5px;color:var(--text-muted);font-size:12px;line-height:1;
  padding:5px 8px;cursor:pointer;margin-left:6px}
.storyclip .sc-del:hover{border-color:#E5484D;color:#E5484D}
.storyclip .sc-scene{padding:9px 12px 4px;font-size:12px;color:var(--text);
  line-height:1.5}
.storyclip .sc-lines{padding:2px 12px 10px;display:flex;flex-direction:column;gap:4px}
.storyclip .sc-line{display:flex;gap:8px;align-items:baseline;font-size:11.5px;line-height:1.45}
.storyclip .sc-spk{flex:0 0 auto;color:var(--gold);font-family:var(--font-mono);
  font-size:9px;font-weight:700;text-transform:uppercase;padding-top:1px}
.storyclip .sc-say{flex:1;min-width:0;color:var(--text-dim)}
.storyclip .sc-noline{padding:0 12px 10px;font-family:var(--font-mono);font-size:9px;
  color:var(--text-muted)}
.storyclip .sc-cont{padding:6px 12px;font-family:var(--font-mono);font-size:9px;
  color:var(--gold);background:var(--gold-dim);border-bottom:1px solid var(--border);
  letter-spacing:.3px}
.storyclip .sc-next{padding:10px 12px;border-top:1px solid var(--border)}
.storyclip .sc-next-lbl{font-family:var(--font-mono);font-size:9px;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px}
.storyclip .sc-next-btns{display:flex;gap:8px}
.storyclip .sc-next-btn{flex:1;background:var(--surface-3);border:1px solid var(--border);
  border-radius:7px;color:var(--text);font-family:var(--font-mono);font-size:10px;
  font-weight:600;padding:9px 8px;cursor:pointer;transition:border-color .12s}
.storyclip .sc-next-btn:hover{border-color:var(--gold);color:var(--gold)}
/* add-clip chooser options */
.addclip-opt{display:flex;gap:12px;align-items:flex-start;width:100%;text-align:left;
  padding:14px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:10px;cursor:pointer;transition:border-color .12s;color:inherit}
.addclip-opt:hover{border-color:var(--gold)}
.addclip-opt .ac-icon{flex:0 0 auto;font-size:22px;line-height:1.2}
.addclip-opt .ac-title{font-size:13px;color:var(--text);font-weight:600;margin-bottom:3px}
.addclip-opt .ac-sub{font-size:11px;color:var(--text-muted);line-height:1.5}
.addclip-opt .ac-badge{background:var(--gold-dim);color:var(--gold);font-size:8px;
  font-family:var(--font-mono);font-weight:700;padding:1px 6px;border-radius:3px;
  vertical-align:middle;margin-left:4px}
.clip-empty{padding:14px;font-family:var(--font-mono);font-size:10px;
  color:var(--text-muted);text-align:center}
/* timeline: persistent add-clip card at the end of the strip */
.sc-addcard{margin-bottom:8px;background:var(--surface-2);border:1px dashed var(--border);
  border-radius:10px;padding:14px;display:flex;gap:10px;align-items:center;
  justify-content:center;cursor:default}
.sc-addcard .sc-addbtn{flex:1;background:var(--surface-3);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:11px;
  font-weight:600;padding:11px 8px;cursor:pointer;transition:border-color .12s}
.sc-addcard .sc-addbtn:hover{border-color:var(--gold);color:var(--gold)}
.sc-addcard .sc-addbtn .a-sub{display:block;font-size:8px;color:var(--text-muted);
  font-weight:400;margin-top:3px;letter-spacing:.2px}
/* editor modal */
.clip-modal-box{background:var(--surface);border:1px solid var(--border);
  border-radius:14px;width:min(680px,96vw);max-height:92vh;display:flex;
  flex-direction:column;overflow:hidden}
.clip-modal-body{padding:18px 20px;overflow-y:auto;display:flex;flex-direction:column;gap:20px;flex:1;min-height:0}
.clip-modal-lbl{display:flex;align-items:center;gap:8px;font-family:var(--font-mono);
  font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);
  font-weight:700;margin-bottom:8px}
.clip-modal-lbl .c{text-transform:none;letter-spacing:0;color:var(--text-muted);
  font-weight:400;opacity:.8}
.clip-modal-ta{width:100%;box-sizing:border-box;min-height:96px;resize:vertical;
  background:var(--surface-2);border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-family:inherit;font-size:13px;line-height:1.6;padding:12px 14px}
.clip-modal-ta:focus{outline:none;border-color:var(--gold)}
/* Next-scene compose: reference images + upload + token badge */
.nc-imgs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.nc-img{position:relative;width:84px;height:84px;border-radius:8px;overflow:hidden;
  border:1px solid var(--border);background:#111}
.nc-img img{width:100%;height:100%;object-fit:cover;display:block}
.nc-img .nc-tag{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.65);
  color:#fff;font-size:8px;text-align:center;padding:2px 0}
.nc-img .nc-x{position:absolute;top:2px;right:2px;width:18px;height:18px;border:none;
  border-radius:4px;background:rgba(0,0,0,.6);color:#fff;font-size:10px;cursor:pointer;
  display:flex;align-items:center;justify-content:center}
.nc-img .nc-x:hover{background:#E5484D}
.nc-upload{display:inline-block;background:var(--surface-3);border:1px dashed var(--border);
  border-radius:8px;color:var(--gold);font-family:var(--font-mono);font-size:11px;
  padding:8px 14px;cursor:pointer}
.nc-upload:hover{border-color:var(--gold)}
.tok-badge{background:var(--gold-dim);color:var(--gold);border:1px solid var(--border);
  border-radius:6px;font-family:var(--font-mono);font-size:11px;padding:5px 10px;
  cursor:pointer;margin-right:8px}
.tok-badge:hover{border-color:var(--gold)}
.tok-badge.low{color:#E5484D;border-color:#E5484D}
.clip-reset{margin-left:auto;background:var(--surface-3);border:1px solid var(--border);
  border-radius:4px;color:var(--gold);font-family:var(--font-mono);font-size:9px;
  padding:4px 8px;cursor:pointer}
.clip-reset:hover{border-color:var(--gold)}
.clip-modal-metarow{display:flex;align-items:center;gap:14px;margin-top:10px}
.clip-frames-row{display:flex;gap:12px;margin-top:12px}
.v2v-sec{margin-top:6px;padding-top:10px;border-top:1px solid var(--border)}
.v2v-drop{width:100%;aspect-ratio:16/9;max-height:160px;border:1px dashed var(--border);
  border-radius:8px;display:flex;align-items:center;justify-content:center;
  overflow:hidden;background:var(--surface-3);cursor:default}
.v2v-drop .cf-empty{color:var(--text-muted);font-family:var(--font-mono);font-size:11px}
.clip-select{width:100%;background:var(--surface-3);border:1px solid var(--border);
  border-radius:7px;color:var(--text);font-family:var(--font-mono);font-size:11px;padding:8px}
.edit-lora-row{display:flex;align-items:center;gap:10px;margin:5px 0}
.edit-lora-row>span:first-child{font-size:11px;color:var(--text-muted);min-width:120px}
.edit-lora-row input[type=range]{flex:1}
.edit-lora-row .sv{font-family:var(--font-mono);font-size:11px;min-width:34px;text-align:right}
.editimg-src{width:100%;max-height:220px;border-radius:8px;overflow:hidden;
  background:#000;display:flex;align-items:center;justify-content:center;cursor:zoom-in;
  border:1px solid var(--border)}
.editimg-src img{max-width:100%;max-height:220px;object-fit:contain;display:block}
#imgFullView{position:fixed;inset:0;z-index:12000;background:rgba(0,0,0,.92);
  display:flex;align-items:center;justify-content:center;cursor:zoom-out}
#imgFullView img{max-width:94vw;max-height:90vh;object-fit:contain;border-radius:6px;cursor:default}
.ifv-close{position:fixed;top:18px;right:22px;background:rgba(0,0,0,.6);border:1px solid var(--border);
  border-radius:8px;color:#fff;font-size:18px;width:40px;height:40px;cursor:pointer;z-index:12001}
.ifv-close:hover{border-color:var(--gold);color:var(--gold)}
.ifv-dl{position:fixed;top:18px;left:22px;background:var(--gold);color:#0d0d0f;font-weight:700;
  border-radius:8px;padding:9px 16px;font-family:var(--font-mono);font-size:12px;
  text-decoration:none;cursor:pointer;z-index:12001}
.ifv-dl:hover{filter:brightness(1.08)}
.clip-frame-slot{flex:1;min-width:0}
.clip-frame-drop{width:100%;aspect-ratio:16/9;border:1px dashed var(--border);border-radius:8px;
  background:var(--surface-2);display:flex;align-items:center;justify-content:center;
  overflow:hidden;cursor:pointer}
.clip-frame-drop:hover{border-color:var(--gold)}
.clip-frame-drop img{width:100%;height:100%;object-fit:cover;display:block}
.clip-frame-drop .cf-empty{color:var(--text-muted);font-family:var(--font-mono);font-size:11px}
/* Clip 1 frames — stacked rows (thumbnail + label + inline actions) */
#clip1Frames{margin-top:10px;display:flex;flex-direction:column;gap:12px}
.c1frame{display:flex;flex-direction:column;gap:8px;background:var(--surface-2);
  border:1px solid var(--border);border-radius:10px;padding:10px}
.c1thumbwrap{position:relative;width:100%}
.c1thumb{width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;
  background:var(--surface-3);border:1px dashed var(--border);cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:border-color .12s}
.c1thumb:hover{border-color:var(--gold)}
.c1thumb img,.c1thumb video{width:100%;height:100%;object-fit:cover;display:block}
.c1thumb .c1plus{color:var(--text-muted);font-size:34px;font-weight:300}
.c1trash{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.65);
  border:1px solid var(--border);border-radius:7px;color:#e8a917;cursor:pointer;
  font-size:15px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;
  transition:all .12s}
.c1trash:hover{background:rgba(0,0,0,.85);border-color:var(--red);color:var(--red)}
.c1editic{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.65);
  border:1px solid var(--border);border-radius:8px;color:var(--text);
  width:28px;height:28px;display:flex;align-items:center;justify-content:center;
  cursor:pointer;font-size:13px;z-index:3}
.c1editic:hover{background:rgba(0,0,0,.85);border-color:var(--gold);color:var(--gold)}
.c1meta{min-width:0}
.c1title{font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--text);margin-bottom:8px}
.c1sub-inline{font-weight:400;font-size:10px;color:var(--text-muted);margin-left:6px}
.c1opt{font-weight:400;font-size:9px;color:var(--text-muted);border:1px solid var(--border);
  border-radius:3px;padding:1px 5px;margin-left:4px;vertical-align:middle}
.c1acts{display:flex;gap:6px;flex-wrap:wrap}
.c1acts .clip-reset{margin:0}
.clip-modal-lenhint{font-family:var(--font-mono);font-size:10px;color:var(--text-muted)}
/* full-size dialogue rows inside the modal */
.cd-row{display:flex;gap:8px;align-items:flex-start;margin-bottom:8px}
.cd-row .dlg-spk{flex:0 0 130px}
.cd-row textarea{flex:1;min-width:0;box-sizing:border-box;min-height:44px;resize:vertical;
  background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  color:var(--text);font-family:inherit;font-size:13px;line-height:1.5;padding:9px 11px}
.cd-row textarea:focus{outline:none;border-color:var(--gold)}
.cd-row .lora-x{flex:0 0 auto;margin-top:4px}
.cd-row .cd-ai{flex:0 0 auto;margin-top:2px;background:var(--surface-3);
  border:1px solid var(--border);border-radius:5px;color:var(--gold);
  font-size:12px;padding:5px 8px;cursor:pointer}
.cd-row .cd-ai:hover{border-color:var(--gold)}
.cd-row .cd-ai:disabled{opacity:.5;cursor:default}
.clip-modal-foot{display:flex;align-items:center;padding:14px 20px;
  border-top:1px solid var(--border);background:var(--surface-2)}
.next-verdict{font-size:12px;line-height:1.6;padding:10px 12px;border-radius:8px;
  border:1px solid var(--border)}
.next-verdict.ok{color:#22C55E;border-color:#22C55E44;background:#22C55E11}
.next-verdict.warn{color:#E8A917;border-color:#E8A91744;background:#E8A91711}
.clip-edit{flex:0 0 auto;background:var(--surface-3);border:1px solid var(--border);
  border-radius:4px;color:var(--gold);font-family:var(--font-mono);font-size:9px;
  padding:4px 9px;cursor:pointer}
.clip-edit:hover{border-color:var(--gold)}
.clip-secs{flex:0 0 54px;background:var(--surface);border:1px solid var(--border);
  border-radius:4px;color:var(--text);font-family:var(--font-mono);font-size:11px;
  padding:5px 6px;text-align:right}
.clip-secs:focus{outline:none;border-color:var(--gold)}
.clip-secs::-webkit-outer-spin-button,.clip-secs::-webkit-inner-spin-button{
  -webkit-appearance:none;appearance:none;margin:0}
.clip-secs{-moz-appearance:textfield;appearance:textfield}
input[type=number]::-webkit-outer-spin-button,input[type=number]::-webkit-inner-spin-button{
  -webkit-appearance:none;appearance:none;margin:0}
input[type=number]{-moz-appearance:textfield;appearance:textfield}
/* Login gate: no visible scrollbar chrome (content still scrolls if needed) */
#mlGate .civ-search-box{overflow:hidden}
#mlGate .civ-search-box>div{scrollbar-width:none;-ms-overflow-style:none}
#mlGate .civ-search-box>div::-webkit-scrollbar{width:0;height:0;display:none}
.clip-unit{color:var(--text-muted);font-family:var(--font-mono);font-size:10px;
  margin-left:-4px}
.spk-gear{margin-left:8px;width:22px;height:22px;background:var(--surface-3);
  border:1px solid var(--border);color:var(--text-muted);border-radius:5px;
  cursor:pointer;font-size:12px;line-height:1;display:inline-flex;
  align-items:center;justify-content:center;flex:0 0 auto}
.spk-gear:hover,.spk-gear.open{border-color:var(--gold);color:var(--gold)}
.spk-panel{background:var(--surface-2);border:1px solid var(--border);
  border-radius:6px;padding:10px;margin-bottom:10px}
.spk-row{border-bottom:1px dashed var(--border);padding-bottom:9px;margin-bottom:9px}
.spk-row:last-of-type{border-bottom:none;margin-bottom:4px;padding-bottom:0}
.spk-row-top{display:flex;gap:6px;align-items:center;margin-bottom:5px}
.spk-name{flex:1;background:var(--surface-3);border:1px solid var(--border);
  border-radius:4px;color:var(--gold);font-family:var(--font-mono);font-size:11px;
  font-weight:700;padding:6px 8px;text-transform:uppercase;min-width:0}
.spk-name:focus{outline:none;border-color:var(--gold)}
.spk-voice{width:100%;background:var(--surface-3);border:1px solid var(--border);
  border-radius:4px;color:var(--text-dim);font-family:var(--font-mono);font-size:10px;
  line-height:1.45;padding:6px 8px;resize:vertical;min-height:40px}
.spk-voice:focus{outline:none;border-color:var(--gold);color:var(--text)}

.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--surface);
  border:1px solid var(--gold);padding:11px 16px;border-radius:6px;font-family:var(--font-mono);
  font-size:11px;color:var(--text);z-index:9500;display:none;max-width:80vw}
.toast.err{border-color:var(--red);color:var(--red)}
/* ── Timeline editor ── */
#editorTab{position:fixed;inset:56px 0 0 0;z-index:8000;background:#09090B;
  display:flex;flex-direction:column}
#edPreviewWrap{flex:1 1 auto;min-height:0;background:#000;position:relative;
  display:flex;align-items:center;justify-content:center}
#edPreview{max-width:100%;max-height:100%;display:none}
#edPreviewEmpty{color:var(--text-muted);font-family:var(--font-mono);font-size:12px;
  padding:20px;text-align:center;line-height:2.4;font-size:28px}
#edTransport{flex:0 0 auto;display:flex;align-items:center;gap:10px;padding:8px 14px;
  background:var(--surface);border-bottom:1px solid var(--border);border-top:1px solid var(--border)}
.ed-tbtn{background:var(--surface-3);border:1px solid var(--border);border-radius:6px;
  color:var(--text);font-family:var(--font-mono);font-size:11px;padding:6px 11px;cursor:pointer}
.ed-tbtn:hover{border-color:var(--gold)}
.ed-time{font-family:var(--font-mono);font-size:11px;color:var(--text-dim)}
#edPreviewEmpty{color:var(--text-muted);font-family:var(--font-mono);
  padding:20px;text-align:center;line-height:1.8;font-size:13px}
.ed-tbtn .ed-plus{font-size:9px;vertical-align:super;margin-left:1px}
.ed-tbtn.ed-export{background:var(--gold);color:#0d0d0f;border-color:var(--gold);
  font-weight:800;font-size:14px;padding:6px 14px}
.ed-tbtn.ed-export:hover{filter:brightness(1.08)}
.ed-zoom{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-muted)}
.ed-zoom input{width:110px}
#edTimelineWrap{flex:1;overflow:auto;position:relative;background:var(--surface-2);min-height:0}
#edRuler{height:22px;position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--border);
  z-index:3;font-family:var(--font-mono);font-size:8px;color:var(--text-muted);white-space:nowrap}
#edRuler .ed-tick{position:absolute;top:0;height:22px;border-left:1px solid var(--border);padding-left:3px;padding-top:5px}
#edTracks{position:relative}
.ed-track{position:relative;height:64px;border-bottom:1px solid var(--border);white-space:nowrap}
.ed-track.ed-audio{height:44px;background:rgba(232,169,23,.03)}
.ed-track-label{position:sticky;left:0;z-index:2;display:inline-flex;align-items:center;gap:6px;
  height:100%;padding:0 8px;background:var(--surface);border-right:1px solid var(--border);
  font-family:var(--font-mono);font-size:9px;color:var(--text-muted);vertical-align:top}
.ed-track-label .ed-mute{background:none;border:1px solid var(--border);border-radius:4px;
  color:var(--text-muted);font-size:10px;padding:2px 5px;cursor:pointer}
.ed-track-label .ed-mute.on{color:var(--gold);border-color:var(--gold)}
.ed-track-label .ed-rmtrack{background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:12px}
.ed-track-label .ed-rmtrack:hover{color:var(--red)}
.ed-clip{position:absolute;top:5px;height:54px;background:var(--surface-3);
  border:1px solid var(--border);border-radius:6px;overflow:hidden;cursor:grab;
  display:flex;align-items:center;box-sizing:border-box}
.ed-audio .ed-clip{top:5px;height:34px;background:var(--gold-dim)}
.ed-clip.sel{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold)}
.ed-clip img{height:100%;width:auto;object-fit:cover;pointer-events:none}
.ed-clip .ed-clip-name{position:absolute;left:4px;bottom:2px;font-family:var(--font-mono);
  font-size:8px;color:#fff;text-shadow:0 1px 3px #000;pointer-events:none;
  max-width:90%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ed-clip .ed-trim{position:absolute;top:0;height:100%;width:8px;cursor:ew-resize;background:rgba(232,169,23,.35)}
.ed-clip .ed-trim.l{left:0}.ed-clip .ed-trim.r{right:0}
.ed-clip .ed-fade{position:absolute;top:0;height:100%;width:0;pointer-events:none;
  background:linear-gradient(90deg,#000,transparent)}
.ed-clip .ed-xfade{position:absolute;right:0;top:0;height:100%;width:0;pointer-events:none;
  background:repeating-linear-gradient(45deg,rgba(232,169,23,.25) 0 3px,transparent 3px 6px)}
#edPlayhead{position:absolute;top:0;width:2px;background:var(--gold);pointer-events:none;z-index:4;display:none}
#edPool{flex:0 0 auto;max-height:100px;overflow-y:auto;padding:8px 14px;
  background:var(--surface);border-top:1px solid var(--border)}
.ed-pool-label{font-family:var(--font-mono);font-size:9px;color:var(--text-muted);margin-bottom:8px}
#edPoolItems{display:flex;gap:8px;flex-wrap:wrap}
.ed-pool-item{width:120px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:7px;overflow:hidden;cursor:pointer;position:relative}
.ed-pool-item:hover{border-color:var(--gold)}
.ed-pool-item .ed-pi-thumb{height:56px;background:#000;display:flex;align-items:center;justify-content:center}
.ed-pool-item .ed-pi-thumb img{width:100%;height:100%;object-fit:cover}
.ed-pool-item .ed-pi-name{padding:4px 6px;font-family:var(--font-mono);font-size:8px;
  color:var(--text-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ed-pool-item .ed-pi-add{position:absolute;top:3px;right:3px;background:var(--gold);color:#0d0d0f;
  border:none;border-radius:4px;font-weight:800;font-size:12px;width:20px;height:20px;cursor:pointer}
.ed-pool-item .ed-pi-badge{position:absolute;top:3px;left:3px;background:rgba(0,0,0,.7);color:var(--gold);
  font-family:var(--font-mono);font-size:7px;padding:1px 4px;border-radius:3px}
/* Floating agent */
#agentFab{position:fixed;right:20px;bottom:52px;z-index:9550;background:var(--gold);
  color:#1a1400;border:none;border-radius:999px;font-family:var(--font-mono);font-weight:700;
  font-size:11px;padding:8px 13px;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,.45);
  opacity:.92;transition:opacity .12s,transform .12s}
#agentFab:hover{opacity:1;transform:translateY(-1px)}
#agentPanel{position:fixed;z-index:9560;width:min(400px,94vw);max-height:72vh;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;
  box-shadow:0 14px 44px rgba(0,0,0,.6);display:flex;flex-direction:column;overflow:hidden}
#agentHead{display:flex;align-items:center;gap:8px;padding:10px 12px;cursor:move;
  background:var(--surface-2);border-bottom:1px solid var(--border);user-select:none}
#agentHead .ag-grip{color:var(--text-muted);letter-spacing:-2px;font-size:11px}
#agentHead .ag-title{font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--gold)}
.ag-btn{background:var(--surface-3);border:1px solid var(--border);border-radius:6px;
  color:var(--text);font-family:var(--font-mono);font-size:10px;padding:5px 9px;cursor:pointer}
.ag-btn:hover{border-color:var(--gold)}
#agentBody{display:flex;flex-direction:column;min-height:0;flex:1}
#agentLog{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;min-height:120px;max-height:46vh}
.agent-hint{color:var(--text-muted);font-size:12px;line-height:1.55}
.agent-msg{padding:8px 11px;border-radius:9px;font-size:12.5px;line-height:1.5;white-space:pre-wrap;max-width:92%}
.am-user{align-self:flex-end;background:var(--gold-dim);color:var(--text);border:1px solid var(--border)}
.am-bot{align-self:flex-start;background:var(--surface-2);color:var(--text-dim);border:1px solid var(--border)}
#agentScriptRow{display:flex;gap:8px;padding:0 12px 8px}
.ag-sbtn{background:var(--surface-3);border:1px solid var(--border);border-radius:7px;color:var(--gold);
  font-family:var(--font-mono);font-size:11px;padding:7px 12px;cursor:pointer;display:inline-block}
.ag-sbtn:hover{border-color:var(--gold)}
#agentInputRow{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border);align-items:flex-end}
#agentInput{flex:1;min-height:44px;max-height:120px;resize:vertical;box-sizing:border-box;
  background:var(--surface-2);border:1px solid var(--border);border-radius:8px;color:var(--text);
  font-family:inherit;font-size:13px;line-height:1.5;padding:10px 12px}
#agentInput:focus{outline:none;border-color:var(--gold)}
#agentSend{flex:0 0 auto;background:var(--gold);color:#1a1400;border:none;border-radius:8px;
  font-size:16px;width:44px;height:44px;cursor:pointer}
#agentSend:hover{filter:brightness(1.08)}
#agentSend:disabled{opacity:.5;cursor:default}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:5px}
::-webkit-scrollbar-track{background:transparent}

/* Docked timeline bar: overlays the bottom of the stage like player
   controls. Translucent, compact, horizontally scrollable. */
.stage-timeline{position:absolute;left:12px;right:12px;bottom:10px;z-index:14;
  background:rgba(12,12,14,.82);backdrop-filter:blur(8px);
  border:1px solid var(--border);border-radius:12px;padding:8px 10px;
  display:flex;flex-direction:column;gap:6px;max-height:200px}
.stl-head{display:flex;align-items:center;gap:8px;flex-wrap:nowrap}
.stl-title{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;color:var(--text-muted);display:flex;align-items:center;gap:6px;white-space:nowrap}
.stl-title .c{display:none}
.stl-spk{width:auto!important;padding:4px 10px!important;font-size:10px!important;margin:0!important}
.stl-title{flex:1 1 auto}
.stl-head>.q-clear:first-of-type{margin-left:auto}
.stl-head .q-clear{font-size:10px;padding:3px 8px}
.stl-strip{display:flex;gap:8px;overflow-x:auto;overflow-y:hidden;padding:2px;
  scroll-behavior:smooth;align-items:stretch}
.stl-strip .clip-empty{flex:0 0 auto;display:flex;align-items:center;
  padding:8px 12px;font-size:11px;color:var(--text-muted);white-space:nowrap}
/* Filmstrip tiles: image-first, 16:9, icon badges. */
.flc{position:relative;flex:0 0 auto;width:148px;height:92px;border-radius:9px;
  overflow:hidden;background:#141416;border:1px solid var(--border);cursor:pointer;
  transition:border-color .12s,transform .12s}
.flc:hover{border-color:var(--gold);transform:translateY(-1px)}
.flc-img{width:100%;height:100%;object-fit:cover;display:block}
.flc-ph{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
  font-size:26px;opacity:.55}
.flc-num{position:absolute;top:5px;left:5px;background:var(--gold);color:#1a1400;
  font-family:var(--font-mono);font-weight:700;font-size:10px;border-radius:5px;
  padding:1px 6px}
.flc-x{position:absolute;top:4px;right:4px;width:18px;height:18px;border:none;
  border-radius:5px;background:rgba(0,0,0,.6);color:var(--text-muted);cursor:pointer;
  font-size:12px;line-height:1;display:none}
.flc:hover .flc-x{display:block}
.flc-x:hover{color:var(--red)}
.flc-bar{position:absolute;left:0;right:0;bottom:0;display:flex;align-items:center;
  justify-content:space-between;padding:3px 6px;background:rgba(0,0,0,.62);
  backdrop-filter:blur(3px)}
.flc-t{font-family:var(--font-mono);font-size:9px;color:var(--text)}
.flc-ic{display:flex;gap:4px;font-size:10px}
.flc-add{border-style:dashed;opacity:.75}
.flc-add:hover{opacity:1}
.flc-ghost{filter:grayscale(.7) brightness(.75);opacity:.6}
.flc-mini{flex:0 0 auto;display:flex;flex-direction:column;gap:6px;
  justify-content:center;width:38px}
.flc-mbtn{width:38px;height:42px;border:1px dashed var(--border);border-radius:8px;
  background:transparent;color:var(--text-muted);font-size:15px;cursor:pointer;
  transition:border-color .12s,color .12s}
.flc-mbtn:hover{border-color:var(--gold);color:var(--gold)}
.stl-hint{display:none}
.stl-strip::-webkit-scrollbar{height:6px}
.stl-strip::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style></head><body>
<div class="app">

  <header class="app-header">
    <!-- LOGO: compact inline SVG placeholder in the studio's gold/black
         theme. To restore your original bitmap logo, replace the whole
         src="..." value below with your data:image/png;base64,... URI. -->
    <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='30' fill='%2318181B' stroke='%23E8A917' stroke-width='4'/><path d='M17 42 V23 l15 14 15-14 v19' fill='none' stroke='%23E8A917' stroke-width='5' stroke-linecap='round' stroke-linejoin='round'/></svg>" alt="MissingLink">
    <h1>MISSINGLINK <span>VIDEO STUDIO</span></h1>
    <div class="stage-tabs">
      <button class="stage-tab" onclick="openGallery()" title="Session gallery">
        <span class="stage-tab-icon">&#127760;</span> Gallery</button>
      <button class="stage-tab active" id="tabStudio" onclick="closeEditor()" title="Workspace">
        <span class="stage-tab-icon">&#127909;</span> Studio</button>
      <button class="stage-tab" id="tabEditor" onclick="openEditor()" title="Timeline editor">
        <span class="stage-tab-icon">&#127916;</span> Editor</button>
    </div>
    <div class="hdr-right">
      <div class="hdr-badge" id="mlBadge" style="cursor:pointer" onclick="showGateIfLocked()" title="MissingLink membership"><span style="color:var(--gold)">&#9670;</span><span id="mlBadgeTxt">Sign in</span></div>
      <div class="hdr-badge" title="VRAM"><span>&#9635;</span><span id="vramPill">&ndash; / &ndash; GB</span></div>
      <div class="hdr-badge" id="connBadge" title="GPU state"><div class="dot cold" id="connDot"></div><span id="connLabel">Connecting</span></div>
      <div class="user-menu-trigger" title="Active GPU">
        <span class="user-menu-fallback">&#9638;</span>
        <span class="user-menu-trigger-email" id="gpuName">GPU</span>
      </div>
    </div>
  </header>

  <aside class="sidebar">
    <div class="sidebar-scroll">
      <div class="mode-sec">
        <div class="mode-cap">Base model</div>
        <div class="mode-toggle">
          <button class="mode-btn" id="engWan" onclick="setEngine('wan')">Wan 2.1</button>
          <button class="mode-btn" id="engWan22" onclick="setEngine('wan22')">Wan 2.2</button>
          <button class="mode-btn active" id="engLtx" onclick="setEngine('ltx')">LTX-2.3 Fast</button>
        </div>
        <div id="wan22Opts" style="display:none;margin-top:8px">
          <label class="civ-nsfw-toggle"><input type="checkbox" id="wan22Lightning" checked onchange="updateDur()"> Lightning (4-step fast)</label>
          <div class="mode-hint">Uncheck for slower, higher-fidelity base sampling. Search LoRAs with base "Wan Video 2.2 I2V-A14B".</div>
        </div>
      </div>

      <div class="mode-sec" id="modeSec">
        <div class="mode-cap">Generation mode</div>
        <div class="mode-toggle">
          <button class="mode-btn active" id="modeI2V" onclick="setMode('i2v')">Image&rarr;Vid</button>
          <button class="mode-btn" id="modeFLF" onclick="setMode('flf2v')">First-Last</button>
          <button class="mode-btn" id="modeVACE" onclick="setMode('vace')">Reference</button>
        </div>
        <div class="mode-hint" id="modeHint">Animate a single still image into a clip. Supports long video via clip chaining.</div>
      </div>

      <div class="mode-sec" id="speedSec">
        <div class="mode-cap">Speed mode</div>
        <div class="mode-toggle">
          <button class="mode-btn active" id="segQual" onclick="qualityMode()">&#9733; Quality</button>
          <button class="mode-btn" id="segFast" onclick="lightningMode()">&#9889; Lightning</button>
        </div>
        <div class="mode-hint">Lightning = the 4-step distill LoRA (fast, seconds per clip). Best choice for long videos. Quality = full multi-step run.</div>
      </div>

      <div class="sec" id="firstSec">
        <div class="sec-label" id="firstLabel"><span class="icon">&#9635;</span> Starting frame <span class="c">required</span></div>
        <div class="dropzone" id="drop" onclick="document.getElementById('file').click()">&#128247;&nbsp; Click or drop an image</div>
        <input type="file" id="file" accept="image/*" hidden>
        <!-- LTX: Clip 1's first + last frame, each upload or generate -->
        <div id="clip1Frames" style="display:none">
          <div class="c1frame">
            <div class="c1thumbwrap">
              <div id="c1StartSlot" class="c1thumb" onclick="$('c1StartFile').click()"><span class="c1plus">+</span></div>
              <button class="c1editic" id="c1StartEditIc" style="display:none" onclick="event.stopPropagation();_c1Edit('start')" title="Edit with AI">&#9998;</button>
              <button class="c1trash" id="c1StartClear" style="display:none" onclick="event.stopPropagation();_c1Clear('start')" title="Remove">&#128465;</button>
            </div>
            <div class="c1meta">
              <div class="c1title">First frame <span class="c1sub-inline">the movie opens on this</span></div>
              <div class="c1acts">
                <button class="clip-reset" onclick="_c1Gen('start')">&#10024; generate</button>
                <button class="clip-reset" onclick="$('c1StartFile').click()">&#8679; upload</button>
                <button class="clip-reset" id="c1StartEdit" style="display:none" onclick="_c1Edit('start')">&#9998; edit</button>
              </div>
            </div>
            <input type="file" id="c1StartFile" accept="image/*" style="display:none" onchange="_c1SetImg('start',this.files)">
          </div>
          <div class="c1frame">
            <div class="c1thumbwrap">
              <div id="c1EndSlot" class="c1thumb" onclick="$('c1EndFile').click()"><span class="c1plus">+</span></div>
              <button class="c1editic" id="c1EndEditIc" style="display:none" onclick="event.stopPropagation();_c1Edit('end')" title="Edit with AI">&#9998;</button>
              <button class="c1trash" id="c1EndClear" style="display:none" onclick="event.stopPropagation();_c1Clear('end')" title="Remove">&#128465;</button>
            </div>
            <div class="c1meta">
              <div class="c1title">Last frame <span class="c1opt">optional</span> <span class="c1sub-inline">the clip lands on this</span></div>
              <div class="c1acts">
                <button class="clip-reset" onclick="_c1Gen('end')">&#10024; generate</button>
                <button class="clip-reset" onclick="$('c1EndFile').click()">&#8679; upload</button>
                <button class="clip-reset" id="c1EndEdit" style="display:none" onclick="_c1Edit('end')">&#9998; edit</button>
              </div>
            </div>
            <input type="file" id="c1EndFile" accept="image/*" style="display:none" onchange="_c1SetImg('end',this.files)">
          </div>
          <div class="c1frame">
            <div class="c1thumbwrap">
              <div id="c1V2VSlot" class="c1thumb" onclick="$('c1V2VFile').click()"><span class="c1plus">&#127909;</span></div>
              <button class="c1trash" id="c1V2VClear" style="display:none" onclick="event.stopPropagation();_c1V2VClear()" title="Remove">&#128465;</button>
            </div>
            <div class="c1meta">
              <div class="c1title">Control video <span class="c1opt">optional</span> <span class="c1sub-inline">drive motion from a video (v2v)</span></div>
              <div class="c1acts">
                <button class="clip-reset" onclick="$('c1V2VFile').click()">&#8679; upload video</button>
                <button class="clip-reset" id="c1V2VEdit" style="display:none" onclick="openClipModal(1)">&#9881; options</button>
                <button class="gen-btn" id="c1FaceSplit" style="display:none;width:100%;margin-top:8px;padding:10px;font-size:13px" onclick="_c1FaceSplit()" title="Detect faces and cut this control video into timeline clips">&#9986; Split into clips by faces</button>
              </div>
            </div>
            <input type="file" id="c1V2VFile" accept="video/*" style="display:none" onchange="_c1V2VSet(this.files)">
          </div>
        </div>
      </div>

      <div class="sec" id="lastSec" style="display:none">
        <div class="sec-label"><span class="icon">&#9209;</span> Last frame <span class="c">required</span></div>
        <div class="dropzone" id="dropLast" onclick="document.getElementById('fileLast').click()">&#127937;&nbsp; Click or drop the ending image</div>
        <input type="file" id="fileLast" accept="image/*" hidden>
        <div class="hintline">FLF2V generates the motion that morphs the first frame into the last one.</div>
      </div>

      <div class="sec" id="refSec" style="display:none">
        <div class="sec-label"><span class="icon">&#128100;</span> Reference image <span class="c">subject to keep</span></div>
        <div class="dropzone" id="dropRef" onclick="document.getElementById('fileRef').click()">&#128100;&nbsp; Click or drop a character/subject</div>
        <input type="file" id="fileRef" accept="image/*" hidden>
        <div class="hintline">VACE keeps this subject consistent across the generated video. The "Start frame" above is optional extra conditioning.</div>
      </div>

      <div class="sec">
        <div class="sec-label"><span class="icon">&#10022;</span> <span id="promptLbl">Prompt</span>
          <button class="spk-gear" id="autoCfgGear" style="margin-left:auto" onclick="openAutoCfg()" title="Auto Prompt settings: standing instructions &amp; extra context">&#9881;</button>
          <button class="q-clear" id="autoBtn" style="margin-left:8px" onclick="autoPrompt()" title="Look at the image with GPT and write the prompt (and dialogue on LTX)">&#10024; Auto Prompt</button></div>
        <textarea class="ta" id="prompt" oninput="if(currentEngine==='ltx'){renderDialog();_updateSceneRoleHint();}">a calm ocean at golden hour, gentle waves rolling, cinematic slow push-in, the water shimmering</textarea>
        <div class="hintline" id="promptRoleHint" style="display:none"></div>
        <div class="sec-label" style="margin:12px 0 8px" id="negLabel"><span class="icon">&#8856;</span> Negative</div>
        <textarea class="ta neg" id="neg">overexposed, static, blurred details, subtitles, worst quality, low quality, JPEG artifacts, ugly, deformed, disfigured, messy background, still picture</textarea>
        <button class="gen-btn" id="addToTimelineBtn" style="display:none;width:100%;margin-top:12px" onclick="addToTimeline()" title="Commits this image &amp; scene as a clip and clears the fields for the next one">&#10133; Create clip</button>
        <label id="appendTlWrap" title="On: add this clip after the existing timeline. Off: start a new timeline." style="display:none;align-items:center;gap:7px;margin-top:8px;font-size:11px;color:var(--text-muted);cursor:pointer;white-space:nowrap">
          <input type="checkbox" id="appendTl"> Append
        </label>
        <div id="splitHint" title="This control video is longer than one segment, so it will be split into chained clips and stitched back after rendering. Adjust seconds-per-clip here." style="display:none;align-items:center;gap:6px;margin-top:8px;font-size:11px;color:var(--gold);white-space:nowrap">
          &#9986;&nbsp;<b id="splitN">2</b>&#215;<b id="splitPer">15</b>s
          <input type="number" id="splitSeg" min="4" max="40" step="1" style="width:46px;padding:2px 4px;background:var(--surface-2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--font-mono);font-size:11px;margin-left:auto">s
        </div>
        <div id="addToTimelineHint" style="display:none"></div>
      </div>

      <!-- timeline moved below the stage -->


      <div class="sec" id="lengthSec">
        <div class="sec-label"><span class="icon">&#9201;</span> Length <span class="c">long video</span></div>
        <div class="slider-row"><span class="sl">Clips</span>
          <input type="range" id="segments" min="1" max="12" value="1"><input class="sv" id="segmentsV" value="1"></div>
        <div class="lenbox"><span>&#9202;</span> <span id="durHint">~5 s &middot; 1 clip</span></div>
        <div class="hintline">Each clip is one ~5 s Wan generation. Extra clips are chained: the last frame of each becomes the start of the next. Quality drifts over many clips &mdash; use <b style="color:var(--gold)">Lightning</b> so long renders stay fast.</div>
      </div>

      <div class="sec">
        <div id="profileWrap">
          <div class="sec-label"><span class="icon">&#9638;</span> Resolution</div>
          <select class="model-select" id="profile">
            <option value="480P">480P &mdash; faster, lighter</option>
            <option value="720P">720P &mdash; sharper, heavier</option>
          </select>
        </div>
        <div id="flfNote" class="hintline" style="display:none">FLF2V runs at 720P only.</div>
        <div id="ltxResHint" class="hintline gold" style="display:none">LTX renders internally then 2&times;-upscales: 480P &rarr; ~704p output, 720P &rarr; ~1088p (1080p-class) output.</div>
        <div id="vaceSizeWrap" style="display:none">
          <div class="sec-label"><span class="icon">&#9638;</span> VACE model</div>
          <select class="model-select" id="vaceSize">
            <option value="1.3B">1.3B &mdash; 480P, light &amp; fast</option>
            <option value="14B">14B &mdash; 720P, heavier</option>
          </select>
        </div>
        <div class="slider-row" style="margin-top:12px"><span class="sl">FPS</span>
          <input type="range" id="fps" min="8" max="30" value="16"><input class="sv" id="fpsV" value="16"></div>
      </div>

      <div class="sec">
        <div class="adv-toggle" id="advToggle" onclick="toggleAdv()"><span class="icon">&#9881;</span> Advanced settings <span id="advCaret">&#9656;</span></div>
        <div class="adv-body" id="advBody">
          <div id="advWanOnly">
            <div class="slider-row"><span class="sl">Steps</span>
              <input type="range" id="steps" min="1" max="60" value="40"><input class="sv" id="stepsV" value="40"></div>
            <div class="slider-row"><span class="sl">Guidance</span>
              <input type="range" id="guid" min="0" max="12" step="0.1" value="5.0"><input class="sv" id="guidV" value="5.0"></div>
            <div class="slider-row"><span class="sl">Flow shift</span>
              <input type="range" id="shift" min="1" max="12" step="0.5" value="3.0"><input class="sv" id="shiftV" value="3.0"></div>
          </div>
          <div class="slider-row"><span class="sl">Frames/clip</span>
            <input type="range" id="frames" min="17" max="81" step="4" value="81"><input class="sv" id="framesV" value="81"></div>
          <div class="hintline" id="frameRuleHint">Frames snap to Wan's 4n+1 rule; 81 is the trained length. Flow shift ~3 for 480P, ~5 for 720P.</div>
          <div id="advWanOnly2">
            <label class="civ-nsfw-toggle" style="margin-top:10px"><input type="checkbox" id="varySeed" checked> Vary seed per clip (chained long video)</label>
            <div class="hintline">On: each chained clip uses seed+N for more varied motion. Off: every clip reuses the same seed &mdash; steadier, but can loop mannerisms.</div>
          </div>
          <div id="advLtx" style="display:none">
            <div class="slider-row"><span class="sl">Img strength</span>
              <input type="range" id="ltxStrength" min="0.5" max="1" step="0.05" value="0.9"><input class="sv" id="ltxStrengthV" value="0.9"></div>
            <div class="hintline">THE motion knob: how hard frame 0 is locked to your photo. 1.0 = strongest identity but can bias toward a frozen shot; 0.8&ndash;0.9 = identity kept with freer motion; lower = looser, livelier, less faithful.</div>
            <div class="slider-row" style="margin-top:10px"><span class="sl">Cond. CRF</span>
              <input type="range" id="ltxCrf" min="0" max="51" step="1" value="33"><input class="sv" id="ltxCrfV" value="33"></div>
            <div class="hintline">H.264 compression applied to the conditioning image before encoding (0 = lossless, package default 33). Lower keeps finer detail from your photo; the default matches how LTX was trained.</div>
            <label class="civ-nsfw-toggle" style="margin-top:10px"><input type="checkbox" id="ltxEnhance"> Enhance prompt (Gemma rewriter)</label>
            <div class="hintline">Gemma expands short prompts into detailed ones. Only use with SHORT prompts &mdash; with long structured prompts it can blow the token budget and degrade output.</div>
            <div class="sec-label" style="margin:12px 0 6px"><span class="icon">&#9881;</span> GPU offload</div>
            <select class="model-select" id="ltxOffload">
              <option value="auto" selected>Auto &mdash; keep on GPU if it fits (~38 GB+), else CPU stream</option>
              <option value="none">None &mdash; keep model on GPU (fastest, needs ~38 GB+)</option>
              <option value="cpu">CPU &mdash; stream weights (fits 24 GB, slower)</option>
            </select>
            <div class="hintline">Audio note: LTX's distilled pipeline ALWAYS generates audio and exposes no audio knobs &mdash; direct it in the prompt instead: quoted dialogue for speech ('she says: "..."'), or 'no music, quiet natural room tone' to suppress a score. Steps, guidance and negative prompt are baked into the distilled schedule.</div>
            <label style="display:flex;align-items:center;gap:8px;margin-top:10px;cursor:pointer;font-size:12px;color:var(--text)">
              <input type="checkbox" id="stripMusic"> <span>&#127925;&#10006; Strip music from the audio <span class="c" style="font-weight:400">(keeps dialogue &amp; sound effects; runs after render)</span></span>
            </label>
            <div class="hintline" style="margin-top:4px">Uses AI source separation (Demucs) to remove a musical score while keeping speech and SFX. Best-effort &mdash; music and ambient effects can overlap, so a faint trace may remain. For a hard guarantee, mute the clip in the Editor and add your own audio.</div>
            <div class="sec-label" style="margin:14px 0 6px"><span class="icon">&#9670;</span> Video-to-video segment length</div>
            <div class="seed-row" style="align-items:center;gap:10px">
              <input type="number" id="v2vSeg" value="20" min="4" max="40" step="1" style="width:90px">
              <span style="font-size:12px;color:var(--text-muted)">seconds per pass</span>
            </div>
            <div class="hintline" style="margin-top:4px">Longer segments = fewer seams &amp; less drift, but more VRAM per pass. A v2v clip longer than this is split into chained segments. Default 20s; drop to 8&ndash;10s if a segment runs out of memory.</div>
          </div>
          <div class="sec-label" style="margin:14px 0 6px"><span class="icon">&#127908;</span> Speakers &amp; voices</div>
          <button class="gen-btn gen-btn-secondary" style="width:100%;padding:9px;font-size:12px" onclick="openSpeakerModal()"><span id="spkBtnLabel">&#127908; Add speakers &amp; voices</span></button>
          <div class="sec-label" style="margin:14px 0 6px"><span class="icon">&#9670;</span> Seed</div>
          <div class="seed-row">
            <input type="number" id="seed" value="42">
            <button class="gen-btn gen-btn-secondary" style="width:auto;padding:8px 12px" onclick="randSeed()">&#127922;</button>
          </div>
        </div>
      </div>

      <div class="sec">
        <div class="sec-label"><span class="icon">&#9880;</span> LoRAs</div>
        <button class="gen-btn gen-btn-secondary" style="width:100%;margin-bottom:8px" onclick="civOpenSearch()">&#128269; Browse Civitai LoRAs</button>
        <input type="text" class="lora-url" id="loraUrl" placeholder="paste HF or Civitai .safetensors URL">
        <div class="lora-row">
          <input type="text" class="lora-url" id="loraName" placeholder="name (optional)">
          <input type="number" class="lora-url lora-scale" id="loraScale" value="1.0" min="0" max="2" step="0.05">
        </div>
        <button class="gen-btn gen-btn-secondary" id="addLoraBtn" style="margin-top:8px;width:100%" onclick="addLora()">+ Add LoRA</button>
        <div id="loraList"></div>
      </div>
    </div>

    <div class="sidebar-foot">
      <button class="gen-btn" id="genBtn" onclick="generate()">&#10022; Generate Video</button>
    </div>
  </aside>

  <main class="stage" id="stage">
    <div class="viewer" id="viewer">
      <div class="ph"><span class="big">&#127909;</span>Your video plays here, full size.<br>Add a <b>starting frame</b>, write a prompt, hit <b>Generate</b>.</div>
    </div>
    <div class="stage-tools" id="stageTools"><a class="chip" id="dlBtn" download="missinglink_video.mp4">&#11015; Download</a><button class="chip" id="vr180Btn" onclick="convertVR180()" title="Convert this clip to 180&deg; VR (mono, headset-ready)">&#129488; 180&deg; VR</button><button class="chip" id="faceKfBtn" onclick="extractFaceKeyframes()" title="Find the first face every 10s and download the frames + timestamps">&#128100; Face frames</button><button class="chip" id="autoClipBtn" onclick="faceAutoClip()" title="Auto-cut into clips at face boundaries; download each clip + its first/last frames for consistent-character reskinning">&#9986; Auto-clip</button><button class="chip" id="nextSceneBtn" onclick="autoNextScene()" title="Look at the last clip and build the next scene's starting image">&#127916; Auto Next Scene</button><button class="chip" id="nextCfgBtn" onclick="openNextCfg()" title="Auto Next Scene settings: story context &amp; instructions">&#9881;</button><button class="chip" id="clearBtn" onclick="clearStage()" title="Clear the stage">&#128465; Clear</button></div>
    <div class="stage-status" id="stage_status"></div>
    <div class="progrow"><div class="prog"><div class="fill" id="fill"></div></div></div>
    <!-- ── STAGE TIMELINE (relocated from the sidebar) ── -->
    <div class="stage-timeline" id="dlgSec" style="display:none">
      <div class="stl-head">
        <span class="stl-title"><span class="icon">&#127916;</span> Timeline <span class="c">clips play in order, continuously</span></span>
        <button class="q-clear" id="charPersistBtn" onclick="openCharPersist()" title="Face-swap a character reference onto every clip's start frame (Qwen faceswap, token-billed)">&#128100; Persist character</button>
        <button class="q-clear" onclick="clearTimeline()" title="Remove all clips and start fresh">&#128465;</button>
        <button class="q-clear" id="stlMin" onclick="toggleStl()" title="Minimize the timeline">&#9662;</button>
      </div>
      <div class="stl-strip" id="clipOverview"></div>
      <div class="hintline stl-hint">Tap a clip to edit, or &#10024; Auto-write to fill it. Add clips at the end of the strip.</div>
    </div>


    <!-- free-floating Jobs panel: queue + history -->
    <div class="float" id="jobsPanel">
      <div class="float-head" id="jobsHandle">
        <span class="float-grip">&#8942;&#8942;</span>
        <span class="float-title"><span class="icon">&#9776;</span> Jobs</span>
        <div class="float-btns"><button class="float-min" id="jobsMin" onclick="toggleMin(event)" title="Minimize">&ndash;</button></div>
      </div>
      <div class="float-inner" id="jobsInner">
        <div class="float-sec q" id="secQueue">
          <div class="float-sec-hd" onclick="toggleSec('secQueue')"><span class="icon">&#9636;</span> Queue <span class="q-count" id="qCount">0</span><span class="caret">&#9662;</span></div>
          <div class="float-sec-body"><div id="queueBody"><div class="empty">No active jobs.</div></div></div>
        </div>
        <div class="float-sec h" id="secHistory">
          <div class="float-sec-hd" onclick="toggleSec('secHistory')"><span class="icon">&#9638;</span> History <span class="q-count" id="hCount">0</span><span class="caret">&#9662;</span></div>
          <div class="float-sec-body"><div id="historyBody"><div class="empty">Finished videos appear here. Click to replay.</div></div></div>
        </div>
      </div>
    </div>
  </main>

  <footer class="dock collapsed" id="dock">
    <div class="dock-head" onclick="toggleDock()">
      <span class="dock-title"><span class="icon">&#9655;</span> Console / debug log</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="copyConsole(event)" id="conCopyBtn">copy</button>
      <button class="q-clear" onclick="clearConsole(event)">clear</button>
      <button class="q-clear" id="dockToggle">&#9652; Show</button>
    </div>
    <div class="console" id="console"></div>
  </footer>
</div>

<div class="modal-scrim" id="overlay">
  <div class="history-modal-panel">
    <div class="history-modal-head">
      <div class="history-modal-title"><span class="icon">&#127760;</span> SESSION GALLERY
        <span class="q-count" id="ovCount" style="margin-left:0">0</span></div>
      <button class="modal-close" onclick="closeGallery()">&#10005;</button>
    </div>
    <div class="history-modal-body"><div class="history-grid" id="ovGrid"></div></div>
  </div>
</div>

<!-- ── TIMELINE EDITOR (full-screen tab) ── -->
<div id="editorTab" style="display:none">
  <div id="edPreviewWrap">
    <video id="edPreview" playsinline></video>
    <div id="edPreviewEmpty">&#127916;<br>Add clips below to build your timeline</div>
  </div>
  <div id="edTransport">
    <button class="ed-tbtn" onclick="edPlayPause()" id="edPlayBtn" title="Play / pause">&#9654;</button>
    <button class="ed-tbtn" onclick="edStop()" title="Stop">&#9209;</button>
    <span id="edTime" class="ed-time">0:00 / 0:00</span>
    <span class="dock-sp"></span>
    <span class="ed-zoom" title="Zoom timeline">&#128269;<input type="range" id="edZoom" min="20" max="400" value="90" oninput="edSetZoom(this.value)"></span>
    <button class="ed-tbtn" onclick="edAddAudioTrack()" title="Add audio track">&#127925;<span class="ed-plus">+</span></button>
    <button class="ed-tbtn ed-export" id="edExportBtn" onclick="edExport()" title="Export the timeline as one MP4">&#8681;</button>
  </div>
  <div id="edTimelineWrap">
    <div id="edRuler"></div>
    <div id="edTracks"></div>
    <div id="edPlayhead"></div>
  </div>
  <div id="edPool">
    <div id="edPoolItems"></div>
  </div>
</div>

<!-- ── CIVITAI LORA SEARCH MODAL (ported from MissingLink SDXL studio) ── -->
<!-- ── LoRA DETAIL / INFO PAGE ── -->
<div class="civ-search-modal" id="loraDetailModal" style="display:none;z-index:9700">
  <div class="civ-search-box" style="max-width:1100px">
    <div class="civ-search-head">
      <span id="loraDetailName">LoRA</span>
      <span class="dock-sp"></span>
      <a id="loraDetailLink" href="#" target="_blank" class="q-clear" style="text-decoration:none">↗ View on Civitai</a>
      <button class="civ-search-close" onclick="closeLoraDetail()">&#10005;</button>
    </div>
    <div id="loraDetailBody" style="flex:1;overflow-y:auto;padding:18px 22px"></div>
    <div class="clip-modal-foot">
      <span id="loraDetailMeta" class="hintline" style="margin:0;flex:1"></span>
      <button class="gen-btn" id="loraDetailAdd" style="width:auto;padding:0 22px" onclick="loraDetailAdd()">+ Add LoRA</button>
    </div>
  </div>
</div>

<div class="civ-search-modal" id="civSearchModal" style="display:none">
  <div class="civ-search-box">
    <div class="civ-search-head">
      <span>Search Civitai LoRAs</span>
      <button class="civ-search-close" onclick="civCloseSearch()">&#10005;</button>
    </div>
    <div class="civ-search-controls">
      <input type="text" id="civSearchQuery" placeholder="Search by name&hellip;" onkeydown="if(event.key==='Enter')civRunSearch(true)">
      <select id="civSearchBase">
        <option value="all">All base models</option>
      </select>
      <select id="civSearchSort">
        <option>Most Downloaded</option>
        <option>Highest Rated</option>
        <option>Newest</option>
      </select>
      <label class="civ-nsfw-toggle"><input type="checkbox" id="civSearchNsfw"> NSFW</label>
      <button class="gen-btn gen-btn-secondary" style="width:auto;padding:9px 16px" onclick="civRunSearch(true)">Search</button>
    </div>
    <div id="civSearchNote" class="hintline" style="display:none;margin:0 0 6px"></div>
    <div class="civ-search-tags" id="civSearchTags"></div>
    <div class="civ-search-grid" id="civSearchGrid"></div>
    <div class="civ-search-status" id="civSearchStatus"></div>
  </div>
</div>
<!-- ── ADD CLIP: extend vs new scene chooser ── -->
<div class="civ-search-modal" id="addClipModal" style="display:none">
  <div class="clip-modal-box" style="width:min(560px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127916;</span>&nbsp; Add a clip</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeAddClip()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div class="hintline" style="margin:0">The new clip continues your movie from the previous clip. Choose how:</div>

      <button class="addclip-opt" onclick="addClipExtend()">
        <div class="ac-icon">&#128172;</div>
        <div class="ac-txt">
          <div class="ac-title">Extend the same scene</div>
          <div class="ac-sub">Keep the exact same shot &amp; setting — continue the moment or keep the conversation going. No new image: at render time it flows straight on from the previous clip's real last frame.</div>
        </div>
      </button>

      <button class="addclip-opt" onclick="addClipNewScene()">
        <div class="ac-icon">&#127917;</div>
        <div class="ac-txt">
          <div class="ac-title">New scene <span class="ac-badge">editable</span></div>
          <div class="ac-sub">Add a fresh clip that continues the story and open its editor. Write the scene &amp; dialogue yourself, or hit ✨ Auto-write to fill it from the story. Generate or upload a start image inside the editor.</div>
        </div>
      </button>

      <div style="display:flex;align-items:center;gap:8px;margin-top:2px">
        <button class="q-clear" onclick="openNextCfg()" title="New-scene image settings">&#9881; Image settings</button>
        <span class="hintline" style="margin:0">Story context &amp; image quality for the new-scene generator.</span>
      </div>
    </div>
  </div>
</div>

<!-- ── SPEAKERS & VOICES MODAL ── -->
<div class="civ-search-modal" id="speakerModal" style="display:none">
  <div class="clip-modal-box" style="width:min(620px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127908;</span>&nbsp; Speakers &amp; voices</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeSpeakerModal()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div class="hintline" style="margin:0">Each speaker's <b>voice description</b> is what LTX uses to synthesize their voice — accent, pitch, age, timbre, pace. Keep it consistent to keep the voice consistent across clips.</div>
      <div id="spkList"></div>
      <button class="gen-btn gen-btn-secondary" style="width:100%;padding:10px;font-size:11px" onclick="addSpeaker()">+ Add speaker</button>
    </div>
    <div class="clip-modal-foot">
      <span class="dock-sp"></span>
      <button class="gen-btn" style="width:auto;padding:0 22px" onclick="closeSpeakerModal()">Done</button>
    </div>
  </div>
</div>

<!-- ── AUTO NEXT SCENE: settings + review modals ── -->
<div class="civ-search-modal" id="nextCfgModal" style="display:none;z-index:9800">
  <div class="clip-modal-box" style="width:min(600px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127916;</span>&nbsp; Auto Next Scene settings</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeNextCfg()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div>
        <label class="clip-modal-lbl" style="margin:0 0 8px">Next shot <span class="c">how this camera setup relates to the last</span></label>
        <select id="nextCfgShot" class="model-select" style="width:100%">
          <option value="auto" selected>Auto — natural next beat (same scene)</option>
          <option value="angle">Same moment, new camera angle</option>
          <option value="location">Same character(s), different location</option>
          <option value="newchar">New character, same world</option>
        </select>
        <div class="hintline" style="margin:6px 0 0">The best available image of the previous shot (your uploaded frame, or the prior clip's start image) is sent to gpt-image-2 as a visual reference, so faces, wardrobe and the world carry over. Returning characters stay identical.</div>
      </div>
      <div>
        <label class="clip-modal-lbl">Story context <span class="c">what's going on, who's who, where this is heading</span></label>
        <textarea id="nextCfgContext" class="clip-modal-ta" placeholder="e.g. Detective Rourke is chasing the suspect through the market. Next he should corner them by the fish stalls. Keep it tense, 1970s New York, overcast light."></textarea>
      </div>
      <div>
        <label class="clip-modal-lbl">Instructions <span class="c">how to build the next image</span></label>
        <textarea id="nextCfgInstr" class="clip-modal-ta" placeholder="e.g. Keep the same wardrobe and face. Photoreal, shallow depth of field. Reverse angle looking back at her."></textarea>
      </div>
      <div>
        <label class="clip-modal-lbl" style="margin:0">Image quality
          <select id="nextCfgQuality" class="model-select" style="margin-left:8px;width:auto;display:inline-block">
            <option value="low">Low (fast, cheap)</option>
            <option value="medium" selected>Medium</option>
            <option value="high">High (slow, best)</option>
          </select>
        </label>
      </div>
      <div class="hintline" style="margin:0">Auto Next Scene reads the last clip, writes a prompt, generates the image with gpt-image-2 (using the last frame as reference), self-checks identity &amp; world consistency, then tells you if it fits. You can accept it, regenerate, or replace it with your own image.</div>
    </div>
    <div class="clip-modal-foot">
      <button class="q-clear" onclick="clearNextCfg()">Clear</button>
      <span class="dock-sp"></span>
      <button class="gen-btn" style="width:auto;padding:0 22px" onclick="closeNextCfg()">Done</button>
    </div>
  </div>
</div>

<!-- ── NEXT SCENE: two-stage COMPOSE (edit prompt + attach images, then generate) ── -->
<!-- ── EDIT IMAGE MODAL (gpt-image-2 edit with reference slots) ── -->
<div class="civ-search-modal" id="editImgModal" style="display:none;z-index:9700">
  <div class="clip-modal-box" style="width:min(680px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#9998;</span>&nbsp; <span id="editImgTitle">Edit image</span></span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeEditImg()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div>
        <label class="clip-modal-lbl">Image being edited <span class="c">click to view full size</span></label>
        <div id="editImgSrc" class="editimg-src" onclick="_editImgFull()"></div>
      </div>
      <div>
        <label class="clip-modal-lbl">Change to make <span class="c">describe the edit — what to add, remove or change</span></label>
        <textarea id="editImgInstr" class="clip-modal-ta" style="min-height:80px" placeholder="e.g. make it night, change the jacket to red, add a crowd behind them…"></textarea>
      </div>
      <div>
        <label class="clip-modal-lbl">Reference images <span class="c">optional — guide the edit (style, subject, object to add)</span></label>
        <div id="editImgRefs" class="nc-imgs"></div>
        <label class="nc-upload">+ Upload reference(s)
          <input type="file" id="editImgRefFile" accept="image/*" multiple style="display:none" onchange="_editImgAddRefs(this.files)">
        </label>
      </div>
      <div id="editLoraPanel" style="border-top:1px solid var(--border);margin-top:6px;padding-top:10px">
        <label class="clip-modal-lbl"><span class="icon">&#127900;</span> LoRA strengths <span class="c">Qwen image-edit adapters — 0 = off</span></label>
        <div class="edit-lora-row"><span>&#128100; Face swap</span>
          <input type="range" id="loraFaceswap" min="0" max="1.5" step="0.05" value="0" oninput="_editLoraSync()">
          <span class="sv" id="loraFaceswapV">0.00</span></div>
        <div class="hintline" id="loraFaceswapHint" style="display:none;margin:-2px 0 6px">Face swap uses reference 1 as the <b>face source</b> (put the character's face there).</div>
        <div class="edit-lora-row"><span>&#128506; Camera angles</span>
          <input type="range" id="loraAngles" min="0" max="1.5" step="0.05" value="0" oninput="_editLoraSync()">
          <span class="sv" id="loraAnglesV">0.00</span></div>
        <div class="edit-lora-row"><span>&#10024; Skin detail</span>
          <input type="range" id="loraSkin" min="0" max="1.5" step="0.05" value="0" oninput="_editLoraSync()">
          <span class="sv" id="loraSkinV">0.00</span></div>
        <div class="edit-lora-row"><span>&#128200; Upscale/refine</span>
          <input type="range" id="loraUpscale" min="0" max="1.5" step="0.05" value="0" oninput="_editLoraSync()">
          <span class="sv" id="loraUpscaleV">0.00</span></div>
        <div class="hintline" style="margin-top:4px">With a LoRA active, the edit runs on the Qwen image-edit worker (Lightning 4-step, token-billed) instead of the default editor.</div>
      </div>
    </div>
    <div class="civ-search-foot">
      <span id="editImgTok" class="tok-badge" onclick="buyTokens()" title="Buy more tokens">— tokens</span>
      <span class="dock-sp"></span>
      <button class="gen-btn" id="editImgGo" style="width:auto;padding:0 22px" onclick="_editImgRun()">&#9998; Apply edit</button>
    </div>
  </div>
</div>
<!-- ── FULLSCREEN IMAGE VIEWER (with download) ── -->
<div id="imgFullView" style="display:none" onclick="_imgFullClose(event)">
  <button class="ifv-close" onclick="_imgFullClose(event,true)" title="Close">&#10005;</button>
  <a id="ifvDownload" class="ifv-dl" download="image.png" onclick="event.stopPropagation()" title="Download">&#8681; Download</a>
  <img id="ifvImg" src="" onclick="event.stopPropagation()">
</div>

<div class="civ-search-modal" id="nextComposeModal" style="display:none;z-index:9600">
  <div class="clip-modal-box" style="width:min(680px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127916;</span>&nbsp; <span id="nextComposeTitle">Compose image</span></span>
      <span class="dock-sp"></span>
      <span id="nextTokBadge" class="tok-badge" onclick="buyTokens()" title="Buy more tokens">— tokens</span>
      <button class="q-clear" onclick="closeNextCompose()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div>
        <label class="clip-modal-lbl">Image prompt <span class="c">what we'll send to gpt-image-2 — edit freely, then Generate</span></label>
        <textarea id="nextComposePrompt" class="clip-modal-ta" style="min-height:120px" placeholder="Describe the image you want…"></textarea>
      </div>
      <div>
        <label class="clip-modal-lbl" id="nextComposeRefLbl">Reference images <span class="c">optional — carried into the generation for consistency</span></label>
        <div id="nextComposeImgs" class="nc-imgs"></div>
        <label class="nc-upload">+ Upload image(s)
          <input type="file" id="nextComposeFile" accept="image/*" multiple style="display:none" onchange="_nextComposeAddFiles(this.files)">
        </label>
        <button class="nc-upload" id="nextCompositeBtn" style="display:none;margin-left:8px;background:none;border:1px solid var(--border)" onclick="_compositeTwoRefs()" title="Combine the first two reference images side-by-side into one — the reliable way to keep TWO characters consistent (LTX pins one frame, so two people must share a single reference)">&#9707; Combine 2 into one ref</button>
      </div>
      <div class="hintline" id="nextComposeCost" style="margin:0"></div>
    </div>
    <div class="clip-modal-foot">
      <button class="q-clear" onclick="openNextCfg()">⚙ Settings</button>
      <span class="dock-sp"></span>
      <button class="q-clear" id="nextComposeRepropose" onclick="_nextRepropose()">↺ Re-propose</button>
      <button class="gen-btn" id="nextComposeGen" style="width:auto;padding:0 22px" onclick="_nextComposeGenerate()">✦ Generate</button>
    </div>
  </div>
</div>

<div class="civ-search-modal" id="nextReviewModal" style="display:none">
  <div class="clip-modal-box" style="width:min(560px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127916;</span>&nbsp; Next scene</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeNextReview()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div id="nextReviewImgWrap" style="text-align:center">
        <img id="nextReviewImg" style="max-width:100%;max-height:46vh;border-radius:10px;border:1px solid var(--border)">
      </div>
      <div id="nextReviewVerdict" class="next-verdict"></div>
      <div id="nextReviewIntent" class="hintline" style="margin:0"></div>
      <div style="margin-top:14px">
        <label class="clip-modal-lbl" style="margin:0 0 6px">Edit this image <span class="c">describe a change and re-generate it</span></label>
        <div style="display:flex;gap:8px;align-items:flex-end">
          <textarea id="nextEditInstr" class="clip-modal-ta" style="min-height:44px;flex:1" placeholder="e.g. make it night, add a red scarf, turn her to face the camera…"></textarea>
          <button class="gen-btn gen-btn-secondary" id="nextEditBtn" style="width:auto;padding:0 16px;flex:0 0 auto" onclick="nextEditImage()" title="Apply this edit to the image">&#9998; Edit</button>
        </div>
      </div>
    </div>
    <div class="clip-modal-foot" style="flex-wrap:wrap;gap:8px">
      <button class="q-clear" onclick="nextReplaceManual()" title="Use your own image instead">&#128247; Use my own image</button>
      <button class="q-clear" onclick="regenNextScene()" title="Try again">&#8635; Regenerate</button>
      <span class="dock-sp"></span>
      <button class="gen-btn" style="width:auto;padding:0 22px" onclick="nextAccept()">Use this image &rarr;</button>
    </div>
  </div>
</div>
<input type="file" id="nextManualFile" accept="image/*" style="display:none">

<!-- ── AUTO PROMPT SETTINGS MODAL ── -->
<div class="civ-search-modal" id="autoCfgModal" style="display:none;z-index:9800">
  <div class="clip-modal-box" style="width:min(600px,96vw)">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#10024;</span>&nbsp; Auto Prompt settings</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeAutoCfg()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div>
        <label class="clip-modal-lbl">Standing instructions <span class="c">always applied when you tap Auto Prompt</span></label>
        <textarea id="autoCfgInstr" class="clip-modal-ta" placeholder="e.g. Always use a slow cinematic push-in. Keep it moody and desaturated. Prefer subtle, realistic motion over big camera moves. Write in present tense."></textarea>
      </div>
      <div>
        <label class="clip-modal-lbl">Extra context <span class="c">facts about the scene, characters, story the model can't see in the image</span></label>
        <textarea id="autoCfgContext" class="clip-modal-ta" placeholder="e.g. This is Detective Rourke, mid-50s, gravel voice, ex-military. The setting is 1970s New York. He's just found a key piece of evidence."></textarea>
      </div>
      <div class="hintline" style="margin:0">These are sent with the image every time. Instructions steer HOW the prompt is written; context adds facts the model should treat as true. Saved for this session.</div>
    </div>
    <div class="clip-modal-foot">
      <button class="q-clear" onclick="clearAutoCfg()" title="Clear both fields">Clear</button>
      <span class="dock-sp"></span>
      <button class="gen-btn" style="width:auto;padding:0 22px" onclick="closeAutoCfg()">Done</button>
    </div>
  </div>
</div>

<!-- ── CLIP EDITOR MODAL ── -->
<div class="civ-search-modal" id="charModal" style="display:none;z-index:9850">
  <div class="civ-search-inner" style="max-width:560px">
    <div class="civ-search-head">
      <span>&#128100; Persist character across clips</span>
      <button class="civ-x" onclick="closeCharPersist()">&#10005;</button>
    </div>
    <div style="padding:14px 16px">
      <div class="hintline" style="margin-bottom:10px">Face-swaps your reference onto each selected clip's <b>start frame</b>, keeping that clip's pose, lighting and framing. Each is a separate <b>100-token</b> job in the queue. Runs on the MissingLink Qwen worker.</div>
      <div class="sec-label" style="margin:6px 0"><span class="icon">&#127912;</span> Character reference (the face to apply)</div>
      <div id="charRefSlot" class="cf-slot" style="height:150px;border:1px dashed var(--border);border-radius:10px;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden" onclick="$('charRefFile').click()">
        <span class="cf-empty">+ upload the character reference</span>
      </div>
      <input type="file" id="charRefFile" accept="image/*" style="display:none" onchange="_charRefUpload(this.files)">
      <div class="seed-row" style="align-items:center;gap:10px;margin-top:12px">
        <span style="font-size:11px;color:var(--text-muted)">Face-swap strength</span>
        <input type="range" id="charStrength" min="0.5" max="1.5" step="0.05" value="1.0" style="flex:1">
        <span class="sv" id="charStrengthV">1.00</span>
      </div>
      <div class="sec-label" style="margin:14px 0 6px"><span class="icon">&#127916;</span> Apply to clips</div>
      <div id="charClipList" style="display:flex;flex-wrap:wrap;gap:6px"></div>
      <button class="gen-btn" id="charRunBtn" style="width:100%;margin-top:14px" onclick="runCharPersist()">&#128100; Swap character into <b id="charN">0</b> clip(s) &middot; <b id="charCost">0</b> tokens</button>
      <div class="hintline" id="charStatus" style="margin-top:8px;display:none"></div>
    </div>
  </div>
</div>
<div class="civ-search-modal" id="clipModal" style="display:none">
  <div class="clip-modal-box">
    <div class="civ-search-head">
      <span><span style="color:var(--gold)">&#127916;</span>&nbsp; <span id="clipModalTitle">Edit Clip</span></span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="closeClipModal()">&#10005; Close</button>
    </div>
    <div class="clip-modal-body">
      <div class="clip-modal-scene">
        <label class="clip-modal-lbl">Scene prompt <span class="c">for this clip — what's happening, camera, mood</span>
          <button class="clip-reset" id="clipSceneReset" onclick="resetClipScene()" title="Reset to the main prompt">reset to main</button>
        </label>
        <textarea id="clipSceneTa" class="clip-modal-ta" placeholder="Describe this clip's scene…"></textarea>
        <div class="clip-frames-row">
          <div class="clip-frame-slot">
            <label class="clip-modal-lbl" style="margin:0 0 6px">Starting image <span class="c">optional — the clip begins on this</span></label>
            <div id="clipStartSlot" class="clip-frame-drop">
              <span class="cf-empty">no start image — chains from the previous clip</span>
            </div>
            <input type="file" id="clipStartFile" accept="image/*" style="display:none" onchange="_clipModalSetImg('start',this.files)">
            <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">
              <button class="clip-reset" onclick="_clipModalGenImg('start')">&#10024; generate</button>
              <button class="clip-reset" onclick="$('clipStartFile').click()">&#8679; upload</button>
              <button class="clip-reset" id="clipStartEdit" style="display:none" onclick="_clipModalEditImg('start')">&#9998; edit</button>
              <button class="clip-reset" id="clipStartClear" style="display:none" onclick="_clipModalClearImg('start')">clear</button>
            </div>
          </div>
          <div class="clip-frame-slot">
            <label class="clip-modal-lbl" style="margin:0 0 6px">End frame <span class="c">optional — the clip lands on this (LTX)</span></label>
            <div id="clipEndSlot" class="clip-frame-drop">
              <span class="cf-empty">no end frame</span>
            </div>
            <input type="file" id="clipEndFile" accept="image/*" style="display:none" onchange="_clipModalSetImg('end',this.files)">
            <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">
              <button class="clip-reset" onclick="_clipModalGenImg('end')">&#10024; generate</button>
              <button class="clip-reset" onclick="$('clipEndFile').click()">&#8679; upload</button>
              <button class="clip-reset" id="clipEndEdit" style="display:none" onclick="_clipModalEditImg('end')">&#9998; edit</button>
              <button class="clip-reset" id="clipEndClear" style="display:none" onclick="_clipModalClearImg('end')">clear</button>
            </div>
          </div>
        </div>
        <!-- ── Video-to-video (IC-LoRA motion/structure control) ── -->
        <div class="v2v-sec">
          <label class="clip-modal-lbl" style="margin:14px 0 6px">&#127909; Video-to-video <span class="c">drive this clip's motion from a video (LTX IC-LoRA)</span></label>
          <div id="clipV2VSlot" class="v2v-drop"><span class="cf-empty">+ upload a control video</span></div>
          <input type="file" id="clipV2VFile" accept="video/*" style="display:none" onchange="_clipV2VSet(this.files)">
          <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap">
            <button class="clip-reset" onclick="$('clipV2VFile').click()">&#8679; upload video</button>
            <button class="clip-reset" id="clipV2VClear" style="display:none" onclick="_clipV2VClear()">clear</button>
          </div>
          <div id="clipV2VOpts" style="display:none;margin-top:8px">
            <label class="clip-modal-lbl" style="margin:0 0 4px">Control type</label>
            <select id="clipV2VType" class="clip-select" onchange="_clipV2VTypeChange()">
              <option value="raw">Raw video — loose conditioning (fastest, no preprocessing)</option>
              <option value="canny">Canny edges — structure transfer</option>
              <option value="depth">Depth — 3D structure / motion transfer</option>
              <option value="motion_track">Motion track — follow spline trajectories</option>
            </select>
            <div class="slider-row" style="margin-top:8px"><span class="sl">Strength</span>
              <input type="range" id="clipV2VStrength" min="0.1" max="1" step="0.05" value="1" oninput="_clipV2VStrengthChange(this.value)">
              <span class="sv" id="clipV2VStrengthV">1.0</span></div>
            <div class="hintline" id="clipV2VSlice" style="display:none;color:var(--gold)"></div>
            <button class="clip-reset" id="clipFaceSplit" style="display:none;margin-top:8px" onclick="clipControlFaceSplit()" title="Cut this control video into clips at face boundaries; each becomes a timeline clip with its first/last frames as anchors">&#9986; Split control video by faces</button>
            <label style="display:flex;align-items:center;gap:8px;margin-top:8px;cursor:pointer;font-size:12px">
              <input type="checkbox" id="clipV2VCopyAudio" onchange="_clipV2VAudioChange(this.checked)"> Use the control video's original audio (instead of generating it)</label>
            <div class="hintline" style="margin-top:6px">The control video's <b>first frame</b> is set as this clip's start image — edit it (✎) to change the scene while keeping the motion. Keep the subject's position when editing so motion stays aligned. <b>Length auto-matches the video</b> (set a clip length to override). The <b>scene prompt is optional</b> — it steers appearance/style, not motion (the video drives that).</div>
          </div>
        </div>
        <div class="clip-modal-metarow">
          <label class="clip-modal-lbl" style="margin:0">Length
            <input type="number" id="clipLenInp" class="clip-secs" min="1" max="60" step="0.5" style="margin-left:6px"> <span class="clip-unit">s</span>
            <button class="clip-reset" id="clipLenReset" onclick="resetClipLen()" title="Auto from dialogue">auto</button>
          </label>
          <span class="clip-modal-lenhint" id="clipLenHint"></span>
        </div>
      </div>
      <div class="clip-modal-dlg">
        <label class="clip-modal-lbl">Dialogue in this clip <span class="c">spoken in order, lip-synced</span></label>
        <div id="clipDlgList"></div>
        <div style="display:flex;gap:8px;margin-top:6px;align-items:stretch">
          <button class="gen-btn gen-btn-secondary" style="flex:1;padding:9px;font-size:11px" onclick="clipModalAddLine()">+ Add line</button>
          <button class="gen-btn gen-btn-secondary" id="clipAutoDlgBtn" style="flex:1;padding:9px;font-size:11px" onclick="clipModalAutoDialog()" title="AI writes this clip's scene AND dialogue, continuing the story from previous clips — builds on whatever you've already typed, looks at the start (and end frame if set)">&#10024; Auto-write scene &amp; dialogue</button>
          <button class="spk-gear" onclick="openAutoCfg()" title="Auto Prompt settings: standing instructions &amp; extra context" style="flex:0 0 auto;padding:0 12px">&#9881;</button>
        </div>
      </div>
    </div>
    <div class="clip-modal-foot">
      <button class="q-clear" id="clipDelBtn" onclick="deleteClipFromModal()" title="Remove this clip and its lines">&#128465; Delete clip</button>
      <span class="dock-sp"></span>
      <button class="gen-btn" style="width:auto;padding:0 22px" onclick="closeClipModal()">Done</button>
    </div>
  </div>
</div>

<!-- ── MISSINGLINK LOGIN GATE (Google sign-in) ── -->
<div class="civ-search-modal" id="mlGate" style="display:none">
  <div class="civ-search-box" style="width:min(460px,94vw);height:auto;max-height:none">
    <div class="civ-search-head"><span><span style="color:var(--gold)">&#9670;</span>&nbsp; MissingLink Video Studio</span></div>
    <div style="padding:22px 24px 24px">

      <!-- signed-out: sign in with Google -->
      <div id="mlGateLogin">
        <div class="hintline" style="font-size:12px;margin:0 0 16px;line-height:1.8">Sign in with your <b style="color:var(--gold)">MissingLink</b> account to use the studio. Studio subscribers get unlimited video generation; new users get <b style="color:var(--gold)">25 free renders</b> to try it.</div>
        <a class="gen-btn" id="mlGoogleBtn" style="display:flex;align-items:center;justify-content:center;gap:8px;text-decoration:none" href="#" target="_blank" onclick="mlOpenSignin(event)">
          <span style="font-size:15px">&#128273;</span> Sign in with Google</a>
        <div class="hintline" style="margin-top:16px;font-size:11px;line-height:1.7">A sign-in window opens &mdash; after you sign in it returns here <b>automatically</b>. If it doesn&rsquo;t, paste the code it shows:</div>
        <div style="display:flex;gap:8px;margin-top:8px">
          <input type="text" class="lora-url" id="mlKey" style="flex:1" name="ml-notebook-code" placeholder="code (only if it doesn't auto-fill)" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" data-lpignore="true" data-1p-ignore="true" data-bwignore="true" data-form-type="other" onkeydown="if(event.key==='Enter')mlLogin()">
          <button class="gen-btn" style="width:auto;padding:0 16px" id="mlLoginBtn" onclick="mlLogin()">Unlock</button>
        </div>
        <div class="hintline" style="margin-top:14px;text-align:center">Not a member yet? <a id="mlSignupA" href="https://www.missinglink.build/" target="_blank" style="color:var(--gold)">See plans at missinglink.build &rarr;</a></div>
      </div>

      <!-- free trial used up: prompt to subscribe -->
      <div id="mlGateLimit" style="display:none">
        <div class="hintline gold" style="font-size:12px;margin:0 0 6px;line-height:1.8">You&rsquo;ve used all <b id="mlLimitN">25</b> free renders<span id="mlLimitEmail"></span>. Subscribe to MissingLink Studio for <b>unlimited</b> video generation.</div>
        <a class="gen-btn" style="display:flex;margin-top:12px;text-decoration:none" id="mlSubBtn" href="https://www.missinglink.build/" target="_blank">&#10024; Subscribe to MissingLink</a>
        <button class="gen-btn gen-btn-secondary" style="margin-top:8px;width:100%" onclick="mlRecheck()">I subscribed &mdash; check again</button>
        <button class="q-clear" style="margin-top:10px;width:100%" onclick="mlSwitchAccount()">Use a different account</button>
      </div>

      <div class="hintline" id="mlGateErr" style="color:var(--red);margin-top:12px;display:none"></div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<!-- ── FLOATING AGENT ── -->
<button id="agentFab" title="Open the storyboard agent" onclick="agentOpen()">&#10024; Agent</button>
<div id="agentPanel" style="display:none">
  <div id="agentHead">
    <span class="ag-grip">&#8942;&#8942;&#8942;</span>
    <span class="ag-title">&#10024; AGENT</span>
    <span class="dock-sp"></span>
    <button class="ag-btn" onclick="agentClear()" title="Clear conversation">&#8635; Reset</button>
    <button class="ag-btn" id="agentMinBtn" onclick="agentMin()" title="Show / hide the chat">&#9650; Show</button>
  </div>
  <div id="agentBody">
    <div id="agentLog"></div>
    <div id="agentScriptRow">
      <button class="ag-sbtn" onclick="downloadScript()" title="Download the full script (images, prompts, dialogue)">&#8681; Script</button>
      <label class="ag-sbtn" title="Upload a script .json">&#8679; Load<input type="file" id="scriptFile" accept="application/json,.json" style="display:none" onchange="uploadScriptFile(this.files)"></label>
    </div>
    <div id="agentInputRow">
      <textarea id="agentInput" placeholder="Ask the agent… (e.g. make clip 2 longer and angrier)" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();agentSend();}"></textarea>
      <button id="agentSend" onclick="agentSend()" title="Send">&#10148;</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);

/* ── MissingLink Google sign-in + free-trial gate ── */
let mlState={authed:false,member:false,email:null,used:0,remaining:25,
  free_limit:25,unlocked:false,reason:"no_session",
  signup:"https://www.missinglink.build/",
  login:"https://missinglink.build/notebook-signin"};
// Intercept every API response: any 401 login_required — from any feature,
// at any time (free trial used up, session lost) — reopens the gate on the
// right screen.
const _fetch=window.fetch.bind(window);
window.fetch=async function(u,o){
  const r=await _fetch(u,o);
  if(r.status===401&&typeof u==="string"&&u.indexOf("/api/")===0
     &&u.indexOf("/api/auth/")!==0){
    try{const j=await r.clone().json();
      if(j&&j.error==="login_required"){mlState.unlocked=false;_mlApply(j);}
    }catch(e){}
  }
  return r;
};
function mlOpenSignin(e){
  if(e)e.preventDefault();
  // Open the Google sign-in as a POPUP so its callback page can post the
  // token straight back to this window (no copy-paste). Google won't run
  // OAuth inside the Colab iframe, so a popup is required — but the code
  // returns automatically via the message listener below.
  const w=680,h=760;
  const y=(window.outerHeight-h)/2+ (window.screenY||0);
  const x=(window.outerWidth-w)/2 + (window.screenX||0);
  const pop=window.open(mlState.login,"missinglink_signin",
    "width="+w+",height="+h+",left="+Math.max(0,x)+",top="+Math.max(0,y));
  if(!pop){
    // Popup blocked — fall back to a normal tab + manual paste.
    window.open(mlState.login,"_blank");
    toast("Allow popups for one-click sign-in, or paste the code below.");
  }
  setTimeout(()=>{const k=$("mlKey");if(k)k.focus();},200);
  // Safety net for the Colab iframe: postMessage from the popup can be
  // blocked by the sandbox, so watch the popup. If it closes and we're
  // still locked, tell the user to paste the code it showed.
  if(pop){
    let ticks=0;
    const iv=setInterval(()=>{
      ticks++;
      if(mlState.unlocked){clearInterval(iv);try{pop.close();}catch(_){}return;}
      if(pop.closed){
        clearInterval(iv);
        if(!mlState.unlocked){
          const k=$("mlKey");if(k)k.focus();
          $("mlGateErr").textContent="Almost there \u2014 paste the code the sign-in window showed you into the box above, then press Unlock.";
          $("mlGateErr").style.color="var(--gold)";
          $("mlGateErr").style.display="block";
        }
      }
      if(ticks>600)clearInterval(iv);   // give up after ~5 min
    },500);
  }
}
// Receive the token the sign-in popup posts back, and log in automatically.
window.addEventListener("message",function(ev){
  const d=ev&&ev.data;
  if(!d)return;
  // Accept several shapes the sign-in page might use:
  //   {type:"missinglink-auth", token}   (primary)
  //   {token} or {key} or {access_token}  (looser)
  //   a bare token string
  let tok=null;
  if(typeof d==="string"&&d.length>16&&d.indexOf(" ")===-1)tok=d;
  else if(typeof d==="object"){
    if((d.type==="missinglink-auth"||d.type==="auth"||!d.type)&&(d.token||d.key||d.access_token))
      tok=d.token||d.key||d.access_token;
  }
  if(!tok)return;
  // Auto path: hand the token straight to mlLogin and DON'T leave it in
  // the visible field (a lingering value is what makes the browser offer
  // to "save password"). The field stays empty on the automatic flow.
  mlLogin(tok);
});
function _mlLinks(){
  if($("mlSignupA"))$("mlSignupA").href=mlState.signup;
  if($("mlSubBtn"))$("mlSubBtn").href=mlState.signup;
  if($("mlGoogleBtn"))$("mlGoogleBtn").href=mlState.login;
}
function _mlBadge(){
  const t=$("mlBadgeTxt");if(!t)return;
  if(mlState.member){t.textContent=mlState.email||"member";t.style.color="var(--gold)";}
  else if(mlState.authed){t.textContent=(mlState.remaining)+" free left";t.style.color="";}
  else{t.textContent="Sign in";t.style.color="";}
}
function _mlApply(s){
  if(typeof s.authed!=="undefined")mlState.authed=!!s.authed;
  if(typeof s.member!=="undefined")mlState.member=!!s.member;
  if(typeof s.email!=="undefined")mlState.email=s.email||mlState.email;
  if(typeof s.used!=="undefined")mlState.used=s.used;
  if(typeof s.remaining!=="undefined")mlState.remaining=s.remaining;
  if(typeof s.free_limit!=="undefined")mlState.free_limit=s.free_limit;
  if(s.signup_url)mlState.signup=s.signup_url;
  if(s.login_url)mlState.login=s.login_url;
  mlState.unlocked=!!(s.unlocked||mlState.member||
    (mlState.authed&&(mlState.remaining>0||mlState.remaining===-1)));
  _mlLinks();_mlBadge();
  if(mlState.unlocked){$("mlGate").style.display="none";}
  else{
    const limit=(s.reason==="free_limit_reached")||
      (mlState.authed&&mlState.remaining===0);
    showGate(limit?"free_limit_reached":"no_session");
  }
}
function showGate(reason){
  const limit=reason==="free_limit_reached";
  $("mlGateLogin").style.display=limit?"none":"block";
  $("mlGateLimit").style.display=limit?"block":"none";
  if($("mlLimitN"))$("mlLimitN").textContent=mlState.free_limit;
  if($("mlLimitEmail"))$("mlLimitEmail").textContent=mlState.email?" ("+mlState.email+")":"";
  _mlLinks();
  $("mlGate").style.display="flex";
}
function showGateIfLocked(){ if(!mlState.unlocked)showGate(
  (mlState.authed&&mlState.remaining===0)?"free_limit_reached":"no_session"); }
function mlSwitchAccount(){ mlState.authed=false;mlState.member=false;
  // Real logout: clears the server's cached session (incl. the on-disk
  // token that survives cell re-runs) so a new sign-in is required.
  try{fetch("/api/auth/logout",{method:"POST"});}catch(e){}
  $("mlGateLimit").style.display="none";$("mlGateLogin").style.display="block"; }
async function mlLogin(tok){
  // tok passed in = automatic postMessage flow (field stays empty).
  // No arg = manual paste, read from the box.
  const key=(typeof tok==="string"?tok:$("mlKey").value).trim();
  const b=$("mlLoginBtn");b.disabled=true;b.textContent="\u2026";
  $("mlGateErr").style.display="none";
  try{
    const j=await(await fetch("/api/auth/login",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({token:key})})).json();
    if(j.unlocked||j.member||(j.authed&&j.remaining!==0)){
      const k=$("mlKey");if(k)k.value="";   // clear so nothing is offered to save
      _mlApply(j);
      toast(j.member?("Welcome"+(j.email?", "+j.email:"")+" \u2014 unlimited access.")
        :("Signed in \u2014 "+j.remaining+" free renders left."));
      refreshLoras();restoreJobs();
    }else if(j.authed){ const k=$("mlKey");if(k)k.value=""; _mlApply(j); }
    else{$("mlGateErr").textContent=j.error||"that code didn't work \u2014 try again";
      $("mlGateErr").style.color="var(--red)";
      $("mlGateErr").style.display="block";}
  }catch(e){$("mlGateErr").textContent="Error: "+e;
    $("mlGateErr").style.color="var(--red)";$("mlGateErr").style.display="block";}
  b.disabled=false;b.textContent="Unlock";
}
async function mlRecheck(){
  try{const j=await(await fetch("/api/auth/status?refresh=1")).json();
    _mlApply(j);
    if(j.member){toast("Subscription active \u2014 welcome!");refreshLoras();restoreJobs();}
    else if(j.authed&&j.remaining>0){toast(j.remaining+" free renders left.");}
    else toast("Still no active subscription \u2014 finish at missinglink.build first.",true);
  }catch(e){toast("Error: "+e,true);}
}
async function mlBoot(){
  try{const j=await(await fetch("/api/auth/status")).json();_mlApply(j);}
  catch(e){}
}

let imgData=null,lastData=null,refData=null;
let _stagedEnd=null;   // staged LAST frame (copied into the clip at create)
let currentMode="i2v";
let queue=[];    // {id,prompt,thumb,status,progress,stage}
let history=[];  // {id,prompt,thumb,url,ts}

function toast(m,e){const t=$("toast");t.textContent=m;
  t.className="toast"+(e?" err":"");t.style.display="block";
  clearTimeout(t._t);t._t=setTimeout(()=>{t.style.display="none";},4500);}
// Toast with an inline Undo action (no native dialogs anywhere).
function toastUndo(m,onUndo){const t=$("toast");t.innerHTML="";
  t.className="toast";t.style.display="block";
  const s=document.createElement("span");s.textContent=m+"  ";
  const a=document.createElement("a");a.textContent="Undo";
  a.style.cssText="color:var(--gold);cursor:pointer;text-decoration:underline";
  a.onclick=()=>{try{onUndo();}catch(_e){}t.style.display="none";};
  t.appendChild(s);t.appendChild(a);
  clearTimeout(t._t);t._t=setTimeout(()=>{t.style.display="none";
    t.textContent="";},7000);}
// Snapshot / restore the full clip-timeline state for Undo.
function _snapClips(){return JSON.stringify({dialog:dialog,
  po:clipPromptOverrides,fo:clipFrameOverrides,si:clipStartImages,
  ei:clipEndImages,sp:clipStartPlaceholders,cv:clipControlVideos,
  ct:clipControlType,cs:clipControlStrength,ca:clipCopySrcAudio,
  cd:clipControlDur,csp:clipControlSplit,sg:clipStartGhosts,cc:_committedClips});}
function _restoreClips(snap){try{const s=JSON.parse(snap);
  dialog=s.dialog;clipPromptOverrides=s.po;clipFrameOverrides=s.fo;
  clipStartImages=s.si;clipEndImages=s.ei;clipStartPlaceholders=s.sp;
  clipControlVideos=s.cv;clipControlType=s.ct;clipControlStrength=s.cs;
  clipCopySrcAudio=s.ca;clipControlDur=s.cd;clipControlSplit=s.csp;
  clipStartGhosts=s.sg||{};_committedClips=s.cc;
  renderSpeakers();renderDialog();renderClipOverview();_renderC1Frames();
  _syncGenerateEnabled();}catch(_e){}}
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

/* slider <-> number box sync + duration estimate */
[["steps","stepsV"],["guid","guidV"],["shift","shiftV"],
 ["frames","framesV"],["fps","fpsV"],["segments","segmentsV"],
 ["ltxStrength","ltxStrengthV"],["ltxCrf","ltxCrfV"]].forEach(([s,v])=>{
  $(s).addEventListener("input",()=>{$(v).value=$(s).value;updateDur();});
  $(v).addEventListener("change",()=>{$(s).value=$(v).value;updateDur();});
});
function setSL(id,val){$(id).value=val;$(id+"V").value=val;}
function randSeed(){$("seed").value=Math.floor(Math.random()*2e9);}
function updateDur(){
  const clips=+$("segmentsV").value||1, frames=+$("framesV").value||81, fps=+$("fpsV").value||16;
  const total=(clips*frames-(clips-1))/fps;
  $("durHint").innerHTML="~"+total.toFixed(1)+" s &middot; "+clips+(clips>1?" clips":" clip");
}

/* speed modes */
async function setLoraScale(name,scale){
  try{await fetch("/api/loras/update",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({name:name,scale:scale})});}catch(e){}
  refreshLoras();
}
async function lightningMode(){
  setSL("steps",4);setSL("guid",1.0);setSL("shift",5.0);updateDur();
  $("segFast").classList.add("active");$("segQual").classList.remove("active");
  await setLoraScale("lightning_4step",1.0);
  toast("Lightning 4-step on \u2014 best for long videos.");
}
async function qualityMode(){
  setSL("steps",30);setSL("guid",5.0);setSL("shift",3.0);updateDur();
  $("segQual").classList.add("active");$("segFast").classList.remove("active");
  await setLoraScale("lightning_4step",0.0);
  toast("Quality mode on.");
}
$("profile").addEventListener("change",e=>{setSL("shift",e.target.value==="720P"?"5.0":"3.0");});

/* advanced toggle */
function toggleAdv(){
  $("advBody").classList.toggle("open");
  $("advCaret").innerHTML=$("advBody").classList.contains("open")?"\u25BE":"\u25B8";
}

/* generation mode switch */
let currentEngine="wan";
// Base-model options for the Civitai LoRA search, per engine. The value
// is the exact Civitai baseModel string (verified); "all"/"wan" are
// convenience filters the worker expands.
const CIV_BASE_OPTS={
  ltx:[
    ["ltx","All LTX"],
    ["LTXV 2.3","LTX 2.3 (LTXV 2.3)"],
    ["LTXV","LTX 2 (LTXV)"],
    ["all","All base models"],
  ],
  wan:[
    ["wan","All Wan 2.1"],
    ["Wan Video 14B i2v 480p","Wan 14B i2v 480p"],
    ["Wan Video 14B i2v 720p","Wan 14B i2v 720p"],
    ["Wan Video 14B t2v","Wan 14B t2v"],
    ["Wan Video 1.3B t2v","Wan 1.3B t2v"],
    ["Wan Video","Wan Video (generic)"],
    ["all","All base models"],
  ],
  wan22:[
    ["Wan Video 2.2 I2V-A14B","Wan 2.2 I2V-A14B"],
    ["Wan Video 2.2 T2V-A14B","Wan 2.2 T2V-A14B"],
    ["Wan Video","Wan Video (generic)"],
    ["all","All base models"],
  ],
};
function _populateCivBase(engine){
  const cb=$("civSearchBase");if(!cb)return;
  const opts=CIV_BASE_OPTS[engine]||CIV_BASE_OPTS.wan;
  cb.innerHTML=opts.map((o,i)=>
    "<option value='"+o[0].replace(/'/g,"&#39;")+"'"+(i===0?" selected":"")
    +">"+o[1]+"</option>").join("");
}

function setEngine(e){
  currentEngine=e;
  try{refreshLoras();}catch(_e){}
  const ltx=e==="ltx";
  const wan22=e==="wan22";
  const wan=e==="wan";
  $("engWan").classList.toggle("active",wan);
  $("engWan22").classList.toggle("active",wan22);
  $("engLtx").classList.toggle("active",ltx);
  $("wan22Opts").style.display=wan22?"block":"none";
  // Rebuild the LoRA-search base-model filter for the active engine so it
  // only offers bases that actually apply (LTX bases on LTX, Wan on Wan).
  _populateCivBase(e);
  if(ltx||wan22)setMode("i2v");     // LTX + Wan 2.2 here are i2v only
  // Wan-only sections: task modes, Lightning/Quality, clip chaining.
  // Wan 2.2 is i2v-only (no flf2v/vace) and its steps are Lightning-fixed,
  // so it hides the task-mode + speed sections but keeps length/LoRAs.
  const showWan=!ltx;                 // both Wan engines show Wan UI
  $("modeSec").style.display=(wan)?"block":"none";   // 2.1 only
  $("speedSec").style.display=(wan)?"block":"none";  // 2.1 only (2.2=Lightning)
  $("lengthSec").style.display=showWan?"block":"none";
  $("ltxResHint").style.display=ltx?"block":"none";
  // Dialogue builder is an LTX feature; Wan (both) uses plain prompt.
  $("dlgSec").style.display=ltx?"flex":"none";
  // On LTX the prompt IS the base scene for the clips below — label it so
  // and show the role hint, so it doesn't look like a duplicate.
  $("promptLbl").textContent=ltx?"Clip 1 \u00b7 scene":"Prompt";
  $("promptRoleHint").style.display=ltx?"block":"none";
  const atb=$("addToTimelineBtn"),ath=$("addToTimelineHint");
  if(atb)atb.style.display=ltx?"block":"none";
  if(ath)ath.style.display=ltx?"block":"none";
  const atw=$("appendTlWrap");if(atw)atw.style.display=ltx?"flex":"none";
  if(ltx){_updateSceneRoleHint();_syncGenerateEnabled();}
  else{const gb=$("genBtn");if(gb){gb.disabled=false;gb.style.opacity="";}}
  // The starting-frame section is Clip 1's first/last frame under LTX.
  const fl=$("firstLabel");
  if(fl)fl.innerHTML=ltx
    ? "<span class='icon'>\u25a3</span> Clip 1 frames"
    : "<span class='icon'>\u25a3</span> Starting frame <span class='c'>required</span>";
  // LTX shows the first+last frame pair; Wan keeps the single dropzone.
  const c1=$("clip1Frames"),dz=$("drop");
  if(c1)c1.style.display=ltx?"block":"none";
  if(dz)dz.style.display=ltx?"none":"block";
  if(ltx)_renderC1Frames();
  $("neg").style.display=ltx?"none":"block";
  $("negLabel").style.display=ltx?"none":"flex";
  if(ltx){renderSpeakers();renderDialog();}
  else _updateGenBtnLabel(1);   // Wan/Wan2.2: plain label
  // Advanced settings: Wan knobs vs LTX knobs.
  $("advWanOnly").style.display=ltx?"none":"block";
  $("advWanOnly2").style.display=ltx?"none":"block";
  $("advLtx").style.display=ltx?"block":"none";
  $("frameRuleHint").textContent=ltx
    ?"Frames snap to LTX's 8k+1 rule; 121 @ 24 fps \u2248 5 s. ~20 s (481 frames) is LTX-2.3's trained max \u2014 longer is allowed but drifts and may run out of memory."
    :"Frames snap to Wan's 4n+1 rule; 81 is the trained length. Flow shift ~3 for 480P, ~5 for 720P.";
  const f=$("frames");
  if(ltx){f.min=33;f.max=1449;f.step=8;setSL("frames",121);setSL("fps",24);}
  else{f.min=17;f.max=81;f.step=4;setSL("frames",81);setSL("fps",16);}
  updateDur();
}
function setMode(m){
  currentMode=m;
  $("modeI2V").classList.toggle("active",m==="i2v");
  $("modeFLF").classList.toggle("active",m==="flf2v");
  $("modeVACE").classList.toggle("active",m==="vace");
  $("lastSec").style.display=m==="flf2v"?"block":"none";
  $("refSec").style.display=m==="vace"?"block":"none";
  $("lengthSec").style.display=m==="i2v"?"block":"none";
  $("profileWrap").style.display=m==="i2v"?"block":"none";
  $("flfNote").style.display=m==="flf2v"?"block":"none";
  $("vaceSizeWrap").style.display=m==="vace"?"block":"none";
  const fl=$("firstLabel");
  if(m==="flf2v"){fl.innerHTML="<span class='icon'>&#9635;</span> First frame <span class='c'>required</span>";
    $("modeHint").textContent="Give a first and last frame; the model fills the motion between them. 720P, single clip.";
    setSL("shift",5.0);}
  else if(m==="vace"){fl.innerHTML="<span class='icon'>&#9635;</span> Start frame <span class='c'>optional</span>";
    $("modeHint").textContent="Reference-to-video: keep a subject/character consistent. Add a Start frame for extra conditioning. Single clip.";}
  else{fl.innerHTML="<span class='icon'>&#9635;</span> Starting frame <span class='c'>required</span>";
    $("modeHint").textContent="Animate a single still image into a clip. Supports long video via clip chaining.";}
}

/* starting / first / start image */
$("file").addEventListener("change",e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=()=>{imgData=r.result;
    $("drop").className="dropzone has";
    $("drop").innerHTML="<img src='"+r.result+"'>";
    if(currentEngine==="ltx"){clipStartImages[1]=r.result;_renderC1Frames();renderClipOverview();}};
  r.readAsDataURL(f);
});
$("fileLast").addEventListener("change",e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=()=>{lastData=r.result;
    $("dropLast").className="dropzone has";
    $("dropLast").innerHTML="<img src='"+r.result+"'>";};
  r.readAsDataURL(f);
});
$("fileRef").addEventListener("change",e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=()=>{refData=r.result;
    $("dropRef").className="dropzone has";
    $("dropRef").innerHTML="<img src='"+r.result+"'>";};
  r.readAsDataURL(f);
});

/* ── LoRAs ── */
async function refreshLoras(){
  const box=$("loraList");let j;
  try{j=await(await fetch("/api/loras")).json();}
  catch(e){box.innerHTML="<div class='hintline'>Could not load LoRAs.</div>";return;}
  window._loraUrls=new Set((j.loras||[]).map(L=>L.url).filter(Boolean));
  box.innerHTML="";
  // Only show LoRAs for the ACTIVE engine — a Wan LoRA (e.g. the Lightning
  // 4-step speed LoRA) does nothing on LTX and vice versa.
  const wantLtx=(typeof currentEngine!=="undefined"&&currentEngine==="ltx");
  const all=j.loras||[];
  const vis=all.filter(L=>wantLtx?(L.engine==="ltx"):(L.engine!=="ltx"));
  const hidden=all.length-vis.length;
  if(!vis.length){
    box.innerHTML="<div class='hintline' style='margin-top:10px'>No LoRAs added"+
      (hidden?" for this engine ("+hidden+" hidden \u2014 they belong to "+(wantLtx?"Wan":"LTX")+")":"")+".</div>";
    return;}
  vis.forEach(L=>{
    const d=document.createElement("div");d.className="lora-card";
    d.innerHTML="<div class='lora-card-top'><span class='lora-card-name'>"+esc(L.name)+
      (L.attached?"":" (not attached)")+"</span>"+
      "<button class='lora-x'>&#10005;</button></div>"+
      "<div class='slider-row' style='margin:8px 0 0'><span class='sl'>strength</span>"+
      "<input type='range' min='0' max='2' step='0.05' value='"+L.scale+"'>"+
      "<input class='sv' value='"+(+L.scale).toFixed(2)+"'></div>";
    const sl=d.querySelector("input[type=range]"),nm=d.querySelector(".sv"),xb=d.querySelector(".lora-x");
    async function push(v){await fetch("/api/loras/update",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name:L.name,scale:parseFloat(v)})});}
    sl.addEventListener("input",()=>nm.value=(+sl.value).toFixed(2));
    sl.addEventListener("change",()=>push(sl.value));
    nm.addEventListener("change",()=>{sl.value=nm.value;push(nm.value);});
    xb.addEventListener("click",async()=>{
      await fetch("/api/loras/remove",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({name:L.name})});refreshLoras();});
    box.appendChild(d);
  });
  if(hidden>0){
    const h=document.createElement("div");h.className="hintline";
    h.style.marginTop="8px";
    h.textContent=hidden+" LoRA"+(hidden>1?"s":"")+" hidden \u2014 belong"+(hidden>1?"":"s")+" to "+(wantLtx?"Wan":"LTX")+".";
    box.appendChild(h);
  }
}
async function addLora(){
  const url=$("loraUrl").value.trim();
  if(!url){toast("Paste a LoRA URL first.",true);return;}
  const b=$("addLoraBtn");b.disabled=true;b.textContent="Downloading...";
  try{
    const r=await fetch("/api/loras/add",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({url:url,name:$("loraName").value.trim(),
        scale:parseFloat($("loraScale").value)||1.0,engine:currentEngine})});
    const j=await r.json();
    if(j.ok){toast("LoRA added.");$("loraUrl").value="";$("loraName").value="";}
    else toast("LoRA failed: "+(j.error||"unknown"),true);
  }catch(e){toast("Error: "+e,true);}
  b.disabled=false;b.textContent="+ Add LoRA";refreshLoras();
}

/* ── 🎬 Auto Next Scene (agentic image for the next clip) ── */
let nextCfg={context:"",instructions:"",quality:"medium",shot:"auto"};
let _nextResult=null;   // {image, verdict, intent, works, corrected}
function openNextCfg(){
  $("nextCfgContext").value=nextCfg.context||"";
  $("nextCfgInstr").value=nextCfg.instructions||"";
  $("nextCfgQuality").value=nextCfg.quality||"medium";
  if($("nextCfgShot"))$("nextCfgShot").value=nextCfg.shot||"auto";
  $("nextCfgContext").oninput=()=>nextCfg.context=$("nextCfgContext").value;
  $("nextCfgInstr").oninput=()=>nextCfg.instructions=$("nextCfgInstr").value;
  $("nextCfgQuality").onchange=()=>nextCfg.quality=$("nextCfgQuality").value;
  if($("nextCfgShot"))$("nextCfgShot").onchange=()=>nextCfg.shot=$("nextCfgShot").value;
  $("nextCfgModal").style.display="flex";
}
function closeNextCfg(){$("nextCfgModal").style.display="none";}
function clearNextCfg(){nextCfg={context:"",instructions:"",quality:nextCfg.quality,shot:nextCfg.shot};
  $("nextCfgContext").value="";$("nextCfgInstr").value="";}
function closeNextReview(){$("nextReviewModal").style.display="none";}
// Grab the last visible frame of the stage video as a data URL.
function _stageLastFrame(){
  return new Promise((resolve,reject)=>{
    const v=$("viewer").querySelector("video");
    if(!v){reject("No clip on the stage yet.");return;}
    const grab=()=>{
      try{
        const c=document.createElement("canvas");
        c.width=v.videoWidth||1024;c.height=v.videoHeight||1024;
        c.getContext("2d").drawImage(v,0,0,c.width,c.height);
        resolve(c.toDataURL("image/png"));
      }catch(e){reject("Couldn't read the video frame ("+e+").");}
    };
    // Seek to the very end so we capture the true last frame.
    try{
      const wasPaused=v.paused;
      const onSeek=()=>{v.removeEventListener("seeked",onSeek);grab();
        if(!wasPaused)v.play().catch(()=>{});};
      if(isFinite(v.duration)&&v.duration>0){
        v.addEventListener("seeked",onSeek);
        v.pause();v.currentTime=Math.max(0,v.duration-0.05);
      }else{grab();}
    }catch(e){grab();}
  });
}
async function autoNextScene(regen){
  try{
    const frame=await _stageLastFrame();
    if(regen)closeNextReview();
    _pendingNewClip=null;   // stage flow appends a new clip on accept
    _openNextCompose(frame,(nextCfg.context||"").trim(),"Next scene from stage");
  }catch(e){toast(typeof e==="string"?e:("Error: "+e),true);}
}
function _showNextReview(j){
  $("nextReviewImg").src=j.image;
  const v=$("nextReviewVerdict");
  v.className="next-verdict "+(j.works?"ok":"warn");
  v.innerHTML=(j.works?"\u2713 ":"\u26a0 ")
    +esc(j.verdict||(j.works?"This should work as the next scene.":"This may not fit the previous clip."))
    +(j.corrected?" <span style='opacity:.7'>(auto-corrected)</span>":"");
  $("nextReviewIntent").textContent=j.intent?("Intent: "+j.intent):"";
  $("nextReviewModal").style.display="flex";
}
// Edit the currently reviewed image with a text instruction (gpt-image-2
// edit). Swaps the result in so the user can accept or keep editing.
async function nextEditImage(){
  if(!_nextResult||!_nextResult.image)return;
  const instr=($("nextEditInstr").value||"").trim();
  if(!instr){toast("Describe the change first.",true);return;}
  const btn=$("nextEditBtn");btn.disabled=true;btn.innerHTML="\u2026";
  toast("Applying your edit\u2026");
  try{
    const r=await fetch("/api/editimage",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:_nextResult.image,instruction:instr,
        quality:(nextCfg.quality||"medium")})});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Edit failed.",true);}
    else{
      _nextResult.image=j.image;
      _nextResult.corrected=true;
      if(j.tokens!==undefined&&j.tokens!==null)_applyTokBadge(j.tokens,j.tokens_per_gen||100,false);
      else if(j.own_key)_applyTokBadge(null,null,true);
      $("nextReviewImg").src=j.image;
      $("nextEditInstr").value="";
      toast("Edit applied \u2014 accept it or keep editing.");
    }
  }catch(e){toast("Error: "+e,true);}
  btn.disabled=false;btn.innerHTML="\u270e Edit";
}
// Accept the generated image. On LTX it becomes a clip's start frame:
// either the specific clip queued by "New scene" (_pendingNewClip), or a
// freshly appended clip when triggered from the stage's Auto Next Scene.
// Where an accepted compose/review image lands. Default target = the
// "new scene" flow (create/open a clip). When opened from a clip's start
// or end slot, _composeTarget routes the image straight into that slot.
let _composeTarget=null;   // null | {kind:"clip_start"|"clip_end", clip:N}
function nextAccept(){
  if(!_nextResult||!_nextResult.image)return;
  // Targeted: write into a specific clip's start/end slot.
  if(_composeTarget){
    const t=_composeTarget;
    if(t.kind==="clip_end"){clipEndImages[t.clip]=_nextResult.image;}
    else{clipStartImages[t.clip]=_nextResult.image;delete clipStartPlaceholders[t.clip];
      if(t.clip===1)_setStartImage(_nextResult.image);}
    _addImageToHistory(_nextResult.image,"Clip "+t.clip+" "+(t.kind==="clip_end"?"end frame":"start image"));
    _composeTarget=null;
    closeNextReview();
    if(_clipModalC===t.clip)_renderClipModalImgs();
    renderDialog();
    toast("Clip "+t.clip+"'s "+(t.kind==="clip_end"?"end frame":"start image")+" updated.");
    return;
  }
  _addImageToHistory(_nextResult.image,"Next scene");
  if(currentEngine==="ltx"){
    if(!speakers.length)addSpeaker();
    const nc=_pendingNewClip || (dialog.reduce((m,l)=>Math.max(m,l.clip||1),0)+1);
    clipStartImages[nc]=_nextResult.image;
    // seed the new clip's scene with the intent so it's not blank
    if(_nextResult.intent)clipPromptOverrides[nc]=_nextResult.intent;
    // only add a dialogue placeholder if this clip doesn't exist yet
    if(!dialog.some(l=>(l.clip||1)===nc))
      dialog.push({speaker:speakers[0].name,text:"",clip:nc});
    _pendingNewClip=null;
    renderDialog();
    closeNextReview();
    toast("Clip "+nc+" starts from the new image \u2014 add its dialogue, then Generate.");
    openClipModal(nc);
  }else{
    _setStartImage(_nextResult.image);
    closeNextReview();
    toast("Next-scene image set as your starting frame. Hit Generate for the next clip.");
  }
}
// Let the user swap in their own image instead of the generated one.
function nextReplaceManual(){
  const inp=$("nextManualFile");
  inp.onchange=()=>{
    const f=inp.files&&inp.files[0];if(!f)return;
    const rd=new FileReader();
    rd.onload=()=>{
      // Targeted at a clip slot (from the clip editor's generate flow).
      if(_composeTarget){
        const t=_composeTarget;
        if(t.kind==="clip_end"){clipEndImages[t.clip]=rd.result;}
        else{clipStartImages[t.clip]=rd.result;delete clipStartPlaceholders[t.clip];
          if(t.clip===1)_setStartImage(rd.result);}
        _composeTarget=null;
        if(_clipModalC===t.clip)_renderClipModalImgs();
        renderDialog();closeNextReview();
        toast("Clip "+t.clip+"'s "+(t.kind==="clip_end"?"end frame":"start image")+" set from your image.");
        return;
      }
      if(currentEngine==="ltx"){
        if(!speakers.length)addSpeaker();
        const nc=_pendingNewClip || (dialog.reduce((m,l)=>Math.max(m,l.clip||1),0)+1);
        clipStartImages[nc]=rd.result;delete clipStartPlaceholders[nc];delete clipStartGhosts[nc];
        if(!dialog.some(l=>(l.clip||1)===nc))
          dialog.push({speaker:speakers[0].name,text:"",clip:nc});
        _pendingNewClip=null;
        renderDialog();closeNextReview();
        toast("Clip "+nc+" starts from your image \u2014 add its dialogue, then Generate.");
        openClipModal(nc);
      }else{
        _setStartImage(rd.result);closeNextReview();
        toast("Your image is set as the starting frame.");
      }
    };
    rd.readAsDataURL(f);inp.value="";
  };
  inp.click();
}
// Route an image into the start-frame slot (imgData + the drop preview),
// matching how a manual upload sets it.
function _setStartImage(dataUrl){
  imgData=dataUrl;
  const d=$("drop");
  if(d){d.className="dropzone has";d.innerHTML="<img src='"+dataUrl+"'>";}
  if(currentEngine==="ltx"){_renderC1Frames();renderClipOverview();}  // clip 1 reflects it
}

// ── Editor tab (timeline editor) ──
// The full drag/trim/transitions/audio/export editor is built separately.
// These handlers open the tab and keep its controls inert until then.
// ══════════════════════════════════════════════════════════════════════
//  TIMELINE EDITOR ENGINE
//  Model: edState.video = ordered clips [{id, url, thumb, name, dur,
//  in, out, fadeIn, xfade}]; edState.audio = [{name, muted, clips:[...]}].
//  Each clip stores its source duration and in/out trim points. Preview
//  plays clips sequentially in a single <video>. Export ships the edit
//  decision list to the server, which stitches with ffmpeg.
// ══════════════════════════════════════════════════════════════════════
let edState={video:[],audio:[],pxPerSec:90,sel:null};
let _edMediaDur={};   // cache: url -> duration seconds

function openEditor(){
  const t=$("editorTab");if(t)t.style.display="block";
  const s=$("tabStudio"),e=$("tabEditor");
  if(s)s.classList.remove("active");if(e)e.classList.add("active");
  const fab=$("agentFab");if(fab)fab.style.display="none";   // don't float over editor
  edRenderPool();edRenderTimeline();
}
function closeEditor(){
  const t=$("editorTab");if(t)t.style.display="none";
  const s=$("tabStudio"),e=$("tabEditor");
  if(e)e.classList.remove("active");if(s)s.classList.add("active");
  const fab=$("agentFab");if(fab&&$("agentPanel").style.display==="none")fab.style.display="";
  edStop();
}
// Probe a media file's duration (cached).
function _edProbe(url){
  return new Promise(res=>{
    if(_edMediaDur[url]!=null)return res(_edMediaDur[url]);
    const v=document.createElement("video");v.preload="metadata";
    v.onloadedmetadata=()=>{_edMediaDur[url]=v.duration||5;res(v.duration||5);};
    v.onerror=()=>{_edMediaDur[url]=5;res(5);};
    v.src=url;
  });
}
// ── Pool (rendered clips + generated audio available to add) ──
function edRenderPool(){
  const box=$("edPoolItems");if(!box)return;box.innerHTML="";
  const vids=history.filter(h=>h.kind!=="image"&&h.url);
  if(!vids.length){box.innerHTML="<div style='color:var(--text-muted);font-family:var(--font-mono);font-size:10px'>No rendered clips yet — generate a video first, then arrange it here.</div>";return;}
  vids.forEach(h=>{
    const d=document.createElement("div");d.className="ed-pool-item";
    d.innerHTML="<div class='ed-pi-thumb'><img src='"+(h.thumb||"")+"'></div>"
      +"<div class='ed-pi-name'>"+_esc((h.prompt||"clip").slice(0,40))+"</div>"
      +"<button class='ed-pi-add' title='Add to timeline'>+</button>";
    d.querySelector(".ed-pi-add").onclick=(e)=>{e.stopPropagation();edAddClip(h);};
    d.onclick=()=>edAddClip(h);
    box.appendChild(d);
  });
}
async function edAddClip(h){
  const dur=await _edProbe(h.url);
  edState.video.push({id:"ec_"+Date.now().toString(36)+Math.random().toString(36).slice(2,5),
    url:h.url,thumb:h.thumb||"",name:(h.prompt||"clip").slice(0,30),
    dur:dur,in:0,out:dur,fadeIn:0,xfade:0});
  edRenderTimeline();
  toast("Added to timeline.");
}
function edAddAudioTrack(){
  edState.audio.push({name:"Audio "+(edState.audio.length+1),muted:false,clips:[]});
  edRenderTimeline();
  // Prompt to attach a file immediately.
  _edAudioUploadFor(edState.audio.length-1);
}
function _edAudioUploadFor(ti){
  const inp=document.createElement("input");inp.type="file";inp.accept="audio/*,video/*";
  inp.onchange=async()=>{
    const f=inp.files&&inp.files[0];if(!f)return;
    const url=URL.createObjectURL(f);const dur=await _edProbe(url);
    edState.audio[ti].clips.push({id:"ea_"+Date.now().toString(36),url:url,
      name:f.name.slice(0,24),dur:dur,in:0,out:dur,start:0,_file:f});
    edRenderTimeline();
  };
  inp.click();
}
// ── Timeline geometry ──
function _edClipLen(c){return Math.max(0.1,(c.out-c.in));}
function edTotalDur(){
  let t=0;edState.video.forEach((c,i)=>{t+=_edClipLen(c)-(i>0?(c.xfade||0):0);});
  return Math.max(t,0);
}
function edSetZoom(v){edState.pxPerSec=+v;edRenderTimeline();}
// ── Render the whole timeline ──
function edRenderTimeline(){
  const pps=edState.pxPerSec;
  const total=Math.max(edTotalDur(),6);
  const wrapW=total*pps+80;
  // Ruler
  const ruler=$("edRuler");ruler.innerHTML="";ruler.style.width=wrapW+"px";
  const step=(pps<40?5:pps<100?2:1);
  for(let s=0;s<=total+1;s+=step){
    const t=document.createElement("div");t.className="ed-tick";t.style.left=(s*pps)+"px";
    t.textContent=_fmtT(s);ruler.appendChild(t);
  }
  const tracks=$("edTracks");tracks.innerHTML="";tracks.style.width=wrapW+"px";
  // Video track
  const vt=document.createElement("div");vt.className="ed-track";
  vt.innerHTML="<div class='ed-track-label'>&#127909; Video</div>";
  let x=0;
  edState.video.forEach((c,i)=>{
    const len=_edClipLen(c);const xf=(i>0?(c.xfade||0):0);
    x-=xf;   // overlap for crossfade
    const el=document.createElement("div");el.className="ed-clip"+(edState.sel===c.id?" sel":"");
    el.style.left=(x*pps)+"px";el.style.width=(len*pps)+"px";
    el.innerHTML=(c.thumb?"<img src='"+c.thumb+"'>":"")
      +"<span class='ed-clip-name'>"+_esc(c.name)+"</span>"
      +"<div class='ed-trim l'></div><div class='ed-trim r'></div>"
      +(c.fadeIn?"<div class='ed-fade' style='width:"+Math.min(len,c.fadeIn)*pps+"px'></div>":"")
      +(xf?"<div class='ed-xfade' style='width:"+xf*pps+"px'></div>":"");
    _edWireClip(el,c,i);
    vt.appendChild(el);
    x+=len;
  });
  tracks.appendChild(vt);
  // Audio tracks
  edState.audio.forEach((tr,ti)=>{
    const at=document.createElement("div");at.className="ed-track ed-audio";
    at.innerHTML="<div class='ed-track-label'>&#127925; "+_esc(tr.name)
      +" <button class='ed-mute"+(tr.muted?" on":"")+"' title='Mute'>"+(tr.muted?"&#128263;":"&#128266;")+"</button>"
      +"<button class='ed-rmtrack' title='Remove track'>&#10005;</button></div>";
    at.querySelector(".ed-mute").onclick=()=>{tr.muted=!tr.muted;edRenderTimeline();};
    at.querySelector(".ed-rmtrack").onclick=()=>{edState.audio.splice(ti,1);edRenderTimeline();};
    tr.clips.forEach(ac=>{
      const len=Math.max(0.1,ac.out-ac.in);
      const el=document.createElement("div");el.className="ed-clip";
      el.style.left=(ac.start*pps)+"px";el.style.width=(len*pps)+"px";
      el.innerHTML="<span class='ed-clip-name'>"+_esc(ac.name)+"</span>";
      _edWireAudioClip(el,ac,tr);
      at.appendChild(el);
    });
    // click empty area of the track label to add a file
    at.ondblclick=()=>_edAudioUploadFor(ti);
    tracks.appendChild(at);
  });
  $("edTime").textContent=_fmtT(0)+" / "+_fmtT(edTotalDur());
  // preview empty state
  $("edPreviewEmpty").style.display=edState.video.length?"none":"block";
}
function _fmtT(s){s=Math.max(0,s|0);return (s/60|0)+":"+String(s%60).padStart(2,"0");}
// ── Clip interactions: drag to reorder, trim edges, select ──
function _edWireClip(el,c,idx){
  const trimL=el.querySelector(".ed-trim.l"),trimR=el.querySelector(".ed-trim.r");
  let mode=null,sx=0,orig=0;
  const down=(e,m)=>{mode=m;sx=e.clientX;
    orig=(m==="inp")?c.in:(m==="out")?c.out:idx;
    e.preventDefault();e.stopPropagation();
    document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);};
  const move=(e)=>{
    const dx=(e.clientX-sx)/edState.pxPerSec;
    if(mode==="inp"){c.in=Math.max(0,Math.min(c.out-0.2,orig+dx));edRenderTimeline();}
    else if(mode==="out"){c.out=Math.min(c.dur,Math.max(c.in+0.2,orig+dx));edRenderTimeline();}
  };
  const up=()=>{document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);mode=null;};
  trimL.addEventListener("mousedown",e=>down(e,"inp"));
  trimR.addEventListener("mousedown",e=>down(e,"out"));
  // HTML5 drag to reorder
  el.draggable=true;
  el.addEventListener("dragstart",e=>{e.dataTransfer.setData("text/plain",String(idx));edState.sel=c.id;});
  el.addEventListener("dragover",e=>e.preventDefault());
  el.addEventListener("drop",e=>{e.preventDefault();
    const from=+e.dataTransfer.getData("text/plain");if(from===idx||isNaN(from))return;
    const moved=edState.video.splice(from,1)[0];edState.video.splice(idx,0,moved);edRenderTimeline();});
  el.addEventListener("click",e=>{if(e.target.closest(".ed-trim"))return;
    edState.sel=c.id;edShowClipMenu(c,idx);edRenderTimeline();});
}
function _edWireAudioClip(el,ac,tr){
  let sx=0,orig=0,drag=false;
  el.addEventListener("mousedown",e=>{drag=true;sx=e.clientX;orig=ac.start;
    document.addEventListener("mousemove",mv);document.addEventListener("mouseup",up);});
  const mv=(e)=>{if(!drag)return;ac.start=Math.max(0,orig+(e.clientX-sx)/edState.pxPerSec);edRenderTimeline();};
  const up=()=>{drag=false;document.removeEventListener("mousemove",mv);document.removeEventListener("mouseup",up);};
  el.addEventListener("dblclick",()=>{tr.clips.splice(tr.clips.indexOf(ac),1);edRenderTimeline();});
}
// Small inline menu for a selected video clip (transition, fade, remove).
function edShowClipMenu(c,idx){
  const cur=c.xfade||0, fi=c.fadeIn||0;
  const opts=["Remove clip","Fade in: "+(fi?fi+"s":"off"),
    idx>0?("Crossfade w/ prev: "+(cur?cur+"s":"off")):"(first clip — no crossfade)"];
  // Cycle values on click via prompt-free toggles.
  const pick=prompt("Clip \""+c.name+"\"\n1 = remove\n2 = fade-in "+(fi?"off":"0.5s")
    +"\n3 = crossfade w/ prev "+(cur?"off":"0.5s")+"\nEnter 1, 2, or 3:","");
  if(pick==="1"){edState.video.splice(idx,1);}
  else if(pick==="2"){c.fadeIn=fi?0:0.5;}
  else if(pick==="3"&&idx>0){c.xfade=cur?0:0.5;}
  edRenderTimeline();
}
// ── Sequential preview playback ──
let _edPlay={on:false,i:0,t0:0};
function edPlayPause(){
  if(!edState.video.length){toast("Add clips to the timeline first.");return;}
  const v=$("edPreview");
  if(_edPlay.on){v.pause();_edPlay.on=false;$("edPlayBtn").innerHTML="&#9654;";return;}
  _edPlay.on=true;$("edPlayBtn").innerHTML="&#10073;&#10073;";
  if(v.style.display==="none"||!v.src)_edPlayFrom(0);else v.play();
}
function _edPlayFrom(i){
  const v=$("edPreview");const c=edState.video[i];
  if(!c){edStop();return;}
  _edPlay.i=i;$("edPreviewEmpty").style.display="none";v.style.display="block";
  v.src=c.url;v.currentTime=c.in||0;
  v.onloadeddata=()=>{v.currentTime=c.in||0;v.play();};
  v.ontimeupdate=()=>{
    if(v.currentTime>=(c.out||c.dur)-0.02){
      v.ontimeupdate=null;
      _edPlay.on&&_edPlayFrom(i+1);
    }
  };
}
function edStop(){
  const v=$("edPreview");if(v){v.pause();v.ontimeupdate=null;}
  _edPlay.on=false;_edPlay.i=0;const b=$("edPlayBtn");if(b)b.innerHTML="&#9654;";
}
// ── Export: send the edit decision list to the server for ffmpeg stitch ──
async function edExport(){
  if(!edState.video.length){toast("Nothing to export — add clips first.",true);return;}
  const btn=$("edExportBtn");btn.disabled=true;const orig=btn.innerHTML;btn.innerHTML="Exporting\u2026";
  try{
    // Upload any local audio files as data URLs; video clips are already
    // server-rendered URLs the backend can fetch, but to be safe we send
    // the clip URLs and trim/transition data as an EDL.
    const audioTracks=[];
    for(const tr of edState.audio){
      const clips=[];
      for(const ac of tr.clips){
        let data=ac.url;
        if(ac._file){data=await _fileToDataUrl(ac._file);}
        clips.push({data:data,start:ac.start,in:ac.in,out:ac.out});
      }
      audioTracks.push({muted:tr.muted,clips:clips});
    }
    const edl={
      video:edState.video.map((c,i)=>({url:c.url,in:c.in,out:c.out,
        fadeIn:c.fadeIn||0,xfade:i>0?(c.xfade||0):0})),
      audio:audioTracks
    };
    const r=await fetch("/api/editor/export",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(edl)});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Export failed.",true);}
    else{
      // Result is a URL to the stitched mp4 — add to history + offer download.
      const h={id:"exp_"+Date.now().toString(36),kind:"video",prompt:"Timeline export",
        thumb:edState.video[0].thumb,url:j.url,ts:Date.now()};
      history.unshift(h);renderHistory();
      toast("Exported! Added to history & gallery.");
      const a=document.createElement("a");a.href=j.url;a.download="timeline_export.mp4";a.click();
    }
  }catch(e){toast("Export error: "+e,true);}
  btn.disabled=false;btn.innerHTML=orig;
}
function _fileToDataUrl(f){return new Promise((res,rej)=>{const r=new FileReader();
  r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(f);});}

// ── Clip 1 first/last frame on the main page ──
// First frame = the uploaded/opening image (imgData + clipStartImages[1]).
// Last frame = clipEndImages[1]. Both upload or generate, mirroring the
// per-clip editor slots, so Clip 1 is configured from the main inputs.
function _renderC1Frames(){
  const s=imgData||null, e=_stagedEnd||null;
  const v=clipControlVideos[1]||null;
  const ss=$("c1StartSlot"),es=$("c1EndSlot"),vs=$("c1V2VSlot");
  if(ss){ss.innerHTML=s?"<img src='"+s+"'>":"<span class='c1plus'>+</span>";
    ss.onclick=s?()=>_showImageFull(s,"clip1-first"):()=>$("c1StartFile").click();}
  if(es){es.innerHTML=e?"<img src='"+e+"'>":"<span class='c1plus'>+</span>";
    es.onclick=e?()=>_showImageFull(e,"clip1-last"):()=>$("c1EndFile").click();}
  if(vs){vs.innerHTML=v?"<video src='"+v+"' muted></video>":"<span class='c1plus'>\ud83c\udfa5</span>";
    vs.onclick=v?()=>{openClipModal(1);}:()=>$("c1V2VFile").click();}
  const sc=$("c1StartClear");if(sc)sc.style.display=s?"flex":"none";
  const ec=$("c1EndClear");if(ec)ec.style.display=e?"flex":"none";
  const vc=$("c1V2VClear");if(vc)vc.style.display=v?"flex":"none";
  const se=$("c1StartEdit");if(se)se.style.display=s?"inline-block":"none";
  const sei=$("c1StartEditIc");if(sei)sei.style.display=s?"flex":"none";
  const eei=$("c1EndEditIc");if(eei)eei.style.display=e?"flex":"none";
  const ee=$("c1EndEdit");if(ee)ee.style.display=e?"inline-block":"none";
  const ve=$("c1V2VEdit");if(ve)ve.style.display=v?"inline-block":"none";
  const vfs=$("c1FaceSplit");if(vfs)vfs.style.display=v?"block":"none";
}
function _c1V2VSet(files){
  const f=(files||[])[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=async()=>{
    clipControlVideos[1]=rd.result;
    if(clipControlType[1]==null)clipControlType[1]="raw";
    if(clipControlStrength[1]==null)clipControlStrength[1]=1;
    if(clipCopySrcAudio[1]==null)clipCopySrcAudio[1]=true;   // default: keep original audio
    _probeVideoDur(rd.result).then(d=>{if(d){clipControlDur[1]=d;renderClipOverview();_updateSplitHint();}});
    toast("Extracting the video's first frame\u2026");
    try{const frame=await _grabFirstFrame(rd.result);
      if(frame){_setStartImage(frame);clipStartImages[1]=frame;}}catch(e){}
    _renderC1Frames();renderClipOverview();
    toast("Control video set on Clip 1 \u2014 its first frame is the opening image (edit to reskin). Use \u2699 options for control type.");
  };
  rd.readAsDataURL(f);
  const inp=$("c1V2VFile");if(inp)inp.value="";
}
function _c1V2VClear(){
  delete clipControlVideos[1];delete clipControlType[1];
  delete clipControlStrength[1];delete clipCopySrcAudio[1];delete clipControlDur[1];
  _renderC1Frames();renderClipOverview();_updateSplitHint();
}
function _c1SetImg(which,files){
  const f=(files||[])[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=()=>{
    if(which==="start"){_setStartImage(rd.result);clipStartImages[1]=rd.result;}
    else _stagedEnd=rd.result;
    _renderC1Frames();renderClipOverview();
    toast("Clip 1 "+(which==="end"?"last":"first")+" frame set.");
  };
  rd.readAsDataURL(f);
  const inp=$(which==="start"?"c1StartFile":"c1EndFile");if(inp)inp.value="";
}
function _c1Clear(which){
  if(which==="start"){imgData=null;
    const d=$("drop");if(d){d.className="dropzone";d.innerHTML="&#128247;&nbsp; Click or drop an image";}}
  else _stagedEnd=null;
  _renderC1Frames();renderClipOverview();
}
function _c1Gen(which){
  // Open the full compose modal targeting Clip 1's first/last frame.
  const cur=(which==="start")?(imgData||null):(_stagedEnd||null);
  const other=(which==="start")?(_stagedEnd||null):(imgData||null);
  const extra=[];if(other)extra.push(other);
  _openNextCompose(cur||"",($("prompt").value||"").trim(),
    "Clip 1 "+(which==="end"?"last frame":"first frame"),
    {target:{kind:(which==="end"?"clip_end":"clip_start"),clip:1},
     extraRefs:extra, propose:!!cur});
}
// ── Edit-image modal (replaces native prompt) ──
// _editImgState.apply(newImageDataUrl) is called with the edited result.
let _editImgState={src:"",refs:[],title:"",apply:null,label:""};
function openEditImageModal(opts){
  _editImgState={src:opts.src||"",refs:(opts.refs||[]).slice(0,5),title:opts.title||"Edit image",
    apply:opts.apply||null,label:opts.label||"image"};
  $("editImgTitle").textContent=_editImgState.title;
  $("editImgSrc").innerHTML=_editImgState.src?"<img src='"+_editImgState.src+"'>":"<span class='cf-empty' style='color:var(--text-muted)'>no image</span>";
  $("editImgInstr").value="";
  ["loraFaceswap","loraAngles","loraSkin","loraUpscale"].forEach(i=>{
    const el=$(i);if(el)el.value=0;});
  try{_editLoraSync();}catch(e){}
  _renderEditImgRefs();_refreshEditTok();
  $("editImgModal").style.display="flex";
  setTimeout(()=>$("editImgInstr").focus(),50);
}
function closeEditImg(){$("editImgModal").style.display="none";}
function _renderEditImgRefs(){
  const box=$("editImgRefs");box.innerHTML="";
  _editImgState.refs.forEach((src,i)=>{
    const d=document.createElement("div");d.className="nc-img";
    d.innerHTML="<img src='"+src+"'><button class='nc-x' title='Remove'>&#10005;</button>";
    d.querySelector(".nc-x").onclick=()=>{_editImgState.refs.splice(i,1);_renderEditImgRefs();};
    box.appendChild(d);
  });
}
function _editImgAddRefs(files){
  [...(files||[])].forEach(f=>{const rd=new FileReader();
    rd.onload=()=>{if(_editImgState.refs.length<5){_editImgState.refs.push(rd.result);_renderEditImgRefs();}
      else toast("Up to 5 reference images.",true);};
    rd.readAsDataURL(f);});
  $("editImgRefFile").value="";
}
async function _refreshEditTok(){
  const b=$("editImgTok");if(!b)return;
  try{
    const j=await(await fetch("/api/tokens")).json();
    if(j.own_key){b.textContent="\ud83d\udd11 your OpenAI key";b.onclick=null;b.style.cursor="default";}
    else{b.onclick=buyTokens;b.style.cursor="pointer";
      b.textContent=(j.tokens!=null?j.tokens.toLocaleString()+" tokens":"tokens");}
  }catch(e){b.textContent="tokens";}
}
function _editLoraSync(){
  [["loraFaceswap","loraFaceswapV"],["loraAngles","loraAnglesV"],
   ["loraSkin","loraSkinV"],["loraUpscale","loraUpscaleV"]].forEach(([i,v])=>{
    const el=$(i);if(el)$(v).textContent=(+el.value).toFixed(2);});
  const fs=parseFloat(($("loraFaceswap")||{}).value)||0;
  const h=$("loraFaceswapHint");if(h)h.style.display=fs>0?"block":"none";
  // Face swap needs a reference face; nudge the button label.
  const go=$("editImgGo");
  if(go){const any=_editLoraActive();
    go.innerHTML=any?"\u2728 Apply (Qwen LoRA)":"\u270e Apply edit";}
}
function _editLoraActive(){
  return ["loraFaceswap","loraAngles","loraSkin","loraUpscale"]
    .some(i=>parseFloat(($(i)||{}).value)>0);
}
async function _editImgRun(){
  const instr=($("editImgInstr").value||"").trim();
  if(!_editImgState.src){toast("No image to edit.",true);return;}
  const loraActive=_editLoraActive();
  if(!instr && !loraActive){toast("Describe the change first.",true);return;}
  const go=$("editImgGo");go.disabled=true;const orig=go.innerHTML;go.innerHTML="Applying\u2026";
  try{
    if(loraActive){
      const fs=parseFloat(($("loraFaceswap")||{}).value)||0;
      if(fs>0 && !(_editImgState.refs&&_editImgState.refs.length)){
        toast("Face swap needs a reference face \u2014 upload one under Reference images.",true);
        go.disabled=false;go.innerHTML=orig;return;
      }
      const r=await fetch("/api/qwenedit/submit",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({image:_editImgState.src,
          references:_editImgState.refs||[],instruction:instr,
          lora_faceswap:fs,
          lora_angles:parseFloat(($("loraAngles")||{}).value)||0,
          lora_skin:parseFloat(($("loraSkin")||{}).value)||0,
          lora_upscale:parseFloat(($("loraUpscale")||{}).value)||0})});
      const j=await r.json();
      if(!j.ok){
        if(j.error==="insufficient_tokens"){toast("Out of tokens.",true);buyTokens();}
        else toast(j.error||"Edit failed.",true);
        go.disabled=false;go.innerHTML=orig;return;
      }
      if(j.tokens!=null)_applyTokBadge(j.tokens,100,false);
      // Queue job so progress shows; apply on completion.
      const applyFn=_editImgState.apply;const label=_editImgState.label;
      queue.unshift({id:j.job_id,status:"running",progress:5,
        stage:"GPU warming up",kind:"qwenedit",_applyLabel:label,
        _applyFn:applyFn,prompt:"Qwen edit",thumb:_editImgState.src,ts:Date.now()});
      if(typeof renderQueue==="function")try{renderQueue();}catch(e){}
      closeEditImg();toast("Qwen edit queued \u2014 watch the Jobs panel.");
      return;
    }
    const r=await fetch("/api/editimage",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:_editImgState.src,instruction:instr,
        references:_editImgState.refs,quality:(nextCfg.quality||"medium")})});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Edit failed.",true);go.disabled=false;go.innerHTML=orig;return;}
    _addImageToHistory(j.image,_editImgState.label);
    if(_editImgState.apply)_editImgState.apply(j.image);
    closeEditImg();toast("Edit applied.");
  }catch(e){toast("Error: "+e,true);}
  go.disabled=false;go.innerHTML=orig;
}
// Fullscreen image viewer with download.
function _showImageFull(src,name){
  $("ifvImg").src=src;
  const dl=$("ifvDownload");dl.href=src;dl.download=(name||"image")+".png";
  $("imgFullView").style.display="flex";
}
function _imgFullClose(e,force){
  if(force||e.target.id==="imgFullView"||e.target.classList.contains("ifv-close"))
    $("imgFullView").style.display="none";
}
function _editImgFull(){if(_editImgState.src)_showImageFull(_editImgState.src,"edit-source");}

async function _c1Edit(which){
  const cur=(which==="start")?(imgData||null):(_stagedEnd||null);
  if(!cur){toast("No image to edit.",true);return;}
  openEditImageModal({
    src:cur,refs:[cur],
    title:"Edit Clip 1 "+(which==="end"?"last":"first")+" frame",
    label:"Clip 1 "+(which==="end"?"last":"first")+" frame",
    apply:(img)=>{
      if(which==="start"){_setStartImage(img);clipStartImages[1]=img;}
      else _stagedEnd=img;
      _renderC1Frames();renderClipOverview();
    }});
}

/* ── ✨ Auto Prompt + LTX dialogue builder ── */
// Standing instructions + extra context the user can attach to Auto
// Prompt via the gear. Sent with the image on every Auto Prompt call.
let autoCfg={instructions:"",context:""};
function openAutoCfg(){
  $("autoCfgInstr").value=autoCfg.instructions||"";
  $("autoCfgContext").value=autoCfg.context||"";
  $("autoCfgInstr").oninput=()=>{autoCfg.instructions=$("autoCfgInstr").value;_autoCfgBadge();};
  $("autoCfgContext").oninput=()=>{autoCfg.context=$("autoCfgContext").value;_autoCfgBadge();};
  $("autoCfgModal").style.display="flex";
  setTimeout(()=>$("autoCfgInstr").focus(),50);
}
function closeAutoCfg(){$("autoCfgModal").style.display="none";}
function clearAutoCfg(){
  autoCfg={instructions:"",context:""};
  $("autoCfgInstr").value="";$("autoCfgContext").value="";_autoCfgBadge();
}
// Gold-tint the gear when custom instructions/context are set, so it's
// obvious Auto Prompt is being steered.
function _autoCfgBadge(){
  const on=!!((autoCfg.instructions||"").trim()||(autoCfg.context||"").trim());
  const g=$("autoCfgGear");if(g)g.classList.toggle("open",on);
}
let speakers=[];   // [{name:"MOTHER", voice:"British, medium-low pitch..."}]
let dialog=[];     // [{speaker:"MOTHER", text:"Look at me, sweetheart."}]

function openSpeakerModal(){
  renderSpeakers();
  $("speakerModal").style.display="flex";
}
function closeSpeakerModal(){
  $("speakerModal").style.display="none";
  renderDialog();          // refresh clip cards (speaker names may have changed)
  _updateSpkBtnLabel();
}
// Show how many speakers are configured on the button.
function _updateSpkBtnLabel(){
  const el=$("spkBtnLabel");if(!el)return;
  const n=speakers.length;
  el.innerHTML=n
    ? "\ud83c\udfa4 Speakers &amp; voices <span style='opacity:.6'>("+n+")</span>"
    : "\ud83c\udfa4 Add speakers &amp; voices";
}
function addSpeaker(){
  speakers.push({name:"SPEAKER "+(speakers.length+1),
    voice:"a natural adult voice, medium pitch, clear timbre, calm pace"});
  renderSpeakers();renderDialog();
}
function renderSpeakers(){
  _updateSpkBtnLabel();
  const box=$("spkList");if(!box)return;box.innerHTML="";
  if(!speakers.length){
    box.innerHTML="<div class='hintline' style='margin:0 0 8px'>No speakers yet \u2014 add one, or let \u2728 Auto Prompt detect them from your image.</div>";
    return;
  }
  speakers.forEach((s,i)=>{
    const row=document.createElement("div");row.className="spk-row";
    const top=document.createElement("div");top.className="spk-row-top";
    const nm=document.createElement("input");nm.className="spk-name";
    nm.value=s.name;nm.placeholder="NAME";
    nm.onchange=()=>{
      const old=s.name,v=(nm.value.trim()||"SPEAKER").toUpperCase();
      s.name=v;dialog.forEach(l=>{if(l.speaker===old)l.speaker=v;});
      renderSpeakers();renderDialog();
    };
    const x=document.createElement("button");x.className="lora-x";
    x.innerHTML="&#10005;";x.title="Remove speaker";
    x.onclick=()=>{
      const gone=s.name;
      speakers.splice(i,1);
      if(speakers.length){
        const fb=speakers[0].name;
        dialog.forEach(l=>{if(l.speaker===gone)l.speaker=fb;});
      }else{
        dialog=[];
      }
      renderSpeakers();renderDialog();
    };
    top.append(nm,x);
    const lk=document.createElement("textarea");lk.className="spk-voice";
    lk.value=s.look||"";lk.placeholder="Appearance (kept identical every shot): age, face, hair, wardrobe…";
    lk.oninput=()=>{s.look=lk.value;};
    const vo=document.createElement("textarea");vo.className="spk-voice";
    vo.value=s.voice;vo.placeholder="Voice: accent, pitch, timbre, pace…";
    vo.oninput=()=>{s.voice=vo.value;};
    row.append(top,lk,vo);box.appendChild(row);
  });
}
function addDlgLine(){
  if(!speakers.length)addSpeaker();
  // New lines default to clip 1; open the editor so it's typed in the
  // roomy modal, not a cramped inline box.
  dialog.push({speaker:speakers[0].name,text:"",clip:1});
  renderDialog();
  openClipModal(1);
}
// Words-per-second heuristic for auto length: ~2.3 wps natural speech,
// plus a small head/tail pad, snapped to LTX's 8k+1 frame rule.
function _clipAutoFrames(words,fps){
  // Dialogue-driven when there are lines (speech ~2.3 words/sec + lead-in),
  // but a scene with NO dialogue should still breathe — default silent/
  // ambient shots to ~6s rather than collapsing to the floor.
  const secs = words>0 ? Math.max(3.0, words/2.3 + 1.2) : 6.0;
  let f=Math.round(secs*fps);
  f=Math.max(9,((f-1)/8|0)*8+1);      // snap to 8k+1
  return Math.min(_ltxMaxFrames(fps),f);
}
// LTX-2.3 supports up to ~20s per clip (≈480 frames at 24fps). Keep it
// under the 8k+1 grid and clamp to the model's real limit for the fps.
function _ltxMaxFrames(fps){
  const cap=Math.floor(20*fps);       // 20 seconds
  let f=((cap-1)/8|0)*8+1;
  return Math.min(f, 481);            // hard ceiling near the 480-frame max
}
// Build the clip plan from the dialogue: one entry per clip number in
// use (min 1 clip), each with its assigned lines and a frame count.
// The timeline is EMPTY until the user commits their first clip via
// "Add to timeline". Until then the top inputs are just a staging area
// and Generate stays disabled.
let _committedClips=0;   // how many clips have been added to the timeline
function clipPlan(){
  const fps=parseInt($("fpsV").value,10)||24;
  if(!_committedClips) return [];      // nothing committed yet
  const maxClip=Math.max(_committedClips,
    dialog.reduce((m,l)=>Math.max(m,l.clip||1),1));
  const plan=[];
  for(let c=1;c<=maxClip;c++){
    const lines=dialog.filter(l=>(l.clip||1)===c&&(l.text||"").trim());
    const words=lines.reduce((s,l)=>s+l.text.trim().split(/\s+/).length,0);
    let auto=_clipAutoFrames(words,fps);
    // v2v: the length comes from the CONTROL VIDEO, not the dialogue.
    if(clipControlVideos[c]&&clipControlDur[c]){
      let f=Math.round(clipControlDur[c]*fps);
      auto=Math.max(9,((f-1)/8|0)*8+1);
    }
    // Respect a manual override the user typed for this clip.
    const ov=clipFrameOverrides[c];
    plan.push({clip:c,lines:lines,frames:(ov||auto),auto:auto,
               overridden:!!ov});
  }
  return plan;
}
let clipFrameOverrides={};   // {clipNumber: frames} — user length edits
let clipPromptOverrides={};  // {clipNumber: sceneText} — per-clip scene edits
let clipStartImages={};      // {clipNumber: dataURL} — start frame per clip (Auto Next Scene / manual)
let clipEndImages={};        // {clipNumber: dataURL} — optional END frame per clip (LTX lands on it)
// ── Video-to-video (IC-LoRA) per-clip control ──
let clipControlVideos={};    // {clipNumber: dataURL} — control video (motion source)
let clipControlType={};      // {clipNumber: 'raw'|'canny'|'depth'|'motion_track'}
let clipControlStrength={};  // {clipNumber: 0..1} — reference conditioning strength
let clipCopySrcAudio={};     // {clipNumber: bool} — copy the control video's audio
let clipControlDur={};       // {clipNumber: seconds} — control video duration
let clipControlSplit={};     // {clipNumber: {index,count,per,total}} — auto-split slice
function _probeVideoDur(dataUrl){
  return new Promise(res=>{
    try{const v=document.createElement("video");
      v.preload="metadata";
      v.onloadedmetadata=()=>res(isFinite(v.duration)?v.duration:null);
      v.onerror=()=>res(null);
      v.src=dataUrl;
    }catch(e){res(null);}
  });
}
let clipStartPlaceholders={}; // {clipNumber: dataURL} — DISPLAY-ONLY placeholder for Extend clips (render uses the real last frame; never sent in the payload)
// The scene text a clip uses: its override if set, else the main prompt.
function clipSceneText(c){
  const ov=clipPromptOverrides[c];
  return (ov!==undefined && ov!==null) ? ov : ($("prompt").value||"");
}
// Live indicator under the scene box. Frames this box as Clip 1's scene
// and points to the timeline for editing any clip — resolving the "is this
// leftover input or live state?" confusion.
function _updateSceneRoleHint(){
  const h=$("promptRoleHint");if(!h)return;
  const plan=(typeof clipPlan==="function")?clipPlan():[];
  const nClips=plan.length;
  let msg;
  if(nClips<=1){
    msg="This is <b>Clip 1</b>\u2019s scene \u2014 your movie's opening shot. Edit it here anytime. Add more clips in the timeline below; each gets its own scene.";
  }else{
    msg="This is <b>Clip 1</b>\u2019s scene. Your other "+(nClips-1)+" clip"+(nClips-1===1?"":"s")+" each have their own scene \u2014 tap any clip in the timeline below to edit it.";
  }
  h.innerHTML=msg;
}

// ── SCRIPT MODEL ──────────────────────────────────────────────────────
// One JSON object capturing the whole storyboard: base scene, engine,
// fps, speakers (name+voice), every clip's scene/length/dialogue/start
// image, and the uploaded first image. Used by the agent (to see and
// edit state) and by script download/upload.
function buildScript(){
  const fps=parseInt($("fpsV").value,10)||24;
  const plan=clipPlan();
  const clips=plan.map(p=>({
    clip:p.clip,
    scene:clipSceneText(p.clip),
    scene_is_custom:(clipPromptOverrides[p.clip]!==undefined
      && clipPromptOverrides[p.clip]!==null
      && clipPromptOverrides[p.clip]!==$("prompt").value),
    length_frames:p.frames,
    length_seconds:+(p.frames/fps).toFixed(2),
    length_overridden:p.overridden,
    start_image:clipStartImages[p.clip]||null,
    end_image:clipEndImages[p.clip]||null,
    lines:dialog.filter(l=>(l.clip||1)===p.clip)
      .map(l=>({speaker:l.speaker,text:l.text||""}))
  }));
  return {
    version:1,
    engine:currentEngine,
    fps:fps,
    base_scene:$("prompt").value||"",
    first_image:imgData||null,
    speakers:speakers.map(s=>({name:s.name,voice:s.voice||"",look:s.look||""})),
    clips:clips
  };
}
// Apply a script object to the live state (replacing it). Tolerant of
// partial scripts. Returns a short summary of what was set.
function applyScript(sc){
  if(!sc||typeof sc!=="object")throw "not a script object";
  if(sc.engine&&["wan","wan22","ltx"].includes(sc.engine))setEngine(sc.engine);
  if(sc.fps){setSL("fps",parseInt(sc.fps,10)||16);}
  if(typeof sc.base_scene==="string")$("prompt").value=sc.base_scene;
  if(sc.first_image){try{_setStartImage(sc.first_image);}catch(e){imgData=sc.first_image;}}
  // speakers
  if(Array.isArray(sc.speakers)){
    speakers=sc.speakers.map(s=>({name:(s.name||"SPEAKER").toUpperCase(),
      voice:s.voice||"a natural adult voice, medium pitch, clear timbre, calm pace",
      look:s.look||""}));
  }
  // clips → dialog + overrides
  if(Array.isArray(sc.clips)){
    dialog=[];clipPromptOverrides={};clipFrameOverrides={};clipStartImages={};clipEndImages={};clipStartPlaceholders={};clipControlVideos={};clipControlType={};clipControlStrength={};clipCopySrcAudio={};clipControlDur={};clipControlSplit={};clipStartGhosts={};
    sc.clips.forEach(cl=>{
      const c=cl.clip||1;
      if(typeof cl.scene==="string" && cl.scene_is_custom!==false
         && cl.scene!==($("prompt").value||""))
        clipPromptOverrides[c]=cl.scene;
      if(cl.length_overridden && cl.length_frames)
        clipFrameOverrides[c]=cl.length_frames;
      if(cl.start_image)clipStartImages[c]=cl.start_image;
      if(cl.end_image)clipEndImages[c]=cl.end_image;
      (cl.lines||[]).forEach(l=>{
        const sp=(l.speaker||(speakers[0]||{}).name||"SPEAKER").toUpperCase();
        if(!speakers.some(x=>x.name===sp))
          speakers.push({name:sp,voice:"a natural adult voice, medium pitch, clear timbre, calm pace"});
        dialog.push({speaker:sp,text:l.text||"",clip:c});
      });
    });
  }
  _committedClips=Array.isArray(sc.clips)?sc.clips.length:0;
  renderSpeakers();renderDialog();_syncGenerateEnabled();
  return "Loaded "+(sc.clips?sc.clips.length:0)+" clip(s), "
    +(speakers.length)+" speaker(s).";
}
// Download the current script as a .json file.
function downloadScript(){
  const sc=buildScript();
  const blob=new Blob([JSON.stringify(sc,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="missinglink_script.json";a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
  toast("Script downloaded.");
}
// Upload a script .json and apply it.
function uploadScriptFile(files){
  const f=(files||[])[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=()=>{
    try{const sc=JSON.parse(rd.result);const msg=applyScript(sc);
      toast(msg);}
    catch(e){toast("Couldn't load script: "+e,true);}
  };
  rd.readAsText(f);
  const inp=$("scriptFile");if(inp)inp.value="";
}

// ── AGENT ACTION EXECUTOR ─────────────────────────────────────────────
// The agent's LLM returns {reply, actions:[...]}. Each action mutates the
// storyboard. Applied immediately; returns a list of human-readable notes
// on what changed (echoed back to the agent + shown to the user).
function _snapFrames(sec,fps){
  let f=Math.round(sec*fps);
  if(currentEngine==="ltx"){f=Math.max(9,((f-1)/8|0)*8+1);f=Math.min(_ltxMaxFrames(fps),f);}
  else{f=Math.max(17,Math.round((f-1)/4)*4+1);f=Math.min(81,f);}
  return f;
}
function applyAgentActions(actions){
  if(!Array.isArray(actions))return [];
  const fps=parseInt($("fpsV").value,10)||24;
  const notes=[];
  const spkName=n=>(n||"SPEAKER").toString().toUpperCase().slice(0,24);
  actions.forEach(a=>{
    try{
      const t=(a.type||a.action||"").toString();
      if(t==="set_base_scene"){
        $("prompt").value=(a.scene||a.text||"").toString();
        notes.push("Base scene updated.");
      }else if(t==="set_engine"){
        if(["wan","wan22","ltx"].includes(a.engine)){setEngine(a.engine);
          notes.push("Engine set to "+a.engine+".");}
      }else if(t==="set_fps"){
        setSL("fps",parseInt(a.fps,10)||fps);notes.push("FPS set to "+a.fps+".");
      }else if(t==="set_clip_scene"){
        const c=parseInt(a.clip,10)||1;
        clipPromptOverrides[c]=(a.scene||a.text||"").toString();
        notes.push("Clip "+c+" scene updated.");
      }else if(t==="reset_clip_scene"){
        const c=parseInt(a.clip,10)||1;delete clipPromptOverrides[c];
        notes.push("Clip "+c+" scene reset to base.");
      }else if(t==="set_clip_length"){
        const c=parseInt(a.clip,10)||1;
        let f=null;
        if(a.seconds!=null)f=_snapFrames(parseFloat(a.seconds),fps);
        else if(a.frames!=null)f=parseInt(a.frames,10);
        if(f){clipFrameOverrides[c]=f;
          notes.push("Clip "+c+" length set to "+(f/fps).toFixed(1)+"s.");}
      }else if(t==="auto_clip_length"){
        const c=parseInt(a.clip,10)||1;delete clipFrameOverrides[c];
        notes.push("Clip "+c+" length back to auto.");
      }else if(t==="add_speaker"){
        const nm=spkName(a.name);
        if(!speakers.some(s=>s.name===nm)){
          speakers.push({name:nm,voice:(a.voice||"a natural adult voice, medium pitch, clear timbre, calm pace")});
          notes.push("Added speaker "+nm+".");}
      }else if(t==="rename_speaker"){
        const from=spkName(a.from||a.name),to=spkName(a.to);
        const s=speakers.find(x=>x.name===from);
        if(s){s.name=to;dialog.forEach(l=>{if(l.speaker===from)l.speaker=to;});
          notes.push("Renamed "+from+" \u2192 "+to+".");}
      }else if(t==="set_voice"){
        const nm=spkName(a.name);const s=speakers.find(x=>x.name===nm);
        if(s){s.voice=(a.voice||"").toString();notes.push("Updated "+nm+"'s voice.");}
      }else if(t==="remove_speaker"){
        const nm=spkName(a.name);const i=speakers.findIndex(s=>s.name===nm);
        if(i>=0){speakers.splice(i,1);
          const fb=(speakers[0]||{}).name||"SPEAKER";
          dialog.forEach(l=>{if(l.speaker===nm)l.speaker=fb;});
          notes.push("Removed speaker "+nm+".");}
      }else if(t==="add_line"){
        const c=parseInt(a.clip,10)||1;const sp=spkName(a.speaker);
        if(!speakers.some(s=>s.name===sp))
          speakers.push({name:sp,voice:"a natural adult voice, medium pitch, clear timbre, calm pace"});
        dialog.push({speaker:sp,text:(a.text||"").toString(),clip:c});
        delete clipFrameOverrides[c];
        notes.push("Added a line for "+sp+" in clip "+c+".");
      }else if(t==="set_line"){
        // Replace the Nth line (1-based) of a clip.
        const c=parseInt(a.clip,10)||1;const idx=(parseInt(a.index,10)||1)-1;
        const rows=dialog.map((l,i)=>({l,i})).filter(o=>(o.l.clip||1)===c);
        if(rows[idx]){const o=rows[idx];
          if(a.text!=null)dialog[o.i].text=a.text.toString();
          if(a.speaker!=null)dialog[o.i].speaker=spkName(a.speaker);
          delete clipFrameOverrides[c];
          notes.push("Updated line "+(idx+1)+" in clip "+c+".");}
      }else if(t==="remove_line"){
        const c=parseInt(a.clip,10)||1;const idx=(parseInt(a.index,10)||1)-1;
        const rows=dialog.map((l,i)=>({l,i})).filter(o=>(o.l.clip||1)===c);
        if(rows[idx]){dialog.splice(rows[idx].i,1);delete clipFrameOverrides[c];
          notes.push("Removed line "+(idx+1)+" from clip "+c+".");}
      }else if(t==="clear_clip_dialogue"){
        const c=parseInt(a.clip,10)||1;
        dialog=dialog.filter(l=>(l.clip||1)!==c);delete clipFrameOverrides[c];
        notes.push("Cleared dialogue in clip "+c+".");
      }else if(t==="add_clip"){
        const nc=_nextClipNum();
        if(a.scene)clipPromptOverrides[nc]=a.scene.toString();
        (a.lines||[]).forEach(l=>{
          const sp=spkName(l.speaker);
          if(!speakers.some(s=>s.name===sp))
            speakers.push({name:sp,voice:"a natural adult voice, medium pitch, clear timbre, calm pace"});
          dialog.push({speaker:sp,text:(l.text||"").toString(),clip:nc});
        });
        if(!(a.lines||[]).length)dialog.push({speaker:(speakers[0]||{name:"SPEAKER"}).name,text:"",clip:nc});
        notes.push("Added clip "+nc+".");
      }else if(t==="delete_clip"){
        const c=parseInt(a.clip,10)||1;deleteClip(c);
        notes.push("Deleted clip "+c+".");
      }else{
        notes.push("(skipped unknown action: "+t+")");
      }
    }catch(e){notes.push("(action failed: "+e+")");}
  });
  renderSpeakers();renderDialog();
  return notes;
}

// ── AGENT CHAT ────────────────────────────────────────────────────────
let _agentHistory=[];   // [{role:"user"|"assistant", content:"..."}]
let _agentBusy=false;
let _agentMin=false;
function agentOpen(){
  const p=$("agentPanel");p.style.display="flex";
  $("agentFab").style.display="none";
  if(!p.dataset.placed){
    p.style.right="24px";p.style.bottom="24px";p.dataset.placed="1";
  }
  // Start COLLAPSED to the header-only view (compact); expand via Show/Hide.
  _agentMin=true;$("agentBody").style.display="none";
  _syncAgentMinLabel();
  _renderAgentLog();
}
function agentMin(){
  // Toggle between the compact header-only view and the full chat.
  _agentMin=!_agentMin;
  $("agentBody").style.display=_agentMin?"none":"flex";
  _syncAgentMinLabel();
}
function _syncAgentMinLabel(){
  const b=$("agentMinBtn");if(!b)return;
  b.innerHTML=_agentMin?"\u25b2 Show":"\u25bc Hide";
}
function agentClose(){
  $("agentPanel").style.display="none";$("agentFab").style.display="";
}
// Drag by the header.
(function _agentDrag(){
  let sx,sy,ox,oy,dragging=false;
  document.addEventListener("mousedown",e=>{
    const h=e.target.closest("#agentHead");if(!h)return;
    if(e.target.closest("button"))return;   // buttons aren't drag handles
    const p=$("agentPanel");const r=p.getBoundingClientRect();
    // Switch to top/left positioning for free dragging.
    p.style.left=r.left+"px";p.style.top=r.top+"px";
    p.style.right="auto";p.style.bottom="auto";
    sx=e.clientX;sy=e.clientY;ox=r.left;oy=r.top;dragging=true;
    document.body.style.userSelect="none";
  });
  document.addEventListener("mousemove",e=>{
    if(!dragging)return;const p=$("agentPanel");
    let nx=ox+(e.clientX-sx),ny=oy+(e.clientY-sy);
    nx=Math.max(4,Math.min(window.innerWidth-80,nx));
    ny=Math.max(4,Math.min(window.innerHeight-40,ny));
    p.style.left=nx+"px";p.style.top=ny+"px";
  });
  document.addEventListener("mouseup",()=>{dragging=false;document.body.style.userSelect="";});
})();

function agentClear(){
  _agentHistory=[];_renderAgentLog();
  toast("Agent conversation cleared.");
}
function _renderAgentLog(){
  const box=$("agentLog");if(!box)return;box.innerHTML="";
  if(!_agentHistory.length){
    box.innerHTML="<div class='agent-hint'>Tell me what to do with the storyboard \u2014 rewrite scenes, change dialogue, adjust clip length, add/rename speakers, add or delete clips. e.g. <i>\u201cmake clip 2 angrier and 2 seconds longer\u201d</i>, <i>\u201cadd a whispered line for the villain in the last clip\u201d</i>.</div>";
    return;
  }
  _agentHistory.forEach(m=>{
    const d=document.createElement("div");
    d.className="agent-msg "+(m.role==="user"?"am-user":"am-bot");
    d.textContent=m.content;box.appendChild(d);
  });
  box.scrollTop=box.scrollHeight;
}
async function agentSend(){
  if(_agentBusy)return;
  const inp=$("agentInput");const msg=(inp.value||"").trim();
  if(!msg)return;
  inp.value="";
  _agentHistory.push({role:"user",content:msg});_renderAgentLog();
  _agentBusy=true;$("agentSend").disabled=true;
  const thinking={role:"assistant",content:"\u2026"};
  _agentHistory.push(thinking);_renderAgentLog();
  try{
    const r=await fetch("/api/agent",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        message:msg,
        history:_agentHistory.slice(0,-2),   // prior turns (exclude this msg + placeholder)
        script:buildScript()})});
    const j=await r.json();
    _agentHistory.pop();   // remove the "…" placeholder
    if(!j.ok){
      _agentHistory.push({role:"assistant",content:"\u26a0 "+(j.error||"agent failed")});
      _renderAgentLog();
    }else{
      let notes=[];
      if(Array.isArray(j.actions)&&j.actions.length)notes=applyAgentActions(j.actions);
      let reply=(j.reply||"").trim()||"Done.";
      if(notes.length)reply+="\n\n\u2713 "+notes.join("\n\u2713 ");
      _agentHistory.push({role:"assistant",content:reply});
      _renderAgentLog();
    }
  }catch(e){
    _agentHistory.pop();
    _agentHistory.push({role:"assistant",content:"\u26a0 Error: "+e});
    _renderAgentLog();
  }
  _agentBusy=false;$("agentSend").disabled=false;
}
// One clean clip list — no separate "plan" panel. Each clip is a single
// row: scene snippet, dialogue preview, an inline length field, and Edit.
function renderDialog(){ renderClipOverview(); }
function _updateSplitHint(){
  const row=$("splitHint");if(!row)return;
  const dur=clipControlDur[1],has=!!clipControlVideos[1];
  const seg=$("splitSeg"),main=$("v2vSeg");
  // Do NOT overwrite the field the user is editing (empty/partial values
  // must persist so they can finish typing).
  const editing=(document.activeElement===seg||document.activeElement===main);
  if(seg&&main&&!seg.value&&!editing)seg.value=main.value||20;
  const s=parseFloat(seg&&seg.value)||20;
  // Visibility depends ONLY on whether this clip will split — a control
  // video longer than the (last committed) segment length. While editing,
  // keep the row shown so the input never disappears under the cursor.
  const willSplit=(has&&dur&&dur>s+0.5&&currentEngine==="ltx");
  row.style.display=(willSplit||(editing&&has&&dur))?"flex":"none";
  if(has&&dur&&s>0){
    const n=Math.max(1,Math.ceil(dur/s));
    $("splitN").textContent=n;
    const rem=dur-(n-1)*s;
    $("splitPer").textContent=(n<=3)
      ?Array.from({length:n},(_,k)=>(k<n-1?s:rem.toFixed(1))).join("+")
      :s;
  }
}
document.addEventListener("DOMContentLoaded",()=>{
  const seg=$("splitSeg"),main=$("v2vSeg");
  if(seg)seg.addEventListener("input",()=>{
    if(main&&document.activeElement!==main)main.value=seg.value;
    _updateSplitHint();});
  if(main)main.addEventListener("input",()=>{
    if(seg&&document.activeElement!==seg)seg.value=main.value;
    _updateSplitHint();});
});
let _stlMin=false;
// ── Persist character across clip start frames (Qwen faceswap) ──
let _charRef=null;              // reference image data URL
let _charSel={};               // {clip:true} selected clips
function openCharPersist(){
  const plan=clipPlan();
  if(!plan.length){toast("Create some clips first.",true);return;}
  _charSel={};plan.forEach(p=>_charSel[p.clip]=true);   // all selected by default
  _renderCharClips();_charUpdateBtn();
  $("charStrengthV").textContent="1.00";$("charStrength").value=1.0;
  const st=$("charStatus");if(st)st.style.display="none";
  $("charModal").style.display="flex";
}
function closeCharPersist(){$("charModal").style.display="none";}
function _renderCharClips(){
  const box=$("charClipList");if(!box)return;box.innerHTML="";
  clipPlan().forEach(p=>{
    const c=p.clip;
    const b=document.createElement("button");
    b.className="chip";b.style.cssText="padding:5px 10px;font-size:11px"
      +(_charSel[c]?";border-color:var(--gold);color:var(--gold)":";opacity:.55");
    b.textContent="Clip "+c;
    b.onclick=()=>{_charSel[c]=!_charSel[c];_renderCharClips();_charUpdateBtn();};
    box.appendChild(b);
  });
}
function _charUpdateBtn(){
  const n=Object.values(_charSel).filter(Boolean).length;
  $("charN").textContent=n;$("charCost").textContent=n*100;
  const btn=$("charRunBtn");if(btn)btn.disabled=(!_charRef||!n);
}
function _charRefUpload(files){
  const f=files&&files[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=()=>{_charRef=rd.result;
    $("charRefSlot").innerHTML="<img src='"+_charRef+"' style='width:100%;height:100%;object-fit:cover'>";
    _charUpdateBtn();};
  rd.readAsDataURL(f);
}
document.addEventListener("DOMContentLoaded",()=>{
  const s=$("charStrength");
  if(s)s.addEventListener("input",()=>{$("charStrengthV").textContent=(+s.value).toFixed(2);});
});
async function runCharPersist(){
  if(!_charRef){toast("Upload a character reference first.",true);return;}
  const clips=Object.keys(_charSel).filter(k=>_charSel[k]).map(Number).sort((a,b)=>a-b);
  if(!clips.length){toast("Select at least one clip.",true);return;}
  const btn=$("charRunBtn");btn.disabled=true;
  let queued=0;
  for(const c of clips){
    const frame=_clipStartImageFor(c);
    if(!frame){toast("Clip "+c+" has no start frame \u2014 skipped.",true);continue;}
    try{
      const sub=await(await fetch("/api/faceswap/submit",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({frame:frame,reference:_charRef,clip:c,
          faceswap:parseFloat($("charStrength").value)||1.0})})).json();
      if(!sub.ok){
        if(sub.error==="insufficient_tokens"){
          toast("Out of tokens \u2014 opening the store.",true);buyTokens();break;
        }
        toast("Clip "+c+": "+(sub.error||"submit failed"),true);continue;
      }
      if(sub.tokens!=null)_applyTokens(sub.tokens);
      // Add to the LOCAL queue so it appears in the Jobs panel; the shared
      // poller drives progress and the done-branch writes it back.
      queue.unshift({id:sub.job_id,status:"running",progress:5,
        stage:"GPU warming up",kind:"faceswap",_clip:c,
        prompt:"Face swap \u2014 clip "+c,thumb:frame,ts:Date.now()});
      queued++;
    }catch(e){toast("Clip "+c+": "+e.message,true);}
  }
  if(typeof renderQueue==="function")try{renderQueue();}catch(e){}
  closeCharPersist();
  if(queued)toast(queued+" face swap"+(queued===1?"":"s")+" queued \u2014 watch the Jobs panel.");
  btn.disabled=false;
}
function _charPoll(jobId, c, st, i, n){
  // Human labels for the studio job's real stages, including the cold-GPU
  // spin-up window (a queued job with no stage yet = GPU warming/claiming).
  const STAGE_LABEL={
    fetching:"loading inputs",preparing:"preparing",
    encoding_prompt:"encoding prompt",encoding_image:"encoding image",
    generating:"generating",decoding:"finishing",upscaling:"upscaling"};
  return new Promise((res)=>{
    let tries=0,sawRunning=false;
    const iv=setInterval(async()=>{
      tries++;
      try{
        const j=await(await fetch("/api/faceswap/poll/"+jobId)).json();
        if(j.status==="done"&&j.url){clearInterval(iv);res(j.url);return;}
        if(!j.ok||j.status==="error"){clearInterval(iv);
          st.textContent="Clip "+c+": "+(j.error||"failed");res(null);return;}
        let label;
        if(j.status==="queued"){
          label="#"+(j.position||"?")+" in queue";
        }else if(j.status==="running"){
          sawRunning=true;
          if(!j.stage){
            // running but no stage yet = GPU spinning up / claiming the job
            label="\u26a1 GPU warming up\u2026";
          }else{
            label=STAGE_LABEL[j.stage]||j.stage;
            if(j.total_steps>0&&j.stage==="generating")
              label+=" "+j.step+"/"+j.total_steps;
          }
        }else{
          label=j.status;
        }
        st.textContent="Clip "+c+" ("+i+"/"+n+"): "+label;
      }catch(e){}
      if(tries>150){clearInterval(iv);res(null);}   // ~5 min cap
    },2000);
  });
}
async function _urlToDataUrl(url){
  const blob=await(await fetch(url)).blob();
  return await new Promise((res)=>{const rd=new FileReader();
    rd.onload=()=>res(rd.result);rd.readAsDataURL(blob);});
}
function _applyTokens(t){
  try{_applyTokBadge(t, 100, false);}catch(e){}
  const el=$("mlTokens");if(el)el.textContent=t;
}
function toggleStl_placeholder(){}

function toggleStl(){
  _stlMin=!_stlMin;
  const strip=$("clipOverview");if(strip)strip.style.display=_stlMin?"none":"flex";
  const b=$("stlMin");if(b)b.innerHTML=_stlMin?"\u9652":"\u9662";
  if(b)b.innerHTML=_stlMin?"\u25b4":"\u25be";
  if(b)b.title=_stlMin?"Expand the timeline":"Minimize the timeline";
}
function renderClipOverview(){
  const box=$("clipOverview");if(!box)return;box.innerHTML="";
  const fps=parseInt($("fpsV").value,10)||24;
  const plan=clipPlan();
  if(!plan.length){
    const empty=document.createElement("div");empty.className="clip-empty";
    empty.innerHTML="Timeline empty \u2014 <b style='color:var(--gold)'>\u2795 Create clip</b> from the left";
    box.appendChild(empty);
    const add=document.createElement("button");add.className="chip";
    add.style.cssText="flex:0 0 auto;align-self:center";
    add.innerHTML="\u2795";add.title="Add an empty clip";
    add.onclick=()=>addClipNewScene();
    box.appendChild(add);
    return;
  }
  plan.forEach((pl,idx)=>{
    const c=pl.clip;
    const spoken=dialog.filter(l=>(l.clip||1)===c&&(l.text||"").trim());
    const scene=(clipSceneText(c)||"").trim();
    const custom=(clipPromptOverrides[c]!==undefined && clipPromptOverrides[c]!==null
                  && clipPromptOverrides[c]!==$("prompt").value);
    const secs=(pl.frames/fps).toFixed(1);
    const startImg=clipStartImages[c]||null;
    const hasCtrl=!!clipControlVideos[c];
    const tile=document.createElement("div");tile.className="flc";
    tile.style.width=Math.max(70,Math.min(320,Math.round(parseFloat(secs)*10)))+"px";
    tile.title=(scene?scene+"\n":"")
      +(c>1?(startImg?"Custom start frame":"Continues clip "+(c-1)+" from its last frame")+"\n":"")
      +secs+"s \u00b7 "+spoken.length+" line"+(spoken.length===1?"":"s")
      +(hasCtrl?" \u00b7 video-to-video":"");
    // Thumbnail-first: the start image IS the card. Chained clips without
    // a custom frame show a chain glyph; icon badges carry the rest.
    const ghost=(!startImg)&&clipStartGhosts[c];
    tile.innerHTML=
      (startImg?"<img class='flc-img' src='"+startImg+"'>"
       :ghost?"<img class='flc-img flc-ghost' src='"+ghost+"'>"
             :"<div class='flc-ph'>"+(c===1?"\ud83c\udfac":"\u26d3")+"</div>")
      +"<span class='flc-num'>"+c+"</span>"
      +"<button class='flc-x' title='Delete clip'>\u00d7</button>"
      +"<div class='flc-bar'>"
        +"<span class='flc-t'>"+secs+"s</span>"
        +"<span class='flc-ic'>"
          +(hasCtrl?"<span title='video-to-video control'>\ud83c\udf9e</span>":"")
          +(spoken.length?"<span title='"+spoken.length+" dialogue line(s)'>\ud83d\udcac</span>":"")
          +(custom?"<span title='custom scene prompt'>\u270e</span>":"")
          +(c>1&&!startImg?"<span title='chains from clip "+(c-1)+"'>\ud83d\udd17</span>":"")
        +"</span>"
      +"</div>";
    tile.querySelector(".flc-x").onclick=(e)=>{e.stopPropagation();deleteClipCard(c);};
    if(ghost){
      // Ghost = predicted start frame: click EXPANDS (view/download);
      // a hover ✎ opens the editor to replace it with a real frame.
      tile.title="Predicted start frame (from the previous clip's control) \u2014 "
        +"click to view/download. It is NOT used at render unless you replace "
        +"the start image in the editor.\n"+tile.title;
      const ed=document.createElement("button");ed.className="flc-x";
      ed.style.right="26px";ed.innerHTML="\u270e";ed.title="Edit this clip";
      ed.onclick=(e)=>{e.stopPropagation();openClipModal(c);};
      tile.appendChild(ed);
      tile.onclick=(e)=>{if(!e.target.closest("button"))
        _showImageFull(ghost,"clip"+c+"-predicted-start");};
    }else{
      tile.onclick=(e)=>{if(!e.target.closest("button"))openClipModal(c);};
    }
    box.appendChild(tile);
  });
  // Slim vertical icon column after the last clip: + add, \u23e9 extend.
  const col=document.createElement("div");col.className="flc-mini";
  const add=document.createElement("button");add.className="flc-mbtn";
  add.innerHTML="\u2795";add.title="Add a fresh clip";
  add.onclick=()=>addClipNewScene();
  const ext=document.createElement("button");ext.className="flc-mbtn";
  ext.innerHTML="\u23e9";ext.title="Extend the same shot from the last frame";
  ext.onclick=()=>addClipExtend();
  col.appendChild(add);col.appendChild(ext);
  box.appendChild(col);
  _updateGenBtnLabel(plan.length);
  _updateSceneRoleHint();
}
function _updateGenBtnLabel(nClips){
  const b=$("genBtn");if(!b)return;
  if(currentEngine==="ltx" && nClips>1)
    b.innerHTML="\u2726 Generate movie ("+nClips+" clips)";
  else if(currentEngine==="ltx")
    b.innerHTML="\u2726 Generate clip";
  else
    b.innerHTML="\u2726 Generate Video";
}
function _esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
// Adding a clip now asks the user's intent: extend the same scene, or a
// new scene (gpt-image-2 generates its start image from prior context).
function addClip(){
  if(!speakers.length)addSpeaker();
  $("addClipModal").style.display="flex";
}
function closeAddClip(){$("addClipModal").style.display="none";}
function _nextClipNum(){
  return dialog.reduce((m,l)=>Math.max(m,l.clip||1),0)+1;
}
// EXTEND: same shot/setting continues. No new image — it chains from the
// previous clip's last frame. Its scene defaults to the previous clip's
// scene so it reads as the same continuous moment; opens the editor so
// you can just keep the conversation going.
// EXTEND: same shot continues smoothly from the previous clip's last
// frame. The AI writes a CONTINUATION prompt describing the ongoing
// action (so motion carries on, not resets), plus optional dialogue.
async function addClipExtend(){
  closeAddClip();
  const nc=_nextClipNum();
  const prevScene=(clipSceneText(nc-1)||"").trim();
  const prevCtxImg=clipStartImages[nc-1]||imgData||null;
  if(prevScene && prevScene!==$("prompt").value) clipPromptOverrides[nc]=prevScene;
  if(!dialog.some(l=>(l.clip||1)===nc))
    dialog.push({speaker:(speakers[0]||{name:"SPEAKER"}).name,text:"",clip:nc});
  // Show a PLACEHOLDER in this clip's start slot. We do NOT try to guess a
  // real frame from history (there's no reliable clip->video mapping), and
  // an extend clip's true starting frame is the previous clip's LAST frame,
  // which is captured automatically at render time. The placeholder is
  // display-only and never enters the render payload. The user can still
  // generate/upload a real start image in the editor to override it.
  clipStartPlaceholders[nc]=_extendPlaceholderFrame(nc);
  // GHOST: if the previous clip is driven by a control video, its slice's
  // LAST frame predicts where this clip starts — show it grayed so the
  // user can inspect/download/edit for consistency. Display-only unless
  // they replace the start image.
  (function(){
    const pcv=clipControlVideos[nc-1];if(!pcv)return;
    const sp=clipControlSplit[nc-1];
    const t=sp?(sp.start+sp.len-0.05)
              :((clipControlDur[nc-1]||9999)-0.05);
    _grabFrameAt(pcv,t).then(img=>{clipStartGhosts[nc]=img;renderClipOverview();})
      .catch(()=>{});
  })();
  renderDialog();
  toast("Writing a smooth continuation for Clip "+nc+"\u2026");
  try{
    const r=await fetch("/api/extendscene",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        prev_scene:prevScene,
        story:_storySoFar(nc),
        instructions:(nextCfg.instructions||"").trim(),
        want_dialogue:true,          // suggest dialogue but don't force it
        image:prevCtxImg})});
    const j=await r.json();
    if(!j.ok){
      toast(j.error||"Couldn't write the continuation \u2014 edit the clip manually.",true);
      openClipModal(nc);return;
    }
    // Continuation scene describes the ongoing action.
    if(j.scene)clipPromptOverrides[nc]=(j.scene||"").trim()
      +(j.camera?" Camera: "+j.camera.trim():"");
    // Add any suggested speakers/lines (may be none — action-only).
    (Array.isArray(j.speakers)?j.speakers:[]).forEach(s=>{
      const nm=(s.name||"SPEAKER").toUpperCase().slice(0,24);
      if(!speakers.some(x=>x.name===nm))
        speakers.push({name:nm,voice:s.voice||"a natural adult voice, medium pitch, calm pace"});
    });
    dialog=dialog.filter(l=>(l.clip||1)!==nc);   // replace this clip's lines
    const newLines=(Array.isArray(j.lines)?j.lines:[]).filter(l=>l&&(l.text||"").trim());
    if(newLines.length){
      newLines.forEach(l=>{
        const sp=(l.speaker||((speakers[0]||{}).name)||"SPEAKER").toUpperCase();
        if(!speakers.some(x=>x.name===sp))
          speakers.push({name:sp,voice:"a natural adult voice, medium pitch, calm pace"});
        dialog.push({speaker:sp,text:String(l.text).trim(),clip:nc});
      });
    }else{
      // action-only continuation: keep an empty placeholder line so the
      // clip exists, but it renders as silent/ambient.
      dialog.push({speaker:(speakers[0]||{name:"SPEAKER"}).name,text:"",clip:nc});
    }
    delete clipFrameOverrides[nc];
    renderSpeakers();renderDialog();
    const nl=dialog.filter(l=>(l.clip||1)===nc&&(l.text||"").trim()).length;
    toast("Clip "+nc+" continues the shot"
      +(nl?(" with "+nl+" line"+(nl===1?"":"s")):" (action continues, no dialogue)")
      +" \u2014 review & Generate.");
    openClipModal(nc);
  }catch(e){toast("Error: "+e,true);openClipModal(nc);}
}
// Commit the staged top inputs (Clip 1 image + scene) as the next clip on
// the timeline, then clear the staging fields for the next clip. This makes
// clips EXPLICIT rather than auto-present, fixing the "is this input or a
// committed clip?" dissonance.
function addToTimeline(){
  const scene=($("prompt").value||"").trim();
  const stagedImg=imgData||null;
  const stagedEnd=_stagedEnd||null;
  if(!scene && !stagedImg){
    toast("Add an image or write a scene first.",true);return;
  }
  // Sidebar default = start a fresh timeline (replace). The user opts into
  // appending with the checkbox. We only wipe when there's something to
  // wipe AND append is off, and we preserve the staged inputs being created.
  const append=($("appendTl")||{}).checked;
  if(!append && _committedClips>0){
    _resetTimelineState();   // silent: replace by default (append box is the opt-in)
  }
  if(!speakers.length)addSpeaker();
  // ── AUTO-SPLIT: a control video longer than the segment length becomes
  // several chained clips. Each gets a time-slice of the control video,
  // the same prompt, and a placeholder last frame (the real last frame is
  // grabbed at render time to chain continuously).
  const stagedCtrl=clipControlVideos[1]||null;
  const segS=parseFloat(($("v2vSeg")||{}).value)||20;
  if(stagedCtrl && clipControlDur[1] && clipControlDur[1] > segS+0.5){
    _autoSplitControl(scene,stagedCtrl,clipControlDur[1],segS,stagedImg);
    return;
  }
  const nc=_committedClips+1;      // next slot
  if(scene)clipPromptOverrides[nc]=scene;
  // Creation COPIES the staged inputs into the clip's own slots — after
  // this the timeline clip is independent of the sidebar (staging area).
  if(stagedImg)clipStartImages[nc]=stagedImg;
  if(stagedEnd)clipEndImages[nc]=stagedEnd;
  if(!dialog.some(l=>(l.clip||1)===nc))
    dialog.push({speaker:(speakers[0]||{name:"SPEAKER 1"}).name,text:"",clip:nc});
  _committedClips=nc;
  // Copy staged CONTROL forward to this clip (staging uses slot-1 maps),
  // then clear staging. After this the clip is fully independent.
  if(nc!==1 && clipControlVideos[1] && !clipControlVideos[nc]){
    clipControlVideos[nc]=clipControlVideos[1];
    clipControlType[nc]=clipControlType[1];
    clipControlStrength[nc]=clipControlStrength[1];
    clipCopySrcAudio[nc]=clipCopySrcAudio[1];
    clipControlDur[nc]=clipControlDur[1];
    delete clipControlVideos[1];delete clipControlType[1];
    delete clipControlStrength[1];delete clipCopySrcAudio[1];
    delete clipControlDur[1];
  }
  // Clear ALL staging (the clip owns its copies now).
  $("prompt").value="";
  imgData=null;_stagedEnd=null;
  const d=$("drop");if(d){d.className="dropzone";d.innerHTML="&#128247;&nbsp; Click or drop an image";}
  _renderC1Frames();
  renderSpeakers();renderClipOverview();_syncGenerateEnabled();
  const _atl=$("appendTl");if(_atl)_atl.checked=true;   // next clip appends
  openClipModal(nc);
}

// Split a long control video into N chained v2v clips. The control video's
// data URL is reused for every clip; the backend slices each clip's window
// by its position + per-clip duration (see _run_v2v_clip segmentation).
function _autoSplitControl(scene,ctrlUrl,dur,segS,stagedImg){
  const base=_committedClips;
  // Cut AT the segment length: full-length clips then the remainder
  // (20s setting on a 30s video = 20s + 10s, not 2x15s).
  const n=Math.ceil(dur/segS);
  for(let k=0;k<n;k++){
    const nc=base+k+1;
    const start=k*segS, len=Math.min(segS, dur-start);
    if(scene)clipPromptOverrides[nc]=scene;      // same prompt across all
    clipControlVideos[nc]=ctrlUrl;               // same source
    clipControlType[nc]=clipControlType[1]||"raw";
    clipControlStrength[nc]=clipControlStrength[1]||1;
    clipCopySrcAudio[nc]=(clipCopySrcAudio[1]!=null)?clipCopySrcAudio[1]:true;
    clipControlDur[nc]=len;                       // this clip's window length
    clipControlSplit[nc]={index:k,count:n,start:start,len:len,total:dur};
    if(k===0 && stagedImg){                       // first clip anchors identity
      clipStartImages[nc]=stagedImg;
    }else{
      clipStartPlaceholders[nc]=_extendPlaceholderFrame(nc);
      // Ghost: this clip's PREDICTED start = its slice's first frame.
      (function(cc,st){
        _grabFrameAt(ctrlUrl,st+0.03)
          .then(img=>{clipStartGhosts[cc]=img;renderClipOverview();})
          .catch(()=>{});
      })(nc,start);
    }
    if(!dialog.some(l=>(l.clip||1)===nc))
      dialog.push({speaker:(speakers[0]||{name:"SPEAKER 1"}).name,text:"",clip:nc});
  }
  _committedClips=base+n;
  // Clear ALL staging (clips own their copies now). For an APPEND split
  // the staged control lived in slot 1 — the split clips have their own
  // copies, so drop the staging one.
  if(base>0){delete clipControlVideos[1];delete clipControlType[1];
    delete clipControlStrength[1];delete clipCopySrcAudio[1];
    delete clipControlDur[1];}
  $("prompt").value="";
  imgData=null;_stagedEnd=null;
  const d=$("drop");if(d){d.className="dropzone";d.innerHTML="&#128247;&nbsp; Click or drop an image";}
  _renderC1Frames();renderSpeakers();renderClipOverview();_syncGenerateEnabled();
  const _atl2=$("appendTl");if(_atl2)_atl2.checked=true;
}
// Enable/disable Generate based on whether any clip is committed (LTX only).
function _syncGenerateEnabled(){
  const gb=$("genBtn");if(!gb)return;
  if(currentEngine!=="ltx"){gb.disabled=false;gb.style.opacity="";gb.title="";return;}
  const ready=_committedClips>0;
  gb.disabled=!ready;gb.style.opacity=ready?"":"0.5";
  gb.title=ready?"":"Add at least one clip to the timeline first.";
}
// NEW SCENE: generate a fresh starting image with gpt-image-2 from the
// previous clip's context, review it, then it becomes this clip's start.
async function addClipNewScene(){
  closeAddClip();
  const nc=Math.max(_committedClips,
    dialog.reduce((m,l)=>Math.max(m,l.clip||1),0))+1;   // next slot
  _pendingNewClip=null;         // clear any stale compose intent
  _composeTarget=null;          // make sure no image-compose flow is armed
  // Create an EMPTY, directly-editable clip. NO image generation happens
  // here — the start image is optional and set inside the editor only if
  // the user chooses generate/upload. A blank scene + one empty dialogue
  // line makes the clip show in the timeline and gives a clean canvas.
  if(!speakers.length)addSpeaker();
  clipPromptOverrides[nc]="";
  dialog.push({speaker:(speakers[0]||{name:"SPEAKER 1"}).name,text:"",clip:nc});
  // Continuation placeholder (real last frame is chained at render).
  if(nc>1)clipStartPlaceholders[nc]=_extendPlaceholderFrame(nc);
  _committedClips=nc;
  renderSpeakers();
  renderClipOverview();         // force the timeline to show the new card
  _syncGenerateEnabled();
  openClipModal(nc);
  toast("Clip "+nc+" added \u2014 edit it here, or \u2728 Auto-write to fill it from the story.");
}
// Two-stage compose: propose a prompt, let the user edit + attach images,
// then generate on demand. _nextCompose holds the working state.
let _nextCompose={context:"",refs:[],label:"",primary:""};
async function _openNextCompose(primaryImg,context,label,opts){
  opts=opts||{};
  _composeTarget=opts.target||null;
  const refs=[];
  if(primaryImg)refs.push(primaryImg);
  (opts.extraRefs||[]).forEach(r=>{if(r&&refs.indexOf(r)<0)refs.push(r);});
  _nextCompose={context:context||"",label:label||"image",
    primary:primaryImg||"",refs:refs,
    hasPrev:!!primaryImg&&opts.propose!==false};
  // Contextual title + labels so the modal reads correctly whether you're
  // creating a first frame, an end frame, or continuing a scene.
  $("nextComposeTitle").textContent=label?("Generate "+label):"Compose image";
  const rl=$("nextComposeRefLbl");
  if(rl)rl.innerHTML=_nextCompose.refs.length
    ? "Reference images <span class='c'>carried into the generation for consistency</span>"
    : "Reference images <span class='c'>optional \u2014 upload images to guide or keep consistency</span>";
  // Re-propose only makes sense when there's a source frame to propose from.
  const rp=$("nextComposeRepropose");
  if(rp)rp.style.display=_nextCompose.hasPrev?"inline-block":"none";
  $("nextComposeModal").style.display="flex";
  $("nextComposePrompt").value=opts.prompt||"";
  $("nextComposePrompt").placeholder=_nextCompose.hasPrev
    ? "Proposing\u2026" : "Describe the image you want\u2026";
  $("nextComposeCost").textContent="";
  _renderNextComposeImgs();
  _refreshTokBadge();
  // Auto-propose only when there's a source frame to build from. For a
  // fresh first frame the user writes their own prompt (no confusing
  // "Proposing…" with nothing to propose from).
  if(opts.propose!==false && _nextCompose.primary){await _nextRepropose();}
}
function closeNextCompose(){$("nextComposeModal").style.display="none";}
async function _nextRepropose(){
  const btn=$("nextComposeRepropose");if(btn){btn.disabled=true;}
  $("nextComposePrompt").placeholder="Proposing a prompt from the previous frame\u2026";
  try{
    const r=await fetch("/api/nextscene/propose",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:_nextCompose.primary,
        context:_nextCompose.context,
        instructions:(nextCfg.instructions||"").trim(),
        shot:nextCfg.shot||"auto"})});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Couldn't propose a prompt.",true);
      $("nextComposePrompt").placeholder="Couldn't propose \u2014 write your own prompt.";}
    else{$("nextComposePrompt").value=j.image_prompt||"";
      _nextCompose.intent=j.intent||"";}
  }catch(e){toast("Error: "+e,true);}
  if(btn)btn.disabled=false;
}
function _renderNextComposeImgs(){
  const box=$("nextComposeImgs");box.innerHTML="";
  _nextCompose.refs.forEach((src,i)=>{
    const d=document.createElement("div");d.className="nc-img";
    const tag=(i===0&&_nextCompose.hasPrev)?"<span class='nc-tag'>prev frame</span>":"";
    d.innerHTML="<img src='"+src+"'>"+tag
      +"<button class='nc-x' title='Remove'>&#10005;</button>";
    d.querySelector(".nc-x").onclick=()=>{_nextCompose.refs.splice(i,1);_renderNextComposeImgs();};
    box.appendChild(d);
  });
  // The composite trick only matters when there are 2+ references.
  const cb=$("nextCompositeBtn");if(cb)cb.style.display=(_nextCompose.refs.length>=2?"inline-block":"none");
}
// Combine the first two references into one side-by-side image. LTX pins a
// single frame, so two characters must share ONE reference to both stay
// consistent (composite-pin pattern from the LTX-2.3 dual-character notes).
function _compositeTwoRefs(){
  const a=_nextCompose.refs[0], b=_nextCompose.refs[1];
  if(!a||!b){toast("Need two reference images to combine.",true);return;}
  const ia=new Image(), ib=new Image();let done=0;
  const draw=()=>{
    if(++done<2)return;
    const H=768;
    const wa=Math.round(ia.width*(H/ia.height)), wb=Math.round(ib.width*(H/ib.height));
    const cv=document.createElement("canvas");cv.width=wa+wb;cv.height=H;
    const x=cv.getContext("2d");
    x.fillStyle="#000";x.fillRect(0,0,cv.width,cv.height);
    x.drawImage(ia,0,0,wa,H);x.drawImage(ib,wa,0,wb,H);
    const out=cv.toDataURL("image/png");
    _nextCompose.refs.splice(0,2,out);   // replace the two with the composite
    _renderNextComposeImgs();
    toast("Combined into one reference \u2014 both characters share a single pinned frame.");
  };
  ia.onload=draw;ib.onload=draw;ia.src=a;ib.src=b;
}
function _nextComposeAddFiles(files){
  [...(files||[])].forEach(f=>{
    const rd=new FileReader();
    rd.onload=()=>{if(_nextCompose.refs.length<6){_nextCompose.refs.push(rd.result);_renderNextComposeImgs();}
      else toast("Up to 6 reference images.",true);};
    rd.readAsDataURL(f);
  });
  $("nextComposeFile").value="";
}
async function _refreshTokBadge(){
  try{
    const j=await(await fetch("/api/tokens")).json();
    if(j.own_key)_applyTokBadge(null,null,true);
    else _applyTokBadge(j.tokens,j.tokens_per_gen||100,false);
  }catch(e){}
}
function _applyTokBadge(tokens,per,ownKey){
  const b=$("nextTokBadge");if(!b)return;
  if(ownKey){
    b.textContent="\ud83d\udd11 your OpenAI key";b.classList.remove("low");
    b.onclick=null;b.style.cursor="default";
    $("nextComposeCost").textContent="Generating on your own OpenAI key \u2014 no MissingLink tokens used.";
    return;
  }
  b.onclick=buyTokens;b.style.cursor="pointer";
  per=per||100;
  if(tokens===null||tokens===undefined){b.textContent="tokens";b.classList.remove("low");
    $("nextComposeCost").textContent="";return;}
  const imgs=Math.floor(tokens/per);
  b.textContent=tokens.toLocaleString()+" tokens ("+imgs+" images)";
  b.classList.toggle("low",tokens<per);
  $("nextComposeCost").innerHTML=(tokens<per)
    ?"<span style='color:#E5484D'>Not enough tokens \u2014 each image costs "+per+". <b onclick=\"buyTokens()\" style='color:var(--gold);cursor:pointer'>Buy tokens</b></span>"
    :"Each image costs "+per+" tokens. You have enough for "+imgs+".";
}
async function buyTokens(){
  toast("Opening secure checkout\u2026");
  try{
    const r=await fetch("/api/checkout",{method:"POST"});
    const j=await r.json();
    if(j.ok&&j.url){window.open(j.url,"_blank");}
    else toast(j.error||"Couldn't open checkout.",true);
  }catch(e){toast("Error: "+e,true);}
}
async function _nextComposeGenerate(){
  const prompt=($("nextComposePrompt").value||"").trim();
  if(!prompt){toast("Write a prompt first.",true);return;}
  closeNextCompose();
  try{
    const r=await fetch("/api/nextscene/generate",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image_prompt:prompt,intent:_nextCompose.intent||"",
        context:_nextCompose.context,quality:nextCfg.quality||"medium",
        images:_nextCompose.refs})});
    const j=await r.json();
    if(!j.ok||!j.job_id){toast(j.error||"Couldn't start generation.",true);return;}
    queue.push({id:j.job_id,status:"queued",progress:0,stage:"queued",
      prompt:_nextCompose.label||"Next-scene image",thumb:"",kind:"image"});
    renderQueue();
    toast("Generating "+(_nextCompose.label||"next scene")+" \u2014 progress in the jobs panel.");
  }catch(e){toast("Error: "+e,true);}
}
// Legacy single-shot path kept for the old auto flow (unused by compose).
async function _startImageJob(payload,label){
  try{
    const r=await fetch("/api/nextscene",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)});
    const j=await r.json();
    if(!j.ok||!j.job_id){toast(j.error||"Couldn't start image build.",true);return;}
    queue.push({id:j.job_id,status:"queued",progress:0,
      stage:"queued",prompt:label||"Next-scene image",thumb:"",kind:"image"});
    renderQueue();
    toast("Queued: "+(label||"next-scene image")+" \u2014 building (~45s), progress in the jobs panel.");
  }catch(e){toast("Error: "+e,true);}
}
let _pendingNewClip=null;   // clip number awaiting an accepted new-scene image
// Regenerate from the correct source: the add-clip flow re-runs from the
// previous clip's context; the stage flow re-reads the stage's last frame.
function regenNextScene(){
  closeNextReview();
  // Reopen compose so the user can tweak the prompt / images and re-generate.
  $("nextComposeModal").style.display="flex";
  _refreshTokBadge();
}

/* ── Clip editor modal ── */
let _clipModalC=null;
function openClipModal(c){
  _clipModalC=c;
  $("clipModalTitle").textContent="Edit Clip "+c;
  // scene
  const ta=$("clipSceneTa");
  ta.value=clipSceneText(c);
  ta.oninput=()=>{clipPromptOverrides[c]=ta.value;_syncClipReset();};
  // start/end images
  _renderClipModalImgs();
  _renderClipV2V();
  // length
  renderClipModalLen();
  // dialogue
  renderClipModalDialog();
  _syncClipReset();
  $("clipModal").style.display="flex";
  setTimeout(()=>ta.focus(),50);
}
// The clip's effective start image (clip 1 = the uploaded image).
function _clipStartImageFor(c){
  // Real set image wins; else the ACTUAL predicted first frame (ghost) so
  // the user can see what the clip starts on; else the schematic extend
  // placeholder; else clip 1's uploaded image.
  if(clipStartImages[c])return clipStartImages[c];
  if(clipStartGhosts[c])return clipStartGhosts[c];
  if(c===1)return imgData||null;
  return clipStartPlaceholders[c]||null;
}
// True when the slot is showing a display-only placeholder (not a real
// image the user set) — used to keep edit/clear controls sensible.
function _clipStartIsPlaceholder(c){
  return !clipStartImages[c] && !!clipStartPlaceholders[c];
}
function _renderClipModalImgs(){
  const c=_clipModalC;if(c==null)return;
  const s=_clipStartImageFor(c), e=clipEndImages[c]||null;
  const isPh=_clipStartIsPlaceholder(c);
  const ss=$("clipStartSlot"),es=$("clipEndSlot");
  const isGhost=(!clipStartImages[c])&&!!clipStartGhosts[c];
  if(ss)ss.innerHTML=s?("<img src='"+s+"'>"+(isGhost?"<span class='nc-tag'>predicted start \u2014 edit to lock a character</span>":(isPh?"<span class='nc-tag'>placeholder \u2014 uses real last frame at render</span>":""))):"<span class='cf-empty'>no start image \u2014 chains from the previous clip</span>";
  if(es)es.innerHTML=e?"<img src='"+e+"'>":"<span class='cf-empty'>no end frame</span>";
  const sc=$("clipStartClear");if(sc)sc.style.display=(clipStartImages[c]?"inline-block":"none");
  const ec=$("clipEndClear");if(ec)ec.style.display=(e?"inline-block":"none");
  // Edit only applies to a REAL image (not the extend placeholder).
  const se=$("clipStartEdit");if(se)se.style.display=(s&&(!isPh||isGhost)?"inline-block":"none");
  const ee=$("clipEndEdit");if(ee)ee.style.display=(e?"inline-block":"none");
}
// Re-edit a clip's image with a text instruction (gpt-image-2 edit).
// Open the full compose modal (prompt + reference images + generate)
// targeting this clip's start or end slot. Seeds references with the
// current slot image and the clip's other frame for consistency.
// ── Video-to-video handlers ──
function _renderClipV2V(){
  const c=_clipModalC;if(c==null)return;
  const cv=clipControlVideos[c]||null;
  const slot=$("clipV2VSlot"),opts=$("clipV2VOpts"),clr=$("clipV2VClear");
  if(cv){
    // Preview: seek to this clip's slice start so you see where it begins.
    const sp0=clipControlSplit[c];
    const seekT=(sp0&&sp0.start!=null)?sp0.start:0;
    slot.innerHTML="<video src='"+cv+"#t="+seekT.toFixed(2)+"' muted "
      +"preload='metadata' style='width:100%;height:100%;object-fit:cover'></video>";
    // If we don't yet have a visible start frame, grab the slice's first
    // frame as the ghost so the START IMAGE slot shows the real content.
    if(!clipStartImages[c] && !clipStartGhosts[c]){
      _grabFrameAt(cv, seekT+0.03).then(img=>{
        if(img){clipStartGhosts[c]=img;
          try{_renderClipModalImgs();}catch(e){}renderClipOverview();}
      }).catch(()=>{});
    }
    opts.style.display="block";clr.style.display="inline-block";
    $("clipV2VType").value=clipControlType[c]||"raw";
    const st=(clipControlStrength[c]!=null?clipControlStrength[c]:1);
    $("clipV2VStrength").value=st;$("clipV2VStrengthV").textContent=(+st).toFixed(2);
    $("clipV2VCopyAudio").checked=!!clipCopySrcAudio[c];
    const fsb=$("clipFaceSplit");if(fsb)fsb.style.display="block";
  }else{
    slot.innerHTML="<span class='cf-empty'>+ upload a control video</span>";
    opts.style.display="none";clr.style.display="none";
    const fsb=$("clipFaceSplit");if(fsb)fsb.style.display="none";
  }
  const sl=$("clipV2VSlice");
  if(sl){const sp=clipControlSplit[c];
    if(sp){const a=(sp.start!=null?sp.start:sp.index*sp.per).toFixed(1),
      b=((sp.start!=null?sp.start:sp.index*sp.per)+(sp.len!=null?sp.len:sp.per)).toFixed(1);
      sl.style.display="block";
      sl.textContent="\ud83c\udf9e slice "+(sp.index+1)+"/"+sp.count
        +" \u00b7 drives "+a+"\u2013"+b+"s of the source video (cut losslessly at render)";}
    else sl.style.display="none";}
}
function _clipV2VSet(files){
  const c=_clipModalC;if(c==null)return;
  const f=(files||[])[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=async()=>{
    clipControlVideos[c]=rd.result;
    if(clipControlType[c]==null)clipControlType[c]="raw";
    if(clipControlStrength[c]==null)clipControlStrength[c]=1;
    if(clipCopySrcAudio[c]==null)clipCopySrcAudio[c]=true;   // default: keep original audio
    _probeVideoDur(rd.result).then(d=>{if(d){clipControlDur[c]=d;
      try{renderClipModalLen();}catch(e){}renderClipOverview();}});
    // Auto-extract the control video's first frame -> this clip's start
    // image, so the user can edit it to reskin while keeping the motion.
    toast("Extracting the video's first frame\u2026");
    try{
      const frame=await _grabFirstFrame(rd.result);
      if(frame){clipStartImages[c]=frame;delete clipStartPlaceholders[c];
        if(c===1)_setStartImage(frame);}
    }catch(e){}
    _renderClipV2V();_renderClipModalImgs();renderClipOverview();
    toast("Control video set \u2014 its first frame is now the start image (edit it to reskin).");
  };
  rd.readAsDataURL(f);
  const inp=$("clipV2VFile");if(inp)inp.value="";
}
function _clipV2VClear(){
  const c=_clipModalC;if(c==null)return;
  delete clipControlVideos[c];delete clipControlType[c];
  delete clipControlStrength[c];delete clipCopySrcAudio[c];delete clipControlDur[c];
  try{renderClipModalLen();}catch(e){}
  _renderClipV2V();renderClipOverview();
}
function _clipV2VTypeChange(){
  const c=_clipModalC;if(c==null)return;
  clipControlType[c]=$("clipV2VType").value;
}
function _clipV2VStrengthChange(v){
  const c=_clipModalC;if(c==null)return;
  clipControlStrength[c]=parseFloat(v);$("clipV2VStrengthV").textContent=(+v).toFixed(2);
}
function _clipV2VAudioChange(on){
  const c=_clipModalC;if(c==null)return;
  clipCopySrcAudio[c]=!!on;
}
// Grab the FIRST frame of a video (data URL) as an image data URL.
function _grabFirstFrame(url){
  return new Promise((res,rej)=>{
    const v=document.createElement("video");v.preload="auto";v.muted=true;
    let done=false;
    const grab=()=>{if(done)return;done=true;
      try{const cv=document.createElement("canvas");
        cv.width=v.videoWidth||768;cv.height=v.videoHeight||432;
        cv.getContext("2d").drawImage(v,0,0,cv.width,cv.height);
        res(cv.toDataURL("image/png"));}catch(e){rej(e);}};
    v.onloadeddata=()=>{v.currentTime=0;};
    v.onseeked=grab;v.onerror=()=>rej("load failed");
    setTimeout(()=>{if(!done)try{grab();}catch(e){rej(e);}},3000);
    v.src=url;
  });
}
function _clipModalGenImg(which){
  const c=_clipModalC;if(c==null)return;
  const cur=(which==="start")?_clipStartImageFor(c):(clipEndImages[c]||null);
  const other=(which==="start")?(clipEndImages[c]||null):_clipStartImageFor(c);
  const extra=[];if(other)extra.push(other);
  _openNextCompose(cur||"",
    (clipSceneText(c)||"").trim(),
    "Clip "+c+" "+(which==="end"?"end frame":"start image"),
    {target:{kind:(which==="end"?"clip_end":"clip_start"),clip:c},
     extraRefs:extra, propose:!!cur});
}
async function _clipModalEditImg(which){
  const c=_clipModalC;if(c==null)return;
  const cur=(which==="start")?_clipStartImageFor(c):(clipEndImages[c]||null);
  if(!cur){toast("No image to edit.",true);return;}
  openEditImageModal({
    src:cur,
    title:"Edit Clip "+c+" "+(which==="end"?"end":"start")+" image",
    label:"Clip "+c+" "+which+" image",
    apply:(img)=>{
      if(which==="start"){clipStartImages[c]=img;delete clipStartPlaceholders[c];if(c===1)_setStartImage(img);}
      else clipEndImages[c]=img;
      _renderClipModalImgs();renderDialog();
    }});
}
function _clipModalSetImg(which,files){
  const c=_clipModalC;if(c==null)return;
  const f=(files||[])[0];if(!f)return;
  const rd=new FileReader();
  rd.onload=()=>{
    if(which==="start"){clipStartImages[c]=rd.result;delete clipStartPlaceholders[c];
      if(c===1)_setStartImage(rd.result);}   // clip 1 also updates the main image
    else clipEndImages[c]=rd.result;
    _renderClipModalImgs();renderDialog();
    toast("Clip "+c+" "+which+" image set.");
  };
  rd.readAsDataURL(f);
  const inp=$(which==="start"?"clipStartFile":"clipEndFile");if(inp)inp.value="";
}
function _clipModalClearImg(which){
  const c=_clipModalC;if(c==null)return;
  if(which==="start")delete clipStartImages[c];
  else delete clipEndImages[c];
  _renderClipModalImgs();renderDialog();
}
function closeClipModal(){
  $("clipModal").style.display="none";_clipModalC=null;
  renderDialog();   // refresh overview + plan with the edits
}
function _syncClipReset(){
  const c=_clipModalC;if(c==null)return;
  const custom=(clipPromptOverrides[c]!==undefined && clipPromptOverrides[c]!==null
                && clipPromptOverrides[c]!==$("prompt").value);
  $("clipSceneReset").style.display=custom?"inline-block":"none";
}
function resetClipScene(){
  const c=_clipModalC;if(c==null)return;
  delete clipPromptOverrides[c];
  $("clipSceneTa").value=$("prompt").value||"";
  _syncClipReset();
}
function renderClipModalLen(){
  const c=_clipModalC;const fps=parseInt($("fpsV").value,10)||24;
  const plan=clipPlan();const pl=plan.find(x=>x.clip===c);
  const frames=pl?pl.frames:((clipControlVideos[c]&&clipControlDur[c])
    ?Math.max(9,((Math.round(clipControlDur[c]*fps)-1)/8|0)*8+1)
    :_clipAutoFrames(0,fps));
  const inp=$("clipLenInp");inp.value=(frames/fps).toFixed(1);
  inp.onchange=()=>{
    let secs=Math.max(1,parseFloat(inp.value)||1);
    const recMax=_ltxMaxFrames(fps)/fps;   // ~20s recommended ceiling
    // Allow going past the recommended max, but be honest: LTX-2.3 was
    // trained for up to ~20s per clip. Beyond that it drifts badly and may
    // run out of VRAM. We warn, cap at a hard 60s safety limit, and proceed.
    if(secs>recMax){
      if(!inp._warned){
        toast("Heads up: LTX-2.3 is trained for ~"+recMax.toFixed(0)+"s per clip. "
          +"Longer single clips drift and may run out of memory \u2014 for long, "
          +"consistent video, chain several shorter clips instead.",true);
        inp._warned=true;
      }
      secs=Math.min(secs,60);              // don't let it go absurd (OOM guard)
    }else{inp._warned=false;}
    let f=Math.round(secs*fps);f=Math.max(9,((f-1)/8|0)*8+1);
    // snap to grid but DON'T clamp to the recommended max when the user
    // explicitly asked for longer.
    if(secs<=recMax)f=Math.min(_ltxMaxFrames(fps),f);
    clipFrameOverrides[c]=f;renderClipModalLen();
  };
  const overridden=clipFrameOverrides[c]!==undefined;
  $("clipLenReset").style.display=overridden?"inline-block":"none";
  const auto=pl?pl.auto:_clipAutoFrames(0,fps);
  const _src=(clipControlVideos[c]&&clipControlDur[c])?"control video":"dialogue";
  $("clipLenHint").textContent="auto from "+_src+" \u2248 "+(auto/fps).toFixed(1)+"s"
    +(overridden?" (overridden)":"");
}
function resetClipLen(){
  const c=_clipModalC;if(c==null)return;
  delete clipFrameOverrides[c];renderClipModalLen();
}
// Auto-write THIS clip's dialogue, continuing the story so far. Closes
// the modal, runs the clip-scoped Auto Prompt, which reopens it with the
// generated lines.
async function clipModalAutoDialog(){
  const c=_clipModalC;if(c==null)return;
  const b=$("clipAutoDlgBtn");
  if(b){b.disabled=true;b._orig=b.innerHTML;b.innerHTML="\u2728 Writing scene & dialogue\u2026";}
  await autoPromptForClip(c,{keepOpen:true});
  if(b){b.disabled=false;b.innerHTML=b._orig||"\u2728 Auto-write scene & dialogue";}
}
function clipModalAddLine(){
  const c=_clipModalC;if(c==null)return;
  if(!speakers.length)addSpeaker();
  dialog.push({speaker:speakers[0].name,text:"",clip:c});
  renderClipModalDialog();renderClipModalLen();
}
// Remove clip c: drop its lines, shift higher clips down, reindex the
// per-clip override maps and any start images so numbering stays clean.
function deleteClip(c){
  if(c==null)return;
  dialog=dialog.filter(l=>(l.clip||1)!==c);
  dialog.forEach(l=>{if((l.clip||1)>c)l.clip=(l.clip||1)-1;});
  delete clipPromptOverrides[c];delete clipFrameOverrides[c];delete clipStartImages[c];delete clipEndImages[c];delete clipStartPlaceholders[c];
  delete clipControlVideos[c];delete clipControlType[c];delete clipControlStrength[c];delete clipCopySrcAudio[c];delete clipControlDur[c];delete clipControlSplit[c];delete clipStartGhosts[c];
  [clipPromptOverrides,clipFrameOverrides,clipStartImages,clipEndImages,clipStartPlaceholders,clipControlVideos,clipControlType,clipControlStrength,clipCopySrcAudio,clipControlDur,clipControlSplit,clipStartGhosts].forEach(map=>{
    Object.keys(map).map(Number).sort((a,b)=>a-b).forEach(k=>{
      if(k>c){map[k-1]=map[k];delete map[k];}
    });
  });
  if(_committedClips>0)_committedClips--;
  _syncGenerateEnabled();
}
function deleteClipFromModal(){
  const c=_clipModalC;if(c==null)return;
  deleteClip(c);
  closeClipModal();
}
// Trash icon on the card. Clip 1 is the base clip — deleting it clears
// the whole storyboard back to a single base clip, so confirm.
function deleteClipCard(c){
  const _snap=_snapClips();
  const nClips=dialog.reduce((m,l)=>Math.max(m,l.clip||1),0);
  if(nClips<=1){
    dialog=[];clipPromptOverrides={};clipFrameOverrides={};clipStartImages={};clipEndImages={};clipStartPlaceholders={};clipControlVideos={};clipControlType={};clipControlStrength={};clipCopySrcAudio={};clipControlDur={};clipControlSplit={};clipStartGhosts={};
    _committedClips=0;
  }else{
    deleteClip(c);
  }
  renderSpeakers();renderDialog();_syncGenerateEnabled();
  toastUndo("Clip "+c+" removed.",()=>_restoreClips(_snap));
}
// Wipe the whole timeline back to a single empty opening clip (fresh start).
function _resetTimelineState(){
  // Timeline clips are independent of staging now — a reset wipes them
  // all. Staged inputs (imgData, prompt, staged control) are untouched;
  // for a fresh clip 1, they are copied in by the caller (Create clip).
  dialog=[];clipPromptOverrides={};clipFrameOverrides={};
  clipStartImages={};clipEndImages={};clipStartPlaceholders={};
  clipControlVideos={};clipControlType={};clipControlStrength={};
  clipCopySrcAudio={};clipControlDur={};clipControlSplit={};
  clipStartGhosts={};
  _committedClips=0;
}
function clearTimeline(){
  const _snap=_snapClips();
  const had=(dialog.length||_committedClips>0);
  dialog=[];clipPromptOverrides={};clipFrameOverrides={};
  clipStartImages={};clipEndImages={};clipStartPlaceholders={};clipControlVideos={};clipControlType={};clipControlStrength={};clipCopySrcAudio={};clipControlDur={};clipControlSplit={};clipStartGhosts={};
  _committedClips=0;
  renderSpeakers();renderDialog();_syncGenerateEnabled();
  const _atl3=$("appendTl");if(_atl3)_atl3.checked=false;
  if(had)toastUndo("Timeline cleared.",()=>_restoreClips(_snap));
}
function renderClipModalDialog(){
  const c=_clipModalC;const box=$("clipDlgList");box.innerHTML="";
  const idxs=[];dialog.forEach((l,i)=>{if((l.clip||1)===c)idxs.push(i);});
  if(!idxs.length){
    box.innerHTML="<div class='clip-empty'>No lines yet \u2014 add one below.</div>";
    return;
  }
  idxs.forEach(i=>{
    const l=dialog[i];
    const row=document.createElement("div");row.className="cd-row";
    const sel=document.createElement("select");sel.className="dlg-spk";
    speakers.forEach(s=>{const o=document.createElement("option");
      o.value=s.name;o.textContent=s.name;
      if(s.name===l.speaker)o.selected=true;sel.appendChild(o);});
    // Inline speaker actions right in the dropdown.
    const oNew=document.createElement("option");oNew.value="__new";oNew.textContent="\u2795 New speaker\u2026";
    const oRen=document.createElement("option");oRen.value="__rename";oRen.textContent="\u270e Rename this speaker\u2026";
    sel.append(oNew,oRen);
    sel.onchange=()=>{
      if(sel.value==="__new"){
        const nm=(prompt("New speaker name:","SPEAKER "+(speakers.length+1))||"").trim();
        if(nm){const up=nm.toUpperCase();
          if(!speakers.some(s=>s.name===up))
            speakers.push({name:up,voice:"a natural adult voice, medium pitch, clear timbre, calm pace"});
          l.speaker=up;}
        renderSpeakers();renderClipModalDialog();return;
      }
      if(sel.value==="__rename"){
        const cur=speakers.find(s=>s.name===l.speaker);
        const nm=(prompt("Rename speaker:",l.speaker)||"").trim();
        if(nm&&cur){const up=nm.toUpperCase(),old=cur.name;
          cur.name=up;dialog.forEach(x=>{if(x.speaker===old)x.speaker=up;});}
        renderSpeakers();renderClipModalDialog();return;
      }
      l.speaker=sel.value;
    };
    const ta=document.createElement("textarea");
    ta.value=l.text;ta.placeholder="What they say in this clip\u2026";
    ta.oninput=()=>{l.text=ta.value;_autoRecalcClipLen(c);renderClipModalLen();};
    // Per-line ✨ auto-write: fill THIS line (for its chosen speaker) from
    // the story so far. Useful after manually adding an empty line.
    const ai=document.createElement("button");ai.className="cd-ai";
    ai.innerHTML="\u2728";ai.title="Auto-write this line from the story";
    ai.onclick=async()=>{
      ai.disabled=true;ai.textContent="\u2026";
      await autoWriteLine(c,i);
      ai.disabled=false;ai.innerHTML="\u2728";
    };
    const x=document.createElement("button");x.className="lora-x";
    x.innerHTML="&#10005;";x.title="Remove line";
    x.onclick=()=>{dialog.splice(i,1);_autoRecalcClipLen(c);renderClipModalDialog();renderClipModalLen();};
    row.append(sel,ta,ai,x);box.appendChild(row);
  });
}
// If the clip's length isn't manually overridden, keep it auto — this
// makes the length track dialogue as lines are added/edited/removed.
// (A manual override stays put; the user can reset it to auto.)
function _autoRecalcClipLen(c){
  // no-op holder: clipPlan() already recomputes auto length live; this
  // exists so callers read clearly. Length re-renders via renderClipModalLen.
}
// Generate a single dialogue line for clip c, line index i, continuing
// the story so far and keeping the speaker the user selected.
async function autoWriteLine(c,i){
  const l=dialog[i];if(!l)return;
  const startImg=clipStartImages[c]||imgData||null;
  const story=_storySoFar(c);
  // Lines already in this clip (context for a natural next line).
  const here=dialog.filter(x=>(x.clip||1)===c && x!==l && (x.text||"").trim())
    .map(x=>x.speaker+': "'+x.text.trim()+'"').join("  ");
  const speaker=l.speaker||"SPEAKER";
  toast("Writing "+speaker+"'s line\u2026");
  try{
    const r=await fetch("/api/autoline",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:startImg,speaker:speaker,
        scene:(clipSceneText(c)||"").trim(),
        lines_here:here,story:story,
        instructions:(autoCfg.instructions||"").trim(),
        context:(autoCfg.context||"").trim()})});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Couldn't write the line.",true);return;}
    l.text=(j.line||"").trim();
    delete clipFrameOverrides[c];   // recompute length from the new dialogue
    _autoRecalcClipLen(c);
    renderClipModalDialog();renderClipModalLen();
    toast(l.text?("Wrote "+speaker+"'s line."):"No line generated.");
  }catch(e){toast("Error: "+e,true);}
}
/* Build the per-clip LTX prompts. Each clip gets: the Scene + Characters
   + music-suppression rules, then ONLY the dialogue lines assigned to it.
   Continuity note is added for clips after the first so the model keeps
   the same characters/setting as the previous clip's last frame. Returns
   [{prompt, frames}] ready for the backend sequence. */
// LTX-2.3 distilled drifts outfits toward white unless the colour token is
// glued to each clothing noun and emphasised. We UPPERCASE colour words in
// a character's appearance so they read as strong, repeated anchors
// ("BLACK velvet gown", "RED dress") — per the LTX2.3 dual-character notes.
function _reinforceColors(look){
  if(!look)return look;
  const colors=["black","white","red","blue","green","yellow","gold","golden",
    "silver","grey","gray","brown","purple","violet","pink","orange","navy",
    "crimson","scarlet","teal","beige","tan","ivory","cream","maroon","olive",
    "turquoise","magenta","charcoal","emerald","burgundy"];
  let out=look;
  colors.forEach(c=>{
    out=out.replace(new RegExp("\\b"+c+"\\b","gi"),c.toUpperCase());
  });
  return out;
}
function _ltxSceneHeader(sceneText,hasDialogue){
  let out="Scene: "+((sceneText!==undefined?sceneText:$("prompt").value)||"").trim();
  if(!/[.!?]$/.test(out))out+=".";
  // Consistent CHARACTER BLOCK: LTX keeps characters consistent across
  // clips best when the SAME fixed description is repeated every clip
  // (image conditioning + a stable prompt block). Colour tokens are
  // uppercased so outfits don't drift toward white on the distilled model.
  if(hasDialogue && speakers.length){
    out+=" Characters (keep identical in every shot, same face, hair, and exact wardrobe colours): "+speakers.map(s=>
      "The "+s.name.toUpperCase()+" \u2014 "
      +(s.look?_reinforceColors(s.look.trim().replace(/\.$/,""))+"; ":"")
      +"voice: "+(s.voice||"a natural adult voice")+".").join(" ");
  }
  // Audio rule adapts: with dialogue, only the speaking voice + room tone;
  // without dialogue, only quiet natural ambience. Music is ALWAYS off —
  // LTX reads audio cues from text and adds a score if music is implied,
  // so we suppress it explicitly and forcefully every clip.
  out+=hasDialogue
    ? " AUDIO: the ONLY sounds are the characters' speaking voices and quiet natural room tone. Absolutely NO music, NO background score, NO soundtrack, NO singing, NO humming, NO instruments, NO melodic underscore of any kind."
    : " AUDIO: the ONLY sound is quiet natural ambience appropriate to the scene. NO dialogue, NO music, NO background score, NO soundtrack, NO singing, NO humming, NO instruments, NO melodic underscore of any kind.";
  return out;
}
function buildLtxClips(){
  const plan=clipPlan();
  return plan.map((pl,idx)=>{
    const hasDlg=pl.lines.length>0;
    // Each clip uses its own scene text (override or main prompt).
    let out=_ltxSceneHeader(clipSceneText(pl.clip),hasDlg);
    if(idx>0)out+=" This shot begins EXACTLY where the previous one ended, on the same frame, and continues in one unbroken take: the same characters, wardrobe, setting, lighting, camera angle and framing, with the motion and pacing carrying on smoothly from the previous moment (no cut, no jump, no reset).";
    if(hasDlg){
      out+=" Storyboard:";
      pl.lines.forEach(l=>{
        out+=' The '+l.speaker.toUpperCase()+' says: "'
          +l.text.trim().replace(/"/g,"'")
          +'", lips and jaw moving in precise sync with the speech audio, blinking naturally.';
      });
      const talking=new Set(pl.lines.map(l=>l.speaker.toUpperCase()));
      speakers.map(s=>s.name.toUpperCase()).filter(n=>!talking.has(n))
        .forEach(n=>{
          out+=" THE "+n+" IS COMPLETELY SILENT in this shot: mouth closed, lips not moving at all, listening and reacting only with the eyes.";
        });
    }
    const cv=clipControlVideos[pl.clip]||null;
    return {prompt:out,frames:pl.frames,frames_overridden:!!pl.overridden,
            start_image:(clipStartImages[pl.clip]||null),
            end_image:(clipEndImages[pl.clip]||null),
            control_split:(clipControlSplit[pl.clip]||null),
            control_video:cv,
            control_type:(cv?(clipControlType[pl.clip]||"raw"):null),
            control_strength:(cv?(clipControlStrength[pl.clip]!=null?clipControlStrength[pl.clip]:1.0):null),
            copy_source_audio:(cv?!!clipCopySrcAudio[pl.clip]:false)};
  });
}
// Back-compat single-prompt builder (used by the prompt preview).
function buildLtxPrompt(){
  const c=buildLtxClips();
  return c.length?c[0].prompt:_ltxSceneHeader();
}
// Build a compact "story so far" from the clips BEFORE clip `upto`, so
// Auto Prompt can continue the narrative instead of starting fresh.
function _storySoFar(upto){
  const plan=clipPlan();
  const parts=[];
  plan.forEach(pl=>{
    if(pl.clip>=upto)return;
    const scene=(clipSceneText(pl.clip)||"").trim();
    const lines=dialog.filter(l=>(l.clip||1)===pl.clip&&(l.text||"").trim())
      .map(l=>l.speaker+': "'+l.text.trim()+'"');
    let seg="Clip "+pl.clip+": "+scene;
    if(lines.length)seg+=" Dialogue \u2014 "+lines.join("  ");
    parts.push(seg);
  });
  return parts.join("\n");
}
// Generate dialogue for a SPECIFIC clip (used by New scene), continuing
// the story so far. Uses the clip's start image + prior-clip context.
// A neutral placeholder frame (data URL) for a clip whose real starting
// frame hasn't been generated yet — gives the vision call a valid image
// while signalling that the true frame is still to come.
// Display-only placeholder shown in an EXTEND clip's start slot. Makes
// clear the real frame comes from the previous clip at render time.
// Grab the final frame of a rendered video as a data URL (canvas capture).
// Seeks slightly before the very end to avoid the LTX tail-smear.
let clipStartGhosts={};   // {clip: dataURL} — PREDICTED start frame (display/edit aid)
function _grabFrameAt(url,t){
  return new Promise((res,rej)=>{
    const v=document.createElement("video");v.preload="auto";v.muted=true;
    let done=false;
    const grab=()=>{if(done)return;done=true;
      try{const cv=document.createElement("canvas");
        cv.width=v.videoWidth||768;cv.height=v.videoHeight||432;
        cv.getContext("2d").drawImage(v,0,0,cv.width,cv.height);
        res(cv.toDataURL("image/png"));}catch(e){rej(e);}};
    v.onloadeddata=()=>{v.currentTime=Math.max(0,Math.min(t,(v.duration||t)-0.03));};
    v.onseeked=grab;
    v.onerror=()=>rej("video load failed");
    setTimeout(()=>{if(!done){try{grab();}catch(e){rej(e);}}},4000);
    v.src=url;
  });
}
function _grabLastFrame(url){
  return new Promise((res,rej)=>{
    const v=document.createElement("video");v.preload="auto";v.muted=true;
    v.crossOrigin="anonymous";
    let done=false;
    const grab=()=>{
      if(done)return;done=true;
      try{
        const cv=document.createElement("canvas");
        cv.width=v.videoWidth||768;cv.height=v.videoHeight||432;
        cv.getContext("2d").drawImage(v,0,0,cv.width,cv.height);
        res(cv.toDataURL("image/png"));
      }catch(e){rej(e);}
    };
    v.onloadeddata=()=>{
      // seek to ~0.15s before the end (skip smeary tail), else 0.
      const t=Math.max(0,(v.duration||0)-0.15);
      v.currentTime=isFinite(t)?t:0;
    };
    v.onseeked=grab;
    v.onerror=()=>rej("video load failed");
    setTimeout(()=>{if(!done){try{grab();}catch(e){rej(e);}}},4000); // safety
    v.src=url;
  });
}
function _extendPlaceholderFrame(c){
  const cv=document.createElement("canvas");cv.width=768;cv.height=432;
  const x=cv.getContext("2d");
  x.fillStyle="#12130f";x.fillRect(0,0,cv.width,cv.height);
  x.strokeStyle="#3a3a2a";x.lineWidth=2;x.setLineDash([8,6]);
  x.strokeRect(10,10,cv.width-20,cv.height-20);x.setLineDash([]);
  x.fillStyle="#c9a227";x.textAlign="center";
  x.font="600 24px system-ui,sans-serif";
  x.fillText("\u23e9 Extends Clip "+(c-1),cv.width/2,cv.height/2-16);
  x.font="15px system-ui,sans-serif";x.fillStyle="#8a8a6a";
  x.fillText("starts from that clip's real last frame",cv.width/2,cv.height/2+12);
  x.fillText("(captured automatically at render)",cv.width/2,cv.height/2+34);
  return cv.toDataURL("image/png");
}
function _placeholderFrame(c){
  const cv=document.createElement("canvas");cv.width=768;cv.height=432;
  const x=cv.getContext("2d");
  x.fillStyle="#141414";x.fillRect(0,0,cv.width,cv.height);
  x.strokeStyle="#333";x.lineWidth=2;x.strokeRect(8,8,cv.width-16,cv.height-16);
  x.fillStyle="#8a8a8a";x.textAlign="center";
  x.font="600 26px system-ui,sans-serif";
  x.fillText("Clip "+c,cv.width/2,cv.height/2-18);
  x.font="16px system-ui,sans-serif";x.fillStyle="#6a6a6a";
  x.fillText("first frame not yet generated",cv.width/2,cv.height/2+14);
  x.fillText("(continues the previous clip)",cv.width/2,cv.height/2+38);
  return cv.toDataURL("image/png");
}
async function autoPromptForClip(c,opts){
  opts=opts||{};
  // Start image: clip 1 = the uploaded image; later clips = their set
  // start image. If this clip continues from a scene that hasn't been
  // rendered/generated yet (no start image), fall back to a PLACEHOLDER
  // frame so Auto Prompt still has something to look at, and tell the
  // model it's a placeholder for a frame that will continue the story.
  let startImg=clipStartImages[c]||null;
  let placeholder=false;
  if(!startImg){startImg=_placeholderFrame(c);placeholder=true;}
  const endImg=clipEndImages[c]||null;
  const story=_storySoFar(c);
  // What's already in THIS clip: the scene text the user has, and any
  // dialogue lines. Auto-write should build on these, not ignore them.
  const curScene=(clipSceneText(c)||"").trim();
  const curLines=dialog.filter(l=>(l.clip||1)===c && (l.text||"").trim())
    .map(l=>l.speaker.toUpperCase()+': "'+l.text.trim()+'"');
  toast("Writing Clip "+c+"'s scene & dialogue\u2026");
  try{
    let ctx=(autoCfg.context||"").trim();
    if(curScene)ctx+="\n\nCURRENT SCENE TEXT for this clip (refine and build on it, keep its intent and specifics; don't discard what's here):\n"+curScene;
    if(curLines.length)ctx+="\n\nDIALOGUE ALREADY IN THIS CLIP (keep these lines and the speakers unless they clearly need adjusting; you may add to them):\n"+curLines.join("\n");
    if(story)ctx+="\n\nSTORY SO FAR (continue it naturally, keep the same characters and voices, do not repeat lines):\n"+story;
    if(placeholder)ctx+="\n\nNOTE: the attached image is a PLACEHOLDER — this clip's real starting frame hasn't been generated yet. It continues the previous clip. Write the scene and dialogue for what happens next based on the story, not the placeholder image.";
    if(endImg)ctx+="\n\nAn END FRAME is attached separately — the clip should LAND on it. Write the scene/motion so it plausibly arrives at that end frame.";
    const payload={image:startImg,engine:"ltx",
      instructions:(autoCfg.instructions||"").trim(),
      context:ctx.trim()};
    if(endImg)payload.end_image=endImg;
    const r=await fetch("/api/autoprompt",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Auto prompt failed.",true);return;}
    // Update this clip's scene, and add its speakers/lines (tagged to c).
    if(j.scene)clipPromptOverrides[c]=(j.scene||"").trim()
      +(j.camera?" Camera: "+j.camera.trim():"");
    (Array.isArray(j.speakers)?j.speakers:[]).forEach(s=>{
      const nm=(s.name||"SPEAKER").toUpperCase().slice(0,24);
      if(!speakers.some(x=>x.name===nm))
        speakers.push({name:nm,voice:s.voice||"a natural adult voice, medium pitch, calm pace"});
    });
    // Replace this clip's dialogue with the new lines.
    dialog=dialog.filter(l=>(l.clip||1)!==c);
    (Array.isArray(j.lines)?j.lines:[]).filter(l=>l&&(l.text||"").trim())
      .forEach(l=>{
        const sp=(l.speaker||((speakers[0]||{}).name)||"SPEAKER").toUpperCase();
        if(!speakers.some(x=>x.name===sp))
          speakers.push({name:sp,voice:"a natural adult voice, medium pitch, calm pace"});
        dialog.push({speaker:sp,text:String(l.text).trim(),clip:c});
      });
    delete clipFrameOverrides[c];   // re-estimate length from new dialogue
    renderSpeakers();renderDialog();
    const nl=dialog.filter(l=>(l.clip||1)===c).length;
    toast(nl?("Clip "+c+": scene + "+nl+" line"+(nl===1?"":"s")+" written from the story."):"Clip "+c+" scene written (add dialogue or Auto-write again).");
    // Refresh the editor in place if it's open, else open it.
    if(opts.keepOpen && _clipModalC===c){
      const ta=$("clipSceneTa");if(ta)ta.value=clipSceneText(c);
      renderClipModalDialog();renderClipModalLen();_renderClipModalImgs();_syncClipReset();
    }else{
      openClipModal(c);
    }
  }catch(e){toast("Error: "+e,true);}
}

async function autoPrompt(){
  if(!imgData){toast("Upload a starting image first.",true);return;}
  const b=$("autoBtn");b.disabled=true;b.textContent="Looking\u2026";
  try{
    const r=await fetch("/api/autoprompt",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:imgData,engine:currentEngine,
        instructions:(autoCfg.instructions||"").trim(),
        context:(autoCfg.context||"").trim()})});
    const j=await r.json();
    if(!j.ok){toast(j.error||"Auto prompt failed.",true);}
    else if(currentEngine==="ltx"){
      $("prompt").value=(j.scene||"").trim()
        +(j.camera?" Camera: "+j.camera.trim():"");
      // Let Auto Prompt fully decide the cast: it may return 0, 1, 2 or
      // more speakers. Replace the roster with exactly what it returned
      // (empty = a no-dialogue scene), don't keep stale speakers around.
      speakers=(Array.isArray(j.speakers)?j.speakers:[]).map(s=>({
        name:(s.name||"SPEAKER").toUpperCase().slice(0,24),
        voice:s.voice||"a natural adult voice, medium pitch, calm pace"}));
      // Same for lines — empty means the clip is ambient only.
      dialog=(Array.isArray(j.lines)?j.lines:[])
        .filter(l=>l&&(l.text||"").trim()).map(l=>({
          speaker:(l.speaker||((speakers[0]||{}).name)||"SPEAKER").toUpperCase(),
          text:String(l.text).trim(),clip:(parseInt(l.clip,10)||1)}));
      clipFrameOverrides={};   // fresh auto lengths for the new dialogue
      clipPromptOverrides={};  // scenes revert to the fresh base prompt
      // Any line naming a speaker not in the roster gets one added, so a
      // 3-speaker scene works even if only 2 were listed in "speakers".
      dialog.forEach(l=>{
        if(!speakers.some(s=>s.name===l.speaker))
          speakers.push({name:l.speaker,
            voice:"a natural adult voice, medium pitch, calm pace"});
      });
      renderSpeakers();renderDialog();
      const ns=speakers.length,nl=dialog.length;
      toast(nl
        ? ("Scene + "+ns+" speaker"+(ns===1?"":"s")+", "+nl+" line"+(nl===1?"":"s")+" \u2014 edit anything, then Generate.")
        : "Scene generated (no dialogue \u2014 ambient) \u2014 edit anything, then Generate.");
    }else{
      $("prompt").value=(j.scene||"").trim();
      toast("Scene prompt generated from your image.");
    }
  }catch(e){toast("Error: "+e,true);}
  b.disabled=false;b.innerHTML="\u2728 Auto Prompt";
}

/* ── Civitai LoRA search (ported from MissingLink SDXL studio) ── */
function fetchT(url,opts,ms){
  const c=new AbortController();const t=setTimeout(()=>c.abort(),ms||20000);
  return fetch(url,Object.assign({},opts||{},{signal:c.signal}))
    .finally(()=>clearTimeout(t));
}
let _civCursor=null,_civLoading=false,_civDone=false,_civActiveTag=null;
function civOpenSearch(){
  const m=$("civSearchModal");m.style.display="flex";
  // Make sure the base filter matches the current engine, and flag the
  // LTX caveat honestly.
  _populateCivBase(currentEngine);
  const note=$("civSearchNote");
  if(note){
    if(currentEngine==="ltx"){
      note.style.display="block";
      note.innerHTML="Browsing LTX (LTXV 2.3 / LTXV) LoRAs. Added LoRAs attach to the LTX model on your next render (the resident worker restarts once to load them).";
    }else{note.style.display="none";}
  }
  m.onclick=e=>{if(e.target===m)civCloseSearch();};
  $("civSearchGrid").innerHTML="";$("civSearchStatus").textContent="";
  _civCursor=null;_civDone=false;
  _civLoadTags();civRunSearch(true);
  const grid=$("civSearchGrid");
  grid.onscroll=()=>{
    if(grid.scrollTop+grid.clientHeight>=grid.scrollHeight-200)civRunSearch(false);
  };
  // Click the media / name / info badge to open the LoRA detail page.
  grid.onclick=(e)=>{
    if(!e.target.closest("[data-info]"))return;
    const card=e.target.closest(".civ-result");if(!card)return;
    const mid=card.getAttribute("data-mid");
    if(mid)openLoraDetail(mid);
  };
}
function civCloseSearch(){$("civSearchModal").style.display="none";}
let _loraDetailDl="";   // download_url of the currently open detail LoRA
function closeLoraDetail(){$("loraDetailModal").style.display="none";}
async function openLoraDetail(mid){
  const m=$("loraDetailModal");m.style.display="flex";
  m.onclick=e=>{if(e.target===m)closeLoraDetail();};
  $("loraDetailName").textContent="Loading\u2026";
  $("loraDetailBody").innerHTML="<div class='clip-empty'>Fetching details from Civitai\u2026</div>";
  $("loraDetailMeta").textContent="";_loraDetailDl="";
  const nsfw=$("civSearchNsfw")&&$("civSearchNsfw").checked?"true":"false";
  try{
    const r=await fetch("/api/civitai/model?id="+encodeURIComponent(mid)+"&nsfw="+nsfw);
    const j=await r.json();
    if(!j.ok){$("loraDetailBody").innerHTML="<div class='clip-empty'>Couldn't load: "+esc(j.error||"error")+"</div>";return;}
    _renderLoraDetail(j);
  }catch(e){$("loraDetailBody").innerHTML="<div class='clip-empty'>Network error loading details.</div>";}
}
function _renderLoraDetail(j){
  $("loraDetailName").textContent=j.name||"LoRA";
  $("loraDetailLink").href=j.civitai_url||"#";
  // Primary version = first (latest). Its download_url powers Add.
  const v=(j.versions&&j.versions[0])||{};
  _loraDetailDl=v.download_url||"";
  const applied=_civIsApplied(_loraDetailDl);
  $("loraDetailAdd").disabled=applied;
  $("loraDetailAdd").textContent=applied?"\u2713 Loaded":"+ Add LoRA";
  const st=j.stats||{};
  $("loraDetailMeta").innerHTML="by "+esc(j.creator||"?")+" \u00b7 "
    +(st.downloads||0).toLocaleString()+" downloads \u00b7 "
    +(st.thumbs_up||0).toLocaleString()+" \ud83d\udc4d \u00b7 "
    +(st.comments||0).toLocaleString()+" comments \u00b7 "
    +"<a href='"+(j.civitai_url||"#")+"' target='_blank' style='color:var(--gold)'>read comments on Civitai \u2197</a>";
  // Trigger words (copyable chips).
  const triggers=(v.triggers||[]);
  let trigHtml="";
  if(triggers.length){
    trigHtml="<div class='ld-sec'><div class='ld-lbl'>Trigger words <span class='c'>click to copy</span></div>"
      +"<div class='ld-trigs'>"+triggers.map(t=>
        "<button class='ld-trig' onclick=\"navigator.clipboard&&navigator.clipboard.writeText('"
        +String(t).replace(/'/g,"\\'")+"');toast('Copied: "+esc(String(t)).replace(/'/g,"")+"')\">"+esc(t)+"</button>"
      ).join("")+"</div></div>";
  }else{
    trigHtml="<div class='ld-sec'><div class='ld-lbl'>Trigger words</div><div class='hintline' style='margin:0'>None \u2014 this LoRA needs no trigger word.</div></div>";
  }
  // Sample gens grid (image + video), from the primary version.
  const samples=(v.samples||[]);
  let sampHtml="<div class='ld-sec'><div class='ld-lbl'>Sample generations</div>";
  if(samples.length){
    sampHtml+="<div class='ld-samples'>"+samples.map(s=>
      s.video
        ? "<video src='"+s.video+"' poster='"+s.thumb+"' autoplay muted loop playsinline></video>"
        : "<img loading='lazy' src='"+s.thumb+"'>"
    ).join("")+"</div>";
  }else{
    sampHtml+="<div class='hintline' style='margin:0'>No sample previews available.</div>";
  }
  sampHtml+="</div>";
  // Base model + version.
  const info="<div class='ld-sec'><div class='ld-lbl'>Details</div>"
    +"<div class='hintline' style='margin:0'>Version: "+esc(v.version_name||"?")
    +" \u00b7 Base model: "+esc(v.base_model||"?")
    +(j.tags&&j.tags.length?" \u00b7 Tags: "+j.tags.slice(0,8).map(esc).join(", "):"")+"</div></div>";
  // Description (Civitai returns HTML; insert as-is inside a scoped box).
  const desc=(j.description||"").trim();
  const descHtml=desc
    ?"<div class='ld-sec'><div class='ld-lbl'>Description</div><div class='ld-desc'>"+desc+"</div></div>"
    :"";
  $("loraDetailBody").innerHTML=sampHtml+trigHtml+info+descHtml;
}
function loraDetailAdd(){
  if(!_loraDetailDl){toast("No downloadable file for this LoRA.",true);return;}
  civPickLora(_loraDetailDl);
  $("loraDetailAdd").disabled=true;$("loraDetailAdd").textContent="\u2713 Loaded";
}
async function _civLoadTags(){
  const box=$("civSearchTags");
  if(box.dataset.loaded)return;
  try{
    const r=await fetchT("/api/civitai/tags",null,15000);
    const d=await r.json();
    if(d.ok){
      box.innerHTML=d.tags.slice(0,24).map(t=>
        "<span class='civ-tag-chip' onclick=\"civToggleTag(this,'"+
        (t.name||"").replace(/'/g,"")+"')\">"+esc(t.name)+"</span>").join("");
      box.dataset.loaded="1";
    }
  }catch(e){}
}
function civToggleTag(el,name){
  document.querySelectorAll("#civSearchTags .civ-tag-chip").forEach(c=>{
    if(c!==el)c.classList.remove("active");});
  if(_civActiveTag===name){_civActiveTag=null;el.classList.remove("active");}
  else{_civActiveTag=name;el.classList.add("active");}
  civRunSearch(true);
}
async function civRunSearch(reset){
  if(_civLoading)return;
  if(reset){_civCursor=null;_civDone=false;$("civSearchGrid").innerHTML="";}
  if(_civDone&&!reset)return;
  _civLoading=true;
  $("civSearchStatus").textContent="Searching\u2026";
  try{
    const payload={query:$("civSearchQuery").value.trim(),
      base_model:$("civSearchBase").value,sort:$("civSearchSort").value,
      nsfw:$("civSearchNsfw").checked,tag:_civActiveTag||undefined,
      cursor:_civCursor||undefined};
    const r=await fetchT("/api/civitai/search",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)},25000);
    if(!r.ok){
      let msg="Search failed (HTTP "+r.status+")";
      try{const ed=await r.json();if(ed&&ed.error)msg=ed.error;}catch(e){}
      $("civSearchStatus").textContent=msg;_civLoading=false;return;
    }
    let d;
    try{d=await r.json();}
    catch(e){$("civSearchStatus").textContent="Search returned a bad response.";
      _civLoading=false;return;}
    if(!d.ok){$("civSearchStatus").textContent=d.error||"Search failed.";
      _civLoading=false;return;}
    const grid=$("civSearchGrid");
    // Keep full item objects — the inline onclick can only carry strings.
    if(!window._civResultMap)window._civResultMap={};
    (d.items||[]).forEach(it=>{
      if(it&&it.download_url)window._civResultMap[it.download_url]=it;});
    grid.insertAdjacentHTML("beforeend",(d.items||[]).map(_civResultHtml).join(""));
    _civCursor=d.next_cursor;_civDone=!d.next_cursor;
    $("civSearchStatus").textContent=grid.children.length
      ?(_civDone?grid.children.length+" results":"Scroll for more\u2026")
      :"No results \u2014 try a different term, base filter, or toggle NSFW.";
  }catch(e){
    $("civSearchStatus").textContent=(e&&e.name==="AbortError")
      ?"Search timed out (Civitai slow). Try again or narrow the search."
      :"Network error during search.";
  }
  _civLoading=false;
}
function _civIsApplied(dl){
  return !!(window._loraUrls&&window._loraUrls.has(dl));
}
function _civResultHtml(it){
  const raw=it.thumb||"";
  const video=it.video||"";
  const proxied=raw?"/api/civitai/thumb?url="+encodeURIComponent(raw):"";
  const dl=(it.download_url||"").replace(/'/g,"");
  const nm=esc(it.name||"(untitled)");
  const applied=_civIsApplied(dl);
  let mediaHtml;
  if(video){
    // Video LoRA: autoplay a muted looping clip. If it fails, fall back to
    // the still poster, then to the proxied still, then to "no preview".
    mediaHtml=
      "<video src=\""+video+"\" poster=\""+raw+"\" autoplay muted loop playsinline "+
      "onerror=\"(function(v){var i=document.createElement('img');i.loading='lazy';"+
      "i.src=v.getAttribute('poster')||'';i.setAttribute('data-proxy','"+proxied+"');"+
      "i.setAttribute('data-stage','raw');i.onerror=function(){if(this.dataset.stage==='raw'){"+
      "this.dataset.stage='proxy';this.src=this.dataset.proxy;}else{this.onerror=null;"+
      "this.replaceWith(Object.assign(document.createElement('div'),{className:'noimg',"+
      "textContent:'no preview'}));}};v.replaceWith(i);})(this)\"></video>"+
      "<span class='vid-badge'>\u25b6</span>";
  }else if(raw){
    mediaHtml=
      "<img loading='lazy' src=\""+raw+"\" data-proxy=\""+proxied+"\" data-stage='raw' "+
      "onerror=\"(function(el){if(el.dataset.stage==='raw'){el.dataset.stage='proxy';"+
      "el.src=el.dataset.proxy;}else{el.onerror=null;el.replaceWith(Object.assign("+
      "document.createElement('div'),{className:'noimg',textContent:'no preview'}));}})(this)\">";
  }else{
    mediaHtml="<div class='noimg'>no preview</div>";
  }
  const appliedTag=applied?" <span style='color:var(--gold);font-weight:700'>\u2713 loaded</span>":"";
  const mid=it.model_id||"";
  return "<div class='civ-result"+(applied?" applied":"")+"' data-url=\""+dl+"\" data-mid=\""+mid+"\">"+
    "<div class='thumbwrap' data-info='1'>"+mediaHtml+"<span class='info-badge' data-info='1'>\u24d8 info</span></div>"+
    "<div class='meta'><div class='nm' data-info='1'>"+nm+"</div>"+
    "<div class='bm'>"+esc(it.base_model||"")+(it.nsfw?" \u00b7 NSFW":"")+appliedTag+"</div></div>"+
    "<button class='add'"+(applied?" disabled":" onclick=\"civPickLora('"+dl+"')\"")+">"+
    (applied?"\u2713 Loaded":"+ Add LoRA")+"</button></div>";
}
async function civPickLora(url){
  const it=(window._civResultMap&&window._civResultMap[url])||{};
  const card=document.querySelector("#civSearchGrid .civ-result[data-url=\""+url+"\"]");
  const btn=card?card.querySelector(".add"):null;
  if(btn){btn.disabled=true;btn.textContent="Downloading\u2026";}
  try{
    // No fetchT timeout here — Wan LoRAs can be hundreds of MB and the
    // add endpoint blocks until the download + attach finish.
    const r=await fetch("/api/loras/add",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({url:url,name:it.name||"",scale:1.0,engine:currentEngine})});
    const j=await r.json();
    if(j.ok){
      toast("LoRA added"+((it.triggers||[]).length?" \u2014 trigger words appended to prompt.":"."));
      // Append trigger words to the prompt so the LoRA actually fires.
      const trig=(it.triggers||[]).filter(Boolean);
      if(trig.length){
        const p=$("prompt");
        trig.forEach(t=>{
          if(p.value.toLowerCase().indexOf(t.toLowerCase())<0)
            p.value=p.value.replace(/\s*$/,"")+(p.value?", ":"")+t;
        });
      }
      await refreshLoras();
      _civRefreshApplied();
    }else{
      toast("LoRA failed: "+(j.error||"unknown"),true);
      if(btn){btn.disabled=false;btn.textContent="+ Add LoRA";}
    }
  }catch(e){
    toast("Error: "+e,true);
    if(btn){btn.disabled=false;btn.textContent="+ Add LoRA";}
  }
}
// Re-evaluate applied state of visible cards in place — keeps scroll
// position and avoids re-requesting thumbnails (matches the SDXL studio).
function _civRefreshApplied(){
  const grid=$("civSearchGrid");if(!grid)return;
  grid.querySelectorAll(".civ-result").forEach(card=>{
    const dl=card.getAttribute("data-url")||"";
    const applied=_civIsApplied(dl);
    card.classList.toggle("applied",applied);
    const b=card.querySelector(".add");
    if(b&&applied){b.disabled=true;b.removeAttribute("onclick");
      b.textContent="\u2713 Loaded";}
  });
}

/* ── generate -> queue ── */
function generate(){
  const mode=currentMode;
  if(currentEngine==="ltx" && !_committedClips){
    toast("Add at least one clip to the timeline first (\u2795 Add to timeline).",true);return;
  }
  if(currentEngine!=="ltx"){
    if(mode==="i2v" && !imgData){toast("Add a starting image first.",true);return;}
    if(mode==="flf2v" && (!imgData||!lastData)){toast("First-Last needs a first AND a last frame.",true);return;}
    if(mode==="vace" && !refData && !imgData){toast("Reference mode needs a reference image (or a start frame).",true);return;}
  }
  const prompt=$("prompt").value;
  const payload={mode:mode,prompt:prompt,negative_prompt:$("neg").value,
    steps:+$("stepsV").value||40,guidance:+$("guidV").value,flow_shift:+$("shiftV").value,
    frames:+$("framesV").value||81,fps:+$("fpsV").value||16,seed:+$("seed").value||0};
  let thumb;
  if(mode==="flf2v"){
    payload.image=imgData;payload.last_image=lastData;payload.profile="720P";payload.segments=1;
    thumb=imgData;
  }else if(mode==="vace"){
    if(refData)payload.reference_image=refData;
    if(imgData)payload.image=imgData;
    payload.vace_size=$("vaceSize").value;payload.segments=1;
    thumb=refData||imgData;
  }else{
    payload.image=imgData;payload.profile=$("profile").value;
    payload.segments=+$("segmentsV").value||1;
    thumb=imgData;
  }
  payload.engine=currentEngine;
  payload.vary_seed=$("varySeed").checked;
  if(currentEngine==="wan22"){
    payload.wan22_lightning=$("wan22Lightning").checked;
    payload.mode="i2v";   // Wan 2.2 is i2v only here
  }
  if(currentEngine==="ltx"){
    payload.segments=1;   // segments is a Wan concept; LTX uses clips
    // THE TIMELINE IS THE CONTRACT: everything rendered comes from the
    // committed clips — nothing in the staging controls affects it.
    const clips=buildLtxClips();
    // Deduplicate control sources: split clips share one video — send it
    // ONCE and let each clip reference a slice of it by index.
    const srcs=[];const seen=new Map();
    clips.forEach(cl=>{
      if(cl.control_video&&typeof cl.control_video==="string"){
        let i=seen.get(cl.control_video);
        if(i===undefined){i=srcs.length;srcs.push(cl.control_video);seen.set(cl.control_video,i);}
        cl.control_video=i;
      }});
    payload.control_sources=srcs;
    payload.image=(clips.length?(clips[0].start_image||null):null);
    payload.clips=clips;
    payload.prompt=clips.length?clips[0].prompt:buildLtxPrompt();
    payload.frames=clips.length?clips[0].frames:(+$("framesV").value||121);
    payload.ltx_strength=parseFloat($("ltxStrengthV").value)||0.9;
    payload.ltx_crf=parseInt($("ltxCrfV").value,10);
    payload.ltx_enhance=$("ltxEnhance").checked;
    payload.ltx_offload=$("ltxOffload").value;
    thumb=(clips.length&&clips[0].start_image)||clipStartGhosts[1]||thumb||null;
    payload.strip_music=$("stripMusic")&&$("stripMusic").checked;
    payload.v2v_seg_seconds=parseFloat(($("v2vSeg")||{}).value)||20;
    if(clips.length>1)toast("Queued a "+clips.length+"-clip sequence.");
  }
  fetch("/api/generate",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)})
    .then(r=>r.json()).then(o=>{
      queue.unshift({id:o.job_id,prompt:prompt,thumb:thumb,
        status:"queued",progress:0,stage:"queued"});
      renderQueue();toast("Queued for generation.");
    }).catch(e=>toast("Error: "+e,true));
}

function renderQueue(){
  $("qCount").textContent=queue.length;
  const box=$("queueBody");
  if(!queue.length){box.innerHTML="<div class='empty'>No active jobs.</div>";return;}
  box.innerHTML="";
  queue.forEach(q=>{
    const badge=q.status==="running"?"<span class='badge b-run'>run</span>"
      :q.status==="cancelling"?"<span class='badge b-err'>cancelling</span>"
      :q.status==="error"?"<span class='badge b-err'>err</span>"
      :"<span class='badge b-queue'>queued</span>";
    const d=document.createElement("div");d.className="q-item";
    // Image builds have no thumbnail — show a film icon tile instead.
    const thumbHtml=(q.kind==="image")
      ? "<div class='q-imgicon'>\ud83c\udfac</div>"
      : "<img src='"+q.thumb+"'>";
    d.innerHTML=thumbHtml+
      "<div class='q-meta'><div class='q-prompt'>"+esc(q.prompt)+"</div>"+
      "<div class='q-stat'>"+badge+"<span>"+esc(q.stage||q.status)+"</span></div>"+
      "<div class='q-bar'><i style='width:"+(q.progress||0)+"%'></i></div></div>"+
      "<button class='q-cancel' title='Cancel'>&#10005;</button>";
    d.querySelector(".q-cancel").addEventListener("click",ev=>{
      ev.stopPropagation();cancelJob(q.id);});
    box.appendChild(d);
  });
}

async function cancelJob(id){
  // Optimistic: drop it from the local queue immediately so the UI is
  // responsive, then tell the server to abort it (running jobs stop at
  // the next step; queued jobs bail when they reach the GPU).
  queue=queue.filter(x=>x.id!==id);renderQueue();
  try{await fetch("/api/cancel/"+id,{method:"POST"});}catch(e){}
  toast("Job cancelled.");
}

function renderHistory(){
  $("hCount").textContent=history.length;
  const box=$("historyBody");
  if(!history.length){box.innerHTML="<div class='empty'>Finished videos and generated images appear here. Click to view.</div>";return;}
  box.innerHTML="<div class='h-grid'></div>";
  const g=box.firstChild;
  history.forEach(h=>{
    const c=document.createElement("div");c.className="history-card";c.title=h.prompt||"";
    const isImg=h.kind==="image";
    c.innerHTML="<div class='history-card-thumb'><img src='"+(h.thumb||h.url||"")+"'>"
      +(isImg?"<div class='h-imgtag'>IMG</div>":"<div class='play'>&#9654;</div>")+"</div>";
    c.onclick=()=>{isImg?showImage(h):showVideo(h);};
    g.appendChild(c);
  });
  renderGallery();
}
// Add a generated image to History (shown alongside videos).
function _addImageToHistory(dataUrl,label){
  if(!dataUrl)return;
  history.unshift({id:"img_"+Date.now().toString(36),kind:"image",
    prompt:label||"Generated image",thumb:dataUrl,url:dataUrl,ts:Date.now()});
  renderHistory();
}
function showImage(h){
  $("viewer").innerHTML="<img src='"+h.url+"' style='max-width:100%;max-height:100%;object-fit:contain;display:block;margin:auto'>";
  if(h.url)$("dlBtn").href=h.url;else $("dlBtn").removeAttribute("href");
  $("stageTools").classList.add("show");
}

const STAGE_PH="<div class='ph'><span class='big'>&#127909;</span>Your video plays here, full size.<br>Add a <b>starting frame</b>, write a prompt, hit <b>Generate</b>.</div>";
function clearStage(){
  $("viewer").innerHTML=STAGE_PH;
  $("stageTools").classList.remove("show");
  $("dlBtn").removeAttribute("href");
}
async function showVideo(h){
  // Restored history entries carry no video data until first viewed —
  // the mp4 stays on the server and is fetched once, on demand.
  if(!h.url){
    try{const r=await(await fetch("/api/result/"+h.id)).json();h.url=r.result;}
    catch(e){}
    if(!h.url){toast("That video is no longer on the server (runtime restarted?).",true);return;}
  }
  $("viewer").innerHTML="<video src='"+h.url+"' controls autoplay loop muted></video>";
  $("dlBtn").href=h.url;
  _stageUrl=h.url;
  $("stageTools").classList.add("show");
}
let _stageUrl=null;
async function clipControlFaceSplit(){
  if(_clipModalC!=null) await _faceSplitClip(_clipModalC, $("clipFaceSplit"));
}
async function _c1FaceSplit(){
  await _faceSplitClip(1, $("c1FaceSplit"));
}
async function _faceSplitClip(c, btn){
  const cv=clipControlVideos[c];
  if(!cv){toast("Upload a control video first.",true);return;}
  const orig=btn?btn.innerHTML:"";
  if(btn){btn.disabled=true;btn.innerHTML="\u2699 Detecting faces\u2026";}
  try{
    const r=await(await fetch("/api/facesplit/start",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({video:cv,window:10})})).json();
    if(!r.ok){toast(r.error||"Could not start face detection.",true);return;}
    queue.unshift({id:r.job_id,status:"running",progress:0,
      stage:"detecting faces",kind:"facesplit",_splitClip:c,_splitCv:cv,
      prompt:"Detecting faces \u2014 clip "+c,thumb:"",ts:Date.now()});
    if(typeof renderQueue==="function")try{renderQueue();}catch(e){}
    try{closeClipModal();}catch(e){}
    toast("Detecting faces \u2014 watch the Jobs panel for progress.");
  }catch(e){toast("Face split failed.",true);}
  finally{if(btn){btn.disabled=false;btn.innerHTML=orig;}}
}
function _buildClipsFromShots(c, cv, data){
  const shots=(data&&data.shots)||[];
  if(!shots.length){toast("No faces found to split on.",true);return;}
  const scene=clipPromptOverrides[c]||"";
  const total=data.total||0;
  if(_committedClips<c)_committedClips=c;
  shots.forEach((s,k)=>{
    const nc=(k===0)?c:(_committedClips+1);
    clipControlVideos[nc]=cv;
    clipControlType[nc]=clipControlType[c]||"raw";
    clipControlStrength[nc]=clipControlStrength[c]||1;
    clipCopySrcAudio[nc]=(clipCopySrcAudio[c]!=null)?clipCopySrcAudio[c]:true;
    clipControlDur[nc]=s.len;
    clipControlSplit[nc]={index:k,count:shots.length,start:s.start,len:s.len,total:total};
    if(scene)clipPromptOverrides[nc]=scene;
    if(!dialog.some(l=>(l.clip||1)===nc))
      dialog.push({speaker:(speakers[0]||{name:"SPEAKER 1"}).name,text:"",clip:nc});
    if(k>0)clipStartPlaceholders[nc]=_extendPlaceholderFrame(nc);
    if(nc>_committedClips)_committedClips=nc;
    (function(cc,st){_grabFrameAt(cv,st+0.05).then(img=>{
      if(img){clipStartGhosts[cc]=img;renderClipOverview();}}).catch(()=>{});
    })(nc,s.start);
  });
  renderClipOverview();renderDialog();_renderC1Frames();_syncGenerateEnabled();
  toast(shots.length+" clip"+(shots.length===1?"":"s")+" created from the control video.");
}

async function faceAutoClip(){
  if(!_stageUrl){toast("No clip on the stage.",true);return;}
  const b=$("autoClipBtn");const orig=b.innerHTML;
  b.disabled=true;b.innerHTML="\u2699 Cutting\u2026";
  try{
    const r=await(await fetch("/api/faceautoclip",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({video:_stageUrl,window:10})})).json();
    if(r.ok&&r.zip){
      const a=document.createElement("a");
      a.href=r.zip;a.download="face_autoclips.zip";
      document.body.appendChild(a);a.click();a.remove();
      toast(r.count+" clip"+(r.count===1?"":"s")+" cut \u2014 zip has each clip + its first/last frames + manifest.json.");
    }else{
      toast(r.error||"No faces found to clip on.",true);
    }
  }catch(e){toast("Auto-clip failed.",true);}
  finally{b.disabled=false;b.innerHTML=orig;}
}
async function extractFaceKeyframes(){
  if(!_stageUrl){toast("No clip on the stage.",true);return;}
  const b=$("faceKfBtn");const orig=b.innerHTML;
  b.disabled=true;b.innerHTML="\u2699 Scanning\u2026";
  try{
    const r=await(await fetch("/api/facekeyframes",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({video:_stageUrl,window:10})})).json();
    if(r.ok&&r.zip){
      const a=document.createElement("a");
      a.href=r.zip;a.download="face_keyframes.zip";
      document.body.appendChild(a);a.click();a.remove();
      toast(r.count+" face frame"+(r.count===1?"":"s")+" saved \u2014 zip downloaded (frames + timestamps.json).");
    }else{
      toast(r.error||"No faces found.",true);
    }
  }catch(e){toast("Face-frame extraction failed.",true);}
  finally{b.disabled=false;b.innerHTML=orig;}
}
async function convertVR180(){
  if(!_stageUrl){toast("No clip on the stage to convert.",true);return;}
  const b=$("vr180Btn");const orig=b.innerHTML;
  b.disabled=true;b.innerHTML="\u2699 Converting\u2026";
  toast("Building 180\u00b0 VR (upscaling + projection)\u2014 this can take a minute.");
  try{
    const r=await(await fetch("/api/vr180",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({video:_stageUrl})})).json();
    if(r.ok&&r.url){
      const h={id:"vr180_"+Date.now(),prompt:"180\u00b0 VR mono "
        +(r.out_w||"")+"\u00d7"+(r.out_h||""),url:r.url,ts:Date.now()};
      history.unshift(h);showVideo(h);renderHistory();
      toast("180\u00b0 mono VR ready \u2014 download and open in your headset "
        +"as 180 monoscopic.");
    }else{
      toast("VR conversion failed: "+(r.error||"unknown"),true);
    }
  }catch(e){toast("VR conversion failed.",true);}
  finally{b.disabled=false;b.innerHTML=orig;}
}

/* ── single poller for all active jobs ──
   Guarded so only one pass runs at a time (status fetches can lag), and
   each finished job is moved to history exactly once (doneIds), which is
   what prevents the duplicate/"30 entries" pile-up. Status is now tiny;
   the heavy base64 video is fetched once, separately, on completion. */
let _polling=false;
const doneIds=new Set();
setInterval(async()=>{
  if(_polling)return;
  _polling=true;
  try{
    const active=queue.filter(q=>q.status==="queued"||q.status==="running"
      ||q.status==="cancelling");
    let running=null,changed=false;
    for(const q of active){
      let j;try{j=await(await fetch("/api/status/"+q.id)).json();}catch(e){continue;}
      if(q.status!==j.status||q.progress!==(j.progress||0))changed=true;
      q.status=j.status;q.progress=j.progress||0;q.stage=j.stage||"";
      // Live clip-by-clip playback: when a new partial clip finishes,
      // fetch and play it on the stage so the movie builds before your eyes.
      if(j.status==="running" && (j.partial_index||0) > (q._partIdx||0)){
        q._partIdx=j.partial_index;
        (async()=>{try{
          const pr=await(await fetch("/api/partial/"+q.id)).json();
          if(pr&&pr.partial){
            showVideo({id:q.id+"_p"+pr.index,prompt:(q.prompt||"")
              +" (clip "+pr.index+"/"+pr.total+")",url:pr.partial,ts:Date.now()});
            toast("Clip "+pr.index+"/"+pr.total+" done \u2014 playing on the stage.");
          }
        }catch(e){}})();
      }
      if(j.status==="done"){
        if(!doneIds.has(q.id)){
          doneIds.add(q.id);
          if(q.kind==="qwenedit"){
            let img=null;
            try{const rr=await(await fetch("/api/result/"+q.id)).json();img=rr.result;}
            catch(e){}
            if(img){
              if(typeof q._applyFn==="function")try{q._applyFn(img);}catch(e){}
              _addImageToHistory(img,q._applyLabel||"Qwen edit");
              toast("Qwen edit applied.");
            }else toast("Qwen edit returned no image.",true);
          }else if(q.kind==="facesplit"){
            let data=null;
            try{const rr=await(await fetch("/api/result/"+q.id)).json();
              data=JSON.parse(rr.result);}catch(e){}
            if(data)_buildClipsFromShots(q._splitClip,q._splitCv,data);
            else toast("Face detection finished but returned nothing.",true);
          }else if(q.kind==="faceswap"){
            let img=null;
            try{const rr=await(await fetch("/api/result/"+q.id)).json();img=rr.result;}
            catch(e){}
            if(img&&q._clip!=null){
              clipStartImages[q._clip]=img;
              delete clipStartPlaceholders[q._clip];delete clipStartGhosts[q._clip];
              try{renderClipOverview();}catch(e){}
              history.unshift({id:q.id,prompt:"Face swap \u2014 clip "+q._clip,
                url:img,thumb:img,ts:Date.now(),isImage:true});
              try{renderHistory();}catch(e){}
              toast("Clip "+q._clip+": character applied.");
            }else{
              toast("Face swap finished but returned no image.",true);
            }
          }else if(q.kind==="image"){
            // Image build: fetch the payload and open the review modal.
            let payload=null;
            try{const rr=await(await fetch("/api/result/"+q.id)).json();
              payload=JSON.parse(rr.result);}catch(e){}
            if(payload&&payload.image){
              _nextResult=payload;_showNextReview(payload);
              if(payload.own_key)_applyTokBadge(null,null,true);
              else if(payload.tokens!==undefined)_applyTokBadge(payload.tokens,payload.tokens_per_gen||100,false);
            }
            else{toast("Image build finished but returned no image.",true);}
          }else{
            let url=null;
            try{const rr=await(await fetch("/api/result/"+q.id)).json();url=rr.result;}
            catch(e){}
            if(url){const h={id:q.id,prompt:q.prompt,thumb:q.thumb,url:url,ts:Date.now()};
              history.unshift(h);showVideo(h);renderHistory();}
          }
        }
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="error"){
        if(j.error==="insufficient_tokens"){
          _applyTokBadge(j.tokens||0,j.tokens_per_gen||100);
          toast("You're out of tokens for image generation \u2014 opening the token store.",true);buyTokens();
        }else{
          toast("Job failed: "+(j.error||"unknown"),true);
        }
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="cancelled"){
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="running"){running=q;}
    }
    const ss=$("stage_status");
    if(running){$("fill").style.width=(running.progress||0)+"%";
      ss.textContent=running.stage||"";ss.classList.add("show");
      // Rendering never blocks new submissions — make that obvious.
      $("genBtn").innerHTML="\u2726 Queue Next Video";}
    else{$("fill").style.width="0%";ss.classList.remove("show");
      $("genBtn").innerHTML="\u2726 Generate Video";}
    if(changed)renderQueue();
  }finally{_polling=false;}
},800);

/* ── gallery modal ── */
function renderGallery(){
  const g=$("ovGrid");if(!g)return;
  $("ovCount").textContent=history.length;
  if(!history.length){g.innerHTML="<div class='empty'>No videos yet this session.</div>";return;}
  g.innerHTML="";
  history.forEach(h=>{
    const c=document.createElement("div");c.className="ov-card";
    c.innerHTML=(h.url
      ?"<video src='"+h.url+"' muted loop playsinline "+
       "onmouseover='this.play()' onmouseout='this.pause()'></video>"
      :"<div class='history-card-thumb'><img src='"+h.thumb+"'>"+
       "<div class='play'>&#9654;</div></div>")+
      "<div class='cap'>"+esc(h.prompt)+"</div>";
    c.onclick=()=>{showVideo(h);closeGallery();};
    g.appendChild(c);
  });
}
function openGallery(){renderGallery();$("overlay").classList.add("open");}
function closeGallery(){$("overlay").classList.remove("open");}

/* ── floating Jobs panel: minimize, section collapse, drag ── */
function toggleMin(e){e.stopPropagation();
  const p=$("jobsPanel");p.classList.toggle("min");
  $("jobsMin").innerHTML=p.classList.contains("min")?"\u25A1":"\u2013";}
function toggleSec(id){$(id).classList.toggle("closed");}
(function makeDraggable(){
  const panel=$("jobsPanel"),handle=$("jobsHandle");let drag=false,sx,sy,ox,oy;
  handle.addEventListener("pointerdown",e=>{
    if(e.target.closest("button"))return;
    drag=true;handle.setPointerCapture(e.pointerId);
    const par=panel.offsetParent.getBoundingClientRect(),r=panel.getBoundingClientRect();
    ox=r.left-par.left;oy=r.top-par.top;sx=e.clientX;sy=e.clientY;
    panel.style.right="auto";panel.style.left=ox+"px";panel.style.top=oy+"px";
  });
  handle.addEventListener("pointermove",e=>{
    if(!drag)return;
    const par=panel.offsetParent;
    let nx=ox+(e.clientX-sx),ny=oy+(e.clientY-sy);
    nx=Math.max(0,Math.min(nx,par.clientWidth-panel.offsetWidth));
    ny=Math.max(0,Math.min(ny,par.clientHeight-panel.offsetHeight));
    panel.style.left=nx+"px";panel.style.top=ny+"px";
  });
  handle.addEventListener("pointerup",()=>{drag=false;});
})();

/* ── console dock ── */
function toggleDock(){
  const d=$("dock");d.classList.toggle("collapsed");
  $("dockToggle").innerHTML=d.classList.contains("collapsed")?"\u25B4 Show":"\u25BE Hide";
}
function clearConsole(e){e.stopPropagation();$("console").innerHTML="";}
async function copyConsole(e){
  if(e)e.stopPropagation();
  const box=$("console");
  // Prefer the authoritative server buffer (full, untruncated); fall back
  // to whatever text is on screen if that fetch fails.
  let text="";
  try{
    const j=await(await fetch("/api/console")).json();
    text=(j.lines||[]).join("\n");
  }catch(_){ text=box?box.innerText:""; }
  if(!text){ if(box)text=box.innerText; }
  const done=()=>{
    const b=$("conCopyBtn");if(!b)return;
    const old=b.textContent;b.textContent="copied \u2713";
    setTimeout(()=>{b.textContent=old;},1400);
  };
  try{
    await navigator.clipboard.writeText(text);done();
  }catch(_){
    // Clipboard API blocked in the Colab iframe — fall back to execCommand
    // via a temporary textarea.
    try{
      const ta=document.createElement("textarea");
      ta.value=text;ta.style.position="fixed";ta.style.opacity="0";
      document.body.appendChild(ta);ta.focus();ta.select();
      document.execCommand("copy");document.body.removeChild(ta);done();
    }catch(err){ toast("Couldn't copy — select the console text manually.",true); }
  }
}

/* ── hardware / GPU state ── */
async function pollHw(){
  try{const j=await(await fetch("/api/hw")).json();
    if(j.gpu){
      $("vramPill").textContent=j.vram_used+" / "+j.vram_total+" GB";
      $("gpuName").textContent=j.gpu.length>20?j.gpu.slice(0,20)+"\u2026":j.gpu;
    }
    const anyRun=queue.some(q=>q.status==="running");
    const res=j.residency&&j.residency!=="unknown"?j.residency:"";
    let cls,label;
    if(!j.gpu){cls="off";label="No GPU";}
    else if(anyRun){cls="warm";label="Generating";}
    else if(res==="gpu"){cls="on";label="GPU resident";}
    else if(res==="cpu-offload"){cls="on";label="CPU offload";}
    else{cls="cold";label="Connecting";}
    $("connDot").className="dot "+cls;$("connLabel").textContent=label;
  }catch(e){}
}

/* ── console log ── */
let _conSeen=0;
async function pollConsole(){
  try{
    const j=await(await fetch("/api/console")).json();
    const box=$("console");if(!box)return;
    const lines=j.lines||[];
    box.innerHTML=lines.map(l=>{
      const s=l.replace(/&/g,"&amp;").replace(/</g,"&lt;");
      let cls="ln";
      if(l.indexOf("[diag]")>=0)cls="ln diag";
      if(l.indexOf("***")>=0||l.toLowerCase().indexOf("warning")>=0)cls="ln warn";
      return "<div class='"+cls+"'>"+s+"</div>";
    }).join("");
    if(lines.length!==_conSeen){box.scrollTop=box.scrollHeight;_conSeen=lines.length;}
  }catch(e){}
}

/* ── restore queue + history from the server after a page reload ── */
async function restoreJobs(){
  try{
    const j=await(await fetch("/api/jobs")).json();
    (j.jobs||[]).forEach(it=>{
      if(it.status==="done"&&it.has_result){
        if(!doneIds.has(it.id)){
          doneIds.add(it.id);
          history.unshift({id:it.id,prompt:it.prompt,thumb:it.thumb,
            url:null,ts:it.ts});   // video fetched lazily on click
        }
      }else if(it.status==="queued"||it.status==="running"
        ||it.status==="cancelling"){
        if(!queue.some(q=>q.id===it.id))
          queue.unshift({id:it.id,prompt:it.prompt,thumb:it.thumb,
            status:it.status,progress:it.progress||0,stage:it.stage||""});
      }
    });
    renderQueue();renderHistory();
  }catch(e){}
}

updateDur();
setEngine("ltx");   // LTX-2.3 is the default engine
setInterval(pollHw,3000);pollHw();
setInterval(pollConsole,1200);pollConsole();
mlBoot();refreshLoras();renderQueue();renderHistory();restoreJobs();
</script></body></html>
"""

# ---- launch ------------------------------------------------------------
print("=" * 60)
print("  Preparing MissingLink Video Studio (Wan 2.1 image-to-video).")
print("=" * 60)
if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info()
    _tot_gb = _total / 1e9
    _thr = _residency_threshold()
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({_tot_gb:.0f} GB)")
    print(f"  quantization: {QUANTIZATION}")
    if QUANTIZATION.lower() == "gguf" and not torch.cuda.is_bf16_supported():
        print("  NOTE: this GPU has no native bf16 (e.g. a T4). GGUF will "
              "still load, but Wan runs best on bf16-capable cards "
              "(L4 / A100).")
    if _tot_gb >= _thr:
        print(f"  -> enough VRAM for FULL GPU RESIDENCY "
              f"(threshold {_thr:.0f} GB for {QUANTIZATION}).")
    else:
        print(f"  -> under {_thr:.0f} GB — will use model CPU offload "
              "(still works, just slower).")
if QUANTIZATION.lower() == "gguf":
    print("Loading Wan 2.1 i2v (first run downloads ~25 GB: the 11.3 GB "
          "Q4_K_M GGUF transformer replaces the 28 GB bf16 shards)...")
else:
    print("Loading Wan 2.1 i2v (first run downloads ~30 GB of weights)...")

# Diagnostic: confirm this diffusers build ships the Wan i2v classes.
try:
    import diffusers as _dfx
    print(f"  diffusers {_dfx.__version__}")
    _need = ["WanImageToVideoPipeline", "AutoencoderKLWan", "WanVACEPipeline",
             "WanTransformer3DModel", "GGUFQuantizationConfig"]
    _missing = [n for n in _need if not hasattr(_dfx, n)]
    if _missing:
        print(f"  WARNING: this diffusers build is missing {_missing}. "
              "Update diffusers (this cell installs from git).")
    else:
        print("  Wan classes available: WanImageToVideoPipeline, "
              "AutoencoderKLWan, WanVACEPipeline, WanTransformer3DModel, "
              "GGUFQuantizationConfig")
except Exception as _e:
    print(f"  (pipeline class check skipped: {_e})")

if _ml_unlocked():
    if not ML.get("member"):
        print(f"  Free trial: {ML.get('remaining', 0)} of "
              f"{ML.get('free_limit', FREE_RENDERS_HINT)} renders left.")
else:
    print("=" * 60)
    print("  LOCKED — sign in with Google to use the studio.")
    print("  Open the studio link below and sign in. New users get "
          f"{FREE_RENDERS_HINT} free video renders.")
    print(f"  Members: unlimited. Sign up: {MISSINGLINK_SIGNUP_URL}")
    print("=" * 60)

# LTX-2.3 is the default engine. Kick off its download + model load NOW,
# in a background thread, so it is already resident when the user clicks
# Generate. The UI comes up immediately; a render submitted while the
# preload is still in flight simply waits on the same locks and then
# reuses the warm worker. (Wan is still built lazily when switched to.)
print("LTX-2.3 engine: preloading in the background now (models download "
      "once, ~55 GB; the model then loads and stays resident so Generate "
      "starts instantly — progress at /api/preload and in the console).")
threading.Thread(target=_ltx_background_preload, daemon=True,
                 name="ltx-preload").start()

# Auto-load preset LoRAs so they appear in the UI menu on first load.
# Done before the server starts so /api/loras already has them.
for _p in PRESET_LORAS:
    print(f"Preset LoRA '{_p['name']}': fetching "
          f"(~0.7 GB, one-time)...")
    try:
        _ok, _why = register_lora(_p["name"], _p["url"],
                                  _p.get("scale", 0.0))
        if _ok:
            print(f"  preset LoRA '{_p['name']}' ready in the menu "
                  f"(strength {_p.get('scale', 0.0)}).")
        else:
            print(f"  preset LoRA '{_p['name']}' skipped — {_why}")
    except Exception as _ple:
        print(f"  preset LoRA '{_p['name']}' skipped — {_ple}")

print("Starting server...")

def _free_port(pref=5000):
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", pref)); return pref
        except OSError:
            s.bind(("0.0.0.0", 0)); return s.getsockname()[1]

PORT = _free_port(5000)
_srv_err = [None]

def _serve():
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
    except Exception as exc:
        _srv_err[0] = exc

threading.Thread(target=_serve, daemon=True, name="flask").start()

_ready = False
for _ in range(80):
    if _srv_err[0]:
        break
    try:
        if _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive",
                         timeout=0.75).status_code == 200:
            _ready = True
            break
    except Exception:
        pass
    time.sleep(0.5)

if not _ready:
    raise RuntimeError(f"Flask never came up on :{PORT} ({_srv_err[0]})")
print(f"Server healthy on localhost:{PORT}")

if IN_COLAB:
    from IPython.display import display, HTML
    _url = None
    for _ in range(20):
        try:
            cand = eval_js(f"google.colab.kernel.proxyPort({PORT},{{'cache':false}})")
            if cand and not cand.startswith("http"):
                cand = "https://" + cand
            if cand and _requests.get(cand.rstrip("/") + "/api/keepalive",
                                      timeout=4).status_code == 200:
                _url = cand
                break
        except Exception:
            pass
        time.sleep(0.5)
    if _url:
        display(HTML(f"""
        <div style="margin:16px 0;padding:16px 24px;background:#141414;
             border:2px solid #f4b740;border-radius:12px;font-family:monospace;">
          <div style="color:#8a8a8a;font-size:13px;margin-bottom:8px;">
            🔗 MissingLink Video Studio is live:</div>
          <a href="{_url}" target="_blank" style="color:#f4b740;font-size:18px;
             font-weight:bold;">{_url}</a></div>"""))
    else:
        from google.colab import output as _co
        _co.serve_kernel_port_as_window(
            PORT, anchor_text="🔗 Click to open MissingLink Video Studio")
else:
    print(f"\nMissingLink Video Studio running at http://localhost:{PORT}\n")

print("=" * 60)
print("  MissingLink Video Studio running. Keep this cell's runtime alive.")
print("=" * 60)