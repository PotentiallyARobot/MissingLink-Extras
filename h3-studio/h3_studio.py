# ======================================================================
# MiniMax H3 · FAST-START UI · BLACKWELL SM120 + A100 80/40GB + T4/16GB LOW-VRAM
# One Colab cell. No prior app cell required.
# PERFORMANCE UPDATE v20:
# - Sage is the forced production attention backend and MUST come from the MissingLink prebuilt SM120 wheel.
# - Includes CUDA/comfy-kitchen ConvRot fast-path audit.
# - Does NOT replace PyTorch inside a live Comfy runtime. If this cell reports CUDA <13,
#   compare against a fresh cu130+ runtime rather than hot-swapping torch underneath Comfy.
# - User-provided Hugging Face base models are supported with live download progress.
# - Fast-start defaults defer optional accelerator downloads and GPU preload until first use.
#
#
# Colab Secrets / optional integrations:
#   MISSING_LINK_TOKEN - required; validated against MissingLink before the UI can start
#   HF_TOKEN           - optional; used for private/gated Hugging Face downloads where applicable
#   CIVITAI_API_KEY    - optional; NEVER required for startup. Public CivitAI installs are attempted anonymously.
#   OPENAI_API_KEY     - optional; used only when the user explicitly invokes Auto Prompt.
# SageAttention is never built from source in this UI cell; Blackwell requires the MissingLink wheel.
#
# It auto-detects the GPU. Blackwell keeps the existing CU130/Sage resident path;
# T4 / <=18.5 GiB cards use a Q4_0 GGUF DiT + LOW_VRAM / Dynamic VRAM path.
# A100 80GB uses the full-quality stack with full-card residency when budget permits.
# A100 40GB uses the same quality DiT with the smaller official Qwen3-VL encoder
# and an explicit TE→DiT→VAE handoff so the active stage stays below 40GB.
# It downloads the matching H3 stack and conditions with profile-specific residency,
# generates a native ~7.29 s H3 clip, and retimes only ~4.17% to an
# exact 7.00-second final MP4 so motion remains natural.
# ======================================================================

# ============================================================================
#  Standalone MiniMax Studio — one cell, pure Python.
#
#  ComfyUI is imported as a LIBRARY. No server, no graph JSON, no object_info,
#  no frontend. Node classes are called as ordinary Python functions:
#
#    UNETLoader -> MiniMaxH3SigmaShift -> MiniMaxH3ImageToVideo -> BasicGuider
#    -> SamplerCustomAdvanced -> VAEDecode + VAEDecodeAudio -> CreateVideo
#
#  Weights: Comfy-Org pruned int8 convrot (~44 GB).
#  Runtime: RTX PRO 6000 Blackwell 96GB + High-RAM. SM120 dense-attention autotune: SageAttention 2.2 vs SDPA.
#
#  LICENSE: MiniMax H3 Community License, territory clause.
#           https://platform.minimax.io/h3-license
# ============================================================================

COMFY_DIR = "/content/ComfyUI"
UI_PORT   = 7860

# Built-in checkpoint configuration. The official MiniMax H3 weights are the default
# built-in model profiles exposed by this studio. Optional acceleration LoRAs
# remain available and are downloaded lazily on first use.
FAST_STARTUP = os.environ.get("H3_FAST_STARTUP", "1").strip().lower() not in {"0", "false", "no", "off"} if "os" in globals() else True
DIT_CHOICE = "stock_quality"
FALLBACK_DIT_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_DIT_FILE = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF2VA_DIT_GIB = 19.53

# Automatic low-VRAM leg.
T4_LOWVRAM_MAX_GIB = 18.5
T4_RESERVE_VRAM_GIB = 2.0
T4_DIT_REPO = "molbal/MiniMax-H3-GGUF"
T4_DIT_FILE = "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
T4_DIT_SHA256 = "50891b806d6d700f4f20931791ca42a083dd9148609838268ccdc782bf899c1c"
T4_DIT_GIB = 10.62
T4_TEXT_ENCODER_FILE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
T4_TEXT_ENCODER_GIB = 14.61

# A100 profiles.
A100_80_MIN_GIB = 60.0
A100_80_RESERVE_VRAM_GIB = 8.0
A100_40_RESERVE_VRAM_GIB = 5.0
A100_40_TEXT_ENCODER_FILE = T4_TEXT_ENCODER_FILE
A100_40_TEXT_ENCODER_GIB = T4_TEXT_ENCODER_GIB

DITS = {
    "stock_quality": ("Comfy-Org/MiniMax-H3", "diffusion_models", FALLBACK_DIT_FILE),
}

# Compatibility sentinels for old saved payloads. They are never exposed or installed.
USING_EROS_MAX = False
USING_REDMIX = False
EROS_MAX_FILE = "__legacy_disabled_checkpoint__.safetensors"
EROS_MAX_REPO = ""
EROS_MAX_SHA256 = ""
EROS_MAX_GIB = 19.53
REDMIX_FILE = "__legacy_disabled_checkpoint_2__.safetensors"
REDMIX_FILE_ALIASES = [REDMIX_FILE]
REDMIX_GIB = 19.53
REDMIX_VERSION = 0
REDMIX_MODEL_ID = 0
REDMIX_SHA256 = ""
REDMIX_PAGE_URL = ""
REDMIX_DIRECT_URL_SECRET = ""
REDMIX_LOCAL_PATH_SECRET = ""
CIVITAI_COOKIE_SECRET = ""
REQUIRE_REDMIX = False
LEGACY_DISABLED_LORA_FILE = "__legacy_disabled_lora__.safetensors"
LEGACY_DISABLED_LORA_STRENGTH = 0.0
LEGACY_DISABLED_LORA_VERSION = 0
LEGACY_DISABLED_LORA_SHA256 = ""

# Optional acceleration LoRAs.
LIGHTNING_REPO = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
LIGHTNING_FILE = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_64_bf16.safetensors"
REF2VA_LIGHTNING_REPO = "Comfy-Org/MiniMax-H3"
REF2VA_LIGHTNING_FILE = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
LIGHTNING_STRENGTH = 1.0
LIGHTNING_DEFAULT = True
LIGHTNING_STEPS = 4
LIGHTNING_SHIFT_VIDEO = 6.0
LIGHTNING_SHIFT_AUDIO = 3.0

# rzgar MiniMax-H3 FL2V 8-Step Motion Enhancer.
MOTION8_REPO = "rzgar/minimax-h3_fl2v_8Step_motion_enhancer"
MOTION8_FILE = "minimax-h3_fl2v_8Step_motion_enhancer.safetensors"
MOTION8_STRENGTH = 1.0

# No built-in specialty catalog is configured.
SPECIALTY_LORA_PRESETS = []
SPECIALTY_LORA_STATES = []
ACTION_LABEL = "Optional LoRA"
ACTION_MODEL_ID = 0
ACTION_REQUESTED_VERSION_ID = 0
ACTION_LINKED_VERSION = 0
ACTION_SOURCE_PAGE = ""
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

# Blackwell memory policy. The full RTX PRO 6000 has enough VRAM for the ~50 GiB
# active H3 stack plus large activation/workspace headroom. Keep a real reserve for
# decode and temporary carriers. Partitioned SM120 cards fall back to NORMAL_VRAM.
RESERVE_VRAM = 8.0
LOWVRAM      = False
BLACKWELL_FULL_CARD_MIN_GIB = 80.0

# Blackwell attention/runtime optimization. Dense/fallback attention is benchmarked
# on the actual SM120 GPU before ComfyUI imports its attention aliases. H3 sparse
# attention is separate and auto-prefers the shipped native Kitchen SM120 backend.
AUTO_OPTIMIZE_ATTENTION = True
SAGEATTN_VERSION = "2.2.0"
ATTN_BENCH_SEQ = 8192
ATTN_BENCH_HEADS = 24
ATTN_BENCH_DIM = 128

# V86 quality-first conditioning encoder. Use the official CUDA13-friendly
# INT8 ConvRot Qwen3-VL-32B instead of the much more aggressively quantized
# NVFP4/AWQ encoder. On a full 96GB Blackwell the INT8 TE + DiT + both VAEs remain warm.
TEXT_ENCODER_FILE = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TEXT_ENCODER_GIB  = 25.24   # ~27.1 GB decimal published size
REDMIX_GIB        = 19.53   # ~20.97 GB decimal on disk
VIDEO_VAE_GIB     = 4.85
AUDIO_VAE_GIB     = 0.57
# ───────────────────────────────────────────────────────────────────────────

# ======================================================================
# CU130 BOOTSTRAP · V86-BW1 · BLACKWELL SM120 · NO VENV / NO ENSUREPIP
# ======================================================================
# Colab's current /usr/bin/python3 can have a broken ensurepip path, which makes
# `python -m venv` fail before the environment even exists. V86 avoids venv
# completely. It installs the cu130 stack into a private --target directory and
# launches a clean child interpreter with that directory first on PYTHONPATH.
import os as _os, sys as _sys, subprocess as _sp, pathlib as _pl, shutil as _shutil, re as _re, textwrap as _textwrap, threading as _threading
FAST_STARTUP = _os.environ.get("H3_FAST_STARTUP", "1").strip().lower() not in {"0", "false", "no", "off"}

# ======================================================================
# MISSINGLINK ACCESS GATE · fail closed
# ======================================================================
# The current public MiniMax H3 notebook instructs users to put their key in the
# Colab Secret MISSING_LINK_TOKEN and advertises a 7-day free trial.  The older
# H3 studio code had ML_OK=True, which meant the UI was not actually protected.
# V88 validates the key against MissingLink's existing non-generation auth endpoint
# before doing expensive CUDA/model setup, and re-checks periodically while the UI
# is running.  The token is never printed or written to disk.
MISSING_LINK_AUTH_URL = (_os.environ.get("MISSING_LINK_AUTH_URL") or
                         "https://missinglink.build/api/cache-token").strip()
MISSING_LINK_TRIAL_URL = "https://www.missinglink.build/pricing.html"
MISSING_LINK_AUTH_TTL_SEC = 600.0
_ML_AUTH_STATE = {"ok": False, "checked": 0.0, "error": "not checked"}

def _read_missinglink_token():
    token = (_os.environ.get("MISSING_LINK_TOKEN") or "").strip()
    if not token:
        try:
            from google.colab import userdata as _ml_userdata
            token = (_ml_userdata.get("MISSING_LINK_TOKEN") or "").strip()
        except Exception:
            token = ""
    if token:
        # Child processes inherit the already-read secret without needing access to
        # Colab's userdata API themselves.
        _os.environ["MISSING_LINK_TOKEN"] = token
    return token

def _validate_missinglink_token(*, force=False):
    """Validate MISSING_LINK_TOKEN without consuming a generation/credit."""
    import json as _ml_json
    import time as _ml_time
    import urllib.request as _ml_urlreq
    import urllib.error as _ml_urlerr

    now = _ml_time.monotonic()
    if (not force and _ML_AUTH_STATE.get("ok") and
            now - float(_ML_AUTH_STATE.get("checked") or 0.0) < MISSING_LINK_AUTH_TTL_SEC):
        return True, ""

    token = _read_missinglink_token()
    if not token:
        msg = (
            "MISSING_LINK_TOKEN is not set. Add it in Colab Secrets (key icon), "
            "enable notebook access, then rerun the cell."
        )
        _ML_AUTH_STATE.update(ok=False, checked=now, error=msg)
        return False, msg

    req = _ml_urlreq.Request(
        MISSING_LINK_AUTH_URL,
        headers={
            "x-api-key": token,
            "Accept": "application/json",
            "User-Agent": "MissingLink-H3-V88-Colab",
        },
        method="GET",
    )
    try:
        with _ml_urlreq.urlopen(req, timeout=15) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read(65536)
        try:
            data = _ml_json.loads(raw.decode("utf-8", "replace")) if raw else {}
        except Exception:
            data = {}
        if 200 <= status < 300 and data.get("ok") is True:
            _ML_AUTH_STATE.update(ok=True, checked=now, error="")
            return True, ""
        msg = f"MissingLink rejected this API key (HTTP {status})."
    except _ml_urlerr.HTTPError as e:
        # Do not echo a response body: auth services sometimes include account data.
        msg = f"MissingLink rejected this API key (HTTP {e.code})."
    except Exception as e:
        # Fail closed: if validity cannot be established, the paid UI should not run.
        msg = f"Could not validate the MissingLink API key: {type(e).__name__}: {e}"

    _ML_AUTH_STATE.update(ok=False, checked=now, error=msg)
    return False, msg

def _require_missinglink_access():
    ok, error = _validate_missinglink_token(force=True)
    if not ok:
        raise SystemExit(
            "\n✗ MissingLink access required.\n"
            f"  {error}\n"
            "  Put a valid key in the Colab Secret MISSING_LINK_TOKEN and rerun.\n"
            f"  Start the 7-day free trial / get access: {MISSING_LINK_TRIAL_URL}\n"
        )
    print("✓ MissingLink API key validated · UI access granted", flush=True)

# Gate the notebook before CUDA builds, weight downloads, or the Flask UI start.
_require_missinglink_access()

def _probe_parent_gpu_no_torch():
    """Cheap GPU/profile probe before importing torch or entering the SM120 bootstrap."""
    try:
        p = _sp.run([
            "nvidia-smi", "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits", "-i", "0"
        ], stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, timeout=8)
        if p.returncode != 0 or not p.stdout.strip():
            return "", 0.0
        row = p.stdout.strip().splitlines()[0]
        name, mem_mib = [x.strip() for x in row.rsplit(",", 1)]
        return name, float(mem_mib) / 1024.0
    except Exception:
        return "", 0.0

_PARENT_IS_CHILD = _os.environ.get("H3_CU130_CHILD") == "1"
_PARENT_GPU_NAME, _PARENT_GPU_GIB = _probe_parent_gpu_no_torch() if not _PARENT_IS_CHILD else ("", 0.0)
_PARENT_LOWVRAM = bool(
    not _PARENT_IS_CHILD
    and _PARENT_GPU_GIB > 0
    and (_PARENT_GPU_GIB <= T4_LOWVRAM_MAX_GIB or "T4" in _PARENT_GPU_NAME.upper())
)
_PARENT_A100 = bool(
    not _PARENT_IS_CHILD
    and _PARENT_GPU_GIB > 0
    and "A100" in _PARENT_GPU_NAME.upper()
)
if _PARENT_LOWVRAM:
    _os.environ["H3_GPU_PROFILE"] = "t4_16gb"
    print(
        f"✓ V89 AUTO: {_PARENT_GPU_NAME} · {_PARENT_GPU_GIB:.1f} GiB detected -> "
        "T4/LOW-VRAM Q4_0 GGUF profile (skip SM120 cu130 bootstrap).",
        flush=True,
    )
elif _PARENT_A100:
    _a100_profile = "a100_80gb" if _PARENT_GPU_GIB >= A100_80_MIN_GIB else "a100_40gb"
    _os.environ["H3_GPU_PROFILE"] = _a100_profile
    print(
        f"✓ V89 AUTO: {_PARENT_GPU_NAME} · {_PARENT_GPU_GIB:.1f} GiB detected -> "
        f"{_a100_profile.upper()} native-runtime profile (skip SM120-only bootstrap).",
        flush=True,
    )

# ======================================================================
# PREBUILT RUNTIME SNAPSHOT · HUGGING FACE STORAGE BUCKET
# ======================================================================
# A successfully-prepared Colab runtime can be snapshotted with the companion
# builder cell and stored in the MissingLinkBuilder/MissingLink_H3_Minimax bucket.
# On a fresh runtime we restore that snapshot before the normal bootstrap. The
# existing runtime probes below remain authoritative: if the restored binary
# environment is stale/incompatible, FAST-REUSE rejects it and the normal builder
# path takes over. Model weights are deliberately NOT part of the snapshot.
H3_RUNTIME_BUCKET = (_os.environ.get("H3_RUNTIME_BUCKET") or
                     "MissingLinkBuilder/MissingLink_H3_Minimax").strip()
H3_RUNTIME_BUNDLE_ENABLED = _os.environ.get("H3_RUNTIME_BUNDLE", "1").strip().lower() not in {"0","false","no","off"}
H3_RUNTIME_BUNDLE_REQUIRED = _os.environ.get("H3_RUNTIME_BUNDLE_REQUIRED", "0").strip().lower() in {"1","true","yes","on"}
H3_RUNTIME_CACHE_DIR = _pl.Path("/content/.missinglink_h3_runtime")
H3_RUNTIME_MARKER = _pl.Path("/content/.missinglink_h3_runtime_restored.json")

def _read_hf_token_parent():
    tok = (_os.environ.get("HF_TOKEN") or _os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
    if not tok:
        try:
            from google.colab import userdata as _hf_userdata
            tok = (_hf_userdata.get("HF_TOKEN") or "").strip()
        except Exception:
            tok = ""
    if tok:
        _os.environ["HF_TOKEN"] = tok
    return tok

def _runtime_profile_key():
    if _PARENT_LOWVRAM:
        return "t4_16gb"
    if _PARENT_A100:
        return "a100_80gb" if _PARENT_GPU_GIB >= A100_80_MIN_GIB else "a100_40gb"
    return "blackwell_sm120"

def _ensure_runtime_bucket_deps():
    try:
        import huggingface_hub as _hfh
        from huggingface_hub import HfFileSystem as _HfFileSystem  # noqa: F401
        from huggingface_hub import get_bucket_file_metadata as _get_bucket_file_metadata  # noqa: F401
        import zstandard as _zstd  # noqa: F401
        return
    except Exception:
        pass
    print("  ↓ preparing HF Storage Bucket runtime client (one-time tiny dependency step)", flush=True)
    _sp.run([
        _sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "huggingface_hub>=1.5.0,<2", "zstandard>=0.22"
    ], check=True)

def _human_bytes(n):
    n = float(max(0, int(n or 0)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def _bucket_download_progress(remote_path, local_path, *, label,
                              expected_size=None, strict_size=False, resume=True):
    """Download one HF Storage Bucket object with visible progress.

    Bucket metadata is treated as a progress hint only. In practice `latest.json` can
    be replaced atomically while an edge/cache still reports the previous object's
    size for a short time. That must never make a valid runtime snapshot fall back to
    the multi-GB bootstrap. Immutable archives can pass `expected_size` from their
    manifest and are still verified by SHA256 after download.
    """
    from huggingface_hub import HfFileSystem, get_bucket_file_metadata
    import time as _time

    token = _read_hf_token_parent() or None
    local_path = _pl.Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    part = _pl.Path(str(local_path) + ".part")

    meta_size = 0
    try:
        meta = get_bucket_file_metadata(H3_RUNTIME_BUCKET, remote_path, token=token)
        meta_size = int(getattr(meta, "size", 0) or 0)
    except Exception as _meta_error:
        print(f"  ↳ {label}: bucket size metadata unavailable ({_meta_error}); streaming anyway", flush=True)

    expected_size = int(expected_size or 0)
    total = expected_size or meta_size
    fs = HfFileSystem(token=token)
    hf_path = f"buckets/{H3_RUNTIME_BUCKET}/{remote_path}"

    if part.exists() and not resume:
        part.unlink(missing_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    # Only an immutable manifest-provided size is authoritative enough to discard a
    # partial download. Bucket metadata can briefly be stale after latest.json changes.
    if expected_size and offset > expected_size:
        part.unlink(missing_ok=True); offset = 0
    mode = "ab" if offset else "wb"
    t0 = _time.time(); last = t0; last_bytes = offset

    with fs.open(hf_path, "rb") as src:
        if offset:
            try:
                src.seek(offset)
                print(f"  ↻ resuming {label} at {_human_bytes(offset)}", flush=True)
            except Exception:
                part.unlink(missing_ok=True); offset = 0; mode = "wb"
                src.seek(0)
        done = offset
        with open(part, mode) as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk); done += len(chunk)
                now = _time.time()
                if now - last >= 0.45 or (total and done >= total):
                    speed = (done - last_bytes) / max(0.001, now - last)
                    if total:
                        pct = min(100.0, 100.0 * done / total)
                        msg = (f"\r  ↓ {label}: {pct:6.2f}% · {_human_bytes(done)}/{_human_bytes(total)} "
                               f"· {speed/1024**2:.1f} MiB/s")
                    else:
                        msg = f"\r  ↓ {label}: {_human_bytes(done)} · {speed/1024**2:.1f} MiB/s"
                    print(msg, end="", flush=True)
                    last, last_bytes = now, done
    print(flush=True)

    actual = part.stat().st_size if part.exists() else 0
    if actual <= 0:
        raise RuntimeError(f"Bucket download returned an empty object: {remote_path}")
    if strict_size and expected_size and actual != expected_size:
        raise RuntimeError(
            f"Runtime archive size mismatch for {remote_path}: {actual} != {expected_size}"
        )
    if not expected_size and meta_size and actual != meta_size:
        print(
            f"  ↳ {label}: bucket metadata was stale ({_human_bytes(meta_size)} reported, "
            f"{_human_bytes(actual)} received); using the current object.",
            flush=True,
        )
    _os.replace(part, local_path)
    return actual

def _sha256_file(path):
    import hashlib as _hashlib
    h = _hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _restore_runtime_bundle():
    if _PARENT_IS_CHILD or not H3_RUNTIME_BUNDLE_ENABLED:
        return False
    if H3_RUNTIME_MARKER.exists() and _os.environ.get("H3_RUNTIME_FORCE_REFRESH", "0").strip().lower() not in {"1","true","yes","on"}:
        try:
            old = __import__("json").loads(H3_RUNTIME_MARKER.read_text())
            print(f"✓ prebuilt runtime already restored in this VM · build {old.get('build_id','unknown')}", flush=True)
            return True
        except Exception:
            pass

    profile = _runtime_profile_key()
    py_tag = f"py{_sys.version_info.major}{_sys.version_info.minor}"
    pointer_remote = f"runtimes/{profile}/{py_tag}/latest.json"
    H3_RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pointer_local = H3_RUNTIME_CACHE_DIR / f"{profile}-{py_tag}-latest.json"

    try:
        _ensure_runtime_bucket_deps()
        _bucket_download_progress(pointer_remote, pointer_local, label="runtime manifest", resume=False)
        import json as _json, platform as _platform, tarfile as _tarfile, time as _time
        import zstandard as _zstd
        manifest = _json.loads(pointer_local.read_text())
        expected_profile = str(manifest.get("profile") or "")
        expected_py = str(manifest.get("python_major_minor") or "")
        expected_machine = str(manifest.get("machine") or "")
        if expected_profile != profile:
            raise RuntimeError(f"runtime profile mismatch: bucket={expected_profile!r}, local={profile!r}")
        if expected_py and expected_py != f"{_sys.version_info.major}.{_sys.version_info.minor}":
            raise RuntimeError(f"runtime Python mismatch: bucket={expected_py}, local={_sys.version_info.major}.{_sys.version_info.minor}")
        if expected_machine and expected_machine != _platform.machine():
            raise RuntimeError(f"runtime machine mismatch: bucket={expected_machine}, local={_platform.machine()}")
        archive_remote = str(manifest.get("archive_path") or "")
        expected_sha = str(manifest.get("archive_sha256") or "").lower()
        if not archive_remote or not expected_sha:
            raise RuntimeError("runtime pointer is missing archive_path/archive_sha256")

        archive_local = H3_RUNTIME_CACHE_DIR / _pl.Path(archive_remote).name
        print(f"⚡ MissingLink runtime snapshot found · {profile} · build {manifest.get('build_id','?')}", flush=True)
        _bucket_download_progress(
            archive_remote, archive_local, label="prebuilt runtime",
            expected_size=int(manifest.get("archive_size") or 0) or None,
            strict_size=bool(int(manifest.get("archive_size") or 0)), resume=True
        )
        print("  ↳ verifying runtime SHA256…", flush=True)
        got_sha = _sha256_file(archive_local)
        if got_sha != expected_sha:
            archive_local.unlink(missing_ok=True)
            raise RuntimeError(f"runtime SHA256 mismatch: expected {expected_sha}, got {got_sha}")

        staging = _pl.Path(f"/content/.h3_runtime_extract_{_os.getpid()}")
        _shutil.rmtree(staging, ignore_errors=True); staging.mkdir(parents=True, exist_ok=True)
        total_payload = int(manifest.get("payload_bytes") or 0)
        extracted = 0; last_notice = 0.0
        print("  ↳ extracting prebuilt runtime…", flush=True)
        with open(archive_local, "rb") as raw:
            with _zstd.ZstdDecompressor().stream_reader(raw) as zr:
                with _tarfile.open(fileobj=zr, mode="r|") as tf:
                    for member in tf:
                        tf.extract(member, path=staging, filter="data")
                        if member.isfile():
                            extracted += int(member.size or 0)
                        now = _time.time()
                        if now - last_notice >= 0.8:
                            if total_payload:
                                print(f"\r  ↳ extract: {min(100.0,100.0*extracted/total_payload):6.2f}% · {extracted/1024**3:.2f}/{total_payload/1024**3:.2f} GiB", end="", flush=True)
                            else:
                                print(f"\r  ↳ extract: {extracted/1024**3:.2f} GiB", end="", flush=True)
                            last_notice = now
        print(flush=True)

        roots = manifest.get("roots") or []
        if not roots:
            roots = [{"path": x, "mode": "replace"} for x in _os.listdir(staging)]
        for item in roots:
            rel = str((item or {}).get("path") or "").strip().strip("/")
            mode = str((item or {}).get("mode") or "replace")
            if not rel:
                continue
            src = staging / rel
            dst = _pl.Path("/content") / rel
            if not src.exists() and not src.is_symlink():
                continue
            if mode == "merge":
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    _shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True); _shutil.copy2(src, dst)
            else:
                if dst.exists() or dst.is_symlink():
                    if dst.is_dir() and not dst.is_symlink(): _shutil.rmtree(dst, ignore_errors=True)
                    else: dst.unlink(missing_ok=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                _shutil.move(str(src), str(dst))
        _shutil.rmtree(staging, ignore_errors=True)
        H3_RUNTIME_MARKER.write_text(_json.dumps({
            "build_id": manifest.get("build_id"), "profile": profile,
            "archive_sha256": expected_sha, "restored_at": _time.time(),
        }, indent=2))
        if _os.environ.get("H3_KEEP_RUNTIME_ARCHIVE", "0").strip().lower() not in {"1","true","yes","on"}:
            archive_local.unlink(missing_ok=True)
        print(f"✓ prebuilt runtime restored · build {manifest.get('build_id','?')} · normal compatibility probes will verify it", flush=True)
        return True
    except Exception as exc:
        msg = f"prebuilt runtime unavailable; falling back to normal bootstrap: {type(exc).__name__}: {exc}"
        if H3_RUNTIME_BUNDLE_REQUIRED:
            raise RuntimeError(msg) from exc
        print("⚠ " + msg, flush=True)
        return False

if not _PARENT_IS_CHILD:
    _restore_runtime_bundle()

# Historical name retained because the main UI body is guarded by it. A100/T4
# execute that body directly; Blackwell still uses the isolated CUDA13 child.
_CU130_CHILD = _PARENT_IS_CHILD or _PARENT_LOWVRAM or _PARENT_A100

def _run_checked(cmd, *, env=None, title=None, cwd=None):
    if title:
        print(f"\n{title}")
    print("  $ " + " ".join(map(str, cmd)))
    p = _sp.run(cmd, env=env, cwd=cwd, text=True,
                stdout=_sp.PIPE, stderr=_sp.STDOUT)
    if p.stdout:
        print(p.stdout[-16000:])
    if p.returncode:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}:\n"
            + " ".join(map(str, cmd))
            + "\n\nLast installer output:\n"
            + (p.stdout[-16000:] if p.stdout else "(no output)")
        )
    return p

def _pick_ui_port(start=7860, stop=7900):
    import socket as _socket
    for _port in range(start, stop):
        with _socket.socket() as _s:
            try:
                _s.bind(("127.0.0.1", _port))
                return _port
            except OSError:
                pass
    raise RuntimeError(f"No free UI port found in {start}..{stop-1}")

def _child_log_tail(_path, n=80):
    try:
        _lines = _pl.Path(_path).read_text(errors="replace").splitlines()
        return "\n".join(_lines[-n:])
    except Exception:
        return "(child log unavailable)"

def _start_ui_child(_self, _env, _port):
    """Start the isolated cu130 UI and stream its startup log into this cell."""
    _env = _env.copy()
    _env["PYTHONUNBUFFERED"] = "1"
    _log_path = f"/content/h3_v86_ui_child_{_port}.log"
    try:
        _pl.Path(_log_path).unlink()
    except FileNotFoundError:
        pass

    _proc = _sp.Popen(
        [_sys.executable, str(_self)],
        env=_env,
        stdout=_sp.PIPE,
        stderr=_sp.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump():
        try:
            with open(_log_path, "a", encoding="utf-8", buffering=1) as _fh:
                if _proc.stdout is not None:
                    for _line in _proc.stdout:
                        _fh.write(_line)
                        print("[V86 child] " + _line.rstrip(), flush=True)
        except Exception as _e:
            try:
                with open(_log_path, "a", encoding="utf-8") as _fh:
                    _fh.write(f"\n[stdout pump error] {_e!r}\n")
            except Exception:
                pass

    _threading.Thread(target=_pump, daemon=True).start()
    return _proc, _log_path

def _wait_for_ui(_proc, _port, _log_path, timeout=1800):
    import socket as _socket, time as _time
    _deadline = _time.time() + timeout
    _last_notice = 0.0
    while _time.time() < _deadline:
        if _proc.poll() is not None:
            raise RuntimeError(
                f"V86 UI child exited early with code {_proc.returncode}.\n\n"
                f"Last child output:\n{_child_log_tail(_log_path)}"
            )
        try:
            with _socket.create_connection(("127.0.0.1", _port), timeout=0.5):
                print(f"✓ V86 child opened port {_port}", flush=True)
                return
        except OSError:
            now = _time.time()
            if now - _last_notice >= 30:
                elapsed = int(timeout - max(0, _deadline - now))
                print(
                    f"  … waiting for V86 model/UI preload: {elapsed}s elapsed · "
                    f"child PID {_proc.pid} · log {_log_path}",
                    flush=True,
                )
                _last_notice = now
            _time.sleep(1.0)

    raise RuntimeError(
        f"V86 UI child is still alive but did not open port {_port} within {timeout}s.\n"
        "The UI intentionally opens only after weights/models are ready.\n\n"
        f"Last child output:\n{_child_log_tail(_log_path)}"
    )

def _show_child_ui(_port, _pid):
    print(f"✓ V86 UI child alive · PID {_pid} · port {_port}", flush=True)
    try:
        from google.colab import output as _co
        print("Opening MissingLink MiniMax Studio · Fast...", flush=True)
        try:
            _co.serve_kernel_port_as_iframe(_port, height="900")
        except Exception as _e:
            print("iframe warning:", _e)
        try:
            _co.serve_kernel_port_as_window(
                _port, anchor_text="◤ Open MissingLink MiniMax Studio · Fast in a new tab")
        except Exception as _e:
            print("window warning:", _e)
    except Exception:
        print(f"Open http://127.0.0.1:{_port}")

if not _CU130_CHILD:
    _sp.run(["pkill", "-9", "-f", "h3_cu130_production_ui_v(39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60|61|62|63|64|65|66|67|68|69|70|71|72|73|74|75|76|77|78|79|80|81|82|83|84|85|86).py"],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, check=False)
    _root = _pl.Path("/content/h3_cu130_target_v35")
    _cuda_pkg_root = _pl.Path("/content/h3_cuda130_compiler_pkgs_v35")
    _cuda_runtime_extra = _pl.Path("/content/h3_cuda130_runtime_extra_v35")

    # MissingLink-published SageAttention binary. Blackwell accepts this exact
    # CPython 3.13 + Torch 2.11/cu130 + SM120 build and no source fallback.
    _ML_SAGE_WHEEL = "sageattention-2.2.0-1sm120cu130-cp313-cp313-linux_x86_64.whl"
    _ML_SAGE_MARKER = ".missinglink_sageattention_2.2.0_sm120cu130"
    _ML_SAGE_URL = "https://missinglink.build/wheel/" + _ML_SAGE_WHEEL

    # V86 rescue path: V23 often got all the way through Torch 2.11+cu130 and
    # CUDA 13, then failed only because stale setuptools 78 metadata won over
    # the requested 74.1.3 build frontend. If that partial target is present
    # and the preferred v35 target has no Torch yet, continue directly from
    # the V23 target instead of downloading several GB again.
    _legacy_v23_root = _pl.Path("/content/h3_cu130_target_v23")
    _legacy_v23_cuda = _pl.Path("/content/h3_cuda130_compiler_pkgs_v23")
    _legacy_v23_extra = _pl.Path("/content/h3_cuda130_runtime_extra_v23")
    if (_legacy_v23_root / "torch").exists() and not (_root / "torch").exists():
        _root = _legacy_v23_root
        if _legacy_v23_cuda.exists():
            _cuda_pkg_root = _legacy_v23_cuda
        if _legacy_v23_extra.exists():
            _cuda_runtime_extra = _legacy_v23_extra
        print(
            "✓ V86 rescue: found partial V23 cu130 target. "
            "Will validate/reuse it, clean stale Sage build metadata, and continue.",
            flush=True,
        )
    # V86: never destroy a valid private CUDA/compiler target just to probe reuse.
    # Repeated notebook runs should be cheap and should preserve a successfully built
    # SageAttention environment. Fresh/invalid roots are cleaned only when required.
    _root.mkdir(parents=True, exist_ok=True)
    _cuda_pkg_root.mkdir(parents=True, exist_ok=True)
    _cuda_runtime_extra.mkdir(parents=True, exist_ok=True)
    _self = _pl.Path("/content/h3_cu130_production_ui_v86.py")
    _src = None
    # Prefer __file__ for uploaded/%run scripts. Using In[-1] first can capture only
    # the "%run ..." wrapper instead of this program.
    try:
        if "__file__" in globals():
            _candidate = _pl.Path(__file__)
            if _candidate.is_file() and _candidate.suffix.lower() == ".py":
                _src = _candidate.read_text()
    except Exception:
        _src = None
    if not _src:
        # exec(open(...).read()) does not reliably point __file__ at this script.
        # Recover from common Colab launch patterns by finding the actual studio file.
        _source_candidates = []
        _env_source = (_os.environ.get("H3_STUDIO_SOURCE") or "").strip()
        if _env_source:
            _source_candidates.append(_pl.Path(_env_source))
        _source_candidates += [
            _pl.Path.cwd() / "h3_studio.py",
            _pl.Path.cwd() / "h3_studio_updated.py",
            _pl.Path("/content/MissingLink-Extras/h3-studio/h3_studio.py"),
            _pl.Path("/content/MissingLink-Extras/h3-studio/h3_studio_updated.py"),
        ]
        try:
            _source_candidates += sorted(_pl.Path.cwd().glob("h3_studio*.py"))
        except Exception:
            pass
        for _candidate in _source_candidates:
            try:
                if _candidate.is_file():
                    _maybe = _candidate.read_text()
                    if "CU130 BOOTSTRAP · V86" in _maybe:
                        _src = _maybe
                        break
            except Exception:
                pass
    if not _src:
        try:
            _src = get_ipython().user_ns["In"][-1]
        except Exception:
            _src = None
    if not _src or "CU130 BOOTSTRAP · V86" not in _src:
        raise RuntimeError(
            "Could not capture the H3 Studio source. Set H3_STUDIO_SOURCE to the .py file, "
            "or run it with %run."
        )
    _self.write_text(_src)

    # ------------------------------------------------------------------
    # V25 FAST REUSE
    # ------------------------------------------------------------------
    # If V24/V25 already built a working Torch 2.11+cu130 + SageAttention
    # SM120 target in this Colab VM, reuse it instead of downloading ~GBs
    # and recompiling Sage again. Fresh runtimes fall through to the full
    # bootstrap automatically.
    _reuse_candidates = [
        (
            _pl.Path("/content/h3_cu130_target_v35"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v35"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v35"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v34"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v34"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v34"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v33"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v33"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v33"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v32"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v32"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v32"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v31"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v31"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v31"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v30"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v30"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v30"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v29"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v29"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v29"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v28"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v28"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v28"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v27"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v27"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v27"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v26"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v26"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v26"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v25"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v25"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v25"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v24"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v24"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v24"),
        ),
        (
            _pl.Path("/content/h3_cu130_target_v23"),
            _pl.Path("/content/h3_cuda130_compiler_pkgs_v23"),
            _pl.Path("/content/h3_cuda130_runtime_extra_v23"),
        ),
    ]

    # V35 FIX: initialize the child environment BEFORE FAST-REUSE probing.
    # V34 referenced _env inside _try_reuse() before the later bootstrap block
    # created it, which only worked accidentally when a previous notebook cell
    # had left _env in globals().
    _env = _os.environ.copy()
    try:
        from google.colab import userdata as _userdata
        for _secret in ("MISSING_LINK_TOKEN", "CIVITAI_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"):
            try:
                _v = (_userdata.get(_secret) or "").strip()
                if _v:
                    _env[_secret] = _v
            except Exception:
                pass
    except Exception:
        pass

    if _env.get("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY found in Colab userdata and forwarded to the isolated UI process.", flush=True)
    else:
        print("⚠ OPENAI_API_KEY is not available in the parent Colab process.", flush=True)

    def _try_reuse(_rr, _cc, _xx):
        if not _rr.exists():
            return None
        _marker = _rr / _ML_SAGE_MARKER
        if not _marker.exists() or _marker.read_text(errors="ignore").strip() != _ML_SAGE_WHEEL:
            print("FAST-REUSE candidate rejected (MissingLink Sage wheel marker absent):", _rr)
            return None
        _e = _env.copy()
        _e["PYTHONPATH"] = str(_rr)
        _e["PYTHONNOUSERSITE"] = "1"

        _nv = [p for p in _cc.glob("nvidia/**/bin/nvcc") if p.is_file()] if _cc.exists() else []
        if _nv:
            _cuda_home_r = str(_nv[0].parent.parent)
            _e["CUDA_HOME"] = _cuda_home_r
            _e["CUDA_PATH"] = _cuda_home_r
            _e["PATH"] = f"{_cuda_home_r}/bin:" + _e.get("PATH","")

        _libs = []
        for _base in (_rr, _cc, _xx):
            if not _base.exists():
                continue
            for _pat in ("nvidia/**/lib","nvidia/**/lib64"):
                for _p in _base.glob(_pat):
                    if _p.is_dir():
                        _q = str(_p.resolve())
                        if _q not in _libs:
                            _libs.append(_q)
        if _libs:
            _e["LD_LIBRARY_PATH"] = ":".join(_libs + ([_e.get("LD_LIBRARY_PATH","")] if _e.get("LD_LIBRARY_PATH") else []))

        _probe = r"""
import torch
assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
assert str(torch.version.cuda).startswith("13.0"), torch.version.cuda
assert torch.cuda.get_device_capability(0) == (12,0)
from sageattention import sageattn
q=torch.randn(1,24,256,128,device="cuda",dtype=torch.bfloat16)
with torch.no_grad():
    y=sageattn(q,q,q,tensor_layout="HND",is_causal=False)
torch.cuda.synchronize()
print("FAST-REUSE OK:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), tuple(y.shape))
"""
        try:
            p = _sp.run([_sys.executable, "-c", _textwrap.dedent(_probe).strip()], env=_e, text=True,
                        stdout=_sp.PIPE, stderr=_sp.STDOUT, timeout=45)
            if p.returncode == 0:
                print(p.stdout.strip())
                return _e
            print("FAST-REUSE candidate rejected:", _rr)
            print((p.stdout or "")[-2500:])
        except Exception as _ex:
            print("FAST-REUSE probe failed:", _rr, repr(_ex))
        return None

    def _parent_download_latest_zip():
        """Run only in the notebook parent process, never the benchmark child."""
        _marker = _pl.Path("/content/h3_v86_latest_zip.txt")
        if not _marker.exists():
            print("⚠ V86 ZIP marker was not written by child.")
            return
        try:
            _zp = _pl.Path(_marker.read_text().strip())
            if not _zp.exists():
                print("⚠ V86 ZIP marker points to missing file:", _zp)
                return
            print("\nV86 RESULT ZIP:", _zp)
            print(f"ZIP size: {_zp.stat().st_size/1024/1024:.2f} MiB")
            try:
                from google.colab import files as _gfiles
                print("Starting ZIP download from the notebook parent process...")
                _gfiles.download(str(_zp))
            except Exception as _e:
                print("⚠ browser auto-download failed:", _e)
                print("ZIP remains at:", _zp)
        except Exception as _e:
            print("⚠ could not process V86 ZIP marker:", _e)

    _reuse_env = None
    for _rr, _cc, _xx in _reuse_candidates:
        _reuse_env = _try_reuse(_rr, _cc, _xx)
        if _reuse_env is not None:
            print("=" * 94)
            print(" V86 FAST REUSE · existing cu130 + MissingLink Sage SM120 wheel environment found")
            print("=" * 94)
            print("Skipping Torch/CUDA downloads; verified MissingLink Sage wheel is already installed.")
            print("V86 will stream child preload/model-download logs while waiting for the UI.")
            print("Reused target:", _rr)
            _reuse_env["H3_CU130_CHILD"] = "1"
            _reuse_env["H3_V86_REUSED_ENV"] = str(_rr)
            _reuse_env["H3_UI_BLOCKING_CHILD"] = "1"
            _ui_port = _pick_ui_port()
            _reuse_env["H3_UI_PORT"] = str(_ui_port)
            print(f"\nH3 V86 PRODUCTION UI · starting persistent cu130 child on port {_ui_port}\n", flush=True)
            _proc, _child_log = _start_ui_child(_self, _reuse_env, _ui_port)
            _wait_for_ui(_proc, _ui_port, _child_log)
            _show_child_ui(_ui_port, _proc.pid)
            # Parent notebook cell ends normally. The isolated child keeps the
            # model + Flask UI alive until this cell is rerun or the runtime stops.
            _os.environ["H3_V86_UI_PID"] = str(_proc.pid)
            _os.environ["H3_V86_UI_PORT"] = str(_ui_port)
            _os.environ["H3_V86_UI_LOG"] = str(_child_log)
            print("✓ V86 ready. No SystemExit is expected.", flush=True)
            # Do not continue into the full bootstrap after successful reuse.
            _V86_PARENT_DONE = True
            break

    if globals().get("_V86_PARENT_DONE"):
        # Keep the notebook parent out of the CUDA bootstrap/body. The persistent
        # child owns the model and UI.
        pass
    else:
        print("=" * 94)
        print(" MiniMax H3 · CUDA 13.0 PRIVATE TARGET BOOTSTRAP · V86-BW1 · BLACKWELL SM120")
        print("=" * 94)
        print("No venv and no ensurepip. cu130 packages are isolated under /content.")
        print(">>> REQUIRED RUN ID: H3-CU130-ML-SAGE-WHEEL-BW1 <<<")
        print(">>> If your log does not say ML-SAGE-WHEEL / BLACKWELL SM120, you are running an OLD cell/file. <<<")
        print(">>> V86 validates existing safetensors and automatically replaces truncated/corrupt weight files. <<<")

        # Use the already-working parent pip; --target does not invoke ensurepip.
        _pip = [_sys.executable, "-m", "pip"]

        # V86 PARTIAL REUSE: a previous run may already have a perfectly valid
        # Torch 2.11+cu130 target but have failed later while preparing Sage's build
        # frontend. Do not throw away/download the multi-GB Torch/CUDA runtime again.
        _partial_env = _env.copy()
        _partial_env["PYTHONPATH"] = str(_root)
        _partial_env["PYTHONNOUSERSITE"] = "1"
        _torch_only_probe = r"""
import torch
print("PARTIAL-TORCH:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
assert str(torch.version.cuda).startswith("13.0"), torch.version.cuda
assert torch.cuda.get_device_capability(0) == (12,0)
x=torch.randn((32,32),device="cuda",dtype=torch.float16); y=x@x
torch.cuda.synchronize()
"""
        _partial_torch_ok = False
        if _root.exists():
            try:
                _pp = _sp.run([_sys.executable, "-c", _textwrap.dedent(_torch_only_probe).strip()],
                              env=_partial_env, text=True, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                              timeout=45)
                _partial_torch_ok = (_pp.returncode == 0)
                if _partial_torch_ok:
                    print((_pp.stdout or "").strip())
                    print(f"✓ V86 partial reuse: keeping existing Torch 2.11+cu130 target at {_root}; skipping Torch reinstall.")
                else:
                    print("V86 partial Torch target rejected; rebuilding private target.")
                    print((_pp.stdout or "")[-2000:])
            except Exception as _e:
                print("V86 partial Torch probe failed:", repr(_e))

        if not _partial_torch_ok:
            _shutil.rmtree(_root, ignore_errors=True)
            _root.mkdir(parents=True, exist_ok=True)

        def _purge_frontend_artifacts(_target):
            """Remove stale build-frontend packages AND dist-info before exact pinning.

            pip --target can leave multiple setuptools-*.dist-info directories after
            repeated upgrades. setuptools.__version__ is metadata-derived, so stale
            78.x metadata can make a real 74.1.3 install report 78.1.0. That is the
            exact failure V45 hit.
            """
            _target = _pl.Path(_target)
            _patterns = (
                "setuptools", "setuptools-*.dist-info", "setuptools-*.egg-info",
                "pkg_resources", "_distutils_hack", "distutils-precedence.pth",
                "wheel", "wheel-*.dist-info", "wheel-*.egg-info",
                "packaging", "packaging-*.dist-info", "packaging-*.egg-info",
            )
            for _pat in _patterns:
                for _p in _target.glob(_pat):
                    try:
                        if _p.is_dir() and not _p.is_symlink():
                            _shutil.rmtree(_p, ignore_errors=True)
                        else:
                            _p.unlink(missing_ok=True)
                    except Exception:
                        pass

        _purge_frontend_artifacts(_root)
        _run_checked(_pip + [
            "install", "--no-cache-dir", "--no-deps", "--upgrade", "--target", str(_root),
            "wheel==0.43.0", "packaging==23.2", "setuptools==74.1.3", "ninja"
        ], title="[1/6] Installing private bootstrap helpers")

        if not _partial_torch_ok:
            _run_checked(_pip + [
                "install", "--no-cache-dir", "--upgrade", "--target", str(_root),
                "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
                "--index-url", "https://download.pytorch.org/whl/cu130"
            ], title="[2/5] Installing official PyTorch 2.11.0 + cu130")
        else:
            print("[2/5] SKIP · existing private Torch 2.11.0+cu130 is valid")

        # Child import isolation: private target FIRST, then normal stdlib/site paths.
        _child_env = _env.copy()
        _child_env["PYTHONPATH"] = str(_root)
        _child_env["PYTHONNOUSERSITE"] = "1"

        _verify_torch = r"""
    import torch, sys
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("torch file:", torch.__file__)
    print("torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CC:", torch.cuda.get_device_capability(0))
    assert torch.__version__.startswith("2.11.0")
    assert str(torch.version.cuda).startswith("13.0"), torch.version.cuda
    assert torch.cuda.get_device_capability(0) == (12,0)
    """
        print("V86 bootstrap fix: embedded python -c probes are de-indented before execution.")
        _run_checked([_sys.executable, "-c", _textwrap.dedent(_verify_torch).strip()], env=_child_env,
                     title="VERIFY · private torch cu130")

        # PyTorch cu130 supplies the CUDA runtime. Sage source compilation still
        # needs nvcc. V86 checks the preserved private CUDA 13 compiler target FIRST
        # so a rerun after a later bootstrap failure does not redownload ~1GB of
        # compiler packages merely because /usr/local/cuda still points at 12.8.
        _private_existing_nvcc = [x for x in _cuda_pkg_root.glob("nvidia/**/bin/nvcc") if x.is_file()]
        _nvcc = str(_private_existing_nvcc[0]) if _private_existing_nvcc else _shutil.which("nvcc")
        if _nvcc:
            _nv = _run_checked([_nvcc, "--version"], env=_child_env)
            _is13 = "release 13." in (_nv.stdout or "")
            if _is13 and _private_existing_nvcc:
                print("✓ V86 compiler reuse: existing private CUDA 13 nvcc found; skipping compiler reinstall.")
        else:
            _is13 = False

        if not _is13:
            _run_checked(_pip + [
                "install", "--no-cache-dir", "--upgrade", "--target", str(_cuda_pkg_root),
                "--extra-index-url", "https://pypi.nvidia.com",
                "cuda-toolkit[nvcc,cccl,cudart,cublas,cusparse,cusolver,nvrtc]==13.0.2",
                "nvidia-nvvm==13.0.88",
                "nvidia-cuda-crt==13.0.88",
                "nvidia-nvjitlink==13.0.88",
            ], title="[3/8] Installing coherent CUDA Toolkit 13.0.2 compiler/runtime support")

        _probe = r"""
    import pathlib, os
    root=pathlib.Path(os.environ["H3_CUDA_COMPILER_ROOT"])
    hits=list(root.glob("nvidia/**/bin/nvcc"))
    if hits: print(hits[0].parent.parent)
    """
        _p = _run_checked([_sys.executable, "-c", _textwrap.dedent(_probe).strip()],
                          env={**_child_env, "H3_CUDA_COMPILER_ROOT": str(_cuda_pkg_root)})
        _lines = [x.strip() for x in (_p.stdout or "").splitlines() if x.strip()]
        if _lines:
            _cuda_home = _lines[-1]
        elif _nvcc and _is13:
            _cuda_home = str(_pl.Path(_nvcc).resolve().parent.parent)
        else:
            raise RuntimeError("CUDA 13 torch works, but no CUDA 13 nvcc could be located.")

        _child_env["CUDA_HOME"] = _cuda_home
        _child_env["CUDA_PATH"] = _cuda_home
        _child_env["PATH"] = f"{_cuda_home}/bin:" + _child_env.get("PATH", "")
        _child_env["TORCH_CUDA_ARCH_LIST"] = "12.0"
        _child_env["MAX_JOBS"] = "8"
        _child_env["EXT_PARALLEL"] = "4"
        _child_env["NVCC_APPEND_FLAGS"] = ""
        print("CUDA_HOME:", _cuda_home)

        # Install Comfy requirements privately without allowing them to replace
        # the selected torch trio.
        _comfy_req = _pl.Path("/content/ComfyUI/requirements.txt")
        if _comfy_req.exists():
            _filtered = _pl.Path("/content/comfy_requirements_no_torch_v35.txt")
            _keep=[]
            for _line in _comfy_req.read_text().splitlines(True):
                _s=_line.strip().lower()
                _pkg=_re.split(r"[<>=!~\[ ;]", _s, 1)[0] if _s else ""
                if _pkg in {"torch","torchvision","torchaudio"}:
                    continue
                _keep.append(_line)
            _filtered.write_text("".join(_keep))
            _run_checked(_pip + [
                "install", "--no-cache-dir", "--upgrade", "--no-deps", "--target", str(_root),
                "-r", str(_filtered)
            ], env=_child_env, title="[4/6] Installing ComfyUI packages WITHOUT replacing pinned torch")
            # V86: requirements are installed with --no-deps, so they cannot replace
            # Torch. Re-downloading Torch + every CUDA dependency here was redundant,
            # slow, and was the step that reintroduced setuptools 78.1.0 in V45.
            _verify_after_comfy = r"""
import torch
print("post-Comfy torch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.__version__.startswith("2.11.0+cu130")
assert str(torch.version.cuda).startswith("13.0")
"""
            _run_checked([_sys.executable, "-c", _textwrap.dedent(_verify_after_comfy).strip()],
                         env=_child_env, title="[5/7] VERIFY · Torch pin survived Comfy install")

        # Sage is wheel-only. No setuptools/ninja preparation is performed for a
        # Sage source build. The CUDA compiler target below remains available for
        # H3-Optimizations and other runtime CUDA extensions.
        print("[6/7] MissingLink Sage policy · source build disabled · prebuilt wheel required")

        # V86: the compiler target was already installed/verified above. Do not
        # force-reinstall ~1GB of CUDA packages a second time. We verify the private
        # nvcc below and only the earlier [3/8] branch installs it when needed.
        print("[7/8] REUSE · coherent CUDA Toolkit compiler target")

        _find_nvcc = r"""
    import pathlib, os
    root = pathlib.Path(os.environ["H3_CUDA_COMPILER_ROOT"])
    hits = [p for p in root.glob("nvidia/**/bin/nvcc") if p.is_file()]
    for p in hits:
        print(p)
    """
        _nv = _run_checked([_sys.executable, "-c", _textwrap.dedent(_find_nvcc).strip()],
                           env={**_child_env, "H3_CUDA_COMPILER_ROOT": str(_cuda_pkg_root)},
                           title="VERIFY · locating isolated CUDA 13 nvcc")
        _nvcc_hits = [x.strip() for x in (_nv.stdout or "").splitlines()
                      if x.strip().endswith("/nvcc")]
        if not _nvcc_hits:
            raise RuntimeError("CUDA 13 compiler package installed, but no nvcc executable exists.")
        _private_nvcc = _nvcc_hits[-1]
        _cuda_home = str(_pl.Path(_private_nvcc).resolve().parent.parent)
        _child_env["CUDA_HOME"] = _cuda_home
        _child_env["CUDA_PATH"] = _cuda_home
        _child_env["PATH"] = f"{_cuda_home}/bin:" + _child_env.get("PATH", "")
        print("FINAL CUDA_HOME:", _cuda_home)
        print("FINAL nvcc:", _private_nvcc)

        # CUDA 13 moved libcudacxx/Thrust/CUB under include/cccl. Sage's host C++
        # compilation includes cuda_fp16.h, which itself includes <nv/target>.
        # The pip nvcc wheel does not make that relocated path implicit, so locate
        # the official CUDA 13.0 CCCL wheel's nv/target and add its parent.
        _nv_target_hits = [p for p in _cuda_pkg_root.glob("nvidia/**/include/**/nv/target") if p.is_file()]
        if not _nv_target_hits:
            _nv_target_hits = [p for p in _cuda_pkg_root.glob("**/nv/target") if p.is_file()]
        if not _nv_target_hits:
            raise RuntimeError(
                "CUDA 13 CCCL headers were installed but nv/target is missing. "
                "Expected nvidia-cuda-cccl==13.0.85 to provide it."
            )
        _nv_target = _nv_target_hits[0].resolve()
        _cccl_include = str(_nv_target.parent.parent)
        print("CCCL nv/target:", _nv_target)
        print("CCCL include root:", _cccl_include)
        _old_cplus = _child_env.get("CPLUS_INCLUDE_PATH", "")
        _child_env["CPLUS_INCLUDE_PATH"] = _cccl_include + ((":" + _old_cplus) if _old_cplus else "")
        _old_cpath = _child_env.get("CPATH", "")

        # Aggregate NVIDIA dev headers from the coherent compiler toolkit and Torch target.
        # v12 fixed <nv/target> but then Torch CUDAContextLight failed on <cusparse.h>.
        _build_include_dirs = []
        for _base in (_cuda_pkg_root, _root):
            for _inc in _base.glob("nvidia/**/include"):
                if _inc.is_dir():
                    _s = str(_inc.resolve())
                    if _s not in _build_include_dirs:
                        _build_include_dirs.append(_s)
        if _cccl_include not in _build_include_dirs:
            _build_include_dirs.insert(0, _cccl_include)
        print("CUDA build include dirs:", len(_build_include_dirs))
        for _d in _build_include_dirs:
            print("  INCLUDE:", _d)
        # v15: DO NOT inject CUDA/CCCL headers through CPATH/CPLUS_INCLUDE_PATH.
        # nvcc must control its own CUDA header order. v14 globally injected
        # CUDA_HOME/include + CCCL libc++ and caused cuda_bf16.hpp builtin failures.
        # Preserve only the caller's original host include environment here.
        if _old_cplus:
            _child_env["CPLUS_INCLUDE_PATH"] = _old_cplus
        else:
            _child_env.pop("CPLUS_INCLUDE_PATH", None)
        if _old_cpath:
            _child_env["CPATH"] = _old_cpath
        else:
            _child_env.pop("CPATH", None)

        _required_headers = ("nv/target", "cuda_fp16.h", "cusparse.h", "cublas_v2.h", "cusolverDn.h")
        for _hdr in _required_headers:
            _hits = []
            for _d in _build_include_dirs:
                _p = _pl.Path(_d) / _hdr
                if _p.is_file():
                    _hits.append(str(_p))
            print("HEADER", _hdr, "=>", _hits[:3])
            if not _hits:
                raise RuntimeError(f"Required CUDA development header missing: {_hdr}")

        # Prove the exact header chains that killed v11 and v12 before building Sage.
        _probe_cpp = _pl.Path("/content/h3_cuda13_cccl_probe.cpp")
        _probe_cpp.write_text("#include <cuda_fp16.h>\n#include <nv/target>\nint main(){return 0;}\n")
        _run_checked([
            "c++", "-std=c++17", "-fsyntax-only",
            "-I", f"{_root}/torch/include",
            "-I", f"{_root}/torch/include/torch/csrc/api/include",
            *sum((["-I", _d] for _d in _build_include_dirs), []),
            str(_probe_cpp)
        ], env=_child_env, title="VERIFY · Torch CUDA headers + CCCL + cuSPARSE host compile")

        _nvver = _run_checked([_private_nvcc, "--version"], env=_child_env,
                              title="VERIFY · isolated private nvcc executes")
        if "release 13." not in (_nvver.stdout or ""):
            raise RuntimeError("Private nvcc is not CUDA 13.x:\\n" + (_nvver.stdout or ""))

        # v12 accidentally mixed nvcc 13.0 with latest nvvm/crt 13.3, producing PTX 9.3
        # that the CUDA-13.0 ptxas (PTX 9.0) could not assemble. Verify a coherent toolchain.
        _ptxas_hits = [p for p in _cuda_pkg_root.glob("nvidia/**/bin/ptxas") if p.is_file()]
        if not _ptxas_hits:
            raise RuntimeError("Coherent CUDA 13.0.2 toolkit has no ptxas.")
        _private_ptxas = str(_ptxas_hits[0].resolve())
        print("FINAL ptxas:", _private_ptxas)
        _run_checked([_private_ptxas, "--version"], env=_child_env,
                     title="VERIFY · coherent CUDA 13.0 ptxas")
        _cuda_probe = _pl.Path("/content/h3_cuda13_sm120_probe.cu")
        _cuda_probe.write_text('extern "C" __global__ void k(float* x){ if(threadIdx.x==0) x[0]+=1.0f; }\n')
        _nvcc_probe_env = dict(_child_env)
        _nvcc_probe_env.pop("CPATH", None)
        _nvcc_probe_env.pop("CPLUS_INCLUDE_PATH", None)
        _run_checked([
            _private_nvcc, "-std=c++17", "-c", "-arch=sm_120",
            str(_cuda_probe), "-o", "/content/h3_cuda13_sm120_probe.o"
        ], env=_nvcc_probe_env, title="VERIFY · clean nvcc→ptxas SM120 object compile")

        # v11 failsafe: install CUPTI in its own immutable target. This never touches
        # Torch's nvidia namespace and guarantees libcupti.so.13 remains available.
        _run_checked(_pip + [
            "install", "--no-cache-dir", "--upgrade", "--force-reinstall",
            "--target", str(_cuda_runtime_extra),
            "--extra-index-url", "https://pypi.nvidia.com",
            "nvidia-cuda-cupti==13.0.85",
        ], env=_child_env, title="[7.5/8] Installing isolated CUDA 13 CUPTI failsafe")

        # v11: runtime libs come from pristine Torch target + isolated CUPTI target.
        _libdirs = []
        for _base in (_root, _cuda_runtime_extra):
            for _pat in ("nvidia/**/lib", "nvidia/**/lib64", "cuda/**/lib", "cuda/**/lib64"):
                for _p in _base.glob(_pat):
                    if _p.is_dir():
                        _s = str(_p.resolve())
                        if _s not in _libdirs:
                            _libdirs.append(_s)
        _old_ld = _child_env.get("LD_LIBRARY_PATH", "")
        _child_env["LD_LIBRARY_PATH"] = ":".join(_libdirs + ([_old_ld] if _old_ld else []))
        print("Private CUDA library dirs:", len(_libdirs))

        _verify_dynamic = r"""
    import os, pathlib, ctypes
    root = pathlib.Path(os.environ["H3_PRIVATE_ROOT"])
    extra = pathlib.Path(os.environ["H3_CUPTI_ROOT"])
    hits = []
    for base in (root, extra):
        hits += list(base.glob("nvidia/**/lib/libcupti.so.13"))
        hits += list(base.glob("nvidia/**/lib64/libcupti.so.13"))
    print("libcupti hits:", [str(x) for x in hits])
    if not hits:
        raise RuntimeError("libcupti.so.13 absent even after isolated CUPTI failsafe install.")
    ctypes.CDLL(str(hits[0]))
    ctypes.CDLL("libcupti.so.13")
    print("libcupti.so.13 load: OK")
    import torch
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("torch file:", torch.__file__)
    assert torch.__version__.startswith("2.11.0+cu130")
    x=torch.randn((64,64), device="cuda", dtype=torch.float16)
    y=x@x
    torch.cuda.synchronize()
    print("torch CUDA smoke: OK", float(y[0,0]))
    """
        _run_checked([_sys.executable, "-c", _textwrap.dedent(_verify_dynamic).strip()],
                     env={**_child_env,
                          "H3_PRIVATE_ROOT": str(_root),
                          "H3_CUPTI_ROOT": str(_cuda_runtime_extra)},
                     title="VERIFY · Torch cu130 + isolated CUPTI + real CUDA matmul")

        # ------------------------------------------------------------------
        # MissingLink SageAttention wheel ONLY — no GitHub/PyPI/source fallback.
        # missinglink.build serves this filename from the dedicated
        # MissingLinkBuilder/wheels HF Storage Bucket.
        # ------------------------------------------------------------------
        _ml_token = (_child_env.get("MISSING_LINK_TOKEN") or "").strip()
        if not _ml_token:
            raise RuntimeError(
                "MISSING_LINK_TOKEN is required to install the MissingLink SageAttention SM120 wheel."
            )

        # Remove Sage left by older source-building cells before installing the
        # published binary. FAST REUSE later requires the provenance marker below.
        for _pat in ("sageattention", "sageattention-*.dist-info", "sageattention-*.egg-info"):
            for _p in _root.glob(_pat):
                try:
                    if _p.is_dir() and not _p.is_symlink():
                        _shutil.rmtree(_p, ignore_errors=True)
                    else:
                        _p.unlink(missing_ok=True)
                except Exception:
                    pass
        (_root / _ML_SAGE_MARKER).unlink(missing_ok=True)

        _sage_wheel_local = _pl.Path("/content") / _ML_SAGE_WHEEL
        print(f"[8/8] Downloading required MissingLink SageAttention wheel -> {_ML_SAGE_WHEEL}", flush=True)
        try:
            import urllib.request as _sage_urlreq
            _req = _sage_urlreq.Request(
                _ML_SAGE_URL,
                headers={
                    "x-api-key": _ml_token,
                    "User-Agent": "MissingLink-H3-SageWheel/1",
                    "Accept": "application/octet-stream",
                },
            )
            with _sage_urlreq.urlopen(_req, timeout=180) as _resp, open(_sage_wheel_local, "wb") as _fh:
                _shutil.copyfileobj(_resp, _fh)
        except Exception as _e:
            raise RuntimeError(
                "Required MissingLink SageAttention wheel download failed. "
                "This UI intentionally has NO source-build fallback. "
                f"Wheel: {_ML_SAGE_WHEEL}\nUnderlying error: {_e}"
            ) from _e

        if not _sage_wheel_local.exists() or _sage_wheel_local.stat().st_size < 5_000_000:
            raise RuntimeError(
                f"MissingLink SageAttention wheel is missing or implausibly small: {_sage_wheel_local}"
            )

        import zipfile as _zipfile
        try:
            with _zipfile.ZipFile(_sage_wheel_local, "r") as _zf:
                _names = set(_zf.namelist())
                _has_pkg = "sageattention/__init__.py" in _names
                _has_fused = any(n.startswith("sageattention/_fused") and n.endswith(".so") for n in _names)
                # Sage 2.2 dispatches Blackwell through its sm89-family FP8 extension.
                _has_qattn = any(n.startswith("sageattention/_qattn_sm89") and n.endswith(".so") for n in _names)
                if not (_has_pkg and _has_fused and _has_qattn):
                    raise RuntimeError(
                        "wheel lacks required compiled Sage extensions "
                        f"(pkg={_has_pkg}, fused={_has_fused}, qattn={_has_qattn})"
                    )
        except Exception as _e:
            raise RuntimeError(f"Invalid MissingLink SageAttention wheel: {_e}") from _e

        _run_checked([
            _sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
            "--force-reinstall", "--target", str(_root), str(_sage_wheel_local),
        ], env=_child_env, title="[8/8] Installing MissingLink SageAttention 2.2.0 · cu130 · sm120 wheel")

        _verify = r"""
    import torch, pathlib
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("torch file:", torch.__file__)
    assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
    assert str(torch.version.cuda).startswith("13.0"), torch.version.cuda
    from sageattention import sageattn
    import sageattention
    print("sage:", sageattention.__file__)
    q=torch.randn(1,24,1024,128,device="cuda",dtype=torch.bfloat16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    with torch.no_grad():
        y=sageattn(q,k,v,tensor_layout="HND",is_causal=False)
    torch.cuda.synchronize()
    print("Sage BF16 sm120 self-test:", tuple(y.shape), "OK")
    """
        _run_checked([_sys.executable, "-c", _textwrap.dedent(_verify).strip()], env=_child_env,
                     title="VERIFY · cu130 + MissingLink SageAttention sm120 wheel")
        (_root / _ML_SAGE_MARKER).write_text(_ML_SAGE_WHEEL)
        print(f"✓ MissingLink Sage wheel provenance marker written: {_root / _ML_SAGE_MARKER}", flush=True)

        print("\nLaunching V86 production UI with private cu130 + MissingLink Sage SM120 wheel first on PYTHONPATH.\n")
        _child_env["H3_CU130_CHILD"] = "1"
        _child_env["H3_UI_BLOCKING_CHILD"] = "1"
        _ui_port = _pick_ui_port()
        _child_env["H3_UI_PORT"] = str(_ui_port)
        print(f"\nH3 V86 PRODUCTION UI · starting persistent freshly-built cu130 child on port {_ui_port}\n", flush=True)
        _proc, _child_log = _start_ui_child(_self, _child_env, _ui_port)
        _wait_for_ui(_proc, _ui_port, _child_log)
        _show_child_ui(_ui_port, _proc.pid)
        _os.environ["H3_V86_UI_PID"] = str(_proc.pid)
        _os.environ["H3_V86_UI_PORT"] = str(_ui_port)
        _os.environ["H3_V86_UI_LOG"] = str(_child_log)
        print("✓ V86 ready. No SystemExit is expected.", flush=True)

# Child continues below with private torch+cu130 and Sage sm120 first on PYTHONPATH.
if _CU130_CHILD:

    import os, re, sys, gc, time, json, uuid, stat, shutil, socket, asyncio, base64, io
    import threading, subprocess, traceback, urllib.request

    # Parent chooses a free port and passes it to this isolated UI process.
    UI_PORT = int(os.environ.get("H3_UI_PORT", str(UI_PORT)))

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
    log("  MiniMax H3 · Standalone Auto GPU Studio   (Blackwell + A100 80/40GB + T4 Q4 Dynamic VRAM)")
    log("="*74)

    # ── MissingLink access gate ───────────────────────────────────────────────
    MACHINE = os.environ.get('MACHINE', 'a100')
    ML_OK, ML_AUTH_ERROR = _validate_missinglink_token(force=False)
    if not ML_OK:
        raise SystemExit(
            "MissingLink API key validation failed before UI startup: "
            + (ML_AUTH_ERROR or "unknown auth error")
            + f"\nStart a 7-day trial / get access: {MISSING_LINK_TRIAL_URL}"
        )
    REQUESTED_GPU_PROFILE = (os.environ.get("H3_GPU_PROFILE") or "blackwell_sm120").strip().lower()
    LOWVRAM_T4_REQUESTED = REQUESTED_GPU_PROFILE == "t4_16gb"
    A100_REQUESTED = REQUESTED_GPU_PROFILE in {"a100_80gb", "a100_40gb"}

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

    def _fetch_source_archive(urls, work_key, min_bytes=10000):
        """Fetch/extract a GitHub source archive without invoking git at all."""
        import tarfile as _tarfile
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(work_key))
        archive = f"/content/_h3_{safe_key}.tar.gz"
        unpack = f"/content/_h3_{safe_key}_unpack"
        errors = []
        for url in urls:
            try:
                try: os.remove(archive)
                except FileNotFoundError: pass
                shutil.rmtree(unpack, ignore_errors=True)
                os.makedirs(unpack, exist_ok=True)
                log(f"  ↓ {work_key} archive -> {url}")
                ok = False
                curl = shutil.which("curl")
                if curl:
                    rr = subprocess.run([
                        curl, "-fL", "--retry", "6", "--retry-delay", "2",
                        "--retry-all-errors", "--connect-timeout", "30",
                        "--max-time", "300", "-A", "Mozilla/5.0 H3-NoGitBootstrap/49",
                        "-o", archive, url,
                    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    ok = rr.returncode == 0 and os.path.isfile(archive) and os.path.getsize(archive) >= min_bytes
                    if not ok:
                        errors.append(f"curl {url}: rc={rr.returncode}\n{rr.stdout[-1200:]}")
                if not ok:
                    try:
                        import urllib.request as _urlreq
                        req = _urlreq.Request(url, headers={
                            "User-Agent": "Mozilla/5.0 H3-NoGitBootstrap/49",
                            "Accept": "application/octet-stream",
                        })
                        with _urlreq.urlopen(req, timeout=120) as resp, open(archive, "wb") as fh:
                            shutil.copyfileobj(resp, fh)
                        ok = os.path.isfile(archive) and os.path.getsize(archive) >= min_bytes
                    except Exception as e:
                        errors.append(f"urllib {url}: {e!r}")
                if not ok:
                    continue

                with _tarfile.open(archive, "r:gz") as tf:
                    try:
                        tf.extractall(unpack, filter="data")
                    except TypeError:
                        tf.extractall(unpack)
                roots = [os.path.join(unpack, n) for n in os.listdir(unpack)
                         if os.path.isdir(os.path.join(unpack, n))]
                if len(roots) == 1:
                    log(f"  ✓ {work_key} source extracted without git")
                    return roots[0]
                # GitHub archives should have a single top-level directory. If not,
                # accept the unpack root only if it contains obvious source files.
                if roots:
                    log(f"  ✓ {work_key} archive extracted ({len(roots)} roots)")
                    return unpack
                errors.append(f"{url}: archive extracted no source directory")
            except Exception as e:
                errors.append(f"{url}: {e!r}")
        raise RuntimeError(
            f"Could not fetch {work_key} source archive without git.\n" +
            "\n---\n".join(errors[-6:])
        )

    def _ensure_comfyui_source():
        global COMFY_DIR

        candidates = []
        for c in (
            os.environ.get("COMFY_DIR", "").strip(),
            COMFY_DIR,
            "/content/Standalone-Extras/ComfyUI",
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
        log("  ↓ ComfyUI source missing — fetching official Comfy-Org/ComfyUI archive (NO GIT)")
        src_root = _fetch_source_archive([
            "https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/refs/heads/master",
            "https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/refs/heads/main",
            "https://github.com/Comfy-Org/ComfyUI/archive/refs/heads/master.tar.gz",
            "https://github.com/Comfy-Org/ComfyUI/archive/refs/heads/main.tar.gz",
        ], "ComfyUI", min_bytes=100000)

        # Merge source into target but never overwrite/delete models/. This works
        # even when the downloader has already created /content/ComfyUI/models.
        for item in os.listdir(src_root):
            if item in (".git", "models"):
                continue
            s = os.path.join(src_root, item)
            d = os.path.join(target, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        log(f"✓ ComfyUI source archive merged into {target} without git")

        if not _is_comfy_source(target):
            raise RuntimeError(
                f"ComfyUI bootstrap finished but required H3 source files are missing in {target}")

        # Install ComfyUI dependencies except torch/torchvision/torchaudio; replacing
        # Colab's CUDA PyTorch build here can break the pinned Blackwell/cu130 runtime.
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

    # Fast-start: avoid invoking pip on every rerun when imports already work.
    import importlib.util as _importlib_util
    _pkg_imports = {"flask":"flask", "nest_asyncio":"nest_asyncio", "av":"av", "huggingface_hub":"huggingface_hub"}
    _missing_pkgs = [pkg for pkg, mod in _pkg_imports.items() if _importlib_util.find_spec(mod) is None]
    if _missing_pkgs:
        log("  ↓ installing missing Python packages: " + ", ".join(_missing_pkgs))
        subprocess.run([sys.executable,"-m","pip","install","-q",*_missing_pkgs], check=False)
    else:
        log("✓ Python runtime packages already present · pip step skipped")

    # T4 / <=18.5 GiB profile: install molbal's maintained ComfyUI-GGUF fork,
    # which includes MiniMax-H3 support and a Dynamic VRAM loader. Keep this out
    # of the Blackwell path so V86-BW1 remains unchanged.
    GGUF_NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-GGUF")
    if LOWVRAM_T4_REQUESTED:
        if FAST_STARTUP and os.path.isfile(os.path.join(GGUF_NODE_DIR, "__init__.py")):
            log("✓ ComfyUI-GGUF reused from prebuilt/local custom_nodes · network refresh skipped")
        else:
            log("  ↓ T4 low-VRAM dependency: molbal/ComfyUI-GGUF (MiniMax-H3 Dynamic VRAM)")
            gguf_src = _fetch_source_archive([
                "https://codeload.github.com/molbal/ComfyUI-GGUF/tar.gz/refs/heads/main",
                "https://github.com/molbal/ComfyUI-GGUF/archive/refs/heads/main.tar.gz",
            ], "ComfyUI-GGUF", min_bytes=10000)
            shutil.rmtree(GGUF_NODE_DIR, ignore_errors=True)
            os.makedirs(os.path.dirname(GGUF_NODE_DIR), exist_ok=True)
            shutil.copytree(gguf_src, GGUF_NODE_DIR, dirs_exist_ok=True)
            req = os.path.join(GGUF_NODE_DIR, "requirements.txt")
            if os.path.isfile(req):
                rr = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if rr.returncode != 0:
                    raise RuntimeError("ComfyUI-GGUF dependency install failed:\n" + rr.stdout[-3000:])
            log("✓ ComfyUI-GGUF ready · molbal/main · T4 Dynamic VRAM loader enabled")

    # Current ComfyUI initializes comfy-aimdo BEFORE torch/model_management during
    # normal main.py startup. This studio imports ComfyUI as a library, so reproduce
    # that ordering explicitly for T4. The previous T4 leg only set args.lowvram,
    # leaving comfy.memory_management.aimdo_enabled=False; molbal's Dynamic GGUF
    # loader correctly refused to run in that state.
    _aimdo_control = None
    _AIMDO_CONTROL_PREINIT = False
    if LOWVRAM_T4_REQUESTED:
        if COMFY_DIR not in sys.path:
            sys.path.insert(0, COMFY_DIR)
        try:
            import comfy.options as _pre_comfy_options
            _pre_comfy_options.enable_args_parsing(False)
            from comfy.cli_args import args as _pre_args
            _pre_args.reserve_vram = T4_RESERVE_VRAM_GIB
            if hasattr(_pre_args, "enable_dynamic_vram"):
                _pre_args.enable_dynamic_vram = True
            if hasattr(_pre_args, "disable_dynamic_vram"):
                _pre_args.disable_dynamic_vram = False
            if hasattr(_pre_args, "highvram"):
                _pre_args.highvram = False
            if hasattr(_pre_args, "gpu_only"):
                _pre_args.gpu_only = False
            if hasattr(_pre_args, "novram"):
                _pre_args.novram = False
            if hasattr(_pre_args, "cpu"):
                _pre_args.cpu = False
            if hasattr(_pre_args, "lowvram"):
                _pre_args.lowvram = True

            import comfy_aimdo.control as _aimdo_control
            _simple_headroom = int(T4_RESERVE_VRAM_GIB * 1024**3)
            try:
                _aimdo_control.init(
                    simple_vram_headroom=_simple_headroom,
                    nvml_pressure=not bool(getattr(_pre_args, "disable_nvml_pressure", False)),
                )
            except TypeError:
                try:
                    _aimdo_control.init(simple_vram_headroom=_simple_headroom)
                except TypeError:
                    _aimdo_control.init()
            _AIMDO_CONTROL_PREINIT = True
            log(f"  ✓ T4 DynamicVRAM controller pre-init complete · headroom {T4_RESERVE_VRAM_GIB:.1f} GiB")
        except Exception as _e:
            raise RuntimeError(
                "Could not pre-initialize Comfy DynamicVRAM/Aimdo for the T4 profile. "
                f"Current ComfyUI + comfy-aimdo are required: {_e}"
            ) from _e

    # ── V30: install/update H3-Optimizations before nodes.init_extra_nodes() ─────
    # H3 sparse optimization nodes are enabled on Blackwell and A100. T4 stays
    # dense SDPA and deliberately does not depend on the sparse-node pack.
    H3OPT_DIR = os.path.join(COMFY_DIR, "custom_nodes", "H3-Optimizations")
    _h3opt_commit = ""
    if not LOWVRAM_T4_REQUESTED:
        try:
            if FAST_STARTUP and os.path.isfile(os.path.join(H3OPT_DIR, "__init__.py")):
                _h3opt_commit = "reused-local"
                log("✓ H3-Optimizations reused from local custom_nodes · network refresh skipped")
            else:
                src_root = _fetch_source_archive([
                    "https://codeload.github.com/Zironic/H3-Optimizations/tar.gz/refs/heads/main",
                    "https://github.com/Zironic/H3-Optimizations/archive/refs/heads/main.tar.gz",
                ], "H3-Optimizations", min_bytes=10000)
                shutil.rmtree(H3OPT_DIR, ignore_errors=True)
                os.makedirs(os.path.dirname(H3OPT_DIR), exist_ok=True)
                shutil.copytree(src_root, H3OPT_DIR, dirs_exist_ok=True)
                if not os.path.isfile(os.path.join(H3OPT_DIR, "__init__.py")):
                    raise RuntimeError("H3-Optimizations archive is missing __init__.py")
                _h3opt_commit = "archive-main"
                log("✓ H3-Optimizations ready · archive main · NO GIT")
        except Exception as _e:
            raise RuntimeError("MiniMax H3 Studio requires H3-Optimizations sparse nodes: " + repr(_e))
    else:
        log("✓ T4 low-VRAM: sparse/Kitchen node pack skipped; dense PyTorch SDPA only")

    import torch, numpy as np
    from PIL import Image, ImageOps
    try:
        import av
    except Exception:
        av = None
    try:
        import torchaudio
    except Exception:
        torchaudio = None

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Runtime -> Change runtime type -> GPU.")

    if LOWVRAM_T4_REQUESTED:
        print(f"✓ V91 T4/LOW-VRAM RUNTIME: Torch {torch.__version__} · CUDA {torch.version.cuda} · "
              f"{torch.cuda.get_device_name(0)}", flush=True)
    elif A100_REQUESTED:
        # A100/SM80 uses the Colab runtime directly; do not force the SM120 cu130 build.
        print(f"✓ V91 A100 NATIVE RUNTIME: Torch {torch.__version__} · CUDA {torch.version.cuda} · "
              f"{torch.cuda.get_device_name(0)}", flush=True)
    else:
        if not torch.__version__.startswith("2.11.0+cu130") or not str(torch.version.cuda).startswith("13.0"):
            raise RuntimeError(
                f"V91 Blackwell requires private Torch 2.11+cu130; got torch={torch.__version__}, "
                f"CUDA={torch.version.cuda}. The parent bootstrap must launch this UI child."
            )
        print(f"✓ V91 BLACKWELL PRIVATE RUNTIME: Torch {torch.__version__} · CUDA {torch.version.cuda} · "
              f"{torch.cuda.get_device_name(0)}", flush=True)

    from flask import Flask, request, jsonify, Response, send_file, session

    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    disk = shutil.disk_usage("/content").free / 1e9
    log(f"  {gpu} · {vram:.0f} GB VRAM · {disk:.0f} GB disk free")
    GPU_CC = torch.cuda.get_device_capability(0)
    PHYSICAL_VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    IS_BLACKWELL_SM120 = GPU_CC == (12, 0)
    IS_A100_SM80 = GPU_CC == (8, 0) and "A100" in gpu.upper()
    LOWVRAM_T4_PROFILE = bool(LOWVRAM_T4_REQUESTED or PHYSICAL_VRAM_GIB <= T4_LOWVRAM_MAX_GIB)
    A100_PROFILE = bool((not LOWVRAM_T4_PROFILE) and (A100_REQUESTED or IS_A100_SM80))
    A100_80_PROFILE = bool(A100_PROFILE and PHYSICAL_VRAM_GIB >= A100_80_MIN_GIB)
    A100_40_PROFILE = bool(A100_PROFILE and not A100_80_PROFILE)
    GPU_PROFILE = (
        "t4_16gb" if LOWVRAM_T4_PROFILE else
        ("a100_80gb" if A100_80_PROFILE else ("a100_40gb" if A100_40_PROFILE else "blackwell_sm120"))
    )

    # Human-readable architecture string used by the browser UI. Keep this derived
    # from the actual detected card rather than a hard-coded Blackwell label.
    _sm_label = f"SM{GPU_CC[0]}{GPU_CC[1]}"
    if LOWVRAM_T4_PROFILE:
        GPU_ARCH_LABEL = f"T4 {_sm_label} · {PHYSICAL_VRAM_GIB:.0f}GB"
    elif A100_80_PROFILE:
        GPU_ARCH_LABEL = f"A100 {_sm_label} · 80GB"
    elif A100_40_PROFILE:
        GPU_ARCH_LABEL = f"A100 {_sm_label} · 40GB"
    else:
        GPU_ARCH_LABEL = f"BLACKWELL {_sm_label} · {PHYSICAL_VRAM_GIB:.0f}GB"

    if LOWVRAM_T4_PROFILE:
        RESERVE_VRAM = T4_RESERVE_VRAM_GIB
        LOWVRAM = True
        AUTO_OPTIMIZE_ATTENTION = False
        TEXT_ENCODER_FILE = T4_TEXT_ENCODER_FILE
        TEXT_ENCODER_GIB = T4_TEXT_ENCODER_GIB
        BLACKWELL_FULL_CARD = False
        log(
            f"  ✓ LOW-VRAM profile verified: {gpu} · CC {GPU_CC} · {PHYSICAL_VRAM_GIB:.1f} GiB · "
            f"Q4_0 GGUF + Dynamic VRAM · reserve {RESERVE_VRAM:.1f} GiB"
        )
    elif A100_PROFILE:
        if not IS_A100_SM80:
            raise RuntimeError(
                f"A100 profile requested but detected GPU={gpu!r}, capability={GPU_CC}, "
                f"VRAM={PHYSICAL_VRAM_GIB:.1f} GiB."
            )
        LOWVRAM = False
        AUTO_OPTIMIZE_ATTENTION = False
        if A100_80_PROFILE:
            RESERVE_VRAM = A100_80_RESERVE_VRAM_GIB
            BLACKWELL_FULL_CARD = True  # compatibility alias: enables HIGH_VRAM path
            log(
                f"  ✓ A100 80GB profile verified: SM80 · {PHYSICAL_VRAM_GIB:.1f} GiB physical · "
                f"quality INT8 TE · full-card residency eligible · reserve {RESERVE_VRAM:.1f} GiB"
            )
        else:
            RESERVE_VRAM = A100_40_RESERVE_VRAM_GIB
            TEXT_ENCODER_FILE = A100_40_TEXT_ENCODER_FILE
            TEXT_ENCODER_GIB = A100_40_TEXT_ENCODER_GIB
            BLACKWELL_FULL_CARD = False
            log(
                f"  ✓ A100 40GB profile verified: SM80 · {PHYSICAL_VRAM_GIB:.1f} GiB physical · "
                f"smaller official conditioning TE · sequential TE→DiT→VAE handoff · "
                f"reserve {RESERVE_VRAM:.1f} GiB"
            )
    else:
        if not IS_BLACKWELL_SM120:
            raise RuntimeError(
                f"V91 supports Blackwell SM120, A100 SM80 (40/80GB), or the <=18.5 GiB/T4 leg; "
                f"got GPU={gpu!r}, capability={GPU_CC}, VRAM={PHYSICAL_VRAM_GIB:.1f} GiB."
            )
        BLACKWELL_FULL_CARD = PHYSICAL_VRAM_GIB >= BLACKWELL_FULL_CARD_MIN_GIB
        log(
            f"  ✓ Blackwell target verified: SM120 · {PHYSICAL_VRAM_GIB:.1f} GiB physical · "
            + ("full-card resident profile" if BLACKWELL_FULL_CARD else "partitioned-card fallback profile")
        )


    # ── GPU runtime + dense attention policy ─────────────────────────────────────
    # Blackwell keeps the SM120 Sage-vs-SDPA autotune. A100 uses native PyTorch SDPA
    # to avoid a CUDA-toolchain build on Colab and prioritize launch reliability.
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
        """Require the prebuilt MissingLink SageAttention 2.2 SM120/cu130 wheel."""
        import importlib, importlib.util, pathlib
        required_wheel = "sageattention-2.2.0-1sm120cu130-cp313-cp313-linux_x86_64.whl"
        marker_name = ".missinglink_sageattention_2.2.0_sm120cu130"
        try:
            importlib.invalidate_caches()
            import sageattention
            from sageattention import sageattn  # noqa: F401
            ext = importlib.util.find_spec("sageattention._qattn_sm89")
            if ext is None:
                raise RuntimeError("sageattention._qattn_sm89 is missing for SM120 FP8 dispatch")
            root = pathlib.Path(sageattention.__file__).resolve().parent.parent
            marker = root / marker_name
            if not marker.exists() or marker.read_text(errors="ignore").strip() != required_wheel:
                raise RuntimeError(
                    "SageAttention is present but is not the MissingLink-published SM120/cu130 wheel"
                )
            note = f"MissingLink wheel · SM120 FP8 extension: {getattr(ext, 'origin', 'found')}"
            log(f"  ✓ SageAttention {SAGEATTN_VERSION} present | {note}")
            return True, note
        except Exception as e:
            log(f"  ✗ REQUIRED MissingLink SageAttention wheel unavailable/broken: {e!r}")
            return False, repr(e)

    def _autotune_attention():
        global ATTN_BACKEND, ATTN_BENCH
        cap = torch.cuda.get_device_capability(0)
        log(f"  dense attention autotune -> GPU capability sm_{cap[0]}{cap[1]} · seq={ATTN_BENCH_SEQ}")
        B,H,S,D = 1, ATTN_BENCH_HEADS, ATTN_BENCH_SEQ, ATTN_BENCH_DIM
        q = torch.randn((B,H,S,D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        try:
            sdpa_ms = _timed_cuda(sdpa, warmup=3, iters=5)
        except Exception as e:
            sdpa_ms = float("inf")
            log(f"  ⚠ SDPA benchmark failed: {e}")

        sage_ms = float("inf")
        sage_ok = False
        install_note = "disabled"
        if AUTO_OPTIMIZE_ATTENTION:
            sage_ok, install_note = _ensure_sageattention()
        if not sage_ok:
            raise RuntimeError(
                "Blackwell runtime requires the MissingLink SageAttention 2.2.0 SM120/cu130 wheel; "
                f"validation failed: {install_note}"
            )
        try:
            from sageattention import sageattn
            sage = lambda: sageattn(q,k,v,tensor_layout="HND",is_causal=False)
            sage_ms = _timed_cuda(sage, warmup=3, iters=5)
        except Exception as e:
            raise RuntimeError(f"Required MissingLink SageAttention SM120 self-test failed: {e}") from e

        ATTN_BENCH = {"sdpa_ms": sdpa_ms, "sage_ms": sage_ms,
                      "sage_ok": bool(sage_ok), "sage_install": install_note,
                      "seq": ATTN_BENCH_SEQ}
        # Use Sage only when it wins by a measurable margin. The sparse H3 path has
        # its own resolver and will prefer native Kitchen SM120 when compatible.
        if sage_ok and sage_ms < sdpa_ms * 0.97:
            ATTN_BACKEND = "sageattention"
            speedup = sdpa_ms / sage_ms
            log(f"  ✓ dense attention: SageAttention {SAGEATTN_VERSION} · {sage_ms:.2f} ms vs SDPA {sdpa_ms:.2f} ms · {speedup:.2f}x")
        else:
            ATTN_BACKEND = "pytorch-sdpa"
            if sage_ok:
                log(f"  ✓ dense attention: PyTorch SDPA · {sdpa_ms:.2f} ms vs Sage {sage_ms:.2f} ms · Sage lacked 3% win")
            else:
                log(f"  ✓ dense attention: PyTorch SDPA · {sdpa_ms:.2f} ms · Sage unavailable")
        del q,k,v
        gc.collect(); torch.cuda.empty_cache()

    if LOWVRAM_T4_PROFILE:
        ATTN_BACKEND = "pytorch-sdpa"
        ATTN_BENCH = {"profile": "t4_16gb", "note": "Sage/Kitchen disabled on low-VRAM leg"}
        log("  ✓ T4/LOW-VRAM attention -> PyTorch SDPA (Sage/Kitchen disabled)")
    elif A100_PROFILE:
        ATTN_BACKEND = "pytorch-sdpa"
        ATTN_BENCH = {"profile": GPU_PROFILE, "note": "native PyTorch SDPA reliability profile on SM80"}
        log(f"  ✓ {GPU_PROFILE.upper()} dense attention -> PyTorch SDPA (native SM80 path)")
    else:
        if FAST_STARTUP:
            _sage_ok, _sage_note = _ensure_sageattention()
            if not _sage_ok:
                raise RuntimeError("Required MissingLink SageAttention wheel unavailable: " + str(_sage_note))
            ATTN_BACKEND = "sageattention"
            ATTN_BENCH = {"profile": "blackwell_sm120", "note": "fast-start: verified Sage wheel; benchmark skipped"}
            log("  ✓ Blackwell fast-start attention -> SageAttention (startup benchmark skipped)")
        else:
            _autotune_attention()

    # ── 1. Weights ─────────────────────────────────────────────────────────────
    from huggingface_hub import hf_hub_download, HfApi, hf_hub_url
    if DIT_CHOICE not in DITS:
        raise RuntimeError(f"DIT_CHOICE must be one of {list(DITS)}")
    _repo, _sub, DIT_FILE = DITS[DIT_CHOICE]

    if "nvfp4" in DIT_CHOICE and not any(k in gpu for k in ("B200","RTX 50","GB200","RTX 60")):
        log(f"  ⚠ {DIT_CHOICE} is a Blackwell profile and {gpu} is not Blackwell.\n"
            f"    Expect a slow emulated path or a load failure — use the stock INT8 profile instead.")

    MODELS = os.path.join(COMFY_DIR, "models")

    def _purge_legacy_adult_assets():
        # User requested an studio: remove the legacy adult assets this notebook
        # itself used to manage, including filenames restored from its old catalog cache.
        exact = {
            "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors",
            "10Eros_Max_h3_TURBO-hybrid_beta5.safetensors",
            "REDMix-MiniMaxH3-A2Ab2-pruned-int8-convrot-ComfyMCP.safetensors",
            "redcraftREDMIXHybridA2A_h3A2AREDBeta2.safetensors",
            "NaughtyTimes_pruned_r256_v2.safetensors",
        }
        lora_dir = os.path.join(MODELS,"loras")
        old_cache = os.path.join(lora_dir,".h3_adult_lora_catalog.json")
        try:
            if os.path.exists(old_cache):
                rows=json.load(open(old_cache,"r",encoding="utf-8"))
                for row in rows if isinstance(rows,list) else []:
                    fn=os.path.basename(str((row or {}).get("file") or ""))
                    if fn: exact.add(fn)
        except Exception:
            pass
        removed=[]
        for folder in (os.path.join(MODELS,"diffusion_models"), lora_dir):
            for fn in exact:
                pth=os.path.join(folder,fn)
                if os.path.isfile(pth):
                    try: os.remove(pth); removed.append(fn)
                    except Exception as e: log(f"  ⚠ could not remove legacy managed asset {fn}: {e}")
        for cache_name in (".h3_adult_lora_catalog.json", ".h3_lora_sources.json"):
            try: os.remove(os.path.join(lora_dir,cache_name))
            except FileNotFoundError: pass
            except Exception: pass
        if removed: log("✓ legacy asset cleanup removed: " + ", ".join(sorted(set(removed))))

    # User-owned model assets are never deleted by the studio.

    # Custom base models are installed explicitly from Hugging Face through the UI.
    def _activate_t4_lowvram_q4():
        global USING_REDMIX, USING_EROS_MAX, DIT_FILE, LIGHTNING_DEFAULT
        global TEXT_ENCODER_FILE, TEXT_ENCODER_GIB
        USING_REDMIX = False
        USING_EROS_MAX = False
        DIT_FILE = T4_DIT_FILE
        TEXT_ENCODER_FILE = T4_TEXT_ENCODER_FILE
        TEXT_ENCODER_GIB = T4_TEXT_ENCODER_GIB
        LIGHTNING_DEFAULT = False
        log("!")
        log("  ✓ T4 / LOW-VRAM DEFAULT selected")
        log(f"  ↳ {T4_DIT_FILE} · Q4_0 GGUF · molbal/ComfyUI-GGUF Dynamic VRAM")
        log(f"  ↳ conditioning TE: {T4_TEXT_ENCODER_FILE} · sequential CPU/GPU handoff")
        log("!")

    def _activate_stock_h3_safe():
        """studio startup using the official Stock MiniMax H3 checkpoint."""
        global USING_REDMIX, USING_EROS_MAX, DIT_FILE, LIGHTNING_DEFAULT
        USING_REDMIX = False
        USING_EROS_MAX = False
        DIT_FILE = FALLBACK_DIT_FILE
        LIGHTNING_DEFAULT = True
        log("!")
        log("  ✓ DEFAULT selected: official Stock MiniMax H3 FL2VA")
        log("  ↳ additional Hugging Face base models can be added from the Studio UI")
        log("  ↳ quality recipe: RES Multistep / Simple · 20 steps · video shift 12 · audio shift 3")
        log("!")

    if LOWVRAM_T4_PROFILE:
        _activate_t4_lowvram_q4()
        FILES = [
                 ("text_encoders", TEXT_ENCODER_FILE,
                  "Comfy-Org/MiniMax-H3", "text_encoders"),
                 ("vae", "minimax_h3_video_vae_fp16.safetensors",
                  "Comfy-Org/MiniMax-H3", "vae"),
                 ("vae", "minimax_h3_audio_vae_fp32.safetensors",
                  "Comfy-Org/MiniMax-H3", "vae"),
                 ("unet", T4_DIT_FILE, T4_DIT_REPO, None),
        ]
    else:
        _activate_stock_h3_safe()
        FILES = [
                 ("text_encoders", TEXT_ENCODER_FILE,
                  "Comfy-Org/MiniMax-H3", "text_encoders"),
                 ("vae", "minimax_h3_video_vae_fp16.safetensors",
                  "Comfy-Org/MiniMax-H3", "vae"),
                 ("vae", "minimax_h3_audio_vae_fp32.safetensors",
                  "Comfy-Org/MiniMax-H3", "vae"),
                 ("diffusion_models", FALLBACK_DIT_FILE,
                  "Comfy-Org/MiniMax-H3", "diffusion_models"),
        ]

    # V86 uses the higher-fidelity INT8 ConvRot text encoder. The initial
    # download is larger, but this removes NVFP4/AWQ conditioning from the
    # quality path.
    # concurrently. The text encoder is ~14.61 GiB instead of ~25.28 GiB INT8, so the
    # first launch still downloads concurrently so the larger quality encoder does not serialize startup.
    # hf_hub_download is thread-safe (per-file locks in the cache), so they
    # now run concurrently: total wall time collapses toward the single
    # largest file. Progress bars interleave; the byte counts still climb.
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _validate_safetensors_file(path):
        """Cheap structural validation: reads the safetensors header/offset table,
        not all tensor payloads. This catches truncated/incomplete weight files
        before Comfy tries to load them.
        """
        if not os.path.exists(path):
            return False, "missing"
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                # Force header/key-table parsing. No tensor payload is materialized.
                _ = list(f.keys())
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _atomic_copy_weight(src_path, dest_path):
        tmp = dest_path + ".incoming"
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        shutil.copyfile(src_path, tmp)
        if dest_path.lower().endswith(".safetensors"):
            ok, err = _validate_safetensors_file(tmp)
            if not ok:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                raise RuntimeError(f"downloaded safetensors failed validation: {err}")
        os.replace(tmp, dest_path)

    def _fetch_one(job):
        sub, fname, repo, remote_sub = job
        os.makedirs(os.path.join(MODELS, sub), exist_ok=True)
        dest = os.path.join(MODELS, sub, fname)

        # V86: existence alone is not enough. A prior interrupted Colab/HF copy can
        # leave a huge-looking .safetensors file whose header says "file not fully
        # covered". Validate every existing safetensors file before accepting it.
        force_download = False
        if os.path.exists(dest):
            if fname.lower().endswith(".safetensors"):
                ok, err = _validate_safetensors_file(dest)
                if ok:
                    log(f"  ✓ {fname} (safetensors validated)")
                    return
                bad_size = os.path.getsize(dest) / 1024**3
                log(f"  ⚠ corrupt/incomplete weight detected: {fname} · {bad_size:.2f} GiB")
                log(f"    ↳ {err}")
                log("    ↳ deleting broken local copy and forcing a clean Hugging Face download")
                try:
                    os.remove(dest)
                except FileNotFoundError:
                    pass
                force_download = True
            else:
                log(f"  ✓ {fname}")
                return

        log(f"  ↓ {fname}  ({repo})" + (" · FORCED CLEAN REDOWNLOAD" if force_download else ""))

        # Two attempts: the first normally uses the HF cache; a validation failure
        # forces a fresh cache download on the second attempt as well.
        last_err = None
        for attempt in range(2):
            try:
                p = hf_hub_download(
                    repo,
                    filename=fname,
                    subfolder=remote_sub,
                    force_download=bool(force_download or attempt > 0),
                )
                if os.path.abspath(p) == os.path.abspath(dest):
                    if fname.lower().endswith(".safetensors"):
                        ok, err = _validate_safetensors_file(dest)
                        if not ok:
                            raise RuntimeError(f"Hugging Face result failed safetensors validation: {err}")
                else:
                    _atomic_copy_weight(p, dest)

                if fname.lower().endswith(".safetensors"):
                    ok, err = _validate_safetensors_file(dest)
                    if not ok:
                        raise RuntimeError(f"installed safetensors failed validation: {err}")
                log(f"  ✓ {fname} (validated)")
                return
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                except Exception:
                    pass
                if attempt == 0:
                    log(f"  ⚠ {fname} validation/download retry: {e}")
                    continue
        raise RuntimeError(f"Could not install a valid {fname}: {last_err}")

    with _TPE(max_workers=4) as _pool:
        # list() propagates the first exception instead of swallowing it —
        # a missing weight must still be a loud failure, exactly as before.
        list(_pool.map(_fetch_one, FILES))
    log(f"✓ transformer: {DIT_FILE}")

    # Integrity check for the active default transformer. Fast-start avoids an
    # unnecessary full-file SHA pass for the official stock safetensors because the
    # file was already structurally validated above and there is no pinned SHA here.
    import hashlib as _hashlib
    if LOWVRAM_T4_PROFILE:
        _active_integrity_path = os.path.join(MODELS, "unet", T4_DIT_FILE)
        _active_expected_sha = T4_DIT_SHA256
        _active_integrity_label = "T4 Q4_0 GGUF"
    else:
        _active_integrity_path = os.path.join(MODELS, "diffusion_models", FALLBACK_DIT_FILE)
        _active_expected_sha = ""
        _active_integrity_label = "official Stock H3 FL2VA"
    if _active_expected_sha:
        _h = _hashlib.sha256()
        with open(_active_integrity_path, "rb") as _f:
            for _chunk in iter(lambda: _f.read(8 * 1024 * 1024), b""):
                _h.update(_chunk)
        _got = _h.hexdigest()
        if _got != _active_expected_sha:
            raise RuntimeError(
                f"{_active_integrity_label} SHA256 mismatch: expected {_active_expected_sha}, got {_got}. "
                "Refusing to run an unknown model revision."
            )
        log(f"  ✓ {_active_integrity_label} SHA256 verified: {_got[:16]}…")
    else:
        log(f"  ✓ {_active_integrity_label} structural validation passed · full-file SHA skipped for faster startup")

    # ── Generic LoRA helpers ───────────────────────────────────────────────
    ldir = os.path.join(MODELS, "loras")
    os.makedirs(ldir, exist_ok=True)

    # CivitAI is an optional user-invoked integration, never a startup dependency.
    # Do not probe/validate its token here; public resources work without one.

    def _civitai_token():
        """Return an optional CivitAI token. Missing/blocked secrets must never stop Studio startup."""
        try:
            tok = (os.environ.get("CIVITAI_API_KEY") or "").strip()
        except Exception:
            tok = ""
        if tok:
            return tok
        try:
            from google.colab import userdata
            try:
                tok = (userdata.get("CIVITAI_API_KEY") or "").strip()
            except Exception:
                tok = ""
        except Exception:
            tok = ""
        if tok:
            os.environ["CIVITAI_API_KEY"] = tok
        return tok

    def _sha256(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest().lower()

    def _civitai_file(version_id, wanted_name, token=""):
        api = f"https://civitai.com/api/v1/model-versions/{version_id}"
        headers = {"User-Agent":"Standalone-MiniMax-H3/1.0", "Accept":"application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(api, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        files = data.get("files") or []
        exact = [f for f in files if (f.get("name") or "").lower() == wanted_name.lower()]
        if not exact:
            raise RuntimeError(f"CivitAI version {version_id} does not expose {wanted_name}.")
        f = exact[0]
        url = f.get("downloadUrl")
        if not url:
            raise RuntimeError(f"CivitAI returned no downloadUrl for {f.get('name')}")
        return f, url

    def _curl_download(url, dest, token="", display_name=None, require_token=False):
        display_name = display_name or os.path.basename(dest)
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
        final_url = url
        if token:
            p = urlsplit(url); q = dict(parse_qsl(p.query, keep_blank_values=True)); q["token"] = token
            final_url = urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))
        elif require_token:
            raise RuntimeError("CivitAI token required for this download path.")
        part = dest + ".part"
        cmd = ["curl","-L","--fail-with-body","--show-error","--progress-bar","--retry","5",
               "--retry-all-errors","--retry-delay","3","--connect-timeout","30","--continue-at","-",
               "--header","User-Agent: Mozilla/5.0","--output",part,final_url]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            try: os.remove(part)
            except OSError: pass
            raise RuntimeError(f"Download failed for {display_name} (curl exit {rc}).")
        if str(dest).lower().endswith(".safetensors"):
            size = os.path.getsize(part) if os.path.exists(part) else 0
            head = open(part,"rb").read(512).lstrip().lower() if size else b""
            if size < 1024*1024 or head.startswith((b"<html",b"<!doctype",b"{")):
                try: os.remove(part)
                except OSError: pass
                raise RuntimeError(f"Download for {display_name} did not return a valid safetensors payload.")
        os.replace(part,dest)

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
            # Lightning is optional; the studio remains usable in QUALITY mode without it.
            log(f"  ⚠ Lightning download failed (optional): {e}")
            return False

    if LOWVRAM_T4_PROFILE:
        LIGHTNING_AVAILABLE = False
        log("  ✓ T4/LOW-VRAM: Lightning unavailable; base Q4_0 profile stays lean")
    elif FAST_STARTUP:
        LIGHTNING_AVAILABLE = True
        log("  ✓ FAST STARTUP: FL2VA Turbo download deferred until FAST is first used")
    else:
        LIGHTNING_AVAILABLE = _ensure_lightning_lora()

    REF2VA_LIGHTNING_PATH = os.path.join(ldir, REF2VA_LIGHTNING_FILE)
    def _ensure_ref2va_lightning_lora():
        try:
            if os.path.exists(REF2VA_LIGHTNING_PATH):
                ok, err = _validate_safetensors_file(REF2VA_LIGHTNING_PATH)
                if ok:
                    log(f"  ✓ Ref2VA Turbo: {REF2VA_LIGHTNING_FILE}")
                    return True
                log(f"  ⚠ Ref2VA Turbo file is corrupt: {err}; replacing it")
                os.remove(REF2VA_LIGHTNING_PATH)
            log(f"  ↓ Ref2VA Turbo LoRA: {REF2VA_LIGHTNING_FILE} ({REF2VA_LIGHTNING_REPO})")
            src = hf_hub_download(
                REF2VA_LIGHTNING_REPO,
                filename=REF2VA_LIGHTNING_FILE,
                subfolder="loras",
            )
            _atomic_copy_weight(src, REF2VA_LIGHTNING_PATH)
            log(f"  ✓ Ref2VA Turbo: {REF2VA_LIGHTNING_FILE}")
            return True
        except Exception as e:
            log(f"  ⚠ Ref2VA Turbo unavailable: {e}")
            return False

    if LOWVRAM_T4_PROFILE:
        REF2VA_LIGHTNING_AVAILABLE = False
        log("  ✓ T4/LOW-VRAM: Ref2VA Turbo unavailable")
    elif FAST_STARTUP:
        REF2VA_LIGHTNING_AVAILABLE = True
        log("  ✓ FAST STARTUP: Ref2VA Turbo download deferred until FAST is first used")
    else:
        REF2VA_LIGHTNING_AVAILABLE = _ensure_ref2va_lightning_lora()

    MOTION8_PATH = os.path.join(ldir, MOTION8_FILE)
    def _ensure_motion8_lora():
        try:
            if os.path.exists(MOTION8_PATH):
                ok, err = _validate_safetensors_file(MOTION8_PATH)
                if ok:
                    log(f"  ✓ Motion 8-Step Enhancer: {MOTION8_FILE}")
                    return True
                log(f"  ⚠ Motion 8-Step file is corrupt: {err}; replacing it")
                os.remove(MOTION8_PATH)
            log(f"  ↓ Motion 8-Step Enhancer: {MOTION8_FILE} ({MOTION8_REPO})")
            src = hf_hub_download(MOTION8_REPO, filename=MOTION8_FILE)
            _atomic_copy_weight(src, MOTION8_PATH)
            log(f"  ✓ Motion 8-Step Enhancer: {MOTION8_FILE}")
            return True
        except Exception as e:
            log(f"  ⚠ Motion 8-Step Enhancer unavailable: {e}")
            return False

    if LOWVRAM_T4_PROFILE:
        MOTION8_AVAILABLE = False
        log("  ✓ T4/LOW-VRAM: Motion8 unavailable")
    elif FAST_STARTUP:
        MOTION8_AVAILABLE = True
        log("  ✓ FAST STARTUP: Motion8 download deferred until first use")
    else:
        MOTION8_AVAILABLE = _ensure_motion8_lora()

    # ── User-installed LoRA metadata helpers ───────────────────────────────
    ACTION_AVAILABLE = False
    SPECIALTY_LORA_STATES = []
    LORA_SOURCE_CACHE_FILE = os.path.join(ldir, ".h3_studio_lora_sources.json")

    def _civitai_json(url, token=""):
        # CivitAI authentication is optional. Public API requests are made anonymously
        # when CIVITAI_API_KEY is absent; only a resource that CivitAI itself gates may
        # reject that specific user-requested install.
        headers = {"User-Agent":"Standalone-MiniMax-H3/1.0", "Accept":"application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if not token and int(getattr(e, "code", 0) or 0) in (401, 403):
                raise RuntimeError(
                    "This specific CivitAI resource requires authentication. "
                    "CIVITAI_API_KEY is optional and is not required to start or use MiniMax H3 Studio; "
                    "only this protected CivitAI download needs it."
                ) from e
            raise

    def _is_h3_version(v):
        v = v or {}
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
        pool.sort(key=lambda f: (str(f.get("type") or "").lower() != "model", -(float(f.get("sizeKB") or 0))))
        return pool[0]

    def _hf_repo_url(repo_id):
        repo_id = str(repo_id or "").strip().strip("/")
        return f"https://huggingface.co/{repo_id}" if repo_id else ""

    def _load_lora_source_cache():
        try:
            if not os.path.exists(LORA_SOURCE_CACHE_FILE): return {}
            raw = json.load(open(LORA_SOURCE_CACHE_FILE,"r",encoding="utf-8"))
            return {os.path.basename(str(k)):str(v).strip() for k,v in raw.items() if k and v}
        except Exception:
            return {}

    def _save_lora_source_cache(cache):
        try:
            keep = {os.path.basename(str(k)):str(v).strip() for k,v in dict(cache or {}).items()
                    if k and v and os.path.exists(os.path.join(ldir,os.path.basename(str(k))))}
            with open(LORA_SOURCE_CACHE_FILE,"w",encoding="utf-8") as fh: json.dump(keep,fh,indent=2)
        except Exception as e:
            log(f"  ⚠ LoRA source cache save skipped: {e}")

    LORA_SOURCE_CACHE = _load_lora_source_cache()

    def _remember_lora_source(file_name, source_url):
        fn=os.path.basename(str(file_name or "")); url=str(source_url or "").strip()
        if fn and url:
            LORA_SOURCE_CACHE[fn]=url; _save_lora_source_cache(LORA_SOURCE_CACHE)

    def _lora_source_map(*_args, **_kwargs):
        out = {}
        if MOTION8_FILE: out[MOTION8_FILE] = _hf_repo_url(MOTION8_REPO)
        if LIGHTNING_FILE: out[LIGHTNING_FILE] = _hf_repo_url(LIGHTNING_REPO)
        if REF2VA_LIGHTNING_FILE: out[REF2VA_LIGHTNING_FILE] = _hf_repo_url(REF2VA_LIGHTNING_REPO)
        for fn,url in dict(LORA_SOURCE_CACHE).items():
            if fn and url: out[fn]=url
        return out

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
    # Full-card Blackwell and A100-80 keep the resident fast path. T4-class cards use Comfy's
    # real DynamicVRAM engine (Aimdo) plus LOW_VRAM text-encoder behavior. Merely
    # setting args.lowvram=True is NOT sufficient in library mode: ComfyUI main.py
    # normally initializes comfy_aimdo and flips memory_management.aimdo_enabled.
    _bw_highvram = bool(BLACKWELL_FULL_CARD and not LOWVRAM_T4_PROFILE)  # includes A100-80 alias
    for _name, _value in {
        "lowvram": bool(LOWVRAM_T4_PROFILE),
        "novram": False,
        "highvram": _bw_highvram,
        "gpu_only": False,
        "normalvram": bool((not LOWVRAM_T4_PROFILE) and (not _bw_highvram)),
        "enable_dynamic_vram": bool(LOWVRAM_T4_PROFILE),
        "disable_dynamic_vram": False,
    }.items():
        if hasattr(args, _name):
            setattr(args, _name, _value)

    # Aimdo was pre-initialized before torch on T4 to match ComfyUI main.py.
    # Keep a defensive fallback for unusual embedded runtimes, but do not reset the
    # controller object here.
    if LOWVRAM_T4_PROFILE and _aimdo_control is None:
        try:
            import comfy_aimdo.control as _aimdo_control
            _simple_headroom = int(float(RESERVE_VRAM) * 1024**3)
            try:
                _aimdo_control.init(
                    simple_vram_headroom=_simple_headroom,
                    nvml_pressure=not bool(getattr(args, "disable_nvml_pressure", False)),
                )
            except TypeError:
                try:
                    _aimdo_control.init(simple_vram_headroom=_simple_headroom)
                except TypeError:
                    _aimdo_control.init()
            log(f"  ✓ T4 DynamicVRAM controller fallback-init complete · headroom {RESERVE_VRAM:.1f} GiB")
        except Exception as _e:
            raise RuntimeError(f"Could not initialize Comfy DynamicVRAM controller: {_e}") from _e

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

    if LOWVRAM_T4_PROFILE:
        import comfy.memory_management as _cmm
        import comfy.model_patcher as _cmp
        if not bool(getattr(_cmm, "aimdo_enabled", False)):
            if _aimdo_control is None:
                raise RuntimeError(
                    "T4 DynamicVRAM requires comfy-aimdo, but it could not be imported. "
                    "Re-run from a fresh runtime so current ComfyUI requirements install cleanly."
                )
            try:
                try:
                    _aimdo_ok = _aimdo_control.init_devices(
                        (d.index, int(float(getattr(args, "vram_headroom", 0.0)) * 1024**3))
                        for d in mm.get_all_torch_devices()
                    )
                except TypeError:
                    _aimdo_ok = _aimdo_control.init_devices(d.index for d in mm.get_all_torch_devices())
            except Exception as _e:
                raise RuntimeError(f"Could not initialize Comfy DynamicVRAM devices: {_e}") from _e
            if not _aimdo_ok:
                raise RuntimeError(
                    "Comfy DynamicVRAM device initialization returned false on this T4. "
                    "A working current comfy-aimdo install is required for the Q4 DynamicVRAM loader."
                )
            _cmp.CoreModelPatcher = _cmp.ModelPatcherDynamic
            _cmm.aimdo_enabled = True

        if not bool(getattr(_cmm, "aimdo_enabled", False)):
            raise RuntimeError("DynamicVRAM did not become active; refusing to start the T4 UI.")
        log("  ✓ Comfy DynamicVRAM ACTIVE · Aimdo enabled · Dynamic GGUF loader may page weights")

    log(f"  vram state: {mm.vram_state} · reserve {RESERVE_VRAM} GB · GPU profile={GPU_PROFILE} · LOWVRAM={LOWVRAM}")
    log(f"  attention active -> {ATTN_BACKEND} (selected before H3 module import)")

    # init_extra_nodes is a coroutine; unawaited it silently registers nothing.
    _r = nodes.init_extra_nodes()
    if asyncio.iscoroutine(_r):
        asyncio.get_event_loop().run_until_complete(_r)
    N = nodes.NODE_CLASS_MAPPINGS
    log(f"✓ {len(N)} nodes registered")
    _required_nodes = ["MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift", "VAEDecodeAudio"]
    if LOWVRAM_T4_PROFILE:
        if "UnetLoaderGGUFDynamicVRAM" not in N and "UnetLoaderGGUF" not in N:
            raise RuntimeError("T4 low-VRAM profile requires molbal/ComfyUI-GGUF loader nodes.")
    else:
        _required_nodes += ["H3SparseAttention", "H3SparseAttentionAdvanced"]
    for req in _required_nodes:
        if req not in N:
            raise RuntimeError(f"node {req} missing — engine/custom node too old")

    SPARSE_NODE_NAME = "" if LOWVRAM_T4_PROFILE else "H3SparseAttention"
    DEFAULT_SPARSE_PERCENT = 0.0
    ZERO_RELOAD_PARTITIONED = False
    ZERO_RELOAD_MIN_FREE_GIB = 4.5

    # V86-BW1 Blackwell full-residency policy.
    #
    # Stock pruned INT8 ConvRot DiT (~19.5 GiB) + Qwen3-VL INT8 ConvRot
    # (~25.2 GiB) + video/audio VAEs (~5.4 GiB) is ~50.2 GiB of weights before
    # activations/workspaces/LoRA patch state. Arm full residency whenever the
    # physical Blackwell has enough room for those weights plus a conservative working
    # headroom budget. The full 96GB G4 profile qualifies; smaller partitions fall back.
    FULL_RESIDENCY_WORKING_HEADROOM_GIB = 18.0
    FULL_RESIDENCY_MIN_FREE_BEFORE_DECODE_GIB = 10.0
    _active_dit_estimate_gib = (
        T4_DIT_GIB if LOWVRAM_T4_PROFILE else
        (T4_DIT_GIB if LOWVRAM_T4_PROFILE else 19.55)
    )
    _full_stack_weights_gib = (
        _active_dit_estimate_gib + TEXT_ENCODER_GIB + VIDEO_VAE_GIB + AUDIO_VAE_GIB
    )
    _full_stack_required_gib = (
        _full_stack_weights_gib + FULL_RESIDENCY_WORKING_HEADROOM_GIB
    )
    try:
        _physical_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        _physical_vram_gib = 0.0

    FULL_STACK_RESIDENCY = bool(
        (not LOWVRAM_T4_PROFILE) and _physical_vram_gib >= _full_stack_required_gib
    )
    # Compatibility alias for older V62–V76 branches below.
    PERSISTENT_FULLCARD_RESIDENCY = FULL_STACK_RESIDENCY
    PERSISTENT_RESIDENCY_MIN_FREE_BEFORE_DECODE_GIB = FULL_RESIDENCY_MIN_FREE_BEFORE_DECODE_GIB

    if LOWVRAM_T4_PROFILE:
        log(
            f"  ✓ T4 DynamicVRAM policy -> physical {_physical_vram_gib:.1f} GiB · "
            f"serialized weights ~{_full_stack_weights_gib:.1f} GiB · never full-stack resident"
        )
        log("  ✓ residency mode -> DYNAMIC GGUF paging + sequential TE→DiT→VAE handoff")
        log("  ✓ sparse backend -> disabled on T4; dense PyTorch SDPA")
    else:
        log(
            f"  ✓ {GPU_PROFILE.upper()} residency policy -> physical {_physical_vram_gib:.1f} GiB · "
            f"estimated weights {_full_stack_weights_gib:.1f} GiB · "
            f"full-resident requirement {_full_stack_required_gib:.1f} GiB"
        )
        log(
            "  ✓ residency mode -> "
            + ("FULL STACK RESIDENT (DiT + Qwen3VL + both VAEs)" if FULL_STACK_RESIDENCY
               else "PARTITIONED-CARD SAFE TE↔DiT HANDOFF")
        )
        if A100_PROFILE:
            log("  ✓ sparse production backend -> H3SparseAttention available on SM80; dense fallback is PyTorch SDPA")
            log("  ✓ A100 path -> quality DiT retained; 40GB uses smaller TE + safe handoff, 80GB can stay resident")
        else:
            log("  ✓ sparse production backend -> H3SparseAttention AUTO · native Kitchen SM120 preferred when checkpoint-compatible")
            log("  ✓ Blackwell fast path -> full-stack residency at 768-class renders; safe fallback retained")

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
    REF2VA_NODE_NAME = (
        "" if LOWVRAM_T4_PROFILE else
        ("MiniMaxH3ReferenceToVideo" if "MiniMaxH3ReferenceToVideo" in N else "")
    )

    def _extract_model_output(x):
        if isinstance(x, (tuple, list)):
            return x[0]
        for attr in ("result", "results", "output", "outputs"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                if isinstance(v, (tuple, list)):
                    return v[0]
                if v is not None:
                    return v
        try:
            vals = list(x)
            if vals:
                return vals[0]
        except Exception:
            pass
        return x

    def _invoke_patch_node(node_name, **kwargs):
        cls = N[node_name]
        inst = cls()
        fn_name = getattr(cls, "FUNCTION", None)
        if fn_name and hasattr(inst, fn_name):
            return _extract_model_output(getattr(inst, fn_name)(**kwargs))
        if hasattr(inst, "execute"):
            return _extract_model_output(inst.execute(**kwargs))
        if hasattr(cls, "execute"):
            return _extract_model_output(cls.execute(**kwargs))
        raise RuntimeError(f"{node_name} exposes neither FUNCTION nor execute().")

    def _apply_sparse_attention(base_model, sparse_percent):
        """Apply flat sparse attention using H3-Optimizations' AUTO backend resolver.

        The old production path hard-selected Advanced/Kitchen INT8. That is a hard
        requirement and crashes when any relevant QKV weight is floating (for
        example after a LoRA/patch composition). The normal node preserves the same
        flat budget when denser_early_late_steps=False, but chooses a checkpoint-
        compatible backend automatically and can fall back safely.
        """
        pct = max(0.0, min(100.0, float(sparse_percent or 0.0)))
        if pct <= 0:
            return base_model, 0.0
        budget = pct / 100.0

        # Current production schema:
        #   model, video_budget, denser_early_late_steps
        # Keep the user's "flat sparse %" semantics by disabling the denser ramp.
        patched = _invoke_patch_node(
            "H3SparseAttention",
            model=base_model,
            video_budget=budget,
            denser_early_late_steps=False,
        )
        return patched, budget

    def _same_patcher(a, b):
        if a is b:
            return True
        for x, y in ((a, b), (b, a)):
            fn = getattr(x, "is_clone", None)
            if callable(fn):
                try:
                    if fn(y):
                        return True
                except Exception:
                    pass
        return False

    def _patcher_from_obj(obj):
        if obj is None:
            return None
        # ModelPatcher itself.
        if hasattr(obj, "model") and hasattr(obj, "load_device") and hasattr(obj, "offload_device"):
            return obj
        # Comfy CLIP / VAE wrappers normally expose their ModelPatcher here.
        for attr in ("patcher", "model_patcher"):
            p = getattr(obj, attr, None)
            if p is not None:
                return p
        return None

    def _pin_persistent_stack(model, clip, vae, avae, reason="resident"):
        """Keep the full stack resident; make the already-resident case nearly free."""
        if not FULL_STACK_RESIDENCY:
            return False
        patchers=[]; seen=set()
        for obj in (model, clip, vae, avae):
            p = _patcher_from_obj(obj)
            if p is None or id(p) in seen:
                continue
            seen.add(id(p)); patchers.append(p)
        if not patchers:
            return False

        loaded = list(getattr(mm, "current_loaded_models", []))
        pending=[]
        for p in patchers:
            lm = next((x for x in loaded if _same_patcher(getattr(x, "model", None), p)), None)
            if lm is None:
                pending.append(p); continue
            try:
                fully_loaded = int(lm.model_loaded_memory()) + (32 << 20) >= int(lm.model_memory())
            except Exception:
                fully_loaded = False
            if not fully_loaded:
                pending.append(p)

        if not pending:
            return True

        try:
            with torch.no_grad():
                try:
                    mm.load_models_gpu(pending, force_full_load=True)
                except TypeError:
                    mm.load_models_gpu(pending)
            torch.cuda.synchronize()
            free_b,total_b = torch.cuda.mem_get_info()
            log(f"  ✓ persistent stack restored ({reason}): {(total_b-free_b)/1024**3:.1f}/{total_b/1024**3:.1f} GiB · {len(pending)} patcher(s)")
            return True
        except Exception as e:
            log(f"  ⚠ persistent-stack pin skipped ({reason}): {e}")
            return False

    def _selective_te_evict_keep_dit(model):
        """Safely transition from the conditioning stack to the H3 DiT.

        V86 retains a partitioned-card fallback for the larger quality text encoder:
        when the DiT was not already resident, the old path attempted to load the
        ~19.5 GiB DiT *before* unloading the ~25.2 GiB Qwen3-VL encoder. On a
        partitioned card that creates an impossible transient residency state and can destabilize
        Comfy's pin/unpin + CUDA allocator path.

        New rule:
          1) synchronize all conditioning work;
          2) if DiT is not resident, unload conditioning models FIRST;
          3) only then load the DiT;
          4) if DiT is already resident, evict everything else while preserving it.
        """
        # Never move/free model storage while asynchronous conditioning kernels may
        # still be referencing it.
        torch.cuda.synchronize()

        loaded = list(getattr(mm, "current_loaded_models", []))
        dit_loaded = None
        for lm in loaded:
            if _same_patcher(getattr(lm, "model", None), model):
                dit_loaded = lm
                break

        t0 = time.perf_counter()

        if dit_loaded is None:
            log("  ↳ partitioned-card handoff: unload conditioning stack BEFORE loading DiT")

            # The Python wrappers remain cached; this only moves/unloads GPU-resident
            # model storage so the 27GB-class text encoder cannot overlap the DiT.
            try:
                mm.unload_all_models()
            except Exception as e:
                log(f"  ⚠ conditioning unload warning: {e}")

            gc.collect()
            try:
                mm.soft_empty_cache()
            except Exception:
                pass

            # Do not call torch.cuda.empty_cache() in the middle of the model-manager
            # handoff. Comfy can reuse allocator blocks directly, and avoiding an
            # extra allocator flush also avoids racing its pin/unpin worker.
            with torch.no_grad():
                if LOWVRAM_T4_PROFILE:
                    # Dynamic VRAM must remain free to page GGUF blocks; forcing a full
                    # 11.4 GB model plus activations defeats the 16 GB profile.
                    mm.load_models_gpu([model])
                else:
                    try:
                        mm.load_models_gpu([model], force_full_load=True)
                    except TypeError:
                        mm.load_models_gpu([model])

            torch.cuda.synchronize()

            loaded = list(getattr(mm, "current_loaded_models", []))
            for lm in loaded:
                if _same_patcher(getattr(lm, "model", None), model):
                    dit_loaded = lm
                    break

            if dit_loaded is None:
                raise RuntimeError("H3 DiT did not become resident after the safe TE→DiT handoff.")
        else:
            # This branch is mainly useful on larger cards / warm jobs.
            mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=[dit_loaded])
            torch.cuda.synchronize()
            gc.collect()
            try:
                mm.soft_empty_cache()
            except Exception:
                pass

        if not any(
            _same_patcher(getattr(lm, "model", None), model)
            for lm in getattr(mm, "current_loaded_models", [])
        ):
            raise RuntimeError("Safe conditioning handoff unexpectedly unloaded the H3 DiT.")

        sec = time.perf_counter() - t0
        free_gib = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"  ✓ safe TE→DiT handoff: {sec:.2f}s · DiT resident · {free_gib:.1f} GiB free")
        return sec

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
    class JobCancelled(RuntimeError):
        pass

    class GenerationRestart(RuntimeError):
        """Abort the current generation attempt and restart from source inputs.

        Used for sampler failures where reusing an already-touched latent/noise/model
        could contaminate the retry. The exception MUST escape _generate so the whole
        Python frame (including latent, positive conditioning and sampler state) dies
        before the next attempt is constructed.
        """
        def __init__(self, params, reason, detail=''):
            super().__init__(reason)
            self.params = dict(params or {})
            self.reason = str(reason or 'restart')
            self.detail = str(detail or '')

    _PIPELINE_STAGE_BASE = {
        "queued": 0.0,
        "loading model": 2.0,
        "applying sparse attention": 7.0,
        "conditioning": 10.0,
        "evicting text encoder": 16.0,
        "sampling": 20.0,
        "OOM fallback / evicting text encoder": 20.0,
        "sparse compatibility fallback / dense": 20.0,
        "sparse runtime-state fallback / dense": 20.0,
        "preparing decode": 77.0,
        "freeing dit for vae": 79.0,
        "decoding video": 82.0,
        "decoding audio": 91.0,
        "muxing": 95.0,
        "saving": 97.0,
        "retiming exact duration": 98.0,
        "stitching timeline": 99.0,
        "done": 100.0,
    }

    def _pipeline_percent(stage, cur=0, total=0):
        stage = str(stage or "queued")
        if stage == "sampling":
            try:
                r = max(0.0, min(1.0, float(cur) / max(1.0, float(total))))
            except Exception:
                r = 0.0
            return 20.0 + 55.0 * r
        return float(_PIPELINE_STAGE_BASE.get(stage, 1.0))

    def _set_job_stage(jid, stage):
        now = time.time()
        old = PROG.get("stage")
        PROG.update(stage=stage, cur=0, total=0)
        j = JOBS.get(jid) if "JOBS" in globals() else None
        if j is not None:
            j["stage"] = stage
            j["stage_started"] = now
            j["last_activity"] = now
            j["cur"] = 0
            j["total"] = 0
        if old != stage:
            log(f"  ↳ job {jid} stage → {stage}")

    def _job_cancel_requested(jid=None):
        if jid is None:
            jid = globals().get("QUEUE_ACTIVE_JOB")
        if not jid or "JOBS" not in globals():
            return False
        return bool(JOBS.get(jid, {}).get("cancel_requested"))

    def _check_job_cancel(jid):
        if _job_cancel_requested(jid):
            raise JobCancelled("Generation cancelled by user.")

    def _h3_progress_hook(cur, total, preview=None, **kw):
        PROG.update(cur=cur, total=total)
        jid = globals().get("QUEUE_ACTIVE_JOB")
        if jid and jid in JOBS:
            JOBS[jid]["cur"] = cur
            JOBS[jid]["total"] = total
            JOBS[jid]["last_activity"] = time.time()
        if jid and _job_cancel_requested(jid):
            raise JobCancelled("Generation cancelled by user.")

    comfy.utils.set_progress_bar_global_hook(_h3_progress_hook)

    # ── 4. Model cache ─────────────────────────────────────────────────────────
    CACHE = {}

    def _invalidate_cached_variant(reason=""):
        """Drop only the job-specific patched MODEL variant.

        H3-Optimizations stores request-local runtime state on descendant
        ModelPatchers. If sparse execution reports a cube-order/session conflict,
        never reuse that descendant on the retry or the next queued job.
        The clean base transformer / CLIP / VAEs remain cached.
        """
        had_variant = CACHE.get("variant_model") is not None
        CACHE.pop("variant_key", None)
        CACHE.pop("variant_model", None)
        CACHE.pop("variant_info", None)
        if had_variant:
            log("  ↳ discarded cached patched MODEL variant"
                + (f" · {reason}" if reason else ""))
        gc.collect()
        try:
            mm.soft_empty_cache()
        except Exception:
            pass

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

    def _ensure_optional_lora_selected(name):
        base = os.path.basename(str(name or ""))
        if base == MOTION8_FILE and not os.path.exists(MOTION8_PATH):
            if not _ensure_motion8_lora(): raise RuntimeError("Motion Enhancer could not be downloaded.")
        elif base == LIGHTNING_FILE and not os.path.exists(LIGHTNING_PATH):
            if not _ensure_lightning_lora(): raise RuntimeError("FL2VA Turbo accelerator could not be downloaded.")
        elif base == REF2VA_LIGHTNING_FILE and not os.path.exists(REF2VA_LIGHTNING_PATH):
            if not _ensure_ref2va_lightning_lora(): raise RuntimeError("Ref2VA Turbo accelerator could not be downloaded.")

    def _apply_lora_checked(model, name, strength, label):
        if not name or name == "none" or float(strength) == 0:
            return model, None
        _ensure_optional_lora_selected(name)
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
                   lightning_strength=LIGHTNING_STRENGTH, unet=None,
                   extra_loras=None):
        """Cache the base H3 weights and stack generation LoRAs on a clone per job.

        Stack order: primary creative LoRA, up to four user rack LoRAs,
        optional Action/CivitAI H3 LoRA, then Turbo/Lightning acceleration.
        """
        unet = unet or DIT_FILE
        if CACHE.get("key") != (unet, weight_dtype):
            if CACHE.get("key") is not None and FULL_STACK_RESIDENCY:
                old_key = CACHE.get("key")
                log(f"  ↳ full-resident profile switch: evicting old active stack {old_key[0]} before loading {unet}")
                torch.cuda.synchronize()
                try:
                    mm.unload_all_models()
                except Exception as e:
                    log(f"  ⚠ profile-switch unload warning: {e}")
                gc.collect()
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
                torch.cuda.synchronize()
            CACHE.clear()
            PROG["stage"] = "loading unet"
            log(f"  loading transformer: {unet}")
            if str(unet).lower().endswith(".gguf"):
                gguf_loader = (
                    "UnetLoaderGGUFDynamicVRAM"
                    if "UnetLoaderGGUFDynamicVRAM" in N
                    else "UnetLoaderGGUF"
                )
                base, = call(gguf_loader, unet_name=unet)
                log(f"  ✓ GGUF loader -> {gguf_loader}")
            else:
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
        norm_extra_loras = []
        for item in (extra_loras or []):
            try:
                name, strength = item
                name = str(name or "none")
                strength = float(strength)
            except Exception:
                continue
            if name == MOTION8_FILE and os.path.basename(str(unet)) != FALLBACK_DIT_FILE:
                log("  ↳ Motion 8-Step Enhancer only targets stock FL2VA; skipping it for this checkpoint")
                continue
            if name != "none" and abs(strength) > 1e-6:
                norm_extra_loras.append((name, strength))

        motion8_active = (
            (str(lora or "none") == MOTION8_FILE and abs(float(lora_strength or 0.0)) > 1e-6)
            or any(name == MOTION8_FILE for name, _ in norm_extra_loras)
        )
        if motion8_active and use_lightning:
            raise RuntimeError(
                "Motion Enhancer and Fast-Mode Accelerator are alternative LoRAs; "
                "refusing to silently disable either one. Turn one OFF in the LoRA rack."
            )

        # NEVER reuse a patched ModelPatcher across jobs. H3-Optimizations and LoRA
        # loaders attach request-local patch/runtime state to descendants. Reusing a
        # descendant can carry stale wrapper/cube-order/patch state into the next clip.
        # Keep the expensive pristine base weights cached, but clone a fresh patcher for
        # every generation attempt. ComfyUI ModelPatcher.clone() is intentionally cheap.
        _base = CACHE["base"]
        _clone = getattr(_base, "clone", None)
        if not callable(_clone):
            raise RuntimeError("Loaded H3 ModelPatcher does not support clone(); refusing cross-job model-state reuse.")
        model = _clone()
        infos = []
        model, info = _apply_lora_checked(model, lora, lora_strength, "primary")
        if info: infos.append(info)

        already_applied = {str(lora or "none")}
        for idx, (extra_name, extra_strength) in enumerate(norm_extra_loras, start=1):
            if extra_name in already_applied:
                log(f"  ↳ extra LoRA slot {idx}: {extra_name} already applied; skipping duplicate")
                continue
            model, info = _apply_lora_checked(model, extra_name, extra_strength, f"extra{idx}")
            if info:
                infos.append(info)
                already_applied.add(extra_name)

        if use_action:
            if not ACTION_AVAILABLE or not ACTION_FILE or not ACTION_PATH or not os.path.exists(ACTION_PATH):
                raise RuntimeError(
                    "The action LoRA is enabled, but no MiniMax-H3-compatible file from "
                    f"CivitAI model {ACTION_MODEL_ID} is available. Check the startup log.")
            if ACTION_FILE in already_applied:
                log("  ↳ action LoRA is already present in the LoRA rack; not applying twice")
            else:
                model, info = _apply_lora_checked(
                    model, ACTION_FILE, action_strength, "action")
                if info:
                    infos.append(info)
                    already_applied.add(ACTION_FILE)

        if use_lightning:
            is_ref2va = os.path.basename(str(unet)).lower().startswith("minimax_h3_ref2va")
            lightning_file = REF2VA_LIGHTNING_FILE if is_ref2va else LIGHTNING_FILE
            lightning_path = REF2VA_LIGHTNING_PATH if is_ref2va else LIGHTNING_PATH
            _ensure_optional_lora_selected(lightning_file)
            if not os.path.exists(lightning_path):
                raise RuntimeError(f"Turbo/Lightning is enabled but {lightning_file} could not be prepared.")
            if lightning_file in already_applied:
                log("  ↳ Turbo/Lightning file is already present in the LoRA rack; not applying twice")
            else:
                model, info = _apply_lora_checked(
                    model, lightning_file, lightning_strength,
                    "Ref2VA Turbo" if is_ref2va else "Lightning")
                if info: infos.append(info)

        variant_info = " | ".join(infos) or "none"
        return model, CACHE["clip"], CACHE["vae"], CACHE["avae"], variant_info

    # Preload the default generation stack before the UI starts. Full-card Blackwell
    # and A100-80 can keep the quality stack resident. A100-40 uses the smaller TE and
    # the partitioned TE↔DiT handoff, while T4 stays on-demand DynamicVRAM.
    # the remaining activation/workspace margin is too small at 1152x768. Keep the
    # Keep the stock DiT resident when the VRAM budget permits; otherwise let Comfy
    # smart-memory swap the text encoder and DiT between conditioning and sampling.
    def _log_vram_budget():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total = 39.5
        dit_gib = T4_DIT_GIB if LOWVRAM_T4_PROFILE else 19.55
        dit_label = "T4 Q4_0" if LOWVRAM_T4_PROFILE else "Stock H3 INT8"
        core = dit_gib + TEXT_ENCODER_GIB
        all_weights = core + VIDEO_VAE_GIB + AUDIO_VAE_GIB
        log(f"  VRAM budget -> {dit_label} ~{dit_gib:.2f} GiB + TE ~{TEXT_ENCODER_GIB:.2f} GiB = ~{core:.2f} GiB")
        log(f"  VRAM budget -> + video/audio VAE ~= {all_weights:.2f} GiB weights vs {total:.1f} GiB physical")
        if FULL_STACK_RESIDENCY:
            log(
                f"  VRAM policy -> FULL-CARD residency: keep DiT + TE + video/audio VAEs warm "
                f"across conditioning, sampling, decode and queued jobs; "
                f"estimated requirement {_full_stack_required_gib:.1f} GiB"
            )
        else:
            log(f"  VRAM policy -> keep Stock H3 + selected LoRAs resident when possible; GPU-swap {TEXT_ENCODER_FILE} for conditioning; reserve {RESERVE_VRAM:.1f} GiB")

    def _preload_default_gpu_stack():
        PROG["stage"] = "startup preload"
        _log_vram_budget()
        log("  ↳ preloading Stock H3 stack to GPU before UI launch")
        try:
            preload_lora = "none"
            preload_lora_strength = 0.0
            preload_lightning = False
            model, clip, vae, avae, info = get_models(
                "default", preload_lora, preload_lora_strength,
                action=False, action_strength=ACTION_STRENGTH,
                lightning=preload_lightning,
                lightning_strength=LIGHTNING_STRENGTH,
                unet=DIT_FILE,
            )
            # V86: if the measured Blackwell budget permits, keep the entire active
            # stack resident. Otherwise preload only the DiT and use the proven
            # partitioned-card TE↔DiT handoff.
            if FULL_STACK_RESIDENCY:
                _pin_persistent_stack(model, clip, vae, avae, reason="startup")
            else:
                if LOWVRAM_T4_PROFILE:
                    mm.load_models_gpu([model])
                else:
                    try:
                        mm.load_models_gpu([model], force_full_load=True)
                    except TypeError:
                        mm.load_models_gpu([model])
                torch.cuda.synchronize()
            free_b, total_b = torch.cuda.mem_get_info()
            used = (total_b-free_b)/1024**3
            total = total_b/1024**3
            log(f"  ✓ startup GPU preload complete: {used:.1f}/{total:.1f} GiB VRAM used · READY before UI launch")
            log(f"  ✓ resident startup stack: {info}")
            PROG["stage"] = "ready"
            return True
        except Exception as e:
            PROG["stage"] = "ready"
            log(f"  ⚠ startup GPU preload warning: {e}")
            log("    UI will still start; Comfy will retry model loading on GENERATE.")
            return False

    if LOWVRAM_T4_PROFILE:
        STARTUP_GPU_PRELOADED = False
        PROG["stage"] = "ready"
        log("✓ T4/LOW-VRAM startup: no force-full GPU preload; Q4_0 DiT loads/pages on first GENERATE")
    elif FAST_STARTUP:
        STARTUP_GPU_PRELOADED = False
        PROG["stage"] = "ready"
        log("✓ FAST STARTUP: GPU model preload deferred until first GENERATE")
    else:
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

    def _load_reference_image(path):
        img = Image.open(path).convert("RGB")
        a = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(a)[None,]

    def _make_audio_dict(waveform, sample_rate):
        if waveform is None:
            return None
        if not torch.is_tensor(waveform):
            waveform = torch.as_tensor(waveform)
        waveform = waveform.detach().float().cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]
        elif waveform.ndim != 2:
            waveform = waveform.reshape(waveform.shape[-2], waveform.shape[-1])
        return {"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": int(sample_rate)}

    def _load_reference_audio(path, required=True):
        last_err = None
        if torchaudio is not None:
            try:
                waveform, sample_rate = torchaudio.load(path)
                return _make_audio_dict(waveform, sample_rate)
            except Exception as e:
                last_err = e
        if av is not None:
            try:
                with av.open(path) as container:
                    stream = next((s for s in container.streams if s.type == "audio"), None)
                    if stream is None:
                        if required:
                            raise ValueError(f"No audio stream found in {os.path.basename(path)}")
                        return None
                    chunks = []
                    sample_rate = int(getattr(stream, "rate", None) or 48000)
                    for frame in container.decode(stream):
                        arr = np.asarray(frame.to_ndarray())
                        if arr.ndim == 1:
                            arr = arr[None, :]
                        elif arr.ndim == 2 and arr.shape[0] > 8 and arr.shape[1] <= 8:
                            arr = arr.T
                        chunks.append(arr.astype(np.float32, copy=False))
                        sample_rate = int(getattr(frame, "sample_rate", None) or sample_rate)
                    if not chunks:
                        if required:
                            raise ValueError(f"No decodable audio found in {os.path.basename(path)}")
                        return None
                    waveform = np.concatenate(chunks, axis=1)
                    return _make_audio_dict(waveform, sample_rate)
            except Exception as e:
                last_err = e
        if required:
            raise RuntimeError(f"Could not decode audio from {os.path.basename(path)}: {last_err}")
        return None

    def _load_reference_video(path, target_fps=MODEL_FPS):
        if av is None:
            raise RuntimeError("PyAV is required for Ref2VA video references but is not available in this runtime.")
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream found in {os.path.basename(path)}")
            src_fps = float(stream.average_rate) if getattr(stream, "average_rate", None) else float(target_fps)
            frames = []
            times = []
            for idx, frame in enumerate(container.decode(stream)):
                frames.append(frame.to_ndarray(format="rgb24"))
                t = frame.time
                if t is None:
                    t = idx / max(src_fps, 1e-6)
                times.append(max(0.0, float(t)))
        if not frames:
            raise ValueError(f"No video frames found in {os.path.basename(path)}")
        if len(frames) == 1:
            selected = frames
        else:
            if not times or len(times) != len(frames):
                times = [i / max(src_fps, 1e-6) for i in range(len(frames))]
            duration = max(times[-1], (len(frames) - 1) / max(src_fps, 1e-6))
            target_count = max(1, int(round(duration * float(target_fps))) + 1)
            if abs(src_fps - float(target_fps)) < 0.25 and target_count == len(frames):
                idxs = list(range(len(frames)))
            else:
                idxs = []
                src_times = np.asarray(times, dtype=np.float32)
                for t in (np.arange(target_count, dtype=np.float32) / float(target_fps)):
                    pos = int(np.searchsorted(src_times, t, side="left"))
                    if pos >= len(frames):
                        pos = len(frames) - 1
                    elif pos > 0 and abs(src_times[pos - 1] - t) <= abs(src_times[pos] - t):
                        pos -= 1
                    idxs.append(pos)
            selected = [frames[i] for i in idxs]
        video = torch.from_numpy(np.stack(selected).astype(np.float32) / 255.0)
        audio = _load_reference_audio(path, required=False)
        return video, audio

    def _frame_like_to_pil(frame):
        if torch.is_tensor(frame):
            arr = frame.detach().float().cpu().numpy()
        else:
            arr = np.asarray(frame)
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3:
            raise ValueError(f"unsupported frame shape: {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            vmax = float(np.nanmax(arr)) if arr.size else 0.0
            if vmax <= 1.001:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _save_stage_frame_pair(jid, images):
        if images is None:
            return None, None
        try:
            first = images[0]
            last = images[-1]
        except Exception as e:
            raise RuntimeError(f"could not index decoded video frames: {e}")
        first_img = _frame_like_to_pil(first)
        last_img = _frame_like_to_pil(last)
        first_path = os.path.join(OUT, f"{jid}_stage_first.png")
        last_path = os.path.join(OUT, f"{jid}_stage_last.png")
        first_img.save(first_path, format="PNG")
        last_img.save(last_path, format="PNG")
        return first_path, last_path

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

    def _probe_media_duration(path):
        ffprobe = shutil.which("ffprobe")
        if not ffprobe or not os.path.exists(path):
            return None
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
            )
            if r.returncode == 0:
                return float((r.stdout or "").strip())
        except Exception:
            pass
        return None

    def _retime_video(src, dest, speed):
        speed = float(speed)
        if abs(speed - 1.0) < 1e-6:
            os.replace(src, dest)
            return True, ""
        if speed <= 0:
            raise ValueError("playback speed must be greater than 0")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to normalize H3 model-time to the requested clip duration")
        filt = f"[0:v]setpts=PTS/{speed:.10g}[v];[0:a]{_atempo_chain(speed)}[a]"
        cmd = [ffmpeg, "-y", "-i", src, "-filter_complex", filt,
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-movflags", "+faststart", dest]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            if os.path.exists(dest):
                os.remove(dest)
            raise RuntimeError(
                "ffmpeg H3 duration/motion retime failed; refusing to silently keep a wrong-speed MP4. "
                + (r.stderr[-700:] if r.stderr else "")
            )
        try: os.remove(src)
        except OSError: pass
        return True, ""

    # ── 5. Generate ────────────────────────────────────────────────────────────
    OUT = "/content/h3_out"; os.makedirs(OUT, exist_ok=True)
    JOBS = {}
    STAGE_LOCK = threading.Lock()
    STAGE_STATE = {
        "first_frame_path": None,
        "last_frame_path": None,
        "video_file": None,
        "job": None,
        "updated": 0.0,
    }
    TIMELINE_LOCK = threading.Lock()
    PROJECTS_DIR = os.path.join(OUT, "timeline_projects")
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    HISTORY_LOCK = threading.Lock()
    HISTORY_FILE = os.path.join(OUT, "generation_history.json")
    HISTORY_STATE = []
    HISTORY_DROP_LOCK = threading.Lock()
    HISTORY_DROP_SEEN = deque(maxlen=512)
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as _hf:
                _rows = json.load(_hf)
            if isinstance(_rows, list):
                HISTORY_STATE = _rows[-300:]
    except Exception as _e:
        log(f"  ⚠ history restore skipped: {_e}")

    def _save_history_unlocked():
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
                json.dump(HISTORY_STATE[-300:], fh, indent=2)
        except Exception as e:
            log(f"  ⚠ history save failed: {e}")

    def _public_history():
        with HISTORY_LOCK:
            rows = list(HISTORY_STATE)
        out_rows = []
        for h in reversed(rows):
            out_rows.append({
                "history_id": h.get("history_id"),
                "job": h.get("job"),
                "file": h.get("file"),
                "duration": h.get("duration"),
                "frames": h.get("frames"),
                "width": h.get("width"),
                "height": h.get("height"),
                "seed": h.get("seed"),
                "prompt": h.get("prompt", ""),
                "last_frame_file": h.get("last_frame_file"),
                "first_frame_file": h.get("first_frame_file"),
                "created": h.get("created"),
                "status": h.get("status", "done"),
            })
        return out_rows

    def _safe_timeline_name(name, fallback):
        txt = re.sub(r"[^A-Za-z0-9._ -]+", "", str(name or "")).strip()
        return (txt[:80] or fallback)

    def _new_sequence(name=None):
        sid = uuid.uuid4().hex[:10]
        now = time.time()
        return {
            "id": sid,
            "name": _safe_timeline_name(name, f"Sequence {sid[:4]}"),
            "segments": [],
            "master_file": None,
            "created": now,
            "updated": now,
        }

    def _blank_timeline_state(project_name="Current Project"):
        first = _new_sequence("Sequence 1")
        now = time.time()
        return {
            "project_id": uuid.uuid4().hex[:12],
            "project_name": _safe_timeline_name(project_name, "Current Project"),
            "project_file": None,
            "sequences": [first],
            "active_sequence_id": first["id"],
            "updated": now,
        }

    TIMELINE_STATE = _blank_timeline_state()

    def _sequence_duration(seq):
        return round(sum(float(s.get("duration") or 0) for s in (seq.get("segments") or [])), 3)

    def _public_segment(seg, idx, seq_id=None):
        return {
            "index": idx,
            "sequence_id": seq_id,
            "segment_id": seg.get("segment_id") or seg.get("job"),
            "job": seg.get("job"),
            "file": seg.get("file"),
            "duration": seg.get("duration"),
            "frames": seg.get("frames"),
            "width": seg.get("width"),
            "height": seg.get("height"),
            "seed": seg.get("seed"),
            "prompt": seg.get("prompt", ""),
            "last_frame_file": seg.get("last_frame_file"),
            "first_frame_file": seg.get("first_frame_file"),
            "continued": bool(seg.get("continued")),
            "continued_from_job": seg.get("continued_from_job"),
        }

    def _sync_stage_from_sequence(seq):
        segs = list((seq or {}).get("segments") or [])
        last = segs[-1] if segs else None
        with STAGE_LOCK:
            if last and last.get("last_frame_path") and os.path.exists(last.get("last_frame_path")):
                STAGE_STATE["first_frame_path"] = last.get("first_frame_path")
                STAGE_STATE["last_frame_path"] = last.get("last_frame_path")
                STAGE_STATE["video_file"] = last.get("file")
                STAGE_STATE["job"] = last.get("job")
                STAGE_STATE["updated"] = time.time()
            else:
                STAGE_STATE["first_frame_path"] = None
                STAGE_STATE["last_frame_path"] = None
                STAGE_STATE["video_file"] = None
                STAGE_STATE["job"] = None
                STAGE_STATE["updated"] = time.time()

    def _active_sequence_unlocked(create=False):
        seqs = TIMELINE_STATE.get("sequences") or []
        active_id = TIMELINE_STATE.get("active_sequence_id")
        for seq in seqs:
            if seq.get("id") == active_id:
                return seq
        if seqs:
            TIMELINE_STATE["active_sequence_id"] = seqs[0]["id"]
            return seqs[0]
        if create:
            seq = _new_sequence("Sequence 1")
            TIMELINE_STATE["sequences"] = [seq]
            TIMELINE_STATE["active_sequence_id"] = seq["id"]
            TIMELINE_STATE["updated"] = time.time()
            return seq
        return None

    def _find_sequence_unlocked(seq_id):
        for seq in (TIMELINE_STATE.get("sequences") or []):
            if seq.get("id") == seq_id:
                return seq
        return None

    def _timeline_snapshot_unlocked():
        state = {
            "project_id": TIMELINE_STATE.get("project_id") or uuid.uuid4().hex[:12],
            "project_name": TIMELINE_STATE.get("project_name") or "Current Project",
            "project_file": TIMELINE_STATE.get("project_file"),
            "active_sequence_id": TIMELINE_STATE.get("active_sequence_id"),
            "updated": TIMELINE_STATE.get("updated", time.time()),
            "sequences": [],
        }
        for seq in (TIMELINE_STATE.get("sequences") or []):
            seq_copy = {
                "id": seq.get("id") or uuid.uuid4().hex[:10],
                "name": seq.get("name") or "Sequence",
                "master_file": seq.get("master_file"),
                "created": seq.get("created", time.time()),
                "updated": seq.get("updated", time.time()),
                "segments": [],
            }
            for seg in (seq.get("segments") or []):
                seg_copy = dict(seg)
                seq_copy["segments"].append(seg_copy)
            state["sequences"].append(seq_copy)
        if not state["sequences"]:
            seq = _new_sequence("Sequence 1")
            state["sequences"] = [seq]
            state["active_sequence_id"] = seq["id"]
        return state

    def _project_json_path_unlocked():
        pid = TIMELINE_STATE.get("project_id") or uuid.uuid4().hex[:12]
        pname = _safe_timeline_name(TIMELINE_STATE.get("project_name"), "Current Project").replace(" ", "_")
        return os.path.join(PROJECTS_DIR, f"{pid}_{pname}.json")

    def _autosave_timeline_unlocked(reason="autosave"):
        path = _project_json_path_unlocked()
        old_name = TIMELINE_STATE.get("project_file")
        snapshot = _timeline_snapshot_unlocked()
        snapshot["save_reason"] = reason
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)
        new_name = os.path.basename(path)
        TIMELINE_STATE["project_file"] = new_name

        # If the same project was renamed, remove the old filename so it does not
        # appear as a duplicate saved project.
        if old_name and old_name != new_name:
            old_path = os.path.join(PROJECTS_DIR, os.path.basename(old_name))
            if os.path.abspath(old_path) != os.path.abspath(path):
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except Exception:
                    pass
        return path

    def _list_saved_projects():
        # Only show actual projects. Older V45-V53 builds wrote an empty
        # "Current Project" JSON on every startup, which created phantom duplicates.
        # Empty projects are shown only when the user explicitly pressed SAVE.
        candidates = []
        for name in sorted(os.listdir(PROJECTS_DIR), reverse=True):
            if not name.endswith(".json"):
                continue
            path = os.path.join(PROJECTS_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                seqs = list(payload.get("sequences") or [])
                clip_count = sum(len(s.get("segments") or []) for s in seqs)
                total_dur = round(sum(float(seg.get("duration") or 0) for s in seqs for seg in (s.get("segments") or [])), 3)
                save_reason = str(payload.get("save_reason") or "")
                if clip_count == 0 and save_reason != "manual_save":
                    continue
                candidates.append({
                    "project_id": payload.get("project_id") or name,
                    "project_name": payload.get("project_name") or name,
                    "project_file": name,
                    "sequence_count": len(seqs),
                    "clip_count": clip_count,
                    "total_duration": total_dur,
                    "updated": payload.get("updated") or os.path.getmtime(path),
                    "save_reason": save_reason,
                })
            except Exception:
                continue

        # One row per logical project id; keep the newest file if an older build
        # left multiple filenames for the same project.
        candidates.sort(key=lambda r: r.get("updated") or 0, reverse=True)
        seen = set()
        rows = []
        for row in candidates:
            pid = row.get("project_id")
            if pid in seen:
                continue
            seen.add(pid)
            rows.append(row)
        return rows

    def _restore_timeline_from_payload_unlocked(payload):
        restored = _blank_timeline_state(payload.get("project_name") or "Loaded Project")
        restored["project_id"] = payload.get("project_id") or restored["project_id"]
        restored["project_name"] = _safe_timeline_name(payload.get("project_name"), "Loaded Project")
        restored["project_file"] = payload.get("project_file")
        restored["updated"] = time.time()
        restored["sequences"] = []
        for raw_seq in (payload.get("sequences") or []):
            seq = _new_sequence(raw_seq.get("name") or f"Sequence {len(restored['sequences']) + 1}")
            seq["id"] = raw_seq.get("id") or seq["id"]
            seq["master_file"] = raw_seq.get("master_file")
            seq["created"] = raw_seq.get("created", time.time())
            seq["updated"] = raw_seq.get("updated", time.time())
            seq["segments"] = []
            for raw_seg in (raw_seq.get("segments") or []):
                seg = dict(raw_seg)
                seg.setdefault("segment_id", uuid.uuid4().hex[:12])
                if not seg.get("first_frame_path") and seg.get("first_frame_file"):
                    seg["first_frame_path"] = os.path.join(OUT, seg["first_frame_file"])
                if not seg.get("last_frame_path") and seg.get("last_frame_file"):
                    seg["last_frame_path"] = os.path.join(OUT, seg["last_frame_file"])
                seq["segments"].append(seg)
            restored["sequences"].append(seq)
        if not restored["sequences"]:
            restored["sequences"] = [_new_sequence("Sequence 1")]
        wanted = payload.get("active_sequence_id")
        active = None
        for seq in restored["sequences"]:
            if seq.get("id") == wanted:
                active = seq
                break
        if active is None:
            active = restored["sequences"][0]
        restored["active_sequence_id"] = active["id"]
        TIMELINE_STATE.clear()
        TIMELINE_STATE.update(restored)
        _autosave_timeline_unlocked("load")
        return active

    def _timeline_public_state():
        with TIMELINE_LOCK:
            active = _active_sequence_unlocked(create=True)
            seqs = list(TIMELINE_STATE.get("sequences") or [])
            seq_public = []
            total_clips = 0
            total_duration = 0.0
            for seq in seqs:
                segs = list(seq.get("segments") or [])
                dur = _sequence_duration(seq)
                total_clips += len(segs)
                total_duration += dur
                seq_public.append({
                    "id": seq.get("id"),
                    "name": seq.get("name") or "Sequence",
                    "clip_count": len(segs),
                    "total_duration": dur,
                    "master_file": seq.get("master_file"),
                    "created": seq.get("created"),
                    "updated": seq.get("updated"),
                    "active": bool(active and seq.get("id") == active.get("id")),
                })
            active_segs = list((active or {}).get("segments") or [])
            active_public = [_public_segment(seg, i, (active or {}).get("id")) for i, seg in enumerate(active_segs)]
            active_duration = _sequence_duration(active or {"segments": []})
            return {
                "project_id": TIMELINE_STATE.get("project_id"),
                "project_name": TIMELINE_STATE.get("project_name") or "Current Project",
                "project_file": TIMELINE_STATE.get("project_file"),
                "updated": TIMELINE_STATE.get("updated", 0.0),
                "active_sequence_id": (active or {}).get("id"),
                "active_sequence": {
                    "id": (active or {}).get("id"),
                    "name": (active or {}).get("name") or "Sequence",
                    "clip_count": len(active_segs),
                    "total_duration": active_duration,
                    "master_file": (active or {}).get("master_file"),
                },
                "sequences": seq_public,
                "segments": active_public,
                "master_file": (active or {}).get("master_file"),
                "total_duration": active_duration,
                "project_total_duration": round(total_duration, 3),
                "project_clip_count": total_clips,
                "saved_projects": _list_saved_projects(),
            }

    # Do NOT create a saved project merely by opening the UI.
    # The working project stays in memory until it has a clip or the user presses SAVE.

    def _concat_mp4_timeline(files, dest):
        """Concatenate final A/V MP4 segments. Copy first; re-encode only if needed."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "ffmpeg not found; timeline master was not stitched"
        files = [os.path.abspath(f) for f in files if f and os.path.exists(f)]
        if not files:
            return False, "timeline has no existing segment files"
        if len(files) == 1:
            return True, "single-segment timeline"

        list_path = dest + ".concat.txt"
        with open(list_path, "w", encoding="utf-8") as fh:
            for f in files:
                fh.write("file '" + f.replace("'", "'\''") + "'\n")

        tmp = dest + ".tmp.mp4"
        for pth in (dest, tmp):
            try:
                os.remove(pth)
            except FileNotFoundError:
                pass

        # All generated segments normally share the same H.264/AAC stream shape,
        # so packet-level concat is nearly instant and does not degrade quality.
        copy_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                    "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                    "-fflags", "+genpts", "-movflags", "+faststart", tmp]
        r = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1024:
            os.replace(tmp, dest)
            try: os.remove(list_path)
            except Exception: pass
            return True, "stream-copy concat"

        # Fallback keeps audio and video together and normalizes timestamps. This
        # should rarely run because continuation forces the same canvas size.
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        enc_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv660p", "-c:a", "aac", "-b:a", "192k",
                   "-fflags", "+genpts", "-movflags", "+faststart", tmp]
        r2 = subprocess.run(enc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try: os.remove(list_path)
        except Exception: pass
        if r2.returncode != 0 or not os.path.exists(tmp):
            return False, "timeline stitch failed: " + (r2.stderr[-700:] if r2.stderr else r.stderr[-700:])
        os.replace(tmp, dest)
        return True, "re-encoded concat fallback"

    def _restitch_sequence(seq_id, reason="edit"):
        with TIMELINE_LOCK:
            seq = _find_sequence_unlocked(seq_id)
            if not seq:
                return False, None, "sequence not found"
            segs = list(seq.get("segments") or [])
            paths = [os.path.join(OUT, s.get("file", "")) for s in segs]
        if not segs:
            master_file = None
            ok, note = True, "empty sequence"
        elif len(segs) == 1:
            master_file = segs[0].get("file")
            ok, note = True, "single-segment sequence"
        else:
            master_file = f"timeline_{uuid.uuid4().hex[:10]}.mp4"
            ok, note = _concat_mp4_timeline(paths, os.path.join(OUT, master_file))
            if not ok:
                master_file = segs[-1].get("file")
        with TIMELINE_LOCK:
            seq = _find_sequence_unlocked(seq_id)
            if seq:
                seq["master_file"] = master_file
                seq["updated"] = time.time()
                TIMELINE_STATE["updated"] = time.time()
                _autosave_timeline_unlocked(reason)
                active = (TIMELINE_STATE.get("active_sequence_id") == seq_id)
            else:
                active = False
        if active:
            _sync_stage_from_sequence(seq)
        return ok, master_file, note

    # Real queued execution: UI can submit many jobs while one GPU job is running.
    QUEUE_LOCK = threading.Lock()
    QUEUE_CV = threading.Condition(QUEUE_LOCK)
    JOB_QUEUE = deque()
    QUEUE_ACTIVE_JOB = None

    def _queue_public_state():
        with QUEUE_LOCK:
            pending = list(JOB_QUEUE)
            active = QUEUE_ACTIVE_JOB
        rows = []
        if active and active in JOBS:
            now = time.time()
            j = JOBS.get(active, {})
            stage = j.get("stage") or PROG.get("stage", "running")
            cur = j.get("cur", PROG.get("cur", 0))
            total = j.get("total", PROG.get("total", 0))
            stage_started = float(j.get("stage_started") or j.get("t0") or now)
            last_activity = float(j.get("last_activity") or stage_started)
            with _GPU_LOCK:
                gpu = dict(_GPU_TELEMETRY)
            gpu_util = gpu.get("util")
            gpu_power = gpu.get("power_w")
            gpu_busy = bool(
                gpu.get("ok") and (
                    (gpu_util is not None and float(gpu_util) >= 20.0)
                    or (gpu_power is not None and float(gpu_power) >= 120.0)
                )
            )
            activity_age = max(0.0, now - last_activity)
            # A stage can legitimately have no Comfy progress callbacks (VAE decode,
            # mux/save). Only flag a possible stall when the GPU is also idle.
            possible_stall = bool(
                not gpu_busy
                and activity_age >= 45.0
                and stage not in ("queued", "saving", "retiming exact duration", "stitching timeline", "done")
            )
            expected_steps = max(0, int(j.get("expected_sample_steps") or 0))
            sample_total = int(total or expected_steps or 0) if stage == "sampling" else 0
            if stage == "sampling" and sample_total > 0:
                # Comfy's progress callback advances after a completed step. While the
                # next kernel is executing, present that in-flight step to the user.
                sample_step = min(sample_total, max(1, int(cur or 0) + (1 if int(cur or 0) < sample_total else 0)))
            else:
                sample_step = 0
            sampling_label = str(j.get("sampling_label") or "").strip()
            stage_label = (
                f"{sampling_label} · sampling" if stage == "sampling" and sampling_label
                else stage
            )
            rows.append({
                "id": active, "status": j.get("status", "running"),
                "prompt": j.get("prompt", ""), "duration": j.get("requested_duration"),
                "created": j.get("t0", 0), "stage": stage, "stage_label": stage_label,
                "cur": cur, "total": total,
                "pipeline_pct": round(_pipeline_percent(stage, cur, sample_total or total), 2),
                "elapsed": max(0, int(now - float(j.get("t0") or now))),
                "stage_elapsed": max(0, int(now - stage_started)),
                "sample_step": sample_step,
                "sample_steps": sample_total,
                "sample_eta": (
                    max(0, int((now - stage_started) / max(1.0, float(cur)) * max(0.0, float(sample_total) - float(cur))))
                    if stage == "sampling" and float(cur or 0) > 0 and float(sample_total or 0) > float(cur or 0)
                    else None
                ),
                "activity_age": round(activity_age, 1),
                "gpu_busy": gpu_busy,
                "gpu_util": gpu_util,
                "gpu_power_w": gpu_power,
                "possible_stall": possible_stall,
                "cancel_requested": bool(j.get("cancel_requested")),
                "thumb_file": j.get("thumb_file"),
            })
        for jid in pending:
            j = JOBS.get(jid, {})
            if j.get("status") != "queued":
                continue
            rows.append({
                "id": jid, "status": "queued", "prompt": j.get("prompt", ""),
                "duration": j.get("requested_duration"), "created": j.get("t0", 0),
                "stage": "queued", "cur": 0, "total": 0, "pipeline_pct": 0.0,
                "thumb_file": j.get("thumb_file"),
            })
        queued_n = sum(1 for r in rows if r["status"] == "queued")
        worker_alive = bool(globals().get("QUEUE_WORKER_THREAD") and QUEUE_WORKER_THREAD.is_alive())
        return {
            "active": active,
            "items": rows,
            "queued": queued_n,
            "worker_alive": worker_alive,
            "worker_waiting_anomaly": bool(queued_n and not active and not worker_alive),
        }

    def _resolve_queued_inputs(p):
        action = (p.get("timeline_action") or "new").lower()
        target_id = p.get("_target_sequence_id")
        if action == "continue":
            with TIMELINE_LOCK:
                seq = _find_sequence_unlocked(target_id) if target_id else _active_sequence_unlocked(create=True)
                segs = list((seq or {}).get("segments") or [])
            if not seq or not segs:
                raise RuntimeError("The target sequence is empty. A queued CONTINUE needs a completed clip before it.")
            prev = segs[-1]
            stage_path = prev.get("last_frame_path")
            if not stage_path or not os.path.exists(stage_path):
                raise RuntimeError("The previous clip has no continuation frame.")
            p["first_frame"] = stage_path
            p["stage_source_job"] = prev.get("job")
            p["use_stage_last"] = "1"
            p["width"] = str(prev.get("width") or p.get("width"))
            p["height"] = str(prev.get("height") or p.get("height"))
        elif str(p.get("use_stage_last") or "0") == "1" and not p.get("first_frame"):
            with STAGE_LOCK:
                stage_path = STAGE_STATE.get("last_frame_path")
                stage_job = STAGE_STATE.get("job")
            if not stage_path or not os.path.exists(stage_path):
                raise RuntimeError("No stage continuation frame is available.")
            p["first_frame"] = stage_path
            p["stage_source_job"] = stage_job
        return p

    def _enqueue_generation(jid, p):
        with QUEUE_CV:
            JOB_QUEUE.append(jid)
            JOBS[jid]["_params"] = p
            QUEUE_CV.notify()

    def _queue_worker():
        global QUEUE_ACTIVE_JOB
        while True:
            with QUEUE_CV:
                while not JOB_QUEUE:
                    QUEUE_CV.wait()
                jid = JOB_QUEUE.popleft()
                QUEUE_ACTIVE_JOB = jid
            j = JOBS.get(jid)
            if not j or j.get("status") == "cancelled":
                with QUEUE_CV:
                    QUEUE_ACTIVE_JOB = None
                continue
            try:
                p = dict(j.pop("_params", {}) or {})
                _resolve_queued_inputs(p)
                generate(jid, p)
            except Exception:
                traceback.print_exc()
                if jid in JOBS:
                    JOBS[jid].update(status="error", msg=traceback.format_exc()[-1600:])
            finally:
                with QUEUE_CV:
                    QUEUE_ACTIVE_JOB = None
                    QUEUE_CV.notify_all()

    # Only one generate may touch the GPU at a time. Without this, every failed or
    # still-running job keeps its models resident and they compete for the card —
    # which looks exactly like a memory leak but is really N pipelines at once.
    GPU_LOCK = threading.Lock()
    CUDA_CONTEXT_POISONED = False

    def generate(jid, p):
        global CUDA_CONTEXT_POISONED
        j = JOBS[jid]
        if CUDA_CONTEXT_POISONED:
            j.update(
                status="error",
                msg="The isolated H3 CUDA context previously reported an illegal memory access. "
                    "Run the V86 cell again to launch a fresh child process before generating."
            )
            return
        if not GPU_LOCK.acquire(blocking=False):
            j.update(status="error",
                     msg="Another generation is still running on the GPU.\n"
                         "Wait for it to finish, or restart the runtime if it is stuck.")
            return
        try:
            # IMPORTANT: use no_grad(), NOT inference_mode(). ComfyUI smart-memory
            # may partially unload/reload ModelPatcher weights while swapping the TE.
            attempt_params = dict(p)
            restarts = 0
            while True:
                try:
                    with torch.no_grad():
                        _generate(jid, attempt_params)
                    break
                except GenerationRestart as restart:
                    restarts += 1
                    if restarts > 2:
                        raise RuntimeError(
                            f"Generation restart limit exceeded after {restart.reason}: {restart.detail}"
                        ) from restart
                    log(
                        f"  ↻ FULL CLEAN GENERATION RESTART · {restart.reason} · "
                        "discarding prior latent/noise/conditioning before rebuilding"
                    )
                    attempt_params = dict(restart.params)
                    j["restart_reason"] = restart.reason
                    j["restart_count"] = restarts
                    gc.collect()
                    try:
                        mm.soft_empty_cache()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        finally:
            GPU_LOCK.release()

    def _generate(jid, p):
        global CUDA_CONTEXT_POISONED
        j = JOBS[jid]
        try:
            j.update(status="running")
            _job_wall0 = time.perf_counter()

            # Keep the startup-preloaded DiT resident. Comfy smart memory will swap it
            # out only when the 32B text encoder needs the card for conditioning, then
            # bring the DiT back for sampling. Unloading everything here would defeat
            # the whole point of startup preloading.
            try:
                gc.collect()
                if not FULL_STACK_RESIDENCY:
                    mm.soft_empty_cache()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            f0 = torch.cuda.mem_get_info()[0]/1e9
            log(f"  job {jid} start: {f0:.1f} GB free")
            j["memlog"] = f"VRAM at job start: {f0:.1f} GB free\n"

            _set_job_stage(jid, "loading model")
            _check_job_cancel(jid)
            model, clip, vae, avae, lora_info = get_models(
                p["weight_dtype"], p.get("lora"), p.get("lora_strength", 0.0),
                action=p.get("action", "0"),
                action_strength=p.get("action_strength", ACTION_STRENGTH),
                lightning=p.get("lightning", "0"),
                lightning_strength=p.get("lightning_strength", LIGHTNING_STRENGTH),
                unet=p.get("unet") or DIT_FILE,
                extra_loras=p.get("extra_loras"))
            j["lora_info"] = lora_info
            _check_job_cancel(jid)

            raw_shift_video = float(p.get("shift_video") or 1.0)
            raw_shift_audio = float(p.get("shift_audio") or 1.0)
            shift_video = raw_shift_video if raw_shift_video > 0.0 else 1.0
            shift_audio = raw_shift_audio if raw_shift_audio > 0.0 else 1.0
            if shift_video != raw_shift_video or shift_audio != raw_shift_audio:
                log(
                    f"  ⚠ invalid zero/negative H3 shift normalized: "
                    f"{raw_shift_video:g}/{raw_shift_audio:g} → {shift_video:g}/{shift_audio:g}; "
                    "1.0 is the unshifted identity schedule"
                )
            p["shift_video"] = str(shift_video)
            p["shift_audio"] = str(shift_audio)
            j["shift_video"] = shift_video
            j["shift_audio"] = shift_audio

            model, = call("MiniMaxH3SigmaShift", model=model,
                          shift_video=shift_video,
                          shift_audio=shift_audio)

            sparse_pct = max(0.0, min(100.0, float(p.get("sparse_percent") or 0.0)))
            if LOWVRAM_T4_PROFILE:
                sparse_pct = 0.0
                p["sparse_percent"] = "0"
            if sparse_pct > 0:
                _set_job_stage(jid, "applying sparse attention")
                model, sparse_budget = _apply_sparse_attention(model, sparse_pct)
                log(f"  sparse attention -> {sparse_pct:.1f}% flat video budget · AUTO backend (checkpoint-safe)")
            else:
                sparse_budget = 0.0
                log("  sparse attention -> OFF / dense")
            j["sparse_percent"] = round(sparse_pct, 3)
            j["sparse_backend_policy"] = "auto" if sparse_pct > 0 else "dense"
            j["sparse_percent_effective"] = round(sparse_pct, 3)
            j["sparse_backend_policy_effective"] = j["sparse_backend_policy"]

            _set_job_stage(jid, "conditioning")
            n_frames, actual_sec, requested_sec = resolve_length(p)
            j["resolved_frames"] = int(n_frames)
            j["requested_final_sec"] = round(requested_sec, 3)
            log(f"  length -> {n_frames} legal frames ({actual_sec:.2f}s model time; requested final {requested_sec:.2f}s)")
            log(f"  render config -> {p.get('width')}x{p.get('height')} | steps={p.get('steps')} | "
                f"Lightning={p.get('lightning','0')} | extras={len(p.get('extra_loras') or [])} | attention={ATTN_BACKEND} | sparse={sparse_pct:.1f}%/AUTO | LOWVRAM={LOWVRAM} | motion_pace={p.get('playback_speed','1.0')}x | prompt={p.get('prompt_source','raw_local')}")
            if n_frames > 362:
                log("  ⚠ long H3 clip: more than 362 model frames; sampling and VAE decode can be substantially slower/more memory-heavy")
            width, height = int(p["width"]), int(p["height"])
            if width % 32 or height % 32:
                raise ValueError("width and height must be multiples of 32")
            if width < 32 or height < 32:
                raise ValueError("width and height must be at least 32")
            mode = (p.get("input_mode") or "fl2va").lower()
            t_cond0 = time.perf_counter()
            if mode == "ref2va":
                if not REF2VA_NODE_NAME:
                    raise RuntimeError("This ComfyUI build does not expose MiniMaxH3ReferenceToVideo, so the Ref2VA tab cannot run.")
                kw = dict(clip=clip, vae=vae, audio_vae=avae, prompt=p["prompt"],
                          width=width, height=height, length=n_frames,
                          ref_image_size=(p.get("ref_image_size") or "match"))
                ref_images = {}
                for idx, path in enumerate(p.get("ref_images") or [], start=1):
                    ref_images[f"ref_image_{idx}"] = _load_reference_image(path)
                ref_videos = {}
                ref_video_audios = {}
                for idx, path in enumerate(p.get("ref_videos") or [], start=1):
                    frames, soundtrack = _load_reference_video(path)
                    ref_videos[f"ref_video_{idx}"] = frames
                    if soundtrack is not None:
                        ref_video_audios[f"ref_video_audio_{idx}"] = soundtrack
                ref_audios = {}
                for idx, path in enumerate(p.get("ref_audios") or [], start=1):
                    ref_audios[f"ref_audio_{idx}"] = _load_reference_audio(path, required=True)
                if not ref_images and not ref_videos:
                    raise ValueError("Ref2VA requires at least one reference image or one reference video.")
                if ref_images:
                    kw["ref_images"] = ref_images
                if ref_videos:
                    kw["ref_videos"] = ref_videos
                if ref_video_audios:
                    kw["ref_video_audios"] = ref_video_audios
                if ref_audios:
                    kw["ref_audios"] = ref_audios
                positive, latent = call(REF2VA_NODE_NAME, **kw)
                j["conditioning_mode"] = "ref2va"
                j["ref_counts"] = {
                    "images": len(ref_images),
                    "videos": len(ref_videos),
                    "video_audios": len(ref_video_audios),
                    "audios": len(ref_audios),
                }
                log(f"  ref2va -> {len(ref_images)} image ref(s), {len(ref_videos)} video ref(s), {len(ref_audios)} audio ref(s)")
            else:
                kw = dict(clip=clip, vae=vae, prompt=p["prompt"],
                          width=width, height=height, length=n_frames)
                fit_mode = p.get("image_fit") or "cover"
                if p.get("first_frame"):
                    kw["first_frame"] = _prepare_frame(p["first_frame"], width, height, fit_mode)
                if p.get("last_frame"):
                    kw["last_frame"] = _prepare_frame(p["last_frame"], width, height, fit_mode)
                positive, latent = call("MiniMaxH3ImageToVideo", **kw)
                j["conditioning_mode"] = "fl2va"
            # Conditioning kernels are asynchronous. Synchronize here before any
            # model-manager unload/move so no kernel can still reference TE storage.
            torch.cuda.synchronize()
            j["conditioning_sec"] = round(time.perf_counter() - t_cond0, 3)
            _check_job_cancel(jid)

            # V86 quality encoder policy:
            # Qwen3-VL INT8 ConvRot (~25.2 GiB binary) plus the ~19.5 GiB H3 DiT
            # fit together on the full Blackwell card; use the TE-only transition only on partitions.
            # Full-card Blackwell retains the persistent-residency path.
            free_gib_before_sample = torch.cuda.mem_get_info()[0] / 1024**3
            j["pre_sample_free_gib"] = round(free_gib_before_sample, 3)
            j["te_evict_sec"] = 0.0
            j["residency_mode"] = "full_resident" if FULL_STACK_RESIDENCY else "zero_reload"

            _force_te_evict = str(p.get("_force_te_evict") or "0").lower() in ("1","true","yes","on")
            if FULL_STACK_RESIDENCY and not _force_te_evict:
                # The legacy small-card branch is bypassed on full-card Blackwell so residency
                # remains persistent across conditioning and sampling.
                # explicitly keeps TE + DiT + VAEs resident through the sample.
                _pin_persistent_stack(model, clip, vae, avae, reason="post-conditioning / pre-sample")
                free_after_pin = torch.cuda.mem_get_info()[0] / 1024**3
                j["pre_sample_free_gib"] = round(free_after_pin, 3)
                log(
                    f"  ✓ FULL-RESIDENT sampler handoff: DiT + TE + VAEs stay on GPU · "
                    f"{free_after_pin:.2f} GiB free"
                )
            elif _force_te_evict or (not ZERO_RELOAD_PARTITIONED) or free_gib_before_sample < ZERO_RELOAD_MIN_FREE_GIB:
                _set_job_stage(jid, "evicting text encoder")
                clip = None
                j["te_evict_sec"] = round(_selective_te_evict_keep_dit(model), 3)
                j["residency_mode"] = "te_only_evict_forced" if _force_te_evict else "te_only_evict"
                log(
                    f"  ↳ {'forced clean-retry' if _force_te_evict else 'partitioned-card'} handoff: "
                    f"{free_gib_before_sample:.2f} GiB free; using safe TE-only eviction"
                )
            else:
                log(
                    f"  ✓ ZERO-RELOAD armed: TE + DiT stay resident · "
                    f"{free_gib_before_sample:.2f} GiB free before sampler"
                )

            guider,  = call("BasicGuider", model=model, conditioning=positive)
            sampler, = call("KSamplerSelect", sampler_name=p["sampler_name"])
            sigmas,  = call("BasicScheduler", model=model, scheduler=p["scheduler"],
                            steps=int(p["steps"]), denoise=float(p["denoise"]))
            noise,   = call("RandomNoise", noise_seed=int(p["seed"]))

            j["expected_sample_steps"] = max(1, int(p["steps"]))
            j["sampling_label"] = "SPARSE" if sparse_pct > 0 else "DENSE"
            _set_job_stage(jid, "sampling")
            _check_job_cancel(jid)
            torch.cuda.synchronize()
            t_sample0 = time.perf_counter()
            try:
                samples = call("SamplerCustomAdvanced", noise=noise, guider=guider,
                               sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
            except torch.cuda.OutOfMemoryError as _oom_exc:
                # NEVER resume sampling with the same latent after an OOM. CUDA/Comfy
                # may have partially touched buffers before raising. Rebuild every
                # conditioning + latent object in a fresh _generate frame instead.
                if str(p.get("_force_te_evict") or "0") == "1":
                    raise
                retry_p = dict(p)
                retry_p["_force_te_evict"] = "1"
                retry_p["_restart_from_sampler"] = "oom"
                j["sampling_label"] = "CLEAN OOM RESTART"
                raise GenerationRestart(retry_p, "oom_clean_restart", str(_oom_exc)) from _oom_exc
            except Exception as _sparse_exc:
                _msg = f"{type(_sparse_exc).__name__}: {_sparse_exc}"
                _msg_low = _msg.lower()
                _convrot_float = (
                    "convrotint8bindingerror" in _msg_low
                    or "floating weight without conversion enabled" in _msg_low
                )
                _cube_order_state = (
                    "h3cubeorderpatcherror" in _msg_low
                    or "finallayer call belongs to another active cube-order state" in _msg_low
                    or "another active cube-order state" in _msg_low
                )
                if sparse_pct <= 0 or not (_convrot_float or _cube_order_state):
                    raise
                if str(p.get("_dense_restart_done") or "0") == "1":
                    raise

                _fallback_code = (
                    "dense_after_cube_order_state_conflict"
                    if _cube_order_state else "dense_after_convrot_float"
                )
                j["sparse_fallback"] = _fallback_code
                j["sparse_fallback_error"] = _msg[:800]
                j["sparse_percent_effective"] = 0.0
                j["sparse_backend_policy_effective"] = "dense_full_restart"
                j["sampling_label"] = "CLEAN DENSE RESTART"

                # Drop any request-local patched descendant. Then unwind this entire
                # frame. The next attempt creates NEW positive conditioning, NEW H3
                # latent, NEW noise object, NEW scheduler and NEW model clone.
                _invalidate_cached_variant(_fallback_code)
                retry_p = dict(p)
                retry_p["sparse_percent"] = "0"
                retry_p["_dense_restart_done"] = "1"
                retry_p["_restart_from_sampler"] = _fallback_code
                raise GenerationRestart(retry_p, _fallback_code, _msg) from _sparse_exc

            torch.cuda.synchronize()
            j["sampling_sec"] = round(time.perf_counter() - t_sample0, 3)
            _check_job_cancel(jid)
            log(
                f"  ✓ sampling wall: {j['sampling_sec']:.3f}s · "
                f"residency={j.get('residency_mode')}"
            )

            # V86 residency handoff.
            # Keep the entire active stack resident whenever the measured Blackwell
            # budget and current decode headroom permit it. Partitioned cards retain
            # the proven VAE handoff.
            _set_job_stage(jid, "preparing decode")
            _unload0 = time.perf_counter()
            free_before_decode_b, total_before_decode_b = torch.cuda.mem_get_info()
            free_before_decode_gib = free_before_decode_b / 1024**3
            j["pre_decode_free_gib"] = round(free_before_decode_gib, 3)

            keep_resident_for_decode = (
                FULL_STACK_RESIDENCY
                and free_before_decode_gib >= FULL_RESIDENCY_MIN_FREE_BEFORE_DECODE_GIB
            )

            # Sampler-only objects can go away; model/clip/VAEs remain held by CACHE
            # and, on full-card Blackwell, remain loaded in Comfy's GPU registry.
            guider = sampler = sigmas = noise = positive = latent = None
            gc.collect()

            if keep_resident_for_decode:
                j["post_sample_unload_sec"] = 0.0
                j["residency_mode"] = "full_resident"
                _pin_persistent_stack(model, clip, vae, avae, reason="pre-decode")
                free_b, total_b = torch.cuda.mem_get_info()
                used_gib = (total_b - free_b) / 1024**3
                MEMLOG = (
                    f"VRAM persistent pre-decode: {used_gib:.1f}/{total_b/1024**3:.1f} GiB used\n"
                    f"loaded models held: {len(mm.current_loaded_models)}\n"
                )
                log(MEMLOG)
                j["memlog"] = j.get("memlog","") + MEMLOG
                j["vram_free"] = round(free_b / 1024**3, 1)
            else:
                _set_job_stage(jid, "freeing dit for vae")
                before = torch.cuda.mem_get_info()[0] / 1024**3
                model = clip = None
                gc.collect()
                unload_err = None
                try:
                    mm.unload_all_models()
                except Exception as _e:
                    unload_err = repr(_e)
                gc.collect()
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
                # V86: avoid an extra raw allocator flush during Comfy's model-manager
                # handoff. Synchronize instead; VAEDecode will request what it needs.
                torch.cuda.synchronize()
                j["post_sample_unload_sec"] = round(time.perf_counter() - _unload0, 3)
                if FULL_STACK_RESIDENCY:
                    j["residency_mode"] = "full_resident_low_headroom_handoff"

                free_b, total_b = torch.cuda.mem_get_info()
                after = free_b / 1024**3
                MEMLOG = (
                    f"VRAM before VAE handoff: {before:.1f} GiB free\n"
                    f"VRAM after  VAE handoff: {after:.1f} GiB free of {total_b/1024**3:.1f} GiB\n"
                    f"unload error: {unload_err}\n"
                    f"loaded models still held: {len(mm.current_loaded_models)}\n"
                )
                log(MEMLOG)
                j["memlog"] = j.get("memlog","") + MEMLOG
                j["vram_free"] = round(after, 1)

            # NOTE: no tiled fallback. The H3 VAE's decode_tiled is a stub that
            # calls decode(), and the latent is a NestedTensor packing video and
            # audio rows, so tiling cannot split it. If this OOMs, cut `length`.
            _set_job_stage(jid, "decoding video")
            _check_job_cancel(jid)
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

            t_vdec0 = time.perf_counter()
            images, = call("VAEDecode", samples=samples, vae=vae)
            torch.cuda.synchronize()
            j["video_decode_sec"] = round(time.perf_counter() - t_vdec0, 3)

            stage_first_png = stage_last_png = None
            try:
                stage_first_png, stage_last_png = _save_stage_frame_pair(jid, images)
                if stage_last_png:
                    log(f"  ✓ staged continuation frame: {os.path.basename(stage_last_png)}")
            except Exception as _e:
                log(f"  ⚠ stage-frame export failed: {_e}")

            _set_job_stage(jid, "decoding audio")
            _check_job_cancel(jid)
            t_adec0 = time.perf_counter()
            audio,  = call("VAEDecodeAudio", samples=samples, vae=avae)
            torch.cuda.synchronize()
            j["audio_decode_sec"] = round(time.perf_counter() - t_adec0, 3)

            if FULL_STACK_RESIDENCY:
                # If the model manager moved anything during VAE decode, restore it
                # immediately so idle VRAM remains warm for the next queued job.
                try:
                    _resident_model, _resident_clip, _resident_vae, _resident_avae, _ = get_models(
                        p["weight_dtype"], p["lora"], float(p["lora_strength"]),
                        action=str(p.get("action") or "0").lower() in ("1","true","yes","on"),
                        action_strength=float(p.get("action_strength") or ACTION_STRENGTH),
                        lightning=str(p.get("lightning") or "0").lower() in ("1","true","yes","on"),
                        lightning_strength=float(p.get("lightning_strength") or LIGHTNING_STRENGTH),
                        unet=p.get("unet") or DIT_FILE,
                        extra_loras=p.get("extra_loras"),
                    )
                    _pin_persistent_stack(
                        _resident_model, _resident_clip, _resident_vae, _resident_avae,
                        reason="post-decode / next-job warm"
                    )
                    free_b, total_b = torch.cuda.mem_get_info()
                    j["resident_post_decode_used_gib"] = round((total_b-free_b)/1024**3, 2)
                except Exception as _e:
                    log(f"  ⚠ post-decode resident restore warning: {_e}")

            _set_job_stage(jid, "muxing")
            # Nothing here needs autograd, and Comfy's server normally runs under
            # inference mode. Without it the audio waveform arrives with a grad
            # graph attached and av's numpy conversion refuses it.
            if isinstance(audio, dict) and "waveform" in audio:
                audio = {**audio, "waveform": audio["waveform"].detach()}
            if torch.is_tensor(images):
                images = images.detach()

            video, = call("CreateVideo", images=images, fps=MODEL_FPS, audio=audio)

            _set_job_stage(jid, "saving")
            _check_job_cancel(jid)
            # Write the container directly. SaveVideo's format/codec are structured
            # V3 values that cannot be synthesised from the schema, and the video
            # object here is already complete.
            t_save0 = time.perf_counter()
            dest = os.path.join(OUT, f"{jid}.mp4")
            requested_playback = max(0.05, min(8.0, float(p.get("playback_speed") or 1.0)))
            requested_final = max(0.01, float(requested_sec))
            # This is the real playback factor after H3 legal-frame snapping. For
            # 7.00 sec at normal speed: 175/24 / 7.00 = 1.0416667x.
            effective_speed = actual_sec / requested_final
            raw_dest = dest if abs(effective_speed - 1.0) < 1e-6 else os.path.join(OUT, f"{jid}.native.mp4")
            video.save_to(raw_dest)
            if not os.path.exists(raw_dest):
                raise RuntimeError("save_to produced no file")
            speed_ok = True
            speed_note = ""
            if raw_dest != dest:
                _set_job_stage(jid, "retiming exact duration")
                speed_ok, speed_note = _retime_video(raw_dest, dest, effective_speed)
                try:
                    if speed_ok and os.path.exists(raw_dest): os.remove(raw_dest)
                except Exception:
                    pass
            final_speed = effective_speed if speed_ok else 1.0
            final_duration = actual_sec / final_speed
            measured_file_duration = _probe_media_duration(dest)
            if measured_file_duration is not None:
                j["file_duration_sec"] = round(measured_file_duration, 4)
                if abs(measured_file_duration - requested_final) > 0.18:
                    speed_note = ((speed_note + " · ") if speed_note else "") + (
                        f"output duration {measured_file_duration:.2f}s differs from requested {requested_final:.2f}s"
                    )
                    log("  ⚠ " + speed_note)
            j["requested_playback_speed"] = round(requested_playback, 5)
            j["effective_playback_speed"] = round(final_speed, 5)
            j["save_retime_sec"] = round(time.perf_counter() - t_save0, 3)
            j["measured_total_sec"] = round(time.perf_counter() - _job_wall0, 3)
            log(
                f"  ✓ V86 STAGES: cond={j.get('conditioning_sec','?')}s · "
                f"TE-transition={j.get('te_evict_sec','?')}s · "
                f"sample={j.get('sampling_sec','?')}s · "
                f"post-unload={j.get('post_sample_unload_sec','?')}s · "
                f"videoVAE={j.get('video_decode_sec','?')}s · "
                f"audioVAE={j.get('audio_decode_sec','?')}s · "
                f"save={j.get('save_retime_sec','?')}s · "
                f"TOTAL={j.get('measured_total_sec','?')}s · "
                f"mode={j.get('residency_mode','?')}"
            )

            with STAGE_LOCK:
                STAGE_STATE.update(
                    first_frame_path=stage_first_png,
                    last_frame_path=stage_last_png,
                    video_file=os.path.basename(dest),
                    job=jid,
                    updated=time.time(),
                )

            # Commit the take to the ACTIVE sequence. GENERATE appends a fresh clip
            # to the current sequence, CONTINUE appends a continuation clip using the
            # previous last frame, and RETRY replaces the last clip in that sequence.
            timeline_action = (p.get("timeline_action") or "new").lower()
            entry = {
                "segment_id": uuid.uuid4().hex[:12],
                "job": jid,
                "file": os.path.basename(dest),
                "duration": round(final_duration, 3),
                "frames": int(n_frames),
                "width": int(width),
                "height": int(height),
                "seed": int(p.get("seed") or 0),
                "prompt": p.get("prompt") or "",
                "first_frame_file": (os.path.basename(stage_first_png) if stage_first_png else None),
                "last_frame_file": (os.path.basename(stage_last_png) if stage_last_png else None),
                "first_frame_path": stage_first_png,
                "last_frame_path": stage_last_png,
                "continued": (timeline_action == "continue" or str(p.get("_retry_continued") or "0") == "1"),
                "continued_from_job": p.get("stage_source_job"),
                "params": dict(p),
            }
            with TIMELINE_LOCK:
                seq = _find_sequence_unlocked(p.get("_target_sequence_id")) if p.get("_target_sequence_id") else _active_sequence_unlocked(create=True)
                if seq is None:
                    seq = _active_sequence_unlocked(create=True)
                segs = list(seq.get("segments") or [])
                if timeline_action == "retry" and segs:
                    segs[-1] = entry
                else:
                    segs.append(entry)
                seq["segments"] = segs
                seq["updated"] = time.time()
                TIMELINE_STATE["updated"] = time.time()
                segment_paths = [os.path.join(OUT, s["file"]) for s in segs]

            stitch0 = time.perf_counter()
            if len(segment_paths) <= 1:
                master_file = os.path.basename(dest)
                stitch_ok, stitch_note = True, "single-segment sequence"
            else:
                master_name = f"timeline_{uuid.uuid4().hex[:10]}.mp4"
                master_path = os.path.join(OUT, master_name)
                _set_job_stage(jid, "stitching timeline")
                stitch_ok, stitch_note = _concat_mp4_timeline(segment_paths, master_path)
                master_file = master_name if stitch_ok else os.path.basename(dest)
            j["stitch_sec"] = round(time.perf_counter() - stitch0, 3)
            j["stitch_note"] = stitch_note
            j["measured_total_sec"] = round(time.perf_counter() - _job_wall0, 3)
            with TIMELINE_LOCK:
                seq = _find_sequence_unlocked(p.get("_target_sequence_id")) if p.get("_target_sequence_id") else _active_sequence_unlocked(create=True)
                if seq is None:
                    seq = _active_sequence_unlocked(create=True)
                seq["master_file"] = master_file
                seq["updated"] = time.time()
                TIMELINE_STATE["updated"] = time.time()
                timeline_count = len(seq.get("segments") or [])
                active_sequence_id = seq.get("id")
                active_sequence_name = seq.get("name") or "Sequence"
                _autosave_timeline_unlocked("generate")
            history_row = dict(entry)
            history_row.update(history_id=uuid.uuid4().hex[:12], created=time.time(), status="done")
            with HISTORY_LOCK:
                HISTORY_STATE.append(history_row)
                if len(HISTORY_STATE) > 300:
                    del HISTORY_STATE[:-300]
                _save_history_unlocked()
            _sync_stage_from_sequence(seq)
            log(
                f"  ✓ V86 TIMELINE: action={timeline_action} · active={active_sequence_name} · clips={timeline_count} · "
                f"stitch={j['stitch_sec']:.3f}s · TOTAL={j['measured_total_sec']:.3f}s · {stitch_note}"
            )

            _set_job_stage(jid, "done")
            j.update(status="done", file=os.path.basename(dest),
                     timeline_master_file=master_file, timeline_count=timeline_count,
                     active_sequence_id=active_sequence_id,
                     active_sequence_name=active_sequence_name,
                     timeline_action=timeline_action,
                     secs=round(time.time()-j["t0"],1), frames=n_frames,
                     model_duration=round(actual_sec, 2), duration=round(final_duration, 2),
                     requested_final_sec=round(requested_sec, 2),
                     playback_speed=round(final_speed, 3), note=speed_note,
                     continued_from_stage=(str(p.get("use_stage_last") or "0") == "1"),
                     stage_last_frame_file=(os.path.basename(stage_last_png) if stage_last_png else None))
        except GenerationRestart:
            raise
        except JobCancelled as e:
            log(f"  ↳ job {jid} cancelled by user")
            j.update(status="cancelled", msg=str(e), secs=round(time.time()-j["t0"], 1))
        except Exception:
            tb = traceback.format_exc()
            traceback.print_exc()
            low_tb = tb.lower()
            if (
                "illegal memory access" in low_tb
                or "cudaerrorillegaladdress" in low_tb
                or "cuda error: an illegal memory access" in low_tb
            ):
                CUDA_CONTEXT_POISONED = True
                log(
                    "  ✗ CUDA context marked POISONED after illegal memory access. "
                    "Queued renders will not be started; rerun the V86 cell for a fresh child process."
                )

            # Put the memory readings AT THE TOP of what the panel shows — the panel
            # truncates long tracebacks from the front, which hid this before.
            head = j.get("memlog", "(failed before the unload step)\n")
            extra = (
                "\nCUDA CONTEXT POISONED — rerun the V86 cell before another render."
                if CUDA_CONTEXT_POISONED else ""
            )
            j.update(status="error", msg=head + "\n" + tb[-1200:] + extra)
        finally:
            PROG["stage"] = "ready"
            gc.collect()
            if not CUDA_CONTEXT_POISONED:
                try:
                    # Normal cleanup only. If CUDA is already poisoned, even this
                    # call can throw and obscure the original error.
                    torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    msg = str(cleanup_err).lower()
                    if "illegal memory access" in msg or "cudaerrorillegaladdress" in msg:
                        CUDA_CONTEXT_POISONED = True
                        log(
                            "  ✗ CUDA allocator cleanup reported illegal memory access; "
                            "context marked POISONED. Rerun the V86 cell."
                        )
                    else:
                        log(f"  ⚠ CUDA cleanup warning: {cleanup_err}")

    QUEUE_WORKER_THREAD = threading.Thread(
        target=_queue_worker, name="h3-generation-queue", daemon=True
    )
    QUEUE_WORKER_THREAD.start()

    # ── 6. UI ──────────────────────────────────────────────────────────────────
    app = Flask(__name__)
    # Flask UI configuration.
    app.config["SECRET_KEY"] = os.environ.get("H3_SESSION_SECRET") or os.urandom(32)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Content-neutral local generation path. The studio performs technical/file
    # validation only; it does not classify or block prompts, checkpoints, or LoRAs.
    def _adult_access_ok():
        return True

    def _prompt_requests_adult(_text):
        return False

    def _asset_blocked(_name):
        return False

    def _is_adult_lora_name(_name):
        return False

    def _is_adult_unet_name(_name):
        return False

    def _profile_from_unet(_unet_name):
        return "stock_quality"

    def _compat_record(*, profiles=None, modes=None, adult=False, label=""):
        return {"profiles":list(profiles or ["stock_quality"]), "modes":list(modes or ["fl2va","ref2va"]),
                "adult":False, "label":str(label or "")}

    def _lora_compatibility_map():
        return {
            MOTION8_FILE:_compat_record(profiles=["stock_quality"],modes=["fl2va"],label="Motion 8-Step Enhancer · Stock H3 FL2VA only"),
            LIGHTNING_FILE:_compat_record(profiles=["stock_quality"],modes=["fl2va"],label="FL2VA Turbo accelerator · managed by FAST preset"),
            REF2VA_LIGHTNING_FILE:_compat_record(profiles=["stock_quality"],modes=["ref2va"],label="Ref2VA Turbo accelerator · managed by FAST preset"),
        }

    def _known_lora_compatible(name, unet_name, mode):
        base=os.path.basename(str(name or ""))
        if not base or base=="none": return True,""
        if _asset_blocked(base): return False,"incompatible with current model/mode"
        info=_lora_compatibility_map().get(base)
        if not info: return True,""
        if "stock_quality" not in info["profiles"] or str(mode or "fl2va") not in info["modes"]:
            return False, info.get("label") or base
        return True,""

    def _adult_catalog():
        return []

    def _adult_request_from_generate_form(_form):
        return False

    def _adult_gate_response():
        return jsonify(ok=True, legacy=True), 200

    @app.post("/api/adult/ack")
    def api_adult_ack():
        return jsonify(ok=True, legacy=True), 200

    @app.post("/api/adult/exit")
    def api_adult_exit():
        return jsonify(ok=True, legacy=True), 200

    @app.before_request
    def _missinglink_ui_gate():
        # Startup already validated the key. Re-check on a short TTL so revoked keys
        # and expired trials do not leave a notebook UI unlocked indefinitely.
        global ML_OK, ML_AUTH_ERROR
        ML_OK, ML_AUTH_ERROR = _validate_missinglink_token(force=False)
        if ML_OK:
            return None
        msg = ML_AUTH_ERROR or "MissingLink API key is not valid."
        if request.path.startswith("/api/"):
            return jsonify(
                error=msg,
                code="missinglink_auth_required",
                trial_url=MISSING_LINK_TRIAL_URL,
            ), 401
        return Response(
            "<!doctype html><meta charset='utf-8'><title>MissingLink access required</title>"
            "<style>body{font:16px system-ui;background:#0d0d0f;color:#eee;padding:48px;max-width:760px;margin:auto}"
            "a{color:#8ab4ff}</style>"
            "<h1>MissingLink access required</h1>"
            f"<p>{msg}</p>"
            "<p>Add a valid <code>MISSING_LINK_TOKEN</code> in Colab Secrets and rerun the cell.</p>"
            f"<p><a href='{MISSING_LINK_TRIAL_URL}' target='_blank'>Start the 7-day free trial / get access</a></p>",
            status=401, mimetype="text/html"
        )

    # ── User-provided Hugging Face base models ─────────────────────────────────
    # Custom checkpoints are explicit user choices. The studio validates transport/file
    # integrity and loader compatibility, but does not make content-policy decisions.
    CUSTOM_MODEL_REGISTRY_FILE = os.path.join(MODELS, ".h3_custom_base_models.json")
    CUSTOM_MODEL_DOWNLOADS = {}
    CUSTOM_MODEL_DOWNLOAD_LOCK = threading.Lock()

    def _hf_token():
        tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
        if tok:
            return tok
        try:
            from google.colab import userdata
            tok = (userdata.get("HF_TOKEN") or "").strip()
        except Exception:
            tok = ""
        if tok:
            os.environ["HF_TOKEN"] = tok
        return tok

    def _custom_model_load_registry():
        try:
            raw = json.load(open(CUSTOM_MODEL_REGISTRY_FILE, "r", encoding="utf-8"))
            if not isinstance(raw, list):
                return []
        except Exception:
            return []
        rows = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            local_name = os.path.basename(str(row.get("local_name") or ""))
            fmt = str(row.get("format") or "safetensors").lower()
            if not local_name:
                continue
            if fmt == "gguf":
                paths = folder_paths.get_folder_paths("unet_gguf") if "unet_gguf" in getattr(folder_paths, "folder_names_and_paths", {}) else []
                exists = any(os.path.exists(os.path.join(p, local_name)) for p in paths)
            else:
                exists = os.path.exists(os.path.join(MODELS, "diffusion_models", local_name))
            if exists:
                row = dict(row)
                row["local_name"] = local_name
                rows.append(row)
        return rows

    def _custom_model_save_registry(rows):
        tmp = CUSTOM_MODEL_REGISTRY_FILE + ".tmp"
        os.makedirs(os.path.dirname(CUSTOM_MODEL_REGISTRY_FILE), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(list(rows), fh, indent=2)
        os.replace(tmp, CUSTOM_MODEL_REGISTRY_FILE)

    def _custom_model_public_rows():
        out = []
        for row in _custom_model_load_registry():
            out.append({
                "profile": "custom:" + row["local_name"],
                "label": row.get("label") or row.get("repo_id") or row["local_name"],
                "repo_id": row.get("repo_id") or "",
                "revision": row.get("revision") or "main",
                "filename": row.get("filename") or row["local_name"],
                "local_name": row["local_name"],
                "mode": row.get("mode") or "both",
                "format": row.get("format") or "safetensors",
                "size": int(row.get("size") or 0),
            })
        return out

    def _hf_repo_candidates(repo_id, revision="main"):
        repo_id = str(repo_id or "").strip().strip("/")
        revision = str(revision or "main").strip() or "main"
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
            raise ValueError("HF repo must look like owner/repository.")
        info = HfApi(token=_hf_token() or None).model_info(repo_id, revision=revision, files_metadata=True)
        rows = []
        for sib in getattr(info, "siblings", []) or []:
            name = str(getattr(sib, "rfilename", "") or "")
            low = name.lower()
            if not low.endswith((".safetensors", ".gguf")):
                continue
            size = int(getattr(sib, "size", 0) or 0)
            if not size:
                lfs = getattr(sib, "lfs", None)
                try:
                    size = int((lfs or {}).get("size") or 0) if isinstance(lfs, dict) else int(getattr(lfs, "size", 0) or 0)
                except Exception:
                    size = 0
            rows.append({"filename": name, "size": size, "format": "gguf" if low.endswith(".gguf") else "safetensors"})
        rows.sort(key=lambda x: (x["size"], x["filename"]), reverse=True)
        return rows

    def _custom_model_destination(repo_id, remote_filename, fmt):
        base = os.path.basename(remote_filename)
        if not base:
            raise ValueError("The selected Hugging Face file has no filename.")
        if fmt == "gguf":
            if "UnetLoaderGGUF" not in N and "UnetLoaderGGUFDynamicVRAM" not in N:
                raise RuntimeError("This runtime has no GGUF UNet loader. Choose a .safetensors H3 checkpoint or enable ComfyUI-GGUF.")
            try:
                roots = folder_paths.get_folder_paths("unet_gguf")
            except Exception:
                roots = []
            if not roots:
                raise RuntimeError("ComfyUI has no unet_gguf model folder in this runtime.")
            root = roots[0]
        else:
            root = os.path.join(MODELS, "diffusion_models")
        os.makedirs(root, exist_ok=True)

        existing = _custom_model_load_registry()
        for row in existing:
            if row.get("repo_id") == repo_id and row.get("filename") == remote_filename:
                return os.path.join(root, row["local_name"]), row["local_name"]
        dest = os.path.join(root, base)
        if os.path.exists(dest):
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id.replace("/", "__"))
            base = slug + "__" + base
            dest = os.path.join(root, base)
        return dest, base

    def _format_bytes(n):
        n = float(n or 0)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024.0 or unit == "TiB":
                return f"{n:.2f} {unit}"
            n /= 1024.0

    def _custom_model_download_worker(download_id, repo_id, revision, filename, mode, expected_size):
        started = time.time()
        fmt = "gguf" if filename.lower().endswith(".gguf") else "safetensors"
        try:
            dest, local_name = _custom_model_destination(repo_id, filename, fmt)
            part = dest + f".{download_id}.part"
            url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
            headers = {"User-Agent": "MissingLink-H3-CustomModel/1"}
            token = _hf_token()
            if token:
                headers["Authorization"] = "Bearer " + token
            req = urllib.request.Request(url, headers=headers)
            downloaded = 0
            with urllib.request.urlopen(req, timeout=180) as resp:
                content_length = int(resp.headers.get("Content-Length") or 0)
                total = int(expected_size or content_length or 0)
                with CUSTOM_MODEL_DOWNLOAD_LOCK:
                    CUSTOM_MODEL_DOWNLOADS[download_id].update(total_bytes=total, local_name=local_name, stage="downloading")
                with open(part, "wb") as fh:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(0.001, time.time() - started)
                        with CUSTOM_MODEL_DOWNLOAD_LOCK:
                            job = CUSTOM_MODEL_DOWNLOADS[download_id]
                            job.update(
                                downloaded_bytes=downloaded,
                                total_bytes=max(int(job.get("total_bytes") or 0), total),
                                speed_bps=downloaded / elapsed,
                                stage="downloading",
                            )
            if expected_size and downloaded != int(expected_size):
                raise RuntimeError(f"Download size mismatch: expected {_format_bytes(expected_size)}, got {_format_bytes(downloaded)}.")
            if fmt == "safetensors":
                ok, err = _validate_safetensors_file(part)
                if not ok:
                    raise RuntimeError("Downloaded checkpoint failed safetensors validation: " + err)
            else:
                with open(part, "rb") as fh:
                    if fh.read(4) != b"GGUF":
                        raise RuntimeError("Downloaded .gguf file does not have a GGUF header.")
            os.replace(part, dest)
            rows = _custom_model_load_registry()
            rows = [r for r in rows if r.get("local_name") != local_name]
            rows.append({
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "local_name": local_name,
                "label": f"{repo_id} · {os.path.basename(filename)}",
                "mode": mode if mode in {"fl2va", "ref2va", "both"} else "both",
                "format": fmt,
                "size": downloaded,
                "installed_at": time.time(),
            })
            _custom_model_save_registry(rows)
            folder_paths.cache_helper.clear()
            with CUSTOM_MODEL_DOWNLOAD_LOCK:
                CUSTOM_MODEL_DOWNLOADS[download_id].update(
                    status="done", stage="done", downloaded_bytes=downloaded,
                    total_bytes=downloaded, speed_bps=0.0, local_name=local_name,
                    profile="custom:" + local_name,
                )
            log(f"  ✓ custom HF base model installed: {repo_id} / {filename} -> {local_name}")
        except Exception as e:
            try:
                if 'part' in locals() and os.path.exists(part):
                    os.remove(part)
            except Exception:
                pass
            with CUSTOM_MODEL_DOWNLOAD_LOCK:
                CUSTOM_MODEL_DOWNLOADS[download_id].update(status="error", stage="error", error=str(e))
            log(f"  ⚠ custom HF base model install failed: {repo_id} / {filename}: {e}")

    @app.post("/api/models/hf/inspect")
    def api_hf_model_inspect():
        body = request.get_json(silent=True) or {}
        repo_id = str(body.get("repo_id") or "").strip()
        revision = str(body.get("revision") or "main").strip() or "main"
        try:
            rows = _hf_repo_candidates(repo_id, revision)
            if not rows:
                return jsonify(error="No .safetensors or .gguf files were found in that repo/revision."), 404
            return jsonify(ok=True, repo_id=repo_id, revision=revision, candidates=rows)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.post("/api/models/hf/install")
    def api_hf_model_install():
        body = request.get_json(silent=True) or {}
        repo_id = str(body.get("repo_id") or "").strip().strip("/")
        revision = str(body.get("revision") or "main").strip() or "main"
        filename = str(body.get("filename") or "").strip().lstrip("/")
        mode = str(body.get("mode") or "both").strip().lower()
        if mode not in {"fl2va", "ref2va", "both"}:
            mode = "both"
        try:
            candidates = _hf_repo_candidates(repo_id, revision)
            by_name = {r["filename"]: r for r in candidates}
            if filename not in by_name:
                return jsonify(error="Choose a model file returned by CHECK REPO."), 400
            selected = by_name[filename]
            free_bytes = shutil.disk_usage("/content").free
            if selected["size"] and free_bytes < selected["size"] + 2 * 1024**3:
                return jsonify(error=f"Not enough disk space. Need about {_format_bytes(selected['size'] + 2 * 1024**3)}, have {_format_bytes(free_bytes)}."), 400
            did = uuid.uuid4().hex[:12]
            with CUSTOM_MODEL_DOWNLOAD_LOCK:
                CUSTOM_MODEL_DOWNLOADS[did] = {
                    "id": did, "status": "queued", "stage": "queued", "error": "",
                    "repo_id": repo_id, "revision": revision, "filename": filename,
                    "mode": mode, "downloaded_bytes": 0, "total_bytes": int(selected["size"] or 0),
                    "speed_bps": 0.0, "started": time.time(), "local_name": "",
                }
            threading.Thread(
                target=_custom_model_download_worker,
                args=(did, repo_id, revision, filename, mode, int(selected["size"] or 0)),
                daemon=True, name=f"hf-model-{did}",
            ).start()
            return jsonify(ok=True, id=did)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.get("/api/models/hf/progress/<download_id>")
    def api_hf_model_progress(download_id):
        with CUSTOM_MODEL_DOWNLOAD_LOCK:
            job = dict(CUSTOM_MODEL_DOWNLOADS.get(download_id) or {})
        if not job:
            return jsonify(error="Unknown model download."), 404
        total = int(job.get("total_bytes") or 0)
        done = int(job.get("downloaded_bytes") or 0)
        job["pct"] = round((100.0 * done / total), 2) if total > 0 else None
        job["downloaded_text"] = _format_bytes(done)
        job["total_text"] = _format_bytes(total) if total else "unknown"
        job["speed_text"] = _format_bytes(job.get("speed_bps") or 0) + "/s" if job.get("speed_bps") else ""
        return jsonify(job)

    @app.get("/api/meta")
    def meta():
        folder_paths.cache_helper.clear()
        all_loras = list(folder_paths.get_filename_list("loras"))
        hidden_accelerators = {LIGHTNING_FILE, REF2VA_LIGHTNING_FILE}
        visible_loras = [x for x in all_loras if x not in hidden_accelerators]
        try:
            gguf_unets = folder_paths.get_filename_list("unet_gguf")
        except Exception:
            gguf_unets = []
        all_unets = sorted(set(folder_paths.get_filename_list("diffusion_models") + gguf_unets))
        custom_models = _custom_model_public_rows()
        allowed_unets = {FALLBACK_DIT_FILE, REF2VA_DIT_FILE, T4_DIT_FILE} | {x["local_name"] for x in custom_models}
        visible_unets = [x for x in all_unets if os.path.basename(str(x)) in allowed_unets]
        # A freshly-installed file may be present before folder_paths refreshes it; keep
        # registry entries visible and let the normal loader provide the technical error.
        for _cm in custom_models:
            if _cm["local_name"] not in visible_unets:
                visible_unets.append(_cm["local_name"])
        visible_unets = sorted(set(visible_unets))
        compat = _lora_compatibility_map()
        profile_state = _model_profile_state()
        return jsonify(
            samplers=SAMPLERS, schedulers=SCHEDULERS, loras=["none"]+visible_loras, custom_models=custom_models,
            lora_default="none", lora_strength_default=0.0,
            lora_compatibility={k:v for k,v in compat.items() if k in visible_loras or k in hidden_accelerators or k==MOTION8_FILE},
            lora_catalog=[], lora_source_map=_lora_source_map(), adult_enabled=False, adult_ack_version="", adult_model_profiles=[],
            gpu_profile=GPU_PROFILE, gpu_name=gpu, gpu_cc=f"{GPU_CC[0]}.{GPU_CC[1]}", gpu_arch_label=GPU_ARCH_LABEL,
            lowvram_t4=bool(LOWVRAM_T4_PROFILE), a100_profile=bool(A100_PROFILE), a100_80=bool(A100_80_PROFILE), a100_40=bool(A100_40_PROFILE),
            t4_default_unet=(T4_DIT_FILE if LOWVRAM_T4_PROFILE else ""), specialty_loras=[],
            lightning_file=LIGHTNING_FILE, lightning_available=bool(not LOWVRAM_T4_PROFILE),
            ref2va_lightning_file=REF2VA_LIGHTNING_FILE, ref2va_lightning_available=bool(not LOWVRAM_T4_PROFILE),
            lightning_default=bool(not LOWVRAM_T4_PROFILE), lightning_strength_default=LIGHTNING_STRENGTH,
            action_label=ACTION_LABEL, action_model_id=0, action_linked_version=0, action_available=False, action_file=None,
            action_strength_default=0.0, action_requested_version_id=0, action_version_id=None, action_source_page="",
            action_resolution="", action_version_name="", action_base_model="", action_trained_words=[],
            sparse_available=bool(SPARSE_NODE_NAME), sparse_node=SPARSE_NODE_NAME or "", sparse_default_percent=DEFAULT_SPARSE_PERCENT,
            attention_backend=ATTN_BACKEND, torch_version=torch.__version__, torch_cuda=str(torch.version.cuda),
            startup_gpu_preloaded=bool(STARTUP_GPU_PRELOADED), fast_startup=bool(FAST_STARTUP),
            full_stack_residency=bool(FULL_STACK_RESIDENCY), full_stack_weights_gib=round(_full_stack_weights_gib,2),
            full_stack_required_gib=round(_full_stack_required_gib,2), physical_vram_gib=round(_physical_vram_gib,2),
            h3opt_commit=_h3opt_commit[:12] if _h3opt_commit else "", ref2va_available=bool(REF2VA_NODE_NAME), ref2va_node=REF2VA_NODE_NAME or "",
            ref2va_unet=REF2VA_DIT_FILE, stock_ref2va_unet=REF2VA_DIT_FILE,
            base_fl2va_unet=(T4_DIT_FILE if LOWVRAM_T4_PROFILE else FALLBACK_DIT_FILE),
            eros_max_unet="", redmix_unet="", eros_max_sha256="", eros_integrated_turbo=False,
            motion8_file=MOTION8_FILE, motion8_available=bool(not LOWVRAM_T4_PROFILE), model_profiles=profile_state,
            naughty_file="", naughty_strength=0.0,
            stock_quality_sampler="res_multistep", stock_quality_scheduler="simple", stock_quality_steps=20,
            stock_quality_shift_video=12.0, stock_quality_shift_audio=3.0,
            quality_text_encoder=TEXT_ENCODER_FILE, quality_sampler="euler", quality_scheduler="simple", quality_steps=8,
            quality_shift_video=12.0, quality_shift_audio=7.0, fast_sampler="lcm", fast_scheduler="simple", fast_steps=6,
            fast_shift_video=1.0, fast_shift_audio=1.0, ref2va_max_images=9, ref2va_max_videos=3, ref2va_max_audios=3,
            unets=visible_unets, unet_default=DIT_FILE, using_redmix=False, using_eros_max=False,
            model_mode=("T4 LOW-VRAM · Q4_0 GGUF · Dynamic VRAM" if LOWVRAM_T4_PROFILE else "MINIMAX H3 STUDIO"),
            fallback_notice="", recommended_steps=20, recommended_sampler="res_multistep", recommended_scheduler="simple",
        )

    AUTO_PROMPT_MODELS = [
        {"id":"gpt-5.6-terra", "label":"GPT-5.6 Terra · balanced"},
        {"id":"gpt-5.6-sol", "label":"GPT-5.6 Sol · highest quality"},
        {"id":"gpt-5.6-luna", "label":"GPT-5.6 Luna · fastest / lowest cost"},
    ]
    AUTO_PROMPT_DEFAULT_MODEL = "gpt-5.6-terra"

    def _openai_api_key():
        try:
            from google.colab import userdata
            key = (userdata.get("OPENAI_API_KEY") or "").strip()
            if key:
                return key
        except Exception:
            pass
        return (os.environ.get("OPENAI_API_KEY") or "").strip()

    def _vision_data_url(source):
        img = Image.open(source).convert("RGB")
        max_edge = 1536
        if max(img.size) > max_edge:
            scale = max_edge / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _extract_response_text(payload):
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        chunks = []
        for item in payload.get("output") or []:
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text"):
                    val = part.get("text")
                    if isinstance(val, str) and val.strip():
                        chunks.append(val.strip())
        return "\n".join(chunks).strip()

    def _auto_prompt_system(mode, duration, extra):
        if mode == "i2va":
            first_rule = ('The final prompt MUST begin exactly with: "For the target video, at 0.00 seconds into the target video, '
                          '<Picture 1> (from [Shot 1]) is fully referenced." Then one blank line.')
        elif mode == "fl2va":
            first_rule = (f'The final prompt MUST begin with: "How the reference pictures align with the target video — '
                          f'Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) '
                          f'aligns with the {duration:.2f}-second mark of the target video." Then one blank line. '
                          'Prefer one continuous shot unless the user explicitly asks for cuts.')
        elif mode == "l2va":
            first_rule = (f'The final prompt MUST begin with: "How the reference pictures align with the target video — '
                          f'<Picture 1> (from [Shot N]) aligns with the {duration:.2f}-second mark of the target video." Then one blank line.')
        else:
            first_rule = "T2VA has no picture-alignment line; begin directly with integrated_multimodal_description."
        extra_block = (extra or "").strip()
        return f"""You are the Auto Prompt director inside a MiniMax H3 audiovisual generation studio.
Return ONLY the finished MiniMax H3 prompt. No markdown fences, preamble, explanation, alternatives, or commentary.

Follow MiniMax H3's documented prompt grammar:
1. {first_rule}
2. Then output exactly these three top-level fields, in this order:
   integrated_multimodal_description: ...
   overall_soundscape: ...
   non_diegetic_music: ...
3. Write the video in playback order. Start with [Shot 1] and no timestamp on Shot 1. Only create later shots when a real cut is useful; later shots use increasing timestamps such as [Shot 2] At 00:03.500, ...
4. Describe observable motion, body/object actions, reactions, spatial relationships, lighting changes, camera behavior, and the final landing state. Avoid keyword soup and contradictory camera moves.
5. Camera movement is natural English: movement type plus meaningful amplitude/speed. Prefer a single controlled camera path when possible.
6. If reference images are supplied, VISUALLY INSPECT EVERY supplied reference image. Preserve visible identity, apparent adult age, wardrobe, body proportions, colors, objects, framing, lighting, and scene geometry unless the user's intent clearly changes them. Do not hallucinate text that is not visible.
7. Reference frames are hard visual anchors, not optional inspiration:
   - If a FIRST FRAME is supplied, Picture 1 must anchor the exact opening composition/state at 0.00 seconds.
   - If a LAST FRAME is supplied, it must anchor the exact final composition/state at the requested end time.
   - If BOTH are supplied, you MUST reference and reason about BOTH images. Describe a physically and visually continuous path from the first image to the last image, preserving identity and scene continuity, and explicitly land on the last-frame composition at the end. Never ignore the last image merely because the first image is more detailed.
8. For I2VA: first-frame anchor -> action onset -> continuous development -> result/reaction. For FL2VA: explicitly describe the continuous physical/compositional path from Picture 1 to Picture 2 and land on Picture 2 at the end.
9. Dialogue belongs in integrated_multimodal_description. Preserve user-provided dialogue verbatim. For speaking subjects use stable IDs like (S1) and H3 dialogue tags such as <d>[English] exact words</d> when appropriate.
10. overall_soundscape is 1-4 sentences of ambience, physical sounds, and non-verbal human sounds; do not duplicate dialogue. Use N/A only if the user explicitly wants complete silence.
11. non_diegetic_music is 1-3 sentences describing instrumentation, tempo/rhythm, and dynamics, or N/A when no background score is wanted.
12. Follow the user's creative direction faithfully; do not add unrelated content or change the requested tone.
13. Keep the prompt precise enough to control H3 but concise enough that the main movement and camera path remain dominant.

Additional user Auto Prompt instructions:
{extra_block if extra_block else '(none)'}
"""

    @app.get("/api/auto_prompt/meta")
    def api_auto_prompt_meta():
        return jsonify(ok=True, key_available=bool(_openai_api_key()), models=AUTO_PROMPT_MODELS, default_model=AUTO_PROMPT_DEFAULT_MODEL)

    @app.post("/api/auto_prompt")
    def api_auto_prompt():
        key = _openai_api_key()
        if not key:
            return jsonify(error="OPENAI_API_KEY is not available to the UI process. V86 forwards it from Colab Secrets at startup; rerun this cell after adding or changing the secret."), 400
        model = (request.form.get("model") or AUTO_PROMPT_DEFAULT_MODEL).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{2,120}", model):
            return jsonify(error="Invalid OpenAI model id."), 400
        rough = (request.form.get("rough_prompt") or "").strip()
        extra = (request.form.get("extra_instructions") or "").strip()[:8000]
        try:
            duration = max(0.21, min(149.7, float(request.form.get("duration") or 7.0)))
        except Exception:
            duration = 7.0

        first_data = last_data = None
        first_upload = request.files.get("first_frame")
        last_upload = request.files.get("last_frame")
        try:
            if first_upload and first_upload.filename:
                first_data = _vision_data_url(first_upload.stream)
            elif str(request.form.get("use_stage_last") or "0") == "1":
                with STAGE_LOCK:
                    stage_path = STAGE_STATE.get("last_frame_path")
                if stage_path and os.path.exists(stage_path):
                    first_data = _vision_data_url(stage_path)
            if last_upload and last_upload.filename:
                last_data = _vision_data_url(last_upload.stream)
        except Exception as e:
            return jsonify(error=f"Could not prepare reference image for Auto Prompt: {e}"), 400

        mode = "fl2va" if first_data and last_data else "i2va" if first_data else "l2va" if last_data else "t2va"
        reference_contract = (
            "Reference contract: "
            + ("FIRST FRAME is a hard opening anchor at 0.00 seconds. " if first_data else "")
            + (f"LAST FRAME is a hard ending anchor at {duration:.2f} seconds. " if last_data else "")
            + ("Both references must be satisfied in one continuous visual trajectory." if first_data and last_data else "")
        ).strip()
        user_text = (f"Create the final MiniMax H3 prompt for a {duration:.2f}-second audiovisual clip.\n"
                     f"Canvas: {request.form.get('width') or '?'}x{request.form.get('height') or '?'}.\n"
                     f"Workflow mode: {mode.upper()}.\n"
                     f"{reference_contract}\n"
                     f"User's rough creative intent:\n{rough if rough else '(No rough text was supplied. Infer a coherent, conservative motion plan from the supplied reference image(s) if present; do not invent dialogue.)'}")
        content = [{"type":"input_text", "text":user_text}]
        if first_data:
            content += [{"type":"input_text", "text":"Picture 1 is the FIRST FRAME reference. Inspect it carefully."},
                        {"type":"input_image", "image_url":first_data}]
        if last_data:
            label = (
                f"Picture 2 is the LAST FRAME reference and a HARD endpoint at {duration:.2f} seconds. "
                "Inspect it as carefully as Picture 1. The generated description must end on this exact visual state/composition, "
                "with a continuous, plausible transition from Picture 1."
                if first_data else
                f"Picture 1 is the LAST FRAME reference and a HARD endpoint at {duration:.2f} seconds. "
                "Inspect it carefully and construct a plausible preceding state and motion path that lands on this exact visual state/composition."
            )
            content += [{"type":"input_text", "text":label}, {"type":"input_image", "image_url":last_data}]

        payload = {"model":model, "instructions":_auto_prompt_system(mode, duration, extra),
                   "input":[{"role":"user", "content":content}], "max_output_tokens":2400, "store":False}
        req = urllib.request.Request("https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization":"Bearer "+key, "Content-Type":"application/json", "User-Agent":"MissingLink-H3-AutoPrompt/50"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[-2000:]
            try: msg = (json.loads(detail).get("error") or {}).get("message") or detail
            except Exception: msg = detail
            return jsonify(error=f"OpenAI Auto Prompt error ({e.code}): {msg}"), 502
        except Exception as e:
            return jsonify(error=f"OpenAI Auto Prompt request failed: {e}"), 502
        final_prompt = _extract_response_text(result)
        if not final_prompt:
            return jsonify(error="The selected OpenAI model returned no prompt text."), 502
        return jsonify(
            ok=True,
            prompt=final_prompt,
            model=model,
            mode=mode,
            used_image=bool(first_data or last_data),
            used_first_image=bool(first_data),
            used_last_image=bool(last_data),
        )

    @app.get("/api/keepalive")
    def keepalive(): return jsonify(ok=True)

    def _clean_upload_ext(name, fallback):
        ext = os.path.splitext(str(name or ""))[1].lower()
        ext = re.sub(r"[^a-z0-9.]", "", ext)[:10]
        if not ext.startswith(".") or ext in {".", ".."}:
            ext = fallback
        return ext or fallback

    def _save_binary_upload(storage, path):
        storage.stream.seek(0)
        with open(path, "wb") as fh:
            shutil.copyfileobj(storage.stream, fh)

    @app.post("/api/generate")
    def api_gen():
        if not ML_OK:
            return jsonify(error="MissingLink token not validated."), 402
        jid = uuid.uuid4().hex[:8]
        p = {k: request.form.get(k) for k in
             ("prompt","width","height","duration","frames","length_mode",
              "playback_speed","image_fit","steps","seed","denoise",
              "shift_video","shift_audio","sparse_percent","sampler_name","scheduler",
              "weight_dtype","lora","lora_strength","motion8","motion8_strength","action","action_strength","lightning",
              "lightning_strength","unet", "use_stage_last", "timeline_action",
              "input_mode", "ref_image_size")}
        input_mode = (p.get("input_mode") or "fl2va").strip().lower()
        if input_mode not in {"fl2va", "ref2va"}:
            input_mode = "fl2va"
        p["input_mode"] = input_mode
        _allowed_unets = {FALLBACK_DIT_FILE, REF2VA_DIT_FILE, T4_DIT_FILE} | {x["local_name"] for x in _custom_model_public_rows()}
        requested_unet = os.path.basename(str(p.get("unet") or DIT_FILE))
        if requested_unet not in _allowed_unets:
            return jsonify(error="That base model is not installed through the studio. Add it with + HF BASE MODEL first."), 400
        p["unet"] = requested_unet
        # Normal GENERATE is raw/local: it never invokes OpenAI Auto Prompt.
        p["prompt_source"] = "raw_local"

        extra_loras = []
        # V96 named LoRA cards submit one explicit JSON stack. Only filenames already
        # present in ComfyUI's LoRA folder are accepted; arbitrary paths are rejected.
        try:
            named_stack = json.loads(request.form.get("lora_stack_json") or "[]")
        except Exception:
            named_stack = []
        installed_loras = set(folder_paths.get_filename_list("loras"))
        if isinstance(named_stack, list):
            for item in named_stack:
                if not isinstance(item, dict):
                    continue
                name = os.path.basename(str(item.get("file") or ""))
                if not name or name == "none":
                    continue
                if name not in installed_loras:
                    return jsonify(error=f"LoRA is not installed: {name}"), 400
                try:
                    strength = float(item.get("strength") or 0.0)
                except Exception:
                    strength = 0.0
                if abs(strength) > 1e-6:
                    extra_loras.append((name, strength))

        for i in range(1, 5):
            name = (request.form.get(f"extra_lora_{i}") or "none").strip()
            try:
                strength = float(request.form.get(f"extra_lora_strength_{i}") or 0.0)
            except Exception:
                strength = 0.0
            if name and name != "none" and abs(strength) > 1e-6:
                extra_loras.append((name, strength))

        # Config-driven specialty presets are mapped by preset key to the verified
        # installed filename. The client cannot inject an arbitrary LoRA path here.
        try:
            requested_specialty = json.loads(request.form.get("specialty_loras_json") or "[]")
        except Exception:
            requested_specialty = []
        specialty_by_key = {str(s.get("key")): s for s in SPECIALTY_LORA_STATES if s.get("available") and s.get("file")}
        for item in requested_specialty if isinstance(requested_specialty, list) else []:
            key = str((item or {}).get("key") or "")
            spec = specialty_by_key.get(key)
            if not spec:
                continue
            try:
                strength = float((item or {}).get("strength") or 0.0)
            except Exception:
                strength = 0.0
            if abs(strength) > 1e-6:
                extra_loras.append((spec["file"], strength))
        # Dedicated Motion Enhancer slider is applied through the same normal
        # LoRA stack/cache path as Extra LoRAs.
        try:
            motion8_strength = float(p.get("motion8_strength") or 0.0)
        except Exception:
            motion8_strength = 0.0
        p["motion8_strength"] = motion8_strength
        p["motion8"] = "1" if abs(motion8_strength) > 1e-6 else "0"

        if abs(motion8_strength) > 1e-6:
            if input_mode != "fl2va":
                return jsonify(error="Motion Enhancer is CURRENT / FL2VA only. Switch to CURRENT · KEYFRAMES."), 400
            extra_loras = [(n, s) for n, s in extra_loras if n != MOTION8_FILE]
            extra_loras.append((MOTION8_FILE, motion8_strength))
            # Never silently cancel another ON switch. Motion8 and the 4-step
            # accelerator are alternatives, so reject an explicit conflicting stack.
            try:
                _lightning_on = str(p.get("lightning") or "0").lower() in ("1", "true", "yes", "on") and abs(float(p.get("lightning_strength") or 0.0)) > 1e-6
            except Exception:
                _lightning_on = False
            if _lightning_on:
                return jsonify(error="Motion Enhancer and Fast-Mode Accelerator are alternative LoRAs. Turn one OFF before generating; neither selection was changed."), 400

        _dedup = {}
        for _name, _strength in extra_loras:
            _dedup[os.path.basename(str(_name))] = float(_strength)
        extra_loras = [(n, v) for n, v in _dedup.items() if n and abs(v) > 1e-6]
        p["extra_loras"] = extra_loras

        # Enforce known compatibility server-side too; disabled browser options are
        # UX, not a security/integrity boundary. Unknown user LoRAs remain allowed.
        selected_loras = []
        try:
            selected_loras.append((p.get("lora") or "none", float(p.get("lora_strength") or 0.0)))
        except Exception:
            selected_loras.append((p.get("lora") or "none", 0.0))
        selected_loras.extend(extra_loras)
        active_unet = p.get("unet") or DIT_FILE
        for lname, lstrength in selected_loras:
            if not lname or lname == "none" or abs(float(lstrength or 0.0)) <= 1e-6:
                continue
            ok, label = _known_lora_compatible(lname, active_unet, input_mode)
            if not ok:
                return jsonify(
                    error=f"LoRA '{lname}' is known to be incompatible with the selected model/mode ({label})."
                ), 400

        if input_mode == "ref2va":
            # studio Ref2VA uses the official MiniMax H3 reference checkpoint.
            if not p.get("unet"):
                p["unet"] = REF2VA_DIT_FILE
            ref_images = []
            ref_videos = []
            ref_audios = []
            p["use_stage_last"] = "0"
            for i in range(1, 10):
                f = request.files.get(f"ref_image_{i}")
                if f and f.filename:
                    path = os.path.join(OUT, f"{jid}_ref_image_{i}.png")
                    Image.open(f.stream).convert("RGB").save(path)
                    ref_images.append(path)
            for i in range(1, 4):
                f = request.files.get(f"ref_video_{i}")
                if f and f.filename:
                    ext = _clean_upload_ext(f.filename, ".mp4")
                    path = os.path.join(OUT, f"{jid}_ref_video_{i}{ext}")
                    _save_binary_upload(f, path)
                    ref_videos.append(path)
            for i in range(1, 4):
                f = request.files.get(f"ref_audio_{i}")
                if f and f.filename:
                    ext = _clean_upload_ext(f.filename, ".wav")
                    path = os.path.join(OUT, f"{jid}_ref_audio_{i}{ext}")
                    _save_binary_upload(f, path)
                    ref_audios.append(path)
            if not ref_images and not ref_videos:
                return jsonify(error="Ref2VA requires at least one reference image or one reference video."), 400
            p["ref_images"] = ref_images
            p["ref_videos"] = ref_videos
            p["ref_audios"] = ref_audios
        else:
            for k in ("first_frame","last_frame"):
                f = request.files.get(k)
                if f and f.filename:
                    path = os.path.join(OUT, f"{jid}_{k}.png")
                    Image.open(f.stream).convert("RGB").save(path)
                    p[k] = path

        timeline_action = (p.get("timeline_action") or "new").lower()
        if timeline_action != "continue":
            p["use_stage_last"] = "0"
        if input_mode != "fl2va" and timeline_action == "continue":
            return jsonify(error="Continue uses the FL2VA first/last-frame path. Switch back to the Current Model tab for continuation renders."), 400
        with TIMELINE_LOCK:
            target_seq = _active_sequence_unlocked(create=True)
            p["_target_sequence_id"] = target_seq.get("id")
            target_has_clips = bool(target_seq.get("segments"))
        if timeline_action == "continue" and not target_has_clips:
            # Allow a continue behind another queued job targeting this sequence.
            with QUEUE_LOCK:
                has_pending_same = any((JOBS.get(qid, {}).get("_params") or {}).get("_target_sequence_id") == p["_target_sequence_id"] for qid in JOB_QUEUE)
                if QUEUE_ACTIVE_JOB and JOBS.get(QUEUE_ACTIVE_JOB, {}).get("target_sequence_id") == p["_target_sequence_id"]:
                    has_pending_same = True
            if not has_pending_same:
                return jsonify(error="The active sequence is empty. Generate or queue its first clip before continuing."), 400
        # Explicit uploaded first frames are already stored above. Stage/continue
        # references are resolved by the worker when this job reaches the GPU.
        thumb_src = p.get("first_frame") or ((p.get("ref_images") or [None])[0])
        thumb_file = os.path.basename(thumb_src) if thumb_src else None
        _now = time.time()
        JOBS[jid] = {
            "status":"queued", "t0":_now, "prompt":p.get("prompt") or "",
            "requested_duration":float(p.get("duration") or 0), "thumb_file":thumb_file,
            "target_sequence_id":p.get("_target_sequence_id"), "timeline_action":timeline_action,
            "stage":"queued", "stage_started":_now, "last_activity":_now, "cur":0, "total":0,
        }
        _enqueue_generation(jid, p)
        return jsonify(id=jid, queued=True)

    @app.get("/api/timeline")
    def api_timeline():
        return jsonify(_timeline_public_state())

    @app.get("/api/timeline/projects")
    def api_timeline_projects():
        return jsonify(projects=_list_saved_projects())

    @app.post("/api/timeline/save")
    def api_timeline_save():
        body = request.get_json(silent=True) or request.form or {}
        with TIMELINE_LOCK:
            if body.get("project_name"):
                TIMELINE_STATE["project_name"] = _safe_timeline_name(body.get("project_name"), "Current Project")
            TIMELINE_STATE["updated"] = time.time()
            path = _autosave_timeline_unlocked("manual_save")
        return jsonify(ok=True, project_file=os.path.basename(path), timeline=_timeline_public_state())

    @app.post("/api/timeline/load")
    def api_timeline_load():
        body = request.get_json(silent=True) or request.form or {}
        project_file = os.path.basename(str(body.get("project_file") or "").strip())
        if not project_file:
            return jsonify(error="Choose a saved project to load."), 400
        path = os.path.join(PROJECTS_DIR, project_file)
        if not os.path.exists(path):
            return jsonify(error=f"Saved project not found: {project_file}"), 404
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        with TIMELINE_LOCK:
            active = _restore_timeline_from_payload_unlocked(payload)
            TIMELINE_STATE["project_file"] = project_file
            TIMELINE_STATE["updated"] = time.time()
            _autosave_timeline_unlocked("load_request")
        _sync_stage_from_sequence(active)
        return jsonify(ok=True, timeline=_timeline_public_state())

    @app.post("/api/timeline/select_sequence")
    def api_timeline_select_sequence():
        body = request.get_json(silent=True) or request.form or {}
        seq_id = str(body.get("sequence_id") or "").strip()
        with TIMELINE_LOCK:
            seq = _find_sequence_unlocked(seq_id)
            if not seq:
                return jsonify(error="Sequence not found."), 404
            TIMELINE_STATE["active_sequence_id"] = seq["id"]
            TIMELINE_STATE["updated"] = time.time()
            _autosave_timeline_unlocked("select_sequence")
        _sync_stage_from_sequence(seq)
        return jsonify(ok=True, timeline=_timeline_public_state())

    @app.post("/api/timeline/new_sequence")
    def api_timeline_new_sequence():
        body = request.get_json(silent=True) or request.form or {}
        with TIMELINE_LOCK:
            seqs = TIMELINE_STATE.get("sequences") or []
            seq = _new_sequence(body.get("name") or f"Sequence {len(seqs)+1}")
            seqs.append(seq)
            TIMELINE_STATE["sequences"] = seqs
            TIMELINE_STATE["active_sequence_id"] = seq["id"]
            TIMELINE_STATE["updated"] = time.time()
            _autosave_timeline_unlocked("new_sequence")
        _sync_stage_from_sequence(seq)
        return jsonify(ok=True, sequence_id=seq["id"], timeline=_timeline_public_state())

    @app.post("/api/timeline/duplicate_active")
    def api_timeline_duplicate_active():
        with TIMELINE_LOCK:
            src = _active_sequence_unlocked(create=True)
            clone = _new_sequence((src.get("name") or "Sequence") + " copy")
            clone["segments"] = [dict(seg) for seg in (src.get("segments") or [])]
            clone["master_file"] = src.get("master_file")
            TIMELINE_STATE["sequences"].append(clone)
            TIMELINE_STATE["active_sequence_id"] = clone["id"]
            TIMELINE_STATE["updated"] = time.time()
            _autosave_timeline_unlocked("duplicate_sequence")
        _sync_stage_from_sequence(clone)
        return jsonify(ok=True, sequence_id=clone["id"], timeline=_timeline_public_state())

    @app.post("/api/timeline/compile_active")
    def api_timeline_compile_active():
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            seq_id = seq.get("id")
            clip_count = len(seq.get("segments") or [])
        if clip_count == 0:
            return jsonify(error="The active sequence has no clips to compile."), 400
        ok, master_file, note = _restitch_sequence(seq_id, "manual_compile")
        if not ok or not master_file:
            return jsonify(error=note or "Could not compile the active sequence."), 500
        return jsonify(
            ok=True,
            file=master_file,
            clip_count=clip_count,
            note=note,
            timeline=_timeline_public_state(),
        )

    @app.post("/api/timeline/retry_last")
    def api_timeline_retry_last():
        if not ML_OK:
            return jsonify(error="MissingLink token not validated."), 402
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            segs = list(seq.get("segments") or [])
        if not segs:
            return jsonify(error="The active sequence is empty; there is nothing to retry."), 400
        last = segs[-1]
        p = dict(last.get("params") or {})
        if not p:
            return jsonify(error="The last segment has no saved generation settings."), 400
        retry_seed = int(uuid.uuid4().hex[:12], 16) % 2147483647
        p["seed"] = str(retry_seed)
        p["timeline_action"] = "retry"
        p["use_stage_last"] = "0"
        p["_retry_continued"] = "1" if last.get("continued") else "0"
        p["_target_sequence_id"] = seq.get("id")
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"status":"queued", "t0":time.time(), "stage":"queued", "stage_started":time.time(), "last_activity":time.time(), "retry_of":last.get("job"),
                     "prompt":p.get("prompt") or "", "requested_duration":float(p.get("duration") or 0),
                     "thumb_file":last.get("last_frame_file")}
        _enqueue_generation(jid,p)
        return jsonify(id=jid, seed=retry_seed, queued=True)

    @app.post("/api/timeline/clear")
    def api_timeline_clear():
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            seq["segments"] = []
            seq["master_file"] = None
            seq["updated"] = time.time()
            TIMELINE_STATE["updated"] = time.time()
            _autosave_timeline_unlocked("clear_active_sequence")
        _sync_stage_from_sequence(seq)
        return jsonify(ok=True, timeline=_timeline_public_state())

    @app.get("/api/history")
    def api_history():
        return jsonify(items=_public_history())

    @app.delete("/api/history/<history_id>")
    def api_history_delete(history_id):
        with HISTORY_LOCK:
            before = len(HISTORY_STATE)
            HISTORY_STATE[:] = [h for h in HISTORY_STATE if h.get("history_id") != history_id]
            _save_history_unlocked()
        return jsonify(ok=(len(HISTORY_STATE) != before))

    def _model_profile_state():
        return dict(
            stock_fl2va_installed=(os.path.exists(os.path.join(MODELS,"unet",T4_DIT_FILE)) if LOWVRAM_T4_PROFILE else os.path.exists(os.path.join(MODELS,"diffusion_models",FALLBACK_DIT_FILE))),
            stock_ref2va_installed=(False if LOWVRAM_T4_PROFILE else os.path.exists(os.path.join(MODELS,"diffusion_models",REF2VA_DIT_FILE))),
            eros_installed=False, redmix_installed=False, naughty_installed=False,
        )

    @app.post("/api/model_profiles/install")
    def api_model_profile_install():
        body=request.get_json(silent=True) or {}
        profile=str(body.get("profile") or "stock_quality").strip().lower()
        mode=str(body.get("mode") or "fl2va").strip().lower()
        if profile != "stock_quality":
            return jsonify(error="The built-in profile installer handles the bundled H3 model only. Use + HF BASE MODEL for custom checkpoints."), 400
        if mode not in {"fl2va","ref2va"}: mode="fl2va"
        try:
            if LOWVRAM_T4_PROFILE:
                if mode=="ref2va": return jsonify(error="Ref2VA is not enabled on the T4 low-VRAM profile."),400
                _fetch_one(("unet",T4_DIT_FILE,T4_DIT_REPO,None))
            elif mode=="ref2va":
                _fetch_one(("diffusion_models",REF2VA_DIT_FILE,"Comfy-Org/MiniMax-H3","diffusion_models"))
            else:
                _fetch_one(("diffusion_models",FALLBACK_DIT_FILE,"Comfy-Org/MiniMax-H3","diffusion_models"))
            folder_paths.cache_helper.clear()
            _allowed_unets={FALLBACK_DIT_FILE,REF2VA_DIT_FILE,T4_DIT_FILE}
            unets=[x for x in folder_paths.get_filename_list("diffusion_models") if os.path.basename(str(x)) in _allowed_unets]
            loras=list(folder_paths.get_filename_list("loras"))
            return jsonify(ok=True,profile="stock_quality",mode=mode,state=_model_profile_state(),unets=unets,loras=["none"]+loras)
        except Exception as e:
            return jsonify(error=str(e),state=_model_profile_state()),400

    @app.post("/api/loras/catalog_install")
    def api_lora_catalog_install():
        return jsonify(error="No built-in LoRA catalog is configured. Install LoRAs from Hugging Face or CivitAI.", code="studio_only"), 404

    @app.post("/api/loras/install")
    def api_lora_install():
        body = request.get_json(silent=True) or {}
        source_url = str(body.get("url") or "").strip()
        if not source_url:
            return jsonify(error="Paste a Hugging Face or CivitAI LoRA URL."), 400

        from urllib.parse import urlsplit, parse_qs, unquote
        parts = urlsplit(source_url)
        host = (parts.hostname or "").lower()
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(lora_dir, exist_ok=True)

        try:
            friendly_source_url = ""
            if host in ("huggingface.co", "www.huggingface.co"):
                seg = [unquote(x) for x in parts.path.split("/") if x]
                if len(seg) < 5 or seg[2] not in ("blob", "resolve"):
                    raise ValueError(
                        "Hugging Face URL must point directly to a .safetensors file "
                        "(.../blob/<revision>/path/file.safetensors or .../resolve/<revision>/...)."
                    )
                repo_id = f"{seg[0]}/{seg[1]}"
                revision = seg[3]
                filename = "/".join(seg[4:])
                base = os.path.basename(filename)
                if not base.lower().endswith(".safetensors"):
                    raise ValueError("Only .safetensors LoRA files are accepted.")
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
                src_path = hf_hub_download(repo_id, filename=filename, revision=revision, token=token)
                dest = os.path.join(lora_dir, base)
                _atomic_copy_weight(src_path, dest)
                installed_name = base
                friendly_source_url = _hf_repo_url(repo_id)

            elif host in ("civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"):
                tok = _civitai_token()
                q = parse_qs(parts.query)
                seg = [x for x in parts.path.split("/") if x]
                version_obj = None
                if len(seg) >= 4 and seg[:3] == ["api", "download", "models"]:
                    version_id = int(seg[3])
                    version_obj = _civitai_json(f"https://civitai.com/api/v1/model-versions/{version_id}", tok)
                    model_id = int(version_obj.get("modelId") or 0)
                    if model_id and version_id:
                        friendly_source_url = f"https://civitai.red/models/{model_id}?modelVersionId={version_id}"
                elif len(seg) >= 2 and seg[0] == "models":
                    model_id = int(seg[1].split("-")[0])
                    model = _civitai_json(f"https://civitai.com/api/v1/models/{model_id}", tok)
                    versions = model.get("modelVersions") or []
                    requested = (q.get("modelVersionId") or [None])[0]
                    if requested:
                        version_obj = next((v for v in versions if int(v.get("id") or 0) == int(requested)), None)
                        if version_obj is None:
                            raise ValueError(f"CivitAI modelVersionId {requested} was not found.")
                    else:
                        compatible = [v for v in versions if _is_h3_version(v) and _pick_model_file(v)]
                        if not compatible:
                            raise ValueError("No MiniMax-H3-compatible .safetensors version was found on that CivitAI model.")
                        compatible.sort(key=lambda v: (str(v.get("createdAt") or ""), int(v.get("id") or 0)), reverse=True)
                        version_obj = compatible[0]
                    version_id = int(version_obj.get("id") or 0)
                    if model_id and version_id:
                        friendly_source_url = f"https://civitai.red/models/{model_id}?modelVersionId={version_id}"
                else:
                    raise ValueError("CivitAI URL must be a civitai.com/civitai.red model page or /api/download/models/<versionId> URL.")

                if not _is_h3_version(version_obj):
                    raise ValueError(
                        f"CivitAI version {version_obj.get('id')} is labeled "
                        f"{version_obj.get('baseModel') or version_obj.get('name') or 'non-H3'}; refusing to install it into MiniMax H3."
                    )
                fobj = _pick_model_file(version_obj)
                if not fobj:
                    raise ValueError("CivitAI version exposes no .safetensors model file.")
                base = os.path.basename(str(fobj.get("name") or ""))
                if not base.lower().endswith(".safetensors"):
                    raise ValueError("Only .safetensors LoRA files are accepted.")
                dl = fobj.get("downloadUrl") or version_obj.get("downloadUrl")
                if not dl:
                    raise ValueError("CivitAI returned no download URL.")
                dest = os.path.join(lora_dir, base)
                _curl_download(dl, dest, tok, base)
                ok, err = _validate_safetensors_file(dest)
                if not ok:
                    try: os.remove(dest)
                    except Exception: pass
                    raise RuntimeError(f"Downloaded LoRA failed safetensors validation: {err}")
                installed_name = base
            else:
                raise ValueError("Only huggingface.co and civitai.com/civitai.red LoRA URLs are accepted.")

            folder_paths.cache_helper.clear()
            _remember_lora_source(installed_name, friendly_source_url or source_url)
            log(f"  ✓ user LoRA installed: {installed_name}")
            return jsonify(ok=True, file=installed_name, source_url=(friendly_source_url or source_url))
        except Exception as e:
            log(f"  ⚠ user LoRA install failed: {e}")
            return jsonify(error=str(e)), 400

    @app.get("/api/queue")
    def api_queue():
        return jsonify(_queue_public_state())

    @app.post("/api/queue/clear")
    def api_queue_clear():
        with QUEUE_CV:
            ids = list(JOB_QUEUE)
            JOB_QUEUE.clear()
            for qid in ids:
                if qid in JOBS and JOBS[qid].get("status") == "queued":
                    JOBS[qid].update(status="cancelled", msg="Removed from queue")
            QUEUE_CV.notify_all()
        return jsonify(ok=True, cleared=len(ids))

    @app.delete("/api/queue/<jid>")
    def api_queue_cancel(jid):
        with QUEUE_CV:
            if jid == QUEUE_ACTIVE_JOB:
                if jid in JOBS:
                    JOBS[jid]["cancel_requested"] = True
                    JOBS[jid]["msg"] = "Cancellation requested; stopping at the next safe sampler/stage boundary."
                QUEUE_CV.notify_all()
                return jsonify(ok=True, running=True, requested=True)
            try:
                JOB_QUEUE.remove(jid)
                if jid in JOBS:
                    JOBS[jid].update(status="cancelled", msg="Removed from queue")
                ok = True
            except ValueError:
                ok = False
            QUEUE_CV.notify_all()
        return jsonify(ok=ok, running=False)

    def _forget_history_drop(drag_id):
        if not drag_id:
            return
        with HISTORY_DROP_LOCK:
            try:
                HISTORY_DROP_SEEN.remove(drag_id)
            except ValueError:
                pass

    @app.post("/api/timeline/add_history")
    def api_timeline_add_history():
        body = request.get_json(silent=True) or {}
        hid = str(body.get("history_id") or "")
        drag_id = str(body.get("drag_id") or "").strip()
        if drag_id:
            with HISTORY_DROP_LOCK:
                if drag_id in HISTORY_DROP_SEEN:
                    return jsonify(ok=True, duplicate_suppressed=True, timeline=_timeline_public_state())
                HISTORY_DROP_SEEN.append(drag_id)
        try:
            insert_index = int(body.get("index")) if body.get("index") is not None else None
        except Exception:
            insert_index = None
        with HISTORY_LOCK:
            src = next((dict(h) for h in HISTORY_STATE if h.get("history_id") == hid), None)
        if not src:
            _forget_history_drop(drag_id)
            return jsonify(error="History clip not found."), 404
        if not src.get("file") or not os.path.exists(os.path.join(OUT, src.get("file"))):
            _forget_history_drop(drag_id)
            return jsonify(error="The history MP4 is no longer available in this runtime."), 404
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            segs = list(seq.get("segments") or [])
            if segs and (int(src.get("width") or 0) != int(segs[0].get("width") or 0) or int(src.get("height") or 0) != int(segs[0].get("height") or 0)):
                _forget_history_drop(drag_id)
                return jsonify(error="That history clip has a different canvas size. Use a matching sequence or start a new sequence."), 400
            src.pop("history_id", None); src.pop("created", None); src.pop("status", None)
            src["segment_id"] = uuid.uuid4().hex[:12]
            src["continued"] = False
            src["continued_from_job"] = None
            src["params"] = dict(src.get("params") or {})
            idx = len(segs) if insert_index is None else max(0, min(len(segs), insert_index))
            segs.insert(idx, src)
            seq["segments"] = segs
            seq["updated"] = time.time()
            seq_id = seq.get("id")
        ok, master, note = _restitch_sequence(seq_id, "history_drop")
        return jsonify(ok=ok, master_file=master, note=note, timeline=_timeline_public_state())

    @app.delete("/api/timeline/segment/<segment_id>")
    def api_timeline_delete_segment(segment_id):
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            before = len(seq.get("segments") or [])
            seq["segments"] = [s for s in (seq.get("segments") or []) if (s.get("segment_id") or s.get("job")) != segment_id]
            if len(seq["segments"]) == before:
                return jsonify(error="Timeline clip not found."), 404
            seq_id = seq.get("id")
        ok, master, note = _restitch_sequence(seq_id, "delete_clip")
        return jsonify(ok=ok, master_file=master, note=note, timeline=_timeline_public_state())

    @app.post("/api/timeline/reorder")
    def api_timeline_reorder():
        body = request.get_json(silent=True) or {}
        order = [str(x) for x in (body.get("order") or [])]
        with TIMELINE_LOCK:
            seq = _active_sequence_unlocked(create=True)
            segs = list(seq.get("segments") or [])
            by_id = {(s.get("segment_id") or s.get("job")): s for s in segs}
            if set(order) != set(by_id.keys()) or len(order) != len(segs):
                return jsonify(error="Timeline changed while reordering. Refresh and try again."), 409
            seq["segments"] = [by_id[sid] for sid in order]
            seq_id = seq.get("id")
        ok, master, note = _restitch_sequence(seq_id, "reorder_clips")
        return jsonify(ok=ok, master_file=master, note=note, timeline=_timeline_public_state())

    @app.get("/api/stage_state")
    def api_stage_state():
        with STAGE_LOCK:
            s = dict(STAGE_STATE)
        last_path = s.get("last_frame_path")
        first_path = s.get("first_frame_path")
        ok = bool(last_path and os.path.exists(last_path))
        return jsonify(ok=ok,
                       job=s.get("job"),
                       updated=s.get("updated"),
                       video_file=s.get("video_file"),
                       first_frame_file=(os.path.basename(first_path) if first_path else None),
                       last_frame_file=(os.path.basename(last_path) if last_path else None))

    @app.get("/api/job/<jid>")
    def api_job(jid):
        j = dict(JOBS.get(jid, {"status":"error","msg":"unknown job"}))
        if j.get("status") in ("running","queued"):
            now = time.time()
            j["el"] = int(now-j["t0"])
            if j.get("status") == "running":
                stage = j.get("stage") or PROG.get("stage","running")
                cur = j.get("cur", PROG.get("cur",0))
                total = j.get("total", PROG.get("total",0))
                j["stage"] = stage
                j["cur"] = cur
                j["total"] = total
                expected_steps = max(0, int(j.get("expected_sample_steps") or 0))
                sample_total = int(total or expected_steps or 0) if stage == "sampling" else 0
                if stage == "sampling" and sample_total > 0:
                    sample_step = min(sample_total, max(1, int(cur or 0) + (1 if int(cur or 0) < sample_total else 0)))
                else:
                    sample_step = 0
                sampling_label = str(j.get("sampling_label") or "").strip()
                j["stage_label"] = (
                    f"{sampling_label} · sampling" if stage == "sampling" and sampling_label
                    else stage
                )
                j["sample_step"] = sample_step
                j["sample_steps"] = sample_total
                j["pipeline_pct"] = round(_pipeline_percent(stage,cur,sample_total or total),2)
                j["stage_elapsed"] = int(max(0, now-float(j.get("stage_started") or j["t0"])))
            else:
                j["stage"]="queued"; j["cur"]=0; j["total"]=0; j["pipeline_pct"]=0.0
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
    <title>MissingLink MiniMax Studio · Fast</title>
    <link rel="icon" href="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2">
    <style>
    *{box-sizing:border-box}
    :root{--bg:#09090b;--panel:#101013;--panel2:#151519;--line:#25252b;--muted:#777982;--text:#ededf0;--accent:#E8A917}
    html,body{height:100%}
    body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
    .wrap{height:100vh;display:grid;grid-template-columns:minmax(350px,390px) minmax(0,1fr)}
    .side{border-right:1px solid var(--line);padding:16px;overflow:auto;background:#0c0c0e}
    .main{padding:8px;overflow:hidden;display:grid;grid-template-rows:36px minmax(0,1fr) 138px;gap:6px;min-width:0;min-height:0;height:100vh}.topdock{display:grid;grid-template-columns:minmax(220px,1fr) auto;align-items:center;gap:10px;min-width:0;height:36px;padding:0 8px;border:1px solid var(--line);border-radius:8px;background:#0e0e11}
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
    .hiddenfile{display:none!important}
    .g2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
    .hint{font-size:9.5px;color:#686a73;margin-top:5px;line-height:1.4}#meta{margin:0;padding:3px 2px 0;font-size:7.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}#meta:empty{display:none}.hint b{color:#a6a7ae}
    .switchrow{display:flex;gap:8px;align-items:center;margin-top:8px}.switchrow input{width:auto;accent-color:var(--accent)}.switchrow label{margin:0;font-size:10.5px;color:#aaa}.sliderline{display:grid;grid-template-columns:minmax(0,1fr) 72px;gap:8px;align-items:center}.sliderline input[type=range]{padding:0;height:24px;accent-color:var(--accent);border:0;background:transparent}.slidervalue{font:10px ui-monospace,Menlo,monospace;color:#d7d8de;text-align:right}.loranumber{width:72px!important;height:29px!important;padding:4px 6px!important;border:1px solid #303139!important;border-radius:6px!important;background:#151519!important;color:#e7e8ec!important;text-align:right!important;-moz-appearance:textfield!important;appearance:textfield!important}.loranumber::-webkit-outer-spin-button,.loranumber::-webkit-inner-spin-button{-webkit-appearance:none!important;margin:0!important}.loranumber:focus{border-color:var(--accent)!important}.loranumber:disabled{opacity:.5;cursor:not-allowed}.lorarack{display:grid;gap:7px;margin-top:8px}.lorarow{border:1px solid #292a30;border-radius:8px;background:#101115;padding:8px}.lorarowhead{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px}.lorarowtitle{font-size:8.5px;letter-spacing:.7px;color:#bfc1c9;font-weight:800;text-transform:uppercase}.loratitlelink{color:inherit;text-decoration:none}.loratitlelink:hover{color:#8ab4ff;text-decoration:underline}.lorarowmeta{font-size:7.5px;color:#696c76}.lorarowgrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,.7fr);gap:8px;align-items:center}.lorarowgrid select{height:32px;padding:5px 7px}.loraslot{display:block}.lorarow.incompatible{opacity:.48}.lorarow.incompatible input{cursor:not-allowed}.lorarow.incompatible .lorarowtitle{color:#70727a}.lorarow.incompatible .lorarowmeta{color:#555861}.loratoggle{width:38px!important;height:21px!important;padding:0!important;margin:0!important;border-radius:999px!important;background:#25262c!important;color:#8b8e97!important;border:1px solid #3a3b43!important;font-size:7px!important;line-height:19px!important}.loratoggle.on{background:var(--accent)!important;color:#111!important;border-color:var(--accent)!important}.loratoggle:disabled{opacity:.42!important}.lorarow select option:disabled{color:#595c65}.loratools{display:grid;grid-template-columns:1fr 1fr 72px 42px;gap:7px;margin-top:8px}.loratools button{margin:0;padding:8px 7px;background:#29292f;color:#ccc;font-size:8px}.motionline{display:grid;grid-template-columns:minmax(0,1fr) 54px;gap:8px;align-items:center}.motionline input[type=range]{padding:0;height:26px;accent-color:var(--accent);border:0;background:transparent}.modetabs{display:grid;grid-template-columns:1fr 1fr;gap:8px}.modetab{margin:0;background:#202126;color:#c8c9cf;border:1px solid #303139}.modetab.active{background:var(--accent);border-color:var(--accent);color:#111}.modetab:disabled{background:#17181c;color:#5a5d66;border-color:#25262b;cursor:not-allowed}.modepanel{display:none}.modepanel.active{display:block}.refimageslots{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:2px}.refslot{height:96px}.reffilelist{display:grid;gap:8px;margin-top:10px}.reffilerow{border:1px solid #2a2b31;border-radius:8px;background:#101116;padding:8px}.reffilerowhead{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px}.reffilerowtitle{font-size:8px;color:#cfd0d5;font-weight:800;letter-spacing:.7px;text-transform:uppercase}.reffilename{flex:1;min-width:0;font-size:8px;color:#8b8d96;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.reffileactions{display:flex;gap:6px;align-items:center}.reffileactions button{width:auto;margin:0;padding:6px 8px;font-size:7px}.refclear{background:#221617;color:#d99595;border:1px solid #573032}.refclear:disabled{background:#17181c;color:#53565f;border-color:#25262b}.refmodehint{margin-top:8px}.refslot .slotbadge{font-size:7px}
    button{border:0;border-radius:7px;background:var(--accent);color:#111;padding:10px 11px;font:inherit;font-weight:800;cursor:pointer}
    button:disabled{background:#29292f;color:#666;cursor:not-allowed}.inlinebtn{background:#29292f;color:#ccc;padding:8px 9px;width:100%;margin-top:8px;font-size:10.5px}.inlinebtn.active{background:var(--accent);color:#111}
    .installed{color:#7cc38c}.missing{color:#d6a56d}
    .preset3{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}
    .hfmodelbox{margin-top:9px;padding:9px;border:1px solid #2b2c32;border-radius:8px;background:#0d0e11;display:none}
    .hfmodelbox.show{display:block}.hfmodelgrid{display:grid;grid-template-columns:minmax(0,1fr) 92px;gap:7px}.hfmodelactions{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:7px}.hfmodelactions button{margin:0}.hfprogress{height:6px;background:#24252a;border-radius:999px;overflow:hidden;margin-top:9px}.hfprogress i{display:block;height:100%;width:0;background:var(--accent);transition:width .18s}.hfprogresstext{font-size:8px;color:#8e9099;margin-top:5px;min-height:12px}.hfmodelbox select,.hfmodelbox input{font-size:10px}
    .modelcardtitle{display:flex;align-items:center;justify-content:space-between;gap:10px}.modelcardtitle>span{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.adultmodebtn{position:relative;width:48px!important;height:23px!important;flex:0 0 48px;margin:0!important;padding:0 17px 0 7px!important;background:#17181c!important;color:#8b8e97!important;border:1px solid #303139!important;border-radius:999px!important;font-size:7px!important;letter-spacing:.35px!important;text-align:left!important}.adultmodebtn::after{content:'';position:absolute;right:6px;top:50%;width:7px;height:7px;border-radius:50%;background:#555862;transform:translateY(-50%);box-shadow:0 0 0 1px #16171a}.adultmodebtn:hover{border-color:#52545d!important;color:#c7c9cf!important}.adultmodebtn.on{background:#211d10!important;color:#edc44a!important;border-color:#66551d!important}.adultmodebtn.on::after{background:var(--accent);box-shadow:0 0 8px #e8a91766}.adultmodebtn:disabled{opacity:.45!important}
    #go{width:100%;font-size:13px;margin-top:4px;padding:12px}
    .autopromptrow{display:grid;grid-template-columns:minmax(0,1fr) 44px;gap:8px;margin-top:8px}.approfilerow{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:6px;align-items:center}.approfilerow button{width:auto;margin:0;padding:8px 9px;font-size:7.5px}.autopromptrow button{margin:0}.autopromptrow .gear{background:#29292f;color:#ddd;font-size:17px;padding:8px}.autopromptrow .gear:hover{color:var(--accent)}#auto_prompt_btn.working{background:#29292f;color:#aaa}.apmodal{display:none;position:fixed;inset:0;z-index:1600;background:rgba(0,0,0,.56);align-items:flex-start;justify-content:center;padding:72px 16px 16px}.apmodal.show{display:flex}.apdialog{width:min(520px,calc(100vw - 28px));background:#111114;border:1px solid #34353d;border-radius:12px;padding:13px;box-shadow:0 18px 60px rgba(0,0,0,.55)}.aphead{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}.aphead b{font-size:11px;letter-spacing:1px;color:#b8bac2}.apclose{width:auto;background:#29292f;color:#bbb;padding:6px 9px}.apdialog textarea{min-height:120px}.apactions{display:flex;justify-content:flex-end;gap:8px;margin-top:10px}.apactions button{width:auto}.apstatus{font-size:9.5px;color:#777982;margin-top:5px}.apstatus.ok{color:#65d78d}.apstatus.err{color:#ff8181}#ap_custom_wrap{display:none}
    .uimodal{display:none;position:fixed;inset:0;z-index:1800;background:rgba(0,0,0,.68);align-items:center;justify-content:center;padding:18px}.uimodal.show{display:flex}.uidialog{width:min(430px,calc(100vw - 30px));background:#111114;border:1px solid #34353d;border-radius:10px;box-shadow:0 20px 70px #000c;overflow:hidden}.uihead{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid #282930}.uihead b{font-size:9.5px;letter-spacing:1.2px;color:#b9bac1;text-transform:uppercase}.uiclose{width:28px;height:28px;padding:0;background:#23242a;color:#aaa;border:1px solid #34353d;border-radius:6px}.uibody{padding:12px}.uimessage{color:#b6b8c0;font-size:10px;line-height:1.55;white-space:pre-wrap}.uiinput{margin-top:10px;height:36px}.uiactions{display:flex;justify-content:flex-end;gap:6px;padding:0 12px 12px}.uiactions button{width:auto;min-width:78px;padding:8px 11px;font-size:9px}.uicancel{background:#25262c;color:#c8c9ce}.uiconfirm.danger{background:#4a1e20;color:#ffb0b0;border:1px solid #7a3034}.uiconfirm.neutral{background:#29292f;color:#ddd}.adultlegal{font-size:9.5px;color:#aeb0b8;line-height:1.5}.adultcheck{display:flex;align-items:flex-start;gap:8px;margin:9px 0;color:#c4c6cd;font-size:9.5px;line-height:1.45}.adultcheck input{width:auto;flex:0 0 auto;margin-top:2px;accent-color:var(--accent)}.adultnotice{margin-top:10px;padding:8px;border:1px solid #37321d;background:#17150d;color:#aaa17d;border-radius:7px;font-size:8.5px;line-height:1.45}
    .imageslots{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px}.imageslot{position:relative;height:112px;border:1px dashed #34353d;border-radius:8px;background:#09090b;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:pointer;outline:none}.imageslot:hover,.imageslot:focus{border-color:#666974}.imageslot.has-image{border-style:solid}.imageslot img{display:none;width:100%;height:100%;object-fit:cover;background:#050506}.imageslot.has-image img{display:block}.slotempty{text-align:center;color:#656872;font-size:9px;letter-spacing:.8px;pointer-events:none}.slotempty b{display:block;color:#9698a1;font-size:10px;margin-bottom:3px}.imageslot.has-image .slotempty{display:none}.slottrash{display:none;position:absolute;right:5px;top:5px;z-index:4;width:24px;height:24px;padding:0;border-radius:6px;background:#18181dcc;color:#d6d7dc;border:1px solid #42434b;font-size:11px;line-height:1}.imageslot.has-image .slottrash{display:block}.slottrash:hover{background:#351719;color:#ff7777;border-color:#763336}.slotbadge{position:absolute;left:5px;bottom:5px;z-index:3;padding:3px 5px;border-radius:4px;background:#08090bcc;color:#ddd;font-size:7.5px;letter-spacing:.5px;pointer-events:none}.imageslot:not(.has-image) .slotbadge{display:none}.imgmodal{display:none;position:fixed;inset:0;z-index:1700;background:rgba(0,0,0,.88);padding:26px;align-items:center;justify-content:center}.imgmodal.show{display:flex}.imgmodal img{max-width:calc(100vw - 52px);max-height:calc(100vh - 52px);object-fit:contain;border-radius:8px;box-shadow:0 20px 80px #000}.imgmodalclose{position:absolute;right:20px;top:18px;width:38px;height:38px;padding:0;border-radius:50%;background:#202126;color:#ddd;border:1px solid #454750;font-size:16px}
    #status{position:relative;min-width:0;height:100%;display:flex;align-items:center;padding-bottom:3px}.row{display:flex;align-items:center;gap:7px;font-size:8.5px;min-width:0;width:100%}.row #stxt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#c1c2c8}.dot{width:6px;height:6px;flex:0 0 6px;border-radius:50%;background:#5fd68a}.dot.live{background:var(--accent);animation:p 1.3s infinite}.dot.err{background:#ff6b6b}@keyframes p{50%{opacity:.25}}.bar{position:absolute;left:0;right:0;bottom:2px;height:2px;background:#1d1e22;border-radius:2px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .25s}
    .gpuoverlay{position:absolute;right:10px;top:10px;z-index:5;width:132px;background:#0c0d11d9;border:1px solid #303139;border-radius:8px;backdrop-filter:blur(8px);box-shadow:0 8px 24px #0007;overflow:hidden}.gpuoverlayhead{height:27px;display:flex;align-items:center;justify-content:space-between;padding:0 6px 0 8px;border-bottom:1px solid #282930}.gpuoverlaytitle{font-size:6.8px;letter-spacing:.9px;color:#858791;font-weight:800}.gpuoverlaytoggle{width:22px!important;height:20px!important;padding:0!important;margin:0!important;border-radius:5px!important;background:#1c1d22!important;color:#aeb0b8!important;border:1px solid #303139!important;font-size:10px!important}.gpugrid{display:flex;flex-direction:column;align-items:stretch;gap:0;min-width:0}.gpucard{height:24px;display:grid;grid-template-columns:34px minmax(0,1fr);align-items:center;gap:6px;padding:0 8px;border-top:1px solid #202126}.gpucard:first-child{border-top:0}.gpuk{font-size:5.8px;color:#63656e;letter-spacing:.45px;text-transform:uppercase}.gpuv{font-size:8px;white-space:nowrap;color:#d1d2d7;text-align:right;overflow:hidden;text-overflow:ellipsis}.gpuv.busy{color:var(--accent)}.gpuv.hot{color:#ffb36b}.gpuoverlay.minimized{width:auto}.gpuoverlay.minimized .gpugrid{display:none}.gpuoverlay.minimized .gpuoverlayhead{border-bottom:0;padding-left:7px}.gpuoverlay.minimized .gpuoverlaytitle{font-size:6.5px}
    .chrometools{display:flex;align-items:center;gap:4px}.chrometools button{width:auto;height:25px;margin:0;padding:0 8px;background:#17181c;color:#bfc0c6;border:1px solid #2d2e34;border-radius:6px;font-size:7px;font-weight:800}.chrometools button:hover{border-color:#50525b;color:#fff}.chrometools .count{color:var(--accent);margin-left:3px}
    .stagearea{position:relative;min-height:0;min-width:0;height:100%;overflow:hidden}.stagearea #vwrap{height:100%;min-height:0}#vwrap{border:1px solid var(--line);border-radius:8px;background:#070708;display:flex;align-items:center;justify-content:center;overflow:hidden}#empty{color:#3a3c43;font-size:8.5px}video{width:100%;height:100%;object-fit:contain;background:#000}.stagecontrols{position:absolute;left:10px;top:10px;z-index:6;display:none;align-items:center;gap:6px}.stageclear,.stagegrab{position:static;width:auto!important;height:27px!important;padding:0 8px!important;background:#111217d9!important;color:#aeb0b8!important;border:1px solid #303139!important;border-radius:6px!important;font-size:7px!important;opacity:.82;backdrop-filter:blur(7px)}.stageclear:hover,.stagegrab:hover{opacity:1;color:#fff!important}.stagegrab{background:#221f11df!important;border-color:#62531f!important;color:#e8d071!important;font-weight:800!important}.stagegrab.working{pointer-events:none;opacity:.55}
    .tclip{position:relative}.tclip.dragging{opacity:.35}.tclip.drop-before{box-shadow:inset 3px 0 0 var(--accent)}.tclip.drop-after{box-shadow:inset -3px 0 0 var(--accent)}.ttools{position:absolute;right:5px;top:5px;display:flex;gap:4px;z-index:3}.ttools button{width:24px;height:24px;padding:0;border-radius:50%;background:#241718;color:#ff6b6b;border:1px solid #6b2c2c;font-size:11px}.dragbadge{position:absolute;left:5px;top:5px;background:#111c;color:#c9c9cf;border:1px solid #363840;border-radius:5px;padding:3px 5px;font-size:7.5px;z-index:3}.timeline.drop-target{outline:1px dashed var(--accent);outline-offset:3px}
    .floatpanel{display:none;position:fixed;z-index:1190;width:min(310px,calc(100vw - 20px));background:#0d0d10;border:1px solid #2a2b31;border-radius:9px;box-shadow:0 14px 40px #000b;overflow:hidden}.floatpanel.history{left:14px;top:70px}.floatpanel.queue{right:14px;bottom:14px;width:min(330px,calc(100vw - 20px))}.floatpanel.minimized .floatbody{display:none}.floathead{display:flex;align-items:center;gap:6px;padding:6px 7px;border-bottom:1px solid #24252b;cursor:move;user-select:none;min-height:31px}.floatpanel.minimized .floathead{border-bottom:0}.floatgrip{color:#80828c;letter-spacing:1px;font-size:8px}.floattitle{font-size:8.5px;color:#989aa4;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;flex:1}.floatcount{color:var(--accent);font-size:8px}.floatactions{display:flex;gap:4px}.floatactions button{width:auto;margin:0;background:#25262c;color:#ddd;padding:4px 6px;font-size:7px}.floatbody{padding:5px;max-height:190px;overflow:auto}.historyitem,.queueitem{display:grid;grid-template-columns:42px minmax(0,1fr) auto;gap:6px;align-items:center;border:1px solid #292a30;border-radius:6px;background:#151518;padding:4px;margin-bottom:4px}.historyitem{cursor:grab}.historyitem:active{cursor:grabbing}.histthumb,.qthumb{width:42px;height:36px;border-radius:4px;background:#090a0c;overflow:hidden}.histthumb img,.qthumb img{width:100%;height:100%;object-fit:cover}.histmain,.qmain{min-width:0}.histstatus,.qstatus{font-size:8.5px;color:#dedfe4;font-weight:800}.histsub,.qsub{font-size:7px;color:#777983;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.historyitem .trash{width:24px;height:24px;border-radius:50%;padding:0;background:#281718;color:#ff6b6b;border:1px solid #782f31;font-size:9px}.historyitem .recallprompt,.tclip .recallprompt{width:auto;height:24px;border-radius:5px;padding:0 7px;background:#24252a;color:#c9cbd2;border:1px solid #363840;font-size:6.5px;font-weight:800;letter-spacing:.4px}.queueitem .cancel{width:auto;height:25px;border-radius:5px;padding:0 8px;background:#281718;color:#ff8585;border:1px solid #782f31;font-size:7px;font-weight:800;letter-spacing:.45px}.queueitem .stoprun{width:auto;height:25px;border-radius:5px;padding:0 8px;background:#3a1818;color:#ff9a9a;border:1px solid #8b3737;font-size:7px;font-weight:800;letter-spacing:.45px}.queuehealth{font-size:6px;color:#8f939d;border:1px solid #30323a;border-radius:4px;padding:2px 4px;margin-left:4px}.queuehealth.busy{color:#f1c34d;border-color:#725b1d}.queuehealth.warn{color:#ff9a9a;border-color:#7d3333;background:#2a1515}.queueprogress{height:3px;border-radius:999px;background:#24252a;margin-top:3px;overflow:hidden}.queueprogress i{display:block;height:100%;background:var(--accent);width:0}.floatempty{padding:15px 8px;text-align:center;color:#5f616a;font-size:8px}.addhist{font-size:6.5px;color:#9a9ca4;margin-top:2px}.queuebadge{color:var(--accent);font-weight:800}
    .timelinebox{position:relative;border:1px solid var(--line);border-radius:8px;background:#0e0e11;padding:6px 8px;min-width:0;height:138px;display:grid;grid-template-rows:27px minmax(0,1fr);overflow:hidden}.timelinehead{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0;min-height:0}.timelineheading{display:flex;align-items:baseline;gap:8px;min-width:0}.timelinehead .title{font-size:9px;font-weight:800;letter-spacing:1px;color:#a0a1a9;text-transform:uppercase;white-space:nowrap}.timelinecontext{font-size:7.5px;color:#62646d;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.timelineactions{display:flex;align-items:center;gap:4px;flex-wrap:wrap;justify-content:flex-end}.timelineactions button,.timelineactions a{height:24px;width:auto;margin:0;border-radius:6px;background:#18191d;color:#bfc0c6;border:1px solid #2b2c31;padding:0 7px;font-size:7px;line-height:25px;text-decoration:none;font-weight:800;white-space:nowrap}.timelineactions button:hover,.timelineactions a:hover{border-color:#555862;color:#fff}.timelineactions button.primary{background:var(--accent);border-color:var(--accent);color:#111}.timelineactions button.compile{background:#24200f;border-color:#66551d;color:#e8ce6b}.timelineactions button.compile:not(:disabled):hover{border-color:var(--accent);color:var(--accent)}.timelineactions button.danger{background:#211516;border-color:#522b2d;color:#d98989}.timelineactions button.danger:not(:disabled):hover{border-color:#8a3e42;color:#ff9a9f}.timelineactions button:disabled,.timelineactions a.disabled{background:#17181c;color:#555761;border-color:#24252a;pointer-events:none}.sequenceselect{display:none;height:22px;max-width:150px;margin:0;padding:2px 24px 2px 6px;border:1px solid #292a2f;border-radius:5px;background:#141519;color:#bfc0c6;font-size:7.2px;line-height:18px}.sequenceselect.show{display:block}.timeline{display:flex;align-items:stretch;gap:6px;overflow-x:auto;min-height:0;height:100%;padding:2px 0}.tclip{min-width:136px;max-width:136px;border:1px solid #2a2b31;border-radius:7px;background:#0b0b0d;overflow:hidden;cursor:pointer;height:100%}.tclip:hover{border-color:#484a53}.tclip:last-child{border-color:#725d25}.tthumb{height:48px;background:#050506;display:flex;align-items:center;justify-content:center}.tthumb img{width:100%;height:100%;object-fit:cover}.tinfo{padding:5px 6px;font-size:7.5px;color:#777982;line-height:1.28}.tinfo b{color:#c7c8cd}.timelineempty{display:flex;align-items:center;justify-content:center;min-width:100%;height:100%;color:#555761;font-size:8.5px}.timelinefoot{display:none}.projectpopover{display:none;position:absolute;right:8px;top:42px;z-index:40;width:min(360px,calc(100% - 16px));padding:9px;background:#0d0d10;border:1px solid #303139;border-radius:8px;box-shadow:0 14px 36px #000c}.projectpopover.show{display:block}.projectpophead{display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;color:#b6b8c0;font-size:9px;letter-spacing:.8px;text-transform:uppercase}.projectpophead button{width:24px;height:24px;padding:0;background:#202126;color:#bbb;border:1px solid #303139}.projectpopover label{font-size:7px;margin:6px 0 3px}.projectline{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:5px;margin-bottom:5px}.projectline input,.projectline select{height:29px;padding:5px 7px;font-size:9px}.projectline button{height:29px;width:auto;padding:0 8px;background:#202126;color:#c7c8cd;border:1px solid #303139;font-size:7.5px}.projectline button:hover{border-color:#555862;color:#fff}.projectline button.disabled{color:#4e5058;border-color:#25262b;pointer-events:none}.projectdanger{width:100%;height:29px;margin-top:4px;background:#1d1516;color:#d98989;border:1px solid #522b2d;font-size:7.5px}.timeline sub{font-size:7px}
    *{scrollbar-width:none}*::-webkit-scrollbar{display:none;width:0;height:0}
    .consolebox{display:none;position:fixed;right:16px;top:72px;z-index:1220;width:min(720px,calc(100vw - 32px));border:1px solid #303139;border-radius:9px;background:#060607;overflow:hidden;box-shadow:0 18px 60px #000c}.consolehead{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:7px 9px;border-bottom:1px solid var(--line);color:#81838c;font-size:9.5px}.consoleactions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.consolehead button{width:auto;margin:0;background:#29292f;color:#bbb;padding:5px 9px;font-size:9px;border-radius:5px}.consolebox pre{margin:0;padding:8px 9px;height:240px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#c8c9ce;font:9.5px/1.4 ui-monospace,Menlo,monospace}.consolebox.collapsed pre{display:none}.consolebox.collapsed .consolehead{border-bottom:0}
    #err{display:none;white-space:pre-wrap;color:#ff8a8a;font-size:10px;max-height:180px;overflow:auto;border:1px solid #3a2020;background:#160e0e;padding:10px;border-radius:8px}
    .footerlink{display:block;text-align:center;color:#62646d;text-decoration:none;font-size:9px;margin:6px 0 2px}.footerlink:hover{color:var(--accent)}
    @media(max-width:1100px){.topdock{grid-template-columns:minmax(160px,1fr) auto}.main{grid-template-rows:36px minmax(0,1fr) 138px}.gpuoverlay{right:7px;top:7px}}
    @media(max-width:850px){body{overflow:auto}.wrap{height:auto;grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line);max-height:none}.main{height:auto;overflow:visible;grid-template-rows:auto 420px 152px;min-height:0}.topdock{grid-template-columns:1fr}.floatpanel.history{left:8px;top:48px}.floatpanel.queue{right:8px;bottom:8px}.timelinehead{align-items:flex-start}.timelineactions{justify-content:flex-start}.projectpopover{left:8px;right:8px;width:auto}.projectline{grid-template-columns:1fr auto}.imageslot{height:126px}.refimageslots{grid-template-columns:1fr 1fr}.consolebox{left:8px;right:8px;top:56px;width:auto}}
    </style></head><body><div class=wrap>
    <div class=side>
    <a class=brand href="https://missinglink.build" target="_blank" rel="noopener"><img id=logo src="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2" alt="" onerror="this.style.display='none'"><span><span class=ml>MISSINGLINK</span> <span class=st>MINIMAX H3</span></span></a>

    <div class=card><div class="cardtitle modelcardtitle"><span>Model Mode · <span id=arch_label>AUTO GPU</span></span><button id=adult_mode_btn type=button hidden aria-hidden=true tabindex=-1></button></div><div class=cardbody>
    <input id=input_mode type=hidden value=fl2va>
    <div class=modetabs><button id=tab_fl2va class="modetab active" type=button>CURRENT · KEYFRAMES</button><button id=tab_ref2va class=modetab type=button>REF2VA · REFERENCES</button></div>

    <label>Base model</label>
    <select id=model_profile_select>
      <option value=stock_quality>BUILT-IN MINIMAX H3</option>
    </select>
    <button id=hf_model_toggle class=inlinebtn type=button>+ HF BASE MODEL</button>
    <div id=hf_model_box class=hfmodelbox>
      <label>Hugging Face repo</label><input id=hf_model_repo type=text placeholder="owner/repository" autocomplete=off>
      <div class=hfmodelgrid><div><label>Revision</label><input id=hf_model_revision type=text value="main" autocomplete=off></div><div><label>Mode</label><select id=hf_model_mode><option value=both selected>both</option><option value=fl2va>current</option><option value=ref2va>ref2va</option></select></div></div>
      <label>Model file</label><select id=hf_model_file disabled><option value="">CHECK REPO first</option></select>
      <div class=hfmodelactions><button id=hf_model_inspect class=inlinebtn type=button>CHECK REPO</button><button id=hf_model_install type=button disabled>DOWNLOAD + USE</button></div>
      <div class=hfprogress><i id=hf_model_progress></i></div><div id=hf_model_progress_text class=hfprogresstext>Paste an HF repo, inspect it, then choose the checkpoint file.</div>
    </div>

    <div class=hint id=mode_hint>The tabs select the input / conditioning workflow. The Base model menu selects the checkpoint.</div>
    <div class=hint id=model_profile_hint>Stock H3 · matching files ready.</div>
    </div></div>

    <div class=card><div class=cardtitle>Prompt</div><div class=cardbody>
    <textarea id=prompt placeholder="Describe the shot, motion, camera, environment, soundscape and music..."></textarea>
    <div class=autopromptrow><button id=auto_prompt_btn type=button>✦ AUTO PROMPT</button><button id=auto_prompt_settings class=gear type=button title="Auto Prompt settings">⚙</button></div>
    <div class=hint id=auto_prompt_hint><b>RAW LOCAL is the default.</b> GENERATE sends the prompt exactly as shown directly to local H3. ✦ AUTO PROMPT is optional and runs only when you click it.</div>
    </div></div>

    <div class=card><div class=cardtitle>Input + Output</div><div class=cardbody>
    <input class=hiddenfile type=file id=first_frame accept="image/*"><input class=hiddenfile type=file id=last_frame accept="image/*">
    <input class=hiddenfile type=file id=ref_image_1 accept="image/*"><input class=hiddenfile type=file id=ref_image_2 accept="image/*"><input class=hiddenfile type=file id=ref_image_3 accept="image/*"><input class=hiddenfile type=file id=ref_image_4 accept="image/*"><input class=hiddenfile type=file id=ref_image_5 accept="image/*"><input class=hiddenfile type=file id=ref_image_6 accept="image/*"><input class=hiddenfile type=file id=ref_image_7 accept="image/*"><input class=hiddenfile type=file id=ref_image_8 accept="image/*"><input class=hiddenfile type=file id=ref_image_9 accept="image/*">
    <input class=hiddenfile type=file id=ref_video_1 accept="video/*"><input class=hiddenfile type=file id=ref_video_2 accept="video/*"><input class=hiddenfile type=file id=ref_video_3 accept="video/*">
    <input class=hiddenfile type=file id=ref_audio_1 accept="audio/*"><input class=hiddenfile type=file id=ref_audio_2 accept="audio/*"><input class=hiddenfile type=file id=ref_audio_3 accept="audio/*">

    <div id=mode_panel_fl2va class="modepanel active">
      <div class=imageslots>
        <div id=first_slot class=imageslot role=button tabindex=0 aria-label="First frame image slot"><img id=first_preview alt="First frame"><div class=slotempty><b>FIRST FRAME</b>click to upload</div><span class=slotbadge>FIRST</span><button id=first_trash class=slottrash type=button title="Clear first frame">⌫</button></div>
        <div id=last_slot class=imageslot role=button tabindex=0 aria-label="Last frame image slot"><img id=last_preview alt="Last frame"><div class=slotempty><b>LAST FRAME</b>click to upload</div><span class=slotbadge>LAST</span><button id=last_trash class=slottrash type=button title="Clear last frame">⌫</button></div>
      </div>
      <div class=hint>CURRENT / KEYFRAMES: zero, one, or two keyframe anchors using the model selected above.</div>
      <div class=switchrow><input id=auto_aspect type=checkbox checked><label for=auto_aspect>fit canvas to first-frame aspect</label></div>
      <div class=g2><div><label>Short edge</label><input id=short_edge type=number value=768 step=32 min=256></div><div><label>Image fit</label><select id=image_fit><option value=cover selected>cover / crop</option><option value=contain>contain</option><option value=stretch>stretch</option></select></div></div>
    </div>

    <div id=mode_panel_ref2va class=modepanel>
      <label>Reference images</label>
      <div class=refimageslots>
        <div id=ref_image_slot_1 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 1"><img id=ref_image_preview_1 alt="Reference image 1"><div class=slotempty><b>REF IMAGE 1</b>click to upload</div><span class=slotbadge>P1</span><button id=ref_image_trash_1 class=slottrash type=button title="Clear reference image 1">⌫</button></div>
        <div id=ref_image_slot_2 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 2"><img id=ref_image_preview_2 alt="Reference image 2"><div class=slotempty><b>REF IMAGE 2</b>click to upload</div><span class=slotbadge>P2</span><button id=ref_image_trash_2 class=slottrash type=button title="Clear reference image 2">⌫</button></div>
        <div id=ref_image_slot_3 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 3"><img id=ref_image_preview_3 alt="Reference image 3"><div class=slotempty><b>REF IMAGE 3</b>click to upload</div><span class=slotbadge>P3</span><button id=ref_image_trash_3 class=slottrash type=button title="Clear reference image 3">⌫</button></div>
        <div id=ref_image_slot_4 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 4"><img id=ref_image_preview_4 alt="Reference image 4"><div class=slotempty><b>REF IMAGE 4</b>click to upload</div><span class=slotbadge>P4</span><button id=ref_image_trash_4 class=slottrash type=button title="Clear reference image 4">⌫</button></div>
        <div id=ref_image_slot_5 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 5"><img id=ref_image_preview_5 alt="Reference image 5"><div class=slotempty><b>REF IMAGE 5</b>click to upload</div><span class=slotbadge>P5</span><button id=ref_image_trash_5 class=slottrash type=button title="Clear reference image 5">⌫</button></div>
        <div id=ref_image_slot_6 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 6"><img id=ref_image_preview_6 alt="Reference image 6"><div class=slotempty><b>REF IMAGE 6</b>click to upload</div><span class=slotbadge>P6</span><button id=ref_image_trash_6 class=slottrash type=button title="Clear reference image 6">⌫</button></div>
        <div id=ref_image_slot_7 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 7"><img id=ref_image_preview_7 alt="Reference image 7"><div class=slotempty><b>REF IMAGE 7</b>click to upload</div><span class=slotbadge>P7</span><button id=ref_image_trash_7 class=slottrash type=button title="Clear reference image 7">⌫</button></div>
        <div id=ref_image_slot_8 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 8"><img id=ref_image_preview_8 alt="Reference image 8"><div class=slotempty><b>REF IMAGE 8</b>click to upload</div><span class=slotbadge>P8</span><button id=ref_image_trash_8 class=slottrash type=button title="Clear reference image 8">⌫</button></div>
        <div id=ref_image_slot_9 class="imageslot refslot" role=button tabindex=0 aria-label="Reference image slot 9"><img id=ref_image_preview_9 alt="Reference image 9"><div class=slotempty><b>REF IMAGE 9</b>click to upload</div><span class=slotbadge>P9</span><button id=ref_image_trash_9 class=slottrash type=button title="Clear reference image 9">⌫</button></div>
      </div>
      <div class=g2><div><label>Ref image size</label><select id=ref_image_size><option value=match selected>match</option><option value=max>max fidelity</option></select></div><div><label>Reference budget</label><div class=hint style="margin-top:9px">Up to 9 images, 3 videos, 3 audio clips.</div></div></div>
      <div class=reffilelist>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Video 1</div><div class=reffilename id=ref_video_name_1>No video selected</div></div><div class=reffileactions><button id=ref_video_pick_1 class=inlinebtn type=button>CHOOSE</button><button id=ref_video_clear_1 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Video 2</div><div class=reffilename id=ref_video_name_2>No video selected</div></div><div class=reffileactions><button id=ref_video_pick_2 class=inlinebtn type=button>CHOOSE</button><button id=ref_video_clear_2 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Video 3</div><div class=reffilename id=ref_video_name_3>No video selected</div></div><div class=reffileactions><button id=ref_video_pick_3 class=inlinebtn type=button>CHOOSE</button><button id=ref_video_clear_3 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Audio 1</div><div class=reffilename id=ref_audio_name_1>No audio selected</div></div><div class=reffileactions><button id=ref_audio_pick_1 class=inlinebtn type=button>CHOOSE</button><button id=ref_audio_clear_1 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Audio 2</div><div class=reffilename id=ref_audio_name_2>No audio selected</div></div><div class=reffileactions><button id=ref_audio_pick_2 class=inlinebtn type=button>CHOOSE</button><button id=ref_audio_clear_2 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
        <div class=reffilerow><div class=reffilerowhead><div class=reffilerowtitle>Reference Audio 3</div><div class=reffilename id=ref_audio_name_3>No audio selected</div></div><div class=reffileactions><button id=ref_audio_pick_3 class=inlinebtn type=button>CHOOSE</button><button id=ref_audio_clear_3 class="inlinebtn refclear" type=button>⌫ CLEAR</button></div></div>
      </div>
      <div class="hint refmodehint" id=ref2va_hint>Prompt labels follow slot order as <code>&lt;Picture 1&gt;</code>… <code>&lt;Video 1&gt;</code>… <code>&lt;Audio 1&gt;</code>. If a reference video contains audio, its soundtrack is forwarded automatically to the matching Ref2VA video-audio slot.</div>
    </div>

    <div class=g2><div><label>Width</label><input id=width type=number value=768 step=32 min=32 autocomplete=off></div><div><label>Height</label><input id=height type=number value=768 step=32 min=32 autocomplete=off></div></div>
    <div class=g2><div><label>Duration</label><input id=duration type=number value=7 step=0.1 min=0.21 max=149.7 autocomplete=off></div><div><label>Motion pace</label><div class=motionline><input id=playback_speed type=range value=1 min=0.75 max=2 step=0.05><span id=motion_pace_value class=slidervalue>1.00×</span></div></div></div>
    <input id=length_mode type=hidden value=seconds><input id=frames type=hidden value=481>
    <div class=hint id=durhint></div>
    <div class=hint><b>Motion pace:</b> 1.00× keeps native H3 timing. Higher values generate more model-time and retime it back to the requested duration, reducing the “slow-motion” feel without changing the final clip length.</div>
    <input id=use_stage_last type=hidden value=0>
    <input id=timeline_action type=hidden value=new>

    <div style="margin-top:11px;padding-top:9px;border-top:1px solid #24252a">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
        <div style="font-size:8px;font-weight:800;letter-spacing:.8px;color:#a7a9b0;text-transform:uppercase">Clip continuity · optional</div>
        <label class=switchrow style="margin:0"><input id=continuity_enabled type=checkbox autocomplete=off><span style="font-size:9px;color:#aaa">USE PREVIOUS LAST FRAME</span></label>
      </div>
      <div class=switchrow><input id=continuity_keep_seed type=checkbox autocomplete=off><label for=continuity_keep_seed>reuse previous clip seed</label></div>
      <div class=hint id=continuity_hint><b>OFF:</b> GENERATE makes a fully independent clip. When enabled, the next GENERATE uses the previous timeline clip's lossless final frame as its first-frame anchor. No latent data is carried between clips.</div>
    </div>
    </div></div>

    <div class=card><div class=cardtitle>Generation Presets</div><div class=cardbody>
    <div class=preset3><button id=fastpreset class="inlinebtn active">FAST</button><button id=ultrafastpreset class=inlinebtn>ULTRA FAST</button><button id=qualitypreset class=inlinebtn>QUALITY</button></div>
    <div class=hint id=preset_hint>Preset recipe follows the model selected above.</div>
    <div class=hint id=modelhint>Model mode will be shown here after startup.</div>
    </div></div>

    <details><summary>Sampling</summary><div>
    <div class=g2><div><label>Steps</label><input id=steps type=number value=4 min=1></div><div><label>Denoise</label><input id=denoise type=number value=1 step=0.01 min=0 max=1></div></div>
    <div class=g2><div><label>Sampler</label><select id=sampler_name></select></div><div><label>Scheduler</label><select id=scheduler></select></div></div>
    <label>Seed</label><div style="display:grid;grid-template-columns:1fr 64px;gap:8px"><input id=seed type=number value=42><button id=rnd class=inlinebtn style="margin:0">RAND</button></div>
    <div class=g2><div><label>Video shift</label><input id=shift_video type=number value=6 step=.01 min=.01 max=100></div><div><label>Audio shift</label><input id=shift_audio type=number value=3 step=.01 min=.01 max=100></div></div>
    <div style="margin-top:10px">
      <div style="display:flex;align-items:end;justify-content:space-between;gap:10px">
        <label style="margin:0">Sparse attention %</label>
        <input id=sparse_percent type=number value=5 min=0 max=100 step=.5 style="width:92px;text-align:right">
      </div>
      <input id=sparse_slider type=range min=0 max=100 step=.5 value=5 style="width:100%;padding:0;margin-top:8px">
      <div class=hint id=sparsehint><b>Ultra Fast uses 5% sparse attention.</b> Fast and Quality use dense attention by default; you can still override this manually.</div>
    </div>
    </div></details>

    <details open><summary>Model + LoRA Rack</summary><div>
    <label>Transformer</label><select id=unet></select>
    <label>Weight dtype</label><select id=weight_dtype><option>default</option><option>fp8_e4m3fn</option><option>fp8_e4m3fn_fast</option><option>fp8_e5m2</option></select>

    <div id=unified_lora_rows class=lorarack></div>
    <div class=loratools style="grid-template-columns:1fr 72px 42px"><button id=install_lora type=button>+ HF / CIVITAI</button><button id=reset_loras type=button>RESET</button><button id=refresh type=button>↻</button></div>
    <div class=hint>Each compatible LoRA has one card. OFF = 0.00 · ON = 1.00 · every slider is 0.00–5.00. The numeric field accepts any finite strength and the slider pins visually to its 0–5 range. Incompatible LoRAs are greyed out.</div>
    </div></details>

    <button id=go>+ ADD GENERATION TO QUEUE</button>
    <a class=footerlink href="https://missinglink.build/studio" target="_blank" rel="noopener">missinglink.build/studio</a>
    </div>

    <div class=main>
    <div class=topdock>
      <div id=status><div class=row><span class=dot></span><span id=stxt>ready</span></div><div class=bar><i id=pb></i></div></div>
      <div class=chrometools>
        <button id=show_history type=button>HISTORY <span id=history_launch_count class=count></span></button>
        <button id=show_queue type=button>QUEUE <span id=queue_launch_count class=count></span></button>
        <button id=show_console type=button>LOG</button>
      </div>
    </div>
    <div class=stagearea>
      <div id=stage_controls class=stagecontrols>
        <button id=clear_stage class=stageclear type=button>CLEAR STAGE</button>
        <button id=stage_last_to_first class=stagegrab type=button title="Use the last visible frame of the staged clip as the FL2VA first-frame input">LAST → FIRST</button>
      </div>
      <aside id=gpu_overlay class=gpuoverlay aria-label="GPU telemetry">
        <div class=gpuoverlayhead><span class=gpuoverlaytitle>GPU STATS</span><button id=gpu_overlay_toggle class=gpuoverlaytoggle type=button title="Minimize GPU stats">−</button></div>
        <div class=gpugrid>
          <div class=gpucard><div class=gpuk>GPU</div><div class=gpuv id=gpu_util>--</div></div>
          <div class=gpucard><div class=gpuk>VRAM</div><div class=gpuv id=gpu_vram>--</div></div>
          <div class=gpucard><div class=gpuk>MEM</div><div class=gpuv id=gpu_memutil>--</div></div>
          <div class=gpucard><div class=gpuk>TEMP</div><div class=gpuv id=gpu_temp>--</div></div>
          <div class=gpucard><div class=gpuk>POWER</div><div class=gpuv id=gpu_power>--</div></div>
          <div class=gpucard><div class=gpuk>CLOCK</div><div class=gpuv id=gpu_clock>--</div></div>
        </div>
      </aside>
      <div id=vwrap><div id=empty>generated video appears here</div></div>
      <div id=meta class=hint></div>
      <div id=err></div>
    </div>
    <div class=timelinebox>
      <div class=timelinehead>
        <div class=timelineheading><div class=title>Timeline</div><div id=timeline_context class=timelinecontext>Sequence 1 · 0 clips · 0.00 s</div><select id=sequence_select class=sequenceselect aria-label="Active sequence"></select></div>
        <div class=timelineactions>
          <button id=timeline_retry>↻ RETRY</button>
          <button id=timeline_new_sequence>+ SEQUENCE</button>
          <button id=project_menu_btn>PROJECT ▾</button>
          <button id=timeline_compile class=compile type=button disabled>⧉ COMPILE</button>
          <button id=timeline_clear class=danger type=button disabled>⌫ CLEAR</button>
        </div>
      </div>
      <div id=timeline class=timeline><div class=timelineempty>Drop a History clip here or generate the first clip.</div></div>
      <div id=timeline_foot class=timelinefoot>0 clips · 0.00 s</div>
      <div id=project_popover class=projectpopover>
        <div class=projectpophead><b>Project</b><button id=project_pop_close type=button>×</button></div>
        <label>Project name</label>
        <div class=projectline><input id=project_name type=text value="Current Project" placeholder="Current Project"><button id=project_save>SAVE</button></div>
        <div class=projectline><button id=project_download class=disabled type=button>EXPORT JSON</button><button id=timeline_duplicate type=button>DUPLICATE SEQUENCE</button></div>
        <label>Saved projects</label>
        <div class=projectline><select id=project_select><option value="">No saved timelines yet</option></select><button id=project_load>LOAD</button><button id=project_refresh title="Refresh">↻</button></div>
        <div id=timeline_project_hint class=hint>Autosaved project.</div>
      </div>
    </div>
    </div></div>
    <div class=consolebox id=consolebox style="display:none"><div class="consolehead floathead"><span class=floatgrip>⠿</span><span style="flex:1">LIVE CONSOLE</span><div class=consoleactions><button id=copyconsole type=button>COPY</button><button id=clearconsole type=button>CLEAR</button><button id=minconsole type=button>HIDE</button></div></div><pre id=console></pre></div>
    <section id=history_panel class="floatpanel history" aria-label="Generation history" style="display:none">
      <div class=floathead><span class=floatgrip>⠿</span><span class=floattitle>History</span><span id=history_count class=floatcount>0</span><div class=floatactions><button id=history_expand type=button>⛶ Expand</button><button id=history_hide type=button>▼ Hide</button></div></div>
      <div id=history_body class=floatbody><div class=floatempty>No completed generations yet.</div></div>
    </section>
    <section id=queue_panel class="floatpanel queue" aria-label="Generation queue" style="display:none">
      <div class=floathead><span class=floatgrip>⠿</span><span class=floattitle>Queue</span><span id=queue_count class=floatcount>0</span><div class=floatactions><button id=queue_clear type=button>✕ Cancel queued</button><button id=queue_hide type=button>▼ Hide</button></div></div>
      <div id=queue_body class=floatbody><div class=floatempty>Queue is empty.</div></div>
    </section>
    <div id=image_view_modal class=imgmodal role=dialog aria-modal=true aria-label="Image preview"><button id=image_view_close class=imgmodalclose type=button>✕</button><img id=image_view_full alt="Full size reference image"></div>
    <div id=auto_prompt_modal class=apmodal role=dialog aria-modal=true aria-labelledby=ap_title>
      <div class=apdialog>
        <div class=aphead><b id=ap_title>AUTO PROMPT SETTINGS</b><button id=ap_close class=apclose type=button>✕</button></div>
        <label>Saved instruction profile</label>
        <select id=ap_profile_select></select>
        <label>Profile name</label>
        <div class=approfilerow><input id=ap_profile_name type=text placeholder="e.g. Handheld realism"><button id=ap_profile_save type=button>SAVE / UPDATE</button><button id=ap_profile_delete type=button>DELETE</button></div>
        <label>OpenAI model</label>
        <select id=ap_model>
          <option value="gpt-5.6-terra" selected>GPT-5.6 Terra · balanced</option>
          <option value="gpt-5.6-sol">GPT-5.6 Sol · highest quality</option>
          <option value="gpt-5.6-luna">GPT-5.6 Luna · fastest / lowest cost</option>
          <option value="__custom__">Custom model ID…</option>
        </select>
        <div id=ap_custom_wrap><label>Custom model ID</label><input id=ap_custom_model type=text placeholder="gpt-5.6-terra"></div>
        <label>Additional Auto Prompt instructions</label>
        <textarea id=ap_extra placeholder="Examples: keep it one continuous handheld shot; no music; preserve wardrobe exactly; emphasize realistic body physics; always end on a close-up..."></textarea>
        <div id=ap_key_status class=apstatus>Checking OPENAI_API_KEY…</div>
        <div class=hint>The generated text follows MiniMax H3's timeline + soundscape + music format and visually inspects attached first/last frames. Your API key is read server-side and is never sent to the browser.</div>
        <div class=apactions><button id=ap_cancel class=inlinebtn type=button>Cancel</button><button id=ap_save type=button>Save settings</button></div>
      </div>
    </div>
    <div id=ui_modal class=uimodal role=dialog aria-modal=true aria-labelledby=ui_modal_title>
      <div class=uidialog>
        <div class=uihead><b id=ui_modal_title>Dialog</b><button id=ui_modal_close class=uiclose type=button>✕</button></div>
        <div class=uibody>
          <div id=ui_modal_message class=uimessage></div>
          <input id=ui_modal_input class=uiinput type=text autocomplete=off style="display:none">
        </div>
        <div class=uiactions>
          <button id=ui_modal_cancel class=uicancel type=button>Cancel</button>
          <button id=ui_modal_confirm class=uiconfirm type=button>OK</button>
        </div>
      </div>
    </div>
    <div id=adult_gate_modal hidden><input id=adult_ack_age type=checkbox><input id=adult_ack_law type=checkbox><input id=adult_ack_consent type=checkbox><button id=adult_gate_close type=button></button><button id=adult_gate_cancel type=button></button><button id=adult_gate_accept type=button disabled></button></div>
    <script>
    const $=i=>document.getElementById(i);
    let job=null;
    let activeQueueJob=null;
    $('rnd').onclick=e=>{e.preventDefault();$('seed').value=Math.floor(Math.random()*1e9)};

    const dot=k=>document.querySelector('.dot').className='dot '+(k||'');
    const say=t=>$('stxt').textContent=t;

    const MODEL_MODE_STORE_KEY='h3_model_mode_v86';
    function currentModelMode(){return $('input_mode').value||localStorage.getItem(MODEL_MODE_STORE_KEY)||'fl2va'}
    function updateContinuityAvailability(){
      const fl=currentModelMode()==='fl2va';
      const toggle=$('continuity_enabled');
      if(!toggle)return;
      toggle.disabled=!fl;
      if(!fl)toggle.checked=false;
      const segs=((window.H3TIMELINE||{}).segments)||[];
      if(!toggle.checked){
        $('continuity_hint').innerHTML='<b>OFF:</b> GENERATE makes a fully independent clip. When enabled, the next GENERATE uses the previous timeline clip\'s lossless final frame as its first-frame anchor. No latent data is carried between clips.';
      }else if(!segs.length){
        $('continuity_hint').innerHTML='<b>ARMED:</b> there is no previous timeline clip yet, so this GENERATE will be independent. The following GENERATE can continue from its final frame.';
      }else{
        $('continuity_hint').innerHTML='<b>LAST-FRAME CONTINUITY:</b> the next GENERATE will use the previous timeline clip\'s lossless final frame as its first-frame anchor. No latent state, overlap, or latent checkpoint is used.';
      }
    }
    function customModelForProfile(profile=ACTIVE_MODEL_PROFILE,m=window.H3META||{}){
      if(!String(profile||'').startsWith('custom:'))return null;
      return (m.custom_models||[]).find(x=>x.profile===profile)||null;
    }
    function activeModelLabel(){
      const c=customModelForProfile();
      return c?(c.label||c.repo_id||c.local_name):'Built-in MiniMax H3';
    }
    function updateModeHint(){
      const m=window.H3META||{},mode=currentModelMode(),el=$('mode_hint'),name=activeModelLabel();
      if(mode==='ref2va'){
        el.innerHTML=m.ref2va_available===false
          ? '<span style="color:#ff8b8b">Ref2VA is unavailable in this ComfyUI build.</span>'
          : `<b>${name}</b> · REF2VA / REFERENCES · up to ${m.ref2va_max_images||9} images, ${m.ref2va_max_videos||3} videos and ${m.ref2va_max_audios||3} audio clips.`;
      }else{
        el.innerHTML=`<b>${name}</b> · CURRENT / KEYFRAMES · text, first-frame, or first+last-frame workflow.`;
      }
    }
    function syncModelModeUI(){const mode=currentModelMode();$('input_mode').value=mode;$('tab_fl2va').classList.toggle('active',mode==='fl2va');$('tab_ref2va').classList.toggle('active',mode==='ref2va');$('mode_panel_fl2va').classList.toggle('active',mode==='fl2va');$('mode_panel_ref2va').classList.toggle('active',mode==='ref2va');if(mode!=='fl2va')stageButtonActive(false);updateModeHint();updateContinuityAvailability()}
    async function setModelMode(mode){
      let m=window.H3META||{};
      if(mode==='ref2va'&&m.ref2va_available===false){
        await uiAlert('MiniMaxH3ReferenceToVideo is not available in this ComfyUI build. Update ComfyUI or install the official H3 extras first.','Ref2VA unavailable');
        return
      }

      const previousMode=currentModelMode();
      $('input_mode').value=mode;
      localStorage.setItem(MODEL_MODE_STORE_KEY,mode);

      try{
        const customModel=customModelForProfile(ACTIVE_MODEL_PROFILE,m);
        if(customModel){
          if(customModel.local_name&&[...$('unet').options].some(x=>x.value===customModel.local_name)){
            $('unet').value=customModel.local_name;
          }else{
            throw new Error('The selected custom base model is no longer installed.');
          }
        // Legacy profile branches retained for old browser state.
        }else if(ACTIVE_MODEL_PROFILE==='eros'){
          if(m.eros_max_unet&&[...$('unet').options].some(x=>x.value===m.eros_max_unet)){
            $('unet').value=m.eros_max_unet;
          }
        }else if(ACTIVE_MODEL_PROFILE==='redmix'){
          if(mode!=='fl2va')throw new Error('REDMIX H3 Beta2 is available in CURRENT · KEYFRAMES mode.');
          if(!(await _ensureProfile('redmix','fl2va')))throw new Error('Could not prepare REDMIX H3 Beta2.');
          await loadMeta();m=window.H3META||{};
          if(!m.redmix_unet||![...$('unet').options].some(x=>x.value===m.redmix_unet))throw new Error('REDMIX H3 Beta2 checkpoint is not available after install.');
          $('unet').value=m.redmix_unet;
        }else{
          // Stock H3 requires the mode-matching official transformer.
          if(!(await _ensureProfile(ACTIVE_MODEL_PROFILE,mode))){
            throw new Error(`Could not prepare ${activeModelLabel()} for ${mode==='ref2va'?'Ref2VA':'Current'} mode.`);
          }
          await loadMeta();
          m=window.H3META||{};
          const target=mode==='ref2va'?m.stock_ref2va_unet:m.base_fl2va_unet;
          if(!target||![...$('unet').options].some(x=>x.value===target)){
            throw new Error(`Matching ${mode==='ref2va'?'Ref2VA':'FL2VA'} transformer is not available for ${activeModelLabel()}.`);
          }
          $('unet').value=target;

        }

        // Reapply the selected model's current performance preset for the new mode.
        await applyPerformancePreset(ACTIVE_PERF_PRESET);
        syncModelModeUI();
        syncModelProfileUI();
        updateModeHint();
        say(`${activeModelLabel()} · ${mode==='ref2va'?'Ref2VA / References':'Current / Keyframes'} · ${ACTIVE_PERF_PRESET.toUpperCase()}`);
      }catch(e){
        $('input_mode').value=previousMode;
        localStorage.setItem(MODEL_MODE_STORE_KEY,previousMode);
        syncModelModeUI();
        await uiAlert(String(e&&e.message?e.message:e),'Mode switch failed');
      }
    }

    $('tab_fl2va').onclick=()=>setModelMode('fl2va');
    $('tab_ref2va').onclick=()=>setModelMode('ref2va');

    // Continuity is opt-in and intentionally uses ONLY the pre-latent last-frame path.
    $('continuity_enabled').checked=false;
    $('continuity_keep_seed').checked=false;
    $('continuity_enabled').addEventListener('change',updateContinuityAvailability);

    let ACTIVE_MODEL_PROFILE='stock_quality';
    let ACTIVE_PERF_PRESET='fast';

    function presetRecipe(profile=ACTIVE_MODEL_PROFILE,mode=currentModelMode(),kind=ACTIVE_PERF_PRESET){
      const label=activeModelLabel();
      if(kind==='ultra'){
        return {short:'ULTRA',label,steps:4,summary:(mode==='ref2va'?'Ref2VA':'FL2VA')+' · 4-step Lightning · Euler / Simple · sparse attention 5%'};
      }
      if(kind==='fast'){
        return {short:'FAST',label,steps:4,summary:(mode==='ref2va'?'Ref2VA':'FL2VA')+' · 4-step Lightning · Euler / Simple · dense attention'};
      }
      return {short:'QUALITY',label,steps:20,summary:(mode==='ref2va'?'Ref2VA':'FL2VA')+' · RES Multistep / Simple · 20 steps · dense attention'};
    }

    function syncSpecialLoraUI(){
      if(typeof syncUnifiedLoraCompatibility==='function')syncUnifiedLoraCompatibility();
    }

    function syncPresetUI(){
      syncSpecialLoraUI();
      const fast=presetRecipe(ACTIVE_MODEL_PROFILE,currentModelMode(),'fast');
      const ultra=presetRecipe(ACTIVE_MODEL_PROFILE,currentModelMode(),'ultra');
      const quality=presetRecipe(ACTIVE_MODEL_PROFILE,currentModelMode(),'quality');
      const active=ACTIVE_PERF_PRESET==='ultra'?ultra:(ACTIVE_PERF_PRESET==='quality'?quality:fast);
      $('fastpreset').classList.toggle('active',ACTIVE_PERF_PRESET==='fast');
      $('ultrafastpreset').classList.toggle('active',ACTIVE_PERF_PRESET==='ultra');
      $('qualitypreset').classList.toggle('active',ACTIVE_PERF_PRESET==='quality');
      $('fastpreset').textContent=`FAST · ${fast.steps}`;
      $('ultrafastpreset').textContent=`ULTRA FAST · ${ultra.steps}`;
      $('qualitypreset').textContent=`QUALITY · ${quality.steps}`;
      $('preset_hint').innerHTML=
        `<b>${active.label}</b> · ${currentModelMode()==='ref2va'?'REF2VA / REFERENCES':'CURRENT / KEYFRAMES'} · ${ACTIVE_PERF_PRESET==='ultra'?'ULTRA FAST':ACTIVE_PERF_PRESET.toUpperCase()}<br>`+
        `${active.summary}<br><span style="color:#70737d">FAST is the default. ULTRA FAST adds 5% sparse attention on top of the 4-step Lightning recipe.</span>`;
    }

    function _modelProfileState(){return (window.H3META||{}).model_profiles||{}}
    function _hasProfileFile(profile,mode=currentModelMode()){
      const s=_modelProfileState();
      if(String(profile||'').startsWith('custom:'))return !!customModelForProfile(profile);
      if(profile==='stock_quality')return mode==='ref2va'?!!s.stock_ref2va_installed:!!s.stock_fl2va_installed;
      return false;
    }
    function syncModelProfileOptions(m=window.H3META||{}){
      const sel=$('model_profile_select');
      const current=ACTIVE_MODEL_PROFILE;
      const s=m.model_profiles||{};
      const mode=currentModelMode();
      const stockReady=mode==='ref2va'?!!s.stock_ref2va_installed:!!s.stock_fl2va_installed;
      const rows=[{value:'stock_quality',label:'BUILT-IN MINIMAX H3'+(stockReady?'':' · DOWNLOAD')}];
      for(const x of (m.custom_models||[])){
        rows.push({value:x.profile,label:'HF · '+(x.repo_id||x.local_name)+' · '+(x.filename||x.local_name)});
      }
      sel.innerHTML=rows.map(x=>`<option value="${esc(x.value)}">${esc(x.label)}</option>`).join('');
      if(rows.some(x=>x.value===current))sel.value=current;
      else{ACTIVE_MODEL_PROFILE='stock_quality';sel.value='stock_quality'}
    }

    function syncModelProfileUI(){
      const mode=currentModelMode(),sel=$('model_profile_select');
      syncModelProfileOptions(window.H3META||{});
      sel.value=ACTIVE_MODEL_PROFILE;
      const cm=customModelForProfile(ACTIVE_MODEL_PROFILE);
      $('model_profile_hint').innerHTML=
        `Selected model: <b>${activeModelLabel()}</b> · `+
        `${cm?('user HF checkpoint · '+(cm.mode||'both')):(mode==='ref2va'?'Ref2VA transformer/reference conditioning':'FL2VA keyframe conditioning')} · `+
        `${_hasProfileFile(ACTIVE_MODEL_PROFILE,mode)?'<span class="installed">ready</span>':'<span class="missing">download on use</span>'}.`;

      $('tab_fl2va').textContent='CURRENT · KEYFRAMES';
      $('tab_ref2va').textContent='REF2VA · REFERENCES';
      updateModeHint();
      syncUnifiedLoraCompatibility();
      syncPresetUI();
    }
    async function _ensureProfile(profile,mode=currentModelMode()){
      if(String(profile||'').startsWith('custom:'))return !!customModelForProfile(profile);
      if(_hasProfileFile(profile,mode))return true;
      say('downloading model profile…');
      const resp=await fetch('/api/model_profiles/install',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({profile,mode})
      });
      const r=await resp.json();
      if(r.adult_ack_required){
        const ok=await requestAdultAcknowledgement();
        return ok?_ensureProfile(profile,mode):false;
      }
      if(r.error){await uiAlert(r.error,'Model profile download failed');return false}
      await loadMeta();
      return true;
    }
    async function applyModelProfile(profile,{install=true}={}){
      const mode=currentModelMode();
      const m=window.H3META||{};
      const cm=customModelForProfile(profile,m);
      if(cm){
        if(![...$('unet').options].some(x=>x.value===cm.local_name)){
          await loadMeta();
        }
        if(![...$('unet').options].some(x=>x.value===cm.local_name))throw new Error('Custom model file is not visible to ComfyUI.');
        $('unet').value=cm.local_name;
        ACTIVE_MODEL_PROFILE=profile;
        await applyPerformancePreset(ACTIVE_PERF_PRESET||'fast');
        syncModelProfileUI();
        return;
      }
      if(install && !(await _ensureProfile('stock_quality',mode)))return;
      const mm=window.H3META||{};
      const target=mode==='ref2va'?mm.stock_ref2va_unet:mm.base_fl2va_unet;
      if(target&&[...$('unet').options].some(x=>x.value===target))$('unet').value=target;
      ACTIVE_MODEL_PROFILE='stock_quality';
      await applyPerformancePreset(ACTIVE_PERF_PRESET||'fast');
      syncModelProfileUI();
    }

    $('model_profile_select').onchange=async()=>{
      const sel=$('model_profile_select');
      const requested=sel.value;
      const previous=ACTIVE_MODEL_PROFILE;
      sel.disabled=true;
      say('switching model · checking matching '+(currentModelMode()==='ref2va'?'Ref2VA':'FL2VA')+' files…');
      try{
        await applyModelProfile(requested);
      }catch(e){
        ACTIVE_MODEL_PROFILE=previous;
        await uiAlert(String(e&&e.message?e.message:e),'Model switch failed');
      }finally{
        sel.disabled=false;
        syncModelProfileUI();
      }
    };


    function _humanBytes(n){n=Number(n||0);const u=['B','KiB','MiB','GiB','TiB'];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return `${n.toFixed(i<2?1:2)} ${u[i]}`}
    $('hf_model_toggle').onclick=e=>{e.preventDefault();$('hf_model_box').classList.toggle('show')};
    $('hf_model_inspect').onclick=async e=>{
      e.preventDefault();
      const repo=$('hf_model_repo').value.trim(),revision=$('hf_model_revision').value.trim()||'main';
      if(!repo){await uiAlert('Enter a Hugging Face repo such as owner/repository.','HF base model');return}
      $('hf_model_inspect').disabled=true;$('hf_model_install').disabled=true;
      $('hf_model_progress').style.width='0%';$('hf_model_progress_text').textContent='Inspecting repo…';
      try{
        const resp=await fetch('/api/models/hf/inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({repo_id:repo,revision})});
        const r=await resp.json();if(!resp.ok||r.error)throw new Error(r.error||'Could not inspect repo.');
        $('hf_model_file').innerHTML=(r.candidates||[]).map(x=>`<option value="${esc(x.filename)}">${esc(x.filename)} · ${x.size?_humanBytes(x.size):'size unknown'}</option>`).join('');
        $('hf_model_file').disabled=!(r.candidates||[]).length;$('hf_model_install').disabled=!(r.candidates||[]).length;
        $('hf_model_progress_text').textContent=`${(r.candidates||[]).length} checkpoint file(s) found. Choose one and download.`;
      }catch(err){$('hf_model_progress_text').textContent=String(err.message||err);await uiAlert(String(err.message||err),'HF repo inspection failed')}
      finally{$('hf_model_inspect').disabled=false}
    };
    async function pollHFModelDownload(id){
      while(true){
        const resp=await fetch('/api/models/hf/progress/'+encodeURIComponent(id),{cache:'no-store'});const r=await resp.json();
        if(!resp.ok||r.error&&r.status!=='error')throw new Error(r.error||'Download status failed.');
        const pct=r.pct==null?0:Math.max(0,Math.min(100,Number(r.pct)));
        $('hf_model_progress').style.width=pct+'%';
        $('hf_model_progress_text').textContent=`${r.stage||r.status} · ${r.downloaded_text||'0 B'} / ${r.total_text||'unknown'}${r.speed_text?' · '+r.speed_text:''}${r.pct==null?'':` · ${pct.toFixed(1)}%`}`;
        if(r.status==='done')return r;
        if(r.status==='error')throw new Error(r.error||'Model download failed.');
        await new Promise(resolve=>setTimeout(resolve,500));
      }
    }
    $('hf_model_install').onclick=async e=>{
      e.preventDefault();
      const repo=$('hf_model_repo').value.trim(),revision=$('hf_model_revision').value.trim()||'main',filename=$('hf_model_file').value,mode=$('hf_model_mode').value||'both';
      if(!repo||!filename){await uiAlert('Check the repo and choose a model file first.','HF base model');return}
      $('hf_model_install').disabled=true;$('hf_model_inspect').disabled=true;
      try{
        const resp=await fetch('/api/models/hf/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({repo_id:repo,revision,filename,mode})});
        const r=await resp.json();if(!resp.ok||r.error)throw new Error(r.error||'Could not start model download.');
        const done=await pollHFModelDownload(r.id);
        await loadMeta();
        ACTIVE_MODEL_PROFILE=done.profile||('custom:'+done.local_name);
        syncModelProfileOptions(window.H3META||{});$('model_profile_select').value=ACTIVE_MODEL_PROFILE;
        await applyModelProfile(ACTIVE_MODEL_PROFILE,{install:false});
        $('hf_model_progress').style.width='100%';$('hf_model_progress_text').textContent=`Installed ${done.local_name}. Selected as the active base model.`;
        say('HF base model installed · '+done.local_name);
      }catch(err){$('hf_model_progress_text').textContent=String(err.message||err);await uiAlert(String(err.message||err),'HF base model download failed')}
      finally{$('hf_model_install').disabled=false;$('hf_model_inspect').disabled=false}
    };

    $('gpu_overlay_toggle').onclick=()=>{
      const p=$('gpu_overlay');
      const mini=p.classList.toggle('minimized');
      $('gpu_overlay_toggle').textContent=mini?'＋':'−';
      $('gpu_overlay_toggle').title=mini?'Expand GPU stats':'Minimize GPU stats';
    };

    // App-owned modal system. Never use browser alert/confirm/prompt dialogs.
    let _uiDialogResolve=null;
    function _closeUIDialog(value){
      $('ui_modal').classList.remove('show');
      const r=_uiDialogResolve; _uiDialogResolve=null;
      if(r)r(value);
    }
    function uiDialog({title='Notice',message='',input=false,value='',confirmLabel='OK',cancelLabel='Cancel',showCancel=true,danger=false}={}){
      if(_uiDialogResolve)_closeUIDialog(null);
      $('ui_modal_title').textContent=title;
      $('ui_modal_message').textContent=message;
      const inp=$('ui_modal_input');
      inp.style.display=input?'block':'none';
      inp.value=input?String(value??''):'';
      $('ui_modal_confirm').textContent=confirmLabel;
      $('ui_modal_confirm').className='uiconfirm '+(danger?'danger':'neutral');
      $('ui_modal_cancel').textContent=cancelLabel;
      $('ui_modal_cancel').style.display=showCancel?'':'none';
      $('ui_modal').classList.add('show');
      return new Promise(resolve=>{
        _uiDialogResolve=resolve;
        requestAnimationFrame(()=>{(input?inp:$('ui_modal_confirm')).focus(); if(input)inp.select()});
      });
    }
    function uiAlert(message,title='Notice'){return uiDialog({title,message,showCancel:false,confirmLabel:'OK'})}
    function uiConfirm(message,{title='Confirm',confirmLabel='Confirm',danger=false}={}){return uiDialog({title,message,showCancel:true,confirmLabel,danger})}
    function uiPrompt(message,value='',{title='New sequence',confirmLabel='Create'}={}){return uiDialog({title,message,input:true,value,showCancel:true,confirmLabel})}
    $('ui_modal_close').onclick=()=>_closeUIDialog(null);
    $('ui_modal_cancel').onclick=()=>_closeUIDialog(null);
    $('ui_modal_confirm').onclick=()=>_closeUIDialog($('ui_modal_input').style.display==='none'?true:$('ui_modal_input').value.trim());
    $('ui_modal').addEventListener('click',e=>{if(e.target===$('ui_modal'))_closeUIDialog(null)});
    $('ui_modal_input').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();$('ui_modal_confirm').click()}else if(e.key==='Escape'){e.preventDefault();_closeUIDialog(null)}});

    // Adult-access acknowledgement is intentionally a signed server-session gate.
    let _adultGateResolve=null;
    function _syncAdultGateAccept(){
      $('adult_gate_accept').disabled=!($('adult_ack_age').checked&&$('adult_ack_law').checked&&$('adult_ack_consent').checked);
    }
    function _closeAdultGate(value){
      $('adult_gate_modal').classList.remove('show');
      const r=_adultGateResolve;_adultGateResolve=null;
      if(r)r(!!value);
    }
    async function requestAdultAcknowledgement(){
      if((window.H3META||{}).adult_enabled)return true;
      for(const id of ['adult_ack_age','adult_ack_law','adult_ack_consent'])$(id).checked=false;
      _syncAdultGateAccept();
      $('adult_gate_modal').classList.add('show');
      return new Promise(resolve=>{_adultGateResolve=resolve;requestAnimationFrame(()=>$('adult_ack_age').focus())});
    }
    for(const id of ['adult_ack_age','adult_ack_law','adult_ack_consent'])$(id).addEventListener('change',_syncAdultGateAccept);
    $('adult_gate_close').onclick=()=>_closeAdultGate(false);
    $('adult_gate_cancel').onclick=()=>_closeAdultGate(false);
    $('adult_gate_modal').addEventListener('click',e=>{if(e.target===$('adult_gate_modal'))_closeAdultGate(false)});
    $('adult_gate_accept').onclick=async()=>{
      if($('adult_gate_accept').disabled)return;
      $('adult_gate_accept').disabled=true;
      say('unlocking adult catalog…');
      try{
        const resp=await fetch('/api/adult/ack',{
          method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({age:true,law:true,consent:true})
        });
        const r=await resp.json();
        if(!resp.ok||r.error){await uiAlert(r.error||'Adult acknowledgement failed.','Adult access');_syncAdultGateAccept();return}
        await loadMeta();
        _closeAdultGate(true);
        say('Adult Mode enabled');
      }catch(e){
        await uiAlert(String(e),'Adult access');
        _syncAdultGateAccept();
      }
    };

    function syncAdultModeButton(m=window.H3META||{}){
      const b=$('adult_mode_btn');if(!b)return;
      const on=!!m.adult_enabled;
      b.classList.toggle('on',on);
      b.textContent='18+';
      b.setAttribute('aria-pressed',on?'true':'false');
      b.title=on?'Adult catalog enabled · click to exit':'Adult models and LoRAs are hidden · click to enable';
    }
    function clearAdultLoraCardState(m=window.H3META||{}){
      for(const card of _namedLoraCards(m)){
        if(card.adult||card.kind==='catalog')LORA_CARD_STATE.delete(_cardKey(card));
      }
    }
    $('adult_mode_btn').onclick=async()=>{
      const m=window.H3META||{};
      if(!m.adult_enabled){
        const ok=await requestAdultAcknowledgement();
        if(ok){syncAdultModeButton(window.H3META||{});renderNamedLoraRows(window.H3META||{});syncModelProfileUI()}
        return;
      }
      if(!(await uiConfirm('Exit Adult Mode? Adult models and adult LoRAs will be hidden and any active adult LoRA selections will be cleared. Installed files stay on disk.',{title:'Exit Adult Mode',confirmLabel:'Exit Adult Mode'})))return;
      clearAdultLoraCardState(m);
      const resp=await fetch('/api/adult/exit',{method:'POST'});
      const r=await resp.json();
      if(!resp.ok||r.error){await uiAlert(r.error||'Could not exit Adult Mode.','Adult Mode');return}
      if(['eros','redmix'].includes(ACTIVE_MODEL_PROFILE))ACTIVE_MODEL_PROFILE='stock_quality';
      await loadMeta();
      syncAdultModeButton(window.H3META||{});
      say('Adult Mode off');
    };

    // One user-facing LoRA stack. There are no legacy LoRA slot/dropdown controls
    // in the DOM. All strengths—including Motion and Fast Accelerator—live here.
    // State is keyed by installed filename or gated catalog key so loadMeta() can
    // refresh compatibility/adult visibility without losing the user's strengths.
    const LORA_CARD_STATE=new Map();

    function _catalogEntryForKey(key,m=window.H3META||{}){
      return (m.lora_catalog||[]).find(x=>String(x.key)===String(key))||null;
    }
    function _catalogEntryForFile(file,m=window.H3META||{}){
      return (m.lora_catalog||[]).find(x=>x.file&&String(x.file)===String(file))||null;
    }
    function _compatForFile(file,m=window.H3META||{}){
      return (m.lora_compatibility||{})[String(file||'')]||null;
    }
    function _compatProfileForUI(m=window.H3META||{}){
      const unet=$('unet')?$('unet').value:'';
      return 'stock_quality';
    }
    function _loraFileCompatible(file,m=window.H3META||{}){
      if(!file)return true;
      const info=_compatForFile(file,m);
      if(!info)return true; // unknown user LoRA: don't invent incompatibility
      return (info.profiles||[]).includes(_compatProfileForUI(m))&&(info.modes||[]).includes(currentModelMode());
    }
    function _humanLoraName(file){
      return String(file||'LoRA').replace(/\.safetensors$/i,'').replace(/[_-]+/g,' ').replace(/\s+/g,' ').trim();
    }
    function _cardKey(card){return card.kind==='catalog'?'catalog:'+card.catalogKey:'file:'+card.file}
    function _loraSourceMap(m=window.H3META||{}){return m.lora_source_map||{}}
    function _cardTitleHTML(card){
      const label=esc(card.label||'LoRA');
      const url=String(card.source_url||'').trim();
      if(!url)return label;
      return `<a class="loratitlelink" href="${esc(url)}" target="_blank" rel="noopener noreferrer" title="Open source page">${label}</a>`;
    }
    function _stateForCard(card){
      const k=_cardKey(card);
      if(!LORA_CARD_STATE.has(k))LORA_CARD_STATE.set(k,{strength:0});
      return LORA_CARD_STATE.get(k);
    }
    function _namedLoraCards(m=window.H3META||{}){
      const cards=[];
      const sourceMap=_loraSourceMap(m);
      const catalogByFile=new Map((m.lora_catalog||[]).filter(x=>x.file).map(x=>[x.file,x]));
      const seen=new Set();

      // Keep the controls that were useful before: Motion Enhancer and the matching
      // Fast-mode accelerator, but place them in this same stack.
      if(m.motion8_file){
        cards.push({kind:'motion8',file:m.motion8_file,label:'Motion Enhancer LoRA',available:!!m.motion8_available,
          detail:'rzgar H3 FL2V 8-step',source_url:sourceMap[m.motion8_file]||''});seen.add(m.motion8_file);
      }
      const accelFile=currentModelMode()==='ref2va'?m.ref2va_lightning_file:m.lightning_file;
      const accelAvailable=currentModelMode()==='ref2va'?m.ref2va_lightning_available:m.lightning_available;
      if(accelFile){
        cards.push({kind:'lightning',file:accelFile,label:'Fast-Mode Accelerator',available:!!accelAvailable,
          detail:currentModelMode()==='ref2va'?'Ref2VA 4-step Turbo':'FL2VA 4-step Turbo',source_url:sourceMap[accelFile]||''});seen.add(accelFile);
      }

      // Installed creative LoRAs become named cards. /api/meta has already stripped
      // adult filenames when Adult Mode is off, so they cannot leak into this list.
      for(const file of (m.loras||[])){
        if(!file||file==='none'||seen.has(file))continue;
        const cat=catalogByFile.get(file),info=_compatForFile(file,m);
        cards.push({kind:'creative',file,label:(cat&&cat.label)||(info&&info.label)||_humanLoraName(file),available:true,
          adult:!!((cat&&cat.adult)||(info&&info.adult)),detail:cat&&cat.adult?'Adult LoRA':(info&&info.label)||'Installed LoRA',
          source_url:(cat&&cat.source_page)||sourceMap[file]||''});
        seen.add(file);
      }

      // Gated adult catalog rows only arrive from /api/meta after acknowledgement.
      // Uninstalled rows still appear as named cards and install only when switched ON.
      for(const cat of (m.lora_catalog||[])){
        if(cat.file&&seen.has(cat.file))continue;
        cards.push({kind:'catalog',catalogKey:cat.key,file:cat.file||'',label:cat.label||cat.key,adult:!!cat.adult,
          available:true,installed:!!cat.installed,detail:cat.installed?'Adult LoRA':'Adult LoRA · download on enable',
          source_url:cat.source_page||sourceMap[cat.file||'']||''});
      }
      return cards;
    }
    function _cardCompatible(card,m=window.H3META||{}){
      if(card.kind==='motion8')return currentModelMode()==='fl2va'&&_compatProfileForUI(m)==='stock_quality'&&!!card.available;
      if(card.kind==='lightning')return _compatProfileForUI(m)==='stock_quality'&&!!card.available;
      if(card.kind==='catalog'&&!card.installed){
        const cat=_catalogEntryForKey(card.catalogKey,m);
        return !!cat&&(cat.profiles||['stock_quality']).includes(_compatProfileForUI(m))&&(cat.modes||['fl2va']).includes(currentModelMode());
      }
      return !!card.available&&_loraFileCompatible(card.file,m);
    }
    function _strengthForKind(kind,m=window.H3META||{}){
      const card=_namedLoraCards(m).find(x=>x.kind===kind);
      if(!card)return 0;
      return Number(_stateForCard(card).strength||0);
    }
    function _setStrengthForKind(kind,strength,m=window.H3META||{}){
      const card=_namedLoraCards(m).find(x=>x.kind===kind);
      if(!card)return;
      _stateForCard(card).strength=Number(strength||0);
    }
    async function _installCatalogCard(card){
      const resp=await fetch('/api/loras/catalog_install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:card.catalogKey})});
      const r=await resp.json();
      if(r.adult_ack_required){const ok=await requestAdultAcknowledgement();if(!ok)return null;return _installCatalogCard(card)}
      if(!resp.ok||r.error){await uiAlert(r.error||'LoRA install failed.','LoRA install failed');return null}
      const oldKey=_cardKey(card),state=LORA_CARD_STATE.get(oldKey)||{strength:0};
      LORA_CARD_STATE.delete(oldKey);LORA_CARD_STATE.set('file:'+r.file,state);
      await loadMeta();
      return r.file;
    }
    function renderNamedLoraRows(m=window.H3META||{}){
      const box=$('unified_lora_rows');if(!box)return;box.innerHTML='';
      const cards=_namedLoraCards(m);
      if(!cards.length){
        box.innerHTML='<div class="lorarow"><div class=lorarowhead><div><div class=lorarowtitle>No LoRAs available</div><div class=lorarowmeta>Install a compatible LoRA to add it here.</div></div></div></div>';
        return;
      }
      for(const card of cards){
        const st=_stateForCard(card),compatible=_cardCompatible(card,m);
        // Compatibility never erases a user's selection. Incompatible cards are
        // disabled/greyed and omitted from generation, then become active again if
        // the user switches back to a compatible model or input mode.
        const selected=Math.abs(Number(st.strength||0))>1e-6;
        const active=compatible&&selected;
        const row=document.createElement('div');row.className='lorarow'+(compatible?'':' incompatible');
        const safeId='lc_'+Math.random().toString(36).slice(2);
        const status=!compatible?`incompatible with current model / mode${selected?` · saved ${Number(st.strength).toFixed(2)}`:''}`:(active?`active · ${Number(st.strength).toFixed(2)}`:(card.kind==='catalog'&&!card.installed?'download on enable':'off'));
        row.innerHTML=`<div class=lorarowhead><div><div class=lorarowtitle>${_cardTitleHTML(card)}</div><div class=lorarowmeta>${esc(status)}</div></div>`+
          `<button class="loratoggle${selected?' on':''}" type=button aria-pressed="${selected?'true':'false'}" ${compatible?'':'disabled'}>${selected?'ON':'OFF'}</button></div>`+
          `<div class=sliderline><input id="${safeId}" type=range min="0" max="5" step=".05" value="${Math.max(0,Math.min(5,Number(st.strength||0)))}" ${compatible?'':'disabled'}><input class="slidervalue loranumber" type=number step="any" value="${Number(st.strength||0).toFixed(2)}" aria-label="${esc(card.label)} strength" ${compatible?'':'disabled'}></div>`+
          `<div class=hint>${esc(card.detail||'')}</div>`;
        box.appendChild(row);
        const slider=row.querySelector('input[type=range]'),out=row.querySelector('.loranumber'),toggle=row.querySelector('.loratoggle'),meta=row.querySelector('.lorarowmeta');
        const sliderMin=0,sliderMax=5;
        const sliderValueFor=n=>Math.max(sliderMin,Math.min(sliderMax,Number(n)));
        const syncState=(n,{normalizeNumber=true}={})=>{
          if(!Number.isFinite(n))return;
          // The numeric field is authoritative and intentionally unbounded. The
          // slider is only a 0–5 convenience control, so out-of-range numbers pin
          // the thumb to the nearest endpoint without changing the stored strength.
          st.strength=n;
          slider.value=String(sliderValueFor(n));
          if(normalizeNumber)out.value=String(n);
          const on=Math.abs(n)>1e-6;
          toggle.classList.toggle('on',on);toggle.textContent=on?'ON':'OFF';toggle.setAttribute('aria-pressed',on?'true':'false');
          meta.textContent=on?`active · ${n}`:'off';
        };
        const commitNumber=v=>{const n=Number(v);if(Number.isFinite(n))syncState(n)};
        slider.addEventListener('input',()=>syncState(Number(slider.value)));
        out.addEventListener('input',()=>{
          const raw=out.value.trim();
          if(raw===''||raw==='-'||raw==='+'||raw==='.'||raw==='-.'||raw==='+.'||/[eE][+-]?$/.test(raw))return;
          const n=Number(raw);if(Number.isFinite(n))syncState(n,{normalizeNumber:false});
        });
        out.addEventListener('change',()=>commitNumber(out.value));
        out.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();commitNumber(out.value);out.blur()}});
        toggle.onclick=async e=>{
          e.preventDefault();
          if(toggle.disabled)return;
          const enabling=!toggle.classList.contains('on');

          // The switch is authoritative: OFF always means strength 0.00 and ON
          // always means strength 1.00. syncState updates the stored strength,
          // numeric field, slider thumb, switch label, and status text together.
          if(card.kind==='catalog'&&!card.installed){
            if(!enabling){syncState(0);return}
            toggle.disabled=true;
            meta.textContent='downloading…';
            const file=await _installCatalogCard(card);
            if(!file){renderNamedLoraRows(window.H3META||{});return}
            const ns=LORA_CARD_STATE.get('file:'+file)||{strength:0};
            ns.strength=1;
            LORA_CARD_STATE.set('file:'+file,ns);
            renderNamedLoraRows(window.H3META||{});
            return;
          }
          syncState(enabling?1:0);
        };
      }
    }
    function syncUnifiedLoraCompatibility(){renderNamedLoraRows(window.H3META||{})}
    $('unet').addEventListener('change',syncUnifiedLoraCompatibility);

    function activeCreativeLoraStack(){
      const m=window.H3META||{},out=[];
      for(const card of _namedLoraCards(m)){
        if(card.kind!=='creative')continue;
        const st=_stateForCard(card),strength=Number(st.strength||0);
        if(card.file&&Math.abs(strength)>1e-6&&_cardCompatible(card,m))out.push({file:card.file,strength});
      }
      return out;
    }
    function submittedSpecialStrength(kind){
      const m=window.H3META||{},card=_namedLoraCards(m).find(x=>x.kind===kind);
      if(!card||!_cardCompatible(card,m))return 0;
      return Number(_stateForCard(card).strength||0);
    }
    function resetNamedLoras(){
      LORA_CARD_STATE.clear();renderNamedLoraRows(window.H3META||{});
    }

    async function _installUserLoraUrl(url){
      let resp=await fetch('/api/loras/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
      let r=await resp.json();
      if(r.adult_ack_required){
        const ok=await requestAdultAcknowledgement();
        if(!ok)return null;
        resp=await fetch('/api/loras/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
        r=await resp.json();
      }
      if(r.error){await uiAlert(r.error,'LoRA install failed');return null}
      return r;
    }
    $('install_lora').onclick=async()=>{
      const url=await uiPrompt('Paste a direct Hugging Face .safetensors file URL or a MiniMax-H3 CivitAI model/version URL.','',{title:'Install LoRA',confirmLabel:'Install'});
      if(!url)return;
      say('installing LoRA…');$('install_lora').disabled=true;
      try{
        const r=await _installUserLoraUrl(url);if(!r)return;
        await loadMeta();
        LORA_CARD_STATE.set('file:'+r.file,{strength:1});
        renderNamedLoraRows(window.H3META||{});
        say('LoRA installed · '+r.file);
      }catch(e){await uiAlert(String(e),'LoRA install failed')}finally{$('install_lora').disabled=false}
    };

    function syncMotionPace(){$('motion_pace_value').textContent=Number($('playback_speed').value||1).toFixed(2)+'×'}
    $('playback_speed').addEventListener('input',syncMotionPace);syncMotionPace();

    const AP_STORE_KEY='h3_auto_prompt_settings_v86';
    const AP_LEGACY_STORE_KEY='h3_auto_prompt_settings_v83';
    const AP_PROFILE_STORE_KEY='h3_auto_prompt_instruction_profiles_v86';
    const AP_ACTIVE_PROFILE_KEY='h3_auto_prompt_active_profile_v86';
    const AP_DEFAULT={model:'gpt-5.6-terra',customModel:'',extra:''};

    function getAPSettings(){try{const raw=localStorage.getItem(AP_STORE_KEY)||localStorage.getItem(AP_LEGACY_STORE_KEY)||'{}';return {...AP_DEFAULT,...JSON.parse(raw)}}catch(e){return {...AP_DEFAULT}}}
    function saveAPSettings(s){localStorage.setItem(AP_STORE_KEY,JSON.stringify({...AP_DEFAULT,...s}))}
    function getAPProfiles(){try{const a=JSON.parse(localStorage.getItem(AP_PROFILE_STORE_KEY)||'[]');return Array.isArray(a)?a.filter(x=>x&&x.id&&x.name):[]}catch(e){return []}}
    function saveAPProfiles(a){localStorage.setItem(AP_PROFILE_STORE_KEY,JSON.stringify(a))}
    function activeAPProfileId(){return localStorage.getItem(AP_ACTIVE_PROFILE_KEY)||''}
    function setActiveAPProfileId(id){if(id)localStorage.setItem(AP_ACTIVE_PROFILE_KEY,id);else localStorage.removeItem(AP_ACTIVE_PROFILE_KEY)}
    function currentAPModel(){const s=getAPSettings();return s.model==='__custom__'?(s.customModel||'').trim():(s.model||AP_DEFAULT.model)}

    function syncAPProfileSelectors(){
      const profiles=getAPProfiles(),active=activeAPProfileId();
      const opts=['<option value="">Current / unsaved</option>'].concat(profiles.map(p=>`<option value="${esc(p.id)}">${esc(p.name)}</option>`)).join('');
      // Saved instruction profiles live only inside the ⚙ settings modal.
      // Keep the main Prompt toolbar intentionally clean: AUTO PROMPT + settings only.
      $('ap_profile_select').innerHTML=opts;
      const valid=profiles.some(p=>p.id===active)?active:'';
      $('ap_profile_select').value=valid;
      const p=profiles.find(x=>x.id===valid);
      $('ap_profile_name').value=p?p.name:'';
      $('ap_profile_delete').disabled=!p;
    }

    function syncAPModal(){
      const s=getAPSettings();
      $('ap_model').value=[...$('ap_model').options].some(o=>o.value===s.model)?s.model:'__custom__';
      $('ap_custom_model').value=s.customModel||((s.model&&!['gpt-5.6-terra','gpt-5.6-sol','gpt-5.6-luna','__custom__'].includes(s.model))?s.model:'');
      $('ap_extra').value=s.extra||'';
      $('ap_custom_wrap').style.display=$('ap_model').value==='__custom__'?'block':'none';
      syncAPProfileSelectors();
    }

    function applyAPProfile(id){
      if(!id){setActiveAPProfileId('');syncAPProfileSelectors();return}
      const p=getAPProfiles().find(x=>x.id===id);if(!p)return;
      saveAPSettings({model:p.model||AP_DEFAULT.model,customModel:p.customModel||'',extra:p.extra||''});
      setActiveAPProfileId(id);syncAPModal();syncAPProfileSelectors();
      $('auto_prompt_hint').innerHTML=`Auto Prompt instructions: <b>${esc(p.name)}</b> · model <b>${esc(currentAPModel())}</b>.`;
      say('Auto Prompt profile selected · '+p.name);
    }

    async function checkAPKey(){const el=$('ap_key_status');el.className='apstatus';el.textContent='Checking OPENAI_API_KEY…';try{const d=await(await fetch('/api/auto_prompt/meta',{cache:'no-store'})).json();if(d.key_available){el.className='apstatus ok';el.textContent='✓ OPENAI_API_KEY available to Auto Prompt.'}else{el.className='apstatus err';el.textContent='OPENAI_API_KEY is not available to this UI process. Add it in Colab Secrets, then rerun V86.'}}catch(e){el.className='apstatus err';el.textContent='Could not check OPENAI_API_KEY.'}}
    function openAPModal(){syncAPModal();$('auto_prompt_modal').classList.add('show');checkAPKey();$('ap_profile_select').focus()}
    function closeAPModal(){$('auto_prompt_modal').classList.remove('show')}
    $('auto_prompt_settings').onclick=openAPModal;$('ap_close').onclick=closeAPModal;$('ap_cancel').onclick=closeAPModal;
    $('auto_prompt_modal').addEventListener('click',e=>{if(e.target===$('auto_prompt_modal'))closeAPModal()});
    $('ap_model').onchange=()=>{$('ap_custom_wrap').style.display=$('ap_model').value==='__custom__'?'block':'none'};
    $('ap_profile_select').onchange=()=>applyAPProfile($('ap_profile_select').value);

    $('ap_profile_save').onclick=()=>{
      const name=$('ap_profile_name').value.trim(),model=$('ap_model').value,customModel=$('ap_custom_model').value.trim(),extra=$('ap_extra').value;
      if(!name){$('ap_key_status').className='apstatus err';$('ap_key_status').textContent='Give this instruction profile a name first.';return}
      if(model==='__custom__'&&!customModel){$('ap_key_status').className='apstatus err';$('ap_key_status').textContent='Enter a custom model ID first.';return}
      const profiles=getAPProfiles();let id=activeAPProfileId();const active=profiles.find(p=>p.id===id);
      if(!active||active.name!==name){const same=profiles.find(p=>p.name.toLowerCase()===name.toLowerCase());id=same?same.id:((crypto&&crypto.randomUUID)?crypto.randomUUID():`${Date.now()}-${Math.random()}`)}
      const row={id,name,model,customModel,extra,updated:Date.now()},idx=profiles.findIndex(p=>p.id===id);
      if(idx>=0)profiles[idx]=row;else profiles.push(row);
      profiles.sort((a,b)=>a.name.localeCompare(b.name));saveAPProfiles(profiles);saveAPSettings({model,customModel,extra});setActiveAPProfileId(id);syncAPProfileSelectors();
      $('ap_key_status').className='apstatus ok';$('ap_key_status').textContent=`✓ Saved instruction profile "${name}".`;
      $('auto_prompt_hint').innerHTML=`Auto Prompt instructions: <b>${esc(name)}</b> · model <b>${esc(model==='__custom__'?customModel:model)}</b>.`;
      say('Auto Prompt instruction profile saved · '+name);
    };

    $('ap_profile_delete').onclick=async()=>{
      const id=activeAPProfileId(),p=getAPProfiles().find(x=>x.id===id);if(!p)return;
      const ok=await uiConfirm(`Delete saved Auto Prompt instruction profile "${p.name}"?`,{title:'Delete Auto Prompt profile',confirmLabel:'Delete'});if(!ok)return;
      saveAPProfiles(getAPProfiles().filter(x=>x.id!==id));setActiveAPProfileId('');syncAPProfileSelectors();$('ap_profile_name').value='';
      $('ap_key_status').className='apstatus';$('ap_key_status').textContent='Profile deleted. Current instructions remain until you change them.';say('Auto Prompt profile deleted');
    };

    $('ap_save').onclick=()=>{
      const model=$('ap_model').value,customModel=$('ap_custom_model').value.trim(),extra=$('ap_extra').value;
      if(model==='__custom__'&&!customModel){$('ap_key_status').className='apstatus err';$('ap_key_status').textContent='Enter a custom model ID first.';return}
      saveAPSettings({model,customModel,extra});
      const active=getAPProfiles().find(p=>p.id===activeAPProfileId());
      if(active&&(active.model!==model||active.customModel!==customModel||active.extra!==extra))setActiveAPProfileId('');
      syncAPProfileSelectors();closeAPModal();
      $('auto_prompt_hint').innerHTML=`Auto Prompt model: <b>${esc(model==='__custom__'?customModel:model)}</b> · current instructions saved locally.`;
      say('Auto Prompt settings saved');
    };

    syncAPProfileSelectors();
    $('auto_prompt_btn').onclick=async()=>{
      const s=getAPSettings(),model=currentAPModel(),btn=$('auto_prompt_btn'),hint=$('auto_prompt_hint');
      if(!model){openAPModal();return}
      const fd=new FormData();
      fd.append('rough_prompt',$('prompt').value||'');
      fd.append('model',model);
      fd.append('extra_instructions',s.extra||'');
      for(const k of ['duration','width','height','playback_speed','use_stage_last'])fd.append(k,$(k).value);
      for(const k of ['first_frame','last_frame'])if($(k).files[0])fd.append(k,$(k).files[0]);
      btn.disabled=true;btn.classList.add('working');btn.textContent='✦ WRITING H3 PROMPT…';
      hint.textContent='Analyzing intent'+(($('first_frame').files[0]||$('last_frame').files[0]||$('use_stage_last').value==='1')?' + reference frame(s)':'')+'…';
      say('Auto Prompt · '+model);
      try{
        const resp=await fetch('/api/auto_prompt',{method:'POST',body:fd});
        const r=await resp.json();
        if(r.adult_ack_required){
          const ok=await requestAdultAcknowledgement();
          if(ok){setTimeout(()=>$('auto_prompt_btn').click(),0);return}
          hint.textContent='Adult Auto Prompt request cancelled; rough prompt preserved.';
          return;
        }
        if(r.error){
          hint.innerHTML=`<span style="color:#ff8181">${esc(r.error)}</span><br>Raw prompt preserved — GENERATE still sends it directly to local H3.`;
          say('Auto Prompt failed · raw prompt preserved');return
        }
        $('prompt').value=r.prompt||'';$('prompt').dispatchEvent(new Event('input',{bubbles:true}));
        const refs=[r.used_first_image?'FIRST':null,r.used_last_image?'LAST':null].filter(Boolean).join(' + ');
        hint.innerHTML=`✓ <b>${esc(r.model)}</b> · ${esc(String(r.mode||'').toUpperCase())}${refs?` · ${refs} vision reference${refs.includes(' + ')?'s':''} used`:''}`;
        say('Auto Prompt ready · prompt field populated')
      }catch(e){
        hint.innerHTML=`<span style="color:#ff8181">Auto Prompt request failed: ${esc(e)}</span><br>Raw prompt preserved — GENERATE remains local.`;
        say('Auto Prompt failed · raw prompt preserved')
      }finally{
        btn.disabled=false;btn.classList.remove('working');btn.textContent='✦ AUTO PROMPT'
      }
    };
    function syncStageClear(){
      const hasVideo=!!$('vwrap').querySelector('video');
      $('stage_controls').style.display=hasVideo?'flex':'none';
    }
    async function _waitForVideoMetadata(v){
      if(Number.isFinite(v.duration)&&v.duration>0&&v.videoWidth&&v.videoHeight)return;
      await new Promise((resolve,reject)=>{
        const done=()=>{cleanup();resolve()};
        const fail=()=>{cleanup();reject(new Error('Could not read staged video metadata.'))};
        const cleanup=()=>{v.removeEventListener('loadedmetadata',done);v.removeEventListener('error',fail)};
        v.addEventListener('loadedmetadata',done,{once:true});
        v.addEventListener('error',fail,{once:true});
        try{v.load()}catch(_){}
      });
    }
    async function stageLastFrameToFirst(){
      const v=$('vwrap').querySelector('video');
      if(!v){await uiAlert('Put a clip on the Stage first.','No staged clip');return}
      const btn=$('stage_last_to_first');
      const originalText=btn.textContent;
      btn.classList.add('working');btn.textContent='CAPTURING…';
      try{
        await _waitForVideoMetadata(v);
        const wasPaused=v.paused;
        const oldTime=Number.isFinite(v.currentTime)?v.currentTime:0;
        try{v.pause()}catch(_){}
        const frameStep=1/24;
        const target=Math.max(0,Number(v.duration)-frameStep);
        if(Math.abs(v.currentTime-target)>0.002){
          await new Promise((resolve,reject)=>{
            let settled=false;
            const finish=()=>{if(settled)return;settled=true;cleanup();resolve()};
            const fail=()=>{if(settled)return;settled=true;cleanup();reject(new Error('Could not seek to the last frame.'))};
            const cleanup=()=>{v.removeEventListener('seeked',finish);v.removeEventListener('error',fail)};
            v.addEventListener('seeked',finish,{once:true});
            v.addEventListener('error',fail,{once:true});
            v.currentTime=target;
          });
        }
        const canvas=document.createElement('canvas');
        canvas.width=v.videoWidth;canvas.height=v.videoHeight;
        const ctx=canvas.getContext('2d',{alpha:false});
        if(!ctx)throw new Error('Canvas capture is unavailable in this browser.');
        ctx.drawImage(v,0,0,canvas.width,canvas.height);
        const blob=await new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('Could not encode the captured frame.')),'image/png'));
        const file=new File([blob],`stage_last_${Date.now()}.png`,{type:'image/png'});
        const dt=new DataTransfer();dt.items.add(file);
        $('first_frame').files=dt.files;

        // The existing first-frame change handler updates the preview, aspect,
        // continuation flag, and generation payload exactly like a normal upload.
        $('first_frame').dispatchEvent(new Event('change',{bubbles:true}));
        if(currentModelMode()!=='fl2va')await setModelMode('fl2va');

        // Restore the viewer position so grabbing a frame does not wreck the Stage.
        try{
          if(Number.isFinite(oldTime)&&Math.abs(oldTime-target)>0.002)v.currentTime=Math.min(oldTime,v.duration||oldTime);
          if(!wasPaused){
            const playPromise=v.play();
            if(playPromise&&playPromise.catch)playPromise.catch(()=>{});
          }
        }catch(_){}
        say('stage last frame assigned to FIRST FRAME');
      }catch(e){
        await uiAlert(String(e&&e.message?e.message:e),'Could not capture last frame');
      }finally{
        btn.classList.remove('working');btn.textContent=originalText;
      }
    }
    $('stage_last_to_first').onclick=stageLastFrameToFirst;
    $('clear_stage').onclick=()=>{$('vwrap').innerHTML='<div id="empty">generated video appears here</div>';$('meta').textContent='';syncStageClear();say('stage cleared')};
    syncStageClear();

    function setPanelMin(panel,on){panel.classList.toggle('minimized',!!on)}
    $('history_hide').onclick=()=>{$('history_panel').style.display='none'};
    $('queue_hide').onclick=()=>{$('queue_panel').style.display='none'};
    $('show_history').onclick=()=>{setPanelMin($('history_panel'),false);$('history_panel').style.display='block';$('history_panel').style.zIndex=1210};
    $('show_queue').onclick=()=>{setPanelMin($('queue_panel'),false);$('queue_panel').style.display='block';$('queue_panel').style.zIndex=1210};
    $('show_console').onclick=()=>{$('consolebox').style.display='block';$('consolebox').style.zIndex=1220};
    $('history_expand').onclick=()=>{const p=$('history_panel');const big=p.dataset.big==='1';p.dataset.big=big?'0':'1';p.style.width=big?'':'min(500px,calc(100vw - 20px))';$('history_expand').textContent=big?'⛶ Expand':'↙ Normal'};
    function makeFloating(panel){
      const head=panel.querySelector('.floathead');let drag=null;
      head.addEventListener('pointerdown',e=>{if(e.target.closest('button'))return;const r=panel.getBoundingClientRect();drag={dx:e.clientX-r.left,dy:e.clientY-r.top};head.setPointerCapture(e.pointerId)});
      head.addEventListener('pointermove',e=>{if(!drag)return;const x=Math.max(4,Math.min(window.innerWidth-panel.offsetWidth-4,e.clientX-drag.dx));const y=Math.max(4,Math.min(window.innerHeight-panel.offsetHeight-4,e.clientY-drag.dy));panel.style.left=x+'px';panel.style.top=y+'px';panel.style.right='auto';panel.style.bottom='auto'});
      head.addEventListener('pointerup',e=>{drag=null;try{head.releasePointerCapture(e.pointerId)}catch(_){}});
    }
    makeFloating($('history_panel'));makeFloating($('queue_panel'));makeFloating($('consolebox'));

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
    $('minconsole').onclick=e=>{e.preventDefault();$('consolebox').style.display='none'};
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
        const finalSec=Math.max(.01,Number($('duration').value||7));
        f=snapFrames(finalSec*speed*24);
      }
      const modelSec=f/24, finalSec=modelSec/speed;
      const trained=f>=124&&f<=362;
      syncMotionPace();
      $('durhint').innerHTML=`→ H3 model: <b>${f} frames / ${modelSec.toFixed(2)}s</b> · final MP4 ≈ <b>${finalSec.toFixed(2)}s</b> · motion pace ${speed.toFixed(2)}×`+
        (trained?'':`<br><span style="color:#c9a227">outside the best-tested 124–362 frame range (≈5.2–15.1s model time); H3 accepts it, but quality/memory are less predictable.</span>`);
    }
    $('duration').addEventListener('input',()=>{$('length_mode').value='seconds';updateDur();});
    $('frames').addEventListener('input',()=>{$('length_mode').value='frames';updateDur();});
    $('playback_speed').addEventListener('input',updateDur);

    function updateSparseUI(source){
      let pct;
      if(source==='slider'){
        pct=Number($('sparse_slider').value||0);
        $('sparse_percent').value=pct;
      }else{
        pct=Math.max(0,Math.min(100,Number($('sparse_percent').value||0)));
        $('sparse_percent').value=pct;
        $('sparse_slider').value=pct;
      }
      const m=window.H3META||{};
      if(m.sparse_available===false){
        $('sparsehint').textContent='Sparse attention is unavailable in this session. Set this to 0% or restart V37.';
      }else if(pct<=0){
        $('sparsehint').innerHTML='<b>Dense / max quality.</b> Sparse attention is disabled.';
      }else{
        $('sparsehint').innerHTML=`<b>${pct.toFixed(1)}% video-attention budget.</b> Lower is faster; raise this if detail or motion quality drops. 0% = dense.`;
      }
    }
    $('sparse_slider').addEventListener('input',()=>updateSparseUI('slider'));
    $('sparse_percent').addEventListener('input',()=>updateSparseUI('number'));
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
    function releaseSlotObjectURL(img){const u=img.dataset.objectUrl;if(u){try{URL.revokeObjectURL(u)}catch(_){}delete img.dataset.objectUrl}}
    function clearImageSlot(kind,{keepContinuation=false}={}){
      const input=$(kind+'_frame'),img=$(kind+'_preview'),slot=$(kind+'_slot');
      releaseSlotObjectURL(img);input.value='';img.removeAttribute('src');slot.classList.remove('has-image');delete slot.dataset.source;
      if(kind==='first'){firstImageDims=null;if(!keepContinuation){stageButtonActive(false);refreshStageState(true)}}
    }
    function showImageModal(src){if(!src)return;$('image_view_full').src=src;$('image_view_modal').classList.add('show')}
    function closeImageModal(){$('image_view_modal').classList.remove('show');$('image_view_full').removeAttribute('src')}
    $('image_view_close').onclick=closeImageModal;$('image_view_modal').addEventListener('click',e=>{if(e.target===$('image_view_modal'))closeImageModal()});
    function bindImageSlot(kind,isFirst){
      const input=$(kind+'_frame'),img=$(kind+'_preview'),slot=$(kind+'_slot'),trash=$(kind+'_trash');
      function openOrPick(){if(slot.classList.contains('has-image')&&img.src)showImageModal(img.src);else input.click()}
      slot.addEventListener('click',e=>{if(e.target.closest('.slottrash'))return;openOrPick()});
      slot.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&!e.target.closest('.slottrash')){e.preventDefault();openOrPick()}});
      trash.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();clearImageSlot(kind);say(kind+' frame cleared')});
      input.addEventListener('change',()=>{
        const f=input.files[0];if(!f){clearImageSlot(kind,{keepContinuation:true});return}
        if(isFirst){stageButtonActive(false);refreshStageState(true)}
        releaseSlotObjectURL(img);const u=URL.createObjectURL(f);img.dataset.objectUrl=u;
        img.onload=()=>{slot.classList.add('has-image');slot.dataset.source='upload';if(isFirst){firstImageDims={w:img.naturalWidth,h:img.naturalHeight};if($('auto_aspect').checked)fitCanvas()}};
        img.src=u;
      });
    }
    bindImageSlot('first',true);bindImageSlot('last',false);
    function clearRefImageSlot(index){const input=$('ref_image_'+index),img=$('ref_image_preview_'+index),slot=$('ref_image_slot_'+index);releaseSlotObjectURL(img);input.value='';img.removeAttribute('src');slot.classList.remove('has-image');delete slot.dataset.source}
    function bindRefImageSlot(index){const input=$('ref_image_'+index),img=$('ref_image_preview_'+index),slot=$('ref_image_slot_'+index),trash=$('ref_image_trash_'+index);function openOrPick(){if(slot.classList.contains('has-image')&&img.src)showImageModal(img.src);else input.click()}slot.addEventListener('click',e=>{if(e.target.closest('.slottrash'))return;openOrPick()});slot.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&!e.target.closest('.slottrash')){e.preventDefault();openOrPick()}});trash.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();clearRefImageSlot(index);say('reference image '+index+' cleared')});input.addEventListener('change',()=>{const f=input.files[0];if(!f){clearRefImageSlot(index);return}releaseSlotObjectURL(img);const u=URL.createObjectURL(f);img.dataset.objectUrl=u;img.onload=()=>{slot.classList.add('has-image');slot.dataset.source='upload'};img.src=u})}
    function bindNamedFileInput(prefix,index,emptyLabel){const input=$(prefix+'_'+index),pick=$(prefix+'_pick_'+index),clear=$(prefix+'_clear_'+index),name=$(prefix+'_name_'+index);const sync=()=>{const f=input.files[0];name.textContent=f?f.name:emptyLabel;clear.disabled=!f};pick.onclick=e=>{e.preventDefault();input.click()};clear.onclick=e=>{e.preventDefault();input.value='';sync();say(prefix.replace('_',' ')+' '+index+' cleared')};input.addEventListener('change',sync);sync()}
    for(let i=1;i<=9;i++)bindRefImageSlot(i);
    for(let i=1;i<=3;i++){bindNamedFileInput('ref_video',i,'No video selected');bindNamedFileInput('ref_audio',i,'No audio selected')}
    function clearFirstPreview(){clearImageSlot('first',{keepContinuation:true})}
    function stageButtonActive(on){$('use_stage_last').value=on?'1':'0'}
    async function refreshStageState(quiet=false){
      try{const s=await (await fetch('/api/stage_state')).json();window.STAGE_STATE=s||{ok:false};return s}catch(e){if(!quiet)console.warn(e);return null}
    }

    let firstMeta=true;
    function loadMeta(){
      const prevSampler=$('sampler_name').value,prevScheduler=$('scheduler').value,prevUnet=$('unet').value;
      return fetch('/api/meta').then(r=>r.json()).then(m=>{
        window.H3META=m;
        window.ADULT_ENABLED=!!m.adult_enabled;
        syncAdultModeButton(m);
        if(firstMeta)ACTIVE_MODEL_PROFILE='stock_quality';

        $('sampler_name').innerHTML=(m.samplers||[]).map(s=>`<option>${esc(s)}</option>`).join('');
        $('scheduler').innerHTML=(m.schedulers||[]).map(s=>`<option>${esc(s)}</option>`).join('');
        if((m.samplers||[]).includes(prevSampler))$('sampler_name').value=prevSampler;
        else if((m.samplers||[]).includes(m.stock_quality_sampler||'res_multistep'))$('sampler_name').value=m.stock_quality_sampler||'res_multistep';
        if((m.schedulers||[]).includes(prevScheduler))$('scheduler').value=prevScheduler;
        else if((m.schedulers||[]).includes(m.stock_quality_scheduler||'simple'))$('scheduler').value=m.stock_quality_scheduler||'simple';

        syncModelProfileOptions(m);

        $('unet').innerHTML=(m.unets||[]).map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
        if(prevUnet&&(m.unets||[]).includes(prevUnet))$('unet').value=prevUnet;
        else if(m.unet_default&&(m.unets||[]).includes(m.unet_default))$('unet').value=m.unet_default;

        renderNamedLoraRows(m);

        if(firstMeta){
          // Startup defaults to the built-in model with creative LoRAs off.
          $('length_mode').value='seconds';
          ACTIVE_MODEL_PROFILE='stock_quality';
          $('model_profile_select').value='stock_quality';
          $('duration').value=m.lowvram_t4?5.17:7;
          $('frames').value=m.lowvram_t4?124:175;
          if(m.lowvram_t4){
            $('width').value=640;$('height').value=480;$('short_edge').value=480;
          }
          LORA_CARD_STATE.clear();
          _setStrengthForKind('motion8',0,m);
          $('playback_speed').value=1.0;$('denoise').value=1.0;
          if(m.lowvram_t4){
            // T4 stays on its lean Q4 profile; the BF16 Lightning LoRA is intentionally unavailable there.
            ACTIVE_PERF_PRESET='fast';
            _setStrengthForKind('lightning',0,m);
            $('steps').value=8;
            if((m.samplers||[]).includes('euler'))$('sampler_name').value='euler';
            if((m.schedulers||[]).includes('simple'))$('scheduler').value='simple';
            $('shift_video').value=12;
            $('shift_audio').value=3;
            say('ready · T4 LOW-VRAM · FAST dense fallback');
          }else{
            // Default setting requested: use the matching Lightning/Turbo LoRA.
            ACTIVE_PERF_PRESET='fast';
            _setStrengthForKind('lightning',m.lightning_strength_default||1.0,m);
            $('steps').value=4;
            if((m.samplers||[]).includes('euler'))$('sampler_name').value='euler';
            if((m.schedulers||[]).includes('simple'))$('scheduler').value='simple';
            $('shift_video').value=currentModelMode()==='ref2va'?12:6;
            $('shift_audio').value=3;
            say('ready · FAST · 4-step Lightning · dense');
          }
          $('sparse_percent').value=0;$('sparse_slider').value=0;updateSparseUI('number');
          updateDur();
          firstMeta=false;
        }

        $('tab_ref2va').disabled=!m.ref2va_available;
        if(!m.ref2va_available && currentModelMode()==='ref2va'){
          localStorage.setItem(MODEL_MODE_STORE_KEY,'fl2va');$('input_mode').value='fl2va';
        }
        syncModelModeUI();
        syncModelProfileUI();
        syncUnifiedLoraCompatibility();

        const arch=$('arch_label');
        if(arch)arch.textContent=(m.gpu_arch_label||m.gpu_profile||'AUTO GPU').toUpperCase();
        const mh=$('modelhint');
        if(mh){
          const profileBlurb=m.lowvram_t4
            ? '<b>T4 / 16GB LOW-VRAM STACK</b> · Q4_0 GGUF + Dynamic VRAM.'
            : (m.a100_80
              ? '<b>A100 80GB QUALITY STACK</b> · native SM80 quality path.'
              : (m.a100_40
                ? '<b>A100 40GB QUALITY STACK</b> · sequential TE→DiT→VAE handoff.'
                : '<b>BLACKWELL SM120 QUALITY STACK</b> · CUDA13 quality path.'));
          mh.innerHTML=profileBlurb
            + `<br><span style="color:#9aa0aa">Residency</span> · ${m.lowvram_t4?'DYNAMIC VRAM / CPU↔GPU paging':(m.full_stack_residency?'FULL STACK RESIDENT':'PARTITIONED TE↔DiT handoff')} · ${m.physical_vram_gib||'?'} GiB physical`
            + `<br><span style="color:#9aa0aa">Conditioning TE</span> · ${m.quality_text_encoder||'Qwen3-VL-32B'}`
            + '<br>' + (m.ref2va_available
              ? `<span style="color:#9aa0aa">Ref2VA available</span> · stock Ref2VA downloads on first use · ${m.ref2va_max_images||9} image / ${m.ref2va_max_videos||3} video / ${m.ref2va_max_audios||3} audio slots.`
              : '<span style="color:#d99191">Ref2VA unavailable</span> · update ComfyUI to expose MiniMaxH3ReferenceToVideo.');
        }
        return m;
      });
    }
    syncModelModeUI();
    loadMeta();
    refreshStageState(true);
    refreshTimeline(true);
    $('project_menu_btn').onclick=()=>{$('project_popover').classList.toggle('show')};
    $('project_pop_close').onclick=()=>{$('project_popover').classList.remove('show')};
    document.addEventListener('click',e=>{const p=$('project_popover');if(!p.classList.contains('show'))return;if(e.target.closest('#project_popover')||e.target.closest('#project_menu_btn'))return;p.classList.remove('show')});
    $('refresh').onclick=e=>{e.preventDefault();loadMeta().then(m=>
      say(`${Math.max(0,(m.loras||[]).length-1)} visible LoRA(s) found`))};

    function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
    function previewTimelineFile(file){
      if(!file)return;
      $('vwrap').innerHTML=`<video controls autoplay src="/out/${encodeURIComponent(file)}?t=${Date.now()}"></video>`;
      syncStageClear();
    }
    async function deleteTimelineClip(segmentId){
      if(!(await uiConfirm('Remove this clip from the active timeline? The render will remain available in History.',{title:'Remove clip',confirmLabel:'Remove',danger:true})))return;
      const r=await(await fetch('/api/timeline/segment/'+encodeURIComponent(segmentId),{method:'DELETE'})).json();
      if(r.error){fail(r.error);return}await refreshTimeline(true);say('clip removed and sequence restitched');
    }
    async function sendTimelineOrder(){
      const order=[...document.querySelectorAll('#timeline .tclip')].map(el=>el.dataset.segment);
      const r=await(await fetch('/api/timeline/reorder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({order})})).json();
      if(r.error){fail(r.error);return}await refreshTimeline(true);say('timeline reordered + restitched');
    }
    const timelineDropInflight=new Set();
    async function addHistoryToTimeline(historyId,index=null,dragId=null){
      const key=dragId||`${historyId}:${index===null?'end':index}`;
      if(timelineDropInflight.has(key))return;
      timelineDropInflight.add(key);
      try{
        const r=await(await fetch('/api/timeline/add_history',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({history_id:historyId,index,drag_id:dragId})
        })).json();
        if(r.error){fail(r.error);return}
        if(!r.duplicate_suppressed){
          await refreshTimeline(true);
          say('1 history clip added to active timeline');
        }
      }finally{
        timelineDropInflight.delete(key);
      }
    }
    function wireTimelineDnD(){
      const tl=$('timeline');
      let dragging=null;

      // Clip-level listeners live on freshly-rendered nodes, so these do not accumulate.
      [...tl.querySelectorAll('.tclip')].forEach(el=>{
        el.draggable=true;
        el.addEventListener('dragstart',e=>{
          dragging=el;
          el.classList.add('dragging');
          e.dataTransfer.effectAllowed='move';
          e.dataTransfer.setData('application/x-h3-segment',el.dataset.segment);
        });
        el.addEventListener('dragend',()=>{
          el.classList.remove('dragging');
          dragging=null;
          [...tl.querySelectorAll('.tclip')].forEach(x=>x.classList.remove('drop-before','drop-after'));
        });
        el.addEventListener('dragover',e=>{
          e.preventDefault();
          e.stopPropagation();
          const r=el.getBoundingClientRect();
          el.classList.toggle('drop-before',e.clientX<r.left+r.width/2);
          el.classList.toggle('drop-after',e.clientX>=r.left+r.width/2);
        });
        el.addEventListener('dragleave',()=>el.classList.remove('drop-before','drop-after'));
        el.addEventListener('drop',async e=>{
          e.preventDefault();
          e.stopPropagation();
          const r=el.getBoundingClientRect();
          const before=e.clientX<r.left+r.width/2;
          const hid=e.dataTransfer.getData('application/x-h3-history');
          const dragId=e.dataTransfer.getData('application/x-h3-drag-id')||null;
          if(hid){
            const clips=[...tl.querySelectorAll('.tclip')];
            const idx=Math.max(0,clips.indexOf(el)+(before?0:1));
            await addHistoryToTimeline(hid,idx,dragId);
            return;
          }
          if(dragging&&dragging!==el){
            el.parentNode.insertBefore(dragging,before?el:el.nextSibling);
            await sendTimelineOrder();
          }
        });
      });

      // IMPORTANT: these are property handlers, not addEventListener().
      // refreshTimeline() calls wireTimelineDnD repeatedly, and addEventListener()
      // used to stack duplicate drop callbacks on the persistent timeline node.
      tl.ondragover=e=>{
        if(e.dataTransfer && [...e.dataTransfer.types].includes('application/x-h3-history')){
          e.preventDefault();
          tl.classList.add('drop-target');
        }
      };
      tl.ondragleave=e=>{
        if(!tl.contains(e.relatedTarget))tl.classList.remove('drop-target');
      };
      tl.ondrop=async e=>{
        tl.classList.remove('drop-target');
        const hid=e.dataTransfer?.getData('application/x-h3-history');
        if(!hid)return;
        e.preventDefault();
        if(e.target.closest('.tclip'))return;
        const dragId=e.dataTransfer.getData('application/x-h3-drag-id')||null;
        await addHistoryToTimeline(hid,null,dragId);
      };
    }
    $('sequence_select').addEventListener('change',async e=>{
      if(!e.target.value)return;
      const r=await(await fetch('/api/timeline/select_sequence',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({sequence_id:e.target.value})
      })).json();
      if(r.error){fail(r.error);return}
      refreshTimeline(true);
      say('active sequence changed');
    });

    async function refreshTimeline(quiet=false){
      let t;
      try{t=await (await fetch('/api/timeline')).json()}catch(e){if(!quiet)console.warn(e);return null}
      window.H3TIMELINE=t;
      const segs=t.segments||[];const seqs=t.sequences||[];const active=t.active_sequence||{};
      if(document.activeElement!==$('project_name')) $('project_name').value=t.project_name||'Current Project';
      updateContinuityAvailability(); $('timeline_retry').disabled=!segs.length;
      $('timeline_compile').disabled=!segs.length;
      $('timeline_clear').disabled=!segs.length;
      const dl=$('project_download');if(t.project_file){dl.classList.remove('disabled');dl.dataset.href='/out/timeline_projects/'+encodeURIComponent(t.project_file)}else{dl.classList.add('disabled');delete dl.dataset.href}
      const ps=$('project_select'),saved=t.saved_projects||[];ps.innerHTML=saved.length?saved.map(p=>`<option value="${esc(p.project_file)}" ${p.project_file===t.project_file?'selected':''}>${esc(p.project_name||p.project_file)} · ${p.clip_count||0} clip(s) · ${(Number(p.total_duration)||0).toFixed(2)}s</option>`).join(''):'<option value="">No saved timelines yet</option>';
      $('timeline_project_hint').textContent=`${t.project_name||'Current Project'} · ${seqs.length} sequence${seqs.length===1?'':'s'} · ${Number(t.project_clip_count||0)} clips · autosaved`;
      $('timeline_foot').textContent=`Active: ${(active.name||'Sequence')} · ${segs.length} clip${segs.length===1?'':'s'} · ${(Number(t.total_duration)||0).toFixed(2)} s`+(t.master_file?` · master ${t.master_file}`:'');
      $('timeline_context').textContent=`${active.name||'Sequence'} · ${segs.length} clip${segs.length===1?'':'s'} · ${(Number(t.total_duration)||0).toFixed(2)} s`;
      const seqSelect=$('sequence_select');
      if(seqs.length>1){
        seqSelect.innerHTML=seqs.map((s,i)=>`<option value="${esc(s.id)}" ${s.active?'selected':''}>${i+1}. ${esc(s.name||'Sequence')}</option>`).join('');
        seqSelect.classList.add('show');
      }else{
        seqSelect.innerHTML='';
        seqSelect.classList.remove('show');
      }
      if(!segs.length){$('timeline').innerHTML='<div class=timelineempty>Drop a History clip here or generate the first clip.</div>';wireTimelineDnD();return t}
      $('timeline').innerHTML=segs.map((s,i)=>{const thumb=s.last_frame_file?`<img src="/out/${encodeURIComponent(s.last_frame_file)}?t=${t.updated||0}">`:'';const p=esc((s.prompt||'').slice(0,58));return `<div class=tclip data-segment="${esc(s.segment_id||s.job)}" data-file="${esc(s.file)}" title="Drag to move · ${esc(s.prompt||'')}"><span class=dragbadge>⠿ ${i+1}</span><div class=ttools><button class=recallprompt data-prompt="${esc(s.prompt||'')}" title="Restore full prompt">P</button><button class=tdelete data-segment="${esc(s.segment_id||s.job)}" title="Delete clip">⌫</button></div><div class=tthumb>${thumb}</div><div class=tinfo><b>${s.continued?'LAST-FRAME CONTINUITY':'SHOT'}</b> · ${Number(s.duration||0).toFixed(2)}s<br>${s.width||'?'}×${s.height||'?'} · seed ${s.seed??'?'}<br>${p||'—'}</div></div>`}).join('');
      [...document.querySelectorAll('.tclip')].forEach(el=>el.addEventListener('click',e=>{if(!e.target.closest('button'))previewTimelineFile(el.dataset.file)}));
      [...document.querySelectorAll('.tclip .recallprompt')].forEach(b=>b.onclick=e=>{e.stopPropagation();restoreClipPrompt(b.dataset.prompt||'')});
      [...document.querySelectorAll('.tdelete')].forEach(b=>b.onclick=e=>{e.stopPropagation();deleteTimelineClip(b.dataset.segment)});
      wireTimelineDnD();return t
    }

    function restoreClipPrompt(prompt){
      $('prompt').value=prompt||'';
      $('prompt').dispatchEvent(new Event('input',{bubbles:true}));
      $('prompt').focus();
      $('prompt').scrollIntoView({behavior:'smooth',block:'center'});
      say('clip prompt restored to editor');
    }

    async function refreshHistory(){
      try{const d=await(await fetch('/api/history',{cache:'no-store'})).json();const rows=d.items||[];$('history_count').textContent=rows.length; $('history_launch_count').textContent=rows.length?rows.length:'';
        if(!rows.length){$('history_body').innerHTML='<div class=floatempty>No completed generations yet.</div>';return}
        $('history_body').innerHTML=rows.map(h=>{const thumb=h.last_frame_file?`<img src="/out/${encodeURIComponent(h.last_frame_file)}?t=${h.created||0}">`:'';return `<article class=historyitem draggable=true data-history="${esc(h.history_id)}" data-file="${esc(h.file)}"><div class=histthumb>${thumb}</div><div class=histmain><div class=histstatus>✅ Done</div><div class=histsub>${Number(h.duration||0).toFixed(2)}s · ${h.width||'?'}×${h.height||'?'} · seed ${h.seed??'?'}</div><div class=addhist>drag to timeline · double-click to preview</div></div><button class=recallprompt data-prompt="${esc(h.prompt||'')}" title="Restore this clip's full prompt">PROMPT</button><button class=trash data-history="${esc(h.history_id)}" title="Remove from history">⌫</button></article>`}).join('');
        [...document.querySelectorAll('.historyitem')].forEach(el=>{
          el.addEventListener('dragstart',e=>{
            const dragId=(crypto&&crypto.randomUUID)?crypto.randomUUID():`${Date.now()}-${Math.random()}`;
            e.dataTransfer.effectAllowed='copy';
            e.dataTransfer.setData('application/x-h3-history',el.dataset.history);
            e.dataTransfer.setData('application/x-h3-drag-id',dragId);
          });
          el.addEventListener('dblclick',()=>previewTimelineFile(el.dataset.file));
        });
        [...document.querySelectorAll('.historyitem .recallprompt')].forEach(b=>b.onclick=e=>{e.stopPropagation();restoreClipPrompt(b.dataset.prompt||'')});
        [...document.querySelectorAll('.historyitem .trash')].forEach(b=>b.onclick=async e=>{e.stopPropagation();await fetch('/api/history/'+encodeURIComponent(b.dataset.history),{method:'DELETE'});refreshHistory()});
      }catch(e){}
    }
    async function refreshQueue(){
      try{
        const d=await(await fetch('/api/queue',{cache:'no-store'})).json();
        const rows=d.items||[];
        activeQueueJob=d.active||null;
        $('queue_count').textContent=rows.length?`${d.active?'1':'0'}/${rows.length}`:'0';
        $('queue_launch_count').textContent=rows.length?`(${rows.length})`:'';

        const active=rows.find(q=>q.status==='running');
        if(active){
          const stage=active.cancel_requested?'stopping…':(active.stage_label||active.stage||'running');
          const step=(active.stage==='sampling'&&Number(active.sample_steps||0)>0)
            ? ` · step ${Number(active.sample_step||0)}/${Number(active.sample_steps||0)}`
            : '';
          const eta=(active.stage==='sampling'&&active.sample_eta!=null)
            ? ` · ETA ~${Number(active.sample_eta)}s`
            : '';
          const gpu=active.gpu_busy
            ? ` · GPU ${Number(active.gpu_util||0).toFixed(0)}%`
            : '';
          const waiting=Number(d.queued||0)>0?` · ${Number(d.queued)} waiting`:'';
          dot(active.possible_stall?'err':'live');
          say(`${stage}${step} · stage ${Number(active.stage_elapsed||0)}s · total ${Number(active.elapsed||0)}s${eta}${gpu}${waiting}`);
          $('pb').style.width=Math.max(0,Math.min(100,Number(active.pipeline_pct||0)))+'%';
        }else if(Number(d.queued||0)>0){
          if(d.worker_alive===false){
            dot('err');
            say(`QUEUE WORKER STOPPED · ${Number(d.queued)} waiting`);
          }else{
            dot('live');
            say(`queue starting · ${Number(d.queued)} waiting`);
          }
          $('pb').style.width='0%';
        }

        if(!rows.length){
          $('queue_body').innerHTML='<div class=floatempty>Queue is empty. Generate repeatedly to stack jobs.</div>';
          return
        }

        $('queue_body').innerHTML=rows.map((q,i)=>{
          const thumb=q.thumb_file?`<img src="/out/${encodeURIComponent(q.thumb_file)}">`:'';
          const pct=Math.max(0,Math.min(100,Number(q.pipeline_pct||0)));
          const stage=q.status==='running'?(q.cancel_requested?'stopping…':(q.stage_label||q.stage||'running')):'queued';

          let health='';
          if(q.status==='running'){
            health=q.possible_stall
              ? '<span class=queuehealth warn>CHECK</span>'
              : q.gpu_busy
                ? `<span class=queuehealth busy>GPU ${Number(q.gpu_util||0).toFixed(0)}%</span>`
                : '<span class=queuehealth>ACTIVE</span>';
          }

          let detail='';
          if(q.status==='running'){
            const sample=(q.stage==='sampling'&&Number(q.sample_steps||0)>0)
              ? ` · step ${Number(q.sample_step||0)}/${Number(q.sample_steps||0)}`
              : '';
            const eta=(q.stage==='sampling'&&q.sample_eta!=null)
              ? ` · ETA ~${Number(q.sample_eta)}s`
              : '';
            detail=`<b>${esc(stage)}</b>${sample} · stage ${Number(q.stage_elapsed||0)}s · total ${Number(q.elapsed||0)}s${eta}`;
          }else{
            detail='waiting for active GPU job';
          }

          const action=q.status==='queued'
            ? `<button class=cancel data-job="${q.id}" title="Cancel this queued generation">CANCEL</button>`
            : `<button class=stoprun data-job="${q.id}" ${q.cancel_requested?'disabled':''} title="Stop at the next safe sampler/stage boundary">${q.cancel_requested?'STOPPING…':'STOP'}</button>`;

          return `<article class=queueitem><div class=qthumb>${thumb}</div><div class=qmain><div class=qstatus>${q.status==='running'?'⚙ Running':'⌛ Queued'} <span class=queuebadge>${i+1}/${rows.length}</span> ${health}</div><div class=qsub>${detail}<br>${esc((q.prompt||'').slice(0,64))||'generation'} · output ${Number(q.duration||0).toFixed(1)}s</div><div class=queueprogress title="${pct.toFixed(0)}% overall pipeline"><i style="width:${pct}%"></i></div></div>${action}</article>`;
        }).join('');

        [...document.querySelectorAll('.queueitem .cancel,.queueitem .stoprun')].forEach(b=>b.onclick=async()=>{
          b.disabled=true;
          b.textContent=b.classList.contains('stoprun')?'STOPPING…':'CANCELLING…';
          const r=await(await fetch('/api/queue/'+encodeURIComponent(b.dataset.job),{method:'DELETE'})).json();
          if(r.error)await uiAlert(r.error,'Queue');
          else say(r.running?'stop requested for running generation':'queued generation cancelled');
          refreshQueue();
        });
      }catch(e){}
    }

    $('queue_clear').onclick=async()=>{await fetch('/api/queue/clear',{method:'POST'});refreshQueue();say('all waiting jobs cancelled · running job left alone')};
    setInterval(refreshQueue,900);setInterval(refreshHistory,1800);refreshQueue();refreshHistory();

    async function applyPerformancePreset(kind){
      const m=window.H3META||{};
      const profile=ACTIVE_MODEL_PROFILE;
      const mode=currentModelMode();
      const accelerated=(kind==='fast'||kind==='ultra');

      if(accelerated && m.motion8_file && Math.abs(_strengthForKind('motion8',m))>1e-6){
        await uiAlert('FAST / ULTRA FAST use the 4-step Lightning accelerator, while Motion Enhancer is an alternative acceleration LoRA. Turn Motion Enhancer OFF first.','Preset conflict');
        return;
      }

      $('denoise').value=1.0;
      $('playback_speed').value=1.0;

      const cm=customModelForProfile(profile,m);
      if(cm&&cm.local_name&&[...$('unet').options].some(x=>x.value===cm.local_name))$('unet').value=cm.local_name;
      else if(!cm){
        const target=mode==='ref2va'?m.stock_ref2va_unet:m.base_fl2va_unet;
        if(target&&[...$('unet').options].some(x=>x.value===target))$('unet').value=target;
      }

      if(accelerated){
        const accelAvailable=mode==='ref2va'?m.ref2va_lightning_available:m.lightning_available;
        if(!accelAvailable){
          // Low-VRAM/T4 currently has no matching Lightning package. Keep FAST selected
          // but use the best available dense base recipe instead of silently breaking.
          _setStrengthForKind('lightning',0,m);
          $('steps').value=8;
          if([...$('sampler_name').options].some(x=>x.value==='euler'))$('sampler_name').value='euler';
          if([...$('scheduler').options].some(x=>x.value==='simple'))$('scheduler').value='simple';
          $('shift_video').value=12;$('shift_audio').value=3;
          $('sparse_percent').value=0;$('sparse_slider').value=0;updateSparseUI('number');
          if(kind==='ultra')await uiAlert('ULTRA FAST requires the 4-step Lightning LoRA, which is unavailable on this runtime. FAST dense fallback was applied instead.','Ultra Fast unavailable');
          kind='fast';
        }else{
          _setStrengthForKind('lightning',m.lightning_strength_default||1.0,m);
          $('steps').value=4;
          if([...$('sampler_name').options].some(x=>x.value==='euler'))$('sampler_name').value='euler';
          if([...$('scheduler').options].some(x=>x.value==='simple'))$('scheduler').value='simple';
          $('shift_video').value=mode==='ref2va'?12:6;
          $('shift_audio').value=3;
          const sparse=kind==='ultra'?5:0;
          $('sparse_percent').value=sparse;$('sparse_slider').value=sparse;updateSparseUI('number');
        }
      }else{
        _setStrengthForKind('lightning',0,m);
        $('steps').value=m.stock_quality_steps||20;
        if([...$('sampler_name').options].some(x=>x.value===(m.stock_quality_sampler||'res_multistep')))$('sampler_name').value=m.stock_quality_sampler||'res_multistep';
        if([...$('scheduler').options].some(x=>x.value===(m.stock_quality_scheduler||'simple')))$('scheduler').value=m.stock_quality_scheduler||'simple';
        $('shift_video').value=m.stock_quality_shift_video??12;$('shift_audio').value=m.stock_quality_shift_audio??3;
        $('sparse_percent').value=0;$('sparse_slider').value=0;updateSparseUI('number');
        if(mode==='ref2va')$('ref_image_size').value='max';
      }

      ACTIVE_PERF_PRESET=kind;
      updateDur();syncPresetUI();renderNamedLoraRows(window.H3META||{});
      const r=presetRecipe(profile,mode,kind);
      say(`${r.label} · ${kind==='ultra'?'ULTRA FAST':kind.toUpperCase()} · ${r.summary}`);
    }

    $('fastpreset').onclick=e=>{e.preventDefault();applyPerformancePreset('fast')};
    $('ultrafastpreset').onclick=e=>{e.preventDefault();applyPerformancePreset('ultra')};
    $('qualitypreset').onclick=e=>{e.preventDefault();applyPerformancePreset('quality')};

    $('reset_loras').onclick=e=>{e.preventDefault();resetNamedLoras();say('LoRA rack reset')};



    function fail(m){dot('err');say('failed');$('err').style.display='block';
      $('err').textContent=m;$('go').disabled=false;$('pb').style.width='0';refreshTimeline(true)}

    async function submitGeneration(){
      const mode=currentModelMode();
      const segs=((window.H3TIMELINE||{}).segments)||[];
      const usePrevious=mode==='fl2va' && $('continuity_enabled').checked && segs.length>0;
      const timelineAction=usePrevious?'continue':'new';
      if(!usePrevious)stageButtonActive(false);

      if(usePrevious && $('continuity_keep_seed').checked){
        const prevSeed=Number(segs[segs.length-1]?.seed);
        if(Number.isFinite(prevSeed))$('seed').value=String(prevSeed);
      }

      $('err').style.display='none';const fd=new FormData();
      for(const k of ['prompt','width','height','duration','frames','length_mode','playback_speed','image_fit','steps','seed','denoise','shift_video','shift_audio','sparse_percent','sampler_name','scheduler','weight_dtype','unet','use_stage_last'])fd.append(k,$(k).value);
      const motion8Submit=submittedSpecialStrength('motion8'),lightningSubmit=submittedSpecialStrength('lightning');
      fd.append('motion8_strength',String(motion8Submit));fd.append('lightning_strength',String(lightningSubmit));
      fd.append('lora','none');fd.append('lora_strength','0');fd.append('lora_stack_json',JSON.stringify(activeCreativeLoraStack()));
      fd.append('timeline_action',timelineAction);fd.append('input_mode',mode);fd.append('ref_image_size',$('ref_image_size').value);fd.append('action','0');fd.append('action_strength','0');fd.append('lightning',Math.abs(lightningSubmit)>1e-6?'1':'0');
      if(mode==='ref2va'){
        for(let i=1;i<=9;i++) if($('ref_image_'+i).files[0]) fd.append('ref_image_'+i,$('ref_image_'+i).files[0]);
        for(let i=1;i<=3;i++) if($('ref_video_'+i).files[0]) fd.append('ref_video_'+i,$('ref_video_'+i).files[0]);
        for(let i=1;i<=3;i++) if($('ref_audio_'+i).files[0]) fd.append('ref_audio_'+i,$('ref_audio_'+i).files[0]);
      }else{
        for(const k of ['first_frame','last_frame'])if($(k).files[0])fd.append(k,$(k).files[0]);
      }
      dot('live');say(timelineAction==='continue'?'adding generation with previous last-frame continuity':(mode==='ref2va'?'adding Ref2VA generation to queue':'adding generation to queue'));
      const resp=await fetch('/api/generate',{method:'POST',body:fd});
      const r=await resp.json();
      if(r.adult_ack_required){
        dot('');say('adult acknowledgement required');
        const ok=await requestAdultAcknowledgement();
        if(ok){
          say('adult catalog unlocked · choose the model/LoRAs you want, then add the generation again');
          return;
        }
        say('adult request cancelled');
        return;
      }
      if(r.error){fail(r.error);refreshTimeline(true);return}
      job=r.id;poll(job);refreshQueue();
    }
    $('go').onclick=()=>submitGeneration();
    $('timeline_compile').onclick=async()=>{
      $('err').style.display='none';
      const b=$('timeline_compile');
      b.disabled=true;
      const old=b.textContent;
      b.textContent='COMPILING…';
      say('compiling active timeline');
      try{
        const r=await(await fetch('/api/timeline/compile_active',{method:'POST'})).json();
        if(r.error){fail(r.error);return}
        $('vwrap').innerHTML=`<video controls autoplay src="/out/${encodeURIComponent(r.file)}?t=${Date.now()}"></video>`;
        $('meta').textContent=`COMPILED · ${r.clip_count} clip${r.clip_count===1?'':'s'} · ${r.file}`;
        syncStageClear();
        say(`compiled ${r.clip_count} timeline clip${r.clip_count===1?'':'s'} · ${r.note||'ready'}`);
        await refreshTimeline(true);
      }finally{
        b.textContent=old;
        b.disabled=!(window.H3TIMELINE?.segments||[]).length;
      }
    };

    $('timeline_retry').onclick=async()=>{
      $('err').style.display='none';
      dot('live');say('adding retry to queue with a new seed');
      const resp=await fetch('/api/timeline/retry_last',{method:'POST'});
      const r=await resp.json();
      if(r.adult_ack_required){
        dot('');const ok=await requestAdultAcknowledgement();
        if(ok)return $('timeline_retry').click();
        say('adult retry cancelled');return;
      }
      if(r.error){fail(r.error);refreshTimeline(true);return}
      $('seed').value=r.seed;
      job=r.id;poll(job);refreshQueue();
    };
    $('timeline_new_sequence').onclick=async()=>{
      const name=await uiPrompt('Name this sequence.',`Sequence ${((window.H3TIMELINE?.sequences||[]).length||0)+1}`,{title:'New sequence',confirmLabel:'Create'});
      if(name===null)return;
      const r=await(await fetch('/api/timeline/new_sequence',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
      if(r.error){fail(r.error);return}
      stageButtonActive(false);
      await refreshTimeline(true);
      say('new active sequence created');
    };
    $('timeline_duplicate').onclick=async()=>{
      const r=await(await fetch('/api/timeline/duplicate_active',{method:'POST'})).json();
      if(r.error){fail(r.error);return}
      await refreshTimeline(true);
      say('active sequence duplicated');
    };
    $('timeline_clear').onclick=async()=>{
      if(!(await uiConfirm('Clear every clip from the active timeline? Rendered clips will remain available in History.',{title:'Clear timeline',confirmLabel:'Clear timeline',danger:true})))return;
      const r=await(await fetch('/api/timeline/clear',{method:'POST'})).json();
      if(r.error){fail(r.error);return}
      stageButtonActive(false);
      await refreshTimeline(true);
      say('timeline cleared');
    };
    $('project_save').onclick=async()=>{
      const r=await(await fetch('/api/timeline/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_name:$('project_name').value||'Current Project'})})).json();
      if(r.error){fail(r.error);return}
      await refreshTimeline(true);
      say('timeline project saved');
    };
    $('project_load').onclick=async()=>{
      const project_file=$('project_select').value;
      if(!project_file){await uiAlert('Choose a saved project first.','Load project');return}
      const r=await(await fetch('/api/timeline/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_file})})).json();
      if(r.error){fail(r.error);return}
      stageButtonActive(false);
      await refreshTimeline(true);
      say('saved timeline loaded');
    };
    $('project_refresh').onclick=async()=>{await refreshTimeline(true);say('saved timeline list refreshed')};
    $('project_download').onclick=()=>{const u=$('project_download').dataset.href;if(u)window.open(u,'_blank')};

    async function poll(jid){
      const j=await(await fetch('/api/job/'+jid)).json();

      // Queue/run status is rendered centrally by refreshQueue(). Do not let the
      // newest submitted waiting jid overwrite the active job with "queued …".
      if(j.status==='queued'||j.status==='running'){
        setTimeout(()=>poll(jid),1300);
        return
      }

      if(j.status==='done'){
        job=jid;
        dot('');
        say(`done in ${j.secs}s · measured ${j.measured_total_sec||j.secs}s · sample ${j.sampling_sec||'?'}s · stitch ${j.stitch_sec||0}s · ${j.residency_mode||''} · ${(j.active_sequence_name||'sequence')} · output ${j.duration}s · ${j.frames} frames`);
        $('pb').style.width='100%';
        const previewFile=j.file;
        $('vwrap').innerHTML=`<video controls autoplay src="/out/${previewFile}?t=${Date.now()}"></video>`;
        syncStageClear();
        $('meta').textContent=j.file+(j.active_sequence_name?` · ${j.active_sequence_name}`:'')+(j.note?` · ${j.note}`:'');
        await refreshTimeline(true);
        await refreshHistory();
        await refreshQueue();
        return
      }

      if(j.status==='cancelled'){
        await refreshQueue();
        return
      }

      if(jid===activeQueueJob||!activeQueueJob)fail(j.msg||'unknown error');
      refreshQueue();
    }
    </script></body></html>"""

    @app.get("/")
    def index(): return Response(PAGE, mimetype="text/html")

    # ── 7. Serve + launch ──────────────────────────────────────────────────────
    if os.environ.get("H3_UI_BLOCKING_CHILD") == "1":
        log("="*74)
        log(f"🚀 V86 isolated CU130/Sage UI is READY on port {UI_PORT}")
        log("  Flask is running BLOCKING in the child process so it stays alive.")
        log("="*74)
        app.run(host="0.0.0.0", port=UI_PORT, threaded=True, use_reloader=False)
    else:
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
                UI_PORT, anchor_text="◤ Open MissingLink MiniMax Studio · Fast in a new tab")
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
        log(f"  ✓ Stock H3 default preloaded; conditioning TE: {TEXT_ENCODER_FILE}.")
    else:
        if LOWVRAM_T4_PROFILE:
            log(f"  ✓ T4/LOW-VRAM on-demand model: {T4_DIT_FILE} · Dynamic VRAM · no startup preload by design.")
        elif FAST_STARTUP:
            log(f"  ✓ FAST STARTUP active · {GPU_PROFILE.upper()} · model loading deferred until first GENERATE.")
        elif A100_PROFILE:
            log(f"  ✓ {GPU_PROFILE.upper()} active · native SM80 runtime · "
                + ("full-card quality residency" if FULL_STACK_RESIDENCY else "safe TE↔DiT↔VAE handoff"))
            log("  Startup preload did not complete; first GENERATE will retry model loading.")
        else:
            log("  Startup preload failed; first GENERATE will retry model loading.")
    log("  Errors come back as a full traceback in the red panel.")
    log("="*74)