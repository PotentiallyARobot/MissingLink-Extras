# ============================================================
#  DiT STUDIO  —  single-cell Colab launcher
#  Qwen-Image 2512 (default)  •  FLUX.1-dev  •  Z-Image Turbo
#  text-to-image + image-to-image, runtime LoRA support
#  (add from URL or the Civitai browser, control strengths live)
#
#  HOW TO USE:
#    1. Runtime -> Change runtime type -> A100 GPU. On an 80 GB A100
#       everything runs fully in VRAM in bf16 (Qwen-Image-2512 ~58 GB,
#       FLUX.1-dev ~34 GB, Z-Image Turbo ~20 GB). On a 40 GB A100 (what
#       Colab usually hands out) the big transformers are auto-loaded in
#       4-bit NF4 + CPU offload instead, so Qwen still works.
#    2. Sidebar -> Secrets: add CIVITAI_API_KEY (toggle "Notebook
#       access" ON). Needed for Civitai LoRA/checkpoint downloads.
#       Add HF_TOKEN too if you want FLUX.1-dev — it's a GATED repo:
#       accept the license on its HuggingFace page first.
#    3. Paste this whole cell into Colab and run it.
#    4. Click the link it prints to open the studio UI.
#    5. txt2img: write a prompt, Generate.
#       img2img: upload an image, set strength, Generate.
#    6. Add LoRAs at runtime: paste a HuggingFace or Civitai
#       .safetensors download URL into "Add LoRA".
# ============================================================

import os, sys, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Installing dependencies (~2 min first run)...")
# Core libs. torchao pinned >=0.16.0 per requirement (older wheels
# trip a Colab/torch compatibility complaint).
# NOTE: do NOT upgrade Pillow here. Colab's preinstalled torchvision is
# built against the resident Pillow; pulling a newer Pillow breaks
# torchvision's import (cannot import name '_Ink'), which cascades into
# transformers/diffusers. PIL is already available, so we leave it alone.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "diffusers>=0.36.0", "transformers", "accelerate",
                "safetensors", "sentencepiece", "peft", "flask",
                "bitsandbytes", "lycoris-lora>=2.2.0", "gguf>=0.10.0",
                "torchao>=0.16.0"],
               check=True)

import io, time, json, uuid, base64, threading, traceback
import collections, re
import socket as _socket
import requests as _requests
from urllib.parse import quote as _url_quote
import torch
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, Response, send_file

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

# ---- LoRA backend probe -------------------------------------------------
# The 'Target modules not found' LoRA failures are usually a PEFT-availability
# problem: diffusers only uses its proper LoRA injection path when
# is_peft_available() AND a recent-enough peft version are present. If that
# check is false at runtime, it falls back to a path that errors on exactly
# the SDXL LoRAs we're loading. Print the ground truth so we stop guessing.
print("Checking LoRA backend (peft/diffusers)...")
try:
    import diffusers as _dfx
    print(f"  diffusers {_dfx.__version__}")
except Exception as e:
    print(f"  diffusers import issue: {e}")
try:
    import peft as _peft
    print(f"  peft {_peft.__version__} imported OK")
except Exception as e:
    print(f"  peft FAILED to import: {type(e).__name__}: {e}")
try:
    from diffusers.utils import is_peft_available as _ipa
    print(f"  diffusers.is_peft_available() -> {_ipa()}")
    try:
        from diffusers.utils import is_peft_version as _ipv
        print(f"  peft >= 0.13.1 -> {_ipv('>=', '0.13.1')}")
    except Exception as e:
        print(f"  is_peft_version check unavailable ({e})")
except Exception as e:
    print(f"  could not query is_peft_available ({e})")

IN_COLAB = False
try:
    from google.colab.output import eval_js
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    eval_js = None
    userdata = None

def _get_secret(key, verbose=True):
    """Read a Colab secret, reporting clearly *why* it is unavailable."""
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
                      "OFF — open the Secrets panel and enable the toggle.")
            elif "SecretNotFound" in name:
                print(f"  secret {key}: not set — add it in the Secrets panel.")
            else:
                print(f"  secret {key}: unavailable ({name}: {e})")
        return None

# ---- API keys ----------------------------------------------------------
print("Resolving API keys...")
CIVITAI_API_KEY = _get_secret("CIVITAI_API_KEY")
HF_TOKEN = _get_secret("HF_TOKEN", verbose=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- models ------------------------------------------------------------
# Three DiT (flow-matching) families. One model is resident at a time, fully
# in VRAM in bf16 on the 80 GB A100; swapping frees the old one first.
MODEL_REGISTRY = {
    "qwen": {
        "label": "Qwen-Image 2512",
        "repo": "Qwen/Qwen-Image-2512",
        "civitai_base": "Qwen",
        "bf16_gb": 64,    # min total VRAM for full bf16 residency
        # Community GGUF quant for low-VRAM cards. Q4_K keeps sensitive
        # tensors (modulation, in/out projections) at higher precision —
        # unlike naive bnb NF4 over every linear, which visibly corrupts
        # DiT outputs (uniform stipple).
        "gguf": {"label": "Q4_K_M (unsloth)", "size_gb": 13.2,
                 "fname": "qwen_image_2512_Q4_K_M.gguf",
                 "url": ("https://huggingface.co/unsloth/Qwen-Image-2512-GGUF"
                         "/resolve/main/qwen-image-2512-Q4_K_M.gguf")},
        "defaults": {"steps": 40, "guidance": 4.0, "supports_negative": True,
                     "guidance_hint": ("Qwen-Image uses TRUE CFG — ~4.0 is the "
                                       "sweet spot; the negative prompt works.")},
    },
    "flux": {
        "label": "FLUX.1-dev",
        "repo": "black-forest-labs/FLUX.1-dev",
        "civitai_base": "Flux.1 D",
        "bf16_gb": 38,
        "gguf": {"label": "Q4_K_S (city96)", "size_gb": 6.8,
                 "fname": "flux1_dev_Q4_K_S.gguf",
                 "url": ("https://huggingface.co/city96/FLUX.1-dev-gguf"
                         "/resolve/main/flux1-dev-Q4_K_S.gguf")},
        "defaults": {"steps": 28, "guidance": 3.5, "supports_negative": False,
                     "guidance_hint": ("FLUX.1-dev uses embedded (distilled) "
                                       "guidance ~3.5. It has NO CFG pass, so "
                                       "the negative prompt is ignored.")},
    },
    "zimage": {
        "label": "Z-Image Turbo",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "civitai_base": "ZImageTurbo",
        "bf16_gb": 24,
        "defaults": {"steps": 8, "guidance": 1.0, "supports_negative": True,
                     "guidance_hint": ("Z-Image Turbo is step-distilled: ~8 "
                                       "steps, guidance 1 (CFG effectively "
                                       "off; negative prompt does little).")},
    },
}
DEFAULT_MODEL_KEY = "qwen"            # Qwen-Image 2512 loads by default
BASE_MODEL = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["repo"]
DTYPE = torch.bfloat16                # all three recommend bf16; A100-native

# LoRAs to download + attach automatically at startup. Each entry is either a
# plain download-URL string, or a dict {"url", "name", "scale", "triggers"}.
# The Civitai API is queried for each (name, base model, trigger words), and
# the FIRST entry's base-model family decides which base model boots — so a
# Qwen LoRA here makes the studio start on the matching Qwen pipeline.
# civitai.red mirror URLs work; the CIVITAI_API_KEY is attached either way.
STARTUP_LORAS = [
    "https://civitai.red/api/download/models/2328988?fileId=2219270",
]

def _detect_arch(*hints):
    """Map a repo id / URL / Civitai base-model string / label to an arch key.
    Returns 'qwen' | 'flux' | 'zimage' | None. 'Flux.2 D' is deliberately
    unsupported (different architecture than FLUX.1)."""
    for sshint in hints:
        t = (sshint or "").lower().replace(" ", "").replace("-", "")
        t = t.replace(".", "").replace("_", "")
        if not t:
            continue
        if "qwen" in t:
            return "qwen"
        if "zimage" in t:
            return "zimage"
        if "flux2" in t:
            return None
        if "flux" in t:
            return "flux"
    return None

# Web image search source for the img2img "find a source image" feature.
# The page is fetched server-side (avoids browser CORS) and image URLs are
# parsed out of its <img> tags (data-src preferred — the markup lazy-loads,
# so src is usually a 1px placeholder). {q} is replaced with the URL-encoded
# query. Edit this to point at the search site you want.
# Default web image search source for img2img. Can be overridden per-search
# from the UI. Use the literal 'duckduckgo' to use DuckDuckGo's image API, or a
# URL template containing {q} for a generic site whose results are <img> tags.
WEB_IMAGE_SEARCH_URL = "best"

# Optional proxy for web-image fetches. Some image CDNs (e.g. PornPics,
# ImageFap) IP-block datacenter ranges like Colab's — proven the block is on
# the IP/ASN, not the request (even a real headless browser gets 403). The
# ONLY fix is routing the fetch through a residential / non-flagged IP. Set
# this to a proxy URL to do that, e.g.:
#   WEB_IMAGE_PROXY = "http://user:pass@host:port"
#   WEB_IMAGE_PROXY = "socks5://user:pass@host:port"   (needs pip pysocks)
# Leave as "" to fetch directly (fine for boorus, which aren't IP-blocked).
# You can also set it at runtime via the WEB_IMAGE_PROXY environment variable.
WEB_IMAGE_PROXY = os.environ.get("WEB_IMAGE_PROXY", "")

# Residency: always try full GPU first (this is built for an 80 GB A100);
# only fall back to CPU offload if the load actually OOMs.

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
    "txt_pipe": None,            # active txt2img pipeline (Qwen/Flux/ZImage)
    "img_pipe": None,            # img2img view sharing the same modules
    "residency": "unknown",
    "arch": DEFAULT_MODEL_KEY,       # 'qwen' | 'flux' | 'zimage'
    "model_name": MODEL_REGISTRY[DEFAULT_MODEL_KEY]["label"],
    "model_ref": BASE_MODEL,         # repo id or local single-file path
    "loras": {},                 # name -> {path, scale, attached}
    "lock": threading.Lock(),    # one GPU job at a time
    "load_lock": threading.Lock(),
    "swap": {"busy": False, "stage": "", "error": None, "result": None},
}
jobs = {}

class _JobCancelled(Exception):
    pass

# ---- download helpers --------------------------------------------------
def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0

def download_safetensors(url, dest_name, out_dir, min_bytes=4096,
                         progress_cb=None):
    """Download a .safetensors file from a HuggingFace or Civitai URL, with
    live progress (percent, speed, ETA) logged to the console roughly every
    2 s. progress_cb(dict) is also called for routing into the swap status.

    Civitai redirects to a CDN that drops the Authorization header, so the
    token is also sent as a ?token= query param to survive the redirect.
    Used for both LoRAs and full base-model checkpoints."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    headers = {}
    full_url = url
    low = url.lower()
    hf_key = HF_TOKEN or _get_secret("HF_TOKEN", verbose=False)
    civitai_key = CIVITAI_API_KEY or _get_secret("CIVITAI_API_KEY", verbose=False)
    if "huggingface" in low and hf_key:
        headers["Authorization"] = f"Bearer {hf_key}"
    elif "civitai" in low:
        if "/api/download/" not in low:
            raise RuntimeError(
                "this looks like a Civitai model PAGE url — use the direct "
                "download link, e.g. "
                "https://civitai.com/api/download/models/<versionId> "
                "(right-click the download button -> Copy link)")
        if not civitai_key:
            _log("  download: no CIVITAI_API_KEY — trying without auth "
                 "(will 401 if the file requires a key).")
        else:
            headers["Authorization"] = f"Bearer {civitai_key}"
            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}token={civitai_key}"
    _fname = (dest_name if dest_name.endswith((".safetensors", ".gguf"))
              else f"{dest_name}.safetensors")
    path = str(Path(out_dir) / _fname)
    # Disk cache: if we already have this file (from earlier in the session or
    # a reconnect that kept /content), don't re-download it. Colab still wipes
    # /content on "delete runtime", but a plain reconnect or repeated load in
    # the same session reuses the file.
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        _log(f"  already on disk — skipping download ({_fmt_bytes(os.path.getsize(path))}).")
        return path
    # Write to a unique temp file then atomically rename — prevents a
    # double-fired download from interleaving writes into the same file.
    tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
    with _requests.get(full_url, headers=headers, stream=True,
                       allow_redirects=True, timeout=900) as r:
        if r.status_code in (401, 403):
            raise RuntimeError(
                f"{r.status_code} unauthorized — for Civitai set the "
                "CIVITAI_API_KEY secret; for gated HF repos set HF_TOKEN")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        last_log = 0.0
        if total:
            _log(f"  download size: {_fmt_bytes(total)}")
        else:
            _log("  download size: unknown (server sent no Content-Length)")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_log >= 2.0:
                    elapsed = max(now - t0, 1e-6)
                    speed = done / elapsed                  # bytes/s
                    if total:
                        pct = 100.0 * done / total
                        eta = (total - done) / speed if speed > 0 else 0
                        msg = (f"  downloading {pct:5.1f}%  "
                               f"{_fmt_bytes(done)} / {_fmt_bytes(total)}  "
                               f"@ {_fmt_bytes(speed)}/s  "
                               f"ETA {int(eta)//60}m{int(eta)%60:02d}s")
                    else:
                        pct = None
                        msg = (f"  downloading {_fmt_bytes(done)}  "
                               f"@ {_fmt_bytes(speed)}/s")
                    _log(msg)
                    if progress_cb:
                        try:
                            progress_cb({"done": done, "total": total,
                                         "pct": pct,
                                         "speed_bps": speed,
                                         "elapsed": elapsed})
                        except Exception:
                            pass
                    last_log = now
        elapsed = max(time.time() - t0, 1e-6)
        _log(f"  download complete: {_fmt_bytes(done)} in "
             f"{int(elapsed)//60}m{int(elapsed)%60:02d}s "
             f"(avg {_fmt_bytes(done/elapsed)}/s)")
    if Path(tmp_path).stat().st_size < min_bytes:
        try: os.remove(tmp_path)
        except Exception: pass
        raise RuntimeError("downloaded file is too small — likely an auth "
                           "or bad-URL error (check the key / that the URL "
                           "is a direct download link)")
    os.replace(tmp_path, path)   # atomic move into final location
    return path

def _hf_predownload(repo_id, token=None):
    """Download an HF repo's files into the hub cache BEFORE from_pretrained,
    with live progress to the console and the swap status line. In Colab,
    huggingface_hub renders its own progress as notebook widgets that our
    web-UI console never sees — so we watch the cache's blobs/ directory
    grow ourselves (blobs only: the snapshots/ entries are symlinks to the
    same files and would double-count). Best-effort: any failure just falls
    through to from_pretrained's own (invisible) download."""
    if not repo_id or "/" not in repo_id or os.path.exists(repo_id):
        return
    try:
        from huggingface_hub import snapshot_download
        try:
            from huggingface_hub.constants import HF_HUB_CACHE as _cache
        except Exception:
            _cache = os.path.expanduser(
                os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
        # Total size for a percent display, when the API will tell us.
        total = None
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo_id, files_metadata=True,
                                      token=token)
            total = sum((f.size or 0) for f in (info.siblings or [])) or None
        except Exception:
            pass
        blob_dir = os.path.join(_cache,
                                "models--" + repo_id.replace("/", "--"),
                                "blobs")

        def _dir_size(d):
            sz = 0
            for root, _dirs, files in os.walk(d):
                for fn in files:
                    try:
                        sz += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            return sz

        already = _dir_size(blob_dir)
        if total and already >= total * 0.99:
            _log(f"  [hf] {repo_id}: already in cache "
                 f"({_fmt_bytes(already)}).")
            return
        if total:
            _log(f"  [hf] {repo_id}: downloading {_fmt_bytes(total)} "
                 "to the hub cache...")
        else:
            _log(f"  [hf] {repo_id}: downloading to the hub cache...")
        done_evt = threading.Event()

        def _watch():
            last, t_last = _dir_size(blob_dir), time.time()
            while not done_evt.wait(3.0):
                sz = _dir_size(blob_dir)
                now = time.time()
                speed = max(0, sz - last) / max(0.5, now - t_last)
                last, t_last = sz, now
                if total:
                    pct = min(100.0, 100.0 * sz / max(1, total))
                    msg = (f"downloading {pct:.0f}%  {_fmt_bytes(sz)} / "
                           f"{_fmt_bytes(total)}  @ {_fmt_bytes(speed)}/s")
                else:
                    msg = (f"downloading {_fmt_bytes(sz)}  "
                           f"@ {_fmt_bytes(speed)}/s")
                STATE["swap"]["stage"] = msg
                _log("  [hf] " + msg)

        th = threading.Thread(target=_watch, daemon=True)
        th.start()
        try:
            snapshot_download(repo_id, token=token)
        finally:
            done_evt.set()
            th.join(timeout=1)
        _log(f"  [hf] {repo_id}: download complete "
             f"({_fmt_bytes(_dir_size(blob_dir))}).")
        STATE["swap"]["stage"] = "loading weights into VRAM"
    except Exception as e:
        _log(f"  [hf] pre-download note ({repo_id}): {e} — continuing; "
             "from_pretrained will fetch what's missing.")

def download_lora(url, dest_name, out_dir="/content/loras"):
    return download_safetensors(url, dest_name, out_dir, min_bytes=4096)


# ---- LoRA helpers ------------------------------------------------------
def inspect_lora_file(path):
    """Read a .safetensors header (metadata + tensor key names) WITHOUT
    loading weights, and infer the LoRA format and intended base model.

    safetensors layout: first 8 bytes = uint64 little-endian header length,
    then that many bytes of JSON. The JSON maps every tensor name to its
    info, plus an optional '__metadata__' dict with training details. We
    only need the header, so this is fast even on multi-GB files."""
    import struct
    info = {"format": "unknown", "base_model": None, "network_module": None,
            "n_tensors": 0, "loadable": None, "notes": [], "meta": {}}
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100 * 1024 * 1024:
                info["notes"].append("header length looks invalid — not a "
                                     "valid safetensors file?")
                return info
            header = json.loads(f.read(n).decode("utf-8", "replace"))
    except Exception as e:
        info["notes"].append(f"could not read header: {e}")
        return info

    meta = header.get("__metadata__", {}) or {}
    keys = [k for k in header.keys() if k != "__metadata__"]
    info["n_tensors"] = len(keys)

    # Training metadata that Kohya / sd-scripts write, when present.
    info["base_model"] = (meta.get("ss_base_model_version")
                          or meta.get("modelspec.architecture")
                          or meta.get("ss_sd_model_name"))
    info["network_module"] = meta.get("ss_network_module")
    # Surface a few useful raw metadata fields if they exist.
    for k in ("ss_base_model_version", "ss_network_module",
              "ss_network_dim", "ss_network_alpha", "modelspec.architecture",
              "modelspec.title", "ss_sd_model_name"):
        if k in meta:
            info["meta"][k] = meta[k]

    joined = "\n".join(keys)
    # Format detection from tensor key patterns — deterministic, works even
    # when __metadata__ is stripped (Civitai sometimes strips it).
    has = lambda sub: sub in joined
    if has("hada_w1") or has("hada_w2"):
        info["format"] = "LoHa (LyCORIS)"
    elif has("lokr_w1") or has("lokr_w2") or has(".lokr_"):
        info["format"] = "LoKr (LyCORIS)"
    elif has("dora_scale") or has(".dora_"):
        info["format"] = "DoRA"
    elif has("lora_down") or has("lora_up") or has(".lora_A") or has(".lora_B"):
        # Distinguish a LyCORIS LoCon (conv layers) from plain LoRA.
        if has("lora_mid") or has("conv_in") or has("_conv"):
            info["format"] = "LoCon (LyCORIS)"
        else:
            info["format"] = "LoRA (standard Kohya)"
    elif has("oft_") or has("boft_"):
        info["format"] = "OFT/BOFT"

    # Model-family hint from key naming. Kohya Flux LoRAs also use a
    # 'lora_unet_' prefix, so only flag UNet-era files when the DiT block
    # names (double/single/transformer blocks) are absent.
    if has("lora_te2") or has("te2_"):
        info["notes"].append("has a second text encoder (lora_te2) — this is "
                             "an SDXL-era LoRA, NOT for Qwen/Flux/Z-Image.")
    elif (has("double_blocks") or has("single_blocks")):
        info["notes"].append("double/single block keys — consistent with a "
                             "FLUX LoRA.")
    elif has("transformer_blocks") or has("transformer."):
        info["notes"].append("transformer-block keys — consistent with a DiT "
                             "(Qwen/Flux/Z-Image) LoRA.")
    elif has("lora_unet_") or has("input_blocks") or has("output_blocks"):
        info["notes"].append("UNet-style keys with no DiT blocks — looks like "
                             "an SD/SDXL LoRA; it will NOT load on "
                             "Qwen/Flux/Z-Image.")

    # Verdict on whether diffusers can load it.
    fmt = info["format"]
    if fmt == "LoRA (standard Kohya)":
        info["loadable"] = True
        info["notes"].append("standard LoRA — diffusers should load this.")
    elif fmt in ("LoCon (LyCORIS)",):
        info["loadable"] = "maybe"
        info["notes"].append("LoCon — recent diffusers loads many LoCons, "
                             "some still fail.")
    elif fmt in ("LoHa (LyCORIS)", "LoKr (LyCORIS)"):
        info["loadable"] = "lycoris"
        info["notes"].append(f"{fmt} — diffusers can't load this directly; "
                             "the studio routes it through the LyCORIS "
                             "loader (wraps the transformer).")
    elif fmt == "OFT/BOFT":
        info["loadable"] = False
        info["notes"].append("OFT/BOFT is not supported by diffusers' "
                             "load_lora_weights nor the LyCORIS inference "
                             "wrapper used here.")
    elif fmt == "DoRA":
        info["loadable"] = "maybe"
        info["notes"].append("DoRA needs a recent peft+diffusers; may fail "
                             "on older stacks.")
    else:
        info["notes"].append("could not recognize the key pattern — unusual "
                             "or non-LoRA file.")
    return info

def _log_lora_inspection(name, info):
    _log(f"  [inspect] LoRA '{name}':")
    _log(f"  [inspect]   detected format : {info['format']}")
    _log(f"  [inspect]   tensors         : {info['n_tensors']}")
    if info.get("base_model"):
        _log(f"  [inspect]   trained on      : {info['base_model']}")
    if info.get("network_module"):
        _log(f"  [inspect]   network module  : {info['network_module']}")
    verdict = {True: "yes", False: "NO", "maybe": "maybe", None: "unknown"}
    _log(f"  [inspect]   diffusers can load: {verdict.get(info.get('loadable'))}")
    for nt in info.get("notes", []):
        _log(f"  [inspect]   - {nt}")

def _detect_lycoris_algo(raw_keys):
    """Classify the LyCORIS algorithm from tensor key names.
    Returns one of: 'loha', 'lokr', 'locon', 'lora'."""
    ks = list(raw_keys)
    if any(".hada_w1" in k or ".hada_w2" in k for k in ks):
        return "loha"
    if any(".lokr_w1" in k or ".lokr_w2" in k for k in ks):
        return "lokr"
    if any("lora_mid" in k or "_conv" in k for k in ks):
        return "locon"
    return "lora"

def _is_bnb_quantized(m):
    """bitsandbytes-quantized modules must NOT be .to()-moved (their packed
    weights treat device moves as quantization events). GGUF-quantized
    modules are different: they load on CPU and MUST be moved to the GPU —
    GGUFParameter supports device transfer losslessly."""
    if m is None:
        return False
    if getattr(m, "is_loaded_in_4bit", False) or \
            getattr(m, "is_loaded_in_8bit", False):
        return True
    try:
        import bitsandbytes as bnb
        return any(isinstance(x, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
                   for x in m.modules())
    except Exception:
        return False

def _transformer_is_quantized(tr):
    """True when the transformer's weights are bitsandbytes-quantized (NF4).
    LyCORIS handles such layers itself (its modules auto-switch to the
    functional 'bypass' path when they wrap a bnb layer) — we only use this
    to sanity-check the installed lycoris version supports that."""
    if tr is None:
        return False
    if getattr(tr, "is_loaded_in_4bit", False) or getattr(tr, "is_loaded_in_8bit", False):
        return True
    if getattr(tr, "hf_quantizer", None) is not None:
        return True
    try:
        import bitsandbytes as bnb
        if any(isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
               for m in tr.modules()):
            return True
    except Exception:
        pass
    try:
        # diffusers GGUF loading keeps weights as GGUFParameter tensors.
        return any(type(prm).__name__ == "GGUFParameter"
                   for prm in tr.parameters())
    except Exception:
        return False

def _load_lycoris_adapter(pipe, path, name, multiplier=1.0):
    """Load a LyCORIS (LoHa/LoKr/LoCon) file using the canonical lycoris-lora
    library, which parses these Civitai files natively (diffusers'
    load_lora_weights cannot load LoHa/LoKr — confirmed current limitation).

    Mirrors the original SDXL-studio approach: build a wrapper per sub-model
    with create_lycoris_from_weights (which owns ALL the algorithm math —
    alphas, dims, kron reconstruction — we do not second-guess it), verify it
    actually bound modules, attach via apply_to() so the network stays
    toggleable, and keep the wrapper objects in STATE so the per-network
    'multiplier' can be changed between generations for live strength
    control. On the DiT pipelines the targets are the transformer (and the
    text encoder, when the file carries keys for it — the wrapper that binds
    0 modules is skipped). Returns the detected algorithm string."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        algo = _detect_lycoris_algo(f.keys())
    _log(f"  LoRA '{name}': LyCORIS algorithm detected = {algo}")

    try:
        from lycoris import create_lycoris_from_weights
    except Exception as e:
        raise RuntimeError(
            "the 'lycoris-lora' library isn't installed — add "
            "`pip install lycoris-lora` to the setup cell to load LoHa/LoKr "
            f"LyCORIS files ({e})") from e

    subs = []
    tr = getattr(pipe, "transformer", None)
    if tr is not None:
        subs.append((tr, "transformer"))
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        subs.append((te, "text_encoder"))
    if not subs:
        raise RuntimeError("pipeline exposes no transformer/text_encoder")

    quantized = _transformer_is_quantized(tr)
    wrappers = []
    total_loaded = 0
    for sub, tag in subs:
        try:
            w, _ = create_lycoris_from_weights(float(multiplier), path, sub)
            if w is None:
                continue
            # Count how many modules the wrapper actually bound. A wrapper
            # that loaded 0 modules is a no-op and must NOT count as a
            # successful attach (otherwise the LoRA silently has no effect
            # while we report 'ready').
            n_mod = 0
            for attr in ("loras", "modules", "_modules_loaded"):
                v = getattr(w, attr, None)
                if v is not None:
                    try: n_mod = len(v)
                    except Exception: pass
                    if n_mod: break
            if n_mod <= 0:
                _log(f"  LoRA '{name}': lycoris bound 0 modules on {tag} — skipping")
                continue
            if tag == "transformer" and quantized:
                # lycoris >= 2.2 auto-runs in bypass mode on bnb-quantized
                # layers (functional delta; packed 4-bit weights untouched).
                # Older versions rebuild weights and would corrupt the
                # output — refuse loudly instead of generating garbage.
                n_bypass = sum(1 for m in (getattr(w, "loras", []) or [])
                               if getattr(m, "bypass_mode", False))
                if n_bypass == 0:
                    raise RuntimeError(
                        "the transformer is NF4-quantized but this "
                        "lycoris-lora version didn't enable bypass mode on "
                        "any module — upgrade with "
                        "`pip install -U 'lycoris-lora>=2.2.0'`")
                _log(f"  LoRA '{name}': bypass mode active on "
                     f"{n_bypass}/{n_mod} quantized modules.")
            w.apply_to()
            try:
                w.to(DEVICE, DTYPE)
            except Exception:
                w.to(DEVICE)
            wrappers.append((tag, w))
            total_loaded += n_mod
            _log(f"  LoRA '{name}': lycoris applied to {tag} ({n_mod} modules)")
        except Exception as e:
            _log(f"  LoRA '{name}': lycoris {tag} attach note — {e}")

    if not wrappers or total_loaded == 0:
        raise RuntimeError("lycoris produced no attached networks — "
                           "the file may target modules not present here")
    STATE.setdefault("_lycoris_wrappers", {})[name] = wrappers
    STATE.setdefault("_lycoris_loras", set()).add(name)
    return algo

def _lycoris_set_scale(name, value):
    """Live strength control: set the multiplier on every wrapper, exactly
    as the SDXL studio did (w.multiplier drives the delta's weight)."""
    wrappers = STATE.get("_lycoris_wrappers", {}).get(name)
    if not wrappers:
        return False
    ok = False
    for tag, w in wrappers:
        try:
            if hasattr(w, "set_multiplier"):
                w.set_multiplier(float(value))
            else:
                w.multiplier = float(value)
                for m in getattr(w, "loras", []) or []:
                    m.multiplier = float(value)
            ok = True
        except Exception as e:
            _log(f"  lycoris '{name}': multiplier set note ({tag}): {e}")
    return ok

def _lycoris_detach(name):
    """Detach every lycoris wrapper for this LoRA (restore base forwards).
    If a wrapper can't unhook, neutralize it at multiplier 0 instead."""
    wrappers = STATE.get("_lycoris_wrappers", {}).pop(name, [])
    STATE.get("_lycoris_loras", set()).discard(name)
    for tag, w in wrappers:
        try:
            w.restore()
            _log(f"  lycoris '{name}': detached from {tag}.")
        except Exception as e:
            try:
                if hasattr(w, "set_multiplier"):
                    w.set_multiplier(0.0)
                else:
                    w.multiplier = 0.0
                    for m in getattr(w, "loras", []) or []:
                        m.multiplier = 0.0
                _log(f"  lycoris '{name}': restore failed on {tag} ({e}); "
                     "strength forced to 0 instead.")
            except Exception:
                _log(f"  lycoris '{name}': could not detach from {tag} ({e}); "
                     "reload the model to fully remove it.")

def register_lora(name, url, scale, triggers=None):
    """Download + attach a LoRA to the live pipeline(s)."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "lora"
    if name in STATE["loras"]:
        return False, f"a LoRA named '{name}' already exists"
    # Guard against a double-click / double-fire downloading the same file
    # twice into the same path concurrently (which corrupts it).
    inflight = STATE.setdefault("_lora_inflight", set())
    if name in inflight:
        return False, f"'{name}' is already being added — please wait"
    inflight.add(name)
    try:
        return _register_lora_inner(name, url, scale, triggers or [])
    finally:
        inflight.discard(name)

def _register_lora_inner(name, url, scale, triggers):
    if not url.lower().startswith("http"):
        return False, "URL must start with http"
    _log(f"  LoRA '{name}': downloading...")
    try:
        path = download_lora(url, dest_name=name)
    except Exception as e:
        return False, f"download failed — {e}"
    # Inspect up front so the format/base model is in the log regardless of
    # whether the attach then succeeds or fails.
    insp = inspect_lora_file(path)
    _log_lora_inspection(name, insp)
    attached = False
    err = None
    pipe = STATE["txt_pipe"]
    if pipe is not None:
        with STATE["load_lock"]:
            # diffusers' per-pipeline LoRA mixins (FluxLoraLoaderMixin,
            # QwenImageLoraLoaderMixin, ZImage's loader) handle both
            # diffusers-format and kohya/ComfyUI-format files, converting key
            # layouts as needed and loading into the transformer (and text
            # encoder where the file carries TE weights). This single call is
            # the whole job for Civitai Qwen / Flux.1 D / Z-Image LoRAs.
            is_lyco_fmt = "LyCORIS" in (insp.get("format") or "")
            if not is_lyco_fmt:
                try:
                    pipe.load_lora_weights(path, adapter_name=name)
                    attached = True
                    _log(f"  LoRA '{name}': attached.")
                except Exception as e:
                    err = str(e)
                    _log(f"  LoRA '{name}': load_lora_weights failed — {e}")
                    try: pipe.delete_adapters(name)
                    except Exception: pass
            # LyCORIS files (LoKr/LoHa, and LoCons diffusers rejected) go
            # through the lycoris-lora wrapper on the transformer.
            if not attached and (is_lyco_fmt or
                                 (err and "lycoris" in err.lower())):
                try:
                    _log(f"  LoRA '{name}': trying LyCORIS loader "
                         f"({insp.get('format')})...")
                    _load_lycoris_adapter(pipe, path, name,
                                          multiplier=float(scale))
                    attached = True
                    err = None
                    _log(f"  LoRA '{name}': attached via lycoris-lora "
                         f"(strength {scale}).")
                except Exception as e2:
                    err = (f"{err} | " if err else "") + f"lycoris: {e2}"
                    _log(f"  LoRA '{name}': LyCORIS loader failed — {e2}")
                    _lycoris_detach(name)
    if not attached:
        # Incompatible / failed: leave NO trace. Delete the file and don't
        # register an entry, so the sidebar never shows a dead LoRA row.
        try:
            if pipe is not None:
                try: pipe.delete_adapters(name)
                except Exception: pass
        finally:
            try: os.remove(path)
            except Exception: pass
        fmt = insp.get("format", "unknown")
        arch = STATE.get("arch", DEFAULT_MODEL_KEY)
        label = MODEL_REGISTRY.get(arch, {}).get("label", arch)
        notes = " ".join(insp.get("notes", []))
        if insp.get("loadable") is False:
            reason = (f"this is a {fmt} file — neither diffusers nor the "
                      "LyCORIS wrapper can load that format")
        elif insp.get("loadable") == "lycoris":
            reason = (f"{fmt} file — the LyCORIS loader couldn't wrap it "
                      f"onto this transformer: {err or 'unknown error'}")
        elif "NOT load on" in notes or "SDXL-era" in notes:
            reason = (f"this LoRA was trained for SD/SDXL, not {label} — "
                      "wrong architecture; pick one whose base model matches "
                      "the active model family")
        else:
            reason = (f"could not attach to {label} — most likely it targets "
                      "a different base-model family (check the base model on "
                      f"its Civitai card). diffusers raised: "
                      f"{err or 'unknown error'}")
        return False, reason
    # Success — register it.
    STATE["loras"][name] = {"path": path, "scale": float(scale),
                            "attached": True, "info": insp, "url": url,
                            "lycoris": name in STATE.get("_lycoris_loras", set()),
                            "triggers": triggers or []}
    _log(f"  LoRA '{name}': ready (strength {scale}).")
    return True, "ok"

def remove_lora(name):
    info = STATE["loras"].pop(name, None)
    if not info:
        return False
    pipe = STATE["txt_pipe"]
    if name in STATE.get("_lycoris_loras", set()):
        _lycoris_detach(name)
    elif pipe is not None and info.get("attached"):
        try: pipe.delete_adapters(name)
        except Exception: pass
    try: os.remove(info["path"])
    except Exception: pass
    return True

def apply_loras(pipe):
    """Activate registered PEFT-adapter LoRAs at their current strengths.
    LyCORIS LoRAs are excluded: their lycoris-lora wrappers are already live
    on the transformer with their own multiplier (set at load / via the
    strength slider), and they aren't PEFT adapters set_adapters knows."""
    lyco = STATE.get("_lycoris_loras", set())
    names = [n for n, i in STATE["loras"].items()
             if i.get("attached") and n not in lyco]
    if not names:
        if not lyco:
            try: pipe.disable_lora()
            except Exception: pass
        return
    weights = [STATE["loras"][n]["scale"] for n in names]
    try:
        pipe.set_adapters(names, adapter_weights=weights)
    except Exception as e:
        _log(f"  set_adapters warning: {e}")

def reattach_all_loras(pipe):
    """After a base-model swap the old transformer (and its LoRA attachments)
    is gone. Re-attach every registered LoRA to the freshly built pipeline. A
    LoRA that no longer fits (e.g. a Flux LoRA after swapping to Qwen) is
    REMOVED (file deleted, entry dropped) rather than left as a dead
    'not attached' row. Returns the list of dropped names."""
    dropped = []
    lyco = STATE.get("_lycoris_loras", set())
    # Old lycoris wrappers point at the previous model's modules; clear them.
    STATE["_lycoris_wrappers"] = {}
    for name, info in list(STATE["loras"].items()):
        try:
            if name in lyco or info.get("lycoris"):
                lyco.discard(name)        # re-added by the loader on success
                _load_lycoris_adapter(pipe, info["path"], name,
                                      multiplier=float(info.get("scale", 1.0)))
            else:
                pipe.load_lora_weights(info["path"], adapter_name=name)
            info["attached"] = True
        except Exception as e:
            _log(f"  LoRA '{name}': not compatible with this model — {e}")
            try: pipe.delete_adapters(name)
            except Exception: pass
            try: os.remove(info["path"])
            except Exception: pass
            STATE["loras"].pop(name, None)
            lyco.discard(name)
            dropped.append(name)
    return dropped

# ---- pipeline ----------------------------------------------------------
def _looks_like_single_file(ref):
    """A local .safetensors path or a direct .safetensors / Civitai download
    URL is a single-file checkpoint; otherwise treat ref as an HF repo id."""
    low = ref.lower()
    if low.endswith(".safetensors"):
        return True
    if low.startswith("http"):
        return ("civitai.com/api/download" in low
                or low.endswith(".safetensors"))
    return False

def _pipeline_classes(arch):
    """(Txt2ImgPipeline, Img2ImgPipeline, TransformerClass) for a family."""
    import diffusers as _df
    if arch == "qwen":
        return (_df.QwenImagePipeline, _df.QwenImageImg2ImgPipeline,
                _df.QwenImageTransformer2DModel)
    if arch == "flux":
        return (_df.FluxPipeline, _df.FluxImg2ImgPipeline,
                _df.FluxTransformer2DModel)
    if arch == "zimage":
        return (_df.ZImagePipeline, _df.ZImageImg2ImgPipeline,
                _df.ZImageTransformer2DModel)
    raise ValueError(f"unknown model family '{arch}'")

def build_pipeline(model_ref=None, model_name=None, arch=None):
    """Build the active DiT txt2img pipeline + an img2img view that reuses the
    same modules (no extra VRAM). Loads from an HF repo id OR grafts a
    single-file (Civitai) transformer checkpoint onto the family's base
    pipeline. bf16, fully GPU-resident on the A100; re-callable for swaps."""
    ref = model_ref or STATE["model_ref"]
    label = model_name or STATE["model_name"]
    arch = (arch or (STATE.get("arch") if model_ref is None else None)
            or _detect_arch(ref, label) or DEFAULT_MODEL_KEY)
    with STATE["load_lock"]:
        if STATE["txt_pipe"] is not None and model_ref is None:
            return STATE["txt_pipe"]

        TxtCls, ImgCls, TrCls = _pipeline_classes(arch)
        base_repo = MODEL_REGISTRY[arch]["repo"]
        common = {}
        quant_kind = "nf4"            # refined to 'gguf-q4' when GGUF loads
        if HF_TOKEN:
            common["token"] = HF_TOKEN
        if arch == "zimage":
            # Per the Z-Image README — avoids a meta-device init issue.
            common["low_cpu_mem_usage"] = False

        # VRAM-adaptive plan. Colab "A100" is usually the 40 GB SXM4 — too
        # small for Qwen-2512's ~41 GB bf16 transformer even with offload
        # (a single module must still fit). Under the family's bf16
        # requirement we load the TRANSFORMER in 4-bit NF4 (bitsandbytes:
        # Qwen ~12 GB, Flux ~7 GB) and run the pipeline with model CPU
        # offload, which keeps one component on the GPU at a time.
        tot_gb = (torch.cuda.mem_get_info()[1] / 1e9
                  if torch.cuda.is_available() else 0)
        quantize = tot_gb and tot_gb < MODEL_REGISTRY[arch]["bf16_gb"]
        quant_cfg = None
        if quantize:
            from diffusers import BitsAndBytesConfig as _DFBnb
            quant_cfg = _DFBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=DTYPE)
            _log(f"  {tot_gb:.0f} GB VRAM < {MODEL_REGISTRY[arch]['bf16_gb']} GB "
                 f"needed for bf16 — loading the {arch} transformer "
                 "quantized (GGUF Q4_K preferred, bnb NF4 fallback), fully "
                 "GPU-resident.")

        if _looks_like_single_file(ref):
            # Civitai-style single-file checkpoints for these families ship
            # the TRANSFORMER weights. Build the family's base pipeline (text
            # encoder + VAE + scheduler from the official repo — cached after
            # the first load), then swap in the checkpoint's transformer.
            _log(f"Loading {arch} base components: {base_repo} ...")
            _hf_predownload(base_repo, token=HF_TOKEN)
            txt = TxtCls.from_pretrained(base_repo, transformer=None,
                                         torch_dtype=DTYPE, **common)
            _log(f"Grafting single-file transformer: {label} ...")
            sf_kwargs = {k: v for k, v in common.items() if k == "token"}
            if quant_cfg is not None:
                sf_kwargs["quantization_config"] = quant_cfg
                _log("  NOTE: low VRAM forces naive bnb NF4 on this "
                     "custom checkpoint — expect added grain vs the "
                     "official GGUF builds.")
            try:
                tr = TrCls.from_single_file(ref, torch_dtype=DTYPE,
                                            config=base_repo,
                                            subfolder="transformer",
                                            **sf_kwargs)
            except Exception as e1:
                _log(f"  from_single_file with base config failed ({e1}); "
                     "retrying without config hint...")
                try:
                    tr = TrCls.from_single_file(ref, torch_dtype=DTYPE,
                                                **sf_kwargs)
                except Exception as e2:
                    raise RuntimeError(
                        f"could not load this checkpoint as a {arch} "
                        f"transformer ({e2}). It is probably for a different "
                        "model family — load checkpoints from the Civitai "
                        "browser so the base-model tag is checked.") from e2
            txt.transformer = tr
            _log("  grafted checkpoint transformer.")
        elif quant_cfg is not None:
            # Low-VRAM transformer. PREFER the community GGUF Q4_K quant:
            # it keeps sensitive tensors (modulation layers, in/out
            # projections) at higher precision, so output quality holds.
            # Naive bnb NF4 quantizes EVERY linear — on these DiT models
            # that produces uniform mosaic/stipple corruption; it remains
            # only as a fallback if the GGUF can't be fetched/loaded.
            tr = None
            gg = MODEL_REGISTRY[arch].get("gguf")
            if gg and ref == MODEL_REGISTRY[arch]["repo"]:
                try:
                    from diffusers import GGUFQuantizationConfig
                    _log(f"Low VRAM: using GGUF {gg['label']} transformer "
                         f"({gg['size_gb']} GB) — higher quality than NF4.")
                    def _gg_prog(p):
                        if p.get("pct") is not None:
                            STATE["swap"]["stage"] = (
                                f"GGUF download {p['pct']:.0f}%  "
                                f"@ {_fmt_bytes(p['speed_bps'])}/s")
                        else:
                            STATE["swap"]["stage"] = (
                                f"GGUF download {_fmt_bytes(p['done'])}  "
                                f"@ {_fmt_bytes(p['speed_bps'])}/s")
                    gpath = download_safetensors(
                        gg["url"], gg["fname"], out_dir="/content/models",
                        min_bytes=1 << 30, progress_cb=_gg_prog)
                    STATE["swap"]["stage"] = "loading GGUF transformer"
                    tr = TrCls.from_single_file(
                        gpath,
                        quantization_config=GGUFQuantizationConfig(
                            compute_dtype=DTYPE),
                        torch_dtype=DTYPE, config=ref,
                        subfolder="transformer")
                    _log("  GGUF transformer loaded.")
                    quant_kind = "gguf-q4"
                except Exception as e:
                    _log(f"  GGUF transformer failed ({e}) — falling back "
                         "to bnb NF4 (expect visible quality loss).")
                    tr = None
            if tr is None:
                _log(f"Loading model (repo, NF4 transformer): {ref} ...")
                STATE["swap"]["stage"] = "quantizing transformer to NF4"
                tr = TrCls.from_pretrained(ref, subfolder="transformer",
                                           quantization_config=quant_cfg,
                                           torch_dtype=DTYPE,
                                           **{k: v for k, v in common.items()
                                              if k == "token"})
            # Text encoder + VAE from the repo (cached after first run).
            _hf_predownload(ref, token=HF_TOKEN)
            STATE["swap"]["stage"] = "loading text encoder + VAE into VRAM"
            txt = TxtCls.from_pretrained(ref, transformer=tr,
                                         torch_dtype=DTYPE, **common)
        else:
            _log(f"Loading model (repo): {ref} ...")
            _hf_predownload(ref, token=HF_TOKEN)
            STATE["swap"]["stage"] = "loading weights into VRAM"
            txt = TxtCls.from_pretrained(ref, torch_dtype=DTYPE, **common)

        # img2img reuses the SAME components — zero extra weights in memory.
        img = None
        try:
            img = ImgCls(**txt.components)
        except Exception as e:
            try:
                img = ImgCls.from_pipe(txt)
            except Exception as e2:
                _log(f"  img2img view unavailable for this model ({e2}).")

        # Residency. 80 GB A100: everything on the GPU in bf16. Smaller GPUs
        # (quantized path): model CPU offload keeps one component resident at
        # a time — the NF4 transformer plus the bf16 text encoder both fit a
        # 40 GB card comfortably this way.
        residency = "cpu-offload"
        if quantize:
            # KEEP THE QUANTIZED PIPELINE FULLY ON THE GPU. The NF4
            # transformer is small (~12 GB for Qwen, ~7 GB for Flux), so the
            # whole pipeline fits a 40 GB card resident: NF4 transformer +
            # bf16 text encoder + VAE ≈ 29 GB for Qwen. Crucially, do NOT use
            # enable_model_cpu_offload here — its hooks round-trip components
            # through .to('cpu')/.to('cuda') after every use, and moving
            # bitsandbytes 4-bit packed weights like that scrambles them
            # (uniform mosaic/stipple corruption in every image).
            try:
                # Move every component to the GPU EXCEPT bitsandbytes ones
                # (a bnb transformer is already on the GPU from loading and
                # must not be .to()-moved). The GGUF transformer loads on
                # CPU and must be moved here, or the first matmul dies on a
                # cuda-vs-cpu device mismatch.
                for _cn, _comp in (txt.components or {}).items():
                    if isinstance(_comp, torch.nn.Module) and \
                            not _is_bnb_quantized(_comp):
                        _comp.to(DEVICE)
                residency = f"gpu ({quant_kind} transformer)"
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    _log(f"  FULL GPU RESIDENCY ({quant_kind} transformer) — "
                         f"{(total - free)/2**30:.1f} GB used of "
                         f"{total/2**30:.0f} GB.")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                txt.enable_model_cpu_offload()
                residency = f"cpu-offload ({quant_kind} transformer)"
                _log("  OOM keeping the quantized pipeline resident — fell "
                     "back to CPU offload. WARNING: offloading bnb 4-bit "
                     "weights is known to corrupt them; if images look "
                     "mosaic-like, lower the resolution instead.")
        else:
            try:
                txt.to(DEVICE)
                residency = "gpu"
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    _log(f"  FULL GPU RESIDENCY — {(total - free)/2**30:.1f} GB "
                         f"used of {total/2**30:.0f} GB.")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                txt.enable_model_cpu_offload()
                _log("  hit OOM on full residency — using CPU offload.")
            except Exception as e:
                torch.cuda.empty_cache()
                try:
                    txt.enable_model_cpu_offload()
                except Exception:
                    pass
                _log(f"  full residency failed ({e}) — using CPU offload.")

        # Trim the VAE's decode peak — decisive on the 40 GB card, where a
        # full-frame decode after a successful denoise was the OOM point.
        # Prefer the VAE-object methods (the pipeline-level ones are
        # deprecated and warn); fall back only when the VAE lacks them.
        _vae = getattr(txt, "vae", None)
        for prim, legacy in (("enable_slicing", "enable_vae_slicing"),
                             ("enable_tiling", "enable_vae_tiling")):
            try:
                if _vae is not None and hasattr(_vae, prim):
                    getattr(_vae, prim)()
                else:
                    getattr(txt, legacy)()
            except Exception:
                pass

        STATE["txt_pipe"] = txt
        STATE["img_pipe"] = img
        STATE["residency"] = residency
        STATE["model_ref"] = ref
        STATE["model_name"] = label
        STATE["arch"] = arch
        _log(f"  Ready: {label} [{arch}] (residency: {residency}).")
        return txt

def swap_base_model(url_or_repo, label=None, base_hint=None):
    """Unload the current model, free VRAM, then load a new one. Accepts a
    registry key ('qwen' / 'flux' / 'zimage'), a Civitai/HF download URL, or
    an HF repo id. base_hint (e.g. the Civitai base-model tag) disambiguates
    which family a download URL belongs to. Existing LoRAs are dropped from
    the old transformer and re-attached to the new one where compatible."""
    ref = (url_or_repo or "").strip()
    if not ref:
        return False, ("give a model key (qwen / flux / zimage), a "
                       "Civitai/HF download URL, or an HF repo id")

    # Registry shortcut.
    key = ref.lower()
    if key in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[key]
        ref = entry["repo"]
        label = label or entry["label"]
        arch = key
    else:
        arch = _detect_arch(base_hint, ref, label)
        if arch is None:
            return False, ("can't tell which model family this is for "
                           "(qwen / flux.1 / z-image). Load checkpoints from "
                           "the Civitai browser, or include the family name "
                           "in the display-name field. Note: Flux.2 is a "
                           "different architecture and is not supported.")

    # If it's a download URL, fetch the checkpoint to disk first.
    if ref.lower().startswith("http"):
        if "civitai" in ref.lower() and "/api/download/" not in ref.lower():
            return False, ("that's a Civitai model PAGE url — use the direct "
                           "download link (/api/download/models/<versionId>)")
        name = label or ("civitai_" + re.sub(r"[^0-9]", "", ref)[-8:] or "model")
        name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "model"
        _log(f"Base model: downloading checkpoint...")
        def _dl_progress(p):
            if p.get("pct") is not None:
                STATE["swap"]["stage"] = (
                    f"downloading {p['pct']:.0f}%  "
                    f"@ {_fmt_bytes(p['speed_bps'])}/s")
            else:
                STATE["swap"]["stage"] = (
                    f"downloading {_fmt_bytes(p['done'])}  "
                    f"@ {_fmt_bytes(p['speed_bps'])}/s")
        try:
            ref_local = download_safetensors(ref, name,
                                             out_dir="/content/models",
                                             min_bytes=1 << 20,
                                             progress_cb=_dl_progress)
        except Exception as e:
            return False, f"download failed — {e}"
        STATE["swap"]["stage"] = "loading model into VRAM"
        disp = label or name
    else:
        ref_local = ref            # HF repo id, load directly
        disp = label or ref

    # Tear down the live pipeline and free GPU memory before reloading —
    # essential here, since two of these models can't coexist in VRAM.
    with STATE["load_lock"]:
        # Drop lycoris wrappers FIRST: their modules hold references to the
        # old transformer, which otherwise survives the teardown (~12 GB of
        # NF4 weights kept alive through the swap — observed as a +9 GB
        # baseline after a re-swap). The names stay in _lycoris_loras so
        # reattach_all_loras knows to rebuild them on the new transformer.
        STATE["_lycoris_wrappers"] = {}
        STATE["txt_pipe"] = None
        STATE["img_pipe"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Mark every LoRA detached; reattach_all_loras re-applies them after.
        for info in STATE["loras"].values():
            info["attached"] = False

    try:
        pipe = build_pipeline(model_ref=ref_local, model_name=disp, arch=arch)
    except Exception as e:
        _log(f"  base-model load failed: {e}")
        return False, f"could not load model — {e}"
    # Remember the original source ref (download URL or repo id) so the browser
    # can tell which checkpoint version is currently resident.
    STATE["model_url"] = ref if ref.lower().startswith("http") else ""

    dropped = reattach_all_loras(pipe)
    survived = [n for n, i in STATE["loras"].items() if i.get("attached")]
    if dropped:
        _log(f"  LoRAs removed (incompatible with new model): {dropped}")
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        _log(f"  post-swap VRAM: {(total - free)/2**30:.1f} GB used "
             f"of {total/2**30:.0f} GB.")
    return True, {"model_name": disp, "survived": survived, "dropped": dropped}

# ---- persistent history (on the T4 disk) -------------------------------
HISTORY_DIR = "/content/outputs"
HISTORY_MANIFEST = "/content/outputs/history.json"
_history_lock = threading.Lock()

def _read_history():
    try:
        with open(HISTORY_MANIFEST) as f:
            return json.load(f)
    except Exception:
        return []

def _append_history(entry):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with _history_lock:
        hist = _read_history()
        hist.insert(0, entry)
        hist = hist[:500]                 # cap manifest length
        tmp = HISTORY_MANIFEST + ".tmp"
        with open(tmp, "w") as f:
            json.dump(hist, f)
        os.replace(tmp, HISTORY_MANIFEST)

# ---- samplers / schedulers ---------------------------------------------
# Qwen-Image, FLUX and Z-Image are flow-matching models: they ship with a
# FlowMatch scheduler and the A1111/Civitai SDXL sampler zoo (DPM++, Karras,
# etc.) does not apply. The default scheduler is right for almost everything;
# we expose just the flow-match variants. Old sampler names from Civitai
# recipes / history fall back to the default. We cache the pipeline's original
# scheduler so we can restore it.
_SAMPLER_MAP = {
    "FlowMatch Euler": ("FlowMatchEulerDiscreteScheduler", {}),
    "FlowMatch Heun":  ("FlowMatchHeunDiscreteScheduler", {}),
}

def _normalize_sampler(name):
    """Best-effort match of a free-form sampler string to a known key."""
    if not name:
        return None
    n = str(name).strip()
    if n in _SAMPLER_MAP:
        return n
    low = n.lower()
    for k in _SAMPLER_MAP:
        if k.lower() == low:
            return k
    # Loose mapping for flow-match aliases only; anything else (DPM++, DDIM,
    # Karras...) is an SDXL-era recipe and intentionally maps to the default.
    if "heun" in low:
        return "FlowMatch Heun"
    if "flow" in low and "euler" in low:
        return "FlowMatch Euler"
    return None

def set_sampler(pipe, name):
    """Swap the pipeline's scheduler to match a sampler name. Caches the
    original on first change so it can be restored. Returns the applied key
    or None if unrecognized (leaves the scheduler unchanged)."""
    key = _normalize_sampler(name)
    if not key:
        return None
    cls_name, kwargs = _SAMPLER_MAP[key]
    try:
        import diffusers as _df
        Sched = getattr(_df, cls_name)
        if "_orig_sched_cfg" not in STATE:
            STATE["_orig_sched_cfg"] = pipe.scheduler.config
        pipe.scheduler = Sched.from_config(pipe.scheduler.config, **kwargs)
        return key
    except Exception as e:
        _log(f"  sampler '{name}' -> {cls_name} failed: {e}")
        return None

# ---- generation worker -------------------------------------------------
def run_job(job_id, params):
    job = jobs[job_id]
    try:
        # If a base-model swap is in progress, wait it out rather than
        # generating against a half-unloaded pipeline.
        if STATE["swap"]["busy"]:
            job.update(status="running", stage="waiting for model swap",
                       progress=2)
            while STATE["swap"]["busy"]:
                if job.get("cancel"):
                    raise _JobCancelled()
                time.sleep(0.4)
        job.update(status="running", stage="loading model", progress=5)
        if job.get("cancel"):
            raise _JobCancelled()

        txt = build_pipeline()
        img_pipe = STATE["img_pipe"]
        arch = STATE.get("arch", DEFAULT_MODEL_KEY)

        mode = params.get("mode", "txt2img")
        init_img = None
        if params.get("image"):
            raw = base64.b64decode(params["image"].split(",")[-1])
            init_img = Image.open(io.BytesIO(raw)).convert("RGB")
        if mode == "img2img" and init_img is None:
            raise ValueError("img2img needs an uploaded image.")
        if mode == "img2img" and img_pipe is None:
            raise ValueError("img2img is unavailable for this model "
                             "(no img2img pipeline could be built).")

        steps = max(1, int(params["steps"]))
        guidance = float(params["guidance"])
        W = int(params.get("width", 1024))
        H = int(params.get("height", 1024))
        # These DiT models want dims divisible by 16 (VAE x8 + patch x2).
        W = max(512, min(2048, (W // 16) * 16))
        H = max(512, min(2048, (H // 16) * 16))
        strength = float(params.get("strength", 0.7))
        seed = int(params.get("seed", 0))
        n_images = max(1, min(int(params.get("batch", 1)), 4))

        _log(f"  [job] arch={arch}; mode={mode}; {W}x{H}; steps={steps}; "
             f"guidance={guidance}; seed={seed}; n={n_images}; "
             + (f"strength={strength}" if mode == "img2img" else ""))
        if arch == "flux" and (params.get("negative_prompt") or "").strip():
            _log("  [job] note: FLUX.1-dev has no CFG pass — the negative "
                 "prompt is ignored.")
        _log(f"  [job] prompt: {params['prompt'][:120]}")

        with STATE["lock"]:
            if job.get("cancel"):
                raise _JobCancelled()
            target = img_pipe if mode == "img2img" else txt
            # Sampler/scheduler: set it to match the requested one (e.g. to
            # reproduce a Civitai recipe, or LCM for DMD2/Lightning models).
            sampler = params.get("sampler")
            if sampler:
                applied = set_sampler(target, sampler)
                _log(f"  [job] sampler: {sampler}"
                     + ("" if applied else " (unrecognized — using current)"))
            elif STATE.get("_orig_sched_cfg") is not None:
                # "Default" selected — restore the pipeline's original scheduler.
                try:
                    import diffusers as _df
                    cfg = STATE["_orig_sched_cfg"]
                    cls = getattr(_df, cfg.get("_class_name",
                                               "EulerDiscreteScheduler"))
                    target.scheduler = cls.from_config(cfg)
                except Exception:
                    pass
            apply_loras(target)
            active = [k for k, i in STATE["loras"].items() if i.get("attached")]
            _log(f"  [job] active LoRAs: {active or 'none'}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            job.update(stage="generating", progress=15)

            def _cb(pipe_self, step, t, kw):
                if job.get("cancel"):
                    raise _JobCancelled()
                frac = (step + 1) / max(1, steps)
                job.update(stage=f"step {step+1}/{steps}",
                           progress=int(15 + frac * 75))
                return kw

            def _run_pipe(target, kw):
                # Step-progress callback when the pipeline supports it; the
                # newest pipelines occasionally lag on callback kwargs, so
                # fall back to a plain call rather than failing the job.
                try:
                    return target(callback_on_step_end=_cb,
                                  callback_on_step_end_tensor_inputs=["latents"],
                                  **kw)
                except TypeError:
                    return target(**kw)

            neg = params.get("negative_prompt") or ""
            results = []
            for i in range(n_images):
                if job.get("cancel"):
                    raise _JobCancelled()
                gen = torch.Generator(device=DEVICE).manual_seed(seed + i)
                call = dict(
                    prompt=params["prompt"],
                    num_inference_steps=steps,
                    generator=gen)
                # Guidance plumbing differs per family:
                #  qwen   — real CFG via true_cfg_scale; negative prompt works
                #           (must be non-empty when true_cfg_scale > 1).
                #  flux   — distilled/embedded guidance via guidance_scale;
                #           there is no CFG pass, so no negative prompt.
                #  zimage — guidance_scale; Turbo is distilled (use ~1, where
                #           the negative prompt has little effect).
                if arch == "qwen":
                    call["true_cfg_scale"] = guidance
                    call["negative_prompt"] = neg if neg.strip() else " "
                elif arch == "zimage":
                    call["guidance_scale"] = guidance
                    call["negative_prompt"] = neg
                else:  # flux
                    call["guidance_scale"] = guidance
                if mode == "img2img":
                    call["image"] = init_img.resize((W, H))
                    call["strength"] = strength
                    out = _run_pipe(img_pipe, call)
                else:
                    call["width"] = W
                    call["height"] = H
                    out = _run_pipe(txt, call)
                results.append(out.images[0])

        job.update(stage="encoding", progress=93)
        encoded = []
        saved_files = []
        hist_dir = "/content/outputs"
        os.makedirs(hist_dir, exist_ok=True)
        for idx, im in enumerate(results):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
            encoded.append("data:image/png;base64,"
                           + base64.b64encode(data).decode())
            # Persist to the T4's disk (tens of GB free) so history survives a
            # browser refresh / reconnect without bloating browser storage.
            fn = f"{job_id}_{idx}.png"
            try:
                with open(os.path.join(hist_dir, fn), "wb") as f:
                    f.write(data)
                saved_files.append(fn)
            except Exception as e:
                _log(f"  [history] could not save {fn}: {e}")
        _log(f"  done — {len(encoded)} image(s).")
        # Append to the on-disk history manifest. Save the full settings so
        # clicking a history image can restore everything that made it.
        try:
            active_loras = [
                {"name": n,
                 "scale": i.get("scale", 1.0),
                 "url": i.get("url"),
                 "triggers": i.get("triggers") or []}
                for n, i in STATE["loras"].items() if i.get("attached")]
            _append_history({
                "id": job_id,
                "prompt": params.get("prompt", ""),
                "files": saved_files,
                "ts": int(time.time() * 1000),
                "settings": {
                    "mode": params.get("mode", "txt2img"),
                    "prompt": params.get("prompt", ""),
                    "negative_prompt": params.get("negative_prompt", ""),
                    "steps": params.get("steps"),
                    "guidance": params.get("guidance"),
                    "width": params.get("width"),
                    "height": params.get("height"),
                    "seed": params.get("seed"),
                    "batch": params.get("batch"),
                    "strength": params.get("strength"),
                    "sampler": params.get("sampler", ""),
                    "model_name": STATE.get("model_name"),
                },
                "loras": active_loras,
            })
        except Exception as e:
            _log(f"  [history] manifest update failed: {e}")
        job.update(status="done", stage="complete", progress=100,
                   result=encoded)
    except _JobCancelled:
        _log("  [job] cancelled by user.")
        job.update(status="cancelled", stage="cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        msg = str(e).strip() or e.__class__.__name__
        if "out of memory" in tb.lower():
            msg = ("CUDA OOM — try a smaller size, fewer images, or fewer "
                   "steps. " + msg)
        job.update(status="error", stage="failed", error=msg)

# ---- Flask app ---------------------------------------------------------
app = Flask(__name__)

# ---- MissingLink sign-in gate ------------------------------------------
# The studio requires a MissingLink account before it will generate:
#   * Google sign-in  — opens missinglink.build/notebook-signin, which runs
#     the site's Google OAuth and hands back a session code (the popup
#     postMessages it to the studio, or the user pastes it manually).
#   * API key        — an existing MissingLink customer token.
# Either credential is verified server-side against /api/notebook/me on
# missinglink.build; nothing is trusted client-side. State is per-runtime
# (this cell), matching how the rest of the studio state works.
ML_BASE      = "https://missinglink.build"
ML_SIGNIN    = ML_BASE + "/notebook-signin"
ML_TRIAL_URL = ML_BASE + "/create-checkout-session"  # 303s to live Stripe Checkout (7-day trial)
AUTH = {"authed": False, "token": None, "kind": None, "email": None,
        "member": False, "used": None, "free_limit": None, "remaining": None}

def _ml_check_token(token):
    """Validate a credential against missinglink.build. Tries it as a
    Google session code (Authorization: Bearer) first, then as a customer
    API key (x-api-key). Returns (ok, info_dict_or_error_str, kind)."""
    token = (token or "").strip()
    if not token:
        return False, "empty sign-in code / API key", None
    last_err = "invalid or expired sign-in code / API key"
    for kind, headers in (("google", {"Authorization": "Bearer " + token}),
                          ("apikey", {"x-api-key": token})):
        try:
            r = _requests.get(ML_BASE + "/api/notebook/me",
                              headers=headers, timeout=15)
        except Exception as e:
            return False, f"could not reach MissingLink ({e})", None
        if r.status_code == 200:
            try:
                j = r.json()
            except Exception:
                j = {}
            if j.get("ok") and j.get("authed"):
                return True, j, kind
    return False, last_err, None

def _require_login():
    """Shared guard for generation endpoints. Returns a Flask response
    if the runtime is not signed in, else None."""
    if not AUTH["authed"]:
        return jsonify(error="login_required",
                       signin_url=ML_SIGNIN, trial_url=ML_TRIAL_URL), 401
    return None


@app.route("/")
def _index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/api/keepalive")
def _keepalive():
    return jsonify(ok=True, t=time.time())

@app.route("/api/auth/status")
def _auth_status():
    return jsonify(authed=AUTH["authed"], email=AUTH["email"],
                   member=AUTH["member"], remaining=AUTH["remaining"],
                   free_limit=AUTH["free_limit"],
                   signin_url=ML_SIGNIN, trial_url=ML_TRIAL_URL)

@app.route("/api/auth/validate", methods=["POST"])
def _auth_validate():
    body = request.get_json(force=True, silent=True) or {}
    ok, info, kind = _ml_check_token(body.get("token"))
    if not ok:
        return jsonify(ok=False, error=info), 401
    if kind == "google" and not info.get("member"):
        # Google sign-in alone isn't enough: the account must be a member
        # (7-day free trial, subscriber, or Pro). Hand the frontend the real
        # Stripe Checkout URL with their email prefilled so /create-checkout-
        # session can decide trial eligibility server-side.
        email = (info.get("email") or "").strip()
        checkout = ML_TRIAL_URL + (("?email=" + _url_quote(email)) if email else "")
        _log(f"  MissingLink sign-in refused \u2014 {email or 'unknown'} has no membership")
        return jsonify(ok=False, error="membership_required", email=email,
                       checkout_url=checkout), 402
    AUTH.update(authed=True, token=(body.get("token") or "").strip(),
                kind=kind, email=info.get("email"),
                member=bool(info.get("member")), used=info.get("used"),
                free_limit=info.get("free_limit"),
                remaining=info.get("remaining"))
    who = AUTH["email"] or "API-key user"
    tier = "member" if AUTH["member"] else "free trial"
    _log(f"  MissingLink sign-in OK \u2014 {who} ({tier})")
    return jsonify(ok=True, email=AUTH["email"], member=AUTH["member"],
                   remaining=AUTH["remaining"],
                   free_limit=AUTH["free_limit"])

@app.route("/api/auth/logout", methods=["POST"])
def _auth_logout():
    AUTH.update(authed=False, token=None, kind=None, email=None,
                member=False, used=None, free_limit=None, remaining=None)
    return jsonify(ok=True)


@app.route("/api/loras", methods=["GET"])
def _list_loras():
    def _vid(u):
        m = re.search(r"/api/download/models/(\d+)", u or "")
        return int(m.group(1)) if m else None
    return jsonify(loras=[{"name": n, "scale": i["scale"],
                           "attached": i["attached"],
                           "triggers": i.get("triggers") or [],
                           "version_id": _vid(i.get("url"))}
                          for n, i in STATE["loras"].items()])

@app.route("/api/loras/loaded", methods=["GET"])
def _loaded_versions():
    """Return the Civitai version ids currently resident: every loaded LoRA
    plus the active base model. The browser uses this to gray out cards for
    things already in VRAM so they aren't downloaded again."""
    lora_vids, ckpt_vids = [], []
    lora_map = {}    # version_id -> {name, scale}
    for nm, info in STATE["loras"].items():
        vid = _version_id_in_url(info.get("url"))
        if vid:
            lora_vids.append(vid)
            lora_map[vid] = {"name": nm, "scale": info.get("scale", 1.0)}
    # Base model: model_ref is either an HF repo id (no version) or a local
    # path like /content/models/civitai_<versionid>.safetensors, or we kept
    # the original download URL — try to recover a version id from any of them.
    ref = STATE.get("model_ref") or ""
    mref_url = STATE.get("model_url") or ""
    for cand in (mref_url, ref):
        vid = _version_id_in_url(cand)
        if not vid:
            m = re.search(r"civitai_(\d+)", cand)
            vid = m.group(1) if m else None
        if vid:
            ckpt_vids.append(vid)
            break
    return jsonify(lora_version_ids=lora_vids,
                   checkpoint_version_ids=ckpt_vids,
                   lora_map=lora_map)

@app.route("/api/loras/add", methods=["POST"])
def _add_lora():
    d = request.get_json(force=True)
    ok, reason = register_lora(d.get("name") or "lora",
                               (d.get("url") or "").strip(),
                               float(d.get("scale", 1.0)),
                               triggers=d.get("triggers") or [])
    return (jsonify(ok=True), 200) if ok else (jsonify(ok=False, error=reason), 400)

@app.route("/api/loras/inspect", methods=["POST"])
def _inspect_lora():
    """Download a LoRA, read its header to detect format + base model, then
    delete the temp file. Lets you check compatibility before a real load."""
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not url.lower().startswith("http"):
        return jsonify(ok=False, error="URL must start with http"), 400
    tmp_name = "inspect_" + uuid.uuid4().hex[:8]
    try:
        path = download_lora(url, dest_name=tmp_name)
    except Exception as e:
        return jsonify(ok=False, error=f"download failed — {e}"), 400
    try:
        info = inspect_lora_file(path)
        _log_lora_inspection(tmp_name, info)
    finally:
        try: os.remove(path)
        except Exception: pass
    return jsonify(ok=True, info=info)

@app.route("/api/loras/update", methods=["POST"])
def _update_lora():
    d = request.get_json(force=True)
    name = d.get("name")
    if name not in STATE["loras"]:
        return jsonify(ok=False, error="No such LoRA."), 404
    new_scale = float(d.get("scale", 1.0))
    STATE["loras"][name]["scale"] = new_scale
    # PEFT adapters pick up the new strength via apply_loras()/set_adapters
    # at the start of the next generation. LyCORIS wrappers are driven by
    # their own multiplier — set it live so the next render uses it.
    if name in STATE.get("_lycoris_loras", set()):
        _lycoris_set_scale(name, new_scale)
    return jsonify(ok=True)

@app.route("/api/loras/remove", methods=["POST"])
def _remove_lora():
    d = request.get_json(force=True)
    return (jsonify(ok=True) if remove_lora(d.get("name"))
            else (jsonify(ok=False, error="No such LoRA."), 404))

def _version_id_in_url(u):
    m = re.search(r"/api/download/models/(\d+)", u or "")
    return m.group(1) if m else None

def _civitai_version_info(version_id):
    """Fetch a Civitai model-version's metadata: display name, base model,
    trigger words. Best-effort; returns {} on any failure."""
    try:
        r = _requests.get(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers=_civitai_headers(), timeout=20)
        if not r.ok:
            return {}
        j = r.json()
        mn = (j.get("model") or {}).get("name")
        vn = j.get("name")
        name = f"{mn} ({vn})" if (mn and vn) else (mn or vn)
        return {"name": name,
                "base_model": j.get("baseModel"),
                "triggers": (j.get("trainedWords") or [])[:6]}
    except Exception:
        return {}

def _resolve_lora_name(version_id):
    """Look up a LoRA's real display name from its Civitai version id, so
    remixed recipes show 'DetailTweaker' instead of 'lora_372220'. Best-effort;
    falls back to the version-id name on any failure."""
    try:
        r = _requests.get(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers=_civitai_headers(), timeout=15)
        if r.ok:
            j = r.json()
            mn = (j.get("model") or {}).get("name")
            vn = j.get("name")
            if mn and vn:
                return f"{mn} ({vn})"
            return mn or vn
    except Exception:
        pass
    return None

@app.route("/api/loras/ensure", methods=["POST"])
def _ensure_loras():
    """Load a set of LoRAs (from a sample's resources). Dedupes by Civitai
    version id so an already-loaded LoRA isn't re-fetched. If replace=True,
    also REMOVES currently-loaded LoRAs that aren't in this set — so remixing a
    different image gives that image's LoRAs, not the union with old ones.
    Resolves real names from Civitai version ids. Returns per-item status."""
    d = request.get_json(force=True)
    items = d.get("loras") or []
    replace = bool(d.get("replace", False))

    # Map currently-loaded LoRAs by version id.
    loaded_by_vid = {}
    for nm, info in STATE["loras"].items():
        vid = _version_id_in_url(info.get("url"))
        if vid:
            loaded_by_vid[vid] = nm
    have = set(loaded_by_vid.keys())
    wanted_vids = set()

    results = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            continue
        vid = _version_id_in_url(url)
        if vid:
            wanted_vids.add(vid)
        if vid and vid in have:
            results.append({"url": url, "status": "already-loaded"})
            continue
        # Resolve a real name; fall back to the supplied name, then lora_<vid>.
        name = it.get("name")
        if not name and vid:
            name = _resolve_lora_name(vid)
        name = name or (f"lora_{vid}" if vid else "lora")
        scale = it.get("weight")
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        ok, reason = register_lora(name, url, scale)
        results.append({"url": url, "status": "added" if ok else "failed",
                        "error": None if ok else reason})
        if ok and vid:
            have.add(vid)

    removed = []
    if replace:
        # Drop any loaded LoRA whose version isn't part of this recipe.
        for vid, nm in loaded_by_vid.items():
            if vid not in wanted_vids:
                if remove_lora(nm):
                    removed.append(nm)
    return jsonify(ok=True, results=results, removed=removed)

# ---- Civitai model browser ---------------------------------------------
def _resize_civitai_url(url, width=320):
    """Civitai image CDN URLs look like
        https://image.civitai.com/<hash>/<uuid>.<ext>
    optionally with a transform segment like '/width=450/' or
    '/anim=false,width=450/' placed as the FIRST path segment after the host.
    Requesting a small width gives a fast thumbnail. If a width transform is
    already present we just rewrite the number; otherwise we insert one right
    after the host (NOT before the filename — the CDN expects it up front)."""
    if not url:
        return url
    # Already has a width transform anywhere in the path: rewrite the number.
    if re.search(r"width=\d+", url):
        return re.sub(r"width=\d+", f"width={width}", url, count=1)
    # Insert a width transform as the first path segment after the host.
    m = re.match(r"(https://image\.civitai\.com)/(.*)", url)
    if m:
        return f"{m.group(1)}/width={width}/{m.group(2)}"
    return url

def _civitai_search(query, base_model, nsfw, cursor, page, sort, mtype="LORA",
                    period="AllTime", tag=None, gen_only=False, limit=60):
    """Proxy Civitai's /api/v1/models. Server-side so the browser doesn't hit
    CORS and so the API key raises rate limits.

    Pagination: full-text `query` REQUIRES cursor pagination (page+query is a
    400, per Civitai docs), and page*limit>1000 is a 429. So we always use
    cursor pagination here and return metadata.nextCursor for infinite scroll.
    `page` is accepted only as a no-query fallback."""
    params = {"types": mtype, "limit": limit,
              "period": period or "AllTime"}
    q = (query or "").strip()
    if q:
        params["query"] = q                     # forces cursor pagination
        # A text query is relevance-ranked by Meilisearch. Pairing it with an
        # expensive global sort (e.g. Highest Rated) makes Civitai time out
        # (408), and the sort would also break cursor continuity across pages.
        # So we only apply an explicit sort when browsing WITHOUT a query.
    else:
        params["sort"] = sort or "Most Downloaded"
    if base_model and base_model != "all":
        params["baseModels"] = base_model
    if tag:
        params["tag"] = tag
    if gen_only:
        params["supportsGeneration"] = "true"
    # NSFW / browsing level. This is the #1 reason adult LoRAs "don't show":
    # nsfw=true alone only includes up to R-rated — X and XXX models are
    # excluded by the API's browsing-level filter (Civitai issue #1795). The
    # real lever is browsingLevel, a BITMASK integer (1=PG 2=PG13 4=R 8=X
    # 16=XXX, additive; 31 = everything) which takes precedence over nsfw.
    # See go-civitai-downloader. We pass the full range when nsfw is on so X/XXX
    # LoRAs are included, and PG+PG13 only when nsfw is off.
    if nsfw:
        params["nsfw"] = "true"
        params["browsingLevel"] = 31      # PG|PG13|R|X|XXX — all content
    else:
        params["nsfw"] = "false"
        params["browsingLevel"] = 3       # PG|PG13 only
    # Cursor wins; only fall back to page when there's no query and no cursor.
    if cursor:
        params["cursor"] = cursor
    elif not q and page:
        params["page"] = page
    headers = {}
    key = CIVITAI_API_KEY or _get_secret("CIVITAI_API_KEY", verbose=False)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def _do(p):
        return _requests.get("https://civitai.com/api/v1/models",
                             params=p, headers=headers, timeout=45)

    r = _do(params)
    if r.status_code == 400 and "browsingLevel" in params:
        # If Civitai's schema rejects browsingLevel (it's ZodError-prone), drop
        # it and fall back to the plain nsfw flag rather than failing the search.
        _log("  [civitai] browsingLevel rejected — retrying without it.")
        params.pop("browsingLevel", None)
        r = _do(params)
    if r.status_code == 400 and not cursor and "page" in params:
        # Defensive: retry once forcing cursor mode if page was rejected.
        params.pop("page", None)
        r = _do(params)
    # Civitai 408/504s when a text `query` is combined with an expensive sort
    # (e.g. Highest Rated) across the whole catalog — Meilisearch + global sort
    # blows past Cloudflare's timeout. Retry without the sort (relevance order,
    # which is the natural ordering for a text search anyway).
    if r.status_code in (408, 504) and q and params.get("sort") not in (None,):
        _log(f"  [civitai] {r.status_code} with query+sort='{params.get('sort')}'"
             " — retrying with relevance sort.")
        p2 = dict(params); p2.pop("sort", None)
        try:
            r2 = _do(p2)
            if r2.status_code == 200:
                r = r2
        except Exception:
            pass
    # Still timing out? One short backoff retry for transient cases.
    if r.status_code in (408, 504):
        time.sleep(1.5)
        try:
            p3 = dict(params); p3.pop("sort", None)
            r3 = _do(p3)
            if r3.status_code == 200:
                r = r3
        except Exception:
            pass
    if r.status_code in (408, 504):
        raise RuntimeError(
            "Civitai timed out on this search. Their full-text search can be "
            "slow when combined with a sort or a very broad term — try a more "
            "specific term, switch the sort to 'Most downloaded', or narrow "
            "the base-model filter.")
    r.raise_for_status()
    data = r.json()

    def _parse_models(d):
        rows = []
        for m in d.get("items", []):
            vers = m.get("modelVersions") or []
            if not vers:
                continue
            v = vers[0]                       # newest/primary version
            raw_thumb = None
            for im in (v.get("images") or []):
                if im.get("type", "image") == "image" and im.get("url"):
                    if (not nsfw) and (im.get("nsfwLevel", 1) or 1) > 4:
                        continue
                    raw_thumb = im["url"]
                    break
            dl = f"https://civitai.com/api/download/models/{v.get('id')}"
            rows.append({
                "model_id": m.get("id"),
                "version_id": v.get("id"),
                "name": m.get("name"),
                "type": m.get("type"),
                "base_model": v.get("baseModel"),
                "nsfw": bool(m.get("nsfw")),
                "nsfw_level": m.get("nsfwLevel"),
                "thumb": _resize_civitai_url(raw_thumb, 320) if raw_thumb else None,
                "download_url": dl,
                "triggers": (v.get("trainedWords") or [])[:6],
                "downloads": (m.get("stats") or {}).get("downloadCount"),
            })
        return rows

    out = _parse_models(data)
    meta = data.get("metadata") or {}

    # Civitai bug (#1848): a text query can return an EMPTY first page that
    # still carries a nextCursor — so the UI shows "no results" when results
    # actually exist further in. Follow the cursor a few times until we get
    # real items (or run out), so a valid query doesn't look like a failure.
    _follow = 0
    while q and not out and meta.get("nextCursor") and _follow < 4:
        _follow += 1
        p2 = dict(params)
        p2["cursor"] = meta["nextCursor"]
        p2.pop("page", None)
        try:
            r2 = _do(p2)
            if r2.status_code != 200:
                break
            data = r2.json()
            out = _parse_models(data)
            meta = data.get("metadata") or {}
        except Exception:
            break
    if _follow:
        _log(f"  [civitai] followed {_follow} empty page(s) for query='{q}'")

    _log(f"  [civitai] params types={params.get('types')} "
         f"base={params.get('baseModels')} sort={params.get('sort')} "
         f"period={params.get('period')} nsfw={params.get('nsfw')} "
         f"tag={params.get('tag')} cursor={'yes' if params.get('cursor') else 'no'}"
         f" -> items={len(data.get('items', []))} "
         f"returned={len(out)} nextCursor={'yes' if meta.get('nextCursor') else 'NONE'} "
         f"totalItems={meta.get('totalItems')}")
    return {"items": out, "next_cursor": meta.get("nextCursor"),
            "total_items": meta.get("totalItems")}

def _parse_img_tags(html):
    """Pull image URLs + dims out of arbitrary HTML by scanning <img> tags.
    Prefers lazy-load attributes (data-src etc.) over src (often a placeholder).
    Best-effort: sites vary, so this stays permissive."""
    results, seen = [], set()
    for tag in re.findall(r"<img\b[^>]*>", html, flags=re.IGNORECASE):
        def _attr(name):
            m = re.search(name + r'\s*=\s*["\']([^"\']+)["\']',
                          tag, flags=re.IGNORECASE)
            return m.group(1).strip() if m else None
        src = (_attr("data-src") or _attr("data-original")
               or _attr("data-lazy-src") or _attr("src"))
        if not src:
            continue
        low = src.lower()
        if (low.startswith("data:") or "1px" in low or "blank" in low
                or "spacer" in low or "placeholder" in low):
            continue
        if not re.search(r"\.(jpe?g|png|webp|gif|bmp)(\?|$)", low):
            if not low.startswith("http"):
                continue
        if src in seen:
            continue
        seen.add(src)
        w = _attr("width") or "0"
        h = _attr("height") or "0"
        try:
            w = int(re.sub(r"[^\d]", "", w) or 0)
            h = int(re.sub(r"[^\d]", "", h) or 0)
        except ValueError:
            w = h = 0
        results.append({"full": src, "thumb": src, "w": w, "h": h,
                        "title": _attr("alt") or ""})
    return results

_WS_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

def _ws_provider_duckduckgo(q, page, cursor=None):
    """DuckDuckGo images. Results come from /i.js (JSON), gated by a 'vqd'
    token scraped from the HTML page. DDG aggressively 403s bare requests, so
    we (1) warm up a session to pick up cookies, then (2) send the full set of
    headers a real browser XHR sends to i.js. Paginates via the 's' offset.
    Note: from datacenter IPs (e.g. Colab) DDG may still 403 — it's an
    unofficial endpoint with bot defenses we can't fully defeat here."""
    s = _requests.Session()
    base_hdrs = {
        "User-Agent": _WS_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    s.headers.update(base_hdrs)
    # 1) warm up: hit the homepage (sets cookies) then the search page (token).
    try:
        s.get("https://duckduckgo.com/", timeout=20,
              headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    except Exception:
        pass
    tok = s.get("https://duckduckgo.com/", params={"q": q, "ia": "images",
                "iax": "images"}, timeout=20,
                headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    tok.raise_for_status()
    m = (re.search(r'vqd=["\']?([\d-]+)', tok.text)
         or re.search(r'vqd=([\d-]+)&', tok.text))
    if not m:
        raise RuntimeError("could not get DuckDuckGo token (vqd)")
    vqd = m.group(1)
    params = {"l": "us-en", "o": "json", "q": q, "vqd": vqd, "f": ",,,,,",
              "p": "1"}
    if cursor:                      # cursor is DDG's 's' offset for the next page
        params["s"] = cursor
    # 2) the i.js XHR — send the exact headers a browser fetch includes, or DDG
    # returns 403.
    rj = s.get("https://duckduckgo.com/i.js", params=params, timeout=20,
               headers={
                   "Accept": "application/json, text/javascript, */*; q=0.01",
                   "Referer": "https://duckduckgo.com/",
                   "X-Requested-With": "XMLHttpRequest",
                   "Sec-Fetch-Dest": "empty",
                   "Sec-Fetch-Mode": "cors",
                   "Sec-Fetch-Site": "same-origin",
               })
    rj.raise_for_status()
    data = rj.json()
    out = []
    for it in (data.get("results") or []):
        full = it.get("image")
        if not full:
            continue
        out.append({"full": full, "thumb": it.get("thumbnail") or full,
                    "w": int(it.get("width") or 0),
                    "h": int(it.get("height") or 0),
                    "title": it.get("title") or ""})
    nxt = data.get("next")
    ncur = None
    if nxt:
        mm = re.search(r"[?&]s=(\d+)", nxt)
        if mm:
            ncur = mm.group(1)
    return out, ncur

def _ws_provider_bing(q, page, cursor=None):
    """Bing Images. Scrapes the results page; paginates via the 'first' offset
    (35 per page). Image URLs live in the murl field of each tile's m=JSON."""
    first = (page - 1) * 35 + 1
    url = ("https://www.bing.com/images/async?q=" + _url_quote(q)
           + f"&first={first}&count=35&mmasync=1")
    r = _requests.get(url, timeout=20,
                      headers={"User-Agent": _WS_UA,
                               "Referer": "https://www.bing.com/"})
    r.raise_for_status()
    out, seen = [], set()
    # Each tile carries a JSON blob in m="..."; murl is the full image.
    for blob in re.findall(r'm="(\{[^"]+\})"', r.text):
        try:
            j = json.loads(blob.replace("&quot;", '"'))
        except Exception:
            continue
        full = j.get("murl")
        if not full or full in seen:
            continue
        seen.add(full)
        out.append({"full": full, "thumb": j.get("turl") or full,
                    "w": 0, "h": 0, "title": j.get("t") or ""})
    has_more = len(out) > 0
    return out, (str(page + 1) if has_more else None)

def _ws_provider_generic(tmpl, q, page, cursor=None):
    """Generic scraped site. Fetches the URL template (with {q} filled) and
    parses <img> tags. Supports an optional {page} placeholder for pagination;
    if absent, only page 1 is available."""
    url = tmpl.replace("{q}", _url_quote(q))
    paged = "{page}" in tmpl
    if paged:
        url = url.replace("{page}", str(page))
    r = _requests.get(url, timeout=20,
                      headers={"User-Agent": _WS_UA,
                               "Accept": "text/html,application/xhtml+xml"})
    r.raise_for_status()
    results = _parse_img_tags(r.text)
    # Only claim "more" if the template can page AND this page returned images.
    has_more = paged and len(results) > 0
    return results, (str(page + 1) if has_more else None)

_DDGS_OK = None      # lazy import flag: None=untried, True/False after first use
def _ensure_ddgs():
    """Lazy-install + import ddgs (the maintained metasearch lib). It aggregates
    DuckDuckGo, Bing, Google etc. with browser impersonation, which is far more
    robust than hand-rolled scraping. Installed on first use so a fresh runtime
    doesn't pay for it unless web search is actually used."""
    global _DDGS_OK
    if _DDGS_OK is not None:
        return _DDGS_OK
    try:
        try:
            import ddgs  # noqa
        except Exception:
            import subprocess, sys
            _log("  [websearch] installing ddgs (one-time)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--upgrade", "ddgs"], check=False)
            import ddgs  # noqa
        _DDGS_OK = True
    except Exception as e:
        _log(f"  [websearch] ddgs unavailable: {e}")
        _DDGS_OK = False
    return _DDGS_OK

def _ddgs_images(q, page, backend):
    """Search images via the ddgs metasearch library. backend is 'auto' (all
    engines, best coverage) or a specific one ('duckduckgo','bing','google').
    Returns (results, next_page) — ddgs paginates by page number."""
    if not _ensure_ddgs():
        raise RuntimeError("ddgs library not available")
    from ddgs import DDGS
    rows = []
    with DDGS() as d:
        # ddgs 9.x signature: images(query=, max_results=, page=, backend=).
        try:
            res = d.images(query=q, max_results=60, page=page, backend=backend,
                           safesearch="off")
        except TypeError:
            # Older signature fallback (keywords=, no page/backend).
            res = d.images(keywords=q, max_results=60)
        for it in (res or []):
            full = it.get("image") or it.get("url")
            if not full:
                continue
            rows.append({"full": full,
                         "thumb": it.get("thumbnail") or full,
                         "w": int(it.get("width") or 0),
                         "h": int(it.get("height") or 0),
                         "title": it.get("title") or ""})
    # ddgs returns a page at a time; assume more exist if this page was full.
    has_more = len(rows) >= 40
    return rows, (str(page + 1) if has_more else None)

def _provider_best(q, page, cursor=None):
    """Primary engine: ddgs in auto mode (aggregates all search backends).
    Falls back to the raw Bing scraper if ddgs is unavailable/empty."""
    try:
        rows, nxt = _ddgs_images(q, page, "auto")
        if rows:
            return rows, nxt
    except Exception as e:
        _log(f"  [websearch] ddgs auto failed ({e}); falling back to Bing scrape")
    return _ws_provider_bing(q, page, cursor)

def _provider_ddgs_ddg(q, page, cursor=None):
    """DuckDuckGo via ddgs (more robust than our raw i.js scrape), with the raw
    scrape as a last resort."""
    try:
        rows, nxt = _ddgs_images(q, page, "duckduckgo")
        if rows:
            return rows, nxt
    except Exception as e:
        _log(f"  [websearch] ddgs/ddg failed ({e}); trying raw i.js")
    return _ws_provider_duckduckgo(q, page, cursor)

def _provider_ddgs_bing(q, page, cursor=None):
    """Bing via ddgs, raw Bing scrape as fallback."""
    try:
        rows, nxt = _ddgs_images(q, page, "bing")
        if rows:
            return rows, nxt
    except Exception as e:
        _log(f"  [websearch] ddgs/bing failed ({e}); trying raw scrape")
    return _ws_provider_bing(q, page, cursor)

def _provider_ddgs_google(q, page, cursor=None):
    """Google Images via ddgs (no good raw fallback, so this is ddgs-only)."""
    return _ddgs_images(q, page, "google")

# ---- booru providers ---------------------------------------------------
# Booru imageboards have real, tag-based JSON APIs with no SafeSearch layer —
# the right source for tag-searchable / adult content that general engines
# filter out. Tags are space-separated; multi-word tags use underscores.
def _booru_query_to_tags(q):
    # Users type natural queries; booru tags use underscores. Convert spaces
    # within the query to '+' between tags, leaving any underscores intact.
    return _url_quote(q.strip())

def _booru_get_json(url, headers):
    """Fetch a booru API and return a clean LIST OF DICT posts, tolerating the
    several shapes booru APIs return: a bare list, a {"post":[...]} or
    {"posts":[...]} wrapper, a single dict, or an error string/empty body.
    Returns [] for anything that isn't usable (this is what was throwing
    'str object has no attribute get' — some boorus return a string on no
    results or on error)."""
    r = _requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    try:
        body = r.json()
    except Exception:
        return []
    # unwrap common wrappers
    if isinstance(body, dict):
        for k in ("post", "posts", "results", "data"):
            if isinstance(body.get(k), list):
                body = body[k]
                break
        else:
            # a single post dict, or a wrapper with no list -> treat as 0/1 post
            body = [body] if body.get("file_url") or body.get("file") else []
    if not isinstance(body, list):
        return []
    # keep only real post dicts; drop strings/None/etc.
    return [it for it in body if isinstance(it, dict)]

def _ws_provider_rule34(q, page, cursor=None):
    """Rule34.xxx — explicit by default, no auth. Paginates by pid (0-based)."""
    pid = page - 1
    url = ("https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1"
           f"&limit=50&pid={pid}&tags={_booru_query_to_tags(q)}")
    items = _booru_get_json(url, {"User-Agent": _WS_UA})
    out = []
    for it in items:
        full = it.get("file_url")
        if not full:
            continue
        out.append({"full": full, "thumb": it.get("preview_url") or full,
                    "w": int(it.get("width") or 0),
                    "h": int(it.get("height") or 0),
                    "title": (it.get("tags") or "")[:80]})
    return out, (str(page + 1) if out else None)

def _ws_provider_gelbooru(q, page, cursor=None):
    """Gelbooru — reads work without keys. Explicit content is included by
    default via the API (the site's SFW default is a browser cookie, not an
    API filter). pid is 0-based; posts are under 'post'."""
    pid = page - 1
    url = ("https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1"
           f"&limit=50&pid={pid}&tags={_booru_query_to_tags(q)}")
    items = _booru_get_json(url, {"User-Agent": _WS_UA})
    out = []
    for it in items:
        full = it.get("file_url")
        if not full:
            continue
        out.append({"full": full, "thumb": it.get("preview_url") or full,
                    "w": int(it.get("width") or 0),
                    "h": int(it.get("height") or 0),
                    "title": (it.get("tags") or "")[:80]})
    return out, (str(page + 1) if out else None)

def _ws_provider_e621(q, page, cursor=None):
    """e621.net — explicit by default. REQUIRES a descriptive User-Agent or it
    403s. Paginates by page (1-based). Image URL is nested under file.url."""
    url = ("https://e621.net/posts.json"
           f"?limit=50&page={page}&tags={_booru_query_to_tags(q)}")
    items = _booru_get_json(
        url, {"User-Agent": "DiTStudio/1.0 (Colab personal use)"})
    out = []
    for it in items:
        f = it.get("file") if isinstance(it.get("file"), dict) else {}
        full = f.get("url")
        if not full:
            continue
        prev = it.get("preview") if isinstance(it.get("preview"), dict) else {}
        tags = it.get("tags") if isinstance(it.get("tags"), dict) else {}
        out.append({"full": full, "thumb": prev.get("url") or full,
                    "w": int(f.get("width") or 0),
                    "h": int(f.get("height") or 0),
                    "title": ", ".join((tags.get("general") or [])[:6])})
    return out, (str(page + 1) if out else None)

def _ws_provider_danbooru(q, page, cursor=None):
    """Danbooru — posts.json (1-based page). NOTE: Danbooru hides explicit posts
    for anonymous callers and limits anonymous searches to 2 tags. For reliable
    adult results prefer Rule34 / Gelbooru / e621."""
    url = ("https://danbooru.donmai.us/posts.json"
           f"?limit=50&page={page}&tags={_booru_query_to_tags(q)}")
    items = _booru_get_json(url, {"User-Agent": _WS_UA})
    out = []
    for it in items:
        full = it.get("file_url") or it.get("large_file_url")
        if not full:
            continue
        out.append({"full": full, "thumb": it.get("preview_file_url") or full,
                    "w": int(it.get("image_width") or 0),
                    "h": int(it.get("image_height") or 0),
                    "title": (it.get("tag_string") or "")[:80]})
    return out, (str(page + 1) if out else None)

# ---- HTML-scrape adult providers (NO API — best-effort, unreliable) ----
# ImageFap and PornPics have no public API and actively block scrapers
# (Cloudflare, hotlink protection, gallery-not-image structure). These parse
# the search page's <img> tags via the generic parser. Expect them to be flaky
# and to sometimes return gallery thumbnails rather than full images — that's
# the nature of scraping these sites, accepted by design here.
def _scrape_site(url, headers, base):
    """Fetch a page and parse image URLs from its <img> tags, resolving
    relative URLs against `base`. Returns the generic-parser result list."""
    r = _requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    rows = _parse_img_tags(r.text)
    out = []
    for it in rows:
        u = it.get("full") or ""
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            u = base.rstrip("/") + u
        elif not u.startswith("http"):
            continue
        it["full"] = u
        th = it.get("thumb") or u
        if th.startswith("//"): th = "https:" + th
        elif th.startswith("/"): th = base.rstrip("/") + th
        it["thumb"] = th
        out.append(it)
    return out

def _ws_provider_pornpics(q, page, cursor=None):
    """PornPics.com SEARCH — returns galleries (not individual images). Each
    search tile is <a class='rel-link' href='/galleries/...'> wrapping an
    <img data-src='cdni...460...jpg'> cover. We return the cover as the thumb
    and stash the gallery URL in 'gallery' so the UI can drill in and pull the
    full-size images. Server-rendered, so this parses reliably. Paginates &page."""
    qs = q.strip().replace(" ", "+")
    url = (f"https://www.pornpics.com/?q={_url_quote(qs).replace('%2B','+')}"
           + (f"&page={page}" if page > 1 else ""))
    r = _requests.get(url, timeout=20, proxies=_ws_proxies(), headers={
        "User-Agent": _WS_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.pornpics.com/",
    })
    r.raise_for_status()
    html = r.text
    out, seen = [], set()
    for a in re.findall(r"<a\s+class='rel-link'[^>]*>.*?</a>", html, re.S):
        href = re.search(r"href=['\"]([^'\"]+)['\"]", a)
        ds = re.search(r"data-src=['\"]([^'\"]+)['\"]", a)
        if not ds:
            continue
        cover = ds.group(1)
        if "cdni.pornpics.com" not in cover:
            continue
        gal = href.group(1) if href else ""
        if gal in seen:
            continue
        seen.add(gal)
        tm = re.search(r"title=['\"]([^'\"]*)['\"]", a)
        out.append({"full": cover, "thumb": cover, "w": 0, "h": 0,
                    "title": (tm.group(1) if tm else "")[:90],
                    "gallery": gal})       # <-- gallery URL to drill into
    pm = re.search(r"P_MAX\s*=\s*(\d+)", html)
    has_more = bool(out) and (not pm or page < int(pm.group(1)))
    return out, (str(page + 1) if has_more else None)

def _pornpics_gallery_images(gallery_url):
    """Parse a PornPics GALLERY page into its full-size images. On a gallery
    page the rel-link href is the full image (cdni.../1280/...jpg) and
    data-pswp-width/height give real dimensions; the inner img data-src is the
    460px thumb. Returns a list of {full, thumb, w, h, title}."""
    if not gallery_url.startswith("http"):
        gallery_url = "https://www.pornpics.com" + gallery_url
    r = _requests.get(gallery_url, timeout=20, proxies=_ws_proxies(), headers={
        "User-Agent": _WS_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.pornpics.com/",
    })
    r.raise_for_status()
    html = r.text
    title = ""
    tm = re.search(r"<title>(.*?)</title>", html, re.S)
    if tm:
        title = tm.group(1).split(" - PornPics")[0].strip()[:90]
    out, seen = [], set()
    for a in re.findall(r"<a\s+class='rel-link'[^>]*>.*?</a>", html, re.S):
        href = re.search(r"href=['\"]([^'\"]+)['\"]", a)
        if not href:
            continue
        full = href.group(1)
        # On a gallery page the href IS the full image (not a /galleries/ link).
        if "cdni.pornpics.com" not in full:
            continue
        if full in seen:
            continue
        seen.add(full)
        ds = re.search(r"data-src=['\"]([^'\"]+)['\"]", a)
        thumb = ds.group(1) if ds else full
        w = re.search(r"data-pswp-width=['\"](\d+)['\"]", a)
        h = re.search(r"data-pswp-height=['\"](\d+)['\"]", a)
        out.append({"full": full, "thumb": thumb,
                    "w": int(w.group(1)) if w else 0,
                    "h": int(h.group(1)) if h else 0,
                    "title": title})
    return out, title

def _ws_provider_imagefap(q, page, cursor=None, gen=""):
    """ImageFap.com SEARCH — returns galleries (server-rendered, reliable).
    Each gallery is a <tr id="GID"> with <a class="link3" href="/gallery.php?
    gid=GID&pgid=" title="View ..."> plus up to 4 cover thumbnails. We return
    the first cover as the thumb and stash the gallery id so the UI can drill
    in. Paginates with &page=N (0-based). Optional gen=<id> filters by category
    (confirmed: search=<q>&gen=<id> filters by both)."""
    pg = page - 1
    url = (f"https://www.imagefap.com/gallery.php?search={_url_quote(q)}"
           f"&page={pg}")
    if gen:
        url += f"&gen={gen}"
    r = _requests.get(url, timeout=20, proxies=_ws_proxies(), headers={
        "User-Agent": _WS_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.imagefap.com/",
    })
    r.raise_for_status()
    html = r.text
    out = []
    # Split into per-gallery blocks on the <tr id="digits"> rows, then within
    # each block grab the title link (gid) and the first cover thumbnail.
    blocks = re.split(r'<tr\s+id="(\d+)"', html)
    # re.split keeps captured ids as alternating list elements after the first.
    for i in range(1, len(blocks), 2):
        gid = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        tl = re.search(r'<a\s+title="([^"]*)"\s+class="link3"\s+href="[^"]*gid='
                       + re.escape(gid), body)
        title = tl.group(1) if tl else ""
        if title.startswith("View "):
            title = title[5:]
        # first real cover thumb (skip /img/unknown.jpg placeholders)
        cover = None
        for m in re.finditer(r'src="(https://cdnc\.imagefap\.com/images/mini/[^"]+)"', body):
            cover = m.group(1); break
        if not cover:
            continue
        out.append({"full": cover, "thumb": cover, "w": 0, "h": 0,
                    "title": (title or "ImageFap gallery")[:90],
                    "gallery": f"imagefap:{gid}"})   # marker the gallery endpoint understands
    # "next" link present if there are more pages.
    has_more = bool(out) and ("search=" in html and ":: next ::" in html)
    return out, (str(page + 1) if has_more else None)

def _imagefap_gallery_images(gid, page=0):
    """Parse ONE page of an ImageFap gallery (by gid) into its photos. Each photo
    is <a name="PHOTO_ID" href="/photo/PHOTO_ID/"><img src="...mini...">. Returns
    (list, has_more). The list items carry the per-photo marker so the UI can
    fetch the full image. Retries once if the page comes back without photos
    (ImageFap occasionally serves a photo-less interstitial)."""
    rx = r'<a\s+name="(\d+)"\s+href="/photo/\1/">\s*<img[^>]+src="([^"]+)"'
    hdrs = {"User-Agent": _WS_UA,
            "Accept": "text/html,application/xhtml+xml",
            "Referer": "https://www.imagefap.com/"}
    url = f"https://www.imagefap.com/pictures/{gid}/x?gid={gid}&page={page}&view=0"
    out, seen = [], set()
    html = ""
    for attempt in range(3):     # retry photo-less pages (transient interstitials)
        try:
            r = _requests.get(url, timeout=20, headers=hdrs, proxies=_ws_proxies())
            r.raise_for_status()
            html = r.text
        except Exception as e:
            _log(f"  [websearch] imagefap gallery {gid} p{page} fetch error: {e}")
            html = ""
        n = len(re.findall(rx, html))
        nlinks = len(re.findall(r'href="/photo/\d+/"', html))
        _log(f"  [websearch] imagefap gallery {gid} p{page} attempt{attempt+1}: "
             f"{len(html)} chars, {nlinks} photo-links, {n} matches")
        if n > 0:
            break
        time.sleep(0.6)
    for m in re.finditer(rx, html):
        pid, thumb = m.group(1), m.group(2)
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"photo": f"imagefapphoto:{pid}:{gid}",
                    "thumb": thumb, "full": thumb,
                    "w": 0, "h": 0, "title": ""})
    # More pages if this page was full (ImageFap shows ~24/page; if we got a
    # healthy page, assume there may be another).
    has_more = len(out) >= 20
    return out, has_more

def _imagefap_full_image(pid, gid=""):
    """Resolve an ImageFap /photo/<id>/ page to its full-size image AND fetch
    the bytes IN THE SAME SESSION — the image CDN only serves the bytes to the
    session whose PHPSESSID cookie was set by loading the photo page (proven:
    cold/other-session fetches 403, same-session fetches return the real image).
    The browser can't do this (CORS locks the bytes), so the backend does it and
    returns a data URL. Returns (data_url, w, h) — data_url is None on failure."""
    s = _requests.Session()
    s.headers.update({"User-Agent": _WS_UA, "Accept-Language": "en-US,en;q=0.9"})
    px = _ws_proxies()
    url = f"https://www.imagefap.com/photo/{pid}/"
    if gid:
        url += f"?gid={gid}"
    r = s.get(url, timeout=20, proxies=px, headers={
        "Accept": "text/html,application/xhtml+xml",
        "Referer": (f"https://www.imagefap.com/pictures/{gid}/x" if gid
                    else "https://www.imagefap.com/"),
    })
    r.raise_for_status()
    html = r.text
    m = re.search(r'id="mainPhoto"[^>]*\ssrc="([^"]+)"', html)
    if not m:
        m = re.search(r'(https://cdnc\.imagefap\.com/images/full/[^"\']+)', html)
    full = m.group(1) if m else None
    w = h = 0
    dm = re.search(r"Dimension</td>\s*<td[^>]*>\s*(\d+)x(\d+)", html)
    if dm:
        w, h = int(dm.group(1)), int(dm.group(2))
    if not full:
        return None, w, h
    # Fetch the image bytes in the SAME session (carries the PHPSESSID cookie).
    ir = s.get(full, timeout=30, proxies=px, headers={
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": url,
    })
    ir.raise_for_status()
    ctype = ir.headers.get("Content-Type", "image/jpeg").split(";")[0]
    if "image" not in ctype or len(ir.content) < 1000:
        return None, w, h
    b64 = base64.b64encode(ir.content).decode("ascii")
    return f"data:{ctype};base64,{b64}", w, h

# Registry of built-in providers. 'All engines' (ddgs auto) is the default and
# most robust for general content. Booru sources are tag-based JSON APIs with
# no SafeSearch — use them for tag-searchable / adult content. The scrape
# providers (PornPics/ImageFap) have NO API and are unreliable by nature.
_WS_PROVIDERS = {
    "best":       {"label": "All engines (best)", "fn": _provider_best},
    "duckduckgo": {"label": "DuckDuckGo",         "fn": _provider_ddgs_ddg},
    "bing":       {"label": "Bing",               "fn": _provider_ddgs_bing},
    "google":     {"label": "Google",             "fn": _provider_ddgs_google},
    "rule34":     {"label": "Rule34",             "fn": _ws_provider_rule34},
    "gelbooru":   {"label": "Gelbooru",           "fn": _ws_provider_gelbooru},
    "e621":       {"label": "e621",               "fn": _ws_provider_e621},
    "danbooru":   {"label": "Danbooru",           "fn": _ws_provider_danbooru},
    "pornpics":   {"label": "PornPics (scrape \u26a0)",  "fn": _ws_provider_pornpics},
    "imagefap":   {"label": "ImageFap (scrape \u26a0)",  "fn": _ws_provider_imagefap},
}

@app.route("/api/websearch/providers", methods=["GET"])
def _web_search_providers():
    """List built-in search providers so the UI can populate its source list."""
    return jsonify(ok=True, providers=[
        {"value": k, "label": v["label"]} for k, v in _WS_PROVIDERS.items()])

_IF_CATS_CACHE = {"cats": None, "ts": 0}
def _imagefap_categories():
    """Scrape ImageFap's live category list from the <select name="gen"> on
    gallery.php. Cached 1h. Returns [{id,name}] with '' = All categories.
    Live so the dropdown never goes stale; falls back to a small built-in set
    if the fetch fails."""
    now = time.time()
    if _IF_CATS_CACHE["cats"] and now - _IF_CATS_CACHE["ts"] < 3600:
        return _IF_CATS_CACHE["cats"]
    cats = [{"id": "", "name": "All categories"}]
    try:
        r = _requests.get("https://www.imagefap.com/gallery.php", timeout=20,
                          proxies=_ws_proxies(), headers={
                              "User-Agent": _WS_UA,
                              "Accept": "text/html,application/xhtml+xml",
                              "Referer": "https://www.imagefap.com/"})
        r.raise_for_status()
        sel = re.search(r'<select[^>]*name="gen"[^>]*>(.*?)</select>', r.text, re.S)
        block = sel.group(1) if sel else ""
        for m in re.finditer(r'<option\s+value="(\d+)/?">([^<\n]+)', block):
            cid, name = m.group(1), m.group(2).strip()
            if cid and name:
                cats.append({"id": cid, "name": name})
    except Exception as e:
        _log(f"  [websearch] imagefap categories fetch failed: {e}")
    if len(cats) <= 1:           # fetch failed — minimal fallback set
        for cid, nm in [("65","Shemale"),("11","Cumshot"),("3","Anal"),("2","Amateur"),
                        ("5","Asian"),("8","Big Tits"),("20","Mature"),("28","Teen"),
                        ("16","Gay"),("1","Lesbian"),("29","Hardcore"),("47","Pornstars")]:
            cats.append({"id": cid, "name": nm})
    _IF_CATS_CACHE["cats"] = cats
    _IF_CATS_CACHE["ts"] = now
    return cats

@app.route("/api/websearch/imagefap_categories", methods=["GET"])
def _web_search_if_categories():
    """Live ImageFap category list for the search-modal dropdown."""
    try:
        return jsonify(ok=True, categories=_imagefap_categories())
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502

@app.route("/api/websearch/gallery", methods=["GET"])
def _web_search_gallery():
    """Fetch a gallery page and return its images, so the UI can drill from a
    search result (a gallery) into the actual photos. Handles PornPics gallery
    URLs and the 'imagefap:<gid>' marker (ImageFap, which needs a further
    per-photo fetch to get full-size — those come back as 'photo' markers)."""
    url = (request.args.get("url") or "").strip()
    if not url:
        return jsonify(ok=False, error="missing gallery url"), 400
    try:
        page = max(0, int(request.args.get("page") or 0))
    except ValueError:
        page = 0
    try:
        if url.startswith("imagefap:"):
            gid = url.split(":", 1)[1]
            images, has_more = _imagefap_gallery_images(gid, page)
            _log(f"  [websearch] imagefap gallery {gid} p{page} -> "
                 f"{len(images)} photo(s){' +more' if has_more else ''}")
            detail = None
            if not images:
                detail = (f"gallery {gid}: parsed 0 photos after retries — see "
                          f"console for page sizes (likely a transient ImageFap "
                          f"interstitial; try reopening)")
            return jsonify(ok=True, results=images, title="",
                           next_page=(page + 1) if has_more else None,
                           detail=detail)
        images, title = _pornpics_gallery_images(url)
        _log(f"  [websearch] gallery -> {len(images)} full image(s)")
        return jsonify(ok=True, results=images, title=title, next_page=None)
    except Exception as e:
        _log(f"  [websearch] gallery fetch failed: {e}")
        return jsonify(ok=False, error=f"gallery fetch failed: {e}"), 502

@app.route("/api/websearch/photo", methods=["GET"])
def _web_search_photo():
    """Resolve an ImageFap 'imagefapphoto:<pid>:<gid>' marker to its full-size
    image, fetched IN-SESSION on the backend and returned as a data URL (the
    browser can't fetch ImageFap's CDN — CORS — and the bytes are only served to
    the session that loaded the photo page, so the backend must do both)."""
    marker = (request.args.get("id") or "").strip()
    if not marker.startswith("imagefapphoto:"):
        return jsonify(ok=False, error="bad photo id"), 400
    parts = marker.split(":")
    pid = parts[1] if len(parts) > 1 else ""
    gid = parts[2] if len(parts) > 2 else ""
    try:
        data_url, w, h = _imagefap_full_image(pid, gid)
        if not data_url:
            return jsonify(ok=False, error="full image not found"), 502
        return jsonify(ok=True, data_url=data_url, w=w, h=h)
    except Exception as e:
        _log(f"  [websearch] photo resolve failed: {e}")
        return jsonify(ok=False, error=f"photo resolve failed: {e}"), 502

@app.route("/api/websearch", methods=["GET"])
def _web_image_search():
    """Search for images server-side (avoids browser CORS). Params:
       q     - the query (required)
       url   - a built-in provider key (e.g. 'duckduckgo','bing') OR a custom
               URL template containing {q} (and optionally {page})
       page  - 1-based page number (for offset-paginated sources)
       cursor- opaque next-page cursor (for cursor-paginated sources like DDG)
    Returns {results, next_cursor, next_page} for infinite scroll."""
    q = (request.args.get("q") or "").strip()
    tmpl = (request.args.get("url") or "").strip() or WEB_IMAGE_SEARCH_URL
    try:
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        page = 1
    cursor = request.args.get("cursor") or None
    if not q:
        return jsonify(ok=False, error="empty query"), 400

    key = tmpl.lower()
    if key in ("ddg",):
        key = "duckduckgo"
    gen = (request.args.get("gen") or "").strip()
    try:
        if key == "imagefap":
            # ImageFap supports a category filter (gen=) combined with search.
            results, ncur = _ws_provider_imagefap(q, page, cursor, gen=gen)
        elif key in _WS_PROVIDERS:
            results, ncur = _WS_PROVIDERS[key]["fn"](q, page, cursor)
        elif "duckduckgo.com" in key:
            results, ncur = _ws_provider_duckduckgo(q, page, cursor)
        else:
            if "{q}" not in tmpl:
                return jsonify(ok=False,
                    error="search URL must contain {q} where the query goes"), 400
            results, ncur = _ws_provider_generic(tmpl, q, page, cursor)
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            msg = ("blocked by the source (403) — it's refusing automated "
                   "requests from this server. Try another source (e.g. Bing) "
                   "or a custom site.")
        _log(f"  [websearch] '{tmpl[:40]}' failed: {e}")
        return jsonify(ok=False, error=f"search failed: {msg}"), 502

    # next_page = page-based "more" (numeric token); next_cursor = opaque cursor.
    next_page = None
    if ncur and str(ncur).isdigit():
        next_page = int(ncur); ncur = None
    _log(f"  [websearch] q='{q}' src='{tmpl[:30]}' p{page} -> {len(results)} "
         f"image(s){' +more' if (ncur or next_page) else ''}")
    return jsonify(ok=True, results=results, query=q,
                   next_cursor=ncur, next_page=next_page)

@app.route("/api/websearch/img", methods=["GET"])
def _web_image_proxy():
    """Stream a remote image through the server with a proper Referer, so booru
    and other hotlink-protected CDNs don't block it the way a direct browser
    <img src> (no-referrer) does. This is why thumbnails were blank — the CDNs
    refused the browser's request but allow a server-side fetch."""
    url = (request.args.get("url") or "").strip()
    if not url.startswith("http"):
        return Response("bad url", status=400)
    try:
        r = _ws_fetch_image(url, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        return Response(r.content, mimetype=ctype,
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        _log(f"  [websearch] img proxy failed ({url[:60]}): {e}")
        return Response("", status=502)

def _ws_image_referer(url):
    """The referer a CDN expects: the parent site, not the image's own host."""
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if "pornpics.com" in host:
        return "https://www.pornpics.com/"
    if "imagefap.com" in host or "fap.to" in host:
        return "https://www.imagefap.com/"
    return "{0.scheme}://{0.netloc}/".format(urlparse(url))

def _ws_proxies():
    """Return a requests-style proxies dict if WEB_IMAGE_PROXY is set, else None.
    Routes image fetches through a residential/non-flagged IP to get past CDN
    IP/ASN blocks (PornPics, ImageFap). Applies to http and https."""
    p = (WEB_IMAGE_PROXY or "").strip()
    if not p:
        return None
    return {"http": p, "https": p}

def _ws_fetch_image(url, stream=False):
    """Fetch a remote image with a full browser-like header set + the right
    Referer, routed through the proxy if one is configured. Bare requests calls
    get 403'd by some CDNs by IP/ASN — a proxy with a residential IP is the only
    thing that gets past that (proven: even a real browser 403s from Colab)."""
    hdrs = {
        "User-Agent": _WS_UA,
        "Referer": _ws_image_referer(url),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Connection": "keep-alive",
    }
    return _requests.get(url, timeout=30, stream=stream, headers=hdrs,
                         proxies=_ws_proxies())

@app.route("/api/websearch/fetch", methods=["GET"])
def _web_image_fetch():
    """Fetch a chosen web image server-side and return it as a base64 data URL,
    so the frontend can set it as the img2img source without tripping browser
    CORS / canvas-taint restrictions."""
    url = (request.args.get("url") or "").strip()
    if not url.startswith("http"):
        return jsonify(ok=False, error="bad url"), 400
    try:
        r = _ws_fetch_image(url)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if "image" not in ctype:
            return jsonify(ok=False, error="not an image"), 415
        b64 = base64.b64encode(r.content).decode("ascii")
        return jsonify(ok=True, data_url=f"data:{ctype};base64,{b64}")
    except Exception as e:
        _log(f"  [websearch] image fetch failed ({url[:60]}): {e}")
        return jsonify(ok=False, error=f"fetch failed: {e}"), 502

@app.route("/api/loras/search", methods=["POST"])
def _search_loras():
    d = request.get_json(force=True)
    mtype = d.get("type") or "LORA"
    if mtype not in ("LORA", "Checkpoint"):
        mtype = "LORA"
    # 'kind' refines a LoRA search to plain LoRA vs LyCORIS. Civitai exposes
    # these as the model types LORA and LoCon (LyCORIS files register as
    # LoCon regardless of LoHa/LoKr subtype).
    kind = d.get("kind") or "any"
    if mtype == "LORA" and kind in ("LORA", "LoCon"):
        mtype = kind
    try:
        res = _civitai_search(
            query=(d.get("query") or "").strip(),
            base_model=(d.get("base_model") or "Qwen"),
            nsfw=bool(d.get("nsfw", False)),
            cursor=(d.get("cursor") or None),
            page=int(d.get("page", 1)),
            sort=(d.get("sort") or "Most Downloaded"),
            period=(d.get("period") or "AllTime"),
            tag=(d.get("tag") or None),
            gen_only=bool(d.get("gen_only", False)),
            mtype=mtype)
    except Exception as e:
        return jsonify(ok=False, error=f"Civitai search failed — {e}"), 502
    return jsonify(ok=True, **res)

# Curated, deduplicated Civitai tags for the quick-filter chips. Kept small and
# organized rather than dumping Civitai's full (redundant) taxonomy. 'primary'
# shows by default; 'extra' appears behind a "more" toggle. Free-text entry
# covers anything not listed.
# Fallback quick-pick tags if Civitai's tag API is unreachable. Normally we
# serve Civitai's real tag taxonomy (popularity-ranked) so you can pick any
# tag the site exposes; this list is only used if that fetch fails.
_TAGS_FALLBACK = [
    "style", "character", "concept", "clothing", "background",
    "anime", "realistic", "cartoon", "3d", "fantasy", "sci-fi",
    "cyberpunk", "vintage", "poster", "landscape", "celebrity",
    "pixel art", "logo", "tattoo", "armor", "dress", "hair", "pose",
    "lighting", "detail", "illustration", "manga", "comic", "vtuber",
]
_TAGS_CACHE = {"all": None, "ts": 0}      # cached full tag list (name list)

def _fetch_civitai_tags():
    """Fetch Civitai's real model-tag list, popularity-ranked. limit=0 returns
    all tags. Cached for an hour so we don't refetch on every UI open. Returns
    a list of {name, count}. Falls back to the curated list on failure."""
    now = time.time()
    if _TAGS_CACHE["all"] and (now - _TAGS_CACHE["ts"] < 3600):
        return _TAGS_CACHE["all"]
    headers = {}
    key = CIVITAI_API_KEY or _get_secret("CIVITAI_API_KEY", verbose=False)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        # limit=200 (the max per page) ordered by popularity covers the tags
        # anyone actually browses; limit=0 ("all") can be thousands and slow.
        r = _requests.get("https://civitai.com/api/v1/tags",
                          params={"limit": 200}, headers=headers, timeout=25)
        r.raise_for_status()
        items = r.json().get("items", [])
        tags = [{"name": t.get("name"), "count": t.get("modelCount") or 0}
                for t in items if t.get("name")]
        # Civitai returns these roughly by popularity already; keep that order.
        if tags:
            _TAGS_CACHE["all"] = tags
            _TAGS_CACHE["ts"] = now
            _log(f"  [tags] loaded {len(tags)} Civitai tags")
            return tags
    except Exception as e:
        _log(f"  [tags] Civitai tag fetch failed ({e}); using fallback list")
    return [{"name": t, "count": 0} for t in _TAGS_FALLBACK]

@app.route("/api/loras/tags", methods=["GET"])
def _list_tags():
    """Serve Civitai's real model tags (popularity-ranked) for the search UI.
    Supports ?q= to filter the list by name (so you can find any tag Civitai
    exposes). Returns the full set plus a 'primary' slice for default display."""
    q = (request.args.get("q") or "").strip().lower()
    tags = _fetch_civitai_tags()
    if q:
        tags = [t for t in tags if q in t["name"].lower()]
    names = [t["name"] for t in tags]
    return jsonify(ok=True,
                   tags=tags,                 # [{name,count}, ...] full/filtered
                   primary=names[:16],        # default chips
                   extra=names[16:48])        # behind "more"

def _civitai_headers():
    h = {}
    key = CIVITAI_API_KEY or _get_secret("CIVITAI_API_KEY", verbose=False)
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h

def _strip_html(s):
    if not s:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p>", "\n\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)            # drop remaining tags
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"&lt;", "<", s).replace("&gt;", ">")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _fetch_samples(model_id, version_id, nsfw, cursor=None, limit=24):
    """Fetch a page of community generations for a model from /images, with
    cursor pagination for infinite scroll. Returns (items, next_cursor).

    Civitai's /images endpoint is flaky (timeouts, 500s, rate-limits) and a
    version-filtered query can come back empty for versions with few posts, so
    we retry transient failures and, on the first page, fall back to a
    model-wide query (no version filter) if the version-filtered one is empty."""
    hdr = _civitai_headers()

    def _do(params):
        last = None
        for attempt in range(3):
            try:
                r = _requests.get("https://civitai.com/api/v1/images",
                                  params=params, headers=hdr, timeout=30)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"Civitai {r.status_code}")
                    time.sleep(1.0 + attempt)      # brief backoff, then retry
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                time.sleep(0.8 + attempt)
        raise last or RuntimeError("Civitai images request failed")

    ip = {"modelId": model_id, "limit": limit, "sort": "Most Reactions",
          "nsfw": "true" if nsfw else "false", "withMeta": "true"}
    if version_id:
        ip["modelVersionId"] = version_id
    if cursor:
        ip["cursor"] = cursor
    body = _do(ip)
    # If a version-filtered first page is empty, retry without the version
    # filter so the gallery still shows the model's images.
    if (not cursor and version_id
            and not (body.get("items") or [])):
        ip.pop("modelVersionId", None)
        body = _do(ip)
    out = []
    # Civitai's nsfwLevel comes back inconsistently: sometimes a string
    # ("None"/"Soft"/"Mature"/"X") and sometimes a bitmask integer
    # (1=PG, 2=PG-13, 4=R, 8/16=X, 32=XXX). The old code only handled the
    # string form, so integer levels (every SFW image!) were wrongly dropped.
    # Treat these as safe-to-show when not in NSFW mode.
    SAFE_STR = {None, "None", "Soft"}
    SAFE_INT = {0, 1, 2}      # unprocessed / PG / PG-13
    for im in (body.get("items") or []):
        if not nsfw:
            lvl = im.get("nsfwLevel")
            is_safe = False
            if isinstance(lvl, str):
                is_safe = lvl in SAFE_STR
            elif isinstance(lvl, (int, float)):
                is_safe = int(lvl) in SAFE_INT
            else:
                is_safe = True   # unknown encoding: trust the nsfw bool below
            if not is_safe or im.get("nsfw") is True:
                continue
        meta = im.get("meta") or {}
        # civitaiResources lists the resources the image used, each with a
        # modelVersionId + type (checkpoint/lora) and sometimes a weight. We
        # keep the LoRAs so "use this prompt" can also load them.
        loras = []
        for res in (meta.get("civitaiResources") or []):
            if (res.get("type") or "").lower() == "lora" and res.get("modelVersionId"):
                loras.append({
                    "version_id": res.get("modelVersionId"),
                    "weight": res.get("weight"),
                    "download_url": f"https://civitai.com/api/download/models/{res.get('modelVersionId')}",
                })
        out.append({
            "thumb": _resize_civitai_url(im.get("url"), 320),
            "full": _resize_civitai_url(im.get("url"), 1024),
            "width": im.get("width"), "height": im.get("height"),
            "nsfw": bool(im.get("nsfw")),
            "prompt": (meta.get("prompt") or "")[:1500],
            "negative": (meta.get("negativePrompt") or "")[:1000],
            "steps": meta.get("steps"),
            "cfg": meta.get("cfgScale"),
            "sampler": meta.get("sampler"),
            "seed": meta.get("seed"),
            "loras": loras,
        })
    next_cursor = (body.get("metadata") or {}).get("nextCursor")
    return out, next_cursor

@app.route("/api/loras/samples", methods=["POST"])
def _model_samples():
    """Paginated community gallery for a model — drives the detail view's
    infinite scroll."""
    d = request.get_json(force=True)
    model_id = d.get("model_id")
    if not model_id:
        return jsonify(ok=False, error="model_id required"), 400
    try:
        items, nxt = _fetch_samples(
            model_id, d.get("version_id"), bool(d.get("nsfw", False)),
            cursor=(d.get("cursor") or None))
    except Exception as e:
        return jsonify(ok=False, error=f"gallery fetch failed — {e}"), 502
    return jsonify(ok=True, samples=items, next_cursor=nxt)

@app.route("/api/loras/detail", methods=["POST"])
def _model_detail():
    """Full detail for one model: metadata from /models/{id} plus a gallery of
    real community generations from /images?modelId=. The gallery images carry
    their generation meta (prompt, sampler, steps, cfg, seed, resources)."""
    d = request.get_json(force=True)
    model_id = d.get("model_id")
    version_id = d.get("version_id")
    nsfw = bool(d.get("nsfw", False))
    hdr = _civitai_headers()
    # Allow opening a card with only a version_id (e.g. clicking a loaded LoRA,
    # whose download URL carries the version id but not the model id). Resolve
    # the model id from the model-versions endpoint.
    if not model_id and version_id:
        try:
            rv = _requests.get(
                f"https://civitai.com/api/v1/model-versions/{version_id}",
                headers=hdr, timeout=20)
            rv.raise_for_status()
            model_id = (rv.json() or {}).get("modelId")
        except Exception as e:
            return jsonify(ok=False,
                           error=f"could not resolve version {version_id} — {e}"), 502
    if not model_id:
        return jsonify(ok=False, error="model_id or version_id required"), 400
    info = {}
    try:
        r = _requests.get(f"https://civitai.com/api/v1/models/{model_id}",
                          headers=hdr, timeout=30)
        r.raise_for_status()
        m = r.json()
        vers = m.get("modelVersions") or []
        v = None
        if version_id is not None:
            vid = str(version_id)
            v = next((x for x in vers if str(x.get("id")) == vid), None)
        v = v or (vers[0] if vers else {})
        # Trigger words live on the version; if the matched version has none,
        # fall back to any version that does, so triggers reliably populate.
        triggers = v.get("trainedWords") or []
        if not triggers:
            for x in vers:
                if x.get("trainedWords"):
                    triggers = x.get("trainedWords"); break
        stats = m.get("stats") or {}
        info = {
            "model_id": m.get("id"),
            "version_id": v.get("id"),
            "name": m.get("name"),
            "type": m.get("type"),
            "base_model": v.get("baseModel"),
            "nsfw": bool(m.get("nsfw")),
            "creator": (m.get("creator") or {}).get("username"),
            "tags": m.get("tags") or [],
            "description": _strip_html(m.get("description"))[:4000],
            "triggers": triggers,
            "version_name": v.get("name"),
            "downloads": stats.get("downloadCount"),
            "thumbs_up": stats.get("thumbsUpCount"),
            "published_at": v.get("publishedAt"),
            "download_url": f"https://civitai.com/api/download/models/{v.get('id')}",
            "civitai_url": f"https://civitai.com/models/{m.get('id')}",
        }
        # File size of the primary file, if present.
        for f in (v.get("files") or []):
            if f.get("primary"):
                kb = f.get("sizeKB") or 0
                info["file_mb"] = round(kb / 1024.0, 1)
                break
        # All versions, so the UI can offer a picker (DMD2 vs Realism vs aBEAST
        # etc. are very different models under one page). Each carries enough to
        # load it directly and show its size/base.
        vlist = []
        for x in vers:
            mb = None
            for f in (x.get("files") or []):
                if f.get("primary"):
                    mb = round((f.get("sizeKB") or 0) / 1024.0, 1); break
            vlist.append({
                "version_id": x.get("id"),
                "version_name": x.get("name"),
                "base_model": x.get("baseModel"),
                "file_mb": mb,
                "triggers": x.get("trainedWords") or [],
                "download_url": f"https://civitai.com/api/download/models/{x.get('id')}",
            })
        info["versions"] = vlist
    except Exception as e:
        return jsonify(ok=False, error=f"model lookup failed — {e}"), 502

    # Community gallery — first page. The detail view pages further via
    # /api/loras/samples using the returned cursor.
    samples, samples_cursor = [], None
    try:
        samples, samples_cursor = _fetch_samples(
            model_id, version_id, nsfw, cursor=None)
    except Exception as e:
        _log(f"  [detail] gallery fetch failed: {e}")
    info["samples"] = samples
    info["samples_cursor"] = samples_cursor
    return jsonify(ok=True, info=info)

# Tiny on-disk thumbnail cache so re-viewed images are instant and we never
# re-fetch the same preview from Civitai twice in a session.
_THUMB_DIR = "/content/thumb_cache"

@app.route("/api/loras/thumb")
def _lora_thumb():
    """Proxy + cache a Civitai preview image so it renders inside the Colab
    iframe (no CORS / mixed-content) and loads fast on repeat views. If a
    resized variant 404s on the CDN, fall back to the original URL with the
    width transform stripped."""
    import hashlib
    u = request.args.get("u", "")
    if not u.startswith("https://image.civitai.com/"):
        return Response("bad url", status=400)
    Path(_THUMB_DIR).mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(u.encode()).hexdigest()
    cached = Path(_THUMB_DIR) / h
    if cached.exists():
        return Response(cached.read_bytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    # Try the requested URL, then a couple of fallbacks. Civitai's CDN serves
    # these without auth but is picky about the User-Agent.
    hdr = {"User-Agent": "Mozilla/5.0 (DiT Studio thumbnail proxy)"}
    candidates = [u]
    # Fallback 1: strip the width transform segment entirely (original image).
    stripped = re.sub(r"/(?:[^/]*,)?width=\d+/", "/", u)
    if stripped != u:
        candidates.append(stripped)
    for cand in candidates:
        try:
            r = _requests.get(cand, headers=hdr, timeout=25)
            if r.status_code == 200 and r.content:
                cached.write_bytes(r.content)
                ct = r.headers.get("Content-Type", "image/jpeg")
                return Response(r.content, mimetype=ct,
                                headers={"Cache-Control": "public, max-age=86400"})
        except Exception:
            pass
    return Response("", status=404)

# ---- base-model swap ---------------------------------------------------
def _do_swap(url_or_repo, label, base_hint=None):
    STATE["swap"] = {"busy": True, "stage": "downloading / loading",
                     "error": None, "result": None}
    try:
        ok, res = swap_base_model(url_or_repo, label, base_hint=base_hint)
        if ok:
            STATE["swap"]["result"] = res
        else:
            STATE["swap"]["error"] = res
    except Exception as e:
        STATE["swap"]["error"] = str(e)
    finally:
        STATE["swap"]["busy"] = False
        STATE["swap"]["stage"] = "done"

@app.route("/api/config", methods=["GET"])
def _get_config():
    """The current replayable setup: base model (url+name) + the LoRA stack
    (name, url, scale each). The browser persists this and, on a fresh runtime,
    replays it via /api/model/swap + /api/loras/add so the last model+LoRA combo
    is restored automatically."""
    loras = []
    for nm, info in STATE["loras"].items():
        loras.append({"name": nm,
                      "url": info.get("url") or "",
                      "scale": float(info.get("scale", 1.0)),
                      "triggers": info.get("triggers") or ""})
    murl = STATE.get("model_url") or STATE.get("model_ref") or ""
    entry = MODEL_REGISTRY.get(STATE.get("arch") or "", {})
    return jsonify(ok=True,
                   model={"url": murl, "name": STATE.get("model_name") or "",
                          "base": entry.get("civitai_base") or ""},
                   loras=loras)

@app.route("/api/model", methods=["GET"])
def _model():
    ref = STATE.get("model_ref") or ""
    murl = STATE.get("model_url") or ""
    vid = None
    for cand in (murl, ref):
        m = re.search(r"/api/download/models/(\d+)", cand or "")
        if not m:
            m = re.search(r"civitai_(\d+)", cand or "")
        if m:
            vid = int(m.group(1)); break
    entry = MODEL_REGISTRY.get(STATE.get("arch") or "", {})
    return jsonify(name=STATE["model_name"], ref=STATE["model_ref"],
                   url=murl or ref, version_id=vid, swap=STATE["swap"],
                   arch=STATE.get("arch"),
                   civitai_base=entry.get("civitai_base"),
                   defaults=entry.get("defaults") or {})

@app.route("/api/model/swap", methods=["POST"])
def _model_swap():
    if STATE["swap"]["busy"]:
        return jsonify(ok=False, error="a model swap is already running"), 409
    if any(j.get("status") in ("queued", "running")
           for j in jobs.values()):
        return jsonify(ok=False,
                       error="finish or cancel active jobs first"), 409
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    label = (d.get("name") or "").strip() or None
    base = (d.get("base") or "").strip() or None
    if not url:
        return jsonify(ok=False,
                       error="give a model key (qwen/flux/zimage), URL, or "
                             "HF repo id"), 400
    threading.Thread(target=_do_swap, args=(url, label, base),
                     daemon=True).start()
    return jsonify(ok=True)

@app.route("/api/model/reset", methods=["POST"])
def _model_reset():
    if STATE["swap"]["busy"]:
        return jsonify(ok=False, error="a model swap is already running"), 409
    if any(j.get("status") in ("queued", "running")
           for j in jobs.values()):
        return jsonify(ok=False,
                       error="finish or cancel active jobs first"), 409
    threading.Thread(target=_do_swap,
                     args=(DEFAULT_MODEL_KEY, None), daemon=True).start()
    return jsonify(ok=True)

def set_vae(which, url=None):
    """VAE swapping was an SDXL-era workaround (fp16 black-image bug).
    Qwen-Image, FLUX and Z-Image each ship their own matched VAE and run it
    in bf16 — there is nothing to fix or swap."""
    return False, ("VAE swapping isn't applicable to Qwen/Flux/Z-Image — "
                   "each model uses its own matched VAE.")

@app.route("/api/model/vae", methods=["POST"])
def _model_vae():
    d = request.get_json(force=True)
    ok, msg = set_vae((d.get("which") or "own"), (d.get("url") or "").strip())
    return (jsonify(ok=True, message=msg) if ok
            else (jsonify(ok=False, error=msg), 400))

@app.route("/api/generate", methods=["POST"])
def _generate():
    gate = _require_login()
    if gate: return gate
    try:
        params = request.get_json(force=True, silent=True)
        if not isinstance(params, dict):
            return jsonify(error="invalid or empty request body"), 400
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"status": "queued", "progress": 0, "stage": "queued",
                        "result": None, "error": None, "cancel": False}
        threading.Thread(target=run_job, args=(job_id, params),
                         daemon=True).start()
        return jsonify(job_id=job_id)
    except Exception as e:
        _log(f"  /api/generate error: {e}")
        return jsonify(error=f"generate failed: {e}"), 500

@app.route("/api/status/<job_id>")
def _status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    slim = {k: v for k, v in job.items() if k != "result"}
    slim["has_result"] = bool(job.get("result"))
    return jsonify(slim)

@app.route("/api/result/<job_id>")
def _result(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="no result"), 404
    return jsonify(result=job["result"])

@app.route("/api/history")
def _history():
    """Return the persisted generation history (most recent first). Images are
    served via /api/history/image/<file> rather than inlined, so the list is
    light and the browser isn't asked to hold megabytes of base64."""
    return jsonify(history=_read_history())

@app.route("/api/history/image/<path:fn>")
def _history_image(fn):
    # Guard against path traversal — only serve files from the outputs dir.
    safe = os.path.basename(fn)
    full = os.path.join(HISTORY_DIR, safe)
    if not os.path.exists(full):
        return jsonify(error="not found"), 404
    return send_file(full, mimetype="image/png")

@app.route("/api/history/clear", methods=["POST"])
def _history_clear():
    with _history_lock:
        try: os.remove(HISTORY_MANIFEST)
        except Exception: pass
    return jsonify(ok=True)

@app.route("/api/cancel/<job_id>", methods=["POST"])
def _cancel(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(ok=False, error="unknown job"), 404
    if job.get("status") in ("done", "error", "cancelled"):
        return jsonify(ok=True, already=True)
    job["cancel"] = True
    job.update(stage="cancelling")
    return jsonify(ok=True)

@app.route("/api/console")
def _console():
    return jsonify(lines=list(console_lines)[-200:])

@app.route("/api/hw")
def _hw():
    info = {"gpu": None, "vram_used": 0, "vram_total": 0,
            "residency": STATE["residency"],
            "model_name": STATE["model_name"],
            "swapping": STATE["swap"]["busy"],
            "swap_stage": STATE["swap"].get("stage", "")}
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
<title>Z-Image Studio — Qwen / Flux / Z-Image</title>
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
.app{display:grid;grid-template-columns:var(--sidebar-w) 1fr;
  grid-template-rows:var(--header-h) minmax(0,1fr) auto;
  grid-template-areas:"header header" "sidebar main" "dock dock";
  height:100vh;width:100vw;overflow:hidden}
.app-header{grid-area:header;display:flex;align-items:center;padding:0 16px;
  background:var(--surface);border-bottom:1px solid var(--border);z-index:100;gap:11px}
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
.hdr-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--text-muted);transition:background .3s}
.hdr-badge .dot.on{background:var(--green);animation:pulse 2s infinite}
.hdr-badge .dot.warm{background:var(--gold);animation:pulse 1.2s infinite}
.hdr-badge .dot.cold{background:var(--text-muted)}
.hdr-badge .dot.off{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
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
.slider-row{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.lock-row{display:flex;align-items:center;gap:7px;margin:2px 0 10px;cursor:pointer;
  font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;user-select:none}
.lock-row input{accent-color:var(--gold);cursor:pointer}
.lock-row .lock-ico{font-size:12px;opacity:.5}
.lock-row.on{color:var(--gold)}
.lock-row.on .lock-ico{opacity:1}
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
.vae-row{display:flex;align-items:center;gap:8px;margin-top:10px}
.vae-label{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;flex:0 0 auto}
.lora-row .lora-scale{flex:0 0 64px}
.lora-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:9px 10px;margin-top:8px}
.lora-card-top{display:flex;align-items:center;gap:8px}
.lora-card-name{font-family:var(--font-mono);font-size:10px;color:var(--gold);font-weight:700;
  flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lora-name-link,.model-name-link{cursor:pointer;text-decoration:underline;
  text-decoration-style:dotted;text-underline-offset:2px}
.lora-name-link:hover,.model-name-link:hover{color:var(--gold)}
.lora-x{background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;
  line-height:1;flex:0 0 auto}
.lora-x:hover{color:var(--red)}
.inspect-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:9px 10px;margin-top:8px}
.inspect-row{display:flex;justify-content:space-between;gap:8px;font-family:var(--font-mono);
  font-size:10px;color:var(--text-muted);padding:2px 0}
.inspect-row b{color:var(--text);font-weight:700;text-align:right;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:62%}
.stage{grid-area:main;position:relative;background:#000;overflow:hidden}
.viewer{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  padding:18px;flex-wrap:wrap;gap:12px;overflow-y:auto}
.viewer img{max-width:100%;max-height:100%;border-radius:8px;background:#000;
  box-shadow:0 8px 40px rgba(0,0,0,.6);object-fit:contain}
.viewer.multi img{max-width:48%;max-height:48%}
.ph{text-align:center;color:var(--text-muted);font-family:var(--font-mono);font-size:12px;
  letter-spacing:.5px;line-height:2;padding:24px}
.ph b{color:var(--gold)}
.ph .big{font-size:32px;display:block;margin-bottom:8px;opacity:.5}
.progrow{position:absolute;left:0;right:0;bottom:0;z-index:15}
.prog{height:4px;background:rgba(255,255,255,.06);overflow:hidden}
.fill{height:100%;width:0%;background:var(--gold);transition:width .3s}
.stage-status{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);
  font-family:var(--font-mono);font-size:10px;color:var(--text-dim);letter-spacing:.5px;
  background:rgba(9,9,11,.7);backdrop-filter:blur(6px);padding:5px 12px;border-radius:14px;
  border:1px solid var(--border);white-space:nowrap;display:none}
.stage-status.show{display:block}
.stage-tools{position:absolute;top:14px;left:14px;z-index:16;display:none;gap:8px}
.stage-tools.show{display:flex}
.stage-clear{position:absolute;top:14px;right:14px;z-index:17;
  display:none;width:34px;height:34px;border-radius:9px;cursor:pointer;
  background:rgba(9,9,11,.72);backdrop-filter:blur(8px);border:1px solid var(--border);
  color:var(--text-muted);font-size:15px;line-height:1;align-items:center;justify-content:center}
.stage-clear.show{display:inline-flex}
.stage-clear:hover{border-color:var(--red,#e5484d);color:var(--red,#e5484d);
  background:rgba(229,72,77,.12)}
.chip{background:rgba(9,9,11,.72);backdrop-filter:blur(8px);border:1px solid var(--border);
  color:var(--text);font-family:var(--font-mono);font-size:10px;font-weight:600;
  letter-spacing:.5px;padding:7px 12px;border-radius:8px;cursor:pointer;transition:all .12s;
  text-decoration:none;display:inline-flex;align-items:center;gap:6px}
.chip:hover{border-color:var(--gold);color:var(--gold)}
.float{position:absolute;top:14px;right:14px;width:262px;z-index:20;
  background:rgba(17,17,20,.9);backdrop-filter:blur(14px);
  border:1px solid var(--border);border-radius:11px;
  box-shadow:0 14px 40px rgba(0,0,0,.55);display:flex;flex-direction:column;
  overflow:hidden;max-height:calc(100% - 28px)}
#historyPanel{top:auto;bottom:14px;max-height:calc(60% - 28px)}
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
.q-item .q-ic{width:40px;height:40px;border-radius:4px;flex:0 0 40px;
  background:var(--surface-3);display:flex;align-items:center;justify-content:center;
  font-size:16px;color:var(--text-muted);overflow:hidden}
.q-item .q-ic img{width:100%;height:100%;object-fit:cover}
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
.history-card-thumb{width:100%;aspect-ratio:1/1;background:var(--surface-3);overflow:hidden;position:relative}
.history-card-thumb img{width:100%;height:100%;object-fit:cover;display:block;cursor:pointer}
.hc-restore{position:absolute;top:5px;right:5px;width:24px;height:24px;border-radius:6px;
  background:rgba(9,9,11,.78);backdrop-filter:blur(6px);border:1px solid var(--border);
  color:var(--text);font-size:14px;line-height:1;cursor:pointer;opacity:0;transition:all .12s;
  display:flex;align-items:center;justify-content:center}
.history-card:hover .hc-restore{opacity:1}
.hc-restore:hover{border-color:var(--gold);color:var(--gold)}
.dock{grid-area:dock;background:#0a0a0c;border-top:1px solid var(--border);
  display:flex;flex-direction:column}
.dock-head{display:flex;align-items:center;gap:8px;padding:9px 14px;cursor:pointer;user-select:none}
.dock-title{font-family:var(--font-mono);font-size:9px;font-weight:700;letter-spacing:1px;
  color:var(--text-muted);text-transform:uppercase}
.dock-title .icon{color:var(--gold)}
.dock-sp{flex:1}
.q-clear{background:var(--surface-3);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:10px;font-weight:600;cursor:pointer;
  letter-spacing:.5px;padding:5px 11px;border-radius:5px;line-height:1;transition:all .12s}
.q-clear:hover{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}
.console{height:180px;overflow-y:auto;padding:2px 16px 12px;font-family:var(--font-mono);
  font-size:11px;line-height:1.55;color:var(--text-dim);user-select:text;cursor:text;
  scrollbar-width:thin;scrollbar-color:var(--surface-3) transparent}
.dock.collapsed .console{display:none}
.console .ln{white-space:pre-wrap;word-break:break-word}
.console .diag{color:var(--gold)}
.console .warn{color:var(--gold-light)}
.modal-scrim{position:fixed;inset:0;background:rgba(0,0,0,.78);
  z-index:8000;display:none;align-items:center;justify-content:center;padding:24px}
.modal-scrim.open{display:flex}
.ws-card{cursor:pointer;position:relative}
.ws-batchbar{display:flex;align-items:center;gap:10px;padding:8px 16px;
  border-bottom:1px solid var(--border);background:var(--surface-2)}
.ws-batch-hint{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);flex:1}
.ws-batch-count{font-family:var(--font-mono);font-size:10px;color:var(--gold);
  letter-spacing:.5px}
.ws-srcbar{display:flex;align-items:center;gap:8px;padding:7px 16px;
  border-bottom:1px solid var(--border);background:var(--surface)}
.ws-src-label{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;flex:0 0 auto}
.ws-src-hint{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
  margin-left:auto;flex:0 0 auto}
/* #wsGrid is the scroll container; the actual grid is .ws-grid-inner inside
   it. Force #wsGrid to plain block so the inner grid spans full width (it was
   inheriting .history-grid's single-column cell, causing one-column layout). */
#wsGrid{display:block !important}
.ws-grid-inner{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
  gap:12px;align-content:start;width:100%}
.ws-more{grid-column:1/-1;text-align:center;padding:16px;font-family:var(--font-mono);
  font-size:11px;color:var(--text-muted)}
.ws-gallery-bar{display:flex;align-items:center;gap:12px;padding:8px 4px 14px;
  flex-wrap:wrap}
.ws-gallery-title{font-family:var(--font-mono);font-size:11px;color:var(--gold)}
.ws-galbadge{position:absolute;bottom:5px;left:5px;background:rgba(235,0,139,.85);
  color:#fff;font-family:var(--font-mono);font-size:9px;padding:2px 6px;
  border-radius:4px;pointer-events:none}
.ws-check{position:absolute;top:5px;left:5px;width:22px;height:22px;border-radius:5px;
  background:rgba(9,9,11,.7);border:1px solid var(--border);color:transparent;
  display:none;align-items:center;justify-content:center;font-size:13px;font-weight:700}
.ws-card.ws-sel{outline:2px solid var(--gold);outline-offset:-2px}
.ws-card.ws-sel .ws-check{display:flex;color:var(--black);background:var(--gold);border-color:var(--gold)}
.ws-card.ws-queued{opacity:.55}
.ws-card.ws-queued::after{content:"queued";position:absolute;top:5px;right:5px;
  font-family:var(--font-mono);font-size:9px;color:var(--gold);
  background:rgba(9,9,11,.8);padding:2px 6px;border-radius:4px}
.ws-dims{position:absolute;bottom:5px;right:5px;background:rgba(9,9,11,.8);
  color:var(--text);font-family:var(--font-mono);font-size:9px;padding:2px 6px;
  border-radius:4px}
.ws-card.ws-loading{opacity:.5;pointer-events:none}
.ws-card.ws-loading::after{content:"loading\u2026";position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);
  font-size:10px;color:var(--gold);background:rgba(9,9,11,.5)}
.history-modal-panel{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;width:min(1200px,95vw);height:min(90vh,900px);display:flex;
  flex-direction:column;overflow:hidden}
.history-modal-head{display:flex;align-items:center;padding:8px 16px;
  border-bottom:1px solid var(--border);gap:12px}
.history-modal-title{font-family:var(--font-mono);font-size:12px;font-weight:700;
  color:var(--text);letter-spacing:1px;flex:1;display:flex;align-items:center;gap:8px}
.history-modal-title .icon{color:var(--gold)}
.modal-close{width:26px;height:26px;background:var(--surface-3);border:1px solid var(--border);
  color:var(--text);border-radius:5px;cursor:pointer;font-size:14px;line-height:1}
.modal-close:hover{border-color:var(--gold);color:var(--gold)}
.history-modal-body{flex:1;overflow-y:auto;padding:12px 14px;contain:layout paint;
  -webkit-overflow-scrolling:touch}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:14px;align-content:start}
.ov-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  overflow:hidden;cursor:pointer;transition:all .15s}
.ov-card:hover{border-color:var(--gold);transform:translateY(-2px)}
.ov-card img{width:100%;display:block;background:#000;aspect-ratio:1/1;object-fit:cover}
.ov-card .cap{padding:8px 10px;font-family:var(--font-mono);font-size:9px;
  color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lora-search-bar{display:flex;gap:7px;align-items:center;padding:8px 16px;
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.lb-input{flex:1;min-width:160px;padding:7px 10px;background:var(--surface-2);
  border:1px solid var(--border);border-radius:5px;color:var(--text);
  font-family:var(--font-mono);font-size:12px;outline:none}
.lb-input:focus{border-color:var(--gold)}
.lb-tagbar{display:flex;align-items:center;gap:8px;padding:6px 16px;
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.lb-tagbar.collapsed{display:none}
.lb-tagbar-label{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;flex:0 0 auto}
.lb-taginput{flex:0 0 auto;width:150px;padding:4px 9px;background:var(--surface-2);
  border:1px solid var(--border);border-radius:13px;color:var(--text);
  font-family:var(--font-mono);font-size:10px;outline:none}
.lb-taginput:focus{border-color:var(--gold)}
.lb-tagchips{display:flex;flex-wrap:wrap;gap:6px;flex:1 1 auto}
.lb-tagchip{font-family:var(--font-mono);font-size:10px;padding:4px 10px;border-radius:13px;
  background:var(--surface-2);color:var(--text-dim);border:1px solid var(--border);
  cursor:pointer;transition:all .12s}
.lb-tagchip:hover{border-color:var(--gold);color:var(--text)}
.lb-tagchip.active{background:var(--gold-dim);color:var(--gold);border-color:var(--gold)}
.lb-tagmore{font-family:var(--font-mono);font-size:10px;padding:4px 10px;border-radius:13px;
  background:transparent;color:var(--gold);border:1px solid var(--gold-dim);cursor:pointer;flex:0 0 auto}
.lb-tagmore:hover{border-color:var(--gold)}
.lb-select{padding:8px 10px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:5px;color:var(--text);font-family:var(--font-mono);font-size:11px;
  cursor:pointer;outline:none}
.lb-select:focus{border-color:var(--gold)}
.lb-nsfw{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
  display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
.lb-tagstoggle{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  background:var(--surface-2);border:1px solid var(--border);border-radius:5px;
  padding:5px 10px;cursor:pointer;transition:all .12s;letter-spacing:.5px}
.lb-tagstoggle:hover{border-color:var(--gold);color:var(--gold)}
.lb-tagstoggle.on{border-color:var(--gold);color:var(--gold);
  background:rgba(212,160,23,.12)}
.lb-card{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  overflow:hidden;cursor:pointer;transition:border-color .15s,transform .15s;position:relative;
  display:flex;flex-direction:column;
  content-visibility:auto;contain-intrinsic-size:auto 300px}
.lb-card:hover{border-color:var(--gold);transform:translateY(-2px)}
.lb-card .lb-thumb{width:100%;aspect-ratio:1/1;background:var(--surface-3);object-fit:cover;
  display:block}
.lb-card .lb-noimg{width:100%;aspect-ratio:1/1;background:var(--surface-3);display:flex;
  align-items:center;justify-content:center;color:var(--text-muted);font-size:24px}
.lb-card .lb-body{padding:8px 10px;display:flex;flex-direction:column;gap:5px}
.lb-card .lb-name{font-family:var(--font-mono);font-size:10px;color:var(--text);font-weight:600;
  line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
  overflow:hidden}
.lb-tags{display:flex;flex-wrap:wrap;gap:4px}
.lb-tag{font-family:var(--font-mono);font-size:7px;letter-spacing:.5px;text-transform:uppercase;
  font-weight:700;padding:2px 6px;border-radius:3px;background:var(--surface-3);color:var(--text-muted)}
.lb-tag.base{background:var(--gold-dim);color:var(--gold)}
.lb-tag.type{background:rgba(59,130,246,.12);color:var(--blue)}
.lb-tag.warn{background:rgba(239,68,68,.12);color:var(--red)}
.lb-tag.nsfw{background:rgba(239,68,68,.18);color:var(--red)}
.lb-tag.loaded{background:rgba(34,197,94,.16);color:var(--green,#22c55e)}
.lb-card.lb-loaded{opacity:.5;filter:grayscale(.55)}
.lb-card.lb-loaded:hover{border-color:var(--border);transform:none}
.lb-card.lb-loaded .lb-loadbtn{background:var(--surface-3);color:var(--text-muted)}
.lb-card .lb-loadbtn{margin:0 10px 10px;padding:7px;background:var(--gold);color:var(--black);
  font-family:var(--font-mono);font-weight:700;font-size:9px;letter-spacing:.5px;
  text-transform:uppercase;border:none;border-radius:5px;cursor:pointer;transition:all .12s}
.lb-card .lb-loadbtn:hover{background:var(--gold-light)}
.lb-card .lb-loadbtn:disabled{opacity:.5;cursor:not-allowed}
.lb-pager{display:flex;align-items:center;justify-content:center;gap:14px;padding:16px;
  font-family:var(--font-mono);font-size:11px;color:var(--text-muted)}
.lb-pager button{background:var(--surface-3);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:11px;cursor:pointer;padding:6px 14px;border-radius:5px}
.lb-pager button:hover:not(:disabled){border-color:var(--gold);color:var(--gold)}
.lb-pager button:disabled{opacity:.4;cursor:not-allowed}
.lbd-head{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}
.lbd-back{background:var(--surface-3);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:11px;cursor:pointer;padding:7px 14px;border-radius:5px}
.lbd-back:hover{border-color:var(--gold);color:var(--gold)}
.lbd-title{font-family:var(--font-mono);font-size:15px;font-weight:700;color:var(--text);flex:1;min-width:160px}
.lbd-load{background:var(--gold);color:var(--black);font-family:var(--font-mono);font-weight:700;
  font-size:11px;letter-spacing:.5px;text-transform:uppercase;border:none;border-radius:6px;
  cursor:pointer;padding:9px 18px}
.lbd-load:hover{background:var(--gold-light)}
.lbd-load:disabled{opacity:.5;cursor:not-allowed}
.lbd-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.lbd-stat{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
  background:var(--surface-2);border:1px solid var(--border);border-radius:5px;padding:5px 9px}
.lbd-stat b{color:var(--text)}
.lbd-desc{font-family:var(--font-sans,inherit);font-size:12px;color:var(--text-dim);
  line-height:1.6;white-space:pre-wrap;background:var(--surface-2);border:1px solid var(--border);
  border-radius:6px;padding:12px;margin-bottom:14px;max-height:180px;overflow-y:auto}
.lbd-section{font-family:var(--font-mono);font-size:10px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted);margin:6px 0 10px}
.lbd-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.lbd-sample{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  overflow:hidden;content-visibility:auto;contain-intrinsic-size:auto 260px}
.lbd-sample img{width:100%;display:block;background:var(--surface-3);object-fit:cover}
.lbd-sample .lbd-samp-body{padding:8px 9px}
.lbd-prompt{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);line-height:1.45;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;margin-bottom:7px}
.lbd-use{width:100%;background:var(--surface-3);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-mono);font-size:8px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;
  cursor:pointer;padding:6px;border-radius:4px}
.lbd-use:hover{border-color:var(--gold);color:var(--gold)}
.lbd-triggers{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.trigchip{font-family:var(--font-mono);font-size:11px;padding:5px 10px;border-radius:14px;
  background:var(--gold-dim);color:var(--gold);border:1px solid transparent;cursor:pointer;
  transition:border-color .15s}
.trigchip:hover{border-color:var(--gold)}
.trigchip.trig-added{background:rgba(34,197,94,.18);color:var(--green,#22c55e)}
.lora-trigs{display:flex;flex-wrap:wrap;gap:5px;align-items:center;margin-top:8px}
.lora-trigs-label{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;margin-right:2px}
.trigchip.sm{font-size:10px;padding:3px 8px}
.lbd-weightrow{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.lbd-verrow{display:flex;align-items:center;gap:10px;margin:10px 0 14px}
.lbd-verlabel{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px;flex:0 0 auto}
.lbd-wlabel{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.5px}
.lbd-weightrow input[type=range]{flex:1;max-width:260px;accent-color:var(--gold)}
.lbd-wval{font-family:var(--font-mono);font-size:12px;color:var(--gold);min-width:38px}
.lightbox{position:fixed;inset:0;z-index:9000;background:rgba(0,0,0,.92);
  display:none;align-items:center;justify-content:center;padding:40px}
.lightbox.open{display:flex}
.lbx-inner{display:flex;flex-direction:column;align-items:center;gap:14px;
  max-width:96vw;max-height:92vh}
.lbx-inner img{max-width:96vw;max-height:74vh;object-fit:contain;border-radius:8px;
  background:#000;box-shadow:0 10px 50px rgba(0,0,0,.6)}
.lbx-info{max-width:760px;width:100%;text-align:center}
.lbx-meta{font-family:var(--font-mono);font-size:11px;color:var(--gold);letter-spacing:.5px;margin-bottom:8px}
.lbx-prompt{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);line-height:1.5;
  max-height:22vh;overflow-y:auto;white-space:pre-wrap;background:var(--surface-2);
  border:1px solid var(--border);border-radius:6px;padding:10px;text-align:left}
.lbx-close{position:fixed;top:18px;right:22px;width:38px;height:38px;border-radius:50%;
  background:rgba(0,0,0,.6);border:1px solid var(--border);color:var(--text);font-size:18px;
  cursor:pointer;z-index:9001}
.lbx-close:hover{border-color:var(--gold);color:var(--gold)}
.lbx-nav{position:fixed;top:50%;transform:translateY(-50%);width:46px;height:46px;border-radius:50%;
  background:rgba(0,0,0,.6);border:1px solid var(--border);color:var(--text);font-size:24px;
  cursor:pointer;z-index:9001;line-height:1}
.lbx-nav:hover{border-color:var(--gold);color:var(--gold)}
.lbx-prev{left:22px}
.lbx-next{right:22px}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--surface);
  border:1px solid var(--gold);padding:11px 16px;border-radius:6px;font-family:var(--font-mono);
  font-size:11px;color:var(--text);z-index:9500;display:none;max-width:80vw}
.toast.err{border-color:var(--red);color:var(--red)}
.model-input-row{display:flex;gap:6px}
.model-input-row #modelUrl{flex:2;min-width:0}
.model-input-row #modelName{flex:1;min-width:0}
.model-btn-row{display:flex;gap:6px;margin-top:6px}
.model-btn-row .gen-btn-secondary.model-active{background:var(--gold);color:var(--black);
  border-color:var(--gold);font-weight:700;
  box-shadow:0 0 0 1px var(--gold),0 0 16px rgba(232,169,23,.35)}
.model-btn-row .gen-btn-secondary.model-active::before{content:"\25CF";font-size:8px;
  margin-right:6px;vertical-align:1.5px}
.model-btn-row .gen-btn-secondary.model-active.custom{background:var(--gold-dim);
  color:var(--gold);box-shadow:none}
.model-btn-row .gen-btn{flex:1;width:auto}
.notify-panel{background:var(--surface);border:1px solid var(--gold);border-radius:10px;
  width:min(440px,92vw);box-shadow:0 18px 50px rgba(0,0,0,.6);overflow:hidden;
  font-family:var(--font-mono)}
.notify-head{padding:14px 18px 4px;font-size:12px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--gold)}
.notify-body{padding:12px 18px 4px;font-size:13px;color:var(--text);line-height:1.55;
  max-height:50vh;overflow-y:auto;white-space:pre-wrap;word-break:break-word}
.notify-foot{padding:14px 18px 16px;display:flex;gap:10px;justify-content:flex-end}
.notify-ok{background:var(--gold);color:#1a1a1a;border:none;border-radius:6px;
  padding:8px 22px;font-family:var(--font-mono);font-size:12px;cursor:pointer;font-weight:600}
.notify-ok:hover{filter:brightness(1.08)}
.notify-cancel{background:var(--surface-2);color:var(--text);border:1px solid var(--border);
  border-radius:6px;padding:8px 18px;font-family:var(--font-mono);font-size:12px;cursor:pointer}
.notify-cancel:hover{border-color:var(--text-muted)}
/* ---- MissingLink sign-in gate ---- */
.login-overlay{position:fixed;inset:0;z-index:9999;background:rgba(5,5,7,.88);
  backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;
  padding:20px;overflow-y:auto}
.login-overlay.hidden{display:none}
.login-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  width:100%;max-width:660px;padding:26px 34px 22px;text-align:center;margin:auto;
  box-shadow:0 30px 80px rgba(0,0,0,.6)}
.login-brand{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:5px;
  text-transform:uppercase;color:var(--white)}
.login-brand span{color:var(--gold)}
.login-head{font-family:var(--font-body);font-size:19px;font-weight:700;color:var(--white);
  margin:9px 0 7px;letter-spacing:.2px}
.login-head span{color:var(--gold)}
.login-sub{font-family:var(--font-mono);font-size:10.5px;color:var(--text-dim);
  line-height:1.7;margin:0 auto 12px;max-width:560px}
.login-gift{display:inline-block;border:1.5px solid var(--gold);border-radius:9px;
  padding:7px 15px;font-family:var(--font-mono);font-size:10px;font-weight:700;
  letter-spacing:1.3px;color:var(--gold);margin-bottom:10px;text-decoration:none;
  transition:all .15s}
a.login-gift:hover{background:var(--gold-dim)}
.login-perks{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);
  margin-bottom:16px;line-height:1.7}
.login-perks b{color:var(--green);font-weight:600}
.login-auth{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:stretch;
  text-align:left}
.login-col{display:flex;flex-direction:column;justify-content:flex-start;min-width:0}
.login-or-v{display:flex;flex-direction:column;align-items:center;gap:8px;
  color:var(--text-muted);font-family:var(--font-mono);font-size:10px;letter-spacing:2px;
  justify-content:center}
.login-or-v::before,.login-or-v::after{content:"";flex:1;width:1px;background:var(--border)}
.login-google{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;
  padding:13px 10px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:9px;color:var(--text);font-family:var(--font-body);font-size:14px;
  font-weight:600;cursor:pointer;transition:all .15s}
.login-google:hover{border-color:var(--border-light);background:var(--surface-3)}
.login-cap{font-family:var(--font-mono);font-size:9.5px;color:var(--text-muted);
  letter-spacing:.4px;margin-top:8px;text-align:center}
.login-label{display:block;text-align:left;font-family:var(--font-mono);font-size:9.5px;
  font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text-muted);
  margin-bottom:6px}
.login-input{width:100%;background:var(--surface-2);border:1px solid var(--border);
  border-radius:8px;padding:11px 12px;color:var(--text);font-family:var(--font-mono);
  font-size:11.5px;outline:none;transition:border-color .15s;margin-bottom:9px}
.login-input:focus{border-color:var(--gold)}
.login-unlock{width:100%;padding:12px;background:var(--gold);color:var(--black);border:none;
  border-radius:8px;font-family:var(--font-mono);font-size:12px;font-weight:700;
  letter-spacing:1.6px;text-transform:uppercase;cursor:pointer;transition:filter .15s;
  text-decoration:none;display:block;text-align:center;box-sizing:border-box}
.login-unlock:hover{filter:brightness(1.08)}
.login-unlock:disabled{opacity:.55;cursor:wait}
.login-trial{margin-top:9px;background:var(--gold-light)}
.login-err{display:none;font-family:var(--font-mono);font-size:10.5px;color:var(--red);
  margin-top:13px;line-height:1.6;word-break:break-word}
.login-paste{display:none;margin-top:13px;text-align:left}
.login-paste .row{display:flex;gap:8px}
.login-paste .login-input{flex:1;margin-bottom:0}
.login-paste .login-unlock{width:auto;flex:0 0 auto;padding:0 22px}
.login-foot{font-family:var(--font-mono);font-size:10.5px;color:var(--text-dim);margin-top:16px}
.login-foot a{color:var(--gold);font-weight:700;text-decoration:none}
.login-foot a:hover{text-decoration:underline}
.login-powered{font-family:var(--font-mono);font-size:9.5px;color:var(--text-muted);margin-top:12px}
.login-powered b{color:var(--gold)}
@media (max-width:620px){
  .login-auth{grid-template-columns:1fr;gap:12px}
  .login-or-v{flex-direction:row}
  .login-or-v::before,.login-or-v::after{height:1px;width:auto;flex:1}
}
#acctBadge{cursor:pointer}
#acctBadge:hover{border-color:var(--border-light)}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:5px}
::-webkit-scrollbar-track{background:transparent}
</style></head><body>
<div class="login-overlay" id="loginOverlay">
  <div class="login-card">
    <div class="login-brand">Z-IMAGE <span>STUDIO</span></div>
    <div class="login-head">One Studio, Every Model. <span>Free for 7 Days</span></div>
    <div class="login-sub">Generate with <b>Qwen-Image 2512</b>, <b>FLUX.1</b> and <b>Z-Image Turbo</b> on your own Colab GPU — built-in <b>Civitai browser</b>, <b>runtime LoRA loading</b> with live strengths, txt2img + img2img.</div>
    <a class="login-gift" href="https://missinglink.build/create-checkout-session" target="_blank" rel="noopener" title="Opens Stripe checkout — 7-day free trial, cancel anytime">&#127873;&nbsp; 7-DAY FREE TRIAL &mdash; NO CHARGE UNTIL DAY 8</a>
    <div class="login-perks"><b>&#10003;</b> Civitai search &amp; one-click LoRAs &nbsp;&middot;&nbsp; <b>&#10003;</b> Qwen / Flux / Z-Image &nbsp;&middot;&nbsp; <b>&#10003;</b> Unlimited renders for members</div>
    <div class="login-auth">
      <div class="login-col">
        <button class="login-google" onclick="mlGoogle()"><svg width="19" height="19" viewBox="0 0 48 48" aria-hidden="true"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg> Continue with Google</button>
        <div class="login-cap">For members — free trial, subscribers &amp; Pro</div>
        <a class="login-unlock login-trial" id="loginTrialGo" target="_blank" rel="noopener" style="display:none">&#127873; Start 7-day free trial &rarr;</a>
      </div>
      <div class="login-or-v">OR</div>
      <div class="login-col">
        <label class="login-label">MissingLink API key</label>
        <input class="login-input" id="loginKey" type="password" placeholder="paste your MissingLink API key" onkeydown="if(event.key==='Enter')mlUseKey()">
        <button class="login-unlock" id="loginUnlock" onclick="mlUseKey()">Sign in</button>
      </div>
    </div>
    <div class="login-paste" id="loginPaste">
      <label class="login-label">Sign-in code</label>
      <div class="row">
        <input class="login-input" id="loginCode" placeholder="popup blocked? paste the code from the sign-in tab" onkeydown="if(event.key==='Enter')mlUseCode()">
        <button class="login-unlock" id="loginCodeBtn" onclick="mlUseCode()">Unlock</button>
      </div>
    </div>
    <div class="login-err" id="loginErr"></div>
    <div class="login-foot">Don't have an account? <a href="https://missinglink.build/create-checkout-session" target="_blank" rel="noopener">Start free trial</a> — $9/mo after, cancel anytime</div>
    <div class="login-powered">powered by <b>MissingLink</b> Triton kernels</div>
  </div>
</div>
<div class="app">

  <header class="app-header">
    <h1>Z-IMAGE <span>STUDIO</span></h1>
    <div class="stage-tabs">
      <button class="stage-tab" onclick="openGallery()" title="Session gallery">
        <span class="stage-tab-icon">&#127760;</span> Gallery</button>
      <button class="stage-tab active" title="Workspace">
        <span class="stage-tab-icon">&#127912;</span> Studio</button>
    </div>
    <div class="hdr-right">
      <div class="hdr-badge" id="acctBadge" style="display:none" title="Signed in with MissingLink — click to sign out" onclick="mlLogout()"><span>&#128100;</span><span id="acctPill"></span></div>
      <div class="hdr-badge" id="modelBadge" title="Active base model"><span>&#9638;</span><span id="modelPill">Qwen-Image 2512</span></div>
      <div class="hdr-badge" title="VRAM"><span>&#9635;</span><span id="vramPill">&ndash; / &ndash; GB</span></div>
      <div class="hdr-badge" id="connBadge" title="GPU state"><div class="dot cold" id="connDot"></div><span id="connLabel">Connecting</span></div>
    </div>
  </header>

  <aside class="sidebar">
    <div class="sidebar-scroll">
      <div class="mode-sec">
        <div class="mode-cap">Generation mode</div>
        <div class="mode-toggle">
          <button class="mode-btn active" id="modeTxt" onclick="setMode('txt2img')">Text&rarr;Image</button>
          <button class="mode-btn" id="modeImg" onclick="setMode('img2img')">Image&rarr;Image</button>
        </div>
        <div class="mode-hint" id="modeHint">Generate an image from a text prompt.</div>
      </div>

      <div class="sec" id="modelSec">
        <div class="sec-label"><span class="icon">&#9638;</span> Base model
          <span class="c" id="modelNow">Qwen-Image 2512</span></div>
        <div class="model-btn-row" style="margin-bottom:6px">
          <button class="gen-btn gen-btn-secondary" id="presetQwen"
            onclick="_startSwap('qwen','')">Qwen 2512</button>
          <button class="gen-btn gen-btn-secondary" id="presetFlux"
            onclick="_startSwap('flux','')">FLUX.1-dev</button>
          <button class="gen-btn gen-btn-secondary" id="presetZ"
            onclick="_startSwap('zimage','')">Z-Image</button>
        </div>
        <button class="gen-btn gen-btn-secondary" style="width:100%;margin-bottom:6px" onclick="openCkptBrowser()">&#128269; Browse Civitai Checkpoints</button>
        <div class="model-input-row">
          <input type="text" class="lora-url" id="modelUrl"
            placeholder="Civitai download URL or HF repo id">
          <input type="text" class="lora-url" id="modelName"
            placeholder="display name (include qwen/flux/z-image for URLs)">
        </div>
        <div class="model-btn-row">
          <button class="gen-btn gen-btn-secondary" id="swapBtn"
            onclick="swapModel()">&#8635; Load model</button>
          <button class="gen-btn gen-btn-secondary" id="resetBtn"
            onclick="resetModel()">Reset to Qwen</button>
        </div>
        <div class="hintline">One model lives in VRAM at a time (full A100 residency, bf16). FLUX.1-dev is a <b>gated</b> HF repo &mdash; accept its license and add HF_TOKEN in Colab Secrets. Civitai checkpoints load as transformer grafts onto the matching family.</div>
      </div>

      <div class="sec" id="initSec" style="display:none">
        <div class="sec-label"><span class="icon">&#9635;</span> Source image <span class="c">required</span></div>
        <div class="dropzone" id="drop" onclick="document.getElementById('file').click()">&#128247;&nbsp; Click or drop an image</div>
        <input type="file" id="file" accept="image/*" hidden>
        <button class="gen-btn gen-btn-secondary" style="width:100%;margin-top:8px;padding:9px"
          onclick="openWebSearch()">&#128269; Search the web for an image</button>
        <div class="slider-row" style="margin-top:12px"><span class="sl">Strength</span>
          <input type="range" id="strength" min="0.1" max="1" step="0.05" value="0.7"><input class="sv" id="strengthV" value="0.70"></div>
        <div class="hintline">Strength = how much to change the source. Low = keep it; high = reinvent it.</div>
      </div>

      <div class="sec">
        <div class="sec-label"><span class="icon">&#10022;</span> Prompt</div>
        <textarea class="ta" id="prompt">a serene mountain lake at golden hour, dramatic clouds, ultra detailed, photorealistic, cinematic lighting</textarea>
        <div class="sec-label" style="margin:12px 0 8px"><span class="icon">&#8856;</span> Negative</div>
        <textarea class="ta neg" id="neg">blurry, low quality, distorted, deformed, watermark, text, jpeg artifacts, oversaturated</textarea>
      </div>

      <div class="sec">
        <div class="sec-label"><span class="icon">&#9638;</span> Size</div>
        <select class="model-select" id="sizePreset" onchange="applySize()">
          <option value="1024x1024">1024 &times; 1024 &mdash; square</option>
          <option value="1152x896">1152 &times; 896 &mdash; landscape</option>
          <option value="896x1152">896 &times; 1152 &mdash; portrait</option>
          <option value="1328x1328">1328 &times; 1328 &mdash; square XL</option>
          <option value="1664x928">1664 &times; 928 &mdash; wide 16:9</option>
          <option value="928x1664">928 &times; 1664 &mdash; tall 9:16</option>
          <option value="custom">Custom</option>
        </select>
        <div class="slider-row" style="margin-top:12px"><span class="sl">Width</span>
          <input type="range" id="width" min="512" max="2048" step="32" value="1024"><input class="sv" id="widthV" value="1024"></div>
        <div class="slider-row"><span class="sl">Height</span>
          <input type="range" id="height" min="512" max="2048" step="32" value="1024"><input class="sv" id="heightV" value="1024"></div>
        <label class="lock-row" id="lockRow" title="Keep the width:height ratio fixed when you change one">
          <input type="checkbox" id="aspectLock"><span class="lock-ico">&#128279;</span>
          <span class="lock-txt">Constrain proportions</span></label>
        <div class="slider-row" style="margin-top:4px"><span class="sl">Images</span>
          <input type="range" id="batch" min="1" max="4" step="1" value="1"><input class="sv" id="batchV" value="1"></div>
      </div>

      <div class="sec">
        <div class="adv-toggle" id="advToggle" onclick="toggleAdv()"><span class="icon">&#9881;</span> Advanced settings <span id="advCaret">&#9656;</span></div>
        <div class="adv-body" id="advBody">
          <div class="sec-label" style="margin:2px 0 6px"><span class="icon">&#9881;</span> Sampler</div>
          <select id="sampler" class="lb-select" style="width:100%">
            <option value="">Default (recommended)</option>
            <option value="FlowMatch Euler">FlowMatch Euler</option>
            <option value="FlowMatch Heun">FlowMatch Heun</option>
          </select>
          <div class="hintline">Qwen / Flux / Z-Image are flow-matching models &mdash; the default scheduler is right for nearly everything. Civitai SDXL sampler recipes (DPM++, Karras&hellip;) don't apply.</div>
          <div class="slider-row"><span class="sl">Steps</span>
            <input type="range" id="steps" min="1" max="60" value="40"><input class="sv" id="stepsV" value="40"></div>
          <div class="slider-row"><span class="sl">Guidance</span>
            <input type="range" id="guid" min="0" max="15" step="0.1" value="4.0"><input class="sv" id="guidV" value="4.0"></div>
          <div class="hintline" id="guidHint">Qwen-Image uses TRUE CFG &mdash; ~4.0 is the sweet spot; the negative prompt works.</div>
          <div class="sec-label" style="margin:14px 0 6px"><span class="icon">&#9670;</span> Seed</div>
          <div class="seed-row">
            <input type="number" id="seed" value="42">
            <button class="gen-btn gen-btn-secondary" style="width:auto;padding:8px 12px" onclick="randSeed()">&#127922;</button>
          </div>
          <label class="lock-row" id="randSeedRow" title="Use a fresh random seed on every Generate for varied results">
            <input type="checkbox" id="randSeedEach" checked><span class="lock-ico">&#127922;</span>
            <span class="lock-txt">Random seed each generation</span></label>
        </div>
      </div>

      <div class="sec">
        <div class="sec-label"><span class="icon">&#9880;</span> LoRAs</div>
        <button class="gen-btn gen-btn-secondary" style="width:100%;margin-bottom:8px" onclick="openLoraBrowser()">&#128269; Browse Civitai LoRAs</button>
        <input type="text" class="lora-url" id="loraUrl" placeholder="paste HF or Civitai .safetensors URL">
        <div class="lora-row">
          <input type="text" class="lora-url" id="loraName" placeholder="name (optional)">
          <input type="number" class="lora-url lora-scale" id="loraScale" value="1.0" min="0" max="2" step="0.05">
        </div>
        <div class="lora-row" style="margin-top:8px">
          <button class="gen-btn gen-btn-secondary" id="inspectLoraBtn" style="flex:0 0 38%" onclick="inspectLora()">&#128269; Inspect</button>
          <button class="gen-btn gen-btn-secondary" id="addLoraBtn" style="flex:1" onclick="addLora()">+ Add LoRA</button>
        </div>
        <div id="loraInspect"></div>
        <div id="loraList"></div>
      </div>
    </div>

    <div class="sidebar-foot">
      <button class="gen-btn" id="genBtn" onclick="generate()">&#10022; Generate Image</button>
    </div>
  </aside>

  <main class="stage" id="stage">
    <div class="viewer" id="viewer">
      <div class="ph"><span class="big">&#127912;</span>Your images appear here, full size.<br>Write a <b>prompt</b> and hit <b>Generate</b>.</div>
    </div>
    <button class="stage-clear" id="stageClear" onclick="clearStage()" title="Clear the stage">&#128465;</button>
    <div class="stage-tools" id="stageTools"><a class="chip" id="dlBtn" download="missinglink_image.png">&#11015; Download</a><button class="chip" id="useAsInputBtn" onclick="useStageAsInput()">&#8631; Use as img2img input</button></div>
    <div class="stage-status" id="stage_status"></div>
    <div class="progrow"><div class="prog"><div class="fill" id="fill"></div></div></div>

    <div class="float" id="jobsPanel">
      <div class="float-head" id="jobsHandle">
        <span class="float-grip">&#8942;&#8942;</span>
        <span class="float-title"><span class="icon">&#9776;</span> Jobs</span>
        <div class="float-btns"><button class="float-min" id="jobsMin" onclick="toggleMin(event,'jobsPanel')" title="Minimize">&ndash;</button></div>
      </div>
      <div class="float-inner" id="jobsInner">
        <div class="float-sec q" id="secQueue">
          <div class="float-sec-hd" onclick="toggleSec('secQueue')"><span class="icon">&#9636;</span> Queue <span class="q-count" id="qCount">0</span><span class="caret">&#9662;</span></div>
          <div class="float-sec-body"><div id="queueBody"><div class="empty">No active jobs.</div></div></div>
        </div>
      </div>
    </div>

    <div class="float" id="historyPanel">
      <div class="float-head" id="histHandle">
        <span class="float-grip">&#8942;&#8942;</span>
        <span class="float-title"><span class="icon">&#9638;</span> History <span class="q-count" id="hCount">0</span></span>
        <div class="float-btns"><button class="float-min" id="histMin" onclick="toggleMin(event,'historyPanel')" title="Minimize">&ndash;</button></div>
      </div>
      <div class="float-inner" id="histInner">
        <div class="float-sec-body"><div id="historyBody"><div class="empty">Finished images appear here. Click to view.</div></div></div>
      </div>
    </div>
  </main>

  <footer class="dock collapsed" id="dock">
    <div class="dock-head" onclick="toggleDock()">
      <span class="dock-title"><span class="icon">&#9655;</span> Console / debug log</span>
      <span class="dock-sp"></span>
      <button class="q-clear" onclick="copyConsole(event)">copy</button>
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

<div class="modal-scrim" id="webSearchOverlay">
  <div class="history-modal-panel">
    <div class="history-modal-head">
      <div class="history-modal-title"><span class="icon">&#128269;</span> WEB IMAGE SEARCH</div>
      <div class="mode-toggle" style="width:auto;margin:0 10px">
        <button class="mode-btn active" id="wsTabSingle" onclick="wsSetMode('single')">Single</button>
        <button class="mode-btn" id="wsTabBatch" onclick="wsSetMode('batch')">Batch</button>
      </div>
      <div class="gallery-search-wrap" style="flex:1;margin:0 12px;display:flex;gap:6px;align-items:center">
        <input type="text" id="wsInput" class="lb-input" placeholder="Describe the image you want\u2026" autocomplete="off">
      </div>
      <button class="modal-close" onclick="closeWebSearch()">&#10005;</button>
    </div>
    <div class="ws-batchbar" id="wsBatchBar" style="display:none">
      <span class="ws-batch-hint">Select images, then run the prompt across each one in turn.</span>
      <span class="ws-batch-count" id="wsSelCount">0 selected</span>
      <button class="hdr-btn" onclick="wsSelectAll()">Select all</button>
      <button class="hdr-btn" onclick="wsClearSel()">Clear</button>
      <button class="gen-btn" id="wsRunBatch" style="width:auto;padding:7px 16px" onclick="wsRunBatch()">&#10022; Run batch</button>
    </div>
    <div class="ws-srcbar">
      <span class="ws-src-label">source</span>
      <select id="wsSource" class="lb-select" onchange="wsOnSourceChange()"></select>
      <select id="wsCat" class="lb-select" style="display:none" title="ImageFap category" onchange="wsOnCatChange()"></select>
      <input type="text" id="wsCustomUrl" class="lb-input" style="flex:1;display:none"
        placeholder="https://site/search?q={q}  \u2014 must contain {q}">
      <button class="hdr-btn" id="wsAddSrc" style="display:none" onclick="wsAddCustomSource()">Save site</button>
      <span class="ws-src-hint" id="wsSrcHint">DuckDuckGo image search</span>
    </div>
    <div class="history-modal-body" id="wsBody">
      <div class="history-grid" id="wsGrid"><div class="empty">Type a search above and hit Enter to find images on the web.</div></div>
    </div>
  </div>
</div>

<div class="modal-scrim" id="loraOverlay">
  <div class="history-modal-panel">
    <div class="history-modal-head">
      <div class="history-modal-title"><span class="icon">&#128269;</span>
        <span id="lbTitle">CIVITAI BROWSER</span></div>
      <div class="mode-toggle" style="width:auto;margin:0 12px">
        <button class="mode-btn active" id="lbTabLora" onclick="lbSetType('LORA')">LoRAs</button>
        <button class="mode-btn" id="lbTabCkpt" onclick="lbSetType('Checkpoint')">Checkpoints</button>
      </div>
      <button class="modal-close" onclick="closeLoraBrowser()">&#10005;</button>
    </div>
    <div class="lora-search-bar">
      <input type="text" id="lbQuery" class="lb-input" placeholder="search by name..." oninput="_lbQueryInput()">
      <select id="lbBase" class="lb-select" onchange="lbSearch()">
        <option value="Qwen">Qwen</option>
        <option value="Flux.1 D">Flux.1 D</option>
        <option value="ZImageTurbo">Z-Image Turbo</option>
        <option value="all">all base models</option>
      </select>
      <select id="lbKind" class="lb-select" onchange="lbSearch()">
        <option value="any">any type</option>
        <option value="LORA">LoRA only</option>
        <option value="LoCon">LyCORIS (LoCon/LoHa/LoKr)</option>
      </select>
      <select id="lbSort" class="lb-select" onchange="lbSearch()">
        <option value="Most Downloaded">Most downloaded</option>
        <option value="Highest Rated">Highest rated</option>
        <option value="Newest">Newest</option>
        <option value="Most Liked">Most liked</option>
      </select>
      <select id="lbPeriod" class="lb-select" onchange="lbSearch()">
        <option value="AllTime">All time</option>
        <option value="Year">This year</option>
        <option value="Month">This month</option>
        <option value="Week">This week</option>
        <option value="Day">Today</option>
      </select>
      <label class="lb-nsfw"><input type="checkbox" id="lbGen" onchange="lbSearch()"> gen-ready</label>
      <label class="lb-nsfw"><input type="checkbox" id="lbNsfw" onchange="lbSearch()"> NSFW</label>
      <button id="lbTagsToggle" class="lb-tagstoggle" onclick="_toggleTagBar()" title="Show/hide tag filters">&#9750; tags</button>
    </div>
    <div class="lb-tagbar collapsed" id="lbTagBar">
      <span class="lb-tagbar-label">tags</span>
      <input type="text" id="lbTagInput" class="lb-taginput" placeholder="type any tag + Enter" />
      <span id="lbTagChips" class="lb-tagchips"></span>
      <button id="lbTagMore" class="lb-tagmore" onclick="_toggleMoreTags()">more +</button>
    </div>
    <div class="history-modal-body" id="lbScroll">
      <div class="history-grid" id="lbGrid"><div class="empty">Search to browse models for your base.</div></div>
      <div id="lbSentinel" style="height:1px"></div>
      <div id="lbStatus" class="lb-pager"></div>
    </div>
    <div class="history-modal-body" id="lbDetail" style="display:none"></div>
  </div>
</div>

<div class="lightbox" id="lightbox" onclick="if(event.target===this)closeLightbox()">
  <button class="lbx-close" onclick="closeLightbox()">&#10005;</button>
  <button class="lbx-nav lbx-prev" onclick="_lbxNav(-1)">&#8249;</button>
  <button class="lbx-nav lbx-next" onclick="_lbxNav(1)">&#8250;</button>
  <div class="lbx-inner">
    <img id="lbxImg" src="">
    <div class="lbx-info">
      <div class="lbx-meta" id="lbxMeta"></div>
      <div class="lbx-prompt" id="lbxPrompt"></div>
      <button class="lbd-use" id="lbxUse" style="margin-top:8px;max-width:280px">use this prompt &amp; settings</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>
<div class="modal-scrim" id="notifyScrim">
  <div class="notify-panel" id="notifyPanel">
    <div class="notify-head"><span id="notifyTitle">Confirm</span></div>
    <div class="notify-body" id="notifyBody"></div>
    <div class="notify-foot">
      <button class="notify-cancel" id="notifyCancel" onclick="_notifyResolve(false)">Cancel</button>
      <button class="notify-ok" id="notifyOk" onclick="_notifyResolve(true)">OK</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let imgData=null;
let currentMode="txt2img";
let queue=[];    // {id,prompt,thumb,status,progress,stage}
let history=[];  // {id,prompt,thumb,urls,ts}

function toast(m,e){const t=$("toast");t.textContent=m;
  t.className="toast"+(e?" err":"");t.style.display="block";
  clearTimeout(t._t);t._t=setTimeout(()=>{t.style.display="none";},4500);}
// In-app confirm modal (replaces the browser's native confirm() dialog, which
// shows the ugly "<host> says" header). Returns a Promise<boolean>.
let _notifyCb=null;
function confirmModal(message,opts){
  opts=opts||{};
  $("notifyTitle").textContent=opts.title||"Confirm";
  $("notifyBody").textContent=message;
  $("notifyOk").textContent=opts.okText||"OK";
  $("notifyCancel").textContent=opts.cancelText||"Cancel";
  $("notifyCancel").style.display=opts.alert?"none":"";
  $("notifyScrim").classList.add("open");
  return new Promise(res=>{_notifyCb=res;});
}
function _notifyResolve(v){
  $("notifyScrim").classList.remove("open");
  const cb=_notifyCb;_notifyCb=null;
  if(cb)cb(v);
}
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

[["steps","stepsV"],["guid","guidV"],["width","widthV"],["height","heightV"],
 ["batch","batchV"],["strength","strengthV"]].forEach(([s,v])=>{
  $(s).addEventListener("input",()=>{$(v).value=$(s).value;});
  $(v).addEventListener("change",()=>{$(s).value=$(v).value;});
});
$("width").addEventListener("input",()=>$("sizePreset").value="custom");
$("height").addEventListener("input",()=>$("sizePreset").value="custom");
function setSL(id,val){$(id).value=val;$(id+"V").value=val;}
function randSeed(){$("seed").value=Math.floor(Math.random()*2e9);}
function applySize(){
  const v=$("sizePreset").value;if(v==="custom")return;
  const [w,h]=v.split("x");setSL("width",w);setSL("height",h);
}

/* ── constrain proportions (aspect lock) ── */
let _aspect=null;       // locked width/height ratio
let _aspectBusy=false;  // guard against the adjust-triggers-adjust loop
function _onAspectLockToggle(){
  if($("aspectLock").checked){
    // Prefer the uploaded image's ratio; else lock to the current box ratio.
    const w=+$("widthV").value||1024, h=+$("heightV").value||1024;
    _aspect=_imgAspect||(w/h);
    $("lockRow").classList.add("on");
  }else{
    _aspect=null;
    $("lockRow").classList.remove("on");
  }
}
function _aspectAdjust(changed){
  if(!_aspect||_aspectBusy)return;
  _aspectBusy=true;
  if(changed==="width"){
    const w=+$("widthV").value||1024;
    setSL("height",_snapDim(w/_aspect));
  }else{
    const h=+$("heightV").value||1024;
    setSL("width",_snapDim(h*_aspect));
  }
  $("sizePreset").value="custom";
  _aspectBusy=false;
}
$("aspectLock").addEventListener("change",_onAspectLockToggle);
$("randSeedEach").addEventListener("change",_saveState);
["input","change"].forEach(ev=>{
  $("width").addEventListener(ev,()=>_aspectAdjust("width"));
  $("widthV").addEventListener(ev,()=>_aspectAdjust("width"));
  $("height").addEventListener(ev,()=>_aspectAdjust("height"));
  $("heightV").addEventListener(ev,()=>_aspectAdjust("height"));
});

function toggleAdv(){
  $("advBody").classList.toggle("open");
  $("advCaret").innerHTML=$("advBody").classList.contains("open")?"\u25BE":"\u25B8";
}

function setMode(m){
  currentMode=m;
  $("modeTxt").classList.toggle("active",m==="txt2img");
  $("modeImg").classList.toggle("active",m==="img2img");
  $("initSec").style.display=m==="img2img"?"block":"none";
  $("modeHint").textContent=m==="img2img"
    ? "Transform an uploaded image guided by your prompt."
    : "Generate an image from a text prompt.";
  $("genBtn").innerHTML="\u2726 Generate Image";
}

let _imgAspect=null;   // width/height of the last uploaded source image
function _snapDim(v){
  // These DiT models want multiples of 16; sliders span 512..2048.
  v=Math.round(v/16)*16;
  return Math.max(512,Math.min(2048,v));
}
function _applyUploadedSize(w,h){
  _imgAspect=(w>0&&h>0)?(w/h):null;
  // Match the output canvas to the source image's dimensions.
  const W=_snapDim(w),H=_snapDim(h);
  _aspectBusy=true;                 // set both without triggering the adjuster
  setSL("width",W);setSL("height",H);
  _aspectBusy=false;
  $("sizePreset").value="custom";
  // If the lock is engaged, adopt the new image's ratio.
  if($("aspectLock") && $("aspectLock").checked && _imgAspect){
    _aspect=_imgAspect;
  }
}
$("file").addEventListener("change",e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();
  r.onload=()=>{imgData=r.result;
    $("drop").className="dropzone has";
    $("drop").innerHTML="<img src='"+r.result+"'>";
    // Read the natural dimensions, then size the output to match.
    const probe=new Image();
    probe.onload=()=>_applyUploadedSize(probe.naturalWidth,probe.naturalHeight);
    probe.src=r.result;
  };
  r.readAsDataURL(f);
});

/* ── Web image search (img2img source) ──
   Opens a modal, searches the configured site via our backend (which fetches
   + parses the page server-side, avoiding CORS), and lets you click a result
   to set it as the single img2img source. The chosen image is fetched through
   the backend too, so cross-origin hosts don't block it. */
let _wsResults=[];
let _wsMode="single";          // 'single' = click sets source; 'batch' = multi-select
let _wsSelected=new Set();      // indices of selected results (batch mode)
// Search sources. Built-in providers come from the backend registry; users
// can also save custom URL templates (persisted). Each: {label, value}.
let _WS_BUILTIN=[{label:"All engines (best)",value:"best"}];
let _wsSources=_WS_BUILTIN.slice();
let _wsSource="best";
let _wsProvidersLoaded=false;
const _WS_SRC_KEY="sdxlstudio_websearch_sources_v1";
async function _wsLoadProviders(){
  if(_wsProvidersLoaded)return;
  try{
    const j=await(await fetch("/api/websearch/providers")).json();
    if(j&&j.ok&&Array.isArray(j.providers)&&j.providers.length)
      _WS_BUILTIN=j.providers.map(p=>({label:p.label,value:p.value}));
  }catch(e){}
  _wsProvidersLoaded=true;
}
function _wsLoadSources(){
  try{
    const saved=JSON.parse(localStorage.getItem(_WS_SRC_KEY)||"[]");
    if(Array.isArray(saved))
      _wsSources=_WS_BUILTIN.concat(saved.filter(s=>s&&s.value&&s.label));
  }catch(e){}
}
function _wsSaveSources(){
  try{
    const custom=_wsSources.filter(s=>!_WS_BUILTIN.some(b=>b.value===s.value));
    localStorage.setItem(_WS_SRC_KEY,JSON.stringify(custom));
  }catch(e){}
}
function _wsRenderSourceOptions(){
  const sel=$("wsSource");if(!sel)return;
  sel.innerHTML=_wsSources.map(s=>
    "<option value='"+esc(s.value)+"'>"+esc(s.label)+"</option>").join("")
    +"<option value='__add__'>+ Add a custom site\u2026</option>";
  sel.value=_wsSource;
}
function wsOnSourceChange(){
  const sel=$("wsSource");
  if(sel.value==="__add__"){
    $("wsCustomUrl").style.display="";$("wsAddSrc").style.display="";
    $("wsSrcHint").textContent="Paste a URL with {q}, then Save site";
    $("wsCustomUrl").focus();
    return;
  }
  _wsSource=sel.value;
  $("wsCustomUrl").style.display="none";$("wsAddSrc").style.display="none";
  $("wsSrcHint").textContent=(_wsSource==="duckduckgo")
    ? "DuckDuckGo image search"
    : "Custom site \u2014 results parsed from its image tags";
  // ImageFap supports a category filter — show + populate the dropdown for it.
  const cat=$("wsCat");
  if(cat){
    if(_wsSource==="imagefap"){
      cat.style.display="";
      _wsLoadImagefapCats();
    }else{cat.style.display="none";}
  }
  // Re-run if there's an active query.
  const q=$("wsInput").value.trim();
  if(q)runWebSearch(q);
}
let _wsCatsLoaded=false;
function wsOnCatChange(){
  const q=$("wsInput").value.trim();
  if(q)runWebSearch(q);
}
async function _wsLoadImagefapCats(){
  const cat=$("wsCat");
  if(!cat||_wsCatsLoaded)return;
  try{
    const j=await(await fetch("/api/websearch/imagefap_categories")).json();
    if(j&&j.ok&&Array.isArray(j.categories)&&j.categories.length){
      cat.innerHTML=j.categories.map(c=>
        "<option value='"+c.id+"'>"+esc(c.name)+"</option>").join("");
      _wsCatsLoaded=true;
    }
  }catch(e){/* leave empty; search still works without a category */}
}
function wsAddCustomSource(){
  const url=$("wsCustomUrl").value.trim();
  if(!url||url.indexOf("{q}")<0){
    toast("The URL must contain {q} where the search term goes.",true);return;}
  let label;
  try{label=new URL(url.replace("{q}","x")).hostname.replace(/^www\./,"");}
  catch(e){label=url.slice(0,30);}
  const entry={label:label,value:url};
  // Replace if same value already saved, else add.
  if(!_wsSources.some(s=>s.value===url))_wsSources.push(entry);
  _wsSaveSources();
  _wsSource=url;
  _wsRenderSourceOptions();
  $("wsSource").value=url;
  $("wsCustomUrl").style.display="none";$("wsAddSrc").style.display="none";
  $("wsCustomUrl").value="";
  $("wsSrcHint").textContent="Saved \u2014 "+label;
  toast("Search site saved: "+label);
  const q=$("wsInput").value.trim();if(q)runWebSearch(q);
}
async function openWebSearch(){
  $("webSearchOverlay").classList.add("open");
  await _wsLoadProviders();
  if(!$("wsSource").options.length){_wsLoadSources();_wsRenderSourceOptions();}
  const inp=$("wsInput");
  if(inp){
    setTimeout(()=>inp.focus(),50);
    if(!inp._wired){
      inp._wired=true;
      inp.addEventListener("keydown",e=>{
        if(e.key==="Enter"){e.preventDefault();runWebSearch(inp.value);}
      });
    }
  }
}
function closeWebSearch(){$("webSearchOverlay").classList.remove("open");}
function wsSetMode(m){
  _wsMode=m;
  $("wsTabSingle").classList.toggle("active",m==="single");
  $("wsTabBatch").classList.toggle("active",m==="batch");
  $("wsBatchBar").style.display=(m==="batch")?"flex":"none";
  if(m==="single")_wsSelected.clear();
  _wsRenderSelection();
}
function _wsUpdateCount(){
  const el=$("wsSelCount");if(el)el.textContent=_wsSelected.size+" selected";
}
function _wsRenderSelection(){
  document.querySelectorAll("#wsGrid .ws-card").forEach(c=>{
    c.classList.toggle("ws-sel",_wsSelected.has(+c.dataset.i));
  });
  _wsUpdateCount();
}
function wsSelectAll(){
  _wsResults.forEach((_,i)=>_wsSelected.add(i));_wsRenderSelection();
}
function wsClearSel(){_wsSelected.clear();_wsRenderSelection();}
let _wsPage=1,_wsCursor=null,_wsHasMore=false,_wsLoadingMore=false,_wsQuery="";
function _wsCardHtml(res,i){
  const dims=(res.w>0&&res.h>0)?("<div class='ws-dims'>"+res.w+"\u00d7"+res.h+"</div>"):"";
  // Load the thumbnail DIRECTLY from the CDN — the browser fetches it from the
  // USER's IP (not Colab's, which some CDNs like PornPics block). If the direct
  // load fails, fall back to the backend proxy (works for boorus); if that also
  // fails, show a "no preview" placeholder. This is why previews were blank:
  // the backend IP was blocked, but the browser's IP is not.
  const raw=res.thumb||res.full;
  const prox="/api/websearch/img?url="+encodeURIComponent(raw);
  const ph="data:image/svg+xml;utf8,"
    +"%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22%3E"
    +"%3Crect width=%22100%22 height=%22100%22 fill=%22%23222%22/%3E"
    +"%3Ctext x=%2250%22 y=%2252%22 fill=%22%23888%22 font-size=%2210%22 "
    +"text-anchor=%22middle%22%3Eno preview%3C/text%3E%3C/svg%3E";
  // onerror chain: direct -> proxy -> placeholder, tracked via data-stage.
  const onerr=
    "var s=this.getAttribute('data-stage')||'direct';"
    +"if(s==='direct'){this.setAttribute('data-stage','proxy');this.src='"+prox+"';}"
    +"else{this.onerror=null;this.classList.add('ws-imgfail');this.src='"+ph+"';}";
  return "<div class='ov-card ws-card' data-i='"+i+"'>"
    +"<div style='position:relative'>"
    +"<img src='"+esc(raw)+"' loading='lazy' referrerpolicy='no-referrer' "
    +"data-stage='direct' onerror=\""+onerr+"\">"
    +dims+"<div class='ws-check'>\u2713</div>"
    +(res.gallery?"<div class='ws-galbadge'>\u25a4 gallery</div>":"")
    +"</div>"
    +"<div class='cap'>"+esc(res.title||"")+"</div></div>";
}
function _wsWireCards(scope){
  (scope||$("wsGrid")).querySelectorAll(".ws-card:not([data-wired])").forEach(c=>{
    c.setAttribute("data-wired","1");
    c.addEventListener("click",()=>{
      const i=+c.dataset.i;
      if(_wsMode==="batch"){
        if(_wsSelected.has(i))_wsSelected.delete(i);else _wsSelected.add(i);
        _wsRenderSelection();
      }else if(_wsResults[i] && _wsResults[i].gallery){
        // This result is a gallery (e.g. PornPics) — drill in to its photos.
        openWsGallery(i);
      }else{useWebImage(i,c);}
    });
  });
}
// ── Gallery drill-in ── For sources that return galleries rather than single
// images (PornPics), clicking a result opens the gallery's full-size photos.
let _wsGalleryImgs=[];
let _wsInGallery=false;
let _wsGalleryUrl=null;          // the imagefap:gid / gallery url being paged
let _wsGalleryPage=0;            // next gallery page to fetch
let _wsGalleryHasMore=false;
let _wsGalleryLoading=false;
// Wire one gallery photo card's click: ImageFap photos resolve+fetch bytes via
// the backend (browser can't); others go through useWebImageObj.
function _wsWireGalleryCard(c){
  c.addEventListener("click",async()=>{
    const k=+c.dataset.i;
    const im=_wsGalleryImgs[k];
    if(!im)return;
    if(im.photo && String(im.photo).startsWith("imagefapphoto:")){
      c.classList.add("ws-loading");
      try{
        const pj=await(await fetch("/api/websearch/photo?id="
          +encodeURIComponent(im.photo))).json();
        if(pj.ok&&pj.data_url){_wsSetSourceFromDataURL(pj.data_url);
          c.classList.remove("ws-loading");return;}
        toast("Couldn't load full image: "+(pj.error||"failed"),true);
        c.classList.remove("ws-loading");return;
      }catch(e){toast("Failed to load image.",true);
        c.classList.remove("ws-loading");return;}
    }
    useWebImageObj(im,c);
  });
}
async function openWsGallery(i){
  const res=_wsResults[i];
  if(!res||!res.gallery)return;
  _wsInGallery=true;
  _wsGalleryUrl=res.gallery;_wsGalleryPage=0;_wsGalleryHasMore=false;
  _wsGalleryLoading=false;_wsGalleryImgs=[];
  const grid=$("wsGrid");
  grid.innerHTML="<div class='empty'>Opening gallery\u2026</div>";
  try{
    const j=await(await fetch("/api/websearch/gallery?url="
      +encodeURIComponent(res.gallery)+"&page=0")).json();
    if(!j.ok||!(j.results||[]).length){
      grid.innerHTML="<div class='empty' style='color:var(--red)'>"
        +esc((j&&(j.detail||j.error))||"no images in gallery")+"</div>"
        +"<div style='text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center'>"
        +"<button class='hdr-btn' onclick='openWsGallery("+i+")'>\u21bb Retry</button>"
        +"<button class='hdr-btn' onclick='wsBackToResults()'>\u2190 Back to results</button></div>";
      return;
    }
    _wsGalleryImgs=j.results;
    _wsGalleryHasMore=!!j.next_page;
    _wsGalleryPage=j.next_page||1;
    grid.innerHTML="<div class='ws-gallery-bar'>"
      +"<button class='hdr-btn' onclick='wsBackToResults()'>\u2190 Back to results</button>"
      +"<span class='ws-gallery-title' id='wsGalTitle'>"+esc(j.title||res.title||"Gallery")
      +" \u2014 "+_wsGalleryImgs.length+" images</span></div>"
      +"<div class='ws-grid-inner' id='wsGalGrid'></div>";
    const inner=$("wsGalGrid");
    inner.innerHTML=_wsGalleryImgs.map((im,k)=>_wsCardHtml(im,k)).join("");
    inner.querySelectorAll(".ws-card").forEach(_wsWireGalleryCard);
  }catch(e){
    grid.innerHTML="<div class='empty' style='color:var(--red)'>Failed to open gallery.</div>";
  }
}
// Load the next page of gallery photos (wired to the modal's infinite scroll).
async function _wsGalleryLoadMore(){
  if(!_wsGalleryHasMore||_wsGalleryLoading||!_wsGalleryUrl)return;
  _wsGalleryLoading=true;
  const grid=$("wsGrid");
  let sentinel=grid.querySelector(".ws-more");
  if(!sentinel){sentinel=document.createElement("div");sentinel.className="ws-more";
    sentinel.textContent="Loading more\u2026";grid.appendChild(sentinel);}
  try{
    const j=await(await fetch("/api/websearch/gallery?url="
      +encodeURIComponent(_wsGalleryUrl)+"&page="+_wsGalleryPage)).json();
    if(j.ok&&(j.results||[]).length){
      const start=_wsGalleryImgs.length;
      _wsGalleryImgs=_wsGalleryImgs.concat(j.results);
      const inner=$("wsGalGrid");
      const frag=j.results.map((im,k)=>_wsCardHtml(im,start+k)).join("");
      inner.insertAdjacentHTML("beforeend",frag);
      // wire only the newly added cards
      Array.from(inner.querySelectorAll(".ws-card")).slice(start).forEach(_wsWireGalleryCard);
      _wsGalleryHasMore=!!j.next_page;
      _wsGalleryPage=j.next_page||(_wsGalleryPage+1);
      const t=$("wsGalTitle");if(t)t.textContent=t.textContent.replace(/\u2014.*/,"\u2014 "+_wsGalleryImgs.length+" images");
    }else{_wsGalleryHasMore=false;}
  }catch(e){_wsGalleryHasMore=false;}
  const s=grid.querySelector(".ws-more");if(s)s.remove();
  _wsGalleryLoading=false;
}
function wsBackToResults(){
  _wsInGallery=false;_wsGalleryImgs=[];_wsGalleryUrl=null;_wsGalleryHasMore=false;
  // Re-render the existing search results without re-fetching.
  const grid=$("wsGrid");
  grid.innerHTML="<div class='ws-grid-inner'></div>";
  const inner=grid.firstChild;
  inner.innerHTML=_wsResults.map((res,i)=>_wsCardHtml(res,i)).join("");
  _wsWireCards(inner);
  _wsRenderSelection();
}
async function runWebSearch(q){
  q=(q||"").trim();
  if(!q)return;
  _wsQuery=q;_wsPage=1;_wsCursor=null;_wsHasMore=false;_wsLoadingMore=false;
  _wsResults=[];_wsSelected.clear();_wsUpdateCount();
  const grid=$("wsGrid");
  grid.innerHTML="<div class='empty'>Searching the web\u2026</div>";
  try{
    const j=await _wsFetchPage();
    if(!j.ok){grid.innerHTML="<div class='empty' style='color:var(--red)'>"
      +esc(j.error||"search failed")+"</div>";return;}
    _wsResults=j.results||[];
    if(!_wsResults.length){
      grid.innerHTML="<div class='empty'>No images found for \""+esc(q)+"\".</div>";return;}
    grid.innerHTML="<div class='ws-grid-inner'></div>";
    const inner=grid.firstChild;
    inner.innerHTML=_wsResults.map((res,i)=>_wsCardHtml(res,i)).join("");
    _wsWireCards(inner);
    _wsRenderSelection();
    _wsUpdateMore(j);
  }catch(e){
    grid.innerHTML="<div class='empty' style='color:var(--red)'>Network error \u2014 try again.</div>";
  }
}
function _wsFetchPage(){
  let u="/api/websearch?q="+encodeURIComponent(_wsQuery)
    +"&url="+encodeURIComponent(_wsSource)+"&page="+_wsPage;
  if(_wsCursor)u+="&cursor="+encodeURIComponent(_wsCursor);
  // ImageFap category filter (gen=) when a category is chosen.
  const cat=$("wsCat");
  if(_wsSource==="imagefap"&&cat&&cat.value)u+="&gen="+encodeURIComponent(cat.value);
  return fetch(u).then(r=>r.json());
}
function _wsUpdateMore(j){
  _wsCursor=j.next_cursor||null;
  if(j.next_page){_wsPage=j.next_page;_wsHasMore=true;}
  else if(_wsCursor){_wsHasMore=true;}
  else{_wsHasMore=false;}
}
async function _wsLoadMore(){
  if(!_wsHasMore||_wsLoadingMore)return;
  _wsLoadingMore=true;
  const inner=$("wsGrid").querySelector(".ws-grid-inner");
  let sentinel=$("wsGrid").querySelector(".ws-more");
  if(!sentinel){sentinel=document.createElement("div");sentinel.className="ws-more";
    sentinel.textContent="Loading more\u2026";$("wsGrid").appendChild(sentinel);}
  try{
    const j=await _wsFetchPage();
    if(j.ok&&(j.results||[]).length){
      const start=_wsResults.length;
      _wsResults=_wsResults.concat(j.results);
      const frag=j.results.map((res,k)=>_wsCardHtml(res,start+k)).join("");
      inner.insertAdjacentHTML("beforeend",frag);
      _wsWireCards(inner);
      _wsRenderSelection();
      _wsUpdateMore(j);
    }else{_wsHasMore=false;}
  }catch(e){_wsHasMore=false;}
  const s=$("wsGrid").querySelector(".ws-more");if(s)s.remove();
  _wsLoadingMore=false;
}
// Infinite scroll: near the bottom, fetch the next page — of the gallery if
// we're drilled into one, otherwise of the search results.
(function _wsWireScroll(){
  const body=$("wsBody");
  if(body)body.addEventListener("scroll",()=>{
    const nearBottom=body.scrollTop+body.clientHeight>=body.scrollHeight-260;
    if(!nearBottom)return;
    if(_wsInGallery){
      if(!_wsGalleryLoading&&_wsGalleryHasMore)_wsGalleryLoadMore();
    }else{
      if(!_wsLoadingMore&&_wsHasMore)_wsLoadMore();
    }
  });
})();
async function useWebImage(i,card){
  const res=_wsResults[i];
  if(!res||!res.full)return;
  return useWebImageObj(res,card);
}
// Fetch an image and return a data URL, trying the BROWSER first (user's IP,
// not the blocked Colab IP) and falling back to the backend proxy (boorus).
async function _wsGetDataURL(url){
  // 1) browser fetch -> blob -> dataURL. Works when the CDN allows CORS reads
  //    (confirmed for PornPics/ImageFap from the user's IP).
  try{
    const resp=await fetch(url,{referrerPolicy:"no-referrer"});
    if(resp.ok){
      const blob=await resp.blob();
      if(blob && blob.size>500 && blob.type.startsWith("image")){
        return await new Promise((res,rej)=>{
          const fr=new FileReader();
          fr.onload=()=>res(fr.result);
          fr.onerror=rej;
          fr.readAsDataURL(blob);
        });
      }
    }
  }catch(e){/* CORS or network — fall through to canvas, then backend */}
  // 2) browser <img> + canvas (handles some cross-origin cases fetch can't).
  try{
    const durl=await new Promise((res,rej)=>{
      const im=new Image();
      im.crossOrigin="anonymous";
      im.onload=()=>{try{
        const c=document.createElement("canvas");
        c.width=im.naturalWidth;c.height=im.naturalHeight;
        c.getContext("2d").drawImage(im,0,0);
        res(c.toDataURL("image/jpeg",0.95));
      }catch(err){rej(err);}};
      im.onerror=rej;
      im.src=url;
    });
    if(durl && durl.length>1000) return durl;
  }catch(e){/* tainted/blocked — fall through to backend */}
  // 3) backend proxy (works for sources whose CDN isn't IP-blocked, e.g. boorus).
  const r=await fetch("/api/websearch/fetch?url="+encodeURIComponent(url));
  const j=await r.json();
  if(j.ok && j.data_url) return j.data_url;
  throw new Error(j.error||"could not load image");
}
// Set the img2img source from a ready data URL (used by ImageFap, whose bytes
// the backend fetched in-session, and by useWebImageObj after it resolves one).
function _wsApplySource(dataUrl){
  imgData=dataUrl;
  if(currentMode!=="img2img")setMode("img2img");
  $("drop").className="dropzone has";
  $("drop").innerHTML="<img src='"+dataUrl+"'>";
  const probe=new Image();
  probe.onload=()=>_applyUploadedSize(probe.naturalWidth,probe.naturalHeight);
  probe.src=dataUrl;
}
function _wsSetSourceFromDataURL(dataUrl){
  _wsApplySource(dataUrl);
  closeWebSearch();
  toast("Source image set from web search.");
}
async function useWebImageObj(res,card){
  if(!res||!res.full)return;
  if(card){card.classList.add("ws-loading");}
  try{
    const dataUrl=await _wsGetDataURL(res.full);
    _wsSetSourceFromDataURL(dataUrl);
  }catch(e){
    toast("Couldn't load that image: "+(e.message||e),true);
    if(card)card.classList.remove("ws-loading");
  }
}
// Run the current prompt + settings across every selected image, one img2img
// job per image. Jobs serialize on the GPU lock server-side, so they process
// one after another — exactly the studio batch behavior. We fetch each image
// through the backend, then submit a job using the current sidebar settings.
async function wsRunBatch(){
  const idxs=[..._wsSelected];
  if(!idxs.length){toast("Select at least one image first.",true);return;}
  const prompt=$("prompt").value;
  if(!prompt.trim()){toast("Enter a prompt before running the batch.",true);return;}
  const btn=$("wsRunBatch");
  if(btn){btn.disabled=true;btn.textContent="Queuing\u2026";}
  // Common settings snapshot for every job (seed handling below).
  const baseStrength=+$("strengthV").value;
  const baseSampler=$("sampler")?$("sampler").value:"";
  const randEach=$("randSeedEach")&&$("randSeedEach").checked;
  let queued=0,failed=0;
  for(const i of idxs){
    const res=_wsResults[i];
    if(!res||!res.full){failed++;continue;}
    const card=document.querySelector("#wsGrid .ws-card[data-i='"+i+"']");
    if(card)card.classList.add("ws-loading");
    try{
      const dataUrl=await _wsGetDataURL(res.full);
      // Per-image seed: random each if the toggle is on, else the fixed field.
      const seed=randEach?Math.floor(Math.random()*2e9):(+$("seed").value||0);
      const payload={mode:"img2img",prompt:prompt,negative_prompt:$("neg").value,
        steps:+$("stepsV").value||30,guidance:+$("guidV").value,
        width:+$("widthV").value||1024,height:+$("heightV").value||1024,
        batch:1,seed:seed,sampler:baseSampler,
        image:dataUrl,strength:baseStrength};
      await _submitJob(payload,dataUrl);
      queued++;
      if(card){card.classList.remove("ws-loading");card.classList.add("ws-queued");}
    }catch(e){failed++;if(card)card.classList.remove("ws-loading");}
  }
  if(btn){btn.disabled=false;btn.innerHTML="\u2726 Run batch";}
  toast("Batch queued: "+queued+" job(s)"+(failed?(", "+failed+" failed"):"")
        +". They'll run one after another \u2014 watch the Jobs panel.",failed>0);
  if(queued){_wsSelected.clear();_wsRenderSelection();closeWebSearch();}
}

/* ── LoRAs ── */
async function refreshLoras(){
  const box=$("loraList");let j;
  try{j=await(await fetch("/api/loras")).json();}
  catch(e){box.innerHTML="<div class='hintline'>Could not load LoRAs.</div>";return;}
  _lastLoras=(j.loras||[]).map(L=>({name:L.name,scale:L.scale,
    version_id:L.version_id||null,triggers:L.triggers||[],
    url:L.version_id?("https://civitai.com/api/download/models/"+L.version_id):null}));
  box.innerHTML="";
  if(!j.loras.length){_lastLoras=[];box.innerHTML="<div class='hintline' style='margin-top:10px'>No LoRAs added.</div>";if(!_comboRestoring)_saveCombo();return;}
  j.loras.forEach(L=>{
    const d=document.createElement("div");d.className="lora-card";
    let trigHtml="";
    if(L.triggers&&L.triggers.length){
      trigHtml="<div class='lora-trigs'>"
        +"<span class='lora-trigs-label'>triggers</span>";
      L.triggers.forEach(t=>{
        trigHtml+="<button class='trigchip sm' data-trig=\""+esc(t)+"\">+ "+esc(t)+"</button>";
      });
      trigHtml+="</div>";
    }
    const nameCls = L.version_id ? "lora-card-name lora-name-link" : "lora-card-name";
    d.innerHTML="<div class='lora-card-top'><span class='"+nameCls+"'"
      +(L.version_id?(" title='Open model card' data-vid='"+L.version_id+"'"):"")
      +">"+esc(L.name)+(L.attached?"":" (not attached)")+"</span>"+
      "<button class='lora-x'>&#10005;</button></div>"+
      "<div class='slider-row' style='margin:8px 0 0'><span class='sl'>strength</span>"+
      "<input type='range' min='0' max='2' step='0.05' value='"+L.scale+"'>"+
      "<input class='sv' value='"+(+L.scale).toFixed(2)+"'></div>"+
      trigHtml;
    const sl=d.querySelector("input[type=range]"),nm=d.querySelector(".sv"),xb=d.querySelector(".lora-x");
    const nameEl=d.querySelector(".lora-name-link");
    if(nameEl)nameEl.addEventListener("click",()=>openCardByVersion(L.version_id,"LORA",L.name));
    async function push(v){await fetch("/api/loras/update",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name:L.name,scale:parseFloat(v)})});}
    sl.addEventListener("input",()=>nm.value=(+sl.value).toFixed(2));
    sl.addEventListener("change",()=>push(sl.value));
    nm.addEventListener("change",()=>{sl.value=nm.value;push(nm.value);});
    xb.addEventListener("click",async()=>{
      await fetch("/api/loras/remove",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({name:L.name})});refreshLoras();refreshLoadedVersions&&refreshLoadedVersions();});
    d.querySelectorAll(".trigchip").forEach(b=>{
      b.addEventListener("click",()=>{
        const t=b.dataset.trig,p=$("prompt");
        p.value=p.value.trim()?(p.value.replace(/\s*$/,"")+", "+t):t;
        b.classList.add("trig-added");b.textContent="\u2713 "+t;
        setTimeout(()=>{b.classList.remove("trig-added");b.textContent="+ "+t;},900);
      });
    });
    box.appendChild(d);
  });
  if(!_comboRestoring)_saveCombo();
}
async function addLora(){
  const url=$("loraUrl").value.trim();
  if(!url){toast("Paste a LoRA URL first.",true);return;}
  const b=$("addLoraBtn");b.disabled=true;b.textContent="Downloading...";
  try{
    const r=await fetch("/api/loras/add",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({url:url,name:$("loraName").value.trim(),
        scale:parseFloat($("loraScale").value)||1.0})});
    const j=await r.json();
    if(j.ok)toast("LoRA added.");
    else toast("LoRA failed: "+(j.error||"unknown"),true);
  }catch(e){toast("Error: "+e,true);}
  // Reset the inputs regardless of outcome — an incompatible LoRA isn't
  // loaded, and leaving its URL/name in the boxes is just clutter.
  $("loraUrl").value="";$("loraName").value="";
  const ic=$("loraInspect");if(ic)ic.innerHTML="";
  b.disabled=false;b.textContent="+ Add LoRA";refreshLoras();
}

async function inspectLora(){
  const url=$("loraUrl").value.trim();
  if(!url){toast("Paste a LoRA URL to inspect.",true);return;}
  const b=$("inspectLoraBtn");b.disabled=true;b.textContent="Reading...";
  const box=$("loraInspect");
  box.innerHTML="<div class='hintline' style='margin-top:8px'>Downloading &amp; reading header...</div>";
  try{
    const r=await fetch("/api/loras/inspect",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({url:url})});
    const j=await r.json();
    if(!j.ok){box.innerHTML="<div class='hintline' style='margin-top:8px;color:var(--red)'>"+esc(j.error||"failed")+"</div>";}
    else{
      const i=j.info;
      const verdict=i.loadable===true?["loadable","var(--green)"]
        :i.loadable===false?["NOT loadable by diffusers","var(--red)"]
        :i.loadable==="maybe"?["might load","var(--gold)"]
        :["unknown","var(--text-muted)"];
      let h="<div class='inspect-card'>";
      h+="<div class='inspect-row'><span>format</span><b>"+esc(i.format)+"</b></div>";
      if(i.base_model)h+="<div class='inspect-row'><span>trained on</span><b>"+esc(i.base_model)+"</b></div>";
      if(i.network_module)h+="<div class='inspect-row'><span>module</span><b>"+esc(i.network_module)+"</b></div>";
      h+="<div class='inspect-row'><span>tensors</span><b>"+i.n_tensors+"</b></div>";
      h+="<div class='inspect-row'><span>verdict</span><b style='color:"+verdict[1]+"'>"+verdict[0]+"</b></div>";
      if(i.notes&&i.notes.length)
        h+="<div class='hintline' style='margin-top:6px'>"+i.notes.map(esc).join("<br>")+"</div>";
      h+="</div>";
      box.innerHTML=h;
    }
  }catch(e){box.innerHTML="<div class='hintline' style='margin-top:8px;color:var(--red)'>Error: "+esc(""+e)+"</div>";}
  b.disabled=false;b.innerHTML="\uD83D\uDD0D Inspect";
}

/* ── generate -> queue ── */
function generate(){
  const mode=currentMode;
  if(mode==="img2img" && !imgData){toast("Upload a source image first.",true);return;}
  // Fresh random seed per generation when the toggle is on — gives varied
  // results without manually editing the seed. The field updates so you can
  // see (and copy/reuse) the seed that was actually used.
  if($("randSeedEach") && $("randSeedEach").checked){ randSeed(); }
  const prompt=$("prompt").value;
  const payload={mode:mode,prompt:prompt,negative_prompt:$("neg").value,
    steps:+$("stepsV").value||30,guidance:+$("guidV").value,
    width:+$("widthV").value||1024,height:+$("heightV").value||1024,
    batch:+$("batchV").value||1,seed:+$("seed").value||0,
    sampler:($("sampler")?$("sampler").value:"")};
  let thumb=null;
  if(mode==="img2img"){payload.image=imgData;payload.strength=+$("strengthV").value;thumb=imgData;}
  _submitJob(payload,thumb);
}
// Submit one generation job and add it to the queue. Returns the fetch promise
// so callers (e.g. batch) can await/sequence. The backend threads each job and
// they serialize on the GPU lock, so multiple submits run one after another.
function _submitJob(payload,thumb){
  return fetch("/api/generate",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)})
    .then(async r=>{
      const txt=await r.text();
      if(r.status===401){_loginShow(true);throw new Error("sign in to MissingLink to generate");}
      if(!r.ok){
        throw new Error("server "+r.status+(txt?(" \u2014 "+txt.slice(0,200)):" (empty response)"));
      }
      if(!txt){throw new Error("empty response from server (the Colab tab may have lost its connection \u2014 re-run the cell)");}
      try{return JSON.parse(txt);}
      catch(e){throw new Error("non-JSON response: "+txt.slice(0,200));}
    }).then(o=>{
      if(!o||!o.job_id){throw new Error(o&&o.error?o.error:"no job id returned");}
      queue.unshift({id:o.job_id,prompt:payload.prompt,thumb:thumb,
        status:"queued",progress:0,stage:"queued",
        settings:_settingsSnapshot(),loras:_loadedLoraSnapshot()});
      renderQueue();
      return o.job_id;
    });
}
// Snapshot the current generation settings (for restoring from history).
function _settingsSnapshot(){
  return {mode:currentMode,
    prompt:$("prompt").value,negative_prompt:$("neg").value,
    steps:+$("stepsV").value,guidance:+$("guidV").value,
    width:+$("widthV").value,height:+$("heightV").value,
    seed:+$("seed").value,batch:+$("batchV").value,
    strength:+$("strengthV").value,
    sampler:($("sampler")?$("sampler").value:"")};
}
// Snapshot currently-loaded LoRAs (name, scale, url) for restoring.
function _loadedLoraSnapshot(){
  return (_lastLoras||[]).map(L=>({name:L.name,scale:L.scale,
    url:L.url||null,version_id:L.version_id||null,triggers:L.triggers||[]}));
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
    const ic=q.thumb?"<img src='"+q.thumb+"'>":"&#127912;";
    const d=document.createElement("div");d.className="q-item";
    d.innerHTML="<div class='q-ic'>"+ic+"</div>"+
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
  queue=queue.filter(x=>x.id!==id);renderQueue();
  try{await fetch("/api/cancel/"+id,{method:"POST"});}catch(e){}
  toast("Job cancelled.");
}

function renderHistory(){
  $("hCount").textContent=history.length;
  const box=$("historyBody");
  if(!history.length){box.innerHTML="<div class='empty'>Finished images appear here. Click to view.</div>";return;}
  box.innerHTML="<div class='h-grid'></div>";
  const g=box.firstChild;
  history.forEach(h=>{
    const c=document.createElement("div");c.className="history-card";c.title=h.prompt;
    const hasSettings=!!(h.settings);
    c.innerHTML="<div class='history-card-thumb'><img src='"+h.urls[0]+"'>"
      +(hasSettings?"<button class='hc-restore' title='Restore prompt, LoRAs &amp; settings'>&#8635;</button>":"")
      +"</div>";
    c.querySelector("img").onclick=()=>showImages(h);
    const rb=c.querySelector(".hc-restore");
    if(rb)rb.onclick=(ev)=>{ev.stopPropagation();restoreFromHistory(h);};
    g.appendChild(c);
  });
  renderGallery();
}

// Restore the prompt, settings, and LoRAs that produced a history image.
async function restoreFromHistory(h){
  const s=h.settings;
  if(!s){toast("This image has no saved settings to restore.",true);return;}
  if(s.mode)setMode(s.mode);
  if(s.prompt!=null)$("prompt").value=s.prompt;
  if(s.negative_prompt!=null)$("neg").value=s.negative_prompt;
  if(s.steps!=null)setSL("steps",s.steps);
  if(s.guidance!=null)setSL("guid",s.guidance);
  if(s.width!=null)setSL("width",s.width);
  if(s.height!=null)setSL("height",s.height);
  if(s.batch!=null)setSL("batch",s.batch);
  if(s.strength!=null)setSL("strength",s.strength);
  if(s.seed!=null)$("seed").value=s.seed;
  if(s.sampler!=null&&$("sampler"))$("sampler").value=s.sampler;
  $("sizePreset").value="custom";
  _saveState();
  // Re-sync the LoRA set to exactly what this image used (replace mode), so
  // restoring also removes any LoRAs that weren't part of this generation.
  const want=(h.loras||[]).filter(L=>L.url).map(L=>({
    url:L.url,name:L.name,weight:L.scale}));
  toast("Restoring settings"+(want.length?(" + "+want.length+" LoRA(s)\u2026"):"\u2026"));
  try{
    const r=await fetch("/api/loras/ensure",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({loras:want,replace:true})});
    const j=await r.json();
    refreshLoras();refreshLoadedVersions&&refreshLoadedVersions();
    if(j&&j.ok){
      const added=(j.results||[]).filter(x=>x.status==="added").length;
      const kept=(j.results||[]).filter(x=>x.status==="already-loaded").length;
      const removed=(j.removed||[]).length;
      let bits=["settings"];
      if(added)bits.push(added+" LoRA(s) loaded");
      if(kept)bits.push(kept+" kept");
      if(removed)bits.push(removed+" removed");
      toast("Restored: "+bits.join(", ")+".");
    }else{
      toast("Settings restored; LoRA sync failed: "+((j&&j.error)||"unknown"),true);
    }
  }catch(e){toast("Settings restored; LoRA sync errored: "+e,true);}
}

let _stageUrl=null;
function showImages(h){
  const v=$("viewer");
  v.classList.toggle("multi",h.urls.length>1);
  v.innerHTML=h.urls.map(u=>"<img src='"+u+"'>").join("");
  $("dlBtn").href=h.urls[0];
  _stageUrl=h.urls[0];
  $("stageTools").classList.add("show");
  $("stageClear").classList.add("show");
}
// Set the image currently shown on the stage as the img2img source. Stage
// images are same-origin server URLs, so the browser can read their bytes.
async function useStageAsInput(){
  if(!_stageUrl){toast("No image on the stage.",true);return;}
  const b=$("useAsInputBtn");const orig=b.innerHTML;
  b.disabled=true;b.textContent="Loading\u2026";
  try{
    const blob=await(await fetch(_stageUrl)).blob();
    const dataUrl=await new Promise((res,rej)=>{
      const fr=new FileReader();fr.onload=()=>res(fr.result);
      fr.onerror=()=>rej(new Error("read failed"));fr.readAsDataURL(blob);});
    _wsApplySource(dataUrl);   // sets imgData, switches to img2img, shows in dropzone
    toast("Stage image set as img2img input.");
  }catch(e){toast("Couldn't load that image as input: "+(e.message||e),true);}
  b.disabled=false;b.innerHTML=orig;
}
function clearStage(){
  const v=$("viewer");
  v.classList.remove("multi");
  _stageUrl=null;
  v.innerHTML="<div class='ph'><span class='big'>\uD83C\uDFA8</span>"
    +"Your images appear here, full size.<br>Write a <b>prompt</b> and hit <b>Generate</b>.</div>";
  $("stageTools").classList.remove("show");
  $("stageClear").classList.remove("show");
}

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
      if(j.status==="done"){
        if(!doneIds.has(q.id)){
          doneIds.add(q.id);
          let urls=null;
          try{const rr=await(await fetch("/api/result/"+q.id)).json();urls=rr.result;}
          catch(e){}
          if(urls&&urls.length){
            const h={id:q.id,prompt:q.prompt,thumb:urls[0],urls:urls,ts:Date.now(),
                     settings:q.settings||null,loras:q.loras||[]};
            history.unshift(h);showImages(h);renderHistory();_saveState();}
        }
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="error"){
        toast("Job failed: "+(j.error||"unknown"),true);
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="cancelled"){
        queue=queue.filter(x=>x.id!==q.id);changed=true;
      }else if(j.status==="running"){running=q;}
    }
    const ss=$("stage_status");
    if(running){$("fill").style.width=(running.progress||0)+"%";
      ss.textContent=running.stage||"";ss.classList.add("show");
      const pend=queue.filter(x=>x.status==="queued").length;
      $("genBtn").innerHTML="\u2726 Generate &mdash; Working"
        +(pend?(" (+"+pend+" queued)"):"")+"\u2026";}
    else{$("fill").style.width="0%";ss.classList.remove("show");
      $("genBtn").innerHTML="\u2726 Generate Image";}
    if(changed)renderQueue();
  }finally{_polling=false;}
},800);

function renderGallery(){
  const g=$("ovGrid");if(!g)return;
  let total=0;history.forEach(h=>total+=h.urls.length);
  $("ovCount").textContent=total;
  if(!history.length){g.innerHTML="<div class='empty'>No images yet this session.</div>";return;}
  g.innerHTML="";
  history.forEach(h=>{
    h.urls.forEach(u=>{
      const c=document.createElement("div");c.className="ov-card";
      c.innerHTML="<img src='"+u+"'><div class='cap'>"+esc(h.prompt)+"</div>";
      c.onclick=()=>{showImages(h);closeGallery();};
      g.appendChild(c);
    });
  });
}
function openGallery(){renderGallery();$("overlay").classList.add("open");}
function closeGallery(){$("overlay").classList.remove("open");}

/* ── Civitai model browser (LoRAs + checkpoints) ── */
// Cursor-based infinite scroll. Civitai requires cursor pagination whenever
// a text query is used (page+query => 400) and caps page*limit at 1000, so
// we drive everything off metadata.nextCursor and an IntersectionObserver.
let _lbType="LORA";
let _lbCursor=null;        // next cursor to request (null = none/end)
let _lbLoading=false;      // a fetch is in flight
let _lbDone=false;         // no more results
let _lbSeq=0;              // bumped on every new search to cancel stale loads
let _lbObserver=null;
let _lbDebounce=null;
let _loadedLoraVids=new Set();    // Civitai version ids already in VRAM
let _loadedCkptVids=new Set();
let _loadedLoraMap={};            // version_id -> {name, scale}
let _lastLoras=[];                // last fetched loaded-LoRA list (for snapshots)
let _activeModelVid=null;         // Civitai version id of the loaded checkpoint
function _applyModelDefaults(j,setSliders){
  if(j.arch)_activeArch=j.arch;
  const d=j.defaults||{};
  const gh=$("guidHint");
  if(gh&&d.guidance_hint)gh.textContent=d.guidance_hint;
  if(setSliders){
    if(d.steps!=null){$("steps").value=d.steps;$("stepsV").value=d.steps;}
    if(d.guidance!=null){$("guid").value=d.guidance;$("guidV").value=d.guidance;}
    _saveState&&_saveState();
  }
  const neg=$("neg");
  if(neg)neg.style.opacity=(d.supports_negative===false)?"0.45":"";
  if(neg)neg.title=(d.supports_negative===false)?"FLUX.1-dev ignores the negative prompt":"";
}
function _markActiveModel(arch,vid){
  const map={qwen:"presetQwen",flux:"presetFlux",zimage:"presetZ"};
  Object.values(map).forEach(id=>{const b=$(id);
    if(b){b.classList.remove("model-active","custom");b.removeAttribute("title");}});
  const b=$(map[arch||""]);if(!b)return;
  b.classList.add("model-active");
  if(vid){b.classList.add("custom");
    b.title="A custom "+arch+"-family Civitai checkpoint is loaded (see name above)";}
  else b.title="This model is currently loaded";
}
async function refreshModelVid(){
  try{
    const j=await(await fetch("/api/model")).json();
    _activeModelVid=j.version_id||null;
    _applyModelDefaults(j,false);
    _markActiveModel(j.arch,j.version_id||null);
    const el=$("modelNow");
    if(el){
      if(_activeModelVid){el.classList.add("model-name-link");el.title="Open model card";}
      else{el.classList.remove("model-name-link");el.removeAttribute("title");}
    }
  }catch(e){}
}
let _lbActiveTag="";              // currently selected quick-pick tag
let _lbTagsLoaded=false;
let _activeArch="qwen";        // family of the resident model (qwen|flux|zimage)
function _familyOf(base){
  // Map a Civitai base-model string ("Qwen", "Flux.1 D", "ZImageTurbo"...)
  // to a supported family key. Flux.2 is a different architecture -> null.
  const b=(base||"").toLowerCase().replace(/[\s.\-_]/g,"");
  if(!b)return null;
  if(b.indexOf("qwen")>=0)return "qwen";
  if(b.indexOf("zimage")>=0)return "zimage";
  if(b.indexOf("flux2")>=0)return null;
  if(b.indexOf("flux")>=0)return "flux";
  return null;
}

let _lbExtraTags=[];          // tags shown only when "more" expanded
let _lbMoreOpen=false;

function _makeTagChip(t,box){
  const b=document.createElement("button");
  b.className="lb-tagchip";b.textContent=t;b.dataset.tag=t;
  if(_lbActiveTag===t)b.classList.add("active");
  b.addEventListener("click",()=>_selectTag(t));
  box.appendChild(b);
  return b;
}
function _selectTag(t){
  const box=$("lbTagChips");
  if(_lbActiveTag===t){_lbActiveTag="";}
  else{_lbActiveTag=t;}
  box.querySelectorAll(".lb-tagchip").forEach(x=>{
    x.classList.toggle("active", x.dataset.tag===_lbActiveTag);
  });
  lbSearch();
}
function _renderTagChips(){
  const box=$("lbTagChips");if(!box)return;
  box.innerHTML="";
  _lbPrimaryTags.forEach(t=>_makeTagChip(t,box));
  if(_lbMoreOpen)_lbExtraTags.forEach(t=>_makeTagChip(t,box));
}
function _toggleTagBar(){
  const bar=$("lbTagBar"),btn=$("lbTagsToggle");
  const nowHidden=bar.classList.toggle("collapsed");
  if(btn)btn.classList.toggle("on",!nowHidden);
}
function _toggleMoreTags(){
  _lbMoreOpen=!_lbMoreOpen;
  const btn=$("lbTagMore");
  if(btn)btn.textContent=_lbMoreOpen?"less \u2212":"more +";
  _renderTagChips();
}
let _lbPrimaryTags=[];
let _lbTagFilterTimer=null;
async function loadTagChips(){
  if(_lbTagsLoaded){_renderTagChips();return;}
  try{
    const j=await(await fetch("/api/loras/tags")).json();
    _lbPrimaryTags=j.primary||[];
    _lbExtraTags=j.extra||[];
    _renderTagChips();
    const inp=$("lbTagInput");
    if(inp && !inp._wired){
      inp._wired=true;
      // Enter = use exactly what's typed as the tag filter.
      inp.addEventListener("keydown",e=>{
        if(e.key==="Enter"){
          const t=inp.value.trim().toLowerCase();
          if(t){_lbActiveTag=t;_renderTagChips();lbSearch();}
        }else if(e.key==="Escape"){inp.value="";_lbFilterTags("");}
      });
      // Typing = live-filter Civitai's full tag set into the chip bar, so you
      // can discover and pick ANY tag the site exposes, not just the defaults.
      inp.addEventListener("input",()=>{
        clearTimeout(_lbTagFilterTimer);
        const q=inp.value.trim();
        _lbTagFilterTimer=setTimeout(()=>_lbFilterTags(q),200);
      });
    }
    _lbTagsLoaded=true;
  }catch(e){}
}
async function _lbFilterTags(q){
  const box=$("lbTagChips");if(!box)return;
  if(!q){_renderTagChips();return;}
  try{
    const j=await(await fetch("/api/loras/tags?q="+encodeURIComponent(q))).json();
    const tags=(j.tags||[]).slice(0,40);
    box.innerHTML="";
    if(!tags.length){
      box.innerHTML="<span class='hintline' style='margin:0'>No Civitai tags match \u201c"+esc(q)+"\u201d \u2014 press Enter to use it anyway.</span>";
      return;
    }
    tags.forEach(t=>{
      const chip=_makeTagChip(t.name,box);
      if(t.count)chip.title=t.count.toLocaleString()+" models";
    });
  }catch(e){_renderTagChips();}
}

function _isSupportedFamily(base){return _familyOf(base)!==null;}
function _matchesActiveArch(base){return _familyOf(base)===_activeArch;}

// Fetch which Civitai versions are currently resident so we can gray out
// their cards. Cheap; called on open and after any load.
async function refreshLoadedVersions(){
  try{
    const j=await(await fetch("/api/loras/loaded")).json();
    _loadedLoraVids=new Set((j.lora_version_ids||[]).map(String));
    _loadedCkptVids=new Set((j.checkpoint_version_ids||[]).map(String));
    _loadedLoraMap=j.lora_map||{};
  }catch(e){}
}
function _isLoadedVersion(it){
  const vid=String(it.version_id||"");
  if(!vid)return false;
  return _lbType==="Checkpoint" ? _loadedCkptVids.has(vid) : _loadedLoraVids.has(vid);
}
function _loadedNameFor(info){
  const e=_loadedLoraMap[String(info.version_id||"")];
  return e?e.name:null;
}
function _loadedScaleFor(info){
  const e=_loadedLoraMap[String(info.version_id||"")];
  return e?e.scale:null;
}
function _lbdWeight(){
  const ws=document.getElementById("lbdWeight");
  return ws?parseFloat(ws.value):1.0;
}

function openLoraBrowser(){lbSetType("LORA");_openBrowser();}
function openCkptBrowser(){lbSetType("Checkpoint");_openBrowser();}
function _openBrowser(){
  $("loraOverlay").classList.add("open");
  const sel=$("lbBase");
  sel.value=({qwen:"Qwen",flux:"Flux.1 D",zimage:"ZImageTurbo"})[_activeArch]||"Qwen";
  _setupLbObserver();
  loadTagChips();
  refreshLoadedVersions().then(()=>lbSearch());
}
function closeLoraBrowser(){$("loraOverlay").classList.remove("open");}

function lbSetType(t){
  _lbType=t;
  $("lbTabLora").classList.toggle("active",t==="LORA");
  $("lbTabCkpt").classList.toggle("active",t==="Checkpoint");
  $("lbTitle").textContent=t==="Checkpoint"?"CIVITAI CHECKPOINTS":"CIVITAI LORAS";
  $("lbQuery").placeholder=t==="Checkpoint"?"search checkpoints by name...":"search LoRAs by name...";
}

// Debounced live search as you type — no Search button needed.
function _lbQueryInput(){
  clearTimeout(_lbDebounce);
  _lbDebounce=setTimeout(()=>lbSearch(),350);
}

// Fresh search: reset cursor + grid, then load the first batch.
function lbSearch(){
  // If the detail view is open, return to the grid for the new search.
  $("lbDetail").style.display="none";
  $("lbScroll").style.display="";
  const seq=++_lbSeq;
  _lbCursor=null;_lbDone=false;_lbLoading=false;
  const grid=$("lbGrid");
  grid.innerHTML="";
  $("lbStatus").textContent="Searching Civitai\u2026";
  lbLoadMore(seq,true);
}

async function lbLoadMore(seq,first){
  if(_lbLoading||_lbDone)return;
  if(seq===undefined)seq=_lbSeq;
  if(seq!==_lbSeq)return;          // a newer search superseded this one
  _lbLoading=true;
  if(!first)$("lbStatus").textContent="Loading more\u2026";
  try{
    const body={type:_lbType,query:$("lbQuery").value.trim(),
      base_model:$("lbBase").value,sort:$("lbSort").value,
      period:$("lbPeriod").value,gen_only:$("lbGen").checked,
      tag:(_lbActiveTag||""),
      kind:(_lbType==="LORA"&&$("lbKind")?$("lbKind").value:"any"),
      nsfw:$("lbNsfw").checked,cursor:_lbCursor};
    const r=await fetch("/api/loras/search",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const j=await r.json();
    if(seq!==_lbSeq){_lbLoading=false;return;}   // stale; drop it
    if(!j.ok){
      $("lbStatus").innerHTML="<span style='color:var(--red)'>"+esc(j.error||"search failed")+"</span>";
      _lbLoading=false;_lbDone=true;return;}
    appendLbResults(j.items||[]);
    _lbCursor=j.next_cursor||null;
    if(!_lbCursor)_lbDone=true;
    const n=$("lbGrid").children.length;
    if(n===0){$("lbStatus").textContent="Nothing found. Try a different search or filters.";}
    else if(_lbDone){$("lbStatus").textContent=n+" result"+(n===1?"":"s")+" \u2014 end.";}
    else{$("lbStatus").textContent=n+" loaded \u2014 scroll for more\u2026";}
  }catch(e){
    if(seq===_lbSeq)$("lbStatus").innerHTML="<span style='color:var(--red)'>Error: "+esc(""+e)+"</span>";
  }
  _lbLoading=false;
  // If the sentinel is still visible (tall viewport, few results), keep going.
  if(!_lbDone&&seq===_lbSeq&&_sentinelVisible())lbLoadMore(seq,false);
}

function _sentinelVisible(){
  const s=$("lbSentinel"),sc=$("lbScroll");
  if(!s||!sc)return false;
  const sr=s.getBoundingClientRect(),cr=sc.getBoundingClientRect();
  return sr.top<=cr.bottom+200;
}
function _setupLbObserver(){
  if(_lbObserver)return;
  const sc=$("lbScroll");
  _lbObserver=new IntersectionObserver((entries)=>{
    if(entries.some(e=>e.isIntersecting))lbLoadMore(_lbSeq,false);
  },{root:sc,rootMargin:"300px"});
  _lbObserver.observe($("lbSentinel"));
}

/* ── model detail view ── */
let _lbDetailItem=null;
function backToSearch(){
  _dSeq++;                       // cancel any in-flight sample loads
  if(_dObserver){_dObserver.disconnect();_dObserver=null;}
  $("lbDetail").style.display="none";
  $("lbScroll").style.display="";
}
async function openDetail(it){
  _lbDetailItem=it;
  const isCkpt=_lbType==="Checkpoint";
  $("lbScroll").style.display="none";
  const d=$("lbDetail");d.style.display="";
  d.scrollTop=0;
  d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button>"
    +"<div class='lbd-title'>"+esc(it.name||"untitled")+"</div></div>"
    +"<div class='empty'>Loading model details &amp; sample gallery\u2026</div>";
  try{
    const r=await fetch("/api/loras/detail",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model_id:it.model_id,version_id:it.version_id,
        nsfw:$("lbNsfw").checked})});
    const j=await r.json();
    if(!j.ok){d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button></div>"
      +"<div class='empty' style='color:var(--red)'>"+esc(j.error||"failed")+"</div>";return;}
    renderDetail(j.info,isCkpt);
  }catch(e){
    d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button></div>"
      +"<div class='empty' style='color:var(--red)'>Error: "+esc(""+e)+"</div>";
  }
}

// Open a model's detail card from anywhere (e.g. clicking a loaded LoRA or the
// active-model name) using only a Civitai version id — the backend resolves the
// model id. `kind` is "LORA" or "Checkpoint" so the card behaves correctly.
async function openCardByVersion(versionId,kind,displayName){
  if(!versionId){toast("No Civitai link for this item.",true);return;}
  _openBrowser();
  lbSetType(kind==="Checkpoint"?"Checkpoint":"LORA");
  // Skip the search list — go straight to the detail view.
  $("lbScroll").style.display="none";
  const d=$("lbDetail");d.style.display="";d.scrollTop=0;
  d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button>"
    +"<div class='lbd-title'>"+esc(displayName||"loading\u2026")+"</div></div>"
    +"<div class='empty'>Loading model details &amp; sample gallery\u2026</div>";
  try{
    const r=await fetch("/api/loras/detail",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({version_id:versionId,nsfw:$("lbNsfw").checked})});
    const j=await r.json();
    if(!j.ok){d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button></div>"
      +"<div class='empty' style='color:var(--red)'>"+esc(j.error||"failed")+"</div>";return;}
    _lbDetailItem=j.info;
    renderDetail(j.info,kind==="Checkpoint");
  }catch(e){
    d.innerHTML="<div class='lbd-head'><button class='lbd-back' onclick='backToSearch()'>\u2190 Back</button></div>"
      +"<div class='empty' style='color:var(--red)'>Error: "+esc(""+e)+"</div>";
  }
}

// Detail-gallery infinite-scroll state.
let _dSamples=[];          // all loaded samples (for lightbox + recipe lookup)
let _dCursor=null,_dLoading=false,_dDone=false,_dSeq=0,_dObserver=null;
let _dModelId=null,_dVersionId=null;

function renderDetail(info,isCkpt){
  const d=$("lbDetail");
  const incompatible=isCkpt&&!_isSupportedFamily(info.base_model);
  const loadLabel=isCkpt?"\u21BB Load as base model":"+ Add this LoRA";
  let h="<div class='lbd-head'>"
    +"<button class='lbd-back' onclick='backToSearch()'>\u2190 Back to search</button>"
    +"<div class='lbd-title'>"+esc(info.name||"untitled")+"</div>"
    +"<button class='lbd-load' id='lbdLoadBtn'>"+loadLabel+"</button></div>";
  h+="<div class='lbd-meta'>";
  if(info.base_model)h+="<span class='lbd-stat"+(incompatible?"' style='color:var(--red)":"")+"'>base <b>"+esc(info.base_model)+"</b></span>";
  if(info.type)h+="<span class='lbd-stat'>type <b>"+esc(info.type)+"</b></span>";
  if(info.version_name)h+="<span class='lbd-stat'>version <b>"+esc(info.version_name)+"</b></span>";
  if(info.creator)h+="<span class='lbd-stat'>by <b>"+esc(info.creator)+"</b></span>";
  if(info.downloads!=null)h+="<span class='lbd-stat'>downloads <b>"+Number(info.downloads).toLocaleString()+"</b></span>";
  if(info.thumbs_up!=null)h+="<span class='lbd-stat'>\u2191 <b>"+Number(info.thumbs_up).toLocaleString()+"</b></span>";
  if(info.file_mb)h+="<span class='lbd-stat'>file <b>"+info.file_mb+" MB</b></span>";
  h+="</div>";
  // Version picker — a model page can have many very different versions
  // (DMD2, Realism, aBEAST, KiSS...). Let the user choose which one loads.
  if(info.versions && info.versions.length>1){
    h+="<div class='lbd-verrow'><span class='lbd-verlabel'>version</span>"
      +"<select id='lbdVersion' class='lb-select' style='flex:1'>";
    info.versions.forEach(v=>{
      const sel=(String(v.version_id)===String(info.version_id))?" selected":"";
      const mb=v.file_mb?(" \u00b7 "+v.file_mb+" MB"):"";
      const bm=v.base_model?(" ["+esc(v.base_model)+"]"):"";
      h+="<option value='"+v.version_id+"'"+sel+">"+esc(v.version_name||("v"+v.version_id))+bm+mb+"</option>";
    });
    h+="</select></div>";
  }
  if(incompatible)h+="<div class='hintline' style='color:var(--red);margin-top:0'>Not a Qwen / Flux.1 / Z-Image checkpoint \u2014 it will fail to load in this pipeline.</div>";
  else if(!isCkpt&&info.base_model&&!_matchesActiveArch(info.base_model))h+="<div class='hintline' style='color:var(--red);margin-top:0'>This LoRA targets a different family ("+esc(_familyOf(info.base_model)||"?")+") than the resident model \u2014 switch the base model first.</div>";
  // Trigger words — clickable chips that insert into the prompt. These are the
  // tokens the LoRA was trained on; the LoRA only expresses its concept when
  // they appear in your prompt, so make them one-click to add.
  if(!isCkpt && info.triggers && info.triggers.length){
    h+="<div class='lbd-section'>Trigger words <span style='font-weight:400;text-transform:none;color:var(--text-dim)'>(click to add to prompt)</span></div>";
    h+="<div class='lbd-triggers'>";
    info.triggers.forEach(t=>{
      h+="<button class='trigchip' data-trig=\""+esc(t)+"\">+ "+esc(t)+"</button>";
    });
    h+="</div>";
  }else if(info.triggers && info.triggers.length){
    h+="<div class='lbd-stat' style='display:inline-block;margin-bottom:12px'>triggers: <b>"+esc(info.triggers.join(", "))+"</b></div>";
  }else if(!isCkpt){
    h+="<div class='lbd-section'>Trigger words</div>"
      +"<div class='hintline' style='margin:0 0 12px'>This LoRA has no trigger words listed on Civitai \u2014 it likely activates without a specific keyword. Just describe what you want.</div>";
  }
  // Weight control for LoRAs: pick the strength to load at, and if already
  // loaded, drag to change it live.
  if(!isCkpt){
    h+="<div class='lbd-weightrow'>"
      +"<span class='lbd-wlabel'>weight</span>"
      +"<input type='range' id='lbdWeight' min='0' max='1.5' step='0.05' value='1'>"
      +"<span class='lbd-wval' id='lbdWeightVal'>1.00</span></div>";
  }
  if(info.civitai_url)
    h+="<div style='margin-bottom:12px'><a href='"+esc(info.civitai_url)+"' target='_blank' style='color:var(--gold);font-family:var(--font-mono);font-size:10px'>view on civitai.com \u2197</a></div>";
  if(info.description)
    h+="<div class='lbd-section'>Description</div><div class='lbd-desc'>"+esc(info.description)+"</div>";
  h+="<div class='lbd-section'>Sample generations</div>";
  h+="<div class='lbd-gallery' id='lbdGallery'></div>";
  h+="<div id='lbdSentinel' style='height:1px'></div>";
  h+="<div id='lbdStatus' class='lb-pager'></div>";
  d.innerHTML=h;

  const lb=$("lbdLoadBtn");
  const alreadyLoaded=_isLoadedVersion(info);
  if(lb&&alreadyLoaded){
    lb.textContent="\u2713 In VRAM";lb.disabled=true;
  }else if(lb){
    lb.addEventListener("click",()=>{
      // Resolve the chosen version: if the picker exists, use its selection;
      // otherwise the item as-is.
      let item=_lbDetailItem;
      const vsel=$("lbdVersion");
      if(vsel && info.versions){
        const chosen=info.versions.find(v=>String(v.version_id)===String(vsel.value));
        if(chosen){
          item=Object.assign({},_lbDetailItem,{
            version_id:chosen.version_id,
            download_url:chosen.download_url,
            base_model:chosen.base_model||_lbDetailItem.base_model,
            triggers:chosen.triggers&&chosen.triggers.length?chosen.triggers:_lbDetailItem.triggers,
          });
        }
      }
      if(isCkpt)lbLoadCkpt(item,lb);
      else lbLoad(item,lb,_lbdWeight());
    });
  }

  // Trigger chips -> insert into the prompt box.
  d.querySelectorAll(".trigchip").forEach(b=>{
    b.addEventListener("click",()=>{
      const t=b.dataset.trig;
      const p=$("prompt");
      p.value = p.value.trim() ? (p.value.replace(/\s*$/,"")+", "+t) : t;
      b.classList.add("trig-added");b.textContent="\u2713 "+t;
      setTimeout(()=>{b.classList.remove("trig-added");b.textContent="+ "+t;},900);
    });
  });

  // Weight slider: live-adjust if this LoRA is already loaded; otherwise it
  // just sets the strength the load button will use.
  const ws=$("lbdWeight");
  if(ws){
    if(alreadyLoaded){
      const cur=_loadedScaleFor(info);
      if(cur!=null){ws.value=cur;$("lbdWeightVal").textContent=Number(cur).toFixed(2);}
    }
    ws.addEventListener("input",()=>{
      $("lbdWeightVal").textContent=Number(ws.value).toFixed(2);
    });
    ws.addEventListener("change",()=>{
      if(alreadyLoaded){
        const nm=_loadedNameFor(info);
        if(nm){
          fetch("/api/loras/update",{method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({name:nm,scale:parseFloat(ws.value)})})
            .then(()=>{refreshLoras();toast("Weight \u2192 "+Number(ws.value).toFixed(2));});
        }
      }
    });
  }

  // Seed the gallery with the first page from the detail call, then paginate.
  _dSamples=[];_dCursor=info.samples_cursor||null;_dLoading=false;_dDone=false;
  _dSeq++;_dModelId=info.model_id;_dVersionId=info.version_id;
  _dAppendSamples(info.samples||[]);
  if(!(info.samples||[]).length && !_dCursor){
    $("lbdStatus").textContent="No community sample images available.";
    _dDone=true;
  }else if(!_dCursor){
    _dDone=true;_dUpdateStatus();
  }else{
    _dUpdateStatus();
  }
  _setupDetailObserver();
  // If the page is tall and the first batch didn't fill it, keep loading.
  if(!_dDone&&_detailSentinelVisible())_dLoadMoreSamples();
}

function _dUpdateStatus(){
  const n=_dSamples.length;
  if(_dDone)$("lbdStatus").textContent=n?(n+" sample"+(n===1?"":"s")+" \u2014 end."):"No samples.";
  else $("lbdStatus").textContent=n+" loaded \u2014 scroll for more\u2026";
}

function _dAppendSamples(items){
  const g=$("lbdGallery");if(!g)return;
  const frag=document.createDocumentFragment();
  items.forEach(s=>{
    const idx=_dSamples.length;
    _dSamples.push(s);
    const cell=document.createElement("div");cell.className="lbd-sample";
    const img=s.thumb
      ?"<img loading='lazy' decoding='async' src='/api/loras/thumb?u="+encodeURIComponent(s.thumb)+"'>":"";
    let recipe="";
    if(s.prompt){
      recipe="<div class='lbd-prompt'>"+esc(s.prompt)+"</div>"
        +"<button class='lbd-use'>use this prompt &amp; settings</button>";
    }
    cell.innerHTML=img+"<div class='lbd-samp-body'>"+recipe+"</div>";
    // Click the image -> fullscreen lightbox.
    const imgEl=cell.querySelector("img");
    if(imgEl)imgEl.addEventListener("click",()=>openLightbox(idx));
    const useBtn=cell.querySelector(".lbd-use");
    if(useBtn)useBtn.addEventListener("click",ev=>{ev.stopPropagation();useRecipe(_dSamples[idx]);});
    frag.appendChild(cell);
  });
  g.appendChild(frag);
}

async function _dLoadMoreSamples(){
  if(_dLoading||_dDone||!_dCursor)return;
  const seq=_dSeq;
  _dLoading=true;
  $("lbdStatus").textContent="Loading more samples\u2026";
  try{
    const r=await fetch("/api/loras/samples",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model_id:_dModelId,version_id:_dVersionId,
        nsfw:$("lbNsfw").checked,cursor:_dCursor})});
    const j=await r.json();
    if(seq!==_dSeq){_dLoading=false;return;}    // detail view changed
    if(!j.ok){$("lbdStatus").innerHTML="<span style='color:var(--red)'>"+esc(j.error||"failed")+"</span>";_dDone=true;_dLoading=false;return;}
    _dAppendSamples(j.samples||[]);
    _dCursor=j.next_cursor||null;
    if(!_dCursor)_dDone=true;
    _dUpdateStatus();
  }catch(e){
    if(seq===_dSeq)$("lbdStatus").innerHTML="<span style='color:var(--red)'>Error: "+esc(""+e)+"</span>";
  }
  _dLoading=false;
  if(!_dDone&&seq===_dSeq&&_detailSentinelVisible())_dLoadMoreSamples();
}

function _detailSentinelVisible(){
  const s=$("lbdSentinel"),sc=$("lbDetail");
  if(!s||!sc)return false;
  const sr=s.getBoundingClientRect(),cr=sc.getBoundingClientRect();
  return sr.top<=cr.bottom+250;
}
function _setupDetailObserver(){
  const sc=$("lbDetail"),sentinel=$("lbdSentinel");
  if(!sentinel)return;
  if(_dObserver)_dObserver.disconnect();
  _dObserver=new IntersectionObserver((entries)=>{
    if(entries.some(e=>e.isIntersecting))_dLoadMoreSamples();
  },{root:sc,rootMargin:"400px"});
  _dObserver.observe(sentinel);
}

/* ── fullscreen lightbox ── */
function openLightbox(idx){
  const s=_dSamples[idx];if(!s)return;
  _lbxIdx=idx;
  const o=$("lightbox");
  const url=s.full||s.thumb;
  let meta="";
  const bits=[];
  if(s.steps)bits.push("steps "+s.steps);
  if(s.cfg)bits.push("cfg "+s.cfg);
  if(s.sampler)bits.push(esc(s.sampler));
  if(s.seed!=null)bits.push("seed "+s.seed);
  meta=bits.join("  \u00b7  ");
  $("lbxImg").src="/api/loras/thumb?u="+encodeURIComponent(url);
  $("lbxPrompt").textContent=s.prompt||"(no prompt metadata)";
  $("lbxMeta").textContent=meta;
  const ub=$("lbxUse");
  ub.style.display=s.prompt?"":"none";
  ub.onclick=()=>{useRecipe(s);closeLightbox();};
  o.classList.add("open");
}
function closeLightbox(){$("lightbox").classList.remove("open");}
function _lbxNav(delta){
  let i=_lbxIdx+delta;
  if(i<0)i=0;
  if(i>=_dSamples.length){
    // try to pull more, then advance if available
    if(!_dDone)_dLoadMoreSamples();
    i=_dSamples.length-1;
  }
  if(i!==_lbxIdx)openLightbox(i);
}
let _lbxIdx=0;

async function useRecipe(s){
  if(!s)return;
  if(s.prompt!=null)$("prompt").value=s.prompt;
  if(s.negative!=null)$("neg").value=s.negative;
  if(s.steps){$("steps").value=Math.min(60,Math.max(1,Math.round(s.steps)));
    $("stepsV").value=$("steps").value;}
  if(s.cfg){$("guid").value=s.cfg;$("guidV").value=s.cfg;}
  if(s.seed!=null&&s.seed!==-1){$("seed").value=s.seed;}
  // Match the reference's sampler if we recognize it (sets the dropdown; the
  // backend maps the name to the right diffusers scheduler).
  if(s.sampler){
    const sel=$("sampler");
    if(sel){
      const want=String(s.sampler).toLowerCase();
      let matched="";
      for(const o of sel.options){
        if(o.value && o.value.toLowerCase()===want){matched=o.value;break;}
      }
      // loose fallback: contains match (e.g. "DPM++ 2M Karras" variants)
      if(!matched){
        for(const o of sel.options){
          if(o.value && want.indexOf(o.value.toLowerCase())>=0){matched=o.value;break;}
        }
      }
      if(matched){sel.value=matched;}
    }
  }

  // Build the list of LoRAs to ensure are loaded:
  //  1) the model being viewed, if it's a LoRA (most reliable match)
  //  2) any LoRAs this sample's metadata recorded (civitaiResources)
  const want=[];
  const seen=new Set();
  const add=(url,weight,name)=>{
    if(!url)return;
    const vid=(url.match(/\/models\/(\d+)/)||[])[1]||url;
    if(seen.has(vid))return;seen.add(vid);
    want.push({url:url,weight:weight,name:name});
  };
  if(_lbDetailItem && _lbType==="LORA" && _lbDetailItem.download_url)
    add(_lbDetailItem.download_url,null,_lbDetailItem.name);
  (s.loras||[]).forEach(l=>add(l.download_url,l.weight,null));

  // "Use this recipe" means make my setup MATCH this image — so replace the
  // current LoRA set rather than accumulate. If the image has no LoRA metadata
  // but we DO have LoRAs loaded, still clear them so old ones don't pollute.
  if(!want.length){
    // Clear any currently-loaded LoRAs so a no-LoRA recipe is actually no-LoRA.
    try{
      await fetch("/api/loras/ensure",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({loras:[],replace:true})});
      refreshLoras();refreshLoadedVersions&&refreshLoadedVersions();
    }catch(e){}
    toast("Prompt & settings copied. (This image lists no LoRAs — cleared any loaded ones.)");
    closeLoraBrowser();return;
  }
  toast("Prompt & settings copied \u2014 syncing "+want.length+" LoRA(s)\u2026");
  try{
    const r=await fetch("/api/loras/ensure",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({loras:want,replace:true})});
    const j=await r.json();
    refreshLoras();refreshLoadedVersions&&refreshLoadedVersions();
    if(j.ok){
      const added=j.results.filter(x=>x.status==="added").length;
      const skip=j.results.filter(x=>x.status==="already-loaded").length;
      const fail=j.results.filter(x=>x.status==="failed").length;
      const removed=(j.removed||[]).length;
      let msg=[];
      if(added)msg.push(added+" loaded");
      if(skip)msg.push(skip+" already present");
      if(removed)msg.push(removed+" removed");
      if(fail)msg.push(fail+" failed");
      toast("Recipe synced \u2014 LoRAs: "+(msg.join(", ")||"none")+".",fail>0);
    }else{
      toast("Settings copied, but LoRA sync failed: "+(j.error||"unknown"),true);
    }
  }catch(e){toast("Settings copied, but LoRA sync errored: "+e,true);}
  closeLoraBrowser();
}

// Append-only render — never rebuilds existing cards, so scrolling stays
// smooth and already-loaded thumbnails are never re-requested.
function appendLbResults(items){
  const grid=$("lbGrid");
  const isCkpt=_lbType==="Checkpoint";
  const frag=document.createDocumentFragment();
  items.forEach(it=>{
    const c=document.createElement("div");c.className="lb-card";
    const loaded=_isLoadedVersion(it);
    if(loaded)c.classList.add("lb-loaded");
    const thumb=it.thumb
      ? "<img class='lb-thumb' loading='lazy' decoding='async' src='/api/loras/thumb?u="+encodeURIComponent(it.thumb)+"'>"
      : "<div class='lb-noimg'>"+(isCkpt?"&#9638;":"&#9880;")+"</div>";
    let typeBadge="",incompatible=false,warnNote="";
    if(isCkpt){
      if(!_isSupportedFamily(it.base_model)){
        incompatible=true;
        warnNote="not a Qwen / Flux.1 / Z-Image checkpoint \u2014 won't load in this pipeline";}
    }else{
      if(it.base_model&&!_matchesActiveArch(it.base_model)){
        const fam=_familyOf(it.base_model);
        warnNote=fam?("for "+fam+" \u2014 loading it will swap nothing; it only attaches when that model is active"):"unsupported base model \u2014 won't load on Qwen/Flux/Z-Image";
        if(!fam)incompatible=true;}
      if(it.type&&it.type!=="LORA"){
        const warn=(it.type==="LoCon"||it.type==="DoRA"||it.type==="LyCORIS");
        typeBadge="<span class='lb-tag "+(warn?"warn":"type")+"'>"+esc(it.type)+"</span>";
        if(warn)warnNote="diffusers may not load this format";}
    }
    let tags="<div class='lb-tags'>";
    if(loaded)tags+="<span class='lb-tag loaded'>\u2713 in VRAM</span>";
    if(it.base_model)tags+="<span class='lb-tag "+(incompatible?"warn":"base")+"'>"+esc(it.base_model)+"</span>";
    tags+=typeBadge;
    if(it.nsfw)tags+="<span class='lb-tag nsfw'>NSFW</span>";
    tags+="</div>";
    const btnLabel=loaded ? "\u2713 Loaded"
      : (isCkpt?"&#8635; Load checkpoint":"+ Load this LoRA");
    c.innerHTML=thumb+
      "<div class='lb-body'><div class='lb-name'>"+esc(it.name||"untitled")+"</div>"+tags+
      (warnNote?"<div class='hintline' style='margin:0;color:var(--red)'>"+esc(warnNote)+"</div>":"")+
      (!isCkpt&&it.triggers&&it.triggers.length
        ? "<div class='hintline' style='margin:0'>trigger: "+esc(it.triggers.join(", "))+"</div>" : "")+
      "</div>"+
      "<button class='lb-loadbtn'"+(loaded?" disabled":"")+">"+btnLabel+"</button>";
    const btn=c.querySelector(".lb-loadbtn");
    if(!loaded){
      const act=()=> isCkpt ? lbLoadCkpt(it,btn) : lbLoad(it,btn);
      btn.addEventListener("click",ev=>{ev.stopPropagation();act();});
    }
    // Clicking the card (not the load button) always opens the detail view,
    // even when loaded — you can still browse its samples.
    c.addEventListener("click",()=>openDetail(it));
    frag.appendChild(c);
  });
  grid.appendChild(frag);
}

async function lbLoad(it,btn,weight){
  if(btn){btn.disabled=true;btn.textContent="Loading...";}
  const scale=(weight==null?1.0:weight);
  try{
    const r=await fetch("/api/loras/add",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({url:it.download_url,name:it.name,scale:scale,
        triggers:it.triggers||[]})});
    const j=await r.json();
    if(j.ok){toast("LoRA loaded: "+(it.name||"")+" at weight "+scale.toFixed(2)+".");
      refreshLoras();refreshLoadedVersions();
      if(btn){btn.textContent="\u2713 Loaded";btn.disabled=true;}}
    else{toast("Couldn't load: "+(j.error||"unknown"),true);
      if(btn){btn.disabled=false;btn.textContent="+ Load this LoRA";}}
  }catch(e){toast("Error: "+e,true);
    if(btn){btn.disabled=false;btn.textContent="+ Load this LoRA";}}
}

async function lbLoadCkpt(it,btn){
  // A checkpoint is a multi-GB download + full pipeline swap, so confirm.
  const fam=_familyOf(it.base_model);
  let msg="Load \""+(it.name||"this checkpoint")+"\" as the base model?\n\n"
    +"This downloads the checkpoint (often 10\u201340 GB for these families) "
    +"and unloads the current model (there's a reload gap).";
  if(!fam)msg+="\n\nWARNING: its base model is \""+(it.base_model||"unknown")
    +"\", which is not Qwen / Flux.1 / Z-Image. It will fail to load.";
  else if(fam!==_activeArch)msg+="\n\nNote: this is a "+fam+" checkpoint; the studio will switch the whole pipeline to that family.";
  if(!(await confirmModal(msg,{title:"Load base model?",okText:"Load model"})))return;
  // Prefill the sidebar field for visibility, then trigger the swap. The
  // swap endpoint refuses if a job/another swap is active.
  $("modelUrl").value=it.download_url;
  $("modelName").value=it.name||"";
  closeLoraBrowser();
  toast("Starting checkpoint load \u2014 watch the console for download progress.");
  _startSwap(it.download_url,it.name||"",false,it.base_model||"");
}

function toggleMin(e,panelId){e.stopPropagation();
  const p=$(panelId||"jobsPanel");p.classList.toggle("min");
  const btn=p.querySelector(".float-min");
  if(btn)btn.innerHTML=p.classList.contains("min")?"\u25A1":"\u2013";}
function toggleSec(id){$(id).classList.toggle("closed");}
function _makeDraggable(panelId,handleId){
  const panel=$(panelId),handle=$(handleId);
  if(!panel||!handle)return;
  let drag=false,sx,sy,ox,oy;
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
}
_makeDraggable("jobsPanel","jobsHandle");
_makeDraggable("historyPanel","histHandle");

function toggleDock(){
  const d=$("dock");d.classList.toggle("collapsed");
  $("dockToggle").innerHTML=d.classList.contains("collapsed")?"\u25B4 Show":"\u25BE Hide";
}
function clearConsole(e){e.stopPropagation();_conLines=[];$("console").innerHTML="";}
async function copyConsole(e){
  e.stopPropagation();
  const text=_conLines.join("\n");
  if(!text){toast("Console is empty.");return;}
  try{
    await navigator.clipboard.writeText(text);
    toast("Console copied ("+_conLines.length+" lines).");
  }catch(err){
    // Clipboard API can be blocked in the Colab proxy iframe — fall back
    // to a hidden textarea + execCommand, which works without permissions.
    const ta=document.createElement("textarea");
    ta.value=text;ta.style.position="fixed";ta.style.opacity="0";
    document.body.appendChild(ta);ta.focus();ta.select();
    let ok=false;try{ok=document.execCommand("copy");}catch(_){}
    document.body.removeChild(ta);
    toast(ok?"Console copied ("+_conLines.length+" lines).":
      "Couldn't auto-copy \u2014 select the text and Ctrl/Cmd+C.",!ok);
  }
}

function _consoleHasSelection(){
  const sel=window.getSelection();
  if(!sel||sel.isCollapsed||!sel.rangeCount)return false;
  const box=$("console");
  return box && box.contains(sel.anchorNode) && box.contains(sel.focusNode);
}

async function pollHw(){
  try{const j=await(await fetch("/api/hw")).json();
    if(j.gpu){
      $("vramPill").textContent=j.vram_used+" / "+j.vram_total+" GB";
    }
    if(j.model_name){$("modelPill").textContent=j.model_name;
      $("modelNow").textContent=j.model_name;}
    const anyRun=queue.some(q=>q.status==="running");
    const res=j.residency&&j.residency!=="unknown"?j.residency:"";
    let cls,label;
    if(!j.gpu){cls="off";label="No GPU";}
    else if(j.swapping){cls="warm";label=j.swap_stage?("Swap: "+j.swap_stage):"Swapping model";}
    else if(anyRun){cls="warm";label="Generating";}
    else if(res==="gpu"){cls="on";label="GPU resident";}
    else if(res==="cpu-offload"){cls="on";label="CPU offload";}
    else{cls="cold";label="Connecting";}
    $("connDot").className="dot "+cls;$("connLabel").textContent=label;
  }catch(e){}
}

let _conSeen=0;
let _conLines=[];
async function pollConsole(){
  try{
    const j=await(await fetch("/api/console")).json();
    const box=$("console");if(!box)return;
    const lines=j.lines||[];
    _conLines=lines;                       // raw text for the Copy button
    // Don't re-render while the user is selecting text inside the console —
    // rebuilding innerHTML would wipe the in-progress selection.
    if(_consoleHasSelection())return;
    if(lines.length===_conSeen)return;     // nothing new; leave DOM alone
    // Preserve the user's scroll position: only snap to bottom if they were
    // already near it (reading live), not if they scrolled up to read back.
    const nearBottom=box.scrollHeight-box.scrollTop-box.clientHeight<40;
    box.innerHTML=lines.map(l=>{
      const s=l.replace(/&/g,"&amp;").replace(/</g,"&lt;");
      let cls="ln";
      if(l.indexOf("[diag]")>=0)cls="ln diag";
      if(l.indexOf("***")>=0||l.toLowerCase().indexOf("warning")>=0)cls="ln warn";
      return "<div class='"+cls+"'>"+s+"</div>";
    }).join("");
    if(nearBottom)box.scrollTop=box.scrollHeight;
    _conSeen=lines.length;
  }catch(e){}
}

/* ── base model swap ── */
let _swapPolling=false;
async function swapModel(){
  const url=$("modelUrl").value.trim();
  if(!url){toast("Paste a model URL or HF repo id first.",true);return;}
  await _startSwap(url,$("modelName").value.trim());
}
async function resetModel(){await _startSwap(null,null,true);}

async function _startSwap(url,name,reset,base){
  const sb=$("swapBtn"),rb=$("resetBtn");
  sb.disabled=true;rb.disabled=true;
  try{
    const ep=reset?"/api/model/reset":"/api/model/swap";
    const body=reset?{}:{url:url,name:name,base:base||""};
    const r=await fetch(ep,{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)});
    const j=await r.json();
    if(!j.ok){toast("Swap failed: "+(j.error||"unknown"),true);
      sb.disabled=false;rb.disabled=false;return;}
    toast(reset?"Resetting to Qwen-Image 2512...":"Loading model... watch the console.");
    sb.textContent="Loading...";
    pollSwap();
  }catch(e){toast("Error: "+e,true);sb.disabled=false;rb.disabled=false;}
}
async function pollSwap(){
  if(_swapPolling)return;_swapPolling=true;
  const sb=$("swapBtn"),rb=$("resetBtn");
  try{
    while(true){
      let j;try{j=await(await fetch("/api/model")).json();}catch(e){await _wait(800);continue;}
      if(j.swap&&j.swap.busy){
        const st=j.swap.stage||"working";
        sb.textContent=st.length>34?st.slice(0,34)+"\u2026":st;
        sb.title=st;
        await _wait(800);continue;}
      // settled
      if(j.swap&&j.swap.error){toast("Model load failed: "+j.swap.error,true);}
      else{
        const res=j.swap&&j.swap.result;
        let msg="Model loaded: "+(j.name||"new model");
        if(res&&res.dropped&&res.dropped.length)
          msg+=" \u2014 "+res.dropped.length+" LoRA(s) not compatible, dropped";
        toast(msg);
        refreshLoras();refreshLoadedVersions();refreshModelVid();
        _applyModelDefaults(j,true);
        if(!_comboRestoring)_saveCombo();
      }
      $("modelPill").textContent=j.name||"model";
      $("modelNow").textContent=j.name||"model";
      break;
    }
  }finally{
    sb.disabled=false;rb.disabled=false;sb.innerHTML="\u21BB Load this model";
    _swapPolling=false;
  }
}
function _wait(ms){return new Promise(r=>setTimeout(r,ms));}

document.addEventListener("keydown",e=>{
  if(e.key==="Escape"&&$("notifyScrim")&&$("notifyScrim").classList.contains("open")){
    _notifyResolve(false);return;}
  if(!$("lightbox").classList.contains("open"))return;
  if(e.key==="Escape")closeLightbox();
  else if(e.key==="ArrowLeft")_lbxNav(-1);
  else if(e.key==="ArrowRight")_lbxNav(1);
});
// Click the dark scrim (outside the panel) = cancel.
(function(){const s=$("notifyScrim");
  if(s)s.addEventListener("click",e=>{if(e.target===s)_notifyResolve(false);});})();

/* ── session persistence ──
   Settings (prompt, sliders, mode) are tiny text — kept in browser
   localStorage so they survive even a runtime restart. Generation HISTORY is
   large (PNG images) and now lives on the T4's disk, fetched from /api/history
   on load; this keeps the browser's ~5MB localStorage limit irrelevant to it. */
const _LS_KEY="sdxlstudio_state_v1";
const _SETTING_IDS=["prompt","neg","stepsV","guidV","widthV","heightV",
  "batchV","strengthV","seed","sizePreset","sampler"];
function _saveState(){
  try{
    const s={mode:currentMode,settings:{},checks:{}};
    _SETTING_IDS.forEach(id=>{const el=$(id);if(el)s.settings[id]=el.value;});
    ["randSeedEach"].forEach(id=>{const el=$(id);if(el)s.checks[id]=el.checked;});
    localStorage.setItem(_LS_KEY,JSON.stringify(s));
  }catch(e){}
}
// ── model + LoRA combo persistence ──
// Remembers the last base-model + LoRA stack and, on a fresh runtime (where the
// live setup is still just the default Qwen base with no LoRAs), replays it so the
// combo you last configured is restored automatically. Saved on every change.
const _CFG_KEY="sdxlstudio_combo_v1";
async function _saveCombo(){
  try{
    const j=await(await fetch("/api/config")).json();
    if(j&&j.ok){
      localStorage.setItem(_CFG_KEY,JSON.stringify(
        {model:j.model||null,loras:j.loras||[],ts:Date.now()}));  // model includes .base
    }
  }catch(e){}
}
let _comboRestoring=false;
async function _restoreCombo(){
  let cfg;
  try{cfg=JSON.parse(localStorage.getItem(_CFG_KEY)||"null");}catch(e){cfg=null;}
  if(!cfg)return;
  // Compare against the LIVE setup; only replay what's missing (fresh runtime).
  let live;
  try{live=await(await fetch("/api/config")).json();}catch(e){return;}
  if(!live||!live.ok)return;
  const liveModel=(live.model&&live.model.url)||"";
  const wantModel=(cfg.model&&cfg.model.url)||"";
  const liveLoras=(live.loras||[]).map(l=>l.url).filter(Boolean);
  const wantLoras=(cfg.loras||[]).filter(l=>l.url);
  const modelMatches=(!wantModel)||(liveModel===wantModel);
  const lorasMatch=wantLoras.every(l=>liveLoras.includes(l.url));
  if(modelMatches&&lorasMatch)return;        // already as configured — nothing to do
  if(_comboRestoring)return;_comboRestoring=true;
  try{
    // 1) base model (skip if it's already the right one or there's no saved url)
    if(wantModel && liveModel!==wantModel){
      toast("Restoring your last model\u2026");
      await fetch("/api/model/swap",{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({url:wantModel,name:(cfg.model&&cfg.model.name)||"",
                             base:(cfg.model&&cfg.model.base)||""})});
      // wait for the swap to settle before adding LoRAs (swap rebuilds the unet)
      await _waitSwapIdle();
    }
    // 2) LoRA stack — add any saved LoRA not already present
    for(const L of wantLoras){
      if(liveLoras.includes(L.url))continue;
      try{
        await fetch("/api/loras/add",{method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({name:L.name,url:L.url,scale:L.scale,
                               triggers:L.triggers||""})});
      }catch(e){}
    }
    refreshLoras();refreshLoadedVersions&&refreshLoadedVersions();refreshModelVid();
    toast("Restored your last model + LoRA setup.");
  }catch(e){}
  _comboRestoring=false;
}
async function _waitSwapIdle(){
  for(let i=0;i<120;i++){           // up to ~100s
    let j;try{j=await(await fetch("/api/model")).json();}catch(e){await _wait(900);continue;}
    if(!(j.swap&&j.swap.busy))return;
    await _wait(900);
  }
}
function _restoreState(){
  let s;
  try{s=JSON.parse(localStorage.getItem(_LS_KEY)||"null");}catch(e){s=null;}
  if(!s)return;
  try{
    if(s.settings){
      _SETTING_IDS.forEach(id=>{
        if(s.settings[id]!=null){const el=$(id);if(el)el.value=s.settings[id];}
      });
      [["steps","stepsV"],["guid","guidV"],["width","widthV"],
       ["height","heightV"],["batch","batchV"],["strength","strengthV"]]
        .forEach(([sl,v])=>{const a=$(sl),b=$(v);if(a&&b)a.value=b.value;});
    }
    if(s.checks){
      ["randSeedEach"].forEach(id=>{
        const el=$(id);if(el&&s.checks[id]!=null)el.checked=s.checks[id];
      });
    }
    if(s.mode)setMode(s.mode);
  }catch(e){}
}
// Pull the persisted history (stored on the T4 disk) and rebuild the gallery.
async function loadServerHistory(){
  try{
    const j=await(await fetch("/api/history")).json();
    if(!Array.isArray(j.history))return;
    history=j.history.map(e=>{
      const urls=(e.files||[]).map(f=>"/api/history/image/"+encodeURIComponent(f));
      return {id:e.id,prompt:e.prompt,thumb:urls[0],urls:urls,ts:e.ts,
              settings:e.settings||null,loras:e.loras||[]};
    });
    renderHistory();
  }catch(e){}
}
// Persist settings whenever a tracked input changes.
_SETTING_IDS.forEach(id=>{
  const el=$(id);
  if(el){el.addEventListener("change",_saveState);
         el.addEventListener("input",_saveState);}
});

// ---- MissingLink sign-in gate ------------------------------------------
const ML_BASE="https://missinglink.build";
let _mlPopup=null;
function _loginShow(on){const o=$("loginOverlay");if(o)o.classList.toggle("hidden",!on);}
function _loginErr(msg){const e=$("loginErr");if(!e)return;
  e.style.display=msg?"block":"none";e.textContent=msg||"";}
function _acctBadge(j){
  const el=$("acctPill");if(!el)return;
  const who=j.email||"API key";
  const tier=j.member?"PRO":(j.remaining!=null?j.remaining+" free left":"trial");
  el.innerHTML=esc(who)+" &middot; <b style='color:var(--gold)'>"+esc(String(tier))+"</b>";
  $("acctBadge").style.display="flex";
}
async function mlCheckAuth(){
  try{
    const j=await(await fetch("/api/auth/status")).json();
    if(j.authed){_loginShow(false);_acctBadge(j);}
    else{_loginShow(true);}
  }catch(e){_loginShow(true);}
}
async function _mlValidate(token,btn){
  _loginErr("");let old=null;
  if(btn){btn.disabled=true;old=btn.textContent;btn.textContent="Checking\u2026";}
  try{
    const r=await fetch("/api/auth/validate",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({token:token})});
    const j=await r.json();
    if(r.ok&&j.ok){_loginShow(false);_acctBadge(j);return true;}
    if(j.error==="membership_required"){
      const t=$("loginTrialGo");
      if(t){t.href=j.checkout_url||(ML_BASE+"/create-checkout-session");
            t.style.display="block";}
      _loginErr((j.email?("Signed in as "+j.email+" \u2014 "):"")
        +"this Google account has no active membership. Start the 7-day free trial below, then press Continue with Google again.");
      return false;
    }
    _loginErr(j.error||"sign-in failed \u2014 try again");
  }catch(e){_loginErr("could not reach the studio server: "+e);}
  finally{if(btn){btn.disabled=false;btn.textContent=old;}}
  return false;
}
function mlGoogle(){
  _loginErr("");
  $("loginPaste").style.display="block";
  const w=520,h=680,x=Math.max(0,(screen.width-w)/2),y=Math.max(0,(screen.height-h)/2);
  _mlPopup=window.open(ML_BASE+"/notebook-signin","ml_signin",
    "width="+w+",height="+h+",left="+x+",top="+y);
  if(!_mlPopup)
    _loginErr("Popup blocked \u2014 open "+ML_BASE+"/notebook-signin in a new tab, sign in with Google, then paste the code above.");
}
window.addEventListener("message",ev=>{
  const d=ev.data;
  if(d&&d.type==="missinglink-auth"&&d.token){
    try{if(_mlPopup)_mlPopup.close();}catch(e){}
    _mlValidate(d.token,null);
  }
});
function mlUseCode(){const v=$("loginCode").value.trim();
  if(v)_mlValidate(v,$("loginCodeBtn"));
  else _loginErr("paste the sign-in code from the Google tab first");}
function mlUseKey(){const v=$("loginKey").value.trim();
  if(v)_mlValidate(v,$("loginUnlock"));
  else _loginErr("paste your MissingLink API key first");}
async function mlLogout(){
  try{await fetch("/api/auth/logout",{method:"POST"});}catch(e){}
  $("acctBadge").style.display="none";_loginShow(true);
}
mlCheckAuth();

setMode("txt2img");
_restoreState();
loadServerHistory();
refreshModelVid();
// Clicking the active model name opens its Civitai card (if it's a Civitai model).
(function(){
  const el=$("modelNow");
  if(el)el.addEventListener("click",()=>{
    if(_activeModelVid)openCardByVersion(_activeModelVid,"Checkpoint",el.textContent);
  });
})();
setInterval(pollHw,3000);pollHw();
setInterval(pollConsole,1200);pollConsole();
refreshLoras();renderQueue();
// Restore the last model+LoRA combo on a fresh runtime (after a short delay so
// the initial model/lora state has loaded; _restoreCombo no-ops if already set).
setTimeout(_restoreCombo,2500);
</script></body></html>
"""

# ---- launch ------------------------------------------------------------
print("=" * 60)
print("  Preparing DiT Studio (default: Qwen-Image 2512).")
print("=" * 60)
if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info()
    _tot_gb = _total / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({_tot_gb:.0f} GB)")
    if _tot_gb >= 70:
        print("  -> 80 GB-class GPU: FULL VRAM residency for all three "
              "families (Qwen ~58 GB, FLUX ~34 GB, Z-Image ~20 GB in bf16).")
    else:
        print("  -> 40 GB-class GPU detected: big transformers load as "
              "community GGUF Q4_K quants (sensitive layers kept high "
              "precision) and the whole pipeline stays resident on the GPU "
              "(Qwen: ~13 GB GGUF transformer + bf16 text encoder ≈ 30 GB).")

# Resolve the startup LoRAs against the Civitai API BEFORE the model load:
# the first one's base-model tag decides which family boots.
_startup = []                 # [(url, name, scale, triggers, family)]
for _entry in STARTUP_LORAS:
    if isinstance(_entry, str):
        _entry = {"url": _entry}
    _u = (_entry.get("url") or "").strip()
    if not _u:
        continue
    _vid = _version_id_in_url(_u)
    _info = _civitai_version_info(_vid) if _vid else {}
    _nm = _entry.get("name") or _info.get("name") or (f"lora_{_vid}" if _vid
                                                      else "startup_lora")
    _fam = _detect_arch(_info.get("base_model"))
    _trg = _entry.get("triggers") or _info.get("triggers") or []
    if _info:
        print(f"  startup LoRA: {_nm}  [base: {_info.get('base_model')}"
              f" -> {_fam or 'unknown family'}]")
    else:
        print(f"  startup LoRA: {_nm} (Civitai lookup failed — will assume "
              "it fits the default model)")
    _startup.append((_u, _nm, float(_entry.get("scale", 1.0)), _trg, _fam))

_boot_key = DEFAULT_MODEL_KEY
for _u, _nm, _sc, _trg, _fam in _startup:
    if _fam in MODEL_REGISTRY:
        _boot_key = _fam
        break
_boot = MODEL_REGISTRY[_boot_key]
print(f"Loading {_boot['label']} (first run downloads the weights — "
      "this takes a while)...")

build_pipeline(model_ref=_boot["repo"], model_name=_boot["label"],
               arch=_boot_key)
print("Model ready.")

# Attach the startup LoRAs to the freshly loaded pipeline.
for _u, _nm, _sc, _trg, _fam in _startup:
    if _fam is not None and _fam != _boot_key:
        print(f"  startup LoRA '{_nm}' skipped — it's for {_fam}, but "
              f"{_boot_key} is loaded.")
        continue
    print(f"  attaching startup LoRA '{_nm}' (strength {_sc})...")
    try:
        _ok, _msg = register_lora(_nm, _u, _sc, triggers=_trg)
        print(f"  startup LoRA '{_nm}': {'ready' if _ok else 'FAILED — ' + str(_msg)}")
    except Exception as _e:
        print(f"  startup LoRA '{_nm}' failed: {_e}")
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
            🔗 DiT Studio is live:</div>
          <a href="{_url}" target="_blank" style="color:#f4b740;font-size:18px;
             font-weight:bold;">{_url}</a></div>"""))
    else:
        from google.colab import output as _co
        _co.serve_kernel_port_as_window(
            PORT, anchor_text="🔗 Click to open DiT Studio")
else:
    print(f"\nDiT Studio running at http://localhost:{PORT}\n")

print("=" * 60)
print("  DiT Studio running. Keep this cell's runtime alive.")
print("=" * 60)