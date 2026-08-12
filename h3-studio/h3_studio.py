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
#  Runtime: A100 40GB + High-RAM.
#
#  LICENSE: MiniMax H3 Community License, territory clause.
#           https://platform.minimax.io/h3-license
# ============================================================================

COMFY_DIR = "/content/ComfyUI"
UI_PORT   = 7860

# Which FL2VA transformer to fetch. All are stock-ComfyUI format and load with
# the built-in loader; the text encoder and both VAEs are shared and unchanged.
# NVFP4 profiles target Blackwell (RTX 50 / B200) — on Ampere or Ada use INT8.
#
# DEFAULT is "base": the Comfy-Org pruned int8 reference paired with the
# Lightning (lightx2v) 4-step turbo LoRA below — the fast path, and the only
# thing downloaded out of the box. The 10Eros_Max quants are NOT fetched
# unless explicitly selected (they are 23 GB+ each and their upstream has
# been throttling to zero). Override without editing this file by setting
# the H3_DIT env var in the notebook before launch, e.g.
#   os.environ["H3_DIT"] = "eros_int8_hq"
import os as _os  # the main import block sits below this config section
DIT_CHOICE = _os.environ.get("H3_DIT", "base")

DITS = {
  # Comfy-Org reference, pruned + int8 convrot
  "base":        ("Comfy-Org/MiniMax-H3", "diffusion_models",
                  "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
  # 10Eros_Max fine-tune (QKV blocks 0-31), quantized by DmitryDB.
  # Served from the disguisequence mirror — a byte-identical duplicate of
  # DmitryDB/MiniMax-H3-10Eros-Max-Quants (same FL2VA/ layout) — because
  # downloads from the DmitryDB origin were throttling to ZERO bytes/sec.
  "eros_int8_hq":("disguisequence/MiniMax-H3-10Eros-Max-Quants", "FL2VA",
                  "10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors"),
  "eros_int8":   ("disguisequence/MiniMax-H3-10Eros-Max-Quants", "FL2VA",
                  "10Eros_Max_H3_FL2VA-INT8-ConvRot.safetensors"),
  "eros_nvfp4_hq":("disguisequence/MiniMax-H3-10Eros-Max-Quants", "FL2VA",
                  "10Eros_Max_H3_FL2VA-NVFP4-HQ.safetensors"),
  "eros_nvfp4":  ("disguisequence/MiniMax-H3-10Eros-Max-Quants", "FL2VA",
                  "10Eros_Max_H3_FL2VA-NVFP4.safetensors"),
}

# 4-step Turbo LoRA (larryvrh, converted for the pruned checkpoint by drbaph).
# ~5x fewer sampling steps. Early preview: under-trained, and audio is its
# weak point. Use steps 4-8, scheduler "beta", strength ~1.0.
TURBO_LORA = True

# cloudflared is off by default — Cloudflare's quick-tunnel API has been
# refusing registrations. Colab's own iframe/window transport is used instead.
TUNNEL_FALLBACK = False

# Memory. In library mode Comfy's manager is unconfigured and tries to keep the
# whole 33B DiT resident, which OOMs regardless of card size. reserve_vram is
# headroom left free for activations and the cast buffers; raise it if sampling
# OOMs. LOWVRAM forces block-by-block streaming — slower, but survives anything.
RESERVE_VRAM = 8.0
LOWVRAM      = True
# ───────────────────────────────────────────────────────────────────────────

import os, re, sys, gc, time, json, uuid, stat, shutil, socket, asyncio
import threading, subprocess, traceback, urllib.request

# Must be set BEFORE torch initialises CUDA. The DiT/VAE handoff fragments the
# allocator badly; expandable segments let freed blocks be reused across sizes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try: sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception: pass
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
from PIL import Image
from flask import Flask, request, jsonify, Response, send_file

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device. Runtime -> Change runtime type -> GPU.")
gpu  = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
disk = shutil.disk_usage("/content").free / 1e9
log(f"  {gpu} · {vram:.0f} GB VRAM · {disk:.0f} GB disk free")

# ── 1. Weights ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
if DIT_CHOICE not in DITS:
    raise RuntimeError(f"DIT_CHOICE must be one of {list(DITS)}")
_repo, _sub, DIT_FILE = DITS[DIT_CHOICE]

if "nvfp4" in DIT_CHOICE and not any(k in gpu for k in ("B200","RTX 50","GB200","RTX 60")):
    log(f"  ⚠ {DIT_CHOICE} is a Blackwell profile and {gpu} is not Blackwell.\n"
        f"    Expect a slow emulated path or a load failure — use eros_int8 instead.")

MODELS = os.path.join(COMFY_DIR, "models")
FILES = [("diffusion_models", DIT_FILE, _repo, _sub),
         ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
          "Comfy-Org/MiniMax-H3", "text_encoders"),
         ("vae", "minimax_h3_video_vae_fp16.safetensors",
          "Comfy-Org/MiniMax-H3", "vae"),
         ("vae", "minimax_h3_audio_vae_fp32.safetensors",
          "Comfy-Org/MiniMax-H3", "vae")]
# The four weight files total ~50 GB and used to download SEQUENTIALLY —
# the 27 GB text encoder only started after the 21 GB DiT finished, so the
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

if TURBO_LORA:
    # Standard-format LoRAs only — backbone weights that stock LoraLoaderModelOnly
    # can apply. The "pruned complete" Turbo conversions carry adaln_t_table and
    # per-block AdaLN projections, which need PR #15353 or ComfyUI-MiniMaxH3;
    # they will fail to load here, so they are deliberately not fetched.
    ldir = os.path.join(MODELS, "loras"); os.makedirs(ldir, exist_ok=True)
    WANT = [
        ("fal/MiniMax-H3-Realism-People-LoRA", "h3-realism-people-t2v.safetensors"),
        ("Kijai/MiniMax-H3_comfy",
         "loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"),
    ]
    for repo, fn in WANT:
        base = os.path.basename(fn)
        if os.path.exists(os.path.join(ldir, base)): 
            log(f"  ✓ {base}"); continue
        try:
            log(f"  ↓ lora: {base}")
            p = hf_hub_download(repo, filename=fn)
            shutil.copy(p, os.path.join(ldir, base))
        except Exception as e:
            log(f"  ⚠ {base} failed: {e}")

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
# NOTE: smart memory is what lets Comfy run this model on a 12 GB card by
# offloading dynamically. Disabling it (which I did earlier) defeats exactly
# the mechanism we need, so it stays on.
if LOWVRAM:
    args.lowvram = True

import nodes, folder_paths, comfy.utils
import comfy.model_management as mm
log(f"  vram state: {mm.vram_state}  · reserve {RESERVE_VRAM} GB")

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
# ComfyUI calls this as hook(current, total, preview, node_id=...) — the extra
# keyword is not optional to accept, so swallow anything else it adds later.
comfy.utils.set_progress_bar_global_hook(
    lambda cur, total, preview=None, **kw: PROG.update(cur=cur, total=total))

# ── 4. Model cache ─────────────────────────────────────────────────────────
CACHE = {}
def get_models(weight_dtype, lora, lora_strength, unet=None):
    """Base weights are cached by (transformer, dtype). The LoRA is applied to a
    clone on every job, so switching or restrengthening a LoRA costs nothing —
    it used to invalidate the cache and reload all 44 GB including the encoder.
    Switching the transformer does reload it, but not the encoder or VAEs."""
    unet = unet or DIT_FILE
    if CACHE.get("key") != (unet, weight_dtype):
        CACHE.clear()
        PROG["stage"] = "loading unet"
        log(f"  loading transformer: {unet}")
        base, = call("UNETLoader", unet_name=unet, weight_dtype=weight_dtype)
        PROG["stage"] = "loading clip"
        clip, = call("CLIPLoader",
                     clip_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                     type="minimax")
        PROG["stage"] = "loading vae"
        vae,  = call("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
        avae, = call("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
        CACHE.update(key=(unet, weight_dtype), base=base, clip=clip,
                     vae=vae, avae=avae)

    model = CACHE["base"]
    if lora and lora != "none":
        PROG["stage"] = "applying lora"
        before = len(getattr(CACHE["base"], "patches", {}) or {})
        model, = call("LoraLoaderModelOnly", model=model, lora_name=lora,
                      strength_model=float(lora_strength))
        after = len(getattr(model, "patches", {}) or {})
        # A LoRA whose keys don't match the model applies zero patches and
        # Comfy only warns — which looks exactly like a working LoRA that
        # does nothing. Treat that as an error instead.
        if after <= before:
            raise RuntimeError(
                f"LoRA '{lora}' applied 0 patches — its keys do not match this "
                f"model. The 'pruned complete' Turbo conversions need PR #15353 "
                f"or the ComfyUI-MiniMaxH3 node; use a standard-format LoRA "
                f"(e.g. h3-realism-people, or Kijai's resized lightx2v turbo).")
        log(f"  lora '{lora}' applied {after-before} patches @ {lora_strength}")
    return model, CACHE["clip"], CACHE["vae"], CACHE["avae"]

def to_tensor(pil):
    a = np.array(Image.open(pil).convert("RGB")).astype(np.float32)/255.0
    return torch.from_numpy(a)[None,]

FPS = 24.0

def frames_from_seconds(sec):
    """Seconds -> frame count on the model's 17k+5 grid (the only counts the
    video VAE can decode). Returns (frames, actual_seconds)."""
    want = float(sec) * FPS
    k = max(0, round((want - 5) / 17))
    frames = 17 * k + 5
    return frames, frames / FPS

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
        # Comfy's server wraps execution this way; library mode does not, which
        # is why decoded tensors came back carrying a grad graph.
        with torch.inference_mode():
            _generate(jid, p)
    finally:
        GPU_LOCK.release()

def _generate(jid, p):
    j = JOBS[jid]
    try:
        j.update(status="running")

        # Whatever a previous failed run left resident, drop it before loading.
        try:
            mm.unload_all_models(); gc.collect(); mm.soft_empty_cache()
            torch.cuda.empty_cache()
        except Exception: pass
        f0 = torch.cuda.mem_get_info()[0]/1e9
        log(f"  job {jid} start: {f0:.1f} GB free")
        j["memlog"] = f"VRAM at job start: {f0:.1f} GB free\n"

        model, clip, vae, avae = get_models(
            p["weight_dtype"], p.get("lora"), p.get("lora_strength", 1.0),
            unet=p.get("unet") or DIT_FILE)

        model, = call("MiniMaxH3SigmaShift", model=model,
                      shift_video=float(p["shift_video"]),
                      shift_audio=float(p["shift_audio"]))

        PROG["stage"] = "conditioning"
        n_frames, actual_sec = frames_from_seconds(p["duration"])
        log(f"  {p['duration']}s requested -> {n_frames} frames ({actual_sec:.2f}s)")
        kw = dict(clip=clip, vae=vae, prompt=p["prompt"],
                  width=int(p["width"]), height=int(p["height"]),
                  length=n_frames)
        if p.get("first_frame"): kw["first_frame"] = to_tensor(p["first_frame"])
        if p.get("last_frame"):  kw["last_frame"]  = to_tensor(p["last_frame"])
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

        video, = call("CreateVideo", images=images, fps=24.0, audio=audio)

        PROG["stage"] = "saving"
        # Write the container directly. SaveVideo's format/codec are structured
        # V3 values that cannot be synthesised from the schema, and the video
        # object here is already complete.
        dest = os.path.join(OUT, f"{jid}.mp4")
        video.save_to(dest)
        if not os.path.exists(dest):
            raise RuntimeError("save_to produced no file")

        j.update(status="done", file=os.path.basename(dest),
                 secs=round(time.time()-j["t0"],1), frames=n_frames,
                 duration=round(actual_sec, 2))
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
                   unets=folder_paths.get_filename_list("diffusion_models"),
                   unet_default=DIT_FILE)

@app.get("/api/keepalive")
def keepalive(): return jsonify(ok=True)

@app.post("/api/generate")
def api_gen():
    if not ML_OK:
        return jsonify(error="MissingLink token not validated."), 402
    jid = uuid.uuid4().hex[:8]
    p = {k: request.form.get(k) for k in
         ("prompt","width","height","duration","steps","seed","denoise",
          "shift_video","shift_audio","sampler_name","scheduler",
          "weight_dtype","lora","lora_strength","unet")}
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

@app.get("/out/<path:f>")
def out(f): return send_file(os.path.join(OUT,f))

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MissingLink MiniMax Studio</title>
<link rel="icon" href="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2">
<style>
*{box-sizing:border-box}
body{margin:0;background:#0a0a0b;color:#e9e9ec;font:13px/1.55 ui-monospace,Menlo,monospace}
.wrap{display:flex;min-height:100vh}
.side{width:430px;flex:0 0 430px;padding:24px;border-right:1px solid #1e1e22;
 height:100vh;overflow-y:auto}
.side::-webkit-scrollbar{width:8px}
.side::-webkit-scrollbar-thumb{background:#26262b;border-radius:4px}
.main{flex:1;padding:24px;display:flex;flex-direction:column;gap:16px}
h1{margin:0 0 24px;display:flex;align-items:center;gap:12px;
 font-size:15px;font-weight:700;letter-spacing:2.5px}
h1 .wm{white-space:nowrap}
h1 .ml{color:#8A8A8A}
h1 .st{color:#E8A917}
#logo{height:32px;width:auto;flex:0 0 auto;display:block}
h2{font-size:10px;color:#6a6a72;margin:24px 0 6px;letter-spacing:1.2px;
 border-bottom:1px solid #1c1c20;padding-bottom:6px}
label{display:block;font-size:10.5px;color:#8b8b93;margin:12px 0 5px}
input,textarea,select{width:100%;background:#141416;border:1px solid #26262b;
 color:#e9e9ec;padding:9px 11px;border-radius:6px;font:inherit;font-size:12.5px;outline:none}
input:focus,textarea:focus,select:focus{border-color:#E8A917}
textarea{min-height:170px;resize:vertical;line-height:1.6}
input[type=file]{padding:7px;font-size:11px}
input[type=file]::file-selector-button{background:#26262b;color:#ccc;border:0;
 padding:6px 12px;border-radius:5px;margin-right:10px;font:inherit;cursor:pointer}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:0 12px}
button{width:100%;margin-top:24px;padding:13px;background:#E8A917;color:#0d0d0e;
 border:0;border-radius:7px;font:inherit;font-weight:700;cursor:pointer;font-size:13px}
button:disabled{background:#2a2a2e;color:#6a6a72;cursor:not-allowed}
.hint{font-size:10.5px;color:#5c5c64;margin-top:5px;line-height:1.5}
#status{padding:13px 16px;border:1px solid #1e1e22;border-radius:8px;background:#101012}
.row{display:flex;align-items:center;gap:9px;font-size:12.5px}
.dot{width:7px;height:7px;border-radius:50%;background:#5fd68a;flex:0 0 7px}
.dot.live{background:#E8A917;animation:p 1.3s infinite}
.dot.err{background:#ff6b6b}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
.bar{height:3px;background:#1e1e22;border-radius:2px;margin-top:9px;overflow:hidden}
.bar i{display:block;height:100%;background:#E8A917;width:0;transition:width .3s}
#vwrap{flex:1;border:1px solid #1e1e22;border-radius:10px;background:#0e0e10;
 display:flex;align-items:center;justify-content:center;min-height:340px;overflow:hidden}
#empty{color:#3f3f47;text-align:center;font-size:12px}
video{width:100%;max-height:74vh;display:block;background:#000}
#err{display:none;white-space:pre-wrap;color:#ff8a8a;font-size:10.5px;
 max-height:280px;overflow:auto;border:1px solid #3a2020;background:#160e0e;
 padding:12px;border-radius:7px}
</style></head><body><div class=wrap>
<div class=side>
<h1><img id=logo src="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink-Extras/main/image-edit-studio/static/app_logo.png?v=2"
 alt="MissingLink" onerror="this.style.display='none'"><span class=wm><span
 class=ml>MISSINGLINK</span> <span class=st>MINIMAX STUDIO</span></span></h1>

<h2>prompt</h2>
<textarea id=prompt placeholder="[Shot 1] Cinematic medium shot, slow push in…

overall_soundscape: …

non_diegetic_music: …"></textarea>
<div class=hint>H3 expects long structured prompts: the shot, then the
soundscape, then the music.</div>

<h2>keyframes</h2>
<label>first frame</label><input type=file id=first_frame accept="image/*">
<label>last frame (optional)</label><input type=file id=last_frame accept="image/*">
<div class=hint>Both optional — leave empty for text-to-video.</div>

<h2>geometry</h2>
<div class=g2>
<div><label>width</label><input id=width type=number value=640 step=32 min=32></div>
<div><label>height</label><input id=height type=number value=384 step=32 min=32></div>
</div>
<label>duration (seconds)</label>
<input id=duration type=number value=5 step=0.5 min=0.2>
<div class=hint id=durhint></div>

<h2>sampling</h2>
<div class=g2>
<div><label>steps</label><input id=steps type=number value=30 min=1></div>
<div><label>denoise</label><input id=denoise type=number value=1.0 step=0.01 min=0 max=1></div>
</div>
<label>sampler</label><select id=sampler_name></select>
<label>scheduler</label><select id=scheduler></select>
<label>seed</label>
<div style="display:flex;gap:10px"><input id=seed type=number value=42>
<button id=rnd style="margin:0;flex:0 0 72px;background:#26262b;color:#aaa">rand</button></div>
<div class=hint>No CFG — guidance is distilled into the weights.</div>

<h2>sigma shift</h2>
<div class=g2>
<div><label>video</label><input id=shift_video type=number value=12.0 step=0.01></div>
<div><label>audio</label><input id=shift_audio type=number value=3.0 step=0.01></div>
</div>
<div class=hint>Separate schedules for the video and audio latents.
12.0 / 3.0 are the released defaults.</div>

<h2>model</h2>
<label>transformer</label>
<select id=unet></select>
<div class=hint>Any FL2VA checkpoint in models/diffusion_models. Switching
reloads the transformer (~21 GB) but not the text encoder or VAEs.</div>
<label>weight dtype</label><select id=weight_dtype>
<option>default</option><option>fp8_e4m3fn</option>
<option>fp8_e4m3fn_fast</option><option>fp8_e5m2</option></select>
<label>lora</label>
<div style="display:flex;gap:10px">
<select id=lora></select>
<button id=refresh style="margin:0;flex:0 0 64px;background:#26262b;color:#aaa">↻</button>
</div>
<label>lora strength</label><input id=lora_strength type=number value=1.0 step=0.01>
<div class=hint><b>h3-realism-people</b> — photoreal humans. Put
<code>r34l1sm</code> at the START of your prompt or it does nothing.
Strength 1.0, or 0.6–0.8 for a lighter touch.<br>
<b>lightx2v turbo</b> — speed. 6–10 steps at 0.6–0.8 strength.<br>
↻ rescans models/loras without restarting.</div>
<button id=turbo style="margin-top:12px;background:#26262b;color:#E8A917">
◆ TURBO PRESET — 8 steps</button>
<div class=hint>Selects the turbo LoRA at 0.7 and sets 8 steps / beta / euler.</div>

<button id=go>GENERATE</button>
</div>

<div class=main>
<div id=status><div class=row><span class=dot></span><span id=stxt>ready</span></div>
<div class=bar><i id=pb></i></div></div>
<div id=err></div>
<div id=vwrap><div id=empty>no render yet</div></div>
<div id=meta class=hint></div>
</div></div>
<script>
const $=i=>document.getElementById(i);
let job=null;
$('rnd').onclick=e=>{e.preventDefault();$('seed').value=Math.floor(Math.random()*1e9)};

const dot=k=>document.querySelector('.dot').className='dot '+(k||'');
const say=t=>$('stxt').textContent=t;

// Mirror the server's 17k+5 snapping so the real duration is visible up front.
function updateDur(){
  const want=parseFloat($('duration').value||0)*24;
  const k=Math.max(0,Math.round((want-5)/17));
  const f=17*k+5, sec=f/24;
  const trained=f>=124&&f<=362;
  $('durhint').innerHTML=
    `→ ${f} frames = <b>${sec.toFixed(2)}s</b> at 24fps (snapped to the VAE's 17k+5 grid)`+
    (trained?'':`<br><span style="color:#c9a227">outside the trained 124–362 frame `+
      `range (5.2–15.1s) — quality will suffer</span>`);
}
$('duration').addEventListener('input',updateDur);
updateDur();

function loadMeta(){
  return fetch('/api/meta').then(r=>r.json()).then(m=>{
    const keep=$('lora').value;
    $('sampler_name').innerHTML=m.samplers.map(s=>
      `<option${s==='euler'?' selected':''}>${s}</option>`).join('');
    $('scheduler').innerHTML=m.schedulers.map(s=>
      `<option${s==='simple'?' selected':''}>${s}</option>`).join('');
    $('lora').innerHTML=m.loras.map(s=>`<option>${s}</option>`).join('');
    if(keep&&m.loras.includes(keep))$('lora').value=keep;
    const ku=$('unet').value;
    $('unet').innerHTML=(m.unets||[]).map(s=>
      `<option${s===m.unet_default?' selected':''}>${s}</option>`).join('');
    if(ku&&(m.unets||[]).includes(ku))$('unet').value=ku;
    return m;
  });
}
loadMeta();
$('refresh').onclick=e=>{e.preventDefault();loadMeta().then(m=>
  say(`${m.loras.length-1} lora(s) found`))};

$('turbo').onclick=e=>{
  e.preventDefault();
  const o=[...$('lora').options].find(x=>/lightx2v|turbo/i.test(x.value));
  if(!o){alert('No turbo LoRA in models/loras — press ↻, or check the startup log.');return}
  $('lora').value=o.value;
  $('lora_strength').value=0.7;
  $('steps').value=8;
  $('scheduler').value='beta';
  $('sampler_name').value='euler';
  say('turbo preset applied');
};


function fail(m){dot('err');say('failed');$('err').style.display='block';
  $('err').textContent=m;$('go').disabled=false;$('pb').style.width='0'}

$('go').onclick=async()=>{
  $('err').style.display='none';
  const fd=new FormData();
  for(const k of ['prompt','width','height','duration','steps','seed','denoise',
    'shift_video','shift_audio','sampler_name','scheduler','weight_dtype',
    'lora','lora_strength','unet'])fd.append(k,$(k).value);
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
    dot('');say(`done in ${j.secs}s · ${j.duration}s / ${j.frames} frames`+
      (j.vram_free!==undefined?` · ${j.vram_free} GB free at decode`:''));
    $('pb').style.width='100%';
    $('vwrap').innerHTML=`<video controls autoplay src="/out/${j.file}"></video>`;
    $('meta').textContent=j.file;$('go').disabled=false;return}
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
log("  Weights load on the first GENERATE, not now — so the first render")
log("  includes load time and later ones reuse the cache.")
log("  Errors come back as a full traceback in the red panel.")
log("="*74)
