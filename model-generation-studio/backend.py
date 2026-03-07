# ============================================================
# 🔺 TRELLIS.2 — Flask Backend (backend.py)
# Pipeline loading, rendering, job management, API routes.
# ============================================================

import os, sys, pathlib, subprocess, re, time, threading, traceback, json, uuid, collections, shutil, gc, math

os.environ["TRELLIS2_DISABLE_REMBG"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask"])
    from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

IN_COLAB = False
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    eval_js = None


# ══════════════════════════════════════════════════════════════
# CONSOLE CAPTURE
# ══════════════════════════════════════════════════════════════

class TeeWriter:
    def __init__(self, original, buf):
        self._original = original
        self._buf = buf

    def write(self, s):
        self._original.write(s)
        if s.strip():
            self._buf.append(s.rstrip('\n'))
        return len(s)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)

console_lines = collections.deque(maxlen=800)
sys.stdout = TeeWriter(sys.__stdout__, console_lines)
sys.stderr = TeeWriter(sys.__stderr__, console_lines)


# ══════════════════════════════════════════════════════════════
# WEIGHT CACHING
# ══════════════════════════════════════════════════════════════

DRIVE_WEIGHTS = pathlib.Path("/content/drive/MyDrive/trellis2_weights_local")
LOCAL_WEIGHTS = pathlib.Path("/content/trellis2_weights_local")
HF_MODEL_ID = "microsoft/TRELLIS.2-4B"


def dir_size_bytes(p):
    total = 0
    for f in pathlib.Path(p).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def copy_weights(src, dst, label=""):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    total = dir_size_bytes(src)
    copied = 0
    file_count = sum(1 for f in src.rglob("*") if f.is_file())
    print(f"  Copying {file_count} files ({total / 1e9:.1f} GB) {label}...")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))
            copied += item.stat().st_size
            pct = int(copied / total * 100) if total else 100
            sys.__stdout__.write(f"\r  {pct}% ({copied / 1e9:.1f} / {total / 1e9:.1f} GB)   ")
            sys.__stdout__.flush()
    sys.__stdout__.write("\n")
    sys.__stdout__.flush()
    print(f"  ✅ Copy complete.")


def resolve_weights():
    if LOCAL_WEIGHTS.exists() and any(LOCAL_WEIGHTS.iterdir()):
        print(f"✅ Local weights found at {LOCAL_WEIGHTS}")
        return str(LOCAL_WEIGHTS)
    if DRIVE_WEIGHTS.exists() and any(DRIVE_WEIGHTS.iterdir()):
        print(f"📂 Found cached weights on Drive: {DRIVE_WEIGHTS}")
        try:
            copy_weights(DRIVE_WEIGHTS, LOCAL_WEIGHTS, label="Drive → local")
            return str(LOCAL_WEIGHTS)
        except Exception as e:
            print(f"  ⚠ Copy failed ({e}), downloading from HuggingFace.")
    print(f"⬇ Downloading weights from {HF_MODEL_ID}...")
    return HF_MODEL_ID


def cache_weights_to_drive():
    if DRIVE_WEIGHTS.exists() and any(DRIVE_WEIGHTS.iterdir()):
        return
    src = LOCAL_WEIGHTS if LOCAL_WEIGHTS.exists() else None
    if not src:
        try:
            from huggingface_hub import snapshot_download
            src = pathlib.Path(snapshot_download(HF_MODEL_ID, local_files_only=True))
        except:
            print("  ⚠ Cannot find weights to cache.")
            return
    weight_size = dir_size_bytes(src)
    print(f"\n💾 Saving weights to Drive ({weight_size / 1e9:.1f} GB)...")
    try:
        usage = shutil.disk_usage("/content/drive/MyDrive")
        if usage.free / 1e9 < weight_size / 1e9 + 1.0:
            print(f"   ⚠ Not enough Drive space. Skipping.")
            return
    except:
        pass
    try:
        copy_weights(src, DRIVE_WEIGHTS, label="local → Drive")
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
        if DRIVE_WEIGHTS.exists():
            try:
                shutil.rmtree(DRIVE_WEIGHTS)
            except:
                pass


# ══════════════════════════════════════════════════════════════
# PIPELINE LOADING
# ══════════════════════════════════════════════════════════════

REPO_DIR = pathlib.Path("/content/TRELLIS.2")
import torch, torch.nn as nn
import numpy as np
from PIL import Image
import cv2, imageio

try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
except:
    pass
try:
    torch.backends.cudnn.conv.fp32_precision = "tf32"
except:
    pass
torch.set_float32_matmul_precision("high")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import missinglink.postprocess_parallel as pp

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9

# Max faces for render — above this the nvdiffrec renderer can trigger
# illegal memory access which poisons the entire CUDA context.
RENDER_MAX_FACES = 16_000_000

print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM:.1f} GB")
print("Loading TRELLIS.2 pipeline...")
weights_path = resolve_weights()
downloaded_from_hf = (weights_path == HF_MODEL_ID)
trellis_pipe = Trellis2ImageTo3DPipeline.from_pretrained(weights_path)
trellis_pipe.cuda()

if downloaded_from_hf:
    try:
        from huggingface_hub import snapshot_download
        hf_cache_path = snapshot_download(HF_MODEL_ID, local_files_only=True)
        if not LOCAL_WEIGHTS.exists():
            print(f"\n📁 Copying HF cache to {LOCAL_WEIGHTS}...")
            copy_weights(hf_cache_path, LOCAL_WEIGHTS, label="HF cache → local")
    except Exception as e:
        print(f"  ⚠ Could not copy HF cache: {e}")
    threading.Thread(target=cache_weights_to_drive, daemon=True).start()

hdri = REPO_DIR / "assets" / "hdri" / "forest.exr"
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread(str(hdri), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device="cuda",
))
print("✅ TRELLIS.2 pipeline loaded")


# ══════════════════════════════════════════════════════════════
# CUDA SAFETY HELPERS
# ══════════════════════════════════════════════════════════════

def cuda_ok():
    try:
        torch.cuda.synchronize()
        return True
    except:
        return False


def safe_cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    gc.collect()


def safe_offload_models():
    """Offload ALL pipeline models to CPU. Logs any failures."""
    freed_before = torch.cuda.memory_allocated() / 1e9
    for name, model in trellis_pipe.models.items():
        try:
            model.to("cpu")
        except Exception as e:
            print(f"    ⚠ offload {name} failed: {e}")
    try:
        trellis_pipe.image_cond_model.to("cpu")
    except Exception as e:
        print(f"    ⚠ offload image_cond_model failed: {e}")
    for attr_name in dir(trellis_pipe):
        try:
            attr = getattr(trellis_pipe, attr_name)
            if isinstance(attr, torch.nn.Module) and any(
                    p.is_cuda for p in attr.parameters()
            ):
                attr.to("cpu")
        except:
            pass
    safe_cleanup()
    freed_after = torch.cuda.memory_allocated() / 1e9
    freed = freed_before - freed_after
    print(f"    📤 Models offloaded: {freed:.1f}GB freed | {TOTAL_VRAM - freed_after:.1f}GB VRAM free")


def safe_reload_models():
    trellis_pipe.cuda()


# ══════════════════════════════════════════════════════════════
# RMBG (LAZY LOADING)
# ══════════════════════════════════════════════════════════════

rmbg_pipe = None
rmbg_lock = threading.Lock()
rmbg_load_error = [None]


def _ensure_monkey_patch():
    """Apply the all_tied_weights_keys monkey patch for RMBG-1.4 compatibility."""
    if not hasattr(torch.nn.Module, "_patched_all_tied_weights_keys"):
        torch.nn.Module._patched_all_tied_weights_keys = True

        @property
        def _atwk(self):
            return {}

        setattr(torch.nn.Module, "all_tied_weights_keys", _atwk)
        print("  🔧 Applied all_tied_weights_keys monkey patch for RMBG-1.4")


def get_rmbg():
    global rmbg_pipe
    if rmbg_pipe is not None:
        return rmbg_pipe
    with rmbg_lock:
        if rmbg_pipe is not None:
            return rmbg_pipe
        print("🔄 Loading RMBG-1.4 background removal model...")
        t0 = time.perf_counter()
        try:
            _ensure_monkey_patch()
            from transformers import pipeline as hf_pipeline
            rmbg_pipe = hf_pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=-1,
            )
            dt = round(time.perf_counter() - t0, 1)
            print(f"✅ RMBG-1.4 loaded in {dt}s")
            rmbg_load_error[0] = None
        except Exception as e:
            rmbg_load_error[0] = str(e)
            print(f"❌ RMBG-1.4 load failed: {e}")
            traceback.print_exc()
            raise
    return rmbg_pipe


def has_transparency(img):
    """Check if a PIL Image has meaningful transparency."""
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    alpha_arr = np.array(alpha)
    non_opaque = np.sum(alpha_arr < 250)
    total = alpha_arr.size
    ratio = non_opaque / total if total > 0 else 0
    return ratio > 0.005


def auto_remove_bg(image_path):
    """Remove background from an image file, return path to transparent PNG."""
    rmbg = get_rmbg()
    result = rmbg(str(image_path))
    p = pathlib.Path(image_path)
    out_p = p.parent / f"{p.stem}_autormbg.png"
    result.save(str(out_p), "PNG")
    return str(out_p)


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

UPLOAD_DIR = pathlib.Path("/content/_trellis_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
jobs = {}
active_jobs = {}
_gpu_lock = threading.Lock()


def safe_stem(name):
    s = pathlib.Path(name).stem.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "image"


def fmt_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(max(n, 0))
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{int(f)} {units[i]}" if i == 0 else f"{f:.2f} {units[i]}"


# ══════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ══════════════════════════════════════════════════════════════

def do_render(mesh, mode, out_path, base, fps=15, resolution=1024,
              sprite_directions=16, sprite_size=256, sprite_pitch=0.52,
              doom_directions=8, doom_size=256, doom_pitch=0.0):
    """
    Render preview using full PBR pipeline, one frame at a time.

    Modes:
      none        — skip
      snapshot    — single PBR grid PNG
      video       — 120-frame bobbing camera MP4
      perspective — clean 360° turntable MP4
      rts_sprite  — transparent-BG sprite sheet for RTS/RPG games
      doom_sprite — Doom/Build-engine style billboard sprite sheet
    """
    n_faces = mesh.faces.shape[0]
    if mode == "none":
        return None, None
    if n_faces > RENDER_MAX_FACES:
        print(f"    ⚠  Skipping render ({n_faces:,} faces > {RENDER_MAX_FACES:,} limit)")
        return None, None

    try:
        # ── Camera paths per mode ──
        if mode == "rts_sprite":
            num_frames = sprite_directions
            yaws = [(-i * 2 * math.pi / num_frames + math.pi / 2) for i in range(num_frames)]
            pitch = [sprite_pitch] * num_frames
            render_res = sprite_size * 2
        elif mode == "doom_sprite":
            num_frames = doom_directions
            yaws = [(-i * 2 * math.pi / num_frames + math.pi / 2) for i in range(num_frames)]
            pitch = [doom_pitch] * num_frames
            render_res = doom_size * 2
        elif mode == "snapshot":
            num_frames = 1
            yaws = (-torch.linspace(0, 2 * 3.1415, num_frames) + np.pi / 2).tolist()
            pitch = [0.35] * num_frames
            render_res = resolution
        elif mode == "perspective":
            num_frames = 120
            yaws = (-torch.linspace(0, 2 * 3.1415, num_frames) + np.pi / 2).tolist()
            pitch = [0.3] * num_frames
            render_res = resolution
        else:  # video
            num_frames = 120
            yaws = (-torch.linspace(0, 2 * 3.1415, num_frames) + np.pi / 2).tolist()
            pitch = (0.25 + 0.5 * torch.sin(
                torch.linspace(0, 2 * 3.1415, num_frames)
            )).tolist()
            render_res = resolution

        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitch, rs=2, fovs=40,
        )

        renderer = render_utils.get_renderer(mesh, resolution=render_res)

        all_frames = {}
        for j in range(num_frames):
            res = renderer.render(mesh, extrinsics[j], intrinsics[j], envmap=envmap)
            for k, v in res.items():
                if k not in all_frames:
                    all_frames[k] = []
                if v.dim() == 2:
                    v = v[None].repeat(3, 1, 1)
                all_frames[k].append(
                    np.clip(v.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                )
            del res
            torch.cuda.empty_cache()

        del renderer
        torch.cuda.empty_cache()

        # ── Post-process per mode ──
        if mode == "snapshot":
            frame = render_utils.make_pbr_vis_frames(all_frames)[0]
            png_path = out_path / f"{base}_preview.png"
            Image.fromarray(frame).save(str(png_path))
            del frame, all_frames
            return str(png_path), "image"

        elif mode == "video":
            frames = render_utils.make_pbr_vis_frames(all_frames)
            mp4_path = out_path / f"{base}.mp4"
            imageio.mimsave(str(mp4_path), frames, fps=fps)
            del frames, all_frames
            return str(mp4_path), "video"

        elif mode == "perspective":
            frames = all_frames.get('shaded', [])
            mp4_path = out_path / f"{base}_perspective.mp4"
            imageio.mimsave(str(mp4_path), frames, fps=fps)
            del frames, all_frames
            return str(mp4_path), "video"

        elif mode == "rts_sprite":
            return _build_rts_spritesheet(
                all_frames, out_path, base,
                sprite_directions, sprite_size,
            )

        elif mode == "doom_sprite":
            return _build_doom_spritesheet(
                all_frames, out_path, base,
                doom_directions, doom_size,
            )

        else:
            del all_frames
            return None, None

    except Exception as e:
        print(f"    ⚠  Render failed ({mode}): {e}")
        if not cuda_ok():
            raise RuntimeError("CUDA context corrupted after render failure")
        return None, None


def _build_rts_spritesheet(all_frames, out_path, base, n_dirs, frame_size):
    """Composite rendered frames into an RTS-compatible sprite sheet."""
    shaded = all_frames.get('shaded', [])
    alpha_frames = all_frames.get('alpha', [])

    if not shaded:
        print("    ⚠  No shaded frames for sprite sheet")
        return None, None

    sprite_dir = out_path / f"{base}_sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)

    dir_labels_8 = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
    dir_labels_16 = ["S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
                     "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE"]

    individual_paths = []
    pil_frames = []

    for i in range(min(n_dirs, len(shaded))):
        rgb = Image.fromarray(shaded[i]).convert("RGB")

        if i < len(alpha_frames):
            alpha_np = alpha_frames[i]
            if alpha_np.ndim == 3 and alpha_np.shape[2] >= 1:
                alpha_ch = alpha_np[:, :, 0]
            else:
                alpha_ch = alpha_np
            alpha_img = Image.fromarray(alpha_ch).convert("L")
        else:
            rgb_np = np.array(rgb).astype(np.float32)
            lum = (rgb_np[..., 0] * 0.299 + rgb_np[..., 1] * 0.587 +
                   rgb_np[..., 2] * 0.114)
            alpha_ch = np.where(lum > 2.0, 255, 0).astype(np.uint8)
            alpha_img = Image.fromarray(alpha_ch).convert("L")

        rgb = rgb.resize((frame_size, frame_size), Image.LANCZOS)
        alpha_img = alpha_img.resize((frame_size, frame_size), Image.LANCZOS)

        rgba = rgb.copy()
        rgba.putalpha(alpha_img)

        bbox = rgba.getbbox()
        if bbox:
            cropped = rgba.crop(bbox)
            canvas = Image.new("RGBA", (frame_size, frame_size), (0, 0, 0, 0))
            cx = (frame_size - cropped.width) // 2
            cy = (frame_size - cropped.height) // 2
            canvas.paste(cropped, (cx, cy), cropped)
            rgba = canvas

        pil_frames.append(rgba)

        if n_dirs <= 8:
            lbl = dir_labels_8[i] if i < len(dir_labels_8) else f"dir{i}"
        elif n_dirs <= 16:
            lbl = dir_labels_16[i] if i < len(dir_labels_16) else f"dir{i}"
        else:
            lbl = f"dir{i:02d}"

        frame_path = sprite_dir / f"{base}_{lbl}.png"
        rgba.save(str(frame_path), "PNG")
        individual_paths.append(str(frame_path))

    # ── Build sprite sheet ──
    n = len(pil_frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    sheet_w = cols * frame_size
    sheet_h = rows * frame_size
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    for idx, frame in enumerate(pil_frames):
        col = idx % cols
        row = idx // cols
        sheet.paste(frame, (col * frame_size, row * frame_size), frame)

    sheet_path = out_path / f"{base}_spritesheet.png"
    sheet.save(str(sheet_path), "PNG")

    print(f"    ✓ Sprite sheet: {cols}×{rows} grid, {n} directions @ {frame_size}px")
    print(f"    ✓ Individual frames saved to {sprite_dir}/")

    del pil_frames, all_frames
    return str(sheet_path), "rts_sprite"


def _build_doom_spritesheet(all_frames, out_path, base, n_dirs, frame_size):
    """Build Doom/Build-engine style billboard sprite sheet."""
    shaded = all_frames.get('shaded', [])
    alpha_frames = all_frames.get('alpha', [])

    if not shaded:
        print("    ⚠  No shaded frames for Doom sprite sheet")
        return None, None

    sprite_dir = out_path / f"{base}_doom_sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)

    individual_paths = []
    pil_frames = []

    for i in range(min(n_dirs, len(shaded))):
        rgb = Image.fromarray(shaded[i]).convert("RGB")

        if i < len(alpha_frames):
            alpha_np = alpha_frames[i]
            if alpha_np.ndim == 3 and alpha_np.shape[2] >= 1:
                alpha_ch = alpha_np[:, :, 0]
            else:
                alpha_ch = alpha_np
            alpha_img = Image.fromarray(alpha_ch).convert("L")
        else:
            rgb_np = np.array(rgb).astype(np.float32)
            lum = (rgb_np[..., 0] * 0.299 + rgb_np[..., 1] * 0.587 +
                   rgb_np[..., 2] * 0.114)
            alpha_ch = np.where(lum > 2.0, 255, 0).astype(np.uint8)
            alpha_img = Image.fromarray(alpha_ch).convert("L")

        rgb = rgb.resize((frame_size, frame_size), Image.LANCZOS)
        alpha_img = alpha_img.resize((frame_size, frame_size), Image.LANCZOS)

        rgba = rgb.copy()
        rgba.putalpha(alpha_img)

        bbox = rgba.getbbox()
        if bbox:
            cropped = rgba.crop(bbox)
            canvas = Image.new("RGBA", (frame_size, frame_size), (0, 0, 0, 0))
            cx = (frame_size - cropped.width) // 2
            cy = frame_size - cropped.height
            canvas.paste(cropped, (cx, max(cy, 0)), cropped)
            rgba = canvas

        pil_frames.append(rgba)

        if n_dirs <= 8:
            lbl = f"A{i + 1}"
        else:
            lbl = f"A{i + 1:02d}"

        frame_path = sprite_dir / f"{base}_{lbl}.png"
        rgba.save(str(frame_path), "PNG")
        individual_paths.append(str(frame_path))

    # ── Build horizontal strip sprite sheet ──
    n = len(pil_frames)
    sheet_w = n * frame_size
    sheet_h = frame_size
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    for idx, frame in enumerate(pil_frames):
        sheet.paste(frame, (idx * frame_size, 0), frame)

    sheet_path = out_path / f"{base}_doom_sheet.png"
    sheet.save(str(sheet_path), "PNG")

    print(f"    ✓ Doom sprite sheet: {n}×1 strip, {n} angles @ {frame_size}px")
    print(f"    ✓ Individual frames saved to {sprite_dir}/")

    del pil_frames, all_frames
    return str(sheet_path), "doom_sprite"


# ══════════════════════════════════════════════════════════════
# GENERATION JOB
# ══════════════════════════════════════════════════════════════

STEPS = [
    ("Loading image...", 0.01),
    ("Running 3D reconstruction...", 0.30),
    ("Preparing mesh...", 0.10),
    ("UV unwrapping (xatlas)...", 0.20),
    ("Baking textures + GLB...", 0.24),
    ("Rendering preview...", 0.15),
]

MAX_RETRIES = 3


def run_generate_job(job_id):
    job = jobs[job_id]
    active_jobs["generate"] = job_id
    s = job["settings"]
    files = job["files"]
    out_path = pathlib.Path(s["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    total = len(files)
    done = 0
    t0_all = time.perf_counter()

    with _gpu_lock:
        for idx, (orig_name, file_path) in enumerate(files):
            base = safe_stem(orig_name)
            glb_out = out_path / f"{base}.glb"

            def set_phase(si):
                label, _ = STEPS[si]
                cum = sum(w for _, w in STEPS[:si])
                pct = (idx + cum) / total
                job["progress"] = {
                    "pct": round(pct * 100, 1),
                    "image_num": idx + 1, "total": total,
                    "name": orig_name, "phase": label,
                    "elapsed": round(time.perf_counter() - t0_all, 1),
                }

            set_phase(0)
            job["log"].append(f"[{idx + 1}/{total}] Processing: {orig_name}")
            t0 = time.perf_counter()

            # ── Auto background removal if enabled ──
            auto_rmbg = s.get("auto_rmbg", True)
            if auto_rmbg:
                try:
                    test_img = Image.open(file_path)
                    if not has_transparency(test_img):
                        job["log"].append(f"  🔍 No transparency detected — auto-removing background...")
                        job["progress"]["phase"] = "Auto-removing background..."
                        file_path = auto_remove_bg(file_path)
                        job["log"].append(f"  ✅ Background removed automatically")
                    else:
                        job["log"].append(f"  ✓ Image already has transparency")
                    del test_img
                except Exception as e:
                    job["log"].append(f"  ⚠ Auto background removal failed: {e} — proceeding with original")
                    traceback.print_exc()

            error = None
            for attempt in range(MAX_RETRIES):
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()

                    image = Image.open(file_path).convert("RGBA")

                    if attempt > 0:
                        job["log"].append(f"  🔄 Retry {attempt + 1}/{MAX_RETRIES}...")
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except:
                            pass
                        safe_reload_models()

                    set_phase(1)
                    out = trellis_pipe.run(
                        [image], image_weights=[1.0],
                        sparse_structure_sampler_params={"steps": 12},
                        shape_slat_sampler_params={"steps": 12},
                        tex_slat_sampler_params={"steps": 12},
                    )
                    if not out:
                        raise RuntimeError("Empty pipeline result")
                    mesh = out[0]

                    mesh.vertices = mesh.vertices.clone()
                    mesh.faces = mesh.faces.clone()
                    if hasattr(mesh, 'attrs') and mesh.attrs is not None:
                        mesh.attrs = mesh.attrs.clone()
                    if hasattr(mesh, 'coords') and mesh.coords is not None:
                        mesh.coords = mesh.coords.clone()

                    recon_s = round(time.perf_counter() - t0, 2)
                    job["log"].append(
                        f"  ✓ Recon: {recon_s}s | "
                        f"{mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces"
                    )

                    # ── Simplify mesh before rendering ──
                    decimate_target = s["decimate_target"]
                    render_limit = min(decimate_target, RENDER_MAX_FACES)
                    n_raw = mesh.faces.shape[0]
                    if n_raw > render_limit:
                        job["log"].append(
                            f"  ▸ Simplifying: {n_raw:,} → {render_limit:,} faces"
                        )
                        mesh.simplify(render_limit)
                        mesh.vertices = mesh.vertices.clone()
                        mesh.faces = mesh.faces.clone()
                        job["log"].append(
                            f"  ✓ Simplified: {mesh.vertices.shape[0]:,} verts, "
                            f"{mesh.faces.shape[0]:,} faces"
                        )

                    # ── Render preview ──
                    set_phase(5)
                    render_mode = s.get("render_mode", "video")
                    media_path, media_type = None, None
                    if render_mode != "none":
                        free_gb = TOTAL_VRAM - torch.cuda.memory_allocated() / 1e9
                        job["log"].append(f"  ▸ Rendering ({render_mode}) | {free_gb:.1f}GB free")
                        media_path, media_type = do_render(
                            mesh, render_mode, out_path, base,
                            fps=s["fps"],
                            sprite_directions=s.get("sprite_directions", 16),
                            sprite_size=s.get("sprite_size", 256),
                            sprite_pitch=s.get("sprite_pitch", 0.52),
                            doom_directions=s.get("doom_directions", 8),
                            doom_size=s.get("doom_size", 256),
                            doom_pitch=s.get("doom_pitch", 0.0),
                        )
                        if media_path:
                            job["log"].append(f"  ✓ Render: {fmt_bytes(pathlib.Path(media_path).stat().st_size)}")
                        torch.cuda.empty_cache()

                    # ── Offload models → CPU for mesh processing ──
                    del out
                    safe_offload_models()

                    # ── Prepare mesh ──
                    set_phase(2)
                    t_prep = time.perf_counter()
                    prepared = pp.prepare_mesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        attr_volume=mesh.attrs,
                        coords=mesh.coords,
                        attr_layout=mesh.layout,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        voxel_size=mesh.voxel_size,
                        decimation_target=s["decimate_target"],
                        texture_size=s["texture_size"],
                        remesh=s["remesh"],
                        remesh_band=s["remesh_band"],
                        verbose=True,
                        name=base,
                    )
                    prep_s = round(time.perf_counter() - t_prep, 2)
                    job["log"].append(f"  ✓ Prepare: {prep_s}s")
                    del mesh
                    safe_cleanup()

                    # ── xatlas UV unwrap ──
                    set_phase(3)
                    t_uv = time.perf_counter()
                    unwrapped = pp.uv_unwrap(prepared, verbose=True)
                    uv_s = round(time.perf_counter() - t_uv, 2)
                    job["log"].append(f"  ✓ xatlas: {uv_s}s")
                    del prepared

                    # ── Texture bake + GLB export ──
                    set_phase(4)
                    t_bake = time.perf_counter()
                    pp.bake_and_export(unwrapped, str(glb_out), verbose=True)
                    bake_s = round(time.perf_counter() - t_bake, 2)
                    glb_size = glb_out.stat().st_size
                    job["log"].append(f"  ✓ Bake: {bake_s}s | GLB: {fmt_bytes(glb_size)}")
                    del unwrapped
                    safe_cleanup()

                    # ── Done ──
                    dt = round(time.perf_counter() - t0, 2)
                    result_entry = {"name": base, "glb": str(glb_out)}
                    if media_path:
                        result_entry["media"] = media_path
                        result_entry["media_type"] = media_type
                    if media_type == "rts_sprite":
                        sprite_dir = out_path / f"{base}_sprites"
                        if sprite_dir.exists():
                            frames = sorted([str(f) for f in sprite_dir.glob("*.png")])
                            result_entry["sprite_frames"] = frames
                            result_entry["sprite_dir"] = str(sprite_dir)
                    if media_type == "doom_sprite":
                        doom_dir = out_path / f"{base}_doom_sprites"
                        if doom_dir.exists():
                            frames = sorted([str(f) for f in doom_dir.glob("*.png")])
                            result_entry["sprite_frames"] = frames
                            result_entry["sprite_dir"] = str(doom_dir)
                    job["log"].append(f"  ✅ {base} — GLB: {fmt_bytes(glb_size)} ({dt}s)")
                    job["results"].append(result_entry)
                    done += 1
                    error = None
                    break

                except Exception as e:
                    err = str(e).lower()
                    retryable = ("storage" in err or "out of memory" in err
                                 or "illegal memory" in err or "cuda error" in err
                                 or "accelerator" in err)
                    if attempt < MAX_RETRIES - 1 and retryable:
                        job["log"].append(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                        try:
                            del out
                        except:
                            pass
                        try:
                            del mesh
                        except:
                            pass
                        try:
                            del prepared
                        except:
                            pass
                        try:
                            del unwrapped
                        except:
                            pass
                        safe_offload_models()
                        gc.collect()
                        try:
                            torch.cuda.synchronize()
                        except:
                            pass
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except:
                            pass
                        free = TOTAL_VRAM - torch.cuda.memory_allocated() / 1e9
                        job["log"].append(f"    Cleanup done | {free:.1f}GB free")
                        time.sleep(3)
                    else:
                        error = str(e)
                        break

            if error:
                job["log"].append(f"  ❌ {orig_name}: {error}")
                traceback.print_exc()

            safe_offload_models()

            if idx < total - 1:
                safe_reload_models()

        dt_total = time.perf_counter() - t0_all
        job["log"].append(f"\nDone — {done}/{total} in {dt_total:.1f}s")
        job["status"] = "done"
        job["progress"] = {
            "pct": 100, "image_num": total, "total": total,
            "name": "Complete", "phase": "All done!",
            "elapsed": round(dt_total, 1),
        }


def run_rmbg_job(job_id):
    job = jobs[job_id]
    active_jobs["rmbg"] = job_id
    files = job["files"]
    total = len(files)
    done = 0
    t0 = time.perf_counter()
    with rmbg_lock:
        job["progress"] = {"pct": 0, "image_num": 0, "total": total,
                           "name": "Loading model...", "phase": "Loading RMBG-1.4...", "elapsed": 0}
        job["log"].append("Loading background removal model...")
        rmbg = get_rmbg()
        job["log"].append("Model loaded.")
        for idx, (orig_name, file_path) in enumerate(files):
            base = safe_stem(orig_name)
            out_p = pathlib.Path(file_path).parent / f"{base}_transparent.png"
            job["progress"] = {"pct": round((idx / total) * 100, 1), "image_num": idx + 1,
                               "total": total, "name": orig_name,
                               "phase": "Removing background...",
                               "elapsed": round(time.perf_counter() - t0, 1)}
            job["log"].append(f"[{idx + 1}/{total}] {orig_name}")
            try:
                rgba = rmbg(str(file_path))
                rgba.save(str(out_p), "PNG")
                job["log"].append(f"  ✅ {base}_transparent.png")
                job["results"].append({"name": base, "file": str(out_p), "original": orig_name})
                done += 1
            except Exception as e:
                job["log"].append(f"  ❌ {orig_name}: {e}")
                traceback.print_exc()
        dt = time.perf_counter() - t0
        job["log"].append(f"\nDone — {done}/{total} in {dt:.1f}s")
        job["status"] = "done"
        job["progress"] = {"pct": 100, "image_num": total, "total": total,
                           "name": "Complete", "phase": "All done!",
                           "elapsed": round(dt, 1)}


# ══════════════════════════════════════════════════════════════
# FLASK APP & ROUTES
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)


@app.route("/api/keepalive")
def api_keepalive():
    return jsonify({"ok": True})


@app.route("/")
def index():
    # Serve the HTML page from a file or inline
    html_path = pathlib.Path(__file__).parent / "index.html"
    if html_path.exists():
        return send_file(str(html_path), mimetype="text/html")
    else:
        return Response("<h1>index.html not found</h1>", mimetype="text/html", status=404)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images"}), 400
    settings = json.loads(request.form.get("settings", "{}"))
    for k, v in [("output_dir", "/content/drive/MyDrive/trellis_models_out"),
                 ("fps", 15), ("texture_size", 4096), ("decimate_target", 1000000),
                 ("remesh", True), ("remesh_band", 1.0), ("render_mode", "video"),
                 ("video_resolution", 512),
                 ("sprite_directions", 16), ("sprite_size", 256), ("sprite_pitch", 0.52),
                 ("doom_directions", 8), ("doom_size", 256), ("doom_pitch", 0.0),
                 ("auto_rmbg", True)]:
        settings.setdefault(k, v)

    # ── Validate output_dir ──
    SAFE_OUTPUT_BASES = ["/content/drive/MyDrive", "/content/"]
    raw_out = settings.get("output_dir", "")
    real_out = os.path.realpath(raw_out)
    out_ok = False
    for base in SAFE_OUTPUT_BASES:
        real_base = os.path.realpath(base)
        if real_out == real_base or real_out.startswith(real_base + os.sep):
            out_ok = True
            break
    if not out_ok:
        return jsonify({"error": f"Output directory must be under Google Drive or /content/. Got: {raw_out}"}), 400
    settings["output_dir"] = real_out

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        safe_name = secure_filename(f.filename) or f"upload_{uuid.uuid4().hex[:8]}.png"
        dest = job_dir / safe_name
        f.save(str(dest))
        saved.append((f.filename, str(dest)))
    jobs[job_id] = {
        "status": "running",
        "progress": {"pct": 0, "image_num": 0, "total": len(saved),
                     "name": "Starting...", "phase": "Preparing...", "elapsed": 0},
        "log": [], "results": [], "files": saved, "settings": settings,
    }
    threading.Thread(target=run_generate_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/rmbg", methods=["POST"])
def api_rmbg():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images"}), 400
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        safe_name = secure_filename(f.filename) or f"upload_{uuid.uuid4().hex[:8]}.png"
        dest = job_dir / safe_name
        f.save(str(dest))
        saved.append((f.filename, str(dest)))
    jobs[job_id] = {
        "status": "running",
        "progress": {"pct": 0, "image_num": 0, "total": len(saved),
                     "name": "Starting...", "phase": "Loading model...", "elapsed": 0},
        "log": [], "results": [], "files": saved, "settings": {},
    }
    threading.Thread(target=run_rmbg_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"],
                    "log": job["log"], "results": job["results"]})


@app.route("/api/console")
def api_console():
    return jsonify({"lines": list(console_lines)[-200:]})


@app.route("/api/active")
def api_active():
    r = {}
    for k in ["generate", "rmbg"]:
        j = active_jobs.get(k)
        if j and j in jobs and jobs[j]["status"] == "running":
            r[k] = j
    return jsonify(r)


@app.route("/api/file")
def api_file():
    p = request.args.get("p", "")
    if not p:
        return "Not found", 404

    try:
        real = os.path.realpath(p)
    except (ValueError, OSError):
        return "Invalid path", 400
    if not os.path.isfile(real):
        return "Not found", 404

    allowed_dirs = set()
    allowed_dirs.add(os.path.realpath(str(UPLOAD_DIR)))

    for job in jobs.values():
        out = job.get("settings", {}).get("output_dir")
        if out:
            allowed_dirs.add(os.path.realpath(out))

    in_allowed = False
    for allowed in allowed_dirs:
        try:
            if real == allowed or real.startswith(allowed + os.sep):
                in_allowed = True
                break
        except (ValueError, TypeError):
            continue

    if not in_allowed:
        return "Access denied", 403

    return send_file(real)
