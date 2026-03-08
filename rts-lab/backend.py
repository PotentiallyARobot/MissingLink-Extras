# ============================================================
# 🌍 RTS-LAB — World Editor Backend
#
# Lightweight Flask server for the 2.5D world editor.
# Provides:
#   /                     → serve worldeditor.html
#   /api/keepalive        → health check
#   /api/models           → list GLBs from trellis output
#   /api/file?p=          → serve files (GLBs, PNGs, etc.)
#   /api/world/save       → persist world JSON to drive
#   /api/world/load       → load world JSON
#   /api/world/list       → list saved worlds
#   /api/world/delete     → delete a saved world
#   /api/bake-sprite      → server-side render a GLB as iso sprite (GPU)
# ============================================================

import os, sys, pathlib, json, uuid, time, threading, traceback, collections, shutil, math, re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# ── Drive mount ──
if not os.path.exists("/content/drive/MyDrive"):
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask"])
    from flask import Flask, request, jsonify, send_file, Response

from werkzeug.utils import secure_filename

IN_COLAB = False
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    eval_js = None

# ── Console capture ──
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

console_lines = collections.deque(maxlen=400)
sys.stdout = TeeWriter(sys.__stdout__, console_lines)
sys.stderr = TeeWriter(sys.__stderr__, console_lines)

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

TRELLIS_OUT = pathlib.Path("/content/drive/MyDrive/trellis_models_out")
WORLDS_DIR  = pathlib.Path("/content/drive/MyDrive/rts_lab_worlds")
UPLOAD_DIR  = pathlib.Path("/content/rts_lab_uploads")

for d in [WORLDS_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)


@app.route("/api/keepalive")
def api_keepalive():
    return jsonify({"ok": True})


@app.route("/")
def index():
    html_path = pathlib.Path(__file__).parent / "index.html"
    if html_path.exists():
        return send_file(str(html_path), mimetype="text/html")
    return Response("<h1>index.html not found</h1>", 404, content_type="text/html")


# ══════════════════════════════════════════════════════════════
# MODEL LISTING — scans trellis output for GLBs
# ══════════════════════════════════════════════════════════════

@app.route("/api/models")
def api_models():
    models = []
    seen = set()

    if TRELLIS_OUT.is_dir():
        for glb in sorted(TRELLIS_OUT.rglob("*.glb")):
            real = str(glb.resolve())
            if real in seen:
                continue
            seen.add(real)
            name = glb.stem
            # Look for existing sprite render
            sprite = glb.parent / f"{name}_iso_sprite.png"
            models.append({
                "name": name,
                "glb_url": f"/api/file?p={real}",
                "glb_path": real,
                "sprite_url": f"/api/file?p={sprite.resolve()}" if sprite.exists() else None,
                "size_kb": round(glb.stat().st_size / 1024),
            })

    return jsonify({"models": models})


# ══════════════════════════════════════════════════════════════
# FILE SERVING
# ══════════════════════════════════════════════════════════════

ALLOWED_ROOTS = set()

def _resolve_allowed():
    ALLOWED_ROOTS.clear()
    for p in [TRELLIS_OUT, WORLDS_DIR, UPLOAD_DIR]:
        try:
            ALLOWED_ROOTS.add(str(p.resolve()))
        except Exception:
            pass

_resolve_allowed()


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

    _resolve_allowed()
    for base in ALLOWED_ROOTS:
        if real == base or real.startswith(base + os.sep):
            return send_file(real)
    return "Access denied", 403


# ══════════════════════════════════════════════════════════════
# WORLD SAVE / LOAD
# ══════════════════════════════════════════════════════════════

def _safe_world_name(name):
    name = re.sub(r'[^\w\s\-]', '', name).strip()
    if not name:
        name = f"world_{uuid.uuid4().hex[:6]}"
    return name


@app.route("/api/world/save", methods=["POST"])
def api_world_save():
    data = request.get_json(force=True)
    name = _safe_world_name(data.get("name", "untitled"))
    world_data = data.get("world", {})

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    path = WORLDS_DIR / f"{name}.json"

    world_data["_meta"] = {
        "name": name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object_count": len(world_data.get("objects", [])),
    }

    with open(str(path), "w") as f:
        json.dump(world_data, f, indent=2)

    return jsonify({"ok": True, "path": str(path), "name": name})


@app.route("/api/world/load")
def api_world_load():
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "No name"}), 400
    path = WORLDS_DIR / f"{_safe_world_name(name)}.json"
    if not path.exists():
        return jsonify({"error": "World not found"}), 404
    with open(str(path)) as f:
        data = json.load(f)
    return jsonify({"ok": True, "world": data, "name": name})


@app.route("/api/world/list")
def api_world_list():
    worlds = []
    if WORLDS_DIR.is_dir():
        for f in sorted(WORLDS_DIR.glob("*.json")):
            try:
                with open(str(f)) as fh:
                    meta = json.load(fh).get("_meta", {})
                worlds.append({
                    "name": f.stem,
                    "saved_at": meta.get("saved_at", ""),
                    "object_count": meta.get("object_count", 0),
                })
            except Exception:
                worlds.append({"name": f.stem, "saved_at": "?", "object_count": 0})
    return jsonify({"worlds": worlds})


@app.route("/api/world/delete", methods=["POST"])
def api_world_delete():
    data = request.get_json(force=True)
    name = _safe_world_name(data.get("name", ""))
    path = WORLDS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


# ══════════════════════════════════════════════════════════════
# BAKE SPRITE — render an isometric sprite for a GLB on GPU
# Uses the TRELLIS render pipeline if available, otherwise
# falls back to a simple Three.js-like render stub
# ══════════════════════════════════════════════════════════════

_bake_lock = threading.Lock()

@app.route("/api/bake-sprite", methods=["POST"])
def api_bake_sprite():
    """
    Render a GLB model as a transparent isometric sprite PNG.
    Body JSON: { glb_path, pitch?, yaw?, size? }
    Returns: { sprite_url, width, height }
    """
    data = request.get_json(force=True)
    glb_path = data.get("glb_path", "")
    pitch = data.get("pitch", 0.52)   # 30° iso default
    yaw = data.get("yaw", 0.785)      # 45° default
    size = data.get("size", 512)

    if not glb_path or not os.path.isfile(glb_path):
        return jsonify({"error": "GLB not found"}), 404

    stem = pathlib.Path(glb_path).stem
    out_dir = pathlib.Path(glb_path).parent
    sprite_path = out_dir / f"{stem}_iso_sprite.png"

    # If already rendered, return it
    if sprite_path.exists():
        return jsonify({
            "sprite_url": f"/api/file?p={sprite_path.resolve()}",
            "width": size, "height": size, "cached": True,
        })

    # Try GPU render using trellis pipeline
    with _bake_lock:
        try:
            import torch
            from trellis.utils import render_utils
            from PIL import Image
            import numpy as np

            # Load the render mesh if cached
            render_pt = out_dir / f"{stem}_render_mesh.pt"
            if render_pt.exists():
                mesh = torch.load(str(render_pt), map_location="cuda")
            else:
                return jsonify({"error": "No render mesh found — generate the model first"}), 400

            # Render single frame
            extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                [yaw], [pitch], rs=2, fovs=40,
            )
            renderer = render_utils.get_renderer(mesh, resolution=size * 2)
            res = renderer.render(mesh, extrinsics[0], intrinsics[0])

            shaded = res.get('shaded')
            alpha = res.get('alpha')

            if shaded is None:
                del renderer, res
                torch.cuda.empty_cache()
                return jsonify({"error": "Render produced no output"}), 500

            rgb_np = np.clip(shaded.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_np).convert("RGB").resize((size, size), Image.LANCZOS)

            if alpha is not None:
                a_np = alpha.detach().cpu().numpy()
                if a_np.ndim == 3:
                    a_np = a_np[0] if a_np.shape[0] == 1 else a_np[:,:,0]
                a_np = np.clip(a_np * 255, 0, 255).astype(np.uint8)
                alpha_img = Image.fromarray(a_np).convert("L").resize((size, size), Image.LANCZOS)
            else:
                rgb_arr = np.array(rgb_img).astype(np.float32)
                lum = rgb_arr[...,0]*0.299 + rgb_arr[...,1]*0.587 + rgb_arr[...,2]*0.114
                alpha_img = Image.fromarray(np.where(lum > 2.0, 255, 0).astype(np.uint8)).convert("L")

            rgba = rgb_img.copy()
            rgba.putalpha(alpha_img)
            rgba.save(str(sprite_path), "PNG")

            del renderer, res, mesh
            torch.cuda.empty_cache()

            return jsonify({
                "sprite_url": f"/api/file?p={sprite_path.resolve()}",
                "width": size, "height": size, "cached": False,
            })

        except ImportError:
            return jsonify({"error": "TRELLIS pipeline not loaded — sprites must be pre-rendered"}), 501
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# CONSOLE
# ══════════════════════════════════════════════════════════════

@app.route("/api/console")
def api_console():
    return jsonify({"lines": list(console_lines)[-200:]})
