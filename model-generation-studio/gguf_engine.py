"""
gguf_engine.py — TRELLIS.2 Q4 GGUF pipeline for the studio backend. Pure Python, no ComfyUI.

Consolidates everything proven in the bring-up cell:
  * ComfyUI shims (folder_paths.models_dir, comfy.utils.ProgressBar)
  * auto-vendor of Aero-Ex's trellis2_gguf engine + model_manager.py
  * file patch 1: fallback GGML Linear inherits GGMLLayer (packed-tensor loading)
  * file patch 2: o_voxel tiled-mesh shim (stock wheels lack the tiled variant)
  * utils3d compat shims (intrinsics_from_fov_xy, get_image_rays)
  * BiRefNet (GPU, fp32) attached as the pipeline's background remover
  * dequant-weight cache (fp16 cached after first forward — big speedup on A100)
  * a run() adapter matching the studio backend's stock call signature

Enable in the studio with:  TRELLIS2_GGUF=1  (and optionally TRELLIS2_GGUF_QUANT).
Measured on A100-40GB: ~127s/model warm before dequant cache, peak VRAM 4.4GB.
"""

import os
import sys
import json
import types
import shutil
import tempfile
import subprocess
import importlib.util

import torch

ENGINE_DIR   = os.environ.get("TRELLIS2_GGUF_DIR", "/content/trellis2_gguf_engine")
MODELS_ROOT  = os.environ.get("TRELLIS2_GGUF_MODELS", "/content/trellis2_gguf_models")
QUANT        = os.environ.get("TRELLIS2_GGUF_QUANT", "Q4_K_M")
CACHE_DEQUANT = os.environ.get("TRELLIS2_GGUF_CACHE_DEQUANT", "1").lower() not in ("0", "false", "no")
WRAPPER_REPO = "https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF"


# ── shims ─────────────────────────────────────────────────────────────────────
def _install_shims():
    os.makedirs(MODELS_ROOT, exist_ok=True)
    fp = sys.modules.get("folder_paths") or types.ModuleType("folder_paths")
    fp.models_dir = MODELS_ROOT
    sys.modules["folder_paths"] = fp
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = types.ModuleType("comfy")
    if "comfy.utils" not in sys.modules:
        cu = types.ModuleType("comfy.utils")

        class ProgressBar:
            def __init__(self, total=0, *a, **k): self.total = total
            def update(self, *a, **k): pass
            def update_absolute(self, *a, **k): pass

        cu.ProgressBar = ProgressBar
        sys.modules["comfy.utils"] = cu
        sys.modules["comfy"].utils = cu


# ── vendoring + file patches ──────────────────────────────────────────────────
def _vendor_engine():
    if os.path.isdir(os.path.join(ENGINE_DIR, "trellis2_gguf")):
        return
    os.makedirs(ENGINE_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp()
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", WRAPPER_REPO, os.path.join(tmp, "w")])
        shutil.copytree(os.path.join(tmp, "w", "trellis2_gguf"), os.path.join(ENGINE_DIR, "trellis2_gguf"))
        shutil.copy2(os.path.join(tmp, "w", "model_manager.py"), os.path.join(ENGINE_DIR, "model_manager.py"))
        for root, _d, fns in os.walk(os.path.join(ENGINE_DIR, "trellis2_gguf")):
            for fn in fns:
                if fn.endswith(".orig"):
                    os.remove(os.path.join(root, fn))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _patch_engine():
    """Apply both proven file patches. Idempotent. Must run before engine import."""
    # Patch 1: standalone GGML Linear (packed Q4 loading without ComfyUI).
    gu = os.path.join(ENGINE_DIR, "trellis2_gguf", "utils", "gguf_utils.py")
    src = open(gu).read()
    if "class Linear(GGMLLayer, torch.nn.Linear)" not in src:
        OLD = (
            "    class GGMLOpsFallback:\n"
            "        class Linear(torch.nn.Module):\n"
            "            def __init__(self, *args, **kwargs):\n"
            "                super().__init__()\n"
        )
        NEW = (
            "    class GGMLOpsFallback:\n"
            "        class Linear(GGMLLayer, torch.nn.Linear):\n"
            "            def __init__(self, in_features=1, out_features=1, bias=True, device=None, dtype=None):\n"
            "                torch.nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)\n"
            "            def forward(self, input, *args, **kwargs):\n"
            "                fn = getattr(self, 'forward_ggml_cast_weights', None)\n"
            "                if fn is not None:\n"
            "                    return fn(input)\n"
            "                weight, bias = self.cast_bias_weight(input)\n"
            "                return torch.nn.functional.linear(input, weight, bias)\n"
            "            def forward_comfy_cast_weights(self, input, *args, **kwargs):\n"
            "                x = input.feats if hasattr(input, 'feats') else input\n"
            "                weight, bias = self.cast_bias_weight(x)\n"
            "                out = torch.nn.functional.linear(x, weight, bias)\n"
            "                return input.replace(out) if hasattr(input, 'replace') else out\n"
        )
        if OLD not in src:
            raise RuntimeError("gguf_utils fallback block not found — upstream changed")
        open(gu, "w").write(src.replace(OLD, NEW, 1))
        if "trellis2_gguf.utils.gguf_utils" in sys.modules:
            raise RuntimeError("trellis2_gguf imported before patching — restart the runtime")

    # Patch 2: o_voxel tiled mesh converter shim (stock wheels lack it).
    fdg = os.path.join(ENGINE_DIR, "trellis2_gguf", "models", "sc_vaes", "fdg_vae.py")
    src = open(fdg).read()
    if "def tiled_flexible_dual_grid_to_mesh" not in src:
        OLD2 = "from o_voxel.convert import flexible_dual_grid_to_mesh, tiled_flexible_dual_grid_to_mesh\n"
        NEW2 = (
            "try:\n"
            "    from o_voxel.convert import flexible_dual_grid_to_mesh, tiled_flexible_dual_grid_to_mesh\n"
            "except ImportError:\n"
            "    from o_voxel.convert import flexible_dual_grid_to_mesh\n"
            "    def tiled_flexible_dual_grid_to_mesh(*args, tile_size=None, **kwargs):\n"
            "        return flexible_dual_grid_to_mesh(*args, **kwargs)\n"
        )
        if OLD2 not in src:
            raise RuntimeError("fdg_vae o_voxel import not found — upstream changed")
        open(fdg, "w").write(src.replace(OLD2, NEW2, 1))
        if "trellis2_gguf.models.sc_vaes.fdg_vae" in sys.modules:
            raise RuntimeError("fdg_vae imported before patching — restart the runtime")


def _shim_utils3d():
    """Recreate functions removed/renamed in newer utils3d (verified numerically)."""
    import utils3d as u3d
    if not hasattr(u3d.torch, "intrinsics_from_fov_xy"):
        def intrinsics_from_fov_xy(fov_x, fov_y):
            fov_x = torch.as_tensor(fov_x); fov_y = torch.as_tensor(fov_y)
            fx = 0.5 / torch.tan(fov_x / 2); fy = 0.5 / torch.tan(fov_y / 2)
            K = torch.zeros(*fx.shape, 3, 3, dtype=torch.float32,
                            device=fx.device if fx.is_cuda else None)
            K[..., 0, 0] = fx; K[..., 1, 1] = fy
            K[..., 0, 2] = 0.5; K[..., 1, 2] = 0.5; K[..., 2, 2] = 1.0
            return K
        u3d.torch.intrinsics_from_fov_xy = intrinsics_from_fov_xy
    if not hasattr(u3d.torch, "get_image_rays"):
        def get_image_rays(extrinsics, intrinsics, width, height):
            dev = extrinsics.device; dt = torch.float32
            R = extrinsics[:3, :3].to(dt); t = extrinsics[:3, 3].to(dt)
            cam_o = -(R.transpose(0, 1) @ t)
            u = (torch.arange(width, device=dev, dtype=dt) + 0.5) / width
            v = (torch.arange(height, device=dev, dtype=dt) + 0.5) / height
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
            dirs = uv1 @ torch.linalg.inv(intrinsics.to(dt)).transpose(0, 1) @ R
            dirs = torch.nn.functional.normalize(dirs, dim=-1)
            return cam_o.expand(height, width, 3).contiguous(), dirs
        u3d.torch.get_image_rays = get_image_rays


def _install_dequant_cache():
    from trellis2_gguf.utils import gguf_utils as gu
    if getattr(gu.GGMLLayer, "_dq_cache_installed", False):
        return
    orig = gu.GGMLLayer.cast_bias_weight

    def cached(self, input=None, dtype=None, device=None):
        d = dtype if dtype is not None else getattr(input, "dtype", torch.float32)
        dev = device if device is not None else getattr(input, "device", None)
        c = getattr(self, "_dq_cache", None)
        if c is not None and c[0] == d and (dev is None or c[1] == dev):
            return c[2], c[3]
        w, b = orig(self, input=input, dtype=dtype, device=device)
        wd = w.device if w is not None else (b.device if b is not None else dev)
        self._dq_cache = (d, wd, w, b)
        return w, b

    gu.GGMLLayer.cast_bias_weight = cached
    gu.GGMLLayer._dq_cache_installed = True


# ── public API ────────────────────────────────────────────────────────────────
_PREPARED = False

def prepare():
    """Shims + vendor + file patches + sys.path + utils3d shims. Idempotent.
    Call before importing anything from trellis2_gguf."""
    global _PREPARED
    if _PREPARED:
        return
    _install_shims()
    _vendor_engine()
    _patch_engine()
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    _shim_utils3d()
    _PREPARED = True


def load_pipeline(log=print):
    """Build the Q4 GGUF pipeline with the studio-compatible run() adapter installed."""
    prepare()

    # model_manager: downloads pipeline.json + quant files on first run.
    spec = importlib.util.spec_from_file_location(
        "trellis2_gguf_model_manager", os.path.join(ENGINE_DIR, "model_manager.py"))
    mm = importlib.util.module_from_spec(spec)
    sys.modules["trellis2_gguf_model_manager"] = mm
    spec.loader.exec_module(mm)
    mm.CURRENT_MODELNAME = "Trellis2"
    model_path = mm.get_models_dir()
    from huggingface_hub import hf_hub_download
    pj = os.path.join(model_path, "pipeline.json")
    if not os.path.exists(pj):
        hf_hub_download(repo_id=mm.GGUF_REPO, filename="pipeline.json", local_dir=model_path)
    log(f"[gguf] ensuring {QUANT} weights (first run downloads ~2.5GB)")
    mm.ensure_model_files(f"GGUF {QUANT}", json.load(open(pj)), gguf_repo=mm.GGUF_REPO)

    from trellis2_gguf.pipelines import Trellis2ImageTo3DPipeline
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        model_path, keep_models_loaded=True, enable_gguf=True, gguf_quant=QUANT,
        precision=None, enable_sdnq=False, sdnq_use_quantized_matmul=True,
        sdnq_torch_compile=False, sdnq_svd_rank=32, isPixal3D=False)
    try:
        pipe.cuda()
    except Exception as e:
        log(f"[gguf] pipe.cuda() note: {e}")

    # GPU background removal (torch/transformers; no onnxruntime).
    try:
        from trellis2_gguf.pipelines.rembg import BiRefNet
        pipe.rembg_model = BiRefNet()
        pipe.rembg_model.cuda()
        pipe.rembg_model.model.float()
        log("[gguf] BiRefNet (GPU, fp32) attached for background removal")
    except Exception as e:
        log(f"[gguf] BiRefNet attach failed ({e}); feed RGBA inputs")

    if CACHE_DEQUANT:
        _install_dequant_cache()
        log("[gguf] dequant cache ON (fp16 cached after first use; faster, more VRAM)")

    # Keep models resident on the GPU when there's room. The fork defaults
    # low_vram=True, which offloads every stage's model to CPU after use and
    # chunks linear ops — pure overhead on a big card. Auto: disable low_vram
    # when free VRAM > 16GB. Override with TRELLIS2_GGUF_LOW_VRAM=0/1.
    try:
        lv_env = os.getenv("TRELLIS2_GGUF_LOW_VRAM", "").strip().lower()
        if lv_env in ("0", "false", "no"):
            low_vram = False
        elif lv_env in ("1", "true", "yes"):
            low_vram = True
        else:
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
            low_vram = free_gb < 16
        pipe.low_vram = low_vram
        log(f"[gguf] low_vram={low_vram} — models {'offloaded per stage' if low_vram else 'stay resident on GPU'}")
        if not low_vram:
            # decode_tex_slat() unconditionally calls move_all_to_cpu() to free
            # VRAM for the decoder, then load_tex_slat_decoder() — which is a
            # no-op when the decoder is already loaded, leaving it stranded on
            # CPU ("mat1 is on cuda:0, different from other tensors on cpu").
            # With models resident and plenty of VRAM the offload is pointless,
            # so neutralize it. Original kept at pipe._orig_move_all_to_cpu.
            pipe._orig_move_all_to_cpu = pipe.move_all_to_cpu
            pipe.move_all_to_cpu = lambda: print(
                "[gguf] move_all_to_cpu skipped (models resident; low_vram off)")
        # Preload every lazily-loaded sub-model NOW (SLat encoder, both flow
        # cascades, decoders, DINOv3) so nothing loads mid-generation. Uses the
        # fork's own load_* methods so GGUF quant flags stay correct. Only when
        # keeping models resident — on low-VRAM cards lazy loading is the point.
        if not low_vram:
            for meth in ("load_sparse_structure_model", "load_image_cond_model",
                         "load_shape_slat_flow_model_512", "load_shape_slat_flow_model_1024",
                         "load_tex_slat_flow_model_512", "load_tex_slat_flow_model_1024",
                         "load_shape_slat_decoder", "load_tex_slat_decoder",
                         "load_shape_slat_encoder"):
                try:
                    getattr(pipe, meth)()
                except Exception as pe:
                    log(f"[gguf] preload {meth} skipped: {pe}")
            loaded = sum(1 for m in pipe.models.values() if m is not None)
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
            log(f"[gguf] preloaded {loaded}/{len(pipe.models)} sub-models to GPU ({free_gb:.1f}GB VRAM still free)")
    except Exception as e:
        log(f"[gguf] low_vram config note: {e}")

    # run() adapter: accept the studio's stock call signature.
    fork_run = pipe.run

    def studio_run(images, image_weights=None, sparse_structure_sampler_params=None,
                   shape_slat_sampler_params=None, tex_slat_sampler_params=None,
                   cache_stages=None, load_stages=None, **kw):
        if load_stages:
            raise RuntimeError("GGUF mode does not support stage reloading (retexture/re-render).")
        if cache_stages:
            log("[gguf] note: stage caching unsupported in GGUF mode — Edit/Re-render disabled for this model")
        if image_weights and any(w != 1.0 for w in image_weights):
            log("[gguf] note: image_weights ignored (unsupported by the GGUF engine)")
        image = images[0] if isinstance(images, (list, tuple)) and len(images) == 1 else images
        # Fork needs a real alpha channel; only preprocess when alpha is absent/opaque.
        need_pre = True
        probe = image[0] if isinstance(image, (list, tuple)) else image
        try:
            if probe.mode == "RGBA" and probe.getchannel("A").getextrema()[0] < 250:
                need_pre = False
        except Exception:
            pass
        return fork_run(
            image,
            sparse_structure_sampler_params=sparse_structure_sampler_params or {"steps": 12},
            shape_slat_sampler_params=shape_slat_sampler_params or {"steps": 12},
            tex_slat_sampler_params=tex_slat_sampler_params or {"steps": 12},
            preprocess_image=need_pre,
            pipeline_type=None, use_tiled=False,
            generate_texture_slat=True, verbose=True,
        )

    pipe.run = studio_run
    log(f"[gguf] pipeline ready ({QUANT}, adapter installed)")
    return pipe
