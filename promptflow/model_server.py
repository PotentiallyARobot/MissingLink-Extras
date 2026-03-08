"""
model_server.py
===============
Unified local server managing two models:

  1. LOCAL LLM  — bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF (Q4_K_M, ~2GB)
                  Used for prompt generation / creative writing without content filters.
                  Runs via llama-cpp-python.
                  GPU state controlled by promptflow config: stays loaded or offloaded
                  to CPU before image generation begins.

  2. Z-IMAGE    — unsloth/Z-Image-Turbo-GGUF (Q4_K_M)
                  Runs via stable-diffusion-cpp.
                  Loaded on demand; GPU memory reclaimed from LLM first when needed.

Endpoints:
  POST /llm/chat            { messages: [...], temperature, max_tokens }
  POST /llm/offload         {} — moves LLM weights to CPU (or deletes ctx)
  POST /llm/reload          {} — restores LLM to GPU layers
  GET  /llm/status          → { loaded, on_gpu, model }
  POST /image/txt2img       { prompt, negative_prompt, width, height, steps, cfg_scale, seed }
  POST /image/img2img       { ...above + init_images: [b64], denoising_strength }
  GET  /image/status        → { loaded, model }
  GET  /health              → { llm, image, gpu_memory_mb }

Usage:
  python model_server.py \\
    --diff_path   /path/z-image-turbo-Q4_K_M.gguf \\
    --llm_path    /path/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf \\
    --vae_path    /path/ae.safetensors \\
    --port        7860

  # For z-image only (no local LLM, use OpenAI via interpreter):
  python model_server.py --diff_path ... --vae_path ... --no_local_llm
"""

import gc
import io
import os
import sys
import json
import base64
import ctypes
import logging
import argparse
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any

from PIL import Image

# ─────────────────────────── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[model-server] %(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model-server")


# ─────────────────────────── GPU memory helpers ────────────────────────────────
def get_gpu_free_mb() -> int:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return -1


def torch_empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ─────────────────────────── Model State Container ───────────────────────────
class ModelState:
    def __init__(self):
        self._lock = threading.RLock()

        # LLM state
        self.llm = None               # llama_cpp.Llama instance
        self.llm_path: str = ""
        self.llm_n_gpu_layers: int = 0
        self.llm_on_gpu: bool = False
        self.llm_loaded: bool = False
        self.llm_ctx_size: int = 4096

        # Image gen state
        self.sd = None                # StableDiffusion instance
        self.diff_path: str = ""
        self.llm_diff_path: str = ""  # LLM component inside z-image
        self.vae_path: str = ""
        self.sd_loaded: bool = False

        # Config
        self.use_local_llm: bool = True
        self.offload_cpu: bool = True
        self.flash_attn: bool = True


STATE = ModelState()


# ─────────────────────────── LLM (llama-cpp-python) ───────────────────────────
def load_llm(path: str, n_gpu_layers: int = 99, n_ctx: int = 4096, verbose: bool = False):
    """Load the uncensored local LLM onto GPU (n_gpu_layers layers on VRAM)."""
    from llama_cpp import Llama
    log.info(f"Loading local LLM: {os.path.basename(path)}  n_gpu_layers={n_gpu_layers}")
    llm = Llama(
        model_path=path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
        chat_format="llama-3",
    )
    with STATE._lock:
        STATE.llm = llm
        STATE.llm_path = path
        STATE.llm_n_gpu_layers = n_gpu_layers
        STATE.llm_on_gpu = n_gpu_layers > 0
        STATE.llm_loaded = True
        STATE.llm_ctx_size = n_ctx
    log.info("✅ Local LLM ready.")


def offload_llm_to_cpu():
    """
    Offload LLM from GPU to CPU.
    Strategy: delete and reload with n_gpu_layers=0 (CPU-only).
    This fully frees VRAM for the diffusion model.
    """
    with STATE._lock:
        if not STATE.llm_loaded:
            return {"status": "not_loaded"}
        if not STATE.llm_on_gpu:
            return {"status": "already_on_cpu"}

        log.info("⬇️  Offloading LLM from GPU → CPU ...")
        path = STATE.llm_path
        ctx = STATE.llm_ctx_size

        # Delete current GPU instance
        del STATE.llm
        STATE.llm = None
        gc.collect()
        torch_empty_cache()

        free_before = get_gpu_free_mb()
        log.info(f"   GPU free after LLM delete: {free_before} MB")

    # Reload on CPU (outside lock to not block requests during load)
    from llama_cpp import Llama
    log.info("   Reloading LLM on CPU (n_gpu_layers=0)...")
    llm_cpu = Llama(
        model_path=path,
        n_gpu_layers=0,
        n_ctx=ctx,
        verbose=False,
        chat_format="llama-3",
    )

    with STATE._lock:
        STATE.llm = llm_cpu
        STATE.llm_on_gpu = False

    log.info("✅ LLM offloaded to CPU.")
    return {"status": "offloaded", "gpu_free_mb": get_gpu_free_mb()}


def reload_llm_to_gpu():
    """Reload LLM back onto GPU (requires image gen to release VRAM first)."""
    with STATE._lock:
        if not STATE.llm_loaded:
            return {"status": "not_loaded"}
        if STATE.llm_on_gpu:
            return {"status": "already_on_gpu"}

        path = STATE.llm_path
        n_gpu = STATE.llm_n_gpu_layers
        ctx = STATE.llm_ctx_size

        del STATE.llm
        STATE.llm = None
        gc.collect()

    from llama_cpp import Llama
    log.info(f"⬆️  Reloading LLM onto GPU (n_gpu_layers={n_gpu})...")
    llm_gpu = Llama(
        model_path=path,
        n_gpu_layers=n_gpu,
        n_ctx=ctx,
        verbose=False,
        chat_format="llama-3",
    )

    with STATE._lock:
        STATE.llm = llm_gpu
        STATE.llm_on_gpu = True

    log.info("✅ LLM reloaded to GPU.")
    return {"status": "reloaded_to_gpu", "gpu_free_mb": get_gpu_free_mb()}


def llm_chat(messages: List[Dict], temperature: float = 0.8,
             max_tokens: int = 1024, stop: Optional[List[str]] = None) -> str:
    """Run chat completion with the local LLM."""
    with STATE._lock:
        if not STATE.llm_loaded or STATE.llm is None:
            raise RuntimeError("Local LLM not loaded")
        llm = STATE.llm

    result = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop or [],
    )
    return result["choices"][0]["message"]["content"]


# ─────────────────────────── Image Gen (stable-diffusion-cpp) ─────────────────
def load_image_gen(diff_path: str, llm_diff_path: str, vae_path: str,
                   offload_cpu: bool = True, flash_attn: bool = True):
    """Load Z-Image Turbo GGUF."""
    from stable_diffusion_cpp import StableDiffusion

    log.info(f"Loading Z-Image Turbo: {os.path.basename(diff_path)}")
    log.info(f"  LLM component : {os.path.basename(llm_diff_path)}")
    log.info(f"  VAE           : {os.path.basename(vae_path)}")

    sd = StableDiffusion(
        diffusion_model_path=diff_path,
        llm_path=llm_diff_path,
        vae_path=vae_path,
        offload_params_to_cpu=offload_cpu,
        diffusion_flash_attn=flash_attn,
    )

    with STATE._lock:
        STATE.sd = sd
        STATE.sd_loaded = True

    log.info("✅ Z-Image Turbo ready.")


def unload_image_gen():
    """Unload image gen model to free VRAM."""
    with STATE._lock:
        if STATE.sd is not None:
            del STATE.sd
            STATE.sd = None
            STATE.sd_loaded = False
            gc.collect()
            torch_empty_cache()
            log.info("Image gen unloaded from memory.")
    return {"status": "unloaded"}


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    cfg_scale: float = 1.0,
    seed: int = -1,
    init_image: Optional[Image.Image] = None,
    strength: float = 0.75,
    ref_images: Optional[List[Image.Image]] = None,
) -> List[str]:
    """Generate image(s), return list of base64 PNG strings."""
    with STATE._lock:
        if not STATE.sd_loaded or STATE.sd is None:
            raise RuntimeError("Image generation model not loaded")
        sd = STATE.sd

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg_scale,
        sample_method="euler",
        seed=seed,
    )

    if init_image is not None:
        # img2img: use init_image + strength for denoising
        kwargs["init_image"] = init_image
        kwargs["strength"] = strength

    if ref_images:
        # ref_images: style reference for Z-Image/FLUX (does not affect denoising)
        kwargs["ref_images"] = ref_images

    result = sd.generate_image(**kwargs)

    # generate_image always returns List[PIL.Image]
    return [pil_to_b64(img) for img in result]


# ─────────────────────────── HTTP Request Handler ─────────────────────────────
class ModelServerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/health":
            self.send_json({
                "llm": {
                    "loaded": STATE.llm_loaded,
                    "on_gpu": STATE.llm_on_gpu,
                    "model": os.path.basename(STATE.llm_path) if STATE.llm_path else None,
                },
                "image": {
                    "loaded": STATE.sd_loaded,
                    "model": os.path.basename(STATE.diff_path) if STATE.diff_path else None,
                },
                "gpu_free_mb": get_gpu_free_mb(),
            })
        elif self.path == "/llm/status":
            self.send_json({
                "loaded": STATE.llm_loaded,
                "on_gpu": STATE.llm_on_gpu,
                "model": os.path.basename(STATE.llm_path) if STATE.llm_path else None,
            })
        elif self.path == "/image/status":
            self.send_json({
                "loaded": STATE.sd_loaded,
                "model": os.path.basename(STATE.diff_path) if STATE.diff_path else None,
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            payload = self.read_json()
        except Exception as e:
            self.send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        path = self.path

        # ── LLM endpoints ──────────────────────────────────────────────────────
        if path == "/llm/chat":
            if not STATE.use_local_llm:
                self.send_json({"error": "Local LLM disabled (use_local_llm=false)"}, 400)
                return
            try:
                messages = payload.get("messages", [])
                temperature = float(payload.get("temperature", 0.8))
                max_tokens = int(payload.get("max_tokens", 1024))
                stop = payload.get("stop", None)
                text = llm_chat(messages, temperature=temperature,
                                max_tokens=max_tokens, stop=stop)
                self.send_json({"content": text})
            except Exception as e:
                log.error(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)

        elif path == "/llm/offload":
            if not STATE.use_local_llm:
                self.send_json({"status": "not_applicable"})
                return
            try:
                result = offload_llm_to_cpu()
                self.send_json(result)
            except Exception as e:
                log.error(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)

        elif path == "/llm/reload":
            if not STATE.use_local_llm:
                self.send_json({"status": "not_applicable"})
                return
            try:
                result = reload_llm_to_gpu()
                self.send_json(result)
            except Exception as e:
                log.error(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)

        # ── Image generation endpoints ─────────────────────────────────────────
        elif path in ("/image/txt2img", "/image/img2img",
                      "/txt2img", "/img2img"):          # legacy compat aliases
            try:
                prompt = payload.get("prompt", "")
                neg    = payload.get("negative_prompt", "")
                width  = int(payload.get("width", 1024))
                height = int(payload.get("height", 1024))
                steps  = int(payload.get("steps", 4))
                cfg    = float(payload.get("cfg_scale", 1.0))
                seed   = int(payload.get("seed", -1))

                init_img  = None
                ref_imgs  = None
                strength  = float(payload.get("denoising_strength", 0.75))

                if path in ("/image/img2img", "/img2img"):
                    raw_list = payload.get("init_images", [])
                    if raw_list:
                        init_img = b64_to_pil(raw_list[0])

                # ref_images: optional style-reference list (Z-Image / FLUX Kontext)
                raw_refs = payload.get("ref_images", [])
                if raw_refs:
                    ref_imgs = [b64_to_pil(r) for r in raw_refs]

                images_b64 = generate_image(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=width, height=height,
                    steps=steps, cfg_scale=cfg, seed=seed,
                    init_image=init_img, strength=strength,
                    ref_images=ref_imgs,
                )
                self.send_json({"images": images_b64})
            except Exception as e:
                log.error(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)

        else:
            self.send_json({"error": f"Unknown endpoint: {path}"}, 404)


# ─────────────────────────── Background loader ───────────────────────────────
def background_load(args):
    """
    Load models in order:
      1. Local LLM first (will use GPU if n_gpu_layers > 0)
      2. Then Z-Image Turbo (uses whatever VRAM remains + offload flag)
    """
    if STATE.use_local_llm and args.llm_path:
        try:
            load_llm(
                path=args.llm_path,
                n_gpu_layers=args.llm_gpu_layers,
                n_ctx=args.llm_ctx,
                verbose=args.llm_verbose,
            )
        except Exception as e:
            log.error(f"Failed to load local LLM: {e}")
            log.error(traceback.format_exc())

    try:
        # For z-image we need all 3 components
        STATE.diff_path = args.diff_path
        load_image_gen(
            diff_path=args.diff_path,
            llm_diff_path=args.llm_diff_path,
            vae_path=args.vae_path,
            offload_cpu=not args.no_offload_cpu,
            flash_attn=not args.no_flash_attn,
        )
    except Exception as e:
        log.error(f"Failed to load image gen: {e}")
        log.error(traceback.format_exc())


# ─────────────────────────── Entry point ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Unified model server: local uncensored LLM + Z-Image Turbo GGUF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Image gen args
    parser.add_argument("--diff_path",     required=True,
                        help="Path to z-image-turbo diffusion GGUF")
    parser.add_argument("--llm_diff_path", required=True,
                        help="Path to LLM component GGUF used inside z-image (e.g. Qwen3-4B)")
    parser.add_argument("--vae_path",      required=True,
                        help="Path to VAE safetensors (ae.safetensors)")
    parser.add_argument("--no_offload_cpu", action="store_true",
                        help="Don't offload diffusion params to CPU (requires more VRAM)")
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Disable flash attention for diffusion model")

    # Local LLM args
    parser.add_argument("--llm_path",
                        default=None,
                        help="Path to local uncensored LLM GGUF "
                             "(default: bartowski/Llama-3.2-3B-Instruct-uncensored Q4_K_M)")
    parser.add_argument("--no_local_llm",  action="store_true",
                        help="Disable local LLM entirely (use OpenAI via interpreter)")
    parser.add_argument("--llm_gpu_layers", type=int, default=99,
                        help="Number of LLM layers to keep on GPU (0=CPU only, 99=all)")
    parser.add_argument("--llm_ctx",        type=int, default=4096,
                        help="LLM context window size")
    parser.add_argument("--llm_verbose",    action="store_true",
                        help="Enable verbose llama.cpp output")

    # Server args
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)

    args = parser.parse_args()

    STATE.use_local_llm = not args.no_local_llm
    STATE.offload_cpu   = not args.no_offload_cpu
    STATE.flash_attn    = not args.no_flash_attn

    # Start model loading in background thread
    loader = threading.Thread(target=background_load, args=(args,), daemon=True)
    loader.start()

    # Start HTTP server
    server = HTTPServer((args.host, args.port), ModelServerHandler)
    log.info(f"🚀 model-server listening at http://{args.host}:{args.port}")
    log.info(f"   Local LLM  : {'enabled' if STATE.use_local_llm else 'DISABLED'}")
    log.info(f"   LLM model  : {os.path.basename(args.llm_path) if args.llm_path else 'none'}")
    log.info(f"   Diff model : {os.path.basename(args.diff_path)}")
    log.info("")
    log.info("Endpoints:")
    log.info("  POST /llm/chat       — generate text with local uncensored LLM")
    log.info("  POST /llm/offload    — move LLM weights to CPU (free VRAM)")
    log.info("  POST /llm/reload     — restore LLM to GPU")
    log.info("  GET  /llm/status     — LLM state")
    log.info("  POST /image/txt2img  — text to image")
    log.info("  POST /image/img2img  — image to image")
    log.info("  GET  /image/status   — image gen state")
    log.info("  GET  /health         — full system status + GPU free MB")
    log.info("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()