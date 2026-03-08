"""
z_image_server.py
=================
Launches a local HTTP server wrapping the stable-diffusion-cpp Z-Image Turbo GGUF model.
Exposes /txt2img and /img2img endpoints compatible with the PromptFlow interpreter.

Usage:
    python z_image_server.py \\
        --diff_path /path/to/z-image-turbo-Q4_K_M.gguf \\
        --llm_path /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\
        --vae_path /path/to/ae.safetensors \\
        --port 7860
"""

import io
import base64
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Model container (loaded once, shared across requests)
# ──────────────────────────────────────────────────────────────────────────────
MODEL = {"sd": None}
MODEL_LOCK = threading.Lock()


def load_model(diff_path: str, llm_path: str, vae_path: str,
               offload_cpu: bool = True, flash_attn: bool = True):
    from stable_diffusion_cpp import StableDiffusion
    print(f"[z-image-server] Loading model...")
    print(f"  Diffusion: {diff_path}")
    print(f"  LLM      : {llm_path}")
    print(f"  VAE      : {vae_path}")
    sd = StableDiffusion(
        diffusion_model_path=diff_path,
        llm_path=llm_path,
        vae_path=vae_path,
        offload_params_to_cpu=offload_cpu,
        diffusion_flash_attn=flash_attn,
    )
    MODEL["sd"] = sd
    print("[z-image-server] ✅ Model loaded and ready.")
    return sd


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
class ZImageHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[z-image-server] {fmt % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ready = MODEL["sd"] is not None
            self.send_json({"status": "ready" if ready else "loading", "model": "z-image-turbo"})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except Exception as e:
            self.send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        if self.path not in ("/txt2img", "/img2img"):
            self.send_json({"error": "Unknown endpoint"}, 404)
            return

        sd = MODEL["sd"]
        if sd is None:
            self.send_json({"error": "Model not loaded yet"}, 503)
            return

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "")
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        steps = int(payload.get("steps", 4))
        cfg_scale = float(payload.get("cfg_scale", 1.0))
        seed = int(payload.get("seed", -1))

        try:
            with MODEL_LOCK:
                if self.path == "/img2img":
                    init_images = payload.get("init_images", [])
                    init_img = b64_to_pil(init_images[0]) if init_images else None
                    strength = float(payload.get("denoising_strength", 0.75))

                    images = sd.img_to_img(
                        image=init_img,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        strength=strength,
                        seed=seed,
                    )
                else:
                    images = sd.txt_to_img(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        seed=seed,
                    )

            result_images = [pil_to_b64(img) for img in images]
            self.send_json({"images": result_images})

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[z-image-server] ERROR: {tb}")
            self.send_json({"error": str(e)}, 500)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run_server(host: str = "0.0.0.0", port: int = 7860,
               diff_path: str = "", llm_path: str = "", vae_path: str = "",
               offload_cpu: bool = True, flash_attn: bool = True):

    # Load model in background thread so server starts immediately
    def _load():
        load_model(diff_path, llm_path, vae_path, offload_cpu, flash_attn)

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    server = HTTPServer((host, port), ZImageHandler)
    print(f"[z-image-server] 🚀 Server running at http://{host}:{port}")
    print(f"[z-image-server]   POST /txt2img  — text to image")
    print(f"[z-image-server]   POST /img2img  — image to image")
    print(f"[z-image-server]   GET  /health   — status check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[z-image-server] Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Image Turbo GGUF local server")
    parser.add_argument("--diff_path", required=True, help="Path to diffusion GGUF file")
    parser.add_argument("--llm_path", required=True, help="Path to LLM GGUF file")
    parser.add_argument("--vae_path", required=True, help="Path to VAE safetensors file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no_offload_cpu", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        diff_path=args.diff_path,
        llm_path=args.llm_path,
        vae_path=args.vae_path,
        offload_cpu=not args.no_offload_cpu,
        flash_attn=not args.no_flash_attn,
    )
