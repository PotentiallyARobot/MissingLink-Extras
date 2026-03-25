# ============================================================================
# Qwen Camera Studio — Flask Backend
# ============================================================================
# Loads Qwen-Image-Edit-2511 GGUF + Lightning + Multi-Angles LoRA
# Serves a web UI with 3D camera control for multi-angle image generation
# ============================================================================

import os, sys, math, gc, io, time, base64, uuid, json, traceback
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

# ── Detect Colab ───────────────────────────────────────────────────────────
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import output as colab_output
    from google.colab.output import eval_js

# ── Flask app ──────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# ── Global pipeline state ──────────────────────────────────────────────────
pipeline = None
_qwen_edit_module = None
_pipeline_loading = False
_pipeline_error = None
_generating = False


def resize_for_qwen(image, max_edge=1024, multiple=8):
    """Resize preserving aspect ratio, dims rounded to multiple of 8."""
    w, h = image.size
    scale = min(max_edge / max(w, h), 1.0)
    new_w = max(round(w * scale / multiple) * multiple, multiple)
    new_h = max(round(h * scale / multiple) * multiple, multiple)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def load_pipeline(gguf_variant="Q4_K_M"):
    """Load the full GGUF + Lightning + Angles pipeline."""
    global pipeline, _qwen_edit_module, _pipeline_loading, _pipeline_error

    if pipeline is not None:
        return
    _pipeline_loading = True
    _pipeline_error = None

    try:
        from diffusers import (
            QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
            FlowMatchEulerDiscreteScheduler,
            GGUFQuantizationConfig,
        )
        from huggingface_hub import hf_hub_download
        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qem

        _qwen_edit_module = qem

        # Lightning scheduler config
        sched_cfg = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

        # 1) Download & load GGUF transformer
        gguf_file = f"qwen-image-edit-2511-{gguf_variant}.gguf"
        print(f"[Pipeline] Downloading {gguf_file}...")
        gguf_path = hf_hub_download(
            repo_id="unsloth/Qwen-Image-Edit-2511-GGUF", filename=gguf_file
        )
        print(f"[Pipeline] Loading GGUF transformer...")
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config="Qwen/Qwen-Image-Edit-2511",
            subfolder="transformer",
        )

        # 2) Build pipeline
        print("[Pipeline] Loading pipeline components...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        # 3) Load LoRAs
        print("[Pipeline] Loading Lightning LoRA...")
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            adapter_name="lightning",
        )
        print("[Pipeline] Loading Multi-Angles LoRA...")
        pipe.load_lora_weights(
            "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
            adapter_name="angles",
        )
        pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 0.9])

        pipeline = pipe
        print("[Pipeline] ✅ Ready!")

    except Exception as e:
        _pipeline_error = str(e)
        traceback.print_exc()
    finally:
        _pipeline_loading = False


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/keepalive")
def keepalive():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/api/status")
def status():
    return jsonify({
        "pipeline_ready": pipeline is not None,
        "pipeline_loading": _pipeline_loading,
        "pipeline_error": _pipeline_error,
        "generating": _generating,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    })


@app.route("/api/load_pipeline", methods=["POST"])
def api_load_pipeline():
    if pipeline is not None:
        return jsonify({"status": "already_loaded"})
    if _pipeline_loading:
        return jsonify({"status": "loading"})

    import threading
    data = request.get_json(silent=True) or {}
    variant = data.get("gguf_variant", "Q4_K_M")
    threading.Thread(target=load_pipeline, args=(variant,), daemon=True).start()
    return jsonify({"status": "started", "variant": variant})


@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower() or ".png"
    uid = str(uuid.uuid4())[:8]
    fname = f"{uid}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    f.save(path)

    img = Image.open(path).convert("RGB")
    w, h = img.size

    return jsonify({
        "id": uid,
        "filename": fname,
        "width": w,
        "height": h,
        "url": f"/api/uploads/{fname}",
    })


@app.route("/api/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/generate", methods=["POST"])
def generate():
    global _generating

    if pipeline is None:
        return jsonify({"error": "Pipeline not loaded. Click 'Load Model' first."}), 503
    if _generating:
        return jsonify({"error": "Already generating. Please wait."}), 429

    _generating = True
    try:
        data = request.get_json()
        image_id = data.get("image_id")
        azimuth = data.get("azimuth", "front view")
        elevation = data.get("elevation", "eye-level shot")
        distance = data.get("distance", "medium shot")
        seed = data.get("seed", 42)
        randomize_seed = data.get("randomize_seed", False)
        guidance_scale = data.get("guidance_scale", 1.0)
        inference_steps = data.get("inference_steps", 4)
        lora_scale = data.get("lora_scale", 0.9)

        if randomize_seed:
            import random
            seed = random.randint(0, 2147483647)

        # Find input image
        matches = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(image_id)]
        if not matches:
            return jsonify({"error": f"Image {image_id} not found"}), 404
        img_path = os.path.join(UPLOAD_DIR, matches[0])
        input_image = Image.open(img_path).convert("RGB")

        # Resize preserving aspect ratio
        input_image = resize_for_qwen(input_image, max_edge=1024)
        img_w, img_h = input_image.size

        # Patch VAE_IMAGE_SIZE bug
        _qwen_edit_module.VAE_IMAGE_SIZE = img_w * img_h

        # Update LoRA scale
        pipeline.set_adapters(["lightning", "angles"], adapter_weights=[1.0, lora_scale])

        # Build prompt
        prompt = f"<sks> {azimuth} {elevation} {distance}"
        print(f"[Generate] {prompt} | seed={seed} steps={inference_steps} cfg={guidance_scale} | {img_w}x{img_h}")

        t0 = time.time()
        with torch.inference_mode():
            output = pipeline(
                image=input_image,
                prompt=prompt,
                generator=torch.manual_seed(seed),
                num_inference_steps=int(inference_steps),
                guidance_scale=1.0,
                true_cfg_scale=float(guidance_scale),
                negative_prompt=" ",
                height=img_h,
                width=img_w,
            )
        elapsed = time.time() - t0

        out_img = output.images[0]
        out_fname = f"out_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        out_img.save(out_path)

        torch.cuda.empty_cache()
        gc.collect()

        return jsonify({
            "url": f"/api/outputs/{out_fname}",
            "width": out_img.size[0],
            "height": out_img.size[1],
            "prompt": prompt,
            "seed": seed,
            "elapsed": round(elapsed, 1),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        _generating = False
