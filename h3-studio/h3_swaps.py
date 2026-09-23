"""Image-only masked edits for H3. Imported lazily; SAM is optional.

SAM 3.1's release adds video tracking. Still images use the official SAM 3
image predictor. No video replacement is advertised by this module.
"""
from __future__ import annotations

import base64
import hashlib
import json
import importlib.util
import io
import os
import tempfile
import threading
import time
import uuid
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000
KINDS = {"face", "head", "body", "clothing", "custom"}


def png(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def read_image(upload, mode="RGB"):
    if upload is None:
        raise ValueError("Choose the required image first.")
    raw = upload.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Each upload must be under 20 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Use a still PNG, JPEG or WebP image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Images must be 16 megapixels or smaller.")
            return ImageOps.exif_transpose(image).convert(mode)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read that image. Use PNG, JPEG or WebP.") from exc


def prepare_mask(mask, size, grow=0, feather=0):
    if mask.size != size:
        raise ValueError("Mask dimensions must match the original image.")
    mask = mask.convert("L").point(lambda v: 255 if v >= 128 else 0)
    if not mask.getbbox():
        raise ValueError("The mask is empty. Select or paint the area to replace.")
    if grow:
        mask = mask.filter(ImageFilter.MaxFilter(2 * grow + 1))
    # Feather inward: the zero region remains exactly unchanged.
    if feather:
        mask = ImageChops.multiply(mask, mask.filter(ImageFilter.GaussianBlur(feather)))
    return mask


def api_mask(mask):
    result = Image.new("RGBA", mask.size, "white")
    result.putalpha(ImageOps.invert(mask))
    return result


def composite(original, generated, mask):
    if generated.size != original.size:
        raise ValueError("The editor returned unexpected dimensions; no stretched result was saved.")
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask)


def public_sam_checkpoint(root):
    """Ungated, pinned fallback for users without private bucket access."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch
    source = "AEmotionStudio/sam3"
    revision = "5eac5d508135b2f19adc3ef095efb7d393236f75"
    directory = root / ("public-" + revision)
    directory.mkdir(exist_ok=True)
    for name in ("LICENSE", "config.json", "sam3.safetensors"):
        downloaded = hf_hub_download(source, name, revision=revision,
                                     local_dir=str(directory), token=False)
    checkpoint = directory / "sam3.pt"
    if not checkpoint.exists():
        weights = load_file(downloaded, device="cpu")
        if not any(key.startswith("detector.") for key in weights):
            raise ValueError("Invalid SAM image checkpoint keys")
        partial = checkpoint.with_suffix(".partial")
        torch.save(weights, partial)
        partial.replace(checkpoint)
    return str(checkpoint)


def sam_checkpoint():
    """Prefer MissingLink's licensed package, with an ungated public fallback."""
    from huggingface_hub import HfApi
    bucket = "MissingLinkBuilder/wheels"
    root = Path(os.environ.get("H3_SAM_CACHE", "/content/h3_sam_weights"))
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest_path = root / "manifest.json"
    try:
        api.download_bucket_files(bucket, files=[("models/sam3/manifest.json", str(manifest_path))])
    except Exception as exc:
        print(f"SAM bucket unavailable ({type(exc).__name__}); using public mirror.")
        return public_sam_checkpoint(root)
    manifest = json.loads(manifest_path.read_text())
    revision = manifest.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Invalid SAM bucket revision.")
    prefix = f"models/sam3/{revision}"
    if manifest.get("source") != "facebook/sam3" or manifest.get("prefix") != prefix:
        raise ValueError("Invalid SAM bucket source.")
    directory = root / revision
    directory.mkdir(exist_ok=True)
    for name in ("LICENSE", "config.json", "sam3.pt"):
        path = directory / name
        expected = manifest["files"][name]
        def valid():
            if not path.is_file() or path.stat().st_size != expected["size"]:
                return False
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
        if not valid():
            partial = directory / (name + ".partial")
            api.download_bucket_files(bucket, files=[(f"{prefix}/{name}", str(partial))])
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if partial.stat().st_size != expected["size"] or digest != expected["sha256"]:
                raise ValueError(f"SAM bucket checksum mismatch: {name}")
            partial.replace(path)
    return str(directory / "sam3.pt")


def segment_image(image, prompt):
    """CPU keeps SAM independent of H3's resident GPU models and queue."""
    dependency_dir = os.environ.get("H3_SWAPS_DEPS", "/content/h3_swaps_deps")
    if dependency_dir not in sys.path:
        sys.path.append(dependency_dir)
    importlib.invalidate_caches()
    if importlib.util.find_spec("sam3") is None:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("setup_swaps.py"))],
                       check=True, timeout=600)
        importlib.invalidate_caches()
    try:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("SAM is not installed. Run the optional SAM setup cell, or paint/upload a mask.") from exc
    # No persistent second model or global autocast context in the H3 process.
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass
    try:
        checkpoint = sam_checkpoint()
        model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint, load_from_HF=False)
    except Exception as exc:
        if "gated" in str(exc).lower() or "403" in str(exc):
            raise RuntimeError("SAM model access is not approved. Request access at huggingface.co/facebook/sam3 using the account for your Colab HF_TOKEN, then retry. You can paint or upload a mask meanwhile.") from exc
        raise
    processor = Sam3Processor(model, device="cpu", confidence_threshold=0.35)
    with torch.inference_mode():
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = result["masks"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy().reshape(-1)
    order = scores.argsort()[::-1][:12]
    return [(Image.fromarray((masks[i].reshape(image.height, image.width) > 0).astype("uint8") * 255),
             float(scores[i])) for i in order]


def register_swaps(app, *, output_dir, api_key, image_request, model="gpt-image-2.5-sunburst", segmenter=None):
    from flask import jsonify, request, send_file
    dependency_dir = os.environ.get("H3_SWAPS_DEPS", "/content/h3_swaps_deps")
    if os.path.isdir(dependency_dir) and dependency_dir not in sys.path:
        sys.path.append(dependency_dir)
    segmenter = segmenter or segment_image
    jobs = {}
    lock = threading.Lock()
    busy = threading.Lock()
    assets = Path(__file__).with_name("swaps")

    @app.get("/swaps/<name>")
    def swaps_asset(name):
        if name not in {"panel.html", "panel.js", "panel.css"}:
            return jsonify(error="Not found"), 404
        return send_file(assets / name)

    @app.get("/api/swaps/status")
    def swaps_status():
        return jsonify(key_available=bool(api_key()), sam_installed=importlib.util.find_spec("sam3") is not None,
                       model=model, media="image", sam_device="cpu")

    @app.get("/api/swaps/jobs/<job_id>")
    def swaps_job(job_id):
        with lock:
            job = jobs.get(job_id)
            return (jsonify(job), 200) if job else (jsonify(error="Edit job expired or not found."), 404)

    def submit(work):
        if not busy.acquire(blocking=False):
            return jsonify(error="Another mask or edit is running. Wait for it to finish."), 409
        job_id = uuid.uuid4().hex
        with lock:
            expired = [k for k, v in jobs.items() if time.time() - v["created"] > 3600]
            for k in expired:
                del jobs[k]
            while len(jobs) >= 32:
                del jobs[next(iter(jobs))]
            jobs[job_id] = dict(status="running", created=time.time())

        def run():
            try:
                result = work()
                with lock:
                    jobs[job_id].update(status="done", **result)
            except Exception as exc:
                with lock:
                    jobs[job_id].update(status="error", error=str(exc))
            finally:
                busy.release()
        threading.Thread(target=run, daemon=True, name="h3-image-edit").start()
        return jsonify(job=job_id), 202

    def check_request():
        # Refuse cross-origin browser posts to endpoints that spend API credits.
        origin = request.headers.get("Origin")
        if (origin and origin.rstrip("/") != request.host_url.rstrip("/")
                and request.headers.get("Sec-Fetch-Site") != "same-origin"):
            raise ValueError("Open this editor from the Studio's own URL.")
        if request.content_length and request.content_length > 3 * MAX_BYTES + 1024 * 1024:
            raise ValueError("The combined upload is too large.")

    @app.post("/api/swaps/mask")
    def swaps_mask():
        try:
            check_request()
            image = read_image(request.files.get("original"))
            prompt = str(request.form.get("target", "")).strip()
            if not prompt or len(prompt) > 200:
                raise ValueError("Describe the region to select in 1–200 characters.")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        def work():
            candidates = segmenter(image, prompt)
            if not candidates:
                raise ValueError("SAM found no matching region. Try a simpler label, or paint/upload a mask.")
            return dict(candidates=[dict(png=base64.b64encode(png(mask)).decode("ascii"), score=score)
                                    for mask, score in candidates])
        return submit(work)

    @app.post("/api/swaps/edit")
    def swaps_edit():
        try:
            check_request()
            key = api_key()
            if not key:
                raise ValueError("Add OPENAI_API_KEY to Colab Secrets (with notebook access) or your environment.")
            original = read_image(request.files.get("original"))
            reference = read_image(request.files.get("reference"))
            mask = read_image(request.files.get("mask"), "L")
            kind = request.form.get("kind", "custom")
            instruction = str(request.form.get("instruction", "")).strip()
            if kind not in KINDS or len(instruction) > 4000:
                raise ValueError("Choose a valid swap type and keep instructions under 4,000 characters.")
            grow, feather = int(request.form.get("grow", 0)), int(request.form.get("feather", 0))
            if not 0 <= grow <= 32 or not 0 <= feather <= 32:
                raise ValueError("Grow and feather must be between 0 and 32 pixels.")
            mask = prepare_mask(mask, original.size, grow, feather)
            quality = request.form.get("quality", "medium")
            if quality not in {"low", "medium", "high"}:
                raise ValueError("Choose low, medium or high quality.")
            destination = Path(output_dir())  # Capture storage selection for this job.
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400

        def work():
            # Fit into a supported canvas without distorting the source. Crop back
            # after editing, then composite at the original upload resolution.
            ratio = original.width / original.height
            canvas_size = (1536, 1024) if ratio > 1.2 else (1024, 1536) if ratio < 0.83 else (1024, 1024)
            fitted = ImageOps.contain(original, canvas_size, Image.Resampling.LANCZOS)
            offset = ((canvas_size[0] - fitted.width) // 2, (canvas_size[1] - fitted.height) // 2)
            canvas = Image.new("RGB", canvas_size, (127, 127, 127))
            canvas.paste(fitted, offset)
            selection = Image.new("L", canvas_size)
            selection.paste(mask.resize(fitted.size, Image.Resampling.NEAREST), offset)
            prompt = (f"Edit image 1 only inside its transparent mask. Replace the selected {kind} using image 2 "
                      "as the replacement reference. Match image 1's pose, perspective, expression and lighting. "
                      "Keep the camera, composition and all unmasked content fixed. Image 2 is a reference, "
                      "not a new scene. Preserve the canvas dimensions and padding. " + instruction)
            with tempfile.TemporaryDirectory(prefix="h3-swap-") as directory:
                first, second, mask_path = [Path(directory) / f"{n}.png" for n in ("original", "reference", "mask")]
                canvas.save(first)
                reference.save(second)
                api_mask(selection).save(mask_path)
                raw, _, _ = image_request(key=key, model=model, prompt=prompt, width=canvas_size[0],
                    height=canvas_size[1], quality=quality, reference_paths=[str(first), str(second)], mask_path=str(mask_path))
            generated = Image.open(io.BytesIO(raw)).convert("RGB")
            if generated.size != canvas_size:
                raise ValueError("The editor returned unexpected dimensions. Try again with a different reference.")
            generated = generated.crop((offset[0], offset[1], offset[0] + fitted.width, offset[1] + fitted.height))
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            result = composite(original, generated, mask)
            destination.mkdir(parents=True, exist_ok=True)
            filename = "swap_" + uuid.uuid4().hex + ".png"
            result.save(destination / filename)
            return dict(file=filename, width=original.width, height=original.height)
        return submit(work)


def inject_swaps(page):
    """Add an independent editor workspace without changing H3 conditioning state."""
    button = '<button id="tab_swaps" class="modetab" type="button">SWAPS / EDIT</button>'
    anchor = '<button id=tab_ref2va class=modetab type=button>REF2VA · REFERENCES</button>'
    if anchor not in page:
        raise RuntimeError("Cannot locate H3 workflow tabs for the Swaps editor.")
    page = page.replace(anchor, anchor + button, 1)
    integration = r'''<style>
    #h3-swaps-workspace{position:fixed;inset:58px 10px 10px;z-index:1100;background:#111218;border:1px solid #353642;border-radius:12px;overflow:hidden;display:none}
    #h3-swaps-workspace iframe{width:100%;height:100%;border:0}
    #h3-swaps-close{position:absolute;right:15px;top:12px;width:auto;padding:8px 12px;z-index:2}
    </style><section id="h3-swaps-workspace" aria-label="Swaps and image editing"><button id="h3-swaps-close" type="button">BACK TO H3 ×</button><iframe title="Swaps / Edit" src="/swaps/panel.html"></iframe></section>
    <script>
    (()=>{const panel=document.getElementById('h3-swaps-workspace'),button=document.getElementById('tab_swaps');
      const close=()=>{panel.style.display='none';button.classList.remove('active');button.setAttribute('aria-selected','false')};
      button.onclick=()=>{panel.style.display='block';button.classList.add('active');button.setAttribute('aria-selected','true')};
      document.getElementById('h3-swaps-close').onclick=close;
      document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});
      window.addEventListener('message',async e=>{
        if(e.origin!==location.origin||e.source!==panel.querySelector('iframe').contentWindow)return;
        if(e.data?.type==='h3-swap-frame'&&/^swap_[a-f0-9]{32}\.png$/.test(e.data.file)&&['first','last'].includes(e.data.kind)){
          try{await assignOutFileToFrame(e.data.file,e.data.kind);close();say('Edited image assigned to '+e.data.kind+' frame')}
          catch(err){panel.querySelector('iframe').contentWindow.postMessage({type:'h3-swap-error',error:String(err)},location.origin)}
        }
      });
    })();</script>'''
    return page.replace("</body>", integration + "</body>", 1)
