"""CPU-only validation and aspect-preserving image preparation."""
import base64
import binascii
import io
import math
from PIL import Image, ImageOps, UnidentifiedImageError


def decode_image(data):
    if not isinstance(data, str) or len(data) > 28_000_000:
        raise ValueError("Choose an image smaller than 20 MB.")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Choose an image smaller than 40 megapixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError("Could not read this image. Try a valid PNG, JPEG or WebP file.") from None


def validate_inputs(body):
    if not isinstance(body, dict) or not isinstance(body.get("images"), dict):
        raise ValueError("Upload at least one image.")
    images = body["images"]
    if not 1 <= len(images) <= 8 or any(not str(k).isdigit() for k in images):
        raise ValueError("Upload between one and eight images.")
    decoded = [decode_image(images[k]) for k in sorted(images, key=int)]
    if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
        raise ValueError("Enter an editing instruction.")
    for key, default, low, high in [("width",512,64,2048),("height",512,64,2048),
                                    ("num_inference_steps",4,1,80),("num_images_per_prompt",1,1,4)]:
        try:
            number = int(body.get(key, default))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Invalid {key}.") from None
        if not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        body[key] = number
    mode = body.get("resize_mode", "auto")
    if mode not in ("auto", "fit", "crop", "stretch"):
        raise ValueError("Choose a valid image sizing mode.")
    body["resize_mode"] = mode
    for key, default, low, high in [("seed", -1, -1, 2**64 - 1), ("mask_blur", 0, 0, 100)]:
        try:
            number = int(body.get(key, default))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Invalid {key}.") from None
        if not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        body[key] = number
    try:
        cfg = float(body.get("true_cfg_scale", 1.0))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid guidance scale.") from None
    if not math.isfinite(cfg) or not 0 <= cfg <= 30:
        raise ValueError("Guidance scale must be between 0 and 30.")
    body["true_cfg_scale"] = cfg
    if body.get("mask"):
        decode_image(body["mask"])  # Fail before reserving GPU time.
    return decoded


def output_size(image, body):
    if body.get("resize_mode", "auto") != "auto":
        return body["width"], body["height"]
    scale = min(math.sqrt((512 * 512) / (image.width * image.height)),
                1024 / max(image.size))
    return tuple(max(64, int(round(d * scale / 32)) * 32) for d in image.size)


def resize_image(image, size, mode="auto"):
    if mode == "stretch":
        return image.resize(size, Image.Resampling.LANCZOS)
    if mode == "crop":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=0)
