# ============================================================
# 🎨 AI Image Edit Studio — GGUF Backend
# ============================================================
# Uses stable_diffusion_cpp with GGUF quantized models.
# Model config lives in config.json (next to server.py).
# ============================================================

import os, subprocess, sys

# ── Kill Xet storage BEFORE anything imports huggingface_hub ──
# Xet causes downloads to stall/hang on large GGUF files.
# Belt AND suspenders: env var + uninstall the package.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Uninstall hf_xet if present (some HF hub versions ignore the env var)
try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "hf_xet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("🗑️  Uninstalled hf_xet (prevents download stalls)")
except Exception:
    pass  # not installed, fine

# ── Server settings ──────────────────────────────────────────
os.environ["PORT"] = "8000"

# ── Path to the image-edit-studio folder ─────────────────────
STUDIO_PATH = "/content/MissingLink-Extras/image-edit-studio"

# ═════════════════════════════════════════════════════════════
# Don't edit below this line
# ═════════════════════════════════════════════════════════════
sys.path.insert(0, STUDIO_PATH)
from server import launch
launch()
