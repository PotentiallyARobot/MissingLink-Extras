# ============================================================
# 🎨 AI Image Edit Studio — GGUF Backend
# ============================================================
# Uses stable_diffusion_cpp with GGUF quantized models.
# Model config lives in config.json (next to server.py).
# ============================================================

import os, subprocess, sys

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
