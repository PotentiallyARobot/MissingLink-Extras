# ============================================================
# 🎨 AI Image Edit Studio — diffusers + GGUF Backend
# ============================================================

import os, subprocess, sys

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# ── Install dependencies (skip if already present) ────────────
_deps = [
    ("huggingface_hub[hf_transfer]", "huggingface_hub"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("gguf", "gguf"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
    ("Pillow", "PIL"),
    ("requests", "requests"),
    ("fastapi", "fastapi"),
    ("uvicorn[standard]", "uvicorn"),
    ("peft", "peft"),
]
for pkg, mod in _deps:
    try:
        __import__(mod)
    except ImportError:
        print(f"📦 Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ── Server settings ──────────────────────────────────────────
os.environ["PORT"] = "8000"

# ── Path to the image-edit-studio folder ─────────────────────
STUDIO_PATH = "/content/MissingLink-Extras/image-edit-studio"

# ═════════════════════════════════════════════════════════════
sys.path.insert(0, STUDIO_PATH)
from server import launch
launch()
