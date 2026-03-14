# ============================================================
# 🎨 AI Image Edit Studio — diffusers + GGUF Backend
# ============================================================

import os, sys, traceback

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ.setdefault("PORT", "8000")

STUDIO_PATH = "/content/MissingLink-Extras/image-edit-studio"

sys.path.insert(0, STUDIO_PATH)

try:
    from server import launch
    launch()
except Exception:
    traceback.print_exc()
    print("\n❌ Launch failed. Check the error above.")
