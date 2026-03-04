# ============================================================
# 🎨 AI Image Edit Studio — Missing Link
# ============================================================
# Model config lives in config.json (next to server.py).
# Edit config.json to change models, then run this cell.
# ============================================================

import os

# ── Server settings ──────────────────────────────────────────
os.environ["PORT"] = "8000"

# ── Path to the image-edit-studio folder ─────────────────────
STUDIO_PATH = "/content/MissingLink-Extras/image-edit-studio"

# ═════════════════════════════════════════════════════════════
# Don't edit below this line
# ═════════════════════════════════════════════════════════════
import subprocess, sys
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])

sys.path.insert(0, STUDIO_PATH)
from server import launch
launch()
