# ============================================================
# 🎨 AI Image Edit Studio — Missing Link (Launch Cell)
# ============================================================
# Paste this into a Colab cell after your dependency install cell.
# Files should be at /content/qwen-studio/
# ============================================================

import sys
sys.path.insert(0, '/content/qwen-studio')
from server import launch
launch()
