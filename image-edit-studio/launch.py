# ============================================================
# 🎨 Qwen Image Edit — Launch Cell
# ============================================================
# Paste this into a Colab cell after your dependency install cell.
# It clones the repo (or you can upload the files), then launches.
# ============================================================

# Option A: If you put the files in Google Drive or uploaded them:
# !cp -r /content/drive/MyDrive/qwen-studio /content/qwen-studio

# Option B: If you have them in the colab already at /content/qwen-studio
# (just skip this)

# Option C: Write them inline (uncomment if needed):
# !mkdir -p /content/qwen-studio/static
# ... then write files ...

# ---------- LAUNCH ----------
%cd /content/qwen-studio
!python server.py
