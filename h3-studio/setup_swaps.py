"""Optional Colab setup. Does not replace H3's torch/torchvision/numpy stack."""
import os
import subprocess
import sys
from pathlib import Path

target = Path(os.environ.get("H3_SWAPS_DEPS", "/content/h3_swaps_deps"))
target.mkdir(parents=True, exist_ok=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--target", str(target), "--no-deps",
    "git+https://github.com/facebookresearch/sam3.git",
    "timm>=1.0.17", "ftfy==6.1.1", "iopath>=0.1.10", "portalocker", "regex",
    "einops", "pycocotools", "decord", "wcwidth",
])
print("SAM image dependencies installed. Restart the Studio to activate them.")
print("SAM weights download automatically from the public MissingLink bucket on first use.")
print("Set OPENAI_API_KEY for replacement generation. Manual masks work without SAM.")
