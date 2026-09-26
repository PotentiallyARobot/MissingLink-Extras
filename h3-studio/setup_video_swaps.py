"""Isolate video dependencies while sharing the notebook's CUDA torch build."""
import os
from pathlib import Path
import subprocess
import sys

SAM_REVISION = '2345a4ad109ac29c569da749c91d84f10dc08c40'
VERSION = 'sam31-vace-1'


def setup():
    root = Path(os.environ.get('H3_VIDEO_ENV', '/content/h3_video_env'))
    marker = root / 'missinglink-version'
    if marker.exists() and marker.read_text() == VERSION:
        return
    # Colab can omit ensurepip. pip's --python option bootstraps the child env.
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(root)], check=True)
    python = root / 'bin/python'
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-deps',
        f'git+https://github.com/facebookresearch/sam3.git@{SAM_REVISION}',
        'diffusers==0.36.0', 'transformers==4.57.6', 'accelerate==1.12.0',
        'huggingface_hub>=0.34,<1.0', 'tokenizers>=0.22,<=0.23',
        'timm>=1.0.17', 'ftfy==6.1.1', 'iopath>=0.1.10', 'portalocker',
        'regex', 'einops', 'pycocotools', 'decord', 'wcwidth', 'safetensors',
        'sentencepiece', 'protobuf', 'opencv-python-headless', 'imageio', 'imageio-ffmpeg'], check=True)
    subprocess.run([str(python), '-c',
        'from sam3.model_builder import build_sam3_multiplex_video_predictor; '
        'from diffusers import WanVACEPipeline; import cv2'], check=True)
    marker.write_text(VERSION)


if __name__ == '__main__':
    setup()
