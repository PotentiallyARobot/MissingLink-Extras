# Swaps / Edit

The H3 Studio adds a Swaps / Edit workspace next to the existing Keyframes and
References buttons. Version 1 edits still images, then hands the result to H3 as
a first or last frame. It does not edit existing video footage.

1. Launch the updated full Extras checkout with `h3_studio_3.py`.
2. Add `OPENAI_API_KEY` in Colab Secrets (enable notebook access), or the environment.
3. Automatic masks: run the notebook's SAM setup cell, then launch/restart
   Studio. `python setup_swaps.py` installs the extra packages separately without
   replacing torch, torchvision, or numpy. Manual masks need no SAM installation.
4. Open **Swaps / Edit**, upload the original and a replacement reference, choose
   a region, and generate a mask. Select a detected candidate, refine with paint/
   erase/undo, or upload a same-size black-and-white mask (white means replace).
5. Generate, compare against the original, download the PNG, or use it as an H3
   first/last frame.

## Behavior and limitations

- Replacement generation uses `gpt-image-2.5-sunburst` via the OpenAI image-edit
  API, using the Studio's existing server-side key lookup. Both images and the
  mask are sent to OpenAI; API billing is separate from H3 access.
- SAM 3.1 is a video tracking release. Still-image masks use the official SAM 3
  image predictor. It runs on CPU to avoid colliding with H3's GPU residency or
  queue. First use downloads public bucket weights and can be slow. This implementation
  has not been benchmarked for SAM inference latency on Colab.
- Face/head/clothing options provide selection prompts and editing instructions;
  they are not specialized identity encoders or a guarantee of exact likeness.
- Images are fitted into a supported generation canvas without stretching, then
  cropped back to their original aspect ratio and resolution. The generated
  region is composited over the original. Pixels outside the expanded mask are
  kept exactly; feathering is inward. Expansion defaults to zero.
- Input limit: 20 MB and 16 MP per image. Up to 12 detected mask candidates are
  returned. Only one SAM/edit job runs at once; H3's queue remains independent.
- Results are saved as `swap_<id>.png` in the active H3 output directory. Source
  and reference files sent to the provider are temporary and cleaned up.
- Install/copy `h3_swaps.py` and the `swaps/` asset directory alongside the main
  script. The notebook downloads the full repository. Encrypted distribution
  packages must be rebuilt by the existing release process before publishing.

## Validation

`python -m unittest discover -s h3-studio -p test_h3_swaps.py -v`

Tests use a fake image provider and segmentation adapter: no API spend or model
downloads. They cover mask polarity, exact preservation outside the mask,
provider errors, concurrency, invalid inputs, assets, and tab integration.
Real SAM inference and live paid image generation require the configured Colab
runtime and credentials; unit tests do not establish visual swap quality.
# MissingLink SAM bucket

The loader checks the public `MissingLinkBuilder/wheels` bucket for
`models/sam3/manifest.json`. Until that package is published, it falls back to
the official `facebook/sam3` download and requires approved model access.
Bucket downloads include the original SAM license and verify SHA256 checksums
before loading a checkpoint. An HF token is not sent for public bucket reads.

With bucket write access, run `python h3-studio/publish_sam_bucket.py` with the
normal HF token. Add `--model sam3.1` to publish the video tracking checkpoint.
Public downloads come from `AEmotionStudio/sam3` and `AEmotionStudio/sam3.1`;
no source-access approval or read token is needed. The image safetensors are
converted without changing tensor names or values to the official builder's
torch checkpoint format. The original safetensors, config and license are
included alongside hashes and exact mirror provenance. The script checks source
LFS hashes, uploads the package under `models/<model>/<revision>/`, and
publishes the manifest pointer last. It does not change bucket visibility.
Redistribution remains under Meta's SAM License, included with the files.
