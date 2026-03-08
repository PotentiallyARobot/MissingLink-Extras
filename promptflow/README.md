# PromptFlow Engine — Game Terrain Generator

A lightweight interpreter for `.promptflow` config files that chains LLM thinking with local image generation (Z-Image Turbo GGUF) to produce complete sets of game asset images.

---

## Project Structure

```
promptflow_engine/
├── terrain_generator.ipynb                 ← Main notebook (start here)
├── model_server.py                         ← Unified HTTP server: local LLM + Z-Image Turbo
├── z_image_server.py                       ← Standalone Z-Image server (alternative)
├── interpreter/
│   ├── __init__.py
│   └── interpreter.py                      ← PromptFlow interpreter engine
├── promptflows/
│   └── game_terrain_generator.promptflow   ← Flow definition (YAML)
├── output/                                 ← Generated images written here
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install stable-diffusion-cpp-python huggingface_hub rich pyyaml requests pillow openai
```

### 2. Run via notebook

Open `terrain_generator.ipynb` and run all cells. Configure in Cell 5:
- `GAME_CONTEXT` — describe your game world
- `TERRAIN_INSTRUCTIONS` — specific terrain requirements
- `TERRAIN_ASSETS_TO_MAKE` — how many tiles to generate (default: 25)
- `SAMPLE_IMAGE` — path to a reference terrain image (optional, enables img2img)
- `OPENAI_API_KEY` — only needed if you switch to `provider: openai` in the .promptflow

### 3. Run via CLI

```bash
# Start the unified model server first
python model_server.py \
  --diff_path   ~/.cache/.../z-image-turbo-Q4_K_M.gguf \
  --llm_diff_path ~/.cache/.../Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --vae_path    ~/.cache/.../ae.safetensors \
  --llm_path    ~/.cache/.../Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf \
  --port 7860

# Then run the interpreter
python -m interpreter.interpreter \
  promptflows/game_terrain_generator.promptflow \
  --game_context 'A sci-fi RTS set on a desert canyon planet' \
  --terrain_asset_instructions 'Include sand dunes, mesa tops, canyon edges, oasis tiles' \
  --terrain_assets_to_make 25 \
  --model_server_url http://localhost:7860 \
  --sample_image sample_terrain.png \
  --out_dir output/my_game_terrain
```

---

## How It Works

### The PromptFlow Format

`.promptflow` files are YAML configs that define a pipeline of steps:

| Step Type       | What it does |
|-----------------|-------------|
| `llm_call`      | Sends a single prompt to the LLM and stores the raw text response |
| `llm_call_loop` | Calls the LLM once per item in a list; collects raw text into a list |
| `code`          | Executes arbitrary Python within the flow context |
| `code_loop`     | Iterates over a list, runs actions (image_gen, log, display_image) per item |

### Key Design Principle

The local 3B LLM is creative but unreliable for structured output. It is **only** asked to write free-form descriptive text. All JSON building, counting, looping, and manifest construction is done entirely in `code` steps.

### Flow for `game_terrain_generator.promptflow`

```
Step 1: describe_art_style        [llm_call]
  └─ LLM writes 2-3 sentences about the visual style

Step 2: list_terrain_categories   [llm_call]
  └─ LLM writes terrain type names, one per line

Step 3: build_tile_plan           [code]
  └─ Parses categories, distributes N tiles across them evenly

Step 4: write_tile_descriptions   [llm_call_loop]
  └─ LLM writes ONE vivid sentence per tile (25 calls)

Step 5: assemble_asset_manifest   [code]
  └─ Combines tile plan + LLM sentences → full image prompts

Step 6: generate_images           [code_loop]
  └─ GPU lifecycle: offload LLM → CPU
  └─ For each asset: generate image, save PNG, display inline

Step 7: save_manifest             [code]
  └─ Writes terrain_manifest.json for game engine integration
```

### GPU Lifecycle

On a 16GB T4, both models can't run at full capacity simultaneously. The flow config declares:

```yaml
gpu_lifecycle:
  offload_before_image_gen: true
  reload_after_image_gen: false
```

Before Step 6 (image generation), the interpreter automatically calls `POST /llm/offload` to move the LLM to CPU, freeing ~2GB VRAM for Z-Image Turbo.

### Template Variables

In `.promptflow` files, use `{{...}}` for dynamic values:

- `{{inputs.key}}` — CLI/config inputs
- `{{steps.step_id.output_key}}` — output from a previous step
- `{{tile.category}}` — current loop item fields
- `{{loop.index}}` / `{{loop.total}}` — loop position
- `{{value | slugify}}` — string filter

---

## Sample Reference Image

If you provide `--sample_image`, the interpreter sends it as an img2img reference to Z-Image Turbo, ensuring all generated tiles share the same perspective, platform shape, lighting, and visual style.

---

## Output

After running, `output/` contains:
- `terrain_001_grassland_01.png` through `terrain_025_*.png`
- `terrain_manifest.json` — machine-readable metadata for game engine integration

```json
{
  "total_assets": 25,
  "art_style": "photorealistic cinematic CGI...",
  "categories": ["Grassland", "Rocky Cliff", ...],
  "assets": [
    {
      "id": "terrain_001",
      "category": "Grassland",
      "name": "Grassland 01",
      "filename": "terrain_001_grassland_01.png",
      "prompt": "..."
    }
  ]
}
```
