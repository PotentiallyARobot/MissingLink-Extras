# ═══════════════════════════════════════════════════════════════════
#  FULL TEST: all on GPU + VAE slicing/tiling
#  RESTART RUNTIME FIRST
# ═══════════════════════════════════════════════════════════════════
import os, shutil, base64, math, time, subprocess, sys, threading, zipfile

CACHE_DIR = "/tmp/torchinductor_root"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Torch / CUDA perf flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Pin torch_key
import torch._inductor.codecache as cc
cc.torch_key = lambda: base64.b64decode("1LeYUP9UUZwkIc+iw44562F1M8GQzyicOOnfGjDIF1E=")
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.bundle_triton_into_fx_graph_cache = True

# Download caches
from huggingface_hub import hf_hub_download, login
from google.colab import userdata
login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)

gpu = torch.cuda.get_device_name(0).replace(" ", "-").replace("/", "-")
cap = torch.cuda.get_device_capability(0)
tag = f"{gpu}_sm{cap[0]}{cap[1]}_torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.','')}"

print("Downloading inductor cache...")
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

zp = hf_hub_download(
    repo_id="TylerF/MissingLinkModelCache",
    filename=f"compile_cache/{tag}_regional.zip",
    repo_type="model",
)
with zipfile.ZipFile(zp, "r") as zf:
    zf.extractall(CACHE_DIR)
n = sum(1 for _, _, fs in os.walk(CACHE_DIR) for _ in fs)
print(f"  {n} files extracted")

print("Downloading mega-cache...")
mc = hf_hub_download(
    repo_id="TylerF/MissingLinkModelCache",
    filename=f"compile_cache/{tag}_megacache_regional.bin",
    repo_type="model",
)
with open(mc, "rb") as f:
    artifact_bytes = f.read()
print(f"  {len(artifact_bytes)/1e6:.0f}MB loaded")
torch.compiler.load_cache_artifacts(artifact_bytes)
del artifact_bytes
print("✅ All caches loaded\n")

# Heartbeat
_alive = True
_status = ["init"]

def _hb():
    t0 = time.time()
    while _alive:
        m, s = divmod(int(time.time() - t0), 60)
        alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        print(f"  💓 [{m:02d}:{s:02d}] {_status[0]} | alloc:{alloc:.2f}GB reserved:{reserved:.2f}GB", flush=True)
        time.sleep(30)

threading.Thread(target=_hb, daemon=True).start()

# Deps
for pkg in ["bitsandbytes", "gguf"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q", "--break-system-packages"])

# Load pipeline
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qem

_status[0] = "loading transformer"
gp = hf_hub_download(
    repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
    filename="qwen-image-edit-2511-Q4_K_M.gguf",
)
tr = QwenImageTransformer2DModel.from_single_file(
    gp,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Qwen/Qwen-Image-Edit-2511",
    subfolder="transformer",
)

_status[0] = "loading 4-bit text encoder"
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    subfolder="text_encoder",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

_status[0] = "building pipeline"
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=tr,
    text_encoder=te,
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
})

_status[0] = "loading LoRAs"
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Edit-2511-Lightning",
    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    adapter_name="lightning",
)
pipe.load_lora_weights(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
    adapter_name="angles",
)
pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 0.9])
pipe.set_progress_bar_config(disable=True)

# Keep entire pipeline on GPU
_status[0] = "moving pipeline to cuda"
pipe.to("cuda")

# Enable VAE memory-saving features
_status[0] = "enabling vae slicing/tiling"
if hasattr(pipe, "vae") and pipe.vae is not None:
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        print("✅ VAE tiling enabled")
    else:
        print("⚠️ VAE tiling not available")

    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
        print("✅ VAE slicing enabled")
    else:
        print("⚠️ VAE slicing not available")

torch.cuda.empty_cache()
print(f"After placement | alloc: {torch.cuda.memory_allocated()/1e9:.2f}GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

# Regional compilation — compile each block individually
_status[0] = "compiling blocks"
torch._dynamo.reset()
for i, block in enumerate(pipe.transformer.transformer_blocks):
    pipe.transformer.transformer_blocks[i] = torch.compile(
        block,
        mode="default",
        fullgraph=False,
    )
print(f"Compiled {len(pipe.transformer.transformer_blocks)} blocks\n")

# ---- test settings ----
TEST_W = 1536
TEST_H = 1536
NUM_RUNS = 3

dummy = Image.new("RGB", (TEST_W, TEST_H), (128, 128, 128))

print(f"Generating {TEST_W}x{TEST_H} x{NUM_RUNS} with all modules on GPU + VAE slicing/tiling\n")
print(f"Before gen | alloc: {torch.cuda.memory_allocated()/1e9:.2f}GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB\n")

results = []
last_result = None

for i in range(NUM_RUNS):
    _status[0] = f"gen {i+1}/{NUM_RUNS}"

    # Set VAE_IMAGE_SIZE dynamically to match the current generation resolution
    qem.VAE_IMAGE_SIZE = TEST_W * TEST_H

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode():
        result = pipe(
            image=dummy,
            prompt="<sks> front view eye-level shot",
            num_inference_steps=4,
            guidance_scale=1.0,
            height=TEST_H,
            width=TEST_W,
            generator=torch.Generator("cuda").manual_seed(42 + i),
        )

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    img = result.images[0]
    save_path = f"/content/output_{TEST_W}x{TEST_H}_sliced_{i+1}.png"
    img.save(save_path)

    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    results.append({
        "run": i + 1,
        "seconds": elapsed,
        "alloc_gb": alloc,
        "reserved_gb": reserved,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "path": save_path,
    })

    print(
        f"  Gen {i+1}: {elapsed:.2f}s | "
        f"alloc:{alloc:.2f}GB reserved:{reserved:.2f}GB | "
        f"peak alloc:{peak_alloc:.2f}GB peak reserved:{peak_reserved:.2f}GB | "
        f"saved:{save_path}"
    )

    torch.cuda.reset_peak_memory_stats()
    last_result = result

from IPython.display import display
display(last_result.images[0])

_alive = False
print(f"\nFinal VRAM | alloc: {torch.cuda.memory_allocated()/1e9:.2f}GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
