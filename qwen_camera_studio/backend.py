# backend.py — MissingLink Qwen Studio (hardened)
import os,sys,math,gc,io,time,base64,uuid,json,traceback,threading,hashlib
IN_COLAB="google.colab" in sys.modules
if IN_COLAB:
    from google.colab import output as colab_output
    from google.colab.output import eval_js

_D=os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if not os.path.isdir(os.path.join(_D,"static")):
    for _c in ["/content/qwen_camera_studio",os.path.join(os.getcwd(),"qwen_camera_studio"),os.getcwd()]:
        if os.path.isdir(os.path.join(_c,"static")): _D=_c; break
PROJECT_DIR=_D; STATIC=os.path.join(_D,"static"); UPLOADS=os.path.join(_D,"uploads"); OUTPUTS=os.path.join(_D,"outputs")
STATE_DIR=os.path.join(_D,".state")
VAE_CACHE_DIR=os.path.join(_D,".vae_cache")
for d in [UPLOADS,OUTPUTS,STATIC,STATE_DIR,VAE_CACHE_DIR]: os.makedirs(d,exist_ok=True)

# ── torch.compile cache: must be set BEFORE torch is imported ──
COMPILE_CACHE_DIR="/tmp/torchinductor_root"
os.makedirs(COMPILE_CACHE_DIR,exist_ok=True)
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"]="1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"]="1"
os.environ["TORCHINDUCTOR_CACHE_DIR"]=COMPILE_CACHE_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
print(f"[backend] project={PROJECT_DIR}")
print(f"[backend] compile cache={COMPILE_CACHE_DIR}")

# ── Pre-built compile cache from HuggingFace ──────
_COMPILE_CACHE_REPO="TylerF/MissingLinkModelCache"
_CACHE_TOKEN_URL="https://missinglink.build/api/cache-token"

def _get_cache_token():
    """Get HF token for compile cache access. Tries API endpoint first, falls back to env vars."""
    # 1. Try fetching from API using user's ML token
    _ml_token=os.environ.get("MISSING_LINK_TOKEN") or os.environ.get("ML_TOKEN")
    if not _ml_token and IN_COLAB:
        try:
            from google.colab import userdata
            _ml_token=userdata.get("MISSING_LINK_TOKEN")
        except:
            try: _ml_token=userdata.get("ML_TOKEN")
            except: pass
    if _ml_token:
        try:
            import urllib.request,json as _j
            req=urllib.request.Request(_CACHE_TOKEN_URL,headers={"x-api-key":_ml_token})
            with urllib.request.urlopen(req,timeout=10) as resp:
                data=_j.loads(resp.read())
                if data.get("ok") and data.get("hf_token"):
                    return data["hf_token"]
        except Exception as e:
            print(f"[compile] Cache token API failed: {e}")

    # 2. Fall back to HF_TOKEN env var
    t=os.environ.get("HF_TOKEN")
    if t: return t
    if IN_COLAB:
        try:
            from google.colab import userdata
            return userdata.get("HF_TOKEN")
        except: pass
    return None

def _get_gpu_tag():
    """Get GPU-specific tag for cache lookup."""
    import torch
    if not torch.cuda.is_available(): return None
    gpu=torch.cuda.get_device_name(0).replace(" ","-").replace("/","-")
    cap=torch.cuda.get_device_capability(0)
    return f"{gpu}_sm{cap[0]}{cap[1]}_torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.','')}"

def _inject_torch_key():
    """Pin torch_key so inductor cache lookups match the compile session."""
    try:
        import torch._inductor.codecache as cc
        import base64 as b64
        SAVED_KEY=b64.b64decode("1LeYUP9UUZwkIc+iw44562F1M8GQzyicOOnfGjDIF1E=")
        cc.torch_key=lambda: SAVED_KEY
        print("[compile] torch_key pinned for cache portability")
    except Exception as e:
        print(f"[compile] torch_key injection failed (non-fatal): {e}")

def _enable_inductor_config():
    """Enable inductor cache configs after torch is imported."""
    try:
        import torch._inductor.config as ic
        ic.fx_graph_cache=True
        ic.bundle_triton_into_fx_graph_cache=True
    except Exception as e:
        print(f"[compile] Inductor config setup failed (non-fatal): {e}")

def _download_compile_cache():
    """Download pre-compiled cache artifacts for this GPU.
    
    Downloads two things:
    1. Inductor cache zip (triton kernels + autotuning configs)
    2. Mega-cache binary (portable bundled fxgraph + aotautograd + triton)
    """
    try:
        import torch
        if not torch.cuda.is_available(): return

        # Inject torch_key and enable configs first
        _inject_torch_key()
        _enable_inductor_config()

        tag=_get_gpu_tag()
        if not tag: return

        # Skip if cache already populated
        existing=sum(1 for _,_,fs in os.walk(COMPILE_CACHE_DIR) for f in fs)
        if existing>100:
            print(f"[compile] Cache already populated ({existing} files), loading mega-cache only")
        else:
            # 1. Download and extract inductor cache zip (triton kernels + autotuning)
            from huggingface_hub import hf_hub_download
            import zipfile, shutil

            # Try regional cache first (smaller, faster), fall back to full cache
            for zip_name in [f"compile_cache/{tag}_regional.zip", f"compile_cache/{tag}.zip"]:
                try:
                    print(f"[compile] Downloading: {_COMPILE_CACHE_REPO}/{zip_name}")
                    zip_path=hf_hub_download(
                        repo_id=_COMPILE_CACHE_REPO, filename=zip_name,
                        repo_type="model", token=_get_cache_token())
                    if os.path.exists(COMPILE_CACHE_DIR):
                        shutil.rmtree(COMPILE_CACHE_DIR)
                    os.makedirs(COMPILE_CACHE_DIR, exist_ok=True)
                    with zipfile.ZipFile(zip_path,"r") as zf:
                        zf.extractall(COMPILE_CACHE_DIR)
                    total=sum(1 for _,_,fs in os.walk(COMPILE_CACHE_DIR) for f in fs)
                    print(f"[compile] Extracted {total} cached files")
                    break
                except Exception as e:
                    print(f"[compile] {zip_name} not found: {e}")

        # 2. Download and load mega-cache (portable bundled artifacts)
        try:
            from huggingface_hub import hf_hub_download
            for mc_name in [f"compile_cache/{tag}_megacache_regional.bin",
                           f"compile_cache/{tag}_megacache.bin"]:
                try:
                    print(f"[compile] Downloading mega-cache: {mc_name}")
                    mc_path=hf_hub_download(
                        repo_id=_COMPILE_CACHE_REPO, filename=mc_name,
                        repo_type="model", token=_get_cache_token())
                    with open(mc_path,"rb") as f:
                        artifact_bytes=f.read()
                    print(f"[compile] Loading {len(artifact_bytes)/1e6:.0f}MB mega-cache...")
                    torch.compiler.load_cache_artifacts(artifact_bytes)
                    del artifact_bytes
                    print("[compile] Mega-cache loaded successfully")
                    break
                except Exception as e:
                    print(f"[compile] {mc_name} not found: {e}")
        except Exception as e:
            print(f"[compile] Mega-cache load failed (non-fatal): {e}")

    except Exception as e:
        print(f"[compile] Cache download failed (non-fatal): {e}")

# ── Persistent state files ─────────────────────────
JOBS_FILE=os.path.join(STATE_DIR,"jobs.json")
UI_STATE_FILE=os.path.join(STATE_DIR,"ui_state.json")

from flask import Flask,request,jsonify,send_from_directory,make_response
app=Flask(__name__,static_folder=STATIC,static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"]=50*1024*1024

# ── CORS: allow Colab proxy origins ────────────────
@app.after_request
def _cors(resp):
    origin=request.headers.get("Origin","")
    # Allow Colab proxy origins and localhost
    if origin and ("colab.dev" in origin or "colab.google" in origin
                    or "localhost" in origin or "127.0.0.1" in origin
                    or "googleusercontent.com" in origin):
        resp.headers["Access-Control-Allow-Origin"]=origin
        resp.headers["Access-Control-Allow-Headers"]="Content-Type"
        resp.headers["Access-Control-Allow-Methods"]="GET,POST,OPTIONS"
    return resp

@app.route("/api/<path:p>",methods=["OPTIONS"])
def _preflight(p):
    r=make_response("",204)
    return r

torch=None; Image=None
def _t():
    global torch,Image
    if torch is None: import torch as t; torch=t
    if Image is None: from PIL import Image as I; Image=I

_logs=[]; _lock=threading.Lock()
def log(msg,level="info"):
    e={"t":time.time(),"msg":str(msg),"level":level}
    with _lock:
        _logs.append(e)
        if len(_logs)>500: _logs.pop(0)
    print(f"[{level}] {msg}")

pipeline=None; _qem=None; _loading=False; _error=None; _generating=False
_progress={"step":0,"total":0,"active":False}

# ── Caches for speed ─────────────────────────────
# File lookup cache: image_id -> filepath (invalidated on upload)
_upload_cache={}
_upload_cache_lock=threading.Lock()

# Resized PIL image cache: (filepath, max_size, mul) -> PIL.Image
_resized_cache={}
_MAX_RESIZED_CACHE=20

# Text encoder prompt cache: (prompt, device) -> (prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds)
_prompt_cache={}
_MAX_PROMPT_CACHE=32

# LoRA scaling modules cache: list of modules with scaling dicts (avoid full tree walk)
_lora_modules=None

# VAE latent cache: hash(image_bytes + dims) -> path to .pt file on disk
_VAE_CACHE_MAX_ENTRIES=200
_VAE_CACHE_MAX_GB=10.0

# ── Persistent jobs store ──────────────────────────
_jobs={}
_jobs_lock=threading.Lock()

def _save_jobs():
    """Persist completed/errored jobs to disk so they survive restarts."""
    try:
        with _jobs_lock:
            persist={k:v for k,v in _jobs.items()
                     if v.get("status") in ("done","error")}
        tmp=JOBS_FILE+".tmp"
        with open(tmp,"w") as f:
            json.dump(persist,f)
        os.replace(tmp,JOBS_FILE)
    except Exception as e:
        print(f"[warn] Failed to save jobs: {e}")

def _load_jobs():
    """Restore completed jobs from disk after restart."""
    global _jobs
    try:
        if os.path.isfile(JOBS_FILE):
            with open(JOBS_FILE) as f:
                loaded=json.load(f)
            with _jobs_lock:
                for k,v in loaded.items():
                    if v.get("status")=="done" and v.get("result",{}).get("url"):
                        fn=os.path.basename(v["result"]["url"])
                        if os.path.isfile(os.path.join(OUTPUTS,fn)):
                            _jobs[k]=v
                    elif v.get("status")=="error":
                        _jobs[k]=v
            log(f"Restored {len(_jobs)} jobs from disk","info")
    except Exception as e:
        print(f"[warn] Failed to load jobs: {e}")

_load_jobs()

def _cleanup_jobs():
    """Remove completed jobs older than 30 minutes after being read."""
    now=time.time()
    with _jobs_lock:
        expired=[k for k,v in _jobs.items()
                 if v.get("read_at") and now-v["read_at"]>1800]
        for k in expired: _jobs.pop(k,None)
    if expired:
        _save_jobs()

# ── UI state persistence ──────────────────────────
def _save_ui_state(state):
    try:
        state["_saved_at"]=time.time()
        tmp=UI_STATE_FILE+".tmp"
        with open(tmp,"w") as f:
            json.dump(state,f)
        os.replace(tmp,UI_STATE_FILE)
    except Exception as e:
        print(f"[warn] Failed to save UI state: {e}")

def _load_ui_state():
    try:
        if os.path.isfile(UI_STATE_FILE):
            with open(UI_STATE_FILE) as f:
                return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to load UI state: {e}")
    return None

# ── Cached psutil values (non-blocking) ────────────
_hw_cache={"cpu_pct":0,"ram":0,"ram_total":0,"disk":0,"disk_total":0,"_t":0}
_hw_lock=threading.Lock()

def _update_hw():
    """Update hardware stats in background — never blocks a request."""
    while True:
        try:
            import psutil
            cpu=round(psutil.cpu_percent(interval=1))
            mem=psutil.virtual_memory()
            dk=psutil.disk_usage('/')
            with _hw_lock:
                _hw_cache["cpu_pct"]=cpu
                _hw_cache["ram"]=round(mem.used/1e6)
                _hw_cache["ram_total"]=round(mem.total/1e6)
                _hw_cache["disk"]=round(dk.used/1e9,1)
                _hw_cache["disk_total"]=round(dk.total/1e9,1)
                _hw_cache["_t"]=time.time()
        except: pass
        time.sleep(2)

threading.Thread(target=_update_hw,daemon=True).start()


def resize_for_qwen(img,mx=1024,mul=8):
    _t(); w,h=img.size; s=min(mx/max(w,h),1.0)
    nw=max(round(w*s/mul)*mul,mul); nh=max(round(h*s/mul)*mul,mul)
    return img.resize((nw,nh),Image.LANCZOS) if (nw,nh)!=(w,h) else img

def _prog_cb(pipe,step,ts,kwargs):
    _progress["step"]=step+1; return kwargs

def load_pipeline(variant="Q4_K_M"):
    global pipeline,_qem,_loading,_error
    _t()
    if pipeline: return
    _loading=True; _error=None
    try:
        # Download pre-built compile cache for this GPU (if available)
        _download_compile_cache()

        try:
            import gguf
            log(f"gguf package: installed")
        except ImportError:
            raise ImportError(
                "gguf package not installed! Run: pip install 'gguf>=0.10.0' and restart runtime"
            )

        # bitsandbytes needed for 4-bit text encoder on L4
        try:
            import bitsandbytes
        except ImportError:
            log("Installing bitsandbytes for 4-bit quantization...")
            import subprocess
            subprocess.check_call([sys.executable,"-m","pip","install",
                "bitsandbytes","-q","--break-system-packages"])

        # ── 1. TF32 + cuDNN: free speed on Ampere+ ──
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            torch.set_float32_matmul_precision("high")
            log("TF32 matmul + cuDNN enabled")
        except Exception as e:
            log(f"TF32 setup skipped: {e}","warn")

        # ── 2. cuDNN benchmark: auto-tune conv kernels ──
        try:
            torch.backends.cudnn.benchmark=True
            log("cuDNN benchmark mode enabled")
        except: pass

        from diffusers import (QwenImageEditPlusPipeline,QwenImageTransformer2DModel,
            FlowMatchEulerDiscreteScheduler,GGUFQuantizationConfig)
        from huggingface_hub import hf_hub_download
        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qem
        _qem_ref=qem
        sc={"base_image_seq_len":256,"base_shift":math.log(3),"invert_sigmas":False,
            "max_image_seq_len":8192,"max_shift":math.log(3),"num_train_timesteps":1000,
            "shift":1.0,"shift_terminal":None,"stochastic_sampling":False,
            "time_shift_type":"exponential","use_beta_sigmas":False,"use_dynamic_shifting":True,
            "use_exponential_sigmas":False,"use_karras_sigmas":False}
        gf=f"qwen-image-edit-2511-{variant}.gguf"
        log(f"Loading GGUF transformer ({gf})...")
        gp=hf_hub_download(repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",filename=gf)
        log(f"GGUF path: {gp}")
        log(f"GGUF file size: {os.path.getsize(gp)/1e9:.2f} GB")
        tr=QwenImageTransformer2DModel.from_single_file(gp,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,config="Qwen/Qwen-Image-Edit-2511",subfolder="transformer")
        log("GGUF loaded, building pipeline...")

        # ── Detect GPU tier and load text encoder accordingly ──
        # A100 (80GB): bf16 text encoder, full GPU, compile reduce-overhead
        # L4 (24GB): 4-bit text encoder, full GPU, compile default
        # T4 (16GB): bf16 text encoder, hybrid offload, no compile
        _use_4bit_text_encoder=False
        _text_encoder=None
        vram_total=0
        if torch.cuda.is_available():
            vram_total=torch.cuda.get_device_properties(0).total_memory/1e9
            if 16<vram_total<40:
                # L4 tier: 4-bit text encoder to fit everything on GPU
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
                    log(f"L4 detected ({vram_total:.0f}GB) — loading 4-bit text encoder...")
                    _text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen-Image-Edit-2511",subfolder="text_encoder",
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4"),
                        torch_dtype=torch.bfloat16)
                    _use_4bit_text_encoder=True
                    log(f"4-bit text encoder loaded — {torch.cuda.memory_allocated()/1e9:.1f}GB VRAM","success")
                except Exception as e:
                    log(f"4-bit text encoder failed ({e}), using bf16","warn")

        if _text_encoder:
            p=QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511",
                transformer=tr,text_encoder=_text_encoder,torch_dtype=torch.bfloat16)
        else:
            p=QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511",
                transformer=tr,torch_dtype=torch.bfloat16)
        p.scheduler=FlowMatchEulerDiscreteScheduler.from_config(sc)

        # Load LoRAs first (before offload decisions)
        log("Loading Lightning LoRA...")
        p.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning",
            weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",adapter_name="lightning")
        log("Loading Multi-Angles LoRA...")
        p.load_lora_weights("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",adapter_name="angles")
        p.set_adapters(["lightning","angles"],adapter_weights=[1.0,0.9])

        # ── 3. LoRA fusing ──
        # NOTE: disabled — fuse_lora + unload_lora_weights is destructive
        # with GGUF quantized models. The _set_lora_scales runtime approach
        # is safer and the overhead is small (840 modules, <1ms per call).

        # ── 4. Smart offload: try best option, fall back on OOM ──
        _used_full_gpu=False
        _used_hybrid=False
        if torch.cuda.is_available():
            vram_total=torch.cuda.get_device_properties(0).total_memory/1e9
            log(f"GPU VRAM: {vram_total:.1f} GB")

            # Try 1: everything on GPU
            try:
                p.to("cuda")
                _used_full_gpu=True
                log(f"Full GPU mode — {vram_total:.0f}GB VRAM","success")
            except Exception as e:
                log(f"Full GPU failed ({e}), trying hybrid...","warn")
                torch.cuda.empty_cache(); gc.collect()

                # Try 2: transformer+VAE on GPU, text encoder CPU-offloaded
                try:
                    p.to("cpu")
                    p.enable_model_cpu_offload()
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(p.transformer, recurse=True)
                    p.transformer.to("cuda")
                    remove_hook_from_module(p.vae, recurse=True)
                    p.vae.to("cuda")
                    _used_hybrid=True
                    log(f"Hybrid GPU mode — transformer+VAE on GPU, text encoder offloaded","success")
                    log(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB of {vram_total:.0f}GB")
                except Exception as e:
                    log(f"Hybrid failed ({e}), trying transformer-only...","warn")
                    torch.cuda.empty_cache(); gc.collect()

                    # Try 3: transformer-only on GPU, VAE+text encoder offloaded (T4 16GB)
                    try:
                        p.to("cpu")
                        p.enable_model_cpu_offload()
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(p.transformer, recurse=True)
                        p.transformer.to("cuda")
                        _used_hybrid=True
                        log(f"Transformer-only GPU mode — VAE+text encoder offloaded","success")
                        log(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB of {vram_total:.0f}GB")
                    except Exception as e:
                        log(f"Transformer-only failed ({e}), full CPU offload","warn")
                        torch.cuda.empty_cache(); gc.collect()
                        try: p.to("cpu")
                        except: pass
                        p.enable_model_cpu_offload()
                        log(f"CPU offload mode — {vram_total:.0f}GB VRAM")
        else:
            p.enable_model_cpu_offload()

        p.set_progress_bar_config(disable=None)

        # ── 5. VAE tiling/slicing/channels_last ──
        # NOTE: disabled for Qwen pipeline — enable_vae_tiling() changes
        # internal return structures causing "not enough values to unpack"
        # during decode. Safe to re-enable if future diffusers versions fix this.

        # ── SDPA attention ──
        # DISABLED: Qwen uses custom dual-output attention processor.
        # AttnProcessor2_0 returns single tensor, breaks unpacking.

        # ── torch.compile ──
        # A100 (80GB): reduce-overhead on full transformer — 4x faster
        # L4 (24GB): regional compilation (per-block) — 2x faster, portable cache
        # T4/offload: skip — incompatible with offload hooks
        _original_transformer=p.transformer
        _compiled_ok=False
        _compile_mode=None
        _regional_compile=False

        if hasattr(torch,'compile') and _used_full_gpu:
            if _use_4bit_text_encoder:
                # L4 path: regional compilation — compile individual transformer blocks
                # Much faster first-run (52s vs 674s) and cache is portable via mega-cache
                _compile_mode="default"
                try:
                    log(f"Applying regional torch.compile (mode={_compile_mode}) to {len(p.transformer.transformer_blocks)} blocks...")
                    torch._dynamo.reset()
                    for i,block in enumerate(p.transformer.transformer_blocks):
                        p.transformer.transformer_blocks[i]=torch.compile(block,mode=_compile_mode,fullgraph=False)
                    log(f"Regional torch.compile applied to {len(p.transformer.transformer_blocks)} blocks","success")
                    _compiled_ok=True
                    _regional_compile=True
                except Exception as e:
                    log(f"Regional torch.compile failed: {e}","warn")
                    # Restore original blocks
                    try:
                        for i,block in enumerate(_original_transformer.transformer_blocks):
                            p.transformer.transformer_blocks[i]=block
                    except: pass
            else:
                # A100 path: full transformer compile with reduce-overhead
                _compile_mode="reduce-overhead"
                try:
                    log(f"Applying torch.compile (mode={_compile_mode})...")
                    p.transformer=torch.compile(p.transformer,mode=_compile_mode,fullgraph=False)
                    log(f"torch.compile wrapper applied","success")
                    _compiled_ok=True
                except Exception as e:
                    log(f"torch.compile failed: {e}","warn")
                    p.transformer=_original_transformer
        elif hasattr(torch,'compile'):
            log("torch.compile skipped (CPU offload mode)")

        global _qem; _qem=_qem_ref
        # NOTE: pipeline is NOT set yet — warmup must finish before requests can run
        log("Pipeline ready — running warmup...")

        # ── Warmup ──
        # Regional compile + mega-cache: ~27s warmup (cache pre-populated)
        # Regional compile, no cache: ~52s warmup
        # A100 full compile: uses CUDA graphs warmup
        # No compile: just prime CUDA kernels
        try:
            _cache_files=sum(1 for _,_,fs in os.walk(COMPILE_CACHE_DIR) for f in fs)
            _warmup_dev="cuda" if (_used_full_gpu or _used_hybrid) else "cpu"

            if _compiled_ok and _regional_compile:
                # L4 regional compilation path — warmup runs full pipeline
                # Cache makes this ~27s, uncached ~52s
                if _cache_files>100:
                    log(f"Running regional compile warmup (cache: {_cache_files} files — should be ~27s)...")
                else:
                    log("Running regional compile warmup (no cache — ~52s first time)...")
                _warmup_sz=768
                _qem.VAE_IMAGE_SIZE=_warmup_sz*_warmup_sz
                dummy_img=Image.new("RGB",(_warmup_sz,_warmup_sz),(128,128,128))
                with torch.inference_mode():
                    _=p(image=dummy_img,prompt="warmup",
                        generator=torch.Generator(device=_warmup_dev).manual_seed(0),
                        num_inference_steps=4,guidance_scale=1.0,
                        height=_warmup_sz,width=_warmup_sz)
                torch.cuda.empty_cache()
                log("Regional compile warmup complete","success")

            elif _compiled_ok:
                # A100 path: full compile warmup
                if _cache_files>100:
                    log(f"Running warmup (compile cache: {_cache_files} files — should be fast)...")
                else:
                    log("Running warmup (NO cache — first compile takes ~13min)...")
                _qem.VAE_IMAGE_SIZE=64*64
                dummy_img=Image.new("RGB",(64,64),(128,128,128))
                with torch.inference_mode():
                    _=p(image=dummy_img,prompt="warmup",
                        generator=torch.Generator(device=_warmup_dev).manual_seed(0),
                        num_inference_steps=1,guidance_scale=1.0,
                        height=64,width=64)
                torch.cuda.empty_cache()
                log("Warmup complete","success")

            else:
                # No compile — just prime CUDA kernels
                log("Running warmup pass (priming CUDA kernels)...")
                _qem.VAE_IMAGE_SIZE=64*64
                dummy_img=Image.new("RGB",(64,64),(128,128,128))
                with torch.inference_mode():
                    _=p(image=dummy_img,prompt="warmup",
                        generator=torch.Generator(device=_warmup_dev).manual_seed(0),
                        num_inference_steps=1,guidance_scale=1.0,
                        height=64,width=64)
                torch.cuda.empty_cache()
                log("Warmup complete","success")
        except Exception as e:
            if _compiled_ok:
                log(f"Compiled warmup failed: {e}","warn")
                log("Restoring uncompiled transformer — pipeline still works","warn")
                if _regional_compile:
                    # Restore original blocks
                    try:
                        for i,block in enumerate(_original_transformer.transformer_blocks):
                            p.transformer.transformer_blocks[i]=block
                    except: pass
                else:
                    p.transformer=_original_transformer
                pipeline=p
                _compiled_ok=False
                torch.cuda.empty_cache(); gc.collect()
                # Try warmup again without compile
                try:
                    _qem.VAE_IMAGE_SIZE=64*64
                    dummy_img=Image.new("RGB",(64,64),(128,128,128))
                    with torch.inference_mode():
                        _=p(image=dummy_img,prompt="warmup",
                            generator=torch.Generator(device=_warmup_dev).manual_seed(0),
                            num_inference_steps=1,guidance_scale=1.0,
                            height=64,width=64)
                    log("Uncompiled warmup succeeded","success")
                except: pass
            else:
                log(f"Warmup failed (non-fatal): {e}","warn")

        # Log final optimization summary
        pipeline=p  # NOW requests can flow — warmup is done
        log("Pipeline ready!","success")
        _opts=[]
        if _used_full_gpu: _opts.append("full-GPU")
        elif _used_hybrid: _opts.append("hybrid-GPU(transformer+VAE)")
        else: _opts.append("CPU-offload")
        if _use_4bit_text_encoder: _opts.append("4bit-text-encoder")
        _opts.append("TF32")
        _opts.append("VAE-cache")
        _opts.append("prompt-cache")
        _opts.append("CUDA-gen")
        if _compiled_ok and _regional_compile:
            _opts.append(f"torch.compile-regional({_compile_mode})")
        elif _compiled_ok:
            _opts.append(f"torch.compile({_compile_mode})")
        log(f"Active optimizations: {', '.join(_opts)}","success")

    except Exception as e:
        import traceback as tb
        full_tb = tb.format_exc()
        _error=str(e)
        log(f"Pipeline failed: {e}","error")
        log(f"Full traceback:\n{full_tb}","error")
        print(full_tb, file=sys.stderr)
    finally: _loading=False

def _find_upload(iid):
    with _upload_cache_lock:
        if iid in _upload_cache:
            p=_upload_cache[iid]
            if os.path.isfile(p): return p
            del _upload_cache[iid]
    ms=[f for f in os.listdir(UPLOADS) if f.startswith(iid)]
    if ms:
        p=os.path.join(UPLOADS,ms[0])
        with _upload_cache_lock:
            _upload_cache[iid]=p
        return p
    return None

def _invalidate_upload_cache(iid=None):
    with _upload_cache_lock:
        if iid: _upload_cache.pop(iid,None)
        else: _upload_cache.clear()

def _load_images(image_ids):
    _t(); imgs=[]
    for iid in image_ids:
        fp=_find_upload(iid)
        if fp:
            cache_key=(fp,os.path.getmtime(fp))
            if cache_key in _resized_cache:
                imgs.append(_resized_cache[cache_key])
            else:
                img=Image.open(fp).convert("RGB")
                resized=resize_for_qwen(img)
                if len(_resized_cache)>=_MAX_RESIZED_CACHE:
                    oldest=next(iter(_resized_cache))
                    del _resized_cache[oldest]
                _resized_cache[cache_key]=resized
                imgs.append(resized)
    return imgs

# ── VAE latent caching ────────────────────────────
def _vae_cache_key(img, width, height):
    """Hash PIL image content + target dims to create a stable cache key.
    Uses raw pixel bytes (fast) instead of PNG encoding (slow)."""
    h=hashlib.md5(img.tobytes())
    h.update(f"{img.size[0]}x{img.size[1]}_{width}x{height}".encode())
    return h.hexdigest()

def _vae_cache_path(key):
    return os.path.join(VAE_CACHE_DIR,f"{key}.pt")

def _vae_cache_get(key):
    """Load cached VAE latents from disk. Returns tensor or None."""
    p=_vae_cache_path(key)
    if os.path.isfile(p):
        try:
            _t()
            # Load directly to GPU if available to avoid CPU→GPU copy later
            dev="cuda" if torch.cuda.is_available() else "cpu"
            data=torch.load(p,map_location=dev,weights_only=True)
            os.utime(p)  # touch for LRU
            log(f"VAE cache HIT: {key[:12]}...")
            return data
        except Exception as e:
            log(f"VAE cache load error: {e}","warn")
            try: os.remove(p)
            except: pass
    return None

def _vae_cache_put(key, tensor):
    """Save VAE latents to disk cache."""
    try:
        _vae_cache_evict()
        p=_vae_cache_path(key)
        torch.save(tensor.cpu(),p)
        log(f"VAE cache STORE: {key[:12]}...")
    except Exception as e:
        log(f"VAE cache save error: {e}","warn")

def _vae_cache_evict():
    """Evict oldest entries if cache exceeds limits."""
    try:
        entries=[]
        total_bytes=0
        for fn in os.listdir(VAE_CACHE_DIR):
            if not fn.endswith(".pt"): continue
            fp=os.path.join(VAE_CACHE_DIR,fn)
            st=os.stat(fp)
            entries.append((fp,st.st_mtime,st.st_size))
            total_bytes+=st.st_size
        entries.sort(key=lambda x:x[1])  # oldest first
        while len(entries)>_VAE_CACHE_MAX_ENTRIES or total_bytes>_VAE_CACHE_MAX_GB*1e9:
            if not entries: break
            fp,_,sz=entries.pop(0)
            try: os.remove(fp)
            except: pass
            total_bytes-=sz
    except: pass

def _snap_dim(v,mul=8):
    return max(mul,round(v/mul)*mul)

MAX_PIXELS=1536*1536

def _cap_resolution(w,h):
    total=w*h
    if total<=MAX_PIXELS:
        return _snap_dim(w),_snap_dim(h)
    scale=math.sqrt(MAX_PIXELS/total)
    w=_snap_dim(int(w*scale))
    h=_snap_dim(int(h*scale))
    return w,h

def _set_lora_scales(scales):
    global _lora_modules
    if _lora_modules is None:
        _lora_modules=[m for m in pipeline.transformer.modules()
                       if hasattr(m,'scaling') and isinstance(m.scaling,dict)]
        log(f"Cached {len(_lora_modules)} LoRA scaling modules")
    for module in _lora_modules:
        for name, weight in scales.items():
            if name in module.scaling:
                module.scaling[name]=weight


# ══════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════

@app.route("/")
def index(): return send_from_directory(STATIC,"index.html")

@app.route("/api/keepalive")
def keepalive(): return jsonify({"status":"ok","t":time.time()})

@app.route("/api/status")
def status():
    _cleanup_jobs()
    gpu=None;vram=0;vt=0
    try:
        _t()
        if torch.cuda.is_available():
            gpu=torch.cuda.get_device_name(0)
            vram=round(torch.cuda.memory_allocated()/1e6)
            vt=round(torch.cuda.get_device_properties(0).total_memory/1e6)
    except: pass
    # Use cached psutil values (never blocks)
    with _hw_lock:
        hw=dict(_hw_cache)
    # VAE cache stats
    vae_count=0;vae_mb=0
    try:
        for fn in os.listdir(VAE_CACHE_DIR):
            if fn.endswith(".pt"):
                vae_count+=1; vae_mb+=os.path.getsize(os.path.join(VAE_CACHE_DIR,fn))
        vae_mb=round(vae_mb/1e6,1)
    except: pass
    return jsonify({"ready":pipeline is not None,"loading":_loading,"error":_error,
        "generating":_generating,"progress":_progress,
        "gpu":gpu,"vram":vram,"vram_total":vt,
        "cpu_pct":hw["cpu_pct"],"ram":hw["ram"],"ram_total":hw["ram_total"],
        "disk":hw["disk"],"disk_total":hw["disk_total"],
        "vae_cache":{"count":vae_count,"size_mb":vae_mb}})

@app.route("/api/logs")
def get_logs():
    s=float(request.args.get("since",0))
    with _lock: n=[e for e in _logs if e["t"]>s]
    return jsonify({"logs":n})

@app.route("/api/load",methods=["POST"])
def api_load():
    if pipeline: return jsonify({"status":"loaded"})
    if _loading: return jsonify({"status":"loading"})
    d=request.get_json(silent=True) or {}
    threading.Thread(target=load_pipeline,args=(d.get("variant","Q4_K_M"),),daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/upload",methods=["POST"])
def upload():
    if "image" not in request.files: return jsonify({"error":"No image"}),400
    f=request.files["image"]; ext=os.path.splitext(f.filename)[1].lower() or ".png"
    uid=uuid.uuid4().hex[:8]; fn=f"{uid}{ext}"; f.save(os.path.join(UPLOADS,fn))
    _invalidate_upload_cache()  # new file added, clear lookup cache
    _t(); img=Image.open(os.path.join(UPLOADS,fn)).convert("RGB")
    return jsonify({"id":uid,"filename":fn,"w":img.size[0],"h":img.size[1],"url":f"/api/uploads/{fn}"})

@app.route("/api/uploads/<path:fn>")
def srv_up(fn): return send_from_directory(UPLOADS,fn)
@app.route("/api/outputs/<path:fn>")
def srv_out(fn): return send_from_directory(OUTPUTS,fn)

# ── Persistent UI state endpoints ──────────────────
@app.route("/api/state",methods=["GET"])
def get_state():
    state=_load_ui_state() or {}
    uploads=[]
    for fn in sorted(os.listdir(UPLOADS)):
        uid=fn.split(".")[0]
        uploads.append({"id":uid,"filename":fn,"url":f"/api/uploads/{fn}"})
    outputs=[]
    for fn in sorted(os.listdir(OUTPUTS)):
        outputs.append({"filename":fn,"url":f"/api/outputs/{fn}"})
    state["_uploads"]=uploads
    state["_outputs"]=outputs
    with _jobs_lock:
        completed={k:v for k,v in _jobs.items() if v.get("status") in ("done","error")}
    state["_completed_jobs"]=completed
    return jsonify(state)

@app.route("/api/state",methods=["POST"])
def save_state():
    d=request.get_json(silent=True) or {}
    d.pop("_uploads",None)
    d.pop("_outputs",None)
    d.pop("_completed_jobs",None)
    _save_ui_state(d)
    return jsonify({"status":"ok"})

@app.route("/api/validate_refs",methods=["POST"])
def validate_refs():
    d=request.get_json(silent=True) or {}
    image_ids=d.get("image_ids",[])
    output_files=d.get("output_files",[])
    valid_images={}
    for iid in image_ids:
        fp=_find_upload(iid)
        if fp:
            fn=os.path.basename(fp)
            valid_images[iid]={"filename":fn,"url":f"/api/uploads/{fn}"}
    valid_outputs={}
    for fn in output_files:
        bf=os.path.basename(fn)
        if os.path.isfile(os.path.join(OUTPUTS,bf)):
            valid_outputs[fn]={"url":f"/api/outputs/{bf}"}
    return jsonify({"valid_images":valid_images,"valid_outputs":valid_outputs})


@app.route("/api/generate",methods=["POST"])
def generate():
    global _generating
    if not pipeline: return jsonify({"error":"Model not loaded"}),503
    if _generating: return jsonify({"error":"Busy"}),429
    _generating=True; _progress["active"]=True; _progress["step"]=0
    d=request.get_json()
    job_id=uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id]={"status":"running","result":None,"error":None,"started_at":time.time()}
    _save_jobs()
    threading.Thread(target=_run_generate,args=(job_id,d),daemon=True).start()
    return jsonify({"job_id":job_id,"status":"started"})

def _run_generate(job_id,d):
    global _generating
    try:
        _t()
        mode=d.get("mode","camera")
        image_ids=d.get("image_ids",[])
        if not image_ids:
            with _jobs_lock:
                _jobs[job_id]={"status":"error","error":"No images provided","finished_at":time.time()}
            _save_jobs(); return
        prompt=d.get("prompt","")
        seed=d.get("seed",42)
        if d.get("randomize_seed"): import random; seed=random.randint(0,2147483647)
        cfg=float(d.get("guidance_scale",1.0))
        steps=int(d.get("inference_steps",4))
        lora_scale=float(d.get("lora_scale",0.9))

        imgs=_load_images(image_ids)
        if not imgs:
            with _jobs_lock:
                _jobs[job_id]={"status":"error","error":"No valid images found","finished_at":time.time()}
            _save_jobs(); return

        iw,ih=imgs[0].size
        ow=int(d.get("width",0)) or iw
        oh=int(d.get("height",0)) or ih
        ow,oh=_cap_resolution(ow,oh)
        _qem.VAE_IMAGE_SIZE=ow*oh
        _progress["total"]=steps

        if mode=='camera':
            angles_w=lora_scale
            log(f"[camera] {prompt} | in:{iw}x{ih} out:{ow}x{oh} seed={seed} steps={steps} angles={angles_w}")
            img_input=imgs[0]
        else:
            angles_w=0.0
            log(f"[edit] {prompt} | in:{iw}x{ih} out:{ow}x{oh} seed={seed} steps={steps} imgs={len(imgs)}")
            img_input=imgs if len(imgs)>1 else imgs[0]

        _set_lora_scales({"lightning":1.0,"angles":angles_w})

        # ── Text encoder prompt caching ───────────────
        # The text encoder (T5) is slow. Cache its output so repeated
        # prompts skip the full forward pass entirely.
        _orig_encode_prompt=None
        _neg_prompt=" "  # always the same
        _prompt_key=(prompt,_neg_prompt)
        if hasattr(pipeline,'encode_prompt') and _prompt_key in _prompt_cache:
            cached_enc=_prompt_cache[_prompt_key]
            _orig_encode_prompt=pipeline.encode_prompt
            def _cached_encode_prompt(*args,**kwargs):
                log("Text encoder skipped (cached)")
                return cached_enc
            pipeline.encode_prompt=_cached_encode_prompt
        elif hasattr(pipeline,'encode_prompt'):
            _orig_encode_prompt=pipeline.encode_prompt
            def _capturing_encode_prompt(*args,**kwargs):
                result=_orig_encode_prompt(*args,**kwargs)
                try:
                    # Cache the result (detach tensors to avoid graph retention)
                    if isinstance(result,tuple):
                        cached=tuple(r.detach() if hasattr(r,'detach') else r for r in result)
                    else:
                        cached=result.detach() if hasattr(result,'detach') else result
                    if len(_prompt_cache)>=_MAX_PROMPT_CACHE:
                        _prompt_cache.pop(next(iter(_prompt_cache)))
                    _prompt_cache[_prompt_key]=cached
                    log(f"Text encoder output cached (prompt len={len(prompt)})")
                except Exception as e:
                    log(f"Prompt cache capture failed: {e}","warn")
                return result
            pipeline.encode_prompt=_capturing_encode_prompt

        # ── VAE encode caching ────────────────────
        # Only cache in camera mode (single image). In edit mode the pipeline
        # batches multiple images into one vae.encode() call — intercepting
        # that with a single-image cache would return wrong-shaped latents.
        _orig_vae_encode=None
        _use_vae_cache = (mode == 'camera' and len(imgs) == 1)

        if _use_vae_cache:
            primary_img=imgs[0]
            vae_key=_vae_cache_key(primary_img,ow,oh)
            cached_latents=_vae_cache_get(vae_key)
            _first_encode_call=[True]

            if cached_latents is not None:
                _orig_vae_encode=pipeline.vae.encode
                cached_on_device=[None]
                def _cached_encode(x,*args,**kwargs):
                    if _first_encode_call[0]:
                        _first_encode_call[0]=False
                        device=x.device
                        if cached_on_device[0] is None:
                            cached_on_device[0]=cached_latents.to(device)
                        log(f"VAE encode skipped (cached)")
                        class _FakeOutput:
                            def __init__(self,lt): self.latent_dist=self; self.sample_=lt
                            def sample(self): return self.sample_
                            def mode(self): return self.sample_
                            @property
                            def latent(self): return self.sample_
                            @property
                            def mean(self): return self.sample_
                        return _FakeOutput(cached_on_device[0])
                    return _orig_vae_encode(x,*args,**kwargs)
                pipeline.vae.encode=_cached_encode
            else:
                _orig_vae_encode=pipeline.vae.encode
                def _caching_encode(x,*args,**kwargs):
                    result=_orig_vae_encode(x,*args,**kwargs)
                    if _first_encode_call[0]:
                        _first_encode_call[0]=False
                        try:
                            sample=result.latent_dist.mode() if hasattr(result,'latent_dist') else result
                            _vae_cache_put(vae_key,sample.detach())
                        except Exception as e:
                            log(f"VAE cache capture failed: {e}","warn")
                    return result
                pipeline.vae.encode=_caching_encode
        else:
            log(f"VAE cache skipped (edit mode, {len(imgs)} images)")

        t0=time.time()
        # Use CUDA generator only in full-GPU mode; CPU offload needs CPU generator
        try:
            _on_gpu = hasattr(pipeline,'_offload_gpu_id') is False and next(pipeline.transformer.parameters()).is_cuda
        except:
            _on_gpu = False
        if _on_gpu:
            gen=torch.Generator(device="cuda").manual_seed(seed)
        else:
            gen=torch.Generator(device="cpu").manual_seed(seed)
        try:
            with torch.inference_mode():
                out=pipeline(image=img_input,prompt=prompt,generator=gen,
                    num_inference_steps=steps,guidance_scale=1.0,true_cfg_scale=cfg,
                    negative_prompt=" ",height=oh,width=ow,callback_on_step_end=_prog_cb)
        finally:
            # Always restore monkey-patched methods
            if _orig_vae_encode is not None:
                pipeline.vae.encode=_orig_vae_encode
            if _orig_encode_prompt is not None:
                pipeline.encode_prompt=_orig_encode_prompt

        el=time.time()-t0; oi=out.images[0]
        of=f"out_{uuid.uuid4().hex[:8]}.png"
        # Save with minimal PNG compression (default level 6 is slow)
        oi.save(os.path.join(OUTPUTS,of),compress_level=1)

        # Smart VRAM management: only flush if >80% utilized
        try:
            if torch.cuda.is_available():
                vram_used=torch.cuda.memory_allocated()
                vram_total=torch.cuda.get_device_properties(0).total_memory
                if vram_used/vram_total>0.8:
                    torch.cuda.empty_cache(); gc.collect()
        except:
            torch.cuda.empty_cache(); gc.collect()

        _cache_status = "HIT" if (_use_vae_cache and cached_latents is not None) else "MISS" if _use_vae_cache else "N/A"
        log(f"Done {el:.1f}s — {oi.size[0]}x{oi.size[1]} (vae_cache={_cache_status})","success")
        result={
            "url":f"/api/outputs/{of}","w":oi.size[0],"h":oi.size[1],
            "prompt":prompt,"seed":seed,"elapsed":round(el,1),"mode":mode}
        with _jobs_lock:
            _jobs[job_id]={"status":"done","result":result,"finished_at":time.time()}
        _save_jobs()

        # ── Pre-cache output in background thread (camera mode only) ──
        if mode == 'camera':
            _oi_copy=oi.copy()
            def _bg_precache():
                try:
                    if _generating: return
                    _precache_output_vae(_oi_copy,ow,oh)
                except Exception as e:
                    log(f"Pre-cache output failed (non-fatal): {e}","warn")
            threading.Thread(target=_bg_precache,daemon=True).start()

    except Exception as e:
        log(f"Generate failed: {e}","error"); traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id]={"status":"error","error":str(e),"finished_at":time.time()}
        _save_jobs()
    finally:
        _generating=False; _progress["active"]=False; _progress["step"]=0

def _precache_output_vae(output_img,ow,oh):
    """Pre-encode the output image through VAE and cache it.
    When the user clicks 'Use as Input', this image gets loaded,
    resized by resize_for_qwen(), then VAE-encoded. We do that
    encoding now so the next generate is a cache hit."""
    _t()
    resized=resize_for_qwen(output_img)
    key=_vae_cache_key(resized,ow,oh)
    # Skip if already cached (shouldn't happen, but be safe)
    if os.path.isfile(_vae_cache_path(key)):
        return
    # Run the actual VAE encode
    import numpy as np
    img_np=np.array(resized).astype(np.float32)/255.0
    img_tensor=torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)
    img_tensor=img_tensor.to(dtype=pipeline.vae.dtype,device=pipeline.vae.device)
    # Normalize to [-1,1] as VAE expects
    img_tensor=2.0*img_tensor-1.0
    with torch.inference_mode():
        enc=pipeline.vae.encode(img_tensor)
        if hasattr(enc,'latent_dist'):
            latent=enc.latent_dist.mode()
        elif hasattr(enc,'latents'):
            latent=enc.latents
        else:
            latent=enc
        _vae_cache_put(key,latent.detach())
    log(f"Pre-cached output VAE encoding ({resized.size[0]}x{resized.size[1]} → {ow}x{oh})")

@app.route("/api/job/<job_id>")
def get_job(job_id):
    with _jobs_lock:
        j=_jobs.get(job_id)
    if not j: return jsonify({"error":"Job not found"}),404
    if j["status"]=="done":
        if "read_at" not in j: j["read_at"]=time.time()
        return jsonify({"status":"done","result":j["result"]})
    elif j["status"]=="error":
        if "read_at" not in j: j["read_at"]=time.time()
        return jsonify({"status":"error","error":j["error"]})
    return jsonify({"status":"running","progress":_progress})

@app.route("/api/jobs",methods=["GET"])
def list_jobs():
    with _jobs_lock:
        summary={}
        for k,v in _jobs.items():
            summary[k]={"status":v.get("status"),"started_at":v.get("started_at"),
                        "finished_at":v.get("finished_at")}
            if v.get("status")=="done":
                summary[k]["result"]=v.get("result")
            elif v.get("status")=="error":
                summary[k]["error"]=v.get("error")
    return jsonify({"jobs":summary,"generating":_generating})

@app.route("/api/use_output",methods=["POST"])
def use_output():
    _t()
    d=request.get_json(); src=d.get("filename","")
    src_path=os.path.join(OUTPUTS,os.path.basename(src))
    if not os.path.isfile(src_path): return jsonify({"error":"Output not found"}),404
    uid=uuid.uuid4().hex[:8]; fn=f"{uid}.png"
    import shutil; shutil.copy2(src_path,os.path.join(UPLOADS,fn))
    _invalidate_upload_cache()
    img=Image.open(os.path.join(UPLOADS,fn)).convert("RGB")
    log(f"Output → Input: {fn} ({img.size[0]}x{img.size[1]})")
    return jsonify({"id":uid,"filename":fn,"w":img.size[0],"h":img.size[1],"url":f"/api/uploads/{fn}"})

@app.route("/api/vae_cache",methods=["GET"])
def vae_cache_info():
    """Return VAE cache stats."""
    try:
        entries=[]
        total=0
        for fn in os.listdir(VAE_CACHE_DIR):
            if fn.endswith(".pt"):
                fp=os.path.join(VAE_CACHE_DIR,fn)
                sz=os.path.getsize(fp)
                entries.append(fn)
                total+=sz
        return jsonify({"count":len(entries),"size_mb":round(total/1e6,1)})
    except:
        return jsonify({"count":0,"size_mb":0})

@app.route("/api/vae_cache",methods=["DELETE"])
def vae_cache_clear():
    """Clear the VAE latent cache."""
    cleared=0
    try:
        for fn in os.listdir(VAE_CACHE_DIR):
            if fn.endswith(".pt"):
                os.remove(os.path.join(VAE_CACHE_DIR,fn))
                cleared+=1
        _resized_cache.clear()
        _prompt_cache.clear()
        log(f"Cleared VAE cache ({cleared} entries)","info")
    except Exception as e:
        log(f"Cache clear error: {e}","warn")
    return jsonify({"status":"ok","cleared":cleared})

# ── Auto-load pipeline on import ──────────────────────
# Pipeline loads and warms up before the Flask server starts,
# so the UI is immediately ready when it opens.
if IN_COLAB:
    log("Auto-loading pipeline...")
    load_pipeline()
    if pipeline:
        log("Pipeline ready — launching UI","success")
    else:
        log("Pipeline failed to load — UI will show error","error")
