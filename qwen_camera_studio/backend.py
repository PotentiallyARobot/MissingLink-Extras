# backend.py — MissingLink Camera Studio
import os,sys,math,gc,io,time,base64,uuid,json,traceback,threading
IN_COLAB="google.colab" in sys.modules
if IN_COLAB:
    from google.colab import output as colab_output
    from google.colab.output import eval_js

_D=os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if not os.path.isdir(os.path.join(_D,"static")):
    for _c in ["/content/qwen_camera_studio",os.path.join(os.getcwd(),"qwen_camera_studio"),os.getcwd()]:
        if os.path.isdir(os.path.join(_c,"static")): _D=_c; break
PROJECT_DIR=_D; STATIC=os.path.join(_D,"static"); UPLOADS=os.path.join(_D,"uploads"); OUTPUTS=os.path.join(_D,"outputs")
for d in [UPLOADS,OUTPUTS,STATIC]: os.makedirs(d,exist_ok=True)
print(f"[backend] project={PROJECT_DIR}")

from flask import Flask,request,jsonify,send_from_directory
app=Flask(__name__,static_folder=STATIC,static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"]=50*1024*1024

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
        # Verify gguf is installed FIRST
        try:
            import gguf
            log(f"gguf package: installed")
        except ImportError:
            raise ImportError(
                "gguf package not installed! Run: pip install 'gguf>=0.10.0' and restart runtime"
            )

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
        p=QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511",transformer=tr,torch_dtype=torch.bfloat16)
        p.scheduler=FlowMatchEulerDiscreteScheduler.from_config(sc)
        p.enable_model_cpu_offload(); p.set_progress_bar_config(disable=None)
        log("Loading Lightning LoRA...")
        p.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning",
            weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",adapter_name="lightning")
        log("Loading Multi-Angles LoRA...")
        p.load_lora_weights("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",adapter_name="angles")
        p.set_adapters(["lightning","angles"],adapter_weights=[1.0,0.9])
        global _qem; _qem=_qem_ref; pipeline=p
        log("Pipeline ready!","success")
    except Exception as e:
        import traceback as tb
        full_tb = tb.format_exc()
        _error=str(e)
        log(f"Pipeline failed: {e}","error")
        log(f"Full traceback:\n{full_tb}","error")
        print(full_tb, file=sys.stderr)
    finally: _loading=False

@app.route("/")
def index(): return send_from_directory(STATIC,"index.html")
@app.route("/api/keepalive")
def keepalive(): return jsonify({"status":"ok"})
@app.route("/api/status")
def status():
    gpu=None;vram=0;vt=0;cpu_pct=0;ram=0;ram_total=0;disk=0;disk_total=0;gpu_temp=0;gpu_util=0
    try:
        _t()
        if torch.cuda.is_available():
            gpu=torch.cuda.get_device_name(0)
            vram=round(torch.cuda.memory_allocated()/1e6)
            vt=round(torch.cuda.get_device_properties(0).total_memory/1e6)
    except: pass
    try:
        import psutil
        cpu_pct=round(psutil.cpu_percent(interval=0))
        mem=psutil.virtual_memory()
        ram=round(mem.used/1e6);ram_total=round(mem.total/1e6)
        dk=psutil.disk_usage('/')
        disk=round(dk.used/1e9,1);disk_total=round(dk.total/1e9,1)
    except: pass
    return jsonify({"ready":pipeline is not None,"loading":_loading,"error":_error,
        "generating":_generating,"progress":_progress,
        "gpu":gpu,"vram":vram,"vram_total":vt,
        "cpu_pct":cpu_pct,"ram":ram,"ram_total":ram_total,
        "disk":disk,"disk_total":disk_total})
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
    _t(); img=Image.open(os.path.join(UPLOADS,fn)).convert("RGB")
    return jsonify({"id":uid,"filename":fn,"w":img.size[0],"h":img.size[1],"url":f"/api/uploads/{fn}"})
@app.route("/api/uploads/<path:fn>")
def srv_up(fn): return send_from_directory(UPLOADS,fn)
@app.route("/api/outputs/<path:fn>")
def srv_out(fn): return send_from_directory(OUTPUTS,fn)
@app.route("/api/generate",methods=["POST"])
def generate():
    global _generating
    if not pipeline: return jsonify({"error":"Model not loaded"}),503
    if _generating: return jsonify({"error":"Busy"}),429
    _generating=True; _progress["active"]=True; _progress["step"]=0
    try:
        _t(); d=request.get_json(); iid=d["image_id"]; prompt=d.get("prompt","<sks> front view eye-level shot medium shot")
        seed=d.get("seed",42)
        if d.get("randomize_seed"): import random; seed=random.randint(0,2147483647)
        cfg=float(d.get("guidance_scale",1.0)); steps=int(d.get("inference_steps",4)); ls=float(d.get("lora_scale",0.9))
        ms=[f for f in os.listdir(UPLOADS) if f.startswith(iid)]
        if not ms: return jsonify({"error":"Image not found"}),404
        inp=Image.open(os.path.join(UPLOADS,ms[0])).convert("RGB"); inp=resize_for_qwen(inp)
        iw,ih=inp.size; _qem.VAE_IMAGE_SIZE=iw*ih
        _progress["total"]=steps
        log(f"Generating: {prompt} | {iw}x{ih} seed={seed} steps={steps}")
        t0=time.time()
        with torch.inference_mode():
            pipeline.set_adapters(["lightning","angles"],adapter_weights=[1.0,ls])
            out=pipeline(image=inp,prompt=prompt,generator=torch.manual_seed(seed),
                num_inference_steps=steps,guidance_scale=1.0,true_cfg_scale=cfg,
                negative_prompt=" ",height=ih,width=iw,callback_on_step_end=_prog_cb)
        el=time.time()-t0; oi=out.images[0]
        of=f"out_{uuid.uuid4().hex[:8]}.png"; oi.save(os.path.join(OUTPUTS,of))
        torch.cuda.empty_cache(); gc.collect()
        log(f"Done {el:.1f}s — {oi.size[0]}x{oi.size[1]}","success")
        return jsonify({"url":f"/api/outputs/{of}","w":oi.size[0],"h":oi.size[1],"prompt":prompt,"seed":seed,"elapsed":round(el,1)})
    except Exception as e:
        log(f"Generate failed: {e}","error"); traceback.print_exc()
        return jsonify({"error":str(e)}),500
    finally: _generating=False; _progress["active"]=False; _progress["step"]=0

@app.route("/api/use_output",methods=["POST"])
def use_output():
    """Copy an output image to uploads so it becomes the new input."""
    _t()
    d=request.get_json(); src=d.get("filename","")
    src_path=os.path.join(OUTPUTS,os.path.basename(src))
    if not os.path.isfile(src_path): return jsonify({"error":"Output not found"}),404
    uid=uuid.uuid4().hex[:8]; fn=f"{uid}.png"
    import shutil; shutil.copy2(src_path,os.path.join(UPLOADS,fn))
    img=Image.open(os.path.join(UPLOADS,fn)).convert("RGB")
    log(f"Output → Input: {fn} ({img.size[0]}x{img.size[1]})")
    return jsonify({"id":uid,"filename":fn,"w":img.size[0],"h":img.size[1],"url":f"/api/uploads/{fn}"})
