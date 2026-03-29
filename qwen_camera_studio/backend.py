# backend.py — MissingLink Qwen Studio (hardened)
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
STATE_DIR=os.path.join(_D,".state")
for d in [UPLOADS,OUTPUTS,STATIC,STATE_DIR]: os.makedirs(d,exist_ok=True)
print(f"[backend] project={PROJECT_DIR}")

# ── Persistent state files ─────────────────────────
JOBS_FILE=os.path.join(STATE_DIR,"jobs.json")
UI_STATE_FILE=os.path.join(STATE_DIR,"ui_state.json")

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

def _find_upload(iid):
    ms=[f for f in os.listdir(UPLOADS) if f.startswith(iid)]
    return os.path.join(UPLOADS,ms[0]) if ms else None

def _load_images(image_ids):
    _t(); imgs=[]
    for iid in image_ids:
        fp=_find_upload(iid)
        if fp:
            img=Image.open(fp).convert("RGB")
            imgs.append(resize_for_qwen(img))
    return imgs

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
    transformer = pipeline.transformer
    for module in transformer.modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            for name, weight in scales.items():
                if name in module.scaling:
                    module.scaling[name] = weight

@app.route("/")
def index(): return send_from_directory(STATIC,"index.html")
@app.route("/api/keepalive")
def keepalive(): return jsonify({"status":"ok"})

@app.route("/api/status")
def status():
    _cleanup_jobs()
    gpu=None;vram=0;vt=0;cpu_pct=0;ram=0;ram_total=0;disk=0;disk_total=0
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

# ── Persistent UI state endpoints ──────────────────
@app.route("/api/state",methods=["GET"])
def get_state():
    """Return saved UI state + inventory of available files."""
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
    """Save UI state from the client."""
    d=request.get_json(silent=True) or {}
    d.pop("_uploads",None)
    d.pop("_outputs",None)
    d.pop("_completed_jobs",None)
    _save_ui_state(d)
    return jsonify({"status":"ok"})

@app.route("/api/validate_refs",methods=["POST"])
def validate_refs():
    """Client sends image IDs; backend confirms which still exist on disk."""
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

        t0=time.time()
        with torch.inference_mode():
            out=pipeline(image=img_input,prompt=prompt,generator=torch.manual_seed(seed),
                num_inference_steps=steps,guidance_scale=1.0,true_cfg_scale=cfg,
                negative_prompt=" ",height=oh,width=ow,callback_on_step_end=_prog_cb)
        el=time.time()-t0; oi=out.images[0]
        of=f"out_{uuid.uuid4().hex[:8]}.png"; oi.save(os.path.join(OUTPUTS,of))
        torch.cuda.empty_cache(); gc.collect()
        log(f"Done {el:.1f}s — {oi.size[0]}x{oi.size[1]}","success")
        result={
            "url":f"/api/outputs/{of}","w":oi.size[0],"h":oi.size[1],
            "prompt":prompt,"seed":seed,"elapsed":round(el,1),"mode":mode}
        with _jobs_lock:
            _jobs[job_id]={"status":"done","result":result,"finished_at":time.time()}
        _save_jobs()
    except Exception as e:
        log(f"Generate failed: {e}","error"); traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id]={"status":"error","error":str(e),"finished_at":time.time()}
        _save_jobs()
    finally:
        _generating=False; _progress["active"]=False; _progress["step"]=0

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
    """List all known jobs (for recovery after reconnect)."""
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
    img=Image.open(os.path.join(UPLOADS,fn)).convert("RGB")
    log(f"Output → Input: {fn} ({img.size[0]}x{img.size[1]})")
    return jsonify({"id":uid,"filename":fn,"w":img.size[0],"h":img.size[1],"url":f"/api/uploads/{fn}"})
