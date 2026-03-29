// app.js — Qwen Studio client (hardened)
const $=id=>document.getElementById(id);

let imageId=null,imageUrl=null,ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;
let currentMode='camera';
let activeJobId=null;

// Edit mode: 3 image slots
let slots=[null,null,null];
let lastResult=null;
let lastOutputFilename=null;

// Debounce timer for saving state
let _saveTimer=null;

// ── Session persistence (dual: server + localStorage) ───
function saveSession(){
    // Debounce: wait 300ms of inactivity before actually saving
    if(_saveTimer) clearTimeout(_saveTimer);
    _saveTimer=setTimeout(_doSaveSession,300);
}

function _doSaveSession(){
    const s=_gatherState();
    // Save to localStorage (survives tab close, fast)
    try{ localStorage.setItem('qwen_studio',JSON.stringify(s)); }catch(e){}
    // Save to server (survives full Colab restart, async fire-and-forget)
    _saveToServer(s);
}

function _gatherState(){
    return {
        mode:currentMode,
        imageId, imageUrl,
        slots,
        activeJobId,
        lastOutputFilename,
        promptInput:$('promptInput')?$('promptInput').value:'',
        customPrompt:$('customPrompt')?$('customPrompt').value:'',
        seed:$('rngSeed')?$('rngSeed').value:'42',
        cfg:$('rngCfg')?$('rngCfg').value:'1',
        steps:$('rngSteps')?$('rngSteps').value:'4',
        lora:$('rngLora')?$('rngLora').value:'0.9',
        width:$('inpWidth')?$('inpWidth').value:'',
        height:$('inpHeight')?$('inpHeight').value:'',
        chkRand:$('chkRand')?$('chkRand').checked:true,
        lastResult:lastResult,
        promptLocked:window._promptLocked,
    };
}

async function _saveToServer(s){
    try{
        await fetch('/api/state',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify(s)
        });
    }catch(e){/* network blip, ignore */}
}

async function restoreSession(){
    // Strategy: try server state first (survives Colab restart),
    // fall back to localStorage (survives page refresh with same origin)
    let s=null;
    try{
        const r=await fetch('/api/state');
        if(r.ok){
            const serverState=await r.json();
            // Server state has _uploads, _outputs, _completed_jobs metadata
            // plus the UI state fields if previously saved
            if(serverState.mode){
                s=serverState;
                s._fromServer=true;
            }
        }
    }catch(e){}

    if(!s){
        try{
            const raw=localStorage.getItem('qwen_studio');
            if(raw) s=JSON.parse(raw);
        }catch(e){}
    }

    if(!s) return;

    // If restoring from server, validate that referenced files still exist
    if(s._fromServer){
        const uploadIds=new Set((s._uploads||[]).map(u=>u.id));
        const outputFiles=new Set((s._outputs||[]).map(o=>o.filename));

        // Validate camera input
        if(s.imageId && !uploadIds.has(s.imageId)){
            s.imageId=null; s.imageUrl=null;
        }

        // Validate edit slots
        if(s.slots){
            for(let i=0;i<s.slots.length;i++){
                if(s.slots[i] && !uploadIds.has(s.slots[i].id)){
                    s.slots[i]=null;
                }
            }
        }

        // Validate last result
        if(s.lastResult && s.lastResult.url){
            const fn=s.lastResult.url.split('/').pop();
            if(!outputFiles.has(fn)){
                s.lastResult=null;
                s.lastOutputFilename=null;
            }
        }

        // Check for completed jobs we might have missed
        const completedJobs=s._completed_jobs||{};
        if(s.activeJobId && completedJobs[s.activeJobId]){
            const job=completedJobs[s.activeJobId];
            if(job.status==='done' && job.result){
                s.lastResult=job.result;
                s.lastOutputFilename=(job.result.url||'').split('/').pop();
                s.activeJobId=null;
            } else if(job.status==='error'){
                s.activeJobId=null;
            }
        }
    } else {
        // localStorage restore: validate against server
        try{
            const refs={image_ids:[],output_files:[]};
            if(s.imageId) refs.image_ids.push(s.imageId);
            if(s.slots){
                s.slots.forEach(sl=>{if(sl&&sl.id) refs.image_ids.push(sl.id);});
            }
            if(s.lastOutputFilename) refs.output_files.push(s.lastOutputFilename);

            if(refs.image_ids.length||refs.output_files.length){
                const vr=await fetch('/api/validate_refs',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify(refs)
                });
                if(vr.ok){
                    const v=await vr.json();
                    if(s.imageId && !v.valid_images[s.imageId]){
                        s.imageId=null; s.imageUrl=null;
                    }
                    if(s.slots){
                        for(let i=0;i<s.slots.length;i++){
                            if(s.slots[i] && !v.valid_images[s.slots[i].id]){
                                s.slots[i]=null;
                            }
                        }
                    }
                    if(s.lastOutputFilename && !v.valid_outputs[s.lastOutputFilename]){
                        s.lastResult=null; s.lastOutputFilename=null;
                    }
                }
            }
        }catch(e){/* validation failed, proceed with what we have */}
    }

    // Apply state to UI
    _applyState(s);
}

function _applyState(s){
    try{
        if(s.mode && s.mode!==currentMode) setMode(s.mode);

        // Camera input
        if(s.imageId && s.imageUrl){
            imageId=s.imageId; imageUrl=s.imageUrl;
            $('dropzone').style.display='none';
            $('inputPreview').style.display='';
            $('inputImg').src=s.imageUrl;
            setViewportImage(s.imageUrl);
        }

        // Edit slots
        if(s.slots){
            for(let i=0;i<3;i++){
                slots[i]=s.slots[i]||null;
                renderSlot(i);
            }
        }

        // Settings
        if(s.promptInput && $('promptInput')) $('promptInput').value=s.promptInput;
        if(s.customPrompt && $('customPrompt')) $('customPrompt').value=s.customPrompt;
        if(s.seed){ $('rngSeed').value=s.seed; $('vSeed').textContent=s.seed; }
        if(s.cfg){ $('rngCfg').value=s.cfg; $('vCfg').textContent=s.cfg; }
        if(s.steps){ $('rngSteps').value=s.steps; $('vSteps').textContent=s.steps; }
        if(s.lora){ $('rngLora').value=s.lora; $('vLora').textContent=s.lora; }
        if(s.width) $('inpWidth').value=s.width;
        if(s.height) $('inpHeight').value=s.height;
        if(s.chkRand!==undefined) $('chkRand').checked=s.chkRand;
        if(s.promptLocked){
            window._promptLocked=true;
            $('promptLock').classList.add('locked');
            $('promptLock').textContent='🔒';
        }

        lastOutputFilename=s.lastOutputFilename||null;

        // Restore last result image
        if(s.lastResult){
            lastResult=s.lastResult;
            showResult(s.lastResult);
        }

        // Resume polling if a job was active
        if(s.activeJobId){
            activeJobId=s.activeJobId;
            resumePolling(s.activeJobId);
        }

        updateGenButton();
    }catch(e){ console.warn('Session restore failed:',e); }
}

// ── Init ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
    initViewport();initDrop();initSlots();pollStatus();pollLogs();
    setInterval(pollStatus,2500);setInterval(pollLogs,1500);
    // Periodic state save as safety net (every 30s)
    setInterval(()=>_saveToServer(_gatherState()),30000);
    restoreSession();
});

// Save state before user leaves
window.addEventListener('beforeunload',()=>{
    const s=_gatherState();
    try{ localStorage.setItem('qwen_studio',JSON.stringify(s)); }catch(e){}
    // Use sendBeacon for reliable save on close
    try{
        navigator.sendBeacon('/api/state',
            new Blob([JSON.stringify(s)],{type:'application/json'}));
    }catch(e){}
});

// ── Mode switching (instant, UI only) ─────────────
function setMode(mode){
    currentMode=mode;
    const isCam=mode==='camera';
    $('btnModeCamera').classList.toggle('active',isCam);
    $('btnModeEdit').classList.toggle('active',!isCam);
    document.querySelectorAll('.camera-only').forEach(el=>el.style.display=isCam?'':'none');
    document.querySelectorAll('.edit-only').forEach(el=>el.style.display=isCam?'none':'');
    $('outPh').textContent=isCam
        ?'Upload an image and generate'
        :'Add image(s) and describe your edit';
    updateGenButton();
    saveSession();
}

function updateGenButton(){
    if(activeJobId){ $('btnGen').disabled=true; return; }
    if(currentMode==='camera'){
        $('btnGen').disabled=!(ready && imageId);
    } else {
        const filled=slots.filter(s=>s!==null);
        const cp=$('customPrompt')?$('customPrompt').value.trim():'';
        $('btnGen').disabled=!(ready && filled.length>0 && cp);
    }
}

// ── Status ────────────────────────────────────────
async function pollStatus(){
    try{
        const r=await fetch('/api/status'),d=await r.json();
        const dot=$('connDot'),lbl=$('connLabel');
        if(d.ready){
            dot.className='dot on'; lbl.textContent='Connected'; ready=true;
            updateGenButton();
        } else if(d.loading){
            dot.className='dot'; dot.style.background='var(--gold)';
            lbl.textContent='Loading model...'; ready=false; $('btnGen').disabled=true;
        } else if(d.error){
            dot.className='dot'; dot.style.background='var(--red)';
            lbl.textContent='Error'; ready=false; $('btnGen').disabled=true;
        } else {
            dot.className='dot'; lbl.textContent='Waiting...'; ready=false; $('btnGen').disabled=true;
        }
        if(d.gpu){
            $('hwGpu').textContent=d.gpu;
            $('hwVram').textContent=`${d.vram} / ${d.vram_total} MB`;
            const vpct=d.vram_total?Math.round(d.vram/d.vram_total*100):0;
            $('hwVramBar').style.width=vpct+'%';
        }
        $('hwCpu').textContent=d.cpu_pct+'%';
        $('hwCpuBar').style.width=d.cpu_pct+'%';
        if(d.ram_total){
            $('hwRam').textContent=`${d.ram} / ${d.ram_total} MB`;
            $('hwRamBar').style.width=Math.round(d.ram/d.ram_total*100)+'%';
        }
        if(d.disk_total){
            $('hwDisk').textContent=`${d.disk} / ${d.disk_total} GB`;
            $('hwDiskBar').style.width=Math.round(d.disk/d.disk_total*100)+'%';
        }
        $('gpuInfo').innerHTML=`<span class="gi-dot" style="background:var(--green)"></span> ${d.gpu||'—'} · VRAM ${d.vram}/${d.vram_total}MB · CPU ${d.cpu_pct}% · RAM ${d.ram}/${d.ram_total}MB`;
        if(d.generating && d.progress.active){
            $('genOverlay').classList.add('active');
            const pct=d.progress.total?Math.round(d.progress.step/d.progress.total*100):0;
            $('genFill').style.width=pct+'%';
            $('genText').textContent=`Step ${d.progress.step}/${d.progress.total}`;
        }
    }catch(e){
        // Server unreachable — show disconnected state
        const dot=$('connDot'),lbl=$('connLabel');
        dot.className='dot'; dot.style.background='var(--red)';
        lbl.textContent='Disconnected';
        ready=false; $('btnGen').disabled=true;
    }
}

// ── Logs ──────────────────────────────────────────
async function pollLogs(){
    try{
        const r=await fetch(`/api/logs?since=${lastLogT}`),d=await r.json();
        const body=$('consoleBody');
        d.logs.forEach(l=>{
            const div=document.createElement('div');
            div.className='log-line'+(l.level==='error'?' error':l.level==='success'?' success':'');
            const ts=new Date(l.t*1000).toLocaleTimeString();
            div.innerHTML=`<span class="log-ts">${ts}</span>${esc(l.msg)}`;
            body.appendChild(div);
            lastLogT=Math.max(lastLogT,l.t);
        });
        if(d.logs.length) body.scrollTop=body.scrollHeight;
    }catch(e){}
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

// ── Console / HW toggle ──────────────────────────
function toggleConsole(){
    consoleOpen=!consoleOpen;
    document.documentElement.style.setProperty('--console-h',consoleOpen?'200px':'0px');
    $('btnConsole').classList.toggle('active',consoleOpen);
    $('btnConsole').textContent=consoleOpen?'▾ Console':'▸ Console';
}
function toggleHW(){
    hwOpen=!hwOpen;
    $('hwPanel').classList.toggle('open',hwOpen);
    $('btnGpu').classList.toggle('active',hwOpen);
}

async function loadModel(){
    $('btnLoad').disabled=true; $('btnLoad').textContent='⏳ STARTING...';
    try{await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({variant:$('selVariant').value})});}catch(e){}
}

// ══════════════════════════════════════════════════
//  CAMERA MODE — single image
// ══════════════════════════════════════════════════
function initDrop(){
    const z=$('dropzone'),inp=$('fileInput');
    ['dragenter','dragover'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.add('over');}));
    ['dragleave','drop'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.remove('over');}));
    z.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length)uploadFile(ev.dataTransfer.files[0]);});
    inp.addEventListener('change',()=>{if(inp.files.length)uploadFile(inp.files[0]);inp.value='';});
}

function setInputImage(id, url, w, h) {
    imageId = id; imageUrl = url;
    $('dropzone').style.display = 'none';
    $('inputPreview').style.display = '';
    $('inputImg').src = url;
    setViewportImage(url);
    if(w && h) setDimensions(w, h);
    updateGenButton();
    saveSession();
}

function setDimensions(w, h){
    $('inpWidth').value = w;
    $('inpHeight').value = h;
}

async function uploadFile(file){
    const fd=new FormData();fd.append('image',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){alert(d.error);return;}
        setInputImage(d.id, d.url, d.w, d.h);
    }catch(e){alert('Upload failed');}
}

function clearInput(){
    imageId=null; imageUrl=null;
    $('dropzone').style.display='';
    $('inputPreview').style.display='none';
    $('inputImg').src='';
    $('inpWidth').value=''; $('inpHeight').value='';
    updateGenButton();
    saveSession();
}

// ══════════════════════════════════════════════════
//  EDIT MODE — 3 stacked image slots
// ══════════════════════════════════════════════════
function initSlots(){
    for(let i=0;i<3;i++){
        const el=$('slot'+i);
        const emptyDiv=el.querySelector('.slot-empty');
        ['dragenter','dragover'].forEach(e=>emptyDiv.addEventListener(e,ev=>{ev.preventDefault();el.classList.add('drag-over');}));
        ['dragleave','drop'].forEach(e=>emptyDiv.addEventListener(e,ev=>{ev.preventDefault();el.classList.remove('drag-over');}));
        emptyDiv.addEventListener('drop',ev=>{
            if(ev.dataTransfer.files.length) uploadToSlot(i,ev.dataTransfer.files[0]);
        });
        const filledDiv=el.querySelector('.slot-filled');
        ['dragenter','dragover'].forEach(e=>filledDiv.addEventListener(e,ev=>{ev.preventDefault();el.classList.add('drag-over');}));
        ['dragleave','drop'].forEach(e=>filledDiv.addEventListener(e,ev=>{ev.preventDefault();el.classList.remove('drag-over');}));
        filledDiv.addEventListener('drop',ev=>{
            if(ev.dataTransfer.files.length) uploadToSlot(i,ev.dataTransfer.files[0]);
        });
    }
}

function slotFileChange(i){
    const inp=$('slotFile'+i);
    if(inp.files.length) uploadToSlot(i,inp.files[0]);
    inp.value='';
}

async function uploadToSlot(i,file){
    const fd=new FormData();fd.append('image',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){alert(d.error);return;}
        slots[i]={id:d.id, url:d.url, filename:d.filename, w:d.w, h:d.h};
        setDimensions(d.w, d.h);
        renderSlot(i);
        updateGenButton();
        saveSession();
    }catch(e){alert('Upload failed');}
}

function removeSlot(i){
    slots[i]=null;
    renderSlot(i);
    updateGenButton();
    saveSession();
}

function renderSlot(i){
    const el=$('slot'+i);
    if(!el) return;
    const emptyDiv=el.querySelector('.slot-empty');
    const filledDiv=el.querySelector('.slot-filled');
    if(slots[i]){
        emptyDiv.style.display='none';
        filledDiv.style.display='';
        $('slotImg'+i).src=slots[i].url;
    } else {
        emptyDiv.style.display='';
        filledDiv.style.display='none';
    }
    const count=slots.filter(s=>s!==null).length;
    if($('slotCounter')) $('slotCounter').textContent=`${count} / 3`;
}

// ── Modal ─────────────────────────────────────────
function openModal(src){
    if(!src)return;
    $('modalImg').src=src;
    $('imgModal').classList.add('open');
}
function closeModal(){
    $('imgModal').classList.remove('open');
}
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });

// ── Use output as input ───────────────────────────
async function useAsInput(){
    if(!lastOutputFilename) return;
    try{
        const r=await fetch('/api/use_output',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filename:lastOutputFilename})});
        const d=await r.json();
        if(d.error){alert(d.error);return;}
        if(currentMode==='camera'){
            setInputImage(d.id, d.url, d.w, d.h);
        } else {
            let target=slots.findIndex(s=>s===null);
            if(target===-1) target=0;
            slots[target]={id:d.id, url:d.url, filename:d.filename, w:d.w, h:d.h};
            setDimensions(d.w, d.h);
            renderSlot(target);
            updateGenButton();
            saveSession();
        }
    }catch(e){alert('Failed');}
}

// ── Prompt lock ───────────────────────────────────
function toggleLock(){
    window._promptLocked=!window._promptLocked;
    $('promptLock').classList.toggle('locked',window._promptLocked);
    $('promptLock').textContent=window._promptLocked?'🔒':'🔓';
    saveSession();
}

// ══════════════════════════════════════════════════
//  GENERATE — polling-based
// ══════════════════════════════════════════════════
async function doGenerate(){
    if(currentMode==='camera') return doGenerateCamera();
    return doGenerateEdit();
}

async function doGenerateCamera(){
    if(!imageId||!ready)return;
    const payload={
        mode:'camera',
        image_ids:[imageId],
        prompt:$('promptInput').value,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
        lora_scale:parseFloat($('rngLora').value),
        width:parseInt($('inpWidth').value)||0,
        height:parseInt($('inpHeight').value)||0,
    };
    await submitAndPoll(payload);
}

async function doGenerateEdit(){
    const filled=slots.filter(s=>s!==null);
    if(!filled.length||!ready)return;
    const prompt=$('customPrompt').value.trim();
    if(!prompt){alert('Please enter an edit instruction.');return;}
    const payload={
        mode:'edit',
        image_ids:filled.map(s=>s.id),
        prompt:prompt,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
        width:parseInt($('inpWidth').value)||0,
        height:parseInt($('inpHeight').value)||0,
    };
    await submitAndPoll(payload);
}

async function submitAndPoll(payload){
    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';
    $('genText').textContent='Starting...';

    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const start=await r.json();
        if(start.error){$('genText').textContent=start.error;finishJob();return;}
        activeJobId=start.job_id;
        saveSession();
        await pollJob(activeJobId);
    }catch(e){$('genText').textContent='Request failed: '+e.message;}
    finally{
        finishJob();
    }
}

// Resume polling for a job (e.g. after page refresh or reconnect)
async function resumePolling(jobId){
    $('btnGen').disabled=true; $('btnGen').textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active');
    $('genText').textContent='Resuming...';

    // First check if the job already completed while we were away
    try{
        const r=await fetch(`/api/job/${jobId}`);
        if(r.ok){
            const j=await r.json();
            if(j.status==='done'){
                lastResult=j.result;
                showResult(j.result);
                finishJob();
                return;
            } else if(j.status==='error'){
                $('genText').textContent=j.error||'Generation failed';
                finishJob();
                return;
            }
        } else if(r.status===404){
            // Job is gone — check if server is still generating
            const sr=await fetch('/api/status');
            if(sr.ok){
                const st=await sr.json();
                if(!st.generating){
                    // Not generating and job gone — check completed jobs list
                    try{
                        const jr=await fetch('/api/jobs');
                        if(jr.ok){
                            const jd=await jr.json();
                            // Look for any recent completed job
                            const recent=Object.entries(jd.jobs||{})
                                .filter(([,v])=>v.status==='done')
                                .sort(([,a],[,b])=>(b.finished_at||0)-(a.finished_at||0));
                            if(recent.length && recent[0][1].result){
                                lastResult=recent[0][1].result;
                                showResult(recent[0][1].result);
                            }
                        }
                    }catch(e){}
                    $('genText').textContent='Job completed or expired';
                    finishJob();
                    return;
                }
            }
        }
    }catch(e){/* network issue, fall through to polling */}

    try{
        await pollJob(jobId);
    }catch(e){$('genText').textContent='Poll failed: '+e.message;}
    finally{
        finishJob();
    }
}

async function pollJob(jobId){
    let consecutiveErrors=0;
    for(let attempt=0;attempt<600;attempt++){
        await new Promise(ok=>setTimeout(ok,1500));
        try{
            const pr=await fetch(`/api/job/${jobId}`);
            consecutiveErrors=0; // reset on any successful fetch
            if(pr.status===404){
                try{
                    const sr=await fetch('/api/status');
                    const st=await sr.json();
                    if(st.generating){
                        if(st.progress && st.progress.total){
                            const pct=Math.round(st.progress.step/st.progress.total*100);
                            $('genFill').style.width=pct+'%';
                            $('genText').textContent=`Step ${st.progress.step}/${st.progress.total}`;
                        }
                        continue;
                    }
                }catch(e){}
                // Check /api/jobs for any recently finished job
                try{
                    const jr=await fetch('/api/jobs');
                    if(jr.ok){
                        const jd=await jr.json();
                        const recent=Object.entries(jd.jobs||{})
                            .filter(([,v])=>v.status==='done')
                            .sort(([,a],[,b])=>(b.finished_at||0)-(a.finished_at||0));
                        if(recent.length && recent[0][1].result){
                            lastResult=recent[0][1].result;
                            showResult(recent[0][1].result);
                            saveSession();
                            return;
                        }
                    }
                }catch(e){}
                $('genText').textContent='Job completed or expired';
                return;
            }
            const j=await pr.json();
            if(j.status==='done'){
                lastResult=j.result;
                showResult(j.result);
                saveSession();
                return;
            } else if(j.status==='error'){
                $('genText').textContent=j.error||'Generation failed';
                return;
            }
            if(j.progress && j.progress.total){
                const pct=Math.round(j.progress.step/j.progress.total*100);
                $('genFill').style.width=pct+'%';
                $('genText').textContent=`Step ${j.progress.step}/${j.progress.total}`;
            }
        }catch(e){
            consecutiveErrors++;
            // After 20 consecutive network failures (~30s), show disconnected
            if(consecutiveErrors>20){
                $('genText').textContent='Lost connection to server — will resume when reconnected';
            } else {
                $('genText').textContent='Reconnecting...';
            }
            // Don't give up — keep polling (server might come back)
        }
    }
    $('genText').textContent='Timed out waiting for result';
}

function finishJob(){
    activeJobId=null;
    $('btnGen').disabled=false; $('btnGen').textContent='⚡ GENERATE';
    $('genOverlay').classList.remove('active');
    updateGenButton();
    saveSession();
}

function showResult(d){
    $('outImg').src=d.url; $('outImg').style.display='';
    $('outPh').style.display='none';
    $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
    $('resultBar').style.display='flex';
    const tag=d.mode==='camera'?'':'✏️ ';
    $('resultText').textContent=`${tag}${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed}`;
    lastOutputFilename=d.url.split('/').pop();
    $('btnUseAsInput').style.display='';
}
