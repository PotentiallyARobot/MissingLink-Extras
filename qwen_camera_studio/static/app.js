// app.js — Qwen Studio client
const $=id=>document.getElementById(id);

let imageId=null,imageUrl=null,ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;
let currentMode='camera';
let activeJobId=null; // currently polling job

// Edit mode: 3 image slots
let slots=[null,null,null];

// ── Session persistence ───────────────────────────
function saveSession(){
    try{
        const s={
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
        };
        sessionStorage.setItem('qwen_studio',JSON.stringify(s));
    }catch(e){}
}

let lastResult=null; // last successful result data

function restoreSession(){
    try{
        const raw=sessionStorage.getItem('qwen_studio');
        if(!raw) return;
        const s=JSON.parse(raw);

        // Restore mode
        if(s.mode && s.mode!==currentMode) setMode(s.mode);

        // Restore camera input
        if(s.imageId && s.imageUrl){
            imageId=s.imageId; imageUrl=s.imageUrl;
            $('dropzone').style.display='none';
            $('inputPreview').style.display='';
            $('inputImg').src=s.imageUrl;
            setViewportImage(s.imageUrl);
        }

        // Restore edit slots
        if(s.slots){
            for(let i=0;i<3;i++){
                slots[i]=s.slots[i]||null;
                renderSlot(i);
            }
        }

        // Restore settings
        if(s.promptInput && $('promptInput')) $('promptInput').value=s.promptInput;
        if(s.customPrompt && $('customPrompt')) $('customPrompt').value=s.customPrompt;
        if(s.seed){ $('rngSeed').value=s.seed; $('vSeed').textContent=s.seed; }
        if(s.cfg){ $('rngCfg').value=s.cfg; $('vCfg').textContent=s.cfg; }
        if(s.steps){ $('rngSteps').value=s.steps; $('vSteps').textContent=s.steps; }
        if(s.lora){ $('rngLora').value=s.lora; $('vLora').textContent=s.lora; }
        if(s.width) $('inpWidth').value=s.width;
        if(s.height) $('inpHeight').value=s.height;
        if(s.chkRand!==undefined) $('chkRand').checked=s.chkRand;

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
    restoreSession();
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
    }catch(e){}
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
let lastOutputFilename = null;

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
        if(start.error){$('genText').textContent=start.error;return;}
        activeJobId=start.job_id;
        saveSession();
        await pollJob(activeJobId);
    }catch(e){$('genText').textContent='Request failed: '+e.message;}
    finally{
        finishJob();
    }
}

// Resume polling for a job (e.g. after page refresh)
async function resumePolling(jobId){
    $('btnGen').disabled=true; $('btnGen').textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active');
    $('genText').textContent='Resuming...';
    try{
        await pollJob(jobId);
    }catch(e){$('genText').textContent='Poll failed: '+e.message;}
    finally{
        finishJob();
    }
}

async function pollJob(jobId){
    // Poll with max 600 attempts (~15 min at 1.5s intervals)
    for(let attempt=0;attempt<600;attempt++){
        await new Promise(ok=>setTimeout(ok,1500));
        try{
            const pr=await fetch(`/api/job/${jobId}`);
            if(pr.status===404){
                // Job gone — was already consumed or server restarted
                // Check if backend is still generating
                try{
                    const sr=await fetch('/api/status');
                    const st=await sr.json();
                    if(st.generating){
                        // Still generating but job ID lost (server restart?)
                        // Just keep waiting and show progress from status
                        if(st.progress && st.progress.total){
                            const pct=Math.round(st.progress.step/st.progress.total*100);
                            $('genFill').style.width=pct+'%';
                            $('genText').textContent=`Step ${st.progress.step}/${st.progress.total}`;
                        }
                        continue;
                    }
                }catch(e){}
                // Not generating and job 404 — it's truly gone
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
            // Still running
            if(j.progress && j.progress.total){
                const pct=Math.round(j.progress.step/j.progress.total*100);
                $('genFill').style.width=pct+'%';
                $('genText').textContent=`Step ${j.progress.step}/${j.progress.total}`;
            }
        }catch(e){
            // Network blip — just retry
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
    $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
    $('resultBar').style.display='flex';
    const tag=d.mode==='camera'?'':'✏️ ';
    $('resultText').textContent=`${tag}${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed}`;
    lastOutputFilename=d.url.split('/').pop();
    $('btnUseAsInput').style.display='';
}
