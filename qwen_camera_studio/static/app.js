// app.js — Camera Studio client (unified)
const $=id=>document.getElementById(id);

let ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;

// ── 3-slot image state ────────────────────────────
const MAX_SLOTS=3;
let slots=[null,null,null]; // each: {id,url,filename} or null

// ── Camera LoRA state ─────────────────────────────
let cameraLoraOn=true;

document.addEventListener('DOMContentLoaded',()=>{
    initViewport();initSlotDragDrop();pollStatus();pollLogs();
    setInterval(pollStatus,2500);setInterval(pollLogs,1500);
    syncUI();
});

// ── Effective mode ────────────────────────────────
function effectiveMode(){
    const cp=$('customPrompt').value.trim();
    if(cp) return 'edit';
    if(cameraLoraOn) return 'camera';
    return 'edit';
}

function filledSlots(){ return slots.filter(s=>s!==null); }

// ── UI sync ───────────────────────────────────────
function syncUI(){
    const mode=effectiveMode();
    const pill=$('promptModePill');
    const hint=$('editHint');

    // Prompt bar
    if(mode==='camera'){
        pill.textContent='🎥 CAMERA';
        pill.className='prompt-mode-pill pill-camera';
        $('promptLock').style.display='';
        hint.textContent='🎥 Camera LoRA active — prompt from pose controls';
        hint.className='edit-hint hint-camera';
    } else {
        pill.textContent='✏️ EDIT';
        pill.className='prompt-mode-pill pill-edit';
        $('promptInput').value=$('customPrompt').value.trim()||'';
        $('promptLock').style.display='none';
        if(cameraLoraOn){
            hint.textContent='✏️ Edit prompt entered — camera LoRA auto-disabled for this generation';
            hint.className='edit-hint hint-edit';
        } else {
            hint.textContent='✏️ Camera LoRA off — using custom edit prompt';
            hint.className='edit-hint hint-edit';
        }
    }

    // Camera panel dim
    $('cameraBody').classList.toggle('disabled',!cameraLoraOn);
    $('loraBadge').textContent=cameraLoraOn?'ON':'OFF';
    $('loraBadge').classList.toggle('off',!cameraLoraOn);

    // Slot count
    $('slotCount').textContent=`${filledSlots().length} / ${MAX_SLOTS}`;

    // Render slots
    for(let i=0;i<MAX_SLOTS;i++){
        const el=$('slot'+i);
        const filled=el.querySelector('.slot-filled');
        const empty=el.querySelector('.slot-empty-inner');
        if(slots[i]){
            el.classList.remove('empty');
            filled.style.display=''; empty.style.display='none';
            $('slotImg'+i).src=slots[i].url;
        } else {
            el.classList.add('empty');
            filled.style.display='none'; empty.style.display='';
        }
    }

    // Viewport image = first slot
    if(slots[0]) setViewportImage(slots[0].url);

    updateGenButton();
}

function updateGenButton(){
    const mode=effectiveMode();
    const has=filledSlots().length>0;
    if(mode==='camera'){
        $('btnGen').disabled=!(ready && has);
    } else {
        const cp=$('customPrompt').value.trim();
        $('btnGen').disabled=!(ready && has && cp);
    }
}

// ── Camera LoRA toggle ────────────────────────────
function toggleCameraLora(){
    cameraLoraOn=!cameraLoraOn;
    $('cameraBody').style.display=cameraLoraOn?'':'none';
    $('cameraChevron').textContent=cameraLoraOn?'▾':'▸';
    if(cameraLoraOn && !$('customPrompt').value.trim()){
        if(typeof updatePromptFromCamera==='function') updatePromptFromCamera();
    }
    syncUI();
}

function onEditPromptChange(){ syncUI(); }

// ── Image slots ───────────────────────────────────
function slotClick(i){
    if(slots[i]){
        // Open modal
        openModal(slots[i].url);
    } else {
        $('slotFile'+i).click();
    }
}

function slotFileChange(i){
    const inp=$('slotFile'+i);
    if(inp.files.length) uploadToSlot(i,inp.files[0]);
    inp.value='';
}

function removeSlot(ev,i){
    ev.stopPropagation();
    slots[i]=null;
    syncUI();
}

async function uploadToSlot(i,file){
    const fd=new FormData();fd.append('image',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){alert(d.error);return;}
        slots[i]={id:d.id, url:d.url, filename:d.filename};
        syncUI();
    }catch(e){alert('Upload failed');}
}

// Drag and drop onto slots
function initSlotDragDrop(){
    for(let i=0;i<MAX_SLOTS;i++){
        const el=$('slot'+i);
        ['dragenter','dragover'].forEach(e=>el.addEventListener(e,ev=>{ev.preventDefault();el.classList.add('drag-over');}));
        ['dragleave','drop'].forEach(e=>el.addEventListener(e,ev=>{ev.preventDefault();el.classList.remove('drag-over');}));
        el.addEventListener('drop',ev=>{
            if(ev.dataTransfer.files.length) uploadToSlot(i,ev.dataTransfer.files[0]);
        });
    }
}

// ── Modal ─────────────────────────────────────────
function openModal(src){
    $('modalImg').src=src;
    $('imgModal').classList.add('open');
}
function closeModal(){
    $('imgModal').classList.remove('open');
}
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });

// ── Use output as input ───────────────────────────
let lastOutputFilename=null;

async function useAsInput(){
    if(!lastOutputFilename) return;
    try{
        const r=await fetch('/api/use_output',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filename:lastOutputFilename})});
        const d=await r.json();
        if(d.error){alert(d.error);return;}
        // Put in first empty slot, or replace slot 0
        let target=slots.findIndex(s=>s===null);
        if(target===-1) target=0;
        slots[target]={id:d.id, url:d.url, filename:d.filename};
        syncUI();
    }catch(e){alert('Failed');}
}

// ── Prompt lock ───────────────────────────────────
function toggleLock(){
    window._promptLocked=!window._promptLocked;
    $('promptLock').classList.toggle('locked',window._promptLocked);
    $('promptLock').textContent=window._promptLocked?'🔒':'🔓';
}

// ── Status poll ───────────────────────────────────
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
            $('hwVramBar').style.width=(d.vram_total?Math.round(d.vram/d.vram_total*100):0)+'%';
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

// ── Logs poll ─────────────────────────────────────
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

// ── Model load (if used externally) ───────────────
async function loadModel(){
    $('btnLoad').disabled=true; $('btnLoad').textContent='⏳ STARTING...';
    try{await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({variant:$('selVariant').value})});}catch(e){}
}

// ── Generate ──────────────────────────────────────
async function doGenerate(){
    const filled=filledSlots();
    if(!filled.length||!ready)return;
    const mode=effectiveMode();

    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

    // Collect image IDs in slot order (preserving order)
    const imageIds=slots.filter(s=>s!==null).map(s=>s.id);

    const payload={
        image_ids:imageIds,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
    };

    if(mode==='camera'){
        payload.mode='camera';
        payload.prompt=$('promptInput').value;
        payload.lora_scale=parseFloat($('rngLora').value);
    } else {
        payload.mode='edit';
        payload.prompt=$('customPrompt').value.trim();
    }

    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const d=await r.json();
        if(d.error){$('genText').textContent=d.error;return;}
        $('outImg').src=d.url; $('outImg').style.display='';
        $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
        $('resultBar').style.display='flex';
        const tag=d.mode==='camera'?'🎥 camera':'✏️ edit';
        $('resultText').textContent=`${tag} · ${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed}`;
        lastOutputFilename=d.url.split('/').pop();
        $('btnUseAsInput').style.display='';
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}
