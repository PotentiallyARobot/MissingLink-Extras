// app.js — Qwen Studio client
const $=id=>document.getElementById(id);

let imageId=null,imageUrl=null,ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;
let currentMode='camera';

// Edit mode: 3 image slots
let slots=[null,null,null];

document.addEventListener('DOMContentLoaded',()=>{
    initViewport();initDrop();initSlots();pollStatus();pollLogs();
    setInterval(pollStatus,2500);setInterval(pollLogs,1500);
});

// ── Mode switching (instant, UI only) ─────────────
function setMode(mode){
    if(mode===currentMode) return;
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
}

function updateGenButton(){
    if(currentMode==='camera'){
        $('btnGen').disabled=!(ready && imageId);
    } else {
        const filled=slots.filter(s=>s!==null);
        const cp=$('customPrompt').value.trim();
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

function setInputImage(id, url) {
    imageId = id; imageUrl = url;
    $('dropzone').style.display = 'none';
    $('inputPreview').style.display = '';
    $('inputImg').src = url;
    setViewportImage(url);
    updateGenButton();
}

async function uploadFile(file){
    const fd=new FormData();fd.append('image',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){alert(d.error);return;}
        setInputImage(d.id, d.url);
    }catch(e){alert('Upload failed');}
}

function clearInput(){
    imageId=null; imageUrl=null;
    $('dropzone').style.display='';
    $('inputPreview').style.display='none';
    $('inputImg').src='';
    updateGenButton();
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
        slots[i]={id:d.id, url:d.url, filename:d.filename};
        renderSlot(i);
        updateGenButton();
    }catch(e){alert('Upload failed');}
}

function removeSlot(i){
    slots[i]=null;
    renderSlot(i);
    updateGenButton();
}

function renderSlot(i){
    const el=$('slot'+i);
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
    $('slotCounter').textContent=`${count} / 3`;
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
            setInputImage(d.id, d.url);
        } else {
            let target=slots.findIndex(s=>s===null);
            if(target===-1) target=0;
            slots[target]={id:d.id, url:d.url, filename:d.filename};
            renderSlot(target);
            updateGenButton();
        }
    }catch(e){alert('Failed');}
}

// ── Prompt lock ───────────────────────────────────
function toggleLock(){
    window._promptLocked=!window._promptLocked;
    $('promptLock').classList.toggle('locked',window._promptLocked);
    $('promptLock').textContent=window._promptLocked?'🔒':'🔓';
}

// ── Generate ──────────────────────────────────────
async function doGenerate(){
    if(currentMode==='camera') return doGenerateCamera();
    return doGenerateEdit();
}

async function doGenerateCamera(){
    if(!imageId||!ready)return;
    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

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
    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const d=await r.json();
        if(d.error){$('genText').textContent=d.error;return;}
        showResult(d);
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}

async function doGenerateEdit(){
    const filled=slots.filter(s=>s!==null);
    if(!filled.length||!ready)return;
    const prompt=$('customPrompt').value.trim();
    if(!prompt){alert('Please enter an edit instruction.');return;}

    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

    const imageIds=slots.filter(s=>s!==null).map(s=>s.id);

    const payload={
        mode:'edit',
        image_ids:imageIds,
        prompt:prompt,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
        width:parseInt($('inpWidth').value)||0,
        height:parseInt($('inpHeight').value)||0,
    };
    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const d=await r.json();
        if(d.error){$('genText').textContent=d.error;return;}
        showResult(d);
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
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
