// app.js — Camera Studio client
const $=id=>document.getElementById(id);

let imageId=null,imageUrl=null,ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;
let currentMode='camera'; // 'camera' or 'normal'

// Multi-image state for normal mode
let multiImages=[]; // [{id, url, filename}]

document.addEventListener('DOMContentLoaded',()=>{
    initViewport();initDrop();initDropMulti();pollStatus();pollLogs();
    setInterval(pollStatus,2500);setInterval(pollLogs,1500);
});

// ── Mode switching ────────────────────────────────
function setMode(mode){
    currentMode=mode;
    const isCam=mode==='camera';

    // Toggle button active state
    $('btnModeCamera').classList.toggle('active',isCam);
    $('btnModeNormal').classList.toggle('active',!isCam);

    // Show/hide mode-specific sections
    document.querySelectorAll('.camera-only').forEach(el=>el.style.display=isCam?'':'none');
    document.querySelectorAll('.normal-only').forEach(el=>el.style.display=isCam?'none':'');

    // Prompt bar: camera mode shows the auto-generated prompt bar, normal hides it
    $('promptBarCamera').style.display=isCam?'':'none';

    // Update placeholder text
    $('outPh').textContent=isCam
        ?'Upload an image and generate'
        :'Upload image(s) and describe your edit';

    // Update generate button state
    updateGenButton();
}

function updateGenButton(){
    if(currentMode==='camera'){
        $('btnGen').disabled=!(ready && imageId);
    } else {
        $('btnGen').disabled=!(ready && multiImages.length>0 && $('customPrompt').value.trim());
    }
}

// ── Status ────────────────────────────────────────
async function pollStatus(){
    try{
        const r=await fetch('/api/status'),d=await r.json();
        const dot=$('connDot'),lbl=$('connLabel'),bg=$('btnGen');
        if(d.ready){
            dot.className='dot on'; lbl.textContent='Connected'; ready=true;
            updateGenButton();
        } else if(d.loading){
            dot.className='dot'; dot.style.background='var(--gold)';
            lbl.textContent='Loading model...'; ready=false; bg.disabled=true;
        } else if(d.error){
            dot.className='dot'; dot.style.background='var(--red)';
            lbl.textContent='Error'; ready=false; bg.disabled=true;
        } else {
            dot.className='dot'; lbl.textContent='Waiting...'; ready=false; bg.disabled=true;
        }
        // HW
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

// ── Console toggle ────────────────────────────────
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

// ── Model load ────────────────────────────────────
async function loadModel(){
    $('btnLoad').disabled=true; $('btnLoad').textContent='⏳ STARTING...';
    try{await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({variant:$('selVariant').value})});}catch(e){}
}

// ── Upload: Camera mode (single image) ────────────
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

// ── Upload: Normal mode (multi image) ─────────────
function initDropMulti(){
    const z=$('dropzoneMulti'),inp=$('fileInputMulti');
    ['dragenter','dragover'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.add('over');}));
    ['dragleave','drop'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.remove('over');}));
    z.addEventListener('drop',ev=>{
        if(ev.dataTransfer.files.length){
            Array.from(ev.dataTransfer.files).forEach(f=>uploadMultiFile(f));
        }
    });
    inp.addEventListener('change',()=>{
        Array.from(inp.files).forEach(f=>uploadMultiFile(f));
        inp.value='';
    });
    // Also listen for custom prompt changes
    $('customPrompt').addEventListener('input',updateGenButton);
}

async function uploadMultiFile(file){
    const fd=new FormData();fd.append('image',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){alert(d.error);return;}
        multiImages.push({id:d.id, url:d.url, filename:d.filename});
        renderMultiPreviews();
        updateGenButton();
    }catch(e){alert('Upload failed');}
}

function removeMultiImage(idx){
    multiImages.splice(idx,1);
    renderMultiPreviews();
    updateGenButton();
}

function renderMultiPreviews(){
    const list=$('multiPreviewList');
    list.innerHTML='';
    multiImages.forEach((img,i)=>{
        const item=document.createElement('div');
        item.className='multi-preview-item';
        item.innerHTML=`
            <img src="${img.url}" alt="">
            <button class="preview-clear" onclick="removeMultiImage(${i})" title="Remove">×</button>
        `;
        list.appendChild(item);
    });
}

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
            multiImages.push({id:d.id, url:d.url, filename:d.filename});
            renderMultiPreviews();
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
    return doGenerateNormal();
}

async function doGenerateCamera(){
    if(!imageId||!ready)return;
    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

    const payload={
        mode:'camera',
        image_id:imageId,
        prompt:$('promptInput').value,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
        lora_scale:parseFloat($('rngLora').value),
    };
    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const d=await r.json();
        if(d.error){$('genText').textContent=d.error;return;}
        $('outImg').src=d.url; $('outImg').style.display='';
        $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
        $('resultBar').style.display='flex';
        $('resultText').textContent=`${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed}`;
        lastOutputFilename=d.url.split('/').pop();
        $('btnUseAsInput').style.display='';
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}

async function doGenerateNormal(){
    if(!multiImages.length||!ready)return;
    const prompt=$('customPrompt').value.trim();
    if(!prompt){alert('Please enter an edit instruction.');return;}

    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

    const payload={
        mode:'normal',
        image_ids:multiImages.map(m=>m.id),
        prompt:prompt,
        seed:parseInt($('rngSeed').value),
        randomize_seed:$('chkRand').checked,
        guidance_scale:parseFloat($('rngCfg').value),
        inference_steps:parseInt($('rngSteps').value),
    };
    try{
        const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const d=await r.json();
        if(d.error){$('genText').textContent=d.error;return;}
        $('outImg').src=d.url; $('outImg').style.display='';
        $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
        $('resultBar').style.display='flex';
        $('resultText').textContent=`${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed}`;
        lastOutputFilename=d.url.split('/').pop();
        $('btnUseAsInput').style.display='';
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}
