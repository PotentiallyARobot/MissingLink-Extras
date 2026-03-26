// app.js — Camera Studio client
const $=id=>document.getElementById(id);

let imageId=null,imageUrl=null,ready=false,consoleOpen=false,hwOpen=false;
window._promptLocked=false;
let lastLogT=0;

document.addEventListener('DOMContentLoaded',()=>{
    initViewport();initDrop();pollStatus();pollLogs();
    setInterval(pollStatus,2500);setInterval(pollLogs,1500);
});

// ── Status ────────────────────────────────────────
async function pollStatus(){
    try{
        const r=await fetch('/api/status'),d=await r.json();
        const dot=$('connDot'),lbl=$('connLabel'),bg=$('btnGen');
        if(d.ready){
            dot.className='dot on'; lbl.textContent='Connected'; ready=true;
            bg.disabled=!imageId;
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
        // CPU
        $('hwCpu').textContent=d.cpu_pct+'%';
        $('hwCpuBar').style.width=d.cpu_pct+'%';
        // RAM
        if(d.ram_total){
            $('hwRam').textContent=`${d.ram} / ${d.ram_total} MB`;
            $('hwRamBar').style.width=Math.round(d.ram/d.ram_total*100)+'%';
        }
        // Disk
        if(d.disk_total){
            $('hwDisk').textContent=`${d.disk} / ${d.disk_total} GB`;
            $('hwDiskBar').style.width=Math.round(d.disk/d.disk_total*100)+'%';
        }
        // Console footer
        $('gpuInfo').innerHTML=`<span class="gi-dot" style="background:var(--green)"></span> ${d.gpu||'—'} · VRAM ${d.vram}/${d.vram_total}MB · CPU ${d.cpu_pct}% · RAM ${d.ram}/${d.ram_total}MB`;
        // Progress
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

// ── Upload (single image) ─────────────────────────
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
    $('btnGen').disabled = !ready;
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
    $('btnGen').disabled=true;
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
        setInputImage(d.id, d.url);
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
    if(!imageId||!ready)return;
    const btn=$('btnGen'); btn.disabled=true; btn.textContent='⏳ GENERATING...';
    $('genOverlay').classList.add('active'); $('genFill').style.width='0';
    $('outImg').style.display='none'; $('outPh').style.display='none';

    const payload={
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
        // Track output for use-as-input
        lastOutputFilename=d.url.split('/').pop();
        $('btnUseAsInput').style.display='';
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}
