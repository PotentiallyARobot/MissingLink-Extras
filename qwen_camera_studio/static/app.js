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
        const dot=$('connDot'),lbl=$('connLabel'),bl=$('btnLoad'),bg=$('btnGen');
        if(d.ready){
            dot.className='dot on'; lbl.textContent='Connected';
            bl.textContent='✅ LOADED'; bl.disabled=true; ready=true;
            bg.disabled=!imageId;
        } else if(d.loading){
            dot.className='dot'; dot.style.background='var(--gold)';
            lbl.textContent='Loading model...'; bl.textContent='⏳ LOADING...'; bl.disabled=true; ready=false; bg.disabled=true;
        } else if(d.error){
            dot.className='dot'; dot.style.background='var(--red)';
            lbl.textContent='Error'; bl.disabled=false; bl.textContent='⚠ RETRY'; ready=false; bg.disabled=true;
        } else {
            dot.className='dot'; lbl.textContent='Model not loaded';
            bl.disabled=false; bl.textContent='LOAD MODEL'; ready=false; bg.disabled=true;
        }
        // HW
        if(d.gpu){
            $('hwGpu').textContent=d.gpu;
            $('hwVram').textContent=`${d.vram} / ${d.vram_total} MB`;
            const pct=d.vram_total?Math.round(d.vram/d.vram_total*100):0;
            $('hwVramBar').style.width=pct+'%';
            $('gpuInfo').innerHTML=`<span class="gi-dot" style="background:var(--green)"></span> ${d.gpu} · VRAM ${d.vram}/${d.vram_total}MB`;
        }
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

// ── Upload ────────────────────────────────────────
function initDrop(){
    const z=$('dropzone'),inp=$('fileInput');
    ['dragenter','dragover'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.add('over');}));
    ['dragleave','drop'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.remove('over');}));
    z.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length)uploadFile(ev.dataTransfer.files[0]);});
    inp.addEventListener('change',()=>{if(inp.files.length)uploadFile(inp.files[0]);inp.value='';});
}
async function uploadFile(file){
    const fd=new FormData();fd.append('image',file);
    $('dropzone').innerHTML='<div class="dz-text">Uploading...</div>';
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();
        if(d.error){$('dropzone').innerHTML=`<div class="dz-text">${d.error}</div>`;return;}
        imageId=d.id; imageUrl=d.url;
        // Thumb
        const ts=$('thumbs'); ts.innerHTML='';
        const th=document.createElement('div');th.className='thumb';
        const img=document.createElement('img');img.src=d.url;th.appendChild(img);ts.appendChild(th);
        $('dropzone').innerHTML='<input type="file" accept="image/*" id="fileInput"><div class="dz-text">Drop images or <b>browse</b></div>';
        document.getElementById('fileInput').addEventListener('change',function(){if(this.files.length)uploadFile(this.files[0]);this.value='';});
        setViewportImage(d.url);
        $('btnGen').disabled=!ready;
    }catch(e){$('dropzone').innerHTML='<div class="dz-text">Upload failed</div>';}
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
        if(d.error){alert(d.error);return;}
        $('outImg').src=d.url; $('outImg').style.display='';
        $('rngSeed').value=d.seed; $('vSeed').textContent=d.seed;
        $('resultBar').style.display=''; $('resultBar').textContent=`${d.w}×${d.h} · ${d.elapsed}s · seed:${d.seed} · ${d.prompt}`;
    }catch(e){alert('Request failed: '+e.message);}
    finally{
        btn.disabled=false; btn.textContent='⚡ GENERATE';
        $('genOverlay').classList.remove('active');
    }
}
