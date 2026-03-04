const API = window.location.origin;
const $ = id => document.getElementById(id);
const rv = (id, v) => $(id).textContent = v;
let generating = false, toastT, S = [], nid = 0;
let lassoOn = false, lassoDrawing = false, lassoPolys = [], curPoly = [], maskDU = null;

function toast(m, e) { const t = $('toast'); t.textContent = m; t.className = 'toast show' + (e ? ' err' : ''); clearTimeout(toastT); toastT = setTimeout(() => t.className = 'toast', 3000) }
function togSec(el) { el.classList.toggle('open'); el.nextElementSibling.classList.toggle('show') }
function lightbox(src) { $('lbi').src = src; $('lb').classList.add('open') }

// ========== CONSOLE ==========
let consoleOpen = false;
let consoleLogs = [];
let consoleInterval = null;
let lastLogId = 0;

function toggleConsole() {
    consoleOpen = !consoleOpen;
    $('consoleDrawer').classList.toggle('open', consoleOpen);
    $('consoleToggle').textContent = consoleOpen ? '▾ Console' : '▸ Console';
    if (consoleOpen && !consoleInterval) {
        fetchLogs();
        consoleInterval = setInterval(fetchLogs, 2000);
    }
    if (!consoleOpen && consoleInterval) {
        clearInterval(consoleInterval);
        consoleInterval = null;
    }
}

async function fetchLogs() {
    try {
        const resp = await fetch(API + '/api/logs?after=' + lastLogId);
        const d = await resp.json();
        if (d.logs && d.logs.length) {
            const body = $('consoleBody');
            d.logs.forEach(log => {
                const line = document.createElement('div');
                let cls = 'log-info';
                if (log.text.startsWith('✅') || log.text.includes('Ready')) cls = 'log-ok';
                else if (log.text.startsWith('⚠') || log.text.includes('ERROR') || log.text.includes('fail')) cls = 'log-err';
                else if (log.text.startsWith('🖥') || log.text.startsWith('🎨') || log.text.startsWith('Loading')) cls = 'log-warn';
                line.className = cls;
                line.textContent = log.text;
                body.appendChild(line);
                lastLogId = Math.max(lastLogId, log.id);
            });
            body.scrollTop = body.scrollHeight;
        }
    } catch (e) { }
}

function clearConsole() {
    $('consoleBody').innerHTML = '';
    fetch(API + '/api/logs/clear', { method: 'POST' });
    lastLogId = 0;
}

// ========== HEALTH POLLING ==========
setInterval(async () => {
    try {
        const d = await (await fetch(API + '/api/health', { signal: AbortSignal.timeout(5000) })).json();
        const p = $('pill'), sp = $('spin'), dt = $('dot'), tx = $('stxt'), pb = $('pbar'), pf = $('pfill');
        if (d.status === 'ready') {
            p.textContent = 'READY'; p.className = 'pill ready'; sp.style.display = 'none'; dt.style.display = 'block'; dt.className = 'sdot green';
            let det = d.detail || 'Ready'; if (d.gpu) det += ' | ' + d.gpu.split(',')[0]; if (d.queue_length > 0) det += ' | ' + d.queue_length + ' queued';
            tx.textContent = det; pb.style.display = 'none'; if (!generating) $('genBtn').disabled = false;
        } else if (d.status === 'loading' || d.status === 'starting') {
            p.textContent = 'LOADING'; p.className = 'pill loading'; sp.style.display = 'block'; dt.style.display = 'none';
            tx.textContent = d.detail || 'Loading...';
            if (d.step && d.total) { pb.style.display = 'block'; pf.style.width = Math.round(d.step / d.total * 100) + '%' } else pb.style.display = 'none';
            if (!generating) $('genBtn').disabled = true;
        } else if (d.status === 'error') {
            p.textContent = 'ERROR'; p.className = 'pill error'; sp.style.display = 'none'; dt.style.display = 'block'; dt.className = 'sdot red'; tx.textContent = d.detail || 'Error';
        }
        // Update LoRA status from health
        if (d.lora) {
            const ls = $('loraStatus');
            if (d.lora.loaded) {
                ls.textContent = '✓ Loaded: ' + d.lora.loaded;
                ls.className = 'lora-status ok';
                $('loraUnloadBtn').style.display = '';
                $('loraScaleRow').style.display = '';
                $('loraRepo').value = d.lora.loaded;
            } else if (d.lora.loading) {
                ls.textContent = '⏳ Loading LoRA...';
                ls.className = 'lora-status busy';
            } else if (d.lora.error) {
                ls.textContent = '✗ ' + d.lora.error;
                ls.className = 'lora-status err';
            }
        }
    } catch (e) { }
}, 3000);

// ========== LORA ==========
async function loraLoad() {
    const repo = $('loraRepo').value.trim();
    if (!repo) { toast('Enter a HuggingFace LoRA repo', 1); return }
    const scale = parseFloat($('loraScale').value) || 1.0;
    $('loraStatus').textContent = '⏳ Loading LoRA...';
    $('loraStatus').className = 'lora-status busy';
    $('loraLoadBtn').disabled = true;
    try {
        const resp = await fetch(API + '/api/lora/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ repo, scale }) });
        const d = await resp.json();
        if (d.ok) {
            $('loraStatus').textContent = '✓ Loaded: ' + repo;
            $('loraStatus').className = 'lora-status ok';
            $('loraUnloadBtn').style.display = '';
            $('loraScaleRow').style.display = '';
            toast('LoRA loaded');
        } else {
            $('loraStatus').textContent = '✗ ' + (d.error || 'Failed');
            $('loraStatus').className = 'lora-status err';
            toast(d.error || 'Failed', 1);
        }
    } catch (e) { $('loraStatus').textContent = '✗ ' + e.message; $('loraStatus').className = 'lora-status err'; }
    finally { $('loraLoadBtn').disabled = false }
}

async function loraUnload() {
    $('loraStatus').textContent = '⏳ Unloading...';
    $('loraStatus').className = 'lora-status busy';
    try {
        const resp = await fetch(API + '/api/lora/unload', { method: 'POST' });
        const d = await resp.json();
        if (d.ok) {
            $('loraStatus').textContent = 'No LoRA loaded';
            $('loraStatus').className = 'lora-status';
            $('loraUnloadBtn').style.display = 'none';
            $('loraScaleRow').style.display = 'none';
            toast('LoRA unloaded');
        } else { toast(d.error || 'Failed', 1) }
    } catch (e) { toast(e.message, 1) }
}

// ========== IMAGE SLOTS ==========
function addSlot(du) { if (!du) return; S.push({ id: nid++, du }); rS() }
function rS() {
    const c = $('slots'); c.innerHTML = '';
    S.forEach((s, i) => {
        const d = document.createElement('div'); d.className = 'slot filled';
        d.innerHTML = `<img src="${s.du}"/><span class="sl">${i + 1}</span><button class="sr" onclick="event.stopPropagation();repS(${s.id})">↻</button><button class="sx" onclick="event.stopPropagation();rmS(${s.id})">×</button>`;
        d.onclick = () => lightbox(s.du);
        d.addEventListener('dragover', e => { e.preventDefault(); d.classList.add('drag-over') });
        d.addEventListener('dragleave', () => d.classList.remove('drag-over'));
        d.addEventListener('drop', e => { e.preventDefault(); d.classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f && f.type.startsWith('image/')) rdF(f, s.id) });
        c.appendChild(d)
    });
    const ab = document.createElement('button'); ab.className = 'slot-add';
    ab.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg> Add Image';
    ab.onclick = pickF; c.appendChild(ab); updLB()
}
function pickF() { const i = document.createElement('input'); i.type = 'file'; i.accept = 'image/*'; i.multiple = true; i.onchange = () => { [...i.files].filter(f => f.type.startsWith('image/')).forEach(f => { const r = new FileReader(); r.onload = e => addSlot(e.target.result); r.readAsDataURL(f) }) }; i.click() }
function rdF(f, sid) { const r = new FileReader(); r.onload = e => { const s = S.find(x => x.id === sid); if (s) { s.du = e.target.result; rS() } }; r.readAsDataURL(f) }
function repS(sid) { const i = document.createElement('input'); i.type = 'file'; i.accept = 'image/*'; i.onchange = () => { if (i.files[0]) rdF(i.files[0], sid) }; i.click() }
function rmS(sid) { S = S.filter(x => x.id !== sid); rS() }
document.addEventListener('paste', e => { [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith('image/')).forEach(f => { const r = new FileReader(); r.onload = ev => addSlot(ev.target.result); r.readAsDataURL(f) }) });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { e.preventDefault(); [...e.dataTransfer.files].filter(f => f.type.startsWith('image/')).forEach(f => { const r = new FileReader(); r.onload = ev => addSlot(ev.target.result); r.readAsDataURL(f) }) });
rS();

// ========== PROGRESS RING HELPER ==========
function setProgress(pct, step, total, perStep, remaining) {
    const overlay = $('genOverlay');
    overlay.classList.add('active');
    // Ring: circumference = 2*PI*45 ≈ 283
    const offset = 283 - (283 * pct / 100);
    $('genRing').style.strokeDashoffset = offset;
    $('genPct').textContent = pct + '%';
    if (step !== undefined && total !== undefined) {
        $('genStep').textContent = `Step ${step} / ${total}`;
    }
    if (perStep !== undefined && perStep !== '?') {
        $('genDetail').textContent = `${perStep}s per step`;
    }
    if (remaining !== undefined && remaining > 0) {
        $('genEta').textContent = `~${Math.round(remaining)}s remaining`;
    } else {
        $('genEta').textContent = '';
    }
}

function hideProgress() {
    $('genOverlay').classList.remove('active');
    $('genRing').style.strokeDashoffset = 283;
    $('genPct').textContent = '0%';
    $('genStep').textContent = '';
    $('genDetail').textContent = '';
    $('genEta').textContent = '';
}

// ========== GENERATE ==========
let genAC = null;
async function gen() {
    if (generating) return;
    const filled = S.filter(s => s.du); if (!filled.length) { toast('Upload at least one image', 1); return }
    const p = $('prompt').value.trim(); if (!p) { toast('Enter a prompt', 1); return }
    const imgs = {}; S.forEach((s, i) => { if (s.du) imgs[String(i)] = s.du });
    const params = {
        images: imgs, prompt: p, negative_prompt: $('neg').value.trim(),
        true_cfg_scale: parseFloat($('cfg').value), guidance_scale: parseFloat($('gs').value),
        num_inference_steps: parseInt($('steps').value), num_images_per_prompt: parseInt($('batch').value),
        width: parseInt($('w').value) || null, height: parseInt($('h').value) || null,
        seed: parseInt($('seed').value), max_sequence_length: parseInt($('msl').value) || null,
        mask: maskDU || null, mask_blur: parseInt($('maskBlur').value) || 0
    };
    genAC = new AbortController(); setG(1);
    let _genJobId = null;
    let _pollTimer = null;
    try {
        const resp = await fetch(API + '/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params), signal: genAC.signal });

        // Try to extract job_id from a custom header (set by server)
        // Also start a polling fallback that updates UI even if SSE is buffered by Colab proxy
        const _pollProgress = async (jid) => {
            if (!jid) return;
            try {
                const pr = await fetch(API + '/api/gen_progress/' + jid, { signal: AbortSignal.timeout(3000) });
                const pd = await pr.json();
                if (pd.type === 'progress') {
                    $('pill').textContent = 'RUNNING'; $('pill').className = 'pill run';
                    const pct = pd.progress || Math.round(pd.step / pd.total * 100);
                    setProgress(pct, pd.step, pd.total, pd.per_step, pd.remaining);
                    $('gpf').style.width = pct + '%';
                    const eta = pd.remaining > 0 ? ' · ~' + Math.round(pd.remaining) + 's' : '';
                    $('gsl').textContent = `Step ${pd.step}/${pd.total} (${pd.per_step || '?'}s/step${eta})`;
                } else if (pd.type === 'queue') {
                    $('genStep').textContent = `Queue position: ${pd.position} of ${pd.queue_length}`;
                }
            } catch(e) {}
        };

        const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = '';
        while (true) {
            const { done, value } = await reader.read(); if (done) break;
            buf += dec.decode(value, { stream: true }); const lines = buf.split('\n'); buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.type === 'error') throw new Error(d.error);
                    else if (d.type === 'init' && d.job_id) {
                        // Got job_id — start polling fallback for Colab proxy buffering
                        _genJobId = d.job_id;
                        if (!_pollTimer) {
                            _pollTimer = setInterval(() => _pollProgress(_genJobId), 1500);
                        }
                    }
                    else if (d.type === 'queue') {
                        setProgress(0, 0, 0, '?', 0);
                        $('genStep').textContent = `Queue position: ${d.position} of ${d.queue_length}`;
                        $('genDetail').textContent = 'Waiting for GPU...';
                        $('gpf').style.width = '0%';
                        $('gsl').textContent = `⏳ Queue: ${d.position}/${d.queue_length}`;
                        $('pill').textContent = 'QUEUED'; $('pill').className = 'pill loading';
                    } else if (d.type === 'done') {
                        hideProgress();
                        showR(d.results); rH();
                        toast('Done!' + (d.elapsed ? ' in ' + d.elapsed + 's' : ''));
                    } else if (d.type === 'progress') {
                        $('pill').textContent = 'RUNNING'; $('pill').className = 'pill run';
                        const pct = d.progress || Math.round(d.step / d.total * 100);
                        setProgress(pct, d.step, d.total, d.per_step, d.remaining);
                        $('gpf').style.width = pct + '%';
                        const eta = d.remaining > 0 ? ' · ~' + Math.round(d.remaining) + 's' : '';
                        $('gsl').textContent = `Step ${d.step}/${d.total} (${d.per_step || '?'}s/step${eta})`;
                    }
                } catch (pe) { if (pe.message && !pe.message.includes('JSON')) throw pe }
            }
        }
    } catch (e) { if (e.name === 'AbortError') toast('Stopped'); else toast(e.message, 1) }
    finally { if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; } genAC = null; setG(0); hideProgress() }
}
function stopGen() { if (genAC) { genAC.abort(); genAC = null } }
function setG(on) {
    generating = !!on;
    $('genBtn').disabled = on;
    $('genBtn').textContent = on ? 'GENERATING...' : '⚡ Generate';
    $('stopBtn').style.display = on ? 'inline-block' : 'none';
    if (on) {
        $('pill').className = 'pill run'; $('pill').textContent = 'RUNNING';
        $('gpf').style.width = '0%'; $('gsl').textContent = 'Starting...';
        setProgress(0); $('genStep').textContent = 'Preparing...'; $('genDetail').textContent = 'Preprocessing image...';
    } else {
        $('gsl').textContent = '';
    }
}

// ========== OUTPUT DISPLAY ==========
function showR(res, si) {
    if (lassoOn) { lassoOn = false; $('lassoT').style.background = ''; $('lassoT').style.color = '' }
    $('lwrap').style.display = 'none'; $('ce').style.display = 'none';
    window._r = res; window._si = si || 0; renO();
}
function renO() {
    const r = window._r; if (!r || !r.length) return; const sel = window._si || 0; const g = $('og'); g.style.display = 'flex'; g.innerHTML = '';
    const m = document.createElement('div'); m.className = 'og-main'; m.innerHTML = `<img src="${r[sel]}"/>`; m.onclick = () => lightbox(r[sel]); g.appendChild(m);
    if (r.length > 1) { const st = document.createElement('div'); st.className = 'og-thumbs'; r.forEach((b, i) => { const t = document.createElement('div'); t.className = 'og-thumb' + (i === sel ? ' active' : ''); t.innerHTML = `<img src="${b}"/>`; t.onclick = () => { window._si = i; renO() }; st.appendChild(t) }); g.appendChild(st) }
    const bar = document.createElement('div'); bar.className = 'og-bar'; let sb = '';
    S.forEach((s, si) => { sb += `<button class="slotb" onclick="event.stopPropagation();useAs(${sel},${s.id})"><img src="${s.du}"/>Replace ${si + 1}</button>` });
    bar.innerHTML = `<button onclick="dlImg(${sel})">💾 Save</button><div class="sep"></div>${sb}<button class="newb" onclick="useAsNew(${sel})">+ New Slot</button>`; g.appendChild(bar)
}
function dlImg(i) { const a = document.createElement('a'); a.href = window._r[i]; a.download = 'output_' + i + '.png'; a.click() }
function useAs(i, sid) { const s = S.find(x => x.id === sid); if (s) { s.du = window._r[i]; rS(); toast('Replaced'); renO() } }
function useAsNew(i) { addSlot(window._r[i]); toast('Added'); renO() }

// ========== HISTORY ==========
async function rH() {
    try {
        const d = await (await fetch(API + '/api/history')).json(); const s = $('hs'); s.innerHTML = '';
        (d.history || []).forEach(h => {
            const e = document.createElement('div'); e.className = 'he';
            let th = (h.input_images || []).slice(0, 1).map(t => `<img class="inp" src="${t}"/>`).join('');
            th += (h.outputs || []).slice(0, 3).map(o => `<img class="out" src="${o}"/>`).join('');
            e.innerHTML = `<div class="hp">${h.prompt || '(no prompt)'}</div><div class="ht">${th}</div><span class="hx" onclick="event.stopPropagation();delH('${h.id}')">×</span>`;
            e.onclick = () => lH(h); s.appendChild(e)
        })
    } catch (e) { }
}
async function delH(id) { try { await fetch(API + '/api/history/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) }); rH(); toast('Removed') } catch (e) { } }
function lH(h) { $('prompt').value = h.prompt || ''; if (h.input_images?.length) { S = []; nid = 0; h.input_images.forEach(i => addSlot(i)) } if (h.outputs?.length) showR(h.outputs); toast('Restored') }
rH();

// ========== LASSO ==========
function toggleLasso() { const f = S.find(s => s.du); if (!f) { toast('Upload an image first', 1); return } lassoOn = !lassoOn; $('lassoT').style.background = lassoOn ? 'var(--gold)' : ''; $('lassoT').style.color = lassoOn ? '#0a0a0a' : ''; if (lassoOn) showLC(f.du); else { $('lwrap').style.display = 'none'; const h = $('og').children.length > 0; $('og').style.display = h ? 'flex' : 'none'; $('ce').style.display = h ? 'none' : '' } }
function showLC(src) {
    $('ce').style.display = 'none'; $('og').style.display = 'none'; $('lwrap').style.display = 'block'; const img = $('limg');
    img.onload = () => {
        const mW = $('C').clientWidth - 48, mH = $('C').clientHeight - 80;
        let w = img.naturalWidth, h = img.naturalHeight; const sc = Math.min(1, mW / w, mH / h);
        w = Math.round(w * sc); h = Math.round(h * sc);
        img.style.width = w + 'px'; img.style.height = h + 'px';
        ['ldraw', 'lmask'].forEach(id => { const c = $(id); c.width = w; c.height = h; c.style.width = w + 'px'; c.style.height = h + 'px' }); renMP()
    }; img.src = src
}
(function () {
    const dc = $('ldraw'); function pt(e) { const r = dc.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top } }
    function endDraw() { if (!lassoDrawing) return; lassoDrawing = false; if (curPoly.length > 2) lassoPolys.push([...curPoly]); curPoly = []; renMP(); drawL() }
    dc.addEventListener('mousedown', e => { if (!lassoOn) return; lassoDrawing = true; curPoly = [pt(e)]; drawL() });
    dc.addEventListener('mousemove', e => { if (!lassoDrawing) return; curPoly.push(pt(e)); drawL() });
    dc.addEventListener('mouseup', endDraw); dc.addEventListener('mouseleave', endDraw);
    dc.addEventListener('touchstart', e => { if (!lassoOn) return; e.preventDefault(); lassoDrawing = true; curPoly = [pt(e.touches[0])]; drawL() }, { passive: false });
    dc.addEventListener('touchmove', e => { if (!lassoDrawing) return; e.preventDefault(); curPoly.push(pt(e.touches[0])); drawL() }, { passive: false });
    dc.addEventListener('touchend', endDraw)
})();
function drawL() {
    const dc = $('ldraw'), ctx = dc.getContext('2d'); ctx.clearRect(0, 0, dc.width, dc.height);
    ctx.fillStyle = 'rgba(212,160,23,0.25)'; ctx.strokeStyle = '#d4a017'; ctx.lineWidth = 2;
    lassoPolys.forEach(p => { ctx.beginPath(); p.forEach((pt, i) => i ? ctx.lineTo(pt.x, pt.y) : ctx.moveTo(pt.x, pt.y)); ctx.closePath(); ctx.fill(); ctx.stroke() });
    if (curPoly.length > 1) { ctx.beginPath(); ctx.strokeStyle = '#f0c840'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]); curPoly.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke(); ctx.setLineDash([]) }
}
function renMP() {
    const mp = $('lmask'), ctx = mp.getContext('2d'); ctx.clearRect(0, 0, mp.width, mp.height);
    if (!lassoPolys.length) { maskDU = null; return } ctx.fillStyle = 'rgba(212,160,23,0.35)';
    lassoPolys.forEach(p => { ctx.beginPath(); p.forEach((pt, i) => i ? ctx.lineTo(pt.x, pt.y) : ctx.moveTo(pt.x, pt.y)); ctx.closePath(); ctx.fill() }); buildMDU()
}
function buildMDU() {
    const f = S.find(s => s.du); if (!f) return; const ti = new window.Image();
    ti.onload = () => {
        const oc = document.createElement('canvas'); oc.width = ti.naturalWidth; oc.height = ti.naturalHeight;
        const ox = oc.getContext('2d'); ox.fillStyle = '#000'; ox.fillRect(0, 0, oc.width, oc.height); ox.fillStyle = '#fff';
        const dc = $('ldraw'); const sx = oc.width / dc.width, sy = oc.height / dc.height;
        lassoPolys.forEach(p => { ox.beginPath(); p.forEach((pt, i) => i ? ox.lineTo(pt.x * sx, pt.y * sy) : ox.moveTo(pt.x * sx, pt.y * sy)); ox.closePath(); ox.fill() });
        maskDU = oc.toDataURL('image/png')
    }; ti.src = f.du
}
function clearMask() { lassoPolys = []; curPoly = []; maskDU = null; renMP(); drawL(); toast('Mask cleared') }
function undoLasso() { if (lassoPolys.length) { lassoPolys.pop(); renMP(); drawL(); toast('Undo') } }
function updLB() { $('lbar').style.display = S.some(s => s.du) ? 'flex' : 'none'; if (!S.some(s => s.du) && lassoOn) { lassoOn = false; $('lwrap').style.display = 'none'; $('lassoT').style.background = ''; $('lassoT').style.color = '' } }
document.addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); gen() } if (e.key === 'Escape' && generating) { e.preventDefault(); stopGen() } });
