// ============================================================
// 🔺 TRELLIS.2 — Client JavaScript (app.js)
// Tab switching, dropzone, polling, render results, keepalive.
// ============================================================

// ── Utility helpers ──
function $(id) { return document.getElementById(id); }
function enc(s) { return encodeURIComponent(s); }
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function show(id) { $(id).classList.add('active'); }
function hide(id) { $(id).classList.remove('active'); }
function fmtTime(s) {
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ':' + String(sec).padStart(2, '0');
}

// ── Tab switching ──
function switchTab(name, el) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    $('tab-' + name).classList.add('active');
}

// ── Console toggle ──
function toggleConsole() {
    $('consoleHead3d').classList.toggle('open');
    $('consoleBody3d').classList.toggle('open');
}

// ══════════════════════════════════════════════════════════════
// RENDER MODE SELECTOR
// ══════════════════════════════════════════════════════════════

let selectedRenderMode = 'video';

function selectRender(el) {
    document.querySelectorAll('.render-mode').forEach(m => m.classList.remove('selected'));
    el.classList.add('selected');
    selectedRenderMode = el.dataset.mode;

    // Show/hide mode-specific settings panels
    const rts = $('rtsSettings');
    const doom = $('doomSettings');
    rts.classList.remove('visible');
    doom.classList.remove('visible');
    if (selectedRenderMode === 'rts_sprite') {
        rts.classList.add('visible');
    } else if (selectedRenderMode === 'doom_sprite') {
        doom.classList.add('visible');
    }
}

// ══════════════════════════════════════════════════════════════
// DROPZONE
// ══════════════════════════════════════════════════════════════

function initDrop(zId, iId, tId, bId, arr) {
    const z = $(zId), inp = $(iId);
    ['dragenter', 'dragover'].forEach(e =>
        z.addEventListener(e, ev => { ev.preventDefault(); z.classList.add('over'); })
    );
    ['dragleave', 'drop'].forEach(e =>
        z.addEventListener(e, ev => { ev.preventDefault(); z.classList.remove('over'); })
    );
    z.addEventListener('drop', ev => addF(ev.dataTransfer.files, arr, tId, bId));
    inp.addEventListener('change', ev => { addF(ev.target.files, arr, tId, bId); inp.value = ''; });
}

function addF(fl, arr, tId, bId) {
    for (const f of fl) {
        if (f.type.startsWith('image/')) arr.push(f);
    }
    renderT(arr, tId, bId);
}

function renderT(arr, tId, bId) {
    const t = $(tId);
    t.innerHTML = '';
    arr.forEach((f, i) => {
        const d = document.createElement('div');
        d.className = 'thumb';
        const img = document.createElement('img');
        img.src = URL.createObjectURL(f);
        const x = document.createElement('button');
        x.className = 'thumb-x';
        x.textContent = '×';
        x.onclick = () => { arr.splice(i, 1); renderT(arr, tId, bId); };
        d.append(img, x);
        t.append(d);
    });
    $(bId).disabled = arr.length === 0;
}

let files3d = [], filesRmbg = [];
initDrop('dropzone3d', 'fileInput3d', 'thumbs3d', 'genBtn3d', files3d);
initDrop('dropzoneRmbg', 'fileInputRmbg', 'thumbsRmbg', 'genBtnRmbg', filesRmbg);

// ══════════════════════════════════════════════════════════════
// KEEPALIVE
// ══════════════════════════════════════════════════════════════

setInterval(async () => {
    try {
        const r = await fetch('/api/keepalive');
        $('keepaliveBadge').style.color = r.ok ? 'var(--green)' : 'var(--red)';
    } catch (e) {
        $('keepaliveBadge').style.color = 'var(--red)';
    }
    setTimeout(() => { $('keepaliveBadge').style.color = 'var(--gray)'; }, 2000);
}, 60000);

// ══════════════════════════════════════════════════════════════
// POLLING
// ══════════════════════════════════════════════════════════════

let timers = {}, localStart = {};

function poll(jobId, type, cfg) {
    if (timers[type]) clearInterval(timers[type]);
    if (!localStart[type]) localStart[type] = Date.now();

    timers[type] = setInterval(async () => {
        try {
            const r = await fetch('/api/status/' + jobId);
            const d = await r.json();
            const p = d.progress || {};
            const elapsed = p.elapsed || ((Date.now() - localStart[type]) / 1000);

            $(cfg.timer).textContent = fmtTime(elapsed);
            const pct = p.pct || 0;
            $(cfg.fill).style.width = pct + '%';
            $(cfg.pct).textContent = Math.round(pct) + '%';
            $(cfg.phase).textContent = p.phase || '';

            if (cfg.image && p.image_num && p.total) {
                $(cfg.image).textContent = 'Image ' + p.image_num + ' of ' + p.total +
                    (p.name ? ' — ' + p.name : '');
            }

            if (d.status === 'done') {
                $(cfg.status).innerHTML = '<span class="done-icon">✅</span> Complete';
            } else {
                $(cfg.status).innerHTML = '<span class="spinner"></span> ' + cfg.statusText;
            }

            if (d.log) {
                const b = $(cfg.log);
                b.textContent = d.log.join('\n');
                b.scrollTop = b.scrollHeight;
            }

            if (cfg.console) {
                const cr = await fetch('/api/console');
                const cd = await cr.json();
                const cel = $(cfg.console);
                cel.textContent = cd.lines.join('\n');
                cel.scrollTop = cel.scrollHeight;
            }

            if (d.status === 'done') {
                clearInterval(timers[type]);
                timers[type] = null;
                delete localStart[type];
                if (cfg.btn) {
                    cfg.btn.disabled = false;
                    cfg.btn.textContent = cfg.btnText;
                }
                cfg.renderFn(d.results || []);
            }
        } catch (e) {
            console.error(e);
        }
    }, 800);
}

// ══════════════════════════════════════════════════════════════
// GENERATE 3D
// ══════════════════════════════════════════════════════════════

async function startGen() {
    if (!files3d.length) return;
    const btn = $('genBtn3d');
    btn.disabled = true;
    btn.textContent = 'Uploading images...';

    const autoRmbg = document.querySelector('input[name="autoRmbg"]:checked').value === 'on';

    const fd = new FormData();
    files3d.forEach(f => fd.append('images', f));
    fd.append('settings', JSON.stringify({
        output_dir: $('sOutDir').value,
        fps: parseInt($('sFps').value),
        texture_size: parseInt($('sTexture').value),
        decimate_target: parseInt($('sDecimate').value),
        remesh: $('sRemesh').checked,
        remesh_band: parseFloat($('sRemeshBand').value),
        render_mode: selectedRenderMode,
        video_resolution: 512,
        sprite_directions: parseInt($('sSpriteDirections').value),
        sprite_size: parseInt($('sSpriteSize').value),
        sprite_pitch: parseFloat($('sSpritePitch').value),
        doom_directions: parseInt($('sDoomDirections').value),
        doom_size: parseInt($('sDoomSize').value),
        doom_pitch: parseFloat($('sDoomPitch').value),
        auto_rmbg: autoRmbg,
    }));

    try {
        const r = await fetch('/api/generate', { method: 'POST', body: fd });
        const d = await r.json();
        if (!d.job_id) throw new Error(d.error || 'Failed');

        btn.textContent = 'Generating...';
        show('progressPanel3d');
        show('logBox3d');
        $('logBox3d').textContent = '';
        hide('results3d');
        $('resultsList3d').innerHTML = '';
        $('pFill3d').style.width = '0%';
        $('pPct3d').textContent = '0%';
        localStart['generate'] = Date.now();

        poll(d.job_id, 'generate', {
            timer: 'pTimer3d', fill: 'pFill3d', pct: 'pPct3d',
            phase: 'pPhase3d', status: 'pStatus3d', statusText: 'Generating...',
            image: 'pImage3d', log: 'logBox3d', console: 'consoleScroll3d',
            btn: btn, btnText: 'Generate 3D models →', renderFn: render3d,
        });
    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Generate 3D models →';
    }
}

// ══════════════════════════════════════════════════════════════
// RENDER 3D RESULTS
// ══════════════════════════════════════════════════════════════

function render3d(results) {
    if (!results.length) return;
    show('results3d');
    const l = $('resultsList3d');
    l.innerHTML = '';

    results.forEach(r => {
        const c = document.createElement('div');
        c.className = 'result-card';
        let mediaHtml = '';

        if ((r.media_type === 'rts_sprite' || r.media_type === 'doom_sprite') && r.media) {
            mediaHtml = `<img class="result-img" src="/api/file?p=${enc(r.media)}" alt="${esc(r.name)} sprite sheet" style="image-rendering:pixelated">`;
        } else if (r.media && r.media_type === 'video') {
            mediaHtml = `<video src="/api/file?p=${enc(r.media)}" controls playsinline autoplay muted loop></video>`;
        } else if (r.media && r.media_type === 'image') {
            mediaHtml = `<img class="result-img" src="/api/file?p=${enc(r.media)}" alt="${esc(r.name)}">`;
        } else {
            mediaHtml = '<div class="no-preview">No preview — GLB only</div>';
        }

        // Action buttons
        let btns = `<a class="dl-btn" href="/api/file?p=${enc(r.glb)}" download="${esc(r.name)}.glb">GLB</a>`;
        if (r.media_type === 'rts_sprite' && r.media) {
            btns = `<a class="dl-btn blue" href="/api/file?p=${enc(r.media)}" download="${esc(r.name)}_spritesheet.png">Sprite Sheet</a>` + btns;
        } else if (r.media_type === 'doom_sprite' && r.media) {
            btns = `<a class="dl-btn crimson" href="/api/file?p=${enc(r.media)}" download="${esc(r.name)}_doom_sheet.png">Doom Sheet</a>` + btns;
        } else if (r.media) {
            const ext = r.media_type === 'video' ? 'mp4' : 'png';
            btns = `<a class="dl-btn" href="/api/file?p=${enc(r.media)}" download="${esc(r.name)}.${ext}">${ext.toUpperCase()}</a>` + btns;
        }

        c.innerHTML = mediaHtml +
            `<div class="result-info"><div class="result-name">${esc(r.name)}</div><div class="result-actions">${btns}</div></div>`;

        // Sprite frame gallery for RTS or Doom mode
        if (r.sprite_frames && r.sprite_frames.length) {
            const isDoom = r.media_type === 'doom_sprite';
            const gal = document.createElement('div');
            gal.className = 'sprite-gallery';
            gal.innerHTML = `<div class="sg-title">${isDoom ? '👹 Doom sprite angles' : '🎮 Individual direction frames'} (click to download)</div>`;

            r.sprite_frames.forEach(fp => {
                const fname = fp.split('/').pop();
                const a = document.createElement('a');
                a.href = `/api/file?p=${enc(fp)}`;
                a.download = fname;
                a.className = 'sprite-frame';
                a.title = fname;
                const img = document.createElement('img');
                img.src = `/api/file?p=${enc(fp)}`;
                img.alt = fname;
                img.loading = 'lazy';
                a.appendChild(img);
                gal.appendChild(a);
            });
            c.appendChild(gal);
        }

        l.append(c);
    });
}

// ══════════════════════════════════════════════════════════════
// REMOVE BACKGROUND
// ══════════════════════════════════════════════════════════════

async function startRmbg() {
    if (!filesRmbg.length) return;
    const btn = $('genBtnRmbg');
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    const fd = new FormData();
    filesRmbg.forEach(f => fd.append('images', f));

    try {
        const r = await fetch('/api/rmbg', { method: 'POST', body: fd });
        const d = await r.json();
        if (!d.job_id) throw new Error(d.error || 'Failed');

        btn.textContent = 'Processing...';
        show('progressPanelRmbg');
        show('logBoxRmbg');
        $('logBoxRmbg').textContent = '';
        hide('resultsRmbg');
        $('resultsListRmbg').innerHTML = '';
        $('pFillRmbg').style.width = '0%';
        $('pPctRmbg').textContent = '0%';
        localStart['rmbg'] = Date.now();

        poll(d.job_id, 'rmbg', {
            timer: 'pTimerRmbg', fill: 'pFillRmbg', pct: 'pPctRmbg',
            phase: 'pPhaseRmbg', status: 'pStatusRmbg', statusText: 'Processing...',
            image: null, log: 'logBoxRmbg', console: null,
            btn: btn, btnText: 'Remove backgrounds →', renderFn: renderRmbg,
        });
    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Remove backgrounds →';
    }
}

function renderRmbg(results) {
    if (!results.length) return;
    show('resultsRmbg');
    const l = $('resultsListRmbg');
    l.innerHTML = '';

    results.forEach(r => {
        const c = document.createElement('div');
        c.className = 'result-card';
        c.innerHTML =
            `<img class="result-img" src="/api/file?p=${enc(r.file)}" alt="${esc(r.name)}">` +
            `<div class="result-info">` +
            `<div class="result-name">${esc(r.name)}_transparent.png</div>` +
            `<div class="result-actions"><a class="dl-btn" href="/api/file?p=${enc(r.file)}" download="${esc(r.name)}_transparent.png">Download PNG</a></div>` +
            `</div>`;
        l.append(c);
    });
}

// ══════════════════════════════════════════════════════════════
// RECONNECT (resume polling if page reloads mid-job)
// ══════════════════════════════════════════════════════════════

async function tryReconnect() {
    try {
        const r = await fetch('/api/active');
        const d = await r.json();

        if (d.generate) {
            show('progressPanel3d');
            show('logBox3d');
            $('genBtn3d').disabled = true;
            $('genBtn3d').textContent = 'Generating...';
            localStart['generate'] = Date.now();
            poll(d.generate, 'generate', {
                timer: 'pTimer3d', fill: 'pFill3d', pct: 'pPct3d',
                phase: 'pPhase3d', status: 'pStatus3d', statusText: 'Generating...',
                image: 'pImage3d', log: 'logBox3d', console: 'consoleScroll3d',
                btn: $('genBtn3d'), btnText: 'Generate 3D models →', renderFn: render3d,
            });
        }

        if (d.rmbg) {
            switchTab('rmbg', document.querySelector('[data-tab=rmbg]'));
            show('progressPanelRmbg');
            show('logBoxRmbg');
            $('genBtnRmbg').disabled = true;
            $('genBtnRmbg').textContent = 'Processing...';
            localStart['rmbg'] = Date.now();
            poll(d.rmbg, 'rmbg', {
                timer: 'pTimerRmbg', fill: 'pFillRmbg', pct: 'pPctRmbg',
                phase: 'pPhaseRmbg', status: 'pStatusRmbg', statusText: 'Processing...',
                image: null, log: 'logBoxRmbg', console: null,
                btn: $('genBtnRmbg'), btnText: 'Remove backgrounds →', renderFn: renderRmbg,
            });
        }
    } catch (e) { }
}

tryReconnect();
