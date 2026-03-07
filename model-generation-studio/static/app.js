// ============================================================
// TRELLIS.2 Studio — Client JavaScript
// Single-page app: sidebar controls, canvas viewport with
// model tab navigation, console drawer, re-render support.
// ============================================================

// ── Helpers ──
const $ = id => document.getElementById(id);
const enc = s => encodeURIComponent(s);
const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
const fmtTime = s => { const m = Math.floor(s / 60), sec = Math.floor(s % 60); return m + ':' + String(sec).padStart(2, '0'); };

// ══════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════

let files3d = [];
let filesRmbg = [];
let selectedRenderMode = 'video';

// completedModels: array of result objects from API, enriched with UI state
// Each: { name, glb, media?, media_type?, sprite_frames?, sprite_dir? }
let completedModels = [];
let activeModelIdx = -1;

// ══════════════════════════════════════════════════════════════
// SIDEBAR TAB SWITCHING
// ══════════════════════════════════════════════════════════════

function switchSidebarTab(btn) {
    document.querySelectorAll('.sidebar-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(btn.dataset.panel).classList.add('active');
}

// ══════════════════════════════════════════════════════════════
// CONSOLE DRAWER
// ══════════════════════════════════════════════════════════════

let consoleOpen = false;

function toggleConsole() {
    consoleOpen = !consoleOpen;
    const drawer = $('consoleDrawer');
    const btn = $('consoleToggle');
    if (consoleOpen) {
        document.documentElement.style.setProperty('--console-h', '200px');
        drawer.classList.add('open');
        btn.classList.add('active');
        btn.textContent = '▾ Console';
    } else {
        document.documentElement.style.setProperty('--console-h', '0px');
        drawer.classList.remove('open');
        btn.classList.remove('active');
        btn.textContent = '▸ Console';
    }
}

function switchConsoleTab(btn) {
    document.querySelectorAll('.console-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.console-content').forEach(c => c.style.display = 'none');
    btn.classList.add('active');
    $(btn.dataset.target).style.display = '';
}

// ══════════════════════════════════════════════════════════════
// RENDER MODE
// ══════════════════════════════════════════════════════════════

function selectRender(el) {
    document.querySelectorAll('.render-opt').forEach(m => m.classList.remove('sel'));
    el.classList.add('sel');
    selectedRenderMode = el.dataset.mode;
    $('rtsSettings').classList.toggle('visible', selectedRenderMode === 'rts_sprite');
    $('doomSettings').classList.toggle('visible', selectedRenderMode === 'doom_sprite');
}

// ══════════════════════════════════════════════════════════════
// DROPZONE
// ══════════════════════════════════════════════════════════════

function initDrop(zId, iId, tId, bId, arr) {
    const z = $(zId), inp = $(iId);
    ['dragenter', 'dragover'].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.remove('over'); }));
    z.addEventListener('drop', ev => addFiles(ev.dataTransfer.files, arr, tId, bId));
    inp.addEventListener('change', ev => { addFiles(ev.target.files, arr, tId, bId); inp.value = ''; });
}

function addFiles(fl, arr, tId, bId) {
    for (const f of fl) if (f.type.startsWith('image/')) arr.push(f);
    renderThumbs(arr, tId, bId);
}

function renderThumbs(arr, tId, bId) {
    const t = $(tId);
    t.innerHTML = '';
    arr.forEach((f, i) => {
        const d = document.createElement('div'); d.className = 'thumb';
        const img = document.createElement('img'); img.src = URL.createObjectURL(f);
        const x = document.createElement('button'); x.className = 'thumb-x'; x.textContent = '×';
        x.onclick = () => { arr.splice(i, 1); renderThumbs(arr, tId, bId); };
        d.append(img, x); t.append(d);
    });
    $(bId).disabled = arr.length === 0;
}

initDrop('dropzone3d', 'fileInput3d', 'thumbs3d', 'genBtn3d', files3d);
initDrop('dropzoneRmbg', 'fileInputRmbg', 'thumbsRmbg', 'genBtnRmbg', filesRmbg);

// ══════════════════════════════════════════════════════════════
// KEEPALIVE
// ══════════════════════════════════════════════════════════════

setInterval(async () => {
    try {
        const r = await fetch('/api/keepalive');
        $('keepaliveBadge').querySelector('.dot').style.background = r.ok ? 'var(--green)' : 'var(--red)';
    } catch (e) {
        $('keepaliveBadge').querySelector('.dot').style.background = 'var(--red)';
    }
}, 60000);

// ══════════════════════════════════════════════════════════════
// MODEL TAB NAVIGATION
// ══════════════════════════════════════════════════════════════

function rebuildModelTabs() {
    const bar = $('canvasTopbar');
    // Keep spacer
    bar.innerHTML = '';
    completedModels.forEach((m, i) => {
        const btn = document.createElement('button');
        btn.className = 'model-tab' + (i === activeModelIdx ? ' active' : '');
        btn.textContent = m.name;
        btn.title = m.name;
        btn.onclick = () => selectModel(i);
        bar.appendChild(btn);
    });
    const spacer = document.createElement('div');
    spacer.className = 'topbar-spacer';
    bar.appendChild(spacer);
}

function selectModel(idx) {
    if (idx < 0 || idx >= completedModels.length) return;
    activeModelIdx = idx;
    rebuildModelTabs();
    showModelInCanvas(completedModels[idx]);
}

function showModelInCanvas(model) {
    const empty = $('canvasEmpty');
    const media = $('canvasMedia');
    const noRender = $('canvasNoRender');
    const strip = $('spriteStrip');

    empty.style.display = 'none';
    media.classList.remove('active');
    noRender.classList.remove('active');
    strip.classList.remove('active');
    media.innerHTML = '';
    strip.innerHTML = '';

    // Show media
    if (model.media && model.media_type === 'video') {
        const vid = document.createElement('video');
        vid.src = '/api/file?p=' + enc(model.media);
        vid.controls = true; vid.autoplay = true; vid.muted = true; vid.loop = true; vid.playsInline = true;
        media.appendChild(vid);
        media.classList.add('active');
    } else if (model.media && (model.media_type === 'image' || model.media_type === 'rts_sprite' || model.media_type === 'doom_sprite')) {
        const img = document.createElement('img');
        img.src = '/api/file?p=' + enc(model.media);
        img.alt = model.name;
        if (model.media_type === 'rts_sprite' || model.media_type === 'doom_sprite') {
            img.className = 'sprite-preview';
        }
        media.appendChild(img);
        media.classList.add('active');
    } else {
        noRender.classList.add('active');
    }

    // Sprite strip
    if (model.sprite_frames && model.sprite_frames.length) {
        model.sprite_frames.forEach(fp => {
            const a = document.createElement('a');
            a.href = '/api/file?p=' + enc(fp);
            a.download = fp.split('/').pop();
            a.className = 'sf';
            const img = document.createElement('img');
            img.src = '/api/file?p=' + enc(fp);
            img.loading = 'lazy';
            a.appendChild(img);
            strip.appendChild(a);
        });
        strip.classList.add('active');
    }

    // Bottom bar
    updateBottomBar(model);
}

function updateBottomBar(model) {
    $('barName').textContent = model.name;
    const acts = $('barActions');
    acts.innerHTML = '';

    // Download GLB
    const dlGlb = document.createElement('a');
    dlGlb.href = '/api/file?p=' + enc(model.glb);
    dlGlb.download = model.name + '.glb';
    dlGlb.className = 'bar-btn gold';
    dlGlb.innerHTML = '↓ GLB';
    acts.appendChild(dlGlb);

    // Download media
    if (model.media) {
        const dlM = document.createElement('a');
        dlM.href = '/api/file?p=' + enc(model.media);
        let ext = 'png';
        let label = 'PNG';
        if (model.media_type === 'video') { ext = 'mp4'; label = 'MP4'; }
        else if (model.media_type === 'rts_sprite') { label = 'Sprites'; }
        else if (model.media_type === 'doom_sprite') { label = 'Doom'; }
        dlM.download = model.name + '.' + ext;
        dlM.className = 'bar-btn blue';
        dlM.innerHTML = '↓ ' + label;
        acts.appendChild(dlM);
    }

    // Re-render button
    const wrap = document.createElement('div');
    wrap.className = 'rerender-wrap';
    const rrBtn = document.createElement('button');
    rrBtn.className = 'bar-btn outline';
    rrBtn.innerHTML = '🎬 Re-render';
    rrBtn.onclick = (e) => {
        e.stopPropagation();
        wrap.querySelector('.rerender-dropdown').classList.toggle('open');
    };
    wrap.appendChild(rrBtn);

    const dd = document.createElement('div');
    dd.className = 'rerender-dropdown';
    const modes = [
        { mode: 'snapshot', icon: '📷', label: 'Snapshot' },
        { mode: 'video', icon: '🎬', label: 'Video 360°' },
        { mode: 'perspective', icon: '🔄', label: 'Turntable' },
        { mode: 'rts_sprite', icon: '🎮', label: 'RTS Sprite' },
        { mode: 'doom_sprite', icon: '👹', label: 'Doom Sprite' },
    ];
    modes.forEach(m => {
        const item = document.createElement('button');
        item.className = 'rd-item';
        item.innerHTML = `<span class="rd-icon">${m.icon}</span><span class="rd-label">${m.label}</span>`;
        item.onclick = () => {
            dd.classList.remove('open');
            requestRerender(model, m.mode);
        };
        dd.appendChild(item);
    });
    wrap.appendChild(dd);
    acts.appendChild(wrap);

    // Close dropdown on outside click
    document.addEventListener('click', () => {
        document.querySelectorAll('.rerender-dropdown.open').forEach(d => d.classList.remove('open'));
    }, { once: false });
}

// ══════════════════════════════════════════════════════════════
// RE-RENDER (POST-HOC)
// ══════════════════════════════════════════════════════════════

async function requestRerender(model, mode) {
    // For now, re-renders require re-running the full generation
    // with the same GLB's source image. We create a new job with
    // the same settings but only the specific render mode.
    // The backend will re-use the cached mesh if we upload the
    // original image again. This is the simplest approach that
    // doesn't require new backend endpoints.

    // TODO: When a dedicated /api/rerender endpoint is added,
    // this can send just the GLB path + render mode.

    alert(
        `Re-rendering "${model.name}" as ${mode}.\n\n` +
        `Note: This requires a dedicated /api/rerender endpoint ` +
        `on the backend (not yet implemented). For now, re-generate ` +
        `the model with the desired render mode selected in the sidebar.`
    );
}

// ══════════════════════════════════════════════════════════════
// POLLING
// ══════════════════════════════════════════════════════════════

let timers = {};
let localStart = {};
let lastResultCount = 0;

function poll(jobId, type, cfg) {
    if (timers[type]) clearInterval(timers[type]);
    if (!localStart[type]) localStart[type] = Date.now();
    lastResultCount = 0;

    timers[type] = setInterval(async () => {
        try {
            const r = await fetch('/api/status/' + jobId);
            const d = await r.json();
            const p = d.progress || {};

            // ── Update progress overlay ──
            if (type === 'generate') {
                const pct = p.pct || 0;
                $('cpPhase').textContent = p.phase || '';
                $('cpDetail').textContent = p.image_num && p.total
                    ? `Image ${p.image_num} of ${p.total}` + (p.name ? ` — ${p.name}` : '')
                    : '';
                $('cpFill').style.width = pct + '%';
                $('cpPct').textContent = Math.round(pct) + '% · ' + fmtTime(p.elapsed || (Date.now() - localStart[type]) / 1000);

                // ── Live results: add new models as they complete ──
                if (d.results && d.results.length > lastResultCount) {
                    for (let i = lastResultCount; i < d.results.length; i++) {
                        completedModels.push(d.results[i]);
                    }
                    lastResultCount = d.results.length;
                    rebuildModelTabs();
                    // Auto-select latest model
                    selectModel(completedModels.length - 1);
                }
            }

            // ── Job log ──
            if (d.log) {
                const jl = $('consoleJob');
                jl.textContent = d.log.join('\n');
                jl.scrollTop = jl.scrollHeight;
            }

            // ── System console ──
            try {
                const cr = await fetch('/api/console');
                const cd = await cr.json();
                const sc = $('consoleSystem');
                sc.textContent = cd.lines.join('\n');
                sc.scrollTop = sc.scrollHeight;
            } catch (e) {}

            // ── Done ──
            if (d.status === 'done') {
                clearInterval(timers[type]);
                timers[type] = null;
                delete localStart[type];

                if (type === 'generate') {
                    $('canvasProgress').classList.remove('active');
                    $('genBtn3d').disabled = false;
                    $('genBtn3d').textContent = 'Generate →';

                    // Final sync of results
                    if (d.results) {
                        for (let i = lastResultCount; i < d.results.length; i++) {
                            completedModels.push(d.results[i]);
                        }
                        lastResultCount = d.results.length;
                        rebuildModelTabs();
                        if (completedModels.length > 0 && activeModelIdx < 0) {
                            selectModel(0);
                        }
                    }
                }

                if (type === 'rmbg') {
                    $('genBtnRmbg').disabled = false;
                    $('genBtnRmbg').textContent = 'Remove BG →';
                    if (d.results && d.results.length) renderRmbgResults(d.results);
                }
            }
        } catch (e) {
            console.error('poll error:', e);
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
    btn.textContent = 'Uploading…';

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

        btn.textContent = 'Generating…';

        // Show progress overlay
        $('canvasEmpty').style.display = 'none';
        $('canvasProgress').classList.add('active');
        $('cpFill').style.width = '0%';
        $('cpPct').textContent = '0%';
        $('cpPhase').textContent = 'Starting…';
        $('cpDetail').textContent = '';

        // Open console automatically
        if (!consoleOpen) toggleConsole();

        localStart['generate'] = Date.now();
        poll(d.job_id, 'generate', {});
    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Generate →';
    }
}

// ══════════════════════════════════════════════════════════════
// REMOVE BACKGROUND
// ══════════════════════════════════════════════════════════════

async function startRmbg() {
    if (!filesRmbg.length) return;
    const btn = $('genBtnRmbg');
    btn.disabled = true;
    btn.textContent = 'Uploading…';

    const fd = new FormData();
    filesRmbg.forEach(f => fd.append('images', f));

    try {
        const r = await fetch('/api/rmbg', { method: 'POST', body: fd });
        const d = await r.json();
        if (!d.job_id) throw new Error(d.error || 'Failed');

        btn.textContent = 'Processing…';
        localStart['rmbg'] = Date.now();
        poll(d.job_id, 'rmbg', {});
    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Remove BG →';
    }
}

function renderRmbgResults(results) {
    const section = $('rmbgResultsSection');
    const list = $('rmbgResultsList');
    section.style.display = '';
    list.innerHTML = '';
    results.forEach(r => {
        const div = document.createElement('div');
        div.className = 'rmbg-result';
        div.innerHTML =
            `<img src="/api/file?p=${enc(r.file)}" alt="${esc(r.name)}">` +
            `<span class="rr-name">${esc(r.name)}</span>` +
            `<a class="rr-dl" href="/api/file?p=${enc(r.file)}" download="${esc(r.name)}_transparent.png">↓</a>`;
        list.appendChild(div);
    });
}

// ══════════════════════════════════════════════════════════════
// RECONNECT (resume if page reloads mid-job)
// ══════════════════════════════════════════════════════════════

async function tryReconnect() {
    try {
        const r = await fetch('/api/active');
        const d = await r.json();

        if (d.generate) {
            $('canvasEmpty').style.display = 'none';
            $('canvasProgress').classList.add('active');
            $('genBtn3d').disabled = true;
            $('genBtn3d').textContent = 'Generating…';
            if (!consoleOpen) toggleConsole();
            localStart['generate'] = Date.now();
            poll(d.generate, 'generate', {});
        }

        if (d.rmbg) {
            $('genBtnRmbg').disabled = true;
            $('genBtnRmbg').textContent = 'Processing…';
            localStart['rmbg'] = Date.now();
            poll(d.rmbg, 'rmbg', {});
        }
    } catch (e) {}
}

tryReconnect();
