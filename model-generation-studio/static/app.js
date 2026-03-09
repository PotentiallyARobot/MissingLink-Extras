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
let lastHwLogPath = null;

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
    bar.innerHTML = '';
    completedModels.forEach((m, i) => {
        const btn = document.createElement('button');
        btn.className = 'model-tab' + (i === activeModelIdx ? ' active' : '');
        btn.title = m.name;
        btn.onclick = () => selectModel(i);

        const label = document.createElement('span');
        label.className = 'mt-label';
        label.textContent = m.name;
        btn.appendChild(label);

        const x = document.createElement('span');
        x.className = 'mt-close';
        x.textContent = '×';
        x.onclick = (e) => { e.stopPropagation(); closeModel(i); };
        btn.appendChild(x);

        bar.appendChild(btn);
    });
    const spacer = document.createElement('div');
    spacer.className = 'topbar-spacer';
    bar.appendChild(spacer);
}

function closeModel(idx) {
    completedModels.splice(idx, 1);
    if (completedModels.length === 0) {
        activeModelIdx = -1;
        rebuildModelTabs();
        // Show empty state
        $('canvasEmpty').style.display = '';
        $('canvasMedia').classList.remove('active');
        $('canvasNoRender').classList.remove('active');
        $('spriteStrip').classList.remove('active');
        if (window.viewer3d) window.viewer3d.hide();
        $('barName').textContent = '—';
        $('barActions').innerHTML = '';
    } else {
        if (activeModelIdx >= completedModels.length) {
            activeModelIdx = completedModels.length - 1;
        } else if (activeModelIdx > idx) {
            activeModelIdx--;
        } else if (activeModelIdx === idx) {
            activeModelIdx = Math.min(idx, completedModels.length - 1);
        }
        rebuildModelTabs();
        selectModel(activeModelIdx);
    }
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

    // Hide 3D viewer if open
    if (window.viewer3d) window.viewer3d.hide();

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

    // View 3D button
    const v3d = document.createElement('button');
    v3d.className = 'bar-btn green';
    v3d.innerHTML = '🔺 View 3D';
    v3d.onclick = () => {
        if (window.viewer3d && model.glb) {
            window.viewer3d.loadFromUrl('/api/file?p=' + enc(model.glb), model.name);
        }
    };
    acts.appendChild(v3d);

    // Edit button — opens edit tab
    if (model.stage_cache) {
        const editBtn = document.createElement('button');
        editBtn.className = 'bar-btn outline';
        editBtn.innerHTML = '🔧 Edit';
        editBtn.onclick = () => {
            // Switch to Edit tab
            const editTab = document.querySelector('.sidebar-tab[data-panel="panelEdit"]');
            if (editTab) switchSidebarTab(editTab);
            refreshEditPanel();
        };
        acts.appendChild(editBtn);
    }

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

    // HW usage log download button
    if (lastHwLogPath) {
        const dlHw = document.createElement('a');
        dlHw.href = '/api/file?p=' + enc(lastHwLogPath);
        dlHw.download = 'hw_usage_log.csv';
        dlHw.className = 'bar-btn outline';
        dlHw.innerHTML = '📊 HW Log';
        dlHw.title = 'Download GPU/CPU usage log (CSV)';
        acts.appendChild(dlHw);
    }

    // Close dropdown on outside click
    document.addEventListener('click', () => {
        document.querySelectorAll('.rerender-dropdown.open').forEach(d => d.classList.remove('open'));
    }, { once: false });
}

// ══════════════════════════════════════════════════════════════
// RE-RENDER (POST-HOC) — uses cached render mesh
// ══════════════════════════════════════════════════════════════

async function requestRerender(model, mode) {
    if (!model.render_mesh) {
        alert(
            `Cannot re-render "${model.name}" — no cached render mesh found.\n\n` +
            `This model was generated before mesh caching was added, ` +
            `or the cache file was deleted. Re-generate the model to enable re-rendering.`
        );
        return;
    }

    // Show progress
    $('canvasProgress').classList.add('active');
    $('cpPhase').textContent = `Re-rendering as ${mode}…`;
    $('cpDetail').textContent = model.name;
    $('cpFill').style.width = '0%';
    $('cpPct').textContent = '0%';
    ensureHwPolling();

    try {
        const r = await fetch('/api/rerender', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                render_mesh: model.render_mesh,
                name: model.name,
                mode: mode,
                output_dir: $('sOutDir').value,
                fps: parseInt($('sFps').value),
                sprite_directions: parseInt($('sSpriteDirections').value),
                sprite_size: parseInt($('sSpriteSize').value),
                sprite_pitch: parseFloat($('sSpritePitch').value),
                doom_directions: parseInt($('sDoomDirections').value),
                doom_size: parseInt($('sDoomSize').value),
                doom_pitch: parseFloat($('sDoomPitch').value),
            }),
        });
        const d = await r.json();
        if (!d.job_id) throw new Error(d.error || 'Failed');

        // Poll the rerender job
        pollRerender(d.job_id, model);
    } catch (e) {
        alert('Re-render failed: ' + e.message);
        $('canvasProgress').classList.remove('active');
        stopHwPolling();
    }
}

function pollRerender(jobId, originalModel) {
    const iv = setInterval(async () => {
        try {
            const r = await fetch('/api/status/' + jobId);
            const d = await r.json();
            const p = d.progress || {};

            $('cpPhase').textContent = p.phase || '';
            $('cpFill').style.width = (p.pct || 0) + '%';
            $('cpPct').textContent = Math.round(p.pct || 0) + '%';

            if (d.log) {
                $('consoleJob').textContent = d.log.join('\n');
                $('consoleJob').scrollTop = $('consoleJob').scrollHeight;
            }

            if (d.status === 'done') {
                clearInterval(iv);
                $('canvasProgress').classList.remove('active');
                stopHwPolling();

                // Update the model in our list with new media
                if (d.results && d.results.length > 0) {
                    const newResult = d.results[0];
                    const idx = completedModels.findIndex(m => m.name === originalModel.name);
                    if (idx >= 0) {
                        // Preserve render_mesh, update media
                        completedModels[idx] = {
                            ...completedModels[idx],
                            ...newResult,
                            render_mesh: completedModels[idx].render_mesh || newResult.render_mesh,
                        };
                        selectModel(idx);
                    } else {
                        completedModels.push(newResult);
                        rebuildModelTabs();
                        selectModel(completedModels.length - 1);
                    }
                }
            }
        } catch (e) {
            console.error('rerender poll error:', e);
        }
    }, 800);
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
                    stopHwPolling();

                    // Store HW log path for download
                    if (d.hw_log) {
                        lastHwLogPath = d.hw_log;
                    }

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

        // Start HW monitoring during generation
        ensureHwPolling();

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
// EDITOR: STAGE INSPECTION + RETEXTURE
// ══════════════════════════════════════════════════════════════

let editRefFile = null;
let selectedLockMode = 'lock_geometry';
let currentStagesData = null;  // cached /api/stages response
let editHistory = [];  // track retexture lineage

function selectLock(el) {
    document.querySelectorAll('.lock-opt').forEach(o => o.classList.remove('sel'));
    el.classList.add('sel');
    selectedLockMode = el.dataset.lock;
    updateRetexButton();
}

// Init dropzone for edit panel reference image
(function initEditDropzone() {
    const z = $('dropzoneEdit'), inp = $('fileInputEdit');
    ['dragenter', 'dragover'].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.remove('over'); }));
    z.addEventListener('drop', ev => {
        const f = ev.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) setEditRef(f);
    });
    inp.addEventListener('change', ev => {
        if (ev.target.files[0]) setEditRef(ev.target.files[0]);
        inp.value = '';
    });
})();

function setEditRef(file) {
    editRefFile = file;
    const t = $('thumbsEdit');
    t.innerHTML = '';
    const d = document.createElement('div'); d.className = 'thumb';
    const img = document.createElement('img'); img.src = URL.createObjectURL(file);
    const x = document.createElement('button'); x.className = 'thumb-x'; x.textContent = '×';
    x.onclick = () => { editRefFile = null; t.innerHTML = ''; updateRetexButton(); };
    d.append(img, x); t.append(d);
    updateRetexButton();
}

function updateRetexButton() {
    const btn = $('retexBtn');
    btn.disabled = !editRefFile || !currentStagesData;
}

// Called whenever activeModelIdx changes — refresh edit panel
async function refreshEditPanel() {
    const noModel = $('editNoModel');
    const modelInfo = $('editModelInfo');
    const locksSection = $('editLocksSection');
    const refSection = $('editRefSection');
    const actionSection = $('editActionSection');
    const historySection = $('editHistorySection');

    if (activeModelIdx < 0 || !completedModels[activeModelIdx]) {
        noModel.style.display = '';
        modelInfo.style.display = 'none';
        locksSection.style.display = 'none';
        refSection.style.display = 'none';
        actionSection.style.display = 'none';
        historySection.style.display = 'none';
        currentStagesData = null;
        return;
    }

    const model = completedModels[activeModelIdx];
    noModel.style.display = 'none';
    modelInfo.style.display = '';
    $('editModelName').textContent = model.name;

    // Fetch stages if stage_cache exists
    if (!model.stage_cache) {
        $('stageList').innerHTML = '<div class="enm-text">No stage cache for this model.<br>Re-generate to enable editing.</div>';
        locksSection.style.display = 'none';
        refSection.style.display = 'none';
        actionSection.style.display = 'none';
        currentStagesData = null;
        return;
    }

    try {
        const r = await fetch('/api/stages?dir=' + enc(model.stage_cache));
        const data = await r.json();
        currentStagesData = data;
        renderStageList(data);
        renderCapabilities(data);
        renderLockControls(data);

        locksSection.style.display = '';
        refSection.style.display = '';
        actionSection.style.display = '';
        updateRetexButton();

        // Build edit history from completedModels lineage
        refreshEditHistory(model);
    } catch (e) {
        console.error('Failed to fetch stages:', e);
        $('stageList').innerHTML = '<div class="enm-text">Error loading stages.</div>';
        currentStagesData = null;
    }
}

function renderStageList(data) {
    const list = $('stageList');
    list.innerHTML = '';

    const STAGE_LABELS = {
        'sparse_structure': '🏗 Sparse Structure',
        'shape_slat': '🔷 Shape Latent',
        'tex_slat': '🎨 Texture Latent',
        'image_cond_512': '🖼 Conditioning 512',
        'image_cond_1024': '🖼 Conditioning 1024',
        'decoded_mesh': '📦 Decoded Mesh',
    };
    const DISPLAY_ORDER = ['sparse_structure', 'shape_slat', 'tex_slat', 'image_cond_512', 'image_cond_1024', 'decoded_mesh'];

    for (const name of DISPLAY_ORDER) {
        const stage = data.stages[name];
        const el = document.createElement('div');
        el.className = 'stage-item ' + (stage ? 'available' : 'missing');
        el.innerHTML = `
            <div class="si-dot"></div>
            <div class="si-name">${STAGE_LABELS[name] || name}</div>
            ${stage ? `<div class="si-size">${stage.size_mb} MB</div>` : ''}
        `;
        list.appendChild(el);
    }

    // Capability badges
    if (data.capabilities && data.capabilities.length > 0) {
        const badges = document.createElement('div');
        badges.className = 'cap-badges';
        const CAP_LABELS = {
            'lock_structure': 'Lock Structure',
            'lock_geometry': 'Lock Geometry',
            'retexture': 'Retexture',
            'full_regen': 'Full Regen',
        };
        data.capabilities.forEach(c => {
            const b = document.createElement('span');
            b.className = 'cap-badge';
            b.textContent = CAP_LABELS[c] || c;
            badges.appendChild(b);
        });
        list.appendChild(badges);
    }
}

function renderCapabilities(data) {
    // (capabilities are now rendered inside renderStageList)
}

function renderLockControls(data) {
    const caps = data.capabilities || [];
    document.querySelectorAll('.lock-opt').forEach(el => {
        const lock = el.dataset.lock;
        if (caps.includes(lock)) {
            el.classList.remove('disabled');
        } else {
            el.classList.add('disabled');
            el.classList.remove('sel');
        }
    });
    // Auto-select best available lock
    if (caps.includes('lock_geometry')) {
        selectedLockMode = 'lock_geometry';
        document.querySelectorAll('.lock-opt').forEach(o => o.classList.remove('sel'));
        document.querySelector('.lock-opt[data-lock="lock_geometry"]')?.classList.add('sel');
    } else if (caps.includes('lock_structure')) {
        selectedLockMode = 'lock_structure';
        document.querySelectorAll('.lock-opt').forEach(o => o.classList.remove('sel'));
        document.querySelector('.lock-opt[data-lock="lock_structure"]')?.classList.add('sel');
    }
}

function refreshEditHistory(activeModel) {
    const section = $('editHistorySection');
    const list = $('editHistoryList');

    // Find models that share the same base lineage (by original name or stage_cache parent)
    const related = completedModels.filter(m =>
        m.name === activeModel.name ||
        m.name.startsWith(activeModel.name + '_retex') ||
        activeModel.name.startsWith(m.name + '_retex')
    );

    if (related.length <= 1) {
        section.style.display = 'none';
        return;
    }

    section.style.display = '';
    list.innerHTML = '';
    related.forEach((m, i) => {
        const isRetex = m.name.includes('_retex');
        const isActive = m === activeModel;
        const el = document.createElement('div');
        el.className = 'edit-history-item' + (isActive ? ' active' : '');
        el.innerHTML = `
            <span class="ehi-icon">${isRetex ? '🔄' : '🔺'}</span>
            <div class="ehi-info">
                <div class="ehi-name">${esc(m.name)}</div>
            </div>
            <span class="ehi-badge ${isRetex ? 'retex' : 'original'}">${isRetex ? 'retex' : 'original'}</span>
        `;
        el.onclick = () => {
            const idx = completedModels.indexOf(m);
            if (idx >= 0) selectModel(idx);
        };
        list.appendChild(el);
    });
}

async function startRetexture() {
    if (!editRefFile || activeModelIdx < 0) return;
    const model = completedModels[activeModelIdx];
    if (!model || !model.stage_cache) return;

    const btn = $('retexBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="retex-icon">⏳</span> Retexturing…';

    // Generate a retex name
    const timestamp = Date.now().toString(36).slice(-4);
    const retexName = model.name + '_retex_' + timestamp;

    const autoRmbg = document.querySelector('input[name="autoRmbg"]:checked')?.value === 'on';

    const fd = new FormData();
    fd.append('image', editRefFile);
    fd.append('stage_cache', model.stage_cache);
    fd.append('lock_mode', selectedLockMode);
    fd.append('name', retexName);
    fd.append('settings', JSON.stringify({
        output_dir: $('sOutDir').value,
        fps: parseInt($('sFps').value),
        texture_size: parseInt($('sTexture').value),
        decimate_target: parseInt($('sDecimate').value),
        remesh: $('sRemesh').checked,
        remesh_band: parseFloat($('sRemeshBand').value),
        render_mode: selectedRenderMode,
        auto_rmbg: autoRmbg,
    }));

    try {
        const r = await fetch('/api/retexture', { method: 'POST', body: fd });
        const d = await r.json();
        if (!d.job_id) throw new Error(d.error || 'Failed');

        // Show progress
        $('canvasEmpty').style.display = 'none';
        $('canvasProgress').classList.add('active');
        $('cpFill').style.width = '0%';
        $('cpPct').textContent = '0%';
        $('cpPhase').textContent = 'Retexturing…';
        $('cpDetail').textContent = retexName;
        if (!consoleOpen) toggleConsole();
        ensureHwPolling();

        localStart['generate'] = Date.now();
        poll(d.job_id, 'generate', {});
    } catch (e) {
        alert('Retexture failed: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="retex-icon">🔄</span> Retexture →';
    }
}

// Hook into selectModel to refresh edit panel
const _origSelectModel = selectModel;
selectModel = function(idx) {
    _origSelectModel(idx);
    refreshEditPanel();
};

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

// ══════════════════════════════════════════════════════════════
// HARDWARE MONITOR
// ══════════════════════════════════════════════════════════════

let hwPanelOpen = false;
let hwPolling = null;
let hwHistory = []; // last N GPU util values for sparkline

function toggleHwPanel() {
    hwPanelOpen = !hwPanelOpen;
    $('hwPanel').classList.toggle('open', hwPanelOpen);
    $('hwToggle').classList.toggle('active', hwPanelOpen);

    if (hwPanelOpen && !hwPolling) {
        // Start polling at 2s interval
        fetchHw();
        hwPolling = setInterval(fetchHw, 2000);
    } else if (!hwPanelOpen && hwPolling) {
        clearInterval(hwPolling);
        hwPolling = null;
    }
}

// Auto-poll HW during active jobs even if panel is closed (for sparkline)
function ensureHwPolling() {
    if (!hwPolling) {
        hwPolling = setInterval(fetchHw, 2000);
    }
}

function stopHwPolling() {
    if (hwPolling && !hwPanelOpen) {
        clearInterval(hwPolling);
        hwPolling = null;
    }
}

async function fetchHw() {
    try {
        const r = await fetch('/api/hw');
        const hw = await r.json();
        updateHwPanel(hw);
        updateHwSparkline(hw);
    } catch (e) {}
}

function updateHwPanel(hw) {
    // GPU Compute
    const gpuUtil = hw.gpu_util_pct ?? 0;
    $('hwGpuUtil').textContent = gpuUtil + '%';
    const gpuBar = $('hwGpuUtilBar');
    gpuBar.style.width = gpuUtil + '%';
    gpuBar.classList.toggle('hot', gpuUtil > 90);

    // VRAM
    const vramPct = hw.gpu_reserved_pct ?? hw.gpu_alloc_pct ?? 0;
    const vramUsed = hw.gpu_reserved_mb ?? hw.gpu_alloc_mb ?? 0;
    const vramTotal = hw.gpu_total_mb ?? 0;
    $('hwVram').textContent = Math.round(vramUsed) + ' / ' + Math.round(vramTotal) + ' MB';
    const vramBar = $('hwVramBar');
    vramBar.style.width = vramPct + '%';
    vramBar.classList.toggle('hot', vramPct > 90);

    // CPU
    const cpuPct = hw.cpu_pct ?? 0;
    $('hwCpu').textContent = Math.round(cpuPct) + '%';
    const cpuBar = $('hwCpuBar');
    cpuBar.style.width = Math.min(cpuPct, 100) + '%';
    cpuBar.classList.toggle('hot', cpuPct > 90);

    // RAM
    const ramPct = hw.ram_pct ?? 0;
    const ramUsed = hw.ram_used_mb ?? 0;
    const ramTotal = hw.ram_total_mb ?? 0;
    $('hwRam').textContent = Math.round(ramUsed) + ' / ' + Math.round(ramTotal) + ' MB';
    const ramBar = $('hwRamBar');
    ramBar.style.width = ramPct + '%';
    ramBar.classList.toggle('hot', ramPct > 90);

    // Details
    if (hw.gpu_temp_c !== undefined) {
        $('hwTemp').textContent = hw.gpu_temp_c + '°C';
    }
    if (hw.gpu_power_w !== undefined && hw.gpu_power_limit_w !== undefined) {
        $('hwPower').textContent = Math.round(hw.gpu_power_w) + ' / ' + Math.round(hw.gpu_power_limit_w) + 'W';
    }
    if (hw.gpu_name) {
        $('hwGpuName').textContent = hw.gpu_name;
    }

    // Phase
    if (hw.phase) {
        $('hwPhase').textContent = hw.phase;
        $('hwPhase').title = hw.phase;
    } else {
        $('hwPhase').textContent = 'Idle';
    }
}

function updateHwSparkline(hw) {
    const val = hw.gpu_util_pct ?? 0;
    hwHistory.push(val);
    if (hwHistory.length > 5) hwHistory.shift();

    const bars = $('hwSpark').querySelectorAll('span');
    for (let i = 0; i < bars.length; i++) {
        const v = hwHistory[hwHistory.length - bars.length + i] ?? 0;
        const h = Math.max(2, Math.round(v / 100 * 10));
        bars[i].style.height = h + 'px';
        // Color code: green < 50, gold < 80, red >= 80
        if (v >= 80) bars[i].style.background = 'var(--red)';
        else if (v >= 50) bars[i].style.background = 'var(--gold)';
        else bars[i].style.background = 'var(--green)';
    }
}

