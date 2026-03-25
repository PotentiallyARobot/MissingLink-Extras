// ============================================================
// app.js — Qwen Camera Studio main application logic
// ============================================================

let uploadedImageId = null;
let uploadedImageUrl = null;
let pipelineReady = false;

// ── Init ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initViewport();
    setupDropZone();
    setupDropdownSync();
    pollStatus();
    setInterval(pollStatus, 3000);
});

// ── Status Polling ────────────────────────────────────────────

async function pollStatus() {
    try {
        const r = await fetch('/api/status');
        const d = await r.json();

        const dot = document.getElementById('status-dot');
        const label = document.getElementById('status-label');
        const btnLoad = document.getElementById('btn-load-model');
        const btnGen = document.getElementById('btn-generate');

        if (d.pipeline_ready) {
            dot.className = 'status-dot ready';
            label.textContent = `Ready · ${d.gpu || 'GPU'} · ${d.vram_used_gb}GB`;
            btnLoad.textContent = '✅ Model Loaded';
            btnLoad.disabled = true;
            pipelineReady = true;
            btnGen.disabled = !uploadedImageId;
        } else if (d.pipeline_loading) {
            dot.className = 'status-dot loading';
            label.textContent = 'Loading model...';
            btnLoad.textContent = '⏳ Loading...';
            btnLoad.disabled = true;
            pipelineReady = false;
            btnGen.disabled = true;
        } else if (d.pipeline_error) {
            dot.className = 'status-dot error';
            label.textContent = `Error: ${d.pipeline_error.substring(0, 60)}`;
            btnLoad.disabled = false;
            btnLoad.textContent = '⚠ Retry Load';
            pipelineReady = false;
            btnGen.disabled = true;
        } else {
            dot.className = 'status-dot';
            label.textContent = 'Model not loaded';
            btnLoad.disabled = false;
            pipelineReady = false;
            btnGen.disabled = true;
        }
    } catch (e) {
        // Server not reachable yet
    }
}

// ── Model Loading ─────────────────────────────────────────────

async function loadModel() {
    const variant = document.getElementById('gguf-variant').value;
    const btn = document.getElementById('btn-load-model');
    btn.disabled = true;
    btn.textContent = '⏳ Starting...';

    try {
        await fetch('/api/load_pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gguf_variant: variant }),
        });
    } catch (e) {
        btn.disabled = false;
        btn.textContent = '⚠ Failed';
    }
}

// ── Image Upload ──────────────────────────────────────────────

function setupDropZone() {
    const zone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    zone.addEventListener('click', () => fileInput.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) uploadFile(fileInput.files[0]);
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('image', file);

    const zone = document.getElementById('drop-zone');
    zone.innerHTML = '<div class="drop-zone-icon">⏳</div><div class="drop-zone-text">Uploading...</div>';

    try {
        const r = await fetch('/api/upload', { method: 'POST', body: formData });
        const d = await r.json();

        if (d.error) {
            zone.innerHTML = `<div class="drop-zone-icon">❌</div><div class="drop-zone-text">${d.error}</div>`;
            return;
        }

        uploadedImageId = d.id;
        uploadedImageUrl = d.url;

        // Show preview
        const preview = document.getElementById('input-preview');
        preview.src = d.url;
        preview.style.display = 'block';
        zone.style.display = 'none';

        // Update 3D viewport
        setViewportImage(d.url);

        // Enable generate button if pipeline ready
        document.getElementById('btn-generate').disabled = !pipelineReady;

    } catch (e) {
        zone.innerHTML = `<div class="drop-zone-icon">❌</div><div class="drop-zone-text">Upload failed</div>`;
    }
}

// ── Dropdown ↔ Viewport Sync ──────────────────────────────────

function setupDropdownSync() {
    ['sel-azimuth', 'sel-elevation', 'sel-distance'].forEach(id => {
        document.getElementById(id).addEventListener('change', syncFromDropdowns);
    });
}

// ── Generation ────────────────────────────────────────────────

async function generate() {
    if (!uploadedImageId || !pipelineReady) return;

    const btn = document.getElementById('btn-generate');
    const overlay = document.getElementById('generating-overlay');
    const outputPreview = document.getElementById('output-preview');
    const placeholder = document.getElementById('output-placeholder');
    const resultInfo = document.getElementById('result-info');

    btn.disabled = true;
    btn.textContent = '⏳ Generating...';
    overlay.style.display = 'flex';

    const payload = {
        image_id: uploadedImageId,
        azimuth: document.getElementById('sel-azimuth').value,
        elevation: document.getElementById('sel-elevation').value,
        distance: document.getElementById('sel-distance').value,
        seed: parseInt(document.getElementById('rng-seed').value),
        randomize_seed: document.getElementById('chk-randomize').checked,
        guidance_scale: parseFloat(document.getElementById('rng-cfg').value),
        inference_steps: parseInt(document.getElementById('rng-steps').value),
        lora_scale: parseFloat(document.getElementById('rng-lora').value),
    };

    try {
        const r = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const d = await r.json();

        if (d.error) {
            alert('Generation error: ' + d.error);
            return;
        }

        // Show output
        outputPreview.src = d.url;
        outputPreview.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';

        // Update seed display
        document.getElementById('rng-seed').value = d.seed;
        document.getElementById('val-seed').textContent = d.seed;

        // Show result info
        resultInfo.style.display = 'block';
        resultInfo.textContent = `${d.width}×${d.height} · ${d.elapsed}s · seed: ${d.seed} · ${d.prompt}`;

    } catch (e) {
        alert('Request failed: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Generate';
        overlay.style.display = 'none';
    }
}

// ── Advanced Settings Toggle ──────────────────────────────────

function toggleAdvanced() {
    const header = document.getElementById('advanced-header');
    const body = document.getElementById('advanced-body');
    header.classList.toggle('open');
    body.classList.toggle('open');
}
