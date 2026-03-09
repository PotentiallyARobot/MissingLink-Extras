// ============================================================
// TRELLIS.2 Studio — 3D GLB Viewer (viewer3d.js)
// GPU-accelerated viewer with:
//   • Interaction-aware quality (fast unlit while orbiting)
//   • 3D paint mask via overlay mesh (no shader recompile)
//   • Keyboard + button camera controls (WASD, arrows, +/-)
//   • UV texture map extraction
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── State ──
let renderer, scene, camera, controls, clock;
let currentModel = null;
let originalMaterials = new Map();
let unlitMaterials = new Map();
let extractedTextures = [];
let isInteracting = false;
let idleTimer = null;
let forcedWireframe = false;
let frameCount = 0, fpsTime = 0, currentFps = 0;
let animFrameId = null;
let isActive = false;

// ── Paint mask state ──
let paintMode = false;
let isPainting = false;
let paintErasing = false;
let brushSize = 24;
let maskCanvas = null;
let maskCtx = null;
const MASK_RESOLUTION = 1024;
let maskTexture = null;
let maskOverlayMesh = null;
let raycaster = new THREE.Raycaster();
let paintPointer = new THREE.Vector2();
let brushCursor = null;
let lastPaintUV = null;
let paintCameraLocked = true;  // when true, orbit is disabled for precise painting
let paintThrottleId = null;    // requestAnimationFrame id for throttled paint

// ── Keyboard ──
const keysDown = new Set();
const PAN_SPEED = 0.015;
const ZOOM_KEY_SPEED = 0.08;

const IDLE_DELAY = 180;

// ── DOM refs ──
const container = document.getElementById('canvas3dViewer');
const canvas = document.getElementById('glbCanvas');
const hudTris = document.getElementById('hudTris');
const hudVerts = document.getElementById('hudVerts');
const hudFps = document.getElementById('hudFps');
const hudMode = document.getElementById('hudMode');
const btnResetCam = document.getElementById('btnResetCam');
const btnWireframe = document.getElementById('btnWireframe');
const btnTexMap = document.getElementById('btnTexMap');
const texOverlay = document.getElementById('texModalOverlay');
const texCloseBtn = document.getElementById('texCloseBtn');
const texDownloadBtn = document.getElementById('texDownloadBtn');
const texPreviewCanvas = document.getElementById('texPreviewCanvas');
const texInfo = document.getElementById('texInfo');
const texModalBody = document.getElementById('texModalBody');

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════

function init() {
    renderer = new THREE.WebGLRenderer({
        canvas, antialias: true, alpha: false,
        powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.shadowMap.enabled = false;
    renderer.setClearColor(0x09090B, 1);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090B);

    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
    camera.position.set(2, 1.5, 3);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.2;
    controls.screenSpacePanning = true;

    controls.addEventListener('start', onInteractionStart);
    controls.addEventListener('end', onInteractionEnd);

    setupLights();
    clock = new THREE.Clock();

    const ro = new ResizeObserver(onResize);
    ro.observe(container);

    btnResetCam.addEventListener('click', resetCamera);
    btnWireframe.addEventListener('click', toggleWireframe);
    btnTexMap.addEventListener('click', openTextureModal);
    texCloseBtn.addEventListener('click', closeTextureModal);
    texOverlay.addEventListener('click', (e) => {
        if (e.target === texOverlay) closeTextureModal();
    });

    canvas.addEventListener('pointerdown', onPaintDown);
    canvas.addEventListener('pointermove', onPaintMove);
    canvas.addEventListener('pointerup', onPaintUp);
    canvas.addEventListener('pointerleave', onPaintUp);

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    brushCursor = document.createElement('div');
    brushCursor.className = 'brush-cursor';
    brushCursor.style.display = 'none';
    container.appendChild(brushCursor);

    initMaskCanvas();
}

function setupLights() {
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(3, 5, 4); scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899bb, 0.6);
    fill.position.set(-3, 2, -2); scene.add(fill);
    const rim = new THREE.DirectionalLight(0xE8A917, 0.3);
    rim.position.set(0, -1, -4); scene.add(rim);
}

// ══════════════════════════════════════════════════════════════
// MASK CANVAS
// ══════════════════════════════════════════════════════════════

function initMaskCanvas() {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = MASK_RESOLUTION;
    maskCanvas.height = MASK_RESOLUTION;
    maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    clearMask();
}

function clearMask() {
    if (!maskCtx) return;
    maskCtx.clearRect(0, 0, MASK_RESOLUTION, MASK_RESOLUTION);
    if (maskTexture) maskTexture.needsUpdate = true;
    lastPaintUV = null;
    window.dispatchEvent(new CustomEvent('maskChanged', { detail: { hasContent: false } }));
}

function getMaskDataURL() { return maskCanvas ? maskCanvas.toDataURL('image/png') : null; }

function getMaskBlob() {
    return new Promise((resolve) => {
        if (!maskCanvas) { resolve(null); return; }
        maskCanvas.toBlob(resolve, 'image/png');
    });
}

function loadMaskFromImage(img) {
    if (!maskCtx) return;
    maskCtx.clearRect(0, 0, MASK_RESOLUTION, MASK_RESOLUTION);
    maskCtx.drawImage(img, 0, 0, MASK_RESOLUTION, MASK_RESOLUTION);
    if (maskTexture) maskTexture.needsUpdate = true;
    window.dispatchEvent(new CustomEvent('maskChanged', { detail: { hasContent: true } }));
}

function hasMaskContent() {
    if (!maskCtx) return false;
    const data = maskCtx.getImageData(0, 0, MASK_RESOLUTION, MASK_RESOLUTION).data;
    for (let i = 3; i < data.length; i += 16) {
        if (data[i] > 10) return true;
    }
    return false;
}

// ══════════════════════════════════════════════════════════════
// PAINT MODE — overlay mesh (fast, no shader recompile)
// ══════════════════════════════════════════════════════════════

function enterPaintMode() {
    if (!currentModel) return;
    paintMode = true;
    paintCameraLocked = true;
    controls.enabled = false;

    // Configure raycaster — ONLY hit front faces to prevent painting wrong side
    raycaster.firstHitOnly = true;

    // Use unlit for speed
    currentModel.traverse((obj) => {
        if (!obj.isMesh) return;
        const unlit = unlitMaterials.get(obj);
        if (unlit) obj.material = unlit;
    });

    if (!maskTexture) {
        maskTexture = new THREE.CanvasTexture(maskCanvas);
        maskTexture.colorSpace = THREE.SRGBColorSpace;
        maskTexture.minFilter = THREE.LinearFilter;
        maskTexture.magFilter = THREE.LinearFilter;
    }

    // Build overlay group
    removeMaskOverlay();
    const group = new THREE.Group();
    group.name = '_maskOverlay';

    currentModel.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry.attributes.uv) return;
        const mat = new THREE.MeshBasicMaterial({
            map: maskTexture,
            transparent: true,
            depthWrite: false,
            depthTest: true,
            side: THREE.FrontSide,
            opacity: 0.6,
        });
        const clone = new THREE.Mesh(obj.geometry, mat);
        clone.matrixAutoUpdate = false;
        obj.updateWorldMatrix(true, false);
        clone.matrix.copy(obj.matrixWorld);
        clone.renderOrder = 10;
        group.add(clone);
    });

    scene.add(group);
    maskOverlayMesh = group;

    brushCursor.style.display = 'block';
    updateBrushCursor();
    canvas.style.cursor = 'none';

    hudMode.textContent = 'PAINT 🔒';
    hudMode.classList.remove('shaded', 'fast', 'wire');
    hudMode.classList.add('paint');

    // Notify UI
    window.dispatchEvent(new CustomEvent('paintModeChanged', { detail: { active: true, cameraLocked: true } }));
}

function removeMaskOverlay() {
    if (maskOverlayMesh) {
        maskOverlayMesh.traverse((obj) => { if (obj.isMesh) obj.material?.dispose?.(); });
        scene.remove(maskOverlayMesh);
        maskOverlayMesh = null;
    }
}

function exitPaintMode() {
    paintMode = false;
    isPainting = false;
    paintCameraLocked = true;
    controls.enabled = true;
    removeMaskOverlay();

    currentModel?.traverse((obj) => {
        if (!obj.isMesh) return;
        const orig = originalMaterials.get(obj);
        if (orig) obj.material = orig;
    });

    brushCursor.style.display = 'none';
    canvas.style.cursor = '';
    if (!forcedWireframe) switchToShadedMode();

    window.dispatchEvent(new CustomEvent('paintModeChanged', { detail: { active: false, cameraLocked: false } }));
}

function togglePaintCameraLock() {
    if (!paintMode) return false;
    paintCameraLocked = !paintCameraLocked;
    controls.enabled = !paintCameraLocked;

    if (paintCameraLocked) {
        canvas.style.cursor = 'none';
        brushCursor.style.display = 'block';
        hudMode.textContent = 'PAINT 🔒';
    } else {
        canvas.style.cursor = '';
        brushCursor.style.display = 'none';
        hudMode.textContent = 'PAINT 🔓';
    }

    window.dispatchEvent(new CustomEvent('paintModeChanged', {
        detail: { active: true, cameraLocked: paintCameraLocked }
    }));
    return paintCameraLocked;
}

function togglePaintMode() {
    if (paintMode) exitPaintMode(); else enterPaintMode();
    return paintMode;
}

function setBrushSize(size) { brushSize = Math.max(4, Math.min(120, size)); updateBrushCursor(); }
function setBrushErasing(erasing) { paintErasing = erasing; updateBrushCursor(); }

function updateBrushCursor() {
    if (!brushCursor) return;
    const d = brushSize * 2;
    brushCursor.style.width = d + 'px';
    brushCursor.style.height = d + 'px';
    brushCursor.style.borderColor = paintErasing ? 'rgba(239,68,68,.8)' : 'rgba(232,169,23,.9)';
    brushCursor.style.background = paintErasing ? 'rgba(239,68,68,.06)' : 'rgba(232,169,23,.06)';
}

// ── Paint Events ──

function onPaintDown(e) {
    if (!paintMode || !paintCameraLocked) return;
    e.preventDefault();
    isPainting = true;
    lastPaintUV = null;
    paintAtScreen(e.clientX, e.clientY);
}

function onPaintMove(e) {
    if (!paintMode) return;
    const rect = canvas.getBoundingClientRect();
    if (paintCameraLocked) {
        brushCursor.style.left = (e.clientX - rect.left - brushSize) + 'px';
        brushCursor.style.top = (e.clientY - rect.top - brushSize) + 'px';
    }
    if (isPainting && paintCameraLocked) {
        // Throttle to rAF for smooth performance
        const cx = e.clientX, cy = e.clientY;
        if (paintThrottleId) cancelAnimationFrame(paintThrottleId);
        paintThrottleId = requestAnimationFrame(() => {
            paintAtScreen(cx, cy);
            paintThrottleId = null;
        });
    }
}

function onPaintUp() {
    if (!paintMode) return;
    isPainting = false;
    lastPaintUV = null;
    if (paintThrottleId) { cancelAnimationFrame(paintThrottleId); paintThrottleId = null; }
    window.dispatchEvent(new CustomEvent('maskChanged', { detail: { hasContent: hasMaskContent() } }));
}

function paintAtScreen(clientX, clientY) {
    if (!currentModel || !maskCtx) return;

    const rect = canvas.getBoundingClientRect();
    paintPointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    paintPointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(paintPointer, camera);

    // ONLY raycast original model meshes, not overlay
    const meshes = [];
    currentModel.traverse((obj) => {
        if (obj.isMesh && obj.geometry.attributes.uv) meshes.push(obj);
    });

    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length === 0) return;

    // Filter: only accept hits on faces facing the camera (dot product check)
    const camDir = new THREE.Vector3();
    camera.getWorldDirection(camDir);
    let hit = null;
    for (const h of hits) {
        if (h.face && h.face.normal) {
            // Transform face normal to world space
            const worldNormal = h.face.normal.clone();
            const normalMatrix = new THREE.Matrix3().getNormalMatrix(h.object.matrixWorld);
            worldNormal.applyMatrix3(normalMatrix).normalize();
            // Face must be pointing toward camera (dot < 0 means facing us)
            if (worldNormal.dot(camDir) < 0.15) {
                hit = h;
                break;
            }
        } else {
            hit = h;
            break;
        }
    }
    if (!hit) return;

    const uv = hit.uv;
    if (!uv) return;

    const mx = uv.x * MASK_RESOLUTION;
    const my = (1 - uv.y) * MASK_RESOLUTION;

    // Fixed radius in UV-pixel space: brushSize slider maps directly
    const r = Math.max(3, brushSize * 0.5);

    // Interpolate for continuous strokes
    const points = [];
    if (lastPaintUV) {
        const dx = mx - lastPaintUV.x;
        const dy = my - lastPaintUV.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const step = Math.max(1, r * 0.35);
        const n = Math.ceil(dist / step);
        for (let i = 0; i <= n; i++) {
            const t = n > 0 ? i / n : 1;
            points.push({ x: lastPaintUV.x + dx * t, y: lastPaintUV.y + dy * t });
        }
    } else {
        points.push({ x: mx, y: my });
    }
    lastPaintUV = { x: mx, y: my };

    // Hard-edge brush for accuracy (no soft falloff = no bleed to unwanted areas)
    for (const pt of points) {
        if (paintErasing) {
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.globalCompositeOperation = 'source-over';
        } else {
            maskCtx.fillStyle = 'rgba(255, 140, 0, 0.92)';
            maskCtx.beginPath();
            maskCtx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            maskCtx.fill();
        }
    }

    if (maskTexture) maskTexture.needsUpdate = true;
}

// ══════════════════════════════════════════════════════════════
// KEYBOARD + BUTTON CAMERA
// ══════════════════════════════════════════════════════════════

function onKeyDown(e) {
    if (!isActive) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    keysDown.add(e.key.toLowerCase());
}
function onKeyUp(e) { keysDown.delete(e.key.toLowerCase()); }

function processKeyboardInput() {
    if (keysDown.size === 0 || !controls) return;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    right.crossVectors(forward, up).normalize();
    const pan = new THREE.Vector3();

    if (keysDown.has('a') || keysDown.has('arrowleft'))  pan.addScaledVector(right, -PAN_SPEED);
    if (keysDown.has('d') || keysDown.has('arrowright')) pan.addScaledVector(right, PAN_SPEED);
    if (keysDown.has('w') || keysDown.has('arrowup'))   pan.y += PAN_SPEED;
    if (keysDown.has('s') || keysDown.has('arrowdown')) pan.y -= PAN_SPEED;
    if (keysDown.has('e') || keysDown.has('=') || keysDown.has('+')) camera.position.addScaledVector(forward, ZOOM_KEY_SPEED);
    if (keysDown.has('q') || keysDown.has('-'))                      camera.position.addScaledVector(forward, -ZOOM_KEY_SPEED);

    if (pan.lengthSq() > 0) { camera.position.add(pan); controls.target.add(pan); }
}

function panCamera(dx, dy) {
    if (!controls || !camera) return;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    camera.getWorldDirection(new THREE.Vector3());
    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    right.crossVectors(fwd, up).normalize();
    const d = new THREE.Vector3();
    d.addScaledVector(right, dx * 0.15);
    d.y += dy * 0.15;
    camera.position.add(d);
    controls.target.add(d);
    controls.update();
}

function zoomIn() {
    if (!controls || !camera) return;
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    camera.position.addScaledVector(dir, controls.minDistance * 2 || 0.3);
    controls.update();
}

function zoomOut() {
    if (!controls || !camera) return;
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    camera.position.addScaledVector(dir, -(controls.minDistance * 2 || 0.3));
    controls.update();
}

// ══════════════════════════════════════════════════════════════
// INTERACTION-AWARE QUALITY
// ══════════════════════════════════════════════════════════════

function onInteractionStart() {
    if (paintMode) return;
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (!isInteracting) { isInteracting = true; if (!forcedWireframe) switchToFastMode(); }
}

function onInteractionEnd() {
    if (paintMode) return;
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { isInteracting = false; if (!forcedWireframe) switchToShadedMode(); }, IDLE_DELAY);
}

function switchToFastMode() {
    if (!currentModel || paintMode) return;
    currentModel.traverse((obj) => { if (obj.isMesh) { const u = unlitMaterials.get(obj); if (u) obj.material = u; } });
    if (!paintMode) { hudMode.textContent = 'FAST'; hudMode.className = 'hud-mode fast'; }
}

function switchToShadedMode() {
    if (!currentModel || paintMode) return;
    currentModel.traverse((obj) => {
        if (!obj.isMesh) return;
        const orig = originalMaterials.get(obj);
        if (orig) { obj.material = orig; if (Array.isArray(orig)) orig.forEach(m => { m.wireframe = false; }); else orig.wireframe = false; }
    });
    hudMode.textContent = 'SHADED'; hudMode.className = 'hud-mode shaded';
}

function toggleWireframe() {
    forcedWireframe = !forcedWireframe;
    btnWireframe.classList.toggle('active', forcedWireframe);
    if (!currentModel) return;
    if (forcedWireframe) {
        if (paintMode) exitPaintMode();
        currentModel.traverse((obj) => { if (!obj.isMesh) return; const orig = originalMaterials.get(obj); if (orig) { obj.material = orig; if (Array.isArray(orig)) orig.forEach(m => { m.wireframe = true; }); else orig.wireframe = true; } });
        hudMode.textContent = 'WIREFRAME'; hudMode.className = 'hud-mode wire';
    } else { if (isInteracting) switchToFastMode(); else switchToShadedMode(); }
}

// ══════════════════════════════════════════════════════════════
// LOAD GLB
// ══════════════════════════════════════════════════════════════

function loadFromUrl(url, name) {
    show();
    if (paintMode) exitPaintMode();
    if (currentModel) { scene.remove(currentModel); disposeModel(currentModel); currentModel = null; }
    originalMaterials.clear(); unlitMaterials.clear(); removeMaskOverlay(); extractedTextures = [];
    clearMask();

    hudTris.textContent = 'Loading…'; hudVerts.textContent = '';
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
        const root = gltf.scene || gltf.scenes?.[0];
        if (!root) { hudTris.textContent = 'Error'; return; }
        let triCount = 0, vertCount = 0, meshCount = 0;
        root.traverse((obj) => {
            if (!obj.isMesh) return;
            meshCount++;
            obj.frustumCulled = true; obj.castShadow = false; obj.receiveShadow = false;
            const geo = obj.geometry;
            if (geo) { const pos = geo.attributes?.position; if (pos) vertCount += pos.count; triCount += geo.index ? geo.index.count / 3 : (pos?.count || 0) / 3; }
            const origMat = obj.material;
            originalMaterials.set(obj, origMat);
            const mats = Array.isArray(origMat) ? origMat : [origMat];
            const unlitArr = mats.map(m => { const b = new THREE.MeshBasicMaterial({ map: m?.map ?? null, color: m?.color?.clone?.() ?? new THREE.Color(0xcccccc), transparent: !!m?.transparent, opacity: m?.opacity ?? 1, side: m?.side ?? THREE.FrontSide, vertexColors: !!m?.vertexColors }); if (b.map) b.map.colorSpace = THREE.SRGBColorSpace; return b; });
            unlitMaterials.set(obj, Array.isArray(origMat) ? unlitArr : unlitArr[0]);
            mats.forEach((m, mi) => { if (m?.map?.image) { if (!extractedTextures.find(t => t.image === m.map.image)) extractedTextures.push({ name: m.name || `mat_${meshCount}_${mi}`, image: m.map.image, width: m.map.image.width || m.map.image.naturalWidth || 0, height: m.map.image.height || m.map.image.naturalHeight || 0 }); } });
        });
        scene.add(root); currentModel = root;
        hudTris.textContent = `▲ ${formatNum(Math.round(triCount))}`;
        hudVerts.textContent = `⬡ ${formatNum(vertCount)}`;
        btnTexMap.style.display = extractedTextures.length > 0 ? '' : 'none';
        frameCamera(root);
        switchToShadedMode();
    }, undefined, (err) => { console.error('GLB load error:', err); hudTris.textContent = 'Load failed'; });
}

function formatNum(n) { if (n >= 1e6) return (n/1e6).toFixed(1)+'M'; if (n >= 1e3) return (n/1e3).toFixed(1)+'K'; return String(n); }

function frameCamera(object) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = maxDim / (2 * Math.tan(fov / 2));
    camera.position.copy(center).add(new THREE.Vector3(dist * 0.8, dist * 0.5, dist * 1.1));
    camera.near = Math.max(dist / 100, 0.001); camera.far = dist * 200;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.minDistance = dist * 0.05; controls.maxDistance = dist * 10;
    controls.update();
}

function resetCamera() { if (currentModel) frameCamera(currentModel); }

function disposeModel(root) {
    root?.traverse((obj) => { if (obj.isMesh) obj.geometry?.dispose?.(); });
    for (const mat of unlitMaterials.values()) { if (Array.isArray(mat)) mat.forEach(m => m?.dispose?.()); else mat?.dispose?.(); }
}

// ══════════════════════════════════════════════════════════════
// TEXTURE MAP MODAL
// ══════════════════════════════════════════════════════════════

function openTextureModal() {
    if (extractedTextures.length === 0) return;
    texOverlay.classList.add('open');
    const body = texModalBody;
    const oldSel = body.querySelector('.tex-selector');
    if (oldSel) oldSel.remove();
    if (extractedTextures.length > 1) {
        const sel = document.createElement('div'); sel.className = 'tex-selector';
        extractedTextures.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'tex-sel-btn' + (i === 0 ? ' active' : '');
            btn.textContent = t.name || `Texture ${i + 1}`;
            btn.onclick = () => { sel.querySelectorAll('.tex-sel-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active'); renderTexturePreview(i); };
            sel.appendChild(btn);
        });
        body.insertBefore(sel, body.firstChild);
    }
    renderTexturePreview(0);
}

function renderTexturePreview(idx) {
    const tex = extractedTextures[idx]; if (!tex) return;
    const img = tex.image;
    const w = img.width || img.naturalWidth || 512, h = img.height || img.naturalHeight || 512;
    texPreviewCanvas.width = w; texPreviewCanvas.height = h;
    const ctx = texPreviewCanvas.getContext('2d');
    const cs = Math.max(8, Math.round(w / 64));
    for (let y = 0; y < h; y += cs) for (let x = 0; x < w; x += cs) { ctx.fillStyle = ((x/cs+y/cs)%2===0)?'#1a1a1e':'#222226'; ctx.fillRect(x,y,cs,cs); }
    ctx.drawImage(img, 0, 0, w, h);
    texInfo.textContent = `${tex.name||'Texture'} — ${w}×${h}px`;
    texDownloadBtn.href = texPreviewCanvas.toDataURL('image/png');
    texDownloadBtn.download = `${tex.name||'texture_map'}.png`;
}

function closeTextureModal() { texOverlay.classList.remove('open'); }

// ══════════════════════════════════════════════════════════════
// RENDER LOOP
// ══════════════════════════════════════════════════════════════

function animate() {
    animFrameId = requestAnimationFrame(animate);
    if (!isActive) return;
    processKeyboardInput();
    controls.update();
    renderer.render(scene, camera);
    frameCount++;
    const elapsed = clock.getElapsedTime();
    if (elapsed - fpsTime >= 0.5) { currentFps = Math.round(frameCount / (elapsed - fpsTime)); hudFps.textContent = currentFps + ' fps'; frameCount = 0; fpsTime = elapsed; }
}

// ══════════════════════════════════════════════════════════════
// SHOW / HIDE
// ══════════════════════════════════════════════════════════════

function show() {
    isActive = true; container.classList.add('active');
    const empty = document.getElementById('canvasEmpty');
    const media = document.getElementById('canvasMedia');
    const noRender = document.getElementById('canvasNoRender');
    if (empty) empty.style.display = 'none';
    if (media) media.classList.remove('active');
    if (noRender) noRender.classList.remove('active');
    onResize();
    if (!animFrameId) animate();
}

function hide() { isActive = false; container.classList.remove('active'); if (paintMode) exitPaintMode(); }

function onResize() {
    if (!renderer || !container) return;
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
}

// ══════════════════════════════════════════════════════════════
// EXPORT API
// ══════════════════════════════════════════════════════════════

init();
animate();

window.viewer3d = {
    loadFromUrl, show, hide, resetCamera,
    togglePaintMode, enterPaintMode, exitPaintMode,
    togglePaintCameraLock,
    setBrushSize, setBrushErasing,
    clearMask, getMaskDataURL, getMaskBlob, hasMaskContent, loadMaskFromImage,
    get isPaintMode() { return paintMode; },
    get isPaintCameraLocked() { return paintCameraLocked; },
    get brushSize() { return brushSize; },
    zoomIn, zoomOut, panCamera,
};
