// ============================================================
// TRELLIS.2 Studio — 3D GLB Viewer (viewer3d.js)
// GPU-accelerated viewer with interaction-aware quality:
//   • While orbiting/zooming: fast unlit MeshBasicMaterial
//   • When idle: full PBR shaded with lights
// Also extracts UV texture maps for 2D preview.
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── State ──
let renderer, scene, camera, controls, clock;
let currentModel = null;
let originalMaterials = new Map();   // mesh → original PBR material
let unlitMaterials = new Map();      // mesh → MeshBasicMaterial clone
let extractedTextures = [];          // { name, image, width, height }
let isInteracting = false;
let idleTimer = null;
let forcedWireframe = false;
let frameCount = 0, fpsTime = 0, currentFps = 0;
let animFrameId = null;
let isActive = false;

const IDLE_DELAY = 180; // ms after interaction stops → switch to shaded

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
// INIT THREE.JS
// ══════════════════════════════════════════════════════════════

function init() {
    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.shadowMap.enabled = false;
    renderer.setClearColor(0x09090B, 1);

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090B);

    // Camera
    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
    camera.position.set(2, 1.5, 3);

    // Controls
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.2;
    controls.screenSpacePanning = true;

    // Track interaction for LOD switching
    controls.addEventListener('start', onInteractionStart);
    controls.addEventListener('end', onInteractionEnd);

    // Lights (used in shaded mode)
    setupLights();

    // Clock for FPS
    clock = new THREE.Clock();

    // Resize observer
    const ro = new ResizeObserver(onResize);
    ro.observe(container);

    // Buttons
    btnResetCam.addEventListener('click', resetCamera);
    btnWireframe.addEventListener('click', toggleWireframe);
    btnTexMap.addEventListener('click', openTextureModal);
    texCloseBtn.addEventListener('click', closeTextureModal);
    texOverlay.addEventListener('click', (e) => {
        if (e.target === texOverlay) closeTextureModal();
    });
}

function setupLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    ambient.name = '_viewer_ambient';
    scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(3, 5, 4);
    key.name = '_viewer_key';
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x8899bb, 0.6);
    fill.position.set(-3, 2, -2);
    fill.name = '_viewer_fill';
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0xE8A917, 0.3);
    rim.position.set(0, -1, -4);
    rim.name = '_viewer_rim';
    scene.add(rim);
}

// ══════════════════════════════════════════════════════════════
// INTERACTION-AWARE QUALITY SWITCHING
// ══════════════════════════════════════════════════════════════

function onInteractionStart() {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (!isInteracting) {
        isInteracting = true;
        if (!forcedWireframe) switchToFastMode();
    }
}

function onInteractionEnd() {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
        isInteracting = false;
        if (!forcedWireframe) switchToShadedMode();
    }, IDLE_DELAY);
}

function switchToFastMode() {
    if (!currentModel) return;
    currentModel.traverse((obj) => {
        if (!obj.isMesh) return;
        const unlit = unlitMaterials.get(obj);
        if (unlit) obj.material = unlit;
    });
    hudMode.textContent = 'FAST';
    hudMode.classList.add('fast');
    hudMode.classList.remove('shaded', 'wire');
}

function switchToShadedMode() {
    if (!currentModel) return;
    currentModel.traverse((obj) => {
        if (!obj.isMesh) return;
        const orig = originalMaterials.get(obj);
        if (orig) {
            obj.material = orig;
            if (Array.isArray(orig)) orig.forEach(m => { m.wireframe = false; });
            else orig.wireframe = false;
        }
    });
    hudMode.textContent = 'SHADED';
    hudMode.classList.add('shaded');
    hudMode.classList.remove('fast', 'wire');
}

function toggleWireframe() {
    forcedWireframe = !forcedWireframe;
    btnWireframe.classList.toggle('active', forcedWireframe);
    if (!currentModel) return;

    if (forcedWireframe) {
        currentModel.traverse((obj) => {
            if (!obj.isMesh) return;
            const orig = originalMaterials.get(obj);
            if (orig) {
                obj.material = orig;
                if (Array.isArray(orig)) orig.forEach(m => { m.wireframe = true; });
                else orig.wireframe = true;
            }
        });
        hudMode.textContent = 'WIREFRAME';
        hudMode.classList.add('wire');
        hudMode.classList.remove('fast', 'shaded');
    } else {
        if (isInteracting) switchToFastMode();
        else switchToShadedMode();
    }
}

// ══════════════════════════════════════════════════════════════
// LOAD GLB
// ══════════════════════════════════════════════════════════════

function loadFromUrl(url, name) {
    // Show the viewer
    show();

    // Clear previous
    if (currentModel) {
        scene.remove(currentModel);
        disposeModel(currentModel);
        currentModel = null;
    }
    originalMaterials.clear();
    unlitMaterials.clear();
    extractedTextures = [];

    hudTris.textContent = 'Loading…';
    hudVerts.textContent = '';

    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
        const root = gltf.scene || gltf.scenes?.[0];
        if (!root) {
            hudTris.textContent = 'Error: no scene';
            return;
        }

        let triCount = 0, vertCount = 0, meshCount = 0;

        root.traverse((obj) => {
            if (!obj.isMesh) return;
            meshCount++;

            // Disable auto-update for perf
            obj.frustumCulled = true;
            obj.castShadow = false;
            obj.receiveShadow = false;

            const geo = obj.geometry;
            if (geo) {
                const pos = geo.attributes?.position;
                if (pos) vertCount += pos.count;
                triCount += geo.index
                    ? geo.index.count / 3
                    : (pos?.count || 0) / 3;
            }

            // Store original material
            const origMat = obj.material;
            originalMaterials.set(obj, origMat);

            // Create unlit clone for fast mode
            const mats = Array.isArray(origMat) ? origMat : [origMat];
            const unlitArr = mats.map(m => {
                const basic = new THREE.MeshBasicMaterial({
                    map: m?.map ?? null,
                    color: m?.color?.clone?.() ?? new THREE.Color(0xcccccc),
                    transparent: !!m?.transparent,
                    opacity: m?.opacity ?? 1,
                    side: m?.side ?? THREE.FrontSide,
                    vertexColors: !!m?.vertexColors,
                });
                if (basic.map) {
                    basic.map.colorSpace = THREE.SRGBColorSpace;
                }
                return basic;
            });
            unlitMaterials.set(obj, Array.isArray(origMat) ? unlitArr : unlitArr[0]);

            // Extract textures
            mats.forEach((m, mi) => {
                if (m?.map?.image) {
                    const existing = extractedTextures.find(t => t.image === m.map.image);
                    if (!existing) {
                        extractedTextures.push({
                            name: m.name || `material_${meshCount}_${mi}`,
                            image: m.map.image,
                            width: m.map.image.width || m.map.image.naturalWidth || 0,
                            height: m.map.image.height || m.map.image.naturalHeight || 0,
                        });
                    }
                }
            });
        });

        scene.add(root);
        currentModel = root;

        hudTris.textContent = `▲ ${formatNum(Math.round(triCount))}`;
        hudVerts.textContent = `⬡ ${formatNum(vertCount)}`;
        btnTexMap.style.display = extractedTextures.length > 0 ? '' : 'none';

        frameCamera(root);
        switchToShadedMode();

    }, undefined, (err) => {
        console.error('GLB load error:', err);
        hudTris.textContent = 'Load failed';
    });
}

function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

function frameCamera(object) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = maxDim / (2 * Math.tan(fov / 2));

    camera.position.copy(center).add(new THREE.Vector3(dist * 0.8, dist * 0.5, dist * 1.1));
    camera.near = Math.max(dist / 100, 0.001);
    camera.far = dist * 200;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.minDistance = dist * 0.05;
    controls.maxDistance = dist * 10;
    controls.update();
}

function resetCamera() {
    if (currentModel) frameCamera(currentModel);
}

function disposeModel(root) {
    root?.traverse((obj) => {
        if (obj.isMesh) {
            obj.geometry?.dispose?.();
            // Don't dispose original materials, just unlit clones
        }
    });
    // Dispose unlit materials
    for (const mat of unlitMaterials.values()) {
        if (Array.isArray(mat)) mat.forEach(m => m?.dispose?.());
        else mat?.dispose?.();
    }
}

// ══════════════════════════════════════════════════════════════
// TEXTURE MAP MODAL
// ══════════════════════════════════════════════════════════════

function openTextureModal() {
    if (extractedTextures.length === 0) return;

    texOverlay.classList.add('open');

    // Build selector if multiple textures
    const body = texModalBody;
    // Remove old selector if any
    const oldSel = body.querySelector('.tex-selector');
    if (oldSel) oldSel.remove();

    if (extractedTextures.length > 1) {
        const sel = document.createElement('div');
        sel.className = 'tex-selector';
        extractedTextures.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'tex-sel-btn' + (i === 0 ? ' active' : '');
            btn.textContent = t.name || `Texture ${i + 1}`;
            btn.onclick = () => {
                sel.querySelectorAll('.tex-sel-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderTexturePreview(i);
            };
            sel.appendChild(btn);
        });
        body.insertBefore(sel, body.firstChild);
    }

    renderTexturePreview(0);
}

function renderTexturePreview(idx) {
    const tex = extractedTextures[idx];
    if (!tex) return;

    const img = tex.image;
    const w = img.width || img.naturalWidth || 512;
    const h = img.height || img.naturalHeight || 512;

    texPreviewCanvas.width = w;
    texPreviewCanvas.height = h;
    const ctx = texPreviewCanvas.getContext('2d');

    // Draw checkerboard background
    const checkSize = Math.max(8, Math.round(w / 64));
    for (let y = 0; y < h; y += checkSize) {
        for (let x = 0; x < w; x += checkSize) {
            ctx.fillStyle = ((x / checkSize + y / checkSize) % 2 === 0) ? '#1a1a1e' : '#222226';
            ctx.fillRect(x, y, checkSize, checkSize);
        }
    }

    // Draw the texture
    ctx.drawImage(img, 0, 0, w, h);

    texInfo.textContent = `${tex.name || 'Texture'} — ${w}×${h}px`;

    // Update download link
    texDownloadBtn.href = texPreviewCanvas.toDataURL('image/png');
    texDownloadBtn.download = `${tex.name || 'texture_map'}.png`;
}

function closeTextureModal() {
    texOverlay.classList.remove('open');
}

// ══════════════════════════════════════════════════════════════
// RENDER LOOP
// ══════════════════════════════════════════════════════════════

function animate() {
    animFrameId = requestAnimationFrame(animate);
    if (!isActive) return;

    controls.update();
    renderer.render(scene, camera);

    // FPS counter
    frameCount++;
    const elapsed = clock.getElapsedTime();
    if (elapsed - fpsTime >= 0.5) {
        currentFps = Math.round(frameCount / (elapsed - fpsTime));
        hudFps.textContent = currentFps + ' fps';
        frameCount = 0;
        fpsTime = elapsed;
    }
}

// ══════════════════════════════════════════════════════════════
// SHOW / HIDE
// ══════════════════════════════════════════════════════════════

function show() {
    isActive = true;
    container.classList.add('active');

    // Hide other canvas states
    const empty = document.getElementById('canvasEmpty');
    const media = document.getElementById('canvasMedia');
    const noRender = document.getElementById('canvasNoRender');
    const progress = document.getElementById('canvasProgress');
    if (empty) empty.style.display = 'none';
    if (media) media.classList.remove('active');
    if (noRender) noRender.classList.remove('active');

    onResize();

    if (!animFrameId) animate();
}

function hide() {
    isActive = false;
    container.classList.remove('active');
}

function onResize() {
    if (!renderer || !container) return;
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

// ══════════════════════════════════════════════════════════════
// EXPORT API
// ══════════════════════════════════════════════════════════════

init();
animate();

window.viewer3d = {
    loadFromUrl,
    show,
    hide,
    resetCamera,
};
