// ============================================================
// TRELLIS.2 Studio — 3D GLB Viewer (viewer3d.js)
// GPU-accelerated viewer with:
//   • Interaction-aware quality (fast unlit while orbiting)
//   • 3D paint mask system for region selection
//   • UV texture map extraction
//   • Zoom +/- controls for laptop users
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

// ── Paint mask state ──
let paintMode = false;          // is paint mode active
let isPainting = false;         // mouse is down and painting
let paintErasing = false;       // eraser mode
let brushSize = 24;             // brush radius in pixels
let maskCanvas = null;          // offscreen canvas for UV mask
let maskCtx = null;
const MASK_RESOLUTION = 1024;   // mask texture resolution
let maskTexture = null;         // THREE.CanvasTexture for overlay
let maskOverlayMaterials = new Map(); // mesh → mask overlay material
let raycaster = new THREE.Raycaster();
let paintPointer = new THREE.Vector2();
let brushCursor = null;         // screen-space brush cursor div
let lastPaintUV = null;         // for interpolating strokes

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
// INIT THREE.JS
// ══════════════════════════════════════════════════════════════

function init() {
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

    // Buttons
    btnResetCam.addEventListener('click', resetCamera);
    btnWireframe.addEventListener('click', toggleWireframe);
    btnTexMap.addEventListener('click', openTextureModal);
    texCloseBtn.addEventListener('click', closeTextureModal);
    texOverlay.addEventListener('click', (e) => {
        if (e.target === texOverlay) closeTextureModal();
    });

    // Paint events on the canvas
    canvas.addEventListener('pointerdown', onPaintDown);
    canvas.addEventListener('pointermove', onPaintMove);
    canvas.addEventListener('pointerup', onPaintUp);
    canvas.addEventListener('pointerleave', onPaintUp);

    // Create brush cursor
    brushCursor = document.createElement('div');
    brushCursor.className = 'brush-cursor';
    brushCursor.style.display = 'none';
    container.appendChild(brushCursor);

    // Init mask canvas
    initMaskCanvas();
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
// MASK CANVAS (offscreen UV-space mask)
// ══════════════════════════════════════════════════════════════

function initMaskCanvas() {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = MASK_RESOLUTION;
    maskCanvas.height = MASK_RESOLUTION;
    maskCtx = maskCanvas.getContext('2d');
    clearMask();
}

function clearMask() {
    if (!maskCtx) return;
    maskCtx.clearRect(0, 0, MASK_RESOLUTION, MASK_RESOLUTION);
    if (maskTexture) {
        maskTexture.needsUpdate = true;
    }
    lastPaintUV = null;
    // Dispatch event so UI can update
    window.dispatchEvent(new CustomEvent('maskChanged', { detail: { hasContent: false } }));
}

function getMaskDataURL() {
    return maskCanvas ? maskCanvas.toDataURL('image/png') : null;
}

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
    for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 10) return true;
    }
    return false;
}

// ══════════════════════════════════════════════════════════════
// PAINT MODE
// ══════════════════════════════════════════════════════════════

function enterPaintMode() {
    if (!currentModel) return;
    paintMode = true;
    controls.enabled = false;

    // Force shaded mode for responsiveness during paint
    switchToFastMode();

    // Create mask texture if needed
    if (!maskTexture) {
        maskTexture = new THREE.CanvasTexture(maskCanvas);
        maskTexture.colorSpace = THREE.SRGBColorSpace;
        maskTexture.wrapS = THREE.ClampToEdgeWrapping;
        maskTexture.wrapT = THREE.ClampToEdgeWrapping;
    }

    // Apply mask overlay to all meshes
    currentModel.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry.attributes.uv) return;
        const orig = originalMaterials.get(obj);
        if (!orig) return;

        // Create overlay material: base texture + mask tint
        const baseMat = Array.isArray(orig) ? orig[0] : orig;
        const overlayMat = new THREE.MeshBasicMaterial({
            map: baseMat?.map ?? null,
            color: baseMat?.color?.clone?.() ?? new THREE.Color(0xcccccc),
            transparent: true,
            side: baseMat?.side ?? THREE.FrontSide,
        });

        // Custom shader chunk to tint masked areas
        overlayMat.onBeforeCompile = (shader) => {
            shader.uniforms.maskTex = { value: maskTexture };
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <map_fragment>',
                `#include <map_fragment>
                 // Mask overlay
                 vec4 maskSample = texture2D(maskTex, vMapUv);
                 float maskAlpha = maskSample.a * 0.55;
                 diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.91, 0.42, 0.09), maskAlpha);
                `
            );
            shader.fragmentShader = 'uniform sampler2D maskTex;\n' + shader.fragmentShader;
        };

        maskOverlayMaterials.set(obj, overlayMat);
        obj.material = overlayMat;
    });

    // Show brush cursor
    brushCursor.style.display = 'block';
    updateBrushCursor();
    canvas.style.cursor = 'none';

    hudMode.textContent = 'PAINT';
    hudMode.classList.remove('shaded', 'fast', 'wire');
    hudMode.classList.add('paint');
}

function exitPaintMode() {
    paintMode = false;
    isPainting = false;
    controls.enabled = true;

    // Restore original materials
    currentModel?.traverse((obj) => {
        if (!obj.isMesh) return;
        const orig = originalMaterials.get(obj);
        if (orig) obj.material = orig;
    });

    // Clean up overlay materials
    for (const mat of maskOverlayMaterials.values()) {
        mat?.dispose?.();
    }
    maskOverlayMaterials.clear();

    brushCursor.style.display = 'none';
    canvas.style.cursor = '';

    if (forcedWireframe) {
        // Restore wireframe state
    } else {
        switchToShadedMode();
    }
}

function togglePaintMode() {
    if (paintMode) exitPaintMode();
    else enterPaintMode();
    return paintMode;
}

function setBrushSize(size) {
    brushSize = Math.max(4, Math.min(120, size));
    updateBrushCursor();
}

function setBrushErasing(erasing) {
    paintErasing = erasing;
    updateBrushCursor();
}

function updateBrushCursor() {
    if (!brushCursor) return;
    const d = brushSize * 2;
    brushCursor.style.width = d + 'px';
    brushCursor.style.height = d + 'px';
    brushCursor.style.borderColor = paintErasing
        ? 'rgba(239, 68, 68, 0.8)'
        : 'rgba(232, 169, 23, 0.9)';
    brushCursor.style.background = paintErasing
        ? 'rgba(239, 68, 68, 0.06)'
        : 'rgba(232, 169, 23, 0.06)';
}

// ── Paint Events ──

function onPaintDown(e) {
    if (!paintMode) return;
    isPainting = true;
    lastPaintUV = null;
    paintAtScreen(e.clientX, e.clientY);
}

function onPaintMove(e) {
    if (!paintMode) return;

    // Update cursor position
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    brushCursor.style.left = (cx - brushSize) + 'px';
    brushCursor.style.top = (cy - brushSize) + 'px';

    if (isPainting) {
        paintAtScreen(e.clientX, e.clientY);
    }
}

function onPaintUp(e) {
    if (!paintMode) return;
    isPainting = false;
    lastPaintUV = null;

    // Check if mask has content and notify
    const has = hasMaskContent();
    window.dispatchEvent(new CustomEvent('maskChanged', { detail: { hasContent: has } }));
}

function paintAtScreen(clientX, clientY) {
    if (!currentModel || !maskCtx) return;

    const rect = canvas.getBoundingClientRect();
    paintPointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    paintPointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(paintPointer, camera);

    // Collect meshes
    const meshes = [];
    currentModel.traverse((obj) => {
        if (obj.isMesh && obj.geometry.attributes.uv) meshes.push(obj);
    });

    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length === 0) return;

    const hit = hits[0];
    const uv = hit.uv;
    if (!uv) return;

    // Convert UV to mask canvas coords
    const mx = uv.x * MASK_RESOLUTION;
    const my = (1 - uv.y) * MASK_RESOLUTION; // flip Y for canvas

    // Brush radius in UV space — approximate from screen brush size
    // Use distance to estimate UV-space brush radius
    const uvRadius = (brushSize / Math.min(rect.width, rect.height)) * 1.8;
    const maskRadius = uvRadius * MASK_RESOLUTION;
    const r = Math.max(2, maskRadius);

    // Interpolate between last paint point for smooth strokes
    const points = [];
    if (lastPaintUV) {
        const dx = mx - lastPaintUV.x;
        const dy = my - lastPaintUV.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const step = Math.max(1, r * 0.3);
        const steps = Math.ceil(dist / step);
        for (let i = 0; i <= steps; i++) {
            const t = steps > 0 ? i / steps : 1;
            points.push({ x: lastPaintUV.x + dx * t, y: lastPaintUV.y + dy * t });
        }
    } else {
        points.push({ x: mx, y: my });
    }
    lastPaintUV = { x: mx, y: my };

    // Paint or erase
    for (const pt of points) {
        if (paintErasing) {
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.globalCompositeOperation = 'source-over';
        } else {
            const grad = maskCtx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, r);
            grad.addColorStop(0, 'rgba(255, 140, 0, 0.85)');
            grad.addColorStop(0.6, 'rgba(255, 140, 0, 0.5)');
            grad.addColorStop(1, 'rgba(255, 140, 0, 0)');
            maskCtx.fillStyle = grad;
            maskCtx.beginPath();
            maskCtx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            maskCtx.fill();
        }
    }

    // Update texture
    if (maskTexture) maskTexture.needsUpdate = true;
}

// ══════════════════════════════════════════════════════════════
// ZOOM CONTROLS (for laptop users)
// ══════════════════════════════════════════════════════════════

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
// INTERACTION-AWARE QUALITY SWITCHING
// ══════════════════════════════════════════════════════════════

function onInteractionStart() {
    if (paintMode) return; // don't switch materials during paint
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (!isInteracting) {
        isInteracting = true;
        if (!forcedWireframe && !paintMode) switchToFastMode();
    }
}

function onInteractionEnd() {
    if (paintMode) return;
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
        isInteracting = false;
        if (!forcedWireframe && !paintMode) switchToShadedMode();
    }, IDLE_DELAY);
}

function switchToFastMode() {
    if (!currentModel || paintMode) return;
    currentModel.traverse((obj) => {
        if (!obj.isMesh) return;
        const unlit = unlitMaterials.get(obj);
        if (unlit) obj.material = unlit;
    });
    if (!paintMode) {
        hudMode.textContent = 'FAST';
        hudMode.classList.add('fast');
        hudMode.classList.remove('shaded', 'wire', 'paint');
    }
}

function switchToShadedMode() {
    if (!currentModel || paintMode) return;
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
    hudMode.classList.remove('fast', 'wire', 'paint');
}

function toggleWireframe() {
    forcedWireframe = !forcedWireframe;
    btnWireframe.classList.toggle('active', forcedWireframe);
    if (!currentModel) return;

    if (forcedWireframe) {
        if (paintMode) exitPaintMode();
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
        hudMode.classList.remove('fast', 'shaded', 'paint');
    } else {
        if (isInteracting) switchToFastMode();
        else switchToShadedMode();
    }
}

// ══════════════════════════════════════════════════════════════
// LOAD GLB
// ══════════════════════════════════════════════════════════════

function loadFromUrl(url, name) {
    show();

    // Exit paint mode if active
    if (paintMode) exitPaintMode();

    // Clear previous
    if (currentModel) {
        scene.remove(currentModel);
        disposeModel(currentModel);
        currentModel = null;
    }
    originalMaterials.clear();
    unlitMaterials.clear();
    maskOverlayMaterials.clear();
    extractedTextures = [];

    // Clear mask for new model
    clearMask();

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
        }
    });
    for (const mat of unlitMaterials.values()) {
        if (Array.isArray(mat)) mat.forEach(m => m?.dispose?.());
        else mat?.dispose?.();
    }
    for (const mat of maskOverlayMaterials.values()) {
        mat?.dispose?.();
    }
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

    const checkSize = Math.max(8, Math.round(w / 64));
    for (let y = 0; y < h; y += checkSize) {
        for (let x = 0; x < w; x += checkSize) {
            ctx.fillStyle = ((x / checkSize + y / checkSize) % 2 === 0) ? '#1a1a1e' : '#222226';
            ctx.fillRect(x, y, checkSize, checkSize);
        }
    }

    ctx.drawImage(img, 0, 0, w, h);

    texInfo.textContent = `${tex.name || 'Texture'} — ${w}×${h}px`;

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

    const empty = document.getElementById('canvasEmpty');
    const media = document.getElementById('canvasMedia');
    const noRender = document.getElementById('canvasNoRender');
    if (empty) empty.style.display = 'none';
    if (media) media.classList.remove('active');
    if (noRender) noRender.classList.remove('active');

    onResize();
    if (!animFrameId) animate();
}

function hide() {
    isActive = false;
    container.classList.remove('active');
    if (paintMode) exitPaintMode();
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
    // Paint mask API
    togglePaintMode,
    enterPaintMode,
    exitPaintMode,
    setBrushSize,
    setBrushErasing,
    clearMask,
    getMaskDataURL,
    getMaskBlob,
    hasMaskContent,
    loadMaskFromImage,
    get isPaintMode() { return paintMode; },
    get brushSize() { return brushSize; },
    // Zoom
    zoomIn,
    zoomOut,
};
