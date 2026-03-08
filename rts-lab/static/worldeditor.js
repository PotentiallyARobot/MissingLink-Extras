// ============================================================
// TRELLIS.2 — World Editor Engine (worldeditor.js)
//
// Full 2.5D level editor with:
//  • RTS-style camera (pan, rotate, zoom)
//  • GLB model loading from TRELLIS output or file drop
//  • Select / Move / Rotate / Scale with gizmo handles
//  • Grid snapping
//  • Scene tree, property editing
//  • Shadow baking → screenshot download
//  • Full scene export as merged GLB
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// ══════════════════════════════════════════════════════════════
// GLOBALS
// ══════════════════════════════════════════════════════════════

let renderer, scene, camera, orbitControls, transformControls;
let clock, raycaster, mouse;
let groundPlane, groundGrid, sunLight, ambientLight;
let gltfLoader;

// Scene objects the user placed
const worldObjects = [];   // { id, name, mesh, url?, type }
let selectedObject = null;
let nextId = 1;

// Tools
let activeTool = 'select';  // select | move | rotate | scale
let snapEnabled = true;
let snapSize = 0.5;

// FPS tracking
let frameCount = 0, fpsTime = 0, currentFps = 0;

// Axis indicator
let axisScene, axisCamera, axisRenderer;

// DOM references
const $ = id => document.getElementById(id);
const viewport = $('weViewport');
const canvas = $('weCanvas');

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════

function init() {
    // ── Renderer ──
    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true,
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x1a1a2e, 1);

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);

    // ── Scene ──
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    scene.fog = new THREE.FogExp2(0x1a1a2e, 0.015);

    // ── Camera — RTS style (isometric-ish perspective) ──
    camera = new THREE.PerspectiveCamera(35, 1, 0.1, 500);
    camera.position.set(12, 14, 12);
    camera.lookAt(0, 0, 0);

    // ── Orbit controls (RMB pan, MMB rotate, scroll zoom) ──
    orbitControls = new OrbitControls(camera, canvas);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.1;
    orbitControls.screenSpacePanning = true;

    // RTS-style: right-click to pan, middle to orbit
    orbitControls.mouseButtons = {
        LEFT: null,   // we handle left-click for selection
        MIDDLE: THREE.MOUSE.ROTATE,
        RIGHT: THREE.MOUSE.PAN,
    };
    orbitControls.touches = {
        ONE: THREE.TOUCH.PAN,
        TWO: THREE.TOUCH.DOLLY_ROTATE,
    };

    orbitControls.minPolarAngle = 0.2;
    orbitControls.maxPolarAngle = Math.PI / 2.2;  // don't go below ground
    orbitControls.minDistance = 3;
    orbitControls.maxDistance = 100;
    orbitControls.target.set(0, 0, 0);

    // ── Transform controls (gizmo) ──
    transformControls = new TransformControls(camera, canvas);
    transformControls.setSize(0.7);
    transformControls.addEventListener('dragging-changed', (e) => {
        orbitControls.enabled = !e.value;
    });
    transformControls.addEventListener('objectChange', () => {
        updatePropertiesPanel();
        updateSceneTree();
    });
    scene.add(transformControls);

    // ── Lights ──
    ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    sunLight = new THREE.DirectionalLight(0xfff5e6, 1.8);
    updateSunPosition(45);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(2048, 2048);
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 80;
    sunLight.shadow.camera.left = -30;
    sunLight.shadow.camera.right = 30;
    sunLight.shadow.camera.top = 30;
    sunLight.shadow.camera.bottom = -30;
    sunLight.shadow.bias = -0.001;
    sunLight.shadow.normalBias = 0.02;
    scene.add(sunLight);
    scene.add(sunLight.target);

    // Subtle hemisphere light
    const hemi = new THREE.HemisphereLight(0x8899bb, 0x445566, 0.3);
    scene.add(hemi);

    // ── Ground ──
    createGround();

    // ── Helpers ──
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    gltfLoader = new GLTFLoader();
    clock = new THREE.Clock();

    // ── Axis indicator ──
    initAxisIndicator();

    // ── Events ──
    setupEvents();

    // ── Resize ──
    const ro = new ResizeObserver(onResize);
    ro.observe(viewport);
    onResize();

    // ── Fetch generated models ──
    refreshGeneratedModels();
}

// ══════════════════════════════════════════════════════════════
// GROUND PLANE & GRID
// ══════════════════════════════════════════════════════════════

function createGround() {
    // Solid ground
    const groundGeo = new THREE.PlaneGeometry(200, 200);
    const groundMat = new THREE.ShadowMaterial({ opacity: 0.35 });
    groundPlane = new THREE.Mesh(groundGeo, groundMat);
    groundPlane.rotation.x = -Math.PI / 2;
    groundPlane.receiveShadow = true;
    groundPlane.name = '__ground';
    scene.add(groundPlane);

    // Visible grid
    groundGrid = new THREE.GridHelper(100, 100, 0x333340, 0x222230);
    groundGrid.position.y = 0.001;
    groundGrid.name = '__grid';
    scene.add(groundGrid);
}

function updateSunPosition(angleDeg) {
    const rad = THREE.MathUtils.degToRad(angleDeg);
    const dist = 25;
    sunLight.position.set(
        dist * Math.cos(rad) * 0.7,
        dist * Math.sin(rad),
        dist * Math.cos(rad)
    );
    sunLight.target.position.set(0, 0, 0);
}

// ══════════════════════════════════════════════════════════════
// AXIS INDICATOR
// ══════════════════════════════════════════════════════════════

function initAxisIndicator() {
    axisScene = new THREE.Scene();
    axisCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10);
    axisCamera.position.set(0, 0, 2.5);

    const axisCanvas = $('axisCanvas');
    axisRenderer = new THREE.WebGLRenderer({ canvas: axisCanvas, alpha: true, antialias: true });
    axisRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    axisRenderer.setSize(80, 80, false);

    // Axis lines
    const axes = new THREE.AxesHelper(0.8);
    axisScene.add(axes);
}

function updateAxisIndicator() {
    if (!axisCamera || !camera) return;
    axisCamera.quaternion.copy(camera.quaternion);
    axisRenderer.render(axisScene, axisCamera);
}

// ══════════════════════════════════════════════════════════════
// EVENTS
// ══════════════════════════════════════════════════════════════

function setupEvents() {
    // ── Left click → select / place ──
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointerup', onPointerUp);

    // ── Keyboard shortcuts ──
    window.addEventListener('keydown', onKeyDown);

    // ── Dropzone ──
    const dz = $('weDropzone');
    const fi = $('weFileInput');
    ['dragenter', 'dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('over'); }));
    dz.addEventListener('drop', ev => handleFileDrop(ev.dataTransfer.files));
    fi.addEventListener('change', ev => { handleFileDrop(ev.target.files); fi.value = ''; });

    // ── Also allow dropping on viewport ──
    viewport.addEventListener('dragover', ev => { ev.preventDefault(); ev.dataTransfer.dropEffect = 'copy'; });
    viewport.addEventListener('drop', ev => { ev.preventDefault(); handleFileDrop(ev.dataTransfer.files); });

    // ── Tool buttons ──
    document.querySelectorAll('.we-tool[data-tool]').forEach(btn => {
        btn.addEventListener('click', () => setTool(btn.dataset.tool));
    });

    // ── Duplicate / Delete buttons ──
    $('toolDuplicate').addEventListener('click', duplicateSelected);
    $('toolDelete').addEventListener('click', deleteSelected);

    // ── Grid snap ──
    $('btnSnapGrid').addEventListener('click', () => {
        snapEnabled = !snapEnabled;
        $('btnSnapGrid').classList.toggle('active', snapEnabled);
        updateSnap();
    });
    $('snapSize').addEventListener('change', () => {
        snapSize = parseFloat($('snapSize').value);
        updateSnap();
    });

    // ── Primitives ──
    document.querySelectorAll('.we-prim').forEach(btn => {
        btn.addEventListener('click', () => addPrimitive(btn.dataset.prim));
    });

    // ── Properties input ──
    ['propPosX','propPosY','propPosZ','propRotX','propRotY','propRotZ','propScX','propScY','propScZ'].forEach(id => {
        $(id).addEventListener('input', onPropertyInput);
    });
    $('propScLock').addEventListener('change', () => {});

    // ── Environment ──
    $('envSkyColor').addEventListener('input', e => {
        const c = new THREE.Color(e.target.value);
        scene.background = c;
        scene.fog.color = c;
        renderer.setClearColor(c, 1);
    });
    $('envAmbient').addEventListener('input', e => { ambientLight.intensity = parseFloat(e.target.value); });
    $('envSunAngle').addEventListener('input', e => { updateSunPosition(parseFloat(e.target.value)); });
    $('envGround').addEventListener('change', e => {
        const v = e.target.value;
        groundGrid.visible = v === 'grid';
        groundPlane.visible = v !== 'none';
    });

    // ── Bake ──
    $('btnBake').addEventListener('click', () => $('bakeModal').classList.add('open'));
    $('bakeClose').addEventListener('click', () => $('bakeModal').classList.remove('open'));
    $('bakeCancel').addEventListener('click', () => $('bakeModal').classList.remove('open'));
    $('bakeConfirm').addEventListener('click', bakeScene);

    // ── Export ──
    $('btnExport').addEventListener('click', exportScene);

    // ── Refresh models ──
    $('btnRefreshModels').addEventListener('click', refreshGeneratedModels);

    // ── Context menu prevention ──
    canvas.addEventListener('contextmenu', e => e.preventDefault());
}

let pointerDownPos = new THREE.Vector2();
let pointerIsDown = false;

function onPointerDown(e) {
    if (e.button !== 0) return; // left click only
    pointerIsDown = true;
    pointerDownPos.set(e.clientX, e.clientY);
}

function onPointerUp(e) {
    if (e.button !== 0 || !pointerIsDown) return;
    pointerIsDown = false;

    // Only select if mouse didn't move much (not a drag from gizmo)
    const dx = e.clientX - pointerDownPos.x;
    const dy = e.clientY - pointerDownPos.y;
    if (Math.sqrt(dx*dx + dy*dy) > 4) return;

    // If transform gizmo is being dragged, don't select
    if (transformControls.dragging) return;

    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Only test world objects
    const meshes = worldObjects.map(o => o.mesh);
    // Need to recurse into children for loaded GLBs
    const testTargets = [];
    worldObjects.forEach(o => {
        o.mesh.traverse(child => {
            if (child.isMesh) testTargets.push(child);
        });
    });

    const hits = raycaster.intersectObjects(testTargets, false);

    if (hits.length > 0) {
        // Find which world object this belongs to
        let hitObj = hits[0].object;
        const wo = worldObjects.find(o => {
            let found = false;
            o.mesh.traverse(c => { if (c === hitObj) found = true; });
            return found;
        });
        if (wo) selectWorldObject(wo);
    } else {
        deselectAll();
    }
}

function onKeyDown(e) {
    // Don't trigger shortcuts when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key.toLowerCase()) {
        case 'v': setTool('select'); break;
        case 'g': setTool('move'); break;
        case 'r': setTool('rotate'); break;
        case 's': if (!e.ctrlKey && !e.metaKey) setTool('scale'); break;
        case 'x': case 'delete': case 'backspace': deleteSelected(); break;
        case 'd': if (e.shiftKey) { e.preventDefault(); duplicateSelected(); } break;
        case 'escape': deselectAll(); break;
        case 'f': if (selectedObject) focusObject(selectedObject); break;
    }
}

// ══════════════════════════════════════════════════════════════
// TOOLS
// ══════════════════════════════════════════════════════════════

function setTool(tool) {
    activeTool = tool;
    document.querySelectorAll('.we-tool[data-tool]').forEach(b => b.classList.toggle('active', b.dataset.tool === tool));

    if (!selectedObject) {
        transformControls.detach();
        return;
    }

    switch (tool) {
        case 'select':
            transformControls.detach();
            break;
        case 'move':
            transformControls.attach(selectedObject.mesh);
            transformControls.setMode('translate');
            break;
        case 'rotate':
            transformControls.attach(selectedObject.mesh);
            transformControls.setMode('rotate');
            break;
        case 'scale':
            transformControls.attach(selectedObject.mesh);
            transformControls.setMode('scale');
            break;
    }
    updateSnap();
}

function updateSnap() {
    if (snapEnabled) {
        transformControls.setTranslationSnap(snapSize);
        transformControls.setRotationSnap(THREE.MathUtils.degToRad(15));
        transformControls.setScaleSnap(0.1);
    } else {
        transformControls.setTranslationSnap(null);
        transformControls.setRotationSnap(null);
        transformControls.setScaleSnap(null);
    }
}

// ══════════════════════════════════════════════════════════════
// SELECTION
// ══════════════════════════════════════════════════════════════

function selectWorldObject(wo) {
    deselectAll(true); // silent
    selectedObject = wo;

    // Highlight
    wo.mesh.traverse(child => {
        if (child.isMesh && child.material) {
            child._origEmissive = child.material.emissive?.clone();
            child._origEmissiveIntensity = child.material.emissiveIntensity;
            if (child.material.emissive) {
                child.material.emissive.set(0xE8A917);
                child.material.emissiveIntensity = 0.08;
            }
        }
    });

    // Attach gizmo if tool is transform
    if (activeTool !== 'select') {
        transformControls.attach(wo.mesh);
        transformControls.setMode(
            activeTool === 'move' ? 'translate' :
            activeTool === 'rotate' ? 'rotate' : 'scale'
        );
    }

    updatePropertiesPanel();
    updateSceneTree();
    updateSelectionInfo();
}

function deselectAll(silent) {
    if (selectedObject) {
        selectedObject.mesh.traverse(child => {
            if (child.isMesh && child.material && child._origEmissive !== undefined) {
                child.material.emissive.copy(child._origEmissive);
                child.material.emissiveIntensity = child._origEmissiveIntensity;
            }
        });
    }
    selectedObject = null;
    transformControls.detach();
    if (!silent) {
        updatePropertiesPanel();
        updateSceneTree();
        updateSelectionInfo();
    }
}

function focusObject(wo) {
    const box = new THREE.Box3().setFromObject(wo.mesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    orbitControls.target.copy(center);
    const dir = camera.position.clone().sub(center).normalize();
    camera.position.copy(center).add(dir.multiplyScalar(maxDim * 3));
    camera.updateProjectionMatrix();
    orbitControls.update();
}

// ══════════════════════════════════════════════════════════════
// ADD OBJECTS
// ══════════════════════════════════════════════════════════════

function addPrimitive(type) {
    let geo, mat, name;
    mat = new THREE.MeshStandardMaterial({
        color: 0x888899,
        roughness: 0.6,
        metalness: 0.1,
    });

    switch (type) {
        case 'box':
            geo = new THREE.BoxGeometry(1, 1, 1);
            name = 'Box';
            break;
        case 'sphere':
            geo = new THREE.SphereGeometry(0.5, 32, 24);
            name = 'Sphere';
            break;
        case 'cylinder':
            geo = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
            name = 'Cylinder';
            break;
        case 'plane':
            geo = new THREE.PlaneGeometry(10, 10);
            mat = new THREE.MeshStandardMaterial({ color: 0x556666, roughness: 0.8, side: THREE.DoubleSide });
            name = 'Ground';
            break;
        case 'cone':
            geo = new THREE.ConeGeometry(0.5, 1, 32);
            name = 'Cone';
            break;
        case 'torus':
            geo = new THREE.TorusGeometry(0.4, 0.15, 16, 48);
            name = 'Torus';
            break;
        default: return;
    }

    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    if (type === 'plane') {
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = 0.001;
        mesh.receiveShadow = true;
        mesh.castShadow = false;
    } else {
        // Place on ground, centered at camera target
        const target = orbitControls.target.clone();
        mesh.position.set(
            snapValue(target.x),
            type === 'box' ? 0.5 : type === 'cylinder' ? 0.5 : type === 'cone' ? 0.5 : 0.5,
            snapValue(target.z)
        );
    }

    const wo = registerObject(mesh, name, 'primitive');
    scene.add(mesh);
    selectWorldObject(wo);
    hideEmptyState();
}

function addGLBToScene(url, name, position) {
    return new Promise((resolve, reject) => {
        gltfLoader.load(url, (gltf) => {
            const root = gltf.scene || gltf.scenes?.[0];
            if (!root) { reject('No scene in GLB'); return; }

            // Normalize: center and scale to reasonable size
            const box = new THREE.Box3().setFromObject(root);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z) || 1;

            // Target ~2 units tall
            const targetSize = 2;
            const scaleFactor = targetSize / maxDim;

            // Create a container group
            const container = new THREE.Group();
            container.name = name;

            root.position.sub(center).multiplyScalar(scaleFactor);
            root.scale.multiplyScalar(scaleFactor);
            root.updateMatrixWorld(true);

            // Recalculate after scaling
            const newBox = new THREE.Box3().setFromObject(root);
            const yOffset = -newBox.min.y * 1;

            // Move so bottom sits on y=0
            root.position.y += yOffset;

            container.add(root);

            // Enable shadows
            container.traverse(child => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            // Place at position
            if (position) {
                container.position.copy(position);
            } else {
                const t = orbitControls.target.clone();
                container.position.set(snapValue(t.x), 0, snapValue(t.z));
            }

            scene.add(container);
            const wo = registerObject(container, name, 'glb', url);
            selectWorldObject(wo);
            hideEmptyState();
            resolve(wo);
        }, undefined, (err) => {
            console.error('GLB load error:', err);
            reject(err);
        });
    });
}

function registerObject(mesh, name, type, url) {
    const id = nextId++;
    const displayName = `${name}_${id}`;
    const wo = { id, name: displayName, mesh, type, url: url || null };
    worldObjects.push(wo);
    mesh.userData.worldObjectId = id;
    updateSceneTree();
    updateHud();
    return wo;
}

function snapValue(v) {
    return snapEnabled ? Math.round(v / snapSize) * snapSize : v;
}

// ══════════════════════════════════════════════════════════════
// DUPLICATE / DELETE
// ══════════════════════════════════════════════════════════════

function duplicateSelected() {
    if (!selectedObject) return;
    const wo = selectedObject;

    const cloned = wo.mesh.clone();
    // Deep clone materials
    cloned.traverse(child => {
        if (child.isMesh && child.material) {
            child.material = child.material.clone();
        }
    });

    // Offset position
    cloned.position.x += 1;
    cloned.position.z += 1;

    scene.add(cloned);
    const newWo = registerObject(cloned, wo.name.replace(/_\d+$/, ''), wo.type, wo.url);
    selectWorldObject(newWo);
}

function deleteSelected() {
    if (!selectedObject) return;
    const wo = selectedObject;
    deselectAll();

    scene.remove(wo.mesh);
    disposeObject(wo.mesh);

    const idx = worldObjects.indexOf(wo);
    if (idx >= 0) worldObjects.splice(idx, 1);

    updateSceneTree();
    updateHud();
    if (worldObjects.length === 0) showEmptyState();
}

function disposeObject(obj) {
    obj.traverse(child => {
        if (child.isMesh) {
            child.geometry?.dispose();
            if (Array.isArray(child.material)) child.material.forEach(m => m?.dispose());
            else child.material?.dispose();
        }
    });
}

// ══════════════════════════════════════════════════════════════
// FILE DROP / MODEL LOADING
// ══════════════════════════════════════════════════════════════

function handleFileDrop(files) {
    for (const file of files) {
        if (!file.name.match(/\.(glb|gltf)$/i)) continue;
        const url = URL.createObjectURL(file);
        const name = file.name.replace(/\.(glb|gltf)$/i, '');
        addGLBToScene(url, name);
    }
}

async function refreshGeneratedModels() {
    const list = $('generatedModelList');
    list.innerHTML = '<div class="we-model-empty">Loading…</div>';

    try {
        // Try to get models from the TRELLIS API
        const r = await fetch('/api/worldeditor/models');
        if (!r.ok) throw new Error('API not available');
        const data = await r.json();

        if (!data.models || data.models.length === 0) {
            list.innerHTML = '<div class="we-model-empty">No generated models found.<br>Generate models in Studio first.</div>';
            return;
        }

        list.innerHTML = '';
        data.models.forEach(m => {
            const item = document.createElement('div');
            item.className = 'we-model-item';
            item.draggable = true;
            item.innerHTML = `
                <span class="mi-icon">📦</span>
                <span class="mi-name" title="${m.name}">${m.name}</span>
                <button class="mi-add" title="Add to scene">+</button>
            `;
            item.querySelector('.mi-add').addEventListener('click', (e) => {
                e.stopPropagation();
                addGLBToScene(m.url, m.name);
            });
            // Drag from panel
            item.addEventListener('dragstart', e => {
                e.dataTransfer.setData('application/x-trellis-model', JSON.stringify(m));
                e.dataTransfer.effectAllowed = 'copy';
            });
            list.appendChild(item);
        });
    } catch (e) {
        // Fallback: scan completed models from the main app if available
        list.innerHTML = '<div class="we-model-empty">Connect via Studio to browse models, or drop GLB files below.</div>';
    }
}

// Handle drop from asset panel onto viewport
viewport.addEventListener('drop', (e) => {
    const modelData = e.dataTransfer.getData('application/x-trellis-model');
    if (modelData) {
        try {
            const m = JSON.parse(modelData);
            // Calculate drop position on ground
            const rect = canvas.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);

            const groundHit = raycaster.intersectObject(groundPlane);
            let pos = null;
            if (groundHit.length > 0) {
                pos = new THREE.Vector3(
                    snapValue(groundHit[0].point.x),
                    0,
                    snapValue(groundHit[0].point.z)
                );
            }
            addGLBToScene(m.url, m.name, pos);
        } catch (err) {}
    }
});

// ══════════════════════════════════════════════════════════════
// PROPERTIES PANEL
// ══════════════════════════════════════════════════════════════

function updatePropertiesPanel() {
    if (!selectedObject) {
        ['propPosX','propPosY','propPosZ','propRotX','propRotY','propRotZ','propScX','propScY','propScZ'].forEach(id => $(id).value = '');
        return;
    }
    const m = selectedObject.mesh;
    $('propPosX').value = round3(m.position.x);
    $('propPosY').value = round3(m.position.y);
    $('propPosZ').value = round3(m.position.z);
    $('propRotX').value = round3(THREE.MathUtils.radToDeg(m.rotation.x));
    $('propRotY').value = round3(THREE.MathUtils.radToDeg(m.rotation.y));
    $('propRotZ').value = round3(THREE.MathUtils.radToDeg(m.rotation.z));
    $('propScX').value = round3(m.scale.x);
    $('propScY').value = round3(m.scale.y);
    $('propScZ').value = round3(m.scale.z);
}

function onPropertyInput(e) {
    if (!selectedObject) return;
    const m = selectedObject.mesh;
    const id = e.target.id;
    const v = parseFloat(e.target.value) || 0;

    if (id === 'propPosX') m.position.x = v;
    else if (id === 'propPosY') m.position.y = v;
    else if (id === 'propPosZ') m.position.z = v;
    else if (id === 'propRotX') m.rotation.x = THREE.MathUtils.degToRad(v);
    else if (id === 'propRotY') m.rotation.y = THREE.MathUtils.degToRad(v);
    else if (id === 'propRotZ') m.rotation.z = THREE.MathUtils.degToRad(v);
    else if (id === 'propScX') {
        m.scale.x = v;
        if ($('propScLock').checked) { m.scale.y = v; m.scale.z = v; $('propScY').value = round3(v); $('propScZ').value = round3(v); }
    }
    else if (id === 'propScY') {
        m.scale.y = v;
        if ($('propScLock').checked) { m.scale.x = v; m.scale.z = v; $('propScX').value = round3(v); $('propScZ').value = round3(v); }
    }
    else if (id === 'propScZ') {
        m.scale.z = v;
        if ($('propScLock').checked) { m.scale.x = v; m.scale.y = v; $('propScX').value = round3(v); $('propScY').value = round3(v); }
    }
}

function round3(v) { return Math.round(v * 1000) / 1000; }

// ══════════════════════════════════════════════════════════════
// SCENE TREE
// ══════════════════════════════════════════════════════════════

function updateSceneTree() {
    const tree = $('sceneTree');
    if (worldObjects.length === 0) {
        tree.innerHTML = '<div class="we-tree-empty">Empty scene</div>';
        return;
    }

    tree.innerHTML = '';
    worldObjects.forEach(wo => {
        const item = document.createElement('div');
        item.className = 'we-tree-item' + (selectedObject === wo ? ' selected' : '');
        const icon = wo.type === 'glb' ? '📦' : '◆';
        item.innerHTML = `
            <span class="ti-icon">${icon}</span>
            <span class="ti-name">${wo.name}</span>
            <button class="ti-vis ${wo.mesh.visible ? '' : 'hidden'}" title="Toggle visibility">👁</button>
        `;
        item.addEventListener('click', (e) => {
            if (e.target.closest('.ti-vis')) return;
            selectWorldObject(wo);
        });
        item.querySelector('.ti-vis').addEventListener('click', () => {
            wo.mesh.visible = !wo.mesh.visible;
            updateSceneTree();
        });
        tree.appendChild(item);
    });
}

function updateSelectionInfo() {
    $('selName').textContent = selectedObject ? selectedObject.name : 'Nothing selected';
}

function updateHud() {
    $('hudObjs').textContent = `Objects: ${worldObjects.length}`;
    let tris = 0;
    worldObjects.forEach(wo => {
        wo.mesh.traverse(child => {
            if (child.isMesh && child.geometry) {
                const idx = child.geometry.index;
                const pos = child.geometry.attributes?.position;
                tris += idx ? idx.count / 3 : (pos?.count || 0) / 3;
            }
        });
    });
    $('hudTriCount').textContent = `Tris: ${formatNum(Math.round(tris))}`;
}

function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

// ══════════════════════════════════════════════════════════════
// EMPTY STATE
// ══════════════════════════════════════════════════════════════

function hideEmptyState() { $('weEmptyState').classList.add('hidden'); }
function showEmptyState() { $('weEmptyState').classList.remove('hidden'); }

// ══════════════════════════════════════════════════════════════
// BAKE — Render high-quality screenshot with baked shadows
// ══════════════════════════════════════════════════════════════

function bakeScene() {
    const res = parseInt($('bakeRes').value);
    const shadowRes = parseInt($('bakeShadow').value);
    const transparent = $('bakeTransparent').checked;

    // Create an offscreen renderer at high res
    const bakeRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: transparent,
        preserveDrawingBuffer: true,
    });
    bakeRenderer.outputColorSpace = THREE.SRGBColorSpace;
    bakeRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    bakeRenderer.toneMappingExposure = 1.1;
    bakeRenderer.shadowMap.enabled = true;
    bakeRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
    bakeRenderer.setSize(res, res);
    bakeRenderer.setPixelRatio(1);

    if (transparent) {
        bakeRenderer.setClearColor(0x000000, 0);
    } else {
        bakeRenderer.setClearColor(scene.background, 1);
    }

    // Temporarily upgrade shadow map
    const origShadowSize = sunLight.shadow.mapSize.clone();
    sunLight.shadow.mapSize.set(shadowRes, shadowRes);
    sunLight.shadow.map?.dispose();
    sunLight.shadow.map = null;

    // Hide grid and transform controls for bake
    const gridWasVisible = groundGrid.visible;
    groundGrid.visible = false;
    const tcWasVisible = transformControls.visible;
    transformControls.visible = false;

    // Make a shadow ground more visible for bake
    const origOpacity = groundPlane.material.opacity;
    groundPlane.material.opacity = 0.5;

    // Create bake camera matching current view
    const bakeCamera = camera.clone();
    bakeCamera.aspect = 1;
    bakeCamera.updateProjectionMatrix();

    // Render
    bakeRenderer.render(scene, bakeCamera);

    // Extract image
    const dataUrl = bakeRenderer.domElement.toDataURL('image/png');

    // Restore
    groundGrid.visible = gridWasVisible;
    transformControls.visible = tcWasVisible;
    groundPlane.material.opacity = origOpacity;
    sunLight.shadow.mapSize.copy(origShadowSize);
    sunLight.shadow.map?.dispose();
    sunLight.shadow.map = null;
    bakeRenderer.dispose();

    // Download
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = 'trellis_world_bake.png';
    a.click();

    $('bakeModal').classList.remove('open');
}

// ══════════════════════════════════════════════════════════════
// EXPORT — Save entire scene as a single GLB
// ══════════════════════════════════════════════════════════════

function exportScene() {
    if (worldObjects.length === 0) {
        alert('Nothing to export — add some objects first.');
        return;
    }

    // Create export group with only user objects
    const exportGroup = new THREE.Group();
    worldObjects.forEach(wo => {
        if (wo.mesh.visible) {
            const clone = wo.mesh.clone();
            exportGroup.add(clone);
        }
    });

    const exporter = new GLTFExporter();
    exporter.parse(exportGroup, (result) => {
        const blob = new Blob([result], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'trellis_world.glb';
        a.click();
        URL.revokeObjectURL(url);
    }, (err) => {
        console.error('Export error:', err);
        alert('Export failed: ' + err.message);
    }, { binary: true });
}

// ══════════════════════════════════════════════════════════════
// RENDER LOOP
// ══════════════════════════════════════════════════════════════

function animate() {
    requestAnimationFrame(animate);

    orbitControls.update();
    renderer.render(scene, camera);
    updateAxisIndicator();

    // FPS
    frameCount++;
    const now = clock.getElapsedTime();
    if (now - fpsTime >= 0.5) {
        currentFps = Math.round(frameCount / (now - fpsTime));
        $('hudFps').textContent = currentFps + ' fps';
        frameCount = 0;
        fpsTime = now;
    }
}

function onResize() {
    const w = viewport.clientWidth || 1;
    const h = viewport.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

// ══════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════

init();
animate();
updateSnap();
