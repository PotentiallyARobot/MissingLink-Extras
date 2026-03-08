// ============================================================
// RTS Lab — World Editor Engine
//
// Features:
//  • RTS 2.5D camera (RMB pan, MMB orbit, scroll zoom)
//  • GLB loading from TRELLIS output or file drop
//  • Transform gizmos (move/rotate/scale) with grid snap
//  • Procedural terrain generator (diamond-square heightmap)
//  • Persistent save/load to Google Drive
//  • Bake high-res screenshot with shadows
//  • Export entire scene as GLB
//  • Full scene graph + property editing
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// ── Helpers ──
const $ = id => document.getElementById(id);
const round3 = v => Math.round(v * 1000) / 1000;
const fmtNum = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n);

// ══════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════

let renderer, scene, camera, orbit, gizmo, gltfLoader;
let clock, raycaster, mouse;
let ground, grid, sunLight, ambient;

const objects = [];     // { id, name, mesh, type, glbPath?, glbUrl? }
let selected = null;
let nextId = 1;
let activeTool = 'select';
let snapOn = true, snapSize = 0.5;
let worldName = 'untitled';

// FPS
let frameCnt = 0, fpsT = 0, fps = 0;

// Axis mini-view
let axScene, axCam, axRenderer;

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════

function init() {
    const vp = $('viewport');
    const canvas = $('canvas3d');

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x1a1a2e, 1);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    scene.fog = new THREE.FogExp2(0x1a1a2e, 0.012);

    camera = new THREE.PerspectiveCamera(35, 1, 0.1, 600);
    camera.position.set(14, 16, 14);

    // RTS controls: RMB pan, MMB orbit, scroll zoom
    orbit = new OrbitControls(camera, canvas);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.1;
    orbit.screenSpacePanning = true;
    orbit.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.ROTATE, RIGHT: THREE.MOUSE.PAN };
    orbit.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
    orbit.minPolarAngle = 0.15;
    orbit.maxPolarAngle = Math.PI / 2.1;
    orbit.minDistance = 3;
    orbit.maxDistance = 120;
    orbit.target.set(0, 0, 0);

    // Transform gizmo
    gizmo = new TransformControls(camera, canvas);
    gizmo.setSize(0.65);
    gizmo.addEventListener('dragging-changed', e => { orbit.enabled = !e.value; });
    gizmo.addEventListener('objectChange', () => { updateProps(); updateTree(); });
    scene.add(gizmo);

    // Lights
    ambient = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambient);

    sunLight = new THREE.DirectionalLight(0xfff5e6, 1.8);
    setSunAngle(45);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(2048, 2048);
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 100;
    const sh = 35;
    sunLight.shadow.camera.left = -sh;
    sunLight.shadow.camera.right = sh;
    sunLight.shadow.camera.top = sh;
    sunLight.shadow.camera.bottom = -sh;
    sunLight.shadow.bias = -0.0008;
    sunLight.shadow.normalBias = 0.02;
    scene.add(sunLight);
    scene.add(sunLight.target);

    scene.add(new THREE.HemisphereLight(0x8899bb, 0x445566, 0.25));

    // Ground
    createGround();

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    gltfLoader = new GLTFLoader();
    clock = new THREE.Clock();

    initAxis();
    setupEvents();

    new ResizeObserver(onResize).observe(vp);
    onResize();

    refreshModels();
    refreshWorlds();
    applySnap();
}

// ══════════════════════════════════════════════════════════════
// GROUND
// ══════════════════════════════════════════════════════════════

function createGround() {
    const geo = new THREE.PlaneGeometry(300, 300);
    const mat = new THREE.ShadowMaterial({ opacity: 0.3 });
    ground = new THREE.Mesh(geo, mat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.name = '__ground';
    scene.add(ground);

    grid = new THREE.GridHelper(120, 120, 0x2a3a4a, 0x1a2530);
    grid.position.y = 0.002;
    grid.name = '__grid';
    scene.add(grid);
}

function setSunAngle(deg) {
    const r = THREE.MathUtils.degToRad(deg);
    const d = 30;
    sunLight.position.set(d * Math.cos(r) * 0.7, d * Math.sin(r), d * Math.cos(r));
    sunLight.target.position.set(0, 0, 0);
}

// ══════════════════════════════════════════════════════════════
// AXIS INDICATOR
// ══════════════════════════════════════════════════════════════

function initAxis() {
    axScene = new THREE.Scene();
    axCam = new THREE.PerspectiveCamera(50, 1, 0.1, 10);
    axCam.position.set(0, 0, 2.5);
    axRenderer = new THREE.WebGLRenderer({ canvas: $('axisCanvas'), alpha: true, antialias: true });
    axRenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    axRenderer.setSize(70, 70, false);
    axScene.add(new THREE.AxesHelper(0.8));
}

// ══════════════════════════════════════════════════════════════
// EVENTS
// ══════════════════════════════════════════════════════════════

let ptrDown = new THREE.Vector2(), ptrIsDown = false;

function setupEvents() {
    const c = $('canvas3d');

    c.addEventListener('pointerdown', e => {
        if (e.button !== 0) return;
        ptrIsDown = true;
        ptrDown.set(e.clientX, e.clientY);
    });

    c.addEventListener('pointerup', e => {
        if (e.button !== 0 || !ptrIsDown) return;
        ptrIsDown = false;
        const dx = e.clientX - ptrDown.x, dy = e.clientY - ptrDown.y;
        if (Math.sqrt(dx*dx+dy*dy) > 5 || gizmo.dragging) return;

        const rect = c.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const targets = [];
        objects.forEach(o => o.mesh.traverse(ch => { if (ch.isMesh) targets.push(ch); }));
        const hits = raycaster.intersectObjects(targets, false);

        if (hits.length) {
            const hit = hits[0].object;
            const wo = objects.find(o => { let f = false; o.mesh.traverse(ch => { if (ch === hit) f = true; }); return f; });
            if (wo) select(wo);
            else deselect();
        } else {
            deselect();
        }
    });

    c.addEventListener('contextmenu', e => e.preventDefault());

    // Viewport drop
    const vp = $('viewport');
    vp.addEventListener('dragover', e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
    vp.addEventListener('drop', e => {
        e.preventDefault();
        const md = e.dataTransfer.getData('application/x-rtslab-model');
        if (md) {
            const m = JSON.parse(md);
            const pos = getGroundHit(e);
            addGLB(m.glb_url, m.name, m.glb_path, pos);
            return;
        }
        handleFileDrop(e.dataTransfer.files);
    });

    // Keyboard
    window.addEventListener('keydown', onKey);

    // Tool buttons
    document.querySelectorAll('.tb[data-tool]').forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));
    $('btnDuplicate').addEventListener('click', duplicate);
    $('btnDelete').addEventListener('click', deleteObj);
    $('btnSnap').addEventListener('click', () => { snapOn = !snapOn; $('btnSnap').classList.toggle('active', snapOn); applySnap(); });
    $('snapSize').addEventListener('change', () => { snapSize = parseFloat($('snapSize').value); applySnap(); });

    // Primitives
    document.querySelectorAll('.prim').forEach(b => b.addEventListener('click', () => addPrimitive(b.dataset.prim)));

    // Terrain
    $('btnGenTerrain').addEventListener('click', generateTerrain);

    // Properties
    ['pPosX','pPosY','pPosZ','pRotX','pRotY','pRotZ','pScX','pScY','pScZ'].forEach(id => {
        $(id).addEventListener('input', onPropInput);
    });

    // Panel tabs
    document.querySelectorAll('.ptab').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('#panelLeft, #panelRight');
            parent.querySelectorAll('.ptab').forEach(b => b.classList.remove('active'));
            parent.querySelectorAll('.panel-body').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            $(btn.dataset.panel).classList.add('active');
        });
    });

    // Environment
    $('envSky').addEventListener('input', e => { const c = new THREE.Color(e.target.value); scene.background = c; scene.fog.color = c; renderer.setClearColor(c); });
    $('envAmbient').addEventListener('input', e => { ambient.intensity = parseFloat(e.target.value); });
    $('envSunAngle').addEventListener('input', e => { setSunAngle(parseFloat(e.target.value)); });
    $('envSunColor').addEventListener('input', e => { sunLight.color.set(e.target.value); });
    $('envFog').addEventListener('input', e => { scene.fog.density = parseFloat(e.target.value); });
    $('envGround').addEventListener('change', e => { grid.visible = e.target.value === 'grid'; ground.visible = e.target.value !== 'none'; });
    $('envGroundColor').addEventListener('input', e => {
        if (ground.material.type !== 'ShadowMaterial') ground.material.color.set(e.target.value);
    });

    // File input dropzone
    const dz = $('dropzone'), fi = $('fileInput');
    ['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('over'); }));
    ['dragleave','drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('over'); }));
    dz.addEventListener('drop', ev => handleFileDrop(ev.dataTransfer.files));
    fi.addEventListener('change', ev => { handleFileDrop(ev.target.files); fi.value = ''; });

    // Save / Load
    $('btnSaveWorld').addEventListener('click', saveWorld);
    $('btnRefreshWorlds').addEventListener('click', refreshWorlds);
    $('btnRefreshModels').addEventListener('click', refreshModels);

    // Bake
    $('btnBake').addEventListener('click', () => $('bakeModal').classList.add('open'));
    $('bakeModalClose').addEventListener('click', () => $('bakeModal').classList.remove('open'));
    $('bakeCancel').addEventListener('click', () => $('bakeModal').classList.remove('open'));
    $('bakeGo').addEventListener('click', bakeView);

    // Export
    $('btnExport').addEventListener('click', exportScene);
}

function onKey(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key.toLowerCase()) {
        case 'v': setTool('select'); break;
        case 'g': setTool('move'); break;
        case 'r': setTool('rotate'); break;
        case 's': if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); setTool('scale'); } break;
        case 'x': case 'delete': case 'backspace': deleteObj(); break;
        case 'd': if (e.shiftKey) { e.preventDefault(); duplicate(); } break;
        case 'escape': deselect(); break;
        case 'f': if (selected) focusObj(selected); break;
    }
}

function getGroundHit(e) {
    const rect = $('canvas3d').getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
    const hits = raycaster.intersectObject(ground);
    if (hits.length) return new THREE.Vector3(snap(hits[0].point.x), 0, snap(hits[0].point.z));
    return null;
}

// ══════════════════════════════════════════════════════════════
// TOOLS
// ══════════════════════════════════════════════════════════════

function setTool(t) {
    activeTool = t;
    document.querySelectorAll('.tb[data-tool]').forEach(b => b.classList.toggle('active', b.dataset.tool === t));
    if (!selected) { gizmo.detach(); return; }
    if (t === 'select') gizmo.detach();
    else { gizmo.attach(selected.mesh); gizmo.setMode(t === 'move' ? 'translate' : t); }
    applySnap();
}

function applySnap() {
    if (snapOn) {
        gizmo.setTranslationSnap(snapSize);
        gizmo.setRotationSnap(THREE.MathUtils.degToRad(15));
        gizmo.setScaleSnap(0.1);
    } else {
        gizmo.setTranslationSnap(null);
        gizmo.setRotationSnap(null);
        gizmo.setScaleSnap(null);
    }
}

function snap(v) { return snapOn ? Math.round(v / snapSize) * snapSize : v; }

// ══════════════════════════════════════════════════════════════
// SELECTION
// ══════════════════════════════════════════════════════════════

function select(wo) {
    deselect(true);
    selected = wo;
    wo.mesh.traverse(ch => {
        if (ch.isMesh && ch.material) {
            ch._savedEmissive = ch.material.emissive?.clone();
            ch._savedEI = ch.material.emissiveIntensity;
            if (ch.material.emissive) { ch.material.emissive.set(0xE8A917); ch.material.emissiveIntensity = 0.07; }
        }
    });
    if (activeTool !== 'select') {
        gizmo.attach(wo.mesh);
        gizmo.setMode(activeTool === 'move' ? 'translate' : activeTool);
    }
    updateProps();
    updateTree();
    $('selName').textContent = wo.name;
}

function deselect(silent) {
    if (selected) {
        selected.mesh.traverse(ch => {
            if (ch.isMesh && ch.material && ch._savedEmissive !== undefined) {
                ch.material.emissive.copy(ch._savedEmissive);
                ch.material.emissiveIntensity = ch._savedEI;
            }
        });
    }
    selected = null;
    gizmo.detach();
    if (!silent) { updateProps(); updateTree(); $('selName').textContent = 'Nothing selected'; }
}

function focusObj(wo) {
    const box = new THREE.Box3().setFromObject(wo.mesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    orbit.target.copy(center);
    const dir = camera.position.clone().sub(center).normalize();
    camera.position.copy(center).add(dir.multiplyScalar(Math.max(size.x, size.y, size.z) * 3));
    orbit.update();
}

// ══════════════════════════════════════════════════════════════
// ADD OBJECTS
// ══════════════════════════════════════════════════════════════

function addPrimitive(type) {
    let geo, mat, name;
    mat = new THREE.MeshStandardMaterial({ color: 0x788899, roughness: 0.7, metalness: 0.05 });

    switch (type) {
        case 'box': geo = new THREE.BoxGeometry(1,1,1); name = 'Box'; break;
        case 'sphere': geo = new THREE.SphereGeometry(0.5,32,24); name = 'Sphere'; break;
        case 'cylinder': geo = new THREE.CylinderGeometry(0.5,0.5,1,32); name = 'Cylinder'; break;
        case 'plane':
            geo = new THREE.PlaneGeometry(10,10);
            mat = new THREE.MeshStandardMaterial({ color: 0x2a3a2a, roughness: 0.9, side: THREE.DoubleSide });
            name = 'Ground'; break;
        case 'cone': geo = new THREE.ConeGeometry(0.5,1,32); name = 'Cone'; break;
        case 'rocks': return addRockCluster();
        default: return;
    }

    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;

    if (type === 'plane') { mesh.rotation.x = -Math.PI/2; mesh.position.y = 0.002; mesh.castShadow = false; }
    else {
        const t = orbit.target.clone();
        mesh.position.set(snap(t.x), 0.5, snap(t.z));
    }

    const wo = register(mesh, name, 'primitive');
    scene.add(mesh);
    select(wo);
    hideEmpty();
}

function addRockCluster() {
    const group = new THREE.Group();
    group.name = 'Rocks';

    const rockCount = 3 + Math.floor(Math.random() * 4);
    for (let i = 0; i < rockCount; i++) {
        const s = 0.2 + Math.random() * 0.6;
        const geo = new THREE.DodecahedronGeometry(s, 1);
        // Deform vertices for organic shape
        const pos = geo.attributes.position;
        for (let j = 0; j < pos.count; j++) {
            pos.setXYZ(j,
                pos.getX(j) * (0.7 + Math.random() * 0.6),
                pos.getY(j) * (0.5 + Math.random() * 0.5),
                pos.getZ(j) * (0.7 + Math.random() * 0.6),
            );
        }
        geo.computeVertexNormals();

        const shade = 0.3 + Math.random() * 0.2;
        const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(shade, shade * 0.95, shade * 0.85), roughness: 0.9 });
        const rock = new THREE.Mesh(geo, mat);
        rock.position.set((Math.random()-0.5)*2, s*0.4, (Math.random()-0.5)*2);
        rock.rotation.set(Math.random()*0.5, Math.random()*Math.PI*2, Math.random()*0.3);
        rock.castShadow = true; rock.receiveShadow = true;
        group.add(rock);
    }

    const t = orbit.target.clone();
    group.position.set(snap(t.x), 0, snap(t.z));
    scene.add(group);
    const wo = register(group, 'Rocks', 'primitive');
    select(wo);
    hideEmpty();
}

function addGLB(url, name, glbPath, position) {
    gltfLoader.load(url, (gltf) => {
        const root = gltf.scene || gltf.scenes?.[0];
        if (!root) return;

        const box = new THREE.Box3().setFromObject(root);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;

        const container = new THREE.Group();
        container.name = name;

        const scaleFactor = 2 / maxDim;
        root.position.sub(center).multiplyScalar(scaleFactor);
        root.scale.multiplyScalar(scaleFactor);
        root.updateMatrixWorld(true);

        const newBox = new THREE.Box3().setFromObject(root);
        root.position.y += -newBox.min.y;

        container.add(root);
        container.traverse(ch => { if (ch.isMesh) { ch.castShadow = true; ch.receiveShadow = true; } });

        if (position) container.position.copy(position);
        else { const t = orbit.target.clone(); container.position.set(snap(t.x), 0, snap(t.z)); }

        scene.add(container);
        const wo = register(container, name, 'glb', glbPath, url);
        select(wo);
        hideEmpty();
    }, undefined, err => console.error('GLB error:', err));
}

function register(mesh, name, type, glbPath, glbUrl) {
    const id = nextId++;
    const displayName = `${name}_${id}`;
    const wo = { id, name: displayName, mesh, type, glbPath: glbPath || null, glbUrl: glbUrl || null };
    objects.push(wo);
    mesh.userData._woId = id;
    updateTree();
    updateHud();
    return wo;
}

// ══════════════════════════════════════════════════════════════
// DUPLICATE / DELETE
// ══════════════════════════════════════════════════════════════

function duplicate() {
    if (!selected) return;
    const wo = selected;
    const cl = wo.mesh.clone();
    cl.traverse(ch => { if (ch.isMesh && ch.material) ch.material = ch.material.clone(); });
    cl.position.x += 1.5;
    cl.position.z += 1.5;
    scene.add(cl);
    select(register(cl, wo.name.replace(/_\d+$/, ''), wo.type, wo.glbPath, wo.glbUrl));
}

function deleteObj() {
    if (!selected) return;
    const wo = selected;
    deselect();
    scene.remove(wo.mesh);
    wo.mesh.traverse(ch => { if (ch.isMesh) { ch.geometry?.dispose(); if (Array.isArray(ch.material)) ch.material.forEach(m => m?.dispose()); else ch.material?.dispose(); } });
    const i = objects.indexOf(wo);
    if (i >= 0) objects.splice(i, 1);
    updateTree(); updateHud();
    if (objects.length === 0) showEmpty();
}

// ══════════════════════════════════════════════════════════════
// TERRAIN GENERATOR (Diamond-Square)
// ══════════════════════════════════════════════════════════════

function generateTerrain() {
    const mapSize = parseInt($('terrainSize').value);
    const maxH = parseFloat($('terrainHeight').value);
    const roughness = parseFloat($('terrainRough').value);
    const seed = parseInt($('terrainSeed').value);

    // Diamond-square heightmap
    const n = Math.pow(2, Math.ceil(Math.log2(mapSize))) + 1;
    const hmap = new Float32Array(n * n);

    // Seeded random
    let _s = seed;
    const srand = () => { _s = (_s * 16807 + 0) % 2147483647; return (_s / 2147483647) - 0.5; };

    // Corners
    hmap[0] = srand() * maxH;
    hmap[n - 1] = srand() * maxH;
    hmap[(n-1)*n] = srand() * maxH;
    hmap[(n-1)*n + n-1] = srand() * maxH;

    let step = n - 1;
    let scale = roughness;

    while (step > 1) {
        const half = step >> 1;

        // Diamond
        for (let y = half; y < n; y += step) {
            for (let x = half; x < n; x += step) {
                const a = hmap[(y-half)*n + (x-half)];
                const b = hmap[(y-half)*n + (x+half)];
                const c = hmap[(y+half)*n + (x-half)];
                const d = hmap[(y+half)*n + (x+half)];
                hmap[y*n + x] = (a+b+c+d)/4 + srand()*scale;
            }
        }

        // Square
        for (let y = 0; y < n; y += half) {
            for (let x = ((y/half) % 2 === 0 ? half : 0); x < n; x += step) {
                let sum = 0, cnt = 0;
                if (y >= half) { sum += hmap[(y-half)*n+x]; cnt++; }
                if (y+half < n) { sum += hmap[(y+half)*n+x]; cnt++; }
                if (x >= half) { sum += hmap[y*n+(x-half)]; cnt++; }
                if (x+half < n) { sum += hmap[y*n+(x+half)]; cnt++; }
                hmap[y*n+x] = sum/cnt + srand()*scale;
            }
        }

        step = half;
        scale *= 0.5;
    }

    // Ensure edges are at 0 for clean borders
    for (let i = 0; i < n; i++) {
        hmap[i] = 0;
        hmap[(n-1)*n+i] = 0;
        hmap[i*n] = 0;
        hmap[i*n+n-1] = 0;
    }

    // Clamp negatives
    for (let i = 0; i < hmap.length; i++) hmap[i] = Math.max(hmap[i], 0);

    // Build mesh
    const geo = new THREE.PlaneGeometry(mapSize, mapSize, n-1, n-1);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const col = i % n;
        const row = Math.floor(i / n);
        pos.setY(i, hmap[row * n + col]);
    }
    geo.computeVertexNormals();

    // Color by height
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
        const h = pos.getY(i);
        const t = maxH > 0 ? h / maxH : 0;
        // Gradient: dark green → green → brown → grey → white
        let r, g, b;
        if (t < 0.15) { r = 0.12; g = 0.18; b = 0.08; }       // valley floor
        else if (t < 0.35) { r = 0.15; g = 0.28; b = 0.1; }    // grass
        else if (t < 0.6) { r = 0.25; g = 0.22; b = 0.14; }    // dirt
        else if (t < 0.8) { r = 0.35; g = 0.33; b = 0.3; }     // rock
        else { r = 0.55; g = 0.55; b = 0.58; }                  // snow/peak
        colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, flatShading: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.y = 0;

    scene.add(mesh);
    const wo = register(mesh, 'Terrain', 'terrain');
    select(wo);
    hideEmpty();

    // Zoom to fit
    focusObj(wo);
}

// ══════════════════════════════════════════════════════════════
// FILE DROP
// ══════════════════════════════════════════════════════════════

function handleFileDrop(files) {
    for (const f of files) {
        if (!f.name.match(/\.(glb|gltf)$/i)) continue;
        const url = URL.createObjectURL(f);
        addGLB(url, f.name.replace(/\.(glb|gltf)$/i, ''));
    }
}

// ══════════════════════════════════════════════════════════════
// MODEL LIST
// ══════════════════════════════════════════════════════════════

async function refreshModels() {
    const list = $('modelList');
    list.innerHTML = '<div class="model-empty">Scanning…</div>';
    try {
        const r = await fetch('/api/models');
        const d = await r.json();
        if (!d.models || d.models.length === 0) {
            list.innerHTML = '<div class="model-empty">No GLBs found in trellis output.<br>Generate models first, or import below.</div>';
            return;
        }
        list.innerHTML = '';
        d.models.forEach(m => {
            const el = document.createElement('div');
            el.className = 'model-item';
            el.draggable = true;
            el.innerHTML = `<span class="mi-icon">📦</span><span class="mi-name" title="${m.name}">${m.name}</span><span class="mi-size">${m.size_kb}KB</span><button class="mi-add" title="Add to world">+</button>`;
            el.querySelector('.mi-add').addEventListener('click', e => { e.stopPropagation(); addGLB(m.glb_url, m.name, m.glb_path); });
            el.addEventListener('dragstart', e => { e.dataTransfer.setData('application/x-rtslab-model', JSON.stringify(m)); e.dataTransfer.effectAllowed = 'copy'; });
            list.appendChild(el);
        });
    } catch (e) {
        list.innerHTML = '<div class="model-empty">Failed to load models.</div>';
    }
}

// ══════════════════════════════════════════════════════════════
// WORLD SAVE / LOAD
// ══════════════════════════════════════════════════════════════

async function saveWorld() {
    const name = $('worldNameInput').value.trim() || 'untitled';
    worldName = name;
    $('worldNameDisplay').textContent = name;

    const worldData = {
        camera: {
            position: camera.position.toArray(),
            target: orbit.target.toArray(),
        },
        environment: {
            sky: '#' + scene.background.getHexString(),
            ambient: ambient.intensity,
            sunAngle: parseFloat($('envSunAngle').value),
            sunColor: '#' + sunLight.color.getHexString(),
            fogDensity: scene.fog.density,
            groundStyle: $('envGround').value,
        },
        objects: objects.map(wo => ({
            name: wo.name,
            type: wo.type,
            glbPath: wo.glbPath || null,
            glbUrl: wo.glbUrl || null,
            position: wo.mesh.position.toArray(),
            rotation: [wo.mesh.rotation.x, wo.mesh.rotation.y, wo.mesh.rotation.z],
            scale: wo.mesh.scale.toArray(),
            visible: wo.mesh.visible,
        })),
    };

    try {
        const r = await fetch('/api/world/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, world: worldData }),
        });
        const d = await r.json();
        if (d.ok) {
            showToast(`💾 Saved "${name}"`);
            refreshWorlds();
        } else {
            showToast('❌ Save failed: ' + (d.error || ''));
        }
    } catch (e) {
        showToast('❌ Save failed');
    }
}

async function loadWorld(name) {
    try {
        const r = await fetch(`/api/world/load?name=${encodeURIComponent(name)}`);
        const d = await r.json();
        if (!d.ok) { showToast('❌ Load failed'); return; }

        // Clear current scene
        while (objects.length) {
            const wo = objects.pop();
            scene.remove(wo.mesh);
        }
        selected = null;
        gizmo.detach();
        nextId = 1;

        const w = d.world;
        worldName = name;
        $('worldNameDisplay').textContent = name;
        $('worldNameInput').value = name;

        // Restore environment
        if (w.environment) {
            const e = w.environment;
            if (e.sky) { const c = new THREE.Color(e.sky); scene.background = c; scene.fog.color = c; renderer.setClearColor(c); $('envSky').value = e.sky; }
            if (e.ambient !== undefined) { ambient.intensity = e.ambient; $('envAmbient').value = e.ambient; }
            if (e.sunAngle !== undefined) { setSunAngle(e.sunAngle); $('envSunAngle').value = e.sunAngle; }
            if (e.sunColor) { sunLight.color.set(e.sunColor); $('envSunColor').value = e.sunColor; }
            if (e.fogDensity !== undefined) { scene.fog.density = e.fogDensity; $('envFog').value = e.fogDensity; }
            if (e.groundStyle) { $('envGround').value = e.groundStyle; grid.visible = e.groundStyle === 'grid'; ground.visible = e.groundStyle !== 'none'; }
        }

        // Restore camera
        if (w.camera) {
            camera.position.fromArray(w.camera.position);
            orbit.target.fromArray(w.camera.target);
            orbit.update();
        }

        // Restore objects
        let loadPromises = [];
        for (const obj of (w.objects || [])) {
            if (obj.type === 'glb' && obj.glbPath) {
                // Check if we can build URL from path
                const url = obj.glbUrl || `/api/file?p=${encodeURIComponent(obj.glbPath)}`;
                loadPromises.push(new Promise(resolve => {
                    gltfLoader.load(url, (gltf) => {
                        const root = gltf.scene || gltf.scenes?.[0];
                        if (!root) { resolve(); return; }

                        const container = new THREE.Group();
                        container.name = obj.name;
                        container.add(root);
                        container.traverse(ch => { if (ch.isMesh) { ch.castShadow = true; ch.receiveShadow = true; } });
                        container.position.fromArray(obj.position);
                        container.rotation.set(...obj.rotation);
                        container.scale.fromArray(obj.scale);
                        container.visible = obj.visible !== false;

                        scene.add(container);
                        register(container, obj.name.replace(/_\d+$/, ''), 'glb', obj.glbPath, url);
                        resolve();
                    }, undefined, () => resolve());
                }));
            } else if (obj.type === 'terrain') {
                // Terrain needs to be re-generated — we save its params eventually
                // For now, place a placeholder
                showToast('⚠ Terrain must be regenerated');
            } else {
                // Primitive recreation
                const primType = obj.name.replace(/_\d+$/, '').toLowerCase();
                // Simplified: recreate using the name hint
                let geo, mat;
                mat = new THREE.MeshStandardMaterial({ color: 0x788899, roughness: 0.7 });
                switch (primType) {
                    case 'box': geo = new THREE.BoxGeometry(1,1,1); break;
                    case 'sphere': geo = new THREE.SphereGeometry(0.5,32,24); break;
                    case 'cylinder': geo = new THREE.CylinderGeometry(0.5,0.5,1,32); break;
                    case 'cone': geo = new THREE.ConeGeometry(0.5,1,32); break;
                    case 'ground':
                        geo = new THREE.PlaneGeometry(10,10);
                        mat = new THREE.MeshStandardMaterial({ color: 0x2a3a2a, roughness: 0.9, side: THREE.DoubleSide });
                        break;
                    default:
                        geo = new THREE.BoxGeometry(1,1,1); break;
                }
                const mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = true; mesh.receiveShadow = true;
                mesh.position.fromArray(obj.position);
                mesh.rotation.set(...obj.rotation);
                mesh.scale.fromArray(obj.scale);
                mesh.visible = obj.visible !== false;
                scene.add(mesh);
                register(mesh, obj.name.replace(/_\d+$/, ''), 'primitive');
            }
        }

        await Promise.all(loadPromises);
        updateTree(); updateHud();
        if (objects.length > 0) hideEmpty();
        showToast(`✅ Loaded "${name}" (${objects.length} objects)`);

    } catch (e) {
        console.error(e);
        showToast('❌ Load failed');
    }
}

async function refreshWorlds() {
    const list = $('worldList');
    list.innerHTML = '<div class="model-empty">Loading…</div>';
    try {
        const r = await fetch('/api/world/list');
        const d = await r.json();
        if (!d.worlds || d.worlds.length === 0) {
            list.innerHTML = '<div class="model-empty">No saved worlds yet.</div>';
            return;
        }
        list.innerHTML = '';
        d.worlds.forEach(w => {
            const el = document.createElement('div');
            el.className = 'world-item';
            el.innerHTML = `<span class="wi-name" title="${w.name}">${w.name}</span><span class="wi-meta">${w.object_count} obj · ${w.saved_at}</span><button class="wi-del" title="Delete">✕</button>`;
            el.addEventListener('click', e => {
                if (e.target.closest('.wi-del')) return;
                loadWorld(w.name);
            });
            el.querySelector('.wi-del').addEventListener('click', async e => {
                e.stopPropagation();
                if (!confirm(`Delete world "${w.name}"?`)) return;
                await fetch('/api/world/delete', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name: w.name}) });
                refreshWorlds();
            });
            list.appendChild(el);
        });
    } catch (e) {
        list.innerHTML = '<div class="model-empty">Failed to load.</div>';
    }
}

// ══════════════════════════════════════════════════════════════
// PROPERTIES PANEL
// ══════════════════════════════════════════════════════════════

function updateProps() {
    if (!selected) {
        ['pPosX','pPosY','pPosZ','pRotX','pRotY','pRotZ','pScX','pScY','pScZ'].forEach(id => $(id).value = '');
        return;
    }
    const m = selected.mesh;
    $('pPosX').value = round3(m.position.x); $('pPosY').value = round3(m.position.y); $('pPosZ').value = round3(m.position.z);
    $('pRotX').value = round3(THREE.MathUtils.radToDeg(m.rotation.x));
    $('pRotY').value = round3(THREE.MathUtils.radToDeg(m.rotation.y));
    $('pRotZ').value = round3(THREE.MathUtils.radToDeg(m.rotation.z));
    $('pScX').value = round3(m.scale.x); $('pScY').value = round3(m.scale.y); $('pScZ').value = round3(m.scale.z);
}

function onPropInput(e) {
    if (!selected) return;
    const m = selected.mesh, id = e.target.id, v = parseFloat(e.target.value) || 0;
    if (id === 'pPosX') m.position.x = v;
    else if (id === 'pPosY') m.position.y = v;
    else if (id === 'pPosZ') m.position.z = v;
    else if (id === 'pRotX') m.rotation.x = THREE.MathUtils.degToRad(v);
    else if (id === 'pRotY') m.rotation.y = THREE.MathUtils.degToRad(v);
    else if (id === 'pRotZ') m.rotation.z = THREE.MathUtils.degToRad(v);
    else if (id === 'pScX') { m.scale.x = v; if ($('pScLock').checked) { m.scale.y = v; m.scale.z = v; $('pScY').value = round3(v); $('pScZ').value = round3(v); } }
    else if (id === 'pScY') { m.scale.y = v; if ($('pScLock').checked) { m.scale.x = v; m.scale.z = v; $('pScX').value = round3(v); $('pScZ').value = round3(v); } }
    else if (id === 'pScZ') { m.scale.z = v; if ($('pScLock').checked) { m.scale.x = v; m.scale.y = v; $('pScX').value = round3(v); $('pScY').value = round3(v); } }
}

// ══════════════════════════════════════════════════════════════
// SCENE TREE
// ══════════════════════════════════════════════════════════════

function updateTree() {
    const tree = $('sceneTree');
    if (!objects.length) { tree.innerHTML = '<div class="tree-empty">Empty world</div>'; return; }
    tree.innerHTML = '';
    objects.forEach(wo => {
        const el = document.createElement('div');
        el.className = 'tree-item' + (selected === wo ? ' sel' : '');
        const icon = wo.type === 'glb' ? '📦' : wo.type === 'terrain' ? '🏔' : '◆';
        el.innerHTML = `<span class="ti-icon">${icon}</span><span class="ti-name">${wo.name}</span><button class="ti-vis ${wo.mesh.visible?'':'off'}" title="Toggle">👁</button>`;
        el.addEventListener('click', e => { if (!e.target.closest('.ti-vis')) select(wo); });
        el.querySelector('.ti-vis').addEventListener('click', () => { wo.mesh.visible = !wo.mesh.visible; updateTree(); });
        tree.appendChild(el);
    });
}

function updateHud() {
    $('hudObjs').textContent = `${objects.length} objects`;
    let tris = 0;
    objects.forEach(wo => wo.mesh.traverse(ch => {
        if (ch.isMesh && ch.geometry) {
            const idx = ch.geometry.index;
            const pos = ch.geometry.attributes?.position;
            tris += idx ? idx.count / 3 : (pos?.count || 0) / 3;
        }
    }));
    $('hudTris').textContent = fmtNum(Math.round(tris)) + ' tris';
}

// ══════════════════════════════════════════════════════════════
// BAKE VIEW
// ══════════════════════════════════════════════════════════════

function bakeView() {
    const res = parseInt($('bakeRes').value);
    const shadowRes = parseInt($('bakeShadowRes').value);
    const transparent = $('bakeTransparent').checked;

    const br = new THREE.WebGLRenderer({ antialias: true, alpha: transparent, preserveDrawingBuffer: true });
    br.outputColorSpace = THREE.SRGBColorSpace;
    br.toneMapping = THREE.ACESFilmicToneMapping;
    br.toneMappingExposure = 1.1;
    br.shadowMap.enabled = true;
    br.shadowMap.type = THREE.PCFSoftShadowMap;
    br.setSize(res, res);
    br.setPixelRatio(1);
    br.setClearColor(transparent ? 0x000000 : scene.background, transparent ? 0 : 1);

    const origShadow = sunLight.shadow.mapSize.clone();
    sunLight.shadow.mapSize.set(shadowRes, shadowRes);
    sunLight.shadow.map?.dispose(); sunLight.shadow.map = null;

    const gv = grid.visible, tv = gizmo.visible;
    grid.visible = false; gizmo.visible = false;
    ground.material.opacity = 0.45;

    const bc = camera.clone();
    bc.aspect = 1; bc.updateProjectionMatrix();

    br.render(scene, bc);

    const url = br.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url; a.download = `${worldName}_bake.png`; a.click();

    grid.visible = gv; gizmo.visible = tv;
    ground.material.opacity = 0.3;
    sunLight.shadow.mapSize.copy(origShadow);
    sunLight.shadow.map?.dispose(); sunLight.shadow.map = null;
    br.dispose();

    $('bakeModal').classList.remove('open');
    showToast('🔥 Baked!');
}

// ══════════════════════════════════════════════════════════════
// EXPORT GLB
// ══════════════════════════════════════════════════════════════

function exportScene() {
    if (!objects.length) { showToast('Nothing to export'); return; }
    const group = new THREE.Group();
    objects.forEach(wo => { if (wo.mesh.visible) group.add(wo.mesh.clone()); });

    const exporter = new GLTFExporter();
    exporter.parse(group, result => {
        const blob = new Blob([result], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${worldName}.glb`; a.click();
        URL.revokeObjectURL(url);
        showToast('↓ Exported GLB');
    }, err => showToast('Export failed'), { binary: true });
}

// ══════════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════════

function showToast(msg) {
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        toast.style.cssText = 'position:fixed;bottom:16px;left:50%;transform:translateX(-50%);padding:8px 18px;background:#18181B;border:1px solid #E8A917;border-radius:6px;font-family:var(--fn-mono);font-size:11px;color:#E8A917;z-index:600;pointer-events:none;opacity:0;transition:opacity .2s';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { toast.style.opacity = '0'; }, 2500);
}

// ══════════════════════════════════════════════════════════════
// EMPTY STATE
// ══════════════════════════════════════════════════════════════

function hideEmpty() { $('emptyState').classList.add('hidden'); }
function showEmpty() { $('emptyState').classList.remove('hidden'); }

// ══════════════════════════════════════════════════════════════
// RENDER LOOP
// ══════════════════════════════════════════════════════════════

function animate() {
    requestAnimationFrame(animate);
    orbit.update();
    renderer.render(scene, camera);

    // Axis
    if (axCam) { axCam.quaternion.copy(camera.quaternion); axRenderer.render(axScene, axCam); }

    // FPS
    frameCnt++;
    const t = clock.getElapsedTime();
    if (t - fpsT >= 0.5) {
        fps = Math.round(frameCnt / (t - fpsT));
        $('hudFps').textContent = fps + ' fps';
        frameCnt = 0; fpsT = t;
    }
}

function onResize() {
    const vp = $('viewport');
    const w = vp.clientWidth || 1, h = vp.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

// ══════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════

init();
animate();
