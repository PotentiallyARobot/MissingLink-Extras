// ============================================================
// viewport.js — Three.js 3D camera control viewport
// Properly draggable handles for azimuth/elevation/distance
// ============================================================

const AZIMUTH_MAP = [
    { deg: 0,   label: "front view" },
    { deg: 45,  label: "front-right quarter view" },
    { deg: 90,  label: "right side view" },
    { deg: 135, label: "back-right quarter view" },
    { deg: 180, label: "back view" },
    { deg: 225, label: "back-left quarter view" },
    { deg: 270, label: "left side view" },
    { deg: 315, label: "front-left quarter view" },
];

const ELEVATION_MAP = [
    { label: "low-angle shot",  angle: -30 },
    { label: "eye-level shot",  angle: 0 },
    { label: "elevated shot",   angle: 20 },
    { label: "high-angle shot", angle: 45 },
];

const DISTANCE_MAP = [
    { label: "close-up",    dist: 2.2 },
    { label: "medium shot", dist: 3.8 },
    { label: "wide shot",   dist: 5.5 },
];

const RING_R = 4.0;
const CTR_Y = 0.6;

let azimuthDeg = 0;
let elevationIdx = 1;
let distanceIdx = 1;

let scene, viewCam, renderer;
let handleAz, handleEl, handleDist;
let cameraModel, imagePlane, rayLine, elevArc;
let raycaster, mouse;
let dragging = null;
let canvasEl;
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

function initViewport() {
    canvasEl = document.getElementById('viewport-canvas');
    if (!canvasEl) return;

    const W = canvasEl.clientWidth || 700;
    const H = canvasEl.clientHeight || 360;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a22);

    viewCam = new THREE.PerspectiveCamera(38, W / H, 0.1, 100);
    viewCam.position.set(7, 5.5, 7);
    viewCam.lookAt(0, CTR_Y, 0);

    renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Grid
    scene.add(new THREE.GridHelper(12, 24, 0x2a2a35, 0x1f1f28));

    // Azimuth ring
    const ring = new THREE.Mesh(
        new THREE.TorusGeometry(RING_R, 0.045, 8, 80),
        new THREE.MeshBasicMaterial({ color: 0x00e5a0 })
    );
    ring.rotation.x = Math.PI / 2;
    scene.add(ring);

    // Elevation arc (rebuilt on each update to follow azimuth)
    elevArc = new THREE.Line(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0xff6eb4 })
    );
    scene.add(elevArc);

    // Handles — bigger spheres
    const hGeo = new THREE.SphereGeometry(0.3, 16, 16);
    handleAz = new THREE.Mesh(hGeo, new THREE.MeshBasicMaterial({ color: 0x00e5a0 }));
    handleAz.userData.h = 'azimuth';
    scene.add(handleAz);

    handleEl = new THREE.Mesh(hGeo, new THREE.MeshBasicMaterial({ color: 0xff6eb4 }));
    handleEl.userData.h = 'elevation';
    scene.add(handleEl);

    handleDist = new THREE.Mesh(hGeo, new THREE.MeshBasicMaterial({ color: 0xffd426 }));
    handleDist.userData.h = 'distance';
    scene.add(handleDist);

    // Camera model
    const cg = new THREE.Group();
    cg.add(new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 0.35, 0.35),
        new THREE.MeshBasicMaterial({ color: 0x3a3a55 })
    ));
    const lens = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.12, 0.22, 8),
        new THREE.MeshBasicMaterial({ color: 0x222233 })
    );
    lens.rotation.x = Math.PI / 2;
    lens.position.z = -0.28;
    cg.add(lens);
    cameraModel = cg;
    scene.add(cameraModel);

    // Ray line
    rayLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0xff8c42, transparent: true, opacity: 0.5 })
    );
    scene.add(rayLine);

    // Image plane
    imagePlane = new THREE.Mesh(
        new THREE.PlaneGeometry(1.6, 1.6),
        new THREE.MeshBasicMaterial({ color: 0x555566, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })
    );
    imagePlane.position.set(0, CTR_Y, 0);
    scene.add(imagePlane);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    canvasEl.addEventListener('pointerdown', onDown);
    canvasEl.addEventListener('pointermove', onMove);
    canvasEl.addEventListener('pointerup', onUp);
    canvasEl.addEventListener('pointerleave', onUp);

    updateScene();
    requestAnimationFrame(loop);
}

function setViewportImage(url) {
    new THREE.TextureLoader().load(url, tex => {
        const ar = tex.image.width / tex.image.height;
        const h = 1.6, w = Math.min(h * ar, 2.4);
        imagePlane.geometry.dispose();
        imagePlane.geometry = new THREE.PlaneGeometry(w, h);
        imagePlane.material.map = tex;
        imagePlane.material.color.set(0xffffff);
        imagePlane.material.opacity = 1;
        imagePlane.material.needsUpdate = true;
    });
}

function updateScene() {
    const azRad = THREE.MathUtils.degToRad(azimuthDeg);
    const elAng = ELEVATION_MAP[elevationIdx].angle;
    const elRad = THREE.MathUtils.degToRad(elAng);
    const dist = DISTANCE_MAP[distanceIdx].dist;

    // Camera position
    const cx = dist * Math.cos(elRad) * Math.sin(azRad);
    const cy = dist * Math.sin(elRad) + CTR_Y;
    const cz = dist * Math.cos(elRad) * Math.cos(azRad);

    cameraModel.position.set(cx, cy, cz);
    cameraModel.lookAt(0, CTR_Y, 0);

    // Azimuth handle on ring
    handleAz.position.set(RING_R * Math.sin(azRad), 0, RING_R * Math.cos(azRad));

    // Elevation arc — semicircle in the azimuth plane
    const arcPts = [];
    for (let a = -40; a <= 55; a += 3) {
        const r = THREE.MathUtils.degToRad(a);
        arcPts.push(new THREE.Vector3(
            RING_R * Math.cos(r) * Math.sin(azRad),
            RING_R * Math.sin(r) + CTR_Y,
            RING_R * Math.cos(r) * Math.cos(azRad)
        ));
    }
    elevArc.geometry.dispose();
    elevArc.geometry = new THREE.BufferGeometry().setFromPoints(arcPts);

    // Elevation handle on arc
    handleEl.position.set(
        RING_R * Math.cos(elRad) * Math.sin(azRad),
        RING_R * Math.sin(elRad) + CTR_Y,
        RING_R * Math.cos(elRad) * Math.cos(azRad)
    );

    // Distance handle — along the ray at 55%
    handleDist.position.set(cx * 0.55, cy * 0.55 + CTR_Y * 0.45, cz * 0.55);

    // Ray line
    const p = rayLine.geometry.attributes.position.array;
    p[0] = cx; p[1] = cy; p[2] = cz;
    p[3] = 0;  p[4] = CTR_Y; p[5] = 0;
    rayLine.geometry.attributes.position.needsUpdate = true;

    updatePromptDisplay();
}

function updatePromptDisplay() {
    const azL = snapAz(azimuthDeg);
    const elL = ELEVATION_MAP[elevationIdx].label;
    const dL = DISTANCE_MAP[distanceIdx].label;
    const el = document.getElementById('prompt-display');
    if (el) el.textContent = `<sks> ${azL} ${elL} ${dL}`;
    const s1 = document.getElementById('sel-azimuth');
    const s2 = document.getElementById('sel-elevation');
    const s3 = document.getElementById('sel-distance');
    if (s1) s1.value = azL;
    if (s2) s2.value = elL;
    if (s3) s3.value = dL;
}

function snapAz(deg) {
    deg = ((deg % 360) + 360) % 360;
    let best = AZIMUTH_MAP[0], bd = 999;
    for (const a of AZIMUTH_MAP) {
        let d = Math.abs(a.deg - deg);
        if (d > 180) d = 360 - d;
        if (d < bd) { bd = d; best = a; }
    }
    return best.label;
}

// ── Pointer events ────────────────────────────────────────────

function getNDC(e) {
    const r = canvasEl.getBoundingClientRect();
    return new THREE.Vector2(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        -((e.clientY - r.top) / r.height) * 2 + 1
    );
}

function hitGround(ndc) {
    raycaster.setFromCamera(ndc, viewCam);
    const v = new THREE.Vector3();
    return raycaster.ray.intersectPlane(groundPlane, v) ? v : null;
}

function onDown(e) {
    const ndc = getNDC(e);
    raycaster.setFromCamera(ndc, viewCam);
    const hits = raycaster.intersectObjects([handleAz, handleEl, handleDist]);
    if (hits.length) {
        dragging = hits[0].object.userData.h;
        canvasEl.setPointerCapture(e.pointerId);
        canvasEl.style.cursor = 'grabbing';
        e.preventDefault();
    }
}

function onMove(e) {
    if (!dragging) {
        const ndc = getNDC(e);
        raycaster.setFromCamera(ndc, viewCam);
        const h = raycaster.intersectObjects([handleAz, handleEl, handleDist]);
        canvasEl.style.cursor = h.length ? 'pointer' : 'grab';
        return;
    }

    const ndc = getNDC(e);

    if (dragging === 'azimuth') {
        const g = hitGround(ndc);
        if (g) {
            let deg = THREE.MathUtils.radToDeg(Math.atan2(g.x, g.z));
            deg = ((deg % 360) + 360) % 360;
            azimuthDeg = Math.round(deg / 45) * 45;
            if (azimuthDeg >= 360) azimuthDeg = 0;
        }
    } else if (dragging === 'elevation') {
        // Screen Y → elevation index  (top=high angle, bottom=low angle)
        const t = 1 - (ndc.y + 1) / 2; // 0=top 1=bottom
        const idx = Math.round(t * (ELEVATION_MAP.length - 1));
        elevationIdx = Math.max(0, Math.min(ELEVATION_MAP.length - 1, idx));
    } else if (dragging === 'distance') {
        const g = hitGround(ndc);
        if (g) {
            const d = Math.sqrt(g.x * g.x + g.z * g.z);
            if (d < 2.8) distanceIdx = 0;
            else if (d < 4.5) distanceIdx = 1;
            else distanceIdx = 2;
        }
    }

    updateScene();
}

function onUp(e) {
    if (dragging) {
        try { canvasEl.releasePointerCapture(e.pointerId); } catch(_) {}
        dragging = null;
        canvasEl.style.cursor = 'grab';
    }
}

function loop() {
    requestAnimationFrame(loop);
    renderer.render(scene, viewCam);
}

// Dropdown → 3D sync
function syncFromDropdowns() {
    const m = AZIMUTH_MAP.find(a => a.label === document.getElementById('sel-azimuth').value);
    if (m) azimuthDeg = m.deg;
    const ei = ELEVATION_MAP.findIndex(e => e.label === document.getElementById('sel-elevation').value);
    if (ei >= 0) elevationIdx = ei;
    const di = DISTANCE_MAP.findIndex(d => d.label === document.getElementById('sel-distance').value);
    if (di >= 0) distanceIdx = di;
    updateScene();
}

window.addEventListener('resize', () => {
    if (!canvasEl || !renderer) return;
    const W = canvasEl.clientWidth, H = canvasEl.clientHeight || 360;
    viewCam.aspect = W / H;
    viewCam.updateProjectionMatrix();
    renderer.setSize(W, H);
});
