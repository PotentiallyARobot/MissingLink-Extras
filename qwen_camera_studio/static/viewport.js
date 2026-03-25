// ============================================================
// viewport.js — Three.js 3D camera control viewport
// Draggable handles for azimuth (green), elevation (pink), distance (orange)
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
    { label: "close-up",    dist: 2.5 },
    { label: "medium shot",  dist: 4.0 },
    { label: "wide shot",    dist: 6.0 },
];

// State
let azimuthDeg = 0;
let elevationIdx = 1;
let distanceIdx = 1;

// Three.js objects
let scene, camera, renderer, orbitGroup;
let azimuthRing, elevationArc;
let handleAzimuth, handleElevation, handleDistance;
let cameraObj, imagePlane, imageTexture;
let raycaster, mouse;
let dragging = null;
let canvasEl;

function initViewport() {
    canvasEl = document.getElementById('viewport-canvas');
    if (!canvasEl) return;

    const rect = canvasEl.getBoundingClientRect();
    const W = rect.width || 700;
    const H = 360;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e1e24);

    // Camera
    camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
    camera.position.set(6, 5, 8);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Grid
    const grid = new THREE.GridHelper(10, 20, 0x333340, 0x222230);
    grid.position.y = -0.5;
    scene.add(grid);

    // Orbit group (rotates with azimuth)
    orbitGroup = new THREE.Group();
    scene.add(orbitGroup);

    // Azimuth ring (teal)
    const ringGeo = new THREE.TorusGeometry(4, 0.06, 8, 64);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5a0 });
    azimuthRing = new THREE.Mesh(ringGeo, ringMat);
    azimuthRing.rotation.x = Math.PI / 2;
    scene.add(azimuthRing);

    // Elevation arc (pink)
    const arcCurve = new THREE.EllipseCurve(0, 0, 4, 4, -Math.PI/2, Math.PI/2, false, 0);
    const arcPoints = arcCurve.getPoints(50);
    const arcGeo = new THREE.BufferGeometry().setFromPoints(
        arcPoints.map(p => new THREE.Vector3(0, p.y, -p.x))
    );
    const arcMat = new THREE.LineBasicMaterial({ color: 0xff6eb4, linewidth: 2 });
    elevationArc = new THREE.Line(arcGeo, arcMat);
    orbitGroup.add(elevationArc);

    // Handles
    const sphereGeo = new THREE.SphereGeometry(0.22, 16, 16);

    handleAzimuth = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x00e5a0 }));
    handleAzimuth.userData.type = 'azimuth';
    scene.add(handleAzimuth);

    handleElevation = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0xff6eb4 }));
    handleElevation.userData.type = 'elevation';
    orbitGroup.add(handleElevation);

    handleDistance = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0xffd426 }));
    handleDistance.userData.type = 'distance';
    orbitGroup.add(handleDistance);

    // Camera object (small box)
    const camGeo = new THREE.BoxGeometry(0.4, 0.3, 0.5);
    const camMat = new THREE.MeshBasicMaterial({ color: 0x444460 });
    cameraObj = new THREE.Mesh(camGeo, camMat);
    orbitGroup.add(cameraObj);

    // Image plane at center
    const planeGeo = new THREE.PlaneGeometry(1.8, 1.8);
    const planeMat = new THREE.MeshBasicMaterial({ color: 0x888888, side: THREE.DoubleSide });
    imagePlane = new THREE.Mesh(planeGeo, planeMat);
    imagePlane.position.set(0, 0.5, 0);
    scene.add(imagePlane);

    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Events
    canvasEl.addEventListener('mousedown', onMouseDown);
    canvasEl.addEventListener('mousemove', onMouseMove);
    canvasEl.addEventListener('mouseup', onMouseUp);
    canvasEl.addEventListener('mouseleave', onMouseUp);

    // Touch support
    canvasEl.addEventListener('touchstart', onTouchStart, { passive: false });
    canvasEl.addEventListener('touchmove', onTouchMove, { passive: false });
    canvasEl.addEventListener('touchend', onMouseUp);

    updatePositions();
    animate();
}

function setViewportImage(url) {
    const loader = new THREE.TextureLoader();
    loader.load(url, (tex) => {
        imageTexture = tex;
        const aspect = tex.image.width / tex.image.height;
        const h = 1.8;
        const w = h * aspect;
        imagePlane.geometry.dispose();
        imagePlane.geometry = new THREE.PlaneGeometry(w, h);
        imagePlane.material.map = tex;
        imagePlane.material.color.set(0xffffff);
        imagePlane.material.needsUpdate = true;
    });
}

function updatePositions() {
    const azRad = THREE.MathUtils.degToRad(azimuthDeg);
    const elev = ELEVATION_MAP[elevationIdx];
    const dist = DISTANCE_MAP[distanceIdx];

    const elevRad = THREE.MathUtils.degToRad(elev.angle);
    const r = dist.dist;

    // Camera position
    const cx = r * Math.cos(elevRad) * Math.sin(azRad);
    const cy = r * Math.sin(elevRad);
    const cz = r * Math.cos(elevRad) * Math.cos(azRad);

    cameraObj.position.set(cx, cy + 0.5, cz);
    cameraObj.lookAt(0, 0.5, 0);

    // Azimuth handle — on the ring at current azimuth
    handleAzimuth.position.set(
        4 * Math.sin(azRad),
        -0.5,
        4 * Math.cos(azRad)
    );

    // Elevation handle — on the arc
    const arcR = 4;
    handleElevation.position.set(0, arcR * Math.sin(elevRad) + 0.5, -arcR * Math.cos(elevRad));

    // Distance handle — between camera and center
    handleDistance.position.copy(cameraObj.position).multiplyScalar(0.5);
    handleDistance.position.y += 0.5;

    // Rotate orbit group for arc orientation
    orbitGroup.rotation.y = azRad;

    // Ray from camera to image
    // (visual line)
    updatePromptDisplay();
}

function updatePromptDisplay() {
    const azLabel = snapToAzimuth(azimuthDeg);
    const elLabel = ELEVATION_MAP[elevationIdx].label;
    const distLabel = DISTANCE_MAP[distanceIdx].label;

    const prompt = `<sks> ${azLabel} ${elLabel} ${distLabel}`;
    const el = document.getElementById('prompt-display');
    if (el) el.textContent = prompt;

    // Sync dropdowns
    const selAz = document.getElementById('sel-azimuth');
    if (selAz) selAz.value = azLabel;
    const selEl = document.getElementById('sel-elevation');
    if (selEl) selEl.value = elLabel;
    const selDist = document.getElementById('sel-distance');
    if (selDist) selDist.value = distLabel;
}

function snapToAzimuth(deg) {
    deg = ((deg % 360) + 360) % 360;
    let best = AZIMUTH_MAP[0];
    let bestDiff = 999;
    for (const a of AZIMUTH_MAP) {
        let diff = Math.abs(a.deg - deg);
        if (diff > 180) diff = 360 - diff;
        if (diff < bestDiff) { bestDiff = diff; best = a; }
    }
    return best.label;
}

function getMousePos(e) {
    const rect = canvasEl.getBoundingClientRect();
    return {
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
    };
}

function onMouseDown(e) {
    const pos = getMousePos(e);
    mouse.set(pos.x, pos.y);
    raycaster.setFromCamera(mouse, camera);

    const hits = raycaster.intersectObjects([handleAzimuth, handleElevation, handleDistance]);
    if (hits.length > 0) {
        dragging = hits[0].object.userData.type;
        canvasEl.style.cursor = 'grabbing';
        e.preventDefault();
    }
}

function onMouseMove(e) {
    if (!dragging) return;
    const pos = getMousePos(e);

    if (dragging === 'azimuth') {
        // Map mouse X to azimuth angle
        azimuthDeg = ((pos.x + 1) / 2 * 360) % 360;
        // Snap to nearest 45°
        azimuthDeg = Math.round(azimuthDeg / 45) * 45;
    } else if (dragging === 'elevation') {
        // Map mouse Y to elevation index
        const idx = Math.round((1 - (pos.y + 1) / 2) * (ELEVATION_MAP.length - 1));
        elevationIdx = Math.max(0, Math.min(ELEVATION_MAP.length - 1, idx));
    } else if (dragging === 'distance') {
        // Map mouse position to distance
        const idx = Math.round(((pos.x + 1) / 2) * (DISTANCE_MAP.length - 1));
        distanceIdx = Math.max(0, Math.min(DISTANCE_MAP.length - 1, idx));
    }

    updatePositions();
}

function onMouseUp() {
    dragging = null;
    if (canvasEl) canvasEl.style.cursor = 'grab';
}

function onTouchStart(e) {
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        onMouseDown({ clientX: touch.clientX, clientY: touch.clientY, preventDefault: () => e.preventDefault() });
    }
}

function onTouchMove(e) {
    if (e.touches.length === 1 && dragging) {
        e.preventDefault();
        const touch = e.touches[0];
        onMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
    }
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// Sync from dropdowns → viewport
function syncFromDropdowns() {
    const azVal = document.getElementById('sel-azimuth').value;
    const match = AZIMUTH_MAP.find(a => a.label === azVal);
    if (match) azimuthDeg = match.deg;

    const elVal = document.getElementById('sel-elevation').value;
    const elIdx = ELEVATION_MAP.findIndex(e => e.label === elVal);
    if (elIdx >= 0) elevationIdx = elIdx;

    const distVal = document.getElementById('sel-distance').value;
    const distIdx = DISTANCE_MAP.findIndex(d => d.label === distVal);
    if (distIdx >= 0) distanceIdx = distIdx;

    updatePositions();
}

// Resize handler
window.addEventListener('resize', () => {
    if (!canvasEl || !renderer) return;
    const rect = canvasEl.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 360;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
});
