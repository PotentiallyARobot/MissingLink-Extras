// viewport.js — Free-drag 3D camera, snaps on release
const AZ=[{d:0,l:"front view"},{d:45,l:"front-right quarter view"},{d:90,l:"right side view"},
  {d:135,l:"back-right quarter view"},{d:180,l:"back view"},{d:225,l:"back-left quarter view"},
  {d:270,l:"left side view"},{d:315,l:"front-left quarter view"}];
const EL=[{l:"low-angle shot",a:-30},{l:"eye-level shot",a:0},{l:"elevated shot",a:20},{l:"high-angle shot",a:45}];
const DI=[{l:"close-up",r:2.2},{l:"medium shot",r:3.8},{l:"wide shot",r:5.5}];
const R=4,CY=.6;

let azDeg=0,elDeg=0,distR=3.8; // continuous values during drag
let snappedAz=0,snappedEl=1,snappedDist=1; // indices after snap
let scene,cam,ren,canv;
let hAz,hEl,hDist,camMdl,imgPlane,ray,elArc;
let rc,ms;
let drag=null,dragStart={x:0,y:0},dragAzStart=0,dragElStart=0,dragDistStart=0;
const GP=new THREE.Plane(new THREE.Vector3(0,1,0),0);

function initViewport(){
    canv=document.getElementById('viewport-canvas'); if(!canv) return;
    const W=canv.clientWidth||700,H=canv.clientHeight||340;
    scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d0d0f);
    cam=new THREE.PerspectiveCamera(38,W/H,.1,100);
    cam.position.set(7,5.5,7); cam.lookAt(0,CY,0);
    ren=new THREE.WebGLRenderer({canvas:canv,antialias:true});
    ren.setSize(W,H); ren.setPixelRatio(Math.min(devicePixelRatio,2));

    scene.add(new THREE.GridHelper(12,24,0x222226,0x18181c));

    // Ring
    const ring=new THREE.Mesh(new THREE.TorusGeometry(R,.04,8,80),new THREE.MeshBasicMaterial({color:0xE8A917,transparent:true,opacity:.5}));
    ring.rotation.x=Math.PI/2; scene.add(ring);

    // Elev arc
    elArc=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xE8A917,transparent:true,opacity:.3}));
    scene.add(elArc);

    // Handles
    const hg=new THREE.SphereGeometry(.3,16,16);
    hAz=new THREE.Mesh(hg,new THREE.MeshBasicMaterial({color:0x22C55E})); hAz.userData.h='az'; scene.add(hAz);
    hEl=new THREE.Mesh(hg,new THREE.MeshBasicMaterial({color:0xE8A917})); hEl.userData.h='el'; scene.add(hEl);
    hDist=new THREE.Mesh(hg,new THREE.MeshBasicMaterial({color:0xEF4444})); hDist.userData.h='dist'; scene.add(hDist);

    // Camera model
    const cg=new THREE.Group();
    cg.add(new THREE.Mesh(new THREE.BoxGeometry(.45,.3,.3),new THREE.MeshBasicMaterial({color:0x3a3a50})));
    const lens=new THREE.Mesh(new THREE.CylinderGeometry(.06,.12,.2,8),new THREE.MeshBasicMaterial({color:0x222233}));
    lens.rotation.x=Math.PI/2; lens.position.z=-.25; cg.add(lens);
    camMdl=cg; scene.add(camMdl);

    // Ray
    ray=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),
        new THREE.LineBasicMaterial({color:0xE8A917,transparent:true,opacity:.35}));
    scene.add(ray);

    // Image plane
    imgPlane=new THREE.Mesh(new THREE.PlaneGeometry(1.6,1.6),
        new THREE.MeshBasicMaterial({color:0x333340,side:THREE.DoubleSide,transparent:true,opacity:.85}));
    imgPlane.position.set(0,CY,0); scene.add(imgPlane);

    rc=new THREE.Raycaster(); ms=new THREE.Vector2();

    canv.addEventListener('pointerdown',onDown);
    canv.addEventListener('pointermove',onMove);
    canv.addEventListener('pointerup',onUp);
    canv.addEventListener('pointerleave',onUp);

    updateScene(); requestAnimationFrame(loop);
}

function setViewportImage(url){
    new THREE.TextureLoader().load(url,tex=>{
        const ar=tex.image.width/tex.image.height,h=1.6,w=Math.min(h*ar,2.4);
        imgPlane.geometry.dispose(); imgPlane.geometry=new THREE.PlaneGeometry(w,h);
        imgPlane.material.map=tex; imgPlane.material.color.set(0xffffff);
        imgPlane.material.opacity=1; imgPlane.material.needsUpdate=true;
    });
}

function updateScene(){
    const azR=THREE.MathUtils.degToRad(azDeg), elR=THREE.MathUtils.degToRad(elDeg);
    const cx=distR*Math.cos(elR)*Math.sin(azR), cy=distR*Math.sin(elR)+CY, cz=distR*Math.cos(elR)*Math.cos(azR);
    camMdl.position.set(cx,cy,cz); camMdl.lookAt(0,CY,0);
    hAz.position.set(R*Math.sin(azR),0,R*Math.cos(azR));

    // Arc in azimuth plane
    const pts=[];
    for(let a=-40;a<=55;a+=3){const r=THREE.MathUtils.degToRad(a);
        pts.push(new THREE.Vector3(R*Math.cos(r)*Math.sin(azR),R*Math.sin(r)+CY,R*Math.cos(r)*Math.cos(azR)));}
    elArc.geometry.dispose(); elArc.geometry=new THREE.BufferGeometry().setFromPoints(pts);

    hEl.position.set(R*Math.cos(elR)*Math.sin(azR),R*Math.sin(elR)+CY,R*Math.cos(elR)*Math.cos(azR));
    hDist.position.set(cx*.55,cy*.55+CY*.45,cz*.55);

    const p=ray.geometry.attributes.position.array;
    p[0]=cx;p[1]=cy;p[2]=cz;p[3]=0;p[4]=CY;p[5]=0;
    ray.geometry.attributes.position.needsUpdate=true;

    updatePromptFromCamera();
}

function updatePromptFromCamera(){
    if(window._promptLocked) return;
    const azL=snapAzLabel(azDeg), elL=snapElLabel(elDeg), dL=snapDistLabel(distR);
    const pi=document.getElementById('promptInput');
    if(pi) pi.value=`<sks> ${azL} ${elL} ${dL}`;
    // Sync dropdowns
    const s1=document.getElementById('selAz'),s2=document.getElementById('selEl'),s3=document.getElementById('selDist');
    if(s1) s1.value=azL; if(s2) s2.value=elL; if(s3) s3.value=dL;
}

function snapAzLabel(d){d=((d%360)+360)%360;let b=AZ[0],bd=999;for(const a of AZ){let df=Math.abs(a.d-d);if(df>180)df=360-df;if(df<bd){bd=df;b=a;}}return b.l;}
function snapElLabel(d){let b=EL[0],bd=999;for(const e of EL){const df=Math.abs(e.a-d);if(df<bd){bd=df;b=e;}}return b.l;}
function snapDistLabel(r){let b=DI[0],bd=999;for(const d of DI){const df=Math.abs(d.r-r);if(df<bd){bd=df;b=d;}}return b.l;}

function snapValues(){
    // Snap to nearest discrete values
    let bd=999,bi=0;AZ.forEach((a,i)=>{let d=Math.abs(((azDeg%360+360)%360)-a.d);if(d>180)d=360-d;if(d<bd){bd=d;bi=i;}});
    azDeg=AZ[bi].d;
    bd=999;bi=0;EL.forEach((e,i)=>{const d=Math.abs(elDeg-e.a);if(d<bd){bd=d;bi=i;}});
    elDeg=EL[bi].a;
    bd=999;bi=0;DI.forEach((d,i)=>{const df=Math.abs(distR-d.r);if(df<bd){bd=df;bi=i;}});
    distR=DI[bi].r;
    updateScene();
}

// ── Pointer events ────────────────────────────────

function ndc(e){
    const r=canv.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1,-((e.clientY-r.top)/r.height)*2+1);
}
function hitGround(n){rc.setFromCamera(n,cam);const v=new THREE.Vector3();return rc.ray.intersectPlane(GP,v)?v:null;}

function onDown(e){
    const n=ndc(e); rc.setFromCamera(n,cam);
    const hits=rc.intersectObjects([hAz,hEl,hDist]);
    if(hits.length){
        drag=hits[0].object.userData.h;
        dragStart={x:e.clientX,y:e.clientY};
        dragAzStart=azDeg; dragElStart=elDeg; dragDistStart=distR;
        canv.setPointerCapture(e.pointerId);
        canv.style.cursor='grabbing'; e.preventDefault();
    }
}

function onMove(e){
    if(!drag){
        const n=ndc(e); rc.setFromCamera(n,cam);
        canv.style.cursor=rc.intersectObjects([hAz,hEl,hDist]).length?'pointer':'grab';
        return;
    }
    const n=ndc(e);
    if(drag==='az'){
        const g=hitGround(n);
        if(g) azDeg=THREE.MathUtils.radToDeg(Math.atan2(g.x,g.z));
    } else if(drag==='el'){
        // Drag UP = higher elevation (positive pitch), drag DOWN = lower
        const dy=e.clientY-dragStart.y;
        elDeg=THREE.MathUtils.clamp(dragElStart-dy*0.3,-35,50);
    } else if(drag==='dist'){
        const g=hitGround(n);
        if(g){const d=Math.sqrt(g.x*g.x+g.z*g.z); distR=THREE.MathUtils.clamp(d*0.9,1.5,7);}
    }
    updateScene();
}

function onUp(e){
    if(drag){
        try{canv.releasePointerCapture(e.pointerId);}catch(_){}
        drag=null; canv.style.cursor='grab';
        snapValues(); // snap to nearest supported pose
    }
}

function loop(){requestAnimationFrame(loop);ren.render(scene,cam);}

function syncFromDropdowns(){
    const m=AZ.find(a=>a.l===document.getElementById('selAz').value); if(m) azDeg=m.d;
    const ei=EL.find(e=>e.l===document.getElementById('selEl').value); if(ei) elDeg=ei.a;
    const di=DI.find(d=>d.l===document.getElementById('selDist').value); if(di) distR=di.r;
    updateScene();
}

window.addEventListener('resize',()=>{
    if(!canv||!ren)return;
    const W=canv.clientWidth,H=canv.clientHeight||340;
    cam.aspect=W/H; cam.updateProjectionMatrix(); ren.setSize(W,H);
});
