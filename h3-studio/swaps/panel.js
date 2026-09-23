'use strict';
const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const mask = document.createElement('canvas'), mctx = mask.getContext('2d', {willReadFrequently:true});
let original = null, originalBlob = null, referenceBlob = null, candidates = [], outputFile = '', outputUrl = '';
let busy = false, erase = false, drawing = false, previous = null, undo = [], hasMask = false, keyReady = false;
const targets = {face:'face', head:'head', body:'person', clothing:'clothing', custom:''};
function status(message, error=false){$('status').textContent=message;$('status').classList.toggle('error',error)}
function controls(){
  for(const id of ['original','reference','kind','target','mask-upload','grow','feather','instruction','quality','paint','erase','brush','candidate'])$(id).disabled=busy;
  $('mask-upload').disabled=busy||!original;$('segment').disabled=busy||!original;
  $('clear').disabled=busy||!original;$('undo').disabled=busy||!undo.length;
  $('generate').disabled=busy||!original||!referenceBlob||!hasMask||!keyReady;
}
function resetResult(){outputFile='';outputUrl='';$('result').hidden=true;$('before').checked=false}
function preview(kind,blob){const img=$(kind+'-preview');if(img.dataset.url)URL.revokeObjectURL(img.dataset.url);delete img.dataset.url;img.removeAttribute('src');img.hidden=!blob;img.parentElement.classList.toggle('loaded',!!blob);if(blob){img.dataset.url=URL.createObjectURL(blob);img.src=img.dataset.url}}
function remember(){undo.push(mctx.getImageData(0,0,mask.width,mask.height));if(undo.length>8)undo.shift()}
function refreshMask(){hasMask=mctx.getImageData(0,0,mask.width,mask.height).data.some((v,i)=>i%4===0&&v>=128);resetResult();controls();render()}
function render(){
  if(!original)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(original,0,0);
  if($('overlay').checked){const layer=document.createElement('canvas');layer.width=mask.width;layer.height=mask.height;const lc=layer.getContext('2d'),data=mctx.getImageData(0,0,mask.width,mask.height);for(let i=0;i<data.data.length;i+=4){const a=data.data[i];data.data[i]=172;data.data[i+1]=130;data.data[i+2]=255;data.data[i+3]=a?110:0}lc.putImageData(data,0,0);ctx.drawImage(layer,0,0)}
}
function blobOf(c){return new Promise(resolve=>c.toBlob(resolve,'image/png'))}
async function imageOf(blob){return createImageBitmap(blob,{imageOrientation:'from-image'})}
async function normalized(file){
  if(!file||!['image/png','image/jpeg','image/webp'].includes(file.type))throw Error('Choose a PNG, JPEG or WebP image.');
  if(file.size>20*1024*1024)throw Error('Each image must be under 20 MB.');
  const img=await imageOf(file);if(img.width*img.height>16000000){img.close();throw Error('Choose an image no larger than 16 megapixels.')}
  const c=document.createElement('canvas');c.width=img.width;c.height=img.height;c.getContext('2d').drawImage(img,0,0);const blob=await blobOf(c);
  if(blob.size>20*1024*1024){img.close();throw Error('Decoded image exceeds 20 MB. Use a smaller image.')}
  return {img,blob};
}
async function configuration(){try{const r=await fetch('/api/swaps/status');if(!r.ok)throw Error('Could not check editor setup');const s=await r.json();keyReady=s.key_available;$('setup').textContent=`Editor: ${s.model}. API key ${keyReady?'ready':'missing'}. SAM ${s.sam_installed?'installed; weights are checked on first use':'not installed; run notebook setup or use a manual mask'}.`;controls()}catch(e){status(e.message,true)}}
async function job(url,form){
  const response=await fetch(url,{method:'POST',body:form});let data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');
  for(;;){await new Promise(r=>setTimeout(r,1500));const r=await fetch('/api/swaps/jobs/'+data.job);if(!r.ok)throw Error('Could not retrieve job status.');const result=await r.json();if(result.status==='error')throw Error(result.error);if(result.status==='done')return result}
}
async function action(task){busy=true;drawing=false;controls();try{await task()}catch(e){status(e.message,true)}finally{busy=false;controls()}}
$('original').onchange=()=>action(async()=>{
  // Clear first so a rejected upload cannot leave a stale mask paired with a new filename.
  if(original)original.close();original=null;originalBlob=null;preview('original',null);preview('mask',null);hasMask=false;undo=[];candidates=[];$('candidate').hidden=$('candidate-label').hidden=true;resetResult();$('canvas').hidden=true;$('empty').hidden=false;
  const loaded=await normalized($('original').files[0]);original=loaded.img;originalBlob=loaded.blob;
  canvas.width=mask.width=original.width;canvas.height=mask.height=original.height;mctx.fillStyle='black';mctx.fillRect(0,0,mask.width,mask.height);
  $('canvas').hidden=false;$('empty').hidden=true;$('dimensions').textContent=`${original.width} × ${original.height} · purple = replace`;
  preview('original',originalBlob);render();status('Select with SAM, paint directly, or upload a mask.');
});
$('reference').onchange=()=>action(async()=>{referenceBlob=null;preview('reference',null);resetResult();const loaded=await normalized($('reference').files[0]);referenceBlob=loaded.blob;loaded.img.close();preview('reference',referenceBlob);status('Replacement reference loaded.')});
$('kind').onchange=()=>{$('target').value=targets[$('kind').value];resetResult()};
$('overlay').onchange=render;$('brush').oninput=()=>{$('brush-value').value=$('brush').value};
function tool(value){erase=value;$('paint').classList.toggle('selected',!value);$('erase').classList.toggle('selected',value);$('paint').setAttribute('aria-pressed',String(!value));$('erase').setAttribute('aria-pressed',String(value))}
$('paint').onclick=()=>tool(false);$('erase').onclick=()=>tool(true);
function point(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}
function stroke(e){const p=point(e);mctx.strokeStyle=mctx.fillStyle=erase?'black':'white';mctx.lineWidth=Number($('brush').value);mctx.lineCap='round';mctx.beginPath();mctx.moveTo(...(previous||p));mctx.lineTo(...p);mctx.stroke();mctx.beginPath();mctx.arc(...p,mctx.lineWidth/2,0,2*Math.PI);mctx.fill();previous=p;render()}
canvas.onpointerdown=e=>{if(busy||!original)return;e.preventDefault();remember();drawing=true;previous=null;canvas.setPointerCapture(e.pointerId);stroke(e)};
canvas.onpointermove=e=>{if(drawing)stroke(e)};
function endStroke(){if(drawing){drawing=false;previous=null;refreshMask()}}
canvas.onpointerup=endStroke;canvas.onpointercancel=endStroke;canvas.onlostpointercapture=endStroke;
$('undo').onclick=()=>{if(!undo.length)return;mctx.putImageData(undo.pop(),0,0);refreshMask()};
$('clear').onclick=()=>{remember();mctx.fillStyle='black';mctx.fillRect(0,0,mask.width,mask.height);refreshMask()};
async function applyMask(blob){const img=await imageOf(blob);try{if(img.width!==mask.width||img.height!==mask.height)throw Error('Mask dimensions must match the original image.');remember();mctx.fillStyle='black';mctx.fillRect(0,0,mask.width,mask.height);mctx.drawImage(img,0,0);const d=mctx.getImageData(0,0,mask.width,mask.height);for(let i=0;i<d.data.length;i+=4){const v=(d.data[i]+d.data[i+1]+d.data[i+2])/3>=128?255:0;d.data[i]=d.data[i+1]=d.data[i+2]=v;d.data[i+3]=255}mctx.putImageData(d,0,0);refreshMask()}finally{img.close()}}
$('mask-upload').onchange=()=>action(async()=>{preview('mask',null);const loaded=await normalized($('mask-upload').files[0]);loaded.img.close();await applyMask(loaded.blob);preview('mask',loaded.blob);status('Mask loaded. White regions will be replaced.')});
$('segment').onclick=()=>action(async()=>{if(!$('target').value.trim())throw Error('Describe the region SAM should select.');status('Finding regions with SAM on CPU. First use downloads the model; this can take several minutes.');const f=new FormData();f.append('original',originalBlob,'original.png');f.append('target',$('target').value.trim());const r=await job('/api/swaps/mask',f);candidates=r.candidates;$('candidate').replaceChildren(...candidates.map((c,i)=>new Option(`Region ${i+1} · ${Math.round(c.score*100)}%`,i)));$('candidate').hidden=$('candidate-label').hidden=false;await chooseCandidate();status(`Found ${candidates.length} region(s). Choose the right one, then refine with the brush.`)});
async function chooseCandidate(){const c=candidates[Number($('candidate').value)];if(c){const raw=Uint8Array.from(atob(c.png),x=>x.charCodeAt(0));await applyMask(new Blob([raw],{type:'image/png'}))}}
$('candidate').onchange=()=>action(chooseCandidate);
$('generate').onclick=()=>action(async()=>{resetResult();status('Generating the replacement. This may take a few minutes…');const f=new FormData();f.append('original',originalBlob,'original.png');f.append('reference',referenceBlob,'reference.png');f.append('mask',await blobOf(mask),'mask.png');for(const id of ['kind','instruction','grow','feather','quality'])f.append(id,$(id).value);const r=await job('/api/swaps/edit',f);outputFile=r.file;outputUrl='/out/'+encodeURIComponent(r.file);$('result-image').src=outputUrl;$('download').href=outputUrl;$('download').download=r.file;$('result').hidden=false;status('Swap ready. Compare the original, then download or send it into H3.');$('result').scrollIntoView({behavior:'smooth',block:'start'})});
let beforeUrl='';$('before').onchange=()=>{if(beforeUrl)URL.revokeObjectURL(beforeUrl);beforeUrl='';if($('before').checked){beforeUrl=URL.createObjectURL(originalBlob);$('result-image').src=beforeUrl}else $('result-image').src=outputUrl};
for(const kind of ['first','last'])$(kind).onclick=()=>{if(outputFile)parent.postMessage({type:'h3-swap-frame',file:outputFile,kind},location.origin)};
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===parent&&e.data?.type==='h3-swap-error')status(e.data.error,true)});
$('refresh').onclick=configuration;configuration();controls();
