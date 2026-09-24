import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const file=readFileSync(new URL('./h3_studio_3.py',import.meta.url),'utf8');
const source=file.slice(file.indexOf('    // Pause network polling after an auth rejection;'),file.indexOf('    const ML_UI_TELEMETRY_QUEUE=[];'));

test('401 pauses all background requests; explicit recovery does not replay a generation',async()=>{
  const calls=[],panels=[];
  let allow=false,reloaded=0;
  const window={location:{href:'https://runtime.example/',origin:'https://runtime.example',reload(){reloaded++}},
    fetch:async(url)=>{calls.push(url);return Response.json({ok:allow}, {status:allow?200:401})}};
  const document={getElementById:id=>panels.find(p=>p.id===id),body:{prepend:p=>panels.push(p)},
    createElement:()=>({setAttribute(){},style:{},append(...children){this.children=children}})};
  vm.runInNewContext(source,{window,document,URL,AbortSignal});
  await window.fetch('/api/generate',{method:'POST'});
  for(let i=0;i<100;i++)await assert.rejects(window.fetch('/api/queue'),/access paused/);
  assert.equal(calls.length,1);assert.equal(panels.length,1);
  const retry=panels[0].children[1];
  await retry.onclick();assert.equal(reloaded,0);assert.equal(retry.disabled,false);
  allow=true;await retry.onclick();assert.equal(reloaded,1);
  assert.deepEqual(calls,['/api/generate','/api/ml/access','/api/ml/access']);
});
