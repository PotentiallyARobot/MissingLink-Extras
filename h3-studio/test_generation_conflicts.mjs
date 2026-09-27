import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(process.env.H3_SOURCE || new URL('./h3_studio_3.py',import.meta.url),'utf8');
const helper=source.slice(source.indexOf('    function accelerationConflict('),source.indexOf('    function resetNamedLoras('));
function context(motion=0,lightning=0,medium=false){
  const values={motion8:motion,lightning};
  const ctx=vm.createContext({window:{MEDIUM8_PRESET_ACTIVE:medium},submittedSpecialStrength:k=>values[k]});
  vm.runInContext(helper,ctx);return ctx;
}
test('mutually exclusive accelerators are blocked in both directions; OFF and creative LoRAs remain possible',()=>{
  assert.ok(context(0,1).accelerationConflict('motion8',1));
  assert.ok(context(1,0).accelerationConflict('lightning',1));
  assert.ok(context(0,0,true).accelerationConflict('motion8',-0.5));
  const legacy=context(1,0);legacy.window.TAOMATE_PRESET_ACTIVE=true;
  assert.ok(legacy.accelerationConflict());
  assert.equal(context(1,1).accelerationConflict('motion8',0),'');
  assert.equal(context(0,1).accelerationConflict(),'');
  assert.equal(context(1,0).accelerationConflict(),'');
  assert.equal(context(0,0).accelerationConflict(),'');
});
test('slider/number/toggle state update refuses the conflict and restores the edited control',()=>{
  const ctx=context(0,1);
  Object.assign(ctx,{card:{kind:'motion8'},st:{strength:0},slider:{value:'1'},out:{value:'1'},meta:{},sliderValueFor:n=>n,
    toggle:{classList:{toggle(){}},setAttribute(){}}});
  const start=source.indexOf('        const syncState=(n,');
  const end=source.indexOf('        const commitNumber=',start);
  vm.runInContext(source.slice(start,end)+'\nsyncState(1);',ctx);
  assert.equal(ctx.st.strength,0);assert.equal(ctx.slider.value,'0');assert.equal(ctx.out.value,'0');
  assert.match(ctx.meta.textContent,/QUALITY/);
});
test('restored conflicting settings cannot submit or consume quota; validation has its own event',async()=>{
  const ctx=context(1,1),events=[],errors=[];
  Object.assign(ctx,{mlTrack:(name)=>events.push(name),fail:m=>errors.push(m),uiAlert:async()=>{},
    fetch:()=>assert.fail('Conflicting request must not reach network'),
    currentModelMode:()=>assert.fail('Validation must precede form and timeline mutation')});
  const start=source.indexOf('    async function submitGeneration(){');
  const end=source.indexOf("    $('go').onclick=",start);
  vm.runInContext(source.slice(start,end),ctx);
  await ctx.submitGeneration();
  assert.equal(errors.length,1);
  assert.deepEqual(events,['notebook_h3_generate_validation_blocked']);
});
test('backend compatibility guard remains intact',()=>{
  assert.ok(source.includes('if _lightning_on or _medium8_on or _taomate_on:'));
  assert.ok(source.includes('Motion Enhancer and the selected acceleration preset are alternative LoRAs.'));
});
