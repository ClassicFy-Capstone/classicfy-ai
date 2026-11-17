const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Choice = require('./study.js');

function fixture() {
  return {definition:{id:'case',candidates:['b','c']}, result:{pack_id:'pack',case_id:'case',
    distance_reference:'A',near_clip:'b',far_clip:'c',candidates:[
      {clip:'b',cosine_distance_to_anchor:.1},{clip:'c',cosine_distance_to_anchor:.7}]}};
}
test('X/Y swaps preserve the selected performance and agreement, with no fixed answer side', () => {
  const {definition,result} = fixture();
  const first=Choice.makeOrder(definition.candidates,0), second=Choice.makeOrder(definition.candidates,1);
  assert.deepEqual(first,['b','c']); assert.deepEqual(second,['c','b']);
  const x=Choice.evaluate('pack',definition,first,'X',result), y=Choice.evaluate('pack',definition,second,'Y',result);
  assert.equal(x.chosen_clip,y.chosen_clip); assert.equal(x.outcome,'same'); assert.equal(y.outcome,'same');
  assert.equal(Choice.evaluate('pack',definition,first,'Y',result).outcome,'opposite');
  assert.equal(Choice.evaluate('pack',definition,second,'X',result).outcome,'opposite');
});
test('ties and uncertainty are excluded from the decisive denominator', () => {
  const {definition,result}=fixture();
  const responses=['X','Y','tie','unsure'].map(choice=>Choice.evaluate('pack',definition,['b','c'],choice,result));
  assert.deepEqual(Choice.summarize(responses),{same:1,opposite:1,tie:1,unsure:1,decisive:2,agreement_fraction:.5});
  assert.equal(Choice.summarize(responses.slice(2)).agreement_fraction,null);
  assert.equal(Choice.summarize([]).agreement_fraction,null);
});
test('stale results, wrong identities, wrong reference and reversed distance labels are rejected', () => {
  const {definition,result}=fixture();
  const score=value=>Choice.evaluate('pack',definition,['b','c'],'X',value);
  assert.throws(()=>score({...result,pack_id:'other'}));
  assert.throws(()=>score({...result,case_id:'other'}));
  assert.throws(()=>score({...result,distance_reference:'B'}));
  assert.throws(()=>score({...result,near_clip:'c',far_clip:'b'}));
  assert.throws(()=>Choice.evaluate('pack',definition,['b','b'],'X',result));
  assert.throws(()=>Choice.evaluate('pack',definition,['b','c'],'',result));
});
test('generated public playlist and per-case answer files agree in both possible X/Y orders', () => {
  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);
  const data=JSON.parse(JSON.stringify(context.window.LISTENING_DATA));
  for(const definition of data.cases){
    const result=JSON.parse(fs.readFileSync(path.join(__dirname,definition.result_url),'utf8'));
    for(const coin of [0,1]){
      const order=Choice.makeOrder(definition.candidates,coin);
      const label=order[0]===result.near_clip?'X':'Y';
      assert.equal(Choice.evaluate(data.id,definition,order,label,result).outcome,'same');
      assert.equal(Choice.evaluate(data.id,definition,order,label==='X'?'Y':'X',result).outcome,'opposite');
    }
  }
});
test('source listening uses the original upload or its verified album-track replacement', () => {
  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);
  for(const clip of Object.values(context.window.LISTENING_DATA.clips)){
    const url=new URL(Choice.watchUrl(clip,clip.start_seconds));
    assert.equal(url.origin,'https://www.youtube.com');
    assert.equal(url.pathname,'/watch');
    assert.equal(url.searchParams.get('v'),clip.video_id==='0QIgscF9ezY'?'V4bTHofOXgE':clip.video_id);
    assert.equal(url.searchParams.get('t'),`${Math.floor(clip.start_seconds)}s`);
  }
});
test('official replacement preserves recording identity, source data, X/Y order and model result', () => {
  const {definition,result}=fixture();
  const data={id:'pack',cases:[definition],clips:{
    b:{video_id:'0QIgscF9ezY',youtube_url:'https://www.youtube.com/watch?v=0QIgscF9ezY',start_seconds:12.75},
    c:{video_id:'BD2nJNAKB4c',youtube_url:'https://www.youtube.com/watch?v=BD2nJNAKB4c',start_seconds:15}}};
  const before=JSON.stringify(data);
  for(const coin of [0,1]){
    const order=Choice.makeOrder(definition.candidates,coin);
    for(const id of order)Choice.watchUrl(data.clips[id],data.clips[id].start_seconds);
    const label=order[0]===result.near_clip?'X':'Y';
    assert.equal(Choice.evaluate(data.id,definition,order,label,result).outcome,'same');
  }
  assert.equal(JSON.stringify(data),before);
  const clip=Object.values(data.clips).find(c=>c.video_id==='0QIgscF9ezY');
  assert.equal(Choice.sourceVideoId(clip),'V4bTHofOXgE');
  assert.equal(Choice.watchUrl(clip,12.75),'https://www.youtube.com/watch?v=V4bTHofOXgE&t=12s');
});
test('source listening rejects another recording, another host, or invalid excerpt times', () => {
  const clip={video_id:'BD2nJNAKB4c',youtube_url:'https://www.youtube.com/watch?v=BD2nJNAKB4c'};
  for(const start of [-1,NaN,Infinity,undefined])assert.throws(()=>Choice.watchUrl(clip,start));
  assert.throws(()=>Choice.watchUrl({...clip,youtube_url:'https://www.youtube.com/watch?v=JuFd49TCYdE'},0));
  assert.throws(()=>Choice.watchUrl({...clip,youtube_url:'https://example.com/watch?v=BD2nJNAKB4c'},0));
  assert.throws(()=>Choice.watchUrl({...clip,video_id:'invalid'},0));
});
test('additive pack migration preserves answers, order, notes and timing without changing the previous save', () => {
  const data={id:'new',compatible_pack_ids:['old'],cases:[{id:'case',anchor:'a',candidates:['b','c']}],clips:{a:{},b:{},c:{}}};
  const c=data.cases[0],order=Choice.makeOrder(c.candidates,1),oldId=data.compatible_pack_ids[0];
  const previous={pack_id:oldId,is_test:true,orders:{[c.id]:order},responses:{[c.id]:{case_id:c.id,choice:'Y',memo:'listener note',xy_order:[...order]}},drafts:{},times:{[c.anchor]:[3,55]}};
  const before=JSON.stringify(previous),result=Choice.migrateSaved(previous,data,'2026-11-04T00:00:00Z');
  assert.equal(result.pack_id,data.id);assert.equal(result.is_test,true);
  assert.deepEqual(result.orders,previous.orders);assert.deepEqual(result.times,previous.times);
  assert.equal(result.responses[c.id].memo,'listener note');assert.equal(result.responses[c.id].choice,'Y');
  assert.equal(result.responses[c.id].submitted_pack_id,oldId);
  assert.equal(Object.keys(result.responses).length,1);assert.equal(Object.keys(result.orders).length,1);
  assert.equal(JSON.stringify(previous),before);
});
test('migration rejects unapproved packs or changed candidate mappings', () => {
  const data={id:'new',compatible_pack_ids:['old'],cases:[{id:'case',candidates:['b','c']}],clips:{b:{},c:{}}};
  const saved={pack_id:'old',orders:{case:['b','c']},responses:{case:{case_id:'case',choice:'X',xy_order:['b','c']}},drafts:{},times:{}};
  assert.equal(Choice.migrateSaved({...saved,pack_id:'unrelated'},data,''),null);
  assert.equal(Choice.migrateSaved({...saved,orders:{case:['b','other']}},data,''),null);
  assert.equal(Choice.migrateSaved({...saved,responses:{case:{...saved.responses.case,xy_order:['c','b']}}},data,''),null);
});
test('replaced questions are archived without becoming answers to the new questions', () => {
  const data={id:'new',compatible_pack_ids:['old'],migration_sources:[{pack_id:'old',case_ids:['keep']}],cases:[{id:'keep',candidates:['b','c']},{id:'new-question',candidates:['f','g']}],clips:{b:{},c:{},f:{},g:{}}};
  const previous={pack_id:'old',orders:{keep:['c','b'],removed:['d','e']},responses:{keep:{case_id:'keep',choice:'X',xy_order:['c','b'],memo:'keep'},removed:{case_id:'removed',choice:'Y',xy_order:['d','e'],memo:'old answer',outcome:'same'}},drafts:{removed:{memo:'old draft'}},times:{d:[1,30],b:[2,40]}};
  const before=JSON.stringify(previous),result=Choice.migrateSaved(previous,data,'now');
  assert.deepEqual(Object.keys(result.responses),['keep']);assert.deepEqual(result.orders,{keep:['c','b']});
  assert.deepEqual(result.times,{b:[2,40]});assert.equal(result.responses['new-question'],undefined);
  assert.equal(result.archived_packs.old.responses.removed.memo,'old answer');
  assert.deepEqual(result.archived_packs.old.times.d,[1,30]);
  assert.equal(Choice.summarize(Object.values(result.responses)).decisive,0);
  assert.equal(JSON.stringify(previous),before);
});
test('all ten selected-theme questions use verified excerpt playback and the requested era quotas', () => {
  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);
  const data=context.window.LISTENING_DATA;
  const audit=JSON.parse(fs.readFileSync(path.join(__dirname,'../../analysis/ae_theme_listening/selected_playback_checks.json'),'utf8'));
  assert.equal(data.cases.length,10);assert.equal(new Set(data.cases.map(c=>c.id)).size,10);
  assert.equal(data.epoch,14);
  assert.deepEqual(JSON.parse(JSON.stringify(data.era_quotas)),{'고전·바로크':4,'낭만':4,'인상주의':2});
  assert.equal(data.compatible_pack_ids.length,0);
  for(const clip of Object.values(data.clips)){
    const check=audit.checks[Choice.sourceVideoId(clip)];
    assert.equal(check.status,'playing');assert.equal(check.video_id_matches,true);
    assert.ok(check.observed_end-check.observed_start>=.5);
    assert.ok(check.duration>=clip.end_seconds);
    assert.equal(check.start,clip.start_seconds);assert.equal(check.end,clip.end_seconds);
  }
});
