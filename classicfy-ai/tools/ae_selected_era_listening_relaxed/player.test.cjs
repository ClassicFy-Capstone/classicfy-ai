const test=require('node:test'),assert=require('node:assert/strict');
const Playback=require('./player.js');
function fixture(repeat=false){
  let raw,tick,cancelled=false;const statuses=[],errors=[],plays=[];
  class Player {
    constructor(element,options){this.options=options;this.position=0;this.state=5;this.loads=[];this.cues=[];this.pauses=0;raw=this;}
    cueVideoById(range){this.cues.push(range);this.position=range.startSeconds;}
    loadVideoById(range){this.loads.push(range);this.position=range.startSeconds;}
    playVideo(){this.state=1;}
    pauseVideo(){this.pauses++;this.state=2;}
    getCurrentTime(){return this.position;}
    getPlayerState(){return this.state;}
    destroy(){this.destroyed=true;}
    emit(state){this.state=state;this.options.events.onStateChange({data:state});}
  }
  const controller=Playback.createPlayer({api:{Player},element:{},videoId:'BD2nJNAKB4c',start:1.25,end:8.75,origin:'http://127.0.0.1:8766',repeat:()=>repeat,onStatus:s=>statuses.push(s),onPlay:()=>plays.push(true),onError:(...args)=>errors.push(args),schedule:f=>{tick=f;return 42;},cancel:id=>{assert.equal(id,42);cancelled=true;}});
  return {controller,get raw(){return raw;},tick:()=>tick(),statuses,errors,plays,cancelled:()=>cancelled};
}
test('official embed hides built-in controls and preserves exact start/end without extension fragments',()=>{
  const f=fixture();assert.equal(f.raw.options.playerVars.controls,0);assert.equal(f.raw.options.playerVars.origin,'http://127.0.0.1:8766');
  f.controller.play();assert.equal(f.raw.loads.length,0);f.raw.options.events.onReady();
  assert.deepEqual(f.raw.loads,[{videoId:'BD2nJNAKB4c',startSeconds:1.25,endSeconds:8.75}]);f.controller.destroy();
});
test('end boundary stops playback when repetition is off',()=>{
  const f=fixture();f.controller.play();f.raw.options.events.onReady();f.raw.emit(1);f.raw.position=8.76;f.tick();
  assert.equal(f.raw.pauses,1);assert.equal(f.controller.getStatus(),'ended');assert.equal(f.raw.cues.at(-1).startSeconds,1.25);
  const loads=f.raw.loads.length;f.raw.emit(0);f.raw.emit(2);assert.equal(f.raw.loads.length,loads);assert.equal(f.controller.getStatus(),'ended');f.controller.destroy();
});
test('repetition reloads bounded excerpts and duplicate end signals cannot double restart',()=>{
  const f=fixture(true);f.controller.play();f.raw.options.events.onReady();f.raw.emit(1);f.raw.position=8.75;f.tick();f.raw.emit(0);
  assert.equal(f.raw.loads.length,2);assert.deepEqual(f.raw.loads[1],f.raw.loads[0]);
  f.raw.emit(1);f.raw.emit(0);assert.equal(f.raw.loads.length,3);f.controller.destroy();
});
test('manual pause, edited ranges and disposal stop pending repetition',()=>{
  const f=fixture(true);f.controller.play();f.raw.options.events.onReady();f.raw.emit(1);f.controller.pause();f.raw.position=10;f.tick();f.raw.emit(0);assert.equal(f.raw.loads.length,1);
  f.controller.setRange(2.5,6.5);assert.deepEqual(f.raw.cues.at(-1),{videoId:'BD2nJNAKB4c',startSeconds:2.5,endSeconds:6.5});
  f.controller.play();assert.equal(f.raw.loads.at(-1).endSeconds,6.5);
  f.controller.destroy();f.raw.emit(1);f.raw.emit(0);f.tick();assert.equal(f.raw.destroyed,true);assert.equal(f.cancelled(),true);assert.equal(f.raw.loads.length,2);
});
test('embed denial and browser autoplay blocks are explicit and do not claim playback',()=>{
  const f=fixture();f.controller.play();f.raw.options.events.onReady();f.raw.options.events.onError({data:150});f.tick();
  assert.equal(f.controller.getStatus(),'error');assert.equal(f.errors[0][0],150);assert.equal(f.plays.length,0);assert.match(f.errors[0][1],/외부 재생/);f.controller.destroy();
  const blocked=fixture();blocked.raw.options.events.onReady();blocked.raw.options.events.onAutoplayBlocked();assert.equal(blocked.controller.getStatus(),'autoplay-blocked');assert.equal(blocked.plays.length,0);blocked.controller.destroy();
});
test('invalid ranges and video identities are rejected before constructing a player',()=>{
  for(const [id,start,end] of [['bad',0,1],['BD2nJNAKB4c',-1,1],['BD2nJNAKB4c',1,1],['BD2nJNAKB4c',0,Infinity]])assert.throws(()=>Playback.validateRange(id,start,end));
});
