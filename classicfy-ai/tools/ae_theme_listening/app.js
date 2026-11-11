"use strict";
const data = window.LISTENING_DATA, Choice = window.ThemeChoice, Playback = window.ThemePlayer;
// Original watch page also reported unavailable in Safari on 2026-11-04.
// Availability checks apply to the resolved playback source, including verified replacements.
const unavailableVideoIds = new Set(["0QIgscF9ezY"]);
const $ = selector => document.querySelector(selector);
const esc = s => String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clockTime = s => `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,"0")}`;
const isTest = new URLSearchParams(location.search).get("qa") === "1";
const key = `classicfy-theme-choice-v2-${isTest ? "qa-" : ""}${data.id}`;
let saved;
try { saved=JSON.parse(localStorage.getItem(key)||"null"); } catch (_) {}
if(!saved)for(const previous of data.compatible_pack_ids||[]) {
  let candidate;
  try {candidate=Choice.migrateSaved(JSON.parse(localStorage.getItem(`classicfy-theme-choice-v2-${isTest?"qa-":""}${previous}`)||"null"),data,new Date().toISOString());} catch (_) {}
  if(candidate){
    const archives={...(saved?.archived_packs||{}),...(candidate.archived_packs||{})};
    if(!saved||Object.keys(candidate.responses).length>Object.keys(saved.responses).length)saved=candidate;
    if(Object.keys(archives).length)saved.archived_packs=archives;
  }
}
if(!saved || saved.pack_id!==data.id || !saved.orders || !saved.responses || !saved.drafts || !saved.times) saved={schema:"ae-theme-choice-responses-v2",pack_id:data.id,is_test:isTest,created_at:new Date().toISOString(),orders:{},responses:{},drafts:{},times:{}};
for(const c of data.cases) {
  if(!Choice.validOrder(c.candidates,saved.orders[c.id])) {
    if(saved.responses[c.id]) throw new Error("Saved response has an invalid X/Y order");
    saved.orders[c.id]=Choice.makeOrder(c.candidates);
  }
}
saved.source_opens ||= {};
saved.listened ||= {};
let caseIndex=0, submitting=false;
let playerGeneration=0,requestedSlot=null;
const players=new Map(),pendingPlayers=new Map();
const playbackLog={};
const revealed=new Map(), currentCase=()=>data.cases[caseIndex];
const clipIds=()=>[currentCase().anchor,...saved.orders[currentCase().id]];
const slotLabel=i=>["A","X","Y"][i];
const bounds=id=>saved.times[id]||[data.clips[id].start_seconds,data.clips[id].end_seconds];
const timingVisible=()=>Boolean(saved.responses[currentCase().id]||saved.timing_visibility?.[currentCase().id]);
const playbackUrl=id=>Choice.watchUrl(data.clips[id],bounds(id)[0]);
const embedCheck=id=>window.EMBED_CHECKS?.checks[Choice.sourceVideoId(data.clips[id])];
const hasUnavailableSource=c=>[c.anchor,...c.candidates].some(id=>unavailableVideoIds.has(Choice.sourceVideoId(data.clips[id])));
const needsListeningConfirmation=c=>hasUnavailableSource(c)&&![c.anchor,...c.candidates].every(id=>saved.listened[c.id]?.[id]);
function persist() {
  try {localStorage.setItem(key,JSON.stringify(saved));$("#save-status").textContent="이 브라우저에 저장했어요.";}
  catch(_){$("#save-status").textContent="브라우저 저장 불가 · JSON 사본을 저장해 주세요.";}
}
persist();
function saveDraft() {
  const c=currentCase(); if(saved.responses[c.id])return;
  saved.drafts[c.id]={choice:$("input[name=choice]:checked")?.value||null,memo:$("#memo").value};persist();updateAnswerControls();
}
function sourceState(slot) {
  const c=currentCase(),id=clipIds()[slot],card=$(`#clip-${slot}`),[start,end]=bounds(id);
  const heard=Boolean(saved.listened[c.id]?.[id]),opened=Boolean(saved.source_opens[c.id]?.[id]);
  card.querySelector(".source-status").textContent=heard?"이 구간을 들었다고 확인했어요.":unavailableVideoIds.has(Choice.sourceVideoId(data.clips[id]))?"11월 4일 Safari 확인 시 원본에서도 재생되지 않았습니다.":opened?"원본을 열었어요. 듣고 이 화면으로 돌아와 주세요.":Choice.sourceVideoId(data.clips[id])!==data.clips[id].video_id?"같은 앨범 트랙의 공식 대체 링크로 듣습니다.":"원본에서 듣고 이 화면으로 돌아오세요.";
  card.querySelector(".listen-confirm").checked=heard;
}
function bindSourceLinks() {
  document.querySelectorAll(".source-open").forEach(link=>{
    const slot=Number(link.dataset.slot),id=clipIds()[slot];
    link.href=playbackUrl(id);
    link.onclick=()=>{
      const c=currentCase();saved.source_opens[c.id] ||= {};
      saved.source_opens[c.id][id]={opened_at:new Date().toISOString(),slot:slotLabel(slot),excerpt_seconds:[...bounds(id)],url:link.href,playback_video_id:Choice.sourceVideoId(data.clips[id])};
      persist();sourceState(slot);
      $("#playback-status").textContent=`${slotLabel(slot)} 원본을 엽니다. 원본 페이지에서는 구간 종료와 반복을 직접 조작하세요.`;
    };
  });
}
function updateAnswerControls(){
  const response=saved.responses[currentCase().id],draft=saved.drafts[currentCase().id];
  document.querySelectorAll("input[name=choice]").forEach(r=>r.disabled=Boolean(response)||submitting);
  $("#memo").disabled=Boolean(response)||submitting;$("#submit").disabled=Boolean(response)||submitting||!draft?.choice||needsListeningConfirmation(currentCase());
  $("#submit").textContent=response?"선택이 저장되었습니다":"선택 저장하고 임베딩 결과 보기";
}
function updateBounds(slot){
  const id=clipIds()[slot],[start,end]=bounds(id),card=$(`#clip-${slot}`);
  card.querySelector(".times").innerHTML=`${clockTime(start)} – ${clockTime(end)} <small>${(end-start).toFixed(1)}초</small>`;
  card.querySelector(".source-stop").textContent=`${clockTime(end)}에서 멈추고 돌아오세요.`;
  card.querySelector(".start-input").value=start;card.querySelector(".end-input").value=end;
}
function updateTimingVisibility(){
  const visible=timingVisible(),answered=Boolean(saved.responses[currentCase().id]);
  $("#show-timing").checked=visible;$("#show-timing").disabled=answered;
  $("#timing-status").textContent=visible?"시간 표시 중 · 구간 위치가 어긋나면 조정할 수 있습니다.":"선택 전에는 재생 시간과 구간 길이를 숨깁니다. 원본 YouTube의 시간 표시는 그대로 보일 수 있습니다.";
  for(const card of document.querySelectorAll(".clip"))for(const selector of [".times",".source-stop",".adjust"])card.querySelector(selector).hidden=!visible;
  bindSourceLinks();
}
function renderClip(id,slot){
  const clip=data.clips[id],theme=clip.theme,identity=revealed.get(currentCase().id)?.recordings[id];
  const artist=identity?.artist||(slot===0?currentCase().anchor_artist:"연주자 정보는 선택 후 공개");
  return `<article class="clip" id="clip-${slot}"><div class="clip-head"><span class="clip-letter">${slotLabel(slot)}</span><span class="role">${slot===0?"기준 연주":"비교할 연주"}</span><h3>${esc(theme.title)}</h3><p class="artist">${esc(artist)}</p><p class="record">${esc(theme.label)} · ${esc(theme.formal_measures)}</p></div>
    <div class="video-wrap" hidden><div class="yt-mount"></div></div>
    <div class="excerpt-controls"><button class="play-excerpt" data-slot="${slot}">▶ ${slotLabel(slot)} 구간 재생${embedCheck(id)?.status==="blocked"?" 시도":""}</button><button class="pause-excerpt" data-slot="${slot}">일시정지</button></div>
    <p class="embed-status" role="status">${embedCheck(id)?.status==="blocked"?"백그라운드 검사에서 외부 재생 제한(150)이 확인됐습니다. 원본 링크를 이용할 수 있습니다.":"이 페이지에서 지정 구간을 재생합니다."}</p>
    <div class="source-panel"><p>삽입 재생이 제한되면 원본에서 듣기</p>
    <a class="source-open source-button" data-slot="${slot}" href="${esc(playbackUrl(id))}" target="_blank" rel="noopener noreferrer">${slotLabel(slot)} 원본 듣기 ↗</a><small class="source-stop"></small></div>
    <div class="clip-body"><p class="times"></p><p class="source-status" role="status"></p>
    <label class="listen-label"><input class="listen-confirm" type="checkbox"> ${slotLabel(slot)} 구간을 들었어요</label>
    <details class="adjust"><summary>재생 시간 조정</summary><label>시작(초) <input class="start-input" type="number" min="0" step="0.1" aria-label="${slotLabel(slot)} 시작 시간"></label><label>종료(초) <input class="end-input" type="number" min="0" step="0.1" aria-label="${slotLabel(slot)} 종료 시간"></label><button class="apply-times">적용</button> <button class="reset-times">추정 시간 복원</button><p class="time-error" role="alert"></p></details>
    <details class="source-detail"><summary>선정 구간 안내</summary><p>${esc(theme.evidence)}</p><p>${esc(theme.mapping_note)}</p>${theme.source_url?`<a href="${esc(theme.source_url)}" target="_blank" rel="noopener noreferrer">${esc(theme.source_title)} ↗</a>`:""}</details></div></article>`;
}
function destroyPlayers(){
  playerGeneration++;requestedSlot=null;for(const player of players.values())player.destroy();players.clear();pendingPlayers.clear();
}
function logPlayback(id,field,value){playbackLog[id] ||= {};playbackLog[id][field]=value;}
async function playExcerpt(slot){
  const id=clipIds()[slot],generation=playerGeneration,card=$(`#clip-${slot}`);
  if(pendingPlayers.has(slot))return;
  requestedSlot=slot;
  for(const [other,player] of players)if(other!==slot)player.pause();
  const existing=players.get(slot);
  if(existing&&existing.getStatus()!=="error"){existing.play();return;}
  const pending={};pendingPlayers.set(slot,pending);
  const button=card.querySelector(".play-excerpt");button.disabled=true;
  card.querySelector(".embed-status").textContent="YouTube 삽입 플레이어를 준비합니다.";
  try{
    const api=await Playback.loadApi(window);if(generation!==playerGeneration||requestedSlot!==slot)return;
    existing?.destroy();
    const wrap=card.querySelector(".video-wrap");wrap.hidden=false;
    wrap.replaceChildren();const mount=document.createElement("div");wrap.append(mount);
    const messages={loading:"플레이어 준비 중",ready:"구간 재생 준비 완료",playing:"지정 구간 재생 중",paused:"일시정지",ended:"구간 재생이 끝났습니다.",repeating:"구간을 처음부터 반복합니다.","autoplay-blocked":"브라우저가 재생을 막았습니다. 플레이어의 재생 버튼을 눌러주세요."};
    const player=Playback.createPlayer({api,element:mount,videoId:Choice.sourceVideoId(data.clips[id]),start:bounds(id)[0],end:bounds(id)[1],origin:location.origin,
      repeat:()=>$("#loop-excerpts").checked,
      onStatus:status=>{if(generation===playerGeneration&&messages[status])card.querySelector(".embed-status").textContent=messages[status];},
      onPlay:()=>{if(generation!==playerGeneration)return;for(const [other,p] of players)if(other!==slot)p.pause();logPlayback(id,"last_embed_playing_at",new Date().toISOString());},
      onRepeat:()=>logPlayback(id,"loop_restarts",(playbackLog[id]?.loop_restarts||0)+1),
      onError:(code,message)=>{if(generation!==playerGeneration)return;card.querySelector(".embed-status").textContent=`${message} (${code})`;wrap.hidden=true;logPlayback(id,"embed_error",code);}
    });
    players.set(slot,player);logPlayback(id,"requested_excerpt",[...bounds(id)]);player.play();
  }catch(error){if(generation===playerGeneration)card.querySelector(".embed-status").textContent=error.message;}
  finally{if(pendingPlayers.get(slot)===pending)pendingPlayers.delete(slot);if(generation===playerGeneration)button.disabled=false;}
}
function render(){
  destroyPlayers();
  submitting=false;
  const c=currentCase();$("#case-label").textContent=c.label;
  $("#case-position").textContent=`${caseIndex+1} / ${data.cases.length}문항`;
  $("#previous-case").disabled=caseIndex===0;$("#next-case").disabled=caseIndex===data.cases.length-1;
  $("#source-warning").hidden=!hasUnavailableSource(c);
  $("#case-tabs").innerHTML=data.cases.map((v,i)=>`<button data-case="${i}" aria-current="${i===caseIndex}">${String(i+1).padStart(2,"0")} &nbsp; ${esc(v.label)}${saved.responses[v.id]?" · 응답 저장됨":""}</button>`).join("");
  $("#players").innerHTML=clipIds().map(renderClip).join("");
  for(let slot=0;slot<3;slot++){
    const id=clipIds()[slot],card=$(`#clip-${slot}`);updateBounds(slot);sourceState(slot);
    card.querySelector(".listen-confirm").onchange=event=>{
      saved.listened[c.id] ||= {};saved.listened[c.id][id]=event.target.checked;persist();sourceState(slot);updateAnswerControls();
    };
    card.querySelector(".apply-times").onclick=()=>{
      const sf=card.querySelector(".start-input"),ef=card.querySelector(".end-input"),start=Number(sf.value),end=Number(ef.value);
      if(!sf.value||!ef.value||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>data.clips[id].track_duration_seconds){card.querySelector(".time-error").textContent="0 ≤ 시작 < 종료 ≤ 원본 길이가 되도록 입력하세요.";return;}
      saved.times[id]=[start,end];players.get(slot)?.setRange(start,end);if(saved.listened[c.id])delete saved.listened[c.id][id];persist();updateBounds(slot);sourceState(slot);bindSourceLinks();updateAnswerControls();card.querySelector(".time-error").textContent="적용했어요.";
    };
    card.querySelector(".reset-times").onclick=()=>{delete saved.times[id];players.get(slot)?.setRange(...bounds(id));if(saved.listened[c.id])delete saved.listened[c.id][id];persist();updateBounds(slot);sourceState(slot);bindSourceLinks();updateAnswerControls();card.querySelector(".time-error").textContent="추정 시간으로 복원했어요.";};
    card.querySelector(".play-excerpt").onclick=()=>playExcerpt(slot);
    card.querySelector(".pause-excerpt").onclick=()=>players.get(slot)?.pause();
  }
  const response=saved.responses[c.id],draft=response||saved.drafts[c.id]||{};
  document.querySelectorAll("input[name=choice]").forEach(r=>r.checked=r.value===draft.choice);$("#memo").value=draft.memo||"";updateAnswerControls();
  $("#answer-status").textContent=response?"거리 공개 전에 저장한 선택입니다. 결과를 본 뒤에는 수정하지 않습니다.":"선택 전에는 거리와 모델 순위를 보여주지 않습니다.";
  $("#playback-status").textContent="A를 먼저 듣고 X와 Y를 비교하세요. 단축키 1 · 2 · 3으로 삽입 구간 재생을 시도합니다.";
  $("#result").hidden=true;$("#retry-result").hidden=true;updateTimingVisibility();updateSummary();
  if(response)reveal(c).catch(e=>resultError(c,e));
}
function updateSummary(){
  const s=Choice.summarize(Object.values(saved.responses)),done=s.same+s.opposite+s.tie+s.unsure;
  $("#summary").textContent=done?`결과 확인 ${done}/${data.cases.length} · 일치 ${s.same} · 반대 ${s.opposite} · 둘이 비슷 ${s.tie} · 판단 어려움 ${s.unsure}${s.decisive?` · 확정 선택 일치 ${s.same}/${s.decisive}`:" · 확정 선택 없음"}`:"아직 확인한 결과가 없습니다.";
  const currentIds=new Set(data.cases.map(c=>c.id)),archived=new Set();
  for(const pack of Object.values(saved.archived_packs||{}))for(const [id,response] of Object.entries(pack.responses||{}))if(!currentIds.has(id))archived.add(`${id}:${response.submitted_at||""}`);
  if(archived.size)$("#summary").textContent+=` · 이전 문항 응답 ${archived.size}개 별도 보관 (JSON에 포함)`;
}
async function reveal(c){
  const response=saved.responses[c.id];if(!response)throw new Error("Save a listener choice before loading model results");
  let answer=revealed.get(c.id);
  if(!answer){const request=await fetch(`${c.result_url}?pack=${data.id}`,{cache:"no-store"});if(!request.ok)throw new Error(`결과 파일을 불러오지 못했어요 (${request.status}).`);answer=await request.json();}
  const evaluated=Choice.evaluate(data.id,c,saved.orders[c.id],response.choice,answer);
  revealed.set(c.id,answer);Object.assign(response,evaluated);response.model_result=answer;persist();updateSummary();if(currentCase().id!==c.id)return;
  const messages={same:["임베딩과 일치했어요.","더 비슷하게 들렸다고 고른 쪽이 임베딩에서도 A에 더 가까웠습니다."],opposite:["임베딩과 반대였어요.","더 비슷하게 들렸다고 고른 쪽은 임베딩에서는 A에서 더 먼 연주였습니다."],tie:["두 후보가 비슷하게 들렸군요.","이 응답은 일치·반대에 포함하지 않습니다. 아래에서 모델의 거리 차이를 확인하세요."],unsure:["판단 어려움으로 기록했어요.","이 응답은 일치·반대에 포함하지 않습니다. 아래에서 모델의 거리 차이를 확인하세요."]};
  const [title,explanation]=messages[evaluated.outcome],labels={X:"X",Y:"Y",tie:"둘이 비슷함",unsure:"판단 어려움"};
  $("#result").innerHTML=`<p class="eyebrow">YOUR CHOICE × EMBEDDING</p><h2>${title}</h2><p>${explanation}</p><p>내 선택: <strong>${labels[response.choice]}</strong> · 임베딩에서 더 가까운 쪽: <strong>${evaluated.model_closer_label}</strong></p><table><thead><tr><th>후보</th><th>실제 연주</th><th>A와의 주제부 거리</th><th>모델의 구분</th></tr></thead><tbody>${saved.orders[c.id].map((id,i)=>{
    const row=answer.candidates.find(r=>r.clip===id),identity=answer.recordings[id];
    return `<tr><th>${i?"Y":"X"}${response.choice===(i?"Y":"X")?" · 내 선택":""}</th><td>${esc(identity.artist)}<small>${esc(identity.performance_id)}</small></td><td>${row.cosine_distance_to_anchor.toFixed(6)}</td><td>${id===answer.near_clip?"가까운 연주":"같은 작품의 먼 연주"}</td></tr>`;
  }).join("")}</tbody></table><p class="result-note">가까운 연주는 다른 작품·악보 후보에서 A와의 거리 1위입니다. 먼 연주는 그 연주와 작품·악보·선정 후보가 같은 나머지 ${answer.far_work_pool}개 연주 중 A와의 거리가 가장 큽니다. 작은 거리일수록 가깝습니다. 세 연주 모두 실제로 듣는 선정 주제부의 임베딩으로 비교했습니다.</p>`;
  $("#result").hidden=false;$("#retry-result").hidden=true;updateTimingVisibility();clipIds().forEach((id,slot)=>$(`#clip-${slot} .artist`).textContent=answer.recordings[id].artist);
}
function resultError(c,error){if(currentCase().id!==c.id)return;$("#answer-status").textContent=`선택은 저장되어 있습니다. ${error.message}`;$("#retry-result").hidden=false;}
$("#submit").onclick=async()=>{
  const c=currentCase(),draft=saved.drafts[c.id];if(submitting||saved.responses[c.id]||!draft?.choice||needsListeningConfirmation(c))return;
  const sourceUrls=Object.fromEntries(clipIds().map(id=>[id,playbackUrl(id)]));
  submitting=true;saved.responses[c.id]={playback_mode:"youtube-embedded-or-original",time_information_shown_before_choice:Boolean(saved.timing_exposures?.[c.id]||saved.timing_visibility?.[c.id]),embedded_player_controls:0,original_page_time_hiding:false,loop_requested:$("#loop-excerpts").checked,playback_events:JSON.parse(JSON.stringify(Object.fromEntries(clipIds().filter(id=>playbackLog[id]).map(id=>[id,playbackLog[id]])))),self_reported_listened:{...saved.listened[c.id]},case_id:c.id,choice:draft.choice,memo:draft.memo,submitted_at:new Date().toISOString(),xy_order:[...saved.orders[c.id]],excerpt_intervals_at_submission:Object.fromEntries(clipIds().map(id=>[id,[...bounds(id)]])),source_urls_at_submission:sourceUrls};
  persist();updateAnswerControls();$("#answer-status").textContent="선택을 저장했어요. 이제 임베딩 결과를 불러옵니다.";
  try{await reveal(c);}catch(e){resultError(c,e);}finally{if(currentCase().id===c.id){submitting=false;updateAnswerControls();}}
};
$("#retry-result").onclick=()=>{const c=currentCase();reveal(c).catch(e=>resultError(c,e));};
$("#case-tabs").onclick=event=>{const button=event.target.closest("button[data-case]");if(!button||submitting)return;caseIndex=Number(button.dataset.case);render();};
$("#previous-case").onclick=()=>{if(!submitting&&caseIndex>0){caseIndex--;render();}};
$("#next-case").onclick=()=>{if(!submitting&&caseIndex<data.cases.length-1){caseIndex++;render();}};
$("#pause-all").onclick=()=>{for(const player of players.values())player.pause();};
$("#memo").addEventListener("input",saveDraft);$("#choice").addEventListener("change",saveDraft);
$("#show-timing").onchange=event=>{
  const id=currentCase().id;if(saved.responses[id])return;
  saved.timing_visibility ||= {};saved.timing_visibility[id]=event.target.checked;
  if(event.target.checked){saved.timing_exposures ||= {};saved.timing_exposures[id]=new Date().toISOString();}
  persist();updateTimingVisibility();
};
$("#export").onclick=()=>{
  const result={...saved,exported_at:new Date().toISOString(),summary:Choice.summarize(Object.values(saved.responses)),distance_scope:"whole-recording mean AE embedding; not theme-only"};
  try{const backups=localStorage.getItem("classicfy-theme-origin-backups-v1");if(backups)result.origin_storage_backups=JSON.parse(backups);}catch(_){}
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="classicfy-near-far-listening.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.addEventListener("keydown",event=>{
  if(/^(INPUT|TEXTAREA|SELECT|BUTTON|SUMMARY|A)$/.test(event.target.tagName)||event.metaKey||event.ctrlKey||event.altKey)return;
  if(["1","2","3"].includes(event.key)){event.preventDefault();playExcerpt(Number(event.key)-1);}
});
$("#model").textContent=`${data.cases.length}문항 · NOTE AE · EPOCH ${data.epoch} / ${data.catalogue.recordings.toLocaleString()}개 연주 / ${data.catalogue.works}개 작품`;$("#pack-id").textContent=`PACK ${data.id.slice(0,10)}`;
if(isTest)$(".edition").textContent="UI 검증용 · 실제 응답 제외";
render();
