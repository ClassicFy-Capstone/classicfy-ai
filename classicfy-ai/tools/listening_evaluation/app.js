"use strict";
(() => {
  const study = window.LISTENING_STUDY;
  const el = id => document.getElementById(id);
  if (!study || study.schema_version !== 1) {
    el("global-message").textContent = "평가 자료가 준비되지 않았습니다. prepare_listening_study.py로 생성한 참가자 pack을 열어 주세요.";
    el("start").disabled = true; return;
  }
  const features = [["tempo", "전체 빠르기"], ["rubato", "국소적인 빠르기 변화"], ["dynamics", "강약의 크기와 대비"], ["articulation", "음의 연결·분리"], ["pedaling", "잔향·울림"]];
  const fields = ["overall", ...features.map(f => f[0]), "confidence"];
  const scale = [["1", "1 · 거의 없음"], ["2", "2 · 작음"], ["3", "3 · 분명함"], ["4", "4 · 큼"], ["5", "5 · 매우 큼"], ["unclear", "판단 어려움"]];
  const choices = [["X", "X"], ["Y", "Y"], ["tie", "둘이 비슷함"], ["unclear", "판단 어려움"]];
  let participant = "", position = 0, answers = {}, storageAvailable = true, feedbackRevealedAt = null;
  const schedule = () => study.participants[participant];
  const task = () => schedule()[position];
  const storageKey = p => `classicfy-listening-${study.study_id}-${p}`;
  el("task-count").textContent = study.task_count;
  if (study.section) {
    document.title = `Classicfy · ${study.section.work_title}`;
    el("page-title").textContent = study.section.work_title;
    el("page-lead").textContent = study.section.label;
    el("study-intro").textContent = `총 ${study.task_count}문항 · 두 연주의 같은 주제 구간을 비교합니다. 참가자 번호를 선택하고 시작하세요.`;
    el("comment").placeholder = "예: X는 선율의 끝에서 더 여유를 두고, Y는 반주가 더 또렷하게 들림.";
    if (study.section.mode === "cross") {
      el("study-intro").textContent = `총 ${study.task_count}문항 · 기준 A를 듣고, 다른 한 작품의 후보 X/Y를 비교합니다. 참가자별 X/Y 순서가 다릅니다. 각자 독립적으로 평가하세요.`;
      el("rating-guide").textContent = "선택: A와 더 비슷한 후보 X 또는 Y. 둘이 비슷하게 느껴지거나 판단하기 어렵다면 해당 선택지를 사용하세요. 답을 기록한 뒤 임베딩 기준과 비교할 수 있습니다.";
      el("comment").placeholder = "예: A와 X는 선율 끝에서 여유를 두는 느낌이 비슷하고, Y는 더 일정하게 들림.";
    }
  }
  function pause() { document.querySelectorAll("audio").forEach(a => a.pause()); }
  function persist() {
    try { localStorage.setItem(storageKey(participant), JSON.stringify({ position, answers, ...(study.feedback ? { feedback_revealed_at: feedbackRevealedAt } : {}) })); storageAvailable = true; }
    catch (e) { storageAvailable = false; }
    el("status").textContent = storageAvailable ? "이 브라우저에 저장됨" : "자동 저장 불가 · JSON 저장 버튼을 사용하세요";
  }
  function update(field, value) {
    if (feedbackRevealedAt) return;
    answers[task().task_id] = { ...(answers[task().task_id] || {}), [field]: value, updated_at: new Date().toISOString() };
    persist();
  }
  function validateAnswers(value, p) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("응답 형식이 올바르지 않습니다.");
    const tasks = study.participants[p];
    for (const [id, answer] of Object.entries(value)) {
      const t = tasks.find(t => t.task_id === id);
      if (!t || !answer || typeof answer !== "object" || Array.isArray(answer)) throw Error("알 수 없는 문항이 있습니다.");
      for (const field of fields) {
        if (answer[field] === undefined) continue;
        const allowed = field === "confidence" ? ["1", "2", "3"] : (t.mode === "cross" ? choices.map(x => x[0]) : scale.map(x => x[0]));
        if (!allowed.includes(answer[field])) throw Error("평가 척도가 올바르지 않습니다.");
      }
      if (answer.comment !== undefined && (typeof answer.comment !== "string" || answer.comment.length > 10000)) throw Error("메모 형식이 올바르지 않습니다.");
    }
  }
  function rating(name, title, options) {
    const wrap = document.createElement("fieldset"); wrap.className = "rating";
    const legend = document.createElement("legend"); legend.textContent = title; wrap.append(legend);
    const row = document.createElement("div"); row.className = "choices";
    for (const [value, label] of options) {
      const lab = document.createElement("label"); lab.className = "choice";
      const input = document.createElement("input"); input.type = "radio"; input.name = name; input.value = value;
      input.checked = (answers[task().task_id] || {})[name] === value;
      input.addEventListener("change", () => update(name, value));
      lab.append(input, document.createTextNode(label)); row.append(lab);
    }
    wrap.append(row); el("questions").append(wrap);
  }
  function render() {
    pause(); const t = task(); el("message").textContent = "";
    el("progress").textContent = `${participant} · ${position + 1} / ${schedule().length} · ${t.task_id}`;
    el("progress-bar").max = schedule().length; el("progress-bar").value = position;
    el("prompt").textContent = t.mode === "pair" ? "X와 Y의 연주 해석은 얼마나 다르게 들리나요?" : "A와 연주 경향이 더 비슷하게 들리는 후보는 어느 쪽인가요?";
    el("detail").textContent = t.mode === "pair" ? "같은 작품의 같은 악보 구간입니다. 구간 1의 X/Y, 구간 2의 X/Y를 비교한 뒤 두 구간을 종합해 평가하세요." : "X와 Y는 같은 작품의 같은 악보 구간이고, A는 다른 작품입니다. 곡의 선율보다 빠르기 변화·강약·연결감·울림의 경향을 비교하세요.";
    if (study.section) el("detail").textContent = study.section.description;
    el("audio").replaceChildren();
    for (const [label, clips] of Object.entries(t.stimuli)) {
      const box = document.createElement("div"); box.className = "stimulus";
      if (t.mode === "cross") box.classList.add(label === "A" ? "reference-stimulus" : "candidate-stimulus");
      const h = document.createElement("h3"); h.textContent = t.mode === "cross" ? `${label} · ${label === "A" ? "기준 연주" : "후보 연주"}` : label; box.append(h);
      for (const clip of clips) {
        const p = document.createElement("p"); p.className = "clip-label"; p.textContent = clip.label || `구간 ${clip.part}`;
        const audio = document.createElement("audio"); audio.controls = true; audio.preload = "metadata"; audio.src = clip.audio;
        audio.setAttribute("aria-label", `${label} 구간 ${clip.part}`);
        audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); }));
        audio.addEventListener("error", () => { el("message").textContent = `${label} 구간 ${clip.part}을 재생할 수 없습니다. pack의 audio 폴더를 확인하세요. 이 문항은 판단 어려움으로 기록할 수 있습니다.`; });
        box.append(p, audio);
      }
      el("audio").append(box);
    }
    el("questions").replaceChildren();
    if (t.mode === "pair") {
      rating("overall", "전체적으로 느껴지는 차이", scale);
      features.forEach(([name, label]) => rating(name, `${label}의 차이`, scale));
    } else {
      rating("overall", "A와 전반적인 연주 경향이 더 비슷한 후보", choices);
      features.forEach(([name, label]) => rating(name, `${label}에서 A와 더 비슷한 후보`, choices));
    }
    rating("confidence", "이 판단에 대한 확신", [["1", "낮음"], ["2", "보통"], ["3", "높음"]]);
    el("comment").value = (answers[t.task_id] || {}).comment || "";
    el("ratings").querySelectorAll("input, textarea").forEach(input => { input.disabled = Boolean(feedbackRevealedAt); });
    if (feedbackRevealedAt) el("message").textContent = "임베딩 기준을 확인한 뒤의 응답은 읽기 전용입니다. 음원은 다시 들을 수 있습니다.";
    el("prev").disabled = position === 0; el("next").textContent = position === schedule().length - 1 ? "평가 마치기" : "다음";
    persist();
  }
  function payload() { return { schema_version: 1, study_id: study.study_id, participant, exported_at: new Date().toISOString(), schedule: schedule(), answers, ...(study.feedback ? { feedback_revealed_at: feedbackRevealedAt } : {}) }; }
  function download(content, type, extension) {
    const url = URL.createObjectURL(new Blob([content], { type })); const a = document.createElement("a");
    a.href = url; a.download = `classicfy-listening-${study.study_id}-${participant}.${extension}`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function exportJson() { download(JSON.stringify(payload(), null, 2), "application/json", "json"); }
  function exportCsv() {
    const columns = ["study_id", "participant", "position", "task_id", "mode", ...fields, "comment", ...(study.feedback ? ["feedback_revealed_at"] : [])];
    const escape = x => '"' + String(x ?? "").replaceAll('"', '""') + '"';
    const rows = schedule().map((t, i) => { const a = answers[t.task_id] || {}; return [study.study_id, participant, i + 1, t.task_id, t.mode, ...fields.map(f => a[f]), a.comment, ...(study.feedback ? [feedbackRevealedAt] : [])]; });
    download("\uFEFF" + [columns, ...rows].map(r => r.map(escape).join(",")).join("\n") + "\n", "text/csv;charset=utf-8", "csv");
  }
  function home() { pause(); ["study", "done"].forEach(id => el(id).classList.add("hidden")); el("welcome").classList.remove("hidden"); }
  function showDone() {
    pause(); persist(); el("study").classList.add("hidden"); el("done").classList.remove("hidden");
    el("completion").textContent = `${participant} · ${schedule().length}문항 완료. JSON 파일을 진행자에게 전달하세요.`;
    el("feedback-section").classList.toggle("hidden", !study.feedback);
    el("feedback-result").replaceChildren();
    el("feedback-message").textContent = feedbackRevealedAt ? "이미 임베딩 기준을 확인한 응답입니다. 다시 확인하거나 음원을 들을 수 있습니다. 응답은 읽기 전용으로 유지됩니다." : "응답을 기록했습니다. 아래 버튼을 누르면 X/Y의 임베딩상 가까움·멀어짐을 공개합니다. 공개한 뒤에는 응답을 바꿀 수 없습니다.";
  }
  function revealDate(value) {
    if (value == null) return null;
    if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw Error("응답의 공개 시각이 올바르지 않습니다.");
    return value;
  }
  el("reveal-feedback").onclick = async () => {
    if (!study.feedback || schedule().some(t => fields.some(f => !(answers[t.task_id] || {})[f]))) return;
    const button = el("reveal-feedback"); button.disabled = true;
    try {
      const response = await fetch(study.feedback.url, { cache: "no-store" });
      if (!response.ok) throw Error("비교 자료를 읽을 수 없습니다.");
      const data = await response.json(); const expected = data.participants?.[participant];
      if (data.study_id !== study.study_id || !expected) throw Error("다른 평가의 비교 자료입니다.");
      for (const field of ["overall", ...features.map(f => f[0])]) {
        const item = expected[field];
        if (!item || !["X", "Y", "tie"].includes(item.choice) || !["X", "Y"].every(label => Number.isFinite(item.distances?.[label]))) throw Error("비교 자료 형식이 올바르지 않습니다.");
      }
      feedbackRevealedAt ||= new Date().toISOString(); persist();
      const result = el("feedback-result"); result.replaceChildren();
      const title = document.createElement("h3"); title.textContent = `임베딩상 A와 더 가까운 후보: ${expected.overall.choice}`;
      const distances = document.createElement("p"); distances.textContent = `A–X 거리 ${expected.overall.distances.X.toFixed(3)} · A–Y 거리 ${expected.overall.distances.Y.toFixed(3)} (작을수록 가까움)`;
      const table = document.createElement("table");
      const head = document.createElement("tr");
      ["항목", "내 선택", "임베딩 기준", "비교"].forEach(text => { const cell = document.createElement("th"); cell.textContent = text; head.append(cell); });
      table.append(head);
      const current = answers[schedule()[0].task_id];
      for (const [field, label] of [["overall", "전체 성향"], ...features]) {
        const choice = current[field]; const target = expected[field].choice;
        const status = choice === "tie" || choice === "unclear" ? "판단 보류" : target === "tie" ? "거리 동점" : choice === target ? "일치" : "다름";
        const row = document.createElement("tr");
        [label, choices.find(c => c[0] === choice)?.[1] || choice, target, status].forEach(text => { const cell = document.createElement("td"); cell.textContent = text; row.append(cell); });
        table.append(row);
      }
      const note = document.createElement("p"); note.className = "muted"; note.textContent = data.interpretation;
      result.append(title, distances, table, note);
      el("feedback-message").textContent = "공개 전 응답을 보존했습니다. 이제 응답은 읽기 전용이며, 음원을 다시 들어볼 수 있습니다.";
    } catch (error) { el("feedback-message").textContent = `비교하지 못했습니다: ${error.message} 로컬 HTTP 서버에서 열었는지 확인하세요.`; }
    finally { button.disabled = false; }
  };
  el("start").onclick = () => {
    participant = el("participant").value; el("global-message").textContent = "";
    if (!study.participants[participant]) { el("global-message").textContent = "참가자 번호를 선택하세요."; return; }
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey(participant)) || "{}");
      answers = stored.answers || {}; validateAnswers(answers, participant);
      feedbackRevealedAt = study.feedback ? revealDate(stored.feedback_revealed_at) : null;
      position = Number.isInteger(stored.position) && stored.position >= 0 ? Math.min(stored.position, schedule().length - 1) : 0;
    } catch (e) {
      if (e instanceof SyntaxError || e.message.startsWith("응답") || e.message.startsWith("알 수") || e.message.startsWith("평가") || e.message.startsWith("메모")) {
        el("global-message").textContent = "임시 응답을 읽을 수 없습니다. 이전 JSON이 있으면 불러와 주세요. 현재 응답은 덮어쓰지 않았습니다."; return;
      }
      answers = {}; position = 0; storageAvailable = false; feedbackRevealedAt = null;
    }
    el("welcome").classList.add("hidden");
    if (feedbackRevealedAt && schedule().every(t => fields.every(f => (answers[t.task_id] || {})[f]))) showDone();
    else { el("study").classList.remove("hidden"); render(); }
  };
  el("comment").oninput = () => update("comment", el("comment").value);
  el("ratings").onsubmit = e => e.preventDefault();
  el("prev").onclick = () => { position--; render(); };
  el("next").onclick = () => {
    const a = answers[task().task_id] || {};
    if (fields.some(f => !a[f])) { el("message").textContent = "모든 항목에 답해 주세요. 듣기 어렵다면 판단 어려움을 선택할 수 있어요."; return; }
    if (position + 1 < schedule().length) { position++; render(); }
    else {
      showDone();
    }
    window.scrollTo(0, 0);
  };
  el("back").onclick = () => { el("done").classList.add("hidden"); el("study").classList.remove("hidden"); render(); };
  ["export-json", "finish-json"].forEach(id => el(id).onclick = exportJson);
  ["export-csv", "finish-csv"].forEach(id => el(id).onclick = exportCsv);
  ["exit", "home"].forEach(id => el(id).onclick = home);
  el("import").onchange = async e => {
    el("global-message").textContent = "";
    try {
      const file = e.target.files[0]; if (!file) return;
      if (file.size > 2 * 1024 * 1024) throw Error("응답 파일이 너무 큽니다.");
      const data = JSON.parse(await file.text()); const p = data.participant;
      if (data.schema_version !== 1 || data.study_id !== study.study_id || !study.participants[p]) throw Error("다른 평가의 응답 파일입니다.");
      if (JSON.stringify(data.schedule) !== JSON.stringify(study.participants[p])) throw Error("평가 순서가 일치하지 않습니다.");
      validateAnswers(data.answers, p);
      if (localStorage.getItem(storageKey(p)) && !window.confirm(`${p}의 이 브라우저 임시 응답을 불러온 파일로 교체할까요?`)) return;
      const tasks = study.participants[p]; const next = tasks.findIndex(t => fields.some(f => !(data.answers[t.task_id] || {})[f]));
      let previous = {};
      try { previous = JSON.parse(localStorage.getItem(storageKey(p)) || "{}"); } catch (e) { /* Restore a saved response even if local storage was corrupt. */ }
      const knownRevealedAt = study.feedback ? revealDate(previous.feedback_revealed_at) || revealDate(data.feedback_revealed_at) : null;
      if (previous.feedback_revealed_at && previous.answers) {
        validateAnswers(previous.answers, p);
        throw Error("이 참가자는 이미 임베딩 기준을 확인했습니다. 공개 전 응답을 바꾸지 않도록 현재 브라우저의 응답을 유지합니다.");
      }
      localStorage.setItem(storageKey(p), JSON.stringify({ position: next < 0 ? tasks.length - 1 : next, answers: data.answers, ...(study.feedback ? { feedback_revealed_at: knownRevealedAt } : {}) }));
      el("participant").value = p; el("global-message").textContent = "응답을 불러왔습니다. 평가 시작 / 이어하기를 눌러 주세요.";
    } catch (error) { el("global-message").textContent = `불러오지 못했습니다: ${error.message}`; }
    e.target.value = "";
  };
})();
