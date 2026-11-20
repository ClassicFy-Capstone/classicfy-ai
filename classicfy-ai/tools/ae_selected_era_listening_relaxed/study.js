/* Shared, side-effect-free choice validation, scoring and X/Y assignment. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ThemeChoice = api;
})(globalThis, function () {
  "use strict";
  function validOrder(candidates, order) {
    return Array.isArray(candidates) && Array.isArray(order) && candidates.length === 2 && order.length === 2 &&
      new Set(candidates).size === 2 && new Set(order).size === 2 && order.every(id => candidates.includes(id));
  }
  function makeOrder(candidates, randomWord) {
    if (!validOrder(candidates, candidates)) throw new Error("Expected two distinct candidate recordings");
    const coin = randomWord === undefined ? crypto.getRandomValues(new Uint32Array(1))[0] : randomWord;
    return coin & 1 ? [...candidates].reverse() : [...candidates];
  }
  function evaluate(packId, definition, order, choice, result) {
    if (!validOrder(definition.candidates, order) || !["X", "Y", "tie", "unsure"].includes(choice)) throw new Error("Invalid displayed order or listener choice");
    if (result.pack_id !== packId || result.case_id !== definition.id || result.distance_reference !== "A") throw new Error("Result belongs to another pack, case or distance reference");
    const rows = result.candidates;
    if (!Array.isArray(rows) || rows.length !== 2 || !validOrder(order, rows.map(r => r.clip))) throw new Error("Candidate identity mismatch");
    const near = rows.find(r => r.clip === result.near_clip), far = rows.find(r => r.clip === result.far_clip);
    if (!near || !far || near === far || ![near, far].every(r => Number.isFinite(r.cosine_distance_to_anchor) && r.cosine_distance_to_anchor >= 0 && r.cosine_distance_to_anchor <= 2) || near.cosine_distance_to_anchor >= far.cosine_distance_to_anchor) throw new Error("Invalid near/far distances");
    const modelLabel = order[0] === result.near_clip ? "X" : "Y", directional = choice === "X" || choice === "Y";
    return {choice, model_closer_label: modelLabel, chosen_clip: directional ? order[choice === "X" ? 0 : 1] : null,
      outcome: directional ? (choice === modelLabel ? "same" : "opposite") : choice,
      distances: Object.fromEntries(order.map((id, i) => [i === 0 ? "X" : "Y", rows.find(r => r.clip === id).cosine_distance_to_anchor]))};
  }
  function summarize(responses) {
    const counts = {same:0, opposite:0, tie:0, unsure:0};
    for (const response of responses) if (response && Object.hasOwn(counts, response.outcome)) counts[response.outcome]++;
    const decisive = counts.same + counts.opposite;
    return {...counts, decisive, agreement_fraction:decisive ? counts.same/decisive : null};
  }
  function migrateSaved(saved, data, timestamp) {
    if (!saved || !data.compatible_pack_ids?.includes(saved.pack_id) || !saved.orders || !saved.responses || !saved.drafts || !saved.times) return null;
    const definitions = new Map(data.cases.map(c=>[c.id,c]));
    const policy = data.migration_sources?.find(source=>source.pack_id===saved.pack_id);
    const keep = new Set(policy ? policy.case_ids : definitions.keys());
    for (const [id,order] of Object.entries(saved.orders)) {
      if (!keep.has(id)) continue;
      if (!definitions.has(id) || !validOrder(definitions.get(id).candidates,order)) return null;
    }
    for (const [id,response] of Object.entries(saved.responses)) {
      if (!keep.has(id)) continue;
      if (!definitions.has(id) || response.case_id!==id || !saved.orders[id] || !["X","Y","tie","unsure"].includes(response.choice) || !Array.isArray(response.xy_order) || response.xy_order.some((clip,i)=>clip!==saved.orders[id][i]) || response.xy_order.length!==2) return null;
    }
    if (!policy && Object.keys(saved.times).some(id=>!data.clips[id])) return null;
    const result = JSON.parse(JSON.stringify(saved));
    const removed = Object.keys(saved.orders).filter(id=>!keep.has(id));
    if (removed.length) {
      result.archived_packs ||= {};
      const snapshot = JSON.parse(JSON.stringify(saved));
      delete snapshot.archived_packs;
      result.archived_packs[saved.pack_id] = snapshot;
    }
    for (const field of ["orders","responses","drafts","source_opens","listened","timing_visibility","timing_exposures"]) {
      if (result[field]) result[field]=Object.fromEntries(Object.entries(result[field]).filter(([id])=>keep.has(id)));
    }
    result.times=Object.fromEntries(Object.entries(result.times).filter(([id])=>data.clips[id]));
    result.migration_history ||= [];
    result.migration_history.push({from_pack_id:saved.pack_id,to_pack_id:data.id,migrated_at:timestamp,retained_cases:[...keep],archived_cases:removed});
    for (const response of Object.values(result.responses)) response.submitted_pack_id ||= saved.pack_id;
    result.pack_id=data.id;
    return result;
  }
  // Verified official upload of the same album track. This changes playback
  // only; the model's recording, distance, pack ID and saved X/Y order stay fixed.
  // Evidence: analysis/ae_theme_listening/source_replacements.json.
  const sourceReplacements = Object.freeze({"0QIgscF9ezY":"V4bTHofOXgE"});
  function sourceVideoId(clip) {
    if (!clip || !/^[A-Za-z0-9_-]{11}$/.test(clip.video_id)) throw new Error("Invalid source video");
    const source = new URL(clip.youtube_url);
    if (source.protocol !== "https:" || source.hostname !== "www.youtube.com" || source.pathname !== "/watch" || source.searchParams.get("v") !== clip.video_id) throw new Error("Original recording identity mismatch");
    return sourceReplacements[clip.video_id] || clip.video_id;
  }
  function watchUrl(clip, start) {
    if (!Number.isFinite(start) || start < 0) throw new Error("Invalid source start time");
    return `https://www.youtube.com/watch?v=${sourceVideoId(clip)}&t=${Math.floor(start)}s`;
  }
  return {validOrder, makeOrder, evaluate, summarize, migrateSaved, watchUrl, sourceVideoId};
});
