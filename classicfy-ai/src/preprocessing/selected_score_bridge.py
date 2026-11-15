"""Audited symbolic MIDI -> original MusicXML note/quarter-coordinate bridges.

Route selection uses score pitches only, never performance expression or AE loss.
Unmatched notes stay unknown; this module never fabricates a notation target.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from difflib import SequenceMatcher
import hashlib
import json
import re

import numpy as np
from numba import njit

from .musicxml_score import _root, read_musicxml_root
from .musicxml_time_repair import normalize_quantized_timing
from .musicxml_navigation_repair import prepare_navigation_recovery
from .score_midi import read_score_midi
from .score_note_mapping import build_tie_mapping


@njit(cache=True)
def _weighted_lcs(a, b, weights):
    n,m=len(a),len(b)
    previous=np.zeros(m+1,np.int32)
    trace=np.zeros((n,m),np.uint8)
    for i in range(n):
        current=np.zeros(m+1,np.int32)
        for j in range(m):
            above,left=previous[j+1],current[j]
            best=max(above,left)
            if a[i]==b[j] and previous[j]+weights[i]>=best:
                current[j+1]=previous[j]+weights[i];trace[i,j]=0
            elif above>=left:
                current[j+1]=above;trace[i,j]=1
            else:
                current[j+1]=left;trace[i,j]=2
        previous=current
    ai,bj=[],[]
    i,j=n-1,m-1
    while i>=0 and j>=0:
        direction=trace[i,j]
        if direction==0:ai.append(i);bj.append(j);i-=1;j-=1
        elif direction==1:i-=1
        else:j-=1
    return ai,bj


def exact_group_pairs(left, right):
    pairs={}
    for block in SequenceMatcher(None,left,right,autojunk=False).get_matching_blocks():
        for k in range(block.size):pairs[block.a+k]=block.b+k
    if len(pairs)<.95*min(len(left),len(right)) and len(left)*len(right)<=40_000_000:
        vocabulary={token:i for i,token in enumerate(dict.fromkeys(left+right))}
        a=np.array([vocabulary[t] for t in left],np.int32)
        b=np.array([vocabulary[t] for t in right],np.int32)
        ai,bj=_weighted_lcs(a,b,np.array([len(t) for t in left],np.int32))
        optimal=dict(zip(ai,bj))
        if sum(len(left[i]) for i in optimal)>sum(len(left[i]) for i in pairs):pairs=optimal
    return pairs


def _groups(notes, *, midi=False, eligible=None):
    groups = defaultdict(list)
    for i, note in enumerate(notes):
        if eligible is not None and not eligible[i]:
            continue
        q = note.onset_beats if midi else float(note.onset)
        groups[round(q, 6)].append(i)
    times = sorted(groups)
    indices = [groups[q] for q in times]
    tokens = [tuple(sorted({notes[i].pitch for i in rows})) for rows in indices]
    return np.array(times), indices, tokens


def align_symbolic_notes(midi_notes, xml):
    ties = build_tie_mapping(xml)
    eligible = ties.is_attack & np.array([not n.grace for n in xml.notes])
    mq, mi, mt = _groups(midi_notes, midi=True)
    xq, xi, xt = _groups(xml.notes, eligible=eligible)
    pairs = exact_group_pairs(mt,xt)
    # Fill a differing chord only when the surrounding exact chord anchors
    # independently agree on its score position. Matching pitches remain required.
    if len(pairs) >= 2:
        a = np.array(sorted(pairs)); b = np.array([pairs[i] for i in a])
        guess = np.interp(mq, mq[a], xq[b])
        used = set(pairs.values())
        for i in range(int(a[0]), int(a[-1]) + 1):
            if i in pairs:
                continue
            j = int(np.searchsorted(xq, guess[i]))
            candidates = [k for k in (j - 1, j) if 0 <= k < len(xq)
                          and k not in used and abs(xq[k] - guess[i]) <= 1 / 64
                          and set(mt[i]) & set(xt[k])]
            if len(candidates) == 1:
                pairs[i] = candidates[0]
                used.add(candidates[0])
    a = np.array(sorted(pairs), dtype=int)
    b = np.array([pairs[i] for i in a], dtype=int)
    if len(a) < 2 or np.any(np.diff(b) <= 0):
        raise ValueError('Cannot establish a monotone symbolic score bridge')
    aliases, mapped_rows = {}, {}
    for i, j in zip(a, b):
        left, right = defaultdict(list), defaultdict(list)
        for k in mi[i]: left[midi_notes[k].pitch].append(k)
        for k in xi[j]: right[xml.notes[k].pitch].append(k)
        for pitch in left.keys() & right.keys():
            # Unison ambiguity is not resolved by arbitrary document order.
            if len(left[pitch]) == len(right[pitch]) == 1:
                m, x = left[pitch][0], right[pitch][0]
                aliases[midi_notes[m].note_id] = xml.notes[x].occurrence_id
                mapped_rows[midi_notes[m].note_id] = x
    coverage = len(aliases) / max(1, len(midi_notes))
    xml_coverage = len(aliases) / max(1, int(eligible.sum()))
    f1 = 2 * coverage * xml_coverage / max(1e-12, coverage + xml_coverage)
    return dict(aliases=aliases, mapped_rows=mapped_rows,
                midi_anchor_q=mq[a], xml_anchor_q=xq[b],
                midi_note_count=len(midi_notes), xml_attack_count=int(eligible.sum()),
                mapped_notes=len(aliases), midi_coverage=coverage,
                xml_coverage=xml_coverage, score=f1)


def score_supported_traversal(midi_notes, written_xml):
    """Propose repeated written measures from long exact score-pitch blocks.

    This is a hypothesis, not an accepted mapping. The caller must parse it and
    verify high bidirectional full-score coverage before using any target.
    """
    ties = build_tie_mapping(written_xml)
    mq, mi, mt = _groups(midi_notes, midi=True)
    xq, xi, xt = _groups(written_xml.notes,
        eligible=ties.is_attack & np.array([not n.grace for n in written_xml.notes]))
    pairs = {}
    intervals = [(0,len(mt))]
    for _ in range(6):
        added = 0
        for left,right in intervals:
            if right-left<4: continue
            for block in SequenceMatcher(None,mt[left:right],xt,autojunk=False).get_matching_blocks():
                if block.size<4: continue
                for k in range(block.size):
                    pairs[left+block.a+k]=block.b+k;added+=1
        if not added: break
        anchors=[-1,*sorted(pairs),len(mt)]
        intervals=[(a+1,b) for a,b in zip(anchors,anchors[1:]) if b-a>4]
    if not pairs: raise ValueError('No repeated measure route supported by symbolic pitch blocks')
    route, previous, previous_q = [], None, None
    for i,j in sorted(pairs.items()):
        measure=written_xml.notes[xi[j][0]].measure
        if previous is None:
            route.extend(range(measure+1))
        elif measure>previous:
            route.extend(range(previous+1,measure+1))
        elif measure<previous or (measure==previous and xq[j]<previous_q):
            route.append(measure)
        previous,previous_q=measure,xq[j]
    route.extend(range(previous+1,len(written_xml.measure_durations)))
    if len(route)>len(written_xml.measure_durations)*8:
        raise ValueError('Inferred route exceeds bounded repeat hypothesis')
    return tuple(route)


def build_selected_score_bridge(xml_path, midi_path):
    from features.note_score_context import extract_note_score_context, normalize_direction_offsets
    root, source_sha = _root(xml_path)
    normalized, time_audit = normalize_quantized_timing(root, prefer_preserve_durations=True)
    midi_notes, _, midi_audit = read_score_midi(midi_path)
    count = len(normalized.findall('part')[0].findall('measure'))
    candidates, errors, seen = [], [], set()

    def attempt(tree, route, name, navigation_audit=None):
        try:
            xml = read_musicxml_root(tree, source_sha, traversal=route)
            if xml.traversal_id in seen:
                return
            seen.add(xml.traversal_id)
            bridge = align_symbolic_notes(midi_notes, xml)
            candidates.append((bridge['score'], name, xml, bridge, navigation_audit))
        except Exception as exc:
            errors.append({'candidate': name, 'error': f'{type(exc).__name__}: {exc}'})

    attempt(normalized, None, 'notated_repeats')
    attempt(normalized, tuple(range(count)), 'written_order')
    needs_repeat = candidates and max(c[3]['midi_coverage'] for c in candidates)<.95 and all(
        c[3]['midi_note_count'] > 1.12*c[3]['xml_attack_count'] for c in candidates)
    if needs_repeat:
        try:
            written=read_musicxml_root(normalized,source_sha,traversal=tuple(range(count)))
            route=score_supported_traversal(midi_notes,written)
            previous_count=len(candidates)
            routes=[('score_midi_supported_repeats',route)]
            for a,b in dict.fromkeys((a,b) for a,b in zip(route,route[1:]) if b<=a):
                routes.append((f'score_supported_repeat_{a}_{b}',tuple(range(a+1))+tuple(range(b,count))))
            written_ties=build_tie_mapping(written)
            measure_counts=np.bincount([n.measure for i,n in enumerate(written.notes)
                if written_ties.is_attack[i] and not n.grace],minlength=count)
            for route_name,proposed in routes:
                estimate=int(measure_counts[list(proposed)].sum())
                if not .90*len(midi_notes)<=estimate<=1.10*len(midi_notes):
                    continue
                attempt(normalized,proposed,route_name,
                        {'policy':'score-pitch-measure-route-v1','performed_expression_used':False,
                         'minimum_midi_and_xml_attack_coverage':.94,'note_content_modified':False})
            candidates[previous_count:]=[c for c in candidates[previous_count:]
                                         if min(c[3]['midi_coverage'],c[3]['xml_coverage'])>=.94]
        except ValueError as exc:
            errors.append({'candidate':'score_midi_supported_repeats','error':str(exc)})
    if not candidates:
        try:
            recovery = prepare_navigation_recovery(normalized)
            for candidate in recovery.candidates:
                attempt(recovery.cleaned_root, candidate.traversal, candidate.name,
                        {**recovery.audit, 'selected_assumptions': candidate.assumptions})
        except ValueError as exc:
            errors.append({'candidate':'navigation_recovery', 'error':str(exc)})
    if not candidates:
        # Some exports encode an inconsistent coda symbol beside a named return.
        # We do not infer a performed jump: offer only written/barline routes and
        # accept one only if BOTH MIDI and XML attack coverage independently >=97%.
        cleaned, edits = deepcopy(normalized), []
        for parent in cleaned.iter():
            for child in list(parent):
                if child.tag in ('coda', 'segno') or (child.tag == 'words' and
                        re.search(r'\b(?:capo|segno|coda|fine)\b', ''.join(child.itertext()), re.I)):
                    edits.append({'tag':child.tag, 'text':''.join(child.itertext())})
                    parent.remove(child)
                elif child.tag == 'sound':
                    for key in ('dacapo','dalsegno','tocoda','fine','segno','coda'):
                        if key in child.attrib:
                            edits.append({'sound':key, 'value':child.attrib.pop(key)})
        audit = {'policy':'symbolic-score-supported-route-only-v1', 'edits':edits,
                 'minimum_midi_and_xml_attack_coverage':.97, 'note_content_modified':False,
                 'performed_navigation_inferred':False}
        attempt(cleaned, None, 'score_supported_barline_route', audit)
        attempt(cleaned, tuple(range(count)), 'score_supported_written_route', audit)
        candidates[:] = [c for c in candidates if min(c[3]['midi_coverage'],c[3]['xml_coverage']) >= .97]
    if not candidates:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    candidates.sort(key=lambda c: (-c[0], -c[3]['mapped_notes'], c[1]))
    _, name, xml, bridge, navigation_audit = candidates[0]
    xml, direction_audit = normalize_direction_offsets(xml)
    if time_audit or navigation_audit:
        # Repaired timing must not collide with an unmodified S cache.
        repair_id = hashlib.sha256(json.dumps([time_audit, navigation_audit], sort_keys=True).encode()).hexdigest()
        xml = replace(xml, traversal_id=hashlib.sha256((xml.traversal_id + repair_id).encode()).hexdigest())
    features = extract_note_score_context(xml)
    bridge.update(source_xml_sha256=source_sha,
                  source_midi_sha256=hashlib.sha256(midi_path.read_bytes()).hexdigest(),
                  selected_candidate=name, time_repair=time_audit,
                  navigation_repair=navigation_audit, direction_repair=direction_audit,
                  candidate_errors=errors,
                  candidate_scores=[{'name': n, 'score': b['score'], 'midi_coverage': b['midi_coverage'],
                                     'xml_coverage': b['xml_coverage']} for _, n, _, b, _ in candidates],
                  score_midi_diagnostics=midi_audit)
    return xml, features, bridge
