"""Opt-in, audited navigation candidates; original MIDI alignment must choose a route.

This module never edits an input tree or decides which repeat route was performed.
It removes parser-blocking navigation only after constructing bounded, explicit
candidate routes. Ambiguous unclosed repeat starts produce labelled hypotheses;
unresolved jump destinations are rejected. Notes and their order are unchanged.
"""

from copy import deepcopy
from dataclasses import dataclass
import html
import re
import xml.etree.ElementTree as ET


VERSION = 'musicxml-navigation-recovery-v1'
_LEGACY_NAV = re.compile(r'\b(?:d\s*\.\s*[cs]\b|da\s+capo\b|dal\s+segno\b|al\s+coda\b|fine\b)', re.I)
_DC = re.compile(r'\b(?:d\s*\.\s*c\b|da\s+capo\b)', re.I)
_DS = re.compile(r'\b(?:d\s*\.\s*s\b|dal\s+segno\b)', re.I)
_NO_REPEAT = re.compile(r'\b(?:senza\s+(?:replica|repetizione|ripetizione)|without\s+repeats?)\b', re.I)
_SECTION = re.compile(r'\b(menuetto|menutto|minuetto|scherzo|scherzino|trio)\b(?:\s+(I{1,3}|IV|V|[12])\b)?', re.I)
_NAV_ATTRIBUTES = ('dacapo', 'dalsegno', 'tocoda', 'fine', 'segno', 'coda')


class UnsupportedNavigation(ValueError):
    """The supplied notation does not define supported bounded jump candidates."""


@dataclass(frozen=True)
class NavigationCandidate:
    name: str
    traversal: tuple[int, ...]
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class NavigationRecovery:
    cleaned_root: ET.Element
    candidates: tuple[NavigationCandidate, ...]
    audit: dict


def _plain_words(node):
    # Some exports store literal HTML inside MusicXML <words>.
    return ' '.join(re.sub(r'<[^>]*>', '', html.unescape(''.join(node.itertext()))).split())


def _section_key(match):
    name = match.group(1).lower()
    if name in {'menutto', 'minuetto'}:
        name = 'menuetto'
    return name, (match.group(2) or '').lower()


def _repeat_routes(part, max_visits):
    measures = part.findall('measure')
    count = len(measures)
    forwards, backwards, endings = set(), {}, {}
    active_ending = None
    for i, measure in enumerate(measures):
        for bar in measure.findall('barline'):
            ending = bar.find('ending')
            if ending is not None:
                if ending.get('type') == 'start':
                    try:
                        active_ending = frozenset(int(x.strip()) for x in ending.get('number', '').split(','))
                    except ValueError as exc:
                        raise UnsupportedNavigation('Unsupported ending numbers') from exc
                elif ending.get('type') in {'stop', 'discontinue'}:
                    if active_ending:
                        endings[i] = active_ending
                    active_ending = None
            repeat = bar.find('repeat')
            if repeat is not None:
                if repeat.get('direction') == 'forward':
                    start = i + (bar.get('location') == 'right')
                    if start >= count:
                        raise UnsupportedNavigation('Forward repeat beyond final measure')
                    forwards.add(start)
                elif repeat.get('direction') == 'backward':
                    end = i - (bar.get('location') == 'left')
                    times = int(repeat.get('times', '2'))
                    if end < 0 or not 2 <= times <= 16:
                        raise UnsupportedNavigation('Invalid backward repeat/count')
                    if end in backwards and backwards[end] != times:
                        raise UnsupportedNavigation('Conflicting repeat counts')
                    backwards[end] = times
        if active_ending:
            endings[i] = active_ending

    stack, pairs = [], []
    for i in range(count):
        if i in forwards:
            stack.append(i)
        if i in backwards:
            start = stack.pop() if stack else 0
            pairs.append((start, i, backwards[i]))
    unclosed = tuple(stack)
    alternatives = [('marked_repeats', pairs, (() if not unclosed else
                    ('unclosed forward starts are nonoperative: ' + ','.join(map(str, unclosed)),)))]
    if unclosed:
        alternatives.append(('unclosed_to_score_end', pairs + [(s, count - 1, 2) for s in unclosed],
                             ('unclosed forward starts close at final measure: ' + ','.join(map(str, unclosed)),)))

    def expand(intervals):
        intervals = sorted(set(intervals), key=lambda x: (x[0], -x[1]))
        nested = any(a <= c <= d <= b and (a, b) != (c, d)
                     for a, b, _ in intervals for c, d, _ in intervals)
        if nested and endings:
            raise UnsupportedNavigation('Nested repeats combined with endings need an explicit reviewed route')
        if not nested:
            ends = {b: (a, n) for a, b, n in intervals}
            if len(ends) != len(intervals):
                raise UnsupportedNavigation('Conflicting repeat intervals')
            path, passes, i, active_pass = [], {}, 0, 1
            steps = 0
            while i < count:
                steps += 1
                if steps > max_visits * 2 or len(path) >= max_visits:
                    raise UnsupportedNavigation('Repeat traversal exceeds visit bound')
                if i not in endings or active_pass in endings[i]:
                    path.append(i)
                if i in ends:
                    left, total = ends[i]
                    passes[i] = passes.get(i, 0) + 1
                    if passes[i] < total:
                        active_pass = passes[i] + 1
                        i = left
                        continue
                i += 1
                if i in forwards:
                    active_pass = 1
            return tuple(path)

        # A nesting tree gives each inner repeat its own count on each outer pass.
        def sequence(left, right, contained):
            out, i = [], left
            outer = []
            for pair in contained:
                if not any(a <= pair[0] and pair[1] <= b for a, b, _ in outer):
                    outer.append(pair)
            for a, b, n in outer:
                if a < i or b > right:
                    raise UnsupportedNavigation('Crossing repeat intervals')
                out.extend(range(i, a))
                inside = [p for p in contained if p != (a, b, n) and a <= p[0] and p[1] <= b]
                phrase = sequence(a, b, inside) if inside else tuple(range(a, b + 1))
                if len(out) + n * len(phrase) > max_visits:
                    raise UnsupportedNavigation('Nested repeat traversal exceeds visit bound')
                out.extend(phrase * n)
                i = b + 1
            out.extend(range(i, right + 1))
            if len(out) > max_visits:
                raise UnsupportedNavigation('Repeat traversal exceeds visit bound')
            return tuple(out)
        return sequence(0, count - 1, intervals)

    routes = [NavigationCandidate('written_order', tuple(range(count)), ('barline repeats omitted by performer',))]
    issues = []
    for name, intervals, assumptions in alternatives:
        try:
            routes.append(NavigationCandidate(name, expand(intervals), assumptions))
        except UnsupportedNavigation as error:
            issues.append({'candidate': name, 'reason': str(error)})
    return routes, {'forward_measures': sorted(forwards), 'backward_measures': backwards,
                    'unclosed_forward_measures': list(unclosed), 'repeat_pairs': pairs,
                    'unsupported_repeat_candidates': issues}


def prepare_navigation_recovery(root: ET.Element, *, max_visits: int = 8192) -> NavigationRecovery:
    """Return a copied tree and candidate routes; none is an approved alignment.

    Callers must retain ``audit``, the original file SHA, and the selected route.
    Every candidate needs independent alignment to the original performance.
    Multiple jumps or unresolved/multiple destinations are deliberately held.
    """
    if max_visits < 1:
        raise ValueError('max_visits must be positive')
    clean = deepcopy(root)
    parts = clean.findall('part')
    if not parts:
        raise UnsupportedNavigation('Score contains no parts')
    count = len(parts[0].findall('measure'))
    if not count or count > max_visits or any(len(p.findall('measure')) != count for p in parts):
        raise UnsupportedNavigation('Invalid or mismatched measure counts')
    routes, repeat_audit = _repeat_routes(parts[0], max_visits)
    events, edits, jumps, fines, segnos, codas, tocodas = [], [], {}, set(), {}, {}, {}
    no_return_repeats = False
    requests_fine, requests_coda = False, False
    named_returns, section_headers = set(), []

    def record(kind, measure, value, source):
        events.append({'kind': kind, 'measure': measure, 'value': value, 'source': source})

    def target(table, label, measure, kind):
        old = table.get(label)
        if old is not None and old != measure:
            raise UnsupportedNavigation(f'Multiple {kind} destinations for {label!r}')
        table[label] = measure

    def jump(measure, kind, label=''):
        old = jumps.get(measure)
        entry = (kind, label)
        if old is not None and old != entry:
            # A words D.S. plus its sound attribute names the same destination.
            if old[0] == kind and (not old[1] or not label):
                entry = (kind, old[1] or label)
            else:
                raise UnsupportedNavigation('Conflicting jump instructions')
        jumps[measure] = entry

    for pi, part in enumerate(parts):
        for mi, measure in enumerate(part.findall('measure')):
            pedal_context = bool(measure.findall('.//pedal')) and any(
                any(0xE000 <= ord(ch) <= 0xF8FF for ch in ''.join(w.itertext()))
                for w in measure.findall('.//words'))
            for direction in measure.findall('direction'):
                sound = direction.find('sound')
                sound_nav = {k: sound.get(k) for k in _NAV_ATTRIBUTES if sound is not None and sound.get(k) is not None}
                for dtype in direction.findall('direction-type'):
                    for item in list(dtype):
                        if item.tag in {'segno', 'coda'}:
                            kind = item.tag
                            label = sound_nav.get(kind, kind)
                            target(segnos if kind == 'segno' else codas, label, mi, kind)
                            record(kind, mi, label, f'part[{pi}]/direction')
                            item.tag, item.text = 'words', f'navigation marker {kind}'
                            edits.append({'kind': kind, 'measure': mi, 'part': pi, 'label': label})
                        elif item.tag == 'words':
                            text = _plain_words(item)
                            expression = False
                            if _DC.search(text) or _DS.search(text):
                                kind = 'dc' if _DC.search(text) else 'ds'
                                jump(mi, kind)
                                no_return_repeats |= bool(_NO_REPEAT.search(text))
                                requests_fine |= bool(re.search(r'\bal\s+fine\b', text, re.I))
                                requests_coda |= bool(re.search(r'\b(?:al\s+coda|poi\s+la\s+coda)\b', text, re.I))
                                section = _SECTION.search(text)
                                if section:
                                    named_returns.add(_section_key(section))
                                record(kind, mi, text, f'part[{pi}]/words')
                            elif re.fullmatch(r'fine[.!]?', text, re.I):
                                fines.add(mi)
                                record('fine', mi, text, f'part[{pi}]/words')
                            elif re.search(r'\b(?:to|al)\s+coda\b', text, re.I):
                                target(tocodas, sound_nav.get('tocoda', 'coda'), mi, 'to-coda trigger')
                                record('tocoda', mi, text, f'part[{pi}]/words')
                            elif re.search(r'\bpoi\s+(?:la\s+)?coda\b', text, re.I):
                                requests_coda = True
                                record('coda_return_requirement', mi, text, f'part[{pi}]/words')
                            elif re.fullmatch(r'coda[.!]?', text, re.I):
                                target(codas, sound_nav.get('coda', 'coda'), mi, 'coda')
                                record('coda', mi, text, f'part[{pi}]/words')
                            elif _LEGACY_NAV.search(text):
                                expression = bool(re.search(r'\b(?:accelerando|rallentando|ritardando|crescendo|diminuendo)\b.*\bfine\b', text, re.I))
                                expression |= bool(re.fullmatch(r'al\s+fine[.!]?', text, re.I) and pedal_context)
                                if not expression:
                                    raise UnsupportedNavigation(f'Unclassified navigation-like words at measure {mi}: {text}')
                                record('expression_not_jump', mi, text, f'part[{pi}]/words')
                            else:
                                header = _SECTION.match(text)
                                if header and pi == 0:
                                    section_headers.append((mi, _section_key(header)))
                                continue
                            edits.append({'kind': 'words', 'measure': mi, 'part': pi, 'original': ''.join(item.itertext())})
                            attributes = dict(item.attrib)
                            item.clear()
                            item.attrib.update(attributes)
                            # Preserve accelerando/dynamics and typography. Only
                            # rephrase the legacy parser's ambiguous "fine" token.
                            replacement = re.sub(r'\bfine\b', 'end', text, flags=re.I) if expression else 'navigation instruction recorded in source audit'
                            item.tag, item.text = 'words', replacement
            for sound in measure.findall('.//sound'):
                attrs = {k: sound.get(k) for k in _NAV_ATTRIBUTES if sound.get(k) is not None}
                for kind, value in attrs.items():
                    if kind == 'dacapo' and value == 'yes':
                        jump(mi, 'dc')
                    elif kind == 'dalsegno':
                        jump(mi, 'ds', value)
                    elif kind == 'fine':
                        fines.add(mi)
                    elif kind == 'segno':
                        target(segnos, value, mi, kind)
                    elif kind == 'coda':
                        target(codas, value, mi, kind)
                    elif kind == 'tocoda':
                        target(tocodas, value, mi, 'to-coda trigger')
                    else:
                        raise UnsupportedNavigation(f'Unsupported sound navigation {kind}={value!r}')
                    record(kind, mi, value, f'part[{pi}]/sound')
                    del sound.attrib[kind]
                if attrs:
                    edits.append({'kind': 'sound', 'measure': mi, 'part': pi, 'original': attrs})

    if len(jumps) > 1:
        raise UnsupportedNavigation('Multiple jump locations require a reviewed traversal')
    candidates = [NavigationCandidate(c.name, c.traversal,
                  c.assumptions + (('notated D.C./D.S. return omitted by performer',) if jumps else ()))
                  for c in routes]
    if jumps:
        at, (kind, label) = next(iter(jumps.items()))
        if kind == 'dc':
            destination = 0
        elif label and label in segnos:
            destination = segnos[label]
        elif not label and len(set(segnos.values())) == 1:
            destination = next(iter(segnos.values()))
        else:
            raise UnsupportedNavigation('D.S. destination cannot be resolved uniquely')
        if destination >= at:
            raise UnsupportedNavigation('Jump destination is not before jump instruction')
        coda_tail = None
        if requests_coda and not (tocodas and codas):
            raise UnsupportedNavigation('Coda requested but trigger/destination is missing')
        if tocodas or codas:
            if len(tocodas) != 1:
                raise UnsupportedNavigation('Coda traversal needs one named to-coda trigger')
            coda_label, finish = next(iter(tocodas.items()))
            if coda_label in codas:
                coda_tail = codas[coda_label]
            elif len(codas) == 1 and coda_label == 'coda':
                coda_tail = next(iter(codas.values()))
            else:
                raise UnsupportedNavigation('Coda destination cannot be resolved uniquely')
            if not destination <= finish < at < coda_tail:
                raise UnsupportedNavigation('Coda/jump ordering is unsupported')
        else:
            stops = sorted(i for i in fines if destination <= i < at)
            if len(stops) > 1:
                raise UnsupportedNavigation('Multiple Fine stops require a reviewed traversal')
            if requests_fine and not stops:
                raise UnsupportedNavigation('Al Fine requested but no earlier Fine stop is present')
            finish = stops[0] if stops else at
            if not stops and named_returns:
                if len(named_returns) != 1:
                    raise UnsupportedNavigation('Multiple named return sections')
                name = next(iter(named_returns))
                starts = {mi for mi, key in section_headers if key == name and mi <= at}
                if starts != {destination}:
                    raise UnsupportedNavigation('Named return section does not match the jump destination')
                boundaries = sorted({mi for mi, _ in section_headers if destination < mi < at})
                if not boundaries:
                    raise UnsupportedNavigation('Named return section has no encoded end boundary')
                finish = boundaries[0] - 1
                record('named_return_section', destination, {'name': name, 'end': finish}, 'section headers')
            if not stops and not named_returns and at < count - 1:
                # The score has music after a bare D.C./D.S. instruction but no
                # encoded return stop. The written-order hypotheses remain
                # available; inventing where the return rejoins the tail would
                # be an unsupported jump interpretation.
                repeat_audit['unsupported_jump_return'] = (
                    'Jump precedes written tail without a Fine or coda trigger; '
                    'only written-order/barline-repeat routes offered')
                routes = []
        for route in routes:
            first = tuple(i for i in route.traversal if i <= at)
            tail = tuple(i for i in route.traversal if coda_tail is not None and i >= coda_tail)
            plain = tuple(range(destination, finish + 1))
            assumption = f'{kind} at {at} returns to {destination}, stops at {finish}'
            if coda_tail is not None:
                assumption += f', then coda {coda_tail}'
            candidates.append(NavigationCandidate(route.name + '_' + kind + '_return_without_repeats',
                                                  first + plain + tail, route.assumptions + (assumption,)))
            if not no_return_repeats:
                repeated = tuple(i for i in route.traversal if destination <= i <= finish)
                candidates.append(NavigationCandidate(route.name + '_' + kind + '_return_with_repeats',
                                                      first + repeated + tail, route.assumptions + (assumption,)))
    elif tocodas:
        raise UnsupportedNavigation('To-coda trigger without a return instruction')

    unique, seen = [], set()
    for candidate in candidates:
        path = candidate.traversal
        if not path or path[0] != 0 or len(path) > max_visits or any(not 0 <= i < count for i in path):
            raise UnsupportedNavigation('Candidate traversal violates measure/visit bounds')
        if path not in seen:
            unique.append(candidate)
            seen.add(path)
    audit = {'version': VERSION, 'selection': 'original-performance-alignment-required',
             'measure_count': count, 'events': events, 'edits': edits, 'repeat_structure': repeat_audit,
             'candidates': [{'name': c.name, 'measure_visits': len(c.traversal),
                             'assumptions': list(c.assumptions)} for c in unique],
             'note_content_modified': False, 'source_tree_modified': False}
    return NavigationRecovery(clean, tuple(unique), audit)
