"""Check all exported MIDI files and independently decode a sample with pretty_midi."""
from collections import Counter
import json
from pathlib import Path
import sys

import mido
import numpy as np
import pandas as pd
import pretty_midi


def verify(root, all_clips=False):
    segments = pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str})
    errors = []
    for row in segments.to_dict('records'):
        path = Path(row['clip_path'])
        counts, last_pedal, now, n = Counter(), {}, 0., 0
        for msg in mido.MidiFile(path):
            now += msg.time
            if msg.type == 'note_on' and msg.velocity:
                counts[(msg.channel,msg.note)] += 1
                n += 1
            elif msg.type in ('note_on','note_off'):
                counts[(msg.channel,msg.note)] -= 1
                if counts[(msg.channel,msg.note)] < 0:
                    errors.append(dict(clip=str(path),reason='orphan_note_off'))
            elif msg.type == 'control_change' and msg.control == 64:
                last_pedal[msg.channel] = msg.value
        if any(counts.values()) or any(v >= 64 for v in last_pedal.values()):
            errors.append(dict(clip=str(path),reason='hanging_notes_or_pedal'))
        if abs(now-row['duration_seconds']) > .002 or n != row['note_count']:
            errors.append(dict(clip=str(path),reason='duration_or_note_count_mismatch'))
    # Deterministic sample: one performance per composition, every segment from it.
    selected = segments.groupby('composition_id',sort=True).perf_id.first()
    sample = segments if all_clips else segments[segments.perf_id.isin(selected)]
    checked_notes = 0
    for source, group in sample.groupby('source_midi'):
        original = pretty_midi.PrettyMIDI(source)
        for row in group.to_dict('records'):
            a,b = row['start_seconds'],row['end_seconds']
            expected = sorted((i.program,i.is_drum,n.pitch,n.velocity,
                               round(max(n.start,a)-a,3),round(min(n.end,b)-a,3))
                              for i in original.instruments for n in i.notes
                              if n.start < b and n.end > a and round(min(n.end,b)-a,3)>round(max(n.start,a)-a,3))
            clip = pretty_midi.PrettyMIDI(row['clip_path'])
            actual = sorted((i.program,i.is_drum,n.pitch,n.velocity,round(n.start,3),round(n.end,3))
                            for i in clip.instruments for n in i.notes)
            same = len(expected)==len(actual)
            if same:
                same = all(x[:4]==y[:4] and np.allclose(x[4:],y[4:],atol=.0011,rtol=0)
                           for x,y in zip(expected,actual))
            if not same:
                errors.append(dict(clip=row['clip_path'],reason='independent_note_content_mismatch',
                                   expected_notes=len(expected),actual_notes=len(actual)))
            checked_notes += len(actual)
    result = dict(all_clips_checked=len(segments), independent_all_clips=all_clips, independent_sample_clips=len(sample),
                  independent_sample_performances=sample.perf_id.nunique(),
                  independent_sample_notes=checked_notes, errors=len(errors), error_details=errors)
    (root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    verify(Path(sys.argv[1]),all_clips='--all' in sys.argv[2:])
