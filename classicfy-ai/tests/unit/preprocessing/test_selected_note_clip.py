import hashlib
from pathlib import Path
import pickle

import mido
import numpy as np

from preprocessing.selected_score_bridge import build_selected_score_bridge, exact_group_pairs
from preprocessing.selected_note_clip import prepare_clip


def write_midi(path, *, performance=False, outside_velocity=110):
    file=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack();file.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',tempo=500000,time=0))
    events=[]
    for i,pitch in enumerate([60,62,64,65]):
        velocity=[20,50,80,outside_velocity][i] if performance else 64
        events.append((480*i,1,mido.Message('note_on',note=pitch,velocity=velocity)))
        events.append((480*i+(384 if performance else 480),0,mido.Message('note_off',note=pitch,velocity=0)))
    if performance:
        for tick,value in [(0,0),(240,100),(1248,30)]:
            events.append((tick,0,mido.Message('control_change',control=64,value=value)))
    previous=0
    for tick,_,message in sorted(events,key=lambda x:(x[0],x[1])):
        track.append(message.copy(time=tick-previous));previous=tick
    file.save(path)


def test_original_pedal_carry_and_clip_end_masks(tmp_path):
    xml=tmp_path/'score.musicxml'
    notes=''.join(f'<note><pitch><step>{p}</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note>' for p in ['C','D','E','F'])
    xml.write_text('<score-partwise><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>0</fifths></key>'
        '<time><beats>4</beats><beat-type>4</beat-type></time></attributes>'+notes+'</measure></part></score-partwise>')
    score=tmp_path/'score.mid';performance=tmp_path/'performance.mid'
    write_midi(score);write_midi(performance,performance=True)
    artifacts=build_selected_score_bridge(xml,score)
    bridge=tmp_path/'bridge.pickle';bridge.write_bytes(pickle.dumps(artifacts))
    alignment=tmp_path/'alignment.npz'
    np.savez(alignment,score_ids=np.array(['0','1','2','3']),performance_ids=np.array(['0','1','2','3']),
        score_onsets=np.arange(4.),score_offsets=np.arange(1.,5.),performance_onsets=np.arange(4.)*.5,
        performance_offsets=np.arange(4.)*.5+.4,pitch=np.array([60,62,64,65]),
        score_beats=np.arange(5.),performance_beats=np.arange(5.)*.5)
    row=dict(perf_id='test',composition_id='work',start_seconds='.5',end_seconds='1.25',
             local_score=str(score),source_sha256=hashlib.sha256(performance.read_bytes()).hexdigest())
    data,report=prepare_clip(row,bridge,performance,alignment)
    assert len(data['R'])==2
    np.testing.assert_allclose(data['R'][:,2],[50/127,80/127])
    np.testing.assert_allclose(data['R'][:,4],[100/127,100/127])  # CC before clip is carried in.
    assert data['M'][0,3] and not data['M'][1,3]  # The second key release is outside clip.
    assert not data['M'][1,[0,5,6,7,8,9]].any()
    write_midi(performance,performance=True,outside_velocity=5)
    row['source_sha256']=hashlib.sha256(performance.read_bytes()).hexdigest()
    changed,_=prepare_clip(row,bridge,performance,alignment)
    np.testing.assert_array_equal(data['M'],changed['M'])
    np.testing.assert_allclose(data['R'][data['M']],changed['R'][changed['M']])


def test_pitch_sequence_alignment_keeps_insertions_unknown():
    left=[(60,),(62,),(64,),(80,),(65,),(67,)]
    right=[(60,),(62,),(64,),(65,),(67,)]
    pairs=exact_group_pairs(left,right)
    assert 3 not in pairs
    assert [pairs[k] for k in sorted(pairs)]==list(range(5))
    assert all(left[i]==right[j] for i,j in pairs.items())


def test_score_midi_repeat_is_verified_before_creating_xml_occurrences(tmp_path):
    xml=tmp_path/'linear.musicxml'
    measures=[]
    pitches=[60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86]
    names={0:'C',2:'D',4:'E',5:'F',7:'G',9:'A',11:'B'}
    for mi in range(4):
        notes=''.join(f'<note><pitch><step>{names[p%12]}</step><octave>{p//12-1}</octave></pitch>'
            '<duration>1</duration><voice>1</voice><type>quarter</type></note>' for p in pitches[mi*4:mi*4+4])
        measures.append(f'<measure number="{mi+1}"><attributes><divisions>1</divisions>'
            '<time><beats>4</beats><beat-type>4</beat-type></time></attributes>'+notes+'</measure>')
    xml.write_text('<score-partwise><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
                   '<part id="P1">'+''.join(measures)+'</part></score-partwise>')
    file=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();file.tracks.append(track)
    for pitch in pitches[:8]+pitches:
        track.append(mido.Message('note_on',note=pitch,velocity=64,time=0))
        track.append(mido.Message('note_off',note=pitch,velocity=0,time=480))
    midi=tmp_path/'repeated.mid';file.save(midi)
    score,features,bridge=build_selected_score_bridge(xml,midi)
    assert tuple(v.measure for v in score.visits)==(0,1,0,1,2,3)
    assert bridge['midi_coverage']==bridge['xml_coverage']==1
    assert len(set(bridge['aliases'].values()))==24
