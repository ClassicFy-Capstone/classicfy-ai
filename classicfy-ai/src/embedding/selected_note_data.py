"""Memory-mapped, complete selected-clip inputs for the existing note AE."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def connected_splits(records, seed=2026):
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(records):
        for field in ('composition_id', 'source_sha256', 'source_xml_sha256', 'score_midi_sha256'):
            key = field, row[field]
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    groups = sorted(components.values(), key=lambda ids:min(records[i]['perf_id'] for i in ids))
    order = np.random.default_rng(seed).permutation(len(groups))
    train, validation = int(.7 * len(groups)), int(.15 * len(groups))
    split = [''] * len(records)
    for position, group in enumerate(order):
        label = 'train' if position < train else 'validation' if position < train+validation else 'test'
        for i in groups[group]: split[i] = label
    for field in ('composition_id', 'source_sha256', 'source_xml_sha256', 'score_midi_sha256'):
        observed = defaultdict(set)
        for row, label in zip(records, split): observed[row[field]].add(label)
        if any(len(values)>1 for values in observed.values()):
            raise AssertionError('Split leakage: '+field)
    return split, len(groups)


def pack_corpus(prepared, destination, *, max_notes=256, expected_clips=5082):
    prepared, destination = Path(prepared), Path(destination)
    reports = sorted(prepared.glob('*.json'))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in reports]
    if len(records) != expected_clips or any(r['status']!='passed' for r in records):
        raise ValueError('Every selected clip must pass preparation before packing')
    if len({r['perf_id'] for r in records}) != expected_clips:
        raise ValueError('Duplicate performance IDs')
    contract = fingerprint({'records':[(r['perf_id'],r['cache_sha256']) for r in records], 'max_notes':max_notes})
    manifest_path = destination/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['id'] != contract: raise ValueError('Existing packed corpus has different inputs')
        return old
    destination.mkdir(parents=True, exist_ok=True)
    total = sum(r['notes'] for r in records)
    arrays = {}
    layout = {'numeric':(np.float32,17), 'categorical':(np.int16,14), 'flags':(np.bool_,17),
              'field_known':(np.bool_,48), 'R':(np.float32,11), 'M':(np.bool_,11),
              'beat_id':(np.int32,None), 'note_to_onset':(np.int32,None),
              'note_q':(np.float64,None), 'original_score_rows':(np.int32,None)}
    for name,(dtype,width) in layout.items():
        arrays[name] = np.lib.format.open_memmap(destination/(name+'.npy'), mode='w+', dtype=dtype,
                                               shape=(total,width) if width else (total,))
    splits, group_count = connected_splits(records)
    clips, windows, offset = [], [], 0
    for ci, (record, split) in enumerate(zip(records, splits)):
        if file_hash(record['cache_path']) != record['cache_sha256']:
            raise ValueError('Prepared clip hash mismatch')
        with np.load(record['cache_path'], allow_pickle=False) as z:
            n = len(z['R'])
            if n != record['notes'] or not z['M'].any(): raise ValueError('Invalid clip observation count')
            for name in layout:
                values = z['onset_q'][z['note_to_onset']] if name=='note_q' else z[name]
                arrays[name][offset:offset+n] = values
            onset_change=np.diff(z['note_to_onset'])!=0
            beat_change=(np.diff(z['beat_id'])!=0)|(z['beat_id'][1:]<0)
            starts = np.r_[0, np.flatnonzero(onset_change&beat_change)+1, n]
            left, owned = 0, []
            while left < n:
                eligible = starts[(starts > left) & (starts <= left+max_notes)]
                right = int(eligible[-1] if len(eligible) else starts[starts>left][0])
                valid = bool(z['M'][left:right].any())
                window = dict(clip_index=ci, start=offset+left, stop=offset+right, N=right-left,
                              observed=valid, split=split)
                windows.append(window); owned.append(window); left=right
            count = sum(w['observed'] for w in owned)
            for window in owned: window['clip_weight'] = 1/count if window['observed'] else 0.
            if sum(w['N'] for w in owned)!=n: raise AssertionError('Clip tail dropped')
        clips.append({**record, 'split':split, 'start':offset, 'stop':offset+n,
                      'window_count':len(owned), 'observed_windows':count})
        offset += n
    for array in arrays.values(): array.flush()
    manifest = dict(id=contract, schema='selected-musicxml-note-ae-v1', clips=clips, windows=windows,
        total_notes=total, max_notes=max_notes, connected_components=group_count,
        split_counts=dict(Counter(splits)), no_dropped_clips=True, no_dropped_note_rows=True,
        split_policy='work + original MIDI hash + XML hash + score MIDI hash connected components',
        files={name:file_hash(destination/(name+'.npy')) for name in arrays})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    return manifest


class SelectedNoteCorpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        self.arrays = {name:np.load(self.path/(name+'.npy'),mmap_mode='r') for name in self.manifest['files']}

    def fit_scalers(self, *, all_data=False):
        s_sum,s_sq,s_count = (np.zeros(17) for _ in range(3))
        r_sum,r_sq,r_count = (np.zeros(11) for _ in range(3))
        seen, fit_ids = {}, []
        for clip in self.manifest['clips']:
            if not all_data and clip['split']!='train': continue
            fit_ids.append(clip['perf_id'])
            section = slice(clip['start'],clip['stop'])
            a = self.arrays
            r,mask = a['R'][section].astype(float),a['M'][section].copy()
            valid = np.flatnonzero(mask[:,0])
            if len(valid):
                _, first = np.unique(a['beat_id'][section][valid],return_index=True)
                mask[:,0]=False;mask[valid[first],0]=True
            r_sum+=(r*mask).sum(0);r_sq+=(r*r*mask).sum(0);r_count+=mask.sum(0)
            key=(clip['source_xml_sha256'],clip['traversal_id'])
            indices=a['original_score_rows'][section]
            previous=seen.get(key,np.zeros(0,bool))
            if len(previous)<=indices.max(): previous=np.pad(previous,(0,int(indices.max()+1-len(previous))))
            unique=~previous[indices];previous[indices]=True;seen[key]=previous
            values=a['numeric'][section][unique].astype(float)
            known=a['field_known'][section][unique,:17]
            s_sum+=(values*known).sum(0);s_sq+=(values*values*known).sum(0);s_count+=known.sum(0)
        def stats(sums,squares,counts):
            mean=sums/np.maximum(counts,1)
            std=np.sqrt(np.maximum(0,squares/np.maximum(counts,1)-mean*mean))
            return mean.tolist(),np.where(std>1e-6,std,1.).tolist()
        sm,ss=stats(s_sum,s_sq,s_count);rm,rs=stats(r_sum,r_sq,r_count)
        result=dict(S_mean=sm,S_scale=ss,R_mean=rm,R_scale=rs,trained_channels=(r_count>0).tolist(),
                    S_counts=s_count.tolist(),R_counts=r_count.tolist(),fit_ids=fit_ids,
                    corpus_id=self.manifest['id'],scope='all_5082_refit' if all_data else 'train_only')
        result['id']=fingerprint(result)
        return result

    def items(self, split=None):
        return [w for w in self.manifest['windows'] if split is None or w['split']==split]

    def collate(self, items, scaler):
        b,n=len(items),max(w['N'] for w in items)
        widths=[]
        for w in items:
            groups=self.arrays['note_to_onset'][w['start']:w['stop']]
            widths.append(int(groups[-1]-groups[0]+1))
        g=max(widths)
        data={name:np.zeros((b,n,width),dtype) for name,width,dtype in
              [('numeric',17,np.float32),('categorical',14,np.int64),('flags',17,np.float32),
               ('field_known',48,bool),('R',11,np.float32),('M',11,bool)]}
        data.update(note_mask=np.zeros((b,n),bool),onset_mask=np.zeros((b,g),bool),
                    note_to_onset=np.zeros((b,n),np.int64),delta_q=np.zeros((b,g),np.float32),
                    beat_id=np.full((b,n),-1,np.int64))
        for i,w in enumerate(items):
            section=slice(w['start'],w['stop']);k=w['N'];a=self.arrays
            for name in ('categorical','flags','field_known'):data[name][i,:k]=a[name][section]
            mask=a['M'][section]&np.array(scaler['trained_channels'])[None,:]
            data['M'][i,:k]=mask
            data['R'][i,:k]=np.where(mask,(a['R'][section]-scaler['R_mean'])/scaler['R_scale'],0)
            data['numeric'][i,:k]=np.where(a['field_known'][section,:17],
                (a['numeric'][section]-scaler['S_mean'])/scaler['S_scale'],0)
            groups=a['note_to_onset'][section];groups=groups-groups[0]
            data['note_to_onset'][i,:k]=groups
            _,first=np.unique(groups,return_index=True)
            data['delta_q'][i,1:widths[i]]=np.diff(a['note_q'][section][first])
            data['beat_id'][i,:k]=a['beat_id'][section]
            data['note_mask'][i,:k]=True;data['onset_mask'][i,:widths[i]]=True
        result={k:torch.from_numpy(v) for k,v in data.items()}
        result.update(P=result['note_mask'],lengths=result['note_mask'].sum(1),
            R_state='train-channel-scaled-once',R_scaler={'fingerprint':scaler['id']},score_scale_id=scaler['id'],
            clip_weight=torch.tensor([w['clip_weight'] for w in items],dtype=torch.float32),
            clip_indices=torch.tensor([w['clip_index'] for w in items]),windows=items)
        return result
