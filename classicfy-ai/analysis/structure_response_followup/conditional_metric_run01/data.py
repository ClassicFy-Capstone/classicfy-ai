"""Audit exact score-coordinate groups; sample train-only, owner-distinct batches."""
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import csv
import gzip
import hashlib
import importlib.util
import json
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'diagnostics/response_model'
REPO = WORK / 'capstone-2026-2-classicfy'
spec = importlib.util.spec_from_file_location('frozen_response_data', PREVIOUS / 'data.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_token(row):
    """Score identity excludes observed performance timing and values."""
    if row['status'] != 'coordinate_verified' or row['valid'] != 'True':
        raise ValueError('Unverified score interval')
    a, b = float(row['written_start_quarters']), float(row['written_stop_quarters'])
    if not np.isfinite([a, b]).all() or b <= a:
        raise ValueError('Invalid written score interval')
    return (int(row['visit_index']), int(row['measure_index']), round(a * 1e8), round(b * 1e8))


@dataclass(frozen=True)
class Passage:
    role: str
    family: str
    piece: str
    grid: str
    kind: int
    signature: str
    rows: tuple


class GroupCatalog:
    def __init__(self, view, mapping):
        self.view = view
        self.data = previous.ViewData(view)
        d = self.data
        candidates = defaultdict(list)
        self.rejected = Counter()
        for j, (owner, kind, span) in enumerate(zip(d.arrays['owner'], d.arrays['type'], d.arrays['spans'])):
            if kind == 0:  # F remains a previous baseline, not a new musical group.
                continue
            meta = d.metadata[owner]
            try:
                tokens = tuple(mapping[int(owner)][k] for k in range(*span))
                if any(t is None for t in tokens):
                    raise ValueError('Unverified interval in passage')
            except (KeyError, ValueError):
                self.rejected['unverified_rows'] += 1
                continue
            signature = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            identity = (meta['evaluation_role'], meta['family_id'], meta['piece'], meta['grid'], int(kind), signature)
            candidates[identity].append(j)
        self.groups = []
        for identity, js in sorted(candidates.items()):
            owners = d.arrays['owner'][js]
            if len(set(owners.tolist())) != len(js):
                self.rejected['ambiguous_owner_groups'] += 1
                continue
            valid = d.arrays['context_valid'][js]
            context = d.arrays['context'][js]
            if not np.array_equal(valid, np.broadcast_to(valid[:1], valid.shape)) or not np.allclose(context, context[:1], atol=1e-6, rtol=0):
                self.rejected['context_mismatch_groups'] += 1
                continue
            self.groups.append(Passage(*identity, tuple(js)))
        self.by_role = {role: [g for g in self.groups if g.role == role and len(g.rows) >= 2]
                        for role in ('train', 'validation', 'test')}
        # Derived amplitude scale fits only train/search view, before any mask.
        dx = np.diff(d.arrays['x'][:, [0, 2]], axis=2)
        self.logamp = np.log1p(np.sqrt(np.mean(dx.astype(float) ** 2, axis=2)))
        ids = d.roles['train']; w = d.role_weights['train']
        mean = w @ self.logamp[ids]
        sd = np.sqrt(np.maximum(w @ (self.logamp[ids] - mean) ** 2, 0))
        self.amplitude_scales = dict(mean=mean.tolist(), sd=sd.tolist(), train_owners=d.gids.tolist())
        self.targets = dict(d.targets)
        self.targets['lag'] = torch.tensor(d.arrays['lag'], dtype=torch.float32)  # Raw [-1,1], not previous z-score.
        self.targets['amplitude'] = torch.tensor(np.tanh((self.logamp - mean) / np.maximum(sd, 1e-8)), dtype=torch.float32)
        self.targets['amplitude_mask'] = torch.tensor(np.broadcast_to(sd > 1e-8, self.logamp.shape).copy())

    def group_weights(self, role):
        """Equal family, work/grid, P/H, H metric, and passage importance."""
        groups = self.by_role[role]
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for i, g in enumerate(groups):
            category = 'P' if g.kind == 1 else 'H'
            tree[g.family][g.piece, g.grid][category][g.kind].append(i)
        weights = np.zeros(len(groups))
        for works in tree.values():
            for categories in works.values():
                for kinds in categories.values():
                    for js in kinds.values():
                        weights[js] = 1 / (len(tree) * len(works) * len(categories) * len(kinds) * len(js))
        return weights

    def sample_train(self, rng, passages=8, recordings=8):
        groups = self.by_role['train']
        if not groups or passages < 1 or recordings < 2:
            raise ValueError('Need train groups, at least one passage, and >=2 owners per passage')
        selected = rng.choice(len(groups), passages, replace=True, p=self.group_weights('train'))
        indices, group_ids = [], []
        for batch_id, index in enumerate(selected):
            g = groups[index]
            rows = rng.choice(g.rows, min(recordings, len(g.rows)), replace=False)
            indices.extend(rows.tolist()); group_ids.extend([batch_id] * len(rows))
        return self.batch(indices, group_ids)

    def batch(self, indices, group_ids):
        ii = np.asarray(indices, int)
        return dict(x=self.data.x[ii], context=self.data.context[ii],
                    targets={k: v[ii] for k, v in self.targets.items()},
                    owners=torch.tensor(self.data.arrays['owner'][ii]),
                    groups=torch.tensor(group_ids), indices=ii, view=self.view)


def load_mapping():
    manifest = json.loads((PREVIOUS / 'data_manifest.json').read_text())
    assert sha(PREVIOUS / 'examples.npz') == manifest['examples_sha256']
    map_path = WORK / 'datasets/score_structure_full_run01/beat_map.csv.gz'
    assert sha(map_path) == manifest['map_sha256']
    with np.load(PREVIOUS / 'examples.npz') as z:
        keys = z['keys'].tolist(); ids = z['eligible'].tolist()
    with (REPO / 'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:
        metadata = {r['key']: r for r in csv.DictReader(f)}
    with (REPO / 'classicfy-ai/analysis/phrase_response/phrase_hypotheses.csv').open() as f:
        parts = {r['piece']: r['part_id'] for r in csv.DictReader(f)}
    owners = {keys[i]: i for i in ids}; result = defaultdict(dict)
    with gzip.open(map_path, 'rt') as f:
        for row in csv.DictReader(f):
            if row['key'] not in owners or row['part_id'] != parts[metadata[row['key']]['piece']]:
                continue
            i, j = owners[row['key']], int(row['index'])
            if j in result[i]:
                raise ValueError('Ambiguous primary-part interval')
            try:
                result[i][j] = coordinate_token(row)
            except ValueError:
                result[i][j] = None
    return result


def audit():
    mapping = load_mapping(); rows = []; summary = []
    for view in (0, 1):
        catalog = GroupCatalog(view, mapping)
        for g in catalog.groups:
            rows.append(dict(view=view, role=g.role, family=g.family, piece=g.piece, grid=g.grid,
                             kind=previous.KINDS[g.kind], signature=g.signature, recordings=len(g.rows),
                             indices=json.dumps(g.rows), owners=json.dumps(catalog.data.arrays['owner'][list(g.rows)].tolist())))
        for role in ('train', 'validation', 'test'):
            for category in ('P', 'H'):
                groups = [g for g in catalog.by_role[role] if (g.kind == 1) == (category == 'P')]
                summary.append(dict(view=view, role=role, category=category, groups=len(groups),
                                    curves=sum(len(g.rows) for g in groups),
                                    min_recordings=min((len(g.rows) for g in groups), default=0),
                                    max_recordings=max((len(g.rows) for g in groups), default=0),
                                    groups_with_three_or_more=sum(len(g.rows) >= 3 for g in groups)))
        print('View', view, 'rejected:', dict(catalog.rejected), flush=True)
    with (OUT / 'passages.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='passed_coordinate_and_context_audit', counts=summary,
                  examples_sha256=sha(PREVIOUS / 'examples.npz'),
                  map_sha256=sha(WORK / 'datasets/score_structure_full_run01/beat_map.csv.gz'),
                  passages_sha256=sha(OUT / 'passages.csv'),
                  note='Audit uses score identities/support only; no test response pair targets or retrieval scores computed.')
    (OUT / 'alignment_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    audit()
