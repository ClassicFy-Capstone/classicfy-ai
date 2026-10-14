"""One train-only optimizer step per control/direction; no fitted model or ranking."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from data import GroupCatalog, load_mapping, sha
from model import ConditionalMetricEncoder, objective, previous


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = Path(__file__).resolve().parent
    mapping = load_mapping(); reports = []
    for view in (0, 1):
        catalog = GroupCatalog(view, mapping)
        batch = catalog.sample_train(np.random.default_rng(20261014 + view))
        train_owners = set(catalog.data.gids.tolist())
        assert set(batch['owners'].tolist()) <= train_owners
        assert not set(batch['owners'].tolist()) & set(catalog.data.qids.tolist())
        for group in batch['groups'].unique():
            rows = batch['groups'] == group
            owners = batch['owners'][rows]
            assert len(owners.unique()) == len(owners)
            context = batch['context'][rows]
            assert torch.allclose(context, context[:1].expand_as(context), atol=1e-6, rtol=0)
        for fused in (False, True):
            for metric in (False, True):
                torch.manual_seed(20261014)
                model = ConditionalMetricEncoder(fused)
                generator = torch.Generator().manual_seed(20261015 + view)
                a = previous.masks(len(batch['x']), generator)
                b = previous.masks(len(batch['x']), generator)
                optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                loss, values, counts = objective(model, batch, a, b, metric)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step()
                model.eval()
                with torch.no_grad():
                    z = model.encode(batch['x'], batch['context'])
                assert z.shape == (len(batch['x']), 32) and torch.isfinite(z).all()
                assert torch.allclose(z.norm(dim=1), torch.ones(len(z)), atol=1e-6)
                reports.append(dict(view=view, fused_context=fused, metric=metric, batch_rows=len(z),
                                    parameters=sum(p.numel() for p in model.parameters()),
                                    **counts, before_step=values, train_owners=sorted(set(batch['owners'].tolist()))))
        (out / f'view{view}_amplitude_scales.json').write_text(json.dumps(catalog.amplitude_scales, indent=2) + '\n')
    code = {name: sha(out / name) for name in ('data.py', 'model.py', 'smoke.py', 'protocol.json')}
    result = dict(status='passed', scope='8 independent one-step train-only smoke checks; weights discarded; no validation/test ranking or checkpoint selection',
                  code_sha256=code, results=reports)
    (out / 'smoke_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(reports), 'parameters': reports[0]['parameters'],
                      'batches': [r['batch_rows'] for r in reports if not r['fused_context'] and r['metric']]}))


if __name__ == '__main__':
    main()
