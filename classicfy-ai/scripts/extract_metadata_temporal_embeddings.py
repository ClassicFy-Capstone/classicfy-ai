"""Re-encode saved #13 models with their original, verified source and inputs.

No training or checkpoint selection. Large exports stay in datasets/, outside Git.
The detached source snapshot is temporary; the working branch is not changed.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20261006, 20261007, 20261008)
SOURCE_REF = 'feat/#13-temporal-autoencoder-experiments'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'datasets')
    parser.add_argument('--source-ref', default=SOURCE_REF)
    parser.add_argument('--out', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.source_ref], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'archive', commit, 'classicfy-ai/src', 'classicfy-ai/scripts'], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='classicfy-metadata-source-') as directory:
        snapshot = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(source)) as archive:
            archive.extractall(snapshot, filter='data')
        # Original prepare() reads Git objects to check the original collector.
        # This pointer is used only for read-only git show/ls-tree operations.
        git_dir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip()
        (snapshot / '.git').write_text(f'gitdir: {git_dir}\n')
        sys.path[:0] = [str(snapshot / 'classicfy-ai/scripts'), str(snapshot / 'classicfy-ai/src')]
        import numpy as np
        import torch
        from compare_sequence_autoencoders import prepare
        from embedding.data import WindowDataset
        from embedding.training import extract_embeddings
        from embedding.extended_experiments import load as load_extended
        from embedding.sequence_models import load_sequence_model
        from train_temporal_ablation import load as load_cnn

        torch.set_num_threads(2)
        sequence_root = args.data / 'sequence_autoencoders_run01'
        extended_root = args.data / 'extended_sequences_run01'
        sequence_protocol = json.loads((sequence_root / 'protocol.json').read_text())
        extended_protocol = json.loads((extended_root / 'protocol.json').read_text())
        for protocol, field in ((sequence_protocol, 'source_sha256'), (extended_protocol, 'sources')):
            for name, expected in protocol[field].items():
                if sha(snapshot / 'classicfy-ai' / name) != expected:
                    raise ValueError(f'Saved model source mismatch: {name}')
            if sha(args.data / 'temporal_embedding_cluster_run01/dataset.json') != protocol['reference_dataset_sha256']:
                raise ValueError('Saved reference manifest changed')
        stored, mc, config, sequences, *_ = prepare(SimpleNamespace(
            reference=args.data / 'temporal_embedding_cluster_run01',
            cache=args.data / 'feature_normalization_cluster_raw.npz',
            asap_root=args.data / 'ASAP', nasap_root=args.data / 'nASAP'))
        windows = WindowDataset(sequences, window_size=mc.window_size, stride=stored['stride'])
        args.out.mkdir(parents=True, exist_ok=True)
        exports, identities = [], None
        for stage, arms in (('best20', ('cnn', 'bilstm', 'transformer')),
                            ('best_extended', ('cnn64', 'bilstm64', 'transformer64', 'bilstm128', 'bilstm256'))):
            for arm in arms:
                for seed in SEEDS:
                    if stage == 'best20':
                        if arm == 'cnn':
                            # Extended trajectory preserves the first-20 CNN;
                            # its best20 artifact also has a recorded checksum.
                            run = extended_root / 'cnn64' / str(seed)
                            checkpoint = run / 'best20.pt'
                            load, checkpoint_field = load_extended, 'best20'
                        else:
                            run = sequence_root / arm / str(seed)
                            checkpoint = run / 'best_value.pt'
                            load, checkpoint_field = load_sequence_model, 'best_value'
                    else:
                        run = extended_root / arm / str(seed)
                        checkpoint = run / 'best.pt'
                        load, checkpoint_field = load_extended, 'best'
                    training = json.loads((run / 'training.json').read_text())
                    if sha(checkpoint) != training['checkpoint_sha256'][checkpoint_field]:
                        raise ValueError(f'Checkpoint changed: {checkpoint}')
                    model, saved = load(checkpoint)
                    rows, vectors, _ = extract_embeddings(model, windows, batch_size=config.batch_size)
                    if identities is None:
                        identities = rows
                    elif rows != identities:
                        raise ValueError('Inference identities changed between models')
                    error = None
                    archive_sha = None
                    if stage == 'best20':
                        old = sequence_root / f'{arm}_{seed}_embeddings.npz'
                        with np.load(old, allow_pickle=False) as z:
                            if list(z['keys']) != [r['key'] for r in rows] or z['vectors'].shape != vectors.shape:
                                raise ValueError('Archived export identities/shapes differ')
                            error = float(np.max(np.abs(z['vectors'] - vectors)))
                            if error > 1e-5:
                                raise ValueError(f'Archive re-encoding differs: {arm} {seed}: {error}')
                        archive_sha = sha(old)
                    name = f'{arm}_{seed}_{stage}.npz'
                    np.savez_compressed(args.out / name, vectors=vectors, keys=np.array([r['key'] for r in rows]))
                    exports.append(dict(file=name, arm=arm, seed=seed, stage=stage,
                        dimensions=vectors.shape[1], performances=len(rows), windows=len(windows),
                        selected_epoch=saved.get('epoch'), checkpoint=str(checkpoint.relative_to(args.data)),
                        checkpoint_sha256=sha(checkpoint), training_sha256=sha(run / 'training.json'),
                        export_sha256=sha(args.out / name), archived_export_sha256=archive_sha,
                        maximum_archive_reencoding_error=error))
                    print(name, vectors.shape, 'archive error', error, flush=True)
        manifest = dict(source_commit=commit, source_reference=args.source_ref, torch_version=str(torch.__version__),
            source_archive_sha256=hashlib.sha256(source).hexdigest(),
            cache_sha256=sha(args.data / 'feature_normalization_cluster_raw.npz'),
            reference_manifest_sha256=sha(args.data / 'temporal_embedding_cluster_run01/dataset.json'),
            sequence_protocol_sha256=sha(sequence_root / 'protocol.json'),
            extended_protocol_sha256=sha(extended_root / 'protocol.json'),
            source_provenance=stored['source_provenance'], identities=identities, exports=exports,
            inference='Original valid masks and 64-beat/stride-32 windows; mean+population SD of window latent vectors; no synthetic masking',
            selection='Saved validation-selected checkpoints; no training or metadata-driven selection',
            code_sha256=sha(__file__))
        (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
