"""Train, export or diagnose the eight-bar conditional note AE; no g updates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.note_ae_data import FixedNoteTriplets, NoteAECorpus
from embedding.note_ae_training import (
    NoteAETrainingConfig, export_note_embeddings, latent_ablation, load_note_ae_model, train_note_ae,
)
from embedding.note_autoencoder import NoteAEConfig
from embedding.note_score_encoder import NoteScoreEncoderConfig
from embedding.style_profile_data import load_style_artifact
from features.note_performance_normalization import load_note_performance_scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'export', 'diagnose'):
        p = commands.add_parser(name)
        p.add_argument('--windows', type=Path, required=True)
        p.add_argument('--validation-windows', type=Path)
        p.add_argument('--out', type=Path, required=True)
        if name == 'train':
            p.add_argument('--config', type=Path)
            p.add_argument('--r-scaler', type=Path)
            p.add_argument('--resume', type=Path)
            for field in ('triplets', 'g-profiles', 'g-scaler', 'g-calibration'):
                p.add_argument('--' + field, type=Path)
        else:
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
            p.add_argument('--batch-size', type=int, default=4)
            p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    windows = load_style_artifact(args.windows, 'windows')
    val = load_style_artifact(args.validation_windows, 'windows') if args.validation_windows else None
    if args.command == 'train':
        supplied = [args.triplets, args.g_profiles, args.g_scaler, args.g_calibration]
        if any(supplied) and not all(supplied):
            parser.error('Triplet training requires --triplets, --g-profiles, --g-scaler and --g-calibration together')
        raw = json.loads(args.config.read_text()) if args.config else {}
        if set(raw) - {'model', 'score', 'training'}:
            raise ValueError('Unknown config section')
        cfg, modelcfg = NoteAETrainingConfig(**raw.get('training', {})), NoteAEConfig(**raw.get('model', {}))
        scorecfg = NoteScoreEncoderConfig(**raw.get('score', {'context_dim': modelcfg.context_dim}))
        corpus = NoteAECorpus(windows, validation_windows=val,
                              r_scaler=load_note_performance_scaler(args.r_scaler) if args.r_scaler else None)
        triples = FixedNoteTriplets(corpus, *(load_style_artifact(p) for p in supplied)) if all(supplied) else None
        _, report = train_note_ae(corpus, args.out, config=cfg, model_config=modelcfg, score_config=scorecfg,
                                  triplets=triples, resume=args.resume)
        print(json.dumps({'train_windows': report['train_windows'], 'validation_windows': report['validation_windows'],
                          'triplets': report['eligible_triplets'], 'style_trained': report['style_training_performed']}))
    else:
        torch.set_num_threads(1)
        model, _, scaler, score = load_note_ae_model(args.checkpoint)
        corpus = NoteAECorpus(windows, validation_windows=val, r_scaler=scaler, score_scale=score)
        if args.command == 'export':
            manifest = export_note_embeddings(model, corpus, args.out, split=args.split, batch_size=args.batch_size,
                device=args.device, checkpoint_id=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
            print(f"Exported {len(manifest['windows'])} window embeddings")
        else:
            result = latent_ablation(model, corpus, split=args.split, device=args.device)
            with args.out.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(json.dumps(result))


if __name__ == '__main__':
    main()
