"""Build reference-relative 8-bar g profiles and train triplets, without an AE."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.style_profile_data import (
    extract_profile_artifact, load_style_artifact, make_window_artifact,
    save_style_artifact, style_report, verify_window_sources,
)
from embedding.style_triplets import build_triplets, calibrate_distances
from features.style_profile import (
    StyleConfig, artifact, check_artifact, fit_profile_scaler, fit_reference, scale_profile, scale_profiles,
)


def run(args):
    if args.command == 'make-windows':
        payload = json.loads(args.manifest.read_text(encoding='utf-8'))
        config = StyleConfig(**(json.loads(args.config.read_text()) if args.config else {}))
        result = make_window_artifact(payload['records'], args.manifest.resolve().parent, config)
    elif args.command == 'fit-reference':
        windows = load_style_artifact(args.windows, 'windows')
        config = check_artifact(windows)
        records = verify_window_sources(windows)
        reference = fit_reference([r for r in records if r.performance.split == 'train'], config)
        result = artifact('reference', config, parents={'windows': windows['id']},
                          **{k: v for k, v in reference.items() if k not in
                             {'id', 'kind', 'schema', 'config', 'config_id', 'parents'}})
    elif args.command == 'extract-profiles':
        result = extract_profile_artifact(load_style_artifact(args.windows, 'windows'),
                                          load_style_artifact(args.reference, 'reference'))
    else:
        profiles = load_style_artifact(args.profiles, 'profiles')
        config = check_artifact(profiles)
        if hasattr(args, 'seed') and args.seed != config.seed:
            raise ValueError('Seed must equal the seed in the window config')
        train = [p for p in profiles['profiles'] if p['window']['split'] == 'train']
        if args.command == 'fit-profile-scaler':
            result = fit_profile_scaler(train, config, profiles_id=profiles['id'])
        else:
            scaler = load_style_artifact(args.scaler, 'profile-scaler', config)
            if args.command == 'scale-profiles':
                if scaler['parents']['profiles'] != profiles['id']:
                    raise ValueError('Scaler/profile dependency mismatch')
                result = artifact('scaled-profiles', config, parents={'profiles': profiles['id'], 'scaler': scaler['id']},
                                  profiles=scale_profiles(profiles['profiles'], scaler))
            elif args.command == 'calibrate-distances':
                result = calibrate_distances(train, scaler, config, profiles_id=profiles['id'])
            else:
                calibration = load_style_artifact(args.calibration, 'distance-calibration', config)
                if args.command == 'build-triplets':
                    result = build_triplets(train, scaler, calibration, config, profiles_id=profiles['id'])
                else:
                    result = style_report(load_style_artifact(args.windows, 'windows'), profiles, scaler, calibration,
                                          load_style_artifact(args.triplets, 'triplets'), load_style_artifact(args.reference, 'reference', config))
    save_style_artifact(result, args.out)
    count = result.get('triplets', result.get('profiles', result.get('windows', [])))
    count = len(count) if isinstance(count, list) else count
    print(json.dumps({'kind': result['kind'], 'id': result['id'], 'out': str(args.out),
                      'count': count}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fields = {'make-windows': ['manifest'], 'fit-reference': ['windows'],
              'extract-profiles': ['windows', 'reference'], 'fit-profile-scaler': ['profiles'],
              'scale-profiles': ['profiles', 'scaler'], 'calibrate-distances': ['profiles', 'scaler'],
              'build-triplets': ['profiles', 'scaler', 'calibration'],
              'report': ['windows', 'reference', 'profiles', 'scaler', 'calibration', 'triplets']}
    for name, required in fields.items():
        command = commands.add_parser(name)
        for field in required:
            command.add_argument('--' + field, type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'make-windows':
            command.add_argument('--config', type=Path)
        if name in {'fit-reference', 'fit-profile-scaler', 'calibrate-distances', 'build-triplets'}:
            command.add_argument('--split', choices=['train'], required=True)
        if name in {'calibrate-distances', 'build-triplets'}:
            command.add_argument('--seed', type=int, default=2026)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
