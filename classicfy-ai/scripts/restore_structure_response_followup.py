"""Restore immutable experiment snapshots and external artifacts in a fresh workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / 'classicfy-ai/analysis/structure_response_followup'
STAGES = {'summary_mean_overlap', 'curve_comparison', 'response_model',
          'conditional_metric_model', 'conditional_metric_run01'}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def plan_restore(archive: Path, artifacts: Path) -> list[tuple[Path, Path]]:
    manifest = json.loads((archive / 'archive_manifest.json').read_text())
    result = []
    seen = set()
    for entry in manifest['files']:
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] not in STAGES:
            raise ValueError(f"Invalid archive path: {entry['path']}")
        if relative.as_posix() in seen:
            raise ValueError(f'Duplicate archive path: {relative}')
        seen.add(relative.as_posix())
        if entry['storage'] == 'git_snapshot':
            source = archive / relative
        elif entry['storage'] == 'external_artifact':
            source = artifacts / relative
        else:
            raise ValueError(f"Unknown storage: {entry['storage']}")
        if not source.is_file():
            raise FileNotFoundError(f'Missing {source}; restore the external file recorded in archive_manifest.json')
        if source.stat().st_size != entry['bytes'] or digest(source) != entry['sha256']:
            raise ValueError(f'Artifact hash mismatch: {source}')
        result.append((source, Path(relative)))
    return result


def restore(plan: list[tuple[Path, Path]], out: Path, repo: Path, datasets: Path) -> None:
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}; choose a new directory')
    if not datasets.is_dir():
        raise FileNotFoundError(f'Dataset directory is missing: {datasets}')
    out.mkdir(parents=True)
    # Existing code expects these historical workspace names. Source/dataset bytes stay unchanged.
    (out / 'capstone-2026-2-classicfy').symlink_to(repo.resolve(), target_is_directory=True)
    (out / 'datasets').symlink_to(datasets.resolve(), target_is_directory=True)
    for source, relative in plan:
        target = out / 'diagnostics' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != digest(source):
            raise ValueError(f'Copy verification failed: {target}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=REPO.parent / 'diagnostics',
                        help='External artifact root with the five original experiment directories')
    parser.add_argument('--datasets', type=Path, default=REPO.parent / 'datasets',
                        help='Existing raw/derived dataset root; linked, not copied')
    parser.add_argument('--out', type=Path, help='New replay workspace; existing paths are refused')
    parser.add_argument('--check-only', action='store_true', help='Verify all archived and external bytes without writing')
    args = parser.parse_args()
    if not args.check_only and args.out is None:
        parser.error('--out is required unless --check-only is set')
    try:
        plan = plan_restore(ARCHIVE, args.artifacts.resolve())
        if not args.datasets.is_dir():
            raise FileNotFoundError(f'Dataset directory is missing: {args.datasets}')
        if not args.check_only:
            restore(plan, args.out.resolve(), REPO, args.datasets)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps({'status': 'verified' if args.check_only else 'restored', 'files': len(plan),
                      'out': None if args.check_only else str(args.out.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
