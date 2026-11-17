"""Compute whole-recording 23D g, split-specific references, and a train-only scaler."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.performance_style_data import prepare_performance_styles
from features.performance_style_profile import PerformanceStyleConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='Existing ae-training-index JSON')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; overwrite is refused')
    parser.add_argument('--config', type=Path, help='Optional whole-performance threshold overrides')
    args = parser.parse_args()
    config = PerformanceStyleConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    report = prepare_performance_styles(args.index, args.out, config)
    print(json.dumps({'complete': True, 'recordings': report['total_recordings'],
                      'splits': {k: {name: row[name] for name in ('recordings', 'core_valid', 'all_23_valid')}
                                 for k, row in report['splits'].items()}}), flush=True)


if __name__ == '__main__':
    main()
