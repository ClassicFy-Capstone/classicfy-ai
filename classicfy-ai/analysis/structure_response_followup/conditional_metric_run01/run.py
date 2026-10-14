"""Reproduce the frozen experiment in a new sibling folder; preserve completed files."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); destination = args.out.resolve()
    if destination.parent != source.parent or destination.exists():
        parser.error('Choose a new direct sibling folder under Classicfy/diagnostics')
    destination.mkdir()
    names = ('data.py', 'model.py', 'protocol.json', 'train.py', 'prepare_judges.py',
             'evaluate.py', 'test_design.py', 'test_pipeline.py', 'verify.py', 'report.py',
             'aligned.py', 'aligned_protocol.json', 'verify_aligned.py', 'report_aligned.py')
    for name in names: shutil.copyfile(source / name, destination / name)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/classicfy-conditional-plots', XDG_CACHE_HOME='/tmp/classicfy-font-cache')
    def call(name): subprocess.run([sys.executable, str(destination / name)], env=env, check=True)
    call('data.py'); call('test_design.py'); call('test_pipeline.py'); call('prepare_judges.py')
    workers = []; logs = []
    seeds = json.loads((destination / 'protocol.json').read_text())['training']['seeds']
    try:
        for seed in seeds:
            log = (destination / f'train_seed{seed}.log').open('w'); logs.append(log)
            workers.append(subprocess.Popen([sys.executable, str(destination / 'train.py'), '--seed', str(seed)], env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers): raise RuntimeError('Training worker failed; inspect seed logs')
            time.sleep(.5)
        if any(p.returncode != 0 for p in workers): raise RuntimeError('Training worker failed')
    finally:
        for p in workers:
            if p.poll() is None: p.terminate()
        for log in logs: log.close()
    call('evaluate.py'); call('verify.py'); call('aligned.py'); call('verify_aligned.py'); call('report.py'); call('report_aligned.py')


if __name__ == '__main__': main()
