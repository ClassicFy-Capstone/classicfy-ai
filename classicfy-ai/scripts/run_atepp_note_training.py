"""Run resumable ATEPP alignment, indexed inputs, and full reconstruction training."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(run, source):
    run, source = run.resolve(), source.resolve()
    python = run / 'runtime/bin/python'
    scripts = source / 'scripts'
    logs = run / 'logs'; logs.mkdir(exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': now(), 'run': str(run), 'source': str(source), 'status': 'running'}
    child = None

    def write_state(**changes):
        state.update(changes, updated_at=now())
        temporary = run / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(run / 'status.json')

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        write_state(status='stopped', signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = {**os.environ, 'PYTHONPATH': str(source / 'src') + ':' + str(source / 'tests'),
                   'MPLCONFIGDIR': str(run / 'matplotlib-cache'), 'NUMBA_CACHE_DIR': str(run / 'numba-cache'),
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CLASSICFY_TORCH_THREADS': '1',
                   'PYTHONUNBUFFERED': '1'}

    def stage(name, arguments):
        nonlocal child
        if shutil.disk_usage(run).free < 5 * 1024 ** 3:
            raise RuntimeError('Less than 5 GiB free; stopping before the next stage')
        command = [str(python), *map(str, arguments)]
        path = logs / (name + '.log')
        write_state(stage=name, command=command, log=str(path))
        with path.open('a') as output:
            child = subprocess.Popen(command, cwd=source, env=environment, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write_state(child_pid=child.pid)
            code = child.wait()
        with (run / 'events.jsonl').open('a') as output:
            output.write(json.dumps({'stage': name, 'finished_at': now(), 'exit_code': code}) + '\n')
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; see {path}')
        child = None

    try:
        write_state(stage='starting')
        pre = run / 'preprocessing'
        common = ['--root', run / 'input', '--out', pre]
        stage('alignment', [scripts / 'validate_atepp_note_pipeline.py', 'align', *common, '--workers', '2',
                            '--max-note-cells', '120000000'])
        stage('alignment-large', [scripts / 'validate_atepp_note_pipeline.py', 'align', *common, '--workers', '1',
                                  '--max-note-cells', '500000000', '--retry-resources'])
        records = {row['perf_id']: row for row in (json.loads(line) for line in (pre / 'alignments.jsonl').read_text().splitlines())}
        if len(records) != 11674 or any(row['status'] == 'error' for row in records.values()):
            raise RuntimeError('Alignment screening is incomplete or contains technical errors; training not started')
        inputs = run / 'ae_inputs'
        stage('prepare-inputs', [scripts / 'prepare_note_ae_index.py', '--preprocessing', pre, '--out', inputs])
        stage('validate-inputs', [scripts / 'validate_note_ae_index.py', '--index', inputs / 'index.json',
                                  '--out', inputs / 'validation.json'])
        training = run / 'training'
        stage('training', [scripts / 'train_note_autoencoder.py', 'train', '--index', inputs / 'index.json',
                            '--config', run / 'training_config.json', '--out', training, '--cache-records', '8'])
        best = training / 'best.pt'
        if not best.is_file():
            raise RuntimeError('Training completed without a validation-selected checkpoint')
        stage('test-evaluation', [scripts / 'train_note_autoencoder.py', 'evaluate', '--index', inputs / 'index.json',
                                  '--checkpoint', best, '--split', 'test', '--out', training / 'test.json'])
        write_state(status='complete', stage='complete', child_pid=None, best_checkpoint=str(best),
                    best_checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(), finished_at=now())
    except Exception as exc:
        write_state(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now())
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_pipeline(args.run, args.source)
