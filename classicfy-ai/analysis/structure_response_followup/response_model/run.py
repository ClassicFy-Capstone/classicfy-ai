"""Reproduce the pilot in a new sibling folder, preserving this completed run."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    source=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New empty folder directly under Classicfy/diagnostics')
    args=parser.parse_args();destination=args.out.resolve()
    if destination.parent!=source.parent:
        parser.error('Choose a direct sibling folder under Classicfy/diagnostics so frozen input paths remain valid.')
    if destination.exists():parser.error('Output already exists; choose a new folder to preserve checkpoints and input hashes.')
    destination.mkdir()
    names=('protocol.json','data.py','model.py','train.py','evaluate.py','verify.py','test_model.py','report.py')
    for name in names:shutil.copyfile(source/name,destination/name)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/classicfy-response-plots',XDG_CACHE_HOME='/tmp/classicfy-font-cache')
    for name in ('data.py','test_model.py','train.py','evaluate.py','verify.py','report.py'):
        print('Running',name,flush=True)
        subprocess.run([sys.executable,str(destination/name)],env=env,check=True)


if __name__=='__main__':main()
