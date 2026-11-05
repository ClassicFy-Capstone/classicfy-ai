"""Export aligned score S and existing performance R for a conditional AE.

Score map CSV: key,work,score_path,score_sha256,split (optional, unassigned).
Native ATEPP-1.2 standardized channel caches and portable quarter caches are
supported. Canonical work/split identity is explicit, never guessed from names.
"""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.score_context_data import export_score_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance-cache',type=Path,required=True)
    parser.add_argument('--score-map',type=Path,required=True)
    parser.add_argument('--score-root',type=Path)
    parser.add_argument('--dataset',choices=('ATEPP','ASAP'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    scale=parser.add_mutually_exclusive_group()
    scale.add_argument('--fit-score-scale',action='store_true')
    scale.add_argument('--score-scale',type=Path)
    args=parser.parse_args()
    manifest=export_score_context(args.performance_cache,args.score_map,args.out,dataset=args.dataset,
        score_root=args.score_root,fit_scale=args.fit_score_scale,scale_path=args.score_scale)
    records=manifest['records']
    print('Exported',len(records),'recordings;',len(manifest['scores']),'score/grid references;',
          sum(r['usable_intervals'] for r in records),'usable original intervals')


if __name__=='__main__':main()
