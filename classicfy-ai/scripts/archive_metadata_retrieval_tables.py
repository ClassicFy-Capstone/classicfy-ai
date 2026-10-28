"""Keep large query-level retrieval tables outside Git without changing bytes."""

import argparse
import json
from pathlib import Path
import shutil

from audit_asap_metadata import ROOT,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_retrieval_results')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/metadata_retrieval_run01')
    args=parser.parse_args();destination=args.cache/'tables';destination.mkdir(parents=True,exist_ok=True)
    execution=json.loads((args.out/'execution.json').read_text());files={}
    for name in ('overlap.csv','metadata_exposure.csv','style_scores.csv','controls.csv','balanced_gallery.csv'):
        source=args.out/name;target=destination/name;expected=execution['csv_sha256'][name]
        if source.exists():
            assert sha256(source)==expected
            if target.exists():
                assert sha256(target)==expected;source.unlink()
            else:shutil.move(str(source),str(target))
        assert sha256(target)==expected
        files[name]=dict(path=str(target.resolve().relative_to(ROOT.parent)),sha256=expected,bytes=target.stat().st_size)
    manifest=dict(execution_sha256=sha256(args.out/'execution.json'),archiver_sha256=sha256(__file__),
        policy='Large query-level tables outside Git; bytes and original execution hashes unchanged; summaries and figures remain in analysis',files=files)
    (args.out/'storage_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Archived',len(files),'tables with identical hashes')


if __name__=='__main__':main()
