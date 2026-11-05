"""Export MusicXML written-note features and optional exact-ID alignment links."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.note_score_context_data import save_note_score_context
from features.note_score_context import extract_note_score_context
from preprocessing.musicxml_score import read_musicxml
from preprocessing.note_alignment import load_match
from preprocessing.score_note_mapping import link_note_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--traversal-json', type=Path,
                        help='Explicit list of zero-based measure indices')
    parser.add_argument('--alignment', type=Path, help='Optional nASAP .match file')
    parser.add_argument('--aliases-json', type=Path,
                        help='Explicit alignment score ID -> written note occurrence ID dictionary')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory already exists')
    if args.aliases_json and not args.alignment:
        parser.error('--aliases-json requires --alignment')
    traversal = json.loads(args.traversal_json.read_text(encoding='utf-8')) if args.traversal_json else None
    score = read_musicxml(args.xml, traversal=traversal)
    features = extract_note_score_context(score)
    links = None
    if args.alignment:
        aliases = json.loads(args.aliases_json.read_text(encoding='utf-8')) if args.aliases_json else None
        links = link_note_alignment(score, features.ties, load_match(args.alignment), score_id_to_occurrence=aliases)
    manifest = save_note_score_context(features, args.out, links=links, source_tables=asdict(score))
    print(f'Exported {len(features.metadata)} written notes, {len(features.onset_q)} onset groups; '
          f'score reference {manifest["cache_key"]}')


if __name__ == '__main__':
    main()
