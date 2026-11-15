"""Resolve selected ATEPP clips to original MusicXML, preserving provenance.

Discovery is not training admission. In particular, PERiScoPe's aligned MIDI
and raw XML share a documented score source, but need a coordinate bridge.
No MIDI-to-XML conversion, source modification, or automatic training occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@lru_cache(maxsize=None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def relative_member(member):
    parts = PurePosixPath(member).parts
    if len(parts) < 3 or parts[1] not in ('data', 'data_aligned'):
        raise ValueError(f'Unexpected archive member: {member}')
    return str(PurePosixPath(*parts[2:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--report-only', action='store_true', help='Render an existing completed audit')
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.out.resolve()
    if args.report_only:
        write_report(out, json.loads((out / 'summary.json').read_text(encoding='utf-8')))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / 'scores').mkdir()
    dataset = root / 'outputs/atepp_candidate_style_selected_recovered_final/selected_segments.csv'
    selected = read_csv(dataset)
    atepp = root / 'ATEPP_dataset/ATEPP-1.2'
    atepp_meta_path = root / 'ATEPP_dataset/ATEPP-metadata-1.2.csv'
    atepp_meta = {x['perf_id']: x for x in read_csv(atepp_meta_path)}
    periscope = root / 'PERiScoPe_dataset/v1.1'
    extraction_path = periscope / 'extraction_manifest.json'
    extraction = json.loads(extraction_path.read_text(encoding='utf-8-sig'))
    by_path, raw_member, metadata_score = defaultdict(list), {}, defaultdict(list)
    for entry in extraction:
        by_path[str(Path(entry['path']))].append(entry)
        if entry['archive'] == 'periscope_raw_v1.1.tar.gz':
            raw_member[relative_member(entry['member'])] = entry
    metadata_seen = set()
    for row in read_csv(periscope / 'metadata_v1.1.csv'):
        if row['score'] and row['xml']:
            key = (row['score'], row['xml'], row['score_source'])
            if key in metadata_seen:
                continue
            metadata_seen.add(key)
            metadata_score[row['score']].append(row)
    repairs = {}
    repairs_path = root / 'ATEPP_dataset/score_repairs.csv'
    if repairs_path.exists():
        repairs = {str(Path(x['repaired_score'])): x for x in read_csv(repairs_path)}

    # Inventory is scoped to the supplied music corpora, excluding software XML.
    inventory = []
    for folder in (root / 'ATEPP_dataset', root / 'PERiScoPe_dataset', root / 'ASAP_dataset'):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.musicxml', '.mxl'):
                inventory.append({'path': str(path), 'bytes': path.stat().st_size,
                                  'source_collection': folder.name})
    write_csv(out / 'local_inventory.csv', inventory)

    candidates, seen = [], set()
    def add(row, path, method, rank, evidence, expected_hash='', score_hash_verified=False):
        path = Path(path)
        key = (row['perf_id'], str(path))
        if key in seen:
            return
        if path.suffix.lower() not in ('.xml', '.musicxml', '.mxl') or not path.is_file():
            return
        digest = sha(str(path))
        if expected_hash and digest != expected_hash:
            raise ValueError(f'Source XML hash mismatch: {path}')
        seen.add(key)
        candidates.append(dict(perf_id=row['perf_id'], composition_id=row['composition_id'],
            composer=row['composer'], track=row['track'], score_midi_path=row['local_score'],
            score_midi_sha256=sha(row['local_score']), source_xml_path=str(path),
            xml_sha256=digest, method=method, rank=rank, provenance=evidence,
            archive_xml_hash_verified=bool(expected_hash),
            archive_score_hash_verified=score_hash_verified))

    @lru_cache(maxsize=None)
    def directory_xml(folder):
        if folder.exists() and folder.name not in ('raw', 'aligned'):
            return tuple(path for path in sorted(folder.iterdir())
                         if path.suffix.lower() in ('.xml', '.musicxml', '.mxl'))
        return ()

    for row in selected:
        score = Path(row['local_score'])
        # Use metadata's explicit score->XML link, not a title similarity guess.
        for entry in by_path[str(score)]:
            if sha(str(score)) != entry['sha256']:
                raise ValueError(f'Source MIDI hash mismatch: {score}')
            relative = relative_member(entry['member'])
            for meta in metadata_score[relative]:
                xml = raw_member.get(meta['xml'])
                if xml:
                    add(row, xml['path'], 'periscope_metadata_exact', 0,
                        json.dumps(dict(score_member=entry['member'], xml_member=xml['member'],
                                        score_source=meta['score_source']), ensure_ascii=False),
                        xml['sha256'], True)
        if str(score) in repairs:
            repair = repairs[str(score)]
            add(row, repair['source_xml'], 'documented_score_repair', 0,
                json.dumps(repair, ensure_ascii=False), repair.get('source_sha256', ''))
        xml_pair = score.with_suffix('')
        if score.suffix.lower() in ('.mid', '.midi') and xml_pair.suffix.lower() in ('.xml', '.musicxml', '.mxl'):
            add(row, xml_pair, 'score_midi_filename_pair', 1, str(score))
        original = atepp_meta[row['perf_id']]
        if original['score_path']:
            add(row, atepp / original['score_path'], 'atepp_metadata_exact', 2,
                json.dumps(dict(perf_id=row['perf_id'], score_path=original['score_path']), ensure_ascii=False))
        for folder in {score.parent, (atepp / original['midi_path']).parent}:
            for path in directory_xml(folder):
                add(row, path, 'same_work_directory_candidate', 3, str(folder))

    print(f'Resolved {len(candidates)} candidate links; beginning XML checks', flush=True)

    from preprocessing.musicxml_score import _root, read_musicxml, PARSER_VERSION
    from features.note_score_context import extract_note_score_context
    checks = {}
    paths = {row['xml_sha256']: row['source_xml_path'] for row in candidates}
    for index, (digest, path) in enumerate(sorted(paths.items()), 1):
        check = dict(xml_sha256=digest, source_xml_path=path,
                     container_xml_status='failed', note_ae_score_context_status='not_run',
                     parser_version=PARSER_VERSION)
        try:
            document, actual_hash = _root(Path(path))
            if actual_hash != digest:
                raise ValueError('File changed during audit')
            check.update(container_xml_status='passed',
                written_pitched_notes=len(document.findall('.//note/pitch')),
                parts=len(document.findall('part')))
            if not check['written_pitched_notes']:
                raise ValueError('No pitched notes in MusicXML')
        except Exception as exc:
            check['container_xml_status'] = 'failed'
            check['container_error'] = f'{type(exc).__name__}: {exc}'
        if check['container_xml_status'] == 'passed':
            target = out / 'scores' / (digest + Path(path).suffix.lower())
            shutil.copyfile(path, target)
            if sha(str(target)) != digest:
                raise ValueError('Collected copy differs from original')
            check['collected_xml_path'] = str(target)
            try:
                symbolic = read_musicxml(path)
                extract_note_score_context(symbolic)
                check.update(note_ae_score_context_status='passed',
                    expanded_notes=len(symbolic.notes),
                    traversal_id=symbolic.traversal_id,
                    diagnostics=json.dumps(symbolic.diagnostics, ensure_ascii=False))
            except Exception as exc:
                check.update(note_ae_score_context_status='failed',
                             note_ae_error=f'{type(exc).__name__}: {exc}')
        checks[digest] = check
        with (out / 'validation_progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(check, ensure_ascii=False) + '\n')
        if index % 25 == 0 or index == len(paths):
            print(f'Validated {index}/{len(paths)} unique MusicXML files', flush=True)
    write_csv(out / 'xml_validation.csv', list(checks.values()))

    by_perf = defaultdict(list)
    for candidate in candidates:
        check = checks[candidate['xml_sha256']]
        candidate.update({k: check.get(k, '') for k in ('container_xml_status',
            'note_ae_score_context_status', 'collected_xml_path', 'note_ae_error')})
        by_perf[candidate['perf_id']].append(candidate)
    write_csv(out / 'all_candidates.csv', candidates)
    resolved = []
    for original in selected:
        rows = sorted(by_perf[original['perf_id']], key=lambda x: (x['rank'], x['source_xml_path']))
        valid = [x for x in rows if x['container_xml_status'] == 'passed']
        best = valid[0] if valid else (rows[0] if rows else {})
        row = {k: original[k] for k in ('perf_id', 'composition_id', 'composer', 'track',
               'start_seconds', 'end_seconds', 'local_score', 'alignment_cache',
               'local_alignment_cache', 'mapping_revalidated', 'source_sha256')}
        row.update({k: best.get(k, '') for k in ('source_xml_path', 'xml_sha256',
            'collected_xml_path', 'method', 'provenance', 'container_xml_status',
            'note_ae_score_context_status', 'note_ae_error', 'archive_xml_hash_verified',
            'archive_score_hash_verified')})
        row.update(xml_found=bool(valid), candidate_count=len(rows),
            any_candidate_context_passes=any(x['note_ae_score_context_status']=='passed' for x in valid),
            training_admission='not_evaluated_requires_clip_alignment_and_R_S_validation')
        resolved.append(row)
    write_csv(out / 'selected_musicxml_manifest.csv', resolved)
    works = []
    grouped = defaultdict(list)
    for row in resolved:
        grouped[row['composition_id']].append(row)
    for work, rows in sorted(grouped.items(), key=lambda x: int(x[0])):
        works.append(dict(composition_id=work, composer=rows[0]['composer'], track=rows[0]['track'],
            selected_segments=len(rows), xml_found=sum(x['xml_found'] for x in rows),
            primary_context_passes=sum(x['note_ae_score_context_status']=='passed' for x in rows),
            source_xml_paths=json.dumps(sorted({x['source_xml_path'] for x in rows}), ensure_ascii=False)))
    write_csv(out / 'work_coverage.csv', works)
    composers = []
    for composer in sorted({row['composer'] for row in resolved}):
        rows = [x for x in resolved if x['composer']==composer]
        composers.append(dict(composer=composer, works=len({x['composition_id'] for x in rows}),
            segments=len(rows), xml_found=sum(x['xml_found'] for x in rows),
            primary_context_passes=sum(x['note_ae_score_context_status']=='passed' for x in rows)))
    write_csv(out / 'composer_coverage.csv', composers)
    summary = dict(selected_segments=len(selected), selected_works=len(grouped),
        score_midi_paths=len({x['local_score'] for x in selected}),
        source_inventory_files=len(inventory), candidate_links=len(candidates),
        unique_xml_files=len(checks), valid_xml_files=sum(x['container_xml_status']=='passed' for x in checks.values()),
        covered_segments=sum(x['xml_found'] for x in resolved),
        fully_covered_works=sum(x['xml_found']==x['selected_segments'] for x in works),
        missing_perf_ids=[x['perf_id'] for x in resolved if not x['xml_found']],
        primary_context_pass_segments=sum(x['note_ae_score_context_status']=='passed' for x in resolved),
        any_candidate_context_pass_segments=sum(x['any_candidate_context_passes'] for x in resolved),
        primary_resolution_methods=dict(Counter(x['method'] for x in resolved)),
        primary_context_errors=dict(Counter(x['note_ae_error'] for x in resolved if x['note_ae_error'])),
        new_downloads=0, midi_to_xml_conversions=0, training_ready_count=None,
        limitations=['Discovery and score-context parsing do not establish note alignment or complete R/S training inputs.',
            'Raw MusicXML and aligned PERiScoPe MIDI can have different repeat paths and note coordinates.',
            'No clips are filtered from the manifest; no scores, annotations, alignments or models are modified.'],
        evidence_hashes={str(p):sha(str(p)) for p in (dataset, atepp_meta_path, extraction_path,
            periscope/'metadata_v1.1.csv', Path(__file__),
            Path(__file__).resolve().parents[1]/'src/preprocessing/musicxml_score.py')})
    summary['git_branch'] = subprocess.check_output(['git','branch','--show-current'], cwd=Path(__file__).parent, text=True).strip()
    summary['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).parent, text=True).strip()
    write_json(out / 'summary.json', summary)
    write_report(out, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence_hashes','primary_context_errors')}, ensure_ascii=False, indent=2))


def write_report(out, summary):
    report = [
        '# 대표구간 데이터셋 MusicXML 탐색 결과', '',
        f"- 대상: {summary['selected_works']:,}개 작품·악장 / {summary['selected_segments']:,}개 대표구간.",
        f"- 정상 MusicXML/MXL 확보: {summary['fully_covered_works']:,}개 작품·악장 / {summary['covered_segments']:,}개 구간.",
        f"- 원본 악보 후보: 해시 중복 제거 후 {summary['unique_xml_files']:,}개. 대체 판본을 포함하므로 작품 수와 다르다.",
        f"- 현재 Note AE 악보 문맥 추출 통과: 우선 대응 악보 기준 {summary['primary_context_pass_segments']:,}개 구간.",
        f"- 같은 작품의 대체 악보까지 고려한 문맥 추출 통과: {summary['any_candidate_context_pass_segments']:,}개 구간.",
        '- 신규 다운로드 0개, MIDI를 XML로 변환한 파일 0개. 기존에 보관된 원본 악보를 찾아 연결했다.',
        '', '## 결과 파일', '',
        '- `selected_musicxml_manifest.csv`: 5,082개 구간 각각의 원본 악보·수집 사본·근거·구간 경계·기존 정렬 경로.',
        '- `work_coverage.csv`, `composer_coverage.csv`: 작품·악장별 및 작곡가별 확보 수.',
        '- `all_candidates.csv`: 우선 대응 외 대체 악보를 포함한 모든 연결 근거.',
        '- `xml_validation.csv`: XML/MXL 구조와 현재 Note AE 악보 문맥 추출 결과·오류.',
        '- `scores/`: 원본과 SHA-256이 동일한 악보 사본. 파일 이름은 내용 해시다.',
        '- `summary.json`: 집계 및 실행 코드·입력 메타데이터의 SHA-256.',
        '', '## 출처 연결 방식', '',
        'PERiScoPe는 추출 목록에서 선택된 score MIDI의 원래 경로를 복원하고, 공식 metadata의 score→xml 연결을 사용한다. '
        'raw와 aligned 디렉터리는 별도 파일로 취급한다. 선택된 MIDI 및 XML의 해시를 각각 추출 목록과 대조한다.',
        'ATEPP는 MusicXML에서 생성된 MIDI의 원본 파일명, 공식 score_path, 기록된 복구 이력을 우선한다. '
        '같은 작품 폴더의 다른 악보는 대체 후보로만 기록한다.',
        '', '| 우선 대응 방식 | 대표구간 수 |', '|---|---:|',
    ]
    report.extend(f'| {method} | {count:,} |' for method, count in summary['primary_resolution_methods'].items())
    report.extend(['', '## 해석 및 다음 단계', '',
        '**악보 확보 수와 실제 학습 가능 수는 다르다.** 이 작업은 학습을 시작하거나 학습 대상 구간을 제외하지 않았다. '
        '문맥 추출 통과는 악보 S를 읽었다는 의미이며, 연주별 R 추출·선정 구간 대응·마스크 검증까지 통과했다는 의미는 아니다.',
        '특히 PERiScoPe의 aligned score MIDI와 raw MusicXML은 반복 전개·장식음·음표 ID·시간 좌표가 다를 수 있다. '
        '기존 정렬을 다른 XML에 그대로 적용하지 말고 두 악보 표현 사이의 대응을 검증해야 한다. '
        '국소 복구가 적용된 구간은 기존 local_alignment_cache와 원본 시간 범위를 보존해야 한다.',
        'XML 구조가 정상이어도 현재 엄격한 파서가 읽지 못하는 악보는 별도 오류로 기록했다. '
        '이것을 원곡 부재나 데이터셋 품질 실패로 해석하지 않는다.',
        '', '## 원본 배포처', '',
        '- [ATEPP](https://github.com/tangjjbetsy/ATEPP)',
        '- [PERiScoPe](https://huggingface.co/datasets/SyMuPe/PERiScoPe)',
        '- [ASAP](https://github.com/fosfrancesco/asap-dataset)',
        '', '악보의 출처·이용 조건은 원본 배포처를 따른다. 수집 사본의 이용 조건을 새로 부여하지 않는다.',
    ])
    (out / 'README.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
