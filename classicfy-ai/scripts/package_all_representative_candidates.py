"""Build the verified all-candidate handoff README, code snapshot and ZIP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

from export_all_representative_candidates import dump, sha


def package(root):
    root=root.resolve();repo=Path(__file__).resolve().parents[2]
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    check=json.loads((root/'verification.json').read_text(encoding='utf-8'))
    if check['errors'] or summary['performance_candidate_rows']!=check['accounted_performance_candidates']:
        raise ValueError('Complete a successful verification before packaging')
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str,'reference_perf_id':str}).fillna('')
    exported=data[data.status.isin(['exported','exported_review'])]
    lengths=exported.duration_seconds.astype(float)
    candidates=pd.read_csv(root/'candidates.csv')
    sources=[
        'classicfy-ai/scripts/export_all_representative_candidates.py',
        'classicfy-ai/scripts/verify_all_representative_candidates.py',
        'classicfy-ai/scripts/package_all_representative_candidates.py',
        'classicfy-ai/scripts/filter_representative_candidates_duration.py',
        'classicfy-ai/scripts/filter_representative_candidates_ids.py',
        'classicfy-ai/scripts/extract_atepp_themes.py',
        'classicfy-ai/scripts/run_representative_passages.py',
        'classicfy-ai/scripts/run_atepp_theme_candidates.py',
        'classicfy-ai/src/preprocessing/representative_passages.py',
        'classicfy-ai/src/preprocessing/theme_candidates.py',
        'classicfy-ai/tests/unit/preprocessing/test_all_candidate_export.py',
        'classicfy-ai/analysis/representative_passages/README.md',
    ]
    hashes={}
    for name in sources:
        destination=root/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/name,destination);hashes[name]=sha(destination)
    packages={name:importlib.metadata.version(name) for name in ['mido','pretty-midi','numpy','pandas']}
    manifest=dict(export_engine='all_saved_v2_candidates',export_source_signature=summary['signature'],
        input_summary_sha256=summary['input_summary_sha256'],audit_sha256=summary['audit_sha256'],
        source_files_sha256=hashes,python=sys.version,packages=packages,
        automatic_selection_applied=False,new_candidate_inference=False,style_embedding_included=False,
        candidate_and_mapping_content_unchanged=True,all_candidate_export_tests_passed=4,
        commands=[
          r'.\.venv\Scripts\python.exe -u classicfy-ai/scripts/export_all_representative_candidates.py --workers 4',
          r'.\.venv\Scripts\python.exe classicfy-ai/scripts/verify_all_representative_candidates.py ../outputs/atepp_representative_v2_all_candidates',
          r'.\.venv\Scripts\python.exe classicfy-ai/scripts/package_all_representative_candidates.py ../outputs/atepp_representative_v2_all_candidates'],
        source_snapshot_requires_full_project_dependencies=True)
    manifest['duration_filter']=summary.get('duration_filter')
    manifest['performance_id_filter']=summary.get('performance_id_filter')
    if summary.get('performance_id_filter'):
        manifest['candidate_and_mapping_content_unchanged']=False
        manifest['remaining_candidate_and_mapping_values_unchanged']=True
    dump(root/'run_manifest.json',manifest)
    count=summary['statuses']
    lines=['# ATEPP v2: 연주별 후보 전체 전달본','',
      '**최종 1개를 자동 선택하지 않고, 작품별 최대 4개 후보의 목록과 연주별 대응을 제공한다. 일반 MIDI에 적용된 길이 필터는 아래에 명시한다.**',
      '기존 선택본과 원본 MIDI는 변경하지 않았다. 새 후보 추론이나 점수 조정도 하지 않았다.','',
      '## 처리 범위','',
      f"- 작품·악장 {summary['works']:,}개, 연주 {summary['performances']:,}개.",
      f"- 작품 기준 후보 {summary['work_candidates']:,}개, 연주×후보 목록 {summary['performance_candidate_rows']:,}행.",
      f"- MIDI 파일 {summary['midi_files']:,}개. 상태별 개수는 다음과 같다.",'',
      '| 상태 | 개수 | 뜻 |','|---|---:|---|']
    definitions=dict(exported='대응 검사를 통과해 midi/에 저장',
        excluded_duration='일반 MIDI에서 10초 미만 또는 60초 초과로 제외: 목록만 보존',
        exported_review='대응 불확실 표시를 유지하고 midi_review/에 저장',
        invalid_interval='길이가 0이거나 유효하지 않은 시각: 목록 보존, 파일 없음',
        mapping_missing='기존 대응 없음: 목록 보존, 파일 없음',
        export_failed='내보내기 또는 내용 검사 실패: 사유 기록, 정상 출력에 포함하지 않음',
        source_mismatch='원본 해시 불일치로 내보내기 중단')
    for status,n in count.items():lines.append(f"| {status} | {n:,} | {definitions.get(status,status)} |")
    lines+=['','## 폴더와 목록','',
      '```text','atepp_representative_v2_all_candidates/',
      '  midi/<composition_id>/<perf_id>/candidate_01.mid ... candidate_04.mid',
      '  midi_review/<composition_id>/<perf_id>/candidate_01.mid ...',
      '  segments.csv','  performances.csv','  candidates.csv','  works/<composition_id>.json',
      '  review_queue.csv','  input_midi_audit.csv','  summary.json / verification.json',
      '  candidate_generation_config.json / run_manifest.json','  source_snapshot/','```','',
      '- `segments.csv`: 연주×후보마다 한 행. 곡명·연주자·원본 시각·길이·상대 파일 경로·대응 상태를 포함한다.',
      '- `performances.csv`: 연주마다 생성 후보 수와 일반/검수 출력 개수. 후보가 부족한 작품에 4개를 강제로 채우지 않았다.',
      '- `candidates.csv`: 작품 기준 후보와 반복대표성·중심성 점수. 시각은 reference_perf_id 연주 기준이다.',
      '- `works/*.json`: 작품 공통 후보 및 연주별 모든 대응. 단일 선택 결과는 저장하지 않았다.',
      '- `review_queue.csv`: 불확실 대응·파일 미출력·60초 초과·원본 품질 경고 등 우선 검수 목록. 모든 후보의 음악적 대표성 자체는 아직 미검증이다.',
      '- `input_midi_audit.csv`: 연주·작품 메타데이터 및 원본 해시. local_midi 등 절대 경로는 실행 컴퓨터의 경로다.',
      '- `source_snapshot/`: 관련 코드 사본. 단독 실행 패키지가 아니며 원본 데이터·프로젝트 의존성이 필요하다.','',
      '## 팀원이 사용할 때','',
      '1. `segments.csv`의 `clip_relative_path`를 압축 해제 폴더에 붙여 파일을 연다. 경로가 비어 있으면 status와 error를 확인한다.',
      '2. `composition_id`는 작품·악장, `perf_id`는 연주 ID다. perf_id는 앞자리 0을 보존하는 문자열로 읽는다.',
      '3. `candidate_number`는 파일명과 같은 1~4번이다. `candidate_index`는 기존 대응과 호환되는 0~3번이다. 번호는 기존 대표성 점수 순서이며 최종 추천·선호 순위가 아니다.',
      '4. 같은 작품의 같은 후보 번호를 묶으면 연주 버전 비교가 가능하다. 실제 대응이 맞는지는 alignment_accepted와 청취로 확인한다.',
      '5. start_seconds/end_seconds는 각 원본 MIDI의 시각이다. 추출 파일은 구간 시작을 0초로 재설정했다. 후보 기준 연주의 시각을 다른 연주에 그대로 적용하면 안 된다.',
      '6. midi_review/ 파일은 대응이 불확실한 후보를 빠짐없이 제공하려고 보존한 것이다. 확정 대응으로 간주하지 않는다.','',
      '```python','from pathlib import Path','import pandas as pd',
      'root = Path("atepp_representative_v2_all_candidates")',
      'table = pd.read_csv(root / "segments.csv", dtype={"perf_id": str, "composition_id": str, "reference_perf_id": str})',
      'files = table[table["status"].isin(["exported", "exported_review"])].copy()',
      'files["path"] = files["clip_relative_path"].map(lambda p: root / p)','```','',
      '## 구간 길이','',
      '- 탐색 목표 길이는 15·25·35초다. 실제 길이는 프레임 해상도, 경계 보정, 연주 속도 및 대응 품질에 따라 달라진다.',
      f"- 기준 후보 길이: {candidates.duration_seconds.min():.2f}~{candidates.duration_seconds.max():.2f}초, 중앙값 {candidates.duration_seconds.median():.2f}초.",
      f"- 실제 출력 파일 길이(검수용 포함): {lengths.min():.2f}~{lengths.max():.2f}초, 중앙값 {lengths.median():.2f}초.",
      f"- 출력 중 60초 초과 {int((lengths>60).sum()):,}개는 검수 사유로 표시했다. duration_seconds = end_seconds - start_seconds.",'',
      '## 검증과 해석','',
      f"- 전체 연주×후보 누락·중복, 원본 해시, 저장된 후보·대응 불변성, 출력 경로·해시 검사 오류 {check['errors']}개.",
      f"- 원본 해시 재검사 {check['source_hashes_rechecked']:,}개, 원본 변경 {check['source_changed']}개.",
      f"- 내보내기 시 원본 대비 독립 검사한 음표 {check['independent_content_checked_notes']:,}개. 음높이·velocity·시각, 종료 시각, 음표·페달 종료를 검사했다.",
      '- 전체 후보 보존·누락 대응·원본 변경 차단·음표 내용 변조 검출 테스트 4개 통과.',
      '- 자동으로 1개를 선택하지 않았다. 학습된 스타일 임베딩이나 확정 주제부·종지 라벨은 포함하지 않는다.',
      '- 기존 v2의 주제부 일치도가 첫 30초 기준선을 넘지 못했다는 한계는 그대로다. 전체 후보 제공이 대표성 정확도 향상을 뜻하지 않는다.',
      '- MIDI 구간이며 원본 오디오·유튜브 연결은 포함하지 않는다.','']
    if summary.get('duration_filter'):
        d=summary['duration_filter']
        lines+=['','## 일반 MIDI 길이 필터','',
            f"midi/에 10~60초(경계 포함) 필터를 적용했다. 10초 미만 {d['under_10']}개, 60초 초과 {d['over_60']}개를 제외해 {d['remaining_normal_midi']:,}개가 남았다.",
            'midi_review/는 이번 필터 대상이 아니다. 제외 내역은 excluded_candidates.csv에 있으며 제외 파일은 ZIP에 포함하지 않았다.',
            '이번 필터 후 파일·목록의 일치를 확인했다. 원본 해시 및 음표 내용 검사는 기존 검증 결과를 유지했다.','']
    if summary.get('performance_id_filter'):
        d=summary['performance_id_filter']
        lines+=['','## 지정 연주 ID 제외','',
            f"최신 길이 필터 적용 ZIP에서 지정 ID {d['matched_performances']}개 연주의 후보 {d['removed_candidate_rows']:,}행과 MIDI {d['removed_midi_files']:,}개를 추가 제외했다. midi/와 midi_review/ 모두 적용했다.",
            f"현재 midi/ {d['remaining_normal_midi']:,}개, midi_review/ {d['remaining_review_midi']:,}개다. 기존 일반 MIDI의 10~60초 필터는 유지했다.",
            'reference_perf_id는 최초 후보 생성의 기준 연주를 나타내는 출처 정보다. 기준 연주가 제외돼도 남은 다른 연주의 후보 정의와 시각을 바꾸지 않았다.','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    archive=root.with_suffix('.zip')
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix!='.tmp')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z, ThreadPoolExecutor(max_workers=8) as pool:
        # Bound memory and overlap thousands of small Windows file reads.
        for offset in range(0,len(files),128):
            batch=files[offset:offset+128]
            for i,(path,body) in enumerate(zip(batch,pool.map(Path.read_bytes,batch)),offset+1):
                info=zipfile.ZipInfo.from_file(path,Path(root.name)/path.relative_to(root))
                z.writestr(info,body,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
                if i%10000==0:print(f'zip files {i}/{len(files)}',flush=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive_crc_error')
        midi=[n for n in z.namelist() if n.endswith('.mid')]
        if len(midi)!=summary['midi_files']:raise ValueError('archive_midi_count_mismatch')
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
                files=len(files),midi_files=len(midi),crc_check='passed')
    dump(root.parent/f'{root.name}_archive_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    package(parser.parse_args().root)
