# ATEPP 후보 전체 → YouTube 구간 매칭

기존 `atepp_representative_v2_all_candidates`의 후보들을 원본 영상의 소수점 초수로 변환하는 CPU 프로그램입니다. 기존 표본과 같은 **50 Hz quantized chroma + DLNCO / MrMsDTW**를 사용하고, chroma-only 정렬로 경계를 교차 검사합니다.

원본 MIDI와 후보 폴더는 읽기만 합니다. 후보 4개마다 영상을 다시 처리하지 않고 **연주 전체를 한 번 정렬한 뒤 그 연주의 모든 후보에 적용**합니다. 각 연주는 별도 프로세스로 실행되어 오류·메모리 문제·시간 초과가 다른 연주에 전파되지 않습니다.

## 전체 실행

저장소 루트 `capstone-2026-2-classicfy`에서 PowerShell로 실행합니다.

```powershell
.\run_youtube_sync.ps1 -Workers 2
```

현재 컴퓨터의 기존 표본용 Python 환경을 사용할 수 있으면 재사용합니다. 없으면 `.venv-youtube-sync`에 필요한 패키지를 설치합니다. 첫 설치 시간은 전체 처리 추정 시간에 포함하지 않았습니다. Python 3.11 또는 3.12와 Node.js를 권장합니다. FFmpeg 실행 파일은 `imageio-ffmpeg`에 포함되어 별도 PATH 설정이 필요하지 않습니다. GPU는 사용하지 않습니다.

PowerShell 실행 정책으로 스크립트 실행이 막힌 경우 다음처럼 이 프로세스에만 적용할 수 있습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_youtube_sync.ps1 -Workers 2
```

기본 입력은 저장소와 나란히 있는 `../outputs/atepp_representative_v2_all_candidates`이고, 기본 출력은 `../outputs/atepp_youtube_full`입니다. 위치를 바꿨다면 명시합니다.

```powershell
.\run_youtube_sync.ps1 -Workers 2 `
  -InputPath "D:\data\atepp_representative_v2_all_candidates" `
  -OutputPath "D:\results\atepp_youtube_full" `
  -MidiRoot "D:\data\ATEPP-1.2"
```

`-MidiRoot`는 원본 전체 MIDI의 위치입니다. 기존 감사표의 `local_midi`가 유효하면 생략할 수 있습니다. 후보 MIDI는 시작이 0초로 재설정되어 있으므로 전체 원본을 대신할 수 없습니다.

## 현재 처리 범위

2026-11-06에 실제 입력 CSV를 다시 집계했습니다. 이전 표본 작업 이후 폴더가 정리되어 현재 수량은 다음과 같습니다.

| 항목 | 수량 |
|---|---:|
| 연주 | 11,186 |
| CSV 후보 목록 전체 | 44,054 |
| 기본 매칭 대상 `exported` | 42,803 |
| 기본 매칭 대상 `exported_review` | 1,071 |
| 길이 필터로 제외된 목록 | 143 |
| 유효하지 않은 구간 | 37 |
| 원본 음원 총길이 | 약 995.67시간 |

기본 실행은 현재 `midi/`, `midi_review/`의 **43,874개 후보**를 대상으로 합니다. 결과표에는 제외된 180행까지 포함해 **입력 44,054행 모두를 유지**합니다. `exported_review`를 정상 후보로 승격하지 않고 검수 사유를 유지합니다. 길이 때문에 제거된 후보의 시각도 계산하려면 `-IncludeExcluded`를 추가합니다. 이 경우 유효한 143개가 추가되며, 유효하지 않은 37개는 여전히 초수를 만들지 않습니다.

## 중단과 재개

Ctrl+C로 중단한 뒤 **같은 명령을 다시 실행**하면 성공한 연주는 재처리하지 않습니다. 완료 결과는 연주마다 원자적으로 저장됩니다. 5분마다 전체 CSV도 갱신하고, 정상 종료나 Ctrl+C 시에는 즉시 갱신합니다. 강제 종료 후에도 이미 저장된 연주 결과는 남습니다.

```powershell
# 이전에 실패한 영상만 다시 시도하며, 정상 완료한 연주는 재사용
.\run_youtube_sync.ps1 -Workers 2 -RetryFailed

# 네트워크 요청 없이 현재 저장 결과로 진행 현황과 CSV 재생성
.\run_youtube_sync.ps1 -Mode status

# 먼저 일부 연주만 시험
.\run_youtube_sync.ps1 -Limit 10 -Workers 2
```

입력 후보·MIDI 파일 상태·알고리즘 코드·출력 설정의 fingerprint를 비교하므로, 결과가 달라질 변경이 있으면 해당 결과를 다시 계산합니다. 매핑 파일과 선택적으로 저장한 WAV의 해시도 확인합니다. 바뀐 설정으로 새 작업을 보존하려면 다른 `-OutputPath`를 쓰는 것이 편합니다.

같은 출력 폴더의 동시 실행은 OS 잠금으로 차단합니다. 비정상 종료 시 잠금은 자동 해제됩니다. 다운로드는 기본 10분, 한 연주는 기본 30분 제한입니다. 실패한 연주는 초수를 비워 두고 사유를 기록합니다. 연속 3회 접근 차단(429/봇 확인 등)이 감지되면 새 작업 시작을 멈추고 결과를 저장합니다. 재개할 때 `-RetryFailed`를 사용할 수 있습니다.

## 결과

- `youtube_matches.csv`: 모든 입력 후보 행 + 매칭 상태 + `youtube_start_seconds`, `youtube_end_seconds`, 영상 링크, 교차 검사 수치.
- `review_queue.csv`: 매칭은 되었지만 검수가 필요한 후보.
- `unmatched.csv`: 아직 처리하지 않은 후보, 삭제 영상, 오류, 제외 후보 등 초수가 없는 결과.
- `progress.json`: 전체 행 수와 상태별 건수.
- `jobs/<perf_id>/result.json`: 연주별 완료 체크포인트, 처리 시간, 취득 영수증.
- `jobs/<perf_id>/mapping.npz`: 원본 MIDI와 영상의 전체 시간 대응.
- `jobs/<perf_id>/worker.log`, `download.log`: 오류 분석용 로그.
- `run_manifest.json`, `last_run.json`: 입력 해시, 실행 옵션과 실행 결과.

`matched_auto`는 자동 기준 통과이고, `review_required`는 교차 검사 불일치·낮은 유사도·기존 MIDI 대응 불확실성 등이 있습니다. 같은 영상에서 잘라낸 여러 악장이나 영상 길이가 크게 긴 경우에는 저해상도 subsequence DTW로 위치를 찾고 고해상도 MrMsDTW를 수행하되, 항상 검수 대상으로 남깁니다. 삭제·비공개·지역 제한 영상은 대체 녹음을 임의로 사용하지 않습니다.

**모든 초수는 자동 추정값입니다.** 0.02초 프레임 간격과 소수점 자릿수는 실제 정렬 정확도를 보장하지 않습니다. `youtube_human_verified=False`, `youtube_exact_accuracy_measured=False`를 유지합니다. 영상 이동 링크는 정수 초로 시작을 내림하며, 구간 링크의 끝은 올림합니다. 정확한 자동 추정치는 CSV를 확인하세요. 곡의 대표성이나 음악적 구절 경계를 새로 검증하는 작업은 포함하지 않습니다.

## 오디오 저장과 디스크

기본값은 링크·초수·매핑만 남깁니다. 처리에 필요한 전체 오디오는 연주 처리 후 정리합니다. 원본 데이터나 외부 오디오 캐시는 삭제하지 않습니다.

```powershell
# 후보 구간 원음 WAV도 저장
.\run_youtube_sync.ps1 -Workers 2 -SaveClips

# 취득한 전체 오디오도 보관 (디스크 사용량 증가)
.\run_youtube_sync.ps1 -Workers 2 -KeepAudio
```

현재 데이터 전체의 22,050 Hz mono 16-bit 원음은 약 **147.2 GiB**, 후보 WAV는 원본 구간 길이 기준 약 **51.8 GiB**입니다. 전체 원음 보관 시 인코딩된 다운로드 파일도 추가됩니다. 기본 실행에서도 매핑·결과·로그와 처리 중 임시 파일을 위해 수 GiB 이상이 필요합니다. 긴 음원 동시 처리와 로그를 고려해 10~20 GiB의 여유 공간을 권장합니다. 이 수치는 파일별 매핑 압축률이나 실패 횟수에 따라 달라집니다.

## 시간 계산과 새 벤치마크

현재 컴퓨터(i5-13500H, 메모리 약 15.6 GiB)에서 6개 연주를 새 프로그램으로 측정했습니다. 원음 길이는 약 198~1,001초였으며, 세 영상은 실제 다운로드하고 나머지는 기존 원음 캐시를 재사용했습니다. 계산 모델은 최초 준비 표본을 제외한 5개, 네트워크 모델은 실제 다운로드한 3개로 맞췄습니다.

| 동시 작업자 | 전체 예상 시간 | 일수 환산 |
|---|---:|---:|
| 1개 | 약 93~187시간 | 약 3.9~7.8일 |
| **2개 (기본)** | **약 52~129시간** | **약 2.2~5.4일** |
| 4개 | 약 30~89시간 | 약 1.3~3.7일 |

단일 처리의 중심 계산값은 계산 약 79.7시간 + 다운로드·디코딩 약 36.9시간 = 약 116.7시간입니다. 병렬 표의 범위는 실제 전체 처리 시간이나 실측 병렬 가속도가 아닌, 처리 효율과 변동을 반영한 계획용 추정입니다. 절전 없이 계속 실행하는 시간을 뜻합니다. 기본 작업자 2개부터 시작하고 메모리와 다른 작업의 부하에 따라 조절할 수 있습니다.

```powershell
# 저장된 벤치마크와 현재 폴더를 기준으로 예상 시간 재계산
.\run_youtube_sync.ps1 -Mode estimate -Workers 2

# 다른 컴퓨터에서 길이가 다른 6개 연주를 골라 새 벤치마크
.\run_youtube_sync.ps1 -Mode benchmark -Workers 1
.\run_youtube_sync.ps1 -Mode estimate -Workers 2
```

기본 벤치마크는 곡 길이 분포의 약 10~99백분위에서 6개 연주를 고릅니다. 영상 삭제 등으로 실패하면 성공한 표본 수가 줄어듭니다. `-Ids`로 대상을 직접 지정할 수도 있습니다. 새 벤치마크 폴더를 쓰거나 삭제한 원음이 필요한 재측정이라면 Python CLI 옵션과 출력 경로를 조절하세요.

예상 시간 모델은 연주당 고정 비용과 음원 길이에 비례하는 계산 비용을 맞추고, 다운로드·디코딩 시간을 별도로 더합니다. 단일 작업자 5개 이상 표본에서는 최초의 캐시·컴파일 준비 표본 1개를 계산 모델에서 제외합니다. 병렬 처리 효율과 0.8~1.6배 변동 폭은 계획용 가정이며 통계적 신뢰구간이 아닙니다. YouTube 제한, 컴퓨터 절전, 다른 프로그램의 CPU 사용, 긴 편집 영상, 수동 검수 시간은 별도입니다.

이번 실행의 실제 벤치마크는 `../outputs/atepp_youtube_sync_benchmark/benchmark.json`, 계산 보고서는 `../outputs/atepp_youtube_full/runtime_estimate.json`에 저장됩니다. 여러 작업자 수의 비교는 같은 폴더의 `runtime_estimates_all_workers.json`에 있습니다. **995시간의 오디오가 995시간의 처리 시간을 뜻하는 것은 아닙니다.**

## Python CLI

```powershell
python -m pip install -r requirements-youtube-sync.txt
python classicfy-ai/scripts/match_atepp_youtube.py run --workers 2
python classicfy-ai/scripts/match_atepp_youtube.py run --ids 06264 09721 --workers 2
python classicfy-ai/scripts/match_atepp_youtube.py estimate --workers 2
python classicfy-ai/scripts/match_atepp_youtube.py status
python classicfy-ai/scripts/match_atepp_youtube.py run --help
```

Python CLI에서 `--download-delay`, `--download-timeout`, `--job-timeout`, `--checkpoint-seconds`, `--benchmark-report` 등을 세부 조절할 수 있습니다. `--audio-cache`는 표본 프로그램처럼 `<perf_id>/source.wav`와 `receipt.json`을 가진 오디오 캐시를 재사용합니다. 영상 ID·0초 기준·SHA-256·샘플레이트가 맞아야 하며, 일반 WAV 파일명을 믿고 임의로 같은 영상으로 간주하지 않습니다. `--offline`은 캐시가 없는 영상에 접근하지 않습니다.

검증 테스트:

```powershell
python -m unittest discover -s classicfy-ai/tests/unit/preprocessing -p test_youtube_sync.py -v
```

방법 참고: [Sync Toolbox 공식 오디오–악보 예제](https://github.com/groupmm/synctoolbox/blob/master/sync_audio_score_full.ipynb), [ATEPP](https://github.com/tangjjbetsy/ATEPP), [yt-dlp](https://github.com/yt-dlp/yt-dlp).
