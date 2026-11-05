# MusicXML 음표별 악보 입력과 ScoreEncoder 구현 계획

2026-10-29 KST. 상태: 구현 전 설계안. 기존 MIDI/quarter 입력의 구현 기록은 유지한다.

사용자 검토 반영: 표기 음표 S와 실제 타건 R의 단위를 분리하고 tie 복원 마스크·전체 음가
계약을 공유한다. KnownMask를 note MLP에 직접 전달하며, onset pooling은
**masked mean + log(1 + 유효 음표 수)**로 고정한다.

## 1. 목적과 범위

MusicXML만으로 결정적으로 추출한 음표별 입력 S_i를 만들고, 주변 악보를 반영하는
음표별 맥락 C_i를 출력한다. 모델 가중치는 후속 연주 복원 손실로 공동 학습한다.
이번 구현 범위는 schema, XML 추출, 구조·정렬 연결, export/batch, ScoreEncoder와 검증이다.
연주 feature R 추출 구현, 전체 AE, 실제 학습과 검색 성능 평가는 별도 후속 범위다.
단, tie의 target 소유권·마스크·기준 음가는 S/R 추출기가 공유할 계약으로 여기서 고정한다.

```text
MusicXML → 원본 음표/쉼/지시/연결 테이블 → 구조 해석 → 음표별 S_i
                                                       ↓
                                     수치 변환 + 범주 임베딩 + KnownMask_i
                                                       ↓
                                                  note MLP → h_i
                                                       ↓
                        같은 악보 onset의 집합 → masked mean + log(1+n_g)
                                                       ↓
                                     onset 간격을 붙여 packed BiGRU
                                                       ↓
                                        onset별 맥락 H_g → 음표로 gather
                                                       ↓
                                         MLP([h_i, H_g(i)]) → C_i
```

기존 `score_context.py`의 30D 집계나 `score_encoder.py`의 quarter 입력에 XML 정보를
끼워 넣지 않는다. 새 schema와 별도 클래스로 구현하고, 기존 경로는 비교 기준으로 둔다.

## 2. 구현 전에 고정할 데이터 계약

### 음표의 단위와 시간축

- S_i의 기본 행은 **반복 회차를 구분한 MusicXML의 pitched 표기 음표 하나**다.
  화음의 각 음도 별도 행이다. 같은 원본 음표가 반복되면 occurrence별 행을 가진다.
- tie의 시작·연장·종료 음표를 원본 테이블에서 삭제하거나 합치지 않는다.
  `tie_chain_id`, 원본 ID 목록, chain 전체 음가, `is_attack`, 연결된 발음 이벤트와
  target 소유 음표는 연결 메타데이터로 보존한다.
  미래 R의 실제 타건과 표기 음표가 1:1이라고 가정하지 않는다.
- rest는 별도 구조 테이블에 보존하고 첫 encoder에서는 pitched note 행의 문맥 계산에 쓴다.
  전곡 쉼, 마지막 쉼의 범위도 보존한다. padding으로 바꾸지 않는다.
- onset, duration, rest, direction distance는 quarter=1이다. XML divisions와 정수 duration을
  유리수로 해석하고, 그룹 생성·정렬 비교가 끝난 뒤 모델 입력에서 float로 변환한다.
- `<backup>`, `<forward>`, `<chord>`, divisions 변경, tuplets, 여러 part의 시간축을 처리한다.
  XML duration에 이미 반영된 잇단음 비율을 다시 곱하지 않는다.
- written 위치와 반복 전개 후 위치를 분리한다. 모델의 event 순서는 선택한 score traversal을
  따른다. traversal은 악보 구조/명시적 경로에서 정하며 실제 연주 onset으로 만들지 않는다.
- grace의 표기 duration=0을 보존한다. 인위적인 양의 duration을 배정하지 않는다.

### feature와 연결 메타데이터 분리

학습 입력에는 아래 3절의 음악적 정보만 넣는다. 별도 연결 테이블에는 다음을 남긴다.

`source_xml_sha256`, `xml_note_id`, ID 부재 시 결정적 source locator, `part_id`,
`voice_id`, `staff_id`, `measure_index`, 원본 measure label, written onset,
`traversal_id`, `occurrence_index`, unfolded onset, tie/slur/beam 연결 ID,
alignment의 score/performance ID와 상태, schema/parser/policy version.

성부 ID는 part 안에서 해석하고, cross-staff 성부 이동을 같은 성부로 추적한다.
표시 마디 번호가 반복되거나 문자가 섞여 있어도 내부 measure index와 혼동하지 않는다.
작품·연주자·파일명·원시 ID 숫자는 S에 넣지 않는다.

### 없음·미확인·padding

- `known absent`: 정상적으로 해당 필드를 읽었고 표기가 없음. 예: staccato=0, known=True.
- `known present`: 표기/적용값을 읽음. 예: staccato=1, known=True.
- `unknown`: 읽기·범위 연결 실패 또는 적용값 미지정. payload는 중립값, known=False.
- `unsupported`: 지원 밖 표기를 발견함. unknown과 함께 reason code를 진단에 기록한다.
- `padding`: 음표 자체가 없음. note_mask=False이며 위 field-known mask와 별개다.

범주는 `NONE`/`UNK`를 구분하고, `PAD`는 batching에만 쓴다.
KnownMask는 저장뿐 아니라 note MLP의 입력에도 포함한다. note_mask는 음표의 존재,
KnownMask는 필드의 관측 여부, R의 target mask는 복원 항목의 유효 여부로 역할이 다르다.
상수 필드도 삭제하지 않는다. 예: 전곡 4/4이면 모든 음표의 박자표 값이 동일하다.

## 3. S_i v1 특징 명세안

아래는 raw 의미의 고정안이다. raw 필드·변환·범주 vocabulary·mask 순서를 schema에
명시하고, 최종 tensor 차원은 이 schema로 산출한다. 차원을 먼저 정해 의미를 줄이지 않는다.

| 묶음 | raw 필드 | 첫 모델 변환 / 규칙 |
|---|---|---|
| 음높이 | MIDI pitch, pitch class | pitch/127, pitch class 12범주 임베딩. 전조 악기는 transpose/옥타브 처리를 명시하고 written pitch도 감사용 보존 |
| 길이·쉼 | written_duration_q, following_rest_q | 비음수 log1p. 쉼은 동일 성부에서 해당 note offset에 접해 시작하는 명시적 연속 rest 길이. 다른 성부 onset이나 단순 다음 onset 차이로 추정하지 않음 |
| 박자·위치 | meter numerator/denominator, beat grouping, nominal_measure_q, actual_measure_q, measure_phase, beat_strength | meter 범주와 grouping, 길이 log1p, phase 0–1, strength 0–4 고정 규칙. 불규칙 마디는 별도 해석 상태 |
| 조표 | fifths 또는 비전통 조표 상태, explicit mode | fifths와 mode 별도 범주. mode는 XML 명시값만 사용. 비전통 조표를 C장조/0으로 대체하지 않음 |
| 음표 표현 | staccato, tenuto, accent, strong-accent, fermata, trill, arpeggiate | 7개 multi-hot + 필드별 known. trill 본음의 실제 연주 분할은 R 연결에서 처리 |
| 연결 | slur start/continue/end, beam start/continue/end, tie start/continue/end | 각 상태 multi-hot + known. 같은 음에 start/end 동시 허용. 중첩 번호·beam level·hook은 원본 연결 테이블에 보존 |
| 기본 템포 | base tempo token, explicit quarter BPM, base distance_q | 명시적 metronome 단위를 quarter BPM으로 환산. 텍스트 Allegro 등을 임의 BPM으로 환산하지 않음 |
| 템포 변화 | relative token, gradual token, local token, 각 active 지시 이후 distance_q | 기본·상대 변화·accel/rit·국소 지시를 독립 슬롯으로 유지. a tempo/tempo primo의 복귀 대상 기록 |
| 기본 셈여림 | dynamic level token, distance_q | ppp…fff 등 고정 범주. 원문 토큰 보존. 미지정은 mf로 채우지 않음 |
| 셈여림 변화 | crescendo/decrescendo 상태, local emphasis token, 각 distance_q | 기본 p와 crescendo 동시 유지. sfz/fp 등을 영구 기본 단계로 덮어쓰지 않음 |
| 꾸밈음 | is_grace, group order, group size, pre/post anchor relation | grace는 길이 0. 일반 음은 known none. 위치·그룹 순서는 악보 구조 기준 |

tie 상태는 표기 음표를 유지하는 설계에서 재타건과 지속을 구별하기 위해 원 제안에 추가한 필드다.
following_rest_q의 0은 음표 끝에 명시적 rest가 없다는 뜻이다. 문서 끝/성부 종료 뒤 범위가
불명확하거나 성부 연결이 끊겼으면 unknown으로 남긴다. 동시음의 길이가 달라 한 음만 먼저
끝나는 경우, 다른 음까지 rest를 소급 적용하지 않는다.

박 강도는 프로젝트 규칙으로 version 관리한다. 초안: 마디 첫 박=4, 명시된 내부 beat group의
시작=3, 나머지 denominator beat=2, 그 절반 subdivision=1, 나머지=0. 단순 4박은 2+2,
단순 2/3박은 기본 beat grouping, 복합박자는 명시된 grouping 또는 지원표를 사용한다.
7/8의 grouping이 불명확하면 임의 2+2+3을 넣지 않고 beat_strength만 unknown으로 둔다.
못갖춘마디의 nominal meter와 실제 길이를 분리하고, phase는 확정한 meter상의 위치를 쓴다.
짧은 마디를 모두 못갖춘마디로 판단하지 않는다.

실제 velocity, 연주 tempo/onset/offset, 수행된 페달, 연주에서 추정한 조성/프레이즈는 S에 없다.
악보의 명시적 metronome BPM은 연주 tempo와 구분해 허용한다.

## 4. 지시 적용 엔진

1. XML 지시를 raw event table로 먼저 보존한다. source 위치, musical offset, part/staff/voice,
   span number, 시작·종료, 원문, 정규화 토큰, 해석 상태를 남긴다. 화면 x좌표를 quarter로 환산하지 않는다.
2. 적용 scope는 명시적인 voice/staff/part/system 정보를 우선한다. 피아노의 staff 없는 지시를
   part 범위로 해석하는 등 fallback은 policy로 공개한다. placement=above/below만으로 scope를 정하지 않는다.
3. 같은 시점에서는 scope별 충돌을 먼저 판정하고 그 시점의 음들에 적용한다.
   더 구체적인 scope를 우선하되 동등 범위 충돌은 unknown+diagnostic으로 남긴다.
4. 기본 템포/셈여림은 다음 유효 기본 지시까지 유지한다. 점진 변화와 국소 강조는 독립 관리한다.
   wedge는 start/stop/number로 연결하고, 후속 기본 지시만 보고 wedge가 끝났다고 가정하지 않는다.
5. 텍스트 지시는 version이 있는 사전으로 해석한다. accel., rit., cresc. 등 열린 종료 지시는
   명시적 종료/복귀 또는 정책에 정한 대체 지시까지 유지한다. 모르는 표현은 원문+unknown이다.
   상대 지시의 누적/대체와 a tempo의 복귀 stack을 규칙표·fixture로 고정한다.
6. distance는 각 슬롯의 실제 적용 시작점부터 unfolded quarter 차이로 따로 계산한다.
   active 지시가 없거나 시작점을 모르면 0을 유효 거리로 주지 않는다.
7. crop 전에 전체 traversal에서 상태를 계산한다. 구간 앞의 p/Allegro가 crop으로 사라지지 않는다.
8. 반복 점프의 carry/reset은 traversal 정책으로 명시한다. 첫 v1은 명확한 반복/ending 경로부터
   지원하며, 지시 상태 복귀가 모호한 D.C./D.S./coda는 미지원으로 보고하고 학습 export를 막는다.

## 5. 표기 음표와 정렬 연결

- 먼저 ID가 포함된 nASAP XML로 추출기를 검증한다. 원본 ASAP XML과 nASAP XML을
  파일명만 같다는 이유로 교환하지 않는다.
- XML ID ↔ TSV/match score ID를 직접 검사한다. 반복 접미사를 임의로 제거하지 않는다.
  source ID와 occurrence ID의 관계를 실제 파일에서 확인한 mapping table로 보존한다.
- 정렬은 S 계산 이후 붙이는 별도 산출물이다. performance_id를 바꿔도 같은 XML/traversal의 S는 같다.
- tie continuation은 별도 S/C를 유지하되 아래 공유 계약으로 복원 target을 연결한다.
  one-to-many trill, insertion, deletion, unresolved mapping은 상태·관계로 보존한다.
- ATEPP의 기존 score MIDI ID를 XML ID로 간주하지 않는다. XML↔정렬 기준 MIDI의 판본·반복·음표
  대응을 확인한 bridge가 있는 자료만 연결한다. XML 부재 자료에 표현 기호를 추측해 채우지 않는다.
- schema/source hash가 바뀌면 새 cache다. XML별 deterministic raw/encoded S는 공유 저장하고,
  traversal이 다른 경우 unfolded 상태 및 occurrence mapping은 별도로 보존한다.

### S/R 추출기가 공유하는 tie 복원 계약

- S_i와 C_i는 모든 표기 음표 occurrence에 존재한다. tie 시작·연속·종료 상태와
  동일 발음 이벤트를 가리키는 관계를 보존한다. 종료 음표도 앞 음에서 tie가 이어지면 재타건이 아니다.
- R은 실제 타건 이벤트에 대응한다. 표기 음표 축으로 배치할 때는 해당 타건을 소유하는
  attack 음표 한 곳에만 velocity·시작 시점·유지 길이 등 이벤트 단위 target을 둔다.
- tie 연속·종료 음표에 동일 velocity나 시작 시점을 복사해 중복 학습하지 않는다.
  해당 복원 항목의 `target_mask[i, channel]`을 False로 둔다. 이 때문에 S/C의 note_mask나
  KnownMask를 끄거나 해당 음표를 onset pooling에서 제거하지 않는다.
- 하나의 타건 유지 길이를 복원할 때 기준 악보 길이는 `tie_chain_duration_q`다.
  이는 해당 traversal에서 연결된 chain 전체의 악보 길이이며 각 표기 조각의 길이와 구분한다.
  articulation 등에서 기대 길이를 초로 바꾸면 전체 chain의 시작부터 끝까지 tempo map으로
  환산한다. raw 유지 길이 target 자체는 실제 타건의 note-off − note-on이다.
- 연결 메타데이터는 `tie_chain_id`, `sounding_event_id`, `attack_owner_occurrence_id`,
  `tie_chain_duration_q`, `tie_chain_known`을 공유한다. 실제 performance event ID는 정렬
  메타데이터에만 있고 S에 넣지 않는다. 반복 회차가 다른 chain/event는 구분한다.
- target mask는 항목별 observation·alignment·event ownership 유효성을 함께 반영한다.
  연결이 미확인인 chain은 영향을 받는 항목을 결측 처리하며 전체 길이를 추측하지 않는다.
  event당 유효한 유지 길이 target은 한 번만 학습한다. 다른 복원 항목의 단위도 후속 R 명세에
  명시하며 연속음의 모든 항목을 일괄 0 또는 일괄 유효로 만들지 않는다.

예: 1 quarter 음표와 2 quarter 음표가 tie로 이어지고 실제 타건이 하나라면 S/C 행은 2개다.
velocity/onset/유지 길이 target은 시작 행에서 한 번 학습하고, 두 번째 행의 해당 mask는 False다.
유지 길이의 기준 음가는 1이나 2가 아닌 3 quarter다.

## 6. 인코더와 batch 계약

입력 예:

```text
numeric[B,N,Dn], categorical[B,N,Dc], flags[B,N,Df]
field_known[B,N,Dknown], note_mask[B,N]
note_to_onset[B,N], onset_mask[B,G], delta_q[B,G]
→ C[B,N,Dcontext]
```

- numeric은 고정 변환을 우선한다. 선택적 mean/SD는 **관측된 train score 값**
  (`note_mask=True`이며 해당 field_known=True)으로만 fit한다. unknown의 채움값과 padding,
  validation/test 값은 통계에서 제외하고 동일 악보가 연주 수만큼 중복 가중되지 않게 한다.
  범주 사전에는 NONE과 UNK를 별도로 두고 validation/test에서도 같은 사전을 사용한다.
- `h_i = NoteMLP(concat(Encode(S_i), KnownMask_i))`로 고정한다. Encode는 수치 변환,
  범주 임베딩과 multi-hot 표현이다. field_known은 batch→forward→concat까지 전달한다.
  known=False 수치 payload와 padding은 모델 전에 중립값으로 정리하고, 범주 미확인은
  UNK를 사용한다. 임의 NaN/Inf를 유효 입력으로 통과시키지 않는다.
- 같은 정확한 악보 onset의 모든 pitched note를 한 그룹으로 묶는다. part/voice가 달라도 같은
  시점이면 같은 그룹이며 pooling은 **masked mean + log(1+n_g)**로 고정한다.
  `m_ig = note_mask_i AND (note_to_onset_i == g)`, `n_g = sum_i m_ig`,
  `u_g = concat(sum_i(m_ig * h_i) / n_g, log(1+n_g))`다. max pooling은 사용하지 않는다.
  n_g는 padding을 제외한 실제 표기 음표 수이며 tie 연속음도 포함한다. 필드 미확인이나 R 결측을
  이유로 음표를 평균/개수에서 제외하지 않는다. 실제 onset 그룹은 n_g>0이어야 하며 빈 padding
  그룹은 안전한 분모 처리 후 0으로 만들고 onset_mask=False로 BiGRU에서 제외한다.
- grace도 첫 버전에서는 같은 표기 onset 그룹에 포함하되 h_i에 그룹 순서·anchor 관계를 준다.
  grace 내부 순서는 note feature에 남고 BiGRU의 별도 시간 step으로는 표현하지 않는다.
  이 한계를 공개하고, grace를 세부 event로 확장하는 실험은 후속으로 분리한다.
- `H = BiGRU(concat(u_g, delta_q_g))`로 onset 간격을 함께 제공한다.
  delta_q는 인접 표기 onset 사이의 quarter 간격이며 실제 연주 초가 아니다.
  첫 그룹은 0과 boundary flag로 구별한다.
- `C_i = MLP(concat(h_i, H[note_to_onset_i]))`. Dcontext 초기값은 128로 제안하며 설정값으로 둔다.
  같은 화음도 h_i가 달라 서로 다른 C_i를 낼 수 있다.
- note와 onset 각각 실제 prefix 길이로 padding을 관리한다. 필드 결측으로 음표를 삭제하거나
  불연속 시간축을 압축하지 않는다. C의 padding은 0이다.
- 처음에는 곡 단위 encoding으로 규칙·padding을 검증한다. 긴 곡은 onset 그룹을 자르지 않는
  chunk+좌우 악보 overlap을 도입하고 중앙 output만 학습에 쓴다. 유한 chunk의 BiGRU가 전곡 맥락과
  같다고 주장하지 않는다. metadata와 directive state는 crop 전 전체 악보 기준이다.
- tie를 포함한 note C→attack target, beat Tempo 조건으로의 연결은 별도 adapter가 맡는다.
  note C와 beat R을 위치 확인 없이 같은 index로 concat하지 않는다.

## 7. 구현 순서와 완료 조건

| 단계 | 산출물 | 다음 단계로 넘어갈 조건 |
|---|---|---|
| 0. 데이터 감사·명세 고정 | XML 구조/ID/반복/지원 기호 inventory, schema JSON, scope·박강도·텍스트 사전, S/R tie 공유 계약 | 대표 복잡 사례와 event 소유권·전체 음가·항목별 마스크를 설명 가능 |
| 1. 구조 파서 | `preprocessing/musicxml_score.py`, 원본 note/rest/direction/span/traversal 테이블 | quarter 위치·화음·성부·tie·grace·반복을 수작업 fixture와 대조 |
| 2. S 추출기 | `features/note_score_context.py`, schema/정규화 모듈, CLI export | 제안된 9묶음과 unknown/none 및 지시 전파 검증 통과 |
| 3. 연결·batch | `preprocessing/score_note_mapping.py`, `embedding/note_score_context_data.py`, cache manifest | ID/occurrence/tie 대응, target mask와 field_known 전달, split/source hash, 공유 cache, padding 검증 통과 |
| 4. NoteScoreEncoder | `embedding/note_score_encoder.py`와 설정·저장 형식 | masked mean·개수·known mask·permutation/padding/gradient 검증, 음표별 C shape 보장 |
| 5. 실데이터 검증 | `scripts/validate_note_score_context.py`, CSV/JSON과 그림 보고서 | 채널별 support와 미지원 사유, 독립 대조, 정렬 coverage 공개 |
| 후속 AE | R 명세/target adapter/전체 복원 학습 | 위 데이터 계약 고정 이후 별도 계획으로 진행 |

파서는 raw XML provenance를 직접 보존하고 구조 계산용 라이브러리 adapter와 분리한다.
저장소의 선택적 alignment 의존성에 partitura가 있으므로 첫 후보로 검토하되, ID/tie/grace/
repeat 정보가 유지되는지 단계 0에서 확인하고 의존성을 고정한다. XML에서 소실된 필드를
라이브러리 추정값으로 조용히 대체하지 않는다.

## 8. 검증 계획

합성 fixture: 4/4→3/4·조표 변경, 6/8·가산박자·못갖춘마디, divisions 변경·잇단음,
서로 다른 성부의 쉼과 cross-staff, 화음의 다른 음가, tie/중첩 slur/다중 beam,
grace·trill·fermata, p+crescendo와 sfz, tempo 복귀, offset이 있는 staff 지시,
단순 반복·ending·미지원 점프, 원본 ID 누락·중복·해석 불가 표기를 포함한다.

반드시 확인할 불변성:

1. 같은 XML/traversal이면 연주 velocity/timing/페달·정렬 파일이 달라도 S와 score cache key가 동일하다.
2. 화음 note 배열 순서를 바꾸면 H는 같고, C는 같은 note identity를 따라 순서만 바뀐다(eval 모드).
3. padding을 늘려도 유효 C가 같고 padding output/gradient가 0이다.
4. crop 앞의 지시가 유지된다. full extraction slice와 crop input의 raw S가 같다.
   full-song C와 chunk C의 동일성을 요구하는 테스트는 만들지 않는다.
5. tie·반복·누락 정렬에서 가짜 타건/음표를 생성하거나 대응 안 되는 target을 정상 관측으로 표시하지 않는다.
   1+2 quarter tie fixture에서 S/C 2행, attack target 1회, 연속음 항목 mask=False,
   chain 기준 음가 3 quarter를 확인한다. crop이 chain 중간에서 시작해도 target 소유권을
   재배정하거나 복사하지 않고 전체 악보의 chain 관계를 유지한다.
6. 같은 수치 채움값이어도 field_known의 없음/미확인이 note MLP 입력에서 구분된다.
   unknown 수치 payload를 바꿔도 중립화된 입력이 같고, 정규화 통계는 unknown·padding을 제외한다.
   NONE과 UNK의 범주 ID가 다르며 field_known이 저장 파일에서 forward까지 보존된다.
7. pooling은 손계산한 masked mean과 log(1+n_g)에 일치한다. 동일 h를 가진 음표를 늘리면
   평균은 같고 개수 항은 변한다. padding 추가는 둘 다 바꾸지 않으며, tie/R 결측 음표도
   실제 표기 음표이면 평균과 개수에 포함된다. 빈 padding 그룹은 0이고 BiGRU에서 제외된다.
8. test-only 작은 decoder의 masked reconstruction loss에서 note MLP·범주 embedding·BiGRU까지
   finite gradient가 도달한다. 이는 실제 AE 학습 완료나 C의 음악적 품질을 뜻하지 않는다.

실데이터는 nASAP 대표 소수 곡에서 원본 XML/정렬과 수작업 대조한 뒤 전체 XML로 확대한다.
독립 구조 계산과 비교하고, 특징별 known/unknown/unsupported 비율 및 repeat/tie mapping coverage를
저장한다. 데이터가 모호하면 오류 위치와 영향을 보고하며 정상 음표로 통과시키지 않는다.
ATEPP는 실제 XML+MIDI bridge가 있는 환경에서 별도 확인한다.

## 9. 근거와 프로젝트 선택의 구분

[Jeong et al., MEC 2019](https://mac.kaist.ac.kr/pubs/JeongKwonKimNam-mec2019.pdf)의
부록은 note pitch/duration, 뒤따르는 쉼, 박자·위치·박 강도, slur/beam 상태,
표현 기호, 템포/셈여림 지시와 그 이후 quarter 거리를 포함한다.
본 계획의 조표/mode 구분, ID 배제, unknown 계약, schema, pooling+BiGRU 구조와 검증 절차는
프로젝트의 설계 선택이다. 논문 전체 입력의 재현으로 표기하지 않는다.

[(n)ASAP 공식 설명](https://github.com/CPJKU/asap-dataset)은 추가된 MusicXML note ID와
연주 note의 정렬을 제공하며 반복 구조가 다른 score variant의 존재를 설명한다.

MusicXML 세부 의미는 공식 사양을 기준으로 한다:
[direction의 voice/staff 및 scope 단서](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/direction/),
[offset의 divisions 및 sound 구분](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/offset/),
[tie와 tied의 구분](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/tie/),
[wedge의 start/stop/number](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/wedge/).
이 사양이 프로젝트의 텍스트 지시 해석이나 fallback 정책까지 정해 주지는 않는다.
