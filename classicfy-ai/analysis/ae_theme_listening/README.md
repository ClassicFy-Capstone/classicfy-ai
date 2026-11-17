# 선정 주제부 청취 구성 — 2026-11-09

**완료: 14 epoch 모델, 고전·바로크 4문항 / 낭만 4문항 / 인상주의 2문항.**
시대 비율은 각 문항의 기준 A에 적용한다. X/Y는 A와 다른 한 작품의 두 연주다.

총 10문항·20개 작품/악장 ID·30개 독립 원본 녹음. 연주자당 최대 2회, 한 문항 안에서는 3명 모두 다르다.
모든 구간을 한 번씩 들으면 약 15분이며 반복·판단·메모 시간은 별도다.
사이트: http://localhost:8766/ (기존 화면이 열려 있다면 새로고침).

## 청취 프로그램

아래 표의 대상 연주자 순서는 가까운/먼 순서다. 실제 사이트는 X/Y 순서를 무작위로 배치하며 선택 후에만 공개한다.

| 번호 | 기준 A 작품 | 연주자 | 대상 작품의 두 연주 |
|---|---|---|---|
| 1 | 바로크 · 바흐 — 영국 모음곡 2번 · 지그 | Ivo Pogorelich | Piano Sonata No. 3 in C Major, Op. 2 No. 3: IV. Allegro assai — Lang Lang / Wilhelm Kempff |
| 2 | 바로크 · 바흐 — 파르티타 3번 · 사라방드 | Claudio Arrau | Piano Sonata No. 14 in C Minor, K. 457: II. Adagio — Zoltán Kocsis / Vladimir Ashkenazy |
| 3 | 고전 · 모차르트 — 피아노 소나타 5번 · 1악장 | Claudio Arrau | Piano Sonata No. 3 In C Major, Op. 2, No. 3: 1. Allegro con brio — Daniel Barenboim / Murray Perahia |
| 4 | 고전 · 베토벤 — 피아노 소나타 15번 «전원» · 4악장 | Emil Gilels | Bach, JS: The Well-Tempered Clavier, Book I, Prelude and Fugue No. 19 in A Major, BWV 864: Fugue — Daniel Barenboim / Glenn Gould |
| 5 | 낭만 · 쇼팽 — 연습곡 Op.10-3 «이별» | Vladimir Ashkenazy | Schumann: Kinderszenen, Op. 15: No. 10, Fast zu ernst (Live) — Martha Argerich / Artur Schnabel |
| 6 | 낭만 · 슈만 — 다비드 동맹 무곡 Op.6 · 6번 | Murray Perahia | Piano Sonata No.30 In E, Op.109: 2. Prestissimo — Emil Gilels / Wilhelm Kempff |
| 7 | 낭만 · 리스트 — 초절기교 연습곡 11번 «저녁의 선율» | Lazar Berman | Piano Sonata No. 32 in C Minor, Op. 111: II. Arietta. Adagio molto semplice e cantabile — Friedrich Gulda / Ivo Pogorelich |
| 8 | 낭만 · 라흐마니노프 — 회화적 연습곡 Op.39 · 4번 | Nikolai Lugansky | Piano Sonata No.2 in A, Op.2 No.2: 1. Allegro vivace — Alfred Brendel / Mikhail Pletnev |
| 9 | 인상주의 · 드뷔시 — 영상 1집 «물의 반영» | Arturo Benedetti Michelangeli | Kinderszenen, Op. 15: No. 7, Träumerei — Vladimir Horowitz / Martha Argerich |
| 10 | 인상주의 · 라벨 — 밤의 가스파르 «스카르보» | Alexandre Tharaud | Fantasie in C, Op. 17: 3. Langsam getragen. Durchweg leise zu halten - Etwas bewegter — Mitsuko Uchida / Sviatoslav Richter |

## 거리와 구간

- 선정 구간 AE 입력 4,898개를 14 epoch checkpoint로 다시 추론했다. 동일 원본 녹음 alias를 합쳐 검색 대상은 4,897개, 작품/악장 ID는 469개다.
- A와 다른 작품·악보 전체에서 실제 최근접 B를 찾는다. C는 B와 같은 작품·악보·반복 경로·선정 후보에서 A까지 가장 먼 다른 녹음이다.
- 10문항의 B는 모두 해당 전체 후보에서 1위다. 영상 상태에 따라 차순위 연주로 바꾸지 않았다.
- 동일 녹음 및 문항 사이 작품/악장 ID를 중복하지 않는다. 바로크 A 둘은 서로 다른 BWV 작품이다.
- 15–90초 구간과 거리 차이 최소 0.05를 적용한다. 기준 A는 고정 난수 순서와 다양성 조건으로 고르고 heldout split을 우선했다.
- 모델이 g와 잘 맞는지, 사람이 듣기에 잘 맞는지로 사례를 골라내지 않았다.
- 재생 구간은 AE `source_segments.csv`의 연주별 원본 MIDI 시작·종료 초를 그대로 사용한다.
- X/Y는 같은 작품·악보·반복 경로·후보 번호로 묶인다. 동일한 악보 경계나 동일한 음표 집합을 보장하는 조건은 아니다.
  현재 10문항 모두 X/Y의 추정 quarter 경계 또는 음표 집합이 다르며, 시작 경계 차이의 최댓값은 약 4.47 quarter다.
  모델 입력과 청취 구간은 유지했고, 경계 차이의 영향과 음악적 대응은 별도 검수가 필요하다.
- 정규화 R/S scaler를 다시 학습하거나 모델 가중치를 바꾸지 않았다.

## 재생과 사용자 응답

30개 모두 임시 Chrome 프로필에서 localhost 공식 YouTube IFrame API로 **실제 선정 구간의 무음 재생과 0.5초 이상 시간 진행**을 확인했다.
영상 ID 일치, 재생 위치, 구간 종료가 영상 길이 안에 있는지도 확인했다.
검사 중 외부 재생 제한을 반환한 영상이 포함된 문항은 제외하고 다른 기준 A로 전체 문항을 교체했다.
검색 상대를 임의의 차순위로 교체하지 않았다. 이전 후보/검사 기록도 workspace 진단 폴더에 보존한다.

이것은 음원과 MIDI의 시작 오프셋 또는 주제 경계를 사람이 청음 검수했다는 뜻은 아니다.
위치가 어긋나면 사이트의 시간 조정 기능으로 수정한다. 인터넷·계정·지역·시점에 따라 재생 상태는 달라질 수 있다.

새 pack ID/storage ID를 사용한다. 이전 모델의 실제 응답을 새 문항에 승계하지 않고 기존 브라우저 저장본을 그대로 보존한다.
이전 사이트 파일은 `diagnostics/atepp_selected_theme_listening/run01/previous_site/`에 전체 보관한다.

## 검증과 파일

- 별도 검증기로 모든 문항의 전체 최근접 및 같은 cohort 최원거리를 재계산했다.
- 4/4/2 구성, 20개 작품/악장 ID·30개 원본 녹음, 연주자 최대 2회, 원래 선정 초 범위를 검사했다.
- JavaScript 17개(선택/저장/배치/실제 파일 11 + 재생 6), 서버 6개 테스트 통과.
- 임시 브라우저의 `?qa=1` 저장 공간에서 10문항 표시, 선택 전 결과 숨김, 선택 저장 후 거리 공개, 다음 문항 이동을 확인했다.
  실제 사용자 응답 저장 공간에는 판단을 입력하지 않았다.
- 현재 `listening.json`, `selected_segment_protocol.json`, `selected_playback_checks.json`, `verification.json`이 새 문항 자료다.
- `themes.json`, `background_checks.json`, `embed_playback_checks.json` 등 예전 파일은 이전 전곡 기반 문항의 이력이며 이번 선정 근거가 아니다.
- 전체 재현 자료와 CSV: `diagnostics/atepp_selected_theme_listening/run01/`.

이 청취 구성은 전체 catalogue를 검색하므로 학습 작품을 포함한다. 청취 응답 일치율은 독립 test 성능이 아니며,
고전·바로크/낭만/인상주의 전체를 대표하는 무작위 표본도 아니다. g 없는 모델 대비 우월성을 검증하는 구성은 아니다.
