G 유효성·거리 순서·악장 유형을 확인한 AXY v2 (2026-11-16)

접속: http://localhost:8770/ (이전 화면이 남아 있으면 새로고침)
실행: python tools/ae_selected_era_listening/serve.py --port 8770

현재는 --selection-policy g-verified-v2로 만든 3/3/4문항을 사용한다.
같은 시대의 다른 작품에서 A의 고정 유효 G 차원을 두 후보가 모두 관측하고,
G와 임베딩 모두 B가 가깝고 C가 멀며 충분한 차이가 있는 사례만 선정했다.
X/Y는 반드시 서로 다른 연주자다. A와 후보의 연주자가 같을 수는 있다.
제목상 장르/악장 유형을 맞추고 일반 소나타는 빠르기 표기와 악장 번호도 맞춘다.
악보 quarter당 onset 밀도와 onset당 음표 수의 작품 중앙값 비율을 각각 1.6배 이하로 제한했다.
이는 거친 짜임새 지표이며 실제 악구·세부 주법이 동일하다는 뜻은 아니다.

고정 조건: 코사인 B<=.40, C>=.65, 차이>=.30;
calibrated G 목표 거리 B<=.55, C>=.65, 차이>=.15;
원래 G 거리비>=1.5, 양쪽 품질>=.25. 품질은 청취 정답 확률이 아니다.
조건을 만족하는 후보 안에서 선정하므로 전체 catalogue 최근접 1위/전체 작품 최원거리라는 보장은 없다.
이 시연 문항의 청취 일치율을 전체 모델 성능으로 일반화하지 않는다.

선정 근거와 수치: analysis/ae_selected_era_listening/selection_report.md, selection_audit.json
재생 확인 범위: analysis/ae_selected_era_listening/browser_playback_checks.json
실제 브라우저 재생 확인과 구간 경계·녹음 동일성의 청음 검수는 구분한다.
블라인드·응답 잠금·반복·자동 종료·JSON 내보내기 엔진은 유지한다.
새 pack은 이전 응답을 승계하지 않는다. 이전 pack 자료는 analysis/ae_selected_era_listening/archive에 보존한다.
기존 브라우저 응답도 이전 pack의 저장 키에 남는다. ?qa=1은 별도 검증 공간이다.

이하 v1의 과거 기록 (현재 선정 규칙은 위의 v2 설명을 따른다)
============================================================

접속 http://localhost:8770/
AI 저장소 폴더에서 실행 python tools/ae_selected_era_listening/serve.py --port 8770
ZIP을 푼 폴더에서 실행 python ae_selected_era_listening/serve.py --port 8770
Python 표준 라이브러리만으로 서버를 실행한다. YouTube 재생에는 인터넷 연결이 필요하다.

원본 feat/#17-ae-cross-work-listening commit d5311e6247b3d8fff43f5c314b6c817c6ad42a64
UI 스타일과 study.js player.js serve.py는 원본과 동일하다.
app.js와 HTML은 대표구간 거리라는 사실과 시대별 범위를 설명하는 문구만 수정했다.
테스트의 예전 문항 전용 fixture는 새 데이터로 교체했다.
기존 tools/ae_theme_listening 프로토타입은 그대로 보존했다.

모델 자료는 전체5082구간으로 최종학습한 보호G 시대별 모델과 64차원 e다.
선정 범위는 시대별 전체 임베딩이다. 다른 시대의 벡터는 비교하지 않는다.
A에서 가장 가까운 다른 작품의 연주 B를 고르고,
B와 같은 작품의 다른 연주 중 A에서 가장 먼 C를 고른다.
C는 B에서 먼 연주가 아니다. byte-identical MIDI는 별도 후보로 쓰지 않는다.
같은 작품의 대표구간 기준은 dataset의 reference/candidate 정보로 확인했다.

B/C 순위를 영상 상태로 바꾸지 않는다. 제공 매핑이 matched_auto가 아니거나
영상이 중복되는 등의 부적합한 문항 전체를 제외한다.
각 시대에서 distance gap 내림차순으로, 작품과 영상 중복을 배제해 선택했다.
총10문항20작품30연주, 바로크3 고전과전환3 낭만과인상주의4문항이다.
선정은 극단 사례 중심의 greedy 방식이다. 모든 문항 조합의 전역 최적화나
전체 catalogue의 무작위 청취 정확도 평가 표본을 의미하지 않는다.

청취 구간은 제공한 youtube_matches.csv의 youtube_start_seconds와
 youtube_end_seconds를 소수 초 그대로 사용한다. start_seconds/end_seconds는
원본 MIDI 시간으로 서로 다른 좌표이므로 영상 재생 시간으로 대체하지 않았다.
30영상 모두 2026-11-15 실제 앱에서 YouTube 플레이어의 playing 상태를 확인했다.
한 영상은 첫 준비 지연 후 재시도 성공했다. 구간 대응의 청각적 정답은 미검수다.

X/Y는 무작위로 정한 뒤 브라우저에 유지한다. 후보 연주자와 거리 및 모델 순위는
선택 저장 뒤 공개한다. 공개 뒤에는 응답과 메모를 바꾸지 못한다.
동점과 판단 어려움은 확정 선택 일치율 분모에서 제외한다.
시간 표시 및 조정과 원래 대표구간 복원, 반복/자동종료, JSON 내보내기는 원본과 같다.
유튜브 영상 안의 제목과 연주자는 보일 수 있다. 소스를 보면 결과 JSON도 볼 수 있으므로
접근 통제가 아닌 UI 수준 블라인드다. 외부 원본 링크는 시작 시각만 전달한다.

모델과 구간이 바뀌었으므로 예전 모델의 응답은 새 pack으로 승계하지 않는다.
?qa=1은 실제 응답과 분리된 검증용 저장 공간이다. UI 검증 응답은 이에만 남겼고
실제 청취 URL에는 검증용 답을 넣지 않았다. 이전 프로토타입 응답도 삭제하지 않았다.

선정 감사 자료는 analysis/ae_selected_era_listening에 보관한다.
배포 ZIP의 organizer 폴더는 정답과 거리를 포함하므로 블라인드 청취 후 열람한다.
