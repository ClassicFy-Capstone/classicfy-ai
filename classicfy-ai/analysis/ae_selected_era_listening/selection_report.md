# G·임베딩 일치 및 짜임새 조건으로 재선정한 AXY 문항

Pack: `621710dd77203aebee59bd1b9451c8412980f3bb0fa44309ad0f0703e9adf70c`

2026-11-16. 세 시대 3/3/4문항. 모두 G 양쪽 유효·순서 일치·명확한 거리 차이·서로 다른 X/Y 연주자 조건을 통과했다. 이 조건으로 선정한 시연용 표본이며 일반 청취 정확도를 검증하는 무작위 표본은 아니다.

| 문항 | 유형 | 임베딩 A–B / A–C | G A–B / A–C | X/Y 연주자(B / C) |
|---|---|---:|---:|---|
| 바로크 01 | 인벤션 | 0.316 / 0.919 | 0.029 / 7.254 | András Schiff / Walter Gieseking |
| 바로크 02 | 알르망드 | 0.130 / 0.835 | 0.837 / 14.777 | Ivo Pogorelich / Vladimir Ashkenazy |
| 바로크 03 | 쿠랑트/코렌테 | 0.075 / 0.661 | 0.355 / 7.644 | Murray Perahia / Walter Gieseking |
| 고전과 전환 01 | 론도 | 0.068 / 0.753 | 0.245 / 5.511 | Alfred Brendel / Daniel Barenboim |
| 고전과 전환 02 | 론도 | 0.206 / 0.976 | 0.323 / 5.559 | Vladimir Horowitz / Radu Lupu |
| 고전과 전환 03 | 소나타 4악장·빠른 표기 | 0.099 / 1.024 | 0.259 / 4.340 | Glenn Gould / Wilhelm Kempff |
| 낭만과 인상주의 01 | 에튀드 | 0.268 / 1.148 | 0.304 / 5.348 | Nikolai Lugansky / Vladimir Ashkenazy |
| 낭만과 인상주의 02 | 에튀드 | 0.378 / 1.127 | 0.461 / 5.694 | Vladimir Horowitz / Vladimir Ashkenazy |
| 낭만과 인상주의 03 | 왈츠 | 0.292 / 0.970 | 0.645 / 6.037 | Claudio Arrau / Dinu Lipatti |
| 낭만과 인상주의 04 | 에튀드 | 0.311 / 0.999 | 0.622 / 5.783 | Vladimir Ashkenazy / Valentina Lisitsa |

- 모든 X/Y는 서로 다른 연주자다. A와 X 또는 Y의 연주자가 같을 수 있다.
- G 신뢰도는 정답 확률이 아니다. 거리차이와 품질 수치는 selection_audit.json에 보존한다.
- 제목상 유형을 맞추고 일반 소나타는 악장 번호도 일치시켰다. quarter당 onset 밀도 및 onset당 음표 수의 작품 중앙값 비율은 각각 1.6배 이하다. 실제 악구의 짜임새 전체를 검증한 것은 아니다.
- 임베딩 거리차 >=0.30, B<=0.40, C>=0.65; calibrated G 목표 거리차 >=0.15, B<=0.55, C>=0.65; 원래 G 거리비 >=1.5; 양쪽 품질 >=0.25를 고정 조건으로 사용했다.
- 기존 전역 최근접/작품 전체 최원거리 규칙 대신 합격 후보 안에서 선정한다. 화면 설명도 변경했다.
- 제공 YouTube 시작/종료 시각을 그대로 사용하고 30개 구간의 원본/매핑 정합성을 검증했다.
- 실제 브라우저 playing 상태는 8/30개만 확인했다. 1개는 준비 지연 상태였고 21개는 미확인이다. 브라우저 연결이 중단되어 부분 확인으로 기록한다.
- 새 pack은 이전 응답을 승계하지 않는다. 이전 문항·검증 기록과 결과는 archive 폴더에 보존한다.

## 문항과 작품

### 바로크 01

- anchor: ATEPP:10592 / András Schiff / Invention No. 3 in D, BWV774
- near: ATEPP:10613 / András Schiff / Invention No. 11 in G minor, BWV782
- far: ATEPP:10615 / Walter Gieseking / Invention No. 11 in G Minor, BWV 782
- 악보 onset 밀도·화음 밀도 비율: 1.142 / 1.126

### 바로크 02

- anchor: ATEPP:02424 / Ivo Pogorelich / English Suite No. 3 in G Minor, BWV 808: II. Allemande
- near: ATEPP:02788 / Ivo Pogorelich / English Suite No. 2 in A Minor, BWV 807: II. Allemande
- far: ATEPP:02782 / Vladimir Ashkenazy / English Suite No. 2 in A Minor, BWV 807: 2. Allemande
- 악보 onset 밀도·화음 밀도 비율: 1.055 / 1.036

### 바로크 03

- anchor: ATEPP:01918 / Murray Perahia / French Suite No. 5 in G Major, BWV 816: II. Courante
- near: ATEPP:02470 / Murray Perahia / Partita No. 5 in G Major, BWV 829: III. Corrente
- far: ATEPP:02468 / Walter Gieseking / Partita No. 5 in G Major, BWV 829: III. Corrente
- 악보 onset 밀도·화음 밀도 비율: 1.010 / 1.007

### 고전과 전환 01

- anchor: ATEPP:05353 / András Schiff / Piano Sonata No. 9 in D, K.311: 3. Rondeau (Allegro)
- near: ATEPP:06478 / Alfred Brendel / Piano Sonata No. 19 in G Minor, Op. 49 No. 1: II. Rondo. Allegro
- far: ATEPP:06469 / Daniel Barenboim / Piano Sonata No. 19 In G Minor, Op. 49, No. 1: 2. Rondo (Allegro)
- 악보 onset 밀도·화음 밀도 비율: 1.182 / 1.140

### 고전과 전환 02

- anchor: ATEPP:09308 / Friedrich Gulda / Piano Sonata No. 16 in G Major, Op. 31 No. 1: III. Rondo. Allegretto
- near: ATEPP:08106 / Vladimir Horowitz / Piano Sonata No. 8 in C Minor, Op. 13 "Pathétique": III. Rondo. Allegro
- far: ATEPP:08143 / Radu Lupu / Piano Sonata No. 8 in C Minor, Op. 13 "Pathétique": 3. Rondo (Allegro)
- 악보 onset 밀도·화음 밀도 비율: 1.440 / 1.034

### 고전과 전환 03

- anchor: ATEPP:07893 / Maurizio Pollini / Piano Sonata No. 28 in A Major, Op. 101: IV. Geschwind, doch nicht zu sehr und mit Entschlossenheit (Allegro)
- near: ATEPP:06240 / Glenn Gould / Sonata No. 3 in C Major, Op. 2, No. 3: IV. Allegro assai - Remastered
- far: ATEPP:06254 / Wilhelm Kempff / Piano Sonata No. 3 in C Major, Op. 2 No. 3: IV. Allegro assai
- 악보 onset 밀도·화음 밀도 비율: 1.075 / 1.239

### 낭만과 인상주의 01

- anchor: ATEPP:09750 / Vladimir Ashkenazy / 12 Études, Op. 25: No. 11 in A Minor "Winter Wind"
- near: ATEPP:09577 / Nikolai Lugansky / Chopin: 12 Études, Op. 10: No. 4 in C-Sharp Minor
- far: ATEPP:09569 / Vladimir Ashkenazy / 12 Etudes, Op. 10: No. 4 in C-Sharp Minor "Torrent"
- 악보 onset 밀도·화음 밀도 비율: 1.430 / 1.120

### 낭만과 인상주의 02

- anchor: ATEPP:00331 / Claudio Arrau / 3 Etudes de Concert, S.144: No. 2 in F Minor "La leggierezza" (A capriccio - Quasi allegretto)
- near: ATEPP:09402 / Vladimir Horowitz / Études, Op. 10: No. 5 in G-Flat Major "Black Keys" - Instrumental
- far: ATEPP:09410 / Vladimir Ashkenazy / 12 Études, Op. 10: No. 5 in G-Flat Major "Black Keys"
- 악보 onset 밀도·화음 밀도 비율: 1.096 / 1.404

### 낭만과 인상주의 03

- anchor: ATEPP:11149 / Claudio Arrau / Waltz No.3 in A Minor, Op.34 No.2
- near: ATEPP:11019 / Claudio Arrau / Waltz No.10 in B Minor, Op.69 No.2
- far: ATEPP:11025 / Dinu Lipatti / Waltz No. 10 in B Minor, Op. 69, No. 2
- 악보 onset 밀도·화음 밀도 비율: 1.051 / 1.228

### 낭만과 인상주의 04

- anchor: ATEPP:09392 / Vladimir Ashkenazy / 12 Études, Op. 10: No. 6 in E-Flat Minor
- near: ATEPP:09723 / Vladimir Ashkenazy / 12 Études, Op. 25: No. 5 in E Minor
- far: ATEPP:09736 / Valentina Lisitsa / 12 Etudes, Op.25: No.5 In E Minor
- 악보 onset 밀도·화음 밀도 비율: 1.021 / 1.088
