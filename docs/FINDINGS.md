# FINDINGS — 확인한 것 vs 추측 (2026-10-11)

표기: **[확인]** 파일·코드 실행으로 직접 검증 / **[추측]** 근거는 있으나 미검증 / **[외부]** 문서·서브에이전트 조사 결과를 인용, 일부만 재확인.

## SinD (공개 샘플, SOTIF-AVLab/SinD @ `data/raw/sind/SOURCE_COMMIT`)

| 항목 | 내용 | 등급 |
|---|---|---|
| 샘플 구성 | 4개 도시 × 1개 녹화: Changchun(Pudong), Chongqing(NR), Tianjin, Xi'an(Shanglin). 각 15–26분 | 확인 |
| 궤적 | `Veh_smoothed_tracks.csv`, 10 Hz(100.1 ms), x/y [m], vx/vy, a_lon, heading, length/width. heading ≈ atan2(vy,vx) (중앙값 차 < 0.2°) | 확인 |
| 신호 코드 | **0 = 적, 1 = 녹, 3 = 황** (`Format.md` 81행 + 모든 파일에서 전이 1→3→0) | 확인 |
| 황색 길이 | ≈ 3.0 s (모든 교차로, Xi'an 포함 — 아래 fps 문제 보정 후) | 확인 |
| 신호 CSV 형식 | 변화가 있을 때만 한 줄인 이벤트표 | 확인 |
| **시간 정렬** | 신호 CSV의 `timestamp(ms)`가 궤적 `timestamp_ms`와 같은 시계. RawFrameID 기준 정렬은 녹화별로 0/30/65/90 s 어긋남 | 확인 (적신호 통과율 검정: cc 1.5%, cqNR 3.8%, tj 4.3%, xasl 7.7% vs RawFrameID 정렬 21/55/4/41%) |
| Xi'an 원본 fps | 59.94 fps (나머지 29.97). 첫 행 timestamp 결측, 중복·역순 행 존재 → 보간·정렬 | 확인 |
| Xi'an 잔여 오프셋 | 최적 +4.5 s(4.0%) vs 0 s(7.7%) — 수 초 어긋남 가능 | 추측 |
| 정지선 | 4개 지도 모두 `type=stop_line` way 4개. UTM(원점 0,0) 투영 후 궤적과 일치 | 확인 (그림 `results/sind/map_*.png`) |
| 신호→접근로 매핑 | 4곳 모두 N/S → light_1, E/W → light_2. 공식 JSON과 cc, xasl, tj 일치; cqNR은 이름 체계가 달라 비교 불가 | 확인 (cqNR 제외) |
| 이벤트 수 | cc 69 (stop 36 / go 33), xasl 1, cqNR 0, tj 0 | 확인 |
| 이벤트가 적은 이유 | 필터를 풀어도 황색 시 상류에서 움직이는 자동차가 cqNR 3, tj 3, xasl 5대뿐 → 교통량·상류 시야 부족 | 확인 |
| LOIO 가능성 | 샘플로는 불가 (사용 가능한 교차로 1곳) | 확인 |
| 창춘 결과 | MLP/LR이 규칙보다 유의하게 높음 (macro-F1 +0.12~0.18, 주기 bootstrap 95% CI > 0). LSTM은 τ=0/0.5에서 유의하지 않음 | 확인 (n=69, 1곳이라 일반화 불가) |
| 규칙이 지는 이유 | 규칙의 stop recall 0.56 — `d/v < t_r`인데도 멈추는 운전자가 많음 | 확인(수치) / 원인 해석은 추측 |

## NGSIM

| 항목 | 내용 | 등급 |
|---|---|---|
| Lankershim (data.transportation.gov) | 지원 데이터 ZIP(`Lankershim-Boulevard-LosAngeles-CA.zip`, 80,179,421 B)이 서버에서 잘려 zip으로 열리지 않음. 2026-10-11 재다운로드해도 같은 크기 · 같은 상태 | 확인 (`BadZipFile`, `unzip -l`: central directory 없음) |
| Lankershim (대체 출처) | **Wayback Machine 사본**: 옛 NGSIM 커뮤니티 사이트의 `Lankershim-Main-Data.zip` (`https://web.archive.org/web/20150324165048id_/http://ngsim-community.org/Lankershim-Main-Data.zip`, 2015-03-24 캡처, 2026-10-11 받음). 375,226,084 B, sha256 `d615aae8…7240c`. 공식 배포처(data.transportation.gov)가 아니라 보관 사본이다 | 확인 (`unzip -t`: 122개 항목 모두 OK, "No errors detected") |
| Lankershim 신호 형식 | 위 ZIP의 `signal-timing/real-time-split-monitor-reports/RSM*-061605.txt`(교차로 10곳): LADOT ATSAC 주기별 리포트. 주기마다 한 줄(시작 시각 1 s 단위, 주기 길이, 위상별 녹색 split 초). 황색 · 적색 시작 시각과 프레임 번호는 없음 | 확인 (파일 내용) |
| Lankershim 신호-궤적 동기화 | `metadata.htm`: 0830-0845 궤적 시작 = RSM 8:27:34, 0845-0900 = 8:44:34 | 외부 (같은 ZIP의 문서) |
| Lankershim 황색 시작 계산 | 주기 시작 + 앞선 위상 split으로 계산할 수 있어 보이나, split에 황색 · 전적색 포함 여부와 위상→접근로 매핑은 timing sheet PDF(`signal-timing-sheets/int*.pdf`)로 확인해야 함 | 추측 |
| Peachtree 신호 | 접근로별 signal-change 로그: 주기마다 BG/BY/BR(녹/황/적 시작) 프레임 번호. 황색 3.6 s(10th), 3.2 s(나머지) | 확인 |
| Peachtree 궤적 | 10 Hz, ft·ft/s. `local_y` = 차량 **앞범퍼 중심** 종방향 위치 (북쪽 증가). direction 2=NB, 4=SB; movement 1=직진 | 확인 (`local_y` 정의는 Lankershim 메타데이터 PDF, Peachtree도 같은 형식으로 가정 — 추측) |
| 두 시간대 구분 | `global_time - 100·frame_id` 값 2개 → 12:45–13:00 / 16:00–16:15 | 확인 (값 2개) |
| 시간대 ↔ 신호 파일 짝 | 1163018600 → 1245-0100, 1163030300 → 0400-0415 | 확인 (반대로 짝지으면 적신호 통과율 43% vs 11.5%) |
| **신호-궤적 프레임 정렬** | "신호 프레임 = 궤적 frame_id + k" 가정. k = 0에서 적신호 통과율 11.5%(전체)/4.3%(5 m/s 이상), 최소는 k ≈ −1.0~−2.0 s (2.9%) | 추측 — 문서 없음. 데이터로 ±1.5 s 범위까지만 좁혀짐 |
| 녹색 출발로 정렬 재검토 (`scripts/ngsim_green_start.py`) | 적신호에 정지선 0–8 m 앞에 서 있던 차로별 선두 차량 107대의 출발 시점(정지 위치에서 0.25 m 벗어난 순간) − 로그상 녹색: 중앙값 1.66 s, IQR 0.85–3.10 s, 범위 −1.7–5.9 s. 주기별 최초 출발만 봐도 IQR 0.83–2.85 s. k = R − 지연이라 반응시간 R(0.5–2 s 가정)에 따라 k = −1.2~+0.3 s → **기존 ±1.5 s보다 좁혀지지 않음**. 시간대별 중앙값은 0400-0415 2.35 s, 1245-0100 1.04 s로 1.3 s 차이. 시간대별 적신호 통과율 최소점은 −1.5~−2.0 s / −1.0 s | 확인 (수치) |
| **한계: 시간대별 어긋남** | 위 결과는 두 시간대의 신호 로그 시계가 서로 다르게 어긋나 있을 가능성을 시사한다(1245-0100은 k = −1.0 s면 R ≈ 0 s로 비현실적, 일부 선두 차량이 로그상 녹색보다 먼저 출발). 반응시간을 독립적으로 알 수 없어 이 데이터로는 가를 수 없음 → k = 0 본 실험 + k = −1.5 s 민감도 분석 유지 | 추측 |
| 정지선 | 데이터에 없음 → 적신호 중 정지한 직진 차량 앞범퍼 위치의 q95로 추정. int_id 경계보다 2–4 m 하류. 14th SB는 정지 차량 0 → 경계+평균오프셋으로 대체 | 추측 (CAD `.dwg`로 검증 필요) |
| 속도/가속도 | 원본 v_acc는 ±12.27 ft/s²에서 잘림 → Savitzky-Golay(1.1 s, 3차)로 위치에서 재계산 | 확인 (잘림은 서브에이전트 조사, 재계산은 코드) |
| 이벤트 수 (k=0) | 203 (stop 71 / go 132; go 중 적색 시작 후 통과 36). 10th 63, 11th 43, 12th 83, 14th 14. 82%가 선행차 있음 | 확인 |
| 결과 (k=0, LOIO) | macro-F1: 규칙 0.82–0.83, LR 0.82–0.86, MLP 0.81–0.86, LSTM 0.80–0.83. **규칙 대비 모든 차이의 95% CI가 0을 포함** | 확인 |
| 결과 (k=−1.5 s) | 이벤트 196 (stop 108 / go 88). 규칙 0.86–0.87이 최고, MLP는 τ=0.5에서 규칙보다 유의하게 낮음 | 확인 |
| **결론의 민감도** | 신호 정렬 ±1.5 s가 라벨 분포와 모델 순위를 바꾼다 → NGSIM 결과는 "LSTM이 규칙보다 낫다는 증거 없음" 이상으로 해석 불가 | 확인 |

## 연구 질문에 대한 현재 답
- "LSTM이 규칙보다 정확한가?" → **현재 데이터로는 아니오/판단 불가.** SinD 창춘(1곳)에서는 순간값 모델(MLP/LR)이 규칙을 이기고 LSTM은 유의하지 않음. Peachtree LOIO에서는 어떤 학습 모델도 규칙을 유의하게 이기지 못함.
- "처음 보는 교차로에서 일반화되는가?" → Peachtree LOIO에서 학습 모델이 within-site와 비슷한 수준을 유지(규칙과 동급). 다만 4곳 모두 같은 도로(Peachtree St.)의 연속 교차로라 "처음 보는 교차로"의 난이도가 낮다 **[추측]**.
- 다음 결정적 실험: SinD 전체 데이터(다수 교차로, 정확한 신호 시계) LOIO.
