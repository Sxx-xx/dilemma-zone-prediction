# 노트 01 — SinD 샘플 로더, 정지선, 황색 정지/통과 라벨링

> 단계 1 결과 정리 (2026-10-11). 표기: **[확인]** = 파일/실행으로 직접 검증, **[추측]** = 근거는 있지만 검증 안 됨.

## 1. 이 단계에서 한 일 (한 줄 요약)
SinD 공개 샘플(4개 도시 × 1개 녹화)로 궤적·신호·Lanelet2 지도를 읽고, **"황색 시작 순간 정지선으로 다가오던 차량"을 뽑아 정지(0)/통과(1) 라벨**을 붙였다.
결과: 파이프라인은 동작하지만 **샘플만으로는 교차로 1곳(창춘, 69건)만 쓸 만함** → LOIO는 전체 데이터가 와야 가능.

## 2. 핵심 개념

### 2.1 신호등 CSV는 "상태표"가 아니라 "이벤트표"
- 매 프레임 상태가 있는 게 아니라, **어느 신호든 바뀔 때만 한 줄**이 기록된다. 임의 시각 t의 상태 = t 이전 마지막 이벤트의 값 (`np.searchsorted`로 조회 → `Recording.light_state_at`).
- 코드: **0 = 적, 1 = 녹, 3 = 황** **[확인]** — `Format.md` 81행 + 실제 파일에서 전이 순서가 항상 1→3→0, 황색 지속 ≈ 3.0 s.

### 2.2 시간 정렬 함정 (이번 단계에서 가장 중요한 발견)
- 신호 CSV에는 `RawFrameID`(원본 영상 프레임)와 `timestamp(ms)` 두 개의 시계가 있다. SinD 공식 시각화 도구는 RawFrameID를, 공식 trajdata 코드는 timestamp(ms)를 쓴다 — **녹화에 따라 둘이 0 / 30 / 65 / 90 s나 어긋난다.**
- 어느 쪽이 맞는지 **데이터로 판정**했다: 정렬이 맞다면 "우회전 아닌 차량이 적신호에 정지선을 넘는 비율"이 거의 0이어야 한다.

| 교차로 | RawFrameID 정렬 시 적신호 통과 비율 | timestamp(ms) 정렬 시 |
|---|---|---|
| Changchun | 21.0 % | **1.5 %** |
| Chongqing NR | 54.6 % | **3.8 %** |
| Tianjin | 4.3 % (두 시계 동일) | **4.3 %** |
| Xi'an | 41 % | **7.7 %** (최적 +4.5 s에서 4.0 %) |

- 결론 **[확인]**: `timestamp(ms)`가 궤적의 `timestamp_ms`와 같은 시계다. Xi'an은 원본 영상이 **59.94 fps**여서(ΔRawFrameID 186 / 3.10 s) RawFrameID/29.97로 계산하면 황색이 6 s로 보였던 것. 중복·역순 행도 있었음 → 시간순 정렬.
- **[추측]** Xi'an은 수 초 정도의 잔여 오프셋이 있을 수 있음(최적 +4.5 s). 샘플 이벤트가 1건이라 영향은 없지만 전체 데이터에서 재검증 필요.

### 2.3 정지선
- 4개 지도 모두 Lanelet2 `type=stop_line` way가 **4개씩 존재** **[확인]**. 좌표는 위경도(원점 0,0) → SinD 도구와 동일하게 UTM 투영 후 원점 빼기 (`lanelet_map.LL2XYProjector`).
- 궤적과 겹쳐 그려 정렬 확인 (`results/sind/map_*.png`): 정지선이 대기열 끝에 정확히 놓임 **[확인, 육안]**.
- 정지선 방향 벡터 `n`(상류→교차로 안쪽)을 정하면: 부호 있는 거리 `d = -(p - mid)·n`, 측방 오프셋 `lat = (p - mid)·t`. 차량 **앞범퍼** 위치 = 중심 + L/2·heading.
- 정지선이 지도에 없을 때의 대안 (NGSIM에서 실제로 사용): **적신호 중 정지한 선두 차량들의 앞범퍼 위치의 상위 분위수(q95)** = 정지선 추정.

### 2.4 어떤 신호가 어떤 접근로를 제어하나?
- 각 접근로에 대해 "통과 차량이 적신호인 비율이 가장 낮은 신호"를 고름 (`events.infer_light_map`). 4곳 모두 **N/S → light_1, E/W → light_2** (2현시).
- 공식 매핑 JSON과 비교: cc, xasl 일치 **[확인]**; tj는 공식이 light 1,3,5,7을 쓰는데 1=4=5=8, 2=3=6=7로 값이 같아 일치 **[확인]**; cqNR은 JSON의 lanelet 이름 체계가 달라 직접 비교 불가.

### 2.5 이벤트 정의와 라벨 (`events.py`)
후보: 황색 시작 직후 첫 프레임에서
- car/truck/bus, 앞범퍼가 정지선 상류 0 < d ≤ 80 m, 정지선 폭 ±1.5 m 안, 교차로 쪽 속도 ≥ 2 m/s
- 우회전 제외 (중국은 적신호 우회전 허용 → 신호와 무관한 결정), 황색 전 2 s 이력 필요(모든 모델이 같은 샘플을 보도록)

라벨 (`label_event`, NGSIM과 공유):
- **go(1)**: 상류에서 멈추지 않고 정지선 통과, 늦어도 적색 시작 후 3 s 이내
- **stop(0)**: 정지선 전(상류)에서 속도 < 1 m/s로 정지
- 그 외(궤적이 중간에 끊김, 멈춤 없이 한참 뒤 통과) → 제외

## 3. 코드 위치
| 기능 | 위치 |
|---|---|
| Lanelet2 파싱·투영·정지선 | `src/dzp/lanelet_map.py` |
| SinD 로더, 신호 이벤트표, 시간 정렬 | `src/dzp/sind.py` (`_load_lights`, `Recording.light_state_at`) |
| 정지선 기하, 통과 검출, 회전각 | `src/dzp/geometry.py` |
| 후보 추출·라벨·시퀀스 | `src/dzp/events.py` (`extract_events`, `label_event`) |
| 정렬 검증 + 지도 그림 | `scripts/sind_check_alignment.py` → `results/sind/alignment.csv`, `map_*.png` |
| 실행 | `scripts/run_sind.py` → `results/sind/` |

## 4. 결과와 해석

**이벤트 수** (`results/sind/extraction_stats.json`)

| 교차로 | 후보 | 우회전 제외 | 이력 부족 | 최종 | stop / go |
|---|---|---|---|---|---|
| Changchun | 129 | 33 | 26 | **69** | 36 / 33 |
| Chongqing NR | 2 | 1 | 1 | 0 | — |
| Tianjin | 3 | 1 | 2 | 0 | — |
| Xi'an | 5 | 4 | 0 | 1 | 0 / 1 |

- 왜 이렇게 적나? 필터 버그가 아님을 확인 **[확인]**: 필터를 다 풀어도 황색 시작 시 상류에서 움직이는 자동차가 cqNR 3대, tj 3대, xasl 5대뿐. 교통량이 적고(정지선 통과 5–16대/분, 창춘은 39대/분) 드론 화면의 상류 구간이 짧다(tj 남북 ≈ 10 m).
- 창춘 이벤트도 d 중앙값 12 m, v 중앙값 5.9 m/s → **저속 도심 접근**이라 고전적 딜레마존(50–60 km/h, 30–80 m)과는 성격이 다름.

**모델 비교 (창춘만, within-site 5-fold, 같은 황색 주기는 같은 fold)** — `results/sind/metrics_within.csv`, macro-F1:

| τ (황색 후) | ① 규칙 | ② LR | ③ MLP | ④ LSTM |
|---|---|---|---|---|
| 0 s | 0.678 | 0.809 | **0.854** | 0.725 |
| 0.5 s | 0.752 | 0.867 | **0.897** | 0.797 |
| 1.0 s | 0.782 | 0.896 | **0.927** | 0.869 |

- 규칙 대비 차이의 95% CI (주기 단위 bootstrap, `bootstrap_within.csv`): MLP +0.15~0.18 (CI가 0을 넘지 않음), LR +0.12~0.14 (유의), **LSTM +0.05~0.09 (τ=0, 0.5에서 CI가 0 포함)**.
- 해석: 규칙의 **stop recall이 0.56**에 불과 — 실제로는 `d/v < t_r`(갈 수 있는데도) **멈추는 운전자가 많다**. 학습 모델은 이 "보수적 정지"를 배운다.
- 주의: n = 69, 교차로 1곳. **일반화에 대해서는 아무 말도 할 수 없다.** LSTM이 MLP보다 못한 것도 데이터가 작아서일 가능성이 큼 **[추측]**.

## 5. 면접 예상 질문

**Q1. 신호 데이터와 궤적 데이터의 시간 정렬을 어떻게 검증했나요?**
- 한: 문서화된 정렬 기준이 두 개(RawFrameID, timestamp)였고 녹화마다 수십 초씩 달랐습니다. 그래서 물리적으로 반드시 성립해야 하는 성질, 즉 "우회전이 아닌 차량은 적신호에 정지선을 거의 넘지 않는다"를 검정 통계로 썼습니다. 오프셋을 ±40 s 스캔해 적신호 통과 비율을 계산했고, timestamp 정렬에서 1.5–4.3%로 최소가 되는 것을 확인했습니다. 이 과정에서 Xi'an 영상이 59.94 fps라는 숨은 차이도 찾았습니다.
- EN: The dataset offered two candidate clocks that disagreed by up to 90 s. I used a physical invariant as the test statistic — non-right-turning vehicles should almost never cross the stop line on red — and scanned time offsets. The timestamp-based alignment minimised the red-crossing rate (1.5–4.3%), and the scan also exposed that one recording's raw video was 59.94 fps, not 29.97.

**Q2. 정지/통과 라벨은 어떻게 정의했고, 애매한 경우는?**
- 한: 황색 시작 시 정지선 상류 80 m 이내에서 2 m/s 이상으로 접근하는 차량을 대상으로, 상류에서 1 m/s 미만으로 멈추면 stop, 멈추지 않고 적색 시작 3 s 이내에 정지선을 넘으면 go로 정의했습니다. 궤적이 끊기거나 멈추지 않고 한참 뒤에 넘는 경우(서행 진입·오할당)는 라벨이 불확실하므로 제외하고 그 수를 기록했습니다. 우회전은 적신호에도 합법이라 신호에 대한 결정이 아니므로 뺐습니다.
- EN: Candidates are vehicles within 80 m upstream moving ≥ 2 m/s at yellow onset. Stop = halts (< 1 m/s) upstream; go = crosses without halting, no later than 3 s into red. Ambiguous cases (track ends, late creeping crossings) are dropped and counted, and right turns are excluded because right-on-red is legal in China, so they are not decisions about the signal.

**Q3. 샘플 데이터 결과에서 MLP가 가장 좋았는데, 그럼 LSTM은 필요 없다는 결론인가요?**
- 한: 아닙니다. 69건, 교차로 1곳이라 LSTM처럼 파라미터가 많은 모델에는 불리하고, bootstrap CI로 보면 LSTM과 규칙의 차이도 τ=0에서 유의하지 않습니다. 이 결과가 말하는 건 "순간값만으로도 규칙보다 낫다"는 것과 "규칙은 보수적 정지를 놓친다"는 것까지입니다. LSTM의 가치는 전체 데이터에서 LOIO로 다시 검증해야 합니다.
- EN: No. With 69 events from a single site, a sequence model is data-starved, and its gain over the rule is not significant at τ = 0 under a cycle-level bootstrap. What the sample supports is narrower: instantaneous features already beat the rule, mainly because the rule misses conservative stops. Whether the LSTM helps has to be tested on the full dataset with leave-one-intersection-out.
