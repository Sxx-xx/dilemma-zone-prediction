# 노트 02 — NGSIM Peachtree 적용과 ①~④ 모델 비교 (within-site / LOIO)

> 단계 2 결과 정리 (2026-10-11). **[확인]** 직접 검증 / **[추측]** 미검증.

## 1. 한 줄 요약
Lankershim은 신호 데이터 파일이 손상돼 불가, **Peachtree(애틀랜타, 신호 교차로 4곳)로 전체 파이프라인을 끝까지 돌렸다.**
203개 이벤트에서 **어떤 학습 모델도 `d/v < t_r` 규칙을 통계적으로 유의하게 이기지 못했고**, 결론은 신호-궤적 시간 정렬(±1.5 s)에 민감했다.

## 2. 핵심 개념

### 2.1 왜 Lankershim이 아니라 Peachtree인가
- Lankershim 지원 데이터 ZIP이 서버에서 잘려 있음(80 MB, 내부 첫 파일만 122 MB) → 신호 파일 복구 불가 **[확인]**. 문서상으로도 주기별 split(1 s)뿐이라 황색 시작 시각을 프레임 단위로 알 수 없음 **[외부]**.
- Peachtree는 **접근로별 signal-change 로그**: 주기마다 녹/황/적 시작 프레임(10 Hz) → 황색 시작 시각과 잔여시간 `t_r`을 정확히 계산 가능 **[확인]**.

### 2.2 같은 파이프라인을 다른 데이터에 얹는 방법 (추상화)
- 데이터마다 다른 부분: 좌표계(SinD 2D 지도 vs NGSIM 1D `local_y`), 정지선 출처, 신호 형식.
- 공통 부분: **라벨 정의(`events.label_event`)**와 **출력 형식** — 이벤트표 + `(N, 30, 4)` 시퀀스 [d, v, a, t_red], 황색 시작 = index 19.
- 그래서 `models.py`, `evaluate.py`는 데이터셋을 모른다. 새 데이터셋 = 로더 + `extract_events`만 작성.

### 2.3 정지선이 없을 때: 대기열 선두 위치로 추정
- 적신호 동안, 교차로 박스 밖(int_id = 0)에서, 속도 < 0.3 m/s인 **직진 차량 앞범퍼 위치의 95% 분위수**. 대기열 선두는 정지선 바로 앞에 서므로 "가장 하류에서 멈춘 위치" ≈ 정지선.
- 결과: 7개 접근로에서 int_id 박스 경계보다 2–4 m 하류 **[확인]**. 14th SB는 정지 차량이 0 → 박스 경계 + 다른 접근로 평균 오프셋 **[추측]**.

### 2.4 NGSIM 품질 이슈
- 가속도 컬럼이 ±12.27 ft/s²에서 잘림, 13–14% 프레임이 |a| > 0.3 g → 비물리적 **[외부, 서브에이전트 확인]**.
- 대응: 위치(`local_y`)를 **Savitzky-Golay 필터(창 1.1 s, 3차)**로 미분해 속도·가속도 재계산 (`ngsim.load_tracks`). 미분은 노이즈를 증폭하므로 "평활 후 미분"이 원칙.

### 2.5 평가 설계
- **within-site**: `GroupKFold`(5-fold), 그룹 = 황색 주기. 같은 주기의 차량들은 같은 신호 이벤트를 공유하고 서로 영향을 주므로(선행차를 따라 정지) train/test에 쪼개지면 누수.
- **LOIO**: 교차로 하나를 통째로 test → "처음 보는 교차로" 일반화.
- **τ = 0 / 0.5 / 1 s**: 황색 후 시간이 지날수록 차량의 반응(감속)이 관측돼 쉬워짐. 실제 활용에서는 빠를수록 가치가 큼.
- positive = **stop**: 앞차의 정지를 놓치는 것(→ 추돌 위험)이 자율차 입장에서 더 위험한 오류라고 봄. 대칭 지표로 macro-F1.
- **paired cluster bootstrap**: 황색 주기 단위로 재표본(2000회) → (모델 − 규칙) macro-F1 차이의 95% CI. 이벤트가 독립이 아니므로 주기 단위로 묶어야 CI가 과도하게 좁아지지 않는다.

### 2.6 시간 정렬 민감도 (이번 단계의 핵심 교훈)
- 문서에 정렬 기준이 없어 SinD와 같은 검정(적신호 통과율)으로 k를 스캔: k = 0에서 4.3%, k ≈ −1.5 s에서 2.9% (5 m/s 이상 통과만) **[확인]**. 즉 데이터로는 ±1.5 s 안으로만 좁혀짐.
- 황색이 3.2 s인데 1.5 s 차이는 치명적: k = −1.5 s로 돌리면 stop/go 분포가 71/132 → 108/88로 바뀌고 모델 순위도 바뀜 **[확인]**.

## 3. 코드 위치
| 기능 | 위치 |
|---|---|
| 궤적 로드·기간 분리·SG 평활 | `src/dzp/ngsim.py: load_tracks` |
| 신호 로그 파싱 | `ngsim.load_signals` |
| 정지선 추정 | `ngsim.estimate_stop_lines`, `fill_missing_stop_lines` |
| 통과 검출·신호 상태 | `ngsim.crossings`, `ngsim.signal_state` |
| 이벤트 추출 (공통 라벨 사용) | `ngsim.extract_events` → `events.label_event` |
| 4개 모델 | `src/dzp/models.py` (`RuleDV`, `LogReg`, `MLP`, `LSTM`) |
| 평가·bootstrap | `src/dzp/evaluate.py` (`run`, `bootstrap_vs_baseline`) |
| 실행 | `scripts/run_ngsim.py [k]` → `results/ngsim_peachtree/` (k=−15 → `sensitivity_k-15/`) |

모델 세부: LR/MLP 입력 = 결정 시점의 [d, v, a, t_r, d/v − t_r] (규칙의 여유값을 특징으로 줘서 공정하게 비교). LSTM = 마지막 2 s(20 step) × [d, v, a, t_red], hidden 32, 5개 seed 앙상블, 클래스 가중 BCE.

## 4. 결과와 해석 (k = 0, n = 203)

**macro-F1** (`metrics_within.csv`, `metrics_loio.csv`)

| 모델 | within τ=0 | 0.5 | 1.0 | LOIO τ=0 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| ① 규칙 | 0.821 | 0.816 | 0.827 | 0.821 | 0.816 | 0.827 |
| ② LR | **0.841** | **0.852** | **0.858** | **0.821** | **0.841** | 0.858 |
| ③ MLP | 0.790 | 0.809 | 0.856 | 0.813 | 0.832 | **0.860** |
| ④ LSTM | 0.776 | 0.760 | 0.788 | 0.799 | 0.800 | 0.833 |

**규칙 대비 차이 95% CI (LOIO, `bootstrap_loio.csv`)**: LR −0.04~+0.08, MLP −0.07~+0.10, LSTM −0.07~+0.06 → **전부 0 포함.**

**LOIO 교차로별 macro-F1 (τ = 0)**: 10th 규칙 0.73 / MLP 0.79, 11th 규칙 0.88 / LR 0.86, 12th 모두 ≈ 0.81–0.83, 14th(n=14) LR·MLP 0.93.

**k = −1.5 s 민감도 (n = 196)**: 규칙 0.86–0.87로 최고, 학습 모델은 동급 이하(MLP는 τ=0.5에서 유의하게 낮음).

해석:
1. **규칙이 생각보다 강하다.** 규칙은 stop recall 0.90–0.94(놓치는 정지가 적음) 대신 stop precision 0.69(갈 차량을 서는 것으로 오판). 학습 모델은 이 균형을 옮길 뿐 전체적으로 크게 낫지 않다.
2. **LSTM이 이득을 못 본 이유 [추측]**: (a) 203건은 시퀀스 모델에 작다, (b) 황색 *이전* 2 s 이력은 아직 반응 전이라 정보가 적고, 반응이 보이는 τ = 1 s에서야 LSTM이 따라붙는다(0.83), (c) NGSIM 속도/가속도 노이즈.
3. **SinD 창춘과의 차이**: 창춘은 저속(중앙값 5.9 m/s, d 12 m)이고 "갈 수 있는데 서는" 운전자가 많아 규칙이 약했다. Peachtree는 11 m/s, d 35 m의 전형적 접근이라 등속 규칙이 잘 맞는다 **[추측]**.
4. **정렬 민감도가 모델 차이보다 크다** → 신호 시계가 정확한 데이터(SinD 전체)가 핵심.

## 5. 면접 예상 질문

**Q1. 딥러닝(LSTM)이 단순 규칙을 이기지 못했습니다. 이 결과의 가치는?**
- 한: 부정적 결과도 설계를 바꾸는 정보입니다. 첫째, 등속 규칙이 전형적 접근 속도에서는 macro-F1 0.82 수준의 강한 기준선이라는 것, 둘째, 학습 모델의 이득은 운전 문화나 속도대(창춘 저속)에 따라 달라질 수 있다는 것, 셋째, 신호 정렬 1.5 s 오차가 모델 간 차이보다 크다는 것을 정량화했습니다. 그래서 다음 실험은 모델 복잡도가 아니라 정확한 SPaT 시계와 교차로 수를 늘리는 데 집중합니다. 주기 단위 bootstrap CI로 차이가 유의하지 않음을 보여 과장된 주장을 피했습니다.
- EN: A negative result still redirects the work. It shows the constant-speed rule is a strong baseline (macro-F1 ≈ 0.82) at typical approach speeds, that any learned gain may depend on speed regime and driver population, and that a 1.5 s signal-alignment error moves results more than the choice of model does. The next experiments therefore target accurate signal timing and more intersections rather than bigger models. The cycle-level bootstrap CIs keep me from overclaiming.

**Q2. 왜 일반 K-fold가 아니라 황색 주기 단위 GroupKFold와 cluster bootstrap을 썼나요?**
- 한: 같은 황색 주기의 차량들은 같은 신호 이벤트를 공유하고, 뒤차는 앞차를 따라 서는 등 결정이 상관돼 있습니다. 행 단위로 섞으면 같은 주기의 앞차가 train, 뒤차가 test에 들어가 사실상 정답을 엿보게 되고, 독립 표본처럼 취급하면 신뢰구간이 지나치게 좁아집니다. 그래서 분할과 재표본 모두 주기를 단위로 했습니다.
- EN: Vehicles in the same yellow cycle share the signal event, and followers often copy the leader's decision, so their outcomes are correlated. Row-level folds would leak a leader's outcome into the test set of its follower, and treating events as independent would make the intervals too narrow. I therefore split and resampled at the cycle level.

**Q3. 문서화되지 않은 신호-궤적 시간 정렬을 어떻게 다뤘나요?**
- 한: "직진 차량은 적신호에 정지선을 거의 넘지 않는다"는 물리적 불변량을 검정 통계로 써 오프셋을 스캔했습니다. 시간대-파일 짝은 반대로 짝지었을 때 적신호 통과율이 11.5%에서 43%로 뛰어 확정했고, 프레임 오프셋은 0 ~ −1.5 s 범위까지만 좁혀졌습니다. 그래서 하나를 정답으로 고르지 않고 k = 0 본 실험과 k = −1.5 s 민감도 분석을 둘 다 보고했고, 결론이 바뀐다는 사실 자체를 핵심 한계로 기록했습니다.
- EN: I used a physical invariant — through vehicles rarely cross on red — as a test statistic and scanned offsets. It settled the period-to-file pairing decisively (red crossings 11.5% vs 43% when swapped), but only narrowed the frame offset to 0 to −1.5 s. Rather than picking one value, I report a main run at k = 0 plus a sensitivity run at k = −1.5 s, and flag that the conclusion changes between them as a key limitation.
