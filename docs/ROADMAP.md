# ROADMAP — C-ITS(SPaT) 기반 주변 유인차량 황색 정지/통과 예측

## 연구 질문
SPaT를 수신하는 자율차가 **주변 사람 운전 차량**의 황색 신호 정지/통과를 예측할 때,
시계열 LSTM이 기존 규칙(`d/v < t_r`)보다 정확하고, **처음 보는 교차로**(leave-one-intersection-out)에서도 일반화되는가?

비교 모델: ① `d/v < t_r` 규칙 ② 로지스틱 회귀(순간값) ③ MLP(순간값) ④ LSTM(황색 직전 2 s 시계열 + 잔여시간)
평가: accuracy / precision / recall / F1 (positive = **stop**), macro-F1, 황색 후 τ = 0 / 0.5 / 1 s, within-site CV vs LOIO,
yellow-cycle 단위 paired cluster bootstrap 95% CI (규칙 대비 차이).

## 현재 상태 (2026-10-11)
| 단계 | 상태 | 결과물 |
|---|---|---|
| 0. 레포/환경 | 완료 | `.venv`, `requirements.txt`, `tests/` |
| 1. SinD 샘플 로더·정지선·라벨링 | 완료 | `src/dzp/{sind,lanelet_map,geometry,events}.py`, `results/sind/` |
| 2. NGSIM 적용 가능성 + ①~④ 전체 비교 | 완료 (Peachtree) | `src/dzp/ngsim.py`, `results/ngsim_peachtree/` |
| 3. SinD 전체 데이터로 본 실험 | **승인 대기** | — |

자세한 확인/추측 구분은 [`docs/FINDINGS.md`](FINDINGS.md), 단계별 공부 노트는 [`docs/notes/`](notes/).

## 다음 단계

### A. SinD 전체 데이터 도착 시 (최우선)
1. `sind.SAMPLE_RECORDINGS`를 전체 녹화 목록으로 일반화 (도시/교차로별 디렉터리 스캔, 지도 매핑 테이블).
2. 녹화마다 `scripts/sind_check_alignment.py` 실행 → 신호-궤적 정렬(red-crossing 비율)과 light→approach 매핑을 자동 검증.
   임계값(예: red-crossing > 8%) 넘는 녹화는 리포트 후 제외/수동 검토.
   - Xi'an 샘플은 최적 오프셋이 +4.5 s(red-crossing 4.0% vs 0 s에서 7.7%) → 전체 데이터에서 녹화별 오프셋 추정이 필요한지 확인.
3. 교차로 ≥ 4곳에서 이벤트 수 확인 (샘플에서는 Changchun만 69건, 나머지는 0–1건) → LOIO 본 실험.
4. 오토바이/삼륜차(중국 데이터에 많음)를 별도 클래스로 넣을지 결정 — 현재는 car/truck/bus만.

### B. 방법론 보강
- **선행차 특징**: 같은 차로 선행차 유무·거리·선행차의 결정(앞차가 서면 뒤차도 선다). 현재 라벨이 "따라가는 차"의 종속 결정을 포함하므로 first-to-stop 차량만 분리한 분석도 추가.
- **확률 출력 + 보정(calibration)**: 자율차 계획기는 0/1보다 P(stop)를 원함 → Brier score, reliability diagram.
- **규칙 개선 베이스라인**: 등속 대신 등가속 `d = v t + ½ a t²` 규칙, 그리고 학습된 임계값 `d/v - t_r < θ` (규칙의 1-파라미터 버전). LSTM이 이것보다 나은지가 진짜 질문.
- **τ 확장**: 황색 *이전* 시점(−0.5 s, SPaT로 황색 시작 예고 가능)에서의 예측 — C-ITS만의 장점.
- **NGSIM 신호 정렬**: Peachtree 신호 로그-궤적 프레임 오프셋 k를 approach별로 추정해 불확실성 전파 (현재 k=0 본 실험, k=−1.5 s 민감도 분석).
- **정지선**: Peachtree는 정지선이 데이터에 없음 → CAD(`Atlanta-Peachtree.dwg`)/정사영상에서 직접 추출해 추정값과 비교. 14th SB는 추정 근거 부족.
- **Lankershim**: 지원 데이터 ZIP이 서버에서 잘려 있음(신호 타이밍 접근 불가) → data.transportation.gov 담당자에게 재업로드 요청 후 재검토.

### C. 논문화
- 데이터: SinD(중국, 드론) + NGSIM Peachtree(미국, 영상) → 국가·센서가 다른 cross-dataset 일반화(train SinD → test Peachtree)도 실험.
- 기여 정리: (1) SPaT 잔여시간을 쓰는 주변차 의도 예측 벤치마크, (2) LOIO 일반화, (3) 규칙 대비 *언제(τ)* 학습 모델이 이득인지.
- 기존 K-City 실차 연구와 연결: 자차 판단 규칙 → 주변차 예측으로 확장.

## 재현
```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -r requirements.txt
# 데이터 배치: README의 "Data" 절 참고 (data/raw/는 git에 포함하지 않음)
.venv/bin/python scripts/sind_check_alignment.py
.venv/bin/python scripts/run_sind.py
.venv/bin/python scripts/run_ngsim.py        # k = 0
.venv/bin/python scripts/run_ngsim.py -15    # 신호 오프셋 민감도 (−1.5 s)
env -u PYTHONPATH .venv/bin/python -m pytest -q tests
```
