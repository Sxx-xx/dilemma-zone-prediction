# Predicting Stop-or-Go Decisions of the Preceding Human-Driven Vehicle at Yellow Onset Using Signal Information: A Comparison of Kinematic Rules and Learned Models

Can an automated vehicle that receives C-ITS **SPaT** (signal phase and timing) predict whether the
**human-driven vehicle ahead of it in the same lane (the preceding vehicle)** will stop or go at yellow onset?

Each through event at yellow onset is split into a platoon **leader** (no unpassed vehicle ahead in its lane) and a
**follower**. Kinematic rules — A: `d / v < t_r` (reach the stop line before red), B: required deceleration
`v² / 2d > b` — are compared with logistic regression, MLP, LSTM and Transformer under the same
leave-one-recording-out cross-validation, with cycle-level bootstrap 95 % CIs.

## Key results
**FLUID FI (Xuancheng, 1 intersection, 151 through events = 88 leaders + 63 followers)**
- **Followers stopped whenever their leader stopped (33 / 33)**, and no model beat rule A on followers →
  the decision worth predicting is the **leader's**.
- Leaders (88, 55 go): rule A (t_r = recorded 3 s yellow) macro-F1 **0.700**, logistic regression on (d, v, a) **0.881**,
  difference **+0.185 [95 % CI +0.075, +0.289]**.

**V2X-Seq TFD (Beijing Yizhuang, roadside sensors, 23 intersections, 1,867 leaders; leave-one-intersection-out)**
- Rule A with t_r = recorded 4 s yellow: macro-F1 **0.673**. Every learned model beats it by +0.16 – +0.19 (MLP 0.862).
- **The gap is mostly the threshold, not the model.** Drivers' go/stop boundary is **d/v ≈ 2.7 s**, 1.3 s shorter than
  the yellow. Rule A with t_r* estimated on the *training* intersections (2.63 – 2.70 s in all 23 folds) reaches
  **0.831**, and at approach speeds ≥ 10 m/s (1,418 events) **0.844 — no learned model beats it**. Learned models
  keep a gain only at < 10 m/s (+0.05 – +0.09) and MLP overall (+0.03 [+0.015, +0.046]).
- The boundary transfers across datasets: t_r* = 2.44 s estimated on FLUID applied unchanged to V2X-Seq gives 0.835,
  the same as V2X-Seq's own calibration.
- Acceleration does not help on V2X-Seq (LR(d, v) − LR(d, v, a) = −0.009, n.s.; the per-frame acceleration is sensor
  noise and smoothing it changes nothing). (x, y, v, a) sequence models collapse when transferred; (d, v, a) sequences
  match logistic regression but never beat it (learning curve: LR saturates at 25 % of the training cycles).
- Dilemma-zone band (d/v 2.5 – 5.5 s, 509 events): calibrated rule A 0.611, band-only LR 0.650, MLP 0.654, LSTM 0.61 – 0.64
  — indistinguishable; only full-training linear models fail there (0.46 – 0.50).

**Limitations:** two cities; roadside-sensor accelerations are noisy; V2X-Seq labels depend on a per-approach stop-line
calibration and 10 – 45 s label horizons; no real-vehicle validation. Figures for the paper: `results/figures/`.

- What was verified vs. assumed (all numbers): [`docs/FINDINGS.md`](docs/FINDINGS.md)
- Plan & status: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Study notes (Korean): [`docs/notes/`](docs/notes/)

## Layout
```
src/dzp/
  sind.py, fluid.py, waterloo.py, ngsim.py   dataset loaders (FLUID/Waterloo converted to the SinD Recording format)
  lanelet_map.py  Lanelet2 .osm reader (UTM projection as in SinD tooling), stop_line ways
  geometry.py     oriented stop lines, signed distance, stop-line crossings, turn angle
  events.py       yellow-onset candidates, shared stop/go labeling, sequences
  features.py     per-event features incl. same-lane leader (ldr_*) columns
  bench_models.py rules A/B, LR(d,v,a), LR(d,v), MLP, LSTM, Transformer (common fit/proba interface)
  bench_eval.py   leave-one-group-out splits, out-of-fold predictions, paired cycle bootstrap
  models.py, evaluate.py   earlier SinD/NGSIM experiment code
scripts/
  run_benchmark.py   dataset-agnostic yellow-onset benchmark (--dataset fluid|sind|v2xseq, --split all|leader|follower)
  v2xseq_extract.py, v2xseq_eval.py, v2xseq_followup.py, paper_materials.py   V2X-Seq leader evaluation, follow-ups (A-E), figures
  leader_follower.py roles, leader features for followers, leader/follower risk pairs
  fluid_check.py, fluid_visibility.py, sind_check_alignment.py, run_sind.py, run_ngsim.py, lankershim_*.py
results/          CSV/JSON outputs + figures (committed; raw data is not)
tests/            labeling / rule unit tests
```

## Data (not redistributed — put under `data/raw/`)
- **FLUID** (Xuancheng, China; Scientific Data 2026; CC BY 4.0): figshare doi `10.6084/m9.figshare.29974954` v2,
  `derived_data` zips 64179079 (TI), 64179085 (FI), 64179082 (FIDRT) → `data/raw/fluid/{TI,FI,FIDRT}/`
  (checksums in `data/raw/fluid/SOURCE.txt`). Main experiments use FI (W/E approaches).
- **V2X-Seq TFD** (DAIR-V2X-Seq, Beijing; single-infrastructure trajectories + traffic-light CSVs + HD maps) →
  `data/raw/v2xseq/full/extracted/V2X-Seq-TFD/`; adapter `src/dzp/v2xseq.py`, calibrated stop lines
  `results/v2xseq/stopline_shift.csv` (see `docs/FINDINGS.md`, "V2X-Seq").
- **SinD** sample: `git clone https://github.com/SOTIF-AVLab/SinD` → copy `Data/*` to `data/raw/sind/`
  (commit used: `data/raw/sind/SOURCE_COMMIT`). Non-commercial license; full dataset by application.
- **NGSIM Peachtree**: trajectories from `https://data.transportation.gov/resource/8ect-6jqj.csv?$where=location='peachtree'`
  (→ `data/raw/ngsim/peachtree/peachtree_trajectories.csv`), and the signal-change text files from the
  "Peachtree-Street-Atlanta-GA.zip" attachment of dataset 8ect-6jqj
  (`peachtree-street-signal-timing/signal-timing/signal-change/Original Text files/*.txt` → `data/raw/ngsim/peachtree/signal_change/`).
- **NGSIM Lankershim** (auxiliary; intersection 2 only): the supporting-data ZIP on data.transportation.gov is
  truncated on the server. An intact copy of `Lankershim-Main-Data.zip` (trajectories, signal timing, CAD) was taken
  from the **Wayback Machine archive of the old NGSIM community site**, not from an official USDOT server:
  `https://web.archive.org/web/20150324165048id_/http://ngsim-community.org/Lankershim-Main-Data.zip`
  (capture 2015-03-24, fetched 2026-10-11, 375,226,084 bytes,
  sha256 `d615aae87efe0b845e46254da34ebe07bfa1f4ebc93206730230a7a6fca7240c`, `unzip -t`: 122 entries OK)
  → `data/raw/ngsim/lankershim/` (with `SOURCE.txt` and `unzip_t.log`).

## Setup / run
```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sind_check_alignment.py
.venv/bin/python scripts/run_sind.py
.venv/bin/python scripts/run_ngsim.py          # signal offset k = 0
.venv/bin/python scripts/run_ngsim.py -15      # sensitivity: k = -1.5 s
# main benchmark on FLUID FI (rules A/B, LR, MLP, LSTM, Transformer; see docs/FINDINGS.md)
FI="--dataset fluid --root data/raw/fluid/FI/derived_data --site FI \
    --approaches W_approach,E_approach --light-map W_approach=light_E_s,E_approach=light_E_s"
.venv/bin/python scripts/run_benchmark.py $FI --out results/fluid_fi --event-dir data/raw/fluid/results/benchmark_fi
.venv/bin/python scripts/run_benchmark.py $FI --split leader --schemes group --out results/fluid_fi/leader_follower/leader \
    --event-dir data/raw/fluid/results/benchmark_fi_leader
.venv/bin/python scripts/leader_follower.py $FI --out results/fluid_fi/leader_follower
env -u PYTHONPATH .venv/bin/python -m pytest -q tests   # (unset PYTHONPATH if ROS is sourced)
```
