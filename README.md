# Predicting Stop-or-Go Decisions of the Preceding Human-Driven Vehicle at Yellow Onset Using Signal Information: A Comparison of Kinematic Rules and Learned Models

Can an automated vehicle that receives C-ITS **SPaT** (signal phase and timing) predict whether the
**human-driven vehicle ahead of it in the same lane (the preceding vehicle)** will stop or go at yellow onset?

Each through event at yellow onset is split into a platoon **leader** (no unpassed vehicle ahead in its lane) and a
**follower**. Kinematic rules — A: `d / v < t_r` (reach the stop line before red), B: required deceleration
`v² / 2d > b` — are compared with logistic regression, MLP, LSTM and Transformer under the same
leave-one-recording-out cross-validation, with cycle-level bootstrap 95 % CIs.

## Key results (FLUID FI, Xuancheng, 151 through events = 88 leaders + 63 followers)
- **Followers stopped whenever their leader stopped (33 / 33)**, and no model beat rule A on followers →
  the decision worth predicting is the **leader's**.
- **Leaders (88, 55 go):** rule A macro-F1 **0.700**, logistic regression on (d, v, a) **0.881**,
  difference **+0.185 [95 % CI +0.075, +0.289]**.
- **Dropping acceleration removes the gain:** logistic regression on (d, v) is +0.063 [−0.036, +0.163] vs rule A
  (not significant) → the benefit comes from deceleration already under way at yellow onset.

**Limitations:** one intersection only; no real-vehicle validation; the multi-intersection SinD dataset
(full release) is still awaiting approval.

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
  run_benchmark.py   dataset-agnostic yellow-onset benchmark (--dataset fluid|sind, --split all|leader|follower)
  leader_follower.py roles, leader features for followers, leader/follower risk pairs
  fluid_check.py, fluid_visibility.py, sind_check_alignment.py, run_sind.py, run_ngsim.py, lankershim_*.py
results/          CSV/JSON outputs + figures (committed; raw data is not)
tests/            labeling / rule unit tests
```

## Data (not redistributed — put under `data/raw/`)
- **FLUID** (Xuancheng, China; Scientific Data 2026; CC BY 4.0): figshare doi `10.6084/m9.figshare.29974954` v2,
  `derived_data` zips 64179079 (TI), 64179085 (FI), 64179082 (FIDRT) → `data/raw/fluid/{TI,FI,FIDRT}/`
  (checksums in `data/raw/fluid/SOURCE.txt`). Main experiments use FI (W/E approaches).
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
.venv/bin/python scripts/run_benchmark.py $FI --split leader --out results/fluid_fi/leader_follower/leader \
    --event-dir data/raw/fluid/results/benchmark_fi_leader
.venv/bin/python scripts/leader_follower.py $FI --out results/fluid_fi/leader_follower
env -u PYTHONPATH .venv/bin/python -m pytest -q tests   # (unset PYTHONPATH if ROS is sourced)
```
