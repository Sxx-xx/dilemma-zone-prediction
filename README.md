# dilemma-zone-prediction

Can an automated vehicle that receives C-ITS **SPaT** (signal phase and timing) predict whether the
**human-driven vehicles around it** will stop or go at yellow onset — and does a sequence model (LSTM)
beat the classic constant-speed rule `d / v < t_r` and still generalize to an **unseen intersection**?

Compared predictors: ① rule `d/v < t_r` ② logistic regression ③ MLP (instantaneous features) ④ LSTM (2 s history + time-to-red).
Evaluated at τ = 0 / 0.5 / 1 s after yellow onset, within-site CV and leave-one-intersection-out (LOIO).

- Plan & status: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- What was verified vs. assumed: [`docs/FINDINGS.md`](docs/FINDINGS.md)
- Study notes (Korean): [`docs/notes/`](docs/notes/)

## Layout
```
src/dzp/
  lanelet_map.py  Lanelet2 .osm reader (UTM projection as in SinD tooling), stop_line ways
  sind.py         SinD loader: tracks, traffic-light event table (timestamp-aligned), map
  geometry.py     oriented stop lines, signed distance, stop-line crossings, turn angle
  events.py       yellow-onset candidates, shared stop/go labeling, (N, T, 4) sequences
  ngsim.py        NGSIM Peachtree loader, signal-change logs, stop-line estimation, events
  models.py       ① RuleDV ② LogReg ③ MLP ④ LSTM (common fit/predict interface)
  evaluate.py     metrics, within-site GroupKFold by cycle, LOIO, cluster bootstrap
scripts/          sind_check_alignment.py, run_sind.py, run_ngsim.py
results/          CSV/JSON outputs + map overlays (committed; raw data is not)
tests/            labeling / rule unit tests
```

## Data (not redistributed — put under `data/raw/`)
- **SinD** sample: `git clone https://github.com/SOTIF-AVLab/SinD` → copy `Data/*` to `data/raw/sind/`
  (commit used: `data/raw/sind/SOURCE_COMMIT`). Non-commercial license; full dataset by application.
- **NGSIM Peachtree**: trajectories from `https://data.transportation.gov/resource/8ect-6jqj.csv?$where=location='peachtree'`
  (→ `data/raw/ngsim/peachtree/peachtree_trajectories.csv`), and the signal-change text files from the
  "Peachtree-Street-Atlanta-GA.zip" attachment of dataset 8ect-6jqj
  (`peachtree-street-signal-timing/signal-timing/signal-change/Original Text files/*.txt` → `data/raw/ngsim/peachtree/signal_change/`).

## Setup / run
```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sind_check_alignment.py
.venv/bin/python scripts/run_sind.py
.venv/bin/python scripts/run_ngsim.py          # signal offset k = 0
.venv/bin/python scripts/run_ngsim.py -15      # sensitivity: k = -1.5 s
env -u PYTHONPATH .venv/bin/python -m pytest -q tests   # (unset PYTHONPATH if ROS is sourced)
```
