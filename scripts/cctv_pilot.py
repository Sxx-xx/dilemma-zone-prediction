"""CCTV pilot: public YOLO weights + ByteTrack on one AI Hub image sequence.

usage: cctv_pilot.py FRAME_DIR [--model yolo11m.pt] [--imgsz 1920] [--tracker bytetrack.yaml] [--tag base]

Writes per-frame tracks to results/cctv_pilot/local/ (gitignored: derived from licensed frames).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}   # COCO ids
OUT = Path(__file__).resolve().parents[1] / "results/cctv_pilot/local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n", type=int, default=0, help="first n frames only (0 = all)")
    a = ap.parse_args()

    files = sorted(Path(a.frames).glob("*.jpg"))
    if a.n:
        files = files[: a.n]
    model = YOLO(a.model)
    rows, t0 = [], time.time()
    for k, f in enumerate(files):
        r = model.track(str(f), persist=True, tracker=a.tracker, imgsz=a.imgsz,
                        classes=list(VEHICLES), conf=0.1, verbose=False)[0]
        b = r.boxes
        if b.id is None:
            continue
        for (x1, y1, x2, y2), tid, c, s in zip(b.xyxy.tolist(), b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist()):
            rows.append((k, tid, VEHICLES[c], s, x1, y1, x2, y2))
    df = pd.DataFrame(rows, columns=["frame", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"tracks_{a.tag}.csv", index=False)
    print(f"{a.tag}: {len(files)} frames in {time.time() - t0:.0f} s, {len(df)} boxes, {df.id.nunique()} ids")


if __name__ == "__main__":
    main()
