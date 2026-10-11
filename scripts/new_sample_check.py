"""Checks on the AI Hub "교차로 신호 체계, 보행자, 차량 이동 복합 데이터" sample (data/raw/New_Sample).

Prints: image sampling inside each source video, label types/attributes, and recall of public YOLO11m
detections against the labelled vehicle points / queue polylines (5 of 25 tasks). Numbers in docs/E_cctv_pilot.md §6.
"""
import collections
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data/raw/New_Sample"


def labels():
    rows, labs, attrs = [], collections.Counter(), collections.Counter()
    for x in sorted(ROOT.glob("라벨링데이터/*/*/*.xml")):
        for im in ET.parse(x).getroot().iter("image"):
            mm, ss = im.findtext("time").split(":")
            for c in im:
                if c.tag in ("points", "polyline"):
                    labs[(c.tag, c.get("label"))] += 1
                    for a in c.iter("attribute"):
                        if a.get("name").lower() != "car_count":
                            attrs[(a.get("name").lower(), a.text)] += 1
            rows.append((x.parent.name, im.get("name"), im.findtext("video_id"), int(mm) * 60 + float(ss)))
    df = pd.DataFrame(rows, columns=["inter", "img", "vid", "t"])
    jpgs = {p.name for p in ROOT.glob("원천데이터/*/*/*/*.jpg")}
    gap = df.sort_values(["vid", "t"]).groupby("vid").t.diff().dropna()
    print(f"xml images={len(df)} jpgs={len(jpgs)} matched={len(jpgs & set(df.img))} videos={df.vid.nunique()}")
    print("images per video:", df.groupby("vid").size().value_counts().sort_index().to_dict())
    print("gap between images of one video [s]:", gap.describe().round(1).to_dict())
    print("filename suffix == <time> seconds:", (df.img.str.extract(r"_(\d+)\.jpg")[0].astype(int) == df.t).mean())
    print("labels:", labs.most_common(8))
    print("attributes:", sorted(attrs.items()))


def detection_recall(step=5):
    from ultralytics import YOLO
    m = YOLO("yolo11m.pt")
    res = collections.defaultdict(lambda: np.zeros(4, int))
    for x in sorted(ROOT.glob("라벨링데이터/*/*/*.xml"))[::step]:
        inter = x.parent.name
        for im in ET.parse(x).getroot().iter("image"):
            p = list(ROOT.glob(f"원천데이터/*/{inter}/*/{im.get('name')}"))
            if not p:
                continue
            b = m.predict(str(p[0]), imgsz=1920, classes=[2, 3, 5, 7], conf=0.25, verbose=False)[0].boxes.xyxy.numpy()
            for c in im:
                if c.tag not in ("points", "polyline"):
                    continue
                for xy in c.get("points").split(";"):
                    X, Y = map(float, xy.split(","))
                    ok = len(b) and ((b[:, 0] - 10 <= X) & (X <= b[:, 2] + 10) & (b[:, 1] - 10 <= Y) & (Y <= b[:, 3] + 10)).any()
                    res[inter] += [ok, 1, 0, 0] if c.tag == "points" else [0, 0, ok, 1]
    for k, (a, n, q, nq) in res.items():
        print(f"{k}: moving-vehicle points {a}/{n} = {a / n:.2f}; queue vertices {q}/{nq} = {q / nq:.2f}")


if __name__ == "__main__":
    labels()
    detection_recall()
