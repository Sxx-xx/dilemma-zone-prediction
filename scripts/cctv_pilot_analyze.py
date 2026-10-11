"""Analyse ByteTrack output of the CCTV pilot (scripts/cctv_pilot.py).

usage: cctv_pilot_analyze.py FRAME_DIR TAG [TAG ...]

- ID continuity: track lengths, suspected ID switches (a track ends and a new one starts nearby within 2 frames),
  continuity of vehicles crossing the far-approach stop line.
- Visible upstream range: flat-road perspective model from the burned-in lane boxes, scaled by queue spacing.
- Writes numbers to results/cctv_pilot/summary_<tag>.json (committed) and figures / video to
  results/cctv_pilot/local/ (gitignored: they contain AI Hub frames).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results/cctv_pilot"
LOCAL = RES / "local"

# Far-approach lane edges measured from the orange ROI overlay (full-res px), rows 222 and 366 (frame 300).
EDGES_222 = np.array([562, 622, 686, 753, 812, 870, 923])
EDGES_366 = np.array([361, 476, 603, 730, 852, 967, 1079])
Y_STOP = 392            # bottom of the orange boxes = far stop line (px row)
Y_H = 78                # horizon row where lane edges converge (lane width w = k (y - Y_H))
QUEUE_SPACING_M = (6.5, 7.5)   # front-to-front spacing of stopped cars [m] (assumption)
LANE_W_M = (3.0, 3.5)


def lane_of(x, y):
    """Lane index 0..5 of the far approach at image point (x, y), or -1; edges extrapolated linearly in y."""
    e = EDGES_222 + (EDGES_366 - EDGES_222) * (y - 222) / (366 - 222)
    i = np.searchsorted(e, x) - 1
    return int(i) if 0 <= i < len(e) - 1 else -1


def load(tag):
    df = pd.read_csv(LOCAL / f"tracks_{tag}.csv")
    df["cx"] = (df.x1 + df.x2) / 2
    df["by"] = df.y2                       # bottom centre ~ ground contact
    df["h"] = df.y2 - df.y1
    return df


def continuity(df, n_frames, W=1920, H=1080):
    g = df.groupby("id")
    life = g.frame.agg(["min", "max", "size"])
    first = df.loc[g.frame.idxmin()].set_index("id")
    last = df.loc[g.frame.idxmax()].set_index("id")
    # suspected switch: track ends (not at border / last frame) and another starts within 2 frames, 1.5 box heights away
    switches = 0
    border = 40
    for tid, r in last.iterrows():
        if r.frame >= n_frames - 1 or r.cx < border or r.cx > W - border or r.by > H - border:
            continue
        cand = first[(first.frame > r.frame) & (first.frame <= r.frame + 2)]
        if len(cand) and (np.hypot(cand.cx - r.cx, cand.by - r.by) < 1.5 * max(r.h, 20)).any():
            switches += 1
    # far-approach stop-line crossings: ids seen upstream (in a lane, above stop line) and later >= 40 px past it
    up = df[(df.by < Y_STOP) & (df.by > 200) & df.apply(lambda r: lane_of(r.cx, r.by) >= 0, axis=1)]
    crossed = []
    for tid, d in df[df.id.isin(up.id.unique())].groupby("id"):
        d = d.sort_values("frame")
        if (d.by > Y_STOP + 40).any():
            pre = d[d.by < Y_STOP]
            crossed.append({"id": tid, "frames_upstream": len(pre), "first_by": d.by.min(),
                            "gaps": int((np.diff(d.frame) > 1).sum())})
    return {
        "ids": int(df.id.nunique()),
        "boxes_per_frame_mean": round(len(df) / n_frames, 1),
        "track_len_frames": life["size"].describe().round(1).to_dict(),
        "short_tracks_le3": int((life["size"] <= 3).sum()),
        "suspected_id_switches": switches,
        "far_stopline_crossings_same_id": len(crossed),
        "crossing_tracks_frames_upstream_median": float(np.median([c["frames_upstream"] for c in crossed])) if crossed else None,
    }, pd.DataFrame(crossed)


def range_model(df):
    """Z(y) = K / (y - Y_H); K from queue spacing of stopped cars in the first frames (red for the far approach)."""
    k_lane = ((EDGES_366[-1] - EDGES_366[0]) / 6) / (366 - Y_H)     # px of lane width per px of (y - Y_H)
    cam_h = [w / k_lane for w in LANE_W_M]
    q = df[(df.frame < 5) & (df.by < Y_STOP + 15) & (df.by > Y_H + 20)].copy()
    q["lane"] = [lane_of(x, y) for x, y in zip(q.cx, q.by)]
    inv = []
    for (f, ln), d in q[q.lane >= 0].groupby(["frame", "lane"]):
        u = np.sort(1.0 / (d.by.values - Y_H))[::-1]          # nearest first
        inv += list(-np.diff(u)[:6])                         # consecutive queue positions
    inv = np.array([v for v in inv if v > 0])
    dinv = float(np.median(inv)) if len(inv) else np.nan
    Ks = [s / dinv for s in QUEUE_SPACING_M]
    z = lambda y, K: K / (y - Y_H)
    # farthest row with reliable detections in far-approach lanes: 5th percentile of bottom rows of in-lane boxes
    lanebox = df[np.array([lane_of(x, y) >= 0 for x, y in zip(df.cx, df.by)]) & (df.by < Y_STOP) & (df.by > Y_H + 5)]
    y_far = float(np.percentile(lanebox.by, 5)) if len(lanebox) else np.nan
    y_box_top = 205
    return {
        "camera_height_m_from_lane_width": [round(h, 1) for h in cam_h],
        "queue_pairs_used": int(len(inv)),
        "K_range": [round(k, 1) for k in Ks],
        "stopline_dist_from_camera_m": [round(z(Y_STOP, k), 1) for k in Ks],
        "roi_box_top_upstream_of_stopline_m": [round(z(y_box_top, k) - z(Y_STOP, k), 1) for k in Ks],
        "y_far_p5": round(y_far, 1),
        "far_detection_upstream_of_stopline_m": [round(z(y_far, k) - z(Y_STOP, k), 1) for k in Ks],
    }


def figures(df, frames, tag, crossed):
    LOCAL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    col = {i: tuple(int(c) for c in rng.integers(60, 255, 3)) for i in df.id.unique()}
    # 1) trajectories over frame 0
    img = cv2.imread(str(frames[0]))
    for tid, d in df.groupby("id"):
        p = d.sort_values("frame")[["cx", "by"]].values.astype(int)
        if len(p) > 1:
            cv2.polylines(img, [p], False, col[tid], 2)
            cv2.putText(img, str(tid), tuple(p[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[tid], 2)
    cv2.line(img, (330, Y_STOP), (1100, Y_STOP), (0, 0, 255), 2)
    cv2.imwrite(str(LOCAL / f"traj_{tag}.jpg"), img)
    # 2) annotated video (2 fps source -> 4 fps playback)
    h, w = img.shape[:2]
    vw = cv2.VideoWriter(str(LOCAL / f"tracks_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (w // 2, h // 2))
    by_f = dict(tuple(df.groupby("frame")))
    for k, f in enumerate(frames):
        im = cv2.imread(str(f))
        for _, r in by_f.get(k, pd.DataFrame()).iterrows():
            c = col[r.id]
            cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
            cv2.putText(im, str(int(r.id)), (int(r.x1), int(r.y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(im, f"frame {k}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        vw.write(cv2.resize(im, (w // 2, h // 2)))
    vw.release()
    # 3) filmstrip of the longest far-approach crossing tracks
    if len(crossed):
        strips = []
        for tid in crossed.sort_values("frames_upstream", ascending=False).id.head(4):
            d = df[df.id == tid].sort_values("frame")
            sel = d.iloc[np.linspace(0, len(d) - 1, min(8, len(d))).astype(int)]
            tiles = []
            for _, r in sel.iterrows():
                im = cv2.imread(str(frames[int(r.frame)]))
                pad = 30
                crop = im[max(int(r.y1) - pad, 0):int(r.y2) + pad, max(int(r.x1) - pad, 0):int(r.x2) + pad]
                crop = cv2.resize(crop, (160, 120))
                cv2.putText(crop, f"id{tid} f{int(r.frame)}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(crop)
            tiles += [np.zeros_like(tiles[0])] * (8 - len(tiles))
            strips.append(np.hstack(tiles))
        cv2.imwrite(str(LOCAL / f"filmstrip_{tag}.jpg"), np.vstack(strips))


def main():
    frames = sorted(Path(sys.argv[1]).glob("*.jpg"))
    for tag in sys.argv[2:]:
        df = load(tag)
        cont, crossed = continuity(df, len(frames))
        out = {"tag": tag, "n_frames": len(frames), **cont, "range": range_model(df)}
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"summary_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        crossed.to_csv(LOCAL / f"crossings_{tag}.csv", index=False)
        figures(df, frames, tag, crossed)
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
