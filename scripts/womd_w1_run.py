"""W1 runner: per shard (download -> extract -> save event tables -> delete shard), resumable, with disk guards.

  nohup setsid nice -n 19 .venv/bin/python scripts/womd_w1_run.py --splits validation,training > data/raw/womd/w1/run.log 2>&1 &

Outputs per shard under --out/<split>/<shard>.{events.csv,feat.csv,seq.npz,meta.json}; state in --out/state.json
(done / failed); markers --out/<SPLIT>_DONE and TRAINING_50. Downloads: --downloads gsutil threads; extraction:
--workers spawn processes (each loads the Michigan index pickle of its split). Stops (marker STOPPED_DISK) when free
space < --keep-free-gb; never holds more than --max-inflight-gb of shards on disk.
"""
import argparse
import json
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

BUCKET = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
_G = {}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def shard_list(split, out):
    cache = f"{out}/shards_{split}.txt"
    if not os.path.exists(cache):
        txt = subprocess.run(["gsutil", "ls", f"{BUCKET}/{split}/"], check=True, capture_output=True, text=True).stdout
        open(cache, "w").write(txt)
    return [l.strip() for l in open(cache) if l.strip().endswith(("-01000", "-00150"))]


def free_gb(path):
    return shutil.disk_usage(path).free / 2 ** 30


def _init(index_pkl):
    os.nice(19)
    from dzp import womd
    _G["womd"] = womd
    _G["idx"] = pickle.load(open(index_pkl, "rb")) if index_pkl and os.path.exists(index_pkl) else None


def extract_one(task):
    path, split, out = task
    womd = _G["womd"]
    name = os.path.basename(path)
    try:
        ev, feat, seq, meta, acc = womd.extract_shard(path, _G["idx"], stitch=True)
        base = f"{out}/{split}/{name}"
        ev.to_csv(base + ".events.csv", index=False)
        feat.to_csv(base + ".feat.csv", index=False)
        np.savez_compressed(base + ".seq.npz", seq=seq)
        meta["accel"] = dict(n=int(len(acc)), da_raw=float(acc["da_raw"].mean()) if len(acc) else None, da_smooth=float(acc["da_smooth"].mean()) if len(acc) else None,
                             da_raw_median=float(acc["da_raw"].median()) if len(acc) else None, da_smooth_median=float(acc["da_smooth"].median()) if len(acc) else None)
        meta["n_events"] = int(len(ev))
        meta["bytes"] = os.path.getsize(path)
        json.dump(meta, open(base + ".meta.json", "w"), default=int)
        return name, None, meta
    except Exception:
        return name, traceback.format_exc(), None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="validation,training")
    ap.add_argument("--out", default="data/raw/womd/w1")
    ap.add_argument("--michigan", default="data/raw/womd/michigan_tls")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--downloads", type=int, default=2)
    ap.add_argument("--keep-free-gb", type=float, default=20.0)
    ap.add_argument("--max-inflight-gb", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many shards per split")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/tmp", exist_ok=True)
    state_p = f"{args.out}/state.json"
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"done": [], "failed": {}}

    def save_state():
        json.dump(state, open(state_p + ".tmp", "w"), indent=1)
        os.replace(state_p + ".tmp", state_p)

    for split in args.splits.split(","):
        os.makedirs(f"{args.out}/{split}", exist_ok=True)
        shards = shard_list(split, args.out)
        if args.limit:
            shards = shards[:args.limit]
        todo = [s for s in shards if os.path.basename(s) not in state["done"]]
        log(f"{split}: {len(shards)} shards, {len(todo)} to do")
        n_total = len(shards)
        idx_pkl = f"{args.michigan}/index_{split}.pkl"
        waited = 0
        while not os.path.exists(idx_pkl) and os.path.isdir(f"{args.michigan}/{split}") and waited < 7200:
            if waited == 0:
                log(f"waiting for {idx_pkl} (Michigan index build)")
            time.sleep(30)
            waited += 30
        if not os.path.exists(idx_pkl):
            log(f"WARNING: {idx_pkl} missing -> original signals only")
        ready, pending = queue.Queue(), queue.Queue()
        for s in todo:
            pending.put(s)
        inflight = {"bytes": 0}
        lock = threading.Lock()
        stop_flag = {"disk": False}

        def downloader():
            while True:
                try:
                    uri = pending.get_nowait()
                except queue.Empty:
                    return
                while True:
                    fr = free_gb(args.out)
                    with lock:
                        over = inflight["bytes"] / 2 ** 30 > args.max_inflight_gb - 0.5
                    if fr < args.keep_free_gb:
                        stop_flag["disk"] = True
                        log(f"STOP: free space {fr:.1f} GB < {args.keep_free_gb} GB")
                        open(f"{args.out}/STOPPED_DISK", "w").write(f"{time.strftime('%F %T')} free {fr:.1f} GB\n")
                        return
                    if not over:
                        break
                    time.sleep(5)
                if stop_flag["disk"]:
                    return
                local = f"{args.out}/tmp/{os.path.basename(uri)}"
                with lock:
                    inflight["bytes"] += 500 * 2 ** 20
                for attempt in range(3):
                    r = subprocess.run(["gsutil", "-q", "cp", uri, local], capture_output=True, text=True)
                    if r.returncode == 0 and os.path.exists(local):
                        break
                    log(f"download retry {attempt + 1} {os.path.basename(uri)}: {r.stderr.strip()[-200:]}")
                    time.sleep(10)
                if os.path.exists(local):
                    with lock:
                        inflight["bytes"] += os.path.getsize(local) - 500 * 2 ** 20
                    ready.put(local)
                else:
                    with lock:
                        inflight["bytes"] -= 500 * 2 ** 20
                    state["failed"][os.path.basename(uri)] = "download failed"
                    save_state()

        threads = [threading.Thread(target=downloader, daemon=True) for _ in range(args.downloads)]
        for t in threads:
            t.start()
        n_done_before = len([s for s in shards if os.path.basename(s) in state["done"]])
        half_marked = os.path.exists(f"{args.out}/TRAINING_50")
        with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(idx_pkl,)) as pool:
            results = []
            submitted = 0
            while submitted < len(todo):
                if stop_flag["disk"] and ready.empty() and all(not t.is_alive() for t in threads):
                    break
                try:
                    local = ready.get(timeout=5)
                except queue.Empty:
                    if all(not t.is_alive() for t in threads) and ready.empty():
                        break
                    continue
                results.append(pool.apply_async(extract_one, ((local, split, args.out),)))
                submitted += 1
                # reap finished
                for r in [r for r in results if r.ready()]:
                    results.remove(r)
                    name, err, meta = r.get()
                    with lock:
                        inflight["bytes"] = max(0, inflight["bytes"] - (meta["bytes"] if meta else 500 * 2 ** 20))
                    if err:
                        state["failed"][name] = err[-500:]
                        log(f"FAILED {name}: {err.splitlines()[-1]}")
                    else:
                        state["done"].append(name)
                        state["failed"].pop(name, None)
                        n_done = len([s for s in shards if os.path.basename(s) in state["done"]])
                        log(f"{split} {n_done}/{n_total} {name} scenarios {meta['n_scenarios']} events {meta['n_events']} stitched {meta['n_multi']} michigan {meta['n_michigan']} free {free_gb(args.out):.0f} GB")
                        if split == "training" and not half_marked and n_done >= n_total / 2:
                            open(f"{args.out}/TRAINING_50", "w").write(time.strftime("%F %T\n"))
                            half_marked = True
                    save_state()
            for r in results:
                name, err, meta = r.get()
                with lock:
                    inflight["bytes"] = max(0, inflight["bytes"] - (meta["bytes"] if meta else 500 * 2 ** 20))
                if err:
                    state["failed"][name] = err[-500:]
                    log(f"FAILED {name}: {err.splitlines()[-1]}")
                else:
                    state["done"].append(name)
                    state["failed"].pop(name, None)
                    log(f"{split} {len([s for s in shards if os.path.basename(s) in state['done']])}/{n_total} {name} events {meta['n_events']}")
                save_state()
        if stop_flag["disk"]:
            log("stopped on disk guard; rerun to resume")
            return
        n_done = len([s for s in shards if os.path.basename(s) in state["done"]])
        log(f"{split} finished: {n_done}/{n_total} done, {sum(1 for k in state['failed'] if k.startswith(split))} failed")
        open(f"{args.out}/{split.upper()}_DONE", "w").write(time.strftime("%F %T\n"))


if __name__ == "__main__":
    main()
