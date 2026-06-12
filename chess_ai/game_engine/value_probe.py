#!/usr/bin/env python3
"""
Value-head decisiveness / calibration probe.

Compares the value head of two checkpoints (default: champion vs candidate) on real
self-play positions, to see whether training sharpened or flattened it. Run from the
chess_ai/ directory:

    python3 game_engine/value_probe.py                      # champ vs candidate, latest iter
    python3 game_engine/value_probe.py --iter 5             # sample from iter_5 data
    python3 game_engine/value_probe.py --champ A.pth --cand B.pth
    python3 game_engine/value_probe.py --cand "" --champ X.pth   # probe a single model

Metrics (value = P(win) - P(loss), from the side-to-move POV):
  mean|v|         average decisiveness (higher = more confident; ~0 = flat/under-confident)
  |v|>=0.9        share of positions the head calls clearly decided
  acc             argmax(WDL) == stored game outcome  (calibration; chance = 33%)
  win/draw/loss   mean predicted prob on the CORRECT class, grouped by actual outcome
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN

CLASS = {0: "win", 1: "draw", 2: "loss"}


def load_model(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model = ChessCNN().eval()
    model.load_state_dict(sd)
    return model


def sample_positions(data_dir, it, n_games, per_game):
    iter_dir = os.path.join(data_dir, f"iter_{it}")
    files = sorted(glob.glob(os.path.join(iter_dir, "*.npz")))
    if not files:
        sys.exit(f"No .npz found in {iter_dir}")
    S, V = [], []
    for f in files[:n_games]:
        d = np.load(f)
        n = d["states"].shape[0]
        idx = np.linspace(0, n - 1, min(n, per_game)).astype(int)
        S.append(d["states"][idx])
        V.append(d["values"][idx])
    return np.concatenate(S), np.concatenate(V).astype(int)


def stats(model, name, x, V):
    with torch.no_grad():
        _, val = model(x)
    wdl = torch.softmax(val, -1).numpy()
    sv = wdl[:, 0] - wdl[:, 2]               # signed value, P(win)-P(loss)
    av = np.abs(sv)
    acc = 100 * (wdl.argmax(1) == V).mean()
    # mean predicted prob on the correct class, per actual outcome
    by = {}
    for c in (0, 1, 2):
        m = V == c
        by[c] = wdl[m, c].mean() if m.any() else float("nan")
    print(f"  {name:14s} mean|v|={av.mean():.3f}  |v|>=0.5:{100*(av>=0.5).mean():4.0f}%  "
          f"|v|>=0.9:{100*(av>=0.9).mean():4.0f}%  acc={acc:3.0f}%  "
          f"| P(win|win)={by[0]:.2f} P(draw|draw)={by[1]:.2f} P(loss|loss)={by[2]:.2f}")


def latest_iter(data_dir):
    its = [int(os.path.basename(d).split("_")[1])
           for d in glob.glob(os.path.join(data_dir, "iter_*")) if d.split("_")[-1].isdigit()]
    return max(its) if its else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ", default="game_engine/model/best_model.pth")
    ap.add_argument("--cand",  default="game_engine/model/candidate.pth",
                    help='set to "" to probe only --champ')
    ap.add_argument("--data",  default="data/self_play")
    ap.add_argument("--iter",  type=int, default=None, help="iteration data dir (default: latest)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--per-game", type=int, default=15)
    a = ap.parse_args()

    it = a.iter if a.iter is not None else latest_iter(a.data)
    if it is None:
        sys.exit(f"No iter_* dirs in {a.data}")

    S, V = sample_positions(a.data, it, a.games, a.per_game)
    x = torch.from_numpy(S).float()
    dist = {CLASS[c]: int((V == c).sum()) for c in (0, 1, 2)}
    print(f"iter_{it}: {len(V)} positions  (outcomes: {dist})\n")

    for path, name in [(a.champ, "champ"), (a.cand, "candidate")]:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"  {name:14s} (skipped — file not found: {path})")
            continue
        stats(load_model(path), name, x, V)


if __name__ == "__main__":
    main()
