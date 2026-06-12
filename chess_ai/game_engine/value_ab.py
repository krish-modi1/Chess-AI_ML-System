#!/usr/bin/env python3
"""
Offline A/B for value-head training knobs. Trains a candidate from best_model.pth on the existing
self-play window with a given (value_weight, position cap), then probes its value head vs the base.
No self-play / arena — ~one training pass (~minutes on GPU). Run from chess_ai/:

    python3 game_engine/value_ab.py --vw 1.0  --cap 0        # control (current behavior)
    python3 game_engine/value_ab.py --vw 0.5  --cap 0        # down-weight value loss
    python3 game_engine/value_ab.py --vw 0.25 --cap 0
    python3 game_engine/value_ab.py --vw 0.5  --cap 150      # + decorrelation
    python3 game_engine/value_ab.py --vw 0.5  --cap 80

Read the probe: a config whose value head gets SHARPER (higher mean|v| / |v|>=0.9) WITHOUT losing
argmax accuracy = the overfitting fix working. If nothing sharpens, the flatness was honest.
"""
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--vw",       type=float, default=1.0, help="VALUE_LOSS_WEIGHT")
ap.add_argument("--cap",      type=int,   default=0,   help="MAX_POSITIONS_PER_GAME (0=off)")
ap.add_argument("--draw-cap", type=int,   default=int(os.environ.get("DRAW_MAX_POSITIONS", 0)))
ap.add_argument("--base",     default="game_engine/model/best_model.pth")
ap.add_argument("--data",     default="data/self_play")
ap.add_argument("--out",      default="/tmp/ab_candidate.pth")
ap.add_argument("--epochs",   type=int,   default=int(os.environ.get("TRAIN_EPOCHS", 1)))
ap.add_argument("--lr",       type=float, default=float(os.environ.get("TRAIN_LR", 1e-4)))
ap.add_argument("--probe-games", type=int, default=40)
a = ap.parse_args()

# Set the knobs BEFORE importing trainer (its module-level constants read os.environ at import).
os.environ["VALUE_LOSS_WEIGHT"]      = str(a.vw)
os.environ["MAX_POSITIONS_PER_GAME"] = str(a.cap)
os.environ["DRAW_MAX_POSITIONS"]     = str(a.draw_cap)

import sys, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN
from trainer import train_model

print(f"\n=== A/B: value_weight={a.vw}  game_cap={a.cap}  draw_cap={a.draw_cap}  "
      f"epochs={a.epochs}  lr={a.lr} ===")
train_model(
    data_path=a.data,
    input_model_path=a.base,
    output_model_path=a.out,
    epochs=a.epochs,
    batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", 2048)),
    lr=a.lr,
    window_size=int(os.environ.get("TRAIN_WINDOW", 20)),
    total_iterations=1000,
)


def load(p):
    raw = torch.load(p, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval(); m.load_state_dict(sd)
    return m


def latest_iter(data_dir):
    its = [int(os.path.basename(d).split("_")[1])
           for d in glob.glob(os.path.join(data_dir, "iter_*")) if d.split("_")[-1].isdigit()]
    return max(its)


it = latest_iter(a.data)
S, V, P = [], [], []
for f in sorted(glob.glob(os.path.join(a.data, f"iter_{it}", "*.npz")))[:a.probe_games]:
    d = np.load(f); n = d["states"].shape[0]
    idx = np.linspace(0, n - 1, min(n, 15)).astype(int)
    S.append(d["states"][idx]); V.append(d["values"][idx]); P.append(d["policies"][idx])
S = np.concatenate(S); V = np.concatenate(V).astype(int); P = np.concatenate(P)
x = torch.from_numpy(S).float()
legal = P > 0                      # MCTS-visited support = legal/considered moves
tgt_top1 = P.argmax(1)


def stats(model, name):
    with torch.no_grad():
        pol, val = model(x)
    w = torch.softmax(val, -1).numpy()
    av = np.abs(w[:, 0] - w[:, 2])
    acc = 100 * (w.argmax(1) == V).mean()
    # policy health: top-1 among legal moves vs the MCTS target (catches policy degradation)
    masked = np.where(legal, pol.numpy(), -1e9)
    ptop1 = 100 * (masked.argmax(1) == tgt_top1).mean()
    print(f"  {name:18s} mean|v|={av.mean():.3f}  |v|>=0.9:{100*(av>=0.9).mean():4.0f}%  "
          f"val_acc={acc:3.0f}%  | policy_top1={ptop1:3.0f}%")


print(f"\n-- value head on iter_{it} ({len(V)} positions) --")
stats(load(a.base), "base (champ)")
stats(load(a.out),  f"vw={a.vw} cap={a.cap}")
