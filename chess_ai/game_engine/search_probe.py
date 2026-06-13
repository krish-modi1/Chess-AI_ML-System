#!/usr/bin/env python3
"""
Search-quality probe: how much does the N-sim MCTS *improve* on the raw network policy?

MCTS is AlphaZero's policy-improvement operator — the loop only learns if the search target
(the stored visit distribution) is BETTER than the net's own policy. If they're nearly identical,
the search isn't adding anything → bumping SIMULATIONS is the lever. If they differ a lot but
candidates still don't improve, the bottleneck is distillation (training), not search.

Run from chess_ai/:
    python3 game_engine/search_probe.py                 # best_model on latest iter data
    python3 game_engine/search_probe.py --iter 7 --games 200
    python3 game_engine/search_probe.py --model game_engine/model/candidate.pth

Compares, on real positions, the model's RAW policy vs the stored MCTS target (visit counts):
  top1-agree   how often net's best legal move == MCTS's pick   (HIGH = search just confirms net)
  KL(MCTS‖net) how far search moves the policy from the prior   (LOW  = search barely improves it)
  Δentropy     net entropy − MCTS entropy                       (>0   = search SHARPENS the policy)
  override     fraction where MCTS top move ≠ net top move      (LOW  = search rarely overrides)
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN


def load(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval(); m.load_state_dict(sd, strict=False)   # tolerate pre-aux checkpoints
    return m


def latest_iter(data_dir):
    its = [int(os.path.basename(d).split("_")[1])
           for d in glob.glob(os.path.join(data_dir, "iter_*")) if d.split("_")[-1].isdigit()]
    return max(its)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="game_engine/model/best_model.pth")
    ap.add_argument("--data",  default="data/self_play")
    ap.add_argument("--iter",  type=int, default=None)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--per-game", type=int, default=15)
    a = ap.parse_args()
    torch.set_num_threads(8)

    it = a.iter if a.iter is not None else latest_iter(a.data)
    S, P = [], []
    for f in sorted(glob.glob(os.path.join(a.data, f"iter_{it}", "*.npz")))[:a.games]:
        d = np.load(f); n = d["states"].shape[0]
        idx = np.linspace(0, n - 1, min(n, a.per_game)).astype(int)
        S.append(d["states"][idx]); P.append(d["policies"][idx])
    S = np.concatenate(S); P = np.concatenate(P)              # P = MCTS targets (sum to 1 over legal)

    model = load(a.model)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(S).float())
    netp = torch.softmax(logits, 1).numpy()

    agree = klsum = ent_m = ent_n = override = nlegal = 0.0
    valid = 0
    for i in range(len(P)):
        legal = P[i] > 0
        k = int(legal.sum())
        if k < 2:                       # need a real choice to talk about "improvement"
            continue
        m = P[i][legal]                 # MCTS dist over legal (sums to 1)
        nl = netp[i][legal]
        nl = nl / nl.sum()              # net dist renormalized to legal
        nl = np.clip(nl, 1e-9, 1.0)
        mt = np.clip(m, 1e-9, 1.0)
        same = np.argmax(m) == np.argmax(nl)
        agree    += same
        override += (not same)
        klsum    += float((m * np.log(mt / nl)).sum())
        ent_m    += float(-(m * np.log(mt)).sum())
        ent_n    += float(-(nl * np.log(nl)).sum())
        nlegal   += k
        valid    += 1

    N = max(valid, 1)
    print(f"\nsearch probe — model={os.path.basename(a.model)}  iter_{it}  ({valid} positions, "
          f"avg {nlegal/N:.1f} legal moves)")
    print(f"  top1-agree (net best == MCTS pick): {100*agree/N:5.1f}%   (HIGH ⇒ search just confirms net)")
    print(f"  override   (MCTS top ≠ net top)   : {100*override/N:5.1f}%   (LOW  ⇒ search rarely overrides)")
    print(f"  KL(MCTS‖net)                      : {klsum/N:6.3f} nats   (LOW  ⇒ search barely moves policy)")
    print(f"  entropy  net={ent_n/N:.2f}  MCTS={ent_m/N:.2f}   Δ={ent_n/N-ent_m/N:+.2f}   "
          f"(Δ>0 ⇒ search SHARPENS)")
    print("\n  read: high agree + low KL + small Δ  → search adds little → bump SIMULATIONS")
    print("        low agree / high KL but candidates still flat → fix TRAINING (distillation), not sims")


if __name__ == "__main__":
    main()
