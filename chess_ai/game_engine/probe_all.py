#!/usr/bin/env python3
"""
Comprehensive value + policy probe of several models on the FULL self-play window (all iters),
in one pass. Mirrors value_probe's value metrics and model_diff's policy metrics so numbers are
directly comparable.

Run from the REPO ROOT, models given as label=path:
    python3 chess_ai/game_engine/probe_all.py \
        pretrained=chess_ai/game_engine/model/pretrained.pth \
        champ=chess_ai/game_engine/model/best_model.pth \
        cand=chess_ai/game_engine/model/candidate.pth \
        --data chess_ai/data/self_play

Value rows (per model): mean|v|, |v|>=0.5/0.9, acc, P(win|win)/P(draw|draw)/P(loss|loss).
Policy pairs: top1-agree + KL(A‖B) over legal moves (k>=2).
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


def load(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else (
         raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval(); m.load_state_dict(sd, strict=False)   # tolerate pre-aux ckpts
    return m


def infer(model, x, bs=1024):
    """Batched forward (whole window won't fit activations in one shot)."""
    pol, val = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg, v = model(x[i:i + bs])
            pol.append(torch.softmax(lg, 1).numpy())
            val.append(torch.softmax(v, -1).numpy())
    return np.concatenate(pol), np.concatenate(val)


def sample_all(data_dir, games_per_iter, per_game):
    iters = sorted(int(os.path.basename(d).split("_")[1])
                   for d in glob.glob(os.path.join(data_dir, "iter_*")) if d.split("_")[-1].isdigit())
    S, P, V = [], [], []
    for it in iters:
        for f in sorted(glob.glob(os.path.join(data_dir, f"iter_{it}", "*.npz")))[:games_per_iter]:
            d = np.load(f); n = d["states"].shape[0]
            idx = np.linspace(0, n - 1, min(n, per_game)).astype(int)
            S.append(d["states"][idx]); P.append(d["policies"][idx]); V.append(d["values"][idx])
    return (np.concatenate(S), np.concatenate(P), np.concatenate(V).astype(int), iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label=path entries (e.g. champ=game_engine/model/best_model.pth)")
    ap.add_argument("--data", default="chess_ai/data/self_play")
    ap.add_argument("--games-per-iter", type=int, default=10**9, help="cap files per iter (default: all)")
    ap.add_argument("--per-game", type=int, default=10)
    ap.add_argument("--policy-sample", type=int, default=8000, help="positions used for the pairwise policy diff")
    a = ap.parse_args()
    torch.set_num_threads(8)

    models = []
    for m in a.models:
        label, path = m.split("=", 1)
        if not os.path.exists(path):
            print(f"  SKIP {label}: not found ({path})"); continue
        models.append((label, path))
    if not models:
        sys.exit("no valid models")

    S, P, V, iters = sample_all(a.data, a.games_per_iter, a.per_game)
    x = torch.from_numpy(S).float()
    dist = {CLASS[c]: int((V == c).sum()) for c in (0, 1, 2)}
    print(f"\nwindow iters {iters[0]}-{iters[-1]} | {len(V):,} positions | outcomes {dist}\n")

    pols = {}
    print("=== VALUE HEAD (all positions) ===")
    for label, path in models:
        pol, wdl = infer(load(path), x)
        av = np.abs(wdl[:, 0] - wdl[:, 2])               # |P(win)-P(loss)|
        acc = 100 * (wdl.argmax(1) == V).mean()
        by = {c: (wdl[V == c, c].mean() if (V == c).any() else float("nan")) for c in (0, 1, 2)}
        print(f"  {label:12s} mean|v|={av.mean():.3f}  |v|>=0.5:{100*(av>=0.5).mean():4.0f}%  "
              f"|v|>=0.9:{100*(av>=0.9).mean():4.0f}%  acc={acc:3.0f}%  "
              f"| P(win|win)={by[0]:.2f} P(draw|draw)={by[1]:.2f} P(loss|loss)={by[2]:.2f}")
        pols[label] = pol

    if len(models) >= 2:
        print(f"\n=== POLICY pairwise (first {min(a.policy_sample, len(P)):,} positions, legal moves, k>=2) ===")
        leg = P[:a.policy_sample] > 0
        labels = [l for l, _ in models]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                A_all, B_all = pols[labels[i]], pols[labels[j]]
                agree = kl = 0.0; valid = 0
                for k in range(leg.shape[0]):
                    lm = leg[k]
                    if lm.sum() < 2:
                        continue
                    A = A_all[k][lm]; A = np.clip(A / A.sum(), 1e-9, 1)
                    B = B_all[k][lm]; B = np.clip(B / B.sum(), 1e-9, 1)
                    agree += (A.argmax() == B.argmax()); kl += float((A * np.log(A / B)).sum()); valid += 1
                valid = max(valid, 1)
                print(f"  {labels[i]:10s} vs {labels[j]:10s}: top1-agree={100*agree/valid:5.1f}%  KL={kl/valid:.3f}")


if __name__ == "__main__":
    main()
