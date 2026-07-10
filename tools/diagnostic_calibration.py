#!/usr/bin/env python3
"""
Offline diagnostic: is the candidate WORSE than the champion, and if so is it the POLICY or the VALUE?
Literature-grounded, runs on checkpoints + self-play npz — no arena, no waiting for iterations.

Metrics (all vs GROUND TRUTH in the npz, per model, same positions):
  VALUE  — accuracy, mean|v|, decisive%%, and CALIBRATION: multiclass Brier score + top-label ECE
           (Guo et al. 2017, "On Calibration of Modern Neural Networks"). Distinguishes a value head
           that is SHARP-but-miscalibrated from one that is FLAT-but-honest.
  POLICY — top-1 agreement of the RAW net policy with the MCTS-improved target (the AlphaZero
           policy-improvement teacher), over legal moves (Maia-style move-match, McIlroy-Young 2020).

Also splits the window into OLD vs NEW iters: if the candidate is calibrated on recent data but
miscalibrated on older data, that is the off-distribution-brittleness signal.

Run from repo root:
    python3 tools/diagnostic_calibration.py \
        champ=chess_ai/game_engine/model/best_model.pth \
        cand=chess_ai/game_engine/model/candidate.pth \
        swa=chess_ai/game_engine/model/swa_model.pth \
        --data chess_ai/data/self_play --files-per-iter 15 --per-game 12
"""
import os, sys, glob, argparse, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_ai", "game_engine"))
from cnn import ChessCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS = {0: "win", 1: "draw", 2: "loss"}


def load(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else (
         raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Guard: if the value head didn't load (arch mismatch), its metrics would be random garbage.
    vh_missing = [k for k in missing if k.startswith("value_head")]
    if vh_missing:
        print(f"  ⚠️  {os.path.basename(path)}: value head NOT loaded ({len(vh_missing)} missing keys) "
              f"— arch mismatch, value metrics are RANDOM, skip this checkpoint for value.")
    return m.to(DEVICE)


def infer(model, x, bs=512):
    pol, val = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg, v = model(x[i:i + bs].to(DEVICE))
            pol.append(torch.softmax(lg, 1).cpu().numpy())
            val.append(torch.softmax(v, -1).cpu().numpy())
    return np.concatenate(pol), np.concatenate(val)


def sample(data_dir, files_per_iter, per_game):
    iters = sorted(int(os.path.basename(d).split("_")[1])
                   for d in glob.glob(os.path.join(data_dir, "iter_*")) if d.split("_")[-1].isdigit())
    S, P, V, IT = [], [], [], []
    for it in iters:
        for f in sorted(glob.glob(os.path.join(data_dir, f"iter_{it}", "*.npz")))[:files_per_iter]:
            try:
                d = np.load(f)
            except Exception:
                continue
            n = d["states"].shape[0]
            if n == 0:
                continue
            idx = np.linspace(0, n - 1, min(n, per_game)).astype(int)
            S.append(d["states"][idx]); P.append(d["policies"][idx])
            V.append(d["values"][idx]); IT.append(np.full(len(idx), it))
    return (np.concatenate(S), np.concatenate(P),
            np.concatenate(V).astype(int), np.concatenate(IT), iters)


def brier_multiclass(wdl, V):
    """Mean over samples of sum_c (p_c - onehot_c)^2. Range [0,2]; lower = better calibrated+accurate."""
    onehot = np.eye(3)[V]
    return float(((wdl - onehot) ** 2).sum(1).mean())


def ece_toplabel(wdl, V, n_bins=15):
    """Guo et al. 2017 top-label ECE: bin by max-confidence, weight |acc - conf| by bin mass."""
    conf = wdl.max(1); pred = wdl.argmax(1); correct = (pred == V).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def value_row(label, wdl, V):
    av = np.abs(wdl[:, 0] - wdl[:, 2])
    acc = 100 * (wdl.argmax(1) == V).mean()
    brier = brier_multiclass(wdl, V)
    ece = 100 * ece_toplabel(wdl, V)
    print(f"  {label:10s} acc={acc:4.1f}%  mean|v|={av.mean():.3f}  dec%={100*(av>=0.9).mean():4.1f}  "
          f"Brier={brier:.4f}  ECE={ece:4.1f}%")
    return brier, ece


def policy_match(label, pol, P, sample_n=8000):
    """Top-1 agreement of raw net policy with the MCTS target top move, over legal moves."""
    n = min(sample_n, len(P)); agree = valid = 0
    for k in range(n):
        lm = P[k] > 0
        if lm.sum() < 2:
            continue
        if pol[k][lm].argmax() == P[k][lm].argmax():
            agree += 1
        valid += 1
    print(f"  {label:10s} policy top-1 == MCTS-teacher top-1 : {100*agree/max(valid,1):5.1f}%  ({valid} pos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default="chess_ai/data/self_play")
    ap.add_argument("--files-per-iter", type=int, default=15)
    ap.add_argument("--per-game", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(8)

    models = []
    for m in a.models:
        label, path = m.split("=", 1)
        if not os.path.exists(path):
            print(f"  SKIP {label}: not found"); continue
        models.append((label, path))

    S, P, V, IT, iters = sample(a.data, a.files_per_iter, a.per_game)
    x = torch.from_numpy(S).float()
    dist = {CLASS[c]: int((V == c).sum()) for c in (0, 1, 2)}
    print(f"\nDevice={DEVICE} | window iters {iters[0]}-{iters[-1]} | {len(V):,} positions | outcomes {dist}")

    # old/new split by iteration median
    med = int(np.median(IT))
    old_m, new_m = IT <= med, IT > med
    print(f"old/new split at iter {med}: old={old_m.sum():,}  new={new_m.sum():,}\n")

    print("=== VALUE  (calibration vs game outcomes — Guo et al. 2017) ===")
    for label, path in models:
        pol, wdl = infer(load(path), x)
        value_row(label, wdl, V)
        if old_m.any() and new_m.any():
            bo, eo = brier_multiclass(wdl[old_m], V[old_m]), 100*ece_toplabel(wdl[old_m], V[old_m])
            bn, en = brier_multiclass(wdl[new_m], V[new_m]), 100*ece_toplabel(wdl[new_m], V[new_m])
            print(f"             └ OLD iters: Brier={bo:.4f} ECE={eo:4.1f}%   |   "
                  f"NEW iters: Brier={bn:.4f} ECE={en:4.1f}%  (gap = off-distribution signal)")

    print("\n=== POLICY  (raw net vs MCTS-improved teacher — Maia move-match) ===")
    for label, path in models:
        pol, _ = infer(load(path), x)
        policy_match(label, pol, P)


if __name__ == "__main__":
    main()
