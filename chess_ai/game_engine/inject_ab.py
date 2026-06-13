#!/usr/bin/env python3
"""
Offline A/B for INJECTING strong decisive games (e.g. Lichess 2500+, or Stockfish) into the
self-play training window. Tests the exact shipping mechanism — drop the injected .npz into the
data window and let train_model see them — so the result transfers 1:1 to production.

It builds a temp data dir that symlinks the last `window` real self-play iters and adds the
injected games as one extra iter folder (replicated to hit --inject-frac), trains a candidate
from best_model, then probes BOTH heads vs the champion:

    python3 game_engine/inject_ab.py --inject-dir data/injected --inject-frac 0.20
    python3 game_engine/inject_ab.py --inject-dir data/injected --inject-frac 0.40 --kl-beta 0.0

Read the probe (decision):
  value head SHARPER (mean|v| / |v|>=0.9 up, val_acc holds)  → injection sharpens value (the bottleneck)
  policy_top1 (vs self-play MCTS target) HOLDS               → self-play policy not degraded
  drift vs champ (KL up, agree down)                          → policy MOVED; check if toward strong play
If value sharpens AND self-play policy_top1 holds → ship it. If policy_top1 collapses → the human
moves are fighting self-play; switch to value-only injection (mask policy loss) before shipping.

Injected .npz must match self-play format exactly:
  states  (N,120,8,8) f32   policies (N,4672) f32 (one-hot on the played move)   values (N,) int8 {0,1,2}
(pretrain_lichess.ipynb already has the 120-plane encoder + 4672 action map — reuse it in Colab.)
"""
import os
import sys
import glob
import shutil
import tempfile
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--inject-dir", required=True, help="folder of injected .npz (one game/file), Colab output")
ap.add_argument("--inject-frac", type=float, default=0.20, help="target fraction of training positions from injected games")
ap.add_argument("--base",   default="game_engine/model/best_model.pth")
ap.add_argument("--data",   default="data/self_play")
ap.add_argument("--out",    default="/tmp/inject_candidate.pth")
ap.add_argument("--window", type=int,   default=int(os.environ.get("TRAIN_WINDOW", 20)))
ap.add_argument("--cap",    type=int,   default=int(os.environ.get("MAX_POSITIONS_PER_GAME", 20)))
ap.add_argument("--kl-beta", type=float, default=float(os.environ.get("KL_ANCHOR_BETA", 1.0)))
ap.add_argument("--epochs", type=int,   default=int(os.environ.get("TRAIN_EPOCHS", 2)))
ap.add_argument("--lr",     type=float, default=float(os.environ.get("TRAIN_LR", 1e-4)))
ap.add_argument("--probe-games", type=int, default=40)
a = ap.parse_args()

# Set knobs BEFORE importing trainer (module-level constants read os.environ at import).
os.environ["MAX_POSITIONS_PER_GAME"] = str(a.cap)
os.environ["KL_ANCHOR_BETA"]         = str(a.kl_beta)


def npz_n(f):
    with np.load(f, allow_pickle=True) as d:
        return int(d["states"].shape[0])


def build_temp_dir(real_data, inject_dir, window, frac, cap):
    """Symlink the last `window` real iters + add injected games as iter_<max+1>, replicated to
    hit `frac`. Returns (tmpdir, realized_inject_pos, window_pos)."""
    subdirs = sorted(
        [d for d in os.listdir(real_data)
         if d.startswith("iter_") and d.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1]))
    win = subdirs[-window:]
    if not win:
        sys.exit(f"no iter_* folders in {real_data}")
    maxit = max(int(d.split("_")[1]) for d in win)

    tmpd = tempfile.mkdtemp(prefix="inject_ab_")
    for d in win:
        os.symlink(os.path.abspath(os.path.join(real_data, d)), os.path.join(tmpd, d))

    # window positions AFTER cap (so the fraction target is honest about what training sees)
    def capped(n):
        return min(n, cap) if cap > 0 else n
    window_pos = sum(capped(npz_n(os.path.join(real_data, d, f)))
                     for d in win for f in os.listdir(os.path.join(real_data, d)) if f.endswith(".npz"))

    pool = sorted(glob.glob(os.path.join(inject_dir, "*.npz")))
    if not pool:
        sys.exit(f"no .npz in {inject_dir}")
    pool_pos = sum(capped(npz_n(f)) for f in pool)

    # target injected positions so injected/(injected+window) == frac
    target = frac / (1.0 - frac) * window_pos
    copies = max(1, int(np.ceil(target / pool_pos)))

    injdir = os.path.join(tmpd, f"iter_{maxit + 1}")
    os.makedirs(injdir)
    realized = 0
    for c in range(copies):
        for f in pool:
            os.symlink(os.path.abspath(f), os.path.join(injdir, f"inj{c}_{os.path.basename(f)}"))
            realized += capped(npz_n(f))
    return tmpd, realized, window_pos


tmpd, inj_pos, win_pos = build_temp_dir(a.data, a.inject_dir, a.window, a.inject_frac, a.cap)
realized_frac = inj_pos / max(inj_pos + win_pos, 1)
print(f"\n=== INJECT A/B  frac(target={a.inject_frac:.2f}, realized={realized_frac:.2f})  "
      f"cap={a.cap}  kl_beta={a.kl_beta}  epochs={a.epochs}  lr={a.lr} ===")
print(f"    window={win_pos:,} pos (capped)  +  injected={inj_pos:,} pos  → train on {win_pos + inj_pos:,}")

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN
from trainer import train_model

try:
    train_model(
        data_path=tmpd,
        input_model_path=a.base,
        output_model_path=a.out,
        epochs=a.epochs,
        batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", 2048)),
        lr=a.lr,
        window_size=a.window + 1,                 # +1 so the injected iter folder is inside the window
        total_iterations=1000,
    )
finally:
    shutil.rmtree(tmpd, ignore_errors=True)


def load(p):
    raw = torch.load(p, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval(); m.load_state_dict(sd, strict=False); return m  # tolerate pre-aux ckpts


def latest_iter(d):
    return max(int(os.path.basename(x).split("_")[1])
               for x in glob.glob(os.path.join(d, "iter_*")) if x.split("_")[-1].isdigit())


# --- probe on REAL self-play positions (the distribution we ultimately care about) ---
it = latest_iter(a.data)
S, V, P = [], [], []
for f in sorted(glob.glob(os.path.join(a.data, f"iter_{it}", "*.npz")))[:a.probe_games]:
    d = np.load(f); n = d["states"].shape[0]; idx = np.linspace(0, n - 1, min(n, 15)).astype(int)
    S.append(d["states"][idx]); V.append(d["values"][idx]); P.append(d["policies"][idx])
S = np.concatenate(S); V = np.concatenate(V).astype(int); P = np.concatenate(P)
x = torch.from_numpy(S).float()
legal = P > 0
tgt_top1 = P.argmax(1)


def probe(model, name, ref_pol=None):
    with torch.no_grad():
        pol, val = model(x)
    pol = pol.numpy()
    w = torch.softmax(val, -1).numpy()
    av = np.abs(w[:, 0] - w[:, 2])
    acc = 100 * (w.argmax(1) == V).mean()
    masked = np.where(legal, pol, -1e9)
    ptop1 = 100 * (masked.argmax(1) == tgt_top1).mean()
    drift = ""
    if ref_pol is not None:
        agree = kl = 0.0; valid = 0
        for i in range(len(P)):
            leg = legal[i]
            if leg.sum() < 2:
                continue
            A = np.clip(_norm(ref_pol[i][leg]), 1e-9, 1); B = np.clip(_norm(pol[i][leg]), 1e-9, 1)
            agree += (A.argmax() == B.argmax()); kl += float((A * np.log(A / B)).sum()); valid += 1
        valid = max(valid, 1)
        drift = f"  | drift vs champ: agree={100*agree/valid:4.0f}%  KL={kl/valid:.3f}"
    print(f"  {name:18s} mean|v|={av.mean():.3f}  |v|>=0.9:{100*(av>=0.9).mean():4.0f}%  "
          f"val_acc={acc:3.0f}%  policy_top1(vs MCTS)={ptop1:3.0f}%{drift}")
    return pol


def _norm(a):
    a = np.asarray(a, dtype=np.float64); s = a.sum(); return a / s if s > 0 else a


print(f"\n-- probe on iter_{it} self-play ({len(V)} positions) --")
champ_pol = probe(load(a.base), "base (champ)")
probe(load(a.out), f"inject f={realized_frac:.2f}", ref_pol=champ_pol)
