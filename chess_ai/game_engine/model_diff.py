#!/usr/bin/env python3
"""
Compare two models' RAW policies on real positions — to test whether the current champion's
policy was 'hit hard' vs the pretrained base, or stayed intact (anchor working).

Run from chess_ai/ (needs the pretrained downloaded next to best_model):
    python3 game_engine/model_diff.py --a pretrained.pth --b game_engine/model/best_model.pth

  top1-agree  how often A's best legal move == B's best legal move (HIGH ⇒ policies ≈ same)
  KL(A‖B)     divergence of the two policies over legal moves       (LOW  ⇒ ≈ same)
If pretrained vs champion is high-agree / low-KL → the policy is INTACT (reset recovers nothing).
"""
import os, sys, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN


def load(p):
    raw = torch.load(p, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = ChessCNN().eval(); m.load_state_dict(sd, strict=False); return m  # tolerate pre-aux ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="model A (e.g. pretrained.pth)")
    ap.add_argument("--b", default="game_engine/model/best_model.pth", help="model B (champion)")
    ap.add_argument("--data", default="data/self_play")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--games", type=int, default=120)
    a = ap.parse_args()
    torch.set_num_threads(8)

    it = a.iter if a.iter is not None else max(
        int(os.path.basename(d).split("_")[1]) for d in glob.glob(os.path.join(a.data, "iter_*")))
    S, P = [], []
    for f in sorted(glob.glob(os.path.join(a.data, f"iter_{it}", "*.npz")))[:a.games]:
        d = np.load(f); n = d["states"].shape[0]; idx = np.linspace(0, n-1, min(n, 15)).astype(int)
        S.append(d["states"][idx]); P.append(d["policies"][idx])
    S = np.concatenate(S); P = np.concatenate(P)
    x = torch.from_numpy(S).float()

    def pol(path):
        with torch.no_grad():
            lg, _ = load(path)(x)
        return torch.softmax(lg, 1).numpy()
    pa, pb = pol(a.a), pol(a.b)

    agree = kl = vp = 0.0; valid = 0
    for i in range(len(P)):
        leg = P[i] > 0
        if leg.sum() < 2:
            continue
        A = pa[i][leg]; A /= A.sum(); A = np.clip(A, 1e-9, 1)
        B = pb[i][leg]; B /= B.sum(); B = np.clip(B, 1e-9, 1)
        agree += (np.argmax(A) == np.argmax(B))
        kl    += float((A * np.log(A / B)).sum())
        valid += 1
    N = max(valid, 1)
    print(f"\nmodel diff — A={os.path.basename(a.a)}  B={os.path.basename(a.b)}  iter_{it} ({valid} positions)")
    print(f"  top1-agree (A best == B best): {100*agree/N:5.1f}%   (HIGH ⇒ policies ≈ identical → INTACT)")
    print(f"  KL(A‖B)                      : {kl/N:6.3f} nats   (LOW  ⇒ ≈ same; HIGH ⇒ B drifted from A)")
    print("\n  read: high agree + low KL → champion policy ≈ pretrained → reset recovers nothing")
    print("        low agree / high KL  → policy DID drift → your reset hypothesis has legs")


if __name__ == "__main__":
    main()
