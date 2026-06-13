#!/usr/bin/env python3
"""
Upgrade pre-aux checkpoints to the new architecture (adds the aux heads with random weights) so
iter-8 training/arena load them with no missing-key / optimizer-mismatch error. Saves MODEL WEIGHTS
ONLY — stale optimizer/scheduler state (built for the old, smaller param set) is dropped, which is
correct across an arch change. Idempotent: re-running on an already-upgraded ckpt is a no-op-ish
(it just re-saves with all keys present).

Run from the REPO ROOT:
    python3 chess_ai/game_engine/upgrade_ckpt.py \
        chess_ai/game_engine/model/best_model.pth \
        chess_ai/game_engine/model/champ_iter8.pth
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ChessCNN


def upgrade(path):
    if not os.path.exists(path):
        print(f"  SKIP (missing): {path}")
        return
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else (
         raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    model = ChessCNN()                                   # new arch (random aux heads)
    res = model.load_state_dict(sd, strict=False)        # trunk+policy+value load; aux stay random
    non_aux_missing = [k for k in res.missing_keys
                       if not any(k.startswith(h) for h in ("material_head", "plies_head", "reply_head"))]
    if non_aux_missing or res.unexpected_keys:
        raise RuntimeError(f"{path}: unexpected non-aux mismatch "
                           f"(missing={non_aux_missing} unexpected={res.unexpected_keys})")

    tmp = path + ".tmp"
    torch.save({"model_state_dict": model.state_dict()}, tmp)   # weights only; no optimizer state
    os.replace(tmp, path)
    print(f"  upgraded: {path}  (+{len(res.missing_keys)} aux params, optimizer state dropped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="checkpoint .pth files to upgrade in place")
    a = ap.parse_args()
    for p in a.paths:
        upgrade(p)
    print("upgrade_ckpt complete.")


if __name__ == "__main__":
    main()
