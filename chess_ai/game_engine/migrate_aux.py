#!/usr/bin/env python3
"""
One-time migration: bake auxiliary labels (material / plies_left / reply) into existing self-play
.npz so iter-8 training can use them. Atomic per file (tmp → verify originals byte-match →
os.replace) and idempotent (skips files that already have the aux keys). The games are the real
backup, so it never mutates a file in place and re-checks every original array before swapping.

Run from the REPO ROOT:
    python3 chess_ai/game_engine/migrate_aux.py --data chess_ai/data/self_play            # all iters
    python3 chess_ai/game_engine/migrate_aux.py --data chess_ai/data/self_play --dry-run   # report only
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aux_labels import derive_aux_labels

AUX_KEYS = ("material", "plies_left", "reply")


def migrate_file(f, dry_run=False):
    with np.load(f, allow_pickle=False) as d:
        # NpzFile.files lists keys WITHOUT decompressing arrays — cheap skip for already-migrated
        # files (so re-running on GCP only loads+rewrites the un-migrated iter, e.g. iter-8).
        if all(k in d.files for k in AUX_KEYS):
            return "skip"
        if dry_run:
            return "would-migrate"
        data = {k: d[k] for k in d.files}   # load arrays only when actually migrating
    mat, plies, reply = derive_aux_labels(data["states"], data["policies"], data.get("values"))
    out = dict(data); out["material"] = mat; out["plies_left"] = plies; out["reply"] = reply

    tmp = f + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    # verify before swapping: every ORIGINAL array must be byte-identical, aux arrays present
    with np.load(tmp, allow_pickle=False) as c:
        for k in data:
            if not np.array_equal(c[k], data[k]):
                os.remove(tmp)
                raise RuntimeError(f"original array '{k}' changed during migrate of {f}")
        for k in AUX_KEYS:
            assert k in c.files, f"aux key {k} missing after write for {f}"
    os.replace(tmp, f)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="chess_ai/data/self_play")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "iter_*", "*.npz")))
    if not files:
        sys.exit(f"no .npz under {a.data}")
    counts = {"done": 0, "skip": 0, "would-migrate": 0}
    for i, f in enumerate(files, 1):
        counts[migrate_file(f, a.dry_run)] += 1
        if i % 500 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] done={counts['done']} skip={counts['skip']} "
                  f"would-migrate={counts['would-migrate']}", flush=True)
    print(f"\nmigrate_aux {'(dry-run) ' if a.dry_run else ''}complete: {counts}")


if __name__ == "__main__":
    main()
