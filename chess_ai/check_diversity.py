"""Self-play game-diversity check — reads an iteration's recorded positions (states) and reports
opening diversity + overall position repetition, so you can tell if self-play is collapsing
(low diversity → over-fit/saturated targets) or staying varied.

Pure numpy/hashlib — no torch/chess deps, runs anywhere fast.

Usage (from repo root):
    python3 chess_ai/check_diversity.py            # latest numeric iter
    python3 chess_ai/check_diversity.py 24          # iter_24
    python3 chess_ai/check_diversity.py iter_aws    # a named bank dir
"""
import sys, glob, os, hashlib, math
from collections import Counter
import numpy as np

DATA = "chess_ai/data/self_play"


def pick_dir(arg):
    if arg and arg.isdigit():
        return f"{DATA}/iter_{arg}"
    if arg:
        return arg if os.path.isdir(arg) else f"{DATA}/{arg}"
    dirs = [p for p in glob.glob(f"{DATA}/iter_*")
            if os.path.isdir(p) and os.path.basename(p).split("_")[-1].isdigit()]
    if not dirs:
        sys.exit(f"no numeric iter dirs under {DATA}")
    return max(dirs, key=lambda p: int(os.path.basename(p).split("_")[-1]))


def H(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).digest()[:10]


d = pick_dir(sys.argv[1] if len(sys.argv) > 1 else None)
files = sorted(glob.glob(f"{d}/*.npz"))
if not files:
    sys.exit(f"no .npz in {d}")

openings, all_pos, total, lens = Counter(), set(), 0, []
for f in files:
    s = np.load(f)["states"]
    if len(s) == 0:
        continue
    openings[H(s[0])] += 1          # first RECORDED (full-search) position ≈ opening/early signature
    for st in s:
        all_pos.add(H(st))
        total += 1
    lens.append(len(s))

ng = sum(openings.values())
ent = -sum((c / ng) * math.log2(c / ng) for c in openings.values())
max_ent = math.log2(ng) if ng > 1 else 1.0

print(f"=== self-play diversity: {os.path.basename(d)} ===")
print(f"games {ng} | recorded positions {total} | unique positions {len(all_pos)} "
      f"({100 * len(all_pos) / total:.1f}%  — low % = repetitive lines)")
print(f"distinct opening signatures: {len(openings)} of {ng} games")
print(f"opening entropy: {ent:.2f} / {max_ent:.2f} bits  ({100 * ent / max_ent:.0f}% of max; "
      f"100% = every game a distinct opening)")
print("top opening clusters (share of games):")
for sig, c in openings.most_common(8):
    print(f"   {sig.hex()}: {c:>4} games  {100 * c / ng:5.1f}%")
print(f"game length (recorded plies): mean {np.mean(lens):.1f}  median {int(np.median(lens))}  "
      f"range [{min(lens)}, {max(lens)}]")
