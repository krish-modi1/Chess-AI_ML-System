"""Per-opening draw-rate check — buckets an iteration's self-play games by their opening
signature (the first RECORDED position, i.e. the post-book line, since book plies aren't recorded)
and reports the draw rate of each, so you can spot openings that constantly draw.

Draw detection is exact from the stored targets: a decisive game stores value 0/2 (win/loss, STM POV)
for every position; a draw stores 1 for every position (main.py:405). So a game is a draw iff its
`values` array is all 1s.

Pure numpy/hashlib — no torch/chess deps.

Usage (from repo root):
    python3 chess_ai/check_draws.py             # latest numeric iter
    python3 chess_ai/check_draws.py 27           # iter_27
    python3 chess_ai/check_draws.py 20-27        # pool iters 20..27 (more games/opening → less noise)
    python3 chess_ai/check_draws.py iter_900     # a named bank dir
    python3 chess_ai/check_draws.py 27 12        # min games per opening to list (default 5)
"""
import sys, glob, os, hashlib, re
from collections import defaultdict
import numpy as np

DATA = "chess_ai/data/self_play"


def pick_dirs(arg):
    if arg and re.fullmatch(r"\d+-\d+", arg):                 # range: pool several iters
        lo, hi = (int(x) for x in arg.split("-"))
        return [f"{DATA}/iter_{i}" for i in range(lo, hi + 1) if os.path.isdir(f"{DATA}/iter_{i}")]
    if arg and arg.isdigit():
        return [f"{DATA}/iter_{arg}"]
    if arg:
        return [arg if os.path.isdir(arg) else f"{DATA}/{arg}"]
    dirs = [p for p in glob.glob(f"{DATA}/iter_*")
            if os.path.isdir(p) and os.path.basename(p).split("_")[-1].isdigit()]
    if not dirs:
        sys.exit(f"no numeric iter dirs under {DATA}")
    return [max(dirs, key=lambda p: int(os.path.basename(p).split("_")[-1]))]


def H(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).digest()[:10]


arg = sys.argv[1] if len(sys.argv) > 1 else None
min_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5
dirs = pick_dirs(arg)
files = sorted(f for dd in dirs for f in glob.glob(f"{dd}/*.npz"))
if not files:
    sys.exit(f"no .npz in {dirs}")
label = arg if (arg and re.fullmatch(r"\d+-\d+", arg)) else os.path.basename(dirs[0])

# per opening: [games, draws, sum_len]
op = defaultdict(lambda: [0, 0, 0])
tot_games = tot_draws = 0
for f in files:
    z = np.load(f)
    s, v = z["states"], z["values"]
    if len(s) == 0:
        continue
    is_draw = bool((v == 1).all())
    sig = H(s[0])
    op[sig][0] += 1
    op[sig][1] += int(is_draw)
    op[sig][2] += len(s)
    tot_games += 1
    tot_draws += int(is_draw)

print(f"=== per-opening draw rate: {label} ({len(dirs)} dir{'s' if len(dirs)!=1 else ''}) ===")
print(f"games {tot_games} | overall draws {tot_draws} ({100*tot_draws/tot_games:.1f}%) | "
      f"distinct openings {len(op)}")
print(f"\nopenings with >= {min_games} games, sorted by draw rate (high = constant-draw line):")
print(f"  {'opening':<22} {'games':>6} {'draws':>6} {'draw%':>7} {'avg_len':>8}")
rows = [(sig, g, dr, ln) for sig, (g, dr, ln) in op.items() if g >= min_games]
for sig, g, dr, ln in sorted(rows, key=lambda r: r[2] / r[1], reverse=True):
    print(f"  {sig.hex():<22} {g:>6} {dr:>6} {100*dr/g:>6.1f}% {ln/g:>8.1f}")

few = sorted([(sig, g, dr) for sig, (g, dr, _) in op.items() if g < min_games],
             key=lambda r: r[1], reverse=True)
if few:
    thin = sum(g for _, g, _ in few)
    print(f"\n(+ {len(few)} openings with < {min_games} games, {thin} games total — too thin to rank)")
