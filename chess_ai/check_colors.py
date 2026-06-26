"""White/Black/Draw outcome distribution per iteration of self-play.

Winner color is recovered from the stored targets (no PGN needed):
  - draw  iff the game's `values` array is all 1s (main.py:405).
  - else the side-to-move at position 0 is winner (value 0) or loser (value 2); side-to-move is
    white iff plane 117 is empty (chess_board.cpp: plane 117 all-1 = black to move). So
    winner_is_white = (values[0]==0) == white_to_move.

A persistent strong skew (e.g. White winning the large majority of decisive games) flags either a
colour-imbalanced opening book or the net playing one colour materially better. Pure numpy.

Usage (from repo root):
    python3 chess_ai/check_colors.py           # every numeric iter, one row each + total
    python3 chess_ai/check_colors.py 20-27      # only iters 20..27
    python3 chess_ai/check_colors.py 27          # single iter
"""
import sys, glob, os, re
import numpy as np

DATA = "chess_ai/data/self_play"


def iter_dirs(arg):
    if arg and re.fullmatch(r"\d+-\d+", arg):
        lo, hi = (int(x) for x in arg.split("-"))
        rng = range(lo, hi + 1)
    elif arg and arg.isdigit():
        rng = [int(arg)]
    else:
        rng = None
    found = [(int(os.path.basename(p).split("_")[-1]), p)
             for p in glob.glob(f"{DATA}/iter_*")
             if os.path.isdir(p) and os.path.basename(p).split("_")[-1].isdigit()]
    found.sort()
    if rng is not None:
        found = [(i, p) for i, p in found if i in rng]
    return found


def tally(d):
    w = b = dr = 0
    for f in sorted(glob.glob(f"{d}/*.npz")):
        try:
            z = np.load(f)
            s, v = z["states"], z["values"]
        except Exception:
            continue
        if len(v) == 0:
            continue
        if bool((v == 1).all()):
            dr += 1
            continue
        white_to_move = s[0, 117].mean() < 0.5
        if (int(v[0]) == 0) == white_to_move:
            w += 1
        else:
            b += 1
    return w, b, dr


def row(label, w, b, dr):
    g = w + b + dr
    if g == 0:
        return f"  {label:<8} {'(no games)':>10}"
    dec = w + b
    ws = f"{100*w/dec:.1f}%" if dec else "—"
    return (f"  {label:<8} {g:>6}  W {w:>5} ({100*w/g:4.1f}%)  B {b:>5} ({100*b/g:4.1f}%)  "
            f"D {dr:>5} ({100*dr/g:4.1f}%)  | decisive W-share {ws:>6}")


dirs = iter_dirs(sys.argv[1] if len(sys.argv) > 1 else None)
if not dirs:
    sys.exit("no matching numeric iter dirs")

print("=== white/black/draw distribution ===")
print("  (W-share = White wins / decisive games; 50% = balanced, >50% = White edge)")
tw = tb = td = 0
for i, d in dirs:
    w, b, dr = tally(d)
    tw += w; tb += b; td += dr
    print(row(f"iter_{i}", w, b, dr))
print("  " + "-" * 78)
print(row("TOTAL", tw, tb, td))
