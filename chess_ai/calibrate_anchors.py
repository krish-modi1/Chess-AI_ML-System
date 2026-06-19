#!/usr/bin/env python3
"""Calibrate fixed-NODES Stockfish anchors against the UCI-1320 reference point, then
write a frozen anchors.json the Elo harness uses. Fast (SF-vs-SF, no model inference).

The node-limited SF (`go nodes N`) is reproducible across machines/versions — unlike UCI_Elo
(CCRL-scaled, 120s+1s, floored at 1320). We tie the scale ONCE to UCI-1320 (the one human-
meaningful point where we measured the model), then anchors are node-defined and frozen forever.

Usage (from repo root, conda env with python-chess):
    conda run -n chessai python chess_ai/calibrate_anchors.py \
        --stockfish /usr/games/stockfish --games 30 \
        --nodes 2,8,24,64,128,256,512,1024 --targets 1200,1450,1750
"""
import argparse, json, math, os, random, sys
import chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish"))
ap.add_argument("--sf-version", default="sf_16.1", help="recorded in anchors.json for reproducibility")
ap.add_argument("--ref-elo", type=int, default=1320, help="UCI_Elo reference (our one human-meaningful tie)")
ap.add_argument("--ref-movetime", type=float, default=0.05, help="seconds/move for the UCI reference SF")
ap.add_argument("--games", type=int, default=30, help="games per node level (vs the ref)")
ap.add_argument("--nodes", default="2,8,24,64,128,256,512,1024", help="node levels to sweep")
ap.add_argument("--targets", default="1200,1450,1750", help="target Elos to pick anchors for")
ap.add_argument("--max-moves", type=int, default=120)
ap.add_argument("--out", default="chess_ai/anchors.json")
args = ap.parse_args()

SF = args.stockfish
node_levels = [int(x) for x in args.nodes.split(",")]
targets = [int(x) for x in args.targets.split(",")]

def new_engine(uci_elo=None):
    e = chess.engine.SimpleEngine.popen_uci(SF)
    if uci_elo is not None:
        e.configure({"UCI_LimitStrength": True, "UCI_Elo": uci_elo})
    return e

def play(node_eng, ref_eng, nodes, node_is_white):
    """One game: node-limited SF vs UCI-ref SF. Returns node-SF score (1/0.5/0)."""
    b = chess.Board()
    while not b.is_game_over() and b.fullmove_number < args.max_moves:
        node_turn = (b.turn == chess.WHITE) == node_is_white
        if node_turn:
            mv = node_eng.play(b, chess.engine.Limit(nodes=nodes)).move
        else:
            mv = ref_eng.play(b, chess.engine.Limit(time=args.ref_movetime)).move
        if mv is None:
            break
        b.push(mv)
    r = b.result()
    if r == "1/2-1/2" or r == "*":
        return 0.5
    return 1.0 if (r == "1-0") == node_is_white else 0.0

def score_to_elo(p, ref):
    """Logistic Elo of a player scoring p vs a `ref`-Elo opponent."""
    p = min(max(p, 0.5 / args.games), 1 - 0.5 / args.games)   # clamp off the 0/1 rails
    return ref + 400.0 * math.log10(p / (1 - p))

print(f"Stockfish: {SF}  |  ref = UCI_Elo {args.ref_elo} @ {args.ref_movetime}s/mv  |  {args.games} games/level")
print(f"{'nodes':>7} {'score':>7} {'est_Elo':>8}")
print("-" * 26)
curve = []   # (nodes, elo)
for N in node_levels:
    node_eng = new_engine()              # full-strength search, limited to N nodes
    ref_eng = new_engine(args.ref_elo)   # UCI-limited reference
    pts = 0.0
    for g in range(args.games):
        pts += play(node_eng, ref_eng, N, node_is_white=(g % 2 == 0))
    node_eng.quit(); ref_eng.quit()
    p = pts / args.games
    elo = score_to_elo(p, args.ref_elo)
    curve.append((N, elo))
    print(f"{N:>7} {p:>7.3f} {elo:>8.0f}")

# Pick the node level whose estimated Elo is closest to each target.
anchors = []
for t in targets:
    N, elo = min(curve, key=lambda ne: abs(ne[1] - t))
    anchors.append({"name": f"A{len(anchors)+1}", "nodes": N, "elo": round(elo)})

cfg = {"sf_version": args.sf_version, "ref_elo": args.ref_elo, "calibrated_with_games": args.games,
       "sweep": [{"nodes": N, "elo": round(e)} for N, e in curve], "anchors": anchors}
with open(args.out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"\nPicked anchors (closest to {targets}):")
for a in anchors:
    print(f"  {a['name']}: nodes={a['nodes']}  Elo≈{a['elo']}")
print(f"Wrote {args.out}")
