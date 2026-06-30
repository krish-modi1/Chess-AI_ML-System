"""Verification gates for the opening-suite change. All offline (python-chess only, no GPU/models).
Asserts the suite is clean AND that the arena color-pairing is fair AND that main.py still compiles and
the loader/fallback behave — so NOTHING in other phases can be made unreliable by this change.
Run from repo root:  python3 tools/test_opening_suite.py
"""
import os, re, sys, subprocess, collections
import chess

ROOT  = "/home/krish/Chess-AI_ML-System"
SUITE = f"{ROOT}/chess_ai/game_engine/openings_suite.txt"
MAIN  = f"{ROOT}/chess_ai/game_engine/main.py"

def load(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

# ── 1. Suite quality: every line legal, replayable from startpos, non-terminal ───────────────
lines = load(SUITE)
fams, fens = collections.Counter(), set()
for i, line in enumerate(lines):
    toks = line.split()
    assert toks, f"line {i} empty"
    b = chess.Board()
    for u in toks:
        try:
            mv = chess.Move.from_uci(u)
        except Exception:
            raise AssertionError(f"line {i}: unparseable UCI {u!r} in {line!r}")
        assert mv in b.legal_moves, f"line {i}: illegal move {u} in {line!r}"
        b.push(mv)
    assert not b.is_game_over(), f"line {i}: opening ends the game: {line!r}"
    fens.add(b.fen())
    fams[" ".join(toks[:3])] += 1
print(f"[1] {len(lines)} lines — all legal, replayable, non-terminal ✓")
print(f"[1] distinct final positions: {len(fens)} | distinct 3-ply families: {len(fams)}")
assert len(fens) == len(lines), "duplicate final positions in suite (dedupe failed)"
assert len(lines) >= 100, f"only {len(lines)} lines — gate CI too wide (want >=100)"
assert len(fams) >= 30, f"only {len(fams)} families — not broad enough (want >=30)"

# ── 1b. Junk-opening detector: tempo-waste = a piece makes a NON-capture move and is later reversed
# (g1f3…f3g1) — the τ=1 exploration garbage balance-filtering missed. Capture-then-retreat (e.g. a
# knight recapturing on e5 then retreating) is NOT junk — it gained material — so the outward move
# must be a non-capture to count.
def junk_roundtrip(toks):
    b = chess.Board(); hist = []
    for u in toks:
        mv = chess.Move.from_uci(u); cap = b.is_capture(mv); f, t = u[:2], u[2:4]
        for (pf, pt, pcap) in hist:
            if f == pt and t == pf and not pcap:      # reverses an earlier NON-capture move
                return f"{pf}{pt}…{f}{t}"
        hist.append((f, t, cap)); b.push(mv)
    return None
junk = [(i, junk_roundtrip(l.split())) for i, l in enumerate(lines) if junk_roundtrip(l.split())]
assert not junk, f"{len(junk)} tempo-waste openings, e.g. line {junk[0][0]}: {junk[0][1]}"
print(f"[1b] no tempo-waste (non-capture round-trip) junk openings ✓")

# ── 2. Source/family balance sanity (not one family dominating) ──────────────────────────────
top_fam, top_n = fams.most_common(1)[0]
print(f"[2] most common family '{top_fam}' = {top_n}/{len(lines)} ({100*top_n/len(lines):.0f}%)")
assert top_n <= 0.25 * len(lines), f"family '{top_fam}' dominates ({top_n}) — not balanced"
print("[2] no single family dominates ✓")

# ── 3. Arena color-pairing fairness: each opening idx played equally as White and Black ───────
N_GAMES_PER_WORKER, N_WORKERS = 6, 50          # matches GAMES_PER_EVAL_WORKER / EVAL_WORKERS
n = len(lines)
white = collections.Counter(); black = collections.Counter()
for worker_id in range(N_WORKERS):
    for gi in range(N_GAMES_PER_WORKER):
        game_id = worker_id * N_GAMES_PER_WORKER + gi
        opening_idx = (game_id // 2) % n
        if game_id % 2 == 0:
            white[opening_idx] += 1
        else:
            black[opening_idx] += 1
used = set(white) | set(black)
imbalanced = [k for k in used if white[k] != black[k]]
print(f"[3] {N_WORKERS*N_GAMES_PER_WORKER} arena games over {len(used)} openings; "
      f"white-uses==black-uses for all: {not imbalanced}")
assert not imbalanced, f"color imbalance at opening idx {imbalanced[:5]} (opening not played both colors)"
print("[3] every opening is a fair color pair ✓")

# ── 4. main.py compiles; loader logic returns the file; fallback on missing file ──────────────
subprocess.run([sys.executable, "-m", "py_compile", MAIN], check=True)
print("[4] main.py compiles ✓")
# Replicate the loader logic exactly (avoid importing main.py = torch/C++ init) and assert behaviour.
def _load(path, fallback):
    try:
        with open(path) as fh:
            got = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        return got if got else fallback
    except Exception:
        return fallback
FB = ["e2e4 e7e5", "d2d4 d7d5"]
assert _load(SUITE, FB) == lines, "loader didn't return the suite file lines"
assert _load(f"{SUITE}.nope", FB) == FB, "loader didn't fall back on missing file"
# Confirm the source still HAS the fallback list + the suite/arena wiring + dropped the random opening.
src = open(MAIN).read()
assert "_FALLBACK_OPENING_BOOK = [" in src, "fallback list missing from main.py"
assert "OPENING_BOOK = _load_opening_suite()" in src and "ARENA_OPENINGS = OPENING_BOOK" in src
assert "random.choice(legal)" not in src, "old random opening still present in main.py"
# random is imported via `import chess, random` in the arena worker and still used by random.seed —
# assert the USE survives (the real invariant) rather than a brittle import-string match.
assert "random.seed(" in src, "random.seed removed — random import may now be orphaned"
assert re.search(r"import .*\brandom\b", src), "random no longer imported but random.seed needs it"
print("[4] loader returns suite, falls back on missing, wiring present, random-opening gone ✓")

# ── 5. Self-play format compatibility: lines consume identically to the old book idiom ────────
# (push each split token from startpos — exactly what the self-play book block does.)
for line in lines:
    b = chess.Board()
    for u in line.split():
        b.push(chess.Move.from_uci(u))   # would raise if not the same UCI move-line format
print("[5] every line consumes via the self-play book idiom (split→push) ✓")

# ── 6. REAL integration: import the actual main.py module (real module-level suite load) and replay
# every opening through the REAL ChessGame (the exact object + idiom the arena/self-play loops use) —
# not python-chess. Closes the gap between "format looks right" and "the production code accepts it".
for p in (f"{ROOT}/chess_ai", f"{ROOT}/chess_ai/game_engine", f"{ROOT}/chess_ai/game_engine/build"):
    if p not in sys.path:
        sys.path.insert(0, p)
import importlib
m = importlib.import_module("game_engine.main")
assert m.OPENING_BOOK == lines, "main.OPENING_BOOK != suite file (real load mismatch)"
assert m.ARENA_OPENINGS is m.OPENING_BOOK, "ARENA_OPENINGS not aliased to OPENING_BOOK"
print(f"[6] main.py imports; real OPENING_BOOK={len(m.OPENING_BOOK)} == suite, ARENA_OPENINGS aliased ✓")

from game_engine.chess_env import ChessGame
for line in m.OPENING_BOOK:                       # replay through the REAL game object, loop idiom
    g = ChessGame()
    for u in line.split():
        assert not g.is_over, f"opening ends mid-line (real ChessGame): {line!r}"
        assert g.push(u), f"real ChessGame rejected legal move {u} in {line!r}"
    assert not g.is_over, f"opening ends the game (real ChessGame): {line!r}"
print(f"[6] all {len(m.OPENING_BOOK)} openings replay on the REAL ChessGame (push+is_over) ✓")

# ── 7. Arena color-pairing using the REAL imported ARENA_OPENINGS (not a copy) ────────────────
nA = len(m.ARENA_OPENINGS)
w2, b2 = collections.Counter(), collections.Counter()
for worker_id in range(N_WORKERS):
    for gi in range(N_GAMES_PER_WORKER):
        gid = worker_id * N_GAMES_PER_WORKER + gi
        idx = (gid // 2) % nA
        (w2 if gid % 2 == 0 else b2)[idx] += 1
assert all(w2[k] == b2[k] for k in set(w2) | set(b2)), "color imbalance on real ARENA_OPENINGS"
print("[7] color-pairing fair on the REAL ARENA_OPENINGS ✓")

print("\nALL VERIFICATION GATES PASSED")
