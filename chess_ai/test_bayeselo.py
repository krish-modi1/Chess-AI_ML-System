#!/usr/bin/env python3
"""BayesElo smoke test — run the binary on a real (git-tracked) PGN and dump EVERYTHING: binary
presence/version, raw stdout, raw stderr, return code, and the parser result. Run on BOTH the local
machine (where it works) and the box (where it 'Failed to parse') and diff the output — that pinpoints
the real issue (empty stdout / nonzero rc = binary incompatible & needs rebuild; table present but
parser None = output-format mismatch).

Usage:  python3 local/test_bayeselo.py [path/to.pgn] [stockfish_elo]
Exit:   0 = BayesElo parsed an Elo,  1 = failed (see dumped raw output).
"""
import os, sys, glob, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
BIN  = os.path.join(ROOT, "chess_ai", "BayesElo", "bayeselo")
SF_ELO = int(sys.argv[2]) if len(sys.argv) > 2 else 1800

pgn = sys.argv[1] if len(sys.argv) > 1 else None
if not pgn:
    cands = sorted(glob.glob(os.path.join(ROOT, "chess_ai/game_engine/evaluation/pgn/*.pgn")))
    pgn = cands[-1] if cands else None

print("=" * 60)
print(f"repo root : {ROOT}")
print(f"bayeselo  : {BIN}")
print(f"  exists={os.path.exists(BIN)}  executable={os.access(BIN, os.X_OK)}  "
      f"size={os.path.getsize(BIN) if os.path.exists(BIN) else 0}")
if os.path.exists(BIN):
    try: os.chmod(BIN, 0o755)
    except Exception: pass
    # `file` tells us the arch the binary was built for (mismatch = won't run on the box)
    try:
        print("  file:", subprocess.run(["file", BIN], capture_output=True, text=True).stdout.strip())
    except Exception: pass
print(f"PGN       : {pgn}")
if not pgn or not os.path.exists(pgn):
    sys.exit("❌ no PGN found — pass one as the first argument (e.g. a git-tracked iter_*.pgn)")
print(f"  games in PGN: {open(pgn).read().count('[Result')}")
print("=" * 60)

cmds = f"readpgn {os.path.abspath(pgn)}\nelo\nmm\ncovariance\nratings\nx\nx\n"
print("\n>>> running bayeselo (readpgn; elo; mm; covariance; ratings; x; x)\n")
try:
    p = subprocess.run([BIN], input=cmds, capture_output=True, text=True, timeout=120, cwd=ROOT)
except Exception as e:
    sys.exit(f"❌ could NOT execute the binary: {e!r}\n"
             f"   → the committed binary doesn't run on this machine — it needs to be built here.")

print(f"return code: {p.returncode}")
print(f"\n--- STDOUT ({len(p.stdout)} chars) ---\n{p.stdout if p.stdout else '(EMPTY — binary produced no output)'}")
print(f"\n--- STDERR ({len(p.stderr)} chars) ---\n{p.stderr if p.stderr else '(empty)'}")

# Now the REAL parser used by the pipeline.
sys.path.insert(0, os.path.join(ROOT, "chess_ai"))
from game_engine.bayeselo_runner import BayesEloRunner
print("\n" + "=" * 60)
res = BayesEloRunner(stockfish_elo=SF_ELO).run(pgn)
if res and res.get("model_elo") is not None:
    print(f"✅ PASS — BayesElo parsed: model_elo = {res['model_elo']:.0f}  (anchor SF {SF_ELO})")
    sys.exit(0)
print("❌ FAIL — parser returned no Elo. Diagnosis from above:")
print("   • STDOUT empty / nonzero rc  → binary incompatible with this machine → BUILD BayesElo here.")
print("   • STDOUT has a ratings table → output-format mismatch → fix the parser regex.")
sys.exit(1)
