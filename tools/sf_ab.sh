#!/usr/bin/env bash
# External Stockfish A/B — the decisive test for "is the arena a self-referential artifact, or is the
# candidate genuinely weaker?". Plays champion / candidate / swa each vs the SAME Stockfish level and
# reports W-D-L. If cand ≈ champ → arena is the artifact (gate is the wall). If cand << champ → the
# candidate genuinely plays weaker (distributional overfit; the training levers are the right track).
#
# Run from repo root, AFTER stopping the training run (frees the GPU + freezes the current candidate.pth):
#     bash tools/sf_ab.sh
# Tunables (env): ELO GAMES SIMS WORKERS MOVE_TIME. --server = 1 shared GPU model + N CPU workers.
set -u
cd "$(dirname "$0")/.."

ELO="${ELO:-2300}"          # SF UCI_Elo. Want champ in the 30-70% band: drop to 2100 if champ<30%, raise to 2500 if >80%.
GAMES="${GAMES:-100}"       # per model; 100 → 95% CI ±5%, enough to resolve a ~7-10% champ↔cand gap.
SIMS="${SIMS:-800}"         # match the arena's eval sims so the strength is comparable to the 43% arena number.
WORKERS="${WORKERS:-25}"    # --server: CPU workers routed to one shared GPU model (25-30 fine on the 4090).
MOVE_TIME="${MOVE_TIME:-0.1}"
OUT="local/sf_ab"; mkdir -p "$OUT"

# CRITICAL: use the SAME from-source SF-16 the pipeline built ($HOME/.local/bin/stockfish, run_gcp.sh),
# NOT apt's /usr/games/stockfish. The apt binary's UCI_Elo is miscalibrated (crushes the champ to ~5%
# at "2300"); the built SF-16 is the one the 80%@2100 anchor was tuned on. See elo-measurement-findings.
SF_BIN="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
if [ ! -x "$SF_BIN" ]; then
  echo "⚠️  SF-16 not found at $SF_BIN — falling back to /usr/games/stockfish (apt, MISCALIBRATED — "
  echo "    champ will floor ~5% and the A/B won't resolve). Build SF-16 first (run_gcp.sh step 3)."
  SF_BIN="/usr/games/stockfish"
fi
echo "SF binary: $SF_BIN  ($("$SF_BIN" --version 2>/dev/null | head -1))"

CFG="--elo $ELO --sims $SIMS --games $GAMES --server --workers $WORKERS --move-time $MOVE_TIME --stockfish $SF_BIN"
echo "=== SF A/B config: elo=$ELO games=$GAMES sims=$SIMS workers=$WORKERS (server) ==="

run() {  # label  ckpt-path
  local label="$1" model="$2"
  [ -f "$model" ] || { echo "  SKIP $label: $model not found"; return; }
  echo; echo "########## $label  vs  SF-$ELO ##########"
  python3 chess_ai/eval_elo.py --model "$model" $CFG --out "$OUT/${label}_vs_sf.pgn" 2>&1 \
    | tee "$OUT/${label}.log" | grep -iE "W-D-L|win rate|score|elo|result:" || true
}

run champ chess_ai/game_engine/model/best_model.pth
run cand  chess_ai/game_engine/model/candidate.pth
run swa   chess_ai/game_engine/model/swa_model.pth

echo; echo "=== SUMMARY (score vs SF-$ELO; higher = stronger) ==="
for l in champ cand swa; do
  [ -f "$OUT/$l.log" ] && echo "  $l:  $(grep -iE 'W-D-L|win rate|score' "$OUT/$l.log" | tail -1)"
done
echo "PGNs + full logs in $OUT/ . Verdict: cand ≈ champ → arena artifact (fix the gate); cand << champ → real overfit (training levers)."
