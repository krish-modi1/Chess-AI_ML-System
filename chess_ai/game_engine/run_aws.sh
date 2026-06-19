#!/usr/bin/env bash
# run_aws.sh — AWS launcher (g6 L4-24GB / g4dn T4-16GB). Wraps run_gcp.sh and injects the AWS/L4
# config via EXTRA_ENV (sourced AFTER hyperparams so it wins), plus a raised FD limit.
# cu126 torch runs as-is on L4/T4 (no Blackwell). Run from the repo root:
#   bash chess_ai/game_engine/run_aws.sh --background
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- AWS / L4 config (sourced after hyperparams via the EXTRA_ENV hook in run_gcp.sh) ---
OVR="$HERE/.aws_overrides.env.sh"
cat > "$OVR" <<'ENV'
# Self-play: 60 workers × 10 games = 600 games/iter (hyperparams defaults; 60×8 → CUDA_BATCH 480,
#   fits L4 24GB and T4 16GB). NUM_WORKERS / GAMES_PER_WORKER left at the hyperparams defaults.
export TRAIN_BATCH_SIZE=2048         # ~21GB on the L4 24GB (also fits T4 16GB)

# Eval: 40 workers × 4 games = 160 games for BOTH arena and Stockfish, routed to a COMMON GPU
#   inference server (server-mode — candidate+champion on 2 servers for arena, 1 for Stockfish;
#   the 40 CPU workers share them via SHM). 4 games/worker keeps color balance (2 White + 2 Black).
export EVAL_WORKERS=40
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_WORKERS=40
export STOCKFISH_GAMES=160

# A_low Elo anchor — fixed nodes → reproducible, same scale as the round-robin ladder.
export STOCKFISH_ELO=1320
export STOCKFISH_NODES=100000
ENV
export EXTRA_ENV="$OVR"

# FD limit — 60 self-play / 40 eval workers × shared servers exhaust the default (~1024) and FDs
# accumulate across iterations (the iter-21 "[Errno 24] Too many open files" crash). Raise to max.
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[aws] FD limit=$(ulimit -n)  |  config → $OVR"

# Hand off to the GCP launcher (privilege/apt/stockfish/build/VRAM-detect/launch all reused).
exec bash "$HERE/run_gcp.sh" "$@"
