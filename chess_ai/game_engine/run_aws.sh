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
# Self-play: 120 workers × 10 games = 1200 games/iter (GAMES_PER_WORKER=10 from hyperparams).
#   Pinned to cores 0-59 (RESERVED_CORES=1 → inference server on core 60 of the 61-vCPU box; 2× oversub).
#   CUDA_BATCH recomputed to 120×8=960 since hyperparams sized it for its default 60 workers.
export NUM_WORKERS=120
export RESERVED_CORES=1
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Training: batch 2048 (fits 24GB), 40 dataloader workers × prefetch 2.
export TRAIN_BATCH_SIZE=2048
export TRAIN_DL_WORKERS=40
export TRAIN_DL_PREFETCH=2

# Eval: 100 workers × 2 games = 200 games for BOTH arena and Stockfish, shared GPU inference
#   servers (server-mode/SHM). 2 games/worker keeps color balance (1 White + 1 Black).
export EVAL_WORKERS=100
export GAMES_PER_EVAL_WORKER=2
export STOCKFISH_WORKERS=100
export STOCKFISH_GAMES=200

# A_low Elo anchor — fixed nodes → reproducible, same scale as the round-robin ladder.
# Measure Elo EVERY iter (not just on promotion) — the absolute-strength trend is the scoreboard,
# and nothing promotes during the plateau, so without this Stockfish would never run.
export STOCKFISH_EVERY_ITER=1
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
