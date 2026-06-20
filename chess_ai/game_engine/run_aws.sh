#!/usr/bin/env bash
# run_aws.sh — cloud launcher (Vast/AWS). Wraps run_gcp.sh and injects the box config via EXTRA_ENV
# (sourced AFTER hyperparams so it wins), plus a raised FD limit. Current target: Vast RTX 4090-24GB,
# 64-core EPYC, cu126 torch (Ada — no Blackwell/cu128 dance). Run from the repo root:
#   bash chess_ai/game_engine/run_aws.sh --background
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- Vast 4090-48GB / 64-core config (sourced after hyperparams via the EXTRA_ENV hook) ---
OVR="$HERE/.aws_overrides.env.sh"
cat > "$OVR" <<'ENV'
# Self-play: 120 workers on the 64-core box — the balanced point (empirically: 240 thrashed on context
#   switches + single-server queue contention; 96 under-fed). CUDA_BATCH = 120×8 = 960 (fits 24GB).
export NUM_WORKERS=120
export RESERVED_CORES=2
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Training: batch 2048 (fits 24GB — 4096 OOMs at ~22GB), 32 DL workers × prefetch 2.
export TRAIN_BATCH_SIZE=2048
export TRAIN_DL_WORKERS=32
export TRAIN_DL_PREFETCH=2
# FRESH-START LANDMINE: hyperparams sets TRAIN_MIN_ITER=8 (drop the old corrupted-run pre-iter-8 data).
# On a clean restart from iter 1 that drops ALL data → training is skipped until iter 8. Keep everything.
export TRAIN_MIN_ITER=0

# Eval: 64 workers × 2 games = 128 games for BOTH arena and Stockfish, shared GPU inference servers
#   (server-mode/SHM). 2 games/worker keeps color balance (1 White + 1 Black).
export EVAL_WORKERS=64
export GAMES_PER_EVAL_WORKER=2
export STOCKFISH_WORKERS=64
export STOCKFISH_GAMES=128

# Elo anchor — time-based UCI_Elo=1320 (NODES=0). At pretrained ~1450-1600 this lands the model in a
# readable ~60-70% band with room to climb; fixed-nodes floors SF ~1700+ (too strong, low resolution
# early). Raise the anchor once the model is clearly past 1320. Measure EVERY iter = the scoreboard.
export STOCKFISH_EVERY_ITER=1
export STOCKFISH_ELO=1320
export STOCKFISH_NODES=0
ENV
export EXTRA_ENV="$OVR"

# FD limit — 120 self-play / 64 eval workers × shared servers exhaust the default (~1024) and FDs
# accumulate across iterations (the iter-21 "[Errno 24] Too many open files" crash). Raise to max.
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[aws] FD limit=$(ulimit -n)  |  config → $OVR"

# Hand off to the GCP launcher (privilege/apt/stockfish/build/VRAM-detect/launch all reused).
exec bash "$HERE/run_gcp.sh" "$@"
