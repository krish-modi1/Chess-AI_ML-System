#!/usr/bin/env bash
# hyperparams.env.sh — shared AlphaZero run-config for GCP and local launchers.
#
# CALLER MUST SET before sourcing:
#   NCPU      — logical CPU count (e.g., from nproc)
#   VRAM_MIB  — GPU VRAM in MiB (from nvidia-smi)
#
# Exports the full env-var contract consumed by game_engine/main.py.

# 60 workers — oversubscribed (workers block on inference); 2 cores reserved for system/server.
export NUM_WORKERS=30

# VRAM tier → worker leaf batch per call, CUDA stream count, VRAM-safe batch cap.
#
#  ≥35 GB  A100 40/80 GB  : WORKER_BATCH_SIZE=256, cap=8192
#  ≥20 GB  A30 / V100-32  : WORKER_BATCH_SIZE=200, cap=6144
#  ≥ 8 GB  T4 / V100-16   : WORKER_BATCH_SIZE=160, cap=4096
#  < 8 GB  laptop (6 GB)  : WORKER_BATCH_SIZE=32,  cap=1536
if   (( VRAM_MIB >= 35000 )); then CUDA_STREAMS=8; VRAM_CAP=24000
elif (( VRAM_MIB >= 20000 )); then CUDA_STREAMS=6; VRAM_CAP=16000
elif (( VRAM_MIB >=  8000 )); then CUDA_STREAMS=4; VRAM_CAP=6000
else                               CUDA_STREAMS=2; VRAM_CAP=1536
fi

# WORKER_BATCH_SIZE = VRAM_CAP / NUM_WORKERS so all workers fill exactly one inference
# batch per MCTS round — no multi-round stalls regardless of worker count.
WORKER_BATCH_SIZE=$(( VRAM_CAP / NUM_WORKERS ))
(( WORKER_BATCH_SIZE < 1 )) && WORKER_BATCH_SIZE=1
export WORKER_BATCH_SIZE CUDA_STREAMS

export CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))

# FIX: 20 ms partial-batch flush (overrides the 1 s hardcoded default in main.py).
export CUDA_TIMEOUT_INFERENCE=0.02

# Loop / eval / rules — identical on all platforms.
export SIMULATIONS=800
export EVAL_SIMULATIONS=800
export GAMES_PER_WORKER=20
export ITERATIONS=1000
export EVAL_WORKERS=10
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_GAMES=50
export STOCKFISH_ELO=1800
export MAX_MOVES_PER_GAME=800
export EVAL_MAX_MOVES_PER_GAME=800
