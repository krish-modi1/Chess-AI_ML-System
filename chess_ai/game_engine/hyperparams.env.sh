#!/usr/bin/env bash
# hyperparams.env.sh — shared AlphaZero run-config for GCP and local launchers.
#
# CALLER MUST SET before sourcing:
#   NCPU      — logical CPU count (e.g., from nproc)
#   VRAM_MIB  — GPU VRAM in MiB (from nvidia-smi)
#
# Exports the full env-var contract consumed by game_engine/main.py.

# Workers: all logical cores minus 2 (inference server + OS), min 1.
export NUM_WORKERS=$(( NCPU > 2 ? NCPU - 2 : 1 ))

# VRAM tier → worker leaf batch per call, CUDA stream count, VRAM-safe batch cap.
#
#  ≥35 GB  A100 40/80 GB  : WORKER_BATCH_SIZE=256, cap=8192
#  ≥20 GB  A30 / V100-32  : WORKER_BATCH_SIZE=200, cap=6144
#  ≥ 8 GB  T4 / V100-16   : WORKER_BATCH_SIZE=160, cap=4096
#  < 8 GB  laptop (6 GB)  : WORKER_BATCH_SIZE=32,  cap=1536
if   (( VRAM_MIB >= 35000 )); then
  CUDA_STREAMS=8; VRAM_CAP=8192
  # a2 instances (A100) are fixed at 12 vCPUs → 10 workers; larger WBS fills bigger batches
  # and halves round-trips (800÷400=2 vs 800÷256=4) at no memory cost.
  if (( NCPU <= 14 )); then WORKER_BATCH_SIZE=400
  else                      WORKER_BATCH_SIZE=256
  fi
elif (( VRAM_MIB >= 20000 )); then WORKER_BATCH_SIZE=200; CUDA_STREAMS=6; VRAM_CAP=6144
elif (( VRAM_MIB >=  8000 )); then WORKER_BATCH_SIZE=160; CUDA_STREAMS=4; VRAM_CAP=4096
else                               WORKER_BATCH_SIZE=32;  CUDA_STREAMS=2; VRAM_CAP=1536
fi
export WORKER_BATCH_SIZE CUDA_STREAMS

# FIX: CUDA_BATCH_SIZE = min(max-concurrent-leaves, VRAM_CAP) so the inference
# server batch always fills immediately — no 1-second timeout stall per MCTS round.
MAX_CONCURRENT=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
export CUDA_BATCH_SIZE=$(( MAX_CONCURRENT < VRAM_CAP ? MAX_CONCURRENT : VRAM_CAP ))

# FIX: 20 ms partial-batch flush (overrides the 1 s hardcoded default in main.py).
export CUDA_TIMEOUT_INFERENCE=0.02

# Loop / eval / rules — identical on all platforms.
export SIMULATIONS=800
export EVAL_SIMULATIONS=800
export GAMES_PER_WORKER=2
export ITERATIONS=1000
export EVAL_WORKERS=10
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_GAMES=50
export STOCKFISH_ELO=1800
export MAX_MOVES_PER_GAME=800
export EVAL_MAX_MOVES_PER_GAME=800
