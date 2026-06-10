#!/usr/bin/env bash
# hyperparams.env.sh — shared AlphaZero run-config for GCP and local launchers.
#
# CALLER MUST SET before sourcing:
#   NCPU      — logical CPU count (e.g., from nproc)
#   VRAM_MIB  — GPU VRAM in MiB (from nvidia-smi)
#
# Exports the full env-var contract consumed by game_engine/main.py.

# ── CRITICAL: pin every process to ONE CPU thread ─────────────────────────────────────────────
# We parallelise across NUM_WORKERS *processes*, not threads. But numpy/MKL/OpenBLAS/torch each
# spin up a thread pool sized to the box (≈16–32) PER process, and idle OpenMP threads spin-wait
# (busy-loop) → 64 workers × ~16 threads ≈ 1000 threads thrashing 32 vCPUs (observed load avg
# ~570, per-worker 11 sim/s vs ~45 with 1 thread). Pinning to 1 thread each removes the thrash;
# the GPU then gets fed and self-play throughput multiplies. Set BEFORE python imports torch.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE   # if any OMP region remains, idle threads sleep instead of spinning

# Heavily oversubscribed: with the small WORKER_BATCH_SIZE below, each search does ~100
# inference round-trips per move, so a worker spends most of its time BLOCKED waiting on the
# GPU. Oversubscribing keeps the CPU busy (ready workers run while others wait) AND feeds the
# GPU a larger pooled batch (NUM_WORKERS×WBS). Target machine: g2-standard-32 = 16 PHYSICAL
# cores / 32 logical (HT), 126 GB RAM, 22 GB L4.
#   - The C++ MCTS traversal releases the GIL, so ~16 workers can compute in parallel; the rest
#     wait on inference. At a ~20-25% CPU duty cycle that balances ~64 workers against 16 cores.
#   - RAM is not the limit: 64 × ~0.4 GB ≈ 25 GB of 126 GB (no per-worker CUDA context; the
#     single server process owns the GPU).
# 64 is calibrated to the 16 physical cores. Tune via `nvidia-smi` + load average: if GPU util
# < ~85% AND load avg < ~28, raise toward 80-96; if load avg pins near/over 32, back off.
export NUM_WORKERS=64

# Per-worker MCTS leaf batch = leaves submitted to the inference server per search iteration.
# num_iterations = SIMULATIONS / WORKER_BATCH_SIZE. 8 → num_iter=100 = full AlphaZero-quality
# sequential search. The 100 inference round-trips/move used to starve the GPU (~13% util) when
# tensors were pickled through mp.Queue, but the SHARED-MEMORY transport (SHM_TRANSPORT, default
# on) moves the bulk tensors into shared buffers and sends only tiny (worker_id, N) signals — so
# the round-trips are cheap and we keep top search quality AND high GPU utilisation.
export WORKER_BATCH_SIZE=8
# Shared-memory inference transport (set 0 to fall back to the legacy pickle-through-queue path).
export SHM_TRANSPORT=1

# VRAM tier → CUDA stream count + a ceiling on the pooled inference batch.
if   (( VRAM_MIB >= 35000 )); then CUDA_STREAMS=8; VRAM_CAP=24000
elif (( VRAM_MIB >= 20000 )); then CUDA_STREAMS=6; VRAM_CAP=16000
elif (( VRAM_MIB >=  8000 )); then CUDA_STREAMS=4; VRAM_CAP=6000
else                               CUDA_STREAMS=2; VRAM_CAP=1536
fi
export CUDA_STREAMS

# Pooled inference batch = one round from every worker (NUM_WORKERS × WORKER_BATCH_SIZE),
# capped by available VRAM.
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# FIX: 20 ms partial-batch flush (overrides the 1 s hardcoded default in main.py).
export CUDA_TIMEOUT_INFERENCE=0.02

# ── Watchdog timeouts ─────────────────────────────────────────────────────────
# Sized for the small-WORKER_BATCH_SIZE search (~100 inference round-trips/move → longer
# moves, games, and iterations than the old 1-round-trip config).
export ITERATION_TIMEOUT=172800        # 48h per-iteration deadlock guard (main.py SIGALRM)
export INFERENCE_TIMEOUT_MS=300000     # 5min max wait for ONE inference round-trip (per call)
export SERVER_DEADLOCK_TIMEOUT=1800    # 30min: server self-kills if it processes no batch

# Training DataLoader workers. The self-play server is dead during training so all 32 vCPUs are
# free; 16 (= physical cores) feeds the GPU a 2048-batch without starving it, and prefetch_factor=4
# (in trainer.py) keeps 4 batches ready per worker. COW fork shares the read-only f16 dataset →
# no RAM multiplication. Local launchers override this lower (tiny datasets).
export TRAIN_DL_WORKERS=16
export TRAIN_DL_PREFETCH=4   # batches buffered per worker (local launchers set 1)

# Loop / eval / rules — identical on all platforms.
export SIMULATIONS=800
export EVAL_SIMULATIONS=800

# KataGo-style decided-game playout: once |P(win)-P(loss)| >= threshold for N consecutive
# moves, cap sims for the rest of the game (played to completion, no resignation → honest
# labels). Saves compute on dead/decided games so the budget buys more diverse games.
export DECIDED_VALUE_THRESHOLD=0.9
export DECIDED_PATIENCE=5
export REDUCED_SIMULATIONS=100
export GAMES_PER_WORKER=10
export ITERATIONS=1000
export EVAL_WORKERS=10
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_GAMES=50
export STOCKFISH_ELO=1800
export MAX_MOVES_PER_GAME=800
export EVAL_MAX_MOVES_PER_GAME=800
