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
# 60 workers on 30 cores (2 per core), with RESERVED_CORES=2 keeping the GPU-feeding inference
# server + OS on the top 2 cores (dedicated, no worker contention). A starved server can fall
# behind, trip its deadlock timeout, self-kill, and hang every worker — reserving cores prevents
# that cascade. Tune via load average + nvidia-smi.
export NUM_WORKERS=60
export RESERVED_CORES=2

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
export EVAL_WORKERS=30           # arena: 30 workers × 5 games = 150 games (stabler promotion gate)
export GAMES_PER_EVAL_WORKER=5
export STOCKFISH_GAMES=100       # stockfish: 100 games for a tighter Elo estimate
export STOCKFISH_WORKERS=25      # 25 CPU-Stockfish workers × 4 games each = 100
export STOCKFISH_ELO=1800
export MAX_MOVES_PER_GAME=800
export EVAL_MAX_MOVES_PER_GAME=800

# RL training recipe — gentle fine-tune of the 1800 Lichess-pretrained net. The raw policy prior
# is weak off the human distribution (gives MCTS-chosen moves ~0.7% median mass), so targets
# demand large shifts; 4 epochs × 3e-4 over a tiny early window overwrote the net (1800→~1300).
# See local/plans/anti-forgetting-levers.md.
export TRAIN_EPOCHS=1          # was 4 — stop hammering tiny data
export TRAIN_LR=1e-4           # was 3e-4 — gentler AdamW fine-tune
export KL_ANCHOR_BETA=1.0      # KL(pretrained prior ‖ candidate) anti-forgetting penalty; 0 disables
export MIN_TRAIN_ITERS=3       # self-play only until this many iterations of data exist, then train
export DRAW_MAX_POSITIONS=0        # subsumed by MAX_POSITIONS_PER_GAME below
export MAX_POSITIONS_PER_GAME=20   # cap EVERY game to 20 positions (value-target decorrelation, AlphaGo Nature'16).
                                   # Offline A/B sweet spot: value head 1.5× sharper + best calibration (56%), policy intact;
                                   # caps ≤5 over-confident (shared trunk ≠ AlphaGo's separate value net).
export VALUE_LOSS_WEIGHT=1.0       # no-op: A/B showed value-loss weight has zero effect (flattening is trunk/BN-driven)
