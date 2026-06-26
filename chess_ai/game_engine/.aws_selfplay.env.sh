# Self-play FARM tuning for g6e.8xlarge (L40S 48GB, 32-core Zen3, 256GB RAM). Self-play ONLY.
export MIN_TRAIN_ITERS=9999          # > ITERATIONS → every iteration is self-play only (no train/eval)
# 120 workers on 30 cores (2 reserved) = ~4x oversubscription — fine for latency-bound self-play.
# RAM is the cap (~0.77 GB/worker): 120w ≈ 93GB → safe on 128GB, comfortable on the 256GB g6e.
# Sims come from hyperparams (2000 full / 200 fast / 25% recorded) — NOT overridden here, so the bank
# matches the Vast run's teacher-test config. CUDA_BATCH auto-tracks NUM_WORKERS.
export NUM_WORKERS=120
export GAMES_PER_WORKER=20           # games each worker plays before the iteration rolls over
export RESERVED_CORES=2
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE
export CUDA_TIMEOUT_INFERENCE=0.02   # latency-bound: shorter gather timeout = more leaves/sec
export TEMP_MOVES=16                 # wide opening book (matches the main run)
# Loose pacing (lead 20 = no real cap at 20 games/worker): fast workers never wait on stragglers,
# maximizing raw game throughput — the farm doesn't need balanced iteration completion.
export MAX_WORKER_LEAD=20
