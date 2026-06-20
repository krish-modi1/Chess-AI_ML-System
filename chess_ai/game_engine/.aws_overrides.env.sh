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
