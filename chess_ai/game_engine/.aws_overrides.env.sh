# Self-play: 120 workers on the 64-core box — the balanced point (empirically: 240 thrashed on context
#   switches + single-server queue contention; 96 under-fed). CUDA_BATCH = 120×8 = 960 (fits 24GB).
export NUM_WORKERS=120
export RESERVED_CORES=2
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Opening exploration: τ=1 sampling for the first 16 plies (hyperparams halved it to 8 "to stay
# on-distribution" — a corrupted-era call, now retired). Restore 16: self-play funneled into ~7
# distinct openings/2000 games (peaked g3 prior); a longer temp window widens the opening book.
export TEMP_MOVES=16

# Worker pacing: cap a fast worker to ≤3 games ahead of the slowest (was 5) — tighter spread so
# fewer workers finish all 10 and idle while stragglers catch up = less tail-idle at iter end.
export MAX_WORKER_LEAD=3

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
