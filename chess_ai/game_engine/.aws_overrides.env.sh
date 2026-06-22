# Self-play: 280 workers on the Genoa box — 24GB VRAM, EPYC 9654P (Zen4), 96 cores, ~148GB RAM.
#   MEASURED at 200w: sm ~72% avg (100% peaks), [server-batch] ~500/1600 @ 100% timeout (queue-STARVED),
#   load only 15/96 (CPU idle). Same starved+idle corner as Romania → more producers fill the batch.
#   NEW CEILING IS RAM, not cores: ~0.4GB/worker → 280w ≈ 115GB + ~11GB/iter creep ≈ 126GB peak, ~22GB
#   margin on 148GB (NO SWAP — do NOT exceed ~300). CUDA_BATCH=280×8=2240 (< VRAM_CAP 16000, inference
#   fits 24GB easily). Watch free -g <130. Latency-bound — see [[selfplay-gpu-bottleneck]].
export NUM_WORKERS=182
# Reserve 8 of the 96 cores for the GPU-feeding inference server (1 gather + ~6 stream executors).
# The server feed isn't the bottleneck (gather sits ~14% idle), but keeping it off the worker cores
# avoids the deadlock-timeout-self-kill failure mode. Workers get the remaining 88.
export RESERVED_CORES=8
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Batch-gather timeout = 0.02s — TESTED best (0.02 beat 0.03 and 0.05 on wall-clock). Self-play is
# latency-bound (workers block on each round-trip), so a SHORTER timeout = lower per-round-trip wait =
# more leaves/sec. Don't raise it — fuller batches at a longer timeout are a vanity metric. [[selfplay-gpu-bottleneck]]
export CUDA_TIMEOUT_INFERENCE=0.02

# Opening exploration: τ=1 sampling for the first 16 plies (hyperparams halved it to 8 "to stay
# on-distribution" — a corrupted-era call, now retired). Restore 16: self-play funneled into ~7
# distinct openings/2000 games (peaked g3 prior); a longer temp window widens the opening book.
export TEMP_MOVES=16

# Worker pacing: cap a fast worker to ≤3 games ahead of the slowest (was 5) — tighter spread so
# fewer workers finish all 10 and idle while stragglers catch up = less tail-idle at iter end.
export MAX_WORKER_LEAD=3

# Training: batch 2048 — fits the 24GB card (4096 OOMs at ~22GB here; that was a 48GB-only setting).
# 50 DL workers × prefetch 2. Genoa box has 193GB RAM so DL/RAM is never the limit.
export TRAIN_BATCH_SIZE=2048
export TRAIN_DL_WORKERS=50
export TRAIN_DL_PREFETCH=2
# FRESH-START LANDMINE: hyperparams sets TRAIN_MIN_ITER=8 (drop the old corrupted-run pre-iter-8 data).
# On a clean restart from iter 1 that drops ALL data → training is skipped until iter 8. Keep everything.
export TRAIN_MIN_ITER=0

# Train-from-lineage (AZ-2017): continue training from candidate.pth, not best_model.pth. NO-OP while
# candidates keep promoting (promotion copies candidate→best, so the two bases are identical). Bites
# only on the FIRST rejection: lineage continues from the rejected candidate (no wasted learning)
# instead of resetting to champion. KL-anchor stays pinned to pretrained. Reversible: flip to 0 to
# reset to champion if a lineage ever stalls (arena gate keeps self-play clean throughout).
export TRAIN_FROM_LINEAGE=1

# Arena: 50 workers × 4 games = 200 games (tighter promotion gate). 4/worker = 2 White + 2 Black,
#   stays color-balanced. Stockfish eval kept at 64×... (its own knobs below).
export EVAL_WORKERS=50
export GAMES_PER_EVAL_WORKER=2
export STOCKFISH_WORKERS=100
export STOCKFISH_GAMES=200

# Stockfish/Elo ONLY on promotion (EVERY_ITER=0): a rejected iter leaves best_model unchanged, so its
# Elo is unchanged — re-measuring it just burns 200 games and adds ±CI noise to the trend. Now that
# BayesElo is reliable, measure the champion once per champion (on promotion); the logger carries the
# last Elo forward on non-promoted iters (elo_measured=false). Arena still runs every iter (the gate).
export STOCKFISH_EVERY_ITER=0
export STOCKFISH_ELO=2800
export STOCKFISH_NODES=0
