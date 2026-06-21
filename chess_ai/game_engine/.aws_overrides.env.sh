# Self-play: 200 workers on the Romania box — 48GB VRAM, EPYC 7V73X (Zen3 + 3D V-cache), 64 vCPUs.
#   MEASURED at 120w: sm ~71% (V-cache lifted it from the old 55%), but [server-batch] showed avg-fill
#   370/960 @ 100% timeout (queue-STARVED) with load only 9.7/64 (CPU idle). So the batch is under-fed
#   and there's CPU to spare → MORE workers fill it. 200 projects ~600/960 + ~85% sm. CUDA_BATCH=1600.
#   Push to 240-280 if avg-fill keeps rising with sm; stop when avg-fill plateaus (gather saturated) or
#   load nears ~50/64. (Timeout is NOT the lever — fixed leaf rate just repackages into bigger batches.)
export NUM_WORKERS=200
# Reserve 8 of the 64 vCPUs for the GPU-feeding inference server (1 gather + 8 stream executors,
# CUDA_STREAMS=8 at the 48GB tier). It's the feed bottleneck and the reason we're on a V-cache box —
# give it dedicated fast cores, off the 120 workers' 56 cores.
export RESERVED_CORES=8
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Batch-gather timeout 0.02→0.03s (the experiment): give partial batches 10ms more to fill before
# dispatch. NO-OP if batches already hit the cap; helps only if they're timing out small. Watch the
# [server-batch] log (avg fill / hit-cap%) + nvidia-smi sm% to judge — NOT load average.
export CUDA_TIMEOUT_INFERENCE=0.03

# Opening exploration: τ=1 sampling for the first 16 plies (hyperparams halved it to 8 "to stay
# on-distribution" — a corrupted-era call, now retired). Restore 16: self-play funneled into ~7
# distinct openings/2000 games (peaked g3 prior); a longer temp window widens the opening book.
export TEMP_MOVES=16

# Worker pacing: cap a fast worker to ≤3 games ahead of the slowest (was 5) — tighter spread so
# fewer workers finish all 10 and idle while stragglers catch up = less tail-idle at iter end.
export MAX_WORKER_LEAD=3

# Training: batch 4096 — was 2048 (capped by the old 24GB card; 4096 OOM'd there). On 48GB with the
# inference servers stopped during training, 4096 fits with headroom. Same LR (1e-4) → training is a
# touch more conservative (half the steps/epoch, less gradient noise) — fine under the β=1.0 KL anchor;
# bump LR ~1.4× later only if value loss stalls. 50 DL workers × prefetch 2.
export TRAIN_BATCH_SIZE=4096
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
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_WORKERS=50
export STOCKFISH_GAMES=200

# Elo anchor — time-based UCI_Elo=2100 (NODES=0). iter-5 scored 94% vs SF-1800 (BayesElo 2294±42),
# nearly a sweep → wide bars. 2100 ∈ SF-16 spec [1320,3190] (verified live: UCI_LimitStrength+UCI_Elo
# applies, no silent default) → puts a ~2294 model at ~75%, a tight readable band. Raise again as it climbs.
export STOCKFISH_EVERY_ITER=1
export STOCKFISH_ELO=2100
export STOCKFISH_NODES=0
