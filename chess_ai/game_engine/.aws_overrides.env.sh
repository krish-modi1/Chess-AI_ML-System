# Self-play on the Genoa box (24GB 4090, EPYC 9654P Zen4 96c, 148GB RAM). RAM IS THE HARD CAP.
#   MEASURED REALITY: ~0.78GB/worker incl SHM. 182w crept past 148GB and got OOM-KILLED mid iter-16
#   (the old "0.4GB/worker → 280 safe" estimate was WRONG — that's what caused the OOM). 160w ≈ 125GB
#   steady → ~23GB margin. Do NOT exceed ~165 here (NO SWAP); watch `free -g`. If it still creeps to
#   OOM over a long run, the per-iteration RAM creep needs a real fix (or a periodic restart).
#   GPU stays queue-starved (latency-bound, sm~72%) but RAM, not the GPU, is the ceiling on this box.
#   CUDA_BATCH auto = NUM_WORKERS×8 (< VRAM_CAP 16000, fits 24GB easily). [[selfplay-gpu-bottleneck]]
export NUM_WORKERS=160
# Reserve 8 of the 96 cores for the GPU-feeding inference server (1 gather + ~6 stream executors).
# The server feed isn't the bottleneck (gather sits ~14% idle), but keeping it off the worker cores
# avoids the deadlock-timeout-self-kill failure mode. Workers get the remaining 88.
export RESERVED_CORES=4
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Batch-gather timeout = 0.02s — TESTED best (0.02 beat 0.03 and 0.05 on wall-clock). Self-play is
# latency-bound (workers block on each round-trip), so a SHORTER timeout = lower per-round-trip wait =
# more leaves/sec. Don't raise it — fuller batches at a longer timeout are a vanity metric. [[selfplay-gpu-bottleneck]]
export CUDA_TIMEOUT_INFERENCE=0.02

# Server self-kill if it processes NO batch for this long (a real hang). Default is 1800s; lowered
# to 600 so a winddown straggler-hang is detected and salvaged (main.py advances to training on the
# games already on disk) in ~10min instead of ~30. The server never legitimately idles this long
# mid-phase with 150 workers always requesting, so 600 won't false-fire. [[selfplay-gpu-bottleneck]]
export SERVER_DEADLOCK_TIMEOUT=600

# Opening exploration: τ=1 sampling for the first 16 plies (hyperparams halved it to 8 "to stay
# on-distribution" — a corrupted-era call, now retired). Restore 16: self-play funneled into ~7
# distinct openings/2000 games (peaked g3 prior); a longer temp window widens the opening book.
export TEMP_MOVES=16

# Opening mix: 5% of games seed from the forced book, 95% play on-distribution (KataGo/Lc0 target).
# 0.05 was tried at iter-41 and collapsed to 96% g1f3 — BUT the root cause was a C++ bug, not the value:
# the played move was argmax(visits) (temperature was inert) and rng reseeded constant every move, so
# τ=1 never actually sampled. FIXED (mcts_engine.cpp samples the move ∝ visits^(1/T) + per-call reseed);
# at 0.5 the on-distribution HALF then held diversity on its own (g1f3 24%, not 96% — iter-42), proving
# sampling now carries it. So the book is no longer the sole source → drop to 0.05. WATCH
# check_diversity.py on the next iter; raise back if the net's sampled openings narrow. [[opening-book-diversity]]
export OPENING_BOOK_PROB=0.05

# Worker pacing: cap a fast worker to ≤3 games ahead of the slowest (was 5) — tighter spread so
# fewer workers finish all 10 and idle while stragglers catch up = less tail-idle at iter end.
export MAX_WORKER_LEAD=10

# Training: batch 2048 — fits the 24GB card (4096 OOMs at ~22GB here; that was a 48GB-only setting).
# DL workers 90→16: the dataset is loaded fully into RAM, so __getitem__ is pure indexing (no disk I/O)
# and the GPU is the bottleneck — 16 workers keep it fed. 90 was the RAM-balloon culprit: each worker
# copy-on-write touches the numpy/list refcounts + holds prefetch buffers, inflating RSS far above the
# printed f16-array size (it under-counts true process RAM). Fewer workers = much less RAM, no speed loss.
export TRAIN_BATCH_SIZE=2048
export TRAIN_DL_WORKERS=16
export TRAIN_DL_PREFETCH=2
# RAM belt: cap each loaded train chunk to ~2M raw pos (~2 chunks at the current window). Since
# main.py now uses sharing_strategy='file_descriptor' (no /dev/shm route — that was the iter-43
# crash), this only bounds peak TRAINING RAM at load (~33GB/chunk), not shm. Trains all data per
# epoch in 2 load passes. Raise toward 3M for single-chunk speed if RAM headroom is confirmed.
export TRAIN_CHUNK_POSITIONS=2000000
# Train on the last 45 iterations of self-play (iter-40: widened 30→45 to give the 3rd training epoch
# more unique positions per pass = less overfitting). Tradeoff vs the old 30: re-admits the older
# lower-sim early iters (mild teacher-signal dilution). Peak training RAM is bounded by the
# TRAIN_CHUNK_POSITIONS=2M cap above, so the wide window stays within the box's RAM.
export TRAIN_WINDOW=50
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
export GAMES_PER_EVAL_WORKER=6
export STOCKFISH_WORKERS=50
export STOCKFISH_GAMES=300

# Stockfish/Elo ONLY on promotion (EVERY_ITER=0): a rejected iter leaves best_model unchanged, so its
# Elo is unchanged — re-measuring it just burns 200 games and adds ±CI noise to the trend. Now that
# BayesElo is reliable, measure the champion once per champion (on promotion); the logger carries the
# last Elo forward on non-promoted iters (elo_measured=false). Arena still runs every iter (the gate).
export STOCKFISH_EVERY_ITER=0
export STOCKFISH_ELO=2800
export STOCKFISH_NODES=0
