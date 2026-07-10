# Self-play on the Genoa box (24GB 4090, EPYC 9654P Zen4 96c, 148GB RAM). RAM IS THE HARD CAP.
#   MEASURED REALITY: ~0.78GB/worker incl SHM. 182w crept past 148GB and got OOM-KILLED mid iter-16
#   (the old "0.4GB/worker → 280 safe" estimate was WRONG — that's what caused the OOM). 160w ≈ 125GB
#   steady → ~23GB margin. Do NOT exceed ~165 here (NO SWAP); watch `free -g`. If it still creeps to
#   OOM over a long run, the per-iteration RAM creep needs a real fix (or a periodic restart).
#   GPU stays queue-starved (latency-bound, sm~72%) but RAM, not the GPU, is the ceiling on this box.
#   CUDA_BATCH auto = NUM_WORKERS×8 (< VRAM_CAP 16000, fits 24GB easily). [[selfplay-gpu-bottleneck]]
export NUM_WORKERS=150
# Reserve 8 of the 96 cores for the GPU-feeding inference server (1 gather + ~6 stream executors).
# The server feed isn't the bottleneck (gather sits ~14% idle), but keeping it off the worker cores
# avoids the deadlock-timeout-self-kill failure mode. Workers get the remaining 88.
export RESERVED_CORES=2
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Batch-gather timeout = 0.02s — TESTED best (0.02 beat 0.03 and 0.05 on wall-clock). Self-play is
# latency-bound (workers block on each round-trip), so a SHORTER timeout = lower per-round-trip wait =
# more leaves/sec. Don't raise it — fuller batches at a longer timeout are a vanity metric. [[selfplay-gpu-bottleneck]]
export CUDA_TIMEOUT_INFERENCE=0.02

# iter-71 OOM fix: training OOM'd at the FIRST forward with ~22.6GB already held and only ~72MB free —
# and batch 1536→1280 freed just 60MB, proving the batch is NOT the hog (self-play VRAM residue +
# fragmentation is). expandable_segments lets the allocator use fragmented gaps instead of failing a
# small alloc when memory is technically available (the error's own suggestion). Set before torch init
# (this override is sourced before python starts). If it STILL OOMs, the residue isn't draining →
# investigate the self-play→training teardown, or raise VRAM_DRAIN_TARGET_GB so training waits for more.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Companion: the pre-training VRAM drain (main.py) defaults to waiting only for 8GB free, then starts
# — but training at batch 1280 needs MORE than that leftover sliver (it OOM'd 100MB short at ~8GB
# free). Wait for 10GB free so self-play residue drains further (the loop calls empty_cache each pass)
# and training has real headroom. iters 68-70 reached ~10GB, so it's attainable; worst case is a 120s
# wait then proceed. Raise toward 12 if it still OOMs.
export VRAM_DRAIN_TARGET_GB=10

# Server self-kill if it processes NO batch for this long (a real hang). Default is 1800s; lowered
# to 600 so a winddown straggler-hang is detected and salvaged (main.py advances to training on the
# games already on disk) in ~10min instead of ~30. The server never legitimately idles this long
# mid-phase with 150 workers always requesting, so 600 won't false-fire. [[selfplay-gpu-bottleneck]]
export SERVER_DEADLOCK_TIMEOUT=600

# Opening exploration: sample the played move ∝ visits^(1/T) for the first TEMP_MOVES plies.
# TEMP_MOVES 16→20: a wider window widens the opening book. SELFPLAY_TEMPERATURE 1.0→0.5 SHARPENS
# the sampling (visits^2) so it stays on the high-visit (sound) moves — cuts the τ=1 junk
# (tempo-wasting knight shuffles, random flank pawns) that made the on-distribution openings noisy,
# while keeping top-move diversity. Wider-but-sharper = more, cleaner opening variety. Watch
# check_diversity.py next iter (don't let openings narrow). Arena/eval still play greedy (T=0).
export TEMP_MOVES=20
export SELFPLAY_TEMPERATURE=1.0

# Opening mix: 5% of games seed from the forced book, 95% play on-distribution (KataGo/Lc0 target).
# 0.05 was tried at iter-41 and collapsed to 96% g1f3 — BUT the root cause was a C++ bug, not the value:
# the played move was argmax(visits) (temperature was inert) and rng reseeded constant every move, so
# τ=1 never actually sampled. FIXED (mcts_engine.cpp samples the move ∝ visits^(1/T) + per-call reseed);
# at 0.5 the on-distribution HALF then held diversity on its own (g1f3 24%, not 96% — iter-42), proving
# sampling now carries it. So the book is no longer the sole source → drop to 0.05. WATCH
# check_diversity.py on the next iter; raise back if the net's sampled openings narrow. [[opening-book-diversity]]
export OPENING_BOOK_PROB=0.1

# Worker pacing: cap a fast worker to ≤3 games ahead of the slowest (was 5) — tighter spread so
# fewer workers finish all 10 and idle while stragglers catch up = less tail-idle at iter end.
export MAX_WORKER_LEAD=4

# Training: batch 2048→1536 for the Net2Net 20x320 net. 320 vs 256 filters = ~1.25× activation/sample,
# AND the KL-anchor (β=1.0) loads a SECOND 320-model copy + forward pass during training — so the 2048
# that fit the 256 net now risks OOM. 1536 ≈ 2048/1.25 with margin for the anchor. If it still OOMs,
# drop to 1280/1024; if VRAM is comfortable on the first iter, you can nudge back up. (LR unaffected:
# arch change → fresh cosine, T_max recomputes for the new batch count.)
# DL workers 90→16: the dataset is loaded fully into RAM, so __getitem__ is pure indexing (no disk I/O)
# and the GPU is the bottleneck — 16 workers keep it fed. 90 was the RAM-balloon culprit: each worker
# copy-on-write touches the numpy/list refcounts + holds prefetch buffers, inflating RSS far above the
# printed f16-array size (it under-counts true process RAM). Fewer workers = much less RAM, no speed loss.
export TRAIN_BATCH_SIZE=1536   # iter-71: 1536→1280 after a CUDA OOM at the START of training (first
                              # forward, 44MB free of 24GB). 1536 was always at the edge for the 20x320
                              # net + the KL-anchor's second model copy; this restart tipped over. 1280
                              # cuts activation mem ~17% (≫ the 120MB overshoot). Drop to 1024 if it recurs.
export TRAIN_DL_WORKERS=120
export TRAIN_DL_PREFETCH=2
# RAM belt: cap each loaded train chunk to ~2M raw pos (~2 chunks at the current window). Since
# main.py now uses sharing_strategy='file_descriptor' (no /dev/shm route — that was the iter-43
# crash), this only bounds peak TRAINING RAM at load (~33GB/chunk), not shm. Trains all data per
# epoch in 2 load passes. Raise toward 3M for single-chunk speed if RAM headroom is confirmed.
export TRAIN_CHUNK_POSITIONS=2500000
# Train on the last 30 iterations of self-play. Reverted 60→30 (was widened to 45/50/60 at iter-40 to
# feed the 3rd epoch + fight overfitting) now that the C++ diversity/temperature bug is fixed (iter-41):
# the wide window kept re-admitting the OLD low-diversity, argmax-temperature (96% g1f3) self-play, which
# dilutes the post-fix teacher signal. 30 was the proven pre-iter-40 default with a tiny train/val gap,
# so reverting is overfitting-safe. NOTE: only iters 41+ are post-fix, so 30 still includes ~pre-fix
# data — narrow further if the goal is purely post-fix data. Now fits ~1 RAM chunk (faster, no chunking).
export TRAIN_WINDOW=20  # iter-92: REVERTED 40→20. The 40/lineage-off/epochs-2 combo was premised on
                        # "candidate is weaker" — the SF-16 A/B DISPROVED that (cand 52% / champ 42% vs SF).
                        # The real bug is the arena gate (see STOCKFISH_GATE below), not training. (hist: 35→10→20)
# FRESH-START LANDMINE: hyperparams sets TRAIN_MIN_ITER=8 (drop the old corrupted-run pre-iter-8 data).
# On a clean restart from iter 1 that drops ALL data → training is skipped until iter 8. Keep everything.
export TRAIN_MIN_ITER=0

# iter-92: TRAIN_EPOCHS reverted to the hyperparams default (1) — the epochs-2 bump was part of the
# false-premise "candidate is weaker" combo (disproven by the SF A/B). No override here.

# Train-from-lineage (AZ-2017): continue training from candidate.pth, not best_model.pth.
# iter-92: REVERTED to 1 (lineage ON). iter-91 flipped it OFF believing the lineage was degrading the
# candidate (arena 0.479→0.441→0.430). But the SF-16 A/B proved the CURRENT lineage candidate is STRONGER
# than the champion vs Stockfish (cand 52% / champ 42%) — the arena "drift" was the head-to-head style
# artifact, NOT real degradation. Lineage was never the problem; the arena gate was. Kept ON.
export TRAIN_FROM_LINEAGE=1

# Eval sizing — BOTH arena and Stockfish at 100 workers × 4 games = 400 games each. 4/worker alternates
#   W,B,W,B = 2 White + 2 Black, color-balanced. 400 games → 95% CI ±4.9% (vs ±5.7% at 300): with the
#   0.50 gate the extra games trim the false-promotion risk of a near-parity net. STOCKFISH_WORKERS
#   mirrors EVAL_WORKERS so both evals size together.
export EVAL_WORKERS=100
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_WORKERS=100
export STOCKFISH_GAMES=150   # iter-92: 400→150. Under STOCKFISH_GATE this runs TWICE/iter (champ + cand),
                            # so 150 each = 300 SF games/iter. ±4% CI each — enough to resolve the ~10-pt
                            # champ↔cand gap (cand 52% vs champ 42% in the A/B). Bump if promotions look noisy.

# STOCKFISH-ANCHORED GATE (iter-92): replace the self-referential arena gate with a paired Stockfish A/B.
# The SF-16 A/B proved the arena rejects candidates that are STRONGER vs a neutral opponent (cand 52% /
# champ 42% @ SF-2300) — it measures a style matchup, not strength, and froze the champion ~80 Elo too low
# for 30+ iters. With the gate ON: each iter measures champ + cand vs SF, promotes iff cand >= champ. The
# arena still runs as a logged DIAGNOSTIC (arena_win_rate stays in metrics.json) but no longer gates.
export STOCKFISH_GATE=1
export STOCKFISH_EVERY_ITER=0   # ignored under the gate (the gate measures Elo every iter itself)
export STOCKFISH_ELO=2300       # the validated A/B band: champ ~42%, cand ~52% → good gate resolution
export STOCKFISH_NODES=0
