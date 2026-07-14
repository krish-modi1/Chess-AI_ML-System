#!/usr/bin/env bash
# run_aws.sh — cloud launcher (Vast/AWS). Wraps run_gcp.sh and injects the box config via EXTRA_ENV
# (sourced AFTER hyperparams so it wins), plus a raised FD limit. Current target: Vast RTX 4090-24GB,
# 64-core EPYC, cu126 torch (Ada — no Blackwell/cu128 dance). Run from the repo root:
#   bash chess_ai/game_engine/run_aws.sh --background
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- Vast 4090-24GB / 64-core config (sourced after hyperparams via the EXTRA_ENV hook) ---
OVR="$HERE/.aws_overrides.env.sh"
cat > "$OVR" <<'ENV'
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

# Opening mix: fraction of games seeded from the forced 20-opening book; the rest play on-distribution.
# iter-42 (book=0.5) the on-dist half held diversity alone (g1f3 24%) — that was taken as proof that τ+
# sampling carries it, so the book was dropped. THAT NO LONGER HOLDS: the net's root policy has since
# collapsed onto Nf3. MEASURED at iter-105+: g1f3 = 83-88% of all games, e2e4/d2d4 < 0.6% EACH — the
# engine essentially never plays the mainlines of chess. Raising FULL_SEARCH_PROB 0.25→0.6 (which doubles
# Dirichlet coverage, since main.py ties `use_dirichlet=is_full`) moved it only 87.9%→83.2% — i.e.
# Dirichlet PERTURBS a peaked root, it cannot FLATTEN one. The book is the only lever that works.
# iter-108: 0.3 → 0.5 (aggressive, the value that gave g1f3 24%). Costs zero compute — book plies still
# get a recorded full-search target. Suspected cause of the value head degrading (val v-loss 0.478→0.504
# while POLICY improves): self-play confined to one opening family. [[opening-book-diversity]]
export OPENING_BOOK_PROB=0.5

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
export TRAIN_WINDOW=25  # iter-100: 35→25. The 35/epochs-2 combo overfit at iter-99 (Va-V rose
                        # 0.477→0.489 while Tr fell), so back off toward the 20 baseline but keep a bit
                        # more data than 20. Near-parity drought persists (it96-98 candidates ~50% vs
                        # SF-2500); the search probe (KL(MCTS‖net)=0.48, Δ=-0.05) says the real lever is
                        # self-play SIMS/teacher strength, not window/epochs. (hist: 35→10→20→35→25)
# FRESH-START LANDMINE: hyperparams sets TRAIN_MIN_ITER=8 (drop the old corrupted-run pre-iter-8 data).
# On a clean restart from iter 1 that drops ALL data → training is skipped until iter 8. Keep everything.
export TRAIN_MIN_ITER=0

# iter-100: TRAIN_EPOCHS 2→1 (back to hyperparams default). iter-99 RAN epochs=2 and it overfit within
# the iteration — the 2nd pass dropped train (Tr-P 1.164→1.146, Tr-V 0.475→0.472) but RAISED val
# (Va-P 1.156→1.158, Va-V 0.477→0.489, Va-Acc 68.8→68.7). So the bounded per-iter overfit is REAL, just
# smaller than lineage-ON's unbounded version. 1 epoch = one clean pass, no 2nd-epoch overfit.
export TRAIN_EPOCHS=1

# Train-from-lineage (AZ-2017): continue training from candidate.pth, not best_model.pth.
# iter-95: FLIPPED OFF. This time the SF gate (GROUND TRUTH, not the arena) shows REAL lineage
# regression: the WR-gate bug froze the self-play generator at the iter-91 champion (it kept rejecting
# stronger candidates), and with the generator frozen the lineage candidate drifted DOWN toward it —
# cand Elo 2592 (iter-94) → 2484 (iter-95), now BELOW the champion (2498). Fresh-from-champion each iter
# = champion + MCTS improvement, robustly >= champion, so promotions unfreeze the generator and it climbs
# instead of spiraling. The old "lineage-off = ~48% mirror" was an ARENA artifact (fresh nets play alike,
# arena can't tell them apart); the SF gate reads the real improvement, so lineage-off is safe now.
export TRAIN_FROM_LINEAGE=0

# SELF-PLAY TEACHER (iter-106). History: SIMS_DIAGNOSTIC (champ @4000 vs same champ @2000, 150 games)
# = 65.7% ± 7.6% (~+113 Elo) → depth MATTERS, the 2000-sim teacher was under-powered. (That also kills the
# it77-79 "target quality is not the wall" verdict, which was reached under the BROKEN arena gate.)
# iter-100 tried 4000 sims @ 0.25 prob to buy depth at flat compute — but 0.25 BACKFIRED on two fronts:
#   1) DATA COLLAPSE: recorded targets/game halved → window fell 1.97M→1.56M pos (heading ~1.0M) and the
#      candidate UNDERFIT (train AND val loss both rose in lockstep, acc fell) → stuck at parity.
#   2) DIVERSITY COLLAPSE: `use_dirichlet=is_full` (main.py) ties root noise to FULL moves ONLY, so 0.25
#      cut exploration noise to 25% of moves → self-play opened g1f3 87.9% of the time (e4/d4/c4 ~0.3%
#      each, measured over 4000 games). Peaked root policy → narrow data → more peaked. Feedback loop.
# FIX: prioritise DIVERSITY + DATA over raw depth — they are the binding constraints (a deeper teacher
# searching the same g1f3 monoculture doesn't help). 2400 sims @ 0.6 prob:
#   cost 0.6·2400 + 0.4·200 = 1520 sims/move (1.38× the old 1100) — cheaper than 4000@0.5 (2100).
#   Dirichlet + recorded coverage 25% → 60% (better than the original 50% on both).
#   Depth 2000→2400 keeps a modest teacher gain (~+30 Elo by log-sims scaling; full +113 needs 4000).
# Revisit depth (→3000/4000) once diversity + window positions recover.
# iter-109: 2400 → 1200, forced by RAM. The virtual-loss fix means a search now expands ~SIMULATIONS
# DISTINCT leaves (was ~26% of them), and expand() materialises a full ChessBoard — carrying the
# repetition-history stack, which grows with game length — for EVERY legal move. Measured tree cost per
# search at 60 plies: 600 sims → 69 MB, 1200 → 141 MB, 2400 → 284 MB (≈ linear). At 100 workers and a
# ~200-ply endgame that overran the 148 GB box. Halving sims halves tree RAM AND halves time per move,
# so games/hour RISES — strictly better than cutting workers.
# NOTE this is still a BETTER teacher than anything that produced iters 1-107: 1200 real sims vs the
# ~636 effective sims that 2400 nominal actually bought under the broken virtual loss.
# Restore toward 2400+ once expand() stops eagerly building a board for every unvisited child (~29 of
# every 30 boards it allocates are never visited — a ~10× cut is available there).
export SIMULATIONS=1200
export FULL_SEARCH_PROB=0.6

# Eval sizing — BOTH arena and Stockfish at 100 workers × 4 games = 400 games each. 4/worker alternates
#   W,B,W,B = 2 White + 2 Black, color-balanced. 400 games → 95% CI ±4.9% (vs ±5.7% at 300): with the
#   0.50 gate the extra games trim the false-promotion risk of a near-parity net. STOCKFISH_WORKERS
#   mirrors EVAL_WORKERS so both evals size together.
export EVAL_WORKERS=100
export GAMES_PER_EVAL_WORKER=4
# iter-109: 150 → 100, mirroring the self-play NUM_WORKERS cut. The virtual-loss fix multiplied the
# nodes each MCTS tree allocates by ~3.8× (a search now expands ~EVAL_SIMULATIONS distinct leaves
# instead of ~26% of them, and expand() materialises a child board per legal move), which OOM'd the
# 148 GB no-swap box in self-play. The SF-gate phase runs the SAME trees plus 150 Stockfish processes,
# so it would OOM next. Colour balance is unaffected: cand_is_white = (game_id % 2 == 0) over
# game_id ∈ [0, 300), so it's exactly 150 W / 150 B regardless of the per-worker split (now 3 each).
export STOCKFISH_WORKERS=100
export STOCKFISH_GAMES=300   # iter-95: champ AND cand both run 300 games, SYMMETRICALLY (champ no
                            # longer 2×). Consistent, tighter BayesElo (±~32 Elo) so the Elo gate
                            # resolves near-parity candidates.

# STOCKFISH-ANCHORED GATE (iter-92): replace the self-referential arena gate with a paired Stockfish A/B.
# The SF-16 A/B proved the arena rejects candidates that are STRONGER vs a neutral opponent (cand 52% /
# champ 42% @ SF-2300) — it measures a style matchup, not strength, and froze the champion ~80 Elo too low
# for 30+ iters. With the gate ON: each iter measures champ + cand vs SF, promotes iff cand >= champ. The
# arena still runs as a logged DIAGNOSTIC (arena_win_rate stays in metrics.json) but no longer gates.
export STOCKFISH_GATE=1
export STOCKFISH_EVERY_ITER=0   # ignored under the gate (the gate measures Elo every iter itself)
# iter-95: gate now compares ANCHORED Elo (cand_elo >= champ_elo), NOT raw win rate — because the anchor
# is raised as the net strengthens and a WR vs SF-2500 isn't comparable to a WR vs SF-2300. The champ
# baseline is re-measured whenever this anchor changes (champ_sf.json keys on sf_elo), so both sides are
# always scored at the SAME SF. Raise STOCKFISH_ELO further once the champion is comfortably beating it.
export STOCKFISH_ELO=2500       # raised 2300→2500: the champion was crushing SF-2300 (WR saturating)
export STOCKFISH_NODES=0
ENV
export EXTRA_ENV="$OVR"

# Seed model checkpoints from Google Drive on a FRESH box (models are gitignored; a new Vast
# instance has the code + self-play data from git but no .pth). Guarded: only runs if best_model
# is missing, or force with SYNC_MODELS=1. Continues even if gdown fails (manual scp fallback).
MODEL_DIR="$HERE/model"
if [[ ! -f "$MODEL_DIR/best_model.pth" || "${SYNC_MODELS:-0}" == "1" ]]; then
  echo "[aws] seeding models from Drive (best_model.pth missing or SYNC_MODELS=1) ..."
  python3 -c "import gdown" 2>/dev/null || python3 -m pip install --user --quiet gdown
  STAGE="$(mktemp -d)"
  if gdown --folder "https://drive.google.com/drive/folders/1J_cfNiIusiVMhUjTZ0rI80gpKo3XEu_u" -O "$STAGE"; then
    mkdir -p "$MODEL_DIR"
    find "$STAGE" -type f \( -name '*.pth' -o -name '*.bak' \) -exec cp -fv {} "$MODEL_DIR/" \;
  else
    echo "[aws] WARNING: gdown failed — scp the .pth files into $MODEL_DIR manually." >&2
  fi
  rm -rf "$STAGE"
else
  echo "[aws] models present ($MODEL_DIR/best_model.pth) — skipping Drive seed."
fi

# FD limit — 120 self-play / 64 eval workers × shared servers exhaust the default (~1024) and FDs
# accumulate across iterations (the iter-21 "[Errno 24] Too many open files" crash). Raise to max.
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[aws] FD limit=$(ulimit -n)  |  config → $OVR"

# Hand off to the GCP launcher (privilege/apt/stockfish/build/VRAM-detect/launch all reused).
exec bash "$HERE/run_gcp.sh" "$@"
