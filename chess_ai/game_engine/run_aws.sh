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
# Self-play: 184 workers on the 96-core box (~1.9/core — same density as the working 120/64, so no
#   240-style per-core thrash). GPU measured at only ~55% sm at 120w → starved, not saturated; more
#   producers aim to fill it. CUDA_BATCH = 184×8 = 1472 (< VRAM_CAP 16000, fits 24GB; train runs 2048).
export NUM_WORKERS=184
# Reserve the top 6 cores for the GPU-feeding inference server (1 gather thread + 6 stream executors).
# At 2 on a 96-core box the server competes with 184 workers for CPU — and the server feed is the
# suspected ~55%-util bottleneck, so give it dedicated cores. Workers get the remaining 90 (~2/core).
export RESERVED_CORES=6
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

# Training: batch 2048 (fits 24GB — 4096 OOMs at ~22GB), 32 DL workers × prefetch 2.
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
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_WORKERS=50
export STOCKFISH_GAMES=200

# Elo anchor — time-based UCI_Elo=1800 (NODES=0). iter-4 SWEPT 1320 200-0 (BayesElo could only
# floor it at ~2154), so 1320 has zero resolution now. 1800 ∈ SF-16 range [1320,3190]; raised to
# pull the score back into a measurable ~30-70% band. Bump again if the model sweeps 1800 too.
export STOCKFISH_EVERY_ITER=1
export STOCKFISH_ELO=1800
export STOCKFISH_NODES=0
ENV
export EXTRA_ENV="$OVR"

# FD limit — 120 self-play / 64 eval workers × shared servers exhaust the default (~1024) and FDs
# accumulate across iterations (the iter-21 "[Errno 24] Too many open files" crash). Raise to max.
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[aws] FD limit=$(ulimit -n)  |  config → $OVR"

# Hand off to the GCP launcher (privilege/apt/stockfish/build/VRAM-detect/launch all reused).
exec bash "$HERE/run_gcp.sh" "$@"
