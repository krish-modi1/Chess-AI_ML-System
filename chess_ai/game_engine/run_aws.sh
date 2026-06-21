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
# Self-play: 280 workers on the Genoa box — 24GB VRAM, EPYC 9654P (Zen4), 96 cores, ~148GB RAM.
#   MEASURED at 200w: sm ~72% avg (100% peaks), [server-batch] ~500/1600 @ 100% timeout (queue-STARVED),
#   load only 15/96 (CPU idle). Same starved+idle corner as Romania → more producers fill the batch.
#   NEW CEILING IS RAM, not cores: ~0.4GB/worker → 280w ≈ 115GB + ~11GB/iter creep ≈ 126GB peak, ~22GB
#   margin on 148GB (NO SWAP — do NOT exceed ~300). CUDA_BATCH=280×8=2240 (< VRAM_CAP 16000, inference
#   fits 24GB easily). Watch free -g <130. Latency-bound — see [[selfplay-gpu-bottleneck]].
export NUM_WORKERS=280
# Reserve 8 of the 96 cores for the GPU-feeding inference server (1 gather + ~6 stream executors).
# The server feed isn't the bottleneck (gather sits ~14% idle), but keeping it off the worker cores
# avoids the deadlock-timeout-self-kill failure mode. Workers get the remaining 88.
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

# Elo anchor — time-based UCI_Elo=2100 (NODES=0). iter-5 scored 94% vs SF-1800 (BayesElo 2294±42),
# nearly a sweep → wide bars. 2100 ∈ SF-16 spec [1320,3190] (verified live: UCI_LimitStrength+UCI_Elo
# applies, no silent default) → puts a ~2294 model at ~75%, a tight readable band. Raise again as it climbs.
export STOCKFISH_EVERY_ITER=1
export STOCKFISH_ELO=2100
export STOCKFISH_NODES=0
ENV
export EXTRA_ENV="$OVR"

# FD limit — 120 self-play / 64 eval workers × shared servers exhaust the default (~1024) and FDs
# accumulate across iterations (the iter-21 "[Errno 24] Too many open files" crash). Raise to max.
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[aws] FD limit=$(ulimit -n)  |  config → $OVR"

# Hand off to the GCP launcher (privilege/apt/stockfish/build/VRAM-detect/launch all reused).
exec bash "$HERE/run_gcp.sh" "$@"
