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
export RESERVED_CORES=4
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
export MAX_WORKER_LEAD=10

# Training: batch 2048 — fits the 24GB card (4096 OOMs at ~22GB here; that was a 48GB-only setting).
# DL workers 90→16: the dataset is loaded fully into RAM, so __getitem__ is pure indexing (no disk I/O)
# and the GPU is the bottleneck — 16 workers keep it fed. 90 was the RAM-balloon culprit: each worker
# copy-on-write touches the numpy/list refcounts + holds prefetch buffers, inflating RSS far above the
# printed f16-array size (it under-counts true process RAM). Fewer workers = much less RAM, no speed loss.
export TRAIN_BATCH_SIZE=2048
export TRAIN_DL_WORKERS=64
export TRAIN_DL_PREFETCH=2
# Train on the last 30 iterations of self-play (was 50). Tightened to age out the old pre-2000-sim data
# faster so the 2000-sim games — recent iters + the iter_900 bank (always in-window as the highest dir) —
# dominate sooner. Per-load RAM is still bounded by the chunk cap (TRAIN_CHUNK_POSITIONS).
export TRAIN_WINDOW=30
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
