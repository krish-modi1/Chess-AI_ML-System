#!/usr/bin/env bash
# run_aws_selfplay.sh — AWS SELF-PLAY FARM (no training, no eval).
#
# Burns the AWS credit generating a bank of champion self-play games, then gathers them into
# data/self_play/iter_aws/ for you to push. NO code changes needed: it runs the stock self-play
# loop (which writes normal numbered iter_<N> dirs) with training+eval disabled, and the
# `consolidate` step moves everything this farm produced into the single iter_aws/ dir.
#
# Recommended box: g6e.8xlarge (L40S 48GB, 32-core Zen3, 256GB) — ~4090-class self-play throughput.
#
#   Provision a fresh box, then from the repo root:
#     bash chess_ai/game_engine/run_aws_selfplay.sh --background          # 1. start the farm
#     # ...let it run on the credit; watch nvidia-smi sm% + `free -g`...
#     kill -TERM "$(cat chess_ai/game_engine/logs/training.pid)"          # 2. stop when done
#     bash chess_ai/game_engine/run_aws_selfplay.sh consolidate           # 3. gather games → iter_aws/
#     # 4. (you handle git) commit + push ONLY data/self_play/iter_aws/
#
# ⚠️ The stock trainer can't read a non-numeric dir like iter_aws yet — that integration is a
#    SEPARATE change (deferred). Do NOT pull iter_aws onto the Vast box until that lands, or Vast's
#    trainer will hit ValueError on int("aws") and skip training. Keep iter_aws off Vast's branch
#    until the consume-side change is in.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA_DIR="$REPO/chess_ai/data/self_play"      # = main.py DATA_DIR (cwd is chess_ai/ at runtime)
BANK="$DATA_DIR/iter_aws"
HOLD="$REPO/chess_ai/data/.iter_aws_hold"     # outside self_play → never matched by iter_* globs
BASELINE="$HERE/.selfplay_baseline"

# ---------------------------------------------------------------------------
# consolidate: move this farm's games (post-baseline numbered iters + any held bank) → iter_aws/
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "consolidate" ]]; then
  mkdir -p "$BANK"
  # restore a previously-held bank first
  if [[ -d "$HOLD" ]]; then
    find "$HOLD" -name '*.npz' -exec mv -t "$BANK/" {} + 2>/dev/null || true
    rmdir "$HOLD" 2>/dev/null || true
  fi
  moved=0
  for d in "$DATA_DIR"/iter_*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"; num="${name#iter_}"
    [[ "$num" =~ ^[0-9]+$ ]] || continue                       # skip iter_aws + any named dir
    [[ -f "$BASELINE" ]] && grep -qx "$name" "$BASELINE" && continue   # skip pre-existing (pulled) iters
    before=$(ls "$d"*.npz 2>/dev/null | wc -l)
    find "$d" -name '*.npz' -exec mv -t "$BANK/" {} + 2>/dev/null || true
    rmdir "$d" 2>/dev/null || true
    moved=$((moved + before))
  done
  echo "[farm] consolidated $moved game files → $BANK"
  echo "[farm] iter_aws now holds $(ls "$BANK"/*.npz 2>/dev/null | wc -l) npz files"
  exit 0
fi

# ---------------------------------------------------------------------------
# farm mode: hold any existing bank aside, snapshot pre-existing iters, run self-play only
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"
# A pre-existing iter_aws (e.g. cloned in) would crash the stock get_start_iteration on int("aws").
# Hold it aside; consolidate restores it. No-op on a clean box.
if [[ -d "$BANK" ]]; then
  mkdir -p "$HOLD"
  find "$BANK" -name '*.npz' -exec mv -t "$HOLD/" {} + 2>/dev/null || true
  rmdir "$BANK" 2>/dev/null || true
  echo "[farm] held existing bank aside ($(ls "$HOLD"/*.npz 2>/dev/null | wc -l) files) → $HOLD"
fi
# Snapshot the numbered iters present now, so consolidate only grabs what THIS farm produces.
ls -d "$DATA_DIR"/iter_*/ 2>/dev/null | xargs -r -n1 basename > "$BASELINE" || : > "$BASELINE"
echo "[farm] baseline iters recorded: $(wc -l < "$BASELINE" 2>/dev/null || echo 0) → $BASELINE"

OVR="$HERE/.aws_selfplay.env.sh"
cat > "$OVR" <<'ENV'
# Self-play FARM tuning for g6e.8xlarge (L40S 48GB, 32-core Zen3, 256GB RAM). Self-play ONLY.
export MIN_TRAIN_ITERS=9999          # > ITERATIONS → every iteration is self-play only (no train/eval)
# 120 workers on 30 cores (2 reserved) = ~4x oversubscription — fine for latency-bound self-play.
# RAM IS THE HARD CAP here, not the GPU: measured ~0.77 GB/worker steady-state (Vast: 184w → ~142GB).
#   120w ≈ 93GB peak → safe on a 128GB box (~35GB margin, NO swap).
#   CEILING on 128GB ≈ 140 workers; do NOT chase the GPU toward ~280 here — that needs ~216GB
#   (a 256GB box). Watch `free -g` and keep peak under ~110GB. CUDA_BATCH auto-tracks NUM_WORKERS.
export NUM_WORKERS=120
export GAMES_PER_WORKER=20           # games each worker plays before the iteration rolls over
export RESERVED_CORES=2
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE
export CUDA_TIMEOUT_INFERENCE=0.02   # latency-bound: shorter gather timeout = more leaves/sec
export TEMP_MOVES=16                 # wide opening book (matches the main run)
# Loose pacing (lead 20 = no real cap at 20 games/worker): fast workers never wait on stragglers,
# maximizing raw game throughput — the farm doesn't need balanced iteration completion.
export MAX_WORKER_LEAD=20
ENV
export EXTRA_ENV="$OVR"

# Seed the champion from Drive — the farm self-plays with best_model.pth (guarded: only if missing).
MODEL_DIR="$HERE/model"
if [[ ! -f "$MODEL_DIR/best_model.pth" || "${SYNC_MODELS:-0}" == "1" ]]; then
  echo "[farm] seeding champion from Drive ..."
  python3 -c "import gdown" 2>/dev/null || python3 -m pip install --user --quiet gdown
  STAGE="$(mktemp -d)"
  if gdown --folder "https://drive.google.com/drive/folders/1J_cfNiIusiVMhUjTZ0rI80gpKo3XEu_u" -O "$STAGE"; then
    mkdir -p "$MODEL_DIR"
    find "$STAGE" -type f \( -name '*.pth' -o -name '*.bak' \) -exec cp -fv {} "$MODEL_DIR/" \;
  else
    echo "[farm] WARNING: gdown failed — scp best_model.pth into $MODEL_DIR manually." >&2
  fi
  rm -rf "$STAGE"
else
  echo "[farm] champion present — skipping Drive seed."
fi

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[farm] FD limit=$(ulimit -n) | SELF-PLAY ONLY (MIN_TRAIN_ITERS=9999) | config → $OVR"

# Hand off to the GCP launcher (apt/stockfish/build/VRAM-detect/launch reused). Self-play only,
# so the build still happens but training/eval never run.
exec bash "$HERE/run_gcp.sh" "$@"
