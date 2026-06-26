#!/usr/bin/env bash
# run_aws_selfplay.sh — AWS SELF-PLAY FARM (no training, no eval).
#
# Generates a bank of champion self-play games at the hyperparams sim config (currently 2000/200),
# writing them DIRECTLY into a single NUMERIC dir data/self_play/iter_<BANK_ITER>/ (via SELFPLAY_FARM=1
# in main.py) so the Vast trainer picks it up AUTOMATICALLY on the next git pull — NO consumer code
# change needed (numeric iter dirs are handled natively; it's the highest-numbered dir so always in window).
#
# BANK_ITER must be HIGHER than any iteration the Vast run will reach during the experiment, else Vast
# eventually writes its own self-play into the same dir (collision). Vast is ~iter-28 at ~6/day, so
# 900 is safe (it never reaches it). Override with `BANK_ITER=<n> bash ...` if needed.
#
# Recommended box: g6e.8xlarge (L40S 48GB, 32-core Zen3, 256GB).
#
#   Provision a fresh box, then from the repo root:
#     bash chess_ai/game_engine/run_aws_selfplay.sh --background       # 1. start farm (writes iter_900 DIRECTLY)
#     kill -TERM "$(pgrep -f game_engine/main.py | head -1)"           # 2. stop when done
#     bash chess_ai/game_engine/run_aws_selfplay.sh consolidate        # 3. OPTIONAL safety sweep (pre-flag stray games)
#     # 4. (you handle git) commit + push ONLY data/self_play/iter_900/
#
# Vast picks it up on the next pull. NOTE: as the highest-numbered dir, iter_900 loads FIRST under the
# MAX_TRAIN_POSITIONS cap, so a large bank dominates the training window — GOOD for the teacher test
# (un-dilutes the 2000-sim signal vs the 50-iter window), but remove iter_900 once the test concludes
# if you don't want the fixed-champion bank permanently weighting training.
# Vast keeps its OWN iteration counter via run_state.json (get_start_iteration would say 901, but it's
# ignored while run_state exists) — iter_900 is just training data, never the "current" iteration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA_DIR="$REPO/chess_ai/data/self_play"      # = main.py DATA_DIR (cwd is chess_ai/ at runtime)
BANK_ITER="${BANK_ITER:-900}"                 # numeric bank dir; must exceed Vast's reach
BANK_NAME="iter_$BANK_ITER"
BANK="$DATA_DIR/$BANK_NAME"
BASELINE="$HERE/.selfplay_baseline"

# ---------------------------------------------------------------------------
# consolidate: move this farm's games (post-baseline numbered iters) → iter_<BANK_ITER>/
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "consolidate" ]]; then
  mkdir -p "$BANK"
  # Gather every game THIS farm produced into the bank: every npz NEWER than the baseline marker
  # (written once at first farm start). Per-FILE (not per-dir) so games main.py appended into a
  # PRE-EXISTING dir — e.g. a cloned iter_28 that's listed in the baseline — are still captured, while
  # older pulled games stay put. Time-based, so it's robust to whatever iteration main.py wrote to.
  moved=0
  for d in "$DATA_DIR"/iter_*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"; num="${name#iter_}"
    [[ "$name" == "$BANK_NAME" ]] && continue                          # never process the bank itself
    [[ "$num" =~ ^[0-9]+$ ]] || continue                              # skip any stray named dir
    while IFS= read -r -d '' f; do
      mv -t "$BANK/" "$f" && moved=$((moved + 1))
    done < <(find "$d" -maxdepth 1 -name '*.npz' -newer "$BASELINE" -print0 2>/dev/null)
    rmdir "$d" 2>/dev/null || true   # drop the dir only if the farm emptied it (baseline dirs keep old games)
  done
  echo "[farm] consolidated $moved game files → $BANK"
  echo "[farm] $BANK_NAME now holds $(ls "$BANK"/*.npz 2>/dev/null | wc -l) npz files"
  echo "[farm] push ONLY data/self_play/$BANK_NAME/ ; Vast auto-consumes it (numeric dir)."
  exit 0
fi

# ---------------------------------------------------------------------------
# farm mode: snapshot pre-existing iters, run self-play only
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"
# Snapshot the numbered iters present now (incl. a pre-existing bank), so consolidate only grabs what
# THIS farm produces. A pre-existing iter_<BANK_ITER> is fine: it's numeric so get_start_iteration
# handles it; consolidate skips it as a source and appends new games into it.
ls -d "$DATA_DIR"/iter_*/ 2>/dev/null | xargs -r -n1 basename > "$BASELINE" || : > "$BASELINE"
echo "[farm] baseline iters recorded: $(wc -l < "$BASELINE" 2>/dev/null || echo 0) → $BASELINE"
echo "[farm] bank target = $BANK_NAME (numeric → Vast auto-consumes on pull)"

OVR="$HERE/.aws_selfplay.env.sh"
cat > "$OVR" <<'ENV'
# Self-play FARM tuning for g6e.8xlarge (L40S 48GB, 32-core Zen3, 256GB RAM). Self-play ONLY.
export MIN_TRAIN_ITERS=9999          # > ITERATIONS → every iteration is self-play only (no train/eval)
# 120 workers on 30 cores (2 reserved) = ~4x oversubscription — fine for latency-bound self-play.
# RAM is the cap (~0.77 GB/worker): 120w ≈ 93GB → safe on 128GB, comfortable on the 256GB g6e.
# Sims come from hyperparams (2000 full / 200 fast / 25% recorded) — NOT overridden here, so the bank
# matches the Vast run's teacher-test config. CUDA_BATCH auto-tracks NUM_WORKERS.
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

# Direct-write: main.py dumps every self-play game straight into iter_$BANK_ITER (the bank), so NO
# consolidate/relabel is needed — push data/self_play/$BANK_NAME/ as-is. (The consolidate step above is
# now just a safety sweep for games written BEFORE this flag existed, e.g. a pre-flag iter_28.)
export SELFPLAY_FARM=1
export FARM_BANK_ITER="$BANK_ITER"

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
echo "[farm] FD limit=$(ulimit -n) | SELF-PLAY ONLY (MIN_TRAIN_ITERS=9999) | sims from hyperparams | config → $OVR"

# Hand off to the GCP launcher (apt/stockfish/build/VRAM-detect/launch reused). Self-play only,
# so the build still happens but training/eval never run.
exec bash "$HERE/run_gcp.sh" "$@"
