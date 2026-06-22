#!/bin/bash
# ==============================================================================
# carc_eval.slurm — ONE-OFF round-robin Elo measurement on CARC (NOT the training loop).
# Rates model checkpoints + the A_low anchor (anchors.json) on one BayesElo scale.
#
#   First time:  mkdir -p chess_ai/logs
#   Run:         sbatch chess_ai/carc_eval.slurm        (from repo root)
#
# Checkpoints are auto-synced from Google Drive at setup (step 0b) into
#   chess_ai/game_engine/model/ — set GDRIVE_MODELS_FOLDER + PLAYERS to match the
#   filenames in that shared Drive folder. No manual scp needed.
# Pure measurement: does NOT touch best_model / metrics.json / the loop.
# ==============================================================================

#SBATCH --job-name=chess-rr
#SBATCH --account=saifhash_1190
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=chess_ai/logs/rr_%x_%j.out
#SBATCH --error=chess_ai/logs/rr_%x_%j.err

set -euo pipefail

# ---- EDIT players here (name:path, comma-separated; paths relative to repo root) ----
# Names must match files in the Drive folder (step 0b): best_model.pth, candidate.pth,
# pretrained.pth, pretrained_anchor.pth, swa_model.pth. Default rates the current run:
# the iter-9 champion vs the rejected iter-10 candidate vs the pretrained base.
PLAYERS="pretrained:game_engine/model/pretrained.pth,champion:game_engine/model/best_model.pth,candidate:game_engine/model/candidate.pth"
GAMES=184        # per pairing (= 92 workers × 2 games each)
RR_WORKERS=92    # SERVER-MODE: workers are CPU procs routing to shared GPU model-servers (2 copies
                 # max on the GPU, like the training loop) → 92 is cheap, no OOM. Server batch auto =
                 # workers × WORKER_BATCH_SIZE.
SIMS=800

# ---------------------------------------------------------------------------
# 0. Paths + modules + env (identical setup to carc_train.slurm; CARC is batch-only)
# ---------------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; CHESS_AI="$REPO/chess_ai"; GAME="$CHESS_AI/game_engine"
mkdir -p "$CHESS_AI/logs"; cd "$REPO"
echo "=== CARC round-robin Elo | $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module purge
module load gcc/13.3.0
module load python/3.11.9
module load cmake/3.29.4
export PATH="$HOME/.local/bin:$PATH"

# deps (pip --user, no-op once present)
if ! python3 -c "import torch, chess, numpy, pybind11, tqdm, psutil" 2>/dev/null; then
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user -r chess_ai/requirements.txt
fi

# Stockfish from source (glibc 2.28 → can't use prebuilt; see carc_train.slurm)
if [[ ! -x "$HOME/.local/bin/stockfish" ]]; then
  mkdir -p "$HOME/.local/bin"; SFB="$(mktemp -d -p "$HOME")"
  curl -fSL --retry 3 -o "$SFB/sf.tgz" "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_16.1.tar.gz"
  tar xzf "$SFB/sf.tgz" -C "$SFB"
  ( cd "$SFB"/Stockfish-*/src && make -j"$(nproc)" build ARCH=x86-64-sse41-popcnt )
  cp "$SFB"/Stockfish-*/src/stockfish "$HOME/.local/bin/stockfish"; chmod +x "$HOME/.local/bin/stockfish"; rm -rf "$SFB"
fi
export STOCKFISH_PATH="$(command -v stockfish || echo "$HOME/.local/bin/stockfish")"

# BayesElo rebuild (committed binary needs glibc 2.34; rebuild in place)
if ! echo quit | "$CHESS_AI/BayesElo/bayeselo" >/dev/null 2>&1; then
  ( cd "$CHESS_AI/BayesElo" && g++ -O3 -Wall -o bayeselo bayeselo.cpp )
fi

# C++ MCTS extension
if ! ls "$GAME"/mcts_engine_cpp*.so >/dev/null 2>&1; then
  PB=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
  rm -rf "$GAME/build" && mkdir -p "$GAME/build"
  ( cd "$GAME/build" && cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(which python3)" -Dpybind11_DIR="$PB" .. && make -j"$(nproc)" )
  cp "$GAME"/build/mcts_engine_cpp*.so "$GAME/"
fi

# ---------------------------------------------------------------------------
# 0b. Sync the latest model checkpoints from Google Drive (so CARC rates the
#     current run's nets without a manual scp). The Drive folder must be shared
#     ("anyone with the link") and contain the .pth/.bak files named in PLAYERS.
# ---------------------------------------------------------------------------
GDRIVE_MODELS_FOLDER="https://drive.google.com/drive/folders/1J_cfNiIusiVMhUjTZ0rI80gpKo3XEu_u"
python3 -c "import gdown" 2>/dev/null || python3 -m pip install --user --quiet gdown
mkdir -p "$GAME/model"
STAGE="$(mktemp -d -p "$HOME")"
echo "[gdown] syncing checkpoints from $GDRIVE_MODELS_FOLDER → $GAME/model"
gdown --folder "$GDRIVE_MODELS_FOLDER" -O "$STAGE"
# find-copy handles gdown's possible subfolder nesting; overwrites stale copies, never deletes.
find "$STAGE" -type f \( -name '*.pth' -o -name '*.bak' \) -exec cp -fv {} "$GAME/model/" \;
rm -rf "$STAGE"
echo "[gdown] model dir now holds:"; ls -la "$GAME/model"/*.pth "$GAME/model"/*.bak 2>/dev/null \
  || echo "[gdown] WARNING: no .pth/.bak fetched — check the folder is shared + PLAYERS names match"

# VERIFY (quit to .err on failure) — deps + cuda + stockfish + .so + bayeselo + checkpoint files
python3 - "$STOCKFISH_PATH" "$CHESS_AI/BayesElo/bayeselo" "$PLAYERS" <<'PY' || { echo "[verify] FAILED — aborting." >&2; exit 1; }
import importlib, subprocess, sys, os
fail = []
for m in ("torch","chess","numpy","pybind11"):
    try: importlib.import_module(m)
    except Exception as e: fail.append(f"import {m}: {e}")
import torch
if not torch.cuda.is_available(): fail.append("cuda not available")
sf, be, players = sys.argv[1], sys.argv[2], sys.argv[3]
if subprocess.run([sf], input="quit\n", capture_output=True, text=True, timeout=15).returncode != 0: fail.append("stockfish broken")
if subprocess.run([be], input="x\n", capture_output=True, text=True, timeout=15).returncode != 0: fail.append("bayeselo broken")
for tok in players.split(","):
    p = tok.split(":",1)[1]; p = p if p.startswith("chess_ai") else os.path.join("chess_ai", p)
    if not os.path.exists(p): fail.append(f"missing checkpoint: {p}")
if fail: sys.stderr.write("VERIFY FAILED:\n  - " + "\n  - ".join(fail) + "\n"); sys.exit(1)
print("[verify] OK")
PY

export PYTHONPATH="$GAME${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Raise the open-file limit — 92 workers × 2 servers/pairing exhaust the default (~1024) and FDs
# accumulate across pairings (same failure that killed the iter-21 arena: "Too many open files").
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[fd] open-files limit = $(ulimit -n)"

# ---------------------------------------------------------------------------
# Run the round-robin
# ---------------------------------------------------------------------------
OUT="$CHESS_AI/../local/rr_${SLURM_JOB_ID:-local}"
echo "[run] round-robin: $PLAYERS | $GAMES games/pairing | $SIMS sims | $RR_WORKERS workers"
python3 chess_ai/round_robin_elo.py \
  --players "$PLAYERS" \
  --anchors-json chess_ai/anchors.json \
  --stockfish "$STOCKFISH_PATH" \
  --bayeselo "$CHESS_AI/BayesElo/bayeselo" \
  --games "$GAMES" --workers "$RR_WORKERS" --sims "$SIMS" \
  --out-dir "$OUT"
echo "[run] done at $(date) → ladder in $OUT/ladder.json"
