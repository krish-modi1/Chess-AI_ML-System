#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Faithful local replica of the GCP AlphaZero RL run.
#
# Sources game_engine/hyperparams.env.sh for the exact same run-config as
# run_gcp.sh, then overrides only the four knobs that control wall-clock
# cost (not code paths). Everything else — CUDA params, move limits, MCTS
# counts, promotion thresholds — is identical to GCP.
#
# Usage:
#   bash run_local.sh               # foreground (Ctrl-C for graceful stop)
#   bash run_local.sh --background  # nohup; tail -f training_log.txt
# =============================================================================
set -euo pipefail

CHESS_AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME_ENGINE_DIR="$CHESS_AI_DIR/game_engine"
cd "$CHESS_AI_DIR"
mkdir -p logs

BACKGROUND=false
for arg in "$@"; do
  [[ "$arg" == "--background" ]] && BACKGROUND=true
done

echo "============================================================"
echo " Chess AI — Local Training (GCP replica)"
echo " Directory : $CHESS_AI_DIR"
echo " Date/Time : $(date)"
echo "============================================================"

# ── Python ─────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-/home/krish/miniconda3/envs/chessai/bin/python3}"
if ! "$PYTHON" --version &>/dev/null; then
  echo "ERROR: Python not found at $PYTHON" >&2; exit 1
fi
echo "Python     : $("$PYTHON" --version 2>&1)"

# ── GPU / VRAM ─────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found — GPU required" >&2; exit 1
fi
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA not available in PyTorch" >&2; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$(( VRAM_MIB / 1024 ))
echo "GPU        : $GPU_NAME (${VRAM_GB} GB / ${VRAM_MIB} MiB VRAM)"

# ── CPU ────────────────────────────────────────────────────────────────────────
NCPU=$(nproc)
PHYS_CORES=$(lscpu | awk '/^Core\(s\) per socket:/{c=$NF} /^Socket\(s\):/{s=$NF} END{print c*s}' 2>/dev/null || echo "$NCPU")
echo "CPU        : $NCPU logical / $PHYS_CORES physical cores"

# ── RAM ────────────────────────────────────────────────────────────────────────
RAM_GB=$(awk '/^MemTotal:/{print int($2/1024/1024)}' /proc/meminfo)
echo "RAM        : ${RAM_GB} GB"

# ── Stockfish ─────────────────────────────────────────────────────────────────
export STOCKFISH_PATH=$(which stockfish 2>/dev/null || echo "/usr/games/stockfish")
if [[ ! -x "$STOCKFISH_PATH" ]]; then
  echo "ERROR: stockfish not found — install: sudo apt-get install stockfish" >&2; exit 1
fi
echo "Stockfish  : $STOCKFISH_PATH"

# ── Model checkpoint ──────────────────────────────────────────────────────────
MODEL_PATH="$GAME_ENGINE_DIR/model/best_model.pth"
if [[ -f "$MODEL_PATH" ]]; then
  echo "Model      : best_model.pth ($(du -h "$MODEL_PATH" | cut -f1))"
else
  echo "WARNING: best_model.pth not found — will initialize from random weights"
fi

# ── Build C++ extension (unconditional clean rebuild, matching run_gcp.sh) ────
echo ""
echo "Building C++ MCTS extension (clean rebuild for GCP parity)..."
PYBIND11_CMAKE_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")
cd "$GAME_ENGINE_DIR"
rm -rf build && mkdir build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  ..
make -j"$NCPU"
cp mcts_engine_cpp*.so "$GAME_ENGINE_DIR/"
cd "$CHESS_AI_DIR"
echo "C++ extension built."

# ── Verify extension loads ─────────────────────────────────────────────────────
"$PYTHON" - <<'PYCHECK'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "game_engine"))
import mcts_engine_cpp
print(f"mcts_engine_cpp OK")
PYCHECK

# ── Shared run-config (identical to GCP) ──────────────────────────────────────
# NCPU and VRAM_MIB must be set before sourcing.
source "$GAME_ENGINE_DIR/hyperparams.env.sh"

# ── Runtime-cost overrides (ONLY these differ from GCP) ───────────────────────
# These change duration, not which code paths run.
export ITERATIONS=1                # one iteration: self-play → train (no eval)
export GAMES_PER_WORKER=1          # 12 workers × 1 game = 12 self-play games (run in parallel)
export TRAIN_EPOCHS=4              # train for 4 epochs on the generated data
export SIMULATIONS=800             # match GCP
export EVAL_SIMULATIONS=800
export SKIP_EVAL=1                 # skip Phase 3 (arena + Stockfish eval)

# ── Local CPU/GPU tuning (this laptop: 16 logical cores, 6 GB GPU, 14 GB RAM) ──
# WORKER_BATCH_SIZE = per-search leaf batch. KEEP SMALL: num_iterations = SIMULATIONS /
# WORKER_BATCH_SIZE. batch=8 → 100 iterations (genuine MCTS refinement, matches GCP);
# a large batch collapses the search to ~1 iteration (target ≈ prior, no learning signal).
# Oversubscribe (16 logical cores): workers mostly block on the 100 inference round-trips/move,
# so 12 workers keep the CPU busy and pool more leaves per inference (12×8 = 96/round).
export NUM_WORKERS=12
export WORKER_BATCH_SIZE=8
export CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))  # 96
# CRITICAL for small GPUs: TRAIN_BATCH_SIZE default (2048) is sized for the 22 GB L4 and
# OOMs on 6 GB. Probed locally: batch=256 → 3.2 GB peak with AMP.
export TRAIN_BATCH_SIZE=256

echo ""
echo "============================================================"
echo " Run configuration (GCP replica)"
echo "============================================================"
echo "  GPU              : $GPU_NAME (${VRAM_GB} GB)"
echo "  NUM_WORKERS      : $NUM_WORKERS"
echo "  WORKER_BATCH_SIZE: $WORKER_BATCH_SIZE"
echo "  CUDA_BATCH_SIZE  : $CUDA_BATCH_SIZE"
echo "  CUDA_STREAMS     : $CUDA_STREAMS"
echo "  CUDA_TIMEOUT     : ${CUDA_TIMEOUT_INFERENCE}s"
echo "  SIMULATIONS      : $SIMULATIONS   (matches GCP)"
echo "  GAMES_PER_WORKER : $GAMES_PER_WORKER   → $(( NUM_WORKERS * GAMES_PER_WORKER )) games total"
echo "  TRAIN_EPOCHS     : $TRAIN_EPOCHS"
echo "  TRAIN_BATCH_SIZE : $TRAIN_BATCH_SIZE   ← local override (GCP: 2048)"
echo "  MAX_MOVES        : $MAX_MOVES_PER_GAME"
echo "  ITERATIONS       : $ITERATIONS   ← local override (GCP: 1000)"
echo "  SKIP_EVAL        : $SKIP_EVAL   (1 = skip arena + Stockfish)"
echo "  Log              : $CHESS_AI_DIR/training_log.txt"
echo "============================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
if $BACKGROUND; then
  echo "Running in background (nohup). PID → logs/training.pid"
  nohup "$PYTHON" game_engine/main.py &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "$CHESS_AI_DIR/logs/training.pid"
  echo "Training PID : $TRAIN_PID"
  echo "Follow logs  : tail -f $CHESS_AI_DIR/training_log.txt"
else
  "$PYTHON" game_engine/main.py
fi
