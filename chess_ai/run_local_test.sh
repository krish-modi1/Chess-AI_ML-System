#!/usr/bin/env bash
# =============================================================================
# run_local_test.sh — Short FULL-PIPELINE local test (GCP-replica code paths).
#
# Runs ONE iteration end-to-end: self-play → train → eval (arena + Stockfish),
# at the small "4 workers / 1 game / 100 sims" config so it finishes quickly.
# Unlike run_local.sh (which skips eval), this exercises the 2-server arena and
# the GPU-server Stockfish eval — the pieces being verified before GCP.
#
#   NO_PROMOTE=1  → dry run: best_model.pth is NEVER replaced, and the Stockfish
#                   phase still runs (it evaluates BEST without mutating it).
#
# Usage:  bash run_local_test.sh
# =============================================================================
set -euo pipefail

CHESS_AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME_ENGINE_DIR="$CHESS_AI_DIR/game_engine"
cd "$CHESS_AI_DIR"

echo "============================================================"
echo " Chess AI — Local FULL-PIPELINE test (self-play→train→eval)"
echo " Date/Time : $(date)"
echo "============================================================"

PYTHON="${PYTHON:-/home/krish/miniconda3/envs/chessai/bin/python3}"
"$PYTHON" --version >/dev/null 2>&1 || { echo "ERROR: Python not found at $PYTHON" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
"$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || { echo "ERROR: CUDA not available in PyTorch" >&2; exit 1; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
NCPU=$(nproc)

export STOCKFISH_PATH=$(which stockfish 2>/dev/null || echo "/usr/games/stockfish")
[[ -x "$STOCKFISH_PATH" ]] || { echo "ERROR: stockfish not found" >&2; exit 1; }

# ── C++ extension: rebuild only if missing or stale vs sources. This is a
#    pipeline-logic test; run_local.sh already covers the clean-rebuild GCP parity. ─
SO=$(ls "$GAME_ENGINE_DIR"/mcts_engine_cpp*.so 2>/dev/null | head -1)
STALE=""
[[ -n "$SO" ]] && STALE=$(find "$GAME_ENGINE_DIR/src" -type f \( -name '*.cpp' -o -name '*.h' \) -newer "$SO" 2>/dev/null | head -1)
if [[ -z "$SO" || -n "$STALE" ]]; then
  echo "Building C++ MCTS extension..."
  PYBIND11_CMAKE_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
  cd "$GAME_ENGINE_DIR"
  rm -rf build && mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON" \
        ${PYBIND11_CMAKE_DIR:+-Dpybind11_DIR="$PYBIND11_CMAKE_DIR"} .. >/dev/null
  make -j"$NCPU" >/dev/null
  cp mcts_engine_cpp*.so "$GAME_ENGINE_DIR/"
  cd "$CHESS_AI_DIR"
  echo "C++ extension built."
else
  echo "C++ extension up to date — skipping rebuild ($(basename "$SO"))"
fi
"$PYTHON" -c "import sys; sys.path.insert(0,'$GAME_ENGINE_DIR'); import mcts_engine_cpp" \
  && echo "mcts_engine_cpp OK" \
  || { echo "ERROR: C++ extension failed to import" >&2; exit 1; }

# ── Shared run-config (identical to GCP), then test overrides ─────────────────
source "$GAME_ENGINE_DIR/hyperparams.env.sh"

# ── Self-play / train (short) ─────────────────────────────────────────────────
export ITERATIONS=1
export NUM_WORKERS=4
export GAMES_PER_WORKER=1
export WORKER_BATCH_SIZE=8
export SIMULATIONS=100
export CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))   # 32
export CUDA_STREAMS=2
export SHM_TRANSPORT=1
export TRAIN_EPOCHS=4
export TRAIN_BATCH_SIZE=256
export TRAIN_DL_WORKERS=4           # GCP uses 16; local data is tiny
export TRAIN_DL_PREFETCH=1          # GCP uses 4; local RAM is tight

# ── Eval: run the FULL pipeline (the point of this test) ──────────────────────
export SKIP_EVAL=0
export NO_PROMOTE=1                 # dry run — never replace best_model.pth
export EVAL_WORKERS=4
export GAMES_PER_EVAL_WORKER=1      # 4 arena games (candidate vs champion)
export EVAL_SIMULATIONS=100
export STOCKFISH_GAMES=4            # 4 stockfish games (BEST model on GPU server)
export STOCKFISH_ELO=1800
export EVAL_MAX_MOVES_PER_GAME=60   # short games for the test

echo ""
echo "============================================================"
echo "  NUM_WORKERS=$NUM_WORKERS  SIMULATIONS=$SIMULATIONS  WORKER_BATCH_SIZE=$WORKER_BATCH_SIZE"
echo "  CUDA_BATCH_SIZE=$CUDA_BATCH_SIZE  CUDA_STREAMS=$CUDA_STREAMS  SHM_TRANSPORT=$SHM_TRANSPORT"
echo "  EVAL_WORKERS=$EVAL_WORKERS x GAMES_PER_EVAL_WORKER=$GAMES_PER_EVAL_WORKER (arena)"
echo "  STOCKFISH_GAMES=$STOCKFISH_GAMES vs Elo $STOCKFISH_ELO"
echo "  NO_PROMOTE=$NO_PROMOTE  SKIP_EVAL=$SKIP_EVAL  EVAL_MAX_MOVES=$EVAL_MAX_MOVES_PER_GAME"
echo "============================================================"
echo ""

"$PYTHON" game_engine/main.py
