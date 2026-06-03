#!/bin/bash
# =============================================================================
# Chess AI Training Job — A100 / A40 (USC CARC Discovery)
#
# Submit from:  ~/Chess-AI_ML-System/FINAL/chess_ai/
#   sbatch slurm/train_a100.sbatch
#
# BEFORE FIRST SUBMIT:
#   1. Fill in YOUR_ACCOUNT below (run: sacctmgr show user $USER)
#   2. Upload model:
#        scp game_engine/model/best_model.pth \
#            krishmod@discovery.usc.edu:~/Chess-AI_ML-System/FINAL/chess_ai/game_engine/model/
#   3. Push code:  git push && on CARC: git pull
# =============================================================================

#SBATCH --job-name=chess-ai-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=23:00:00
#SBATCH --account=saifhash_1190
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[a100|a40]"       # prefer A100 (40 GB) or A40 (48 GB)
#SBATCH --output=slurm/chess-ai-train-%j.out
#SBATCH --error=slurm/chess-ai-train-%j.err
#SBATCH --requeue                       # requeue on preemption; main.py auto-resumes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=krish.modi.2000@gmail.com

set -euo pipefail

# =============================================================================
# MODULES  (match the version set used in other CARC projects)
# =============================================================================
module purge
module load ver/2506
module load gcc/14.3.0
module load cuda/12.9.1
module load cmake/3.29.4

# =============================================================================
# DIAGNOSTICS
# =============================================================================
echo "============================================================"
echo "Job:    $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"
echo "CUDA:   $(nvcc --version 2>&1 | grep -oP 'release \K[0-9.]+')"
echo "GCC:    $(gcc --version | head -1)"
echo "CPUs:   $SLURM_CPUS_PER_TASK   Mem: ${SLURM_MEM_PER_NODE:-unset} MB"
echo "Python: $(python3 --version 2>&1)"
echo "Dir:    $(pwd)"
echo "Started: $(date)"
echo "============================================================"

# =============================================================================
# PATHS
# =============================================================================
CHESS_AI_DIR="$HOME/Chess-AI_ML-System/FINAL/chess_ai"
cd "$CHESS_AI_DIR"

# Tee stderr to slurm/ alongside the .out/.err files
exec 2> >(tee -a "slurm/stderr_${SLURM_JOB_ID}.log" >&2)

# =============================================================================
# PYTHON ENVIRONMENT
# Strategy: use conda if the chessai env exists; otherwise use module Python
# + pip install --user.  Conda is more reproducible for PyTorch CUDA builds.
# =============================================================================
if command -v conda &>/dev/null || module load conda 2>/dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^chessai "; then
        conda activate chessai
        echo "Python (conda/chessai): $(python3 --version)"
    else
        echo "chessai conda env not found — creating from requirements.txt"
        conda create -y -n chessai python=3.11
        conda activate chessai
        # Install PyTorch with CUDA 12.x wheel first
        pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
        pip install --quiet -r requirements.txt
    fi
else
    # Fall back: module Python + pip --user
    module load python/3.11.9 2>/dev/null || true
    echo "Python (module): $(python3 --version)"
    pip install --user --quiet --disable-pip-version-check \
        torch --index-url https://download.pytorch.org/whl/cu124
    pip install --user --quiet --disable-pip-version-check -r requirements.txt
fi

# Verify PyTorch sees the GPU
python3 - <<'PYCHECK'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available — check cuda module"
print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} "
      f"| VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
PYCHECK

# =============================================================================
# STOCKFISH
# =============================================================================
if module load stockfish 2>/dev/null && command -v stockfish &>/dev/null; then
    export STOCKFISH_PATH="$(which stockfish)"
elif command -v stockfish &>/dev/null; then
    export STOCKFISH_PATH="$(which stockfish)"
else
    echo "WARNING: stockfish not found — evaluation phase will be skipped"
    export STOCKFISH_PATH="/usr/games/stockfish"
fi
echo "Stockfish: $STOCKFISH_PATH"

# =============================================================================
# MODEL CHECKPOINT CHECK
# =============================================================================
MODEL_PATH="game_engine/model/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: $MODEL_PATH not found."
    echo "Upload it with:"
    echo "  scp game_engine/model/best_model.pth \\"
    echo "      krishmod@discovery.usc.edu:$CHESS_AI_DIR/game_engine/model/"
    exit 1
fi
echo "Model checkpoint: $(du -h $MODEL_PATH | cut -f1)  ($MODEL_PATH)"

# =============================================================================
# BUILD C++ MCTS EXTENSION  (must compile natively on compute node)
# =============================================================================
cd "$CHESS_AI_DIR/game_engine"
rm -rf build && mkdir -p build && cd build

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DPython3_EXECUTABLE="$(which python3)" \
      -Dpybind11_DIR="$PYBIND11_DIR" \
      ..
make -j"$SLURM_CPUS_PER_TASK"
cp mcts_engine_cpp*.so ..
cd "$CHESS_AI_DIR"

python3 -c "
import sys; sys.path.insert(0, 'game_engine')
import mcts_engine_cpp
e = mcts_engine_cpp.MCTSEngine(1, 1)
assert hasattr(e, 'advance_root'), 'advance_root missing'
print('C++ extension OK')
"

# =============================================================================
# HYPERPARAMETERS  — A100 (40 GB) / A40 (48 GB) with 32 CPUs
#
# GPU sizing rationale:
#   A100 40GB VRAM: model 469 MB + 8192 positions × 30 KB each ≈ 2 GB batch
#   → CUDA_BATCH_SIZE=8192 uses ~12% VRAM; safe headroom for activations.
#
# CPU/worker rationale:
#   32 CPUs − 2 (GPU server + main) = 30 MCTS workers.
#   Each worker batches 256 positions for inference (800 sims / 3 iters).
#   30 workers × 256 positions = 7,680 in-flight; GPU processes 8,192 at once.
#   → GPU stays fully loaded; no idle time.
#
# Timing estimate (A100):
#   ~8–12 min/game × 2 games × 30 workers in parallel ≈ 25 min self-play/iter
#   + 10–15 min training + 10 min eval ≈ 45–50 min/iteration
#   23 h job ≈ 27 iterations; 30-day window ≈ 810+ iterations.
# =============================================================================
export CUDA_BATCH_SIZE=8192
export CUDA_STREAMS=8
export NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))
export WORKER_BATCH_SIZE=256
export GAMES_PER_WORKER=2
export SIMULATIONS=800
export EVAL_SIMULATIONS=800
export EVAL_WORKERS=$(( SLURM_CPUS_PER_TASK / 8 ))
export GAMES_PER_EVAL_WORKER=4
export STOCKFISH_GAMES=20
export STOCKFISH_ELO=1320
export MAX_MOVES_PER_GAME=512
export EVAL_MAX_MOVES_PER_GAME=512
export ITERATIONS=1000

echo "------------------------------------------------------------"
echo "Config: workers=$NUM_WORKERS  sims=$SIMULATIONS  gpu_batch=$CUDA_BATCH_SIZE"
echo "        worker_batch=$WORKER_BATCH_SIZE  games/worker=$GAMES_PER_WORKER"
echo "------------------------------------------------------------"

# =============================================================================
# RUN
# =============================================================================
python3 game_engine/main.py

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
