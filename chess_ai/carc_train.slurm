#!/bin/bash
# ==============================================================================
# carc_train.slurm — ONE self-contained SLURM job for USC CARC.
# Installs deps → verifies (quits to .err on failure) → builds the C++ ext →
# runs EXACTLY 3 AlphaZero iterations, with GCP-identical outputs.
#
#   First time:   mkdir -p chess_ai/logs
#   Every batch:  sbatch chess_ai/carc_train.slurm     (run from the repo root)
#
# Why everything is in the job: CARC is batch-only — login-node setup does NOT
# carry to the compute node, so install + build + run must all happen here.
# Deps go to ~/.local (shared home → persist across jobs; pip is a no-op once present).
# Outputs land exactly where GCP put them: chess_ai/training_log.txt,
# chess_ai/game_engine/model/{best_model,candidate,metrics.json}, data/self_play/.
# ==============================================================================

#SBATCH --job-name=chess-az
#SBATCH --account=saifhash_1190         # CARC project allocation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1               # or gpu:1 / gpu:a40:1  (check: sinfo -o "%P %G")
#SBATCH --cpus-per-task=16              # self-play CPU workers scale to this
#SBATCH --mem=64G
#SBATCH --time=24:00:00                 # generous for 3 iters; raise to partition max if needed
#SBATCH --signal=B:TERM@300             # SIGTERM 300s before time limit → graceful save
#SBATCH --output=chess_ai/logs/carc_%x_%j.out
#SBATCH --error=chess_ai/logs/carc_%x_%j.err

set -euo pipefail
ITERS_PER_JOB=3

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
CHESS_AI="$REPO/chess_ai"
GAME="$CHESS_AI/game_engine"
mkdir -p "$CHESS_AI/logs"
cd "$REPO"

echo "============================================================"
echo " CARC AlphaZero job  |  $(date)"
echo " Node: $(hostname)   |  Job: ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Modules  (verify exact strings once with: module avail gcc python cmake)
# ---------------------------------------------------------------------------
module purge
module load gcc/11.3.0    || module load gcc
module load python/3.11.3 || module load python
module load cmake         || true
export PATH="$HOME/.local/bin:$PATH"      # pip --user scripts + stockfish live here

# ---------------------------------------------------------------------------
# 2. Python deps → ~/.local (only if missing; torch's cu126 wheel bundles CUDA)
# ---------------------------------------------------------------------------
if ! python3 -c "import torch, chess, numpy, pybind11, tqdm, psutil" 2>/dev/null; then
  echo "[deps] installing into ~/.local (first run / missing packages)..."
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user -r chess_ai/requirements.txt
else
  echo "[deps] python packages already present — skipping pip."
fi

# ---------------------------------------------------------------------------
# 3. Stockfish → ~/.local/bin (official prebuilt static binary; no apt/conda)
# ---------------------------------------------------------------------------
if ! command -v stockfish >/dev/null 2>&1; then
  echo "[sf] downloading Stockfish ..."
  mkdir -p "$HOME/.local/bin"
  TMP="$(mktemp -d)"
  for variant in avx2 sse41-popcnt; do
    url="https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-${variant}.tar"
    if curl -fSL --retry 3 -o "$TMP/sf.tar" "$url"; then
      tar xf "$TMP/sf.tar" -C "$TMP"
      bin="$(find "$TMP" -type f -name 'stockfish-ubuntu-*' | head -1)"
      chmod +x "$bin"
      if echo quit | "$bin" >/dev/null 2>&1; then
        cp "$bin" "$HOME/.local/bin/stockfish"; chmod +x "$HOME/.local/bin/stockfish"
        echo "[sf] installed ($variant)"; break
      fi
    fi
  done
  rm -rf "$TMP"
fi
export STOCKFISH_PATH="$(command -v stockfish || echo "$HOME/.local/bin/stockfish")"

# ---------------------------------------------------------------------------
# 4. VERIFY environment — on ANY failure, write to .err and QUIT before the run
# ---------------------------------------------------------------------------
echo "[verify] checking environment ..."
python3 - "$STOCKFISH_PATH" <<'PY' || { echo "[verify] FAILED — see errors above; aborting job." >&2; exit 1; }
import importlib, subprocess, sys, os
fail = []
for m in ("torch", "chess", "numpy", "pybind11", "tqdm", "psutil"):
    try:
        importlib.import_module(m)
    except Exception as e:
        fail.append(f"import {m}: {e}")
try:
    import torch
    print(f"[verify] torch {torch.__version__}  cuda_build={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        fail.append("torch.cuda.is_available() is False on a GPU node (driver/wheel mismatch)")
except Exception as e:
    fail.append(f"torch/cuda: {e}")
sf = sys.argv[1]
try:
    r = subprocess.run([sf], input="quit\n", capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        fail.append(f"stockfish exited {r.returncode}")
    else:
        print(f"[verify] stockfish OK ({sf})")
except Exception as e:
    fail.append(f"stockfish run: {e}")
if fail:
    sys.stderr.write("VERIFY FAILED:\n  - " + "\n  - ".join(fail) + "\n")
    sys.exit(1)
print("[verify] python deps + cuda + stockfish OK")
PY

# ---------------------------------------------------------------------------
# 5. Build the C++ MCTS extension if missing (in-job = correct python ABI)
# ---------------------------------------------------------------------------
if ! ls "$GAME"/mcts_engine_cpp*.so >/dev/null 2>&1 || [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  echo "[build] compiling mcts_engine_cpp ..."
  PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
  rm -rf "$GAME/build" && mkdir -p "$GAME/build"
  ( cd "$GAME/build" \
    && cmake -DCMAKE_BUILD_TYPE=Release \
             -DPython3_EXECUTABLE="$(which python3)" \
             -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" .. \
    && make -j"$(nproc)" )
  cp "$GAME"/build/mcts_engine_cpp*.so "$GAME/"
  echo "[build] done."
else
  echo "[build] mcts_engine_cpp.so present — skipping (FORCE_REBUILD=1 to force)."
fi
python3 -c "import sys; sys.path.insert(0,'$GAME'); import mcts_engine_cpp" \
  || { echo "[verify] mcts_engine_cpp failed to import — aborting job." >&2; exit 1; }
echo "[verify] mcts_engine_cpp import OK"

# ---------------------------------------------------------------------------
# 6. Hyperparameters + cap this job to 3 fresh iterations (from run_state.json)
# ---------------------------------------------------------------------------
set -a; source "$GAME/hyperparams.env.sh"; set +a
export PYTHONPATH="$GAME${PYTHONPATH:+:$PYTHONPATH}"

START_ITER=$(python3 - "$CHESS_AI" <<'PY'
import json, os, sys
data = os.path.join(sys.argv[1], "data", "self_play")
rs   = os.path.join(data, "run_state.json")
phases = ("self_play", "training", "eval")
if os.path.exists(rs):
    st = json.load(open(rs)); it = int(st["iteration"]); done = phases.index(st["phase"]) + 1
    start = it + 1 if done >= len(phases) else it
else:
    its = [int(d.split("_")[1]) for d in os.listdir(data)
           if d.startswith("iter_") and os.path.isdir(os.path.join(data, d))]
    start = (max(its) + 1) if its else 1
print(start)
PY
)
export ITERATIONS=$(( START_ITER + ITERS_PER_JOB - 1 ))
echo "[run] resuming at iter $START_ITER → through iter $ITERATIONS ($ITERS_PER_JOB iters this job)"

# ---------------------------------------------------------------------------
# 7. Run the loop EXACTLY as GCP did (cwd=chess_ai → Logger writes
#    chess_ai/training_log.txt; model/ + data/self_play/ updated in place).
#    Forward SLURM's SIGTERM to python for graceful phase-resume.
# ---------------------------------------------------------------------------
cd "$CHESS_AI"
python3 -m game_engine.main &
PID=$!
graceful() { echo "[run] SIGTERM (time limit) → graceful stop"; kill -TERM "$PID" 2>/dev/null || true; }
trap graceful TERM
set +e
wait "$PID"
while kill -0 "$PID" 2>/dev/null; do sleep 5; done
echo "[run] python finished at $(date)"
