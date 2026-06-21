#!/usr/bin/env bash
# run_gcp.sh — GCP GPU setup + training launcher for Chess AI AlphaZero
# Usage: bash run_gcp.sh [--background]
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 0. Pre-flight: sudo check (warn-only — packages may already be installed)
# ──────────────────────────────────────────────────────────────────────────────
# Privilege detection — portable across GCP (sudo), Vast.ai/RunPod Docker (root, no sudo binary), local.
if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  HAS_SUDO=true
  sudo_run() { "$@"; }                          # already root (Docker) — run directly, no sudo
elif sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
  HAS_SUDO=true
  sudo_run() { sudo "$@"; }
else
  HAS_SUDO=false
  sudo_run() { echo "  [skip — no sudo/root] $*"; }
  echo "WARNING: not root and no sudo — skipping apt installs (deps must be pre-installed)."
fi

if ! command -v git &>/dev/null; then
  echo "[pre] git not found — installing..."
  sudo_run apt-get update -qq && sudo_run apt-get install -y git
fi

# ──────────────────────────────────────────────────────────────────────────────
# 1. Resolve script directory so this works from any working directory
# ──────────────────────────────────────────────────────────────────────────────
CHESS_AI_DIR="$(cd "$(dirname "$0")" && pwd)"   # = chess_ai/game_engine (this script lives here)
PARENT_DIR="$(dirname "$CHESS_AI_DIR")"          # = chess_ai/ — canonical home of training_log.txt
cd "$CHESS_AI_DIR"
mkdir -p logs

BACKGROUND=false
for arg in "$@"; do
  [[ "$arg" == "--background" ]] && BACKGROUND=true
done

echo "============================================================"
echo " Chess AI — GCP Training Setup"
echo " Directory : $CHESS_AI_DIR"
echo " Date/Time : $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Check CUDA / GPU
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/9] Checking CUDA / GPU availability..."

if ! command -v nvidia-smi &>/dev/null; then
  if $HAS_SUDO; then
    echo "[0/9] NVIDIA driver not found — installing for Ubuntu..."
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-driver-550
    echo ""
    echo "================================================================"
    echo " Driver installed. A reboot is required to load the kernel module."
    echo " Run:  sudo reboot"
    echo " Then SSH back in and re-run:  bash run_gcp.sh"
    echo "================================================================"
    exit 0
  else
    echo "ERROR: nvidia-smi not found and no sudo to install driver." >&2
    exit 1
  fi
fi

nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$(( VRAM_MIB / 1024 ))

echo ""
echo "  GPU  : $GPU_NAME"
echo "  VRAM : ${VRAM_GB} GB (${VRAM_MIB} MiB)"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Install system dependencies if missing
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/9] Checking system dependencies..."

MISSING_PKGS=()

command -v stockfish      &>/dev/null || MISSING_PKGS+=(stockfish)
command -v cmake          &>/dev/null || MISSING_PKGS+=(cmake)
command -v make           &>/dev/null || MISSING_PKGS+=(build-essential)
command -v tmux           &>/dev/null || MISSING_PKGS+=(tmux)
command -v nvtop          &>/dev/null || MISSING_PKGS+=(nvtop)
command -v htop           &>/dev/null || MISSING_PKGS+=(htop)
command -v git            &>/dev/null || MISSING_PKGS+=(git)
dpkg -s python3-dev &>/dev/null 2>&1  || MISSING_PKGS+=(python3-dev)
dpkg -s python3-pip &>/dev/null 2>&1  || MISSING_PKGS+=(python3-pip)

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo "  Missing packages: ${MISSING_PKGS[*]}"
  if $HAS_SUDO; then
    sudo_run apt-get update -qq
    sudo_run apt-get install -y "${MISSING_PKGS[@]}"
  else
    echo "  WARNING: Cannot install — no sudo/root. Continuing anyway (may fail later)."
  fi
else
  echo "  All system dependencies already installed."
fi

# Ensure pip --user binaries are on PATH
export PATH="$HOME/.local/bin:$PATH"

# ──────────────────────────────────────────────────────────────────────────────
# 3. Stockfish 16 — BUILD FROM SOURCE (pinned, identical to the local SF 16 the anchor was tuned on).
#    The box's apt stockfish is version-dependent: older Ubuntu ships SF 14.1, whose UCI_Elo floor is
#    1350 (not SF 16's 1320). Requesting 1320 on SF 14.1 throws, gets swallowed by main.py's except,
#    and SF runs UNLIMITED (the iter-3 0/128). Building SF 16 makes UCI_Elo=1320 valid + calibration
#    identical to local. ~3-5 min; reuses the cmake/make toolchain already installed for the .so.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
SF_BIN="$HOME/.local/bin/stockfish"
# NB: capture --version into a var and string-match — a piped `grep -q` SIGPIPEs Stockfish, which
# under `set -o pipefail` makes the pipeline non-zero and false-fails the check.
SF_VER="$("$SF_BIN" --version 2>/dev/null | head -1 || true)"
if [[ "$SF_VER" == *"Stockfish 16"* ]]; then
  echo "[3/9] Stockfish 16 already built at $SF_BIN"
else
  echo "[3/9] Building Stockfish 16 from source..."
  rm -rf /tmp/Stockfish
  git clone --depth 1 --branch sf_16 https://github.com/official-stockfish/Stockfish.git /tmp/Stockfish
  ( cd /tmp/Stockfish/src && make -j"$(nproc)" build ARCH=x86-64-sse41-popcnt )
  mkdir -p "$HOME/.local/bin"
  cp /tmp/Stockfish/src/stockfish "$SF_BIN"
  SF_VER="$("$SF_BIN" --version 2>/dev/null | head -1 || true)"
fi
export STOCKFISH_PATH="$SF_BIN"
echo "[3/9] STOCKFISH_PATH=$STOCKFISH_PATH  ($SF_VER)"

if [[ ! -x "$STOCKFISH_PATH" || "$SF_VER" != *"Stockfish 16"* ]]; then
  echo "ERROR: Stockfish 16 build/verify failed at $STOCKFISH_PATH" >&2
  exit 1
fi

# BayesElo — build NATIVE from source (one-file compile per its makefile). The committed binary is a
# foreign dynamic build: on a box with different libs it fails to run → empty stdout → the recurring
# "Failed to parse". A native build (g++ already installed for the .so) makes the Elo check work here.
echo ""
echo "[3b/9] Building BayesElo from source..."
if ( cd "$PARENT_DIR/BayesElo" && g++ -o bayeselo -O3 -w bayeselo.cpp ); then
  echo "[3b/9] BayesElo built native at $PARENT_DIR/BayesElo/bayeselo"
else
  echo "[3b/9] WARNING: BayesElo build failed — Elo measurement will report errors." >&2
fi

# ──────────────────────────────────────────────────────────────────────────────
# 4. Python — install deps system-wide (dedicated server, no venv needed)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/9] Installing Python dependencies ($(python3 --version))..."

PYTHON="$(which python3)"  # used only for cmake PYTHON3_EXE below

# --break-system-packages is REQUIRED on PEP-668 systems (Ubuntu 24.04 / py3.12) but is an UNKNOWN
# option on older pip (Ubuntu 22.04 / py3.10). Try with the flag, fall back without it.
REQ="$(dirname "$CHESS_AI_DIR")/requirements.txt"
pip3 install -q --break-system-packages -r "$REQ" 2>/dev/null \
  || pip3 install -q -r "$REQ"
echo "  pip install complete."

# ──────────────────────────────────────────────────────────────────────────────
# 5. Model checkpoint
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/9] Checking model checkpoint..."

MODEL_PATH="$CHESS_AI_DIR/model/best_model.pth"
GDRIVE_FILE_ID="${GDRIVE_MODEL_ID:-1FHQQI9hNmIxAZd6zmX6QO8oow5ekjgGs}"

if [[ -f "$MODEL_PATH" ]]; then
  echo "  best_model.pth found ($(du -h "$MODEL_PATH" | cut -f1))"
elif [[ -n "$GDRIVE_FILE_ID" ]]; then
  echo "  Downloading best_model.pth from Google Drive..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  gdown "$GDRIVE_FILE_ID" -O "$MODEL_PATH"
  echo "  Downloaded: $(du -h "$MODEL_PATH" | cut -f1)"
else
  echo "  WARNING: best_model.pth not found."
  echo "  Upload it with:"
  echo "    gdown <YOUR_GDRIVE_FILE_ID> -O $MODEL_PATH"
  echo "  Or copy it manually, then re-run this script."
  echo "  Training will fail without it."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 6. Build C++ MCTS extension
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/9] Building C++ MCTS extension..."

GAME_ENGINE_DIR="$CHESS_AI_DIR"
PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
PYTHON3_EXE=python3

cd "$GAME_ENGINE_DIR"
rm -rf build
mkdir build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PYTHON3_EXE" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  ..

make -j"$(nproc)"

# Copy the built .so back to game_engine/
cp mcts_engine_cpp*.so "$GAME_ENGINE_DIR/"

cd "$CHESS_AI_DIR"
echo "  C++ extension built and copied to $GAME_ENGINE_DIR/"

# ──────────────────────────────────────────────────────────────────────────────
# 6. Verify C++ extension loads
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[7/9] Verifying C++ extension loads..."

python3 - <<'PYCHECK'
import sys, os
sys.path.insert(0, os.getcwd())
import mcts_engine_cpp
print(f"  mcts_engine_cpp loaded OK: {mcts_engine_cpp}")
PYCHECK

# ──────────────────────────────────────────────────────────────────────────────
# 7. Auto-detect VRAM and set hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[8/9] Setting hyperparameters based on GPU VRAM (${VRAM_GB} GB)..."

NCPU=$(nproc)
source "$GAME_ENGINE_DIR/hyperparams.env.sh"

# Optional per-platform override file, sourced AFTER hyperparams so its values WIN over the defaults.
# run_aws.sh sets EXTRA_ENV to inject AWS/L4 config (eval workers, batch, anchor). No-op on GCP/local.
if [[ -n "${EXTRA_ENV:-}" && -f "${EXTRA_ENV}" ]]; then
  echo "  [env] sourcing overrides ← $EXTRA_ENV"
  source "$EXTRA_ENV"
fi

# GPU tier label (display only).
if   (( VRAM_MIB >= 35000 )); then GPU_TIER="A100"
elif (( VRAM_MIB >= 20000 )); then GPU_TIER="A30/V100-32"
else                                GPU_TIER="T4/V100-16"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 8. Print final config summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[9/9] Final configuration:"
echo "  GPU tier         : $GPU_TIER ($GPU_NAME, ${VRAM_GB} GB VRAM)"
echo "  CPU cores        : $NCPU  →  NUM_WORKERS=$NUM_WORKERS"
echo "  CUDA_BATCH_SIZE  : $CUDA_BATCH_SIZE"
echo "  CUDA_STREAMS     : $CUDA_STREAMS"
echo "  WORKER_BATCH_SIZE: $WORKER_BATCH_SIZE"
echo "  SIMULATIONS      : $SIMULATIONS"
echo "  GAMES_PER_WORKER : $GAMES_PER_WORKER"
echo "  ITERATIONS       : $ITERATIONS"
echo "  STOCKFISH_PATH   : $STOCKFISH_PATH"
echo "  Log file         : $PARENT_DIR/training_log.txt"
echo "  EVAL_SIMULATIONS : $EVAL_SIMULATIONS"
echo "  STOCKFISH_GAMES  : $STOCKFISH_GAMES"
echo "  CUDA_TIMEOUT     : ${CUDA_TIMEOUT_INFERENCE}s"
echo "  MAX_MOVES        : $MAX_MOVES_PER_GAME"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 9. Launch training
# ──────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Launching training..."
echo "============================================================"
echo ""

# Run from chess_ai/ so 'game_engine' package is importable.
# PYTHONPATH also includes chess_ai/game_engine/ so bare 'import mcts_engine_cpp' finds the .so.
cd "$PARENT_DIR"
export PYTHONPATH="$CHESS_AI_DIR${PYTHONPATH:+:$PYTHONPATH}"

if $BACKGROUND; then
  echo "  Running in background (nohup). PID will be written to logs/training.pid"
  # main.py's Logger writes ALL stdout to $PARENT_DIR/training_log.txt (chess_ai/). Send the
  # process stdout to /dev/null (Logger already handles it) and append only stderr (segfaults /
  # C++) to that SAME file → exactly one log, no duplicate under game_engine/.
  nohup python3 -m game_engine.main >/dev/null 2>> "$PARENT_DIR/training_log.txt" &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "$CHESS_AI_DIR/logs/training.pid"
  echo "  Training started with PID $TRAIN_PID"
  echo "  Follow logs: tail -f $PARENT_DIR/training_log.txt"
else
  python3 -m game_engine.main
fi
