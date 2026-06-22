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
#SBATCH --cpus-per-task=96              # self-play CPU workers + training dataloaders scale to this
#SBATCH --mem=192G
#SBATCH --time=48:00:00                 # generous for 3 iters; raise to partition max if needed
#SBATCH --signal=B:TERM@300             # SIGTERM 300s before time limit → graceful save
#SBATCH --output=chess_ai/logs/carc_%x_%j.out
#SBATCH --error=chess_ai/logs/carc_%x_%j.err

set -euo pipefail
ITERS_PER_JOB=15

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
module load gcc/13.3.0
module load python/3.11.9
module load cmake/3.29.4
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
# 3. Stockfish → BUILD FROM SOURCE into ~/.local/bin.
#    The official prebuilt "ubuntu" binaries need glibc >=2.29; CARC (Rocky 8)
#    has glibc 2.28, so we compile against the node's toolchain (gcc module).
#    Builds under $HOME (avoids any noexec /tmp); NNUE net is embedded.
# ---------------------------------------------------------------------------
if [[ ! -x "$HOME/.local/bin/stockfish" ]]; then
  echo "[sf] building Stockfish from source ..."
  mkdir -p "$HOME/.local/bin"
  SFB="$(mktemp -d -p "$HOME")"
  curl -fSL --retry 3 -o "$SFB/sf.tgz" \
    "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_16.1.tar.gz"
  tar xzf "$SFB/sf.tgz" -C "$SFB"
  ( cd "$SFB"/Stockfish-*/src && make -j"$(nproc)" build ARCH=x86-64-sse41-popcnt )
  cp "$SFB"/Stockfish-*/src/stockfish "$HOME/.local/bin/stockfish"
  chmod +x "$HOME/.local/bin/stockfish"
  rm -rf "$SFB"
  echo "[sf] built → $HOME/.local/bin/stockfish"
else
  echo "[sf] stockfish present — skipping build."
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
# 5b. BayesElo — the committed binary needs glibc >=2.34; Rocky 8 has 2.28, so
#     rebuild it from the in-repo source if it can't run here (else Stockfish
#     Elo silently fails to parse). Unity build: g++ on bayeselo.cpp alone.
# ---------------------------------------------------------------------------
if ! echo quit | "$CHESS_AI/BayesElo/bayeselo" >/dev/null 2>&1; then
  echo "[bayeselo] committed binary won't run here — rebuilding from source ..."
  ( cd "$CHESS_AI/BayesElo" && g++ -O3 -Wall -o bayeselo bayeselo.cpp )
  echo quit | "$CHESS_AI/BayesElo/bayeselo" >/dev/null 2>&1 \
    && echo "[bayeselo] rebuilt OK" \
    || echo "[bayeselo] rebuild FAILED — Stockfish Elo won't compute" >&2
else
  echo "[bayeselo] binary runs — OK."
fi

# ---------------------------------------------------------------------------
# 5c. Seed model checkpoints from Google Drive (latest nets live there). Runs ONLY
#     on the first job (no best_model.pth yet) or when SYNC_MODELS=1 — so a resuming
#     job never clobbers CARC's own newer checkpoints. Folder must be shared.
# ---------------------------------------------------------------------------
GDRIVE_MODELS_FOLDER="https://drive.google.com/drive/folders/1J_cfNiIusiVMhUjTZ0rI80gpKo3XEu_u"
if [[ ! -f "$GAME/model/best_model.pth" ]] || [[ "${SYNC_MODELS:-0}" == "1" ]]; then
  echo "[gdown] seeding checkpoints from Drive → $GAME/model"
  python3 -c "import gdown" 2>/dev/null || python3 -m pip install --user --quiet gdown
  mkdir -p "$GAME/model"
  STAGE="$(mktemp -d -p "$HOME")"
  gdown --folder "$GDRIVE_MODELS_FOLDER" -O "$STAGE"
  # find-copy handles gdown's folder nesting; overwrites stale copies, never deletes.
  find "$STAGE" -type f \( -name '*.pth' -o -name '*.bak' \) -exec cp -fv {} "$GAME/model/" \;
  rm -rf "$STAGE"
  ls -la "$GAME/model"/*.pth 2>/dev/null || echo "[gdown] WARNING: no .pth fetched — check folder is shared"
else
  echo "[gdown] best_model.pth present — skipping Drive sync (SYNC_MODELS=1 to force-refresh)."
fi

# ---------------------------------------------------------------------------
# 6. Hyperparameters + cap this job to 3 fresh iterations (from run_state.json)
# ---------------------------------------------------------------------------
# hyperparams.env.sh needs these set BEFORE sourcing (VRAM tier + worker counts).
export NCPU="$(nproc)"
export VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
set -a; source "$GAME/hyperparams.env.sh"; set +a
export PYTHONPATH="$GAME${PYTHONPATH:+:$PYTHONPATH}"

# ─── A100 + CARC calibration (overrides win over hyperparams defaults) ──────────────────────────
# Training batch leverages the A100 but is VRAM-adaptive: --gres=gpu:a100 may land a 40GB OR 80GB
# card, and 4096 OOMs below ~80GB (Vast's 24GB card OOM'd at 4096 → 2048). Sized off detected VRAM.
if   (( VRAM_MIB >= 70000 )); then export TRAIN_BATCH_SIZE=4096
elif (( VRAM_MIB >= 38000 )); then export TRAIN_BATCH_SIZE=2048
else                               export TRAIN_BATCH_SIZE=1024; fi
export TRAIN_DL_WORKERS=32
export TRAIN_DL_PREFETCH=8

# Self-play scaling for the requested 96-core / A100 node. 184 workers ≈ 2× the 96 cores — deliberate
# oversubscription so the latency-bound search keeps the A100's inference queue full (Vast finding:
# self-play is round-trip-bound, more producers fill the batch). If SLURM only grants 48 cores it's
# 4× — still runs (workers block on GPU), just more CPU contention. NUM_WORKERS overrides hyperparams'
# 60; CUDA_BATCH_SIZE MUST be recomputed to NUM_WORKERS×WORKER_BATCH_SIZE (capped by VRAM_CAP) or the
# inference server stalls 1s/round (Vast lesson). 184 workers × 10 games = 1840 games/iter.
export NUM_WORKERS=184
export GAMES_PER_WORKER=10         # self-play games each worker plays per iteration (pinned; matches hyperparams)
export RESERVED_CORES=4            # reserve 4 of 96 for the GPU-feeder threads (Vast parity; avoids deadlock-self-kill)
export CUDA_TIMEOUT_INFERENCE=0.02 # Vast-tuned best; self-play is latency-bound, shorter timeout = more leaves/s
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# Vast-parity fixes that hyperparams.env.sh's stale (iter-3-rollback-era) defaults would otherwise
# undo — without these the opening collapse partly returns and the data window is wrong:
export TEMP_MOVES=16              # restore the 16-ply temperature window (hyperparams halved it to 8)
export MAX_WORKER_LEAD=3          # tighter game-lead spread → less iter-end tail-idle
export TRAIN_MIN_ITER=0           # keep ALL self-play data (default 8 drops the clean post-restart iters)
export TRAIN_FROM_LINEAGE=1       # continue training from candidate.pth (AZ-2017), matches the Vast run
# Opening book (USE_OPENING_BOOK defaults to 1) forces 20 diverse openings — the core diversity fix.

# Eval scaling: 92 workers × 2 games = 184 arena games (2/worker = 1 White + 1 Black, color-balanced).
export EVAL_WORKERS=92
export GAMES_PER_EVAL_WORKER=2
export STOCKFISH_WORKERS=92
export STOCKFISH_GAMES=200
# Elo ONLY on promotion (EVERY_ITER=0): a rejected iter leaves best_model unchanged, so re-measuring
# burns 200 games + adds CI noise to a flat trend (the Vast decision once BayesElo became reliable).
export STOCKFISH_EVERY_ITER=0
# Anchor: SF-2800 (the proven Vast value — gave a clean champion Elo 2422±32, cross-confirmed). The
# ~2422 champion scores ~8% here; low WR but BayesElo rates it fine and it leaves headroom as the net
# climbs toward 2800. (SF-1320 would be a 100% sweep → BayesElo can't rate it.) NODES=0 = UCI time control.
export STOCKFISH_ELO=2800
export STOCKFISH_NODES=0

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
# Raise the open-file limit. 92 workers × 2 arena servers × (queues + semaphores + SHM) exhaust
# the default (~1024) and FDs accumulate across iterations — job 9511845 died at iter-21 arena
# with "[Errno 24] Too many open files" → server deadlock. Set to the hard max for headroom.
# ---------------------------------------------------------------------------
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[fd] open-files limit = $(ulimit -n)"

# ---------------------------------------------------------------------------
# 7. Run the loop EXACTLY as GCP did (cwd=chess_ai → Logger writes
#    chess_ai/training_log.txt; model/ + data/self_play/ updated in place).
#    Forward SLURM's SIGTERM to python for graceful phase-resume.
# ---------------------------------------------------------------------------
cd "$CHESS_AI"

# Background resource sampler for next-run tuning. Every 60s, across all 3 phases:
# GPU util %, GPU mem used (MiB), 1-min load avg, live python-process count.
# Prints '[usage] ...' to the SLURM .out (grep it) AND a clean CSV at logs/usage_<jobid>.csv.
USAGE_CSV="$CHESS_AI/logs/usage_${SLURM_JOB_ID:-local}.csv"
echo "time,gpu_util_pct,gpu_mem_MiB,load_1min,py_procs" > "$USAGE_CSV"
( while true; do
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    load=$(awk '{print $1}' /proc/loadavg 2>/dev/null)
    npy=$(pgrep -u "$(id -un)" -fc python 2>/dev/null || echo 0)
    line="$(date +%T),${g:-NA},${load:-NA},${npy:-0}"
    echo "$line" >> "$USAGE_CSV"
    echo "[usage] $line"
    sleep 60
  done ) &
SAMPLER_PID=$!

python3 -m game_engine.main &
PID=$!
graceful() { echo "[run] SIGTERM (time limit) → graceful stop"; kill -TERM "$PID" 2>/dev/null || true; }
trap graceful TERM
set +e
wait "$PID"
while kill -0 "$PID" 2>/dev/null; do sleep 5; done
kill "$SAMPLER_PID" 2>/dev/null
echo "[run] python finished at $(date)"
