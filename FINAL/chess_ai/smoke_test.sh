#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Chess AI/ML System Full Verification
#
# Checks every layer of the system: files, model weights, chess env, C++
# extension, MCTS tree reuse, data augmentation, trainer, evaluation,
# Stockfish, metrics.json, self-play data, source code, and SLURM scripts.
#
# Usage:  bash smoke_test.sh          (from FINAL/chess_ai/)
#         PYTHON=/path/to/python bash smoke_test.sh
# =============================================================================

CHESS_AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/krish/miniconda3/envs/chessai/bin/python3.11}"

PASS=0; FAIL=0; WARN=0

# ── Terminal colors ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    G='\033[0;32m' R='\033[0;31m' Y='\033[1;33m'
    B='\033[1;34m' BOLD='\033[1m' NC='\033[0m'
else
    G='' R='' Y='' B='' BOLD='' NC=''
fi

header() { printf "\n${BOLD}${B}── %s${NC}\n" "$1"; }

ok() {
    printf "  ${G}✅  %s${NC}\n" "$1"
    PASS=$((PASS+1))
}

fail() {
    printf "  ${R}❌  %s${NC}\n" "$1"
    [[ -n "${2:-}" ]] && printf "     ${R}↳ %s${NC}\n" "$2"
    FAIL=$((FAIL+1))
}

warn() {
    printf "  ${Y}⚠   %s${NC}\n" "$1"
    [[ -n "${2:-}" ]] && printf "     ${Y}↳ %s${NC}\n" "$2"
    WARN=$((WARN+1))
}

# Parse Python output lines: PASS:/FAIL:/WARN: [description] [| detail]
parse() {
    while IFS= read -r line; do
        local full msg detail
        case "$line" in
            PASS:*)
                full="${line:5}"
                ok "$full"
                ;;
            FAIL:*)
                full="${line:5}"
                if [[ "$full" == *" | "* ]]; then
                    msg="${full%% | *}"; detail="${full##* | }"
                    fail "$msg" "$detail"
                else
                    fail "$full"
                fi
                ;;
            WARN:*)
                full="${line:5}"
                if [[ "$full" == *" | "* ]]; then
                    msg="${full%% | *}"; detail="${full##* | }"
                    warn "$msg" "$detail"
                else
                    warn "$full"
                fi
                ;;
        esac
    done <<< "$1"
}

printf "${BOLD}==============================================\n"
printf " Chess AI/ML System — Smoke Test\n"
printf " Dir:    %s\n" "$CHESS_AI_DIR"
printf " Python: %s\n" "$PYTHON"
printf "==============================================${NC}\n"

# =============================================================================
# 1. FILES & DIRECTORIES
# =============================================================================
header "1. Files & Directories"

file_check() {
    local path="$1"
    if [[ -f "$CHESS_AI_DIR/$path" ]]; then
        ok "$path"
    else
        fail "$path" "not found"
    fi
}

file_check "game_engine/main.py"
file_check "game_engine/chess_env.py"
file_check "game_engine/cnn.py"
file_check "game_engine/trainer.py"
file_check "game_engine/mcts_worker_cpp.py"
file_check "game_engine/evaluation.py"
file_check "game_engine/neural_net.py"
file_check "game_engine/src/mcts_engine.cpp"
file_check "game_engine/src/mcts_engine.h"
file_check "game_engine/src/python_bridge.cpp"
file_check "game_engine/model/best_model.pth"
file_check "game_engine/model/metrics.json"
file_check "slurm/train_a100.sbatch"
file_check "slurm/train_v100.sbatch"

# C++ extension (.so for running Python version)
PY_TAG=$("$PYTHON" -c "import sys; print(f'cpython-{sys.version_info.major}{sys.version_info.minor}')")
SO_PATH="$CHESS_AI_DIR/game_engine/mcts_engine_cpp.${PY_TAG}"*".so"
if ls $SO_PATH &>/dev/null; then
    ok "C++ extension (.so for $PY_TAG)"
else
    fail "C++ extension (.so for $PY_TAG)" "no mcts_engine_cpp.${PY_TAG}*.so in game_engine/"
fi

# Data directory (warn only — fresh run has no data yet)
if [[ -d "$CHESS_AI_DIR/data/self_play" ]]; then
    iter_count=$(find "$CHESS_AI_DIR/data/self_play" -maxdepth 1 -name "iter_*" -type d | wc -l)
    ok "data/self_play/ ($iter_count iteration dirs)"
else
    warn "data/self_play/ does not exist" "expected for a fresh run before any self-play"
fi

# =============================================================================
# 2. PYTHON ENVIRONMENT
# =============================================================================
header "2. Python Environment"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

def warn_check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"WARN: {name} | {str(e).replace(chr(10),' ')[:120]}")

check("Python 3.11",
    lambda: None if sys.version_info[:2] == (3, 11)
            else (_ for _ in ()).throw(Exception(f"got {sys.version_info.major}.{sys.version_info.minor}")))

import torch
check("torch importable",   lambda: __import__("torch"))
check("numpy importable",   lambda: __import__("numpy"))
check("chess importable",   lambda: __import__("chess"))
check("pybind11 importable", lambda: __import__("pybind11"))

warn_check("CUDA available",
    lambda: None if torch.cuda.is_available()
            else (_ for _ in ()).throw(Exception("no GPU — CPU-only mode")))

if torch.cuda.is_available():
    check("CUDA device name",
        lambda: print(f"PASS: CUDA device: {torch.cuda.get_device_name(0)}") or None)
PY
)
parse "$out"

# =============================================================================
# 3. MODEL WEIGHTS (best_model.pth)
# =============================================================================
header "3. Model Weights"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import torch, sys
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

model_path = "game_engine/model/best_model.pth"

raw = None
def load_raw():
    global raw
    raw = torch.load(model_path, map_location="cpu")
check("best_model.pth loads", load_raw)
if raw is None:
    sys.exit(0)

check("has model_state_dict key",
    lambda: None if "model_state_dict" in raw
            else (_ for _ in ()).throw(Exception(f"keys: {list(raw.keys())}")))

state = raw.get("model_state_dict", raw)
w = state.get("input_conv.0.weight")

check("input_conv.0.weight exists", lambda: None if w is not None else (_ for _ in ()).throw(Exception("key missing")))
if w is None:
    sys.exit(0)

check("input_conv.0.weight shape (256,120,3,3)",
    lambda: None if tuple(w.shape) == (256, 120, 3, 3)
            else (_ for _ in ()).throw(Exception(f"got {tuple(w.shape)}")))

check("channels 0-15 non-zero (original weights preserved)",
    lambda: None if (w[:, :16, :, :] != 0).any()
            else (_ for _ in ()).throw(Exception("all zeros — original weights lost!")))

check("channels 16-119 zero-initialized",
    lambda: None if (w[:, 16:, :, :] == 0).all()
            else (_ for _ in ()).throw(Exception(
                f"{(w[:, 16:, :, :] != 0).sum().item()} non-zero elements found")))

# Check all expected layer groups present
expected_prefixes = ["input_conv", "res_blocks", "policy_head", "value_head"]
for prefix in expected_prefixes:
    check(f"state dict has {prefix} keys",
        lambda p=prefix: None if any(k.startswith(p) for k in state)
                else (_ for _ in ()).throw(Exception(f"no keys starting with '{p}'")))

# Load into model and run forward pass
from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame

model = None
def load_model():
    global model
    m = ChessCNN(upgraded=True)
    m.load_state_dict(state)
    m.eval()
    model = m
check("ChessCNN(upgraded=True) loads from checkpoint", load_model)

if model:
    t = torch.zeros(1, 120, 8, 8)
    with torch.no_grad():
        policy, value = model(t)

    check("policy output shape (1, 8192)",
        lambda: None if tuple(policy.shape) == (1, 8192)
                else (_ for _ in ()).throw(Exception(f"got {tuple(policy.shape)}")))
    check("value output shape (1, 1)",
        lambda: None if tuple(value.shape) == (1, 1)
                else (_ for _ in ()).throw(Exception(f"got {tuple(value.shape)}")))
    check("value is finite",
        lambda: None if torch.isfinite(value).all()
                else (_ for _ in ()).throw(Exception(f"value={value.item()}")))
    s = torch.softmax(policy, dim=1).sum().item()
    check("policy softmax sums to 1.0",
        lambda: None if abs(s - 1.0) < 1e-3
                else (_ for _ in ()).throw(Exception(f"sum={s:.6f}")))

# Check scheduler and optimizer keys present (persistent LR)
check("optimizer_state_dict saved",
    lambda: None if "optimizer_state_dict" in raw
            else (_ for _ in ()).throw(Exception("missing — LR won't persist across iterations")))
# Warn (not fail): pre-CARC migration checkpoint may not have it yet;
# trainer.py does save it, so it will appear after the first training run.
warn_check("scheduler_state_dict saved (expect after first training run)",
    lambda: None if "scheduler_state_dict" in raw
            else (_ for _ in ()).throw(Exception(
                "not in checkpoint yet — OK if this is the migrated model before first train")))
PY
)
parse "$out"

# =============================================================================
# 4. CHESS ENVIRONMENT (chess_env.py)
# =============================================================================
header "4. Chess Environment"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, numpy as np
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

from game_engine.chess_env import ChessGame
import chess

game = ChessGame()

# Legal moves
check("initial position: 20 legal moves",
    lambda: None if len(game.legal_moves()) == 20
            else (_ for _ in ()).throw(Exception(f"got {len(game.legal_moves())}")))

# to_tensor() without history
t = game.to_tensor()
check("to_tensor() shape is (120, 8, 8)",
    lambda: None if t.shape == (120, 8, 8)
            else (_ for _ in ()).throw(Exception(f"got {t.shape}")))
check("to_tensor() dtype is float32",
    lambda: None if t.dtype == np.float32
            else (_ for _ in ()).throw(Exception(f"got {t.dtype}")))

# Auxiliary planes at start position (White to move, all castling rights)
check("castling plane 112 (WK) = 1.0 at start",
    lambda: None if t[112].max() == 1.0 else (_ for _ in ()).throw(Exception(f"plane112 max={t[112].max()}")))
check("castling plane 113 (WQ) = 1.0 at start",
    lambda: None if t[113].max() == 1.0 else (_ for _ in ()).throw(Exception(f"plane113 max={t[113].max()}")))
check("castling plane 114 (BK) = 1.0 at start",
    lambda: None if t[114].max() == 1.0 else (_ for _ in ()).throw(Exception(f"plane114 max={t[114].max()}")))
check("castling plane 115 (BQ) = 1.0 at start",
    lambda: None if t[115].max() == 1.0 else (_ for _ in ()).throw(Exception(f"plane115 max={t[115].max()}")))
check("side-to-move plane 117 = 0.0 (White) at start",
    lambda: None if t[117].max() == 0.0 else (_ for _ in ()).throw(Exception(f"plane117={t[117].max()}")))

# to_tensor() with history
boards = [chess.Board()] * 3
t_hist = game.to_tensor(history=boards)
check("to_tensor(history=3 boards) shape is (120, 8, 8)",
    lambda: None if t_hist.shape == (120, 8, 8)
            else (_ for _ in ()).throw(Exception(f"got {t_hist.shape}")))
check("history planes 14-27 non-zero (history frame 1 populated)",
    lambda: None if t_hist[14:28].sum() != 0
            else (_ for _ in ()).throw(Exception("history planes are all zeros")))
check("current position planes 0-13 non-zero",
    lambda: None if t_hist[0:14].sum() != 0
            else (_ for _ in ()).throw(Exception("current position planes are all zeros")))

# push / is_over / turn_player
turn_before = game.turn_player
check("push('e2e4') returns True",
    lambda: None if game.push("e2e4") is True
            else (_ for _ in ()).throw(Exception("push returned False or None")))
check("is_over is False after e2e4",
    lambda: None if not game.is_over
            else (_ for _ in ()).throw(Exception("game ended after one move")))
check("turn_player changed after push",
    lambda: None if game.turn_player != turn_before
            else (_ for _ in ()).throw(Exception("turn_player unchanged")))

# copy() independence
g2 = game.copy()
g2.push("e7e5")
check("copy() is independent (original unaffected by mutations to copy)",
    lambda: None if len(game.moves) != len(g2.moves)
            else (_ for _ in ()).throw(Exception("copy shares state with original")))
PY
)
parse "$out"

# =============================================================================
# 5. C++ EXTENSION
# =============================================================================
header "5. C++ Extension (mcts_engine_cpp)"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

import mcts_engine_cpp

engine = None
def make_engine():
    global engine
    engine = mcts_engine_cpp.MCTSEngine(800, 8)
check("MCTSEngine(800, 8) instantiates", make_engine)

if engine:
    for method in ("search", "advance_root", "reset_cache"):
        check(f"MCTSEngine.{method} exists",
            lambda m=method: None if hasattr(engine, m)
                    else (_ for _ in ()).throw(Exception("method missing")))
    check("MCTSEngine.simulations read/write",
        lambda: None if (setattr(engine, "simulations", 400) or engine.simulations == 400)
                else (_ for _ in ()).throw(Exception("r/w failed")))
    check("MCTSEngine.batch_size read/write",
        lambda: None if (setattr(engine, "batch_size", 4) or engine.batch_size == 4)
                else (_ for _ in ()).throw(Exception("r/w failed")))
PY
)
parse "$out"

# =============================================================================
# 6. MCTS WORKER + TREE REUSE
# =============================================================================
header "6. MCTS Worker & Tree Reuse"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, torch
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

from game_engine.chess_env import ChessGame
from game_engine.cnn import ChessCNN
from game_engine.mcts_worker_cpp import MCTSWorker

raw = torch.load("game_engine/model/best_model.pth", map_location="cpu")
model = ChessCNN(upgraded=True)
model.load_state_dict(raw["model_state_dict"])
model.eval()

worker = MCTSWorker(worker_id=0, input_queue=None, output_queue=None,
                    simulations=50, batch_size=8)

check("MCTSWorker creates", lambda: None)
check("reset_cache() runs without error", lambda: worker.reset_cache())
check("advance_root method present", lambda: None if hasattr(worker, "advance_root")
      else (_ for _ in ()).throw(Exception("missing")))

# Play 5 moves, verify tree reuse kicks in from move 2
game = ChessGame()
worker.reset_cache()
reuse_results = []
policy_sums = []

for i in range(5):
    best_move, policy = worker.search_direct(game, model, temperature=1.0)
    reused = worker.advance_root(best_move)
    reuse_results.append(reused)
    policy_sums.append(float(policy.sum()))
    game.push(best_move)

check("search_direct() completes 5 moves", lambda: None)

# After the first search, advance_root finds the played move in the freshly-built
# tree and returns True (correct). Moves 2-5 reuse increasingly explored subtrees.
check("all 5 advance_root calls return True (tree populated by search)",
    lambda: None if all(reuse_results)
            else (_ for _ in ()).throw(Exception(f"reuse results: {reuse_results}")))

for i, s in enumerate(policy_sums):
    check(f"move {i+1} policy sums to ≈1.0 (got {s:.4f})",
        lambda sv=s: None if abs(sv - 1.0) < 0.01
                else (_ for _ in ()).throw(Exception(f"sum={sv:.6f}")))

# After reset, advance_root should return False (no cached tree)
worker.reset_cache()
check("advance_root returns False after reset_cache (no stale tree)",
    lambda: None if worker.advance_root("e2e4") is False
            else (_ for _ in ()).throw(Exception("returned True — stale tree not cleared")))
PY
)
parse "$out"

# =============================================================================
# 7. DATA AUGMENTATION (horizontal board flip)
# =============================================================================
header "7. Data Augmentation"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, torch
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

from game_engine.trainer import _H_FLIP_PERM

check("_H_FLIP_PERM shape is (8192,)",
    lambda: None if _H_FLIP_PERM.shape == (8192,)
            else (_ for _ in ()).throw(Exception(f"got {_H_FLIP_PERM.shape}")))
check("_H_FLIP_PERM dtype is long (index tensor)",
    lambda: None if _H_FLIP_PERM.dtype == torch.long
            else (_ for _ in ()).throw(Exception(f"got {_H_FLIP_PERM.dtype}")))
check("_H_FLIP_PERM is a permutation (all indices in 0..8191)",
    lambda: None if _H_FLIP_PERM.max().item() < 8192 and _H_FLIP_PERM.min().item() >= 0
            else (_ for _ in ()).throw(Exception("out-of-range indices")))

# Double-flip identity: applying perm twice must recover original
policy = torch.rand(8192)
double_flipped = policy[_H_FLIP_PERM][_H_FLIP_PERM]
check("double-flip identity (flip ∘ flip = identity)",
    lambda: None if torch.allclose(policy, double_flipped)
            else (_ for _ in ()).throw(Exception(
                f"max diff = {(policy - double_flipped).abs().max().item():.6f}")))

# Spot check: e2e4 (file e=4, rank 1→3) flips to d2d4 (file d=3, rank 1→3)
# src e2 = file 4 + rank 1 * 8 = 12   dst e4 = file 4 + rank 3 * 8 = 28  → idx 12*64+28 = 796
# src d2 = file 3 + rank 1 * 8 = 11   dst d4 = file 3 + rank 3 * 8 = 27  → idx 11*64+27 = 731
e2e4_idx = 12 * 64 + 28   # 796
d2d4_idx = 11 * 64 + 27   # 731
check("e2e4 (idx 796) flips to d2d4 (idx 731): perm[731] == 796",
    lambda: None if _H_FLIP_PERM[d2d4_idx].item() == e2e4_idx
            else (_ for _ in ()).throw(Exception(
                f"perm[{d2d4_idx}] = {_H_FLIP_PERM[d2d4_idx].item()}, expected {e2e4_idx}")))

# Functional: set policy[796]=1, flip, expect result[731]=1
p = torch.zeros(8192); p[e2e4_idx] = 1.0
p_flipped = p[_H_FLIP_PERM]
check("e2e4 policy → d2d4 after flip (functional)",
    lambda: None if p_flipped[d2d4_idx].item() == 1.0 and p_flipped[e2e4_idx].item() == 0.0
            else (_ for _ in ()).throw(Exception(
                f"d2d4={p_flipped[d2d4_idx].item()}, e2e4={p_flipped[e2e4_idx].item()}")))
PY
)
parse "$out"

# =============================================================================
# 8. TRAINER
# =============================================================================
header "8. Trainer"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, os
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

from game_engine.trainer import train_model, ChessDataset, _H_FLIP_PERM
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

check("train_model importable", lambda: None)
check("ChessDataset importable", lambda: None)

# Dataset with missing dir: 0 positions, no crash
ds = ChessDataset("/nonexistent/path/abc123", window_size=5)
check("ChessDataset(missing dir) = 0 positions, no crash",
    lambda: None if len(ds) == 0
            else (_ for _ in ()).throw(Exception(f"got {len(ds)} positions")))

# Dataset shape validation: trainer skips 16-channel old data
import numpy as np, tempfile, glob
with tempfile.TemporaryDirectory() as tmpdir:
    iter_dir = os.path.join(tmpdir, "iter_1")
    os.makedirs(iter_dir)
    # Write an old-format (16-channel) file — should be skipped
    np.savez_compressed(os.path.join(iter_dir, "old.npz"),
                        states=np.zeros((10, 16, 8, 8), dtype=np.float32),
                        policies=np.zeros((10, 8192), dtype=np.float32),
                        values=np.zeros(10, dtype=np.float32))
    ds_old = ChessDataset(tmpdir, window_size=1)
    check("ChessDataset skips 16-channel (old format) files",
        lambda: None if len(ds_old) == 0
                else (_ for _ in ()).throw(Exception(f"accepted old data: {len(ds_old)} positions")))

    # Write a new-format (120-channel) file — should be loaded
    np.savez_compressed(os.path.join(iter_dir, "new.npz"),
                        states=np.zeros((5, 120, 8, 8), dtype=np.float32),
                        policies=np.zeros((5, 8192), dtype=np.float32),
                        values=np.zeros(5, dtype=np.float32))
    ds_new = ChessDataset(tmpdir, window_size=1)
    check("ChessDataset accepts 120-channel (new format) files",
        lambda: None if len(ds_new) == 5
                else (_ for _ in ()).throw(Exception(f"got {len(ds_new)} positions")))

# LR scheduler: verify CosineAnnealingLR is used (not ExponentialLR)
import inspect, game_engine.trainer as tr_mod
src = inspect.getsource(tr_mod.train_model)
check("train_model uses CosineAnnealingLR",
    lambda: None if "CosineAnnealingLR" in src
            else (_ for _ in ()).throw(Exception("CosineAnnealingLR not found in train_model source")))
check("train_model saves scheduler_state_dict",
    lambda: None if "scheduler_state_dict" in src
            else (_ for _ in ()).throw(Exception("scheduler_state_dict not saved to checkpoint")))
check("train_model accepts total_iterations param",
    lambda: None if "total_iterations" in src
            else (_ for _ in ()).throw(Exception("total_iterations param missing")))
PY
)
parse "$out"

# =============================================================================
# 9. EVALUATION
# =============================================================================
header "9. Evaluation"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

from game_engine.evaluation import EvalMCTS, Arena, StockfishEvaluator
import inspect

check("EvalMCTS importable", lambda: None)
check("Arena importable", lambda: None)
check("StockfishEvaluator importable", lambda: None)

# EvalMCTS must have tree-reuse methods
src_eval = inspect.getsource(EvalMCTS)
check("EvalMCTS.advance_root defined",
    lambda: None if "def advance_root" in src_eval
            else (_ for _ in ()).throw(Exception("method not found")))
check("EvalMCTS.reset_cache defined",
    lambda: None if "def reset_cache" in src_eval
            else (_ for _ in ()).throw(Exception("method not found")))

src_arena = inspect.getsource(Arena.play_game)
check("Arena.play_game calls reset_cache at game start",
    lambda: None if "reset_cache" in src_arena
            else (_ for _ in ()).throw(Exception("reset_cache not called")))
check("Arena.play_game calls advance_root after each move",
    lambda: None if "advance_root" in src_arena
            else (_ for _ in ()).throw(Exception("advance_root not called")))

src_sf = inspect.getsource(StockfishEvaluator.evaluate_with_bayeselo)
check("StockfishEvaluator.evaluate_with_bayeselo calls advance_root",
    lambda: None if "advance_root" in src_sf
            else (_ for _ in ()).throw(Exception("advance_root not called")))
check("StockfishEvaluator.evaluate_with_bayeselo calls reset_cache",
    lambda: None if "reset_cache" in src_sf
            else (_ for _ in ()).throw(Exception("reset_cache not called")))
PY
)
parse "$out"

# =============================================================================
# 10. STOCKFISH
# =============================================================================
header "10. Stockfish"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, os
sys.path.insert(0, "game_engine")
sys.path.insert(0, ".")

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

def warn_check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"WARN: {name} | {str(e).replace(chr(10),' ')[:120]}")

sf_path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
warn_check(f"stockfish binary exists ({sf_path})",
    lambda: None if os.path.exists(sf_path)
            else (_ for _ in ()).throw(Exception("not found — eval phase will fail")))

if os.path.exists(sf_path):
    import chess as chess_lib
    import chess.engine
    def test_sf():
        with chess.engine.SimpleEngine.popen_uci(sf_path) as eng:
            board = chess_lib.Board()
            result = eng.play(board, chess.engine.Limit(time=0.05))
            assert result.move is not None, "no move returned"
    warn_check("stockfish responds to UCI play command", test_sf)
PY
)
parse "$out"

# =============================================================================
# 11. METRICS.JSON
# =============================================================================
header "11. metrics.json"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import json, math, sys

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

metrics_path = "game_engine/model/metrics.json"

data = None
def load_metrics():
    global data
    with open(metrics_path) as f:
        data = json.load(f)
check("metrics.json parses as valid JSON", load_metrics)
if data is None:
    sys.exit(0)

check("metrics.json is a list (empty = fresh start is OK)",
    lambda: None if isinstance(data, list)
            else (_ for _ in ()).throw(Exception(f"expected list, got {type(data).__name__}")))

if not isinstance(data, list) or len(data) == 0:
    sys.exit(0)

required_keys = ["iteration", "timestamp", "policy_loss", "value_loss",
                 "arena_win_rate", "model_elo", "stockfish_elo"]
entry = data[-1]

for k in required_keys:
    check(f"last entry has key '{k}'",
        lambda k=k: None if k in entry
                else (_ for _ in ()).throw(Exception(f"missing; found: {list(entry.keys())}")))

# Verify bug fix: key is 'model_elo', NOT 'elo'
check("uses 'model_elo' key (not 'elo' — W7 bug fix)",
    lambda: None if "model_elo" in entry and "elo" not in entry
            else (_ for _ in ()).throw(Exception(
                f"model_elo={'model_elo' in entry}, elo={'elo' in entry}")))

# Value sanity checks
if "policy_loss" in entry:
    check(f"policy_loss is finite ({entry['policy_loss']:.4f})",
        lambda: None if math.isfinite(entry["policy_loss"])
                else (_ for _ in ()).throw(Exception("NaN or Inf")))
if "value_loss" in entry:
    check(f"value_loss is finite ({entry['value_loss']:.4f})",
        lambda: None if math.isfinite(entry["value_loss"])
                else (_ for _ in ()).throw(Exception("NaN or Inf")))
if "model_elo" in entry and entry["model_elo"] is not None:
    check(f"model_elo > 0 ({entry['model_elo']:.0f})",
        lambda: None if float(entry["model_elo"]) > 0
                else (_ for _ in ()).throw(Exception(f"got {entry['model_elo']}")))

print(f"PASS: {len(data)} total entries, latest: iter {entry.get('iteration')} | ELO {entry.get('model_elo')}")
PY
)
parse "$out"

# =============================================================================
# 12. SELF-PLAY DATA
# =============================================================================
header "12. Self-Play Data"

out=$(cd "$CHESS_AI_DIR" && "$PYTHON" - 2>&1 <<'PY'
import sys, os, glob, numpy as np

def check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name} | {str(e).replace(chr(10),' ')[:120]}")

def warn_check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"WARN: {name} | {str(e).replace(chr(10),' ')[:120]}")

data_dir = "data/self_play"

warn_check("data/self_play/ exists",
    lambda: None if os.path.isdir(data_dir)
            else (_ for _ in ()).throw(Exception("run self-play phase first")))

if not os.path.isdir(data_dir):
    sys.exit(0)

subdirs = sorted([d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("iter_")],
                 key=lambda x: int(x.split("_")[1]))

warn_check(f"self-play data present ({len(subdirs)} iter dirs) — OK if starting fresh",
    lambda: None if len(subdirs) > 0
            else (_ for _ in ()).throw(Exception("no self-play data yet — first training run will populate this")))

if not subdirs:
    sys.exit(0)

latest = subdirs[-1]
npz_files = glob.glob(os.path.join(data_dir, latest, "*.npz"))
warn_check(f"latest dir ({latest}) has .npz files ({len(npz_files)} found)",
    lambda: None if len(npz_files) > 0
            else (_ for _ in ()).throw(Exception("no .npz files in latest iter dir")))

if not npz_files:
    sys.exit(0)

# Load one sample file
sample = np.load(npz_files[0], allow_pickle=True)
s_shape = sample["states"].shape
p_shape = sample["policies"].shape
v = sample["values"]

check("sample .npz has 'states' key", lambda: None if "states" in sample else (_ for _ in ()).throw(Exception("missing")))
check("sample .npz has 'policies' key", lambda: None if "policies" in sample else (_ for _ in ()).throw(Exception("missing")))
check("sample .npz has 'values' key", lambda: None if "values" in sample else (_ for _ in ()).throw(Exception("missing")))
check("states is 4-dimensional",
    lambda: None if len(s_shape) == 4
            else (_ for _ in ()).throw(Exception(f"got {len(s_shape)}D: {s_shape}")))

# Warn if old 16-channel format (will be skipped by trainer)
if len(s_shape) == 4:
    ch = s_shape[1]
    if ch == 120:
        check(f"states shape channels = 120 (new format) — {s_shape}",
            lambda: None)
    elif ch == 16:
        print(f"WARN: states shape has 16 channels (OLD format) — {s_shape} | trainer will SKIP these files; new 120ch data needed")
    else:
        check(f"states channel count = 120",
            lambda: (_ for _ in ()).throw(Exception(f"unexpected channel count: {ch} in {s_shape}")))

check(f"policies shape (N, 8192) — got {p_shape}",
    lambda: None if len(p_shape) == 2 and p_shape[1] == 8192
            else (_ for _ in ()).throw(Exception(f"got {p_shape}")))
check(f"values in [-1, 1] — min={v.min():.2f} max={v.max():.2f}",
    lambda: None if v.min() >= -1.0 and v.max() <= 1.0
            else (_ for _ in ()).throw(Exception(f"out of range: [{v.min():.2f}, {v.max():.2f}]")))
PY
)
parse "$out"

# =============================================================================
# 13. SOURCE CODE CHECKS (grep-based)
# =============================================================================
header "13. Source Code Integrity"

src_check() {
    local description="$1"
    local file="$2"
    local pattern="$3"
    if grep -q "$pattern" "$CHESS_AI_DIR/$file"; then
        ok "$description"
    else
        fail "$description" "pattern '$pattern' not found in $file"
    fi
}

src_absent() {
    local description="$1"
    local file="$2"
    local pattern="$3"
    if ! grep -q "$pattern" "$CHESS_AI_DIR/$file"; then
        ok "$description"
    else
        fail "$description" "unwanted pattern '$pattern' found in $file"
    fi
}

# main.py: env var overrides
src_check "main.py: NUM_WORKERS uses os.environ.get"      "game_engine/main.py" 'os.environ.get.*NUM_WORKERS'
src_check "main.py: STOCKFISH_PATH uses os.environ.get"   "game_engine/main.py" 'os.environ.get.*STOCKFISH_PATH'
src_check "main.py: CUDA_BATCH_SIZE uses os.environ.get"  "game_engine/main.py" 'os.environ.get.*CUDA_BATCH_SIZE'
src_check "main.py: ITERATIONS uses os.environ.get"       "game_engine/main.py" 'os.environ.get.*ITERATIONS'

# main.py: dynamic CPU affinity (not hardcoded 44)
src_check  "main.py: uses sched_getaffinity for dynamic CPU count" "game_engine/main.py" 'sched_getaffinity'
src_absent "main.py: no hardcoded % 44 CPU affinity"               "game_engine/main.py" '% 44'

# main.py: tree reuse hooks
src_check "main.py: calls worker.advance_root"    "game_engine/main.py" 'worker\.advance_root'
src_check "main.py: calls worker.reset_cache"     "game_engine/main.py" 'worker\.reset_cache'

# main.py: ELO metrics bug fix (model_elo not elo)
src_check  "main.py: uses 'model_elo' key for ELO lookup"  "game_engine/main.py" '"model_elo"'
src_absent "main.py: no bare 'elo' key lookup"              "game_engine/main.py" 'entry\.get("elo")'

# main.py: passes total_iterations to train_model
src_check "main.py: passes total_iterations to train_model" "game_engine/main.py" 'total_iterations='

# evaluation.py: tree reuse
src_check "evaluation.py: Arena calls advance_root"  "game_engine/evaluation.py" 'advance_root'
src_check "evaluation.py: Arena calls reset_cache"   "game_engine/evaluation.py" 'reset_cache'

# trainer.py: LR scheduler
src_check  "trainer.py: uses CosineAnnealingLR"      "game_engine/trainer.py" 'CosineAnnealingLR'
src_absent "trainer.py: no ExponentialLR"            "game_engine/trainer.py" 'ExponentialLR'
src_check  "trainer.py: saves scheduler_state_dict"  "game_engine/trainer.py" 'scheduler_state_dict'
src_check  "trainer.py: AMP uses torch.amp.autocast" "game_engine/trainer.py" 'torch\.amp\.autocast'

# chess_env.py: 120-plane encoding
src_check "chess_env.py: to_tensor accepts history param" "game_engine/chess_env.py" 'def to_tensor.*history'
src_check "chess_env.py: 120 planes defined"              "game_engine/chess_env.py" '120'

# cnn.py: 120 input channels
src_check "cnn.py: input_channels = 120" "game_engine/cnn.py" 'input_channels = 120'

# =============================================================================
# 14. SLURM SCRIPTS
# =============================================================================
header "14. SLURM Scripts"

slurm_check() {
    local description="$1"
    local file="$2"
    local pattern="$3"
    if grep -qe "$pattern" "$CHESS_AI_DIR/$file"; then
        ok "$description"
    else
        fail "$description" "pattern '$pattern' not found in $file"
    fi
}

# CARC-specific: GPU is selected via --gres=gpu:1 + --constraint, NOT --gres=gpu:a100:1
slurm_check "a100: gres=gpu:1 (CARC syntax)"     "slurm/train_a100.sbatch" 'gres=gpu:1'
slurm_check "a100: constraint targets a100/a40"  "slurm/train_a100.sbatch" 'constraint.*a100'
slurm_check "a100: --account= filled in"         "slurm/train_a100.sbatch" '--account=saifhash_1190'
slurm_check "a100: CARC module ver/2506"         "slurm/train_a100.sbatch" 'module load ver/2506'
slurm_check "a100: CARC cuda module"             "slurm/train_a100.sbatch" 'module load cuda'
slurm_check "a100: requests 32 CPUs"             "slurm/train_a100.sbatch" 'cpus-per-task=32'
slurm_check "a100: CUDA_BATCH_SIZE=8192"         "slurm/train_a100.sbatch" 'CUDA_BATCH_SIZE=8192'
slurm_check "a100: NUM_WORKERS set from CPUs"    "slurm/train_a100.sbatch" 'NUM_WORKERS='
slurm_check "a100: conda activate"               "slurm/train_a100.sbatch" 'conda activate'
slurm_check "a100: cmake build step"             "slurm/train_a100.sbatch" 'cmake'
slurm_check "a100: make -j build"                "slurm/train_a100.sbatch" 'make -j'
slurm_check "a100: copies .so to game_engine"    "slurm/train_a100.sbatch" 'cp mcts_engine_cpp'
slurm_check "a100: runs game_engine/main.py"     "slurm/train_a100.sbatch" 'game_engine/main.py'
slurm_check "a100: --requeue for preemption"     "slurm/train_a100.sbatch" 'requeue'
slurm_check "a100: stderr tee to slurm/"         "slurm/train_a100.sbatch" 'tee.*slurm/stderr'

slurm_check "v100: gres=gpu:1 (CARC syntax)"     "slurm/train_v100.sbatch" 'gres=gpu:1'
slurm_check "v100: constraint targets v100"      "slurm/train_v100.sbatch" 'constraint.*v100'
slurm_check "v100: --account= filled in"         "slurm/train_v100.sbatch" '--account=saifhash_1190'
slurm_check "v100: CARC module ver/2506"         "slurm/train_v100.sbatch" 'module load ver/2506'
slurm_check "v100: CUDA_BATCH_SIZE=4096"         "slurm/train_v100.sbatch" 'CUDA_BATCH_SIZE=4096'
slurm_check "v100: runs game_engine/main.py"     "slurm/train_v100.sbatch" 'game_engine/main.py'
slurm_check "v100: --requeue for preemption"     "slurm/train_v100.sbatch" 'requeue'

# =============================================================================
# SUMMARY
# =============================================================================
TOTAL=$((PASS + FAIL + WARN))
echo ""
printf "${BOLD}==============================================\n"
printf " Summary: %d checks run\n" "$TOTAL"
printf "${G} ✅  Passed: %d${NC}\n" "$PASS"
[[ $FAIL -gt 0 ]] && printf "${R} ❌  Failed: %d${NC}\n" "$FAIL" || printf "${G} ❌  Failed: 0${NC}\n"
[[ $WARN -gt 0 ]] && printf "${Y} ⚠   Warned: %d${NC}\n" "$WARN" || printf " ⚠   Warned: 0\n"
printf "${BOLD}==============================================${NC}\n"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
