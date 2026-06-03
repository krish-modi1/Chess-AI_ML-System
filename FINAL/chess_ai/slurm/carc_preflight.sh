#!/bin/bash
# =============================================================================
# carc_preflight.sh — Pre-submission verification for USC CARC Discovery
#
# Run this on the CARC LOGIN NODE before sbatch to confirm:
#   • Your account name and allocation
#   • GPU availability and correct constraint syntax
#   • Module versions
#   • Conda environment
#   • Model checkpoint presence
#   • sbatch script has --account filled in
#
# Usage (on CARC login node):
#   bash ~/EE542-Project/FINAL/chess_ai/slurm/carc_preflight.sh
# =============================================================================

set -uo pipefail

CHESS_AI_DIR="${CHESS_AI_DIR:-$HOME/EE542-Project/FINAL/chess_ai}"

PASS=0; FAIL=0; WARN=0

ok()   { printf "  ✅  %s\n" "$1"; PASS=$((PASS+1)); }
fail() { printf "  ❌  %s\n" "$1"; [[ -n "${2:-}" ]] && printf "     ↳ %s\n" "$2"; FAIL=$((FAIL+1)); }
warn() { printf "  ⚠   %s\n" "$1"; [[ -n "${2:-}" ]] && printf "     ↳ %s\n" "$2"; WARN=$((WARN+1)); }
header() { printf "\n── %s\n" "$1"; }

# =============================================================================
header "1. SLURM account"
# =============================================================================

ACCOUNT=$(sacctmgr show user "$USER" format=account --noheader -P 2>/dev/null | head -1 | tr -d '[:space:]')
if [[ -n "$ACCOUNT" ]]; then
    ok "Account: $ACCOUNT"
    echo "     → Add this to --account= in your sbatch scripts"
else
    fail "Could not detect SLURM account" "run: sacctmgr show user \$USER"
fi

# Check sbatch scripts have account filled in
for script in train_a100.sbatch train_v100.sbatch; do
    f="$CHESS_AI_DIR/slurm/$script"
    if [[ ! -f "$f" ]]; then
        warn "$script not found" "expected at $f"
    elif grep -q 'YOUR_ACCOUNT' "$f"; then
        fail "$script: --account=YOUR_ACCOUNT not filled in" \
             "edit slurm/$script and replace YOUR_ACCOUNT with: $ACCOUNT"
    else
        ok "$script: --account is set"
    fi
done

# =============================================================================
header "2. GPU partition and constraints"
# =============================================================================

if sinfo -p gpu --noheader &>/dev/null; then
    ok "gpu partition exists"
    echo ""
    echo "  Available GPU nodes (gpu partition):"
    sinfo -p gpu -o "  %10N  %10G  %12f  %5a  %5t" | head -20
else
    fail "gpu partition not found" "run: sinfo -p gpu"
fi

echo ""
echo "  GPU feature/constraint strings (use in --constraint=):"
sinfo -p gpu -o "  %f" --noheader 2>/dev/null | sort -u | head -20

# =============================================================================
header "3. Module availability"
# =============================================================================

check_module() {
    local mod="$1"
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
        ok "module $mod available"
    else
        fail "module $mod NOT found" "run: module avail $mod"
    fi
}

module purge 2>/dev/null
module load ver/2506 2>/dev/null && ok "module ver/2506 loaded" || fail "module ver/2506 not available"

check_module "gcc/14.3.0"
check_module "cuda/12.9.1"
check_module "cmake/3.29.4"

# Check stockfish
if module load stockfish 2>/dev/null && command -v stockfish &>/dev/null; then
    ok "stockfish module available: $(which stockfish)"
else
    warn "stockfish module not found" "evaluation phase will be skipped during training"
fi

# =============================================================================
header "4. Python / conda environment"
# =============================================================================

if command -v conda &>/dev/null; then
    ok "conda available: $(conda --version)"
    if conda env list 2>/dev/null | grep -q "^chessai "; then
        ok "chessai conda env exists"
        PYVER=$(conda run -n chessai python3 --version 2>/dev/null)
        ok "Python in chessai: $PYVER"
    else
        warn "chessai conda env NOT found" \
             "sbatch script will auto-create it from requirements.txt (adds ~5 min first run)"
    fi
else
    warn "conda not available on login node" \
         "sbatch script will try 'module load conda' — verify it exists"
    if module avail conda 2>&1 | grep -q "conda"; then
        ok "conda module found"
    else
        fail "conda module not found either" "contact CARC support: carc.usc.edu/user-information/ticket-submission"
    fi
fi

# Check pybind11 in env
if conda env list 2>/dev/null | grep -q "^chessai "; then
    if conda run -n chessai python3 -c "import pybind11; print(pybind11.get_cmake_dir())" &>/dev/null; then
        ok "pybind11 installed in chessai env"
    else
        fail "pybind11 NOT installed in chessai env" \
             "conda run -n chessai pip install pybind11"
    fi
fi

# =============================================================================
header "5. Model checkpoint"
# =============================================================================

MODEL_PATH="$CHESS_AI_DIR/game_engine/model/best_model.pth"
if [[ -f "$MODEL_PATH" ]]; then
    SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    ok "best_model.pth found ($SIZE)"

    # Verify it's 120-channel
    CHAN=$(conda run -n chessai python3 - <<'PYCHECK' 2>/dev/null
import torch, sys
ckpt = torch.load("'"$MODEL_PATH"'", map_location="cpu", weights_only=True)
w = ckpt.get("model_state_dict", {}).get("input_conv.0.weight")
print(w.shape[1] if w is not None else "MISSING")
PYCHECK
    )
    if [[ "$CHAN" == "120" ]]; then
        ok "model is 120-channel (correct)"
    elif [[ "$CHAN" == "16" ]]; then
        fail "model is still 16-channel — run the migration script locally first"
    else
        warn "could not verify channel count (got: $CHAN)"
    fi
else
    fail "best_model.pth NOT found at $MODEL_PATH"
    cat <<EOF

  Upload from your local machine with:
    scp FINAL/chess_ai/game_engine/model/best_model.pth \\
        krishmod@discovery.usc.edu:$MODEL_PATH
EOF
fi

# =============================================================================
header "6. Project directory structure"
# =============================================================================

for d in game_engine slurm logs; do
    if [[ -d "$CHESS_AI_DIR/$d" ]]; then
        ok "directory $d exists"
    else
        fail "directory $d missing" "expected $CHESS_AI_DIR/$d"
    fi
done

for f in game_engine/main.py game_engine/CMakeLists.txt requirements.txt; do
    if [[ -f "$CHESS_AI_DIR/$f" ]]; then
        ok "file $f exists"
    else
        fail "file $f missing"
    fi
done

# =============================================================================
header "7. Disk quota"
# =============================================================================

echo "  Home directory usage:"
df -h "$HOME" | awk 'NR==2 {printf "  %s used of %s (%s)\n", $3, $2, $5}'
lfs quota -u "$USER" "$HOME" 2>/dev/null || true

# =============================================================================
printf "\n══════════════════════════════════════════\n"
printf " Preflight summary: %d pass  %d fail  %d warn\n" "$PASS" "$FAIL" "$WARN"
printf "══════════════════════════════════════════\n"

if [[ $FAIL -gt 0 ]]; then
    printf " Fix failures before submitting.\n"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    printf " Warnings present — review before submitting.\n"
    exit 0
else
    printf " All checks passed. Ready to sbatch.\n"
    exit 0
fi
