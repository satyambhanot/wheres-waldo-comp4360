#!/bin/bash
# =============================================================================
# Full Pipeline Runner - Where's Waldo (COMP 4360)
#
# Runs the entire project from scratch: environment setup, data processing,
# code quality checks, training, and evaluation.
#
# Usage:
#   ./scripts/run_pipeline.sh              Full run (200 epochs)
#   ./scripts/run_pipeline.sh --quick      Quick test (10 epochs)
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EPOCHS=200
QUICK=false

if [ "$1" = "--quick" ]; then
    QUICK=true
    EPOCHS=10
fi

step_pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
step_fail() { echo -e "  ${RED}FAIL${NC} $1"; exit 1; }
step_info() { echo -e "\n${BLUE}[$1]${NC} $2"; }

echo ""
echo "========================================"
echo "  Where's Waldo - Full Pipeline"
echo "========================================"
if [ "$QUICK" = true ]; then
    echo -e "  Mode: ${YELLOW}QUICK TEST${NC} ($EPOCHS epochs)"
else
    echo "  Mode: FULL RUN ($EPOCHS epochs)"
fi
echo ""

# ----- Step 1: Check Python -----
step_info "1/10" "Checking Python version..."
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MINOR" -ge 11 ]; then
        step_pass "$PY_VERSION"
    else
        step_fail "Python 3.11+ required, found $PY_VERSION"
    fi
else
    step_fail "python3 not found in PATH"
fi

# ----- Step 2: Check raw images -----
step_info "2/10" "Checking raw data..."
IMG_COUNT=$(ls data/raw/images/*.jpg 2>/dev/null | wc -l | tr -d ' ')
ANN_COUNT=$(ls data/raw/annotations/*.xml 2>/dev/null | wc -l | tr -d ' ')

if [ "$IMG_COUNT" -eq 0 ]; then
    step_fail "No images found in data/raw/images/. See README.md for data setup instructions."
fi
if [ "$ANN_COUNT" -eq 0 ]; then
    step_fail "No annotations found in data/raw/annotations/. See README.md for data setup instructions."
fi
step_pass "$IMG_COUNT images, $ANN_COUNT annotations"

# ----- Step 3: Setup venv -----
step_info "3/10" "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    . venv/bin/activate && pip install --upgrade pip -q
    . venv/bin/activate && pip install -r requirements.txt -q
    step_pass "Created venv and installed dependencies"
else
    step_pass "venv already exists"
fi

# ----- Step 4: Code quality -----
step_info "4/10" "Running code quality checks..."
if . venv/bin/activate && ./scripts/check.sh --fix >/dev/null 2>&1; then
    step_pass "All 8 checks passed"
else
    step_fail "Code quality checks failed. Run: ./scripts/check.sh --fix"
fi

# ----- Step 5: Tiling pipeline -----
step_info "5/10" "Running tiling pipeline..."
. venv/bin/activate && python -m src.preprocess
step_pass "Tiles generated at 64/128/256 px"

# ----- Step 6: Dataset sanity check -----
step_info "6/10" "Verifying dataset..."
. venv/bin/activate && python -m src.dataset
step_pass "Dataset loads correctly"

# ----- Step 7: Model sanity check -----
step_info "7/10" "Verifying model..."
. venv/bin/activate && python -m src.model
step_pass "Model forward pass works"

# ----- Step 8: Training -----
step_info "8/10" "Training baseline model ($EPOCHS epochs)..."
. venv/bin/activate && python -c "
from src.train import train
train(config_overrides={
    'num_epochs': $EPOCHS,
    'resolution': 256,
    'use_sobel': True,
})
"
step_pass "Training complete"

# ----- Step 9: Evaluation -----
step_info "9/10" "Evaluating on test set..."
. venv/bin/activate && python -m src.evaluate
step_pass "Evaluation complete"

# ----- Step 10: Summary -----
step_info "10/10" "Pipeline complete!"
echo ""
echo "========================================"
echo -e "  ${GREEN}All steps completed successfully.${NC}"
echo "========================================"
echo ""
echo "Results:"
echo "  Checkpoints:  results/checkpoints/"
echo "  Plots:        results/plots/"
echo "  Metrics:      (see evaluation output above)"
echo ""
if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}Note: This was a quick test run ($EPOCHS epochs)."
    echo -e "For full training, run without --quick.${NC}"
fi
echo ""
