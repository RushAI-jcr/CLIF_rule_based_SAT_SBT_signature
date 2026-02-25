#!/usr/bin/env bash
# ==============================================================
# run_aggregation.sh — Coordinating center aggregation pipeline
# ==============================================================
# Run at the main site after collecting all site outputs.
#
# Prerequisites:
#   1. All site outputs placed under output/ following the structure
#      described in SITE_INSTRUCTIONS.md
#   2. Python 3.9+ with full requirements.txt (including rpy2 for
#      Fine-Gray models)
#
# Usage:
#   cd /path/to/repo
#   bash scripts/run_aggregation.sh
# ==============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="$REPO_ROOT/code"
OUTPUT_DIR="$REPO_ROOT/output"
POOLED_DIR="$OUTPUT_DIR/final/pooled"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() { echo -e "\n${GREEN}[STEP $1]${NC} $2"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

PYTHON=$(command -v python3 || command -v python)

# ----------------------------------------------------------
# Step 0: Verify site data exists
# ----------------------------------------------------------
log_step 0 "Checking for site data..."

SITE_COUNT=0
for site_dir in "$OUTPUT_DIR"/final/*/; do
    site_name=$(basename "$site_dir")
    [ "$site_name" = "pooled" ] && continue
    [ "$site_name" = "esm" ] && continue
    [ "$site_name" = "sensitivity" ] && continue
    SITE_COUNT=$((SITE_COUNT + 1))
    echo "  Found site: $site_name"
done

if [ "$SITE_COUNT" -eq 0 ]; then
    log_fail "No site outputs found in $OUTPUT_DIR/final/. Place site directories there first."
fi
echo "  Total sites: $SITE_COUNT"

# Check intermediate files
for f in "final_df_SAT.csv" "final_df_SBT.csv"; do
    if [ ! -f "$OUTPUT_DIR/intermediate/$f" ]; then
        log_warn "Missing $OUTPUT_DIR/intermediate/$f — some analyses will be limited"
    fi
done

# ----------------------------------------------------------
# Step 1: Aggregate site outputs (07)
# ----------------------------------------------------------
log_step 1 "Running 07_aggregate_sites.py..."
cd "$CODE_DIR"
$PYTHON 07_aggregate_sites.py \
    --data-dir "$OUTPUT_DIR" \
    --output-dir "$POOLED_DIR"
echo -e "  ${GREEN}Aggregation complete${NC}"

# ----------------------------------------------------------
# Step 2: Outcome models (03)
# ----------------------------------------------------------
log_step 2 "Running 03_outcome_models.py..."
if [ -f "$CODE_DIR/03_outcome_models.py" ]; then
    $PYTHON 03_outcome_models.py || log_warn "03_outcome_models.py failed (check data availability)"
else
    log_warn "03_outcome_models.py not found, skipping"
fi

# ----------------------------------------------------------
# Step 3: Hospital variation (04)
# ----------------------------------------------------------
log_step 3 "Running 04_hospital_variation.py..."
if [ -f "$CODE_DIR/04_hospital_variation.py" ]; then
    $PYTHON 04_hospital_variation.py || log_warn "04_hospital_variation.py failed"
else
    log_warn "04_hospital_variation.py not found, skipping"
fi

# ----------------------------------------------------------
# Step 4: Sensitivity analyses (06)
# ----------------------------------------------------------
log_step 4 "Running 06_sensitivity_analyses.py..."
if [ -f "$CODE_DIR/06_sensitivity_analyses.py" ]; then
    $PYTHON 06_sensitivity_analyses.py || log_warn "06_sensitivity_analyses.py failed"
else
    log_warn "06_sensitivity_analyses.py not found, skipping"
fi

# ----------------------------------------------------------
# Step 5: Manuscript figures (05)
# ----------------------------------------------------------
log_step 5 "Running 05_manuscript_figures.py..."
if [ -f "$CODE_DIR/05_manuscript_figures.py" ]; then
    $PYTHON 05_manuscript_figures.py || log_warn "05_manuscript_figures.py failed"
else
    log_warn "05_manuscript_figures.py not found, skipping"
fi

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Aggregation pipeline complete${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Outputs:"
echo "  Pooled results:  $POOLED_DIR/"
echo "  ESM tables:      $OUTPUT_DIR/final/esm/"
echo "  Figures:         $OUTPUT_DIR/final/figures/"
echo ""
echo "Key files for manuscript:"
echo "  - $POOLED_DIR/manuscript_numbers.json"
echo "  - $POOLED_DIR/pooled_delivery_rates.csv"
echo "  - $POOLED_DIR/pooled_concordance.csv"
echo "  - $POOLED_DIR/consort_numbers.json"
