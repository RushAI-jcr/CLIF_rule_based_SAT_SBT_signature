#!/usr/bin/env bash
# ==============================================================
# run_site.sh — Site-level pipeline runner
# ==============================================================
# Run this script at each CLIF site to produce all required outputs.
#
# Prerequisites:
#   1. Python 3.9+ with packages in requirements.txt
#   2. config/config.json populated from config/config_template.json
#   3. CLIF 2.1 tables at the path specified in config.json
#
# Usage:
#   cd /path/to/repo
#   bash scripts/run_site.sh
#
# What it does (in order):
#   Step 0: Validates config and environment
#   Step 1: 00_cohort_id.py      — builds study_cohort.parquet
#   Step 2: 01_SAT_standard.py   — SAT phenotyping + concordance
#   Step 3: 02_SBT_Standard.py   — SBT phenotyping + concordance
#   Step 4: 02_SBT variants      — Hemodynamic, Respiratory, Both
#   Step 5: Validates outputs against manifest
# ==============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="$REPO_ROOT/code"
CONFIG="$REPO_ROOT/config/config.json"
OUTPUT_DIR="$REPO_ROOT/output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() { echo -e "\n${GREEN}[STEP $1]${NC} $2"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ----------------------------------------------------------
# Step 0: Validate environment
# ----------------------------------------------------------
log_step 0 "Validating environment..."

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    log_fail "Python not found. Install Python 3.9+."
fi
PYTHON=$(command -v python3 || command -v python)
echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"

if [ ! -f "$CONFIG" ]; then
    log_fail "Config not found at $CONFIG. Copy config_template.json and fill in your site details."
fi

SITE_NAME=$($PYTHON -c "import json; print(json.load(open('$CONFIG'))['site_name'])")
echo "  Site: $SITE_NAME"

TABLES_PATH=$($PYTHON -c "import json; print(json.load(open('$CONFIG'))['tables_path'])")
if [ ! -d "$TABLES_PATH" ]; then
    log_fail "CLIF tables path does not exist: $TABLES_PATH"
fi
echo "  CLIF tables: $TABLES_PATH"

# Check key Python dependencies
$PYTHON -c "import pandas, numpy, scipy, lifelines" 2>/dev/null || \
    log_fail "Missing Python packages. Run: pip install -r requirements.txt"

echo -e "  ${GREEN}Environment OK${NC}"

# ----------------------------------------------------------
# Step 1: Cohort identification
# ----------------------------------------------------------
log_step 1 "Running 00_cohort_id.py (cohort identification)..."
cd "$CODE_DIR"
$PYTHON 00_cohort_id.py
echo "  Checking output..."
if [ ! -f "$OUTPUT_DIR/intermediate/study_cohort.parquet" ] && \
   [ ! -f "$OUTPUT_DIR/intermediate/study_cohort.csv" ]; then
    log_fail "study_cohort not produced. Check 00_cohort_id.py output."
fi
echo -e "  ${GREEN}Cohort built${NC}"

# ----------------------------------------------------------
# Step 2: SAT phenotyping
# ----------------------------------------------------------
log_step 2 "Running 01_SAT_standard.py (SAT phenotyping)..."
$PYTHON 01_SAT_standard.py
if [ ! -f "$OUTPUT_DIR/intermediate/final_df_SAT.csv" ]; then
    log_fail "final_df_SAT.csv not produced. Check 01_SAT_standard.py output."
fi
echo -e "  ${GREEN}SAT phenotyping complete${NC}"

# ----------------------------------------------------------
# Step 3: SBT Standard phenotyping
# ----------------------------------------------------------
log_step 3 "Running 02_SBT_Standard.py (SBT phenotyping)..."
$PYTHON 02_SBT_Standard.py
if [ ! -f "$OUTPUT_DIR/intermediate/final_df_SBT.csv" ]; then
    log_fail "final_df_SBT.csv not produced. Check 02_SBT_Standard.py output."
fi
echo -e "  ${GREEN}SBT phenotyping complete${NC}"

# ----------------------------------------------------------
# Step 4: SBT variants (optional but recommended)
# ----------------------------------------------------------
log_step 4 "Running SBT variant notebooks..."
for variant in "02_SBT_Hemodynamic_Stability.py" \
               "02_SBT_Respiratory_Stability.py" \
               "02_SBT_Both_stabilities.py"; do
    if [ -f "$CODE_DIR/$variant" ]; then
        echo "  Running $variant..."
        $PYTHON "$variant" || log_warn "$variant failed (non-critical)"
    fi
done
echo -e "  ${GREEN}SBT variants complete${NC}"

# ----------------------------------------------------------
# Step 5: Validate outputs
# ----------------------------------------------------------
log_step 5 "Validating site outputs..."

MISSING=0
check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $(basename "$1")"
    else
        echo -e "  ${RED}[MISSING]${NC} $1"
        MISSING=$((MISSING + 1))
    fi
}

echo ""
echo "=== Required Intermediate Files ==="
check_file "$OUTPUT_DIR/intermediate/study_cohort.parquet"
check_file "$OUTPUT_DIR/intermediate/final_df_SAT.csv"
check_file "$OUTPUT_DIR/intermediate/final_df_SBT.csv"

echo ""
echo "=== Required Final Files (SAT) ==="
SAT_DIR="$OUTPUT_DIR/final/$SITE_NAME/SAT_standard"
if [ -d "$SAT_DIR" ]; then
    for f in "$SAT_DIR"/*.csv; do
        [ -f "$f" ] && echo "  [OK] $(basename "$f")"
    done
else
    echo -e "  ${YELLOW}SAT final directory not found: $SAT_DIR${NC}"
fi

echo ""
echo "=== Required Final Files (SBT) ==="
SBT_DIR="$OUTPUT_DIR/final/$SITE_NAME/SBT_standard"
if [ -d "$SBT_DIR" ]; then
    for f in "$SBT_DIR"/*.csv; do
        [ -f "$f" ] && echo "  [OK] $(basename "$f")"
    done
else
    echo -e "  ${YELLOW}SBT final directory not found: $SBT_DIR${NC}"
fi

echo ""
if [ "$MISSING" -gt 0 ]; then
    log_fail "$MISSING required files missing. Fix errors above before sending outputs."
else
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Site pipeline complete for: $SITE_NAME${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review outputs in: $OUTPUT_DIR/"
    echo "  2. Send the following to the coordinating center:"
    echo "     - output/intermediate/study_cohort.parquet"
    echo "     - output/intermediate/final_df_SAT.csv"
    echo "     - output/intermediate/final_df_SBT.csv"
    echo "     - output/final/$SITE_NAME/  (entire directory)"
    echo ""
    echo "  Transfer via your institution's approved secure method."
fi
