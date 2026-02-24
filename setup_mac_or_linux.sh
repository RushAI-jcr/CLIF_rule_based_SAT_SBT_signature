#!/usr/bin/env bash
# setup.sh — Bash version of setup.bat
# Stop on first error
set -e

# ── ANSI colour codes ──────────────────────────────────────────────────────────
YELLOW="\033[33m"
CYAN="\033[36m"
GREEN="\033[32m"
RESET="\033[0m"

separator() { echo -e "${YELLOW}==================================================${RESET}"; }

# ── 1. Create virtual environment ──────────────────────────────────────────────
separator
echo -e "${CYAN}Creating virtual environment (.SBT)…${RESET}"
python -m venv .SBT

# ── 2. Activate virtual environment ────────────────────────────────────────────
separator
echo -e "${CYAN}Activating virtual environment…${RESET}"
# shellcheck source=/dev/null
source .SBT/bin/activate   # (Unix path; note the forward slashes)

# ── 3. Upgrade pip ─────────────────────────────────────────────────────────────
separator
echo -e "${CYAN}Upgrading pip…${RESET}"
python -m pip install --upgrade pip

# ── 4. Install required packages ───────────────────────────────────────────────
separator
echo -e "${CYAN}Installing required packages…${RESET}"
pip install -r requirements.txt

# ── 5. Install Jupyter + IPython kernel ────────────────────────────────────────
separator
echo -e "${CYAN}Installing Jupyter and IPykernel…${RESET}"
pip install jupyter ipykernel

separator
echo -e "${CYAN}Registering the virtual environment as a Jupyter kernel…${RESET}"
python -m ipykernel install --user --name ".SBT" --display-name "Python (SBT 2025)"

# ── 6. Change to code directory ────────────────────────────────────────────────
separator
echo -e "${CYAN}Changing directory to code folder…${RESET}"
cd code || { echo "❌  'code' directory not found."; exit 1; }

echo -e "${CYAN}==================================================${RESET}"
echo -e "${CYAN}IMPORTANT: Please run the notebook named 00_* first to generate the cohort.${RESET}"
echo -e "${CYAN}Then, run all 01_* and 02_* notebooks to generate results. These can be triggered/run in parallel.${RESET}"
echo -e "${CYAN}==================================================${RESET}"
