#!/bin/bash
# Reproduce neural analysis (single-unit + fMRI + EEG)
#
# Prerequisites:
#   - Steinmetz Neuropixels data cached locally
#   - ABIDE fMRI data preprocessed
#   - EEGBCI data downloaded
#
# Usage: bash scripts/run_neural.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Neural Analysis Pipeline ==="
echo "Repository: $REPO_DIR"
echo ""

# Verify data files exist
echo "Checking data files..."
for f in data/su_cohort.csv data/fmri_cohort.csv data/eeg_sensor_cov_summary.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done
echo ""

# Run reproduction notebook
echo "Running reproduction notebook..."
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/01_reproduce_results.ipynb \
    --output 01_reproduce_results_executed.ipynb

echo ""
echo "=== Neural analysis complete ==="
echo "Results in: notebooks/01_reproduce_results_executed.ipynb"
