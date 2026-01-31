#!/bin/bash
# Reproduce AI (GPT-2) SPD analysis
#
# Prerequisites:
#   - Python environment with torch, transformers, sklearn
#   - GPU recommended but not required (CPU works, slower)
#
# Usage: bash scripts/run_ai.sh [--synthetic]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== AI SPD Pipeline ==="
echo "Repository: $REPO_DIR"
echo ""

# Check for --synthetic flag
SYNTHETIC=""
if [[ "${1:-}" == "--synthetic" ]]; then
    SYNTHETIC="--synthetic"
    echo "Mode: Synthetic activations (no GPU required)"
else
    echo "Mode: Real GPT-2 activations"
fi
echo ""

# Verify pre-computed results exist
echo "Checking pre-computed results..."
for f in data/corrected_gpt2_results.json data/robustness_results.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done
echo ""

# Run the AI pipeline
echo "Running AI SPD pipeline..."
python -m src.ai_spd_pipeline $SYNTHETIC

echo ""
echo "=== AI analysis complete ==="
