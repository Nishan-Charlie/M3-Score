#!/bin/bash
# Full Experimental Evaluation Pipeline
# ======================================
# 1. Regenerate clean images (with fixed pipeline)
# 2. Run all experiments
# 3. Generate summary report

set -e
trap 'echo "❌ Experiment pipeline failed!" && exit 1' ERR

PYTHON="python3"
REAL_DIR="data_mri/brats_axial"
GEN_DIR="output/generated_1000_clean"
OUTPUT_DIR="results/experiments_output"
DEVICE="cuda:0"
NUM_IMAGES=500
TIMESTAMP=$(date +"%Y%m%d_%H%M")

echo "╔══════════════════════════════════════════════╗"
echo "║  Experimental Evaluation Pipeline            ║"
echo "║  Started: $(date)           ║"
echo "╚══════════════════════════════════════════════╝"

# Step 0: Check if clean images exist, regenerate if not
if [ ! -d "$GEN_DIR" ] || [ $(ls "$GEN_DIR"/*.png 2>/dev/null | wc -l) -lt 500 ]; then
    echo ""
    echo "Step 0: Regenerating clean images..."
    bash scripts/regenerate_clean.sh
fi

# Step 1: Run all experiments
echo ""
echo "Step 1: Running all experiments..."
$PYTHON evaluation/run_experiments.py \
    --real_dir "$REAL_DIR" \
    --gen_dir "$GEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --experiments all \
    --device "$DEVICE" \
    --num_images "$NUM_IMAGES" \
    --K 10 \
    --noise_num_images 200 \
    --no_tqdm

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                          ║"  
echo "║  Results: $OUTPUT_DIR                        ║"
echo "╚══════════════════════════════════════════════╝"
