#!/bin/bash
# Regenerate clean images from the best checkpoint with the fixed pipeline
# The rescale_from_01 bug has been fixed — no longer applied by default

set -e

PYTHON="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
CHECKPOINT_DIR="test_checkpoints_unet/best"
OUTPUT_DIR="output/generated_1000_clean"
NUM_IMAGES=1000
BATCH_SIZE=16
DEVICE="cuda:0"

echo "============================================="
echo "  Regenerating Clean Images"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Images:     $NUM_IMAGES"
echo "============================================="

# Remove old generated images if they exist
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing old output: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

$PYTHON generate.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_images "$NUM_IMAGES" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --no_tqdm

echo ""
echo "============================================="
echo "  Verifying generated images..."
echo "============================================="

# Quick quality check
$PYTHON -c "
from PIL import Image
import numpy as np
import glob, os

paths = sorted(glob.glob('$OUTPUT_DIR/*.png'))
print(f'Total images: {len(paths)}')

dark = 0
for p in paths:
    arr = np.array(Image.open(p))
    if arr.mean() < 10:
        dark += 1

print(f'Near-black images: {dark}')
if dark > len(paths) * 0.1:
    print('WARNING: >10% dark images — generation may still have issues')
else:
    print('SUCCESS: Images look good!')
"

echo ""
echo "Done! Clean images saved to: $OUTPUT_DIR"
