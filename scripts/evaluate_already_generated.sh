#!/bin/bash

# evaluate_already_generated.sh
# A convenient script to evaluate an existing folder of generated MRI images using evaluate.py.
# This script computes standard metrics, including the newly added M3-Score ("m3").

# Exit on any error
set -e

# Real images path
REAL_DIR="datasets/brats_axial"
PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"

# # Check arguments
# if [ -z "$1" ]; then
#     echo "Usage: ./evaluate_already_generated.sh <path_to_generated_images_dir> [num_images]"
#     echo "Example: ./evaluate_already_generated.sh output/generated_with_metrics 100"
#     exit 1
# fi

GEN_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/output/output_unet_brats_axial/generated_images"
NUM_IMAGES="${2:-1000}"  # Default to 100 if not provided
OUTPUT_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/output/output_unet_brats_axial/evaluation_results"

echo "--------------------------------------------------------"
echo "Evaluating already generated images"
echo "Generated Directory: $GEN_DIR"
echo "Real Directory:      $REAL_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Number of Images:    $NUM_IMAGES"
echo "Metrics to run:      fid, ssim, m3"
echo "--------------------------------------------------------"

cuda=1
$PYTHON_EXE evaluate.py \
    --real_dir "$REAL_DIR" \
    --gen_dir "$GEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --metrics fid ssim m3 \
    --num_images "$NUM_IMAGES" \
    --device cuda:$cuda

echo "--------------------------------------------------------"
echo "Evaluation complete! Results are saved in: $OUTPUT_DIR/evaluation_report.json"
