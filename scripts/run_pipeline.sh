#!/bin/bash

# Exit on any error
set -e
trap '~/whatsappAutomation/notify.sh "❌ run_pipeline.sh failed!" || true' ERR

# Default values
DATA_DIR="./data_mri/brats_axial"
EPOCHS=300
NUM_IMAGES=1000
DEVICE="cuda:0"
BATCH_SIZE=8
NO_TQDM=true
MODEL_TYPE="unet"
EARLY_STOPPING_PATIENCE=30
PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"

# Parse arguments correctly
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --num_images) NUM_IMAGES="$2"; shift 2 ;;
        --checkpoint_dir) CUSTOM_CHECKPOINT_DIR="$2"; shift 2 ;;
        --output_dir) CUSTOM_OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --early_stopping_patience) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --no_tqdm) NO_TQDM=true; shift 1 ;;
        --rescale_from_01) RESCALE_FLAG="--rescale_from_01"; shift 1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done


# Define New Reorganized Structure
BASE_OUTPUT_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/output/output_${MODEL_TYPE}_brats_axial"
# mkdir $BASE_OUTPUT_DIR
# CHECKPOINT_DIR="checkpoints"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/checkpoints"
# mkdir $CHECKPOINT_DIR
OUTPUT_DIR="${BASE_OUTPUT_DIR}/generated_images"
# mkdir $OUTPUT_DIR
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
# mkdir $LOG_DIR

# Ensure directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# --- 1. Generate Dynamic Log Filename ---
TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")
PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Redirect all subsequent output to both the console and the log file
exec > >(tee -a "$PIPELINE_LOG") 2>&1

# Prepare tqdm flag for python
TQDM_FLAG=""
if [ "$NO_TQDM" = true ]; then
    TQDM_FLAG="--no_tqdm"
fi

echo "------------------------------------------------"
echo "Log file created: $PIPELINE_LOG"
echo "Starting MRI-Diffuser Pipeline"
echo "Data Directory: $DATA_DIR"
echo "Epochs:         $EPOCHS"
echo "Target Images:  $NUM_IMAGES"
echo "Device:         $DEVICE"
echo "Batch Size:     $BATCH_SIZE"
echo "No TQDM:        $NO_TQDM"
echo "Model Type:     $MODEL_TYPE"
echo "------------------------------------------------"

# 1. Train the model
echo "Step 1: Training the $MODEL_TYPE model..."
$PYTHON_EXE train.py \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --model_type "$MODEL_TYPE" \
    --num_workers 4 \
    $TQDM_FLAG

# 2. Get the last checkpoint using natural version sort
LAST_CHECKPOINT=$(ls -vd "${CHECKPOINT_DIR}"/epoch_*/ 2>/dev/null | tail -n 1)
LAST_CHECKPOINT="${LAST_CHECKPOINT%/}"

if [ -z "$LAST_CHECKPOINT" ]; then
    echo "Error: No checkpoint directory found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Step 2: Using Checkpoint: $LAST_CHECKPOINT"
echo "Generating $NUM_IMAGES images and calculating metrics..."
$PYTHON_EXE generate.py \
    --checkpoint_dir "$LAST_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --num_images "$NUM_IMAGES" \
    --batch_size "$BATCH_SIZE" \
    --data_dir "$DATA_DIR" \
    --calculate_metrics \
    --device "$DEVICE" \
    $TQDM_FLAG \
    $RESCALE_FLAG

echo "------------------------------------------------"
echo "Pipeline execution finished!"
echo "Generated images: $OUTPUT_DIR"
echo "Full log saved to: $PIPELINE_LOG"
echo "------------------------------------------------"

~/whatsappAutomation/notify.sh "✅ run_pipeline.sh finished successfully! Logs: $PIPELINE_LOG" || true