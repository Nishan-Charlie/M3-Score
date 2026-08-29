#!/bin/bash

# Exit on any error
set -e
trap '~/whatsappAutomation/notify.sh "❌ run_finetune_classification.sh failed!" || true' ERR

PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
SCRIPT_NAME="finetune_raddino_classification.py"

CUDA_DEVICE="0" # Default device
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

LOG_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/logs/raddino_classification"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"

# Define Paths
DATASET_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/data_mri/Training"
OUTPUT_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/output/raddino_classification"

echo "------------------------------------------------"
echo "Log file created: $LOG_FILE"
echo "Starting Brain Tumor Classification Fine-Tuning"
echo "------------------------------------------------"

# Redirect all subsequent output to both the console and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Run the enhanced script with hyperparameters
$PYTHON_EXE $SCRIPT_NAME \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 1 \
    --batch_size 16 \
    --lr_backbone 1e-5 \
    --lr_head 1e-4 \
    --patience 10

echo "------------------------------------------------"
echo "Pipeline execution finished!"
echo "Full log saved to: $LOG_FILE"
echo "------------------------------------------------"

~/whatsappAutomation/notify.sh "✅ Brain tumor classification training completed! Logs: $LOG_FILE" || true
