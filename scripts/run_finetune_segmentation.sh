#!/bin/bash

# Exit on any error
set -e
trap '~/whatsappAutomation/notify.sh "❌ run_finetune_segmentation.sh failed!" || true' ERR

PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
SCRIPT_NAME="finetune_raddino_segmentation.py"

CUDA_DEVICE="0" # Default device

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda|--device) CUDA_DEVICE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

LOG_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models/logs/raddino_segmentation"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"

echo "------------------------------------------------"
echo "Log file created: $LOG_FILE"
echo "Starting BraTS Segmentation Fine-Tuning"
echo "------------------------------------------------"

# Redirect all subsequent output to both the console and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

$PYTHON_EXE $SCRIPT_NAME

echo "------------------------------------------------"
echo "Pipeline execution finished!"
echo "Full log saved to: $LOG_FILE"
echo "------------------------------------------------"

~/whatsappAutomation/notify.sh "✅ BraTS segmentation training completed! Logs: $LOG_FILE" || true
