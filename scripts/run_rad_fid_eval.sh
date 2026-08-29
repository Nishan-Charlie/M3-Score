#!/bin/bash
# =============================================================================
# run_rad_fid_eval.sh
# Runs full metric evaluation (FID, KID, SSIM, M3, Rad-FID finetuned)
# on UNet-Axial and Attention-UNet-Axial generated images vs BraTS real images.
# =============================================================================

set -e
trap '~/whatsappAutomation/notify.sh "❌ run_rad_fid_eval.sh failed!" || true' ERR

PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
BASE_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models"
REAL_DIR="$BASE_DIR/data_mri/brats_axial"
BACKBONE="$BASE_DIR/output/raddino_segmentation/backbone_final.pth"
CUDA_DEVICE="${1:-cuda:1}"  # Default GPU 1 (23 GB free), override with first arg

LOG_DIR="$BASE_DIR/logs/eval"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")

echo "============================================================"
echo "  Rad-FID Full Evaluation Pipeline"
echo "  $(date)"
echo "  Device: $CUDA_DEVICE"
echo "  Backbone: $BACKBONE"
echo "  Real dir: $REAL_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# EVAL 1: UNet-Axial (1000 generated images)
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  EVAL 1: UNet-Axial"
echo "------------------------------------------------------------"

UNET_GEN="$BASE_DIR/output/output_unet_brats_axial/generated_images"
UNET_EVAL="$BASE_DIR/output/output_unet_brats_axial/evaluation_results"
mkdir -p "$UNET_EVAL"

$PYTHON_EXE evaluation/eval_pipeline.py \
    --real_dir   "$REAL_DIR"   \
    --gen_dir    "$UNET_GEN"   \
    --output_dir "$UNET_EVAL"  \
    --metrics    fid m3 ssim rad_fid \
    --num_images 500 \
    --rad_fid_checkpoint "$BACKBONE" \
    --batch_size 32 \
    --device "$CUDA_DEVICE" \
    --no_tqdm \
    2>&1 | tee "$LOG_DIR/eval_unet_brats_axial_${TIMESTAMP}.log"

echo "[OK] UNet-Axial evaluation complete."

# ---------------------------------------------------------------------------
# EVAL 2: Attention-UNet-Axial (500 generated images)
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  EVAL 2: Attention-UNet-Axial"
echo "------------------------------------------------------------"

ATTN_GEN="$BASE_DIR/output/output_attention_unet_brats_axial/generated_images"
ATTN_EVAL="$BASE_DIR/output/output_attention_unet_brats_axial/evaluation_results"
mkdir -p "$ATTN_EVAL"

$PYTHON_EXE evaluation/eval_pipeline.py \
    --real_dir   "$REAL_DIR"    \
    --gen_dir    "$ATTN_GEN"    \
    --output_dir "$ATTN_EVAL"   \
    --metrics    fid m3 ssim rad_fid \
    --num_images 500 \
    --rad_fid_checkpoint "$BACKBONE" \
    --batch_size 32 \
    --device "$CUDA_DEVICE" \
    --no_tqdm \
    2>&1 | tee "$LOG_DIR/eval_attention_unet_brats_axial_${TIMESTAMP}.log"

echo "[OK] Attention-UNet-Axial evaluation complete."

echo ""
echo "============================================================"
echo "  Both evaluations finished!  $(date)"
echo "  UNet-Axial report    : $UNET_EVAL/evaluation_report.json"
echo "  Attention-UNet report: $ATTN_EVAL/evaluation_report.json"
echo "============================================================"

~/whatsappAutomation/notify.sh "✅ Rad-FID evaluation done! Check $UNET_EVAL and $ATTN_EVAL" || true
