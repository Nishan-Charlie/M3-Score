#!/bin/bash
# =============================================================================
# run_multislice_pipeline.sh
# Extracts multi-slice BraTS data (slices 55-85) and retrains both
# UNet-Axial and Attention-UNet-Axial models on GPU 0.
# =============================================================================
# // turbo-all

set -e
trap '~/whatsappAutomation/notify.sh "❌ run_multislice_pipeline.sh failed at step: $CURRENT_STEP" || true' ERR

PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
BASE_DIR="/home/e21283/mediGAN/mri-diffuser/huggingface_models"
DEVICE="cuda:0"
CURRENT_STEP="init"

LOG_DIR="$BASE_DIR/logs/multislice_pipeline"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  Multi-Slice BraTS Retraining Pipeline"
echo "  $(date)"
echo "  Device: $DEVICE"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1 — Extract slices 55–85 from 3D BraTS NIfTI volumes
# ---------------------------------------------------------------------------
CURRENT_STEP="slice_extraction"
echo ""
echo "------------------------------------------------------------"
echo "  STEP 1: Extracting BraTS slices 55–85 (FLAIR, axial)"
echo "------------------------------------------------------------"

BRATS_3D_DIR="$BASE_DIR/data_mri/brats"
MULTISLICE_DIR="$BASE_DIR/data_mri/brats_axial_multislice"
mkdir -p "$MULTISLICE_DIR"

# Check if already extracted
EXISTING=$(ls "$MULTISLICE_DIR"/*.png 2>/dev/null | wc -l || echo 0)
if [ "$EXISTING" -gt 10000 ]; then
    echo "[SKIP] Found $EXISTING slices already in $MULTISLICE_DIR. Skipping extraction."
else
    echo "Extracting slices 55–85 from $BRATS_3D_DIR..."
    $PYTHON_EXE prepare_brats_slices.py \
        --data_dir  "$BRATS_3D_DIR"      \
        --output_dir "$MULTISLICE_DIR"   \
        --modality  flair                \
        --axis      2                    \
        --threshold 0.05                 \
        --depth     55 85
    echo "[OK] Extraction complete."
fi

SLICE_COUNT=$(ls "$MULTISLICE_DIR"/*.png 2>/dev/null | wc -l || echo 0)
echo "  Total slices available: $SLICE_COUNT"

# ---------------------------------------------------------------------------
# STEP 2 — Train UNet on multi-slice data (300 epochs)
# ---------------------------------------------------------------------------
CURRENT_STEP="train_unet"
echo ""
echo "------------------------------------------------------------"
echo "  STEP 2: Training UNet on multi-slice BraTS data"
echo "------------------------------------------------------------"

~/whatsappAutomation/notify.sh "🚀 Starting UNet multi-slice training (${SLICE_COUNT} slices, 300 epochs)" || true

UNET_OUTPUT="$BASE_DIR/output/output_unet_brats_multislice"
mkdir -p "$UNET_OUTPUT/checkpoints" "$UNET_OUTPUT/logs"

$PYTHON_EXE train.py \
    --data_dir          "$MULTISLICE_DIR"              \
    --model_type        unet                           \
    --epochs            300                            \
    --batch_size        16                             \
    --lr                1e-5                           \
    --img_size          256                            \
    --in_channels       1                              \
    --out_channels      1                              \
    --checkpoint_dir    "$UNET_OUTPUT/checkpoints"     \
    --log_file          "$UNET_OUTPUT/logs/train_log_${TIMESTAMP}.csv" \
    --device            "$DEVICE"                      \
    --num_workers       4                              \
    --early_stopping_patience 999999                   \
    --early_stopping_min_delta 0.0                     \
    --no_tqdm

echo "[OK] UNet training complete."
~/whatsappAutomation/notify.sh "✅ UNet multi-slice training done! Starting Attention-UNet..." || true

# ---------------------------------------------------------------------------
# STEP 3 — Train Attention-UNet on multi-slice data (300 epochs)
# ---------------------------------------------------------------------------
CURRENT_STEP="train_attention_unet"
echo ""
echo "------------------------------------------------------------"
echo "  STEP 3: Training Attention-UNet on multi-slice BraTS data"
echo "------------------------------------------------------------"

ATTN_OUTPUT="$BASE_DIR/output/output_attention_unet_brats_multislice"
mkdir -p "$ATTN_OUTPUT/checkpoints" "$ATTN_OUTPUT/logs"

$PYTHON_EXE train.py \
    --data_dir          "$MULTISLICE_DIR"              \
    --model_type        attention_unet                 \
    --epochs            300                            \
    --batch_size        16                             \
    --lr                1e-5                           \
    --img_size          256                            \
    --in_channels       1                              \
    --out_channels      1                              \
    --attention_head_dim 8                             \
    --checkpoint_dir    "$ATTN_OUTPUT/checkpoints"     \
    --log_file          "$ATTN_OUTPUT/logs/train_log_${TIMESTAMP}.csv" \
    --device            "$DEVICE"                      \
    --num_workers       4                              \
    --early_stopping_patience 999999                   \
    --early_stopping_min_delta 0.0                     \
    --no_tqdm

echo "[OK] Attention-UNet training complete."
~/whatsappAutomation/notify.sh "✅ Attention-UNet multi-slice training done! Starting generation + eval..." || true

# ---------------------------------------------------------------------------
# STEP 4 — Generate 500 images from each model
# ---------------------------------------------------------------------------
CURRENT_STEP="generate"
echo ""
echo "------------------------------------------------------------"
echo "  STEP 4: Generating 500 images from each model"
echo "------------------------------------------------------------"

# Pick last checkpoint for UNet
UNET_CKPT=$(ls -vd "$UNET_OUTPUT/checkpoints"/epoch_*/ 2>/dev/null | tail -1 | sed 's|/$||')
ATTN_CKPT=$(ls -vd "$ATTN_OUTPUT/checkpoints"/epoch_*/ 2>/dev/null | tail -1 | sed 's|/$||')

echo "  UNet checkpoint   : $UNET_CKPT"
echo "  Attn-UNet checkpoint: $ATTN_CKPT"

UNET_GEN="$UNET_OUTPUT/generated_images"
ATTN_GEN="$ATTN_OUTPUT/generated_images"

$PYTHON_EXE generate.py \
    --checkpoint_dir "$UNET_CKPT" \
    --output_dir     "$UNET_GEN"  \
    --num_images     500          \
    --batch_size     16           \
    --data_dir       "$MULTISLICE_DIR" \
    --device         "$DEVICE"    \
    --no_tqdm        \
    --rescale_from_01

$PYTHON_EXE generate.py \
    --checkpoint_dir "$ATTN_CKPT" \
    --output_dir     "$ATTN_GEN"  \
    --num_images     500          \
    --batch_size     16           \
    --data_dir       "$MULTISLICE_DIR" \
    --device         "$DEVICE"    \
    --no_tqdm        \
    --rescale_from_01

echo "[OK] Generation complete."

# ---------------------------------------------------------------------------
# STEP 5 — Evaluate both models (FID, M3, SSIM, Rad-FID)
# ---------------------------------------------------------------------------
CURRENT_STEP="evaluate"
echo ""
echo "------------------------------------------------------------"
echo "  STEP 5: Evaluating both models"
echo "------------------------------------------------------------"

BACKBONE="$BASE_DIR/output/raddino_segmentation/backbone_final.pth"
REAL_DIR="$MULTISLICE_DIR"   # use multi-slice real data as reference!

for MODEL_NAME in unet attention_unet; do
    if [ "$MODEL_NAME" = "unet" ]; then
        GEN_DIR="$UNET_GEN"
        EVAL_DIR="$UNET_OUTPUT/evaluation_results"
    else
        GEN_DIR="$ATTN_GEN"
        EVAL_DIR="$ATTN_OUTPUT/evaluation_results"
    fi

    mkdir -p "$EVAL_DIR"
    echo "  Evaluating $MODEL_NAME..."

    $PYTHON_EXE evaluation/eval_pipeline.py \
        --real_dir   "$REAL_DIR"   \
        --gen_dir    "$GEN_DIR"    \
        --output_dir "$EVAL_DIR"   \
        --metrics    fid m3 ssim rad_fid \
        --num_images 500 \
        --rad_fid_checkpoint "$BACKBONE" \
        --batch_size 32 \
        --device     "$DEVICE" \
        --no_tqdm
    echo "  [OK] $MODEL_NAME evaluation done → $EVAL_DIR/evaluation_report.json"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Full pipeline finished!  $(date)"
echo "  Multi-slice data   : $MULTISLICE_DIR ($SLICE_COUNT slices)"
echo "  UNet results       : $UNET_OUTPUT/evaluation_results/evaluation_report.json"
echo "  Attn-UNet results  : $ATTN_OUTPUT/evaluation_results/evaluation_report.json"
echo "  Full log           : $LOG_FILE"
echo "============================================================"

~/whatsappAutomation/notify.sh "🎉 Full multi-slice pipeline complete! Check logs at: $LOG_FILE" || true
