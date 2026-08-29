#!/bin/bash
# =============================================================================
# run_attention_unet.sh
# Full train → generate → evaluate pipeline for the attention_unet architecture
# 300 epochs, no early stopping.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable defaults (override via CLI flags)
# ---------------------------------------------------------------------------
DATA_DIR="./data_mri/Training"
EPOCHS=300
BATCH_SIZE=16
LR="1e-5"
IMG_SIZE=256
IN_CHANNELS=1
OUT_CHANNELS=1
ATTENTION_HEAD_DIM=8
NUM_IMAGES=1000
DEVICE="cuda:1"
NUM_WORKERS=4
NO_TQDM=true
RESCALE_FLAG="--rescale_from_01"
EVAL_METRICS="all"
EVAL_K=5

# --- Optional RL / Reward flags (all off by default) ---
ENABLE_RL_FINETUNE=false
RL_REWARD_TYPE="deep_cosine_diversity"  # trajectory_efficiency | deep_cosine_diversity
RL_EPOCHS=50
RL_LR="1e-6"
RL_BATCH_SIZE=4
RL_GRAD_TIMESTEPS=10
RL_KL_COEFF=0.05
# Best-of-N at generation time (active when GEN_REWARD_TYPE != none)
GEN_REWARD_TYPE="none"    # none | trajectory_efficiency | deep_cosine_diversity
GEN_BEST_OF_N=1

PYTHON_EXE="/home/e21283/miniconda3/envs/mri-diffuser/bin/python3"
MODEL_TYPE="attention_unet"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir)             DATA_DIR="$2";             shift 2 ;;
        --epochs)               EPOCHS="$2";               shift 2 ;;
        --batch_size)           BATCH_SIZE="$2";           shift 2 ;;
        --lr)                   LR="$2";                   shift 2 ;;
        --img_size)             IMG_SIZE="$2";             shift 2 ;;
        --in_channels)          IN_CHANNELS="$2";          shift 2 ;;
        --out_channels)         OUT_CHANNELS="$2";         shift 2 ;;
        --attention_head_dim)   ATTENTION_HEAD_DIM="$2";   shift 2 ;;
        --num_images)           NUM_IMAGES="$2";           shift 2 ;;
        --device)               DEVICE="$2";               shift 2 ;;
        --num_workers)          NUM_WORKERS="$2";          shift 2 ;;
        --no_tqdm)              NO_TQDM=true;              shift 1 ;;
        --rescale_from_01)      RESCALE_FLAG="--rescale_from_01"; shift 1 ;;
        --eval_metrics)         EVAL_METRICS="$2";         shift 2 ;;
        --eval_k)               EVAL_K="$2";               shift 2 ;;
        # RL / Reward flags
        --enable_rl_finetune)   ENABLE_RL_FINETUNE=true;   shift 1 ;;
        --rl_reward_type)       RL_REWARD_TYPE="$2";        shift 2 ;;
        --rl_epochs)            RL_EPOCHS="$2";             shift 2 ;;
        --rl_lr)                RL_LR="$2";                 shift 2 ;;
        --rl_batch_size)        RL_BATCH_SIZE="$2";         shift 2 ;;
        --rl_grad_timesteps)    RL_GRAD_TIMESTEPS="$2";     shift 2 ;;
        --rl_kl_coeff)          RL_KL_COEFF="$2";           shift 2 ;;
        --gen_reward_type)      GEN_REWARD_TYPE="$2";       shift 2 ;;
        --gen_best_of_n)        GEN_BEST_OF_N="$2";         shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +"%d_%m_%Y_%H_%M")
BASE_OUTPUT_DIR="output/output_${MODEL_TYPE}"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/checkpoints"
GEN_DIR="${BASE_OUTPUT_DIR}/generated_images"
EVAL_DIR="${BASE_OUTPUT_DIR}/evaluation"
RL_CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/rl_checkpoints"
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
TRAIN_LOG_CSV="${LOG_DIR}/training_log_${TIMESTAMP}.csv"
RL_LOG_CSV="${LOG_DIR}/rl_training_log_${TIMESTAMP}.csv"

mkdir -p "$CHECKPOINT_DIR" "$GEN_DIR" "$EVAL_DIR" "$LOG_DIR"

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Tqdm flag
TQDM_FLAG=""
if [ "$NO_TQDM" = true ]; then
    TQDM_FLAG="--no_tqdm"
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  MRI Diffuser — Attention UNet Pipeline"
echo "  $(date)"
echo "============================================================"
echo "  Architecture  : $MODEL_TYPE  (benetraco/brain_ddpm_256)"
echo "  Data dir      : $DATA_DIR"
echo "  Epochs        : $EPOCHS  (no early stopping)"
echo "  Batch size    : $BATCH_SIZE"
echo "  Learning rate : $LR"
echo "  Image size    : ${IMG_SIZE}x${IMG_SIZE}"
echo "  Channels      : in=$IN_CHANNELS  out=$OUT_CHANNELS"
echo "  Device        : $DEVICE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Generated dir : $GEN_DIR"
echo "  Eval dir      : $EVAL_DIR"
echo "  Log           : $LOG_FILE"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1 — Train
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  STEP 1: Training $MODEL_TYPE for $EPOCHS epochs"
echo "------------------------------------------------------------"

$PYTHON_EXE train.py \
    --data_dir          "$DATA_DIR"          \
    --epochs            "$EPOCHS"            \
    --batch_size        "$BATCH_SIZE"        \
    --lr                "$LR"                \
    --img_size          "$IMG_SIZE"          \
    --in_channels       "$IN_CHANNELS"       \
    --out_channels      "$OUT_CHANNELS"      \
    --attention_head_dim "$ATTENTION_HEAD_DIM" \
    --model_type        "$MODEL_TYPE"        \
    --checkpoint_dir    "$CHECKPOINT_DIR"    \
    --log_file          "$TRAIN_LOG_CSV"     \
    --device            "$DEVICE"            \
    --num_workers       "$NUM_WORKERS"       \
    --early_stopping_patience 999999        \
    --early_stopping_min_delta 0.0          \
    $TQDM_FLAG

echo "[OK] Training complete."

# ---------------------------------------------------------------------------
# STEP 2 — Pick best checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  STEP 2: Selecting checkpoint"
echo "------------------------------------------------------------"

LAST_CHECKPOINT=$(ls -vd "${CHECKPOINT_DIR}"/epoch_*/ 2>/dev/null | tail -n 1)
LAST_CHECKPOINT="${LAST_CHECKPOINT%/}"

if [ -z "$LAST_CHECKPOINT" ]; then
    echo "[ERROR] No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "[OK] Using checkpoint: $LAST_CHECKPOINT"

# ---------------------------------------------------------------------------
# STEP 3 — Generate images + base metrics (FID, KID, SSIM, PSNR)
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  STEP 3: Generating $NUM_IMAGES images + base metrics"
echo "------------------------------------------------------------"

# Build optional reward flags for generate.py
GEN_REWARD_FLAGS=""
if [ "$GEN_REWARD_TYPE" != "none" ] && [ "$GEN_BEST_OF_N" -gt 1 ]; then
    GEN_REWARD_FLAGS="--reward_type $GEN_REWARD_TYPE --best_of_n $GEN_BEST_OF_N"
    echo "  [Reward] Best-of-${GEN_BEST_OF_N} with '${GEN_REWARD_TYPE}' enabled for generation."
fi

$PYTHON_EXE generate.py \
    --checkpoint_dir "$LAST_CHECKPOINT" \
    --output_dir     "$GEN_DIR"          \
    --num_images     "$NUM_IMAGES"       \
    --batch_size     "$BATCH_SIZE"       \
    --data_dir       "$DATA_DIR"         \
    --calculate_metrics                  \
    --device         "$DEVICE"           \
    $TQDM_FLAG                           \
    $RESCALE_FLAG                        \
    $GEN_REWARD_FLAGS

echo "[OK] Generation and base metrics complete."

# ---------------------------------------------------------------------------
# STEP 4 — Advanced evaluation (α-precision, β-recall, t-SNE, anomaly, etc.)
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  STEP 4: Advanced evaluation metrics"
echo "------------------------------------------------------------"

$PYTHON_EXE evaluate.py \
    --real_dir   "$DATA_DIR"        \
    --gen_dir    "$GEN_DIR"         \
    --output_dir "$EVAL_DIR"        \
    --metrics    $EVAL_METRICS      \
    --num_images "$NUM_IMAGES"      \
    --k          "$EVAL_K"          \
    --device     "$DEVICE"          \
    --batch_size "$BATCH_SIZE"      \
    --downstream_mode augment       \
    --anomaly_method  both          \
    $TQDM_FLAG

echo "[OK] Advanced evaluation complete."

# ---------------------------------------------------------------------------
# STEP 5 [OPTIONAL] — DDPO RL Fine-tuning
# ---------------------------------------------------------------------------
if [ "$ENABLE_RL_FINETUNE" = true ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "  STEP 5: DDPO RL Fine-tuning (reward=${RL_REWARD_TYPE}, epochs=${RL_EPOCHS})"
    echo "------------------------------------------------------------"

    $PYTHON_EXE finetune_rl.py \
        --checkpoint_dir  "$LAST_CHECKPOINT"    \
        --reward_type     "$RL_REWARD_TYPE"     \
        --rl_epochs       "$RL_EPOCHS"          \
        --lr              "$RL_LR"              \
        --batch_size      "$RL_BATCH_SIZE"      \
        --grad_timesteps  "$RL_GRAD_TIMESTEPS"  \
        --kl_coeff        "$RL_KL_COEFF"        \
        --output_dir      "$RL_CHECKPOINT_DIR"  \
        --log_file        "$RL_LOG_CSV"         \
        --device          "$DEVICE"             \
        $TQDM_FLAG

    echo "[OK] RL fine-tuning complete."
else
    echo ""
    echo "  [INFO] RL Fine-tuning skipped (pass --enable_rl_finetune to enable)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline finished successfully!"
echo "  $(date)"
echo "------------------------------------------------------------"
echo "  Checkpoint      : $LAST_CHECKPOINT"
echo "  Generated images: $GEN_DIR"
echo "  Base metrics    : $GEN_DIR/metrics.json"
echo "  Eval report     : $EVAL_DIR/evaluation_report.json"
echo "  Radar chart     : $EVAL_DIR/metrics_radar.png"
echo "  t-SNE plot      : $EVAL_DIR/tsne_plot.png"
if [ "$ENABLE_RL_FINETUNE" = true ]; then
echo "  RL checkpoints  : $RL_CHECKPOINT_DIR"
echo "  RL training log : $RL_LOG_CSV"
fi
echo "  Full log        : $LOG_FILE"
echo "============================================================"
