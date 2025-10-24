#!/bin/bash
#
# HelmARC Phase 3: Self-Correction Fine-tuning (Repulsion)
#
# Enable repulsion mechanism to avoid failed latents
# Maximum 16 self-correction attempts
#

set -e

ARC_DATA_DIR=/data/arc/training_processed
TRM_CKPT=/data/trm/checkpoints/pretrain_att_arc1concept_4/step_362650
OUTPUT_DIR=/data/hybrid_training
PHASE2_CKPT=${1:-$OUTPUT_DIR/phase2_joint/checkpoint_step_10000.pt}
DEVICE=${2:-cuda:6}

echo "=========================================="
echo "HelmARC Phase 3: Repulsion Training"
echo "=========================================="
echo "Phase 2 checkpoint: $PHASE2_CKPT"
echo "Device: $DEVICE"
echo ""

if [ ! -f "$PHASE2_CKPT" ]; then
    echo "❌ Error: Phase 2 checkpoint not found!"
    echo "Expected: $PHASE2_CKPT"
    echo ""
    echo "Please run Phase 2 first:"
    echo "  ./run_helmarc_phase2.sh <phase1_ckpt> $DEVICE"
    exit 1
fi

cd /home/ubuntu/TinyRecursiveModels/hybrid_pipeline/experiments

python run_joint_training.py \
    --data_path $ARC_DATA_DIR \
    --output_dir $OUTPUT_DIR/phase3_repulsion \
    --trm_checkpoint $TRM_CKPT \
    --load_checkpoint $PHASE2_CKPT \
    --use_attention_pooling True \
    --enable_repulsion True \
    --repulsion_weight 0.5 \
    --max_attempts 16 \
    --batch_size 2 \
    --epochs 5 \
    --lr 1e-5 \
    --log_interval 5 \
    --checkpoint_interval 100 \
    --device $DEVICE \
    --project_name arc-phase3-repulsion \
    --run_name helmarc_repulsion_training_$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "✅ Phase 3 completed!"
echo "Checkpoints: $OUTPUT_DIR/phase3_repulsion/"
echo "=========================================="
