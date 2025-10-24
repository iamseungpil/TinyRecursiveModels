#!/bin/bash
#
# HelmARC Phase 2: Joint Training (End-to-End)
#
# LLaMA (frozen) + Adapters (trainable) + TRM (trainable)
# Self-correction loop with 4 attempts
#

set -e

DATA_DIR=/data/helmarc_trm_v3
TRM_CKPT=/data/trm/checkpoints/pretrain_att_arc1concept_4/step_362650
OUTPUT_DIR=/data/hybrid_training
PHASE1_CKPT=${1:-$OUTPUT_DIR/phase1/adapter_trm_supervised.pt}
DEVICE=${2:-cuda:6}

echo "=========================================="
echo "HelmARC Phase 2: Joint Training"
echo "=========================================="
echo "Phase 1 checkpoint: $PHASE1_CKPT"
echo "Device: $DEVICE"
echo ""

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "❌ Error: Phase 1 checkpoint not found!"
    echo "Expected: $PHASE1_CKPT"
    echo ""
    echo "Please run Phase 1 first:"
    echo "  ./run_helmarc_phase1.sh $DEVICE"
    exit 1
fi

cd /home/ubuntu/TinyRecursiveModels/hybrid_pipeline/experiments

python run_joint_training.py \
    --data_path $DATA_DIR \
    --output_dir $OUTPUT_DIR/phase2_joint \
    --trm_checkpoint $TRM_CKPT \
    --adapter_checkpoint $PHASE1_CKPT \
    --use_attention_pooling True \
    --max_attempts 4 \
    --batch_size 4 \
    --epochs 10 \
    --lr 5e-5 \
    --device $DEVICE \
    --llama_model openai/gpt-oss-20b \
    --use_precomputed_embeddings True \
    --embeddings_path /data/helmarc_gptoss_embeddings.pt

echo ""
echo "=========================================="
echo "✅ Phase 2 completed!"
echo "Checkpoints: $OUTPUT_DIR/phase2_joint/"
echo "=========================================="
