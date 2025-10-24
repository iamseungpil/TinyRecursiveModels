#!/bin/bash
#
# HelmARC Phase 1: Adapter Pretraining
#
# Step 1: Precompute GPT-OSS embeddings from analysis
# Step 2: Train adapters with precomputed embeddings (fast)
#

set -e

BASE_DIR=/home/ubuntu/TinyRecursiveModels/hybrid_pipeline
DATA_DIR=/data/helmarc_trm_v3
OUTPUT_DIR=/data/hybrid_training
CHECKPOINT_DIR=$OUTPUT_DIR/phase1
DEVICE=${1:-cuda:6}

echo "=========================================="
echo "HelmARC Phase 1: Adapter Pretraining"
echo "=========================================="
echo "Device: $DEVICE"
echo ""

# Step 1: Precompute embeddings
echo "Step 1/2: Precomputing GPT-OSS embeddings..."
echo "------------------------------------------"

cd $BASE_DIR

python precompute_helmarc_embeddings.py \
    --analysis_dir /data/helmarc/analysis \
    --identifiers_path $DATA_DIR/identifiers.json \
    --output_path /data/helmarc_gptoss_embeddings.pt \
    --model_name openai/gpt-oss-20b \
    --device $DEVICE \
    --trust_remote_code True \
    --use_fast_tokenizer False \
    --torch_dtype bfloat16 \
    --device_map auto

echo ""
echo "✅ Embeddings saved to /data/helmarc_gptoss_embeddings.pt"
echo ""

# Step 2: Train adapters with autoencoder objective
echo "Step 2/2: Training adapters (autoencoder)..."
echo "------------------------------------------"

cd $BASE_DIR/experiments

mkdir -p $CHECKPOINT_DIR

python train_adapter_autoencoder.py \
    --embeddings_path /data/helmarc_gptoss_embeddings.pt \
    --data_path $DATA_DIR \
    --output_dir $CHECKPOINT_DIR \
    --batch_size 128 \
    --epochs 10 \
    --lr 1e-4 \
    --device $DEVICE \
    --use_attention_pooling

echo ""
echo "✅ Autoencoder checkpoint: $CHECKPOINT_DIR/adapter_autoencoder.pt"
echo ""

echo "Step 3/3: Fitting adapters with TRM supervision (augmented data)..."
echo "------------------------------------------"

python train_adapter_with_trm.py \
    --latents_path /data/helmarc_gptoss_embeddings.pt \
    --data_path $DATA_DIR \
    --trm_checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_362650 \
    --adapter_checkpoint $CHECKPOINT_DIR/adapter_autoencoder.pt \
    --output_path $CHECKPOINT_DIR/adapter_trmfit.pt \
    --batch_size 32 \
    --epochs 10 \
    --lr 5e-5 \
    --device $DEVICE \
    --use_augmented

echo ""
echo "=========================================="
echo "✅ Phase 1 completed!"
echo "Checkpoints: $CHECKPOINT_DIR/"
echo "=========================================="
