#!/bin/bash
#
# TRM Pretraining Script
#
# Usage: ./run_trm_pretrain.sh [DATA_PATH] [OUTPUT_DIR] [DEVICE]
#
# Example:
#   ./run_trm_pretrain.sh /data/arc/processed /data/trm/pretrain cuda:0

set -e

# Default values
DATA_PATH="${1:-/data/arc/processed}"
OUTPUT_DIR="${2:-/data/trm/pretrain}"
DEVICE="${3:-cuda:0}"
BATCH_SIZE="${4:-32}"
EPOCHS="${5:-50}"

echo "========================================="
echo "TRM Pretraining"
echo "========================================="
echo "Data path:    $DATA_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Device:       $DEVICE"
echo "Batch size:   $BATCH_SIZE"
echo "Epochs:       $EPOCHS"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
cd "$(dirname "$0")/../trm_pretrain"

python train_trm.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    2>&1 | tee "$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ TRM pretraining complete!"
echo "   Checkpoints saved in: $OUTPUT_DIR"
