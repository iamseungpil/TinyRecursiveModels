#!/bin/bash
#
# TRM Evaluation Script
#
# Usage: ./run_trm_eval.sh [CHECKPOINT] [DATA_PATH] [DEVICE]
#
# Example:
#   ./run_trm_eval.sh /data/trm/pretrain/checkpoint_step_5000.pt /data/arc/processed cuda:0

set -e

# Arguments
CHECKPOINT="${1:?Error: Checkpoint path required}"
DATA_PATH="${2:-/data/arc/processed}"
DEVICE="${3:-cuda:0}"
OUTPUT_FILE="${4:-/data/trm/pretrain/eval_results.json}"

echo "========================================="
echo "TRM Evaluation"
echo "========================================="
echo "Checkpoint:   $CHECKPOINT"
echo "Data path:    $DATA_PATH"
echo "Device:       $DEVICE"
echo "Output:       $OUTPUT_FILE"
echo "========================================="

# Run evaluation
cd "$(dirname "$0")/../trm_pretrain"

python eval_trm.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --device "$DEVICE" \
    --output "$OUTPUT_FILE"

echo ""
echo "✅ TRM evaluation complete!"
echo "   Results saved in: $OUTPUT_FILE"
