#!/bin/bash
#
# Joint Training Script (LLaMA + TRM)
#
# Usage: ./run_joint_training.sh --data_path DATA --output_dir OUTPUT [--trm_checkpoint CKPT]
#
# Example:
#   ./run_joint_training.sh \
#       --data_path /data/arc/processed \
#       --output_dir /data/trm/joint_training \
#       --trm_checkpoint /data/trm/pretrain/checkpoint_step_5000.pt \
#       --device cuda:0

set -e

echo "========================================="
echo "Joint Training (LLaMA + TRM)"
echo "========================================="

# Run training
cd "$(dirname "$0")"

python run_joint_training.py "$@"

echo ""
echo "✅ Joint training launched!"
