#!/bin/bash
# Phase 1 MVP Training Script
# Run on GPU 3

set -e

echo "🚀 Starting Phase 1 MVP Training"
echo "================================"

# Set environment
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

# Activate environment (if needed)
# conda activate your_env

# Navigate to project directory
cd /home/ubuntu/TinyRecursiveModels/gpt-integration

# Run training
python training/train_phase1_mvp.py

echo "✅ Training complete!"
