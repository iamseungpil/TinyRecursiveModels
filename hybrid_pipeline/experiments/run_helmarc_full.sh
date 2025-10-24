#!/bin/bash
#
# HelmARC Full Pipeline
#
# Runs all 3 phases sequentially:
#   Phase 1: Adapter Pretraining (precomputed embeddings)
#   Phase 2: Joint Training (end-to-end)
#   Phase 3: Repulsion Training (self-correction)
#
# Usage: ./run_helmarc_full.sh [DEVICE]
# Example: ./run_helmarc_full.sh cuda:6
#

set -e

DEVICE=${1:-cuda:6}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "HelmARC Full Training Pipeline"
echo "=========================================="
echo "Device: $DEVICE"
echo "Estimated total time: 4-6 days"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Phase 1: Adapter Pretraining
echo ""
echo "=========================================="
echo "Starting Phase 1..."
echo "=========================================="
$SCRIPT_DIR/run_helmarc_phase1.sh $DEVICE

# Phase 2: Joint Training
echo ""
echo "=========================================="
echo "Starting Phase 2..."
echo "=========================================="
$SCRIPT_DIR/run_helmarc_phase2.sh auto $DEVICE

# Phase 3: Repulsion Training
echo ""
echo "=========================================="
echo "Starting Phase 3..."
echo "=========================================="
$SCRIPT_DIR/run_helmarc_phase3.sh auto $DEVICE

echo ""
echo "=========================================="
echo "✅ Full pipeline completed!"
echo "=========================================="
echo ""
echo "Checkpoints:"
echo "  Phase 1: /data/hybrid_training/phase1_adapters/"
echo "  Phase 2: /data/hybrid_training/phase2_joint/"
echo "  Phase 3: /data/hybrid_training/phase3_repulsion/"
echo ""
echo "WandB projects:"
echo "  Phase 1: arc-phase1-adapters"
echo "  Phase 2: arc-phase2-joint"
echo "  Phase 3: arc-phase3-repulsion"
echo ""
