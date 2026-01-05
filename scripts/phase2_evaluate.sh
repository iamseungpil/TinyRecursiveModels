#!/bin/bash
# =============================================================================
# TRM-Titans v7 Phase 2: Standard Evaluation with pass@k Metrics
# =============================================================================
# Description: Evaluate a trained checkpoint using the standard ARC evaluator
# Usage: ./scripts/phase2_evaluate.sh [CHECKPOINT_PATH] [--help]
#
# This script runs evaluation using a custom Python evaluation script,
# computing pass@k accuracy and generating submission.json for Kaggle.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Signal Handler for Cleanup
# -----------------------------------------------------------------------------
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] Caught signal, cleaning up..."
    pkill -P $$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Default checkpoint path
DEFAULT_CHECKPOINT_DIR="/data/TinyRecursiveModels/checkpoints/trm_titans_v7"
LOG_DIR="${PROJECT_ROOT}/logs/evaluate"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/evaluate_${TIMESTAMP}"

# GPU settings
NUM_GPUS=1
GPU_ID=0

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*"
}

# -----------------------------------------------------------------------------
# Help Function
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
TRM-Titans v7 Phase 2: Standard Evaluation

Usage: $(basename "$0") [CHECKPOINT_PATH] [OPTIONS]

Arguments:
    CHECKPOINT_PATH     Path to checkpoint file (e.g., step_1000)
                        If not provided, uses latest checkpoint from default directory

Options:
    --gpu ID            GPU ID to use (default: 0)
    --output PATH       Output directory for results
    --dry-run           Print command without executing
    --help              Show this help message

Examples:
    # Evaluate latest checkpoint
    ./scripts/phase2_evaluate.sh

    # Evaluate specific checkpoint
    ./scripts/phase2_evaluate.sh /path/to/checkpoints/step_1000

    # Use specific GPU
    ./scripts/phase2_evaluate.sh --gpu 1

Output Files:
    - submission.json       Kaggle submission file
    - metrics.json          pass@k accuracy results
    - evaluate_*.log        Detailed evaluation logs

EOF
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
CHECKPOINT_PATH=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            # Positional argument: checkpoint path
            CHECKPOINT_PATH="$1"
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Find Checkpoint
# -----------------------------------------------------------------------------
find_latest_checkpoint() {
    local checkpoint_dir="$1"

    if [[ ! -d "${checkpoint_dir}" ]]; then
        log_error "Checkpoint directory not found: ${checkpoint_dir}"
        return 1
    fi

    # Find latest step_* file
    local latest
    latest=$(ls -t "${checkpoint_dir}"/step_* 2>/dev/null | head -1)

    if [[ -z "${latest}" ]]; then
        log_error "No checkpoints found in: ${checkpoint_dir}"
        return 1
    fi

    echo "${latest}"
}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    log "No checkpoint specified, searching for latest..."
    CHECKPOINT_PATH=$(find_latest_checkpoint "${DEFAULT_CHECKPOINT_DIR}") || exit 1
    log "Found latest checkpoint: ${CHECKPOINT_PATH}"
fi

# Verify checkpoint exists
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    log_error "Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TORCH_NCCL_BLOCKING_WAIT=1

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/evaluate_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
log "Starting TRM-Titans v7 Standard Evaluation"
log "Project root: ${PROJECT_ROOT}"
log "Checkpoint: ${CHECKPOINT_PATH}"
log "Output directory: ${OUTPUT_DIR}"
log "GPU: ${GPU_ID}"

# Check GPU availability
if ! nvidia-smi -i "${GPU_ID}" &>/dev/null; then
    log_error "GPU ${GPU_ID} not available"
    exit 1
fi
log "GPU ${GPU_ID} verified"

# Check data directory
DATA_PATH="${PROJECT_ROOT}/data/arc-aug-1000"
if [[ ! -d "${DATA_PATH}" ]]; then
    log_error "Data directory not found: ${DATA_PATH}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Run Evaluation via Python Script
# -----------------------------------------------------------------------------
# Create a temporary Python script for evaluation-only mode
EVAL_SCRIPT="${OUTPUT_DIR}/run_evaluation.py"

cat > "${EVAL_SCRIPT}" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Evaluation-only script for TRM-Titans.
Loads checkpoint and runs ARC evaluator to compute pass@k metrics.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import yaml

# Add project root to path
# Use cwd since the bash script cd's to project root before running this script
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from torch.utils.data import DataLoader
from utils.functions import load_model_class


def load_config_from_checkpoint(checkpoint_path: str):
    """Load training config from checkpoint directory."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/arc-aug-1000")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config_from_checkpoint(args.checkpoint)
    if config is None:
        print("Warning: Could not load config, using defaults")
        config = {"arch": {"name": "recursive_reasoning.trm_titans@TRM_Titans"}}

    # Load dataset metadata
    metadata_path = os.path.join(args.data_path, "test", "dataset.json")
    with open(metadata_path, "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))

    print(f"Metadata: vocab_size={metadata.vocab_size}, seq_len={metadata.seq_len}")

    # Build model config
    arch_config = config.get("arch", {})
    model_cfg = {
        "batch_size": args.batch_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "causal": False,
        **{k: v for k, v in arch_config.items() if k not in ["name", "loss"]}
    }

    # Create model with loss head
    model_cls = load_model_class(arch_config.get("name", "recursive_reasoning.trm_titans@TRM_Titans"))
    loss_config = arch_config.get("loss", {"name": "recursive_reasoning.trm_titans@TRM_Titans_ACTLossHead"})
    loss_head_cls = load_model_class(loss_config.get("name", "recursive_reasoning.trm_titans@TRM_Titans_ACTLossHead"))

    print("Creating model...")
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **{k: v for k, v in loss_config.items() if k != "name"})

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Create eval dataloader
    print("Creating evaluation dataloader...")
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[args.data_path],
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=args.batch_size
    ), split="test")

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True
    )

    # Create evaluator
    from evaluators.arc import ARC
    evaluator = ARC(
        data_path=args.data_path,
        eval_metadata=metadata,
        pass_Ks=(1, 2, 5, 10, 100, 1000)
    )
    evaluator.begin_eval()

    # Run evaluation
    print("Running evaluation...")
    total_batches = 0

    with torch.no_grad():
        for set_name, batch, global_batch_size in dataloader:
            total_batches += 1
            if total_batches % 100 == 0:
                print(f"  Processed {total_batches} batches...")

            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)

            # ACT loop
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=evaluator.required_outputs
                )
                if all_finish:
                    break

            evaluator.update_batch(batch, preds)

    print(f"Total batches processed: {total_batches}")

    # Get results
    print("Computing final metrics...")
    os.makedirs(args.output_dir, exist_ok=True)

    results = evaluator.result(
        save_path=args.output_dir,
        rank=0,
        world_size=1,
        group=None
    )

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for key, value in sorted(results.items()):
        if "pass@" in key:
            print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
PYTHON_EOF

# -----------------------------------------------------------------------------
# Execute Evaluation
# -----------------------------------------------------------------------------
log "Running evaluation..."

CMD="python ${EVAL_SCRIPT} \
    --checkpoint ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 24"

log "Command: ${CMD}"

if [[ "${DRY_RUN}" == "true" ]]; then
    log "Dry run mode - not executing"
    exit 0
fi

cd "${PROJECT_ROOT}"

# Execute with logging
{
    ${CMD} 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
}

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "Evaluation completed successfully"
    log "Results saved to: ${OUTPUT_DIR}"

    # Display summary
    if [[ -f "${OUTPUT_DIR}/metrics.json" ]]; then
        echo ""
        echo "=== Metrics Summary ==="
        cat "${OUTPUT_DIR}/metrics.json"
        echo ""
    fi

    # Check for submission file
    if [[ -f "${OUTPUT_DIR}/submission.json" ]]; then
        log "Submission file: ${OUTPUT_DIR}/submission.json"
    fi
else
    log_error "Evaluation failed with exit code: ${EXIT_CODE}"
    log_error "Check logs: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi
