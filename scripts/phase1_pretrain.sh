#!/bin/bash
# =============================================================================
# TRM-Titans v7 Phase 1: Distributed Pretraining
# =============================================================================
# Description: Run 4-GPU distributed pretraining with TRM-Titans v7 architecture
# Usage: ./scripts/phase1_pretrain.sh [--config CONFIG_NAME] [--epochs EPOCHS] [--help]
#
# Environment: Requires 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)
# Outputs: Checkpoints saved to /data/TinyRecursiveModels/checkpoints/trm_titans_v7
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

# Default values
CONFIG_NAME="cfg_trm_titans_fast"
CHECKPOINT_PATH="/data/TinyRecursiveModels/checkpoints/trm_titans_v7"
LOG_DIR="${PROJECT_ROOT}/logs/pretrain"
NUM_GPUS=4
GLOBAL_BATCH_SIZE=96  # Adjusted for 4 GPUs (24 per GPU)

# Training overrides (can be passed via CLI)
EPOCHS=""
EVAL_INTERVAL=""
LR=""

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
TRM-Titans v7 Phase 1: Distributed Pretraining

Usage: $(basename "$0") [OPTIONS]

Options:
    --config NAME       Hydra config name (default: cfg_trm_titans_fast)
    --checkpoint PATH   Checkpoint save directory
                        (default: /data/TinyRecursiveModels/checkpoints/trm_titans_v7)
    --gpus N            Number of GPUs to use (default: 4)
    --batch-size N      Global batch size (default: 96)
    --epochs N          Number of training epochs (overrides config)
    --eval-interval N   Evaluation interval in epochs (overrides config)
    --lr RATE           Learning rate (overrides config)
    --dry-run           Print command without executing
    --help              Show this help message

Examples:
    # Run with defaults
    ./scripts/phase1_pretrain.sh

    # Custom epochs and checkpoint path
    ./scripts/phase1_pretrain.sh --epochs 64 --checkpoint /custom/path

    # Quick test with fewer epochs
    ./scripts/phase1_pretrain.sh --epochs 2 --eval-interval 2

Environment Variables:
    CUDA_VISIBLE_DEVICES    Override GPU selection (default: 0,1,2,3)
    TORCH_NCCL_BLOCKING_WAIT Set to 1 for NCCL debugging (auto-set by script)
    DISABLE_COMPILE         Set to disable torch.compile for debugging

EOF
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
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
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# GPU selection
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# NCCL settings for stability
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

# Master port for distributed training (configurable for multi-job scenarios)
MASTER_PORT="${MASTER_PORT:-29500}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Log files with timestamp
STDOUT_LOG="${LOG_DIR}/pretrain_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/pretrain_${TIMESTAMP}.err"

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
log "Starting TRM-Titans v7 Pretraining"
log "Project root: ${PROJECT_ROOT}"
log "Config: ${CONFIG_NAME}"
log "Checkpoint path: ${CHECKPOINT_PATH}"

# Check GPU availability
log "Checking GPU availability..."
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
if [[ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]]; then
    log_error "Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available"
    exit 1
fi
log "GPUs available: ${AVAILABLE_GPUS}, using: ${NUM_GPUS}"

# Check data directory
DATA_PATH="${PROJECT_ROOT}/data/arc-aug-1000"
if [[ ! -d "${DATA_PATH}" ]]; then
    log_error "Data directory not found: ${DATA_PATH}"
    exit 1
fi
log "Data path verified: ${DATA_PATH}"

# Check pretrain.py exists
if [[ ! -f "${PROJECT_ROOT}/pretrain.py" ]]; then
    log_error "pretrain.py not found in ${PROJECT_ROOT}"
    exit 1
fi

# Create checkpoint directory
mkdir -p "${CHECKPOINT_PATH}"
log "Checkpoint directory: ${CHECKPOINT_PATH}"

# -----------------------------------------------------------------------------
# Build Hydra Override String
# -----------------------------------------------------------------------------
HYDRA_OVERRIDES=""

# Always override these
HYDRA_OVERRIDES+=" global_batch_size=${GLOBAL_BATCH_SIZE}"
HYDRA_OVERRIDES+=" checkpoint_path=${CHECKPOINT_PATH}"
HYDRA_OVERRIDES+=" run_name=trm_titans_v7_${TIMESTAMP}"

# Optional overrides
if [[ -n "${EPOCHS}" ]]; then
    HYDRA_OVERRIDES+=" epochs=${EPOCHS}"
fi

if [[ -n "${EVAL_INTERVAL}" ]]; then
    HYDRA_OVERRIDES+=" eval_interval=${EVAL_INTERVAL}"
fi

if [[ -n "${LR}" ]]; then
    HYDRA_OVERRIDES+=" lr=${LR}"
fi

# -----------------------------------------------------------------------------
# Build and Execute Command
# -----------------------------------------------------------------------------
CMD="torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    ${PROJECT_ROOT}/pretrain.py \
    --config-name=${CONFIG_NAME} \
    ${HYDRA_OVERRIDES}"

log "Command to execute:"
echo "  ${CMD}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    log "Dry run mode - not executing"
    exit 0
fi

log "Starting training..."
log "Logs: stdout -> ${STDOUT_LOG}"
log "Logs: stderr -> ${STDERR_LOG}"

# Change to project root and execute
cd "${PROJECT_ROOT}"

# Run with tee for real-time output and file logging
# Use process substitution to capture both streams
{
    ${CMD} 2>&1 | tee "${STDOUT_LOG}"
    EXIT_CODE=${PIPESTATUS[0]}
} 2>&1

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "Training completed successfully"
    log "Checkpoint saved to: ${CHECKPOINT_PATH}"

    # List saved checkpoints
    log "Saved checkpoints:"
    ls -la "${CHECKPOINT_PATH}/" | tail -10
else
    log_error "Training failed with exit code: ${EXIT_CODE}"
    log_error "Check logs: ${STDOUT_LOG}"
    exit ${EXIT_CODE}
fi
