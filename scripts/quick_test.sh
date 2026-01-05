#!/bin/bash
# =============================================================================
# TRM-Titans v7 Quick Test: Fast Validation Pipeline
# =============================================================================
# Description: Quick validation of training and TTT evaluation with minimal data
# Usage: ./scripts/quick_test.sh [--mode MODE] [--help]
#
# Modes:
#   train     - Quick training test (2 epochs, single GPU)
#   ttt       - Quick TTT evaluation (5 steps, 10 puzzles)
#   full      - Both train and TTT test (default)
#
# This script is designed for rapid iteration and debugging.
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

# Quick test settings
QUICK_EPOCHS=2
QUICK_EVAL_INTERVAL=2
QUICK_BATCH_SIZE=24
QUICK_TTT_STEPS=5
QUICK_TTT_LR=0.01
QUICK_MAX_PUZZLES=10

# Paths
CHECKPOINT_DIR="/data/TinyRecursiveModels/checkpoints/quick_test_${TIMESTAMP}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/quick_test_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/logs/quick_test"

# GPU settings
GPU_ID=0

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUICK-TEST] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*"
}

log_section() {
    echo ""
    echo "========================================"
    echo "$*"
    echo "========================================"
}

# -----------------------------------------------------------------------------
# Help Function
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
TRM-Titans v7 Quick Test: Fast Validation Pipeline

Usage: $(basename "$0") [OPTIONS]

Options:
    --mode MODE         Test mode: train, ttt, or full (default: full)
    --gpu ID            GPU ID to use (default: 0)
    --epochs N          Quick training epochs (default: 2)
    --puzzles N         Number of puzzles for TTT test (default: 10)
    --ttt-steps N       TTT adaptation steps (default: 5)
    --checkpoint PATH   Existing checkpoint for ttt mode
    --verbose           Enable verbose output
    --dry-run           Print commands without executing
    --help              Show this help message

Modes:
    train   Run quick training (${QUICK_EPOCHS} epochs, single GPU)
            Useful for validating training pipeline and config

    ttt     Run quick TTT evaluation (${QUICK_MAX_PUZZLES} puzzles, ${QUICK_TTT_STEPS} steps)
            Requires existing checkpoint (--checkpoint) or uses latest

    full    Run both train and TTT test (default)
            Complete pipeline validation

Examples:
    # Full quick test
    ./scripts/quick_test.sh

    # Only test training
    ./scripts/quick_test.sh --mode train

    # Only test TTT with existing checkpoint
    ./scripts/quick_test.sh --mode ttt --checkpoint /path/to/checkpoint

    # Verbose output for debugging
    ./scripts/quick_test.sh --verbose

Expected Time:
    train:  ~5-10 minutes (2 epochs on 1 GPU)
    ttt:    ~2-5 minutes (10 puzzles, 5 steps each)
    full:   ~10-15 minutes total

EOF
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
MODE="full"
VERBOSE=false
DRY_RUN=false
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --epochs)
            QUICK_EPOCHS="$2"
            shift 2
            ;;
        --puzzles)
            QUICK_MAX_PUZZLES="$2"
            shift 2
            ;;
        --ttt-steps)
            QUICK_TTT_STEPS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
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

# Validate mode
case "${MODE}" in
    train|ttt|full)
        ;;
    *)
        log_error "Invalid mode: ${MODE}. Use: train, ttt, or full"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TORCH_NCCL_BLOCKING_WAIT=1
export DISABLE_COMPILE=1  # Disable torch.compile for faster startup

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/quick_test_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
log_section "TRM-Titans v7 Quick Test"
log "Mode: ${MODE}"
log "GPU: ${GPU_ID}"
log "Timestamp: ${TIMESTAMP}"
log "Log file: ${LOG_FILE}"

# Check GPU
if ! nvidia-smi -i "${GPU_ID}" &>/dev/null; then
    log_error "GPU ${GPU_ID} not available"
    exit 1
fi

# Check data
DATA_PATH="${PROJECT_ROOT}/data/arc-aug-1000"
if [[ ! -d "${DATA_PATH}" ]]; then
    log_error "Data directory not found: ${DATA_PATH}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Quick Training Function
# -----------------------------------------------------------------------------
run_quick_train() {
    log_section "Quick Training Test"
    log "Epochs: ${QUICK_EPOCHS}"
    log "Batch size: ${QUICK_BATCH_SIZE}"
    log "Checkpoint: ${CHECKPOINT_DIR}"

    # Build torchrun command for single GPU
    CMD="python ${PROJECT_ROOT}/pretrain.py \
        --config-name=cfg_trm_titans_fast \
        epochs=${QUICK_EPOCHS} \
        eval_interval=${QUICK_EVAL_INTERVAL} \
        global_batch_size=${QUICK_BATCH_SIZE} \
        checkpoint_path=${CHECKPOINT_DIR} \
        run_name=quick_test_${TIMESTAMP} \
        checkpoint_every_eval=True"

    log "Command: ${CMD}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log "Dry run - skipping execution"
        return 0
    fi

    cd "${PROJECT_ROOT}"

    START_TIME=$(date +%s)

    if ${CMD} 2>&1 | tee -a "${LOG_FILE}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log_success "Training completed in ${DURATION} seconds"

        # Find the saved checkpoint
        LATEST_CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/step_* 2>/dev/null | head -1 || echo "")
        if [[ -n "${LATEST_CHECKPOINT}" ]]; then
            log "Checkpoint saved: ${LATEST_CHECKPOINT}"
            echo "${LATEST_CHECKPOINT}" > "${OUTPUT_DIR}/latest_checkpoint.txt"
        fi
        return 0
    else
        log_error "Training failed"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Quick TTT Evaluation Function
# -----------------------------------------------------------------------------
run_quick_ttt() {
    log_section "Quick TTT Evaluation Test"
    log "TTT steps: ${QUICK_TTT_STEPS}"
    log "TTT learning rate: ${QUICK_TTT_LR}"
    log "Max puzzles: ${QUICK_MAX_PUZZLES}"

    # Find checkpoint
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        # Try to find from quick train output
        if [[ -f "${OUTPUT_DIR}/latest_checkpoint.txt" ]]; then
            CHECKPOINT_PATH=$(cat "${OUTPUT_DIR}/latest_checkpoint.txt")
            log "Using checkpoint from quick train: ${CHECKPOINT_PATH}"
        else
            # Find latest from default location
            local default_dir="/data/TinyRecursiveModels/checkpoints/trm_titans_v7"
            CHECKPOINT_PATH=$(ls -t "${default_dir}"/step_* 2>/dev/null | head -1 || echo "")

            if [[ -z "${CHECKPOINT_PATH}" ]]; then
                # Check quick test checkpoint
                CHECKPOINT_PATH=$(ls -t "${CHECKPOINT_DIR}"/step_* 2>/dev/null | head -1 || echo "")
            fi

            if [[ -z "${CHECKPOINT_PATH}" ]]; then
                log_error "No checkpoint found. Run training first or specify --checkpoint"
                return 1
            fi
            log "Using latest checkpoint: ${CHECKPOINT_PATH}"
        fi
    fi

    # Verify checkpoint
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        log_error "Checkpoint not found: ${CHECKPOINT_PATH}"
        return 1
    fi

    TTT_OUTPUT="${OUTPUT_DIR}/ttt_results"
    mkdir -p "${TTT_OUTPUT}"

    # Build command
    CMD="python ${PROJECT_ROOT}/evaluate_ttt.py \
        --checkpoint ${CHECKPOINT_PATH} \
        --data_path ${DATA_PATH} \
        --ttt_steps ${QUICK_TTT_STEPS} \
        --ttt_lr ${QUICK_TTT_LR} \
        --output_path ${TTT_OUTPUT} \
        --device cuda:0 \
        --max_puzzles ${QUICK_MAX_PUZZLES}"

    if [[ "${VERBOSE}" == "true" ]]; then
        CMD+=" --verbose"
    fi

    log "Command: ${CMD}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log "Dry run - skipping execution"
        return 0
    fi

    cd "${PROJECT_ROOT}"

    START_TIME=$(date +%s)

    if ${CMD} 2>&1 | tee -a "${LOG_FILE}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log_success "TTT evaluation completed in ${DURATION} seconds"

        # Show results
        if [[ -f "${TTT_OUTPUT}/metrics.json" ]]; then
            echo ""
            echo "=== TTT Metrics ==="
            cat "${TTT_OUTPUT}/metrics.json"
            echo ""
        fi
        return 0
    else
        log_error "TTT evaluation failed"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
OVERALL_START=$(date +%s)
TRAIN_SUCCESS=true
TTT_SUCCESS=true

case "${MODE}" in
    train)
        run_quick_train || TRAIN_SUCCESS=false
        ;;
    ttt)
        run_quick_ttt || TTT_SUCCESS=false
        ;;
    full)
        run_quick_train || TRAIN_SUCCESS=false

        if [[ "${TRAIN_SUCCESS}" == "true" ]]; then
            run_quick_ttt || TTT_SUCCESS=false
        else
            log_error "Skipping TTT test due to training failure"
            TTT_SUCCESS=false
        fi
        ;;
esac

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))

log_section "Quick Test Summary"
log "Mode: ${MODE}"
log "Total time: ${TOTAL_DURATION} seconds"

if [[ "${MODE}" == "train" || "${MODE}" == "full" ]]; then
    if [[ "${TRAIN_SUCCESS}" == "true" ]]; then
        log_success "Training: PASSED"
    else
        log_error "Training: FAILED"
    fi
fi

if [[ "${MODE}" == "ttt" || "${MODE}" == "full" ]]; then
    if [[ "${TTT_SUCCESS}" == "true" ]]; then
        log_success "TTT Evaluation: PASSED"
    else
        log_error "TTT Evaluation: FAILED"
    fi
fi

log "Output directory: ${OUTPUT_DIR}"
log "Log file: ${LOG_FILE}"

# Exit with appropriate code
if [[ "${TRAIN_SUCCESS}" == "true" && "${TTT_SUCCESS}" == "true" ]]; then
    log_success "All tests passed!"
    exit 0
else
    log_error "Some tests failed"
    exit 1
fi
