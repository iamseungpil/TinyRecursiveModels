#!/bin/bash
# =============================================================================
# TRM-Titans v7 Phase 3: Test-Time Training Evaluation
# =============================================================================
# Description: Evaluate model with test-time training adaptation on ARC puzzles
# Usage: ./scripts/phase3_ttt_evaluate.sh [CHECKPOINT] [TTT_STEPS] [TTT_LR] [OPTIONS]
#
# This script uses evaluate_ttt.py to:
# 1. Load a pretrained checkpoint
# 2. For each puzzle: adapt on training examples, then predict test examples
# 3. Compute pass@k accuracy with TTT-adapted predictions
# 4. Save submission.json for Kaggle
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
DEFAULT_CHECKPOINT_DIR="/data/TinyRecursiveModels/checkpoints/trm_titans_v7"
DEFAULT_TTT_STEPS=10
DEFAULT_TTT_LR=0.01
DEFAULT_DATA_PATH="${PROJECT_ROOT}/data/arc-aug-1000"

# Output configuration
OUTPUT_BASE="${PROJECT_ROOT}/outputs/ttt_eval"
LOG_DIR="${PROJECT_ROOT}/logs/ttt_evaluate"

# GPU settings
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
TRM-Titans v7 Phase 3: Test-Time Training Evaluation

Usage: $(basename "$0") [CHECKPOINT] [TTT_STEPS] [TTT_LR] [OPTIONS]

Positional Arguments:
    CHECKPOINT          Path to checkpoint file (default: latest from default dir)
    TTT_STEPS           Number of TTT adaptation steps (default: 10)
    TTT_LR              Learning rate for TTT adaptation (default: 0.01)

Options:
    --gpu ID            GPU ID to use (default: 0)
    --data PATH         Path to ARC data directory
    --output PATH       Output directory for results
    --max-puzzles N     Maximum puzzles to evaluate (0 = all)
    --no-accumulate     Disable memory accumulation across demos
    --verbose           Print detailed progress
    --dry-run           Print command without executing
    --help              Show this help message

Examples:
    # Evaluate with defaults (latest checkpoint, 10 steps, lr=0.01)
    ./scripts/phase3_ttt_evaluate.sh

    # Custom TTT parameters
    ./scripts/phase3_ttt_evaluate.sh /path/to/checkpoint 20 0.005

    # Quick test with fewer puzzles
    ./scripts/phase3_ttt_evaluate.sh --max-puzzles 10 --verbose

    # Disable memory accumulation
    ./scripts/phase3_ttt_evaluate.sh --no-accumulate

Output Files:
    outputs/ttt_eval_TIMESTAMP/
    |-- submission.json     Kaggle submission file
    |-- metrics.json        pass@k accuracy results
    |-- eval_config.json    Evaluation configuration

TTT Memory Modes:
    --accumulate (default)  Memory persists across demo examples
    --no-accumulate         Memory resets for each demo example

EOF
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
CHECKPOINT_PATH=""
TTT_STEPS=""
TTT_LR=""
DATA_PATH=""
OUTPUT_DIR=""
MAX_PUZZLES=0
ACCUMULATE_MEMORY=true
VERBOSE=false
DRY_RUN=false

# Parse arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-puzzles)
            MAX_PUZZLES="$2"
            shift 2
            ;;
        --no-accumulate)
            ACCUMULATE_MEMORY=false
            shift
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
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Process positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    CHECKPOINT_PATH="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    TTT_STEPS="${POSITIONAL_ARGS[1]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
    TTT_LR="${POSITIONAL_ARGS[2]}"
fi

# Apply defaults
TTT_STEPS="${TTT_STEPS:-${DEFAULT_TTT_STEPS}}"
TTT_LR="${TTT_LR:-${DEFAULT_TTT_LR}}"
DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

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

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/ttt_evaluate_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
log "Starting TRM-Titans v7 TTT Evaluation"
log "Project root: ${PROJECT_ROOT}"
log "Checkpoint: ${CHECKPOINT_PATH}"
log "TTT steps: ${TTT_STEPS}"
log "TTT learning rate: ${TTT_LR}"
log "Memory accumulation: ${ACCUMULATE_MEMORY}"
log "Output directory: ${OUTPUT_DIR}"
log "GPU: ${GPU_ID}"

# Check GPU availability
if ! nvidia-smi -i "${GPU_ID}" &>/dev/null; then
    log_error "GPU ${GPU_ID} not available"
    exit 1
fi
log "GPU ${GPU_ID} verified"

# Check evaluate_ttt.py exists
EVAL_SCRIPT="${PROJECT_ROOT}/evaluate_ttt.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    log_error "evaluate_ttt.py not found: ${EVAL_SCRIPT}"
    exit 1
fi

# Check data directory
if [[ ! -d "${DATA_PATH}" ]]; then
    log_error "Data directory not found: ${DATA_PATH}"
    exit 1
fi

# Check for test_puzzles.json
if [[ ! -f "${DATA_PATH}/test_puzzles.json" ]]; then
    log_error "test_puzzles.json not found in: ${DATA_PATH}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Build Command
# -----------------------------------------------------------------------------
CMD="python ${EVAL_SCRIPT} \
    --checkpoint ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_lr ${TTT_LR} \
    --output_path ${OUTPUT_DIR} \
    --device cuda:0"

# Add optional flags
if [[ "${ACCUMULATE_MEMORY}" == "false" ]]; then
    CMD+=" --no_accumulate_memory"
fi

if [[ "${VERBOSE}" == "true" ]]; then
    CMD+=" --verbose"
fi

if [[ "${MAX_PUZZLES}" -gt 0 ]]; then
    CMD+=" --max_puzzles ${MAX_PUZZLES}"
fi

# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
log "Command to execute:"
echo "  ${CMD}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    log "Dry run mode - not executing"
    exit 0
fi

log "Starting TTT evaluation..."
log "Log file: ${LOG_FILE}"

cd "${PROJECT_ROOT}"

# Execute with logging
{
    ${CMD} 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
}

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "TTT Evaluation completed successfully"

    # Display results summary
    echo ""
    echo "========================================"
    echo "TTT Evaluation Results"
    echo "========================================"

    if [[ -f "${OUTPUT_DIR}/metrics.json" ]]; then
        echo "Metrics:"
        cat "${OUTPUT_DIR}/metrics.json"
        echo ""
    fi

    echo "Output files:"
    ls -la "${OUTPUT_DIR}/"

    log "Results saved to: ${OUTPUT_DIR}"
    log "Submission file: ${OUTPUT_DIR}/submission.json"
else
    log_error "TTT Evaluation failed with exit code: ${EXIT_CODE}"
    log_error "Check logs: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi
