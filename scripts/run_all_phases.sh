#!/bin/bash
# =============================================================================
# TRM-Titans v7 Full Pipeline: All Phases Orchestration
# =============================================================================
# Description: Run the complete TRM-Titans v7 pipeline sequentially
#   Phase 1: Distributed pretraining on 4 GPUs
#   Phase 2: Standard evaluation with pass@k metrics
#   Phase 3: Test-time training evaluation
#
# Usage: ./scripts/run_all_phases.sh [OPTIONS]
#
# This script orchestrates all phases, automatically passing checkpoints
# between phases and generating a comprehensive summary.
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

# Default paths
CHECKPOINT_DIR="/data/TinyRecursiveModels/checkpoints/trm_titans_v7"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/full_pipeline_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/logs/full_pipeline"

# Phase control
SKIP_PHASE1=false
SKIP_PHASE2=false
SKIP_PHASE3=false

# Phase 1 settings
P1_NUM_GPUS=4
P1_BATCH_SIZE=96
P1_EPOCHS=""

# Phase 2 settings
P2_GPU=0

# Phase 3 settings
P3_GPU=0
P3_TTT_STEPS=10
P3_TTT_LR=0.01

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PIPELINE] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*"
}

log_phase() {
    echo ""
    echo "########################################################################"
    echo "# PHASE $1: $2"
    echo "########################################################################"
    echo ""
}

log_summary_header() {
    echo ""
    echo "========================================================================"
    echo "                    PIPELINE EXECUTION SUMMARY"
    echo "========================================================================"
}

# -----------------------------------------------------------------------------
# Help Function
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
TRM-Titans v7 Full Pipeline: All Phases Orchestration

Usage: $(basename "$0") [OPTIONS]

Phase Control:
    --skip-phase1       Skip pretraining (use existing checkpoint)
    --skip-phase2       Skip standard evaluation
    --skip-phase3       Skip TTT evaluation
    --checkpoint PATH   Use specific checkpoint (skips phase 1)

Phase 1 Options (Pretraining):
    --gpus N            Number of GPUs for pretraining (default: 4)
    --batch-size N      Global batch size (default: 96)
    --epochs N          Training epochs (uses config default if not specified)

Phase 2 Options (Standard Eval):
    --p2-gpu ID         GPU for standard evaluation (default: 0)

Phase 3 Options (TTT Eval):
    --p3-gpu ID         GPU for TTT evaluation (default: 0)
    --ttt-steps N       TTT adaptation steps (default: 10)
    --ttt-lr RATE       TTT learning rate (default: 0.01)

General Options:
    --output PATH       Output directory for all results
    --dry-run           Print commands without executing
    --help              Show this help message

Examples:
    # Run all phases with defaults
    ./scripts/run_all_phases.sh

    # Skip training, use existing checkpoint
    ./scripts/run_all_phases.sh --skip-phase1 --checkpoint /path/to/step_1000

    # Custom training epochs
    ./scripts/run_all_phases.sh --epochs 64

    # Only run evaluation phases
    ./scripts/run_all_phases.sh --skip-phase1

Pipeline Flow:
    Phase 1 (Pretraining)
        |
        v  [checkpoint]
    Phase 2 (Standard Eval)
        |
        v  [checkpoint]
    Phase 3 (TTT Eval)
        |
        v
    Summary & Results

EOF
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
DRY_RUN=false
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-phase1)
            SKIP_PHASE1=true
            shift
            ;;
        --skip-phase2)
            SKIP_PHASE2=true
            shift
            ;;
        --skip-phase3)
            SKIP_PHASE3=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            SKIP_PHASE1=true
            shift 2
            ;;
        --gpus)
            P1_NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            P1_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            P1_EPOCHS="$2"
            shift 2
            ;;
        --p2-gpu)
            P2_GPU="$2"
            shift 2
            ;;
        --p3-gpu)
            P3_GPU="$2"
            shift 2
            ;;
        --ttt-steps)
            P3_TTT_STEPS="$2"
            shift 2
            ;;
        --ttt-lr)
            P3_TTT_LR="$2"
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
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
SUMMARY_FILE="${OUTPUT_DIR}/pipeline_summary.txt"

# Track phase results
declare -A PHASE_STATUS
declare -A PHASE_DURATION
PIPELINE_START=$(date +%s)

# Redirect all output to log file while showing on console
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log "Starting TRM-Titans v7 Full Pipeline"
log "Timestamp: ${TIMESTAMP}"
log "Output directory: ${OUTPUT_DIR}"
log "Log file: ${PIPELINE_LOG}"

# -----------------------------------------------------------------------------
# Find Latest Checkpoint Function
# -----------------------------------------------------------------------------
find_latest_checkpoint() {
    local search_dir="$1"

    if [[ ! -d "${search_dir}" ]]; then
        return 1
    fi

    local latest
    latest=$(ls -td "${search_dir}"/step_* 2>/dev/null | head -1)

    if [[ -z "${latest}" ]]; then
        return 1
    fi

    echo "${latest}"
}

# -----------------------------------------------------------------------------
# Phase 1: Pretraining
# -----------------------------------------------------------------------------
run_phase1() {
    log_phase "1" "Distributed Pretraining"

    if [[ "${SKIP_PHASE1}" == "true" ]]; then
        log "Skipping Phase 1 (pretraining)"
        PHASE_STATUS["phase1"]="SKIPPED"
        PHASE_DURATION["phase1"]=0
        return 0
    fi

    local start_time=$(date +%s)

    # Build command
    local cmd="${SCRIPT_DIR}/phase1_pretrain.sh"
    cmd+=" --gpus ${P1_NUM_GPUS}"
    cmd+=" --batch-size ${P1_BATCH_SIZE}"
    cmd+=" --checkpoint ${CHECKPOINT_DIR}"

    if [[ -n "${P1_EPOCHS}" ]]; then
        cmd+=" --epochs ${P1_EPOCHS}"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=" --dry-run"
    fi

    log "Executing: ${cmd}"

    if ${cmd}; then
        local end_time=$(date +%s)
        PHASE_DURATION["phase1"]=$((end_time - start_time))
        PHASE_STATUS["phase1"]="SUCCESS"

        # Find the checkpoint
        CHECKPOINT_PATH=$(find_latest_checkpoint "${CHECKPOINT_DIR}") || true
        if [[ -n "${CHECKPOINT_PATH}" ]]; then
            log "Checkpoint saved: ${CHECKPOINT_PATH}"
        fi
        return 0
    else
        PHASE_STATUS["phase1"]="FAILED"
        PHASE_DURATION["phase1"]=$(($(date +%s) - start_time))
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Phase 2: Standard Evaluation
# -----------------------------------------------------------------------------
run_phase2() {
    log_phase "2" "Standard Evaluation"

    if [[ "${SKIP_PHASE2}" == "true" ]]; then
        log "Skipping Phase 2 (standard evaluation)"
        PHASE_STATUS["phase2"]="SKIPPED"
        PHASE_DURATION["phase2"]=0
        return 0
    fi

    # Find checkpoint if not set
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        CHECKPOINT_PATH=$(find_latest_checkpoint "${CHECKPOINT_DIR}") || true
        if [[ -z "${CHECKPOINT_PATH}" ]]; then
            log_error "No checkpoint found for evaluation"
            PHASE_STATUS["phase2"]="FAILED (no checkpoint)"
            PHASE_DURATION["phase2"]=0
            return 1
        fi
    fi

    local start_time=$(date +%s)

    # Build command
    local cmd="${SCRIPT_DIR}/phase2_evaluate.sh"
    cmd+=" ${CHECKPOINT_PATH}"
    cmd+=" --gpu ${P2_GPU}"
    cmd+=" --output ${OUTPUT_DIR}/phase2_results"

    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=" --dry-run"
    fi

    log "Using checkpoint: ${CHECKPOINT_PATH}"
    log "Executing: ${cmd}"

    if ${cmd}; then
        PHASE_STATUS["phase2"]="SUCCESS"
        PHASE_DURATION["phase2"]=$(($(date +%s) - start_time))
        return 0
    else
        PHASE_STATUS["phase2"]="FAILED"
        PHASE_DURATION["phase2"]=$(($(date +%s) - start_time))
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Phase 3: TTT Evaluation
# -----------------------------------------------------------------------------
run_phase3() {
    log_phase "3" "Test-Time Training Evaluation"

    if [[ "${SKIP_PHASE3}" == "true" ]]; then
        log "Skipping Phase 3 (TTT evaluation)"
        PHASE_STATUS["phase3"]="SKIPPED"
        PHASE_DURATION["phase3"]=0
        return 0
    fi

    # Find checkpoint if not set
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        CHECKPOINT_PATH=$(find_latest_checkpoint "${CHECKPOINT_DIR}") || true
        if [[ -z "${CHECKPOINT_PATH}" ]]; then
            log_error "No checkpoint found for TTT evaluation"
            PHASE_STATUS["phase3"]="FAILED (no checkpoint)"
            PHASE_DURATION["phase3"]=0
            return 1
        fi
    fi

    local start_time=$(date +%s)

    # Build command
    local cmd="${SCRIPT_DIR}/phase3_ttt_evaluate.sh"
    cmd+=" ${CHECKPOINT_PATH}"
    cmd+=" ${P3_TTT_STEPS}"
    cmd+=" ${P3_TTT_LR}"
    cmd+=" --gpu ${P3_GPU}"
    cmd+=" --output ${OUTPUT_DIR}/phase3_results"

    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=" --dry-run"
    fi

    log "Using checkpoint: ${CHECKPOINT_PATH}"
    log "Executing: ${cmd}"

    if ${cmd}; then
        PHASE_STATUS["phase3"]="SUCCESS"
        PHASE_DURATION["phase3"]=$(($(date +%s) - start_time))
        return 0
    else
        PHASE_STATUS["phase3"]="FAILED"
        PHASE_DURATION["phase3"]=$(($(date +%s) - start_time))
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Generate Summary
# -----------------------------------------------------------------------------
generate_summary() {
    log_summary_header

    local pipeline_end=$(date +%s)
    local total_duration=$((pipeline_end - PIPELINE_START))

    {
        echo "Pipeline Execution Summary"
        echo "=========================="
        echo ""
        echo "Timestamp: ${TIMESTAMP}"
        echo "Total Duration: ${total_duration} seconds ($(( total_duration / 60 )) minutes)"
        echo ""
        echo "Phase Results:"
        echo "--------------"

        for phase in phase1 phase2 phase3; do
            local status="${PHASE_STATUS[${phase}]:-NOT RUN}"
            local duration="${PHASE_DURATION[${phase}]:-0}"
            local phase_name=""

            case "${phase}" in
                phase1) phase_name="Pretraining" ;;
                phase2) phase_name="Standard Eval" ;;
                phase3) phase_name="TTT Eval" ;;
            esac

            printf "  %-15s: %-20s (%d seconds)\n" "${phase_name}" "${status}" "${duration}"
        done

        echo ""
        echo "Checkpoint: ${CHECKPOINT_PATH:-N/A}"
        echo ""
        echo "Output Files:"
        echo "-------------"

        if [[ -d "${OUTPUT_DIR}/phase2_results" ]]; then
            echo "  Phase 2 (Standard Eval):"
            ls -la "${OUTPUT_DIR}/phase2_results/" 2>/dev/null | grep -E "\.json$" | awk '{print "    - " $NF}'
        fi

        if [[ -d "${OUTPUT_DIR}/phase3_results" ]]; then
            echo "  Phase 3 (TTT Eval):"
            ls -la "${OUTPUT_DIR}/phase3_results/" 2>/dev/null | grep -E "\.json$" | awk '{print "    - " $NF}'
        fi

        echo ""
        echo "Metrics Summary:"
        echo "----------------"

        # Show Phase 2 metrics
        if [[ -f "${OUTPUT_DIR}/phase2_results/metrics.json" ]]; then
            echo "  Standard Evaluation:"
            cat "${OUTPUT_DIR}/phase2_results/metrics.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for k, v in sorted(data.items()):
    if 'pass@' in k:
        print(f'    {k}: {v:.4f} ({v*100:.2f}%)')
" 2>/dev/null || echo "    (unable to parse metrics)"
        fi

        # Show Phase 3 metrics
        if [[ -f "${OUTPUT_DIR}/phase3_results/metrics.json" ]]; then
            echo "  TTT Evaluation:"
            cat "${OUTPUT_DIR}/phase3_results/metrics.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for k, v in sorted(data.items()):
    if 'pass@' in k:
        print(f'    {k}: {v:.4f} ({v*100:.2f}%)')
" 2>/dev/null || echo "    (unable to parse metrics)"
        fi

        echo ""
        echo "Log Files:"
        echo "----------"
        echo "  Pipeline log: ${PIPELINE_LOG}"
        echo ""
        echo "========================================================================"

    } | tee "${SUMMARY_FILE}"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
OVERALL_SUCCESS=true

# Run phases sequentially
if ! run_phase1; then
    log_error "Phase 1 failed"
    OVERALL_SUCCESS=false
fi

# Continue with phase 2 even if phase 1 was skipped (with existing checkpoint)
if [[ "${OVERALL_SUCCESS}" == "true" || "${SKIP_PHASE1}" == "true" ]]; then
    if ! run_phase2; then
        log_error "Phase 2 failed"
        OVERALL_SUCCESS=false
    fi
fi

# Continue with phase 3
if [[ "${OVERALL_SUCCESS}" == "true" || "${SKIP_PHASE2}" == "true" ]]; then
    if ! run_phase3; then
        log_error "Phase 3 failed"
        OVERALL_SUCCESS=false
    fi
fi

# Generate summary
generate_summary

# Final status
if [[ "${OVERALL_SUCCESS}" == "true" ]]; then
    log_success "Pipeline completed successfully!"
    exit 0
else
    log_error "Pipeline completed with errors"
    exit 1
fi
