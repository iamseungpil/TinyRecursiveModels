#!/bin/bash
#
# Baseline Runner Script (LLM-only, no TRM)
#
# Usage: ./run_baseline.sh [ARC_JSON_PREFIX] [OUTPUT] [SUBSET] [MAX_ATTEMPTS]
#
# Example:
#   ./run_baseline.sh /path/to/arc_agi /data/trm/baseline_results.json evaluation 3

set -e

# Arguments
ARC_JSON_PREFIX="${1:?Error: ARC_JSON_PREFIX required (e.g., /path/to/arc_agi)}"
OUTPUT="${2:-/data/trm/baseline_results.json}"
SUBSET="${3:-evaluation}"
MAX_ATTEMPTS="${4:-3}"
NUM_PROBLEMS="${5:-10}"
DEVICE="${6:-cuda}"

echo "========================================="
echo "ARC Baseline (LLM-only)"
echo "========================================="
echo "JSON prefix:   $ARC_JSON_PREFIX"
echo "Subset:        $SUBSET"
echo "Output:        $OUTPUT"
echo "Max attempts:  $MAX_ATTEMPTS"
echo "Num problems:  $NUM_PROBLEMS"
echo "Device:        $DEVICE"
echo "========================================="

# Run baseline
cd "$(dirname "$0")/../gpt_oss_port"

python run_baseline.py \
    --arc_json_prefix "$ARC_JSON_PREFIX" \
    --subset "$SUBSET" \
    --max_attempts "$MAX_ATTEMPTS" \
    --num_problems "$NUM_PROBLEMS" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    2>&1 | tee "$(dirname "$OUTPUT")/baseline_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Baseline evaluation complete!"
echo "   Results: $OUTPUT"
