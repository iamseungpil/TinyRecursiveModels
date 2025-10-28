#!/bin/bash
# Launch full dataset generation V2 across 3 GPUs
# IMPROVEMENTS:
#   - 67 complete DSL descriptions (vs 19 previously)
#   - CORRECTED flipx/flipy definitions
#   - Enhanced validation logic
#   - Improved in-context examples

echo "================================================"
echo "LAUNCHING FULL DATASET GENERATION V2"
echo "================================================"
echo "IMPROVEMENTS:"
echo "  - 67 complete DSL primitive descriptions"
echo "  - CORRECTED flipx/flipy definitions"
echo "  - flipx = VERTICAL flip (top↔bottom)"
echo "  - flipy = HORIZONTAL flip (left↔right)"
echo "  - Enhanced validation logic"
echo "  - Improved in-context examples with corrections"
echo "================================================"
echo "Total samples: 8,572"
echo "Split across GPU 1, 2, 3"
echo "Expected output: ~17,144 samples (8,572 × 2 types)"
echo "================================================"

# Create logs directory in /data
mkdir -p /data/helmarc_gpt_analysis_v2/logs

# GPU 1: samples 0-2857
echo "Launching GPU 1: samples 0-2857..."
CUDA_VISIBLE_DEVICES=1 nohup python3 generate_full_dataset_v2.py --gpu 1 --start 0 --end 2857 > /data/helmarc_gpt_analysis_v2/logs/gpu1_generation.log 2>&1 &
GPU1_PID=$!
echo "GPU 1 PID: $GPU1_PID"

# GPU 2: samples 2857-5714
echo "Launching GPU 2: samples 2857-5714..."
CUDA_VISIBLE_DEVICES=2 nohup python3 generate_full_dataset_v2.py --gpu 2 --start 2857 --end 5714 > /data/helmarc_gpt_analysis_v2/logs/gpu2_generation.log 2>&1 &
GPU2_PID=$!
echo "GPU 2 PID: $GPU2_PID"

# GPU 3: samples 5714-8572
echo "Launching GPU 3: samples 5714-8572..."
CUDA_VISIBLE_DEVICES=3 nohup python3 generate_full_dataset_v2.py --gpu 3 --start 5714 --end 8572 > /data/helmarc_gpt_analysis_v2/logs/gpu3_generation.log 2>&1 &
GPU3_PID=$!
echo "GPU 3 PID: $GPU3_PID"

echo ""
echo "================================================"
echo "ALL PROCESSES LAUNCHED"
echo "================================================"
echo "GPU 1 PID: $GPU1_PID (samples 0-2857)"
echo "GPU 2 PID: $GPU2_PID (samples 2857-5714)"
echo "GPU 3 PID: $GPU3_PID (samples 5714-8572)"
echo ""
echo "Monitor progress:"
echo "  tail -f /data/helmarc_gpt_analysis_v2/logs/gpu1_generation.log"
echo "  tail -f /data/helmarc_gpt_analysis_v2/logs/gpu2_generation.log"
echo "  tail -f /data/helmarc_gpt_analysis_v2/logs/gpu3_generation.log"
echo ""
echo "Check processes:"
echo "  ps aux | grep generate_full_dataset_v2"
echo "================================================"

# Save PIDs to file
echo "GPU1=$GPU1_PID" > /data/helmarc_gpt_analysis_v2/logs/generation_pids.txt
echo "GPU2=$GPU2_PID" >> /data/helmarc_gpt_analysis_v2/logs/generation_pids.txt
echo "GPU3=$GPU3_PID" >> /data/helmarc_gpt_analysis_v2/logs/generation_pids.txt

echo "PIDs saved to /data/helmarc_gpt_analysis_v2/logs/generation_pids.txt"
echo ""
echo "Output will be saved to: /data/helmarc_gpt_analysis_v2/"
echo ""
echo "Expected completion time: ~14 hours"
echo ""
echo "Key improvements in V2:"
echo "  ✓ flipx correctly described as 'vertical flip' (not horizontal)"
echo "  ✓ flipy correctly described as 'horizontal flip' (not vertical)"
echo "  ✓ All 67 DSL primitives have descriptions"
echo "  ✓ Validation logic rejects incorrect flipx/flipy descriptions"
echo "================================================"
