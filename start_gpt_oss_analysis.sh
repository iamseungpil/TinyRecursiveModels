#!/bin/bash

# Start GPT-OSS analysis on GPU 0-3
cd /home/ubuntu/dreamcoder-arc/InternVL

for gpu in 0 1 2 3; do
  start_idx=$((gpu * 2143))
  end_idx=$(((gpu + 1) * 2143))

  # Set CUDA_VISIBLE_DEVICES to ensure each process uses only its assigned GPU
  CUDA_VISIBLE_DEVICES=${gpu} nohup conda run -n dream python src/batch_analyze_helmarc.py \
    --input /data/helmarc_correct/20251024_062500/samples.json \
    --output /data/helmarc_analyzed/gpu${gpu} \
    --gpu 0 \
    --start ${start_idx} \
    --end ${end_idx} \
    > /data/helmarc_analyzed/gpu${gpu}.log 2>&1 &

  echo "GPU ${gpu}: Started (samples ${start_idx}-${end_idx}), PID: $!"
  sleep 1
done

echo ""
echo "All processes started. Checking status..."
sleep 3
ps aux | grep batch_analyze_helmarc.py | grep -v grep
