# TRM Step-by-Step Visualization - Quick Start Guide

## Overview

This guide will help you run the proof-of-concept (POC) implementation to visualize how TRM's reasoning evolves across H-cycles when solving ARC puzzles.

## Prerequisites

- TRM checkpoint: `/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071` ✅
- ARC dataset: `/data/arc1concept-aug-1000` ✅
- GPU 4 available (24GB VRAM)
- Python packages: torch, numpy, matplotlib, sklearn

## Quick Start (5 minutes)

### 1. Run POC on Single Puzzle

```bash
cd /home/ubuntu/TinyRecursiveModels
conda activate trm  # Or your TRM environment

# Analyze puzzle 0 (should be solved)
python latent_analysis/scripts/step_by_step_inference_poc.py \
    --puzzle_idx 0 \
    --output_dir latent_analysis/results/poc_puzzle_0
```

**Expected Output**:
```
================================================================================
TRM Step-by-Step Analysis - Proof of Concept
================================================================================
Puzzle Index: 0
Output Dir: latent_analysis/results/poc_puzzle_0
================================================================================
📦 Loading checkpoint from /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071...
✅ Model loaded (H_cycles=3, L_cycles=6)
✅ Loaded puzzle 0: <puzzle_name>
   Examples: 5 (last is test)

🔬 Running step-by-step inference (max_h_steps=3)...
  H-step 0...
  H-step 1...
  H-step 2...
✅ Captured 3 H-step states

🎨 Generating grid evolution visualization...
💾 Saved: latent_analysis/results/poc_puzzle_0/grid_evolution.png

🎨 Generating latent trajectory visualization...
💾 Saved: latent_analysis/results/poc_puzzle_0/latent_trajectory.png

📊 Metrics:
   Solved: True
   Final Accuracy: 100.0%
   Grid Changes: [0.15, 0.05]
   Latent Movements: ['2.345', '0.876']

✅ Analysis complete! Results in: latent_analysis/results/poc_puzzle_0
================================================================================
```

**Generated Files**:
- `grid_evolution.png`: How the predicted grid changes at each H-step
- `latent_trajectory.png`: Latent space trajectory and movement metrics
- `metrics.json`: Numerical convergence statistics

### 2. Visualize Results

```bash
# View the grid evolution
eog latent_analysis/results/poc_puzzle_0/grid_evolution.png

# View the latent trajectory
eog latent_analysis/results/poc_puzzle_0/latent_trajectory.png

# Check metrics
cat latent_analysis/results/poc_puzzle_0/metrics.json
```

### 3. Analyze Multiple Puzzles

To compare solved vs unsolved puzzles:

```bash
# Find some unsolved puzzles first
python latent_analysis/scripts/find_unsolved_puzzles.py

# Then analyze specific indices
for idx in 0 5 10 15 20; do
    python latent_analysis/scripts/step_by_step_inference_poc.py \
        --puzzle_idx $idx \
        --output_dir latent_analysis/results/poc_puzzle_$idx
done
```

## Understanding the Visualizations

### Grid Evolution (`grid_evolution.png`)

**Layout**:
```
Row 1: [Input] [Ground Truth] [H-step 0] [H-step 1] [H-step 2]
Row 2: [Empty] [Empty]        [Errors]   [Errors]   [Errors]
```

**What to Look For**:
- ✅ **Convergence**: Errors decrease over H-steps (red cells disappear)
- ❌ **Oscillation**: Errors increase or fluctuate (prediction unstable)
- 🎯 **Early Solve**: H-step 0 or 1 already correct (fast reasoning)
- 🔄 **Late Convergence**: H-step 2 correct but earlier steps wrong (gradual refinement)

**Example Interpretations**:
- **Errors: 50 → 20 → 0**: Smooth convergence, model confident
- **Errors: 10 → 30 → 10**: Oscillation, model uncertain
- **Errors: 100 → 95 → 90**: Minimal progress, likely unsolved

### Latent Trajectory (`latent_trajectory.png`)

**4 Subplots**:

1. **Top-Left: PCA Trajectory**
   - Shows path through latent space
   - Green circle = Start (H-step 0)
   - Red X = End (H-step 2)
   - Look for: Smooth path vs erratic movement

2. **Top-Right: Movement Magnitude**
   - Bars show ||z_H(t) - z_H(t-1)||
   - Look for: Decreasing trend (convergence) vs constant (oscillation)

3. **Bottom-Left: PC Coordinates Over Time**
   - Shows first 5 PCs evolving
   - Look for: Monotonic trends (directed reasoning) vs oscillation

4. **Bottom-Right: Variance Explained**
   - PCA quality indicator
   - PC1+PC2 should explain >10% variance

**What to Look For**:
- ✅ **Converging**: Movement decreases (2.5 → 1.2 → 0.3)
- ❌ **Diverging**: Movement increases (1.0 → 2.0 → 3.0)
- 🔄 **Oscillating**: Movement stays constant (2.0 → 2.1 → 1.9)

### Metrics JSON

```json
{
  "num_h_steps": 3,
  "grid_changes": [0.15, 0.05],      // Fraction of cells changed per step
  "latent_movements": [2.345, 0.876], // L2 distance in latent space
  "accuracies": [0.85, 0.95, 1.0],   // Grid accuracy vs ground truth
  "is_solved": true,
  "final_accuracy": 1.0
}
```

**Key Metrics**:
- `grid_changes`: Should decrease for convergence
- `latent_movements`: Should decrease for stability
- `accuracies`: Should increase monotonically

## Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Use CPU or reduce batch processing
```bash
python step_by_step_inference_poc.py --puzzle_idx 0 --device cpu
```

### Error: "Checkpoint not found"

**Solution**: Verify checkpoint path
```bash
ls -lh /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071
```

### Error: "Dataset not found"

**Solution**: Verify dataset path
```bash
ls -lh /data/arc1concept-aug-1000/test/
```

### Warning: "Final prediction doesn't match extract_latents_corrected.py"

**Cause**: This is a validation check. If final predictions differ, there may be a bug.

**Solution**: Report discrepancy with puzzle_idx for debugging.

## Next Steps

Once POC is validated:

1. **Batch Analysis**: Run on 50 solved + 50 unsolved puzzles
2. **Statistical Tests**: Compare convergence metrics (t-tests)
3. **Pattern Discovery**: Cluster analysis on latent trajectories
4. **Interactive Viz**: Port to Plotly for web-based exploration

## Advanced Usage

### Custom H-Steps

Override default 3 H-cycles:

```python
# In step_by_step_inference_poc.py, modify:
history = step_by_step_inference(model, batch, device, max_h_steps=5)
```

### Export Latents for Analysis

```python
# Add after line 385 in POC script:
import pickle
with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history, f)
```

### Batch Processing Script

```bash
# Create batch_analyze.sh
#!/bin/bash
for idx in {0..99}; do
    python latent_analysis/scripts/step_by_step_inference_poc.py \
        --puzzle_idx $idx \
        --output_dir latent_analysis/results/batch_100/puzzle_$idx \
        2>&1 | tee latent_analysis/results/batch_100/puzzle_${idx}.log
done
```

## Expected Runtime

- Single puzzle: ~5-10 seconds (GPU)
- Single puzzle: ~30-60 seconds (CPU)
- 100 puzzles: ~10-15 minutes (GPU)

## Validation Checklist

Before running on many puzzles, validate on 1-2 examples:

- [ ] Final prediction matches `extract_latents_corrected.py`
- [ ] Grid evolution shows logical progression
- [ ] Latent trajectory is smooth (not random)
- [ ] Metrics JSON contains expected fields
- [ ] Visualizations render correctly (no blank/corrupted images)

## Contact

For questions or issues:
- Check `/home/ubuntu/TinyRecursiveModels/latent_analysis/STEP_BY_STEP_VISUALIZATION_PLAN.md`
- Review POC code: `step_by_step_inference_poc.py`
- Validate against existing extraction: `extract_latents_corrected.py`
