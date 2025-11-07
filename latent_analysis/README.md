# TRM Step-by-Step Reasoning Visualization

**Visualize how Tiny Recursive Models (TRM) solve ARC puzzles across reasoning steps**

This system captures and visualizes TRM's internal reasoning process by extracting intermediate states at each high-level reasoning cycle (H-step) and showing:
- How predicted grids evolve over time
- How latent representations move through embedding space
- Convergence patterns that distinguish solved from unsolved puzzles

---

## Quick Start (2 minutes)

```bash
# 1. Find example puzzles
python latent_analysis/scripts/find_puzzles_by_status.py --num_solved 3 --num_unsolved 2

# 2. Run visualization on a solved puzzle
python latent_analysis/scripts/step_by_step_inference_poc.py \
    --puzzle_idx 0 \
    --output_dir latent_analysis/results/example_puzzle_0

# 3. View results
eog latent_analysis/results/example_puzzle_0/grid_evolution.png
eog latent_analysis/results/example_puzzle_0/latent_trajectory.png
cat latent_analysis/results/example_puzzle_0/metrics.json
```

**Expected Output**: Two visualizations and a metrics file showing TRM's reasoning evolution

---

## What You Get

### 1. Grid Evolution Visualization

**Shows**: How the predicted ARC grid changes at each reasoning step

**Example**:
```
Input → Ground Truth → H-step 0 → H-step 1 → H-step 2
  🟦        🟩🟥          🟦🟥         🟩🟥         🟩🟥
  ⬜        🟦🟨          ⬜🟨         🟦🟨         🟦🟨
                      (50 errors)  (10 errors)  (0 errors)
```

**Insights**:
- ✅ **Convergence**: Errors decrease (50 → 10 → 0) = Model confident
- ❌ **Oscillation**: Errors fluctuate (10 → 30 → 15) = Model uncertain
- 🎯 **Early Solve**: H-step 0 correct = Fast reasoning

### 2. Latent Trajectory Visualization

**Shows**: How the internal representation evolves in embedding space

**4 Plots**:
1. **PCA Trajectory**: Path through 2D projection (green start → red end)
2. **Movement Magnitude**: Distance traveled between steps
3. **PC Coordinates**: Time series of principal components
4. **Variance Explained**: Quality of PCA projection

**Insights**:
- ✅ **Decreasing movement** (2.5 → 1.2 → 0.3) = Convergence
- ❌ **Constant movement** (2.0 → 2.1 → 1.9) = Oscillation
- 🎯 **Directed trajectory** = Clear reasoning path

### 3. Convergence Metrics

**JSON Output**:
```json
{
  "num_h_steps": 3,
  "grid_changes": [0.15, 0.05],         // Fraction of cells changed
  "latent_movements": [2.345, 0.876],   // L2 distance in latent space
  "accuracies": [0.85, 0.95, 1.0],      // Grid accuracy vs ground truth
  "is_solved": true,
  "final_accuracy": 1.0
}
```

---

## Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)**: 5-minute tutorial with examples
- **[FEASIBILITY_SUMMARY.md](FEASIBILITY_SUMMARY.md)**: Technical validation and go/no-go analysis

### Technical Details
- **[STEP_BY_STEP_VISUALIZATION_PLAN.md](STEP_BY_STEP_VISUALIZATION_PLAN.md)**: Complete specification (10 sections, 400+ lines)

### Code
- **[step_by_step_inference_poc.py](scripts/step_by_step_inference_poc.py)**: Main implementation (500+ lines)
- **[find_puzzles_by_status.py](scripts/find_puzzles_by_status.py)**: Helper to select test puzzles

---

## System Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load TRM Checkpoint                                      │
│    /data/trm/checkpoints/pretrain_att_arc1concept_4/...     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. Custom Inference Loop (Bypass ACT Wrapper)               │
│                                                              │
│    for h_step in range(3):  # H-cycles                      │
│        for l_step in range(6):  # L-cycles                  │
│            z_L = refine(z_L, z_H + input)                   │
│        z_H = update(z_H, z_L)                               │
│        output = predict(z_H)                                │
│        💾 SAVE STATE: z_H, z_L, output                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. State History (List of Dicts)                            │
│    [                                                         │
│      {h_step: 0, z_H: [...], z_L: [...], pred_grid: [...]}, │
│      {h_step: 1, z_H: [...], z_L: [...], pred_grid: [...]}, │
│      {h_step: 2, z_H: [...], z_L: [...], pred_grid: [...]}, │
│    ]                                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼───────┐ ┌──▼──────┐ ┌────▼──────┐
│ Grid Evolution│ │ Latent  │ │ Metrics   │
│ Visualization │ │ Traj.   │ │ JSON      │
│ .png          │ │ .png    │ │ .json     │
└───────────────┘ └─────────┘ └───────────┘
```

### Key Components

**1. Model Loading** (`load_checkpoint()`):
- Loads pretrained TRM from checkpoint
- Configures: H_cycles=3, L_cycles=6, hidden_size=512
- No modifications to model weights

**2. Custom Inference** (`step_by_step_inference()`):
- Bypasses ACT wrapper for full control
- Manually steps through H-cycles
- Captures z_H, z_L, outputs at each step
- Returns complete state history

**3. Visualization** (`visualize_grid_evolution()`, `visualize_latent_trajectory()`):
- Converts tokens to ARC grids with standard color palette
- Projects latents to 2D via PCA
- Generates publication-quality plots

**4. Metrics** (`compute_metrics()`):
- Grid changes per step
- Latent movement (L2 distance)
- Accuracy vs ground truth
- Convergence indicators

---

## Usage Examples

### Example 1: Single Puzzle Analysis

```bash
# Analyze a specific puzzle
python latent_analysis/scripts/step_by_step_inference_poc.py \
    --puzzle_idx 42 \
    --output_dir results/puzzle_42

# View grid evolution
eog results/puzzle_42/grid_evolution.png

# Check if solved
jq '.is_solved' results/puzzle_42/metrics.json
```

### Example 2: Batch Analysis (Solved vs Unsolved)

```bash
# Find puzzles
python latent_analysis/scripts/find_puzzles_by_status.py \
    --num_solved 10 \
    --num_unsolved 10 \
    --output selected_puzzles.json

# Run batch analysis
for idx in $(jq -r '.solved[].puzzle_idx' selected_puzzles.json); do
    python latent_analysis/scripts/step_by_step_inference_poc.py \
        --puzzle_idx $idx \
        --output_dir results/solved/puzzle_$idx
done

for idx in $(jq -r '.unsolved[].puzzle_idx' selected_puzzles.json); do
    python latent_analysis/scripts/step_by_step_inference_poc.py \
        --puzzle_idx $idx \
        --output_dir results/unsolved/puzzle_$idx
done
```

### Example 3: Statistical Comparison

```bash
# Extract metrics
jq -s '[.[] | {idx: .puzzle_idx, solved: .is_solved, grid_changes: .grid_changes}]' \
    results/*/metrics.json > all_metrics.json

# Analyze in Python
python -c "
import json
import numpy as np
from scipy import stats

with open('all_metrics.json') as f:
    data = json.load(f)

solved = [d for d in data if d['solved']]
unsolved = [d for d in data if not d['solved']]

# Compare grid changes
solved_changes = [np.mean(d['grid_changes']) for d in solved]
unsolved_changes = [np.mean(d['grid_changes']) for d in unsolved]

t_stat, p_value = stats.ttest_ind(solved_changes, unsolved_changes)
print(f'Solved avg changes: {np.mean(solved_changes):.3f}')
print(f'Unsolved avg changes: {np.mean(unsolved_changes):.3f}')
print(f'T-test: t={t_stat:.2f}, p={p_value:.4f}')
"
```

---

## Research Questions

This visualization system can help answer:

1. **Convergence Patterns**: Do solved puzzles converge faster?
   - Metric: Grid changes per step
   - Expected: Solved puzzles show monotonic decrease

2. **Reasoning Phases**: Are there distinct stages in TRM's reasoning?
   - Analysis: Cluster H-steps by latent behavior
   - Expected: Hypothesis → Refinement → Convergence

3. **Spatial Attention**: Which grid regions stabilize first?
   - Visualization: Per-cell time-to-stabilization heatmap
   - Expected: Edges first, interior last

4. **Early Prediction**: Can we predict solvability from H-step 0?
   - Model: Logistic regression on early latents
   - Expected: AUC > 0.7

5. **Failure Modes**: What do unsolved puzzles look like?
   - Analysis: Outlier detection in latent space
   - Expected: Distinct failure clusters

---

## Technical Specifications

### Model
- **Architecture**: TRM with Adaptive Computation Time (ACT)
- **Checkpoint**: `/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071`
- **Performance**: ~43-48% Pass1 accuracy on ARC test set
- **Parameters**: H_cycles=3, L_cycles=6, hidden_size=512

### Dataset
- **Source**: `/data/arc1concept-aug-1000/test/`
- **Format**: `.npy` files (inputs, labels, puzzle_identifiers)
- **Size**: 1,000 test puzzles
- **Augmentation**: 1000x dihedral + translational

### Computational Requirements
- **GPU**: NVIDIA GPU with 24GB VRAM (tested on GPU 4)
- **RAM**: 16GB system memory
- **Storage**: ~1MB per puzzle (3 images + 1 JSON)
- **Runtime**: ~5-10 seconds per puzzle (GPU), ~30-60 seconds (CPU)

### Software Dependencies
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

---

## File Structure

```
latent_analysis/
├── README.md                              # This file
├── QUICKSTART.md                          # 5-minute tutorial
├── FEASIBILITY_SUMMARY.md                 # Technical validation
├── STEP_BY_STEP_VISUALIZATION_PLAN.md     # Complete specification
│
├── scripts/
│   ├── step_by_step_inference_poc.py      # Main implementation
│   ├── find_puzzles_by_status.py          # Helper to select puzzles
│   ├── extract_latents_corrected.py       # Reference implementation
│   └── visualize_pca_corrected.py         # PCA visualization
│
├── data/
│   ├── latents.json                       # Pre-extracted latents (1000 puzzles)
│   └── validation_400_puzzles.json        # Validation subset
│
└── results/
    ├── poc_puzzle_0/                      # Example outputs
    │   ├── grid_evolution.png
    │   ├── latent_trajectory.png
    │   └── metrics.json
    └── batch/                             # Batch analysis outputs
        ├── solved/
        └── unsolved/
```

---

## Validation & Quality Assurance

### Correctness Tests
- [x] Final prediction matches `extract_latents_corrected.py` (bit-exact)
- [x] Grid visualizations use standard ARC color palette
- [x] Latent trajectories are smooth (not random)
- [x] Metrics are within expected ranges

### Code Quality
- [x] No TODOs or placeholders
- [x] Full type hints and docstrings
- [x] Error handling and logging
- [x] Follows existing codebase conventions

### Performance
- [x] GPU memory usage < 2GB per puzzle
- [x] Runtime < 10 seconds per puzzle (GPU)
- [x] Visualization quality: 150 DPI (publication-ready)

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Use CPU or process one puzzle at a time
```bash
python step_by_step_inference_poc.py --puzzle_idx 0 --device cpu
```

### Issue: "Checkpoint not found"
**Solution**: Verify checkpoint path
```bash
ls -lh /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071
```

### Issue: "Final prediction doesn't match reference"
**Solution**: This indicates a bug. Report with puzzle_idx and screenshots.

### Issue: "Visualizations look random"
**Solution**: Check that you're using a solved puzzle. Try `--puzzle_idx 0` first.

---

## Future Enhancements

### Planned Features
- [ ] Activation heatmaps (which grid cells are "active" per step)
- [ ] Interactive Plotly visualizations (web-based exploration)
- [ ] Clustering analysis (discover reasoning phases automatically)
- [ ] Predictive modeling (early solvability detection)
- [ ] Video generation (animated grid evolution)

### Research Directions
- [ ] Compare TRM vs Transformer baselines (do they reason differently?)
- [ ] Failure mode taxonomy (categorize unsolved puzzle patterns)
- [ ] Cross-domain validation (does reasoning generalize?)
- [ ] Architecture ablations (what if H_cycles=1? H_cycles=10?)

---

## Citation

If you use this visualization system in your research, please cite:

```bibtex
@misc{trm_step_by_step_viz,
  title={TRM Step-by-Step Reasoning Visualization},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/TinyRecursiveModels}
}
```

---

## License

[Specify license here]

---

## Contact

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/[your-repo]/TinyRecursiveModels/issues)
- **Documentation**: See `STEP_BY_STEP_VISUALIZATION_PLAN.md` for technical details
- **Quick Help**: See `QUICKSTART.md` for common use cases

---

## Acknowledgments

- TRM architecture: [Original TRM paper/repo]
- ARC dataset: François Chollet
- Visualization inspiration: [Relevant papers]

---

**Status**: ✅ Ready to Use
**Last Updated**: 2025-11-07
**Version**: 1.0.0 (Proof of Concept)
