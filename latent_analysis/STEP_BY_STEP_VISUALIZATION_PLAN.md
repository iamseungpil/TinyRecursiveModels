# TRM Step-by-Step Visualization: Feasibility Analysis & Implementation Plan

**Date**: 2025-11-07
**Checkpoint**: `/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071`
**Goal**: Visualize how TRM's reasoning evolves across H-cycles when solving ARC puzzles

---

## 1. TECHNICAL FEASIBILITY ANALYSIS

### ✅ FEASIBLE - Infrastructure Exists

**Model Architecture** (`models/recursive_reasoning/trm.py`):
```python
# Forward pass structure (lines 204-230)
for H_step in range(H_cycles-1):  # Without gradient
    for L_step in range(L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings)
    z_H = self.L_level(z_H, z_L)
# Final iteration with gradient
for L_step in range(L_cycles):
    z_L = self.L_level(z_L, z_H + input_embeddings)
z_H = self.L_level(z_H, z_L)
output = self.lm_head(z_H)  # Final prediction
```

**Key Configuration** (from checkpoint):
- `H_cycles`: 3 (high-level reasoning steps)
- `L_cycles`: 6 (low-level refinements per H-step)
- `halt_max_steps`: 16 (max ACT steps, but default is 3 H-cycles)
- `hidden_size`: 512 (latent dimension)
- `seq_len`: 900 (30×30 grid flattened)

**Current Limitations**:
1. ❌ **No intermediate state extraction**: Current forward pass only returns final output
2. ❌ **Gradient detachment**: H_cycles-1 iterations run without gradients (no_grad context)
3. ❌ **ACT wrapper opacity**: The outer ACT wrapper manages halting, obscuring internal iterations

**What CAN Be Extracted**:
- ✅ `z_H`: High-level representation [batch, 900+puzzle_emb_len, 512]
- ✅ `z_L`: Low-level representation [batch, 900+puzzle_emb_len, 512]
- ✅ `output`: Prediction logits [batch, 900, 12] (vocab: PAD, EOS, 0-9)
- ✅ `q_logits`: Halting signal [batch, 2] (halt vs continue)

### Checkpoint & Data Availability

**Checkpoint** ✅:
- Location: `/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071`
- Size: 1.7GB
- Performance: ~43-48% Pass1 accuracy
- Successfully loaded in `extract_latents_corrected.py`

**Test Data** ✅:
- Location: `/data/arc1concept-aug-1000/test/`
- Format: `.npy` files (inputs, labels, puzzle_identifiers)
- Total puzzles: 1000 (from `latents.json`)
- Solved rate: ~93.6% (from extraction logs)

**Existing Infrastructure** ✅:
- Model loader: `extract_latents_corrected.py` (working)
- Grid visualization: `eval_and_visualize_trm.py` (archived)
- PCA visualization: `visualize_pca_corrected.py` (working)
- Dataset utilities: `dataset/build_arc_dataset.py` (inverse_aug, _crop)

---

## 2. IMPLEMENTATION PLAN

### Phase 1: Custom Forward Pass with Intermediate Capture

**Approach**: Modify the inner forward pass to expose intermediate states

**Implementation Options**:

#### Option A: Hook-Based Capture (Non-Invasive) ⭐ RECOMMENDED
```python
class IntermediateStateCapture:
    """Capture intermediate states during forward pass."""

    def __init__(self):
        self.h_step_states = []  # List of (z_H, z_L, output, step_idx)

    def register_hooks(self, model):
        """Register forward hooks on L_level modules."""
        # Hook after each H-cycle to capture z_H, z_L
        pass
```

**Pros**:
- No model modification required
- Clean separation of concerns
- Easy to toggle on/off

**Cons**:
- Hooks may not capture all intermediate states cleanly
- Requires careful hook placement

#### Option B: Custom Inference Loop (Full Control) ⭐⭐ RECOMMENDED
```python
def step_by_step_inference(model, batch, max_h_steps=3):
    """
    Manual stepping through H-cycles with full state capture.

    Returns:
        history: List of dicts with {
            'h_step': int,
            'z_H': [batch, seq_len, hidden],
            'z_L': [batch, seq_len, hidden],
            'output_logits': [batch, seq_len, vocab],
            'pred_grid': numpy array
        }
    """
    # Initialize carry
    carry = model.inner.empty_carry(batch_size, device)
    carry = model.inner.reset_carry(torch.ones(batch_size, dtype=bool), carry)

    # Prepare inputs
    seq_info = dict(cos_sin=model.inner.rotary_emb())
    input_embeddings = model.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    history = []
    z_H, z_L = carry.z_H, carry.z_L

    # Step through H-cycles
    for h_step in range(max_h_steps):
        # L-cycles
        for l_step in range(model.config.L_cycles):
            z_L = model.inner.L_level(z_L, z_H + input_embeddings, **seq_info)

        # H-cycle update
        z_H = model.inner.L_level(z_H, z_L, **seq_info)

        # Generate output
        output_logits = model.inner.lm_head(z_H)[:, model.inner.puzzle_emb_len:]
        pred_tokens = output_logits.argmax(dim=-1)

        # Save state
        history.append({
            'h_step': h_step,
            'z_H': z_H.clone().detach().float(),
            'z_L': z_L.clone().detach().float(),
            'output_logits': output_logits.clone().detach(),
            'pred_tokens': pred_tokens.clone().detach(),
        })

    return history
```

**Pros**:
- Full control over iteration
- Easy to extract any intermediate state
- Clear logic flow

**Cons**:
- Duplicates some model logic
- Must keep in sync with model implementation

### Phase 2: Visualization System

#### 2.1 Grid Evolution Visualization

**Goal**: Show how predicted grid changes at each H-step

```python
def visualize_grid_evolution(history, ground_truth, task_id):
    """
    Create visualization showing grid prediction at each H-step.

    Layout:
        Row 1: Input | Ground Truth | H-step 0 | H-step 1 | H-step 2
        Row 2: Diff from GT (color intensity = error magnitude)
    """
    num_steps = len(history)
    fig, axes = plt.subplots(2, num_steps + 2, figsize=(4 * (num_steps + 2), 8))

    # Color palette (ARC standard)
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

    # Draw input and ground truth
    draw_arc_grid(axes[0, 0], input_grid, "Input")
    draw_arc_grid(axes[0, 1], ground_truth, "Ground Truth")
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    # Draw each H-step prediction
    for h_step, state in enumerate(history):
        pred_grid = tokens_to_grid(state['pred_tokens'], height, width)

        # Prediction
        draw_arc_grid(axes[0, h_step + 2], pred_grid, f"H-step {h_step}")

        # Difference heatmap
        diff = compute_grid_diff(pred_grid, ground_truth)
        axes[1, h_step + 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[1, h_step + 2].set_title(f"Error: {diff.sum():.0f} cells")
```

**Metrics to Display**:
- Cell-level accuracy per step
- Number of cells that changed from previous step
- Convergence indicator (did prediction stabilize?)

#### 2.2 Latent Space Trajectory Visualization

**Goal**: Show how z_H and z_L embeddings evolve in latent space

```python
def visualize_latent_trajectory(history, output_path):
    """
    Show trajectory of latent representations through PCA space.

    Plots:
        1. z_H trajectory in 2D PCA space
        2. z_L trajectory in 2D PCA space
        3. z_H evolution over time (first 10 PCs)
        4. Distance metrics: ||z_H(t) - z_H(t-1)||
    """
    # Extract latents (average over sequence length)
    z_H_sequence = [state['z_H'].mean(dim=1).cpu().numpy() for state in history]  # [steps, batch, 512]
    z_L_sequence = [state['z_L'].mean(dim=1).cpu().numpy() for state in history]

    # Stack for PCA
    all_z_H = np.vstack(z_H_sequence)  # [steps * batch, 512]
    all_z_L = np.vstack(z_L_sequence)

    # PCA projection
    pca_H = PCA(n_components=2)
    z_H_2d = pca_H.fit_transform(all_z_H)

    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: z_H trajectory in PCA space
    for batch_idx in range(z_H_sequence[0].shape[0]):
        trajectory = [z_H_2d[step * batch_size + batch_idx] for step in range(len(history))]
        trajectory = np.array(trajectory)

        axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], '-o', alpha=0.6, linewidth=2)
        axes[0, 0].scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        axes[0, 0].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='X', label='End')

    axes[0, 0].set_title('z_H Trajectory in PCA Space')
    axes[0, 0].set_xlabel(f'PC1 ({pca_H.explained_variance_ratio_[0]:.1%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca_H.explained_variance_ratio_[1]:.1%})')

    # Plot 2: Distance over time
    distances = []
    for i in range(1, len(z_H_sequence)):
        dist = np.linalg.norm(z_H_sequence[i] - z_H_sequence[i-1], axis=-1).mean()
        distances.append(dist)

    axes[0, 1].plot(range(1, len(history)), distances, '-o', linewidth=2)
    axes[0, 1].set_title('Latent Space Movement (z_H)')
    axes[0, 1].set_xlabel('H-step')
    axes[0, 1].set_ylabel('||z_H(t) - z_H(t-1)||')
    axes[0, 1].grid(True, alpha=0.3)
```

**Insights to Extract**:
- Do latents converge? (decreasing movement over time)
- Are there distinct phases? (large jumps then stabilization)
- Do solved puzzles show different patterns than unsolved?

#### 2.3 Attention/Activation Heatmaps (Optional)

**Goal**: Show which parts of the grid TRM is "focusing on"

```python
def visualize_activation_patterns(history, input_grid, output_path):
    """
    Visualize spatial activation patterns in z_H.

    For each H-step, show heatmap of activation magnitude across grid positions.
    """
    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 6))

    for h_step, state in enumerate(history):
        z_H = state['z_H'][0]  # [seq_len, hidden_size]

        # Compute activation magnitude per position
        activation_magnitude = z_H.norm(dim=-1).cpu().numpy()  # [seq_len]

        # Reshape to 30x30 grid (skip puzzle embedding positions)
        grid_activations = activation_magnitude[puzzle_emb_len:].reshape(30, 30)

        # Crop to actual puzzle size
        grid_activations_cropped = grid_activations[:height, :width]

        # Heatmap
        im = axes[h_step].imshow(grid_activations_cropped, cmap='hot', interpolation='nearest')
        axes[h_step].set_title(f'H-step {h_step} Activation')
        plt.colorbar(im, ax=axes[h_step])
```

### Phase 3: Comprehensive Analysis Pipeline

#### Main Script: `step_by_step_analysis.py`

```python
def analyze_puzzle_evolution(
    checkpoint_path: str,
    puzzle_id: int,
    output_dir: str,
    max_h_steps: int = 3
):
    """
    Complete analysis pipeline for single puzzle.

    Outputs:
        - grid_evolution.png: Grid predictions over time
        - latent_trajectory.png: Latent space movement
        - activation_heatmaps.png: Spatial attention patterns
        - metrics.json: Numerical convergence metrics
    """
    # 1. Load model and data
    model = load_checkpoint(checkpoint_path)
    batch = load_puzzle(puzzle_id)

    # 2. Run step-by-step inference
    history = step_by_step_inference(model, batch, max_h_steps)

    # 3. Generate visualizations
    visualize_grid_evolution(history, batch['labels'], output_dir)
    visualize_latent_trajectory(history, output_dir)
    visualize_activation_patterns(history, batch['inputs'], output_dir)

    # 4. Compute and save metrics
    metrics = compute_convergence_metrics(history)
    save_metrics(metrics, output_dir)
```

#### Batch Analysis: Multiple Puzzles

```python
def batch_analysis(
    checkpoint_path: str,
    num_solved: int = 10,
    num_unsolved: int = 10,
    output_dir: str = "step_by_step_analysis"
):
    """
    Analyze multiple puzzles to identify patterns.

    Comparison Questions:
        1. Do solved puzzles converge faster than unsolved?
        2. Do solved puzzles show less latent movement in final steps?
        3. Are there common activation patterns in solved cases?
    """
    # Load all test puzzles
    test_data = load_test_dataset()

    # Categorize
    solved_puzzles = [p for p in test_data if p['solved']][:num_solved]
    unsolved_puzzles = [p for p in test_data if not p['solved']][:num_unsolved]

    # Analyze each
    for category, puzzles in [("solved", solved_puzzles), ("unsolved", unsolved_puzzles)]:
        for idx, puzzle in enumerate(puzzles):
            puzzle_output_dir = Path(output_dir) / category / f"puzzle_{idx}"
            analyze_puzzle_evolution(checkpoint_path, puzzle, puzzle_output_dir)

    # Comparative analysis
    compare_solved_vs_unsolved(output_dir)
```

---

## 3. EXPECTED INSIGHTS

### 3.1 Convergence Patterns

**Hypothesis**: Solved puzzles converge smoothly, unsolved puzzles oscillate

**Metrics to Track**:
- Grid change rate: `cells_changed(step_t) / total_cells`
- Latent movement: `||z_H(t) - z_H(t-1)||_2`
- Prediction stability: Steps until no more grid changes

**Expected Observations**:
- ✅ **Solved**: Monotonic decrease in grid changes, latent stabilization
- ❌ **Unsolved**: Oscillation, large latent movements even in final steps

### 3.2 Reasoning Phases

**Question**: Are there distinct phases in TRM's reasoning?

**Potential Phases**:
1. **Initial Hypothesis** (H-step 0): Large latent movement, major grid changes
2. **Refinement** (H-step 1): Smaller changes, local corrections
3. **Convergence** (H-step 2): Minimal changes, stabilization

**Detection Method**:
- Plot latent movement over time
- Identify inflection points in change rate
- Cluster steps by behavior (PCA on delta vectors)

### 3.3 Spatial Reasoning Patterns

**Question**: Which parts of the grid does TRM process first?

**Analysis**:
- Track per-cell prediction changes over steps
- Identify which cells stabilize early vs late
- Correlate with input patterns (edges, corners, symmetries)

**Visualization**:
- Heatmap showing "time to stabilization" per cell
- Animation of grid evolution (GIF)

### 3.4 Latent Space Structure

**Question**: Do different H-steps occupy different regions in latent space?

**Analysis**:
- PCA on all z_H states from all puzzles and steps
- Color by: (1) H-step, (2) Solved/unsolved, (3) Puzzle ID
- Check for clustering or trajectories

**Expected Findings**:
- Early steps may cluster by puzzle similarity
- Late steps may separate by correctness
- Trajectories may converge to "attractor" regions

---

## 4. IMPLEMENTATION TIMELINE

### Week 1: Core Infrastructure
- [ ] Implement `step_by_step_inference()` function
- [ ] Test on 5 solved puzzles, verify outputs match final prediction
- [ ] Create basic grid evolution visualization

### Week 2: Visualization Suite
- [ ] Implement latent trajectory visualization
- [ ] Implement activation heatmap visualization
- [ ] Add convergence metrics computation

### Week 3: Batch Analysis
- [ ] Run on 50 solved + 50 unsolved puzzles
- [ ] Generate comparative analysis
- [ ] Statistical testing (t-tests, effect sizes)

### Week 4: Insights & Writeup
- [ ] Identify key patterns
- [ ] Create publication-quality figures
- [ ] Write findings report

---

## 5. TECHNICAL CHALLENGES & SOLUTIONS

### Challenge 1: Memory Constraints

**Problem**: Storing full z_H, z_L for multiple steps and batches

**Solution**:
- Process one puzzle at a time (batch_size=1)
- Store only aggregated statistics (means, norms)
- Use float16 instead of float32 where possible

### Challenge 2: Gradient Computation

**Problem**: Original forward pass disables gradients for H-1 cycles

**Solution**:
- Use model in eval mode (no training needed)
- No gradients required for visualization
- All operations can be in `torch.no_grad()` context

### Challenge 3: ACT Wrapper Complexity

**Problem**: ACT wrapper manages halting, obscures iterations

**Solution**:
- Bypass ACT wrapper, directly call `model.inner`
- Manually implement H/L cycle iteration
- Ignore halting logic (always run full 3 H-cycles)

### Challenge 4: Visualization Scalability

**Problem**: Generating figures for 100+ puzzles

**Solution**:
- Parallel processing with multiprocessing
- Generate thumbnails for overview, full resolution on demand
- Use matplotlib figure caching

---

## 6. CODE STRUCTURE

```
latent_analysis/
├── scripts/
│   ├── step_by_step_inference.py      # Core inference with state capture
│   ├── visualize_grid_evolution.py    # Grid prediction over time
│   ├── visualize_latent_trajectory.py # Latent space movement
│   ├── visualize_activation_maps.py   # Spatial attention heatmaps
│   ├── batch_analysis.py              # Multi-puzzle comparison
│   └── comparative_metrics.py         # Statistical analysis
├── utils/
│   ├── grid_utils.py                  # Token↔Grid conversion
│   ├── arc_colors.py                  # Standard ARC color palette
│   └── metrics.py                     # Convergence/stability metrics
├── data/
│   └── step_by_step_results/          # Output directory
│       ├── solved/
│       │   ├── puzzle_0/
│       │   │   ├── grid_evolution.png
│       │   │   ├── latent_trajectory.png
│       │   │   ├── activation_maps.png
│       │   │   └── metrics.json
│       └── unsolved/
└── analysis/
    ├── convergence_patterns.ipynb     # Interactive analysis
    └── final_report.md                # Findings summary
```

---

## 7. VALIDATION CHECKLIST

Before running full analysis:

- [ ] Test `step_by_step_inference()` on 1 puzzle, verify:
  - [ ] Final output matches original extraction
  - [ ] History contains 3 H-steps (or halt_max_steps)
  - [ ] Latent dimensions correct [1, 900+puzzle_emb_len, 512]
  - [ ] Predictions are valid grids (no PAD/EOS in content)

- [ ] Test visualization on 1 puzzle, verify:
  - [ ] Grids render correctly with ARC colors
  - [ ] PCA projection looks reasonable
  - [ ] Figures save without errors

- [ ] Memory profiling:
  - [ ] Run on 10 puzzles, check peak memory usage
  - [ ] Ensure fits in GPU memory (24GB on GPU 4)

- [ ] Correctness validation:
  - [ ] Compare final prediction with `extract_latents_corrected.py`
  - [ ] Should be identical (same model, same data)

---

## 8. SUCCESS CRITERIA

**Minimum Viable Product**:
1. ✅ Successfully extract intermediate states for 100 puzzles
2. ✅ Generate grid evolution visualization showing convergence
3. ✅ Demonstrate difference in convergence between solved/unsolved

**Stretch Goals**:
1. 🎯 Identify specific "reasoning phases" in latent space
2. 🎯 Find spatial attention patterns (which cells processed first)
3. 🎯 Predict solvability from early H-steps (H-step 0 or 1)
4. 🎯 Create interactive visualization (Plotly/Bokeh)

---

## 9. FINAL RECOMMENDATIONS

### ⭐ START HERE

1. **Implement Option B** (Custom Inference Loop)
   - Most reliable and maintainable
   - Full control over state capture
   - Easy to debug

2. **Select 5 Test Puzzles**
   - 3 solved (high confidence)
   - 2 unsolved (near-miss vs far-miss)
   - Use existing `latents.json` to identify good candidates

3. **Build Incrementally**
   - Start with grid evolution only
   - Add latent trajectory once grid visualization works
   - Add advanced features (attention, activation) last

4. **Validate Early and Often**
   - Compare final output with `extract_latents_corrected.py`
   - Visual inspection of grids (sanity check)
   - Print shapes and statistics at each step

### AVOID PITFALLS

- ❌ Don't modify the checkpoint model code
- ❌ Don't try to visualize all 1000 puzzles at once
- ❌ Don't skip validation (easy to have subtle bugs)
- ✅ DO save intermediate results frequently
- ✅ DO use descriptive filenames (puzzle_id, h_step, metric)
- ✅ DO document unexpected findings immediately

---

## 10. QUESTIONS TO ANSWER

Once implementation is complete, answer these research questions:

1. **Convergence**: Do solved puzzles stabilize faster? (Quantify with metrics)
2. **Phases**: Are there distinct reasoning stages? (Cluster analysis on latent deltas)
3. **Spatial Patterns**: Which cells converge first? (Heatmap analysis)
4. **Predictability**: Can we predict solvability from early steps? (Logistic regression)
5. **Failure Modes**: What do unsolved puzzles look like in latent space? (Outlier analysis)

These insights will be valuable for:
- Understanding TRM's reasoning process
- Improving architecture design
- Debugging failure cases
- Designing better training objectives
