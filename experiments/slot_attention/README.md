# TRM with Slot Attention + Contrastive Learning

This experiment extends the Tiny Recursive Reasoning Model (TRM) with **Slot Attention** for compositional decomposition and **Contrastive Learning** for learning structured representations.

## Overview

### Motivation

ARC tasks often involve compositional rules (e.g., "diagonal symmetry + color inversion + fill empty spaces"). The baseline TRM learns a monolithic representation `z_H` that doesn't explicitly decompose these components.

**Goal:** Decompose `z_H` into **slots** where each slot represents a compositional rule or pattern.

### Architecture

```
TRM Reasoning
    ↓
z_H [B, 916, 512]
    ↓
Slot Attention (iterative)
    ↓
slots [B, 8, 256]
    ↓         ↓
Direct    Slot Decoder
    ↓         ↓
logits    logits_slots
```

### Key Components

1. **Slot Attention Module** (`models/slot_attention.py`)
   - Decomposes `z_H` into `num_slots` slots
   - Uses iterative attention refinement (3 iterations)
   - Permutation-invariant binding

2. **Dual Prediction Heads**
   - **Direct Head**: Predicts from `z_H` directly (baseline)
   - **Slot Head**: Reconstructs from slots (encourages meaningful decomposition)

3. **Contrastive Learning with Hungarian Matching**
   - Finds same `puzzle_id` examples in batch
   - Uses Hungarian algorithm for optimal slot assignment
   - Maximizes cosine similarity between matched slots

### Loss Function

```python
total_loss = (
    lm_loss_direct +                      # Pixel-level loss (direct)
    0.5 * lm_loss_slots +                 # Slot reconstruction loss
    0.1 * slot_contrastive_loss +         # Contrastive loss (NEW!)
    0.5 * q_halt_loss                     # Halting loss
)
```

**Slot Contrastive Loss:**
```python
# For same puzzle_id examples
slots_1 = [s1_0, s1_1, ..., s1_7]  # Example 1
slots_2 = [s2_0, s2_1, ..., s2_7]  # Example 2

# Hungarian matching to handle permutation
matching = hungarian_match(slots_1, slots_2)

# Maximize similarity for matched slots
for (i, j) in matching:
    loss += -cosine_sim(slots_1[i], slots_2[j])
```

## Configuration

See `configs/trm_slots.yaml` for full configuration.

**Key parameters:**
- `num_slots: 8` - Number of slots
- `slot_dim: 256` - Dimension of each slot
- `slot_iterations: 3` - Iterative refinement steps
- `slot_recon_weight: 0.5` - Weight for reconstruction loss
- `slot_contrastive_weight: 0.1` - Weight for contrastive loss
- `use_hungarian_matching: true` - Enable Hungarian matching

## Usage

### Training

```bash
# From project root
cd experiments/slot_attention

# Single GPU
python train_slot_attention.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train_slot_attention.py

# Override config parameters
python train_slot_attention.py arch.num_slots=16 arch.loss.slot_contrastive_weight=0.2
```

### Monitoring

Training metrics logged to WandB:
- `train/lm_loss` - Direct LM loss
- `train/lm_loss_slots` - Slot reconstruction loss
- `train/slot_contrastive_loss` - Contrastive loss
- `train/accuracy` - Pixel accuracy
- `train/exact_accuracy` - Exact match accuracy

### Evaluation

```bash
# Evaluate checkpoint
python ../../pretrain.py \
    --config-path experiments/slot_attention/configs \
    --config-name trm_slots \
    load_checkpoint=/path/to/checkpoint \
    eval_interval=1 \
    epochs=1
```

## Expected Results

### Hypotheses

1. **Compositional Decomposition**
   - Slots should learn to represent different sub-rules
   - Slot 0: "symmetry", Slot 1: "color transform", etc.

2. **Improved Generalization**
   - Better systematic generalization on unseen tasks
   - Similar tasks → similar slot patterns

3. **Interpretability**
   - Visualize slot assignments
   - Understand what each slot represents

### Evaluation Metrics

- **ARC pass@K** - Accuracy on ARC evaluation set
- **Slot utilization** - How many slots are actively used
- **Slot stability** - Consistency of slot assignments across examples
- **Contrastive alignment** - Cosine similarity of matched slots

## Analysis

### Slot Visualization

```python
# Extract slot representations
slots = model.slot_attention(z_H)  # [B, 8, 256]

# Compute similarity matrix
for i, j in enumerate(same_task_examples):
    sim_matrix[i, j] = cosine_similarity(slots[i], slots[j])

# Expected: High similarity for same task
```

### Potential Issues

1. **Slot Collapse**
   - **Problem:** All slots learn same representation
   - **Solution:** Add diversity regularization

2. **Permutation Instability**
   - **Problem:** Slot assignments change across iterations
   - **Solution:** Hungarian matching helps, but may need additional regularization

3. **Computational Overhead**
   - **Problem:** Hungarian algorithm is O(n³)
   - **Solution:** Batch only contains few same-puzzle pairs, overhead minimal

4. **Contrastive Loss Batching** ⚠️ IMPORTANT
   - **Problem:** Contrastive loss requires ≥2 examples with same puzzle_id in batch
   - **Impact:** With random sampling, most batches won't have duplicates → loss = 0
   - **Solutions:**
     - Use custom sampler that ensures same-puzzle pairs in each batch
     - Increase batch size to increase probability of duplicates
     - Use memory bank to store recent slot representations
   - **Status:** Current implementation uses random sampling (may need improvement)

## File Structure

```
experiments/slot_attention/
├── README.md                           # This file
├── train_slot_attention.py             # Training script
└── configs/
    └── trm_slots.yaml                  # Configuration

models/
├── slot_attention.py                   # Slot Attention module
├── slot_losses.py                      # SlotContrastiveLossHead (separate from baseline)
├── losses.py                           # Baseline loss functions (ACTLossHead)
└── recursive_reasoning/
    ├── trm.py                          # Baseline TRM
    └── trm_with_slots.py               # TRM + Slots model
```

**Note**: Slot Attention components are fully separated from baseline TRM:
- `slot_losses.py` contains slot-specific losses (requires scipy)
- Baseline TRM users can use `losses.py` without scipy dependency

## Dependencies

Additional dependencies for this experiment:
```bash
pip install scipy  # For Hungarian matching (linear_sum_assignment)
```

## References

1. **Slot Attention**
   - Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020
   - https://arxiv.org/abs/2006.15055

2. **Contrastive Learning**
   - Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
   - https://arxiv.org/abs/2002.05709

3. **Hungarian Algorithm**
   - For optimal bipartite matching with O(n³) complexity
   - https://en.wikipedia.org/wiki/Hungarian_algorithm

## TODO / Future Work

- [ ] Add diversity regularization to prevent slot collapse
- [ ] Experiment with Sinkhorn matching (differentiable alternative to Hungarian)
- [ ] Curriculum learning: start with fewer slots, gradually increase
- [ ] Visualize slot attention maps
- [ ] Analyze slot semantic meaning
- [ ] Test on compositional generalization benchmarks

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainers.
