# JEPA Contrastive Learning for TRM

This document explains the JEPA-style contrastive learning implementation for Tiny Recursive Models (TRM) and how to use it to improve representation learning on ARC tasks.

## Overview

### What is JEPA?

JEPA (Joint-Embedding Predictive Architecture) is a self-supervised learning approach that learns representations by predicting embeddings rather than reconstructing pixels. The key idea:

- **Same task, different inputs** → similar latent representations
- **Different tasks** → different latent representations
- **Latent space learning** → no pixel reconstruction needed

### Why JEPA for TRM?

TRM learns to solve ARC puzzles from training examples. However, the challenge is:

1. **Generalization gap**: Training and validation examples come from the same tasks but with different input grids
2. **Weak spatial encoding**: Current input encoding uses simple lookup embeddings without spatial inductive bias
3. **Limited supervision**: Only supervised on final outputs, not on learned representations

JEPA contrastive learning addresses these by:
- Explicitly encouraging similar representations for examples from the same task
- Providing additional self-supervised signal beyond task loss
- Improving generalization to novel inputs from known tasks

## Architecture

### JEPA Contrastive TRM

```
Input 1 (task A) ──→ Online Encoder ──→ Projector ──→ Predictor ──→ z1_pred
                                                                        │
                                                                        │ Cosine
                                                                        │ similarity
                                                                        │ loss
Input 2 (task A) ──→ Target Encoder ──→ Projector ────────────────→ z2_proj
                     (EMA, no grad)
```

**Components**:

1. **Online Encoder**: TRM model with gradient updates
2. **Target Encoder**: EMA copy of online encoder (no gradients)
3. **Projector**: Maps hidden states to projection space (shared)
4. **Predictor**: Maps online projection to target space (online only)

**Loss**: Cosine similarity between predicted and target representations
```python
loss = 2 - 2 * cosine_similarity(z1_pred, z2_proj)
```

### InfoNCE Contrastive TRM

Alternative approach using batch negatives:

```
Input 1 (task A) ──→ Encoder ──→ Projector ──→ z1
                                                 │
                                                 │ InfoNCE
                                                 │ (contrastive)
Input 2 (task A) ──→ Encoder ──→ Projector ──→ z2
                                                 │
Other inputs     ──→ Encoder ──→ Projector ──→ negatives
```

**Loss**: Cross-entropy over similarity matrix
```python
similarity = z1 @ z2.T / temperature
loss = cross_entropy(similarity, labels=diagonal)
```

## Implementation

### File Structure

```
models/recursive_reasoning/
├── jepa_contrastive.py          # JEPA implementation
├── trm.py                        # Base TRM model

examples/
├── train_trm_with_jepa.py       # Training example

docs/
├── JEPA_CONTRASTIVE_LEARNING.md # This file
```

### Key Classes

#### `JEPAContrastiveTRM`

Main JEPA implementation with EMA target encoder.

```python
from models.recursive_reasoning.jepa_contrastive import JEPAContrastiveTRM

# Create JEPA wrapper
jepa_model = JEPAContrastiveTRM(
    trm_model=base_trm_model,
    hidden_size=512,
    proj_dim=256,
    ema_decay=0.99,
    pool_method='mean'
)

# Forward pass
batch = {
    "inputs": torch.randint(0, 11, (32, 900)),
    "inputs_aug": torch.randint(0, 11, (32, 900)),  # Different example, same task
    "puzzle_identifiers": torch.randint(0, 800, (32,))
}

contrastive_loss = jepa_model(batch)

# Update target encoder (after optimizer step)
jepa_model.update_target_encoder()
```

#### `InfoNCEContrastiveTRM`

Alternative using batch negatives.

```python
from models.recursive_reasoning.jepa_contrastive import InfoNCEContrastiveTRM

infonce_model = InfoNCEContrastiveTRM(
    trm_model=base_trm_model,
    hidden_size=512,
    proj_dim=256,
    temperature=0.07
)

contrastive_loss = infonce_model(batch)
```

### Data Preparation

The key requirement is providing **pairs of examples from the same task**.

#### Strategy 1: Use Different Training Examples (Recommended)

Modify your dataloader to sample two different training examples from the same task:

```python
class PairPuzzleDataset(Dataset):
    def __getitem__(self, idx):
        puzzle_id = self.puzzles[idx]

        # Sample two different training examples
        train_examples = self.puzzle_to_examples[puzzle_id]
        ex1, ex2 = random.sample(train_examples, 2)

        return {
            "inputs": ex1.input,
            "inputs_aug": ex2.input,  # Different example, same task
            "labels": ex1.output,
            "puzzle_identifiers": puzzle_id
        }
```

#### Strategy 2: Data Augmentation

Apply task-preserving transformations:

```python
from models.recursive_reasoning.jepa_contrastive import augment_arc_grid

# Augment inputs (rotation, flip)
inputs_aug = augment_arc_grid(
    batch["inputs"],
    apply_rotation=True,
    apply_flip=True,
    rotation_prob=0.5,
    flip_prob=0.5
)

batch["inputs_aug"] = inputs_aug
```

**Warning**: Many ARC tasks are NOT rotation/flip invariant. Using different training examples is safer.

### Training Integration

#### Combined Loss

Combine task loss (standard TRM objective) with contrastive loss:

```python
# Task loss (cross-entropy on predictions)
carry = trm_model.initial_carry(batch)
carry, task_loss, metrics, _, _ = trm_model(carry, batch, return_keys=[])

# Contrastive loss
contrastive_loss = jepa_model(batch)

# Combined loss
total_loss = task_loss / batch_size + contrastive_weight * contrastive_loss

# Backward and optimize
total_loss.backward()
optimizer.step()

# Update target encoder (JEPA only)
jepa_model.update_target_encoder()
```

#### Hyperparameters

Recommended starting values:

- **contrastive_weight**: 0.1 (tune in [0.01, 0.5])
- **ema_decay**: 0.99 (for JEPA)
- **proj_dim**: 256
- **temperature**: 0.07 (for InfoNCE)
- **pool_method**: 'mean' (alternatives: 'max', 'first', 'last')

### Full Example

See `examples/train_trm_with_jepa.py` for a complete training script.

```bash
python examples/train_trm_with_jepa.py \
    --data_path /path/to/arc_dataset \
    --contrastive_weight 0.1 \
    --contrastive_type jepa \
    --hidden_size 512 \
    --batch_size 32 \
    --lr 1e-4
```

## Design Rationale

### Why Not CNN Input Encoder?

The conversation explored whether CNN input encoding could help. Key insights:

1. **Learned embeddings** provide value semantics (what each color means)
2. **CNN** would add spatial pattern recognition (edges, corners, symmetries)
3. **Both are complementary**, not redundant

JEPA is orthogonal to input encoding improvements and can be combined with CNN encoders.

### Static Input Embeddings vs Dynamic Latent States

Important clarification from architecture analysis:

- **Input embeddings**: Computed once, injected every L-cycle (STATIC)
- **z_H, z_L**: Recursively updated each cycle (DYNAMIC)
- **Relationship**: Input is raw data; z represents processing/reasoning

JEPA operates on z_H (dynamic reasoning state), not input embeddings.

### JEPA vs InfoNCE

**JEPA**:
- ✅ Doesn't require large batches
- ✅ EMA target provides stable targets
- ✅ Predictor prevents collapse
- ❌ More complex (two encoders)

**InfoNCE**:
- ✅ Simpler architecture
- ✅ Strong contrastive signal with negatives
- ❌ Requires diverse batch (different tasks)
- ❌ May be too strict for ARC (tasks have subtle differences)

**Recommendation**: Start with JEPA for ARC tasks.

## Expected Results

### Metrics to Track

1. **Task accuracy**: Primary metric (should not degrade)
2. **Contrastive loss**: Should decrease over training
3. **Representation similarity**:
   - Same task, different inputs → high cosine similarity
   - Different tasks → low cosine similarity

### Expected Improvements

- Better generalization on validation set (same tasks, novel inputs)
- More robust to input variations
- Faster convergence on new tasks (transfer learning)

### Ablation Studies

Recommended experiments:

1. **Weight sweep**: contrastive_weight ∈ {0.0, 0.01, 0.05, 0.1, 0.3, 0.5}
2. **Architecture**: JEPA vs InfoNCE
3. **Data strategy**: Paired examples vs augmentation
4. **Pooling method**: mean vs max vs first vs last

## Advanced Usage

### Custom Augmentation

Define task-specific augmentations:

```python
def custom_arc_augmentation(inputs):
    """Custom augmentation for specific ARC task properties."""
    # Example: Only augment if task is rotation-invariant
    # (requires task metadata)
    if is_rotation_invariant(task_id):
        return augment_arc_grid(inputs, apply_rotation=True)
    else:
        return inputs.clone()

batch = create_augmented_batch(batch, augmentation_fn=custom_arc_augmentation)
```

### Multi-View Contrastive Learning

Extend to multiple views:

```python
# Create 3 views
views = [batch["inputs"], batch["inputs_aug1"], batch["inputs_aug2"]]

# Pairwise contrastive loss
loss = 0
for i in range(len(views)):
    for j in range(i+1, len(views)):
        loss += contrastive_loss(views[i], views[j])
```

### Freezing Components

Freeze encoder, only train projection heads (for transfer learning):

```python
# Freeze TRM encoder
for param in jepa_model.encoder.parameters():
    param.requires_grad = False

# Only train projector and predictor
optimizer = torch.optim.Adam([
    {'params': jepa_model.projector.parameters()},
    {'params': jepa_model.predictor.parameters()}
])
```

## Troubleshooting

### High Contrastive Loss

**Symptom**: Contrastive loss stays high or increases

**Possible causes**:
- contrastive_weight too high (reduce to 0.01-0.1)
- EMA decay too high (reduce to 0.95-0.98)
- inputs and inputs_aug are identical (check data pipeline)
- Model hasn't converged on task loss yet (reduce weight early in training)

### Task Performance Degradation

**Symptom**: Task accuracy decreases when adding contrastive loss

**Solutions**:
- Reduce contrastive_weight (try 0.01)
- Ensure task loss is well-tuned first
- Check gradient norms (contrastive gradients may dominate)
- Use gradient clipping

### Representation Collapse

**Symptom**: All representations become identical

**Solutions**:
- Use predictor (JEPA) to prevent collapse
- Increase projection dimension
- Check EMA update is working
- Use InfoNCE with negatives instead

## References

1. **JEPA Paper**: [A Path Towards Autonomous Machine Intelligence](https://arxiv.org/abs/2301.08243)
2. **SimSiam**: [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)
3. **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
4. **MoCo**: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{trm_jepa_contrastive,
  title={JEPA-style Contrastive Learning for Tiny Recursive Models},
  author={TinyRecursiveModels Contributors},
  year={2025},
  howpublished={\url{https://github.com/TinyRecursiveModels}}
}
```

## Contributing

Contributions welcome! Areas for improvement:

- Task-adaptive augmentation strategies
- Better pooling methods for sequential reasoning
- Integration with curriculum learning
- Multi-task contrastive learning

Please submit issues or pull requests on GitHub.
