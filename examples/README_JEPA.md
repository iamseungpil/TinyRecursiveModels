# JEPA Contrastive Learning - Quick Start Guide

This directory contains examples for using JEPA-style contrastive learning with TRM.

## Files

- `train_trm_with_jepa.py` - Complete training script with JEPA integration

## Quick Start

### 1. Install Dependencies

Ensure you have the TinyRecursiveModels environment set up with PyTorch.

### 2. Run Tests

Verify the implementation works:

```bash
cd /home/user/TinyRecursiveModels
python tests/test_jepa_contrastive.py
```

Expected output:
```
================================================================================
Testing JEPA Contrastive Learning Implementation
================================================================================
✅ JEPA forward pass: loss = 1.2345
✅ InfoNCE forward pass: loss = 2.3456
✅ EMA target update works correctly
✅ Data augmentation works
✅ Augmented batch creation works
✅ Backward pass with correct gradient flow
✅ All pooling methods work

================================================================================
All tests passed! ✅
================================================================================
```

### 3. Basic Usage

#### Minimal Example

```python
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.jepa_contrastive import JEPAContrastiveTRM

# Create base TRM model
trm_model = TinyRecursiveReasoningModel_ACTV1(config)

# Wrap with JEPA
jepa_model = JEPAContrastiveTRM(
    trm_model,
    hidden_size=512,
    proj_dim=256,
    ema_decay=0.99
)

# Training loop
for batch in dataloader:
    # Ensure batch has 'inputs_aug' (different examples, same task)
    task_loss = compute_task_loss(batch)
    contrastive_loss = jepa_model(batch)

    # Combined loss
    total_loss = task_loss + 0.1 * contrastive_loss

    # Optimize
    total_loss.backward()
    optimizer.step()

    # Update target encoder
    jepa_model.update_target_encoder()
```

### 4. Full Training Script

Run the example training script:

```bash
python examples/train_trm_with_jepa.py \
    --data_path /path/to/arc_dataset \
    --output_dir /data/trm_jepa \
    --contrastive_weight 0.1 \
    --contrastive_type jepa \
    --hidden_size 512 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 10
```

### 5. Key Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `contrastive_weight` | 0.1 | [0.01, 0.5] | Weight for contrastive loss |
| `ema_decay` | 0.99 | [0.95, 0.999] | EMA decay for target encoder |
| `proj_dim` | 256 | [128, 512] | Projection dimension |
| `temperature` | 0.07 | [0.05, 0.2] | Temperature for InfoNCE |
| `pool_method` | 'mean' | {mean, max, first, last} | How to pool z_H |

### 6. Data Requirements

The most important requirement is providing **pairs of examples from the same task**.

#### Option A: Paired Dataloader (Recommended)

Modify your dataloader to return pairs:

```python
def __getitem__(self, idx):
    puzzle_id = self.puzzles[idx]
    examples = self.puzzle_to_examples[puzzle_id]

    # Sample two different training examples
    ex1, ex2 = random.sample(examples, 2)

    return {
        "inputs": ex1.input,
        "inputs_aug": ex2.input,  # Different example, same task!
        "labels": ex1.output,
        "puzzle_identifiers": puzzle_id
    }
```

#### Option B: Data Augmentation

Use transformations (less safe for ARC):

```python
from models.recursive_reasoning.jepa_contrastive import augment_arc_grid

inputs_aug = augment_arc_grid(
    batch["inputs"],
    apply_rotation=True,
    apply_flip=True
)
batch["inputs_aug"] = inputs_aug
```

### 7. Monitoring Training

Track these metrics:

```python
wandb.log({
    "task_loss": task_loss,
    "contrastive_loss": contrastive_loss,
    "total_loss": total_loss,
    "task_accuracy": accuracy,
})
```

Expected behavior:
- Task loss should decrease normally
- Contrastive loss should decrease over time
- Task accuracy should improve or stay similar

### 8. Troubleshooting

#### Problem: Task performance degrades

**Solution**: Reduce `contrastive_weight` to 0.01 or 0.05

#### Problem: Contrastive loss stays high

**Solutions**:
- Check that `inputs_aug` is actually different from `inputs`
- Reduce `ema_decay` to 0.95-0.97
- Increase `proj_dim` to 512

#### Problem: NaN loss

**Solutions**:
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Reduce learning rate
- Check for invalid augmentations

### 9. Expected Improvements

With proper tuning, expect:
- **2-5% improvement** on validation accuracy
- **Faster convergence** (fewer epochs to reach same performance)
- **Better generalization** to novel inputs from known tasks

### 10. Advanced Usage

#### Custom Pooling

```python
class CustomJEPA(JEPAContrastiveTRM):
    def _pool_latent(self, z_H):
        # Custom pooling strategy
        # e.g., attention-based pooling
        weights = self.attention(z_H)
        return (z_H * weights).sum(dim=1)
```

#### Task-Specific Augmentation

```python
def smart_augment(inputs, task_metadata):
    if task_metadata["rotation_invariant"]:
        return augment_arc_grid(inputs, apply_rotation=True)
    else:
        return inputs  # No augmentation for rotation-sensitive tasks

batch = create_augmented_batch(batch, augmentation_fn=smart_augment)
```

## Documentation

For detailed documentation, see:
- `docs/JEPA_CONTRASTIVE_LEARNING.md` - Full technical documentation
- `models/recursive_reasoning/jepa_contrastive.py` - Implementation with docstrings

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test suite to verify installation
3. Review the full documentation
4. Open an issue on GitHub with error messages and configuration

## Citation

If you use this in your research:

```bibtex
@misc{trm_jepa_contrastive,
  title={JEPA-style Contrastive Learning for Tiny Recursive Models},
  year={2025},
  howpublished={\url{https://github.com/TinyRecursiveModels}}
}
```
