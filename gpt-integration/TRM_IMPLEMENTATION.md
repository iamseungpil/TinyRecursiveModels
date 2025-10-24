# TRM-Style Hybrid Model Implementation

## Overview

This directory contains a TRM (Tiny Recursive Model)-style hybrid architecture for ARC-AGI tasks, combining:
1. **LLaMA-8B** (frozen) for text reasoning → z_init extraction
2. **TRM hierarchical reasoning** with carry state persistence
3. **Non-causal grid generation** with EOS markers

## Architecture Summary

### Key Differences from Original Implementation

| Feature | Original (self_correction.py) | TRM Hybrid (trm_hybrid.py) |
|---------|------------------------------|----------------------------|
| Text Generation | LLaMA causal generation | LLaMA frozen z_init extraction |
| Grid Generation | Autoregressive with self-correction | Non-causal with hierarchical reasoning |
| State Management | No carry state | Persistent carry state (z_H, z_L) |
| Reasoning Cycles | Single pass | H_cycles × L_cycles hierarchical |
| Evaluation | Direct comparison | EOS cropping for variable sizes |

## Files Implemented

### 1. models/trm_hybrid.py

**Core Components:**

```python
@dataclass
class HybridTRMInnerCarry:
    z_H: torch.Tensor  # [batch, 916, hidden_size] - High-level reasoning
    z_L: torch.Tensor  # [batch, 916, hidden_size] - Low-level reasoning

class HybridTRMConfig:
    # TRM parameters
    H_cycles: int = 2        # High-level reasoning cycles
    L_cycles: int = 1        # Low-level cycles per H cycle
    L_layers: int = 4        # Transformer layers
    hidden_size: int = 512   # TRM hidden size
    text_hidden_size: int = 4096  # LLaMA hidden size

class HybridTRM_Inner(nn.Module):
    # Key methods:
    # - _input_embeddings(z_init, problem_grid): Project z_init and embed grid
    # - forward(carry, z_init, problem_grid): Hierarchical reasoning
    # - reset_carry(reset_flag, carry): Reset state for new problems

class HybridTRMModel(nn.Module):
    # Wrapper combining LLaMA + TRM Inner
    # - text_module: Frozen LLaMA-8B for z_init extraction
    # - inner: TRM hierarchical reasoning module
```

**Architecture Flow:**

```
Input: problem_texts, problem_grids
  ↓
[LLaMA-8B (frozen)]
  ↓ z_init [batch, 4096]
  ↓
[z_projection] → [batch, 512]
  ↓
[Grid Embedding] + z_init
  ↓
[TRM Hierarchical Reasoning]
  H_cycles-1: no_grad (efficiency)
  Last cycle: with_grad (training)
    L_cycles × [L_level reasoning]
    → Update z_H, z_L
  ↓
[LM Head] → logits [batch, 900, 12]
```

**Key Implementation Details:**

1. **Non-causal Attention**: `causal=False` in Attention layers for bidirectional grid reasoning
2. **Carry State Persistence**: z_H and z_L persist across reasoning cycles
3. **Gradient Efficiency**: Only last H_cycle has gradients enabled
4. **Input Injection**: z_init is broadcast and added to every position

### 2. evaluation/eos_utils.py

**EOS Cropping Logic:**

```python
@numba.njit
def _crop(grid: np.ndarray):
    """Find maximum rectangle without EOS tokens"""
    # Scan for EOS=1 or PAD=0 or invalid (>11)
    # Return cropped grid with colors 0-9

def compute_loss_with_crop(logits, targets):
    """Cross-entropy loss only on valid positions"""
    # Mask: (targets >= 2) & (targets <= 11)
    # Ignore PAD=0 and EOS=1 positions

def compute_exact_match(pred_grids, target_grids):
    """Exact match after cropping both predictions and targets"""
```

**Token Encoding:**
- PAD = 0
- EOS = 1
- Colors 0-9 = tokens 2-11
- Vocab size = 12

### 3. training/train_trm_hybrid.py

**Training Loop Structure:**

```python
def train_step(model, batch, carry, optimizer, config):
    # 1. Initialize carry if None (first step)
    if carry is None:
        carry = model.initial_carry(batch_size)
        carry = model.inner.reset_carry(reset_flag, carry)

    # 2. Forward (single step for Phase 1)
    carry, logits = model(carry, batch_input)

    # 3. Compute loss with EOS cropping
    loss = compute_loss_with_crop(logits, targets)

    # 4. Backward + optimizer step
    loss.backward()
    optimizer.step()

    return carry, loss, metrics

def evaluate(model, val_loader, config):
    # Use EOS cropping for evaluation
    preds_cropped = crop_grid_from_eos(logits.argmax(-1))
    targets_cropped = crop_grid_from_eos(targets)
    exact_match = compute_exact_match(preds_cropped, targets_cropped)
```

**Training Configuration:**

```python
config = TrainConfig(
    # Model
    batch_size=1,
    H_cycles=2,
    L_cycles=1,
    L_layers=4,
    hidden_size=512,

    # Training
    num_train_problems=10,
    num_val_problems=5,
    num_epochs=10,
    lr=1e-4,
    warmup_steps=100,
    gradient_clip=1.0
)
```

## Testing

### Component Tests

```bash
# Test EOS utilities
python evaluation/eos_utils.py

# Test TRM inner model (without LLaMA)
python test_trm_components.py

# Expected output:
✓ Inner model created
✓ Carry reset successful
✓ Forward pass successful
  Logits shape: torch.Size([2, 900, 12])
  Carry z_H shape: torch.Size([2, 900, 512])
  Carry z_L shape: torch.Size([2, 900, 512])
✓ Backward pass successful
✅ TRM Hybrid Inner Model test passed!
```

### Full Model Test (with LLaMA)

```bash
# Note: Requires GPU and LLaMA model access
CUDA_VISIBLE_DEVICES=0 python models/trm_hybrid.py

# Expected output:
Loading meta-llama/Llama-3.2-8B-Instruct...
✓ LLaMA-8B frozen
✓ Hybrid TRM Model initialized
✓ Logits shape: torch.Size([2, 900, 12])
✓ Carry z_H shape: torch.Size([2, 900, 512])
✓ Carry z_L shape: torch.Size([2, 900, 512])
✅ HybridTRMModel test passed!
```

## Training

### Basic Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_trm_hybrid.py \
    --num_train_problems 10 \
    --num_val_problems 5 \
    --num_epochs 10 \
    --batch_size 1 \
    --lr 1e-4 \
    --project_name arc-trm-hybrid \
    --run_name experiment_1
```

### Advanced Training Configuration

```bash
python training/train_trm_hybrid.py \
    --H_cycles 3 \
    --L_cycles 2 \
    --L_layers 6 \
    --hidden_size 768 \
    --num_train_problems 50 \
    --num_val_problems 10 \
    --num_epochs 20 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --gradient_clip 0.5
```

## Implementation Checklist

- ✅ **models/trm_hybrid.py**: TRM hierarchical reasoning with LLaMA integration
  - ✅ HybridTRMInnerCarry dataclass
  - ✅ HybridTRMConfig with all TRM parameters
  - ✅ HybridTRM_Inner with hierarchical reasoning
  - ✅ HybridTRMModel wrapper with frozen LLaMA

- ✅ **evaluation/eos_utils.py**: EOS cropping utilities
  - ✅ `_crop()`: Numba-optimized grid cropping
  - ✅ `crop_grid_from_eos()`: Torch wrapper
  - ✅ `compute_loss_with_crop()`: Masked loss computation
  - ✅ `compute_exact_match()`: Accuracy metric

- ✅ **training/train_trm_hybrid.py**: Training loop
  - ✅ `train_step()`: Single training step with carry state
  - ✅ `evaluate()`: Validation with EOS cropping
  - ✅ Cosine learning rate schedule with warmup
  - ✅ Gradient clipping
  - ✅ Wandb logging
  - ✅ Checkpoint saving

- ✅ **Supporting files**:
  - ✅ data/arc_loader.py (already correct)
  - ✅ models/text_reasoning.py (already correct)
  - ✅ models/common.py (copied from TRM)
  - ✅ models/layers.py (copied from TRM)

## Design Rationale

### Why TRM Architecture?

1. **Carry State Persistence**: Maintains reasoning state across cycles, enabling multi-step reasoning
2. **Hierarchical Reasoning**: H_cycles (high-level) × L_cycles (low-level) enables compositional thinking
3. **Gradient Efficiency**: Only last cycle has gradients, reducing memory and computation
4. **Non-causal Generation**: Grid positions can attend to all other positions (not sequential)

### Why Remove Self-Correction Loop?

The original implementation had:
```python
for attempt in range(max_attempts):
    generate_grid()
    if correct:
        break
```

TRM approach:
- Single forward pass with hierarchical reasoning cycles
- More efficient: H_cycles × L_cycles reasoning instead of max_attempts trial-and-error
- Better gradient flow: All reasoning is differentiable
- Matches TRM's proven architecture on other reasoning tasks

### EOS Cropping vs Fixed Size

- **Problem**: ARC grids have variable sizes (1x1 to 30x30)
- **Solution**: EOS markers indicate grid boundaries
- **Benefit**: Model learns to generate appropriately sized grids
- **Implementation**: Numba-optimized rectangle finding algorithm

## Next Steps (Phase 2+)

1. **Multi-step Reasoning**: Enable `halt_max_steps > 1` for adaptive computation
2. **Cross-attention**: Use full hidden states from LLaMA for richer conditioning
3. **Curriculum Learning**: Start with small grids, progress to larger ones
4. **Data Augmentation**: Rotation, reflection, color permutation
5. **Ensemble Methods**: Multiple reasoning paths with voting

## References

- **TRM Paper**: Tiny Recursive Models (original architecture)
- **TRM Code**: `/home/ubuntu/TinyRecursiveModels/models/recursive_reasoning/trm.py`
- **ARC-AGI**: Abstraction and Reasoning Corpus dataset
- **Implementation**: Based on `pretrain.py` and `arc.py` from TinyRecursiveModels

## File Locations

```
/home/ubuntu/TinyRecursiveModels/gpt-integration/
├── models/
│   ├── trm_hybrid.py          # Main TRM hybrid model
│   ├── text_reasoning.py      # LLaMA wrapper
│   ├── common.py              # Initialization utilities
│   └── layers.py              # Transformer layers
├── training/
│   └── train_trm_hybrid.py    # Training loop
├── evaluation/
│   └── eos_utils.py           # EOS cropping utilities
├── data/
│   └── arc_loader.py          # ARC dataset loader
└── test_trm_components.py     # Component tests
```
