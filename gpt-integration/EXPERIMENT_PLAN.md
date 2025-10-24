# Experiment Plan: Hybrid LLaMA + TRM for ARC-AGI

## Goal
Combine LLaMA-8B's text reasoning with TinyRecursiveModels' hierarchical grid generation for ARC-AGI.

## Hypothesis
Text reasoning provides high-level understanding, while hierarchical grid generation provides structured output.

## Phases

### Phase 1: MVP (Week 1)
**Objective**: Validate basic architecture

**Model**:
- Text: Frozen LLaMA-8B (meta-llama/Llama-3.2-8B)
- Grid: Trainable hierarchical decoder (1024-dim, L=4, H_cycles=2, L_cycles=3)
- Connection: z_init = last_hidden_state[:, -1, :]

**Data**: 10 ARC problems

**Hyperparameters**:
```yaml
batch_size: 1
grad_accumulation: 8
learning_rate: 1e-4
epochs: 50
max_attempts: 3
```

**Success Criteria**:
- ✅ >50% exact match on 10 problems
- ✅ <40GB GPU memory
- ✅ <6 hours training time

### Phase 2: Cross-Attention (Week 2)
**Addition**: Cross-attention from grid decoder to full text hidden states

**Expected improvement**: +20% exact match

### Phase 3: Self-Correction (Week 3)
**Addition**:
- Unfreeze LLaMA-8B (fine-tune)
- Error feedback loop
- Iterative refinement

**Expected improvement**: +10% exact match

### Phase 4: Q-Learning (Week 4)
**Addition**:
- Q-head for adaptive halting
- Exploration strategy
- Compute optimization

**Expected improvement**: -30% compute, maintain accuracy

## Technical Details

### Information Flow
```
Text (variable length) → z_init [1, 4096]
                         ↓ (linear projection)
                    z_init [1, 1024]
                         ↓ (broadcast)
            z_H, z_L [batch, 916, 1024]
                         ↓ (hierarchical reasoning)
                  Grid [batch, 900]
```

### Memory Budget
- LLaMA-8B: ~16GB
- Grid decoder: ~8GB
- Activations: ~12GB
- Buffer: ~4GB
- Total: ~40GB (fits A100)

### Training Strategy
1. Freeze LLaMA-8B (Phase 1-2)
2. Fine-tune LLaMA-8B with small LR (Phase 3-4)
3. Gradient checkpointing for all phases

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Information bottleneck | Phase 2 cross-attention |
| Memory overflow | Gradient checkpointing, batch_size=1 |
| Training instability | Small LR, gradient clipping |
| Overfitting (10 problems) | Data augmentation, early stopping |

## Evaluation Metrics

1. **Exact Match**: % of problems with 100% correct grid
2. **Pixel Accuracy**: Average % of correct pixels
3. **Pass@K**: Success rate with K attempts
4. **Avg Attempts**: Average number of attempts needed
5. **Reasoning Quality**: Manual inspection of text output

## Timeline

- **Day 1-2**: Implementation (data loader, models)
- **Day 3-4**: Phase 1 training
- **Day 5**: Phase 1 evaluation & analysis
- **Day 6-7**: Phase 2 implementation (if Phase 1 succeeds)

## Hardware
- **GPU**: CUDA_VISIBLE_DEVICES=3 (40GB A100)
- **CPU**: 32 cores
- **RAM**: 128GB

## Monitoring
- W&B project: `gpt-integration-arc`
- Metrics logged every 10 steps
- Checkpoints saved every epoch
