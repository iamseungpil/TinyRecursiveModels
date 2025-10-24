# GPT-Integration: Hybrid Text Reasoning + Grid Generation for ARC-AGI

## Overview

This project combines LLaMA-8B's text reasoning capabilities with TinyRecursiveModels' hierarchical grid generation to solve ARC-AGI puzzles.

## Architecture

```
Text Reasoning (LLaMA-8B) → z_init → Grid Generation (TRM-style) → Grid Prediction
         ↓                                                                ↓
    Reasoning text                                                   Verification
         ↑                                                                ↓
    Self-correction ←──────────────────────────────────────────── If wrong
```

## Phases

### Phase 1: MVP (Current)
- **Model**: Frozen LLaMA-8B + Trainable grid decoder
- **Data**: 10 ARC problems
- **Target**: >50% exact match

### Phase 2: Cross-Attention
- **Addition**: Cross-attention from grid to text
- **Target**: >70% exact match

### Phase 3: Self-Correction
- **Addition**: Iterative refinement with error feedback
- **Target**: >80% exact match

### Phase 4: Q-Learning
- **Addition**: Adaptive halting with Q-head
- **Target**: >85% exact match, reduced compute

## Quick Start

```bash
# Setup environment
cd /home/ubuntu/TinyRecursiveModels/gpt-integration
bash scripts/setup_environment.sh

# Run Phase 1 MVP
CUDA_VISIBLE_DEVICES=3 bash scripts/run_phase1_mvp.sh

# Evaluate
bash scripts/evaluate_checkpoint.sh experiments/phase1_mvp/checkpoints/best_model.pt
```

## Requirements

- PyTorch 2.1+
- Transformers 4.35+
- CUDA 11.8+
- 40GB GPU (A100)

## Directory Structure

See `EXPERIMENT_PLAN.md` for detailed structure.

## Results

| Phase | Problems | Exact Match | Pass@3 | Time |
|-------|----------|-------------|--------|------|
| Phase 1 MVP | 10 | TBD | TBD | TBD |

## Citation

Based on TinyRecursiveModels and LLaMA-3.2.
