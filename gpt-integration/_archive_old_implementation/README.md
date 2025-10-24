# Archived: Old MVP Implementation

This directory contains the superseded "Phase 1 MVP" implementation.

## Archived Date
2025-10-11

## Reason for Archival
Architectural pivot from self-correction MVP approach to TRM-style hybrid architecture.

## Old Architecture (Archived)
- **Self-correction loop**: Multiple attempts with feedback
- **GridGenerationModule**: Separate grid decoder module
- **HybridARCModel_MVP**: Original hybrid model design
- **Causal grid generation**: Sequential token generation

## New Architecture (Active)
- **TRM hierarchical reasoning**: z_H (high-level) + z_L (low-level) carry states
- **Non-causal grid generation**: Bidirectional attention
- **Frozen LLaMA-8B**: Text reasoning to z_init
- **EOS cropping**: Variable-sized grid support

## Contents
- `hybrid_model.py` - Old MVP model with self-correction
- `grid_generation.py` - Old grid generation module
- `train_phase1_mvp.py` - Old training script
- `test_self_correction.py` - Old test file
- `smoke_test.py` - Old smoke test
- `components/` - Old component modules
- `phase1_mvp/` - Old experiment directory
- `wandb_runs/` - Old experiment logs
- `old_logs/` - Old training logs
- `SELF_CORRECTION_IMPLEMENTATION.md` - Old design documentation
- `self_correction_flow.txt` - Old architecture diagram

## Active Implementation
See:
- `../models/trm_hybrid.py` - Current TRM-style hybrid model
- `../training/train_trm_hybrid.py` - Current training script
- `../TRM_IMPLEMENTATION.md` - Current architecture documentation
