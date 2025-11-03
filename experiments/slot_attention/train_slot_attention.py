#!/usr/bin/env python3
"""
Training script for TRM with Slot Attention + Contrastive Learning.

This experiment extends TRM with:
1. Slot Attention for compositional decomposition of z_H
2. Dual prediction heads (direct + slot-based)
3. Contrastive learning with Hungarian matching for slot alignment

Usage:
    # From project root
    cd experiments/slot_attention
    python train_slot_attention.py

    # Or with distributed training
    torchrun --nproc_per_node=4 train_slot_attention.py
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig

# Import main training loop from pretrain.py
from pretrain import launch


@hydra.main(config_path="configs", config_name="trm_slots", version_base=None)
def main(config: DictConfig):
    """
    Launch training with Slot Attention configuration.

    The main training loop is reused from pretrain.py. This script just
    provides the specific configuration for the slot attention experiment.
    """
    print("=" * 80)
    print("TRM with Slot Attention + Contrastive Learning")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Model: {config.arch.name}")
    print(f"  Loss: {config.arch.loss.name}")
    print(f"  Num Slots: {config.arch.num_slots}")
    print(f"  Slot Dim: {config.arch.slot_dim}")
    print(f"  Slot Iterations: {config.arch.slot_iterations}")
    print(f"  Slot Recon Weight: {config.arch.loss.slot_recon_weight}")
    print(f"  Slot Contrastive Weight: {config.arch.loss.slot_contrastive_weight}")
    print(f"  Hungarian Matching: {config.arch.loss.use_hungarian_matching}")
    print()
    print(f"  Batch Size: {config.global_batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.lr}")
    print()
    print("=" * 80)
    print()

    # Launch training
    launch(config)


if __name__ == "__main__":
    main()
