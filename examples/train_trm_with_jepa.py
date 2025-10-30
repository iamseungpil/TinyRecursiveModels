"""
Example: Training TRM with JEPA Contrastive Learning

This script demonstrates how to integrate JEPA-style contrastive learning
with the existing TRM training pipeline. It combines:
1. Standard TRM task loss (cross-entropy on predictions)
2. JEPA contrastive loss (same task → similar latents)

Usage:
    python examples/train_trm_with_jepa.py --data_path /path/to/data --config config.json

Key differences from standard TRM training:
- Requires pairs of examples from same task (inputs_aug)
- Adds contrastive loss term with tunable weight
- Updates EMA target encoder each step
- Can significantly improve generalization
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent
sys.path.insert(0, str(trm_root))

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config
)
from models.recursive_reasoning.jepa_contrastive import (
    JEPAContrastiveTRM,
    InfoNCEContrastiveTRM,
    create_augmented_batch,
    augment_arc_grid
)
from utils.functions import load_model_class


class JEPATRMTrainer:
    """
    Trainer that combines TRM task loss with JEPA contrastive loss.

    Args:
        trm_model: Base TRM model with loss head
        contrastive_weight: Weight for contrastive loss term
        ema_decay: EMA decay for target encoder
        contrastive_type: 'jepa' or 'infonce'
        use_augmentation: Whether to use data augmentation
    """

    def __init__(
        self,
        trm_model: nn.Module,
        hidden_size: int = 512,
        contrastive_weight: float = 0.1,
        ema_decay: float = 0.99,
        contrastive_type: str = 'jepa',
        use_augmentation: bool = False,
        device: str = 'cuda'
    ):
        self.device = device

        # Extract base TRM model (remove loss head for contrastive learning)
        # Assumes trm_model is wrapped with ACTLossHead
        if hasattr(trm_model, 'model'):
            base_model = trm_model.model
        else:
            base_model = trm_model

        # Keep original model for task loss
        self.task_model = trm_model

        # Create contrastive model
        if contrastive_type == 'jepa':
            self.contrastive_model = JEPAContrastiveTRM(
                base_model,
                hidden_size=hidden_size,
                ema_decay=ema_decay
            ).to(device)
        elif contrastive_type == 'infonce':
            self.contrastive_model = InfoNCEContrastiveTRM(
                base_model,
                hidden_size=hidden_size
            ).to(device)
        else:
            raise ValueError(f"Unknown contrastive_type: {contrastive_type}")

        self.contrastive_weight = contrastive_weight
        self.use_augmentation = use_augmentation
        self.contrastive_type = contrastive_type

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        carry: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Single training step with both task loss and contrastive loss.

        Args:
            batch: Dictionary with 'inputs', 'labels', 'puzzle_identifiers'
            optimizer: Optimizer for model parameters
            carry: TRM carry state (optional)

        Returns:
            Dictionary of metrics
        """
        self.task_model.train()
        self.contrastive_model.encoder.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # === 1. Task loss (standard TRM objective) ===
        if carry is None:
            carry = self.task_model.initial_carry(batch)

        carry, task_loss, metrics, _, _ = self.task_model(
            carry=carry,
            batch=batch,
            return_keys=[]
        )

        # === 2. Contrastive loss ===
        contrastive_loss = torch.tensor(0.0, device=self.device)

        if self.contrastive_weight > 0:
            # Create augmented batch
            if self.use_augmentation:
                batch = create_augmented_batch(
                    batch,
                    augmentation_fn=augment_arc_grid
                )
            else:
                # Assume dataloader provides inputs_aug
                if "inputs_aug" not in batch:
                    # Fallback: use same inputs (no contrastive learning)
                    batch["inputs_aug"] = batch["inputs"]

            # Compute contrastive loss
            contrastive_loss = self.contrastive_model(batch)

        # === 3. Combined loss ===
        # Note: task_loss is summed over batch (from ACTLossHead)
        batch_size = batch["inputs"].shape[0]
        normalized_task_loss = task_loss / batch_size

        total_loss = normalized_task_loss + self.contrastive_weight * contrastive_loss

        # === 4. Backward and optimize ===
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.task_model.parameters(), 1.0)
        optimizer.step()

        # === 5. Update EMA target encoder (JEPA only) ===
        if self.contrastive_type == 'jepa':
            self.contrastive_model.update_target_encoder()

        # === 6. Collect metrics ===
        result = {
            "loss": task_loss.item(),
            "task_loss": task_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "total_loss": total_loss.item() * batch_size,  # Unnormalize for logging
        }

        # Add other metrics from task model
        for k, v in metrics.items():
            if not v.requires_grad:
                result[k] = v.item()

        return result, carry


def create_trm_model(
    metadata: Dict[str, Any],
    hidden_size: int = 512,
    num_heads: int = 8,
    H_cycles: int = 3,
    L_cycles: int = 6,
    L_layers: int = 4,
    device: str = 'cuda'
) -> nn.Module:
    """Create TRM model with loss head."""
    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=32,
        seq_len=metadata["seq_len"],
        vocab_size=metadata["vocab_size"],
        num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
        puzzle_emb_ndim=0,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        H_layers=1,
        L_layers=L_layers,
        hidden_size=hidden_size,
        expansion=4.0,
        num_heads=num_heads,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
    )

    # Create base model
    model = TinyRecursiveReasoningModel_ACTV1(model_config)

    # Wrap with loss head
    loss_head_cls = load_model_class("losses@ACTLossHead")
    model = loss_head_cls(model, loss_type="stablemax_cross_entropy")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train TRM with JEPA contrastive learning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="/data/trm_jepa", help="Output directory")
    parser.add_argument("--config", type=str, help="Config JSON file")

    # Training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_type", type=str, default="jepa", choices=["jepa", "infonce"])
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")

    # Model args
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--L_layers", type=int, default=4)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

    print("=" * 80)
    print("Training TRM with JEPA Contrastive Learning")
    print("=" * 80)
    print(f"Contrastive type: {args.contrastive_type}")
    print(f"Contrastive weight: {args.contrastive_weight}")
    print(f"Use augmentation: {args.use_augmentation}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 80)

    # Set seed
    torch.manual_seed(args.seed)

    # Create metadata (load from dataset)
    # This is a placeholder - in real training, load from dataset
    metadata = {
        "seq_len": 916,  # 16 (puzzle) + 900 (input)
        "vocab_size": 11,  # ARC colors 0-10
        "num_puzzle_identifiers": 1200  # Total puzzles
    }

    # Create model
    print("Creating model...")
    model = create_trm_model(
        metadata,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        L_layers=args.L_layers,
        device=args.device
    )

    # Create trainer
    trainer = JEPATRMTrainer(
        model,
        hidden_size=args.hidden_size,
        contrastive_weight=args.contrastive_weight,
        contrastive_type=args.contrastive_type,
        use_augmentation=args.use_augmentation,
        device=args.device
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    print("Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Initialize wandb
    wandb.init(
        project="arc-trm-jepa",
        config=vars(args)
    )

    print("\n" + "=" * 80)
    print("Training loop would start here...")
    print("=" * 80)
    print("\nTo complete integration:")
    print("1. Load your dataset with DataLoader")
    print("2. Ensure dataset provides 'inputs_aug' (different examples, same task)")
    print("3. Call trainer.train_step(batch, optimizer) in training loop")
    print("4. Log metrics and save checkpoints")
    print("\nExample:")
    print("    for epoch in range(epochs):")
    print("        for batch in dataloader:")
    print("            metrics, carry = trainer.train_step(batch, optimizer)")
    print("            wandb.log(metrics)")


if __name__ == "__main__":
    main()
