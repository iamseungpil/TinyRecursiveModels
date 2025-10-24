"""
TRM Pretraining - Simplified single-GPU version of pretrain.py

NO DUPLICATION: Imports TRM model and losses from /home/ubuntu/TinyRecursiveModels/models/
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

# Import existing modules (NO DUPLICATION)
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry
)
from utils.functions import load_model_class
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


class TRMPretrainConfig:
    """Configuration for TRM pretraining."""

    def __init__(
        self,
        # Data
        data_path: str,
        output_dir: str = "/data/trm/pretrain",
        # Architecture
        hidden_size: int = 512,
        num_heads: int = 8,
        expansion: float = 4.0,
        H_cycles: int = 3,
        L_cycles: int = 6,
        L_layers: int = 4,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        pos_encodings: str = "rope",
        # Training
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-4,
        lr_min_ratio: float = 0.1,
        lr_warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        # Logging
        project_name: str = "arc-trm-pretrain",
        run_name: Optional[str] = None,
        log_interval: int = 10,
        eval_interval: int = 1000,
        checkpoint_interval: int = 5000,
        # Hardware
        device: str = "cuda:0",
        seed: int = 42
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.L_layers = L_layers
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.pos_encodings = pos_encodings
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min_ratio = lr_min_ratio
        self.lr_warmup_steps = lr_warmup_steps
        self.weight_decay = weight_decay
        self.project_name = project_name
        self.run_name = run_name or f"trm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.seed = seed

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items()}


def create_model(config: TRMPretrainConfig, metadata: Dict[str, Any]) -> nn.Module:
    """
    Create TRM model with loss head.

    Reuses existing model from models/recursive_reasoning/trm.py
    """
    model_config = {
        "batch_size": config.batch_size,
        "seq_len": metadata["seq_len"],
        "vocab_size": metadata["vocab_size"],
        "num_puzzle_identifiers": metadata["num_puzzle_identifiers"],
        "puzzle_emb_ndim": 0,  # No puzzle embeddings
        "puzzle_emb_len": 0,   # No puzzle embedding length
        "H_cycles": config.H_cycles,
        "L_cycles": config.L_cycles,
        "H_layers": 1,  # Not used
        "L_layers": config.L_layers,
        "hidden_size": config.hidden_size,
        "expansion": config.expansion,
        "num_heads": config.num_heads,
        "pos_encodings": config.pos_encodings,
        "halt_max_steps": config.halt_max_steps,
        "halt_exploration_prob": config.halt_exploration_prob,
    }

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(model_config)

    # Wrap with loss head (import from existing losses.py)
    loss_head_cls = load_model_class("losses@ACTLossHead")
    model = loss_head_cls(model, loss_type="stablemax_cross_entropy")

    # Move to device
    model = model.to(config.device)

    return model


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_ratio: float = 0.1
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    carry: Any,
    optimizer: torch.optim.Optimizer,
    config: TRMPretrainConfig,
    step: int,
    total_steps: int
) -> Dict[str, float]:
    """Single training step."""
    model.train()

    # Move batch to device
    batch = {k: v.to(config.device) for k, v in batch.items()}

    # Initialize carry if needed (model already on device)
    if carry is None:
        carry = model.initial_carry(batch)

    # Forward pass
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

    # Backward pass (normalize by batch size for consistent gradients)
    # ACTLossHead sums loss over batch, so divide by batch_size
    normalized_loss = loss / config.batch_size
    normalized_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update learning rate
    lr = cosine_schedule_with_warmup(
        step, total_steps, config.lr_warmup_steps, config.lr, config.lr_min_ratio
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Collect metrics
    result = {
        "loss": loss.item(),
        "lr": lr,
    }

    # Add other metrics
    for k, v in metrics.items():
        if not v.requires_grad:
            result[k] = v.item()

    return result, carry


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TRMPretrainConfig
):
    """Save model checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict()
    }, checkpoint_path)

    print(f"💾 Checkpoint saved: {checkpoint_path}")


def train(config: TRMPretrainConfig):
    """Main training loop."""
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create dataset
    print("📦 Loading dataset...")
    dataset_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[config.data_path],
        global_batch_size=config.batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    train_dataset = PuzzleDataset(dataset_config, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True
    )

    metadata = train_dataset.metadata

    # Create model
    print("🔧 Creating TRM model...")
    model = create_model(config, metadata.__dict__)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Calculate total steps
    total_steps = config.epochs * (metadata.total_groups * metadata.mean_puzzle_examples) // config.batch_size
    print(f"📊 Total training steps: {total_steps}")

    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.to_dict()
    )

    # Training loop
    step = 0
    carry = None

    for epoch in range(config.epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{config.epochs}")

        for set_name, batch, global_batch_size in train_loader:
            step += 1

            # Train step
            metrics, carry = train_step(
                model, batch, carry, optimizer, config, step, total_steps
            )

            # Log metrics
            if step % config.log_interval == 0:
                wandb.log(metrics, step=step)
                print(f"Step {step}/{total_steps}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}")

            # Save checkpoint
            if step % config.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step, config)

            if step >= total_steps:
                break

        if step >= total_steps:
            break

    # Final checkpoint
    save_checkpoint(model, optimizer, step, config)

    wandb.finish()
    print("✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="TRM Pretraining (Single GPU)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ARC dataset")
    parser.add_argument("--output_dir", type=str, default="/data/trm/pretrain", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = TRMPretrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed
    )

    train(config)


if __name__ == "__main__":
    main()
