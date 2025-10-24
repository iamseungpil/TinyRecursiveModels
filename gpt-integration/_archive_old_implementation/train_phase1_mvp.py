#!/usr/bin/env python3
"""
Phase 1 MVP Training Script
Frozen LLaMA-8B + Trainable Grid Decoder
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels/gpt-integration')
sys.path.insert(0, '/home/ubuntu/gpt_oss_arc_final')

from models.hybrid_model import HybridARCModel_MVP
from data.arc_loader import create_arc_dataloaders
from arc import train_problems


def load_config(config_path: str) -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_loss(outputs: dict, target_grids: torch.Tensor) -> dict:
    """
    Compute training loss

    Args:
        outputs: Model outputs dict
        target_grids: [batch, 900] target grids

    Returns:
        loss_dict with total_loss and components
    """
    grid_logits = outputs['grid_logits']  # [batch, 900, vocab_size]
    target_grids = target_grids.to(grid_logits.device)

    # Cross-entropy loss
    loss = F.cross_entropy(
        grid_logits.view(-1, grid_logits.size(-1)),
        target_grids.view(-1),
        reduction='mean'
    )

    # Compute metrics
    grid_pred = outputs['grid_pred']
    exact_match = (grid_pred == target_grids).all(dim=1).float().mean()
    pixel_accuracy = (grid_pred == target_grids).float().mean()

    return {
        'total_loss': loss,
        'ce_loss': loss.item(),
        'exact_match': exact_match.item(),
        'pixel_accuracy': pixel_accuracy.item()
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_accumulation_steps: int,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_exact_match = 0.0
    total_pixel_acc = 0.0
    total_attempts = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Forward
        outputs = model(
            problem_texts=batch['problem_text'],
            problem_grids=batch['problem_grid'],
            target_grids=batch['target_grid']
        )

        # Compute loss
        loss_dict = compute_loss(outputs, batch['target_grid'])
        loss = loss_dict['total_loss']

        # Backward (with gradient accumulation)
        loss = loss / grad_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Track average attempts
        avg_attempts = outputs['attempts'].float().mean().item()

        # Accumulate metrics
        total_loss += loss_dict['ce_loss']
        total_exact_match += loss_dict['exact_match']
        total_pixel_acc += loss_dict['pixel_accuracy']
        total_attempts += avg_attempts
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['ce_loss']:.4f}",
            'exact': f"{loss_dict['exact_match']:.2f}",
            'pixel': f"{loss_dict['pixel_accuracy']:.3f}",
            'attempts': f"{avg_attempts:.1f}"
        })

    return {
        'train/loss': total_loss / num_batches,
        'train/exact_match': total_exact_match / num_batches,
        'train/pixel_accuracy': total_pixel_acc / num_batches,
        'train/avg_attempts': total_attempts / num_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> dict:
    """Evaluate on validation set"""
    model.eval()

    total_exact_match = 0.0
    total_pixel_acc = 0.0
    total_attempts = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        outputs = model.generate(
            problem_texts=batch['problem_text'],
            problem_grids=batch['problem_grid']
        )

        target_grids = batch['target_grid'].to(outputs['grid_pred'].device)
        exact_match = (outputs['grid_pred'] == target_grids).all(dim=1).float().mean()
        pixel_acc = (outputs['grid_pred'] == target_grids).float().mean()
        avg_attempts = outputs['attempts'].float().mean()

        total_exact_match += exact_match.item()
        total_pixel_acc += pixel_acc.item()
        total_attempts += avg_attempts.item()
        num_batches += 1

    return {
        'val/exact_match': total_exact_match / num_batches,
        'val/pixel_accuracy': total_pixel_acc / num_batches,
        'val/avg_attempts': total_attempts / num_batches
    }


def main():
    # Load config
    config_path = "/home/ubuntu/TinyRecursiveModels/gpt-integration/configs/phase1_mvp_llama8b.yaml"
    config = load_config(config_path)

    # Set device
    device = config['hardware']['device']
    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]

    # Initialize wandb
    wandb.init(
        project=config['logging']['wandb_project'],
        name=config['logging']['wandb_run_name'],
        config=config
    )

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_arc_dataloaders(
        num_train_problems=config['data']['num_train_problems'],
        num_val_problems=config['data']['num_val_problems'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        augmentation=config['data']['augmentation']
    )

    # Create model
    print("Creating model...")
    model = HybridARCModel_MVP(
        text_model_name=config['model']['text_model'],
        text_hidden_size=config['model']['text_hidden_size'],
        grid_hidden_size=config['model']['grid_hidden_size'],
        L_layers=config['model']['L_layers'],
        H_cycles=config['model']['H_cycles'],
        L_cycles=config['model']['L_cycles'],
        num_heads=config['model']['num_heads'],
        expansion=config['model']['expansion'],
        dropout=config['model']['dropout'],
        vocab_size=config['data']['vocab_size'],
        seq_len=config['data']['seq_len'],
        max_attempts=config['model']['max_attempts'],
        device=device
    )

    # Log parameter counts
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v/1e6:.1f}M")
    wandb.log(param_counts)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['optimization']['beta1'], config['optimization']['beta2'])
    )

    # Training loop
    print("\nStarting training...")
    best_val_exact_match = 0.0
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            grad_accumulation_steps=config['training']['grad_accumulation_steps'],
            device=device,
            epoch=epoch
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        wandb.log(metrics)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
        print(f"  Train Exact Match: {train_metrics['train/exact_match']:.2%}")
        print(f"  Train Avg Attempts: {train_metrics['train/avg_attempts']:.2f}")
        print(f"  Val Exact Match: {val_metrics['val/exact_match']:.2%}")
        print(f"  Val Avg Attempts: {val_metrics['val/avg_attempts']:.2f}")

        # Save best model
        if val_metrics['val/exact_match'] > best_val_exact_match:
            best_val_exact_match = val_metrics['val/exact_match']
            checkpoint_path = output_dir / "checkpoints" / "best_model.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_exact_match': best_val_exact_match,
                'config': config
            }, checkpoint_path)
            print(f"  💾 Saved best model (exact_match: {best_val_exact_match:.2%})")

    print(f"\n✅ Training complete! Best val exact match: {best_val_exact_match:.2%}")
    wandb.finish()


if __name__ == "__main__":
    main()
