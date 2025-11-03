"""
Train CNN Puzzle Embedding Predictor

Trains a CNN to predict TRM's puzzle embeddings from input grids.

Usage:
    python train.py \
        --data-path ./data/training_pairs/training_pairs.pt \
        --output-dir ./checkpoints \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --gpu 0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels/puzzle_embedding_predictor')

from models.cnn_encoder import PuzzleEmbeddingCNN, PuzzleEmbeddingLoss


class PuzzleEmbeddingDataset(Dataset):
    """
    Dataset for (grid, embedding) pairs.
    """

    def __init__(self, training_pairs: List[Dict], max_grid_size: int = 30):
        self.pairs = training_pairs
        self.max_grid_size = max_grid_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        grid = pair['grid']
        embedding = pair['embedding']

        # Pad grid to max_grid_size
        H, W = grid.shape
        padded_grid = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded_grid[:H, :W] = grid

        return {
            'grid': torch.from_numpy(padded_grid).long(),
            'embedding': torch.from_numpy(embedding).float(),
            'puzzle_id': pair['puzzle_id'],
            'puzzle_name': pair['puzzle_name']
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: PuzzleEmbeddingLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        grid = batch['grid'].to(device)
        target_embedding = batch['embedding'].to(device)

        # Forward
        pred_embedding = model(grid)

        # Compute loss
        losses = loss_fn(pred_embedding, target_embedding)

        # Backward
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        # Accumulate
        total_loss += losses['loss'].item()
        total_mse += losses['mse'].item()
        total_cosine += losses['cosine'].item()
        total_cosine_sim += losses['cosine_similarity'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': losses['loss'].item(),
            'cos_sim': losses['cosine_similarity'].item()
        })

    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'cosine': total_cosine / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: PuzzleEmbeddingLoss,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        grid = batch['grid'].to(device)
        target_embedding = batch['embedding'].to(device)

        # Forward
        pred_embedding = model(grid)

        # Compute loss
        losses = loss_fn(pred_embedding, target_embedding)

        # Accumulate
        total_loss += losses['loss'].item()
        total_mse += losses['mse'].item()
        total_cosine += losses['cosine'].item()
        total_cosine_sim += losses['cosine_similarity'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': losses['loss'].item(),
            'cos_sim': losses['cosine_similarity'].item()
        })

    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'cosine': total_cosine / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save latest
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save best
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best checkpoint (cos_sim={metrics['val_cosine_similarity']:.4f})")

    # Save periodic
    if epoch % 10 == 0:
        epoch_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)


def main():
    parser = argparse.ArgumentParser(description="Train puzzle embedding predictor")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/training_pairs/training_pairs.pt",
        help="Path to training pairs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=30,
        help="Maximum grid size (padding)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="CNN embedding dimension"
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=256,
        help="CNN hidden channels"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of residual blocks"
    )
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*70)
    print("Train Puzzle Embedding Predictor")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")

    # Load data
    print(f"\n📥 Loading training pairs from {args.data_path}")
    training_pairs = torch.load(args.data_path)
    print(f"✅ Loaded {len(training_pairs)} training pairs")

    # Create dataset
    dataset = PuzzleEmbeddingDataset(training_pairs, max_grid_size=args.max_grid_size)

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = PuzzleEmbeddingCNN(
        vocab_size=12,
        embedding_dim=args.embedding_dim,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        output_dim=512
    ).to(device)

    print(f"\n🏗️  Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = PuzzleEmbeddingLoss(mse_weight=1.0, cosine_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Training loop
    print(f"\n🚀 Starting training...")
    best_cosine_sim = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, epoch)

        # Step scheduler
        scheduler.step()

        # Log
        metrics = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_metrics['loss'],
            'train_cosine_similarity': train_metrics['cosine_similarity'],
            'val_loss': val_metrics['loss'],
            'val_cosine_similarity': val_metrics['cosine_similarity']
        }
        history.append(metrics)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, cos_sim={train_metrics['cosine_similarity']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, cos_sim={val_metrics['cosine_similarity']:.4f}")

        # Save checkpoint
        is_best = val_metrics['cosine_similarity'] > best_cosine_sim
        if is_best:
            best_cosine_sim = val_metrics['cosine_similarity']

        save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best)

    # Save training history
    history_path = output_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"   Best val cosine similarity: {best_cosine_sim:.4f}")
    print(f"   Checkpoints saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
