"""
TRM Evaluation - Test pretrained TRM model on ARC

NO DUPLICATION: Imports from existing models/ and dataset/
"""

import sys
import os
from pathlib import Path
import argparse
import json
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.functions import load_model_class
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def load_checkpoint(checkpoint_path: str, device: str = "cuda:0"):
    """Load pretrained TRM checkpoint."""
    print(f"📦 Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model with same config
    model_config = {
        "batch_size": config.get("batch_size", 32),
        "seq_len": 900,  # Will be updated from metadata
        "vocab_size": 12,  # Will be updated
        "num_puzzle_identifiers": 1,  # Will be updated
        "puzzle_emb_ndim": 0,
        "H_cycles": config.get("H_cycles", 3),
        "L_cycles": config.get("L_cycles", 6),
        "H_layers": 1,
        "L_layers": config.get("L_layers", 4),
        "hidden_size": config.get("hidden_size", 512),
        "expansion": config.get("expansion", 4.0),
        "num_heads": config.get("num_heads", 8),
        "pos_encodings": config.get("pos_encodings", "rope"),
        "halt_max_steps": config.get("halt_max_steps", 16),
        "halt_exploration_prob": 0.0,  # No exploration during eval
    }

    model = TinyRecursiveReasoningModel_ACTV1(model_config)

    # Wrap with loss head
    loss_head_cls = load_model_class("losses@SeqLoss")
    model = loss_head_cls(model)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded from step {checkpoint['step']}")

    return model


def evaluate(
    model: torch.nn.Module,
    data_path: str,
    device: str = "cuda:0",
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate TRM model on test set."""

    # Load test dataset
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[data_path],
        rank=0,
        num_replicas=1
    )
    test_dataset = PuzzleDataset(dataset_config, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True
    )

    # Evaluation loop
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_steps = 0

    carry = None

    with torch.no_grad():
        for set_name, batch, global_batch_size in test_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Initialize carry
            if carry is None:
                carry = model.initial_carry(batch)

            # Multi-step inference (TRM style with ACT halting)
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[]
                )
                inference_steps += 1

                if all_finish:
                    break

            # Accumulate metrics
            total_loss += loss.item()
            total_steps += inference_steps
            total_examples += global_batch_size

            # Check accuracy (exact match)
            if "logits" in preds:
                pred_tokens = preds["logits"].argmax(dim=-1)
                target_tokens = batch["labels"]
                correct = (pred_tokens == target_tokens).all(dim=-1).sum().item()
                total_correct += correct

            print(f"Batch: {set_name}, Steps: {inference_steps}, Loss: {loss.item():.4f}")

    # Compute final metrics
    results = {
        "avg_loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "avg_steps": total_steps / total_examples,
        "total_examples": total_examples
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained TRM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Load model
    model = load_checkpoint(args.checkpoint, args.device)

    # Evaluate
    print("\n🔍 Starting evaluation...")
    results = evaluate(model, args.data_path, args.device)

    print("\n📊 Evaluation Results:")
    print(f"  Average Loss: {results['avg_loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Avg Inference Steps: {results['avg_steps']:.2f}")
    print(f"  Total Examples: {results['total_examples']}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
