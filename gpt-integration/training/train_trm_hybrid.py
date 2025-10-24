"""
Training Loop for TRM Hybrid Model

Based on TinyRecursiveModels/pretrain.py but adapted for ARC-AGI hybrid architecture
"""
import os
import sys
import math
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import argparse
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append('/home/ubuntu/TinyRecursiveModels/gpt-integration')
sys.path.append('/home/ubuntu/TinyRecursiveModels')

from models.trm_hybrid import HybridTRMModel
from data.arc_loader import create_arc_dataloaders, tokens_to_grid, grid_to_string
from evaluation.eos_utils import compute_loss_with_crop, compute_exact_match


@dataclass
class TrainConfig:
    """Training configuration"""
    # Model config
    batch_size: int = 1
    seq_len: int = 900
    vocab_size: int = 12
    H_cycles: int = 2
    L_cycles: int = 1
    L_layers: int = 4
    hidden_size: int = 512
    text_hidden_size: int = 4096
    expansion: float = 2.0
    num_heads: int = 8
    pos_encodings: str = 'rope'
    text_model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'

    # Training hyperparams
    num_train_problems: int = 400  # Expanded from 10 to avoid overfitting
    num_val_problems: int = 100    # Expanded from 5 for better evaluation
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # Self-correction hyperparams
    max_correction_attempts: int = 16  # Maximum correction attempts per sample (matching TRM halt_max_steps)
    correction_loss_weight: float = 0.5  # Weight for correction attempt losses

    # Text generation hyperparameters
    text_max_new_tokens: int = 128
    text_temperature: float = 0.6
    text_top_p: float = 0.9

    # Logging
    project_name: str = "arc-trm-hybrid"
    run_name: Optional[str] = None
    log_interval: int = 10
    eval_interval: int = 1  # Evaluate every N epochs
    save_dir: str = "./checkpoints"

    # System
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


def grid_to_feedback(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> str:
    """Translate token-level mismatches into concise textual feedback.

    Simplified to avoid overwhelming LLaMA with long grid strings.
    """

    pred_grid = tokens_to_grid(pred_tokens.cpu())
    target_grid = tokens_to_grid(target_tokens.cpu())

    if pred_grid.shape != target_grid.shape:
        return (
            f"\nSize mismatch: predicted {pred_grid.shape[0]}x{pred_grid.shape[1]}, "
            f"but target is {target_grid.shape[0]}x{target_grid.shape[1]}."
        )

    diff = pred_grid != target_grid
    coords = list(zip(*np.where(diff)))

    if not coords:
        return "\nPrediction is correct!"

    # Show only top 5 coords to keep feedback concise
    sample = coords[:5]
    num_errors = len(coords)
    coord_text = ", ".join(f"({r},{c})" for r, c in sample)

    more_text = f" ({num_errors - 5} more)" if num_errors > 5 else ""

    return (
        f"\n{num_errors} cells incorrect. Key errors at: {coord_text}{more_text}. "
        "Re-evaluate the transformation rule."
    )


def run_self_correction(
    model: HybridTRMModel,
    base_texts: List[str],
    problem_grids: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
    carry: Optional[Any] = None,
    training: bool = True
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, Any]:
    """Shared self-correction routine for both training and evaluation.

    Returns carry for TRM-style batch-to-batch persistence.
    """

    device = problem_grids.device
    batch_size = problem_grids.shape[0]

    # Initialize carry if None (TRM-style)
    if carry is None:
        carry = model.initial_carry(batch_size)
        reset_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        carry = model.inner.reset_carry(reset_mask, carry)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    feedbacks = ["" for _ in range(batch_size)]
    latent_prefixes: List[Optional[torch.Tensor]] = [None] * batch_size
    latent_cache = torch.zeros(
        batch_size,
        model.text_module.hidden_size,
        device=device,
        dtype=torch.float32
    )
    attempt_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    weighted_losses: List[torch.Tensor] = []
    weight_sum = 0.0
    final_preds = None

    for attempt in range(config.max_correction_attempts):
        if not active_mask.any():
            break

        prompts = []
        prefixes = []
        indices = []
        for idx in range(batch_size):
            if not active_mask[idx]:
                continue
            prompts.append(base_texts[idx] + feedbacks[idx])
            prefixes.append(latent_prefixes[idx])
            indices.append(idx)

        if not prompts:
            break

        z_active, _ = model.text_module(
            prompts,
            latent_prefixes=prefixes,
            max_length=config.text_max_new_tokens,
            temperature=config.text_temperature,
            top_p=config.text_top_p,
        )

        # Update latent cache without in-place operations
        latent_cache_new = latent_cache.clone()
        for local_idx, sample_idx in enumerate(indices):
            latent_cache_new[sample_idx] = z_active[local_idx].to(device)
        latent_cache = latent_cache_new

        reset_mask = torch.ones(batch_size, dtype=torch.bool, device=device) if attempt == 0 else torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(training):
            carry, logits = model.forward_with_latent(
                carry,
                problem_grids,
                latent_cache,
                reset_mask=reset_mask
            )

            masked_targets = targets.clone()
            inactive_mask = (~active_mask).unsqueeze(1)
            masked_targets = torch.where(inactive_mask, torch.zeros_like(masked_targets), masked_targets)
            loss = compute_loss_with_crop(logits, masked_targets)

        weight = 1.0 if attempt == 0 else config.correction_loss_weight
        weighted_losses.append(loss * weight)
        weight_sum += weight

        preds = logits.argmax(dim=-1)
        final_preds = preds
        exact_matches = compute_exact_match(preds, targets).to(device)

        attempt_counts = torch.where(active_mask, torch.full_like(attempt_counts, attempt + 1), attempt_counts)

        carry_latents = model.carry_to_latent(carry).cpu()  # Keep gradient flow to TRM

        for idx in range(batch_size):
            if not active_mask[idx]:
                continue
            if exact_matches[idx]:
                feedbacks[idx] = ""
                latent_prefixes[idx] = None
            else:
                feedbacks[idx] = grid_to_feedback(preds[idx], targets[idx])
                latent_prefixes[idx] = carry_latents[idx]

        active_mask = active_mask & (~exact_matches)

    if final_preds is None:
        raise RuntimeError("Self-correction loop did not run any attempts")

    attempt_counts = torch.where(
        attempt_counts == 0,
        torch.full_like(attempt_counts, config.max_correction_attempts),
        attempt_counts
    )

    total_loss = torch.stack(weighted_losses).sum() / weight_sum
    accuracy = compute_exact_match(final_preds, targets).float().mean().item()
    avg_attempts = attempt_counts.float().mean().item()

    metrics = {
        'loss': total_loss.item() if not training else None,
        'accuracy': accuracy,
        'attempts': avg_attempts
    }

    return total_loss, metrics, final_preds, carry


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr_ratio: float = 0.1
) -> float:
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return lr * float(step) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_step(
    model: HybridTRMModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    carry = None
) -> Tuple[Dict[str, float], any]:
    """Single training step that leverages the shared self-correction loop.

    Returns metrics and carry for TRM-style batch-to-batch persistence.
    """

    model.train()

    problem_grids = batch['problem_grid'].to(config.device)
    targets = batch['target_grid'].to(config.device)
    base_texts = batch['problem_text']

    total_loss, metrics, _, carry = run_self_correction(
        model,
        base_texts,
        problem_grids,
        targets,
        config,
        carry=carry,
        training=True
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    optimizer.zero_grad()

    metrics['loss'] = total_loss.item()
    return metrics, carry


@torch.no_grad()
def evaluate(
    model: HybridTRMModel,
    val_loader: DataLoader,
    config: TrainConfig
) -> Dict[str, float]:
    """Evaluate the model with the same self-correction behaviour used in training.

    Note: For evaluation, we reset carry for each problem (TRM-style evaluation).
    """

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_attempts = 0.0
    num_batches = 0

    for batch in tqdm.tqdm(val_loader, desc="Validation", leave=False):
        problem_grids = batch['problem_grid'].to(config.device)
        targets = batch['target_grid'].to(config.device)
        base_texts = batch['problem_text']

        with torch.no_grad():
            # Reset carry for each validation problem (TRM-style)
            loss, metrics, _, _ = run_self_correction(
                model,
                base_texts,
                problem_grids,
                targets,
                config,
                carry=None,  # Reset for each problem
                training=False
            )

        total_loss += loss.item()
        total_accuracy += metrics['accuracy']
        total_attempts += metrics['attempts']
        num_batches += 1

    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
        'val_accuracy': total_accuracy / max(num_batches, 1),
        'val_attempts': total_attempts / max(num_batches, 1)
    }

    return metrics


def train(config: TrainConfig):
    """
    Main training loop

    Args:
        config: Training configuration
    """
    # Set random seed
    torch.manual_seed(config.seed)

    # Create dataloaders
    print("Loading ARC dataset...")
    train_loader, val_loader = create_arc_dataloaders(
        num_train_problems=config.num_train_problems,
        num_val_problems=config.num_val_problems,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        augmentation=True
    )

    # Calculate total steps
    total_steps = len(train_loader) * config.num_epochs
    print(f"Total training steps: {total_steps}")

    # Create model
    print("Initializing model...")
    model_config = {
        'batch_size': config.batch_size,
        'seq_len': config.seq_len,
        'vocab_size': config.vocab_size,
        'H_cycles': config.H_cycles,
        'L_cycles': config.L_cycles,
        'H_layers': 0,  # ignored
        'L_layers': config.L_layers,
        'hidden_size': config.hidden_size,
        'text_hidden_size': config.text_hidden_size,
        'expansion': config.expansion,
        'num_heads': config.num_heads,
        'pos_encodings': config.pos_encodings,
        'text_model_name': config.text_model_name,
        'num_puzzle_identifiers': 1,
        'puzzle_emb_ndim': 0
    }

    model = HybridTRMModel(model_config).to(config.device)

    # Optimizer (only train TRM parameters, LLaMA is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Initialize wandb
    if config.run_name is None:
        config.run_name = f"trm_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
        dir="/data/trm/wandb"
    )

    # Training loop
    global_step = 0
    best_val_accuracy = 0.0

    # Initialize carry for TRM-style batch-to-batch persistence
    train_carry = None

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Training
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_attempts = 0.0

        progress_bar = tqdm.tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Update learning rate
            lr = cosine_schedule_with_warmup(
                global_step,
                total_steps,
                config.warmup_steps,
                config.lr
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Train step with carry persistence (TRM-style)
            metrics, train_carry = train_step(
                model, batch, optimizer, config, carry=train_carry
            )

            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
            epoch_attempts += metrics.get('attempts', 1)
            global_step += 1

            # Logging
            if batch_idx % config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': metrics['loss'],
                    'acc': metrics['accuracy'],
                    'lr': lr
                })

                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/accuracy': metrics['accuracy'],
                    'train/attempts': metrics.get('attempts', 1),
                    'train/lr': lr,
                    'step': global_step
                })

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        avg_attempts = epoch_attempts / len(train_loader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Attempts: {avg_attempts:.2f}")

        # Validation
        if (epoch + 1) % config.eval_interval == 0:
            print("Running validation...")
            val_metrics = evaluate(model, val_loader, config)
            print(
                f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                f"Accuracy: {val_metrics['val_accuracy']:.4f}, "
                f"Attempts: {val_metrics['val_attempts']:.2f}"
            )

            wandb.log({
                **val_metrics,
                'epoch': epoch + 1
            })

            # Save best model
            if val_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_accuracy']
                save_path = os.path.join(config.save_dir, config.run_name, "best_model.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                    'config': model_config
                }, save_path)
                print(f"✓ Saved best model to {save_path}")

    # Final save
    final_path = os.path.join(config.save_dir, config.run_name, "final_model.pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config
    }, final_path)
    print(f"✓ Saved final model to {final_path}")

    wandb.finish()
    print("\n✅ Training completed!")


def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train TRM Hybrid Model for ARC-AGI")

    # Model args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--H_cycles", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=1)
    parser.add_argument("--L_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=512)

    # Training args
    parser.add_argument("--num_train_problems", type=int, default=10)
    parser.add_argument("--num_val_problems", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Self-correction args
    parser.add_argument("--max_correction_attempts", type=int, default=3)
    parser.add_argument("--correction_loss_weight", type=float, default=0.5)

    # Text generation args
    parser.add_argument("--text_max_new_tokens", type=int, default=128)
    parser.add_argument("--text_temperature", type=float, default=0.6)
    parser.add_argument("--text_top_p", type=float, default=0.9)

    # Logging
    parser.add_argument("--project_name", type=str, default="arc-trm-hybrid")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="/data/trm/checkpoints")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create config
    config = TrainConfig(**vars(args))

    # Start training
    train(config)


if __name__ == "__main__":
    main()
