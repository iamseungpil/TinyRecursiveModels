"""
Contrastive Loss for TYPE 2 Training (Wrong vs Correct Programs)

Implements margin-based contrastive learning:
- Wrong program should produce HIGHER loss than correct program
- Margin ensures sufficient separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


def compute_contrastive_loss(
    executor,
    adapter,
    correct_steps: list,
    wrong_steps: list,
    input_grid: torch.Tensor,
    target_grid: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    margin: float = 1.0,
    return_grad: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute contrastive loss between wrong and correct programs.

    Goal: loss(wrong_program) > loss(correct_program) + margin

    Args:
        executor: SequentialExecutor instance
        adapter: TextToLatentAdapter (not used directly, but passed for consistency)
        correct_steps: List of correct instruction strings
        wrong_steps: List of wrong instruction strings
        input_grid: [batch, seq] input grid tensor
        target_grid: [batch, seq] target grid tensor
        batch: Full batch dict
        margin: Contrastive margin (default 1.0)
        return_grad: Whether to compute gradients

    Returns:
        loss: Contrastive loss tensor
        metrics: Dict with detailed metrics
    """
    device = input_grid.device

    # 1. Execute correct program
    try:
        final_grid_correct, _, metrics_correct = executor.execute_plan(
            steps=correct_steps,
            input_grid=input_grid,
            batch=batch,
            target_grid=target_grid.cpu().numpy()[0],
            return_grad=return_grad,
        )
        loss_correct = metrics_correct['total_loss']

        # Handle case where total_loss might be 0 (perfect match)
        if not isinstance(loss_correct, torch.Tensor):
            loss_correct = torch.tensor(loss_correct, device=device)

    except Exception as e:
        print(f"    ⚠ Correct program execution failed: {e}")
        # Return high loss for failed correct program
        return torch.tensor(10.0, device=device), {
            'correct_loss': 10.0,
            'wrong_loss': 0.0,
            'margin_violation': 10.0,
        }

    # 2. Execute wrong program
    try:
        final_grid_wrong, _, metrics_wrong = executor.execute_plan(
            steps=wrong_steps,
            input_grid=input_grid,
            batch=batch,
            target_grid=target_grid.cpu().numpy()[0],
            return_grad=return_grad,
        )
        loss_wrong = metrics_wrong['total_loss']

        if not isinstance(loss_wrong, torch.Tensor):
            loss_wrong = torch.tensor(loss_wrong, device=device)

    except Exception as e:
        print(f"    ⚠ Wrong program execution failed: {e}")
        # Wrong program failing is actually good (higher implicit loss)
        loss_wrong = torch.tensor(10.0, device=device)
        metrics_wrong = {'total_loss': 10.0}

    # 3. Compute contrastive loss
    # Goal: loss_wrong > loss_correct + margin
    # If loss_wrong < loss_correct + margin, penalize
    margin_violation = torch.clamp(loss_correct - loss_wrong + margin, min=0.0)

    # Total contrastive loss
    contrastive_loss = margin_violation

    # 4. Collect metrics
    metrics = {
        'correct_loss': loss_correct.item() if isinstance(loss_correct, torch.Tensor) else loss_correct,
        'wrong_loss': loss_wrong.item() if isinstance(loss_wrong, torch.Tensor) else loss_wrong,
        'margin_violation': margin_violation.item(),
        'correct_accuracy': metrics_correct.get('cell_accuracy', 0.0),
        'wrong_accuracy': metrics_wrong.get('cell_accuracy', 0.0),
    }

    return contrastive_loss, metrics


def compute_prediction_loss(
    executor,
    steps: list,
    input_grid: torch.Tensor,
    target_grid: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    return_grad: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute standard prediction loss for TYPE 1 training.

    Args:
        executor: SequentialExecutor instance
        steps: List of instruction strings
        input_grid: [batch, seq] input grid tensor
        target_grid: [batch, seq] target grid tensor
        batch: Full batch dict
        return_grad: Whether to compute gradients

    Returns:
        loss: Prediction loss tensor
        metrics: Dict with accuracy metrics
    """
    device = input_grid.device

    try:
        final_grid, intermediate_results, metrics = executor.execute_plan(
            steps=steps,
            input_grid=input_grid,
            batch=batch,
            target_grid=target_grid.cpu().numpy()[0],
            return_grad=return_grad,
        )

        loss = metrics['total_loss']
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=device)

        result_metrics = {
            'prediction_loss': loss.item(),
            'accuracy': metrics.get('cell_accuracy', 0.0),
            'exact_match': metrics.get('exact_match', 0.0),
            'num_steps': len(steps),
            'total_trm_steps': metrics.get('total_trm_steps', 0),
        }

        return loss, result_metrics

    except Exception as e:
        print(f"    ❌ Prediction execution failed: {e}")
        # Return high loss for failed execution
        return torch.tensor(10.0, device=device), {
            'prediction_loss': 10.0,
            'accuracy': 0.0,
            'exact_match': 0.0,
            'error': str(e),
        }


def compute_mixed_loss_type2(
    executor,
    adapter,
    correct_steps: list,
    wrong_steps: list,
    input_grid: torch.Tensor,
    target_grid: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    prediction_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    margin: float = 1.0,
    return_grad: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute mixed loss for TYPE 2: prediction + contrastive.

    Args:
        executor: SequentialExecutor instance
        adapter: TextToLatentAdapter
        correct_steps: Correct instruction list
        wrong_steps: Wrong instruction list
        input_grid: Input grid tensor
        target_grid: Target grid tensor
        batch: Full batch dict
        prediction_weight: Weight for prediction loss
        contrastive_weight: Weight for contrastive loss
        margin: Contrastive margin
        return_grad: Compute gradients

    Returns:
        total_loss: Combined loss
        metrics: Detailed metrics dict
    """
    # 1. Prediction loss (on correct program)
    loss_pred, metrics_pred = compute_prediction_loss(
        executor=executor,
        steps=correct_steps,
        input_grid=input_grid,
        target_grid=target_grid,
        batch=batch,
        return_grad=return_grad,
    )

    # 2. Contrastive loss (wrong vs correct)
    loss_contrast, metrics_contrast = compute_contrastive_loss(
        executor=executor,
        adapter=adapter,
        correct_steps=correct_steps,
        wrong_steps=wrong_steps,
        input_grid=input_grid,
        target_grid=target_grid,
        batch=batch,
        margin=margin,
        return_grad=return_grad,
    )

    # 3. Combine losses
    total_loss = (
        prediction_weight * loss_pred +
        contrastive_weight * loss_contrast
    )

    # 4. Merge metrics
    metrics = {
        'total_loss': total_loss.item(),
        'prediction_loss': metrics_pred['prediction_loss'],
        'contrastive_loss': loss_contrast.item(),
        'prediction_weight': prediction_weight,
        'contrastive_weight': contrastive_weight,
        **metrics_pred,
        **metrics_contrast,
    }

    return total_loss, metrics


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Contrastive Loss Module Test")
    print("=" * 60)

    # This requires full model setup, so just verify imports
    print("\n✓ All functions imported successfully")
    print("✓ compute_contrastive_loss")
    print("✓ compute_prediction_loss")
    print("✓ compute_mixed_loss_type2")

    print("\nNote: Full testing requires model components.")
    print("Run from train_compositional.py for integration test.")
