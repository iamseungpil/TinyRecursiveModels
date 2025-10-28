"""
Training epoch functions for TYPE1 and TYPE2 HelmARC data.

Separated into its own file to keep train_compositional.py clean.
"""

import torch
from typing import Dict, Any
from tqdm import tqdm


def train_epoch_type1(
    config,
    model_components: Dict[str, Any],
    dataloader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> Dict[str, float]:
    """
    Train one epoch on TYPE1 data (correct programs only).

    Args:
        config: CompositionalConfig instance
        model_components: Dict with llm, adapter, trm, executor
        dataloader: DataLoader for TYPE1 HelmARC dataset
        optimizer: Optimizer
        epoch: Current epoch number

    Returns:
        metrics: Aggregated epoch metrics
    """
    from contrastive_loss import compute_prediction_loss
    from gpt_oss_port.plan_parser import parse_multi_step_plan

    executor = model_components["executor"]
    adapter = model_components["adapter"]

    epoch_metrics = {
        'total_loss': 0.0,
        'total_accuracy': 0.0,
        'total_exact_match': 0.0,
        'num_batches': 0,
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"TYPE1 Epoch {epoch+1}")):
        # Move to device
        input_grids = batch['input_grids'].to(config.device)
        output_grids = batch['output_grids'].to(config.device)
        programs = batch['programs']
        explanations = batch['explanations']

        # Process each sample in batch
        batch_loss = 0.0
        batch_accuracy = 0.0
        batch_exact_match = 0.0

        for i in range(input_grids.shape[0]):
            # Extract single sample
            input_grid_2d = input_grids[i].unsqueeze(0)  # [1, H, W]
            output_grid_2d = output_grids[i].unsqueeze(0)  # [1, H, W]
            explanation = explanations[i]

            # ✅ FIX: Flatten grids for TRM
            input_grid = input_grid_2d.view(1, -1)  # [1, H*W]
            output_grid = output_grid_2d.view(1, -1)  # [1, H*W]

            # Parse steps from explanation
            steps = parse_multi_step_plan(explanation)

            if not steps:
                print(f"    ⚠ No steps parsed from explanation: {explanation[:100]}")
                continue

            # Prepare batch dict for executor
            executor_batch = {
                'inputs': input_grid,  # [1, 900]
                'labels': output_grid,  # [1, 900]
                'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=config.device),
            }

            # Compute prediction loss
            loss, metrics = compute_prediction_loss(
                executor=executor,
                steps=steps,
                input_grid=input_grid,  # [1, 900]
                target_grid=output_grid,  # [1, 900]
                batch=executor_batch,
                return_grad=True,
            )

            batch_loss += loss
            batch_accuracy += metrics.get('accuracy', 0.0)
            batch_exact_match += metrics.get('exact_match', 0.0)

        # Average over batch
        if input_grids.shape[0] > 0:
            batch_loss = batch_loss / input_grids.shape[0]
            batch_accuracy = batch_accuracy / input_grids.shape[0]
            batch_exact_match = batch_exact_match / input_grids.shape[0]

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_components["adapter"].parameters()] +
                [p for p in model_components["trm"].parameters()],
                max_norm=1.0
            )
            optimizer.step()

            # Update epoch metrics
            epoch_metrics['total_loss'] += batch_loss.item()
            epoch_metrics['total_accuracy'] += batch_accuracy
            epoch_metrics['total_exact_match'] += batch_exact_match
            epoch_metrics['num_batches'] += 1

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}: loss={batch_loss.item():.4f}, acc={batch_accuracy:.4f}")

    # Average over epoch
    if epoch_metrics['num_batches'] > 0:
        epoch_metrics['avg_loss'] = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
        epoch_metrics['avg_accuracy'] = epoch_metrics['total_accuracy'] / epoch_metrics['num_batches']
        epoch_metrics['avg_exact_match'] = epoch_metrics['total_exact_match'] / epoch_metrics['num_batches']

    print(f"\n  📊 TYPE1 Epoch {epoch+1} Summary:")
    print(f"    Loss: {epoch_metrics.get('avg_loss', 0):.4f}")
    print(f"    Accuracy: {epoch_metrics.get('avg_accuracy', 0):.4f}")
    print(f"    Exact Match: {epoch_metrics.get('avg_exact_match', 0):.4f}")

    return epoch_metrics


def train_epoch_type2(
    config,
    model_components: Dict[str, Any],
    dataloader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> Dict[str, float]:
    """
    Train one epoch on TYPE2 data (wrong vs correct programs).

    Args:
        config: CompositionalConfig instance
        model_components: Dict with llm, adapter, trm, executor
        dataloader: DataLoader for TYPE2 HelmARC dataset
        optimizer: Optimizer
        epoch: Current epoch number

    Returns:
        metrics: Aggregated epoch metrics
    """
    from contrastive_loss import compute_mixed_loss_type2
    from gpt_oss_port.plan_parser import parse_multi_step_plan

    executor = model_components["executor"]
    adapter = model_components["adapter"]

    epoch_metrics = {
        'total_loss': 0.0,
        'prediction_loss': 0.0,
        'contrastive_loss': 0.0,
        'total_accuracy': 0.0,
        'num_batches': 0,
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"TYPE2 Epoch {epoch+1}")):
        # Move to device
        input_grids = batch['input_grids'].to(config.device)
        output_grids = batch['output_grids'].to(config.device)
        correct_programs = batch['correct_programs']
        wrong_programs = batch['wrong_programs']
        explanations = batch['explanations']

        # Process each sample in batch
        batch_loss = 0.0
        batch_pred_loss = 0.0
        batch_contrast_loss = 0.0
        batch_accuracy = 0.0

        for i in range(input_grids.shape[0]):
            # Extract single sample
            input_grid_2d = input_grids[i].unsqueeze(0)  # [1, H, W]
            output_grid_2d = output_grids[i].unsqueeze(0)  # [1, H, W]
            explanation = explanations[i]

            # ✅ FIX: Flatten grids for TRM
            input_grid = input_grid_2d.view(1, -1)  # [1, H*W]
            output_grid = output_grid_2d.view(1, -1)  # [1, H*W]

            # Parse correct and wrong steps from explanation
            # TYPE2 explanation format: "**Incorrect transformation:** ... **Correct transformation:** ..."
            parts = explanation.split("**Correct transformation")
            if len(parts) < 2:
                print(f"    ⚠ Failed to split TYPE2 explanation")
                continue

            correct_part = parts[1]
            correct_steps = parse_multi_step_plan(correct_part)

            # For wrong steps, we can use the wrong_program directly
            # or parse from the incorrect part
            wrong_part = parts[0]
            wrong_steps = parse_multi_step_plan(wrong_part)

            # If wrong_steps is empty, create a single-step plan from wrong_program
            if not wrong_steps and wrong_programs[i]:
                wrong_steps = [wrong_programs[i]]

            if not correct_steps:
                print(f"    ⚠ No correct steps parsed")
                continue

            # Prepare batch dict for executor
            executor_batch = {
                'inputs': input_grid,  # [1, 900]
                'labels': output_grid,  # [1, 900]
                'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=config.device),
            }

            # Compute mixed loss (prediction + contrastive)
            loss, metrics = compute_mixed_loss_type2(
                executor=executor,
                adapter=adapter,
                correct_steps=correct_steps,
                wrong_steps=wrong_steps if wrong_steps else correct_steps,  # Fallback
                input_grid=input_grid,  # [1, 900]
                target_grid=output_grid,  # [1, 900]
                batch=executor_batch,
                prediction_weight=config.prediction_weight,
                contrastive_weight=config.contrastive_weight,
                margin=config.contrastive_margin,
                return_grad=True,
            )

            batch_loss += loss
            batch_pred_loss += metrics.get('prediction_loss', 0.0)
            batch_contrast_loss += metrics.get('contrastive_loss', 0.0)
            batch_accuracy += metrics.get('accuracy', 0.0)

        # Average over batch
        if input_grids.shape[0] > 0:
            batch_loss = batch_loss / input_grids.shape[0]
            batch_pred_loss = batch_pred_loss / input_grids.shape[0]
            batch_contrast_loss = batch_contrast_loss / input_grids.shape[0]
            batch_accuracy = batch_accuracy / input_grids.shape[0]

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_components["adapter"].parameters()] +
                [p for p in model_components["trm"].parameters()],
                max_norm=1.0
            )
            optimizer.step()

            # Update epoch metrics
            epoch_metrics['total_loss'] += batch_loss.item()
            epoch_metrics['prediction_loss'] += batch_pred_loss
            epoch_metrics['contrastive_loss'] += batch_contrast_loss
            epoch_metrics['total_accuracy'] += batch_accuracy
            epoch_metrics['num_batches'] += 1

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}: loss={batch_loss.item():.4f}, "
                      f"pred={batch_pred_loss:.4f}, contrast={batch_contrast_loss:.4f}")

    # Average over epoch
    if epoch_metrics['num_batches'] > 0:
        epoch_metrics['avg_loss'] = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
        epoch_metrics['avg_pred_loss'] = epoch_metrics['prediction_loss'] / epoch_metrics['num_batches']
        epoch_metrics['avg_contrast_loss'] = epoch_metrics['contrastive_loss'] / epoch_metrics['num_batches']
        epoch_metrics['avg_accuracy'] = epoch_metrics['total_accuracy'] / epoch_metrics['num_batches']

    print(f"\n  📊 TYPE2 Epoch {epoch+1} Summary:")
    print(f"    Total Loss: {epoch_metrics.get('avg_loss', 0):.4f}")
    print(f"    Prediction Loss: {epoch_metrics.get('avg_pred_loss', 0):.4f}")
    print(f"    Contrastive Loss: {epoch_metrics.get('avg_contrast_loss', 0):.4f}")
    print(f"    Accuracy: {epoch_metrics.get('avg_accuracy', 0):.4f}")

    return epoch_metrics
