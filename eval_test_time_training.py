"""
Comprehensive evaluation of test-time training on validation set.

Compares performance with and without test-time adaptation.
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import argparse

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from test_time_adapter import TestTimeAdapter, TestTimeConfig
from dataset.build_arc_dataset import inverse_aug, arc_grid_to_np
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_model(checkpoint_path: str, device: str):
    """Load TRM model."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    config = {
        "batch_size": 1,
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 876406,
        "puzzle_emb_ndim": 512,
        "puzzle_emb_len": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
        "L_cycles": 4,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
        "no_ACT_continue": True,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, config


def prepare_grid(grid: np.ndarray) -> torch.Tensor:
    """Convert grid to model input format."""
    # Pad to 30x30 and flatten
    input_padded = np.pad(
        grid + 2,
        ((0, 30 - grid.shape[0]), (0, 30 - grid.shape[1])),
        constant_values=0
    )
    return torch.from_numpy(input_padded.reshape(-1)).long()


def predict(model, input_tensor: torch.Tensor, puzzle_id: int, device: str) -> np.ndarray:
    """Run model prediction."""
    with torch.no_grad():
        inputs = input_tensor.unsqueeze(0).to(device)
        puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long, device=device)

        batch = {
            'inputs': inputs,
            'puzzle_identifiers': puzzle_ids
        }

        # Initialize carry
        carry = model.initial_carry(batch)

        # Move carry to device
        if hasattr(carry, 'inner_carry'):
            InnerCarry = type(carry.inner_carry)
            Carry = type(carry)
            carry = Carry(
                inner_carry=InnerCarry(
                    z_H=carry.inner_carry.z_H.to(device),
                    z_L=carry.inner_carry.z_L.to(device)
                ),
                steps=carry.steps.to(device),
                halted=carry.halted.to(device),
                current_data={k: v.to(device) for k, v in carry.current_data.items()}
            )

        # Run all 16 ACT steps
        for _ in range(16):
            carry, outputs = model(carry, batch)

        # Get prediction
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)

        # Crop and convert to grid
        pred_grid = _crop(preds.cpu().numpy()[0])

    return pred_grid


def evaluate_puzzle(
    model,
    puzzle_name: str,
    puzzle_data: Dict,
    adapter: TestTimeAdapter,
    device: str,
    use_adaptation: bool = True,
    puzzle_id: int = None
) -> Dict:
    """Evaluate a single puzzle."""
    # Use separate IDs for with/without adaptation to avoid interference
    if puzzle_id is None:
        puzzle_id = 0 if use_adaptation else 1

    # Prepare training examples
    train_examples = []
    for ex in puzzle_data['train']:
        input_grid = arc_grid_to_np(ex['input'])
        output_grid = arc_grid_to_np(ex['output'])

        train_examples.append({
            'input': prepare_grid(input_grid),
            'output': prepare_grid(output_grid),
        })

    # Adapt if enabled
    if use_adaptation:
        returned_puzzle_id, history = adapter.adapt(train_examples, puzzle_id=puzzle_id, device=device)
        adaptation_loss = history['loss'][-1]
        adaptation_steps = len(history['loss'])
    else:
        # Initialize random embedding but don't adapt
        adapter.initialize_puzzle_embedding(puzzle_id)

        # Compute initial loss without training
        with torch.no_grad():
            total_loss = 0.0
            for example in train_examples:
                inputs = example['input'].unsqueeze(0).to(device)
                labels = example['output'].unsqueeze(0).to(device)
                puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long, device=device)

                batch = {
                    'inputs': inputs,
                    'labels': labels,
                    'puzzle_identifiers': puzzle_ids
                }

                carry = model.initial_carry(batch)
                if hasattr(carry, 'inner_carry'):
                    InnerCarry = type(carry.inner_carry)
                    Carry = type(carry)
                    carry = Carry(
                        inner_carry=InnerCarry(
                            z_H=carry.inner_carry.z_H.to(device),
                            z_L=carry.inner_carry.z_L.to(device)
                        ),
                        steps=carry.steps.to(device),
                        halted=carry.halted.to(device),
                        current_data={k: v.to(device) for k, v in carry.current_data.items()}
                    )

                carry, outputs = model(carry, batch)
                logits = outputs['logits']
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                total_loss += loss.item()

        adaptation_loss = total_loss / len(train_examples)
        adaptation_steps = 0

    # Evaluate on test examples
    results = []
    for test_ex in puzzle_data['test']:
        input_grid = arc_grid_to_np(test_ex['input'])
        output_grid = arc_grid_to_np(test_ex['output'])

        input_tensor = prepare_grid(input_grid)
        pred_grid = predict(model, input_tensor, puzzle_id, device)

        # Check if correct
        is_correct = (pred_grid.shape == output_grid.shape and
                     np.array_equal(pred_grid, output_grid))

        results.append({
            'correct': is_correct,
            'pred_shape': list(pred_grid.shape),
            'true_shape': list(output_grid.shape),
            'pred_sample': pred_grid[:min(3, pred_grid.shape[0]), :min(3, pred_grid.shape[1])].tolist(),
            'true_sample': output_grid[:min(3, output_grid.shape[0]), :min(3, output_grid.shape[1])].tolist()
        })

    return {
        'puzzle_name': puzzle_name,
        'num_train': len(train_examples),
        'num_test': len(results),
        'adaptation_loss': adaptation_loss,
        'adaptation_steps': adaptation_steps,
        'test_results': results,
        'accuracy': sum(r['correct'] for r in results) / len(results)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071')
    parser.add_argument('--data-path', type=str,
                       default='/data/arc1concept-aug-1000')
    parser.add_argument('--num-puzzles', type=int, default=30,
                       help='Number of puzzles to evaluate')
    parser.add_argument('--output', type=str, default='eval_results.json')
    args = parser.parse_args()

    device = "cuda"

    print("=" * 80)
    print("Test-Time Training Evaluation")
    print("=" * 80)

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Create adapter
    test_time_config = TestTimeConfig(
        reserved_puzzle_id=0,
        learning_rate=1e-3,
        max_steps=50,
        patience=5
    )
    adapter = TestTimeAdapter(model, test_time_config)

    # Load test puzzles
    with open(os.path.join(args.data_path, "test_puzzles.json"), 'r') as f:
        test_puzzles = json.load(f)

    # Select subset of puzzles
    puzzle_names = list(test_puzzles.keys())[:args.num_puzzles]

    print(f"\nEvaluating {len(puzzle_names)} puzzles...")
    print(f"Comparing: WITH adaptation vs WITHOUT adaptation\n")

    results_with_adaptation = []
    results_without_adaptation = []

    for i, puzzle_name in enumerate(puzzle_names, 1):
        print(f"[{i}/{len(puzzle_names)}] {puzzle_name}")

        puzzle_data = test_puzzles[puzzle_name]

        # With adaptation
        result_with = evaluate_puzzle(
            model, puzzle_name, puzzle_data, adapter, device, use_adaptation=True
        )
        results_with_adaptation.append(result_with)

        # Without adaptation
        result_without = evaluate_puzzle(
            model, puzzle_name, puzzle_data, adapter, device, use_adaptation=False
        )
        results_without_adaptation.append(result_without)

        print(f"  With adaptation:    {result_with['accuracy']:.1%} "
              f"(loss: {result_with['adaptation_loss']:.2f}, steps: {result_with['adaptation_steps']})")
        print(f"  Without adaptation: {result_without['accuracy']:.1%}")
        print()

    # Compute overall statistics
    avg_acc_with = np.mean([r['accuracy'] for r in results_with_adaptation])
    avg_acc_without = np.mean([r['accuracy'] for r in results_without_adaptation])

    print("=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"With adaptation:    {avg_acc_with:.1%}")
    print(f"Without adaptation: {avg_acc_without:.1%}")
    print(f"Improvement:        {avg_acc_with - avg_acc_without:+.1%}")
    print("=" * 80)

    # Save results
    output_data = {
        'config': {
            'checkpoint': args.checkpoint,
            'num_puzzles': len(puzzle_names),
            'test_time_config': {
                'learning_rate': test_time_config.learning_rate,
                'max_steps': test_time_config.max_steps,
                'patience': test_time_config.patience
            }
        },
        'summary': {
            'avg_accuracy_with_adaptation': float(avg_acc_with),
            'avg_accuracy_without_adaptation': float(avg_acc_without),
            'improvement': float(avg_acc_with - avg_acc_without)
        },
        'results_with_adaptation': results_with_adaptation,
        'results_without_adaptation': results_without_adaptation
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
