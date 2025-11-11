"""
Test V3 adapter with retrieval-based initialization.

Key differences from V2:
- Uses TestTimeAdapterV3 with K-NN retrieval
- Initializes from similar puzzles (not global mean)
- Expected: Better initial loss, faster convergence, higher accuracy
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from test_time_adapter_v3 import TestTimeAdapterV3, TestTimeConfigV3
from dataset.build_arc_dataset import arc_grid_to_np
from evaluators.arc import _crop
from prepare_grid_helpers import prepare_grid, prepare_grid_label

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_model(checkpoint_path: str, device: str):
    """Load TRM model."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    config = dict(full_config['arch'])
    config['batch_size'] = 1
    config['seq_len'] = 900
    config['vocab_size'] = 12
    config['num_puzzle_identifiers'] = 876406

    print(f"Loaded config: LSTM={config.get('use_lstm_gating', False)}, L_cycles={config.get('L_cycles', 'N/A')}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, config


def predict(model, input_tensor: torch.Tensor, puzzle_id: int, device: str) -> np.ndarray:
    """Run inference."""
    inputs = input_tensor.unsqueeze(0).to(device)
    puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long, device=device)

    batch = {
        'inputs': inputs,
        'puzzle_identifiers': puzzle_ids
    }

    carry = model.initial_carry(batch)

    if hasattr(carry, 'inner_carry'):
        InnerCarry = type(carry.inner_carry)
        Carry = type(carry)

        inner_carry_dict = {
            'z_H': carry.inner_carry.z_H.to(device),
            'z_L': carry.inner_carry.z_L.to(device)
        }
        if hasattr(carry.inner_carry, 'c_H') and carry.inner_carry.c_H is not None:
            inner_carry_dict['c_H'] = carry.inner_carry.c_H.to(device)

        carry = Carry(
            inner_carry=InnerCarry(**inner_carry_dict),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data={k: v.to(device) for k, v in carry.current_data.items()}
        )

    # Run ACT loop until halt
    act_step = 0
    while act_step < 16:
        carry, outputs = model(carry, batch)
        act_step += 1
        if carry.halted.all():
            break

    logits = outputs['logits'].squeeze(0)
    pred = logits.argmax(dim=-1).cpu().numpy()
    pred_grid = _crop(pred)

    return pred_grid


def evaluate_puzzle_with_test_time_training(
    model,
    puzzle_name: str,
    puzzle_data: dict,
    adapter: TestTimeAdapterV3,
    new_puzzle_id: int,
    device: str
):
    """Evaluate with test-time training."""
    # Prepare training examples for training (tensors)
    train_examples_for_training = []

    for ex in puzzle_data['train']:
        input_grid = arc_grid_to_np(ex['input'])
        output_grid = arc_grid_to_np(ex['output'])

        # For training (tensors)
        train_examples_for_training.append({
            'input': prepare_grid(input_grid),
            'output': prepare_grid_label(output_grid)
        })

    # Run test-time adaptation
    puzzle_id, history = adapter.adapt(
        train_examples_for_training,
        puzzle_id=new_puzzle_id,
        device=device,
        exclude_puzzle_name=puzzle_name  # Exclude this puzzle from K-NN search
    )

    # Evaluate on test examples
    test_results = []
    for test_ex in puzzle_data['test']:
        input_grid = arc_grid_to_np(test_ex['input'])
        output_grid = arc_grid_to_np(test_ex['output'])

        input_tensor = prepare_grid(input_grid)
        pred_grid = predict(model, input_tensor, puzzle_id, device)

        is_correct = (
            pred_grid.shape == output_grid.shape and
            np.array_equal(pred_grid, output_grid)
        )

        test_results.append({
            'correct': is_correct,
            'pred_shape': list(pred_grid.shape),
            'true_shape': list(output_grid.shape)
        })

    num_correct = sum(r['correct'] for r in test_results)
    accuracy = num_correct / len(test_results) if test_results else 0.0

    return {
        'puzzle_name': puzzle_name,
        'num_train': len(train_examples_for_training),
        'num_test': len(test_results),
        'correct': num_correct,
        'accuracy': accuracy,
        'final_loss': history['loss'][-1],
        'adaptation_steps': len(history['loss']),
        'test_results': test_results,
        'history': history
    }


def main():
    checkpoint_path = '/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071'
    data_path = '/data/arc1concept-aug-1000'
    device = 'cuda'

    test_puzzles = ['50a16a69', '66e6c45b', '1a2e2828']

    config = TestTimeConfigV3(
        reserved_puzzle_id=876410,
        learning_rate=1e-3,
        max_steps=50,
        patience=5,
        halt_max_steps=16,
        k_neighbors=10  # Use 10 nearest neighbors
    )

    print("=" * 80)
    print("Test-Time Training V3: Retrieval-Based Initialization")
    print("=" * 80)
    print(f"\nKey innovation:")
    print(f"  1. Find K={config.k_neighbors} most similar puzzles from training set")
    print(f"  2. Initialize from their average (not global mean)")
    print(f"  3. Expected: Lower initial loss, faster convergence")
    print(f"\nConfiguration:")
    print(f"  Max training steps: {config.max_steps}")
    print(f"  Max ACT steps: {config.halt_max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  K neighbors: {config.k_neighbors}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}")
    model, model_config = load_model(checkpoint_path, device)

    # Load puzzle data
    with open(os.path.join(data_path, "test_puzzles.json"), 'r') as f:
        all_puzzles = json.load(f)

    # Create adapter with data path for retrieval
    adapter = TestTimeAdapterV3(model, config, data_path=data_path)

    # Test each puzzle
    results = []
    for i, puzzle_name in enumerate(test_puzzles):
        print(f"\n{'='*80}")
        print(f"Puzzle {i+1}/{len(test_puzzles)}: {puzzle_name}")
        print(f"{'='*80}")

        if puzzle_name not in all_puzzles:
            print(f"ERROR: Puzzle {puzzle_name} not found")
            continue

        puzzle_data = all_puzzles[puzzle_name]
        new_puzzle_id = config.reserved_puzzle_id + i

        print(f"  Train examples: {len(puzzle_data['train'])}")
        print(f"  Test examples: {len(puzzle_data['test'])}")
        print(f"  Using NEW puzzle_id: {new_puzzle_id}")

        start_time = datetime.now()
        result = evaluate_puzzle_with_test_time_training(
            model, puzzle_name, puzzle_data, adapter, new_puzzle_id, device
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\nResults:")
        print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['num_test']})")
        print(f"  Adaptation steps: {result['adaptation_steps']}")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        result['elapsed_seconds'] = elapsed
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_correct = sum(r['correct'] for r in results)
    total_test = sum(r['num_test'] for r in results)
    avg_accuracy = total_correct / total_test if total_test > 0 else 0
    total_time = sum(r['elapsed_seconds'] for r in results)

    print(f"\nTest-Time Training V3 (Retrieval-Based Init):")
    print(f"  Overall accuracy: {avg_accuracy:.1%} ({total_correct}/{total_test})")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average per puzzle: {total_time/len(results):.1f}s")

    print(f"\nPer-puzzle results:")
    for r in results:
        print(f"  {r['puzzle_name']}: {r['accuracy']:.1%} ({r['correct']}/{r['num_test']}) - {r['adaptation_steps']} steps, loss {r['final_loss']:.4f}")

    # Save results
    output_path = Path("/data/test_realistic_v3.json")
    with open(output_path, 'w') as f:
        json.dump({
            'version': 'v3_retrieval_init',
            'config': {
                'max_steps': config.max_steps,
                'halt_max_steps': config.halt_max_steps,
                'learning_rate': config.learning_rate,
                'patience': config.patience,
                'k_neighbors': config.k_neighbors
            },
            'results': results,
            'summary': {
                'total_correct': total_correct,
                'total_test': total_test,
                'accuracy': float(avg_accuracy),
                'total_time_seconds': total_time
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Compare with V2
    print(f"\n{'='*80}")
    print("COMPARISON WITH V2")
    print(f"{'='*80}")
    print(f"\nV2 (global mean init):")
    print(f"  - Accuracy: 0.0% (0/3)")
    print(f"  - Initial loss: ~200 (avg)")
    print(f"  - Final loss: ~21.28 (avg)")
    print(f"  - Time: 78.2s")
    print(f"\nV3 (retrieval-based init):")
    print(f"  - Accuracy: {avg_accuracy:.1%} ({total_correct}/{total_test})")

    # Calculate initial loss (first loss value)
    if results:
        initial_losses = [r['history']['loss'][0] for r in results if r['history']['loss']]
        avg_initial_loss = np.mean(initial_losses) if initial_losses else 0
        print(f"  - Initial loss: ~{avg_initial_loss:.2f} (avg)")

    avg_final_loss = np.mean([r['final_loss'] for r in results]) if results else 0
    print(f"  - Final loss: ~{avg_final_loss:.2f} (avg)")
    print(f"  - Time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
