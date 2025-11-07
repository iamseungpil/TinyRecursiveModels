"""
TRM Step-by-Step Analysis WITH ACT (Adaptive Computation Time)

This script properly uses ACT to let the model decide when to halt,
while capturing intermediate states at each step.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_model_and_data(checkpoint_path, data_path, puzzle_idx, device='cuda'):
    """Load TRM model and test puzzle."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Clean state dict
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    # Config
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
        "H_cycles": 3,  # Training config
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,  # ACT max steps
        "halt_exploration_prob": 0.0,  # No exploration during inference
        "pos_encodings": "rope",
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded with ACT (halt_max_steps={config['halt_max_steps']})")

    # Load puzzle
    test_dir = os.path.join(data_path, 'test')
    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    start_idx = puzzle_indices[puzzle_idx]
    end_idx = puzzle_indices[puzzle_idx + 1]

    batch = {
        'inputs': torch.from_numpy(inputs[start_idx:end_idx].astype(np.int64)).to(device),
        'labels': torch.from_numpy(labels[start_idx:end_idx].astype(np.int64)).to(device),
        'puzzle_identifiers': torch.full(
            (end_idx - start_idx,),
            puzzle_identifiers[puzzle_idx],
            dtype=torch.long,
            device=device
        )
    }

    print(f"✅ Loaded puzzle {puzzle_idx}: {end_idx - start_idx} examples")

    return model, batch, config


def run_act_with_history(model, batch, device):
    """
    Run ACT inference while capturing intermediate states.

    This modifies the ACT loop to record z_H, z_L at each step.

    NOTE: During eval mode, the model ignores Q-head halting and always runs
    for halt_max_steps. The initial carry starts with halted=True to trigger
    reset on first forward pass.
    """
    history = []

    # Initialize carry (starts with halted=True by design)
    carry = model.initial_carry(batch)

    with torch.no_grad():
        # Run for halt_max_steps (model ignores Q-head during eval)
        for step in range(model.config.halt_max_steps):
            # Record state BEFORE this step
            z_H_norm = carry.inner_carry.z_H.norm().item()
            z_L_norm = carry.inner_carry.z_L.norm().item()
            print(f"  Step {step}: halted={carry.halted[0].item()}, steps={carry.steps[0].item()}, "
                  f"||z_H||={z_H_norm:.3f}, ||z_L||={z_L_norm:.3f}")

            history.append({
                'step': step,
                'z_H': carry.inner_carry.z_H.float().cpu().clone(),
                'z_L': carry.inner_carry.z_L.float().cpu().clone(),
                'halted': carry.halted.cpu().clone(),
                'steps': carry.steps.cpu().clone(),
            })

            # Debug: Check state values
            if step <= 2:
                print(f"    z_H[0,0,:5] before: {carry.inner_carry.z_H[0,0,:5]}")

            # Run one ACT step
            carry, outputs = model(carry, batch)

            # Debug: Check if z_H actually changed
            new_z_H_norm = carry.inner_carry.z_H.norm().item()
            new_z_L_norm = carry.inner_carry.z_L.norm().item()
            if step <= 2:
                print(f"    z_H[0,0,:5] after: {carry.inner_carry.z_H[0,0,:5]}")
            print(f"    After forward: ||z_H||={new_z_H_norm:.3f}, ||z_L||={new_z_L_norm:.3f}, "
                  f"halted={carry.halted[0].item()}, steps={carry.steps[0].item()}")

            # Record output
            history[-1]['logits'] = outputs['logits'].cpu().clone()
            history[-1]['q_halt'] = outputs['q_halt_logits'].cpu().clone()

        # Final state
        history.append({
            'step': model.config.halt_max_steps,
            'z_H': carry.inner_carry.z_H.float().cpu().clone(),
            'z_L': carry.inner_carry.z_L.float().cpu().clone(),
            'halted': carry.halted.cpu().clone(),
            'steps': carry.steps.cpu().clone(),
            'logits': outputs['logits'].cpu().clone(),
            'q_halt': None,  # No Q-head after final step
        })

    return history, carry


def analyze_act_results(history, batch, output_dir):
    """Analyze ACT results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Calculate accuracies
    labels_np = batch['labels'][0].cpu().numpy()  # [900]

    accuracies = []
    q_halt_values = []
    for h in history[:-1]:  # Exclude final state (has no logits from forward pass)
        if 'logits' in h:
            pred = h['logits'][0].argmax(dim=-1).numpy()  # [900]
            acc = (pred == labels_np).mean()
            accuracies.append(acc)

            # Q-halt values (higher = wants to halt)
            if h['q_halt'] is not None:
                q_halt_values.append(float(h['q_halt'][0].item()))

    # NOTE: During eval mode, model runs for halt_max_steps regardless of Q-values
    # During training, it would halt when q_halt > 0 (if no_ACT_continue=True)
    num_steps = len(accuracies)

    # Check if Q-head would have wanted to halt early
    hypothetical_halt = None
    for i, q_val in enumerate(q_halt_values):
        if q_val > 0:
            hypothetical_halt = i
            break

    print(f"\n{'='*80}")
    print(f"ACT Analysis Results")
    print(f"{'='*80}")
    print(f"Total ACT steps: {num_steps}")
    if hypothetical_halt is not None:
        print(f"Q-head wanted to halt at: step {hypothetical_halt} (but ignored in eval mode)")
    else:
        print(f"Q-head never signaled halt (all q_halt <= 0)")
    print(f"\nAccuracies and Q-values by step:")
    for i, acc in enumerate(accuracies):
        q_str = f", q_halt={q_halt_values[i]:+.3f}" if i < len(q_halt_values) else ""
        marker = " ← Q-HALT SIGNAL" if i == hypothetical_halt else ""
        print(f"  Step {i}: {acc:.1%}{q_str}{marker}")
    print(f"\nFinal accuracy: {accuracies[-1]:.1%}")
    print(f"Solved: {accuracies[-1] == 1.0}")

    # Save metrics
    metrics = {
        'num_steps': num_steps,
        'hypothetical_halt_step': hypothetical_halt,
        'q_halt_values': q_halt_values,
        'accuracies': [float(a) for a in accuracies],
        'final_accuracy': float(accuracies[-1]),
        'is_solved': bool(accuracies[-1] == 1.0),
    }

    with open(os.path.join(output_dir, 'act_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot accuracy progression
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Accuracy
    steps = list(range(len(accuracies)))
    ax1.plot(steps, [a * 100 for a in accuracies], 'o-', linewidth=2, markersize=8)
    if hypothetical_halt is not None:
        ax1.axvline(x=hypothetical_halt, color='r', linestyle='--',
                    label=f'Q-head wanted to halt at step {hypothetical_halt}')
    ax1.set_xlabel('ACT Step', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('TRM with ACT: Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if hypothetical_halt is not None:
        ax1.legend()

    # Q-halt values
    ax2.plot(range(len(q_halt_values)), q_halt_values, 's-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.fill_between(range(len(q_halt_values)), 0, q_halt_values,
                      where=[q > 0 for q in q_halt_values], alpha=0.3, color='red', label='Wants to halt')
    ax2.set_xlabel('ACT Step', fontsize=12)
    ax2.set_ylabel('Q-halt Value', fontsize=12)
    ax2.set_title('Q-head Halting Signal (>0 = wants to halt)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'act_accuracy_progression.png'), dpi=150)
    plt.close()

    print(f"\n✅ Results saved to {output_dir}")
    print(f"{'='*80}\n")

    return metrics


def main():
    checkpoint_path = '/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071'
    data_path = '/data/arc1concept-aug-1000'

    # Test puzzles (start with just 0 and 10 for debugging)
    test_puzzles = [0, 10]

    for puzzle_idx in test_puzzles:
        print(f"\n\n{'#'*80}")
        print(f"# Analyzing Puzzle {puzzle_idx} with ACT")
        print(f"{'#'*80}\n")

        output_dir = f'/home/ubuntu/TinyRecursiveModels/latent_analysis/results/act_analysis/puzzle_{puzzle_idx}'

        # Load
        model, batch, config = load_model_and_data(checkpoint_path, data_path, puzzle_idx)

        # Run ACT with history
        history, final_carry = run_act_with_history(model, batch, 'cuda')

        # Analyze
        metrics = analyze_act_results(history, batch, output_dir)


if __name__ == "__main__":
    main()
