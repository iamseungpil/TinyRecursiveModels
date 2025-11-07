"""
Comprehensive ACT Analysis for All 400 Validation Puzzles

This script analyzes TRM's ACT (Adaptive Computation Time) behavior across
the entire validation set, tracking z_H changes, prediction evolution, and
ACT halting patterns at each step.

KEY FIX: Uses H_cycles=1 to prevent the 48 H-cycle problem.
- With H_cycles=1: 16 ACT steps = 16 H-cycles (close to training: ~3)
- With H_cycles=3: 16 ACT steps = 48 H-cycles (way beyond training!)

Usage:
    python act_analysis_400_validation.py --output_dir results/act_400/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

# GPU 4 for analysis
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# ============================================================================
# 1. MODEL & DATA LOADING
# ============================================================================

def load_checkpoint(checkpoint_path: str, device: str):
    """Load TRM model with FIXED configuration (H_cycles=1)."""
    print(f"📦 Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    # CRITICAL FIX: H_cycles=1 (not 3!)
    # This ensures 16 ACT steps = 16 H-cycles (not 48)
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
        "H_cycles": 1,  # ⚠️ CRITICAL: 1 ACT step = 1 H-cycle
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded with ACT:")
    print(f"   H_cycles = {config['H_cycles']} (1 ACT step = 1 H-cycle)")
    print(f"   halt_max_steps = {config['halt_max_steps']}")
    print(f"   Total H-cycles possible = {config['H_cycles'] * config['halt_max_steps']}")

    return model, config


def load_all_validation_puzzles(data_path: str, identifiers_path: str):
    """Load all 400 validation puzzles."""
    print(f"📂 Loading validation puzzles from {data_path}...")

    test_dir = os.path.join(data_path, 'test')

    # Load memory-mapped arrays
    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    # Load identifier map
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}

    num_puzzles = len(puzzle_indices) - 1
    print(f"✅ Found {num_puzzles} validation puzzles")

    return {
        'inputs': inputs,
        'labels': labels,
        'puzzle_identifiers': puzzle_identifiers,
        'puzzle_indices': puzzle_indices,
        'identifier_map': identifier_map,
        'num_puzzles': num_puzzles,
    }


def get_puzzle_batch(data_dict: Dict, puzzle_idx: int, device: str):
    """Extract a single puzzle as a batch."""
    start_idx = data_dict['puzzle_indices'][puzzle_idx]
    end_idx = data_dict['puzzle_indices'][puzzle_idx + 1]

    batch = {
        'inputs': torch.from_numpy(
            data_dict['inputs'][start_idx:end_idx].astype(np.int64)
        ).to(device),
        'labels': torch.from_numpy(
            data_dict['labels'][start_idx:end_idx].astype(np.int64)
        ).to(device),
        'puzzle_identifiers': torch.full(
            (end_idx - start_idx,),
            data_dict['puzzle_identifiers'][puzzle_idx],
            dtype=torch.long,
            device=device
        ),
        'puzzle_id': int(data_dict['puzzle_identifiers'][puzzle_idx]),
        'puzzle_name': data_dict['identifier_map'].get(
            data_dict['puzzle_identifiers'][puzzle_idx],
            f"<unknown>"
        ),
        'num_examples': end_idx - start_idx,
    }

    return batch


# ============================================================================
# 2. ACT INFERENCE WITH TRACKING
# ============================================================================

def run_act_with_tracking(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    device: str,
    max_steps: int = 16
) -> List[Dict]:
    """
    Run ACT inference while tracking all intermediate states.

    Returns:
        history: List of dicts (one per ACT step) containing:
            - act_step: int
            - z_H: [batch, seq_len, hidden_size]
            - z_L: [batch, seq_len, hidden_size]
            - z_H_mean: [batch, hidden_size]
            - halted: [batch]
            - steps: [batch]
            - logits: [batch, seq_len, vocab_size]
            - prediction: [seq_len] (argmax tokens)
            - q_halt_logit: float
    """
    history = []

    # Initialize carry
    carry = model.initial_carry(batch)

    with torch.no_grad():
        for act_step in range(max_steps):
            # Record state BEFORE this ACT step
            history.append({
                'act_step': act_step,
                'z_H': carry.inner_carry.z_H.float().cpu().clone(),
                'z_L': carry.inner_carry.z_L.float().cpu().clone(),
                'z_H_mean': carry.inner_carry.z_H.mean(dim=1).float().cpu().clone(),
                'z_L_mean': carry.inner_carry.z_L.mean(dim=1).float().cpu().clone(),
                'halted': carry.halted.cpu().clone(),
                'steps': carry.steps.cpu().clone(),
            })

            # Run one ACT step (forward pass)
            carry, outputs = model(carry, batch)

            # Record outputs
            history[-1]['logits'] = outputs['logits'].cpu().clone()
            history[-1]['prediction'] = outputs['logits'].argmax(dim=-1)[0].cpu().numpy()
            history[-1]['q_halt_logit'] = float(outputs['q_halt_logits'][0].item())

    return history


# ============================================================================
# 3. GRID UTILITIES
# ============================================================================

def tokens_to_grid(tokens: torch.Tensor, puzzle_name: str) -> np.ndarray:
    """Convert token sequence to grid with inverse augmentation."""
    orig_name, inverse_fn = inverse_aug(puzzle_name)

    if isinstance(tokens, torch.Tensor):
        tokens_np = tokens.cpu().numpy()
    else:
        tokens_np = tokens

    grid = inverse_fn(_crop(tokens_np))
    return grid


# ============================================================================
# 4. ANALYSIS FUNCTIONS
# ============================================================================

def analyze_puzzle_act(
    history: List[Dict],
    batch: Dict,
    puzzle_idx: int
) -> Dict:
    """
    Analyze ACT behavior for a single puzzle.

    Returns:
        metrics: Dict containing:
            - puzzle_idx, puzzle_name, num_examples
            - num_act_steps: int
            - accuracies: List[float] (per ACT step)
            - q_halt_values: List[float]
            - z_H_movements: List[float] (L2 norm of z_H changes)
            - prediction_changes: List[int] (number of tokens changed)
            - is_solved: bool
            - final_accuracy: float
            - convergence_step: int (first step where prediction stops changing)
    """
    test_idx = batch['num_examples'] - 1
    ground_truth_tokens = batch['labels'][test_idx].cpu().numpy()
    ground_truth_grid = tokens_to_grid(ground_truth_tokens, batch['puzzle_name'])

    metrics = {
        'puzzle_idx': puzzle_idx,
        'puzzle_name': batch['puzzle_name'],
        'num_examples': batch['num_examples'],
        'num_act_steps': len(history),
        'accuracies': [],
        'q_halt_values': [],
        'z_H_movements': [],
        'prediction_changes': [],
        'grid_accuracy': [],
    }

    prev_pred = None
    prev_z_H = None

    for h in history:
        # Prediction accuracy (token-level)
        pred_tokens = torch.from_numpy(h['prediction'])
        accuracy = (pred_tokens == torch.from_numpy(ground_truth_tokens)).float().mean().item()
        metrics['accuracies'].append(accuracy)

        # Grid-level accuracy (after inverse augmentation)
        try:
            pred_grid = tokens_to_grid(pred_tokens, batch['puzzle_name'])
            grid_acc = (pred_grid == ground_truth_grid).mean()
            metrics['grid_accuracy'].append(float(grid_acc))
        except:
            metrics['grid_accuracy'].append(0.0)

        # Q-halt value
        metrics['q_halt_values'].append(h['q_halt_logit'])

        # Prediction changes from previous step
        if prev_pred is not None:
            changes = (h['prediction'] != prev_pred).sum()
            metrics['prediction_changes'].append(int(changes))

        # z_H movement
        if prev_z_H is not None:
            movement = torch.norm(h['z_H_mean'][test_idx] - prev_z_H[test_idx]).item()
            metrics['z_H_movements'].append(float(movement))

        prev_pred = h['prediction']
        prev_z_H = h['z_H_mean']

    # Final metrics
    metrics['is_solved'] = bool(metrics['accuracies'][-1] == 1.0)
    metrics['final_accuracy'] = float(metrics['accuracies'][-1])
    metrics['final_grid_accuracy'] = float(metrics['grid_accuracy'][-1])

    # Convergence detection (when predictions stop changing)
    metrics['convergence_step'] = None
    for i, changes in enumerate(metrics['prediction_changes']):
        if changes == 0:
            metrics['convergence_step'] = i + 1
            break

    return metrics


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def visualize_single_puzzle(
    history: List[Dict],
    batch: Dict,
    metrics: Dict,
    output_path: str
):
    """Create visualization for a single puzzle showing ACT progression."""

    # ARC color palette
    ARC_COLORS = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    def draw_arc_grid(ax, grid, title):
        grid = np.array(grid)
        height, width = grid.shape
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

        for i in range(height):
            for j in range(width):
                color_idx = int(grid[i, j]) % len(ARC_COLORS)
                color = ARC_COLORS[color_idx]
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=0,
                    facecolor=color
                )
                ax.add_patch(rect)

    # Get grids
    test_idx = batch['num_examples'] - 1
    input_tokens = batch['inputs'][test_idx].cpu()
    label_tokens = batch['labels'][test_idx].cpu()

    input_grid = tokens_to_grid(input_tokens, batch['puzzle_name'])
    ground_truth = tokens_to_grid(label_tokens, batch['puzzle_name'])

    # Get predictions at key steps (0, 3, 7, 15)
    key_steps = [0, 3, 7, 15]
    key_steps = [s for s in key_steps if s < len(history)]

    predictions = [tokens_to_grid(history[s]['prediction'], batch['puzzle_name'])
                   for s in key_steps]

    # Create figure
    num_cols = len(key_steps) + 2
    fig, axes = plt.subplots(3, num_cols, figsize=(3 * num_cols, 9))

    # Row 1: Grids
    draw_arc_grid(axes[0, 0], input_grid, "Input")
    draw_arc_grid(axes[0, 1], ground_truth, "Ground Truth")

    for i, (step, pred) in enumerate(zip(key_steps, predictions)):
        acc = metrics['accuracies'][step]
        draw_arc_grid(axes[0, i + 2], pred, f"Step {step}\n({acc:.1%})")

    # Row 2: Accuracy progression
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    ax = axes[1, 2]
    steps = list(range(len(metrics['accuracies'])))
    ax.plot(steps, [a * 100 for a in metrics['accuracies']], 'o-', linewidth=2)
    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Progression')
    ax.grid(True, alpha=0.3)

    # Row 2: Q-halt values
    ax = axes[1, 3]
    ax.plot(steps, metrics['q_halt_values'], 's-', linewidth=2, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Q-halt Logit')
    ax.set_title('Halting Signal')
    ax.grid(True, alpha=0.3)

    # Row 3: Prediction changes
    if len(axes.shape) > 1 and axes.shape[0] > 2:
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')

        if len(metrics['prediction_changes']) > 0:
            ax = axes[2, 2]
            ax.bar(range(1, len(metrics['prediction_changes']) + 1),
                   metrics['prediction_changes'])
            ax.set_xlabel('ACT Step')
            ax.set_ylabel('Tokens Changed')
            ax.set_title('Prediction Changes')
            ax.grid(True, alpha=0.3, axis='y')

        # Row 3: z_H movements
        if len(metrics['z_H_movements']) > 0:
            ax = axes[2, 3]
            ax.plot(range(1, len(metrics['z_H_movements']) + 1),
                    metrics['z_H_movements'], 'o-', linewidth=2, color='purple')
            ax.set_xlabel('ACT Step')
            ax.set_ylabel('||Δz_H||')
            ax.set_title('Latent Movement')
            ax.grid(True, alpha=0.3)

    # Title
    status = "✅ SOLVED" if metrics['is_solved'] else "❌ UNSOLVED"
    fig.suptitle(
        f"{status} - {batch['puzzle_name']} (Final: {metrics['final_accuracy']:.1%})",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_summary_plots(all_metrics: List[Dict], output_dir: str):
    """Generate summary statistics and visualizations."""
    print(f"\n📊 Generating summary statistics...")

    # Separate solved vs unsolved
    solved = [m for m in all_metrics if m['is_solved']]
    unsolved = [m for m in all_metrics if not m['is_solved']]

    summary = {
        'total_puzzles': len(all_metrics),
        'solved_count': len(solved),
        'unsolved_count': len(unsolved),
        'solve_rate': len(solved) / len(all_metrics) if all_metrics else 0,
        'avg_final_accuracy': np.mean([m['final_accuracy'] for m in all_metrics]),
        'avg_convergence_step_solved': np.mean([m['convergence_step'] for m in solved
                                                 if m['convergence_step'] is not None]),
        'avg_convergence_step_unsolved': np.mean([m['convergence_step'] for m in unsolved
                                                   if m['convergence_step'] is not None]),
    }

    # Save summary JSON
    summary_path = os.path.join(output_dir, 'summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary Statistics:")
    print(f"   Total puzzles: {summary['total_puzzles']}")
    print(f"   Solved: {summary['solved_count']} ({summary['solve_rate']:.1%})")
    print(f"   Avg final accuracy: {summary['avg_final_accuracy']:.1%}")

    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Solve rate
    ax = axes[0, 0]
    ax.bar(['Solved', 'Unsolved'], [len(solved), len(unsolved)], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Solve Rate: {summary["solve_rate"]:.1%}')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Final accuracy distribution
    ax = axes[0, 1]
    accuracies = [m['final_accuracy'] for m in all_metrics]
    ax.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Final Accuracy Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Convergence step distribution
    ax = axes[0, 2]
    conv_steps_solved = [m['convergence_step'] for m in solved if m['convergence_step'] is not None]
    conv_steps_unsolved = [m['convergence_step'] for m in unsolved if m['convergence_step'] is not None]

    if conv_steps_solved or conv_steps_unsolved:
        ax.hist([conv_steps_solved, conv_steps_unsolved],
                bins=16, label=['Solved', 'Unsolved'],
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('Convergence Step')
        ax.set_ylabel('Count')
        ax.set_title('Convergence Step Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Average accuracy progression (solved vs unsolved)
    ax = axes[1, 0]
    if solved:
        max_len = max(len(m['accuracies']) for m in solved)
        acc_matrix_solved = np.zeros((len(solved), max_len))
        for i, m in enumerate(solved):
            acc_matrix_solved[i, :len(m['accuracies'])] = m['accuracies']
        avg_acc_solved = acc_matrix_solved.mean(axis=0)
        ax.plot(range(max_len), avg_acc_solved * 100, 'o-', linewidth=2,
                label=f'Solved (n={len(solved)})', color='green')

    if unsolved:
        max_len = max(len(m['accuracies']) for m in unsolved)
        acc_matrix_unsolved = np.zeros((len(unsolved), max_len))
        for i, m in enumerate(unsolved):
            acc_matrix_unsolved[i, :len(m['accuracies'])] = m['accuracies']
        avg_acc_unsolved = acc_matrix_unsolved.mean(axis=0)
        ax.plot(range(max_len), avg_acc_unsolved * 100, 's-', linewidth=2,
                label=f'Unsolved (n={len(unsolved)})', color='red')

    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Accuracy Progression: Solved vs Unsolved')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Q-halt behavior (solved vs unsolved)
    ax = axes[1, 1]
    if solved:
        max_len = max(len(m['q_halt_values']) for m in solved)
        q_matrix_solved = np.zeros((len(solved), max_len))
        for i, m in enumerate(solved):
            q_matrix_solved[i, :len(m['q_halt_values'])] = m['q_halt_values']
        avg_q_solved = q_matrix_solved.mean(axis=0)
        ax.plot(range(max_len), avg_q_solved, 'o-', linewidth=2,
                label='Solved', color='green')

    if unsolved:
        max_len = max(len(m['q_halt_values']) for m in unsolved)
        q_matrix_unsolved = np.zeros((len(unsolved), max_len))
        for i, m in enumerate(unsolved):
            q_matrix_unsolved[i, :len(m['q_halt_values'])] = m['q_halt_values']
        avg_q_unsolved = q_matrix_unsolved.mean(axis=0)
        ax.plot(range(max_len), avg_q_unsolved, 's-', linewidth=2,
                label='Unsolved', color='red')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Average Q-halt Logit')
    ax.set_title('Q-halt Signal: Solved vs Unsolved')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: z_H movement patterns
    ax = axes[1, 2]
    if solved:
        max_len = max(len(m['z_H_movements']) for m in solved if m['z_H_movements'])
        z_matrix_solved = np.zeros((len(solved), max_len))
        for i, m in enumerate(solved):
            if m['z_H_movements']:
                z_matrix_solved[i, :len(m['z_H_movements'])] = m['z_H_movements']
        avg_z_solved = z_matrix_solved.mean(axis=0)
        ax.plot(range(1, max_len + 1), avg_z_solved, 'o-', linewidth=2,
                label='Solved', color='green')

    if unsolved:
        max_len = max(len(m['z_H_movements']) for m in unsolved if m['z_H_movements'])
        z_matrix_unsolved = np.zeros((len(unsolved), max_len))
        for i, m in enumerate(unsolved):
            if m['z_H_movements']:
                z_matrix_unsolved[i, :len(m['z_H_movements'])] = m['z_H_movements']
        avg_z_unsolved = z_matrix_unsolved.mean(axis=0)
        ax.plot(range(1, max_len + 1), avg_z_unsolved, 's-', linewidth=2,
                label='Unsolved', color='red')

    ax.set_xlabel('ACT Step')
    ax.set_ylabel('Average ||Δz_H||')
    ax.set_title('Latent Movement: Solved vs Unsolved')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_fig_path = os.path.join(output_dir, 'summary_plots.png')
    plt.savefig(summary_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved summary plots: {summary_fig_path}")


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ACT Analysis for All 400 Validation Puzzles"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
                        help="Path to TRM checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--identifiers", type=str,
                        default="/data/arc1concept-aug-1000/identifiers.json",
                        help="Path to identifiers.json")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/TinyRecursiveModels/latent_analysis/results/act_400",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--save_individual_plots", action="store_true",
                        help="Save individual puzzle visualizations")
    parser.add_argument("--num_puzzles", type=int, default=400,
                        help="Number of puzzles to analyze (default: all 400)")

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    puzzles_dir = os.path.join(output_dir, 'puzzles')
    if args.save_individual_plots:
        Path(puzzles_dir).mkdir(exist_ok=True)

    print("="*80)
    print("TRM ACT Analysis - 400 Validation Puzzles")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device)

    # Load all validation puzzles
    data_dict = load_all_validation_puzzles(args.data_path, args.identifiers)

    # Analyze each puzzle
    all_metrics = []
    num_to_analyze = min(args.num_puzzles, data_dict['num_puzzles'])

    print(f"\n🔬 Analyzing {num_to_analyze} puzzles...")

    for puzzle_idx in tqdm(range(num_to_analyze)):
        # Load puzzle
        batch = get_puzzle_batch(data_dict, puzzle_idx, args.device)

        # Run ACT with tracking
        history = run_act_with_tracking(model, batch, args.device, max_steps=16)

        # Analyze
        metrics = analyze_puzzle_act(history, batch, puzzle_idx)
        all_metrics.append(metrics)

        # Save individual visualization
        if args.save_individual_plots:
            puzzle_output_path = os.path.join(
                puzzles_dir,
                f"puzzle_{puzzle_idx:03d}_{batch['puzzle_name']}.png"
            )
            visualize_single_puzzle(history, batch, metrics, puzzle_output_path)

    # Save all metrics
    metrics_path = os.path.join(output_dir, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n💾 Saved metrics: {metrics_path}")

    # Generate summary plots
    generate_summary_plots(all_metrics, output_dir)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
