"""
Proof-of-Concept: Step-by-Step TRM Inference with Intermediate State Capture

This script demonstrates the feasibility of extracting and visualizing
intermediate reasoning states from TRM during ARC puzzle solving.

Key Features:
1. Custom inference loop bypassing ACT wrapper
2. Full state capture at each H-cycle step
3. Grid evolution visualization
4. Latent space trajectory visualization

Usage:
    python step_by_step_inference_poc.py --puzzle_idx 0 --output_dir results/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA

# Add project root to path
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
    """Load TRM model with correct configuration."""
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

    # Config matching checkpoint
    config = {
        "batch_size": 1,  # Process one at a time
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 876406,
        "puzzle_emb_ndim": 512,
        "puzzle_emb_len": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
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

    print(f"✅ Model loaded (H_cycles={config['H_cycles']}, L_cycles={config['L_cycles']})")
    return model, config


def load_test_puzzle(data_path: str, puzzle_idx: int, identifier_map: Dict[int, str]):
    """Load a single test puzzle by index."""
    test_dir = os.path.join(data_path, 'test')

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    # Get puzzle range
    start_idx = puzzle_indices[puzzle_idx]
    end_idx = puzzle_indices[puzzle_idx + 1]

    # Load all examples for this puzzle
    batch = {
        'inputs': torch.from_numpy(inputs[start_idx:end_idx].astype(np.int64)),
        'labels': torch.from_numpy(labels[start_idx:end_idx].astype(np.int64)),
        'puzzle_identifiers': torch.full(
            (end_idx - start_idx,),
            puzzle_identifiers[puzzle_idx],
            dtype=torch.long
        ),
        'puzzle_id': int(puzzle_identifiers[puzzle_idx]),
        'puzzle_name': identifier_map.get(puzzle_identifiers[puzzle_idx], f"<unknown>"),
    }

    print(f"✅ Loaded puzzle {puzzle_idx}: {batch['puzzle_name']}")
    print(f"   Examples: {len(batch['inputs'])} (last is test)")
    return batch


# ============================================================================
# 2. STEP-BY-STEP INFERENCE ENGINE
# ============================================================================

def step_by_step_inference(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    device: str,
    max_h_steps: int = 3
) -> List[Dict]:
    """
    Run TRM inference with intermediate state capture.

    This bypasses the ACT wrapper and manually steps through H-cycles,
    capturing z_H, z_L, and output predictions at each step.

    Returns:
        history: List of dicts (one per H-step) containing:
            - h_step: int (0, 1, 2)
            - z_H: [batch, seq_len, hidden_size]
            - z_L: [batch, seq_len, hidden_size]
            - output_logits: [batch, seq_len, vocab_size]
            - pred_tokens: [batch, seq_len]
            - z_H_mean: [batch, hidden_size] (averaged over sequence)
    """
    print(f"\n🔬 Running step-by-step inference (max_h_steps={max_h_steps})...")

    # Move batch to device
    inputs = batch['inputs'].to(device)
    labels = batch['labels'].to(device)
    puzzle_ids = batch['puzzle_identifiers'].to(device)

    batch_dict = {
        'inputs': inputs,
        'labels': labels,
        'puzzle_identifiers': puzzle_ids
    }

    batch_size = inputs.shape[0]

    # Access inner model (bypass ACT wrapper)
    inner = model.inner

    # Initialize carry (empty -> will be reset on first use)
    z_H = inner.H_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)
    z_L = inner.L_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)

    # Prepare sequence info (RoPE embeddings)
    seq_info = dict(
        cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
    )

    # Input embeddings (constant across all H-steps)
    input_embeddings = inner._input_embeddings(inputs, puzzle_ids)

    history = []

    with torch.no_grad():
        for h_step in range(max_h_steps):
            print(f"  H-step {h_step}...")

            # L-cycles (low-level refinement)
            for l_step in range(model.config.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_embeddings, **seq_info)

            # H-cycle update (high-level reasoning)
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Generate output predictions
            output_logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]  # Remove puzzle emb positions
            pred_tokens = output_logits.argmax(dim=-1)

            # Save state snapshot
            history.append({
                'h_step': h_step,
                'z_H': z_H.clone().float().cpu(),  # [batch, seq_len, hidden]
                'z_L': z_L.clone().float().cpu(),
                'output_logits': output_logits.clone().cpu(),
                'pred_tokens': pred_tokens.clone().cpu(),
                'z_H_mean': z_H.mean(dim=1).float().cpu(),  # [batch, hidden] - for trajectory
                'z_L_mean': z_L.mean(dim=1).float().cpu(),
            })

    print(f"✅ Captured {len(history)} H-step states")
    return history


# ============================================================================
# 3. GRID UTILITIES
# ============================================================================

def tokens_to_grid(tokens: torch.Tensor, puzzle_name: str) -> np.ndarray:
    """Convert token sequence to grid (apply inverse augmentation)."""
    # Get inverse augmentation function
    orig_name, inverse_fn = inverse_aug(puzzle_name)

    # Crop and apply inverse
    tokens_np = tokens.numpy()
    grid = inverse_fn(_crop(tokens_np))

    return grid


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def visualize_grid_evolution(
    history: List[Dict],
    batch: Dict,
    output_path: str
):
    """
    Visualize how the predicted grid evolves across H-steps.

    Layout:
        Row 1: Input | Ground Truth | H-step 0 | H-step 1 | H-step 2
        Row 2: Diff heatmap (error magnitude per cell)
    """
    print(f"\n🎨 Generating grid evolution visualization...")

    # ARC color palette (0-9)
    ARC_COLORS = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    def draw_arc_grid(ax, grid, title, show_grid=True):
        """Draw a single ARC grid."""
        grid = np.array(grid)
        height, width = grid.shape

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight='bold')
        if not show_grid:
            ax.axis('off')

        # Draw cells
        for i in range(height):
            for j in range(width):
                color_idx = int(grid[i, j]) % len(ARC_COLORS)
                color = ARC_COLORS[color_idx]
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=0.5 if show_grid else 0,
                    edgecolor='lightgray' if show_grid else 'none',
                    facecolor=color
                )
                ax.add_patch(rect)

    # Get test example (last in batch)
    test_idx = len(batch['inputs']) - 1
    input_tokens = batch['inputs'][test_idx]
    label_tokens = batch['labels'][test_idx]
    puzzle_name = batch['puzzle_name']

    # Convert to grids
    input_grid = tokens_to_grid(input_tokens, puzzle_name)
    ground_truth = tokens_to_grid(label_tokens, puzzle_name)

    # Get predictions from history
    predictions = [tokens_to_grid(h['pred_tokens'][test_idx], puzzle_name) for h in history]

    # Create figure
    num_steps = len(history)
    fig, axes = plt.subplots(2, num_steps + 2, figsize=(4 * (num_steps + 2), 8))

    # Row 1: Grids
    draw_arc_grid(axes[0, 0], input_grid, "Input")
    draw_arc_grid(axes[0, 1], ground_truth, "Ground Truth")

    for h_step, pred in enumerate(predictions):
        draw_arc_grid(axes[0, h_step + 2], pred, f"H-step {h_step}")

    # Row 2: Error heatmaps
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    for h_step, pred in enumerate(predictions):
        # Compute error mask
        error_mask = (pred != ground_truth).astype(float)
        num_errors = error_mask.sum()
        accuracy = 100 * (1 - num_errors / error_mask.size)

        # Heatmap
        im = axes[1, h_step + 2].imshow(error_mask, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        axes[1, h_step + 2].set_title(f"Errors: {int(num_errors)}\nAccuracy: {accuracy:.1f}%", fontsize=9)
        axes[1, h_step + 2].axis('off')

    # Overall title
    is_solved = np.array_equal(predictions[-1], ground_truth)
    status = "✅ SOLVED" if is_solved else "❌ UNSOLVED"
    fig.suptitle(f"{status} - {puzzle_name}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")


def visualize_latent_trajectory(
    history: List[Dict],
    output_path: str
):
    """
    Visualize latent space trajectory across H-steps.

    Plots:
        1. z_H trajectory in 2D PCA space
        2. z_H movement magnitude over time
        3. First 10 PC coordinates over time
    """
    print(f"\n🎨 Generating latent trajectory visualization...")

    # Extract latent means
    z_H_sequence = [h['z_H_mean'].numpy() for h in history]  # List of [batch, hidden]
    z_L_sequence = [h['z_L_mean'].numpy() for h in history]

    # Stack for analysis
    all_z_H = np.vstack(z_H_sequence)  # [num_steps * batch, hidden]
    batch_size = z_H_sequence[0].shape[0]
    num_steps = len(history)

    # PCA projection (use min of 10 or available samples)
    n_pca_components = min(10, all_z_H.shape[0] - 1)
    pca = PCA(n_components=n_pca_components)
    z_H_pca = pca.fit_transform(all_z_H)  # [num_steps * batch, n_pca_components]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Trajectory in PCA space (2D)
    ax = axes[0, 0]
    for batch_idx in range(batch_size):
        trajectory = np.array([z_H_pca[step * batch_size + batch_idx, :2] for step in range(num_steps)])

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], '-o', alpha=0.7, linewidth=2, markersize=8)
        # Start point
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=200, marker='o', edgecolors='black', linewidths=2, zorder=10)
        # End point
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)

        # Annotate steps
        for step in range(num_steps):
            ax.annotate(f'{step}', trajectory[step], fontsize=8, ha='center', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('z_H Trajectory in PCA Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(['Trajectory', 'Start (H=0)', 'End (H=2)'], loc='best')

    # Plot 2: Movement magnitude over time
    ax = axes[0, 1]
    for batch_idx in range(batch_size):
        distances = []
        for step in range(1, num_steps):
            prev = z_H_sequence[step - 1][batch_idx]
            curr = z_H_sequence[step][batch_idx]
            dist = np.linalg.norm(curr - prev)
            distances.append(dist)

        ax.plot(range(1, num_steps), distances, '-o', linewidth=2, markersize=8, label=f'Example {batch_idx}')

    ax.set_xlabel('H-step Transition', fontsize=12)
    ax.set_ylabel('||z_H(t) - z_H(t-1)||', fontsize=12)
    ax.set_title('Latent Space Movement Magnitude', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, num_steps))
    ax.set_xticklabels([f'{i-1}→{i}' for i in range(1, num_steps)])
    ax.grid(True, alpha=0.3)
    if batch_size <= 5:
        ax.legend()

    # Plot 3: PC coordinates over time (first 5 PCs)
    ax = axes[1, 0]
    for pc_idx in range(min(5, pca.n_components_)):
        pc_values = [z_H_pca[step * batch_size, pc_idx] for step in range(num_steps)]
        ax.plot(range(num_steps), pc_values, '-o', linewidth=2, markersize=8,
                label=f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.1%})')

    ax.set_xlabel('H-step', fontsize=12)
    ax.set_ylabel('PC Coordinate Value', fontsize=12)
    ax.set_title('Principal Components Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_steps))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Plot 4: Variance explained
    ax = axes[1, 1]
    n_bars = min(10, pca.n_components_)
    ax.bar(range(1, n_bars + 1), pca.explained_variance_ratio_[:n_bars], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")


# ============================================================================
# 5. METRICS
# ============================================================================

def compute_metrics(history: List[Dict], batch: Dict, puzzle_name: str) -> Dict:
    """Compute convergence and stability metrics."""
    test_idx = len(batch['inputs']) - 1
    ground_truth = tokens_to_grid(batch['labels'][test_idx], puzzle_name)

    metrics = {
        'num_h_steps': len(history),
        'grid_changes': [],
        'latent_movements': [],
        'accuracies': [],
        'is_solved': False,
    }

    prev_pred = None
    prev_z_H = None

    for h_step, state in enumerate(history):
        # Grid accuracy
        pred = tokens_to_grid(state['pred_tokens'][test_idx], puzzle_name)
        accuracy = (pred == ground_truth).mean()
        metrics['accuracies'].append(float(accuracy))

        # Grid changes from previous step
        if prev_pred is not None:
            changes = (pred != prev_pred).sum() / pred.size
            metrics['grid_changes'].append(float(changes))

        # Latent movement
        if prev_z_H is not None:
            z_H_curr = state['z_H_mean'][test_idx].numpy()
            z_H_prev = prev_z_H[test_idx].numpy()
            movement = np.linalg.norm(z_H_curr - z_H_prev)
            metrics['latent_movements'].append(float(movement))

        prev_pred = pred
        prev_z_H = state['z_H_mean']

    # Final solution status
    final_pred = tokens_to_grid(history[-1]['pred_tokens'][test_idx], puzzle_name)
    metrics['is_solved'] = bool(np.array_equal(final_pred, ground_truth))
    metrics['final_accuracy'] = float((final_pred == ground_truth).mean())

    return metrics


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def analyze_single_puzzle(
    checkpoint_path: str,
    data_path: str,
    identifiers_path: str,
    puzzle_idx: int,
    output_dir: str,
    device: str = "cuda"
):
    """Complete analysis pipeline for a single puzzle."""

    print("="*80)
    print(f"TRM Step-by-Step Analysis - Proof of Concept")
    print("="*80)
    print(f"Puzzle Index: {puzzle_idx}")
    print(f"Output Dir: {output_dir}")
    print("="*80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load identifier map
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}

    # Load model
    model, config = load_checkpoint(checkpoint_path, device)

    # Load puzzle
    batch = load_test_puzzle(data_path, puzzle_idx, identifier_map)

    # Run step-by-step inference
    history = step_by_step_inference(model, batch, device, max_h_steps=config['H_cycles'])

    # Generate visualizations
    visualize_grid_evolution(
        history,
        batch,
        os.path.join(output_dir, 'grid_evolution.png')
    )

    visualize_latent_trajectory(
        history,
        os.path.join(output_dir, 'latent_trajectory.png')
    )

    # Compute metrics
    metrics = compute_metrics(history, batch, batch['puzzle_name'])

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📊 Metrics:")
    print(f"   Solved: {metrics['is_solved']}")
    print(f"   Final Accuracy: {metrics['final_accuracy']:.1%}")
    print(f"   Grid Changes: {metrics['grid_changes']}")
    print(f"   Latent Movements: {[f'{m:.3f}' for m in metrics['latent_movements']]}")

    print(f"\n✅ Analysis complete! Results in: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="TRM Step-by-Step Inference POC")
    parser.add_argument("--checkpoint", type=str,
                        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
                        help="Path to TRM checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--identifiers", type=str,
                        default="/data/arc1concept-aug-1000/identifiers.json",
                        help="Path to identifiers.json")
    parser.add_argument("--puzzle_idx", type=int, default=0,
                        help="Puzzle index to analyze (0-999)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/TinyRecursiveModels/latent_analysis/results/poc",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    analyze_single_puzzle(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        identifiers_path=args.identifiers,
        puzzle_idx=args.puzzle_idx,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
