"""
Comprehensive TRM Latent Space Analysis

This script extends the POC to:
1. Track BOTH z_H and z_L dynamics
2. Extract and visualize puzzle embeddings
3. Analyze diverse puzzle types
4. Compare reasoning patterns across tasks

Usage:
    python comprehensive_analysis.py --num_puzzles 30 --output_dir results/comprehensive/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    return batch


def sample_diverse_puzzles(data_path: str, num_puzzles: int, seed: int = 42) -> List[int]:
    """
    Sample diverse puzzles from the test set.

    Strategy: Sample evenly across the puzzle index space to get variety.
    """
    test_dir = os.path.join(data_path, 'test')
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))
    total_puzzles = len(puzzle_indices) - 1

    # Sample evenly
    np.random.seed(seed)
    sampled_indices = np.linspace(0, total_puzzles - 1, num_puzzles, dtype=int)

    # Add some randomness
    sampled_indices = sampled_indices + np.random.randint(-100, 100, size=num_puzzles)
    sampled_indices = np.clip(sampled_indices, 0, total_puzzles - 1)

    return sampled_indices.tolist()


# ============================================================================
# 2. ENHANCED INFERENCE ENGINE (z_H + z_L + puzzle_emb)
# ============================================================================

def comprehensive_inference(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    device: str,
    max_h_steps: int = 3
) -> Dict:
    """
    Run TRM inference with comprehensive state capture.

    Returns:
        Dict containing:
            - history: List of H-step states (z_H, z_L, predictions)
            - puzzle_embedding: [puzzle_emb_len, hidden_size] - task encoding
            - input_embeddings: [seq_len + puzzle_emb_len, hidden_size]
    """
    print(f"\n🔬 Running comprehensive inference (max_h_steps={max_h_steps})...")

    # Move batch to device
    inputs = batch['inputs'].to(device)
    labels = batch['labels'].to(device)
    puzzle_ids = batch['puzzle_identifiers'].to(device)

    batch_size = inputs.shape[0]

    # Access inner model
    inner = model.inner

    # Initialize carry
    z_H = inner.H_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)
    z_L = inner.L_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)

    # Prepare sequence info
    seq_info = dict(
        cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
    )

    # Input embeddings (constant across all H-steps)
    input_embeddings = inner._input_embeddings(inputs, puzzle_ids)

    # Extract puzzle embedding (first 16 positions)
    puzzle_embedding = input_embeddings[0, :inner.puzzle_emb_len].clone().cpu()  # [16, 512]

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
            output_logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
            pred_tokens = output_logits.argmax(dim=-1)

            # Save comprehensive state snapshot
            history.append({
                'h_step': h_step,
                # Full representations
                'z_H': z_H.clone().float().cpu(),  # [batch, 916, 512]
                'z_L': z_L.clone().float().cpu(),  # [batch, 916, 512]
                # Separated components
                'z_H_puzzle': z_H[:, :inner.puzzle_emb_len].clone().float().cpu(),  # [batch, 16, 512]
                'z_H_grid': z_H[:, inner.puzzle_emb_len:].clone().float().cpu(),    # [batch, 900, 512]
                'z_L_puzzle': z_L[:, :inner.puzzle_emb_len].clone().float().cpu(),
                'z_L_grid': z_L[:, inner.puzzle_emb_len:].clone().float().cpu(),
                # Aggregated for trajectory
                'z_H_mean': z_H.mean(dim=1).float().cpu(),  # [batch, 512]
                'z_L_mean': z_L.mean(dim=1).float().cpu(),
                # Predictions
                'output_logits': output_logits.clone().cpu(),
                'pred_tokens': pred_tokens.clone().cpu(),
            })

    print(f"✅ Captured {len(history)} H-step states + puzzle embedding")

    return {
        'history': history,
        'puzzle_embedding': puzzle_embedding,  # [16, 512]
        'input_embeddings': input_embeddings.clone().cpu(),  # [batch, 916, 512]
    }


# ============================================================================
# 3. GRID UTILITIES
# ============================================================================

def tokens_to_grid(tokens: torch.Tensor, puzzle_name: str) -> np.ndarray:
    """Convert token sequence to grid (apply inverse augmentation)."""
    orig_name, inverse_fn = inverse_aug(puzzle_name)
    tokens_np = tokens.numpy()
    grid = inverse_fn(_crop(tokens_np))
    return grid


# ============================================================================
# 4. ENHANCED VISUALIZATIONS
# ============================================================================

def visualize_joint_trajectory(
    results: Dict,
    batch: Dict,
    output_path: str
):
    """
    Visualize joint z_H and z_L trajectories.

    Plots:
        1. z_H and z_L in shared PCA space
        2. z_H and z_L movement magnitudes
        3. Divergence between z_H and z_L over time
        4. PC coordinates for both
    """
    print(f"\n🎨 Generating joint z_H + z_L trajectory visualization...")

    history = results['history']

    # Extract sequences
    z_H_sequence = [h['z_H_mean'].numpy() for h in history]  # List of [batch, 512]
    z_L_sequence = [h['z_L_mean'].numpy() for h in history]

    # Stack for PCA
    all_z_H = np.vstack(z_H_sequence)  # [num_steps * batch, 512]
    all_z_L = np.vstack(z_L_sequence)
    all_z = np.vstack([all_z_H, all_z_L])  # Combined for shared PCA space

    batch_size = z_H_sequence[0].shape[0]
    num_steps = len(history)

    # Shared PCA
    n_components = min(10, all_z.shape[0] - 1)
    pca = PCA(n_components=n_components)
    all_z_pca = pca.fit_transform(all_z)

    # Split back
    z_H_pca = all_z_pca[:len(all_z_H)]
    z_L_pca = all_z_pca[len(all_z_H):]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Joint trajectory in PCA space
    ax = axes[0, 0]
    for batch_idx in range(min(batch_size, 5)):  # Limit to 5 examples
        # z_H trajectory
        traj_H = np.array([z_H_pca[step * batch_size + batch_idx, :2] for step in range(num_steps)])
        ax.plot(traj_H[:, 0], traj_H[:, 1], '-o', alpha=0.7, linewidth=2, markersize=8, label=f'z_H Ex{batch_idx}')

        # z_L trajectory
        traj_L = np.array([z_L_pca[step * batch_size + batch_idx, :2] for step in range(num_steps)])
        ax.plot(traj_L[:, 0], traj_L[:, 1], '--s', alpha=0.7, linewidth=2, markersize=6, label=f'z_L Ex{batch_idx}')

        # Start/end markers
        ax.scatter(traj_H[0, 0], traj_H[0, 1], c='green', s=200, marker='o', edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(traj_H[-1, 0], traj_H[-1, 1], c='red', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Joint z_H + z_L Trajectory in Shared PCA Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    # Plot 2: Movement magnitudes
    ax = axes[0, 1]
    batch_idx = 0  # Focus on first example

    distances_H = []
    distances_L = []
    for step in range(1, num_steps):
        prev_H = z_H_sequence[step - 1][batch_idx]
        curr_H = z_H_sequence[step][batch_idx]
        dist_H = np.linalg.norm(curr_H - prev_H)
        distances_H.append(dist_H)

        prev_L = z_L_sequence[step - 1][batch_idx]
        curr_L = z_L_sequence[step][batch_idx]
        dist_L = np.linalg.norm(curr_L - prev_L)
        distances_L.append(dist_L)

    x = range(1, num_steps)
    ax.plot(x, distances_H, '-o', linewidth=2, markersize=8, label='z_H movement')
    ax.plot(x, distances_L, '--s', linewidth=2, markersize=8, label='z_L movement')

    ax.set_xlabel('H-step Transition', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Latent Space Movement Magnitude', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i-1}→{i}' for i in range(1, num_steps)])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Divergence between z_H and z_L
    ax = axes[1, 0]
    divergences = []
    for step in range(num_steps):
        z_H = z_H_sequence[step][batch_idx]
        z_L = z_L_sequence[step][batch_idx]
        div = np.linalg.norm(z_H - z_L)
        divergences.append(div)

    ax.plot(range(num_steps), divergences, '-o', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('H-step', fontsize=12)
    ax.set_ylabel('||z_H - z_L||', fontsize=12)
    ax.set_title('z_H vs z_L Divergence Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_steps))
    ax.grid(True, alpha=0.3)

    # Plot 4: PC coordinates comparison
    ax = axes[1, 1]
    for pc_idx in range(min(3, pca.n_components_)):
        pc_values_H = [z_H_pca[step * batch_size, pc_idx] for step in range(num_steps)]
        pc_values_L = [z_L_pca[step * batch_size, pc_idx] for step in range(num_steps)]

        ax.plot(range(num_steps), pc_values_H, '-o', linewidth=2, markersize=8,
                label=f'z_H PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.1%})')
        ax.plot(range(num_steps), pc_values_L, '--s', linewidth=2, markersize=6,
                label=f'z_L PC{pc_idx+1}')

    ax.set_xlabel('H-step', fontsize=12)
    ax.set_ylabel('PC Coordinate Value', fontsize=12)
    ax.set_title('Principal Components: z_H vs z_L', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_steps))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")


def visualize_puzzle_embedding_space(
    all_results: List[Dict],
    puzzle_metadata: List[Dict],
    output_path: str
):
    """
    Visualize puzzle embedding space using t-SNE.

    Shows how different puzzles cluster in the learned embedding space.
    Color-code by whether the puzzle was solved.
    """
    print(f"\n🎨 Generating puzzle embedding space visualization...")

    # Extract puzzle embeddings
    puzzle_embeddings = []
    is_solved = []
    puzzle_names = []

    for result, meta in zip(all_results, puzzle_metadata):
        # Average over puzzle embedding positions
        emb = result['puzzle_embedding'].mean(dim=0).detach().numpy()  # [512]
        puzzle_embeddings.append(emb)
        is_solved.append(meta['is_solved'])
        puzzle_names.append(meta['puzzle_name'])

    puzzle_embeddings = np.array(puzzle_embeddings)  # [num_puzzles, 512]
    is_solved = np.array(is_solved)

    # Dimensionality reduction
    # PCA for initial reduction
    pca = PCA(n_components=50)
    emb_pca = pca.fit_transform(puzzle_embeddings)

    # t-SNE for final 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(puzzle_embeddings) - 1))
    emb_2d = tsne.fit_transform(emb_pca)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Color by solution status
    ax = axes[0]
    colors = ['red' if not solved else 'green' for solved in is_solved]
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')

    # Annotate some points
    for i in range(min(10, len(puzzle_names))):
        ax.annotate(puzzle_names[i][:10], (emb_2d[i, 0], emb_2d[i, 1]),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Puzzle Embedding Space (Colored by Solution Status)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Solved'),
        Patch(facecolor='red', edgecolor='black', label='Unsolved')
    ]
    ax.legend(handles=legend_elements, loc='best')

    # Plot 2: PCA variance
    ax = axes[1]
    ax.bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
           pca.explained_variance_ratio_[:20], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('Puzzle Embedding PCA Variance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")


def visualize_grid_evolution(
    results: Dict,
    batch: Dict,
    output_path: str
):
    """Visualize grid evolution (same as POC)."""
    print(f"\n🎨 Generating grid evolution visualization...")

    history = results['history']

    ARC_COLORS = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    def draw_arc_grid(ax, grid, title, show_grid=True):
        grid = np.array(grid)
        height, width = grid.shape

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight='bold')
        if not show_grid:
            ax.axis('off')

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

    test_idx = len(batch['inputs']) - 1
    input_tokens = batch['inputs'][test_idx]
    label_tokens = batch['labels'][test_idx]
    puzzle_name = batch['puzzle_name']

    input_grid = tokens_to_grid(input_tokens, puzzle_name)
    ground_truth = tokens_to_grid(label_tokens, puzzle_name)
    predictions = [tokens_to_grid(h['pred_tokens'][test_idx], puzzle_name) for h in history]

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
        error_mask = (pred != ground_truth).astype(float)
        num_errors = error_mask.sum()
        accuracy = 100 * (1 - num_errors / error_mask.size)

        axes[1, h_step + 2].imshow(error_mask, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        axes[1, h_step + 2].set_title(f"Errors: {int(num_errors)}\nAccuracy: {accuracy:.1f}%", fontsize=9)
        axes[1, h_step + 2].axis('off')

    is_solved = np.array_equal(predictions[-1], ground_truth)
    status = "✅ SOLVED" if is_solved else "❌ UNSOLVED"
    fig.suptitle(f"{status} - {puzzle_name}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")


# ============================================================================
# 5. CROSS-PUZZLE ANALYSIS
# ============================================================================

def analyze_trajectory_patterns(all_results: List[Dict], puzzle_metadata: List[Dict]) -> Dict:
    """
    Analyze common patterns across puzzles.

    Returns statistics about:
    - Average movement magnitudes
    - Convergence patterns
    - Differences between solved vs unsolved
    """
    print("\n📊 Analyzing trajectory patterns across puzzles...")

    solved_movements_H = []
    unsolved_movements_H = []
    solved_movements_L = []
    unsolved_movements_L = []

    for result, meta in zip(all_results, puzzle_metadata):
        history = result['history']
        is_solved = meta['is_solved']

        # Compute movements
        movements_H = []
        movements_L = []
        for step in range(1, len(history)):
            prev_H = history[step - 1]['z_H_mean'][0].numpy()
            curr_H = history[step]['z_H_mean'][0].numpy()
            movements_H.append(np.linalg.norm(curr_H - prev_H))

            prev_L = history[step - 1]['z_L_mean'][0].numpy()
            curr_L = history[step]['z_L_mean'][0].numpy()
            movements_L.append(np.linalg.norm(curr_L - prev_L))

        if is_solved:
            solved_movements_H.append(movements_H)
            solved_movements_L.append(movements_L)
        else:
            unsolved_movements_H.append(movements_H)
            unsolved_movements_L.append(movements_L)

    stats = {
        'solved': {
            'count': len(solved_movements_H),
            'avg_movement_H': np.mean([np.mean(m) for m in solved_movements_H]) if solved_movements_H else 0,
            'avg_movement_L': np.mean([np.mean(m) for m in solved_movements_L]) if solved_movements_L else 0,
        },
        'unsolved': {
            'count': len(unsolved_movements_H),
            'avg_movement_H': np.mean([np.mean(m) for m in unsolved_movements_H]) if unsolved_movements_H else 0,
            'avg_movement_L': np.mean([np.mean(m) for m in unsolved_movements_L]) if unsolved_movements_L else 0,
        }
    }

    print(f"  Solved puzzles: {stats['solved']['count']}")
    print(f"    Avg z_H movement: {stats['solved']['avg_movement_H']:.3f}")
    print(f"    Avg z_L movement: {stats['solved']['avg_movement_L']:.3f}")
    print(f"  Unsolved puzzles: {stats['unsolved']['count']}")
    print(f"    Avg z_H movement: {stats['unsolved']['avg_movement_H']:.3f}")
    print(f"    Avg z_L movement: {stats['unsolved']['avg_movement_L']:.3f}")

    return stats


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def comprehensive_analysis(
    checkpoint_path: str,
    data_path: str,
    identifiers_path: str,
    num_puzzles: int,
    output_dir: str,
    device: str = "cuda"
):
    """
    Complete comprehensive analysis across multiple puzzles.
    """
    print("="*80)
    print(f"TRM Comprehensive Latent Space Analysis")
    print("="*80)
    print(f"Number of puzzles: {num_puzzles}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load identifier map
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}

    # Load model
    model, config = load_checkpoint(checkpoint_path, device)

    # Sample diverse puzzles
    puzzle_indices = sample_diverse_puzzles(data_path, num_puzzles)
    print(f"\n📋 Selected puzzle indices: {puzzle_indices[:10]}..." if len(puzzle_indices) > 10 else f"\n📋 Selected puzzle indices: {puzzle_indices}")

    # Analyze each puzzle
    all_results = []
    puzzle_metadata = []

    for i, puzzle_idx in enumerate(puzzle_indices):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{num_puzzles}] Analyzing puzzle {puzzle_idx}")
        print(f"{'='*80}")

        try:
            # Load puzzle
            batch = load_test_puzzle(data_path, puzzle_idx, identifier_map)
            print(f"Puzzle: {batch['puzzle_name']}")
            print(f"Examples: {len(batch['inputs'])}")

            # Run inference
            result = comprehensive_inference(model, batch, device, max_h_steps=config['H_cycles'])

            # Check if solved
            test_idx = len(batch['inputs']) - 1
            final_pred = tokens_to_grid(result['history'][-1]['pred_tokens'][test_idx], batch['puzzle_name'])
            ground_truth = tokens_to_grid(batch['labels'][test_idx], batch['puzzle_name'])
            is_solved = np.array_equal(final_pred, ground_truth)

            # Save results
            all_results.append(result)
            puzzle_metadata.append({
                'puzzle_idx': puzzle_idx,
                'puzzle_name': batch['puzzle_name'],
                'puzzle_id': batch['puzzle_id'],
                'is_solved': is_solved,
                'num_examples': len(batch['inputs']),
            })

            # Generate individual visualizations
            puzzle_output_dir = os.path.join(output_dir, f'puzzle_{puzzle_idx}')
            Path(puzzle_output_dir).mkdir(parents=True, exist_ok=True)

            visualize_grid_evolution(result, batch, os.path.join(puzzle_output_dir, 'grid_evolution.png'))
            visualize_joint_trajectory(result, batch, os.path.join(puzzle_output_dir, 'joint_trajectory.png'))

            print(f"✅ Puzzle {puzzle_idx}: {'SOLVED' if is_solved else 'UNSOLVED'}")

        except Exception as e:
            print(f"❌ Error analyzing puzzle {puzzle_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("Generating cross-puzzle analyses...")
    print(f"{'='*80}")

    # Cross-puzzle visualizations
    visualize_puzzle_embedding_space(
        all_results,
        puzzle_metadata,
        os.path.join(output_dir, 'puzzle_embedding_space.png')
    )

    # Trajectory pattern analysis
    trajectory_stats = analyze_trajectory_patterns(all_results, puzzle_metadata)

    # Save comprehensive summary
    summary = {
        'num_puzzles_analyzed': len(all_results),
        'num_solved': sum(m['is_solved'] for m in puzzle_metadata),
        'puzzle_metadata': puzzle_metadata,
        'trajectory_stats': trajectory_stats,
        'config': config,
    }

    summary_path = os.path.join(output_dir, 'comprehensive_summary.json')
    with open(summary_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(summary, f, indent=2, default=convert)

    print(f"\n📊 Summary:")
    print(f"   Puzzles analyzed: {summary['num_puzzles_analyzed']}")
    print(f"   Puzzles solved: {summary['num_solved']} ({100*summary['num_solved']/summary['num_puzzles_analyzed']:.1f}%)")
    print(f"\n💾 Comprehensive summary saved to: {summary_path}")
    print(f"\n✅ Analysis complete! Results in: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="TRM Comprehensive Latent Space Analysis")
    parser.add_argument("--checkpoint", type=str,
                        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
                        help="Path to TRM checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--identifiers", type=str,
                        default="/data/arc1concept-aug-1000/identifiers.json",
                        help="Path to identifiers.json")
    parser.add_argument("--num_puzzles", type=int, default=30,
                        help="Number of puzzles to analyze")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/TinyRecursiveModels/latent_analysis/results/comprehensive",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    comprehensive_analysis(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        identifiers_path=args.identifiers,
        num_puzzles=args.num_puzzles,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
