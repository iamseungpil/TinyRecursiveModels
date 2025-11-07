"""
Generate cross-puzzle visualizations from saved results.
"""

import os
import sys
import json
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

def load_results(results_dir: str):
    """Load all saved puzzle results."""
    all_results = []
    puzzle_metadata = []

    # Find all puzzle directories
    puzzle_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith('puzzle_')])

    print(f"Found {len(puzzle_dirs)} puzzle directories")

    for puzzle_dir in puzzle_dirs:
        puzzle_idx = int(puzzle_dir.split('_')[1])
        puzzle_path = os.path.join(results_dir, puzzle_dir)

        # Check if analysis completed (has joint_trajectory.png)
        if not os.path.exists(os.path.join(puzzle_path, 'joint_trajectory.png')):
            print(f"Skipping {puzzle_dir} (incomplete)")
            continue

        # Load saved data if exists
        data_file = os.path.join(puzzle_path, 'analysis_data.pkl')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            all_results.append(data['result'])
            puzzle_metadata.append(data['metadata'])

    print(f"Loaded {len(all_results)} complete puzzle analyses")
    return all_results, puzzle_metadata


def visualize_puzzle_embedding_space(
    all_results, puzzle_metadata, output_path: str
):
    """Visualize puzzle embedding space using t-SNE."""
    print(f"\n🎨 Generating puzzle embedding space visualization...")

    # Extract puzzle embeddings
    puzzle_embeddings = []
    is_solved = []
    puzzle_names = []

    for result, meta in zip(all_results, puzzle_metadata):
        # Average over puzzle embedding positions
        emb_tensor = result['puzzle_embedding']
        if isinstance(emb_tensor, torch.Tensor):
            emb = emb_tensor.mean(dim=0).detach().numpy()
        else:
            emb = emb_tensor.mean(axis=0)
        puzzle_embeddings.append(emb)
        is_solved.append(meta['is_solved'])
        puzzle_names.append(meta['puzzle_name'])

    puzzle_embeddings = np.array(puzzle_embeddings)
    is_solved = np.array(is_solved)

    # PCA for initial reduction
    pca = PCA(n_components=min(50, len(puzzle_embeddings) - 1))
    emb_pca = pca.fit_transform(puzzle_embeddings)

    # t-SNE for final 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(puzzle_embeddings) - 1))
    emb_2d = tsne.fit_transform(emb_pca)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Color by solution status
    ax = axes[0]
    colors = ['red' if not solved else 'green' for solved in is_solved]
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')

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
        Patch(facecolor='green', edgecolor='black', label=f'Solved ({is_solved.sum()})'),
        Patch(facecolor='red', edgecolor='black', label=f'Unsolved ({(~is_solved).sum()})')
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


def analyze_trajectory_patterns(all_results, puzzle_metadata):
    """Analyze common patterns across puzzles."""
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
            prev_H = history[step - 1]['z_H_mean'][0]
            curr_H = history[step]['z_H_mean'][0]

            if isinstance(prev_H, torch.Tensor):
                prev_H = prev_H.numpy()
                curr_H = curr_H.numpy()

            movements_H.append(np.linalg.norm(curr_H - prev_H))

            prev_L = history[step - 1]['z_L_mean'][0]
            curr_L = history[step]['z_L_mean'][0]

            if isinstance(prev_L, torch.Tensor):
                prev_L = prev_L.numpy()
                curr_L = curr_L.numpy()

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
            'avg_movement_H': float(np.mean([np.mean(m) for m in solved_movements_H])) if solved_movements_H else 0,
            'avg_movement_L': float(np.mean([np.mean(m) for m in solved_movements_L])) if solved_movements_L else 0,
        },
        'unsolved': {
            'count': len(unsolved_movements_H),
            'avg_movement_H': float(np.mean([np.mean(m) for m in unsolved_movements_H])) if unsolved_movements_H else 0,
            'avg_movement_L': float(np.mean([np.mean(m) for m in unsolved_movements_L])) if unsolved_movements_L else 0,
        }
    }

    print(f"  Solved puzzles: {stats['solved']['count']}")
    print(f"    Avg z_H movement: {stats['solved']['avg_movement_H']:.3f}")
    print(f"    Avg z_L movement: {stats['solved']['avg_movement_L']:.3f}")
    print(f"  Unsolved puzzles: {stats['unsolved']['count']}")
    print(f"    Avg z_H movement: {stats['unsolved']['avg_movement_H']:.3f}")
    print(f"    Avg z_L movement: {stats['unsolved']['avg_movement_L']:.3f}")

    return stats


def main():
    results_dir = '/home/ubuntu/TinyRecursiveModels/latent_analysis/results/comprehensive_30_puzzles'

    print("="*80)
    print("Generating Cross-Puzzle Visualizations")
    print("="*80)

    # Load results
    all_results, puzzle_metadata = load_results(results_dir)

    if len(all_results) == 0:
        print("ERROR: No complete puzzle analyses found!")
        return

    # Generate visualizations
    visualize_puzzle_embedding_space(
        all_results,
        puzzle_metadata,
        os.path.join(results_dir, 'puzzle_embedding_space.png')
    )

    # Analyze trajectories
    trajectory_stats = analyze_trajectory_patterns(all_results, puzzle_metadata)

    # Save summary
    summary = {
        'num_puzzles_analyzed': len(all_results),
        'num_solved': sum(m['is_solved'] for m in puzzle_metadata),
        'puzzle_metadata': puzzle_metadata,
        'trajectory_stats': trajectory_stats,
    }

    summary_path = os.path.join(results_dir, 'comprehensive_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Summary:")
    print(f"   Puzzles analyzed: {summary['num_puzzles_analyzed']}")
    print(f"   Puzzles solved: {summary['num_solved']} ({100*summary['num_solved']/summary['num_puzzles_analyzed']:.1f}%)")
    print(f"\n💾 Summary saved to: {summary_path}")
    print("="*80)


if __name__ == "__main__":
    main()
