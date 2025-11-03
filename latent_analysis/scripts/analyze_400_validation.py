"""
Analyze all 400 validation puzzles - PCA + Nearest Neighbors
"""

import sys
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path


def main():
    data_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/validation_400_puzzles.json"
    test_puzzles_path = "/data/arc1concept-aug-1000/test_puzzles.json"
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("400 Validation Puzzle Analysis")
    print("="*70)

    # Load data
    print(f"\n📥 Loading data...")
    with open(data_path, 'r') as f:
        results = json.load(f)
    print(f"✅ Loaded {len(results)} puzzles")

    # Extract latents and labels
    latents = np.array([r['latent'] for r in results])
    solved = np.array([r['solved'] for r in results])
    puzzle_names = [r['puzzle_name'] for r in results]

    print(f"\n📊 Statistics:")
    print(f"   Total puzzles: {len(results)}")
    print(f"   Solved: {solved.sum()} ({solved.sum()/len(results)*100:.1f}%)")
    print(f"   Unsolved: {(~solved).sum()} ({(~solved).sum()/len(results)*100:.1f}%)")

    # PCA
    print(f"\n🔬 Running PCA...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    print(f"✅ PCA complete:")
    print(f"   PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"   Total: {pca.explained_variance_ratio_.sum():.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Binary solved/unsolved
    ax1 = axes[0]
    solved_mask = solved
    unsolved_mask = ~solved

    ax1.scatter(latents_2d[unsolved_mask, 0], latents_2d[unsolved_mask, 1],
                c='red', alpha=0.7, s=80, label=f'Unsolved ({unsolved_mask.sum()})',
                edgecolors='darkred', linewidths=1)
    ax1.scatter(latents_2d[solved_mask, 0], latents_2d[solved_mask, 1],
                c='green', alpha=0.7, s=80, label=f'Solved ({solved_mask.sum()})',
                edgecolors='darkgreen', linewidths=1)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('All 400 Validation Puzzles - Latent Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Density
    ax2 = axes[1]
    ax2.hexbin(latents_2d[:, 0], latents_2d[:, 1],
               gridsize=30, cmap='viridis', mincnt=1)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax2.set_title('Density Map (400 Puzzles)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'pca_400_validation_puzzles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved to {output_path}")

    # Compute separation
    if unsolved_mask.sum() > 0 and solved_mask.sum() > 0:
        solved_centroid = latents_2d[solved_mask].mean(axis=0)
        unsolved_centroid = latents_2d[unsolved_mask].mean(axis=0)
        separation = np.linalg.norm(solved_centroid - unsolved_centroid)

        print(f"\n🎯 Cluster Separation:")
        print(f"   Solved centroid: [{solved_centroid[0]:.3f}, {solved_centroid[1]:.3f}]")
        print(f"   Unsolved centroid: [{unsolved_centroid[0]:.3f}, {unsolved_centroid[1]:.3f}]")
        print(f"   Distance: {separation:.3f}")

    # Find nearest neighbors across decision boundary
    print(f"\n🔍 Finding nearest neighbor pairs across decision boundary...")

    # Compute distances
    distances = euclidean_distances(latents)

    # Find solved-unsolved pairs
    boundary_pairs = []
    for i in range(len(results)):
        if solved[i]:  # If this is solved
            # Find nearest unsolved
            unsolved_indices = np.where(~solved)[0]
            if len(unsolved_indices) > 0:
                dists_to_unsolved = distances[i, unsolved_indices]
                nearest_unsolved_idx = unsolved_indices[np.argmin(dists_to_unsolved)]

                boundary_pairs.append({
                    'solved_idx': i,
                    'unsolved_idx': nearest_unsolved_idx,
                    'solved_name': puzzle_names[i],
                    'unsolved_name': puzzle_names[nearest_unsolved_idx],
                    'distance': distances[i, nearest_unsolved_idx],
                    'pos_solved': latents_2d[i],
                    'pos_unsolved': latents_2d[nearest_unsolved_idx]
                })

    # Get unique pairs (avoid duplicates)
    seen = set()
    unique_pairs = []
    for pair in boundary_pairs:
        pair_key = tuple(sorted([pair['solved_name'], pair['unsolved_name']]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(pair)

    # Sort by distance
    unique_pairs.sort(key=lambda x: x['distance'])

    print(f"✅ Found {len(unique_pairs)} boundary pairs")

    # Print top 10 closest boundary pairs
    print(f"\n📍 Top 10 closest solved-unsolved pairs:")
    for idx, pair in enumerate(unique_pairs[:10], 1):
        print(f"{idx}. {pair['solved_name']} (solved) ↔ {pair['unsolved_name']} (unsolved)")
        print(f"   Distance: {pair['distance']:.3f}")
        print(f"   Position: [{pair['pos_solved'][0]:.2f}, {pair['pos_solved'][1]:.2f}] → [{pair['pos_unsolved'][0]:.2f}, {pair['pos_unsolved'][1]:.2f}]")
        print()

    # Load test puzzles for visualization
    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)

    # Visualize top 3 boundary pairs
    print(f"\n🎨 Visualizing top 3 boundary pairs...")
    for pair_idx, pair in enumerate(unique_pairs[:3]):
        solved_name = pair['solved_name']
        unsolved_name = pair['unsolved_name']

        solved_data = test_puzzles.get(solved_name)
        unsolved_data = test_puzzles.get(unsolved_name)

        if not solved_data or not unsolved_data:
            print(f"⚠️  Skipping pair {pair_idx+1} - data not found")
            continue

        # Get first train example
        ex_solved = solved_data['train'][0] if solved_data['train'] else solved_data['test'][0]
        ex_unsolved = unsolved_data['train'][0] if unsolved_data['train'] else unsolved_data['test'][0]

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Solved puzzle
        axes[0, 0].imshow(ex_solved['input'], cmap='tab10', vmin=0, vmax=9)
        axes[0, 0].set_title(f"SOLVED: {solved_name}\nInput", fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(ex_solved['output'], cmap='tab10', vmin=0, vmax=9)
        axes[0, 1].set_title(f"Output", fontsize=10)
        axes[0, 1].axis('off')

        # Unsolved puzzle
        axes[1, 0].imshow(ex_unsolved['input'], cmap='tab10', vmin=0, vmax=9)
        axes[1, 0].set_title(f"UNSOLVED: {unsolved_name}\nInput", fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ex_unsolved['output'], cmap='tab10', vmin=0, vmax=9)
        axes[1, 1].set_title(f"Output", fontsize=10)
        axes[1, 1].axis('off')

        fig.suptitle(f'Decision Boundary Pair #{pair_idx+1} (Distance: {pair["distance"]:.3f})\\n' +
                     f'Latent positions: Solved [{pair["pos_solved"][0]:.2f}, {pair["pos_solved"][1]:.2f}] | ' +
                     f'Unsolved [{pair["pos_unsolved"][0]:.2f}, {pair["pos_unsolved"][1]:.2f}]',
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        output_path = Path(output_dir) / f'boundary_pair_{pair_idx+1}_{solved_name[:8]}_{unsolved_name[:8]}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Saved pair {pair_idx+1} to {output_path}")
        plt.close()

    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
