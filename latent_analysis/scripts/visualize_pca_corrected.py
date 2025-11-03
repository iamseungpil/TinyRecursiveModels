"""
PCA visualization of TRM latents with CORRECTED configuration.

This uses latents extracted with:
- Real puzzle IDs (not zeros)
- Augmented .npy data
- Correct L_cycles=6
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def load_latents(data_path: str):
    """Load latent vectors and metadata."""
    print(f"📥 Loading data from {data_path}...")

    with open(data_path, 'r') as f:
        results = json.load(f)

    # Extract latents and labels
    latents = np.array([r['latent'] for r in results])
    solved = np.array([r['solved'] for r in results])
    puzzle_ids = np.array([r['puzzle_id'] for r in results])

    print(f"✅ Loaded:")
    print(f"   Samples: {len(results)}")
    print(f"   Latent dimension: {latents.shape[1]}")
    print(f"   Solved: {solved.sum()} ({solved.sum()/len(solved)*100:.1f}%)")
    print(f"   Unsolved: {(~solved).sum()} ({(~solved).sum()/len(solved)*100:.1f}%)")

    return latents, solved, puzzle_ids


def run_pca_analysis(latents, solved, output_dir: str):
    """Run PCA and create visualizations."""
    print(f"\n🔬 Running PCA analysis...")

    # Run PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    print(f"✅ PCA complete:")
    print(f"   Explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    print(f"   Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by solved/unsolved
    ax1 = axes[0]
    colors = ['red' if not s else 'green' for s in solved]
    ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, alpha=0.6, s=20)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('TRM Latent Space (Corrected Configuration)\nColored by Solved/Unsolved')
    ax1.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label=f'Solved ({solved.sum()})'),
        Patch(facecolor='red', alpha=0.6, label=f'Unsolved ({(~solved).sum()})')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Plot 2: Density plot
    ax2 = axes[1]
    ax2.hexbin(latents_2d[:, 0], latents_2d[:, 1], gridsize=30, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('TRM Latent Space Density')
    ax2.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    output_path = Path(output_dir) / 'pca_visualization_corrected.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization to {output_path}")

    # Also save a separate comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Separate solved/unsolved
    solved_mask = solved
    unsolved_mask = ~solved

    ax.scatter(latents_2d[unsolved_mask, 0], latents_2d[unsolved_mask, 1],
               c='red', alpha=0.6, s=30, label=f'Unsolved ({unsolved_mask.sum()})')
    ax.scatter(latents_2d[solved_mask, 0], latents_2d[solved_mask, 1],
               c='green', alpha=0.6, s=30, label=f'Solved ({solved_mask.sum()})')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('TRM Latent Space - Corrected Configuration\n(Real IDs + Augmented Data + L_cycles=6)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = Path(output_dir) / 'pca_comparison_corrected.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comparison to {output_path2}")

    return latents_2d, pca


def analyze_clusters(latents_2d, solved):
    """Analyze spatial clustering of solved vs unsolved."""
    print(f"\n📊 Cluster Analysis:")

    solved_coords = latents_2d[solved]
    unsolved_coords = latents_2d[~solved]

    print(f"\nSolved examples:")
    print(f"   Mean: [{solved_coords[:, 0].mean():.3f}, {solved_coords[:, 1].mean():.3f}]")
    print(f"   Std:  [{solved_coords[:, 0].std():.3f}, {solved_coords[:, 1].std():.3f}]")

    print(f"\nUnsolved examples:")
    print(f"   Mean: [{unsolved_coords[:, 0].mean():.3f}, {unsolved_coords[:, 1].mean():.3f}]")
    print(f"   Std:  [{unsolved_coords[:, 0].std():.3f}, {unsolved_coords[:, 1].std():.3f}]")

    # Compute distance between centroids
    solved_centroid = solved_coords.mean(axis=0)
    unsolved_centroid = unsolved_coords.mean(axis=0)
    distance = np.linalg.norm(solved_centroid - unsolved_centroid)

    print(f"\n🎯 Separation:")
    print(f"   Distance between centroids: {distance:.3f}")

    # Compute overlap
    from scipy.spatial.distance import cdist

    # Sample for efficiency
    if len(solved_coords) > 100:
        solved_sample = solved_coords[np.random.choice(len(solved_coords), 100, replace=False)]
    else:
        solved_sample = solved_coords

    if len(unsolved_coords) > 100:
        unsolved_sample = unsolved_coords[np.random.choice(len(unsolved_coords), 100, replace=False)]
    else:
        unsolved_sample = unsolved_coords

    # Compute average nearest neighbor distance
    if len(unsolved_sample) > 0:
        dists = cdist(solved_sample, unsolved_sample)
        min_dists = dists.min(axis=1)
        avg_nearest = min_dists.mean()
        print(f"   Avg nearest neighbor distance (solved→unsolved): {avg_nearest:.3f}")


def main():
    # Paths
    data_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/quick_test_results.json"
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TRM Latent PCA Visualization - CORRECTED")
    print("="*70)
    print("Configuration:")
    print("  ✅ Real puzzle IDs (362365-730514)")
    print("  ✅ Augmented .npy data")
    print("  ✅ L_cycles=6")
    print("="*70)

    # Load data
    latents, solved, puzzle_ids = load_latents(data_path)

    # Run PCA
    latents_2d, pca = run_pca_analysis(latents, solved, output_dir)

    # Analyze clusters
    analyze_clusters(latents_2d, solved)

    print(f"\n{'='*70}")
    print("✅ Visualization complete!")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  - {output_dir}/pca_visualization_corrected.png")
    print(f"  - {output_dir}/pca_comparison_corrected.png")


if __name__ == "__main__":
    main()
