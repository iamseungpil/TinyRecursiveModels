"""
Analyze unique puzzle representations (removing augmentation effects).

This script:
1. Groups augmented versions of same puzzle
2. Computes average representation per unique puzzle
3. Runs PCA on unique puzzles
4. Finds nearest neighbors in latent space
5. Visualizes similar puzzles
"""

import sys
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
from dataset.build_arc_dataset import inverse_aug, PuzzleIdSeparator


def load_and_group_by_unique_puzzle(data_path: str, identifiers_path: str):
    """
    Load latents and group by unique puzzle (removing augmentation).

    Returns:
        unique_puzzles: dict mapping original_puzzle_id -> {
            'latents': list of latent vectors,
            'solved': list of bool,
            'puzzle_ids': list of augmented IDs
        }
    """
    print(f"📥 Loading data...")

    with open(data_path, 'r') as f:
        results = json.load(f)

    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
        identifier_map = {i: name for i, name in enumerate(identifier_list)}

    print(f"✅ Loaded {len(results)} samples")

    # Group by original puzzle
    unique_puzzles = {}

    for r in results:
        puzzle_id = r['puzzle_id']
        puzzle_name = identifier_map.get(puzzle_id, f"<unknown_{puzzle_id}>")

        # Remove augmentation to get original name
        original_name, _ = inverse_aug(puzzle_name)

        if original_name not in unique_puzzles:
            unique_puzzles[original_name] = {
                'latents': [],
                'solved': [],
                'puzzle_ids': []
            }

        unique_puzzles[original_name]['latents'].append(r['latent'])
        unique_puzzles[original_name]['solved'].append(r['solved'])
        unique_puzzles[original_name]['puzzle_ids'].append(puzzle_id)

    print(f"✅ Found {len(unique_puzzles)} unique puzzles")

    # Compute statistics per unique puzzle
    unique_puzzle_data = []
    for orig_name, data in unique_puzzles.items():
        latents = np.array(data['latents'])
        solved = np.array(data['solved'])

        # Average latent representation
        avg_latent = latents.mean(axis=0)

        # Majority vote for solved
        majority_solved = solved.mean() > 0.5
        solve_rate = solved.mean()

        unique_puzzle_data.append({
            'original_name': orig_name,
            'avg_latent': avg_latent,
            'solved': majority_solved,
            'solve_rate': solve_rate,
            'num_augmentations': len(latents),
            'latent_std': latents.std(axis=0).mean()  # Average std across dimensions
        })

    print(f"\n📊 Unique puzzle statistics:")
    solve_rates = [p['solve_rate'] for p in unique_puzzle_data]
    print(f"   Solve rate distribution:")
    print(f"     - Mean: {np.mean(solve_rates):.3f}")
    print(f"     - Median: {np.median(solve_rates):.3f}")
    print(f"     - Std: {np.std(solve_rates):.3f}")

    solved_count = sum(1 for p in unique_puzzle_data if p['solved'])
    print(f"   Solved puzzles: {solved_count}/{len(unique_puzzle_data)} ({solved_count/len(unique_puzzle_data)*100:.1f}%)")

    return unique_puzzle_data


def run_pca_unique_puzzles(puzzle_data, output_dir: str):
    """Run PCA on unique puzzle representations."""
    print(f"\n🔬 Running PCA on unique puzzles...")

    # Extract latents and labels
    latents = np.array([p['avg_latent'] for p in puzzle_data])
    solved = np.array([p['solved'] for p in puzzle_data])
    solve_rates = np.array([p['solve_rate'] for p in puzzle_data])

    # Run PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    print(f"✅ PCA complete:")
    print(f"   Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"   Total: {pca.explained_variance_ratio_.sum():.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Binary solved/unsolved
    ax1 = axes[0]
    solved_mask = solved
    unsolved_mask = ~solved

    ax1.scatter(latents_2d[unsolved_mask, 0], latents_2d[unsolved_mask, 1],
                c='red', alpha=0.7, s=80, label=f'Unsolved ({unsolved_mask.sum()})', edgecolors='darkred', linewidths=1)
    ax1.scatter(latents_2d[solved_mask, 0], latents_2d[solved_mask, 1],
                c='green', alpha=0.7, s=80, label=f'Solved ({solved_mask.sum()})', edgecolors='darkgreen', linewidths=1)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('Unique Puzzle Latent Space\n(Averaged over augmentations)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Continuous solve rate
    ax2 = axes[1]
    scatter = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1],
                         c=solve_rates, cmap='RdYlGn', alpha=0.8, s=80,
                         vmin=0, vmax=1, edgecolors='black', linewidths=0.5)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Solve Rate', fontsize=11)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax2.set_title('Unique Puzzle Solve Rate Heatmap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'pca_unique_puzzles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to {output_path}")

    # Compute separation metrics
    if unsolved_mask.sum() > 0 and solved_mask.sum() > 0:
        solved_centroid = latents_2d[solved_mask].mean(axis=0)
        unsolved_centroid = latents_2d[unsolved_mask].mean(axis=0)
        separation = np.linalg.norm(solved_centroid - unsolved_centroid)

        print(f"\n🎯 Cluster Separation:")
        print(f"   Solved centroid: [{solved_centroid[0]:.3f}, {solved_centroid[1]:.3f}]")
        print(f"   Unsolved centroid: [{unsolved_centroid[0]:.3f}, {unsolved_centroid[1]:.3f}]")
        print(f"   Distance: {separation:.3f}")

    return latents_2d, latents, pca


def find_nearest_neighbors(puzzle_data, latents, latents_2d, k=5):
    """
    Find nearest neighbors in latent space.

    Returns pairs of (puzzle_i, puzzle_j, distance)
    """
    print(f"\n🔍 Finding nearest neighbors...")

    # Compute pairwise distances
    distances = euclidean_distances(latents)

    # For each puzzle, find nearest different puzzle
    neighbor_pairs = []

    for i in range(len(puzzle_data)):
        # Get distances to all other puzzles
        dists = distances[i].copy()
        dists[i] = np.inf  # Exclude self

        # Find k nearest
        nearest_indices = np.argsort(dists)[:k]

        for j in nearest_indices:
            if i < j:  # Avoid duplicates
                neighbor_pairs.append({
                    'puzzle_i': puzzle_data[i]['original_name'],
                    'puzzle_j': puzzle_data[j]['original_name'],
                    'distance': dists[j],
                    'solved_i': puzzle_data[i]['solved'],
                    'solved_j': puzzle_data[j]['solved'],
                    'solve_rate_i': puzzle_data[i]['solve_rate'],
                    'solve_rate_j': puzzle_data[j]['solve_rate'],
                    'pos_i': latents_2d[i],
                    'pos_j': latents_2d[j]
                })

    # Sort by distance
    neighbor_pairs.sort(key=lambda x: x['distance'])

    print(f"✅ Found {len(neighbor_pairs)} neighbor pairs")

    # Print top 10 closest pairs
    print(f"\n📍 Top 10 closest pairs:")
    for idx, pair in enumerate(neighbor_pairs[:10], 1):
        print(f"{idx}. {pair['puzzle_i']} ↔ {pair['puzzle_j']}")
        print(f"   Distance: {pair['distance']:.3f}")
        print(f"   Solved: {pair['solved_i']} ({pair['solve_rate_i']:.1%}) ↔ {pair['solved_j']} ({pair['solve_rate_j']:.1%})")
        print()

    return neighbor_pairs


def visualize_similar_puzzles(neighbor_pairs, test_puzzles_path: str, output_dir: str, num_pairs=3):
    """
    Visualize input/output grids of similar puzzles.
    """
    print(f"\n🎨 Visualizing similar puzzle pairs...")

    # Load test puzzles
    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)

    # Select interesting pairs (mix of same/different solve status)
    same_status_pairs = [p for p in neighbor_pairs if p['solved_i'] == p['solved_j']][:num_pairs]
    diff_status_pairs = [p for p in neighbor_pairs if p['solved_i'] != p['solved_j']][:num_pairs]

    selected_pairs = same_status_pairs + diff_status_pairs

    for pair_idx, pair in enumerate(selected_pairs):
        puzzle_i_name = pair['puzzle_i']
        puzzle_j_name = pair['puzzle_j']

        # Find puzzles in test_puzzles
        puzzle_i_data = test_puzzles.get(puzzle_i_name)
        puzzle_j_data = test_puzzles.get(puzzle_j_name)

        if not puzzle_i_data or not puzzle_j_data:
            print(f"⚠️  Skipping {puzzle_i_name} or {puzzle_j_name} - not in test set")
            continue

        # Get first train example from each
        example_i = puzzle_i_data['train'][0] if puzzle_i_data['train'] else puzzle_i_data['test'][0]
        example_j = puzzle_j_data['train'][0] if puzzle_j_data['train'] else puzzle_j_data['test'][0]

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Puzzle i
        axes[0, 0].imshow(example_i['input'], cmap='tab10', vmin=0, vmax=9)
        axes[0, 0].set_title(f"Puzzle {puzzle_i_name}\nInput", fontsize=10)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(example_i['output'], cmap='tab10', vmin=0, vmax=9)
        axes[0, 1].set_title(f"Output\nSolved: {pair['solved_i']} ({pair['solve_rate_i']:.0%})", fontsize=10)
        axes[0, 1].axis('off')

        # Puzzle j
        axes[1, 0].imshow(example_j['input'], cmap='tab10', vmin=0, vmax=9)
        axes[1, 0].set_title(f"Puzzle {puzzle_j_name}\nInput", fontsize=10)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(example_j['output'], cmap='tab10', vmin=0, vmax=9)
        axes[1, 1].set_title(f"Output\nSolved: {pair['solved_j']} ({pair['solve_rate_j']:.0%})", fontsize=10)
        axes[1, 1].axis('off')

        # Add more examples if available
        for col, ex_idx in enumerate([1, 2], start=2):
            if ex_idx < len(puzzle_i_data['train']):
                ex = puzzle_i_data['train'][ex_idx]
                axes[0, col].imshow(ex['input'], cmap='tab10', vmin=0, vmax=9)
                axes[0, col].set_title(f"Train {ex_idx}", fontsize=9)
                axes[0, col].axis('off')
            else:
                axes[0, col].axis('off')

            if ex_idx < len(puzzle_j_data['train']):
                ex = puzzle_j_data['train'][ex_idx]
                axes[1, col].imshow(ex['input'], cmap='tab10', vmin=0, vmax=9)
                axes[1, col].set_title(f"Train {ex_idx}", fontsize=9)
                axes[1, col].axis('off')
            else:
                axes[1, col].axis('off')

        fig.suptitle(f'Similar Puzzles in Latent Space (Distance: {pair["distance"]:.3f})\n'
                     f'Position i: [{pair["pos_i"][0]:.2f}, {pair["pos_i"][1]:.2f}] | '
                     f'Position j: [{pair["pos_j"][0]:.2f}, {pair["pos_j"][1]:.2f}]',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = Path(output_dir) / f'similar_pair_{pair_idx+1}_{puzzle_i_name[:8]}_{puzzle_j_name[:8]}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Saved pair {pair_idx+1} to {output_path}")
        plt.close()


def main():
    # Paths
    data_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/quick_test_results.json"
    identifiers_path = "/data/arc1concept-aug-1000/identifiers.json"
    test_puzzles_path = "/data/arc1concept-aug-1000/test_puzzles.json"
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Unique Puzzle Analysis - Removing Augmentation Effects")
    print("="*70)

    # Step 1: Group by unique puzzle
    puzzle_data = load_and_group_by_unique_puzzle(data_path, identifiers_path)

    # Step 2: PCA on unique puzzles
    latents_2d, latents, pca = run_pca_unique_puzzles(puzzle_data, output_dir)

    # Step 3: Find nearest neighbors
    neighbor_pairs = find_nearest_neighbors(puzzle_data, latents, latents_2d, k=5)

    # Step 4: Visualize similar puzzles
    visualize_similar_puzzles(neighbor_pairs, test_puzzles_path, output_dir, num_pairs=3)

    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}")
    print(f"\nKey insights:")
    print(f"1. Unique puzzle count: {len(puzzle_data)}")
    print(f"2. Solve rate: {np.mean([p['solve_rate'] for p in puzzle_data]):.1%}")
    print(f"3. Nearest neighbors found: {len(neighbor_pairs)}")
    print(f"\nFiles generated:")
    print(f"  - pca_unique_puzzles.png")
    print(f"  - similar_pair_*.png (visualization of close puzzles)")


if __name__ == "__main__":
    main()
