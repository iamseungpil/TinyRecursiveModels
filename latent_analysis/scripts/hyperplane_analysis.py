"""
Hyperplane-based analysis of 400 validation puzzles.

This script:
1. Learns a hyperplane in 512D space to separate solved/unsolved
2. Finds three groups:
   - Solved-Solved similar pairs
   - Unsolved-Unsolved similar pairs
   - Boundary pairs (closest across decision boundary)
3. Visualizes PCA with hyperplane
4. Shows examples from each group
"""

import sys
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path


def main():
    data_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/validation_400_puzzles.json"
    test_puzzles_path = "/data/arc1concept-aug-1000/test_puzzles.json"
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Hyperplane-Based Decision Boundary Analysis")
    print("="*70)

    # Load data
    print(f"\n📥 Loading data...")
    with open(data_path, 'r') as f:
        results = json.load(f)

    latents = np.array([r['latent'] for r in results])
    solved = np.array([r['solved'] for r in results])
    puzzle_names = [r['puzzle_name'] for r in results]

    print(f"✅ Loaded {len(results)} puzzles")
    print(f"   Solved: {solved.sum()} ({solved.sum()/len(results)*100:.1f}%)")
    print(f"   Unsolved: {(~solved).sum()} ({(~solved).sum()/len(results)*100:.1f}%)")

    # Step 1: Learn Hyperplane in 512D
    print(f"\n🔬 Learning hyperplane in 512D space...")
    svm = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    svm.fit(latents, solved)

    w = svm.coef_[0]  # Weight vector [512]
    b = svm.intercept_[0]  # Bias

    print(f"✅ Hyperplane learned")
    print(f"   Training accuracy: {svm.score(latents, solved):.3f}")

    # Step 2: Compute distances to hyperplane
    print(f"\n📏 Computing distances to hyperplane...")
    distances = svm.decision_function(latents)
    abs_distances = np.abs(distances)

    print(f"✅ Distance statistics:")
    print(f"   Solved side (distance > 0): {(distances > 0).sum()}")
    print(f"   Unsolved side (distance < 0): {(distances < 0).sum()}")
    print(f"   Min distance: {abs_distances.min():.3f}")
    print(f"   Max distance: {abs_distances.max():.3f}")
    print(f"   Median distance: {np.median(abs_distances):.3f}")

    # Step 3: Find three groups
    print(f"\n🔍 Finding similar pairs in three groups...")

    solved_indices = np.where(solved)[0]
    unsolved_indices = np.where(~solved)[0]

    # Group 1: Solved-Solved pairs
    print(f"\n   Group 1: Solved-Solved similar pairs")
    solved_latents = latents[solved_indices]
    solved_distances = euclidean_distances(solved_latents)

    solved_pairs = []
    for i in range(len(solved_indices)):
        for j in range(i+1, len(solved_indices)):
            solved_pairs.append({
                'idx_i': solved_indices[i],
                'idx_j': solved_indices[j],
                'name_i': puzzle_names[solved_indices[i]],
                'name_j': puzzle_names[solved_indices[j]],
                'distance': solved_distances[i, j],
                'hyperplane_dist_i': distances[solved_indices[i]],
                'hyperplane_dist_j': distances[solved_indices[j]]
            })
    solved_pairs.sort(key=lambda x: x['distance'])
    print(f"   Found {len(solved_pairs)} solved-solved pairs")

    # Group 2: Unsolved-Unsolved pairs
    print(f"\n   Group 2: Unsolved-Unsolved similar pairs")
    unsolved_latents = latents[unsolved_indices]
    unsolved_distances = euclidean_distances(unsolved_latents)

    unsolved_pairs = []
    for i in range(len(unsolved_indices)):
        for j in range(i+1, len(unsolved_indices)):
            unsolved_pairs.append({
                'idx_i': unsolved_indices[i],
                'idx_j': unsolved_indices[j],
                'name_i': puzzle_names[unsolved_indices[i]],
                'name_j': puzzle_names[unsolved_indices[j]],
                'distance': unsolved_distances[i, j],
                'hyperplane_dist_i': distances[unsolved_indices[i]],
                'hyperplane_dist_j': distances[unsolved_indices[j]]
            })
    unsolved_pairs.sort(key=lambda x: x['distance'])
    print(f"   Found {len(unsolved_pairs)} unsolved-unsolved pairs")

    # Group 3: Boundary pairs (closest across hyperplane)
    print(f"\n   Group 3: Boundary pairs (solved-unsolved near hyperplane)")

    # Get puzzles near boundary (top 30 from each side)
    boundary_solved_idx = solved_indices[np.argsort(abs_distances[solved_indices])[:30]]
    boundary_unsolved_idx = unsolved_indices[np.argsort(abs_distances[unsolved_indices])[:30]]

    boundary_pairs = []
    for i in boundary_solved_idx:
        for j in boundary_unsolved_idx:
            dist = np.linalg.norm(latents[i] - latents[j])
            boundary_pairs.append({
                'idx_i': i,
                'idx_j': j,
                'name_i': puzzle_names[i],
                'name_j': puzzle_names[j],
                'distance': dist,
                'hyperplane_dist_i': distances[i],
                'hyperplane_dist_j': distances[j],
                'boundary_gap': abs(distances[i]) + abs(distances[j])  # Total distance across boundary
            })

    # Sort by distance in latent space
    boundary_pairs.sort(key=lambda x: x['distance'])
    print(f"   Found {len(boundary_pairs)} boundary pairs")

    # Print top 5 from each group
    print(f"\n📍 Top 5 closest pairs from each group:")

    print(f"\n   Solved-Solved:")
    for idx, pair in enumerate(solved_pairs[:5], 1):
        print(f"   {idx}. {pair['name_i']} ↔ {pair['name_j']}")
        print(f"      Distance: {pair['distance']:.3f}")

    print(f"\n   Unsolved-Unsolved:")
    for idx, pair in enumerate(unsolved_pairs[:5], 1):
        print(f"   {idx}. {pair['name_i']} ↔ {pair['name_j']}")
        print(f"      Distance: {pair['distance']:.3f}")

    print(f"\n   Boundary Pairs:")
    for idx, pair in enumerate(boundary_pairs[:5], 1):
        print(f"   {idx}. {pair['name_i']} (solved) ↔ {pair['name_j']} (unsolved)")
        print(f"      Distance: {pair['distance']:.3f}")
        print(f"      Hyperplane dists: {pair['hyperplane_dist_i']:.3f} | {pair['hyperplane_dist_j']:.3f}")

    # Step 4: PCA with Hyperplane
    print(f"\n🎨 Creating PCA visualization with hyperplane...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Project hyperplane to 2D
    w_pca = pca.components_ @ w  # Project weight vector
    # Hyperplane in 2D: w_pca[0]*x + w_pca[1]*y + b_projected = 0

    # To draw the line, we need to adjust bias for PCA transformation
    # Since PCA centers data, we need to find where the hyperplane intersects the PCA space
    center = pca.mean_
    b_pca = b + np.dot(w, center)  # Adjust bias for centered data

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: Standard PCA with hyperplane
    ax1 = axes[0]
    ax1.scatter(latents_2d[~solved, 0], latents_2d[~solved, 1],
                c='red', alpha=0.6, s=60, label=f'Unsolved ({(~solved).sum()})',
                edgecolors='darkred', linewidths=0.8)
    ax1.scatter(latents_2d[solved, 0], latents_2d[solved, 1],
                c='green', alpha=0.6, s=60, label=f'Solved ({solved.sum()})',
                edgecolors='darkgreen', linewidths=0.8)

    # Draw hyperplane
    xlim = ax1.get_xlim()
    x_line = np.linspace(xlim[0], xlim[1], 100)
    if abs(w_pca[1]) > 1e-6:
        y_line = -(w_pca[0] * x_line + b_pca) / w_pca[1]
        ax1.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Hyperplane', alpha=0.8)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax1.set_title('PCA with Decision Hyperplane', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance to hyperplane (heatmap)
    ax2 = axes[1]
    scatter = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1],
                         c=distances, cmap='RdBu_r', s=60, alpha=0.7,
                         edgecolors='black', linewidths=0.5,
                         vmin=-np.percentile(abs_distances, 95),
                         vmax=np.percentile(abs_distances, 95))

    # Draw hyperplane
    if abs(w_pca[1]) > 1e-6:
        ax2.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Distance to Hyperplane\n(+ = Solved, - = Unsolved)', fontsize=10)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax2.set_title('Signed Distance to Hyperplane', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Boundary region (zoom)
    ax3 = axes[2]

    # Only show puzzles near boundary (top 50 from each side)
    near_boundary_solved = solved_indices[np.argsort(abs_distances[solved_indices])[:50]]
    near_boundary_unsolved = unsolved_indices[np.argsort(abs_distances[unsolved_indices])[:50]]

    ax3.scatter(latents_2d[near_boundary_unsolved, 0], latents_2d[near_boundary_unsolved, 1],
                c='red', alpha=0.7, s=100, label='Unsolved (near boundary)',
                edgecolors='darkred', linewidths=1)
    ax3.scatter(latents_2d[near_boundary_solved, 0], latents_2d[near_boundary_solved, 1],
                c='green', alpha=0.7, s=100, label='Solved (near boundary)',
                edgecolors='darkgreen', linewidths=1)

    # Draw hyperplane
    if abs(w_pca[1]) > 1e-6:
        ax3.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8, label='Hyperplane')

    # Mark top 3 boundary pairs
    for idx, pair in enumerate(boundary_pairs[:3]):
        i, j = pair['idx_i'], pair['idx_j']
        ax3.plot([latents_2d[i, 0], latents_2d[j, 0]],
                [latents_2d[i, 1], latents_2d[j, 1]],
                'purple', linewidth=2, alpha=0.6)
        mid_x = (latents_2d[i, 0] + latents_2d[j, 0]) / 2
        mid_y = (latents_2d[i, 1] + latents_2d[j, 1]) / 2
        ax3.text(mid_x, mid_y, f'{idx+1}', fontsize=10, fontweight='bold',
                color='purple', ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax3.set_title('Boundary Region (Top 50 each side)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'hyperplane_pca_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to {output_path}")
    plt.close()

    # Step 5: Visualize examples from each group
    print(f"\n🎨 Visualizing examples from each group...")

    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)

    # Load predictions from validation data
    predictions_map = {r['puzzle_name']: np.array(r['prediction']) for r in results}

    # Visualize top 3 from each group
    for group_name, pairs_list in [
        ('solved_solved', solved_pairs[:3]),
        ('unsolved_unsolved', unsolved_pairs[:3]),
        ('boundary', boundary_pairs[:3])
    ]:
        for pair_idx, pair in enumerate(pairs_list):
            name_i = pair['name_i']
            name_j = pair['name_j']

            data_i = test_puzzles.get(name_i)
            data_j = test_puzzles.get(name_j)

            if not data_i or not data_j:
                print(f"⚠️  Skipping {name_i} or {name_j}")
                continue

            # Get first train example for input/output
            ex_i = data_i['train'][0] if data_i['train'] else data_i['test'][0]
            ex_j = data_j['train'][0] if data_j['train'] else data_j['test'][0]

            # Determine if solved or unsolved
            status_i = "SOLVED" if name_i in [r['puzzle_name'] for r in results if r['solved']] else "UNSOLVED"
            status_j = "SOLVED" if name_j in [r['puzzle_name'] for r in results if r['solved']] else "UNSOLVED"

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Puzzle i
            axes[0, 0].imshow(ex_i['input'], cmap='tab10', vmin=0, vmax=9)
            color_i = 'green' if status_i == "SOLVED" else 'red'
            axes[0, 0].set_title(f"{status_i}: {name_i}\nInput", fontsize=10, fontweight='bold', color=color_i)
            axes[0, 0].axis('off')

            # Output for puzzle i: Ground Truth if solved, Model Prediction if unsolved
            if status_i == "SOLVED":
                axes[0, 1].imshow(ex_i['output'], cmap='tab10', vmin=0, vmax=9)
                axes[0, 1].set_title(f"Ground Truth", fontsize=10, color='green', fontweight='bold')
            else:
                pred_i = predictions_map.get(name_i)
                if pred_i is not None:
                    axes[0, 1].imshow(pred_i, cmap='tab10', vmin=0, vmax=9)
                    axes[0, 1].set_title(f"Model Prediction (Wrong)", fontsize=10, color='red', fontweight='bold')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No prediction', ha='center', va='center')
                    axes[0, 1].set_title(f"Model Prediction", fontsize=10)
            axes[0, 1].axis('off')

            # Puzzle j
            axes[1, 0].imshow(ex_j['input'], cmap='tab10', vmin=0, vmax=9)
            color_j = 'green' if status_j == "SOLVED" else 'red'
            axes[1, 0].set_title(f"{status_j}: {name_j}\nInput", fontsize=10, fontweight='bold', color=color_j)
            axes[1, 0].axis('off')

            # Output for puzzle j: Ground Truth if solved, Model Prediction if unsolved
            if status_j == "SOLVED":
                axes[1, 1].imshow(ex_j['output'], cmap='tab10', vmin=0, vmax=9)
                axes[1, 1].set_title(f"Ground Truth", fontsize=10, color='green', fontweight='bold')
            else:
                pred_j = predictions_map.get(name_j)
                if pred_j is not None:
                    axes[1, 1].imshow(pred_j, cmap='tab10', vmin=0, vmax=9)
                    axes[1, 1].set_title(f"Model Prediction (Wrong)", fontsize=10, color='red', fontweight='bold')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No prediction', ha='center', va='center')
                    axes[1, 1].set_title(f"Model Prediction", fontsize=10)
            axes[1, 1].axis('off')

            title = f'{group_name.upper()} Pair #{pair_idx+1} (Distance: {pair["distance"]:.3f})'
            if group_name == 'boundary':
                title += f'\nHyperplane: {pair["hyperplane_dist_i"]:.3f} | {pair["hyperplane_dist_j"]:.3f}'

            fig.suptitle(title, fontsize=13, fontweight='bold')

            plt.tight_layout()
            output_path = Path(output_dir) / f'{group_name}_pair_{pair_idx+1}_{name_i[:8]}_{name_j[:8]}.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"💾 Saved {group_name} pair {pair_idx+1}")
            plt.close()

    print(f"\n{'='*70}")
    print("✅ Hyperplane analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
