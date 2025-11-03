"""
Visualize TRM latents using PCA and find nearest neighbor pairs.

Reads latents.json and creates:
1. PCA 2D scatter plot (solved vs unsolved)
2. Finds 2 nearest neighbor pairs
3. Visualizes the grids for these pairs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def load_latents(latents_path: str):
    """Load latents from JSON file."""
    print(f"📥 Loading latents from {latents_path}...")
    with open(latents_path, 'r') as f:
        data = json.load(f)

    latents = np.array([d['latent'] for d in data])  # [N, 512]
    solved_labels = np.array([d['solved'] for d in data])  # [N]

    print(f"✅ Loaded {len(latents)} latents")
    print(f"  Solved: {solved_labels.sum()} ({solved_labels.sum()/len(solved_labels)*100:.1f}%)")
    print(f"  Unsolved: {(~solved_labels).sum()} ({(~solved_labels).sum()/len(solved_labels)*100:.1f}%)")

    return data, latents, solved_labels


def perform_pca(latents: np.ndarray):
    """Perform PCA to reduce 512D to 2D."""
    print(f"\n🔬 Performing PCA: {latents.shape[1]}D → 2D...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    print(f"✅ PCA completed")
    print(f"  Explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%, {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return latents_2d, pca


def plot_pca_scatter(latents_2d: np.ndarray, solved_labels: np.ndarray, output_path: str):
    """Create PCA scatter plot with solved/unsolved coloring."""
    print(f"\n📊 Creating PCA scatter plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot unsolved (red)
    unsolved_mask = ~solved_labels
    ax.scatter(
        latents_2d[unsolved_mask, 0],
        latents_2d[unsolved_mask, 1],
        c='red',
        alpha=0.6,
        s=30,
        label=f'Unsolved ({unsolved_mask.sum()})',
        edgecolors='darkred',
        linewidth=0.5
    )

    # Plot solved (green)
    solved_mask = solved_labels
    if solved_mask.sum() > 0:
        ax.scatter(
            latents_2d[solved_mask, 0],
            latents_2d[solved_mask, 1],
            c='green',
            alpha=0.6,
            s=30,
            label=f'Solved ({solved_mask.sum()})',
            edgecolors='darkgreen',
            linewidth=0.5
        )

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('TRM Latent Space (PCA 2D Projection)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved scatter plot to {output_path}")


def group_by_task(data):
    """Group problems by task_id (base task without train/test suffix)."""
    from collections import defaultdict
    tasks = defaultdict(list)
    for i, item in enumerate(data):
        # Extract base task_id (remove _train_X or _test_X)
        parts = item['task_id'].split('_')
        # Task ID is everything except last 2 parts (train/test and index)
        task_id = '_'.join(parts[:-2])
        tasks[task_id].append(i)
    return tasks


def find_nearest_tasks(data, latents: np.ndarray, n_tasks: int = 2):
    """Find N nearest neighbor task pairs by centroid distance."""
    print(f"\n🔍 Finding {n_tasks} nearest task pairs...")

    # Group by task
    tasks = group_by_task(data)
    print(f"  Total tasks: {len(tasks)}")

    # Compute task centroids (mean of all problems in task)
    task_centroids = {}
    for task_id, indices in tasks.items():
        task_latents = latents[indices]
        task_centroids[task_id] = task_latents.mean(axis=0)

    # Find nearest task pairs
    task_ids = list(task_centroids.keys())
    centroids = np.array([task_centroids[tid] for tid in task_ids])

    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nbrs.fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)

    # Get all task pairs with distances
    pairs = []
    for i in range(len(centroids)):
        j = indices[i, 1]  # Skip self (index 0)
        distance = distances[i, 1]

        # Only add pair once (smaller index first)
        if i < j:
            pairs.append((task_ids[i], task_ids[j], distance))

    # Sort by distance and get top N
    pairs = sorted(pairs, key=lambda x: x[2])[:n_tasks]

    print(f"✅ Found {n_tasks} nearest task pairs:")
    for idx, (task1, task2, dist) in enumerate(pairs):
        n_problems_1 = len(tasks[task1])
        n_problems_2 = len(tasks[task2])
        print(f"  Pair {idx+1}: {task1} ({n_problems_1} problems) ↔ {task2} ({n_problems_2} problems)")
        print(f"           Distance: {dist:.4f}")

    return pairs, tasks


def _crop(grid):
    """Official ARC evaluator crop function."""
    grid = grid.reshape(30, 30)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


def visualize_single_task(data, task_id, tasks, output_path: str):
    """Visualize all problems from a single task."""
    # ARC color palette
    colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    task_indices = tasks[task_id]
    solved_count = sum(1 for idx in task_indices if data[idx]['solved'])

    # Show up to 6 problems (2 rows x 3 columns)
    n_problems = min(6, len(task_indices))
    n_cols = 3
    n_rows = (n_problems + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Task: {task_id} ({solved_count}/{len(task_indices)} solved)',
                 fontsize=14, fontweight='bold')

    for idx in range(n_problems):
        row = idx // n_cols
        col_offset = (idx % n_cols) * 3

        item = data[task_indices[idx]]

        # Input, Output, Pred
        for sub_col, (grid_data, title) in enumerate([
            (item['input_grid'], 'Input'),
            (item['output_grid'], 'Target'),
            (item['pred_grid'], f"Pred ({'✓' if item['solved'] else '✗'})")
        ]):
            ax = axes[row, col_offset + sub_col]

            # Apply _crop to pred_grid to get correct values
            if title.startswith('Pred'):
                grid = _crop(np.array(grid_data).flatten())
            else:
                grid = np.array(grid_data)

            # Create colored grid
            colored_grid = np.zeros((*grid.shape, 3))
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    color_idx = grid[i, j] % 10
                    hex_color = colors[color_idx]
                    colored_grid[i, j] = [int(hex_color[k:k+2], 16)/255 for k in (1, 3, 5)]

            ax.imshow(colored_grid, interpolation='nearest')
            ax.set_title(f"{item['task_id'].split('_')[-2]}_{item['task_id'].split('_')[-1]}\n{title}",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)

    # Hide unused subplots
    for idx in range(n_problems, n_rows * n_cols):
        row = idx // n_cols
        col_offset = (idx % n_cols) * 3
        for sub_col in range(3):
            axes[row, col_offset + sub_col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved task visualization to {output_path}")


def visualize_task_pair(data, task1_id, task2_id, tasks, output_path: str):
    """Visualize all problems from two nearest tasks."""
    # ARC color palette
    colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    task1_indices = tasks[task1_id]
    task2_indices = tasks[task2_id]

    # Get first problem from each task for visualization
    item1 = data[task1_indices[0]]
    item2 = data[task2_indices[0]]

    # Count solved/unsolved in each task
    task1_solved = sum(1 for idx in task1_indices if data[idx]['solved'])
    task2_solved = sum(1 for idx in task2_indices if data[idx]['solved'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Nearest Task Pair: {task1_id} ({task1_solved}/{len(task1_indices)} solved) ↔ {task2_id} ({task2_solved}/{len(task2_indices)} solved)',
                 fontsize=12, fontweight='bold')

    # Task 1 - first problem
    for col, (grid_data, title) in enumerate([
        (item1['input_grid'], f'{task1_id}\nInput'),
        (item1['output_grid'], 'Output (target)'),
        (item1['pred_grid'], f"Pred ({'✓' if item1['solved'] else '✗'})")
    ]):
        ax = axes[0, col]
        grid = np.array(grid_data)

        # Create colored grid
        colored_grid = np.zeros((*grid.shape, 3))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color_idx = grid[i, j] % 10
                hex_color = colors[color_idx]
                colored_grid[i, j] = [int(hex_color[k:k+2], 16)/255 for k in (1, 3, 5)]

        ax.imshow(colored_grid, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)

    # Task 2 - first problem
    for col, (grid_data, title) in enumerate([
        (item2['input_grid'], f'{task2_id}\nInput'),
        (item2['output_grid'], 'Output (target)'),
        (item2['pred_grid'], f"Pred ({'✓' if item2['solved'] else '✗'})")
    ]):
        ax = axes[1, col]
        grid = np.array(grid_data)

        # Create colored grid
        colored_grid = np.zeros((*grid.shape, 3))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color_idx = grid[i, j] % 10
                hex_color = colors[color_idx]
                colored_grid[i, j] = [int(hex_color[k:k+2], 16)/255 for k in (1, 3, 5)]

        ax.imshow(colored_grid, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved task pair visualization to {output_path}")


def main():
    # Paths
    latents_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/latents.json"
    vis_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/visualizations"

    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TRM Latent Visualization (TEST ONLY)")
    print("="*60)

    # Load latents
    data, latents, solved_labels = load_latents(latents_path)

    # Filter TEST examples only
    test_indices = [i for i, d in enumerate(data) if '_test_' in d['task_id']]
    data = [data[i] for i in test_indices]
    latents = latents[test_indices]
    solved_labels = solved_labels[test_indices]

    print(f"\n📊 Filtered to TEST examples only:")
    print(f"  Total: {len(data)}")
    print(f"  Solved: {solved_labels.sum()} ({solved_labels.sum()/len(solved_labels)*100:.1f}%)")
    print(f"  Unsolved: {(~solved_labels).sum()} ({(~solved_labels).sum()/len(solved_labels)*100:.1f}%)")

    # PCA
    latents_2d, pca = perform_pca(latents)

    # Plot PCA scatter
    plot_pca_scatter(
        latents_2d,
        solved_labels,
        f"{vis_dir}/pca_scatter.png"
    )

    # Group by task
    tasks = group_by_task(data)

    # Find tasks with high solve rates (instead of nearest pairs)
    print("\n🔍 Finding tasks with highest solve rates...")
    task_solve_rates = []
    for task_id, indices in tasks.items():
        solved_count = sum(1 for idx in indices if data[idx]['solved'])
        total_count = len(indices)
        solve_rate = solved_count / total_count
        task_solve_rates.append((task_id, solved_count, total_count, solve_rate))

    # Sort by solve rate
    task_solve_rates.sort(key=lambda x: x[3], reverse=True)

    print(f"✅ Top tasks by solve rate:")
    for i in range(min(5, len(task_solve_rates))):
        task_id, solved, total, rate = task_solve_rates[i]
        print(f"  {i+1}. {task_id}: {solved}/{total} ({rate*100:.1f}%)")

    # Visualize top 3 tasks with highest solve rate
    for idx in range(min(3, len(task_solve_rates))):
        task_id, solved_count, total_count, solve_rate = task_solve_rates[idx]
        print(f"\n📊 Visualizing task {idx+1}: {task_id} ({solved_count}/{total_count} solved)...")

        # Visualize this task
        visualize_single_task(
            data,
            task_id,
            tasks,
            f"{vis_dir}/solved_task_{idx+1}_{task_id}.png"
        )

    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("="*60)
    print(f"Output directory: {vis_dir}")


if __name__ == "__main__":
    main()
