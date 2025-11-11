"""
Similarity metrics for puzzle retrieval.

Used to find similar puzzles from training set for better initialization.
"""

import numpy as np
from typing import List, Tuple


def color_histogram_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """
    Compute similarity based on color distribution.

    Args:
        grid1, grid2: np.ndarray with values 0-9 (ARC colors)

    Returns:
        similarity score in [0, 1] (higher = more similar)
    """
    # Flatten grids
    flat1 = grid1.flatten()
    flat2 = grid2.flatten()

    # Compute color histograms (10 colors: 0-9)
    hist1 = np.bincount(flat1, minlength=10).astype(float)
    hist2 = np.bincount(flat2, minlength=10).astype(float)

    # Normalize to probability distributions
    hist1_norm = hist1 / (hist1.sum() + 1e-8)
    hist2_norm = hist2 / (hist2.sum() + 1e-8)

    # Cosine similarity
    dot_product = np.dot(hist1_norm, hist2_norm)
    norm1 = np.linalg.norm(hist1_norm)
    norm2 = np.linalg.norm(hist2_norm)

    similarity = dot_product / (norm1 * norm2 + 1e-8)

    return float(similarity)


def grid_size_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """
    Compute similarity based on grid dimensions.

    Returns:
        similarity score in [0, 1]
    """
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape

    # Jaccard similarity of dimensions
    h_sim = min(h1, h2) / max(h1, h2)
    w_sim = min(w1, w2) / max(w1, w2)

    return (h_sim + w_sim) / 2


def combined_similarity(grid1: np.ndarray, grid2: np.ndarray,
                       color_weight: float = 0.7,
                       size_weight: float = 0.3) -> float:
    """
    Combined similarity using both color and size.

    Args:
        grid1, grid2: Input grids
        color_weight: Weight for color histogram similarity
        size_weight: Weight for size similarity

    Returns:
        Combined similarity score in [0, 1]
    """
    color_sim = color_histogram_similarity(grid1, grid2)
    size_sim = grid_size_similarity(grid1, grid2)

    return color_weight * color_sim + size_weight * size_sim


def compute_example_similarity(examples1: List[dict],
                               examples2: List[dict]) -> float:
    """
    Compute average similarity between two sets of examples.

    Args:
        examples1, examples2: Lists of dicts with 'input' and 'output' grids

    Returns:
        Average similarity score
    """
    if not examples1 or not examples2:
        return 0.0

    total_sim = 0.0
    count = 0

    # Compare each pair of examples
    for ex1 in examples1:
        for ex2 in examples2:
            # Similarity on input grids
            input_sim = combined_similarity(ex1['input'], ex2['input'])

            # Similarity on output grids (optional)
            output_sim = combined_similarity(ex1['output'], ex2['output'])

            # Average
            pair_sim = (input_sim + output_sim) / 2

            total_sim += pair_sim
            count += 1

    return total_sim / count if count > 0 else 0.0


def precompute_color_histograms(puzzle_examples: List[dict]) -> np.ndarray:
    """
    Pre-compute average color histogram for a puzzle.

    Args:
        puzzle_examples: List of examples with 'input' and 'output'

    Returns:
        Average normalized histogram (shape: (10,))
    """
    histograms = []

    for ex in puzzle_examples:
        # Combine input and output
        combined = np.concatenate([ex['input'].flatten(), ex['output'].flatten()])

        # Histogram
        hist = np.bincount(combined, minlength=10).astype(float)
        hist_norm = hist / (hist.sum() + 1e-8)

        histograms.append(hist_norm)

    # Average across examples
    avg_hist = np.mean(histograms, axis=0) if histograms else np.zeros(10)

    return avg_hist
