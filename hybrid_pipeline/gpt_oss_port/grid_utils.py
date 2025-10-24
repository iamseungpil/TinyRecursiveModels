"""
Grid Utilities - Extracted from GPT-OSS sequential_validation_v4.py

Provides grid formatting, parsing, and validation WITHOUT LLM inference.
"""

import numpy as np
from typing import Optional


def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)


def string_to_grid(grid_str: str) -> Optional[np.ndarray]:
    """Convert string grid back to numpy array."""
    try:
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split()]
                if row:
                    grid.append(row)
        return np.array(grid)
    except Exception as e:
        print(f"❌ Error parsing grid: {e}")
        return None


def compare_grids(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Compare two grids for exact match."""
    if predicted is None or target is None:
        return False
    if predicted.shape != target.shape:
        return False
    return np.array_equal(predicted, target)


def grid_shape_matches(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Check if grid shapes match."""
    if predicted is None or target is None:
        return False
    return predicted.shape == target.shape
