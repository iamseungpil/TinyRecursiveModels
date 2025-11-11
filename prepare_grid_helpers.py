"""
Grid preparation helpers for test-time training.
"""

import torch
import numpy as np


def prepare_grid(grid: np.ndarray, add_eos: bool = True) -> torch.Tensor:
    """Convert grid to model input format."""
    h, w = grid.shape

    # Pad with 0 and shift values by 2
    padded = np.pad(grid + 2, ((0, 30 - h), (0, 30 - w)), constant_values=0)

    if add_eos:
        eos_row, eos_col = h, w
        if eos_row < 30:
            padded[eos_row, 0:eos_col] = 1
        if eos_col < 30:
            padded[0:eos_row, eos_col] = 1

    return torch.from_numpy(padded.reshape(-1)).long()


def prepare_grid_label(grid: np.ndarray, add_eos: bool = True) -> torch.Tensor:
    """
    Convert grid to label format with proper masking.

    Returns:
        Tensor with:
        - -100 for padding/EOS positions (ignored in loss)
        - 2-11 for actual grid colors
    """
    h, w = grid.shape

    # Create padded grid with -100 for padding
    padded = np.full((30, 30), -100, dtype=np.int64)

    # Fill valid region with shifted colors (0-9 → 2-11)
    padded[:h, :w] = grid + 2

    if add_eos:
        # Mark EOS positions as -100
        eos_row, eos_col = h, w
        if eos_row < 30:
            padded[eos_row, 0:eos_col] = -100
        if eos_col < 30:
            padded[0:eos_row, eos_col] = -100

    return torch.from_numpy(padded.reshape(-1)).long()
