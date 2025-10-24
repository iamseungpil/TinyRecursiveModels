"""
Grid Verifier - Validation and feedback generation

Compares predicted grids with targets and generates feedback for self-correction.
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch

from .grid_utils import compare_grids, grid_shape_matches
from adapters.feedback_formatter import grid_to_feedback, tokens_to_grid


class GridVerifier:
    """
    Verifies grid predictions and generates feedback.

    Uses grid_utils for comparison and feedback_formatter for error messages.
    """

    def __init__(
        self,
        target_grid: np.ndarray,
        max_errors_shown: int = 5
    ):
        """
        Args:
            target_grid: Ground truth output grid
            max_errors_shown: Maximum error coordinates to show in feedback
        """
        self.target_grid = target_grid
        self.max_errors_shown = max_errors_shown

    def verify_tokens(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify predicted tokens against target tokens.

        Args:
            pred_tokens: [seq_len] predicted token IDs
            target_tokens: [seq_len] target token IDs

        Returns:
            success: True if exact match
            feedback: Feedback string for next attempt
            metrics: Dictionary with accuracy metrics
        """
        # Convert tokens to grids
        pred_grid = tokens_to_grid(pred_tokens)
        target_grid = tokens_to_grid(target_tokens)

        # Check exact match
        success = compare_grids(pred_grid, target_grid)

        # Generate feedback
        if success:
            feedback = "Correct! Grid prediction matches target."
        else:
            feedback = grid_to_feedback(
                pred_grid,
                target_grid,
                max_errors_shown=self.max_errors_shown
            )

        # Compute metrics
        shape_match = grid_shape_matches(pred_grid, target_grid)

        if shape_match:
            accuracy = (pred_grid == target_grid).mean()
        else:
            accuracy = 0.0

        metrics = {
            "exact_match": success,
            "shape_match": shape_match,
            "cell_accuracy": float(accuracy),
            "pred_shape": pred_grid.shape,
            "target_shape": target_grid.shape
        }

        return success, feedback, metrics

    def verify_grid(
        self,
        pred_grid: np.ndarray
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify predicted grid directly.

        Args:
            pred_grid: Predicted output grid

        Returns:
            success: True if exact match
            feedback: Feedback string
            metrics: Accuracy metrics
        """
        success = compare_grids(pred_grid, self.target_grid)

        if success:
            feedback = "Correct! Grid prediction matches target."
            shape_match = True
            accuracy = 1.0
        else:
            feedback = grid_to_feedback(
                pred_grid,
                self.target_grid,
                max_errors_shown=self.max_errors_shown
            )
            shape_match = grid_shape_matches(pred_grid, self.target_grid)
            accuracy = (pred_grid == self.target_grid).mean() if shape_match else 0.0

        metrics = {
            "exact_match": success,
            "shape_match": shape_match,
            "cell_accuracy": float(accuracy),
            "pred_shape": pred_grid.shape,
            "target_shape": self.target_grid.shape
        }

        return success, feedback, metrics
