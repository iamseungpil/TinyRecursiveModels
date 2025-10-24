"""
Feedback Formatter - Grid validation and feedback generation

Clean interface: works with grids (np.ndarray) only.
Token conversion is handled by caller (GridVerifier).
"""

import numpy as np
import torch


def tokens_to_grid(tokens: torch.Tensor, max_grid_size: int = 30) -> np.ndarray:
    """
    Convert tokenized grid back to 2D numpy array.

    Token format:
    - PAD: 0
    - <eos>: 1
    - digits 0-9: tokens 2-11

    Args:
        tokens: [seq_len] tensor with token IDs
        max_grid_size: Maximum grid dimension (default 30 for ARC)

    Returns:
        2D numpy array representing the grid

    Raises:
        ValueError: If tokens shape is invalid
    """
    # Validate input shape
    if tokens.ndim != 1:
        raise ValueError(
            f"Expected 1D tensor, got {tokens.ndim}D tensor with shape {tokens.shape}"
        )

    expected_len = max_grid_size * max_grid_size
    if tokens.shape[0] != expected_len:
        raise ValueError(
            f"Expected {expected_len} tokens ({max_grid_size}×{max_grid_size}), "
            f"got {tokens.shape[0]} tokens"
        )

    seq = tokens.cpu().numpy()
    grid = seq.reshape(max_grid_size, max_grid_size)

    # Subtract 2 to convert tokens (2-11) back to values (0-9)
    # Handle PAD (0) and EOS (1) by clamping
    grid = np.clip(grid - 2, 0, 9).astype(np.uint8)

    # Find actual grid bounds using EOS markers
    rows_with_eos = np.any(seq.reshape(max_grid_size, max_grid_size) == 1, axis=1)
    cols_with_eos = np.any(seq.reshape(max_grid_size, max_grid_size) == 1, axis=0)

    if rows_with_eos.any():
        end_row = np.argmax(rows_with_eos)
    else:
        end_row = max_grid_size

    if cols_with_eos.any():
        end_col = np.argmax(cols_with_eos)
    else:
        end_col = max_grid_size

    return grid[:end_row, :end_col]


def grid_to_feedback(
    pred_grid: np.ndarray,
    target_grid: np.ndarray,
    max_errors_shown: int = 5
) -> str:
    """
    Generate concise textual feedback from grid comparison.

    Simplified to avoid overwhelming LLaMA with long grid strings.

    Args:
        pred_grid: [H, W] predicted grid
        target_grid: [H, W] target grid
        max_errors_shown: Maximum number of error coordinates to display

    Returns:
        Concise feedback string describing errors
    """
    if pred_grid.shape != target_grid.shape:
        return (
            f"\nSize mismatch: predicted {pred_grid.shape[0]}x{pred_grid.shape[1]}, "
            f"but target is {target_grid.shape[0]}x{target_grid.shape[1]}."
        )

    diff = pred_grid != target_grid
    coords = list(zip(*np.where(diff)))

    if not coords:
        return "\nPrediction is correct!"

    # Show only top N coords to keep feedback concise
    sample = coords[:max_errors_shown]
    num_errors = len(coords)
    coord_text = ", ".join(f"({r},{c})" for r, c in sample)

    more_text = f" ({num_errors - max_errors_shown} more)" if num_errors > max_errors_shown else ""

    return (
        f"\n{num_errors} cells incorrect. Key errors at: {coord_text}{more_text}. "
        "Re-evaluate the transformation rule."
    )


def format_problem_text(problem_description: str, history: str = "") -> str:
    """
    Format ARC problem and history for LLM consumption.

    Adds "Let me reason step by step" suffix ONCE (not in TextReasoningModule).

    Args:
        problem_description: ARC problem with examples
        history: Optional history of previous attempts

    Returns:
        Formatted prompt string
    """
    prompt = problem_description

    if history:
        prompt += f"\n\nPrevious attempts:\n{history}"

    # Add reasoning prompt ONCE (not in TextReasoningModule._prepare_prompt)
    prompt += "\n\nLet me reason step by step before predicting the grid."

    return prompt
