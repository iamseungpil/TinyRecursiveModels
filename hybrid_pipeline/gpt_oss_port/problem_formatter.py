"""
Problem Formatter - Convert ARC problems to LLM prompts

Formats training examples and test input into clear text for LLM planning.
"""

import numpy as np
from typing import List, Tuple

try:
    from .grid_utils import grid_to_string
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from grid_utils import grid_to_string


def format_arc_problem(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    include_instructions: bool = True,
) -> str:
    """
    Format ARC problem with training examples for LLM.

    Args:
        train_pairs: List of (input_grid, output_grid) training examples
        test_input: Test input grid to transform
        include_instructions: Whether to include task instructions

    Returns:
        Formatted prompt string for LLM

    Example:
        >>> train = [(np.array([[0,1],[1,0]]), np.array([[1,0],[0,1]]))]
        >>> test = np.array([[0,1],[1,0]])
        >>> print(format_arc_problem(train, test))
        Solve this ARC puzzle by finding the transformation pattern.

        Training Example 1:
        Input:
        0 1
        1 0
        Output:
        1 0
        0 1

        Test Input:
        0 1
        1 0

        Provide a step-by-step transformation plan.
        Format each step as: 'Step N: <instruction>'
        Plan:
    """
    lines = []

    if include_instructions:
        lines.append("Solve this ARC puzzle by finding the transformation pattern.\n")

    # Format training examples
    for i, (input_grid, output_grid) in enumerate(train_pairs, 1):
        lines.append(f"Training Example {i}:")
        lines.append("Input:")
        lines.append(grid_to_string(input_grid))
        lines.append("Output:")
        lines.append(grid_to_string(output_grid))
        lines.append("")  # Blank line between examples

    # Format test input
    lines.append("Test Input:")
    lines.append(grid_to_string(test_input))
    lines.append("")

    if include_instructions:
        lines.append("Provide a step-by-step transformation plan.")
        lines.append("Format each step as: 'Step N: <instruction>'")
        lines.append("Plan:")

    return "\n".join(lines)


def format_arc_problem_with_feedback(
    original_problem: str,
    feedback: str,
) -> str:
    """
    Add feedback to original problem for plan refinement.

    Args:
        original_problem: Original problem description
        feedback: Execution feedback from previous attempt

    Returns:
        Problem + feedback for LLM
    """
    return (
        f"{original_problem}\n\n"
        f"Previous Attempt Feedback:\n"
        f"{feedback}\n\n"
        f"Based on this feedback, provide a REVISED step-by-step plan.\n"
        f"Plan:"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Problem Formatter Test")
    print("=" * 60)

    # Test case
    train_pairs = [
        (
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        ),
        (
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
        ),
    ]

    test_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    problem = format_arc_problem(train_pairs, test_input)
    print(problem)

    print("\n" + "=" * 60)
    print("With Feedback:")
    print("=" * 60)

    feedback = "Step 1 changed too many cells. Try a more conservative approach."
    problem_with_feedback = format_arc_problem_with_feedback(problem, feedback)
    print(problem_with_feedback)

    print("\n" + "=" * 60)
    print("✓ Problem formatter test complete!")
    print("=" * 60)
