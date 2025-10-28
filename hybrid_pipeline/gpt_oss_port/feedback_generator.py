"""
Feedback Generator - Create detailed feedback from intermediate grids

Converts step-by-step execution trace into human-readable (LLM-readable) feedback.
Shows what happened at each step so LLM can diagnose and fix the plan.
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from .grid_utils import grid_to_string
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from grid_utils import grid_to_string


def format_step_by_step_feedback(
    steps: List[str],
    intermediate_results: List[Dict[str, Any]],
    target_grid: np.ndarray,
    original_input: np.ndarray,
) -> str:
    """
    Generate detailed feedback showing execution trace of all steps.

    Args:
        steps: List of instruction strings
        intermediate_results: List of dicts with keys:
            - 'step_idx': int
            - 'instruction': str
            - 'input_grid': np.ndarray
            - 'output_grid': np.ndarray
        target_grid: Expected final output
        original_input: Original problem input

    Returns:
        Formatted feedback string for LLM
    """
    lines = []

    lines.append("=" * 60)
    lines.append("EXECUTION TRACE")
    lines.append("=" * 60)

    # Show original input
    lines.append("\nOriginal Input:")
    lines.append(grid_to_string(original_input))
    lines.append("")

    # Show each step
    for i, (step, result) in enumerate(zip(steps, intermediate_results), 1):
        lines.append(f"Step {i}: {step}")
        lines.append(f"  Input shape: {result['input_grid'].shape}")
        lines.append(f"  Input:\n{_indent(grid_to_string(result['input_grid']))}")

        lines.append(f"  Output shape: {result['output_grid'].shape}")
        lines.append(f"  Output:\n{_indent(grid_to_string(result['output_grid']))}")

        # Analyze this step's transformation
        analysis = _analyze_grid_transformation(
            result['input_grid'],
            result['output_grid']
        )
        if analysis:
            lines.append(f"  Analysis: {analysis}")

        lines.append("")

    # Show target
    lines.append("Expected Target:")
    lines.append(grid_to_string(target_grid))
    lines.append("")

    # Compare final output with target
    if len(intermediate_results) > 0:
        final_output = intermediate_results[-1]['output_grid']
        comparison = _compare_grids(final_output, target_grid)
        lines.append("Final Comparison:")
        lines.append(comparison)
    else:
        lines.append("No steps were executed!")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Please refine your plan based on this feedback.")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_success_feedback(
    steps: List[str],
    num_steps_executed: int,
) -> str:
    """
    Generate feedback for successful execution.

    Args:
        steps: List of instruction strings
        num_steps_executed: How many steps were actually run

    Returns:
        Success message
    """
    return (
        f"✓ Success! Solved in {num_steps_executed} steps.\n"
        f"Plan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    )


def _indent(text: str, spaces: int = 4) -> str:
    """Indent each line of text."""
    indent_str = " " * spaces
    return "\n".join(indent_str + line for line in text.split("\n"))


def _analyze_grid_transformation(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> str:
    """
    Analyze what changed between input and output.

    Returns brief description of transformation.
    """
    analysis_parts = []

    # Shape change
    if input_grid.shape != output_grid.shape:
        analysis_parts.append(
            f"Shape changed from {input_grid.shape} to {output_grid.shape}"
        )

    # Color distribution
    input_colors = _get_color_distribution(input_grid)
    output_colors = _get_color_distribution(output_grid)

    if input_colors != output_colors:
        added_colors = set(output_colors.keys()) - set(input_colors.keys())
        removed_colors = set(input_colors.keys()) - set(output_colors.keys())

        if added_colors:
            analysis_parts.append(f"Added colors: {sorted(added_colors)}")
        if removed_colors:
            analysis_parts.append(f"Removed colors: {sorted(removed_colors)}")

    # Cell changes (if same shape)
    if input_grid.shape == output_grid.shape:
        changed_cells = (input_grid != output_grid).sum()
        total_cells = input_grid.size
        pct = 100 * changed_cells / total_cells if total_cells > 0 else 0

        if changed_cells > 0:
            analysis_parts.append(
                f"{changed_cells}/{total_cells} cells changed ({pct:.1f}%)"
            )
        else:
            analysis_parts.append("No changes")

    return "; ".join(analysis_parts) if analysis_parts else "Transformation applied"


def _compare_grids(
    predicted: np.ndarray,
    target: np.ndarray,
) -> str:
    """
    Compare predicted grid with target.

    Returns detailed comparison message.
    """
    lines = []

    # Shape comparison
    if predicted.shape != target.shape:
        lines.append(
            f"  ✗ Shape mismatch: got {predicted.shape}, expected {target.shape}"
        )
        return "\n".join(lines)

    # Exact match
    if np.array_equal(predicted, target):
        lines.append("  ✓ Perfect match!")
        return "\n".join(lines)

    # Cell-level comparison
    diff_mask = predicted != target
    num_errors = diff_mask.sum()
    total_cells = target.size
    accuracy = 100 * (1 - num_errors / total_cells)

    lines.append(f"  ✗ {num_errors}/{total_cells} cells incorrect ({accuracy:.1f}% accuracy)")

    # Find error locations (show first few)
    error_positions = np.argwhere(diff_mask)
    num_to_show = min(5, len(error_positions))

    if num_to_show > 0:
        lines.append(f"  First {num_to_show} errors:")
        for pos in error_positions[:num_to_show]:
            row, col = pos
            lines.append(
                f"    Position ({row},{col}): got {predicted[row,col]}, expected {target[row,col]}"
            )

    return "\n".join(lines)


def _get_color_distribution(grid: np.ndarray) -> Dict[int, int]:
    """Get count of each color in grid."""
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


# Self-review questions:
#
# Q1: Should we visualize grids with colors or just numbers?
# A1: Numbers for now (grid_to_string). Could add ANSI colors later.
#     Decision: Keep simple, LLMs read text well.
#
# Q2: How much detail in analysis?
# A2: Balance between too verbose and too terse.
#     Current: Shape, color changes, cell differences.
#     Decision: Good starting point, can add more if needed.
#
# Q3: Should we show ALL error positions or just a few?
# A3: Large grids could have 100+ errors. Show first 5.
#     Decision: Limit to 5, enough to see pattern.
#
# Q4: What if intermediate_results is empty?
# A4: Handle gracefully with "No steps executed" message.
#     Decision: Added check.
#
# Q5: Should we suggest specific fixes?
# A5: Tempting, but risk being wrong. Let LLM diagnose.
#     Decision: Show data, let LLM reason.
#
# Q6: Performance with large grids (30x30)?
# A6: grid_to_string can be slow. But feedback is generated rarely (only on failure).
#     Decision: Acceptable for now.


if __name__ == "__main__":
    print("=" * 60)
    print("Feedback Generator Test")
    print("=" * 60)

    # Create test data
    original_input = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    target_grid = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ])

    steps = [
        "Flip colors (0↔1)",
    ]

    # Simulate execution
    intermediate_results = [
        {
            'step_idx': 0,
            'instruction': steps[0],
            'input_grid': original_input.copy(),
            'output_grid': np.array([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ]),
        }
    ]

    # Test 1: Success case
    print("\nTest 1: Successful execution")
    print("-" * 60)
    feedback = format_step_by_step_feedback(
        steps,
        intermediate_results,
        target_grid,
        original_input,
    )
    print(feedback)

    # Test 2: Failure case
    print("\n\nTest 2: Failed execution")
    print("-" * 60)

    steps_fail = [
        "Rotate 90 degrees",
        "Flip vertically",
    ]

    intermediate_results_fail = [
        {
            'step_idx': 0,
            'instruction': steps_fail[0],
            'input_grid': original_input.copy(),
            'output_grid': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ]),
        },
        {
            'step_idx': 1,
            'instruction': steps_fail[1],
            'input_grid': np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ]),
            'output_grid': np.array([
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]),
        }
    ]

    feedback_fail = format_step_by_step_feedback(
        steps_fail,
        intermediate_results_fail,
        target_grid,
        original_input,
    )
    print(feedback_fail)

    print("\n" + "=" * 60)
    print("✓ Feedback generator tests complete!")
    print("=" * 60)
