"""
Plan Parser - Extract step-by-step instructions from LLM output

Parses multi-step plans in format:
    "Step 1: Do this
     Step 2: Do that
     Step 3: Do something else"

Returns list of instruction strings for sequential execution.
"""

import re
from typing import List, Optional


def parse_multi_step_plan(llm_output: str) -> List[str]:
    """
    Extract step-by-step instructions from LLM generated plan.

    Supports formats:
        - "Step 1: instruction"
        - "Step 1. instruction"
        - "1. instruction"
        - "1) instruction"

    Args:
        llm_output: Raw text from LLM containing step-by-step plan

    Returns:
        List of instruction strings (without step numbers)

    Examples:
        >>> parse_multi_step_plan("Step 1: Rotate grid\\nStep 2: Flip colors")
        ['Rotate grid', 'Flip colors']

        >>> parse_multi_step_plan("1. Extract objects\\n2. Sort by size")
        ['Extract objects', 'Sort by size']
    """
    # Try multiple patterns in order of specificity
    patterns = [
        r"Step\s+(\d+):\s*(.+?)(?=\n\s*Step\s+\d+:|$)",  # "Step 1: ..."
        r"Step\s+(\d+)\.\s*(.+?)(?=\n\s*Step\s+\d+\.|$)",  # "Step 1. ..."
        r"^(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)",  # "1. ..."
        r"^(\d+)\)\s*(.+?)(?=\n\s*\d+\)|$)",  # "1) ..."
    ]

    for pattern in patterns:
        matches = re.findall(pattern, llm_output, re.MULTILINE | re.DOTALL)
        if matches:
            # Extract just the instruction text (second group)
            steps = [match[1].strip() for match in matches]
            # Filter out empty steps
            steps = [s for s in steps if s]
            if steps:
                return steps

    # Fallback: no structured steps found, treat entire output as single step
    clean_output = llm_output.strip()
    if clean_output:
        return [clean_output]

    return []


def validate_plan(steps: List[str], min_steps: int = 1, max_steps: int = 10) -> bool:
    """
    Validate that plan has reasonable number of steps.

    Args:
        steps: List of instruction strings
        min_steps: Minimum acceptable number of steps
        max_steps: Maximum acceptable number of steps

    Returns:
        True if valid, False otherwise
    """
    if not steps:
        return False

    if len(steps) < min_steps or len(steps) > max_steps:
        return False

    # Check that each step has some content
    for step in steps:
        if not step or len(step.strip()) < 3:
            return False

    return True


def format_plan_for_display(steps: List[str]) -> str:
    """
    Format steps for human-readable display.

    Args:
        steps: List of instruction strings

    Returns:
        Formatted string with numbered steps
    """
    if not steps:
        return "(No steps)"

    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"Step {i}: {step}")

    return "\n".join(lines)


# Self-review questions for this implementation:
#
# Q1: What if LLM outputs steps out of order (e.g., Step 3, Step 1, Step 2)?
# A1: Current implementation extracts in document order, not by step number.
#     Should we sort by step number? Trade-off: LLM might intentionally reorder.
#     Decision: Keep document order for now, add sorting option if needed.
#
# Q2: What if LLM includes sub-steps (e.g., "Step 1a: ..., Step 1b: ...")?
# A2: Current regex won't match. Should we support hierarchical steps?
#     Decision: Start simple, add if needed. Most ARC tasks are linear.
#
# Q3: How to handle very long instructions (e.g., 500+ words)?
# A3: No length limit currently. Could truncate or warn.
#     Decision: Let TRM/adapter handle it, they'll truncate if needed.
#
# Q4: What about multilingual instructions?
# A4: Regex is language-agnostic for numbers. Should work for Korean, Chinese, etc.
#     Decision: Good for now.
#
# Q5: Should we cache parsed plans to avoid re-parsing?
# A5: Premature optimization. Plans are short, parsing is fast.
#     Decision: Add if profiling shows it's a bottleneck.


if __name__ == "__main__":
    # Test cases
    test_cases = [
        """Step 1: Segment objects by color
Step 2: Extract each region
Step 3: Arrange horizontally""",

        """Step 1. Identify the pattern
Step 2. Apply transformation
Step 3. Verify output""",

        """1. Rotate 90 degrees
2. Flip vertically
3. Change color 1 to 2""",

        """1) First do this
2) Then do that
3) Finally do something else""",

        "Single step instruction without numbering",
    ]

    print("=" * 60)
    print("Plan Parser Test Cases")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input:\n{test}")
        print(f"\nParsed steps:")
        steps = parse_multi_step_plan(test)
        for j, step in enumerate(steps, 1):
            print(f"  {j}. {step}")
        print(f"Valid: {validate_plan(steps)}")

    print("\n" + "=" * 60)
    print("✓ All test cases completed")
    print("=" * 60)
