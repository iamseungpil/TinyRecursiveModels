"""
Unit tests for planner module with mock LLM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from gpt_oss_port.llm import LatentOutput
from gpt_oss_port.planner import ARCPlanner


class MockLLM:
    """Mock LLM for testing planner logic."""

    def __init__(self, hidden_size: int = 4096):
        self.hidden_size = hidden_size
        self.call_count = 0

    def generate_latent(self, prompt: str, latent_prefix=None, max_length: int = 128, **_kwargs):
        """Generate mock hidden state and text."""
        self.call_count += 1
        z_init = torch.randn(self.hidden_size)
        text = f"Mock reasoning attempt {self.call_count}: {prompt[:50]}..."
        return LatentOutput(
            final_hidden=z_init,
            generated_text=text,
            hidden_sequence=None,
            attention_mask=None,
        )


def test_planner_basic():
    """Test basic planner initialization and reset."""
    print("🧪 Testing planner basic functionality...")

    mock_llm = MockLLM()
    planner = ARCPlanner(
        llm_module=mock_llm,
        max_attempts=3,
        max_tokens_per_attempt=100
    )

    assert planner.max_attempts == 3
    assert planner.attempt_count == 0
    assert len(planner.history) == 0

    planner.attempt_count = 5
    planner.history = ["test"]
    planner.reset()

    assert planner.attempt_count == 0
    assert len(planner.history) == 0

    print("  ✅ Basic functionality works")


def test_planner_problem_description():
    """Test problem description generation."""
    print("\n🧪 Testing problem description generation...")

    mock_llm = MockLLM()
    planner = ARCPlanner(llm_module=mock_llm, max_attempts=3)

    train_pairs = [
        (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])),
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]]))
    ]
    test_input = np.array([[0, 0], [1, 1]])

    desc = planner.create_problem_description(train_pairs, test_input)

    assert "Example 1:" in desc
    assert "Example 2:" in desc
    assert "Test Input:" in desc
    assert "0 1" in desc  # Check grid formatting

    print("  ✅ Problem description generated correctly")


def test_planner_single_step():
    """Test single planning step."""
    print("\n🧪 Testing single planning step...")

    mock_llm = MockLLM()
    planner = ARCPlanner(llm_module=mock_llm, max_attempts=3)

    problem_desc = "Test problem"
    latent = planner.plan_step(problem_desc)

    assert latent.final_hidden.shape == (4096,)
    assert isinstance(latent.generated_text, str)
    assert planner.attempt_count == 1
    assert mock_llm.call_count == 1

    print(f"  ✅ Planning step executed (z_init shape: {latent.final_hidden.shape})")


def test_planner_multi_attempt():
    """Test multi-attempt solving loop."""
    print("\n🧪 Testing multi-attempt loop...")

    mock_llm = MockLLM()
    planner = ARCPlanner(llm_module=mock_llm, max_attempts=3)

    problem_desc = "Test problem"

    # Mock verifier that succeeds on 2nd attempt
    attempt_counter = [0]

    def mock_verifier(z_init, attempt):
        attempt_counter[0] += 1
        success = attempt_counter[0] == 2
        feedback = "Try again" if not success else "Correct!"
        metrics = {"attempt": attempt, "success": success}
        return success, feedback, metrics

    # Run multi-attempt
    results = planner.multi_attempt_solve(
        problem_desc,
        verifier_fn=mock_verifier
    )

    assert results["success"] == True
    assert results["total_attempts"] == 2
    assert len(results["reasoning_history"]) == 2
    assert len(results["metrics"]) == 2

    print(f"  ✅ Multi-attempt loop worked (stopped at attempt 2)")


def test_planner_max_attempts():
    """Test that planner respects max_attempts limit."""
    print("\n🧪 Testing max attempts limit...")

    mock_llm = MockLLM()
    planner = ARCPlanner(llm_module=mock_llm, max_attempts=3)

    problem_desc = "Test problem"

    # Mock verifier that always fails
    def always_fail_verifier(z_init, attempt):
        return False, "Failed", {"attempt": attempt}

    results = planner.multi_attempt_solve(
        problem_desc,
        verifier_fn=always_fail_verifier
    )

    assert results["success"] == False
    assert results["total_attempts"] == 3
    assert len(results["reasoning_history"]) == 3

    print(f"  ✅ Max attempts respected (stopped at 3)")


def test_planner_with_feedback():
    """Test planner with feedback accumulation."""
    print("\n🧪 Testing feedback accumulation...")

    mock_llm = MockLLM()
    planner = ARCPlanner(llm_module=mock_llm, max_attempts=3)

    problem_desc = "Test problem"

    # Mock verifier with different feedback each time
    def feedback_verifier(z_init, attempt):
        feedback = f"Feedback for attempt {attempt + 1}"
        return False, feedback, {"attempt": attempt}

    results = planner.multi_attempt_solve(
        problem_desc,
        verifier_fn=feedback_verifier
    )

    # Check that feedback was accumulated in history
    assert len(planner.history) > 0
    assert results["total_attempts"] == 3

    print(f"  ✅ Feedback accumulated ({len(planner.history)} items in history)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Planner Tests")
    print("=" * 60)

    test_planner_basic()
    test_planner_problem_description()
    test_planner_single_step()
    test_planner_multi_attempt()
    test_planner_max_attempts()
    test_planner_with_feedback()

    print("\n" + "=" * 60)
    print("✅ All planner tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
