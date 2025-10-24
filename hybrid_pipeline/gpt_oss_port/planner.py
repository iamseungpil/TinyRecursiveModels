"""
ARC Problem Planner - Core reasoning loop

Extracted from sequential_validation_v4.py, modularized for hybrid pipeline.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .llm import LatentOutput, TextReasoningModule
from .grid_utils import grid_to_string
from adapters.feedback_formatter import format_problem_text


class ARCPlanner:
    """
    Plans and executes multi-attempt problem solving with LLM.

    Maintains conversation history and coordinates with verifier for feedback.
    """

    def __init__(
        self,
        llm_module: TextReasoningModule,
        max_attempts: int = 16,
        max_tokens_per_attempt: int = 128
    ):
        """
        Args:
            llm_module: Text reasoning module (LLaMA)
            max_attempts: Maximum reasoning attempts per problem
            max_tokens_per_attempt: Maximum tokens to generate per attempt
        """
        self.llm = llm_module
        self.max_attempts = max_attempts
        self.max_tokens_per_attempt = max_tokens_per_attempt

        # Conversation state
        self.history: List[str] = []
        self.attempt_count = 0

    def reset(self):
        """Reset planner state for new problem."""
        self.history = []
        self.attempt_count = 0

    def create_problem_description(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> str:
        """
        Create ARC problem description from examples.

        Args:
            train_pairs: List of (input_grid, output_grid) training examples
            test_input: Test input grid

        Returns:
            Problem description string
        """
        examples = []
        for i, (inp, out) in enumerate(train_pairs, 1):
            examples.append(
                f"Example {i}:\n"
                f"Input:\n{grid_to_string(inp)}\n"
                f"Output:\n{grid_to_string(out)}"
            )

        examples_text = '\n\n'.join(examples)

        problem_desc = (
            f"Solve this ARC puzzle:\n\n"
            f"{examples_text}\n\n"
            f"Test Input:\n{grid_to_string(test_input)}\n\n"
            f"What transformation rule do you see? Apply it to predict the output grid."
        )

        return problem_desc

    def plan_step(
        self,
        problem_description: str,
        feedback: Optional[str] = None,
        latent_prefix: Optional[Any] = None,
        use_chat_template: bool = False
    ) -> LatentOutput:
        """
        Execute one planning step (LLM reasoning).

        Args:
            problem_description: Original problem description
            feedback: Optional feedback from previous attempt
            latent_prefix: Optional latent prefix from TRM
            use_chat_template: Use apply_chat_template (for GPT-OSS)

        Returns:
            z_init: LLM hidden state for adapter
            reasoning_text: Generated reasoning text
        """
        self.attempt_count += 1

        # Build prompt with history
        if feedback:
            self.history.append(feedback)

        history_str = "\n".join(self.history) if self.history else ""
        full_prompt = format_problem_text(problem_description, history_str)

        # Generate reasoning
        latent_output = self.llm.generate_latent(
            full_prompt,
            latent_prefix=latent_prefix,
            max_length=self.max_tokens_per_attempt,
            use_chat_template=use_chat_template
        )

        return latent_output

    def multi_attempt_solve(
        self,
        problem_description: str,
        verifier_fn: callable,
        get_latent_prefix_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-attempt solving with self-correction.

        Args:
            problem_description: ARC problem description
            verifier_fn: Function (z_init, attempt) -> (success, feedback, metrics)
            get_latent_prefix_fn: Optional function () -> latent_prefix

        Returns:
            results: Dictionary with success, attempts, reasoning history, metrics
        """
        self.reset()

        results = {
            "success": False,
            "total_attempts": 0,
            "reasoning_history": [],
            "final_feedback": None,
            "metrics": []
        }

        for attempt in range(self.max_attempts):
            # Get latent prefix from TRM if available
            latent_prefix = None
            if get_latent_prefix_fn is not None:
                latent_prefix = get_latent_prefix_fn()

            # Plan step
            latent_output = self.plan_step(
                problem_description,
                feedback=results["final_feedback"],
                latent_prefix=latent_prefix
            )

            results["reasoning_history"].append(latent_output.generated_text)
            results["total_attempts"] = attempt + 1

            # Verify
            success, feedback, metrics = verifier_fn(latent_output.final_hidden, attempt)

            results["metrics"].append(metrics)
            results["final_feedback"] = feedback

            if success:
                results["success"] = True
                break

        return results


class SimpleARCPlanner:
    """Simplified planner for baseline (LLM-only)."""

    def __init__(
        self,
        model_name: str = "unsloth/gpt-oss-mxfp4-20b",
        max_attempts: int = 3,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: LLaMA model name
            max_attempts: Maximum attempts per problem
            device: Device to load model
        """
        self.llm = TextReasoningModule(
            model_name=model_name,
            freeze=True,
            device=device
        )
        self.max_attempts = max_attempts

    def solve_problem(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Dict[str, Any]:
        """
        Solve ARC problem with LLM-only baseline.

        Args:
            train_pairs: Training examples
            test_input: Test input grid
            test_output: Test output grid (for verification)

        Returns:
            results: Solving results with success flag and metrics
        """
        # Create problem description
        planner = ARCPlanner(
            llm_module=self.llm,
            max_attempts=self.max_attempts
        )

        problem_desc = planner.create_problem_description(train_pairs, test_input)

        # Define simple verifier (just checks if LLM mentions correct pattern)
        def simple_verifier(z_init, attempt):
            # Placeholder: real implementation would parse LLM output
            # and compare with test_output
            success = False
            feedback = f"Attempt {attempt + 1} failed. Try again."
            metrics = {"attempt": attempt, "z_init_norm": z_init.norm().item()}
            return success, feedback, metrics

        # Solve
        results = planner.multi_attempt_solve(
            problem_desc,
            verifier_fn=simple_verifier
        )

        return results
