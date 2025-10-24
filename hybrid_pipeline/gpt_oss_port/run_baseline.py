"""
Baseline Runner - LLM-only baseline without TRM

CLI tool to run simple LLM baseline on ARC dataset.
Uses dataset_access to load original ARC problems.
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

# Add hybrid_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt_oss_port.planner import SimpleARCPlanner
from gpt_oss_port.grid_utils import grid_to_string
from gpt_oss_port.dataset_access import load_puzzles_arcagi, DataProcessConfig, ARCPuzzle


def load_arc_problems(
    arc_json_prefix: str,
    subset: str = "evaluation",
    num_problems: int = 10
) -> List[dict]:
    """
    Load ARC problems using dataset_access.

    Args:
        arc_json_prefix: Path prefix to ARC JSON files (e.g., /path/to/arc_agi)
        subset: Subset name (training, evaluation, etc.)
        num_problems: Number of problems to load

    Returns:
        problems: List of problem dictionaries with train/test pairs
    """
    # Create config for load_puzzles_arcagi
    config = DataProcessConfig(
        input_file_prefix=arc_json_prefix,
        output_dir="",  # Not used for loading only
        subsets=[subset],
        test_set_name=subset,
        test_set_name2="",
        num_aug=0
    )

    # Load puzzles
    print(f"Loading puzzles from {arc_json_prefix}_{subset}_challenges.json...")
    all_puzzles, test_puzzles = load_puzzles_arcagi(config)

    # Convert to planner format
    problems = []
    for puzzle_id, puzzle in list(all_puzzles[subset].items())[:num_problems]:
        # Extract train pairs
        train_pairs = [
            (np.array(ex.input_grid, dtype=np.uint8),
             np.array(ex.output_grid, dtype=np.uint8))
            for ex in puzzle.train
        ]

        # Extract test pairs (use first test example)
        if puzzle.test:
            test_input = np.array(puzzle.test[0].input_grid, dtype=np.uint8)
            test_output = np.array(puzzle.test[0].output_grid, dtype=np.uint8)

            problems.append({
                "id": puzzle_id,
                "train": train_pairs,
                "test_input": test_input,
                "test_output": test_output
            })

    return problems


def run_baseline(
    arc_json_prefix: str,
    subset: str = "evaluation",
    model_name: str = "unsloth/gpt-oss-mxfp4-20b",
    max_attempts: int = 3,
    num_problems: int = 10,
    output_path: Optional[str] = None,
    device: str = "cuda"
):
    """
    Run LLM-only baseline on ARC problems.

    Args:
        arc_json_prefix: Path prefix to ARC JSON files (e.g., /path/to/arc_agi)
        subset: ARC subset (training, evaluation, etc.)
        model_name: LLaMA model name
        max_attempts: Max attempts per problem
        num_problems: Number of problems to evaluate
        output_path: Optional output JSON path
        device: Device to use
    """
    print("=" * 60)
    print("ARC Baseline Runner (LLM-only)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Subset: {subset}")
    print(f"Max attempts: {max_attempts}")
    print(f"Problems: {num_problems}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load problems using dataset_access
    print(f"\n📦 Loading problems from {arc_json_prefix}...")
    problems = load_arc_problems(arc_json_prefix, subset, num_problems)

    if not problems:
        print("❌ No problems loaded. Check data path.")
        return

    print(f"✅ Loaded {len(problems)} problems")

    # Create planner
    print(f"\n🔧 Initializing planner...")
    planner = SimpleARCPlanner(
        model_name=model_name,
        max_attempts=max_attempts,
        device=device
    )

    # Run evaluation
    results = []
    correct_count = 0

    for i, problem in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(problems)}: {problem['id']}")
        print(f"{'='*60}")

        # Solve
        result = planner.solve_problem(
            train_pairs=problem["train"],
            test_input=problem["test_input"],
            test_output=problem["test_output"]
        )

        result["problem_id"] = problem["id"]
        results.append(result)

        if result["success"]:
            correct_count += 1
            print(f"✅ CORRECT (attempt {result['total_attempts']})")
        else:
            print(f"❌ FAILED ({result['total_attempts']} attempts)")

    # Summary
    accuracy = correct_count / len(problems) if problems else 0
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {len(problems)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"{'='*60}")

    # Save results
    if output_path:
        output_data = {
            "config": {
                "model_name": model_name,
                "subset": subset,
                "max_attempts": max_attempts,
                "num_problems": num_problems,
                "arc_json_prefix": arc_json_prefix
            },
            "summary": {
                "total": len(problems),
                "correct": correct_count,
                "accuracy": accuracy
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ARC LLM-only Baseline using dataset_access"
    )
    parser.add_argument(
        "--arc_json_prefix", type=str, required=True,
        help="Path prefix to ARC JSON files (e.g., /path/to/arc_agi)"
    )
    parser.add_argument(
        "--subset", type=str, default="evaluation",
        help="ARC subset (training, evaluation, etc.)"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="unsloth/gpt-oss-mxfp4-20b",
        help="HF model name"
    )
    parser.add_argument(
        "--max_attempts", type=int, default=3,
        help="Maximum attempts per problem"
    )
    parser.add_argument(
        "--num_problems", type=int, default=10,
        help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )

    args = parser.parse_args()

    run_baseline(
        arc_json_prefix=args.arc_json_prefix,
        subset=args.subset,
        model_name=args.model_name,
        max_attempts=args.max_attempts,
        num_problems=args.num_problems,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
