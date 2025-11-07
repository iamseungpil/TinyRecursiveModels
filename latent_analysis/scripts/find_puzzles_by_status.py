"""
Helper script to find solved/unsolved puzzles for testing.

Reads the existing latents.json and identifies puzzle indices by solve status.
Useful for selecting diverse test cases for step-by-step visualization.

Usage:
    python find_puzzles_by_status.py --num_solved 5 --num_unsolved 5
"""

import json
import argparse
import random
from pathlib import Path


def load_latents(latents_path: str):
    """Load latent results and categorize by solve status."""
    print(f"📥 Loading latents from {latents_path}...")

    with open(latents_path, 'r') as f:
        results = json.load(f)

    # Categorize by solve status (using test examples only)
    solved_puzzles = {}
    unsolved_puzzles = {}

    for result in results:
        if 'test' not in result['task_id']:
            continue  # Skip training examples

        puzzle_id = result['puzzle_id']
        is_solved = result['solved']

        if is_solved:
            if puzzle_id not in solved_puzzles:
                solved_puzzles[puzzle_id] = result
        else:
            if puzzle_id not in unsolved_puzzles:
                unsolved_puzzles[puzzle_id] = result

    print(f"✅ Found:")
    print(f"   Solved puzzles: {len(solved_puzzles)}")
    print(f"   Unsolved puzzles: {len(unsolved_puzzles)}")

    return solved_puzzles, unsolved_puzzles


def find_puzzle_index_from_id(data_path: str, puzzle_id: int):
    """Find puzzle index from puzzle_id."""
    import numpy as np

    puzzle_identifiers = np.load(f"{data_path}/test/all__puzzle_identifiers.npy")

    for idx, pid in enumerate(puzzle_identifiers):
        if pid == puzzle_id:
            return idx

    return None


def main():
    parser = argparse.ArgumentParser(description="Find puzzles by solve status")
    parser.add_argument("--latents", type=str,
                        default="/home/ubuntu/TinyRecursiveModels/latent_analysis/data/latents.json",
                        help="Path to latents.json")
    parser.add_argument("--data_path", type=str,
                        default="/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--num_solved", type=int, default=5,
                        help="Number of solved puzzles to select")
    parser.add_argument("--num_unsolved", type=int, default=5,
                        help="Number of unsolved puzzles to select")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selection")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (optional)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load latents
    solved_puzzles, unsolved_puzzles = load_latents(args.latents)

    # Select random samples
    solved_ids = random.sample(list(solved_puzzles.keys()), min(args.num_solved, len(solved_puzzles)))
    unsolved_ids = random.sample(list(unsolved_puzzles.keys()), min(args.num_unsolved, len(unsolved_puzzles)))

    # Map to indices
    print(f"\n🔍 Finding puzzle indices...")
    selected = {
        'solved': [],
        'unsolved': []
    }

    for puzzle_id in solved_ids:
        idx = find_puzzle_index_from_id(args.data_path, puzzle_id)
        if idx is not None:
            result = solved_puzzles[puzzle_id]
            selected['solved'].append({
                'puzzle_idx': int(idx),
                'puzzle_id': int(puzzle_id),
                'puzzle_name': result['puzzle_name'],
                'task_id': result['task_id']
            })

    for puzzle_id in unsolved_ids:
        idx = find_puzzle_index_from_id(args.data_path, puzzle_id)
        if idx is not None:
            result = unsolved_puzzles[puzzle_id]
            selected['unsolved'].append({
                'puzzle_idx': int(idx),
                'puzzle_id': int(puzzle_id),
                'puzzle_name': result['puzzle_name'],
                'task_id': result['task_id']
            })

    # Display results
    print(f"\n✅ Selected Puzzles:")
    print(f"\n{'='*80}")
    print("SOLVED PUZZLES:")
    print(f"{'='*80}")
    for i, puzzle in enumerate(selected['solved'], 1):
        print(f"{i}. puzzle_idx={puzzle['puzzle_idx']:3d}  |  {puzzle['puzzle_name']}")

    print(f"\n{'='*80}")
    print("UNSOLVED PUZZLES:")
    print(f"{'='*80}")
    for i, puzzle in enumerate(selected['unsolved'], 1):
        print(f"{i}. puzzle_idx={puzzle['puzzle_idx']:3d}  |  {puzzle['puzzle_name']}")

    # Command examples
    print(f"\n{'='*80}")
    print("EXAMPLE COMMANDS:")
    print(f"{'='*80}")
    print("\n# Run on all solved puzzles:")
    for puzzle in selected['solved']:
        print(f"python latent_analysis/scripts/step_by_step_inference_poc.py --puzzle_idx {puzzle['puzzle_idx']}")

    print("\n# Run on all unsolved puzzles:")
    for puzzle in selected['unsolved']:
        print(f"python latent_analysis/scripts/step_by_step_inference_poc.py --puzzle_idx {puzzle['puzzle_idx']}")

    print("\n# Batch script:")
    print("#!/bin/bash")
    all_indices = [p['puzzle_idx'] for p in selected['solved']] + [p['puzzle_idx'] for p in selected['unsolved']]
    print(f"for idx in {' '.join(map(str, all_indices))}; do")
    print("    python latent_analysis/scripts/step_by_step_inference_poc.py \\")
    print("        --puzzle_idx $idx \\")
    print("        --output_dir latent_analysis/results/batch/puzzle_$idx")
    print("done")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(selected, f, indent=2)

        print(f"\n💾 Saved selection to: {output_path}")

    print(f"\n{'='*80}")
    print(f"Total selected: {len(selected['solved'])} solved + {len(selected['unsolved'])} unsolved")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
