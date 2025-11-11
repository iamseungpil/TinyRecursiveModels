"""
Pre-compute puzzle features for K-NN retrieval.

This script:
1. Loads all available puzzles (from test_puzzles.json)
2. Computes color histogram features for each
3. Creates mapping from puzzle names to features
4. Saves for fast K-NN lookup
"""

import os
import json
import numpy as np
from tqdm import tqdm
import pickle

from dataset.build_arc_dataset import arc_grid_to_np
from similarity_metrics import precompute_color_histograms


def extract_puzzle_name_from_identifier(identifier: str) -> str:
    """
    Extract original puzzle name from augmented identifier.

    Example: "8be77c9e|||t7|||0612397845" → "8be77c9e"
    """
    if '|||' in identifier:
        return identifier.split('|||')[0]
    return identifier


def load_identifiers(data_path: str):
    """Load puzzle identifiers and create mapping."""
    identifiers_path = os.path.join(data_path, "identifiers.json")

    with open(identifiers_path, 'r') as f:
        identifiers = json.load(f)

    # Create mapping: puzzle_name → [puzzle_ids]
    puzzle_name_to_ids = {}

    for puzzle_id, identifier in enumerate(identifiers):
        if identifier == "<blank>":
            continue

        puzzle_name = extract_puzzle_name_from_identifier(identifier)

        if puzzle_name not in puzzle_name_to_ids:
            puzzle_name_to_ids[puzzle_name] = []

        puzzle_name_to_ids[puzzle_name].append(puzzle_id)

    return puzzle_name_to_ids


def main():
    data_path = '/data/arc1concept-aug-1000'

    print("=" * 80)
    print("Pre-computing Puzzle Features for K-NN Retrieval")
    print("=" * 80)

    # Load puzzle name to ID mapping
    print("\nLoading puzzle identifiers...")
    puzzle_name_to_ids = load_identifiers(data_path)
    print(f"Found {len(puzzle_name_to_ids)} unique puzzle names")
    print(f"Total puzzle IDs: {sum(len(ids) for ids in puzzle_name_to_ids.values())}")

    # Load test puzzles
    print("\nLoading test puzzles...")
    test_puzzles_path = os.path.join(data_path, "test_puzzles.json")

    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)

    print(f"Loaded {len(test_puzzles)} test puzzles")

    # Compute features for each puzzle
    print("\nComputing color histogram features...")
    puzzle_features = {}
    puzzle_to_puzzle_ids = {}

    for puzzle_name, puzzle_data in tqdm(test_puzzles.items(), desc="Processing puzzles"):
        # Convert train examples to numpy
        train_examples = []
        for ex in puzzle_data['train']:
            input_grid = arc_grid_to_np(ex['input'])
            output_grid = arc_grid_to_np(ex['output'])
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })

        # Compute average color histogram
        avg_histogram = precompute_color_histograms(train_examples)
        puzzle_features[puzzle_name] = avg_histogram

        # Store puzzle IDs for embedding lookup
        if puzzle_name in puzzle_name_to_ids:
            # Use first puzzle ID as representative
            puzzle_to_puzzle_ids[puzzle_name] = puzzle_name_to_ids[puzzle_name][0]
        else:
            # Puzzle not in training set, use hash as fallback
            puzzle_to_puzzle_ids[puzzle_name] = hash(puzzle_name) % 876406

    print(f"\nComputed features for {len(puzzle_features)} puzzles")

    # Count how many have valid puzzle IDs
    valid_ids = sum(1 for name in puzzle_features
                   if name in puzzle_name_to_ids)
    print(f"Puzzles with valid training IDs: {valid_ids}/{len(puzzle_features)}")

    # Save features
    output_dir = '/data/arc1concept-aug-1000'
    features_path = os.path.join(output_dir, 'puzzle_features.pkl')
    ids_path = os.path.join(output_dir, 'puzzle_to_ids.pkl')

    with open(features_path, 'wb') as f:
        pickle.dump(puzzle_features, f)

    with open(ids_path, 'wb') as f:
        pickle.dump(puzzle_to_puzzle_ids, f)

    print(f"\nSaved features to: {features_path}")
    print(f"Saved ID mapping to: {ids_path}")

    # Print sample
    print("\nSample features:")
    for i, (name, hist) in enumerate(list(puzzle_features.items())[:5]):
        puzzle_id = puzzle_to_puzzle_ids.get(name, -1)
        hist_str = ' '.join(f'{h:.3f}' for h in hist[:5])
        print(f"  {name}: ID={puzzle_id}, histogram=[{hist_str}...]")

    print("\n" + "=" * 80)
    print("Feature pre-computation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
