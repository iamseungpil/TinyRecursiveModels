"""
Extract (input_grid, puzzle_embedding) pairs from TRM checkpoint

This script:
1. Loads learned puzzle embeddings from TRM checkpoint
2. Extracts input grids from dataset for each puzzle
3. Creates training pairs: grid → embedding
4. Saves as PyTorch dataset for CNN training

Usage:
    python extract_training_pairs.py \
        --checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
        --data-path /data/arc1concept-aug-1000 \
        --output-dir ./data/training_pairs \
        --max-examples-per-puzzle 10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop


def load_puzzle_embeddings(checkpoint_path: str) -> torch.Tensor:
    """
    Load learned puzzle embeddings from TRM checkpoint.

    Returns:
        embeddings: (num_puzzles, 512) tensor of learned embeddings
    """
    print(f"\n📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try different key names
    possible_keys = [
        'model.inner.puzzle_emb.weights',
        '_orig_mod.model.inner.puzzle_emb.weights',
        'puzzle_emb.weights'
    ]

    puzzle_emb = None
    for key in possible_keys:
        if key in checkpoint:
            puzzle_emb = checkpoint[key]
            print(f"✅ Found puzzle embeddings: key='{key}'")
            break

    if puzzle_emb is None:
        # Try state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            for key in possible_keys:
                if key in state_dict:
                    puzzle_emb = state_dict[key]
                    print(f"✅ Found puzzle embeddings in state_dict: key='{key}'")
                    break

    if puzzle_emb is None:
        available_keys = list(checkpoint.keys())[:10]
        raise KeyError(f"Puzzle embedding not found. Available keys: {available_keys}")

    print(f"✅ Loaded embeddings: shape={puzzle_emb.shape}")
    print(f"   Embedding dim: {puzzle_emb.shape[1]}")
    print(f"   Num puzzles: {puzzle_emb.shape[0]}")

    return puzzle_emb


def load_dataset_metadata(data_path: str) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset metadata and memory-mapped arrays.

    Returns:
        identifier_map: {puzzle_id: puzzle_name}
        inputs_mmap: Input grids
        labels_mmap: Output grids
        puzzle_identifiers: Puzzle IDs for each example
        puzzle_indices: Start/end indices for each puzzle
    """
    print(f"\n📥 Loading dataset from {data_path}")

    # Load identifier map
    identifiers_path = os.path.join(data_path, "identifiers.json")
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}
    print(f"✅ Loaded {len(identifier_map)} identifiers")

    # Load train set (has more examples per puzzle)
    train_dir = os.path.join(data_path, 'train')
    inputs_mmap = np.load(os.path.join(train_dir, 'all__inputs.npy'), mmap_mode='r')
    labels_mmap = np.load(os.path.join(train_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(train_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(train_dir, 'all__puzzle_indices.npy'))

    print(f"✅ Dataset loaded:")
    print(f"   Total examples: {len(inputs_mmap)}")
    print(f"   Unique puzzles: {len(puzzle_identifiers)}")
    print(f"   Grid shape: {inputs_mmap.shape}")

    return identifier_map, inputs_mmap, labels_mmap, puzzle_identifiers, puzzle_indices


def extract_training_pairs(
    puzzle_embeddings: torch.Tensor,
    identifier_map: Dict,
    inputs_mmap: np.ndarray,
    puzzle_identifiers: np.ndarray,
    puzzle_indices: np.ndarray,
    max_examples_per_puzzle: int = 10,
    min_examples_per_puzzle: int = 1
) -> List[Dict]:
    """
    Extract (grid, embedding) pairs from dataset.

    Args:
        puzzle_embeddings: (num_puzzles, 512) learned embeddings
        identifier_map: Puzzle ID to name mapping
        inputs_mmap: Input grids
        puzzle_identifiers: Puzzle IDs
        puzzle_indices: Start/end indices for each puzzle
        max_examples_per_puzzle: Max training examples per puzzle
        min_examples_per_puzzle: Min examples to include puzzle

    Returns:
        List of dicts with 'grid', 'embedding', 'puzzle_id', 'puzzle_name'
    """
    print(f"\n🔬 Extracting training pairs...")
    print(f"   Max examples per puzzle: {max_examples_per_puzzle}")
    print(f"   Min examples per puzzle: {min_examples_per_puzzle}")

    training_pairs = []
    skipped_puzzles = 0
    zero_norm_puzzles = 0

    for puzzle_idx in tqdm(range(len(puzzle_identifiers)), desc="Processing puzzles"):
        puzzle_id = puzzle_identifiers[puzzle_idx]
        puzzle_name = identifier_map.get(puzzle_id, f"unknown_{puzzle_id}")

        # Get puzzle embedding
        if puzzle_id >= len(puzzle_embeddings):
            skipped_puzzles += 1
            continue

        embedding = puzzle_embeddings[puzzle_id].numpy()

        # Skip zero embeddings (padding/unused)
        if np.linalg.norm(embedding) < 1e-6:
            zero_norm_puzzles += 1
            continue

        # Get all examples for this puzzle
        start_idx = puzzle_indices[puzzle_idx]
        end_idx = puzzle_indices[puzzle_idx + 1]
        num_examples = end_idx - start_idx

        if num_examples < min_examples_per_puzzle:
            skipped_puzzles += 1
            continue

        # Sample examples (limit to max_examples_per_puzzle)
        if num_examples > max_examples_per_puzzle:
            # Randomly sample
            example_indices = np.random.choice(
                range(start_idx, end_idx),
                size=max_examples_per_puzzle,
                replace=False
            )
        else:
            example_indices = range(start_idx, end_idx)

        # Extract grids
        for example_idx in example_indices:
            input_grid = inputs_mmap[example_idx]

            # Crop to actual size (remove padding)
            input_cropped = _crop(input_grid)

            training_pairs.append({
                'grid': input_cropped,
                'embedding': embedding,
                'puzzle_id': int(puzzle_id),
                'puzzle_name': puzzle_name,
                'example_idx': int(example_idx)
            })

    print(f"\n📊 Extraction results:")
    print(f"   Total pairs: {len(training_pairs)}")
    print(f"   Unique puzzles: {len(set(p['puzzle_id'] for p in training_pairs))}")
    print(f"   Skipped (insufficient examples): {skipped_puzzles}")
    print(f"   Skipped (zero norm embedding): {zero_norm_puzzles}")

    return training_pairs


def save_training_pairs(training_pairs: List[Dict], output_dir: str):
    """Save training pairs as PyTorch dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as torch file (more efficient for loading)
    torch_path = output_path / "training_pairs.pt"
    torch.save(training_pairs, torch_path)
    print(f"\n💾 Saved PyTorch dataset: {torch_path}")
    print(f"   Size: {torch_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata
    metadata = {
        'num_pairs': len(training_pairs),
        'num_puzzles': len(set(p['puzzle_id'] for p in training_pairs)),
        'embedding_dim': training_pairs[0]['embedding'].shape[0] if training_pairs else 0,
        'sample_puzzles': [p['puzzle_name'] for p in training_pairs[:10]]
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract training pairs for CNN puzzle embedding predictor")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
        help="Path to TRM checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/arc1concept-aug-1000",
        help="Path to ARC dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/TinyRecursiveModels/puzzle_embedding_predictor/data/training_pairs",
        help="Output directory for training pairs"
    )
    parser.add_argument(
        "--max-examples-per-puzzle",
        type=int,
        default=10,
        help="Maximum training examples per puzzle"
    )
    parser.add_argument(
        "--min-examples-per-puzzle",
        type=int,
        default=1,
        help="Minimum examples to include puzzle"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*70)
    print("Extract Training Pairs for Puzzle Embedding Predictor")
    print("="*70)

    # Load puzzle embeddings from checkpoint
    puzzle_embeddings = load_puzzle_embeddings(args.checkpoint)

    # Load dataset
    identifier_map, inputs_mmap, labels_mmap, puzzle_identifiers, puzzle_indices = \
        load_dataset_metadata(args.data_path)

    # Extract pairs
    training_pairs = extract_training_pairs(
        puzzle_embeddings,
        identifier_map,
        inputs_mmap,
        puzzle_identifiers,
        puzzle_indices,
        max_examples_per_puzzle=args.max_examples_per_puzzle,
        min_examples_per_puzzle=args.min_examples_per_puzzle
    )

    # Save
    save_training_pairs(training_pairs, args.output_dir)

    print("\n" + "="*70)
    print("✅ Extraction complete!")
    print("="*70)


if __name__ == "__main__":
    main()
