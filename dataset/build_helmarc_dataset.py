"""
Build HelmARC dataset in TRM format.

Similar to build_arc_dataset.py but reads HelmARC JSON structure.
"""

from typing import List, Tuple, Dict
import os
import json
import hashlib
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform

cli = ArgParser()

ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5
PuzzleIdSeparator = "|||"


class HelmARCConfig(BaseModel):
    raw_dir: str  # /data/helmarc/raw
    analysis_dir: str  # /data/helmarc/analysis
    output_dir: str  # Output directory
    seed: int = 42
    num_aug: int = 500  # Fewer than real ARC (HelmARC is smaller)
    puzzle_identifiers_start: int = 1


from dataclasses import dataclass

@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])
    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def convert_single_helmarc_puzzle(results: dict, name: str, puzzle: dict, aug_count: int):
    """
    Convert single HelmARC puzzle to TRM format.

    HelmARC structure:
    {
        "examples": [{"input": [...], "output": [...]}, ...],  # Training examples
        "test": [{"input": [...], "output": [...]}, ...]        # Test examples
    }
    """
    # Convert to ARCPuzzle format, filtering out examples with empty outputs
    train_examples = []
    for ex in puzzle["examples"]:
        if ex["output"] and len(ex["output"]) > 0:  # Skip empty outputs
            try:
                train_examples.append((arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"])))
            except AssertionError:
                # Skip invalid grids (too large, wrong dimensions, etc.)
                continue

    test_examples = []
    for ex in puzzle["test"]:
        if ex["output"] and len(ex["output"]) > 0:  # Skip empty outputs
            try:
                test_examples.append((arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"])))
            except AssertionError:
                # Skip invalid grids
                continue

    # Skip puzzles with no valid examples
    if not train_examples or not test_examples:
        return

    # Create puzzle objects for train and test
    train_puzzle = ARCPuzzle(name, train_examples)
    test_puzzle = ARCPuzzle(name, test_examples)

    converted = {
        ("train", "all"): train_puzzle,
        ("test", "all"): test_puzzle
    }

    group = [converted]

    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Augment train only, keep test original for evaluation
            augmented = {
                ("train", "all"): ARCPuzzle(aug_name, [(_map_grid(input), _map_grid(label)) for (input, label) in train_examples]),
                ("test", "all"): ARCPuzzle(name, test_examples)  # Keep original test
            }
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append to results
    for dest in [("train", "all"), ("test", "all")]:
        dest_split, dest_set = dest
        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])

        # Train: add all augmented puzzles
        # Test: add only original puzzle (group[0]) to avoid evaluation distribution skew
        if dest_split == "train":
            results[dest_split][dest_set].append([converted[dest] for converted in group])
        else:  # test
            results[dest_split][dest_set].append([group[0][dest]])


def load_helmarc_puzzles(config: HelmARCConfig):
    """Load HelmARC puzzles from raw directory."""
    results = {}

    raw_dir = Path(config.raw_dir)
    sample_files = sorted(raw_dir.glob("sample_*.json"))

    print(f"Loading {len(sample_files)} HelmARC puzzles...")

    for sample_file in sample_files:
        with open(sample_file) as f:
            puzzle_data = json.load(f)

        task_id = puzzle_data["task_id"]

        # Convert to TRM format with augmentation
        convert_single_helmarc_puzzle(results, task_id, puzzle_data, config.num_aug)

    print(f"Total puzzles loaded: {len(sample_files)}")
    return results


def convert_dataset(config: HelmARCConfig):
    np.random.seed(config.seed)

    # Load HelmARC puzzles
    data = load_helmarc_puzzles(config)

    # Map global puzzle identifiers
    num_identifiers = config.puzzle_identifiers_start
    identifier_map = {}

    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    # Extract base task_id (remove augmentation suffix)
                    # e.g., "helmholtz_arc_dc_0001|||t6|||..." → "helmholtz_arc_dc_0001"
                    base_task_id = puzzle.id.split(PuzzleIdSeparator)[0]
                    if base_task_id not in identifier_map:
                        identifier_map[base_task_id] = num_identifiers
                        num_identifiers += 1

    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Translational augmentations only for train
        enable_translational_augment = split_name == "train"

        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        inp, out = np_grid_to_seq_translational_augment(
                            inp, out,
                            do_translation=enable_translational_augment and _idx_ex != no_aug_id
                        )

                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        example_id += 1
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)

                    # Use base task_id for identifier (same as mapping above)
                    base_task_id = puzzle.id.split(PuzzleIdSeparator)[0]
                    results["puzzle_identifiers"].append(identifier_map[base_task_id])

                    puzzle_id += 1
                    total_puzzles += 1

                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)

                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys())
        )

        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    print(f"✅ HelmARC dataset built successfully!")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total examples: {total_examples}")
    print(f"   Total puzzles: {total_puzzles}")


@cli.command(singleton=True)
def main(config: HelmARCConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
