"""
ARC-AGI Dataset Loader for Hybrid Model
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

import json
from pathlib import Path

# ARC dataset paths
ARC_TRAIN_CHALLENGES = "/home/ubuntu/barc_feedback/SOAR-main-cleanup/arc-prize-2025/arc-agi_training_challenges.json"
ARC_TRAIN_SOLUTIONS = "/home/ubuntu/barc_feedback/SOAR-main-cleanup/arc-prize-2025/arc-agi_training_solutions.json"
ARC_EVAL_CHALLENGES = "/home/ubuntu/barc_feedback/SOAR-main-cleanup/arc-prize-2025/arc-agi_evaluation_challenges.json"
ARC_EVAL_SOLUTIONS = "/home/ubuntu/barc_feedback/SOAR-main-cleanup/arc-prize-2025/arc-agi_evaluation_solutions.json"


def load_arc_problems(challenges_path: str, solutions_path: str) -> List[Dict]:
    """Load ARC problems from JSON files"""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    # Try to load solutions if they exist
    solutions = {}
    if Path(solutions_path).exists():
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

    problems = []
    for problem_id, problem_data in challenges.items():
        # Each problem has train and test pairs
        problem = {
            'uid': problem_id,
            'train_pairs': [],
            'test_pairs': []
        }

        # Load training pairs
        for pair in problem_data.get('train', []):
            problem['train_pairs'].append({
                'input': np.array(pair['input'], dtype=np.uint8),
                'output': np.array(pair['output'], dtype=np.uint8)
            })

        # Load test pairs
        for i, pair in enumerate(problem_data.get('test', [])):
            test_pair = {
                'input': np.array(pair['input'], dtype=np.uint8),
                'output': None  # Test outputs may not be available
            }

            # Check if solution exists
            if problem_id in solutions and i < len(solutions[problem_id]):
                test_pair['output'] = np.array(solutions[problem_id][i], dtype=np.uint8)
            elif 'output' in pair:  # Some test sets include output
                test_pair['output'] = np.array(pair['output'], dtype=np.uint8)

            problem['test_pairs'].append(test_pair)

        problems.append(problem)

    return problems


def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)


def grid_to_tokens(grid: np.ndarray, max_size: int = 30, pad_value: int = 10, eos_value: int = 11) -> torch.Tensor:
    """
    Convert 2D grid to flattened token sequence

    Args:
        grid: [H, W] numpy array with values 0-9
        max_size: Maximum grid dimension
        pad_value: Padding token ID
        eos_value: End-of-sequence token ID

    Returns:
        tokens: [max_size * max_size] tensor with grid + padding + eos markers
    """
    H, W = grid.shape
    assert H <= max_size and W <= max_size, f"Grid size {H}x{W} exceeds max {max_size}x{max_size}"

    # Create empty grid with padding
    full_grid = np.full((max_size, max_size), pad_value, dtype=np.int64)

    # Place actual grid (add 2 to shift 0-9 -> 2-11, avoiding pad=10 and eos=11)
    # Wait, that conflicts. Let's use: values 0-9, pad=10, eos=11
    # But then we can't distinguish... Let's use standard encoding:
    # PAD=0, EOS=1, colors 0-9 -> 2-11
    full_grid = np.zeros((max_size, max_size), dtype=np.int64)  # PAD=0
    full_grid[:H, :W] = grid + 2  # Shift colors to 2-11

    # Add EOS markers at boundaries
    if H < max_size:
        full_grid[H, :W] = 1  # EOS at bottom boundary
    if W < max_size:
        full_grid[:H, W] = 1  # EOS at right boundary

    # Flatten to sequence
    tokens = torch.from_numpy(full_grid.flatten())
    return tokens


def tokens_to_grid(tokens: torch.Tensor, max_size: int = 30) -> np.ndarray:
    """
    Convert flattened token sequence back to 2D grid

    Args:
        tokens: [max_size * max_size] tensor
        max_size: Grid dimension

    Returns:
        grid: [H, W] numpy array with values 0-9
    """
    # Reshape to 2D
    full_grid = tokens.reshape(max_size, max_size).cpu().numpy()

    # Find actual dimensions by locating EOS markers
    eos_rows = np.where((full_grid == 1).any(axis=1))[0]
    eos_cols = np.where((full_grid == 1).any(axis=0))[0]

    H = eos_rows[0] if len(eos_rows) > 0 else max_size
    W = eos_cols[0] if len(eos_cols) > 0 else max_size

    # Extract actual grid and shift back to 0-9
    grid = full_grid[:H, :W] - 2
    grid = np.clip(grid, 0, 9).astype(np.uint8)

    return grid


class ARCDataset(Dataset):
    """
    ARC-AGI Dataset for hybrid model

    Returns:
        problem_text: Text description of the problem
        problem_grid: Input grid as tokens [900]
        target_grid: Target output grid as tokens [900]
        problem_id: Problem unique identifier
    """

    def __init__(
        self,
        split: str = "train",
        num_problems: Optional[int] = None,
        seed: int = 42,
        augmentation: bool = False
    ):
        self.split = split
        self.augmentation = augmentation
        self.max_size = 30

        # Load problems from JSON files
        if split == "train":
            self.problems = load_arc_problems(ARC_TRAIN_CHALLENGES, ARC_TRAIN_SOLUTIONS)
        elif split == "eval":
            self.problems = load_arc_problems(ARC_EVAL_CHALLENGES, ARC_EVAL_SOLUTIONS)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter out problems without any solvable test outputs
        filtered_problems = []
        for problem in self.problems:
            valid_pairs = [pair for pair in problem['test_pairs'] if pair.get('output') is not None]
            if valid_pairs:
                problem['test_pairs'] = valid_pairs
                filtered_problems.append(problem)
        self.problems = filtered_problems

        # Shuffle and limit by number of unique problems
        random.seed(seed)
        random.shuffle(self.problems)
        if num_problems is not None:
            self.problems = self.problems[:num_problems]

        # Build flat sample index over (problem, test_pair)
        self.samples: List[Tuple[int, int]] = []
        for problem_idx, problem in enumerate(self.problems):
            for test_idx, test_pair in enumerate(problem['test_pairs']):
                if test_pair.get('output') is not None:
                    self.samples.append((problem_idx, test_idx))

        print(
            f"Loaded {len(self.problems)} {split} problems covering {len(self.samples)} test grids"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        problem_idx, test_idx = self.samples[idx]
        problem = self.problems[problem_idx]

        # Format text prompt
        examples = []
        for i, train_pair in enumerate(problem['train_pairs'], 1):
            examples.append(
                f"Example {i}:\n"
                f"Input:\n{grid_to_string(train_pair['input'])}\n"
                f"Output:\n{grid_to_string(train_pair['output'])}"
            )

        test_pair = problem['test_pairs'][test_idx]
        test_input = np.array(test_pair['input'], copy=True)
        test_output = np.array(test_pair['output'], copy=True)

        examples_text = '\n\n'.join(examples)
        problem_text = f"""Solve this ARC puzzle by identifying the transformation rule.

Training Examples:
{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern and provide reasoning:"""

        # Apply augmentation (rotation/flip) if enabled
        if self.augmentation and random.random() < 0.3:
            aug_type = random.choice(['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v'])
            test_input, test_output = self._augment(test_input, test_output, aug_type)

        # Convert grids to tokens
        input_tokens = grid_to_tokens(test_input, self.max_size)
        target_tokens = grid_to_tokens(test_output, self.max_size)

        return {
            'problem_text': problem_text,
            'problem_grid': input_tokens,  # [900]
            'target_grid': target_tokens,   # [900]
            'problem_id': f"{problem['uid']}_test{test_idx}",
            'raw_input': test_input,  # For visualization
            'raw_target': test_output
        }

    def _augment(self, input_grid: np.ndarray, output_grid: np.ndarray, aug_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to both input and output"""
        if aug_type == 'rot90':
            return np.rot90(input_grid), np.rot90(output_grid)
        elif aug_type == 'rot180':
            return np.rot90(input_grid, 2), np.rot90(output_grid, 2)
        elif aug_type == 'rot270':
            return np.rot90(input_grid, 3), np.rot90(output_grid, 3)
        elif aug_type == 'flip_h':
            return np.fliplr(input_grid), np.fliplr(output_grid)
        elif aug_type == 'flip_v':
            return np.flipud(input_grid), np.flipud(output_grid)
        else:
            return input_grid, output_grid


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for ARC dataset

    Since problem_text has variable length, we keep it as a list
    """
    return {
        'problem_text': [item['problem_text'] for item in batch],
        'problem_grid': torch.stack([item['problem_grid'] for item in batch]),
        'target_grid': torch.stack([item['target_grid'] for item in batch]),
        'problem_id': [item['problem_id'] for item in batch],
        'raw_input': [item['raw_input'] for item in batch],
        'raw_target': [item['raw_target'] for item in batch]
    }


def create_arc_dataloaders(
    num_train_problems: int = 10,
    num_val_problems: int = 5,
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Returns:
        train_loader, val_loader
    """
    train_dataset = ARCDataset(
        split="train",
        num_problems=num_train_problems,
        seed=seed,
        augmentation=augmentation
    )

    val_dataset = ARCDataset(
        split="eval",
        num_problems=num_val_problems,
        seed=seed,
        augmentation=False  # No augmentation for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# Test code
if __name__ == "__main__":
    print("Testing ARC DataLoader...")

    # Create dataset
    dataset = ARCDataset(split="train", num_problems=2)

    # Get sample
    sample = dataset[0]
    print(f"\nProblem ID: {sample['problem_id']}")
    print(f"Problem text length: {len(sample['problem_text'])}")
    print(f"Problem grid shape: {sample['problem_grid'].shape}")
    print(f"Target grid shape: {sample['target_grid'].shape}")
    print(f"\nFirst 100 chars of problem text:\n{sample['problem_text'][:100]}...")

    # Test dataloader
    train_loader, val_loader = create_arc_dataloaders(
        num_train_problems=2,
        num_val_problems=1,
        batch_size=1
    )

    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch problem_grid shape: {batch['problem_grid'].shape}")

    print("\n✅ ARC DataLoader test passed!")
