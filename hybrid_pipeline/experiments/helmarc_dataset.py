"""
HelmARC Dataset Loader for Compositional Reasoning Training

Combines:
1. GPT-generated analysis (TYPE 1: correct programs, TYPE 2: wrong+correct programs)
2. Actual grid data from HelmARC dataset

Data sources:
- GPT analysis: /data/helmarc_gpt_analysis_v2/
- Prototype data: /home/ubuntu/TinyRecursiveModels/prototype_v2_20251025_150856/
- Grid data: /data/helmarc_archive/helmarc_trm_v3/ (legacy mapping)
- Original HelmARC: Need to locate actual input/output grids

Usage:
    # Load TYPE 1 (correct programs only)
    dataset = HelmARCDataset(dataset_type="type1", split="train")

    # Load TYPE 2 (wrong + correct programs)
    dataset = HelmARCDataset(dataset_type="type2", split="train")

    # Access samples
    sample = dataset[0]
    # sample = {
    #     'task_id': str,
    #     'program': str,  # DSL program
    #     'explanation': str,  # Natural language explanation
    #     'input_grid': np.ndarray,  # [H, W] input grid
    #     'output_grid': np.ndarray,  # [H, W] output grid
    #     'primitive_count': int,
    #     # For TYPE 2 only:
    #     'wrong_program': str (optional),
    #     'wrong_explanation': str (optional)
    # }
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings


class HelmARCDataset(Dataset):
    """
    Dataset for HelmARC compositional reasoning training.

    Combines GPT-generated program analysis with actual grid transformations.
    """

    def __init__(
        self,
        dataset_type: str = "type1",  # "type1" or "type2"
        split: str = "train",  # "train" or "test"
        data_dir: Optional[str] = None,
        use_integrated: bool = True,  # Use pre-integrated datasets
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_type: "type1" (correct only) or "type2" (wrong+correct)
            split: "train" or "test"
            data_dir: Directory with integrated dataset or GPT analysis results
            use_integrated: Use pre-integrated dataset (grids already loaded)
            max_samples: Limit number of samples (for debugging)
        """
        self.dataset_type = dataset_type
        self.split = split
        self.use_integrated = use_integrated

        # Set default paths
        if data_dir is None:
            if use_integrated:
                data_dir = "/home/ubuntu/TinyRecursiveModels/hybrid_pipeline/data/helmarc_integrated"
            else:
                # Try prototype first, then GPU analysis
                prototype_dir = Path("/home/ubuntu/TinyRecursiveModels/prototype_v2_20251025_150856")
                if prototype_dir.exists():
                    data_dir = str(prototype_dir)
                else:
                    # Use latest GPU analysis results
                    data_dir = self._find_latest_gpu_analysis()

        self.data_dir = Path(data_dir)

        # Load dataset
        self.samples = self._load_data()

        # Limit samples if requested
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"✓ Loaded {len(self.samples)} samples from {dataset_type} ({split})")

    def _find_latest_gpu_analysis(self) -> str:
        """Find the most recent GPU analysis directory."""
        analysis_root = Path("/data/helmarc_gpt_analysis_v2")
        if not analysis_root.exists():
            raise FileNotFoundError(f"GPU analysis directory not found: {analysis_root}")

        # Find all GPU directories
        gpu_dirs = sorted(analysis_root.glob("full_dataset_gpu*"), reverse=True)
        if not gpu_dirs:
            raise FileNotFoundError(f"No GPU analysis directories found in {analysis_root}")

        return str(gpu_dirs[0])

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset (integrated or raw analysis)."""
        if self.use_integrated:
            # Load pre-integrated dataset (grids already included)
            filepath = self.data_dir / f"{self.dataset_type}_integrated.json"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Integrated dataset not found: {filepath}\n"
                    f"Run integrate_helmarc_grids.py to create it."
                )

            with open(filepath, 'r') as f:
                data = json.load(f)

            print(f"  Loaded {len(data)} integrated samples from {filepath.name}")
            return data

        else:
            # Load raw analysis data (will need to load grids separately)
            return self._load_analysis_data()

    def _load_analysis_data(self) -> List[Dict[str, Any]]:
        """Load GPT analysis results."""
        if self.dataset_type == "type1":
            filepath = self.data_dir / "type1_results.json"
            if not filepath.exists():
                # Try checkpoint files from GPU analysis
                filepath = self._find_latest_checkpoint("type1")
        elif self.dataset_type == "type2":
            filepath = self.data_dir / "type2_results.json"
            if not filepath.exists():
                filepath = self._find_latest_checkpoint("type2")
        else:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

        if not filepath.exists():
            raise FileNotFoundError(f"Analysis data not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Filter successful samples only
        successful = [s for s in data if s.get('status') == 'success']

        print(f"  Loaded {len(successful)}/{len(data)} successful samples from {filepath.name}")

        return successful

    def _find_latest_checkpoint(self, dtype: str) -> Path:
        """Find the latest checkpoint file for given type."""
        pattern = f"{dtype}_checkpoint_*.json"
        checkpoints = sorted(self.data_dir.glob(pattern), reverse=True)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files matching {pattern}")
        return checkpoints[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        task_id = sample['task_id']

        # Load grids
        if self.use_integrated:
            # Grids already in the sample (convert from list to numpy)
            input_grid = np.array(sample['input_grid'], dtype=np.int64)
            output_grid = np.array(sample['output_grid'], dtype=np.int64)
        else:
            # Legacy non-integrated mode - not fully implemented
            warnings.warn("Non-integrated mode not fully implemented. Use use_integrated=True")
            input_grid = np.zeros((30, 30), dtype=np.int64)
            output_grid = np.zeros((30, 30), dtype=np.int64)

        # Build return dict
        result = {
            'task_id': task_id,
            'input_grid': input_grid,
            'output_grid': output_grid,
        }

        # Get explanation field (differs between integrated and non-integrated)
        explanation_field = 'explanation' if self.use_integrated else 'final_response'

        if self.dataset_type == "type1":
            # TYPE 1: Correct program only
            result.update({
                'program': sample['program'],
                'explanation': sample.get(explanation_field, ''),
                'primitive_count': sample.get('primitive_count', 0),
            })
        else:
            # TYPE 2: Wrong + Correct program
            result.update({
                'correct_program': sample['correct_program'],
                'wrong_program': sample['wrong_program'],
                'explanation': sample.get(explanation_field, ''),
                'correct_primitive_count': sample.get('correct_primitive_count', 0),
            })

        return result


class HelmARCBatch:
    """
    Batch collator for HelmARC dataset.

    Handles variable-size grids by padding to max size in batch.
    """

    def __init__(self, max_grid_size: int = 30, pad_value: int = 0):
        """
        Args:
            max_grid_size: Maximum grid dimension (will pad to this)
            pad_value: Value to use for padding
        """
        self.max_grid_size = max_grid_size
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Returns:
            {
                'task_ids': List[str],
                'programs': List[str],
                'explanations': List[str],
                'input_grids': torch.Tensor [B, H, W],
                'output_grids': torch.Tensor [B, H, W],
                'grid_sizes': List[Tuple[int, int]],  # Original sizes
                ...
            }
        """
        batch_size = len(batch)

        # Collect metadata
        task_ids = [s['task_id'] for s in batch]

        # Determine max grid size in this batch
        max_h = max(s['input_grid'].shape[0] for s in batch)
        max_w = max(s['input_grid'].shape[1] for s in batch)
        max_h = min(max_h, self.max_grid_size)
        max_w = min(max_w, self.max_grid_size)

        # Pad grids
        input_grids = []
        output_grids = []
        grid_sizes = []

        for s in batch:
            in_grid = s['input_grid']
            out_grid = s['output_grid']

            # Record original size
            grid_sizes.append((in_grid.shape[0], in_grid.shape[1]))

            # Pad to batch max size
            in_padded = np.full((max_h, max_w), self.pad_value, dtype=np.int64)
            out_padded = np.full((max_h, max_w), self.pad_value, dtype=np.int64)

            h, w = in_grid.shape
            in_padded[:h, :w] = in_grid
            out_padded[:h, :w] = out_grid

            input_grids.append(in_padded)
            output_grids.append(out_padded)

        # Convert to tensors
        input_grids = torch.from_numpy(np.stack(input_grids))
        output_grids = torch.from_numpy(np.stack(output_grids))

        # Build result dict
        result = {
            'task_ids': task_ids,
            'input_grids': input_grids,
            'output_grids': output_grids,
            'grid_sizes': grid_sizes,
        }

        # Add type-specific fields
        if 'program' in batch[0]:
            # TYPE 1
            result['programs'] = [s['program'] for s in batch]
            result['explanations'] = [s['explanation'] for s in batch]
            result['primitive_counts'] = [s['primitive_count'] for s in batch]
        else:
            # TYPE 2
            result['correct_programs'] = [s['correct_program'] for s in batch]
            result['wrong_programs'] = [s['wrong_program'] for s in batch]
            result['explanations'] = [s['explanation'] for s in batch]
            result['correct_primitive_counts'] = [s['correct_primitive_count'] for s in batch]

        return result


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("HelmARC Dataset Loader Test")
    print("=" * 60)

    # Test TYPE 1
    print("\n[TEST 1] Loading TYPE 1 dataset...")
    try:
        dataset_type1 = HelmARCDataset(dataset_type="type1", max_samples=5)
        print(f"✓ Loaded {len(dataset_type1)} TYPE 1 samples")

        sample = dataset_type1[0]
        print(f"\nSample 0:")
        print(f"  task_id: {sample['task_id']}")
        print(f"  program: {sample['program']}")
        print(f"  input_grid shape: {sample['input_grid'].shape}")
        print(f"  output_grid shape: {sample['output_grid'].shape}")
        print(f"  explanation: {sample['explanation'][:100]}...")
    except Exception as e:
        print(f"✗ Failed to load TYPE 1: {e}")

    # Test TYPE 2
    print("\n[TEST 2] Loading TYPE 2 dataset...")
    try:
        dataset_type2 = HelmARCDataset(dataset_type="type2", max_samples=5)
        print(f"✓ Loaded {len(dataset_type2)} TYPE 2 samples")

        sample = dataset_type2[0]
        print(f"\nSample 0:")
        print(f"  task_id: {sample['task_id']}")
        print(f"  correct_program: {sample['correct_program']}")
        print(f"  wrong_program: {sample['wrong_program']}")
        print(f"  input_grid shape: {sample['input_grid'].shape}")
    except Exception as e:
        print(f"✗ Failed to load TYPE 2: {e}")

    # Test batch collation
    print("\n[TEST 3] Testing batch collation...")
    try:
        from torch.utils.data import DataLoader

        collator = HelmARCBatch(max_grid_size=30)
        loader = DataLoader(
            dataset_type1,
            batch_size=2,
            collate_fn=collator,
            shuffle=False
        )

        batch = next(iter(loader))
        print(f"✓ Batch created:")
        print(f"  task_ids: {batch['task_ids']}")
        print(f"  input_grids shape: {batch['input_grids'].shape}")
        print(f"  output_grids shape: {batch['output_grids'].shape}")
        print(f"  programs: {len(batch['programs'])} items")
    except Exception as e:
        print(f"✗ Failed batch collation: {e}")

    print("\n" + "=" * 60)
    print("✓ HelmARC Dataset Loader test complete!")
    print("=" * 60)
    print("\nNote: Grid loading is currently using dummy data.")
    print("TODO: Implement actual grid loading from original HelmARC source.")
