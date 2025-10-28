"""
Integrate HelmARC GPT Analysis with Grid Data

Maps GPT analysis task IDs to actual grid data from Helmholtz samples.

Task ID Format:
- GPT analysis: helmholtz_6aa20dc0_4320
- Helmholtz samples: helmholtz_6aa20dc0_5152 (same hash, different suffix)

The hash (middle part) identifies the original ARC task.
The suffix identifies a specific test case variation.

This script:
1. Loads Helmholtz samples (grid data)
2. Builds a hash -> samples mapping
3. For each GPT analysis task ID, finds matching grid data
4. Saves the integrated dataset
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class GridIntegrator:
    """Integrates GPT analysis with Helmholtz grid data."""

    def __init__(
        self,
        helmholtz_samples_path: str = "/data/dreamcoder-arc/helmholtz_samples/20250512_131838/samples.json",
        verbose: bool = True,
    ):
        """
        Args:
            helmholtz_samples_path: Path to Helmholtz samples.json
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.helmholtz_samples_path = Path(helmholtz_samples_path)

        # Load Helmholtz samples
        if self.verbose:
            print(f"Loading Helmholtz samples from {self.helmholtz_samples_path}...")

        with open(self.helmholtz_samples_path, 'r') as f:
            self.helmholtz_samples = json.load(f)

        if self.verbose:
            print(f"✓ Loaded {len(self.helmholtz_samples)} Helmholtz samples")

        # Build hash -> samples mapping
        self.hash_to_samples = self._build_hash_mapping()

        if self.verbose:
            print(f"✓ Built mapping for {len(self.hash_to_samples)} unique hashes")

    def _build_hash_mapping(self) -> Dict[str, List[Dict]]:
        """Build mapping from hash to all samples with that hash."""
        mapping = defaultdict(list)

        for sample in self.helmholtz_samples:
            task_id = sample['task_id']
            # Extract hash (middle part)
            parts = task_id.split('_')
            if len(parts) == 3:
                hash_part = parts[1]
                mapping[hash_part].append(sample)

        return dict(mapping)

    def find_grid_data(
        self,
        task_id: str,
        prefer_exact_match: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Find grid data for a GPT analysis task ID.

        Args:
            task_id: GPT analysis task ID (e.g., "helmholtz_6aa20dc0_4320")
            prefer_exact_match: Try to find exact task ID match first

        Returns:
            input_grid: [H, W] numpy array
            output_grid: [H, W] numpy array
            metadata: Dict with original task info
        """
        # Parse task ID
        parts = task_id.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid task ID format: {task_id}")

        prefix, hash_part, suffix = parts

        # Try exact match first
        if prefer_exact_match:
            for sample in self.helmholtz_samples:
                if sample['task_id'] == task_id:
                    return self._extract_grids(sample)

        # Try hash match
        if hash_part in self.hash_to_samples:
            samples_with_hash = self.hash_to_samples[hash_part]

            # Try to find with same suffix
            for sample in samples_with_hash:
                if sample['task_id'].endswith(f"_{suffix}"):
                    return self._extract_grids(sample)

            # Use first sample with this hash
            if samples_with_hash:
                sample = samples_with_hash[0]
                if self.verbose:
                    print(f"  Using approximate match: {task_id} -> {sample['task_id']}")
                return self._extract_grids(sample)

        raise ValueError(f"No matching grid data found for task ID: {task_id}")

    def _extract_grids(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Extract input/output grids from Helmholtz sample."""
        # Use first test case (test is a list of test cases)
        test_cases = sample['test']
        if not test_cases:
            raise ValueError(f"No test cases found for {sample['task_id']}")

        # Use first test case
        test = test_cases[0]
        input_grid = np.array(test['input'], dtype=np.int64)
        output_grid = np.array(test['output'], dtype=np.int64)

        # Metadata
        metadata = {
            'task_id': sample['task_id'],
            'original_arc_task_id': sample.get('original_arc_task_id'),
            'program': sample.get('program'),
            'num_examples': len(sample.get('examples', [])),
            'num_test_cases': len(test_cases),
        }

        return input_grid, output_grid, metadata

    def integrate_dataset(
        self,
        analysis_path: str,
        dataset_type: str = "type1",
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Integrate GPT analysis with grid data.

        Args:
            analysis_path: Path to GPT analysis results (type1_results.json or type2_results.json)
            dataset_type: "type1" or "type2"
            output_path: Optional path to save integrated dataset

        Returns:
            List of integrated samples
        """
        if self.verbose:
            print(f"\nIntegrating {dataset_type} dataset from {analysis_path}...")

        # Load GPT analysis
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)

        # Filter successful samples
        successful = [s for s in analysis_data if s.get('status') == 'success']

        if self.verbose:
            print(f"  Found {len(successful)}/{len(analysis_data)} successful samples")

        # Integrate with grid data
        integrated = []
        stats = {'exact_match': 0, 'hash_match': 0, 'not_found': 0}

        for sample in successful:
            task_id = sample['task_id']

            try:
                input_grid, output_grid, metadata = self.find_grid_data(task_id)

                # Build integrated sample
                integrated_sample = {
                    'task_id': task_id,
                    'input_grid': input_grid.tolist(),  # Convert to list for JSON serialization
                    'output_grid': output_grid.tolist(),
                    'grid_shape': list(input_grid.shape),
                    'metadata': metadata,
                }

                # Add analysis data
                if dataset_type == "type1":
                    integrated_sample.update({
                        'program': sample['program'],
                        'explanation': sample['final_response'],
                        'primitive_count': sample.get('primitive_count', 0),
                    })
                else:  # type2
                    integrated_sample.update({
                        'correct_program': sample['correct_program'],
                        'wrong_program': sample['wrong_program'],
                        'explanation': sample['final_response'],
                        'correct_primitive_count': sample.get('correct_primitive_count', 0),
                    })

                integrated.append(integrated_sample)

                # Update stats
                if metadata['task_id'] == task_id:
                    stats['exact_match'] += 1
                else:
                    stats['hash_match'] += 1

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Failed to integrate {task_id}: {e}")
                stats['not_found'] += 1

        if self.verbose:
            print(f"\n✓ Integrated {len(integrated)}/{len(successful)} samples")
            print(f"  Exact matches: {stats['exact_match']}")
            print(f"  Hash matches: {stats['hash_match']}")
            print(f"  Not found: {stats['not_found']}")

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(integrated, f, indent=2)

            if self.verbose:
                print(f"\n✓ Saved integrated dataset to {output_path}")

        return integrated


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrate HelmARC GPT analysis with grid data")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="/home/ubuntu/TinyRecursiveModels/prototype_v2_20251025_150856",
        help="Directory with GPT analysis results"
    )
    parser.add_argument(
        "--helmholtz-samples",
        type=str,
        default="/data/dreamcoder-arc/helmholtz_samples/20250512_131838/samples.json",
        help="Path to Helmholtz samples.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/TinyRecursiveModels/hybrid_pipeline/data/helmarc_integrated",
        help="Output directory for integrated datasets"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["type1", "type2"],
        choices=["type1", "type2"],
        help="Dataset types to integrate"
    )

    args = parser.parse_args()

    # Create integrator
    integrator = GridIntegrator(
        helmholtz_samples_path=args.helmholtz_samples,
        verbose=True
    )

    # Process each type
    for dtype in args.types:
        print(f"\n{'=' * 60}")
        print(f"Integrating {dtype.upper()} Dataset")
        print(f"{'=' * 60}")

        analysis_path = Path(args.analysis_dir) / f"{dtype}_results.json"
        output_path = Path(args.output_dir) / f"{dtype}_integrated.json"

        integrated = integrator.integrate_dataset(
            analysis_path=str(analysis_path),
            dataset_type=dtype,
            output_path=str(output_path)
        )

        print(f"\n✓ {dtype.upper()} integration complete!")
        print(f"  Total samples: {len(integrated)}")

        if integrated:
            print(f"  Example grid shape: {integrated[0]['grid_shape']}")

    print(f"\n{'=' * 60}")
    print("✓ All datasets integrated successfully!")
    print(f"{'=' * 60}")
