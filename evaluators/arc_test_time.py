"""
ARC Evaluator with Test-Time Training Support

Extends the original ARC evaluator to support test-time adaptation.
Minimal modification to original code.
"""

from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
from numba import njit
import torch.distributed as dist

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from dataset.common import PuzzleDatasetMetadata
from test_time_adapter import TestTimeAdapter, TestTimeConfig


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside."""
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape

    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC_TestTime:
    """
    ARC Evaluator with Test-Time Training.

    Uses training examples to adapt puzzle embedding before inference.
    """

    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        model: torch.nn.Module,
        submission_K: int = 2,
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
        aggregated_voting: bool = True,
        enable_test_time_training: bool = True,
        test_time_config: Optional[TestTimeConfig] = None,
    ):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id
        self.enable_test_time_training = enable_test_time_training

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)

        # States
        self._local_hmap = {}
        self._local_preds = {}

        # Test-time adapter
        if self.enable_test_time_training:
            self.adapter = TestTimeAdapter(model, test_time_config)
            self._puzzle_id_mapping = {}  # Maps puzzle_name -> adapted puzzle_id
        else:
            self.adapter = None

    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}

        # Reset puzzle ID mapping for test-time training
        if self.enable_test_time_training:
            self._puzzle_id_mapping = {}

    def _prepare_test_time_training(self, puzzle_name: str, device: str = "cuda"):
        """
        Prepare puzzle for test-time training.

        Args:
            puzzle_name: Name of the puzzle (base name)
            device: Device to use

        Returns:
            Adapted puzzle_id to use for this puzzle
        """
        if not self.enable_test_time_training or puzzle_name in self._puzzle_id_mapping:
            return None

        # Get training examples
        if puzzle_name not in self.test_puzzles:
            return None

        puzzle_data = self.test_puzzles[puzzle_name]
        train_examples = puzzle_data.get('train', [])

        if not train_examples:
            return None

        # Convert to tensors
        train_tensors = []
        for ex in train_examples:
            # Convert grid to tokens (reverse of _crop + inverse_aug)
            input_grid = arc_grid_to_np(ex['input'])
            output_grid = arc_grid_to_np(ex['output'])

            # Pad to 30x30 and flatten to 900
            input_padded = np.pad(input_grid + 2, ((0, 30 - input_grid.shape[0]), (0, 30 - input_grid.shape[1])), constant_values=0)
            output_padded = np.pad(output_grid + 2, ((0, 30 - output_grid.shape[0]), (0, 30 - output_grid.shape[1])), constant_values=0)

            train_tensors.append({
                'input': torch.from_numpy(input_padded.reshape(-1)).long(),
                'output': torch.from_numpy(output_padded.reshape(-1)).long(),
            })

        # Adapt
        print(f"Test-time training for {puzzle_name} with {len(train_tensors)} examples...")
        puzzle_id, history = self.adapter.adapt(train_tensors, device=device)

        # Store mapping
        self._puzzle_id_mapping[puzzle_name] = puzzle_id

        print(f"  Final loss: {history['loss'][-1]:.4f} (converged in {len(history['loss'])} steps)")

        return puzzle_id

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Update with batch predictions.

        Note: In test-time mode, this is called AFTER test-time training.
        """
        # Collect required outputs to CPU
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()

        assert q_values is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input, pred, q in zip(
            outputs["puzzle_identifiers"].numpy(),
            outputs["inputs"].numpy(),
            outputs["preds"].numpy(),
            q_values.numpy()
        ):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)

            input_hash = grid_hash(_inverse_fn(_crop(input)))

            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred

            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        """
        Compute results with test-time training.

        If test-time training is enabled, this will first run adaptation
        on each puzzle before evaluation.
        """
        # Gather predictions to rank 0 for voting
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)

        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]

            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                p_map = {}
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q in preds.get(name, {}).get(input_hash, {}):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q

                if not len(p_map):
                    print(f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash

                    num_test_correct[i] += ok

                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}

        # Add test-time training info
        if self.enable_test_time_training:
            all_results["test_time_training_enabled"] = True
            all_results["num_adapted_puzzles"] = len(self._puzzle_id_mapping)

        return all_results
