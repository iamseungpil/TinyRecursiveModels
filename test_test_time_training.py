"""
Test script for test-time training implementation.

Tests the test-time adapter on a small subset of puzzles.
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from test_time_adapter import TestTimeAdapter, TestTimeConfig
from dataset.build_arc_dataset import inverse_aug, arc_grid_to_np
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_model(checkpoint_path: str, device: str):
    """Load TRM model."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    config = {
        "batch_size": 1,
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 876406,
        "puzzle_emb_ndim": 512,
        "puzzle_emb_len": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
        "L_cycles": 4,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
        "no_ACT_continue": True,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, config


def test_adapter():
    """Test the adapter on a few puzzles."""
    device = "cuda"
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    data_path = "/data/arc1concept-aug-1000"

    print("=" * 80)
    print("Test-Time Training Implementation Test")
    print("=" * 80)

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Create adapter
    # Use ID outside training range to avoid conflicts
    test_time_config = TestTimeConfig(
        reserved_puzzle_id=876410,
        learning_rate=1e-3,
        max_steps=20,
        patience=3
    )
    adapter = TestTimeAdapter(model, test_time_config)

    # Load test puzzles
    with open(os.path.join(data_path, "test_puzzles.json"), 'r') as f:
        test_puzzles = json.load(f)

    # Test on first 3 puzzles
    puzzle_names = list(test_puzzles.keys())[:3]

    print(f"\nTesting on {len(puzzle_names)} puzzles...")

    for puzzle_name in puzzle_names:
        print(f"\n{'-' * 80}")
        print(f"Puzzle: {puzzle_name}")
        print(f"{'-' * 80}")

        puzzle_data = test_puzzles[puzzle_name]

        # Prepare training examples
        train_examples = []
        for ex in puzzle_data['train']:
            input_grid = arc_grid_to_np(ex['input'])
            output_grid = arc_grid_to_np(ex['output'])

            # Get original size
            h, w = output_grid.shape

            # Pad to 30x30 with PAD=0, colors+2
            input_padded = np.pad(
                input_grid + 2,
                ((0, 30 - h), (0, 30 - w)),
                constant_values=0
            )

            # Add EOS tokens on boundaries (match dataset encoding)
            eos_row, eos_col = h, w
            if eos_row < 30:  # Bottom edge
                input_padded[eos_row, 0:eos_col] = 1
            if eos_col < 30:  # Right edge
                input_padded[0:eos_row, eos_col] = 1

            # Output: pad with -100 to ignore padding in loss calculation
            output_padded = np.full((30, 30), -100, dtype=np.int64)
            output_padded[:h, :w] = output_grid + 2

            train_examples.append({
                'input': torch.from_numpy(input_padded.reshape(-1)).long(),
                'output': torch.from_numpy(output_padded.reshape(-1)).long(),
            })

        print(f"Training examples: {len(train_examples)}")

        # Adapt
        try:
            puzzle_id, history = adapter.adapt(train_examples, device=device)

            print(f"✅ Adaptation successful!")
            print(f"   Puzzle ID: {puzzle_id}")
            print(f"   Steps: {len(history['loss'])}")
            print(f"   Final loss: {history['loss'][-1]:.4f}")
            print(f"   Loss progression: {history['loss'][:5]}... → {history['loss'][-1]:.4f}")

        except Exception as e:
            print(f"❌ Adaptation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_adapter()
