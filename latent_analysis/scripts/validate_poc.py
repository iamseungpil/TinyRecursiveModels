"""
Validation script for step-by-step inference POC.

This script verifies that our custom inference loop produces identical
final predictions to the reference implementation (extract_latents_corrected.py).

Usage:
    python validate_poc.py --puzzle_idx 0

This will:
1. Run step-by-step inference
2. Extract final prediction
3. Compare with reference implementation
4. Report pass/fail with detailed diagnostics
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_checkpoint(checkpoint_path: str, device: str):
    """Load TRM model."""
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
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


def load_test_puzzle(data_path: str, puzzle_idx: int, identifier_map: dict):
    """Load a single test puzzle."""
    test_dir = os.path.join(data_path, 'test')

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    start_idx = puzzle_indices[puzzle_idx]
    end_idx = puzzle_indices[puzzle_idx + 1]

    batch = {
        'inputs': torch.from_numpy(inputs[start_idx:end_idx].astype(np.int64)),
        'labels': torch.from_numpy(labels[start_idx:end_idx].astype(np.int64)),
        'puzzle_identifiers': torch.full(
            (end_idx - start_idx,),
            puzzle_identifiers[puzzle_idx],
            dtype=torch.long
        ),
        'puzzle_id': int(puzzle_identifiers[puzzle_idx]),
        'puzzle_name': identifier_map.get(puzzle_identifiers[puzzle_idx], f"<unknown>"),
    }

    return batch


def reference_inference(model, batch, device):
    """
    Reference implementation using official ACT wrapper.
    This is the "ground truth" we're validating against.
    """
    inputs = batch['inputs'].to(device)
    labels = batch['labels'].to(device)
    puzzle_ids = batch['puzzle_identifiers'].to(device)

    batch_dict = {
        'inputs': inputs,
        'labels': labels,
        'puzzle_identifiers': puzzle_ids
    }

    with torch.no_grad():
        carry = model.initial_carry(batch_dict)
        carry, outputs = model(carry, batch_dict)

    pred_tokens = outputs['logits'].argmax(dim=-1)
    return pred_tokens.cpu()


def custom_inference(model, batch, device, max_h_steps=3):
    """
    Custom implementation bypassing ACT wrapper.
    This is what we're testing.
    """
    inputs = batch['inputs'].to(device)
    puzzle_ids = batch['puzzle_identifiers'].to(device)

    batch_size = inputs.shape[0]
    inner = model.inner

    # Initialize carry
    z_H = inner.H_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)
    z_L = inner.L_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, model.config.seq_len + inner.puzzle_emb_len, -1)

    seq_info = dict(
        cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
    )

    input_embeddings = inner._input_embeddings(inputs, puzzle_ids)

    with torch.no_grad():
        for h_step in range(max_h_steps):
            for l_step in range(model.config.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)

        # Final output
        output_logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
        pred_tokens = output_logits.argmax(dim=-1)

    return pred_tokens.cpu()


def tokens_to_grid(tokens: torch.Tensor, puzzle_name: str):
    """Convert tokens to grid with inverse augmentation."""
    orig_name, inverse_fn = inverse_aug(puzzle_name)
    tokens_np = tokens.numpy()
    grid = inverse_fn(_crop(tokens_np))
    return grid


def validate_puzzle(
    checkpoint_path: str,
    data_path: str,
    identifiers_path: str,
    puzzle_idx: int,
    device: str = "cuda"
):
    """
    Validate that custom inference matches reference implementation.

    Returns:
        dict with validation results
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION: Puzzle {puzzle_idx}")
    print(f"{'='*80}\n")

    # Load identifier map
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}

    # Load model
    print("📦 Loading model...")
    model, config = load_checkpoint(checkpoint_path, device)
    print(f"✅ Model loaded (H_cycles={config['H_cycles']}, L_cycles={config['L_cycles']})")

    # Load puzzle
    print(f"\n📥 Loading puzzle {puzzle_idx}...")
    batch = load_test_puzzle(data_path, puzzle_idx, identifier_map)
    print(f"✅ Loaded: {batch['puzzle_name']} ({len(batch['inputs'])} examples)")

    # Run reference inference
    print(f"\n🔬 Running REFERENCE inference (ACT wrapper)...")
    ref_pred = reference_inference(model, batch, device)
    print(f"✅ Reference complete")

    # Run custom inference
    print(f"\n🔬 Running CUSTOM inference (manual H-cycles)...")
    custom_pred = custom_inference(model, batch, device, max_h_steps=config['H_cycles'])
    print(f"✅ Custom complete")

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}\n")

    test_idx = len(batch['inputs']) - 1  # Last example is test
    ref_grid = tokens_to_grid(ref_pred[test_idx], batch['puzzle_name'])
    custom_grid = tokens_to_grid(custom_pred[test_idx], batch['puzzle_name'])
    ground_truth = tokens_to_grid(batch['labels'][test_idx], batch['puzzle_name'])

    # Token-level comparison
    token_match = torch.equal(ref_pred, custom_pred)
    num_diff_tokens = (ref_pred != custom_pred).sum().item()
    total_tokens = ref_pred.numel()

    # Grid-level comparison
    grid_match = np.array_equal(ref_grid, custom_grid)
    num_diff_cells = (ref_grid != custom_grid).sum()
    total_cells = ref_grid.size

    # Accuracy
    ref_correct = np.array_equal(ref_grid, ground_truth)
    custom_correct = np.array_equal(custom_grid, ground_truth)
    ref_acc = (ref_grid == ground_truth).mean()
    custom_acc = (custom_grid == ground_truth).mean()

    # Results
    results = {
        'puzzle_idx': puzzle_idx,
        'puzzle_name': batch['puzzle_name'],
        'token_match': bool(token_match),
        'num_diff_tokens': int(num_diff_tokens),
        'total_tokens': int(total_tokens),
        'grid_match': bool(grid_match),
        'num_diff_cells': int(num_diff_cells),
        'total_cells': int(total_cells),
        'ref_solved': bool(ref_correct),
        'custom_solved': bool(custom_correct),
        'ref_accuracy': float(ref_acc),
        'custom_accuracy': float(custom_acc),
    }

    # Display
    print(f"Token-level:")
    print(f"  Match: {'✅ YES' if token_match else '❌ NO'}")
    print(f"  Differences: {num_diff_tokens}/{total_tokens} ({num_diff_tokens/total_tokens*100:.2f}%)")

    print(f"\nGrid-level (test example):")
    print(f"  Match: {'✅ YES' if grid_match else '❌ NO'}")
    print(f"  Differences: {num_diff_cells}/{total_cells} cells ({num_diff_cells/total_cells*100:.2f}%)")

    print(f"\nAccuracy vs Ground Truth:")
    print(f"  Reference:  {'✅ SOLVED' if ref_correct else '❌ UNSOLVED'}  (acc={ref_acc:.2%})")
    print(f"  Custom:     {'✅ SOLVED' if custom_correct else '❌ UNSOLVED'}  (acc={custom_acc:.2%})")

    # Overall verdict
    print(f"\n{'='*80}")
    if token_match and grid_match:
        print(f"VERDICT: ✅ PASS - Perfect match!")
        print(f"{'='*80}")
        return results
    elif grid_match:
        print(f"VERDICT: ⚠️  PASS (with warning) - Grids match but tokens differ")
        print(f"  This may be due to padding/EOS differences")
        print(f"{'='*80}")
        return results
    else:
        print(f"VERDICT: ❌ FAIL - Outputs do not match")
        print(f"\n🔍 Debugging info:")
        print(f"  First difference at token index: {(ref_pred != custom_pred).nonzero()[0].item() if num_diff_tokens > 0 else 'N/A'}")
        print(f"  Reference token: {ref_pred.flatten()[0].item()}")
        print(f"  Custom token: {custom_pred.flatten()[0].item()}")
        print(f"{'='*80}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Validate step-by-step inference POC")
    parser.add_argument("--checkpoint", type=str,
                        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
                        help="Path to TRM checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--identifiers", type=str,
                        default="/data/arc1concept-aug-1000/identifiers.json",
                        help="Path to identifiers.json")
    parser.add_argument("--puzzle_idx", type=int, default=0,
                        help="Puzzle index to validate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    results = validate_puzzle(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        identifiers_path=args.identifiers,
        puzzle_idx=args.puzzle_idx,
        device=args.device
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    # Exit code
    if results['grid_match']:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
