"""
Extract TRM latents from ARC test set with proper augmentation handling.

This script properly uses:
1. Dataset loader with test split (includes all augmentations)
2. Actual puzzle_identifiers from dataset
3. inverse_aug to restore original orientation
4. Official _crop for evaluation
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Add project root to path
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug, grid_hash
from evaluators.arc import _crop

# GPU 4 for extraction
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_test_dataset(data_path: str, batch_size: int = 8):
    """
    Load test dataset using numpy files directly (mimicking puzzle_dataset.py).

    Returns batches of (inputs, labels, puzzle_identifiers) for test set.
    """
    # Load test data
    test_dir = os.path.join(data_path, 'test')

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    print(f"📊 Loaded test set:")
    print(f"   Total examples: {len(inputs)}")
    print(f"   Total puzzles: {len(puzzle_identifiers)}")
    print(f"   Input shape: {inputs.shape}")

    # Create batches by puzzle (all examples from same puzzle in one batch)
    batches = []
    for puzzle_id in range(len(puzzle_identifiers)):
        start_idx = puzzle_indices[puzzle_id]
        end_idx = puzzle_indices[puzzle_id + 1]

        batch_inputs = torch.from_numpy(inputs[start_idx:end_idx].astype(np.int64))
        batch_labels = torch.from_numpy(labels[start_idx:end_idx].astype(np.int64))
        batch_puzzle_ids = torch.full((end_idx - start_idx,), puzzle_identifiers[puzzle_id], dtype=torch.long)

        batches.append({
            'inputs': batch_inputs,
            'labels': batch_labels,
            'puzzle_identifiers': batch_puzzle_ids
        })

    return batches


def load_checkpoint(checkpoint_path: str, device: str):
    """Load TRM model from checkpoint."""
    print(f"\n🔧 Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip _orig_mod.model. prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        cleaned_state_dict[k] = v

    # Model config (from checkpoint metadata)
    config = {
        "batch_size": 32,
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
    }

    print(f"✅ Model config: L={config['L_layers']}, H={config['H_layers']}, cycles={config['L_cycles']}, hidden={config['hidden_size']}")

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    return model


def extract_latents_and_evaluate(model, batches, identifier_map, device):
    """
    Extract latents and evaluate predictions using official method.

    Returns list of dicts with:
    - task_id: original task ID (after inverse_aug)
    - latent: 512D latent vector
    - solved: bool
    - ponder_steps: int
    - input_grid, output_grid, pred_grid: numpy arrays
    """
    results = []

    print(f"\n🔬 Processing {len(batches)} puzzles...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Extracting latents")):
            inputs = batch['inputs'].to(device)  # [num_examples, 900]
            labels = batch['labels'].to(device)
            puzzle_ids = batch['puzzle_identifiers']  # [num_examples]

            # Get puzzle name and inverse function
            puzzle_id = puzzle_ids[0].item()
            puzzle_name = identifier_map[puzzle_id]
            orig_name, inverse_fn = inverse_aug(puzzle_name)

            # Forward pass on each example
            for i in range(len(inputs)):
                input_seq = inputs[i:i+1]  # [1, 900]
                label_seq = labels[i:i+1]

                # TRM forward
                outputs = model(
                    input_seq,
                    puzzle_identifiers=puzzle_ids[i:i+1].to(device)
                )

                # Extract latent (mean pool over sequence)
                if hasattr(outputs, 'z_H') and outputs.z_H is not None:
                    latent = outputs.z_H.mean(dim=1).cpu().numpy()[0]  # [512]
                else:
                    # Fallback: use z_L
                    latent = outputs.z_L.mean(dim=1).cpu().numpy()[0]

                # Get prediction
                pred_logits = outputs.logits  # [1, 900, vocab_size]
                predictions = pred_logits.argmax(dim=-1).cpu().numpy()  # [1, 900]

                # Prepare grids
                input_np = input_seq.cpu().numpy()[0]  # [900]
                label_np = label_seq.cpu().numpy()[0]  # [900]
                pred_np = predictions[0]  # [900]

                # Apply inverse augmentation
                input_cropped = inverse_fn(_crop(input_np))
                label_cropped = inverse_fn(_crop(label_np))
                pred_cropped = inverse_fn(_crop(pred_np))

                # Evaluate
                solved = np.array_equal(pred_cropped, label_cropped)

                # Determine if this is train or test example
                # In test split, both train demonstrations and test problems are included
                # We'll include all for now, can filter later if needed
                example_type = "train" if i < len(inputs) - 1 else "test"

                results.append({
                    'task_id': f"{orig_name}_{example_type}_{i}",
                    'latent': latent.tolist(),
                    'solved': bool(solved),
                    'ponder_steps': int(outputs.ponder_steps[0].item()) if hasattr(outputs, 'ponder_steps') else 1,
                    'input_grid': input_cropped.tolist(),
                    'output_grid': label_cropped.tolist(),
                    'pred_grid': pred_cropped.tolist(),
                })

    solved_count = sum(1 for r in results if r['solved'])
    print(f"\n✅ Extraction complete:")
    print(f"   Total examples: {len(results)}")
    print(f"   Solved: {solved_count} ({solved_count/len(results)*100:.1f}%)")

    return results


def main():
    # Paths
    data_path = "/data/arc1concept-aug-1000"
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    identifiers_path = os.path.join(data_path, "identifiers.json")
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load identifier map
    print(f"\n📥 Loading identifier map from {identifiers_path}...")
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}
    print(f"✅ Loaded {len(identifier_map)} puzzle identifiers")

    # Load test dataset
    print(f"\n📥 Loading test dataset from {data_path}...")
    batches = load_test_dataset(data_path)

    # Load model
    model = load_checkpoint(checkpoint_path, device)

    # Extract latents and evaluate
    results = extract_latents_and_evaluate(model, batches, identifier_map, device)

    # Save results
    output_file = os.path.join(output_dir, "latents_fixed.json")
    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Done! Results saved to {output_file}")
    print(f"\n📊 Summary:")
    print(f"   Total examples: {len(results)}")
    print(f"   Solved: {sum(1 for r in results if r['solved'])} ({sum(1 for r in results if r['solved'])/len(results)*100:.1f}%)")
    print(f"   Average ponder steps: {np.mean([r['ponder_steps'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
