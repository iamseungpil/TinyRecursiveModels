"""
Extract TRM latents from ARC test set with FULLY CORRECTED configuration.

This script fixes ALL three critical issues:
1. ✅ Uses actual puzzle_identifiers (not zeros)
2. ✅ Loads augmented .npy data (not raw test_puzzles.json)
3. ✅ Correct L_cycles=6 config (matching arch/trm.yaml)

Expected performance: ~48% Pass1 accuracy (matching reported results)
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
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

# GPU 4 for extraction
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_test_dataset(data_path: str):
    """
    Load test dataset from .npy files (properly augmented data).

    Returns batches organized by puzzle (all examples from same puzzle together).
    """
    test_dir = os.path.join(data_path, 'test')

    print(f"📥 Loading test dataset from {test_dir}...")

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    print(f"✅ Loaded:")
    print(f"   Total examples: {len(inputs)}")
    print(f"   Total puzzles: {len(puzzle_identifiers)}")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Puzzle ID range: {puzzle_identifiers.min()} - {puzzle_identifiers.max()}")

    # Create batches by puzzle
    batches = []
    for puzzle_idx in range(len(puzzle_identifiers)):
        start_idx = puzzle_indices[puzzle_idx]
        end_idx = puzzle_indices[puzzle_idx + 1]

        batch = {
            'inputs': torch.from_numpy(inputs[start_idx:end_idx].astype(np.int64)),
            'labels': torch.from_numpy(labels[start_idx:end_idx].astype(np.int64)),
            'puzzle_identifiers': torch.full(
                (end_idx - start_idx,),
                puzzle_identifiers[puzzle_idx],
                dtype=torch.long
            )
        }
        batches.append(batch)

    return batches


def load_checkpoint(checkpoint_path: str, device: str):
    """
    Load TRM model with CORRECT configuration matching arch/trm.yaml.

    Key fix: L_cycles = 6 (not 4!)
    """
    print(f"\n🔧 Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip prefixes from torch.compile
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    # ✅ CORRECT config matching arch/trm.yaml
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
        "L_cycles": 6,  # ✅ FIXED: was 4, now 6 (matching arch/trm.yaml)
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    print(f"✅ Model config:")
    print(f"   L_layers={config['L_layers']}, H_layers={config['H_layers']}")
    print(f"   L_cycles={config['L_cycles']}, H_cycles={config['H_cycles']}")
    print(f"   hidden_size={config['hidden_size']}")
    print(f"   num_puzzle_identifiers={config['num_puzzle_identifiers']}")

    # Create and load model
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    return model


def extract_latents_and_evaluate(model, batches, identifier_map, device):
    """
    Extract latents and evaluate using official evaluation method.

    Returns list of dicts with:
    - task_id: original task ID (after inverse_aug)
    - latent: 512D vector
    - solved: bool
    - ponder_steps: int
    - input_grid, output_grid, pred_grid: cropped numpy arrays
    """
    results = []

    print(f"\n🔬 Processing {len(batches)} puzzles...")

    with torch.no_grad():
        for batch in tqdm(batches, desc="Extracting latents"):
            # Move batch to device
            inputs = batch['inputs'].to(device)  # [num_examples, 900]
            labels = batch['labels'].to(device)
            puzzle_ids = batch['puzzle_identifiers'].to(device)

            # Get puzzle name and inverse augmentation function
            puzzle_id = puzzle_ids[0].item()
            puzzle_name = identifier_map.get(puzzle_id, f"<unknown_{puzzle_id}>")
            orig_name, inverse_fn = inverse_aug(puzzle_name)

            # Prepare batch dict (using correct API)
            batch_dict = {
                'inputs': inputs,
                'labels': labels,
                'puzzle_identifiers': puzzle_ids
            }

            # TRM forward pass (correct API: initial_carry + forward)
            carry = model.initial_carry(batch_dict)
            carry, outputs = model(carry, batch_dict)

            # Extract z_H latent (high-level representation)
            # Shape: [num_examples, seq_len, hidden_size]
            z_H = carry.inner_carry.z_H.float()  # Convert bfloat16 to float32
            latents = z_H.mean(dim=1).cpu().numpy()  # [num_examples, 512]

            # Get predictions
            predictions = outputs['logits'].argmax(dim=-1).cpu().numpy()  # [num_examples, 900]
            labels_np = labels.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            # Process each example
            for i in range(len(inputs)):
                # Apply inverse augmentation and crop
                input_cropped = inverse_fn(_crop(inputs_np[i]))
                label_cropped = inverse_fn(_crop(labels_np[i]))
                pred_cropped = inverse_fn(_crop(predictions[i]))

                # Evaluate
                solved = np.array_equal(pred_cropped, label_cropped)

                # Determine example type (train demo vs test)
                # In test split: first N-1 are train demos, last is test
                example_type = "train" if i < len(inputs) - 1 else "test"

                results.append({
                    'task_id': f"{orig_name}_{example_type}_{i}",
                    'puzzle_id': int(puzzle_id),
                    'puzzle_name': puzzle_name,
                    'latent': latents[i].tolist(),
                    'solved': bool(solved),
                    'ponder_steps': int(carry.steps[i].item()),
                    'input_grid': input_cropped.tolist(),
                    'output_grid': label_cropped.tolist(),
                    'pred_grid': pred_cropped.tolist(),
                })

    # Summary statistics
    solved_count = sum(1 for r in results if r['solved'])
    test_examples = [r for r in results if 'test' in r['task_id']]
    test_solved = sum(1 for r in test_examples if r['solved'])

    print(f"\n✅ Extraction complete:")
    print(f"   Total examples: {len(results)}")
    print(f"   Total solved: {solved_count} ({solved_count/len(results)*100:.1f}%)")
    print(f"   Test examples: {len(test_examples)}")
    print(f"   Test solved: {test_solved} ({test_solved/len(test_examples)*100:.1f}% Pass1)")
    print(f"   Average ponder steps: {np.mean([r['ponder_steps'] for r in results]):.2f}")

    return results


def main():
    # Paths
    data_path = "/data/arc1concept-aug-1000"
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    identifiers_path = os.path.join(data_path, "identifiers.json")
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("TRM Latent Extraction - FULLY CORRECTED VERSION")
    print("="*70)
    print(f"🖥️  Device: {device}")
    print(f"📂 Data: {data_path}")
    print(f"🔧 Checkpoint: {checkpoint_path}")
    print("="*70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load identifier map
    print(f"\n📥 Loading identifier map...")
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}
    print(f"✅ Loaded {len(identifier_map)} puzzle identifiers")

    # Load test dataset
    batches = load_test_dataset(data_path)

    # Load model
    model = load_checkpoint(checkpoint_path, device)

    # Extract latents and evaluate
    results = extract_latents_and_evaluate(model, batches, identifier_map, device)

    # Save results
    output_file = os.path.join(output_dir, "latents_corrected.json")
    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! Results saved to {output_file}")
    print(f"\n📊 Final Summary:")
    print(f"   Expected Pass1: ~48%")
    test_results = [r for r in results if 'test' in r['task_id']]
    actual_pass1 = sum(1 for r in test_results if r['solved']) / len(test_results) * 100
    print(f"   Actual Pass1: {actual_pass1:.1f}%")

    if actual_pass1 > 40:
        print(f"\n🎉 SUCCESS! Performance matches expected ~48% Pass1")
    else:
        print(f"\n⚠️  WARNING: Performance below expected. Check configuration.")


if __name__ == "__main__":
    main()
