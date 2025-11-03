"""
Extract latents for all 400 validation puzzles (one representation per puzzle).

This script:
1. Loads all 400 validation puzzle names from test_puzzles.json
2. For each puzzle, finds its test example in the dataset
3. Extracts latent representation
4. Saves results for PCA analysis
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_checkpoint(checkpoint_path: str, device: str):
    """Load TRM model with correct L_cycles=6 config."""
    print(f"\n🔧 Loading checkpoint...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip prefixes
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
        "L_cycles": 6,  # ✅ CORRECT
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    print(f"✅ Config: L_cycles={config['L_cycles']}, L_layers={config['L_layers']}")

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded")
    return model


def find_puzzle_in_dataset(puzzle_name: str, identifier_map: dict,
                           inputs_mmap, labels_mmap, puzzle_identifiers, puzzle_indices):
    """Find the first test example of a puzzle in the dataset."""

    # Search for this puzzle name in identifiers
    for puzzle_idx in range(len(puzzle_identifiers)):
        puzzle_id = puzzle_identifiers[puzzle_idx]
        mapped_name = identifier_map.get(puzzle_id, "")

        # Check if this is the puzzle we want (remove augmentation suffix)
        original_name, _ = inverse_aug(mapped_name)

        if original_name == puzzle_name:
            # Found it! Get the test example (last one in the batch)
            start_idx = puzzle_indices[puzzle_idx]
            end_idx = puzzle_indices[puzzle_idx + 1]

            # Return test example (last one)
            test_idx = end_idx - 1

            return {
                'input': inputs_mmap[test_idx],
                'label': labels_mmap[test_idx],
                'puzzle_id': puzzle_id,
                'puzzle_idx': puzzle_idx
            }

    return None


def main():
    data_path = "/data/arc1concept-aug-1000"
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    identifiers_path = os.path.join(data_path, "identifiers.json")
    test_puzzles_path = os.path.join(data_path, "test_puzzles.json")
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("Extract Latents for All 400 Validation Puzzles")
    print("="*70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load test puzzles (400 total)
    print(f"\n📥 Loading validation puzzle names...")
    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)
    print(f"✅ Found {len(test_puzzles)} validation puzzles")

    # Load identifier map
    print(f"\n📥 Loading identifier map...")
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}
    print(f"✅ Loaded {len(identifier_map)} identifiers")

    # Load dataset (memory-mapped for efficiency)
    print(f"\n📥 Loading dataset...")
    test_dir = os.path.join(data_path, 'test')
    inputs_mmap = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels_mmap = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))
    print(f"✅ Dataset loaded: {len(inputs_mmap)} examples, {len(puzzle_identifiers)} unique puzzle IDs")

    # Load model
    model = load_checkpoint(checkpoint_path, device)

    # Extract latents for all 400 puzzles
    print(f"\n🔬 Extracting latents for 400 puzzles...")
    results = []
    not_found = []

    with torch.no_grad():
        for puzzle_name in tqdm(test_puzzles.keys(), desc="Processing"):
            # Find puzzle in dataset
            puzzle_data = find_puzzle_in_dataset(
                puzzle_name, identifier_map,
                inputs_mmap, labels_mmap,
                puzzle_identifiers, puzzle_indices
            )

            if puzzle_data is None:
                not_found.append(puzzle_name)
                continue

            # Prepare batch
            inputs = torch.from_numpy(puzzle_data['input'].astype(np.int64)).unsqueeze(0).to(device)
            labels = torch.from_numpy(puzzle_data['label'].astype(np.int64)).unsqueeze(0).to(device)
            puzzle_ids = torch.tensor([puzzle_data['puzzle_id']], dtype=torch.long, device=device)

            batch_dict = {
                'inputs': inputs,
                'labels': labels,
                'puzzle_identifiers': puzzle_ids
            }

            # Extract latent
            carry = model.initial_carry(batch_dict)
            carry, outputs = model(carry, batch_dict)

            z_H = carry.inner_carry.z_H.float()
            latent = z_H.mean(dim=1).cpu().numpy()[0]  # Average over sequence

            # Check if solved
            prediction = outputs['logits'].argmax(dim=-1).cpu().numpy()[0]
            label_np = labels.cpu().numpy()[0]
            input_np = inputs.cpu().numpy()[0]

            # Crop and check
            original_name, inverse_fn = inverse_aug(identifier_map[puzzle_data['puzzle_id']])
            input_cropped = inverse_fn(_crop(input_np))
            label_cropped = inverse_fn(_crop(label_np))
            pred_cropped = inverse_fn(_crop(prediction))

            solved = np.array_equal(pred_cropped, label_cropped)

            results.append({
                'puzzle_name': puzzle_name,
                'puzzle_id': int(puzzle_data['puzzle_id']),
                'solved': bool(solved),
                'ponder_steps': int(carry.steps[0].item()),
                'latent': latent.tolist(),
                'prediction': pred_cropped.tolist()
            })

    # Statistics
    solved_count = sum(1 for r in results if r['solved'])
    print(f"\n📊 Results:")
    print(f"   Total puzzles: {len(results)}")
    print(f"   Solved (Pass1): {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")
    print(f"   Not found: {len(not_found)}")
    if not_found:
        print(f"   Not found puzzles: {not_found[:10]}...")

    # Save
    output_file = os.path.join(output_dir, "validation_400_puzzles.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
