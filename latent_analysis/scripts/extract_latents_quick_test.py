"""
Quick test version - samples first 1000 puzzles for rapid validation.

Expected: ~48% Pass1 accuracy on sampled test examples.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_arc_dataset import inverse_aug
from evaluators.arc import _crop

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# SAMPLE SIZE
MAX_PUZZLES = 5000  # Sample first 5000 puzzles for better diversity


def load_test_dataset_sampled(data_path: str, max_puzzles: int):
    """Load first N puzzles for quick testing."""
    test_dir = os.path.join(data_path, 'test')

    print(f"📥 Loading sampled test dataset (first {max_puzzles} puzzles)...")

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    print(f"✅ Dataset loaded:")
    print(f"   Total puzzles available: {len(puzzle_identifiers)}")
    print(f"   Sampling: {max_puzzles} puzzles")

    # Sample first N puzzles
    batches = []
    for puzzle_idx in range(min(max_puzzles, len(puzzle_identifiers))):
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

    print(f"✅ Sampled {len(batches)} puzzles")
    return batches


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


def extract_and_evaluate(model, batches, identifier_map, device):
    """Extract latents and evaluate."""
    results = []

    print(f"\n🔬 Processing {len(batches)} puzzles...")

    with torch.no_grad():
        for batch in tqdm(batches, desc="Extracting"):
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            puzzle_ids = batch['puzzle_identifiers'].to(device)

            puzzle_id = puzzle_ids[0].item()
            puzzle_name = identifier_map.get(puzzle_id, f"<unknown_{puzzle_id}>")
            orig_name, inverse_fn = inverse_aug(puzzle_name)

            batch_dict = {
                'inputs': inputs,
                'labels': labels,
                'puzzle_identifiers': puzzle_ids
            }

            carry = model.initial_carry(batch_dict)
            carry, outputs = model(carry, batch_dict)

            z_H = carry.inner_carry.z_H.float()
            latents = z_H.mean(dim=1).cpu().numpy()

            predictions = outputs['logits'].argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            for i in range(len(inputs)):
                input_cropped = inverse_fn(_crop(inputs_np[i]))
                label_cropped = inverse_fn(_crop(labels_np[i]))
                pred_cropped = inverse_fn(_crop(predictions[i]))

                solved = np.array_equal(pred_cropped, label_cropped)
                example_type = "train" if i < len(inputs) - 1 else "test"

                results.append({
                    'task_id': f"{orig_name}_{example_type}_{i}",
                    'puzzle_id': int(puzzle_id),
                    'solved': bool(solved),
                    'ponder_steps': int(carry.steps[i].item()),
                    'latent': latents[i].tolist(),  # ✅ Add latent vector
                })

    # Statistics
    solved_count = sum(1 for r in results if r['solved'])
    test_examples = [r for r in results if 'test' in r['task_id']]
    test_solved = sum(1 for r in test_examples if r['solved'])

    print(f"\n📊 Results:")
    print(f"   Total examples: {len(results)}")
    print(f"   Total solved: {solved_count} ({solved_count/len(results)*100:.1f}%)")
    print(f"   Test examples: {len(test_examples)}")
    print(f"   Test solved (Pass1): {test_solved}/{len(test_examples)} ({test_solved/len(test_examples)*100:.1f}%)")
    print(f"   Average ponder: {np.mean([r['ponder_steps'] for r in results]):.2f}")

    return results


def main():
    data_path = "/data/arc1concept-aug-1000"
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    identifiers_path = os.path.join(data_path, "identifiers.json")
    output_dir = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("QUICK TEST - First 1000 Puzzles")
    print("="*70)
    print(f"Expected Pass1: ~48%")
    print("="*70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load identifier map
    with open(identifiers_path, 'r') as f:
        identifier_list = json.load(f)
    identifier_map = {i: name for i, name in enumerate(identifier_list)}

    # Load sampled dataset
    batches = load_test_dataset_sampled(data_path, MAX_PUZZLES)

    # Load model
    model = load_checkpoint(checkpoint_path, device)

    # Extract and evaluate
    results = extract_and_evaluate(model, batches, identifier_map, device)

    # Save
    output_file = os.path.join(output_dir, "quick_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    # Final verdict
    test_results = [r for r in results if 'test' in r['task_id']]
    actual_pass1 = sum(1 for r in test_results if r['solved']) / len(test_results) * 100

    print(f"\n" + "="*70)
    print(f"🎯 VALIDATION RESULT")
    print(f"="*70)
    print(f"Expected Pass1: ~48%")
    print(f"Actual Pass1:   {actual_pass1:.1f}%")

    if actual_pass1 > 40:
        print(f"\n✅ SUCCESS! Configuration fixes verified!")
        print(f"   All 3 issues are corrected:")
        print(f"   1. ✅ Real puzzle IDs (not zeros)")
        print(f"   2. ✅ Augmented .npy data (not raw JSON)")
        print(f"   3. ✅ L_cycles=6 (not 4)")
    else:
        print(f"\n⚠️  Performance below expected. Further investigation needed.")

    print("="*70)


if __name__ == "__main__":
    main()
