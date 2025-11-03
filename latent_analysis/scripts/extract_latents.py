"""
Extract TRM latents from ARC validation set for PCA visualization.

Uses checkpoint_step_91.pt and processes ~400 validation problems.
Extracts the final z_H (high-level) latent representation from each problem.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# GPU 4 강제 사용
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_arc_data(arc_data_path: str) -> List[Dict]:
    """Load ARC test puzzles from test_puzzles.json (400 validation problems)."""
    with open(arc_data_path, 'r') as f:
        test_puzzles = json.load(f)

    all_problems = []
    for task_id, task_data in test_puzzles.items():
        # Extract train examples (demonstrations) for each task
        for idx, example in enumerate(task_data.get('train', [])):
            all_problems.append({
                'task_id': f"{task_id}_train_{idx}",
                'input': example['input'],
                'output': example['output']
            })

        # Also extract test examples
        for idx, example in enumerate(task_data.get('test', [])):
            all_problems.append({
                'task_id': f"{task_id}_test_{idx}",
                'input': example['input'],
                'output': example['output']
            })

    print(f"📊 Loaded {len(all_problems)} problems from {len(test_puzzles)} tasks")
    return all_problems


def preprocess_grid(grid: List[List[int]], max_size: int = 30) -> torch.Tensor:
    """
    Convert ARC grid to TRM input format.

    Grid values: 0-9 (ARC colors)
    TRM expects: values + 2 (because 0=PAD, 1=EOS in training)
    Pad to 30x30 and flatten.
    """
    grid_np = np.array(grid, dtype=np.uint8)
    h, w = grid_np.shape

    # Pad to max_size x max_size
    padded = np.zeros((max_size, max_size), dtype=np.uint8)
    padded[:h, :w] = grid_np + 2  # Shift by 2 (PAD=0, EOS=1, colors=2-11)

    # Flatten
    return torch.from_numpy(padded.flatten()).unsqueeze(0)  # [1, 900]


def _crop(grid: np.ndarray) -> np.ndarray:
    """
    Official ARC evaluator crop function.
    Find maximum-sized rectangle without any EOS token inside.

    From /home/ubuntu/TinyRecursiveModels/evaluators/arc.py
    """
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


def load_trm_model(checkpoint_path: str, device: str = 'cuda') -> TinyRecursiveReasoningModel_ACTV1:
    """
    Load TRM model with correct config matching checkpoint_step_91.pt.

    Key fixes from train_compositional.py:
    - vocab_size: 12 (not 10)
    - pos_encodings: 'rope' (not 'learned')
    - Strip "model." prefix from state_dict
    """
    print(f"📥 Loading TRM from {checkpoint_path}...")

    # Config matching step_518071 (from all_config.yaml)
    config = {
        "batch_size": 32,
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 876406,  # ← Checkpoint has 876406 puzzle IDs
        "puzzle_emb_ndim": 512,  # ← step_518071 uses puzzle embeddings
        "puzzle_emb_len": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
        "L_cycles": 4,  # ← Different from checkpoint_step_91
        "H_layers": 0,  # ← Different!
        "L_layers": 2,  # ← Different! (was 4)
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        # Format: checkpoint_step_91.pt
        state_dict = checkpoint["model_state_dict"]
    else:
        # Format: step_518071 (direct state_dict)
        state_dict = checkpoint

    # Strip "_orig_mod.model." or "model." prefix
    if any(k.startswith("_orig_mod.model.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.model.", "", 1): v for k, v in state_dict.items()}
    elif any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    print(f"✓ TRM loaded (hidden_size={config['hidden_size']})")
    return model


def extract_latents(
    model: TinyRecursiveReasoningModel_ACTV1,
    problems: List[Dict],
    device: str = 'cuda',
    batch_size: int = 32
) -> List[Dict]:
    """
    Extract final z_H latent from each problem.

    Returns list of:
    {
        'task_id': str,
        'latent': np.ndarray [512],
        'ponder_steps': int,
        'solved': bool (if output matches target),
        'input_grid': List[List[int]],
        'output_grid': List[List[int]],
        'pred_grid': np.ndarray
    }
    """
    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(problems), batch_size), desc="Extracting latents"):
            batch = problems[i:i+batch_size]

            # Prepare batch inputs
            inputs = []
            labels = []
            for prob in batch:
                inp = preprocess_grid(prob['input'])
                out = preprocess_grid(prob['output'])
                inputs.append(inp)
                labels.append(out)

            inputs = torch.cat(inputs, dim=0).to(device)  # [B, 900]
            labels = torch.cat(labels, dim=0).to(device)  # [B, 900]

            # Prepare batch dict
            puzzle_ids = torch.zeros(len(batch), dtype=torch.long, device=device)
            batch_dict = {
                'inputs': inputs,
                'labels': labels,
                'puzzle_identifiers': puzzle_ids
            }

            # Initialize carry and forward pass
            carry = model.initial_carry(batch_dict)
            carry, outputs = model(carry, batch_dict)

            # Extract z_H (high-level latent) from final carry
            # carry.inner_carry.z_H has shape [B, seq_len, hidden_size]
            # Take mean pooling over sequence dimension
            z_H = carry.inner_carry.z_H  # [B, 900, 512]
            latents = z_H.mean(dim=1).float().cpu().numpy()  # [B, 512] - convert bfloat16 to float32

            # Predictions
            predictions = outputs['logits'].argmax(dim=-1).cpu().numpy()  # [B, 900]
            labels_np = labels.cpu().numpy()

            # Store results
            for j, prob in enumerate(batch):
                # Use official _crop function from evaluators/arc.py
                pred_cropped = _crop(predictions[j])
                label_cropped = _crop(labels_np[j])

                # Compare using official method
                solved = np.array_equal(pred_cropped, label_cropped)

                results.append({
                    'task_id': prob['task_id'],
                    'latent': latents[j].tolist(),
                    'ponder_steps': int(carry.steps[j].item()),
                    'solved': solved,
                    'input_grid': prob['input'],
                    'output_grid': prob['output'],
                    'pred_grid': predictions[j].reshape(30, 30).tolist()
                })

    return results


def main():
    # Paths
    checkpoint_path = "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071"
    arc_data_path = "/data/arc1concept-aug-1000/test_puzzles.json"  # 400 validation tasks
    output_path = "/home/ubuntu/TinyRecursiveModels/latent_analysis/data/latents.json"

    device = 'cuda'  # Will be cuda:0 after CUDA_VISIBLE_DEVICES=4

    print("="*60)
    print("TRM Latent Extraction")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {arc_data_path}")
    print(f"Device: GPU 4 (CUDA_VISIBLE_DEVICES=4)")
    print("="*60)

    # Load data
    problems = load_arc_data(arc_data_path)

    # Load model
    model = load_trm_model(checkpoint_path, device)

    # Extract latents
    print(f"\n🔬 Extracting latents...")
    results = extract_latents(model, problems, device, batch_size=32)

    # Save results
    print(f"\n💾 Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    solved = sum(1 for r in results if r['solved'])
    print(f"\n📊 Summary:")
    print(f"  Total problems: {len(results)}")
    if len(results) > 0:
        print(f"  Solved: {solved} ({solved/len(results)*100:.1f}%)")
        print(f"  Unsolved: {len(results)-solved} ({(len(results)-solved)/len(results)*100:.1f}%)")
    print(f"  Latent dim: 512")
    print(f"\n✅ Done! Results saved to {output_path}")


if __name__ == "__main__":
    main()
