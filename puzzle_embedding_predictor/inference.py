"""
Inference with CNN Puzzle Embedding Predictor

Predicts puzzle embeddings for unseen tasks and runs TRM inference.

Usage:
    # Single puzzle
    python inference.py \
        --cnn-checkpoint ./checkpoints/run_xxx/checkpoint_best.pt \
        --trm-checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
        --puzzle-json /path/to/puzzle.json \
        --output prediction.json \
        --gpu 0

    # Batch evaluation on ARC test set
    python inference.py \
        --cnn-checkpoint ./checkpoints/run_xxx/checkpoint_best.pt \
        --trm-checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
        --test-dir /data/arc_test \
        --output-dir ./predictions \
        --gpu 0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels/puzzle_embedding_predictor')

from models.cnn_encoder import PuzzleEmbeddingCNN
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from evaluators.arc import _crop


def load_cnn_model(checkpoint_path: str, device: str) -> PuzzleEmbeddingCNN:
    """Load trained CNN puzzle embedding predictor."""
    print(f"\n🔧 Loading CNN predictor from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with default config (should match training)
    model = PuzzleEmbeddingCNN(
        vocab_size=12,
        embedding_dim=64,
        hidden_channels=256,
        num_blocks=4,
        output_dim=512
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ CNN model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"   Val cosine similarity: {metrics.get('val_cosine_similarity', 'N/A')}")

    return model


def load_trm_model(checkpoint_path: str, device: str) -> TinyRecursiveReasoningModel_ACTV1:
    """Load TRM model for inference."""
    print(f"\n🔧 Loading TRM model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Strip prefixes
    cleaned_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('_orig_mod.model.'):
            k = k.replace('_orig_mod.model.', '')
        elif k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        elif k.startswith('model.'):
            k = k.replace('model.', '')
        cleaned_state_dict[k] = v

    # TRM config (should match training)
    config = {
        "batch_size": 1,
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 1,  # We'll override this dynamically
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

    # Load weights (skip puzzle_emb - we'll inject predicted embeddings)
    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"✅ TRM model loaded")
    if incompatible.missing_keys:
        print(f"   Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"   Unexpected keys: {len(incompatible.unexpected_keys)}")

    model = model.to(device)
    model.eval()

    return model


def predict_puzzle_embedding(
    cnn_model: PuzzleEmbeddingCNN,
    input_grid: np.ndarray,
    device: str,
    max_grid_size: int = 30
) -> torch.Tensor:
    """
    Predict puzzle embedding from input grid using CNN.

    Args:
        cnn_model: Trained CNN predictor
        input_grid: (H, W) numpy array
        device: Device
        max_grid_size: Max grid size for padding

    Returns:
        predicted_embedding: (512,) tensor
    """
    # Pad grid
    H, W = input_grid.shape
    padded_grid = np.zeros((max_grid_size, max_grid_size), dtype=np.int64)
    padded_grid[:H, :W] = input_grid

    # Convert to tensor
    grid_tensor = torch.from_numpy(padded_grid).long().unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        embedding = cnn_model(grid_tensor)  # (1, 512)

    return embedding.squeeze(0)  # (512,)


def inject_puzzle_embedding(
    trm_model: TinyRecursiveReasoningModel_ACTV1,
    predicted_embedding: torch.Tensor
):
    """
    Inject predicted puzzle embedding into TRM model.

    This replaces the learned embedding lookup with our CNN prediction.
    """
    # Get puzzle_emb layer
    puzzle_emb_layer = trm_model.inner.puzzle_emb

    # Create a temporary embedding table with single entry
    # We'll use puzzle_id = 0
    new_weights = predicted_embedding.unsqueeze(0)  # (1, 512)

    # Replace the weights temporarily
    # Note: This is a bit hacky, but works for inference
    with torch.no_grad():
        if puzzle_emb_layer.weights.shape[0] == 1:
            puzzle_emb_layer.weights.copy_(new_weights)
        else:
            # If table is larger, just update first entry
            puzzle_emb_layer.weights[0].copy_(predicted_embedding)


def run_trm_inference(
    trm_model: TinyRecursiveReasoningModel_ACTV1,
    input_grid: np.ndarray,
    device: str,
    max_seq_len: int = 900
) -> Dict:
    """
    Run TRM inference with injected puzzle embedding.

    Args:
        trm_model: TRM model
        input_grid: (H, W) input grid
        device: Device
        max_seq_len: Max sequence length

    Returns:
        Dict with prediction, ponder_steps, etc.
    """
    # Prepare input (flatten and pad)
    H, W = input_grid.shape
    flattened = input_grid.flatten()

    # Pad to max_seq_len
    if len(flattened) > max_seq_len:
        flattened = flattened[:max_seq_len]
    padded = np.zeros(max_seq_len, dtype=np.int64)
    padded[:len(flattened)] = flattened

    # Convert to tensors
    inputs = torch.from_numpy(padded).unsqueeze(0).to(device)  # (1, seq_len)
    labels = inputs.clone()  # Dummy labels
    puzzle_ids = torch.tensor([0], dtype=torch.long, device=device)  # Use ID 0

    batch_dict = {
        'inputs': inputs,
        'labels': labels,
        'puzzle_identifiers': puzzle_ids
    }

    # Run inference
    with torch.no_grad():
        carry = trm_model.initial_carry(batch_dict)
        carry, outputs = trm_model(carry, batch_dict)

    # Get prediction
    prediction = outputs['logits'].argmax(dim=-1).cpu().numpy()[0]
    ponder_steps = carry.steps[0].item()

    # Reshape to grid (try to recover original shape)
    # This is approximate - we don't know exact output size
    pred_grid = prediction[:H*W].reshape(H, W)

    return {
        'prediction': pred_grid,
        'ponder_steps': ponder_steps,
        'raw_prediction': prediction.tolist()
    }


def load_puzzle_from_json(puzzle_path: str) -> Dict:
    """Load ARC puzzle from JSON file."""
    with open(puzzle_path, 'r') as f:
        puzzle = json.load(f)
    return puzzle


def process_single_puzzle(
    cnn_model: PuzzleEmbeddingCNN,
    trm_model: TinyRecursiveReasoningModel_ACTV1,
    puzzle: Dict,
    device: str
) -> List[Dict]:
    """
    Process a single puzzle with multiple test examples.

    Args:
        cnn_model: CNN predictor
        trm_model: TRM model
        puzzle: Puzzle dict with 'train' and 'test' keys
        device: Device

    Returns:
        List of predictions for each test example
    """
    results = []

    # Get first train example to predict puzzle embedding
    if len(puzzle['train']) > 0:
        first_train_input = np.array(puzzle['train'][0]['input'])
    else:
        # Fallback: use test input
        first_train_input = np.array(puzzle['test'][0]['input'])

    # Predict puzzle embedding from first example
    print("  Predicting puzzle embedding from input grid...")
    predicted_embedding = predict_puzzle_embedding(
        cnn_model, first_train_input, device
    )

    # Inject into TRM
    inject_puzzle_embedding(trm_model, predicted_embedding)

    # Process each test example
    for test_idx, test_example in enumerate(puzzle['test']):
        test_input = np.array(test_example['input'])

        print(f"  Running TRM inference on test example {test_idx + 1}/{len(puzzle['test'])}...")
        result = run_trm_inference(trm_model, test_input, device)

        results.append({
            'test_index': test_idx,
            'prediction': result['prediction'].tolist(),
            'ponder_steps': result['ponder_steps'],
            'input_shape': test_input.shape,
            'output_shape': result['prediction'].shape
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with CNN puzzle embedding predictor")
    parser.add_argument(
        "--cnn-checkpoint",
        type=str,
        required=True,
        help="Path to trained CNN checkpoint"
    )
    parser.add_argument(
        "--trm-checkpoint",
        type=str,
        default="/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
        help="Path to TRM checkpoint"
    )
    parser.add_argument(
        "--puzzle-json",
        type=str,
        help="Path to single puzzle JSON"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        help="Directory with test puzzles"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for single puzzle"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch evaluation"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device"
    )
    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("CNN Puzzle Embedding Predictor - Inference")
    print("="*70)
    print(f"Device: {device}")

    # Load models
    cnn_model = load_cnn_model(args.cnn_checkpoint, device)
    trm_model = load_trm_model(args.trm_checkpoint, device)

    # Single puzzle mode
    if args.puzzle_json:
        print(f"\n📝 Processing single puzzle: {args.puzzle_json}")
        puzzle = load_puzzle_from_json(args.puzzle_json)

        results = process_single_puzzle(cnn_model, trm_model, puzzle, device)

        # Save
        output_path = args.output or "prediction.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved predictions to {output_path}")

    # Batch mode
    elif args.test_dir:
        print(f"\n📁 Processing test directory: {args.test_dir}")
        test_dir = Path(args.test_dir)
        output_dir = Path(args.output_dir or "./predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        puzzle_files = sorted(test_dir.glob("*.json"))
        print(f"   Found {len(puzzle_files)} puzzles")

        for puzzle_file in tqdm(puzzle_files, desc="Processing puzzles"):
            puzzle = load_puzzle_from_json(puzzle_file)
            results = process_single_puzzle(cnn_model, trm_model, puzzle, device)

            # Save individual result
            output_path = output_dir / f"{puzzle_file.stem}_prediction.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        print(f"\n💾 Saved {len(puzzle_files)} predictions to {output_dir}")

    else:
        print("Error: Must specify either --puzzle-json or --test-dir")
        return

    print("\n" + "="*70)
    print("✅ Inference complete!")
    print("="*70)


if __name__ == "__main__":
    main()
