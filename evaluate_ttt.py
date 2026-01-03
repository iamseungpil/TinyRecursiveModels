#!/usr/bin/env python3
"""
Test-Time Training (TTT) Evaluation Script for TRM-Titans on ARC Puzzles.

This script evaluates TRM_Titans_TestTime on ARC 400 puzzles by:
1. Loading a pretrained TRM-Titans checkpoint
2. For each puzzle: adapting on training examples, then predicting test examples
3. Computing pass@k accuracy using the ARC evaluator
4. Saving submission.json for Kaggle submission

Usage:
    python evaluate_ttt.py \
        --checkpoint outputs/2025-12-29/run_name/step_1000 \
        --data_path data/arc-aug-1000 \
        --ttt_steps 10 \
        --ttt_lr 0.01 \
        --output_path outputs/ttt_eval \
        --device cuda:0 \
        --verbose
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.recursive_reasoning.trm_titans import (
    TRM_Titans,
    TRM_Titans_Config,
    TRM_Titans_TestTime,
    IGNORE_LABEL_ID
)
from dataset.build_arc_dataset import (
    arc_grid_to_np,
    np_grid_to_seq_translational_augment,
    inverse_aug,
    grid_hash,
    ARCMaxGridSize,
    PuzzleIdSeparator
)
from dataset.common import PuzzleDatasetMetadata
from evaluators.arc import _crop


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for TTT evaluation."""
    checkpoint: str
    data_path: str
    ttt_steps: int = 10
    ttt_lr: float = 0.01
    output_path: str = "outputs/ttt_eval"
    device: str = "cuda:0"
    max_puzzles: int = 0  # 0 means all puzzles
    verbose: bool = False
    pass_ks: Tuple[int, ...] = (1, 2)
    accumulate_memory: bool = True  # Whether to accumulate memory across demos


# =============================================================================
# Grid Conversion Utilities
# =============================================================================

def example_to_tensors(
    inp_grid: List[List[int]],
    out_grid: List[List[int]],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an ARC example (input, output) pair to model tensors.

    Args:
        inp_grid: Input grid as list of lists
        out_grid: Output grid as list of lists
        device: Target device

    Returns:
        (input_tokens, label_tokens) each of shape [1, 900]
    """
    inp_arr = arc_grid_to_np(inp_grid)
    out_arr = arc_grid_to_np(out_grid)

    # Convert to token sequences (no translation for evaluation)
    inp_seq, out_seq = np_grid_to_seq_translational_augment(inp_arr, out_arr, do_translation=False)

    # Create tensors
    input_tokens = torch.from_numpy(inp_seq).unsqueeze(0).to(device)  # [1, 900]
    label_tokens = torch.from_numpy(out_seq).unsqueeze(0).to(device)  # [1, 900]

    # Set padding positions in labels to IGNORE_LABEL_ID
    label_tokens = label_tokens.clone()
    label_tokens[label_tokens == 0] = IGNORE_LABEL_ID  # Ignore PAD tokens in loss

    return input_tokens.long(), label_tokens.long()


def tokens_to_grid(tokens: torch.Tensor) -> np.ndarray:
    """
    Convert model output tokens back to ARC grid.

    Uses the same _crop algorithm as evaluators/arc.py to find the maximum
    rectangle without EOS tokens and extract the grid.

    Args:
        tokens: Token tensor of shape [900] or [1, 900]

    Returns:
        Grid as numpy array with values 0-9
    """
    if tokens.dim() == 2:
        tokens = tokens[0]

    tokens = tokens.cpu().numpy()

    # Use the same _crop function as evaluators/arc.py for consistency
    return _crop(tokens)


# =============================================================================
# Model Loading
# =============================================================================

def load_model_config_from_checkpoint(checkpoint_path: str, data_path: str) -> dict:
    """
    Load model configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., outputs/.../step_1000)
        data_path: Path to data directory for metadata

    Returns:
        Model configuration dictionary
    """
    # Load dataset metadata for vocab_size, seq_len, etc.
    metadata_path = os.path.join(data_path, "test", "dataset.json")
    with open(metadata_path, "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))

    # Try to load config from checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.yaml")

    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        arch_config = config.get("arch", {})
    else:
        # Use default TRM-Titans config
        arch_config = {}

    # Get puzzle_emb_ndim first to determine puzzle_emb_len
    puzzle_emb_ndim = arch_config.get("puzzle_emb_ndim", 0)

    # IMPORTANT: When puzzle_emb_ndim=0, puzzle_emb_len must also be 0
    # to avoid dimension mismatch between carry (seq_len + puzzle_emb_len)
    # and input_embeddings (seq_len only when no puzzle embeddings)
    if puzzle_emb_ndim == 0:
        puzzle_emb_len = 0
    else:
        puzzle_emb_len = arch_config.get("puzzle_emb_len", 16)

    # Build model config with required fields
    model_cfg = {
        "batch_size": 1,  # For single-puzzle evaluation
        "seq_len": metadata.seq_len,
        "vocab_size": metadata.vocab_size,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,

        # TRM-Titans specific (use arch_config values or defaults)
        "H_cycles": arch_config.get("H_cycles", 2),
        "L_cycles": arch_config.get("L_cycles", 2),
        "H_layers": arch_config.get("H_layers", 1),  # Ignored in v6
        "L_layers": arch_config.get("L_layers", 4),
        "hidden_size": arch_config.get("hidden_size", 512),
        "expansion": arch_config.get("expansion", 2.0),
        "num_heads": arch_config.get("num_heads", 8),
        "pos_encodings": arch_config.get("pos_encodings", "rope"),
        "halt_max_steps": arch_config.get("halt_max_steps", 1),
        "halt_exploration_prob": arch_config.get("halt_exploration_prob", 0.0),
        "puzzle_emb_ndim": puzzle_emb_ndim,
        "puzzle_emb_len": puzzle_emb_len,
        "memory_hidden_mult": arch_config.get("memory_hidden_mult", 4),
        "surprise_loss_weight": arch_config.get("surprise_loss_weight", 0.1),
        "integration_type": arch_config.get("integration_type", "mag"),
        "forward_dtype": arch_config.get("forward_dtype", "bfloat16"),
    }

    return model_cfg


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model weights from checkpoint.

    Handles torch.compile wrapped models by stripping _orig_mod prefix if needed.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Target device
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix if present
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        # Also handle model.inner. -> inner. mapping
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v

    # Try loading with strict=False to handle shape mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print(f"Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                print(f"  - {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"  - {k}")


# =============================================================================
# Evaluation Logic
# =============================================================================

def evaluate_single_puzzle(
    ttt: TRM_Titans_TestTime,
    puzzle_name: str,
    puzzle_data: dict,
    config: EvalConfig,
    puzzle_id: int = 0
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Evaluate a single puzzle using test-time training.

    Args:
        ttt: TRM_Titans_TestTime wrapper
        puzzle_name: Name/ID of the puzzle
        puzzle_data: Dict with 'train' and 'test' examples
        config: Evaluation configuration
        puzzle_id: Puzzle identifier for embedding lookup

    Returns:
        List of (input_hash, predicted_grid, confidence) for each test example
    """
    device = torch.device(config.device)

    # Prepare demo pairs from training examples
    demo_pairs = []
    for example in puzzle_data["train"]:
        inp_tokens, label_tokens = example_to_tensors(
            example["input"],
            example["output"],
            device
        )
        demo_pairs.append((inp_tokens, label_tokens))

    if not demo_pairs:
        print(f"Warning: Puzzle {puzzle_name} has no training examples")
        return []

    # Adapt on training examples
    # Note: test_time_adapt() internally calls reset_all_memory() at the start
    ttt.test_time_adapt(
        demo_pairs=demo_pairs,
        n_steps=config.ttt_steps,
        lr=config.ttt_lr,
        puzzle_id=puzzle_id,
        verbose=config.verbose,
        accumulate_memory=config.accumulate_memory
    )

    # Predict on test examples
    results = []
    for test_example in puzzle_data["test"]:
        # Convert test input to tokens
        test_inp = arc_grid_to_np(test_example["input"])
        inp_seq, _ = np_grid_to_seq_translational_augment(
            test_inp,
            np.zeros((1, 1), dtype=np.uint8),  # Dummy output
            do_translation=False
        )
        test_input = torch.from_numpy(inp_seq).unsqueeze(0).long().to(device)

        # Get prediction (without resetting memory to use adapted state)
        predictions = ttt.predict(
            test_input,
            update_during_prediction=False,
            puzzle_id=puzzle_id,
            reset_memory=False  # Keep adapted memory
        )

        # Convert prediction to grid
        pred_grid = tokens_to_grid(predictions)

        # Compute input hash for matching
        input_hash = grid_hash(test_inp)

        # Confidence placeholder (could use q_halt_logits if available)
        confidence = 1.0

        results.append((input_hash, pred_grid, confidence))

    return results


def evaluate_all_puzzles(
    model: TRM_Titans,
    puzzles: Dict[str, dict],
    config: EvalConfig
) -> Dict[str, any]:
    """
    Evaluate all puzzles and compute metrics.

    Args:
        model: TRM_Titans model
        puzzles: Dict mapping puzzle names to puzzle data
        config: Evaluation configuration

    Returns:
        Dictionary with metrics and submission data
    """
    device = torch.device(config.device)
    model.to(device)
    model.eval()

    # Create test-time training wrapper
    ttt = TRM_Titans_TestTime(model, device=device)

    # Storage for results
    all_predictions = {}  # puzzle_name -> {input_hash -> [(pred_hash, confidence), ...]}
    pred_grids = {}  # pred_hash -> grid

    # Load test puzzles ground truth for accuracy computation
    test_puzzles_path = os.path.join(config.data_path, "test_puzzles.json")
    with open(test_puzzles_path, "r") as f:
        test_puzzles = json.load(f)

    # Process each puzzle (optionally limit for testing)
    puzzle_names = list(puzzles.keys())
    if config.max_puzzles > 0:
        puzzle_names = puzzle_names[:config.max_puzzles]
        print(f"Limiting evaluation to first {config.max_puzzles} puzzles")

    pbar = tqdm(puzzle_names, desc="Evaluating puzzles")

    for puzzle_idx, puzzle_name in enumerate(pbar):
        puzzle_data = puzzles[puzzle_name]

        # Evaluate puzzle
        results = evaluate_single_puzzle(
            ttt=ttt,
            puzzle_name=puzzle_name,
            puzzle_data=puzzle_data,
            config=config,
            puzzle_id=puzzle_idx % model.config.num_puzzle_identifiers
        )

        # Store predictions
        all_predictions[puzzle_name] = {}
        for input_hash, pred_grid, confidence in results:
            pred_hash = grid_hash(pred_grid)
            pred_grids[pred_hash] = pred_grid

            all_predictions[puzzle_name].setdefault(input_hash, [])
            all_predictions[puzzle_name][input_hash].append((pred_hash, confidence))

    # Compute pass@k metrics
    correct = {k: 0.0 for k in config.pass_ks}

    for puzzle_name, puzzle_data in test_puzzles.items():
        if puzzle_name not in all_predictions:
            continue

        num_test = len(puzzle_data["test"])
        puzzle_correct = {k: 0 for k in config.pass_ks}

        for test_example in puzzle_data["test"]:
            input_hash = grid_hash(arc_grid_to_np(test_example["input"]))
            label_hash = grid_hash(arc_grid_to_np(test_example["output"]))

            preds = all_predictions[puzzle_name].get(input_hash, [])

            # Sort by confidence
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

            # Check pass@k for each k
            for k in config.pass_ks:
                for pred_hash, _ in preds_sorted[:k]:
                    if pred_hash == label_hash:
                        puzzle_correct[k] += 1
                        break

        # Average over test examples in this puzzle
        for k in config.pass_ks:
            correct[k] += puzzle_correct[k] / num_test

    # Average over all puzzles
    num_puzzles = len(test_puzzles)
    metrics = {
        f"pass@{k}": correct[k] / num_puzzles if num_puzzles > 0 else 0.0
        for k in config.pass_ks
    }

    # Build submission
    submission = {}
    for puzzle_name, puzzle_data in test_puzzles.items():
        submission[puzzle_name] = []

        for test_example in puzzle_data["test"]:
            input_hash = grid_hash(arc_grid_to_np(test_example["input"]))
            preds = all_predictions.get(puzzle_name, {}).get(input_hash, [])
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

            # Get top-2 predictions
            attempts = {}
            for i, (pred_hash, _) in enumerate(preds_sorted[:2]):
                if pred_hash in pred_grids:
                    attempts[f"attempt_{i+1}"] = pred_grids[pred_hash].tolist()

            # Pad if needed
            if len(attempts) == 0:
                attempts["attempt_1"] = [[0]]
                attempts["attempt_2"] = [[0]]
            elif len(attempts) == 1:
                attempts["attempt_2"] = attempts["attempt_1"]

            submission[puzzle_name].append(attempts)

    return {
        "metrics": metrics,
        "submission": submission
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TRM-Titans with Test-Time Training on ARC puzzles"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/arc-aug-1000",
        help="Path to ARC data directory"
    )
    parser.add_argument(
        "--ttt_steps", type=int, default=10,
        help="Number of test-time adaptation steps"
    )
    parser.add_argument(
        "--ttt_lr", type=float, default=0.01,
        help="Learning rate for test-time adaptation"
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs/ttt_eval",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--no_accumulate_memory", action="store_true",
        help="Disable memory accumulation across demo examples"
    )
    parser.add_argument(
        "--max_puzzles", type=int, default=0,
        help="Maximum number of puzzles to evaluate (0 = all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create config
    config = EvalConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        output_path=args.output_path,
        device=args.device,
        verbose=args.verbose,
        accumulate_memory=not args.no_accumulate_memory,
        max_puzzles=args.max_puzzles
    )

    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)

    # Load model configuration
    print(f"Loading model configuration from: {config.checkpoint}")
    model_cfg = load_model_config_from_checkpoint(config.checkpoint, config.data_path)

    # Print configuration
    print("\nModel Configuration:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")

    # Create model
    print("\nCreating model...")
    device = torch.device(config.device)
    with torch.device(device):
        model = TRM_Titans(model_cfg)

    # Load checkpoint
    load_checkpoint(model, config.checkpoint, device)

    # Freeze memory templates for test-time adaptation
    model.freeze_memory_templates()

    # Load test puzzles
    print(f"\nLoading test puzzles from: {config.data_path}")
    test_puzzles_path = os.path.join(config.data_path, "test_puzzles.json")
    with open(test_puzzles_path, "r") as f:
        puzzles = json.load(f)
    print(f"Loaded {len(puzzles)} puzzles")

    # Run evaluation
    print(f"\nStarting evaluation with TTT (steps={config.ttt_steps}, lr={config.ttt_lr})")
    print(f"Memory accumulation: {config.accumulate_memory}")

    results = evaluate_all_puzzles(model, puzzles, config)

    # Print metrics
    print("\n" + "="*50)
    print("Results:")
    for metric_name, value in results["metrics"].items():
        print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")

    # Save submission
    submission_path = os.path.join(config.output_path, "submission.json")
    with open(submission_path, "w") as f:
        json.dump(results["submission"], f)
    print(f"\nSubmission saved to: {submission_path}")

    # Save metrics
    metrics_path = os.path.join(config.output_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save config
    config_save_path = os.path.join(config.output_path, "eval_config.json")
    with open(config_save_path, "w") as f:
        json.dump({
            "checkpoint": config.checkpoint,
            "data_path": config.data_path,
            "ttt_steps": config.ttt_steps,
            "ttt_lr": config.ttt_lr,
            "accumulate_memory": config.accumulate_memory,
            "max_puzzles": config.max_puzzles,
            "device": config.device,
            "pass_ks": config.pass_ks,
        }, f, indent=2)
    print(f"Config saved to: {config_save_path}")


if __name__ == "__main__":
    main()
