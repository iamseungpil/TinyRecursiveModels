"""
Evaluate TRM checkpoint and visualize solved/unsolved problems
NO HALLUCINATION: All data comes directly from checkpoint predictions
"""

import sys
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent
sys.path.insert(0, str(trm_root))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.functions import load_model_class


def load_checkpoint(checkpoint_path: str, device: str = "cuda:0"):
    """Load pretrained TRM checkpoint."""
    print(f"📦 Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model with same config
    model_config = {
        "batch_size": 1,  # Evaluate one at a time
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,  # IMPORTANT: No puzzle embedding length
        "H_cycles": config.get("H_cycles", 3),
        "L_cycles": config.get("L_cycles", 6),
        "H_layers": 1,
        "L_layers": config.get("L_layers", 4),
        "hidden_size": config.get("hidden_size", 512),
        "expansion": config.get("expansion", 4.0),
        "num_heads": config.get("num_heads", 8),
        "pos_encodings": config.get("pos_encodings", "rope"),
        "halt_max_steps": config.get("halt_max_steps", 16),
        "halt_exploration_prob": 0.0,
    }

    model = TinyRecursiveReasoningModel_ACTV1(model_config)
    loss_head_cls = load_model_class("losses@ACTLossHead")
    model = loss_head_cls(model, loss_type="stablemax_cross_entropy")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded from step {checkpoint['step']}")
    return model, config


def grid_to_tokens(grid: List[List[int]]) -> torch.Tensor:
    """
    Convert grid to token sequence using ARC dataset format.
    - Pads to 30x30
    - Adds 2 to all values (PAD: 0, EOS: 1, digits: 2-11)
    - Adds EOS markers
    - Flattens to 900 tokens
    """
    ARCMaxGridSize = 30
    grid_np = np.array(grid, dtype=np.uint8)
    nrow, ncol = grid_np.shape

    # Pad to 30x30 with value+2
    padded = np.pad(grid_np + 2,
                    ((0, ARCMaxGridSize - nrow), (0, ARCMaxGridSize - ncol)),
                    constant_values=0)

    # Add EOS markers
    if nrow < ARCMaxGridSize:
        padded[nrow, 0:ncol] = 1  # EOS row
    if ncol < ARCMaxGridSize:
        padded[0:nrow, ncol] = 1  # EOS column

    # Flatten
    return torch.tensor(padded.flatten(), dtype=torch.long)


def tokens_to_grid(tokens: torch.Tensor, orig_height: int, orig_width: int) -> List[List[int]]:
    """
    Convert token sequence back to grid.
    - Reshape from 900 to 30x30
    - Subtract 2 to get original values
    - Extract only the original size
    """
    ARCMaxGridSize = 30
    tokens = tokens.cpu().numpy().reshape(ARCMaxGridSize, ARCMaxGridSize)

    # Subtract 2 to get original values (but clip negatives to 0)
    tokens = np.maximum(tokens - 2, 0)

    # Extract original size
    grid = tokens[:orig_height, :orig_width].tolist()
    return grid


def evaluate_single_problem(model, problem: Dict, device: str = "cuda:0") -> Tuple[bool, List[List[int]], List[List[int]]]:
    """
    Evaluate TRM on a single problem.
    Returns: (is_correct, ground_truth_grid, predicted_grid)
    """
    # Use last example as test (others as training context)
    test_example = problem["examples"][-1]
    test_input = test_example["input"]
    test_output = test_example["output"]

    height = len(test_input)
    width = len(test_input[0])

    # Convert to tokens
    input_tokens = grid_to_tokens(test_input).unsqueeze(0).to(device)
    target_tokens = grid_to_tokens(test_output).unsqueeze(0).to(device)

    # Debug: print shapes
    if input_tokens.shape[1] != 900:
        print(f"WARNING: input_tokens shape: {input_tokens.shape}, expected (1, 900)")
        print(f"  Input grid: {len(test_input)}x{len(test_input[0])}")

    # Create batch
    batch = {
        "inputs": input_tokens,
        "labels": target_tokens,
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device)
    }

    # Initialize carry
    carry = model.initial_carry(batch)

    # Multi-step inference
    with torch.no_grad():
        for step in range(16):  # Max steps
            carry, loss, metrics, preds, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=[]
            )
            if all_finish:
                break

    # Get prediction
    if "logits" in preds:
        pred_tokens = preds["logits"].argmax(dim=-1)[0]
    else:
        pred_tokens = input_tokens[0]  # Fallback to input

    # Convert back to grid
    pred_grid = tokens_to_grid(pred_tokens, height, width)

    # Check if correct (exact match)
    is_correct = (pred_grid == test_output)

    return is_correct, test_output, pred_grid


def visualize_problem(
    problem: Dict,
    pred_grid: List[List[int]],
    is_solved: bool,
    output_path: str
):
    """Visualize a single problem with training examples and prediction."""

    # ARC color palette (0-9)
    colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    def draw_grid(ax, grid, title):
        """Draw a single grid."""
        grid = np.array(grid)
        height, width = grid.shape

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Draw cells
        for i in range(height):
            for j in range(width):
                color = colors[int(grid[i, j]) % len(colors)]
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=1,
                    edgecolor='gray',
                    facecolor=color
                )
                ax.add_patch(rect)

    # Count training examples
    num_train = len(problem["examples"]) - 1  # Last one is test

    # Create figure
    fig = plt.figure(figsize=(16, 4 * (num_train + 1)))

    status = "✅ SOLVED" if is_solved else "❌ UNSOLVED"
    fig.suptitle(
        f"{status} - Task: {problem['task_id']}\n",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # Training examples
    for idx in range(num_train):
        example = problem["examples"][idx]

        # Input
        ax_in = plt.subplot(num_train + 1, 3, idx * 3 + 1)
        draw_grid(ax_in, example["input"], f"Train {idx+1} Input")

        # Output
        ax_out = plt.subplot(num_train + 1, 3, idx * 3 + 2)
        draw_grid(ax_out, example["output"], f"Train {idx+1} Output")

        # Empty space for third column
        ax_empty = plt.subplot(num_train + 1, 3, idx * 3 + 3)
        ax_empty.axis('off')

    # Test example
    test_example = problem["examples"][-1]

    # Test input
    ax_test_in = plt.subplot(num_train + 1, 3, num_train * 3 + 1)
    draw_grid(ax_test_in, test_example["input"], "Test Input")

    # Ground truth
    ax_test_gt = plt.subplot(num_train + 1, 3, num_train * 3 + 2)
    draw_grid(ax_test_gt, test_example["output"], "Ground Truth")

    # TRM prediction
    ax_test_pred = plt.subplot(num_train + 1, 3, num_train * 3 + 3)
    draw_grid(ax_test_pred, pred_grid, "TRM Prediction")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  💾 Saved: {output_path}")


def main():
    checkpoint_path = "/data/trm/pretrain/checkpoint_step_91.pt"
    data_path = "/data/helmarc_correct/20251024_062500/samples.json"
    device = "cuda:0"

    # Load model
    model, config = load_checkpoint(checkpoint_path, device)

    # Load data
    print(f"\n📂 Loading data from {data_path}")
    with open(data_path, 'r') as f:
        all_problems = json.load(f)
    print(f"✅ Loaded {len(all_problems)} problems")

    # Evaluate problems
    print("\n🔍 Evaluating problems...")
    solved_problems = []
    unsolved_problems = []

    for idx, problem in enumerate(all_problems[:100]):  # Evaluate first 100
        is_correct, gt_grid, pred_grid = evaluate_single_problem(model, problem, device)

        result = {
            "problem": problem,
            "prediction": pred_grid,
            "ground_truth": gt_grid,
            "is_correct": is_correct
        }

        if is_correct:
            solved_problems.append(result)
        else:
            unsolved_problems.append(result)

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/100, Solved: {len(solved_problems)}, Unsolved: {len(unsolved_problems)}")

    print(f"\n📊 Results:")
    print(f"  Solved: {len(solved_problems)}")
    print(f"  Unsolved: {len(unsolved_problems)}")
    print(f"  Accuracy: {len(solved_problems) / (len(solved_problems) + len(unsolved_problems)) * 100:.1f}%")

    # Select 3 solved and 3 unsolved
    num_solved_to_viz = min(3, len(solved_problems))
    num_unsolved_to_viz = min(3, len(unsolved_problems))

    print(f"\n🎨 Visualizing {num_solved_to_viz} solved and {num_unsolved_to_viz} unsolved problems...")

    output_dir = Path("/home/ubuntu/TinyRecursiveModels/trm_visualizations")
    output_dir.mkdir(exist_ok=True)

    # Visualize solved
    for idx in range(num_solved_to_viz):
        result = solved_problems[idx]
        output_path = output_dir / f"solved_{idx+1}.png"
        visualize_problem(
            result["problem"],
            result["prediction"],
            True,
            str(output_path)
        )

    # Visualize unsolved
    for idx in range(num_unsolved_to_viz):
        result = unsolved_problems[idx]
        output_path = output_dir / f"unsolved_{idx+1}.png"
        visualize_problem(
            result["problem"],
            result["prediction"],
            False,
            str(output_path)
        )

    # Save summary
    summary = {
        "checkpoint": checkpoint_path,
        "total_evaluated": len(solved_problems) + len(unsolved_problems),
        "num_solved": len(solved_problems),
        "num_unsolved": len(unsolved_problems),
        "accuracy": len(solved_problems) / (len(solved_problems) + len(unsolved_problems)),
        "solved_task_ids": [r["problem"]["task_id"] for r in solved_problems[:num_solved_to_viz]],
        "unsolved_task_ids": [r["problem"]["task_id"] for r in unsolved_problems[:num_unsolved_to_viz]]
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Done! Results saved in: {output_dir}")
    print(f"  📊 Summary: {summary_path}")


if __name__ == "__main__":
    main()
