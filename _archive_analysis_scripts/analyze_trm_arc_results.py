"""
Analyze TRM step 518071 evaluation results on ARC-AGI 1 validation set
Extract solved and unsolved problems, then visualize them
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_data():
    """Load submission and test puzzles"""
    print("📂 Loading evaluation results...")

    # Load TRM submission
    with open("/data/trm/checkpoints/pretrain_att_arc1concept_4/evaluator_ARC_step_518071/submission.json") as f:
        submission = json.load(f)

    # Load test puzzles (ground truth)
    with open("/data/arc1concept-aug-1000/test_puzzles.json") as f:
        test_puzzles = json.load(f)

    print(f"✅ Loaded {len(submission)} submissions")
    print(f"✅ Loaded {len(test_puzzles)} test puzzles")

    return submission, test_puzzles


def compare_grids(pred, gt):
    """Check if two grids are exactly equal"""
    if pred is None or gt is None:
        return False

    pred_arr = np.array(pred)
    gt_arr = np.array(gt)

    # Check shape
    if pred_arr.shape != gt_arr.shape:
        return False

    # Check exact match
    return np.array_equal(pred_arr, gt_arr)


def analyze_results(submission, test_puzzles):
    """Compare submission with ground truth"""
    print("\n🔍 Analyzing results...")

    solved = []
    unsolved = []

    for task_id, predictions in submission.items():
        if task_id not in test_puzzles:
            continue

        # Get ground truth
        gt_output = test_puzzles[task_id]["test"][0]["output"]

        # Check if any prediction matches
        is_correct = False
        pred_output = None

        if isinstance(predictions, list) and len(predictions) > 0:
            # Check first attempt
            pred_dict = predictions[0]
            if "attempt_1" in pred_dict:
                pred_output = pred_dict["attempt_1"]
                is_correct = compare_grids(pred_output, gt_output)

        # Store result
        result = {
            "task_id": task_id,
            "puzzle": test_puzzles[task_id],
            "prediction": pred_output,
            "ground_truth": gt_output,
            "is_correct": is_correct
        }

        if is_correct:
            solved.append(result)
        else:
            unsolved.append(result)

    print(f"\n📊 Results:")
    print(f"  Solved: {len(solved)} ({len(solved)/len(submission)*100:.1f}%)")
    print(f"  Unsolved: {len(unsolved)} ({len(unsolved)/len(submission)*100:.1f}%)")

    return solved, unsolved


def visualize_problem(result, output_path, problem_type="solved"):
    """Visualize a single ARC problem"""

    # ARC color palette
    colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]

    def draw_grid(ax, grid, title):
        """Draw a single grid"""
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

    puzzle = result["puzzle"]
    train_examples = puzzle["train"]
    test_input = puzzle["test"][0]["input"]
    ground_truth = result["ground_truth"]
    prediction = result["prediction"]

    # Create figure
    num_train = len(train_examples)
    fig = plt.figure(figsize=(16, 4 * (num_train + 1)))

    status = "✅ SOLVED" if problem_type == "solved" else "❌ UNSOLVED"
    fig.suptitle(
        f"{status} - Task: {result['task_id']}\n",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # Training examples
    for idx, example in enumerate(train_examples):
        # Input
        ax_in = plt.subplot(num_train + 1, 3, idx * 3 + 1)
        draw_grid(ax_in, example["input"], f"Train {idx+1} Input")

        # Output
        ax_out = plt.subplot(num_train + 1, 3, idx * 3 + 2)
        draw_grid(ax_out, example["output"], f"Train {idx+1} Output")

        # Empty for third column
        ax_empty = plt.subplot(num_train + 1, 3, idx * 3 + 3)
        ax_empty.axis('off')

    # Test example
    # Test input
    ax_test_in = plt.subplot(num_train + 1, 3, num_train * 3 + 1)
    draw_grid(ax_test_in, test_input, "Test Input")

    # Ground truth
    ax_test_gt = plt.subplot(num_train + 1, 3, num_train * 3 + 2)
    draw_grid(ax_test_gt, ground_truth, "Ground Truth")

    # TRM prediction
    ax_test_pred = plt.subplot(num_train + 1, 3, num_train * 3 + 3)
    if prediction is not None:
        draw_grid(ax_test_pred, prediction, "TRM Prediction")
    else:
        ax_test_pred.text(0.5, 0.5, "No prediction", ha='center', va='center')
        ax_test_pred.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  💾 Saved: {output_path}")


def main():
    # Load data
    submission, test_puzzles = load_data()

    # Analyze results
    solved, unsolved = analyze_results(submission, test_puzzles)

    # Create output directory
    output_dir = Path("/home/ubuntu/TinyRecursiveModels/trm_arc_visualizations")
    output_dir.mkdir(exist_ok=True)

    # Visualize 3 solved and 3 unsolved
    num_solved = min(3, len(solved))
    num_unsolved = min(3, len(unsolved))

    print(f"\n🎨 Visualizing {num_solved} solved and {num_unsolved} unsolved problems...")

    # Visualize solved
    for idx in range(num_solved):
        result = solved[idx]
        output_path = output_dir / f"solved_{idx+1}_{result['task_id']}.png"
        visualize_problem(result, str(output_path), "solved")

    # Visualize unsolved
    for idx in range(num_unsolved):
        result = unsolved[idx]
        output_path = output_dir / f"unsolved_{idx+1}_{result['task_id']}.png"
        visualize_problem(result, str(output_path), "unsolved")

    # Save summary
    summary = {
        "checkpoint": "/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071",
        "step": 518071,
        "total_evaluated": len(submission),
        "num_solved": len(solved),
        "num_unsolved": len(unsolved),
        "accuracy": len(solved) / len(submission) if len(submission) > 0 else 0,
        "solved_task_ids": [r["task_id"] for r in solved[:num_solved]],
        "unsolved_task_ids": [r["task_id"] for r in unsolved[:num_unsolved]]
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Done! Results saved in: {output_dir}")
    print(f"  📊 Summary: {summary_path}")
    print(f"\n📈 Final Accuracy: {summary['accuracy']:.2%} ({summary['num_solved']}/{summary['total_evaluated']})")


if __name__ == "__main__":
    main()
