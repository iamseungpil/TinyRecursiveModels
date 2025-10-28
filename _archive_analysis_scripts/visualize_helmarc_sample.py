#!/usr/bin/env python3
"""
Visualize HelmARC sample with GPT-OSS analysis
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky
    '#870C25',  # 9: brown
]

def plot_grid(ax, grid, title):
    """Plot a single grid with ARC colors"""
    grid = np.array(grid)
    height, width = grid.shape

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Draw cells
    for i in range(height):
        for j in range(width):
            color = ARC_COLORS[int(grid[i, j])]
            rect = patches.Rectangle((j, height-1-i), 1, 1,
                                     linewidth=1, edgecolor='white',
                                     facecolor=color)
            ax.add_patch(rect)

    # Draw grid lines
    for i in range(height + 1):
        ax.plot([0, width], [i, i], 'w-', linewidth=0.5)
    for j in range(width + 1):
        ax.plot([j, j], [0, height], 'w-', linewidth=0.5)

def visualize_helmarc_sample(helmarc_data, analysis_data, output_dir):
    """
    Visualize HelmARC sample with analysis

    Args:
        helmarc_data: HelmARC sample dict
        analysis_data: GPT-OSS analysis dict
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_id = helmarc_data['task_id']

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'HelmARC Sample: {task_id}\nProgram: {helmarc_data["program"]}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Get examples
    examples = helmarc_data['examples']
    test = helmarc_data['test'][0]

    # Calculate number of training examples
    n_train = len(examples)

    # Create subplot grid
    # Top row: Training examples
    # Bottom row: Test input and output
    for idx, example in enumerate(examples[:2]):  # Show first 2 training examples
        # Train input
        ax = plt.subplot(2, 4, idx*2 + 1)
        plot_grid(ax, example['input'], f'Train {idx+1} Input')

        # Train output
        ax = plt.subplot(2, 4, idx*2 + 2)
        plot_grid(ax, example['output'], f'Train {idx+1} Output')

    # Test input
    ax = plt.subplot(2, 4, 5)
    plot_grid(ax, test['input'], 'Test Input')

    # Test output (Ground Truth)
    ax = plt.subplot(2, 4, 6)
    plot_grid(ax, test['output'], 'Ground Truth')

    # Add analysis summary as text
    ax = plt.subplot(2, 4, (7, 8))
    ax.axis('off')

    # Extract key info
    inference_time = analysis_data.get('inference_time', 0)
    attempts = analysis_data.get('attempts_needed', 0)

    # Get first few lines of analysis
    full_response = analysis_data.get('full_response', '')
    lines = full_response.split('\n')[:15]
    summary_text = '\n'.join(lines)
    if len(lines) >= 15:
        summary_text += '\n...'

    info_text = f"""GPT-OSS Analysis Summary:

Model: {analysis_data.get('model_name', 'N/A')}
Inference Time: {inference_time:.1f}s
Attempts: {attempts}
Timestamp: {analysis_data.get('timestamp', 'N/A')}

Analysis Preview:
{summary_text}
"""

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    img_path = output_dir / f'{task_id}_visualization.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {img_path}")
    plt.close()

    # Save analysis as JSON
    json_path = output_dir / f'{task_id}_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved analysis JSON: {json_path}")

    # Save full analysis as text file for easy reading
    txt_path = output_dir / f'{task_id}_analysis.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Program: {helmarc_data['program']}\n")
        f.write(f"Model: {analysis_data.get('model_name', 'N/A')}\n")
        f.write(f"Inference Time: {inference_time:.1f}s\n")
        f.write(f"Attempts: {attempts}\n")
        f.write(f"Timestamp: {analysis_data.get('timestamp', 'N/A')}\n")
        f.write("="*70 + "\n\n")
        f.write("FULL ANALYSIS:\n\n")
        f.write(full_response)
    print(f"✅ Saved analysis text: {txt_path}")

    return img_path, json_path, txt_path

if __name__ == "__main__":
    # Load data
    helmarc_path = '/data/helmarc_correct/20251024_062500/samples.json'
    checkpoint_path = '/data/helmarc_analyzed/gpu1/analysis_checkpoint_9.json'
    output_dir = '/home/ubuntu/TinyRecursiveModels/helmarc_visualization'

    print("Loading HelmARC data...")
    with open(helmarc_path, 'r') as f:
        helmarc_samples = json.load(f)

    print("Loading GPT-OSS analysis...")
    with open(checkpoint_path, 'r') as f:
        analysis_samples = json.load(f)

    # Visualize first 5 samples
    print(f"\nGenerating visualizations for first 5 samples...")
    for i in range(min(5, len(analysis_samples))):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/5")
        print(f"{'='*70}")

        analysis = analysis_samples[i]
        sample_idx = analysis['sample_index']
        task_id = analysis['task_id']

        # Get corresponding HelmARC data
        helmarc_data = helmarc_samples[sample_idx]

        # Verify task IDs match
        assert helmarc_data['task_id'] == task_id, f"Task ID mismatch: {helmarc_data['task_id']} != {task_id}"

        # Visualize
        img_path, json_path, txt_path = visualize_helmarc_sample(
            helmarc_data, analysis, output_dir
        )

    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}")
