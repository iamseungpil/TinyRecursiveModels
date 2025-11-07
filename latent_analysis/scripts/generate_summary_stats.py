"""
Generate summary statistics from completed puzzle analyses.
"""

import os
import json
from pathlib import Path

def main():
    results_dir = '/home/ubuntu/TinyRecursiveModels/latent_analysis/results/comprehensive_30_puzzles'

    # Find completed analyses
    puzzle_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith('puzzle_')])

    completed = []
    for puzzle_dir in puzzle_dirs:
        puzzle_path = os.path.join(results_dir, puzzle_dir)
        if os.path.exists(os.path.join(puzzle_path, 'joint_trajectory.png')):
            puzzle_idx = int(puzzle_dir.split('_')[1])
            completed.append({
                'puzzle_idx': puzzle_idx,
                'puzzle_dir': puzzle_dir,
                'has_grid_evolution': os.path.exists(os.path.join(puzzle_path, 'grid_evolution.png')),
                'has_joint_trajectory': True,
            })

    # Create summary
    summary = {
        'total_attempted': len(puzzle_dirs),
        'total_completed': len(completed),
        'completion_rate': len(completed) / len(puzzle_dirs) if puzzle_dirs else 0,
        'completed_puzzles': completed,
    }

    # Save summary
    output_path = os.path.join(results_dir, 'analysis_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("="*80)
    print("TRM Latent Space Analysis - Summary Statistics")
    print("="*80)
    print(f"\nTotal puzzles attempted: {summary['total_attempted']}")
    print(f"Successfully completed: {summary['total_completed']}")
    print(f"Completion rate: {summary['completion_rate']*100:.1f}%")
    print(f"\nCompleted puzzle indices:")
    for p in completed:
        print(f"  - {p['puzzle_idx']}")
    print(f"\nSummary saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
