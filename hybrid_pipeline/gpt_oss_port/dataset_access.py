"""
Dataset Access Wrapper - Interfaces with existing dataset/ module

NO DUPLICATION: Imports from /home/ubuntu/TinyRecursiveModels/dataset/
"""

import sys
import os
from pathlib import Path

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

from dataset.build_arc_dataset import (
    arc_grid_to_np,
    ARCMaxGridSize,
    ARCPuzzle,
    load_puzzles_arcagi,
    DataProcessConfig
)
from dataset.common import (
    PuzzleDatasetMetadata,
    dihedral_transform,
    inverse_dihedral_transform
)

# Re-export for convenience
__all__ = [
    'arc_grid_to_np',
    'ARCMaxGridSize',
    'ARCPuzzle',
    'load_puzzles_arcagi',
    'DataProcessConfig',
    'PuzzleDatasetMetadata',
    'dihedral_transform',
    'inverse_dihedral_transform'
]
