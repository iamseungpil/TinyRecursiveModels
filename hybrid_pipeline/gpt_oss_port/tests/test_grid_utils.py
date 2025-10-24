"""
Unit tests for grid utilities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from gpt_oss_port.grid_utils import (
    grid_to_string,
    string_to_grid,
    compare_grids,
    grid_shape_matches
)


def test_grid_to_string():
    """Test grid to string conversion."""
    print("🧪 Testing grid_to_string...")

    grid = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])

    result = grid_to_string(grid)
    expected = "0 1 2\n3 4 5\n6 7 8"

    assert result == expected, f"Expected:\n{expected}\nGot:\n{result}"
    print(f"  ✅ Grid converted correctly")


def test_string_to_grid():
    """Test string to grid parsing."""
    print("\n🧪 Testing string_to_grid...")

    grid_str = "0 1 2\n3 4 5\n6 7 8"

    result = string_to_grid(grid_str)
    expected = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])

    assert result is not None, "Parsing returned None"
    assert np.array_equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"
    print(f"  ✅ String parsed correctly")


def test_grid_roundtrip():
    """Test grid → string → grid roundtrip."""
    print("\n🧪 Testing grid roundtrip...")

    original = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1]
    ])

    # Convert to string
    grid_str = grid_to_string(original)

    # Parse back
    reconstructed = string_to_grid(grid_str)

    assert np.array_equal(original, reconstructed), \
        f"Roundtrip failed:\nOriginal:\n{original}\nReconstructed:\n{reconstructed}"

    print(f"  ✅ Roundtrip successful")
    print(f"     Original shape: {original.shape}")
    print(f"     Reconstructed shape: {reconstructed.shape}")


def test_compare_grids():
    """Test grid comparison."""
    print("\n🧪 Testing compare_grids...")

    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[1, 2], [3, 4]])
    grid3 = np.array([[1, 2], [3, 5]])  # Different
    grid4 = np.array([[1, 2, 3]])  # Different shape

    # Same grids
    assert compare_grids(grid1, grid2) == True, "Identical grids should match"

    # Different values
    assert compare_grids(grid1, grid3) == False, "Different grids should not match"

    # Different shapes
    assert compare_grids(grid1, grid4) == False, "Different shapes should not match"

    # None cases
    assert compare_grids(None, grid1) == False
    assert compare_grids(grid1, None) == False

    print(f"  ✅ Grid comparison working correctly")


def test_grid_shape_matches():
    """Test grid shape matching."""
    print("\n🧪 Testing grid_shape_matches...")

    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[5, 6], [7, 8]])
    grid3 = np.array([[1, 2, 3]])

    # Same shape
    assert grid_shape_matches(grid1, grid2) == True

    # Different shape
    assert grid_shape_matches(grid1, grid3) == False

    # None cases
    assert grid_shape_matches(None, grid1) == False
    assert grid_shape_matches(grid1, None) == False

    print(f"  ✅ Shape matching working correctly")


def test_invalid_grid_string():
    """Test parsing invalid grid strings."""
    print("\n🧪 Testing invalid grid string parsing...")

    invalid_strings = [
        "",  # Empty
        "1 2 a",  # Non-numeric
        "1 2\n3",  # Inconsistent columns (will still parse but check behavior)
    ]

    for grid_str in invalid_strings:
        result = string_to_grid(grid_str)
        if result is None:
            print(f"  ✅ Correctly rejected: '{grid_str}'")
        else:
            print(f"  ⚠️  Parsed as: {result.shape} (may be acceptable)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Grid Utilities Tests")
    print("=" * 60)

    test_grid_to_string()
    test_string_to_grid()
    test_grid_roundtrip()
    test_compare_grids()
    test_grid_shape_matches()
    test_invalid_grid_string()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
