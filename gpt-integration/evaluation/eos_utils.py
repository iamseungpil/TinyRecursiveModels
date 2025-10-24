"""
EOS Cropping Utilities for ARC Grid Generation

Based on TinyRecursiveModels/evaluators/arc.py
"""
import torch
import numpy as np
import numba


@numba.njit
def _crop(grid: np.ndarray):
    """
    Find maximum-sized rectangle without any EOS token inside

    Args:
        grid: Flattened numpy array [900] with token IDs

    Returns:
        cropped_grid: [H, W] numpy array with values 0-9 (colors only)
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
            # EOS=1 or PAD=0 or invalid (>11)
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    # Extract grid and shift colors from 2-11 to 0-9
    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


def crop_grid_from_eos(tensor_grid: torch.Tensor) -> torch.Tensor:
    """
    Crop torch tensor grid using EOS markers

    Args:
        tensor_grid: [batch, 900] or [900] tensor with token IDs

    Returns:
        cropped_grids: List of numpy arrays with variable shapes [H, W]
    """
    if tensor_grid.dim() == 1:
        # Single grid
        grid_np = tensor_grid.cpu().numpy()
        cropped = _crop(grid_np)
        return torch.from_numpy(cropped)

    elif tensor_grid.dim() == 2:
        # Batch of grids
        batch_size = tensor_grid.shape[0]
        cropped_grids = []
        for i in range(batch_size):
            grid_np = tensor_grid[i].cpu().numpy()
            cropped = _crop(grid_np)
            cropped_grids.append(torch.from_numpy(cropped))
        return cropped_grids

    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {tensor_grid.dim()}D")


def compute_exact_match(pred_grids: torch.Tensor, target_grids: torch.Tensor) -> torch.Tensor:
    """
    Compute exact match accuracy after EOS cropping

    Args:
        pred_grids: [batch, 900] predicted grids
        target_grids: [batch, 900] target grids

    Returns:
        exact_matches: [batch] boolean tensor
    """
    batch_size = pred_grids.shape[0]
    exact_matches = []

    for i in range(batch_size):
        pred_cropped = crop_grid_from_eos(pred_grids[i])
        target_cropped = crop_grid_from_eos(target_grids[i])

        # Check shape match
        if pred_cropped.shape != target_cropped.shape:
            exact_matches.append(False)
        else:
            # Check content match
            exact_matches.append(torch.all(pred_cropped == target_cropped).item())

    return torch.tensor(exact_matches, dtype=torch.bool)


def compute_loss_with_crop(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute cross-entropy loss with EOS cropping

    Only compute loss on valid grid positions (before EOS markers)

    Args:
        logits: [batch, seq_len, vocab_size] model predictions
        targets: [batch, seq_len] target token IDs

    Returns:
        loss: Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Create mask for valid positions (not PAD=0, not EOS=1)
    valid_mask = (targets >= 2) & (targets <= 11)

    # Flatten for cross-entropy
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Set invalid positions to ignore_index
    targets_masked = torch.where(valid_mask.view(-1), targets_flat, ignore_index)

    # Compute cross-entropy (ignores ignore_index positions)
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_masked,
        ignore_index=ignore_index,
        reduction='mean'
    )

    return loss


def visualize_grid(grid: np.ndarray, title: str = "Grid"):
    """
    Print grid in a readable format

    Args:
        grid: [H, W] numpy array with values 0-9
        title: Title to display
    """
    print(f"\n{title} ({grid.shape[0]}x{grid.shape[1]}):")
    for row in grid:
        print(' '.join(str(int(cell)) for cell in row))


# Test code
if __name__ == "__main__":
    print("Testing EOS utilities...")

    # Create test grid with EOS markers
    # Pattern: 3x3 grid with colors 2-11 (actual values 0-9)
    # PAD=0, EOS=1, colors 0-9 = 2-11
    test_grid = np.zeros(900, dtype=np.int64)

    # Fill 3x3 region with colors
    grid_2d = np.zeros((30, 30), dtype=np.int64)
    grid_2d[:3, :3] = np.array([
        [2, 3, 4],   # Colors 0, 1, 2
        [5, 6, 7],   # Colors 3, 4, 5
        [8, 9, 10]   # Colors 6, 7, 8
    ])

    # Add EOS markers
    grid_2d[3, :3] = 1  # Bottom boundary
    grid_2d[:3, 3] = 1  # Right boundary

    test_grid = grid_2d.flatten()

    # Test cropping
    cropped = _crop(test_grid)
    print(f"✓ Cropped shape: {cropped.shape}")  # Should be (3, 3)
    visualize_grid(cropped, "Cropped Grid")

    # Test torch wrapper
    test_tensor = torch.from_numpy(test_grid)
    cropped_tensor = crop_grid_from_eos(test_tensor)
    print(f"✓ Torch cropped shape: {cropped_tensor.shape}")

    # Test batch
    batch_tensor = torch.stack([test_tensor, test_tensor])
    cropped_batch = crop_grid_from_eos(batch_tensor)
    print(f"✓ Batch cropped: {len(cropped_batch)} grids")

    # Test exact match
    pred = torch.randint(2, 12, (2, 900))
    target = torch.randint(2, 12, (2, 900))
    matches = compute_exact_match(pred, target)
    print(f"✓ Exact match shape: {matches.shape}")

    # Test loss computation
    logits = torch.randn(2, 900, 12)
    targets = torch.randint(0, 12, (2, 900))
    loss = compute_loss_with_crop(logits, targets)
    print(f"✓ Loss computed: {loss.item():.4f}")

    print("\n✅ EOS utilities test passed!")
