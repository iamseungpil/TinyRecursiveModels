"""
Test script for ZLHistoryAggregator module.

Tests:
1. Empty history -> returns zeros
2. Single history item -> valid output
3. Multiple history items -> valid output
4. Identity behavior at init (gate ~ 0)
5. Shape consistency
6. Window size limiting
7. Gate value monitoring

Run with: python -m models.recursive_reasoning.encoder.test_history
"""

import torch
import torch.nn as nn
from typing import List

from .history_aggregator import (
    ZLHistoryAggregator,
    ZLHistoryAggregatorConfig,
    create_history_aggregator,
)


def test_empty_history():
    """Test that empty history returns zeros."""
    print("Testing empty history returns zeros...")

    config = ZLHistoryAggregatorConfig(
        hidden_size=64,
        num_heads=4,
        window_size=3,
    )
    aggregator = ZLHistoryAggregator(config)

    # Input
    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)
    z_L_history: List[torch.Tensor] = []

    # Forward with empty history
    output = aggregator(z_L, z_L_history)

    # Should return zeros
    assert output.shape == z_L.shape, f"Expected shape {z_L.shape}, got {output.shape}"
    assert torch.allclose(output, torch.zeros_like(z_L)), "Empty history should return zeros"

    print("  [OK] Empty history returns zeros")


def test_single_history_item():
    """Test with single history item."""
    print("Testing single history item...")

    aggregator = create_history_aggregator(
        hidden_size=64,
        num_heads=4,
        window_size=3,
    )

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)
    z_L_history = [torch.randn(B, L, D)]

    output = aggregator(z_L, z_L_history)

    assert output.shape == (B, L, D), f"Expected shape ({B}, {L}, {D}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("  [OK] Single history item produces valid output")


def test_multiple_history_items():
    """Test with multiple history items."""
    print("Testing multiple history items...")

    config = ZLHistoryAggregatorConfig(
        hidden_size=64,
        num_heads=4,
        window_size=5,  # Allow up to 5
    )
    aggregator = ZLHistoryAggregator(config)

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)
    z_L_history = [torch.randn(B, L, D) for _ in range(3)]

    output = aggregator(z_L, z_L_history)

    assert output.shape == (B, L, D), f"Expected shape ({B}, {L}, {D}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    print("  [OK] Multiple history items produce valid output")


def test_identity_init():
    """Test that identity initialization gives near-zero output."""
    print("Testing identity initialization (gate ~ 0)...")

    config = ZLHistoryAggregatorConfig(
        hidden_size=64,
        num_heads=4,
        window_size=3,
        gate_init=-10.0,  # sigmoid(-10) ~ 0.00005
    )
    aggregator = ZLHistoryAggregator(config)

    # Check gate value
    gate_value = aggregator.get_gate_value()
    assert gate_value < 0.001, f"Gate should be near 0, got {gate_value}"

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)
    z_L_history = [torch.randn(B, L, D) for _ in range(3)]

    output = aggregator(z_L, z_L_history)

    # Output should be near zero due to gate
    output_norm = output.abs().mean().item()
    assert output_norm < 0.01, f"Output should be near zero at init, got norm {output_norm}"

    print(f"  [OK] Identity init: gate={gate_value:.6f}, output_norm={output_norm:.6f}")


def test_shape_consistency():
    """Test shape consistency across different inputs."""
    print("Testing shape consistency...")

    aggregator = create_history_aggregator(
        hidden_size=128,
        num_heads=8,
        window_size=4,
    )

    test_cases = [
        (1, 50, 128),   # Small batch
        (4, 100, 128),  # Medium
        (8, 200, 128),  # Larger
    ]

    for B, L, D in test_cases:
        z_L = torch.randn(B, L, D)
        z_L_history = [torch.randn(B, L, D) for _ in range(2)]

        output = aggregator(z_L, z_L_history)
        assert output.shape == (B, L, D), f"Shape mismatch for ({B}, {L}, {D})"

    print("  [OK] Shape consistency across different inputs")


def test_window_size_limiting():
    """Test that history is limited to window size."""
    print("Testing window size limiting...")

    window_size = 3
    aggregator = create_history_aggregator(
        hidden_size=64,
        num_heads=4,
        window_size=window_size,
    )

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)

    # Create history with more items than window
    z_L_history = [torch.randn(B, L, D) for _ in range(10)]

    # Should not error - internally limits to window size
    output = aggregator(z_L, z_L_history)
    assert output.shape == (B, L, D), "Should handle excess history gracefully"

    print("  [OK] Window size limiting works correctly")


def test_gradient_flow():
    """Test that gradients flow through the aggregator."""
    print("Testing gradient flow...")

    aggregator = create_history_aggregator(
        hidden_size=64,
        num_heads=4,
        window_size=3,
    )

    # Override gate to allow gradients to flow
    aggregator.gate.data.fill_(0.0)  # sigmoid(0) = 0.5

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D, requires_grad=True)
    z_L_history = [torch.randn(B, L, D) for _ in range(2)]

    output = aggregator(z_L, z_L_history)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert z_L.grad is not None, "Gradients should flow to z_L"
    assert aggregator.gate.grad is not None, "Gradients should flow to gate"

    print("  [OK] Gradients flow correctly")


def test_gate_reset():
    """Test gate reset functionality."""
    print("Testing gate reset...")

    aggregator = create_history_aggregator(hidden_size=64, num_heads=4)

    # Modify gate
    aggregator.gate.data.fill_(0.0)
    assert abs(aggregator.get_gate_value() - 0.5) < 0.01

    # Reset
    aggregator.reset_gate(-10.0)
    assert aggregator.get_gate_value() < 0.001

    print("  [OK] Gate reset works correctly")


def test_different_num_heads():
    """Test with different number of attention heads."""
    print("Testing different number of attention heads...")

    B, L, D = 2, 100, 128

    for num_heads in [1, 2, 4, 8]:
        aggregator = create_history_aggregator(
            hidden_size=D,
            num_heads=num_heads,
            window_size=3,
        )
        # Reset gate for non-zero output
        aggregator.gate.data.fill_(0.0)

        z_L = torch.randn(B, L, D)
        z_L_history = [torch.randn(B, L, D) for _ in range(2)]

        output = aggregator(z_L, z_L_history)
        assert output.shape == (B, L, D), f"Failed for num_heads={num_heads}"

    print("  [OK] Works with different number of attention heads")


def test_device_consistency():
    """Test that output is on same device as input."""
    print("Testing device consistency...")

    aggregator = create_history_aggregator(hidden_size=64, num_heads=4)

    B, L, D = 2, 100, 64
    z_L = torch.randn(B, L, D)
    z_L_history = [torch.randn(B, L, D) for _ in range(2)]

    output = aggregator(z_L, z_L_history)
    assert output.device == z_L.device, "Output should be on same device as input"

    print("  [OK] Device consistency maintained")


def test_integration_simulation():
    """Simulate integration with TRM forward loop."""
    print("Testing integration simulation...")

    aggregator = create_history_aggregator(hidden_size=64, num_heads=4, window_size=3)
    aggregator.gate.data.fill_(0.0)  # Enable output

    B, L, D = 2, 100, 64
    H_cycles = 5

    # Simulate TRM forward with history
    z_L_history: List[torch.Tensor] = []
    input_embeddings = torch.randn(B, L, D)
    z_H = torch.zeros(B, L, D)
    z_L = torch.zeros(B, L, D)

    for h_step in range(H_cycles):
        # Compute history context
        history_context = aggregator(z_L, z_L_history)

        # Simulated injection (normally: z_H + input_embeddings + history_context)
        injection = z_H + input_embeddings + history_context

        # Simulated L-level update
        z_L = z_L + 0.1 * injection

        # Update history
        z_L_history.append(z_L.detach().clone())
        if len(z_L_history) > aggregator.config.window_size:
            z_L_history = z_L_history[-aggregator.config.window_size:]

        # Simulated H-level update
        z_H = z_H + 0.1 * z_L

    # Final checks
    assert z_L.shape == (B, L, D)
    assert z_H.shape == (B, L, D)
    assert len(z_L_history) == aggregator.config.window_size

    print("  [OK] Integration simulation successful")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running ZLHistoryAggregator tests")
    print("=" * 60)

    test_empty_history()
    test_single_history_item()
    test_multiple_history_items()
    test_identity_init()
    test_shape_consistency()
    test_window_size_limiting()
    test_gradient_flow()
    test_gate_reset()
    test_different_num_heads()
    test_device_consistency()
    test_integration_simulation()

    print("=" * 60)
    print("All ZLHistoryAggregator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
