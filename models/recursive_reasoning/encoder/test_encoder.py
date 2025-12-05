"""
Test script for EnhancedTRM encoder module.

This script verifies:
1. CLS token is at position 0 for q_head compatibility
2. Prefix length is exactly 16 tokens (matching puzzle_emb_len)
3. lm_head output skips first 16 positions
4. All stages work correctly
5. Parameter groups are properly separated

Usage:
    python -m models.recursive_reasoning.encoder.test_encoder
"""

import torch
import torch.nn as nn
from typing import Dict


def create_mock_batch(batch_size: int = 4, seq_len: int = 100) -> Dict[str, torch.Tensor]:
    """Create a mock batch for testing."""
    return {
        "inputs": torch.randint(0, 11, (batch_size, seq_len)),
        "labels": torch.randint(0, 11, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }


def create_mock_batch_with_fewshot(
    batch_size: int = 4,
    seq_len: int = 100,
    num_examples: int = 3,
) -> Dict[str, torch.Tensor]:
    """Create a mock batch with few-shot data."""
    batch = create_mock_batch(batch_size, seq_len)
    batch["few_shot_inputs"] = torch.randint(0, 11, (batch_size, num_examples, seq_len))
    batch["few_shot_outputs"] = torch.randint(0, 11, (batch_size, num_examples, seq_len))
    return batch


def test_grid_encoder():
    """Test GridEncoder standalone."""
    print("=" * 60)
    print("Testing GridEncoder")
    print("=" * 60)

    from .grid_encoder import GridEncoder, GridEncoderConfig

    config = GridEncoderConfig(
        vocab_size=11,
        hidden_size=512,
        num_layers=2,
        num_heads=4,
    )
    encoder = GridEncoder(config)

    # Test single grid
    grid = torch.randint(0, 11, (4, 100))  # [B, L]

    global_repr = encoder(grid, return_type="global")
    print(f"  Global repr shape: {global_repr.shape}")  # [4, 512]
    assert global_repr.shape == (4, 512), f"Expected (4, 512), got {global_repr.shape}"

    token_repr = encoder(grid, return_type="tokens")
    print(f"  Token repr shape: {token_repr.shape}")  # [4, 100, 512]
    assert token_repr.shape == (4, 100, 512), f"Expected (4, 100, 512), got {token_repr.shape}"

    # Test batched few-shot
    batched_grid = torch.randint(0, 11, (4, 3, 100))  # [B, N, L]
    global_repr = encoder(batched_grid, return_type="global")
    print(f"  Batched global repr shape: {global_repr.shape}")  # [4, 3, 512]
    assert global_repr.shape == (4, 3, 512), f"Expected (4, 3, 512), got {global_repr.shape}"

    print("  GridEncoder tests passed!")
    print()


def test_goal_encoder():
    """Test GoalEncoder."""
    print("=" * 60)
    print("Testing GoalEncoder")
    print("=" * 60)

    from .grid_encoder import GridEncoder, GridEncoderConfig
    from .goal_encoder import GoalEncoder, GoalEncoderConfig

    grid_config = GridEncoderConfig(hidden_size=512)
    grid_encoder = GridEncoder(grid_config)

    goal_config = GoalEncoderConfig(
        hidden_size=512,
        num_goal_tokens=16,
    )
    goal_encoder = GoalEncoder(grid_encoder, goal_config)

    few_shot_inputs = torch.randint(0, 11, (4, 3, 100))
    few_shot_outputs = torch.randint(0, 11, (4, 3, 100))

    # Input-only mode
    goal = goal_encoder(few_shot_inputs, use_output_info=False)
    print(f"  Goal (input-only) shape: {goal.shape}")  # [4, 16, 512]
    assert goal.shape == (4, 16, 512), f"Expected (4, 16, 512), got {goal.shape}"

    # Goal conditioning mode
    goal = goal_encoder(few_shot_inputs, few_shot_outputs, use_output_info=True)
    print(f"  Goal (conditioning) shape: {goal.shape}")  # [4, 16, 512]
    assert goal.shape == (4, 16, 512), f"Expected (4, 16, 512), got {goal.shape}"

    print("  GoalEncoder tests passed!")
    print()


def test_enhanced_trm_stage1():
    """Test EnhancedTRM Stage 1 (GridEncoder for input)."""
    print("=" * 60)
    print("Testing EnhancedTRM Stage 1")
    print("=" * 60)

    # Import TRM
    import sys
    sys.path.insert(0, "/home/ubuntu/TinyRecursiveModels")
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    from .wrappers import create_stage1_grid_encoder

    # Create base TRM
    trm_config = {
        "batch_size": 4,
        "seq_len": 100,
        "puzzle_emb_ndim": 512,
        "num_puzzle_identifiers": 1000,
        "vocab_size": 11,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
    }

    base_trm = TinyRecursiveReasoningModel_ACTV1(trm_config)
    puzzle_emb_len = base_trm.inner.puzzle_emb_len
    print(f"  puzzle_emb_len: {puzzle_emb_len}")
    assert puzzle_emb_len == 16, f"Expected puzzle_emb_len=16, got {puzzle_emb_len}"

    # Create Stage 1 EnhancedTRM
    enhanced = create_stage1_grid_encoder(base_trm, freeze_trm=False)

    # Check CLS token exists
    assert enhanced.cls_token is not None, "CLS token should exist for Stage 1"
    print(f"  CLS token shape: {enhanced.cls_token.shape}")  # [1, 1, 512]
    assert enhanced.cls_token.shape == (1, 1, 512), f"Expected (1, 1, 512), got {enhanced.cls_token.shape}"

    # Check prefix padding exists
    assert enhanced.prefix_padding is not None, "Prefix padding should exist for Stage 1"
    print(f"  Prefix padding shape: {enhanced.prefix_padding.shape}")  # [1, 15, 512]
    assert enhanced.prefix_padding.shape == (1, 15, 512), f"Expected (1, 15, 512), got {enhanced.prefix_padding.shape}"

    # Test forward pass
    batch = create_mock_batch(batch_size=4, seq_len=100)
    for k, v in batch.items():
        batch[k] = v.cuda() if torch.cuda.is_available() else v

    if torch.cuda.is_available():
        enhanced = enhanced.cuda()

    carry = enhanced.initial_carry(batch)
    new_carry, outputs = enhanced(carry, batch)

    print(f"  Output logits shape: {outputs['logits'].shape}")
    expected_logits_shape = (4, 100, 11)  # [B, L, vocab_size], skips prefix
    assert outputs['logits'].shape == expected_logits_shape, f"Expected {expected_logits_shape}, got {outputs['logits'].shape}"

    print(f"  q_halt_logits shape: {outputs['q_halt_logits'].shape}")
    expected_q_shape = (4,)
    assert outputs['q_halt_logits'].shape == expected_q_shape, f"Expected {expected_q_shape}, got {outputs['q_halt_logits'].shape}"

    print("  Stage 1 tests passed!")
    print()


def test_parameter_groups():
    """Test parameter group separation for optimizer."""
    print("=" * 60)
    print("Testing Parameter Groups")
    print("=" * 60)

    import sys
    sys.path.insert(0, "/home/ubuntu/TinyRecursiveModels")
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    from .wrappers import create_stage1_grid_encoder

    # Create base TRM
    trm_config = {
        "batch_size": 4,
        "seq_len": 100,
        "puzzle_emb_ndim": 512,
        "num_puzzle_identifiers": 1000,
        "vocab_size": 11,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
    }

    base_trm = TinyRecursiveReasoningModel_ACTV1(trm_config)
    enhanced = create_stage1_grid_encoder(base_trm, freeze_trm=False)

    # Get parameter groups
    param_groups = enhanced.get_parameter_groups(
        encoder_lr=1e-4,
        prefix_lr=1e-2,
        trm_lr=1e-4,
    )

    print(f"  Number of parameter groups: {len(param_groups)}")

    for group in param_groups:
        name = group["name"]
        lr = group["lr"]
        params = list(group["params"])
        num_params = sum(p.numel() for p in params)
        print(f"    {name}: {num_params:,} params, lr={lr}")

    # Verify groups exist
    group_names = {g["name"] for g in param_groups}
    assert "encoder" in group_names, "Should have encoder group"
    assert "prefix" in group_names, "Should have prefix group"
    assert "trm" in group_names, "Should have trm group"

    print("  Parameter group tests passed!")
    print()


def test_prefix_semantics():
    """Test that position 0 is properly used for q_head."""
    print("=" * 60)
    print("Testing Prefix Semantics (CLS at position 0)")
    print("=" * 60)

    import sys
    sys.path.insert(0, "/home/ubuntu/TinyRecursiveModels")
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    from .wrappers import create_stage1_grid_encoder, EnhancedTRM

    # Create base TRM
    trm_config = {
        "batch_size": 4,
        "seq_len": 100,
        "puzzle_emb_ndim": 512,
        "num_puzzle_identifiers": 1000,
        "vocab_size": 11,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
    }

    base_trm = TinyRecursiveReasoningModel_ACTV1(trm_config)
    enhanced = create_stage1_grid_encoder(base_trm, freeze_trm=False)

    # Test _get_prefix
    batch_size = 4
    prefix = enhanced._get_prefix(batch_size=batch_size, goal=None, puzzle_identifiers=None)

    print(f"  Prefix shape: {prefix.shape}")
    expected_prefix_shape = (4, 16, 512)
    assert prefix.shape == expected_prefix_shape, f"Expected {expected_prefix_shape}, got {prefix.shape}"

    # Verify CLS is at position 0
    cls_expanded = enhanced.cls_token.expand(batch_size, -1, -1)
    assert torch.allclose(prefix[:, 0:1, :], cls_expanded), "Position 0 should be CLS token"
    print("  Position 0 is CLS token: VERIFIED")

    # Verify padding is at positions 1-15
    pad_expanded = enhanced.prefix_padding.expand(batch_size, -1, -1)
    assert torch.allclose(prefix[:, 1:16, :], pad_expanded), "Positions 1-15 should be padding"
    print("  Positions 1-15 are padding: VERIFIED")

    print("  Prefix semantics tests passed!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EnhancedTRM Encoder Module Tests")
    print("=" * 60 + "\n")

    test_grid_encoder()
    test_goal_encoder()
    test_prefix_semantics()
    test_parameter_groups()

    # Skip GPU-requiring test if no GPU available
    if torch.cuda.is_available():
        test_enhanced_trm_stage1()
    else:
        print("Skipping Stage 1 test (no GPU available)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
