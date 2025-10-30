"""
Unit tests for JEPA contrastive learning implementation.

Run with:
    python -m pytest tests/test_jepa_contrastive.py -v
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config
)
from models.recursive_reasoning.jepa_contrastive import (
    JEPAContrastiveTRM,
    InfoNCEContrastiveTRM,
    create_augmented_batch,
    augment_arc_grid
)


def create_mock_trm_model(hidden_size=512, device='cpu'):
    """Create a small TRM model for testing."""
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=4,
        seq_len=916,  # 16 + 900
        vocab_size=11,
        num_puzzle_identifiers=100,
        puzzle_emb_ndim=0,
        H_cycles=2,
        L_cycles=2,
        H_layers=1,
        L_layers=2,
        hidden_size=hidden_size,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=8,
        halt_exploration_prob=0.1,
    )
    return TinyRecursiveReasoningModel_ACTV1(config).to(device)


def test_jepa_contrastive_forward():
    """Test JEPA contrastive forward pass."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256

    # Create model
    trm_model = create_mock_trm_model(hidden_size=hidden_size, device=device)
    jepa_model = JEPAContrastiveTRM(
        trm_model,
        hidden_size=hidden_size,
        proj_dim=128,
        ema_decay=0.99
    ).to(device)

    # Create batch
    batch = {
        "inputs": torch.randint(0, 11, (4, 900), device=device),
        "inputs_aug": torch.randint(0, 11, (4, 900), device=device),
        "puzzle_identifiers": torch.randint(0, 100, (4,), device=device)
    }

    # Forward pass
    loss = jepa_model(batch)

    # Assertions
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert loss.item() <= 4, "Cosine loss should be in [0, 4]"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.requires_grad, "Loss should have gradient"

    print(f"✅ JEPA forward pass: loss = {loss.item():.4f}")


def test_infonce_contrastive_forward():
    """Test InfoNCE contrastive forward pass."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256

    # Create model
    trm_model = create_mock_trm_model(hidden_size=hidden_size, device=device)
    infonce_model = InfoNCEContrastiveTRM(
        trm_model,
        hidden_size=hidden_size,
        proj_dim=128,
        temperature=0.07
    ).to(device)

    # Create batch
    batch = {
        "inputs": torch.randint(0, 11, (4, 900), device=device),
        "inputs_aug": torch.randint(0, 11, (4, 900), device=device),
        "puzzle_identifiers": torch.randint(0, 100, (4,), device=device)
    }

    # Forward pass
    loss = infonce_model(batch)

    # Assertions
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.requires_grad, "Loss should have gradient"

    print(f"✅ InfoNCE forward pass: loss = {loss.item():.4f}")


def test_ema_target_update():
    """Test EMA target encoder update."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256

    # Create model
    trm_model = create_mock_trm_model(hidden_size=hidden_size, device=device)
    jepa_model = JEPAContrastiveTRM(
        trm_model,
        hidden_size=hidden_size,
        ema_decay=0.9  # Lower for faster change
    ).to(device)

    # Get initial target weights
    target_param = list(jepa_model.target_encoder.parameters())[0]
    online_param = list(jepa_model.encoder.parameters())[0]
    initial_target = target_param.clone()
    initial_online = online_param.clone()

    # Simulate gradient update on online encoder
    online_param.data += 0.1

    # Update target encoder
    jepa_model.update_target_encoder()

    # Check target has changed
    updated_target = target_param.clone()
    assert not torch.allclose(initial_target, updated_target, atol=1e-5), \
        "Target should have changed after EMA update"

    # Check EMA formula: target = 0.9 * target + 0.1 * online
    expected = 0.9 * initial_target + 0.1 * online_param
    assert torch.allclose(updated_target, expected, atol=1e-5), \
        "EMA update formula incorrect"

    print("✅ EMA target update works correctly")


def test_augmentation():
    """Test data augmentation."""
    batch_size = 8
    inputs = torch.randint(0, 11, (batch_size, 900))

    # Apply augmentation
    inputs_aug = augment_arc_grid(
        inputs,
        grid_height=30,
        grid_width=30,
        apply_rotation=True,
        apply_flip=True
    )

    # Assertions
    assert inputs_aug.shape == inputs.shape, "Shape should be preserved"
    assert not torch.equal(inputs, inputs_aug), "Augmented should differ from original"

    print("✅ Data augmentation works")


def test_create_augmented_batch():
    """Test augmented batch creation."""
    batch = {
        "inputs": torch.randint(0, 11, (4, 900)),
        "puzzle_identifiers": torch.randint(0, 100, (4,))
    }

    # Test with augmentation function
    batch_aug = create_augmented_batch(batch, augmentation_fn=augment_arc_grid)
    assert "inputs_aug" in batch_aug, "Should add inputs_aug"
    assert batch_aug["inputs_aug"].shape == batch["inputs"].shape

    # Test with pre-existing inputs_aug
    batch_with_aug = {
        "inputs": torch.randint(0, 11, (4, 900)),
        "inputs_aug": torch.randint(0, 11, (4, 900)),
        "puzzle_identifiers": torch.randint(0, 100, (4,))
    }
    result = create_augmented_batch(batch_with_aug)
    assert torch.equal(result["inputs_aug"], batch_with_aug["inputs_aug"]), \
        "Should preserve existing inputs_aug"

    print("✅ Augmented batch creation works")


def test_backward_pass():
    """Test backward pass with gradient flow."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256

    # Create model
    trm_model = create_mock_trm_model(hidden_size=hidden_size, device=device)
    jepa_model = JEPAContrastiveTRM(
        trm_model,
        hidden_size=hidden_size,
        proj_dim=128
    ).to(device)

    # Create batch
    batch = {
        "inputs": torch.randint(0, 11, (4, 900), device=device),
        "inputs_aug": torch.randint(0, 11, (4, 900), device=device),
        "puzzle_identifiers": torch.randint(0, 100, (4,), device=device)
    }

    # Forward + backward
    loss = jepa_model(batch)
    loss.backward()

    # Check gradients exist for online encoder
    has_grads = False
    for param in jepa_model.encoder.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break

    assert has_grads, "Online encoder should have gradients"

    # Check target encoder has no gradients
    for param in jepa_model.target_encoder.parameters():
        assert not param.requires_grad, "Target encoder should not require gradients"

    print("✅ Backward pass with correct gradient flow")


def test_pooling_methods():
    """Test different pooling methods."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256

    batch = {
        "inputs": torch.randint(0, 11, (4, 900), device=device),
        "inputs_aug": torch.randint(0, 11, (4, 900), device=device),
        "puzzle_identifiers": torch.randint(0, 100, (4,), device=device)
    }

    for pool_method in ['mean', 'max', 'first', 'last']:
        trm_model = create_mock_trm_model(hidden_size=hidden_size, device=device)
        jepa_model = JEPAContrastiveTRM(
            trm_model,
            hidden_size=hidden_size,
            pool_method=pool_method
        ).to(device)

        loss = jepa_model(batch)
        assert not torch.isnan(loss), f"Loss NaN with pool_method={pool_method}"

    print("✅ All pooling methods work")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing JEPA Contrastive Learning Implementation")
    print("=" * 80)

    test_jepa_contrastive_forward()
    test_infonce_contrastive_forward()
    test_ema_target_update()
    test_augmentation()
    test_create_augmented_batch()
    test_backward_pass()
    test_pooling_methods()

    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
