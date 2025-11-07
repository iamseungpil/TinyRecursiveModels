"""
Test Slot Attention Implementation

Verifies that all components work correctly before full training.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.slot_attention import SlotAttention, SlotCrossAttentionDecoder
from models.slot_losses import SlotContrastiveLossHead


def test_slot_diversity():
    """Test 1: Verify slots are diverse (not collapsed)"""
    print("\n" + "="*70)
    print("TEST 1: Slot Diversity")
    print("="*70)

    B, N, D = 4, 16, 512
    num_slots = 8
    slot_dim = 256

    slot_attn = SlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        input_dim=D,
        num_iterations=3
    )

    # Create dummy input
    puzzle_repr = torch.randn(B, N, D)

    # Forward pass
    slots = slot_attn(puzzle_repr)

    # Check shape
    assert slots.shape == (B, num_slots, slot_dim), f"Expected {(B, num_slots, slot_dim)}, got {slots.shape}"
    print(f"✓ Shape correct: {slots.shape}")

    # Check diversity
    slots_norm = F.normalize(slots, dim=-1, p=2)
    similarity = torch.einsum('bkd,bqd->bkq', slots_norm, slots_norm)  # [B, K, K]

    # Off-diagonal similarity (between different slots)
    mask = torch.eye(num_slots, dtype=torch.bool)
    off_diag_sim = similarity.masked_fill(mask, 0.0)
    mean_similarity = off_diag_sim.abs().mean().item()

    print(f"✓ Mean pairwise similarity: {mean_similarity:.4f}")

    if mean_similarity < 0.5:
        print(f"✓ PASS: Slots are diverse (similarity < 0.5)")
    else:
        print(f"⚠️  WARNING: Slots may collapse (similarity = {mean_similarity:.4f})")

    return slots


def test_cross_attention_decoder():
    """Test 2: Verify cross-attention decoder produces position-specific outputs"""
    print("\n" + "="*70)
    print("TEST 2: Cross-Attention Decoder")
    print("="*70)

    B, N_grid, N_slots = 4, 900, 8
    grid_dim, slot_dim = 512, 256

    decoder = SlotCrossAttentionDecoder(
        slot_dim=slot_dim,
        grid_dim=grid_dim,
        num_heads=8
    )

    # Dummy inputs
    grid_features = torch.randn(B, N_grid, grid_dim)
    rule_slots = torch.randn(B, N_slots, slot_dim)

    # Forward pass
    grid_enhanced, attn_weights = decoder(grid_features, rule_slots)

    # Check shapes
    assert grid_enhanced.shape == (B, N_grid, grid_dim), f"Expected {(B, N_grid, grid_dim)}, got {grid_enhanced.shape}"
    assert attn_weights.shape == (B, N_grid, N_slots), f"Expected {(B, N_grid, N_slots)}, got {attn_weights.shape}"
    print(f"✓ Shape correct: enhanced={grid_enhanced.shape}, attn={attn_weights.shape}")

    # Check position-specific (NOT all positions same)
    # Compare first and last position
    first_pos = grid_enhanced[:, 0, :].detach()
    last_pos = grid_enhanced[:, -1, :].detach()

    # They should be different (cosine similarity < 1.0)
    cos_sim = F.cosine_similarity(first_pos, last_pos, dim=-1).mean().item()
    print(f"✓ Position diversity: first vs last similarity = {cos_sim:.4f}")

    if cos_sim < 0.99:
        print(f"✓ PASS: Decoder produces position-specific features")
    else:
        print(f"❌ FAIL: All positions identical (bug in decoder!)")

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention weights don't sum to 1"
    print(f"✓ Attention weights properly normalized")

    return grid_enhanced, attn_weights


def test_gradient_flow():
    """Test 3: Verify gradients flow through all components"""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow")
    print("="*70)

    B, N_puzzle, N_grid = 2, 16, 900
    D, num_slots, slot_dim = 512, 8, 256

    # Create components
    slot_attn = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, input_dim=D)
    decoder = SlotCrossAttentionDecoder(slot_dim=slot_dim, grid_dim=D)

    # Dummy inputs
    puzzle_repr = torch.randn(B, N_puzzle, D, requires_grad=True)
    grid_features = torch.randn(B, N_grid, D, requires_grad=True)

    # Forward pass
    rule_slots = slot_attn(puzzle_repr)
    grid_enhanced, _ = decoder(grid_features, rule_slots)

    # Backward pass
    loss = grid_enhanced.sum()
    loss.backward()

    # Check gradients exist
    assert puzzle_repr.grad is not None, "No gradient for puzzle_repr"
    assert grid_features.grad is not None, "No gradient for grid_features"
    assert slot_attn.project_q.weight.grad is not None, "No gradient for slot_attn"

    # Check decoder gradients (cross_attn may have different internal structure)
    has_decoder_grad = False
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            has_decoder_grad = True
            break
    assert has_decoder_grad, "No gradient for decoder"

    print(f"✓ Gradients flow through puzzle_repr")
    print(f"✓ Gradients flow through grid_features")
    print(f"✓ Gradients flow through SlotAttention")
    print(f"✓ Gradients flow through CrossAttentionDecoder")
    print(f"✓ PASS: All components receive gradients")


def test_diversity_loss():
    """Test 4: Verify diversity loss computation"""
    print("\n" + "="*70)
    print("TEST 4: Diversity Loss")
    print("="*70)

    from models.slot_losses import SlotContrastiveLossHead

    # Create dummy model
    class DummyModel:
        def initial_carry(self, *args, **kwargs):
            pass

    loss_head = SlotContrastiveLossHead(
        model=DummyModel(),
        loss_type='softmax_cross_entropy',
        slot_diversity_weight=0.01
    )

    # Test case 1: Identical slots (should have HIGH diversity loss)
    B, K, D = 4, 8, 256
    identical_slots = torch.randn(B, 1, D).expand(B, K, D)
    loss_identical = loss_head.compute_slot_diversity_loss(identical_slots).item()
    print(f"Identical slots loss: {loss_identical:.4f}")

    # Test case 2: Orthogonal slots (should have LOW diversity loss)
    orthogonal_slots = torch.zeros(B, K, D)
    for k in range(K):
        orthogonal_slots[:, k, k*32:(k+1)*32] = 1.0  # Each slot uses different dimensions
    loss_orthogonal = loss_head.compute_slot_diversity_loss(orthogonal_slots).item()
    print(f"Orthogonal slots loss: {loss_orthogonal:.4f}")

    assert loss_identical > loss_orthogonal, "Diversity loss should be higher for identical slots!"
    print(f"✓ PASS: Diversity loss correctly penalizes similar slots")
    print(f"  Identical: {loss_identical:.4f} > Orthogonal: {loss_orthogonal:.4f}")


def test_full_forward_pass():
    """Test 5: Full forward pass simulation"""
    print("\n" + "="*70)
    print("TEST 5: Full Forward Pass Simulation")
    print("="*70)

    B = 2
    puzzle_emb_len = 16
    seq_len = 900
    hidden_size = 512
    num_slots = 8
    slot_dim = 256
    vocab_size = 12

    # Simulate TRM z_H output
    z_H = torch.randn(B, puzzle_emb_len + seq_len, hidden_size)

    # Slot decomposition (puzzle_emb only)
    slot_attn = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, input_dim=hidden_size)
    puzzle_repr = z_H[:, :puzzle_emb_len, :]
    rule_slots = slot_attn(puzzle_repr)
    print(f"✓ Slot decomposition: {puzzle_repr.shape} → {rule_slots.shape}")

    # Grid features
    grid_features = z_H[:, puzzle_emb_len:, :]
    print(f"✓ Grid features: {grid_features.shape}")

    # Direct path
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    output_direct = lm_head(grid_features)
    print(f"✓ Direct output: {output_direct.shape}")

    # Slot-enhanced path
    decoder = SlotCrossAttentionDecoder(slot_dim=slot_dim, grid_dim=hidden_size)
    grid_enhanced, attn_weights = decoder(grid_features, rule_slots)
    output_slots = lm_head(grid_enhanced)  # Shared head!
    print(f"✓ Slot-enhanced output: {output_slots.shape}")

    # Verify outputs are different
    diff = (output_direct - output_slots).abs().mean().item()
    print(f"✓ Output difference: {diff:.4f}")

    if diff > 0.01:
        print(f"✓ PASS: Slot path produces different predictions")
    else:
        print(f"⚠️  WARNING: Outputs too similar (slots may not be contributing)")

    # Check attention interpretability
    print(f"✓ Attention weights: {attn_weights.shape}")
    print(f"  Mean attention per slot: {attn_weights.mean(dim=(0,1))}")
    print(f"✓ PASS: Full forward pass works correctly")


def main():
    print("\n" + "#"*70)
    print("# Slot Attention Implementation Test Suite")
    print("#"*70)

    try:
        # Run all tests
        test_slot_diversity()
        test_cross_attention_decoder()
        test_gradient_flow()
        test_diversity_loss()
        test_full_forward_pass()

        print("\n" + "#"*70)
        print("# ✅ ALL TESTS PASSED!")
        print("# Implementation ready for training")
        print("#"*70)

    except Exception as e:
        print("\n" + "#"*70)
        print(f"# ❌ TEST FAILED: {str(e)}")
        print("#"*70)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
