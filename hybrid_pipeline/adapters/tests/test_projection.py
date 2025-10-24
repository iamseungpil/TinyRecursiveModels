"""
Unit tests for adapter projection layers
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from adapters.text_to_latent import TextToLatentAdapter
from adapters.latent_to_text import LatentToTextAdapter


def test_text_to_latent_adapter():
    """Test TextToLatentAdapter forward pass."""
    print("🧪 Testing TextToLatentAdapter...")

    batch_size = 2
    llm_hidden_size = 4096
    trm_hidden_size = 512
    trm_seq_len = 900

    adapter = TextToLatentAdapter(
        llm_hidden_size=llm_hidden_size,
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len
    )

    llm_hidden = torch.randn(batch_size, llm_hidden_size)
    z_H, z_L = adapter(llm_hidden)

    assert z_H.shape == (batch_size, trm_seq_len, trm_hidden_size)
    assert z_L.shape == (batch_size, trm_seq_len, trm_hidden_size)
    assert z_H.dtype == torch.float32
    assert z_L.dtype == torch.float32
    assert not torch.allclose(z_H, torch.zeros_like(z_H), atol=1e-6)

    print(f"  ✅ z_H shape: {z_H.shape}, range: [{z_H.min():.4f}, {z_H.max():.4f}]")


def test_text_to_latent_adapter_cross_attention():
    """Test TextToLatentAdapter with cross-attention bridge."""
    print("\n🧪 Testing TextToLatentAdapter (Cross-Attention)...")

    batch_size = 2
    llm_hidden_size = 4096
    trm_hidden_size = 512
    trm_seq_len = 900
    llm_seq_len = 128

    adapter = TextToLatentAdapter(
        llm_hidden_size=llm_hidden_size,
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len,
        use_cross_attention=True,
        cross_attention_layers=1,
        cross_attention_heads=4
    )

    llm_hidden = torch.randn(batch_size, llm_hidden_size)
    llm_sequence = torch.randn(batch_size, llm_seq_len, llm_hidden_size)
    attention_mask = torch.ones(batch_size, llm_seq_len, dtype=torch.bool)

    z_H, z_L = adapter(llm_hidden, llm_hidden_sequence=llm_sequence, attention_mask=attention_mask)

    assert z_H.shape == (batch_size, trm_seq_len, trm_hidden_size)
    assert z_L.shape == (batch_size, trm_seq_len, trm_hidden_size)
    assert torch.isfinite(z_H).all()
    assert torch.isfinite(z_L).all()

    print(f"  ✅ Cross-attention output shapes OK ({z_H.shape})")


def test_latent_to_text_adapter():
    """Test LatentToTextAdapter forward pass."""
    print("\n🧪 Testing LatentToTextAdapter...")

    batch_size = 2
    trm_hidden_size = 512
    trm_seq_len = 900
    llm_hidden_size = 4096

    adapter = LatentToTextAdapter(
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len,
        llm_hidden_size=llm_hidden_size
    )

    z_H = torch.randn(batch_size, trm_seq_len, trm_hidden_size)
    z_L = torch.randn(batch_size, trm_seq_len, trm_hidden_size)

    latent_prefix = adapter(z_H, z_L)

    assert latent_prefix.shape == (batch_size, llm_hidden_size)
    assert latent_prefix.dtype == torch.float32
    assert not torch.allclose(latent_prefix, torch.zeros_like(latent_prefix), atol=1e-6)

    print(f"  ✅ Latent prefix shape: {latent_prefix.shape}, range: [{latent_prefix.min():.4f}, {latent_prefix.max():.4f}]")


def test_adapter_roundtrip():
    """Test text→latent→text roundtrip."""
    print("\n🧪 Testing adapter roundtrip...")

    batch_size = 2
    llm_hidden_size = 4096
    trm_hidden_size = 512
    trm_seq_len = 900

    text_to_latent = TextToLatentAdapter(
        llm_hidden_size=llm_hidden_size,
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len
    )

    latent_to_text = LatentToTextAdapter(
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len,
        llm_hidden_size=llm_hidden_size
    )

    llm_hidden = torch.randn(batch_size, llm_hidden_size)
    z_H, z_L = text_to_latent(llm_hidden)
    reconstructed = latent_to_text(z_H, z_L)

    assert reconstructed.shape == llm_hidden.shape

    correlation = torch.corrcoef(torch.stack([
        llm_hidden.flatten(),
        reconstructed.flatten()
    ]))[0, 1]

    print(f"  ✅ Roundtrip correlation: {correlation:.4f}")


def test_gradient_flow():
    """Test gradient flow through adapters."""
    print("\n🧪 Testing gradient flow...")

    batch_size = 2
    llm_hidden_size = 4096
    trm_hidden_size = 512
    trm_seq_len = 900

    text_to_latent = TextToLatentAdapter(
        llm_hidden_size=llm_hidden_size,
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=trm_seq_len
    )

    llm_hidden = torch.randn(batch_size, llm_hidden_size, requires_grad=True)
    z_H, z_L = text_to_latent(llm_hidden)
    loss = (z_H.sum() + z_L.sum())
    loss.backward()

    assert llm_hidden.grad is not None
    assert not torch.allclose(llm_hidden.grad, torch.zeros_like(llm_hidden.grad))

    for name, param in text_to_latent.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    print(f"  ✅ Gradients propagated correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Adapter Projection Tests")
    print("=" * 60)

    test_text_to_latent_adapter()
    test_latent_to_text_adapter()
    test_text_to_latent_adapter_cross_attention()
    test_adapter_roundtrip()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
