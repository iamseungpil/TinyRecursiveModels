"""
Slot Attention module for compositional representation learning.

Based on "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
https://arxiv.org/abs/2006.15055
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlotAttention(nn.Module):
    """
    Slot Attention module that decomposes input representations into slots.

    This module uses iterative attention to bind input features to a fixed number
    of slots, enabling compositional and object-centric representations.

    Args:
        num_slots: Number of slots (e.g., 8 for decomposing into 8 components)
        slot_dim: Dimension of each slot
        input_dim: Dimension of input features
        num_iterations: Number of iterative refinement steps
        mlp_hidden_dim: Hidden dimension for slot update MLP
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        mlp_hidden_dim: Optional[int] = None,
        epsilon: float = 1e-8,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.dtype = dtype

        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(slot_dim, input_dim)

        # Slot initialization parameters (learnable) - use target dtype
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim, dtype=dtype))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim, dtype=dtype))

        # Layer normalization
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Linear projections for attention
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )

        # Scale factor for attention
        self.scale = slot_dim ** -0.5

        # Convert all modules to target dtype
        self.to(dtype=self.dtype)

    @torch.compiler.disable()
    def forward(self, inputs: torch.Tensor, num_slots: Optional[int] = None) -> torch.Tensor:
        """
        Apply slot attention to decompose inputs into slots.

        Args:
            inputs: Input features [batch_size, num_inputs, input_dim]
            num_slots: Optional override for number of slots (for curriculum learning)

        Returns:
            slots: Decomposed slot representations [batch_size, num_slots, slot_dim]
        """
        B, N, D_in = inputs.shape

        if num_slots is None:
            num_slots = self.num_slots

        # Initialize slots using learned mean and variance
        mu = self.slots_mu.expand(B, num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Normalize and project inputs
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative attention refinement
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Normalize slots before attention
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)  # [B, num_slots, slot_dim]

            # Compute attention weights: softmax over slots
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k) * self.scale  # [B, num_slots, N]
            attn = F.softmax(attn_logits, dim=1)  # Normalize over slots

            # Weighted mean of values
            attn_sum = attn.sum(dim=-1, keepdim=True) + self.epsilon
            attn_wts = attn / attn_sum  # [B, num_slots, N]
            updates = torch.einsum('bsn,bnd->bsd', attn_wts, v)  # [B, num_slots, slot_dim]

            # Update slots using GRU
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, num_slots, self.slot_dim)

            # Refine with MLP
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SlotDecoder(nn.Module):
    """
    Decoder that reconstructs features from slots.

    DEPRECATED: This decoder uses mean pooling which destroys slot information.
    Use SlotCrossAttentionDecoder instead for position-specific reconstruction.

    Args:
        slot_dim: Dimension of input slots
        output_dim: Dimension of output features
        hidden_dim: Hidden dimension
        broadcast_size: Spatial size to broadcast slots to
    """

    def __init__(
        self,
        slot_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        broadcast_size: Optional[int] = None
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.output_dim = output_dim
        self.broadcast_size = broadcast_size

        if hidden_dim is None:
            hidden_dim = max(slot_dim, output_dim)

        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Decode slots to output features.

        Args:
            slots: [batch_size, num_slots, slot_dim]

        Returns:
            outputs: [batch_size, broadcast_size, output_dim] if broadcast_size is set,
                    otherwise [batch_size, num_slots, output_dim]
        """
        B, num_slots, D = slots.shape

        # Decode each slot
        slot_features = self.decoder(slots)  # [B, num_slots, output_dim]

        if self.broadcast_size is None:
            # Return per-slot features
            return slot_features
        else:
            # Aggregate slots and broadcast to spatial dimensions
            # Mean pooling over slots
            aggregated = slot_features.mean(dim=1, keepdim=True)  # [B, 1, output_dim]
            # Broadcast
            outputs = aggregated.expand(B, self.broadcast_size, self.output_dim)
            return outputs


class SlotCrossAttentionDecoder(nn.Module):
    """
    Cross-attention decoder where grid positions attend to rule slots.

    This decoder allows each spatial position to selectively attend to relevant
    compositional rule slots, enabling position-specific reconstruction while
    maintaining slot compositionality.

    Key difference from SlotDecoder:
    - SlotDecoder: Mean pools slots → all positions get same features (BUGGY)
    - SlotCrossAttentionDecoder: Each position attends to slots → position-specific features

    Args:
        slot_dim: Dimension of slot representations
        grid_dim: Dimension of grid features (hidden_size in TRM)
        num_heads: Number of attention heads
        dropout: Dropout probability

    Example:
        >>> decoder = SlotCrossAttentionDecoder(slot_dim=256, grid_dim=512, num_heads=8)
        >>> grid_features = torch.randn(4, 900, 512)  # [B, 900 positions, 512]
        >>> rule_slots = torch.randn(4, 8, 256)       # [B, 8 slots, 256]
        >>> enhanced, attn = decoder(grid_features, rule_slots)
        >>> enhanced.shape  # [4, 900, 512] - each position enhanced differently
        >>> attn.shape      # [4, 900, 8] - which rule each position uses
    """

    def __init__(
        self,
        slot_dim: int,
        grid_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.grid_dim = grid_dim
        self.num_heads = num_heads
        self.dtype = dtype

        # Cross-attention: grid queries rule slots
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=grid_dim,
            num_heads=num_heads,
            kdim=slot_dim,
            vdim=slot_dim,
            dropout=dropout,
            batch_first=False  # PyTorch expects [seq, batch, dim]
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(grid_dim)
        self.norm2 = nn.LayerNorm(grid_dim)

        # Feed-forward network (using same pattern as TRM)
        from models.layers import SwiGLU
        self.mlp = SwiGLU(hidden_size=grid_dim, expansion=2.0)

        # Convert all modules to target dtype
        self.to(dtype=self.dtype)

    def forward(
        self,
        grid_features: torch.Tensor,
        rule_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention from grid positions to rule slots.

        Args:
            grid_features: [batch_size, num_positions, grid_dim]
                          Spatial grid features (e.g., 900 positions)
            rule_slots: [batch_size, num_slots, slot_dim]
                       Compositional rule slots (e.g., 8 slots)

        Returns:
            grid_enhanced: [batch_size, num_positions, grid_dim]
                         Grid features enhanced with rule information
            attn_weights: [batch_size, num_positions, num_slots]
                        Attention weights showing which rule each position uses
        """
        B, N, D_grid = grid_features.shape
        _, K, D_slot = rule_slots.shape

        # PyTorch MultiheadAttention expects [seq, batch, dim]
        # Query: grid positions want to attend to rules
        # Key/Value: rule slots provide information
        query = grid_features.transpose(0, 1)  # [N, B, grid_dim]
        key = rule_slots.transpose(0, 1)       # [K, B, slot_dim]
        value = rule_slots.transpose(0, 1)     # [K, B, slot_dim]

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=True  # Average over heads for interpretability
        )
        attn_out = attn_out.transpose(0, 1)  # [B, N, grid_dim]

        # Residual connection + LayerNorm
        grid_features = self.norm1(grid_features + attn_out)

        # Feed-forward + Residual
        grid_enhanced = self.norm2(grid_features + self.mlp(grid_features))

        # attn_weights: [B, N, K] - which rule slot each position attends to
        return grid_enhanced, attn_weights


def test_slot_attention():
    """Test slot attention module."""
    print("Testing Slot Attention...")

    batch_size = 4
    num_inputs = 100
    input_dim = 512
    num_slots = 8
    slot_dim = 256

    # Create module
    slot_attn = SlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        input_dim=input_dim,
        num_iterations=3
    )

    # Test input
    inputs = torch.randn(batch_size, num_inputs, input_dim)

    # Forward pass
    slots = slot_attn(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Slots shape: {slots.shape}")
    assert slots.shape == (batch_size, num_slots, slot_dim)

    # Test decoder
    decoder = SlotDecoder(
        slot_dim=slot_dim,
        output_dim=input_dim,
        broadcast_size=num_inputs
    )

    reconstructed = decoder(slots)
    print(f"Reconstructed shape: {reconstructed.shape}")
    assert reconstructed.shape == (batch_size, num_inputs, input_dim)

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_slot_attention()
