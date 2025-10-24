"""
Latent to Text Adapter - Projects TRM carry state to LLM space

Separated from text_to_latent.py for modularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to preserve important grid cell information.

    Instead of simple average pooling (which loses 99.9% information),
    this learns which grid positions are important and weights them accordingly.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, seq_len, hidden] tensor

        Returns:
            pooled: [batch, hidden] weighted sum based on attention
        """
        # Compute attention scores: [batch, seq_len, 1]
        attn_scores = self.attention(z)

        # Softmax over sequence: [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum: [batch, hidden]
        pooled = (z * attn_weights).sum(dim=1)

        return pooled


class LatentToTextAdapter(nn.Module):
    """
    Projects TRM carry state back to LLM latent space.

    This allows LLM to "read" TRM's reasoning state and generate
    feedback for re-injection as latent prefix.

    Uses attention-based pooling to preserve important grid information.
    """

    def __init__(
        self,
        trm_hidden_size: int,
        trm_seq_len: int,
        llm_hidden_size: int,
        puzzle_emb_len: int = 0,
        use_attention_pooling: bool = True
    ):
        """
        Args:
            trm_hidden_size: TRM hidden dimension
            trm_seq_len: TRM sequence length
            llm_hidden_size: LLaMA hidden dimension (4096)
            puzzle_emb_len: Additional puzzle embedding length
            use_attention_pooling: Use attention pooling (default) vs average pooling
        """
        super().__init__()

        self.trm_hidden_size = trm_hidden_size
        self.trm_seq_len = trm_seq_len + puzzle_emb_len
        self.llm_hidden_size = llm_hidden_size
        self.use_attention_pooling = use_attention_pooling

        # Pooling + projection architecture to reduce parameters
        # Old: 213B params → 10M params (average pooling)
        # New: ~10.5M params (attention pooling, +500K for attention networks)

        # Stage 1: Sequence pooling
        if use_attention_pooling:
            # NEW: Attention-based pooling (learns importance weights)
            self.pool_H = AttentionPooling(trm_hidden_size)
            self.pool_L = AttentionPooling(trm_hidden_size)
        else:
            # OLD: Simple average pooling (loses spatial information)
            self.pool = nn.AdaptiveAvgPool1d(1)

        # Stage 2: Projection to LLM space
        intermediate_dim = 2048
        self.projection = nn.Sequential(
            nn.Linear(2 * trm_hidden_size, intermediate_dim),  # 1024 → 2048
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, llm_hidden_size)  # 2048 → 4096
        )

        # Initialize to small values for stable training
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor
    ) -> torch.Tensor:
        """
        Project TRM carry state to LLM latent space.

        Args:
            z_H: [batch, seq_len, trm_dim] high-level carry
            z_L: [batch, seq_len, trm_dim] low-level carry

        Returns:
            latent_prefix: [batch, llm_dim] latent for LLM prefix
        """
        batch_size = z_H.shape[0]

        # Stage 1: Pool sequences
        if self.use_attention_pooling:
            # NEW: Attention-based pooling (preserves important grid cells)
            # [batch, seq_len, 512] → [batch, 512]
            z_H_pooled = self.pool_H(z_H)
            z_L_pooled = self.pool_L(z_L)
        else:
            # OLD: Average pooling (loses 99.9% information)
            # [batch, seq_len, 512] → [batch, 512, seq_len] → [batch, 512, 1] → [batch, 512]
            z_H_pooled = self.pool(z_H.transpose(1, 2)).squeeze(-1)
            z_L_pooled = self.pool(z_L.transpose(1, 2)).squeeze(-1)

        # Stage 2: Concatenate pooled features
        # [batch, 512] + [batch, 512] → [batch, 1024]
        z_combined = torch.cat([z_H_pooled, z_L_pooled], dim=-1)

        # Stage 3: Project to LLM space
        # [batch, 1024] → [batch, 4096]
        latent_prefix = self.projection(z_combined)

        return latent_prefix
