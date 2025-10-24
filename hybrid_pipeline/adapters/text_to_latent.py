"""
Text to Latent Adapter - Projects LLM hidden states to TRM initial carry

Maps LLaMA hidden state (4096-dim) → TRM z_H, z_L initial states

Now with POSITION-AWARE expansion for grid-level reasoning!
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossAttentionLayer(nn.Module):
    """Single cross-attention block with residual connections."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        query = query + attn_out
        query = query + self.ff(query)
        return query


class CrossAttentionBridge(nn.Module):
    """Maps LLM hidden sequences to TRM slots using cross-attention."""

    def __init__(
        self,
        num_slots: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        llm_hidden: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = llm_hidden.shape[0]
        slots = self.slots.unsqueeze(0).expand(batch_size, -1, -1)

        x = slots
        for layer in self.layers:
            x = layer(x, llm_hidden, key_padding_mask=key_padding_mask)

        return x


class TextToLatentAdapter(nn.Module):
    """
    Projects LLM text representation to TRM latent space.

    Architecture:
        LLM hidden state [batch, llm_dim=4096]
        → Compression → [batch, bottleneck_dim]
        → Position-aware expansion → [batch, seq_len, bottleneck_dim]
        → Projection → z_H [batch, seq_len, trm_dim], z_L [batch, seq_len, trm_dim]

    NEW: Position embeddings allow each grid cell to have distinct features!
    """

    def __init__(
        self,
        llm_hidden_size: int,
        trm_hidden_size: int,
        trm_seq_len: int,
        puzzle_emb_len: int = 0,
        use_position_embeddings: bool = True,
        use_cross_attention: bool = False,
        cross_attention_layers: int = 2,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.0,
    ):
        """
        Args:
            llm_hidden_size: LLaMA hidden dimension (4096 for LLaMA-8B)
            trm_hidden_size: TRM hidden dimension (e.g., 512)
            trm_seq_len: TRM sequence length (900 for ARC 30x30)
            puzzle_emb_len: Additional puzzle embedding length
            use_position_embeddings: Enable position-aware expansion (recommended)
        """
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.trm_hidden_size = trm_hidden_size
        self.trm_seq_len = trm_seq_len + puzzle_emb_len
        self.puzzle_emb_len = puzzle_emb_len
        self.use_position_embeddings = use_position_embeddings
        self.use_cross_attention = use_cross_attention

        # Bottleneck architecture to reduce parameters
        # 1.9B params → 6M params
        self.bottleneck_dim = 1024

        # Stage 1: Compression (4096 → 1024)
        self.compress = nn.Sequential(
            nn.Linear(llm_hidden_size, self.bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(self.bottleneck_dim)
        )

        if self.use_cross_attention:
            self.sequence_proj = nn.Linear(llm_hidden_size, self.bottleneck_dim)
            self.cross_attention = CrossAttentionBridge(
                num_slots=self.trm_seq_len,
                hidden_dim=self.bottleneck_dim,
                num_layers=cross_attention_layers,
                num_heads=cross_attention_heads,
                dropout=cross_attention_dropout,
            )
        else:
            self.sequence_proj = None

        # NEW: Position embeddings for grid-aware reasoning
        if use_position_embeddings:
            # Learnable position embeddings for each grid position
            self.position_embeddings = nn.Parameter(
                torch.randn(self.trm_seq_len, self.bottleneck_dim) * 0.02
            )
        else:
            self.position_embeddings = None

        # Stage 2: To TRM latent (1024 → 1024 for z_H + z_L)
        self.to_trm = nn.Linear(self.bottleneck_dim, 2 * trm_hidden_size)

        # Initialize to small values for stable training
        modules_to_init = [self.compress, self.to_trm]
        if self.sequence_proj is not None:
            modules_to_init.append(self.sequence_proj)

        for module in modules_to_init:
            for m in module.modules() if hasattr(module, 'modules') else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        llm_hidden_state: torch.Tensor,
        llm_hidden_sequence: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project LLM hidden state to TRM initial carry.

        Args:
            llm_hidden_state: [batch, llm_dim] final hidden state from LLM

        Returns:
            z_H: [batch, seq_len, trm_dim] high-level carry state
            z_L: [batch, seq_len, trm_dim] low-level carry state
        """
        batch_size = llm_hidden_state.shape[0]

        # Stage 1: Compress global LLM hidden state
        compressed = self.compress(llm_hidden_state)

        if self.use_cross_attention and llm_hidden_sequence is not None:
            seq_tokens = self.sequence_proj(llm_hidden_sequence)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.to(torch.bool)

            seq_features = self.cross_attention(seq_tokens, key_padding_mask=key_padding_mask)
            seq_features = seq_features + compressed.unsqueeze(1)
            if self.position_embeddings is not None:
                seq_features = seq_features + self.position_embeddings.unsqueeze(0)
        else:
            if self.use_position_embeddings:
                seq_features = compressed.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
            else:
                seq_features = compressed.unsqueeze(1).expand(-1, self.trm_seq_len, -1)

        # Stage 3: Project to TRM dimensions
        trm_latent = self.to_trm(seq_features)

        # Split into z_H and z_L
        z_H, z_L = trm_latent.chunk(2, dim=-1)

        return z_H, z_L
