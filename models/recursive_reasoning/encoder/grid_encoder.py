"""
GridEncoder: Unified encoder for all grid representations.

This module provides a single encoder that can process any ARC grid
(input, output, few-shot examples) into either:
- Global representation [B, D] for goal computation
- Token-level representation [B, L, D] for TRM reasoning

Design Principles:
- Standalone module (no dependency on TRM internals)
- Can optionally share embed_tokens with TRM for consistency
- Configurable output type via return_type parameter

Output Characteristics:
=======================
- Token output (return_type="tokens"): Normalized by LayerNorm, std ~ 1.0
  * NO CLS token added (CLS is handled externally in wrappers.py)
  * Suitable for direct concatenation with prefix tokens

- Global output (return_type="global"): Normalized by LayerNorm
  * Uses CLS token or mean pooling for aggregation
  * Suitable for goal computation

Position Encoding Strategy:
===========================
GridEncoder's pos_embed and TRM's RoPE serve DIFFERENT semantic purposes:

1. GridEncoder pos_embed (position_encoding_mode):
   - "additive": Add learned positions to token embeddings (default)
     * Captures 2D grid spatial structure (row/column awareness)
     * Applied BEFORE transformer layers process tokens
   - "none": No position encoding (for use with TRM RoPE only)
     * Grid tokens have no spatial awareness from GridEncoder
     * Relies entirely on TRM's RoPE for position information
   - "offset": Add positions offset by prefix_length
     * Aligns GridEncoder positions with TRM sequence positions
     * pos_embed[offset:offset+L] used instead of pos_embed[0:L]

2. TRM RoPE (controlled by EnhancedTRM):
   - Applied in attention computation to Q,K
   - Captures relative sequence position for attention patterns
   - Applied to ALL tokens (prefix + grid) at positions 0 to N-1

Recommended Configurations:
- "additive" + RoPE: Hybrid approach (both spatial and sequence info)
- "none" + RoPE: Pure sequence-based (like standard transformer)
- "additive" + no RoPE: Pure spatial-based (grid structure only)

The hybrid approach ("additive" + RoPE) is the default and recommended
setting as it allows the model to leverage both:
- Grid spatial structure (from pos_embed)
- Sequence relationships (from RoPE)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, Union
from dataclasses import dataclass


@dataclass
class GridEncoderConfig:
    """
    Configuration for GridEncoder.

    Position Encoding Options:
    - position_encoding_mode: Controls how/if GridEncoder adds position info
        * "additive": Add pos_embed[0:L] to grid tokens (default, grid-aware)
        * "none": No position encoding (rely on TRM's RoPE only)
        * "offset": Add pos_embed[offset:offset+L] to align with TRM positions
    - position_offset: Offset for "offset" mode (typically puzzle_emb_len=16)

    The combination of GridEncoder position encoding and TRM's RoPE determines
    the overall position encoding strategy. See module docstring for details.
    """
    vocab_size: int = 11  # 0-10 for ARC colors
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 4
    max_grid_size: int = 900  # 30x30
    dropout: float = 0.0
    use_cls_token: bool = True  # Only used for "global" return type

    # Position encoding configuration
    position_encoding_mode: str = "additive"  # "additive", "none", or "offset"
    position_offset: int = 16  # Offset for "offset" mode (puzzle_emb_len)


class GridEncoder(nn.Module):
    """
    Unified encoder for ARC grids.

    Can encode any grid into:
    - "global": Single vector [B, D] representing the entire grid
    - "tokens": Per-token representations [B, L, D] for detailed reasoning
    - "both": Tuple of (global, tokens)

    Output Properties:
    ==================
    - All outputs go through LayerNorm (normalized, std ~ 1.0)
    - Token outputs do NOT include CLS token (handled externally)

    Position Encoding Modes:
    ========================
    - "additive": pos_embed[0:L] added to tokens (grid spatial awareness)
    - "none": No position encoding (for TRM RoPE-only strategy)
    - "offset": pos_embed[offset:offset+L] added (aligned with TRM positions)

    See module docstring for detailed explanation of position encoding strategy.

    Args:
        config: GridEncoderConfig with model hyperparameters
        embed_tokens: Optional pre-trained token embedding to share with TRM

    Example:
        >>> # Default: additive position encoding (grid-aware)
        >>> encoder = GridEncoder(GridEncoderConfig())
        >>> grid = torch.randint(0, 11, (4, 900))  # [B, L]
        >>> token_repr = encoder(grid, return_type="tokens")   # [B, L, D]
        >>>
        >>> # No position encoding (for RoPE-only strategy)
        >>> config = GridEncoderConfig(position_encoding_mode="none")
        >>> encoder = GridEncoder(config)
    """

    def __init__(
        self,
        config: GridEncoderConfig,
        embed_tokens: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # Validate position encoding mode
        valid_modes = ("additive", "none", "offset")
        if config.position_encoding_mode not in valid_modes:
            raise ValueError(
                f"position_encoding_mode must be one of {valid_modes}, "
                f"got '{config.position_encoding_mode}'"
            )

        # Token embedding: use provided or create new
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
            self._shared_embed = True
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size
            )
            self._shared_embed = False
            self._init_embedding()

        # Position embedding for grid structure awareness
        # Only create if position encoding is enabled
        if config.position_encoding_mode != "none":
            # For "offset" mode, we need extra positions to handle the offset
            max_pos = config.max_grid_size
            if config.position_encoding_mode == "offset":
                max_pos = config.max_grid_size + config.position_offset
            self.pos_embed = nn.Parameter(
                torch.randn(max_pos, config.hidden_size) * 0.02
            )
        else:
            # Register as None for "none" mode
            self.register_parameter("pos_embed", None)

        # CLS token for global pooling (only used with return_type="global")
        if config.use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, config.hidden_size) * 0.02
            )
        else:
            self.cls_token = None

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output normalization - ensures consistent output scale
        self.norm = nn.LayerNorm(config.hidden_size)

    def _init_embedding(self):
        """Initialize embedding with truncated normal."""
        nn.init.trunc_normal_(self.embed_tokens.weight, std=0.02)

    def _embed_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Embed grid tokens with optional position information.

        Position encoding depends on config.position_encoding_mode:
        - "additive": Add pos_embed[0:L] (grid spatial awareness)
        - "none": No position encoding (for TRM RoPE-only strategy)
        - "offset": Add pos_embed[offset:offset+L] (aligned with TRM positions)

        Args:
            grid: [B, L] or [B*N, L] token IDs

        Returns:
            [B, L, D] embedded tokens (with or without position encoding)
        """
        # Token embedding
        if hasattr(self.embed_tokens, 'forward'):
            # Handle both nn.Embedding and custom embedding modules
            x = self.embed_tokens(grid.long())
        else:
            x = F.embedding(grid.long(), self.embed_tokens.weight)

        seq_len = x.size(1)

        # Apply position encoding based on mode
        # Cast pos_embed to the same dtype as x to avoid dtype mismatch
        # (pos_embed is float32, but x may be bfloat16 in mixed precision training)
        mode = self.config.position_encoding_mode
        if mode == "additive":
            # Standard additive: positions 0 to L-1
            # Captures grid spatial structure (row/column positions)
            x = x + self.pos_embed[:seq_len].to(x.dtype)
        elif mode == "offset":
            # Offset additive: positions offset to offset+L-1
            # Aligns with TRM sequence positions (after prefix)
            offset = self.config.position_offset
            x = x + self.pos_embed[offset:offset + seq_len].to(x.dtype)
        # mode == "none": no position encoding added

        return x

    def forward(
        self,
        grid: torch.Tensor,
        return_type: Literal["global", "tokens", "both"] = "global",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode grid into representations.

        Args:
            grid: Token IDs of shape [B, L] or [B, N, L] for batched few-shot
            return_type:
                - "global": Return [B, D] global representation
                - "tokens": Return [B, L, D] per-token representations
                           NOTE: Does NOT include CLS token
                - "both": Return tuple (global, tokens)

        Returns:
            Depending on return_type:
                - "global": [B, D] or [B, N, D] for batched input
                - "tokens": [B, L, D] or [B, N, L, D] for batched input
                - "both": Tuple of above

        Output Properties:
        - All outputs are normalized (LayerNorm applied)
        - Token outputs have position encoding built-in
        - Token outputs do NOT include CLS (handled by caller)
        """
        # Handle batched few-shot input [B, N, L]
        original_shape = grid.shape
        is_batched = grid.dim() == 3

        if is_batched:
            B, N, L = grid.shape
            grid = grid.view(B * N, L)
        else:
            B, L = grid.shape
            N = None

        # Embed tokens with position encoding
        x = self._embed_grid(grid)  # [B, L, D]

        # Different processing based on return type
        if return_type == "tokens":
            # For tokens: process WITHOUT CLS, return normalized tokens
            x = self.transformer(x)
            x = self.norm(x)
            token_repr = x  # [B, L, D]

            # Restore batch shape if needed
            if is_batched:
                token_repr = token_repr.view(B, N, L, -1)  # [B, N, L, D]

            return token_repr

        elif return_type == "global":
            # For global: use CLS token for aggregation
            if self.cls_token is not None:
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)  # [B, 1+L, D]
                x = self.transformer(x)
                x = self.norm(x)
                global_repr = x[:, 0]  # [B, D] - CLS position
            else:
                # Mean pooling as fallback
                x = self.transformer(x)
                x = self.norm(x)
                global_repr = x.mean(dim=1)  # [B, D]

            # Restore batch shape if needed
            if is_batched:
                global_repr = global_repr.view(B, N, -1)  # [B, N, D]

            return global_repr

        else:  # "both"
            # For both: need CLS for global, but tokens without CLS
            if self.cls_token is not None:
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x_with_cls = torch.cat([cls, x], dim=1)  # [B, 1+L, D]
                x_with_cls = self.transformer(x_with_cls)
                x_with_cls = self.norm(x_with_cls)
                global_repr = x_with_cls[:, 0]  # [B, D]
                token_repr = x_with_cls[:, 1:]  # [B, L, D] - exclude CLS
            else:
                x = self.transformer(x)
                x = self.norm(x)
                global_repr = x.mean(dim=1)  # [B, D]
                token_repr = x  # [B, L, D]

            # Restore batch shape if needed
            if is_batched:
                global_repr = global_repr.view(B, N, -1)  # [B, N, D]
                token_repr = token_repr.view(B, N, L, -1)  # [B, N, L, D]

            return global_repr, token_repr

    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.config.hidden_size

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# Convenience function for quick instantiation
def create_grid_encoder(
    hidden_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 4,
    embed_tokens: Optional[nn.Module] = None,
    position_encoding_mode: str = "additive",
    position_offset: int = 16,
    **kwargs
) -> GridEncoder:
    """
    Create a GridEncoder with common defaults.

    Args:
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        embed_tokens: Optional shared embedding module
        position_encoding_mode: Position encoding strategy
            - "additive": Add pos_embed[0:L] (default, grid spatial awareness)
            - "none": No position encoding (for TRM RoPE-only strategy)
            - "offset": Add pos_embed[offset:offset+L] (aligned with TRM positions)
        position_offset: Offset for "offset" mode (default: 16 for puzzle_emb_len)
        **kwargs: Additional config parameters

    Returns:
        Configured GridEncoder instance

    Example:
        >>> # Default: hybrid (GridEncoder pos_embed + TRM RoPE)
        >>> encoder = create_grid_encoder()
        >>>
        >>> # RoPE only (no GridEncoder position encoding)
        >>> encoder = create_grid_encoder(position_encoding_mode="none")
        >>>
        >>> # Aligned positions (GridEncoder + TRM positions match)
        >>> encoder = create_grid_encoder(position_encoding_mode="offset")
    """
    config = GridEncoderConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        position_encoding_mode=position_encoding_mode,
        position_offset=position_offset,
        **kwargs
    )
    return GridEncoder(config, embed_tokens=embed_tokens)
