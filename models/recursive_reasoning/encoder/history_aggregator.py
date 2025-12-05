"""
ZLHistoryAggregator: Allow z_L to reference past z_L states.

Design rationale:
- z_H is already recurrent (accumulates past info), so z_H history is redundant
- z_L's detail reasoning info is lost when compressed into z_H
- z_L history gives direct access to past detailed reasoning

Key features:
- Cross-attention: current z_L attends to past K z_L states
- Identity initialization: gate initialized to large negative value so sigmoid(gate) ~ 0
- Configurable window size K

Usage:
    >>> config = ZLHistoryAggregatorConfig(hidden_size=512, window_size=3)
    >>> aggregator = ZLHistoryAggregator(config)
    >>>
    >>> # During inference, accumulate z_L history
    >>> z_L_history = []
    >>> for h_step in range(H_cycles):
    >>>     # ... compute z_L ...
    >>>     history_context = aggregator(z_L, z_L_history)
    >>>     # Add history_context to injection
    >>>     z_L_history.append(z_L.detach())
    >>>     z_L_history = z_L_history[-config.window_size:]  # Keep window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ZLHistoryAggregatorConfig:
    """
    Configuration for ZLHistoryAggregator.

    Attributes:
        hidden_size: Hidden dimension (must match z_L dimension)
        num_heads: Number of attention heads for cross-attention
        window_size: Maximum number of past z_L states to keep (K)
        dropout: Dropout rate for attention
        gate_init: Initial value for gate parameter (large negative for identity init)
    """
    hidden_size: int = 512
    num_heads: int = 8
    window_size: int = 3
    dropout: float = 0.0
    gate_init: float = -10.0  # sigmoid(-10) ~ 0.00005, effectively 0


class ZLHistoryAggregator(nn.Module):
    """
    Aggregates information from past z_L states via cross-attention.

    The aggregator allows the current z_L to attend to previous z_L states,
    providing direct access to past detailed reasoning information that would
    otherwise be lost when compressed into z_H.

    Key design decisions:
    1. Cross-attention (not self-attention): Current z_L queries past states
    2. Identity initialization: Gate starts near 0, so output starts as zeros
    3. Additive integration: Output is ADDED to injection, not replacing anything

    Args:
        config: ZLHistoryAggregatorConfig with hyperparameters

    Example:
        >>> aggregator = ZLHistoryAggregator(ZLHistoryAggregatorConfig())
        >>> z_L = torch.randn(2, 100, 512)  # [B, L, D]
        >>> z_L_history = [torch.randn(2, 100, 512) for _ in range(3)]
        >>> history_context = aggregator(z_L, z_L_history)
        >>> # Add to injection: z_H + input_embeddings + history_context
    """

    def __init__(self, config: ZLHistoryAggregatorConfig):
        super().__init__()
        self.config = config

        # Cross-attention: z_L (query) attends to z_L_history (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Layer norm for query and key/value
        self.query_norm = nn.LayerNorm(config.hidden_size)
        self.kv_norm = nn.LayerNorm(config.hidden_size)

        # Output projection with layer norm
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)

        # Gate for identity initialization
        # sigmoid(gate_init) ~ 0, so initial output is effectively 0
        self.gate = nn.Parameter(torch.tensor(config.gate_init))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        # Small initialization for output projection
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_L: torch.Tensor,
        z_L_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute history context from past z_L states.

        Args:
            z_L: Current low-level state [B, L, D]
            z_L_history: List of past z_L states, each [B, L, D]
                         Maximum K items (older items are discarded)

        Returns:
            history_context: [B, L, D] context to ADD to injection
                            Returns zeros if no history available
        """
        # Handle empty history case
        if not z_L_history:
            return torch.zeros_like(z_L)

        B, L, D = z_L.shape
        K = len(z_L_history)

        # Limit to window size
        if K > self.config.window_size:
            z_L_history = z_L_history[-self.config.window_size:]
            K = self.config.window_size

        # Stack history: [B, K, L, D]
        history_stack = torch.stack(z_L_history, dim=1)

        # Flatten for attention: [B, K*L, D]
        history_flat = history_stack.view(B, K * L, D)

        # Normalize query and key/value
        query = self.query_norm(z_L)  # [B, L, D]
        kv = self.kv_norm(history_flat)  # [B, K*L, D]

        # Cross-attention: current z_L attends to past z_L states
        attn_output, _ = self.cross_attention(
            query=query,
            key=kv,
            value=kv,
        )  # [B, L, D]

        # Output projection
        output = self.output_proj(attn_output)
        output = self.output_norm(output)

        # Apply gate for identity initialization
        # At init, sigmoid(gate) ~ 0, so output ~ 0
        gated_output = torch.sigmoid(self.gate) * output

        return gated_output

    def get_gate_value(self) -> float:
        """Return current gate value (for monitoring training)."""
        return torch.sigmoid(self.gate).item()

    def reset_gate(self, value: float = -10.0):
        """Reset gate to initial value (useful for transfer learning)."""
        self.gate.data.fill_(value)


def create_history_aggregator(
    hidden_size: int = 512,
    num_heads: int = 8,
    window_size: int = 3,
    **kwargs
) -> ZLHistoryAggregator:
    """
    Convenience function to create ZLHistoryAggregator.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        window_size: History window size K
        **kwargs: Additional config parameters

    Returns:
        Configured ZLHistoryAggregator instance
    """
    config = ZLHistoryAggregatorConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window_size,
        **kwargs
    )
    return ZLHistoryAggregator(config)
