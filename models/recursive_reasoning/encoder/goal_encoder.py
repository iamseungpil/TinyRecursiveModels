"""
GoalEncoder: Extract goal representation from few-shot examples.

This module computes a goal representation that captures the transformation
pattern from few-shot input-output pairs. The goal can be used to condition
TRM's reasoning process.

Supports multiple modes:
- Input-only: Goal from few-shot inputs (replaces puzzle_emb)
- Goal conditioning: Goal from (output - input) transformation

Design Principles:
- Uses GridEncoder as backbone (composition, not inheritance)
- Outputs puzzle_emb-compatible shape [B, num_goal_tokens, hidden_size]
- Supports staged testing via use_output_info flag
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .grid_encoder import GridEncoder, GridEncoderConfig


@dataclass
class GoalEncoderConfig:
    """Configuration for GoalEncoder."""
    hidden_size: int = 512
    num_goal_tokens: int = 16  # Match puzzle_emb_len
    num_heads: int = 4
    use_cross_attention: bool = True
    aggregation: str = "attention"  # "attention", "mean", "max"
    use_channel_embedding: bool = True


class GoalEncoder(nn.Module):
    """
    Encode few-shot examples into goal representation.

    The goal can be computed in two modes:
    1. Input-only mode (use_output_info=False):
       - goal = aggregate(encoder(few_shot_inputs))
       - Simple replacement for puzzle_emb

    2. Goal conditioning mode (use_output_info=True):
       - goal = aggregate(encoder(outputs) - encoder(inputs))
       - Captures transformation pattern

    Args:
        grid_encoder: GridEncoder instance for encoding grids
        config: GoalEncoderConfig with hyperparameters

    Example:
        >>> grid_encoder = GridEncoder(GridEncoderConfig())
        >>> goal_encoder = GoalEncoder(grid_encoder, GoalEncoderConfig())
        >>> goal = goal_encoder(few_shot_inputs, few_shot_outputs)
        >>> print(goal.shape)  # [B, 16, 512]
    """

    def __init__(
        self,
        grid_encoder: GridEncoder,
        config: GoalEncoderConfig,
    ):
        super().__init__()
        self.grid_encoder = grid_encoder
        self.config = config

        hidden_size = config.hidden_size

        # Channel embedding: distinguish input (0) vs output (1)
        if config.use_channel_embedding:
            self.channel_embed = nn.Embedding(2, hidden_size)
        else:
            self.channel_embed = None

        # Goal query tokens (learnable)
        self.goal_queries = nn.Parameter(
            torch.randn(config.num_goal_tokens, hidden_size) * 0.02
        )

        # Cross-attention for aggregating pair representations into goal
        if config.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=config.num_heads,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_size)

        # MLP for transformation features (when using output info)
        self.transform_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def _aggregate_representations(
        self,
        repr: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Aggregate representations across examples into goal tokens.

        Args:
            repr: [B, N, D] or [B, 2N, D] representations
            batch_size: Original batch size B

        Returns:
            [B, num_goal_tokens, D] goal representation
        """
        if self.config.use_cross_attention:
            # Goal queries attend to representations
            # Cast goal_queries to the same dtype as repr to avoid dtype mismatch
            # (goal_queries is float32, but repr may be bfloat16 in mixed precision training)
            queries = self.goal_queries.unsqueeze(0).expand(batch_size, -1, -1).to(repr.dtype)
            goal, _ = self.cross_attn(queries, repr, repr)
            goal = self.attn_norm(goal + queries)  # Residual
        else:
            # Simple aggregation
            if self.config.aggregation == "mean":
                agg = repr.mean(dim=1)  # [B, D]
            elif self.config.aggregation == "max":
                agg = repr.max(dim=1)[0]  # [B, D]
            else:
                agg = repr.mean(dim=1)

            # Expand to goal tokens
            goal = agg.unsqueeze(1).expand(-1, self.config.num_goal_tokens, -1)

        return goal

    def forward(
        self,
        few_shot_inputs: torch.Tensor,
        few_shot_outputs: Optional[torch.Tensor] = None,
        use_output_info: bool = True,
    ) -> torch.Tensor:
        """
        Compute goal from few-shot examples.

        Args:
            few_shot_inputs: [B, N, L] few-shot input grids
            few_shot_outputs: [B, N, L] few-shot output grids (optional)
            use_output_info: Whether to use output information for goal

        Returns:
            goal: [B, num_goal_tokens, hidden_size] goal representation
                  Compatible with puzzle_emb shape
        """
        B, N, L = few_shot_inputs.shape

        # Encode inputs (always needed)
        in_repr = self.grid_encoder(
            few_shot_inputs,
            return_type="global"
        )  # [B, N, D]

        if use_output_info and few_shot_outputs is not None:
            # Goal conditioning mode: use transformation pattern
            out_repr = self.grid_encoder(
                few_shot_outputs,
                return_type="global"
            )  # [B, N, D]

            # Compute transformation: output - input
            transform = out_repr - in_repr  # [B, N, D]
            transform = self.transform_mlp(transform)

            # Add channel embeddings if configured
            if self.channel_embed is not None:
                in_channel = self.channel_embed(
                    torch.zeros(B, N, dtype=torch.long, device=in_repr.device)
                )
                out_channel = self.channel_embed(
                    torch.ones(B, N, dtype=torch.long, device=out_repr.device)
                )
                in_repr = in_repr + in_channel
                out_repr = out_repr + out_channel

            # Combine input, output, and transform representations
            combined = torch.cat([in_repr, out_repr, transform], dim=1)  # [B, 3N, D]
            goal = self._aggregate_representations(combined, B)

        else:
            # Input-only mode: simple replacement for puzzle_emb
            if self.channel_embed is not None:
                in_channel = self.channel_embed(
                    torch.zeros(B, N, dtype=torch.long, device=in_repr.device)
                )
                in_repr = in_repr + in_channel

            goal = self._aggregate_representations(in_repr, B)

        # Output projection and normalization
        goal = self.output_proj(goal)
        goal = self.output_norm(goal)

        return goal  # [B, num_goal_tokens, D]

    def get_output_shape(self) -> Tuple[int, int]:
        """Return output shape (num_tokens, hidden_size)."""
        return (self.config.num_goal_tokens, self.config.hidden_size)


class ContrastiveGoalEncoder(GoalEncoder):
    """
    GoalEncoder with contrastive learning support.

    Adds methods for contrastive pre-training where same-task
    representations should cluster together.
    """

    def __init__(
        self,
        grid_encoder: GridEncoder,
        config: GoalEncoderConfig,
        temperature: float = 0.07,
    ):
        super().__init__(grid_encoder, config)
        self.temperature = temperature

        # Projection head for contrastive learning
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def compute_contrastive_loss(
        self,
        few_shot_inputs: torch.Tensor,
        few_shot_outputs: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for representation learning.

        Same-task examples should have similar representations.

        Args:
            few_shot_inputs: [B, N, L]
            few_shot_outputs: [B, N, L]
            task_ids: [B] task identifiers

        Returns:
            Contrastive loss scalar
        """
        B, N, L = few_shot_inputs.shape

        # Get goal representations
        goal = self.forward(few_shot_inputs, few_shot_outputs)  # [B, 16, D]

        # Pool to single vector for contrastive
        goal_pooled = goal.mean(dim=1)  # [B, D]
        goal_proj = self.contrastive_proj(goal_pooled)  # [B, D]
        goal_proj = F.normalize(goal_proj, dim=-1)

        # Compute similarity matrix
        sim_matrix = goal_proj @ goal_proj.T / self.temperature  # [B, B]

        # Positive mask: same task
        pos_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)
        pos_mask = pos_mask & ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(
            torch.eye(B, dtype=torch.bool, device=sim_matrix.device),
            0
        )

        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # Avoid division by zero
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        # Only count samples with at least one positive
        has_positive = pos_mask.sum(dim=1) > 0
        if has_positive.sum() > 0:
            loss = loss[has_positive].mean()
        else:
            loss = torch.tensor(0.0, device=sim_matrix.device)

        return loss


# Convenience function
def create_goal_encoder(
    grid_encoder: GridEncoder,
    num_goal_tokens: int = 16,
    hidden_size: int = 512,
    **kwargs
) -> GoalEncoder:
    """
    Create a GoalEncoder with common defaults.

    Args:
        grid_encoder: GridEncoder instance
        num_goal_tokens: Number of goal tokens (should match puzzle_emb_len)
        hidden_size: Hidden dimension
        **kwargs: Additional config parameters

    Returns:
        Configured GoalEncoder instance
    """
    config = GoalEncoderConfig(
        hidden_size=hidden_size,
        num_goal_tokens=num_goal_tokens,
        **kwargs
    )
    return GoalEncoder(grid_encoder, config)
