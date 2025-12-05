"""
Encoder module for TRM enhancement.

This module provides encoder components for goal conditioning:
- GridEncoder: Unified encoder for all ARC grids
- GoalEncoder: Extract goal representation from few-shot examples
- ZLHistoryAggregator: Cross-attention to past z_L states
- EnhancedTRM: Wrapper that adds encoder capabilities to original TRM

Architecture Overview:
=====================
Original TRM uses puzzle_emb as a 16-token prefix:
  - Input: [puzzle_emb(16), grid_tokens(L)]
  - q_head uses position 0 (first puzzle_emb token) as CLS
  - lm_head skips first 16 positions

EnhancedTRM supports TWO modes:

1. ORIGINAL MODE (use_grid_encoder_for_input=False):
   - Exact same behavior as main branch TRM
   - Uses puzzle_emb + embed_tokens + embed_scale
   - Learned position encoding if configured

2. ENCODER MODE (use_grid_encoder_for_input=True):
   - GridEncoder for grid tokens (normalized, has own pos encoding)
   - Prefix options:
     a) use_original_puzzle_emb=True: Use original puzzle_emb (scaled)
     b) use_original_puzzle_emb=False: Use CLS + padding tokens (not scaled)
   - NO additional TRM position encoding
   - NO embed_scale on GridEncoder output

Staged Testing:
==============
>>> # Stage 0: Baseline (original TRM behavior)
>>> model = create_stage0_baseline(base_trm)
>>>
>>> # Stage 1: GridEncoder for input embedding only
>>> model = create_stage1_grid_encoder(base_trm)
>>>
>>> # Stage 2: GoalEncoder from few-shot inputs (no output info)
>>> model = create_stage2_goal_encoder(base_trm)
>>>
>>> # Stage 3: GoalEncoder with full goal conditioning
>>> model = create_stage3_goal_conditioning(base_trm)
>>>
>>> # Stage 4: Full integration with z_L history
>>> model = create_stage4_with_history(base_trm)

Usage Example:
=============
>>> from models.recursive_reasoning.encoder import (
...     EnhancedTRM, EnhancedTRMConfig,
...     create_stage1_grid_encoder,
...     create_stage0_baseline,
... )
>>>
>>> # Load pretrained TRM
>>> base_trm = TinyRecursiveReasoningModel_ACTV1(config)
>>> base_trm.load_state_dict(checkpoint)
>>>
>>> # Wrap with Stage 1 encoder (CLS + padding prefix)
>>> enhanced = create_stage1_grid_encoder(base_trm, freeze_trm=False)
>>>
>>> # Or with Stage 1 encoder keeping puzzle_emb
>>> enhanced = create_stage1_grid_encoder(base_trm, use_original_puzzle_emb=True)
>>>
>>> # Get parameter groups for optimizer
>>> param_groups = enhanced.get_parameter_groups(
...     encoder_lr=1e-4,
...     prefix_lr=1e-2,
...     trm_lr=1e-4,
... )
>>> optimizer = AdamW(param_groups)
"""

from .grid_encoder import (
    GridEncoder,
    GridEncoderConfig,
    create_grid_encoder,
)

from .goal_encoder import (
    GoalEncoder,
    GoalEncoderConfig,
    ContrastiveGoalEncoder,
    create_goal_encoder,
)

from .history_aggregator import (
    ZLHistoryAggregator,
    ZLHistoryAggregatorConfig,
    create_history_aggregator,
)

from .wrappers import (
    EnhancedTRM,
    EnhancedTRMConfig,
    create_enhanced_trm,
    # Stage factories
    create_stage0_baseline,
    create_stage1_baseline,  # Legacy alias for stage0
    create_stage1_grid_encoder,
    create_stage2_goal_encoder,
    create_stage3_goal_conditioning,
    create_stage4_with_history,
    # Aliases for backward compatibility
    create_stage2_encoder_only,
    create_stage4_full_integration,
)

__all__ = [
    # GridEncoder
    "GridEncoder",
    "GridEncoderConfig",
    "create_grid_encoder",
    # GoalEncoder
    "GoalEncoder",
    "GoalEncoderConfig",
    "ContrastiveGoalEncoder",
    "create_goal_encoder",
    # History Aggregator
    "ZLHistoryAggregator",
    "ZLHistoryAggregatorConfig",
    "create_history_aggregator",
    # EnhancedTRM
    "EnhancedTRM",
    "EnhancedTRMConfig",
    "create_enhanced_trm",
    # Stage factories
    "create_stage0_baseline",
    "create_stage1_baseline",  # Legacy alias
    "create_stage1_grid_encoder",
    "create_stage2_goal_encoder",
    "create_stage3_goal_conditioning",
    "create_stage4_with_history",
    # Aliases
    "create_stage2_encoder_only",
    "create_stage4_full_integration",
]
