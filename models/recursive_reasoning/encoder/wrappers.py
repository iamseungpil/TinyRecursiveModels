"""
EnhancedTRM: Wrapper that adds encoder capabilities to original TRM.

This module wraps the original TRM model and adds:
- GridEncoder for input embedding
- GoalEncoder for goal conditioning
- ZLHistoryAggregator for z_L history

CRITICAL DESIGN REQUIREMENT:
============================
The original TRM uses puzzle_emb as a 16-token prefix:
  - Input structure: [puzzle_emb(16), grid_tokens(L)]
  - q_head uses: z_H[:, 0] (first puzzle_emb token as CLS/global token)
  - lm_head uses: z_H[:, puzzle_emb_len:] (skip prefix)

This wrapper supports TWO modes:

1. ORIGINAL MODE (use_grid_encoder_for_input=False):
   - Exact same behavior as main branch TRM
   - Uses puzzle_emb + embed_tokens + embed_scale
   - Learned position encoding if configured

2. ENCODER MODE (use_grid_encoder_for_input=True):
   - GridEncoder processes input grid
   - CLS token at position 0 (for q_head)
   - Padding tokens at positions 1-15
   - Grid tokens at positions 16+
   - NO embed_scale applied (GridEncoder output is normalized)

Position Encoding Strategy (Encoder Mode):
==========================================
GridEncoder and TRM can each contribute position information.
These serve DIFFERENT semantic purposes:

1. GridEncoder pos_embed (grid_position_encoding_mode):
   - "additive": Positions 0-L added to tokens (grid spatial awareness)
   - "none": No position encoding (rely on TRM RoPE only)
   - "offset": Positions offset-to-offset+L (aligned with TRM sequence)

2. TRM RoPE (disable_rope_for_encoder):
   - False (default): RoPE applied in attention to all tokens
   - True: RoPE disabled when using GridEncoder

Recommended Configurations:
- Hybrid (default): grid_position_encoding_mode="additive", disable_rope=False
  * Best of both: grid spatial + sequence position
- RoPE Only: grid_position_encoding_mode="none", disable_rope=False
  * Like standard transformer, no grid spatial awareness
- GridEncoder Only: grid_position_encoding_mode="additive", disable_rope=True
  * Grid spatial only, no sequence position from RoPE
- Aligned: grid_position_encoding_mode="offset", disable_rope=False
  * Both encodings use same position indices

Scale Logic:
============
- GridEncoder output: Already normalized by LayerNorm (std ~ 1.0)
- embed_tokens output: Raw embeddings, scaled by embed_scale * sqrt(hidden_size)
- CLS/padding tokens: Learnable, initialized to small values (std=0.02)
  They will learn to match GridEncoder magnitude during training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

from .grid_encoder import GridEncoder, GridEncoderConfig, create_grid_encoder
from .goal_encoder import GoalEncoder, GoalEncoderConfig, create_goal_encoder
from .history_aggregator import ZLHistoryAggregator, ZLHistoryAggregatorConfig, create_history_aggregator

# Import TRM carry types at module level to avoid import inside forward pass
from ..trm import TinyRecursiveReasoningModel_ACTV1InnerCarry, TinyRecursiveReasoningModel_ACTV1Carry


@dataclass
class EnhancedTRMConfig:
    """
    Configuration for EnhancedTRM wrapper.

    Attributes:
        use_goal_encoder: Replace puzzle_emb with goal from encoder
        use_goal_conditioning: Use output info for goal (output - input)
        use_grid_encoder_for_input: Replace embed_tokens with GridEncoder
        use_original_puzzle_emb: When True AND use_grid_encoder_for_input=True,
                                 use original puzzle_emb instead of CLS+padding
                                 This allows testing GridEncoder for grid tokens
                                 while keeping original prefix behavior.
        use_history: Enable z_L history aggregation
        freeze_trm: Freeze original TRM parameters
        freeze_goal_encoder: Freeze goal encoder parameters

        # GridEncoder config
        grid_encoder_layers: Number of transformer layers in GridEncoder
        grid_encoder_heads: Number of attention heads in GridEncoder

        # Position Encoding Strategy (see module docstring for details)
        grid_position_encoding_mode: GridEncoder position encoding strategy
            - "additive": pos_embed[0:L] added to grid tokens (default)
            - "none": No position encoding from GridEncoder
            - "offset": pos_embed[offset:offset+L] aligned with TRM positions
        disable_rope_for_encoder: Disable TRM RoPE when using GridEncoder
            - False (default): RoPE still applied in attention
            - True: No RoPE, rely on GridEncoder position only

        # GoalEncoder config
        num_goal_tokens: Number of goal tokens (should match puzzle_emb_len=16)
        goal_aggregation: Aggregation method ("attention", "mean", "max")

        # History config
        history_window_size: Number of past z_L states to keep
    """
    # Feature flags
    use_goal_encoder: bool = False
    use_goal_conditioning: bool = False
    use_grid_encoder_for_input: bool = False
    use_original_puzzle_emb: bool = True  # When True with encoder mode, keep puzzle_emb
    use_history: bool = False

    # Training control
    freeze_trm: bool = True
    freeze_goal_encoder: bool = False

    # GridEncoder config
    grid_encoder_layers: int = 2
    grid_encoder_heads: int = 4

    # Position Encoding Strategy
    grid_position_encoding_mode: str = "additive"  # "additive", "none", "offset"
    disable_rope_for_encoder: bool = False  # True to disable RoPE in encoder mode

    # GoalEncoder config
    num_goal_tokens: int = 16  # MUST match puzzle_emb_len
    goal_aggregation: str = "attention"

    # History config
    history_window_size: int = 3


class EnhancedTRM(nn.Module):
    """
    Wrapper around original TRM with encoder capabilities.

    This wrapper allows incremental testing of encoder features
    without modifying the original TRM implementation.

    IMPORTANT: This wrapper preserves q_head semantics by ensuring
    position 0 is always a CLS/global representation token.

    Two Modes:
    ==========
    1. Original Mode (use_grid_encoder_for_input=False):
       - Exact same behavior as main branch TRM
       - puzzle_emb + embed_tokens + embed_scale + position encoding

    2. Encoder Mode (use_grid_encoder_for_input=True):
       - GridEncoder for grid tokens (normalized, with its own pos encoding)
       - Prefix options:
         a) use_original_puzzle_emb=True: Use original puzzle_emb (scaled)
         b) use_original_puzzle_emb=False: Use CLS + padding tokens

    Args:
        base_trm: Original TRM model instance
        config: EnhancedTRMConfig

    Example:
        >>> # Load original TRM
        >>> base_trm = TinyRecursiveReasoningModel_ACTV1(trm_config)
        >>> base_trm.load_state_dict(checkpoint)
        >>>
        >>> # Wrap with encoder (Stage 1)
        >>> config = EnhancedTRMConfig(use_grid_encoder_for_input=True)
        >>> enhanced = EnhancedTRM(base_trm, config)
        >>>
        >>> # Forward (no few-shot data needed for Stage 1)
        >>> output = enhanced(carry, batch)
    """

    def __init__(
        self,
        base_trm: nn.Module,
        config: EnhancedTRMConfig,
    ):
        super().__init__()
        self.base_trm = base_trm
        self.config = config

        # Get hidden size and puzzle_emb_len from base TRM
        self.hidden_size = base_trm.inner.config.hidden_size
        self.puzzle_emb_len = base_trm.inner.puzzle_emb_len  # Should be 16
        self.forward_dtype = base_trm.inner.forward_dtype

        # Store embed_scale from base TRM for scaling prefix embeddings
        self.embed_scale = base_trm.inner.embed_scale

        # Validate num_goal_tokens matches puzzle_emb_len
        if config.num_goal_tokens != self.puzzle_emb_len:
            raise ValueError(
                f"num_goal_tokens ({config.num_goal_tokens}) must match "
                f"puzzle_emb_len ({self.puzzle_emb_len})"
            )

        # Create GridEncoder if needed
        if config.use_goal_encoder or config.use_grid_encoder_for_input:
            self.grid_encoder = create_grid_encoder(
                hidden_size=self.hidden_size,
                num_layers=config.grid_encoder_layers,
                num_heads=config.grid_encoder_heads,
                # Position encoding strategy from config
                position_encoding_mode=config.grid_position_encoding_mode,
                position_offset=self.puzzle_emb_len,  # Offset by prefix length
                # Don't share embed_tokens - GridEncoder has its own
            )
        else:
            self.grid_encoder = None

        # Create GoalEncoder if needed
        if config.use_goal_encoder:
            self.goal_encoder = create_goal_encoder(
                grid_encoder=self.grid_encoder,
                num_goal_tokens=config.num_goal_tokens,
                hidden_size=self.hidden_size,
                aggregation=config.goal_aggregation,
            )
        else:
            self.goal_encoder = None

        # Create prefix tokens for encoder mode without puzzle_emb
        # Only create CLS + padding when:
        # - Using GridEncoder for input AND
        # - NOT using original puzzle_emb AND
        # - NOT using GoalEncoder (which provides its own prefix)
        if (config.use_grid_encoder_for_input and
            not config.use_original_puzzle_emb and
            not config.use_goal_encoder):
            # CLS token - will be used by q_head
            # Initialize small, will learn to match GridEncoder magnitude
            # NOTE: Do NOT specify dtype here - let model's device/dtype determine it
            # when moved to CUDA. This avoids bfloat16 CPU tensor issues.
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * 0.02
            )
            # Padding tokens (15 tokens to complete the 16-token prefix)
            self.prefix_padding = nn.Parameter(
                torch.randn(1, self.puzzle_emb_len - 1, self.hidden_size) * 0.02
            )
        else:
            self.cls_token = None
            self.prefix_padding = None

        # Create history aggregator if needed
        if config.use_history:
            self.history_aggregator = create_history_aggregator(
                hidden_size=self.hidden_size,
                num_heads=8,
                window_size=config.history_window_size,
            )
        else:
            self.history_aggregator = None

        # Apply freezing
        self._apply_freezing()

    def _apply_freezing(self):
        """Apply freezing based on config."""
        if self.config.freeze_trm:
            for param in self.base_trm.parameters():
                param.requires_grad = False

        if self.config.freeze_goal_encoder and self.goal_encoder is not None:
            for param in self.goal_encoder.parameters():
                param.requires_grad = False

    def unfreeze_trm(self, lr_scale: float = 0.1):
        """
        Unfreeze TRM for fine-tuning.

        Args:
            lr_scale: Suggested LR scale (for reference, not enforced)
        """
        for param in self.base_trm.parameters():
            param.requires_grad = True
        self.config.freeze_trm = False

    def get_trainable_parameters(self):
        """Get only trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_parameter_groups(
        self,
        encoder_lr: float = 1e-4,
        trm_lr: float = 1e-6,
        prefix_lr: float = 1e-2,
        puzzle_emb_lr: Optional[float] = None,
    ):
        """
        Get parameter groups for optimizer with different LRs.

        Args:
            encoder_lr: Learning rate for encoder components (GridEncoder)
            trm_lr: Learning rate for TRM components
            prefix_lr: Learning rate for CLS and prefix tokens
            puzzle_emb_lr: Learning rate for puzzle_emb (defaults to prefix_lr)

        Returns:
            List of parameter group dicts for optimizer

        Parameter Group Logic:
        ======================
        - Encoder mode with use_original_puzzle_emb=False:
          * GridEncoder params -> encoder_lr
          * CLS + padding -> prefix_lr
          * TRM (excluding puzzle_emb) -> trm_lr
          * puzzle_emb is NOT included (not used)

        - Encoder mode with use_original_puzzle_emb=True:
          * GridEncoder params -> encoder_lr
          * TRM (excluding puzzle_emb) -> trm_lr
          * puzzle_emb -> puzzle_emb_lr (high LR)

        - Original mode:
          * TRM (including puzzle_emb) -> trm_lr
          * puzzle_emb gets separate group with puzzle_emb_lr
        """
        if puzzle_emb_lr is None:
            puzzle_emb_lr = prefix_lr

        groups = []

        # Encoder parameters (GridEncoder, GoalEncoder)
        encoder_params = []
        if self.grid_encoder is not None:
            encoder_params.extend(self.grid_encoder.parameters())
        if self.goal_encoder is not None:
            encoder_params.extend(self.goal_encoder.parameters())

        if encoder_params:
            groups.append({
                "params": encoder_params,
                "lr": encoder_lr,
                "name": "encoder"
            })

        # Prefix tokens (CLS and padding) - only when using CLS+padding prefix
        prefix_params = []
        if self.cls_token is not None:
            prefix_params.append(self.cls_token)
        if self.prefix_padding is not None:
            prefix_params.append(self.prefix_padding)

        if prefix_params:
            groups.append({
                "params": prefix_params,
                "lr": prefix_lr,
                "name": "prefix"
            })

        # History aggregator
        if self.history_aggregator is not None:
            groups.append({
                "params": list(self.history_aggregator.parameters()),
                "lr": encoder_lr,
                "name": "history"
            })

        # TRM parameters handling
        if not self.config.freeze_trm:
            trm_params = []
            puzzle_emb_params = []

            for name, param in self.base_trm.named_parameters():
                if 'puzzle_emb' in name:
                    # puzzle_emb handling depends on mode
                    if self.config.use_grid_encoder_for_input:
                        if self.config.use_original_puzzle_emb:
                            # Using puzzle_emb with GridEncoder - include with high LR
                            puzzle_emb_params.append(param)
                        else:
                            # Using CLS+padding instead - EXCLUDE puzzle_emb
                            continue
                    else:
                        # Original mode - include puzzle_emb with high LR
                        puzzle_emb_params.append(param)
                else:
                    trm_params.append(param)

            if trm_params:
                groups.append({
                    "params": trm_params,
                    "lr": trm_lr,
                    "name": "trm"
                })

            if puzzle_emb_params:
                groups.append({
                    "params": puzzle_emb_params,
                    "lr": puzzle_emb_lr,
                    "name": "puzzle_emb"
                })

        return groups

    def _compute_goal(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Compute goal from few-shot examples.

        Args:
            batch: Batch dict with few_shot_inputs/outputs

        Returns:
            goal [B, num_goal_tokens, D] or None if not using goal encoder
        """
        if not self.config.use_goal_encoder:
            return None

        few_shot_inputs = batch.get("few_shot_inputs")
        few_shot_outputs = batch.get("few_shot_outputs")

        if few_shot_inputs is None:
            return None

        goal = self.goal_encoder(
            few_shot_inputs=few_shot_inputs,
            few_shot_outputs=few_shot_outputs,
            use_output_info=self.config.use_goal_conditioning,
        )

        return goal

    def _get_puzzle_emb_prefix(
        self,
        puzzle_identifiers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get puzzle_emb prefix using original TRM logic.

        This is the ORIGINAL prefix computation - same as main branch TRM.
        Returns scaled embeddings with position encoding applied.

        Args:
            puzzle_identifiers: [B] tensor of puzzle IDs

        Returns:
            [B, puzzle_emb_len, D] scaled prefix embeddings
        """
        inner = self.base_trm.inner

        # Get puzzle embedding
        puzzle_emb = inner.puzzle_emb(puzzle_identifiers)

        # Pad if needed
        pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_emb.shape[-1]
        if pad_count > 0:
            puzzle_emb = F.pad(puzzle_emb, (0, pad_count))

        # Reshape to [B, puzzle_emb_len, D]
        puzzle_emb = puzzle_emb.view(-1, self.puzzle_emb_len, self.hidden_size)

        return puzzle_emb

    def _get_cls_prefix(
        self,
        batch_size: int,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Get CLS + padding prefix.

        This is the REPLACEMENT prefix for encoder mode.
        CLS token at position 0 (for q_head), padding at 1-15.

        Args:
            batch_size: Batch size B
            dtype: Target dtype for the prefix (e.g., bfloat16 for mixed precision)

        Returns:
            [B, puzzle_emb_len, D] prefix tokens (NOT scaled)
        """
        # Cast to target dtype if specified to avoid dtype mismatch
        # (cls_token/prefix_padding are float32, but may need bfloat16 in mixed precision)
        cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        pad = self.prefix_padding.expand(batch_size, -1, -1)  # [B, 15, D]
        if dtype is not None:
            cls = cls.to(dtype)
            pad = pad.to(dtype)
        prefix = torch.cat([cls, pad], dim=1)  # [B, 16, D]
        return prefix

    def _get_input_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get input embeddings with proper prefix.

        TWO MODES:
        ==========

        1. ORIGINAL MODE (use_grid_encoder_for_input=False):
           - Uses original TRM _input_embeddings logic exactly
           - embed_tokens for grid, puzzle_emb for prefix
           - Learned position encoding if configured
           - embed_scale applied to everything

        2. ENCODER MODE (use_grid_encoder_for_input=True):
           - GridEncoder for grid tokens (already normalized, has own pos encoding)
           - Prefix options:
             a) goal from GoalEncoder (if use_goal_encoder=True)
             b) puzzle_emb (if use_original_puzzle_emb=True) - scaled
             c) CLS + padding (otherwise) - NOT scaled (matches GridEncoder)
           - NO additional TRM position encoding
           - NO embed_scale on GridEncoder output

        Args:
            batch: Batch dict with inputs, puzzle_identifiers
            goal: Optional goal tensor from GoalEncoder

        Returns:
            Combined embeddings [B, puzzle_emb_len + L, D]
        """
        inputs = batch["inputs"]
        B = inputs.shape[0]

        if not self.config.use_grid_encoder_for_input:
            # ================================================================
            # ORIGINAL MODE: Exact same as main branch TRM
            # ================================================================
            return self._original_input_embeddings(batch)

        else:
            # ================================================================
            # ENCODER MODE: GridEncoder for grid, various prefix options
            # ================================================================

            # Get grid embeddings from GridEncoder
            # GridEncoder output is already:
            # - Has its own position encoding (pos_embed in GridEncoder)
            # - Normalized by LayerNorm (std ~ 1.0)
            grid_emb = self.grid_encoder(inputs, return_type="tokens")  # [B, L, D]

            # Get prefix based on configuration
            if goal is not None:
                # GoalEncoder provides the prefix
                prefix = goal.to(self.forward_dtype)  # [B, 16, D]
            elif self.config.use_original_puzzle_emb:
                # Use original puzzle_emb - needs scaling to match original behavior
                prefix = self._get_puzzle_emb_prefix(batch["puzzle_identifiers"])
                # Apply position encoding and scaling like original TRM
                inner = self.base_trm.inner
                if inner.config.pos_encodings == "learned":
                    pos_emb = inner.embed_pos.embedding_weight[:self.puzzle_emb_len]
                    prefix = 0.707106781 * (prefix + pos_emb.to(self.forward_dtype))
                prefix = self.embed_scale * prefix
            else:
                # Use CLS + padding tokens
                # These are learnable and NOT scaled - they'll learn to match GridEncoder
                # Pass grid_emb.dtype to ensure consistent dtype with GridEncoder output
                prefix = self._get_cls_prefix(B, dtype=grid_emb.dtype)  # [B, 16, D]

            # Combine prefix and grid embeddings
            # IMPORTANT: NO scale or position encoding on grid_emb
            # GridEncoder already handles both internally
            embedding = torch.cat([prefix, grid_emb], dim=1)  # [B, 16+L, D]

            return embedding

    def _original_input_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get input embeddings using ORIGINAL TRM logic.

        This is an EXACT copy of TinyRecursiveReasoningModel_ACTV1_Inner._input_embeddings.
        Used to ensure original mode produces identical results to main branch.

        Args:
            batch: Batch dict with inputs, puzzle_identifiers

        Returns:
            [B, puzzle_emb_len + L, D] embeddings (scaled, with position encoding)
        """
        inner = self.base_trm.inner

        # Token embedding
        embedding = inner.embed_tokens(batch["inputs"].to(torch.int32))

        # Puzzle embeddings
        if inner.config.puzzle_emb_ndim > 0:
            puzzle_embedding = inner.puzzle_emb(batch["puzzle_identifiers"])

            pad_count = inner.puzzle_emb_len * inner.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, inner.puzzle_emb_len, inner.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if inner.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + inner.embed_pos.embedding_weight.to(inner.forward_dtype))

        # Scale
        return inner.embed_scale * embedding

    def forward(
        self,
        carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass with enhanced features.

        Args:
            carry: TRM carry state
            batch: Batch dict with inputs, labels, puzzle_identifiers,
                   and optionally few_shot_inputs/outputs

        Returns:
            Tuple of (new_carry, outputs_dict)
        """
        # Compute goal if using goal encoder
        goal = self._compute_goal(batch)

        # If using custom embeddings, we need to modify the forward
        if self.config.use_goal_encoder or self.config.use_grid_encoder_for_input:
            # Get custom embeddings
            input_embeddings = self._get_input_embeddings(batch, goal)

            # Run TRM inner forward with custom embeddings
            return self._forward_with_custom_embeddings(carry, batch, input_embeddings)
        else:
            # Use original TRM forward
            return self.base_trm(carry, batch)

    def _forward_with_custom_embeddings(
        self,
        carry,
        batch: Dict[str, torch.Tensor],
        input_embeddings: torch.Tensor,
    ):
        """
        Run TRM forward with custom input embeddings.

        This replicates the TRM inner forward logic but uses
        custom input embeddings instead of the default.

        IMPORTANT: Position 0 is CLS token (used by q_head).

        Position Encoding Strategy:
        ===========================
        RoPE in TRM attention is controlled by disable_rope_for_encoder:
        - False (default): RoPE applied via cos_sin in attention
        - True: cos_sin=None, no RoPE applied

        Combined with GridEncoder's position_encoding_mode, this allows:
        - Hybrid: GridEncoder pos_embed + TRM RoPE (both contribute)
        - RoPE Only: No GridEncoder pos + TRM RoPE
        - GridEncoder Only: GridEncoder pos + no RoPE
        - Aligned: GridEncoder offset pos + TRM RoPE (same indices)
        """
        inner = self.base_trm.inner
        config = inner.config

        # Get sequence info for RoPE
        # Controlled by disable_rope_for_encoder when using GridEncoder
        if (self.config.use_grid_encoder_for_input and
            self.config.disable_rope_for_encoder):
            # Disable RoPE when using GridEncoder (if configured)
            cos_sin = None
        else:
            # Use RoPE (default behavior)
            cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None

        seq_info = dict(cos_sin=cos_sin)

        # Initialize or reset carry
        new_inner_carry = inner.reset_carry(carry.halted, carry.inner_carry)
        z_H, z_L = new_inner_carry.z_H, new_inner_carry.z_L

        # History tracking for z_L history aggregation
        z_L_history: List[torch.Tensor] = []

        # Forward iterations
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(config.H_cycles - 1):
                for _L_step in range(config.L_cycles):
                    injection = z_H + input_embeddings
                    # Add history context if using history
                    if self.history_aggregator is not None:
                        history_context = self.history_aggregator(z_L, z_L_history)
                        injection = injection + history_context
                    z_L = inner.L_level(z_L, injection, **seq_info)

                # Update history after L cycles complete
                if self.history_aggregator is not None:
                    z_L_history.append(z_L.detach())
                    z_L_history = z_L_history[-self.config.history_window_size:]

                z_H = inner.L_level(z_H, z_L, **seq_info)

        # 1 cycle with grad
        for _L_step in range(config.L_cycles):
            injection = z_H + input_embeddings
            # Add history context if using history
            if self.history_aggregator is not None:
                history_context = self.history_aggregator(z_L, z_L_history)
                injection = injection + history_context
            z_L = inner.L_level(z_L, injection, **seq_info)

        z_H = inner.L_level(z_H, z_L, **seq_info)

        # Create new carry (using module-level import)
        new_inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach()
        )

        # LM outputs
        # IMPORTANT: lm_head skips first puzzle_emb_len positions
        output = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]

        # Q head uses position 0 (CLS token)
        # IMPORTANT: Position 0 must be CLS/global representation
        q_logits = inner.q_head(z_H[:, 0]).to(torch.float32)

        outputs = {
            "logits": output,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
        }

        # Update outer carry
        new_steps = torch.where(carry.halted, 0, carry.steps) + 1
        is_last_step = new_steps >= config.halt_max_steps
        halted = is_last_step

        if self.training and config.halt_max_steps > 1:
            if config.no_ACT_continue:
                halted = halted | (q_logits[..., 0] > 0)
            else:
                halted = halted | (q_logits[..., 0] > q_logits[..., 1])

            min_halt_steps = (
                (torch.rand_like(q_logits[..., 0]) < config.halt_exploration_prob) *
                torch.randint_like(new_steps, low=2, high=config.halt_max_steps + 1)
            )
            halted = halted & (new_steps >= min_halt_steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # Create outer carry (using module-level import)
        new_carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )

        return new_carry, outputs

    @property
    def inner(self):
        """Access base TRM inner for compatibility."""
        return self.base_trm.inner

    @property
    def puzzle_emb(self):
        """Access puzzle_emb for compatibility."""
        return self.base_trm.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Create initial carry state."""
        return self.base_trm.initial_carry(batch)


def create_enhanced_trm(
    base_trm: nn.Module,
    use_goal_encoder: bool = True,
    use_goal_conditioning: bool = False,
    freeze_trm: bool = True,
    **kwargs
) -> EnhancedTRM:
    """
    Convenience function to create EnhancedTRM.

    Args:
        base_trm: Original TRM model
        use_goal_encoder: Enable goal encoder
        use_goal_conditioning: Use output info for goal
        freeze_trm: Freeze TRM parameters
        **kwargs: Additional config parameters

    Returns:
        EnhancedTRM instance
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=use_goal_encoder,
        use_goal_conditioning=use_goal_conditioning,
        freeze_trm=freeze_trm,
        **kwargs
    )
    return EnhancedTRM(base_trm, config)


# ============================================================================
# Stage-specific factory functions for easy testing
# ============================================================================

def create_stage0_baseline(base_trm: nn.Module) -> EnhancedTRM:
    """
    Stage 0: Original TRM behavior (baseline).

    This is a passthrough wrapper - no encoder features enabled.
    EXACT same behavior as main branch TRM.
    Useful for comparison testing.

    Configuration:
    - use_grid_encoder_for_input=False
    - use_original_puzzle_emb=True (but irrelevant since no encoder)
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=False,
        use_goal_conditioning=False,
        use_grid_encoder_for_input=False,
        use_original_puzzle_emb=True,
        use_history=False,
        freeze_trm=False,  # Allow training
    )
    return EnhancedTRM(base_trm, config)


def create_stage1_grid_encoder(
    base_trm: nn.Module,
    freeze_trm: bool = False,
    use_original_puzzle_emb: bool = False,
    grid_encoder_layers: int = 2,
    grid_encoder_heads: int = 4,
    grid_position_encoding_mode: str = "additive",
    disable_rope_for_encoder: bool = False,
) -> EnhancedTRM:
    """
    Stage 1: GridEncoder for input embedding only.

    This stage:
    - Uses GridEncoder to embed input grids (replaces embed_tokens)
    - Position 0 = CLS token (for q_head compatibility)
    - No few-shot data required

    Prefix Options:
    - use_original_puzzle_emb=False (default): CLS(1) + padding(15)
      * Learnable tokens that match GridEncoder magnitude
      * puzzle_emb is NOT used (excluded from optimizer)

    - use_original_puzzle_emb=True: Original puzzle_emb
      * Keeps original prefix behavior
      * puzzle_emb IS used (included in optimizer)

    Position Encoding Strategy:
    - grid_position_encoding_mode: How GridEncoder adds position info
      * "additive" (default): pos_embed[0:L] added (grid spatial awareness)
      * "none": No position encoding (rely on TRM RoPE only)
      * "offset": pos_embed[16:16+L] added (aligned with TRM positions)
    - disable_rope_for_encoder: Whether to disable TRM RoPE
      * False (default): RoPE still applied in attention
      * True: No RoPE, rely on GridEncoder position only

    Args:
        base_trm: Original TRM model
        freeze_trm: Whether to freeze TRM parameters
        use_original_puzzle_emb: If True, use puzzle_emb; if False, use CLS+padding
        grid_encoder_layers: Number of GridEncoder layers
        grid_encoder_heads: Number of GridEncoder attention heads
        grid_position_encoding_mode: GridEncoder position encoding strategy
        disable_rope_for_encoder: Disable TRM RoPE when using GridEncoder

    Returns:
        EnhancedTRM configured for Stage 1
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=False,           # No goal encoder
        use_goal_conditioning=False,      # No goal conditioning
        use_grid_encoder_for_input=True,  # Use GridEncoder for input
        use_original_puzzle_emb=use_original_puzzle_emb,  # Toggle puzzle_emb
        use_history=False,                # No history
        freeze_trm=freeze_trm,
        grid_encoder_layers=grid_encoder_layers,
        grid_encoder_heads=grid_encoder_heads,
        # Position encoding strategy
        grid_position_encoding_mode=grid_position_encoding_mode,
        disable_rope_for_encoder=disable_rope_for_encoder,
    )
    return EnhancedTRM(base_trm, config)


def create_stage2_goal_encoder(
    base_trm: nn.Module,
    freeze_trm: bool = True,
) -> EnhancedTRM:
    """
    Stage 2: GoalEncoder for puzzle_emb replacement (input-only goal).

    This stage:
    - Uses GoalEncoder to create 16 goal tokens from few-shot inputs
    - Goal tokens replace puzzle_emb as prefix
    - Position 0 = first goal token (acts as CLS)
    - Requires few_shot_inputs in batch

    Args:
        base_trm: Original TRM model
        freeze_trm: Whether to freeze TRM parameters

    Returns:
        EnhancedTRM configured for Stage 2
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=True,
        use_goal_conditioning=False,  # No output info - input only
        use_grid_encoder_for_input=False,
        use_original_puzzle_emb=True,  # Irrelevant when goal encoder is used
        use_history=False,
        freeze_trm=freeze_trm,
    )
    return EnhancedTRM(base_trm, config)


def create_stage3_goal_conditioning(
    base_trm: nn.Module,
    freeze_trm: bool = True,
) -> EnhancedTRM:
    """
    Stage 3: GoalEncoder with full goal conditioning.

    This stage:
    - Uses GoalEncoder with output info (output - input transformation)
    - Goal captures the transformation pattern from few-shot pairs
    - Requires both few_shot_inputs and few_shot_outputs in batch

    Args:
        base_trm: Original TRM model
        freeze_trm: Whether to freeze TRM parameters

    Returns:
        EnhancedTRM configured for Stage 3
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=True,
        use_goal_conditioning=True,  # Use output info
        use_grid_encoder_for_input=False,
        use_original_puzzle_emb=True,
        use_history=False,
        freeze_trm=freeze_trm,
    )
    return EnhancedTRM(base_trm, config)


def create_stage4_with_history(
    base_trm: nn.Module,
    freeze_trm: bool = False,
    use_original_puzzle_emb: bool = False,
    history_window_size: int = 3,
    grid_position_encoding_mode: str = "additive",
    disable_rope_for_encoder: bool = False,
) -> EnhancedTRM:
    """
    Stage 4: Full integration with z_L history aggregation.

    This stage:
    - All Stage 3 features (goal conditioning)
    - GridEncoder for input (with proper normalization)
    - z_L history aggregation via cross-attention
    - Past z_L states provide detailed reasoning context

    Position Encoding Strategy:
    - grid_position_encoding_mode: How GridEncoder adds position info
      * "additive" (default): pos_embed[0:L] added (grid spatial awareness)
      * "none": No position encoding (rely on TRM RoPE only)
      * "offset": pos_embed[16:16+L] added (aligned with TRM positions)
    - disable_rope_for_encoder: Whether to disable TRM RoPE
      * False (default): RoPE still applied in attention
      * True: No RoPE, rely on GridEncoder position only

    Args:
        base_trm: Original TRM model
        freeze_trm: Whether to freeze TRM parameters
        use_original_puzzle_emb: If True, use puzzle_emb; if False, use CLS+padding
        history_window_size: Number of past z_L states to keep
        grid_position_encoding_mode: GridEncoder position encoding strategy
        disable_rope_for_encoder: Disable TRM RoPE when using GridEncoder

    Returns:
        EnhancedTRM configured for Stage 4
    """
    config = EnhancedTRMConfig(
        use_goal_encoder=True,
        use_goal_conditioning=True,
        use_grid_encoder_for_input=True,
        use_original_puzzle_emb=use_original_puzzle_emb,
        use_history=True,
        freeze_trm=freeze_trm,
        history_window_size=history_window_size,
        # Position encoding strategy
        grid_position_encoding_mode=grid_position_encoding_mode,
        disable_rope_for_encoder=disable_rope_for_encoder,
    )
    return EnhancedTRM(base_trm, config)


# Legacy compatibility - keep old name pointing to new function
def create_stage1_baseline(base_trm: nn.Module) -> EnhancedTRM:
    """
    Stage 0: Original TRM behavior (baseline).

    Legacy alias for create_stage0_baseline.
    """
    return create_stage0_baseline(base_trm)


# Alias for backward compatibility
create_stage2_encoder_only = create_stage2_goal_encoder
create_stage4_full_integration = create_stage4_with_history
