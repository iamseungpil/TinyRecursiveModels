from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.lstm_context import LSTMStyleContext

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    c_H: Optional[torch.Tensor] = None  # LSTM cell state (only used if use_lstm_gating=True)


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # LSTM-style gating for context accumulation
    use_lstm_gating: bool = False
    lstm_init_forget_bias: float = 1.0

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # LSTM-style context gating (optional)
        # Fixed: Use FP32 for LSTM gates (numerical stability)
        # BF16 has only 7-bit mantissa which can cause gate saturation
        if self.config.use_lstm_gating:
            self.lstm_context = LSTMStyleContext(
                hidden_dim=self.config.hidden_size,
                init_forget_bias=self.config.lstm_init_forget_bias,
                dtype=torch.float32  # Always FP32 for LSTM, regardless of forward_dtype
            )

            # Puzzle-aware c_H initialization: convert puzzle_emb to c_H initial state
            # Only if puzzle_emb is enabled (puzzle_emb_ndim > 0)
            if self.config.puzzle_emb_ndim > 0:
                self.puzzle_emb_to_c_H = nn.Linear(
                    self.config.puzzle_emb_ndim,
                    self.config.hidden_size,
                    bias=True,
                    dtype=torch.float32  # Match LSTM dtype for numerical stability
                )
                # Initialize with small weights to avoid dominating initial state
                nn.init.normal_(self.puzzle_emb_to_c_H.weight, std=0.01)
                nn.init.zeros_(self.puzzle_emb_to_c_H.bias)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        if self.config.use_lstm_gating:
            # LSTM cell state MUST be FP32 for numerical stability (not BF16)
            self.C_init = nn.Buffer(torch.zeros(self.config.hidden_size, dtype=torch.float32), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """Initialize carry state using learnable buffers (H_init, L_init, C_init).

        Fixed: Changed from torch.empty (uninitialized memory) to proper initialization
        using learnable buffers. This ensures deterministic behavior and reproducibility.

        Note: This is kept for backward compatibility. Use empty_carry_with_puzzle() for
        puzzle-aware c_H initialization.
        """
        c_H = None
        if self.config.use_lstm_gating:
            c_H = self.C_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone()

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self.H_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone(),
            z_L=self.L_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone(),
            c_H=c_H,
        )

    def empty_carry_with_puzzle(self, puzzle_identifiers: torch.Tensor):
        """Initialize carry state with puzzle-aware c_H initialization.

        Uses puzzle_emb to generate initial c_H state specific to each puzzle.
        This allows the model to start with puzzle-specific context without
        maintaining a large cache of states.

        Args:
            puzzle_identifiers: (batch_size,) tensor of puzzle IDs

        Returns:
            InnerCarry with puzzle-aware c_H initialization
        """
        batch_size = puzzle_identifiers.shape[0]

        # Initialize z_H and z_L as usual
        z_H = self.H_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone()
        z_L = self.L_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone()

        # Initialize c_H with puzzle-aware state
        c_H = None
        if self.config.use_lstm_gating:
            if self.config.puzzle_emb_ndim > 0 and hasattr(self, 'puzzle_emb_to_c_H'):
                # Puzzle-aware initialization: puzzle_emb → c_H
                puzzle_embs = self.puzzle_emb(puzzle_identifiers)  # (B, emb_ndim), BF16

                # Convert to FP32 to match puzzle_emb_to_c_H dtype
                puzzle_embs_fp32 = puzzle_embs.to(torch.float32)
                c_H_init = self.puzzle_emb_to_c_H(puzzle_embs_fp32)  # (B, hidden_size), FP32

                # Expand to sequence length: (B, seq_len, hidden_size)
                c_H = c_H_init.unsqueeze(1).expand(
                    -1, self.config.seq_len + self.puzzle_emb_len, -1
                ).contiguous().clone()
            else:
                # Fallback to C_init if puzzle_emb not available
                c_H = self.C_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1).clone()

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H,
            z_L=z_L,
            c_H=c_H,
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, puzzle_identifiers: Optional[torch.Tensor] = None):
        """Reset carry state for halted sequences.

        Args:
            reset_flag: Boolean tensor indicating which sequences to reset
            carry: Current carry state
            puzzle_identifiers: Optional puzzle IDs for puzzle-aware c_H initialization
        """
        c_H = None
        if self.config.use_lstm_gating:
            if puzzle_identifiers is not None and self.config.puzzle_emb_ndim > 0 and hasattr(self, 'puzzle_emb_to_c_H'):
                # Puzzle-aware reset: use puzzle_emb for reset sequences
                puzzle_embs = self.puzzle_emb(puzzle_identifiers)  # (B, emb_ndim), BF16
                puzzle_embs_fp32 = puzzle_embs.to(torch.float32)
                c_H_reset = self.puzzle_emb_to_c_H(puzzle_embs_fp32)  # (B, hidden_size), FP32

                # Expand to sequence length
                c_H_reset_expanded = c_H_reset.unsqueeze(1).expand(
                    -1, self.config.seq_len + self.puzzle_emb_len, -1
                )

                # Use puzzle-aware c_H for reset, keep old c_H for non-reset
                c_H = torch.where(reset_flag.view(-1, 1, 1), c_H_reset_expanded, carry.c_H)
            else:
                # Fallback: use C_init
                c_H = torch.where(reset_flag.view(-1, 1, 1), self.C_init, carry.c_H)

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            c_H=c_H,
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        c_H = carry.c_H if self.config.use_lstm_gating else None

        # Fixed: Separate gradient strategies for LSTM vs baseline
        if self.config.use_lstm_gating:
            # LSTM path: All H-cycles WITH gradients (required for LSTM learning)
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H, c_H = self.lstm_context(c_H, z_H, z_L)
        else:
            # Baseline path: H_cycles-1 without grad, 1 with grad (original optimization)
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles-1):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

            # Final H-cycle with gradients
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        # Fixed: Detach z_H/z_L but NOT c_H to allow LSTM BPTT across ACT steps
        # - z_H/z_L: Detached to prevent baseline path graph accumulation
        # - c_H: NOT detached - LSTM needs gradients across ACT steps for long-term memory
        # - This allows LSTM to learn how cell state should evolve across puzzle-solving steps
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            c_H=c_H if self.config.use_lstm_gating else None  # NO detach for LSTM BPTT!
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        # FIXED: For pretrain, always use C_init (learnable constant) for c_H initialization
        # This allows LSTM to learn from scratch without puzzle-specific bias
        # puzzle_emb is still used in input embeddings (as in original TRM)
        inner_carry = self.inner.empty_carry(batch_size)

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,  # Uses C_init for c_H (puzzle-agnostic)

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        # FIXED: Don't pass puzzle_identifiers - use C_init for reset (puzzle-agnostic)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry, puzzle_identifiers=None)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            # Preserve input halted state (critical for forced halting in training)
            # Once a sample halts, it should stay halted for this batch
            halted = carry.halted | is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
