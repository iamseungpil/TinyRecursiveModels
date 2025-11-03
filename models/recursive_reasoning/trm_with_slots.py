"""
Tiny Recursive Reasoning Model with Slot Attention.

Extends TRM with:
1. Slot Attention for compositional decomposition
2. Dual prediction heads (direct + slot-based)
3. Support for slot-level contrastive learning
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.slot_attention import SlotAttention, SlotDecoder

IGNORE_LABEL_ID = -100


@dataclass
class TRM_WithSlots_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    slots: Optional[torch.Tensor] = None  # Store slots for contrastive learning


@dataclass
class TRM_WithSlots_Carry:
    inner_carry: TRM_WithSlots_InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TRM_WithSlots_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
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

    # TRM specific
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # Slot Attention config
    num_slots: int = 8
    slot_dim: int = 256
    slot_iterations: int = 3
    use_slot_decoder: bool = True


class TRM_WithSlots_Block(nn.Module):
    def __init__(self, config: TRM_WithSlots_Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
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
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRM_WithSlots_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRM_WithSlots_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRM_WithSlots_Inner(nn.Module):
    def __init__(self, config: TRM_WithSlots_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Dual prediction heads
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)  # Direct from z_H
        self.lm_head_slots = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)  # From slots

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
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
        self.L_level = TRM_WithSlots_ReasoningModule(layers=[TRM_WithSlots_Block(self.config) for _i in range(self.config.L_layers)])

        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=self.config.num_slots,
            slot_dim=self.config.slot_dim,
            input_dim=self.config.hidden_size,
            num_iterations=self.config.slot_iterations
        )

        # Slot Decoder
        if self.config.use_slot_decoder:
            self.slot_decoder = SlotDecoder(
                slot_dim=self.config.slot_dim,
                output_dim=self.config.hidden_size,
                broadcast_size=self.config.seq_len  # Exclude puzzle_emb_len
            )
        else:
            # Projection for slot aggregation when decoder is disabled
            if self.config.slot_dim != self.config.hidden_size:
                self.slot_proj = nn.Linear(self.config.slot_dim, self.config.hidden_size)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

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
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = None):
        """Create empty carry state."""
        device = device if device is not None else torch.device('cpu')
        return TRM_WithSlots_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            slots=None
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRM_WithSlots_InnerCarry):
        return TRM_WithSlots_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            slots=None  # Reset slots
        )

    def forward(self, carry: TRM_WithSlots_InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_WithSlots_InnerCarry, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations (TRM reasoning)
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Slot Attention decomposition
        slots = self.slot_attention(z_H)  # [B, num_slots, slot_dim]

        # Direct prediction from z_H
        output_direct = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Slot-based prediction
        if self.config.use_slot_decoder:
            slot_features = self.slot_decoder(slots)  # [B, seq_len, hidden_size]
            output_slots = self.lm_head_slots(slot_features)
        else:
            # Simple aggregation: mean over slots
            slot_agg = slots.mean(dim=1, keepdim=True).expand(-1, self.config.seq_len, -1)
            # Project to hidden_size if needed
            if hasattr(self, 'slot_proj'):
                slot_agg = self.slot_proj(slot_agg)
            output_slots = self.lm_head_slots(slot_agg)

        # Q-head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # New carry with slots
        new_carry = TRM_WithSlots_InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            slots=slots.detach()
        )

        return new_carry, output_direct, output_slots, slots, (q_logits[..., 0], q_logits[..., 1])


class TRM_WithSlots_ACTV1(nn.Module):
    """ACT wrapper for TRM with Slots."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_WithSlots_Config(**config_dict)
        self.inner = TRM_WithSlots_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRM_WithSlots_Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),

            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TRM_WithSlots_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_WithSlots_Carry, Dict[str, torch.Tensor]]:
        # Update data, carry
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, output_direct, output_slots, slots, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": output_direct,
            "logits_slots": output_slots,
            "slots": slots,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    _, _, _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TRM_WithSlots_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
