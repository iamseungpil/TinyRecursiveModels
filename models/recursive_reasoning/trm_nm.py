"""
TRM-NM: Tiny Recursive Model with Neural Memory (Titans-style)

Key changes from TRM:
1. Added Neural Memory module that learns K->V associations
2. Memory is updated via gradient descent (surprise-based)
3. Test-time: only memory + puzzle_emb are learned, rest is frozen
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from contextlib import nullcontext
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import einops

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from torch.nn.functional import scaled_dot_product_attention

IGNORE_LABEL_ID = -100


# =============================================================================
# Neural Memory Module (Titans-style)
# =============================================================================

class NeuralMemory(nn.Module):
    """
    Neural Memory that stores K->V associations in MLP weights.

    Unlike attention which stores K,V as activations,
    Neural Memory stores associations in learnable weights
    and updates them via gradient descent.

    Key insight from Titans: Memory weights are updated in-place during forward pass.
    For meta-learning (create_graph=True), we track "persistent state" tensors
    that maintain gradient flow.

    IMPORTANT: Memory is batch-aware - each sample in the batch has its own memory state.
    This is achieved by storing per-batch weights with shape [B, out_dim, in_dim].
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dtype = dtype

        # Memory MLP (2-layer) - these are the "template" weights
        self.memory_up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.memory_down = nn.Linear(hidden_dim, output_dim, bias=False)

        # Learnable hyperparameters for memory update
        self.mem_lr = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.mem_decay = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))

        # Current state weights (for in-flight updates with grad tracking)
        # Shape: [B, out_dim, in_dim] for batch-aware memory
        # These are NOT nn.Parameters - they're tensors that get updated each step
        self._current_up_weight: Optional[torch.Tensor] = None  # [B, hidden_dim, input_dim]
        self._current_down_weight: Optional[torch.Tensor] = None  # [B, output_dim, hidden_dim]
        self._batch_size: int = 0

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.memory_up.weight, std=0.02)
        nn.init.normal_(self.memory_down.weight, std=0.02)

    def reset(self, batch_size: int = None):
        """Reset memory to initial (learned) state for all samples in batch.

        Args:
            batch_size: Number of samples in batch. If None, uses last known batch size.
        """
        if batch_size is not None:
            self._batch_size = batch_size

        if self._batch_size == 0:
            # Fallback for non-batch case
            self._current_up_weight = self.memory_up.weight.clone()
            self._current_down_weight = self.memory_down.weight.clone()
        else:
            # Batch-aware: expand template weights to [B, out_dim, in_dim]
            # memory_up.weight: [hidden_dim, input_dim]
            # memory_down.weight: [output_dim, hidden_dim]
            self._current_up_weight = self.memory_up.weight.unsqueeze(0).expand(
                self._batch_size, -1, -1
            ).clone()
            self._current_down_weight = self.memory_down.weight.unsqueeze(0).expand(
                self._batch_size, -1, -1
            ).clone()

    def reset_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory only for samples where reset_mask is True.

        Args:
            reset_mask: Boolean tensor [B] indicating which samples to reset.
        """
        if self._current_up_weight is None:
            # No memory state yet, initialize with the batch size from mask
            self.reset(batch_size=reset_mask.shape[0])
            return

        # Ensure batch sizes match
        if self._current_up_weight.dim() == 2:
            # Non-batch weights, expand to batch
            self._batch_size = reset_mask.shape[0]
            self._current_up_weight = self._current_up_weight.unsqueeze(0).expand(
                self._batch_size, -1, -1
            ).clone()
            self._current_down_weight = self._current_down_weight.unsqueeze(0).expand(
                self._batch_size, -1, -1
            ).clone()

        # Get template weights expanded to match batch
        template_up = self.memory_up.weight.unsqueeze(0).expand_as(self._current_up_weight)
        template_down = self.memory_down.weight.unsqueeze(0).expand_as(self._current_down_weight)

        # Selectively reset only halted samples
        # reset_mask: [B] -> [B, 1, 1] for broadcasting
        mask = reset_mask.view(-1, 1, 1)
        self._current_up_weight = torch.where(mask, template_up, self._current_up_weight)
        self._current_down_weight = torch.where(mask, template_down, self._current_down_weight)

    def _get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current weights (either template or in-flight state)"""
        if self._current_up_weight is None:
            return self.memory_up.weight, self.memory_down.weight
        return self._current_up_weight, self._current_down_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through memory MLP using current state weights.

        Handles both batch-aware weights [B, out_dim, in_dim] and
        non-batch weights [out_dim, in_dim].
        """
        up_w, down_w = self._get_weights()

        if up_w.dim() == 3:
            # Batch-aware weights: up_w [B, hidden_dim, input_dim], x [B, L, input_dim]
            # Use einsum for batch matrix multiplication: out[b,l,h] = sum_d x[b,l,d] * up_w[b,h,d]
            h = F.silu(torch.einsum('bld,bhd->blh', x.to(up_w.dtype), up_w))
            # down_w [B, output_dim, hidden_dim], h [B, L, hidden_dim]
            out = torch.einsum('blh,boh->blo', h, down_w)
        else:
            # Non-batch weights: use standard F.linear
            h = F.silu(F.linear(x.to(up_w.dtype), up_w))
            out = F.linear(h, down_w)

        return out.to(x.dtype)

    def compute_surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute surprise = prediction error

        Args:
            k: Keys [B, L, D] - what to query
            v: Values [B, L, D] - expected output

        Returns:
            surprise: scalar loss
        """
        pred = self.forward(k)
        surprise = (pred - v.to(pred.dtype)).pow(2).mean()
        return surprise

    def update_memory(self, k: torch.Tensor, v: torch.Tensor,
                      create_graph: bool = False) -> torch.Tensor:
        """
        Update memory weights based on surprise.

        When create_graph=True, the weight updates maintain gradient flow
        for meta-learning (learning to learn good initial memory weights).

        IMPORTANT FIX (Issue 2 - Gradient Flow Bug):
        - When create_graph=True, we detach the updated weights to prevent
          non-leaf tensor issues during backpropagation.
        - The surprise loss is computed fresh each time and returned for
          the outer optimization loop.
        - This prevents weight tensors from becoming non-leaf while still
          allowing gradient flow through the surprise loss.

        Args:
            k: Keys [B, L, D]
            v: Values [B, L, D]
            create_graph: Whether to create graph for meta-learning

        Returns:
            surprise: the computed surprise (for logging/loss)
        """
        batch_size = k.shape[0]

        # Ensure we have current state weights
        if self._current_up_weight is None:
            self.reset(batch_size=batch_size)

        up_w, down_w = self._get_weights()

        # Ensure batch dimensions match
        if up_w.dim() == 2:
            # Expand to batch if needed
            self._batch_size = batch_size
            up_w = up_w.unsqueeze(0).expand(batch_size, -1, -1).clone()
            down_w = down_w.unsqueeze(0).expand(batch_size, -1, -1).clone()
            self._current_up_weight = up_w
            self._current_down_weight = down_w

        # Compute surprise with current weights (batch-aware)
        # up_w: [B, hidden_dim, input_dim], k: [B, L, input_dim]
        h = F.silu(torch.einsum('bld,bhd->blh', k.to(up_w.dtype), up_w))
        # down_w: [B, output_dim, hidden_dim], h: [B, L, hidden_dim]
        pred = torch.einsum('blh,boh->blo', h, down_w)
        surprise = (pred - v.to(pred.dtype)).pow(2).mean()

        # Compute gradients w.r.t. current state weights
        # CRITICAL FIX: Only compute gradients if weights require grad
        if up_w.requires_grad or down_w.requires_grad:
            grads = torch.autograd.grad(
                surprise,
                [up_w, down_w],
                create_graph=create_graph,
                retain_graph=True  # Keep graph for the surprise loss
            )
            grad_up, grad_down = grads
        else:
            # Weights don't require grad (e.g., during no_grad context)
            # Skip gradient computation
            return surprise.detach()

        # Learnable hyperparams
        lr = self.mem_lr.abs()
        decay = self.mem_decay.abs().clamp(0, 0.1)

        # Update weights
        # CRITICAL FIX (Issue 2): When create_graph=True, we need to be careful
        # about the computation graph. The updated weights should be detached
        # to prevent them from becoming non-leaf tensors that cause issues
        # in subsequent backward passes. The surprise loss itself carries
        # the gradient information we need.
        new_up_w = (1 - decay) * up_w - lr * grad_up
        new_down_w = (1 - decay) * down_w - lr * grad_down

        if create_graph:
            # Detach updated weights to prevent non-leaf tensor issues
            # The gradient flow for meta-learning goes through the surprise loss
            self._current_up_weight = new_up_w.detach().requires_grad_(True)
            self._current_down_weight = new_down_w.detach().requires_grad_(True)
        else:
            self._current_up_weight = new_up_w.detach()
            self._current_down_weight = new_down_w.detach()

        return surprise


# =============================================================================
# Attention with Memory
# =============================================================================

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class AttentionWithMemory(nn.Module):
    """
    Attention module with integrated Neural Memory.

    Following Titans design:
    - Memory uses hidden_states directly (not attention K/V)
    - Memory learns to map hidden_states -> hidden_states
    - This is simpler and maintains hidden_size throughout
    """

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int,
                 num_key_value_heads: int, memory_hidden_mult: int = 4,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        # QKV projection (same as original TRM)
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

        # Neural Memory - uses hidden_states directly
        self.memory = NeuralMemory(
            input_dim=hidden_size,
            hidden_dim=hidden_size * memory_hidden_mult,
            output_dim=hidden_size,
            dtype=dtype
        )

        # Learnable gate to balance attention vs memory
        self.mem_gate = nn.Parameter(torch.zeros(1))

    def reset_memory(self, batch_size: int = None):
        """Reset memory state for all samples."""
        self.memory.reset(batch_size=batch_size)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory only for samples where reset_mask is True."""
        self.memory.reset_for_samples(reset_mask)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor,
                update_memory: bool = True,
                create_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention and memory.

        Args:
            cos_sin: Rotary position embeddings
            hidden_states: Input tensor [B, L, D]
            update_memory: Whether to update memory weights
            create_graph: Whether to create graph for meta-learning

        Returns:
            output: Combined attention + memory output [B, L, D]
            surprise: Memory surprise loss (scalar)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Apply RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # === Attention ===
        query_attn = einops.rearrange(query, 'B S H D -> B H S D')
        key_attn = einops.rearrange(key, 'B S H D -> B H S D')
        value_attn = einops.rearrange(value, 'B S H D -> B H S D')

        attn_output = scaled_dot_product_attention(
            query=query_attn, key=key_attn, value=value_attn, is_causal=False
        )
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S (H D)')
        attn_output = self.o_proj(attn_output)

        # === Neural Memory ===
        # Use hidden_states directly for memory (following Titans)
        # Memory learns: hidden_states -> hidden_states transformation
        surprise = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        if update_memory:
            # Update memory: learn to predict hidden_states from hidden_states
            # This captures patterns in the input sequence
            surprise = self.memory.update_memory(hidden_states, hidden_states, create_graph=create_graph)

        # Memory retrieval
        mem_output = self.memory(hidden_states)

        # Gated combination of attention and memory
        gate = torch.sigmoid(self.mem_gate)
        output = attn_output + gate * mem_output

        return output, surprise


# =============================================================================
# TRM-NM Block
# =============================================================================

class TRM_NM_Block(nn.Module):
    """TRM Block with Neural Memory"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Attention with Memory
        self.self_attn = AttentionWithMemory(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            memory_hidden_mult=4,
            dtype=getattr(torch, config.forward_dtype)
        )

        # MLP (same as original)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def reset_memory(self, batch_size: int = None):
        """Reset memory state for all samples."""
        self.self_attn.reset_memory(batch_size=batch_size)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory only for samples where reset_mask is True."""
        self.self_attn.reset_memory_for_samples(reset_mask)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor,
                update_memory: bool = True,
                create_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            hidden_states: Output tensor
            surprise: Memory surprise loss
        """
        # Attention + Memory
        attn_out, surprise = self.self_attn(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            update_memory=update_memory,
            create_graph=create_graph
        )
        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # MLP
        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)

        return hidden_states, surprise


# =============================================================================
# TRM-NM Reasoning Module
# =============================================================================

class TRM_NM_ReasoningModule(nn.Module):
    """Reasoning module with multiple TRM-NM blocks"""

    def __init__(self, layers: List[TRM_NM_Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def reset_memory(self, batch_size: int = None):
        """Reset memory in all layers for all samples."""
        for layer in self.layers:
            layer.reset_memory(batch_size=batch_size)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory in all layers only for samples where reset_mask is True."""
        for layer in self.layers:
            layer.reset_memory_for_samples(reset_mask)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor,
                update_memory: bool = True, create_graph: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all layers.

        Returns:
            hidden_states: Output tensor
            total_surprise: Sum of surprise from all layers
        """
        hidden_states = hidden_states + input_injection
        total_surprise = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

        for layer in self.layers:
            hidden_states, surprise = layer(
                hidden_states=hidden_states,
                update_memory=update_memory,
                create_graph=create_graph,
                **kwargs
            )
            total_surprise = total_surprise + surprise

        return hidden_states, total_surprise


# =============================================================================
# TRM-NM Config
# =============================================================================

class TRM_NM_Config(BaseModel):
    """Configuration for TRM-NM model"""
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
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # Memory specific
    memory_hidden_mult: int = 4
    surprise_loss_weight: float = 0.1


# =============================================================================
# TRM-NM Carry (State)
# =============================================================================

@dataclass
class TRM_NM_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRM_NM_Carry:
    inner_carry: TRM_NM_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# =============================================================================
# TRM-NM Inner Model
# =============================================================================

class TRM_NM_Inner(nn.Module):
    """Inner model for TRM-NM"""

    def __init__(self, config: TRM_NM_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O Embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embeddings
        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )

        # Reasoning Layers with Memory
        self.L_level = TRM_NM_ReasoningModule(
            layers=[TRM_NM_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Q head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def reset_memory(self, batch_size: int = None):
        """Reset all memory states for all samples."""
        self.L_level.reset_memory(batch_size=batch_size)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory only for samples where reset_mask is True."""
        self.L_level.reset_memory_for_samples(reset_mask)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings with puzzle context"""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = None):
        device = device or self.H_init.device
        return TRM_NM_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRM_NM_InnerCarry):
        return TRM_NM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TRM_NM_InnerCarry, batch: Dict[str, torch.Tensor],
                update_memory: bool = True,
                create_graph: bool = False) -> Tuple[TRM_NM_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Returns:
            new_carry: Updated carry state
            output: Logits
            (q_halt_logits, q_continue_logits): Halting logits
            total_surprise: Sum of surprise losses

        CRITICAL FIX (Issue 3 - Computation Graph Accumulation):
        - Previously: H_cycles iterations accumulated computation graphs through
          memory updates, causing OOM even with torch.no_grad() on hidden states
        - Now: Detach z_H and z_L between H_cycles to prevent graph accumulation
        - Only the final cycle maintains gradient flow for backpropagation
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        total_surprise = torch.tensor(0.0, device=z_H.device, dtype=torch.float32)

        # Determine if we should create graph (only for last cycle during training)
        should_create_graph = create_graph and self.training

        # H_cycles-1 without grad (for efficiency)
        # CRITICAL FIX: Use torch.no_grad() AND detach between cycles to prevent
        # any gradient accumulation through memory updates
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L, surprise = self.L_level(
                        z_L, z_H + input_embeddings,
                        update_memory=update_memory,
                        create_graph=False,
                        **seq_info
                    )
                z_H, surprise = self.L_level(z_H, z_L, update_memory=update_memory, create_graph=False, **seq_info)

                # CRITICAL FIX (Issue 3): Detach between H_cycles to prevent graph accumulation
                # This ensures memory updates don't chain across cycles
                z_H = z_H.detach()
                z_L = z_L.detach()

        # Last cycle with grad
        for _L_step in range(self.config.L_cycles):
            z_L, surprise = self.L_level(
                z_L, z_H + input_embeddings,
                update_memory=update_memory,
                create_graph=should_create_graph,
                **seq_info
            )
            total_surprise = total_surprise + surprise

            # CRITICAL FIX (Issue 3): Detach z_L between L_cycles to prevent
            # accumulation within the last H_cycle as well
            if not should_create_graph:
                z_L = z_L.detach()

        z_H, surprise = self.L_level(z_H, z_L, update_memory=update_memory, create_graph=should_create_graph, **seq_info)
        total_surprise = total_surprise + surprise

        # Outputs
        new_carry = TRM_NM_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), total_surprise


# =============================================================================
# TRM-NM Main Model (with ACT wrapper)
# =============================================================================

class TRM_NM(nn.Module):
    """TRM with Neural Memory - ACT wrapper"""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_NM_Config(**config_dict)
        self.inner = TRM_NM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def reset_memory(self, batch_size: int = None):
        """Reset all memory states for all samples."""
        self.inner.reset_memory(batch_size=batch_size)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory only for samples where reset_mask is True."""
        self.inner.reset_memory_for_samples(reset_mask)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRM_NM_Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TRM_NM_Carry, batch: Dict[str, torch.Tensor],
                update_memory: bool = True,
                create_graph: bool = False) -> Tuple[TRM_NM_Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT.

        Returns:
            new_carry: Updated carry
            outputs: Dict with logits, q_logits, surprise

        CRITICAL FIX (Issue 1 - Memory State Isolation):
        - Previously: if ANY sample halted, ALL memory was reset
        - Now: Only reset memory for the specific samples that halted
        - This ensures each puzzle has isolated memory context
        """
        # FIXED: Selectively reset memory only for samples that just halted
        # This prevents interference between samples in the same batch
        if carry.halted.any():
            self.inner.reset_memory_for_samples(carry.halted)

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), surprise = self.inner(
            new_inner_carry, new_current_data,
            update_memory=update_memory,
            create_graph=create_graph
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "surprise": surprise,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(
                        new_inner_carry, new_current_data,
                        update_memory=False,
                        create_graph=False
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits,
                                   torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    )

        return TRM_NM_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


# =============================================================================
# Test-time Learning Utilities
# =============================================================================

class TRM_NM_TestTime:
    """
    Utilities for test-time learning.

    At test-time, only memory and puzzle_emb are learned.
    The rest of the model is frozen.
    """

    def __init__(self, model: TRM_NM, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device

        # Freeze all parameters except memory and puzzle_emb
        self._freeze_pretrained()

    def _freeze_pretrained(self):
        """Freeze all parameters except memory and puzzle_emb"""
        for name, param in self.model.named_parameters():
            # Keep trainable: memory weights, puzzle_emb, memory hyperparams, and memory gate
            if 'memory' in name or 'puzzle_emb' in name or 'mem_lr' in name or 'mem_decay' in name or 'mem_gate' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_learnable_params(self):
        """Get parameters that are learned at test-time"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def create_test_puzzle_emb(self, batch_size: int) -> torch.Tensor:
        """
        Create a new puzzle embedding for unseen puzzles.

        Returns:
            puzzle_emb: Learnable tensor for this puzzle
        """
        emb_dim = self.model.config.puzzle_emb_ndim
        puzzle_emb = torch.zeros(batch_size, emb_dim, device=self.device, requires_grad=True)
        return puzzle_emb

    def test_time_adapt(self, demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                        n_steps: int = 5, lr: float = 0.01):
        """
        Adapt model to a new puzzle using demo pairs.

        Args:
            demo_pairs: List of (demo_input, demo_output) tensors
            n_steps: Number of adaptation steps
            lr: Learning rate for adaptation
        """
        # Reset memory
        self.model.reset_memory()

        # Create optimizer for learnable params
        optimizer = torch.optim.Adam(self.get_learnable_params(), lr=lr)

        self.model.train()

        for step in range(n_steps):
            total_loss = 0

            for demo_x, demo_y in demo_pairs:
                # Ensure tensors are on correct device
                demo_x = demo_x.to(self.device)
                demo_y = demo_y.to(self.device)

                # Create batch
                batch = {
                    "inputs": demo_x,
                    "labels": demo_y,
                    "puzzle_identifiers": torch.zeros(demo_x.shape[0], dtype=torch.long, device=self.device)
                }

                # Forward pass (initial_carry now handles device correctly)
                carry = self.model.initial_carry(batch)

                carry, outputs = self.model(carry, batch, update_memory=True, create_graph=True)

                # Compute loss
                logits = outputs["logits"]
                labels = demo_y[:, :logits.shape[1]]  # Align shapes

                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=IGNORE_LABEL_ID
                )
                loss = loss + self.model.config.surprise_loss_weight * outputs["surprise"]

                total_loss = total_loss + loss

            # Backward and update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        self.model.eval()

    def predict(self, test_input: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on test input after adaptation.

        Args:
            test_input: Test input tensor

        Returns:
            predictions: Predicted output tokens
        """
        self.model.eval()

        with torch.no_grad():
            # Ensure input is on correct device
            test_input = test_input.to(self.device)

            batch = {
                "inputs": test_input,
                "labels": torch.zeros_like(test_input),
                "puzzle_identifiers": torch.zeros(test_input.shape[0], dtype=torch.long, device=self.device)
            }

            # initial_carry now handles device correctly
            carry = self.model.initial_carry(batch)

            # Run full ACT loop
            for _ in range(self.model.config.halt_max_steps):
                carry, outputs = self.model(carry, batch, update_memory=False, create_graph=False)
                if carry.halted.all():
                    break

            predictions = outputs["logits"].argmax(dim=-1)

        return predictions


# =============================================================================
# ACT Loss Head for TRM-NM
# =============================================================================

class TRM_NM_ACTLossHead(nn.Module):
    """
    ACT Loss Head for TRM-NM that includes surprise loss.

    Extends the base ACTLossHead to handle surprise loss from neural memory.
    """

    def __init__(self, model: TRM_NM, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type] if loss_type in globals() else self._get_loss_fn(loss_type)

    def _get_loss_fn(self, loss_type: str):
        """Import loss function from losses module."""
        from models.losses import stablemax_cross_entropy, softmax_cross_entropy
        return {"stablemax_cross_entropy": stablemax_cross_entropy,
                "softmax_cross_entropy": softmax_cross_entropy}[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys,
        carry: TRM_NM_Carry,
        batch: Dict[str, torch.Tensor],
    ):
        """
        Forward pass with loss computation.

        Returns:
            new_carry: Updated carry state
            loss: Total loss (lm_loss + q_halt_loss + surprise_loss)
            metrics: Dict of metrics
            detached_outputs: Outputs for evaluation
            all_halted: Whether all sequences have halted
        """
        # Forward model
        new_carry, outputs = self.model(carry, batch, update_memory=True, create_graph=self.training)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum"
        )

        # Surprise loss (from neural memory)
        surprise_loss = outputs.get("surprise", torch.tensor(0.0, device=lm_loss.device))
        surprise_weight = self.model.config.surprise_loss_weight

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "surprise_loss": surprise_loss.detach() if isinstance(surprise_loss, torch.Tensor) else surprise_loss,
        })

        # Q continue loss
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Total loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + surprise_weight * surprise_loss

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


# Loss functions (for standalone use)
def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    """Stablemax cross entropy loss."""
    def s(x, epsilon=1e-30):
        return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)

    def log_stablemax(x, dim=-1):
        s_x = s(x)
        return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    """Softmax cross entropy loss."""
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none"
    ).view(labels.shape)
