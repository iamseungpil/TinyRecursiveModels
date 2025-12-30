"""
TRM-Titans: Tiny Recursive Model with Titans-style Implicit Memory

This module implements the Titans architecture where:
- Memory weights ARE the state (no explicit z_H, z_L tensors)
- Two MLP memories replace z_H and z_L roles
- Memory updates happen via gradient descent on prediction error

=== TITANS CORE CONCEPTS ===
1. Implicit State: State is stored in MLP weights, not in activation tensors
2. Memory Update: M_t = (1 - alpha) * M_{t-1} - eta * gradient(loss(M; x))
3. K->V Mapping: Memory learns input->output associations
4. MAG (Memory as Gate): Combines memory output with attention via learned gate

=== KEY DIFFERENCE FROM TRM-NM ===
- TRM-NM: Uses z_H, z_L tensors as explicit state + memory for pattern learning
- TRM-Titans: NO z_H, z_L tensors - memory weights ARE the implicit state

=== ARCHITECTURE ===
- memory_H: High-level reasoning memory (replaces z_H)
- memory_L: Low-level computation memory (replaces z_L)
- Each memory is a 2-layer MLP with batch-aware per-sample weights
- H_cycles: Number of high-level memory updates
- L_cycles: Number of low-level memory updates per H_cycle
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from contextlib import nullcontext
import math
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from pydantic import BaseModel
import einops


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def _is_distributed() -> bool:
    """Check if we are in distributed training mode with multiple GPUs."""
    return dist.is_initialized() and dist.get_world_size() > 1


def _sync_if_distributed():
    """
    Synchronize CUDA streams if in distributed mode.

    This prevents GPU desync when using operations that may complete at
    different times across GPUs (like autograd.grad with varying tensor sizes).
    """
    if _is_distributed():
        torch.cuda.synchronize()

# Disable donated buffer for torch.compile compatibility with autograd.grad
# This is required for Titans-style memory update which uses torch.autograd.grad
# Note: We now use retain_graph=False for distributed training stability
try:
    import torch._functorch.config
    torch._functorch.config.donated_buffer = False
except (ImportError, AttributeError):
    pass

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from torch.nn.functional import scaled_dot_product_attention

IGNORE_LABEL_ID = -100


# =============================================================================
# Titans Memory Module (Implicit State in Weights)
# =============================================================================

class TitansMemory(nn.Module):
    """
    Titans-style memory where weights ARE the state.

    Unlike attention which stores K,V as activations, or TRM which uses z tensors,
    TitansMemory stores all state information in MLP weights themselves.

    Key properties:
    - Per-sample batch-aware weights: [B, out_dim, in_dim]
    - Weights updated via gradient descent on prediction error
    - No explicit state tensors - weights encode everything

    Memory Update Equation (from Titans paper):
        M_t = (1 - alpha) * M_{t-1} - eta * grad(||M(k) - v||^2)
    where:
        - alpha: decay rate (forgetting factor)
        - eta: learning rate
        - k: key (input)
        - v: value (target)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dtype = dtype

        # Template weights (learned initial state)
        # These are the "default" weights that memory resets to
        self.template_up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.template_down = nn.Linear(hidden_dim, output_dim, bias=False)

        # Memory update hyperparameters (stored in log-space)
        # NOTE: These are nn.Parameters for convenience but do NOT receive gradients
        # in the current design (we return pre-update surprise, not post-update).
        # exp(-4.6) ~= 0.01 (learning rate), exp(-6.9) ~= 0.001 (decay)
        self.mem_lr = nn.Parameter(torch.tensor(-4.6, dtype=torch.float32))
        self.mem_decay = nn.Parameter(torch.tensor(-6.9, dtype=torch.float32))

        # Current state weights (per-batch, updated during forward)
        # Shape: [B, out_dim, in_dim]
        self._current_up_weight: Optional[torch.Tensor] = None
        self._current_down_weight: Optional[torch.Tensor] = None
        self._batch_size: int = 0

        self._init_weights()

    def _init_weights(self):
        """Initialize template weights with small values for stable learning."""
        nn.init.normal_(self.template_up.weight, std=0.02)
        nn.init.normal_(self.template_down.weight, std=0.02)

    def reset(self, batch_size: int, device: torch.device = None):
        """
        Reset memory to template (initial) state for all samples.

        This creates fresh per-sample weights from the learned templates.
        """
        if device is None:
            device = self.template_up.weight.device

        self._batch_size = batch_size

        # Expand template weights to batch: [out, in] -> [B, out, in]
        self._current_up_weight = self.template_up.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone().to(device).requires_grad_(True)

        self._current_down_weight = self.template_down.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone().to(device).requires_grad_(True)

    def reset_for_samples(self, reset_mask: torch.Tensor):
        """
        Selectively reset memory only for samples where reset_mask is True.

        Uses gradient-preserving mask arithmetic to maintain gradient flow
        for non-reset samples.
        """
        # Get device from reset_mask for multi-GPU compatibility
        mask_device = reset_mask.device if reset_mask is not None else None

        if self._current_up_weight is None:
            self.reset(batch_size=reset_mask.shape[0], device=mask_device)
            return

        device = self._current_up_weight.device
        reset_mask = reset_mask.to(device)

        # Ensure batch dimensions match
        if self._current_up_weight.dim() == 2:
            self.reset(batch_size=reset_mask.shape[0], device=device)
            return

        # Get template weights expanded to batch
        template_up = self.template_up.weight.unsqueeze(0).expand_as(self._current_up_weight)
        template_down = self.template_down.weight.unsqueeze(0).expand_as(self._current_down_weight)

        # Gradient-preserving update using mask arithmetic
        mask_float = reset_mask.view(-1, 1, 1).to(self._current_up_weight.dtype)
        keep_mask = 1.0 - mask_float

        new_up = keep_mask * self._current_up_weight + mask_float * template_up
        new_down = keep_mask * self._current_down_weight + mask_float * template_down

        if torch.is_grad_enabled():
            self._current_up_weight = new_up
            self._current_down_weight = new_down
        else:
            self._current_up_weight = new_up.detach().requires_grad_(True)
            self._current_down_weight = new_down.detach().requires_grad_(True)

    def _get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current state weights."""
        if self._current_up_weight is None:
            return self.template_up.weight, self.template_down.weight
        return self._current_up_weight, self._current_down_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through memory using current state weights.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            output: Memory output [B, L, output_dim]
        """
        up_w, down_w = self._get_weights()

        if up_w.dim() == 3:
            # Batch-aware: [B, hidden, input] x [B, L, input] -> [B, L, hidden]
            h = F.silu(torch.einsum('bld,bhd->blh', x.to(up_w.dtype), up_w))
            # [B, L, hidden] x [B, output, hidden] -> [B, L, output]
            out = torch.einsum('blh,boh->blo', h, down_w)
        else:
            # Non-batch: use standard linear
            h = F.silu(F.linear(x.to(up_w.dtype), up_w))
            out = F.linear(h, down_w)

        return out.to(x.dtype)

    @torch.compiler.disable  # Disable compilation for this method (uses autograd.grad)
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        create_graph: bool = False
    ) -> torch.Tensor:
        """
        Update memory weights based on prediction error (Titans-style).

        Implements: M_t = (1 - alpha) * M_{t-1} - eta * grad(||M(k) - v||^2)

        Args:
            k: Key (input) [B, L, D]
            v: Value (target) [B, L, D]
            create_graph: Whether to create computation graph for meta-learning.
                          When True, gradients will flow to mem_lr and mem_decay.

        Returns:
            surprise: Mean squared prediction error (for logging/loss).
                      When create_graph=True, returns with gradient attached for backprop.
                      When create_graph=False, returns detached value.

        Note on gradient flow:
            - When create_graph=True: retain_graph=True to allow backprop through surprise
            - mem_lr and mem_decay receive gradients through the surprise loss
            - Weight updates are computed but weights are detached to prevent infinite graphs
        """
        batch_size = k.shape[0]
        device = k.device

        if batch_size == 0:
            raise ValueError("Cannot update memory with empty batch")

        # Ensure we have per-sample weights
        if self._current_up_weight is None:
            self.reset(batch_size=batch_size, device=device)

        up_w, down_w = self._get_weights()

        # Handle batch size mismatch
        if up_w.dim() == 2:
            self.reset(batch_size=batch_size, device=device)
            up_w, down_w = self._get_weights()
        elif up_w.shape[0] != batch_size:
            if batch_size < up_w.shape[0]:
                # FIX #3: Warn when batch size shrinks (may indicate a bug)
                warnings.warn(
                    f"TitansMemory: Batch size shrunk from {up_w.shape[0]} to {batch_size}. "
                    "Truncating memory weights. This may indicate a bug in batch handling.",
                    stacklevel=2
                )
                up_w = up_w[:batch_size].contiguous()
                down_w = down_w[:batch_size].contiguous()
                self._current_up_weight = up_w
                self._current_down_weight = down_w
                self._batch_size = batch_size
            else:
                # Expand with templates for new samples
                old_size = up_w.shape[0]
                new_count = batch_size - old_size
                template_up = self.template_up.weight.unsqueeze(0).expand(new_count, -1, -1).clone().to(device)
                template_down = self.template_down.weight.unsqueeze(0).expand(new_count, -1, -1).clone().to(device)
                up_w = torch.cat([up_w, template_up], dim=0)
                down_w = torch.cat([down_w, template_down], dim=0)
                self._current_up_weight = up_w
                self._current_down_weight = down_w
                self._batch_size = batch_size

        # Forward pass to compute prediction
        h = F.silu(torch.einsum('bld,bhd->blh', k.to(up_w.dtype), up_w))
        pred = torch.einsum('blh,boh->blo', h, down_w)

        # Compute per-sample surprise (prediction error)
        surprise_per_sample = (pred - v.to(pred.dtype)).pow(2).mean(dim=(1, 2))
        surprise = surprise_per_sample.mean()

        # Skip gradient computation if not enabled
        if not torch.is_grad_enabled():
            return surprise.detach()

        if not (up_w.requires_grad and down_w.requires_grad):
            return surprise.detach()

        # Compute gradients for memory update
        # When create_graph=True:
        #   - retain_graph=True: Keep graph for backprop through surprise to mem_lr/mem_decay
        #   - create_graph=True: Allow second-order gradients
        # When create_graph=False:
        #   - retain_graph=False: Free graph immediately for memory efficiency
        #   - create_graph=False: No second-order gradients
        grads = torch.autograd.grad(
            surprise,
            [up_w, down_w],
            create_graph=create_graph,
            retain_graph=create_graph  # Keep graph when we need gradients to flow
        )
        grad_up, grad_down = grads

        # Gradient clipping for stability
        # Note: When create_graph=True, we keep gradients flowing through clip operation
        if create_graph:
            grad_norm = torch.sqrt(grad_up.pow(2).sum() + grad_down.pow(2).sum() + 1e-8)
            max_norm = 1.0
            if grad_norm > max_norm:
                scale = max_norm / grad_norm
                grad_up = grad_up * scale
                grad_down = grad_down * scale

            # Get learning rate and decay from log-space parameters
            # These ops are differentiable so mem_lr and mem_decay get gradients
            lr = torch.exp(self.mem_lr).clamp(min=0.001, max=0.1)
            decay = torch.exp(self.mem_decay).clamp(min=0.0, max=0.1)

            # Titans update equation: M_t = (1 - alpha) * M_{t-1} - eta * grad
            new_up = (1 - decay) * up_w - lr * grad_up
            new_down = (1 - decay) * down_w - lr * grad_down

            # Store updated weights (detach to prevent infinite graph growth)
            # The memory update is a side effect; we don't need the lookahead surprise.
            self._current_up_weight = new_up.detach().requires_grad_(True)
            self._current_down_weight = new_down.detach().requires_grad_(True)

            # Return pre-update surprise (current prediction error)
            #
            # Design decision: We return the CURRENT prediction error (surprise), not
            # the post-update error (new_surprise). This means:
            # - The model optimizes for current prediction accuracy (standard learning)
            # - NOT for how well the memory update improves predictions (meta-learning)
            #
            # Note on mem_lr/mem_decay: These are nn.Parameters for convenience but
            # do NOT receive gradients with this design. The memory update is a side
            # effect that happens during forward pass. If meta-learning is desired,
            # return new_surprise instead and mem_lr/mem_decay will receive gradients.
            return surprise
        else:
            # No gradient flow needed - use torch.no_grad for efficiency
            with torch.no_grad():
                grad_norm = torch.sqrt(grad_up.pow(2).sum() + grad_down.pow(2).sum() + 1e-8)
                max_norm = 1.0
                if grad_norm > max_norm:
                    scale = max_norm / grad_norm
                    grad_up = grad_up * scale
                    grad_down = grad_down * scale

                # Get learning rate and decay from log-space parameters
                lr = torch.exp(self.mem_lr).clamp(min=0.001, max=0.1)
                decay = torch.exp(self.mem_decay).clamp(min=0.0, max=0.1)

                # Titans update equation: M_t = (1 - alpha) * M_{t-1} - eta * grad
                new_up = (1 - decay) * up_w - lr * grad_up
                new_down = (1 - decay) * down_w - lr * grad_down

                # Store updated weights
                self._current_up_weight = new_up.detach().requires_grad_(True)
                self._current_down_weight = new_down.detach().requires_grad_(True)

            return surprise.detach()


# =============================================================================
# Titans Block with Memory-Attention Integration (MAG)
# =============================================================================

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings."""
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


class TitansAttentionWithMAG(nn.Module):
    """
    Attention with Memory-as-Gate (MAG) integration.

    MAG combines memory output with attention output:
        output = gate * memory_out + (1 - gate) * attn_out

    The gate is computed from memory confidence (inverse of surprise).
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        memory_hidden_mult: int = 4,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        # QKV projection
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

        # Titans Memory for this attention layer
        self.memory = TitansMemory(
            input_dim=hidden_size,
            hidden_dim=hidden_size * memory_hidden_mult,
            output_dim=hidden_size,
            dtype=dtype
        )

        # Temperature for surprise-based gating
        self.surprise_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def reset_memory(self, batch_size: int, device: torch.device = None):
        """Reset memory state."""
        self.memory.reset(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory for specific samples."""
        self.memory.reset_for_samples(reset_mask)

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention and MAG integration.

        Returns:
            output: Combined output [B, L, D]
            surprise: Memory prediction error
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

        # Attention
        query_attn = einops.rearrange(query, 'B S H D -> B H S D')
        key_attn = einops.rearrange(key, 'B S H D -> B H S D')
        value_attn = einops.rearrange(value, 'B S H D -> B H S D')

        attn_output = scaled_dot_product_attention(
            query=query_attn, key=key_attn, value=value_attn, is_causal=False
        )
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S (H D)')
        attn_output = self.o_proj(attn_output)

        # Memory retrieval
        mem_output = self.memory(hidden_states)

        # Compute surprise for gating
        surprise_per_sample = (mem_output - attn_output.to(mem_output.dtype)).pow(2).mean(dim=(1, 2))
        surprise = surprise_per_sample.mean()

        # Update memory if requested
        if update_memory:
            surprise = self.memory.update(hidden_states, attn_output, create_graph=create_graph)

        # MAG: Surprise-based confidence gating
        temperature = F.softplus(self.surprise_temperature) + 0.1
        surprise_clamped = surprise_per_sample.clamp(max=50.0 / (temperature + 1e-6))
        confidence = torch.exp(-surprise_clamped * temperature)  # [B]
        confidence = confidence.view(-1, 1, 1)  # [B, 1, 1]

        # Combine: high confidence -> use memory, low confidence -> use attention
        output = confidence * mem_output + (1 - confidence) * attn_output

        return output, surprise


# =============================================================================
# TRM-Titans Block
# =============================================================================

class TRM_Titans_Block(nn.Module):
    """
    Single TRM-Titans block with attention+MAG and MLP.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Attention with MAG
        self.self_attn = TitansAttentionWithMAG(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            memory_hidden_mult=config.memory_hidden_mult,
            dtype=getattr(torch, config.forward_dtype)
        )

        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def reset_memory(self, batch_size: int, device: torch.device = None):
        """Reset memory state."""
        self.self_attn.reset_memory(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory for specific samples."""
        self.self_attn.reset_memory_for_samples(reset_mask)

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through block."""
        # Attention + MAG
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
# TRM-Titans Reasoning Module (Multi-block)
# =============================================================================

class TRM_Titans_ReasoningModule(nn.Module):
    """Reasoning module with multiple TRM-Titans blocks."""

    def __init__(self, layers: List[TRM_Titans_Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def reset_memory(self, batch_size: int, device: torch.device = None):
        """Reset memory in all layers."""
        for layer in self.layers:
            layer.reset_memory(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory in all layers."""
        for layer in self.layers:
            layer.reset_memory_for_samples(reset_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through all layers."""
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
# TRM-Titans Config
# =============================================================================

class TRM_Titans_Config(BaseModel):
    """Configuration for TRM-Titans model."""
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

    # Titans Memory specific
    memory_hidden_mult: int = 4
    surprise_loss_weight: float = 0.1


# =============================================================================
# TRM-Titans Carry (NO z_H, z_L - Memory weights ARE the state)
# =============================================================================

@dataclass
class TRM_Titans_InnerCarry:
    """
    Inner carry for TRM-Titans.

    CRITICAL: NO z_H, z_L tensors!
    The memory weights in TitansMemory modules ARE the implicit state.
    This carry only tracks minimal information needed for ACT.
    """
    # Placeholder tensor for shape/device consistency
    # The actual state is in the TitansMemory modules
    _device_placeholder: torch.Tensor


@dataclass
class TRM_Titans_Carry:
    """Outer carry for ACT wrapper."""
    inner_carry: TRM_Titans_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# =============================================================================
# TRM-Titans Inner Model
# =============================================================================

class TRM_Titans_Inner(nn.Module):
    """
    Inner model for TRM-Titans.

    Key difference from TRM/TRM-NM:
    - NO z_H, z_L tensors in carry
    - memory_H and memory_L modules store state in their weights
    - H_cycles update memory_H, L_cycles update memory_L within each H_cycle
    """

    def __init__(self, config: TRM_Titans_Config) -> None:
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

        # Titans Memory Modules (REPLACES z_H, z_L)
        # memory_H: High-level reasoning memory
        self.memory_H = TitansMemory(
            input_dim=self.config.hidden_size,
            hidden_dim=self.config.hidden_size * self.config.memory_hidden_mult,
            output_dim=self.config.hidden_size,
            dtype=self.forward_dtype
        )

        # memory_L: Low-level computation memory
        self.memory_L = TitansMemory(
            input_dim=self.config.hidden_size,
            hidden_dim=self.config.hidden_size * self.config.memory_hidden_mult,
            output_dim=self.config.hidden_size,
            dtype=self.forward_dtype
        )

        # Reasoning Layers (shared between H and L cycles)
        self.L_level = TRM_Titans_ReasoningModule(
            layers=[TRM_Titans_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # Q head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def reset_all_memory(self, batch_size: int, device: torch.device = None):
        """Reset all memory states (memory_H, memory_L, and layer memories)."""
        self.memory_H.reset(batch_size=batch_size, device=device)
        self.memory_L.reset(batch_size=batch_size, device=device)
        self.L_level.reset_memory(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory for specific samples."""
        self.memory_H.reset_for_samples(reset_mask)
        self.memory_L.reset_for_samples(reset_mask)
        self.L_level.reset_memory_for_samples(reset_mask)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings with puzzle context."""
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
        """Create empty carry (minimal, as state is in memory weights)."""
        if device is None:
            device = self.lm_head.weight.device
        return TRM_Titans_InnerCarry(
            _device_placeholder=torch.empty(1, device=device, dtype=self.forward_dtype)
        )

    def forward(
        self,
        carry: TRM_Titans_InnerCarry,
        batch: Dict[str, torch.Tensor],
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[TRM_Titans_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass using Titans-style memory iteration.

        Key insight: No z_H, z_L tensors passed through.
        Instead, memory_H and memory_L store state in their weights.

        Forward structure:
        - Input embeddings provide the "query" to memory
        - H_cycles: Update memory_H based on memory_L output
        - L_cycles: Update memory_L based on memory_H output + input

        Returns:
            new_carry: Updated carry (minimal)
            output: LM logits
            (q_halt_logits, q_continue_logits): Halting logits
            total_surprise: Sum of memory surprises
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        total_surprise = torch.tensor(0.0, device=device, dtype=torch.float32)
        should_create_graph = create_graph and self.training

        # Initialize memory state if needed (check all memory modules for consistency)
        # Check top-level memories
        top_level_needs_reset = (
            self.memory_H._current_up_weight is None or
            self.memory_L._current_up_weight is None or
            self.memory_H._batch_size != batch_size or
            self.memory_L._batch_size != batch_size or
            self.memory_H._batch_size != self.memory_L._batch_size
        )
        # Check layer memories for consistency
        layer_needs_reset = any(
            layer.self_attn.memory._current_up_weight is not None and
            layer.self_attn.memory._batch_size != batch_size
            for layer in self.L_level.layers
        )
        needs_reset = top_level_needs_reset or layer_needs_reset
        if needs_reset:
            self.reset_all_memory(batch_size=batch_size, device=device)

        # Get current memory outputs as "state" tensors (HIERARCHICAL ORDER)
        # L queries with input, H queries with L output (creates proper hierarchy)
        # OPTIMIZATION: Initial states are overwritten in the loop, so no gradients needed here
        with torch.no_grad():
            l_state = self.memory_L(input_embeddings)  # L: input -> L representation
            h_state = self.memory_H(l_state)           # H: L -> H representation (hierarchical!)

        # H_cycles-1 without grad (INTENTIONAL for efficiency)
        # This matches original TRM design where only final cycle has gradients.
        # Memory weights are still updated but without backprop tracking.
        # The final H_cycle with gradients provides sufficient learning signal.
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                # L_cycles: update L state based on H state + input
                for _L_step in range(self.config.L_cycles):
                    # Match Original TRM: injection = z_H + input (NOT l_state + z_H + input)
                    l_injection = h_state + input_embeddings
                    l_state, surprise = self.L_level(
                        l_state, l_injection,
                        update_memory=update_memory,
                        create_graph=False,
                        **seq_info
                    )
                    # Update memory_L based on new state
                    if update_memory:
                        self.memory_L.update(input_embeddings, l_state, create_graph=False)

                # Update H state based on L state
                # Match Original TRM: injection = z_L only (NOT h_state + z_L)
                h_injection = l_state
                h_state, surprise = self.L_level(
                    h_state, h_injection,
                    update_memory=update_memory,
                    create_graph=False,
                    **seq_info
                )
                # Update memory_H based on new state (key=l_state for hierarchy)
                if update_memory:
                    self.memory_H.update(l_state, h_state, create_graph=False)

        # Final H_cycle with grad
        for _L_step in range(self.config.L_cycles):
            # Match Original TRM: injection = z_H + input (NOT l_state + z_H + input)
            l_injection = h_state + input_embeddings
            l_state, surprise = self.L_level(
                l_state, l_injection,
                update_memory=update_memory,
                create_graph=should_create_graph,
                **seq_info
            )
            total_surprise = total_surprise + surprise

            if update_memory:
                mem_surprise = self.memory_L.update(input_embeddings, l_state, create_graph=should_create_graph)
                total_surprise = total_surprise + mem_surprise

        # Final H update
        # Match Original TRM: injection = z_L only (NOT h_state + z_L)
        h_injection = l_state
        h_state, surprise = self.L_level(
            h_state, h_injection,
            update_memory=update_memory,
            create_graph=should_create_graph,
            **seq_info
        )
        total_surprise = total_surprise + surprise

        if update_memory:
            mem_surprise = self.memory_H.update(l_state, h_state, create_graph=should_create_graph)
            total_surprise = total_surprise + mem_surprise

        # Detach states after ALL computations complete (moved from after L_cycles)
        if not self.training:
            l_state = l_state.detach()
            h_state = h_state.detach()

        # Synchronize all GPUs after memory updates before final output computation
        # This prevents NCCL timeout from GPU desync during distributed training
        _sync_if_distributed()

        # Normalize surprise by the number of surprise terms accumulated
        #
        # Surprise counting in the final H_cycle (with gradients):
        # - L_cycles calls to L_level.forward(): each returns SUM of n_layers surprises
        # - 1 H_update call to L_level.forward(): returns SUM of n_layers surprises
        # - If update_memory:
        #   - L_cycles calls to memory_L.update(): each returns 1 surprise
        #   - 1 call to memory_H.update(): returns 1 surprise
        #
        # Total individual surprise terms:
        # - Layer surprises: (L_cycles + 1) * n_layers
        # - Memory surprises: (L_cycles + 1) if update_memory
        n_layers = len(self.L_level.layers)
        n_forward_calls = self.config.L_cycles + 1
        if update_memory:
            # (L_cycles + 1) * n_layers layer surprises + (L_cycles + 1) memory surprises
            num_surprise_terms = n_forward_calls * (n_layers + 1)
        else:
            # Only layer surprises
            num_surprise_terms = n_forward_calls * n_layers
        total_surprise = total_surprise / max(num_surprise_terms, 1)

        # Outputs (use H state for final prediction)
        new_carry = TRM_Titans_InnerCarry(
            _device_placeholder=torch.empty(1, device=device, dtype=self.forward_dtype)
        )
        output = self.lm_head(h_state)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(h_state[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), total_surprise


# =============================================================================
# TRM-Titans Main Model (ACT Wrapper)
# =============================================================================

class TRM_Titans(nn.Module):
    """
    TRM with Titans-style Memory - ACT wrapper.

    Key feature: No z_H, z_L tensors.
    Memory weights ARE the state, stored in TitansMemory modules.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_Titans_Config(**config_dict)
        self.inner = TRM_Titans_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def reset_all_memory(self, batch_size: int = None, device: torch.device = None):
        """Reset all memory states."""
        self.inner.reset_all_memory(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory for specific samples."""
        self.inner.reset_memory_for_samples(reset_mask)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRM_Titans_Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: TRM_Titans_Carry,
        batch: Dict[str, torch.Tensor],
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[TRM_Titans_Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT.

        Memory reset for halted samples happens here before processing.
        """
        # Reset memory for halted samples (new puzzle)
        # NOTE: Always call reset_memory_for_samples to ensure all GPUs execute
        # the same code path. The mask arithmetic inside handles empty masks.
        # This prevents NCCL timeout from asymmetric collective operations.
        self.inner.reset_memory_for_samples(carry.halted)

        # Update data for new puzzles
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), surprise = self.inner(
            carry.inner_carry, new_current_data,
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

        # Reset memory for samples that just halted
        # NOTE: Always call to maintain GPU synchronization (see note above)
        self.inner.reset_memory_for_samples(halted)

        return TRM_Titans_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


# =============================================================================
# TRM-Titans ACT Loss Head
# =============================================================================

class TRM_Titans_ACTLossHead(nn.Module):
    """
    ACT Loss Head for TRM-Titans with surprise loss integration.
    """

    def __init__(self, model: TRM_Titans, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = self._get_loss_fn(loss_type)

    def _get_loss_fn(self, loss_type: str):
        """Get loss function."""
        if loss_type == "stablemax_cross_entropy":
            return stablemax_cross_entropy
        elif loss_type == "softmax_cross_entropy":
            return softmax_cross_entropy
        else:
            # Try importing from models.losses
            from models.losses import stablemax_cross_entropy as sm_ce, softmax_cross_entropy as sf_ce
            return {"stablemax_cross_entropy": sm_ce, "softmax_cross_entropy": sf_ce}.get(loss_type, sm_ce)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys,
        carry: TRM_Titans_Carry,
        batch: Dict[str, torch.Tensor],
    ):
        """
        Forward pass with loss computation.

        Returns:
            new_carry: Updated carry
            loss: Total loss
            metrics: Dict of metrics
            detached_outputs: Outputs for evaluation
            all_halted: Whether all sequences halted
        """
        # Forward model
        new_carry, outputs = self.model(carry, batch, update_memory=True, create_graph=self.training)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Predictions
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

        # Surprise loss
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


# =============================================================================
# Loss Functions
# =============================================================================

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


# =============================================================================
# Test-time Learning Utilities
# =============================================================================

class TRM_Titans_TestTime:
    """
    Utilities for test-time learning.

    At test-time, only memory and puzzle_emb are learned.
    """

    def __init__(self, model: TRM_Titans, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._freeze_pretrained()

    def _freeze_pretrained(self):
        """Freeze all parameters except memory modules and puzzle embeddings.

        Test-time adaptation uses two mechanisms:

        1. Gradient-based optimization (via optimizer.step()):
           - inner.puzzle_emb.* - receives gradients from LM/surprise loss
           - Memory template weights are included but typically don't receive
             useful gradients due to detach() in memory.update()

        2. Fast weight updates (via memory.update() during forward):
           - Memory _current_*_weight are updated via Titans memory update rule
           - This is the primary memory adaptation mechanism

        Learnable parameters (requires_grad=True):
        - inner.memory_H.* (high-level memory templates)
        - inner.memory_L.* (low-level memory templates)
        - *.self_attn.memory.* (attention layer memory templates)
        - *.mem_lr, *.mem_decay (memory hyperparameters)
        - inner.puzzle_emb.* (puzzle embeddings)

        Note: The actual memory state (_current_up_weight, _current_down_weight)
        is updated via the Titans update rule during forward pass, not through
        the optimizer. Template weights serve as initialization for memory reset.
        """
        for name, param in self.model.named_parameters():
            # Specific patterns for memory-related parameters
            is_titans_memory = (
                '.memory_H.' in name or
                '.memory_L.' in name or
                '.self_attn.memory.' in name
            )
            is_mem_hyperparam = '.mem_lr' in name or '.mem_decay' in name
            is_puzzle_emb = '.puzzle_emb.' in name

            if is_titans_memory or is_mem_hyperparam or is_puzzle_emb:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_learnable_params(self):
        """Get parameters that are learned at test-time."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def test_time_adapt(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        n_steps: int = 5,
        lr: float = 0.01,
        puzzle_id: int = 0,
        verbose: bool = False,
        accumulate_memory: bool = True,
        use_lm_loss_for_memory: bool = False
    ):
        """Adapt model to a new puzzle using demo pairs.

        This method performs test-time adaptation by learning from demonstration
        pairs. The key design choice is how memory accumulates across demos:

        Memory Accumulation Behavior (accumulate_memory=True, default):
            - Memory is initialized once at the START of each adaptation step
            - All demos within a step share and build upon the same memory state
            - Information from earlier demos is retained and refined by later demos
            - This enables few-shot learning where demos progressively teach the model

        Non-accumulating Behavior (accumulate_memory=False):
            - Memory is reset to template weights before EACH demo
            - Each demo is processed independently with fresh memory
            - No information transfer between demos within a step
            - Useful for independent demo processing

        Loss Design:
            - Surprise loss: Always applied, drives memory to predict attention output
            - LM loss: Controlled by use_lm_loss_for_memory parameter
              - False (default): LM loss gradients are blocked from memory params
              - True: LM loss also optimizes memory (original behavior)

        Args:
            demo_pairs: List of (input, target) demo pairs. Each pair contains:
                - input: Token IDs tensor [B, L]
                - target: Label IDs tensor [B, L] (use IGNORE_LABEL_ID=-100 for padding)
            n_steps: Number of adaptation steps (epochs over all demos)
            lr: Learning rate for adaptation optimizer
            puzzle_id: Puzzle identifier for puzzle embedding (should be consistent
                      with predict() calls)
            verbose: Whether to print loss progression
            accumulate_memory: If True (default), memory accumulates across demos
                              within each step. If False, memory resets before each demo.
            use_lm_loss_for_memory: If False (default), LM loss does not affect memory
                                   template weights (only surprise loss does). If True,
                                   LM loss also optimizes memory (may cause instability).

        Raises:
            ValueError: If demo_pairs is empty

        Example:
            >>> ttt = TRM_Titans_TestTime(model)
            >>> demos = [(x1, y1), (x2, y2), (x3, y3)]  # 3 demo pairs
            >>> ttt.test_time_adapt(demos, n_steps=10, lr=0.01)
            >>> predictions = ttt.predict(test_input)
        """
        # Validate input
        if not demo_pairs:
            raise ValueError("demo_pairs cannot be empty for test-time adaptation")

        # Validate batch size consistency BEFORE training (only warn once)
        first_batch_size = demo_pairs[0][0].shape[0]
        for demo_idx, (demo_x, demo_y) in enumerate(demo_pairs):
            if demo_x.shape[0] != first_batch_size:
                raise ValueError(
                    f"All demos must have the same batch size. "
                    f"Expected {first_batch_size}, got {demo_x.shape[0]} at demo {demo_idx}. "
                    f"Memory state is tied to batch size and cannot handle mismatches."
                )

        optimizer = torch.optim.Adam(self.get_learnable_params(), lr=lr)
        self.model.train()

        num_demos = len(demo_pairs)

        for step in range(n_steps):
            optimizer.zero_grad()
            total_loss_value = 0.0

            # Create carry ONCE at the start of each step (before demo loop)
            # This ensures memory accumulates across all demos within the step
            first_demo_x, first_demo_y = demo_pairs[0]
            first_demo_x = first_demo_x.to(self.device)
            first_batch = {
                "inputs": first_demo_x,
                "labels": first_demo_y.to(self.device),
                "puzzle_identifiers": torch.full(
                    (first_demo_x.shape[0],), puzzle_id,
                    dtype=torch.long, device=self.device
                )
            }
            carry = self.model.initial_carry(first_batch)

            # Process all demos in this step
            for demo_idx, (demo_x, demo_y) in enumerate(demo_pairs):
                demo_x = demo_x.to(self.device)
                demo_y = demo_y.to(self.device)

                batch = {
                    "inputs": demo_x,
                    "labels": demo_y,
                    "puzzle_identifiers": torch.full(
                        (demo_x.shape[0],), puzzle_id,
                        dtype=torch.long, device=self.device
                    )
                }

                # Forward pass
                carry, outputs = self.model(carry, batch, update_memory=True, create_graph=True)

                # Memory accumulation control:
                # - accumulate_memory=True: Memory persists across all demos in this step
                # - accumulate_memory=False: Memory resets before each demo
                #
                # The halted flag controls memory reset in model.forward():
                # - halted=True -> memory gets reset
                # - halted=False -> memory persists
                if accumulate_memory and demo_idx == 0:
                    # Prevent memory reset for all subsequent demos in this step
                    carry = TRM_Titans_Carry(
                        inner_carry=carry.inner_carry,
                        steps=carry.steps,
                        halted=torch.zeros_like(carry.halted),  # halted=False prevents reset
                        current_data=carry.current_data
                    )
                elif not accumulate_memory and demo_idx > 0:
                    # Force memory reset for each demo when not accumulating
                    # After first forward, carry.halted becomes False automatically,
                    # so we must explicitly set it to True to trigger reset
                    carry = TRM_Titans_Carry(
                        inner_carry=carry.inner_carry,
                        steps=carry.steps,
                        halted=torch.ones_like(carry.halted),  # halted=True forces reset
                        current_data=carry.current_data
                    )

                logits = outputs["logits"]
                labels = demo_y[:, :logits.shape[1]]

                # Compute LM loss
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=IGNORE_LABEL_ID
                )

                # Surprise loss (always applied to memory)
                # This is the primary learning signal for memory - it learns to predict
                # attention output, which is the core Titans design principle
                surprise_loss = self.model.config.surprise_loss_weight * outputs["surprise"]

                # Total loss computation
                # Design rationale:
                # - Surprise loss: Memory's primary objective (predict attention output)
                # - LM loss: Language modeling objective (predict next token)
                #
                # When use_lm_loss_for_memory=False (default):
                #   Memory templates only receive gradients from surprise loss.
                #   This keeps memory focused on its core function (attention prediction)
                #   while puzzle_emb still learns from both losses.
                #
                # When use_lm_loss_for_memory=True:
                #   Both losses optimize all learnable params including memory templates.
                #   May lead to memory learning task-specific shortcuts instead of
                #   general attention prediction patterns.
                if use_lm_loss_for_memory:
                    # Original behavior: both losses optimize everything
                    loss = lm_loss + surprise_loss
                    (loss / num_demos).backward()
                else:
                    # Memory templates learn from surprise only
                    # Implementation: Separate backward passes with gradient accumulation
                    #
                    # Step 1: Backward surprise loss (affects memory params)
                    # Step 2: Save memory grads
                    # Step 3: Backward LM loss (affects all params including memory)
                    # Step 4: Restore memory grads (overwrite LM loss contribution)
                    #
                    # This ensures memory only learns from surprise, while puzzle_emb
                    # and other params learn from both losses.

                    # Backward surprise loss first
                    (surprise_loss / num_demos).backward(retain_graph=True)

                    # Save memory gradients
                    memory_grads = {}
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            is_memory = (
                                '.memory_H.' in name or
                                '.memory_L.' in name or
                                '.self_attn.memory.' in name or
                                '.mem_lr' in name or
                                '.mem_decay' in name
                            )
                            if is_memory:
                                memory_grads[name] = param.grad.clone()

                    # Backward LM loss (will add to existing grads)
                    (lm_loss / num_demos).backward()

                    # Restore memory gradients (block LM loss contribution to memory)
                    # Also handle edge case: if a memory param didn't have grad after
                    # surprise backward but does after LM backward, zero it out
                    for name, param in self.model.named_parameters():
                        is_memory = (
                            '.memory_H.' in name or
                            '.memory_L.' in name or
                            '.self_attn.memory.' in name or
                            '.mem_lr' in name or
                            '.mem_decay' in name
                        )
                        if is_memory:
                            if name in memory_grads:
                                # Restore saved surprise-only gradient
                                param.grad = memory_grads[name]
                            elif param.grad is not None:
                                # Memory param has LM grad but no surprise grad - zero it out
                                # This prevents LM loss from affecting memory params that
                                # didn't receive surprise gradients
                                param.grad.zero_()

                total_loss_value += (lm_loss.item() + surprise_loss.item())

            optimizer.step()

            if verbose:
                avg_loss = total_loss_value / num_demos
                print(f"Step {step+1}/{n_steps}, Loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(
        self,
        test_input: torch.Tensor,
        update_during_prediction: bool = False,
        puzzle_id: int = 0
    ) -> torch.Tensor:
        """Make prediction after adaptation.

        Args:
            test_input: Input tensor for prediction [B, L]
            update_during_prediction: Whether to continue updating memory during prediction.
                                      True allows online refinement, False gives deterministic output.
            puzzle_id: Puzzle identifier (should match the one used in test_time_adapt)

        Returns:
            predictions: Predicted token IDs (argmax of logits) [B, L]
        """
        self.model.eval()

        with torch.no_grad():
            test_input = test_input.to(self.device)

            batch = {
                "inputs": test_input,
                "labels": torch.zeros_like(test_input),
                "puzzle_identifiers": torch.full((test_input.shape[0],), puzzle_id, dtype=torch.long, device=self.device)
            }

            carry = self.model.initial_carry(batch)

            for _ in range(self.model.config.halt_max_steps):
                carry, outputs = self.model(carry, batch, update_memory=update_during_prediction, create_graph=False)
                if carry.halted.all():
                    break

            predictions = outputs["logits"].argmax(dim=-1)

        return predictions
