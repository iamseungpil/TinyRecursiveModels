"""
TRM-Titans v4: Strategy Memory + Latent Reasoning

This module implements the TRM-Titans v4 architecture with a new design philosophy:

=== TRM-TITANS V4 CORE DESIGN ===

1. h_state = memory_H(l_state): Strategy Retrieval
   - memory_H weights encode "solving strategies"
   - h_state is retrieved based on current l_state
   - NO separate MLP for h_state generation
   - memory_H learns via BACKPROP from LM loss

2. l_state: Latent Reasoning (CoCoNut-like)
   - Evolves through L_cycles via attention
   - Uses h_state as guidance/context
   - Continuous thought process
   - Layer memories still learn via SURPRISE

3. Learning:
   - memory_H: Learns via BACKPROP (LM loss teaches "what strategy works")
   - Layer memories: Learn via SURPRISE (pattern learning, unchanged)

4. Output from h_state (strategy -> answer)
   - Like original TRM where output comes from z_H
   - h_state = memory_H(l_state) provides strategy representation
   - LM head produces answer from strategy

=== KEY DIFFERENCES FROM V3 ===
| v3 Design                      | v4 Design                       |
|--------------------------------|---------------------------------|
| h_state = MLP(l_state)         | h_state = memory_H(l_state)     |
| memory_H.update(l, MLP(l))     | No surprise update (backprop)   |
| memory learns MLP behavior     | memory learns strategies        |
| memory_H templates frozen      | memory_H templates TRAINABLE    |

=== TITANS CONCEPTS (ADAPTED) ===
1. Implicit State: Layer memories store fast contextual patterns
2. Strategy Memory: memory_H stores slow strategic knowledge
3. K->V Mapping: Layer memories learn input->output associations
4. Integration Strategies (plug-and-play for layer memories):
   - MAG (Memory as Gate): Confidence-based mixing
   - MAC (Memory as Context): Memory as attention KV context
   - MAL (Memory as Layer): Sequential processing

=== ARCHITECTURE ===
- Only L_level exists (H_layers config is ignored)
- L_level.attention_forward(): Fast l_state updates (L_cycles per H_cycle)
- memory_H (TitansMemory): Strategy retrieval (once per H_cycle, backprop learning)
- H_cycles: Number of high-level reasoning cycles
- L_cycles: Number of low-level computation cycles per H_cycle

=== LEARNING DESIGN ===

1. LM Loss:
   - Optimizes ALL parameters including memory_H templates
   - memory_H.requires_grad=True for backprop learning
   - Gradients flow: loss -> lm_head -> h_state -> memory_H weights

2. Layer Memory Learning (during forward pass):
   - Layer memories update via surprise-based gradient descent
   - Uses: M_t = (1 - alpha) * M_{t-1} - eta * grad(surprise)
   - Happens automatically during forward pass

3. memory_H Learning (via backprop):
   - memory_H does NOT use surprise-based updates
   - Learns from LM loss via standard backpropagation
   - Templates are TRAINABLE (requires_grad=True)

4. Surprise Loss:
   - Monitored for logging (layer memories only)
   - NOT added to total_loss

Usage:
    # During training
    model = TRM_Titans(config)
    model.freeze_layer_memory_templates()  # Freeze layer memories only
    # memory_H remains trainable for backprop learning
    loss_head = TRM_Titans_ACTLossHead(model, loss_type)
    # ... training loop ...

    # During test-time adaptation
    ttt = TRM_Titans_TestTime(model)
    ttt.test_time_adapt(demo_pairs)  # Layer memories adapt automatically
    predictions = ttt.predict(test_input)
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from contextlib import nullcontext
from abc import ABC, abstractmethod
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
# Integration Strategy ABC and Implementations (Plug-and-Play)
# =============================================================================

class IntegrationStrategy(ABC):
    """
    Abstract base class for Memory-Attention integration strategies.

    This enables plug-and-play integration of different memory-attention
    combination methods without modifying core attention logic.

    Supported strategies:
    - MAG (Memory as Gate): confidence-based mixing
    - MAC (Memory as Context): memory as attention KV context
    - MAL (Memory as Layer): sequential processing
    """

    @abstractmethod
    def combine(
        self,
        memory: 'TitansMemory',
        hidden_states: torch.Tensor,
        mem_output: torch.Tensor,
        attn_output: torch.Tensor,
        update_memory: bool,
        create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine memory and attention outputs.

        Args:
            memory: TitansMemory module for updates
            hidden_states: Input to memory [B, L, D]
            mem_output: Memory forward output [B, L, D]
            attn_output: Attention output [B, L, D]
            update_memory: Whether to update memory
            create_graph: Whether to create computation graph

        Returns:
            output: Combined output [B, L, D]
            surprise: Memory surprise value
        """
        pass


class MAGIntegration(IntegrationStrategy):
    """
    Memory as Gate (MAG): output = confidence * mem + (1-confidence) * attn

    Confidence is computed from surprise: conf = exp(-surprise * temperature)
    High confidence (low surprise) -> use memory
    Low confidence (high surprise) -> use attention

    This is the default Titans integration strategy.
    """

    def __init__(self, temperature_param: nn.Parameter):
        """
        Initialize MAG integration.

        Args:
            temperature_param: Learnable temperature parameter for gating
        """
        self.temperature_param = temperature_param

    def combine(
        self,
        memory: 'TitansMemory',
        hidden_states: torch.Tensor,
        mem_output: torch.Tensor,
        attn_output: torch.Tensor,
        update_memory: bool,
        create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine using surprise-based confidence gating."""
        # Compute surprise for gating (using current memory state that produced mem_output)
        # This measures how different the memory prediction is from attention output
        surprise_per_sample = (mem_output - attn_output.to(mem_output.dtype)).pow(2).mean(dim=(1, 2))
        surprise = surprise_per_sample.mean()

        # Update memory if requested (surprise for logging is returned from update)
        if update_memory:
            surprise = memory.update(hidden_states, attn_output, create_graph=create_graph)
        # Note: We intentionally use pre-update surprise_per_sample for gating
        # because it reflects the memory state that produced mem_output.
        # Recomputing after update would be semantically incorrect (stale mem_output).

        # MAG: Surprise-based confidence gating
        temperature = F.softplus(self.temperature_param) + 0.1
        surprise_clamped = surprise_per_sample.clamp(max=50.0 / (temperature + 1e-6))
        confidence = torch.exp(-surprise_clamped * temperature)  # [B]
        confidence = confidence.view(-1, 1, 1)  # [B, 1, 1]

        # Combine: high confidence -> use memory, low confidence -> use attention
        output = confidence * mem_output + (1 - confidence) * attn_output

        return output, surprise


class MACIntegration(IntegrationStrategy):
    """
    Memory as Context (MAC): output = Attention(Q=x, K=[x,M(x)], V=[x,M(x)])

    Memory output is concatenated to Key and Value, allowing attention
    to selectively incorporate memory information via attention weights.

    In MAC mode, the attention computation is modified to use memory as context,
    so the attn_output passed to combine() is already computed with memory context.
    """

    def __init__(self):
        """Initialize MAC integration (no additional parameters needed)."""
        pass

    def combine(
        self,
        memory: 'TitansMemory',
        hidden_states: torch.Tensor,
        mem_output: torch.Tensor,
        attn_output: torch.Tensor,  # Already computed with context in MAC
        update_memory: bool,
        create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine using memory as context.

        In MAC, attn_output is already computed with memory as context.
        We just need to update memory and compute surprise.
        """
        # Compute surprise: difference between pure memory output and attention-with-context output
        # This measures how much attention modifies/refines the memory representation.
        # High surprise = attention significantly changed memory's prediction
        # Low surprise = memory output was already close to the final result
        surprise_per_sample = (mem_output - attn_output.to(mem_output.dtype)).pow(2).mean(dim=(1, 2))
        surprise = surprise_per_sample.mean()

        # Update memory if requested (K=hidden_states, V=attn_output)
        if update_memory:
            surprise = memory.update(hidden_states, attn_output, create_graph=create_graph)

        # MAC: attention output is the final output (memory was used as context)
        return attn_output, surprise


class MALIntegration(IntegrationStrategy):
    """
    Memory as Layer (MAL): Sequential processing

    Two modes:
    - memory_first: output = Attention(Memory(x))
    - attention_first: output = Memory(Attention(x))

    This creates a sequential pipeline where one module processes
    the output of the other.
    """

    def __init__(self, order: str = "memory_first"):
        """
        Initialize MAL integration.

        Args:
            order: Processing order - "memory_first" or "attention_first"
        """
        assert order in ("memory_first", "attention_first"), \
            f"order must be 'memory_first' or 'attention_first', got {order}"
        self.order = order

    def combine(
        self,
        memory: 'TitansMemory',
        hidden_states: torch.Tensor,
        mem_output: torch.Tensor,  # Already computed based on order
        attn_output: torch.Tensor,  # Already computed based on order
        update_memory: bool,
        create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine using sequential processing.

        The mem_output and attn_output are pre-computed based on order:
        - memory_first: mem_output = M(x), attn_output = A(M(x))
        - attention_first: attn_output = A(x), mem_output = M(A(x))
        """
        # Determine which is the final output based on order
        if self.order == "memory_first":
            # Memory(x) -> Attention(Memory(x))
            # attn_output is computed from mem_output, so it's the final output
            final_output = attn_output
            # Surprise: how well memory predicts what attention wants
            surprise_per_sample = (mem_output - attn_output.to(mem_output.dtype)).pow(2).mean(dim=(1, 2))
            # Memory update: K=hidden_states, V=what attention produces
            update_k, update_v = hidden_states, attn_output
        else:  # attention_first
            # Attention(x) -> Memory(Attention(x))
            # mem_output is computed from attn_output, so it's the final output
            final_output = mem_output
            # Surprise: difference between memory's refinement and attention output
            surprise_per_sample = (mem_output - attn_output.to(mem_output.dtype)).pow(2).mean(dim=(1, 2))
            # Memory update: Learn full input -> final output mapping
            # Memory learns to predict the complete transformation from raw input
            # to final output, giving it a consistent learning signal
            update_k, update_v = hidden_states, mem_output

        surprise = surprise_per_sample.mean()

        # Update memory if requested
        if update_memory:
            surprise = memory.update(update_k, update_v, create_graph=create_graph)

        return final_output, surprise


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

        # Handle batch size changes (e.g., last batch in epoch or after error recovery)
        if self._batch_size != reset_mask.shape[0]:
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

    def forward(self, x: torch.Tensor, force_templates: bool = False) -> torch.Tensor:
        """
        Forward pass through memory using current state weights.

        Args:
            x: Input tensor [B, L, D]
            force_templates: If True, always use template weights (for v4 backprop learning).
                            This ensures gradients flow to template weights.

        Returns:
            output: Memory output [B, L, output_dim]
        """
        if force_templates:
            # v4 mode: Always use templates for backprop learning
            # This ensures gradients flow: loss -> output -> templates
            up_w = self.template_up.weight
            down_w = self.template_down.weight
        else:
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


class TitansAttention(nn.Module):
    """
    Attention with pluggable Memory integration strategy.

    Supports three integration strategies:
    - MAG (Memory as Gate): confidence-based mixing (default)
    - MAC (Memory as Context): memory as attention KV context
    - MAL (Memory as Layer): sequential processing

    The integration strategy can be configured at initialization time,
    allowing different experiments without code changes.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        memory_hidden_mult: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        integration_type: str = "mag",
        mal_order: str = "memory_first"
    ):
        """
        Initialize TitansAttention with pluggable integration strategy.

        Args:
            hidden_size: Model hidden dimension
            head_dim: Attention head dimension
            num_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            memory_hidden_mult: Memory hidden dimension multiplier
            dtype: Data type for memory computations
            integration_type: Integration strategy - "mag", "mac", or "mal"
            mal_order: For MAL - "memory_first" or "attention_first"
        """
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

        # Temperature for surprise-based gating (used by MAG)
        self.surprise_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Integration strategy (pluggable)
        self.integration_type = integration_type
        self.integration = self._create_integration(integration_type, mal_order)

    def _create_integration(self, integration_type: str, mal_order: str) -> IntegrationStrategy:
        """
        Create integration strategy based on type.

        Args:
            integration_type: Strategy type - "mag", "mac", or "mal"
            mal_order: For MAL - "memory_first" or "attention_first"

        Returns:
            IntegrationStrategy instance

        Raises:
            ValueError: If integration_type is not recognized
        """
        if integration_type == "mag":
            return MAGIntegration(self.surprise_temperature)
        elif integration_type == "mac":
            return MACIntegration()
        elif integration_type == "mal":
            return MALIntegration(order=mal_order)
        else:
            raise ValueError(
                f"Unknown integration type: {integration_type}. "
                "Must be 'mag', 'mac', or 'mal'."
            )

    def reset_memory(self, batch_size: int, device: torch.device = None):
        """Reset memory state."""
        self.memory.reset(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """Selectively reset memory for specific samples."""
        self.memory.reset_for_samples(reset_mask)

    def _standard_attention(
        self,
        hidden_states: torch.Tensor,
        cos_sin: CosSin
    ) -> torch.Tensor:
        """
        Standard self-attention computation.

        Args:
            hidden_states: Input tensor [B, L, D]
            cos_sin: Rotary position embeddings (cos, sin) or None

        Returns:
            attn_output: Attention output [B, L, D]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Apply RoPE (handle potential length mismatch)
        if cos_sin is not None:
            cos, sin = cos_sin
            # RoPE tensors may be 2D [max_len, head_dim] or 3D [batch, max_len, head_dim]
            # Slice to match input sequence length
            if cos.ndim == 2:
                # [max_len, head_dim] - slice on dim 0
                cos = cos[:seq_len]
                sin = sin[:seq_len]
            else:
                # [batch, max_len, head_dim] - slice on dim 1
                cos = cos[:, :seq_len]
                sin = sin[:, :seq_len]
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

        return attn_output

    def _attention_with_context(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        cos_sin: CosSin
    ) -> torch.Tensor:
        """
        Compute attention with memory context (for MAC).

        Q: from hidden_states (with RoPE)
        K: [hidden_states (with RoPE), context (NO RoPE)]
        V: [hidden_states, context]

        Memory context is TIMELESS and does NOT receive RoPE.
        This allows attention to selectively incorporate memory information
        without imposing positional semantics on memory.

        Args:
            hidden_states: Input tensor [B, L, D]
            context: Memory context tensor [B, ctx_len, D]
            cos_sin: Rotary position embeddings (cos, sin) or None

        Returns:
            attn_output: Attention output [B, L, D]
        """
        batch_size, seq_len, _ = hidden_states.shape
        ctx_len = context.shape[1]

        # Project Q from hidden_states
        qkv_q = self.qkv_proj(hidden_states)
        qkv_q = qkv_q.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv_q[:, :, :self.num_heads]

        # Project K, V separately from hidden_states and context
        # Hidden states K/V (will get RoPE on K)
        qkv_hidden = self.qkv_proj(hidden_states)
        qkv_hidden = qkv_hidden.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        key_hidden = qkv_hidden[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value_hidden = qkv_hidden[:, :, self.num_heads + self.num_key_value_heads:]

        # Context K/V (NO RoPE - memory is timeless)
        qkv_ctx = self.qkv_proj(context)
        qkv_ctx = qkv_ctx.view(batch_size, ctx_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        key_ctx = qkv_ctx[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value_ctx = qkv_ctx[:, :, self.num_heads + self.num_key_value_heads:]

        # Apply RoPE to query and hidden_states key ONLY (NOT to context)
        if cos_sin is not None:
            cos, sin = cos_sin
            # Handle both 2D [L, D] and 3D [B, L, D] RoPE tensors
            if cos.ndim == 2:
                cos_seq = cos[:seq_len]
                sin_seq = sin[:seq_len]
            else:  # 3D: [B, L, D]
                cos_seq = cos[:, :seq_len]
                sin_seq = sin[:, :seq_len]
            query, key_hidden = apply_rotary_pos_emb(query, key_hidden, cos_seq, sin_seq)

        # Concatenate K/V: [hidden (with RoPE), context (no RoPE)]
        key = torch.cat([key_hidden, key_ctx], dim=1)
        value = torch.cat([value_hidden, value_ctx], dim=1)

        # Attention
        query_attn = einops.rearrange(query, 'B S H D -> B H S D')
        key_attn = einops.rearrange(key, 'B S H D -> B H S D')
        value_attn = einops.rearrange(value, 'B S H D -> B H S D')

        attn_output = scaled_dot_product_attention(
            query=query_attn, key=key_attn, value=value_attn, is_causal=False
        )
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S (H D)')
        attn_output = self.o_proj(attn_output)

        return attn_output

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention and memory integration.

        The integration strategy determines how memory and attention outputs
        are combined:
        - MAG: Confidence-based gating
        - MAC: Memory as attention context
        - MAL: Sequential processing

        Args:
            cos_sin: Rotary position embeddings (cos, sin) or None
            hidden_states: Input tensor [B, L, D]
            update_memory: Whether to update memory state
            create_graph: Whether to create computation graph for backprop

        Returns:
            output: Combined output [B, L, D]
            surprise: Memory prediction error
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute memory and attention based on integration type
        # (Conditional computation to avoid wasted work)
        if self.integration_type == "mal" and isinstance(self.integration, MALIntegration):
            if self.integration.order == "memory_first":
                # MAL memory_first: Memory(x) -> Attention(Memory(x))
                mem_output = self.memory(hidden_states)
                attn_output = self._standard_attention(mem_output, cos_sin)
            else:
                # MAL attention_first: Attention(x) -> Memory(Attention(x))
                attn_output = self._standard_attention(hidden_states, cos_sin)
                mem_output = self.memory(attn_output)
        elif self.integration_type == "mac":
            # MAC: Use memory output as context for attention
            mem_output = self.memory(hidden_states)
            attn_output = self._attention_with_context(hidden_states, mem_output, cos_sin)
        else:
            # MAG (default): Standard parallel computation
            mem_output = self.memory(hidden_states)
            attn_output = self._standard_attention(hidden_states, cos_sin)

        # Apply integration strategy
        output, surprise = self.integration.combine(
            memory=self.memory,
            hidden_states=hidden_states,
            mem_output=mem_output,
            attn_output=attn_output,
            update_memory=update_memory,
            create_graph=create_graph
        )

        return output, surprise


# Backward compatibility alias
TitansAttentionWithMAG = TitansAttention


# =============================================================================
# TRM-Titans Block
# =============================================================================

class TRM_Titans_Block(nn.Module):
    """
    Single TRM-Titans block with pluggable memory integration and MLP.

    TRM-Titans v3 Architecture:
    - attention_forward(): Attention only -> produces l_state update (called L_cycles times)
    - mlp_forward(): MLP only -> produces h_state (called once per H_cycle)
    - forward(): Combined for backward compatibility (calls both)

    The integration strategy (MAG, MAC, or MAL) is determined by the config.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Attention with pluggable integration strategy
        self.self_attn = TitansAttention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            memory_hidden_mult=config.memory_hidden_mult,
            dtype=getattr(torch, config.forward_dtype),
            integration_type=getattr(config, 'integration_type', 'mag'),
            mal_order=getattr(config, 'mal_order', 'memory_first')
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

    def attention_forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-only forward pass - produces l_state update.

        TRM-Titans v3: This is called L_cycles times per H_cycle.
        Attention captures fast, contextual information.

        Args:
            cos_sin: Rotary position embeddings (cos, sin) or None
            hidden_states: Input tensor [B, L, D]
            update_memory: Whether to update layer memory state
            create_graph: Whether to create computation graph for backprop

        Returns:
            l_state: Updated l_state after attention [B, L, D]
            surprise: Memory prediction error
        """
        attn_out, surprise = self.self_attn(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            update_memory=update_memory,
            create_graph=create_graph
        )
        l_state = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        return l_state, surprise

    def mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MLP-only forward pass - produces h_state.

        TRM-Titans v3: This is called once per H_cycle.
        MLP captures slow, abstract information.

        Args:
            hidden_states: Input tensor [B, L, D]

        Returns:
            h_state: Updated h_state after MLP [B, L, D]
        """
        mlp_out = self.mlp(hidden_states)
        h_state = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        return h_state

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through block (backward compatibility).

        Calls both attention_forward and mlp_forward sequentially.
        Returns h_state as main output.

        Args:
            cos_sin: Rotary position embeddings (cos, sin) or None
            hidden_states: Input tensor [B, L, D]
            update_memory: Whether to update memory state
            create_graph: Whether to create computation graph for backprop

        Returns:
            h_state: Output after attention + MLP [B, L, D]
            surprise: Memory prediction error
        """
        # Attention -> l_state
        l_state, surprise = self.attention_forward(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            update_memory=update_memory,
            create_graph=create_graph
        )
        # MLP -> h_state
        h_state = self.mlp_forward(l_state)
        return h_state, surprise


# =============================================================================
# TRM-Titans Reasoning Module (Multi-block)
# =============================================================================

class TRM_Titans_ReasoningModule(nn.Module):
    """
    Reasoning module with multiple TRM-Titans blocks.

    TRM-Titans v3 Architecture:
    - attention_forward(): Run attention through all layers (fast, L_cycles times)
    - mlp_forward(): Run MLP through all layers (slow, once per H_cycle)
    - forward(): Combined for backward compatibility
    """

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

    def attention_forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run attention through all layers (L-level fast processing).

        TRM-Titans v3: This is called L_cycles times per H_cycle.
        Captures contextual, fast information through attention only.

        Args:
            hidden_states: Current l_state [B, L, D]
            input_injection: Injection from h_state + input_embeddings [B, L, D]
            update_memory: Whether to update layer memories
            create_graph: Whether to create computation graph for backprop
            **kwargs: Additional arguments (e.g., cos_sin for RoPE)

        Returns:
            l_state: Updated l_state after attention through all layers [B, L, D]
            total_surprise: Sum of surprise values from all layers
        """
        hidden_states = hidden_states + input_injection
        total_surprise = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

        for layer in self.layers:
            hidden_states, surprise = layer.attention_forward(
                hidden_states=hidden_states,
                update_memory=update_memory,
                create_graph=create_graph,
                **kwargs
            )
            total_surprise = total_surprise + surprise

        return hidden_states, total_surprise

    def mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Run MLP through all layers (H-level slow processing).

        TRM-Titans v3: This is called once per H_cycle.
        Captures abstract, slow information through MLP only.

        Args:
            hidden_states: Current state [B, L, D]

        Returns:
            h_state: Updated h_state after MLP through all layers [B, L, D]
        """
        for layer in self.layers:
            hidden_states = layer.mlp_forward(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        update_memory: bool = True,
        create_graph: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward through all layers (backward compatibility).

        Calls both attention and MLP for each layer sequentially.

        Args:
            hidden_states: Input tensor [B, L, D]
            input_injection: Injection tensor [B, L, D]
            update_memory: Whether to update memory state
            create_graph: Whether to create computation graph for backprop
            **kwargs: Additional arguments (e.g., cos_sin for RoPE)

        Returns:
            output: Output after all layers [B, L, D]
            total_surprise: Sum of surprise values from all layers
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
# TRM-Titans Config
# =============================================================================

class TRM_Titans_Config(BaseModel):
    """
    Configuration for TRM-Titans model.

    TRM-Titans v4 Architecture: Strategy Memory + Latent Reasoning
    - h_state = memory_H(l_state): Strategy retrieval from memory
    - l_state evolves through L_cycles via attention (CoCoNut-like)
    - memory_H learns via BACKPROP from LM loss
    - Layer memories learn via SURPRISE (pattern learning)
    - Output from h_state (strategy -> answer)

    Integration types (MAG/MAC/MAL) are plug-and-play for LAYER memories.
    memory_H uses simple forward pass (backprop learning, no integration strategy).
    """
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored (like original TRM) - only L_level is used
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

    # Integration type for memory-attention combination (plug-and-play)
    # Options: "mag" (Memory as Gate), "mac" (Memory as Context), "mal" (Memory as Layer)
    integration_type: str = "mag"
    # For MAL integration: "memory_first" or "attention_first"
    mal_order: str = "memory_first"

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        # Validate integration_type
        valid_integration_types = {"mag", "mac", "mal"}
        if self.integration_type not in valid_integration_types:
            raise ValueError(
                f"integration_type must be one of {valid_integration_types}, "
                f"got '{self.integration_type}'"
            )

        # Validate mal_order
        valid_mal_orders = {"memory_first", "attention_first"}
        if self.mal_order not in valid_mal_orders:
            raise ValueError(
                f"mal_order must be one of {valid_mal_orders}, "
                f"got '{self.mal_order}'"
            )

        # Validate cycles
        if self.H_cycles < 1:
            raise ValueError(f"H_cycles must be >= 1, got {self.H_cycles}")
        if self.L_cycles < 1:
            raise ValueError(f"L_cycles must be >= 1, got {self.L_cycles}")


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
    Inner model for TRM-Titans v4.

    TRM-Titans v4 Architecture: Strategy Memory + Latent Reasoning
    - h_state = memory_H(l_state): Strategy retrieval from memory
    - l_state evolves through L_cycles via attention (CoCoNut-like)
    - memory_H learns via BACKPROP from LM loss
    - Layer memories learn via SURPRISE (unchanged from v3)
    - Output from h_state (strategy -> answer)

    Key components:
    - L_level: Reasoning module with attention_forward() (MLP not used for h_state)
    - memory_H: TitansMemory that encodes "solving strategies"
    - H_init, L_init: Learnable initial states

    Forward loop structure:
        For each H_cycle:
            1. Strategy retrieval: h_state = memory_H(l_state)
            2. L_cycles of latent reasoning:
               l_state = L_level.attention_forward(l_state, h_state + input)
            (NOTE: No memory_H.update() - memory_H learns via backprop only)
        Final strategy retrieval: h_state = memory_H(l_state)
        Output from h_state

    Key Design Principles:
    - memory_H encodes STRATEGIES, not MLP predictions
    - memory_H learns via BACKPROP: loss -> lm_head -> h_state -> memory_H
    - Layer memories learn via SURPRISE during forward pass
    - h_state provides guidance for l_state evolution

    Properties:
    - NO z_H, z_L tensors in carry (layer state is in memory weights)
    - H_layers config is ignored (like original TRM)
    - Integration types (MAG/MAC/MAL) are plug-and-play at attention layer level
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

        # L_level: Single reasoning module (like original TRM)
        # Layer memories inside each block serve as implicit state (replacing z_H, z_L)
        self.L_level = TRM_Titans_ReasoningModule(
            layers=[TRM_Titans_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # Global H-level memory (replaces z_H, provides h_state)
        self.memory_H = TitansMemory(
            input_dim=self.config.hidden_size,
            hidden_dim=self.config.hidden_size * self.config.memory_hidden_mult,
            output_dim=self.config.hidden_size,
            dtype=self.forward_dtype
        )

        # Learnable initial states (can be loaded from TRM checkpoint)
        self.H_init = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(self.config.hidden_size))

        # Initialize with small random values
        nn.init.normal_(self.H_init, std=0.02)
        nn.init.normal_(self.L_init, std=0.02)

        # Q head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def reset_memory_H(self, batch_size: int, device: torch.device = None):
        """
        Reset global H-level memory.

        Note: In v4, memory_H uses templates directly for backprop learning,
        so this is a no-op. Kept for API compatibility.
        """
        # In v4, memory_H doesn't use _current_weights (backprop learning)
        # This is a no-op but kept for backward compatibility
        pass

    def reset_all_memory(self, batch_size: int, device: torch.device = None):
        """
        Reset all memory states (layer memories only in v4).

        In v4, memory_H uses templates directly for backprop learning,
        so only layer memories need resetting.
        """
        # Only reset layer memories (memory_H uses templates directly in v4)
        self.L_level.reset_memory(batch_size=batch_size, device=device)

    def reset_memory_for_samples(self, reset_mask: torch.Tensor):
        """
        Selectively reset memory for specific samples (layer memories only in v4).

        In v4, memory_H uses templates directly for backprop learning,
        so only layer memories need resetting.
        """
        # Only reset layer memories (memory_H uses templates directly in v4)
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
        Forward pass using TRM-Titans v4 architecture.

        TRM-Titans v4 Core Design: Strategy Memory + Latent Reasoning
            - h_state = memory_H(l_state): Strategy retrieval from memory
            - l_state evolves through L_cycles via attention (CoCoNut-like)
            - memory_H learns via BACKPROP from LM loss
            - Layer memories learn via SURPRISE during forward pass
            - Output from h_state (strategy -> answer)

        Architecture:
            For each H_cycle:
                1. Strategy retrieval: h_state = memory_H(l_state)
                2. L_cycles of latent reasoning:
                   l_state = L_level.attention_forward(l_state, h_state + input)
            Final strategy retrieval: h_state = memory_H(l_state)
            Output from h_state (not l_state!)

        Key design principles:
        - memory_H encodes STRATEGIES learned via backprop
        - Layer memories learn via SURPRISE (forward-pass updates)
        - h_state guides l_state evolution as context
        - Output comes from h_state (like original TRM outputs from z_H)

        Note: memory_H does NOT use surprise-based updates!
        - memory_H.update() is NOT called
        - memory_H learns via: loss -> lm_head -> h_state -> memory_H weights
        - memory_H templates must have requires_grad=True

        Returns:
            new_carry: Updated carry (minimal)
            output: LM logits (from h_state = memory_H output)
            (q_halt_logits, q_continue_logits): Halting logits
            total_surprise: Sum of layer memory surprises (NOT memory_H)
        """
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]
        device = batch["inputs"].device

        # Validate batch size (empty batches will cause memory update failures)
        if batch_size == 0:
            raise ValueError("TRM_Titans: Cannot process empty batch (batch_size=0)")

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        total_surprise = torch.tensor(0.0, device=device, dtype=torch.float32)
        should_create_graph = create_graph and self.training

        # Initialize layer memory state if needed
        # Note: memory_H does NOT need _current_weights for v4 (uses templates directly via backprop)
        # Only layer memories need per-batch state for surprise-based updates
        l_level_needs_reset = any(
            layer.self_attn.memory._current_up_weight is None or
            layer.self_attn.memory._batch_size != batch_size
            for layer in self.L_level.layers
        )

        if l_level_needs_reset:
            self.L_level.reset_memory(batch_size=batch_size, device=device)

        # Initialize l_state from L_init + input_embeddings (latent reasoning state)
        # Note: input_embeddings shape is [B, actual_seq_len, D] where actual_seq_len
        # includes puzzle_emb_len ONLY if puzzle_emb_ndim > 0
        l_state = self.L_init.view(1, 1, -1) + input_embeddings
        actual_seq_len = input_embeddings.shape[1]

        # H_cycles-1 without grad (INTENTIONAL for efficiency)
        # This matches original TRM design where only final cycle has gradients.
        # Layer memory weights are still updated but without backprop tracking.
        # The final H_cycle with gradients provides sufficient learning signal.
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                # 1. Strategy retrieval: h_state from memory_H
                # memory_H encodes "solving strategies" - retrieved based on current l_state
                # Note: Using templates directly (no _current_weights needed for backprop learning)
                h_state = self.memory_H(l_state, force_templates=True)

                # 2. L_cycles: Latent reasoning with h_state guidance
                # l_state evolves through attention, using h_state as context
                for _L_step in range(self.config.L_cycles):
                    l_injection = h_state + input_embeddings
                    l_state, surprise = self.L_level.attention_forward(
                        l_state, l_injection,
                        update_memory=update_memory,
                        create_graph=False,
                        **seq_info
                    )

                # NOTE: No memory_H.update() here!
                # memory_H learns via BACKPROP from LM loss, NOT surprise

        # Final H_cycle with grad
        # 1. Strategy retrieval (final, with gradient tracking for backprop)
        h_state = self.memory_H(l_state, force_templates=True)

        # 2. L_cycles: Latent reasoning with gradient tracking
        for _L_step in range(self.config.L_cycles):
            l_injection = h_state + input_embeddings
            l_state, surprise = self.L_level.attention_forward(
                l_state, l_injection,
                update_memory=update_memory,
                create_graph=should_create_graph,
                **seq_info
            )
            total_surprise = total_surprise + surprise

        # 3. Final strategy retrieval for output
        # This h_state will be used for LM output (strategy -> answer)
        # Gradients flow: loss -> lm_head -> h_state -> memory_H weights
        h_state = self.memory_H(l_state, force_templates=True)

        # Synchronize all GPUs after memory updates before final output computation
        # This prevents NCCL timeout from GPU desync during distributed training
        _sync_if_distributed()

        # Normalize surprise by the number of surprise terms accumulated
        # Only layer memories contribute to surprise in v4 (memory_H uses backprop)
        #
        # Surprise counting for the FINAL H_cycle only (previous cycles run without grad):
        # - L_cycles calls to L_level.attention_forward: each returns SUM of L_layers surprises
        # Total: L_cycles * L_layers (layer memories only, no memory_H surprise)
        n_L_layers = len(self.L_level.layers)
        num_surprise_terms = self.config.L_cycles * n_L_layers
        total_surprise = total_surprise / max(num_surprise_terms, 1)

        # OUTPUT FROM h_state (strategy -> answer)
        # h_state = memory_H(l_state) provides the strategy representation
        # LM head transforms strategy to answer logits
        output = self.lm_head(h_state)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(h_state[:, 0]).to(torch.float32)

        # Create carry (only the placeholder - layer state is in memory weights)
        new_carry = TRM_Titans_InnerCarry(
            _device_placeholder=torch.empty(1, device=device, dtype=self.forward_dtype)
        )

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), total_surprise


# =============================================================================
# TRM-Titans Main Model (ACT Wrapper)
# =============================================================================

class TRM_Titans(nn.Module):
    """
    TRM-Titans v4 with ACT (Adaptive Computation Time) wrapper.

    TRM-Titans v4 Core Design: Strategy Memory + Latent Reasoning
    - h_state = memory_H(l_state): Strategy retrieval from memory
    - l_state evolves through L_cycles via attention (CoCoNut-like)
    - memory_H learns via BACKPROP from LM loss
    - Layer memories learn via SURPRISE (pattern learning)
    - Output from h_state (strategy -> answer)

    Architecture:
    - Only L_level exists (H_layers is ignored, like original TRM)
    - L_level.attention_forward(): Fast l_state updates (L_cycles per H_cycle)
    - memory_H: TitansMemory that encodes solving strategies (backprop learning)
    - Integration types (MAG/MAC/MAL) are plug-and-play via config (for layer memories)

    Memory Design (v4):
    - memory_H templates are TRAINABLE (requires_grad=True for backprop)
    - memory_H learns via: loss -> lm_head -> h_state -> memory_H weights
    - Layer memory templates are frozen (surprise-based updates only)
    - Layer _current_weights adapt during forward pass via surprise

    Key v4 Design Principles:
    - memory_H encodes STRATEGIES, not MLP predictions
    - Gradient flow: loss -> lm_head -> h_state -> memory_H -> strategies learned
    - Layer memories handle fast pattern learning via surprise
    - Call freeze_layer_memory_templates() before training (NOT freeze_memory_templates)
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_Titans_Config(**config_dict)
        self.inner = TRM_Titans_Inner(self.config)

    @property
    def puzzle_emb(self):
        """Access puzzle embedding (returns None if puzzle_emb_ndim=0)."""
        if not hasattr(self.inner, 'puzzle_emb'):
            return None
        return self.inner.puzzle_emb

    @property
    def memory_H(self):
        """Access global H-level memory."""
        return self.inner.memory_H

    def reset_memory_H(self, batch_size: int = None, device: torch.device = None):
        """Reset global H-level memory."""
        self.inner.reset_memory_H(batch_size=batch_size, device=device)

    def freeze_layer_memory_templates(self):
        """
        Freeze only layer memory template weights (NOT memory_H).

        TRM-Titans v4 Design:
        - Layer memories learn via SURPRISE during forward pass
        - Layer memory templates should NOT receive gradients from LM loss
        - memory_H learns via BACKPROP, so it remains trainable

        Call this method before training to enforce v4 design.
        The layer _current_weights (runtime state) are NOT affected and
        continue to adapt during forward pass via surprise.

        Memory parameters frozen:
        - *.self_attn.memory.template_up/down (layer memories)
        - *.self_attn.memory.mem_lr, mem_decay (layer memory hyperparameters)

        NOT frozen (for backprop learning):
        - *.memory_H.* (global strategy memory - learns via backprop)
        """
        for name, param in self.named_parameters():
            is_layer_memory = (
                '.self_attn.memory.' in name
            )
            if is_layer_memory:
                param.requires_grad = False

    def freeze_memory_templates(self):
        """
        Freeze all memory template weights including memory_H.

        WARNING: This is the v3 design. For v4, use freeze_layer_memory_templates()
        which keeps memory_H trainable for backprop learning.

        This method freezes ALL memory-related nn.Parameters including memory_H.
        Use only if you want to completely disable backprop to memory_H.

        Memory parameters frozen:
        - *.self_attn.memory.template_up/down (layer memories)
        - *.memory_H.* (global H-level memory)
        - *.mem_lr, *.mem_decay (memory hyperparameters)
        """
        for name, param in self.named_parameters():
            is_memory = (
                '.self_attn.memory.' in name or
                '.memory_H.' in name or
                '.mem_lr' in name or
                '.mem_decay' in name
            )
            if is_memory:
                param.requires_grad = False

    def unfreeze_memory_templates(self):
        """
        Unfreeze all memory template weights (for meta-learning or fine-tuning).

        This reverses freeze_memory_templates() and allows ALL memory templates
        to receive gradients from the optimizer.
        """
        for name, param in self.named_parameters():
            is_memory = (
                '.self_attn.memory.' in name or
                '.memory_H.' in name or
                '.mem_lr' in name or
                '.mem_decay' in name
            )
            if is_memory:
                param.requires_grad = True

    def unfreeze_layer_memory_templates(self):
        """
        Unfreeze only layer memory template weights (NOT memory_H).

        This reverses freeze_layer_memory_templates() and allows layer memory
        templates to receive gradients from the optimizer.
        """
        for name, param in self.named_parameters():
            is_layer_memory = (
                '.self_attn.memory.' in name
            )
            if is_layer_memory:
                param.requires_grad = True

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
# TRM Checkpoint Loading Utilities
# =============================================================================

def load_from_trm_checkpoint(
    model: TRM_Titans,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Load compatible weights from a pretrained TRM checkpoint.

    This enables transfer learning from TRM to TRM-Titans:
    - Attention weights (qkv_proj, o_proj) are directly compatible
    - MLP weights are compatible if expansion factor matches
    - H_init from TRM initializes memory_H pattern
    - L_init from TRM initializes l_state starting point

    Args:
        model: TRM_Titans model to load weights into
        checkpoint_path: Path to TRM checkpoint file
        strict: If True, raise error on incompatible weights
        verbose: Print loading statistics

    Returns:
        Dict with 'loaded', 'skipped', 'missing' key lists
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and any(k.startswith('_orig_mod') for k in checkpoint.keys()):
        trm_state = checkpoint
    elif 'model' in checkpoint:
        trm_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        trm_state = checkpoint['state_dict']
    else:
        trm_state = checkpoint

    # Key mapping from TRM to TRM-Titans
    # TRM format: _orig_mod.model.inner.X
    # TRM-Titans format: inner.X

    loaded_keys = []
    skipped_keys = []
    missing_keys = []

    model_state = model.state_dict()
    new_state = {}

    for trm_key, trm_value in trm_state.items():
        # Remove _orig_mod.model. prefix if present
        clean_key = trm_key.replace('_orig_mod.model.', '')

        # Special handling for H_init and L_init
        if clean_key == 'inner.H_init':
            # Use H_init to initialize memory_H template
            # H_init is [hidden_size], need to use it for memory initialization
            if 'inner.H_init' in model_state:
                new_state['inner.H_init'] = trm_value
                loaded_keys.append(f'{trm_key} -> inner.H_init')
            continue

        if clean_key == 'inner.L_init':
            if 'inner.L_init' in model_state:
                new_state['inner.L_init'] = trm_value
                loaded_keys.append(f'{trm_key} -> inner.L_init')
            continue

        # Skip memory-related keys (TRM doesn't have them)
        if '.memory.' in clean_key or 'memory_H' in clean_key:
            continue

        # Check if key exists in TRM-Titans model
        if clean_key in model_state:
            if trm_value.shape == model_state[clean_key].shape:
                new_state[clean_key] = trm_value
                loaded_keys.append(f'{trm_key} -> {clean_key}')
            else:
                skipped_keys.append(
                    f'{trm_key}: shape mismatch '
                    f'(TRM: {trm_value.shape}, Titans: {model_state[clean_key].shape})'
                )
        else:
            missing_keys.append(f'{trm_key} (no match in TRM-Titans)')

    # Load the compatible weights
    model.load_state_dict(new_state, strict=False)

    if verbose:
        print(f"=== TRM Checkpoint Loading Results ===")
        print(f"Loaded: {len(loaded_keys)} weights")
        print(f"Skipped (shape mismatch): {len(skipped_keys)} weights")
        print(f"Missing (no match): {len(missing_keys)} weights")

        if skipped_keys:
            print(f"\nSkipped weights:")
            for k in skipped_keys[:5]:
                print(f"  - {k}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")

    return {
        'loaded': loaded_keys,
        'skipped': skipped_keys,
        'missing': missing_keys
    }


# =============================================================================
# TRM-Titans ACT Loss Head
# =============================================================================

class TRM_Titans_ACTLossHead(nn.Module):
    """
    ACT Loss Head for TRM-Titans v4 with surprise loss integration.

    TRM-Titans v4 Loss Design:
    - LM loss: Optimizes all parameters including memory_H templates
    - memory_H learns via BACKPROP (gradients flow through h_state)
    - Layer memory surprise: Used for layer memory update during forward pass
    - Q losses: Optimize all parameters for halting decisions

    Implementation:
    - Layer memory templates have requires_grad=False (surprise-based learning)
    - memory_H templates have requires_grad=True (backprop learning)
    - Call model.freeze_layer_memory_templates() before training
    - Surprise loss is NOT included in total_loss (layer memories learn via forward pass)
    """

    def __init__(self, model: TRM_Titans, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = self._get_loss_fn(loss_type)

        # Auto-freeze layer memory templates (NOT memory_H) to enforce v4 design
        # memory_H learns via backprop, layer memories learn via surprise
        self._enforce_v4_loss_design()

    def _enforce_v4_loss_design(self):
        """
        Enforce v4 loss design by freezing only layer memory template parameters.

        TRM-Titans v4 design:
        - Layer memories learn via forward-pass surprise updates (frozen templates)
        - memory_H learns via backprop from LM loss (trainable templates)

        This method ensures layer memory templates don't receive gradients,
        while memory_H remains trainable for strategy learning.
        """
        # Check if layer memory templates are already frozen
        layer_memory_params_unfrozen = []
        memory_h_params = []
        for name, param in self.model.named_parameters():
            is_layer_memory = '.self_attn.memory.' in name
            is_memory_h = '.memory_H.' in name

            if is_layer_memory and param.requires_grad:
                layer_memory_params_unfrozen.append(name)
            if is_memory_h:
                memory_h_params.append((name, param.requires_grad))

        if layer_memory_params_unfrozen:
            # Auto-freeze layer memory templates
            if hasattr(self.model, 'freeze_layer_memory_templates'):
                self.model.freeze_layer_memory_templates()
                print(
                    f"[TRM-Titans v4] Auto-frozen {len(layer_memory_params_unfrozen)} layer memory parameters. "
                    "Layer memories will learn via forward-pass surprise updates."
                )
            else:
                # Manual freeze if method doesn't exist
                for name, param in self.model.named_parameters():
                    if '.self_attn.memory.' in name:
                        param.requires_grad = False
                print(
                    f"[TRM-Titans v4] Auto-frozen {len(layer_memory_params_unfrozen)} layer memory parameters."
                )

        # Ensure memory_H is trainable
        memory_h_frozen = [name for name, grad in memory_h_params if not grad]
        if memory_h_frozen:
            print(
                f"[TRM-Titans v4] WARNING: memory_H parameters are frozen but should be trainable. "
                f"Unfreezing {len(memory_h_frozen)} memory_H parameters for backprop learning."
            )
            for name, param in self.model.named_parameters():
                if '.memory_H.' in name:
                    param.requires_grad = True

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
        Forward pass with loss computation following TRM-Titans v4 design.

        TRM-Titans v4 Loss Design:
        - LM loss: Optimizes all parameters including memory_H templates
        - memory_H learns via BACKPROP (gradients flow: loss -> lm_head -> h_state -> memory_H)
        - Layer memory surprise: Layer memories learn during forward pass via memory.update()
        - Q losses: Optimize all parameters for halting decisions

        Gradient Flow:
        - memory_H templates receive gradients from LM loss (backprop learning)
        - Layer memory templates are frozen (surprise-based learning only)

        Returns:
            new_carry: Updated carry
            loss: Total loss (memory_H receives gradients, layer memories don't)
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

        # Surprise loss - this is logged but NOT added to total_loss
        # Layer memories learn via surprise during forward pass (memory.update())
        # memory_H learns via backprop (included in lm_loss gradient flow)
        # The surprise value is from layer memories only (monitoring)
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

        # Total loss for backward pass
        # NOTE: surprise_loss is NOT included here
        # - Layer memories learn via forward pass (memory.update()), not via optimizer
        # - memory_H learns via backprop (gradients flow through lm_loss -> h_state -> memory_H)
        #
        # Q losses ARE included because halting decisions can legitimately use memory state
        # (the model needs to know when memory is confident to decide when to halt)
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # TRM-Titans v4 Design:
        # - memory_H templates have requires_grad=True (backprop learning)
        # - Layer memory templates have requires_grad=False (surprise-based learning)
        #
        # This is enforced by freeze_layer_memory_templates() called in __init__.
        # memory_H receives gradients from lm_loss via: loss -> lm_head -> h_state -> memory_H

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
    Utilities for test-time learning following TRM-Titans v4 design.

    TRM-Titans v4 Test-Time Adaptation Design:
    - Layer memories: Adapt automatically during forward pass via surprise-based update
    - memory_H: Uses frozen templates (no adaptation at test-time)
    - Puzzle_emb: Optionally trained via optimizer (optional)

    Key Differences from v3:
    - memory_H is frozen at test-time (learned strategies from training)
    - Layer memories still adapt via surprise during forward pass
    - Only layer memories benefit from accumulate_memory setting
    """

    def __init__(self, model: TRM_Titans, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._freeze_pretrained()

    def _freeze_pretrained(self):
        """
        Freeze parameters for test-time adaptation following v4 design.

        TRM-Titans v4 Design:
        - Layer memories adapt via forward-pass surprise update (NOT optimizer)
        - memory_H is frozen (uses learned strategies from training)
        - Only puzzle_emb uses optimizer.step() (optional)
        - All other parameters are frozen

        This method:
        1. Freezes ALL parameters (requires_grad=False)
        2. Unfreezes only puzzle_emb (requires_grad=True)

        Note: Layer memory _current_weights are still updated during forward
        via the Titans update rule, independent of requires_grad.
        """
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only puzzle_emb
        for name, param in self.model.named_parameters():
            if '.puzzle_emb.' in name:
                param.requires_grad = True

    def get_learnable_params(self):
        """
        Get parameters that are learned via optimizer at test-time.

        Following v4 design, only puzzle_emb is optimized.
        Layer memories learn during forward pass via memory.update().
        memory_H uses frozen templates (strategies learned during training).
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset_memory(self, batch_size: int = 1):
        """
        Explicitly reset layer memory to clean template state.

        Note: In v4, this only resets layer memories. memory_H uses templates
        directly and doesn't have per-batch state.

        Call this method to:
        - Reset layer memory before adapting to a new puzzle (automatic in test_time_adapt)
        - Reset layer memory between independent predictions
        - Clear layer memory if you want a fresh start

        Args:
            batch_size: Batch size for the memory state
        """
        self.model.reset_all_memory(batch_size=batch_size, device=self.device)

    def test_time_adapt(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        n_steps: int = 5,
        lr: float = 0.01,
        puzzle_id: int = 0,
        verbose: bool = False,
        accumulate_memory: bool = True
    ):
        """
        Adapt model to a new puzzle using demo pairs.

        TRM-Titans v4 Test-Time Adaptation:
        - Layer memories automatically adapt during forward pass via surprise-based update
        - memory_H uses frozen templates (strategies learned during training)
        - Puzzle_emb optionally trained via optimizer

        Memory Accumulation Behavior (accumulate_memory=True, default):
            - Layer memory is initialized once at the START of each adaptation step
            - All demos within a step share and build upon the same layer memory state
            - Information from earlier demos is retained and refined by later demos
            - This enables few-shot learning where demos progressively teach the model

        Non-accumulating Behavior (accumulate_memory=False):
            - Layer memory is reset to template weights before EACH demo
            - Each demo is processed independently with fresh layer memory
            - No information transfer between demos within a step
            - Useful for independent demo processing

        Note: memory_H is NOT affected by accumulate_memory setting (uses templates).

        Args:
            demo_pairs: List of (input, target) demo pairs. Each pair contains:
                - input: Token IDs tensor [B, L]
                - target: Label IDs tensor [B, L] (use IGNORE_LABEL_ID=-100 for padding)
            n_steps: Number of adaptation steps (epochs over all demos)
            lr: Learning rate for puzzle_emb optimizer
            puzzle_id: Puzzle identifier for puzzle embedding (should be consistent
                      with predict() calls)
            verbose: Whether to print loss progression
            accumulate_memory: If True (default), layer memory accumulates across demos
                              within each step. If False, layer memory resets before each demo.

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

        # Validate batch size consistency FIRST (before any state modification)
        first_batch_size = demo_pairs[0][0].shape[0]
        for demo_idx, (demo_x, demo_y) in enumerate(demo_pairs):
            if demo_x.shape[0] != first_batch_size:
                raise ValueError(
                    f"All demos must have the same batch size. "
                    f"Expected {first_batch_size}, got {demo_x.shape[0]} at demo {demo_idx}. "
                    f"Memory state is tied to batch size and cannot handle mismatches."
                )

        # Reset memory to clean state before adaptation (after validation passes)
        # This ensures no interference from previous puzzle adaptations
        self.model.reset_all_memory(batch_size=first_batch_size, device=self.device)

        # Get learnable params (only puzzle_emb following Titans design)
        learnable_params = self.get_learnable_params()

        # If no learnable params (no puzzle_emb), just do forward passes for memory adaptation
        if not learnable_params:
            if verbose:
                print("No learnable parameters (puzzle_emb). Running forward passes for memory adaptation only.")

            for step in range(n_steps):
                # Create carry at the start of each step
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

                total_surprise = 0.0
                with torch.no_grad():
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

                        # Forward pass - memory adapts automatically
                        carry, outputs = self.model(carry, batch, update_memory=True, create_graph=False)

                        # Control memory accumulation
                        if accumulate_memory and demo_idx == 0:
                            carry = TRM_Titans_Carry(
                                inner_carry=carry.inner_carry,
                                steps=carry.steps,
                                halted=torch.zeros_like(carry.halted),
                                current_data=carry.current_data
                            )
                        elif not accumulate_memory and demo_idx > 0:
                            carry = TRM_Titans_Carry(
                                inner_carry=carry.inner_carry,
                                steps=carry.steps,
                                halted=torch.ones_like(carry.halted),
                                current_data=carry.current_data
                            )

                        total_surprise += outputs["surprise"].item()

                if verbose:
                    avg_surprise = total_surprise / len(demo_pairs)
                    print(f"Step {step+1}/{n_steps}, Avg Surprise: {avg_surprise:.4f}")

            self.model.eval()
            return

        # We have learnable params (puzzle_emb) - use optimizer
        optimizer = torch.optim.Adam(learnable_params, lr=lr)
        # NOTE: Keep model in eval mode to prevent ACT halting logic from interfering
        # with our explicit memory control via halted flag. Memory updates still work
        # via update_memory parameter (not dependent on training mode).
        self.model.eval()

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

                # Forward pass - memory adapts automatically via memory.update()
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

                # Compute LM loss for puzzle_emb optimization
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=IGNORE_LABEL_ID
                )

                # Only LM loss for puzzle_emb (memory learns via forward pass)
                # Note: Memory templates have requires_grad=False, so they don't get gradients
                (lm_loss / num_demos).backward()

                total_loss_value += lm_loss.item()

            optimizer.step()

            if verbose:
                avg_loss = total_loss_value / num_demos
                print(f"Step {step+1}/{n_steps}, Loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(
        self,
        test_input: torch.Tensor,
        update_during_prediction: bool = False,
        puzzle_id: int = 0,
        reset_memory: bool = False
    ) -> torch.Tensor:
        """Make prediction after adaptation.

        Args:
            test_input: Input tensor for prediction [B, L]
            update_during_prediction: Whether to continue updating memory during prediction.
                                      True allows online refinement, False gives deterministic output.
            puzzle_id: Puzzle identifier (should match the one used in test_time_adapt)
            reset_memory: If True, reset memory before prediction for independent predictions.
                         If False (default), use adapted memory state from test_time_adapt.

        Returns:
            predictions: Predicted token IDs (argmax of logits) [B, L]
        """
        self.model.eval()

        with torch.no_grad():
            test_input = test_input.to(self.device)

            # Optionally reset memory for independent predictions
            if reset_memory:
                self.model.reset_all_memory(batch_size=test_input.shape[0], device=self.device)

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
