"""
TRM-Style Hybrid Model for ARC-AGI Tasks

Architecture:
- LLaMA-8B frozen for text reasoning -> z_init extraction
- TRM hierarchical reasoning with carry state
- Non-causal grid generation with EOS markers
"""
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import sys
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

# Import from local models directory (copied from TinyRecursiveModels)
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


@dataclass
class HybridTRMInnerCarry:
    """Inner carry state for hierarchical reasoning"""
    z_H: torch.Tensor  # [batch, seq_len, hidden_size] - High-level reasoning state
    z_L: torch.Tensor  # [batch, seq_len, hidden_size] - Low-level reasoning state


class HybridTRMConfig(BaseModel):
    """Configuration for Hybrid TRM Model"""
    batch_size: int
    seq_len: int  # Grid sequence length (900 for 30x30)
    puzzle_emb_ndim: int = 0  # No puzzle embeddings in Phase 1
    num_puzzle_identifiers: int = 1  # Placeholder
    vocab_size: int  # 12 (PAD, EOS, colors 0-9)

    H_cycles: int  # Number of high-level reasoning cycles
    L_cycles: int  # Number of low-level reasoning cycles per H cycle

    H_layers: int  # ignored (for compatibility)
    L_layers: int  # Number of transformer layers

    # Transformer config
    hidden_size: int  # TRM hidden size (e.g., 512)
    text_hidden_size: int = 4096  # LLaMA hidden size
    expansion: float  # MLP expansion ratio
    num_heads: int
    pos_encodings: str  # "rope" or "learned"

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting config (not used in Phase 1)
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    forward_dtype: str = "bfloat16"

    # Additional config
    mlp_t: bool = False
    puzzle_emb_len: int = 0
    no_ACT_continue: bool = True


class HybridTRMBlock(nn.Module):
    """Single transformer block for TRM reasoning"""
    def __init__(self, config: HybridTRMConfig) -> None:
        super().__init__()
        self.config = config

        # Self-attention (non-causal for grid generation)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Non-causal for bidirectional reasoning
        )

        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm architecture (same as TRM)
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        # MLP
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class HybridTRMReasoningModule(nn.Module):
    """Stack of transformer layers with input injection"""
    def __init__(self, layers: List[HybridTRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HybridTRM_Inner(nn.Module):
    """Inner TRM model with hierarchical reasoning"""
    def __init__(self, config: HybridTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O layers
        self.embed_scale = math.sqrt(self.config.hidden_size)

        # Replace token embedding with z_init projection
        self.z_projection = CastedLinear(
            self.config.text_hidden_size,
            self.config.hidden_size,
            bias=False
        )

        # Grid token embedding (for problem_grid input)
        embed_init_std = 1.0 / self.embed_scale
        self.grid_embed = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Output head
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Positional encodings
        self.puzzle_emb_len = 0  # No puzzle embeddings in Phase 1
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

        # Reasoning layers
        self.L_level = HybridTRMReasoningModule(
            layers=[HybridTRMBlock(self.config) for _i in range(self.config.L_layers)]
        )

        # Initial states (broadcasted to batch)
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

    def _input_embeddings(self, z_init: torch.Tensor, problem_grid: torch.Tensor):
        """
        Create input embeddings from z_init and problem_grid

        Args:
            z_init: [batch, text_hidden_size] from LLaMA
            problem_grid: [batch, seq_len] grid tokens

        Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        # Project z_init to TRM hidden size
        z_projected = self.z_projection(z_init)  # [batch, hidden_size]

        # Embed grid tokens
        grid_embedding = self.grid_embed(problem_grid.to(torch.int32))  # [batch, seq_len, hidden_size]

        # Broadcast z_projected and add to grid embeddings
        # This injects text reasoning into every position
        embedding = grid_embedding + z_projected.unsqueeze(1)  # [batch, seq_len, hidden_size]

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """Create empty carry state"""
        return HybridTRMInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HybridTRMInnerCarry):
        """Reset carry state to initial values based on reset_flag"""
        # Move all tensors to same device as reset_flag
        device = reset_flag.device
        carry_z_H = carry.z_H.to(device)
        carry_z_L = carry.z_L.to(device)
        return HybridTRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init.to(device), carry_z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init.to(device), carry_z_L),
        )

    def forward(self, carry: HybridTRMInnerCarry, z_init: torch.Tensor, problem_grid: torch.Tensor) -> Tuple[HybridTRMInnerCarry, torch.Tensor]:
        """
        Forward pass with hierarchical reasoning

        Args:
            carry: Previous reasoning state
            z_init: [batch, text_hidden_size] from LLaMA
            problem_grid: [batch, seq_len] grid tokens

        Returns:
            new_carry: Updated reasoning state
            logits: [batch, seq_len, vocab_size] output logits
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(z_init, problem_grid)

        # Hierarchical reasoning
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad (for efficiency)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Last cycle with grad (for training)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Output logits
        new_carry = HybridTRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H)  # [batch, seq_len, vocab_size]

        return new_carry, logits


class HybridTRMModel(nn.Module):
    """
    Hybrid TRM Model combining LLaMA text reasoning with TRM grid generation

    Architecture:
    1. LLaMA-8B (frozen) generates text reasoning and z_init
    2. TRM inner model performs hierarchical reasoning with carry state
    3. Non-causal grid generation with EOS markers
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HybridTRMConfig(**config_dict)
        self.inner = HybridTRM_Inner(self.config)

        # LLaMA wrapper for text reasoning
        sys.path.append('/home/ubuntu/TinyRecursiveModels/gpt-integration')
        from models.text_reasoning import TextReasoningModule
        self.text_module = TextReasoningModule(
            model_name=config_dict.get('text_model_name', 'meta-llama/Llama-3.1-8B-Instruct'),
            freeze=True
        )

        # Projection to map TRM hidden states back to text latent space when needed
        self.trm_to_text = nn.Linear(
            self.config.hidden_size,
            self.text_module.hidden_size,
            bias=False
        )
        print("✓ Hybrid TRM Model initialized")

    def initial_carry(self, batch_size: int):
        """Create initial empty carry (will be reset on first forward)"""
        return self.inner.empty_carry(batch_size)

    def forward_with_latent(
        self,
        carry: Optional[HybridTRMInnerCarry],
        problem_grids: torch.Tensor,
        z_init: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[HybridTRMInnerCarry, torch.Tensor]:
        """Forward pass that reuses externally provided latents."""

        batch_size = problem_grids.shape[0]
        if carry is None:
            carry = self.initial_carry(batch_size)

        if reset_mask is not None:
            carry = self.inner.reset_carry(reset_mask, carry)

        z_init = z_init.to(problem_grids.device).to(torch.float32)
        new_carry, logits = self.inner(carry, z_init, problem_grids)
        return new_carry, logits

    def forward(self, carry: HybridTRMInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HybridTRMInnerCarry, torch.Tensor]:
        """
        Forward pass

        Args:
            carry: Previous reasoning state (or None for first step)
            batch: Dict with keys:
                - problem_texts: List[str] problem descriptions
                - problem_grids: [batch, 900] input grid tokens

        Returns:
            new_carry: Updated reasoning state
            logits: [batch, 900, vocab_size] output logits
        """
        batch_size = batch['problem_grids'].shape[0]

        # Reset carry if None
        if carry is None:
            carry = self.initial_carry(batch_size)
            # Reset all sequences (first forward pass)
            reset_flag = torch.ones(batch_size, dtype=torch.bool, device=batch['problem_grids'].device)
            carry = self.inner.reset_carry(reset_flag, carry)

        # 1. Get z_init from LLaMA (frozen)
        z_init, _ = self.text_module(batch['problem_texts'])

        # 2. Forward through TRM inner
        return self.forward_with_latent(carry, batch['problem_grids'], z_init)

    def carry_to_latent(self, carry: HybridTRMInnerCarry) -> torch.Tensor:
        """Project TRM carry back into text latent space."""
        summary = carry.z_H.to(torch.float32).mean(dim=1)
        return self.trm_to_text(summary)


# Test code
if __name__ == "__main__":
    print("Testing HybridTRMModel...")

    # Config matching TRM architecture
    config = {
        'batch_size': 2,
        'seq_len': 900,  # 30x30 grid
        'vocab_size': 12,  # PAD, EOS, 0-9
        'H_cycles': 2,
        'L_cycles': 1,
        'H_layers': 0,  # ignored
        'L_layers': 4,
        'hidden_size': 512,
        'text_hidden_size': 4096,
        'expansion': 2.0,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'text_model_name': 'meta-llama/Llama-3.2-8B-Instruct',
        'num_puzzle_identifiers': 1
    }

    model = HybridTRMModel(config).cuda()

    # Test batch
    batch = {
        'problem_texts': [
            "Solve this ARC puzzle: rotate 90 degrees clockwise",
            "Solve this ARC puzzle: reflect horizontally"
        ],
        'problem_grids': torch.randint(0, 12, (2, 900)).cuda()
    }

    # Initial forward
    carry = model.initial_carry(2)
    new_carry, logits = model(carry, batch)

    print(f"✓ Logits shape: {logits.shape}")  # Should be [2, 900, 12]
    print(f"✓ Carry z_H shape: {new_carry.z_H.shape}")  # Should be [2, 900, 512]
    print(f"✓ Carry z_L shape: {new_carry.z_L.shape}")  # Should be [2, 900, 512]

    print("\n✅ HybridTRMModel test passed!")
