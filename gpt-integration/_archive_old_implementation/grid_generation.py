"""
Grid Generation Module (TRM-style hierarchical reasoning)
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import math

from models.components.hierarchical_layers import (
    HierarchicalReasoningBlock,
    HierarchicalReasoningModule,
    RotaryPositionalEmbedding,
    RMSNorm
)


class GridGenerationModule(nn.Module):
    """
    Hierarchical grid generation from z_init

    Architecture:
        z_init [batch, text_hidden] -> project -> z_init [batch, grid_hidden]
        -> broadcast -> z_H, z_L [batch, seq_len, grid_hidden]
        -> hierarchical reasoning (H_cycles, L_cycles)
        -> grid prediction [batch, 900, vocab_size]
    """

    def __init__(
        self,
        text_hidden_size: int = 4096,  # LLaMA-8B hidden size
        grid_hidden_size: int = 1024,
        L_layers: int = 4,
        H_cycles: int = 2,
        L_cycles: int = 3,
        num_heads: int = 8,
        expansion: float = 4.0,
        dropout: float = 0.1,
        vocab_size: int = 12,  # 0-9 colors + pad + eos
        seq_len: int = 900,    # 30x30 grid
        puzzle_emb_len: int = 16
    ):
        super().__init__()

        self.text_hidden_size = text_hidden_size
        self.grid_hidden_size = grid_hidden_size
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.total_seq_len = seq_len + puzzle_emb_len  # 916

        # Project z_init from text_hidden to grid_hidden
        self.z_projection = nn.Linear(text_hidden_size, grid_hidden_size)

        # Problem embedding (encode input grid)
        self.problem_embedding = nn.Embedding(vocab_size, grid_hidden_size)

        # Position embedding (RoPE)
        self.rope = RotaryPositionalEmbedding(
            dim=grid_hidden_size // num_heads,
            max_seq_len=self.total_seq_len
        )

        # L-level layers (shared for both z_H and z_L in TRM style)
        self.L_level = HierarchicalReasoningModule(
            layers=nn.ModuleList([
                HierarchicalReasoningBlock(
                    hidden_size=grid_hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    use_rope=True
                )
                for _ in range(L_layers)
            ])
        )

        # Output head
        self.output_norm = RMSNorm(grid_hidden_size)
        self.lm_head = nn.Linear(grid_hidden_size, vocab_size, bias=False)

        # Initial states
        self.H_init = nn.Parameter(torch.randn(grid_hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(grid_hidden_size) * 0.02)

        # Embedding scale
        self.embed_scale = math.sqrt(grid_hidden_size)

    def initialize_from_text(
        self,
        z_init: torch.Tensor,
        problem_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize z_H and z_L from z_init and problem_grid

        Args:
            z_init: [batch, text_hidden_size]
            problem_grid: [batch, 900] tokenized input grid

        Returns:
            z_H: [batch, 916, grid_hidden_size]
            z_L: [batch, 916, grid_hidden_size]
            problem_emb: [batch, 916, grid_hidden_size]
        """
        batch_size = z_init.shape[0]

        # Convert z_init to float32 if needed (LLaMA outputs BFloat16)
        if z_init.dtype != torch.float32:
            z_init = z_init.float()

        # Project z_init to grid_hidden_size
        z_init_proj = self.z_projection(z_init)  # [batch, grid_hidden_size]

        # Encode problem grid
        problem_emb = self.problem_embedding(problem_grid)  # [batch, 900, grid_hidden_size]
        problem_emb = problem_emb * self.embed_scale

        # Add puzzle embedding space (prepend)
        puzzle_emb = z_init_proj.unsqueeze(1).expand(-1, self.puzzle_emb_len, -1)
        problem_emb = torch.cat([puzzle_emb, problem_emb], dim=1)  # [batch, 916, grid_hidden_size]

        # Initialize z_H and z_L by broadcasting z_init
        z_H = z_init_proj.unsqueeze(1).expand(-1, self.total_seq_len, -1)  # [batch, 916, grid_hidden_size]
        z_L = z_init_proj.unsqueeze(1).expand(-1, self.total_seq_len, -1)

        # Add small position-dependent noise for diversity
        position_noise_H = torch.randn_like(z_H) * 0.01
        position_noise_L = torch.randn_like(z_L) * 0.01

        z_H = z_H + position_noise_H
        z_L = z_L + position_noise_L

        return z_H, z_L, problem_emb

    def forward(
        self,
        z_init: torch.Tensor,
        problem_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate grid from z_init and problem_grid

        Args:
            z_init: [batch, text_hidden_size]
            problem_grid: [batch, 900]

        Returns:
            grid_logits: [batch, 900, vocab_size]
            grid_pred: [batch, 900]
        """
        # Initialize states
        z_H, z_L, problem_emb = self.initialize_from_text(z_init, problem_grid)

        # Get RoPE embeddings
        cos_sin = self.rope(self.total_seq_len)

        # Hierarchical reasoning with gradient checkpointing
        # Most cycles without gradient (memory efficient)
        with torch.no_grad():
            for h_cycle in range(self.H_cycles - 1):
                for l_cycle in range(self.L_cycles):
                    # L-level update with H+input injection
                    z_L = self.L_level(z_L, z_H + problem_emb, cos_sin)
                # H-level update with L injection
                z_H = self.L_level(z_H, z_L, cos_sin)

        # Last cycle with gradient
        for l_cycle in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + problem_emb, cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin)

        # Generate grid (skip puzzle embedding positions)
        z_H_grid = z_H[:, self.puzzle_emb_len:, :]  # [batch, 900, grid_hidden_size]

        # Output projection
        z_H_grid = self.output_norm(z_H_grid)
        grid_logits = self.lm_head(z_H_grid)  # [batch, 900, vocab_size]

        # Get predictions
        grid_pred = grid_logits.argmax(dim=-1)  # [batch, 900]

        return grid_logits, grid_pred


# Test
if __name__ == "__main__":
    print("Testing GridGenerationModule...")

    batch_size = 2
    text_hidden_size = 4096
    grid_hidden_size = 1024

    module = GridGenerationModule(
        text_hidden_size=text_hidden_size,
        grid_hidden_size=grid_hidden_size,
        L_layers=4,
        H_cycles=2,
        L_cycles=3,
        num_heads=8,
        vocab_size=12,
        seq_len=900
    )

    # Test forward
    z_init = torch.randn(batch_size, text_hidden_size)
    problem_grid = torch.randint(0, 12, (batch_size, 900))

    grid_logits, grid_pred = module(z_init, problem_grid)

    print(f"z_init shape: {z_init.shape}")
    print(f"problem_grid shape: {problem_grid.shape}")
    print(f"grid_logits shape: {grid_logits.shape}")
    print(f"grid_pred shape: {grid_pred.shape}")
    print("✅ GridGenerationModule test passed!")
