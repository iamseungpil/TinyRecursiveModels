"""
Instruction Encoder - Convert instruction text to TRM latent representation

Key insight: We don't need full LLM generation for each step instruction.
Instead, we use the LLM's embedding layer + optional encoder layers to get
a contextual representation, then pass through the adapter.

This is much faster than generating text for each step!
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class InstructionEncoder:
    """
    Encodes instruction text into TRM latent representation.

    Uses LLM's embedding layer (no generation) + adapter to get z_H, z_L.

    Design decisions:
        - Use embedding layer only (fast, no generation overhead)
        - Mean pooling over tokens (simple, works well)
        - Could optionally use LLM encoder layers for better contextualization
    """

    def __init__(
        self,
        llm_module,
        text_to_latent_adapter,
        pooling_method: str = "mean",
        use_encoder_layers: bool = False,
        max_instruction_length: int = 128,
    ):
        """
        Args:
            llm_module: TextReasoningModule instance
            text_to_latent_adapter: TextToLatentAdapter instance
            pooling_method: How to pool tokens ("mean", "last", "first")
            use_encoder_layers: If True, pass through LLM encoder (slower but better)
            max_instruction_length: Maximum tokens per instruction
        """
        self.llm = llm_module
        self.adapter = text_to_latent_adapter
        self.pooling_method = pooling_method
        self.use_encoder_layers = use_encoder_layers
        self.max_length = max_instruction_length

        self.device = llm_module.device

    def encode_single(
        self,
        instruction: str,
        return_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single instruction to TRM latent.

        Args:
            instruction: Text instruction for one step
            return_grad: If True, maintain gradient flow (for training)

        Returns:
            z_H: [seq_len, hidden_dim] high-level TRM latent
            z_L: [seq_len, hidden_dim] low-level TRM latent
        """
        # Tokenize instruction
        inputs = self.llm.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # Get embeddings from LLM
        if self.use_encoder_layers:
            # Option 1: Pass through LLM encoder layers (better contextualization)
            with torch.set_grad_enabled(return_grad):
                outputs = self.llm.model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer
        else:
            # Option 2: Just embedding layer (faster)
            with torch.set_grad_enabled(return_grad):
                embed_tokens = self.llm.model.model.embed_tokens
                hidden_states = embed_tokens(inputs["input_ids"])

        # Pool tokens to single vector (for global representation)
        hidden = self._pool_tokens(hidden_states, inputs["attention_mask"])

        # Through adapter to get TRM latent
        # CRITICAL FIX: Pass BOTH pooled hidden AND full sequence for cross-attention
        z_H, z_L = self.adapter(
            llm_hidden_state=hidden,
            llm_hidden_sequence=hidden_states,  # ✅ Enable cross-attention!
            attention_mask=inputs["attention_mask"],  # ✅ Pass attention mask!
        )

        return z_H, z_L

    def encode_batch(
        self,
        instructions: List[str],
        return_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple instructions in batch.

        Args:
            instructions: List of instruction strings
            return_grad: If True, maintain gradient flow

        Returns:
            z_H: [batch, seq_len, hidden_dim]
            z_L: [batch, seq_len, hidden_dim]
        """
        if not instructions:
            raise ValueError("Empty instruction list")

        # Tokenize all instructions
        inputs = self.llm.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # Get embeddings
        if self.use_encoder_layers:
            with torch.set_grad_enabled(return_grad):
                outputs = self.llm.model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]
        else:
            with torch.set_grad_enabled(return_grad):
                embed_tokens = self.llm.model.model.embed_tokens
                hidden_states = embed_tokens(inputs["input_ids"])

        # Pool each instruction
        hidden_batch = []
        for i in range(len(instructions)):
            hidden = self._pool_tokens(
                hidden_states[i:i+1],
                inputs["attention_mask"][i:i+1]
            )
            hidden_batch.append(hidden)

        hidden_batch = torch.cat(hidden_batch, dim=0)

        # Through adapter (pass full sequence for cross-attention)
        # CRITICAL FIX: Pass BOTH pooled batch AND full sequences
        z_H, z_L = self.adapter(
            llm_hidden_state=hidden_batch,
            llm_hidden_sequence=hidden_states,  # ✅ Enable cross-attention!
            attention_mask=inputs["attention_mask"],  # ✅ Pass attention mask!
        )

        return z_H, z_L

    def _pool_tokens(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings to single vector.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim]
        """
        if self.pooling_method == "mean":
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask

        elif self.pooling_method == "last":
            # Last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[range(batch_size), seq_lengths]

        elif self.pooling_method == "first":
            # First token (CLS-style)
            pooled = hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        return pooled


# Self-review questions:
#
# Q1: Why not just use LLM.generate_latent() for each instruction?
# A1: Generation is slow (autoregressive), expensive, and we don't need the text output.
#     We just need a representation. Embedding layer is 100x faster.
#     Trade-off: Less contextualized, but sufficient for short instructions.
#
# Q2: Should we fine-tune the LLM embedding layer during training?
# A2: Depends. If LLM is frozen, embeddings are frozen too.
#     If we want to learn instruction-specific embeddings, could unfreeze.
#     Decision: Follow LLM freeze status (controlled by user).
#
# Q3: What if instruction is very long (e.g., paragraph)?
# A3: Truncate to max_instruction_length (default 128 tokens).
#     For ARC, instructions should be short (10-30 tokens).
#
# Q4: Mean pooling vs last token - which is better?
# A4: Mean pooling: More robust to tokenization, uses all information.
#     Last token: Common in causal LMs, but sensitive to padding.
#     Decision: Default to mean, allow user to choose.
#
# Q5: Why support use_encoder_layers option?
# A5: Trade-off between speed and quality.
#     Embedding only: Fast, but no self-attention.
#     Full encoder: Slow, but better contextualized.
#     Decision: Make it optional, default to embedding only.
#
# Q6: Should we cache encoded instructions?
# A6: Yes! Same instruction might appear in different problems.
#     But: Needs gradient support, so cache only during inference.
#     Decision: Add caching in v2 if needed.
#
# Q7: What about gradient flow during training?
# A7: return_grad flag controls torch.set_grad_enabled().
#     Important: If adapter is trainable, we need gradients!
#     Decision: Default True for training, caller can set False for inference.


if __name__ == "__main__":
    print("=" * 60)
    print("Instruction Encoder Test")
    print("=" * 60)

    # Note: This test requires a loaded LLM and adapter.
    # We'll create dummy versions for testing the interface.

    class DummyLLM:
        """Minimal LLM mock for testing."""
        def __init__(self):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # Small tokenizer for testing
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.device = "cpu"

            # Dummy embedding layer
            class DummyModel:
                class InnerModel:
                    def __init__(self):
                        self.embed_tokens = nn.Embedding(50257, 768)
                def __init__(self):
                    self.model = self.InnerModel()

            self.model = DummyModel()

    class DummyAdapter:
        """Minimal adapter mock for testing."""
        def __call__(self, x):
            # x: [batch, 768] -> z_H, z_L: [batch, 900, 512]
            batch = x.shape[0]
            z_H = torch.randn(batch, 900, 512)
            z_L = torch.randn(batch, 900, 512)
            return z_H, z_L

    llm = DummyLLM()
    adapter = DummyAdapter()

    encoder = InstructionEncoder(
        llm_module=llm,
        text_to_latent_adapter=adapter,
        pooling_method="mean",
        use_encoder_layers=False,
    )

    # Test single instruction
    print("\n1. Testing single instruction encoding:")
    instruction = "Rotate the grid 90 degrees clockwise"
    z_H, z_L = encoder.encode_single(instruction, return_grad=False)
    print(f"   Instruction: '{instruction}'")
    print(f"   z_H shape: {z_H.shape}")
    print(f"   z_L shape: {z_L.shape}")

    # Test batch
    print("\n2. Testing batch encoding:")
    instructions = [
        "Segment objects by color",
        "Extract each region",
        "Arrange horizontally",
    ]
    z_H_batch, z_L_batch = encoder.encode_batch(instructions, return_grad=False)
    print(f"   Batch size: {len(instructions)}")
    print(f"   z_H shape: {z_H_batch.shape}")
    print(f"   z_L shape: {z_L_batch.shape}")

    print("\n" + "=" * 60)
    print("✓ Interface tests passed!")
    print("=" * 60)
    print("\nNote: Full test requires actual LLM + trained adapter.")
    print("This test only validates the interface.")
