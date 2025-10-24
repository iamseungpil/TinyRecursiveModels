"""
Hybrid Model: Text Reasoning (LLaMA) + Grid Generation (TRM)
Phase 1 MVP: Frozen text, trainable grid decoder
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.text_reasoning import TextReasoningModule
from models.grid_generation import GridGenerationModule


class HybridARCModel_MVP(nn.Module):
    """
    Phase 1 MVP: Hybrid model for ARC-AGI

    Architecture:
        1. Text Reasoning (Frozen LLaMA-8B)
           Input: problem_text -> Output: z_init [batch, 4096]

        2. Grid Generation (Trainable TRM-style)
           Input: z_init + problem_grid -> Output: grid_pred [batch, 900]

        3. Verification & Retry
           If wrong: retry (max 3 attempts)
    """

    def __init__(
        self,
        text_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        text_hidden_size: int = 4096,
        grid_hidden_size: int = 1024,
        L_layers: int = 4,
        H_cycles: int = 2,
        L_cycles: int = 3,
        num_heads: int = 8,
        expansion: float = 4.0,
        dropout: float = 0.1,
        vocab_size: int = 12,
        seq_len: int = 900,
        max_attempts: int = 3,
        device: str = "cuda:3"
    ):
        super().__init__()

        # When CUDA_VISIBLE_DEVICES is set, use "cuda" instead of specific device number
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_attempts = max_attempts

        # Text reasoning module (frozen)
        print("Initializing Text Reasoning Module...")
        self.text_module = TextReasoningModule(
            model_name=text_model_name,
            freeze=True,
            device=self.device
        )

        # Grid generation module (trainable)
        print("Initializing Grid Generation Module...")
        self.grid_module = GridGenerationModule(
            text_hidden_size=text_hidden_size,
            grid_hidden_size=grid_hidden_size,
            L_layers=L_layers,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            vocab_size=vocab_size,
            seq_len=seq_len
        ).to(self.device)

        print("✓ Hybrid Model MVP initialized")

    def forward(
        self,
        problem_texts: List[str],
        problem_grids: torch.Tensor,
        target_grids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with self-correction loop (up to max_attempts)

        Args:
            problem_texts: List of problem descriptions
            problem_grids: [batch, 900] input grids
            target_grids: [batch, 900] target grids (for training)

        Returns:
            dict with:
                - grid_logits: [batch, 900, vocab_size]
                - grid_pred: [batch, 900]
                - attempts: [batch] number of attempts used per sample
                - is_correct: [batch] boolean (if target provided)
        """
        batch_size = problem_grids.shape[0]

        # Move to device
        problem_grids = problem_grids.to(self.device)
        if target_grids is not None:
            target_grids = target_grids.to(self.device)

        # Track which samples are still incorrect
        # Initially, all samples need to be solved
        samples_to_retry = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        attempts_used = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Store final outputs
        final_grid_logits = None
        final_grid_pred = None
        final_z_init = None

        # Self-correction loop
        for attempt in range(self.max_attempts):
            # Prepare text prompts for this attempt
            if attempt == 0:
                # First attempt: use original prompts
                current_texts = problem_texts
            else:
                # Subsequent attempts: add feedback about previous failure
                current_texts = []
                for i, text in enumerate(problem_texts):
                    if samples_to_retry[i]:
                        feedback_text = (
                            f"{text}\n\n"
                            f"[FEEDBACK] Previous attempt {attempt} was incorrect. "
                            f"Please reconsider the pattern and try a different approach."
                        )
                        current_texts.append(feedback_text)
                    else:
                        # Keep original text for already-correct samples
                        current_texts.append(text)

            # Phase 1: Text Reasoning (no gradient)
            with torch.no_grad():
                z_init, _ = self.text_module(
                    current_texts,
                    max_length=256  # Shorter for MVP
                )

            # Phase 2: Grid Generation
            # During training, only compute gradients on the LAST attempt to save memory
            if self.training and attempt < self.max_attempts - 1:
                with torch.no_grad():
                    grid_logits, grid_pred = self.grid_module(z_init, problem_grids)
            else:
                # Last attempt or inference mode: compute with gradients
                grid_logits, grid_pred = self.grid_module(z_init, problem_grids)

            # Update attempts counter for all samples being retried
            attempts_used[samples_to_retry] = attempt + 1

            # Verification: check correctness
            if target_grids is not None:
                is_correct = (grid_pred == target_grids).all(dim=1)

                # Update which samples still need retry
                samples_to_retry = samples_to_retry & (~is_correct)

                # If all samples are correct, we can stop early
                if not samples_to_retry.any():
                    # Store final outputs
                    final_grid_logits = grid_logits
                    final_grid_pred = grid_pred
                    final_z_init = z_init
                    break

            # Store outputs for this attempt (in case it's the last one)
            final_grid_logits = grid_logits
            final_grid_pred = grid_pred
            final_z_init = z_init

        # Compute final correctness
        final_is_correct = None
        if target_grids is not None:
            final_is_correct = (final_grid_pred == target_grids).all(dim=1)

        return {
            'grid_logits': final_grid_logits,
            'grid_pred': final_grid_pred,
            'attempts': attempts_used,
            'is_correct': final_is_correct,
            'z_init': final_z_init  # For debugging
        }

    def generate(
        self,
        problem_texts: List[str],
        problem_grids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generation mode (inference only)"""
        self.eval()
        with torch.no_grad():
            return self.forward(problem_texts, problem_grids, target_grids=None)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable vs frozen parameters"""
        text_params = sum(p.numel() for p in self.text_module.parameters())
        grid_params = sum(p.numel() for p in self.grid_module.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'text_module': text_params,
            'grid_module': grid_params,
            'total': text_params + grid_params,
            'trainable': trainable_params,
            'frozen': text_params + grid_params - trainable_params
        }


# Test
if __name__ == "__main__":
    print("Testing Hybrid Model MVP...")

    model = HybridARCModel_MVP(
        text_model_name="meta-llama/Llama-3.2-8B-Instruct",
        grid_hidden_size=1024,
        L_layers=2,  # Small for testing
        H_cycles=1,
        L_cycles=1,
        device="cuda:3"
    )

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v/1e6:.1f}M")

    # Test forward (mock data)
    batch_size = 1
    problem_texts = ["Solve this ARC puzzle: rotate 90 degrees"]
    problem_grids = torch.randint(0, 12, (batch_size, 900))
    target_grids = torch.randint(0, 12, (batch_size, 900))

    print("\nTesting forward pass...")
    outputs = model(problem_texts, problem_grids, target_grids)

    print(f"grid_logits shape: {outputs['grid_logits'].shape}")
    print(f"grid_pred shape: {outputs['grid_pred'].shape}")
    print(f"is_correct: {outputs['is_correct']}")

    print("\n✅ Hybrid Model MVP test passed!")
