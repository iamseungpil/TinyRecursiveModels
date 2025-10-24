"""
Loss-Guided LLM Representation Selection

When TRM loss is high → explore distant representations (diversity)
When TRM loss is low → exploit nearby representations (refinement)
"""

import torch
from typing import Dict, List, Tuple

from .llm import LatentOutput


class LossGuidedGenerator:
    """
    Generates multiple LLM reasoning candidates and selects based on TRM loss.

    Strategy:
    - If previous loss high → select representation far from previous (exploration)
    - If previous loss low → select representation close to previous (exploitation)
    """

    def __init__(
        self,
        llm_module,
        text_to_latent,
        trm_model,
        high_loss_threshold: float = 2.0,
        low_loss_threshold: float = 0.5,
        num_candidates: int = 3,
        temperature: float = 0.8
    ):
        """
        Args:
            llm_module: LLM for generation
            text_to_latent: Adapter (LLM → TRM)
            trm_model: TRM for loss computation
            high_loss_threshold: Loss above this → exploration
            low_loss_threshold: Loss below this → exploitation
            num_candidates: Number of candidate generations
            temperature: Sampling temperature for diversity
        """
        self.llm = llm_module
        self.text_to_latent = text_to_latent
        self.trm = trm_model
        self.high_loss_threshold = high_loss_threshold
        self.low_loss_threshold = low_loss_threshold
        self.num_candidates = num_candidates
        self.temperature = temperature

        self.previous_z_llm = None
        self.previous_loss = None

    def reset(self):
        """Reset previous state."""
        self.previous_z_llm = None
        self.previous_loss = None

    def generate_candidates(
        self,
        problem_text: str,
        num_candidates: int,
        max_length: int = 128,
        use_chat_template: bool = False
    ) -> List[LatentOutput]:
        """Generate multiple reasoning candidates with optional diversity."""

        candidates: List[LatentOutput] = []

        for i in range(num_candidates):
            latent = self.llm.generate_latent(
                problem_text,
                latent_prefix=None,
                max_length=max_length,
                use_chat_template=use_chat_template,
                do_sample=True if i > 0 else False,
                temperature=self.temperature if i > 0 else 1.0,
            )
            candidates.append(latent)

        return candidates

    def compute_trm_loss(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute TRM forward loss.

        Args:
            z_H: [batch, seq_len, trm_dim]
            z_L: [batch, seq_len, trm_dim]
            batch: Batch with labels

        Returns:
            loss: Scalar loss
        """
        # Initialize carry
        carry = self.trm.initial_carry(batch)
        carry.inner_carry.z_H = z_H
        carry.inner_carry.z_L = z_L

        # Forward pass
        carry, loss, metrics, _, _ = self.trm(
            carry=carry,
            batch=batch,
            return_keys=[]
        )

        return loss

    def select_by_loss_and_distance(
        self,
        candidates: List[LatentOutput],
        losses: List[torch.Tensor]
    ) -> Tuple[LatentOutput, torch.Tensor]:
        """
        Select candidate based on loss and distance to previous.

        Strategy:
        - If previous loss was high → select farthest from previous (exploration)
        - If previous loss was low → select closest to previous (exploitation)
        - If no previous → select lowest loss

        Args:
            candidates: List of latent outputs
            losses: List of TRM losses

        Returns:
            Selected latent output and its loss
        """
        if self.previous_z_llm is None or self.previous_loss is None:
            # No previous → greedy (lowest loss)
            best_idx = torch.argmin(torch.stack(losses))
            return candidates[best_idx], losses[best_idx]

        # Compute distances to previous
        distances = []
        for latent in candidates:
            dist = torch.norm(latent.final_hidden - self.previous_z_llm)
            distances.append(dist)

        distances = torch.stack(distances)

        # Decision based on previous loss
        if self.previous_loss > self.high_loss_threshold:
            # High loss → EXPLORATION (select farthest)
            selected_idx = torch.argmax(distances)
            strategy = "exploration"
        elif self.previous_loss < self.low_loss_threshold:
            # Low loss → EXPLOITATION (select closest with lowest loss)
            # Weighted score: prefer low loss + close distance
            loss_tensor = torch.stack(losses)
            normalized_loss = (loss_tensor - loss_tensor.min()) / (loss_tensor.max() - loss_tensor.min() + 1e-8)
            normalized_dist = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

            # Score: lower is better (low loss + close distance)
            scores = 0.7 * normalized_loss + 0.3 * normalized_dist
            selected_idx = torch.argmin(scores)
            strategy = "exploitation"
        else:
            # Medium loss → GREEDY (just lowest loss)
            selected_idx = torch.argmin(torch.stack(losses))
            strategy = "greedy"

        selected_latent = candidates[selected_idx]
        selected_loss = losses[selected_idx]

        print(
            f"  Loss-guided selection: {strategy} (prev_loss={self.previous_loss:.3f}, "
            f"selected_loss={selected_loss:.3f}, distance={distances[selected_idx]:.3f})"
        )

        return selected_latent, selected_loss

    def generate_with_loss_guidance(
        self,
        problem_text: str,
        batch: Dict[str, torch.Tensor],
        max_length: int = 128,
        use_chat_template: bool = False
    ) -> LatentOutput:
        """
        Generate LLM representation with loss-guided selection.

        Args:
            problem_text: Problem description
            batch: Batch for TRM loss computation
            max_length: Max generation length
            use_chat_template: Use apply_chat_template

        Returns:
            LatentOutput describing the selected candidate
        """
        adapter_device = next(self.text_to_latent.parameters()).device
        # Generate multiple candidates
        candidates = self.generate_candidates(
            problem_text,
            self.num_candidates,
            max_length,
            use_chat_template
        )

        # Compute TRM loss for each candidate
        losses = []
        for latent in candidates:
            z_llm_batch = latent.final_hidden.unsqueeze(0).to(adapter_device)

            adapter_kwargs = {}
            if latent.hidden_sequence is not None:
                adapter_kwargs["llm_hidden_sequence"] = latent.hidden_sequence.unsqueeze(0).to(adapter_device)
            if latent.attention_mask is not None:
                adapter_kwargs["attention_mask"] = latent.attention_mask.unsqueeze(0).to(adapter_device)

            if hasattr(self.text_to_latent, 'forward'):
                z_H, z_L = self.text_to_latent(z_llm_batch, **adapter_kwargs)
            else:
                z_H, z_L, _, _ = self.text_to_latent(
                    z_llm_batch,
                    deterministic=True,
                    **adapter_kwargs,
                )

            # Compute loss
            loss = self.compute_trm_loss(z_H, z_L, batch)
            losses.append(loss)

        # Select based on loss and distance
        selected_latent, selected_loss = self.select_by_loss_and_distance(
            candidates, losses
        )

        # Update previous
        self.previous_z_llm = selected_latent.final_hidden.detach()
        self.previous_loss = selected_loss.detach()

        return LatentOutput(
            final_hidden=selected_latent.final_hidden,
            generated_text=selected_latent.generated_text,
            hidden_sequence=selected_latent.hidden_sequence,
            attention_mask=selected_latent.attention_mask,
        )
