"""
Slot Attention Loss Functions

Loss heads for TRM with Slot Attention and Contrastive Learning.
Separated from models/losses.py to avoid scipy dependency for baseline TRM.
"""

from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

# Check scipy availability for Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    linear_sum_assignment = None

# Import common loss functions from baseline
from models.losses import softmax_cross_entropy, stablemax_cross_entropy, IGNORE_LABEL_ID


class SlotContrastiveLossHead(nn.Module):
    """
    Loss head for TRM with Slot Attention and Contrastive Learning.

    Combines:
    1. Direct LM loss (from z_H)
    2. Slot reconstruction loss (from slots)
    3. Slot contrastive loss with Hungarian matching
    4. Q-learning losses
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        slot_recon_weight: float = 0.5,
        slot_contrastive_weight: float = 0.1,
        slot_diversity_weight: float = 0.01,
        use_hungarian_matching: bool = True
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.slot_recon_weight = slot_recon_weight
        self.slot_contrastive_weight = slot_contrastive_weight
        self.slot_diversity_weight = slot_diversity_weight
        self.use_hungarian_matching = use_hungarian_matching

        # Check scipy availability for Hungarian matching
        if self.use_hungarian_matching and not HAS_SCIPY:
            raise ImportError(
                "scipy is required for Hungarian matching. "
                "Install with: pip install scipy"
            )

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    @torch.compiler.disable()
    def hungarian_matching(self, slots_1: torch.Tensor, slots_2: torch.Tensor):
        """
        Find optimal slot assignment using Hungarian algorithm.

        Args:
            slots_1: [B, num_slots, slot_dim]
            slots_2: [B, num_slots, slot_dim]

        Returns:
            matched_indices: List of (row_idx, col_idx) tuples for each batch element
        """
        B, num_slots, D = slots_1.shape
        matched_indices = []

        for b in range(B):
            # Cost matrix: negative cosine similarity
            cost_matrix = -F.cosine_similarity(
                slots_1[b:b+1, :, None, :],  # [1, num_slots, 1, D]
                slots_2[b:b+1, None, :, :],  # [1, 1, num_slots, D]
                dim=-1
            ).squeeze(0)  # [num_slots, num_slots]

            # Hungarian algorithm (detach for numpy, matching is not differentiable)
            # Convert to float32 since scipy doesn't support bfloat16
            row_idx, col_idx = linear_sum_assignment(cost_matrix.detach().cpu().float().numpy())
            matched_indices.append((row_idx, col_idx))

        return matched_indices

    def compute_slot_contrastive_loss(self, carry, slots: torch.Tensor, temperature: float = 0.07):
        """
        Compute slot-level contrastive loss with InfoNCE.

        Strategy:
        - For each pair of examples in batch, use Hungarian matching to find semantic slot alignment
        - Matched slots = positive pairs (same semantic meaning across different puzzles)
        - Non-matched slots = negative pairs (different semantic meanings)
        - Apply InfoNCE loss: -log(exp(pos/τ) / (exp(pos/τ) + Σexp(neg/τ)))

        This encourages:
        1. Slots with same semantics (e.g., "rotation") to be similar across all puzzles
        2. Slots with different semantics (e.g., "rotation" vs "mirroring") to be dissimilar

        Args:
            carry: Model carry (not used but kept for interface compatibility)
            slots: [B, num_slots, slot_dim]
            temperature: Temperature parameter for InfoNCE loss (default: 0.07)

        Returns:
            Contrastive loss (scalar tensor)
        """
        B, num_slots, D = slots.shape

        if B < 2:
            # Need at least 2 examples for contrastive learning
            return torch.tensor(0.0, device=slots.device)

        total_loss = 0.0
        count = 0

        # For each pair of examples
        for i in range(B):
            for j in range(i+1, B):
                slots_i = slots[i:i+1]  # [1, num_slots, D]
                slots_j = slots[j:j+1]  # [1, num_slots, D]

                if self.use_hungarian_matching:
                    # Hungarian matching to find semantic alignment
                    matched = self.hungarian_matching(slots_i, slots_j)[0]
                    row_idx, col_idx = matched

                    # For each matched slot pair
                    for k in range(num_slots):
                        anchor = slots_i[0, row_idx[k]]  # [D]
                        positive = slots_j[0, col_idx[k]]  # [D] - matched slot

                        # Negatives: all OTHER slots from example j
                        negative_mask = torch.ones(num_slots, dtype=torch.bool, device=slots.device)
                        negative_mask[col_idx[k]] = False
                        negatives = slots_j[0, negative_mask]  # [num_slots-1, D]

                        # Compute similarities
                        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0), dim=1) / temperature
                        neg_sims = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=1) / temperature  # [num_slots-1]

                        # InfoNCE loss
                        logits = torch.cat([pos_sim, neg_sims])  # [num_slots]
                        labels = torch.zeros(1, dtype=torch.long, device=slots.device)  # Positive is index 0
                        loss = F.cross_entropy(logits.unsqueeze(0), labels)

                        total_loss += loss
                        count += 1
                else:
                    # Without Hungarian matching, assume slot order is aligned
                    # Still use InfoNCE loss for proper negative sampling
                    for k in range(num_slots):
                        anchor = slots_i[0, k]  # [D]
                        positive = slots_j[0, k]  # [D] - aligned slot

                        # Negatives: all OTHER slots from example j
                        negative_mask = torch.ones(num_slots, dtype=torch.bool, device=slots.device)
                        negative_mask[k] = False
                        negatives = slots_j[0, negative_mask]  # [num_slots-1, D]

                        # Compute similarities
                        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0), dim=1) / temperature
                        neg_sims = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=1) / temperature

                        # InfoNCE loss
                        logits = torch.cat([pos_sim, neg_sims])
                        labels = torch.zeros(1, dtype=torch.long, device=slots.device)
                        loss = F.cross_entropy(logits.unsqueeze(0), labels)

                        total_loss += loss
                        count += 1

        if count > 0:
            total_loss = total_loss / count
        else:
            total_loss = torch.tensor(0.0, device=slots.device)

        return total_loss

    def compute_slot_diversity_loss(self, slots: torch.Tensor):
        """
        Encourage diversity among slots to prevent slot collapse.

        Penalizes high pairwise similarity between different slots within same example.
        This prevents all slots from converging to the same representation.

        Args:
            slots: [B, num_slots, slot_dim]

        Returns:
            Diversity loss (scalar) - lower is more diverse
        """
        B, K, D = slots.shape

        # Normalize slots for cosine similarity
        slots_norm = F.normalize(slots, dim=-1, p=2)  # [B, K, D]

        # Compute pairwise cosine similarity within each batch
        # similarity[b, i, j] = cosine_similarity(slots[b, i], slots[b, j])
        similarity = torch.einsum('bkd,bqd->bkq', slots_norm, slots_norm)  # [B, K, K]

        # Mask out diagonal (self-similarity = 1.0)
        mask = torch.eye(K, device=slots.device, dtype=torch.bool)
        similarity_off_diag = similarity.masked_fill(mask, 0.0)

        # Penalize high off-diagonal similarity
        # Mean absolute similarity (excluding diagonal)
        diversity_loss = similarity_off_diag.abs().sum() / (B * K * (K - 1))

        return diversity_loss

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model forward
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Extract outputs
        logits_direct = outputs["logits"]  # Direct from z_H
        logits_slots = outputs.get("logits_slots", None)  # From slots
        slots = outputs.get("slots", None)  # Slot representations

        with torch.no_grad():
            # Preds (use direct logits)
            outputs["preds"] = torch.argmax(logits_direct, dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(logits_direct, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)

        # 1. Direct LM loss
        lm_loss_direct = (self.loss_fn(logits_direct, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

        # 2. Slot reconstruction loss
        lm_loss_slots = torch.tensor(0.0, device=logits_direct.device)
        if logits_slots is not None:
            lm_loss_slots = (self.loss_fn(logits_slots, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

        # 3. Slot contrastive loss
        slot_contrastive_loss = torch.tensor(0.0, device=logits_direct.device)
        if slots is not None and self.slot_contrastive_weight > 0:
            slot_contrastive_loss = self.compute_slot_contrastive_loss(new_carry, slots)

        # 4. Slot diversity loss (prevent slot collapse)
        slot_diversity_loss = torch.tensor(0.0, device=logits_direct.device)
        if slots is not None and self.slot_diversity_weight > 0:
            slot_diversity_loss = self.compute_slot_diversity_loss(slots)

        # 5. Q-learning losses
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum"
        )

        q_continue_loss = torch.tensor(0.0, device=logits_direct.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            )

        # Total loss
        total_loss = (
            lm_loss_direct +
            self.slot_recon_weight * lm_loss_slots +
            self.slot_contrastive_weight * slot_contrastive_loss +
            self.slot_diversity_weight * slot_diversity_loss +
            0.5 * (q_halt_loss + q_continue_loss)
        )

        # Metrics
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
            "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            "lm_loss": lm_loss_direct.detach(),
            "lm_loss_slots": lm_loss_slots.detach(),
            "slot_contrastive_loss": slot_contrastive_loss.detach(),
            "slot_diversity_loss": slot_diversity_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }

        if "target_q_continue" in outputs:
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
