from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

# Check scipy availability for Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    linear_sum_assignment = None

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


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
        use_hungarian_matching: bool = True
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.slot_recon_weight = slot_recon_weight
        self.slot_contrastive_weight = slot_contrastive_weight
        self.use_hungarian_matching = use_hungarian_matching

        # Check scipy availability for Hungarian matching
        if self.use_hungarian_matching and not HAS_SCIPY:
            raise ImportError(
                "scipy is required for Hungarian matching. "
                "Install with: pip install scipy"
            )

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

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

            # Hungarian algorithm
            row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            matched_indices.append((row_idx, col_idx))

        return matched_indices

    def compute_slot_contrastive_loss(self, carry, slots: torch.Tensor):
        """
        Compute contrastive loss between slots of same puzzle.

        Strategy:
        - Group examples by puzzle_id in the batch
        - For same puzzle_id, compute pairwise slot similarity with optional Hungarian matching
        - Maximize similarity between matched slots

        Args:
            carry: Model carry with current_data containing puzzle_identifiers
            slots: [B, num_slots, slot_dim]

        Returns:
            Contrastive loss (scalar tensor)
        """
        puzzle_ids = carry.current_data["puzzle_identifiers"]  # [B]
        B, num_slots, D = slots.shape

        total_loss = 0.0
        count = 0

        # Find pairs of same puzzle_id in batch
        unique_ids = torch.unique(puzzle_ids)

        for puzzle_id in unique_ids:
            mask = (puzzle_ids == puzzle_id)
            indices = torch.where(mask)[0]

            if len(indices) < 2:
                continue  # Need at least 2 examples

            # Pairwise contrastive
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i].item(), indices[j].item()

                    slots_1 = slots[idx1:idx1+1]  # [1, num_slots, D]
                    slots_2 = slots[idx2:idx2+1]  # [1, num_slots, D]

                    if self.use_hungarian_matching:
                        # Hungarian matching
                        matched = self.hungarian_matching(slots_1, slots_2)[0]
                        row_idx, col_idx = matched

                        # Matched slots should be similar
                        for r, c in zip(row_idx, col_idx):
                            sim = F.cosine_similarity(
                                slots_1[0, r], slots_2[0, c], dim=0
                            )
                            total_loss += -sim  # Maximize similarity = minimize negative similarity
                            count += 1
                    else:
                        # Simple pairwise similarity without matching
                        # Assume slot order is already aligned
                        for s in range(num_slots):
                            sim = F.cosine_similarity(
                                slots_1[0, s], slots_2[0, s], dim=0
                            )
                            total_loss += -sim
                            count += 1

        if count > 0:
            total_loss = total_loss / count
        else:
            total_loss = torch.tensor(0.0, device=slots.device)

        return total_loss

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
        lm_loss_direct = (self.loss_fn(logits_direct, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        # 2. Slot reconstruction loss
        lm_loss_slots = torch.tensor(0.0, device=logits_direct.device)
        if logits_slots is not None:
            lm_loss_slots = (self.loss_fn(logits_slots, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        # 3. Slot contrastive loss
        slot_contrastive_loss = torch.tensor(0.0, device=logits_direct.device)
        if slots is not None and self.slot_contrastive_weight > 0:
            slot_contrastive_loss = self.compute_slot_contrastive_loss(new_carry, slots)

        # 4. Q-learning losses
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
            "q_halt_loss": q_halt_loss.detach(),
        }

        if "target_q_continue" in outputs:
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

