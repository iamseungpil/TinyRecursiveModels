"""
JEPA-style Contrastive Learning for TRM

This module implements Joint-Embedding Predictive Architecture (JEPA) style
contrastive learning to improve TRM's representation learning on ARC tasks.

Key Features:
- Same task, different inputs → similar latent representations
- Different tasks → different latent representations
- Self-supervised learning in latent space (no pixel reconstruction)
- EMA target encoder for stable training
- Compatible with existing TRM training pipeline

References:
- JEPA: https://arxiv.org/abs/2301.08243
- SimSiam: https://arxiv.org/abs/2011.10566
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class JEPAContrastiveTRM(nn.Module):
    """
    TRM with JEPA-style contrastive learning.

    This wrapper adds contrastive learning capabilities to any TRM model
    by learning representations where examples from the same task have
    similar embeddings in latent space.

    Args:
        trm_model: Base TRM model (TinyRecursiveReasoningModel_ACTV1)
        hidden_size: Hidden dimension size of TRM
        proj_dim: Projection dimension for contrastive learning
        ema_decay: Exponential moving average decay for target encoder
        pool_method: Method to pool z_H ['mean', 'max', 'first', 'last']
    """

    def __init__(
        self,
        trm_model: nn.Module,
        hidden_size: int = 512,
        proj_dim: int = 256,
        ema_decay: float = 0.99,
        pool_method: str = 'mean'
    ):
        super().__init__()

        # Online encoder (with gradient)
        self.encoder = trm_model

        # Target encoder (EMA, no gradient)
        self.target_encoder = copy.deepcopy(trm_model)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Projection head (shared between online and target)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )

        # Predictor (only for online branch)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )

        self.ema_decay = ema_decay
        self.pool_method = pool_method

    def _pool_latent(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Pool z_H to fixed-size representation.

        Args:
            z_H: [batch, seq_len, hidden_size]

        Returns:
            Pooled representation: [batch, hidden_size]
        """
        if self.pool_method == 'mean':
            return z_H.mean(dim=1)
        elif self.pool_method == 'max':
            return z_H.max(dim=1)[0]
        elif self.pool_method == 'first':
            return z_H[:, 0]
        elif self.pool_method == 'last':
            return z_H[:, -1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

    def encode(
        self,
        inputs: torch.Tensor,
        puzzle_ids: torch.Tensor,
        encoder: nn.Module
    ) -> torch.Tensor:
        """
        Encode inputs to latent representation using given encoder.

        Args:
            inputs: Input tokens [batch, seq_len]
            puzzle_ids: Puzzle identifiers [batch]
            encoder: TRM encoder to use

        Returns:
            Pooled latent representation: [batch, hidden_size]
        """
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_ids
        }

        # Forward through TRM
        carry = encoder.initial_carry(batch)
        carry, output = encoder(carry, batch)

        # Pool z_H to fixed size
        z_H = carry.inner_carry.z_H  # [batch, seq_len, hidden_size]
        z_pooled = self._pool_latent(z_H)  # [batch, hidden_size]

        return z_pooled

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with contrastive loss.

        Args:
            batch: Dictionary with keys:
                - 'inputs': [batch, seq_len] primary input
                - 'inputs_aug': [batch, seq_len] augmented input (same task)
                - 'puzzle_identifiers': [batch] task IDs

        Returns:
            Contrastive loss scalar
        """
        input1 = batch["inputs"]  # [batch, seq_len]
        input2 = batch["inputs_aug"]  # [batch, seq_len] - different example, same task
        puzzle_ids = batch["puzzle_identifiers"]  # [batch]

        # === Online branch (with gradient) ===
        z1 = self.encode(input1, puzzle_ids, self.encoder)  # [batch, hidden_size]
        z1_proj = self.projector(z1)  # [batch, proj_dim]
        z1_pred = self.predictor(z1_proj)  # [batch, proj_dim]

        # === Target branch (stop gradient) ===
        with torch.no_grad():
            z2 = self.encode(input2, puzzle_ids, self.target_encoder)  # [batch, hidden_size]
            z2_proj = self.projector(z2)  # [batch, proj_dim]

        # === Similarity loss ===
        # Normalize
        z1_pred = F.normalize(z1_pred, dim=-1)  # [batch, proj_dim]
        z2_proj = F.normalize(z2_proj, dim=-1)  # [batch, proj_dim]

        # Cosine similarity loss
        # loss = 2 - 2 * cos_sim (range [0, 4])
        loss = 2 - 2 * (z1_pred * z2_proj).sum(dim=-1).mean()

        return loss

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with exponential moving average."""
        for param_online, param_target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_target.data = (
                self.ema_decay * param_target.data +
                (1 - self.ema_decay) * param_online.data
            )


class InfoNCEContrastiveTRM(nn.Module):
    """
    Alternative: InfoNCE loss for contrastive learning.

    Uses negative pairs from the batch for stronger contrastive signal.
    Good when you have large batches with diverse tasks.

    Args:
        trm_model: Base TRM model
        hidden_size: Hidden dimension size
        proj_dim: Projection dimension
        temperature: Temperature parameter for InfoNCE loss
        pool_method: Method to pool z_H
    """

    def __init__(
        self,
        trm_model: nn.Module,
        hidden_size: int = 512,
        proj_dim: int = 256,
        temperature: float = 0.07,
        pool_method: str = 'mean'
    ):
        super().__init__()

        self.encoder = trm_model

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )

        self.temperature = temperature
        self.pool_method = pool_method

    def _pool_latent(self, z_H: torch.Tensor) -> torch.Tensor:
        """Pool z_H to fixed-size representation."""
        if self.pool_method == 'mean':
            return z_H.mean(dim=1)
        elif self.pool_method == 'max':
            return z_H.max(dim=1)[0]
        elif self.pool_method == 'first':
            return z_H[:, 0]
        elif self.pool_method == 'last':
            return z_H[:, -1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

    def encode(
        self,
        inputs: torch.Tensor,
        puzzle_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode inputs to latent representation."""
        batch_dict = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_ids
        }

        carry = self.encoder.initial_carry(batch_dict)
        carry, _ = self.encoder(carry, batch_dict)

        z_H = carry.inner_carry.z_H
        z_pooled = self._pool_latent(z_H)

        return z_pooled

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Positive pairs: same task, different inputs
        Negative pairs: different examples in batch

        Args:
            batch: Dictionary with 'inputs', 'inputs_aug', 'puzzle_identifiers'

        Returns:
            InfoNCE loss scalar
        """
        input1 = batch["inputs"]
        input2 = batch["inputs_aug"]
        puzzle_ids = batch["puzzle_identifiers"]

        # Encode both views
        z1 = self.encode(input1, puzzle_ids)  # [batch, hidden_size]
        z2 = self.encode(input2, puzzle_ids)  # [batch, hidden_size]

        # Project
        z1_proj = self.projector(z1)  # [batch, proj_dim]
        z2_proj = self.projector(z2)  # [batch, proj_dim]

        # Normalize
        z1_proj = F.normalize(z1_proj, dim=-1)
        z2_proj = F.normalize(z2_proj, dim=-1)

        # Compute similarity matrix [batch, batch]
        similarity = torch.mm(z1_proj, z2_proj.t()) / self.temperature

        # Positive pairs: diagonal
        # Negatives: off-diagonal
        batch_size = z1_proj.shape[0]
        labels = torch.arange(batch_size, device=z1_proj.device)

        # Symmetric loss
        loss_12 = F.cross_entropy(similarity, labels)
        loss_21 = F.cross_entropy(similarity.t(), labels)

        return (loss_12 + loss_21) / 2


def create_augmented_batch(
    batch: Dict[str, torch.Tensor],
    augmentation_fn: Optional[callable] = None
) -> Dict[str, torch.Tensor]:
    """
    Create augmented batch for contrastive learning.

    Strategy 1: Use different training examples from same task (preferred)
    Strategy 2: Apply task-preserving transformations

    Args:
        batch: Input batch with 'inputs' and 'puzzle_identifiers'
        augmentation_fn: Optional augmentation function

    Returns:
        Batch with 'inputs_aug' added
    """
    # Check if augmented inputs already provided
    if "inputs_aug" in batch:
        return batch

    # Apply augmentation function if provided
    if augmentation_fn is not None:
        inputs_aug = augmentation_fn(batch["inputs"])
    else:
        # Default: simple copy (assumes dataloader provides different examples)
        inputs_aug = batch["inputs"].clone()

    batch["inputs_aug"] = inputs_aug
    return batch


def augment_arc_grid(
    inputs: torch.Tensor,
    grid_height: int = 30,
    grid_width: int = 30,
    apply_rotation: bool = True,
    apply_flip: bool = True,
    rotation_prob: float = 0.5,
    flip_prob: float = 0.5
) -> torch.Tensor:
    """
    Augment ARC grids with task-preserving transformations.

    WARNING: Only use transformations that preserve task semantics!
    For many ARC tasks, rotation/flip changes the correct answer.
    Safest approach: use different training examples from same task.

    Args:
        inputs: Input tokens [batch, seq_len]
        grid_height: Height of grid
        grid_width: Width of grid
        apply_rotation: Whether to apply rotation
        apply_flip: Whether to apply flipping
        rotation_prob: Probability of rotation
        flip_prob: Probability of flipping

    Returns:
        Augmented inputs [batch, seq_len]
    """
    batch_size = inputs.shape[0]
    inputs_aug = inputs.clone()

    for i in range(batch_size):
        grid = inputs[i].view(grid_height, grid_width)  # Reshape to 2D

        # Random rotation (90, 180, 270 degrees)
        if apply_rotation and torch.rand(1).item() < rotation_prob:
            k = torch.randint(1, 4, (1,)).item()
            grid = torch.rot90(grid, k=k, dims=[0, 1])

        # Random flip
        if apply_flip and torch.rand(1).item() < flip_prob:
            if torch.rand(1).item() > 0.5:
                grid = torch.flip(grid, dims=[0])  # Vertical flip
            else:
                grid = torch.flip(grid, dims=[1])  # Horizontal flip

        inputs_aug[i] = grid.flatten()

    return inputs_aug
