"""
CNN-based Puzzle Embedding Predictor

Predicts TRM's 512-dim puzzle embedding from input grid alone.
This enables unseen task inference without learned puzzle identifiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block for grid feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PuzzleEmbeddingCNN(nn.Module):
    """
    CNN encoder: ARC grid (H x W) → 512-dim puzzle embedding

    Architecture:
    1. Embedding layer (vocab_size=12 → channels)
    2. Conv layers with residual blocks
    3. Global average pooling
    4. FC layer → 512-dim

    Args:
        vocab_size: ARC color vocabulary (default: 12 for 0-9 + padding/mask)
        embedding_dim: Initial embedding dimension
        hidden_channels: CNN hidden channels
        num_blocks: Number of residual blocks
        output_dim: Output dimension (512 for TRM puzzle embedding)
    """

    def __init__(
        self,
        vocab_size: int = 12,
        embedding_dim: int = 64,
        hidden_channels: int = 256,
        num_blocks: int = 4,
        output_dim: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)

        # Initial conv to expand channels
        self.conv_in = nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, H, W) long tensor of color indices

        Returns:
            embedding: (batch, output_dim) predicted puzzle embedding
        """
        batch_size, H, W = grid.shape

        # Embed tokens: (B, H, W) → (B, H, W, emb_dim)
        x = self.token_embed(grid)

        # Permute to conv format: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Initial conv
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Global pooling: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        # Output projection: (B, C) → (B, output_dim)
        embedding = self.fc(x)

        return embedding


class PuzzleEmbeddingLoss(nn.Module):
    """
    Combined loss for puzzle embedding prediction.

    1. MSE loss: Direct reconstruction of learned embedding
    2. Cosine similarity loss: Preserve directional information
    """

    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(
        self,
        pred_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_embedding: (batch, 512) predicted embedding
            target_embedding: (batch, 512) learned TRM embedding

        Returns:
            dict with 'loss', 'mse', 'cosine' components
        """
        # MSE loss
        mse_loss = F.mse_loss(pred_embedding, target_embedding)

        # Cosine similarity loss (maximize similarity = minimize negative)
        cosine_sim = F.cosine_similarity(pred_embedding, target_embedding, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        return {
            'loss': total_loss,
            'mse': mse_loss,
            'cosine': cosine_loss,
            'cosine_similarity': cosine_sim.mean()
        }


if __name__ == "__main__":
    # Test model
    model = PuzzleEmbeddingCNN(
        vocab_size=12,
        embedding_dim=64,
        hidden_channels=256,
        num_blocks=4,
        output_dim=512
    )

    # Test forward pass
    batch_size = 4
    grid = torch.randint(0, 10, (batch_size, 30, 30))  # Random ARC grid
    embedding = model(grid)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {grid.shape}")
    print(f"Output shape: {embedding.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test loss
    target = torch.randn(batch_size, 512)
    loss_fn = PuzzleEmbeddingLoss()
    losses = loss_fn(embedding, target)
    print(f"\nLoss components: {losses}")
