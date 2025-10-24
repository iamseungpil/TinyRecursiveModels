"""
VAE-based Text-Latent Adapter

Adds reconstruction loss to ensure adapters preserve information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAETextToLatentAdapter(nn.Module):
    """
    VAE encoder: LLM → TRM latent space

    Adds variational bottleneck with mean/variance parameterization.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        trm_hidden_size: int,
        trm_seq_len: int,
        puzzle_emb_len: int = 0,
        use_position_embeddings: bool = True,
        latent_dim: int = 1024
    ):
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.trm_hidden_size = trm_hidden_size
        self.trm_seq_len = trm_seq_len + puzzle_emb_len
        self.latent_dim = latent_dim
        self.use_position_embeddings = use_position_embeddings

        # Encoder to latent parameters
        self.fc_mu = nn.Linear(llm_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(llm_hidden_size, latent_dim)

        # Position embeddings
        if use_position_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.randn(self.trm_seq_len, latent_dim) * 0.02
            )
        else:
            self.position_embeddings = None

        # Projection to TRM space
        self.to_trm = nn.Linear(latent_dim, 2 * trm_hidden_size)

        # Initialize
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        Args:
            mu: [batch, latent_dim] mean
            logvar: [batch, latent_dim] log variance

        Returns:
            z: [batch, latent_dim] sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        llm_hidden: torch.Tensor,
        deterministic: bool = False,
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode LLM hidden state to TRM latent.

        Args:
            llm_hidden: [batch, llm_dim] LLM hidden state
            deterministic: If True, use mu (no sampling) for inference

        Returns:
            z_H: [batch, seq_len, trm_dim] high-level carry
            z_L: [batch, seq_len, trm_dim] low-level carry
            mu: [batch, latent_dim] latent mean
            logvar: [batch, latent_dim] latent log variance
        """
        batch_size = llm_hidden.shape[0]

        # Encode to latent parameters
        mu = self.fc_mu(llm_hidden)
        logvar = self.fc_logvar(llm_hidden)

        # Sample or use mean
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        # Position-aware expansion
        if self.use_position_embeddings:
            seq_features = z.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:
            seq_features = z.unsqueeze(1).expand(-1, self.trm_seq_len, -1)

        # Project to TRM space
        trm_latent = self.to_trm(seq_features)
        z_H, z_L = torch.chunk(trm_latent, 2, dim=-1)

        return z_H, z_L, mu, logvar


class VAELatentToTextAdapter(nn.Module):
    """
    VAE decoder: TRM latent → LLM space

    Reconstructs LLM hidden state for reconstruction loss.
    """

    def __init__(
        self,
        trm_hidden_size: int,
        trm_seq_len: int,
        llm_hidden_size: int,
        puzzle_emb_len: int = 0,
        use_attention_pooling: bool = True,
        latent_dim: int = 1024
    ):
        super().__init__()

        self.trm_hidden_size = trm_hidden_size
        self.trm_seq_len = trm_seq_len + puzzle_emb_len
        self.llm_hidden_size = llm_hidden_size
        self.latent_dim = latent_dim
        self.use_attention_pooling = use_attention_pooling

        # Pooling
        if use_attention_pooling:
            from .latent_to_text import AttentionPooling
            self.pool_H = AttentionPooling(trm_hidden_size)
            self.pool_L = AttentionPooling(trm_hidden_size)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

        # Decode to latent
        self.to_latent = nn.Sequential(
            nn.Linear(2 * trm_hidden_size, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

        # Decode to LLM space
        self.to_llm = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, llm_hidden_size)
        )

        # Initialize
        for module in [self.to_latent, self.to_llm]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode TRM latent to LLM hidden state.

        Args:
            z_H: [batch, seq_len, trm_dim] high-level carry
            z_L: [batch, seq_len, trm_dim] low-level carry

        Returns:
            llm_hidden: [batch, llm_dim] reconstructed LLM hidden
        """
        # Pool
        if self.use_attention_pooling:
            z_H_pooled = self.pool_H(z_H)
            z_L_pooled = self.pool_L(z_L)
        else:
            z_H_pooled = self.pool(z_H.transpose(1, 2)).squeeze(-1)
            z_L_pooled = self.pool(z_L.transpose(1, 2)).squeeze(-1)

        # Concatenate
        z_combined = torch.cat([z_H_pooled, z_L_pooled], dim=-1)

        # Decode
        latent = self.to_latent(z_combined)
        llm_hidden = self.to_llm(latent)

        return llm_hidden


def vae_loss(
    llm_hidden: torch.Tensor,
    llm_reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KL divergence).

    Args:
        llm_hidden: [batch, llm_dim] original LLM hidden state
        llm_reconstructed: [batch, llm_dim] reconstructed LLM hidden state
        mu: [batch, latent_dim] latent mean
        logvar: [batch, latent_dim] latent log variance
        beta: Weight for KL loss (beta-VAE)

    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(llm_reconstructed, llm_hidden, reduction='mean')

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
