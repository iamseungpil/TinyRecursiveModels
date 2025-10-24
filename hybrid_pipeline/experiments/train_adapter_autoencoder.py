"""Train Text↔Latent adapters as an autoencoder using precomputed LLM embeddings."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adapters.text_to_latent import TextToLatentAdapter
from adapters.latent_to_text import LatentToTextAdapter


class EmbeddingDataset(Dataset):
    """Dataset that matches embeddings to augmented puzzle data."""
    def __init__(self, embeddings: Dict[int, torch.Tensor], data_path: str):
        import numpy as np

        # Load puzzle identifiers from augmented dataset
        puzzle_ids_path = Path(data_path) / "train" / "all__puzzle_identifiers.npy"
        all_puzzle_ids = np.load(puzzle_ids_path)

        # Store embeddings dict
        self.embeddings = {k: v.squeeze(0).to(torch.float32) for k, v in embeddings.items()}

        # Filter to keep only samples with available embeddings
        valid_indices = [i for i, pid in enumerate(all_puzzle_ids) if int(pid) in self.embeddings]
        self.puzzle_ids = all_puzzle_ids[valid_indices]

        print(f"📊 Loaded {len(self.puzzle_ids)} augmented samples from {len(self.embeddings)} puzzles")
        print(f"⚠️  Filtered out {len(all_puzzle_ids) - len(self.puzzle_ids)} samples with missing embeddings")

    def __len__(self) -> int:
        return len(self.puzzle_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        puzzle_id = int(self.puzzle_ids[idx])
        # Return the embedding for this puzzle (same embedding used for all augmentations)
        return self.embeddings[puzzle_id]


class AdapterAutoencoder(nn.Module):
    def __init__(self, encoder: TextToLatentAdapter, decoder: LatentToTextAdapter):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        z_H, z_L = self.encoder(llm_hidden)
        reconstructed = self.decoder(z_H, z_L)
        return reconstructed


def load_metadata(data_path: str) -> Dict[str, int]:
    metadata_path = Path(data_path) / "train" / "dataset.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return {
        "seq_len": metadata["seq_len"],
        "trm_hidden_size": metadata.get("hidden_size", 512),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapter autoencoder training")
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to TRM dataset (for seq_len metadata)")
    parser.add_argument("--output_dir", type=str, default="/data/hybrid_training/phase1_autoencoder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_attention_pooling", action="store_true")
    parser.add_argument("--save_name", type=str, default="adapter_autoencoder.pt")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("📥 Loading precomputed embeddings...")
    embeddings = torch.load(args.embeddings_path)
    dataset = EmbeddingDataset(embeddings, args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    metadata = load_metadata(args.data_path)
    seq_len = metadata["seq_len"]
    trm_hidden_size = metadata["trm_hidden_size"]
    llm_hidden_size = dataset[0].shape[-1]

    print(f"📐 LLM hidden size: {llm_hidden_size}")
    print(f"📐 TRM seq len: {seq_len}, hidden size: {trm_hidden_size}")

    encoder = TextToLatentAdapter(
        llm_hidden_size=llm_hidden_size,
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=seq_len,
        use_position_embeddings=True,
        use_cross_attention=False,
    ).to(device)

    decoder = LatentToTextAdapter(
        trm_hidden_size=trm_hidden_size,
        trm_seq_len=seq_len,
        llm_hidden_size=llm_hidden_size,
        use_attention_pooling=args.use_attention_pooling,
    ).to(device)

    model = AdapterAutoencoder(encoder, decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting autoencoder training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = F.mse_loss(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}: recon_loss={avg_loss:.6f}")

    checkpoint = {
        "text_to_latent": encoder.state_dict(),
        "latent_to_text": decoder.state_dict(),
        "llm_hidden_size": llm_hidden_size,
        "trm_seq_len": seq_len,
        "trm_hidden_size": trm_hidden_size,
    }
    checkpoint_path = Path(args.output_dir) / args.save_name
    torch.save(checkpoint, checkpoint_path)

    print(f"✅ Adapter autoencoder saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
