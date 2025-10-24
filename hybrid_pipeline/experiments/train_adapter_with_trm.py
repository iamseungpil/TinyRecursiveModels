"""Phase 1.5: Train adapter so TRM solves HelmARC latents."""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adapters.text_to_latent import TextToLatentAdapter
from adapters.latent_to_text import LatentToTextAdapter
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.functions import load_model_class


class LatentDataset(Dataset):
    def __init__(self, embeddings_path: str, targets_path: str):
        latents = torch.load(embeddings_path)
        self.latent_order = sorted(latents.keys())
        self.latents = latents
        self.targets = torch.load(targets_path)

    def __len__(self):
        return len(self.latent_order)

    def __getitem__(self, idx):
        key = self.latent_order[idx]
        latent = self.latents[key].squeeze(0).to(torch.float32)
        target = self.targets[key]
        puzzle_id = torch.tensor(key, dtype=torch.long)
        return latent, target, puzzle_id


class AugmentedLatentDataset(Dataset):
    """Dataset that uses augmented data with puzzle-level embeddings."""
    def __init__(self, embeddings_path: str, data_path: str):
        import numpy as np

        # Load puzzle-level embeddings
        embeddings = torch.load(embeddings_path)
        self.embeddings = {k: v.squeeze(0).to(torch.float32) for k, v in embeddings.items()}

        # Load augmented data
        train_dir = Path(data_path) / "train"
        puzzle_ids_path = train_dir / "all__puzzle_identifiers.npy"
        inputs_path = train_dir / "all__inputs.npy"
        labels_path = train_dir / "all__labels.npy"

        all_puzzle_ids = np.load(puzzle_ids_path)
        self.inputs = np.load(inputs_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

        # Filter to keep only samples with available embeddings
        valid_indices = [i for i, pid in enumerate(all_puzzle_ids) if int(pid) in self.embeddings]
        self.puzzle_ids = all_puzzle_ids[valid_indices]
        self.valid_indices = valid_indices

        print(f"📊 Loaded {len(self.puzzle_ids):,} augmented samples from {len(self.embeddings)} puzzles")
        print(f"⚠️  Filtered out {len(all_puzzle_ids) - len(self.puzzle_ids):,} samples with missing embeddings")

    def __len__(self):
        return len(self.puzzle_ids)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        puzzle_id = int(self.puzzle_ids[idx])

        # Get embedding for this puzzle (same embedding for all augmentations)
        latent = self.embeddings[puzzle_id]

        # Get augmented input/label
        inputs = torch.from_numpy(self.inputs[actual_idx].copy())
        labels = torch.from_numpy(self.labels[actual_idx].copy())
        puzzle_id_tensor = torch.tensor(puzzle_id, dtype=torch.long)

        return latent, inputs, labels, puzzle_id_tensor


def main():
    parser = argparse.ArgumentParser(description="Adapter training with TRM supervision")
    parser.add_argument("--latents_path", type=str, required=True)
    parser.add_argument("--targets_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset root to fetch metadata")
    parser.add_argument("--trm_checkpoint", type=str, required=True)
    parser.add_argument("--adapter_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/data/hybrid_training/phase2_adapter.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_augmented", action="store_true",
                        help="Use augmented dataset instead of puzzle-level data")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_augmented:
        print("📊 Using augmented dataset")
        dataset = AugmentedLatentDataset(args.latents_path, args.data_path)
    else:
        print("📊 Using puzzle-level dataset")
        if args.targets_path is None:
            raise ValueError("--targets_path is required when not using augmented data")
        dataset = LatentDataset(args.latents_path, args.targets_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    metadata_path = Path(args.data_path) / "train" / "dataset.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    seq_len = metadata["seq_len"]
    vocab_size = metadata["vocab_size"]

    encoder = TextToLatentAdapter(
        llm_hidden_size=dataset[0][0].shape[-1],
        trm_hidden_size=512,
        trm_seq_len=seq_len,
        use_position_embeddings=True,
    ).to(device)

    decoder = LatentToTextAdapter(
        trm_hidden_size=512,
        trm_seq_len=seq_len,
        llm_hidden_size=dataset[0][0].shape[-1],
        use_attention_pooling=True,
    ).to(device)

    checkpoint = torch.load(args.adapter_checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["text_to_latent"], strict=False)
    decoder.load_state_dict(checkpoint["latent_to_text"], strict=False)

    trm_config = {
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_puzzle_identifiers": metadata.get("num_puzzle_identifiers", 1),
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 1,
        "L_layers": 4,
        "hidden_size": 512,
        "expansion": 4.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
    }

    trm_model = TinyRecursiveReasoningModel_ACTV1(trm_config)
    loss_head_cls = load_model_class("losses@ACTLossHead")
    trm_model = loss_head_cls(trm_model, loss_type="stablemax_cross_entropy").to(device)
    trm_model.load_state_dict(torch.load(args.trm_checkpoint, map_location=device), strict=False)
    trm_model.eval()
    for param in trm_model.parameters():
        param.requires_grad = False

    trainable_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        total_trm_loss = 0.0
        total_recon_loss = 0.0
        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if args.use_augmented:
                latents, inputs, labels, puzzle_ids = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
            else:
                latents, targets, puzzle_ids = batch_data
                inputs = labels = targets.to(device)

            latents = latents.to(device)
            puzzle_ids = puzzle_ids.to(device)

            z_H, z_L = encoder(latents)
            batch = {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_ids}
            carry = trm_model.initial_carry(batch)
            carry.inner_carry.z_H = z_H
            carry.inner_carry.z_L = z_L
            carry, trm_loss, metrics, _, _ = trm_model(carry=carry, batch=batch, return_keys=[])

            decoded_latents = decoder(carry.inner_carry.z_H, carry.inner_carry.z_L)
            recon_loss = F.mse_loss(decoded_latents.float(), latents.float())
            combined_loss = trm_loss + recon_loss

            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            batch_size = latents.size(0)
            total_loss += combined_loss.item() * batch_size
            total_trm_loss += trm_loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size

        avg_loss = total_loss / len(dataset)
        avg_trm = total_trm_loss / len(dataset)
        avg_recon = total_recon_loss / len(dataset)
        print(f"Epoch {epoch+1}: trm_loss={avg_trm:.6f} recon_loss={avg_recon:.6f} total_loss={avg_loss:.6f}")

    torch.save({
        "text_to_latent": encoder.state_dict(),
        "latent_to_text": decoder.state_dict(),
    }, args.output_path)
    print(f"✅ Adapter (with TRM supervision) saved to {args.output_path}")


if __name__ == "__main__":
    main()
