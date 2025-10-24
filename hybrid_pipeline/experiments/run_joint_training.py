"""
Joint Training: LLaMA + TRM with Adapters

Combines text reasoning (LLaMA) with grid generation (TRM) through thin adapters.
Uses TRM's official API (initial_carry + forward) instead of manual carry creation.
"""

import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

# Import from hybrid_pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpt_oss_port.llm import LatentOutput, TextReasoningModule
from gpt_oss_port.planner import ARCPlanner
from gpt_oss_port.verifier import GridVerifier
from gpt_oss_port.grid_utils import grid_to_string
from gpt_oss_port.loss_guided_generation import LossGuidedGenerator
from adapters.text_to_latent import TextToLatentAdapter
from adapters.latent_to_text import LatentToTextAdapter
from adapters.vae_adapter import VAETextToLatentAdapter, VAELatentToTextAdapter, vae_loss
from adapters.feedback_formatter import tokens_to_grid

# Import TRM from existing models/ module
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.functions import load_model_class
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class JointModelConfig:
    """Configuration for joint LLaMA + TRM training."""

    # Data
    data_path: str
    output_dir: str = "/data/trm/joint_training"

    # LLaMA
    llama_model: str = "unsloth/gpt-oss-mxfp4-20b"
    llama_device: str = "cuda"
    llama_frozen: bool = True
    llama_torch_dtype: str = "bfloat16"
    llama_device_map: str = "auto"
    llama_trust_remote_code: bool = True
    llama_use_fast_tokenizer: bool = False

    # Phase 1: Pre-computed embeddings (for HelmARC)
    use_precomputed_embeddings: bool = False
    embeddings_path: Optional[str] = None

    # TRM
    trm_checkpoint: Optional[str] = None
    trm_hidden_size: int = 512
    trm_num_heads: int = 8
    trm_expansion: float = 4.0
    trm_H_cycles: int = 3
    trm_L_cycles: int = 6
    trm_L_layers: int = 4
    trm_halt_max_steps: int = 16

    # Training
    batch_size: int = 1
    max_attempts: int = 16
    epochs: int = 10
    lr: float = 1e-4
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01

    # Recursive solving (Phase 3)
    enable_repulsion: bool = False
    repulsion_weight: float = 0.5

    # Adapter improvements
    use_attention_pooling: bool = True  # Use attention pooling in LatentToText (default: True)
    use_vae_adapter: bool = False  # Use VAE adapter with reconstruction loss
    vae_beta: float = 0.1  # KL loss weight for VAE
    use_cross_attention_bridge: bool = False  # Enable cross-attention bridge (LLM seq → TRM slots)
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.0
    adapter_checkpoint: Optional[str] = None

    # Checkpoint loading for full model
    load_checkpoint: Optional[str] = None

    # LLM improvements
    use_chat_template: bool = False  # Use apply_chat_template (for GPT-OSS)
    use_loss_guided_generation: bool = False  # Use loss-guided representation selection
    loss_guide_num_candidates: int = 3  # Number of candidates for loss-guided generation
    loss_guide_high_threshold: float = 2.0  # Loss threshold for exploration
    loss_guide_low_threshold: float = 0.5  # Loss threshold for exploitation

    # Logging
    project_name: str = "arc-hybrid-joint"
    run_name: Optional[str] = None
    log_interval: int = 5
    checkpoint_interval: int = 100

    # Hardware
    device: str = "cuda:0"
    seed: int = 42


class HybridModel(nn.Module):
    """
    Hybrid LLaMA + TRM model with adapters.

    Uses TRM's official API: initial_carry() + forward()
    """

    def __init__(
        self,
        llm_module: TextReasoningModule,
        trm_model: nn.Module,
        text_to_latent: nn.Module,
        latent_to_text: nn.Module,
        config: JointModelConfig,
        loss_guided_generator: Optional[LossGuidedGenerator] = None
    ):
        super().__init__()

        self.llm = llm_module
        self.trm = trm_model
        self.text_to_latent = text_to_latent
        self.latent_to_text = latent_to_text
        self.config = config
        self.loss_guided_generator = loss_guided_generator


def create_hybrid_model(config: JointModelConfig, metadata: Dict[str, Any]) -> HybridModel:
    """Create joint LLaMA + TRM model with adapters."""

    print(f"📦 Loading LLaMA: {config.llama_model}")
    torch_dtype = config.llama_torch_dtype
    if isinstance(torch_dtype, str):
        if torch_dtype.lower() == "auto":
            torch_dtype_value = None
        else:
            torch_dtype_value = getattr(torch, torch_dtype)
    else:
        torch_dtype_value = torch_dtype

    llm_module = TextReasoningModule(
        model_name=config.llama_model,
        freeze=config.llama_frozen,
        device=config.llama_device,
        extract_full_sequence=config.use_cross_attention_bridge,
        torch_dtype=torch_dtype_value,
        device_map=config.llama_device_map,
        trust_remote_code=config.llama_trust_remote_code,
        use_fast_tokenizer=config.llama_use_fast_tokenizer,
    )
    llama_hidden_size = llm_module.hidden_size

    print("🔧 Creating TRM model...")
    trm_config_dict = {
        "batch_size": config.batch_size,
        "seq_len": metadata["seq_len"],
        "vocab_size": metadata["vocab_size"],
        "num_puzzle_identifiers": metadata["num_puzzle_identifiers"],
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "H_cycles": config.trm_H_cycles,
        "L_cycles": config.trm_L_cycles,
        "H_layers": 1,
        "L_layers": config.trm_L_layers,
        "hidden_size": config.trm_hidden_size,
        "expansion": config.trm_expansion,
        "num_heads": config.trm_num_heads,
        "pos_encodings": "rope",
        "halt_max_steps": config.trm_halt_max_steps,
        "halt_exploration_prob": 0.1,
    }

    # Create model
    trm_model = TinyRecursiveReasoningModel_ACTV1(trm_config_dict)

    if config.trm_checkpoint:
        print(f"📥 Loading TRM checkpoint: {config.trm_checkpoint}")
        checkpoint = torch.load(config.trm_checkpoint, map_location=config.device)
        if "model_state_dict" in checkpoint:
            trm_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            trm_model.load_state_dict(checkpoint, strict=False)

    # Wrap with loss head (following pretrain.py pattern)
    loss_head_cls = load_model_class("losses@ACTLossHead")
    trm_model = loss_head_cls(trm_model, loss_type="stablemax_cross_entropy")

    # Move to device
    trm_model = trm_model.to(config.device)

    print("🔌 Creating adapters...")
    if config.use_cross_attention_bridge and config.use_vae_adapter:
        raise ValueError("Cross-attention bridge is not supported with the VAE adapter. Disable one of the features.")
    # Choose adapter type based on config
    if config.use_vae_adapter:
        print("  Using VAE adapters with reconstruction loss")
        text_to_latent = VAETextToLatentAdapter(
            llm_hidden_size=llama_hidden_size,
            trm_hidden_size=config.trm_hidden_size,
            trm_seq_len=metadata["seq_len"],
            puzzle_emb_len=metadata.get("puzzle_emb_len", 0),
            use_position_embeddings=True
        ).to(device=config.device, dtype=torch.bfloat16)

        latent_to_text = VAELatentToTextAdapter(
            trm_hidden_size=config.trm_hidden_size,
            trm_seq_len=metadata["seq_len"],
            llm_hidden_size=llama_hidden_size,
            puzzle_emb_len=metadata.get("puzzle_emb_len", 0),
            use_attention_pooling=config.use_attention_pooling
        ).to(device=config.device, dtype=torch.bfloat16)
    else:
        print("  Using standard adapters")
        text_to_latent = TextToLatentAdapter(
            llm_hidden_size=llama_hidden_size,
            trm_hidden_size=config.trm_hidden_size,
            trm_seq_len=metadata["seq_len"],
            use_cross_attention=config.use_cross_attention_bridge,
            cross_attention_layers=config.cross_attention_layers,
            cross_attention_heads=config.cross_attention_heads,
            cross_attention_dropout=config.cross_attention_dropout,
        ).to(device=config.device, dtype=torch.bfloat16)

        latent_to_text = LatentToTextAdapter(
            trm_hidden_size=config.trm_hidden_size,
            trm_seq_len=metadata["seq_len"],
            llm_hidden_size=llama_hidden_size,
            use_attention_pooling=config.use_attention_pooling
        ).to(device=config.device, dtype=torch.bfloat16)

    # Create loss-guided generator if enabled
    loss_guided_generator = None
    if config.use_loss_guided_generation:
        print("🎯 Creating loss-guided generator...")
        loss_guided_generator = LossGuidedGenerator(
            llm_module=llm_module,
            text_to_latent=text_to_latent,
            trm_model=trm_model,
            high_loss_threshold=config.loss_guide_high_threshold,
            low_loss_threshold=config.loss_guide_low_threshold,
            num_candidates=config.loss_guide_num_candidates,
            temperature=0.8
        )

    hybrid_model = HybridModel(
        llm_module=llm_module,
        trm_model=trm_model,
        text_to_latent=text_to_latent,
        latent_to_text=latent_to_text,
        config=config,
        loss_guided_generator=loss_guided_generator
    )

    # Load adapter weights if provided
    if config.adapter_checkpoint is not None:
        print(f"📥 Loading adapter checkpoint: {config.adapter_checkpoint}")
        checkpoint = torch.load(config.adapter_checkpoint, map_location=config.device)

        # Extract state dict from checkpoint
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load text_to_latent adapter
        text_to_latent_state = {k.replace("text_to_latent.", ""): v
                                for k, v in state_dict.items()
                                if k.startswith("text_to_latent.")}
        if text_to_latent_state:
            hybrid_model.text_to_latent.load_state_dict(text_to_latent_state, strict=False)
            print(f"  ✓ Loaded text_to_latent adapter ({len(text_to_latent_state)} keys)")

        # Load latent_to_text adapter
        latent_to_text_state = {k.replace("latent_to_text.", ""): v
                                for k, v in state_dict.items()
                                if k.startswith("latent_to_text.")}
        if latent_to_text_state:
            hybrid_model.latent_to_text.load_state_dict(latent_to_text_state, strict=False)
            print(f"  ✓ Loaded latent_to_text adapter ({len(latent_to_text_state)} keys)")

        # Optionally load TRM weights (only if not already loaded from trm_checkpoint)
        if not config.trm_checkpoint:
            trm_state = {k.replace("trm.", ""): v
                        for k, v in state_dict.items()
                        if k.startswith("trm.")}
            if trm_state:
                # Remove loss head wrapper to access base model
                if hasattr(hybrid_model.trm, 'model'):
                    hybrid_model.trm.model.load_state_dict(trm_state, strict=False)
                else:
                    hybrid_model.trm.load_state_dict(trm_state, strict=False)
                print(f"  ✓ Loaded TRM weights from adapter checkpoint ({len(trm_state)} keys)")

        print(f"✅ Adapter checkpoint loaded successfully")

        # Ensure adapters match TRM dtype (bfloat16)
        hybrid_model.text_to_latent = hybrid_model.text_to_latent.to(dtype=torch.bfloat16)
        hybrid_model.latent_to_text = hybrid_model.latent_to_text.to(dtype=torch.bfloat16)
        print(f"  ✓ Converted adapters to bfloat16")

    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"✅ Hybrid model created with {trainable_params:,} trainable parameters")

    return hybrid_model


def extract_problem_data(batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Convert a PuzzleDataset batch into lightweight ARC problem descriptions."""

    batch_size = batch["inputs"].shape[0]
    problems: List[Dict[str, Any]] = []

    for i in range(batch_size):
        input_grid = tokens_to_grid(batch["inputs"][i])
        target_grid = tokens_to_grid(batch["labels"][i])

        problem_description = (
            "Solve this ARC puzzle from a single example.\n\n"
            "Example 1:\n"
            f"Input grid:\n{grid_to_string(input_grid)}\n"
            "Predict the correct output grid that matches the transformation."
        )

        problems.append(
            {
                "description": problem_description,
                "input_grid": input_grid,
                "target_grid": target_grid,
            }
        )

    return problems


def train_step(
    model: HybridModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    precomputed_embeddings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Single training step with self-correction loop.

    Uses TRM's official API: initial_carry() + forward()

    Args:
        precomputed_embeddings: Optional pre-computed LLaMA embeddings for Phase 1 training
    """
    model.train()
    device = model.config.device

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_size = batch["inputs"].shape[0]
    target_tokens = batch["labels"]

    # Extract problem data from batch
    problems = extract_problem_data(batch)

    planners: List[ARCPlanner] = []
    feedback_history: List[Optional[str]] = []
    for problem in problems:
        planner = ARCPlanner(
            llm_module=model.llm,
            max_attempts=model.config.max_attempts
        )
        planner.reset()
        planners.append(planner)
        feedback_history.append(None)

    total_loss = 0.0
    attempt_count = 0
    correct_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    latent_prefix_list: List[Optional[torch.Tensor]] = [None] * batch_size

    # Collect GridVerifier metrics across batch
    batch_ver_metrics = {
        "shape_match": [],
        "cell_accuracy": []
    }
    trm_steps_total = 0

    # Self-correction loop
    for attempt in range(model.config.max_attempts):
        attempt_count += 1

        # Step 1: LLaMA generates reasoning for each problem
        z_init_list: List[torch.Tensor] = []
        reasoning_texts: List[str] = []
        hidden_sequence_list: List[Optional[torch.Tensor]] = []
        attention_mask_list: List[Optional[torch.Tensor]] = []

        for i, problem in enumerate(problems):
            if correct_mask[i]:
                # Already correct, skip
                z_init_list.append(torch.zeros(model.llm.hidden_size, device=device))
                reasoning_texts.append("")
                hidden_sequence_list.append(None)
                attention_mask_list.append(None)
                continue

            # Phase 1: Use pre-computed embeddings (skip LLaMA inference)
            if precomputed_embeddings is not None and attempt == 0:
                # Get puzzle_identifier from batch
                puzzle_id = batch["puzzle_identifiers"][i].item()
                if puzzle_id in precomputed_embeddings:
                    embedding = precomputed_embeddings[puzzle_id].to(device).squeeze(0).to(torch.bfloat16)
                    latent_output = LatentOutput(
                        final_hidden=embedding,
                        generated_text="[Pre-computed embedding]",
                        hidden_sequence=None,
                        attention_mask=None,
                    )
                else:
                    # Fallback: generate normally
                    latent_output = planners[i].plan_step(
                        problem_description=problem["description"],
                        feedback=None,
                        latent_prefix=None,
                        use_chat_template=model.config.use_chat_template
                    )
            else:
                # Phase 2/3: Loss-guided generation, repulsion, or normal generation
                if model.config.use_loss_guided_generation and model.loss_guided_generator is not None:
                    # Loss-guided generation (exploration/exploitation)
                    from adapters.feedback_formatter import format_problem_text
                    history_str = "\n".join(planners[i].history) if planners[i].history else ""
                    full_prompt = format_problem_text(problem["description"], history_str)

                    latent_output = model.loss_guided_generator.generate_with_loss_guidance(
                        problem_text=full_prompt,
                        batch=batch,
                        max_length=128,
                        use_chat_template=model.config.use_chat_template
                    )
                elif model.config.enable_repulsion and latent_prefix_list[i] is not None:
                    # Repulsion-based generation (avoid failed latent)
                    # Update planner history to maintain self-correction context
                    planners[i].attempt_count += 1
                    if feedback_history[i]:
                        planners[i].history.append(feedback_history[i])

                    # Format problem with history for self-correction context
                    from adapters.feedback_formatter import format_problem_text
                    history_str = "\n".join(planners[i].history) if planners[i].history else ""
                    full_prompt = format_problem_text(problem["description"], history_str)

                    latent_output = model.llm.generate_with_repulsion(
                        problem_description=full_prompt,
                        failed_latent=latent_prefix_list[i],
                        repulsion_weight=model.config.repulsion_weight
                    )
                else:
                    # Standard generation with optional latent prefix
                    latent_output = planners[i].plan_step(
                        problem_description=problem["description"],
                        feedback=feedback_history[i],
                        latent_prefix=latent_prefix_list[i],
                        use_chat_template=model.config.use_chat_template
                    )

            feedback_history[i] = None
            z_init_list.append(latent_output.final_hidden.to(device))
            reasoning_texts.append(latent_output.generated_text)
            hidden_sequence_list.append(
                latent_output.hidden_sequence.to(device) if latent_output.hidden_sequence is not None else None
            )
            attention_mask_list.append(
                latent_output.attention_mask.to(device) if latent_output.attention_mask is not None else None
            )

        z_init_batch = torch.stack(z_init_list).to(dtype=torch.bfloat16)

        # Prepare optional LLM hidden sequences for cross-attention bridge
        if any(seq is None for seq in hidden_sequence_list):
            llm_hidden_sequence_batch = None
            llm_attention_mask_batch = None
        else:
            valid_sequences = [seq for seq in hidden_sequence_list if seq is not None]
            seq_lengths = [seq.shape[0] for seq in valid_sequences]
            hidden_dim = valid_sequences[0].shape[-1] if valid_sequences else model.llm.hidden_size
            max_seq_len = max(seq_lengths) if seq_lengths else 1

            llm_hidden_sequence_batch = torch.zeros(
                batch_size,
                max_seq_len,
                hidden_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            llm_attention_mask_batch = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

            for idx, seq in enumerate(hidden_sequence_list):
                if seq is None:
                    continue
                length = seq.shape[0]
                llm_hidden_sequence_batch[idx, :length] = seq.to(dtype=torch.bfloat16)
                mask = attention_mask_list[idx]
                if mask is None:
                    llm_attention_mask_batch[idx, :length] = True
                else:
                    llm_attention_mask_batch[idx, :mask.shape[0]] = mask

        # Step 2: Adapter projects LLaMA → TRM
        # Handle VAE vs standard adapters
        adapter_kwargs = {}
        if llm_hidden_sequence_batch is not None:
            adapter_kwargs["llm_hidden_sequence"] = llm_hidden_sequence_batch
            adapter_kwargs["attention_mask"] = llm_attention_mask_batch

        if model.config.use_vae_adapter:
            z_H, z_L, mu, logvar = model.text_to_latent(z_init_batch, deterministic=False, **adapter_kwargs)
            # Store mu, logvar for VAE loss computation
            vae_mu_batch = mu
            vae_logvar_batch = logvar
        else:
            z_H, z_L = model.text_to_latent(z_init_batch, **adapter_kwargs)
            vae_mu_batch = None
            vae_logvar_batch = None

        # Step 3: Run TRM forward using official API (initial_carry + forward)
        carry = model.trm.initial_carry(batch)

        # Ensure z_H, z_L match TRM dtype/device
        z_H = z_H.to(device=device, dtype=model.trm.model.inner.forward_dtype)
        z_L = z_L.to(device=device, dtype=model.trm.model.inner.forward_dtype)

        carry.inner_carry.z_H = z_H
        carry.inner_carry.z_L = z_L

        trm_steps = 0
        while True:
            carry, loss, metrics, preds, all_finish = model.trm(
                carry=carry,
                batch=batch,
                return_keys=["logits"]
            )
            trm_steps += 1

            if all_finish or trm_steps >= model.config.trm_halt_max_steps:
                break

        logits = preds.get("logits", carry.current_data.get("logits"))
        if logits is None:
            # Fallback: extract from model output
            print(f"Warning: logits not in preds, using loss computation")
            logits = torch.zeros_like(target_tokens).unsqueeze(-1).expand(-1, -1, batch["inputs"].max().item() + 1)

        # Step 4: Add VAE reconstruction loss if using VAE adapters
        vae_recon_loss = torch.tensor(0.0, device=device)
        vae_kl_loss = torch.tensor(0.0, device=device)
        if model.config.use_vae_adapter and vae_mu_batch is not None:
            # Reconstruct LLM hidden state from TRM latent
            llm_reconstructed = model.latent_to_text(
                carry.inner_carry.z_H,
                carry.inner_carry.z_L
            )

            # Compute VAE loss
            vae_total, vae_recon, vae_kl = vae_loss(
                llm_hidden=z_init_batch,
                llm_reconstructed=llm_reconstructed,
                mu=vae_mu_batch,
                logvar=vae_logvar_batch,
                beta=model.config.vae_beta
            )

            # Add VAE loss to total loss
            total_loss += loss + vae_total
            vae_recon_loss = vae_recon
            vae_kl_loss = vae_kl
        else:
            total_loss += loss

        trm_steps_total += trm_steps

        # Step 5: Verify predictions and generate feedback
        pred_tokens = logits.argmax(dim=-1) if logits.dim() > 2 else logits

        for i in range(batch_size):
            if correct_mask[i]:
                continue

            pred_grid = tokens_to_grid(pred_tokens[i])
            verifier = GridVerifier(target_grid=problems[i]["target_grid"])

            success, feedback, ver_metrics = verifier.verify_grid(pred_grid)

            correct_mask[i] = success

            # Collect metrics for this batch item
            batch_ver_metrics["shape_match"].append(ver_metrics["shape_match"])
            batch_ver_metrics["cell_accuracy"].append(ver_metrics["cell_accuracy"])
            feedback_history[i] = feedback if not success else "Correct!"

        # Early stopping if all correct
        if correct_mask.all():
            break

        # Step 6: Generate feedback latent for next attempt
        latent_prefix_batch = model.latent_to_text(
            carry.inner_carry.z_H,
            carry.inner_carry.z_L
        )
        latent_prefix_list = [latent_prefix_batch[i] for i in range(batch_size)]

    # Average loss
    avg_loss = total_loss / attempt_count

    # Backward
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()
    optimizer.zero_grad()

    # Aggregate GridVerifier metrics
    avg_shape_match = np.mean(batch_ver_metrics["shape_match"]) if batch_ver_metrics["shape_match"] else 0.0
    avg_cell_accuracy = np.mean(batch_ver_metrics["cell_accuracy"]) if batch_ver_metrics["cell_accuracy"] else 0.0
    avg_trm_steps = trm_steps_total / max(attempt_count, 1)

    # Metrics
    result_metrics = {
        "loss": avg_loss.item(),
        "exact_match": correct_mask.float().mean().item(),
        "avg_attempts": attempt_count,
        "avg_trm_steps": avg_trm_steps,
        "avg_reasoning_length": np.mean([len(t) for t in reasoning_texts if t]) if reasoning_texts else 0,
        "shape_match": avg_shape_match,
        "cell_accuracy": avg_cell_accuracy
    }

    # Add VAE loss metrics if using VAE adapters
    if model.config.use_vae_adapter:
        result_metrics["vae_recon_loss"] = vae_recon_loss.item() if isinstance(vae_recon_loss, torch.Tensor) else 0.0
        result_metrics["vae_kl_loss"] = vae_kl_loss.item() if isinstance(vae_kl_loss, torch.Tensor) else 0.0

    return result_metrics


def train(config: JointModelConfig):
    """Main joint training loop."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load pre-computed embeddings if using HelmARC Phase 1
    precomputed_embeddings = None
    if config.use_precomputed_embeddings:
        if config.embeddings_path is None:
            raise ValueError("embeddings_path must be provided when use_precomputed_embeddings=True")
        print(f"📥 Loading pre-computed embeddings from {config.embeddings_path}")
        precomputed_embeddings = torch.load(config.embeddings_path)
        print(f"✅ Loaded {len(precomputed_embeddings)} pre-computed embeddings")

    print("📦 Loading ARC dataset...")
    dataset_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[config.data_path],
        global_batch_size=config.batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    train_dataset = PuzzleDataset(dataset_config, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True
    )

    metadata = train_dataset.metadata.__dict__

    # Create model
    model = create_hybrid_model(config, metadata)

    # Optimizer (only adapters + TRM)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    start_step = 0
    if config.load_checkpoint is not None:
        print(f"📥 Loading hybrid checkpoint: {config.load_checkpoint}")
        checkpoint = torch.load(config.load_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)

    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.run_name or f"joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=asdict(config)
    )

    print("\n🚀 Starting joint training...")
    step = start_step

    for epoch in range(config.epochs):
        print(f"\n📖 Epoch {epoch + 1}/{config.epochs}")

        for set_name, batch, global_batch_size in train_loader:
            step += 1

            # Train step
            metrics = train_step(model, batch, optimizer, step, precomputed_embeddings)

            # Log
            if step % config.log_interval == 0:
                wandb.log(metrics, step=step)
                print(
                    f"Step {step}: loss={metrics['loss']:.4f}, "
                    f"acc={metrics['exact_match']:.2%}, "
                    f"attempts={metrics['avg_attempts']:.1f}, "
                    f"trm_steps={metrics['avg_trm_steps']:.1f}"
                )

            # Checkpoint
            if step % config.checkpoint_interval == 0:
                os.makedirs(config.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    config.output_dir,
                    f"checkpoint_step_{step}.pt"
                )
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config)
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

    wandb.finish()
    print("✅ Joint training complete!")


def evaluate(config: JointModelConfig):
    """Evaluate hybrid model on test set."""

    if config.adapter_checkpoint is None and config.load_checkpoint is None:
        raise ValueError("adapter_checkpoint or load_checkpoint must be provided for evaluation")

    print("\n🔍 Starting evaluation...")

    # Load dataset metadata
    metadata_path = Path(config.data_path) / "train" / "dataset.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Create model (adapter checkpoint will be loaded inside create_hybrid_model)
    print("\n🏗️ Building hybrid model...")
    model = create_hybrid_model(config, metadata)
    model.eval()

    if config.load_checkpoint is not None:
        checkpoint = torch.load(config.load_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    # Load test dataset
    print("\n📚 Loading test dataset...")
    dataset_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=[config.data_path],
        global_batch_size=config.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_config, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True
    )

    # Evaluation metrics
    total_samples = 0
    correct = 0
    total_attempts = 0
    total_trm_steps = 0

    print("\n🧪 Running evaluation...")

    with torch.no_grad():
        for set_name, batch, global_batch_size in dataloader:
            problems = extract_problem_data(batch)

            for idx, problem in enumerate(problems):
                sample_batch = {k: v[idx:idx + 1].to(config.device) for k, v in batch.items()}

                llm_output = model.llm.generate_latent(
                    problem["description"],
                    max_length=128,
                    use_chat_template=config.use_chat_template,
                    do_sample=False,
                )

                adapter_kwargs = {}
                if llm_output.hidden_sequence is not None:
                    adapter_kwargs["llm_hidden_sequence"] = llm_output.hidden_sequence.unsqueeze(0).to(config.device)
                if llm_output.attention_mask is not None:
                    adapter_kwargs["attention_mask"] = llm_output.attention_mask.unsqueeze(0).to(config.device)

                input_hidden = llm_output.final_hidden.unsqueeze(0).to(config.device)

                if config.use_vae_adapter:
                    z_H, z_L, _, _ = model.text_to_latent(input_hidden, deterministic=True, **adapter_kwargs)
                else:
                    z_H, z_L = model.text_to_latent(input_hidden, **adapter_kwargs)

                carry = model.trm.initial_carry(sample_batch)
                carry.inner_carry.z_H = z_H
                carry.inner_carry.z_L = z_L
                carry, loss, metrics, preds, all_finish = model.trm(
                    carry=carry,
                    batch=sample_batch,
                    return_keys=["logits"]
                )

                logits = preds.get("logits", carry.current_data.get("logits"))
                pred_tokens = logits.argmax(dim=-1)
                target_tokens = sample_batch["labels"]
                is_correct = torch.all(pred_tokens == target_tokens).item()

                total_samples += 1
                correct += int(is_correct)
                total_attempts += 1
                if isinstance(metrics, dict) and "steps" in metrics:
                    total_trm_steps += metrics["steps"].item()

                if total_samples % 10 == 0:
                    print(f"  Evaluated {total_samples} samples, accuracy: {correct/total_samples:.2%}")

    # Final metrics
    accuracy = correct / total_samples
    avg_attempts = total_attempts / total_samples
    avg_trm_steps = total_trm_steps / total_samples

    print("\n✅ Evaluation complete!")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total_samples})")
    print(f"  Avg attempts: {avg_attempts:.2f}")
    print(f"  Avg TRM steps: {avg_trm_steps:.1f}")

    # Save results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total_samples,
        "avg_attempts": avg_attempts,
        "avg_trm_steps": avg_trm_steps,
    }

    results_path = os.path.join(config.output_dir, "eval_results.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved: {results_path}")


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Joint LLaMA + TRM Training")

    # Data and paths
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/trm/joint_training")
    parser.add_argument("--trm_checkpoint", type=str, default=None)
    parser.add_argument("--adapter_checkpoint", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Load hybrid model checkpoint (adapters + optionally TRM)")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_attempts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # LLaMA configuration
    parser.add_argument("--llama_model", type=str, default="unsloth/gpt-oss-mxfp4-20b")
    parser.add_argument("--llama_device", type=str, default="cuda")
    parser.add_argument("--llama_frozen", type=str2bool, default=True)
    parser.add_argument("--llama_torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--llama_device_map", type=str, default="auto")
    parser.add_argument("--llama_trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--llama_use_fast_tokenizer", type=str2bool, default=False)

    # Pre-computed embeddings (Phase 1)
    parser.add_argument("--use_precomputed_embeddings", type=str2bool, default=False)
    parser.add_argument("--embeddings_path", type=str, default=None)

    # Adapter configuration
    parser.add_argument("--use_attention_pooling", type=str2bool, default=True)
    parser.add_argument("--use_vae_adapter", type=str2bool, default=False)
    parser.add_argument("--vae_beta", type=float, default=0.1)
    parser.add_argument("--use_cross_attention_bridge", type=str2bool, default=False)

    # LLM improvements
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--use_loss_guided_generation", type=str2bool, default=False)
    parser.add_argument("--loss_guide_num_candidates", type=int, default=3)
    parser.add_argument("--loss_guide_high_threshold", type=float, default=2.0)
    parser.add_argument("--loss_guide_low_threshold", type=float, default=0.5)

    # Repulsion (Phase 3)
    parser.add_argument("--enable_repulsion", type=str2bool, default=False)
    parser.add_argument("--repulsion_weight", type=float, default=0.5)

    # Evaluation mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (requires --load_checkpoint)")

    args = parser.parse_args()

    # Validation
    if args.eval_only and not args.load_checkpoint:
        raise ValueError("--eval_only requires --load_checkpoint")

    config = JointModelConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        trm_checkpoint=args.trm_checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        load_checkpoint=args.load_checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        device=args.device,
        max_attempts=args.max_attempts,
        seed=args.seed,
        llama_model=args.llama_model,
        llama_device=args.llama_device,
        llama_frozen=args.llama_frozen,
        llama_torch_dtype=args.llama_torch_dtype,
        llama_device_map=args.llama_device_map,
        llama_trust_remote_code=args.llama_trust_remote_code,
        llama_use_fast_tokenizer=args.llama_use_fast_tokenizer,
        use_precomputed_embeddings=args.use_precomputed_embeddings,
        embeddings_path=args.embeddings_path,
        use_attention_pooling=args.use_attention_pooling,
        use_vae_adapter=args.use_vae_adapter,
        vae_beta=args.vae_beta,
        use_cross_attention_bridge=args.use_cross_attention_bridge,
        use_chat_template=args.use_chat_template,
        use_loss_guided_generation=args.use_loss_guided_generation,
        loss_guide_num_candidates=args.loss_guide_num_candidates,
        loss_guide_high_threshold=args.loss_guide_high_threshold,
        loss_guide_low_threshold=args.loss_guide_low_threshold,
        enable_repulsion=args.enable_repulsion,
        repulsion_weight=args.repulsion_weight,
    )

    if args.eval_only:
        evaluate(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
