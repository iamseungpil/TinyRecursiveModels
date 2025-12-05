"""
Stage 1 Training Script: TRM with GridEncoder for input only.

This script trains the original TRM model with GridEncoder replacing
the embed_tokens for input embedding. The key enhancement is:

1. GridEncoder processes input grids with self-attention
2. Learnable CLS token at position 0 (for q_head compatibility)
3. Learnable padding tokens at positions 1-15 (total prefix = 16)
4. Original TRM architecture and hyperparameters preserved

Architecture (with use_original_puzzle_emb=False):
    Input: [CLS(1), padding(15), grid_tokens(L)]
    - Position 0 = CLS token (used by q_head)
    - Positions 1-15 = learnable padding
    - Positions 16+ = GridEncoder output for input grid
    - puzzle_emb is NOT used (excluded from optimizer)

Architecture (with use_original_puzzle_emb=True):
    Input: [puzzle_emb(16), grid_tokens(L)]
    - Position 0-15 = original puzzle_emb (scaled)
    - Positions 16+ = GridEncoder output for input grid
    - puzzle_emb IS used (included in optimizer)

Position Encoding Strategy:
===========================
GridEncoder and TRM each contribute position information.
These serve DIFFERENT semantic purposes:

1. GridEncoder pos_embed (grid_position_encoding_mode):
   - "additive" (default): pos_embed[0:L] added to grid tokens
     * Captures 2D grid spatial structure (row/column awareness)
   - "none": No position encoding from GridEncoder
     * Rely entirely on TRM RoPE for position info
   - "offset": pos_embed[16:16+L] added to grid tokens
     * Aligned with TRM sequence positions (after prefix)

2. TRM RoPE (disable_rope_for_encoder):
   - False (default): RoPE applied in attention to all tokens
   - True: No RoPE when using GridEncoder

Recommended Configurations:
- Hybrid (default): additive + RoPE (both spatial and sequence info)
- RoPE Only: none + RoPE (like standard transformer)
- GridEncoder Only: additive + no RoPE (grid spatial only)
- Aligned: offset + RoPE (both use same position indices)

Hyperparameters (matching original TRM):
    - H_cycles: 3, L_cycles: 6, L_layers: 2
    - hidden_size: 512, num_heads: 8, expansion: 4
    - puzzle_emb_len: 16 (preserved)
    - global_batch_size: 768, lr: 1e-4, puzzle_emb_lr: 1e-2

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pretrain_stage1.py

    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python pretrain_stage1.py

    # Test different position encoding strategies:
    # Hybrid (default)
    python pretrain_stage1.py grid_position_encoding_mode=additive

    # RoPE only
    python pretrain_stage1.py grid_position_encoding_mode=none

    # GridEncoder only (no RoPE)
    python pretrain_stage1.py disable_rope_for_encoder=True
"""

from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

try:
    from adam_atan2 import AdamATan2
except ImportError:
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# Import EnhancedTRM for Stage 1
from models.recursive_reasoning.encoder import (
    EnhancedTRM,
    EnhancedTRMConfig,
    create_stage1_grid_encoder,
    create_stage0_baseline,
)

# ============================================================================
# CONFIGURATION - Modify these paths and tokens as needed
# ============================================================================
DATA_BASE_PATH = "/data/TinyRecursiveModels"
WANDB_TOKEN = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# ============================================================================


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding / prefix token learning rate
    # Used for CLS/prefix tokens in encoder mode or puzzle_emb in original mode
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Stage 1 specific options
    use_grid_encoder: bool = True  # Set to False for baseline comparison
    use_original_puzzle_emb: bool = False  # If True, use puzzle_emb with GridEncoder

    # Position Encoding Strategy (only applies when use_grid_encoder=True)
    # See config/cfg_grid_encoder.yaml for detailed documentation
    grid_position_encoding_mode: str = "additive"  # "additive", "none", "offset"
    disable_rope_for_encoder: bool = False  # True to disable TRM RoPE

    # Gradient clipping for training stability
    grad_clip_norm: float = 1.0  # Max gradient norm; set to 0 to disable

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    """
    Create EnhancedTRM model for Stage 1 training.

    Two modes supported:
    1. use_grid_encoder=True: GridEncoder for input embedding
       - use_original_puzzle_emb=False: CLS + padding as prefix (puzzle_emb excluded)
       - use_original_puzzle_emb=True: puzzle_emb as prefix (puzzle_emb included)

    2. use_grid_encoder=False: Original TRM baseline
       - Exact same behavior as main branch
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    # Instantiate base TRM model
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        base_model: nn.Module = model_cls(model_cfg)
        print(f"Base TRM model created:")
        print(base_model)

        # Choose mode based on config
        if config.use_grid_encoder:
            # Stage 1: GridEncoder for input embedding
            enhanced_model = create_stage1_grid_encoder(
                base_trm=base_model,
                freeze_trm=False,  # Train TRM as well
                use_original_puzzle_emb=config.use_original_puzzle_emb,
                grid_encoder_layers=2,
                grid_encoder_heads=4,
                # Position encoding strategy
                grid_position_encoding_mode=config.grid_position_encoding_mode,
                disable_rope_for_encoder=config.disable_rope_for_encoder,
            )

            print(f"\nEnhanced TRM (Stage 1) created:")
            print(f"  use_grid_encoder_for_input: True")
            print(f"  use_original_puzzle_emb: {config.use_original_puzzle_emb}")
            print(f"  GridEncoder layers: 2")
            print(f"  GridEncoder heads: 4")
            # Position encoding strategy info
            print(f"\n  Position Encoding Strategy:")
            print(f"    grid_position_encoding_mode: {config.grid_position_encoding_mode}")
            print(f"    disable_rope_for_encoder: {config.disable_rope_for_encoder}")
            if config.grid_position_encoding_mode == "additive":
                print(f"    -> GridEncoder adds pos_embed[0:L] to grid tokens")
            elif config.grid_position_encoding_mode == "none":
                print(f"    -> GridEncoder: NO position encoding")
            elif config.grid_position_encoding_mode == "offset":
                print(f"    -> GridEncoder adds pos_embed[16:16+L] to grid tokens")
            if config.disable_rope_for_encoder:
                print(f"    -> TRM RoPE: DISABLED")
            else:
                print(f"    -> TRM RoPE: Applied in attention")
            # Prefix info
            if config.use_original_puzzle_emb:
                print(f"\n  Prefix structure: puzzle_emb(16) = 16 tokens")
                print(f"  puzzle_emb: INCLUDED in optimizer")
            else:
                print(f"\n  Prefix structure: [CLS(1), padding(15)] = 16 tokens")
                print(f"  Position 0 = CLS token (for q_head)")
                print(f"  puzzle_emb: EXCLUDED from optimizer")
        else:
            # Stage 0: Original TRM baseline
            enhanced_model = create_stage0_baseline(base_trm=base_model)
            print(f"\nOriginal TRM (Stage 0 Baseline):")
            print(f"  use_grid_encoder_for_input: False")
            print(f"  Using original embed_tokens + puzzle_emb")

        # Wrap with loss head
        model = loss_head_cls(enhanced_model, **config.arch.loss.__pydantic_extra__)

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

        # Load checkpoint if specified
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # ========================================================================
    # Optimizer setup with proper parameter groups
    # ========================================================================
    # IMPORTANT: puzzle_emb uses CastedSparseEmbedding which stores weights
    # as buffers, not parameters. It requires a special optimizer:
    # CastedSparseEmbeddingSignSGD_Distributed
    #
    # Parameter groups depend on mode:
    #
    # GridEncoder mode (use_original_puzzle_emb=False):
    #   - AdamATan2 for: encoder, prefix (CLS+padding), trm
    #   - puzzle_emb: NOT included (not used)
    #
    # GridEncoder mode (use_original_puzzle_emb=True):
    #   - CastedSparseEmbeddingSignSGD for: puzzle_emb
    #   - AdamATan2 for: encoder, trm
    #
    # Original mode:
    #   - CastedSparseEmbeddingSignSGD for: puzzle_emb
    #   - AdamATan2 for: trm
    # ========================================================================

    # Get enhanced_trm from model (after torch.compile)
    enhanced_trm = None
    if hasattr(model, '_orig_mod'):
        if hasattr(model._orig_mod, 'model'):
            enhanced_trm = model._orig_mod.model
    elif hasattr(model, 'model'):
        enhanced_trm = model.model

    # Small initial lr for adam_atan2_pytorch compatibility (lr > 0 required)
    init_lr = 1e-10

    # Determine if we should use puzzle_emb optimizer
    use_puzzle_emb_optimizer = False
    if enhanced_trm is not None and isinstance(enhanced_trm, EnhancedTRM):
        # Check if puzzle_emb actually exists (puzzle_emb_ndim > 0 in base TRM config)
        has_puzzle_emb = (
            hasattr(enhanced_trm, 'base_trm') and
            hasattr(enhanced_trm.base_trm, 'inner') and
            hasattr(enhanced_trm.base_trm.inner, 'puzzle_emb') and
            enhanced_trm.base_trm.inner.config.puzzle_emb_ndim > 0
        )

        if has_puzzle_emb:
            # Use puzzle_emb optimizer when:
            # 1. Original mode (not using GridEncoder) - always use puzzle_emb
            # 2. GridEncoder mode with use_original_puzzle_emb=True
            if not enhanced_trm.config.use_grid_encoder_for_input:
                use_puzzle_emb_optimizer = True
            elif enhanced_trm.config.use_original_puzzle_emb:
                use_puzzle_emb_optimizer = True

    if enhanced_trm is not None and isinstance(enhanced_trm, EnhancedTRM):
        # Get parameter groups from EnhancedTRM (excludes puzzle_emb which is buffer)
        param_groups = enhanced_trm.get_parameter_groups(
            encoder_lr=init_lr,
            prefix_lr=init_lr,
            trm_lr=init_lr,
            puzzle_emb_lr=init_lr,  # Won't be used since puzzle_emb is not in named_parameters
        )

        # Filter out empty groups and convert generators to lists
        valid_groups = []
        for g in param_groups:
            params = list(g["params"])
            if len(params) > 0:
                g["params"] = params
                valid_groups.append(g)
                print(f"  Param group '{g['name']}': {sum(p.numel() for p in params):,} params")

        param_groups = valid_groups

        if use_puzzle_emb_optimizer:
            # Two optimizers: puzzle_emb (special) + others (AdamATan2)
            print(f"\n  puzzle_emb: Using CastedSparseEmbeddingSignSGD_Distributed")

            optimizers = [
                CastedSparseEmbeddingSignSGD_Distributed(
                    enhanced_trm.base_trm.inner.puzzle_emb.buffers(),
                    lr=init_lr,
                    weight_decay=config.puzzle_emb_weight_decay,
                    world_size=world_size
                ),
                AdamATan2(
                    param_groups,
                    lr=init_lr,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]

            # LRs: [puzzle_emb_lr, {group_name: lr}]
            optimizer_lrs = [
                config.puzzle_emb_lr,  # For puzzle_emb optimizer
                {  # For AdamATan2 optimizer
                    "encoder": config.lr,
                    "prefix": config.puzzle_emb_lr,
                    "trm": config.lr,
                    "history": config.lr,
                }
            ]
        else:
            # Single optimizer: AdamATan2 only (puzzle_emb not used)
            print(f"\n  puzzle_emb: EXCLUDED (using CLS+padding instead)")

            optimizers = [
                AdamATan2(
                    param_groups,
                    lr=init_lr,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]

            optimizer_lrs = [{
                "encoder": config.lr,
                "prefix": config.puzzle_emb_lr,
                "trm": config.lr,
                "history": config.lr,
            }]

        # Log configuration
        print("\nOptimizer configuration:")
        if use_puzzle_emb_optimizer:
            print(f"  puzzle_emb: CastedSparseEmbeddingSignSGD, target_lr={config.puzzle_emb_lr}")
        for g in param_groups:
            target_lr = optimizer_lrs[-1].get(g['name'], config.lr) if isinstance(optimizer_lrs[-1], dict) else config.lr
            print(f"  {g['name']}: {sum(p.numel() for p in g['params']):,} params, target_lr={target_lr}")

    else:
        # Fallback: original pretrain.py logic
        print("Warning: Could not access EnhancedTRM for parameter groups")
        print("  Using original pretrain.py optimizer setup")

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=init_lr,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=init_lr,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Handle puzzle_emb resizing if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        try:
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape
            if puzzle_emb_name in state_dict:
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
        except AttributeError:
            pass  # puzzle_emb may not exist in enhanced model

        # Load with strict=False to handle new encoder parameters
        model.load_state_dict(state_dict, assign=True, strict=False)


def compute_lr(base_lr, config: PretrainConfig, train_state: TrainState):
    """
    Compute learning rate with cosine schedule.

    Args:
        base_lr: Can be either:
            - float: Single base LR
            - dict: Mapping of group_name -> base_lr
    """
    if isinstance(base_lr, dict):
        # Return dict of computed LRs for each group
        return {
            name: cosine_schedule_with_warmup_lr_lambda(
                current_step=train_state.step,
                base_lr=lr,
                num_warmup_steps=round(config.lr_warmup_steps),
                num_training_steps=train_state.total_steps,
                min_ratio=config.lr_min_ratio
            )
            for name, lr in base_lr.items()
        }
    else:
        return cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(config.lr_warmup_steps),
            num_training_steps=train_state.total_steps,
            min_ratio=config.lr_min_ratio
        )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )
            evaluators.append(cls)

    return evaluators


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Gradient clipping for training stability
    # Applied after all_reduce to ensure consistent clipping across ranks
    if config.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), max_norm=config.grad_clip_norm)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_computed = compute_lr(base_lr, config, train_state)

        if isinstance(lr_computed, dict):
            # Set LR for each parameter group by name
            for param_group in optim.param_groups:
                group_name = param_group.get("name", "default")
                if group_name in lr_computed:
                    param_group['lr'] = lr_computed[group_name]
                    # Use TRM or encoder LR for logging
                    if lr_this_step is None:
                        lr_this_step = lr_computed.get("trm", lr_computed.get("encoder"))
        else:
            for param_group in optim.param_groups:
                param_group['lr'] = lr_computed
            lr_this_step = lr_computed

        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}
        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")

            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)

        # Override data paths to use /data - use the path from config
        # config.data_paths is already set in cfg_grid_encoder.yaml

        # Naming
        if config.project_name is None:
            config.project_name = "TRM-Stage1-GridEncoder"
        if config.run_name is None:
            if config.use_grid_encoder:
                prefix = "stage1-encoder"
                if config.use_original_puzzle_emb:
                    prefix += "-puzzle_emb"
                else:
                    prefix += "-cls_prefix"
                # Add position encoding strategy to name
                pos_mode = config.grid_position_encoding_mode
                if pos_mode == "additive":
                    prefix += "-hybrid"  # additive + RoPE (default)
                elif pos_mode == "none":
                    prefix += "-rope_only"
                elif pos_mode == "offset":
                    prefix += "-aligned"
                if config.disable_rope_for_encoder:
                    prefix += "-no_rope"
            else:
                prefix = "stage0-baseline"
            config.run_name = f"{prefix}-{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(DATA_BASE_PATH, "checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


@hydra.main(config_path="config", config_name="cfg_grid_encoder", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        # Login to wandb with configured token
        wandb.login(key=WANDB_TOKEN)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        # Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            # Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            # Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
