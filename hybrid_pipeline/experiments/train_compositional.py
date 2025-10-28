"""
Compositional Training - Step-by-step ARC solving with TRM

New approach: LLM plans multi-step transformation, TRM executes each step sequentially.
Intermediate grids are fed back to LLM for plan refinement.

This is fundamentally different from the old approach:
    OLD: LLM → one representation → TRM → final grid (one-shot)
    NEW: LLM → plan → [step1 → TRM → grid1] → [step2 → TRM → grid2] → ...
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np

# Add TinyRecursiveModels to path
trm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(trm_root))

# Import hybrid_pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpt_oss_port.llm import TextReasoningModule, LatentOutput
from gpt_oss_port.plan_parser import parse_multi_step_plan, validate_plan
from gpt_oss_port.instruction_encoder import InstructionEncoder
from gpt_oss_port.sequential_executor import SequentialExecutor
from gpt_oss_port.feedback_generator import format_step_by_step_feedback, format_success_feedback
from gpt_oss_port.grid_utils import grid_to_string
from adapters.text_to_latent import TextToLatentAdapter

# Import HelmARC dataset and contrastive loss
from helmarc_dataset import HelmARCDataset, HelmARCBatch
from contrastive_loss import compute_prediction_loss, compute_mixed_loss_type2
from train_epoch_funcs import train_epoch_type1, train_epoch_type2

# Import TRM from existing models/ module
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.functions import load_model_class
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class CompositionalConfig:
    """Configuration for compositional training."""

    # Data
    data_path: str
    output_dir: str = "/data/trm/compositional_training"

    # HelmARC dataset configuration
    use_helmarc: bool = True  # Use HelmARC integrated dataset
    helmarc_data_dir: str = "/home/ubuntu/TinyRecursiveModels/hybrid_pipeline/data/helmarc_integrated"
    dataset_type: str = "mixed"  # "type1", "type2", or "mixed" (both)
    train_type1_ratio: float = 0.7  # Ratio of TYPE1 samples in mixed mode

    # LLM
    llama_model: str = "openai/gpt-oss-20b"
    llama_device: str = "cuda"
    llama_frozen: bool = True
    llama_torch_dtype: str = "bfloat16"
    llama_use_lora: bool = False  # Enable LoRA for LLM fine-tuning
    llama_lora_r: int = 16  # LoRA rank
    llama_lora_alpha: int = 32  # LoRA alpha
    llama_lora_dropout: float = 0.05  # LoRA dropout

    # TRM
    trm_checkpoint: Optional[str] = None
    trm_hidden_size: int = 512
    trm_num_heads: int = 8
    trm_expansion: float = 4.0
    trm_H_cycles: int = 3
    trm_L_cycles: int = 6
    trm_L_layers: int = 4
    trm_max_steps_per_instruction: int = 5  # ACT steps for each instruction

    # Adapter
    adapter_use_position_embeddings: bool = True
    adapter_use_cross_attention: bool = False

    # Training
    batch_size: int = 1
    max_attempts: int = 8  # Max plan refinement attempts
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01

    # Loss weights (for TYPE2 contrastive learning)
    prediction_weight: float = 1.0
    contrastive_weight: float = 0.5
    contrastive_margin: float = 1.0

    # Instruction encoding
    instruction_pooling: str = "mean"  # "mean", "last", or "first"
    instruction_use_encoder_layers: bool = False  # If True, slower but better

    # LLM inference control (NEW - for chat template and reasoning)
    llm_use_chat_template: bool = True  # Enable apply_chat_template for reasoning_effort support
    llm_reasoning_effort: Optional[str] = None  # "low", "medium", "high" for o1-style models

    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "arc-compositional"
    save_interval: int = 100

    # Device
    device: str = "cuda"


def create_compositional_model(config: CompositionalConfig):
    """
    Create compositional model with all components.

    Components:
        1. LLM (for planning)
        2. Adapter (instruction → TRM latent)
        3. TRM (grid transformation)
        4. InstructionEncoder (wraps LLM + adapter)
        5. SequentialExecutor (step-by-step execution)
    """
    print("🔧 Creating compositional model...")

    # 1. Load LLM
    print(f"📥 Loading LLM: {config.llama_model}")
    llm = TextReasoningModule(
        model_name=config.llama_model,
        freeze=config.llama_frozen,
        device=config.device,  # Use the specified GPU device
        device_map={"": config.device},  # Force load on specific GPU
        torch_dtype=getattr(torch, config.llama_torch_dtype),
    )
    print(f"  ✓ LLM loaded (hidden_size={llm.hidden_size})")

    # Apply LoRA if requested
    if config.llama_use_lora:
        print(f"🔧 Applying LoRA to LLM (r={config.llama_lora_r}, alpha={config.llama_lora_alpha})...")
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.llama_lora_r,
                lora_alpha=config.llama_lora_alpha,
                lora_dropout=config.llama_lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA to the underlying model
            llm.model = get_peft_model(llm.model, lora_config)
            llm.model.print_trainable_parameters()
            print("  ✓ LoRA applied successfully")

        except ImportError:
            print("  ⚠ peft library not found. Install with: pip install peft")
            print("  ✗ Continuing without LoRA")
        except Exception as e:
            print(f"  ⚠ Failed to apply LoRA: {e}")
            print("  ✗ Continuing without LoRA")

    # 2. Create TRM
    print("🔧 Creating TRM model...")
    trm_config_dict = {
        "batch_size": config.batch_size,
        "seq_len": 900,  # ARC 30x30 max
        "vocab_size": 12,  # Match checkpoint (includes special tokens)
        "num_puzzle_identifiers": 1,  # Not used in compositional mode
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": config.trm_hidden_size,
        "num_heads": config.trm_num_heads,
        "expansion": config.trm_expansion,
        "H_cycles": config.trm_H_cycles,
        "L_cycles": config.trm_L_cycles,
        "H_layers": 1,  # Not used
        "L_layers": config.trm_L_layers,
        "halt_max_steps": 16,  # Global max
        "halt_exploration_prob": 0.0,  # No exploration during compositional training
        "pos_encodings": "rope",  # Match checkpoint (RoPE instead of learned)
    }

    # Direct import instead of load_model_class (clearer, avoids string parsing)
    trm_model = TinyRecursiveReasoningModel_ACTV1(trm_config_dict)

    if config.trm_checkpoint:
        print(f"📥 Loading TRM checkpoint: {config.trm_checkpoint}")
        checkpoint = torch.load(config.trm_checkpoint, map_location=config.device)

        # Strip "model." prefix from checkpoint keys if present
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        trm_model.load_state_dict(state_dict)
        print("  ✓ TRM checkpoint loaded")

    trm_model = trm_model.to(config.device)
    print(f"  ✓ TRM created (hidden_size={config.trm_hidden_size})")

    # 3. Create Adapter
    print("🔌 Creating text-to-latent adapter...")
    adapter = TextToLatentAdapter(
        llm_hidden_size=llm.hidden_size,
        trm_hidden_size=config.trm_hidden_size,
        trm_seq_len=900,  # ARC max 30x30
        use_position_embeddings=config.adapter_use_position_embeddings,
        use_cross_attention=config.adapter_use_cross_attention,
    ).to(config.device)
    print(f"  ✓ Adapter created")

    # 4. Create Instruction Encoder
    print("🔧 Creating instruction encoder...")
    encoder = InstructionEncoder(
        llm_module=llm,
        text_to_latent_adapter=adapter,
        pooling_method=config.instruction_pooling,
        use_encoder_layers=config.instruction_use_encoder_layers,
    )
    print(f"  ✓ Encoder created (pooling={config.instruction_pooling})")

    # 5. Create Sequential Executor
    print("🔧 Creating sequential executor...")
    executor = SequentialExecutor(
        instruction_encoder=encoder,
        trm_model=trm_model,
        max_trm_steps_per_instruction=config.trm_max_steps_per_instruction,
        device=config.device,
    )
    print("  ✓ Executor created")

    return {
        "llm": llm,
        "adapter": adapter,
        "trm": trm_model,
        "encoder": encoder,
        "executor": executor,
        "config": config,
    }


def generate_plan(
    llm: TextReasoningModule,
    problem_description: str,
    feedback: Optional[str] = None,
    max_length: int = 256,
    use_chat_template: bool = False,
    reasoning_effort: Optional[str] = None,
) -> LatentOutput:
    """
    Generate multi-step plan from LLM.

    Args:
        llm: TextReasoningModule
        problem_description: ARC problem description
        feedback: Optional feedback from previous attempt
        max_length: Max tokens to generate
        use_chat_template: Enable apply_chat_template (for GPT-OSS)
        reasoning_effort: Reasoning effort level ("low", "medium", "high") for o1-style models

    Returns:
        LatentOutput with generated plan
    """
    # Build prompt
    prompt = (
        f"{problem_description}\n\n"
        "Provide a step-by-step transformation plan.\n"
        "Format each step as: 'Step N: <instruction>'\n"
        "Plan:\n"
    )

    if feedback:
        prompt = f"{prompt}\nPrevious attempt feedback:\n{feedback}\n\nRevised plan:\n"

    # Generate with chat template support
    latent_output = llm.generate_latent(
        problem_text=prompt,
        max_length=max_length,
        use_chat_template=use_chat_template,
        reasoning_effort=reasoning_effort,
        do_sample=False,  # Deterministic for reproducibility
    )

    return latent_output


def train_step(
    model_components: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Single training step with multi-attempt refinement.

    Returns:
        metrics: Training metrics
    """
    llm = model_components["llm"]
    executor = model_components["executor"]
    config = model_components["config"]

    device = config.device

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}

    input_grid = batch["inputs"]
    target_grid = batch["labels"]

    # Extract problem description (simplified for now)
    # TODO: Real implementation should format from train/test pairs
    problem_description = "Transform the input grid to match the output pattern."

    # Multi-attempt loop
    total_loss = None
    success = False
    feedback = None

    for attempt in range(config.max_attempts):
        # 1. Generate plan (with chat template and reasoning effort if configured)
        plan_output = generate_plan(
            llm,
            problem_description,
            feedback,
            use_chat_template=config.llm_use_chat_template,
            reasoning_effort=config.llm_reasoning_effort,
        )
        plan_text = plan_output.generated_text

        # 2. Parse plan
        steps = parse_multi_step_plan(plan_text)

        if not validate_plan(steps, min_steps=1, max_steps=10):
            print(f"  ⚠ Invalid plan (attempt {attempt+1})")
            feedback = "Plan format invalid. Use 'Step N: instruction' format."
            continue

        print(f"  📋 Plan (attempt {attempt+1}): {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            print(f"    {i}. {step}")

        # 3. Execute plan
        try:
            final_grid, intermediate_results, metrics = executor.execute_plan(
                steps=steps,
                input_grid=input_grid,
                batch=batch,
                target_grid=target_grid.cpu().numpy()[0],
                return_grad=True,  # Training mode
            )
        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
            feedback = f"Execution error: {e}. Simplify the plan."
            continue

        # 4. Check success
        if metrics.get('exact_match', 0.0) > 0.99:
            success = True
            total_loss = metrics['total_loss']
            print(f"  ✅ Success in {attempt+1} attempts!")
            break

        # 5. Generate feedback
        feedback = format_step_by_step_feedback(
            steps=steps,
            intermediate_results=intermediate_results,
            target_grid=target_grid.cpu().numpy()[0],
            original_input=input_grid.cpu().numpy()[0],
        )

        total_loss = metrics['total_loss']

        print(f"  ❌ Failed (attempt {attempt+1}): {metrics['cell_accuracy']:.2%} accuracy")

    # Backward pass
    if total_loss is not None and total_loss.requires_grad:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Return metrics
    return {
        "success": float(success),
        "attempts": attempt + 1,
        "loss": total_loss.item() if total_loss is not None else 0.0,
        "num_steps": len(steps) if steps else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compositional ARC Training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/trm/compositional_training")
    parser.add_argument("--trm_checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_attempts", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true")

    # HelmARC dataset
    parser.add_argument("--use_helmarc", action="store_true", default=True,
                       help="Use HelmARC integrated dataset")
    parser.add_argument("--helmarc_data_dir", type=str,
                       default="/home/ubuntu/TinyRecursiveModels/hybrid_pipeline/data/helmarc_integrated")
    parser.add_argument("--dataset_type", type=str, default="mixed",
                       choices=["type1", "type2", "mixed"],
                       help="Dataset type: type1 (correct only), type2 (wrong+correct), or mixed")

    # LLM LoRA
    parser.add_argument("--llama_use_lora", action="store_true",
                       help="Enable LoRA for LLM fine-tuning")
    parser.add_argument("--llama_lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--llama_lora_alpha", type=int, default=32,
                       help="LoRA alpha")

    # LLM inference control
    parser.add_argument("--llm_use_chat_template", action="store_true", default=True,
                       help="Enable apply_chat_template for reasoning_effort support")
    parser.add_argument("--llm_reasoning_effort", type=str, default=None,
                       choices=["low", "medium", "high"],
                       help="Reasoning effort for o1-style models")

    # Loss weights
    parser.add_argument("--prediction_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--contrastive_margin", type=float, default=1.0)

    args = parser.parse_args()

    # Create config
    config = CompositionalConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_helmarc=args.use_helmarc,
        helmarc_data_dir=args.helmarc_data_dir,
        dataset_type=args.dataset_type,
        trm_checkpoint=args.trm_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_attempts=args.max_attempts,
        device=args.device,
        use_wandb=args.use_wandb,
        llama_use_lora=args.llama_use_lora,
        llama_lora_r=args.llama_lora_r,
        llama_lora_alpha=args.llama_lora_alpha,
        llm_use_chat_template=args.llm_use_chat_template,
        llm_reasoning_effort=args.llm_reasoning_effort,
        prediction_weight=args.prediction_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
    )

    # Initialize wandb
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=asdict(config))

    # Create model
    model_components = create_compositional_model(config)

    # Create optimizer
    trainable_params = (
        list(model_components["adapter"].parameters()) +
        list(model_components["trm"].parameters())
    )

    # Add LLM parameters if using LoRA
    if config.llama_use_lora:
        print("  ✓ Including LLM LoRA parameters in optimizer")
        trainable_params += list(model_components["llm"].parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    print(f"📊 Optimizer created with {sum(p.numel() for p in trainable_params):,} trainable parameters")

    # Load HelmARC dataset
    print("\n📊 Loading HelmARC dataset...")

    if config.use_helmarc:
        if config.dataset_type == "mixed":
            # Load both TYPE1 and TYPE2
            dataset_type1 = HelmARCDataset(
                dataset_type="type1",
                data_dir=config.helmarc_data_dir,
                use_integrated=True,
            )
            dataset_type2 = HelmARCDataset(
                dataset_type="type2",
                data_dir=config.helmarc_data_dir,
                use_integrated=True,
            )

            print(f"  ✓ TYPE1: {len(dataset_type1)} samples")
            print(f"  ✓ TYPE2: {len(dataset_type2)} samples")

            # Create dataloaders
            collator = HelmARCBatch(max_grid_size=30)
            dataloader_type1 = DataLoader(
                dataset_type1,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collator,
            )
            dataloader_type2 = DataLoader(
                dataset_type2,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collator,
            )

        else:
            # Load single dataset type
            dataset = HelmARCDataset(
                dataset_type=config.dataset_type,
                data_dir=config.helmarc_data_dir,
                use_integrated=True,
            )

            print(f"  ✓ {config.dataset_type.upper()}: {len(dataset)} samples")

            collator = HelmARCBatch(max_grid_size=30)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collator,
            )

            # For unified interface
            if config.dataset_type == "type1":
                dataloader_type1 = dataloader
                dataloader_type2 = None
            else:
                dataloader_type1 = None
                dataloader_type2 = dataloader
    else:
        print("  ⚠ HelmARC disabled - using dummy data")
        dataloader_type1 = None
        dataloader_type2 = None

    # Training loop
    print("\n🚀 Starting training...")
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")

        # Train on TYPE1 if available
        if dataloader_type1 is not None:
            print(f"\n📘 Training on TYPE1 (Correct programs only)...")
            train_epoch_type1(
                config=config,
                model_components=model_components,
                dataloader=dataloader_type1,
                optimizer=optimizer,
                epoch=epoch,
            )

        # Train on TYPE2 if available
        if dataloader_type2 is not None:
            print(f"\n📕 Training on TYPE2 (Wrong vs Correct)...")
            train_epoch_type2(
                config=config,
                model_components=model_components,
                dataloader=dataloader_type2,
                optimizer=optimizer,
                epoch=epoch,
            )

        # Save checkpoint periodically
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_dir = Path(config.output_dir) / f"checkpoint_epoch_{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save TRM
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_components["trm"].state_dict(),
            }, checkpoint_dir / "trm.pt")

            # Save Adapter
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_components["adapter"].state_dict(),
            }, checkpoint_dir / "adapter.pt")

            # Save LLM if using LoRA
            if config.llama_use_lora:
                model_components["llm"].model.save_pretrained(checkpoint_dir / "llm_lora")

            print(f"  💾 Checkpoint saved to {checkpoint_dir}")

    print("\n✅ Training complete!")
    print(f"📁 Output directory: {config.output_dir}")


# Self-review questions:
#
# Q1: Why not batch multiple problems together?
# A1: Each problem has different number of refinement attempts.
#     Sample 1 might succeed in 2 attempts, Sample 2 in 5 attempts.
#     Batching would require padding/masking.
#     Decision: Keep batch_size=1, parallelize across GPUs instead.
#
# Q2: Should we cache LLM plans across epochs?
# A2: No - plans should evolve as adapter/TRM improve.
#     Same problem might need different plan with better TRM.
#     Decision: Regenerate plans every time.
#
# Q3: What if LLM always generates same invalid plan?
# A3: Current: Keep trying up to max_attempts.
#     Better: Add temperature/sampling after N failed attempts.
#     Decision: Add in v2 if this becomes an issue.
#
# Q4: How to handle very long plans (e.g., 20 steps)?
# A4: validate_plan() limits to max_steps=10.
#     Rationale: ARC tasks should be solvable in <10 steps.
#     If not, problem decomposition is wrong.
#     Decision: Hard limit at 10 steps.
#
# Q5: Should we save intermediate checkpoints during refinement?
# A5: No - only save after successful training step.
#     Intermediate states during refinement are transient.
#     Decision: Save at epoch level only.
#
# Q6: What about curriculum learning (easy tasks first)?
# A6: Not implemented yet, but would be beneficial.
#     Decision: Add task difficulty sorting in v2.
#
# Q7: Memory usage with max_attempts=8?
# A7: Each attempt creates computation graph.
#     With gradient accumulation, could OOM.
#     Decision: Use return_grad=False for early attempts,
#              only True for final successful attempt.
#              TODO: Implement this optimization.


if __name__ == "__main__":
    main()

