"""
Sequential Executor - Step-by-step compositional TRM execution

Key innovation: Instead of one-shot grid generation, execute plan step-by-step.
Each TRM call transforms current_grid to next_grid, creating a chain.

This enables:
    1. Compositional reasoning (complex task = sequence of simple operations)
    2. Intermediate supervision (see what each step produces)
    3. Better credit assignment (know which step failed)
    4. Easier debugging (LLM can inspect intermediate grids)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from .instruction_encoder import InstructionEncoder
    from .grid_utils import grid_to_string
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from instruction_encoder import InstructionEncoder
    from grid_utils import grid_to_string


class SequentialExecutor:
    """
    Executes multi-step plan compositionally.

    Each step:
        instruction → InstructionEncoder → (z_H, z_L) → TRM → output_grid

    Chaining:
        output_grid[i] becomes input_grid[i+1]
    """

    def __init__(
        self,
        instruction_encoder: InstructionEncoder,
        trm_model,
        max_trm_steps_per_instruction: int = 5,
        device: str = "cuda",
    ):
        """
        Args:
            instruction_encoder: InstructionEncoder instance
            trm_model: TinyRecursiveReasoningModel instance
            max_trm_steps_per_instruction: Max ACT steps for each instruction
            device: Device for computation
        """
        self.encoder = instruction_encoder
        self.trm = trm_model
        self.max_trm_steps = max_trm_steps_per_instruction
        self.device = device

    def execute_plan(
        self,
        steps: List[str],
        input_grid: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        target_grid: Optional[np.ndarray] = None,
        return_grad: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute multi-step plan sequentially with carry state chaining.

        CRITICAL: Carry state is initialized ONCE and chained through all steps,
        enabling true compositional reasoning in latent space.

        Args:
            steps: List of instruction strings
            input_grid: [batch, seq] initial input grid
            batch: Full batch dict (for TRM)
            target_grid: Optional target for metrics
            return_grad: Whether to maintain gradient flow

        Returns:
            final_grid: [batch, seq] final output after all steps
            intermediate_results: List of dicts with execution trace
            metrics: Aggregated metrics
        """
        if not steps:
            raise ValueError("Empty plan - no steps to execute")

        current_grid = input_grid.clone()
        intermediate_results = []
        total_trm_steps = 0
        step_losses = []

        # ✅ CRITICAL FIX: Initialize carry ONCE for compositional reasoning
        carry = self.trm.initial_carry(batch)

        for step_idx, instruction in enumerate(steps):
            # 1. Encode instruction to latent
            with torch.set_grad_enabled(return_grad):
                z_H, z_L = self.encoder.encode_single(
                    instruction,
                    return_grad=return_grad
                )

            # 2. Prepare batch with current grid as input
            batch_step = self._prepare_batch_for_step(batch, current_grid)

            # 3. Execute TRM with carry chaining
            carry, output_grid, trm_steps, step_loss = self._execute_trm_step(
                carry, z_H, z_L,
                batch_step,
                return_grad=return_grad
            )

            # 4. Store intermediate result
            intermediate_results.append({
                'step_idx': step_idx,
                'instruction': instruction,
                'input_grid': self._tensor_to_numpy(current_grid),
                'output_grid': self._tensor_to_numpy(output_grid),
                'z_H': z_H.detach().cpu() if return_grad else z_H.cpu(),
                'z_L': z_L.detach().cpu() if return_grad else z_L.cpu(),
                'trm_steps': trm_steps,
                'loss': step_loss.item() if step_loss is not None else 0.0,
            })

            total_trm_steps += trm_steps
            if step_loss is not None:
                step_losses.append(step_loss)

            # 5. Chain: output becomes next input
            current_grid = output_grid

        # Final grid is output of last step
        final_grid = current_grid

        # Aggregate metrics
        metrics = {
            'num_steps': len(steps),
            'total_trm_steps': total_trm_steps,
            'avg_trm_steps_per_instruction': total_trm_steps / len(steps),
            'total_loss': sum(step_losses) if step_losses else torch.tensor(0.0),
            'avg_loss_per_step': (sum(step_losses) / len(step_losses)) if step_losses else torch.tensor(0.0),
        }

        # Add accuracy metrics if target provided
        if target_grid is not None:
            final_grid_np = self._tensor_to_numpy(final_grid)
            metrics.update(self._compute_accuracy_metrics(final_grid_np, target_grid))

        return final_grid, intermediate_results, metrics

    def _prepare_batch_for_step(
        self,
        original_batch: Dict[str, torch.Tensor],
        current_grid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Create batch dict with current_grid as input.

        Preserves other fields from original batch (labels, metadata, etc.)
        but replaces 'inputs' with current step's grid.
        """
        batch_step = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in original_batch.items()}
        batch_step['inputs'] = current_grid
        return batch_step

    def _execute_trm_step(
        self,
        carry,  # ✅ CRITICAL: Accept carry as input
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        return_grad: bool = True,
    ) -> Tuple[object, torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Execute TRM for one instruction with carry state continuity.

        CRITICAL FIX: No longer initializes carry - uses the passed carry
        from previous step, enabling compositional reasoning in latent space.

        Args:
            carry: TRM carry state from previous step (or initial_carry for first step)
            z_H: High-frequency latent instruction representation
            z_L: Low-frequency latent instruction representation
            batch: Batch dict with inputs and optional labels
            return_grad: Whether to maintain gradient flow

        Returns:
            carry: Updated carry state to pass to next step
            output_grid: [batch, seq] predicted grid
            trm_steps: Number of ACT steps taken
            loss: Loss value if labels available
        """
        # ✅ CRITICAL FIX: Update instruction in EXISTING carry (don't reset!)
        # Inject latent representations into the existing carry state
        z_H = z_H.to(device=self.device, dtype=self.trm.inner.forward_dtype)
        z_L = z_L.to(device=self.device, dtype=self.trm.inner.forward_dtype)

        carry.inner_carry.z_H = z_H
        carry.inner_carry.z_L = z_L

        # Run TRM with ACT (raw TRM interface: forward(carry, batch) -> (carry, outputs))
        # The carry is updated in-place by TRM and returned
        trm_steps = 0
        with torch.set_grad_enabled(return_grad):
            while True:
                # ✅ Call raw TRM - it returns UPDATED carry
                carry, outputs = self.trm(carry, batch)
                trm_steps += 1

                # Check if all sequences halted
                all_finish = carry.halted.all().item()

                if all_finish or trm_steps >= self.max_trm_steps:
                    break

        # Extract output grid from outputs dict
        logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("TRM did not produce logits")

        output_grid = logits.argmax(dim=-1) if logits.dim() > 2 else logits

        # Compute loss if labels available (for training)
        loss = None
        if "labels" in batch:
            # Import here to avoid circular dependency
            import torch.nn.functional as F

            target_labels = batch["labels"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_labels.view(-1),
                ignore_index=-100  # Standard ignore index
            )

        # ✅ CRITICAL: Return updated carry for next step
        return carry, output_grid, trm_steps, loss

    def _tensor_to_numpy(self, grid: torch.Tensor) -> np.ndarray:
        """Convert grid tensor to numpy array."""
        if grid.dim() == 1:
            # Flat tensor, need to reshape (assume square grid for now)
            size = int(np.sqrt(grid.numel()))
            grid = grid.reshape(size, size)

        return grid.detach().cpu().numpy() if grid.requires_grad else grid.cpu().numpy()

    def _compute_accuracy_metrics(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, float]:
        """Compute accuracy metrics."""
        metrics = {}

        # Ensure 2D
        if predicted.ndim == 1:
            size = int(np.sqrt(predicted.size))
            predicted = predicted.reshape(size, size)
        if target.ndim == 1:
            size = int(np.sqrt(target.size))
            target = target.reshape(size, size)

        # Shape match
        metrics['shape_match'] = float(predicted.shape == target.shape)

        if predicted.shape == target.shape:
            # Exact match
            metrics['exact_match'] = float(np.array_equal(predicted, target))

            # Cell accuracy
            correct_cells = (predicted == target).sum()
            total_cells = target.size
            metrics['cell_accuracy'] = float(correct_cells / total_cells) if total_cells > 0 else 0.0
        else:
            metrics['exact_match'] = 0.0
            metrics['cell_accuracy'] = 0.0

        return metrics


# Self-review questions:
#
# Q1: Should we stop early if one step's output is clearly wrong?
# A1: No - execute all steps to gather full trace for LLM feedback.
#     LLM needs to see entire execution to diagnose.
#     Decision: Execute all steps regardless of intermediate errors.
#
# Q2: What if TRM gets stuck in ACT loop (doesn't halt)?
# A2: max_trm_steps_per_instruction limits ACT iterations.
#     Decision: Default 5 steps per instruction (enough for simple ops).
#
# Q3: Should each step use the same max_trm_steps or different?
# A3: Could make it instruction-specific, but adds complexity.
#     Decision: Uniform limit for now, optimize later if needed.
#
# Q4: How to handle gradient flow through all steps?
# A4: return_grad flag controls torch.set_grad_enabled().
#     Training: True (backprop through entire chain)
#     Inference: False (faster, less memory)
#     Decision: Caller controls via flag.
#
# Q5: What if grid shape changes between steps?
# A5: Valid transformation! Store actual shapes.
#     TRM can change grid dimensions (e.g., crop, expand).
#     Decision: Support it, feedback shows shape changes.
#
# Q6: Should we cache instruction encodings?
# A6: Same instruction in different positions has different meaning.
#     Also, gradients complicate caching.
#     Decision: No caching for now.
#
# Q7: Loss computation - sum or average across steps?
# A7: Store both: total_loss and avg_loss_per_step.
#     Caller decides which to use for optimization.
#     Decision: Provide both metrics.


if __name__ == "__main__":
    print("=" * 60)
    print("Sequential Executor Test")
    print("=" * 60)
    print("\nNote: Full test requires loaded LLM, adapter, and TRM.")
    print("This validates the interface only.")

    # Create minimal mocks
    class DummyEncoder:
        def encode_single(self, instruction, return_grad=True):
            return torch.randn(1, 900, 512), torch.randn(1, 900, 512)

    class DummyTRM:
        class DummyCarry:
            class InnerCarry:
                z_H = None
                z_L = None
            inner_carry = InnerCarry()
            current_data = {}

        class DummyModel:
            class Inner:
                forward_dtype = torch.float32
            inner = Inner()
        model = DummyModel()

        def initial_carry(self, batch):
            return self.DummyCarry()

        def __call__(self, carry, batch, return_keys):
            logits = torch.randint(0, 10, (1, 900))
            preds = {"logits": logits}
            loss = torch.tensor(1.0)
            return carry, loss, {}, preds, True

    encoder = DummyEncoder()
    trm = DummyTRM()

    executor = SequentialExecutor(
        instruction_encoder=encoder,
        trm_model=trm,
        max_trm_steps_per_instruction=3,
        device="cpu"
    )

    # Test execution
    steps = ["Rotate 90 degrees", "Flip colors"]
    input_grid = torch.randint(0, 10, (1, 900))
    batch = {"inputs": input_grid, "labels": input_grid}

    print("\nExecuting 2-step plan...")
    final_grid, intermediate_results, metrics = executor.execute_plan(
        steps=steps,
        input_grid=input_grid,
        batch=batch,
        return_grad=False
    )

    print(f"✓ Final grid shape: {final_grid.shape}")
    print(f"✓ Intermediate steps: {len(intermediate_results)}")
    print(f"✓ Metrics: {metrics}")

    for i, result in enumerate(intermediate_results):
        print(f"\nStep {i+1}: {result['instruction']}")
        print(f"  Input shape: {result['input_grid'].shape}")
        print(f"  Output shape: {result['output_grid'].shape}")
        print(f"  TRM steps: {result['trm_steps']}")

    print("\n" + "=" * 60)
    print("✓ Sequential executor interface test passed!")
    print("=" * 60)
