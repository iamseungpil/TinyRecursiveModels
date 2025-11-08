"""
Test-Time Adaptation for TRM

Adapts puzzle embedding using training examples at test time.
Minimal modification to original code.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TestTimeConfig:
    """Configuration for test-time adaptation."""
    reserved_puzzle_id: int = 0  # ID slot for new puzzles
    learning_rate: float = 1e-3
    max_steps: int = 50
    patience: int = 5  # Early stopping
    min_loss_improvement: float = 1e-4


class TestTimeAdapter:
    """
    Adapts puzzle embedding using training examples.

    This class implements test-time training by:
    1. Initializing a puzzle embedding for new puzzles
    2. Training on available training examples
    3. Using learned embedding for test inference
    """

    def __init__(self, model: nn.Module, config: TestTimeConfig = None):
        """
        Args:
            model: TRM model with puzzle_emb attribute
            config: Test-time adaptation configuration
        """
        self.model = model
        self.config = config or TestTimeConfig()

        # Validate model has puzzle embedding
        if not hasattr(model, 'puzzle_emb'):
            # Try to access through compiled model
            if hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'puzzle_emb'):
                self.puzzle_emb = model._orig_mod.puzzle_emb
            elif hasattr(model, 'model') and hasattr(model.model, 'puzzle_emb'):
                self.puzzle_emb = model.model.puzzle_emb
            else:
                raise ValueError("Model does not have puzzle_emb attribute")
        else:
            self.puzzle_emb = model.puzzle_emb

    def initialize_puzzle_embedding(self, puzzle_id: int = None):
        """
        Initialize puzzle embedding for a new puzzle.

        Args:
            puzzle_id: ID to use (default: reserved_puzzle_id)
        """
        if puzzle_id is None:
            puzzle_id = self.config.reserved_puzzle_id

        # Small random initialization
        with torch.no_grad():
            if hasattr(self.puzzle_emb, 'weights'):
                # Direct weights access
                emb_dim = self.puzzle_emb.weights.shape[1]
                self.puzzle_emb.weights[puzzle_id] = torch.randn(emb_dim) * 0.01
            else:
                raise ValueError("Cannot access puzzle embedding weights")

    def adapt(
        self,
        train_examples: List[Dict[str, torch.Tensor]],
        puzzle_id: int = None,
        device: str = "cuda"
    ) -> Tuple[int, Dict[str, List[float]]]:
        """
        Adapt puzzle embedding using training examples.

        Args:
            train_examples: List of dicts with 'input' and 'output' tensors
            puzzle_id: ID to use (default: reserved_puzzle_id)
            device: Device to use

        Returns:
            Tuple of (puzzle_id, training_history)
        """
        if puzzle_id is None:
            puzzle_id = self.config.reserved_puzzle_id

        # Initialize embedding
        self.initialize_puzzle_embedding(puzzle_id)

        # Create learnable Parameter from puzzle embedding
        # This is necessary because CastedSparseEmbedding uses torch.no_grad()
        # and returns Buffers instead of Parameters
        if not hasattr(self.puzzle_emb, 'weights'):
            raise ValueError("Cannot access puzzle embedding weights")

        puzzle_emb_param = nn.Parameter(
            self.puzzle_emb.weights[puzzle_id].clone().detach().to(device)
        )

        # Setup optimizer for the learnable parameter
        optimizer = torch.optim.AdamW([puzzle_emb_param], lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_steps
        )

        # Training history
        history = {
            'loss': [],
            'lr': []
        }

        best_loss = float('inf')
        patience_counter = 0

        # Set model to eval mode to avoid torch.no_grad() contexts
        original_training_mode = self.model.training
        self.model.eval()

        # Store original forward method and replace with custom one
        original_forward = self.puzzle_emb.forward

        def custom_forward(inputs):
            """Custom forward that uses our learnable parameter."""
            # Create output tensor - flat embeddings (batch_size, puzzle_emb_ndim)
            batch_size = inputs.shape[0]
            emb_ndim = self.puzzle_emb.weights.shape[1]

            output = torch.zeros(batch_size, emb_ndim, device=inputs.device, dtype=puzzle_emb_param.dtype)

            # For each input, check if it's our puzzle_id
            for i in range(batch_size):
                if inputs[i] == puzzle_id:
                    # Use learnable parameter (preserves gradient)
                    output[i] = puzzle_emb_param
                else:
                    # Use original weights for other puzzles
                    output[i] = self.puzzle_emb.weights[inputs[i]].to(puzzle_emb_param.dtype)

            return output.to(self.puzzle_emb.cast_to)

        try:
            # Replace forward method temporarily
            self.puzzle_emb.forward = custom_forward

            for step in range(self.config.max_steps):
                optimizer.zero_grad()
                total_loss = 0.0

                # Train on all examples
                for example in train_examples:
                    # Prepare batch
                    inputs = example['input'].unsqueeze(0).to(device)
                    labels = example['output'].unsqueeze(0).to(device)
                    puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long, device=device)

                    # Forward pass
                    batch = {
                        'inputs': inputs,
                        'labels': labels,
                        'puzzle_identifiers': puzzle_ids
                    }

                    # Get model output
                    # For TRM, we need to run through the full forward pass
                    carry = self.model.initial_carry(batch)

                    # Move carry to device if needed
                    if hasattr(carry, 'inner_carry'):
                        InnerCarry = type(carry.inner_carry)
                        Carry = type(carry)
                        carry = Carry(
                            inner_carry=InnerCarry(
                                z_H=carry.inner_carry.z_H.to(device),
                                z_L=carry.inner_carry.z_L.to(device)
                            ),
                            steps=carry.steps.to(device),
                            halted=carry.halted.to(device),
                            current_data={k: v.to(device) for k, v in carry.current_data.items()}
                        )

                    # Single ACT step for efficiency
                    carry, outputs = self.model(carry, batch)

                    # Compute loss
                    logits = outputs['logits']

                    # Cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100
                    )

                    # Accumulate loss (divide by num examples for averaging)
                    (loss / len(train_examples)).backward()
                    total_loss += loss.item()

                # Average loss
                avg_loss = total_loss / len(train_examples)

                # Optimizer step
                optimizer.step()
                scheduler.step()

                # Record history
                history['loss'].append(avg_loss)
                history['lr'].append(optimizer.param_groups[0]['lr'])

                # Early stopping
                if avg_loss < best_loss - self.config.min_loss_improvement:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"Early stopping at step {step+1}, loss: {avg_loss:.4f}")
                    break

        finally:
            # Restore original forward method
            self.puzzle_emb.forward = original_forward

            # Keep final learned embedding
            with torch.no_grad():
                self.puzzle_emb.weights[puzzle_id] = puzzle_emb_param.data

            # Restore original training mode
            self.model.train(original_training_mode)

        return puzzle_id, history

    def reset_puzzle_embedding(self, puzzle_id: int = None):
        """Reset puzzle embedding to zero."""
        if puzzle_id is None:
            puzzle_id = self.config.reserved_puzzle_id

        with torch.no_grad():
            if hasattr(self.puzzle_emb, 'weights'):
                self.puzzle_emb.weights[puzzle_id].zero_()
