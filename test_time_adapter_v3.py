"""
Test-Time Adapter V3 - Retrieval-Based Initialization

Key improvement over V2:
- Uses K-NN retrieval to find similar puzzles from training set
- Initializes embedding from average of K similar puzzles (not global mean)
- Expected: Better initial loss, faster convergence, higher accuracy
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from similarity_metrics import compute_example_similarity
from dataset.build_arc_dataset import arc_grid_to_np


@dataclass
class TestTimeConfigV3:
    """Configuration for test-time adaptation with retrieval."""
    reserved_puzzle_id: int = 876410
    learning_rate: float = 1e-3
    max_steps: int = 50
    patience: int = 5
    min_loss_improvement: float = 1e-4
    halt_max_steps: int = 16  # Max ACT steps (match model config)
    k_neighbors: int = 10  # Number of similar puzzles for initialization


class TestTimeAdapterV3:
    """
    Test-Time Adapter with Retrieval-Based Initialization.

    Key innovation:
    - Finds K most similar puzzles from training set (876,406 puzzles)
    - Initializes from their average embedding (not global mean)
    - Should start with lower initial loss and converge faster
    """

    def __init__(self, model: nn.Module, config: TestTimeConfigV3 = None, data_path: str = None):
        self.model = model
        self.config = config or TestTimeConfigV3()
        self.data_path = data_path

        # Validate model has puzzle embedding
        if not hasattr(model, 'puzzle_emb'):
            if hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'puzzle_emb'):
                self.puzzle_emb = model._orig_mod.puzzle_emb
            else:
                raise ValueError("Model does not have puzzle_emb attribute")
        else:
            self.puzzle_emb = model.puzzle_emb

        # Load all puzzle data for retrieval
        print("Loading puzzle data for retrieval...")
        self.all_puzzle_data = self._load_all_puzzles()
        if self.all_puzzle_data and 'features' in self.all_puzzle_data:
            num_puzzles = len(self.all_puzzle_data['features'])
            print(f"Loaded {num_puzzles} puzzles for K-NN search")
        else:
            print(f"Loaded 0 puzzles for K-NN search")

        # Freeze ALL model parameters (we only train embedding)
        print("Freezing model parameters...")
        frozen_count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                frozen_count += 1
        print(f"Frozen {frozen_count} model parameters")

    def _load_all_puzzles(self) -> Dict[int, Dict]:
        """
        Load pre-computed puzzle features for K-NN retrieval.

        Returns:
            Dict with 'features' (name→histogram) and 'ids' (name→puzzle_id)
        """
        if self.data_path is None:
            print("WARNING: No data_path provided, using global mean initialization")
            return {}

        import pickle

        features_path = os.path.join(self.data_path, 'puzzle_features.pkl')
        ids_path = os.path.join(self.data_path, 'puzzle_to_ids.pkl')

        if not os.path.exists(features_path) or not os.path.exists(ids_path):
            print(f"WARNING: Pre-computed features not found")
            print(f"         Run precompute_puzzle_features.py first")
            return {}

        # Load pre-computed features
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        with open(ids_path, 'rb') as f:
            puzzle_to_ids = pickle.load(f)

        print(f"Loaded pre-computed features for {len(features)} puzzles")

        return {
            'features': features,
            'ids': puzzle_to_ids
        }

    def find_k_nearest_puzzles(self, query_histogram: np.ndarray, K: int = 10, exclude_names: List[str] = None) -> List[int]:
        """
        Find K most similar puzzles using pre-computed color histograms.

        Args:
            query_histogram: Color histogram of query puzzle (shape: (10,))
            K: Number of nearest neighbors to return
            exclude_names: Puzzle names to exclude (e.g., the test puzzle itself)

        Returns:
            List of K puzzle_ids sorted by similarity (most similar first)
        """
        if not self.all_puzzle_data or 'features' not in self.all_puzzle_data:
            print("WARNING: No puzzle data loaded, cannot perform retrieval")
            return None  # Signal to caller to use global mean fallback

        features = self.all_puzzle_data['features']
        puzzle_to_ids = self.all_puzzle_data['ids']

        exclude_names = exclude_names or []

        similarities = []

        # Compute cosine similarity with all puzzles
        for puzzle_name, puzzle_hist in features.items():
            # Skip excluded puzzles
            if puzzle_name in exclude_names:
                continue

            # Cosine similarity
            dot_product = np.dot(query_histogram, puzzle_hist)
            norm_query = np.linalg.norm(query_histogram)
            norm_puzzle = np.linalg.norm(puzzle_hist)

            sim = dot_product / (norm_query * norm_puzzle + 1e-8)

            puzzle_id = puzzle_to_ids.get(puzzle_name, -1)
            similarities.append((sim, puzzle_id, puzzle_name))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Return top K puzzle IDs
        top_k_ids = [pid for _, pid, _ in similarities[:K]]

        # Print top 5 for debugging
        print(f"Top {min(5, K)} similar puzzles:")
        for i, (sim, pid, name) in enumerate(similarities[:min(5, K)]):
            print(f"  {i+1}. {name} (ID={pid}): similarity={sim:.4f}")

        return top_k_ids

    def initialize_puzzle_embedding_with_retrieval(
        self,
        train_examples: List[Dict],
        puzzle_id: int,
        device: str,
        exclude_puzzle_name: str = None
    ):
        """
        Initialize puzzle embedding from K nearest neighbors (retrieval-based).

        Args:
            train_examples: Training examples (tensors) to find similar puzzles
            puzzle_id: Puzzle ID to initialize
            device: Device to run on
        """
        with torch.no_grad():
            # Extend embedding table if needed
            current_size = self.puzzle_emb.weights.shape[0]
            if puzzle_id >= current_size:
                emb_dim = self.puzzle_emb.weights.shape[1]
                new_size = max(puzzle_id + 10, int(current_size * 1.1))

                # Create new weights (expanded table)
                new_weights = torch.zeros(new_size, emb_dim, dtype=self.puzzle_emb.weights.dtype, device=device)
                new_weights[:current_size] = self.puzzle_emb.weights.to(device)

                # Update embedding table
                # Note: This replaces the Buffer with a regular Tensor, but it's OK because:
                # 1. We don't use this for gradient-based optimization (we copy to separate Parameter)
                # 2. We only need it as a lookup table for initialization
                # 3. Changes are written back via direct indexing: weights[puzzle_id] = ...
                self.puzzle_emb.weights = new_weights  # Not a Parameter, just a tensor
                print(f"Extended embedding table from {current_size} to {new_size}")

            # Convert tensors to numpy and compute query histogram
            numpy_examples = []
            for ex in train_examples:
                # Tensors are flattened (900,), reshape to (30, 30)
                input_tensor = ex['input'].cpu().numpy().reshape(30, 30)
                output_tensor = ex['output'].cpu().numpy().reshape(30, 30)

                # Input: extract valid grid (remove padding, EOS), shift from [2,11] to [0,9]
                # Find actual grid size by looking for EOS markers (value 1)
                # Valid data is in range [2, 11], EOS is 1, padding is 0
                valid_data = (input_tensor >= 2) & (input_tensor <= 11)
                if valid_data.any():
                    rows = np.where(valid_data.any(axis=1))[0]
                    cols = np.where(valid_data.any(axis=0))[0]
                    if len(rows) > 0 and len(cols) > 0:
                        h_end, w_end = rows[-1] + 1, cols[-1] + 1
                        input_grid = input_tensor[:h_end, :w_end] - 2  # Shift back to [0,9]
                        # Clip to valid range just in case
                        input_grid = np.clip(input_grid, 0, 9)
                    else:
                        input_grid = np.array([[0]])
                else:
                    input_grid = np.array([[0]])

                # Output: extract valid grid (ignore -100 padding)
                valid_mask = output_tensor >= 0
                if valid_mask.any():
                    rows = np.where(valid_mask.any(axis=1))[0]
                    cols = np.where(valid_mask.any(axis=0))[0]
                    if len(rows) > 0 and len(cols) > 0:
                        h_end, w_end = rows[-1] + 1, cols[-1] + 1
                        output_grid = output_tensor[:h_end, :w_end] - 2  # Shift back to [0,9]
                        # Clip to valid range
                        output_grid = np.clip(output_grid, 0, 9)
                    else:
                        output_grid = np.array([[0]])
                else:
                    output_grid = np.array([[0]])

                numpy_examples.append({
                    'input': input_grid,
                    'output': output_grid
                })

            # Compute query histogram from training examples
            from similarity_metrics import precompute_color_histograms
            query_histogram = precompute_color_histograms(numpy_examples)

            # Find K nearest neighbors (excluding current puzzle if specified)
            exclude_names = [exclude_puzzle_name] if exclude_puzzle_name else []
            similar_ids = self.find_k_nearest_puzzles(
                query_histogram,
                K=self.config.k_neighbors,
                exclude_names=exclude_names
            )

            # If retrieval failed (returned None), fall back to global mean
            if similar_ids is None:
                # Use global mean of all trained embeddings
                # Avoid hardcoded puzzle count for different checkpoints
                current_table_size = self.puzzle_emb.weights.shape[0]
                num_trained = min(876406, current_table_size)
                trained_embeddings = self.puzzle_emb.weights[:num_trained]
                init_emb = trained_embeddings.mean(dim=0)
                print(f"Using global mean initialization (retrieval failed, {num_trained} puzzles)")
            else:
                # Get embeddings of similar puzzles
                similar_embeddings = self.puzzle_emb.weights[similar_ids]

                # Initialize as average of similar embeddings
                init_emb = similar_embeddings.mean(dim=0)

            # Add small noise for regularization
            emb_dim = self.puzzle_emb.weights.shape[1]
            noise = torch.randn(emb_dim, device=device) * 0.1

            self.puzzle_emb.weights[puzzle_id] = init_emb.to(device) + noise

            # Report initialization quality
            init_norm = init_emb.norm().item()
            print(f"Initialized from {len(similar_ids)} similar puzzles, norm={init_norm:.4f}")

    def adapt(self, train_examples: List[Dict], puzzle_id: int, device: str, exclude_puzzle_name: str = None) -> Tuple[int, Dict]:
        """
        Adapt puzzle embedding using training examples.

        Args:
            train_examples: List of dicts with 'input' and 'output' tensors
            puzzle_id: Puzzle ID to adapt (should be in reserved range)
            device: Device to run on

        Returns:
            (puzzle_id, history) where history contains loss curve
        """
        if not train_examples:
            raise ValueError("No training examples provided")

        # Initialize embedding with retrieval
        self.initialize_puzzle_embedding_with_retrieval(
            train_examples,
            puzzle_id,
            device,
            exclude_puzzle_name=exclude_puzzle_name
        )

        # Create learnable parameter
        puzzle_emb_param = nn.Parameter(
            self.puzzle_emb.weights[puzzle_id].clone().detach()
        )
        puzzle_emb_param.requires_grad_(True)

        # Setup optimizer for puzzle embedding only
        optimizer = torch.optim.AdamW([puzzle_emb_param], lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_steps
        )

        history = {
            'loss': [],
            'lr': []
        }

        best_loss = float('inf')
        patience_counter = 0

        # Set model to eval mode
        original_training_mode = self.model.training
        self.model.eval()

        # Replace forward method
        original_forward = self.puzzle_emb.forward

        def custom_forward(inputs):
            """Custom forward that uses learnable parameter."""
            batch_size = inputs.shape[0]
            emb_ndim = self.puzzle_emb.weights.shape[1]

            output = torch.zeros(batch_size, emb_ndim, device=inputs.device, dtype=puzzle_emb_param.dtype)

            for i in range(batch_size):
                if inputs[i] == puzzle_id:
                    output[i] = puzzle_emb_param
                else:
                    output[i] = self.puzzle_emb.weights[inputs[i]].to(puzzle_emb_param.dtype)

            return output.to(self.puzzle_emb.cast_to)

        try:
            self.puzzle_emb.forward = custom_forward

            for step in range(self.config.max_steps):
                optimizer.zero_grad()
                total_loss = 0.0

                # Stack all examples into a single batch
                inputs_list = [ex['input'] for ex in train_examples]
                labels_list = [ex['output'] for ex in train_examples]

                inputs_batch = torch.stack(inputs_list).to(device)
                labels_batch = torch.stack(labels_list).to(device)

                batch_size = len(train_examples)
                puzzle_ids = torch.full((batch_size,), puzzle_id, dtype=torch.long, device=device)

                batch = {
                    'inputs': inputs_batch,
                    'labels': labels_batch,
                    'puzzle_identifiers': puzzle_ids
                }

                # Initialize carry ONCE
                carry = self.model.initial_carry(batch)

                # Move carry to device
                if hasattr(carry, 'inner_carry'):
                    InnerCarry = type(carry.inner_carry)
                    Carry = type(carry)

                    inner_carry_dict = {
                        'z_H': carry.inner_carry.z_H.to(device),
                        'z_L': carry.inner_carry.z_L.to(device)
                    }
                    if hasattr(carry.inner_carry, 'c_H') and carry.inner_carry.c_H is not None:
                        inner_carry_dict['c_H'] = carry.inner_carry.c_H.to(device)

                    carry = Carry(
                        inner_carry=InnerCarry(**inner_carry_dict),
                        steps=carry.steps.to(device),
                        halted=carry.halted.to(device),
                        current_data={k: v.to(device) for k, v in carry.current_data.items()}
                    )

                # Run ACT loop until all sequences halt (max 16 steps)
                act_step = 0
                while act_step < self.config.halt_max_steps:
                    carry, outputs = self.model(carry, batch)
                    act_step += 1

                    # Check if all sequences halted
                    if carry.halted.all():
                        break

                # Compute loss on FINAL outputs (after full ACT reasoning)
                logits = outputs['logits']

                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels_batch.reshape(-1),
                    ignore_index=-100
                )

                # Backward with proper scaling
                (loss / len(train_examples)).backward()
                total_loss = loss.item()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([puzzle_emb_param], max_norm=1.0)

                # Optimizer step
                optimizer.step()
                scheduler.step()

                # Record history
                avg_loss = total_loss
                history['loss'].append(avg_loss)
                history['lr'].append(scheduler.get_last_lr()[0])

                # Print progress
                if (step + 1) % 10 == 0:
                    print(f"Step {step+1}/{self.config.max_steps}: loss={avg_loss:.4f}, ACT steps={act_step}")

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
            # Restore original forward
            self.puzzle_emb.forward = original_forward
            self.model.train(original_training_mode)

        # Update embedding table with learned parameters
        with torch.no_grad():
            self.puzzle_emb.weights[puzzle_id] = puzzle_emb_param.data

        return puzzle_id, history
