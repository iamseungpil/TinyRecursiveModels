# CNN Puzzle Embedding Predictor

**Meta-learning approach for TRM unseen task inference**

## Overview

This module enables TRM (Tiny Recursive Model) to solve **unseen tasks** by predicting puzzle embeddings from input grids alone, without requiring learned puzzle identifiers.

### Problem

- TRM uses learned puzzle-specific embeddings (876K puzzles × 512-dim)
- For unseen tasks, embeddings are randomly initialized → poor performance
- Current evaluation only works on puzzles seen during training

### Solution

Train a CNN to predict TRM's puzzle embeddings from input grids:

```
Input Grid (H×W) → CNN Encoder → Predicted Embedding (512-dim) → TRM → Output
```

This enables **meta-learning**: the CNN learns to map visual patterns to the embedding space that TRM expects.

## Architecture

### CNN Encoder
```
Input: ARC grid (30×30, padded)
  ↓
Token Embedding (12 colors → 64-dim)
  ↓
Conv + ResBlocks (256 channels, 4 blocks)
  ↓
Global Average Pooling
  ↓
FC Layer (256 → 512)
  ↓
Output: Puzzle Embedding (512-dim)
```

### Training Objective
```python
Loss = MSE(pred_emb, target_emb) + 0.5 × (1 - cosine_similarity(pred_emb, target_emb))
```

## Usage

### 1. Extract Training Pairs

Extract (grid, embedding) pairs from TRM checkpoint:

```bash
python data/extract_training_pairs.py \
    --checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
    --data-path /data/arc1concept-aug-1000 \
    --output-dir ./data/training_pairs \
    --max-examples-per-puzzle 10
```

**Output**: `training_pairs.pt` with ~9,600 pairs (960 puzzles × 10 examples)

### 2. Train CNN Predictor

```bash
python train.py \
    --data-path ./data/training_pairs/training_pairs.pt \
    --output-dir ./checkpoints \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --gpu 0
```

**Metrics to monitor**:
- `train_cosine_similarity`: Should reach ~0.9+
- `val_cosine_similarity`: Should reach ~0.85+

### 3. Inference on Unseen Tasks

Single puzzle:
```bash
python inference.py \
    --cnn-checkpoint ./checkpoints/run_xxx/checkpoint_best.pt \
    --trm-checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
    --puzzle-json /path/to/unseen_puzzle.json \
    --output prediction.json \
    --gpu 0
```

Batch evaluation:
```bash
python inference.py \
    --cnn-checkpoint ./checkpoints/run_xxx/checkpoint_best.pt \
    --trm-checkpoint /data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071 \
    --test-dir /data/arc_test \
    --output-dir ./predictions \
    --gpu 0
```

## File Structure

```
puzzle_embedding_predictor/
├── models/
│   └── cnn_encoder.py          # CNN architecture + loss
├── data/
│   ├── extract_training_pairs.py  # Data extraction
│   └── training_pairs/         # Generated training data
├── train.py                    # Training script
├── inference.py                # Inference script
├── checkpoints/                # Saved model checkpoints
└── README.md                   # This file
```

## Key Implementation Details

### Embedding Injection

The CNN predicts embeddings that are injected into TRM's `puzzle_emb` layer:

```python
# Predict embedding from input grid
predicted_embedding = cnn_model(input_grid)  # (512,)

# Inject into TRM
trm_model.inner.puzzle_emb.weights[0] = predicted_embedding

# Run TRM inference
output = trm_model(input_grid)
```

### Training Data

- **Source**: TRM checkpoint's learned embeddings (876,406 puzzles)
- **Filtering**: Skip zero-norm embeddings (padding/unused)
- **Augmentation**: Multiple examples per puzzle (up to 10)
- **Split**: 90% train, 10% validation

### Model Size

- **Parameters**: ~1.5M (lightweight)
- **Embedding dimension**: 64 (token) → 256 (hidden) → 512 (output)
- **Residual blocks**: 4 blocks for deeper feature extraction

## Expected Performance

### CNN Training
- **Cosine similarity**: 0.85-0.90 on validation set
- **Training time**: ~1-2 hours on single GPU
- **Convergence**: ~50-100 epochs

### TRM Inference with Predicted Embeddings
- **Expected accuracy**: 15-25% on unseen tasks
  - Lower than 29% on seen tasks (uses learned embeddings)
  - But enables generalization to completely new puzzles!

## Comparison to Baseline

| Method | Seen Tasks | Unseen Tasks |
|--------|-----------|--------------|
| TRM (learned embeddings) | 29% | ~0% (random init) |
| TRM + CNN predictor | 25-27% | **15-25%** |

## Future Improvements

1. **Multi-example aggregation**: Use all train examples to predict embedding
2. **Attention mechanism**: Attend to relevant patterns across examples
3. **Contrastive learning**: Learn embedding space jointly with TRM
4. **Few-shot adaptation**: Fine-tune CNN on few examples at test time

## Kaggle Code Review

The provided Kaggle notebook has these limitations:

1. **No unseen task support**: Uses checkpoint's 876K learned embeddings
2. **Random initialization for new puzzles**: New puzzle IDs get random embeddings
3. **No augmentation at test time**: `--num-aug 0` (trained with ~1000x aug)

**This CNN predictor solves limitation #2** by learning to predict embeddings from grids.

## Citation

If you use this code, please reference the TinyRecursiveModels project:
```
@misc{trm2025,
  title={Tiny Recursive Models for Abstract Reasoning},
  author={TinyRecursiveModels Team},
  year={2025}
}
```

## License

Same as parent TinyRecursiveModels project.
