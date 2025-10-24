# Hybrid LLaMA + TRM Pipeline for ARC-AGI

Complete pipeline for training hybrid models that combine:
- **LLaMA-8B** for text reasoning
- **TRM (Tiny Recursive Models)** for grid generation
- **Thin adapters** for LLM↔TRM communication

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID MODEL PIPELINE                     │
└─────────────────────────────────────────────────────────────┘

1. Text Reasoning (LLaMA-8B, frozen)
   ├─ Input: ARC problem description
   └─ Output: Reasoning text + hidden state (4096-dim)

2. Text→Latent Adapter (trainable)
   ├─ Input: LLaMA hidden state [batch, 4096]
   └─ Output: TRM initial carry (z_H, z_L) [batch, 900, 512]

3. Grid Generation (TRM, trainable)
   ├─ Input: z_init from adapter + tokenized problem
   ├─ Process: ACT halting (max 16 steps)
   └─ Output: Predicted grid tokens

4. Latent→Text Adapter (trainable)
   ├─ Input: TRM carry state (z_H, z_L)
   └─ Output: Feedback latent [batch, 4096] for re-injection

5. Self-Correction Loop
   ├─ If prediction incorrect: Generate feedback
   ├─ Re-inject feedback as latent prefix to LLaMA
   └─ Repeat up to 16 attempts (like TRM halt_max_steps)
```

## 📁 Directory Structure

```
hybrid_pipeline/
├── gpt_oss_port/           # GPT-OSS utilities (NO LLM inference)
│   ├── grid_utils.py       # Grid formatting, parsing, validation
│   └── dataset_access.py   # Wrapper for existing dataset/ module
├── adapters/               # LLM↔TRM interface adapters
│   ├── text_to_latent.py   # LLaMA → TRM projection
│   ├── feedback_formatter.py  # Grid → feedback text
│   └── tests/
├── trm_pretrain/           # TRM pretraining (single GPU)
│   ├── train_trm.py        # Simplified pretrain.py
│   ├── eval_trm.py         # TRM evaluation
│   └── tests/
├── experiments/            # Joint training orchestration
│   ├── run_joint_training.py   # Main joint training script
│   ├── config_joint.yaml       # Configuration file
│   ├── run_trm_pretrain.sh     # TRM pretraining script
│   ├── run_trm_eval.sh         # TRM evaluation script
│   └── run_joint_training.sh   # Joint training script
└── docs/
    └── README.md           # This file
```

## 🚀 Quick Start

### 1. Prepare ARC Dataset

```bash
cd /home/ubuntu/TinyRecursiveModels/dataset
python build_arc_dataset.py \
    --input_file_prefix /path/to/arc \
    --output_dir /data/arc/processed \
    --subsets training evaluation \
    --test_set_name evaluation \
    --num_aug 100
```

### 2. Pretrain TRM (Optional but Recommended)

```bash
cd /home/ubuntu/TinyRecursiveModels/hybrid_pipeline/experiments
./run_trm_pretrain.sh /data/arc/processed /data/trm/pretrain cuda:0
```

This trains TRM on grid generation (without LLaMA) for 50 epochs.

### 3. Evaluate Pretrained TRM

```bash
./run_trm_eval.sh \
    /data/trm/pretrain/checkpoint_step_5000.pt \
    /data/arc/processed \
    cuda:0
```

### 4. Joint Training (LLaMA + TRM)

```bash
./run_joint_training.sh \
    /data/arc/processed \
    /data/trm/joint_training \
    /data/trm/pretrain/checkpoint_step_5000.pt \
    cuda:0
```

This trains adapters + TRM with frozen LLaMA.

## 🔧 Configuration

Edit `experiments/config_joint.yaml` to customize:

```yaml
# LLaMA Configuration
llama:
  model: "meta-llama/Llama-3.2-8B-Instruct"
  frozen: true  # Keep LLaMA frozen

# TRM Configuration
trm:
  hidden_size: 512
  halt_max_steps: 16  # Max self-correction attempts

# Training
training:
  batch_size: 1
  max_attempts: 16
  epochs: 10
  lr: 0.0001
```

## 📊 Logging and Monitoring

All experiments log to WandB:
- **TRM Pretraining**: Project `arc-trm-pretrain`
- **Joint Training**: Project `arc-hybrid-joint`

Logs and checkpoints are saved to `/data/trm/` by default.

## 🧪 Running Tests

```bash
# Test adapters
cd hybrid_pipeline/adapters/tests
python test_projection.py

# Test TRM forward pass
cd hybrid_pipeline/trm_pretrain/tests
python test_trm_forward.py
```

## 🔑 Key Design Principles

### 1. **No Code Duplication**
- All imports from existing `/home/ubuntu/TinyRecursiveModels/models/` and `dataset/`
- NO reimplementation of TRM, losses, or data loading

### 2. **Modular Architecture**
- Clean separation: text reasoning | adapters | grid generation
- Thin adapters (2-3 layer MLPs) for efficient training

### 3. **Gradient Flow**
- Adapters + TRM trainable
- LLaMA frozen (optional fine-tuning)
- NO `.detach()` breaking gradient flow

### 4. **TRM-Style Self-Correction**
- Max 16 attempts (matches TRM `halt_max_steps`)
- Feedback re-injection as latent prefix
- ACT halting in TRM grid generation

### 5. **Data Handling**
- All paths via CLI arguments (no hard-coded paths)
- Logs/checkpoints to `/data/` directory
- Reuses existing ARC data processing pipeline

## 📚 Module Reference

### `gpt_oss_port/grid_utils.py`
Utilities extracted from GPT-OSS sequential_validation_v4.py:
- `grid_to_string()`: Convert numpy array → string
- `string_to_grid()`: Parse string → numpy array
- `compare_grids()`: Validate prediction accuracy

### `adapters/text_to_latent.py`
- `TextToLatentAdapter`: LLaMA [4096] → TRM (z_H, z_L) [900, 512]
- `LatentToTextAdapter`: TRM (z_H, z_L) → LLaMA feedback [4096]

### `adapters/feedback_formatter.py`
- `tokens_to_grid()`: Convert TRM tokens → 2D grid
- `grid_to_feedback()`: Generate concise feedback text (max 5 errors)
- `format_problem_text()`: Format ARC problem with history

### `trm_pretrain/train_trm.py`
Simplified single-GPU TRM pretraining:
- Reuses `TinyRecursiveReasoningModel_ACTV1` from `models/`
- Cosine learning rate schedule with warmup
- Checkpoint saving every N steps
- WandB logging

### `experiments/run_joint_training.py`
Main joint training orchestration:
- `HybridModel`: Combined LLaMA + adapters + TRM
- Self-correction loop with feedback re-injection
- ACT-style multi-attempt training (max 16 steps)
- Gradient flow through adapters + TRM only

## 🐛 Debugging Tips

### Problem: CUDA out of memory
**Solution**: Reduce batch size to 1, reduce `max_attempts` to 8

### Problem: TRM not learning
**Solution**: Check gradient flow, verify no `.detach()` calls, ensure learning rate not too low

### Problem: LLaMA generating random text
**Solution**: Verify frozen LLaMA, check latent prefix injection, use greedy decoding

### Problem: Slow training
**Solution**: Use pretrained TRM checkpoint, reduce TRM `halt_max_steps`, use mixed precision

## 📝 Citation

This pipeline combines:
- **TRM**: Tiny Recursive Models (existing `/home/ubuntu/TinyRecursiveModels/`)
- **LLaMA**: Meta LLaMA-3.2-8B-Instruct
- **ARC-AGI**: Abstract Reasoning Corpus

## 🔗 Related Files

- Original TRM: `/home/ubuntu/TinyRecursiveModels/models/recursive_reasoning/trm.py`
- Original pretrain: `/home/ubuntu/TinyRecursiveModels/pretrain.py`
- Dataset module: `/home/ubuntu/TinyRecursiveModels/dataset/build_arc_dataset.py`
- GPT-OSS reference: `/home/ubuntu/gpt_oss_arc_final/sequential_validation_v4.py`
- Previous attempt: `/home/ubuntu/TinyRecursiveModels/gpt-integration/` (deprecated)

## ⚠️ Important Notes

1. **Data locations**: All datasets/logs/checkpoints MUST be in `/data/` directory
2. **No hard-coded paths**: All paths via CLI arguments or config files
3. **Module reuse**: NEVER duplicate code from `models/` or `dataset/`
4. **Gradient flow**: Always check gradient flow through adapters
5. **Memory**: Use batch_size=1 for 8B LLaMA + TRM on single GPU

## 📞 Support

For issues or questions:
1. Check logs in `/data/trm/*/train_*.log`
2. Verify WandB dashboard for metrics
3. Test individual components with unit tests
4. Review architecture diagram above

---

**Last updated**: 2025-10-13
**Author**: Claude Code
**Status**: Production-ready hybrid pipeline
