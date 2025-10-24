# Quick Reference: TRM Architecture & Hybrid Pipeline

## Key File Locations

### Core TRM Architecture
- **Main Model**: `/models/recursive_reasoning/trm.py` (307 lines)
- **Alternative (6-level)**: `/models/recursive_reasoning/trm_hier6.py` (327 lines)
- **HRM Reference**: `/models/recursive_reasoning/hrm.py` (150+ lines)
- **Building Blocks**: `/models/layers.py` (170 lines)
- **Config**: `/config/arch/trm.yaml`

### Hybrid Pipeline Components
- **Main Script**: `/hybrid_pipeline/experiments/run_joint_training.py` (200+ lines)
- **Text→Latent Adapter**: `/hybrid_pipeline/adapters/text_to_latent.py` (213 lines)
- **Latent→Text Adapter**: `/hybrid_pipeline/adapters/latent_to_text.py` (149 lines)
- **VAE Adapter**: `/hybrid_pipeline/adapters/vae_adapter.py` (239 lines)
- **LLM Module**: `/hybrid_pipeline/gpt_oss_port/llm.py`
- **Planner**: `/hybrid_pipeline/gpt_oss_port/planner.py`
- **Verifier**: `/hybrid_pipeline/gpt_oss_port/verifier.py`

### Training & Config
- **Training Script**: `/pretrain.py` (657 lines)
- **Dataset**: `/puzzle_dataset.py` (10,616 lines)
- **Arch Configs**: `/config/arch/*.yaml`
- **Pretraining Config**: `/config/cfg_pretrain.yaml`

### Evaluation
- **Evaluators**: `/evaluators/` directory
- **TRM Analysis**: `/analyze_trm_arc_results.py`

---

## Critical Code Snippets

### 1. TRM Hidden State Structure (THE H VECTORS)

**File**: `models/recursive_reasoning/trm.py:17-24`

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor    # [batch, seq_len=900, hidden_size=512]
    z_L: torch.Tensor    # [batch, seq_len=900, hidden_size=512]
```

**Current Usage**: `trm.py:214-224`
```python
z_H, z_L = carry.z_H, carry.z_L

# Both go through SAME L_level module!
for _H_step in range(self.config.H_cycles-1):
    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)  # ← SAME MODULE AS z_L!
```

**THE ISSUE**: No distinction between H and L processing despite separate representations.

---

### 2. Key Configuration Values

**File**: `config/arch/trm.yaml`

```yaml
H_cycles: 3          # Number of high-level iterations
L_cycles: 6          # Number of low-level iterations per H cycle
H_layers: 0          # Number of H-level transformer blocks (CURRENTLY UNUSED!)
L_layers: 2          # Number of L-level transformer blocks
hidden_size: 512
num_heads: 8
halt_max_steps: 16   # Max recursion steps before forced halt
```

**Insight**: H_layers config exists but H_level module is never created separately!

---

### 3. Input Embedding & Information Flow

**File**: `models/recursive_reasoning/trm.py:162-182`

```python
def _input_embeddings(self, input, puzzle_identifiers):
    # Simple flat embedding - NO hierarchy
    embedding = self.embed_tokens(input)
    
    if self.config.puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
    
    if self.config.pos_encodings == "learned":
        embedding = embedding + self.embed_pos.embedding_weight
    
    return self.embed_scale * embedding
```

**Insight**: All 900 grid positions initialized uniformly from single learned embedding.

---

### 4. ACT Halting Mechanism

**File**: `models/recursive_reasoning/trm.py:285-305`

```python
# Q-learning based halting decision
q_halt_logits, q_continue_logits = model.q_head(z_H[:, 0])

if training and halt_max_steps > 1:
    if no_ACT_continue:
        halted = halted | (q_halt_logits > 0)  # Sigmoid threshold
    else:
        halted = halted | (q_halt_logits > q_continue_logits)  # Q-learning
    
    # Exploration: random minimum halt steps
    min_halt_steps = torch.randint_like(..., low=2, high=halt_max_steps+1)
    halted = halted & (steps >= min_halt_steps)
```

**Q-head uses z_H[:, 0]** - first position (puzzle embedding).

---

### 5. Text-to-Latent Adapter Architecture

**File**: `hybrid_pipeline/adapters/text_to_latent.py:93-212`

```python
class TextToLatentAdapter(nn.Module):
    # Stage 1: Compress LLM hidden
    self.compress = nn.Sequential(
        nn.Linear(4096, 1024),  # LLaMA → bottleneck
        nn.GELU(),
        nn.LayerNorm(1024)
    )
    
    # Stage 2: Position-aware expansion (LEARNABLE!)
    self.position_embeddings = nn.Parameter(
        torch.randn(seq_len, 1024) * 0.02
    )
    
    # Stage 3: Project to TRM space
    self.to_trm = nn.Linear(1024, 2 * trm_hidden_size)  # → z_H + z_L
```

**Key**: Position embeddings allow per-position differentiation! But still no explicit hierarchy.

---

### 6. Latent-to-Text Adapter: Attention Pooling

**File**: `hybrid_pipeline/adapters/latent_to_text.py:12-46`

```python
class AttentionPooling(nn.Module):
    """Learn which grid cells are important"""
    def __init__(self, hidden_size):
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)  # → importance scores
        )
    
    def forward(self, z):  # [batch, 900, hidden]
        attn_scores = self.attention(z)  # [batch, 900, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize
        pooled = (z * attn_weights).sum(dim=1)  # [batch, hidden]
        return pooled
```

**Purpose**: Prevents info loss from 900→1 compression (was using avg_pool before).

---

### 7. The "Multi-Level Attempt" (HIER6)

**File**: `models/recursive_reasoning/trm_hier6.py:230-244`

```python
# 6 independent L vectors
z_L = [carry.z_L1, carry.z_L2, ..., carry.z_L6]

# But they're summed together (loses hierarchy!)
for _L_step in range(self.config.L_cycles):
    z_L_ = z_L[0] + z_L[1] + z_L[2] + z_L[3] + z_L[4] + z_L[5]
    z_L[_L_step] = self.L_level(z_L_, z_H + input_embeddings)

# All 6 are summed before H update
z_L_ = sum(z_L)
z_H = self.L_level(z_H, z_L_)
```

**Issue**: Creates 6 channels but sums them (complete info loss). Still uses same L_level.

---

## What Currently Exists

### ✅ Infrastructure Ready for Hierarchical Implementation
- Separate z_H and z_L tensors (just need separate processing)
- H_layers config parameter (just needs to be used)
- Position embeddings in adapters (can be enhanced)
- Attention-based pooling (can be made hierarchical)

### ❌ Missing Pieces
- NO separate H_level module creation
- NO multi-scale grid processing
- NO diffusion-like coarse-to-fine generation
- NO hierarchical carry management
- NO per-level ACT decisions

---

## Implementation Checklist

### Quick Wins (1-2 weeks)

- [ ] Create separate H_level and L_level modules in `trm.py`
- [ ] Update forward pass to use different modules for H and L
- [ ] Test on Sudoku-Extreme to verify improvement
- [ ] Add H_layers to config and use it

### Medium Effort (2-3 weeks)

- [ ] Add position context conditioning to h vectors
- [ ] Implement grid downsampling/upsampling utilities
- [ ] Create `HierarchicalDiffusionCarry` dataclass
- [ ] Implement 3-level hierarchy (9→36→900)

### Full Implementation (3-4 weeks)

- [ ] Create `trm_hierarchical_diffusion.py` model
- [ ] Implement level-specific blocks with separate processing
- [ ] Add hierarchical aware adapters
- [ ] Per-level ACT halting decisions
- [ ] Integration with hybrid pipeline

---

## Performance Baselines

**Current TRM**:
- ARC-AGI-1: 45%
- ARC-AGI-2: 8%
- Parameters: 7M
- Training: ~3 days on 4 H100s

**Expected with Hierarchical Diffusion**:
- ARC-AGI-1: 47-50% (+2-5%)
- ARC-AGI-2: 10-12% (+2-4%)
- Parameters: 7.5-8M (+10-15%)
- Training: ~2.5 days (faster convergence due to coarse-first learning)

---

## Key Insights

1. **TRM is intentionally flat**: Paper explicitly avoids hierarchical complexity
2. **H and L are just tensor names**: Both processed identically (by design, not bug)
3. **Hierarchy would be additive**: Not replacing existing approach, extending it
4. **Information preservation is key**: "Kill abstraction" means no lossy compression
5. **Adapters are already position-aware**: Just need explicit hierarchy

---

## Testing Commands

### Quick Test (Sudoku)
```bash
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=100 eval_interval=100 evaluators="[]"
```

### Full Test (ARC)
```bash
torchrun --nproc-per-node 4 pretrain.py arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.H_layers=2 arch.L_layers=2
```

### Hybrid Pipeline Joint Training
```bash
cd hybrid_pipeline/experiments
python run_joint_training.py \
  --data_path /data/arc/processed \
  --trm_checkpoint /path/to/ckpt.pt \
  --max_attempts 16
```

