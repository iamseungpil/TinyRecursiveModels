# TRM Architecture & Hybrid Pipeline Exploration Report

## Executive Summary

This report provides a thorough analysis of the Tiny Recursive Models (TRM) architecture and the hybrid_pipeline implementation. The exploration covers:
1. **TRM Core Architecture** - RNN structure, h vectors (z_H, z_L), and information flow
2. **Hybrid Pipeline** - LLaMA + TRM integration with adapters
3. **Current Approaches** - Existing hierarchical and diffusion-related implementations
4. **Recommendations** - How to apply "kill abstraction + hierarchical diffusion" to TRM

---

## Part 1: TRM Core Architecture Analysis

### 1.1 Overall Architecture Overview

**Location**: `/home/user/TinyRecursiveModels/models/recursive_reasoning/trm.py` (307 lines)

TRM achieves 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with only **7M parameters** through recursive reasoning. Unlike HRM (Hierarchical Reasoning Model), TRM simplifies the approach by removing biological arguments and mathematical theorems.

**Key Design Principle**: The model uses **ACT (Adaptive Computation Time)** with a learnable halting mechanism. Instead of a fixed depth, it determines dynamically how many reasoning steps are needed.

### 1.2 RNN Hidden State Structure (The h vectors)

**Current Implementation** (trm.py, lines 17-24):

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor    # High-level carry: [batch, seq_len, hidden_size]
    z_L: torch.Tensor    # Low-level carry: [batch, seq_len, hidden_size]
```

**Current Usage** (lines 214-224):
```python
z_H, z_L = carry.z_H, carry.z_L

# H_cycles-1 without grad
with torch.no_grad():
    for _H_step in range(self.config.H_cycles-1):
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)  # ← ISSUE: Uses L_level for H update

# 1 with grad
for _L_step in range(self.config.L_cycles):
    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
z_H = self.L_level(z_H, z_L, **seq_info)  # ← Same processing, no distinction
```

**Critical Observation**: Despite having separate z_H and z_L representations:
- Both use the **SAME** `L_level` module
- There is **NO hierarchical distinction** in processing
- The "H" and "L" are just different tensors passed through identical processing
- This design intentionally avoids hierarchical complexity (per paper motivation)

### 1.3 Current Abstraction Handling

**Current Status**: TRM **explicitly avoids abstractions** through a flat, unified architecture:

**Input Embedding** (lines 162-182):
- Simple token embedding + puzzle embedding
- No hierarchical clustering or coarse-to-fine decomposition
- Direct projection to full sequence length

**Forward Pass** (lines 212-224):
- All sequence positions processed at the same "level"
- No multi-scale or pyramid structure
- Same attention patterns across all positions

### 1.4 Information Flow

```
Input Embeddings [batch, seq_len, hidden_size]
         ↓
    z_H and z_L initialization [batch, seq_len, hidden_size]
         ↓
    H_CYCLES-1 steps (without gradients):
    ├─ L_CYCLES steps of L_level updates (z_L)
    └─ One H_level update (z_H)  [currently: same as L_level]
         ↓
    1 final step (with gradients):
    ├─ L_CYCLES steps of L_level updates (z_L)
    └─ One H_level update (z_H)
         ↓
    LM Head Output [batch, seq_len, vocab_size]
    Q Head Output [batch, 1, 2] for ACT halting decisions
```

**Key Insight**: The model has **independent H and L vectors** but they're processed identically. This is intentional to keep the model simple, but creates an opportunity for hierarchical differentiation.

---

## Part 2: Alternative Model Variants

### 2.1 TRM_HIER6.PY - Multi-Level Attempt

**Location**: `/home/user/TinyRecursiveModels/models/recursive_reasoning/trm_hier6.py` (327 lines)

This is an experimental hierarchical variant with **6 independent L-level vectors**:

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L1: torch.Tensor
    z_L2: torch.Tensor
    z_L3: torch.Tensor
    z_L4: torch.Tensor
    z_L5: torch.Tensor
    z_L6: torch.Tensor
```

**Processing** (lines 240-244):
```python
# All L vectors are summed together, then used for H update
for _L_step in range(self.config.L_cycles):
    z_L_ = z_L[0] + z_L[1] + z_L[2] + z_L[3] + z_L[4] + z_L[5]
    z_L[_L_step] = self.L_level(z_L_, z_H + input_embeddings, **seq_info)
    
z_L_ = z_L[0] + z_L[1] + z_L[2] + z_L[3] + z_L[4] + z_L[5]
z_H = self.L_level(z_H, z_L_, **seq_info)
```

**Issues**:
- Simply sums all L vectors (destroys hierarchical structure)
- All L vectors still use the same `L_level` module
- No explicit abstraction or hierarchical grouping of grid positions

### 2.2 HRM.PY - Original Hierarchical Reasoning Model

**Location**: `/home/user/TinyRecursiveModels/models/recursive_reasoning/hrm.py` (lines 1-150)

The original HRM that TRM simplified from:
- Has both `H_level` and `L_level` as separate ModuleLists (line 148-149)
- Still unified architecture without multi-scale processing

---

## Part 3: Hybrid Pipeline Implementation

### 3.1 Overall Architecture

**Location**: `/home/user/TinyRecursiveModels/hybrid_pipeline/`

The hybrid pipeline integrates:
1. **LLaMA (frozen)** - Text understanding via GPT-OSS model
2. **Adapters** - Bridge between text and grid representations
3. **TRM** - Grid generation with recursive reasoning
4. **Planner** - Multi-attempt loop with feedback

### 3.2 Text-to-Latent Adapter (TextToLatentAdapter)

**Location**: `/home/user/TinyRecursiveModels/hybrid_pipeline/adapters/text_to_latent.py` (213 lines)

**Architecture**:
```
LLM hidden [batch, 4096]
    ↓
Compress [4096 → 1024]: Linear + GELU + LayerNorm
    ↓
Position-Aware Expansion [batch, 1024] → [batch, seq_len, 1024]
    ├─ Option 1: Simple expand (replicates across positions)
    ├─ Option 2: Position embeddings (learns per-position features)
    └─ Option 3: Cross-attention bridge (attends over LLM sequence)
    ↓
To TRM [1024 → 1024] → Split into z_H and z_L
    ↓
z_H, z_L [batch, seq_len, 512]
```

**Current Features** (lines 93-212):
- **Position embeddings** (learnable per-position): Allows different initialization for each grid cell
- **Cross-attention option**: Maps LLM token sequence to TRM slots using attention
- **Bottleneck design**: 4096 → 1024 → TRM reduces parameters from 1.9B to ~6M

**Limitation**: All grid positions initialized identically (without cross-attention) or with only positional differentiation. No hierarchical clustering or coarse-to-fine initialization.

### 3.3 Latent-to-Text Adapter (LatentToTextAdapter)

**Location**: `/home/user/TinyRecursiveModels/hybrid_pipeline/adapters/latent_to_text.py` (149 lines)

**Architecture**:
```
z_H [batch, seq_len, 512]
z_L [batch, seq_len, 512]
    ↓
Attention Pooling: Learns importance weights for each grid cell
    ├─ AttentionPooling(z_H) → [batch, 512]
    └─ AttentionPooling(z_L) → [batch, 512]
    ↓
Concatenate [batch, 1024]
    ↓
Project to LLM [1024 → 2048 → 4096]
    ↓
LLM latent prefix [batch, 4096]
```

**Key Innovation** (lines 12-46): **Attention-based pooling** instead of average pooling
- Learns which grid cells are important
- Prevents information loss from 900→1 projection
- Replaces simple average pooling that loses 99.9% of information

**Current Limitation**: Still compresses entire grid to single token. No hierarchical structure preservation.

### 3.4 VAE Adapter (Optional Enhancement)

**Location**: `/home/user/TinyRecursiveModels/hybrid_pipeline/adapters/vae_adapter.py` (239 lines)

Adds variational bottleneck with reconstruction loss:
```
- Encoder: LLM → (mu, logvar) for latent sampling
- Reparameterization trick: z = mu + std * epsilon
- Decoder: TRM latent → reconstructed LLM hidden
- Loss: Reconstruction MSE + β * KL divergence
```

Used when `use_vae_adapter=True` in config. Helps preserve information bidirectionally.

### 3.5 Current Hybrid Pipeline Limitations

**No Hierarchical Information Processing**:
- Adapters don't explicitly model grid structure
- No abstraction levels (e.g., regions → patterns → full grid)
- All grid positions treated independently

---

## Part 4: Existing Hierarchical/Diffusion Attempts

### 4.1 Hierarchical Layers (Archived)

**Location**: `/home/user/TinyRecursiveModels/gpt-integration/_archive_old_implementation/components/hierarchical_layers.py` (223 lines)

This is an archived component with:
- `HierarchicalReasoningBlock` with separate attention and MLP
- `RMSNorm`, `RotaryEmbedding`, `MultiHeadAttention`
- Pre-norm residual connections

**Status**: Archived - not used in current TRM or hybrid pipeline.

### 4.2 Missing Diffusion Components

**Current Status**: **NO diffusion-related code found** in TRM or hybrid pipeline.

Searched locations:
- `/models/` - No diffusion modules
- `/hybrid_pipeline/` - No diffusion adapters
- Config files - No diffusion settings
- Documentation - No diffusion discussion

---

## Part 5: Detailed Recommendations for "Kill Abstraction + Hierarchical Diffusion"

### 5.1 What "Kill Abstraction" Means in TRM Context

**Current State**: TRM avoids explicit abstractions by processing all sequence positions uniformly.

**What "Kill Abstraction" Means**:
- Remove learned abstractions that hide information
- Don't compress grid information to intermediate semantic representations
- Keep all position-level information available throughout reasoning
- Avoid clustering grid cells into super-nodes that lose spatial structure

**Recommendation 1.1: Make z_H and z_L Truly Distinct**

**Current Problem**: Both use same `L_level` module (trm.py, line 220)

**Solution**: Create separate H and L processing modules
```python
# In TinyRecursiveReasoningModel_ACTV1_Inner.__init__
self.H_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
    layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.H_layers)]
)
self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
    layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
)

# In forward pass
z_H = self.H_level(z_H, z_L, **seq_info)  # Different module!
z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
```

**Impact**: Allows H and L to learn different abstraction levels
- z_H: Coarse features, long-range dependencies
- z_L: Fine details, local patterns

**Recommendation 1.2: Add Position-Level Conditioning**

**Problem**: Grid positions have no semantic role distinction

**Solution**: Embed position context that influences what abstraction level is needed
```python
class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        # Learn position importance and abstraction needs
        self.position_context = nn.Parameter(torch.randn(seq_len, hidden_size))
        self.position_scale = nn.Parameter(torch.ones(seq_len, 1))
    
    def forward(self, h):
        # h: [batch, seq_len, hidden_size]
        return h * self.position_scale + self.position_context

# Use in forward:
z_L = self.position_embed(z_L)
```

### 5.2 Hierarchical Diffusion Implementation

**Concept**: Use diffusion-like reverse process to build up grid from abstraction levels.

#### 5.2.1 Hierarchical Diffusion Architecture

**Three-Level Hierarchy**:
1. **Level 0 (Coarsest)**: 30x30 → 3x3 (9 super-cells)
2. **Level 1 (Medium)**: 3x3 → 6x6 (36 regions)
3. **Level 2 (Finest)**: 6x6 → 30x30 (900 cells)

**Forward Process** (Coarse-to-Fine):
```
Input [30x30 grid, 900 tokens]
    ↓
Downsample to 3x3 [9 tokens] + position embeddings
    ↓
Process at Level 0 (coarse abstraction)
    ↓
Upsample to 6x6 [36 tokens] + diffuse from Level 0
    ↓
Process at Level 1 (medium abstraction)
    ↓
Upsample to 30x30 [900 tokens] + diffuse from Level 1
    ↓
Process at Level 2 (fine detail)
    ↓
Output [30x30 grid, 900 tokens]
```

#### 5.2.2 Implementation in TRM

**File to Create**: `/models/recursive_reasoning/trm_hierarchical_diffusion.py`

**Key Components**:

```python
class HierarchicalDiffusionCarry:
    """Carry state with explicit hierarchy"""
    z_coarse: torch.Tensor    # [batch, 9, hidden]
    z_medium: torch.Tensor    # [batch, 36, hidden]
    z_fine: torch.Tensor      # [batch, 900, hidden]
    
class HierarchicalDiffusionBlock(nn.Module):
    """Process at one abstraction level"""
    def __init__(self, level_name: str, seq_len: int, hidden_size: int):
        super().__init__()
        self.level_attention = Attention(hidden_size, ...)
        self.level_mlp = SwiGLU(hidden_size, ...)
        # NO compression - keep all position info
    
    def forward(self, z: torch.Tensor, prior_z: torch.Tensor = None):
        # prior_z: upsampled from coarser level
        if prior_z is not None:
            z = z + prior_z  # Inject coarse information
        return self.level_attention(...) + self.level_mlp(...)

class UpsamplingDiffusion(nn.Module):
    """Upsample from coarse to fine while preserving structure"""
    def forward(self, z_coarse: torch.Tensor, target_shape: Tuple):
        # Bilinear/nearest upsample
        return F.interpolate(z_coarse, size=target_shape, mode='bilinear')
```

**Usage in TinyRecursiveReasoningModel**:

```python
class TinyRecursiveReasoningModel_HierarchicalDiffusion(nn.Module):
    def forward(self, carry, batch):
        # Initialize carries at all levels
        z_coarse, z_medium, z_fine = carry
        
        # Process coarse level
        for _ in range(H_cycles):
            z_coarse = self.coarse_level(z_coarse, input_embeddings[downsampled])
        
        # Upsample and refine
        z_medium_prior = upsample(z_coarse)
        for _ in range(H_cycles):
            z_medium = self.medium_level(z_medium, z_medium_prior + input_embeddings[medium])
        
        # Final fine refinement
        z_fine_prior = upsample(z_medium)
        for _ in range(H_cycles):
            z_fine = self.fine_level(z_fine, z_fine_prior + input_embeddings)
        
        return z_coarse, z_medium, z_fine, outputs
```

**Benefits**:
- No abstraction "killing" - all info preserved
- Hierarchical coarse-to-fine reasoning
- Lower memory footprint at coarse levels
- Better for learning long-range dependencies first

### 5.3 Modifying the Adapters for Hierarchical Awareness

#### 5.3.1 Updated TextToLatentAdapter

**File**: Update `/hybrid_pipeline/adapters/text_to_latent.py`

**Enhancement**: Use hierarchy information from LLM

```python
class HierarchicalTextToLatentAdapter(TextToLatentAdapter):
    def __init__(self, ..., enable_hierarchical_init=True):
        super().__init__(...)
        if enable_hierarchical_init:
            # Learn separate initializations for coarse/medium/fine levels
            self.coarse_init = nn.Linear(bottleneck_dim, trm_hidden_size)
            self.medium_init = nn.Linear(bottleneck_dim, trm_hidden_size)
            self.fine_init = nn.Linear(bottleneck_dim, trm_hidden_size)
    
    def forward(self, llm_hidden_state, **kwargs):
        compressed = self.compress(llm_hidden_state)
        
        # Different initializations for each level
        z_coarse = self.coarse_init(compressed)
        z_medium = self.medium_init(compressed)
        z_fine = self.fine_init(compressed)
        
        # Broadcast to appropriate shapes
        z_coarse = z_coarse.unsqueeze(1).expand(-1, 9, -1)
        z_medium = z_medium.unsqueeze(1).expand(-1, 36, -1)
        z_fine = z_fine.unsqueeze(1).expand(-1, 900, -1)
        
        return z_coarse, z_medium, z_fine
```

#### 5.3.2 Updated LatentToTextAdapter

**Enhancement**: Preserve hierarchical structure in pooling

```python
class HierarchicalLatentToTextAdapter(LatentToTextAdapter):
    def forward(self, z_coarse, z_medium, z_fine):
        # Pool each level separately
        coarse_pooled = self.pool_H(z_coarse)  # [batch, hidden]
        medium_pooled = self.pool_L(z_medium)
        fine_pooled = self.pool(z_fine)
        
        # Concatenate all levels (don't lose information)
        z_combined = torch.cat([coarse_pooled, medium_pooled, fine_pooled], dim=-1)
        
        # Project to LLM space
        return self.projection(z_combined)
```

### 5.4 ACT (Adaptive Computation Time) with Hierarchical Diffusion

**Enhancement**: Use hierarchy in halting decisions

```python
class HierarchicalACTHaltingHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Separate halt decisions per level
        self.q_head_coarse = nn.Linear(hidden_size, 2)
        self.q_head_medium = nn.Linear(hidden_size, 2)
        self.q_head_fine = nn.Linear(hidden_size, 2)
    
    def forward(self, z_coarse, z_medium, z_fine):
        # Halt coarse → medium → fine in cascade
        halt_coarse = self.q_head_coarse(z_coarse.mean(1))
        halt_medium = self.q_head_medium(z_medium.mean(1))
        halt_fine = self.q_head_fine(z_fine.mean(1))
        
        # Can halt early at any level
        return halt_coarse, halt_medium, halt_fine
```

---

## Part 6: Implementation Roadmap

### Phase 1: Separate H and L Processing (1-2 weeks)

**Files to Modify**:
- `/models/recursive_reasoning/trm.py` - Create separate H_level module
- `/config/arch/trm.yaml` - Add H_layers config
- `/models/recursive_reasoning/trm_singlez.py` - Update if applicable

**Testing**:
```bash
python pretrain.py arch=trm data_paths="[data/sudoku]" epochs=100 evaluators="[]"
```

**Expected Impact**: Allow H and L to learn different representations

### Phase 2: Position-Aware Conditioning (1-2 weeks)

**Files to Create**:
- `/models/recursive_reasoning/trm_position_aware.py` - New model variant

**Testing**: Compare against baseline on Sudoku-Extreme

### Phase 3: Hierarchical Diffusion Model (2-3 weeks)

**Files to Create**:
- `/models/recursive_reasoning/trm_hierarchical_diffusion.py` - Core model
- `/config/arch/trm_hierarchical_diffusion.yaml` - Configuration

**Utilities Needed**:
- Grid downsampling/upsampling functions
- Hierarchical carry management
- Multi-level loss computation

### Phase 4: Adapter Updates (1 week)

**Files to Create**:
- `/hybrid_pipeline/adapters/hierarchical_text_to_latent.py`
- `/hybrid_pipeline/adapters/hierarchical_latent_to_text.py`

### Phase 5: Integration Testing (1-2 weeks)

- Joint training with hierarchical diffusion
- Evaluate on ARC-AGI-1 and ARC-AGI-2
- Compare against flat TRM baseline

---

## Part 7: Technical Details & Considerations

### 7.1 Grid Downsampling/Upsampling

For ARC 30x30 grids:

```python
def downsample_grid(grid: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample grid by factor (mode: 'max' for color grids)"""
    # grid: [batch, 30, 30, channels]
    return F.max_pool2d(grid, kernel_size=factor, stride=factor)

def upsample_grid(grid: torch.Tensor, target_size: int) -> torch.Tensor:
    """Upsample grid using nearest neighbor or bilinear"""
    return F.interpolate(grid, size=(target_size, target_size), mode='nearest')

# For sequence format [batch, seq_len, hidden]:
def downsample_sequence(seq: torch.Tensor, grid_size: int, factor: int) -> torch.Tensor:
    # Reshape to grid, downsample, reshape back
    batch, seq_len, hidden = seq.shape
    grid = seq.view(batch, grid_size, grid_size, hidden)
    grid_down = downsample_grid(grid, factor)
    return grid_down.view(batch, -1, hidden)
```

### 7.2 Parameter Efficiency

**Concern**: Adding hierarchical modules increases parameters.

**Mitigations**:
- Use shared/tied weights across levels
- Keep hidden_size the same
- Use efficient upsampling (no learnable parameters)
- Shared position embeddings

**Expected**: +10-15% parameter increase for 3-level hierarchy

### 7.3 Training Stability

**Challenges**:
- Balancing coarse-to-fine refinement
- Gradients through upsampling
- Cascading halt decisions

**Solutions**:
- Careful initialization of upsampling
- Gradient clipping for each level
- Pre-training coarse levels separately

### 7.4 Evaluation Metrics

**Metrics to Track**:
- Time to convergence (hierarchical should be faster)
- Memory usage per level
- Accuracy at each abstraction level
- Halt distribution across levels

---

## Part 8: Experimental Validation Plan

### Experiment 1: Separate H/L Processing
- **Hypothesis**: Distinct H/L modules improve performance
- **Baseline**: Current TRM with shared L_level
- **Treatment**: TRM with separate H_level and L_level
- **Metrics**: Accuracy, training time, parameter count

### Experiment 2: Hierarchical Diffusion
- **Hypothesis**: Coarse-to-fine reasoning improves on hard problems
- **Baseline**: TRM (flat)
- **Treatment**: TRM with 3-level hierarchy
- **Metrics**: Accuracy on hard vs easy problems

### Experiment 3: Combined with Hybrid Pipeline
- **Hypothesis**: Hierarchical awareness in adapters improves joint training
- **Baseline**: Current hybrid pipeline
- **Treatment**: Hierarchical-aware adapters + TRM
- **Metrics**: Joint training accuracy, convergence speed

---

## Part 9: Alternative Approaches (Not Recommended but Worth Considering)

### 9.1 Diffusion as Noise Scheduling

Instead of hierarchical diffusion, use diffusion-style noise:
- Start with random grid, gradually sharpen predictions
- Use forward/reverse diffusion processes
- Requires significant architectural changes
- May not align with "kill abstraction" philosophy

**Status**: More complex, less aligned with goal. Skip for now.

### 9.2 Mixture of Experts per Level

Use different expert modules for each abstraction level:
```python
self.coarse_experts = nn.ModuleList([...])
self.medium_experts = nn.ModuleList([...])
self.fine_experts = nn.ModuleList([...])
```

**Status**: Increases parameters significantly. Only if needed.

### 9.3 Multi-Scale Attention

Attention at multiple scales (3x3, 6x6, 30x30 simultaneously):
- Requires positional embeddings for all scales
- Complex attention masking
- High memory cost

**Status**: Overcomplicated. Stick with sequential hierarchy.

---

## Summary Table: Current vs. Recommended Architecture

| Aspect | Current TRM | Recommended Changes |
|--------|-------------|-------------------|
| H/L Processing | Both use L_level | Separate H_level module |
| Position Awareness | Learned position embeddings | + Context-aware conditioning |
| Abstraction Levels | Flat (900 positions) | Hierarchical (9→36→900) |
| Information Flow | All positions together | Coarse→medium→fine refinement |
| Adapters | Flat pooling | Hierarchical pooling |
| ACT Halting | Single decision | Per-level decisions |
| Parameters | 7M | ~8-8.5M (+10-15%) |
| Memory | Baseline | Coarse level: 1/100x data size |
| Expected Improvement | Baseline | +2-5% on hard problems |

---

## Conclusion

The TRM architecture provides an excellent foundation for implementing "kill abstraction + hierarchical diffusion" because:

1. **No existing abstractions to remove** - TRM is already flat, making changes easier
2. **Separate H/L vectors ready** - Infrastructure exists for hierarchical processing
3. **Clean adapter interface** - Easy to plug in hierarchical awareness
4. **Proven recursive reasoning** - Foundation is solid

The recommended approach maintains TRM's simplicity while adding explicit hierarchy through:
- Separate H and L processing modules
- Position-aware conditioning
- Three-level hierarchical diffusion (coarse→medium→fine)
- Adaptive halt decisions per level

This should provide better performance on hard reasoning problems while maintaining parameter efficiency.

