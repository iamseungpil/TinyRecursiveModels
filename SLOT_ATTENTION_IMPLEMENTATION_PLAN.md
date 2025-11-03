# Slot Attention + Contrastive Learning 구현 계획

## 📁 디렉토리 구조

```
experiments/slot_attention/
├── README.md                           # 실험 설명
├── train_slot_attention.py             # 학습 스크립트 (pretrain.py 기반)
├── eval_slot_attention.py              # 평가 스크립트
└── configs/
    └── trm_slots.yaml                  # 설정 파일

models/
├── slot_attention.py                   # Slot Attention 모듈 (새로 작성)
├── recursive_reasoning/
│   └── trm_with_slots.py              # TRM + Slot Attention (새로 작성)
└── losses.py                           # SlotContrastiveLossHead 추가
```

---

## 🎯 구현 목표

**핵심 아이디어:**
1. TRM의 z_H representation을 Slot Attention으로 분해
2. 같은 task의 다른 examples끼리 slots를 contrastive learning
3. Hungarian matching으로 permutation-invariant 처리

---

## 📐 Architecture 설계

### 1. Slot Attention 모듈 (`models/slot_attention.py`)

```python
class SlotAttention(nn.Module):
    """
    Slot Attention module for decomposing representations.

    Paper: "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
    """

    def __init__(
        self,
        num_slots: int,          # 8 (slot 개수)
        slot_dim: int,           # 256 (slot dimension)
        input_dim: int,          # 512 (TRM hidden_size)
        num_iterations: int = 3, # Iterative refinement
        mlp_hidden_dim: int = 512
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # Slot initialization (learnable)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Attention
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(input_dim, slot_dim)
        self.project_v = nn.Linear(input_dim, slot_dim)

        # Slot update MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_inputs, input_dim]  # z_H: [B, 916, 512]

        Returns:
            slots: [batch, num_slots, slot_dim]     # [B, 8, 256]
        """
        B, N, D = inputs.shape

        # Initialize slots
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # [B, num_slots, slot_dim]

        # Normalize inputs
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Compute attention weights
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k)  # [B, num_slots, N]
            attn_logits = attn_logits / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots

            # Weighted mean
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum('bsn,bnd->bsd', attn_norm, v)  # [B, num_slots, slot_dim]

            # Update slots
            slots = slots_prev + self.mlp(updates)

        return slots  # [B, num_slots, slot_dim]
```

---

### 2. TRM + Slot Attention (`models/recursive_reasoning/trm_with_slots.py`)

```python
class TRM_WithSlots_Inner(nn.Module):
    """TRM with Slot Attention decomposition."""

    def __init__(self, config):
        super().__init__()
        # ... 기존 TRM 초기화 ...

        # Slot Attention module
        self.slot_attention = SlotAttention(
            num_slots=config.num_slots,        # 8
            slot_dim=config.slot_dim,          # 256
            input_dim=config.hidden_size,      # 512
            num_iterations=config.slot_iterations  # 3
        )

        # Slot decoder (slots → pixels for reconstruction)
        self.slot_decoder = nn.Sequential(
            nn.Linear(config.slot_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Dual prediction heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)  # 기존
        self.slot_lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)  # 새로

    def forward(self, carry, batch):
        # ... 기존 TRM forward (z_H 생성) ...

        # Slot decomposition
        slots = self.slot_attention(z_H)  # [B, num_slots, slot_dim]

        # Reconstruct from slots
        slot_features = self.slot_decoder(slots)  # [B, num_slots, hidden_size]
        slot_features_pooled = slot_features.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # Dual outputs
        output_direct = self.lm_head(z_H)[:, self.puzzle_emb_len:]     # 기존 방식
        output_slots = self.slot_lm_head(slot_features_pooled)         # Slot 기반

        return new_carry, output_direct, output_slots, slots, (q_halt_logits, q_continue_logits)
```

**핵심:**
- z_H [B, 916, 512]를 slots [B, 8, 256]로 분해
- 2개 prediction heads: 기존 + slot 기반

---

### 3. Contrastive Loss with Hungarian Matching (`models/losses.py`)

```python
class SlotContrastiveLossHead(nn.Module):
    """
    Loss head with:
    1. Pixel-level reconstruction loss (기존)
    2. Slot reconstruction loss (새로)
    3. Slot contrastive loss with Hungarian matching (새로)
    """

    def __init__(self, model, loss_type: str,
                 slot_recon_weight: float = 0.5,
                 slot_contrastive_weight: float = 0.1):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.slot_recon_weight = slot_recon_weight
        self.slot_contrastive_weight = slot_contrastive_weight

    def hungarian_matching(self, slots_1, slots_2):
        """
        Find optimal slot assignment using Hungarian algorithm.

        Args:
            slots_1: [B, num_slots, slot_dim]
            slots_2: [B, num_slots, slot_dim]

        Returns:
            matched_indices: List of (row_idx, col_idx) for each batch
        """
        from scipy.optimize import linear_sum_assignment

        B, num_slots, D = slots_1.shape
        matched_indices = []

        for b in range(B):
            # Cost matrix: negative cosine similarity
            cost_matrix = -F.cosine_similarity(
                slots_1[b:b+1, :, None, :],  # [1, num_slots, 1, D]
                slots_2[b:b+1, None, :, :],  # [1, 1, num_slots, D]
                dim=-1
            ).squeeze(0)  # [num_slots, num_slots]

            # Hungarian algorithm
            row_idx, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            matched_indices.append((row_idx, col_idx))

        return matched_indices

    def compute_slot_contrastive_loss(self, carry, slots):
        """
        Compute contrastive loss between slots of same task.

        Strategy:
        - Group examples by puzzle_id in the batch
        - For same puzzle_id, compute pairwise slot similarity with Hungarian matching
        """
        puzzle_ids = carry.current_data["puzzle_identifiers"]  # [B]
        B, num_slots, D = slots.shape

        total_loss = 0
        count = 0

        # Find pairs of same puzzle_id in batch
        unique_ids = torch.unique(puzzle_ids)

        for puzzle_id in unique_ids:
            mask = (puzzle_ids == puzzle_id)
            indices = torch.where(mask)[0]

            if len(indices) < 2:
                continue  # Need at least 2 examples

            # Pairwise contrastive
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]

                    slots_1 = slots[idx1:idx1+1]  # [1, num_slots, D]
                    slots_2 = slots[idx2:idx2+1]  # [1, num_slots, D]

                    # Hungarian matching
                    matched = self.hungarian_matching(slots_1, slots_2)[0]
                    row_idx, col_idx = matched

                    # Matched slots should be similar
                    for r, c in zip(row_idx, col_idx):
                        sim = F.cosine_similarity(
                            slots_1[0, r], slots_2[0, c], dim=0
                        )
                        total_loss += -sim  # Maximize similarity
                        count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def forward(self, return_keys, **model_kwargs):
        # Model forward
        new_carry, output_direct, output_slots, slots, (q_halt_logits, q_continue_logits) = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # 1. Direct LM loss (기존)
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        lm_loss_direct = (self.loss_fn(output_direct, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        # 2. Slot reconstruction loss
        lm_loss_slots = (self.loss_fn(output_slots, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        # 3. Slot contrastive loss
        slot_contrastive_loss = self.compute_slot_contrastive_loss(new_carry, slots)

        # 4. Q-learning losses (기존)
        with torch.no_grad():
            is_correct = mask & (torch.argmax(output_direct, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum"
        )

        # Total loss
        total_loss = (
            lm_loss_direct +
            self.slot_recon_weight * lm_loss_slots +
            self.slot_contrastive_weight * slot_contrastive_loss +
            0.5 * q_halt_loss
        )

        # Metrics
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": valid_metrics.sum(),
            "lm_loss_direct": lm_loss_direct.detach(),
            "lm_loss_slots": lm_loss_slots.detach(),
            "slot_contrastive_loss": slot_contrastive_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }

        outputs = {
            "logits": output_direct,
            "slots": slots,
            "q_halt_logits": q_halt_logits,
        }

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
```

**핵심:**
- Hungarian matching으로 optimal slot assignment
- 같은 puzzle_id끼리만 contrastive
- 3가지 loss 결합: direct + slot recon + contrastive

---

## 🔧 Config 설정 (`experiments/slot_attention/configs/trm_slots.yaml`)

```yaml
arch:
  name: models.recursive_reasoning.trm_with_slots@TRM_WithSlots_ACTV1

  # TRM config (기존)
  H_cycles: 16
  L_cycles: 6
  H_layers: 0
  L_layers: 2
  hidden_size: 512
  expansion: 2.0
  num_heads: 8
  pos_encodings: rope
  puzzle_emb_ndim: 8192
  puzzle_emb_len: 16
  halt_max_steps: 16
  halt_exploration_prob: 0.2
  mlp_t: true
  no_ACT_continue: true
  forward_dtype: bfloat16

  # Slot Attention config (새로)
  num_slots: 8
  slot_dim: 256
  slot_iterations: 3

  # Loss config
  loss:
    name: models.losses@SlotContrastiveLossHead
    loss_type: softmax_cross_entropy
    slot_recon_weight: 0.5
    slot_contrastive_weight: 0.1

# Training config (기존과 동일)
data_paths:
  - data/arc-agi/arc_dataset_train

global_batch_size: 64
epochs: 200
lr: 2e-4
lr_min_ratio: 0.1
lr_warmup_steps: 1000
weight_decay: 0.01
beta1: 0.9
beta2: 0.98

puzzle_emb_lr: 1e-3
puzzle_emb_weight_decay: 0.01

eval_interval: 10
checkpoint_every_eval: true

evaluators:
  - name: evaluators.arc@ARC
```

---

## 🚀 Training Script (`experiments/slot_attention/train_slot_attention.py`)

```python
#!/usr/bin/env python3
"""
Training script for TRM with Slot Attention + Contrastive Learning.

Usage:
    cd experiments/slot_attention
    python train_slot_attention.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import hydra
from omegaconf import DictConfig

# Import pretrain logic
from pretrain import launch

@hydra.main(config_path="configs", config_name="trm_slots", version_base=None)
def main(config: DictConfig):
    """Launch training with slot attention config."""
    launch(config)

if __name__ == "__main__":
    main()
```

**핵심:**
- pretrain.py의 launch() 재사용
- 별도 config만 사용

---

## 📊 학습 흐름

```
1. Batch 준비
   ├─ inputs: [B, 900]
   ├─ labels: [B, 900]
   └─ puzzle_identifiers: [B]  # 같은 puzzle_id 여러 개 포함 가능

2. TRM Forward
   ├─ z_H: [B, 916, 512]  ← TRM reasoning
   └─ Slot Attention
       └─ slots: [B, 8, 256]  ← Decomposition

3. Dual Prediction
   ├─ output_direct: [B, 900, 12]  ← z_H에서 직접
   └─ output_slots: [B, 900, 12]   ← slots에서 복원

4. Loss 계산
   ├─ lm_loss_direct: Direct prediction
   ├─ lm_loss_slots: Slot reconstruction
   ├─ slot_contrastive_loss:
   │   └─ 같은 puzzle_id끼리 Hungarian matching
   │       └─ Matched slots → cosine similarity 최대화
   └─ q_halt_loss: Halting loss
```

---

## 🎯 기대 효과

### 1. Compositional Structure 학습
```
Slot 0: "대각선 대칭" 규칙
Slot 1: "색상 반전" 규칙
Slot 2: "빈 공간 채우기" 규칙
Slot 3-7: Unused or finer details
```

### 2. Task Similarity Encoding
```
같은 task의 examples → Similar slot assignments
다른 task → Different slot patterns
```

### 3. Interpretability
```
Slot별로 어떤 규칙을 담당하는지 분석 가능
```

---

## 🔍 평가 방법

### 1. Quantitative
- ARC accuracy (기존 평가)
- Slot contrastive loss 수렴
- Slot reconstruction accuracy

### 2. Qualitative
```python
# Slot visualization
def visualize_slots(model, puzzle_examples):
    slots = []
    for example in puzzle_examples:
        z_H = model.forward(example)
        slot = model.slot_attention(z_H)
        slots.append(slot)

    # Compute slot similarity matrix
    similarity = cosine_similarity(slots)
    # → 같은 task면 similar slots 기대
```

---

## ⚠️ 잠재적 문제와 해결책

### 1. Slot Collapse
**문제:** 모든 slots가 같은 representation 학습

**해결책:**
```python
# Diversity regularization
diversity_loss = -torch.pdist(slots.mean(dim=0)).mean()
```

### 2. Hungarian Matching Overhead
**문제:** O(n³) 복잡도

**해결책:**
- 배치 내 같은 puzzle 적을 때는 괜찮음
- 필요시 Sinkhorn (differentiable) 대체

### 3. Slot 수 선택
**문제:** num_slots=8이 적절한가?

**해결책:**
- Curriculum: 처음엔 2-3개, 점진적 증가
- Sparsity: 필요한 slot만 활성화

---

## 📝 구현 순서

1. ✅ **계획 설명** (현재)
2. 승인 후:
   - [ ] `models/slot_attention.py` 작성
   - [ ] `models/recursive_reasoning/trm_with_slots.py` 작성
   - [ ] `models/losses.py`에 SlotContrastiveLossHead 추가
   - [ ] `experiments/slot_attention/` 디렉토리 생성
   - [ ] Config 파일 작성
   - [ ] Training script 작성
   - [ ] README 작성
3. 테스트:
   - [ ] 단위 테스트 (slot attention 단독)
   - [ ] 통합 테스트 (전체 pipeline)
   - [ ] Small-scale 학습 (10 epochs)
4. Full training

---

## 질문 사항

1. **num_slots = 8** 적절한가요? (조정 가능)
2. **slot_contrastive_weight = 0.1** 적절한가요?
3. **Hungarian vs Sinkhorn** 어떤 게 좋을까요?
4. 다른 regularization (diversity, sparsity) 추가할까요?

---

이 계획으로 진행해도 될까요?
