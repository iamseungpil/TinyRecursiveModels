# Slot Attention의 Permutation Problem 분석

## 문제: Slot Assignment의 Arbitrary 특성

### 1️⃣ Slot Attention의 Permutation Invariance

```python
# Task A, Example 1
input_1 = [grid with 대칭 + 색상반전 rules]
slots_1 = slot_attention(input_1)
# → slots_1 = [slot_0: "대칭", slot_1: "색상반전", slot_2: unused, ...]

# Task A, Example 2 (같은 task, 다른 example)
input_2 = [another grid with same rules]
slots_2 = slot_attention(input_2)
# → slots_2 = [slot_0: "색상반전", slot_1: "대칭", slot_2: unused, ...]
#             ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^
#             순서가 바뀜! (Permutation invariant)
```

**왜 이런 일이 발생하는가?**
- Slot Attention은 **reconstruction loss**로 학습됨
- Reconstruction은 "어떤 slot이 어떤 정보를 담는가"에 무관 (permutation invariant)
- 따라서 매 iteration/example마다 slot assignment가 바뀔 수 있음

---

## 문제: 단순 Contrastive Loss는 실패

### ❌ 잘못된 접근
```python
# 같은 task의 slots끼리 contrastive
slots_1 = [s1_0, s1_1, s1_2]  # Task A, Example 1
slots_2 = [s2_0, s2_1, s2_2]  # Task A, Example 2

# 단순 index-wise contrastive
loss = 0
for i in range(num_slots):
    loss += -cosine_sim(slots_1[i], slots_2[i])  # ❌ 틀림!

# s1_0 = "대칭", s2_0 = "색상반전" → similarity 낮음!
# 같은 task인데도 loss가 높아짐
```

---

## 해결책 1: Hungarian Matching

### ✅ Optimal Assignment 찾기

```python
# Cost matrix: 모든 slot 쌍의 similarity
cost_matrix = torch.zeros(num_slots, num_slots)
for i in range(num_slots):
    for j in range(num_slots):
        cost_matrix[i, j] = -cosine_sim(slots_1[i], slots_2[j])  # Negative (minimize cost)

# Hungarian algorithm으로 optimal matching
from scipy.optimize import linear_sum_assignment
row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())

# Matched pairs만 contrastive loss
loss = 0
for i, j in zip(row_indices, col_indices):
    loss += -cosine_sim(slots_1[i], slots_2[j])
```

**장점:**
✅ Permutation-invariant contrastive learning 가능
✅ Optimal assignment 보장

**단점:**
❌ Computational overhead: O(n³) 복잡도
❌ Non-differentiable: Hungarian은 discrete optimization
❌ Training instability: 매 step마다 assignment 바뀔 수 있음

---

## 해결책 2: Sinkhorn Matching (Differentiable)

```python
def sinkhorn_matching(slots_1, slots_2, num_iters=10, temperature=0.1):
    """Differentiable soft matching using Sinkhorn-Knopp algorithm."""
    # Similarity matrix
    sim_matrix = torch.matmul(slots_1, slots_2.T)  # [num_slots, num_slots]
    sim_matrix = sim_matrix / temperature

    # Sinkhorn iterations
    log_P = F.log_softmax(sim_matrix, dim=-1)
    for _ in range(num_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)

    P = torch.exp(log_P)  # Soft assignment matrix

    # Soft contrastive loss
    loss = -(P * sim_matrix).sum()
    return loss, P
```

**장점:**
✅ Differentiable
✅ Faster than Hungarian

**단점:**
❌ Soft assignment (not exact matching)
❌ 여전히 slot consistency 문제 있음

---

## 근본적 문제: Semantic Consistency 부족

**문제:**
```python
# Iteration 1
slots = [s0: "대칭", s1: "색상반전", s2: "채우기"]

# Iteration 100
slots = [s0: "색상반전", s1: "채우기", s2: "대칭"]  # 완전히 바뀜!

# Iteration 1000
slots = [s0: "대칭+색상반전 혼합", s1: "??", s2: "채우기"]  # 붕괴
```

**왜 이런 일이?**
- Reconstruction + Contrastive만으로는 **"Slot i는 항상 특정 규칙을 담는다"** 보장 못함
- Slot들이 서로 역할을 바꿔도 loss 동일 (symmetry)

---

## 질문: Slot + Contrastive만으로 개선이 될까?

### 제 답변: ⚠️ **아마도 충분하지 않음**

**이유:**

1. **No explicit supervision on rules**
   - Pixel-level loss만 있음
   - 어떤 규칙이 있는지 알려주지 않음
   - 모델이 "규칙"을 자동으로 발견해야 함

2. **Slot의 의미가 불명확**
   ```python
   # 모델이 배울 수 있는 것들:
   # 옵션 A: slot_0 = "대칭", slot_1 = "색상반전"  (우리가 원하는 것)
   # 옵션 B: slot_0 = "왼쪽 절반", slot_1 = "오른쪽 절반"  (spatial decomposition)
   # 옵션 C: slot_0 = "빨간 픽셀", slot_1 = "파란 픽셀"  (color-based)
   # 옵션 D: 의미 없는 분해

   # Reconstruction loss는 모두 동일!
   ```

3. **Training dynamics 불안정**
   - Slot assignment가 계속 바뀜
   - Contrastive가 안정화시키려 하지만 약함

---

## 해결책: 추가 Regularization 필요

### Option A: Slot Diversity + Sparsity

```python
class SlotAttentionWithRegularization(nn.Module):
    def forward(self, z_H):
        slots = self.slot_attention(z_H)  # [B, num_slots, slot_dim]

        # 1. Diversity loss: 각 slot이 서로 다르게
        diversity_loss = 0
        for i in range(num_slots):
            for j in range(i+1, num_slots):
                diversity_loss += cosine_sim(slots[:, i], slots[:, j]).abs().mean()
        diversity_loss = -diversity_loss  # Minimize similarity

        # 2. Sparsity: 필요한 slot만 사용
        slot_importance = self.importance_head(slots)  # [B, num_slots]
        sparsity_loss = slot_importance.abs().sum(dim=-1).mean()

        # 3. Slot utilization: 모든 slot이 골고루 사용되게
        utilization_loss = -slot_importance.std(dim=0).mean()

        return slots, diversity_loss, sparsity_loss, utilization_loss
```

### Option B: Curriculum Learning

```python
# Phase 1: 단순 task (1개 rule) → num_active_slots = 1
# Phase 2: 중간 task (2 rules) → num_active_slots = 2
# Phase 3: 복잡 task (3+ rules) → num_active_slots = 3+

# 각 phase에서 점진적으로 slot이 sub-rule 학습
```

### Option C: Prototype-based Slots

```python
# Learnable rule prototypes
rule_prototypes = nn.Parameter(torch.randn(num_rules, slot_dim))
# → prototype_0: "대칭" template
# → prototype_1: "색상반전" template

# Slots는 prototypes의 조합으로 표현
slot_weights = attention(z_H, rule_prototypes)  # [B, num_slots, num_rules]
slots = torch.matmul(slot_weights, rule_prototypes)

# Contrastive on prototypes (not slots)
# → Semantic consistency 보장!
```

---

## 더 실용적 대안: TRM hier6 개선

현재 `trm_hier6.py`는 이미 6개 L-level states를 사용:

```python
# 현재: trm_hier6.py:235
z_L_ = z_L[0] + z_L[1] + z_L[2] + z_L[3] + z_L[4] + z_L[5]  # 단순 sum
```

**개선 아이디어:**

```python
# 1. Learned composition weights
composition_logits = self.composition_head(z_H)  # [B, 6]
composition_weights = F.softmax(composition_logits, dim=-1)

z_L_ = sum(composition_weights[:, i:i+1, None] * z_L[i] for i in range(6))

# 2. Sparsity regularization
sparsity_loss = composition_weights.abs().sum(dim=-1).mean()

# 3. Diversity: 각 z_L[i]가 서로 다른 패턴 학습
diversity_loss = -sum(cosine_sim(z_L[i], z_L[j]) for i<j)

# 4. Contrastive: 같은 task는 같은 composition weights
# (이건 실제로 효과 있을 가능성 높음!)
task_A_weights_1 = composition_weights_from_example_1  # [B, 6]
task_A_weights_2 = composition_weights_from_example_2  # [B, 6]

contrastive_loss = -cosine_sim(task_A_weights_1, task_A_weights_2)
```

**왜 이게 더 나은가?**
- ✅ 6개 states는 이미 존재 (architecture 변경 최소)
- ✅ Composition weights는 low-dim (6차원) → stable
- ✅ Contrastive on weights는 permutation 문제 없음
- ✅ Curriculum 적용 쉬움 (처음엔 1-2개만 활성화)

---

## 결론: Slot + Contrastive만으로는 부족

### ❌ 단순 Slot Attention + Contrastive
- Permutation 문제
- Semantic consistency 부족
- Training instability

### ✅ 추가 필요한 것들
1. **Regularization**: Diversity, Sparsity, Utilization
2. **Curriculum Learning**: 단순→복잡 task 순서
3. **Prototype/Anchor**: Semantic consistency 보장
4. 또는 **TRM hier6 개선** (더 실용적)

### 추천 순서
1. **먼저 TRM hier6 개선** 시도
   - Learned composition weights
   - Sparsity + Diversity loss
   - Contrastive on composition weights
2. 효과 없으면 **Slot Attention + Prototypes**
3. 근본적 해결은 **Neuro-symbolic** (장기)

---

어떤 방향으로 진행하시겠습니까?
1. TRM hier6 개선 구현
2. Slot Attention + Hungarian/Sinkhorn 구현
3. 다른 접근?
