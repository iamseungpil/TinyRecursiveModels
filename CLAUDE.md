# TinyRecursiveModels - Claude Code Memory

## TRM-Titans v6 Architecture Documentation

### 1. 목표 (Goals)

TRM-Titans는 **TRM (Tiny Recursive Model)의 계층적 추론 구조**와 **Titans의 Neural Memory**를 결합하여 ARC (Abstraction and Reasoning Corpus) 퍼즐 해결을 위한 test-time learning 능력을 갖춘 모델입니다.

**핵심 목표:**
1. **Test-Time Adaptation**: 새로운 퍼즐에 대해 demo examples로 빠르게 적응
2. **Hierarchical Reasoning**: H-level (추상화) ↔ L-level (구체화) 양방향 추론
3. **Pattern Memory**: Surprise 기반 학습으로 input→output 패턴 저장
4. **Plug-and-Play Integration**: MAG/MAC/MAL 전략 간 쉬운 전환

---

### 2. 모델 구현 (Implementation)

#### 2.1 Block-Level Memory MLP 구조

각 `TRM_Titans_Block`에는 `TitansAttention`이 있고, 그 안에 `TitansMemory` (MLP)가 있습니다:

```
TRM_Titans_Block
├── self_attn: TitansAttention
│   ├── memory: TitansMemory (MLP)
│   │   ├── template_up: Linear(D → H)     # 초기 상태 (frozen)
│   │   ├── template_down: Linear(H → D)   # 초기 상태 (frozen)
│   │   ├── _current_up_weight: [B, H, D]  # 런타임 상태 (surprise 업데이트)
│   │   └── _current_down_weight: [B, D, H]
│   ├── qkv_proj: Linear
│   └── o_proj: Linear
└── mlp: SwiGLU
```

**Memory Update 수식 (Titans):**
```
M_t = (1 - α) * M_{t-1} - η * ∇(||M(k) - v||²)

where:
  M: Memory weights
  α: decay rate (mem_decay)
  η: learning rate (mem_lr)
  k: input (hidden_states)
  v: target (attention output)
```

#### 2.2 Integration Strategies (MAG/MAC/MAL)

##### MAG (Memory As Gate) - Default
```
     x (input)
        │
   ┌────┴────┐
   ▼         ▼
Memory(x)  Attn(x)    ← 병렬 계산
   │         │
   └────┬────┘
        ▼
surprise = ||M(x) - A(x)||²
conf = exp(-surprise * τ)
output = conf * M(x) + (1-conf) * A(x)
```
- **학습**: Memory가 Attention 출력을 예측하도록 학습
- **추론**: confidence 높으면 Memory 사용 (빠름), 낮으면 Attention 사용 (정확)

##### MAC (Memory As Context)
```
     x (input)
        │
        ▼
    Memory(x) = context
        │
        ▼
Attention(Q=x, K=[x, context], V=[x, context])
        │
        ▼
     output
```
- **특징**: Memory context는 "timeless" - RoPE 적용 안 함
- **학습**: Memory가 유용한 context를 제공하도록 학습

##### MAL (Memory As Layer)
```
memory_first:  x → Memory(x) → Attention(M(x)) → output
attention_first: x → Attention(x) → Memory(A(x)) → output
```
- **특징**: 순차적 처리
- **학습**: 각 단계가 다음 단계에 유용한 representation 제공

#### 2.3 Config 설정
```yaml
# config/arch/trm_titans.yaml
integration_type: mag    # mag, mac, mal
mal_order: memory_first  # MAL 전용: memory_first, attention_first
```

---

### 3. 상태 관리 (State Management)

#### 현재 v6 구현 (z_H + z_L + Memory)
```
┌─────────────────────────────────────────────────────────┐
│  Explicit State (Carry)                                  │
│  ├── z_H: [B, L, D] - High-level state tensor           │
│  └── z_L: [B, L, D] - Low-level state tensor            │
│                                                          │
│  Implicit State (TitansMemory per block)                 │
│  ├── _current_up_weight: [B, H, D]                      │
│  └── _current_down_weight: [B, D, H]                    │
└─────────────────────────────────────────────────────────┘
```

#### Forward Loop (v6)
```python
for H_step in range(H_cycles - 1):
    for L_step in range(L_cycles):
        z_L = L_level(z_L, z_H + input)  # Attn + MLP with TitansMemory
    z_H = L_level(z_H, z_L)              # Attn + MLP with TitansMemory

output = lm_head(z_H)
```

#### 순수 Titans 설계 (대안)
만약 z_H를 제거하고 Memory weights만 H-level 역할을 하게 하면:
```python
for H_step in range(H_cycles):
    for L_step in range(L_cycles):
        l_state = L_level(l_state, input)  # Memory weights가 H-level 대체
        # Memory.update() 호출 → weights에 패턴 저장

output = lm_head(l_state)
```
이 설계에서는 Memory weights가 "slow knowledge"를 저장하여 z_H 역할을 대신함.

---

### 4. TRM vs Titans vs TRM-Titans 비교

| 측면 | TRM (원본) | Titans | TRM-Titans v6 |
|------|-----------|--------|---------------|
| **H-level 상태** | z_H tensor | Memory weights | z_H tensor + Memory weights |
| **L-level 상태** | z_L tensor | Activations | z_L tensor |
| **H↔L 상호작용** | z_H ↔ z_L via L_level | Memory ↔ Attention | z_H ↔ z_L + Memory ↔ Attention |
| **Memory 학습** | N/A | Surprise (forward) | Surprise (forward) |
| **Attention 학습** | Backprop | Backprop | Backprop |
| **MLP 학습** | Backprop | Backprop | Backprop |
| **Test-Time Learning** | No | Yes (memory) | Yes (memory + puzzle_emb) |
| **Integration** | N/A | MAG/MAC/MAL | MAG/MAC/MAL |

### 4.1 TRM과의 차이점
1. **TitansMemory 추가**: 각 block에 Memory MLP 추가
2. **Surprise 학습**: Memory는 backprop 없이 forward pass에서 학습
3. **Integration Strategy**: MAG/MAC/MAL 선택 가능
4. **Test-Time Adaptation**: TTT 지원

### 4.2 Titans와의 차이점
1. **z_H/z_L 유지**: 원본 TRM의 양방향 구조 유지
2. **H_cycles/L_cycles**: 계층적 반복 구조 유지
3. **ACT (Adaptive Computation Time)**: 동적 halting 지원
4. **Puzzle Embedding**: Task-specific embedding 학습

---

### 5. 학습 파이프라인

#### 5.1 Pretraining (`pretrain.py`)
```python
# Optimizer 분리
optimizers = [
    CastedSparseEmbeddingSignSGD(puzzle_emb),  # Puzzle embedding
    AdamW(model.parameters())                   # 나머지
]

# Training loop
for batch in dataloader:
    carry, loss, metrics = model(carry, batch, update_memory=True)
    loss.backward()  # Memory는 forward에서 이미 업데이트됨
    optimizer.step()
```

**학습 대상:**
| 컴포넌트 | 학습 방식 |
|----------|----------|
| Puzzle Embedding | SignSGD (sparse) |
| Attention (Q,K,V,O) | AdamW (backprop) |
| MLP (SwiGLU) | AdamW (backprop) |
| Memory template | Frozen |
| Memory current weights | Surprise (forward) |
| mem_lr, mem_decay | Frozen |

#### 5.2 Test-Time Training (`evaluate_ttt.py`)
```python
ttt = TRM_Titans_TestTime(model)

for puzzle in puzzles:
    # 1. Memory reset
    ttt.reset_memory()

    # 2. Demo pairs로 적응
    ttt.test_time_adapt(demo_pairs, n_steps=10, lr=0.01)

    # 3. Test 예측
    predictions = ttt.predict(test_input)
```

---

### 6. 파일 구조

```
TinyRecursiveModels/
├── models/recursive_reasoning/
│   ├── trm_titans.py      # TRM-Titans v6 모델 (메인)
│   ├── trm.py             # 원본 TRM
│   └── hrm.py             # HRM (비교용)
├── pretrain.py            # 학습 스크립트
├── evaluate_ttt.py        # TTT 평가 스크립트
├── evaluators/arc.py      # ARC 평가기
└── config/
    ├── arch/trm_titans.yaml     # 아키텍처 설정
    └── cfg_trm_titans_fast.yaml # 빠른 학습 설정
```

---

### 7. 주요 API

#### 7.1 Model
```python
# 생성
model = TRM_Titans(config)
loss_head = TRM_Titans_ACTLossHead(model, "stablemax_cross_entropy")

# Forward
carry = model.initial_carry(batch)
carry, loss, metrics, outputs, all_halted = loss_head(carry, batch)
```

#### 7.2 Test-Time Training
```python
ttt = TRM_Titans_TestTime(model)
ttt.test_time_adapt(demo_pairs, n_steps=10, lr=0.01)
predictions = ttt.predict(test_input)
```

#### 7.3 Memory Control
```python
model.reset_all_memory(batch_size, device)       # 전체 reset
model.freeze_memory_templates()                   # Template freeze
```

---

### 8. 실행 명령어

#### Pretraining
```bash
torchrun --nproc_per_node=8 pretrain.py --config-name=cfg_trm_titans_fast
```

#### TTT Evaluation
```bash
python evaluate_ttt.py \
    --checkpoint /path/to/checkpoint \
    --data_path data/arc-aug-1000 \
    --ttt_steps 10 \
    --ttt_lr 0.01 \
    --verbose
```

---

### 9. 참고 논문

1. **Titans**: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
   - Neural Memory with surprise-based learning
   - MAG/MAC/MAL integration strategies

2. **TRM**: Original Tiny Recursive Model
   - z_H ↔ z_L bidirectional structure
   - H_cycles / L_cycles hierarchical reasoning

3. **ACT**: Adaptive Computation Time
   - Dynamic halting based on q_halt_logits
