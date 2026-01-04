# TinyRecursiveModels - Claude Code Memory

## TRM-Titans v7 Architecture Documentation

### 1. 목표 (Goals)

TRM-Titans는 **TRM (Tiny Recursive Model)의 계층적 추론 구조**와 **Titans의 Neural Memory**를 결합하여 ARC (Abstraction and Reasoning Corpus) 퍼즐 해결을 위한 test-time learning 능력을 갖춘 모델입니다.

**핵심 목표:**
1. **Test-Time Adaptation**: 새로운 퍼즐에 대해 demo examples로 빠르게 적응
2. **Simplified State**: z_L만 관리, Memory weights가 H-level 역할 대체
3. **Pattern Memory**: Surprise 기반 학습으로 input→output 패턴 저장 (H_cycle마다)
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

#### v7 구현 (z_L만 + Memory weights)
```
┌─────────────────────────────────────────────────────────┐
│  Explicit State (Carry)                                  │
│  └── z_L: [B, L, D] - 유일한 state tensor               │
│                                                          │
│  Implicit State (TitansMemory per block) - H-level 역할  │
│  ├── _current_up_weight: [B, H, D]                      │
│  └── _current_down_weight: [B, D, H]                    │
└─────────────────────────────────────────────────────────┘
```

#### Forward Loop (v7 Option 2A)
```python
z_L = carry.z_L
zero_injection = torch.zeros_like(input_embeddings)  # H_step용

for H_step in range(H_cycles):
    z_L_start = z_L.clone()  # Cycle 시작 상태 저장

    # L_cycles: Attention만 사용 (Memory 미사용)
    for L_step in range(L_cycles):
        z_L = L_level(z_L, input_embeddings, use_memory=False)

    # H_step: Attention + Memory 통합 (Memory 업데이트 없음)
    z_L = L_level(z_L, zero_injection, use_memory=True, update_memory=False)

    # Cycle-level Memory 업데이트: K=z_L_start, V=z_L
    if update_memory:
        L_level.update_all_memory(k=z_L_start, v=z_L)

output = lm_head(z_L)  # z_L에서 직접 출력
```

#### Option 2A 설계 핵심
- **L_step**: Attention만 사용 (Memory 미사용) - 빠른 처리
- **H_step**: Attention + Memory 통합 - 느린 처리
- **Memory 업데이트**: Cycle-level 변환 학습 (z_L_start → z_L_end)
- **z_L**: Fast state - 매 L_step마다 업데이트
- **Memory weights**: Slow knowledge - H_cycle마다 1회 업데이트 (z_H 대체)
- z_H tensor 없음 → Memory weights가 implicit H-level state 역할

---

### 4. TRM vs Titans vs TRM-Titans v7 비교

| 측면 | TRM (원본) | Titans | TRM-Titans v7 |
|------|-----------|--------|---------------|
| **H-level 상태** | z_H tensor | Memory weights | Memory weights (z_H 제거) |
| **L-level 상태** | z_L tensor | Activations | z_L tensor |
| **H↔L 상호작용** | z_H ↔ z_L via L_level | Memory ↔ Attention | Memory ↔ z_L (H_cycle마다) |
| **Memory 학습** | N/A | Surprise (매 step) | Surprise (H_cycle 끝) |
| **Attention 학습** | Backprop | Backprop | Backprop (final cycle) |
| **MLP 학습** | Backprop | Backprop | Backprop (final cycle) |
| **Test-Time Learning** | No | Yes (memory) | Yes (memory + puzzle_emb) |
| **Integration** | N/A | MAG/MAC/MAL | MAG/MAC/MAL |

### 4.1 TRM과의 차이점
1. **z_H 제거**: Memory weights가 H-level implicit state 역할
2. **TitansMemory 추가**: 각 block에 Memory MLP 추가
3. **Surprise 학습**: Memory는 H_cycle 끝에서 학습
4. **Integration Strategy**: MAG/MAC/MAL 선택 가능
5. **Test-Time Adaptation**: TTT 지원

### 4.2 Titans와의 차이점
1. **Memory 업데이트 타이밍**: H_cycle 끝에만 (매 step X)
2. **H_cycles/L_cycles**: 계층적 반복 구조 유지
3. **ACT (Adaptive Computation Time)**: 동적 halting 지원
4. **Puzzle Embedding**: Task-specific embedding 학습

---

### 5. 학습/추론 파이프라인 (4가지 코드 경로)

TRM-Titans는 **4가지 코드 경로**를 지원합니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRM-Titans Code Paths                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Pretraining (pretrain.py)                                               │
│     └── 대규모 데이터셋으로 모델 사전학습                                      │
│         • Dual optimizer (SignSGD + AdamW)                                  │
│         • Memory surprise 업데이트 (forward)                                 │
│         • Attention/MLP backprop 학습                                       │
│                                                                             │
│  2. Standard Evaluation (evaluators/arc.py)                                 │
│     └── 학습 중 주기적 평가 (pretrain.py에서 호출)                            │
│         • 모든 augmentation에 대해 예측                                      │
│         • Voting으로 최종 예측 선택                                          │
│         • pass@k 정확도 계산                                                │
│                                                                             │
│  3. Test-Time Training (TRM_Titans_TestTime in trm_titans.py)               │
│     └── 퍼즐별 적응 (demo examples로 memory 업데이트)                        │
│         • Memory만 업데이트 (model weights frozen)                          │
│         • Surprise 기반 학습                                                │
│         • Accumulate 또는 Reset 옵션                                        │
│                                                                             │
│  4. TTT Evaluation (evaluate_ttt.py)                                        │
│     └── TTT 성능 독립 평가 스크립트                                          │
│         • 체크포인트 로드 → TTT 적응 → 예측 → pass@k 계산                    │
│         • submission.json 생성 (Kaggle 제출용)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### 5.1 Pretraining (`pretrain.py`)

대규모 ARC 데이터셋으로 모델을 사전학습합니다.

**핵심 구조:**
```python
# 1. Dual Optimizer 설정
optimizers = [
    CastedSparseEmbeddingSignSGD_Distributed(  # Puzzle embedding
        model.puzzle_emb.buffers(),
        lr=puzzle_emb_lr,  # 1e-2 (높은 lr)
        weight_decay=0.1
    ),
    AdamW(                                      # 나머지 모델 파라미터
        model.parameters(),
        lr=lr,  # 1e-4 (일반 lr)
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
]

# 2. Training Loop
carry = None  # Carry는 배치 간 persist (Memory 상태 유지)
for epoch in epochs:
    for batch in train_loader:
        # Init carry if None
        if carry is None:
            carry = model.initial_carry(batch)

        # Forward (Memory surprise 업데이트 포함)
        carry, loss, metrics, _, _ = model(carry, batch)

        # Backward (Attention, MLP만 학습)
        loss.backward()

        # Optimizer step
        for optim in optimizers:
            optim.step()
            optim.zero_grad()
```

**학습 대상 정리:**
| 컴포넌트 | 학습 방식 | Optimizer | 비고 |
|----------|----------|-----------|------|
| Puzzle Embedding | Gradient | SignSGD (sparse) | lr=1e-2, task별 embedding |
| Attention (Q,K,V,O) | Backprop | AdamW | lr=1e-4 |
| MLP (SwiGLU) | Backprop | AdamW | lr=1e-4 |
| Memory template | **Frozen** | - | 초기화 후 변경 없음 |
| Memory current weights | **Surprise** | - | Forward에서 업데이트 |
| mem_lr, mem_decay | **Frozen** | - | 고정 하이퍼파라미터 |

**Carry 관리 (v7):**
- `carry`는 `z_L` 텐서만 포함 (z_H 제거됨)
- 배치 간 **persist** (epoch 전체에서 유지)
- Memory weights (`_current_up/down_weight`)는 모델 내부에 저장 → H-level 역할

---

#### 5.2 Standard Evaluation (`evaluators/arc.py`)

`pretrain.py`에서 주기적으로 호출되는 평가 로직입니다.

**평가 흐름:**
```python
# 1. Evaluator 초기화
evaluator = ARC(data_path, eval_metadata, pass_Ks=(1, 2, 5, 10, 100, 1000))

# 2. 각 배치 처리
for batch in eval_loader:
    with torch.no_grad():
        carry = model.initial_carry(batch)  # 각 배치마다 새 carry

        # ACT loop (all_finish될 때까지)
        while True:
            carry, loss, metrics, preds, all_finish = model(carry, batch)
            if all_finish:
                break

        # 예측 수집
        evaluator.update_batch(batch, preds)

# 3. 결과 집계 (voting)
results = evaluator.result(save_path, rank, world_size, group)
```

**Voting 메커니즘:**
```python
# 여러 augmentation의 예측을 voting으로 집계
for input_hash, predictions in all_predictions.items():
    # q_halt_logits sigmoid를 confidence로 사용
    p_map = {}
    for pred_hash, q_value in predictions:
        p_map[pred_hash] = (count, avg_confidence)

    # Confidence 순으로 정렬
    sorted_preds = sorted(p_map.items(), key=confidence, reverse=True)

    # pass@k 계산
    for k in pass_Ks:
        if label_hash in top_k_preds:
            correct[k] += 1
```

**출력:**
- `ARC/pass@1`, `ARC/pass@2`, ... `ARC/pass@1000`
- `submission.json` (Kaggle 제출용)

---

#### 5.3 Test-Time Training (`TRM_Titans_TestTime`)

퍼즐별로 demo examples로 적응하는 TTT 래퍼 클래스입니다.

**클래스 구조 (`trm_titans.py`):**
```python
class TRM_Titans_TestTime:
    def __init__(self, model: TRM_Titans, device: torch.device):
        self.model = model
        self.device = device

    def reset_all_memory(self, batch_size: int = 1):
        """모든 Memory를 template 상태로 리셋"""
        self.model.reset_all_memory(batch_size, self.device)

    def test_time_adapt(
        self,
        demo_pairs: List[Tuple[Tensor, Tensor]],  # (input, label) pairs
        n_steps: int = 10,
        lr: float = 0.01,
        puzzle_id: int = 0,
        accumulate_memory: bool = True,
        verbose: bool = False
    ):
        """Demo examples로 Memory 적응"""
        # 1. Memory 리셋 (accumulate=False일 때)
        if not accumulate_memory:
            self.reset_all_memory()

        # 2. 각 demo pair로 Memory 업데이트
        for step in range(n_steps):
            for inp, label in demo_pairs:
                # Forward (surprise 계산 및 Memory 업데이트)
                carry = self.model.initial_carry(batch)
                carry, loss, metrics, preds, _ = self.model(
                    carry, batch, update_memory=True
                )

    def predict(
        self,
        test_input: Tensor,
        update_during_prediction: bool = False,
        puzzle_id: int = 0,
        reset_memory: bool = False
    ) -> Tensor:
        """적응된 Memory로 예측"""
        if reset_memory:
            self.reset_all_memory()

        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            carry, _, _, preds, _ = self.model(
                carry, batch, update_memory=update_during_prediction
            )

        return preds["preds"]
```

**Memory Accumulation 옵션:**
| 옵션 | 동작 | 사용 상황 |
|------|------|----------|
| `accumulate_memory=True` | Demo 간 Memory 누적 | 관련 패턴이 많을 때 |
| `accumulate_memory=False` | 각 demo마다 리셋 | 독립적인 패턴일 때 |

---

#### 5.4 TTT Evaluation (`evaluate_ttt.py`)

TTT 성능을 독립적으로 평가하는 스크립트입니다.

**실행 흐름:**
```python
# 1. 모델 및 체크포인트 로드
model = TRM_Titans(config)
load_checkpoint(model, checkpoint_path)
model.freeze_memory_templates()  # Template freeze

# 2. TTT 래퍼 생성
ttt = TRM_Titans_TestTime(model, device)

# 3. 각 퍼즐 평가
for puzzle_name, puzzle_data in puzzles.items():
    # Demo pairs 준비
    demo_pairs = [
        (inp_tokens, label_tokens)
        for example in puzzle_data["train"]
    ]

    # TTT 적응
    ttt.test_time_adapt(
        demo_pairs=demo_pairs,
        n_steps=10,      # 적응 스텝 수
        lr=0.01,         # 적응 learning rate
        puzzle_id=idx,
        accumulate_memory=True
    )

    # Test 예측
    for test_example in puzzle_data["test"]:
        pred = ttt.predict(
            test_input,
            update_during_prediction=False,  # 예측 시 Memory 고정
            reset_memory=False               # 적응된 Memory 유지
        )
        results.append(pred)

# 4. Pass@k 계산 및 저장
metrics = compute_pass_at_k(results, labels, pass_ks=(1, 2))
save_submission(submission_path)
```

**CLI 옵션:**
```bash
python evaluate_ttt.py \
    --checkpoint /path/to/checkpoint \
    --data_path data/arc-aug-1000 \
    --ttt_steps 10 \              # 적응 스텝 수
    --ttt_lr 0.01 \               # 적응 lr
    --output_path outputs/ttt_eval \
    --device cuda:0 \
    --verbose \
    --no_accumulate_memory        # Memory 누적 비활성화 (선택)
```

**출력 파일:**
- `submission.json`: Kaggle 제출용 (attempt_1, attempt_2)
- `metrics.json`: pass@1, pass@2 정확도
- `eval_config.json`: 평가 설정 기록

---

#### 5.5 코드 경로별 Memory 동작 비교

| 코드 경로 | Memory 리셋 시점 | Memory 업데이트 | 용도 |
|-----------|-----------------|----------------|------|
| **Pretraining** | 없음 (persist) | Surprise (forward) | 패턴 학습 |
| **Std Evaluation** | 각 배치 | No (no_grad) | 성능 측정 |
| **TTT Adapt** | 각 퍼즐 시작 | Surprise (forward) | 퍼즐 적응 |
| **TTT Predict** | 선택적 | 선택적 | 예측 생성 |

---

#### 5.6 학습 컴포넌트별 역할 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Learning Component Roles                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Puzzle Embedding (SignSGD)                                                 │
│  └── 각 퍼즐 ID별 task-specific representation 학습                         │
│      • Sparse update (해당 퍼즐 ID만 업데이트)                               │
│      • 높은 lr (1e-2) 사용                                                  │
│                                                                             │
│  Attention + MLP (AdamW)                                                    │
│  └── 일반적인 패턴 인식 및 변환 학습                                         │
│      • Q, K, V, O projection + SwiGLU MLP                                   │
│      • 표준 lr (1e-4) 사용                                                  │
│                                                                             │
│  Memory Template (Frozen)                                                   │
│  └── 초기 memory state 정의                                                │
│      • reset_all_memory() 시 이 값으로 복원                                 │
│      • 학습되지 않음                                                        │
│                                                                             │
│  Memory Current Weights (Surprise)                                          │
│  └── 런타임 패턴 저장                                                       │
│      • Forward pass에서 자동 업데이트                                       │
│      • M_t = (1-α)*M_{t-1} - η*∇(||M(k)-v||²)                              │
│      • Backprop 그래프에 포함되지 않음                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 6. 파일 구조

```
TinyRecursiveModels/
├── models/recursive_reasoning/
│   ├── trm_titans.py      # TRM-Titans v7 모델 (메인)
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
