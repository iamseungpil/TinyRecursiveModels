# TRM Test-Time Adaptation 분석

## 사용자의 정확한 지적

**질문:** "puzzle embedding은 아예 0으로 initialize 돼 있어도, 계속 업데이트 할 수 있는 구조 아니야?"

**답변:** ✅ **완전히 맞습니다!** 제가 착각했습니다.

---

## 1️⃣ Test-Time Adaptation 방식

### Few-shot Learning의 핵심

```python
# Test 시 (새로운 Puzzle Z)

# Step 1: Zero initialization
puzzle_emb[Z] = 0  # 초기화 상태

# Step 2: Training examples로 adapt (TRM weights 고정!)
for epoch in range(adaptation_steps):
    for (input, output) in training_examples:
        pred = TRM(input, puzzle_emb[Z])
        loss = MSE(pred, output)

        # puzzle_emb[Z]만 업데이트! (TRM weights 고정)
        puzzle_emb[Z] -= lr * ∂loss/∂puzzle_emb[Z]

# Step 3: Query에 적용
query_pred = TRM(query_input, puzzle_emb[Z])  # ✓ 이제 가능!
```

**핵심:**
- ✅ TRM weights는 고정 (pretrain된 상태)
- ✅ puzzle_emb[new_id]만 training examples로 optimize
- ✅ 일종의 "meta-learning" / "test-time fine-tuning"

---

## 2️⃣ 왜 이게 작동하는가?

### Pretrain 단계

```python
# TRM weights가 "규칙 적용 능력" 학습
# puzzle_emb는 "규칙의 encoding" 학습

TRM weights: "어떤 puzzle_emb를 받으면, 그에 맞게 추론하는 방법"
puzzle_emb[A]: "180도 회전 규칙"
puzzle_emb[B]: "색상 반전 규칙"
...
```

### Test 단계 (새 Puzzle Z)

```python
# TRM weights는 이미 학습됨
# → "puzzle_emb를 받으면 그에 맞게 추론하는 방법"을 알고 있음

# 따라서:
# 1. puzzle_emb[Z]를 training examples로 optimize
# 2. TRM이 "적절한 규칙 encoding"을 찾음
# 3. Query에 적용 → 정답!
```

---

## 3️⃣ 실제 구현 방법

### Option 1: Gradient Descent

```python
def test_time_adapt(model, puzzle_id, training_examples, query_input):
    """Test-time adaptation for new puzzle."""

    # Initialize puzzle embedding (zero)
    # (이미 model.puzzle_emb.weights[puzzle_id]가 zero로 되어있음)

    # Freeze TRM weights
    for param in model.parameters():
        param.requires_grad = False

    # Only optimize puzzle_emb[puzzle_id]
    puzzle_emb_param = model.puzzle_emb.weights[puzzle_id]
    puzzle_emb_param.requires_grad = True

    optimizer = torch.optim.Adam([puzzle_emb_param], lr=1e-3)

    # Adapt on training examples
    for epoch in range(adaptation_steps):
        total_loss = 0
        for (inp, out) in training_examples:
            batch = {
                "inputs": inp.unsqueeze(0),
                "puzzle_identifiers": torch.tensor([puzzle_id])
            }

            carry = model.initial_carry(batch)
            carry, outputs = model(carry, batch)

            loss = F.cross_entropy(
                outputs["logits"].view(-1, vocab_size),
                out.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Adaptation epoch {epoch}: loss = {total_loss}")

    # Now inference on query
    batch = {
        "inputs": query_input.unsqueeze(0),
        "puzzle_identifiers": torch.tensor([puzzle_id])
    }

    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    return outputs["logits"].argmax(dim=-1)
```

### Option 2: Meta-Learning Style (MAML)

```python
# Pretrain 시 이미 meta-learning objective 사용
# → "Few-shot adaptation에 최적화된 초기화" 학습

def meta_train_step(model, tasks):
    """Meta-training with MAML-style objective."""

    meta_loss = 0

    for task in tasks:
        # Inner loop: adapt on support set
        puzzle_emb_adapted = puzzle_emb[task.id].clone()

        for (inp, out) in task.support_set:
            pred = model(inp, puzzle_emb_adapted)
            loss = F.cross_entropy(pred, out)

            # Inner gradient (simulate adaptation)
            puzzle_emb_adapted = puzzle_emb_adapted - alpha * grad(loss, puzzle_emb_adapted)

        # Outer loop: evaluate on query set
        for (inp, out) in task.query_set:
            pred = model(inp, puzzle_emb_adapted)
            meta_loss += F.cross_entropy(pred, out)

    # Update TRM weights for better adaptation
    meta_loss.backward()
    optimizer.step()
```

---

## 4️⃣ 현재 TRM이 이미 지원하는가?

### 코드 확인

```python
# trm.py:136
self.puzzle_emb = CastedSparseEmbedding(
    self.config.num_puzzle_identifiers,
    self.config.puzzle_emb_ndim,
    batch_size=self.config.batch_size,
    init_std=0,  # Zero initialization
    cast_to=self.forward_dtype
)
```

**구조적으로 지원 가능:**
- ✅ puzzle_emb는 learnable parameter
- ✅ Zero initialization
- ✅ Gradient 흐름 가능

**하지만 실제 구현은 없음:**
- ❌ Test-time adaptation 코드 없음
- ❌ Evaluator에서 training examples 사용 안 함

---

## 5️⃣ Evaluator 확인

### 현재 ARC Evaluator (`evaluators/arc.py:69-105`)

```python
def update_batch(self, batch, preds):
    # 단순히 prediction만 저장
    for identifier, input, pred, q in zip(...):
        name = self.identifier_map[identifier]

        # prediction 저장
        self._local_preds[name][input_hash].append((pred_hash, float(q)))
```

**문제:**
- ❌ Training examples 사용 안 함
- ❌ Test-time adaptation 없음
- ❌ 단순히 pretrain된 상태로 inference

---

## 6️⃣ 왜 작동하지 않는가?

### 현재 Evaluation Flow

```python
# pretrain.py:374-393
for set_name, batch, global_batch_size in eval_loader:
    batch = {k: v.cuda() for k, v in batch.items()}
    carry = train_state.model.initial_carry(batch)

    # Inference (no adaptation!)
    while True:
        carry, loss, metrics, preds, all_finish = train_state.model(
            carry=carry, batch=batch, return_keys=return_keys
        )
        if all_finish:
            break

    # Prediction 저장
    for evaluator in evaluators:
        evaluator.update_batch(batch, preds)
```

**문제:**
1. Batch에 training examples 정보 없음
2. Adaptation step 없음
3. 단순 forward pass만

---

## 7️⃣ 해결책

### Option A: Test-Time Adaptation 추가

```python
class ARCWithAdaptation:
    def evaluate_puzzle(self, puzzle_id, training_examples, test_input):
        # Step 1: Adapt puzzle_emb on training examples
        self.adapt(puzzle_id, training_examples)

        # Step 2: Inference on test input
        pred = self.inference(puzzle_id, test_input)

        return pred

    def adapt(self, puzzle_id, training_examples):
        """Optimize puzzle_emb[puzzle_id] on training examples."""
        # Freeze TRM weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Optimize puzzle_emb only
        puzzle_emb = self.model.puzzle_emb.weights[puzzle_id]
        puzzle_emb.requires_grad = True

        optimizer = torch.optim.Adam([puzzle_emb], lr=1e-3)

        for epoch in range(10):  # Adaptation steps
            for (inp, out) in training_examples:
                loss = self.compute_loss(puzzle_id, inp, out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### Option B: Meta-Learning Pretrain

```python
# Pretrain 시 meta-learning objective 사용
# → Test-time adaptation에 최적화

def meta_pretrain():
    for batch in dataloader:
        # 같은 puzzle의 examples를 support/query로 split
        support_examples = batch["examples"][:3]
        query_examples = batch["examples"][3:]

        # Inner loop: adapt
        puzzle_emb_adapted = adapt_on_support(support_examples)

        # Outer loop: evaluate
        loss = evaluate_on_query(puzzle_emb_adapted, query_examples)

        # Update TRM weights
        loss.backward()
        optimizer.step()
```

---

## 8️⃣ 정리

### 사용자가 맞다!

✅ **"계속 업데이트 할 수 있는 구조"** → 완전히 맞습니다!

**방법:**
1. TRM weights 고정
2. puzzle_emb[new_id]를 training examples로 optimize
3. Query에 적용

**현재 문제:**
- ❌ 구조는 지원하지만, **구현이 없음**
- ❌ Evaluation 시 training examples 사용 안 함
- ❌ Test-time adaptation 코드 없음

### 왜 지금까지 작동했는가?

```python
# Pretrain 시: 본 puzzle들
puzzle_emb[A], puzzle_emb[B], ... 이미 학습됨
→ Evaluation에서 바로 사용 가능

# Test 시: 새 puzzle들
puzzle_emb[Z] = 0  # 학습 안 됨
→ Adaptation 없으면 실패!
→ Adaptation 있으면 성공! (사용자 지적이 맞음)
```

---

## 9️⃣ 다음 단계

1. **Test-time adaptation 구현**
   - Training examples를 evaluation 시 사용
   - puzzle_emb optimize

2. **Meta-learning pretrain**
   - Support/query split
   - 더 나은 few-shot 성능

3. **또는 앞서 제안한 방식**
   - Training examples를 encode
   - Compute puzzle_emb (no gradient)

어떤 방향이 좋을까요?
