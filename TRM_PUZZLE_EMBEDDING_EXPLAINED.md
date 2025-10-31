# TRM Puzzle Embedding 문제 상세 분석

## 1️⃣ ARC Task의 본질

### ARC는 Few-shot Learning Task
```json
// 하나의 ARC Puzzle
{
  "train": [
    {"input": [[1,2], [3,4]], "output": [[4,3], [2,1]]},  // Example 1
    {"input": [[5,6], [7,8]], "output": [[8,7], [6,5]]},  // Example 2
    {"input": [[0,1], [2,3]], "output": [[3,2], [1,0]]}   // Example 3
  ],
  "test": [
    {"input": [[9,0], [1,2]], "output": ???}  // Query: 규칙을 유추해서 풀어라
  ]
}
```

**인간이 푸는 방법:**
1. Training examples (1,2,3)를 관찰
2. 규칙 발견: "180도 회전"
3. Test input에 규칙 적용 → output 생성

**핵심:** Training examples가 필수!

---

## 2️⃣ 현재 TRM의 처리 방식

### Step 1: Dataset 생성 (`build_arc_dataset.py:268-285`)

```python
# Puzzle A (3개 training examples)
for puzzle in group:
    for (inp, out) in puzzle.examples:  # 각 example 독립적
        results["inputs"].append(inp)         # [900] 저장
        results["labels"].append(out)         # [900] 저장

    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
    # ↑ 모든 examples가 같은 puzzle_id를 공유
```

**생성되는 데이터:**
```
Example 1: input_1 [900], output_1 [900], puzzle_id=A
Example 2: input_2 [900], output_2 [900], puzzle_id=A
Example 3: input_3 [900], output_3 [900], puzzle_id=A
```

**문제:**
- ❌ "이 input이 Example 1이다"라는 정보 없음
- ❌ "다른 examples는 뭐였는지" 정보 없음
- ❌ Examples 간의 관계 없음
- ✅ 오직 `puzzle_id=A`만 있음

---

### Step 2: Pretrain 중 Batch 구성 (`puzzle_dataset.py:231-235`)

```python
batch = {
    "inputs": dataset["inputs"][batch_indices],              # [B, 900]
    "labels": dataset["labels"][batch_indices],              # [B, 900]
    "puzzle_identifiers": dataset["puzzle_identifiers"][...] # [B]
}
```

**Batch 예시:**
```python
batch = {
    "inputs": [
        input_1,  # Puzzle A, Example 1
        input_5,  # Puzzle B, Example 2
        input_2,  # Puzzle A, Example 2  ← 같은 Puzzle!
        input_9,  # Puzzle C, Example 1
    ],
    "puzzle_identifiers": [A, B, A, C]
}
```

**문제:**
- ❌ Batch 안에 Puzzle A의 다른 examples가 섞여 있어도 **서로 모름**
- ❌ "input_1과 input_2는 같은 puzzle"이라는 정보 없음
- ❌ Training examples context 전혀 없음

---

### Step 3: TRM Forward (`trm.py:162-210`)

```python
def _input_embeddings(self, input, puzzle_identifiers):
    # 1. Token embedding
    embedding = self.embed_tokens(input)  # [B, 900, 512]

    # 2. Puzzle embedding (단순 lookup!)
    puzzle_embedding = self.puzzle_emb(puzzle_identifiers)  # [B, 8192]
    #                                   ^^^^^^^^^^^^^^^^^
    #                                   오직 puzzle_id만 사용!

    # 3. Concatenate
    embedding = torch.cat((puzzle_embedding.view(-1, 16, 512), embedding), dim=-2)
    return embedding  # [B, 916, 512]

def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(
        batch["inputs"],           # [B, 900]
        batch["puzzle_identifiers"] # [B] ← 오직 ID만!
    )

    # Recursive reasoning
    for _H_step in range(H_cycles):
        for _L_step in range(L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, ...)
        z_H = self.L_level(z_H, z_L, ...)

    # Prediction
    output = self.lm_head(z_H)  # [B, 900, 12]
    return output
```

**실제 동작:**
```python
# Batch 안에 Puzzle A의 2개 examples
input_1, puzzle_id=A  →  puzzle_emb[A] + embed(input_1) → predict output_1
input_2, puzzle_id=A  →  puzzle_emb[A] + embed(input_2) → predict output_2
                         ^^^^^^^^^^^^^ 같은 embedding!

# 하지만 input_1과 input_2가 서로 관련 있다는 정보 없음!
```

---

## 3️⃣ 학습이 어떻게 작동하는가?

### Pretrain 중

```python
# Puzzle A (규칙: 180도 회전)
# Gradient가 puzzle_emb[A]를 업데이트

Step 1: input_1 → TRM(input_1, puzzle_emb[A]) → pred_1
        Loss: MSE(pred_1, output_1)
        Gradient: ∂Loss/∂puzzle_emb[A]  ← 업데이트!

Step 2: input_2 → TRM(input_2, puzzle_emb[A]) → pred_2
        Loss: MSE(pred_2, output_2)
        Gradient: ∂Loss/∂puzzle_emb[A]  ← 또 업데이트!

Step 3: input_3 → TRM(input_3, puzzle_emb[A]) → pred_3
        Loss: MSE(pred_3, output_3)
        Gradient: ∂Loss/∂puzzle_emb[A]  ← 또 업데이트!

# 결과: puzzle_emb[A]가 "180도 회전 규칙"을 암묵적으로 학습
```

**학습되는 것:**
```python
puzzle_emb[A] ≈ "180도 회전 규칙의 representation"
puzzle_emb[B] ≈ "색상 반전 규칙의 representation"
puzzle_emb[C] ≈ "대각선 대칭 규칙의 representation"
...
```

**핵심:**
- ✅ TRM weights는 "규칙 적용 능력" 학습
- ✅ `puzzle_emb[id]`는 "특정 규칙의 encoding" 학습
- ❌ Training examples를 보고 추론하는 능력은 학습 안 됨!

---

## 4️⃣ Test 시 문제

### 새로운 Puzzle Z (학습 안 한 ID)

```python
# Test 시
test_input, puzzle_id=Z  # Z는 처음 보는 ID!

# Forward
puzzle_embedding = self.puzzle_emb(puzzle_id=Z)
# → puzzle_emb.weights[Z]는 초기화 상태 (거의 zero vector)

input_embeddings = puzzle_embedding + embed(test_input)
# → "어떤 규칙인지" 정보가 없음!

output = TRM(input_embeddings)
# → 추론 실패!
```

**왜 실패하는가?**
```python
# Pretrain 때:
puzzle_emb[A] = "학습된 180도 회전 규칙"  ✓

# Test 때:
puzzle_emb[Z] = "초기화 상태 (zero)"  ✗
# → TRM은 "어떤 규칙을 적용해야 하는지" 모름!
```

---

## 5️⃣ 근본 문제: Memorization vs Reasoning

### 현재 TRM: Memorization

```python
학습 시:
  puzzle_emb[A] ← "180도 회전 규칙" 암기
  puzzle_emb[B] ← "색상 반전 규칙" 암기
  ...

Test 시:
  puzzle_emb[Z] ← 처음 봄! 암기 안 됨!
  → 실패
```

### 올바른 접근: Reasoning

```python
학습 시:
  TRM weights ← "Examples를 보고 규칙을 유추하는 능력" 학습

Test 시:
  Training examples = [ex1, ex2, ex3]  # ARC가 제공!

  # Examples를 context로 사용
  context = encode_examples([ex1, ex2, ex3])

  # TRM이 규칙을 유추
  output = TRM(test_input, context)  # Zero-shot 가능! ✓
```

---

## 6️⃣ 왜 Training Examples를 사용하지 않는가?

### 코드 전체를 확인한 결과

**Dataset 생성 (`build_arc_dataset.py`):**
- ❌ Training examples 정보 저장 안 함
- ✅ 각 example을 독립적으로 저장

**Dataset 로딩 (`puzzle_dataset.py`):**
- ❌ Training examples grouping 없음
- ✅ Random shuffle로 batch 구성

**Model Forward (`trm.py`):**
- ❌ Training examples 입력 없음
- ✅ 오직 `puzzle_id` lookup

**어디에도 training examples를 활용하는 코드가 없습니다!**

---

## 7️⃣ 사용자의 질문에 대한 답변

### Q1: "왜 학습하지 않은 task id는 못 푸는 거야?"

**A:**
- `puzzle_emb[new_id]`가 학습 안 됨 (zero vector)
- TRM은 "어떤 규칙인지" 정보 없이는 추론 불가

### Q2: "weight 고정시켜두고 inference하면 되잖아"

**A:**
- ✅ **맞습니다!** TRM weights는 고정되어도 됩니다
- ❌ **하지만** training examples를 활용해야 가능합니다
- ❌ **현재는** training examples를 전혀 안 써서 불가능합니다

### Q3: "어떻게 compute 해?"

**A:**
```python
# 제안: Training examples를 encode
training_examples = get_training_examples(puzzle_id)
# → ARC는 항상 train/test split으로 제공됨

# Encode each example
example_embeds = [encode(ex) for ex in training_examples]

# Aggregate (mean pooling or attention)
puzzle_emb = aggregate(example_embeds)

# 이제 zero-shot 가능!
```

### Q4: "지금 모델에도 구현이 돼 있지 않아?"

**A:**
- ❌ **전혀 구현 안 돼 있습니다!**
- Dataset에서 examples grouping 없음
- Model에서 examples 입력받지 않음
- 오직 `puzzle_id` lookup만 사용

---

## 8️⃣ 해결책

### Option A: Dataset 재구성

```python
# 현재: 각 example 독립
batch = {
    "inputs": [input_1, input_5, input_2, ...],  # Random shuffle
    "puzzle_identifiers": [A, B, A, ...]
}

# 제안: Examples grouping
batch = {
    "puzzle_id": A,
    "training_examples": [
        (input_1, output_1),
        (input_2, output_2),
        (input_3, output_3)
    ],
    "query_input": input_test,
    "query_output": output_test
}
```

### Option B: Inference 시 Examples 활용

```python
# Pretrain: 기존 방식 유지 (backward compatibility)
if self.training:
    puzzle_emb = self.puzzle_emb_table(puzzle_id)  # Lookup
else:
    # Inference: Training examples로부터 compute
    training_examples = load_training_examples(puzzle_id)
    puzzle_emb = self.encode_examples(training_examples)
```

---

## 9️⃣ 정리

| | **현재 TRM** | **올바른 접근** |
|---|---|---|
| **Paradigm** | Memorization | Reasoning |
| **Training examples** | ❌ 사용 안 함 | ✅ Context로 활용 |
| **puzzle_emb** | Learnable lookup | Computed from examples |
| **새 puzzle** | ❌ 못 풂 | ✅ Zero-shot 가능 |
| **ARC 본질** | ❌ 무시 | ✅ Few-shot learning |

**핵심:**
- 현재 TRM은 ARC를 "classification task"로 취급
- 올바르게는 "few-shot reasoning task"로 접근해야 함
- Training examples를 활용하면 zero-shot generalization 가능

---

이제 명확하신가요?
