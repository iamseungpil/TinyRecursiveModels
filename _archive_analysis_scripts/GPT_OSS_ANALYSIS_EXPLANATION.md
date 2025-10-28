# GPT-OSS 분석 생성 코드 설명

## 1. 코드 위치 및 구조

**메인 파일**: `/home/ubuntu/dreamcoder-arc/InternVL/src/gpt_oss_analyzer.py` (562줄)

**주요 클래스**: `GPTOSSARCAnalyzer`

## 2. DSL Primitive 설명 추가 방식

### 2.1 System Prompt에 전체 Primitive 목록 포함

**위치**: `gpt_oss_analyzer.py` lines 203-234 (system message)

**중요**: **사용된 DSL만 추가하는 것이 아니라, 모든 가능한 primitive의 전체 목록을 system prompt에 포함**

```python
{
    "role": "system",
    "content": """You are an ARC Puzzle Solver.

# DSL Primitive Functions Reference

These transformations may use combinations of the following operations:

## Geometric Transformations
- rot90: Rotate grid 90 degrees clockwise
- rot180: Rotate grid 180 degrees
- rot270: Rotate grid 270 degrees clockwise (90 degrees counterclockwise)
- flipx: Flip grid vertically (top-bottom mirror, along horizontal axis)
- flipy: Flip grid horizontally (left-right mirror, along vertical axis)
- swapxy: Transpose grid (swap rows and columns)
- identity: Return grid unchanged

## Gravity Operations
- gravity_left: Move all colored (non-zero) cells to the left within their row while preserving order
- gravity_right: Move all colored (non-zero) cells to the right within their row while preserving order
- gravity_up: Move all colored (non-zero) cells upward within their column while preserving order
- gravity_down: Move all colored (non-zero) cells downward within their column while preserving order

## Repetition & Mirroring
- repeatX: Repeat the grid horizontally (side by side)
- repeatY: Repeat the grid vertically (stacked)
- mirrorX: Append a horizontal reflection of the grid
- mirrorY: Append a vertical reflection of the grid

# Valid channels: analysis, commentary, final"""
}
```

### 2.2 개별 Transformation Hint

**메서드**: `get_transformation_hint()` (lines 93-185)

이 메서드는 **사용된 특정 DSL program을 분석**하여 해당 변환에 대한 힌트를 생성합니다:

```python
def get_transformation_hint(self, program: str) -> str:
    """Extract transformation hint from DSL program without revealing the DSL itself"""
    dsl_semantics = {
        "gravity_left": "The transformation moves all colored (non-zero) cells to the left...",
        "gravity_right": "The transformation moves all colored (non-zero) cells to the right...",
        # ... 더 많은 primitive 설명
    }
```

**특징**:
- 프로그램에서 사용된 primitive를 추출
- 해당 primitive의 자연어 설명을 제공
- "gravity_left", "DSL", "program" 같은 technical term은 언급하지 말라고 지시

## 3. User Prompt 구조

**위치**: lines 236-282

```python
{
    "role": "user",
    "content": f"""# ARC Puzzle Solver - Training Data Generation

## Example of correct format:
{incontext_example}  # In-context learning example

## Task
Solve this ARC puzzle:

{examples_text}  # Train examples

Test Input:
{self.grid_to_string(test_input)}

GIVEN INFORMATION FOR GUIDANCE:
- Expected Output: {expected_output}
- Transformation Hint: {transformation_hint}  # 사용된 DSL의 힌트

CRITICAL INSTRUCTIONS:
- ALWAYS use <|channel|>analysis first to work through the pattern
- ALWAYS switch to <|channel|>final for DETAILED, VERBOSE step-by-step reasoning
- DO NOT mention technical terms like "gravity_left", "DSL", "program"
- Explain the transformation by analyzing visual/logical patterns directly
- Write at least 6-10 detailed reasoning steps in the final channel
"""
}
```

## 4. 전체 프로세스 흐름

```
1. Helmholtz Sample 로드
   ↓
2. DSL Program 추출 (예: "(lambda (gravity_left $0))")
   ↓
3. Transformation Hint 생성
   - get_transformation_hint() 호출
   - 프로그램에서 "gravity_left" 추출
   - 자연어 힌트 반환
   ↓
4. System Prompt 준비
   - 전체 DSL primitive 목록 포함 (모든 가능한 primitive)
   ↓
5. User Prompt 준비
   - Train examples (입력/출력 그리드)
   - Test input
   - Expected output
   - Transformation hint (해당 DSL의 자연어 설명)
   ↓
6. GPT-OSS 추론
   - <|channel|>analysis: 패턴 분석
   - <|channel|>final: 상세한 단계별 추론
   ↓
7. 응답 파싱 및 저장
   - Final channel 내용 추출
   - JSON 파일로 저장
```

## 5. 핵심 설계 결정

### 5.1 왜 모든 Primitive를 System Prompt에 포함?

**이유**:
1. **일관성**: 모든 샘플이 동일한 primitive 정의를 참조
2. **컨텍스트**: 모델이 다양한 변환 유형을 인지
3. **일반화**: 복합 변환(여러 primitive 조합) 이해 향상

**단점**:
- Token 수 증가 (~1,000 tokens 추가)
- 하지만 DSL reference가 추론 품질을 크게 향상시킴

### 5.2 Transformation Hint vs DSL Reference

**Transformation Hint** (User prompt):
- 해당 샘플의 **특정 DSL에 대한 자연어 설명**
- 예: "The transformation moves all colored cells to the left..."
- 모델에게 정답 방향 제시

**DSL Reference** (System prompt):
- **모든 가능한 primitive의 간략한 정의**
- 모델이 다양한 변환 유형을 학습
- In-context learning 강화

### 5.3 Technical Term 금지

**중요 지시사항** (line 274-278):
```
- DO NOT mention technical terms like "gravity_left", "DSL", "program", or any code-like names
- Explain the transformation by analyzing the visual/logical patterns directly
- Focus on what changes between input and output (positions, colors, arrangements)
```

**목적**:
- 자연어 설명 생성 (코드가 아닌)
- 인간이 이해하기 쉬운 분석
- TRM 학습에 적합한 형태

## 6. 실제 예시

### 6.1 Helmholtz Sample

```json
{
  "task_id": "helmholtz_82819916_2096",
  "program": "(lambda (gravity_left $0))",
  "examples": [
    {"input": [[0,2,0,0,0,8,0,0,0,0], ...], "output": [[2,8,0,0,0,0,0,0,0,0], ...]}
  ],
  "test": [{"input": [...], "output": [...]}]
}
```

### 6.2 생성된 Transformation Hint

```
"The transformation moves all colored (non-zero) cells to the left within their row,
compressing them together while preserving their original left-to-right order."
```

### 6.3 모델 출력 (Final Channel)

```
Step 1: Understanding the transformation pattern
By examining both training examples, I observe that this transformation moves all
colored cells to the leftmost positions in each row...

Step 2: Analyzing the grid structure
In the test input grid (10 rows × 10 columns), I identify colored cells (values 2 and 8)
scattered across different positions...

Step 3: Applying the transformation
For each row, I move all non-zero values to the left side while maintaining their
original left-to-right order...
```

## 7. 데이터 통계

**생성된 데이터셋**:
- 총 샘플: 8,572개
- Train examples: 20,187개
- Test examples: 15,961개
- 사용된 primitive 다양성: 20+개 (첫 20개 샘플에서)

**주요 primitive 분포** (첫 20개 샘플):
- gravity_left (2회)
- mirrorY (2회)
- rot270 (2회)
- swapxy (2회)
- ic_erasecol, ic_connectX, rot180, ic_compress2, ic_fill, bottom_half, gravity_right, ic_makeborder, repeatY, mirrorX (각 1회)

## 8. 파일 위치

**코드**:
- `/home/ubuntu/dreamcoder-arc/InternVL/src/gpt_oss_analyzer.py` - GPT-OSS 분석기
- `/home/ubuntu/dreamcoder-arc/ec/src/generator/helmholtz_generator.py` - 샘플 생성기

**데이터**:
- `/data/helmarc_correct/20251024_062500/samples.json` - 원본 Helmholtz 샘플 (8,572개)
- `/data/helmarc_analyzed/gpu*/` - GPT-OSS 분석 결과 (진행 중)
- `/home/ubuntu/TinyRecursiveModels/helmarc_sample_examples.json` - 예시 3개

**설정**:
- `/home/ubuntu/dreamcoder-arc/ec/configs/helmarc_10k.yaml` - Generation 설정
