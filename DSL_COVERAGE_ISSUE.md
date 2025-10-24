# ⚠️ DSL Hint Coverage 문제 발견

## 문제 요약

**현재 상황**: 10개 primitive만 transformation hint를 가지고 있지만, 실제로는 훨씬 더 많은 primitive가 사용됨

## 1. 정의된 Transformation Hints (10개)

`gpt_oss_analyzer.py:97-108`의 `dsl_semantics` 딕셔너리:

```python
dsl_semantics = {
    "gravity_left": "...",
    "gravity_right": "...",
    "gravity_up": "...",
    "gravity_down": "...",
    "ic_erasecol": "...",
    "flipx": "...",
    "flipy": "...",
    "rot90": "...",
    "rot180": "...",
    "rot270": "...",
}
```

## 2. 실제 사용된 Primitives (14개, 첫 20개 샘플 기준)

1. ✅ gravity_left - **hint 있음**
2. ✅ gravity_right - **hint 있음**
3. ✅ ic_erasecol - **hint 있음**
4. ✅ rot180 - **hint 있음**
5. ✅ rot270 - **hint 있음**
6. ❌ **bottom_half** - **hint 없음**
7. ❌ **ic_compress2** - **hint 없음**
8. ❌ **ic_connectX** - **hint 없음**
9. ❌ **ic_fill** - **hint 없음**
10. ❌ **ic_makeborder** - **hint 없음**
11. ❌ **mirrorX** - **hint 없음**
12. ❌ **mirrorY** - **hint 없음**
13. ❌ **repeatY** - **hint 없음**
14. ❌ **swapxy** - **hint 없음**

## 3. Fallback Hint (generic, 도움 안 됨)

Hint가 없는 primitive들은 다음 generic 메시지를 받습니다:

```python
# Line 115
return "The transformation changes the grid according to a consistent pattern. Observe the examples carefully to identify the rule."
```

**문제점**: 이것은 사실상 "잘 모르겠다"는 의미입니다.

## 4. System Prompt vs Transformation Hint 불일치

### System Prompt (lines 213-232)에 포함된 내용:
```
## Geometric Transformations
- rot90, rot180, rot270, flipx, flipy, swapxy, identity

## Gravity Operations
- gravity_left, gravity_right, gravity_up, gravity_down

## Repetition & Mirroring
- repeatX, repeatY, mirrorX, mirrorY
```

**문제**: System prompt에는 `mirrorX`, `mirrorY`, `repeatX`, `repeatY`, `swapxy`가 **있지만**, transformation hint dictionary에는 **없습니다**!

## 5. 영향 분석

### 5.1 Coverage 부족

**전체 8,572개 샘플 중**:
- 약 35-40%가 hint 없는 primitive 사용할 가능성
- 이들은 generic fallback hint만 받음

### 5.2 품질 저하

**Hint가 없는 경우**:
- GPT-OSS가 패턴을 자체적으로 찾아야 함
- 더 긴 추론 시간
- 잘못된 패턴 식별 가능성
- 덜 구조화된 분석

### 5.3 일관성 문제

**System prompt와 transformation hint 불일치**:
- System: "mirrorX는 horizontal reflection을 append"
- Transformation hint: "Observe examples carefully..." (도움 안 됨)

## 6. 실제 영향 확인 필요

현재 생성 중인 샘플들 중:
- 샘플 1-5: 어떤 primitive 사용했는지 확인 필요
- 특히 hint 없는 primitive 샘플의 분석 품질 확인

## 7. 권장 조치

### 즉시 조치 (Critical):

1. **모든 primitive에 대한 hint 추가**
   - bottom_half
   - ic_compress2
   - ic_connectX
   - ic_fill
   - ic_makeborder
   - mirrorX, mirrorY
   - repeatX, repeatY
   - swapxy

2. **arcPrimitivesIC2.py에서 docstring 추출**
   - 각 primitive의 공식 설명 활용
   - 자동으로 hint dictionary 생성

### 장기 조치:

3. **자동 coverage 체크**
   - 생성된 샘플의 primitive 분포 분석
   - Hint 없는 primitive 자동 감지
   - 경고 메시지 출력

4. **Hint 품질 평가**
   - Hint 있는 샘플 vs 없는 샘플 분석 비교
   - 생성된 reasoning 길이 및 품질 측정

## 8. 현재 진행 중인 분석의 영향

**현재 GPU 0에서 생성 중인 5개 샘플**:
- 몇 개가 hint 없는 primitive 사용했을지 불확실
- Checkpoint 생성 후 확인 필요
- 품질이 낮은 샘플은 재생성 고려

## 9. Mapping 방식

**현재 방식** (line 111-115):
```python
# Extract operation name from program like "(lambda (gravity_left $0))"
for op_name, hint in dsl_semantics.items():
    if op_name in program:
        return hint

return "The transformation changes the grid according to a consistent pattern..."
```

**방식**:
- Simple string matching (`if op_name in program`)
- 첫 번째 매칭된 primitive의 hint 반환
- 매칭 실패 시 generic fallback

**문제**:
- 복합 primitive (여러 개 조합)에 대한 처리 없음
- Fallback이 너무 generic
- Dictionary 기반이라 확장성 낮음

## 10. 통계

**Coverage**:
- Defined hints: 10개
- Used primitives (첫 20개 샘플): 14개
- **Missing hints: 9개 (64% 누락!)**

**Impact**:
- 만약 전체 8,572개 샘플에서 동일한 분포라면
- **약 3,000-4,000개 샘플이 generic hint만 받음**
- 이는 데이터셋 품질에 큰 영향
