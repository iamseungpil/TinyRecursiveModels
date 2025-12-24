---
name: iterative-code-review
description: Iteratively improve code quality by using modular-code-architect agent to fix issues and code-reviewer agent to validate quality. Use when implementing new features, after bug fixes, during refactoring, or when preparing code for production deployment. Loops until code-reviewer reports no critical issues.
---

# Iterative Code Review

코드 아키텍트와 코드 리뷰어를 반복적으로 사용하여 코드를 production-ready 수준까지 개선하는 스킬입니다.

## 사용 시점

다음과 같은 상황에서 이 스킬을 사용합니다:
- 새로운 기능 구현 후 품질 검증이 필요할 때
- 복잡한 버그 수정 후 side effect 확인이 필요할 때
- 리팩토링 후 코드 품질 검증이 필요할 때
- Production 배포 전 최종 검증이 필요할 때

## 워크플로우

### Phase 1: 초기 분석
1. 대상 파일/브랜치 식별
2. 현재 상태 파악 (git status, git diff)
3. Todo 리스트 생성

### Phase 2: 반복 개선 루프

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────────────┐     ┌─────────────────────┐   │
│  │  Code Architect │────▶│   Code Reviewer     │   │
│  │    (수정)       │     │     (검토)          │   │
│  └─────────────────┘     └──────────┬──────────┘   │
│           ▲                         │              │
│           │                         ▼              │
│           │              ┌─────────────────────┐   │
│           │              │  Critical Issues?   │   │
│           │              └──────────┬──────────┘   │
│           │                         │              │
│           │         Yes             │    No        │
│           └─────────────────────────┘    │         │
│                                          ▼         │
│                              ┌─────────────────┐   │
│                              │   Complete!     │   │
│                              └─────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Phase 3: 완료
1. 모든 변경사항 커밋
2. 브랜치 push
3. 최종 상태 보고

## 에이전트 호출 지침

### 1. modular-code-architect 에이전트 호출

수정이 필요할 때 Task 도구로 호출:
```
Task tool 사용:
- subagent_type: "modular-code-architect"
- prompt: 수정해야 할 이슈들의 상세 설명
- 파일 경로, 라인 번호, 구체적인 수정 방향 포함
```

### 2. code-reviewer 에이전트 호출

수정 후 검토할 때 Task 도구로 호출:
```
Task tool 사용:
- subagent_type: "code-reviewer"
- prompt:
  - main 브랜치와 비교 또는 특정 커밋과 비교
  - ultrathink 수준의 심층 분석 요청
  - Critical, Warning, Suggestion 구분하여 보고 요청
```

## 반복 종료 조건

다음 조건이 모두 충족되면 반복을 종료합니다:

1. **Critical 이슈 없음**: code-reviewer가 Critical 이슈를 보고하지 않음
2. **Gradient Flow 정상**: (ML 코드의 경우) gradient가 올바르게 흐름
3. **Edge Case 처리 완료**: 모든 엣지 케이스가 처리됨
4. **Production Ready 평가**: code-reviewer가 production-ready로 판정

## 사용 예시

### 예시 1: 새 기능 검증
```
사용자: "이 브랜치의 코드를 iterative-code-review로 검증해줘"

Claude:
1. git diff main...HEAD로 변경사항 확인
2. modular-code-architect로 초기 이슈 수정
3. code-reviewer로 검토
4. Critical 이슈 발견 시 수정 반복
5. Production-ready 판정 시 완료
```

### 예시 2: 특정 파일 개선
```
사용자: "/iterative-code-review models/trm_nm.py"

Claude:
1. 해당 파일 분석
2. 아키텍트로 개선점 구현
3. 리뷰어로 검증
4. 반복...
```

## Best Practices

1. **Todo 리스트 활용**: TodoWrite 도구로 진행 상황 추적
2. **커밋 분리**: 각 수정 사이클마다 의미 있는 커밋 생성
3. **점진적 개선**: 한 번에 모든 것을 수정하지 않고 단계적으로 진행
4. **문서화**: 수정 이유와 결정 사항을 커밋 메시지에 기록

## 주의사항

- main 브랜치에 직접 머지하지 않음 (별도 지시가 없는 한)
- 파괴적 git 명령어 사용 금지 (force push, hard reset 등)
- 테스트가 있는 경우 수정 후 테스트 실행 권장
