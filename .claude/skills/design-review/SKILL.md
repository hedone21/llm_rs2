---
name: design-review
description: 코드 구조를 분석하고 SOLID 원칙 준수 여부를 검토한다. Architect 에이전트가 사용.
allowed-tools: Read, Glob, Grep
argument-hint: "<module_or_file_path>"
---

# Design Review

지정된 모듈/파일의 구조를 분석하고 설계 품질을 검토한다.

## 검토 항목

### 1. SOLID 원칙
- **SRP**: 모듈/구조체가 단일 책임을 가지는가?
- **OCP**: 새 구현 추가 시 기존 코드 수정이 최소화되는가?
- **LSP**: 트레이트 구현체가 계약을 준수하는가?
- **ISP**: 클라이언트가 불필요한 메서드에 의존하지 않는가?
- **DIP**: 구체 타입이 아닌 트레이트에 의존하는가?

### 2. 테스트 용이성
- 의존성 주입이 가능한 구조인가?
- Mock/Stub으로 대체 가능한 트레이트 바운드인가?
- 부작용(side effect)이 격리되어 있는가?

### 3. 모듈 의존성
- 순환 의존이 없는가?
- 레이어 위반이 없는가? (core ← layers ← models ← bin)
- 불필요한 pub 노출이 없는가?

### 4. 프로젝트 특화 규칙
- `Backend` 트레이트 변경 시 CPU/OpenCL 양쪽 구현 확인
- `EvictionPolicy`/`CachePressureHandler` 트레이트 일관성
- KV cache 레이아웃(HeadMajor) 가정이 유지되는가?

## 보고서 형식

```markdown
## Design Review: [대상]

### 구조 요약
(현재 모듈/파일 구조)

### 발견 사항
| # | 원칙 | 심각도 | 설명 | 제안 |
|---|------|--------|------|------|

### 의존성 그래프
(해당 모듈의 주요 의존 관계)

### 변경 리스크
| # | 리스크 | 심각도 | 발생 가능성 | 완화 방안 |
|---|--------|--------|-------------|-----------|

### 권장 조치
(우선순위 순)
```

## 분석 대상 참고 경로

- 핵심 트레이트: `engine/src/core/backend.rs`, `engine/src/core/kv_cache.rs`
- 압력 파이프라인: `engine/src/core/pressure/`
- 모델: `engine/src/models/llama/`
- 레이어: `engine/src/layers/`
