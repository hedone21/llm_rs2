---
name: research
description: 논문, 기술 자료, 경쟁 프레임워크를 조사하고 프로젝트 적용 가능성을 평가한다. Researcher 에이전트가 사용.
allowed-tools: Read, Glob, Grep, WebSearch, WebFetch
argument-hint: "<topic or paper title>"
---

# Research

지정된 주제에 대해 논문/기술 자료를 조사하고 프로젝트 적용 가능성을 평가한다.

## 조사 절차

1. **키워드 검색**: 주제 관련 논문, 블로그, 레포지토리 검색
2. **핵심 메커니즘 분석**: 알고리즘/기법의 작동 원리 파악
3. **코드 매핑**: 기존 코드베이스에서 대응되는 모듈/트레이트 식별
4. **적용 가능성 평가**: 난이도, 예상 효과, 제약사항 분석
5. **보고서 작성**: 구조화된 형식으로 결과 반환

## 보고서 형식

```markdown
## 조사: [주제]

### 요약
(핵심 아이디어 1-3줄)

### 핵심 메커니즘
(알고리즘 작동 원리 — 수식/다이어그램 포함)

### 프로젝트 매핑
| 논문 개념 | llm.rs 대응 모듈 | 현재 상태 |
|-----------|------------------|-----------|

### 적용 가능성
- **난이도**: 낮음/중간/높음
- **예상 효과**: 성능/메모리/품질 개선 정도
- **제약사항**: 1B 모델, 2048 컨텍스트, ARM64 등에서의 한계
- **구현 방향**: 추가/수정할 모듈

### 참고 자료
(논문 링크, 레포지토리, 관련 이슈)
```

## 프로젝트 컨텍스트 (조사 시 참고)

- **모델**: Llama 3.2 1B (dim=2048, 32 Q-heads, 8 KV-heads, 16 layers, max_seq=2048)
- **타겟**: ARM64 Android, Adreno GPU (OpenCL)
- **KV cache**: HeadMajor, Q4_0/F16/F32, CachePressurePipeline
- **구현 완료 기법**: H2O, D2O, Sliding Window, Flash Attention, SnapKV(분석)
- **핵심 제약**: 1B 모델에서 attention score 분포가 편향적 (BOS 지배)

## 조사 우선순위 키워드

- KV cache: compression, quantization, eviction, paging
- Decoding: speculative, draft model, early exit
- Attention: PagedAttention, FlashAttention, linear attention
- On-device: memory-efficient inference, quantization-aware
- 경쟁 프레임워크: llama.cpp, MLC-LLM, ExecuTorch

## 기존 조사 기록 확인

조사 시작 전 memory에서 기존 분석 여부를 확인:
- `snapkv_analysis.md` — SnapKV 분석 완료
- `project_proxy_degradation.md` — QCF 시스템 설계
- MEMORY.md의 Research Notes 섹션
