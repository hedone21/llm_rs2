# Phase 5 — clEnqueueMigrateMemObjects Prewarm **NEGATIVE**

## Setup
- Device: Galaxy S25 (Adreno 830)
- Buffer: 25MB ALLOC_HOST_PTR (fresh per run)
- Path A (cold): create buffer → first write_buffer (timed)
- Path B (prewarm): create buffer → migrate(CONTENT_UNDEFINED) + finish → first write_buffer (timed)
- n=60 (30 per path interleaved)

## 결과 (n=30/path)

| Path | mean | median | σ/mean | vs Path A |
|------|-----:|-------:|-------:|----------:|
| **Cold first-write** | 7.77 ms | 7.15 ms | 13.4% | reference |
| **Prewarm + first-write** | 7.68 ms | 7.12 ms | 13.1% | -1.2% (noise) |

## 핵심 분석

### 결과 해석
- 차이 -1.2%는 σ/mean=13%에 묻혀 noise
- Adreno driver는 `clEnqueueMigrateMemObjects(CONTENT_UNDEFINED)` hint를 무시
- migrate API는 spec compliant (call returns CL_SUCCESS) but no effect on subsequent access cost

### 흥미로운 부수 데이터
- 25MB cold first-write = 7.77ms (warm은 ~1ms — Phase 4)
- **First-touch penalty ~ 7ms per 25MB buffer**
- 600MB swap = 25 × cold buffers = 25 × 7ms = 175ms 추정
- 하지만 production에선 buffer pool로 첫 swap 후 warm 상태 유지

### Production 적용 평가
- prewarm 효과 noise level → 통합 가치 없음
- 단, **buffer pool 재사용은 매우 중요** (cold → warm 차이 8x)

## 결론
**Phase 5 SKIP integration**. migrate hint는 Adreno에서 no-op.

부수 finding: **buffer pool 재사용이 first-touch latency 우회의 유일한 방법**. LISWAP-3 zero-copy pool은 이미 이 원칙을 따른다 — 추가 작업 불필요.

---
2026-05-09 (Phase 5 완료)
