# llm.rs2 vs llama.cpp 성능 비교 리포트

**날짜**: 2026-03-13
**디바이스**: Samsung Galaxy S24 (SM-S931N), Snapdragon 8 Gen 3, Adreno 830
**모델**: Llama 3.2 1B (HuggingFace Safetensors / GGUF F16)
**프롬프트**: "tell me short story" (5 tokens prefill)
**KV cache**: F16, HeadMajor layout

## 1. 벤치마크 결과

### Decode 128 토큰

| 시스템 | Backend | Weight | TTFT (ms) | TBT (ms) | tok/s | vs llama.cpp CPU |
|--------|---------|--------|-----------|----------|-------|-----------------|
| llama.cpp | CPU (-ngl 0) | F16 | 92 | 41.0 | **24.4** | baseline |
| llama.cpp | GPU (-ngl 99) | F16 | 377 | 44.1 | **22.7** | -7% |
| llm.rs2 | CPU | Q4 | 180 | 57.0 | **17.5** | -28% |
| llm.rs2 | CPU | F16 | 288 | 65.9 | **15.2** | -38% |
| **llm.rs2** | **OpenCL** | **Q4** | **163** | **33.3** | **30.1** | **+23%** |
| llm.rs2 | OpenCL | F16 | 448 | 584.7 | 1.7 | -93% |

### Decode 256 토큰

| 시스템 | Backend | Weight | TBT (ms) | tok/s | vs llama.cpp CPU |
|--------|---------|--------|----------|-------|-----------------|
| llama.cpp | CPU (-ngl 0) | F16 | 42.4 | **23.6** | baseline |
| llama.cpp | GPU (-ngl 99) | F16 | 44.6 | **22.4** | -5% |
| llm.rs2 | CPU | Q4 | 67.1 | **14.9** | -37% |
| **llm.rs2** | **OpenCL** | **Q4** | **35.9** | **27.8** | **+18%** |

### 측정 조건
- 각 실행 전 10-15초 쿨다운 (시작 온도 ~40°C)
- llama.cpp: `--temp 0` (greedy), llm.rs2: `--temp 0.8 --top-p 0.9`
- llama.cpp 모델: `Llama-3.2-1B-Instruct-f16.gguf`
- llm.rs2 모델: HuggingFace Safetensors (load-time Q4_0 quantization or F16)

## 2. 분석

### 2.1 llm.rs2 OpenCL+Q4가 최고 성능인 이유

`mul_mv_q4_0_f32.cl` 커널이 Adreno 830에 고도로 최적화:
- **서브그룹 dispatch**: 64-thread wavefront, `N_DST=4` rows/subgroup
- **이미지 버퍼**: `read_imageh()` — Adreno 텍스처 캐시 활용 (~240 GB/s)
- **인라인 양자화 해제**: shift-mask-scale, 별도 스테이징 없음
- **레지스터 블로킹**: `half8` 누적기 4개, 배리어 0개

llama.cpp GPU는 범용 OpenCL이라 1B 소형 모델에서 오버헤드 > 이득.

### 2.2 llm.rs2 CPU가 느린 이유 (-28%)

llama.cpp의 NEON Q4×Q8 dot product 대비 주요 병목:

| 병목 | 영향 | 설명 |
|------|------|------|
| 수평 합산 (`vaddlvq_s16`) | ~30% | 블록당 4회, 각 8-10 사이클 |
| 활성화 양자화 (A→Q8_0) | ~23% | 12 사이클/32 floats, 전체의 ~23% |
| 스칼라 스케일 곱셈 | ~10% | `d * isum as f32` 직렬 처리 |
| 블록 배칭 부족 | ~10% | 반복당 64값 vs llama.cpp 128+ |
| Rayon 태스크 오버헤드 | ~5-8% | 청크 디스패치 비용 |

### 2.3 OpenCL F16 matmul이 극도로 느린 이유 (1.7 tok/s)

`mul_mat_f16_f32.cl`은 범용 tiled GEMM 템플릿:
- Adreno 서브그룹 미사용 (선언만 하고 실제 로직 없음)
- 로컬 메모리 3KB + 배리어 2회/K-tile
- 스칼라 F16→F32 변환 (하드웨어 가속 없음, 10+ cycles/convert)
- 2D 워크그룹 (16x8) — Adreno 선호 패턴 아님
- 레지스터 압박 (80 floats/thread) → 낮은 점유율

### 2.4 Prefill 속도 차이

| 시스템 | ms/token (prefill) |
|--------|-------------------|
| llama.cpp CPU | 6.6 |
| llm.rs2 CPU Q4 | 36.0 |
| llm.rs2 OpenCL Q4 | 32.7 |

llama.cpp 대비 5.5배 느림. Batch matmul 최적화 부재가 주 원인.

## 3. 성능 개선 계획

### Phase 1: CPU Q4 Dot Product 최적화 (목표: 17.5 → 24+ tok/s)

**P1-1. `vec_dot_q4_0_q8_0` NEON 재작성** (예상 효과: +30-40%)
- 현재: `vaddlvq_s16()` 4회/블록 (수평 합산 병목)
- 개선: llama.cpp 패턴 — `vpaddlq_s16` + `vpadal_s16` 누적기 체인
- 4블록 배칭 (128값/반복) + 다중 누적기로 ILP 극대화
- `sdot` (dotprod) 경로도 동일하게 최적화

**P1-2. 활성화 양자화 최적화** (예상 효과: +10-15%)
- 현재: 전체 activation row를 Q8_0으로 양자화 후 dot product
- 개선: 양자화와 dot product 융합 (fused quantize-dot)
- 또는: NEON `abs → max → scale → cvt` 파이프라인 4x 언롤

**P1-3. Matmul 병렬화 개선** (예상 효과: +5-8%)
- decode (M=1): N 차원 병렬 → Rayon 대신 직접 스레드 분할
- 청크 크기 최적화: L2 캐시 적합 크기로 조정

### Phase 2: OpenCL F16 Matmul 커널 재작성 (목표: 1.7 → 25+ tok/s)

**P2-1. Adreno 특화 F16 커널 작성** (`mul_mv_f16_f32_adreno.cl`)
- Q4_0 커널(`mul_mv_q4_0_f32.cl`) 구조를 기반으로 F16 버전 작성
- 서브그룹 dispatch (64-thread, `N_DST=4`)
- 이미지 버퍼 `read_imageh()` 활용
- 배리어 제거, 레지스터 블로킹

**P2-2. F16→F32 변환 최적화**
- `vload_half4()` + 벡터 FMA 사용
- 또는 F16 누적 유지 후 최종 변환만 수행

**P2-3. Dispatch 코드 수정** (`mod.rs`)
- 1D 워크그룹 (64 threads)
- `matmul_f16()` 함수에서 새 커널 호출

### Phase 3: Prefill Batch Matmul 최적화 (목표: 36 → 10 ms/tok)

**P3-1. CPU Batch Matmul 타일링**
- 현재: M×K 전체를 한 번에 처리
- 개선: L1/L2 캐시에 맞는 M×K 타일 블로킹
- 다중 output row 동시 처리 (M=128 → 4-8 row 배치)

**P3-2. OpenCL Prefill Matmul**
- 배치 처리에 특화된 GEMM 커널 (현재 decode용 GEMV만 최적화됨)
- M>1일 때 별도 dispatch 경로

## 4. 우선순위 및 예상 일정

| Phase | 작업 | 예상 효과 | 의존성 |
|-------|------|----------|--------|
| **P1-1** | NEON dot product 재작성 | 17.5 → 22-24 tok/s | 없음 |
| **P1-2** | 양자화 최적화 | +1-2 tok/s | P1-1 이후 |
| **P1-3** | 병렬화 개선 | +0.5-1 tok/s | 독립 |
| **P2-1** | Adreno F16 커널 | 1.7 → 25+ tok/s | 없음 |
| **P2-2** | F16 변환 최적화 | P2-1에 포함 | P2-1 |
| **P2-3** | Dispatch 수정 | P2-1에 포함 | P2-1 |
| **P3-1** | CPU batch tiling | prefill 3-5x 개선 | P1-1 이후 |
| **P3-2** | GPU batch GEMM | prefill 5-10x 개선 | P2-1 이후 |

## 5. 성공 기준

| 지표 | 현재 | 목표 | llama.cpp 대비 |
|------|------|------|---------------|
| CPU Q4 decode tok/s | 17.5 | **24+** | 100%+ |
| OpenCL Q4 decode tok/s | 30.1 | **30+** | 유지 |
| OpenCL F16 decode tok/s | 1.7 | **25+** | 100%+ |
| CPU prefill ms/tok | 36.0 | **10** | 65% |
| OpenCL prefill ms/tok | 32.7 | **5** | 75% |
