# Qwen2.5-1.5B Q4_0 GPU Prefill SOA Repack — 실측 결과

**날짜**: 2026-04-20  
**디바이스**: Galaxy S25 (Snapdragon 8 Elite, Adreno 830), serial `R3CY408S5SB`  
**모델**: `qwen2.5-1.5b-q4_0.gguf` (131 token prompt / 128 token decode, 6T, resilience+manager)  
**비교**: 이 작업 전(AOS + byte-scalar 로드) vs 이 작업 후(SOA + uchar4 벡터 로드)

## 변경 요약
- `engine/kernels/mul_mm_q4_0_f32_l4_lm.cl`: llama.cpp 원본 시그니처 복원
  (`src0_q uchar4*` + `src0_d half*`, `uchar4 q = *qs;` aligned load)
- `engine/src/backend/opencl/mod.rs`:
  - `NoshuffleSoaEntry`에 `gemm_q_buf` / `gemm_d_buf` 필드 추가
  - `convert_q4_0_to_plain_soa()` 신규 추가 — `kernel_convert_block_q4_0` (cvt.cl 기존)
    을 이용해 AOS 버퍼에서 SOA 쌍을 한 번에 생성
  - GEMM dispatch에서 SOA 쌍을 lookup, 가용 시 새 2-버퍼 시그니처로 호출, 미가용 시 GEMV 폴백
- `engine/src/models/transformer.rs:prepare_noshuffle_buffers()`: plain SOA 쌍도
  함께 빌드하여 엔트리에 저장 (호환 유지 — 실패 시 None)

## 실측 결과 (3회 평균 ± std)

| 지표 | AOS baseline (master) | SOA patch | 개선 |
|------|----------------------|-----------|------|
| Prefill | 115.0 ± 0.4 t/s (1139 ± 3.6 ms) | 116.5 ± 0.1 t/s (1124 ± 1.0 ms) | **+1.3%** |
| Decode  | 27.83 ± 0.06 t/s | 27.87 ± 0.06 t/s | 노이즈 범위 |
| llama.cpp 참조 | 615 t/s prefill | — | 여전히 -81% |

원본 AOS 로그: `.agent/research/2026-04-20_qwen15b_host_vs_device_bench/llmrs_gpu_q4_0_run{1,2,3}.log`  
SOA 로그: `bench/llmrs_gpu_q4_0_SOA_run{1,2,3}.log`  
RUST_LOG=info: `bench/llmrs_gpu_rustlog.log` — GEMM dispatch가 `(SOA uchar4)` 경로로 들어감을 확인

## 정합성 검증
- 출력 텍스트: 정상 (Qwen2 대화체 문장 완성, "Paris" 등 greedy 결과 정확)
- 호스트 `cargo test --workspace`: 기존 스킵 목록 + 사전 존재 panic 테스트
  (`test_f32_to_f16_roundtrip`) 제외 후 모두 통과
- `cargo fmt`: clean
- `cargo clippy --workspace -- -D warnings`: 이 패치와 무관한 **master 사전 존재 에러 6건**
  (`plan.rs`의 `manual_is_multiple_of`, `transformer.rs`의 `collapsible_if`, manager
  `compute.rs`의 `collapsible_if`, `forward.rs`의 `type_complexity`).
  stash 후 master에서 동일 에러 재현 확인됨.
- Tier2 `test_backend --backends opencl,scalar`: OpenCL 쪽 모든 MatMul FAIL — 그러나
  master에서도 **동일하게 FAIL** (stash 후 재현 확인). 사전 존재 문제이며 이 패치와 무관.

## 결론 — 가설 기각

지시사항의 가설 ("AOS의 byte-scalar 로드 vs SOA uchar4 aligned 로드가 5.3x 병목의
주원인") 은 **실측으로 기각**됨:

- SOA 경로가 실제로 dispatch되고 있음 (RUST_LOG=info로 확인)
- prefill GEMM 이득은 +1.3%에 불과 — llama.cpp의 615 t/s (5.3x) 격차를 설명하지 못함
- **Q4_0 prefill 병목은 이 커널의 로드 경로가 아님**

가능한 실제 병목 후보 (후속 조사 필요):
1. `matmul_q4_0` dispatch 조건 (`m >= 32` 게이트) — 실제로 GEMM이 타는지 재확인은 OK
2. flash attention의 Q4_0 경로 (KV 디퀀트) — Qwen2 layers=28, head_dim=128,
   `[Prefill] flash_attn dispatch` 로그에서 F16 KV로 dispatch — GEMM 외 구간 영향
3. Norm/residual/softmax 등 지원 OP의 GPU dispatch 오버헤드 — llama.cpp는 다수의
   OP를 fused 커널로 처리
4. Workspace / command queue flush 전략 — llama.cpp는 plan-level batching이 다름
5. `plan.rs`의 scatter/gather 커널 (`kernel_kv_scatter_f32_to_f16` 등) — 이 작업 범위 밖

Adreno register 사용량 측정 (`clGetKernelWorkGroupInfo`)은 코드 추가가 필요해
이번 범위에서는 수행하지 못함. 단 커널 자체는 llama.cpp 원본과 동일하므로 
llama.cpp 대비 추가 spill 발생 가능성은 낮음 (구조 동일).

## 미해결 / 후속 과제
- **5x 격차의 실제 원인 규명 프로파일 작업** (flash_attn + per-op breakdown)
- GEMV Q4_0 SOA 통일 (현재 decode는 여전히 noshuffle SOA + image 경로, 별개)
- F16 GEMM (-37% 격차) 동일 분석 — 이 경로는 이미 parity 커널이라 원인이 더 좁음
- Tier2 OpenCL MatMul FAIL (master HEAD 사전 존재) — 별도 조사 이슈로 분리 필요

## 커밋 여부
지시사항 "실패/중단 시 커밋 없이 보고만"에 따라 **커밋 보류**.
단, 변경된 코드는 회귀 없이 +1.3% 이득 + llama.cpp 구조 parity 확보라는 정성적
가치는 있음. 최종 승인은 오케스트레이터 판단에 위임.
