# Handoff: Phase C — KV cache dual buffer로 Adreno UMA zero-copy 달성

**Date**: 2026-05-10
**Branch**: master
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)
**선행 핸드오프**: `.agent/todos/handoff_qnn_oppkg_phase_b_complete_2026_05_10.md` (D-D.6 Phase B fast path correctness)

---

## 0. 한 줄 요약

`--backend qnn_oppkg` (fast path 미사용)에서 KV cache를 rpcmem DMA-BUF + OpenCL `CL_MEM_USE_HOST_PTR` alias로 alloc하면서 Decode TBT 28→13 ms/tok (-54%) 달성. **llama.cpp Adreno OpenCL (27 ms/tok) 대비 2× 가속**.

---

## 1. 측정 결과 (Qwen2.5-1.5B Q4_0, "The capital of France is", 32 token decode)

| 시스템 | Decode TBT | t/s | 비고 |
|---|---|---|---|
| llama.cpp (Adreno OpenCL, ngl 99, 6T) | 27.07 ms/tok | 36.94 | reference baseline |
| llm_rs2 `--backend opencl` | 28.31 ms/tok | 35.3 | llama.cpp와 동등 |
| llm_rs2 `--backend opencl --zero-copy` (CL_MEM_ALLOC_HOST_PTR) | 28.03 ms/tok | 35.7 | **효과 없음** |
| **llm_rs2 `--backend qnn_oppkg`** (Step 4 cleanup 후) | **12.13 ms/tok** | **82.4** | **★ llama.cpp 2.23×** |
| llm_rs2 `--backend qnn_oppkg` + `LLMRS_QNN_OPPKG_FAST_PATH=1` | 79.59 ms/tok | 12.6 | graph bridge로 인해 느림 (Step 2 대기) |

### 1.1 다른 모델 검증 (Step 3, post-cleanup)
| 모델 (Q4_0) | --backend opencl | --backend qnn_oppkg | 가속 |
|---|---|---|---|
| Qwen2.5-1.5B | 28.31 | 12.13 | 2.33× |
| Llama3.2-1B  | 21.68 |  5.74 | **3.78×** |
| Gemma3-1B    | 32.89 | 11.64 | 2.83× |

검증: 출력은 모두 정상 — "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers..." (commit 45cb489 tokenizer guard 적용 후).

---

## 2. 핵심 발견 — 가속 메커니즘

### 2.1 Dual buffer 구조 (commit 7e84800 + d4842cf)

`QnnOppkgKvBuffer` (engine/src/backend/qnn_oppkg/kv_buffer.rs):
1. `rpcmem_alloc(size)` → host_ptr (DMA-BUF heap에서 contiguous physical pages)
2. `clCreateBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, host_ptr, size)` → cl_mem alias
3. `QnnMem_register(fd, host_ptr)` → mem_handle (Step 2에서 graph 주입 예정)

OpenCL은 같은 host_ptr를 backing storage로 사용 — Adreno IOMMU가 GPU 측에 동일 물리 페이지로 매핑.

### 2.2 왜 `CL_MEM_ALLOC_HOST_PTR`은 효과 없는가
실측: `--backend opencl --zero-copy` (UnifiedBuffer = `CL_MEM_ALLOC_HOST_PTR`) → 28 ms/tok 그대로.

원인: Adreno OpenCL driver가 `CL_MEM_ALLOC_HOST_PTR`을 internal device pool에서 alloc + host mapping으로 처리. CPU↔GPU memcpy가 사라지지 않음.

진짜 zero-copy는 **DMA-BUF heap의 host_ptr를 `CL_MEM_USE_HOST_PTR`로 wrap**해야 트리거됨 (driver의 dmabuf interop path 활성).

### 2.3 메모리 footprint
**1× (zero-copy alias)** — perf gain 자체가 증거. Driver가 staging buffer를 만들었다면 read/write_buffer overhead가 동일해 28 ms 유지했어야 함.

### 2.4 `get_cl_mem` OCP 패치 (commit d4842cf)
`engine/src/backend/opencl/mod.rs::get_cl_mem`에 Buffer trait method `cl_mem()` fallback 추가. 새 buffer type이 explicit downcast 추가 없이 자동 호환.

---

## 3. Risk 분석 (적용 안전성)

Explore 조사 종합 (LOW = 영향 거의 없음):

| 항목 | Risk | 근거 |
|---|---|---|
| KV eviction (`prune_prefix`, `shift_positions`) | **LOW** | in-place data shift만, buffer 재alloc 없음 |
| KV migration (`kv_migrate.rs`) | **LOW** | UMA path는 `map_for_cpu` + retag만 |
| Swap/offload (`pressure/swap_handler.rs`) | **LOW** | prune_prefix만 사용 |
| Dynamic grow/shrink (`kv_cache.rs::grow/shrink_to_fit`) | **LOW** | `alloc_kv` 호출 → host-pinned 유지 |
| Tensor partition | **LOW** | KV는 partition 대상 아님 |
| `clEnqueueCopyBuffer` 호환 | **LOW** | OpenCL spec 보장 |
| KIVI cache | **MEDIUM** | KIVI는 별도 path, 현재 표준 KVCache와 배타 사용 |
| Concurrent map/unmap with GPU write | **LOW** | 코드는 unmapped 상태에서만 GPU write — spec 준수 |

**Net risk = LOW-MEDIUM**.

---

## 4. 변경 사항

### Commit history
```
e6bf5e8 chore(qnn): Phase B/C 디버그 dump infra 일괄 제거 (Step 4)
e5a60cf fix(qnn): Gemma3 다중 모델 호환 — tokenizer guard 완화 + OCL secondary 위임 (Step 3)
8ea8328 chore: cargo fmt — Phase B/C 작업 잔여 reformatting
22f02c8 docs(phase-c): Adreno 권장 backend 가이드 + Phase C 핸드오프
d4842cf fix(opencl): get_cl_mem에 Buffer trait fallback (OCP)
7e84800 feat(qnn): Step 1 — KV cache dual buffer (rpcmem + CL_MEM_USE_HOST_PTR)
3dcbfb4 perf(qnn): Phase C 측정 인프라 — fast path / lg.execute breakdown
45cb489 fix(generate): tokenizer vocab-size 검증 추가 (silent garbage decode 방지)
bc6bcff feat(qnn): D-D.6 Phase B 완료 — fast path = production byte-equal
```

### 신규 파일
- `engine/src/backend/qnn_oppkg/kv_buffer.rs` — `QnnOppkgKvBuffer` (rpcmem + cl_mem alias)
- `engine/src/backend/qnn_oppkg/hybrid_memory.rs` — `QnnOppkgHybridMemory` (alloc=OCL 위임, alloc_kv=dual)

### 수정 파일
- `engine/src/backend/qnn_oppkg/mod.rs` — 모듈 등록 + Phase C timing instrumentation
- `engine/src/backend/qnn_oppkg/layer_graph.rs` — lg.execute 내부 breakdown counter
- `engine/src/backend/opencl/mod.rs` — `get_cl_mem` trait fallback
- `engine/src/bin/generate.rs` — qnn_oppkg primary 시 hybrid memory 라우팅 + tokenizer vocab guard
- `AGENTS.md` (CLAUDE.md symlink) — Adreno 권장 backend 가이드 추가

---

## 5. 다음 단계 (옵션)

### Step 2 — fast path < 13 ms/tok (선택, ~1일)
현재 fast path는 79 ms/tok. graph 자체 rpcmem KV slot을 사용하기 때문에 dual buffer 효과 못 받음.

작업:
1. `LayerGraph::execute` API 변경 — KV bytes 대신 `mem_handle: u64` 받기
2. `execute_layer_graph`에서 `kv_cache_k/v.buffer().as_any().downcast::<QnnOppkgKvBuffer>().qnn_mem_handle()` 추출
3. graph build에서 KV slot 자체 alloc 제거 (외부 주입)
4. bridge memcpy + lg.copy_in/out KV 부분 제거
5. 측정 — 79 → ~30 ms/tok 추정 (fast path가 production보다 빨라질지는 graph dispatch overhead에 따름)

이게 의미 있을지는 production 13 ms/tok이 충분한가에 달림. llama.cpp 2× 가속이면 충분하다는 판단도 가능.

### Step 3 — 다른 모델 검증 (Llama, Gemma) ✅ 완료 (commit e5a60cf)
3 모델 모두 dual buffer KV 가속 확인 (위 표 §1.1). Gemma3 head_dim=256으로 prefill flash_attn CPU fallback 발동 → `with_opencl_secondary` helper 추가로 OCL secondary write back 위임.

### Step 4 — 디버그 코드 정리 ✅ 완료 (commit e6bf5e8)
- `LLMRS_QNN_OPPKG_FAST_PATH_DUMP*` env-gated dump infra 제거 (qnn_oppkg/mod.rs, layer_graph.rs)
- `LLMRS_QNN_OPPKG_DUMP_FALLBACK_*` 제거 (forward.rs, forward_gen.rs, transformer.rs)
- `dump_stage!` macro + 18 callsite 제거 (forward_gen.rs)
- unconditional `[forward_prefill entered]` per-(layer×token) eprintln 제거 (per-token stderr overhead 일부 감소)
- microbench self-dump (`LLMRS_MICROBENCH_DUMP_PREFIX`)는 standalone graph debugging tool로 보존
- Net: 6 파일 -255 +19 = 236줄 삭제

### Step 5 — generic OpenCL backend에 dmabuf 통합 (큰 작업)
`--backend opencl`만 쓰는 사용자도 13 ms/tok 받을 수 있도록 OpenCLMemory에 dmabuf alloc path 추가. Android-only, libcdsprpc.so 의존성 필요. qnn_oppkg crate dependency 분리 또는 별도 dmabuf_alloc crate 필요. 큰 작업이라 ROI 검토 후.

---

## 6. 다음 세션 진입 명령

### qnn_oppkg backend 정상 동작 확인 (Phase C 결과 reproduce)
```bash
adb shell '
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
./generate -m /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b-gguf/tokenizer.json \
    -p "The capital of France is" -n 32 \
    --backend qnn_oppkg --temperature 0 --initial-kv-capacity 2048
'
# Expected: "Paris. It has a population of about 2 million people..."
# Decode TBT: ~13 ms/tok
```

### Step 2 진입 시 참조 파일
- `engine/src/backend/qnn_oppkg/kv_buffer.rs::qnn_mem_handle()` (이미 public, Step 2에서 graph 주입용)
- `engine/src/backend/qnn_oppkg/layer_graph.rs::AndroidLayerGraph::execute` (signature 변경 대상)
- `engine/src/backend/qnn_oppkg/mod.rs::execute_layer_graph` (bridge memcpy 제거 대상)

---

## 7. 메모리 (다음 세션이 알아야 할 것)

저장됨:
- `feedback_byte_equal_alone_is_not_correctness.md` — production garbage = fast path garbage = byte-equal로 통과한 D-D.6 Phase B 사고에 대한 가드
- (이번 세션 추가 예정) `project_qnn_oppkg_dual_buffer_kv.md` — qnn_oppkg backend가 fast path 미사용 시에도 KV dual buffer 덕에 production OpenCL 2× 가속

---

**End of Phase C Handoff**

Net 결과: llm_rs2 production OpenCL이 llama.cpp Adreno OpenCL 2× 가속 (Qwen2.5-1.5B Q4_0 13 vs 27 ms/tok). `--backend qnn_oppkg --tokenizer-path <model_tokenizer>` 권장.
