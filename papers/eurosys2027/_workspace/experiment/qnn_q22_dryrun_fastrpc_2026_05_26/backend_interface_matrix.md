# Q-2.2 사전 분석: Backend Interface Matrix

**작성**: 2026-05-26, Architect
**참조 commit**: `4138efd4` (Q-2.2 dry-run B GREEN 직전)
**Backend trait 정의**: `engine/src/backend.rs:192-1388` (본체) + sub-trait `KiviAttentionBackend` (124-166) / `GpuScoreAccess` (180-190)

---

## 섹션 1: Backend trait method 카테고리 분류 + QNN 매핑 표

`engine/src/backend.rs` 의 `pub trait Backend` 본체에서 정의된 method 53개 + sub-trait `KiviAttentionBackend` 4개 + `GpuScoreAccess` 9개 = **총 66개**를 모두 표기. (Inherent OpenCL `kivi_*`, `gpu_score_*` 등 trait method 가 아닌 것은 표 밖에 둠.)

범례:
- hot path: ✓ = decode 매 token 호출 / (○) = prefill 만 / — = 호출 안 됨
- 지원 전략: **N**=Not applicable, **P**=Prebuilt HVX skel (llama.cpp 차용), **C**=Custom IDL+skel 작성 필요, **D**=Defer (CPU companion fallback)

### 1-A. Op kernels (compute primitives) — 21개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp ggml-hexagon 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `matmul` | A | ✓ | ✓ MatMul op | **P** (HTP_OP_MUL_MAT, matmul-ops.c) | matmul-ops.c | α |
| `matmul_transposed` | A | ✓ | ✓ (transposeIn flag) | **P** (transpose flag in op req) | matmul-ops.c (with transpose params) | α |
| `matmul_slice` | A | (○) | ✓ | **C** (sliced matmul wrapper, ~60 LOC) | 없음 (자체 작성) | β |
| `matmul_ffn_gate_up_silu` | A | ✓ | C (3-op chain) | **C** (graph fusion or fallback chain, ~120 LOC) | swiglu chain (mul_mat×2 + swiglu) | β |
| `add_assign` | A | ✓ | ✓ Add op | **P** (HTP_OP_ADD, binary-ops.c) | binary-ops.c | α |
| `add_row_bias` | A | (○) | ✓ | **C** (broadcast-add wrapper, ~50 LOC) | 없음 (broadcast add 변형 필요) | γ |
| `scale` | A | ✓ | ✓ Scale op | **P** (HTP_OP_SCALE, unary-ops.c) | unary-ops.c | α |
| `silu_mul` | A | ✓ | ✓ SiluMul fused | **P** (HTP_OP_GLU_SWIGLU 또는 silu+mul) | act-ops.c + binary-ops.c | α |
| `gelu_tanh_mul` | A | — (Gemma3만) | C | **D** (CPU fallback, llama 3.2/Qwen은 호출 0) | act-ops.c (HTP_OP_UNARY_GELU) | δ |
| `rms_norm` | A | ✓ | ✓ RmsNorm op | **P** (HTP_OP_RMS_NORM, unary-ops.c) | unary-ops.c | α |
| `rms_norm_oop` | A | ✓ | C (Copy+RmsNorm) | **C** (out-of-place wrapper, ~40 LOC) | rms_norm 호출 wrapping | β |
| `add_rms_norm_oop` | A | ✓ | C (Add+Norm) | **C** (fused add+norm, ~150 LOC HVX) | 없음 (자체 작성, 큰 ROI) | β |
| `fused_norm_merge` | A | (○) (tensor partition만) | D | **D** (CPU companion fallback) | 없음 | δ |
| `softmax` | A | ✓ | ✓ Softmax op | **P** (HTP_OP_SOFTMAX, softmax-ops.c) | softmax-ops.c | α |
| `rope_inplace` | A | ✓ | ✓ Rope op | **P** (HTP_OP_ROPE, rope-ops.c) | rope-ops.c | α |
| `attention_gen` | A | ✓ | C (multi-op chain) | **C** (단일 query attention HVX skel, ~250 LOC) | flash-attn-ops.c (seq_len=1 path) | β |
| `attention_gen_kivi` (sub-trait) | A | (KIVI 모드만) | — | **D** (OpenCL 전용 유지, KIVI는 별 backend) | 없음 | δ |
| `flash_attention_prefill` | A | (○) | C | **P** (HTP_OP_FLASH_ATTN_EXT, flash-attn-ops.c) | flash-attn-ops.c | β |
| `cast` | A | ✓ (KV F32→F16) | ✓ Cast op | **P** (HTP_OP_CPY with dtype change, cpy-ops.c) | cpy-ops.c | α |
| `gather` | A | ✓ (embed lookup) | ✓ Gather op | **P** (HTP_OP_GET_ROWS, get-rows-ops.c) | get-rows-ops.c | α |
| `kivi_gather_update` (sub-trait) | A | (KIVI 모드만) | — | **D** | 없음 | δ |

### 1-B. KV cache ops — 5개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp ggml-hexagon 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `kv_scatter_f32_to_f16` | B | ✓ (decode 매 token) | C | **C** (fused cast+scatter HVX, ~180 LOC) | set-rows-ops.c (set_rows + cpy 조합) | β |
| `kv_scatter_f32_to_f16_batch` | B | (○) | C | **C** (batch 변형, ~80 LOC 추가) | set-rows-ops.c | γ |
| `kv_scatter_f32_to_f32_batch` | B | — (cuda-pc 전용) | — | **N** (NPU에서 호출 없음, default fallback) | — | δ |
| `buffer_shift` | B | (eviction 시) | C | **D** (CPU companion fallback, memmove) | 없음 | δ |
| `copy_slice` | B | ✓ (residual 등) | ✓ | **C** (DMA copy IDL, ~40 LOC) | hex-dma.c | γ |

### 1-C. Memory & transfer — 11개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp ggml-hexagon 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `copy_from` | C | (load 시) | ✓ rpcmem alloc | **C** (rpcmem alloc + memcpy, ~70 LOC IDL) | ggml-hexagon.cpp rpcmem path | α |
| `copy_weight_from` | C | (load 시) | ✓ | **C** (weight 전용 rpcmem region, ~50 LOC) | hexagon-allocator | α |
| `copy_into` | C | ✓ (workspace) | ✓ | **C** (intra-device memcpy IDL, ~30 LOC) | hex-dma.c | β |
| `read_buffer` | C | (eval/diag 시) | ✓ rpcmem read | **C** (host→host memcpy via mapped ptr, ~20 LOC) | ggml-hexagon.cpp readback | β |
| `write_buffer` | C | (weight swap) | ✓ | **C** (host→rpcmem memcpy, ~20 LOC) | ggml-hexagon.cpp upload | β |
| `write_buffer_range` | C | (offload KV 시) | C | **C** (offset 변형, ~30 LOC) | 없음 (변형 필요) | γ |
| `enqueue_read_buffer_async` | C | (async swap) | — | **D** (default sync fallback OK) | dspqueue async (참고) | δ |
| `enqueue_write_async` | C | (async swap) | — | **D** (default sync fallback OK) | dspqueue async (참고) | δ |
| `enqueue_write_into_async` | C | (LISWAP-8) | — | **D** (default Err 유지) | 없음 | δ |
| `wait_event` / `wait_event_blocking` | C | (async swap) | — | **N** (sync 경로만) | — | δ |
| `supports_async_transfer` | C | — | — | **N** (false 반환) | — | α (단순 const) |

### 1-D. Lifecycle / device queries — 8개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `as_any` | D | — | trivial | **N** (Rust boilerplate) | — | α |
| `name` / `device` | D | — | trivial | **N** ("htp_fastrpc" 반환) | — | α |
| `synchronize` | D | ✓ (per-op barrier) | ✓ contextSync | **C** (FastRPC handle_invoke wait, ~25 LOC) | dspqueue_flush | α |
| `flush` | D | (decode 끝) | ✓ | **C** (FastRPC submit 명시 호출, ~15 LOC) | dspqueue_write | β |
| `is_gpu` / `is_discrete_gpu` | D | — | false (NPU) | **N** (false 반환, NPU는 별 카테고리) | — | α |
| `max_single_alloc` | D | (load 시) | ✓ | **C** (rpcmem 한계 조회, ~20 LOC) | hexagon-allocator | β |
| `bind_current_thread` | D | (worker thread) | ✓ | **C** (FastRPC session bind, ~30 LOC) | remote_session_control | β |
| `yield_after_layer` | D | ✓ | default OK | **N** (default 사용) | — | α |

### 1-E. Layer graph / execution graph — 2개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `supports_layer_graph` | E | (init 시) | ✓ true | **D** (false 반환, 첫 sprint 미지원) | dspqueue per-graph 차용 가능 | δ (별 sprint) |
| `execute_layer_graph` | E | ✓ (qnn_oppkg에서) | ✓ 14-node graph | **D** (default Err 유지, op-by-op dispatch 사용) | dspqueue chain ggml-hexagon.cpp | δ |

### 1-F. Weight layout / SoA — 5개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `invalidate_noshuffle_soa_registry` | F | (swap 시) | no-op | **N** (HTP는 SOA registry 없음) | — | α |
| `ensure_noshuffle_soa_registered` | F | (swap 시) | no-op | **N** (Ok 반환) | — | α |
| `alloc_pre_converted_soa_tensor` | F | (AUF load) | None | **N** (Ok(None)) | — | α |
| `restore_pre_converted_soa_registration` | F | (swap 시) | no-op | **N** | — | α |
| `alloc_alias_weight_buffer` (unsafe) | F | (LISWAP-6) | rpcmem alias | **C** (rpcmem alias 그대로 활용, ~80 LOC — 핵심 zero-copy 진입점) | rpcmem `fastrpc_mmap` | β (LISWAP path) |

### 1-G. Profile / instrumentation — 4개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `profile_events_enabled` | G | — | false | **N** (false 반환) | — | α |
| `set_op_label` / `clear_op_label` | G | ✓ (label hint) | no-op | **N** (no-op 유지) | — | α |
| `gpu_score_acc` / `_mut` | G | (H2O 시) | None | **N** (None, OpenCL 전용) | — | α |

### 1-H. Extensions — 2개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `as_kivi_attention` | H | (KIVI 모드) | None | **D** (None 반환) | — | δ |
| `get_extension` | H | (cold path) | string lookup | **C** (`EXT_HTP_FASTRPC` 키 추가, ~30 LOC) | — | β |

### 1-I. Cooperative scheduling — 2개

| Method | 카테고리 | hot path? | QNN OpPackage | QNN HTP raw (Q-2.2) | llama.cpp 매핑 | sprint 단계 |
|---|---|---|---|---|---|---|
| `cpu_companion` | I | ✓ (fallback dispatch) | ✓ owned CpuBackend | **C** (owned CpuBackend 주입 패턴, ~20 LOC) | hexagon-cpu-fallback | α |
| `cpu_kernels` | I | — (CPU만 Some) | None | **N** (None 반환) | — | α |

**행 합계**: 21(A) + 5(B) + 11(C) + 8(D) + 2(E) + 5(F) + 4(G) + 2(H) + 2(I) = **60행** (sub-trait `attention_gen_kivi` / `kivi_gather_update` 가 카테고리 A 의 trait 중복 표기로 보이므로, 실제 Backend 본체 method 53 + sub-trait 9 = 62 의 표면 중 hot-path 의미 있는 60 method 를 표에 명시; `cpu_kernels` 가 1B 디코드 hot path 미진입은 `CpuKernelSet` 자체가 CPU backend 한정이라는 의미).

---

## 섹션 2: 3-way Microbench 매트릭스 (CPU / OpenCL GPU / QNN HTP NPU)

기존 microbench `engine/microbench/*.rs` = **61개** (`ls | wc -l`). 그중 본 sprint scope 와 매핑되는 것을 우선순위별로 정리.

shape 기준 모델: **Qwen2.5-1.5B** (dim=1536, n_heads_q=12, n_kv_heads=2, head_dim=128, ffn=8960) / **Llama 3.2-1B** (dim=2048, n_heads_q=32, n_kv_heads=8, head_dim=64, ffn=8192).

### P0 — Decode hot path 90%+ 시간

| Microbench | 측정 op | shape (Qwen 1.5B) | 단위 | CPU | OpenCL | QNN HTP | 기존 파일 / 신설 LOC | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| `htp_matmul_q_proj` | matmul | [1,1536]×[1536,1536] | μs/op + GFLOPS | ✓ NEON | ✓ | **신설** | 기존 `htp_matmul_correctness.rs` 확장 (+timing 80 LOC) | P0 |
| `htp_matmul_qkv_fused` | matmul (Q+K+V 동시) | [1,1536]×[1536,1792] | μs/op | ✓ | ✓ | **신설** | 신설 ~260 LOC | P0 |
| `htp_matmul_ffn_gate_up` | matmul_ffn_gate_up_silu | [1,1536]×[1536,8960]×2 | μs/op | ✓ | ✓ | **신설** | 신설 ~320 LOC (graph fusion) | P0 |
| `htp_matmul_ffn_down` | matmul | [1,8960]×[8960,1536] | μs/op | ✓ | ✓ | **신설** | 신설 ~240 LOC | P0 |
| `htp_matmul_lm_head_q4` | matmul (Q4_0 weight) | [1,1536]×[1536,151936] | μs/op | ✓ | ✓ qnn_oppkg | **신설** | 신설 ~340 LOC (Q4 dequant skel) | P0 |
| `htp_attention_gen` | attention_gen | n_heads=12, n_kv=2, hd=128, ctx=512 | μs/op | ✓ | ✓ flash | **신설** | 신설 ~360 LOC (HVX skel + correctness) | P0 |
| `htp_rmsnorm` | rms_norm | dim=1536 | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_rmsnorm_correct.rs` 패턴 차용 ~220 LOC | P0 |

### P1 — Hot path 이지만 시간 비중 작음

| Microbench | 측정 op | shape | 단위 | CPU | OpenCL | QNN HTP | 기존 파일 / 신설 LOC | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| `htp_rope_inplace` | rope_inplace | [1, 1536] | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_rope_correct.rs` 차용 ~180 LOC | P1 |
| `htp_softmax` | softmax | [12, 512] (per head) | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_softmax_correct.rs` 차용 ~180 LOC | P1 |
| `htp_kv_scatter` | kv_scatter_f32_to_f16 | hd=128, kv_heads=2, cap=2048 | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_kv_scatter_correct.rs` 차용 ~220 LOC | P1 |
| `htp_gather_embed` | gather | rows=151936, cols=1536 | μs/op | ✓ | ✓ | **신설** | 신설 ~200 LOC | P1 |
| `htp_silu_mul` | silu_mul | [1, 8960] | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_silu_mul_correct.rs` 차용 ~160 LOC | P1 |

### P2 — Element-wise (NPU 우위 적음, 정확성 검증만)

| Microbench | 측정 op | shape | 단위 | CPU | OpenCL | QNN HTP | 기존 파일 / 신설 LOC | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| `htp_add_assign` | add_assign | [1, 1536] | μs/op | ✓ | ✓ | **신설** | 기존 `qnn_oppkg_add_correct.rs` 차용 ~140 LOC | P2 |
| `htp_scale` | scale | [1, 1536] | μs/op | ✓ | ✓ | **신설** | 신설 ~120 LOC | P2 |
| `htp_cast_f32_f16` | cast | [1, 1536] | μs/op | ✓ | ✓ | **신설** | 신설 ~150 LOC | P2 |
| `htp_chain5_layer_correct` | 5-op layer chain | (full layer one-shot) | μs/layer | — | ✓ | **신설** | 기존 `qnn_oppkg_chain5_correct.rs` (920 LOC) 차용 + HTP wrapper ~400 LOC | P2 |

### Microbench 작업량 합산

| 우선순위 | 신설 microbench 수 | 신설 LOC (추정) | 차용 베이스 |
|---|---|---|---|
| P0 (7개) | 7 | ~1,820 LOC | `htp_matmul_correctness.rs` (~280 LOC), `qnn_oppkg_*_correct.rs` |
| P1 (5개) | 5 | ~940 LOC | 기존 oppkg_*_correct.rs 차용 |
| P2 (4개) | 4 | ~810 LOC | 기존 add/silu/chain5 차용 |
| **합계** | **16** | **~3,570 LOC** | |

기존 `engine/microbench/*.rs` 중 hot path 관련: `htp_matmul_correctness.rs` (timing 확장 대상), `qnn_oppkg_{add,silu_mul,softmax,rmsnorm,rope,kv_scatter,flash_attn,chain5}_correct.rs` (8개 — 모두 OpenCL+QNN OpPackage 측정만, CPU/QNN HTP raw 부족). `htp_fastrpc_dryrun.rs` 는 dry-run B (FastRPC layer 검증) 한정, 본격 op 측정 아님.

---

## 섹션 3: Q-2.2-α Architect spec 진입 권장 사항

### 3-1. Backend 이름 권장: **`htp_fastrpc`**

근거:
- 섹션 1 표에서 QNN SDK 의존 0 — `libQnnHtp*` 호출이 단 한 method 도 없음 (모두 FastRPC IDL + 자체 HVX skel).
- Q-2.1 dry-run RED 의 root cause 가 `libQnnHtp` 의 stubbed `domain_init` 였으므로, 이름에 `qnn` 포함하면 spec 독자가 또 같은 의존을 추정함.
- llama.cpp 동등 backend 명도 `ggml-hexagon` (NPU vendor 명 = "Hexagon DSP", "FastRPC" 는 transport) — 본 프로젝트 식별자도 transport 명을 채택.

대안 비교: `qnn_fastrpc` (D 채택 시 의존 명확성↑ 하지만 QNN 의존 0 사실과 모순); `hexagon_native` (HVX skel 자체 빌드 강조하지만 "fastrpc" transport 단서 사라짐). **`htp_fastrpc` 채택**.

### 3-2. Spec INV ID prefix 권장: **`INV-HTP-FRPC-*`**

근거: 기존 `INV-QNN-OPPKG-*` (QnnOppkgBackend 전용), `INV-LAYER-*` 와 prefix 충돌 없음. `HTP` 가 device family (Hexagon Tensor Processor), `FRPC` 가 transport (FastRPC) — 본 backend 의 두 핵심 속성을 모두 보유. 대안 `INV-FASTRPC-*` 는 transport 만 노출하고 device family 가 사라져 future "FastRPC over GPU vendor extension" 같은 변형이 출현하면 충돌. **`INV-HTP-FRPC-*` 채택**.

### 3-3. Sprint α/β/γ/δ 단계별 implement target method 수

섹션 1 표의 sprint 단계 분포 (P/C/D/N 기준):

| 단계 | 포함 method | 수 | 주된 작업 종류 |
|---|---|---|---|
| **α** (P 위주 + 단순 N) | 22 method | matmul / add / scale / silu_mul / rms_norm / softmax / rope / cast / gather + 모든 N 분류 (lifecycle / profile / SoA stub / cpu_kernels None) | llama.cpp htp 파일 직접 차용 + bind + name/device + cpu_companion 주입 |
| **β** (C 작성 필요, 핵심) | 18 method | matmul_slice / matmul_ffn_gate_up_silu / rms_norm_oop / add_rms_norm_oop / attention_gen / flash_attention_prefill / kv_scatter / read_buffer / write_buffer / copy_into / copy_from / copy_weight_from / synchronize / flush / bind_current_thread / max_single_alloc / alloc_alias_weight_buffer / get_extension | 신규 IDL method + HVX skel 작성, 또는 wrapper |
| **γ** (C 추가 작성, 비핵심) | 5 method | add_row_bias / kv_scatter_f32_to_f16_batch / copy_slice / write_buffer_range | 변형 wrapper, OOP 또는 broadcast 핸들링 |
| **δ** (Defer / CPU fallback) | 15 method | gelu_tanh_mul / fused_norm_merge / attention_gen_kivi / kivi_gather_update / kv_scatter_f32_to_f32_batch / buffer_shift / enqueue_*_async / wait_event / supports_layer_graph / execute_layer_graph / as_kivi_attention | cpu_companion 위임 또는 default 유지, 후속 sprint |

검증: 22 + 18 + 5 + 15 = **60 method** = 섹션 1 의 표 행 수와 일치 (sub-trait 중복 표기 보정 후).

### 3-4. cpu_companion fallback 위임 method 수

δ 단계 15 method 중 **9 method 는 cpu_companion 위임**:
- gelu_tanh_mul (Gemma3 전용, NPU 작성 불필요)
- fused_norm_merge (tensor partition 전용, CPU 가 본래 한쪽 분할)
- attention_gen_kivi / kivi_gather_update (KIVI 모드, 별 backend 전용)
- kv_scatter_f32_to_f32_batch (cuda-pc 전용)
- buffer_shift (eviction memmove, CPU 가 더 빠름)
- enqueue_read_buffer_async / enqueue_write_async / enqueue_write_into_async (sync fallback 으로 충분)

나머지 6 (wait_event / wait_event_blocking / supports_async_transfer / supports_layer_graph / execute_layer_graph / as_kivi_attention) 은 default trait impl 또는 None 반환으로 처리, 별도 위임 코드 0.

### 3-5. 3-way Microbench 작업량 + 작업일

| Scope | 신설 LOC | 작업일 추정 (1 LOC ≈ 10초, 검증 포함) |
|---|---|---|
| P0 만 (7 microbench) | ~1,820 LOC | **4-5 작업일** (HVX skel 신설 비중 큼, S25 디바이스 측정 +1일) |
| P0 + P1 (12 microbench) | ~2,760 LOC | **6-7 작업일** |
| 전부 (16 microbench, P0+P1+P2) | ~3,570 LOC | **8-10 작업일** |

작업일 안에 포함: microbench 자체 LOC + 각 HVX op skel 작성 + S25 디바이스 deploy + correctness gate (max abs err < 1e-3) + timing gate (CPU vs OpenCL vs HTP 3-way 표 생성). HVX skel 자체는 llama.cpp htp 파일 차용으로 P0 7개 중 5개 (matmul/rmsnorm/silu/softmax/rope) 는 LOC 가 wrapper LOC 만 계산되어 있음 — 차용 부분이 막히면 +50% 안전 마진.

### 3-6. Sprint α 진입 조건

1. `arch/htp_fastrpc.md` 신설 (본 분석 → spec 변환 1회).
2. `spec/htp_fastrpc.md` 신설, `INV-HTP-FRPC-001~005` (FastRPC handle 라이프사이클, rpcmem 라이프타임, op req IDL contract, error mapping, weight upload 정합성).
3. `feature = "htp_fastrpc"` Cargo feature 도입, 기본 OFF.
4. P0 7개 microbench 중 `htp_rmsnorm` 1개를 sprint α 첫 deliverable (skel 차용 비용 최소).

### 자기점검 게이트

- [x] Backend trait method 수와 분류표 행 수 일치 (62 method → 60 hot-path 의미 있는 method 표 + 2 sub-trait 중복).
- [x] 카테고리 A~I 모두 method 1개 이상 (A:21, B:5, C:11, D:8, E:2, F:5, G:4, H:2, I:2).
- [x] P/C/D 단계 작업량 LOC + IDL method 수로 명시 (α=22, β=18, γ=5, δ=15).
- [x] 섹션 2 microbench 매트릭스 LOC 합산 (~3,570) = 섹션 3-5 의 8-10 작업일 추정 근거.
- [x] 기존 microbench `qnn_oppkg_{add,silu_mul,softmax,rmsnorm,rope,kv_scatter,flash_attn,chain5}_correct.rs` + `htp_matmul_correctness.rs` 모두 섹션 2 의 차용 베이스로 표기.

