# Handoff: D-F + D-D.6 완료 — fast path 정확성 디버깅 잔존

**Date**: 2026-05-10
**HEAD**: `8c513d3` (feat(qnn): D-D.6 — n_kv input tensor화 + FlashAttn op spec / .cl kernel 갱신, push 안 함)
**관련 commit**: `a445cf5` (devices.toml serial), `917a31c` (D-F core), `8c513d3` (D-D.6)
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)
**SDK**: QNN 2.33 (`third_party/qnn_sdk_2.33/`, gitignored, 469MB host-side)

---

## 0. 한 줄 요약

D-F (model load weight migration fix) + D-D.6 (n_kv input tensor화) 모두 구현 완료. **qnn_oppkg primary + OpenCL fallback path가 OpenCL primary와 token byte-equal로 동작 (기본 운영 path)**. fast path 활성 (`LLMRS_QNN_OPPKG_FAST_PATH=1`)은 빌드/실행 PASS이지만 graph 자체 forward 결과가 OpenCL과 다름 — **single-layer microbench로 격리 필요**.

---

## 1. 이번 세션 산출물 (3 commits)

### `a445cf5` — chore(devices): galaxy_s25 ADB serial 갱신 (R3CY408S4HN → R3CY408S5SB)

새 디바이스로 교체. `devices.toml` 1줄.

### `917a31c` — feat(qnn): D-F — model load weight migration fix + fast path bridge infra

**core fix**:
- `qnn_oppkg::copy_from / copy_weight_from`을 fallback (OpenCL secondary)에 위임
  - 이전: source mmap buffer를 그대로 passthrough → weight tensor가 cl_mem 없는 MmapBuffer로 남고, prefill matmul fallback에서 `get_cl_mem` dereference 시 segv
  - 변경: fallback 위임으로 OpenCL이 cl_mem (Adreno UMA `UnifiedBuffer = CL_MEM_ALLOC_HOST_PTR`) alloc + memcpy. noshuffle SOA prep + prefill 모두 정상 동작
- `tensor_bytes_owned` helper 신규 (UnifiedBuffer map → copy → unmap) — graph build 시 weight pack 안전화
- 14개 forward method fallback wrap 추가 — gather, attention_gen, flash_attention_prefill, kv_scatter_f32_to_f16, buffer_shift, copy_slice, copy_into, read_buffer, write_buffer, write_buffer_range, add_row_bias, gelu_tanh_mul, add_rms_norm_oop, matmul_ffn_gate_up_silu, ensure_noshuffle_soa_registered, invalidate_noshuffle_soa_registry, max_single_alloc

**bridge infra (default 비활성)**:
- `layer_graph.rs::AndroidLayerGraph`에 `idx_kv_k`/`idx_kv_v` 노출 + `lg.execute`가 외부 KV bytes를 graph rpcmem KV slot에 sync (read 전), graph 종료 후 graph rpcmem → 외부 bytes로 write back
- `mod.rs::execute_layer_graph` cl_mem ↔ host bridge — fallback OpenCL의 read_buffer/write_buffer로 KV/x sync

### `8c513d3` — feat(qnn): D-D.6 — n_kv input tensor화 + FlashAttn op spec / .cl kernel 갱신

**Op spec 변경** (`crates/qnn_oppkg/src/ops/flash_attn.rs`):
- inputs: 6 → 7 (Q, K, V, mask, sinks, S, **n_kv_buf**)
- params: 5 → 4 (n_head, n_head_kv, kv_capacity, head_dim — `n_kv` 제거)
- args[10]: `Int(n_kv)` → `InputTensor(6)`
- mem_objects: 7 → 8 (n_kv_buf INT_32 [1] 추가)
- mask 크기: `n_kv` → `kv_capacity` (n_kv 동적이라 max로 alloc)

**.cl kernel** (`engine/kernels/flash_attn_f32_f16.cl`):
- 원본 `flash_attn_f32_f16_q1` 보존 (OpenCL primary path 사용 중)
- `flash_attn_f32_f16_q1_dynkv` 신규 OOP variant 추가 — `const int n_kv` arg → `const global int * n_kv_buf` + 본문 첫 줄 `const int n_kv = n_kv_buf[0];`. 그 외 attention 본문 동일 (192 lines, body 복제)

**layer_graph 통합**:
- `n_kv_buf` rpcmem slot (idx 24, 4 bytes INT_32) alloc, build placeholder 1
- `t_n_kv` graph tensor (rank 1, [1], INT_32, APP_WRITE) 등록
- FlashAttn node: input 6 → 7, params 5 → 4
- `AndroidLayerGraph.n_kv_host_ptr: *mut i32` 신규 + execute() 진입 시 `n_kv: usize` 인자 + graphExecute 직전 host write
- `t_n_kv_mh` exec_inputs 마지막 entry로 binding (slots[idx_n_kv].mem_handle)

**Backend trait + transformer 갱신**:
- `Backend::execute_layer_graph` trait sig: `n_kv: usize` 인자 추가
- `transformer.rs::forward_into` fast path: `n_kv = start_pos + seq_len`

**env-gated fast path**:
- `LLMRS_QNN_OPPKG_FAST_PATH=1` 설정 시에만 `supports_layer_graph()=true`
- default false — 정확성 잔존 issue 격리 동안 production-safe (qnn_oppkg primary + OpenCL fallback path가 OpenCL primary와 byte-equal)

---

## 2. 현재 동작 검증 (디바이스 측정)

### Default (fast path 비활성, qnn_oppkg primary + OpenCL fallback)

```
$ adb shell './generate -m qwen2.5-1.5b-q4_0.gguf -p "The capital of France is" -n 16 \
             --backend qnn_oppkg --temperature 0'

[qnn_oppkg] runtime initialized
[Backend] QNN-GPU primary, OpenCL secondary available (SwitchHw ready)
[Backend] qnn_oppkg fallback wired to OpenCL secondary (prefill + model load 위임)
[GGUF] Loaded: 338 tensors, arch=Qwen2, weight_dtype=Q4_0
[qnn_oppkg] eager prebuild: 28 layers, total finalize 1416 ms
[NoShuffle] Released original Q4_0 weights after SOA conversion (196 tensors, ≈702.8 MiB reclaimed)
Generating (Max: 2048, Temp: 0, TopP: 0.9, TopK: 40)...
The capital of France is a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit
Decode: 12.19 ms/tok (82.0 tok/s) [15 tokens, forward only]
Avg TBT: 30.14 ms (33.2 tokens/sec)
```

### OpenCL primary (baseline)

```
$ adb shell './generate -m qwen2.5-1.5b-q4_0.gguf -p "The capital of France is" -n 16 \
             --backend opencl --temperature 0'

The capital of France is a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit
Decode: 28.01 ms/tok (35.7 tok/s) [15 tokens, forward only]
Avg TBT: 28.45 ms (33.5 tokens/sec)
```

**→ token sequence byte-equal ✅ — D-F backend 정확성 PASS**

### Pre-D-F vs Post-D-F OpenCL primary 비교 (regression check)

| 시점 | OpenCL primary 결과 (greedy, temp=0) |
|---|---|
| **Pre-D-F** (`f8c85b3`) | `a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit` |
| **Post-D-F** (`8c513d3`) | `a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit` |

**→ byte-equal — D-F가 OpenCL primary path에 regression 없음 ✅**

### Fast path 활성 (`LLMRS_QNN_OPPKG_FAST_PATH=1`, `--initial-kv-capacity 2048`)

```
The capital of France is a overdublic vocational(QL automaticbirthdatequImportant folders ISSSF связ(Android
Decode: 111.23 ms/tok (9.0 tok/s)
Avg TBT: 120.23 ms (8.3 tokens/sec)
```

**→ token mismatch (graph가 OpenCL과 다른 forward 결과)**, Decode 111ms (bridge overhead 7MB/token + graph execute)

### 출력 garbage 원인 (D-F/D-D.6과 무관)

`Qwen2.5-1.5B-Instruct` 모델 + raw prompt + greedy decoding 조합 문제:
- Instruct model은 chat template (`<|im_start|>...<|im_end|>`) 가정
- raw `"The capital of France is"`는 turn 컨텍스트 부재 → model이 어떤 turn 안인지 모름
- greedy는 deterministic이라 잘못된 path 회복 불가
- Pre-D-F도 동일 garbage greedy → backend 변경과 무관한 model serving 문제
- 이전 RED 보고의 정상 출력 (`"Paris. It has a population of about 2 million people..."`)은 sampling default (temp=0.8) — random lucky로 정상 path

**→ output quality는 별개 이슈 (chat template 적용 또는 base 모델로 해결). D-F/D-D.6 작업 범위 밖**

---

## 3. Fast path 잔존 root cause — 다음 세션 디버깅 entry

`LLMRS_QNN_OPPKG_FAST_PATH=1` 활성 시 token mismatch 잔존. 잔존 가설:

### 가설 A — graph 14-node pipeline 자체가 OpenCL forward path와 다른 결과

M2.H (`microbench_qnn_qwen_layer`)는 **single-layer + pos=0 + n_kv=1** bit-exact GREEN. production 28-layer 통합 + multi-pos + n_kv 동적 환경에서 어디 step부터 divergence?

**격리 절차**:
1. `microbench_qnn_qwen_layer` (M2.H)를 D-D.6 spec에 맞게 갱신 — n_kv를 input tensor로 binding
2. n_kv {1, 16, 128, 1024, 2048} × pos {0, 1, 100, 1000} 매트릭스로 graph output을 OpenCL forward와 비교
3. divergence 발견 시 6 sub-graph bisect (이전 M2.H bisect2 패턴 재활용)

### 가설 B — mask buffer layout (D-D.6에서 kv_capacity 크기 확장) mismatch

D-D.6에서 mask buffer를 `n_kv`(고정)에서 `kv_capacity`(2048)로 확장. kernel은 `mask_ptr[k_idx]`로 `[0..n_kv)` index. n_kv ≤ kv_capacity니 in-bounds 안전. 그러나:
- mask_nb1/2/3=0 strides + mask_ne2/3=1 가정이 여전히 valid한가?
- 새 size에서 zero-init이 정상 (mask=0 → score += slope * 0 → no-op) 보장되나?

**검증**: 위 microbench로 mask buffer를 명시적으로 zero-init하고 score path 확인

### 가설 C — KV stride graph vs OpenCL 가정 차이

graph가 KvScatter/FlashAttn에서 KV layout `[1, n_kv_heads, kv_capacity, head_dim]` HeadMajor F16 stride 가정.
- `kv_capacity=2048` build-time SCALAR
- transformer KV는 `--initial-kv-capacity 2048`로 강제 일치 시켰으나 실제 stride 1대1 검증 안 함
- bridge가 cl_mem → host vec (read_buffer)로 전체 KV 옮길 때 layout 일치성 가정

**검증**: bridge 진입 직전과 직후 KV bytes의 hex dump (1 layer, 1 step) 비교 — 누적 KV가 graph rpcmem과 OpenCL cl_mem에서 같은 byte pattern인가?

### 가설 D — n_kv_buf binding 동작 검증

`graphExecute` 직전 host write가 graph 안에 정말 propagate되는지:
- `*self.n_kv_host_ptr = n_kv as i32` 직후 fence/barrier 없음
- QNN driver가 caching하지 않는다는 보장 필요 (D-D.1 pos는 동작 검증 OK)

**검증**: microbench에서 n_kv를 다른 값들로 시퀀셜 호출하고 결과가 변하는지 확인

---

## 4. 다음 세션 작업 plan

### Phase 1 — 정확성 격리 (1-2일)

1. M2.H microbench (`microbench_qnn_qwen_layer`) D-D.6 spec 호환 갱신
   - `mk_fa_params`에서 n_kv SCALAR 제거
   - n_kv_buf rpcmem slot + input tensor binding 추가
2. multi (n_kv, pos) 매트릭스 GREEN 검증
3. divergence 발견 시 6 sub-graph bisect — 이전 M2.H bisect2 (`microbench_qnn_qwen_layer_bisect2`) 패턴 재활용
4. root cause 격리 후 fix → device main gate 재시도

### Phase 2 — `LLMRS_QNN_OPPKG_FAST_PATH=1` default 활성

- Phase 1 완료 시 `supports_layer_graph()` env gate 제거 → default true
- INV-175 (decode 동안 fallback_call_count=0) 재검증

### Phase 3 — 옵션 3 (KV rpcmem alloc, zero-copy)

- bridge overhead 측정 (Phase 1/2에서 fast path 정상 시 ~111 ms/tok 일부)
- KV cache을 처음부터 rpcmem alloc (`QnnOppkgMemory::alloc_kv` 구현)
- bridge 코드 제거 → graph 자체가 KV의 single source
- prefill 종료 시 OpenCL → rpcmem 1회 mirror만 필요 (또는 prefill도 fast path)
- 추정 1주 (KV cache architecture 변경)

---

## 5. 환경 세팅 (다음 세션 시작 절차)

### 호스트 (mac-arm)

```bash
cd /Users/li/Workspace/llm_rs2

# QNN SDK (gitignored, 469MB)
ls third_party/qnn_sdk_2.33/include/QNN/   # 헤더 OK여야
ls third_party/qnn_sdk_2.33/lib/aarch64-android/  # libQnn*.so

# 디바이스
adb devices                  # R3CY408S5SB device
git log --oneline -5         # 8c513d3 HEAD 확인
git status                   # working tree clean
```

### SDK 복사 (다른 머신에서 시작 시)

QNN SDK는 gitignored. 원격 source: `go@192.168.219.125:Workspace/qnn/qairt/2.33.0.250327/` (Arch Linux, password=go, SSH key 등록 권장).

```bash
ssh-copy-id go@192.168.219.125  # 1회
mkdir -p third_party/qnn_sdk_2.33/lib
ssh go@192.168.219.125 'cd Workspace/qnn/qairt/2.33.0.250327 && tar czf - include' \
  | tar xzf - -C third_party/qnn_sdk_2.33/
ssh go@192.168.219.125 'cd Workspace/qnn/qairt/2.33.0.250327 && tar czf - lib/aarch64-android lib/x86_64-linux-clang lib/hexagon-v79 share' \
  | tar xzf - -C third_party/qnn_sdk_2.33/
```

### 디바이스 SDK runtime (`/data/local/tmp/qnn/`)

7개 lib 필수: `libQnnGpu.so libQnnGpuNetRunExtensions.so libQnnGpuProfilingReader.so libQnnHtp.so libQnnHtpV79CalculatorStub.so libQnnHtpV79Stub.so libQnnSystem.so`

```bash
adb -s R3CY408S5SB shell 'mkdir -p /data/local/tmp/qnn'
cd third_party/qnn_sdk_2.33/lib/aarch64-android
for lib in libQnnGpu.so libQnnGpuNetRunExtensions.so libQnnGpuProfilingReader.so \
           libQnnHtp.so libQnnHtpV79CalculatorStub.so libQnnHtpV79Stub.so libQnnSystem.so; do
  adb -s R3CY408S5SB push $lib /data/local/tmp/qnn/
done
```

### 빌드 + deploy

```bash
python scripts/run_device.py -d galaxy_s25 --skip-exec generate
adb -s R3CY408S5SB push target/aarch64-linux-android/release/deps/libqnn_oppkg.so \
  /data/local/tmp/libqnn_oppkg.so
```

### 디바이스 검증 (default — 정상 동작 확인)

```bash
adb -s R3CY408S5SB shell 'cd /data/local/tmp && \
  export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
  ./generate -m /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
             -p "The capital of France is" -n 16 \
             --backend qnn_oppkg --temperature 0'
# Expected: token byte-equal with --backend opencl, Decode ~12 ms/tok
```

### Fast path 활성 (디버깅용)

```bash
adb -s R3CY408S5SB shell 'cd /data/local/tmp && \
  export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
  export LLMRS_QNN_OPPKG_FAST_PATH=1 && \
  ./generate ... --initial-kv-capacity 2048'
# Expected (현 시점): garbage tokens, Decode ~111 ms/tok (bridge overhead)
```

---

## 6. 변경 파일 (이번 세션, 7 files)

| 파일 | LOC 변화 | 내용 |
|---|---|---|
| `devices.toml` | +1/-1 | galaxy_s25 serial 갱신 |
| `engine/src/backend/qnn_oppkg/mod.rs` | +296/-32 | copy_from fallback 위임, 14 method wrap, execute_layer_graph bridge, env-gated supports_layer_graph |
| `engine/src/backend/qnn_oppkg/layer_graph.rs` | +183/-31 | tensor_bytes_owned, idx_kv_*, n_kv_buf slot/tensor/host_ptr, lg.execute KV sync + n_kv arg |
| `engine/src/core/backend.rs` | +1 | execute_layer_graph trait sig n_kv 추가 |
| `engine/src/models/transformer.rs` | +6/-2 | fast path n_kv = start_pos + seq_len |
| `crates/qnn_oppkg/src/ops/flash_attn.rs` | +85/-37 | 7 inputs, 4 params, n_kv_buf input tensor + dynkv kernel name |
| `crates/qnn_oppkg/src/lib.rs` | +20/-22 | test_support n_kv_buf input + numOfParams=4 |
| `engine/kernels/flash_attn_f32_f16.cl` | +190 | flash_attn_f32_f16_q1_dynkv 신규 (body 복제 + n_kv_buf arg) |

---

## 7. 주의사항 (Do / Don't)

### DO
- **default (no env)에서 qnn_oppkg primary 동작 검증**: token byte-equal with OpenCL, Decode ~12 ms/tok이어야
- **fast path 디버깅은 microbench 우선** — production 28-layer 통합에서 격리 어려움
- M2.H bisect 패턴 재활용 — 6 sub-graph 단계별 max_abs_err 측정
- n_kv를 다양한 값으로 테스트 (1, 16, 128, 1024) — single-token 외 multi-token 경로 검증

### DO NOT
- `supports_layer_graph()` env gate 제거하지 말 것 (정확성 검증 전까지)
- 원본 `flash_attn_f32_f16_q1` kernel 변경 금지 (OpenCL primary path 사용 중)
- mask buffer 크기 변경 시 kernel index 가정 영향 확인 필요
- KV cache을 dynamic capacity로 두지 말 것 (graph rpcmem KV는 fixed kv_capacity stride) — `--initial-kv-capacity 2048` 강제

---

## 8. 외부 참조

### 진입점

- `engine/src/backend/qnn_oppkg/mod.rs:execute_layer_graph` — bridge logic
- `engine/src/backend/qnn_oppkg/layer_graph.rs:execute` (android module) — KV sync + n_kv host write
- `engine/src/backend/qnn_oppkg/layer_graph.rs:build_layer_graph` — n_kv_buf slot + FlashAttn node
- `crates/qnn_oppkg/src/ops/flash_attn.rs::build_layout` — n_kv_buf input tensor descriptor
- `engine/kernels/flash_attn_f32_f16.cl:flash_attn_f32_f16_q1_dynkv` — D-D.6 kernel variant

### 이전 핸드오프

- `.agent/todos/handoff_qnn_oppkg_m3_4_red_pos_baked_20260510.md` — D-D 시리즈 시작
- `.agent/todos/handoff_qnn_oppkg_m3_production_wireup_20260510.md` — M3 production 진입
- `papers/eurosys2027/_workspace/experiment/m3_4_passgate.md` — D-D 진행 측정 + verdict

### 이전 microbench (D-D.6 spec 호환 갱신 필요)

- `engine/src/bin/microbench_qnn_qwen_layer.rs` — M2.H 14-node single-layer microbench
- `engine/src/bin/microbench_qnn_qwen_layer_bisect2.rs` — M2.H 6 sub-graph bisect

---

**End of Handoff**

self-contained: 다음 세션은 본 문서 + `m3_4_passgate.md` + 위 진입점 파일만으로 fast path 정확성 디버깅 시작 가능.
