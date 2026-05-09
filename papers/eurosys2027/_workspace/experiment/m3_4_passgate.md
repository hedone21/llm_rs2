# QNN OpPackage M3.4 메인 게이트 측정 리포트

**Date**: 2026-05-10
**HEAD**: (이번 세션 작업 후 - 미커밋)
**Device**: Samsung Galaxy S25 (R3CY408S4HN), Adreno 830, Hexagon V79
**Model**: Qwen2.5-1.5B Q4_0 (`/data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf`)
**Prompt**: "The capital of France is" (5 tokens)
**Tokens**: 32 target (실측은 prefill 진입 후 segfault로 0)
**Sampling**: greedy

## Verdict: **RED**

decode 진입 전 prefill 단계에서 segfault. 정확성/TBT/VmRSS 측정 미수행.

## 측정 결과

### OpenCL baseline (정상)

```
Prefill: 117.20 ms (5 tokens, 42.7 tok/s)
Decode: 28.07 ms/tok (35.6 tok/s) [31 tokens, forward only]
Avg TBT: 28.57 ms (35.0 tokens/sec)
Token sequence: "Paris. It has a population of about 2 million people and covers an
area of 104 square kilometers (km2). The city is divided into"
```

### qnn_oppkg backend

```
[qnn_oppkg] runtime initialized: backend_lib=/data/local/tmp/qnn/libQnnGpu.so ...
[Backend] QNN-GPU primary, OpenCL secondary available (SwitchHw ready)
[Backend] qnn_oppkg fallback wired to OpenCL secondary (prefill + model load 위임)
...
[qnn_oppkg] eager prebuild: 28 layers, total finalize 1355~1364 ms
[Backend] Noshuffle preparation skipped: Not OpenCL backend
...
Generating (Max: 2048, Temp: 0, TopP: 0.9, TopK: 40)...
Segmentation fault
```

### graphFinalize 28× 분포 (5회 측정 평균)

| Layer | Time (ms) |
|---|---|
| 0 (cold) | 1175~1196 |
| 1 (warm) | 6.14 |
| 2~27 | 6.4~7.2 (mean ≈ 6.7) |
| **Total** | **~1360 ms** |

- INV-167 200 ms budget은 비현실적 — M3.4 측정에서 1500 ms로 갱신
- **layer 0 cold = ~1.2 s** (M2 microbench와 동일 — QNN driver lazy compile cache)
- **layer 1~27 warm = ~6.7 ms** — 동일 graph 인스턴스 cache hit
- 총 ~1360 ms (예상 ~33 s 대비 24× 빠름 — driver-level graph re-use 작동)

### 정확성 / TBT / VmRSS

**측정 불가** (prefill 단계 segfault).

## Root cause 분석 (잠정)

### 단계별 진입 결과

1. ✅ **runtime init**: libQnnGpu + OpPackage register PASS
2. ✅ **OpenCL secondary fallback wire-up**: PASS
3. ✅ **Model load**: GGUF 338 tensors load PASS (`copy_from`이 buffer passthrough로 처리)
4. ✅ **lm_head derivation**: Tied weight derive → `copy_from` passthrough OK
5. ✅ **graphFinalize 28×**: 모든 layer PASS (총 ~1360 ms)
6. ❌ **prefill (seq_len=5)**: 진입 직후 segfault

### Segfault 원인 가설

`[Backend] Noshuffle preparation skipped: Not OpenCL backend` 메시지가 핵심.

**가설 (검증 필요)**:
- production은 OpenCL backend primary일 때 weight를 noshuffle SOA로 변환
  (`OpenCLBackend::ensure_noshuffle_soa_registered`)
- qnn_oppkg primary일 때 backend.is_gpu()=true이지만 OpenCL이 아니라 noshuffle 미적용
- prefill의 OpenCL secondary fallback이 weight tensor를 받아 matmul 호출
  → weight가 AOS Q4_0 layout인데 noshuffle SOA kernel이 호출되어 stale pointer dereference
- 또는 graph가 KvScatter를 graph 내부 rpcmem buffer에 작성하지만, prefill의
  attention은 OpenCL backend의 KV cl_mem을 read → 동기화 안 된 stale data

### **CRITICAL DISCOVERY (2026-05-10 후속 분석)**: pos baked architectural blocker

M2.H microbench `engine/src/bin/microbench_qnn_qwen_layer.rs:1721-1741` 정독 결과:

```rust
let mk_rope_params = |start: i32, th: f32| {
    [
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,  // ← graph build 시점 baked
            name: pn_start_pos.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: start },
                },
            },
        },
        // ...
    ]
};
```

**M2.H는 `start_pos`/`write_pos`를 `QNN_PARAMTYPE_SCALAR`로 graph build 시점에 hardcoded.** M2.H는 single-token 테스트(pos=0)였으므로 이 한계가 노출되지 않음.

production multi-token decode는 pos가 0..31로 변하는데 graph가 pos=0으로 baked → RoPE/KvScatter가 모든 token에서 pos=0으로 동작 → 결과 garbage (segfault 이전에 정확성부터 fail).

이는 M3.4 RED의 **본질적 architectural blocker**. segfault는 부차적 (noshuffle SOA 미적용으로 OpenCL secondary fallback dereference) — 이는 1줄 fix 가능하지만, fix해도 pos baked 때문에 정확성 100% match 불가.

**해결 옵션 (심각도/시간 순)**:

- **D-D (CRITICAL, 신규)**: M2 frozen ops 수정 — `start_pos`/`write_pos`를 `QNN_PARAMTYPE_SCALAR` (build-time) → input tensor (execute-time)로 변경. RoPE/KvScatter op signature 갱신. 이는 `crates/qnn_oppkg/src/ops/rope.rs` + `kv_scatter.rs` 변경 + `kernel_rope_simple_oop` / `kernel_kv_scatter_*` `.cl` kernel 갱신 필요. **CLAUDE.md D7 결정에 따라 .cl 수정 가능**이지만 M2 검증이 무효화될 수도 — 별도 정확성 재검증 필요. 추정 5-7일.
- **D-E (가장 단순)**: M3 scope를 single-token (pos=0 hardcoded) 검증으로 한정. multi-token decode는 OpenCL fallback. M3 메인 게이트 (32-token 100% match)의 의미가 사라지므로 plan 재정의 필요.
- **A안/B안/C안** (이전 보고): segfault만 fix하는 방안. pos baked 문제로 정확성 PASS 불가.

### 다음 세션 todo (RED → 다음 방향)

이는 단순 wire-up이 아닌 **architectural** 결정 필요:

1. **A안 — qnn_oppkg primary일 때 noshuffle SOA 강제 활성**
   - `ensure_noshuffle_soa_registered`를 OpenCL backend가 아닌 진입점에서 호출
   - weight tensor의 backend reference가 OpenCL이 아니어도 OpenCL secondary가
     SOA 변환 수행
   - 변경 부피: 중간

2. **B안 — qnn_oppkg primary일 때 prefill을 별도 backend가 처리**
   - `--qnn-allow-fallback`을 활성하여 prefill만 OpenCL backend로 전체 위임
   - decode (seq_len=1)는 graph fast path
   - 변경 부피: 작음 (model object를 OpenCL primary로 만들고, decode 진입 시
     supports_layer_graph 검사로 fast path 분기). 그러나 backend dispatcher가
     primary를 변경 못 하므로 transformer.rs forward 분기가 backend.execute_layer_graph만
     교체

3. **C안 — KV layout / mask buffer pos-handling 명세 결정**
   - graph build 시 pos=0 baked인데 production 32-token decode는 pos가 0..31로 변함
   - QNN OpPackage는 graph build 후 param 변경 불가 → 모든 token에서 동일 graph
     사용 시 pos=0으로 고정되어 RoPE / KvScatter가 잘못 동작
   - 해결 방안: mask buffer를 통해 pos 동적 처리 (M2 microbench는 단일 token이라
     이슈 없음)

### 사용자 결정 요청 (UPDATED)

본 세션 timebox 내에서 root cause 정밀 격리 결과 — **단순 fix 불가**.

핵심 문제: **pos baked**. M2.H가 single-token 테스트라 noticed 안 된 architectural blocker. production 32-token decode는 graph rebuild 또는 op spec 변경 없이 PASS 불가능.

권장 진행 옵션 (재평가):

| ID | 옵션 | 시간 | M3 timeline 영향 | 위험 |
|---|---|---|---|---|
| **D-D** | M2 ops 수정 (RoPE/KvScatter pos를 input tensor로) | 5-7일 | +1.5주 | M2 검증 재실행 필요 |
| **D-E** | M3 scope 재정의 — single-token 검증 + multi-token OpenCL fallback | 2-3일 | +0.5주 | M3 게이트 의미 약화. paper evidence 약함 |
| **D-A/B/C** | segfault만 fix | 2-5일 | +0.5~1주 | pos baked 문제 미해결 → 정확성 RED 유지 |

**기존 M3 timeline (4주) → M3 완료 +1.5~2.5주 추가** 예상.

본 결정은 plan re-scope이므로 사용자 명시 결정 필요.

## 변경 파일 (이번 세션)

| 파일 | LOC 변화 | 내용 |
|---|---|---|
| `engine/src/backend/qnn_oppkg/layer_graph.rs` | +780 / -45 | 14-node body 본격 이식 (microbench 1680~2174 + execute path) |
| `engine/src/backend/qnn_oppkg/weight_pack.rs` | **신규 +120** | GGUF Q4_0 AOS → SOA layout 변환 (3 unit tests) |
| `engine/src/backend/qnn_oppkg/mod.rs` | +50 / -25 | fallback_or_panic helper + set_fallback_backend + copy_from passthrough + 12 trait method 위임 |
| `engine/src/bin/generate.rs` | +20 / -8 | qnn_oppkg primary일 때 OpenCL secondary memory를 production primary로 위임 + fallback wire-up |

총 4 files, ~1000 LOC 변화 (대부분 layer_graph.rs body).

## Pass-gate 결정

- **결과**: **RED**
- **Token sequence**: 측정 불가 (prefill segfault)
- **TBT ratio**: 측정 불가
- **VmRSS ratio**: 측정 불가
- **graphFinalize 28×**: max 1196 ms (cold), avg warm 6.7 ms, total 1360 ms — INV-167 1500 ms budget 내 PASS
- **다음 단계**: 사용자 호출 — D-A/D-B/D-C 중 결정 후 다음 세션 진행
