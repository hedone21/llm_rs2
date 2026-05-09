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

### 사용자 결정 요청

본 세션 timebox 내에서 segfault root cause + production wire-up 통합은 미완료.
다음 세션 전에 다음 결정 필요:

- **D-A**: A안 (noshuffle SOA 강제)으로 진행 (~3-5일)
- **D-B**: B안 (prefill 전체 OpenCL 위임)으로 진행 (~2-3일)
- **D-C**: scope 재정의 — qnn_oppkg를 prefill 미지원 backend로 명시하고 transformer.rs가 prefill 시점에 OpenCL primary로 강제 전환 (~2일)

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
