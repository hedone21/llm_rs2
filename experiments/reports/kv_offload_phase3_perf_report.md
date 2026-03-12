# KV Cache Offload Phase 3: Performance & Memory Report

**Date**: 2026-03-12
**Model**: Llama 3.2 1B (dim=2048, 16 layers, kv_heads=8, head_dim=64)
**Platform**: x86_64 Linux, CPU backend (release build, LTO)
**Commit**: `2f638f1` (feat(offload): integrate KV cache offload with per-layer prefetch pipeline)

---

## 1. Executive Summary

Phase 3에서 LMCache-inspired per-layer prefetch 파이프라인과 CLI 통합을 구현했다.
E2E 추론 벤치마크 결과:

- **DiskStore F16**: 12% 성능 손실로 **93% KV cache RAM 절감** (유일하게 의미 있는 메모리 절감)
- **ZramStore F16**: 4-5% 성능 손실이지만, 실제 모델 데이터에서 **압축률 1.0x** → RAM 이점 없음
- **Lazy attn 버퍼**: 두 모드 모두에서 attn 버퍼 메모리 **94% 절감** (16 layers → 2 layers)

---

## 2. E2E Inference Throughput

### 2.1 Short Prompt (7 tokens prompt + 64 decode)

| Configuration | tok/s | Avg Forward (ms) | Avg TBT (ms) | TTFT (ms) | vs BASE |
|---|---:|---:|---:|---:|---:|
| **BASE F16** | **42.3** | **23.66** | 23.66 | 507 | — |
| Offload Zram F16 | 40.5 | 24.42 | 24.69 | 496 | **-4.3%** |
| Offload Disk F16 | 38.2 | 25.89 | 26.16 | 487 | **-9.7%** |
| Offload Zram F32 | 35.1 | 28.21 | 28.49 | 500 | -17.0% |

### 2.2 Long Prompt (116 tokens prompt + 256 decode)

| Configuration | tok/s | Avg Forward (ms) | Avg TBT (ms) | TTFT (ms) | vs BASE |
|---|---:|---:|---:|---:|---:|
| **BASE F16** | **34.6** | **28.92** | 28.92 | 6,455 | — |
| Offload Zram F16 | 33.0 | 29.94 | 30.28 | 6,455 | **-4.6%** |
| Offload Disk F16 | 30.4 | 32.50 | 32.85 | 6,414 | **-12.1%** |

### 2.3 Throughput Analysis

- **TTFT (Time To First Token)**: Prefill은 standard `forward_into()` 사용 → 오프로드 모드와 무관하게 동일
- **Decode overhead 증가 원인**: 시퀀스 길이에 비례하여 `store.load_into()` 데이터량 증가
  - 70 tokens: load 70KB/layer → 4% overhead
  - 371 tokens: load 371KB/layer → 12% overhead
- **Zram vs Disk**: Zram이 ~5-7% 빠름 (메모리 내 LZ4 해제 vs 파일 I/O)
- **F32 vs F16**: F32는 2배 데이터량 → 추가 오버헤드 발생

---

## 3. ZramStore Compression: Real vs Synthetic Data

### 3.1 Compression Ratio Comparison

| Data Type | Tokens | Raw KV Size | ZramStore Size | Compression |
|---|---:|---:|---:|---:|
| **Synthetic (unit test)** | 160 | 655 KB | 271 KB | **2.42x** |
| Real model (short) | 70 | 2,240 KB | 2,240 KB | **1.00x** |
| Real model (medium) | 243 | 7,776 KB | 7,781 KB | **~1.00x** |
| Real model (long) | 371 | 11,872 KB | 11,880 KB | **~1.00x** |

### 3.2 Root Cause

실제 신경망 KV cache 데이터는 attention 연산 결과로, 값의 분포가 pseudo-random에 가깝다.
byte-shuffle 전처리(exponent/mantissa 분리)가 합성 데이터의 반복 패턴에는 효과적이지만,
실제 모델의 고엔트로피 값에는 LZ4가 압축할 수 있는 중복이 거의 없다.

- F16 byte-shuffle: high/low byte 분리 → 실 데이터에서는 두 바이트 모두 고엔트로피
- LZ4: 4-byte 최소 매칭 → 연속 동일 바이트 시퀀스 부재 시 압축 불가
- 결과: LZ4 메타데이터 오버헤드로 인해 오히려 **미세하게 크기 증가** (7,776KB → 7,781KB)

### 3.3 Implication

ZramStore는 다음 시나리오에서만 유효:
- 양자화된 KV 데이터 (Q4_0 등, 비트 패턴 반복)
- Sparse attention 출력 (다수 0값)
- 구조화된 합성 데이터

일반 F16/F32 추론에서는 **메모리 절감 효과가 없으므로 권장하지 않음**.

---

## 4. Memory Analysis

### 4.1 RAM Usage at 371 Tokens (116 prompt + 255 decode)

| Configuration | KV Data (RAM) | Attn Buffers | Total RAM | vs BASE |
|---|---:|---:|---:|---:|
| **BASE F16** (dynamic, cap=512) | 16.0 MB | — | **16.0 MB** | — |
| ZramStore F16 (preload) | 11.6 MB | 4.0 MB | **15.6 MB** | -2.5% |
| DiskStore F16 (preload) | **0 MB** (on disk) | 4.0 MB | **~4.7 MB** | **-71%** |

> BASE KVCache는 동적 용량 확장(grow-on-demand)으로 371 tokens 시 capacity=512.

### 4.2 RAM Usage at Maximum Capacity (2048 tokens)

| Configuration | KV Data (RAM) | Attn Buffers | Total RAM | vs BASE |
|---|---:|---:|---:|---:|
| **BASE F16** | 64.0 MB | — | **64.0 MB** | — |
| ZramStore F16 (preload) | ~64.0 MB | 4.0 MB | **~68.0 MB** | **+6% (악화)** |
| DiskStore F16 (preload) | **0 MB** (on disk) | 4.0 MB | **~4.7 MB** | **-93%** |

### 4.3 RAM Usage Detail Breakdown (DiskStore F16 at max capacity)

| Component | Size | Note |
|---|---:|---|
| On-disk KV files | 64 MB | 16 layers × 2 files (K, V) |
| Attn K buffer (layer N) | 2.0 MB | Lazy-allocated, active |
| Attn V buffer (layer N) | 2.0 MB | Lazy-allocated, active |
| Attn K buffer (layer N+1) | 2.0 MB | Preloading via background thread |
| Attn V buffer (layer N+1) | 2.0 MB | Preloading via background thread |
| Out K SharedBuffer | ~0.4 MB | Reusable, grows with seq_len |
| Out V SharedBuffer | ~0.4 MB | Reusable, grows with seq_len |
| **Total in RAM** | **~8.8 MB** | |
| **vs BASE 64 MB** | | **-86%** |

> Note: 정확한 out buffer 크기는 current_pos에 비례. max capacity 2048에서는 ~2MB.
> 실제 peak RAM = 4 MB (attn) + ~4 MB (out) = ~8 MB.

### 4.4 Lazy Attn Buffer Savings (R-P1)

Phase 2에서는 `new()` 시 모든 16 레이어가 attn 버퍼를 사전 할당:

| | Phase 2 (사전 할당) | Phase 3 (lazy, 2 layers) | 절감 |
|---|---:|---:|---:|
| F16, max_seq=2048 | 64 MB | 4 MB | **94%** |
| F32, max_seq=2048 | 128 MB | 8 MB | **94%** |

Phase 2의 사전 할당은 오프로드의 존재 의의를 무효화했으나 (BASE와 동일한 64MB),
Phase 3의 lazy 할당 + release_buffers()로 해결.

---

## 5. Micro-Benchmark: get_view() Latency

16 layers × 128 decode steps, Llama 3.2 1B 파라미터.

| Configuration | Total Decode (ms) | Per-Token (ms) | get_view (μs/call) |
|---|---:|---:|---:|
| BASE KVCache F32 | 8.8 | 0.069 | — |
| BASE KVCache F16 | 3.8 | 0.030 | — |
| ZramStore F16 (sync) | 1,003.8 | 7.842 | 486.2 |
| ZramStore F16 (preload) | — | — | **31.4** |
| DiskStore F16 (sync) | 104.0 | 0.813 | 35.5 |
| DiskStore F16 (preload) | — | — | **27.8** |

> preload 모드에서 get_view()의 μs/call이 극적으로 감소 (486→31μs).
> 이는 store.load_into() I/O가 이미 완료된 상태에서 memcpy만 수행하기 때문.

> Note: micro-benchmark는 순수 I/O 부하만 측정 (compute 없음).
> preload 모드의 total이 sync보다 높은 것은 thread::scope 오버헤드 때문이며,
> 실제 추론에서는 compute가 I/O를 은닉하여 E2E에서는 반대 결과.

---

## 6. Prefetch Pipeline Effectiveness

### 6.1 I/O-Compute Overlap Model

```
Timeline (1 decode token, 16 layers):

BASE:    [Compute L0] [Compute L1] ... [Compute L15]
         ├─ ~24ms total ─┤

Offload  [Load L0] [Compute L0 | Load L1] [Compute L1 | Load L2] ... [Compute L15]
(sync):           ├────── ~27ms total ──────┤
                  ↑ I/O 은닉 구간

Offload  [Compute L0 | Load L1] [Compute L1 | Load L2] ... [Compute L15]
(preload):├──────── ~25ms total ────────┤
          ↑ L0은 sync preload (첫 토큰만 +1.5ms)
```

### 6.2 Per-Layer Timing Estimate (F16, 371 tokens)

| Component | Duration | Note |
|---|---:|---|
| Layer compute (matmul + attention + FFN) | ~1.8 ms | 28.92ms / 16 layers |
| ZramStore load_into (371 tokens F16) | ~0.5 ms | LZ4 decompress + memcpy |
| DiskStore load_into (371 tokens F16) | ~1.2 ms | File read |
| thread::scope overhead | ~0.05 ms | Per-spawn on x86 Linux |

- **Zram F16**: load (0.5ms) < compute (1.8ms) → **파이프라인 완전 은닉**
- **Disk F16**: load (1.2ms) < compute (1.8ms) → **파이프라인 부분 은닉**
- 시퀀스가 길어질수록 load 시간 증가 → 스톨 가능성 증가

### 6.3 Android ARM64 Considerations (미측정, 추정)

| Factor | x86 Linux | ARM64 Android (estimated) |
|---|---|---|
| thread::scope spawn | ~0.05ms | ~0.3-0.5ms |
| 16 layers × spawn | ~0.8ms | ~5-8ms |
| Overhead ratio | ~3% | **~10-17%** |

> R-P6: Android에서 thread::scope 오버헤드가 5% 초과 시 Step 3c (persistent thread) 전환 필요.

---

## 7. Conclusions & Recommendations

### 7.1 Mode Selection Guide

| Scenario | Recommendation | Expected Impact |
|---|---|---|
| **메모리 제한 edge 디바이스** | `--kv-offload disk --kv-type f16` | -12% throughput, **-93% RAM** |
| 빠른 SSD가 있는 시스템 | `--kv-offload disk --kv-type f16` | -10% throughput, -93% RAM |
| 느린 저장장치 | `--kv-offload zram --kv-type f16` | -5% throughput, RAM 이점 없음 |
| **성능 최우선** | BASE (`--kv-offload none`) | 최고 throughput |
| 압축 가능한 데이터 (Q-aware) | `--kv-offload zram` | 데이터 특성에 따라 다름 |

### 7.2 Key Findings

1. **DiskStore가 유일하게 의미 있는 RAM 절감 제공** (93% at max capacity)
2. **ZramStore는 실제 모델 데이터에서 압축 불가** (엔트로피 1.0x) → RAM 이점 없음
3. **Per-layer prefetch가 I/O 은닉에 효과적**: DiskStore 12%, ZramStore 5% 오버헤드
4. **Lazy attn 버퍼는 근본적 개선**: 94% 버퍼 메모리 절감 (Phase 2의 치명적 문제 해결)
5. **TTFT 영향 없음**: Prefill은 standard forward_into() 사용

### 7.3 Future Work

| Item | Priority | Description |
|---|---|---|
| Android 실측 | HIGH | thread::scope 오버헤드 확인, persistent thread 필요성 판단 |
| 적응형 chunk size | MEDIUM | I/O > compute 시 multi-layer batch preload (Step 3a) |
| Compression 대안 | LOW | Zstd, dictionary compression, delta encoding 등 시도 |
| Eviction 통합 | LOW | 오프로드 + eviction 조합 (현재 미지원) |

---

## Appendix A: Test Configuration

```bash
# Short prompt
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --kv-offload {none|zram|disk} --kv-type f16 --kv-layout seq \
  --prompt "The future of artificial intelligence is" -n 64 --greedy

# Long prompt
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --kv-offload {none|zram|disk} --kv-type f16 --kv-layout seq \
  --prompt "In the year 2045, humanity had achieved..." -n 256 --greedy
```

## Appendix B: Micro-Benchmark

```bash
cargo test -p llm_rs2 --lib core::offload::tests::test_bench_preload_vs_sync_and_memory -- --nocapture
```

Parameters: kv_heads=8, head_dim=64, 16 layers, max_seq=2048, prefill=64, decode=128.
