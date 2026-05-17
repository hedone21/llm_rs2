# Phase 4-3 vtable microbench — host CPU baseline

**Date**: 2026-05-17
**Phase**: 4-3 C3 (probe_inference_loop microbench)
**HEAD**: `c63190d1 feat(session): probe_inference_loop microbench (Phase 4-3 C3)`
**Bench binary**: `engine/src/bin/probe_inference_loop.rs`

## 측정 목적

`DecodeLoop + ModelForward` (vtable path) vs `model.forward_into()` 직접 호출
(baseline)의 **avg_tbt 회귀 ≤ 5%** 게이트(arch §7.3). 회귀 > 5% 시 escape hatch
(`DecodeLoop<F: Forward, T: TokenSampler>` 2-trait monomorphize) 트리거.

본 측정은 **호스트 CPU baseline** — 디바이스 (S25 OpenCL + Jetson CUDA) 측정은
Tester (`/deploy-test` + `/profile`) 별도 위임.

## 호스트 환경

- Arch Linux 7.0.3-arch1-2, x86_64 AVX2+FMA
- Cargo workspace release build (`opt-level=3`, `lto=fat`, `codegen-units=1`)
- 모델: `Qwen2.5-1.5b-Q4_0.gguf` (`/home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/`)
- Tokenizer: `qwen2.5-1.5b/tokenizer.json`
- Backend: `--backend cpu` (NEON/AVX2 fallback)
- Prompt: `"The capital of France is"` (5 tokens)
- KV dtype: `f16`, max_seq_len: 64

## CLI

```bash
./target/release/probe_inference_loop \
    --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
    --backend cpu --gen 4 --runs 1 --max-seq-len 64
```

## 결과 (gen=4, runs=1)

| 메트릭 | DecodeLoop path | Direct forward_into | Δ |
|---|---:|---:|---:|
| avg_tbt_ms | **62.30** | **61.36** | **+1.53%** |
| tok0_ms | 176.11 | 177.88 | -1.00% |
| total_ms | 249.19 | 245.45 | +1.53% |
| tokens | `[12095, 13, 576, 6722]` | `[12095, 13, 576, 6722]` | **bit-identical** |

**Verdict**: `PASS` (`delta_pct=1.53% ≤ 5%` AND `bit_identical=true`).

## 해석

- vtable indirect call(매 step 6 trait 호출 + observer N개) 비용이 호스트 CPU에서
  **1.53%** 측정. arch §7.2 이론치(0.0003~0.0014%)보다 크지만 게이트 통과.
- 호스트 CPU forward는 GPU 대비 무거우므로 vtable 비율 묻힘 효과가 약함. **GPU
  backend에서는 더 낮은 비율 예상** (디바이스 측정으로 확인 필요).
- `bit_identical=true` → ModelForward 래퍼의 tensor upload / kv borrow plumbing /
  logits read-back이 직접 호출과 동등. paradigm은 양 path가 DecodeLoop 방식
  (prompt-last를 first step에 재입력) 통일.

## 다음 측정 (Tester C4)

| Device | Backend | 모델 | gen | runs | 게이트 |
|---|---|---|---:|---:|---|
| Galaxy S25 | `--backend opencl` (Adreno) | Qwen2.5-1.5b-Q4_0 | 32 | 5 | bit-identical + Δ≤5% |
| Jetson Xavier | `--backend cuda` | Qwen2.5-1.5b-Q4_0 | 32 | 5 | bit-identical + Δ≤5% |

S25 명령:
```bash
python scripts/run_device.py -d s25 probe_inference_loop -- \
    --model-path /data/local/tmp/Qwen2.5-1.5b-Q4_0.gguf \
    --tokenizer-path /data/local/tmp/tokenizer.json \
    --backend opencl --gen 32 --runs 5 --max-seq-len 512
```

Jetson (보드 내 빌드):
```bash
ssh jetson 'cd llm_rs2 && cargo build --release --bin probe_inference_loop \
    --features cuda-embedded \
    && ./target/release/probe_inference_loop --backend cuda \
       --model-path Qwen2.5-1.5b-Q4_0.gguf --tokenizer-path tokenizer.json \
       --gen 32 --runs 5 --max-seq-len 512'
```

회귀 ≥ 5% 시: arch §7.3 escape hatch — `DecodeLoop<F: Forward, T: TokenSampler>`
부분 generic화 PoC + 재측정.

## 한계

- gen=4 단발 (runs=1) — 디바이스 측정에선 gen=32, runs=5로 medians 사용
- 호스트 CPU는 production 대상이 아니므로 본 baseline은 "sanity 통과" 의미만
- `DecodeLoop::prefill`이 `()` 반환 → direct path도 DecodeLoop paradigm
  (prompt-last 두 번 forward) 사용. production decode 패턴 (prefill last logits
  → argmax → first generated)과 다름 — Phase 4-3.5 또는 4-4에서 prefill
  시그니처 정리 시 paradigm 통일 가능.
