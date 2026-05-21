# Phase 4-3 vtable microbench — host CPU baseline + S25 OpenCL

**Date**: 2026-05-17
**Phase**: 4-3 C3 (probe_inference_loop microbench) + C4 (디바이스)
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

## S25 OpenCL 환경

- Galaxy S25 (R3CY408S5SB), Adreno 830 (Snapdragon 8 Elite)
- Cross-compile: `cargo build --release --features opencl,vulkan,qnn --no-default-features --target aarch64-linux-android --bin probe_inference_loop`
- 모델: `/data/local/tmp/qwen2.5-1.5b-q4_0.gguf` (pure Q4_0, Q6_K 없는 변형)
- Tokenizer: `/data/local/tmp/qwen-tokenizer.json` (호스트에서 push)
- Backend: `--backend opencl` (Adreno)
- KV dtype: `f16`, max_seq_len: 512

### S25 명령

```bash
adb -s R3CY408S5SB shell 'cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp ./probe_inference_loop \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --gen 32 --runs 5 --max-seq-len 512'
```

### S25 결과 (gen=32, runs=5)

| 메트릭 | DecodeLoop path | Direct forward_into | Δ |
|---|---:|---:|---:|
| avg_tbt_ms | **33.18** | **32.44** | **+2.29%** |
| tok0_ms | 118.61 | 116.48 | +1.83% |
| total_ms | 1061.84 | 1038.11 | +2.29% |
| tokens (32) | identical | identical | **bit-identical** |

토큰 시퀀스 (32 토큰 양 path 동일):
`[12095, 13, 576, 6722, 315, 9625, 374, 12095, ...반복...]`

**Verdict**: `PASS` (`delta_pct=2.29% ≤ 5%` AND `bit_identical_first_n=true`).

## Jetson CUDA (pending)

Jetson Orin은 본 호스트의 `~/.ssh/config`에 `jetson` alias가 미등록 → 보드 내
직접 빌드 + 측정 필요. 별도 task로 분리. PASS 게이트 동일 (`Δ≤5%` +
`bit_identical_first_n`).

권장 명령 (보드 내):
```bash
cargo build --release --bin probe_inference_loop --features cuda-embedded
./target/release/probe_inference_loop --backend cuda \
    --model-path qwen2.5-1.5b-q4_0.gguf --tokenizer-path tokenizer.json \
    --gen 32 --runs 5 --max-seq-len 512
```

## 종합 평가

| Device/Backend | Δ% | bit-identical | Verdict |
|---|---:|:---:|:---:|
| Host CPU (x86_64 AVX2) | 1.53% | true | PASS |
| S25 Adreno OpenCL | **2.29%** | **true (32 toks)** | **PASS** |
| Jetson CUDA | — | — | pending |

vtable indirect call overhead가 양 측정 환경에서 게이트(5%) 통과. GPU forward가
무거워질수록 vtable 비율 묻힘 효과가 작아질 줄 알았으나, 실측은 호스트 1.53%,
S25 OpenCL 2.29%로 S25에서 약간 더 큼. 가설:
- 디바이스 GPU forward TBT (~33 ms) 자체가 호스트 CPU TBT (~60 ms)보다 절반 →
  분모가 작으니 vtable absolute 비용의 비율이 더 크게 보임
- 양쪽 모두 5%의 1/2 이내 — escape hatch 불필요 확정

회귀 ≥ 5% 가능성 거의 없으므로 **Phase 4-4 main() 조립자화 진입 가능** (Jetson
측정은 후속 보충).

## 한계

- 호스트 CPU 측정은 gen=4 단발 (sanity), S25는 gen=32 runs=5 (정식)
- Jetson CUDA 미측정 (SSH alias 미설정, 별도 task)
- 호스트 CPU는 production 대상이 아니므로 본 baseline은 "sanity 통과" 의미만
- `DecodeLoop::prefill`이 `()` 반환 → direct path도 DecodeLoop paradigm
  (prompt-last 두 번 forward) 사용. production decode 패턴 (prefill last logits
  → argmax → first generated)과 다름 — Phase 4-3.5 또는 4-4에서 prefill
  시그니처 정리 시 paradigm 통일 가능.
