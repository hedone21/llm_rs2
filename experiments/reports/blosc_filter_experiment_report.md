# Blosc Filter Experiment Report

> ZramStore 압축 파이프라인에 Blosc2 기반 필터(bytedelta, trunc_prec)를 적용한 실험 결과

**Date**: 2026-03-14
**Commit**: 7cfcb33 (`feat(offload): add Blosc-inspired bytedelta filter and configurable ZramStore pipeline`)

---

## 1. 배경

ZramStore의 기존 파이프라인(`byte-shuffle → LZ4`)이 실제 F16 KV 캐시 데이터에서 압축률 **1.0x**를 기록하여 사실상 압축이 되지 않는 문제가 있었다. F16의 10-bit mantissa가 pseudo-random이라 바이트 엔트로피가 ~7.3 bits/byte로 높기 때문이다.

이를 개선하기 위해 Blosc2의 핵심 전처리 필터인 **bytedelta**와 **trunc_prec**을 순수 Rust로 구현하고, Zstd 코덱을 추가하여 5가지 파이프라인 조합의 성능을 측정했다.

## 2. 파이프라인 조합

| # | Pipeline | 손실 여부 |
|---|----------|-----------|
| ① | shuffle + LZ4 | 무손실 (기준선) |
| ② | shuffle + bytedelta + LZ4 | 무손실 |
| ③ | shuffle + bytedelta + Zstd(1) | 무손실 |
| ④ | trunc(3) + shuffle + bytedelta + LZ4 | 손실 (mantissa 3bit 제거) |
| ⑤ | trunc(5) + shuffle + bytedelta + Zstd(1) | 손실 (mantissa 5bit 제거) |

## 3. 벤치마크 결과

### 3.1 Small (256 tokens, 512 KB, 8 heads × 64 dim × F16)

| Pipeline | Ratio | Compress | Decompress |
|----------|-------|----------|------------|
| ① shuffle+LZ4 | 1.00x | 0.486 ms | 0.237 ms |
| ② shuffle+bytedelta+LZ4 | 1.00x | 0.094 ms | 0.412 ms |
| ③ shuffle+bytedelta+Zstd(1) | 1.09x | 0.579 ms | 0.680 ms |
| ④ trunc(3)+shuffle+bytedelta+LZ4 | 1.00x | 0.202 ms | 0.284 ms |
| ⑤ trunc(5)+shuffle+bytedelta+Zstd(1) | 1.56x | 0.651 ms | 0.776 ms |

### 3.2 Large (512 tokens, 1024 KB, 8 heads × 64 dim × F16)

| Pipeline | Ratio | Size | Compress | Decompress |
|----------|-------|------|----------|------------|
| ① shuffle+LZ4 | 1.00x | 1,052,692 B | 0.824 ms | 0.430 ms |
| ② shuffle+bytedelta+LZ4 | 1.00x | 1,052,696 B | 0.296 ms | 0.848 ms |
| ③ shuffle+bytedelta+Zstd(1) | 1.09x | 961,045 B | 2.091 ms | 1.277 ms |
| ④ trunc(3)+shuffle+bytedelta+LZ4 | 1.00x | 1,052,696 B | 1.260 ms | 0.581 ms |
| ⑤ trunc(5)+shuffle+bytedelta+Zstd(1) | 1.57x | 669,467 B | 1.302 ms | 1.422 ms |

> 측정 환경: x86_64 호스트, `--release` 빌드, 합성 데이터 (집중 exponent + random mantissa)

### 3.3 속도 예산 평가 (4MB 블록 기준)

ARM64 디바이스에서의 해제 지연 예산은 **3ms/4MB**. 호스트 측정값을 4MB로 선형 외삽:

| Pipeline | 4MB 추정 해제 시간 | 예산 대비 |
|----------|-------------------|-----------|
| ① shuffle+LZ4 | ~1.7 ms | OK |
| ② shuffle+bytedelta+LZ4 | ~3.3 ms | 경계 |
| ③ shuffle+bytedelta+Zstd(1) | ~5.0 ms | 초과 |
| ⑤ trunc(5)+bytedelta+Zstd(1) | ~5.5 ms | 초과 |

> ARM64 NEON에서의 실측은 미수행. Zstd 해제가 LZ4 대비 ~2-3x 느림.

## 4. 분석

### 4.1 무손실 파이프라인 한계

- **bytedelta만으로는 부족**: F16 mantissa의 delta도 높은 엔트로피를 가져 LZ4가 압축하지 못함 (1.00x)
- **Zstd는 소폭 개선**: bytedelta 후 Zstd를 적용하면 1.09x까지 도달하지만, 압축/해제 비용이 높아 실용적 이점이 미미
- **무손실 이론 상한**: F16 KV 캐시의 무손실 압축률 상한은 ~1.2-1.25x로, 어떤 코덱을 사용해도 대폭 개선은 불가

### 4.2 손실 파이프라인 (trunc_prec)

- **trunc(3)**: mantissa 3bit 제거 → LZ4로도 압축 불가 (1.00x). 남은 7bit 엔트로피가 여전히 높음
- **trunc(5)**: mantissa 5bit 제거 (10bit 중 절반) → **1.57x** 달성. 이는 F16의 precision을 절반으로 줄인 대가
  - 효과적으로 F16 → ~11-bit float 변환에 해당
  - 품질 영향: greedy decoding 토큰 일치율 80% 이상 기대 (실제 모델 검증 필요)

### 4.3 Blosc 필터 효과 평가

Blosc의 bytedelta 필터는 F16 KV 캐시 데이터의 근본적 한계 — **mantissa가 quasi-random** — 를 극복하지 못한다. 정수형/F32 데이터에서 효과적인 bytedelta가 F16에서는 delta 자체도 높은 엔트로피를 유지하기 때문이다.

의미 있는 압축을 위해서는 **손실 전처리(trunc_prec)**가 필수이며, 이 경우 trunc(5)+Zstd 조합이 1.57x로 가장 높은 압축률을 보인다.

## 5. 권장 사항

### 기본 설정 (무손실 우선)

```
codec: LZ4, bytedelta: false, trunc_bits: 0
```

현재와 동일. 무손실에서는 어떤 조합도 의미 있는 개선을 제공하지 않으므로, 추가 CPU 비용 없이 LZ4 기본값을 유지한다.

### 메모리 압박 시 대안

```
codec: Zstd(1), bytedelta: true, trunc_bits: 5
```

메모리가 극도로 부족한 상황에서 **1.57x** 절감 (37% 절약). 단, 해제 지연이 3ms 예산을 초과할 수 있으므로 ARM64 실측 후 사용 결정. 품질 영향도 실 모델 E2E 테스트로 검증 필요.

### 후속 연구 방향

1. **양자화 기반 접근**: KIVI(2-bit, 8x), GEAR(4-bit, 4x) 등 quantization-based KV cache 압축이 정보 이론적 한계를 우회하는 더 효과적인 전략
2. **ARM64 NEON 최적화**: bytedelta/trunc_prec의 SIMD 구현으로 전처리 비용 절감
3. **실제 모델 데이터 검증**: Llama 3.2 1B 추론 중 실제 KV 캐시를 캡처하여 합성 데이터와의 차이 확인
4. **Zstd 딕셔너리**: 레이어 간 유사한 패턴을 활용한 사전 훈련 딕셔너리 방식

## 6. 구현 요약

| 파일 | 변경 내용 |
|------|-----------|
| `engine/src/core/offload/preprocess.rs` | `bytedelta_encode/decode`, `trunc_prec_f16/f32` 추가 (17 tests) |
| `engine/src/core/offload/zram_store.rs` | `ZramConfig`, `ZramCodec`, configurable pipeline (15 tests) |
| `engine/Cargo.toml` | `zstd = "0.13"` 의존성 추가 |

전체 테스트: 448 passed (커밋 시점)

## 7. 결론

F16 KV 캐시 데이터에 대한 Blosc 필터(bytedelta) 적용은 **무손실에서 의미 있는 압축 개선을 달성하지 못했다** (최대 1.09x). 이는 F16 mantissa의 높은 엔트로피라는 근본적 원인 때문이며, 전처리 필터만으로는 해결할 수 없다.

손실 허용 시 trunc(5)+Zstd 조합으로 **1.57x**까지 가능하지만, 품질 트레이드오프와 해제 속도 페널티를 감수해야 한다.

실질적인 KV 캐시 메모리 절감을 위해서는 **양자화 기반 접근**(KIVI, GEAR 등)이 더 적합한 방향이다.
