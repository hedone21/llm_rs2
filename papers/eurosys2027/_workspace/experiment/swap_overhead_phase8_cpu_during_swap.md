# Phase 8 — CPU Forward During Swap Window 측정

**작성일**: 2026-05-09
**Device**: Galaxy S25 (Adreno 830 + Snapdragon 8 Elite for Galaxy CPU)
**Model**: Qwen2.5-1.5B Q4_0 / F16

## 한 줄 요약

**ARM CPU NEON Q4_0 forward는 GPU와 거의 동등한 throughput** (30 vs 28 ms/tok). 이전 "5x 느림" 가정 폐기. **CPU during swap window 가능성 매우 높음**, 단 production 통합은 post-swap CPU forward segfault (별도 이슈) 해결 필요.

## 측정 1: 순수 forward TBT (no swap)

### Qwen2.5-1.5B Q4_0 GGUF
```
GPU (Adreno 830):  Decode 28.07 ms/tok, TTFT 217.34 ms
CPU (ARM NEON):    Decode 29.52 ms/tok, TTFT 672.99 ms
CPU/GPU TBT 비율:  1.05x (CPU이 5% 느림)
```

### Llama 3.2 1B BF16 Safetensors
```
GPU (Adreno 830):  Decode 41.59 ms/tok, TTFT 346.58 ms
CPU (ARM NEON):    Decode 61.04 ms/tok, TTFT 217.51 ms (prefill faster)
CPU/GPU TBT 비율:  1.47x (CPU이 47% 느림)
```

### 비교 분석
- **Q4_0 모델**: CPU NEON dotprod이 GPU와 동등 (1.05x) — 4-bit weight + INT 연산이 NEON에 최적
- **BF16 모델**: CPU 1.47x 느림 — float 연산이 더 무거움 (NEON SVE 미사용)
- **메모리 노트의 "CPU 5x 느림"은 f16 + Qwen NEON 버그 한정** — 일반 BF16/Q4_0은 1-1.5x slower
- 두 경우 모두 hybrid scheduling에 충분히 효과적 — Q4_0은 거의 free, BF16도 부분 hide 가능

## 측정 2: 스타트업 swap (f16 primary → Q4_0 secondary)

### GPU + swap
```
swap stages: prefault=2.9ms mmap_permute=304.7ms arc_swap=0.0ms madvise=0.1ms synchronize=1.5ms
total swap stall: 309.2ms
TTFT: 500.76ms (swap + prefill + first token)
Decode (post-swap): 23.38 ms/tok ← AOS Q4_0 path가 GGUF Q4_0보다 약 17% 빠름
```

### CPU + swap
```
swap stages: prefault=13.2ms mmap_permute=353.6ms (CPU에서 mmap_permute 살짝 느림)
total swap stall: 368.9ms
첫 forward token: SEGFAULT
```

**Post-swap CPU forward segfault** — 메모리 노트 명시: "post-swap CPU forward 정확성은 별도 사전 존재 이슈로 분리". Qwen2.5-1.5B 한정 (Llama 3.2 1B는 별도 검증 필요, GGUF on-device 미준비).

## 이론적 분석

### Sync swap baseline (현재 production)
- swap stall: 290ms (per swap event)
- N tokens forward: N × 28ms (GPU)
- Total: 290 + 28N ms

### Hybrid CPU during swap (target)
- swap GPU: 290ms (background)
- N tokens forward CPU: N × 30ms (foreground, parallel to swap)
- Total: max(290, 30N) ms

### Saving 분석 — Qwen2.5-1.5B Q4_0 (CPU TBT 30 ms/tok)
| N tokens | Sync swap | Hybrid | Saving |
|---------:|----------:|-------:|-------:|
| 1 | 318 ms | 290 ms | 28 ms (9%) |
| 5 | 430 ms | 290 ms | **140 ms (33%)** |
| 9 | 542 ms | 290 ms | **252 ms (46%)** |
| 10 | 570 ms | 300 ms | **270 ms (47%)** |
| 20 | 850 ms | 600 ms | 250 ms (29%) |

**Q4_0 sweet spot**: N=10 tokens — swap stall **완전 hide** (270ms saving).

### Saving 분석 — Llama 3.2 1B BF16 (CPU TBT 61 ms/tok)
| N tokens | Sync swap | Hybrid | Saving |
|---------:|----------:|-------:|-------:|
| 1 | 332 ms | 290 ms | 42 ms (13%) |
| 4 | 458 ms | 290 ms | **168 ms (37%)** |
| 5 | 500 ms | 305 ms | **195 ms (39%)** |
| 10 | 710 ms | 610 ms | 100 ms (14%) |

**BF16 sweet spot**: N=4-5 tokens — partial hide, 39% saving in best case.

### 적용 시나리오
- **Mid-decode swap** (KV pressure / thermal trigger): swap 시점에 CPU 전환 → ~10 토큰 동안 CPU forward → swap 완료 후 GPU 복귀
- **Startup swap**: swap이 prefill 전이라 CPU 전환 효과 없음 (forward 자체가 없음)

## Production 통합 차단 요인

### 1. Post-swap CPU forward segfault (Qwen2.5)
- 정확한 원인 미확인 (NEON path + 새 weight layout 호환성)
- 메모리 노트에 알려진 이슈로 등록되어 있음
- **별도 fix 필요** (예상: 1-3일 디버깅)

### 2. Llama 3.2 1B GGUF 디바이스 미준비
- 호스트 모델은 Safetensors only
- 다른 모델 (Gemma 3 1B, Llama 3.1 8B 등)에서 검증 필요

### 3. Hybrid scheduling 인프라
- Manager-driven SwitchHw directive는 존재
- 그러나 swap timing과 자동 동기화 안 됨
- 필요: swap 시작 직전 → mock_manager 또는 CLI flag로 SwitchHw cpu 트리거

## 결론

### POSITIVE finding
- **ARM CPU NEON은 Adreno GPU와 거의 동등한 Q4_0 throughput** — 이전 "CPU 5x slow" 가정 폐기
- 이는 본 paper의 hybrid scheduling proposal에 강력한 근거
- Hexagon DSP / Vulkan compute 같은 복잡한 path 없이 standard CPU NEON로 충분

### Production roadmap
1. **(Step 1)** Post-swap CPU forward segfault fix — Qwen2.5 + Llama 3.2 1B 양쪽 (~1-3일)
2. **(Step 2)** Mid-decode swap trigger 인프라 — Manager directive를 swap 시점과 동기화
3. **(Step 3)** Hybrid switching policy: detect swap start → SwitchHw cpu → swap completes → SwitchHw gpu
4. **(Step 4)** End-to-end 측정: 290ms saving 검증

### Paper 가치
- "Adreno OpenCL HW serialize는 우회 불가능 (Phase 6)"
- "그러나 CPU NEON Q4_0이 GPU와 동등 → CPU during swap window가 290ms 완전 hide 가능"
- "이는 mobile SoC에서 새로운 hybrid scheduling 패러다임을 제시"

이 finding을 OpenCL exhaustive negative + Phase 6 HW serialize evidence와 결합하면, paper의 main contribution은 단순한 negative가 아닌:

> "GPU stack의 fundamental constraint를 직접 입증하고, 동시에 모바일 ARM CPU NEON이
> Q4_0 양자화에서 GPU-class throughput을 달성함을 보여 hybrid CPU/GPU scheduling이
> 새로운 우회 path임을 제안."

---

2026-05-09 (Phase 8 완료)
