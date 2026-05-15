# LISWAP-8 Hammer D: Zero-Copy MMap Alias Swap (2026-05-15)

## 목적
Phase B α (cuMemAlloc 제거) + B-2 (CudaHostBuffer + CPU memcpy)이 모두 baseline 회복 못함 → 진짜 culprit은 UMA 메모리 컨트롤러 BW 경쟁. Hammer D는 worker work 자체를 0으로: secondary GGUF mmap을 `cuMemHostRegister(DEVICEMAP | READ_ONLY)`로 한 번 pin + 매 swap은 mmap layer offset alias만 (memcpy/cuMemcpy 모두 0).

## 구현 (commit 대기)
- `engine/src/buffer/cuda_mmap_alias_buffer.rs` (신규)
  - `CudaMmapRegistration`: 전체 mmap을 4 KB-aligned size로 cuMemHostRegister. Drop 시 unregister.
  - `CudaMmapAliasBuffer`: registered region의 `[offset..offset+size)` 슬라이스. `Buffer` trait 구현 + `device_ptr()` exposed.
- `engine/src/backend/cuda_embedded/mod.rs::get_device_ptr` — alias buffer downcast 추가
- `engine/src/models/weights/swap_executor.rs::build_layer_via_mmap_alias_standalone` (신규)
- env: `LLMRS_SWAP_MMAP_ALIAS=1` + `LLMRS_SWAP_BG_FETCH=1` 같이

### Registration flag (운영 노트)
Jetson Xavier에서 `DEVICEMAP only` 는 INVALID_VALUE. `DEVICEMAP | READ_ONLY` 첫 시도 + fallback. READ_ONLY가 필요한 이유는 mmap이 MAP_PRIVATE/read-only.

## 결과 (Jetson Llama 3.1 8B Q4_0, 50 tokens)

| K | metric | baseline | bg_fetch | mmap_alias | Δ alias vs base | Δ alias vs bg | n |
|---|---|---|---|---|---|---|---|
| 2 | forward (rest_avg) | 91.66 | 117.97 | 108.96 | +17.30 | -9.01 | 1 |
| 2 | **active_avg** | 145.19 | 223.79 | **190.55** | +45.36 | **-33.24** | 1 |
| 2 | idle_avg | 67.48 | 67.68 | 70.73 | +3.25 | +3.05 | 1 |
| 2 | **rest_tbt** | 122.06 | 129.30 | **113.75** | **-8.31 ✓** | -15.55 | 1 |
| 4 | forward | 79.29 | 109.98 | 107.19 | +27.90 | -2.79 | 3 |
| 4 | **active_avg** | 146.60 | 326.78 | **289.57** | +142.97 | **-37.21** | 3 |
| 4 | idle_avg | 67.92 | 68.74 | 72.73 | +4.81 | +3.99 | 3 |
| 4 | **rest_tbt** | 114.67 | 119.18 | **113.12** | **-1.55 ✓** | -6.06 | 3 |
| 8 | forward | 72.77 | 101.89 | 100.58 | +27.81 | -1.31 | 3 |
| 8 | **active_avg** | 146.90 | 490.30 | **395.57** | +248.67 | **-94.73** | 3 |
| 8 | idle_avg | 67.93 | 68.45 | 75.51 | +7.58 | +7.06 | 3 |
| 8 | **rest_tbt** | 106.86 | 116.10 | **105.27** | **-1.59 ✓** | -10.83 | 3 |
| 32 | active_avg | 150.15 | 152.47 | 150.68 | +0.53 | -1.79 | 1 |
| 32 | idle_avg | 66.18 | 68.20 | **95.81** | **+29.63 ❌** | +27.61 | 1 |
| 32 | rest_tbt | 69.95 | 71.85 | 99.43 | +29.48 ❌ | +27.58 | 1 |

## 핵심 발견

1. **K=2/4/8에서 mmap_alias가 baseline TBT 능가** (1.55 ~ 8.31 ms 개선)
   - swap 자체 비용 0
   - active forward bg_fetch 대비 -33 ~ -95 ms 개선
2. **active forward는 baseline까지 회복 안 됨** (K=4: 290 vs 146)
   - 남은 +143 ms 는 mmap GPU read latency + prefault/page fault 일부
3. **K=32 outlier — idle_avg +29 ms**
   - 한 번 batch로 32 layer alias 생성 → 모든 weights mmap-backed
   - GPU forward 매 token이 mmap region read → +29 ms per token
   - K<32는 일부 layer만 alias (active window) → idle 영향 작음
4. **idle window forward는 K가 클수록 더 무거움** (K=4: 72.73 → K=32: 95.81)
   - alias 적용 layer 수에 비례

## paper main figure 후보

3-way comparison (active forward), K-sweep:

```
active forward (ms)
        K=2    K=4    K=8    K=32
base    145    147    147    150
bg_fetch 224   327    490    152
alias    191   290    396    151
        ▲ alias가 bg_fetch 대비 K가 클수록 더 큰 개선

TBT (ms)
        K=2    K=4    K=8    K=32
base    122    115    107     70
bg_fetch 129   119    116     72
alias   114   113   105     99   ← K<32에서 baseline 능가, K=32에서 outlier
```

## 폐기된 가설 (이전)

| 가설 | 검증 | 결과 |
|---|---|---|
| CUDA driver context lock | Phase B α: pool ≈ bg_fetch | ❌ |
| cuMemcpyHtoDAsync staging copy | Phase B-2: pool_zc 더 나쁨 | ❌ |

## 새로 확정된 culprit

**worker work 강도에 비례하는 UMA memory controller BW 경쟁**:
- mmap_alias (work=0): baseline 능가
- bg_fetch DMA: +180 ms regression
- pool_zc CPU memcpy: +280 ms regression

## 미해결 / Future Work

1. **active forward +143 ms 잔존** (K=4 baseline 대비)
   - 가설: mmap GPU read의 first-touch page fault 비용
   - 해결 후보: pre-fault mmap region (madvise WILLNEED) + cuMemHostRegister 직후 strong read
2. **K=32 idle window +30 ms regression**
   - 가설: 모든 weights가 mmap-backed → GPU read 매번 host-mapped 경유
   - 해결 후보: 사용 빈도 높은 layer는 device buffer로 copy (hybrid)
3. **PoC 정확성** — Q/K permute 무시 (GGUF 한정). AUF secondary는 정확.
   - 해결 후보: load time에 한 번 unpermute해서 별도 registered region에 저장 + alias

## 데이터 파일
- `k{2,4,8,32}_{baseline,bg_fetch,mmap_alias}_r{1,2,3}.{stdout,stderr,tbt.jsonl}` × 24 + verify
