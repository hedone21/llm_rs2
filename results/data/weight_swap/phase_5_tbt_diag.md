# Phase 5 — WSWAP-5-TBT-DIAG cl_mem fragmentation 진단 리포트

- **측정일**: 2026-04-26
- **브랜치/HEAD (instrumentation)**: `feat/weight` (instrumentation patch on top of `aee9adc`)
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel 6.6.77, 6 threads (`--threads 6`), `--profile` **미사용 (TBT)** / **사용 (op breakdown만)**
- **모델**:
  - F16 GGUF: `Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB, 16 layers)
  - Q4_0 GGUF: `Llama-3.2-1B-Instruct-q4_0.gguf` (703 MB)
  - AUF (Phase 3.7b SOA payload): `Llama-3.2-1B-Instruct.auf` (sha256 `1a1ead0c1f532b26034989deb7dbfece4f5b7ed41b881491cfee857ae014c5ec`, mtime 2026-04-26 00:40)
- **CLI**: `--num-tokens 128 --protected-prefix 4 --prompt "The capital of France is" --threads 6`
- **반복**: N=3/configuration, intra-config sleep 12s, inter-config sleep 40s
- **Instrumentation 활성화**: env `LLMRS_CL_MEM_DIAG=1`

## 1. Instrumentation 추가/변경 파일

| 파일 | 변경 내용 |
|---|---|
| `engine/src/backend/opencl/mod.rs` | `OpenCLBackend`에 `cl_mem_diag_enabled: bool` + `cl_mem_diag: UnsafeCell<HashMap<&'static str, ClMemDiagBucket>>` 필드 추가. 알로케이션 hook (`record_cl_mem_alloc`)을 `convert_q4_0_to_noshuffle` (q_buf, d_buf, q_img), `register_pre_converted_soa` (auf_soa_*), `copy_from` (weight_*_copy)에 삽입. release hook은 `clear_noshuffle_soa_registry`에서 registry 잔여 q/d/img 산입. dump 메서드 `dump_cl_mem_diagnostics(prefix)`. 환경변수 `LLMRS_CL_MEM_DIAG=1` 활성화 시에만 카운팅 동작. |
| `engine/src/bin/generate.rs` | 3 stage 시점에서 `dump_cl_mem_diagnostics` 호출 — (1) 모델 로드 + noshuffle prep 직후, (2) `--force-swap-ratio` 강제 swap 직후, (3) generate 종료 직후. |

추가된 카테고리:
- `weight_q4_aos_copy` / `weight_f16_copy` / `weight_f32_copy` / `weight_other_copy` — `Backend::copy_from` (primary loader, AUF placeholder 등)
- `weight_q4_soa_q` / `weight_q4_soa_d` / `weight_q4_soa_img` — `convert_q4_0_to_noshuffle` (GGUF SOA 경로)
- `auf_soa_q` / `auf_soa_d` / `auf_soa_img` — `register_pre_converted_soa` (AUF SOA bypass)
- `noshuffle_registry_q` / `_d` / `_img` — `clear_noshuffle_soa_registry` 시 registry drop release

원본 GGUF mmap에서 직접 잡은 weight cl_mem (예: `gguf::create_buffer_from_mmap`) 같은 경로는 `copy_from` 통과 시 잡히지만, mmap-direct backed cl_mem (UnifiedBuffer)는 `OpenCLMemory::alloc`을 거치며, 본 작업에서는 그 직접 경로에 hook을 두지 않고 상위 `copy_from`에서 분류하였다. 따라서 본 카운트는 `Backend::copy_from`을 거친 cl_mem의 정확한 분포이다.

## 2. cl_mem 개수 분포 표 (load 직후 + swap 직후)

(N=3 평균, 모든 iteration이 동일 카운트 — 운영 결정적이므로 변동 없음)

### 2-A. Q4 baseline (`q4_0.gguf` 단독)

| 카테고리 | count | alive_bytes |
|---|---:|---:|
| weight_q4_aos_copy (primary loader) | 113 | 695,107,584 |
| weight_q4_soa_q | 113 | 617,873,408 |
| weight_q4_soa_d | 113 | 77,234,176 |
| weight_q4_soa_img (`q_buf` view) | 113 | 617,873,408 |
| weight_f32_copy (norms) | 33 | 270,336 |
| **TOTAL after_noshuffle_prep** | **485** | **2,008,358,912 (1.87 GB)** |

`stage=after_generate`에서 weight_f32_copy +32, weight_other_copy +2 (norm 추가 alloc — embed/lm_head dequant 등) → **TOTAL 519 cl_mem**.

주의 1: `weight_q4_aos_copy` 113은 alloc 시점 카운트. `prepare_noshuffle_buffers(swap_to_placeholder=true)`가 NoshuffleWeightBuffer로 교체하면서 원본 AOS cl_mem 113개 중 SOA 진입한 113개는 사실상 drop됨 ("[NoShuffle] Released original Q4_0 weights … 113 tensors, 662.9 MiB reclaimed" 로그). 본 카운터는 destructor release를 추적하지 않으므로 alive_bytes에 잔존하나, 실제 driver-side cl_mem은 reclaim된다. 따라서 **실제 alive cl_mem ≈ 372 (113×3 SOA + 33 norms)**, 실제 alive bytes ≈ 1.31 GB.

주의 2: `weight_q4_soa_img`는 `image1d_buffer_t` view로, `q_buf`와 메모리를 공유한다. driver-side로는 cl_mem object 1개씩 추가되지만 실제 backing memory는 q_buf와 동일.

### 2-B. Mixed ratio=1.0 (`f16.gguf` primary + `.auf` secondary + `--force-swap-ratio 1.0`)

| 카테고리 | count | alive_bytes |
|---|---:|---:|
| weight_f16_copy (F16 primary loader) | 145 | 2,471,755,776 |
| weight_q4_aos_copy (AUF placeholder) | 112 | 547,356,672 |
| auf_soa_q | 112 | 486,539,264 |
| auf_soa_d | 112 | 60,817,408 |
| auf_soa_img | 112 | 486,539,264 |
| weight_f32_copy (norms) | 33 | 270,336 |
| **TOTAL after_force_swap** | **626** | **4,053,278,720 (3.77 GB)** |

`stage=after_generate`에서 weight_f32_copy +32, weight_other_copy +2 → **TOTAL 660 cl_mem**.

여기서 중요한 것:
- F16 primary 145 cl_mem이 swap 후에도 **그대로 alive**. swap path는 `LayerSlot::swap_weights`로 새 `Arc<LayerWeights>`를 publish하지만, 원본 layer Arc가 다른 곳 (model.layers의 ArcSwap 이전 슬롯, embed_tokens, lm_head 등)에서 잡고 있으면 drop되지 않는다. F16 GGUF의 비-swap 영역 (token embedding, lm_head, output_norm)이 정상 alive이고, **swap된 16 layer × 7 weight = 112 F16 weight도 ArcSwap publish 이후 strong_count > 1로 madvise 우회 (madvise=0.0ms 측정 일치)** → drop되지 않은 채 alive.
- `weight_q4_aos_copy` 112는 `materialise_auf_soa_weight`에서 alloc된 **placeholder cl_mem**. AUF SOA bytes를 담아 GPU에 업로드하지만, GEMV 경로는 노트북 noshuffle SOA registry의 `auf_soa_*`만 읽고 이 placeholder의 데이터는 한 번도 읽지 않는다. 그러나 cl_mem object 자체는 등록 키로 사용되어 alive 유지된다.

### 2-C. 차이 (Mixed - Baseline)

| 항목 | Baseline (a) | Mixed (b) | Δ (b - a) |
|---|---:|---:|---:|
| TOTAL cl_mem object | 485 | 626 | **+141 (+29%)** |
| TOTAL alive_bytes | 2.00 GB | 4.05 GB | **+2.05 GB (+102%)** |
| 실제 alive cl_mem (placeholder/F16 primary 제외) | 372 | 369 | 거의 동일 |
| 실제 GEMV에 사용되는 SOA buffer | 113×3 = 339 | 112×3 = 336 | 거의 동일 |
| 사용되지 않는 alive cl_mem | 0 | **257** (F16 primary 145 + AUF placeholder 112) | **+257** |

**핵심 사실**: GEMV가 실제로 읽는 SOA 데이터 자체의 cl_mem 수는 baseline과 mixed가 거의 동일 (113 vs 112). 차이는 **사용되지 않으나 driver page table에 alive로 등록된 cl_mem 257개**다.

비교 (참고): 본 sprint 가설 시점에 추정한 224 (16 × 7 × 2)는 SOA q+d만 합산한 수치였는데, 실제 측정에서는 SOA q+d+img(공유) + placeholder + F16 잔여까지 합치면 **실제 driver alive cl_mem 차이가 +257**로 나타나 가설보다 더 크다.

llama.cpp 단일 cl_mem 비교: llama.cpp는 모든 weight를 단일 `cl_mem` + 텐서별 offset으로 관리한다 (`ggml_backend_alloc_ctx_tensors_from_buft_impl`). 1B 모델의 경우 ~700 MB 단일 cl_mem 1개. **본 구현은 baseline에서도 372 vs 1, mixed에서는 626 vs 1로 두 자릿수 fragmentation gap.**

## 3. Per-op timing breakdown (`--profile`)

`--profile`은 driver-specific sync 오버헤드 (~54 ms/tok)를 포함하므로 절대값은 inflate되어 있다 (CLAUDE.md 정책). 단, baseline과 mixed는 **동일 sync 오버헤드 기반선**이므로 per-op delta는 fragmentation 영향 시그널로 유효하다.

i2 iteration 비교 (1008 layer-calls 누적):

| op | Q4 baseline (Avg µs/call) | Mixed ratio=1.0 (Avg µs/call) | Δ µs (%) |
|---|---:|---:|---:|
| matmul_qkv | 437 | 534 | **+97 µs (+22%)** |
| matmul_wo | 323 | 323 | 0 |
| matmul_ffn | 1209 | 1150 | -59 (-5%) |
| attention | 245 | 267 | +22 µs (+9%) |
| rms_norm | 516 | 517 | 0 |
| rope | 247 | 247 | 0 |
| kv_update | 196 | 192 | -4 |
| add_assign | 187 | 187 | 0 |
| **TOTAL (per layer-call)** | **3363** | **3422** | **+59 µs (+1.7%)** |

### 핵심 관찰

- **matmul_qkv가 단독으로 +97 µs (+22%) 증가**한다. 다른 matmul (wo, ffn)은 거의 변화가 없다 (0 ~ -5%).
- 이는 fragmentation의 단순한 "전체 cl_mem 개수" 영향이 아니라 **weight access의 첫 entry가 cold하다**는 신호다. matmul_qkv는 layer당 첫 weight load (Q/K/V 3개 weight를 동시에 load) 이므로 driver TLB / texture cache가 cold하다. matmul_wo, ffn은 같은 layer 안에서 후속 access이므로 warm.
- matmul_qkv 증가분 +97 µs × 16 layers = +1.55 ms/tok. 본 측정 N=3 (full TBT) 차이 (Q4 i2 23.64 vs Mixed i2 27.21 = +3.57 ms/tok)의 **약 43%**를 설명한다.
- attention +22 µs × 16 layers = +0.35 ms/tok 추가 설명. 합 +1.9 ms/tok 설명, 실측 +3.6 ms/tok 중 **약 53% 설명**.
- 나머지 ~47%는 본 instrumentation으로는 식별되지 않는 영역 — 추정: KV cache fragmentation (`project_kv_fragmentation` finding +1.32 µs/n_kv × n_kv 누적), F16 primary mmap pages가 alive함으로써 발생하는 OS page cache pressure.

## 4. 가설 판정

**판정: 가설 (c) 복합 원인 — fragmentation은 부분 원인이며 단독으로 -20.7%를 설명하지 못함**

근거:

1. cl_mem 수 차이 (Q4 baseline 485 vs Mixed 626, +141)는 큰 차이가 맞으나, 실제로 GEMV에 의해 읽히는 SOA 데이터의 cl_mem 수는 양쪽 모두 거의 동일 (113 vs 112). **사용되지 않는 cl_mem 257개 (F16 primary 145 + AUF placeholder 112)**가 alive로 남아 있는 것이 진짜 차이.

2. per-op profile은 fragmentation의 **balanced 영향 (모든 matmul/op가 비슷하게 증가)**이 아니라 **matmul_qkv 단독 +22% 증가**를 보인다. 이는 일반 fragmentation 이론이 아니라 specific access pattern (layer 내 첫 weight load가 cold) 시그널.

3. matmul_qkv + attention 증가만으로 -20.7% 중 ~53%만 설명. 나머지는 KV cache fragmentation (이미 알려진 +1.32 µs/n_kv) + OS page cache pressure (alive F16 mmap pages 2.4 GB가 swap 도중에도 page cache에 잔존, 이후 generation의 다른 알로케이션 (KV grow, scratch)에서 page reclaim cost 발생) 같은 부수적 요인의 합.

4. F16 primary GGUF 145 cl_mem이 alive로 유지되는 것은 본 sprint에서 발견된 **새 finding**이다. swap path는 layer weight만 ArcSwap하지만 embed_tokens, lm_head, output_norm은 여전히 primary F16을 가리키고, swap된 16 layer의 F16 weight도 madvise=0.0ms (strong_count > 1 우회)로 실제 RSS reclaim 없이 alive.

## 5. 후속 sprint scope design note

본 진단 결과 **단일 cl_mem 통합 (KV cache fragmentation 사례와 동등한 처치)** 만으로 -20.7% gap 회복은 불가하다. 효과적인 sprint 설계는 다음 두 갈래로 나뉜다.

### Sprint 후보 1 — `WSWAP-5-AUF-PLACEHOLDER-DROP` (권장 P1, S effort)

**목표**: AUF SOA bypass에서 placeholder cl_mem 112개 alloc을 제거.

**현행**: `materialise_auf_soa_weight`가 AUF AOS-크기 placeholder cl_mem을 alloc하고 `register_pre_converted_soa`가 별도 cl_mem 키로 SOA를 등록한다. placeholder는 **registry key로만** 사용된다.

**수정안**: registry 키를 `(layer_idx, subname)` 같은 stable 식별자로 바꾸고, weight tensor가 placeholder cl_mem을 보유하는 대신 `Tensor::buffer()`가 **SOA q_buf 자체**를 가리키도록 한다. matmul_q4_0 lookup 경로 (`b_buf.as_ptr() as usize` 키)를 (layer_idx, subname) 또는 SOA q_buf cl_mem ptr 기반으로 변경.

**영향 범위**:
- `engine/src/backend/opencl/mod.rs` — `register_pre_converted_soa`, `lookup_noshuffle_soa`, `matmul_q4_0` lookup logic
- `engine/src/models/weights/swap_executor.rs` — `materialise_auf_soa_weight` placeholder 제거, SOA buf 직접 wrap
- `engine/src/buffer/noshuffle_weight_buffer.rs` — 이미 q_buf/d_buf wrap 가능, key 노출 메서드 추가

**예상 효과**: 112 cl_mem 제거, ~547 MB alloc traffic 절약. matmul_qkv +97 µs 중 일부 회복 예상 (driver TLB/page table 부담 감소).

**Risk**: registry key 식별자 변경 시 stale lookup race window. ENG-ALG-221 (ratio_generation bump) invariant 유지 필요. INV-130/131 회귀 가드 필수.

**Effort**: S (수일).

### Sprint 후보 2 — `WSWAP-5-PRIMARY-DROP` (P2, M effort)

**목표**: swap 직후 alive로 남은 F16 primary GGUF 145 cl_mem을 explicit drop.

**현행**: `LayerSlot::swap_weights`가 새 `Arc<LayerWeights>`를 publish하고 old Arc는 strong_count 검사 + madvise(MADV_DONTNEED)만 시도. F16 primary는 `model.layers[i].load_weights()` 외에도 `embed_tokens`, `lm_head`, `output_norm`, 그리고 swap 도중 잠시 잡힌 snapshot 들이 strong_count > 1을 만들어 reclaim 안됨.

**수정안**:
1. swap 종료 후 강제로 layer-내 F16 weight Arc를 drop (model 내부의 모든 referencing path를 audit). 
2. cl_mem level에서 explicit `clReleaseMemObject` (Mem `Drop` 통한 자동 release).
3. 또는 secondary mmap (AUF SOA bypass) 도입 시 model을 처음부터 "secondary가 primary"인 양 빌드 — F16 primary 자체를 alloc하지 않음 (lazy primary load).

**영향 범위**:
- `engine/src/models/transformer.rs` — TransformerWeights/LayerSlot 라이프사이클 audit
- `engine/src/models/loader/gguf.rs` — F16 primary 경로 분기
- `engine/src/bin/generate.rs` — load order 변경

**예상 효과**: F16 primary 145 cl_mem (~2.4 GB) 제거. PSS/RSS 감소 + driver page table 부담 추가 감소. 그러나 본 sprint에서 측정된 -20.7% 중 fragmentation 외 요인이라 어느 정도 throughput 회복할지 측정 전 확정 불가.

**Risk**: F16 primary가 lm_head/embed_tokens에 사용되므로 단순 drop 불가. 부분 drop (16 layer × 7 weight만) 시 lm_head 정상성 유지. 정확성 측정 후 진입 결정.

**Effort**: M (1주~). primary 경로 라이프사이클 수정은 광범위.

### Sprint 후보 3 — `WSWAP-5-KV-FRAG-INTEGRATE` (P3, L effort, 조건부)

**조건**: 후보 1을 진행해도 -20.7% 잔여가 큰 경우.

`project_kv_fragmentation.md`의 KV 단일 cl_mem 통합. 16 layer × 2 (K, V) × ratio scope = 56 → 1 + view. KV alloc은 본 swap path와 무관하므로 baseline에서도 동일 효과. matmul_qkv 회복 후 잔여 +0.35 ms/tok (attention +22 µs × 16) 영역을 다룬다. 별도 sprint로 분리.

### 권장 우선순위

1. **WSWAP-5-AUF-PLACEHOLDER-DROP (P1, S)** — 즉시 진행. 효과/effort 비율 가장 좋음.
2. 진행 후 재측정 → 잔여 gap에 따라 후보 2 또는 3 결정.
3. 최종 목표: Phase 4 -20.7%를 -5% 이하로.

## 6. 산출물

- 원시 로그 (TBT): `/tmp/wswap5_diag/results/wswap5/{f16_baseline,q4_baseline,d3_ratio_1}_i{1,2,3}.log` (host)
- 원시 로그 (--profile breakdown): `/tmp/wswap5_diag/results/wswap5_prof/{q4_baseline,d3_ratio_1}_prof_i{1,2}.log` (host)
- 측정 스크립트: `/tmp/wswap5_diag/run_measurements.sh`, `run_profile_breakdown.sh`
- 파서: `/tmp/wswap5_diag/parse_results.sh`, `parse_clmem.sh`

## 7. 결론 요약

| 질문 | 답 |
|---|---|
| cl_mem 개수가 fragmentation 영향을 만드는가 | 부분적으로. 실제 GEMV 사용 cl_mem은 baseline과 mixed가 거의 동일. 차이는 **사용되지 않으나 alive로 남은 cl_mem 257개**. |
| 가설 (a) "fragmentation 주 원인" | **부분 지지** — 직접 영향은 통계적 유의 (matmul_qkv +22%) 있으나 -20.7% 전체를 설명 못함. |
| 가설 (b) "다른 원인" | **부분 지지** — first-weight-access TLB cold 신호. |
| 가설 (c) "복합 원인" | **가장 설득력** — fragmentation + alive-but-unused cl_mem + 알려진 KV fragmentation + OS page cache pressure. |
| -20.7% 회복 가능? | 후보 1 (S effort) 단독으로 일부, 후보 1+2 또는 1+3 조합으로 -5% 이하 달성 시도 가치 있음. |
| INV-122 차단 요인인가 | 아니다. throughput 이슈로 정확성 측정과 무관. 본 진단 후속 sprint와 INV-122는 병렬 진행 가능. |
