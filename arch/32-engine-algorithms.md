# Engine Algorithms -- Architecture

> spec/32-engine-algorithms.md의 구현 상세.

## 코드 매핑

### 3.1 KV Cache Eviction Algorithms

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-010 (H2O) | `engine/src/core/eviction/h2o.rs` | `H2OPolicy::evict()`, `evict_with_scores()`, `evict_with_head_scores()` | compact_keep_positions batch 최적화 |
| ENG-ALG-010 | `engine/src/core/kv_cache.rs:920` | `compact_keep_positions()` — 연속 keep 위치 batch shift | `compact_keep_positions_for_head()` (per-head 변형) |
| ENG-ALG-010 | `engine/src/core/kv_cache.rs:844` | `release_unused_pages()` — eviction 후 물리 메모리 해제 | shrink_to_fit 우선, madvise fallback |
| ENG-ALG-011 (Sliding) | `engine/src/core/eviction/sliding_window.rs` | `SlidingWindowPolicy::evict()` — `prune_prefix()` 호출 | |
| ENG-ALG-011 | `engine/src/core/kv_cache.rs:601` | `prune_prefix(count)` — shift + release_unused_pages | |
| ENG-ALG-012 (D2O) | `engine/src/core/pressure/d2o_handler.rs` | `D2OHandler` — CachePressureHandler 구현체 | H2O eviction + cosine merge + EMA threshold |
| ENG-ALG-012 | `engine/src/core/pressure/d2o_layer_alloc.rs` | `D2OVarianceCollector` — per-layer variance 수집 | Phase B: layer-level dynamic allocation |

### 3.2 KV Cache Quantization (KIVI)

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-020 (Flush) | `engine/src/core/kivi_cache.rs:465` | `flush_residual()` — CPU path | per-channel Key, per-token Value 양자화 |
| ENG-ALG-020 | `engine/src/core/kivi_cache.rs:886` | `flush_residual_gpu()` — GPU path | CPU에서 양자화 후 GPU 저장소에 복사 |
| ENG-ALG-021 (Bit Transition) | `engine/src/core/kivi_cache.rs:587` | `transition_bits(new_bits)` — dequant → requant | Q2/Q4/Q8 전환, `q2_deq_tokens=0` 초기화 |
| ENG-ALG-022 (Incremental Dequant) | `engine/src/core/kivi_cache.rs:727` | `assemble_view()` — CPU path | incremental flush dequant + residual copy |
| ENG-ALG-022 | `engine/src/core/kivi_cache.rs:1160` | `assemble_view_gpu()` — GPU path | |
| ENG-ALG-023 (GPU Mode) | `engine/src/core/kivi_cache.rs` | `KiviCache::new_gpu()` 생성자 | gpu_res_k/v, gpu_attn_k/v, gpu_q2k/v 6종 GPU 버퍼 |

### 3.3 Layer Skip

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-030 (SkipConfig) | `engine/src/core/skip_config.rs:43` | `SkipConfig::uniform_init(num_layers, skip_ratio)` | 홀수 인덱스 우선, layer 0/L-1 보호 |
| ENG-ALG-032 (Layer Importance) | `engine/src/core/qcf/layer_importance.rs:150` | `ImportanceCollector` — prefill 시 `snapshot_before()` → `record_after()` → `build()` | cosine_similarity 기반 |
| ENG-ALG-032 | `engine/src/core/qcf/layer_importance.rs:37` | `ImportanceTable` — `compute_qcf()`, `estimate_qcf_for_count()` | |

### 3.4 QCF 계산

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-041 (Eviction QCF Attn) | `engine/src/core/qcf/eviction_qcf.rs:20` | `compute_eviction_qcf_attn()` | attn * V-norm, per-head → Mean/Defensive 집계 |
| ENG-ALG-042 (Sliding QCF) | `engine/src/core/qcf/eviction_qcf.rs:112` | `compute_sliding_qcf_attn()` | V-norm only, attention score 불필요 |
| ENG-ALG-043 (CAOTE) | `engine/src/core/qcf/eviction_qcf.rs` | `compute_eviction_qcf_caote()` (추정) | amplification * weighted_residual / output_norm |
| ENG-ALG-044 (QCF Attn V2) | `engine/src/core/qcf/eviction_qcf.rs` | attention score only 경량 variant | V-norm 계산 없음 |
| ENG-ALG-045 (KIVI NMSE) | `engine/src/core/qcf/quant_qcf.rs:69` | `compute_flush_qcf()` | NMSE per-block, 0.6K+0.4V 가중 합산 |
| ENG-ALG-046 (KIVI OPR) | `engine/src/core/qcf/quant_qcf.rs:187` | `compute_flush_opr()` | V cache delta_sum / orig_sum |
| ENG-ALG-047 (AWQE) | `engine/src/core/qcf/quant_qcf.rs:316` | `compute_flush_awqe()` | attention-weighted V 양자화 오차, 기본 비활성 |
| ENG-ALG-048 (Layer Skip QCF) | `engine/src/core/qcf/skip_qcf.rs:13` | `SkipQcfTracker` — rejection rate sliding window | `record()`, `current_proxy()` |

### 3.5 RequestQcf -> QcfEstimate

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-050 | **미구현** | EngineCommand enum과 CommandExecutor에 미등록 | 프로토콜 정의만 존재 (MSG-036b, SEQ-095~098) |

### 3.6 DegradationEstimator

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-060 | `engine/src/core/qcf/estimator.rs:55` | `DegradationEstimator` struct | PiecewiseLinear curves + EMA correction |
| ENG-ALG-060 | `engine/src/core/qcf/estimator.rs:136` | `estimate(&QcfMetric) -> f32` | `clamp(base * correction, 0, d_max)` |

### 3.7 madvise / release_unused_pages

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-071 (CPU madvise) | `engine/src/core/kv_cache.rs:1015` | `madvise_dontneed()` — page-aligned MADV_DONTNEED | high_water_pos 기반 범위 제한 |
| ENG-ALG-071 | `engine/src/core/kv_cache.rs:844` | `release_unused_pages()` — shrink_to_fit 우선, madvise fallback | Layout별 (SeqMajor/HeadMajor) 분기 |
| ENG-ALG-072 (shrink_to_fit) | `engine/src/core/kv_cache.rs:363` | `shrink_to_fit()` — `next_power_of_2(current_pos)` 재할당 | dynamic cache (`memory.is_some()`) 전용 |
| ENG-ALG-073 (GPU madvise) | `engine/src/buffer/madviseable_gpu_buffer.rs:15` | `MadviseableGPUBuffer` — `CL_MEM_USE_HOST_PTR` + `is_host_managed()=true` | Adreno pin 문제 시 shrink_to_fit 대안 |

### 3.8 Chunked Prefill

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-080 | `engine/src/bin/generate.rs:101` | CLI: `--prefill-chunk-size` | 0=비활성 |
| ENG-ALG-080 | `engine/src/bin/generate.rs:1311-1341` | chunked prefill 루프 — `logits_last_only=true` per chunk | `model.forward()` 반복 호출 |

### 3.9 CachePressurePipeline

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-091 (Pipeline) | `engine/src/core/pressure/mod.rs:149` | `CachePressurePipeline` — `Vec<PressureStageConfig>` min_level 오름차순 | |
| ENG-ALG-091 | `engine/src/core/pressure/mod.rs:167` | `execute(&mut HandlerContext)` — `min_level <= ctx.pressure_level` 필터 | |
| ENG-ALG-092 (Handler 6종) | `engine/src/core/pressure/eviction_handler.rs` | `EvictionHandler` — 완료 | |
| ENG-ALG-092 | `engine/src/core/pressure/d2o_handler.rs` | `D2OHandler` — 완료 | |
| ENG-ALG-092 | `engine/src/core/pressure/swap_handler.rs` | `SwapHandler` — 완료 | |
| ENG-ALG-092 | `engine/src/core/pressure/quantize_handler.rs` | `QuantizeHandler` — 완료 | |
| ENG-ALG-092 | `engine/src/core/pressure/merge_handler.rs` | `MergeHandler` — stub | |
| ENG-ALG-092 | `engine/src/core/pressure/sparse_handler.rs` | `SparseHandler` — stub | |
| ENG-ALG-093 (EvictionHandler QCF) | `engine/src/core/pressure/eviction_handler.rs` | `compute_and_push_proxy()` — eviction 전 QCF 계산 → qcf_sink push | |

### 3.10 Inference Loop Resilience Checkpoint

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ALG-095 | `engine/src/resilience/executor.rs:152` | `CommandExecutor::poll(&KVSnapshot) -> ExecutionPlan` | 토큰당 최대 1회 |
| ENG-ALG-095 | `engine/src/bin/generate.rs` | decode 루프 내 `poll()` 호출 → plan 필드별 소비 | suspend → evict → switch → skip → quant → throttle → restore |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `d2o.keep_ratio` | f32 | 0.75 | ENG-ALG-012 |
| `d2o.ema_alpha` | f32 | 0.5 | ENG-ALG-012 |
| `d2o.ema_beta` | f32 | 0.5 | ENG-ALG-012 |
| `d2o.protected_prefix` | usize | 4 | ENG-ALG-012 |
| `degradation_estimator.d_max` | f32 | 5.0 | ENG-ALG-060 |
| `degradation_estimator.ema_alpha` | f32 | 0.1 | ENG-ALG-060 |
| `degradation_estimator.curves` | JSON | action별 piecewise-linear | ENG-ALG-060 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--eviction-policy` | eviction 정책 선택 (none/sliding/streaming/h2o/h2o_plus/d2o) | ENG-ALG-010~012 |
| `--h2o-keep-ratio` | H2O heavy hitter 보존 비율 | ENG-ALG-010 |
| `--h2o-decay` | H2O score 지수 감쇠 | ENG-ALG-010 |
| `--d2o-keep-ratio` | D2O heavy hitter 비율 | ENG-ALG-012 |
| `--d2o-ema-alpha` | D2O EMA old-threshold 가중치 | ENG-ALG-012 |
| `--d2o-ema-beta` | D2O EMA new-mean 가중치 | ENG-ALG-012 |
| `--d2o-layer-alloc` | D2O per-layer 동적 할당 | ENG-ALG-012 |
| `--kivi` | KIVI 양자화 활성 | ENG-ALG-020 |
| `--kivi-residual-size` | KIVI residual 버퍼 크기 | ENG-ALG-020 |
| `--skip-ratio` | layer skip 비율 (uniform_init) | ENG-ALG-030 |
| `--skip-layers` | 명시적 skip 레이어 | ENG-ALG-030 |
| `--dump-importance` | importance table 출력 후 종료 | ENG-ALG-032 |
| `--qcf-mode` | QCF proxy 모드 (attn/caote/both) | ENG-ALG-041~043 |
| `--prefill-chunk-size` | chunked prefill 크기 (0=비활성) | ENG-ALG-080 |
| `--eviction-target-ratio` | eviction 보존 비율 | ENG-ALG-095 |