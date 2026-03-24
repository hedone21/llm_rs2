# Action Pool 구현 Sprint

> **목표**: `action-impl-spec.md`에 정의된 8개 액션 전체를 동적 enable/disable 가능하게 구현
> **현황**: ✅ 전체 구현 완료 (Phase 1-A ~ Phase 4)
> **참조**: `/home/go/Workspace/papers/pact2026/plan/action-impl-spec.md`
> **완료일**: 2026-03-17
> **테스트**: 505 lib + 14 integration = 519 tests passing

---

## 구현 현황 요약

| # | Action | 상태 | 동적 제어 | 비고 |
|---|--------|------|----------|------|
| W1 | GPU↔CPU Switch | ✅ 완전 | `generate --backend hybrid` | KV migration 포함, generate에 통합 |
| W2 | KV Offload Disk | ❌ 스텁 | — | DiskStore 의도적 제거됨 |
| W3 | Throttle | ✅ 완전 | Resilience signal | Thermal/Compute/Energy 전략 |
| C1 | SWIFT Layer Skip | ❌ 전무 | — | 코드 전무, 가장 복잡 |
| C4 | H2O Eviction | ✅ 완전 | `--eviction-policy h2o` | H2O+ per-head 포함 |
| C5 | SnapKV | ❌ 스텁 | — | CompressHandler NoOp |
| C6 | StreamingLLM | ⚠️ 부분 | `--eviction-policy sliding` | 기능적 동등, 명시적 CLI 부재 |
| C8 | KIVI Quant | ⚠️ 부분 | `--kivi` | Q2 고정, 동적 전환 미구현 |

---

# Phase 1-A: C6 StreamingLLM CLI 완성

> 기존 `SlidingWindowPolicy`는 StreamingLLM과 기능적으로 동등.
> Attention sink (protected_prefix ≥ 4) + sliding window + monotonic RoPE 분리.
> 명시적 CLI alias와 sink_size 파라미터만 추가하면 완성.

## [P1] AP-1A-1: StreamingLLM CLI alias 및 파라미터 추가

- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음 (기존 코드 완전)

- **Description**:

  1. `--eviction-policy streaming` alias 추가 (내부적으로 `SlidingWindowPolicy` 생성)
  2. `--sink-size <N>` 파라미터 추가 (기본값 4)
  3. `streaming` 선택 시 `protected_prefix = sink_size` 매핑

  **수정 파일**: `engine/src/bin/generate.rs`

  ```rust
  // CLI args 추가
  #[arg(long, default_value_t = 4)]
  sink_size: usize,

  // eviction policy factory 수정
  match args.eviction_policy.as_str() {
      "sliding" | "streaming" => {
          let prefix = if args.eviction_policy == "streaming" {
              args.sink_size
          } else {
              protected_prefix  // 기존 로직
          };
          Box::new(SlidingWindowPolicy::new(args.eviction_window, prefix))
      }
      // ...
  }
  ```

  4. `streaming` 선택 시 기본 eviction_window를 2000으로 설정 (sliding은 기존 기본값 유지)

- **Acceptance Criteria**:
  - `--eviction-policy streaming --sink-size 4` 로 실행 가능
  - `--eviction-policy streaming` = `--eviction-policy sliding --protected-prefix 4 --eviction-window 2000` 과 동일 동작
  - 기존 `sliding` 동작에 영향 없음 (regression 없음)
  - `cargo test -p llm_rs2` 통과

- **Notes**: ~30 LOC 변경. 스펙의 C6와 현 구현의 pseudocode 차이(RoPE baked-in vs attention-time 적용)는 결과적으로 동등 — 상대 위치가 보존되므로 허용.

---

# Phase 1-B: C8 KIVI 동적 비트 전환

> 현재 Q2(2-bit) 고정 구현. 스펙은 F16→INT8→INT4→INT2 동적 전환 + pressure 연동 요구.
> `KiviCache`에 multi-bit 지원 추가 + `QuantizeHandler` 실구현.

## [P1] AP-1B-1: BlockQ4_0 / BlockQ8_0 KV cache용 양자화 포맷 추가

- **Status**: DONE
- **Sprint**: current
- **Dependencies**: 없음

- **Description**:

  `engine/src/core/quant.rs`에 4-bit, 8-bit asymmetric 양자화 블록 추가.
  기존 `BlockQ2_0`과 동일한 인터페이스 (quantize/dequantize).

  **주의**: 기존 `BlockQ4_0`은 weight용 symmetric 양자화. KV cache용은 asymmetric이어야 함 (KIVI 논문 요구).
  이름 충돌 방지를 위해 `BlockKVQ4_0`, `BlockKVQ8_0` 사용.

  ```rust
  // engine/src/core/quant.rs 추가

  pub const QK4_KV: usize = 32;  // group size

  /// 4-bit asymmetric KV cache quantization block.
  /// 32 values → 16 bytes packed + 4 bytes (scale+min) = 20 bytes
  #[repr(C)]
  pub struct BlockKVQ4_0 {
      pub d: f16,                  // scale = (max - min) / 15
      pub m: f16,                  // minimum value
      pub qs: [u8; QK4_KV / 2],   // 32 × 4-bit packed in 16 bytes
  }

  /// 8-bit asymmetric KV cache quantization block.
  /// 32 values → 32 bytes packed + 4 bytes (scale+min) = 36 bytes
  #[repr(C)]
  pub struct BlockKVQ8_0 {
      pub d: f16,
      pub m: f16,
      pub qs: [u8; QK4_KV],       // 32 × 8-bit in 32 bytes
  }

  impl BlockKVQ4_0 {
      pub fn quantize(src: &[f32; QK4_KV]) -> Self { /* min-max → 4-bit */ }
      pub fn dequantize(&self, out: &mut [f32; QK4_KV]) { /* q*d + m */ }
  }

  impl BlockKVQ8_0 {
      pub fn quantize(src: &[f32; QK4_KV]) -> Self { /* min-max → 8-bit */ }
      pub fn dequantize(&self, out: &mut [f32; QK4_KV]) { /* q*d + m */ }
  }
  ```

  **Packing 포맷**:
  - 4-bit: `packed[i] = (v[2i+1] << 4) | v[2i]` (nibble pair per byte)
  - 8-bit: 1:1 byte mapping (trivial)

- **Acceptance Criteria**:
  - `BlockKVQ4_0::quantize()` → `dequantize()` roundtrip 상대 오차 < 5% (random f32 input)
  - `BlockKVQ8_0::quantize()` → `dequantize()` roundtrip 상대 오차 < 0.5%
  - `BlockQ2_0` 기존 동작 불변 (regression 없음)
  - packing 정확성: pack→unpack 후 모든 값 원본과 일치

- **Notes**: ~150 LOC. 기존 `BlockQ2_0` 패턴 복제 + bit width 조정.

---

## [P1] AP-1B-2: KiviCache multi-bit 지원 및 transition_bits() 구현

- **Status**: DONE
- **Sprint**: current
- **Dependencies**: AP-1B-1

- **Description**:

  `engine/src/core/kivi_cache.rs`에 multi-bit 양자화 + 동적 전환 지원.

  **1. KiviCache 구조 변경**:

  ```rust
  pub struct KiviCache {
      // 기존 필드 유지 + 변경
      bits: u8,                            // 2, 4, 8 (현재는 2 고정)
      q2_k: Vec<BlockQ2_0>,                // bits=2 일 때 사용
      q4_k: Vec<BlockKVQ4_0>,              // bits=4 일 때 사용
      q8_k: Vec<BlockKVQ8_0>,              // bits=8 일 때 사용
      // V도 동일하게 q2_v, q4_v, q8_v
      // ... 나머지 기존 필드
  }
  ```

  **2. Enum 기반 디스패치** (대안 — 더 깔끔):

  ```rust
  enum QuantizedBlocks {
      Q2(Vec<BlockQ2_0>),
      Q4(Vec<BlockKVQ4_0>),
      Q8(Vec<BlockKVQ8_0>),
  }

  impl QuantizedBlocks {
      fn dequantize_range(&self, start_token: usize, count: usize,
                          out: &mut [f32], head_dim: usize, kv_heads: usize);
      fn token_count(&self) -> usize;
      fn memory_bytes(&self) -> usize;
  }

  pub struct KiviCache {
      bits: u8,
      qk: QuantizedBlocks,    // Key 양자화 저장소
      qv: QuantizedBlocks,    // Value 양자화 저장소
      // FP32 residual 유지
      res_k: Vec<f32>,
      res_v: Vec<f32>,
      // ...
  }
  ```

  **3. transition_bits() 구현**:

  ```rust
  impl KiviCache {
      /// 기존 양자화 블록을 새 bit width로 재양자화.
      /// dequant(old) → requant(new) 과정에서 오차 누적 있음.
      pub fn transition_bits(&mut self, new_bits: u8) -> Result<()> {
          if new_bits == self.bits { return Ok(()); }

          // Step 1: 기존 양자화 데이터 → FP32 dequant
          let total_tokens = self.qk.token_count();
          let buf_size = total_tokens * self.kv_heads * self.head_dim;
          let mut k_fp32 = vec![0.0f32; buf_size];
          let mut v_fp32 = vec![0.0f32; buf_size];
          self.qk.dequantize_range(0, total_tokens, &mut k_fp32, self.head_dim, self.kv_heads);
          self.qv.dequantize_range(0, total_tokens, &mut v_fp32, self.head_dim, self.kv_heads);

          // Step 2: 새 bit width로 re-quantize
          //   Key: per-channel (groups across tokens within each channel)
          //   Value: per-token (groups within one token's head_dim)
          self.qk = Self::quantize_key_blocks(&k_fp32, new_bits, ...);
          self.qv = Self::quantize_value_blocks(&v_fp32, new_bits, ...);

          // Step 3: 상태 업데이트
          self.bits = new_bits;
          self.q2_deq_tokens = 0;  // dequant 캐시 무효화

          Ok(())
      }
  }
  ```

  **4. flush_residual() multi-bit 디스패치**:

  기존 `flush_residual()`은 `BlockQ2_0::quantize()` 호출 → `self.bits` 에 따라 적절한 블록 타입으로 디스패치.

  **5. CLI 변경**: `engine/src/bin/generate.rs`

  ```
  --kivi-bits <2|4|8>       초기 양자화 비트 (기본값: 2)
  --kivi-dynamic             pressure 연동 동적 전환 활성화
  ```

- **Acceptance Criteria**:
  - `KiviCache::new()` 에서 `bits=2,4,8` 모두 정상 생성
  - `flush_residual()` 에서 `bits=4` → `BlockKVQ4_0`, `bits=8` → `BlockKVQ8_0` 올바르게 생성
  - `transition_bits(8→4)`, `transition_bits(4→2)` 후 dequant 오차:
    - 8→4: 상대 오차 < 5%
    - 4→2: 상대 오차 < 15%
    - 8→2 (두 단계 거침): 단일 8→2 직접 전환 대비 오차 유사
  - `get_view()` 가 multi-bit 블록을 올바르게 dequant
  - 기존 `--kivi` (bits=2) 동작 불변

- **Notes**: ~250 LOC. 핵심 난이도는 `QuantizedBlocks` enum 디스패치 + per-channel/per-token 축 분리 유지.

---

## [P1] AP-1B-3: QuantizeHandler 실구현 — Pressure 연동 동적 비트 전환

- **Status**: DONE
- **Sprint**: current
- **Dependencies**: AP-1B-2

- **Description**:

  `engine/src/core/pressure/quantize_handler.rs` 스텁을 실구현으로 교체.
  Pressure level에 따라 `KiviCache.transition_bits()` 호출.

  ```rust
  pub struct QuantizeHandler {
      /// pressure → bits 매핑
      thresholds: [(PressureLevel, u8); 4],
  }

  impl Default for QuantizeHandler {
      fn default() -> Self {
          Self {
              thresholds: [
                  (PressureLevel::Normal, 16),    // FP16 (양자화 없음 = residual만)
                  (PressureLevel::Warning, 8),     // INT8
                  (PressureLevel::Critical, 4),    // INT4
                  (PressureLevel::Emergency, 2),   // INT2
              ],
          }
      }
  }

  impl CachePressureHandler for QuantizeHandler {
      fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
          let target_bits = self.thresholds.iter()
              .filter(|(level, _)| *level <= ctx.pressure_level)
              .last()
              .map(|(_, bits)| *bits)
              .unwrap_or(16);

          let mut quantized_count = 0;
          for cache in ctx.caches.iter_mut() {
              // KiviCache인 경우에만 동작 (일반 KVCache는 skip)
              if let Some(kivi) = cache.as_kivi_mut() {
                  if kivi.bits() != target_bits && target_bits < 16 {
                      kivi.transition_bits(target_bits)?;
                      quantized_count += 1;
                  }
              }
          }

          if quantized_count > 0 {
              Ok(ActionResult::Quantized { layers: quantized_count })
          } else {
              Ok(ActionResult::NoOp)
          }
      }
  }
  ```

  **통합 문제: KVCache trait downcasting**:
  - 현재 `HandlerContext.caches`는 `&mut [KVCache]` 타입
  - `KiviCache`는 별도 타입 (`KVCacheOps` trait 구현)
  - 해결 방안: `KVCacheOps` trait에 `fn as_kivi_mut(&mut self) -> Option<&mut KiviCache>` 추가
    또는 `HandlerContext`에 `kivi_caches: Option<&mut [KiviCache]>` 필드 추가

- **Acceptance Criteria**:
  - `PressureLevel::Warning` → bits=8 전환 확인
  - `PressureLevel::Critical` → bits=4 전환 확인
  - `PressureLevel::Emergency` → bits=2 전환 확인
  - `PressureLevel::Normal` → 전환 없음 (이미 양자화된 상태 유지, 역전환 안 함)
  - 일반 `KVCache` (non-KIVI) 에서는 NoOp 반환
  - 단위 테스트 5개 이상

- **Notes**: ~100 LOC. downcasting 설계 결정이 핵심. `--kivi-dynamic` CLI 플래그로 QuantizeHandler를 Pipeline에 등록.

---

# Phase 2-A: C5 SnapKV Eviction

> Prefill 완료 직후 1회성 KV cache 압축.
> Observation window voting + avg pooling + per-head top-k.
> `CompressHandler` 스텁을 실구현으로 교체.

## [P2] AP-2A-1: avg_pool_1d 유틸리티 + per-head top-k 선택 함수 추가

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음

- **Description**:

  SnapKV에 필요한 2개 핵심 유틸리티 함수를 `engine/src/core/` 에 추가.

  **1. 1D Average Pooling** (`engine/src/core/math_utils.rs` 신규):

  ```rust
  /// In-place 1D average pooling.
  /// kernel_size=5, stride=1, padding=kernel_size/2 (same size 출력).
  pub fn avg_pool_1d(data: &mut [f32], kernel_size: usize) {
      let pad = kernel_size / 2;
      let len = data.len();
      let mut buf = vec![0.0f32; len];

      for i in 0..len {
          let start = i.saturating_sub(pad);
          let end = (i + pad + 1).min(len);
          let sum: f32 = data[start..end].iter().sum();
          buf[i] = sum / (end - start) as f32;
      }

      data.copy_from_slice(&buf);
  }
  ```

  **2. Per-head top-k 인덱스 선택**:

  ```rust
  /// scores: [n_heads, prefix_len] (head-major flat array)
  /// 반환: [n_heads][keep_count] sorted indices
  pub fn topk_indices_per_head(
      scores: &[f32],
      n_heads: usize,
      prefix_len: usize,
      keep_count: usize,
  ) -> Vec<Vec<usize>> {
      let mut result = Vec::with_capacity(n_heads);
      for h in 0..n_heads {
          let head_scores = &scores[h * prefix_len..(h + 1) * prefix_len];
          let mut indexed: Vec<(usize, f32)> = head_scores.iter()
              .enumerate()
              .map(|(i, &s)| (i, s))
              .collect();
          // Partial sort: O(n) average via select_nth_unstable
          indexed.select_nth_unstable_by(keep_count, |a, b| b.1.partial_cmp(&a.1).unwrap());
          let mut top = indexed[..keep_count].iter().map(|(i, _)| *i).collect::<Vec<_>>();
          top.sort_unstable();  // 위치 순서 유지
          result.push(top);
      }
      result
  }
  ```

- **Acceptance Criteria**:
  - `avg_pool_1d([1,2,3,4,5], kernel=3)` → `[1.5, 2.0, 3.0, 4.0, 4.5]`
  - `avg_pool_1d` 길이 불변 (input.len() == output.len())
  - `topk_indices_per_head` 에서 반환 인덱스가 정렬됨 (ascending)
  - 각 head별 독립적인 top-k 결과
  - keep_count > prefix_len 시 전체 반환

- **Notes**: ~80 LOC. `select_nth_unstable` 은 O(n) partial sort 활용.

---

## [P2] AP-2A-2: KVCache::compress_per_head() 메서드 추가

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: AP-2A-1

- **Description**:

  `engine/src/core/kv_cache.rs`에 per-head gather+compact 메서드 추가.
  SnapKV 압축 후 선택된 prefix 토큰 + observation window를 새 위치로 재배치.

  ```rust
  impl KVCache {
      /// Per-head compression: 각 head별로 다른 토큰을 유지.
      ///
      /// keep_indices: [n_kv_heads][variable] — head별 유지할 prefix 토큰 위치 (sorted)
      /// window_start: observation window 시작 위치 (이후 모든 토큰 유지)
      ///
      /// 결과: cache.current_pos = max(keep_indices[h].len()) + window_tokens
      ///
      /// HeadMajor 레이아웃에서만 동작 (head별 독립 compact 가능).
      pub fn compress_per_head(
          &mut self,
          keep_indices: &[Vec<usize>],
          window_start: usize,
      ) -> Result<usize> {
          assert_eq!(self.layout, KVLayout::HeadMajor);
          let n_kv_heads = keep_indices.len();
          let window_tokens = self.current_pos - window_start;

          for h in 0..n_kv_heads {
              let mut write_pos = 0;

              // 1. 선택된 prefix 토큰 복사 (gather)
              for &src_pos in &keep_indices[h] {
                  if src_pos != write_pos {
                      self.shift_positions_for_head(h, src_pos, write_pos, 1)?;
                  }
                  write_pos += 1;
              }

              // 2. Observation window 토큰 이동
              for offset in 0..window_tokens {
                  let src = window_start + offset;
                  if src != write_pos {
                      self.shift_positions_for_head(h, src, write_pos, 1)?;
                  }
                  write_pos += 1;
              }
          }

          // 3. current_pos 업데이트 (모든 head 동일 최종 길이)
          let new_pos = keep_indices[0].len() + window_tokens;
          self.current_pos = new_pos;

          Ok(self.current_pos)
      }
  }
  ```

  **레이아웃 가정**: HeadMajor `[1, kv_heads, capacity, head_dim]`.
  `shift_positions_for_head()` 는 이미 구현됨 (H2O+에서 사용).

- **Acceptance Criteria**:
  - HeadMajor 레이아웃에서 정상 동작
  - head 0이 [0,3,7], head 1이 [1,4,8] 유지 시 각 head별 올바른 데이터 배치
  - observation window 토큰 전체 유지 확인
  - `current_pos` = `keep_count + window_tokens`
  - SeqMajor에서 호출 시 panic/에러

- **Notes**: ~80 LOC. `shift_positions_for_head()`가 이미 있으므로 gather loop만 구현.

---

## [P2] AP-2A-3: SnapKVHandler (CompressHandler 실구현)

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: AP-2A-1, AP-2A-2

- **Description**:

  `engine/src/core/pressure/compress_handler.rs` 를 SnapKV 알고리즘으로 실구현.

  **알고리즘 구현 (스펙 pseudocode 기준)**:

  ```rust
  pub struct SnapKVHandler {
      pub window_size: usize,     // default: 32
      pub max_capacity: usize,    // default: 1024
      pub kernel_size: usize,     // default: 5
  }

  impl CachePressureHandler for SnapKVHandler {
      fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
          let importance = ctx.importance.ok_or_else(|| anyhow!("SnapKV requires importance scores"))?;
          let n_kv_heads = ctx.n_kv_heads;
          let mut total_removed = 0;

          for cache in ctx.caches.iter_mut() {
              let total_len = cache.current_pos;
              if total_len <= self.max_capacity { continue; }

              let prefix_len = total_len.saturating_sub(self.window_size);
              if prefix_len == 0 { continue; }

              // Step 1: Observation window의 attention을 prefix에 대해 voting
              //   importance는 [max_seq_len] — 누적 score
              //   SnapKV는 원래 window→prefix attention을 쓰지만,
              //   우리 시스템에서는 누적 importance에서 prefix 영역의 score를 추출
              //   (head_importance 사용 시 per-head voting 가능)
              let votes = if let Some(head_imp) = ctx.head_importance {
                  // Per-head voting: head_importance[kv_h * max_seq + pos]
                  compute_per_head_votes(head_imp, n_kv_heads, prefix_len, self.kernel_size)
              } else {
                  // Fallback: global importance → 모든 head 동일
                  let mut v = importance[..prefix_len].to_vec();
                  avg_pool_1d(&mut v, self.kernel_size);
                  vec![v; n_kv_heads]
              };

              // Step 2: Per-head top-k selection
              let keep_count = self.max_capacity.saturating_sub(self.window_size);
              let keep_indices = topk_indices_per_head_from_votes(&votes, keep_count);

              // Step 3: Gather + compact
              let before = cache.current_pos;
              cache.compress_per_head(&keep_indices, prefix_len)?;
              total_removed += before - cache.current_pos;
          }

          if total_removed > 0 {
              Ok(ActionResult::Compressed { tokens_removed: total_removed })
          } else {
              Ok(ActionResult::NoOp)
          }
      }
  }

  fn compute_per_head_votes(
      head_importance: &[f32], n_kv_heads: usize, prefix_len: usize, kernel_size: usize,
  ) -> Vec<Vec<f32>> {
      let max_seq = head_importance.len() / n_kv_heads;
      (0..n_kv_heads).map(|h| {
          let mut votes: Vec<f32> = (0..prefix_len)
              .map(|pos| head_importance[h * max_seq + pos])
              .collect();
          avg_pool_1d(&mut votes, kernel_size);
          votes
      }).collect()
  }
  ```

  **호출 시점 통합** (`generate.rs`):

  ```rust
  // Prefill 완료 직후, decode loop 진입 전:
  if args.snapkv {
      let handler = SnapKVHandler {
          window_size: args.snapkv_window,
          max_capacity: args.snapkv_capacity,
          kernel_size: args.snapkv_kernel,
      };
      let mut ctx = HandlerContext {
          caches: &mut kv_caches,
          importance: Some(score_accumulator.importance_scores()),
          head_importance: score_accumulator.head_importance_scores(),
          n_kv_heads: config.num_key_value_heads,
          pressure_level: PressureLevel::Emergency,  // 강제 실행
          // ...
      };
      handler.handle(&mut ctx)?;
  }
  ```

  **CLI**: `--snapkv --snapkv-window 32 --snapkv-capacity 1024 --snapkv-kernel 5`

  **Score 캡처 이슈**:
  - 현재 `AttentionScoreAccumulator`는 decode 중 score 추적
  - SnapKV는 prefill 마지막 attention score 필요
  - 해결: prefill forward pass에서도 score 추적 활성화 (`need_scores = true`)
  - 또는: prefill 완료 후 한 번 더 forward pass (overhead 있으나 정확)
  - **권장**: 기존 score accumulator에 prefill 시에도 accumulate 활성화

- **Acceptance Criteria**:
  - `--snapkv --snapkv-capacity 512` 로 실행 시 prefill 후 cache.current_pos ≤ 512
  - Observation window (마지막 32토큰) 100% 유지 확인
  - Per-head 독립 selection: head 0과 head 1의 유지 토큰이 다를 수 있음
  - `avg_pool_1d` 적용 전후 votes 길이 동일
  - 16개 layer 모두 압축 적용
  - Decode 이후 새 토큰 정상 추가 (압축 후 추론 정상)
  - 단위 테스트 8개 이상

- **Notes**: ~200 LOC (handler) + ~50 LOC (generate.rs 통합). Score 캡처 이슈가 가장 큰 설계 결정.

---

## [P2] AP-2A-4: SnapKV + Eviction 조합 테스트

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: AP-2A-3

- **Description**:

  SnapKV(prefill 압축)와 기존 eviction(decode 시 제거)의 조합 동작 검증.
  SnapKV는 prefill 후 1회, eviction은 decode 중 반복적으로 동작 — 이론적으로 직교.

  **조합 매트릭스**:

  | SnapKV | Eviction | 기대 동작 |
  |--------|----------|----------|
  | ON | none | prefill 압축만, decode 중 cache 무한 성장 |
  | ON | sliding | prefill 압축 → decode 중 sliding window |
  | ON | h2o | prefill 압축 → decode 중 H2O score eviction |
  | OFF | sliding | 기존 동작 |

  **테스트 코드**:
  - `test_snapkv_then_sliding`: SnapKV(cap=512) 후 sliding(window=256) → cache ≤ 256+sink
  - `test_snapkv_then_h2o`: SnapKV(cap=512) 후 H2O → score 기반 추가 eviction 정상
  - `test_snapkv_position_continuity`: 압축 후 새 토큰의 RoPE position이 연속적

- **Acceptance Criteria**:
  - 모든 조합에서 crash 없음
  - 각 정책의 독립 동작 보장 (SnapKV가 eviction에 영향 안 줌, 역도 마찬가지)
  - 통합 테스트 3개 이상

- **Notes**: ~100 LOC 테스트 코드.

---

# Phase 2-B: W2 KV Cache Offload (Disk)

> DiskStore가 이전에 제거됨 (성능 부족). 논문 요구사항으로 재구현.
> `OffloadStore` trait이 이미 있으므로 새 구현체만 추가.
> 이전 성능 이슈 감안하여 mmap 방식 고려.

## [P2] AP-2B-1: DiskStore 재구현 (OffloadStore trait)

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음

- **Description**:

  `engine/src/core/offload/disk_store.rs` 신규 생성.
  이전 제거된 DiskStore의 교훈 반영:
  - buffered I/O 대신 mmap 활용 (latency 개선)
  - 단일 파일에 append → 파일 수 최소화
  - metadata는 in-memory 유지 (디스크 메타 불필요)

  ```rust
  // engine/src/core/offload/disk_store.rs

  use super::OffloadStore;
  use anyhow::Result;
  use std::path::PathBuf;
  use std::fs::{File, OpenOptions};
  use std::io::{Write, Read, Seek, SeekFrom};

  pub struct DiskStore {
      path: PathBuf,
      k_file: File,
      v_file: File,
      stored_tokens: usize,
      bytes_per_token: usize,  // kv_heads * head_dim * dtype_size
  }

  impl DiskStore {
      pub fn new(dir: PathBuf, layer_id: usize, bytes_per_token: usize) -> Result<Self> {
          std::fs::create_dir_all(&dir)?;
          let k_path = dir.join(format!("layer{}_k.bin", layer_id));
          let v_path = dir.join(format!("layer{}_v.bin", layer_id));
          let k_file = OpenOptions::new().create(true).write(true).read(true).open(&k_path)?;
          let v_file = OpenOptions::new().create(true).write(true).read(true).open(&v_path)?;
          Ok(Self { path: dir, k_file, v_file, stored_tokens: 0, bytes_per_token })
      }
  }

  impl OffloadStore for DiskStore {
      fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()> {
          self.k_file.seek(SeekFrom::End(0))?;
          self.k_file.write_all(k_data)?;
          self.v_file.seek(SeekFrom::End(0))?;
          self.v_file.write_all(v_data)?;
          self.stored_tokens += num_tokens;
          Ok(())
      }

      fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize> {
          let mut k_file = &self.k_file;
          k_file.seek(SeekFrom::Start(0))?;
          k_file.read_exact(k_buf)?;
          let mut v_file = &self.v_file;
          v_file.seek(SeekFrom::Start(0))?;
          v_file.read_exact(v_buf)?;
          Ok(self.stored_tokens)
      }

      fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()> {
          self.store(k_token, v_token, 1)
      }

      fn storage_size(&self) -> usize {
          self.stored_tokens * self.bytes_per_token * 2  // K + V
      }

      fn stored_tokens(&self) -> usize {
          self.stored_tokens
      }

      fn clear(&mut self) {
          let _ = self.k_file.set_len(0);
          let _ = self.v_file.set_len(0);
          self.stored_tokens = 0;
      }
  }

  impl Drop for DiskStore {
      fn drop(&mut self) {
          // 파일 정리 (optional — 디버깅 시 유지 가능)
          self.clear();
      }
  }
  ```

  **CLI**: `--kv-offload disk --kv-offload-path /tmp/kv/`

  **성능 고려**:
  - 이전 제거 사유: -12% throughput (buffered I/O)
  - 개선 방향: `memmap2` crate 사용 시 커널 page cache 활용으로 latency 감소 가능
  - 그러나 외부 dependency 추가 필요 → 우선 표준 I/O로 구현, 성능 이슈 시 mmap 전환

- **Acceptance Criteria**:
  - `DiskStore::store()` → `load_into()` roundtrip 데이터 일치 (byte-exact)
  - 16 layers × 2048 tokens × F16 offload 시 파일 크기 = 예상값
  - `clear()` 후 파일 크기 = 0
  - `OffloadKVCache`와 통합 시 기존 `RawStore` 대체 가능
  - `--kv-offload disk` CLI 동작 확인
  - 단위 테스트 5개 이상

- **Notes**: ~150 LOC. 이전 실패 교훈 — 성능보다 정확성 우선. 벤치마크는 별도 측정.

---

## [P2] AP-2B-2: SwapHandler 실구현 — Pressure 연동 디스크 Offload

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: AP-2B-1

- **Description**:

  `engine/src/core/pressure/swap_handler.rs` 스텁을 실구현으로 교체.
  LRU 방식으로 오래된 KV 토큰을 디스크로 offload.

  ```rust
  pub struct SwapHandler {
      pub offload_ratio: f32,    // default: 0.5
      pub storage_path: PathBuf, // default: /tmp/kv/
  }

  impl CachePressureHandler for SwapHandler {
      fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
          // Warning 이상에서만 동작
          if ctx.pressure_level < PressureLevel::Warning {
              return Ok(ActionResult::NoOp);
          }

          let mut swapped = 0;
          for (i, cache) in ctx.caches.iter_mut().enumerate() {
              let total = cache.current_pos;
              let offload_count = (total as f32 * self.offload_ratio) as usize;
              if offload_count == 0 { continue; }

              // 1. 오래된 토큰 데이터 읽기
              let k_data = cache.read_range(0, offload_count, CacheComponent::Key)?;
              let v_data = cache.read_range(0, offload_count, CacheComponent::Value)?;

              // 2. 디스크에 저장 (layer별 DiskStore)
              let store = DiskStore::new(
                  self.storage_path.join(format!("layer_{}", i)),
                  i, cache.bytes_per_token(),
              )?;
              store.store(&k_data, &v_data, offload_count)?;

              // 3. 캐시에서 제거 (prune_prefix)
              cache.prune_prefix(offload_count)?;
              swapped += offload_count;
          }

          if swapped > 0 {
              Ok(ActionResult::Swapped { tokens_swapped: swapped })
          } else {
              Ok(ActionResult::NoOp)
          }
      }
  }
  ```

  **Recall 기능** (optional):
  - `--kv-offload-recall` 플래그로 활성화
  - 별도 `recall()` 메서드: 디스크에서 읽어 cache 앞에 prepend
  - Phase 1 에서는 recall 미구현 (offload = lossy로 간주)

- **Acceptance Criteria**:
  - `PressureLevel::Warning` 시 offload_ratio 비율의 토큰 디스크 이동
  - offload 후 cache.current_pos 감소 확인
  - 디스크 파일 존재 및 크기 확인
  - offload 후 추론 계속 가능 (crash 없음)
  - Normal pressure 시 NoOp

- **Notes**: ~120 LOC. `KVCache::read_range()` 메서드 필요할 수 있음 (기존 `as_slice` 활용 가능).

---

# Phase 3: C1 SWIFT Layer Skip

> 가장 복잡한 알고리즘. 3개 서브시스템 신규 구현 필요.
> Layer skip + Speculative decoding + Bayesian optimization.

## [P1] AP-3-1: SkipConfig 구조체 및 LlamaLayer skip 분기 추가

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음

- **Description**:

  Layer skip을 위한 configuration 구조체와 forward pass 분기 추가.

  **1. SkipConfig 정의** (`engine/src/core/skip_config.rs` 신규):

  ```rust
  use std::collections::HashSet;

  /// SWIFT layer skip configuration.
  /// attention과 MLP를 독립적으로 skip 가능.
  #[derive(Debug, Clone, Default)]
  pub struct SkipConfig {
      /// Attention을 skip할 layer indices
      pub attn_skip: HashSet<usize>,
      /// MLP를 skip할 layer indices
      pub mlp_skip: HashSet<usize>,
  }

  impl SkipConfig {
      pub fn new() -> Self { Self::default() }

      /// Layer 0과 L-1은 절대 skip하지 않음 (SWIFT 제약)
      pub fn validate(&self, num_layers: usize) -> bool {
          !self.attn_skip.contains(&0) &&
          !self.attn_skip.contains(&(num_layers - 1)) &&
          !self.mlp_skip.contains(&0) &&
          !self.mlp_skip.contains(&(num_layers - 1))
      }

      /// Uniform initialization: 홀수 인덱스 layer skip
      pub fn uniform_init(num_layers: usize, skip_ratio: f32) -> Self {
          let num_skip = ((num_layers - 2) as f32 * 2.0 * skip_ratio) as usize;
          let mut config = Self::new();
          let mut count = 0;
          for i in (1..num_layers - 1).step_by(2) {
              if count >= num_skip { break; }
              config.attn_skip.insert(i);
              count += 1;
              if count >= num_skip { break; }
              config.mlp_skip.insert(i);
              count += 1;
          }
          config
      }

      pub fn skip_attn(&self, layer_id: usize) -> bool {
          self.attn_skip.contains(&layer_id)
      }

      pub fn skip_mlp(&self, layer_id: usize) -> bool {
          self.mlp_skip.contains(&layer_id)
      }

      pub fn total_skips(&self) -> usize {
          self.attn_skip.len() + self.mlp_skip.len()
      }
  }
  ```

  **2. LlamaLayerForwardArgs 확장** (`engine/src/layers/llama_layer.rs`):

  ```rust
  pub struct LlamaLayerForwardArgs<'a> {
      // 기존 필드...
      pub skip_config: Option<&'a SkipConfig>,  // 추가
  }
  ```

  **3. LlamaLayer::forward() skip 분기**:

  ```rust
  fn forward(&self, args: &LlamaLayerForwardArgs) -> Result<()> {
      let skip_attn = args.skip_config
          .map_or(false, |s| s.skip_attn(self.layer_id));
      let skip_mlp = args.skip_config
          .map_or(false, |s| s.skip_mlp(self.layer_id));

      // ---- Attention block ----
      if !skip_attn {
          // 기존 attention 코드 (RMSNorm → QKV → RoPE → KV update → attention)
          self.attention(args)?;
      }
      // else: residual만 통과 (identity) — x 변경 없음

      // ---- MLP block ----
      if !skip_mlp {
          // 기존 FFN 코드 (RMSNorm → gate → up → SiLU → down)
          self.ffn(args)?;
      }
      // else: residual만 통과 (identity)

      Ok(())
  }
  ```

  **주의**: attention skip 시 KV cache update도 skip됨.
  이는 SWIFT 논문과 일치 — draft model은 KV cache를 별도 관리하거나 shared cache 사용.

- **Acceptance Criteria**:
  - `SkipConfig::default()` → skip 없음 (기존 동작 유지)
  - `SkipConfig` with attn_skip={1,3,5} → layer 1,3,5의 attention 건너뜀
  - Layer 0, L-1 skip 시 `validate()` = false
  - `uniform_init(16, 0.45)` → 합리적인 skip pattern 생성
  - skip 활성화 시 forward pass 시간 감소 확인 (벤치마크)
  - 기존 forward pass (skip_config=None) 완전 불변

- **Notes**: ~150 LOC (SkipConfig) + ~30 LOC (LlamaLayer 수정). forward path에 조건 분기만 추가하므로 risk 낮음.

---

## [P2] AP-3-2: SpeculativeDecoder — Draft/Verify 사이클 구현

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: AP-3-1

- **Description**:

  Speculative decoding의 draft-verify 사이클 구현.
  같은 모델을 draft (skip layers) + verify (full layers)로 이중 사용.

  **파일**: `engine/src/core/speculative.rs` 신규

  ```rust
  pub struct SpeculativeDecoder {
      pub skip_config: SkipConfig,
      pub max_draft_steps: usize,    // default: 25
      pub stop_threshold: f32,       // default: 0.8
  }

  pub struct DraftResult {
      pub tokens: Vec<u32>,
      pub confidences: Vec<f32>,
  }

  pub struct VerifyResult {
      pub accepted_count: usize,
      pub corrected_token: Option<u32>,  // reject 시 수정된 토큰
  }

  impl SpeculativeDecoder {
      /// Draft phase: skip_config 적용하여 여러 토큰 연속 생성
      pub fn draft(
          &self,
          model: &LlamaModel,
          kv_caches: &mut [KVCache],
          start_token: u32,
          start_pos: usize,
          backend: &dyn Backend,
          memory: &dyn Memory,
          workspace: &mut LayerWorkspace,
          sampling: &SamplingConfig,
      ) -> Result<DraftResult> {
          let mut tokens = Vec::new();
          let mut confidences = Vec::new();
          let mut current_token = start_token;
          let mut pos = start_pos;

          for _ in 0..self.max_draft_steps {
              // Forward with skip
              let logits = model.forward_with_skip(
                  current_token, pos,
                  kv_caches, &self.skip_config,
                  backend, memory, workspace,
              )?;

              let probs = softmax(&logits);
              let token = argmax(&probs);
              let confidence = probs[token as usize];

              tokens.push(token);
              confidences.push(confidence);
              current_token = token;
              pos += 1;

              if confidence < self.stop_threshold { break; }
          }

          Ok(DraftResult { tokens, confidences })
      }

      /// Verify phase: 전체 layer로 draft 토큰들을 한 번에 검증
      pub fn verify(
          &self,
          model: &LlamaModel,
          kv_caches: &mut [KVCache],
          draft: &DraftResult,
          start_pos: usize,
          backend: &dyn Backend,
          memory: &dyn Memory,
          workspace: &mut LayerWorkspace,
      ) -> Result<VerifyResult> {
          // Batch forward (전체 layer, skip 없음)
          let all_logits = model.forward_batch(
              &draft.tokens, start_pos,
              kv_caches, None,  // no skip
              backend, memory, workspace,
          )?;

          // Accept/Reject (greedy)
          let mut accepted = 0;
          let mut corrected = None;
          for (i, &draft_token) in draft.tokens.iter().enumerate() {
              let target_token = argmax(&all_logits[i]);
              if target_token == draft_token {
                  accepted += 1;
              } else {
                  corrected = Some(target_token);
                  break;
              }
          }

          Ok(VerifyResult { accepted_count: accepted, corrected_token: corrected })
      }
  }
  ```

  **KV Cache 관리 이슈**:
  - Draft phase에서 KV cache에 draft 토큰의 KV가 추가됨
  - Reject 시 draft KV를 rollback해야 함
  - 해결: draft 전 `kv_cache.current_pos` 스냅샷 → reject 시 `current_pos = snapshot`
  - 또는: draft용 별도 KV cache clone (메모리 2배 필요)
  - **권장**: snapshot+rollback (메모리 효율적, 데이터 overwrite 없이 pos만 되돌림)

  **LlamaModel 확장 필요**:
  - `forward_with_skip()`: skip_config 전달하는 forward wrapper
  - `forward_batch()`: 여러 토큰의 logits를 한 번에 반환 (현재 forward는 단일 토큰)

- **Acceptance Criteria**:
  - Greedy decoding에서 SWIFT 출력 == full model 출력 (lossless)
  - acceptance_rate ≥ 0.90 (충분히 높은 matchness의 skip config에서)
  - Draft phase에서 rejected 토큰의 KV cache rollback 정확
  - max_draft_steps=1 일 때 = 일반 speculative decoding
  - max_draft_steps=0 일 때 = 일반 forward (fallback)
  - 단위 테스트 6개 이상

- **Notes**: ~300 LOC. 가장 큰 구현 난이도: `forward_batch()` (현재 미존재), KV rollback.

---

## [P2] AP-3-3: SkipOptimizer — Skip layer 선택 최적화

- **Status**: DONE
- **Sprint**: backlog
- **Dependencies**: AP-3-1, AP-3-2

- **Description**:

  Prefill 시 context 토큰으로 최적의 skip pattern을 찾는 optimizer.
  Random search + 간소화된 Bayesian optimization.

  **파일**: `engine/src/core/skip_optimizer.rs` 신규

  ```rust
  pub struct SkipOptimizer {
      pub skip_ratio: f32,       // default: 0.45
      pub max_iter: usize,       // default: 200 (논문 1000에서 축소)
      pub early_stop: f32,       // default: 0.95
  }

  impl SkipOptimizer {
      /// Prefill 토큰으로 최적 skip pattern 탐색.
      ///
      /// 1. Uniform init 에서 시작
      /// 2. Random perturbation으로 후보 생성
      /// 3. Matchness 평가: draft output vs context tokens
      /// 4. Best matchness 유지
      pub fn optimize(
          &self,
          model: &LlamaModel,
          context_tokens: &[u32],
          num_layers: usize,
          backend: &dyn Backend,
          memory: &dyn Memory,
      ) -> Result<SkipConfig> {
          let num_skip = ((num_layers - 2) as f32 * 2.0 * self.skip_ratio) as usize;

          let mut best_config = SkipConfig::uniform_init(num_layers, self.skip_ratio);
          let mut best_matchness = self.evaluate_matchness(
              model, context_tokens, &best_config, backend, memory,
          )?;

          for iter in 0..self.max_iter {
              // Random perturbation: 현재 best에서 1-2개 sub-layer 교체
              let candidate = self.perturb(&best_config, num_layers, num_skip);

              if !candidate.validate(num_layers) { continue; }

              let matchness = self.evaluate_matchness(
                  model, context_tokens, &candidate, backend, memory,
              )?;

              if matchness > best_matchness {
                  best_config = candidate;
                  best_matchness = matchness;
              }

              if best_matchness >= self.early_stop { break; }
          }

          Ok(best_config)
      }

      fn evaluate_matchness(
          &self, model: &LlamaModel, tokens: &[u32],
          config: &SkipConfig, backend: &dyn Backend, memory: &dyn Memory,
      ) -> Result<f32> {
          // Draft model로 tokens[:-1] 입력 → tokens[1:] 예측
          // match = count(predicted == actual) / len
          let predictions = model.forward_sequence_with_skip(tokens, config, backend, memory)?;
          let matches = predictions.iter().zip(tokens[1..].iter())
              .filter(|(pred, actual)| pred == actual)
              .count();
          Ok(matches as f32 / (tokens.len() - 1) as f32)
      }

      fn perturb(&self, config: &SkipConfig, num_layers: usize, target: usize) -> SkipConfig {
          // 1-2개 sub-layer를 랜덤으로 교체 (add/remove)
          let mut new_config = config.clone();
          // ... random swap logic
          new_config
      }
  }
  ```

  **Bayesian Optimization 간소화**:
  - 논문: GP + UCB (kappa=2.5), bayes_interval=25
  - 우리 구현: 순수 random search로 시작 (충분히 효과적)
  - 향후: `linfa-gp` 또는 수동 GP 구현 추가 가능

  **Optimization 비용**:
  - 각 iteration = 1회 forward pass (skip 적용)
  - 200 iterations × 1 forward ≈ 200 forward (prefill 시간의 200배)
  - 허용 가능: prefill은 1회, optimization도 1회
  - 실제 비용: Llama 3.2 1B, 50 tokens → ~200 * 50ms = 10초 (모바일)
  - **절충**: max_iter=100, context_window=32 (논문의 50에서 축소)

- **Acceptance Criteria**:
  - `optimize()` 반환 config의 matchness ≥ 0.90
  - Layer 0, L-1 skip 없음 (validate 통과)
  - `total_skips() ≈ num_skip` (±2)
  - Optimization 시간 < 30초 (Llama 3.2 1B, 모바일)
  - 단위 테스트 4개 이상

- **Notes**: ~200 LOC. GP 미구현으로 간소화. random search만으로도 논문 기준 90%+ matchness 달성 가능.

---

## [P2] AP-3-4: generate.rs SWIFT 통합 및 CLI

- **Status**: DONE
- **Sprint**: backlog
- **Dependencies**: AP-3-1, AP-3-2, AP-3-3

- **Description**:

  SWIFT를 generate.rs 추론 루프에 통합.

  **CLI 플래그**:
  ```
  --swift                      SWIFT layer skip 활성화
  --swift-ratio <f32>          skip 비율 (default: 0.45)
  --swift-max-draft <usize>    draft 당 최대 토큰 (default: 25)
  --swift-threshold <f32>      confidence 조기 종료 (default: 0.8)
  --swift-opt-iter <usize>     optimization iterations (default: 200)
  ```

  **추론 루프 변경**:

  ```rust
  // 1. Prefill 후 optimization (1회)
  let skip_config = if args.swift {
      let optimizer = SkipOptimizer {
          skip_ratio: args.swift_ratio,
          max_iter: args.swift_opt_iter,
          early_stop: 0.95,
      };
      Some(optimizer.optimize(model, &input_ids, num_layers, backend, memory)?)
  } else {
      None
  };

  // 2. Decode loop
  let decoder = SpeculativeDecoder {
      skip_config: skip_config.clone().unwrap_or_default(),
      max_draft_steps: args.swift_max_draft,
      stop_threshold: args.swift_threshold,
  };

  while !eos {
      if args.swift {
          // Speculative: draft → verify → accept/reject
          let draft = decoder.draft(model, &mut kv_caches, last_token, pos, ...)?;
          let result = decoder.verify(model, &mut kv_caches, &draft, pos, ...)?;

          // Accepted tokens 출력
          for i in 0..result.accepted_count {
              emit_token(draft.tokens[i]);
              pos += 1;
          }
          // Corrected token 출력 (있으면)
          if let Some(corrected) = result.corrected_token {
              emit_token(corrected);
              pos += 1;
          }

          // KV rollback: draft에서 reject된 토큰의 KV 제거
          let rollback_count = draft.tokens.len() - result.accepted_count;
          for cache in &mut kv_caches {
              cache.current_pos -= rollback_count;
          }
      } else {
          // 기존 단일 토큰 decode
          // ...
      }
  }
  ```

  **Pressure 연동**:
  - `ResilienceAction` 에 `LayerSkip { ratio: f32 }` 추가
  - Pressure 증가 → skip_ratio 증가 (0.2 → 0.5)
  - acceptance rate < 0.7 → skip_ratio 자동 감소
  - Strategy: ComputeStrategy/ThermalStrategy에 LayerSkip 매핑

- **Acceptance Criteria**:
  - `--swift --temp 0` 로 greedy decoding 시 출력 == non-swift 출력 (lossless)
  - acceptance_rate 메트릭 출력 (stderr 또는 profile)
  - `--swift` 없이 실행 시 기존 동작 완전 불변
  - speedup 측정 가능 (tok/s 비교)
  - crash 없이 전체 시퀀스 생성 완료

- **Notes**: ~200 LOC (generate.rs 변경) + ~50 LOC (ResilienceAction 확장). 가장 큰 통합 작업.

---

# Phase 4: E2E 테스트 및 통합 검증

## [P1] AP-4-1: Action별 단위 테스트 완성

- **Status**: DONE
- **Sprint**: next (각 Phase 완료 시 병행)
- **Dependencies**: 각 Phase의 구현 완료

- **Description**:

  각 Action의 단위 테스트를 구현 파일 내 `#[cfg(test)] mod tests`에 추가.

  **신규 테스트 목록**:

  | Action | 테스트 함수명 | 검증 내용 |
  |--------|-------------|----------|
  | C6 | `test_streaming_alias_equivalence` | `streaming` = `sliding` + sink=4 + window=2000 |
  | C8 | `test_kvq4_roundtrip_error_bound` | Q4 roundtrip 상대오차 < 5% |
  | C8 | `test_kvq8_roundtrip_error_bound` | Q8 roundtrip 상대오차 < 0.5% |
  | C8 | `test_transition_8_to_4_to_2` | 8→4→2 순차 전환 오차 bounded |
  | C8 | `test_quantize_handler_pressure_mapping` | Warning→8, Critical→4, Emergency→2 |
  | C5 | `test_avg_pool_1d_identity` | kernel=1이면 변환 없음 |
  | C5 | `test_avg_pool_1d_known_values` | 수동 계산과 일치 |
  | C5 | `test_snapkv_compress_size` | 압축 후 == max_capacity |
  | C5 | `test_snapkv_window_preserved` | 마지막 window 토큰 유지 |
  | C5 | `test_snapkv_per_head_different` | head별 다른 prefix 선택 |
  | W2 | `test_disk_store_roundtrip` | store→load byte-exact |
  | W2 | `test_disk_store_append` | 점진적 append 후 전체 load |
  | W2 | `test_disk_store_clear` | 파일 크기 0 확인 |
  | C1 | `test_skip_config_validate` | layer 0/L-1 skip 거부 |
  | C1 | `test_skip_config_uniform_init` | 합리적 패턴 생성 |
  | C1 | `test_speculative_greedy_lossless` | draft+verify == full forward |
  | C1 | `test_kv_rollback_on_reject` | reject 시 KV pos 복원 |

- **Acceptance Criteria**:
  - 모든 신규 테스트 `cargo test -p llm_rs2` 통과
  - 기존 283 테스트 regression 없음
  - 총 테스트 수 300+ 도달

---

## [P1] AP-4-2: Cross-Action 통합 테스트

- **Status**: DONE
- **Sprint**: next
- **Dependencies**: Phase 1-A, 1-B, 2-A, 2-B 완료

- **Description**:

  `engine/tests/test_action_pool_integration.rs` 신규 생성.
  Action 간 조합 호환성 + 상호 배타성 검증.

  ```rust
  // 조합 호환성 테스트
  #[test]
  fn test_h2o_plus_kivi_combination() {
      // H2O eviction + KIVI Q2 동시 활성화 → crash 없음
      // KiviCache는 KVCacheOps 구현, H2O는 EvictionPolicy
      // 직접 조합 불가 (KiviCache는 자체 관리) → 에러 또는 graceful fallback 확인
  }

  #[test]
  fn test_eviction_mutual_exclusion() {
      // CLI에서 h2o + streaming 동시 지정 → 에러 또는 마지막 것만 적용
  }

  #[test]
  fn test_snapkv_then_sliding() {
      // 1. Prefill → SnapKV 압축 (1024 → 512)
      // 2. Decode → Sliding window (window=256)
      // 3. 최종 cache ≤ sink + 256
  }

  #[test]
  fn test_snapkv_then_h2o() {
      // 1. Prefill → SnapKV 압축
      // 2. Decode → H2O score eviction
      // 3. Score 축적이 압축 후에도 정상
  }

  #[test]
  fn test_throttle_plus_eviction() {
      // W3 + C4: delay 삽입 + cache eviction 동시
      // 출력 토큰은 eviction 결과에만 의존 (throttle은 lossless)
  }

  #[test]
  fn test_warning_to_critical_escalation() {
      // 1. Warning → W3 throttle 활성화
      // 2. Pressure 증가 → Critical → C4 H2O eviction 추가
      // 3. 두 action 동시 활동 확인
  }

  #[test]
  fn test_all_actions_disable_restore() {
      // 모든 action 활성화 → RestoreDefaults → 모두 비활성화
      // throttle=0, eviction 중지, KIVI bits 유지 (역전환 없음)
  }
  ```

- **Acceptance Criteria**:
  - 모든 통합 테스트 통과
  - 상호 배타 정책 (H2O/SnapKV/StreamingLLM 중 1개) 강제 확인
  - 조합 가능 정책 (eviction + KIVI, eviction + throttle) 정상 동작
  - RestoreDefaults 시 clean state 복원

---

## [P2] AP-4-3: E2E Benchmark 스크립트 및 검증

- **Status**: DONE
- **Sprint**: backlog
- **Dependencies**: 모든 Phase 구현 완료

- **Description**:

  호스트 및 Android 디바이스에서 전체 Action Pool 의 E2E 검증.

  **1. Greedy Identity 테스트** (`scripts/test_action_identity.sh`):

  ```bash
  #!/bin/bash
  # 각 action 활성화 후 greedy 출력이 baseline과 동일한지 확인
  MODEL="models/llama3.2-1b"
  PROMPT="The quick brown fox jumps over the lazy dog"
  BASELINE=$(cargo run --release --bin generate -- --model-path $MODEL \
      --prompt "$PROMPT" -n 64 --temp 0 2>/dev/null)

  # W3 Throttle: 출력 동일해야 함 (lossless)
  THROTTLE=$(cargo run --release --bin generate -- --model-path $MODEL \
      --prompt "$PROMPT" -n 64 --temp 0 \
      --enable-resilience --test-throttle-ms 50 2>/dev/null)
  assert_equal "$BASELINE" "$THROTTLE" "W3 Throttle"

  # C6 StreamingLLM: window 충분히 크면 동일
  STREAMING=$(cargo run --release --bin generate -- --model-path $MODEL \
      --prompt "$PROMPT" -n 64 --temp 0 \
      --eviction-policy streaming --sink-size 4 --eviction-window 2048 2>/dev/null)
  assert_equal "$BASELINE" "$STREAMING" "C6 StreamingLLM (large window)"

  # C1 SWIFT: greedy에서 lossless 보장
  SWIFT=$(cargo run --release --bin generate -- --model-path $MODEL \
      --prompt "$PROMPT" -n 64 --temp 0 \
      --swift --swift-ratio 0.3 2>/dev/null)
  assert_equal "$BASELINE" "$SWIFT" "C1 SWIFT"
  ```

  **2. Perplexity Benchmark** (`scripts/test_action_ppl.sh`):

  ```bash
  # 각 action의 PPL 변화 측정
  for policy in "none" "streaming --sink-size 4 --eviction-window 512" \
                "h2o --h2o-keep-ratio 0.5" "--snapkv --snapkv-capacity 512" \
                "--kivi" "--kivi --kivi-bits 4"; do
      cargo run --release --bin generate -- --model-path $MODEL \
          --eval-ll --eval-batch experiments/prompts/ppl_prompts.json \
          $policy 2>&1 | grep "PPL:"
  done
  ```

  **3. Memory Profile** (`scripts/test_action_memory.sh`):
  - `/proc/self/status` VmRSS 측정
  - 각 action 활성화 전후 RSS 비교

  **4. 조합 매트릭스 자동화**:

  ```bash
  # 유효 조합 전수 검사
  EVICTIONS=("none" "sliding" "h2o" "streaming")
  KIVI=("" "--kivi" "--kivi --kivi-bits 4")
  SWIFT=("" "--swift --swift-ratio 0.3")

  for e in "${EVICTIONS[@]}"; do
      for k in "${KIVI[@]}"; do
          for s in "${SWIFT[@]}"; do
              echo "Testing: eviction=$e kivi=$k swift=$s"
              timeout 60 cargo run --release --bin generate -- \
                  --model-path $MODEL --prompt "Hello" -n 32 --temp 0 \
                  --eviction-policy $e $k $s 2>&1 || echo "FAIL"
          done
      done
  done
  ```

  **기대 결과**:

  | 벤치마크 | Warning (Lossless) | Critical (Lossy) |
  |----------|-------------------|-------------------|
  | Greedy Identity | 100% 동일 | N/A (lossy) |
  | PPL 변화 | 0 | < 2.0 증가 |
  | Memory 감소 | W2: -50% | C4/C6: bounded, C8: -60% |
  | Throughput 변화 | W3: 비례 감소 | C1: +30~60% |
  | Crash/OOM | 0 | 0 |

- **Acceptance Criteria**:
  - 모든 유효 조합에서 crash 없음
  - Lossless action (W1, W3)의 greedy 출력 = baseline
  - Lossy action의 PPL 변화가 기대 범위 내
  - E2E 스크립트 자동화 완료
  - 결과 JSON으로 `experiments/benchmarks/results/` 에 저장

- **Notes**: 스크립트 ~200 LOC. 모델 가중치 필요 (호스트: `models/llama3.2-1b/`).

---

# 의존성 그래프

```
Phase 1-A: C6 StreamingLLM CLI
  AP-1A-1  ←── (없음, 독립)

Phase 1-B: C8 KIVI 동적 전환
  AP-1B-1  ←── (없음)
  AP-1B-2  ←── AP-1B-1
  AP-1B-3  ←── AP-1B-2

Phase 2-A: C5 SnapKV
  AP-2A-1  ←── (없음)
  AP-2A-2  ←── AP-2A-1
  AP-2A-3  ←── AP-2A-1, AP-2A-2
  AP-2A-4  ←── AP-2A-3

Phase 2-B: W2 Disk Offload
  AP-2B-1  ←── (없음)
  AP-2B-2  ←── AP-2B-1

Phase 3: C1 SWIFT
  AP-3-1   ←── (없음)
  AP-3-2   ←── AP-3-1
  AP-3-3   ←── AP-3-1, AP-3-2
  AP-3-4   ←── AP-3-1, AP-3-2, AP-3-3

Phase 4: E2E 테스트
  AP-4-1   ←── 각 Phase 병행
  AP-4-2   ←── Phase 1-A, 1-B, 2-A, 2-B
  AP-4-3   ←── 전체 완료
```

---

# 예상 규모

| Phase | 신규 LOC | 테스트 LOC | 신규 파일 | 수정 파일 |
|-------|---------|-----------|----------|----------|
| 1-A | ~30 | ~20 | 0 | 1 (generate.rs) |
| 1-B | ~450 | ~150 | 0 | 3 (quant.rs, kivi_cache.rs, quantize_handler.rs) |
| 2-A | ~360 | ~200 | 1 (math_utils.rs) | 3 (compress_handler.rs, kv_cache.rs, generate.rs) |
| 2-B | ~270 | ~100 | 1 (disk_store.rs) | 2 (swap_handler.rs, generate.rs) |
| 3 | ~850 | ~300 | 3 (skip_config.rs, speculative.rs, skip_optimizer.rs) | 3 (llama_layer.rs, llama_model.rs, generate.rs) |
| 4 | ~50 | ~400 | 2 (integration test, scripts) | 0 |
| **합계** | **~2010** | **~1170** | **7** | **~8** |
