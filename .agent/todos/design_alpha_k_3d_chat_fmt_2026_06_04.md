# Phase α-K BC (3d) — chat-fmt eviction 배선 설계 (확정)

**작성**: 2026-06-04 (메인 세션, 워크플로 `wf_ba60a394-33f`: 3 설계 + 3 적대검증 렌즈 + 종합)
**목적**: 5-F(legacy+KVCacheOps trait 삭제, 비가역)의 **선결** — chat-standard 의 OLD `forward_into<C>` 의존 제거(사용자 결정: chat fmt 이주 = A1 보존).
**SSOT cross-ref**: `arch/pipeline_stage_design_v2.md` §9.1-EVICT-DECISION(γ — device sanity 한정 근거)

---

## 결론: **Approach B (Unwrap-Evict-Rewrap, UER)** — 적대검증 3 렌즈 만장일치

OLD eviction 경로(`CacheManager::execute_dispatch` → `CachePressurePipeline::execute(&mut [KVCache])`)를 **글자 그대로 재사용**한다. fmt_caches 의 inner `KVCache` 들을 일시적으로 연속 `Vec` 로 꺼내(`take_inner`) `cm.force_evict*`/`maybe_evict*` 실행 후 다시 넣는다(`put_inner`).

### 왜 B인가 (A/C 기각 근거)
- **B2 (blocking)**: chat 은 `--eviction-policy d2o` 실사용(session.rs:606 → `with_pipeline([D2OHandler])`). **D2O 는 `EvictionPolicy` 가 아니라 `CachePressureHandler`** 라 `plan_keep` 메서드 자체가 없다(keep 선택이 cross-layer `Mutex<D2OState>` EMA + Eq.11 merge 와 분리 불가). A/C(plan_keep→compact)는 D2O eviction 을 **통째로 잃고** `ensure_capacity` 가 "exceed max_seq_len even after eviction" bail(session.rs:214) → 멀티턴 chat 이 paper-core 정책에서 죽는다. B 는 `execute_dispatch` 재사용으로 D2O merge 자동 보존.
- **maybe_evict pressure-gate**: A/C plan_keep 은 무조건 keep-list 산출 → OLD 가 `Normal` 이라 no-evict 하는 상황과 발산. B 는 `cm.maybe_evict` 그대로라 pressure-gate 동일.
- **side-effect 보존**: `execute_dispatch` 의 `release_unused_pages()` madvise(:302, 멀티턴 메모리 회수) + `new_pos` 유도(:296) + `CacheEvent` 로그(observability) — A/C 우회 시 전부 손실. B 보존.
- **device 게이트 정합(B3)**: SSOT γ(RPN 90, 2026-06-03)가 World-split bit-identical 을 F3 타이밍으로 기각 + host compact_parity=1차/device=sanity 못박음. A/C 의 "logits/byte bit-identical" device 게이트는 γ 가 vacuous/impossible 로 refute 한 영역. B 게이트(eviction succeeded/pos↓/no panic/sane)가 γ 정합.

### selection 동일성 = code-path 동일성
B 는 OLD `cm.force_evict*` 를 같은 인자로 재호출하므로 evict 선택·pipeline·madvise·new_pos 가 **정의상 보존**. 별도 bit-identical 증명 불요(compact_parity 가 plan_keep 의 buffer 결과만 증명한 A/C 와 대조).

---

## Blocking 4건 반영

| # | blocking | 반영 |
|---|---|---|
| **B1** | chat `fmt_eligible=false`(session.rs:439) → `ensure_fmt_wrapped` 영구 no-op → fmt 경로 dead. fmt_eligible flip 원자적 동반 필수 | **S3 step 명시**. S2(try_evict UER)와 같은 PR. 플립 없이는 dead code, 플립만 하고 UER 없으면 회계 붕괴. |
| **B2** | A/C 가 D2O selection 재현 불가 | **B 채택으로 해소** |
| **B3** | A/C device 게이트가 γ 와 충돌 | **device 게이트 = γ sanity 한정**(bit-identical 미요구) |
| **B4** | execute_dispatch side-effect 우회 손실 | **B 채택으로 해소** |

**Warning**: W1 layer idx 정렬 불변식(`ensure_fmt_wrapped` enumerate 순서 == layer idx, D2O cross-layer 전제) / W2 Arc 보존(into_inner+재wrap 대신 with_cache_mut 기반 take_inner/put_inner — Arc 재할당 0) / W3 turn-2 panic 주석 stale(①-b 이후 해소, 플립 시 갱신 + 좌표 정합 device 재확인) / W4 H2O+ chat 미사용 carve-out(chat try_evict 가 flat scores 만 전달 — session.rs:195/287, head_scores 미전달 → H2O+ 는 flat-score H2O dispatch = dead branch).

---

## try_evict 배선 (코드 스케치)

```rust
// standard_format.rs — Arc 보존 seam (with_cache_mut 위, mem::take 변형)
impl StandardFormat {
    pub(crate) fn take_inner(&self) -> KVCache {       // KVCache: Default(빈 placeholder)
        std::mem::take(&mut self.inner.lock().unwrap().cache)
    }
    pub(crate) fn put_inner(&self, cache: KVCache) {
        self.inner.lock().unwrap().cache = cache;
    }
}
```

```rust
// model_forward.rs::try_evict — fmt 분기 추가 (시그니처 무변)
if let Some(fmts) = &self.fmt_caches {
    // W1: fmts = ensure_fmt_wrapped enumerate 순서 == layer idx.
    let before_pos = fmts.first().map(|f| f.with_cache_mut(|c| c.current_pos)).unwrap_or(0);
    let mut temp: Vec<KVCache> = fmts.iter().map(|f| f.take_inner()).collect();
    let result = (|| -> Result<EvictionResult> {        // 클로저 캡처 → ? 전파를 rewrap 이후로
        if force {
            match scores { Some(sc)=>cm.force_evict_with_scores(&mut temp,target_ratio,sc), None=>cm.force_evict(&mut temp,target_ratio) }
        } else {
            match scores { Some(sc)=>cm.maybe_evict_with_scores(&mut temp,sc), None=>cm.maybe_evict(&mut temp) }
        }
    })();
    for (f, c) in fmts.iter().zip(temp.into_iter()) { f.put_inner(c); }   // rewrap 항상 실행
    let result = result?;
    return if result.evicted { Ok((before_pos.saturating_sub(result.new_pos), result.new_pos)) } else { Ok((0, before_pos)) };
}
// 기존 kv_caches 경로 (fmt 비활성) — 무변
```

**전제**: `KVCache: Default`(빈 placeholder for mem::take). 미충족 시 `Option<KVCache>` 또는 placeholder 생성자 필요 — S1 착수 시 확인.

---

## 정책 커버리지 (bail 0)

| 정책 | (3d) 처리 | 비고 |
|---|---|---|
| none/sliding/streaming/h2o | **지원** (UER, OLD 동일) | sliding=Round15 1B 최적 |
| h2o_plus | **지원 (flat-only carve-out)** | chat try_evict 가 head_scores 미전달 → flat-score H2O dispatch. per-head 발화는 (3d) 범위 밖. Round15 worthless → 영향 0 |
| d2o | **지원** (UER → D2OHandler pipeline 그대로, merge 보존) | A/C 불가, **B 결정적 우위** |

---

## 구조 변경 + LOC

| 파일 | 변경 | LOC |
|---|---|---|
| `standard_format.rs` | take_inner/put_inner 신규 | ~12 |
| `model_forward.rs::try_evict` | fmt 분기(unwrap+evict 캡처+rewrap), kv_caches 경로 무변 | ~40 |
| `session.rs`(chat builder) | fmt_eligible false→true(:439) + turn-2 panic 주석 갱신 | ~3 |
| **구현 소계** | | **~55** |
| `compact_parity.rs` 확장 | cm.force_evict 대비 multi-layer + maybe_evict pressure-gate | ~60 |
| `test_chat_session_multi_turn` 확장 | fmt_eligible=true 멀티턴 eviction 회계(pos/evicted_total/logits ON≡OFF) | ~30 |

**무변경**: `cache_manager.rs`(신규 public 메서드 0), `pressure/eviction/*`, `kv_cache.rs`, `KVCacheFormat`, `CachePressurePipeline`, `D2OHandler`, `decode_loop.rs`.

---

## sub-step 시퀀싱

| step | 내용 | 게이트 | 가역성 |
|---|---|---|---|
| **S1** | standard_format take_inner/put_inner (unwired, 호출처 0) | host build + unit(take/put round-trip = identity) | 가역 additive |
| **S2** | try_evict fmt 분기(UER). fmt_eligible=false라 production 미발화(S3까지 dead) | host: compact_parity 확장 + build/clippy | 가역 |
| **S3 (원자)** | chat fmt_eligible false→true + 주석 갱신. **B1 해소, fmt 실발화** | host: chat spec test 멀티턴 sliding/h2o/d2o fmt-ON≡OFF | 가역(env 격리) |
| **S4 (device)** | S25 eviction-fmt sanity 게이트 = **acceptance** | device γ sanity | RED 시 S3 revert |

**원자성**: S2+S3 한 PR(S2만=dead code, S3만=회계 붕괴). S1 additive 선행 커밋 가능.

---

## device 게이트 spec (S25, γ = sanity 한정, **bit-identical 아님**)
- 디바이스: `galaxy_s25` opencl `--opencl-rpcmem` 단독.
- 모델/KV: qwen2.5-1.5b-q4_0 또는 llama-1b **F16-weight** × kv-type {f16,f32,q4}.
- 정책: chat 선택가능 6종(none/sliding/streaming/h2o/h2o_plus/d2o) 각 1회.
- prompt: **multi-turn ≥3턴, max_seq 축소로 압력 유발**(single-turn=vacuous).
- 합격(γ sanity): (1) eviction succeeded(no bail) (2) pos 감소 (3) evicted_total>0 (4) logits valid(NaN/Inf 0, sane) (5) **unwrap/rewrap panic 0**(Mutex 안전) (6) eviction(pos↓)↔turn-2 prefill append q_start_pos 좌표 정합(W3, garbage 아님).
- sliding/h2o/streaming = host compact_parity 등가 보장(device 보조); H2O+/D2O = UER 보존 확인.

---

## 잔여 위험
1. **rewrap placeholder 잔존(medium)**: evict 도중 panic(unwind)이 일부 fmt 를 placeholder 로 남길 수 있음 → 클로저 캡처로 `?` 전파를 rewrap 이후로(완화). evict 는 실용상 panic-free. 추가 강건화 시 take 인덱스 추적 drop guard(5-B DrainGuard 패턴) 고려 — (3d) 범위엔 클로저로 충분.
2. **H2O+ per-head 미발화(low)**: chat flat-only. (3d) 범위 밖, Round15 영향 0, carve-out.
3. **W1 layer idx 정렬 의존(low)**: D2O cross-layer 가 fmt_caches 순서==layer idx 의존. enumerate 보장, spec test 로 고정.
4. **device sanity 가 selection 발산 직접 미포착(accepted, γ 정합)**: host compact_parity(distinct-byte)가 1차로 잡음.
5. **(3d) ≠ KVCacheOps 삭제**: chat decode fmt화 + eviction fmt화가 목적. offload/chunked-prefill/KiviForward OLD-chain 잔여는 5-F.

---

**관련 파일**: `engine/src/pressure/standard_format.rs`(take/put_inner), `engine/src/session/forward/model_forward.rs:636`(try_evict), `engine/src/session/chat/session.rs:439`(fmt_eligible), `engine/src/pressure/eviction/compact_parity.rs`(host gate), `engine/tests/spec/test_chat_session_multi_turn.rs`(멀티턴), `arch/pipeline_stage_design_v2.md` §9.1-EVICT-DECISION(γ).
