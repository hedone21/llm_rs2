# Host OpenCL on NVIDIA CUDA: Q5+ NaN / CL_OUT_OF_RESOURCES

**Date filed**: 2026-04-14
**Filed by**: feat/gemma3-4b-support branch investigation
**Scope**: NVIDIA host OpenCL (RTX 3090 Ti, OpenCL 1.2), all Gemma3 models (1B + 4B 확인)
**Follow-up branch**: fix/host-opencl-nvidia-fallback
**Status**: Gemma3 1B 해결 (commit 24789da) · Gemma3 4B 잔존 (→ follow-up 참고)

## Symptom

- Gemma3 4B on NVIDIA OpenCL: eval-ll Q1–Q4 NLLs match CPU to 3 decimals, Q5+ all NaN. Single-prompt generation produces immediate EOS (blank).
- Gemma3 1B on NVIDIA OpenCL: eval-ll Q1–Q4 numeric, Q5 crashes with `clEnqueueReadBuffer → CL_OUT_OF_RESOURCES`.
- Gemma3 4B on PoCL (`OCL_PLATFORM=portable`): eval-ll 40/40 numeric (655s), single-prompt "a city that never sleeps...", exit 0. **Engine state management is correct.**
- CPU backend: eval-ll 40/40 numeric (447s).

## Root cause (confirmed, 2026-04-14 afternoon investigation)

초기 가설(커널 JIT 실패 + nosub fallback 경로의 state 오염)은 **부분적으로만 맞음**.
재현 시나리오를 좁히는 bisect 과정에서 전혀 다른 root cause가 드러났다:

**근본 원인 — `eval-ll` full-prefill 모드의 eviction 가드 로직 버그**

`engine/src/eval/eviction_hook.rs` `post_prefill`/`post_decode_step`의 가드:

```rust
if caches.is_empty() || max_cache_pos(caches) <= self.effective_budget {
    return;
}
```

`--eval-ll`를 `--kv-budget`/`--kv-budget-ratio` 없이 돌리면 `effective_budget = 0`
으로 설정된다 (의도: "budget 없음 = eviction 없음"). 그러나 `max_cache_pos > 0`인
순간 위 가드를 통과해 `force_evict(ratio = 0/before_len = 0)`가 실행된다.

이때 파이프라인은 다음 경로를 탄다:
1. `EvictionHandler::handle` → policy='none'이어도 `target_len = (current_pos*0).max(1) = 1`
2. `NoEvictionPolicy::evict`는 noop이지만 `ActionResult::Evicted { tokens_removed: 0 }` 리턴
3. `pipeline_results_to_eviction_result`에서 `Evicted` variant를 매치 → `any_action=true`
4. `CacheManager::execute_dispatch` (`engine/src/core/cache_manager.rs:245`):
   `eviction_result.evicted == true` → `release_unused_pages()` 호출
5. `KVCache::release_unused_pages` (`engine/src/core/kv_cache.rs:903`)은
   `current_pos < capacity/2`이면 `shrink_to_fit()`을 실행 → 레이어 N개 모두
   새 GPU 버퍼 alloc + old→new copy_slice + old 버퍼 drop
6. 다음 질문 prefill 중 NVIDIA 드라이버에서 `clEnqueueReadBuffer → CL_OUT_OF_RESOURCES`

CPU/PoCL은 shrink-reallocation 이후에도 문제없이 동작하지만, NVIDIA 드라이버에서는
레이어 수만큼 연속된 alloc/free/copy 후 드라이버 상태가 깨진다.

### JIT compile 실패 경고는 red herring였나?

아니. **9개 커널의 JIT 실패 + fallback 활성화는 사실이며** `mod.rs:418-591`에서
4개는 nosub fallback으로 자동 대체되고, 5개(flash_attn DK=64/128 · mul_mv_f16_f32_l4
· mul_mm_f16_f32_l4_lm · kivi_q2)는 해당 경로 자체가 disable된다.
다만 **"Q1-Q4 OK, Q5+ NaN"의 직접 트리거는 JIT 실패 자체가 아니라 shrink 경로**였다.
fallback 경로는 정상적으로 동작하며, shrink reallocation만 없다면 NVIDIA에서도
수치적으로 CPU/PoCL과 일치한다 (Gemma3 1B 40/40 numeric, Gemma3 4B 1-4 numeric).

---

### 원 가설 (참고 — 실제 트리거 아님)

NVIDIA OpenCL 1.2 (no `cl_khr_subgroups`, no native `convert_float(half)`) fails to JIT-compile 9 Adreno-specific kernels at model load time:

1. `mul_mv_f32_f32.cl` — `get_sub_group_local_id`, `sub_group_reduce_add` undeclared
2. `simple_ops.cl` — same subgroup intrinsics
3. `mul_mv_q4_0_f32.cl` — undeclared macros `N_SIMDGROUP`, `N_DST`, `N_SIMDWIDTH`
4. `mul_mv_f16_f32_l4.cl` — `convert_float(half)` no match (x2 variants)
5. `flash_attn_f32_f16.cl` DK=64 — `convert_float(half)` no match
6. `flash_attn_f32_f16.cl` DK=128 — `convert_float(half)` no match
7. `mul_mm_f16_f32_l4_lm.cl` — `convert_float(half)` no match
8. `kivi_q2.cl` — same

Engine reports "4 warnings and 8 errors generated. / 2 warnings and 7 errors generated. (x2) / 1 error generated." at model load, activates fallback paths. Fallback runs on NVIDIA but accumulates state corruption from Q5 onward (exact trigger op TBD).

## Reproduction (x86_64 Linux + NVIDIA CUDA OpenCL 1.2)

```bash
# 4B — Q5+ NaN
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll \
  --eval-batch /tmp/race_h_smoke_10q.json --greedy

# 1B — Q5 CL_OUT_OF_RESOURCES
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-1b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll \
  --eval-batch /tmp/race_h_smoke_10q.json --greedy
```

## Fix (applied, commit 24789da)

`engine/src/eval/eviction_hook.rs` `post_prefill`/`post_decode_step` 양쪽에
`effective_budget == 0` short-circuit을 추가:

```rust
if caches.is_empty()
    || self.effective_budget == 0
    || max_cache_pos(caches) <= self.effective_budget
{
    return;
}
```

논리적으로도 맞는 수정 — full-prefill 모드(budget 없음)에서 eviction은
무의미하다. 모든 백엔드에 무해하며 NVIDIA에서는 shrink 경로가 제거되어
근본 문제 해소.

### Verification (NVIDIA RTX 3090 Ti, OpenCL 1.2)

- Gemma3 1B eval-ll 40/40 numeric, wall 55.6s (수정 전: Q5 OOR)
- Gemma3 4B eval-ll Q1-Q8 numeric (수정 전: Q5+ NaN, 현재는 10Q 이상 배치에서만 Q5 OOR — follow-up 참고)
- 유닛 테스트: 133 eviction-related tests pass

## Remaining follow-up

Gemma3 4B는 위 수정 후에도 **배치 크기 ≥ 10일 때만 Q5에서 `clEnqueueReadBuffer → CL_OUT_OF_RESOURCES`** 재현. 원인은 shrink 경로가 아닌 별도의 NVIDIA 드라이버 resource accumulation으로 추정. 별도 이슈로 분리:

→ `issues/gemma3_4b_nvidia_batch_accumulation_20260414.md`

## Workaround (4B only)

Gemma3 4B NVIDIA 호스트 개발 환경은 `OCL_PLATFORM=portable OCL_DEVICE_TYPE=cpu`
(PoCL)로 우회 가능. Gemma3 1B는 본 fix로 복구되어 우회 불필요.

## Artifacts

- `notes/gemma3_4b_task10_status.md` — Task 10 raw output, NVIDIA 실행 로그 발췌
- `docs/40_gemma3_support.md` §9.3 — 최종 백엔드별 검증 결과 표
- `notes/gemma3_4b_final_status.md` — Phase 4 전체 최종 상태
- commit 24789da — `fix(eval): skip post-eviction when effective_budget is zero`

## Impact

Gemma3-4B PR (`feat/gemma3-4b-support`)은 **CPU 및 PoCL 경로 기준 merge-ready**. 프로덕션 타겟인 Android Adreno는 별도 테스트 파이프라인에서 검증. 본 이슈는 4B 지원의 차단 요인이 아니라 호스트 개발 환경 제약.
Gemma3 1B는 본 fix로 NVIDIA 호스트에서도 완전 복구.
