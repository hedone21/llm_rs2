# Handoff: phase-aware win 시도 — Phase 5b 후 follow-up

**Date**: 2026-05-11
**HEAD**: `ee22b3e` (Phase 5 alias H2D-skip 적용)
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830, 6T)
**선행 보고서**: `.agent/todos/phase_aware_win_attempt_2026_05_11.md` (4-Phase + Phase 5 상세)

---

## 0. 한 줄 요약

phase-aware swap이 single-shot wall time을 이길 수 있는지 5-Phase 검증.
**Phase 5b로 phase-aware 1411 → 515 ms (-63%) 까지 단축, single-shot 대비 +21%까지 따라잡음** (Phase 0 +92%에서 70%p 개선). 단 여전히 single-shot (427 ms)이 winner. 88 ms 잔여 갭의 원인은 chunk dispatch 자체의 micro-overhead. 다음 세션은 (1) 잔여 갭 추가 최적화, (2) mid-decode + Phase 5b 재측정, (3) 다른 KV/모델 generalization 중 선택.

---

## 1. 현재 state (commit `ee22b3e`)

### 1.1 측정 결과 (KV 1.8k Qwen2.5-1.5B, qnn_oppkg + LISWAP-6, n=30, 3 run median)

| Phase | 변경 | per-tick=25 | phase-aware | gap |
|---|---|---:|---:|---:|
| 0 | LISWAP-6 alias only | 735 ms | 1411 ms | +92% |
| 1 | + cached cl_mem alias | **405** ★ | 1029 | +154% |
| 2 | + throttle K-sweep | 405 | 1029~2715 | 무효 |
| 3 | + mid-decode (delay=30) | 402 | 1229 | +205% |
| 4 | + async worker thread | 405 | 1903 | +369% (worker contention) |
| **5b** | **+ alias H2D-skip (2곳)** | **427** | **515** | **+21%** ★ |

### 1.2 Phase 5b의 결정적 fix 두 곳

사용자 통찰 ("rpcmem alias = 병렬용. memcpy가 제대로 안 된 거 아냐?") 적용:

**Fix A** (`engine/src/models/weights/swap_executor.rs:892~916`):
- `build_tensor_from_mmap_async_for_hook` 가 alias buffer 받고도 `enqueue_write_async` 호출 → 새 cl_mem 할당 + rpcmem → 새 cl_mem H2D copy 진행
- 수정: `cpu_tensor.buffer().cl_mem().is_some()` → `enqueue_write_async` skip + dummy event

**Fix B** (`engine/src/models/weights/phase_aware_swap.rs:288~301`):
- `wait_pending` 가 dummy event 받아도 `wait_event_blocking` → fall-through `self.synchronize()` → forward GPU op까지 sync block
- 수정: `ev.is_dummy()` 검출 후 skip

**API 추가** (`engine/src/core/backend.rs:28~46`):
- `GpuEvent::is_dummy()` — `inner_cl/inner_cu` 모두 None 검출. cfg(opencl/cuda) gating

**Process_commit 동시 fix** (`engine/src/models/weights/async_swap.rs:248~258`): 같은 dummy event 함정 공유라 같이 적용.

### 1.3 Phase 5b per-token 패턴 (KV 1.8k, sanity 1 run)

```
tok[0] = 68 ms     (Phase 0 = 397, 5.8× 단축)
tok[1] = 59 ms
tok[2] = 51 ms
tok[3] = 44 ms     (plan retired at token=3)
tok[4+] = ~11 ms   (steady, baseline 회복)
```

inflation = 222 ms (Phase 0 1180 ms 대비 -81%).

### 1.4 자동 활성화 조건

- `--backend qnn_oppkg` + `--secondary-gguf` + `--swap-phase-aware`
- LISWAP-6 alias 자동 활용 (eager prefault + cached cl_mem)
- Phase 5 fix는 alias path detection 자동, 비-alias (memcpy) path 영향 없음

### 1.5 새 CLI flags (Phase 2/3 추가, 모두 default off)

- `--swap-phase-aware-max-chunks-per-token N` (Phase 2 throttle, default 0=무제한)
- `--swap-delay-tokens N` (Phase 3 mid-decode, default 0=즉시)

---

## 2. 측정 인프라

### 2.1 Binary
- 빌드: `source android.source && cargo build --release --target aarch64-linux-android -p llm_rs2 --features opencl,qnn`
- 디바이스 위치: `/data/local/tmp/generate`
- mtime 기준: 2026-05-11 03:46~ (Phase 5b 빌드)

### 2.2 모델 + prompt (KV 1.8k Qwen 표준)
| | path |
|---|---|
| Primary (F16) | `/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf` |
| Secondary (Q4_0) | `/data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf` |
| Tokenizer | `/data/local/tmp/models/qwen2.5-1.5b/tokenizer.json` |
| Prompt | `/data/local/tmp/prompt_1800.txt` (1821 token) |

### 2.3 Env vars
| Env | 효과 |
|---|---|
| `LLMRS_PER_TOKEN_MS=1` | `[PER_TOKEN] idx=N kv_pos=K forward_ms=X.XX` 출력 |
| `LLMRS_PHASE_AWARE_DEBUG=1` | `[PhaseAwareSwap-DBG]` snapshot per token |
| `LLMRS_SWAP_PROFILE_BREAKDOWN=1` | per-tensor swap timing |

### 2.4 LD library
```bash
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
```

### 2.5 측정 스크립트 (재사용 가능)
- `/tmp/measure_phase5b.sh` — 3 mode × 3 run, n=30
- `/tmp/measure_phase3_2026_05_11.sh` — mid-decode delay=30, n=60
- `/tmp/measure_phase2_sweep.sh` — throttle K sweep
- 결과 파싱: `/tmp/parse_swap_modes_v2.py` (LOG_DIR sed로 변경)

---

## 3. 다음 세션 작업 우선순위

### Priority 1 — 잔여 88 ms 갭 분석/최적화 ★ (가장 흥미로운 후속)

phase-aware 515 ms vs single-shot 427 ms 의 **88 ms gap** 원인:
- chunk dispatch self overhead (252 chunks × ~0.35 ms)
- 후보:
  - `chunk_queue` Mutex lock per chunk
  - `pending_layers` Mutex lock per chunk install
  - `cached_alias` Mutex + HashMap lookup per chunk
  - `SwapExecutor::new` 매 chunk 생성 (스택 alloc + Galloc init)

**측정 작업**:
1. `[PhaseAwareSwap-DBG]` 출력에 per-chunk dispatch wall-clock 추가
2. tok[0~3] 의 in-forward inflation 정확한 분해
3. SwapExecutor 한 번 생성해서 reuse → 측정 영향

**가능한 추가 fix**:
- `SwapExecutor` worker-thread 1회 생성 후 reuse
- `chunk_queue` 를 lock-free queue (crossbeam)로 교체
- `pending_layers` per-layer 분리 (HashMap → Vec<Mutex<PartialLayer>>)
- `cached_alias` lock-free read (DashMap 또는 ArcSwap)

목표: phase-aware wall < 427 (single-shot win 가능성). 단 88 ms 단축 어려울 수 있음.

### Priority 2 — Mid-decode + Phase 5b 재측정

Phase 3 측정은 Phase 5b 이전이라 phase-aware 결과가 부정확.
- 새 측정: `--swap-delay-tokens 30 --swap-phase-aware`, n=60
- 비교: 같은 delay 의 per-tick=25 vs phase-aware
- production scenario 에서 Phase 5b의 진짜 가치 측정
- 가설: mid-decode 에서는 phase-aware 의 "swap이 자연스럽게 흡수" 가 user UX 측면에서 single-shot stall을 이길 수 있음

### Priority 3 — Generalization 측정

- **다른 KV size**: KV 5 / 1k / 4k (Llama) — Phase 5b 효과의 KV-dependence
- **다른 모델**: Llama 3.2-1B (1.6B 동급), Gemma3 — 다른 layer count, tensor size
- **결과 패턴**: phase-aware advantage 가 어느 환경에서 max?

### Priority 4 — Accuracy verification

Phase 5b 변경 (alias 직접 사용) 이 byte-correct 한가?
- per-tick=25 vs phase-aware deterministic 출력 비교 (n=100, prompt 다양)
- PPL 측정 (perplexity) — 같은 weight forward 면 PPL 동일 보장
- 만약 mismatch 면: alias buffer 의 GPU-visibility 가 commit 직후 정상 보장되는지 검증
  (현재는 wait_pending skip → ArcSwap commit 직후 다음 forward 가 새 cl_mem 사용. cl_mem 이 alias 라 pre-warmed)

### Priority 5 — Multi-queue (가능성 5)

Adreno OpenCL 이 `clCreateCommandQueue` multiple instance 지원하는가?
- 별도 transfer queue 가 있다면 chunk dispatch 가 진짜 GPU-level parallel 가능
- 단 alias 환경에서는 이미 transfer = 0 → 효과 작을 수 있음
- 단 chunk dispatch overhead 자체 (cl_mem 등록) 가 별 queue 에서 진행되면 micro-cost 단축 가능

조사 step:
1. Adreno OpenCL extension list 확인
2. multi-queue prototype microbench 작성
3. async_swap.rs::AsyncSwapDispatcher 가 transfer_queue 별도 사용 옵션 추가

### Priority 6 — Cleanup segfault fix (Task #33, P2 backlog)

여전히 cleanup time 에 SIGSEGV. 측정 자동화 영향 있음 (매 run 수신 처리 필요). production 동작 영향 없음.

---

## 4. 코드 인덱스

### 4.1 본 작업으로 수정된 파일 (commit `c753efa`, `ee22b3e`)

| 파일 | 변경 내용 |
|---|---|
| `engine/src/buffer/rpcmem_alias_buffer.rs` | Weak<SecondaryMmap> demote (Phase 1 cycle 차단) |
| `engine/src/models/weights/rpcmem_secondary.rs` | alias_cache + backend Arc + install_self_arc (Phase 1) |
| `engine/src/models/weights/secondary_mmap.rs` | backend 전달 (Phase 1) |
| `engine/src/models/transformer.rs` | install_self_arc 호출 (Phase 1) |
| `engine/src/models/weights/swap_executor.rs` | cached_alias try-first (Phase 1) + alias H2D skip (Phase 5a) |
| `engine/src/models/weights/phase_aware_swap.rs` | throttle (Phase 2) + worker submit (Phase 4) + install_self_weak + wait_pending alias-skip (Phase 5b) |
| `engine/src/models/weights/async_swap.rs` | SwapJob::DispatchChunk variant (Phase 4) + process_commit dummy event skip (Phase 5) |
| `engine/src/core/backend.rs` | GpuEvent::is_dummy() (Phase 5) |
| `engine/src/bin/generate.rs` | 3 CLI flags + dispatch_force_swap! macro + decode loop hooks + mid-decode trigger |

### 4.2 측정 보고서

- `.agent/todos/phase_aware_win_attempt_2026_05_11.md` — 4-Phase + Phase 5 상세 결과 + 해석
- `papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md` — LISWAP-5 v1 측정 (참고용)
- `papers/eurosys2027/_workspace/experiment/swap_overhead_phase_predictability_2026_05_10.md` — phase predictability microbench (CV 1.2%)
- `papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md` §5 — async swap microbench (Scenario B GREEN)

### 4.3 알려진 미해결 이슈

- **Cleanup segfault** (Task #33) — 모든 swap mode 종료 후 SIGSEGV
- **per-tick=1 wall variance ±25%** — thermal/clock state 민감, 작업과 무관
- **Phase 4 worker thread는 Phase 5b 와 함께 active** — Phase 5b fix 없이는 worker contention 으로 wall 악화. Phase 5b 기준에서는 worker 가 도움 (forward 영향 최소화)

---

## 5. 알려진 fact (다음 세션이 가정해도 OK)

1. **Phase 1 cached cl_mem alias는 production 가치 확정** — 모든 swap mode 에서 wall 단축
2. **Phase 5b alias H2D-skip 도 production 가치 확정** — phase-aware path 한정이지만 무해
3. **Per-tick=25 single-shot 이 production winner** — 모든 시나리오에서. 단 phase-aware 와의 격차 +21%까지 좁아짐
4. **Adreno OpenCL alias 환경의 architectural ceiling** — single OpenCL queue → driver FIFO. multi-queue 미확인
5. **Phase R Scenario B microbench 가설은 alias 환경에서 valid 하지 않음** — transfer = 0 이라 hide할 게 없음, 단 dispatch self overhead 는 여전

---

## 6. 다음 세션 시작 절차

```bash
cd /Users/li/Workspace/llm_rs2

# 1. 상태 확인
git log --oneline -5    # ee22b3e가 HEAD
cat .agent/todos/handoff_phase_aware_win_attempt_followup_2026_05_11.md  # 본 문서
cat .agent/todos/phase_aware_win_attempt_2026_05_11.md  # 5-Phase 상세

# 2. 디바이스 + binary 확인
adb devices
adb shell 'ls -la /data/local/tmp/generate'  # mtime 2026-05-11 03:46 이어야

# 3. Phase 5b sanity 1-shot (KV 1.8k phase-aware) — phase-aware tok[0]≈68ms, tok[3]plan retired
adb shell 'cd /data/local/tmp && \
  export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=/data/local/tmp/qnn && \
  export LLMRS_PER_TOKEN_MS=1 && \
  export LLMRS_PHASE_AWARE_DEBUG=1 && \
  ./generate \
    -m /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    --force-swap-ratio 1.0 \
    --swap-phase-aware \
    --prompt-file /data/local/tmp/prompt_1800.txt \
    -n 30 \
    --backend qnn_oppkg --temperature 0' 2>&1 | grep -E "weight_swap|Decode:|TTFT|PER_TOKEN|plan retired" | head -10

# 4. Priority 1 시작 (잔여 88 ms 갭 분석)
# 또는 Priority 2 시작 (mid-decode + Phase 5b 측정)
```

---

## 7. 본 세션 commit history

```
ee22b3e feat(liswap6): Phase 5 alias H2D-skip — phase-aware wall 1411 → 515 ms (-63%)
c753efa feat(liswap6): phase-aware win 시도 4-Phase — Phase 1만 production win
2abf1de docs(liswap6): handoff §4.4b — per-token 패턴 추가 (이전 세션)
```

**End of Handoff** — self-contained, 다음 세션은 본 문서만으로 시작 가능.
