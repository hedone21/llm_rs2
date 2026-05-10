# Handoff: LISWAP-6 swap mode 심도 분석 (per-tick=25 / per-tick=1 / phase-aware)

**Date**: 2026-05-11
**HEAD**: `6a48449` (eager prefault + outlier 정정 commit)
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830, 6T)
**선행 문서**:
- `papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md` §7.7 (LISWAP-6 측정 + outlier 정정)
- `.claude/plans/purrfect-jumping-kettle.md` (LISWAP-6 design plan)
- 메모리: `project_liswap6_alias_production.md`, `project_qnn_oppkg_dual_buffer_kv.md`

---

## 0. 한 줄 요약

LISWAP-6 (DMA-BUF alias + eager prefault) 적용 후 3개 swap mode (per-tick=25 / per-tick=1 / phase-aware) 측정 — 단 **per-tick=25에서 bimodal 패턴 (tail 3.7 ↔ 10.27 ms, 4× variance) 발견, root cause 미확정**. 다음 세션이 root cause 분석 + 정확도 (출력 quality) + production scenario (mid-decode swap) 깊이 검증.

---

## 1. 현재 state (commit `6a48449`)

### 1.1 구현 완료

| Commit | 내용 |
|---|---|
| `5b6e022` | LISWAP-6 Phase A~E 구현 (RpcmemSecondaryStore + alias buffer + try_alias_materialise) |
| `d81cbb2` | qnn_oppkg lm_head Q4_0 quantize GPU dispatch fix (pre-existing latent bug) |
| `f038301` | Eager prefault on model load — swap stall 700→400 ms |
| `6a48449` | postmortem 갱신 + outlier 정정 |

### 1.2 자동 활성화 조건
- `--backend qnn_oppkg` + `--secondary-gguf` 둘 다 active 시 LISWAP-6 alias 자동
- target_layers 결정 직후 `secondary.prefault_layers()` 자동 호출 (eager prefault)
- 기타 backend (opencl/cpu/cuda) 또는 secondary 없음 → 기존 mmap path fallback

### 1.3 검증된 production path
```bash
./generate -m <f16>.gguf --tokenizer-path <tok>.json \
    --secondary-gguf <q4>.gguf --force-swap-ratio 0.X \
    --swap-incremental-per-tick {25|1} \
    --backend qnn_oppkg --temperature 0
```

---

## 2. 측정 인프라

### 2.1 Binary
- 빌드: `source android.source && cargo build --release --target aarch64-linux-android -p llm_rs2 --features opencl,qnn`
- 디바이스 위치: `/data/local/tmp/generate`
- mtime: 2026-05-11 (commit `f038301` 시점)

### 2.2 모델 + prompt
| 모델 (KV 1.8k 측정) | path |
|---|---|
| Primary (F16) | `/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf` |
| Secondary (Q4_0) | `/data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf` |
| Tokenizer | `/data/local/tmp/models/qwen2.5-1.5b/tokenizer.json` |
| Prompt (KV 1.8k) | `/data/local/tmp/prompt_1800.txt` (1821 token) |
| Prompt (KV 4K) | `/data/local/tmp/prompt_4096.txt` (4070 token, Llama 측정용) |
| Prompt (short) | `-p "The capital of France is"` (5 token) |

### 2.3 Env vars
| Env | 효과 |
|---|---|
| `LLMRS_PER_TOKEN_MS=1` | 매 token forward_ms 출력 (`[PER_TOKEN] idx=N kv_pos=K forward_ms=X.XX`) |
| `LLMRS_SWAP_PROFILE_BREAKDOWN=1` | per-tensor swap timing + alias source (`[swap-prof] ... source=rpcmem-alias`) |
| `LLMRS_PHASE_AWARE_DEBUG=1` | phase-aware dispatcher snapshot |
| `RUST_BACKTRACE=full` | crash 시 backtrace |

### 2.4 LD library
```bash
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
```

---

## 3. 측정 데이터 (KV 1.8k Qwen, n=20, 3 run)

### 3.1 Mode 비교 (median, 본 세션 측정값)

| Mode | Prefill | tok[0] | tail | swap latency | E2E | CV (tail) |
|---|---:|---:|---:|---:|---:|---:|
| **Q4 only baseline** | 13852 | ~10 | **10.78** | — | 14066 | ~2% |
| **per-tick=25** | 11744~12172 | 7~11 | **3.70 ↔ 10.27** ⚠️ | 408~553 | 11819~12379 | **~50% bimodal** |
| **per-tick=1** | 13328 | 9 | **8.34** ✅ | 19×~50ms = 945ms | 13503 | ~2% |
| **phase-aware** | 14086 | **403** | **45.41** | (chunk dispatch) | 15397 | ~3% |

(prefault: 모든 swap mode에서 250~390 ms model load 시 발생)

### 3.2 Eager prefault 효과 (commit `f038301` 전후)

| Stage | lazy (이전) | eager (`f038301`) |
|---|---:|---:|
| Model load 시 prefault | — | **250~362 ms** (한 번만) |
| Swap blocking total | **700 ms** | **403 ms** (-42%) |
| - swap stages.prefault | 420 ms | **0 ms** |
| - swap stages.mmap_permute | 300 ms | 403 ms (variance noise) |

### 3.3 정확도 측정 (deterministic, n=40, prompt="The capital of France is")

| Mode | 출력 첫 줄 | Fact 정확도 |
|---|---|---|
| F16 only | "...French are known for love of food and wine..." | general |
| Q4 only | "...**2 million people**, 104 sq km, 8 districts" | ❌ Q4 양자화 손실 |
| **per-tick=25** | "...**67 million people**, **Île-de-France (Paris)**" | **✅ 가장 정확** |
| per-tick=1 | "...**6 million** inside city (2013)" | △ partial |
| phase-aware | "...67M, **largest city, London**" | ❌ **hallucinate** (intra-layer mixed weight) |

**phase-aware hallucinate 원인**: chunk granularity가 layer보다 작아 (per-tensor) → layer N의 attn_q는 Q4, attn_v는 F16 인 시점 존재 → attention output 손상.

---

## 4. **bimodal 패턴 (per-tick=25)** — 다음 세션 핵심 디버그

### 4.1 관찰

같은 명령 같은 환경에서 per-tick=25 tail이 두 cluster:
- **Fast cluster**: tail ~3.7 ms (CV 약함)
- **Normal cluster**: tail ~10 ms (Q4 baseline과 동일)

3 run 측정 (#2 segfault, #1 vs #3):
| Run | swap latency | tail | tok[0] |
|---|---:|---:|---:|
| #1 | **553 ms** | **3.70 ms** | 10.82 |
| #3 | **408 ms** | **10.27 ms** | 7.16 |

**Inverse relationship**: swap이 더 길면 → decode 더 빠름. 의외.

### 4.2 가설 (검증 필요)

#### Hypothesis A — GPU clock state (DVFS)
- 첫 cold run: process init + swap 더 무거움 → GPU 부하 indicator → DVFS high frequency 유지 → fast tail
- 후속 warm run: thermal/cache warmer → 짧은 swap → low frequency baseline → normal tail

#### Hypothesis B — Page cache 영향
- cold run의 mmap fault가 GPU page cache에도 영향
- secondary mmap region이 page cache hot 되면 GPU access도 빠름

#### Hypothesis C — OpenCL context state
- 첫 swap 시 cl_mem 등록이 driver internal cache에 hot
- 후속 forward에서 cache hit → fast tail

#### Hypothesis D — LayerSlot ArcSwap visibility
- per-tick=25 single-shot 후 ArcSwap commit 시점에 따라 forward thread가 새 alias buffer 또는 stale buffer 보는 race
- stale buffer (BorrowedMmapBuffer) → 10 ms, alias buffer → 3 ms

### 4.3 다음 세션이 진행 중인 검증 측정

**Background task (commit `6a48449` 시점)**: `bjqlg19o8`
- `LLMRS_PER_TOKEN_MS=1` + per-tick=25 × 5 run (10s idle) + 50s cool + 2 run (60s idle)
- 출력: `/private/tmp/claude-501/-Users-li-Workspace-llm-rs2/1c559a8b-b5a0-4f95-a91a-af4471a16c75/tasks/bjqlg19o8.output`
- 상태: 아직 실행 중 (process running) — 다음 세션이 결과 확인

**확인 가능한 신호**:
- bimodal vs continuous distribution
- cool 시간이 늘어나면 fast cluster 더 빈번한가
- 첫 token부터 fast인가, 어느 시점부터 slow로 떨어지는가

### 4.4 추가 디버그 옵션

1. **n=200+ long decode** — 시간이 흐르면 GPU clock 떨어지는지
2. **GPU frequency monitor** — `cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq` 동시 측정
3. **forward_into 안에 sync 추가** — 매 layer 후 synchronize → driver scheduler effect 분리
4. **LayerSlot trace** — slot.load_weights() 가 alias vs non-alias 어떤 buffer 반환하는지 확인

---

## 5. 다음 세션 작업 우선순위

### Priority 1 — bimodal root cause (가장 큰 의문)
- §4.4 디버그 옵션 진행
- 5+ run distribution 분석 (`bjqlg19o8` 결과)
- 가설 A/B/C/D 검증 또는 새 가설

### Priority 2 — 정확도 정밀 검증
- per-tick=25 / per-tick=1 / phase-aware × 다양 prompt × n=100+ 출력 비교
- semantic plausibility 외 perplexity (PPL) 측정 추가 — same weight 가정 시 PPL 동일해야
- LISWAP-6의 weight access path 차이가 forward output에 byte-level 영향 주는지 확인 (있으면 silent corruption 가능성)
- phase-aware hallucinate (London) 패턴이 random 인지 deterministic 인지

### Priority 3 — Mid-decode swap 측정 (production scenario)
- 현재 측정은 swap이 token 0 강제 (force_swap_ratio + decode 시작 시점)
- 진짜 production: token N (decode 진행 도중) swap signal 수신 → swap 후 계속
- 옵션:
  - `--swap-delay-tokens N` CLI flag 추가
  - mock_manager IPC trigger 활용 (`engine/src/bin/mock_manager`)
- 측정: token N 시점 swap stall + 그 후 forward time 회복

### Priority 4 — KV 무관 일반화
- KV 5 / 1k / 1.8k / 4k 모두 재측정 (eager prefault 적용 후)
- KV 1.8k Qwen 측정값을 Llama / Gemma 다른 모델로 확장
- KV-dependence 패턴 정밀 정량

### Priority 5 — Cleanup segfault fix (P2 backlog)
- 측정 자동화 영향 (segfault 매 run 처리 필요)
- production 동작 영향 없음
- 별도 task #33

---

## 6. 관련 코드 / docs 인덱스

### 본 작업으로 수정 가능성 높은 파일
- `engine/src/bin/generate.rs:2867~2897` — eager prefault hook (이미 적용)
- `engine/src/models/weights/rpcmem_secondary.rs` — `ensure_layer_loaded`, `prefault_layers`
- `engine/src/buffer/rpcmem_alias_buffer.rs` — alias buffer + Drop ordering
- `engine/src/models/weights/swap_executor.rs::try_alias_materialise` — alias path 분기

### 참고 패턴 (수정 X)
- `engine/src/backend/qnn_oppkg/hybrid_memory.rs:104-171` — KV cache rpcmem alloc + alias 패턴 (LISWAP-6의 모델)
- `engine/src/backend/qnn_oppkg/kv_buffer.rs:80-92` — Drop ordering 검증된 패턴

### 측정 보고서
- `papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md`
  - §7.6 fair comparison (LISWAP-5)
  - §7.7 LISWAP-6 측정 + 정확도 + KV 4K + eager prefault + outlier 정정

### TODO 추적
- `.agent/todos/backlog.md` — LISWAP-5 v2 CLOSED, LISWAP-6 cleanup segfault P2

---

## 7. Open questions (다음 세션 답변 필요)

1. **per-tick=25 bimodal**: GPU clock / page cache / OpenCL context / ArcSwap race — 어느 가설이 맞는가?
2. **per-tick=1 일관성 (CV 2%)**: 왜 per-tick=25보다 일관적인가? 매 token swap이 GPU clock state 안정화?
3. **phase-aware tail 45ms vs Q4 baseline 10.78ms**: 4.2× 손해의 정확한 분해 (chunk dispatch vs scheduler vs SOA registry race vs intra-layer mixed weight)
4. **LISWAP-6 alias 효과**: 진짜 decode 가속이 있는가, 아니면 측정 noise (bimodal에서 fast cluster만)?
5. **정확도**: per-tick=25 출력 ("Île-de-France")이 정말 Q4 weight forward 결과인가, 아니면 alias buffer가 silently 다른 weight 사용하는가?

---

## 8. 다음 세션 시작 절차

```bash
cd /Users/li/Workspace/llm_rs2

# 1. 상태 확인
git log --oneline -5    # 6a48449 가 HEAD
cat .agent/todos/handoff_liswap6_swap_modes_deep_analysis_2026_05_11.md  # 본 문서
cat papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md  # 측정 보고서 §7.7

# 2. 디바이스 + binary 확인
adb devices
adb shell 'ls -la /data/local/tmp/generate'  # mtime 2026-05-11 이어야

# 3. 진행 중 측정 결과 확인 (bjqlg19o8)
cat /private/tmp/claude-501/-Users-li-Workspace-llm-rs2/1c559a8b-b5a0-4f95-a91a-af4471a16c75/tasks/bjqlg19o8.output
adb shell 'ps -A | grep generate'  # 끝났으면 비어있음

# 4. bimodal distribution 분석부터 시작
# 또는 mid-decode swap CLI option 추가 작업
```

---

## 9. 본 세션 commit history

```
6a48449 docs(liswap6): eager prefault + outlier 정정
f038301 fix(liswap6): eager prefault on model load — swap stall 700→400ms
9edbffb docs(liswap6): KV 4K Llama 측정 + KV 1.8k Qwen 비교
8d3abc5 docs(liswap6): 정확도 비교 추가 — phase-aware는 intra-layer mixed weight로 hallucinate
d883e21 docs(liswap6): KV 1.8k 측정 결과 + LISWAP-5 폐기 확정
d81cbb2 fix(liswap6): qnn_oppkg lm_head Q4_0 quantize GPU dispatch
5b6e022 feat(liswap6): DMA-BUF alias weight swap (zero H2D copy)
```

**End of Handoff** — self-contained, 다음 세션은 본 문서만으로 시작 가능.
