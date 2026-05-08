# Predictive Pre-staging Swap 옵션 X v2 측정 (BG thread spawn at decode tok 0)

**측정 일자**: 2026-05-08
**디바이스**: Galaxy S25 (`R3CY408S5SB`), Adreno 830, OpenCL
**모델**: Qwen2.5-1.5B F16 GGUF (primary) + Q4_0 AOS .auf (secondary)
**프롬프트**: "The quick brown fox jumps", n=100, --ignore-eos, threads=6
**바이너리**: `aarch64-linux-android` release (HEAD `f3dc4a4` 기준 worktree)

---

## TL;DR — **NEGATIVE: v2 deterministic crash (5/5)**

옵션 X v2는 디코드 토큰 0에서 `std::thread::spawn`으로 진짜 BG OS thread를 띄워 swap을 진행하는 동안 main thread가 decode를 계속하도록 설계되었다. 5/5 run 모두 BG thread spawn 후 약 5~10 token decode 진행 중에 **`AdrenoOsLib` driver thread에서 SIGABRT(`Scudo ERROR: invalid chunk state`)로 deterministic crash**한다.

이는 Adreno OpenCL driver(`libCB.so`/`libgsl.so`)의 command buffer 관리자가 두 application thread에서 동시에 enqueue 호출되는 상황(그리고 동시에 swap이 cl_mem을 release하거나 ArcSwap commit으로 변경하는 상황)을 견디지 못한다는 LISWAP-2 (Adreno multi-queue serialize) 결과와 동일한 root cause로 보인다. 더 강한 형태(Scudo invalid free 패닉)로 나타났다.

| 시나리오 | n | Prefill (ms) | Decode (ms/tok) | tok[0] (ms) | Swap (ms) | TTFT (ms) | 상태 |
|----------|---|-------------:|----------------:|------------:|----------:|----------:|------|
| sync_baseline | 5 | 105.7 ± 1.3 | 25.91 ± 0.08 | 26.66 ± 0.27 | 424.1 ± 100 | 638.7 ± 99 | **PASS 5/5** |
| pre_stage_v2 | 5 | 221.3 ± 2.1 | — | — | — | — | **CRASH 5/5** |

---

## 측정 시나리오

### CLI 공통

```
./generate \
  --model-path qwen2.5-1.5b-f16.gguf \
  --secondary-gguf qwen2.5-1.5b-q4_0-aos.auf \
  --secondary-layout aos \
  --force-swap-ratio 0.9 \
  --threads 6 \
  --backend opencl \
  --ignore-eos \
  -p "The quick brown fox jumps" -n 100
```

- **sync_baseline**: 위 CLI 그대로 (swap이 prefill 전 single-shot으로 발생)
- **pre_stage_v2**: 위에 `--swap-pre-stage` 추가 (swap이 decode token 0 시점 BG thread로 deferred)

### v2 구현 (engine/src/bin/generate.rs:5190~)

- decode token 0 시점에 25개 target layer 정보 capture
- `std::thread::spawn(move || { SwapExecutor::execute_on_slots(...) })` 호출
- caller thread는 즉시 다음 decode iteration 진행
- BG thread 완료 시 `[PreStage-BG] BG thread work completed in {bg_ms}ms (BG-internal wall)` 로그
- decode loop 종료 후 `handle.join()` → `[PreStage-BG] join complete: ...` 로그

---

## sync_baseline 결과 (5/5 PASS)

| run | Prefill (ms) | Swap (ms) | TTFT (ms) | tok[0] (ms) | Decode (ms/tok) | Avg TBT (ms) |
|-----|-------------:|----------:|----------:|------------:|----------------:|-------------:|
| 1   | 106.50 | 269.6 | 481.9 | 26.76 | 25.95 | 28.50 |
| 2   | 103.97 | 392.9 | 604.0 | 26.52 | 25.76 | 27.26 |
| 3   | 105.96 | 488.3 | 701.4 | 26.36 | 25.93 | 29.61 |
| 4   | 106.31 | 484.8 | 700.8 | 26.63 | 25.94 | 29.17 |
| 5   | 107.57 | 485.1 | 705.5 | 27.05 | 25.96 | 28.63 |
| **mean** | **105.7** | **424.1** | **638.7** | **26.66** | **25.91** | **28.63** |
| **stdev** | 1.30 | 100.4 | 99.1 | 0.27 | 0.08 | 0.86 |

- swap stage breakdown은 `mmap_permute=82~99%` (예: run 4 = `prefault=4.2ms mmap_permute=478.9ms arc_swap=0.0ms`)
- Decode tok[0]도 정상 수준 (≈26.6 ms)으로 swap 후 첫 토큰 inflate 없음

---

## pre_stage_v2 결과 (5/5 CRASH)

| run | Prefill (ms) | BG_done | join | scudo_abort (logcat) | Exit | 마지막 출력 |
|-----|-------------:|--------:|-----:|---------------------:|-----:|------------|
| 1 | 219.38 | 0 | 0 | 2 | 134 (SIGABRT) | "the lazy dog. 1" |
| 2 | 219.45 | 0 | 0 | 2 | 134 (SIGABRT) | "the lazy dog. The first" |
| 3 | 221.83 | 0 | 0 | 0* | 134 (SIGABRT) | "the lazy dog. If" |
| 4 | 224.74 | 0 | 0 | 2 | 134 (SIGABRT) | "the lazy dog. How" |
| 5 | 221.07 | 0 | 0 | 2 | 134 (SIGABRT) | "def ca" |

\*run 3은 logcat 캡처 타이밍이 어긋나 Scudo 로그 미수집. 종료 패턴(exit 134, 같은 prefill 시간, 동일한 짧은 decode 후 abort)은 동일.

### Crash 분석 (tombstone)

```
pid: 12357, tid: 12364, name: AdrenoOsLib
signal 6 (SIGABRT), code -1 (SI_QUEUE)
Abort message: 'Scudo ERROR: invalid chunk state when deallocating address 0x200007b8ecacc00'

backtrace:
  #00 abort+160                                                      /libc.so
  #01 scudo::die()+12                                                 /libc.so
  #02 scudo::reportRawError                                           /libc.so
  #03 scudo::reportInvalidChunkState                                  /libc.so
  #04 scudo::Allocator::deallocate                                    /libc.so
  #05~#08 libCB.so   (Adreno command buffer manager, 4 frames)
  #09 libgsl.so::os_thread_launcher                                   /vendor/lib64/libgsl.so
  #10 __pthread_start                                                 /libc.so
  #11 __start_thread                                                  /libc.so
```

- 죽는 thread: `AdrenoOsLib` — Adreno OpenCL driver의 자체 background thread (libgsl spawned)
- 원인 frame: `libCB.so` (Adreno command buffer manager)가 chunk를 free하는데 Scudo가 invalid state 감지
- 즉, application code가 cl_mem 또는 OpenCL queue를 driver thread와 race하면서 driver 내부 메모리 구조 손상

### Decode와 BG swap의 timing

- v2 prefill은 sync보다 2배 느림 (220 vs 106 ms). 추정 원인: secondary AOS mmap의 lazy fault가 prefill 중 처리됨 (sync는 prefill 전 swap으로 미리 fault 처리)
- BG thread spawn 직후 decode 토큰 약 5~10개 정상 출력 (sync와 같은 "the lazy dog" 시퀀스)
- 그 직후 driver thread crash → main process kill
- `BG thread work completed` 메시지는 5/5 모두 미출력 → BG thread는 290ms 짜리 work 완료 못 함
- `join complete` 메시지도 5/5 모두 미출력 → main thread도 join 도달 못 함

---

## 판정

| 기준 | 관측 | 결론 |
|------|------|------|
| Strong success (TBT 인플레 거의 없음, BG total ≈ 290ms) | — | 미관측 |
| Partial success | — | 미관측 |
| Negative (Adreno serialize) | — | 측정 불가 (crash 선행) |
| CPU memcpy bound | — | 측정 불가 (crash 선행) |
| **CRASH (driver thread-safety violation)** | **5/5** | **Confirmed** |

**Final**: NEGATIVE — driver-level deterministic crash.

---

## Root cause 추정 및 LISWAP-2와의 관계

LISWAP-2는 multi-queue 시도 중 Adreno가 application multi-queue를 직렬화한다는 결과로 종결됐다. v2는 이보다 한 단계 강한 race를 시도(application multi-queue가 아니라 multi-thread에서 같은 context의 cl_mem release/enqueue 중첩) → driver 자체가 thread-safety를 보장하지 않는 영역에 진입.

구체 race 후보:
1. **cl_mem release race**: BG thread의 `executor.execute_on_slots`가 옛 primary cl_mem을 release하는 사이, main thread의 decode가 같은 cl_mem(혹은 그것이 가리키는 BO/host mapping)을 kernel arg로 enqueue
2. **command buffer manager double-touch**: `libCB.so`는 process-wide lock을 가정한 듯, 두 application thread가 enqueue를 동시에 호출하면 chunk lifetime이 깨짐
3. **ArcSwap mid-flight commit**: BG thread가 LayerSlot의 ArcSwap에 새 weights pointer를 swap하는 순간, main thread가 같은 layer의 forward pass에서 옛 weights pointer를 읽고 있을 가능성

(2)가 직접 원인일 가능성이 가장 높다 — backtrace가 `libCB.so` 4 frame에서 시작하므로, swap의 cl_mem release 또는 새 cl_mem allocate가 driver의 command buffer 자료구조를 손상시켰다.

---

## 권장 후속 조치

1. **v2 완전 보류** (동기화 없는 BG thread + GPU enqueue는 Adreno에서 production 불가능)
2. 만약 BG swap을 시도한다면:
   - BG thread는 **CPU-only 작업**(mmap fault, permute, F16→Q4 변환)만 수행
   - cl_mem allocate/release/upload는 main thread로 옮기거나 mutex로 직렬화
   - ArcSwap commit도 main thread의 decode iteration 경계에서만 수행
3. 또는 LISWAP-4 (intra-forward layer-aligned swap, 핸드오프: `.agent/todos/handoff_liswap_4_intra_forward_swap.md`) 트랙으로 우회. forward path 안에서 same-thread로 swap을 layer-by-layer 진행하면 driver multi-thread race 회피 가능.

---

## Artifacts

- 측정 스크립트: `/tmp/swap_v2_measurements/run_sync.sh`, `run_v2.sh`
- 측정 raw 로그: `/tmp/swap_v2_measurements/sync_run_{1..5}.log`, `v2_run_{1..5}.log`, `v2_run_{1..5}_logcat.log`
- Tombstone(예시): `R3CY408S5SB:/data/tombstones/tombstone_07`, `tombstone_12`
- 빌드: `target/aarch64-linux-android/release/generate` (worktree HEAD `f3dc4a4`)
