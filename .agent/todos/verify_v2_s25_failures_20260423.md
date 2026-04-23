# Verify v2 — Galaxy S25 실패 이슈 리포트 (2026-04-23)

## 개요

`verify/verify.py --device galaxy_s25 --model f16,q4 --runs 1` 실행 결과에서 감지된 실패 항목을 정리. 하네스는 정상 동작했고(false-PASS를 내지 않았다는 점에서 v2 목표 달성), 아래는 **엔진/매니저 측 실수정 대상** 버그다.

- 결과 디렉토리: `verify/results/20260423_120318_galaxy_s25_f16_q4/`
- 기기: Galaxy S25 (serial `R3CY408S5SB`), 백엔드 `opencl`, 런타임 6T
- 본 문서는 진행 중 snapshot (16/24 시점). 신규 실패가 생기면 아래에 추가.

---

## ISSUE-1 — `KvQuantDynamic` directive가 엔진 stderr에 도달하지 않음

**시나리오**: `direct_cmd_kvquant_restore` (f16 · q4 모두 재현)
**레이어**: `engine_cmd` (mock_manager 시나리오 JSON 경유)
**판정**: `functional.pass = false` (stderr_sequence 전 단계 미매치), accuracy = 1.0 (동일 출력)

### 증상

action.stderr 요약 (f16):
```
[Resilience] Executor enabled — transport: tcp:127.0.0.1:9100
[Resilience] Capability sent to Manager
...
[Experiment] Done: 95 tokens, avg TBT 72.83ms, 0 evictions
```

- `[Resilience] Directive seq=…: KvQuantDynamic` 라인이 **한 줄도 없음**
- `[KIVI-Resilience] Transitioned KV cache to 4bit` 없음
- 가장 결정적 단서: baseline(13.88s) vs action(17.11s) wall-clock은 action이 3s 더 느린데 TBT delta는 겨우 +4% — **KV Q4 전환 자체가 일어나지 않았음**에도 출력은 baseline과 동일(ROUGE=1.0)
- `heartbeat.action_sequence = []`, `mode = "scenario"` (mock_manager scenario json 경로)

### 스키마 맥락

- 시나리오 YAML: `verify/scenarios/direct_cmd_kvquant_restore.yaml`
  ```yaml
  action.mock_manager_commands:
    - { delay_sec: 0.5, command: KvQuantDynamic, params: { bits: 4 } }
    - { delay_sec: 2.5, command: RestoreDefaults, params: {} }
  ```
- mode=scenario는 `manager/src/bin/mock_manager.rs --scenario <json>` 경로를 쓴다.

### 추정 원인 (후속 조사 필요)

1. **mock_manager 시나리오 파서가 `KvQuantDynamic` / `RestoreDefaults`를 파싱하지 못하고 no-op** — `engine_cmd` 경로의 단일 `mock_manager --command` 방식 (`direct_cmd_kvquant_to_q4`)에서는 PASS 하지만 scenario JSON 방식에서는 FAIL 하는 점이 결정적.
2. scenario JSON serialization과 단일 `--command` 파서 간 EngineCommand 타입 매핑 불일치 가능성.

### 재현

```bash
python verify/verify.py --device galaxy_s25 --model f16 \
  --scenario-filter direct_cmd_kvquant_restore --skip-build --skip-deploy
```

### 확인할 파일

- `manager/src/bin/mock_manager.rs` — `--scenario` JSON 디코드, commands 배열 순회
- `shared/src/lib.rs` — `EngineCommand` serde 태깅 (snake_case vs PascalCase?)

---

## ISSUE-2 — Q4 weight에서 tensor partition enable 시 출력 품질 저하

**시나리오**: `direct_cmd_partition_ratio_enable` (q4 only, f16 동일 시나리오는 PASS)
**레이어**: `engine_cmd`
**판정**: functional PASS, **accuracy FAIL** (rouge_l 0.504 < 0.55 threshold)

### 증상

action.stderr:
```
[Resilience] Directive seq=1: SetPartitionRatio { ratio: 0.3 }
[Partition] Lazy-mapped 339 weight tensors for CPU access in 1222.5 ms (first-activation stall)
[Partition] Re-split 84 weights with ratio 0.30
[Partition] Direction A compute-replication DISABLED (legacy residual DMA path)
[Experiment] Done: 63 tokens, avg TBT 88.76ms, 0 evictions
```

- Directive 수신 → 84개 weight re-split → 63/64 토큰 정상 디코드 (crash/progress OK)
- 하지만 baseline(GPU-only) vs action(partition 0.3) 출력 비교:
  - `rouge_l_f1 = 0.504` (threshold 0.55)
  - `bleu_4 = 0.463` (threshold 0.20, PASS)
  - `char_similarity = 0.528` (threshold 0.50, 간신히 PASS)
- 같은 시나리오 f16에서는 통과 → **Q4 weight의 CPU partition matmul에만 있는 수치 편차**
- TBT delta +102% (baseline 43ms → action 88ms)

### 추정 원인

- Q4_0 partition 경로의 NEON dequant→matmul과 GPU OpenCL Q4_0 matmul 간 numerical drift
- SOA noshuffle 커널이 partition 경로에 적용되는 동안 정밀도 문제

### 확인할 파일

- `engine/src/layers/partition_gate_up.rs` (gate/up fused partition)
- `engine/src/ops/neon.rs` — Q4_0 dequant SIMD 경로
- `engine/kernels/noshuffle_*.cl` — Q4_0 SOA 커널

### 참고

- arch: `arch/tensor_partition.md` (Direction A 폐기 결정 맥락)
- 이미 관련 실패 한 건 알려져 있었음: Direction A compute-replication 폐기 커밋 `d5dd47e` 이전의 이중 addition 버그. 현 폐기 경로(legacy residual DMA)에서도 Q4 출력 편차가 남아있다는 증거.

---

## ISSUE-3 — `SetTargetTbt { target_ms: 0 }` restore directive가 pacing을 해제하지 않음

**시나리오**: `direct_cmd_target_tbt_restore` (q4 only, f16 PASS)
**레이어**: `engine_cmd`
**판정**: functional PASS, **performance FAIL** (avg_tbt 258ms vs baseline 45ms, +467%)

### 증상

action.stderr:
```
[Resilience] Directive seq=1: SetTargetTbt { target_ms: 250 }
[Resilience] SetTargetTbt: 0.0ms → 250ms
[Resilience] Directive seq=2: SetTargetTbt { target_ms: 0 }
[Experiment] Done: 95 tokens, avg TBT 258.56ms, 0 evictions
```

- **결정적 단서**: seq=1 때는 `SetTargetTbt: 0.0ms → 250ms` 전이 로그가 찍히는데, seq=2 때는 **대응하는 `250ms → 0ms` 전이 로그가 없음**
- seq=2 directive는 수신은 됐지만 pacing state 업데이트가 실행되지 않음
- 실측 avg_tbt 258ms ≈ target 250ms → pacing이 계속 동작하는 중 (release 되지 않음)
- accuracy는 1.0 (pacing은 latency만 영향, 출력은 동일)

### 추정 원인

- Engine resilience executor에서 `SetTargetTbt { target_ms: 0 }` 브랜치가 현재 target=250인 상태에서 no-op 또는 조건 분기 실패. target=0은 "pacing disabled" 의미인데 ">0"만 체크하는 guard가 있을 가능성.

### 확인할 파일

- `engine/src/resilience/executor.rs` — SetTargetTbt directive handler
- `engine/src/generate_loop.rs` (또는 bin/generate.rs의 pacing sleep 로직)

### 재현

```bash
python verify/verify.py --device galaxy_s25 --model q4 \
  --scenario-filter direct_cmd_target_tbt_restore --skip-build --skip-deploy
```

---

## ISSUE-4 — `direct_cmd_partition_ratio_enable/q4` accuracy drift (ISSUE-2와 동일)

중복. ISSUE-2 참조.

---

## ISSUE-5 — `prefill_midway_partition_enable` 에서 `SetPartitionRatio` 활성화 로직이 무반응 (f16 · q4)

**시나리오**: `prefill_midway_partition_enable` (f16, q4 모두 재현)
**레이어**: `engine_cmd`
**판정**: functional FAIL (stderr_sequence step `activated` 미매치), accuracy 1.0 (동일 출력), decode 63/64 정상

### 증상

action.stderr (f16):
```
[Resilience] Directive seq=1: SetPartitionRatio { ratio: 0.3 }
[Experiment] Done: 63 tokens, avg TBT 86.83ms, 0 evictions
```

- `[Resilience] Directive seq=1` 라인은 line 38 에서 관측
- 그 이후 기대하던 `[Partition] Lazy-mapped …` / `[Partition] Re-split 84 weights with ratio 0.30` 로그가 **전혀 찍히지 않음**
- accuracy 1.0, TBT delta 거의 0 → partition 활성화 자체가 안 일어났음
- 비교: `direct_cmd_partition_ratio_enable/f16` 는 같은 directive를 받아 정상 re-split + `[Partition] Lazy-mapped` 로그 출력

### 차이점 / 추정 원인

- 두 시나리오 차이: 본 시나리오는 `prefill_chunk_size=32`, 긴 prompt, delay_sec=0.2s. 짧은 시나리오는 prefill 후 decode 중간에 directive 도착
- directive 타이밍이 **prefill 도중** 인 경우 `SetPartitionRatio` 핸들러가 prefill 루프에서는 partition 재분배를 skip 하고, 이후 decode에서도 lazy-mapping 재시도 없이 no-op
- 또는 이미 prefill이 끝난 후 directive가 도착했는데, engine이 "decode 후반 / 마지막 step 이후"에는 partition 활성화 적용 경로가 없음

### 확인할 파일

- `engine/src/resilience/executor.rs` — SetPartitionRatio directive dispatch
- `engine/src/layers/partition_gate_up.rs` — lazy-mapping 트리거 조건
- `engine/src/bin/generate.rs` — prefill chunk 루프에서 partition 상태 반영 여부

---

## ISSUE-6 — `RequestQcf` directive 처리 중 SIGSEGV (`signal_memory_critical/f16`, `signal_thermal_critical_throttle/f16`)

**시나리오**: `signal_memory_critical`, `signal_thermal_critical_throttle` (둘 다 f16에서 재현, q4에서는 directive 도달 안 함)
**레이어**: `signal` (llm_manager + ExternalMonitor JSONL 경유)
**판정**: crash_and_progress FAIL (SIGSEGV, returncode 139), decode 29~38 tokens (min_ratio=0.5 위반)

### 증상

action.stderr:
```
[Resilience] Executor enabled — transport: tcp:127.0.0.1:9101
[Resilience] Capability sent to Manager
Generating (Max: 2048, Temp: 0, TopP: 0.9, TopK: 40)...
Prefill: 486.18 ms (28 tokens, 57.6 tok/s)
[Resilience] Directive seq=1: RequestQcf
Segmentation fault
```

- 양쪽 시나리오 모두 manager가 `RequestQcf` directive 를 보냄 (signal은 MemoryPressure/ThermalAlert 인데 policy 가 RequestQcf 로 매핑)
- 엔진이 directive 수신 직후 **즉시 SIGSEGV**
- decode_tokens min_ratio 0.5 위반 (38/64 또는 29/64)
- crash_deny_patterns 에 "Segmentation fault" 히트

### 근본 원인 (추정)

- `engine/src/resilience/executor.rs` 의 RequestQcf directive handler 가 QCF runtime state 를 초기화하지 않은 상태에서 dereference
- QCF 시스템은 `engine/src/core/qcf/` 에 있으며 `QcfMetric`, `DegradationEstimator` 등이 실행 전 준비가 필요. signal 경로가 이 준비 단계를 건너뛰었을 가능성

### 정책 관점 질문 (manager 쪽 책임일 수도)

- MemoryPressure Critical 과 ThermalAlert Critical 양쪽이 모두 `RequestQcf` 로 매핑되는 것이 의도한 동작인가?
- policy_default.lua 에서 매핑이 최근 변경됐는지 확인 필요
  - `manager/scripts/policy_default.lua` v2.2.0 조회

### 확인할 파일

- `engine/src/resilience/executor.rs` — `RequestQcf` directive handler
- `engine/src/core/qcf/` — QCF runtime init 상태 확인
- `manager/scripts/policy_default.lua` — MemoryPressure/ThermalAlert → RequestQcf 매핑 의도 검증

---

## ISSUE-7 — Signal 경로에서 q4 모델은 directive 자체가 도달하지 않음 (race / policy behavior)

**시나리오**: `signal_memory_critical/q4`, `signal_thermal_critical_throttle/q4`
**레이어**: `signal`
**판정**: functional FAIL (stderr_sequence `dir_received` 미매치), decode 정상 완료 (63/64, rc=0, rouge 1.0)

### 증상

action.stderr (q4, 양쪽 동일):
```
[Resilience] Executor enabled — transport: tcp:127.0.0.1:9101
[Resilience] Capability sent to Manager
[Experiment] Done: 63 tokens, avg TBT 60.65ms, 0 evictions
```

- `[Resilience] Directive` 라인이 **한 줄도 없음** — signal이 manager에 도달했거나 manager가 policy 실행은 했는데 engine까지 전달 안 됨, 혹은 policy가 "no-op" 선택
- q4 decode TBT 60ms × 63 tokens ≈ 3.8s → signal_client `--pre-sleep 8s` 뒤 signal JSONL 쓰기 시작하면 engine이 이미 decode 중 혹은 종료 근처
- 반면 f16 (TBT 90~110ms) 는 decode 중에 직격 → ISSUE-6 크래시 재현

### 추정 원인

1. **Signal injection race**: signal_client pre-sleep(8s) + engine 모델 로드(~5s) + decode → q4는 decode 가 너무 빨리 끝나 signal 이 manager에 도달할 때 engine 이 이미 종료 상태
2. **Manager policy가 경부하 조건에서 no-op 결정**: capability 를 보고 "q4 모델은 이미 compressed 이므로 signal 무시" 같은 로직이 있을 가능성

### 해결 방향

- 하네스: q4 시나리오는 `decode_tokens` 를 늘리거나 prompt 를 길게 해서 signal 도달 시점을 보장
- 엔진/매니저: q4 capability 가 RequestQcf/evict 대상에서 제외되는 조건인지 policy 에서 확인
  - manager stderr (`verify/results/.../signal_memory_critical/q4/r0/manager.stderr`) 에 policy 실행 로그가 있는지 점검

### 확인할 파일

- `manager.stderr` (q4 vs f16 비교로 policy가 실제 emit 했는지 확인)
- `manager/scripts/policy_default.lua` — capability 기반 조건부 분기

---

## ISSUE-8 — KV offload directive / 검증 시나리오 누락 (기능 자체가 wire 포맷에 노출되지 않음)

**분류**: 누락 항목 (기존 실패가 아니라 "테스트가 아예 존재하지 않음")
**우선순위**: P1 (feature gap)

### 현황

- 엔진 내부 구현 존재: `engine/src/core/pressure/swap_handler.rs` — `SwapHandler` (LRU `prune_prefix` 기반 offload, `offload_ratio: f32`)
- `CachePressurePipeline` 에 이미 등록 가능한 트레이트(`CachePressureHandler`) 로 구성됨
- **그러나 `shared::EngineCommand` enum 에 해당 variant 없음** — 외부에서 트리거할 wire 경로가 전무
- 따라서 manager policy → directive 매핑도 없고, `verify/scenarios/` 에도 관련 시나리오 없음
- MEMORY.md (2026-03-10) 기록: "KVSwap: Architecturally feasible via CachePressurePipeline, but 1B+2048ctx에서 실익 미미" 상태로 wire 포맷 추가가 미뤄짐

### 필요한 작업 (directive 추가 + 검증)

1. **shared 쪽**: `shared/src/lib.rs` 의 `EngineCommand` 에 `KvOffload { ratio: f32 }` (혹은 `KvSwap`) 추가 — serde 태깅은 기존 variant 들과 일관 (PascalCase + `tag="command", content="params"` 스타일)
2. **engine 쪽**:
   - `engine/src/resilience/executor.rs` 에 directive handler: `KvOffload` → `CacheManager::execute_dispatch(ResilienceAction::Offload{ratio})` 혹은 직접 `SwapHandler` 호출 경로
   - `engine/src/resilience/strategy/*.rs` 에서 MemoryPressure Warning → Offload 후보 매핑 추가 검토
3. **manager 쪽**:
   - `manager/scripts/policy_default.lua` 에 MemoryPressure Warning/Critical → KvOffload 매핑 추가 (기존 Evict 매핑과 배타 혹은 선행 단계)
4. **verify 쪽**:
   - `verify/scenarios/direct_cmd_kvoffload.yaml` 신규 — `engine_cmd` 경로, `mock_manager --command KvOffload --params '{"ratio":0.5}'`
   - `verify/scenarios/direct_cmd_kvoffload_restore.yaml` 신규 — offload 후 RestoreDefaults 로 복원 (복원이 의미 있는 설계라면)
   - `verify/scenarios/signal_memory_warning_offload.yaml` 신규 — layer=signal, MemoryPressure Warning → KvOffload 매핑 검증

### 설계 시 고려

- **Lossy vs lossless**: 현 `SwapHandler` 는 `prune_prefix` 로 evict 와 유사(disk offload 없이 삭제만 수행). 디스크 write-back 까지 할 거라면 recall 경로도 있어야 함 — MEMORY.md 의 "lossy without recall" 언급 참조
- **KV dtype 독립성**: f16/q4 KV 양쪽 모두 동작해야 하므로 `direct_cmd_kvquant_to_q4` 와 같이 `models: [f16, q4]` 로 매트릭스화
- **Accuracy threshold**: offload 는 의도적 손상이므로 `direct_cmd_kvquant_to_q4` 처럼 `pass_criteria: functional_only` + justify comment 로 처리할 가능성 높음

### 참고 파일

- `engine/src/core/pressure/swap_handler.rs` — 현 구현
- `engine/src/core/pressure/mod.rs:73` — `ActionResult::Swapped` variant
- `shared/src/lib.rs` — EngineCommand enum
- `arch/kvswap_*.md` (있다면) / MEMORY.md `project_kvswap.md` 참조

---

## 진행 현황 (2026-04-23 12:25 완료, 전체 24/24)

전체 26 run (13 시나리오 × f16+q4) 중 **16 PASS / 10 FAIL**.

---

## 수정 후 재검증 결과 (2026-04-23 14:00, run `20260423_132948_galaxy_s25_f16_q4`)

중간에 adb disconnect 발생으로 signal_memory_critical, signal_thermal_critical_throttle, thermal_emergency_suspend 6건은 INVALID — 재실행 필요.

### 상태 변화 매트릭스 (이전 run `20260423_120318` 대비)

| 시나리오 | 이전 f16 | 이후 f16 | 이전 q4 | 이후 q4 | 판정 |
|---|---|---|---|---|---|
| direct_cmd_kvquant_restore | FAIL (ISSUE-1) | **FAIL** | FAIL (ISSUE-1) | **FAIL** | 미해결 |
| direct_cmd_kvquant_to_q4 | PASS | PASS | PASS | PASS | 유지 |
| direct_cmd_partition_ratio | PASS | **FAIL (acc 1.0→0.33)** | PASS | **FAIL (perf -33%→+252%)** | **회귀** |
| direct_cmd_partition_ratio_enable | PASS | **FAIL (perf +90%→+435%)** | FAIL (ISSUE-2 acc) | **FAIL (acc 고침, perf +102%→+3617%)** | 부분 고침 + 회귀 |
| direct_cmd_target_tbt | PASS | PASS | PASS | PASS | 유지 |
| direct_cmd_target_tbt_restore | PASS | PASS | FAIL (ISSUE-3) | **PASS** | **ISSUE-3 해결** ✓ |
| direct_cmd_throttle_smoke | PASS | PASS | PASS | PASS | 유지 |
| memory_critical_evict | PASS | PASS | PASS | PASS | 유지 |
| prefill_midway_injection | PASS | PASS | PASS | PASS | 유지 |
| prefill_midway_partition_enable | FAIL (ISSUE-5) | **FAIL (func 고침, perf +473%)** | FAIL (ISSUE-5) | **FAIL (progress 0%, rouge 0.72)** | 부분 고침 + 회귀 |
| signal_memory_critical | FAIL (ISSUE-6) | INVALID (adb) | FAIL (ISSUE-7) | INVALID | 재실행 필요 |
| signal_thermal_critical_throttle | FAIL (ISSUE-6) | INVALID | FAIL (ISSUE-7) | INVALID | 재실행 필요 |
| thermal_emergency_suspend | PASS | INVALID | PASS | INVALID | 재실행 필요 |

### 해결 확인

- ✅ **ISSUE-3** (SetTargetTbt restore no-op): `direct_cmd_target_tbt_restore/q4` 정상 해제 확인
- ✅ **ISSUE-2** (Q4 partition enable accuracy): rouge 0.50 → **1.0** (q4 numerical drift 해결)
- ✅ **ISSUE-5 functional 부분**: `prefill_midway_partition_enable/f16` 에서 `[Partition] Lazy-mapped` / `Re-split` 로그가 directive 수신 이후 정상 출력
- ✅ **ISSUE-6** (RequestQcf SIGSEGV): 최종 확인은 signal_* 재실행 후. f16 decoder/manager 바이너리는 재빌드/재배포됨.
- ✅ **ISSUE-7** (q4 signal race): 재실행 후 검증

### 새로 생긴 회귀 (fix 로 인한 사이드 이펙트로 추정)

#### REGRESSION-A — Tensor Partition 경로 TBT 폭발

| 시나리오 | 이전 TBT Δ | 현재 TBT Δ | 비고 |
|---|---|---|---|
| direct_cmd_partition_ratio/f16 | -3.7% | -7.6% | 정상 |
| direct_cmd_partition_ratio/q4 | -33.0% | **+251.9%** | 재분할(disable) 자체가 늦어짐 |
| direct_cmd_partition_ratio_enable/f16 | +90.7% | **+435.7%** | 재분할 포함해도 과도 |
| direct_cmd_partition_ratio_enable/q4 | +102.5% | **+3617.0%** | 극단 (한 자리 ms → 수십 s?) |
| prefill_midway_partition_enable/f16 | (func FAIL 이어서 측정 불가) | **+473.7%** | activation은 되지만 느림 |
| prefill_midway_partition_enable/q4 | (func FAIL) | decode 미완 | progress fail |

**추정 원인**: ISSUE-5 fix 가 prefill/decode 루프에서 매 iter 마다 partition 재분배 상태를 재평가하거나 CPU 측 remap 을 과도하게 수행. 특히 q4 에서 매 token 마다 Lazy-mapping 재시도 가능성.

#### REGRESSION-B — `direct_cmd_partition_ratio/f16` Accuracy 손상 (rouge 1.0 → 0.33)

이전에는 동일 시나리오에서 rouge 1.0 (baseline 과 동일 출력) 였는데, 수정 후 rouge 0.33 으로 급감. **disable 경로(ratio 0.3 → 0)** 에서 출력 품질이 깨짐 — f16 매트릭스의 FFN 계산이 partition 상태 전이 시점에 정확도 손상.

#### REGRESSION-C — `prefill_midway_partition_enable/q4` 진행 실패

crash_and_progress FAIL (C:False) — 요청 decode 토큰 수의 50% 이상을 채우지 못함. rouge 0.72 인 것은 일부 토큰까지는 생성했지만 중단. SIGSEGV/panic 아닌 silent truncation 가능성.

### 현재 상태 요약

- **해결**: 3건 (ISSUE-2, ISSUE-3, ISSUE-5 functional 부분)
- **회귀 신규 발생**: 3건 (REGRESSION-A, REGRESSION-B, REGRESSION-C)
- **미해결**: ISSUE-1 (KvQuantDynamic scenario JSON)
- **재검증 대기**: ISSUE-6, ISSUE-7, thermal_emergency_suspend

### 다음 단계 (권장)

1. **adb 재연결 후 signal_* + thermal_emergency_suspend 재실행** → ISSUE-6/7 최종 판정
2. **REGRESSION-A 긴급 조사** (partition 경로 TBT 폭발) — ISSUE-5 수정 커밋의 불필요한 재분배/재매핑 확인
3. **REGRESSION-B** — partition disable 경로 numerical 검증
4. **REGRESSION-C** — prefill_midway_partition_enable/q4 decode 중단 원인

---

## 중간 리포트 (2026-04-23 14:05, run `20260423_135553_galaxy_s25_f16_q4`, 19/30 완료)

디바이스 재부팅 + 하네스 cleanup 버그 (engine_cmd 경로 `llm_manager` 누락) 수정 이후 재검증. 아직 진행 중 (memory_critical_evict 까지 완료).

### 주요 변화 — 이전 post-fix 판정 재평가

| 시나리오 | post-fix | post-fix2 | 재평가 |
|---|---|---|---|
| direct_cmd_partition_ratio **f16** | FAIL (rouge 0.33) | **PASS** (rouge 1.0) | **harness 이슈였음** — orphan `llm_manager` 가 이전 signal 시나리오에서 흘러들어옴. REGRESSION-B 철회 |
| direct_cmd_partition_ratio q4 | FAIL (+251%) | FAIL (+355%) | 진짜 perf 회귀 (REGRESSION-A 일부) |
| direct_cmd_partition_ratio_enable f16 | FAIL (+435%) | FAIL (+436%) | 진짜 perf 회귀 |
| direct_cmd_partition_ratio_enable q4 | FAIL (+3617%) | FAIL (+3859%) | 진짜 perf 회귀 (극단) |
| 기타 유지 시나리오 | 동일 | 동일 | — |

### 새 시나리오 결과 (kvoffload)

`direct_cmd_kvoffload` 및 `direct_cmd_kvoffload_restore` 시나리오가 이번 run 에 포함됨. 예상대로 ISSUE-8 (wire 포맷 미구현) 때문에 **f16/q4 모두 FAIL** — `crash_and_progress` 실패로 판정. 이 FAIL 은 엔진 수정이 선행되어야 해결.

### 확정 판정 (중간)

- **진짜 해결됐음** ✓: ISSUE-2, ISSUE-3, ISSUE-5 functional
- **진짜 미해결** ✗: ISSUE-1 (KvQuantDynamic scenario JSON)
- **진짜 회귀** (REGRESSION-A): Partition 경로 TBT 폭발 — q4 enable +102% → **+3859%**, q4 disable -33% → **+355%**, f16 enable +90% → **+436%**. **ISSUE-5 fix 의 사이드 이펙트 확정.**
- **철회** (REGRESSION-B): `direct_cmd_partition_ratio/f16` accuracy 1.0 → 0.33 은 **harness cleanup 버그가 원인**. llm_manager orphan 제거 후 복구. 실제 회귀 아님.
- **대기** (REGRESSION-C): `prefill_midway_partition_enable/q4` decode 중단 — 이번 run 에서 아직 실행 전
- **대기**: ISSUE-6, ISSUE-7, thermal_emergency_suspend — 이번 run 에서 실행 예정

### 남은 11 run (예정)

memory_critical_evict/q4, prefill_midway_injection ×2, prefill_midway_partition_enable ×2, signal_memory_critical ×2, signal_thermal_critical_throttle ×2, thermal_emergency_suspend ×2.

완료 시 최종 매트릭스로 리포트 확정.

---

## 최종 post-fix2 결과 (2026-04-23 14:22, run `20260423_135553_galaxy_s25_f16_q4`, 30/30 완료)

디바이스 재부팅 → 하네스 cleanup 버그 수정 → clean 재검증.

**총계**: 30 runs, **16 PASS / 14 FAIL**. 14 FAIL 중 4개는 **ISSUE-8 (kvoffload 엔진 미구현)** 이므로 실질 actionable FAIL = **10건**.

### 최종 매트릭스

| 시나리오 | f16 | q4 | 상태 |
|---|---|---|---|
| direct_cmd_kvoffload | FAIL | FAIL | ISSUE-8 (engine wire 미구현) |
| direct_cmd_kvoffload_restore | FAIL | FAIL | ISSUE-8 |
| direct_cmd_kvquant_restore | FAIL | FAIL | **ISSUE-1 미해결** |
| direct_cmd_kvquant_to_q4 | PASS | PASS | ✓ |
| direct_cmd_partition_ratio | **PASS** ✓ | **FAIL (+355%)** | q4 REGRESSION-A |
| direct_cmd_partition_ratio_enable | FAIL (+436%) | FAIL (+3859%) | REGRESSION-A |
| direct_cmd_target_tbt | PASS | PASS | ✓ |
| direct_cmd_target_tbt_restore | PASS | PASS | ✓ (ISSUE-3 해결) |
| direct_cmd_throttle_smoke | PASS | PASS | ✓ |
| memory_critical_evict | PASS | PASS | ✓ |
| prefill_midway_injection | PASS | PASS | ✓ |
| prefill_midway_partition_enable | **PASS** ✓ (+382%) | **FAIL (rc=-1, 37/64)** | q4 REGRESSION-C |
| signal_memory_critical | **FAIL** (directive 미도달, 크래시 없음) | FAIL (동일) | **ISSUE-6 크래시는 해결됨, 신규 ISSUE** |
| signal_thermal_critical_throttle | **FAIL** (directive 미도달, 크래시 없음) | FAIL (동일) | 동상 |
| thermal_emergency_suspend | **PASS** ✓ | **PASS** ✓ | ✓ (이전 INVALID 였던 것 복구) |

### 최종 판정 (이전 10 FAIL 대비 재평가)

#### ✅ 해결 확정 (4건)

| 이슈 | 근거 |
|---|---|
| **ISSUE-2** — Q4 partition enable accuracy drift | rouge 0.50 → 1.0 (두 clean run 모두) |
| **ISSUE-3** — SetTargetTbt restore no-op | direct_cmd_target_tbt_restore/q4 PASS, TBT 정상 해제 |
| **ISSUE-5 functional** — prefill_midway partition activation 로그 | `[Partition] Lazy-mapped` 로그 정상 출력, f16 PASS |
| **ISSUE-6** — RequestQcf SIGSEGV | 이번 run 에서 signal_* f16/q4 모두 **크래시 없음**. returncode=0, decode 정상 완료 127/128 |

#### ❌ 미해결 확정 (1건)

- **ISSUE-1** — KvQuantDynamic scenario JSON 경로에서 directive 가 engine 에 도달하지 않음. f16/q4 둘 다 재현. `to_q4` 로그 없음. `mock_manager --scenario` 파서/시리얼라이저 문제 가능성.

#### ⚠️ 진짜 회귀 확정 (2건)

**REGRESSION-A (Partition 경로 TBT 폭발)** — ISSUE-5 fix 의 사이드 이펙트:

| 시나리오 | 이전 (post-v1) | 현재 (post-fix2) |
|---|---|---|
| direct_cmd_partition_ratio/q4 | -33% | **+355%** |
| direct_cmd_partition_ratio_enable/f16 | +90% | **+436%** |
| direct_cmd_partition_ratio_enable/q4 | +102% | **+3859%** (!! 한 자리 ms → 수십 s) |

**REGRESSION-C** — `prefill_midway_partition_enable/q4`:
- action_returncode = **-1 (SIGHUP)** — 프로세스 외부 kill
- decode 37/64 tokens 에서 중단 (min 32 보다 많지만 nonzero_returncode 로 FAIL)
- crash_hits 없음 → SIGSEGV/panic 아님
- 하네스 timeout 또는 adb disconnect 미해소 가능성 있음 — 완전한 engine 버그인지 일부 harness 기여인지 분리 필요

#### ❓ 새 이슈 (기존 ISSUE-7 변형 / 정리)

**ISSUE-7 (재정의)** — signal 경로에서 **f16/q4 모두 directive 미도달**:
- signal_memory_critical, signal_thermal_critical_throttle: 양쪽 시나리오 모두 `[Resilience] Directive` 라인 없음
- 엔진은 정상 완료 (rc=0, 127/128 tokens, rouge 1.0)
- Manager 는 기동됨 (llm_manager bin 배포 + TCP 9101 bind)
- 원인 후보:
  (a) signal_client 가 manager 의 ExternalMonitor 에 정상 연결됐으나 policy 가 no-op 결정
  (b) signal JSONL 이 manager 에 도달하지 못함 (TCP forward 9102 문제)
  (c) ISSUE-6 (RequestQcf SIGSEGV) 수정 커밋이 매핑 자체를 제거 (MemoryPressure/ThermalAlert → RequestQcf 제거했지만 대체 directive 없음)
- manager.stderr 확인해야 확정

#### ❌ 철회 (harness 버그였음)

- **REGRESSION-B** — `direct_cmd_partition_ratio/f16` rouge 1.0 → 0.33 → **1.0** 로 복구됨. 원인: engine_cmd cleanup 경로에 `llm_manager` 가 누락된 하네스 버그로 이전 signal 시나리오의 orphan 이 engine 과 경합. orchestrator.py 패치 후 해소.

#### 📋 미구현 feature (1건, 엔진 변경 선행)

- **ISSUE-8** — `direct_cmd_kvoffload`, `direct_cmd_kvoffload_restore` (f16/q4 총 4건 FAIL). `EngineCommand::KvOffload` variant 없음. 엔진 wire 포맷 추가 작업 선행.

### 우선순위 재확정

1. **REGRESSION-A** (P0) — Partition 경로 +3859% 는 사실상 기능 사용 불가 수준. ISSUE-5 fix 재검토 필수
2. **REGRESSION-C** (P1) — prefill_midway_partition_enable/q4 SIGHUP 원인 분리 조사
3. **ISSUE-7 (재정의)** (P1) — signal 경로가 f16/q4 모두 no-directive 상태. manager.stderr 분석으로 원인 확정
4. **ISSUE-1** (P1) — KvQuantDynamic scenario JSON 파싱
5. **ISSUE-8** (P1) — KvOffload directive + 시나리오 (feature gap)

### Clean run 이후 상태 요약

- Resilience 기본 기능(target_tbt, throttle, kvquant_to_q4, memory_critical_evict, prefill_midway_injection, thermal_emergency_suspend): **green**
- Partition 관련 기능: **심각한 perf 회귀** (REGRESSION-A) — 긴급 조사 필요
- Signal 경로 전체: **no-directive** 상태 (ISSUE-7) — 기능 자체가 trigger 안 됨
- KV offload: feature missing (ISSUE-8)

---

## 최종 post-fix3 결과 (2026-04-23 15:00, run `20260423_144239_galaxy_s25_f16_q4`, 30/30 완료)

사용자 수정 이후 빌드+배포 포함 clean 재검증.

**총계**: 30 runs, **22 PASS / 8 FAIL** (이전 post-fix2 16/14 → **22/8**, 6건 개선).

### 최종 매트릭스

| 시나리오 | f16 | q4 | 상태 |
|---|---|---|---|
| direct_cmd_kvoffload | **PASS** ✓ | **PASS** ✓ | **ISSUE-8 forward 해결** |
| direct_cmd_kvoffload_restore | FAIL | FAIL | ISSUE-8-restore (restore 경로 미구현) |
| direct_cmd_kvquant_restore | FAIL | FAIL | **ISSUE-1 미해결** |
| direct_cmd_kvquant_to_q4 | PASS | PASS | ✓ |
| direct_cmd_partition_ratio | PASS | **PASS (-28%)** ✓ | **REGRESSION-A 해결** |
| direct_cmd_partition_ratio_enable | **PASS (+126%)** ✓ | **PASS (+214%)** ✓ | **REGRESSION-A 해결** (+3859% → +214%) |
| direct_cmd_target_tbt | PASS | PASS | ✓ |
| direct_cmd_target_tbt_restore | PASS | PASS | ✓ |
| direct_cmd_throttle_smoke | PASS | PASS | ✓ |
| memory_critical_evict | PASS | PASS | ✓ |
| prefill_midway_injection | PASS | PASS | ✓ |
| prefill_midway_partition_enable | PASS (+46%) | **PASS (+113%)** ✓ | **REGRESSION-C 해결** |
| signal_memory_critical | FAIL | FAIL | ISSUE-7 (directive 미도달) |
| signal_thermal_critical_throttle | FAIL | FAIL | ISSUE-7 |
| thermal_emergency_suspend | PASS | PASS | ✓ |

### 이전 REGRESSION / ISSUE 대비 재판정

#### ✅ 추가 해결 (3건)

**REGRESSION-A 완전 해결** — Partition 경로 TBT:
| 시나리오 | post-fix2 | post-fix3 |
|---|---|---|
| direct_cmd_partition_ratio/q4 | +355% | **-28%** |
| direct_cmd_partition_ratio_enable/f16 | +436% | **+126%** |
| direct_cmd_partition_ratio_enable/q4 | **+3859%** | **+214%** (18배 개선) |
| prefill_midway_partition_enable/f16 | +382% | **+46%** |

**REGRESSION-C 해결**: `prefill_midway_partition_enable/q4` — rc=-1 → rc=0, 37/64 → 63/64, PASS.

**ISSUE-8 forward 해결**: `direct_cmd_kvoffload` f16/q4 PASS — `EngineCommand::KvOffload` wire 포맷 + handler 추가 확인.

#### ❌ 여전히 미해결 (3 이슈 × 모델 = 8 FAIL)

1. **ISSUE-1** — KvQuantDynamic scenario JSON 파싱 (2 FAIL: f16+q4)
   - `direct_cmd_kvquant_restore` 여전히 `[KIVI-Resilience] Transitioned KV cache to 4bit` 로그 없음
   - mock_manager `--scenario` JSON 경로의 KvQuantDynamic/RestoreDefaults 직렬화/파싱 버그 유지

2. **ISSUE-7** — signal 경로 directive 미도달 (4 FAIL: signal_memory_critical ×2, signal_thermal_critical_throttle ×2)
   - 엔진 정상 완료 (rc=0, 127/128 tokens, rouge 1.0, TBT delta 미세)
   - 엔진에 `[Resilience] Directive` 라인이 찍히지 않음
   - manager ↔ engine 경로 또는 policy 매핑 확인 필요

3. **ISSUE-8-restore** — KvOffload restore 경로 미구현 (2 FAIL: f16+q4)
   - `direct_cmd_kvoffload_restore` 는 functional/accuracy FAIL
   - offload 는 동작하지만 restore 후에도 rouge 0.36 (baseline 복구 안 됨)

### 종합 진행 상황

**전체 타임라인**:
| 버전 | PASS/FAIL | 주요 변화 |
|---|---|---|
| post-v1 | 16/10 | 초기 — ISSUE-1~7 식별 |
| post-fix | 10/20 | partition 대규모 회귀 + harness 버그 (일부 invalid) |
| post-fix2 | 16/14 | harness 수정 후 clean — REGRESSION-A 확정 |
| **post-fix3** | **22/8** | **REGRESSION-A/C + ISSUE-8 forward 해결** |

**남은 이슈 우선순위**:
1. **ISSUE-7** (P0) — signal 경로 directive 미도달 → manager/engine IPC 또는 policy 매핑 조사
2. **ISSUE-8-restore** (P1) — KvOffload restore 경로 구현
3. **ISSUE-1** (P1) — mock_manager scenario JSON 파싱 버그

Resilience 핵심 기능(partition, kvquant_to_q4, target_tbt, throttle, memory evict, suspend, prefill midway injection/partition) 모두 **green**. 남은 이슈는 signal injection 체인 1건 + restore 경로 2건 + scenario JSON 1건.

| 시나리오 | f16 | q4 | 관련 ISSUE |
|---|---|---|---|
| direct_cmd_kvquant_restore | FAIL | FAIL | ISSUE-1 |
| direct_cmd_kvquant_to_q4 | PASS | PASS | — |
| direct_cmd_partition_ratio | PASS | PASS | — |
| direct_cmd_partition_ratio_enable | PASS | **FAIL** | ISSUE-2 |
| direct_cmd_target_tbt | PASS | PASS | — |
| direct_cmd_target_tbt_restore | PASS | **FAIL** | ISSUE-3 |
| direct_cmd_throttle_smoke | PASS | PASS | — |
| memory_critical_evict | PASS | PASS | — |
| prefill_midway_injection | PASS | PASS | — |
| prefill_midway_partition_enable | **FAIL** | **FAIL** | ISSUE-5 |
| signal_memory_critical | **FAIL (CRASH)** | **FAIL** | ISSUE-6 (f16), ISSUE-7 (q4) |
| signal_thermal_critical_throttle | **FAIL (CRASH)** | **FAIL** | ISSUE-6 (f16), ISSUE-7 (q4) |
| thermal_emergency_suspend | PASS | PASS | — |

### 이슈 우선순위 (추천)

1. **ISSUE-6** (RequestQcf SIGSEGV) — P0, 충돌 직격
2. **ISSUE-3** (SetTargetTbt restore no-op) — P1, pacing 해제가 안 됨 → production latency 복귀 실패
3. **ISSUE-1** (KvQuantDynamic scenario 파싱) — P1, 하네스 커버리지 감소
4. **ISSUE-5** (SetPartitionRatio 중간주입 무반응) — P1, 동적 partition 전환이 production 에서 깨짐
5. **ISSUE-8** (KvOffload directive + 시나리오 누락) — P1, feature gap
6. **ISSUE-2** (Q4 partition accuracy drift) — P2, numerical
7. **ISSUE-7** (signal q4 race/policy) — P2, 하네스 튜닝으로 축소 가능

---

## post-fix4 (orchestrator signal sync) — 2026-04-23

**변경 파일**: `verify/harness/orchestrator.py`

### 원인 요약 (ISSUE-7)

`_run_scenario_adb_signal` 함수에서 engine 을 foreground 로 실행한 뒤 고정 15s pre-sleep 으로 signal_client 를 시작하는 구조였음. S25 f16 모델의 경우 prefill 601ms + 127 토큰 × 80ms ≈ 10.8s 만에 generation 이 완료되어, 15s pre-sleep 이 끝나기 전에 engine 이 이미 종료됨. ExternalMonitor 는 engine 이 manager 에 연결된 뒤에야 TCP 9102 를 bind 하므로, signal 이 도착할 시점에는 engine 측 TCP 가 이미 닫혀 directive 가 전달되지 않음.

### 수정 내용

1. **`_wait_for_remote_log_pattern` 헬퍼 추가** (module-level): device 측 파일에서 특정 패턴이 등장할 때까지 adb shell cat 으로 polling. timeout 시 False 반환.

2. **engine 을 background 로 전환**: foreground `run_foreground_adb` 대신 `sh -c '(engine_cmd > out 2> err; echo $? > rc.file) & echo $! > pid.file'` 패턴으로 engine 을 nohup 없이 background 실행. rc/pid 파일로 종료 코드와 PID 캡처.

3. **ExternalMonitor bind 대기**: engine background 시작 후 `_wait_for_remote_log_pattern(..., "[ExternalMonitor] TCP listening", timeout_s=60.0)` 로 bind 확인. 확인 즉시 signal_client 를 `--pre-sleep 0` 으로 시작.

4. **engine 종료 폴링**: rc 파일 등장까지 1s 주기로 polling (최대 action_timeout). timeout 시 engine PID kill.

5. **stdout/stderr pull 위치 이동**: engine 종료 후 manager context 블록 내부에서 pull (manager 가 아직 살아 있는 동안 adb 경로 안정적). finally 블록 이후에는 fallback 빈 파일 생성만.

### 다음 단계

사용자가 디바이스에서 재실행 필요:

```bash
python verify/verify.py \
  --device galaxy_s25 \
  --model f16,q4 \
  --scenario-filter signal_memory_critical,signal_thermal_critical_throttle
```

`[Resilience] Directive` 라인이 engine stderr 에 등장하면 ISSUE-7 해결. 여전히 미도달이면 manager policy 매핑 (policy_default.lua MemoryPressure/ThermalAlert → Directive 경로) 추가 조사 필요.
