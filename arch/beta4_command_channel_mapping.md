# β-4 — EngineCommand → 신 채널 전수 매핑표 (α-W-3b 흡수 선행 산출물)

**작성**: 2026-06-10 (Architect). Phase β-4 코드 diff 전 단독 커밋되는 arch/ 부속 문서.
**SSOT**: `arch/pipeline_stage_design_v2.md` §5.4(2-source 모델) + §5.2.1(driver↔Stage 계약).
**정본 명세**: `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md` §β-4.
**코드 권위 기준**: HEAD=`428ed2d6`(코드 상태 `1f93d6b1`). 모든 라인 번호는 본 문서 작성 시점 **실측**이며, roadmap 기재와 다르면 `(roadmap :N → 실측 :M)` 으로 정오 표기한다.

이 문서는 β-4 host 게이트 **"매핑표 행별 등가 test"** 의 명세다 — 18-variant 전수 + 19필드 전수, 누락 0.

---

## 0. 범위와 정의

β-4 는 v1 의 `CommandSource → ExecutionPlan → run() 인라인 소비` 경로를 v2 의
`CommandSource(pure) → CommandDispatcher → {registry.submit(OneShot Stage) | LoopControl}` 로 분해한다.

- **구 채널** = `CommandExecutor::poll`(executor.rs:289-358)이 `apply_command`(executor.rs:360-571)를 통해 채우는 `ExecutionPlan`(executor.rs:19-65) 필드 + `DecodeLoop::run`(decode_loop.rs:205-485)의 인라인 소비.
- **신 채널** = v2 §5.4 A-1 의 3분류:
  - **① OneShot EvictionStage** — evict-family 4종 → `registry.submit(EvictionStage::one_shot(...))` (β-3 신설 `engine/src/stages/kv/eviction.rs`).
  - **② LoopControl** — control 7종(신설 구조체) + 과도기 5종(deprecated 필드 잔존).
  - **③ Hardware resolve seam** — SwitchHw/PrepareComputeUnit.

**[정오 — roadmap 라인 drift 요약]** (상세는 각 부의 표):

| 대상 | roadmap 기재 | 실측 | 비고 |
|---|---|---|---|
| EngineCommand variant 정의 | `:189-278` | `:189-278`(enum 본문), variant `:192-277` | 일치 |
| ExecutionPlan | `:19-65`(19필드) | `:19-65`(19필드) | 일치 |
| apply_command | `:360-571` | `:360-571` | 일치 |
| sticky carry(directives empty) | `:289-358` 안의 carry | `:306-317`(empty 분기) + `:337-342`(non-empty carry-forward) | roadmap 은 poll 전체 범위만 지시 — 실 carry 로직은 2곳 |
| RestoreDefaults reset | `:484-502` | `:484-502` | 일치 |
| heartbeat 송출 | `:292-296` | `:293-296` | roadmap `:292` 는 주석 라인, 실 발화 `:293-295` |
| `evict_applied` 필드 | decode_loop `:48` | `:53` | **drift +5** |
| `evict_applied` set | decode_loop `:169` | `:256-257` | **drift +87** (β-2/β-3 phase 삽입으로 run 본문 하향) |
| run() | decode_loop `:124-346` | `:205-485` | **drift** (β-1~3 누적) |
| prefill() | decode_loop `:79-105` | `:155-186` | **drift** |
| run_until_stop() | decode_loop `:356-497` | `:495-636` | **drift** |
| run_until_stop 소비부 | decode_loop `:382-389` | command poll `:512-528` | **drift** — roadmap `:382-389` 는 구버전 run_until_stop 의 poll 위치 |
| CmdSrcWrapper::poll | resilience_adapter `:37/:72` | 정의 `:70`, poll impl `:72-79`; `ResilienceAdapter::poll` `:37-41`(`:39` 가 executor.poll) | roadmap `:37` = ResilienceAdapter impl, `:72` = CmdSrcWrapper impl |
| on_kv_prune | model_forward `:474` | sig `:477`, 본문 `:477-489` | **drift +3** |
| NoOpCommandSource | defaults `:39` | `:37`(struct), impl `:39-43` | 일치(impl 기준) |
| ~~batch orphan poll~~ **삭제됨 (γ-4)** | ~~batch/runner `:495`~~ | **n/a** | **[정오 2026-06-11]** `session/batch/` 모듈이 γ-4(`2e53cf44`)에서 순수 삭제됨 — 이 orphan poll 호출처(`executor.poll` 직접 호출)가 소멸. SetPrefillPolicy 의 prefill 경로 유일 소비자가 사라져 **소비자 0** (1.2/1.3 참조) |
| chat run_until_stop 호출 | chat/session `:100` | `:109` | **drift +9** |
| chat turn-boundary try_evict | chat/session `:194/:286` | `:194`/`:286` | 일치 |

---

## 1부. EngineCommand → 신 채널 전수 매핑표 (18/18)

`EngineCommand`(shared/src/lib.rs:189-278) 18-variant. 각 variant 의 **구 채널**(apply_command 라인 → ExecutionPlan 필드) → **신 채널**(v2 §5.4) → **소비 지점**(decode_loop run()/run_until_stop()) 을 적시한다.

### 1.1 ① evict-family 4종 → `registry.submit(OneShot EvictionStage)`

| # | variant (정의 라인) | apply_command (라인) | ExecutionPlan 필드 | 신 채널 | run() 소비 지점 |
|---|---|---|---|---|---|
| 1 | `KvEvictH2o`(:212) | :397-409 (`EvictMethod::H2o`) | `evict: Option<EvictPlan>`(:21) | `EvictionStage::one_shot(handles, cm, keep_ratio)` → `registry.submit` | (a.5) :255-281 → β-4: dispatcher 가 submit, PreEviction(:315) 에서 소비 |
| 2 | `KvEvictSliding`(:215) | :411-426 (`EvictMethod::Sliding`) | `evict`(:21) | 동상 | 동상 |
| 3 | `KvStreaming`(:218) | :428-449 (`EvictMethod::Streaming` + `StreamingParams`) | `evict`(:21) (target_ratio=0.0) | 동상 (streaming sink/window 는 stage 가 보유한 cm 의 CLI 정책) | 동상 |
| 4 | `KvMergeD2o`(:224) | :451-463 (`EvictMethod::D2o`) | `evict`(:21) | 동상 | 동상 |

**설계 결정 (① 4종 공통)**: 4 variant 는 v1 에서 모두 `EvictPlan` 1개 필드(`evict`)로 수렴되고, `apply_command` 가 `EvictMethod` enum 으로 분기를 표현했다. v2 에서는 **method 정보가 신 채널을 통과하지 않는다** — (a.5) 가 directive 의 method 를 무시하고 CacheManager 의 CLI 구성 정책으로 force_evict 하기 때문(3부 참조). 따라서 4종 모두 동일한 `EvictionStage::one_shot(handles, cm, target_ratio)` 로 사상되며, `target_ratio` 만 directive 의 `keep_ratio`(KvStreaming 은 0.0 → cm CLI sink/window)에서 가져온다. EvictionStage 가 보유한 `cm` 이 정책(sliding/h2o/streaming/d2o)을 결정한다.

**sticky 등가**: v1 의 `self.evict_plan`(executor.rs:123) sticky carry(:311, :338) + driver 의 `evict_applied`(decode_loop.rs:53) 1회-게이트 = v2 의 OneShot Consumed GC 1회성. 상세는 2부.

### 1.2 ② control 7종 → `LoopControl` (신설)

| # | variant (정의 라인) | apply_command (라인) | ExecutionPlan 필드 | LoopControl 필드(제안) | run() 소비 지점 |
|---|---|---|---|---|---|
| 5 | `Throttle`(:192) | :362-372 | `throttle_delay_ms: u64`(:27) | `throttle_delay_ms` | (a) 직후 :244-246 (sleep) |
| 6 | `SetTargetTbt`(:196) | :373-389 | `target_tbt_ms`(:30) + `target_tbt_set`(:35) | `target_tbt_ms` + `target_tbt_set` | (h) pacing :462-470 |
| 7 | `Suspend`(:246) | :512-516 | `suspended: bool`(:37) | `suspended`(또는 `LoopControl::Stop`) | (a) 직후 :240-243 (break, **[G6] break 보존**) |
| 8 | `Resume`(:248) | :517-526 | `resumed: bool`(:39) | `resumed` | (현 run() 미소비 — executor 내부 state 만; LoopControl seam 잔존) |
| 9 | `RestoreDefaults`(:240) | :484-502 | `restore_defaults`(:52) + `recall_offload`(:50) + throttle/tbt reset | `restore_defaults` + reset 묶음 | (a.6) recall :301-310; throttle/tbt reset 은 executor→LoopControl |
| 10 | `RequestQcf`(:236) | :527-530 | `request_qcf: bool`(:54) | `request_qcf` | (현 run() 미소비 — EngineReport 경로; QCF 송출은 generate 측. LoopControl tick) |
| 11 | `SetPrefillPolicy`(:260) | :536-551 | `prefill_chunk_size`(:58) + `prefill_yield_ms`(:59) + `prefill_cpu_chunk_size`(:60) | prefill 3필드 | **[정오 2026-06-11] 소비자 0** — run() 미소비 + ~~prefill 경로(batch/runner.rs:495-509)~~ 가 γ-4(`2e53cf44`)에서 batch 모듈과 함께 삭제됨. LoopControl.prefill_* 매핑은 신설하나 **현 소비자 0** (chunked prefill 의 동적 정책 변경 경로 부재). 후속 substep 에서 prefill chunked 경로 재배선 시 부활 |

**설계 결정 (② 7종)**: control 명령은 KV cache 를 prune 하지 않고 **루프의 제어 파라미터**(pacing/lifecycle/query)를 바꾼다 → Stage(phase-반응)가 아니라 driver-local 제어 상태다. `LoopControl` 신설 구조체가 이를 담고, dispatcher 가 `apply_command` 의 plan-쓰기 대신 `LoopControl` 의 필드를 갱신한다. run() 은 매 step `LoopControl` 을 읽어 sleep/break/pacing 한다 — v1 의 `plan.throttle_delay_ms`/`plan.suspended`/`plan.target_tbt_ms` 읽기와 1:1.

**예외 — SetPrefillPolicy(11)**: 이 명령의 ExecutionPlan 3필드는 **decode_loop.rs::run() 에서 소비되지 않는다**. ~~prefill chunked 경로(batch/runner.rs:495-509, `executor.poll` 직접 호출)가 유일 소비자다.~~ **[정오 2026-06-11]** 그 유일 소비자(`batch/runner.rs`)가 γ-4(`2e53cf44`)에서 batch 모듈과 함께 **순수 삭제**되어, SetPrefillPolicy → LoopControl.prefill_* 매핑의 **소비자가 현재 0** 이다. β-4 dispatcher cutover 시 LoopControl 에 prefill 3필드는 등가 보존 차원에서 두되, run() 소비부도 prefill shim 도 추가하지 않는다(읽는 쪽 부재). roadmap §β-4 item 3 의 "batch/runner.rs:495 동시 마이그레이션/shim" 항목은 batch 삭제로 **moot** — chunked prefill 경로 재배선 substep 이 도입될 때 LoopControl.prefill_* 의 소비처를 신설한다.

**예외 — Resume(8)/RequestQcf(10)**: run() 본문에 직접 소비 코드가 없다. Resume 은 executor 내부 state(`engine_state`) 전이만(executor.rs:517-526), RequestQcf 는 `plan.request_qcf` 플래그를 generate 측(또는 EngineReport)이 읽어 QCF 송출. LoopControl 에 필드는 두되, β-4 에서는 등가 seam 잔존(소비 무변)으로 처리한다.

### 1.3 과도기 5종 → LoopControl 과도기 필드 (deprecated 표기, G1)

| # | variant (정의 라인) | apply_command (라인) | ExecutionPlan 필드 | run() 소비 지점 (live 여부) | 이전 예정 |
|---|---|---|---|---|---|
| 12 | `KvQuantDynamic`(:227) | :465-475 | ~~`kv_quant_bits: Option<u8>`(:43)~~ **삭제 (AB-2)** | **① OneShot KiviQuantStage** — `submit_kv_quant(target_bits)` → `KiviQuantStage::one_shot(kivi_handles, target_bits)` → `registry.submit`; `KvMutate`(eviction 과 동일 KV phase) 에서 소비(transition_bits 적용, KV bit-width 전환). sticky = dispatcher `last_quant_bits` 비교 게이트(값 변경 시만 submit, evict_armed 복사 금지), RestoreDefaults → `last=None`(guard clear 만 — 16bit 복원 submit **없음**, partition `submit_partition_full` 과 비대칭, v1 등가) | **✅ AB-2 완료** — 자세히는 `pipeline_stage_design_v2.md §5.7`. `LoopControl.kv_quant_bits` 필드 삭제(write-only dead-end 소멸 — KvQuantDynamic 이 v2 에서 silent no-op 였던 것 해소). KV 축 Stage(`stages/kv/kivi_quant.rs`, EvictionStage 형제) — weight 축 partition/swap 과 구분. 로그 marker = `[KIVI-Resilience] Transitioned KV cache to {N}bit`(verify YAML 글자단위 계약). heartbeat kv_dtype = adapter handle query(§5.7.6) |
| 13 | `KvOffload`(:231) | :476-483 | `offload_ratio: Option<f32>`(:47) | **(a.6) :288-300 (live — `forward.try_offload`)** | AB- offload Stage |
| 14 | `SwapWeights`(:209) | :562-569 | ~~`swap_weights: Option<(f32,DtypeTag)>`(:64)~~ **삭제 (AB-6)** | **① OneShot WeightSwapStage** — `submit_swap(ratio, target_dtype)` → `WeightSwapStage::one_shot(layer_slots, swap_runtime, importance, ratio, target_dtype)` → `registry.submit`; `WeightMutate`(구 PostEviction rename) 에서 소비(Incremental multi-tick drain, D5). sticky = **transient**(last-applied/armed 게이트 불요 — in-flight 가드가 R-1 동시활성화 차단) | **✅ AB-6 완료** — 자세히는 `pipeline_stage_design_v2.md §5.6`. `LoopControl.swap_weights` 필드 삭제(write-only dead-end 소멸). 4-mode: Incremental=Stage drain / IntraForward·LayerImmediate·PhaseAware=layer-tier hook 설치(§5.6.3). manager report = `WeightSwapReport`(executor.rs:248 EngineMessage) |
| 15 | `SetPartitionRatio`(:254) | :531-535 | ~~`partition_ratio: Option<f32>`(:56)~~ **삭제 (AB-4)** | **① OneShot PartitionStage** — `submit_partition(ratio)` → `PartitionStage::one_shot(layer_slots, hardware, ratio)` → `registry.submit`; `PreForward` 에서 소비(weight slot re-slice, INV-120) | **✅ AB-4 완료** — 자세히는 `pipeline_stage_design_v2.md §5.5`. sticky = dispatcher `last_partition_ratio` 비교 게이트(값 변경 시만 submit, evict_armed 복사 금지), RestoreDefaults → `last=None` + Full 복원 OneShot |
| 16 | `LayerSkip`(:199) | :390-396 | `layer_skip: Option<f32>`(:41) | run() 인라인 소비 없음 — forward skip_config 측 (sticky) | AB- layer-skip Stage |

**설계 결정 (과도기 5종, [G1])**: 이 5종은 각자 대응 Stage(KIVI/offload/weight-swap/partition/layer-skip)가 아직 미구현이다. β 가 owner 이나 AB-2/4/6 트랙은 β-4 확정 후 후속 substep 에서 Stage 모델로 구현한다 — **미실재 stage 선축조 금지**(roadmap §β-4 item 2). 따라서 β-4 에서는 LoopControl 에 **deprecated 과도기 필드**로 잔존시키고, 구==신 등가를 보장한다. 특히 **KvOffload(13)/SwapWeights(14) 는 live 소비 경로**(decode_loop (a.6) / take_pending_swap_weights)를 가지므로, β-4 게이트가 이 2종의 구==신 등가를 명시 검증한다(roadmap §β-4 게이트 "live 소비 경로 보유 SwapWeights(:209)/KvQuantDynamic(:227)/KvOffload(:231) 포함").

**[정오 2026-06-11 — AB-2/4/6 완료]**: 위 5종 중 **partition(15)은 AB-4 에서, swap(14)은 AB-6 에서, quant(12)는 AB-2 에서 OneShot Stage 로 이전**됐다(`LoopControl.partition_ratio`·`swap_weights`·`kv_quant_bits` 필드 3종 삭제). 따라서 과도기 잔존은 **KvOffload(13)/LayerSkip(16) 2종**으로 줄었다(AB-/offload·layer-skip Stage 대기). SwapWeights 의 v1 `take_pending_swap_weights`(executor.rs:258) live 경로는 `WeightSwapStage` commit 경로(`handle_swap_weights` §1~7 이전, §5.6.2)로 수렴. **KvQuantDynamic(12)은 v2 에서 live 소비 경로가 0**(`control.kv_quant_bits` set 후 decode_loop 미배선 = silent no-op)이었으므로 β-4 게이트의 "live 소비 등가" 대상이 아니었다 — AB-2 가 v1 등가(`generate.rs`(d5ed71d2^) L4392-4407 transition_bits loop)를 KiviQuantStage commit 으로 *신규 배선*(no-op 해소)하고, host 게이트(dispatcher sticky 단위테스트 + transition_bits 등가)로 검증한다(§5.7).

**[정오 — roadmap 라인]**: roadmap §β-4 게이트 본문은 `SwapWeights(:246)`/`KvQuantDynamic(:465)`/`KvOffload(:476)` 로 표기한다. 이는 **혼합 참조** — `:246` 은 SwapWeights 의 *variant 정의가 아니라 Suspend 정의 라인*이고(SwapWeights variant 정의는 :209), `:465`/`:476` 은 *variant 정의가 아니라 apply_command 핸들러 라인*이다(KvQuantDynamic variant 정의 :227 / apply :465; KvOffload variant 정의 :231 / apply :476). 본 표는 variant 정의 라인(shared/src/lib.rs)과 apply_command 라인(executor.rs)을 분리 표기한다.

### 1.4 ③ Hardware resolve seam 2종

| # | variant (정의 라인) | apply_command (라인) | ExecutionPlan 필드 | 신 채널 | run() 소비 지점 |
|---|---|---|---|---|---|
| 17 | `SwitchHw`(:242) | :503-507 | `switch_device: Option<String>`(:23) | Hardware resolve seam (`Arc<Hardware>.resolve(target)`, §3.5) | run() 인라인 소비 없음 — generate 측 device switch (seam 만) |
| 18 | `PrepareComputeUnit`(:244) | :508-511 | `prepare_device: Option<String>`(:25) | Hardware resolve seam | 동상 |

**설계 결정 (③ 2종)**: device switch/prewarm 은 scalar 환원 불가한 mode 명령(§5.4 이산 채널)이며 KV/loop 상태가 아니라 hardware 토폴로지를 만진다. v2 의 hardware 축은 `Arc<Hardware>` register-시점 보유 + `resolve(target)` read-only(§5.4, §3.5). β-4 에서는 **seam 만** 둔다 — 실제 switch 구현은 별도. run() 본문에 인라인 소비 없음(현재도 없음 — `plan.switch_device`/`plan.prepare_device` 는 generate 측이 읽음).

### 1.5 역방향 검증표 (ExecutionPlan 19필드 → variant, 누락 0)

apply_command 가 쓰는 19필드 각각이 적어도 1개 variant 에서 set 됨을 역으로 확인한다 (1부 표에서 빠진 필드 0).

| # | ExecutionPlan 필드 (정의 라인) | set 하는 variant(들) | 신 채널 |
|---|---|---|---|
| 1 | `evict`(:21) | KvEvictH2o/KvEvictSliding/KvStreaming/KvMergeD2o + sticky carry(:311,:338) | ① OneShot EvictionStage |
| 2 | `switch_device`(:23) | SwitchHw(:505) | ③ Hardware seam |
| 3 | `prepare_device`(:25) | PrepareComputeUnit(:509) | ③ Hardware seam |
| 4 | `throttle_delay_ms`(:27) | Throttle(:363) / Resume(:524 reset) / RestoreDefaults(:487 reset) / sticky(:308) | ② LoopControl |
| 5 | `target_tbt_ms`(:30) | SetTargetTbt(:374) / RestoreDefaults(:488 reset) / sticky(:309) | ② LoopControl |
| 6 | `target_tbt_set`(:35) | SetTargetTbt(:375 true) / RestoreDefaults(:489 주석 — false 유지) / sticky(:310) | ② LoopControl |
| 7 | `suspended`(:37) | Suspend(:513) | ② LoopControl (break) |
| 8 | `resumed`(:39) | Resume(:518) | ② LoopControl |
| 9 | `layer_skip`(:41) | LayerSkip(:391) | 과도기 (LoopControl deprecated) |
| 10 | ~~`kv_quant_bits`(:43)~~ **삭제 (AB-2)** | (구) KvQuantDynamic(:467) / sticky(:312,:341) → 신: dispatcher `submit_kv_quant` + `last_quant_bits` 게이트 | **① OneShot KiviQuantStage** (LoopControl 필드 제거 — write-only dead-end 소멸; §5.7.3) |
| 11 | `offload_ratio`(:47) | KvOffload(:478) | 과도기 (LoopControl deprecated, live (a.6)) |
| 12 | `recall_offload`(:50) | RestoreDefaults(:486) | ② LoopControl (RestoreDefaults 묶음, live (a.6)) |
| 13 | `restore_defaults`(:52) | RestoreDefaults(:485) | ② LoopControl |
| 14 | `request_qcf`(:54) | RequestQcf(:528) | ② LoopControl |
| 15 | ~~`partition_ratio`(:56)~~ **삭제 (AB-4)** | (구) SetPartitionRatio(:533) / sticky(:315) → 신: dispatcher `submit_partition` + `last_partition_ratio` 게이트 | **① OneShot PartitionStage** (LoopControl 필드 제거 — write-only dead-end 소멸; §5.5.2) |
| 16 | `prefill_chunk_size`(:58) | SetPrefillPolicy(:542) | ② LoopControl (prefill 경로) |
| 17 | `prefill_yield_ms`(:59) | SetPrefillPolicy(:545) | ② LoopControl (prefill 경로) |
| 18 | `prefill_cpu_chunk_size`(:60) | SetPrefillPolicy(:548) | ② LoopControl (prefill 경로) |
| 19 | ~~`swap_weights`(:64)~~ **삭제 (AB-6)** | (구) SwapWeights(:567) → 신: dispatcher `submit_swap` + in-flight 가드(transient) | **① OneShot WeightSwapStage** (LoopControl 필드 제거 — write-only dead-end 소멸; §5.6.4) |

**검증 결론**: 19필드 중 `partition_ratio`(#15, AB-4 삭제)·`swap_weights`(#19, AB-6 삭제)·`kv_quant_bits`(#10, AB-2 삭제) 3필드가 OneShot Stage(PartitionStage/WeightSwapStage/KiviQuantStage)로 이전돼 LoopControl 에서 제거됐다. 잔존 16필드는 전부 1개 이상 variant 에 매핑됨. 누락 0. (sticky carry-only 필드도 set source variant 가 존재한다 — sticky 는 같은 variant 의 재현일 뿐.)

---

## 2부. sticky 3-상태 전이 등가표

v1 의 "한 번 받은 directive 가 다음 step 까지 유지되는" sticky 시맨틱은 3곳에 분산되어 있다. 각각을 신 OneShot 모델 등가로 사상한다.

### 2.1 v1 sticky 3-상태

#### 상태 A — `evict_applied`(driver 1회-게이트)

| 항목 | 위치 | roadmap 기재 → 실측 |
|---|---|---|
| 필드 | decode_loop.rs:53 | (roadmap :48 → 실측 :53, drift +5) |
| set (true) | decode_loop.rs:256-257 (`if !self.evict_applied { self.evict_applied = true;`) | (roadmap :169 → 실측 :256, drift +87) |
| reset (false) | decode_loop.rs:280 (`} else { self.evict_applied = false; }`) | — |

**전이 조건**:
- **set**: `plan.evict.is_some() && !evict_applied` → `evict_applied=true` + `forward.try_evict` 1회 발동 (decode_loop.rs:255-278).
- **reset**: `plan.evict.is_none()`(RestoreDefaults 로 executor 의 evict_plan 이 None 으로 복귀하여 carry 가 끊긴 step) → `evict_applied=false` (decode_loop.rs:279-281).
- **재적용 방지**: sticky `evict_plan` 이 매 poll 같은 EvictPlan 을 carry 해도, `evict_applied==true` 인 동안 try_evict 재발동 안 함(cache 과다 축소 방지 — decode_loop.rs:253 주석).

#### 상태 B — `evict_plan` sticky carry(executor 보유)

| 항목 | 위치 | roadmap 기재 → 실측 |
|---|---|---|
| 필드 | executor.rs:123 (`evict_plan: Option<EvictPlan>`) | — |
| carry (directives empty) | executor.rs:311 (`plan.evict = self.evict_plan.clone();`) | (roadmap "carry executor.rs:306-317" → 실측 :306-317 분기 안의 :311) |
| carry-forward (non-empty, 미override) | executor.rs:337-339 (`if plan.evict.is_none() { plan.evict = self.evict_plan.clone(); }`) | roadmap 미세분 — 실 carry 는 2곳 |
| set | apply_command KvEvict* :404/:418/:441/:458 (`self.evict_plan = Some(...)`) | — |

**전이 조건**:
- **set**: 어떤 KvEvict* directive 도착 → `self.evict_plan = Some(evict)` (새 method 가 기존 override — executor.rs test :1480-1507).
- **carry**: directive 없는 poll → `plan.evict = self.evict_plan.clone()` (sticky 유지).
- **reset**: RestoreDefaults → `self.evict_plan = None` (executor.rs:495).

#### 상태 C — RestoreDefaults reset(executor)

| 항목 | 위치 | roadmap 기재 → 실측 |
|---|---|---|
| reset 블록 | executor.rs:484-502 | (roadmap :484-502 → 실측 :484-502, 일치) |

reset 대상(:485-500): `restore_defaults=true`(:485), `recall_offload=true`(:486), throttle/tbt 0(:487-488,:492-494), `evict_plan=None`(:495), `kv_quant_bits=None`(:496), `partition_ratio_sticky=None`(:497), level Normal(:498-499), `active_actions.clear()`(:500).

### 2.2 신 OneShot 모델 등가

| v1 상태 | 신 모델 등가 | 사상 근거 |
|---|---|---|
| **A. evict_applied=true (1회 발동)** | `EvictionStage::one_shot` 제출 → `on_phase` 1회 발화 → `StageOutcome::Consumed` → registry GC (eviction.rs:99-103, registry dispatch GC) | **제출 1회 = 발화 1회 = GC** 가 v1 의 "evict_applied 게이트로 active 구간당 1회" 와 등가. v1 의 sticky carry 가 매 step 같은 plan 을 줘도 1회만 발동하던 것 = OneShot 이 Consumed 후 registry 에서 사라져 재발화 불가 |
| **B. evict_plan sticky carry** | dispatcher 가 KvEvict* directive 수신 시 `registry.submit(one_shot(...))` 1회. carry(다음 step 재제출) **없음** — 신 모델은 sticky 를 폐기하고 directive 도착=제출 1회로 단순화 | v1 sticky 의 목적은 "directive 가 한 번 와도 evict_applied 게이트로 1회 발동 보장" 이었다. OneShot Consumed GC 가 동일 보장을 제공하므로, executor 의 evict_plan sticky carry 는 신 모델에서 **불필요**(dispatcher 가 매 poll drain 한 directive 만 submit). 회귀 위험 = directive 가 prefill 중 도착해 decode 진입 전 1회 drain 되는 타이밍 — frozen_baseline(directive prefill 중 도착 → decode step 0 적용)이 이 타이밍을 고정 검증 |
| **C. RestoreDefaults reset** | RestoreDefaults 도착 → dispatcher 가 (i) LoopControl reset(throttle/tbt/restore_defaults/recall_offload) + (ii) **재제출 가능 상태 복귀** (다음 KvEvict* directive 가 새 OneShot 제출 가능) | v1 의 `evict_plan=None`(B reset) + `evict_applied=false`(A reset, decode_loop.rs:279-281 의 `plan.evict.is_none()` 분기) 의 합 = 신 모델의 "RestoreDefaults 도착 시 OneShot 재제출 가능". OneShot 은 이미 Consumed/GC 되었으므로 별도 clear 불요 — 다음 directive 가 새 stage 를 submit 하면 됨 |

**핵심 등가 명제** (β-4 host 게이트 검증 대상):
```text
v1:  [KvEvict* 도착] → evict_plan=Some → (carry) → [step] evict_applied?false→try_evict 1회·true
                                                  → [step+1] evict_applied?true → skip (sticky 재적용 방지)
     [RestoreDefaults] → evict_plan=None → [step] plan.evict?None → evict_applied=false (재발동 준비)

v2:  [KvEvict* 도착] → dispatcher submit one_shot → [PreEviction] on_phase 1회·Consumed → GC
                                                  → [step+1] registry len==0 → 무발화 (재적용 자동 방지)
     [RestoreDefaults] → dispatcher LoopControl reset → (다음 KvEvict* 가 새 one_shot submit 가능)
```
양쪽 모두: **active 구간당 정확히 1회 prune + RestoreDefaults 후 재무장**. 행별 등가 test 가 이 시퀀스를 mock_manager directive 로 재현한다.

---

## 3부. (a.5) method-drop 시맨틱

### 3.1 v1 거동 — directive method 무시

v1 (a.5)(decode_loop.rs:255-278)는 `plan.evict`(EvictPlan)의 `method` 필드를 **읽지 않는다**. 발동 코드는 오직 `evict_plan.target_ratio` 만 사용한다:

```rust
// decode_loop.rs:259-261
let (removed, new_pos) =
    self.forward
        .try_evict(cm, None, true, evict_plan.target_ratio)?;
```

`try_evict(cm, scores=None, force=true, target_ratio)`(model_forward.rs:505-548)는 `force=true, scores=None` 경로로 `cache_manager.force_evict(&mut temp, target_ratio)`(model_forward.rs:530)만 호출한다. **어느 eviction 알고리즘을 쓸지는 directive 의 `EvictMethod` 가 아니라 CacheManager 가 보유한 정책**(CLI `eviction <policy>` 로 `build_resilience_cache_manager`(build_bench_loop.rs:46-138)가 구성)이 결정한다.

즉 `EvictPlan.method`(executor.rs:82, apply_command 이 KvEvictH2o→H2o / KvEvictSliding→Sliding 등으로 채움)는 (a.5) 소비 시점에 **드롭**된다 — directive 가 H2O 를 요청해도 CLI 가 sliding 으로 구성했으면 sliding 으로 evict 된다.

**코드 증거**: frozen_baseline_beta_evict_2026_06.md 의 시나리오 — `KvEvictSliding{keep_ratio:0.5}` directive 가 CLI `eviction sliding` CacheManager 로 흘러 `[CacheEvent] policy='sliding@Warning'` 를 emit. directive 의 method(Sliding)가 우연히 CLI 정책(sliding)과 일치하지만, 일치를 강제하는 코드는 없다 — method 가 CLI 정책으로 override 된다.

### 3.2 신 모델 — 동일 시맨틱 보존

OneShot `EvictionStage`(eviction.rs:74-103)도 **동일하게 method 를 사용하지 않는다**:
- `EvictionStage::one_shot(handles, cache_manager, target_ratio)`(eviction.rs:51-62)는 `target_ratio` 만 받는다 — method 인자 없음.
- `on_phase`(eviction.rs:74-103)는 보유한 `cache_manager`(CLI 정책)로 `force_evict(&mut temp, self.target_ratio)`(eviction.rs:88-92) 호출 — UER 가 v1 try_evict 의 inner op 와 byte-identical(eviction.rs:8 주석).

**dispatcher 설계 (3부 결론)**: β-4 CommandDispatcher 가 KvEvict* 4종을 OneShot 으로 사상할 때, directive 의 `keep_ratio`(KvStreaming 은 0.0)만 `target_ratio` 로 전달하고 **method 는 무시**한다. EvictionStage 가 보유한 cm(= build_bench_loop 가 CLI 로 구성)이 정책을 결정한다. 이는 **의도적 시맨틱 보존** — v1 의 method-drop 이 버그가 아니라 "manager 는 강도(keep_ratio)를 지시하고, 알고리즘은 엔진 CLI 구성이 고정한다"는 책임 분리의 코드화다.

**β-4 게이트 함의**: method-drop 때문에, 행별 등가 test 에서 KvEvictH2o directive 를 보내도 CLI 가 sliding 이면 sliding 산출이 나온다. test 는 이를 정상으로 검증해야 한다(directive method ↔ 산출 정책 일치를 가정하지 말 것). frozen_baseline 의 `policy='sliding@Warning'` 가 이 게이트의 anchor.

---

## 4부. heartbeat 송출 경로 보존 설계

### 4.1 현 유일 트리거 (단절 위험)

heartbeat 의 현 송출 경로:

```text
DecodeLoop::run() (a) command poll
  → cmd_source.poll(&ctx, &kv_snap)                      [decode_loop.rs:239]
    → CmdSrcWrapper::poll(ctx, kv_snap)                  [resilience_adapter.rs:72-79]
      → ResilienceAdapter::poll(ctx, kv_snap)            [resilience_adapter.rs:37-41]
        → executor.poll(kv_snap)                         [resilience_adapter.rs:39]
          → if last_heartbeat.elapsed() >= interval {    [executor.rs:293]
                send_heartbeat(kv_snap);                 [executor.rs:294 — 유일 송출]
            }
```

**단절 위험**: roadmap §β-4 item 1 은 `CommandSource::poll` 을 `poll() -> Vec<EngineCommand>`(pure)로 retarget 한다. pure 화하면:
1. `kv_snap` 인자가 사라진다 — heartbeat payload(KVSnapshot)의 운반 경로 소멸.
2. heartbeat 송출(`send_heartbeat`)이 poll 내부에서 일어나는데, poll 이 순수 command drain 만 하면 이 부수효과가 사라진다.

`EngineReport`(traits.rs:178-182)는 `send_capability`/`send_qcf_estimate`/`send_swap_report` 만 가지며 **heartbeat 메서드가 없다**(roadmap §β-4 item 4 확인). 또한 `DecodeLoop.report`(decode_loop.rs:43-44)는 `#[allow(dead_code)]` 로 현재 미사용.

### 4.2 후보 분석

| 후보 | 설명 | 침습도 | 평가 |
|---|---|---|---|
| **(가) ManagerCommandSource drain 시 송출 유지** | pure `poll()->Vec<EngineCommand>` 가 drain 직전 내부에서 heartbeat 를 계속 송출. kv_snap 은 poll 인자 대신 `ManagerCommandSource` 가 보유한 held-handle(KVCacheFormat)에서 query | **최소** — poll 시그니처는 pure(Vec 반환)이되, heartbeat 부수효과는 source 내부에 잔존. kv_snap 운반 경로만 인자→held-handle query 로 교체 | **채택 권고** |
| (나) EngineReport 채널로 이관 | `EngineReport` 에 `send_heartbeat(status)` 추가, driver 가 매 step interval 체크 후 report.send_heartbeat 호출 | 중 — trait 확장 + driver 에 heartbeat 타이머 로직 신설 + kv_snap 구성을 driver 가 담당 | trait 확장·driver god-loop 화. β 범위 초과 |
| (다) LoopControl 틱 | LoopControl 이 heartbeat interval 을 관리하고 driver 가 틱마다 송출 | 중-고 — LoopControl 책임 비대(제어+보고 혼재, SRP 위반) | 비권장 |

### 4.3 채택안 — (가) ManagerCommandSource 내부 송출 유지 + kv_snap held-handle query

**설계 결정**: heartbeat 송출은 **`ManagerCommandSource`(구 ResilienceAdapter/CommandExecutor) 내부에 잔존**시킨다. poll 을 pure(`poll()->Vec<EngineCommand>`)화하되, heartbeat 부수효과(interval 체크 + send_heartbeat)는 source 가 drain 직전에 자체 수행한다. 변경되는 것은 **kv_snap 의 운반 경로**뿐이다:

- **현재**: `kv_snap` 이 poll 인자로 driver→adapter→executor 로 흐름(decode_loop.rs:238 `build_kv_snapshot()` → :239 poll 인자).
- **신**: `ManagerCommandSource` 가 KV 상태를 **held-handle**(register 시점 보유한 `Arc<dyn KVCacheFormat>` — §5.2.1 (가)의 kv_pos_handle 과 동일 패턴)에서 query 하여 KVSnapshot 을 자체 구성. 이는 SSOT §5.2.1 의 "ctx 로 흘리지 말고 held-handle query"(god-ctx 회피, INV-STAGE-LAYER-HANDLE) 정신과 정합.

**정당화 (왜 가장 침습이 적은가)**:
1. **trait 표면 무확장** — EngineReport 에 메서드 추가 없음. CommandSource 만 pure 화(이미 roadmap 계획).
2. **driver 무변** — run() 에 heartbeat 타이머/송출 로직 신설 없음. driver 는 여전히 `cmd_source.poll()` 만 호출(반환 타입만 Vec).
3. **송출 코드 이동 0** — `send_heartbeat`(executor.rs:573-643) + interval 체크(executor.rs:293-296)는 그대로 ManagerCommandSource 내부에 잔존. 옮기는 것은 kv_snap 의 **출처**(poll 인자 → held-handle query)뿐.
4. **NoOpCommandSource(defaults.rs:39-43) 무영향** — heartbeat 를 송출하지 않는 source 는 pure poll 만 반환(현재도 ExecutionPlan::default 반환 = 무동작).

**held-handle 주입**: build_bench_loop(build_bench_loop.rs:185-188)가 이미 layer-0 `Arc<dyn KVCacheFormat>`(kv_pos_handle)를 만든다. ManagerCommandSource 도 동일 handle 을 register 시점 받아 `current_pos`/capacity 등을 query 해 KVSnapshot 구성. (total_bytes 등 일부 필드는 현재도 placeholder — decode_loop.rs:193 `total_bytes: 0`, build_kv_snapshot 도 partial. 등가 보존에 영향 없음.)

### 4.4 host 게이트 — heartbeat 연속성 test (검증 가능 형태)

β-4 host 게이트의 "heartbeat 연속성 test(actual_throughput 송출 등가)" 명세:

```text
GIVEN  mock_manager 가 짧은 heartbeat_interval(예: 10ms)로 ManagerCommandSource 구성
       + held-handle(KVCacheFormat) 주입 + on_token_generated 로 throughput EMA 적재
WHEN   decode N step 진행 (각 step poll() pure 호출)
THEN   (1) heartbeat(EngineMessage::Heartbeat) 가 interval 마다 ≥1회 송출됨 (resp_rx 수신)
       (2) 송출된 EngineStatus.actual_throughput 가 0 이 아님 (EMA 적재 확인)
       (3) EngineStatus.kv_cache_tokens == held-handle.current_pos() (kv_snap held-handle query 정합)
       (4) v1(executor.poll(kv_snap)) 대비 송출 횟수·payload 동치
```

**등가 anchor**: v1 의 `test_executor_heartbeat`(executor.rs:1061-1096) + `test_executor_heartbeat_active_actions`(executor.rs:1098-1136). 신 ManagerCommandSource 가 동일 입력(token tick + interval 경과)에서 동일 heartbeat payload(active_device/kv_cache_tokens/state/actual_throughput)를 송출함을 검증. (3)의 kv_cache_tokens 정합이 held-handle query 전환의 핵심 회귀 가드.

---

## 5. β-4 host 게이트 명세 요약 (이 문서가 정의)

이 문서를 명세로 하는 host 게이트(roadmap §β-4 게이트):

1. **매핑표 행별 등가 test** (1부) — mock_manager directive 시퀀스로 18-variant 중 live 소비 경로 보유분(suspend/throttle/tbt/evict 4종 + SwapWeights/KvQuantDynamic/KvOffload)의 구==신 산출 동치.
2. **dispatcher EngineCommand exhaustive match** — 18-variant 컴파일 강제(누락 시 컴파일 실패).
3. **sticky 전이 unit** (2부) — A/B/C 3-상태 → OneShot Consumed GC + RestoreDefaults 재무장 등가.
4. **method-drop test** (3부) — KvEvict* directive ↔ CLI 정책 산출(directive method 무시) 검증.
5. **heartbeat 연속성 test** (4부) — pure poll 전환 후 heartbeat 송출·payload 연속성.

**Implementer 테스트 작성 요청** (오케스트레이터 경유): 위 5종 host test + `tests/spec/` 의 INV-DECODE-STAGE 계열에 method-drop·OneShot Consumed 등가 항목 추가.

## 참조

- SSOT: `arch/pipeline_stage_design_v2.md` §5.4(:653-666+ 2-source; 핵심 :655-660) · §5.2.1(:611-641 driver↔Stage 계약)
- roadmap: `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md` §β-4(:83-94)
- frozen baseline: `.agent/todos/frozen_baseline_beta_evict_2026_06.md` (command-driven evict anchor)
- 코드: `shared/src/lib.rs`(EngineCommand) · `engine/src/resilience/executor.rs`(apply_command/ExecutionPlan/heartbeat) · `engine/src/session/decode_loop.rs`(run/run_until_stop 소비) · `engine/src/session/resilience_adapter.rs`(heartbeat conduit) · `engine/src/stages/kv/eviction.rs`(OneShot EvictionStage) · `engine/src/session/forward/model_forward.rs`(try_evict method-drop)
