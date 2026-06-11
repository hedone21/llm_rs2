# Handoff: argus-bench AB-0/AB-1/AB-3 ✅ (host) → AB-2/AB-4/AB-6 + AB-5 device phase

**작성**: 2026-06-05 (메인 세션)
**HEAD**: `95908f57` feat(engine): argus-bench AB-3 — resilience KvOffload / recall 재배선 — **3 커밋 미푸시**(origin/master=`e23518e3`)
**브랜치**: `master`
**다음 세션 진입 문장**: **"argus-bench AB-2 (KIVI quant) 진행"** 또는 **"argus-bench AB-5 verify 재배선 + S25 device 게이트"**

---

## TL;DR
argus-bench 트랙(legacy decode_fallback 폐기 모드를 fmt 경로 재구현 + verify 재배선) 중 **host-검증 가능한 3개 완료**: AB-0(experiment-output bin + throttle/suspend/target_tbt) / AB-1(mid-decode eviction) / AB-3(KvOffload/recall). **host verify 6 시나리오 green**. 남은 AB-2(KIVI quant)/AB-4(partition)/AB-6(weight-swap)은 **verify 시나리오가 전부 device 전용**(galaxy_s25/jetson)이라 host verify-green 불가 — 구현 + device 검증(AB-5 verify 재배선 + S25 실행)이 한 phase. **왜 멈췄나**: 사용자 결정 "엔진 구현만(host build/smoke, device 검증 보류)"에 따라 host-검증분 완료. AB-2/4/6은 device-only-verify + 높은 복잡도(blind 구현 위험)라 device phase 로 분리.

---

## 완료 (커밋, host-검증)
| 단계 | commit | 동작 | host 게이트 |
|---|---|---|---|
| AB-0 | `f18d0b6e` | argus_bench bin + experiment-output JSONL + `[Experiment] Done` + target_tbt pacing(DecodeLoop) + SetTargetTbt 로그(executor) + Suspend 로그 + verify host binary 재배선(ENGINE_BIN) | f16 4/4: throttle_smoke·target_tbt·target_tbt_restore·thermal_emergency_suspend |
| AB-1 | `94ef643f` | resilience KvEvict* mid-decode eviction (DecodeLoop cache_manager+evict_applied, forward.try_evict, build_bench_loop, build_resilience_cache_manager, on_kv_prune gpu_plan invalidate, verify RUST_LOG=info) | f16+q4: memory_critical_evict·prefill_midway_injection |
| AB-3 | `95908f57` | resilience KvOffload/recall (Forward try_offload/try_recall UER, DecodeLoop 소비, build_resilience_cache_manager `--swap-dir`→enable_swap) | host smoke: kvoffload(1-cmd) PASS, offload 로그+disk 검증 |

**핵심 아키텍처(재사용)**: `DecodeLoop`(decode_loop.rs)가 매 step `cmd_source.poll()→plan` 후 plan.{evict,offload_ratio,recall_offload,target_tbt_ms,throttle,suspended}을 소비. `forward.try_evict/try_offload/try_recall`(ModelForward fmt UER: take_inner→cm.op→put_inner). `build_bench_loop`(assembly/build_bench_loop.rs) + `build_resilience_cache_manager`가 CLI→CacheManager. argus_bench `bench_supported` 가드(is_standard_happy_path 에서 eviction 만 허용).

---

## 남은 작업 (device phase — verify 시나리오 전부 device 전용)
**디바이스**: S25(R3CY408S5SB) adb 연결됨 + 모델(f16/q4 gguf) 존재 + hosts.toml android-ndk + devices.toml galaxy_s25(aarch64-linux-android/adreno). Jetson ssh 오프라인.

### AB-2 KIVI dynamic quant (kvquant_to_q4, kvquant_restore — galaxy_s25/jetson)
- **로그(정확)**: `[KIVI-Resilience] Transitioned KV cache to {bits}bit`(legacy generate.rs:4405), `[KIVI-Resilience] RestoreDefaults`. heartbeat kv_dtype→q4.
- **plan**: `plan.kv_quant_bits: Option<u8>`(executor 완비, sticky). DecodeLoop 미소비.
- **구현**: (1) argus_bench 가드 `--kv-dynamic-quant` 허용 + KIVI 라우팅. (2) build_bench_loop KIVI 분기 — `alloc_kivi_kv_caches`(kivi_forward.rs:321; CPU=`kivi:None`, OpenCL=`caps.get::<dyn KiviAttentionBackend>()` from `SessionInitCtx.caps`)+`KiviForward::new`(initial bits=16). StandardHappyCtx.kv_caches(Vec<KVCache>)는 drop되므로 KIVI 분기가 자체 alloc — caps+residual_size 를 ctx→run_experiment_path→build 로 thread 필요. (3) `Forward::try_quant(bits)` default no-op + KiviForward override(self.kv_caches iter `cache.transition_bits(bits)`(kivi_cache.rs:1034)+로그). (4) DecodeLoop plan.kv_quant_bits 소비 + `kivi_last_quant_bits` sticky 가드(bits 변경 시만 transition) + RestoreDefaults reset. (5) build_kv_snapshot kv_dtype(KIVI bits 기반).
- **host smoke**: CPU `--kv-dynamic-quant` + mock KvQuantDynamic → transition_bits CPU + `[KIVI-Resilience] Transitioned` 로그. GPU realloc(F16→Q4 GPU)+heartbeat kv_dtype 는 device.
- **landmine**: build_inference_ctx 가 caps discard(현재) → OpenCL KIVI dispatch 실패 위험, caps thread 필수. restore `pass_criteria: all`(accuracy min_rouge_l 0.30) — F16→Q4 fidelity 필요. Forward trait try_quant 는 all-impl 변경(default no-op 비파괴).

### AB-4 tensor partition 동적 enable (partition_ratio×2, prefill_midway_partition — galaxy_s25/jetson, **GPU 전용**)
- **로그(정확, legacy 글자단위 보존 필수)**: `[Partition] Re-split {n} weights with ratio {:.2}`(generate.rs:2807), `[Partition] Lazy-mapped {n} weight tensors...`(2789), `[Partition] Disabled (ratio=0)`(2743, f32 0.0 Display=`0`), `[Partition] RestoreDefaults: re-split...`(3132).
- **plan**: `plan.partition_ratio: Option<f32>`(executor sticky 완비). DecodeLoop 미소비.
- **구현**: 가드 완화(bench_supported tensor_partition==0 제거+reject 제거) / DecodeLoop `last_applied_partition_ratio` 가드 + `Forward::try_set_partition(ratio,&Hardware)` / ModelForward `prepare_tensor_partition`+`map_weights_for_cpu`+decode_workspace.partition_ws 부착 + gpu_plan invalidate / build_bench_loop+ModelForward::new 에 `hardware:Arc<Hardware>` 인자 추가. 참조 구현: batch/runner.rs:172-211.
- **landmine(중대)**: `prepare_tensor_partition(&mut self)` vs ModelForward 의 `Arc<TransformerModel>`(get_mut 불가) — LayerSlot ArcSwap 으로 &self 토글 가능한지 확인 필요, 불가 시 비국소 변경. build_standard_loop/build_bench_loop 시그니처에 hardware 추가(non-local). baseline `--tensor-partition 0.3` 가 ModelForward 에 partition_ws 부착 안 됨 → prologue 블록 F(lazy 1회 부착) 필수.
- **host smoke**: **불가**(GPU 부재, CPU=silent no-op). build/clippy + CLI parse + executor sticky 단위테스트까지만. 실제 partition decode·로그·정확성 = device 필수.

### AB-6 weight-swap 8종 SwapStage glue (verify 직접 시나리오 없음, 사용자 명시 요구)
- legacy dispatch_force_swap!(generate.rs:1393) 4-way(single-shot/incremental/intra-forward/phase-aware) + swap_dispatch(run_incremental_dispatch/retire_*) fmt 재배선. 생존 인프라: IncrementalSwapPlan/IntraForwardSwapHook/PhaseAwareSwapDispatcher/run_layer_swap(qcf_runtime.rs:52, cpu fallback)/EngineSwapRuntime(swap_runtime.rs:131).
- **SwapStage 인터페이스 판정(핵심)**: 현 `before_step/after_step(&StepCtx)` 로는 plan.swap_weights + model layer/secondary Arc 접근 불가. **trait 시그니처 변경 필요**(사용자 수용): (i) StepCtx/SwapStage 에 swap directive 노출, (ii) ModelForward 가 IntraForwardSwapHook 을 forward_into `layer_boundary_hook` 으로 주입(Forward 협조 — 가장 비국소). report 슬롯(decode_loop.rs:39 dead_code) drain 배선.
- **host smoke**: single-shot/incremental CPU swap(run_layer_swap cpu fallback)으로 `weight_swap: ...` 로그+LayerSlot dtype Q4_0 전이 검증. intra-forward/phase-aware(GPU materialise)+post-swap remap(Arc refcount≥2 → get_mut 불가)은 device.

### ~~AB-5 verify.py 원격 재배선 + 16/16 device 게이트~~ ✅ **종결 (2026-06-11, S25)**
- ~~PARKED 해제~~ ✅ `171fe98f`(이력 복원 + argus_bench 치환, 원격 3경로 + pkill + RUST_LOG 주입. `_run_scenario_adb` "부재" 기록은 stale — 기존재).
- **최종 게이트: 15 시나리오 × f16,q4 = 30 run → 28 PASS + 2 known-fail.** AB-2/4 신규 capability 시나리오(kvquant 2종·partition 3종) 전부 PASS.
- **수정 체인** (1차 매트릭스 16/30 → triage 엔진 회귀 0 → 점진 해소):
  - `f1dd7d0b` argv 서브커맨드 순서(`--threads`가 `eviction` 뒤 → clap 거부, 6건) + target_tbt YAML v2 marker 정합(4건) + restore delay race(2건).
  - `bf6230e8`+`226d154b` **QCF estimate 역방향 IPC 재배선**(arch v2 §5.8 — dispatcher 직결 `compute_and_send_qcf` + report_tx, `LoopControl.request_qcf` 삭제, v1 lift-and-shift→`session/qcf_runtime.rs`) — β-7 제거 후 유일 미재배선 잔여였음.
  - `e267bd50` device-only KV read-back fallback(UnifiedBuffer unmapped `as_ptr()=null` → `VDataSource::from_buffer(.., Some(cpu_bytes))` seam) + thermal decode_tokens 256.
  - `e55835ae` signal 주입 anchor 고정 sleep(10s)→event-driven(Capability marker 대기) + thermal `functional_only`(policy의 SwitchHw{cpu}=의도적 발산, kvquant 선례).
  - `1aee5497` **signal root cause**: remote_run_dir에 model key 부재 → 같은 pid의 2번째 model이 1번째의 stale `engine.rc`를 0.0s에 읽고 조기 pull(stderr가 prefill에서 절단돼 functional 오판) — 계측(`[signal] engine.rc=N after Xs`)으로 실증 후 spawn 전 rm -f.
- **known-fail 2건** = `signal_memory_critical` f16/q4: 엔진 경로 정상 입증(RequestQcf→QcfEstimate(1 action)→policy LayerSkip 선택) — h2o/d2o estimate에 필요한 **AttentionScoreAccumulator가 argus_bench 미장착**(본 문서 위 AB-1 landmine 그대로)이라 policy가 KvEvict 대신 LayerSkip. backlog `[P2] argus_bench AttentionScoreAccumulator 배선` 등록 — 그 항목 해소 시 본 시나리오가 회귀 게이트.

---

## Landmines / 미해결 (R6)
- **AB-1 eviction pos sync 잠재 이슈**: try_evict 후 `self.pos = new_pos`(physical) — RoPE-after-prune 의미가 functional_only 로 가려짐(rouge 0.082 통과). score-driven eviction(AttentionScoreAccumulator) 미장착(force_evict score-free, H2O≡recency). `pass_criteria: all` 시나리오(signal_memory_critical)는 device 에서 별도 확인.
- **AB-3 recall host 미검증**: scenario-mode mock + 느린 /tmp 동기 offload blocking → `read length prefix`(transport 아티팩트). 과거 galaxy legacy 는 644 swapped→644 recalled 통과(로그 문자열 동일). device 검증 필요.
- **verify RUST_LOG=info**(AB-1, orchestrator `_engine_env`): `[CacheEvent]`/KIVI 로그가 log::info! 라 필요. 원격(adb/ssh)도 동일 주입 필요(AB-5).
- **non-opencl 회귀 게이트**: `cargo test -p llm_rs2 --lib -- --skip opencl` = 1217 PASS. `test_prune_prefix_calls_release_unused_pages` 병렬압박 flaky(격리 PASS=비-회귀). opencl=host GPU 부재 abort.
- **상세 맵**: AB-2/4/6 전체 조사 = workflow `waq2nsy1b` 결과(legacy 로그 글자단위 + 생존 인프라 file:line + needed_changes + host_testable + risks). AB-2/3/4/6 조사 원본 보존.

---

## 자기점검
- 진입 문장? ✓ "argus-bench AB-2 진행" / "AB-5 verify 재배선 + S25 device 게이트"
- 왜 멈췄나? ✓ host-검증 가능분(AB-0/1/3) 완료, AB-2/4/6 은 device-only-verify(사용자 device 검증 보류 결정)
- 최대 landmine? ✓ AB-4 prepare_tensor_partition(&mut) vs Arc<Model>(get_mut 불가) — 비국소 변경 위험 + GPU host smoke 불가
- 게이트 수치? ✓ host verify 6 시나리오 green(AB-0 4 + AB-1 2) + AB-3 offload smoke / non-opencl 1217 PASS / clippy+fmt clean
- 길이? ✓ 3 커밋 메시지 + workflow waq2nsy1b/wpgh0h1oh 맵이 상세 보존
