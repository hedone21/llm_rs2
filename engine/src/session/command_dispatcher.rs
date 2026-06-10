//! `CommandDispatcher` + `LoopControl` — v2 §5.4 A-1 의 2-source 명령 분배자 (Phase β-4).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.4 (2-source 모델) +
//! `arch/beta4_command_channel_mapping.md` (18-variant × 19필드 전수 매핑 = 구현 명세).
//! 정본 명세: `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md` §β-4.
//!
//! [`CommandSource::poll`](super::traits::CommandSource) 가 pure 생산한 [`EngineCommand`] 들을
//! 받아 v2 §5.4 의 3분류로 분배한다:
//!
//! - **① OneShot EvictionStage** — evict-family 4종(KvEvictH2o/KvEvictSliding/KvStreaming/
//!   KvMergeD2o) → `registry.submit(EvictionStage::one_shot(...))` (method-drop 시맨틱 — directive
//!   의 method 는 무시하고 `keep_ratio`→`target_ratio` 만 사용, 정책은 CM 의 CLI 구성, 3부).
//! - **② LoopControl** — control 7종(throttle/tbt/suspend/resume/restore/qcf/prefill) + 과도기
//!   5종(offload/recall/quant/swap/partition/layer-skip — deprecated, 등가 보존 G1).
//! - **③ Hardware resolve seam** — SwitchHw/PrepareComputeUnit (seam 만, run() 인라인 소비 없음).
//!
//! **sticky 등가 (2부)**: v1 `evict_plan` sticky carry + driver `evict_applied` 1회-게이트 =
//! OneShot Consumed GC 1회성. directive 1회 = submit 1회 = 발화 1회 = GC. RestoreDefaults →
//! 재제출 가능 reset. v1 `evict_applied` 는 dispatcher 내부 sticky 상태로 흡수된다.

use std::sync::{Arc, Mutex};

use llm_shared::{EngineCapability, EngineCommand, QcfEstimate, WeightSwapReport};

use crate::pressure::cache_manager::CacheManager;
use crate::pressure::standard_format::StandardFormat;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::stages::kv::eviction::EvictionStage;

/// External command channel (manager IPC, schedule, stdin, ...).
///
/// **Phase β-7**: moved here from the deleted `session::traits` — this is the
/// dispatcher's input seam.
///
/// **β-4 retarget (v2 §5.4 A-1)**: `poll` 은 **pure 생산자**다 — drain 한
/// [`EngineCommand`] 들을 그대로 반환할 뿐, `ExecutionPlan` 으로 번역하지 않고
/// registry 도 모른다. 번역(① OneShot Stage submit / ② LoopControl / ③ Hardware seam)은
/// [`CommandDispatcher`] 책임이다.
///
/// heartbeat 등 부수효과(매핑 문서 4부 채택안 (가))는 source 구현체 내부에 잔존한다 —
/// `kv_snap` 운반은 poll 인자가 아니라 source 가 register 시점 보유한 held-handle query 로
/// 교체된다(`ManagerCommandSource`). pure poll 은 `ctx`/`kv_snap` 인자가 없다.
pub trait CommandSource {
    /// Per-step poll — 도착한 manager command 들을 drain 하여 반환한다 (pure).
    /// Default Noop 은 빈 `Vec` 을 반환.
    fn poll(&mut self) -> anyhow::Result<Vec<EngineCommand>>;
}

/// Outbound reporting channel (engine → manager).
///
/// **Phase β-7**: moved here from the deleted `session::traits`. Implemented by
/// `ResilienceAdapter` (the surviving `CommandExecutor` side).
pub trait EngineReport {
    fn send_capability(&mut self, _cap: EngineCapability) {}
    fn send_qcf_estimate(&mut self, _qcf: QcfEstimate) {}
    fn send_swap_report(&mut self, _report: WeightSwapReport) {}
}

/// ExecutionPlan 축소판 — driver-local 루프 제어 상태 (v2 §5.4 ② channel).
///
/// `CommandDispatcher::dispatch` 가 매 step 갱신하고, `DecodeLoop::run` 이 읽어 sleep/break/pacing
/// 한다. v1 `ExecutionPlan` 의 control 필드와 1:1 (매핑 문서 1.2/1.3).
///
/// **과도기 5종 필드(offload_ratio/recall_offload/restore_defaults/kv_quant_bits/partition_ratio/
/// layer_skip/swap_weights)** 는 대응 Stage(AB-2/4/6) 미구현이라 deprecated 로 잔존한다(G1) —
/// 구==신 등가만 보장하고, run() 인라인 소비는 (a.6) offload/recall 만 live 다(나머지는 generate/
/// forward 측 sticky).
#[derive(Debug, Clone, Default)]
pub struct LoopControl {
    // ── ② control 핵심 (run() live 소비) ──
    /// Throttle delay between tokens (ms). 0 = no throttle. (Throttle)
    pub throttle_delay_ms: u64,
    /// Target TBT in ms. 0 = disabled. (SetTargetTbt)
    pub target_tbt_ms: u64,
    /// True iff `target_tbt_ms` was explicitly set at least once. (SetTargetTbt / RestoreDefaults)
    pub target_tbt_set: bool,
    /// Whether inference should be suspended → loop break (G6 보존). (Suspend)
    pub suspended: bool,

    // ── ② control 비-live (seam 잔존, run() 미소비) ──
    /// Whether inference should resume from suspension. (Resume — executor 내부 state 만)
    pub resumed: bool,
    /// Whether Engine should compute and send QCF estimates. (RequestQcf — EngineReport 경로)
    pub request_qcf: bool,
    /// Prefill policy update (batch/runner.rs 경로 소비). (SetPrefillPolicy)
    pub prefill_chunk_size: Option<usize>,
    pub prefill_yield_ms: Option<u32>,
    pub prefill_cpu_chunk_size: Option<usize>,

    // ── ② RestoreDefaults 묶음 (a.6 live recall) ──
    /// True when a RestoreDefaults directive arrives and offloaded KV should be recalled.
    pub recall_offload: bool,
    /// Whether to restore all action-induced state to defaults.
    pub restore_defaults: bool,

    // ── 과도기 5종 (deprecated, G1 — AB-2/4/6 Stage 이전 예정) ──
    /// KV cache offload ratio. (KvOffload — (a.6) live `forward.try_offload`)
    pub offload_ratio: Option<f32>,
    /// KV quantization bits. (KvQuantDynamic — sticky, generate/forward 측)
    pub kv_quant_bits: Option<u8>,
    /// Tensor partition ratio. (SetPartitionRatio — sticky, forward/transformer 측)
    pub partition_ratio: Option<f32>,
    /// Layer skip ratio. (LayerSkip — sticky, forward skip_config 측)
    pub layer_skip: Option<f32>,
    /// Weight swap request. (SwapWeights — take_pending_swap_weights 경로)
    pub swap_weights: Option<(f32, llm_shared::DtypeTag)>,

    // ── ③ Hardware resolve seam (run() 미소비) ──
    /// Device to switch to. (SwitchHw)
    pub switch_device: Option<String>,
    /// Device to pre-warm. (PrepareComputeUnit)
    pub prepare_device: Option<String>,
}

/// v2 §5.4 A-1 의 명령 분배자. `EngineCommand` 를 ① OneShot Stage submit / ② LoopControl /
/// ③ Hardware seam 으로 분배한다.
///
/// **L4 동층 합성** (INV-LAYER-006 BANNED 비해당): dispatcher 는 driver(`DecodeLoop`)와 같은 L4 에서
/// 합성되며, registry(L4)·CacheManager(L3, `Arc<Mutex>`)·held-handle(`Arc<StandardFormat>`) 를 보유한다.
pub struct CommandDispatcher {
    /// ① evict directive 가 EvictionStage 를 submit 할 stage registry (driver 와 공유).
    registry: Arc<PipelineRegistry>,
    /// ① EvictionStage 가 prune 할 KV handle (register 시점 보유, INV-STAGE-LAYER-HANDLE).
    kv_handles: Vec<Arc<StandardFormat>>,
    /// ① EvictionStage 들이 공유하는 단일 CacheManager (CLI 정책·sticky eviction 상태).
    /// `None` 이면 evict directive 가 와도 submit 안 함(happy/chat 동등 — eviction 미구성).
    cache_manager: Option<Arc<Mutex<CacheManager>>>,
    /// ② 누적 루프 제어 상태 (sticky control — throttle/tbt 유지, evict 는 OneShot 으로 분리).
    control: LoopControl,

    // ── sticky 상태 (2부 — v1 executor 의 sticky carry/게이트 흡수) ──
    /// 상태 A 등가(v1 `evict_applied`): active 구간당 evict OneShot 1회만 submit.
    /// evict directive 도착 시 true, RestoreDefaults 도착 시 false 로 재무장.
    evict_armed: bool,
}

impl CommandDispatcher {
    /// dispatcher 생성. `cache_manager` 가 `None` 이면 evict directive 는 무시된다(미구성).
    pub fn new(
        registry: Arc<PipelineRegistry>,
        kv_handles: Vec<Arc<StandardFormat>>,
        cache_manager: Option<Arc<Mutex<CacheManager>>>,
    ) -> Self {
        Self {
            registry,
            kv_handles,
            cache_manager,
            control: LoopControl::default(),
            evict_armed: false,
        }
    }

    /// dispatcher 가 보유한 공유 `CacheManager` 에 대한 접근자.
    ///
    /// (a.6) KvOffload/recall 이 `forward.try_offload`/`try_recall` 에 `&mut CacheManager` 를 넘길 때
    /// driver 가 lock 후 사용한다. `None` 이면 미구성(happy/chat).
    pub fn cache_manager(&self) -> Option<&Arc<Mutex<CacheManager>>> {
        self.cache_manager.as_ref()
    }

    /// 마지막 [`Self::dispatch`] 가 갱신한 누적 [`LoopControl`] 읽기 (driver 가 (a.6) 등에서 사용).
    pub fn control(&self) -> &LoopControl {
        &self.control
    }

    /// 도착한 command 들을 분배하고 갱신된 [`LoopControl`] 을 반환한다.
    ///
    /// 구 `CommandExecutor::apply_command`(executor.rs:360-571) + `poll` 후처리(:344-355) 로직 이동:
    /// - **transient reset**: control 의 1-step 필드(evict 트리거 제외 transient)는 매 dispatch 진입
    ///   시 초기화하되, sticky 필드(throttle/tbt/quant/partition)는 carry. v1 `ExecutionPlan::default`
    ///   에서 시작 후 sticky carry 하던 것과 등가.
    /// - **suspend override**: suspended 면 evict 미submit + device seam clear (v1 :344-352 등가).
    pub fn dispatch(&mut self, cmds: Vec<EngineCommand>) -> &LoopControl {
        // transient(매 step 새로 결정되는) 필드만 초기화 — sticky(throttle/tbt/quant/partition)는 carry.
        self.control.suspended = false;
        self.control.resumed = false;
        self.control.request_qcf = false;
        self.control.recall_offload = false;
        self.control.restore_defaults = false;
        self.control.offload_ratio = None;
        self.control.swap_weights = None;
        self.control.switch_device = None;
        self.control.prepare_device = None;
        self.control.prefill_chunk_size = None;
        self.control.prefill_yield_ms = None;
        self.control.prefill_cpu_chunk_size = None;

        for cmd in &cmds {
            self.apply(cmd);
        }

        // 상태 A/B 재무장 시맨틱 (2부 핵심 등가 명제, line 186-194): v1 의 disarm 은 **빈 batch 가
        // 아니라 RestoreDefaults** 에서만 일어난다. v1 evict_plan 이 sticky carry 되어 빈 poll 에서도
        // plan.evict 가 Some 유지 → evict_applied 가 reset 되지 않기 때문(decode_loop.rs:279 의
        // plan.evict.is_none() 분기는 evict_plan=None 일 때만 = RestoreDefaults 후). 따라서 disarm 은
        // apply 의 RestoreDefaults arm 에서만 수행한다(빈 batch 에서 disarm 금지 — active 구간 유지).

        // suspend override (v1 poll :344-352): suspend 면 device seam 무효 + throttle 0.
        if self.control.suspended {
            self.control.switch_device = None;
            self.control.prepare_device = None;
            self.control.throttle_delay_ms = 0;
            self.control.resumed = false;
        }

        &self.control
    }

    /// 단일 command 분배.
    fn apply(&mut self, cmd: &EngineCommand) {
        match cmd {
            // ── ① evict-family 4종 → OneShot EvictionStage submit (method-drop, 3부) ──
            EngineCommand::KvEvictH2o { keep_ratio }
            | EngineCommand::KvEvictSliding { keep_ratio }
            | EngineCommand::KvMergeD2o { keep_ratio } => {
                self.submit_evict(*keep_ratio);
            }
            EngineCommand::KvStreaming { .. } => {
                // StreamingLLM 은 target_len 무시 → target_ratio=0.0 (CM CLI sink/window 정책 사용).
                self.submit_evict(0.0);
            }

            // ── ② control 7종 → LoopControl ──
            EngineCommand::Throttle { delay_ms } => {
                self.control.throttle_delay_ms = *delay_ms;
            }
            EngineCommand::SetTargetTbt { target_ms } => {
                self.control.target_tbt_ms = *target_ms;
                self.control.target_tbt_set = true;
            }
            EngineCommand::Suspend => {
                self.control.suspended = true;
            }
            EngineCommand::Resume => {
                self.control.resumed = true;
                self.control.throttle_delay_ms = 0;
            }
            EngineCommand::RestoreDefaults => {
                // v1 RestoreDefaults(:484-502) 등가 — reset 묶음.
                self.control.restore_defaults = true;
                self.control.recall_offload = true;
                self.control.throttle_delay_ms = 0;
                self.control.target_tbt_ms = 0;
                // target_tbt_set 은 false 로 (CLI fallback branch — v1 :489 주석 등가).
                self.control.target_tbt_set = false;
                self.control.kv_quant_bits = None;
                self.control.partition_ratio = None;
                // 상태 C 재무장: 다음 KvEvict* directive 가 새 OneShot submit 가능 (2부).
                self.evict_armed = false;
            }
            EngineCommand::RequestQcf => {
                self.control.request_qcf = true;
            }
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                if let Some(v) = chunk_size {
                    self.control.prefill_chunk_size = Some(*v);
                }
                if let Some(v) = yield_ms {
                    self.control.prefill_yield_ms = Some(*v);
                }
                if let Some(v) = cpu_chunk_size {
                    self.control.prefill_cpu_chunk_size = Some(*v);
                }
            }

            // ── 과도기 5종 → LoopControl deprecated 필드 (G1) ──
            EngineCommand::KvOffload { ratio } => {
                self.control.offload_ratio = Some(ratio.clamp(0.0, 1.0));
            }
            EngineCommand::KvQuantDynamic { target_bits } => {
                self.control.kv_quant_bits = Some(*target_bits);
            }
            EngineCommand::SetPartitionRatio { ratio } => {
                self.control.partition_ratio = Some(*ratio);
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                self.control.layer_skip = Some(*skip_ratio);
            }
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                self.control.swap_weights = Some((*ratio, *target_dtype));
            }

            // ── ③ Hardware resolve seam ──
            EngineCommand::SwitchHw { device } => {
                self.control.switch_device = Some(device.clone());
            }
            EngineCommand::PrepareComputeUnit { device } => {
                self.control.prepare_device = Some(device.clone());
            }
        }
    }

    /// ① evict directive 1건을 OneShot `EvictionStage` 로 submit (method-drop).
    ///
    /// 상태 A/B 등가(2부): `evict_armed` 게이트로 active 구간당 1회만 submit. CacheManager 미구성
    /// (`None`)이거나 handle 이 없으면 no-op(happy/chat 동등 — v1 `cache_manager=None` 분기).
    fn submit_evict(&mut self, target_ratio: f32) {
        if self.evict_armed {
            return; // 이미 active 구간 내 submit 됨 (재적용 방지 — v1 evict_applied 등가).
        }
        let Some(cm) = self.cache_manager.as_ref() else {
            return; // 미구성 — directive 무시 (happy/chat).
        };
        if self.kv_handles.is_empty() {
            return;
        }
        self.evict_armed = true;
        let stage = EvictionStage::one_shot(self.kv_handles.clone(), Arc::clone(cm), target_ratio);
        self.registry.submit(Arc::new(stage));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::KVCacheFormat;
    use crate::memory::host::shared::SharedBuffer;
    use crate::pressure::eviction::sliding_window::SlidingWindowPolicy;
    use crate::pressure::kv_cache::KVCache;
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    const N_TOKENS: usize = 120;

    fn make_handle(n_tokens: usize) -> Arc<StandardFormat> {
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut cache = KVCache::new(k, v, MAX_SEQ);
        cache.current_pos = n_tokens;
        Arc::new(StandardFormat::new(0, cache))
    }

    fn make_cm() -> Arc<Mutex<CacheManager>> {
        let policy = Box::new(SlidingWindowPolicy::new(10, 4));
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            0.3,
        )))
    }

    fn make_dispatcher() -> (
        CommandDispatcher,
        Arc<PipelineRegistry>,
        Arc<StandardFormat>,
    ) {
        let registry = Arc::new(PipelineRegistry::new());
        let handle = make_handle(N_TOKENS);
        let cm = make_cm();
        let d = CommandDispatcher::new(Arc::clone(&registry), vec![handle.clone()], Some(cm));
        (d, registry, handle)
    }

    // ── ② control: throttle/tbt/suspend ──

    #[test]
    fn throttle_sets_control() {
        let (mut d, _r, _h) = make_dispatcher();
        let c = d.dispatch(vec![EngineCommand::Throttle { delay_ms: 50 }]);
        assert_eq!(c.throttle_delay_ms, 50);
        // sticky: 다음 dispatch 에서도 유지.
        let c2 = d.dispatch(vec![]);
        assert_eq!(c2.throttle_delay_ms, 50);
    }

    #[test]
    fn set_target_tbt_dirty_flag() {
        let (mut d, _r, _h) = make_dispatcher();
        let c0 = d.dispatch(vec![]);
        assert!(!c0.target_tbt_set);
        let c1 = d.dispatch(vec![EngineCommand::SetTargetTbt { target_ms: 250 }]);
        assert!(c1.target_tbt_set);
        assert_eq!(c1.target_tbt_ms, 250);
        // RestoreDefaults → set=false, ms=0.
        let c2 = d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert!(!c2.target_tbt_set);
        assert_eq!(c2.target_tbt_ms, 0);
        assert!(c2.restore_defaults);
    }

    #[test]
    fn suspend_overrides_device_seam() {
        let (mut d, _r, _h) = make_dispatcher();
        let c = d.dispatch(vec![
            EngineCommand::SwitchHw {
                device: "cpu".to_string(),
            },
            EngineCommand::Suspend,
        ]);
        assert!(c.suspended);
        assert!(c.switch_device.is_none(), "suspend → device seam clear");
    }

    // ── ① evict OneShot submit + sticky 1회성 (상태 A) ──

    #[test]
    fn evict_submits_one_shot_once() {
        let (mut d, registry, _h) = make_dispatcher();
        assert_eq!(registry.len(), 0);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1, "첫 evict directive → OneShot 1개 submit");
        // sticky carry(빈 batch)에도 재submit 안 함 (상태 A — evict_armed 게이트).
        d.dispatch(vec![]);
        assert_eq!(registry.len(), 1, "빈 batch — 재submit 없음");
        // 같은 directive 반복도 재submit 안 함.
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            1,
            "active 구간 내 directive 반복 — 재submit 없음"
        );
    }

    #[test]
    fn restore_defaults_rearms_evict() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1);
        // RestoreDefaults → 재무장 (다음 KvEvict* 가 새 OneShot submit 가능).
        d.dispatch(vec![EngineCommand::RestoreDefaults]);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            2,
            "RestoreDefaults 후 재무장 → 새 OneShot submit"
        );
    }

    /// v1 sticky carry 등가: evict directive 없는 batch(빈/다른 directive)에서는 disarm 되지
    /// 않는다 — v1 evict_plan 이 sticky carry 되어 plan.evict 가 Some 유지되므로 evict_applied 도
    /// reset 안 됨. disarm 은 오직 RestoreDefaults 에서만 (2부 핵심 등가 명제).
    #[test]
    fn evict_stays_armed_without_restore_defaults() {
        let (mut d, registry, _h) = make_dispatcher();
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 1);
        // evict directive 없는 batch (다른 directive) → 여전히 armed (sticky carry 등가, disarm 아님).
        d.dispatch(vec![EngineCommand::Throttle { delay_ms: 10 }]);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(
            registry.len(),
            1,
            "RestoreDefaults 없이는 disarm 안 됨 — 재submit 없음"
        );
    }

    // ── method-drop (3부): KvEvictH2o directive 라도 stage 는 method 안 받음 (target_ratio 만) ──

    #[test]
    fn method_drop_h2o_directive_submits_target_ratio_only() {
        let (mut d, registry, handle) = make_dispatcher();
        // CLI 정책 = sliding(make_cm). KvEvictH2o directive 가 와도 sliding 으로 prune 됨.
        d.dispatch(vec![EngineCommand::KvEvictH2o { keep_ratio: 0.3 }]);
        assert_eq!(registry.len(), 1);
        // stage 발화 (PreEviction dispatch) → sliding prune.
        let mut profiler = crate::observability::profile::OpProfiler::new();
        let mut ctx = crate::pipeline::StageContext {
            step: crate::pipeline::StepInfo {
                pos: N_TOKENS,
                decode_step: 0,
                pressure: crate::pipeline::Pressure::new(0),
                prev_token: 0,
            },
            profiler: &mut profiler,
        };
        use crate::pipeline::{LifecyclePhase, PipelineDispatcher};
        registry.dispatch(LifecyclePhase::PreEviction, &mut ctx);
        assert!(
            handle.current_pos() < N_TOKENS,
            "method-drop: directive method 무시, CM 정책(sliding)으로 prune"
        );
    }

    // ── 과도기 5종 등가 (KvOffload/SwapWeights/KvQuantDynamic) ──

    #[test]
    fn transitional_fields_equivalence() {
        let (mut d, _r, _h) = make_dispatcher();
        let c = d.dispatch(vec![
            EngineCommand::KvOffload { ratio: 0.5 },
            EngineCommand::KvQuantDynamic { target_bits: 4 },
            EngineCommand::SwapWeights {
                ratio: 0.9,
                target_dtype: llm_shared::DtypeTag::Q4_0,
            },
            EngineCommand::SetPartitionRatio { ratio: 0.3 },
            EngineCommand::LayerSkip { skip_ratio: 0.25 },
        ]);
        assert_eq!(c.offload_ratio, Some(0.5));
        assert_eq!(c.kv_quant_bits, Some(4));
        assert_eq!(c.swap_weights, Some((0.9, llm_shared::DtypeTag::Q4_0)));
        assert_eq!(c.partition_ratio, Some(0.3));
        assert_eq!(c.layer_skip, Some(0.25));
        // sticky: quant/partition 은 다음 dispatch 에 carry, offload/swap 은 transient.
        let c2 = d.dispatch(vec![]);
        assert_eq!(c2.kv_quant_bits, Some(4), "quant sticky");
        assert_eq!(c2.partition_ratio, Some(0.3), "partition sticky");
        assert_eq!(c2.offload_ratio, None, "offload transient");
        assert_eq!(c2.swap_weights, None, "swap transient");
    }

    #[test]
    fn restore_defaults_clears_sticky_quant_partition() {
        let (mut d, _r, _h) = make_dispatcher();
        d.dispatch(vec![
            EngineCommand::KvQuantDynamic { target_bits: 8 },
            EngineCommand::SetPartitionRatio { ratio: 0.4 },
        ]);
        let c = d.dispatch(vec![EngineCommand::RestoreDefaults]);
        assert_eq!(c.kv_quant_bits, None);
        assert_eq!(c.partition_ratio, None);
        assert!(c.recall_offload, "RestoreDefaults → recall_offload");
    }

    // ── exhaustive match 컴파일 강제 (게이트 2): 18 variant 가 apply 에서 누락되면 컴파일 실패 ──
    // (apply 의 match 가 non-exhaustive `_` 없이 18 arm 을 전부 다루므로, variant 추가 시 컴파일 에러.)

    #[test]
    fn cache_manager_none_ignores_evict() {
        let registry = Arc::new(PipelineRegistry::new());
        let handle = make_handle(N_TOKENS);
        let mut d = CommandDispatcher::new(Arc::clone(&registry), vec![handle], None);
        d.dispatch(vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }]);
        assert_eq!(registry.len(), 0, "CM 미구성 → evict directive 무시");
    }
}
