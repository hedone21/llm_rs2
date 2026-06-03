//! [`ModelForward`] — first concrete [`Forward`] implementation (Phase 4-3).
//!
//! Wraps [`TransformerModel::forward_into`] for the standard `KVCache` path.
//! Owns the backend handle, model `Arc`, KV caches, decode workspace, lazy
//! prefill workspace, and two reusable logits tensors.
//!
//! Out of scope for 4-3 (all kept as `None` in the forward args):
//! `score_accumulator`, `skip_config`, `profiler`, `importance_collector`,
//! `variance_collector`, `layer_boundary_hook`. These are absorbed by
//! `EvictionStage` / `SwapStage` / `DecodeObserver` in Phase 4-4+.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
#[cfg(feature = "opencl")]
use crate::backend::opencl::plan::FullKernelPlan;
use crate::buffer::DType;
use crate::format::KVCacheFormat;
use crate::kv_cache_ops::KVLayout;
use crate::layers::workspace::{LayerWorkspace, PrefillWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
#[cfg(feature = "opencl")]
use crate::model_config::ModelArch;
use crate::models::transformer::{
    TransformerModel, TransformerModelForwardArgs, TransformerModelForwardFmtArgs,
};
use crate::pressure::kv_cache::KVCache;
use crate::pressure::standard_format::StandardFormat;
use crate::session::traits::{Forward, StepCtx};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Standard `Forward` implementation backed by [`TransformerModel::forward_into`]
/// and a `Vec<KVCache>`.
///
/// Workspace policy (Phase 4-3 §P4 "Hybrid"):
/// - `decode_workspace` is allocated eagerly in [`Self::new`] (small,
///   `[1, 1, *]`-shaped).
/// - `prefill_workspace` is allocated lazily on the first `prefill()` call
///   (large, `[1, seq_len, *]`-shaped). Reallocated if a longer prompt
///   arrives.
pub struct ModelForward {
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: Arc<TransformerModel>,
    kv_caches: Vec<KVCache>,

    decode_workspace: LayerWorkspace,
    // Phase 4-4.5: paradigm equivalence requires `prefill_workspace: None`
    // in `forward_into` args so production owned-ws path is hit. These two
    // fields are kept for future caller-reuse re-enable after the regression
    // is closed; suppress the dead-code lint until then.
    #[allow(dead_code)]
    prefill_workspace: Option<PrefillWorkspace>,
    #[allow(dead_code)]
    max_seq_len: usize,

    // Owned single-token decode input + per-token x_gen scratch + logits.
    // Allocated once to keep the vtable microbench signal clean (no per-step
    // GPU buffer creation).
    decode_input: Tensor,        // [1, 1] U8 (u32 token id)
    decode_x_gen: Tensor,        // [1, 1, hidden]
    logits_decode: Tensor,       // [1, 1, vocab]
    logits_prefill_last: Tensor, // [1, 1, vocab] (logits_last_only=true)

    vocab_size: usize,

    // Phase α-K substep (3c): fmt-cache wiring. `LLMRS_KV_FMT` 게이트 ON 시 prefill 직후(첫 step
    // lazy) `kv_caches` 를 `Vec<Arc<StandardFormat>>` 로 wrap(by-value move, 단일 물리 캐시) →
    // decode fallback 을 `forward_into_fmt`(trait object) 로 전환. 게이트 OFF(None) 시 기존 경로
    // (production 무변). happy-path 전용(eviction=none → NoOpEvictionStage, --no-gpu-plan 강제).
    fmt_caches: Option<Vec<Arc<StandardFormat>>>,

    // fmt-cache 게이트 자격 — **single-prompt happy-path 빌더(build_standard_loop)만 true**.
    // chat(build_chat_standard)/eval 등 prefill 이 멀티턴 재호출되는 경로는 false 로 주입하여
    // `LLMRS_KV_FMT` 가 set 돼 있어도 wrap 이 발동하지 않게 한다(turn2 prefill 이 mem::take 로 빈
    // kv_caches 를 인덱싱하는 panic + eviction 회계 붕괴 차단 — 적대 검증 wiring-safety lens).
    fmt_eligible: bool,

    // Phase 4-4.7 (A1): plan-aware decode. step()이 production fallback
    // (generate.rs l.4351~4477)과 동일하게 execute_plan → forward_into fallback
    // → 다음 step lazy rebuild를 자체적으로 수행한다.
    //
    // `gpu_plan`: 현재 보유 중인 plan (lazy build, invalidation 시 None).
    // `sticky_disabled`: 한 번 build 실패 또는 invalidation lock-out 발동 시
    //   매 step rebuild를 spam하지 않도록 차단 (generate.rs l.4213 패턴).
    // `plan_enabled`: 호출자(`build_standard_loop`)가 `!args.no_gpu_plan`을
    //   전달. CLI `--no-gpu-plan` 활성 시 false → plan path 완전 우회.
    #[cfg(feature = "opencl")]
    gpu_plan: Option<FullKernelPlan>,
    #[cfg(feature = "opencl")]
    sticky_disabled: bool,
    #[cfg(feature = "opencl")]
    plan_enabled: bool,
}

impl ModelForward {
    /// Build a `ModelForward` ready to be passed to
    /// [`crate::session::DecodeLoopBuilder::with_forward`].
    ///
    /// `max_seq_len` caps the lazy `PrefillWorkspace` allocation. KV caches
    /// must already be sized for the same context window.
    // 생성자 — 소유 의존성 다수 (substep 3c 에서 fmt_eligible 추가로 8 인자).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<dyn Backend>,
        memory: Arc<dyn Memory>,
        cpu_backend: Arc<dyn Backend>,
        model: Arc<TransformerModel>,
        kv_caches: Vec<KVCache>,
        max_seq_len: usize,
        #[cfg_attr(not(feature = "opencl"), allow(unused_variables))] plan_enabled: bool,
        // Phase α-K (3c): fmt-cache 게이트 자격. single-prompt happy-path 만 true(아래 필드 doc).
        fmt_eligible: bool,
    ) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let vocab_size = model.config.vocab_size;

        let decode_workspace = LayerWorkspace::new(
            workspace_config_for(&model, max_seq_len),
            memory.as_ref(),
            backend.clone(),
        )?;

        let decode_input_buf = memory.alloc(4, DType::U8)?;
        let decode_input = Tensor::new(Shape::new(vec![1, 1]), decode_input_buf, backend.clone());

        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let decode_x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );

        let logits_decode = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;
        let logits_prefill_last = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;

        Ok(Self {
            backend,
            memory,
            cpu_backend,
            model,
            kv_caches,
            decode_workspace,
            prefill_workspace: None,
            max_seq_len,
            decode_input,
            decode_x_gen,
            logits_decode,
            logits_prefill_last,
            vocab_size,
            fmt_caches: None,
            fmt_eligible,
            #[cfg(feature = "opencl")]
            gpu_plan: None,
            #[cfg(feature = "opencl")]
            sticky_disabled: false,
            #[cfg(feature = "opencl")]
            plan_enabled,
        })
    }

    /// Phase 4-4.7 (A1): plan eligibility 검사 + build 시도.
    ///
    /// production fallback (`generate.rs` l.4186~4199) 가드와 동치 — backend가
    /// OpenCL이고 `--no-gpu-plan` 비활성이며 Gemma3 아닐 때만. score
    /// accumulator / partition / swap_intra_forward 등 추가 가드는 호출자
    /// `is_standard_happy_path`에서 사전 차단되어 도달 시점에 모두 false 보장.
    ///
    /// 결과가 None일 때 `sticky_disabled = true`로 lock-out하여 매 step rebuild
    /// spam을 차단. invalidation 발생 시 호출자가 `gpu_plan = None`으로 set하면
    /// 다음 step 진입에서 sticky_disabled가 false인 경우에만 자동 rebuild.
    #[cfg(feature = "opencl")]
    fn try_build_plan(&mut self) -> Option<FullKernelPlan> {
        // 환경변수 `LLMRS_FWD_TRACE=1` 시 plan path 진입/거부/실패 stderr 로그.
        // Phase 4-4.7 device 측정에서 `build_plan returned None` 위치 진단용.
        // 후속 Phase 4-4.8 plan-path 진단 sprint에서 활용.
        let trace = std::env::var_os("LLMRS_FWD_TRACE").is_some();
        if !self.plan_enabled {
            if trace {
                eprintln!("[fwd-trace] skip: plan_enabled=false");
            }
            return None;
        }
        if self.sticky_disabled {
            if trace {
                eprintln!("[fwd-trace] skip: sticky_disabled");
            }
            return None;
        }
        if self.backend.name() != "OpenCL" {
            if trace {
                eprintln!("[fwd-trace] skip: backend.name()={}", self.backend.name());
            }
            return None;
        }
        if matches!(self.model.config.arch, ModelArch::Gemma3) {
            if trace {
                eprintln!("[fwd-trace] skip: arch=Gemma3");
            }
            return None;
        }
        let plan = self.model.build_plan(
            &self.decode_x_gen,
            &self.logits_decode,
            &self.decode_workspace,
            &mut self.kv_caches,
            &self.backend,
        );
        if plan.is_none() {
            // build_plan이 None 반환 → 본 모델/상태에서 plan path 미지원.
            // 매 step 시도를 막기 위해 sticky lock-out.
            if trace {
                eprintln!("[fwd-trace] build_plan returned None → sticky lock");
            }
            self.sticky_disabled = true;
        } else if trace {
            eprintln!("[fwd-trace] build_plan SUCCESS");
        }
        plan
    }

    /// Borrow the underlying KV caches (Phase 4-4 `EvictionStage` will reach
    /// in via `&mut self` accessors once those traits are wired).
    pub fn kv_caches(&self) -> &[KVCache] {
        &self.kv_caches
    }

    pub fn kv_caches_mut(&mut self) -> &mut [KVCache] {
        &mut self.kv_caches
    }

    pub fn model(&self) -> &Arc<TransformerModel> {
        &self.model
    }

    /// Phase α-K (3c): `LLMRS_KV_FMT` 게이트 ON 시 `kv_caches` 를 `StandardFormat` 으로 1회 wrap.
    ///
    /// prefill 이 채운 캐시를 **by-value move**(`mem::take`)하므로 물리 캐시는 fmt 안에 단 한 벌만
    /// 존재(dual-ownership 부재 — interior mutability 로 forward/eviction 모두 `&self` 통과, ADR-0001
    /// §4.2). 게이트 OFF / 이미 wrap / `kv_caches` 빈 경우 no-op. 첫 `step()` 에서 lazy 호출되므로
    /// prefill(→ `kv_caches` 직접 write) 이후 시점이 보장된다.
    fn ensure_fmt_wrapped(&mut self) {
        // fmt_eligible=false(chat/eval 빌더) 면 env 무관 no-op — 멀티턴 prefill panic 방지.
        if !self.fmt_eligible
            || !standard_format_gate_enabled()
            || self.fmt_caches.is_some()
            || self.kv_caches.is_empty()
        {
            return;
        }
        let caches = std::mem::take(&mut self.kv_caches);
        let fmts: Vec<Arc<StandardFormat>> = caches
            .into_iter()
            .enumerate()
            .map(|(i, c)| Arc::new(StandardFormat::new(i, c)))
            .collect();
        if std::env::var_os("LLMRS_FWD_TRACE").is_some() {
            eprintln!(
                "[fwd-trace] KV_FMT ON: wrapped {} KVCache → StandardFormat (decode = forward_into_fmt)",
                fmts.len()
            );
        }
        self.fmt_caches = Some(fmts);
    }

    /// Construct the input `[1, seq_len]` U32 tensor on the active backend.
    /// CPU-side buffer is built via `Galloc` and uploaded with
    /// `backend.copy_from`, matching the existing prefill path in
    /// `generate.rs`.
    fn build_input_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let cpu_buf = Galloc::new().alloc(seq_len * 4, DType::U8)?;
        // SAFETY: cpu_buf is a freshly allocated [u8] of size seq_len*4 with
        // alignment from Galloc which satisfies u32 alignment (Galloc returns
        // 64B-aligned blocks). We immediately initialise it.
        unsafe {
            let dst = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, seq_len);
        }
        let cpu_tensor = Tensor::new(
            Shape::new(vec![1, seq_len]),
            cpu_buf,
            self.cpu_backend.clone(),
        );
        self.backend.copy_from(&cpu_tensor)
    }

    /// Lazy allocator for `prefill_workspace` with a seq_len realloc guard
    /// (Phase 4-3 §R4). Reuses the existing workspace when its capacity is
    /// already ≥ `seq_len`; otherwise drops and re-allocates.
    #[allow(dead_code)] // Phase 4-4.5: see struct comment.
    fn ensure_prefill_workspace(&mut self, seq_len: usize) -> Result<()> {
        let needs_alloc = match self.prefill_workspace.as_ref() {
            None => true,
            Some(ws) => ws.seq_len() < seq_len,
        };
        if needs_alloc {
            self.prefill_workspace = None; // drop old GPU buffers first
            let config = workspace_config_for(&self.model, self.max_seq_len);
            let ws = PrefillWorkspace::new(
                &config,
                seq_len.min(self.max_seq_len),
                self.memory.as_ref(),
                self.backend.clone(),
            )?;
            self.prefill_workspace = Some(ws);
        }
        Ok(())
    }

    /// Derive a safe `chunk_size` for prefill. CPU (max_single_alloc=usize::MAX)
    /// returns `seq_len` (no chunking needed). GPU mirrors the heuristic in
    /// `generate.rs::auto_gpu_chunk` — `min(budget/(vocab*4), max_alloc/(hidden*4), 512)`
    /// so neither the logits buffer nor activation buffers exceed device limits.
    fn derive_chunk_size(&self, seq_len: usize) -> usize {
        if !self.backend.is_gpu() {
            return seq_len;
        }
        let max_alloc = self.backend.max_single_alloc();
        if max_alloc == 0 || max_alloc == usize::MAX {
            return seq_len;
        }
        let hidden = self.model.config.hidden_size;
        let budget = max_alloc / 2;
        let by_vocab = (budget / (self.vocab_size * 4)).max(1);
        let by_hidden = (max_alloc / (hidden * 4)).max(1);
        by_vocab.min(by_hidden).min(512).min(seq_len)
    }

    /// Read a `[1, 1, vocab]` logits tensor off the backend into a `Vec<f32>`.
    /// Forces a backend sync first so async backends (CUDA/OpenCL) produce a
    /// stable snapshot.
    fn read_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        self.backend.synchronize()?;
        let mut out = vec![0.0f32; self.vocab_size];
        // SAFETY: `out` is a freshly initialised f32 slice of length vocab_size;
        // reinterpreting as [u8; vocab_size*4] is sound for read_buffer (which
        // writes f32 bytes from the GPU buffer back into host memory). The
        // backend implementation does not retain the pointer past the call.
        unsafe {
            let bytes =
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, self.vocab_size * 4);
            self.backend.read_buffer(logits, bytes)?;
        }
        Ok(out)
    }
}

impl Forward for ModelForward {
    fn prefill(&mut self, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("ModelForward::prefill received zero tokens");
        }
        let seq_len = tokens.len();
        let chunk_size = self.derive_chunk_size(seq_len);
        // Phase 4-4.5: pass `prefill_workspace: None` so `forward_into` allocates
        // its own owned workspace per chunk — matches the production prefill
        // path (`generate.rs:3371`) bit-for-bit. The caller-reuse optimisation
        // is deferred until the paradigm-equivalence regression is closed.

        // Phase α-K ①-b: `LLMRS_KV_FMT` 게이트 ON(+ fmt_eligible) 시 prefill 도 fmt 경로로 통일.
        // chunk loop **전** wrap — (3c)는 step()=decode 에서 lazy wrap(prefill 이후)했으나, prefill flip
        // 으로 prefill 시작으로 이동. 이후 decode step() 의 ensure_fmt_wrapped 는 idempotent no-op
        // (fmt_caches 이미 Some). 게이트 OFF / fmt_eligible=false(chat·eval) 면 no-op → 기존 forward_into.
        self.ensure_fmt_wrapped();

        let mut chunk_start = 0;
        while chunk_start < seq_len {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            let chunk = &tokens[chunk_start..chunk_end];
            let input_tensor = self.build_input_tensor(chunk)?;

            // Split mutable handles to avoid double-borrowing `self` inside
            // the FnArgs literal.
            let backend = self.backend.clone();
            let memory_ref: *const dyn Memory = self.memory.as_ref();
            // SAFETY: `self.memory` is owned by `self` and lives across this
            // forward_into call; the raw pointer is dereferenced only on the
            // current stack frame.
            let memory: &dyn Memory = unsafe { &*memory_ref };

            // fmt 게이트 ON → forward_into_fmt(write_kv_batch + multi-token causal attention).
            // concrete Arc clone → transient dyn Vec (step():423-427 패턴). 게이트 ON 시 kv_caches 는
            // ensure_fmt_wrapped 가 mem::take 로 비웠으므로 **반드시** fmt 분기여야 한다(forward_into 는
            // 빈 slice 인덱싱 panic) — 검증 wfceex20u 정정 E.
            let dyn_fmts: Option<Vec<Arc<dyn KVCacheFormat>>> =
                self.fmt_caches.as_ref().map(|fmts| {
                    fmts.iter()
                        .map(|f| f.clone() as Arc<dyn KVCacheFormat>)
                        .collect()
                });
            if let Some(dyn_fmts) = dyn_fmts {
                self.model
                    .forward_into_fmt(TransformerModelForwardFmtArgs {
                        input_tokens: &input_tensor,
                        start_pos: start_pos + chunk_start,
                        fmts: &dyn_fmts,
                        backend: &backend,
                        memory,
                        logits_out: &mut self.logits_prefill_last,
                        x_gen: None,
                        workspace: None,
                        logits_last_only: true,
                        // Phase α-K ①-c: eval feature 필드 (production 은 비활성).
                        score_accumulator: None,
                        skip_config: None,
                        importance_collector: None,
                        cache_self_need_scores: false,
                    })?;
            } else {
                self.model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &input_tensor,
                    start_pos: start_pos + chunk_start,
                    kv_caches: &mut self.kv_caches,
                    backend: &backend,
                    memory,
                    logits_out: &mut self.logits_prefill_last,
                    x_gen: None,
                    workspace: None,
                    prefill_workspace: None,
                    score_accumulator: None,
                    profiler: None,
                    skip_config: None,
                    importance_collector: None,
                    logits_last_only: true,
                    variance_collector: None,
                    layer_boundary_hook: None,
                })?;
            }

            chunk_start = chunk_end;
        }

        // Only the last chunk's last-token logits are kept; intermediate
        // chunks reused the same `logits_prefill_last` buffer in-place.
        self.read_logits(&self.logits_prefill_last)
    }

    fn step(&mut self, ctx: &StepCtx, token: u32) -> Result<Vec<f32>> {
        // Write the single token into the persistent decode_input buffer.
        // `write_buffer` is the same upload path used by the existing decode
        // loop in `generate.rs:2836`.
        let bytes = token.to_ne_bytes();
        self.backend.write_buffer(&mut self.decode_input, &bytes)?;

        // Phase α-K (3c): fmt-cache 게이트. `LLMRS_KV_FMT` ON 시 plan 우회 + `forward_into_fmt`(trait
        // object) 로 decode. 게이트는 `--no-gpu-plan` 동반 강제 전제(plan 활성 + fmt 동시 미지원 —
        // plan 이 `&mut Vec<KVCache>` 를 보는데 fmt 는 move 후 빈 Vec). 게이트 OFF 시 아래 기존 경로
        // (production 무변). transient `Vec<Arc<dyn KVCacheFormat>>` = concrete Arc clone(escape 0,
        // 호출 종료 시 drop) — cold path 라 N Arc clone 비용 무관.
        self.ensure_fmt_wrapped();
        // concrete Arc clone → transient dyn Vec (fmt_caches borrow 는 map 클로저 안에서 종료되어
        // 아래 &mut self 필드 borrow 와 충돌하지 않는다).
        let dyn_fmts: Option<Vec<Arc<dyn KVCacheFormat>>> = self.fmt_caches.as_ref().map(|fmts| {
            fmts.iter()
                .map(|f| f.clone() as Arc<dyn KVCacheFormat>)
                .collect()
        });
        if let Some(dyn_fmts) = dyn_fmts {
            let backend = self.backend.clone();
            let memory_ref: *const dyn Memory = self.memory.as_ref();
            // SAFETY: `self.memory` 는 self 소유, 본 call stack 동안 유효 (기존 fallback 동일 패턴).
            let memory: &dyn Memory = unsafe { &*memory_ref };
            self.model
                .forward_into_fmt(TransformerModelForwardFmtArgs {
                    input_tokens: &self.decode_input,
                    start_pos: ctx.pos,
                    fmts: &dyn_fmts,
                    backend: &backend,
                    memory,
                    logits_out: &mut self.logits_decode,
                    x_gen: Some(&mut self.decode_x_gen),
                    workspace: Some(&mut self.decode_workspace),
                    logits_last_only: false,
                    // Phase α-K ①-c: eval feature 필드 (production 은 비활성).
                    score_accumulator: None,
                    skip_config: None,
                    importance_collector: None,
                    cache_self_need_scores: false,
                })?;
            return self.read_logits(&self.logits_decode);
        }

        // Phase 4-4.7 (A1): plan path 우선 시도. invalidation 또는 build 실패 시
        // forward_into fallback. production fallback (generate.rs l.4351~4376) 패턴.
        #[cfg(feature = "opencl")]
        {
            // Lazy build: gpu_plan이 None이고 sticky_disabled가 false일 때만 시도.
            if self.gpu_plan.is_none() && !self.sticky_disabled {
                self.gpu_plan = self.try_build_plan();
            }

            // borrow 충돌 회피: plan을 step scope로 take, 결과에 따라 복귀/drop.
            // (execute_plan은 `&plan` + `&mut kv_caches` 동시 borrow 필요)
            let plan_opt = self.gpu_plan.take();
            let plan_result = if let Some(plan) = plan_opt.as_ref() {
                let backend = self.backend.clone();
                self.model.execute_plan(
                    plan,
                    &self.decode_input,
                    ctx.pos,
                    &mut self.decode_x_gen,
                    &mut self.kv_caches,
                    &mut self.logits_decode,
                    &backend,
                )
            } else {
                Ok(false)
            };

            match plan_result {
                Ok(true) => {
                    // 성공: plan을 다시 보유 + logits 반환.
                    self.gpu_plan = plan_opt;
                    return self.read_logits(&self.logits_decode);
                }
                Ok(false) | Err(_) => {
                    // Invalidated (KV resize 등) 또는 execute 오류 — plan_opt drop,
                    // gpu_plan은 take()로 이미 None. fallback으로 진행.
                    // 다음 step 진입부에서 lazy rebuild 자동 시도.
                }
            }
        }

        // Fallback: forward_into 직접 호출 (production l.4380~4438과 동치).
        // Same trick as prefill: split &mut borrows so we do not hold &self
        // and &mut self.kv_caches simultaneously inside the args literal.
        let backend = self.backend.clone();
        let memory_ref: *const dyn Memory = self.memory.as_ref();
        let memory: &dyn Memory = unsafe { &*memory_ref };

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.decode_input,
            start_pos: ctx.pos,
            kv_caches: &mut self.kv_caches,
            backend: &backend,
            memory,
            logits_out: &mut self.logits_decode,
            x_gen: Some(&mut self.decode_x_gen),
            workspace: Some(&mut self.decode_workspace),
            prefill_workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            layer_boundary_hook: None,
        })?;

        self.read_logits(&self.logits_decode)
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn on_kv_prune(&mut self, _new_pos: usize) {
        // Phase 4-3 wires `NoOpEvictionStage`, so this hook never fires.
        // When `EvictionStage` learns to reach into `ModelForward::kv_caches_mut`
        // in Phase 4-4, this default no-op is overridden to keep the KV cache
        // `current_pos` in sync with the loop counter.
    }

    fn reset_kv(&mut self) -> anyhow::Result<()> {
        // fmt 활성 시 inner cache 는 StandardFormat 안 → with_cache_mut seam 으로 reset.
        if let Some(fmts) = &self.fmt_caches {
            for f in fmts {
                f.with_cache_mut(|c| c.current_pos = 0);
            }
        } else {
            for cache in &mut self.kv_caches {
                cache.current_pos = 0;
            }
        }
        Ok(())
    }

    fn try_evict(
        &mut self,
        cache_manager: &crate::pressure::cache_manager::CacheManager,
        scores: Option<&[f32]>,
        force: bool,
        target_ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        let before_pos = self.kv_caches.first().map(|c| c.current_pos).unwrap_or(0);

        let result = if force {
            match scores {
                Some(sc) => {
                    cache_manager.force_evict_with_scores(&mut self.kv_caches, target_ratio, sc)?
                }
                None => cache_manager.force_evict(&mut self.kv_caches, target_ratio)?,
            }
        } else {
            match scores {
                Some(sc) => cache_manager.maybe_evict_with_scores(&mut self.kv_caches, sc)?,
                None => cache_manager.maybe_evict(&mut self.kv_caches)?,
            }
        };

        if result.evicted {
            let removed = before_pos.saturating_sub(result.new_pos);
            Ok((removed, result.new_pos))
        } else {
            Ok((0, before_pos))
        }
    }
}

/// `LLMRS_KV_FMT` 게이트 (Phase α-K 3c, 기본 OFF). OnceLock 캐시 → per-step 비용 ~0.
///
/// ON 시 ModelForward decode fallback 이 `forward_into_fmt`(KVCacheFormat trait object) 로 전환.
/// device 검증 전용 임시 게이트 — production 무회귀 우선이라 CLI Args 표면 미오염(env only).
fn standard_format_gate_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("LLMRS_KV_FMT").is_some())
}

fn workspace_config_for(model: &TransformerModel, max_seq_len: usize) -> WorkspaceConfig {
    let head_dim = model.config.head_dim;
    let kv_dim = model.config.num_key_value_heads * head_dim;
    WorkspaceConfig {
        batch_size: 1,
        dim: model.config.hidden_size,
        q_dim: model.config.num_attention_heads * head_dim,
        k_dim: kv_dim,
        v_dim: kv_dim,
        ffn_hidden: model.config.intermediate_size,
        n_heads: model.config.num_attention_heads,
        max_seq_len,
    }
}

fn alloc_logits(
    memory: &dyn Memory,
    backend: Arc<dyn Backend>,
    vocab_size: usize,
) -> Result<Tensor> {
    let buf = memory.alloc(vocab_size * 4, DType::F32)?;
    Ok(Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        buf,
        backend,
    ))
}

/// Allocate a standard `KVCache` per layer using the same recipe as
/// `generate.rs:406` — `HeadMajor` layout, dynamic grow, `kv_buf_size`
/// derived from `dtype`. Exposed for `bin/probe_inference_loop.rs` so the
/// microbench does not need to copy this block.
pub fn alloc_standard_kv_caches(
    model: &TransformerModel,
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    initial_capacity: usize,
    max_seq_len: usize,
    dtype: DType,
) -> Result<Vec<KVCache>> {
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let n_values = initial_capacity * kv_heads * head_dim;
    let kv_buf_size = match dtype {
        DType::Q4_0 => {
            use crate::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * dtype.size(),
    };

    let mut caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let v_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let shape = Shape::new(vec![1, kv_heads, initial_capacity, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(KVLayout::HeadMajor),
        );
    }
    Ok(caches)
}
