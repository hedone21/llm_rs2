pub mod assembly;
pub mod batch;
pub mod chat;
pub mod chat_ipc;
pub mod chat_template;
pub mod cli;
pub mod decode_fallback;
pub mod decode_loop;
pub mod defaults;
pub mod dump_importance;
pub mod eval;
pub mod forward;
pub mod init;
pub mod ppl;
pub mod prefill;
pub mod qcf_runtime;
pub mod resilience_adapter;
pub mod resilience_init;
pub mod samplers;
pub mod standard_happy;
pub mod swap_runtime;
pub mod traits;
pub mod warmup;

pub use assembly::{build_standard_loop, is_standard_happy_path};
pub use decode_loop::{DecodeLoop, DecodeLoopBuilder, HasForward, NoForward};
pub use defaults::{
    GreedySampler, NoOpCommandSource, NoOpEngineReport, NoOpEvictionStage, NoOpObserver,
    NoOpSwapStage, NoOpTokenTickSink,
};
pub use samplers::RepetitionPenaltySampler;
pub use traits::{
    CommandSource, DecodeObserver, DecodeResult, EngineReport, EvictionOutcome, EvictionStage,
    Forward, ResilienceBundle, SkipReason, StepCtx, StopReason, SwapStage, TokenSampler,
    TokenTickSink,
};
