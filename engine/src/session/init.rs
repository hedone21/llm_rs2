use std::sync::Arc;

use crate::core::backend::Backend;
use crate::core::memory::Memory;
use crate::core::sampling::SamplingConfig;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;

/// Session 초기화 컨텍스트 (Phase 4-1 외곽 추출).
///
/// `SessionInitCtx::build(&args)`가 generate.rs main()의 L1249~L2191 (args 검증,
/// 환경변수 전파, Rayon 초기화, Backend/Memory init, 모델 로드, weight probe)을
/// 수행한다. build() 완료 후 ctx는 모든 하위 decode 경로에서 필요한 값들을 보유한다.
///
/// - `args`는 main()이 owned 보유; build()는 `&Args` borrow + 필요 필드만 clone.
/// - `model`은 owned으로 보유 (KIVI/SwitchHw가 model.layers를 mutation).
/// - `is_gpu`, `weights_on_gpu`는 main()이 mutation할 수 있도록 pub으로 노출.
pub struct SessionInitCtx {
    /// sampling 파라미터 (greedy override 처리 완료).
    pub sampling_config: SamplingConfig,
    /// 모델 파일 경로 (GGUF 또는 safetensors 디렉토리).
    pub model_path: String,
    /// 모델이 GGUF 포맷이면 true.
    pub is_gguf: bool,

    /// 주 backend (CPU 또는 GPU).
    pub backend: Arc<dyn Backend>,
    /// 주 메모리 할당자.
    pub memory: Arc<dyn Memory>,
    /// SwitchHw 용 GPU secondary backend (CPU primary일 때 Some).
    pub gpu_backend_arc: Option<Arc<dyn Backend>>,
    /// SwitchHw 용 GPU secondary 메모리 할당자.
    pub gpu_memory_arc: Option<Arc<dyn Memory>>,
    /// 현재 GPU가 primary backend이면 true (main()이 SwitchHw 후 mutation).
    pub is_gpu: bool,
    /// 모델 weights가 GPU cl_mem에 있으면 true.
    pub weights_on_gpu: bool,

    /// 로드된 모델 (KIVI/SwitchHw가 model.layers를 swap하므로 owned).
    pub model: TransformerModel,
}

impl SessionInitCtx {
    pub fn build(_args: &Args) -> anyhow::Result<Self> {
        unimplemented!(
            "SessionInitCtx::build() — Phase 4-1 C2에서 generate.rs L1249~L2191 이동 예정"
        )
    }
}
