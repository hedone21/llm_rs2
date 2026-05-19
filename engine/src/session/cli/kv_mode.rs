use clap::{Args, ValueEnum};

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvMode {
    #[default]
    Standard,
    Kivi,
    Offload,
}

#[derive(Args, Debug, Clone, Default)]
pub struct KvModeArgs {
    /// KV cache mode (default: standard)
    #[arg(long, value_enum, default_value_t = KvMode::Standard)]
    pub kv_mode: KvMode,

    /// KIVI quantization bits (kv-mode=kivi 한정)
    #[arg(long = "kv-kivi-bits", default_value_t = 2)]
    pub kv_kivi_bits: u32,

    /// KIVI residual buffer length (kv-mode=kivi 한정)
    #[arg(long = "kv-kivi-residual-len", default_value_t = 128)]
    pub kv_kivi_residual_len: usize,

    /// Offload storage backend: mmap | tmpfs | ... (kv-mode=offload 한정)
    #[arg(long = "kv-offload-storage", default_value = "mmap")]
    pub kv_offload_storage: String,

    /// Max prefetch depth for offload (kv-mode=offload 한정)
    #[arg(long = "kv-max-prefetch-depth", default_value_t = 2)]
    pub kv_max_prefetch_depth: usize,
}
