#[cfg(all(feature = "opencl", feature = "cuda"))]
compile_error!("Features 'opencl' and 'cuda' are mutually exclusive. Enable only one.");
#[cfg(all(feature = "opencl", feature = "cuda-embedded"))]
compile_error!("Features 'opencl' and 'cuda-embedded' are mutually exclusive. Enable only one.");

pub mod auf;
pub mod backend;
pub mod buffer;
pub mod cpu_kernels;
pub mod hybrid_attention;
pub mod inference;
pub mod instrument;
pub mod kv_cache_ops;
pub mod layer_boundary_hook;
pub mod layers;
pub mod memory;
pub mod model_config;
pub mod models;
pub mod op_kind;
pub mod partition_workspace;
pub mod pressure;
pub mod qcf;
pub mod qcf_collector;
pub mod qcf_computer;
pub mod qcf_types;
pub mod quant;
pub mod shape;
pub mod tensor;
pub mod thread_pool;
pub mod yield_policy;

pub mod experiment;
pub mod observability;
pub mod resilience;
pub mod session;
