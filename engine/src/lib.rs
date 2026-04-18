#[cfg(all(feature = "opencl", feature = "cuda"))]
compile_error!("Features 'opencl' and 'cuda' are mutually exclusive. Enable only one.");
#[cfg(all(feature = "opencl", feature = "cuda-embedded"))]
compile_error!("Features 'opencl' and 'cuda-embedded' are mutually exclusive. Enable only one.");

pub mod backend;
pub mod buffer;
pub mod core;
pub mod layers;
pub mod memory;
pub mod models;

pub mod eval;
pub mod experiment;
pub mod profile;
pub mod resilience;
