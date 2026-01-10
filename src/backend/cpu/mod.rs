use anyhow::{Result, anyhow};
use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::buffer::DType;

pub mod common;
pub use common::CpuBackendCommon;

#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "x86_64")]
pub use x86::CpuBackendAVX2;

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon::CpuBackendNeon;

#[cfg(target_arch = "x86_64")]
pub type CpuBackend = CpuBackendAVX2;

#[cfg(target_arch = "aarch64")]
pub type CpuBackend = CpuBackendNeon;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type CpuBackend = CpuBackendCommon;

