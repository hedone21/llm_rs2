pub mod cpu;

// Optional GPU backends (mutually exclusive)
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;
