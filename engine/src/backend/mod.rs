pub mod cpu;

// Optional GPU backends (mutually exclusive)
#[cfg(feature = "opencl")]
pub mod opencl;
