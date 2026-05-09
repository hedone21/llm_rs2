pub mod cpu;

// Optional GPU backends (mutually exclusive)
#[cfg(all(feature = "cuda", feature = "cuda-embedded"))]
compile_error!(
    "Features `cuda` and `cuda-embedded` are mutually exclusive — pick one (cuda = PC/discrete GPU, cuda-embedded = Jetson/UMA)"
);

#[cfg(feature = "cuda")]
pub mod cuda_pc;
#[cfg(feature = "cuda")]
pub use cuda_pc as cuda;

#[cfg(feature = "cuda-embedded")]
pub mod cuda_embedded;
#[cfg(feature = "cuda-embedded")]
pub use cuda_embedded as cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

// QNN OpPackage backend (M3.1 skeleton, ENG-QNN-201~210, INV-166~180).
#[cfg(feature = "qnn")]
pub mod qnn_oppkg;
