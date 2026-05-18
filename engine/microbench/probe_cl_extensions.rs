//! Probe Adreno OpenCL extensions for AHardwareBuffer / DMA-BUF support.
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin probe_cl_extensions`
//! Run on device: `adb push ... && adb shell ./probe_cl_extensions`
//!
//! Phase 0 enhancements (2026-05-09):
//!   - CL_DEVICE_SVM_CAPABILITIES (0x1053) — coarse/fine-grain buffer/system, atomics
//!   - CL_DEVICE_IL_VERSION (0x105B) — SPIR-V support
//!   - cl_khr_command_buffer + mutable_dispatch + cl_qcom_recordable_queues
//!   - Phase 1/2/3 SKIP 조건 결정 input

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::ffi;

    let platform = ocl::Platform::default();
    eprintln!("OpenCL Platform: {}", platform.name()?);
    eprintln!("Vendor: {}", platform.vendor()?);
    eprintln!("Version: {}", platform.version()?);

    let device = ocl::Device::first(platform)?;
    eprintln!("\nDevice: {}", device.name()?);
    eprintln!("Vendor: {}", device.vendor()?);
    eprintln!("Version: {}", device.version()?);

    if let Ok(cv) = device.info(ocl::core::DeviceInfo::OpenclCVersion) {
        eprintln!("OpenCL C Version: {}", cv);
    }

    let extensions = device
        .info(ocl::core::DeviceInfo::Extensions)
        .map(|v| v.to_string())
        .unwrap_or_default();

    eprintln!("\nAll extensions ({} bytes):", extensions.len());
    for ext in extensions.split_whitespace() {
        eprintln!("  - {}", ext);
    }

    eprintln!("\n=== Key extension presence check ===");
    let candidates = [
        // Host pointer / zero-copy (Phase 3)
        "cl_qcom_android_ahardwarebuffer_host_ptr",
        "cl_qcom_android_native_buffer_host_ptr",
        "cl_qcom_ext_host_ptr_iocoherent",
        "cl_qcom_ext_host_ptr",
        "cl_qcom_ion_host_ptr",
        "cl_qcom_dma_buf_host_ptr",
        "cl_khr_external_memory",
        "cl_khr_external_memory_dma_buf",
        "cl_khr_external_memory_android_hardware_buffer",
        "cl_arm_import_memory",
        "cl_arm_import_memory_host",
        "cl_arm_import_memory_dma_buf",
        // Command buffer / record-replay (Phase 2)
        "cl_khr_command_buffer",
        "cl_khr_command_buffer_mutable_dispatch",
        "cl_khr_command_buffer_multi_device",
        "cl_qcom_recordable_queues",
        // SPIR-V / IL (auxiliary)
        "cl_khr_il_program",
        // Performance hints / priority
        "cl_khr_priority_hints",
        "cl_qcom_priority_hint",
        "cl_qcom_perf_hint",
    ];
    for c in candidates {
        let supported = extensions.contains(c);
        eprintln!("  {} : {}", c, if supported { "YES" } else { "no" });
    }

    eprintln!("\n=== SVM capabilities (CL_DEVICE_SVM_CAPABILITIES = 0x1053) ===");
    const CL_DEVICE_SVM_CAPABILITIES: u32 = 0x1053;
    const CL_DEVICE_SVM_COARSE_GRAIN_BUFFER: u64 = 1 << 0;
    const CL_DEVICE_SVM_FINE_GRAIN_BUFFER: u64 = 1 << 1;
    const CL_DEVICE_SVM_FINE_GRAIN_SYSTEM: u64 = 1 << 2;
    const CL_DEVICE_SVM_ATOMICS: u64 = 1 << 3;

    let mut svm_caps: u64 = 0;
    let mut size_ret: usize = 0;
    let svm_err = unsafe {
        ffi::clGetDeviceInfo(
            device.as_core().as_raw(),
            CL_DEVICE_SVM_CAPABILITIES,
            std::mem::size_of::<u64>(),
            &mut svm_caps as *mut _ as *mut std::ffi::c_void,
            &mut size_ret,
        )
    };
    if svm_err == 0 {
        eprintln!("  raw bitfield: 0x{:016x}", svm_caps);
        eprintln!(
            "  CL_DEVICE_SVM_COARSE_GRAIN_BUFFER : {}",
            if svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER != 0 {
                "YES"
            } else {
                "no"
            }
        );
        eprintln!(
            "  CL_DEVICE_SVM_FINE_GRAIN_BUFFER   : {}",
            if svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0 {
                "YES"
            } else {
                "no"
            }
        );
        eprintln!(
            "  CL_DEVICE_SVM_FINE_GRAIN_SYSTEM   : {}",
            if svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0 {
                "YES"
            } else {
                "no"
            }
        );
        eprintln!(
            "  CL_DEVICE_SVM_ATOMICS             : {}",
            if svm_caps & CL_DEVICE_SVM_ATOMICS != 0 {
                "YES"
            } else {
                "no"
            }
        );
        // Phase 1 SKIP/PROCEED summary
        let phase1_proceed = svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0;
        eprintln!(
            "  >> Phase 1 (SVM fine-grain swap path): {}",
            if phase1_proceed {
                "PROCEED"
            } else {
                "SKIP (no fine-grain buffer)"
            }
        );
    } else {
        eprintln!(
            "  clGetDeviceInfo SVM_CAPABILITIES failed (err={}). Likely OpenCL 1.2 device.",
            svm_err
        );
        eprintln!("  >> Phase 1 (SVM): SKIP (query failed, likely 1.2-only)");
    }

    eprintln!("\n=== IL version (CL_DEVICE_IL_VERSION = 0x105B) ===");
    const CL_DEVICE_IL_VERSION: u32 = 0x105B;
    let mut il_buf = vec![0u8; 256];
    let mut il_size: usize = 0;
    let il_err = unsafe {
        ffi::clGetDeviceInfo(
            device.as_core().as_raw(),
            CL_DEVICE_IL_VERSION,
            il_buf.len(),
            il_buf.as_mut_ptr() as *mut std::ffi::c_void,
            &mut il_size,
        )
    };
    if il_err == 0 && il_size > 0 {
        let s = String::from_utf8_lossy(&il_buf[..il_size.min(il_buf.len()).saturating_sub(1)]);
        eprintln!("  IL_VERSION: \"{}\"", s.trim_matches('\0'));
    } else {
        eprintln!(
            "  clGetDeviceInfo IL_VERSION failed (err={}, size={}). SPIR-V not supported.",
            il_err, il_size
        );
    }

    eprintln!("\n=== Phase decision summary ===");
    let cmd_buf = extensions.contains("cl_khr_command_buffer");
    let mut_dispatch = extensions.contains("cl_khr_command_buffer_mutable_dispatch");
    let recordable_q = extensions.contains("cl_qcom_recordable_queues");
    let phase2_proceed = (cmd_buf && mut_dispatch) || recordable_q;
    eprintln!(
        "  Phase 2 (command_buffer mutable_dispatch): {}",
        if phase2_proceed { "PROCEED" } else { "SKIP" }
    );

    let ion = extensions.contains("cl_qcom_ion_host_ptr");
    let ahb = extensions.contains("cl_qcom_android_ahardwarebuffer_host_ptr");
    let phase3_proceed = ion || ahb;
    eprintln!(
        "  Phase 3 (ION/AHB iocoherent): {} (ion={}, ahb={})",
        if phase3_proceed { "PROCEED" } else { "SKIP" },
        ion,
        ahb
    );

    Ok(())
}

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("This probe requires the `opencl` feature. Rebuild with --features opencl.");
}
