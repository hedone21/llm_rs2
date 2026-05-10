#![allow(clippy::all, dead_code, unused_variables, unused_parens)]
//! Probe Qualcomm OpenCL shared-memory allocation paths.
//!
//! Tries multiple `allocation_type` × `host_cache_policy` × flag combinations
//! to find one that Adreno 830 driver accepts without ION/DMA-BUF/ANB
//! infrastructure.
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin probe_qcom_alloc`

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use std::ffi::c_void;
    use std::ptr;

    let platform = ocl::Platform::default();
    eprintln!("Platform: {}", platform.name()?);
    let device = ocl::Device::first(platform)?;
    eprintln!("Device: {}", device.name()?);

    let context = ocl::Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let ctx_ptr = context.as_ptr();

    // Query Adreno-specific device info
    use ocl::core::DeviceInfo;
    if let Ok(v) = device.info(DeviceInfo::Extensions) {
        let s = v.to_string();
        for keyword in [
            "ext_host_ptr",
            "iocoherent",
            "ahardwarebuffer",
            "dmabuf",
            "ion",
            "external_memory",
            "android_native",
            "direct_import",
        ] {
            for ext in s.split_whitespace() {
                if ext.contains(keyword) {
                    eprintln!("  ext: {}", ext);
                    break;
                }
            }
        }
    }

    // Try CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM (0x40A0) and CL_DEVICE_PAGE_SIZE_QCOM (0x40A1)
    {
        let mut padding: u32 = 0;
        let mut page_size: u32 = 0;
        let mut sz: usize = 0;
        unsafe {
            let _ = ocl::ffi::clGetDeviceInfo(
                device.as_raw(),
                0x40A0, // CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
                std::mem::size_of::<u32>(),
                &mut padding as *mut _ as *mut c_void,
                &mut sz,
            );
            let _ = ocl::ffi::clGetDeviceInfo(
                device.as_raw(),
                0x40A1, // CL_DEVICE_PAGE_SIZE_QCOM
                std::mem::size_of::<u32>(),
                &mut page_size as *mut _ as *mut c_void,
                &mut sz,
            );
        }
        eprintln!("EXT_MEM_PADDING_QCOM = {} bytes", padding);
        eprintln!("PAGE_SIZE_QCOM = {} bytes", page_size);
    }

    // Constants
    const CL_MEM_READ_ONLY: u64 = 1 << 2;
    const CL_MEM_READ_WRITE: u64 = 1 << 0;
    const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
    const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
    const CL_MEM_EXT_HOST_PTR_QCOM: u64 = 1 << 29;

    #[repr(C)]
    struct ExtHostPtr {
        allocation_type: u32,
        host_cache_policy: u32,
    }

    // Sub-extension struct for ION (matches cl_ext.h)
    #[repr(C)]
    struct ExtHostPtrIon {
        ext: ExtHostPtr,
        ion_filedesc: i32,
        ion_hostptr: *mut c_void,
    }

    // Sub-extension struct for ANB (Android Native Buffer)
    #[repr(C)]
    struct ExtHostPtrAnb {
        ext: ExtHostPtr,
        anb_ptr: *mut c_void,
    }

    let size: usize = 4096; // small probe buffer

    // Helper: try clCreateBuffer and report
    let try_alloc = |label: &str, flags: u64, host_ptr: *mut c_void, sz: usize| -> bool {
        let mut errcode: i32 = 0;
        let raw =
            unsafe { ocl::ffi::clCreateBuffer(ctx_ptr, flags as u64, sz, host_ptr, &mut errcode) };
        let ok = errcode == 0 && !raw.is_null();
        if ok {
            unsafe { ocl::ffi::clReleaseMemObject(raw) };
        }
        eprintln!(
            "  [{}] {}: errcode={}, ptr={}",
            if ok { "OK " } else { "FAIL" },
            label,
            errcode,
            !raw.is_null()
        );
        ok
    };

    eprintln!("\n=== Baseline ===");
    try_alloc(
        "ALLOC_HOST_PTR (no ext)",
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        ptr::null_mut(),
        size,
    );

    eprintln!("\n=== Probe allocation_type values with iocoherent cache policy ===");
    let cache_iocoherent: u32 = 0x40A9;
    let alloc_types = [
        (0u32, "0 (default)"),
        (0x40A4, "HOST_UNCACHED (cache value)"),
        (0x40A5, "HOST_WRITEBACK (cache value)"),
        (0x40A8, "ION_HOST_PTR_QCOM"),
        (0x40A9, "HOST_IOCOHERENT (cache value)"),
        (0x40AA, "0x40AA"),
        (0x40AB, "0x40AB"),
        (0x40AC, "0x40AC"),
        (0x40AD, "0x40AD"),
        (0x40AE, "0x40AE"),
        (0x40AF, "0x40AF"),
        (0x40B0, "0x40B0"),
        (0x40B1, "0x40B1"),
        (0x40B2, "0x40B2"),
        (0x40B3, "0x40B3"),
        (0x40B4, "0x40B4"),
        (0x40B5, "0x40B5"),
        (0x40B6, "0x40B6"),
        (0x40B7, "0x40B7"),
        (0x40B8, "0x40B8"),
        (0x40B9, "0x40B9"),
        (0x40BA, "0x40BA"),
        (0x40BB, "0x40BB"),
        (0x40BC, "0x40BC"),
        (0x40BD, "0x40BD"),
        (0x40BE, "0x40BE"),
        (0x40BF, "0x40BF"),
        (0x40C0, "0x40C0"),
        (0x40C1, "0x40C1"),
        (0x40C2, "0x40C2"),
        (0x40C3, "0x40C3"),
        (0x40C4, "0x40C4"),
        (0x40C5, "0x40C5"),
        (0x40C6, "ANB_HOST_PTR_QCOM"),
        (0x40C7, "0x40C7"),
        (0x40C8, "0x40C8"),
    ];
    for (alloc_type, label) in alloc_types {
        let ext = ExtHostPtr {
            allocation_type: alloc_type,
            host_cache_policy: cache_iocoherent,
        };
        try_alloc(
            &format!("ext_host_ptr alloc_type={} ({})", alloc_type, label),
            CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM,
            &ext as *const _ as *mut c_void,
            size,
        );
    }

    eprintln!("\n=== ION-style struct (alloc=0x40A8) with various cache policies ===");
    for (cache, label) in [
        (0u32, "0"),
        (0x40A4, "UNCACHED"),
        (0x40A5, "WRITEBACK"),
        (0x40A6, "WRITETHROUGH"),
        (0x40A7, "WRITE_COMBINING"),
        (0x40A9, "IOCOHERENT"),
    ] {
        let ext = ExtHostPtrIon {
            ext: ExtHostPtr {
                allocation_type: 0x40A8,
                host_cache_policy: cache,
            },
            ion_filedesc: -1,
            ion_hostptr: ptr::null_mut(),
        };
        try_alloc(
            &format!("ION-style cache=0x{:X} ({})", cache, label),
            CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM,
            &ext as *const _ as *mut c_void,
            size,
        );
    }

    eprintln!("\n=== ANB-style struct (alloc=0x40C6) — anb_ptr = NULL ===");
    let ext_anb = ExtHostPtrAnb {
        ext: ExtHostPtr {
            allocation_type: 0x40C6,
            host_cache_policy: 0x40A9,
        },
        anb_ptr: ptr::null_mut(),
    };
    try_alloc(
        "ANB-style anb_ptr=NULL",
        CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM,
        &ext_anb as *const _ as *mut c_void,
        size,
    );

    eprintln!("\n=== USE_HOST_PTR (user-provided buffer) variants ===");
    // posix_memalign to page-aligned
    let mut aligned: *mut c_void = ptr::null_mut();
    unsafe {
        let _ = libc::posix_memalign(&mut aligned, 4096, size);
    }
    if !aligned.is_null() {
        try_alloc(
            "USE_HOST_PTR aligned",
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            aligned,
            size,
        );
        // EXT + USE_HOST_PTR with iocoherent
        for (alloc_type, label) in [
            (0u32, "0"),
            (0x40A8, "ION"),
            (0x40A9, "IOC"),
            (0x40C6, "ANB"),
        ] {
            let ext = ExtHostPtr {
                allocation_type: alloc_type,
                host_cache_policy: 0x40A9,
            };
            let ext_ptr = &ext as *const _ as *mut c_void;
            // EXT struct passed as host_ptr alongside USE_HOST_PTR — note the
            // user pointer is the EXT struct here (driver reads alloc_type).
            try_alloc(
                &format!(
                    "USE_HOST_PTR + EXT_QCOM alloc=0x{:X} ({})",
                    alloc_type, label
                ),
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                ext_ptr,
                size,
            );
            // Just EXT_QCOM with the EXT struct (driver allocates internally)
            try_alloc(
                &format!("EXT_QCOM-only alloc=0x{:X} ({})", alloc_type, label),
                CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM,
                ext_ptr,
                size,
            );
        }
        unsafe { libc::free(aligned) };
    }

    eprintln!("\n=== READ_WRITE variants ===");
    let ext_rw = ExtHostPtr {
        allocation_type: 0,
        host_cache_policy: 0x40A9,
    };
    try_alloc(
        "READ_WRITE + EXT_QCOM alloc=0",
        CL_MEM_READ_WRITE | CL_MEM_EXT_HOST_PTR_QCOM,
        &ext_rw as *const _ as *mut c_void,
        size,
    );

    eprintln!("\n=== Wider sweep 0x40C8 .. 0x40FF (alloc_type) ===");
    for v in 0x40C8u32..=0x40FFu32 {
        let ext = ExtHostPtr {
            allocation_type: v,
            host_cache_policy: 0x40A9,
        };
        let mut errcode: i32 = 0;
        let raw = unsafe {
            ocl::ffi::clCreateBuffer(
                ctx_ptr,
                (CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM) as u64,
                size,
                &ext as *const _ as *mut c_void,
                &mut errcode,
            )
        };
        if errcode != -30 || !raw.is_null() {
            eprintln!(
                "  alloc_type=0x{:X}: errcode={}, ptr={}",
                v,
                errcode,
                !raw.is_null()
            );
            if !raw.is_null() {
                unsafe { ocl::ffi::clReleaseMemObject(raw) };
            }
        }
    }

    eprintln!("\n=== DMA-BUF heap test (/dev/dma_heap/system) ===");
    use std::os::unix::io::RawFd;
    #[repr(C)]
    struct DmaHeapAllocationData {
        len: u64,
        fd: u32,
        fd_flags: u32,
        heap_flags: u64,
    }
    const DMA_HEAP_IOCTL_MAGIC: u8 = b'H';
    const DMA_HEAP_IOCTL_ALLOC: u64 = ((2u64 << 30) /* IOC_READ|WRITE */
                                       | ((DMA_HEAP_IOCTL_MAGIC as u64) << 8)
                                       | 0x00
                                       | ((std::mem::size_of::<DmaHeapAllocationData>() as u64) << 16));
    // Note: above is approximate; correct ioctl encoding for Linux is:
    //   _IOWR('H', 0, struct dma_heap_allocation_data)
    // = ((1+2)<<30) | (sizeof << 16) | ('H' << 8) | 0
    let ioctl_alloc: u64 = (3u64 << 30)
        | ((std::mem::size_of::<DmaHeapAllocationData>() as u64) << 16)
        | ((b'H' as u64) << 8);
    eprintln!("DMA_HEAP_IOCTL_ALLOC = 0x{:X}", ioctl_alloc);

    let heap_fd: RawFd = unsafe { libc::open(c"/dev/dma_heap/system".as_ptr(), libc::O_RDONLY) };
    if heap_fd < 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        eprintln!("  open(/dev/dma_heap/system) failed: errno={}", errno);
    } else {
        eprintln!("  /dev/dma_heap/system opened, fd={}", heap_fd);
        let mut data = DmaHeapAllocationData {
            len: 4096,
            fd: 0,
            fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            heap_flags: 0,
        };
        unsafe extern "C" {
            fn ioctl(fd: libc::c_int, request: libc::c_ulong, ...) -> libc::c_int;
        }
        let rc = unsafe { ioctl(heap_fd, ioctl_alloc as libc::c_ulong, &mut data) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            eprintln!("  ioctl ALLOC failed: rc={}, errno={}", rc, errno);
        } else {
            let dmabuf_fd = data.fd as i32;
            eprintln!("  DMA-BUF allocated: fd={}, len={}", dmabuf_fd, data.len);

            // mmap the DMA-BUF for CPU access
            let mmap_ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    dmabuf_fd,
                    0,
                )
            };
            if mmap_ptr == libc::MAP_FAILED {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                eprintln!("  mmap failed: errno={}", errno);
            } else {
                eprintln!("  DMA-BUF mmap ok at {:p}", mmap_ptr);

                // cl_qcom_dmabuf_host_ptr struct (canonical layout)
                #[repr(C)]
                struct ExtHostPtrDmabuf {
                    ext: ExtHostPtr,
                    dmabuf_fd: i32,
                    _pad: u32,
                    dmabuf_hostptr: *mut c_void,
                }

                eprintln!("  Probing alloc_type values for DMA-BUF binding:");
                for v in [
                    0x40A8u32, 0x40A9, 0x40C6, 0x40C7, 0x40C8, 0x40C9, 0x40CA, 0x40CB, 0x40CC,
                    0x40CD, 0x40CE, 0x40CF, 0x40D0, 0x40D1, 0x40D2, 0x40D3, 0x40D4, 0x40D5, 0x40D6,
                    0x40D7, 0x40D8, 0x40D9, 0x40DA,
                ] {
                    let ext = ExtHostPtrDmabuf {
                        ext: ExtHostPtr {
                            allocation_type: v,
                            host_cache_policy: 0x40A9,
                        },
                        dmabuf_fd,
                        _pad: 0,
                        dmabuf_hostptr: mmap_ptr,
                    };
                    let mut errcode: i32 = 0;
                    let raw = unsafe {
                        ocl::ffi::clCreateBuffer(
                            ctx_ptr,
                            (CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM) as u64,
                            4096,
                            &ext as *const _ as *mut c_void,
                            &mut errcode,
                        )
                    };
                    if errcode != -30 {
                        eprintln!(
                            "    alloc_type=0x{:X}: errcode={}, ptr={}",
                            v,
                            errcode,
                            !raw.is_null()
                        );
                        if !raw.is_null() {
                            unsafe { ocl::ffi::clReleaseMemObject(raw) };
                        }
                    }
                }

                eprintln!(
                    "\n  === clCreateBufferWithProperties + cl_khr_external_memory_dma_buf ==="
                );
                // CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR = 0x2067
                // Properties: pairs of (key, value), 0-terminated.
                let props: [u64; 3] = [0x2067, dmabuf_fd as u64, 0];

                unsafe extern "C" {
                    fn clCreateBufferWithProperties(
                        context: *mut c_void,
                        properties: *const u64,
                        flags: u64,
                        size: usize,
                        host_ptr: *mut c_void,
                        errcode_ret: *mut i32,
                    ) -> *mut c_void;
                }

                for (flags, label) in [
                    (CL_MEM_READ_ONLY, "READ_ONLY"),
                    (CL_MEM_READ_WRITE, "READ_WRITE"),
                    (
                        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                        "READ_ONLY|USE_HOST_PTR",
                    ),
                ] {
                    let mut errcode: i32 = 0;
                    let host_ptr_arg = if flags & CL_MEM_USE_HOST_PTR != 0 {
                        mmap_ptr
                    } else {
                        ptr::null_mut()
                    };
                    let raw = unsafe {
                        clCreateBufferWithProperties(
                            ctx_ptr as *mut _,
                            props.as_ptr(),
                            flags,
                            4096,
                            host_ptr_arg,
                            &mut errcode,
                        )
                    };
                    eprintln!(
                        "    flags={}: errcode={}, ptr={}",
                        label,
                        errcode,
                        !raw.is_null()
                    );
                    if !raw.is_null() {
                        unsafe { ocl::ffi::clReleaseMemObject(raw as *mut _) };
                    }
                }

                unsafe { libc::munmap(mmap_ptr, 4096) };
                unsafe { libc::close(dmabuf_fd) };
            }
        }
        unsafe { libc::close(heap_fd) };
    }

    Ok(())
}

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("Requires opencl feature.");
}
