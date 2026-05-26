//! microbench_htp_matmul_correctness — Phase 32b-1: R1 + R2
//!
//! 목적: HTP의 prebuilt MatMul op를 활용해 정확성 검증.
//! 1B 모델 FFN gate matmul scale (1×K, K×N → 1×N)으로 측정.
//!
//! Pass-gate (R1+R2):
//!   - prebuilt MatMul op 호출 OK
//!   - max abs error < 1e-3 (FP32)
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_matmul_correctness
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_matmul_correctness [K] [N]`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_matmul_correctness requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

// HTP-specific device custom config (from third_party/qnn_sdk_2.33/include/QNN/HTP/QnnHtpDevice.h).
// bindgen은 QnnInterface.h 만 처리하므로 HTP 헤더 타입은 여기에 직접 정의한다.
// Q-2 dry-run gate: stock S25 에서 unsigned-PD 명시가 없으면 contextCreate err=0x36b1.
// Q-2.1 추가: PlatformInfo block (Executorch HtpDevicePlatformInfoConfig.cpp:13-58 차용).
#[cfg(feature = "qnn")]
#[allow(non_snake_case, non_camel_case_types, dead_code)]
mod qnn_htp {
    pub const QNN_HTP_DEVICE_ARCH_V79: u32 = 79;

    pub const QNN_HTP_DEVICE_CONFIG_OPTION_ARCH: u32 = 1;
    pub const QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD: u32 = 2;

    // QnnHtpDevice_DeviceType_t (QnnHtpDevice.h:84-87)
    pub const QNN_HTP_DEVICE_TYPE_ON_CHIP: u32 = 0;

    // SocInfo: SM8750 (Snapdragon 8 Elite Gen ?). Executorch utils.py:1278-1315 mapping.
    pub const QNN_SOC_MODEL_SM8750: u32 = 57;

    // V79 VTCM (per Executorch SocInfo, V79 = 8 MB)
    pub const QNN_HTP_V79_VTCM_SIZE_MB: usize = 8;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct QnnHtpDevice_Minimum_Arch_t {
        pub deviceId: u32,
        pub arch: u32, // QnnHtpDevice_Arch_t
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct QnnHtpDevice_UseSignedProcessDomain_t {
        pub deviceId: u32,
        pub useSignedProcessDomain: bool,
    }

    // SDK header에 있는 anonymous union의 최대 크기를 커버하도록 8바이트 align +
    // 충분한 storage (Minimum_Arch_t 가 8B, UseSignedProcessDomain_t 가 8B, socModel u32).
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub union QnnHtpDevice_CustomConfig_Union_t {
        pub socModel: u32,
        pub arch: QnnHtpDevice_Minimum_Arch_t,
        pub useSignedProcessDomain: QnnHtpDevice_UseSignedProcessDomain_t,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct QnnHtpDevice_CustomConfig_t {
        pub option: u32, // QnnHtpDevice_ConfigOption_t
        pub payload: QnnHtpDevice_CustomConfig_Union_t,
    }

    // ── PlatformInfo block (Executorch HtpDevicePlatformInfoConfig.cpp 차용) ──
    // QnnHtpDevice_OnChipDeviceInfoExtension_t (QnnHtpDevice.h:113-120)
    // 주의: `vtcmSize: size_t` 가 첫 필드. ARM64 에서 size_t=u64.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct QnnHtpDevice_OnChipDeviceInfoExtension_t {
        pub vtcmSize: usize,
        pub socModel: u32,
        pub signedPdSupport: bool,
        pub dlbcSupport: bool,
        pub arch: u32, // QnnHtpDevice_Arch_t
    }

    // QnnHtpDevice_DeviceInfoExtension_t (QnnHtpDevice.h:126-131)
    // bindgen 의 _QnnDevice_DeviceInfoExtension_t 는 opaque struct 이지만,
    // 실제 storage 는 HTP backend 가 본 struct 로 캐스트한다.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct QnnHtpDevice_DeviceInfoExtension_t {
        pub devType: u32, // QnnHtpDevice_DeviceType_t
        pub onChipDevice: QnnHtpDevice_OnChipDeviceInfoExtension_t,
    }
}

// Q-2.1 patch: RTLD_GLOBAL dlopen (Executorch QnnImplementation.cpp:60-80 차용).
// libloading 0.8 default 는 RTLD_LAZY | RTLD_LOCAL (Adreno OpenCL 과 동일 default).
// HTP backend 는 inter-lib symbol resolution (Skel ↔ Stub ↔ Calculator) 가 필요할 수 있어
// RTLD_GLOBAL 명시. RTLD_NOW 도 함께 (lazy binding 부작용 회피).
#[cfg(feature = "qnn")]
unsafe fn dlopen_global(path: &str) -> anyhow::Result<libloading::Library> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
    // SAFETY: caller responsible for path validity. Library 는 process lifetime 까지 leak 의도.
    let lib = unsafe { UnixLibrary::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }
        .map_err(|e| anyhow::anyhow!("dlopen RTLD_GLOBAL fail {}: {}", path, e))?;
    Ok(libloading::Library::from(lib))
}

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::Symbol;
    use std::ffi::CString;
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;
    use qnn_htp::*;

    // FFN gate scale: Qwen2.5-1.5B has dim=1536, ffn_dim=8960. We use simpler 1024×4096.
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let m: usize = 1; // single-token decode

    println!("=== microbench_htp_matmul_correctness (Phase 32b-1) ===\n");
    println!(
        "MatMul: A[{}, {}] × B[{}, {}] = C[{}, {}]",
        m, k, k, n, m, n
    );

    // ── HTP setup ──
    // Q-2.1: RTLD_GLOBAL 로 preload (Executorch 동일 패턴).
    // 순서: System → Prepare → V79 (skel/stub/calculator) → Htp main.
    // 각 lib 의 export symbol 이 후속 lib resolve 단계에서 가시화되어야 함.
    let preload_libs = [
        "/data/local/tmp/qnn/libQnnSystem.so",
        "/data/local/tmp/qnn/libQnnHtpPrepare.so",
        "/data/local/tmp/qnn/libQnnHtpV79Stub.so",
        "/data/local/tmp/qnn/libQnnHtpV79.so",
        "/data/local/tmp/qnn/libQnnHtpV79Skel.so",
        "/data/local/tmp/qnn/libQnnHtpV79CalculatorStub.so",
    ];
    let mut _preloaded: Vec<libloading::Library> = Vec::new();
    for p in preload_libs.iter() {
        match unsafe { dlopen_global(p) } {
            Ok(l) => {
                println!("preload RTLD_GLOBAL OK: {}", p);
                _preloaded.push(l);
            }
            Err(e) => {
                println!("preload RTLD_GLOBAL skip: {} ({})", p, e);
            }
        }
    }
    let lib = unsafe { dlopen_global("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { dlopen_global("libQnnHtp.so") })?;
    println!("dlopen libQnnHtp.so RTLD_GLOBAL OK");
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> = unsafe { lib.get(b"QnnInterface_getProviders\0")? };
    let mut providers: *mut *const QnnInterface_t = ptr::null_mut();
    let mut num: c_uint = 0;
    let err = unsafe { get_providers(&mut providers, &mut num) };
    anyhow::ensure!(
        err == 0 && num > 0,
        "QnnInterface_getProviders err=0x{:x}",
        err
    );
    let v = unsafe { (**providers).__bindgen_anon_1.v2_25 };

    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut backend) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);

    // ── Q-2 dry-run gate: HTP device with custom config (unsigned PD + V79 arch) ──
    // stock S25 (R3CY408S5SB) 는 deviceCreate 없이 contextCreate(NULL device) 호출 시
    // contextCreate err=0x36b1 (AEE_ENOSUCHMOD): DSP가 application PD에 Skel.so publish 실패.
    // Executorch HtpDevice.cpp:304~321 패턴: arch=V79 + useSignedProcessDomain=false.
    let mut htp_cfg_arch = QnnHtpDevice_CustomConfig_t {
        option: QNN_HTP_DEVICE_CONFIG_OPTION_ARCH,
        payload: QnnHtpDevice_CustomConfig_Union_t {
            arch: QnnHtpDevice_Minimum_Arch_t {
                deviceId: 0,
                arch: QNN_HTP_DEVICE_ARCH_V79,
            },
        },
    };
    // LLMRS_HTP_SIGNED_PD=1 → signed PD path (stock device 차단 가설 검증용)
    let use_signed = std::env::var("LLMRS_HTP_SIGNED_PD").is_ok();
    let mut htp_cfg_unsigned = QnnHtpDevice_CustomConfig_t {
        option: QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD,
        payload: QnnHtpDevice_CustomConfig_Union_t {
            useSignedProcessDomain: QnnHtpDevice_UseSignedProcessDomain_t {
                deviceId: 0,
                useSignedProcessDomain: use_signed,
            },
        },
    };
    println!(
        "HTP custom config: arch=V79, useSignedProcessDomain={}",
        use_signed
    );
    let dev_cfg_arch = QnnDevice_Config_t {
        option: QnnDevice_ConfigOption_t_QNN_DEVICE_CONFIG_OPTION_CUSTOM,
        __bindgen_anon_1: QnnDevice_Config_t__bindgen_ty_1 {
            customConfig: &mut htp_cfg_arch as *mut _ as *mut std::os::raw::c_void,
        },
    };
    let dev_cfg_unsigned = QnnDevice_Config_t {
        option: QnnDevice_ConfigOption_t_QNN_DEVICE_CONFIG_OPTION_CUSTOM,
        __bindgen_anon_1: QnnDevice_Config_t__bindgen_ty_1 {
            customConfig: &mut htp_cfg_unsigned as *mut _ as *mut std::os::raw::c_void,
        },
    };
    // ── Q-2.1 patch: PlatformInfo block (Executorch HtpDevicePlatformInfoConfig.cpp:13-58) ──
    // socModel/vtcm/arch/signedPdSupport/dlbcSupport 을 device-side dispatch 에 명시 전달.
    // CustomConfig (ARCH + SIGNEDPD) 만으로는 deviceCreate err=0x36b1 (AEE_ENOSUCHMOD) 발생.
    let mut on_chip = QnnHtpDevice_OnChipDeviceInfoExtension_t {
        vtcmSize: QNN_HTP_V79_VTCM_SIZE_MB,
        socModel: QNN_SOC_MODEL_SM8750,
        signedPdSupport: use_signed,
        dlbcSupport: true,
        arch: QNN_HTP_DEVICE_ARCH_V79,
    };
    let mut dev_info_ext = QnnHtpDevice_DeviceInfoExtension_t {
        devType: QNN_HTP_DEVICE_TYPE_ON_CHIP,
        onChipDevice: on_chip,
    };
    // QnnDevice_CoreInfo_t: version=1 + v1 { coreId=0, coreType=0, ext=NULL }.
    let mut core_info = QnnDevice_CoreInfo_t {
        version: QnnDevice_CoreInfoVersion_t_QNN_DEVICE_CORE_INFO_VERSION_1,
        __bindgen_anon_1: QnnDevice_CoreInfo_t__bindgen_ty_1 {
            v1: QnnDevice_CoreInfoV1_t {
                coreId: 0,
                coreType: 0,
                coreInfoExtension: ptr::null_mut(),
            },
        },
    };
    // QnnDevice_HardwareDeviceInfo_t: v1 { deviceId=0, deviceType=0, numCores=1, cores=&core_info,
    //                                       deviceInfoExtension=&dev_info_ext (HTP onChip block) }.
    let mut hw_dev_info = QnnDevice_HardwareDeviceInfo_t {
        version: QnnDevice_HardwareDeviceInfoVersion_t_QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1,
        __bindgen_anon_1: QnnDevice_HardwareDeviceInfo_t__bindgen_ty_1 {
            v1: QnnDevice_HardwareDeviceInfoV1_t {
                deviceId: 0,
                deviceType: 0,
                numCores: 1,
                cores: &mut core_info as *mut QnnDevice_CoreInfo_t,
                // HTP backend가 본 opaque 포인터를 QnnHtpDevice_DeviceInfoExtension_t* 로 캐스트.
                deviceInfoExtension: &mut dev_info_ext as *mut _
                    as *mut _QnnDevice_DeviceInfoExtension_t,
            },
        },
    };
    let mut platform_info = QnnDevice_PlatformInfo_t {
        version: QnnDevice_PlatformInfoVersion_t_QNN_DEVICE_PLATFORM_INFO_VERSION_1,
        __bindgen_anon_1: QnnDevice_PlatformInfo_t__bindgen_ty_1 {
            v1: QnnDevice_PlatformInfoV1_t {
                numHwDevices: 1,
                hwDevices: &mut hw_dev_info as *mut QnnDevice_HardwareDeviceInfo_t,
            },
        },
    };
    let dev_cfg_platform = QnnDevice_Config_t {
        option: QnnDevice_ConfigOption_t_QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO,
        __bindgen_anon_1: QnnDevice_Config_t__bindgen_ty_1 {
            hardwareInfo: &mut platform_info as *mut QnnDevice_PlatformInfo_t,
        },
    };

    let dev_cfg_arch_ptr: *const QnnDevice_Config_t = &dev_cfg_arch;
    let dev_cfg_unsigned_ptr: *const QnnDevice_Config_t = &dev_cfg_unsigned;
    let dev_cfg_platform_ptr: *const QnnDevice_Config_t = &dev_cfg_platform;
    // QNN convention: NULL-terminated array of *const QnnDevice_Config_t.
    // 순서: ARCH → SIGNEDPD → PLATFORM_INFO → NULL.
    // PlatformInfo 가 마지막 — backend 가 custom config 로 baseline 설정 후 platform block 적용.
    let mut dev_cfg_list: [*const QnnDevice_Config_t; 4] = [
        dev_cfg_arch_ptr,
        dev_cfg_unsigned_ptr,
        dev_cfg_platform_ptr,
        ptr::null(),
    ];
    // suppress unused-mut warnings on the helper struct held by reference.
    let _ = &mut on_chip;
    let mut device: Qnn_DeviceHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.deviceCreate.unwrap())(ptr::null_mut(), dev_cfg_list.as_mut_ptr(), &mut device)
    };
    anyhow::ensure!(err == 0, "deviceCreate err=0x{:x}", err);
    println!("deviceCreate (V79 + unsigned PD + PlatformInfo): OK");

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(backend, device, ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);
    println!("contextCreate (with device): OK");

    let graph_name = CString::new("htp_matmul").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    // ── Tensors: A [m, k], B [k, n], C [m, n] ──
    let dims_a: Vec<u32> = vec![m as u32, k as u32];
    let dims_b: Vec<u32> = vec![k as u32, n as u32];
    let dims_c: Vec<u32> = vec![m as u32, n as u32];

    let mk_v1_raw = |name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]| -> Qnn_TensorV1_t {
        Qnn_TensorV1_t {
            id: 0,
            name: name.as_ptr(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            quantizeParams: Qnn_QuantizeParams_t {
                encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
                quantizationEncoding:
                    Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
                __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
                    scaleOffsetEncoding: Qnn_ScaleOffset_t {
                        scale: 0.0,
                        offset: 0,
                    },
                },
            },
            rank: dims.len() as u32,
            dimensions: dims.as_ptr() as *mut u32,
            memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                clientBuf: Qnn_ClientBuffer_t {
                    data: ptr::null_mut(),
                    dataSize: 0,
                },
            },
        }
    };
    let name_a = CString::new("A").unwrap();
    let name_b = CString::new("B").unwrap();
    let name_c = CString::new("C").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut t_a), ("B", &mut t_b), ("C", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreateGraphTensor({}) err=0x{:x}", l, err);
    }

    // ── MatMul op ──
    let op_name = CString::new("matmul0").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("MatMul").unwrap();
    let mut inputs = [t_a, t_b];
    let mut outputs = [t_c];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: inputs.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
    anyhow::ensure!(err == 0, "graphAddNode(MatMul) err=0x{:x}", err);
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
    println!("HTP graph (MatMul) finalize: OK");

    // ── Host data ──
    // A: small values to keep magnitudes manageable
    let host_a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0 - 0.5).collect();
    let host_b: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.0007 + 0.13) % 1.0 - 0.5)
        .collect();
    let mut host_c: Vec<f32> = vec![0.0; m * n];

    unsafe {
        inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_a.as_ptr() as *mut _,
            dataSize: (host_a.len() * 4) as u32,
        };
        inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_b.as_ptr() as *mut _,
            dataSize: (host_b.len() * 4) as u32,
        };
        outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_c.as_mut_ptr() as *mut _,
            dataSize: (host_c.len() * 4) as u32,
        };
    }

    // Warmup + measure
    let mut samples = Vec::new();
    for _ in 0..3 {
        let _ = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                inputs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }
    for _ in 0..10 {
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                inputs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "graphExecute mean: {:.2} ms over {} iters",
        mean,
        samples.len()
    );

    // ── CPU NEON ground truth ──
    println!("\nComputing CPU reference...");
    let t0 = Instant::now();
    let mut ref_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += host_a[mi * k + ki] * host_b[ki * n + ni];
            }
            ref_c[mi * n + ni] = acc;
        }
    }
    println!(
        "CPU reference: {:.2} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ── Compare ──
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut idx_max = 0;
    for i in 0..(m * n) {
        let abs = (host_c[i] - ref_c[i]).abs();
        let rel = abs / ref_c[i].abs().max(1.0);
        if abs > max_abs_err {
            max_abs_err = abs;
            idx_max = i;
        }
        if rel > max_rel_err {
            max_rel_err = rel;
        }
    }
    println!(
        "\nMax abs err: {:.6e} (at idx {}: HTP={:.6} ref={:.6})",
        max_abs_err, idx_max, host_c[idx_max], ref_c[idx_max]
    );
    println!("Max rel err: {:.6e}", max_rel_err);

    let pass_strict = max_abs_err < 1e-3;
    let pass_acceptable = max_abs_err < 1e-2;
    println!(
        "\nR1: prebuilt MatMul: {}",
        if mean > 0.0 { "✓ exists" } else { "✗" }
    );
    println!(
        "R2: numerical correctness: {}",
        if pass_strict {
            "✓ PASS (<1e-3)"
        } else if pass_acceptable {
            "△ ACCEPTABLE (<1e-2)"
        } else {
            "✗ FAIL"
        }
    );

    if pass_strict || pass_acceptable {
        println!("\n=> Phase 32b-1 PASS-gate cleared. Proceed to Phase 32b-2 (R3 concurrent).");
    } else {
        println!("\n=> Phase 32b-1 BLOCKED. Investigate HTP precision option.");
    }

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.deviceFree.unwrap())(device);
        let _ = (v.backendFree.unwrap())(backend);
    }
    if pass_acceptable {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
