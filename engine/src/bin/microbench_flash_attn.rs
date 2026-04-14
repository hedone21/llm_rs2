//! microbench_flash_attn — flash_attn_f32_f16_q1 단독 격리 측정
//!
//! 목적: long-context decode 갭(llm_rs2 12.45 vs llama.cpp 5.70 μs/n_kv)이
//! KV stride layout 차이에서 오는지 격리 검증.
//!
//! 두 layout을 같은 production 커널에 던져서 attention μs 기울기 비교:
//!   - HeadMajor: `[kv_h, cap, dk]` (현재 production)
//!     k_nb1 = dk*2 = 256B   (한 head 내 pos 연속)
//!     k_nb2 = cap*dk*2      (head 사이 큰 stride)
//!   - PosMajor:  `[cap, kv_h, dk]` (llama.cpp permute view 등가)
//!     k_nb1 = kv_h*dk*2 = 512B  (pos 사이에 다른 kv_head data interleave)
//!     k_nb2 = dk*2 = 256B       (head 사이 작은 stride)
//!
//! Phase A에서 K-loop 바이트 동일 결론이 났으므로, 두 layout이 의미 있는
//! 차이를 보이면 갭 원인은 **host가 주는 stride argument를 통한
//! L1/L2/prefetcher behavior 차이**로 확정된다.
//!
//! 설계 원칙:
//!   - production `flash_attn_f32_f16.cl` 무수정 사용
//!   - production 빌드 옵션 (`-DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32`
//!     + fast-math + CL2.0)
//!   - 단일 dispatch만 측정 → thermal 누적 제로, 506-dispatch decode 노이즈 배제
//!   - CL_QUEUE_PROFILING_ENABLE event 측정 (production --profile-events 동등)
//!   - mask=NULL, sinks=NULL (decode dispatch도 동일)
//!
//! Qwen 2.5-1.5B 파라미터: n_h_q=12, n_kv_h=2, dk=128, gqa=6.

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_flash_attn requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::{ArgVal, Event, Mem, ProfilingInfo};
    use ocl::flags;
    use ocl::{Context, Device, Platform, Program, Queue};

    // === 모델 파라미터 (Qwen 2.5-1.5B) ===
    const N_H_Q: usize = 12;
    const N_H_KV: usize = 2;
    const DK: usize = 128;
    const DV: usize = 128;
    const CAP: usize = 8192; // 최대 n_kv 4472 + 마진
    const Q1_WG_SIZE: usize = 64;

    // === 측정 파라미터 ===
    const N_KV_VALUES: &[i32] = &[258, 1025, 2047, 4472];
    const WARMUP_ITERS: usize = 5;
    const MEASURE_ITERS: usize = 30;

    // === 1. Platform / Device / Context / Profiling Queue ===
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
    let device_name = device.name().unwrap_or_else(|_| "unknown".into());
    println!("# microbench_flash_attn — KV layout slope isolation");
    println!("# Platform: {}", platform_name);
    println!("# Device:   {}", device_name);
    println!(
        "# Model: n_h_q={} n_h_kv={} dk={} dv={} cap={} Q1_WG_SIZE={}",
        N_H_Q, N_H_KV, DK, DV, CAP, Q1_WG_SIZE
    );

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // CL_QUEUE_PROFILING_ENABLE 필수 (event μs 추출)
    let queue = Queue::new(&context, device, Some(flags::QUEUE_PROFILING_ENABLE))?;

    // === 2. Build production flash_attn_f32_f16.cl with DK=128 macros ===
    let flash_attn_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
    let cl_c_version_str = device
        .info(ocl::core::DeviceInfo::OpenclCVersion)
        .map(|v| v.to_string())
        .unwrap_or_default();
    let supports_cl_c_2 = cl_c_version_str
        .split_whitespace()
        .nth(2)
        .and_then(|ver| {
            let mut parts = ver.split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next()?.parse().ok()?;
            Some((major, minor) >= (2, 0))
        })
        .unwrap_or(false);
    let fast_math = "-cl-mad-enable -cl-unsafe-math-optimizations \
                     -cl-finite-math-only -cl-fast-relaxed-math";
    let cl_opts = if supports_cl_c_2 {
        format!("-cl-std=CL2.0 {}", fast_math)
    } else {
        fast_math.to_string()
    };
    let dk128_defines = format!("-DDK={} -DDV={} -DBLOCK_M=32 -DBLOCK_N=32", DK, DV);
    let full_opts = format!("{} {}", dk128_defines, cl_opts);
    println!("# Build opts: {}", full_opts);

    let program = Program::builder()
        .devices(device)
        .src(flash_attn_src)
        .cmplr_opt(&full_opts)
        .build(&context)?;
    let kernel = ocl::core::create_kernel(&program, "flash_attn_f32_f16_q1")?;

    // === 3. Buffer allocation ===
    // KV 버퍼 크기는 layout과 무관 (총 elements = kv_h * cap * dk).
    // F16 데이터 = 2 byte/elem. 0.125 (F16 = 0x3000) 로 채워서 NaN 회피.
    let kv_elems = N_H_KV * CAP * DK;
    let kv_filler: u16 = 0x3000; // F16 0.125
    let kv_host: Vec<u16> = vec![kv_filler; kv_elems];

    let alloc_kv = || -> anyhow::Result<Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, u16>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                kv_elems,
                None,
            )?
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &buf,
                true,
                0,
                std::slice::from_raw_parts(kv_host.as_ptr() as *const u8, kv_elems * 2),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(buf)
    };
    let k_buf = alloc_kv()?;
    let v_buf = alloc_kv()?;

    // Q: F32 [1, 1, n_h_q, dk]. 0.125 로 채움.
    let q_elems = N_H_Q * DK;
    let q_host: Vec<f32> = vec![0.125_f32; q_elems];
    let q_buf = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            q_elems,
            None,
        )?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(
            &queue,
            &q_buf,
            true,
            0,
            std::slice::from_raw_parts(q_host.as_ptr() as *const u8, q_elems * 4),
            None::<&Event>,
            None::<&mut Event>,
        )?;
    }

    // O: F32 [1, 1, n_h_q, dv]. 결과 무관.
    let o_elems = N_H_Q * DV;
    let o_buf = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            o_elems,
            None,
        )?
    };

    // === 4. Stride 사전 계산 ===
    // Q strides (F32 [1,1,n_h_q,dk])
    let q_nb1 = (N_H_Q * DK * 4) as u64;
    let q_nb2 = (DK * 4) as u64;
    let q_nb3 = q_nb1;
    // O strides
    let o_nb1 = (DV * 4) as u64;
    let o_nb2 = (N_H_Q * DV * 4) as u64;
    let o_nb3 = o_nb2;

    let kv_elem: u64 = 2;
    // HeadMajor: [kv_h, cap, dk]
    let head_major = (
        (DK as u64) * kv_elem,                // nb1 = dk*2 = 256
        (CAP * DK) as u64 * kv_elem,          // nb2 = cap*dk*2
        (N_H_KV * CAP * DK) as u64 * kv_elem, // nb3
    );
    // PosMajor: [cap, kv_h, dk]  (llama.cpp permute view 등가)
    let pos_major = (
        (N_H_KV * DK) as u64 * kv_elem,       // nb1 = kv_h*dk*2 = 512
        (DK as u64) * kv_elem,                // nb2 = dk*2 = 256
        (N_H_KV * CAP * DK) as u64 * kv_elem, // nb3 (동일)
    );

    // === 5. Static args (layout-independent) — set once ===
    let scale = 1.0f32 / (DK as f32).sqrt();
    let n_q = 1i32;
    let is_causal = 0i32;
    let n_head = N_H_Q as i32;
    let n_head_kv = N_H_KV as i32;
    let max_bias = 0.0f32;
    let m0 = 0.0f32;
    let m1 = 0.0f32;
    let n_head_log2 = 0i32;
    let logit_softcap = 0.0f32;
    let zero_u64 = 0u64;
    let zero_i32 = 0i32;

    {
        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&k_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::mem(&v_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::mem(&o_buf))?;
        ocl::core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&n_q))?;
        // arg 10 = n_kv (per-iter, set in loop)
        ocl::core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&is_causal))?;
        ocl::core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&n_head))?;
        ocl::core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&q_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&q_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 15, ArgVal::scalar(&q_nb3))?;
        // args 16-21 = K/V strides (per-layout, set per outer)
        ocl::core::set_kernel_arg(&kernel, 22, ArgVal::scalar(&o_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 23, ArgVal::scalar(&o_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 24, ArgVal::scalar(&o_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 25, ArgVal::scalar(&max_bias))?;
        ocl::core::set_kernel_arg(&kernel, 26, ArgVal::scalar(&m0))?;
        ocl::core::set_kernel_arg(&kernel, 27, ArgVal::scalar(&m1))?;
        ocl::core::set_kernel_arg(&kernel, 28, ArgVal::scalar(&n_head_log2))?;
        ocl::core::set_kernel_arg(&kernel, 29, ArgVal::scalar(&logit_softcap))?;
        ocl::core::set_kernel_arg(&kernel, 30, ArgVal::scalar(&n_head_kv))?;
        ocl::core::set_kernel_arg(&kernel, 31, ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 32, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 33, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 34, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 35, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 36, ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 37, ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 38, ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 39, ArgVal::scalar(&zero_u64))?;
    }

    // === 6. Measurement matrix ===
    let layouts: &[(&str, (u64, u64, u64))] = &[("HeadMajor", head_major), ("PosMajor", pos_major)];
    let gws = [Q1_WG_SIZE, N_H_Q, 1];
    let lws: Option<[usize; 3]> = Some([Q1_WG_SIZE, 1, 1]);

    println!();
    println!("layout,n_kv,median_us,mean_us,min_us,max_us,iters");

    let mut slope_data: Vec<(String, Vec<(i32, f64)>)> = Vec::new();

    for (layout_name, (k_nb1, k_nb2, k_nb3)) in layouts {
        // Set K/V strides for this layout
        ocl::core::set_kernel_arg(&kernel, 16, ArgVal::scalar(k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 17, ArgVal::scalar(k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 18, ArgVal::scalar(k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 19, ArgVal::scalar(k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 20, ArgVal::scalar(k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 21, ArgVal::scalar(k_nb3))?;

        let mut points: Vec<(i32, f64)> = Vec::new();

        for &n_kv in N_KV_VALUES {
            ocl::core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&n_kv))?;

            // Warmup
            for _ in 0..WARMUP_ITERS {
                unsafe {
                    ocl::core::enqueue_kernel(
                        &queue,
                        &kernel,
                        2,
                        None,
                        &gws,
                        lws,
                        None::<&Event>,
                        None::<&mut Event>,
                    )?;
                }
            }
            ocl::core::finish(&queue)?;

            // Measurement: per-iter event 측정
            let mut samples_us: Vec<f64> = Vec::with_capacity(MEASURE_ITERS);
            for _ in 0..MEASURE_ITERS {
                let mut ev: Event = Event::null();
                unsafe {
                    ocl::core::enqueue_kernel(
                        &queue,
                        &kernel,
                        2,
                        None,
                        &gws,
                        lws,
                        None::<&Event>,
                        Some(&mut ev),
                    )?;
                }
                ocl::core::finish(&queue)?;
                let start_ns =
                    ocl::core::get_event_profiling_info(&ev, ProfilingInfo::Start)?.time()?;
                let end_ns =
                    ocl::core::get_event_profiling_info(&ev, ProfilingInfo::End)?.time()?;
                let delta_ns = end_ns.saturating_sub(start_ns) as f64;
                samples_us.push(delta_ns / 1000.0);
            }

            // Stats
            let mut sorted = samples_us.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            let mean: f64 = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
            let min = sorted[0];
            let max = *sorted.last().unwrap();

            println!(
                "{},{},{:.2},{:.2},{:.2},{:.2},{}",
                layout_name,
                n_kv,
                median,
                mean,
                min,
                max,
                samples_us.len()
            );
            points.push((n_kv, median));
        }

        slope_data.push((layout_name.to_string(), points));
    }

    // === 7. Slope (median μs vs n_kv, simple least squares) ===
    println!();
    println!("# Slope (μs / n_kv) via least squares on median samples");
    for (name, pts) in &slope_data {
        let n = pts.len() as f64;
        let sx: f64 = pts.iter().map(|(x, _)| *x as f64).sum();
        let sy: f64 = pts.iter().map(|(_, y)| *y).sum();
        let sxx: f64 = pts.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|(x, y)| (*x as f64) * *y).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;
        println!(
            "# {:10}  slope = {:.4} μs/n_kv   intercept = {:.2} μs",
            name, slope, intercept
        );
    }

    // === 8. Comparison ===
    if slope_data.len() == 2 {
        let s_head = {
            let pts = &slope_data[0].1;
            let n = pts.len() as f64;
            let sx: f64 = pts.iter().map(|(x, _)| *x as f64).sum();
            let sy: f64 = pts.iter().map(|(_, y)| *y).sum();
            let sxx: f64 = pts.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
            let sxy: f64 = pts.iter().map(|(x, y)| (*x as f64) * *y).sum();
            (n * sxy - sx * sy) / (n * sxx - sx * sx)
        };
        let s_pos = {
            let pts = &slope_data[1].1;
            let n = pts.len() as f64;
            let sx: f64 = pts.iter().map(|(x, _)| *x as f64).sum();
            let sy: f64 = pts.iter().map(|(_, y)| *y).sum();
            let sxx: f64 = pts.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
            let sxy: f64 = pts.iter().map(|(x, y)| (*x as f64) * *y).sum();
            (n * sxy - sx * sy) / (n * sxx - sx * sx)
        };
        let ratio = s_head / s_pos;
        println!(
            "# HeadMajor / PosMajor slope ratio = {:.3}x ({})",
            ratio,
            if ratio > 1.10 {
                "PosMajor wins → KV stride 가설 SUPPORT"
            } else if ratio < 0.90 {
                "HeadMajor wins → 현재 layout 유지가 정답"
            } else {
                "차이 < 10% → KV stride 가설 REJECT (다른 곳 찾아야)"
            }
        );
    }

    Ok(())
}
