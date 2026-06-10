//! microbench_score_readback — 커스텀 fold KV-eviction plugin 이 OpenCL 백엔드에서
//! per-step attention score(`score_buf`)를 GPU→CPU readback 하는 비용(D2H DMA) 측정.
//!
//! 배경 (observe-hook PoC 후속): 내장 H2O 는 `score_buf` 를 GPU 에 두고 GPU reduce
//! (`GpuScoreAccumulator`)로 누적 → CPU 는 evict 시점에 최종 importance 만 1회 `sync_to_cpu`.
//! 그러나 커스텀 fold(variance/windowed/entropy)는 GPU reduce 커널(H2O sum 전용)을 못 써서
//! raw `score_buf` 를 **step 마다 CPU 로 readback** 해 plugin 안에서 fold 해야 한다. 그 per-step
//! readback 이 production OpenCL 경로에서 observe-hook 의 진짜 추가비용 (CPU fold 비용은 별도 PoC).
//!
//! `score_buf` = `[n_layers × n_heads_q × max_seq]` f32, **MEM_READ_WRITE** (엔진
//! `gpu_score.rs:155` 와 동일 flag — device-local, ALLOC_HOST_PTR 아님 → 진짜 D2H DMA copy).
//! per-step readback = `n_layers × n_heads_q × cache_seq_len` f32.
//!
//! 측정 (csl ∈ {512,1024,2048}, n=30 median, mobile thermal noise 흡수):
//!   (A) GPU-resident baseline : write kernel + finish (readback 없음 = 내장 H2O GPU 경로)
//!   (B) readback(MEM_READ_WRITE): write kernel + enqueue_read_buffer(per-step slice) + finish
//!   (C) readback(ALLOC_HOST_PTR): host-visible alloc 시 readback 이 싸지는지(설계 레버)
//!   delta(B−A) = 순수 D2H readback 비용. 32ms TBT(S25 decode) 대비 % 환산.
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_score_readback`
//! Run:   `adb shell ./microbench_score_readback [N_LAYERS] [N_HEADS_Q] [MAX_SEQ] [N_ITERS]`
//!
//! ## 실측 결과 (2026-06-10, Galaxy S25 / Adreno 830, n=30 median)
//!
//! Qwen2.5-1.5B 구성 (28L×12H, production):
//! ```text
//!  csl  | bytes   | A base   B copy  | C HOSTPTR  %TBT(B) GB/s | D rpcmem zc
//!  512  | 0.66 MB | 0.128ms  0.169ms | 0.172ms    0.5%    16.6 | 0.923ms
//!  1024 | 1.31 MB | 0.170ms  0.301ms | 0.341ms    0.9%    10.5 | 1.717ms
//!  2048 | 2.62 MB | 0.210ms  0.877ms | 1.062ms    2.7%     4.1 | 3.731ms
//! ```
//! Llama 3.2 1B 구성 (16L×32H): csl=2048 에서 B=1.400ms(4.4% TBT), D=4.483ms.
//!
//! **결론**: (1) per-step readback(B)은 production 구성 full-context 에서도 **TBT 의
//! ≤2.7%**(+fold ~360μs ≈ 3.9%) — 커스텀 fold plugin 은 OpenCL 경로에서 **실용 가능**.
//! (2) C(ALLOC_HOST_PTR)는 B 대비 무이득 — CLAUDE.md 기존 교훈(Adreno 에서 zero-copy
//! 효과 없음) 재확인. (3) D(rpcmem map+coherent read)는 B 의 **3~5× 느림**(uncached
//! DRAM coherent read 가 DMA copy 보다 비쌈; GARBAGE 0 = coherency 는 정상) —
//! score readback 용도로는 rpcmem 부적합, **plain D2H copy(B)가 설계 채택안**.

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_score_readback requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::{ArgVal, Mem};
    use ocl::{Context, Device, Platform, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_heads_q: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n_iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);
    const TBT_MS: f64 = 32.0; // S25 Qwen2.5-1.5B Q4_0 decode TBT 기준 (CLAUDE.md)

    let buf_floats = n_layers * n_heads_q * max_seq;
    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Platform: {}", platform.name()?);
    println!("Device:   {}", device.name()?);
    println!(
        "score_buf: [{} layers × {} heads × {} max_seq] = {} floats ({:.1} MB), n_iters={}",
        n_layers,
        n_heads_q,
        max_seq,
        buf_floats,
        (buf_floats * 4) as f64 / 1024.0 / 1024.0,
        n_iters
    );

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;

    // write kernel — score_buf 를 GPU 가 매 step 새로 쓰게 해(=attention 이 score 쓰는 것 모사)
    // readback 이 진짜 GPU→CPU sync 를 강제하도록.
    let kernel_src = r#"
        __kernel void wr(__global float* b) {
            uint i = get_global_id(0);
            b[i] = (float)(i & 1023) * 0.001f;
        }
    "#;
    let program = ocl::Program::builder()
        .devices(device)
        .src(kernel_src)
        .build(&context)?;
    let kernel = ocl::core::create_kernel(&program, "wr")?;

    let mk_buf = |host_ptr: bool| -> anyhow::Result<Mem> {
        let flags = if host_ptr {
            ocl::core::MEM_READ_WRITE | ocl::core::MEM_ALLOC_HOST_PTR
        } else {
            ocl::core::MEM_READ_WRITE
        };
        let m = unsafe {
            ocl::core::create_buffer::<_, f32>(context.as_core(), flags, buf_floats, None)?
        };
        Ok(m)
    };

    let median = |v: &mut [f64]| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    // 한 config(buf, csl, do_read) 를 n_iters 돌려 per-step median ms 반환.
    let run = |buf: &Mem, n_read: usize, do_read: bool| -> anyhow::Result<f64> {
        let mut host = vec![0.0f32; n_read.max(1)];
        let gws = [n_read.max(1), 1, 1];
        // warmup
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(buf))?;
            ocl::core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &gws,
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            if do_read {
                ocl::core::enqueue_read_buffer(
                    &queue,
                    buf,
                    true,
                    0,
                    &mut host,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }
        ocl::core::finish(&queue)?;
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            unsafe {
                ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(buf))?;
                ocl::core::enqueue_kernel(
                    &queue,
                    &kernel,
                    1,
                    None,
                    &gws,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
                if do_read {
                    ocl::core::enqueue_read_buffer(
                        &queue,
                        buf,
                        true,
                        0,
                        &mut host,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
            }
            ocl::core::finish(&queue)?;
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(median(&mut samples))
    };

    let buf_dev = mk_buf(false)?;
    let buf_host = mk_buf(true)?;

    // D: rpcmem DMA-BUF + CL_MEM_USE_HOST_PTR alias (엔진 zero-copy 경로, gpu_score.rs 대신).
    // GPU write 후 host_ptr 직접 read (DMA copy 없음). Adreno UMA cache-coherent (kv_buffer.rs:68).
    // host(non-Android) build 에선 RpcmemAllocator::new()=Err → D 생략.
    use llm_rs2::memory::rpcmem::allocator::RpcmemAllocator;
    let bytes = buf_floats * 4;
    let rpcmem: Option<(RpcmemAllocator, *mut u8, Mem)> = match RpcmemAllocator::new() {
        Ok(alloc) => match unsafe { alloc.alloc(bytes) } {
            Ok((hp, _fd)) => {
                let slice = unsafe { std::slice::from_raw_parts_mut(hp, bytes) };
                match unsafe {
                    ocl::core::create_buffer::<_, u8>(
                        context.as_core(),
                        ocl::core::MEM_READ_WRITE | ocl::core::MEM_USE_HOST_PTR,
                        bytes,
                        Some(slice),
                    )
                } {
                    Ok(cl) => {
                        println!(
                            "rpcmem: zero-copy alias OK ({:.1} MB DMA-BUF)",
                            bytes as f64 / 1048576.0
                        );
                        Some((alloc, hp, cl))
                    }
                    Err(e) => {
                        eprintln!("rpcmem alias 실패: {e}");
                        unsafe { alloc.free(hp) };
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("rpcmem alloc 실패: {e}");
                None
            }
        },
        Err(e) => {
            println!("[D] rpcmem unavailable: {e} — D(zero-copy) 생략");
            None
        }
    };

    // rpcmem zero-copy read: write kernel + enqueue_map_buffer(READ, 코히런시 sync) + read + unmap.
    // map = DMA copy 없이 host_ptr 노출 + 코히런시 보장. (raw pointer 직접 read 는 stale → garbage.)
    let run_rpcmem = |cl: &Mem, _host_ptr: *mut u8, n_read: usize| -> anyhow::Result<(f64, bool)> {
        use ocl::core::MapFlags;
        let gws = [n_read.max(1), 1, 1];
        let map_and_check = |check: bool| -> anyhow::Result<bool> {
            unsafe {
                ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(cl))?;
                ocl::core::enqueue_kernel(
                    &queue,
                    &kernel,
                    1,
                    None,
                    &gws,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
                let map: ocl::core::MemMap<f32> = ocl::core::enqueue_map_buffer::<f32, _, _, _>(
                    &queue,
                    cl,
                    true,
                    MapFlags::new().read(),
                    0,
                    n_read,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
                let mptr = map.as_ptr();
                let mut s = 0.0f32;
                for i in 0..n_read {
                    s += *mptr.add(i);
                }
                std::hint::black_box(s);
                let mut coherent = true;
                if check {
                    for &i in &[1usize, 5, 100, 1023, 1024] {
                        if i < n_read && (*mptr.add(i) - ((i & 1023) as f32) * 0.001).abs() > 1e-4 {
                            coherent = false;
                        }
                    }
                }
                ocl::core::enqueue_unmap_mem_object(
                    &queue,
                    cl,
                    &map,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
                ocl::core::finish(&queue)?;
                Ok(coherent)
            }
        };
        let coherent = map_and_check(true)?; // warmup + coherency 검증
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            map_and_check(false)?;
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((median(&mut samples), coherent))
    };

    println!(
        "\n{:>6} | {:>9} | {:>10} {:>10} | {:>10} {:>8} {:>8} | {:>13}",
        "csl", "bytes", "A baseline", "B copy", "C HOSTPTR", "% TBT(B)", "GB/s(B)", "D rpcmem zc"
    );
    println!("{}", "-".repeat(100));
    for &csl in &[512usize, 1024, 2048] {
        if csl > max_seq {
            continue;
        }
        let n_read = n_layers * n_heads_q * csl;
        let mb = (n_read * 4) as f64 / 1024.0 / 1024.0;
        let a = run(&buf_dev, n_read, false)?; // GPU-resident (write+finish, no read)
        let b = run(&buf_dev, n_read, true)?; // MEM_READ_WRITE D2H copy
        let c = run(&buf_host, n_read, true)?; // ALLOC_HOST_PTR readback
        let delta = (b - a).max(0.0);
        let gbs = if delta > 0.0 {
            (n_read * 4) as f64 / (delta / 1000.0) / 1e9
        } else {
            0.0
        };
        let pct = b / TBT_MS * 100.0;
        let d_str = if let Some((_, hp, cl)) = &rpcmem {
            let (d, coh) = run_rpcmem(cl, *hp, n_read)?;
            format!("{:.3}ms{}", d, if coh { "" } else { " GARBAGE!" })
        } else {
            "n/a".to_string()
        };
        println!(
            "{:>6} | {:>6.2} MB | {:>8.3}ms {:>8.3}ms | {:>8.3}ms {:>7.1}% {:>7.1} | {:>13}",
            csl, mb, a, b, c, pct, gbs, d_str
        );
    }

    if let Some((alloc, hp, cl)) = rpcmem {
        drop(cl); // alias 먼저 release
        unsafe { alloc.free(hp) };
    }

    println!("\n해석:");
    println!(
        "  A = 내장 H2O GPU-resident(readback 없음).  B = 커스텀 plugin MEM_READ_WRITE D2H copy."
    );
    println!(
        "  C = ALLOC_HOST_PTR readback.  D = rpcmem DMA-BUF zero-copy(host_ptr 직접, DMA copy 없음)."
    );
    println!(
        "  % TBT = B 가 {TBT_MS}ms decode 의 몇 %.  D 가 B 보다 작으면 rpcmem zero-copy 가 readback 비용 절감."
    );
    println!(
        "  주의: B/C 는 copy 만(이후 plugin fold ~360μs 추가). D 는 finish+coherent read(fold-read 포함)."
    );
    println!(
        "        D 의 coherent read 가 uncached DRAM hit 면 느릴 수 있음 — GARBAGE 표시는 coherency 깨짐."
    );
    println!(
        "        실제 score_buf 는 [layer][head][pos] strided — 본 측정은 동일 byte contiguous(하한)."
    );
    Ok(())
}
