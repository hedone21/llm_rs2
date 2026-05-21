//! microbench_vulkan_two_queue — Vulkan PoC Phase B (Q2: async swap feasibility)
//!
//! 목적: Vulkan의 두 queue가 동시 submit될 때 wall-clock이 single-queue 1.0x
//! 인지 2.0x인지 측정. OpenCL Phase 6 (microbench_two_queue_concurrent.rs)와
//! 1:1 매칭. Adreno HW 직렬화가 OpenCL ICD 한정인지 HW level인지 확정.
//!
//! Configs (n=30):
//!   C1: Single queue baseline (family 0 queue 0)
//!   C2: Same family x2  (family 0 queue 0 + queue 1)
//!   C3: Cross family    (family 0 queue 0 + family 1 queue 0)  ← Q2 핵심
//!   C4: Cross family + binary semaphore (compute_q signal -> compute2 wait)
//!
//! Build: `cargo build --release --features vulkan --target aarch64-linux-android --bin microbench_vulkan_two_queue`
//! Run:   `adb shell /data/local/tmp/microbench_vulkan_two_queue [N_ITERS]`

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("microbench_vulkan_two_queue requires --features vulkan");
    std::process::exit(2);
}

#[cfg(feature = "vulkan")]
fn main() -> anyhow::Result<()> {
    use ash::{Entry, vk};
    use std::ffi::CString;
    use std::time::Instant;

    const GSIZE: u32 = 1024;
    const LOCAL_X: u32 = 1024;
    const SPV: &[u8] = include_bytes!("../shaders/busy.spv");

    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    // ── Instance ──
    let entry = unsafe { Entry::load() }.map_err(|e| anyhow::anyhow!("Entry::load: {:?}", e))?;
    let app_name = CString::new("microbench_vulkan_two_queue").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let inst_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&inst_ci, None)? };

    // ── Pick Adreno ──
    let pds = unsafe { instance.enumerate_physical_devices()? };
    let phys = pds
        .iter()
        .copied()
        .find(|&pd| {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            let name = unsafe { std::ffi::CStr::from_ptr(p.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            name.contains("Adreno") || name.contains("Qualcomm")
        })
        .unwrap_or(pds[0]);
    let phys_props = unsafe { instance.get_physical_device_properties(phys) };
    let phys_name = unsafe { std::ffi::CStr::from_ptr(phys_props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    println!("Device: {}", phys_name);
    println!("n_iters per config: {}", n_iters);

    // ── Identify queue families ──
    let qfs = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let family0 = 0u32;
    let family0_count = qfs[0].queue_count;
    let mut family_compute_only: Option<u32> = None;
    for (i, qf) in qfs.iter().enumerate() {
        if qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            family_compute_only = Some(i as u32);
            break;
        }
    }
    let family1 = family_compute_only
        .ok_or_else(|| anyhow::anyhow!("No compute-only queue family on this device"))?;
    println!(
        "Family 0 (universal): count={}, family 1 (compute-only): index={}",
        family0_count, family1
    );

    // ── Logical device with queues from both families ──
    let priorities = [1.0f32, 1.0, 1.0];
    let q0_count = family0_count.min(2); // we need queue 0 + queue 1 for C2
    let q0_priorities = &priorities[..q0_count as usize];
    let q1_priorities = &priorities[..1];
    let queue_cis = [
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family0)
            .queue_priorities(q0_priorities),
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family1)
            .queue_priorities(q1_priorities),
    ];
    let dev_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_cis);
    let device = unsafe { instance.create_device(phys, &dev_ci, None)? };
    let q0a = unsafe { device.get_device_queue(family0, 0) };
    let q0b = if q0_count >= 2 {
        unsafe { device.get_device_queue(family0, 1) }
    } else {
        q0a
    };
    let q1 = unsafe { device.get_device_queue(family1, 0) };

    // ── Memory type (HOST_VISIBLE | HOST_COHERENT) ──
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    let pick_mem_type = |bits: u32, want: vk::MemoryPropertyFlags| -> Option<u32> {
        (0..mem_props.memory_type_count).find(|&i| {
            let t = mem_props.memory_types[i as usize];
            (bits & (1 << i)) != 0 && t.property_flags.contains(want)
        })
    };

    // ── Allocate two disjoint output buffers (one per queue) ──
    let buf_size = (GSIZE as usize * std::mem::size_of::<f32>()) as u64;
    let mk_buf = |fams: &[u32]| -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let sharing = if fams.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let mut ci = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(sharing);
        if fams.len() > 1 {
            ci = ci.queue_family_indices(fams);
        }
        let b = unsafe { device.create_buffer(&ci, None)? };
        let r = unsafe { device.get_buffer_memory_requirements(b) };
        let mt = pick_mem_type(
            r.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| anyhow::anyhow!("No HOST_VISIBLE|HOST_COHERENT memory type"))?;
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(r.size)
            .memory_type_index(mt);
        let m = unsafe { device.allocate_memory(&ai, None)? };
        unsafe { device.bind_buffer_memory(b, m, 0)? };
        Ok((b, m))
    };
    // Buffer A: usable by family0 queues only (no transfer needed for same-family use)
    let (buf_a, mem_a) = mk_buf(&[family0])?;
    // Buffer B: shared across family0 and family1 (CONCURRENT)
    let queue_fams_concurrent = [family0, family1];
    let (buf_b, mem_b) = mk_buf(&queue_fams_concurrent)?;

    // ── Shader module + pipeline (push_constant int iters) ──
    let spv_u32: Vec<u32> = SPV
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let sm_ci = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
    let sm = unsafe { device.create_shader_module(&sm_ci, None)? };

    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let dsl = unsafe { device.create_descriptor_set_layout(&dsl_ci, None)? };
    let dsls = [dsl];
    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(4)];
    let pl_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls)
        .push_constant_ranges(&pc_range);
    let pl = unsafe { device.create_pipeline_layout(&pl_ci, None)? };

    let entry_name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(sm)
        .name(&entry_name);
    let pp_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pl);
    let pps = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pp_ci], None) }
        .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {:?}", e))?;
    let pipeline = pps[0];

    // ── Two descriptor sets (one per buffer) ──
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(2)];
    let dp_ci = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(2);
    let dp = unsafe { device.create_descriptor_pool(&dp_ci, None)? };
    let layouts2 = [dsl, dsl];
    let set_ai = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(dp)
        .set_layouts(&layouts2);
    let sets = unsafe { device.allocate_descriptor_sets(&set_ai)? };
    let set_a = sets[0];
    let set_b = sets[1];
    let bi_a = [vk::DescriptorBufferInfo::default()
        .buffer(buf_a)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let bi_b = [vk::DescriptorBufferInfo::default()
        .buffer(buf_b)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set_a)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi_a),
        vk::WriteDescriptorSet::default()
            .dst_set(set_b)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi_b),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // ── Command pools (one per queue family) ──
    let cp_ci_0 = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family0)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cp_ci_1 = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family1)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cp0 = unsafe { device.create_command_pool(&cp_ci_0, None)? };
    let cp1 = unsafe { device.create_command_pool(&cp_ci_1, None)? };

    let alloc_cb = |cp: vk::CommandPool| -> anyhow::Result<vk::CommandBuffer> {
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(cp)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe { device.allocate_command_buffers(&ai)? }[0])
    };
    let cb_0a = alloc_cb(cp0)?; // for q0a, set_a
    let cb_0b = alloc_cb(cp0)?; // for q0b, set_b
    let cb_1 = alloc_cb(cp1)?; // for q1, set_b

    // ── Record helper: bind pipeline + descriptor + push constant + dispatch ──
    let record = |cb: vk::CommandBuffer,
                  set: vk::DescriptorSet,
                  iters: i32|
     -> anyhow::Result<()> {
        let begin = vk::CommandBufferBeginInfo::default();
        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cb, &begin)?;
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[]);
            let bytes = iters.to_le_bytes();
            device.cmd_push_constants(cb, pl, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            let n_groups = GSIZE.div_ceil(LOCAL_X);
            device.cmd_dispatch(cb, n_groups, 1, 1);
            device.end_command_buffer(cb)?;
        }
        Ok(())
    };

    // ── Fences ──
    let mk_fence = || -> anyhow::Result<vk::Fence> {
        Ok(unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? })
    };
    let f0 = mk_fence()?;
    let f1 = mk_fence()?;

    // ── Tune iters to ~1ms per kernel on q0a ──
    let mut iters: i32 = 30_000;
    println!("\nTuning kernel iterations to ~1ms on family 0 queue 0...");
    for _ in 0..7 {
        record(cb_0a, set_a, iters)?;
        let cmd_arr = [cb_0a];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
        unsafe {
            device.reset_fences(&[f0])?;
            let t0 = Instant::now();
            device.queue_submit(q0a, &[submit], f0)?;
            device.wait_for_fences(&[f0], true, u64::MAX)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  iters={}: {:.2} ms", iters, ms);
            if ms < 0.7 {
                iters = (iters as f64 * 1.0 / ms.max(0.1)) as i32;
            } else if ms > 1.5 {
                iters = (iters as f64 / (ms / 1.0)) as i32;
            } else {
                break;
            }
        }
    }
    println!("  Final iters={} (~1ms target)", iters);

    enum Cfg {
        Single,
        SameFamily,
        CrossFamily,
        CrossFamilyOrdered,
    }
    let configs: Vec<(&str, Cfg)> = vec![
        ("C1: Single queue (family 0 q0)", Cfg::Single),
        ("C2: Same family × 2 (family 0 q0 + q1)", Cfg::SameFamily),
        ("C3: Cross family (family 0 + family 1)", Cfg::CrossFamily),
        (
            "C4: Cross family + binary semaphore",
            Cfg::CrossFamilyOrdered,
        ),
    ];

    println!(
        "\n=== Two-queue concurrent kernel (Vulkan, busy ~1ms × {} iters/cfg) ===",
        n_iters
    );
    println!("Hypothesis: ratio_to_C1 ≈ 1.0 (parallel) vs ≈ 2.0 (serial).\n");

    let mut summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for (label, cfg) in &configs {
        // Setup per-config (semaphore for C4)
        let sem_signal: Option<vk::Semaphore> = match cfg {
            Cfg::CrossFamilyOrdered => {
                Some(unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? })
            }
            _ => None,
        };

        // Warmup
        for _ in 0..5 {
            record(cb_0a, set_a, iters)?;
            let cmd_arr_0a = [cb_0a];
            unsafe {
                device.reset_fences(&[f0])?;
                device.queue_submit(
                    q0a,
                    &[vk::SubmitInfo::default().command_buffers(&cmd_arr_0a)],
                    f0,
                )?;
                device.wait_for_fences(&[f0], true, u64::MAX)?;
            }
        }

        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            // record cb_0a always; record cb_0b or cb_1 conditionally
            record(cb_0a, set_a, iters)?;
            let (qb, cbb) = match cfg {
                Cfg::Single => (q0a, cb_0a), // unused second submit
                Cfg::SameFamily => {
                    record(cb_0b, set_b, iters)?;
                    (q0b, cb_0b)
                }
                Cfg::CrossFamily | Cfg::CrossFamilyOrdered => {
                    record(cb_1, set_b, iters)?;
                    (q1, cb_1)
                }
            };

            unsafe {
                device.reset_fences(&[f0, f1])?;
            }
            let cmd_arr_a = [cb_0a];
            let cmd_arr_b = [cbb];
            let t0 = Instant::now();
            match cfg {
                Cfg::Single => {
                    let s = vk::SubmitInfo::default().command_buffers(&cmd_arr_a);
                    unsafe {
                        device.queue_submit(q0a, &[s], f0)?;
                        device.wait_for_fences(&[f0], true, u64::MAX)?;
                    }
                }
                Cfg::SameFamily | Cfg::CrossFamily => {
                    let s_a = vk::SubmitInfo::default().command_buffers(&cmd_arr_a);
                    let s_b = vk::SubmitInfo::default().command_buffers(&cmd_arr_b);
                    unsafe {
                        device.queue_submit(q0a, &[s_a], f0)?;
                        device.queue_submit(qb, &[s_b], f1)?;
                        device.wait_for_fences(&[f0, f1], true, u64::MAX)?;
                    }
                }
                Cfg::CrossFamilyOrdered => {
                    let sem = sem_signal.unwrap();
                    let signal_arr = [sem];
                    let wait_arr = [sem];
                    let stage_arr = [vk::PipelineStageFlags::COMPUTE_SHADER];
                    let s_a = vk::SubmitInfo::default()
                        .command_buffers(&cmd_arr_a)
                        .signal_semaphores(&signal_arr);
                    let s_b = vk::SubmitInfo::default()
                        .command_buffers(&cmd_arr_b)
                        .wait_semaphores(&wait_arr)
                        .wait_dst_stage_mask(&stage_arr);
                    unsafe {
                        device.queue_submit(q0a, &[s_a], f0)?;
                        device.queue_submit(qb, &[s_b], f1)?;
                        device.wait_for_fences(&[f0, f1], true, u64::MAX)?;
                    }
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            samples.push(ms);
        }

        if let Some(sem) = sem_signal {
            unsafe { device.destroy_semaphore(sem, None) };
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "{:<45} mean={:.3} median={:.3} σ={:.3} σ/mean={:.3}",
            label, mean, median, stddev, cv
        );
        summary.push((label.to_string(), mean, median, cv));
    }

    let baseline = summary[0].1;
    println!("\n=== Phase B summary (Q2 answer) ===");
    println!(
        "{:<45} {:>10} {:>10} {:>10} {:>10}",
        "Config", "mean", "median", "σ/mean", "ratio_C1"
    );
    println!("{}", "-".repeat(95));
    for (label, mean, median, cv) in &summary {
        println!(
            "{:<45} {:>8.3}ms {:>8.3}ms {:>9.3} {:>9.3}x",
            label,
            mean,
            median,
            cv,
            mean / baseline
        );
    }

    // Q2 answer interpretation
    let c3_ratio = summary[2].1 / baseline;
    println!("\nC3 (Cross family) ratio_to_C1 = {:.3}x", c3_ratio);
    if c3_ratio <= 1.10 {
        println!(
            "=> Q2 ANSWER: ✓ async OK (parallel). OpenCL ICD-specific serialization narrowed."
        );
    } else if c3_ratio < 1.80 {
        println!(
            "=> Q2 ANSWER: partial overlap. Driver mixed schedule. (treated as ✗ for Go/NoGo)"
        );
    } else {
        println!("=> Q2 ANSWER: ✗ HW-level serialize. Phase 6 OpenCL evidence STRENGTHENED.");
    }

    // ── Cleanup ──
    unsafe {
        device.destroy_fence(f0, None);
        device.destroy_fence(f1, None);
        device.destroy_command_pool(cp0, None);
        device.destroy_command_pool(cp1, None);
        device.destroy_descriptor_pool(dp, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pl, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(sm, None);
        device.destroy_buffer(buf_a, None);
        device.destroy_buffer(buf_b, None);
        device.free_memory(mem_a, None);
        device.free_memory(mem_b, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
