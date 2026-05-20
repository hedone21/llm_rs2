//! microbench_vulkan_h2d_baseline — Vulkan PoC Phase C (Q3: throughput parity)
//!
//! 목적: Vulkan H2D throughput이 OpenCL ALLOC_HOST_PTR + clEnqueueWriteBuffer
//! (Phase 0 baseline 22.28 ms / 27.5 GB/s @ 600MB)와 동급인지 확인.
//!
//! 측정:
//!   T1: HOST_VISIBLE | HOST_COHERENT 600MB 직접 write (UMA 활용)
//!       — host pointer로 memcpy. Adreno UMA에선 별도 H2D 없음
//!   T2: HOST_VISIBLE staging → DEVICE_LOCAL device buffer copy via vkCmdCopyBuffer
//!       — OpenCL clEnqueueWriteBuffer와 1:1 매칭
//!
//! Build: `cargo build --release --features vulkan --target aarch64-linux-android --bin microbench_vulkan_h2d_baseline`
//! Run:   `adb shell /data/local/tmp/microbench_vulkan_h2d_baseline [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("microbench_vulkan_h2d_baseline requires --features vulkan");
    std::process::exit(2);
}

#[cfg(feature = "vulkan")]
fn main() -> anyhow::Result<()> {
    use ash::{Entry, vk};
    use std::ffi::CString;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let size_bytes = (size_mb * 1024 * 1024) as u64;

    let entry = unsafe { Entry::load() }.map_err(|e| anyhow::anyhow!("Entry::load: {:?}", e))?;
    let app_name = CString::new("microbench_vulkan_h2d_baseline").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let inst_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&inst_ci, None)? };

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
    println!("Buffer: {} MB, n_iters: {}", size_mb, n_iters);

    let qfs = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let _ = qfs[0];
    let qfi = 0u32;
    let priorities = [1.0f32];
    let queue_cis = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&priorities)];
    let dev_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_cis);
    let device = unsafe { instance.create_device(phys, &dev_ci, None)? };
    let queue = unsafe { device.get_device_queue(qfi, 0) };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    let pick_mem_type = |bits: u32, want: vk::MemoryPropertyFlags| -> Option<u32> {
        (0..mem_props.memory_type_count).find(|&i| {
            let t = mem_props.memory_types[i as usize];
            (bits & (1 << i)) != 0 && t.property_flags.contains(want)
        })
    };

    // ── Source data ──
    let n_floats = (size_bytes / 4) as usize;
    let host_src: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 1.0e-6).collect();

    // === Test 1: HOST_VISIBLE | HOST_COHERENT direct write (UMA-style) ===
    println!("\n=== Test 1: HOST_VISIBLE | HOST_COHERENT direct write ===");
    println!(
        "(UMA: write to mapped pointer; no explicit H2D copy. closest to OpenCL ALLOC_HOST_PTR analogy)"
    );

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size_bytes)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf_host = unsafe { device.create_buffer(&buf_ci, None)? };
    let req = unsafe { device.get_buffer_memory_requirements(buf_host) };
    let mt_host = pick_mem_type(
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| anyhow::anyhow!("No HOST_VISIBLE|HOST_COHERENT memory type"))?;
    let ai = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mt_host);
    let mem_host = unsafe { device.allocate_memory(&ai, None)? };
    unsafe { device.bind_buffer_memory(buf_host, mem_host, 0)? };
    println!(
        "  buf_host alloc OK (memory type {}, flags={:?})",
        mt_host, mem_props.memory_types[mt_host as usize].property_flags
    );

    // Persistent map
    let host_ptr = unsafe {
        device.map_memory(mem_host, 0, size_bytes, vk::MemoryMapFlags::empty())? as *mut u8
    };

    let mut t1_samples = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_src.as_ptr() as *const u8,
                host_ptr,
                size_bytes as usize,
            );
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        t1_samples.push(ms);
        println!("  iter {:2}: {:7.2} ms", i, ms);
    }
    report_stats("HOST_VISIBLE direct write", size_mb, &t1_samples);

    // === Test 2: HOST_VISIBLE staging → DEVICE_LOCAL via vkCmdCopyBuffer ===
    println!("\n=== Test 2: HOST_VISIBLE staging → DEVICE_LOCAL via vkCmdCopyBuffer ===");

    // Find DEVICE_LOCAL-only or DEVICE_LOCAL non-host_visible memory type.
    // On Adreno UMA, DEVICE_LOCAL alone (without HOST_VISIBLE) is type 0/1/2/7.
    let buf_dev_ci = vk::BufferCreateInfo::default()
        .size(size_bytes)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf_dev = unsafe { device.create_buffer(&buf_dev_ci, None)? };
    let req_dev = unsafe { device.get_buffer_memory_requirements(buf_dev) };
    // Prefer DEVICE_LOCAL-only (no HOST_VISIBLE) to model discrete GPU pattern.
    let mt_dev = pick_mem_type_strict(
        &mem_props,
        req_dev.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )
    .or_else(|| {
        // Fallback: any DEVICE_LOCAL
        pick_mem_type(
            req_dev.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    })
    .ok_or_else(|| anyhow::anyhow!("No DEVICE_LOCAL memory type"))?;
    let ai_dev = vk::MemoryAllocateInfo::default()
        .allocation_size(req_dev.size)
        .memory_type_index(mt_dev);
    let mem_dev = unsafe { device.allocate_memory(&ai_dev, None)? };
    unsafe { device.bind_buffer_memory(buf_dev, mem_dev, 0)? };
    println!(
        "  buf_dev alloc OK (memory type {}, flags={:?})",
        mt_dev, mem_props.memory_types[mt_dev as usize].property_flags
    );

    // Command pool + buffer for copy
    let cp_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qfi)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cp = unsafe { device.create_command_pool(&cp_ci, None)? };
    let cb_ai = vk::CommandBufferAllocateInfo::default()
        .command_pool(cp)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&cb_ai)? }[0];
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    let region = [vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(0)
        .size(size_bytes)];

    // Warmup 3x
    for _ in 0..3 {
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            device.cmd_copy_buffer(cmd, buf_host, buf_dev, &region);
            device.end_command_buffer(cmd)?;
            device.reset_fences(&[fence])?;
            let cmd_arr = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
            device.queue_submit(queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
    }

    let mut t2_samples = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        // Refresh staging contents (mimics swap_executor calling write per layer)
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_src.as_ptr() as *const u8,
                host_ptr,
                size_bytes as usize,
            );
        }
        let t0 = Instant::now();
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            device.cmd_copy_buffer(cmd, buf_host, buf_dev, &region);
            device.end_command_buffer(cmd)?;
            device.reset_fences(&[fence])?;
            let cmd_arr = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
            device.queue_submit(queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        t2_samples.push(ms);
        println!("  iter {:2}: {:7.2} ms", i, ms);
    }
    report_stats("staging → device_local copy", size_mb, &t2_samples);

    // === Q3 verdict ===
    let opencl_baseline_ms = 22.28; // Phase 0
    let n_t1 = t1_samples.len() as f64;
    let n_t2 = t2_samples.len() as f64;
    let mean_t1 = t1_samples.iter().sum::<f64>() / n_t1;
    let mean_t2 = t2_samples.iter().sum::<f64>() / n_t2;
    let ratio_t1 = mean_t1 / opencl_baseline_ms;
    let ratio_t2 = mean_t2 / opencl_baseline_ms;

    println!("\n=== Phase C summary (Q3 answer) ===");
    println!(
        "OpenCL baseline (ALLOC_HOST_PTR + clEnqueueWriteBuffer 600MB): {:.2} ms",
        opencl_baseline_ms
    );
    println!(
        "Vulkan T1 (HOST_VISIBLE direct write):                          {:.2} ms ({:.3}x)",
        mean_t1, ratio_t1
    );
    println!(
        "Vulkan T2 (staging → DEVICE_LOCAL copy):                        {:.2} ms ({:.3}x)",
        mean_t2, ratio_t2
    );

    let bands_t2 = if (0.90..=1.10).contains(&ratio_t2) {
        "✓ parity (within ±10%)"
    } else if ratio_t2 < 0.90 {
        "✓ Vulkan superior"
    } else {
        "✗ Vulkan slower"
    };
    println!(
        "\nQ3 ANSWER (T2 vs OpenCL baseline): {} — ratio {:.3}x",
        bands_t2, ratio_t2
    );
    println!(
        "(T1 measures UMA host write only; not 1:1 to OpenCL clEnqueueWriteBuffer.\n T2 is the apples-to-apples comparison.)"
    );

    // Cleanup
    unsafe {
        device.unmap_memory(mem_host);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cp, None);
        device.destroy_buffer(buf_host, None);
        device.destroy_buffer(buf_dev, None);
        device.free_memory(mem_host, None);
        device.free_memory(mem_dev, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn pick_mem_type_strict(
    mp: &ash::vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    must_have: ash::vk::MemoryPropertyFlags,
    must_not_have: ash::vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mp.memory_type_count).find(|&i| {
        let t = mp.memory_types[i as usize];
        let f = t.property_flags;
        (bits & (1 << i)) != 0 && f.contains(must_have) && !f.intersects(must_not_have)
    })
}

#[cfg(feature = "vulkan")]
fn report_stats(label: &str, size_mb: usize, samples_ms: &[f64]) {
    let n = samples_ms.len() as f64;
    let mean = samples_ms.iter().sum::<f64>() / n;
    let var = samples_ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let mut sorted: Vec<f64> = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let cv = stddev / mean;
    let bw = (size_mb as f64) / 1024.0 / (mean / 1000.0);
    println!("\n[{}] {} MB, n={}", label, size_mb, samples_ms.len());
    println!("  mean   : {:7.2} ms", mean);
    println!("  median : {:7.2} ms", median);
    println!("  stddev : {:7.2} ms ({:.1}% σ/mean)", stddev, cv * 100.0);
    println!("  effective BW: {:5.2} GB/s", bw);
}
