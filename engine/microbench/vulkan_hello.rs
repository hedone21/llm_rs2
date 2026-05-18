//! microbench_vulkan_hello — Vulkan PoC Phase A
//!
//! 목적 (Q1): Adreno 830에서 Vulkan compute kernel이 정확한 결과를 내는지 확인.
//! 동시에 queue family enumeration 결과로 Phase B(Q2 async swap) 분기 결정.
//!
//! 측정/출력:
//! - VkInstance/VkDevice 생성
//! - vkGetPhysicalDeviceQueueFamilyProperties: graphics/compute/transfer family 분리 여부
//! - Memory heap/type list (DEVICE_LOCAL + HOST_VISIBLE 분리 여부)
//! - simple_add kernel 1024 element c[i] = a[i] + b[i] 전수 검증
//!
//! Build: `cargo build --release --features vulkan --target aarch64-linux-android --bin microbench_vulkan_hello`
//! Run:   `adb shell /data/local/tmp/microbench_vulkan_hello`

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("microbench_vulkan_hello requires --features vulkan");
    std::process::exit(2);
}

#[cfg(feature = "vulkan")]
fn main() -> anyhow::Result<()> {
    use ash::{Entry, vk};
    use std::ffi::CString;

    const N_FLOATS: usize = 1024;
    const LOCAL_X: u32 = 256;
    const SPV: &[u8] = include_bytes!("../shaders/simple_add.spv");

    println!("=== microbench_vulkan_hello ===\n");

    // ── 1. Load Vulkan loader ──
    let entry = unsafe { Entry::load() }
        .map_err(|e| anyhow::anyhow!("Entry::load failed (libvulkan.so not found?): {:?}", e))?;
    println!("Vulkan loader: OK");

    // ── 2. Create instance ──
    let app_name = CString::new("microbench_vulkan_hello").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None)? };

    // ── 3. Pick physical device (Adreno preferred) ──
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.is_empty() {
        anyhow::bail!("No Vulkan physical devices found");
    }
    println!("\nPhysical devices ({}):", physical_devices.len());
    let mut chosen: Option<vk::PhysicalDevice> = None;
    for &pd in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let api_major = vk::api_version_major(props.api_version);
        let api_minor = vk::api_version_minor(props.api_version);
        let api_patch = vk::api_version_patch(props.api_version);
        println!(
            "  - {} | type={:?} | api={}.{}.{} | vendor=0x{:x} | device=0x{:x}",
            name,
            props.device_type,
            api_major,
            api_minor,
            api_patch,
            props.vendor_id,
            props.device_id
        );
        if chosen.is_none() && (name.contains("Adreno") || name.contains("Qualcomm")) {
            chosen = Some(pd);
        }
    }
    let phys = chosen.unwrap_or(physical_devices[0]);
    let phys_props = unsafe { instance.get_physical_device_properties(phys) };
    let phys_name = unsafe { std::ffi::CStr::from_ptr(phys_props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    println!("\nSelected: {}", phys_name);

    // ── 4. Queue family enumeration (KEY OUTPUT for Q2) ──
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    println!("\nQueue families ({}):", queue_families.len());
    let mut compute_qf: Option<u32> = None;
    let mut transfer_only_qf: Option<u32> = None;
    let mut compute_only_qf: Option<u32> = None;
    for (i, qf) in queue_families.iter().enumerate() {
        let g = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let c = qf.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let t = qf.queue_flags.contains(vk::QueueFlags::TRANSFER);
        let s = qf.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING);
        println!(
            "  family {}: count={}, flags=[{}{}{}{}] timestamp_bits={}",
            i,
            qf.queue_count,
            if g { "G" } else { "-" },
            if c { "C" } else { "-" },
            if t { "T" } else { "-" },
            if s { "S" } else { "-" },
            qf.timestamp_valid_bits
        );
        if c && compute_qf.is_none() {
            compute_qf = Some(i as u32);
        }
        if c && !g && compute_only_qf.is_none() {
            compute_only_qf = Some(i as u32);
        }
        if t && !c && !g && transfer_only_qf.is_none() {
            transfer_only_qf = Some(i as u32);
        }
    }
    println!(
        "\nQ2 dispatch decision: compute_qf={:?}, compute_only_qf={:?}, transfer_only_qf={:?}",
        compute_qf, compute_only_qf, transfer_only_qf
    );
    if transfer_only_qf.is_some() {
        println!("  -> transfer-only family EXISTS. Phase B can test compute+transfer split.");
    } else if compute_only_qf.is_some() {
        println!("  -> compute-only family exists. Phase B falls back to graphics+compute pair.");
    } else {
        println!("  -> only universal queue family. Phase B reduces to same-family × 2 only.");
    }
    let qf_idx = compute_qf.ok_or_else(|| anyhow::anyhow!("No compute queue family"))?;

    // ── 5. Memory heaps ──
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    println!("\nMemory heaps ({}):", mem_props.memory_heap_count);
    for i in 0..mem_props.memory_heap_count as usize {
        let h = mem_props.memory_heaps[i];
        let dl = h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        println!(
            "  heap {}: size={} MB, flags={}",
            i,
            h.size / (1024 * 1024),
            if dl { "DEVICE_LOCAL" } else { "(host)" }
        );
    }
    println!("Memory types ({}):", mem_props.memory_type_count);
    for i in 0..mem_props.memory_type_count as usize {
        let t = mem_props.memory_types[i];
        let mut tags = Vec::new();
        if t.property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            tags.push("DEVICE_LOCAL");
        }
        if t.property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            tags.push("HOST_VISIBLE");
        }
        if t.property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            tags.push("HOST_COHERENT");
        }
        if t.property_flags
            .contains(vk::MemoryPropertyFlags::HOST_CACHED)
        {
            tags.push("HOST_CACHED");
        }
        if t.property_flags
            .contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
        {
            tags.push("LAZILY_ALLOCATED");
        }
        println!("  type {}: heap={}, [{}]", i, t.heap_index, tags.join("|"));
    }

    // ── 6. Create logical device + queue ──
    let priorities = [1.0_f32];
    let queue_ci = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qf_idx)
        .queue_priorities(&priorities);
    let queue_cis = [queue_ci];
    let device_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_cis);
    let device = unsafe { instance.create_device(phys, &device_ci, None)? };
    let queue = unsafe { device.get_device_queue(qf_idx, 0) };
    println!("\nLogical device + queue: OK (qf={})", qf_idx);

    // ── 7. Allocate 3 buffers (A, B, C) on HOST_VISIBLE | HOST_COHERENT ──
    let size_bytes = (N_FLOATS * std::mem::size_of::<f32>()) as u64;
    let make_buffer = |dev: &ash::Device| -> anyhow::Result<vk::Buffer> {
        let ci = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        Ok(unsafe { dev.create_buffer(&ci, None)? })
    };
    let buf_a = make_buffer(&device)?;
    let buf_b = make_buffer(&device)?;
    let buf_c = make_buffer(&device)?;

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buf_a) };
    let mem_type_idx = (0..mem_props.memory_type_count)
        .find(|&i| {
            let t = mem_props.memory_types[i as usize];
            mem_reqs.memory_type_bits & (1 << i) != 0
                && t.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .ok_or_else(|| anyhow::anyhow!("No HOST_VISIBLE|HOST_COHERENT memory type"))?;
    println!("Memory type for buffers: {}", mem_type_idx);

    let alloc_one = |dev: &ash::Device, buf: vk::Buffer| -> anyhow::Result<vk::DeviceMemory> {
        let req = unsafe { dev.get_buffer_memory_requirements(buf) };
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type_idx);
        let mem = unsafe { dev.allocate_memory(&ai, None)? };
        unsafe { dev.bind_buffer_memory(buf, mem, 0)? };
        Ok(mem)
    };
    let mem_a = alloc_one(&device, buf_a)?;
    let mem_b = alloc_one(&device, buf_b)?;
    let mem_c = alloc_one(&device, buf_c)?;

    // Write input data via mapping
    let host_a: Vec<f32> = (0..N_FLOATS).map(|i| i as f32 * 0.5).collect();
    let host_b: Vec<f32> = (0..N_FLOATS).map(|i| i as f32 * 0.25 + 1.0).collect();
    unsafe {
        let ptr_a =
            device.map_memory(mem_a, 0, size_bytes, vk::MemoryMapFlags::empty())? as *mut f32;
        std::ptr::copy_nonoverlapping(host_a.as_ptr(), ptr_a, N_FLOATS);
        device.unmap_memory(mem_a);
        let ptr_b =
            device.map_memory(mem_b, 0, size_bytes, vk::MemoryMapFlags::empty())? as *mut f32;
        std::ptr::copy_nonoverlapping(host_b.as_ptr(), ptr_b, N_FLOATS);
        device.unmap_memory(mem_b);
    }

    // ── 8. Shader module ──
    let spv_u32: Vec<u32> = SPV
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let sm_ci = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
    let shader_module = unsafe { device.create_shader_module(&sm_ci, None)? };
    println!("Shader module: OK ({} u32 words)", spv_u32.len());

    // ── 9. Descriptor set layout, pipeline layout, pipeline ──
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let dsl = unsafe { device.create_descriptor_set_layout(&dsl_ci, None)? };
    let dsls = [dsl];
    let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls);
    let pl = unsafe { device.create_pipeline_layout(&pl_ci, None)? };

    let entry_name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name);
    let pipeline_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pl);
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None) }
            .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {:?}", e))?;
    let pipeline = pipelines[0];
    println!("Compute pipeline: OK");

    // ── 10. Descriptor pool + set ──
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let dp_ci = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    let dp = unsafe { device.create_descriptor_pool(&dp_ci, None)? };
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(dp)
        .set_layouts(&dsls);
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
    let set = sets[0];

    let bi_a = [vk::DescriptorBufferInfo::default()
        .buffer(buf_a)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let bi_b = [vk::DescriptorBufferInfo::default()
        .buffer(buf_b)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let bi_c = [vk::DescriptorBufferInfo::default()
        .buffer(buf_c)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi_a),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi_b),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi_c),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // ── 11. Command pool + buffer, record dispatch ──
    let cp_ci = vk::CommandPoolCreateInfo::default().queue_family_index(qf_idx);
    let cp = unsafe { device.create_command_pool(&cp_ci, None)? };
    let cb_ai = vk::CommandBufferAllocateInfo::default()
        .command_pool(cp)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmds = unsafe { device.allocate_command_buffers(&cb_ai)? };
    let cmd = cmds[0];

    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    let n_groups = (N_FLOATS as u32).div_ceil(LOCAL_X);
    unsafe {
        device.begin_command_buffer(cmd, &begin)?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[]);
        device.cmd_dispatch(cmd, n_groups, 1, 1);
        device.end_command_buffer(cmd)?;
    }

    // ── 12. Submit + wait fence ──
    let cmd_arr = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
    let fence_ci = vk::FenceCreateInfo::default();
    let fence = unsafe { device.create_fence(&fence_ci, None)? };
    unsafe {
        device.queue_submit(queue, &[submit], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    println!(
        "\nDispatch + wait_for_fences: OK ({} groups of {} threads)",
        n_groups, LOCAL_X
    );

    // ── 13. Read back and verify ──
    let mut host_c = vec![0.0_f32; N_FLOATS];
    unsafe {
        let ptr =
            device.map_memory(mem_c, 0, size_bytes, vk::MemoryMapFlags::empty())? as *const f32;
        std::ptr::copy_nonoverlapping(ptr, host_c.as_mut_ptr(), N_FLOATS);
        device.unmap_memory(mem_c);
    }
    let mut mismatch = 0usize;
    let mut first_bad = None;
    for i in 0..N_FLOATS {
        let exp = host_a[i] + host_b[i];
        if (host_c[i] - exp).abs() > 1e-6 * exp.abs().max(1.0) {
            mismatch += 1;
            if first_bad.is_none() {
                first_bad = Some((i, host_c[i], exp));
            }
        }
    }
    if mismatch == 0 {
        println!(
            "\n=== Q1 ANSWER: ✓ correct ({}/{} elements match) ===",
            N_FLOATS, N_FLOATS
        );
    } else {
        println!("\n=== Q1 ANSWER: ✗ MISMATCH {}/{} ===", mismatch, N_FLOATS);
        if let Some((i, got, exp)) = first_bad {
            println!("  first bad: c[{}]={} (expected {})", i, got, exp);
        }
    }

    // ── 14. Cleanup ──
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cp, None);
        device.destroy_descriptor_pool(dp, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pl, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_buffer(buf_a, None);
        device.destroy_buffer(buf_b, None);
        device.destroy_buffer(buf_c, None);
        device.free_memory(mem_a, None);
        device.free_memory(mem_b, None);
        device.free_memory(mem_c, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    if mismatch != 0 {
        std::process::exit(1);
    }
    Ok(())
}
