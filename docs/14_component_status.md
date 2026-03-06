# 14. Component Quality Gates

This document tracks component-level quality gates for the llm.rs (llm_rs2) inference framework. Each component is assigned a tier that determines its testing requirements and gate criteria.

> **Auto-update**: Sections 3 and 4 are automatically maintained by `scripts/update_test_status.py`.

---

## 1. Component Diagram

```mermaid
graph TB
    subgraph BinaryLayer ["Binaries"]
        Generate["generate"]
        GenerateHybrid["generate_hybrid"]
        MicroBench["micro_bench"]
        TestBackend["test_backend"]
    end

    subgraph ModelComponent ["Model"]
        LlamaModel["LlamaModel"]
        LlamaLayer["LlamaLayer"]
        LayerWS["LayerWorkspace"]
        Attention["Attention (CPU fallback)"]
    end

    subgraph CoreComponent ["Core"]
        BackendTrait["Backend trait"]
        MemoryTrait["Memory trait"]
        BufferTrait["Buffer trait"]
        KVCache["KVCache"]
        Tensor["Tensor"]
        Shape["Shape"]
        Quant["Quant (Q4_0)"]
    end

    subgraph EvictionSubsystem ["KV Cache Eviction"]
        CacheManager["CacheManager"]
        EvictionPolicy["EvictionPolicy trait"]
        NoEviction["NoEvictionPolicy"]
        SlidingWindow["SlidingWindowPolicy"]
        H2O["H2OPolicy (3-partition)"]
        SysMonitor["SystemMonitor trait"]
        LinuxMonitor["LinuxSystemMonitor"]
    end

    subgraph BackendComponent ["Compute Backends"]
        CpuBackend["CpuBackend"]
        OpenCLBackend["OpenCLBackend"]
    end

    subgraph MemoryComponent ["Memory Management"]
        Galloc["Galloc"]
        SharedBuffer["SharedBuffer"]
    end

    Generate --> LlamaModel
    GenerateHybrid --> LlamaModel
    LlamaModel --> LlamaLayer
    LlamaLayer --> LayerWS
    LlamaLayer --> Attention
    LlamaLayer --> KVCache

    LlamaModel --> CacheManager
    CacheManager --> EvictionPolicy
    CacheManager --> SysMonitor
    NoEviction -.->|impl| EvictionPolicy
    SlidingWindow -.->|impl| EvictionPolicy
    H2O -.->|impl| EvictionPolicy
    LinuxMonitor -.->|impl| SysMonitor
    EvictionPolicy --> KVCache

    Tensor --> BufferTrait
    Tensor --> BackendTrait
    Tensor --> Shape

    CpuBackend -.-> BackendTrait
    OpenCLBackend -.-> BackendTrait
    Galloc -.-> MemoryTrait
    SharedBuffer -.-> BufferTrait
    Galloc --> SharedBuffer
```

---

## 2. Quality Gate Definition

### Tier Classification

| Tier | Scope | Components | Gate Criteria |
|:-----|:------|:-----------|:--------------|
| **T1: Foundation** | Data structures, memory primitives | Shape, Tensor, Buffer/DType, Quant, SharedBuffer, Galloc | Host unit tests required, all must PASS |
| **T2: Algorithm** | Algorithms, policies, CPU-testable logic | KVCache, NoEvictionPolicy, SlidingWindowPolicy, H2OPolicy, CacheManager, SystemMonitor, Attention | Host unit tests required, all must PASS |
| **T3: Backend** | Hardware-specific backends | CpuBackend, OpenCLBackend | Device verification via `test_backend`, host N/A |
| **T4: Integration** | Model layers, GPU buffers | LlamaLayer, LayerWorkspace, LlamaModel, UnifiedBuffer | E2E device verification, host N/A |

### Gate Status

| Status | Meaning |
|:-------|:--------|
| PASS | All tests pass |
| **FAIL** | One or more tests fail |
| **BLOCKED** | T1/T2 component with zero tests — quality unknown |
| N/A | T3/T4 component — requires device, not testable on host |

### Maturity Levels

| Level | Meaning |
|:------|:--------|
| Stable | Production-ready, well-tested |
| Beta | Functional but under active development |
| Stub | Placeholder implementation |

### Overall Gate Rule

The overall gate is **FAIL** if any T1 or T2 component has status BLOCKED or FAIL. T3/T4 components are excluded from the overall gate since they require device access.

---

## 3. Component Quality Status

<!-- AUTO-GENERATED:TEST_STATUS:START -->
_Last updated: 2026-03-06 12:39:06_

### Quality Gate Summary

| Component | Tier | Maturity | Tests | Passed | Skipped | Gate |
|:----------|:-----|:---------|------:|-------:|--------:|:-----|
| Buffer/DType | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Galloc | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Quant | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Shape | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| SharedBuffer | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Tensor | T1 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Attention | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| CacheManager | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| H2OPolicy | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| KVCache | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| NoEvictionPolicy | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| OperatingMode | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| ResilienceManager | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Signal/Level | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| SlidingWindowPolicy | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| Strategy | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| SystemMonitor | T2 | Stable | 0 | 0 | 0 | **BLOCKED** |
| CpuBackend | T3 | Stable | 0 | 0 | 0 | N/A |
| OpenCLBackend | T3 | Stable | 0 | 0 | 0 | N/A |
| LayerWorkspace | T4 | Stable | 0 | 0 | 0 | N/A |
| LlamaLayer | T4 | Stable | 0 | 0 | 0 | N/A |
| LlamaModel | T4 | Stable | 0 | 0 | 0 | N/A |
| UnifiedBuffer | T4 | Stable | 0 | 0 | 0 | N/A |
| **Overall** | | | **0** | **0** | **0** | **FAIL** |

### Test Details

| Test | Component | Result |
|:-----|:----------|:------:|
<!-- AUTO-GENERATED:TEST_STATUS:END -->

---

## 4. Test History

<!-- AUTO-GENERATED:TEST_HISTORY:START -->
| Date | Total | Passed | Failed | Pass Rate |
|:-----|------:|-------:|-------:|----------:|
| 2026-03-02T23:48:47 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:48:51 | 86 | 85 | 1 | 98.8% |
| 2026-03-02T23:48:56 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:01 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:05 | 85 | 85 | 0 | 100.0% |
| 2026-03-02T23:49:10 | 85 | 85 | 0 | 100.0% |
| 2026-03-02T23:49:14 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:19 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:24 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:28 | 85 | 85 | 0 | 100.0% |
| 2026-03-02T23:49:33 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:37 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:42 | 85 | 85 | 0 | 100.0% |
| 2026-03-02T23:49:47 | 85 | 85 | 0 | 100.0% |
| 2026-03-02T23:49:52 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:49:57 | 88 | 85 | 3 | 96.6% |
| 2026-03-02T23:50:01 | 86 | 85 | 1 | 98.8% |
| 2026-03-02T23:50:06 | 86 | 85 | 1 | 98.8% |
| 2026-03-02T23:52:33 | 88 | 85 | 3 | 96.6% |
| 2026-03-06T12:39:06 | 0 | 0 | 0 | 0.0% |
<!-- AUTO-GENERATED:TEST_HISTORY:END -->
