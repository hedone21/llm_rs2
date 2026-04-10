# impl-request: Tensor Partition 확장 — 전 Linear Layer + Adaptive Cooperative Prefill

> **상태**: 요청  
> **작성**: 2026-04-07  
> **우선순위**: P0 (논문 핵심 실험 차단)  
> **선행 요구**: 없음 (현재 `9a8f37c` 기반으로 확장)  
> **참조**: HeteroInfer (SOSP '25), `arch/tensor_partition.md` (v2 설계)

---

## 배경: 왜 확장이 필요한가

### 현재 문제

S1 시나리오(PUBG + LLM 공존)에서 **SwitchHw(GPU→CPU 전환)** 실험 결과:
- 게임 FPS: −35% → −10% (GPU 해방 효과 ✓)
- LLM decode TBT: 99ms → 187ms (1.9x 악화 ✗)
- 원인: CPU에서 LLM + PUBG 게임로직이 경쟁 (c0_util: 31% → 91%)

**SwitchHw는 올-오어-낫싱**: GPU를 100% 반환하고 CPU에서 100% 부담.
Tensor partition은 이 사이의 **연속적 트레이드오프**를 제공한다.
GPU를 게임과 공유하면서 LLM이 GPU+CPU를 동시에 사용할 수 있다.

### 현재 구현 한계

| 항목 | 현재 상태 | 필요한 상태 |
|------|----------|-----------|
| Decode FFN gate/up | ✅ partitioned | 유지 |
| Decode FFN down | ❌ GPU only | partition 필요 |
| Decode Attention QKV | ❌ GPU only | partition 필요 |
| Decode Attention output (wo) | ❌ GPU only | partition 필요 |
| Prefill GPU 독점 | ❌ chunk=256, 6.7s 연속 점유 | dynamic chunk + yield + CPU interleave |
| Prefill 제어 | ❌ 정적 chunk_size만 | 동적 contention level (None/Low/High) |

현재 decode에서 gate/up만 partition하면 **전체 matmul 시간의 ~40%만** 커버.
나머지 60%(QKV, output, down)는 여전히 GPU를 100% 점유한다.
이 60% 구간에서 게임 FPS가 여전히 하락한다.

### HeteroInfer 참조 포인트

HeteroInfer(SOSP '25)는 GPU-NPU 병렬이지만, 설계 원칙이 CPU-GPU partition에 직접 적용 가능:

1. **모든 Linear layer를 partition 대상으로 삼음** — QKV, O, FFN up/gate/down 전부
2. **Prefill과 Decode에 다른 전략 적용**:
   - Prefill: NPU(=compute 강자) 주도, GPU 보조. 우리: GPU 주도, CPU 보조 (GPU가 MatMul에 강함)
   - Decode: GPU(=bandwidth 강자) 주도, NPU 보조. 우리: GPU 주도, CPU 보조 (동일 논리)
3. **Weight-centric partition**: weight를 row 방향으로 정적 분할 → 현재 llm_rs2와 동일
4. **비-Linear 연산(RMSNorm, SwiGLU, Softmax, RoPE)은 GPU 전용** → 현재 llm_rs2와 동일
5. **UMA 기반 zero-copy sync** → 현재 ClSubBuffer/SliceBuffer와 동일 아이디어
6. **Offline solver로 최적 ratio 결정** → 우리는 runtime calibration (v2 설계)으로 대체

---

## §1. Decode: 전 Linear Layer Partition 확장

### 1.1 목표

Decode (seq_len=1) 경로의 **모든 6개 Linear projection**을 partition 대상으로 확장:

```
Transformer Layer 내 Linear projections:
  Attention:  Q, K, V projections  (w_q, w_k, w_v)    ← 신규
              Output projection    (w_o)               ← 신규
  FFN:        Gate projection      (w_gate)            ✅ 기존
              Up projection        (w_up)              ✅ 기존
              Down projection      (w_down)            ← 신규
```

### 1.2 PartitionContext 확장

현재:
```rust
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    pub gate: PartitionedWeight,
    pub up: PartitionedWeight,
}
```

변경:
```rust
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    // Attention
    pub wq: PartitionedWeight,
    pub wk: PartitionedWeight,
    pub wv: PartitionedWeight,
    pub wo: PartitionedWeight,
    // FFN
    pub gate: PartitionedWeight,
    pub up: PartitionedWeight,
    pub down: PartitionedWeight,
}
```

### 1.3 PartitionWorkspace 확장

현재 `PartitionWorkspace`는 gate/up GPU/CPU partial + residual_cpu만 보유.
Attention 4개 + down 1개의 GPU/CPU partial buffer를 추가해야 한다.

```rust
pub struct PartitionWorkspace {
    // 기존 (FFN gate/up)
    pub gate_gpu: Tensor,       // [1, 1, gate_split_row] F32
    pub gate_cpu: Tensor,       // [1, 1, gate_cpu_rows] F32
    pub up_gpu: Tensor,
    pub up_cpu: Tensor,
    pub residual_cpu: Tensor,   // [1, 1, dim] F32 — CPU용 입력 복사본

    // 신규 (Attention)
    pub q_gpu: Tensor,          // [1, 1, q_split_row] F32
    pub q_cpu: Tensor,          // [1, 1, q_cpu_rows] F32
    pub k_gpu: Tensor,
    pub k_cpu: Tensor,
    pub v_gpu: Tensor,
    pub v_cpu: Tensor,
    pub o_gpu: Tensor,          // wo의 output partial
    pub o_cpu: Tensor,

    // 신규 (FFN down)
    pub down_gpu: Tensor,
    pub down_cpu: Tensor,

    // 신규 — attention output 후 wo 입력용 CPU 복사본
    pub attn_out_cpu: Tensor,   // [1, 1, dim] F32
    // 신규 — FFN down 입력용 CPU 복사본 (gelu_tanh_mul 결과)
    pub ffn_mid_cpu: Tensor,    // [1, 1, ffn_dim] F32
}
```

**메모리 추정** (Qwen 2.5 1.5B, dim=1536, ffn_dim=8960):
- Attention partials (Q/K/V/O × GPU+CPU): 8 × 1536 × 4B = ~49KB
  - 단, K/V는 GQA head 수에 따라 더 작음 (n_kv_heads=2 → dim=256)
  - Q: 1536, K: 256, V: 256, O: 1536 → (1536+256+256+1536)×2×4 = ~29KB
- FFN down partials: 2 × 1536 × 4B = ~12KB
- CPU 복사 버퍼 (attn_out_cpu, ffn_mid_cpu): (1536 + 8960) × 4B = ~42KB
- **총 추가**: ~83KB/layer × 28 layers = **~2.3MB** (무시할 수 있는 수준)

### 1.4 Decode Forward 수정 (forward_gen.rs)

#### Attention QKV partition

현재 코드 (대략 line 670~720):
```rust
// Q, K, V projections — 현재 전부 GPU
backend.matmul_transposed(&ws.residual, &self.w_q, &mut ws.q)?;
backend.matmul_transposed(&ws.residual, &self.w_k, &mut ws.k)?;
backend.matmul_transposed(&ws.residual, &self.w_v, &mut ws.v)?;
```

변경 패턴 (FFN gate/up과 동일 패턴):
```rust
if let Some(ref part) = self.partition_ctx {
    let pw = ws.partition_ws.as_mut().unwrap();

    // 0. residual → CPU 복사 (이미 FFN에서 하던 것과 동일, 위치만 이동)
    backend.synchronize()?;
    backend.read_buffer(&ws.residual, &mut pw.residual_cpu)?;

    // 1. GPU: 3개 matmul enqueue (non-blocking)
    backend.matmul_transposed(&ws.residual, &part.wq.gpu_slice, &mut pw.q_gpu)?;
    backend.matmul_transposed(&ws.residual, &part.wk.gpu_slice, &mut pw.k_gpu)?;
    backend.matmul_transposed(&ws.residual, &part.wv.gpu_slice, &mut pw.v_gpu)?;
    backend.flush()?;

    // 2. CPU: 3개 matmul blocking (GPU 병렬 실행)
    let cpu = &part.cpu_backend;
    cpu.matmul_transposed(&pw.residual_cpu, &part.wq.cpu_slice, &mut pw.q_cpu)?;
    cpu.matmul_transposed(&pw.residual_cpu, &part.wk.cpu_slice, &mut pw.k_cpu)?;
    cpu.matmul_transposed(&pw.residual_cpu, &part.wv.cpu_slice, &mut pw.v_cpu)?;

    // 3. Merge → ws.q, ws.k, ws.v
    merge_partials(backend, &pw.q_gpu, &pw.q_cpu, &mut ws.q)?;
    merge_partials(backend, &pw.k_gpu, &pw.k_cpu, &mut ws.k)?;
    merge_partials(backend, &pw.v_gpu, &pw.v_cpu, &mut ws.v)?;
} else {
    // 기존 경로 유지
}
```

**`merge_partials` 헬퍼** (반복 코드 제거):
```rust
fn merge_partials(
    backend: &dyn Backend,
    gpu_part: &Tensor,
    cpu_part: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let gpu_elems = gpu_part.size() / 4;
    let cpu_elems = cpu_part.size() / 4;
    backend.copy_slice(gpu_part, output, 0, 0, gpu_elems)?;
    backend.copy_slice(cpu_part, output, 0, gpu_elems, cpu_elems)?;
    Ok(())
}
```

#### Attention Output (wo) partition

wo projection은 attention 연산 완료 후 실행된다.
입력이 `ws.attn_out` (attention 결과)이므로 별도의 CPU 복사가 필요:

```rust
// Attention 완료 후, wo projection
if let Some(ref part) = self.partition_ctx {
    let pw = ws.partition_ws.as_mut().unwrap();

    backend.synchronize()?;
    backend.read_buffer(&ws.attn_out, &mut pw.attn_out_cpu)?;

    backend.matmul_transposed(&ws.attn_out, &part.wo.gpu_slice, &mut pw.o_gpu)?;
    backend.flush()?;

    let cpu = &part.cpu_backend;
    cpu.matmul_transposed(&pw.attn_out_cpu, &part.wo.cpu_slice, &mut pw.o_cpu)?;

    merge_partials(backend, &pw.o_gpu, &pw.o_cpu, &mut ws.x)?;  // or wherever wo output goes
}
```

#### FFN down partition

down의 입력은 `gelu_tanh_mul(gate, up)` 결과 (`ws.gate`에 in-place 저장).
이 결과를 CPU로 복사한 후 partition 실행:

```rust
// gelu_tanh_mul 완료 후
if let Some(ref part) = self.partition_ctx {
    let pw = ws.partition_ws.as_mut().unwrap();

    backend.synchronize()?;
    backend.read_buffer(&ws.gate, &mut pw.ffn_mid_cpu)?;  // gelu_tanh_mul 결과

    backend.matmul_transposed(&ws.gate, &part.down.gpu_slice, &mut pw.down_gpu)?;
    backend.flush()?;

    let cpu = &part.cpu_backend;
    cpu.matmul_transposed(&pw.ffn_mid_cpu, &part.down.cpu_slice, &mut pw.down_cpu)?;

    merge_partials(backend, &pw.down_gpu, &pw.down_cpu, &mut ws.down)?;
} else {
    backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
}
```

### 1.5 `prepare_tensor_partition` 확장

현재 `TransformerModel::prepare_tensor_partition()`은 gate/up만 split.
7개 weight 전부를 split하도록 확장:

```rust
// models/transformer.rs
pub fn prepare_tensor_partition(
    &mut self,
    gpu_ratio: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<usize> {
    let mut count = 0;
    for layer in &mut self.layers {
        let gate = split_weight(&layer.w_gate, gpu_ratio, cpu_backend)?;
        let up = split_weight(&layer.w_up, gpu_ratio, cpu_backend)?;
        let down = split_weight(&layer.w_down, gpu_ratio, cpu_backend)?;
        let wq = split_weight(&layer.w_q, gpu_ratio, cpu_backend)?;
        let wk = split_weight(&layer.w_k, gpu_ratio, cpu_backend)?;
        let wv = split_weight(&layer.w_v, gpu_ratio, cpu_backend)?;
        let wo = split_weight(&layer.w_o, gpu_ratio, cpu_backend)?;

        layer.partition_ctx = Some(PartitionContext {
            gpu_ratio,
            cpu_backend: cpu_backend.clone(),
            wq, wk, wv, wo,
            gate, up, down,
        });
        count += 7;
    }
    Ok(count)
}
```

### 1.6 GQA(Grouped Query Attention) 고려사항

Qwen 2.5 1.5B: `n_heads=12, n_kv_heads=2`.
- w_q: [1536, 1536] (12 heads × 128 dim)
- w_k: [256, 1536]  (2 kv_heads × 128 dim)
- w_v: [256, 1536]  (2 kv_heads × 128 dim)
- w_o: [1536, 1536]

w_k, w_v의 out_dim=256이므로 ROW_ALIGNMENT=128 기준으로 **최소 2개 partition만 가능**.
gpu_ratio=0.5 → split_row=128, cpu_rows=128. 이 경우 partition 효과가 크지 않을 수 있다.

**처리 방안**: out_dim < ROW_ALIGNMENT × 4 (512 미만)인 weight는 partition을 skip하고
GPU-only로 실행. `split_weight()`에서 `None`을 반환하는 옵션 추가:

```rust
pub fn split_weight_optional(
    weight: &Tensor,
    gpu_ratio: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<Option<PartitionedWeight>> {
    let out_dim = weight.shape().dims()[0];
    if out_dim < ROW_ALIGNMENT * 4 {
        return Ok(None);  // 너무 작아 partition 비효율
    }
    split_weight(weight, gpu_ratio, cpu_backend).map(Some)
}
```

`PartitionContext`에서 해당 필드를 `Option<PartitionedWeight>`로:
```rust
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    pub wq: Option<PartitionedWeight>,
    pub wk: Option<PartitionedWeight>,  // GQA에서 None일 수 있음
    pub wv: Option<PartitionedWeight>,  // GQA에서 None일 수 있음
    pub wo: Option<PartitionedWeight>,
    pub gate: PartitionedWeight,        // FFN은 항상 큼, 항상 Some
    pub up: PartitionedWeight,
    pub down: PartitionedWeight,
}
```

Forward에서 `Option` 분기:
```rust
if let Some(ref wq_part) = part.wq {
    // partitioned path
} else {
    // GPU-only fallback
    backend.matmul_transposed(&ws.residual, &self.w_q, &mut ws.q)?;
}
```

---

## §2. Adaptive Cooperative Prefill — 동적 GPU 양보 + CPU 활용

### 2.1 문제: Prefill이 FPS 하락의 주범

| 특성 | Decode | Prefill |
|------|--------|---------|
| 연산 유형 | MatVec (memory-bound) | MatMul (compute-bound) |
| GPU 점유 시간 | 짧음 (~수백μs/token) | 긺 (chunk당 **6.7초**) |
| 게임 FPS 영향 | 중간 (−35%) | **심각** (−44%, 최저 24fps) |

S1 실험에서 prefill 구간이 decode보다 FPS 하락이 심각:
- chunk_size=256 → chunk당 6.7초 GPU 연속 점유 → 게임 프레임 렌더링 불가
- GPU prefill: 34.0 tok/s, CPU prefill: 12.7 tok/s (**GPU가 2.7x 빠름**)

Tensor partition으로 prefill의 각 matmul을 분할하는 것은 복잡도가 높고
(2D merge, 동적 workspace, copy_slice_2d 커널 필요) 효과도 제한적이다.
**GPU를 "짧게 쓰고 자주 반환"하는 것이 "부분만 쓰고 계속 점유"보다 게임 FPS에 유리하다.**

### 2.2 핵심 아이디어: 3가지 메커니즘 결합

```
┌─────────────────────────────────────────────────────────────┐
│ Adaptive Cooperative Prefill                                 │
│                                                              │
│  ① Dynamic Chunk Size  — chunk 크기로 GPU 점유 상한 제어     │
│  ② Inter-layer Yield   — layer 사이에 GPU를 게임에 양보      │
│  ③ GPU-CPU Interleave  — GPU yield 동안 CPU가 prefill 계속   │
│                                                              │
│  Manager → contention level → Engine 자율 조정               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 메커니즘 ①: Dynamic Chunk Size

chunk 사이에 resilience directive 체크 + GPU command queue drain이 발생한다.
chunk 크기를 줄이면 이 gap이 더 자주 발생하여 게임이 GPU를 사용할 수 있다.

```
현재 (chunk=256): [======6.7s GPU======][gap][======6.7s======][gap]
                  게임 프레임 0~2개           게임 프레임 0~2개

chunk=64:         [=1.7s=][gap][=1.7s=][gap][=1.7s=][gap]...
                  4~5 fr   4~5 fr   4~5 fr

chunk=32:         [=0.9s=][gap][=0.9s=][gap]...
                  5~6 fr   5~6 fr
```

**구현**: `--prefill-chunk-size`는 이미 CLI로 존재. 동적 변경만 추가.

```rust
// engine/src/inference/generate.rs — prefill 루프 내부
// 현재: 고정 chunk_size로 루프
// 변경: 매 chunk 시작 전 contention level 확인하여 chunk_size 조정

let effective_chunk_size = match self.prefill_contention {
    ContentionLevel::None => args.prefill_chunk_size,  // 256 (최대 성능)
    ContentionLevel::Low  => 64,                        // 적당한 양보
    ContentionLevel::High => 32,                        // 자주 양보
};
```

### 2.4 메커니즘 ②: Inter-layer Yield

chunk 안에서도 각 transformer layer 처리 후 GPU를 yield한다.
**chunk 크기를 줄이지 않고도** 게임 프레임을 확보할 수 있다.

```
Chunk 64 tokens, 28 layers:

yield 없음:   [L0 L1 L2 ... L27] = 1.7s 연속 GPU 점유

yield 5ms:    [L0]·[L1]·[L2]·...·[L27]  (·=5ms yield)
              = 1.7s + 28×5ms = 1.84s (+8%)
              매 60ms마다 게임 프레임 1개 → ~17fps 유지

yield 10ms:   [L0]··[L1]··[L2]··...··[L27]
              = 1.7s + 28×10ms = 1.98s (+16%)
              매 70ms마다 게임 프레임 1~2개 → ~14fps 유지
```

**구현 위치**: `TransformerModel::forward()` 또는 `forward_gen()` 내부,
layer 루프의 끝에 yield 삽입.

```rust
// models/transformer.rs — forward() 내 layer 루프
for (i, layer) in self.layers.iter().enumerate() {
    layer.forward(backend, ws, ...)?;

    // Inter-layer yield: prefill 중에만, yield_ms > 0일 때만
    if !is_decode && self.prefill_yield_ms > 0 {
        backend.synchronize()?;  // GPU 커널 완료 대기
        std::thread::sleep(Duration::from_millis(self.prefill_yield_ms as u64));
    }
}
```

**동적 조정**: yield 시간도 contention level에 따라 변경.

```rust
let yield_ms = match self.prefill_contention {
    ContentionLevel::None => 0,   // yield 없음 (최대 성능)
    ContentionLevel::Low  => 5,   // 5ms (매 layer 후)
    ContentionLevel::High => 10,  // 10ms (더 긴 양보)
};
```

### 2.5 메커니즘 ③: GPU-CPU Chunk Interleaving

**핵심 아이디어**: GPU가 게임에 양보하는 동안, CPU가 다음 chunk을 처리.
GPU와 CPU가 **chunk 단위로 교대**하여 prefill을 진행한다.

```
전체 1073 tokens을 GPU chunk과 CPU chunk으로 교대 처리:

[GPU chunk 48 tok] → [CPU chunk 16 tok] → [GPU chunk 48 tok] → ...
  GPU: 28 layers      GPU: 게임 전용!       GPU: 28 layers
  CPU: 유휴            CPU: 28 layers        CPU: 유휴
  ~1.4s                ~1.3s                 ~1.4s

Timeline:
  GPU: ████████░░░░░░░░████████░░░░░░░░████████
  CPU: ░░░░░░░░████████░░░░░░░░████████░░░░░░░░
  Game:░░░░░░░░████████░░░░░░░░████████░░░░░░░░
       ↑ GPU turn       ↑ CPU turn (GPU 100% free for game)
```

**장점**:
- GPU turn 동안: GPU가 대형 MatMul로 고속 prefill (34 tok/s)
- CPU turn 동안: **게임이 GPU 100% 사용** (FPS 완전 복구), CPU가 저속 prefill (12.7 tok/s)
- 기존 SwitchHw 인프라 재사용 (backend 전환 불필요 — 그냥 CPU backend로 forward 호출)
- Attention 토큰 간 의존성 문제 없음 — 각 chunk가 독립적으로 전 layer 통과

**구현**: prefill 루프에서 GPU chunk과 CPU chunk을 번갈아 실행.

```rust
// generate.rs — prefill 루프
let mut pos = 0;
while pos < prompt_len {
    // GPU chunk
    let gpu_end = (pos + gpu_chunk_size).min(prompt_len);
    if gpu_end > pos {
        let chunk = &prompt_tokens[pos..gpu_end];
        model.forward(gpu_backend, chunk, pos, ...)?;  // GPU prefill
        pos = gpu_end;
    }

    // CPU chunk (GPU가 게임에 양보하는 동안)
    let cpu_end = (pos + cpu_chunk_size).min(prompt_len);
    if cpu_end > pos && cpu_chunk_size > 0 {
        let chunk = &prompt_tokens[pos..cpu_end];
        model.forward(cpu_backend, chunk, pos, ...)?;  // CPU prefill
        pos = cpu_end;
    }
}
```

**KV cache 호환성**: prefill의 각 chunk는 KV cache에 순차적으로 append된다.
GPU chunk이든 CPU chunk이든 결과적으로 동일한 KV cache에 기록되므로 정확성에 영향 없다.
단, **KV cache가 GPU 메모리에 있을 때 CPU chunk의 KV write는 UMA를 통해 직접 접근** 가능
(ClWrappedBuffer, CL_MEM_USE_HOST_PTR).

**CPU forward 호환**: 현재 `TransformerLayer::forward()`는 backend argument로
GPU 또는 CPU를 받을 수 있다. Weight tensor가 dual-access(ClWrappedBuffer)이므로
CPU backend에서도 동일 weight에 접근 가능하다.
SwitchHw 구현에서 이미 검증된 경로 — `--resilience-prealloc-switch`로
weight가 CPU/GPU 양쪽에서 접근 가능하도록 re-wrap됨.

### 2.6 수치 예상: Interleaving 효과

```
GPU-only:      1073 tok / 34 tok/s = 31.6s,  FPS drop: −44%
CPU-only:      1073 tok / 12.7 tok/s = 84.5s, FPS drop:  −0%

Interleaved (GPU 48 + CPU 16 = 64 tok/cycle):
  GPU phase: 48/34 = 1.41s  (FPS −44%)
  CPU phase: 16/12.7 = 1.26s (FPS ~0%, GPU free)
  Cycle: 2.67s per 64 tokens
  Total: 1073/64 × 2.67 = 44.8s (+42%)
  시간 가중 평균 FPS drop: 53% × (−44%) + 47% × 0% = −23%
```

GPU:CPU 비율 조정으로 FPS/속도 트레이드오프를 제어:

| GPU:CPU 비율 | 총 Prefill 시간 | 평균 FPS drop | 비고 |
|-------------|----------------|--------------|------|
| 100:0 (현재) | 31.6s | −44% | 게임 불가 |
| 75:25 (48:16) | 44.8s (+42%) | −23% | 균형 |
| 67:33 (42:21) | 47.2s (+49%) | −20% | 게임 우선 |
| 50:50 (32:32) | 56.2s (+78%) | −15% | 게임 중심 |
| 0:100 (SwitchHw) | 84.5s (+167%) | −0% | 최대 양보 |

### 2.7 종합 설계: ContentionLevel에 따른 3단계

```rust
/// Prefill contention 제어 레벨 (Manager → Engine)
pub enum PrefillContention {
    /// 게임 없음: 최대 성능
    /// chunk=256, yield=0, CPU chunk=0
    None,

    /// 경미한 경쟁: chunk 축소 + yield
    /// chunk=64, yield=5ms, CPU chunk=0
    Low,

    /// 심각한 경쟁: chunk 축소 + yield + CPU interleave
    /// GPU chunk=48, yield=10ms, CPU chunk=16
    High,
}
```

각 레벨의 실행 패턴:

```
None:
  [=====256 tok GPU=====][=====256 tok GPU=====]...
  Total: 31.6s, FPS: −44%

Low:
  [=64 GPU=]·[=64 GPU=]·[=64 GPU=]·...  (·=yield 5ms×28 layers)
  Total: ~35s, FPS: ~−30%

High:
  [48 GPU+yield]→[16 CPU (GPU free)]→[48 GPU+yield]→[16 CPU]→...
  Total: ~45s, FPS: ~−23%
```

### 2.8 Manager 통합: EngineCommand 확장

```rust
// shared/src/lib.rs — EngineCommand에 추가
/// Prefill 중 GPU 양보 레벨 설정
SetPrefillContention { level: PrefillContention },
```

**SwitchHw와의 관계**:
- `SwitchHw { device: "cpu" }` 수신 시: CPU-only prefill (interleave 비활성)
- `SetPrefillContention { High }` 수신 시: GPU-CPU interleave 활성
- 두 커맨드는 상호 배타: SwitchHw가 활성이면 contention 무시

**Lua policy 예시**:
```lua
function decide(ctx)
    local actions = {}

    if ctx.engine.is_prefilling then
        local gpu_busy = sys.gpu_busy()
        if gpu_busy > 60 then
            table.insert(actions, { type = "set_prefill_contention", level = "high" })
        elseif gpu_busy > 30 then
            table.insert(actions, { type = "set_prefill_contention", level = "low" })
        else
            table.insert(actions, { type = "set_prefill_contention", level = "none" })
        end
    end

    return actions
end
```

### 2.9 CLI

```
--prefill-contention <none|low|high>    # 정적 설정 (기본: none)
--prefill-chunk-size <N>                # 기존, Low/None에서 사용
--prefill-gpu-chunks <N>                # High에서 GPU chunk 크기 (기본: 48)
--prefill-cpu-chunks <N>                # High에서 CPU chunk 크기 (기본: 16)
```

### 2.10 `is_prefilling` 상태 보고

Engine이 현재 prefill 중인지를 Manager에 보고해야 contention level을 동적 조절할 수 있다.

```rust
// EngineStatus에 추가
#[serde(default)]
pub is_prefilling: bool,
```

prefill 시작 시 `true`, decode 시작 시 `false`로 전환.
Heartbeat에 포함되어 Manager에 전달.

---

## §3. `split_weight` 최적화: 모든 weight 동시 split

현재 `split_weight`는 weight 하나당 호출. 7개 weight를 split하면 7번의
ClSubBuffer 생성 + SliceBuffer 생성이 발생한다.

**최적화**: 불필요. ClSubBuffer/SliceBuffer는 parent buffer의 view이므로
추가 메모리 할당이 없다. 7개 split의 총 overhead는 무시 가능.

그러나 **에러 처리 일관성**을 위해 `split_all_weights` 헬퍼를 제공:

```rust
pub fn split_all_weights(
    layer: &TransformerLayer,
    gpu_ratio: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<PartitionContext> {
    Ok(PartitionContext {
        gpu_ratio,
        cpu_backend: cpu_backend.clone(),
        wq: split_weight_optional(&layer.w_q, gpu_ratio, cpu_backend)?,
        wk: split_weight_optional(&layer.w_k, gpu_ratio, cpu_backend)?,
        wv: split_weight_optional(&layer.w_v, gpu_ratio, cpu_backend)?,
        wo: split_weight_optional(&layer.w_o, gpu_ratio, cpu_backend)?,
        gate: split_weight(&layer.w_gate, gpu_ratio, cpu_backend)?,
        up: split_weight(&layer.w_up, gpu_ratio, cpu_backend)?,
        down: split_weight(&layer.w_down, gpu_ratio, cpu_backend)?,
    })
}
```

---

## §4. Sync 최적화: read_buffer 호출 최소화

### 문제

현재 gate/up partition에서 `read_buffer` (GPU→CPU 복사)가 1회 발생.
§1 확장 후에는 **3회** 발생:
1. Attention QKV 전: `residual → residual_cpu`
2. Attention wo 전: `attn_out → attn_out_cpu`
3. FFN down 전: `gate(=gelu_tanh_mul 결과) → ffn_mid_cpu`

각 `read_buffer`는 `synchronize()` + DMA 복사 → ~100-200μs overhead.

### 해결: 조건부 복사 + 재사용

```
Token 실행 흐름:
  RMSNorm(x) → residual
  ┌─ QKV partition (residual → CPU 복사 ①)
  │  Attention (GPU only: RoPE, softmax, ...)
  │  wo partition (attn_out → CPU 복사 ②)
  └─ Residual add
  RMSNorm → residual
  ┌─ gate/up partition (residual → CPU 복사 ③, ①과 다른 값!)
  │  gelu_tanh_mul (GPU)
  │  down partition (gate → CPU 복사 ④)
  └─ Residual add
```

**최적화**:
- ① `residual_cpu`를 QKV에서 복사, FFN gate/up에서도 재사용? → **불가**. 두 번째 RMSNorm 후 값이 다름.
- ② 와 ③ 은 반드시 각각 필요.
- ④ `gelu_tanh_mul` 결과는 GPU에서 in-place 계산 → CPU 복사 필수.

**결론**: 4회 read_buffer는 피할 수 없지만, 각각 **작은 크기**:
- ① ③: dim × 4B = 1536 × 4 = 6KB
- ②: dim × 4B = 6KB
- ④: ffn_dim × 4B = 8960 × 4 = 35KB

총 ~53KB/token. DMA 속도 ~40GB/s 기준 **~1.3μs**. 무시 가능.
실제 bottleneck은 `synchronize()` (~100μs). 이 횟수를 줄이는 것이 핵심.

**synchronize 최소화 전략**: 
- ① 전에 1회 sync (RMSNorm 완료 보장)
- ② 전에 sync 불필요 — attention 연산이 이미 in-order queue에서 완료
  → `read_buffer` 자체가 in-order queue에서 이전 커널 완료를 보장
- ③ 전에 sync 불필요 — 동일 이유
- ④ 전에 sync 불필요 — gelu_tanh_mul이 같은 queue에서 실행

**결론**: `backend.synchronize()`는 **layer 시작 시 1회만** 필요.
나머지는 in-order queue 특성을 활용하여 `read_buffer` 만으로 충분.

---

## §5. 구현 순서 및 검증

### Phase 1: Decode 전 Linear (§1) — P0

1. `PartitionContext` 7개 weight 확장 + `split_all_weights` 함수
2. `PartitionWorkspace` 확장 (attention + down partials)
3. `forward_gen.rs` 수정:
   - QKV partition 분기
   - wo partition 분기
   - down partition 분기
   - `merge_partials` 헬퍼 추출
4. `prepare_tensor_partition` 7개 weight split
5. GQA skip 로직 (out_dim < 512 → None)

**검증**:
```bash
# Smoke test: 텍스트 생성 정확성 (partition 유/무 비교)
./generate -m qwen2.5-1.5b --tensor-partition 0.5 -n 20 --greedy
# 출력이 --tensor-partition 0.0 (GPU only)과 동일해야 함

# 성능: decode TBT 비교
./generate -m qwen2.5-1.5b --tensor-partition 0.6 --profile --profile-probes latency -n 100
# partition 없는 GPU-only 대비 TBT 변화 확인
```

### Phase 2: Adaptive Cooperative Prefill (§2) — P1

#### Phase 2a: Dynamic Chunk Size + Inter-layer Yield (1~2일)

1. `PrefillContention` enum 정의 (`None` / `Low` / `High`)
2. `--prefill-contention` CLI 추가
3. Prefill 루프에서 `effective_chunk_size` 동적 결정
4. Layer 루프에 inter-layer yield 삽입 (`synchronize() + sleep`)
5. `is_prefilling` 상태를 EngineStatus에 추가

**검증**:
```bash
# 정확성: chunk_size 변경이 출력에 영향 없음 확인
./generate -m qwen2.5-1.5b --prefill-chunk-size 32 -n 20 --greedy
# chunk_size=256과 동일 출력

# FPS 영향: S25에서 PUBG + prefill 구간 FPS 측정
bash pilot_s1_switchhw.sh pubg 300  # --prefill-contention low
bash pilot_s1_switchhw.sh pubg 300  # --prefill-contention high
```

#### Phase 2b: GPU-CPU Chunk Interleaving (2~3일)

1. Prefill 루프에 CPU chunk 교대 실행 로직 추가
2. CPU backend로 `model.forward()` 호출 경로 검증 (SwitchHw에서 검증된 경로 재사용)
3. KV cache에 GPU/CPU chunk 결과가 올바르게 append되는지 검증
4. `--prefill-gpu-chunks`, `--prefill-cpu-chunks` CLI 추가
5. `SetPrefillContention` EngineCommand 추가 (Manager 연동)

**검증**:
```bash
# 정확성: interleave 결과가 GPU-only와 동일한지 확인
./generate -m qwen2.5-1.5b --prefill-contention high -n 20 --greedy
# GPU-only와 동일 출력

# E2E: S25에서 PUBG + 전체 시나리오 (prefill + decode)
# FPS 시계열, prefill 시간, decode TBT 모두 측정
```

### Phase 3: 동적 ratio 제어 (v2 설계) — P2

`tensor_partition.md` v2 설계(SetGpuBudget, Calibration, RatioController)는
Phase 1-2 완료 후 별도 진행. 논문 제출 전에는 **정적 ratio + 정적 contention level**로 충분.

---

## 수치 예상

### Decode: §1 Partition 효과 (S1 시나리오, PUBG + Qwen 2.5 1.5B)

| 구성 | GPU busy (LLM) | 게임 FPS drop | Decode TBT |
|------|---------------|--------------|------------|
| GPU only (현재 Mode A) | 59% | −35% | 137ms |
| SwitchHw (현재 Mode C) | 0% | −10% | 187ms |
| **Partition 0.5 (gate/up only)** | ~45% | ~−25% | ~120ms |
| **Partition 0.5 (전 Linear, §1)** | ~25-30% | ~−15% | ~110-130ms |
| **Partition 0.3 (전 Linear)** | ~15-20% | ~−10% | ~140-160ms |

### Prefill: §2 Cooperative Prefill 효과

| 구성 | Prefill 시간 | Prefill FPS drop | 비고 |
|------|-------------|-----------------|------|
| GPU only (현재) | 31.6s | −44% | chunk=256 |
| **Low** (chunk=64, yield=5ms) | ~35s (+10%) | ~−30% | 자주 양보 |
| **High** (GPU 48 + CPU 16 interleave) | ~45s (+42%) | ~−23% | CPU 활용 |
| **High** (GPU 32 + CPU 32 interleave) | ~56s (+78%) | ~−15% | 게임 중심 |
| CPU only (SwitchHw) | 84.5s (+167%) | −0% | 최대 양보 |

### 종합: §1 + §2 결합 시

| Phase | 구성 | FPS drop | 시간 |
|-------|------|---------|------|
| Prefill | High contention (GPU 48 + CPU 16) | −23% | ~45s |
| Decode | Partition 0.4 (전 Linear) | −12% | ongoing |
| **가중 평균** | | **~−15%** | |

**핵심**: §1(decode partition) + §2(cooperative prefill)을 결합하면
prefill/decode 모든 구간에서 게임 FPS를 15% 이내로 보호하면서
LLM 성능도 수용 가능한 수준으로 유지할 수 있다.

---

## Architect 검토 의견

> **검토자**: Architect Agent  
> **검토일**: 2026-04-07 (개정판 재검토)  
> **판정**: **Approve with conditions**

### 1. 전체 판정

**§1(Decode 전 Linear partition)**: 이전 검토와 동일하게 **승인**. 기존 gate/up 패턴의 자연스러운 확장이며, 기술적 리스크가 낮다.

**§2(Adaptive Cooperative Prefill)**: 이전 Prefill Partition 대비 **대폭 개선된 설계**. 3가지 메커니즘의 단계적 구조가 합리적이며, 구현 복잡도가 현저히 낮다. 단, GPU-CPU Interleave(메커니즘 3)는 KV cache 정합성 검증이 필수 조건이다.

**§3, §4**: 변경 없음. 이전 검토 결론 유지.

**조건부 승인 사항**:
1. §1: `forward_gen.rs` 코드 복잡도 관리 -- partition 로직을 별도 함수로 분리 (이전과 동일)
2. §1: GQA skip 임계값 -- 프로파일링 기반 결정 (이전과 동일)
3. §2: GPU-CPU Interleave 착수 전, KV cache UMA write 안전성 검증 테스트 작성 필수
4. §2: `is_prefilling` 추가 시 shared 크레이트 하위호환성 확보 (`#[serde(default)]`)

---

### 2. §1 Decode 전 Linear Partition -- 검토 (간략)

이전 검토 결론을 유지한다. 핵심 요약:

- **기술적 타당성**: 적합. forward_gen.rs:883~924의 gate/up 패턴을 QKV/wo/down에 동일하게 적용.
- **수학적 정확성**: matmul 선형성에 의해 `concat(x*W_top^T, x*W_bot^T) = x*W^T` 보장.
- **wo 변수명 주의**: impl-request는 `ws.attn_out`으로 기술했으나, 실제 코드에서 wo의 **입력**은 `ws.out_attn`, **출력**은 `ws.attn_out`이다 (forward_gen.rs:844). 구현 시 혼동 주의.
- **GQA skip**: 초기 구현에서는 `split_weight` 호출 가능 여부(out_dim >= 256)만으로 판단. skip 임계값은 프로파일링 후 튜닝.
- **PartialPair 구조체**: `PartitionWorkspace` 확장 시 15개 이상 필드를 관리하려면 `PartialPair { gpu: Tensor, cpu: Tensor }` 도입 권장.

세부 분석은 이전 검토(§2.1~2.6)를 참조하라.

---

### 3. §2 Adaptive Cooperative Prefill -- 집중 검토

#### 3.1 설계 철학 전환에 대한 평가: **적절**

이전 §2(Prefill Partition)의 핵심 문제는 "각 matmul을 GPU/CPU로 분할"하는 접근이었다.
Prefill에서 matmul은 `[seq_len, dim] x [out_dim, dim]^T`이므로 output이 2D이고,
merge에 `copy_slice_2d` 커널이 필요했다.

개정된 §2는 이 접근을 포기하고, **chunk 단위로 GPU/CPU를 교대**하는 방식으로 전환했다.
이는 "부분적으로 GPU를 사용"하는 대신 "짧게 GPU를 점유하고 자주 반환"하는 전략이며,
게임 FPS 보호 관점에서 더 효과적이다.

| 비교 | 기존 Prefill Partition | Adaptive Cooperative Prefill |
|------|----------------------|------------------------------|
| 구현 복잡도 | 높음 (2D merge, 커널) | **낮음** (루프 제어, sleep) |
| 새 커널 필요 | copy_slice_2d 필수 | **없음** |
| 정확도 리스크 | 2D merge 버그 가능 | **없음** (기존 forward 재사용) |
| FPS 보호 방식 | GPU 일부 사용 (연속 점유) | **GPU 전체 반환** (교대 점유) |
| Prefill 시간 | ~0% overhead | +10~78% (contention level 의존) |

**핵심 판단**: prefill 시간 증가(+10~78%)는 게임 공존 시나리오에서 수용 가능한 트레이드오프이다.
GPU를 부분 점유하면서 연속 실행하는 것보다, GPU를 완전 반환하고 CPU로 진행하는 것이
게임 프레임 렌더링에 더 유리하다. 이 설계 전환에 **동의한다**.

#### 3.2 메커니즘 1: Dynamic Chunk Size -- **즉시 적용 가능**

**기존 인프라 활용도**: **높음**.

- `--prefill-chunk-size` CLI는 이미 존재하고, generate.rs:2080~2086에서 사용 중이다.
- 현재 코드에서 `chunk_size`는 prefill 루프 시작 전에 한 번 결정되고 고정된다.
- 동적 변경은 `while chunk_start < process_len` 루프(generate.rs:2113) 내부에서 매 iteration마다 `effective_chunk_size`를 재계산하면 된다.
- **변경 범위**: generate.rs의 prefill 루프 내부 ~5줄. 기존 코드 구조를 거의 건드리지 않는다.

**FPS 효과 예상**: chunk 크기 축소의 FPS 효과는 **chunk 사이의 gap 빈도**에 의존한다.
현재 chunked prefill에서 chunk 사이에 `backend.synchronize()` (generate.rs:2153) + resilience checkpoint poll (generate.rs:2161~2222)이 있다. 이 gap은 ~1-5ms 정도이므로:

- chunk=256 (현재): gap 빈도 = 1073/256 = ~4회, 총 gap ~4-20ms. 게임이 GPU를 확보할 기회 거의 없음.
- chunk=64: gap 빈도 = ~17회, 총 gap ~17-85ms. 1-5 프레임 렌더링 가능.
- chunk=32: gap 빈도 = ~34회, 총 gap ~34-170ms. 2-10 프레임 렌더링 가능.

**결론**: chunk 축소만으로도 FPS 개선이 기대되며, 구현 비용이 거의 0이다.
다만 chunk가 너무 작으면 prefill throughput이 저하된다 (커널 launch overhead 증가).
**chunk=64가 실용적 하한**이다.

#### 3.3 메커니즘 2: Inter-layer Yield -- **효과적이나 tuning 주의**

**synchronize() 비용**:

impl-request에서 `synchronize() + sleep()` 패턴을 제안한다. 이에 대한 분석:

- `synchronize()`는 OpenCL backend에서 `clFinish()`를 호출한다. Adreno에서 ~50-150us.
- 28 layers x yield_ms=5: 28 x (5ms + ~0.1ms sync) = ~143ms 추가 per chunk.
- chunk=64 기준 1 chunk의 GPU 시간 ~1.7s이므로, +143ms = +8.4%. 합리적.

**yield_ms tuning 관련 우려**:

- yield_ms=5에서 실제 GPU idle 시간이 5ms라 해도, 게임이 이 5ms 안에 프레임을 렌더링하려면 GPU context switch + 프레임 제출 + VSync 대기가 필요하다.
- 일반적으로 모바일 게임의 GPU 프레임 시간은 ~10-16ms (60fps 기준). 5ms yield로는 프레임 1개를 완성하기 어렵다.
- **권장**: yield_ms의 기본값을 **16** (1 프레임 시간)으로 설정하고, 실측으로 최적값을 찾을 것. impl-request의 5ms/10ms는 보수적이다.

**구현 위치에 대한 의견**:

impl-request에서 `TransformerModel::forward()` 내 layer 루프에 yield를 삽입하자고 제안한다.
그러나 `forward_into()` (transformer.rs:1207~1315)의 layer 루프는 prefill과 decode 모두를 처리한다.
`seq_len > 1` 조건으로 prefill만 분기할 수 있지만, 더 깔끔한 방법은 **generate.rs의 chunked prefill 루프에서 model.forward_into() 호출 전후에 yield를 삽입**하는 것이다.
이렇게 하면 model 코드가 yield 로직에 오염되지 않는다.

```
현재:  chunk → forward_into → synchronize → resilience poll → 다음 chunk
제안:  chunk → forward_into → synchronize → yield(sleep) → resilience poll → 다음 chunk
```

단, 이 경우 yield는 chunk 간(~64 tokens 단위)에서만 발생하고, 레이어 간은 아니다.
레이어 간 yield가 필요한 이유는 chunk 크기가 클 때(예: 256) 하나의 chunk 내에서도
GPU를 너무 오래 점유하기 때문이다. **chunk=64 이하에서는 chunk 간 yield만으로 충분할 수 있다.**

**권장**: Phase 2a에서는 **chunk 간 yield만 구현**하고, 실측에서 부족하면 레이어 간 yield를 추가하라.
레이어 간 yield는 model 코드에 contention 인식을 주입해야 하므로 구현 비용이 더 높다.

#### 3.4 메커니즘 3: GPU-CPU Chunk Interleaving -- **집중 검증 필요**

이 메커니즘이 §2의 핵심 혁신이자 최대 리스크 요소이다. 아래 4가지 검증 포인트를 상세 분석한다.

##### 3.4.1 KV cache write 안전성 (검증 포인트 1)

**질문**: CPU chunk이 MadviseableGPUBuffer 기반 KV cache에 UMA로 직접 write할 때,
GPU가 이전 chunk의 KV를 아직 캐시에 가지고 있으면 race condition이 발생하는가?

**코드 분석 결과**: **안전하다 (조건 충족 시)**.

근거:

1. `MadviseableGPUBuffer`는 `CL_MEM_USE_HOST_PTR`로 생성된다 (madviseable_gpu_buffer.rs:40).
   이는 app이 소유한 host memory를 OpenCL이 접근하는 방식이다.

2. KV cache update (kv_cache.rs:586~617)는 `can_direct_copy` 경로에서
   `std::ptr::copy_nonoverlapping`으로 직접 host memory에 write한다.
   GPU buffer의 `as_mut_ptr()`가 유효하므로 (madviseable_gpu_buffer.rs:73~74),
   CPU에서 직접 write가 가능하다.

3. **핵심 조건**: GPU chunk 완료 후 반드시 `synchronize()`를 호출해야 한다.
   이는 현재 chunked prefill 코드에서 이미 수행된다 (generate.rs:2153: `backend.synchronize()?`).
   `synchronize()` 후에는 GPU가 해당 KV buffer에 대한 모든 read/write를 완료했으므로,
   CPU가 **다른 position range에** write하는 것은 안전하다.

4. **ARM 캐시 비일관성 문제**: `CL_MEM_USE_HOST_PTR` 버퍼에서 CPU write 후 GPU가
   해당 데이터를 읽으려면, GPU 커널 dispatch 전에 캐시가 일관성 있어야 한다.
   OpenCL 드라이버는 `clEnqueueNDRangeKernel` 호출 시 필요한 캐시 flush/invalidate를
   자동으로 수행한다 (OpenCL 1.2 spec, Section 5.12).
   따라서 CPU chunk이 KV에 write한 후 GPU chunk이 해당 KV를 읽을 때 정합성이 보장된다.

5. **다만 주의점**: CPU chunk이 write하는 position range와 GPU chunk이 읽는 position range가
   **겹치지 않아야** 한다. Prefill에서 각 chunk은 자신의 position range에만 write하고,
   attention은 `[0..current_pos]` 전체를 읽는다. CPU chunk 이후 GPU chunk의 attention이
   CPU가 write한 KV를 읽게 되는데, 이때 OpenCL 드라이버가 캐시 동기화를 보장해야 한다.
   이는 위 4번에서 설명한 대로 보장된다.

**결론**: `synchronize()` 후 CPU write는 안전하다. 단, **반드시 디바이스 E2E 테스트로 검증**할 것.
ARM SoC 벤더별로 `CL_MEM_USE_HOST_PTR`의 캐시 일관성 구현이 다를 수 있다.

##### 3.4.2 CPU forward 경로 호환성 (검증 포인트 2)

**질문**: SwitchHw에서는 `migrate_kv_caches()` + workspace 교체를 수행하는데,
Interleave에서는 KV migrate 없이 CPU가 GPU KV buffer에 직접 write한다. 호환되는가?

**코드 분석 결과**: **조건부 호환**.

SwitchHw에서 CPU forward가 동작하려면:
1. **Weight 접근**: `rewrap_weights_for_dual_access()` 후 weight는 ClWrappedBuffer
   (CL_MEM_USE_HOST_PTR)이므로 CPU `as_ptr()`가 유효하다. -- 이미 충족.
2. **Workspace**: SwitchHw에서는 spare CPU workspace로 교체한다 (generate.rs:3167~3170).
   Interleave에서는 **prefill workspace가 GPU backend로 할당**되어 있다.
   CPU backend로 `forward_prefill()`을 호출하면, workspace tensor의 backend가 GPU인 채로
   CPU matmul이 실행된다. `CpuBackend::matmul_transposed`는 `as_ptr()`로 데이터에 접근하는데,
   GPU workspace의 `as_ptr()`가 유효한지가 핵심이다.

**여기서 문제 발생 가능**:

- `PrefillWorkspace`는 `TransformerModel::forward_into()` 내부에서 **현재 backend의 memory로 할당**된다 (transformer.rs:1189: `PrefillWorkspace::new(&ws_cfg, seq_len, memory, backend.clone())`).
- GPU backend에서 할당하면 `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR)가 사용된다.
- `UnifiedBuffer`의 `as_ptr()`는 **map이 활성화된 경우에만** 유효하다. Map이 없으면 null을 반환할 수 있다.
- SwitchHw에서는 `retag_backend()`로 workspace를 re-tag하거나 spare workspace로 교체하여 이 문제를 회피한다.

**해결 방안**:

Interleave에서 CPU chunk 실행 시:
1. `forward_into()`를 CPU backend로 호출하면 새 PrefillWorkspace가 **CPU memory로 할당**된다 (transformer.rs:1177: `if seq_len > 1 && backend.is_gpu()` -- CPU backend이면 이 조건 false, prefill_ws=None).
2. prefill_ws=None이면 forward_prefill이 아닌 **fallback forward path** (transformer.rs:1271)가 실행되며, 이 경로에서는 매 layer마다 임시 버퍼를 할당한다.
3. 이 fallback path는 기능적으로 올바르지만, **매 layer마다 alloc/free** 하므로 성능이 저하된다.

**권장**: CPU chunk에서는 별도의 CPU PrefillWorkspace를 **사전 할당**하고 재사용하라.
또는 `PrefillWorkspace`를 CPU memory로 한 번 할당하여 CPU chunk 전용으로 유지.
이는 SwitchHw의 spare workspace 패턴과 동일한 접근이다.

##### 3.4.3 SwitchHw 인프라 재사용 가능성

impl-request에서 "SwitchHw 인프라 재사용"을 언급하지만, 실제로 재사용되는 부분은 제한적이다:

| SwitchHw 인프라 | Interleave에서 재사용 | 비고 |
|----------------|---------------------|------|
| `rewrap_weights_for_dual_access()` | **O** (이미 tensor-partition에서 활성화) | weight CPU 접근 보장 |
| `migrate_kv_caches()` | **X** (KV는 GPU buffer에 유지) | 불필요 |
| spare workspace 교체 (swap) | **부분** (별도 CPU prefill workspace 필요) | 동일 개념, 다른 구현 |
| backend/memory 변수 교체 | **X** (교대 실행이므로 명시적 backend 전달) | generate.rs에서 처리 |

실질적으로 새로 구현해야 하는 것:
- CPU chunk 전용 PrefillWorkspace 사전 할당
- generate.rs prefill 루프에서 GPU/CPU chunk 교대 실행 로직
- CPU backend로 `model.forward_into()` 호출 시 올바른 arguments 전달

##### 3.4.4 동적 chunk_size 변경의 정확성 (검증 포인트 3)

**질문**: prefill 중간에 chunk 크기가 바뀌어도 KV cache position이 올바르게 연속되는가?

**코드 분석 결과**: **안전하다**.

- 현재 prefill 루프(generate.rs:2113~2226)에서 `chunk_start`는 매 iteration마다 `chunk_end`로 갱신된다.
- `chunk_start_pos = start_pos + chunk_start` (generate.rs:2132)가 RoPE position으로 사용된다.
- KV cache의 `update()`는 내부적으로 `current_pos`를 `seq_len`만큼 증가시킨다.
- chunk 크기가 바뀌어도 `chunk_start`의 연속성은 유지되므로, KV position은 올바르다.

**결론**: 동적 chunk_size 변경은 안전하다. 추가 검증 불필요.

##### 3.4.5 ContentionLevel 설계 적절성

`PrefillContention` enum (None/Low/High) 3단계 설계에 대한 검토:

**장점**:
- 단순하고 이해하기 쉽다.
- Manager에서 GPU 사용률 기반으로 discrete level을 결정하는 것이 연속값보다 안정적이다.
- 각 level의 동작이 명확하게 정의되어 있다.

**우려**:
- 3단계가 **충분한 granularity를 제공하는가**? Low(chunk=64, yield=5ms)와 High(GPU 48 + CPU 16 + yield=10ms)의 차이가 크다. 그 사이에 중간 단계가 필요할 수 있다.
- **High level에서 GPU chunk과 CPU chunk 크기가 하드코딩**되어 있다. `--prefill-gpu-chunks`와 `--prefill-cpu-chunks` CLI로 오버라이드 가능하지만, Manager 동적 제어 시 이 값을 매번 전달하기 어렵다.

**권장**:
- 초기 구현에서는 3단계로 충분하다. 실측 후 필요하면 Medium 추가.
- High level의 GPU:CPU chunk 비율을 `PrefillContention` 내에 포함시키는 것을 고려하라:
  ```rust
  High { gpu_chunks: usize, cpu_chunks: usize }
  ```
  이렇게 하면 Manager가 비율을 동적으로 조정할 수 있다.
  단, 단순성을 위해 초기에는 하드코딩 기본값으로 시작해도 된다.

#### 3.5 수치 예상에 대한 검증

impl-request의 수치 예상(§2.6)을 검토한다:

- GPU-only prefill: 34 tok/s -- 이전 S1 실험 데이터에 기반. **타당**.
- CPU-only prefill: 12.7 tok/s -- SwitchHw 실험 데이터. **타당**.
- Interleaved (GPU 48 + CPU 16): 총 44.8s (+42%) -- 계산이 정확하다.
- 시간 가중 평균 FPS drop: -23% -- GPU phase 53%, CPU phase 47%로 계산. **타당**.

**주의**: 이 수치는 GPU turn과 CPU turn이 **완전히 비중첩**이라고 가정한다.
실제로는 CPU turn 동안 GPU가 게임 렌더링을 하므로, CPU prefill이 메모리 bandwidth를
소비하여 게임 FPS에 미미한 영향을 줄 수 있다. 그러나 prefill의 CPU matmul은
compute-bound이지 memory-bound가 아니므로, 이 영향은 무시 가능하다.

---

### 4. §3 split_weight 최적화 -- 검토

이전 검토 결론 유지. `split_all_weights` 헬퍼는 코드 품질 개선이므로 그대로 진행.

---

### 5. §4 Sync 최적화 -- 검토

이전 검토 결론 유지. 핵심 요약:

- OpenCL in-order queue에서 blocking `read_buffer`는 이전 커맨드 완료를 보장한다. `synchronize()` 생략이 이론적으로 가능.
- 그러나 Adreno 드라이버 구현의 100% spec 준수가 보장되지 않으므로, **방어적 sync를 기본으로 하고 프로파일링 후 제거**하라.
- 4회 sync의 총 오버헤드 200-600us/token은 decode TBT ~130ms 대비 ~0.5% 미만. 최적화 우선순위 낮음.

---

### 6. 리스크 분석 (개정)

| 리스크 | 심각도 | 발생 가능성 | 완화 방안 |
|--------|--------|-----------|----------|
| forward_gen.rs 코드 복잡도 폭증 | 중간 | 높음 | partition 로직 함수 추출 필수 |
| GQA wk/wv partition 오버헤드 (512 dim) | 낮음 | 중간 | Option skip + 프로파일링 후 판단 |
| synchronize 제거 시 Adreno 특이 동작 | 중간 | 낮음 | 디바이스 테스트 + 방어적 sync 유지 |
| **GPU-CPU interleave: KV cache coherence** | **높음** | **중간** | GPU chunk 후 synchronize() 필수. CL_MEM_USE_HOST_PTR의 cache coherence는 OpenCL spec이 보장하나, **Adreno/Mali 실기기에서 E2E 검증 필수**. Phase 2b 착수 조건으로 설정. |
| **CPU chunk: PrefillWorkspace 미사전할당 시 alloc/free 반복** | **중간** | **높음** | CPU 전용 PrefillWorkspace를 사전 할당하여 CPU chunk에서 재사용. Fallback path (매 layer alloc)는 사용하지 않을 것. |
| **CPU chunk 중 게임 CPU 경쟁** | 중간 | 중간 | CPU chunk은 12.7 tok/s (S1 측정값). 게임이 CPU도 사용하면 추가 저하 가능. 실측 필요. |
| Dynamic chunk size에서 chunk가 너무 작아 throughput 저하 | 낮음 | 중간 | chunk >= 64 하한 설정 |
| is_prefilling 추가로 shared 크레이트 호환성 깨짐 | 낮음 | 높음 | `#[serde(default)]` 속성으로 하위호환 보장 |
| Inter-layer yield의 yield_ms 최적값 미지 | 낮음 | 중간 | S25 실측으로 tuning. 기본값 16ms (1프레임) 권장 |
| PartitionWorkspace 메모리 ~2.3MB 추가 | 낮음 | 확정 | 수용 가능 |
| CPU matmul이 GPU보다 현저히 느려 partition 효과 저하 | 중간 | 중간 | 프로파일링 후 per-projection ON/OFF |

---

### 7. 권장 구현 순서

impl-request의 Phase 1 -> 2a -> 2b -> 3 순서에 **대체로 동의**하되, 세부 조정을 제안한다.

#### Phase 1a: FFN down partition (1~2일)

- 이전 검토와 동일. down projection만 추가.
- `PartitionWorkspace`에 down_gpu, down_cpu, ffn_mid_cpu 3개 필드 추가.
- 검증: greedy decode 정확도 + TBT 프로파일링.

#### Phase 1b: Attention QKV + wo partition (2~3일)

- 이전 검토와 동일. 7개 weight 전체 partition.
- 검증: greedy decode 정확도 + GQA 모델(Qwen) 테스트.

#### Phase 1c: 코드 정리 (1일)

- partition 로직 함수 추출 + PartialPair 구조체.

#### Phase 2a: Dynamic Chunk Size + chunk 간 Yield (0.5~1일)

impl-request의 Phase 2a와 **유사하나**, 레이어 간 yield 대신 **chunk 간 yield를 우선 구현**하라.

- `PrefillContention` enum 정의 (None/Low/High).
- `--prefill-contention` CLI 추가.
- generate.rs prefill 루프에서:
  - `effective_chunk_size` 동적 결정 (contention level 기반).
  - chunk 간 `thread::sleep(yield_ms)` 삽입 (synchronize() 직후).
- `is_prefilling` 상태를 `EngineStatus`에 추가 (`#[serde(default)]`).
- **레이어 간 yield는 chunk 간 yield의 효과가 불충분할 때만 추가** (Phase 2a-2로 분리).

**이유**: chunk 간 yield는 generate.rs에서만 변경하면 되므로 model 코드에 영향이 없다.
레이어 간 yield는 `TransformerModel::forward_into()`에 contention 인식을 주입해야 하며,
이는 SRP(Single Responsibility)를 약화시킨다.

#### Phase 2b: GPU-CPU Chunk Interleaving (2~3일)

**착수 조건**: Phase 2a 실측에서 FPS 개선이 -20% 이내로 불충분할 경우.

- CPU 전용 PrefillWorkspace 사전 할당.
- generate.rs prefill 루프에 GPU/CPU chunk 교대 로직.
- CPU backend로 `model.forward_into()` 호출 (weight는 dual-access 보장됨).
- KV cache UMA write 안전성 E2E 테스트.
- `SetPrefillContention` EngineCommand 추가.

#### Phase 3: 동적 ratio -- 논문 이후

동의.

---

### 8. 추가 권장사항

1. **per-projection 프로파일링**: partition overhead (sync + read_buffer + merge)를 별도 probe로 분리. `prof_record!(t, partition_sync)` 등.

2. **정확도 회귀 테스트**: partition 유/무에서 greedy decode 출력이 **bit-exact**해야 한다. E2E 레벨의 10-token 정확도 비교 테스트를 CI에 포함하라.

3. **Feature flag 유지**: `--tensor-partition 1.0`(또는 미지정)일 때 partition 코드를 skip하는 현재 설계를 유지. `if let Some(part)` guard로 zero-cost.

4. **Phase 2b KV cache 검증 테스트**: Interleave 구현 전에 다음을 검증하는 **독립 테스트**를 먼저 작성하라:
   - MadviseableGPUBuffer에 GPU write 후 synchronize() 후 CPU read가 정확한가
   - MadviseableGPUBuffer에 CPU write 후 GPU kernel이 해당 데이터를 정확히 읽는가
   - 교대로 GPU write(pos 0~47) → synchronize → CPU write(pos 48~63) → GPU attention(pos 0~63)이 정확한가

5. **SetPrefillContention과 SwitchHw 상호작용**: impl-request에서 두 커맨드가 상호 배타라고 기술했다. 이에 동의하며, **SwitchHw가 활성이면 contention을 무시**하는 로직을 generate.rs에 명시적으로 구현하라. SwitchHw CPU 상태에서 prefill은 이미 CPU-only이므로 contention이 무의미하다.
