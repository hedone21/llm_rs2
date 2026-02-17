# Chapter 10: Model Inference Pipeline

> **이전**: [09. GPU 어텐션](09_attention_mechanism.md) | **다음**: [11. KV 캐시 관리](11_kv_cache_management.md)

## 10.1 Overview

이 문서는 Llama 3.2 모델의 전체 추론 파이프라인을 설명합니다.

---

## 10.2 모델 아키텍처

### Llama 3.2 1B 구성
```
vocab_size = 128256
hidden_dim = 2048
num_layers = 16
num_heads = 32
num_kv_heads = 8 (GQA)
head_dim = 64
intermediate_size = 8192
max_position = 131072
rope_theta = 500000.0
rms_norm_eps = 1e-5
```

### 레이어 구조
```
Input Token IDs
      ↓
┌─────────────────────────────────────┐
│           Embedding                  │
│   embed_tokens: [vocab, hidden_dim]  │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│         LlamaLayer × 16             │
│  ┌─────────────────────────────┐    │
│  │ attention_norm (RMSNorm)    │    │
│  │ self_attn:                  │    │
│  │   q_proj, k_proj, v_proj    │    │
│  │   o_proj                    │    │
│  │ ffn_norm (RMSNorm)          │    │
│  │ mlp:                        │    │
│  │   gate_proj, up_proj        │    │
│  │   down_proj                 │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│           Final Norm                 │
│   norm: RMSNorm                      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│           LM Head                    │
│   lm_head: [hidden_dim, vocab]       │
│   (embed_tokens weight tied)         │
└─────────────────────────────────────┘
      ↓
Output Logits
```

---

## 10.3 추론 흐름

### 10.3.1 Prefill Phase (seq_len > 1)
```rust
// 전체 프롬프트를 한 번에 처리
fn prefill(&mut self, token_ids: &[u32]) -> Result<Tensor> {
    // 1. Embedding
    let mut hidden = self.embed(token_ids)?;
    
    // 2. 각 레이어 순회
    for layer in &mut self.layers {
        hidden = layer.forward(&mut hidden, kv_cache, 0)?;
    }
    
    // 3. Final Norm + LM Head
    self.norm(&mut hidden)?;
    let logits = self.lm_head(&hidden)?;
    
    Ok(logits)
}
```

### 10.3.2 Generation Phase (seq_len = 1)
```rust
// 토큰 하나씩 생성
fn generate_token(&mut self, token_id: u32, pos: usize) -> Result<u32> {
    // 1. Embedding (1 token)
    let mut hidden = self.embed(&[token_id])?;
    
    // 2. 각 레이어 (forward_gen 사용)
    for layer in &mut self.layers {
        layer.forward_gen(&mut hidden, kv_cache, pos)?;
    }
    
    // 3. LM Head → argmax
    let logits = self.lm_head(&hidden)?;
    let next_token = argmax(&logits);
    
    Ok(next_token)
}
```

---

## 10.4 LlamaLayer::forward_gen 상세

### 파일 위치
`src/layers/llama_layer.rs`

### 전체 흐름
```
Input: x [1, hidden_dim]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 1. Residual Save + RMS Norm                               │
│    residual = copy_from(x)                                │
│    rms_norm(x, attention_norm, eps)                       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 2. QKV Projection (Q4_0 quantized weights)                │
│    q = matmul_transposed(x, wq)   → [1, dim]              │
│    k = matmul_transposed(x, wk)   → [1, kv_dim]           │
│    v = matmul_transposed(x, wv)   → [1, kv_dim]           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 3. Reshape                                                │
│    q_rope = reshape(q, [1, heads_q, head_dim])            │
│    k_rope = reshape(k, [1, heads_kv, head_dim])           │
│    v = reshape(v, [1, heads_kv, head_dim])                │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 4. RoPE Inplace                                           │
│    rope_inplace(q_rope, start_pos, theta)                 │
│    rope_inplace(k_rope, start_pos, theta)                 │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 5. KV Cache Update                                        │
│    kv_cache.update(k_rope, v)                             │
│    (k, v) = kv_cache.get_view()                           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 6. Attention (GPU!)                                       │
│    attention_gen(q_rope, k_cache, v_cache, out_attn)      │
│    Output: [heads_q, head_dim]                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 7. Output Projection                                      │
│    attn_out = matmul_transposed(out_attn, wo)             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 8. Residual 1                                             │
│    add_assign(attn_out, residual)                         │
│    x = copy_from(attn_out)                                │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 9. FFN Norm                                               │
│    residual = copy_from(x)                                │
│    rms_norm(x, ffn_norm, eps)                             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 10. FFN (SwiGLU)                                          │
│     gate = matmul_transposed(x, w_gate)                   │
│     up = matmul_transposed(x, w_up)                       │
│     silu_mul(gate, up)   // gate = SiLU(gate) * up        │
│     down = matmul_transposed(gate, w_down)                │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ 11. Residual 2                                            │
│     add_assign(down, residual)                            │
│     x = copy_from(down)                                   │
└───────────────────────────────────────────────────────────┘
        │
        ▼
Output: x [1, hidden_dim]
```

---

## 10.5 LayerWorkspace

**파일**: `src/layers/workspace.rs`

### 목적
Decode 단계(seq_len=1)에서 매 토큰마다 버퍼를 할당하면 성능 저하가 발생한다. LayerWorkspace는 **한 번 할당하고 모든 레이어에서 재사용**하는 사전 할당 버퍼 집합이다.

### 구조

```rust
pub struct LayerWorkspace {
    // Attention
    pub q: Tensor,           // [batch, 1, q_dim]       — q_dim = num_heads * head_dim
    pub k: Tensor,           // [batch, 1, k_dim]       — k_dim = num_kv_heads * head_dim
    pub v: Tensor,           // [batch, 1, v_dim]       — v_dim = num_kv_heads * head_dim
    pub out_attn: Tensor,    // [batch, 1, q_dim]       — attention 출력 (projection 전)
    pub attn_out: Tensor,    // [batch, 1, dim]         — output projection 결과
    pub residual: Tensor,    // [batch, 1, dim]         — residual connection 저장

    // FFN
    pub gate: Tensor,        // [batch, 1, ffn_hidden]  — gate_proj 결과
    pub up: Tensor,          // [batch, 1, ffn_hidden]  — up_proj 결과
    pub down: Tensor,        // [batch, 1, dim]         — down_proj 결과

    // Attention scores (CPU-side)
    pub scores: Vec<f32>,    // [n_heads * max_seq_len]
}
```

> **주의**: `q`, `out_attn`의 shape은 `[batch, 1, q_dim]`이며, `q_dim = num_heads * head_dim`으로 **head 차원이 flatten**된 형태이다. `[batch, 1, num_heads, head_dim]`이 아니다. `out_attn`은 attention 연산의 직접 출력(projection 전), `attn_out`은 output projection 후 dim 크기이다.

### WorkspaceConfig

`LayerWorkspace::new()`는 `WorkspaceConfig`를 받아 버퍼를 할당한다:

```rust
pub struct WorkspaceConfig {
    pub batch_size: usize,     // 보통 1
    pub dim: usize,            // hidden_dim (2048)
    pub q_dim: usize,          // num_heads * head_dim (32 * 64 = 2048)
    pub k_dim: usize,          // num_kv_heads * head_dim (8 * 64 = 512)
    pub v_dim: usize,          // num_kv_heads * head_dim (8 * 64 = 512)
    pub ffn_hidden: usize,     // intermediate_size (8192)
    pub n_heads: usize,        // num_heads (32)
    pub max_seq_len: usize,    // KV cache 최대 길이
}
```

### Llama 3.2 1B 메모리 사용량

| 필드 | Shape | 크기 (F32) |
|------|-------|-----------|
| `q` | [1, 1, 2048] | 8 KB |
| `k` | [1, 1, 512] | 2 KB |
| `v` | [1, 1, 512] | 2 KB |
| `out_attn` | [1, 1, 2048] | 8 KB |
| `gate` | [1, 1, 8192] | 32 KB |
| `up` | [1, 1, 8192] | 32 KB |
| `down` | [1, 1, 2048] | 8 KB |
| `residual` | [1, 1, 2048] | 8 KB |
| `attn_out` | [1, 1, 2048] | 8 KB |
| `scores` | [32 * max_seq_len] | max_seq_len에 비례 |
| **합계** | | ~108 KB + scores |

모든 버퍼는 F32로 할당되며, `Memory::alloc(size * 4, DType::F32)`를 통해 생성된다.

---

## 10.6 KV Cache 관리

### 구조
```rust
pub struct KVCache {
    pub k_buffer: Tensor,    // [max_seq_len, kv_heads, head_dim]
    pub v_buffer: Tensor,    // [max_seq_len, kv_heads, head_dim]
    pub current_pos: usize,  // 현재 토큰 수
}
```

### 업데이트 로직
```rust
fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
    // new_k: [1, kv_heads, head_dim]
    // k_buffer[current_pos, :, :] = new_k
    
    backend.copy_slice(
        new_k, 
        &mut self.k_buffer,
        0,                              // src_offset
        self.current_pos * kv_heads * head_dim,  // dst_offset
        kv_heads * head_dim             // count
    )?;
    
    // same for V
    
    self.current_pos += 1;
    Ok(())
}
```

---

## 10.7 Quantization 지원

### Q4_0 Weight Format
```
원본: [out_features, in_features] (F32)
양자화: [out_features, in_features/32] (BlockQ4_0)

BlockQ4_0:
  d: f16      // scale
  qs: [16]u8  // 32 x 4-bit values packed
```

### Matmul with Dequantization
```rust
fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if b.dtype() == DType::Q4_0 {
        // Q4_0 전용 커널 사용 (내부에서 dequantization)
        self.matmul_q4_0(a, b, out)
    } else {
        // F32 matmul
        self.matmul(a, b, out)
    }
}
```

---

## 10.8 Generate Binary

### 파일 위치
`src/bin/generate.rs`

### 사용법
```bash
./generate -b opencl -n 20 -p "Hello, world!"
```

### 주요 출력
```
TTFT: 143.08 ms      # Time To First Token
Avg TBT: 19.19 ms    # Time Between Tokens
(52.1 tokens/sec)
```

---

## 10.9 성능 프로파일

### 레이어당 연산 비중 (추정)
```
┌─────────────────────────────────────────────┐
│ Operation          │ Est. Time │ % of Layer │
├─────────────────────────────────────────────┤
│ QKV Projection     │  3-4 ms   │   20%      │
│ RoPE               │  <1 ms    │   3%       │
│ KV Cache Update    │  <1 ms    │   2%       │
│ Attention (GPU)    │  2-3 ms   │   15%      │
│ Output Projection  │  2 ms     │   12%      │
│ FFN (3 matmuls)    │  6-8 ms   │   45%      │
│ Other (norm, add)  │  <1 ms    │   3%       │
└─────────────────────────────────────────────┘
```

### 병목 분석
1. **FFN**: 3개의 큰 matmul (8192 intermediate)
2. **QKV Projection**: 3개의 matmul
3. **Attention**: 최적화 완료 (GPU)
