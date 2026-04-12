#!/usr/bin/env python3
"""Convert safetensors F16 model to pure Q4_0 GGUF for llm.rs testing.

Usage:
    python scripts/convert_safetensors_to_gguf.py \
        models/llama3.2-1b \
        models/llama3.2-1b-q4_0.gguf
"""
import sys, json, struct, os
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import load_file as st_load_file

QK4_0 = 32  # Q4_0 block size

def quantize_q4_0(data_f32: np.ndarray) -> bytes:
    """Quantize F32 array to Q4_0 blocks (matches llm.rs/ggml BlockQ4_0)."""
    assert data_f32.size % QK4_0 == 0, f"Size {data_f32.size} not divisible by {QK4_0}"
    n_blocks = data_f32.size // QK4_0
    data = data_f32.reshape(n_blocks, QK4_0)

    result = bytearray()
    for block in data:
        amax = np.max(np.abs(block))
        d = amax / 7.0
        id_ = 1.0 / d if d != 0 else 0.0

        # Quantize to 4-bit signed [-8, 7], store as unsigned [0, 15]
        q = np.clip(np.round(block * id_), -8, 7).astype(np.int8) + 8

        # Pack d as f16 (2 bytes)
        d_f16 = np.float16(d)
        result.extend(d_f16.tobytes())

        # Pack 32 nibbles into 16 bytes (lower 16 in low nibble, upper 16 in high nibble)
        qs = bytearray(16)
        for i in range(16):
            qs[i] = int(q[i]) | (int(q[i + 16]) << 4)
        result.extend(qs)

    return bytes(result)


def write_gguf_string(buf: bytearray, s: str):
    """Write GGUF string: u64 len + bytes."""
    encoded = s.encode('utf-8')
    buf.extend(struct.pack('<Q', len(encoded)))
    buf.extend(encoded)


def write_gguf_kv(buf: bytearray, key: str, value, vtype: int):
    """Write a single KV pair."""
    write_gguf_string(buf, key)
    buf.extend(struct.pack('<I', vtype))

    if vtype == 4:    # u32
        buf.extend(struct.pack('<I', value))
    elif vtype == 6:  # f32
        buf.extend(struct.pack('<f', value))
    elif vtype == 8:  # string
        write_gguf_string(buf, value)
    elif vtype == 10: # u64
        buf.extend(struct.pack('<Q', value))


# GGUF tensor name mapping: HF name → GGUF name
GGUF_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
for i in range(256):  # enough layers
    GGUF_NAMES.update({
        f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        # QKV bias (Qwen2)
        f"model.layers.{i}.self_attn.q_proj.bias": f"blk.{i}.attn_q.bias",
        f"model.layers.{i}.self_attn.k_proj.bias": f"blk.{i}.attn_k.bias",
        f"model.layers.{i}.self_attn.v_proj.bias": f"blk.{i}.attn_v.bias",
        # Gemma3 extra norms
        f"model.layers.{i}.self_attn.q_norm.weight": f"blk.{i}.attn_q_norm.weight",
        f"model.layers.{i}.self_attn.k_norm.weight": f"blk.{i}.attn_k_norm.weight",
        f"model.layers.{i}.pre_feedforward_layernorm.weight": f"blk.{i}.pre_ffn_norm.weight",
        f"model.layers.{i}.post_feedforward_layernorm.weight": f"blk.{i}.post_ffn_norm.weight",
    })

# ggml_type constants
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2

ALIGNMENT = 32


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <safetensors_dir> <output.gguf>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_path = sys.argv[2]

    # Load config.json
    config = json.loads((model_dir / "config.json").read_text())

    # Detect architecture from model_type
    model_type = config.get("model_type", "llama").lower()
    ARCH_MAP = {"llama": "llama", "qwen2": "qwen2", "gemma": "gemma", "gemma2": "gemma2",
                 "gemma3": "gemma3", "gemma3_text": "gemma3"}
    arch = ARCH_MAP.get(model_type, "llama")
    print(f"Architecture: {arch} (model_type={model_type})")

    # Find safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"No .safetensors files found in {model_dir}")
        sys.exit(1)

    # Open all safetensors
    handles = []
    tensor_names = {}
    for f in st_files:
        h = safe_open(str(f), framework="numpy")
        for name in h.keys():
            tensor_names[name] = len(handles)
        handles.append(h)

    print(f"Found {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Determine which tensors to quantize vs keep as F32
    weight_2d = set()  # 2D weight tensors → Q4_0
    norm_1d = set()    # 1D norm tensors → F32
    embed = set()      # embed/lm_head → F32 (too important to quantize)

    for name in tensor_names:
        h = handles[tensor_names[name]]
        shape = h.get_slice(name).get_shape()
        if len(shape) == 1:
            norm_1d.add(name)
        elif "embed_tokens" in name or "lm_head" in name:
            embed.add(name)
        else:
            if shape[-1] % QK4_0 == 0:
                weight_2d.add(name)
            else:
                embed.add(name)  # can't quantize, keep as F32

    print(f"  Q4_0: {len(weight_2d)}, F32 norm: {len(norm_1d)}, F32 embed: {len(embed)}")

    # Prepare tensor data
    tensors = []  # (gguf_name, dims_gguf, ggml_type, data_bytes)

    for hf_name in sorted(tensor_names.keys()):
        gguf_name = GGUF_NAMES.get(hf_name)
        if gguf_name is None:
            print(f"  SKIP: {hf_name} (no GGUF mapping)")
            continue

        h = handles[tensor_names[hf_name]]
        # Handle BF16: numpy doesn't support bf16 natively
        try:
            tensor = h.get_tensor(hf_name)
        except TypeError:
            # BF16 fallback: read raw uint16 bytes, convert to f32 via bit manipulation
            sl = h.get_slice(hf_name)
            shape_orig = sl.get_shape()
            # Read raw bytes from safetensors file
            st_path = st_files[tensor_names[hf_name]]
            with open(str(st_path), 'rb') as raw_f:
                header_size = struct.unpack('<Q', raw_f.read(8))[0]
                header = json.loads(raw_f.read(header_size))
                info = header[hf_name]
                offsets = info['data_offsets']
                raw_f.seek(8 + header_size + offsets[0])
                raw_bytes = raw_f.read(offsets[1] - offsets[0])
            # BF16 → F32: shift left 16 bits
            bf16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            f32_bits = bf16.astype(np.uint32) << 16
            tensor = np.frombuffer(f32_bits.tobytes(), dtype=np.float32).reshape(shape_orig)
        shape = tensor.shape

        # GGUF dims are reversed (innermost first)
        dims_gguf = list(reversed(shape))

        if hf_name in weight_2d:
            # Quantize to Q4_0
            f32 = tensor.astype(np.float32).flatten()
            data = quantize_q4_0(f32)
            tensors.append((gguf_name, dims_gguf, GGML_Q4_0, data))
        else:
            # Keep as F32
            f32 = tensor.astype(np.float32)
            data = f32.tobytes()
            tensors.append((gguf_name, dims_gguf, GGML_F32, data))

    print(f"Prepared {len(tensors)} tensors for GGUF")

    # Build GGUF file
    # 1. Header
    header = bytearray()
    header.extend(struct.pack('<I', 0x46554747))  # magic: GGUF LE
    header.extend(struct.pack('<I', 3))            # version: 3
    header.extend(struct.pack('<Q', len(tensors))) # tensor_count
    # KV count (will fill after building KV section)

    # 2. KV metadata
    kv = bytearray()
    kv_count = 0

    def add_kv_str(key, val):
        nonlocal kv_count
        write_gguf_kv(kv, key, val, 8)
        kv_count += 1

    def add_kv_u32(key, val):
        nonlocal kv_count
        write_gguf_kv(kv, key, val, 4)
        kv_count += 1

    def add_kv_f32(key, val):
        nonlocal kv_count
        write_gguf_kv(kv, key, val, 6)
        kv_count += 1

    add_kv_str("general.architecture", arch)
    add_kv_str("general.name", config.get("_name_or_path", "llama"))
    add_kv_u32(f"{arch}.embedding_length", config["hidden_size"])
    add_kv_u32(f"{arch}.block_count", config["num_hidden_layers"])
    add_kv_u32(f"{arch}.attention.head_count", config["num_attention_heads"])
    add_kv_u32(f"{arch}.attention.head_count_kv", config.get("num_key_value_heads", config["num_attention_heads"]))
    add_kv_u32(f"{arch}.feed_forward_length", config["intermediate_size"])
    add_kv_u32(f"{arch}.vocab_size", config["vocab_size"])
    add_kv_u32(f"{arch}.context_length", config.get("max_position_embeddings", 2048))
    add_kv_f32(f"{arch}.attention.layer_norm_rms_epsilon", config.get("rms_norm_eps", 1e-5))
    add_kv_f32(f"{arch}.rope.freq_base", config.get("rope_theta", 10000.0))

    # Gemma3 extra metadata
    if "head_dim" in config:
        add_kv_u32(f"{arch}.attention.head_dim", config["head_dim"])
    if "sliding_window" in config:
        add_kv_u32(f"{arch}.attention.sliding_window", config["sliding_window"])
    if "sliding_window_pattern" in config:
        add_kv_u32(f"{arch}.attention.sliding_window_pattern", config["sliding_window_pattern"])
    if "query_pre_attn_scalar" in config:
        add_kv_f32(f"{arch}.attention.query_pre_attn_scalar", config["query_pre_attn_scalar"])

    # Finalize header with kv_count
    header.extend(struct.pack('<Q', kv_count))

    # 3. Tensor info section
    tensor_info = bytearray()
    current_offset = 0
    tensor_offsets = []

    for gguf_name, dims, ggml_type, data in tensors:
        write_gguf_string(tensor_info, gguf_name)
        tensor_info.extend(struct.pack('<I', len(dims)))
        for d in dims:
            tensor_info.extend(struct.pack('<Q', d))
        tensor_info.extend(struct.pack('<I', ggml_type))
        tensor_info.extend(struct.pack('<Q', current_offset))
        tensor_offsets.append(current_offset)
        current_offset += len(data)
        # Align next tensor
        padding = (ALIGNMENT - (current_offset % ALIGNMENT)) % ALIGNMENT
        current_offset += padding

    # 4. Compute data start (aligned)
    meta_size = len(header) + len(kv) + len(tensor_info)
    data_start_padding = (ALIGNMENT - (meta_size % ALIGNMENT)) % ALIGNMENT

    # 5. Write file
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(kv)
        f.write(tensor_info)
        f.write(b'\x00' * data_start_padding)

        for i, (gguf_name, dims, ggml_type, data) in enumerate(tensors):
            f.write(data)
            # Align
            padding = (ALIGNMENT - (len(data) % ALIGNMENT)) % ALIGNMENT
            if padding and i < len(tensors) - 1:
                f.write(b'\x00' * padding)

    file_size = os.path.getsize(output_path)
    print(f"Written {output_path}: {file_size / 1024 / 1024:.1f} MB ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
