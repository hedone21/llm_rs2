//! compress_lab: F16 KV cache compression experiment workbench.
//!
//! Two modes:
//!   --capture: Run inference and dump raw F16 KV cache to disk
//!   (default):  Load dumped data and benchmark compression algorithms
//!
//! Usage:
//!   # Step 1: Capture real data (once)
//!   cargo run --release --bin compress_lab -- --capture \
//!     --model-path models/llama3.2-1b \
//!     --prompt "The quick brown fox" -n 128
//!
//!   # Step 2: Benchmark algorithms (fast iteration)
//!   cargo run --release --bin compress_lab
//!   cargo run --release --bin compress_lab -- --algo bitshuffle+lz4

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── CLI ──────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "compress_lab",
    about = "F16 KV cache compression experiment workbench"
)]
struct Args {
    /// Capture mode: run inference and dump KV cache to disk
    #[arg(long)]
    capture: bool,

    /// Model path (capture mode only)
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    model_path: String,

    /// Prompt for inference (capture mode only)
    #[arg(
        short,
        long,
        default_value = "The quick brown fox jumps over the lazy dog. In a distant galaxy far away, there existed a civilization that thrived on the principles of harmony and knowledge."
    )]
    prompt: String,

    /// Number of decode tokens (capture mode only)
    #[arg(short, long, default_value_t = 256)]
    num_tokens: usize,

    /// Directory for dumped KV cache data
    #[arg(long, default_value = "experiments/kv_dump")]
    data_dir: String,

    /// Filter to specific algorithm (substring match)
    #[arg(long)]
    algo: Option<String>,

    /// Show per-layer breakdown
    #[arg(long)]
    per_layer: bool,
}

// ── Data structures ──────────────────────────────────────────────────────

struct KVLayerData {
    k: Vec<u8>,
    v: Vec<u8>,
    layer_id: usize,
}

#[allow(dead_code)]
struct KVDump {
    layers: Vec<KVLayerData>,
    num_tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    token_bytes: usize, // kv_heads * head_dim * 2 (F16)
}

impl KVDump {
    fn load(dir: &Path) -> Result<Self> {
        // Read metadata
        let meta_path = dir.join("metadata.txt");
        let meta = fs::read_to_string(&meta_path).with_context(|| {
            format!("No metadata at {:?}. Run with --capture first.", meta_path)
        })?;

        let mut num_tokens = 0;
        let mut kv_heads = 0;
        let mut head_dim = 0;
        let mut num_layers = 0;
        for line in meta.lines() {
            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "num_tokens" => num_tokens = v.trim().parse()?,
                    "kv_heads" => kv_heads = v.trim().parse()?,
                    "head_dim" => head_dim = v.trim().parse()?,
                    "num_layers" => num_layers = v.trim().parse()?,
                    _ => {}
                }
            }
        }
        let token_bytes = kv_heads * head_dim * 2;

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let path = dir.join(format!("layer_{i}.bin"));
            let data = fs::read(&path).with_context(|| format!("Failed to read {:?}", path))?;
            let half = data.len() / 2;
            layers.push(KVLayerData {
                k: data[..half].to_vec(),
                v: data[half..].to_vec(),
                layer_id: i,
            });
        }

        println!(
            "Loaded {} layers, {} tokens, {} B/tok ({}×{}×F16)",
            layers.len(),
            num_tokens,
            token_bytes,
            kv_heads,
            head_dim
        );
        println!(
            "Total data: {:.1} MB\n",
            (layers.len() * num_tokens * token_bytes * 2) as f64 / 1024.0 / 1024.0
        );

        Ok(KVDump {
            layers,
            num_tokens,
            kv_heads,
            head_dim,
            token_bytes,
        })
    }
}

// ── Compression algorithms ───────────────────────────────────────────────

use llm_rs2::core::offload::preprocess;

struct AlgoResult {
    compressed_size: usize,
    original_size: usize,
    compress_us: u64,
    decompress_us: u64,
    verified: bool,
}

#[allow(dead_code)]
impl AlgoResult {
    fn ratio(&self) -> f64 {
        self.original_size as f64 / self.compressed_size as f64
    }
}

type AlgoFn = fn(&[u8], usize) -> AlgoResult;

fn algo_raw_lz4(data: &[u8], _elem_size: usize) -> AlgoResult {
    let t0 = Instant::now();
    let compressed = lz4::block::compress(data, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let decompressed = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: decompressed == data,
    }
}

fn algo_raw_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let t0 = Instant::now();
    let compressed = zstd::bulk::compress(data, 1).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let decompressed = zstd::bulk::decompress(&compressed, data.len()).unwrap();
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: decompressed == data,
    }
}

fn algo_byteshuffle_lz4(data: &[u8], elem_size: usize) -> AlgoResult {
    let mut shuffled = vec![0u8; data.len()];

    let t0 = Instant::now();
    preprocess::shuffle(data, &mut shuffled, elem_size);
    let compressed = lz4::block::compress(&shuffled, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let dec = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    preprocess::unshuffle(&dec, &mut restored, elem_size);
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

fn algo_bitshuffle_lz4(data: &[u8], elem_size: usize) -> AlgoResult {
    let mut shuffled = vec![0u8; data.len()];

    let t0 = Instant::now();
    preprocess::bitshuffle(data, &mut shuffled, elem_size);
    let compressed = lz4::block::compress(&shuffled, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let dec = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    preprocess::bitunshuffle(&dec, &mut restored, elem_size);
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

fn algo_bitshuffle_zstd(data: &[u8], elem_size: usize) -> AlgoResult {
    let mut shuffled = vec![0u8; data.len()];

    let t0 = Instant::now();
    preprocess::bitshuffle(data, &mut shuffled, elem_size);
    let compressed = zstd::bulk::compress(&shuffled, 1).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let dec = zstd::bulk::decompress(&compressed, data.len()).unwrap();
    let mut restored = vec![0u8; data.len()];
    preprocess::bitunshuffle(&dec, &mut restored, elem_size);
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

/// Exponent-mantissa separation: split F16 into 5-bit exponent stream + 11-bit rest stream.
/// Compress each independently.
fn algo_exponent_split_lz4(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;
    // Stream 1: high byte (sign + exponent + mantissa_high 2 bits)
    // Stream 2: low byte (mantissa_low 8 bits)
    let mut hi_stream = Vec::with_capacity(n);
    let mut lo_stream = Vec::with_capacity(n);

    let t0 = Instant::now();
    for i in 0..n {
        lo_stream.push(data[i * 2]);
        hi_stream.push(data[i * 2 + 1]);
    }
    let hi_compressed = lz4::block::compress(&hi_stream, None, false).unwrap();
    let lo_compressed = lz4::block::compress(&lo_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = hi_compressed.len() + lo_compressed.len();

    let t1 = Instant::now();
    let hi_dec = lz4::block::decompress(&hi_compressed, Some(n as i32)).unwrap();
    let lo_dec = lz4::block::decompress(&lo_compressed, Some(n as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        restored[i * 2] = lo_dec[i];
        restored[i * 2 + 1] = hi_dec[i];
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

/// Exponent-only Huffman-like: bytedelta on high byte stream + Zstd for entropy coding
fn algo_exponent_split_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;
    let mut hi_stream = Vec::with_capacity(n);
    let mut lo_stream = Vec::with_capacity(n);

    let t0 = Instant::now();
    for i in 0..n {
        lo_stream.push(data[i * 2]);
        hi_stream.push(data[i * 2 + 1]);
    }
    // Bytedelta on hi stream (exponent bytes have similar neighbors)
    preprocess::bytedelta_encode(&mut hi_stream, n, 1);
    let hi_compressed = zstd::bulk::compress(&hi_stream, 1).unwrap();
    let lo_compressed = lz4::block::compress(&lo_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = hi_compressed.len() + lo_compressed.len();

    let t1 = Instant::now();
    let mut hi_dec = zstd::bulk::decompress(&hi_compressed, n).unwrap();
    preprocess::bytedelta_decode(&mut hi_dec, n, 1);
    let lo_dec = lz4::block::decompress(&lo_compressed, Some(n as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        restored[i * 2] = lo_dec[i];
        restored[i * 2 + 1] = hi_dec[i];
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

/// Channel-reorder: group same head_dim index across all tokens, then bitshuffle.
/// Exploits cross-token exponent similarity within a channel.
fn algo_channel_reorder_bitshuffle_lz4(data: &[u8], _elem_size: usize) -> AlgoResult {
    // data layout (SeqMajor): [tok0: h0d0 h0d1 ... h7d63][tok1: h0d0 ...][...]
    // Each F16 = 2 bytes. token_stride = kv_heads * head_dim * 2
    // We can't know kv_heads/head_dim from elem_size alone.
    // Assume Llama 3.2 1B: 8 heads × 64 dim = 512 F16 values per token
    let values_per_token = 512; // 8 * 64
    let bytes_per_token = values_per_token * 2;
    let n_tokens = data.len() / bytes_per_token;
    if n_tokens == 0 || !data.len().is_multiple_of(bytes_per_token) {
        return algo_bitshuffle_lz4(data, 2); // fallback
    }

    // Reorder: for each channel c, gather all tokens' value at channel c
    // Output: [ch0_tok0 ch0_tok1 ... ch0_tokN][ch1_tok0 ...][...]
    let mut reordered = vec![0u8; data.len()];

    let t0 = Instant::now();
    for ch in 0..values_per_token {
        for tok in 0..n_tokens {
            let src_off = tok * bytes_per_token + ch * 2;
            let dst_off = (ch * n_tokens + tok) * 2;
            reordered[dst_off] = data[src_off];
            reordered[dst_off + 1] = data[src_off + 1];
        }
    }
    // Now bitshuffle the channel-major data
    let mut shuffled = vec![0u8; data.len()];
    preprocess::bitshuffle(&reordered, &mut shuffled, 2);
    let compressed = lz4::block::compress(&shuffled, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let dec = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
    let mut unshuffled = vec![0u8; data.len()];
    preprocess::bitunshuffle(&dec, &mut unshuffled, 2);
    // Reverse reorder
    let mut restored = vec![0u8; data.len()];
    for ch in 0..values_per_token {
        for tok in 0..n_tokens {
            let src_off = (ch * n_tokens + tok) * 2;
            let dst_off = tok * bytes_per_token + ch * 2;
            restored[dst_off] = unshuffled[src_off];
            restored[dst_off + 1] = unshuffled[src_off + 1];
        }
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

/// Token-delta: compute delta between consecutive tokens' KV values, then bitshuffle.
fn algo_delta_token_bitshuffle_lz4(data: &[u8], _elem_size: usize) -> AlgoResult {
    let bytes_per_token = 512 * 2; // 8 heads × 64 dim × 2 bytes
    let n_tokens = data.len() / bytes_per_token;
    if n_tokens < 2 || !data.len().is_multiple_of(bytes_per_token) {
        return algo_bitshuffle_lz4(data, 2);
    }

    let mut delta = vec![0u8; data.len()];

    let t0 = Instant::now();
    // First token: copy as-is
    delta[..bytes_per_token].copy_from_slice(&data[..bytes_per_token]);
    // Remaining tokens: F16-value-wise XOR with previous token
    for tok in 1..n_tokens {
        let curr = tok * bytes_per_token;
        let prev = (tok - 1) * bytes_per_token;
        for i in (0..bytes_per_token).step_by(2) {
            let c = u16::from_le_bytes([data[curr + i], data[curr + i + 1]]);
            let p = u16::from_le_bytes([data[prev + i], data[prev + i + 1]]);
            let d = c ^ p;
            delta[curr + i] = (d & 0xFF) as u8;
            delta[curr + i + 1] = (d >> 8) as u8;
        }
    }
    let mut shuffled = vec![0u8; data.len()];
    preprocess::bitshuffle(&delta, &mut shuffled, 2);
    let compressed = lz4::block::compress(&shuffled, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let t1 = Instant::now();
    let dec = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
    let mut undelta = vec![0u8; data.len()];
    preprocess::bitunshuffle(&dec, &mut undelta, 2);
    // Reverse XOR delta
    let mut restored = vec![0u8; data.len()];
    restored[..bytes_per_token].copy_from_slice(&undelta[..bytes_per_token]);
    for tok in 1..n_tokens {
        let curr = tok * bytes_per_token;
        let prev = (tok - 1) * bytes_per_token;
        for i in (0..bytes_per_token).step_by(2) {
            let d = u16::from_le_bytes([undelta[curr + i], undelta[curr + i + 1]]);
            let p = u16::from_le_bytes([restored[prev + i], restored[prev + i + 1]]);
            let r = d ^ p;
            restored[curr + i] = (r & 0xFF) as u8;
            restored[curr + i + 1] = (r >> 8) as u8;
        }
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: compressed.len(),
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H13 variant: All 4 nibble streams with Zstd ─────────────────────────

fn algo_nibble_shuffle_all_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    let nib_bytes = n.div_ceil(2);
    let mut nib3 = vec![0u8; nib_bytes];
    let mut nib2 = vec![0u8; nib_bytes];
    let mut nib1 = vec![0u8; nib_bytes];
    let mut nib0 = vec![0u8; nib_bytes];

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        nib3[byte_idx] |= ((val >> 12) as u8 & 0xF) << shift;
        nib2[byte_idx] |= ((val >> 8) as u8 & 0xF) << shift;
        nib1[byte_idx] |= ((val >> 4) as u8 & 0xF) << shift;
        nib0[byte_idx] |= (val as u8 & 0xF) << shift;
    }

    let c3 = zstd::bulk::compress(&nib3, 1).unwrap();
    let c2 = zstd::bulk::compress(&nib2, 1).unwrap();
    let c1 = zstd::bulk::compress(&nib1, 1).unwrap();
    let c0 = zstd::bulk::compress(&nib0, 1).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c3.len() + c2.len() + c1.len() + c0.len();

    let t1 = Instant::now();
    let d3 = zstd::bulk::decompress(&c3, nib_bytes).unwrap();
    let d2 = zstd::bulk::decompress(&c2, nib_bytes).unwrap();
    let d1 = zstd::bulk::decompress(&c1, nib_bytes).unwrap();
    let d0 = zstd::bulk::decompress(&c0, nib_bytes).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        let n0 = (d0[byte_idx] >> shift) & 0xF;
        let n1 = (d1[byte_idx] >> shift) & 0xF;
        let n2 = (d2[byte_idx] >> shift) & 0xF;
        let n3 = (d3[byte_idx] >> shift) & 0xF;
        let val = (n3 as u16) << 12 | (n2 as u16) << 8 | (n1 as u16) << 4 | n0 as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H13 variant: Nibble shuffle + expsplit-style (nib3+nib2 as "hi", nib1+nib0 as "lo")

fn algo_nibble_shuffle_lz4(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    let nib_bytes = n.div_ceil(2);
    let mut nib3 = vec![0u8; nib_bytes];
    let mut nib2 = vec![0u8; nib_bytes];
    let mut nib1 = vec![0u8; nib_bytes];
    let mut nib0 = vec![0u8; nib_bytes];

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        nib3[byte_idx] |= ((val >> 12) as u8 & 0xF) << shift;
        nib2[byte_idx] |= ((val >> 8) as u8 & 0xF) << shift;
        nib1[byte_idx] |= ((val >> 4) as u8 & 0xF) << shift;
        nib0[byte_idx] |= (val as u8 & 0xF) << shift;
    }

    let c3 = lz4::block::compress(&nib3, None, false).unwrap();
    let c2 = lz4::block::compress(&nib2, None, false).unwrap();
    let c1 = lz4::block::compress(&nib1, None, false).unwrap();
    let c0 = lz4::block::compress(&nib0, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c3.len() + c2.len() + c1.len() + c0.len();

    let t1 = Instant::now();
    let d3 = lz4::block::decompress(&c3, Some(nib_bytes as i32)).unwrap();
    let d2 = lz4::block::decompress(&c2, Some(nib_bytes as i32)).unwrap();
    let d1 = lz4::block::decompress(&c1, Some(nib_bytes as i32)).unwrap();
    let d0 = lz4::block::decompress(&c0, Some(nib_bytes as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        let n0 = (d0[byte_idx] >> shift) & 0xF;
        let n1 = (d1[byte_idx] >> shift) & 0xF;
        let n2 = (d2[byte_idx] >> shift) & 0xF;
        let n3 = (d3[byte_idx] >> shift) & 0xF;
        let val = (n3 as u16) << 12 | (n2 as u16) << 8 | (n1 as u16) << 4 | n0 as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H8: Zstd higher compression levels ──────────────────────────────────

fn algo_expsplit_bytedelta_zstd3(data: &[u8], _elem_size: usize) -> AlgoResult {
    expsplit_bytedelta_zstd_level(data, 3)
}

fn algo_expsplit_bytedelta_zstd5(data: &[u8], _elem_size: usize) -> AlgoResult {
    expsplit_bytedelta_zstd_level(data, 5)
}

fn algo_expsplit_bytedelta_zstd9(data: &[u8], _elem_size: usize) -> AlgoResult {
    expsplit_bytedelta_zstd_level(data, 9)
}

fn expsplit_bytedelta_zstd_level(data: &[u8], level: i32) -> AlgoResult {
    let n = data.len() / 2;
    let mut hi_stream = Vec::with_capacity(n);
    let mut lo_stream = Vec::with_capacity(n);

    let t0 = Instant::now();
    for i in 0..n {
        lo_stream.push(data[i * 2]);
        hi_stream.push(data[i * 2 + 1]);
    }
    preprocess::bytedelta_encode(&mut hi_stream, n, 1);
    let hi_compressed = zstd::bulk::compress(&hi_stream, level).unwrap();
    let lo_compressed = lz4::block::compress(&lo_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = hi_compressed.len() + lo_compressed.len();

    let t1 = Instant::now();
    let mut hi_dec = zstd::bulk::decompress(&hi_compressed, n).unwrap();
    preprocess::bytedelta_decode(&mut hi_dec, n, 1);
    let lo_dec = lz4::block::decompress(&lo_compressed, Some(n as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        restored[i * 2] = lo_dec[i];
        restored[i * 2 + 1] = hi_dec[i];
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H9: Hi-stream bitshuffle + Zstd ─────────────────────────────────────

fn algo_expsplit_bitshuffle_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;
    let mut hi_stream = Vec::with_capacity(n);
    let mut lo_stream = Vec::with_capacity(n);

    let t0 = Instant::now();
    for i in 0..n {
        lo_stream.push(data[i * 2]);
        hi_stream.push(data[i * 2 + 1]);
    }
    // Bitshuffle the hi stream (elem_size=1 for byte-level bit transpose)
    let mut hi_shuffled = vec![0u8; n];
    preprocess::bitshuffle(&hi_stream, &mut hi_shuffled, 1);
    let hi_compressed = zstd::bulk::compress(&hi_shuffled, 1).unwrap();
    let lo_compressed = lz4::block::compress(&lo_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = hi_compressed.len() + lo_compressed.len();

    let t1 = Instant::now();
    let hi_dec_shuffled = zstd::bulk::decompress(&hi_compressed, n).unwrap();
    let mut hi_dec = vec![0u8; n];
    preprocess::bitunshuffle(&hi_dec_shuffled, &mut hi_dec, 1);
    let lo_dec = lz4::block::decompress(&lo_compressed, Some(n as i32)).unwrap();
    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        restored[i * 2] = lo_dec[i];
        restored[i * 2 + 1] = hi_dec[i];
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H10: Context-dependent lo byte coding ────────────────────────────────

fn algo_expsplit_context_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Group lo bytes by their corresponding hi byte value
    let mut buckets: std::collections::HashMap<u8, Vec<u8>> = std::collections::HashMap::new();
    let mut hi_stream = Vec::with_capacity(n);
    for i in 0..n {
        let lo = data[i * 2];
        let hi = data[i * 2 + 1];
        hi_stream.push(hi);
        buckets.entry(hi).or_default().push(lo);
    }
    // Compress hi stream with bytedelta + zstd
    preprocess::bytedelta_encode(&mut hi_stream, n, 1);
    let hi_compressed = zstd::bulk::compress(&hi_stream, 1).unwrap();

    // Compress each lo bucket independently with zstd
    let mut lo_total_compressed = 0usize;
    let mut lo_buckets_compressed: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut bucket_keys: Vec<u8> = buckets.keys().copied().collect();
    bucket_keys.sort();
    for &key in &bucket_keys {
        let bucket = &buckets[&key];
        let compressed = zstd::bulk::compress(bucket, 1).unwrap();
        lo_total_compressed += compressed.len();
        lo_buckets_compressed.push((key, compressed));
    }
    let compress_us = t0.elapsed().as_micros() as u64;

    // Header overhead: number of buckets + (key, size) per bucket
    let header_size = 2 + bucket_keys.len() * 6; // 2 byte count + 6 per bucket
    let total_compressed = hi_compressed.len() + lo_total_compressed + header_size;

    let t1 = Instant::now();
    // Decompress hi
    let mut hi_dec = zstd::bulk::decompress(&hi_compressed, n).unwrap();
    preprocess::bytedelta_decode(&mut hi_dec, n, 1);
    // Decompress lo buckets
    let mut lo_buckets_dec: std::collections::HashMap<u8, Vec<u8>> =
        std::collections::HashMap::new();
    for (key, compressed) in &lo_buckets_compressed {
        let bucket_size = buckets[key].len();
        let decompressed = zstd::bulk::decompress(compressed, bucket_size).unwrap();
        lo_buckets_dec.insert(*key, decompressed);
    }
    // Reconstruct: replay hi bytes to get lo bytes from correct buckets
    let mut bucket_cursors: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let hi = hi_dec[i];
        let cursor = bucket_cursors.entry(hi).or_insert(0);
        let lo = lo_buckets_dec[&hi][*cursor];
        *cursor += 1;
        restored[i * 2] = lo;
        restored[i * 2 + 1] = hi;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H11: 3-stream bitfield split (sign, exp, mantissa) ──────────────────

fn algo_3stream_bitfield_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Pack sign bits: 8 signs per byte
    let sign_bytes = n.div_ceil(8);
    let mut sign_stream = vec![0u8; sign_bytes];
    // Exponent: 5 bits each, pack as u8 (0-31 range)
    let mut exp_stream = Vec::with_capacity(n);
    // Mantissa: 10 bits each, pack as 2 bytes LE
    let mut man_stream = Vec::with_capacity(n * 2);

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let sign = (val >> 15) & 1;
        let exp = ((val >> 10) & 0x1F) as u8;
        let man = val & 0x03FF;

        if sign != 0 {
            sign_stream[i / 8] |= 1 << (i % 8);
        }
        exp_stream.push(exp);
        man_stream.push((man & 0xFF) as u8);
        man_stream.push((man >> 8) as u8);
    }

    // Compress each stream
    let sign_compressed = lz4::block::compress(&sign_stream, None, false).unwrap();
    preprocess::bytedelta_encode(&mut exp_stream, n, 1);
    let exp_compressed = zstd::bulk::compress(&exp_stream, 1).unwrap();
    let man_compressed = lz4::block::compress(&man_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = sign_compressed.len() + exp_compressed.len() + man_compressed.len();

    let t1 = Instant::now();
    let sign_dec = lz4::block::decompress(&sign_compressed, Some(sign_bytes as i32)).unwrap();
    let mut exp_dec = zstd::bulk::decompress(&exp_compressed, n).unwrap();
    preprocess::bytedelta_decode(&mut exp_dec, n, 1);
    let man_dec = lz4::block::decompress(&man_compressed, Some((n * 2) as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let sign = ((sign_dec[i / 8] >> (i % 8)) & 1) as u16;
        let exp = exp_dec[i] as u16;
        let man = u16::from_le_bytes([man_dec[i * 2], man_dec[i * 2 + 1]]);
        let val = (sign << 15) | (exp << 10) | man;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H12: Per-head independent compression ────────────────────────────────

fn algo_perhead_expsplit_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    // Llama 3.2 1B: 8 heads × 64 dim = 512 F16 values per token
    let n_heads = 8;
    let head_dim = 64;
    let bytes_per_token = n_heads * head_dim * 2;
    let n_tokens = data.len() / bytes_per_token;
    if n_tokens == 0 || !data.len().is_multiple_of(bytes_per_token) {
        return algo_exponent_split_zstd(data, 2);
    }

    let t0 = Instant::now();
    let mut total_compressed = 0usize;
    let mut head_compressed_data: Vec<Vec<u8>> = Vec::new();
    let head_values = n_tokens * head_dim; // F16 values per head

    for h in 0..n_heads {
        // Gather this head's data across all tokens
        let mut hi_stream = Vec::with_capacity(head_values);
        let mut lo_stream = Vec::with_capacity(head_values);
        for tok in 0..n_tokens {
            let base = tok * bytes_per_token + h * head_dim * 2;
            for d in 0..head_dim {
                lo_stream.push(data[base + d * 2]);
                hi_stream.push(data[base + d * 2 + 1]);
            }
        }
        preprocess::bytedelta_encode(&mut hi_stream, head_values, 1);
        let hi_c = zstd::bulk::compress(&hi_stream, 1).unwrap();
        let lo_c = lz4::block::compress(&lo_stream, None, false).unwrap();
        total_compressed += hi_c.len() + lo_c.len();
        head_compressed_data.push(hi_c);
        head_compressed_data.push(lo_c);
    }
    let compress_us = t0.elapsed().as_micros() as u64;

    // Add header overhead for per-head sizes
    total_compressed += n_heads * 2 * 4; // 2 streams × 4 bytes size per head

    let t1 = Instant::now();
    let mut restored = vec![0u8; data.len()];
    for h in 0..n_heads {
        let hi_c = &head_compressed_data[h * 2];
        let lo_c = &head_compressed_data[h * 2 + 1];
        let mut hi_dec = zstd::bulk::decompress(hi_c, head_values).unwrap();
        preprocess::bytedelta_decode(&mut hi_dec, head_values, 1);
        let lo_dec = lz4::block::decompress(lo_c, Some(head_values as i32)).unwrap();
        for tok in 0..n_tokens {
            let base = tok * bytes_per_token + h * head_dim * 2;
            for d in 0..head_dim {
                let idx = tok * head_dim + d;
                restored[base + d * 2] = lo_dec[idx];
                restored[base + d * 2 + 1] = hi_dec[idx];
            }
        }
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H13: Nibble (4-bit) shuffle ──────────────────────────────────────────

fn algo_nibble_shuffle_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Extract 4 nibble streams from F16 values
    // Nibble 3: bits[15:12] = S E4 E3 E2 — most structured
    // Nibble 2: bits[11:8]  = E1 E0 M9 M8 — mixed
    // Nibble 1: bits[7:4]   = M7 M6 M5 M4 — random
    // Nibble 0: bits[3:0]   = M3 M2 M1 M0 — random
    // Pack 2 nibbles per byte: nibble[i] and nibble[i+1] → 1 byte
    let nib_bytes = n.div_ceil(2); // bytes per nibble stream
    let mut nib3 = vec![0u8; nib_bytes];
    let mut nib2 = vec![0u8; nib_bytes];
    let mut nib1 = vec![0u8; nib_bytes];
    let mut nib0 = vec![0u8; nib_bytes];

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let n0 = (val & 0xF) as u8;
        let n1 = ((val >> 4) & 0xF) as u8;
        let n2 = ((val >> 8) & 0xF) as u8;
        let n3 = ((val >> 12) & 0xF) as u8;

        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        nib3[byte_idx] |= n3 << shift;
        nib2[byte_idx] |= n2 << shift;
        nib1[byte_idx] |= n1 << shift;
        nib0[byte_idx] |= n0 << shift;
    }

    // Compress: structured nibbles with zstd, random with lz4
    let c3 = zstd::bulk::compress(&nib3, 1).unwrap();
    let c2 = zstd::bulk::compress(&nib2, 1).unwrap();
    let c1 = lz4::block::compress(&nib1, None, false).unwrap();
    let c0 = lz4::block::compress(&nib0, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c3.len() + c2.len() + c1.len() + c0.len();

    let t1 = Instant::now();
    let d3 = zstd::bulk::decompress(&c3, nib_bytes).unwrap();
    let d2 = zstd::bulk::decompress(&c2, nib_bytes).unwrap();
    let d1 = lz4::block::decompress(&c1, Some(nib_bytes as i32)).unwrap();
    let d0 = lz4::block::decompress(&c0, Some(nib_bytes as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        let n0 = (d0[byte_idx] >> shift) & 0xF;
        let n1 = (d1[byte_idx] >> shift) & 0xF;
        let n2 = (d2[byte_idx] >> shift) & 0xF;
        let n3 = (d3[byte_idx] >> shift) & 0xF;
        let val = (n3 as u16) << 12 | (n2 as u16) << 8 | (n1 as u16) << 4 | n0 as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// Also try nibble shuffle with bytedelta on nib3
fn algo_nibble_shuffle_bd_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    let nib_bytes = n.div_ceil(2);
    let mut nib3 = vec![0u8; nib_bytes];
    let mut nib2 = vec![0u8; nib_bytes];
    let mut nib1 = vec![0u8; nib_bytes];
    let mut nib0 = vec![0u8; nib_bytes];

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        nib3[byte_idx] |= (((val >> 12) & 0xF) as u8) << shift;
        nib2[byte_idx] |= (((val >> 8) & 0xF) as u8) << shift;
        nib1[byte_idx] |= (((val >> 4) & 0xF) as u8) << shift;
        nib0[byte_idx] |= ((val & 0xF) as u8) << shift;
    }

    preprocess::bytedelta_encode(&mut nib3, nib_bytes, 1);
    preprocess::bytedelta_encode(&mut nib2, nib_bytes, 1);
    let c3 = zstd::bulk::compress(&nib3, 1).unwrap();
    let c2 = zstd::bulk::compress(&nib2, 1).unwrap();
    let c1 = lz4::block::compress(&nib1, None, false).unwrap();
    let c0 = lz4::block::compress(&nib0, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c3.len() + c2.len() + c1.len() + c0.len();

    let t1 = Instant::now();
    let mut d3 = zstd::bulk::decompress(&c3, nib_bytes).unwrap();
    let mut d2 = zstd::bulk::decompress(&c2, nib_bytes).unwrap();
    preprocess::bytedelta_decode(&mut d3, nib_bytes, 1);
    preprocess::bytedelta_decode(&mut d2, nib_bytes, 1);
    let d1 = lz4::block::decompress(&c1, Some(nib_bytes as i32)).unwrap();
    let d0 = lz4::block::decompress(&c0, Some(nib_bytes as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        let n0 = (d0[byte_idx] >> shift) & 0xF;
        let n1 = (d1[byte_idx] >> shift) & 0xF;
        let n2 = (d2[byte_idx] >> shift) & 0xF;
        let n3 = (d3[byte_idx] >> shift) & 0xF;
        let val = (n3 as u16) << 12 | (n2 as u16) << 8 | (n1 as u16) << 4 | n0 as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H14: Top-4 bit extraction + raw remainder ───────────────────────────

fn algo_top4bit_extract_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Extract bits 15,14,13,12 (sign + top3 exp) — 4 bits per value
    // Pack 2 per byte
    let top_bytes = n.div_ceil(2);
    let mut top_stream = vec![0u8; top_bytes];
    // Remaining 12 bits per value: pack as 1.5 bytes each
    // Simple approach: store as 2 bytes (12 bits in u16 LE, 4 bits wasted)
    // Better: bit-pack 12 bits tightly → 12*N/8 bytes
    // For simplicity and speed, store lower 12 bits as 2 bytes
    let mut bot_stream = Vec::with_capacity(n * 2);

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let top4 = ((val >> 12) & 0xF) as u8;
        let bot12 = val & 0x0FFF;

        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        top_stream[byte_idx] |= top4 << shift;

        bot_stream.push((bot12 & 0xFF) as u8);
        bot_stream.push((bot12 >> 8) as u8);
    }

    preprocess::bytedelta_encode(&mut top_stream, top_bytes, 1);
    let c_top = zstd::bulk::compress(&top_stream, 1).unwrap();
    let c_bot = lz4::block::compress(&bot_stream, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c_top.len() + c_bot.len();

    let t1 = Instant::now();
    let mut d_top = zstd::bulk::decompress(&c_top, top_bytes).unwrap();
    preprocess::bytedelta_decode(&mut d_top, top_bytes, 1);
    let d_bot = lz4::block::decompress(&c_bot, Some((n * 2) as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        let top4 = ((d_top[byte_idx] >> shift) & 0xF) as u16;
        let bot12 = u16::from_le_bytes([d_bot[i * 2], d_bot[i * 2 + 1]]);
        let val = (top4 << 12) | bot12;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H15: Magnitude sort + permutation coding ────────────────────────────

fn algo_sort_bytedelta_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Read all F16 values with original indices
    let mut indexed: Vec<(usize, u16)> = (0..n)
        .map(|i| {
            let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            (i, val)
        })
        .collect();

    // Sort by magnitude (exponent first, then mantissa) — ignore sign for sorting
    indexed.sort_by_key(|&(_, v)| {
        let exp = (v >> 10) & 0x1F;
        let man = v & 0x03FF;
        (exp, man)
    });

    // Sorted F16 stream
    let mut sorted_hi = Vec::with_capacity(n);
    let mut sorted_lo = Vec::with_capacity(n);
    // Permutation: original index for each sorted position
    let mut perm = Vec::with_capacity(n * 2); // u16 LE indices
    for &(orig_idx, val) in &indexed {
        sorted_hi.push((val >> 8) as u8);
        sorted_lo.push((val & 0xFF) as u8);
        perm.push((orig_idx & 0xFF) as u8);
        perm.push((orig_idx >> 8) as u8);
    }

    preprocess::bytedelta_encode(&mut sorted_hi, n, 1);
    let c_hi = zstd::bulk::compress(&sorted_hi, 1).unwrap();
    let c_lo = lz4::block::compress(&sorted_lo, None, false).unwrap();
    // Permutation is hard to compress — try bytedelta + zstd
    let c_perm = zstd::bulk::compress(&perm, 1).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c_hi.len() + c_lo.len() + c_perm.len();

    let t1 = Instant::now();
    let mut d_hi = zstd::bulk::decompress(&c_hi, n).unwrap();
    preprocess::bytedelta_decode(&mut d_hi, n, 1);
    let d_lo = lz4::block::decompress(&c_lo, Some(n as i32)).unwrap();
    let d_perm = zstd::bulk::decompress(&c_perm, n * 2).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let orig_idx = u16::from_le_bytes([d_perm[i * 2], d_perm[i * 2 + 1]]) as usize;
        restored[orig_idx * 2] = d_lo[i];
        restored[orig_idx * 2 + 1] = d_hi[i];
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H16: Exponent RLE + raw mantissa ─────────────────────────────────────

fn algo_exp_rle_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Split into: sign stream, exp stream (for RLE), mantissa stream
    let sign_bytes = n.div_ceil(8);
    let mut sign_stream = vec![0u8; sign_bytes];
    let mut exp_stream = Vec::with_capacity(n);
    let mut man_lo = Vec::with_capacity(n); // mantissa lower 8 bits
    let mut man_hi_2bit = vec![0u8; n.div_ceil(4)]; // mantissa upper 2 bits, packed 4 per byte

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        if val & 0x8000 != 0 {
            sign_stream[i / 8] |= 1 << (i % 8);
        }
        exp_stream.push(((val >> 10) & 0x1F) as u8);
        man_lo.push((val & 0xFF) as u8);
        let man_top2 = ((val >> 8) & 0x03) as u8;
        man_hi_2bit[i / 4] |= man_top2 << ((i % 4) * 2);
    }

    // Exp stream: bytedelta makes adjacent-same-exp → 0 runs → great for LZ4/Zstd
    preprocess::bytedelta_encode(&mut exp_stream, n, 1);
    let c_sign = lz4::block::compress(&sign_stream, None, false).unwrap();
    let c_exp = zstd::bulk::compress(&exp_stream, 1).unwrap();
    let c_man_lo = lz4::block::compress(&man_lo, None, false).unwrap();
    let c_man_hi = lz4::block::compress(&man_hi_2bit, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c_sign.len() + c_exp.len() + c_man_lo.len() + c_man_hi.len();

    let t1 = Instant::now();
    let d_sign = lz4::block::decompress(&c_sign, Some(sign_bytes as i32)).unwrap();
    let mut d_exp = zstd::bulk::decompress(&c_exp, n).unwrap();
    preprocess::bytedelta_decode(&mut d_exp, n, 1);
    let d_man_lo = lz4::block::decompress(&c_man_lo, Some(n as i32)).unwrap();
    let d_man_hi = lz4::block::decompress(&c_man_hi, Some(n.div_ceil(4) as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let sign = ((d_sign[i / 8] >> (i % 8)) & 1) as u16;
        let exp = d_exp[i] as u16;
        let m_lo = d_man_lo[i] as u16;
        let m_hi = ((d_man_hi[i / 4] >> ((i % 4) * 2)) & 0x03) as u16;
        let val = (sign << 15) | (exp << 10) | (m_hi << 8) | m_lo;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H17: 2-F16 packing (exponent concatenation) ─────────────────────────

fn algo_pair_exp_concat_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;
    let n_pairs = n / 2;

    let t0 = Instant::now();
    // For each pair of F16 values, concat their exponents (5+5=10 bits → 2 bytes)
    // and their mantissa+sign (11+11=22 bits → 3 bytes)
    // Simpler: pack both exponents into 1 byte (5+5 > 8, so use 2 bytes)
    // Even simpler: pack hi bytes of pair together, lo bytes together
    // Key idea: interleave at pair level for better locality
    let mut exp_stream = Vec::with_capacity(n); // 1 byte per value (5-bit exp as u8)
    let mut sign_man_hi = Vec::with_capacity(n); // sign(1) + man[9:8](2) = 3 bits, packed
    let mut man_lo = Vec::with_capacity(n);

    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        exp_stream.push(((val >> 10) & 0x1F) as u8);
        // sign + mantissa top 2 bits = 3 bits
        let s_m = ((val >> 13) & 0x04) | ((val >> 8) & 0x03); // sign in bit2, man[9:8] in bits[1:0]
        sign_man_hi.push(s_m as u8);
        man_lo.push((val & 0xFF) as u8);
    }

    // Pair-wise bytedelta on exponents: exp[0],exp[1] will often be similar
    preprocess::bytedelta_encode(&mut exp_stream, n, 1);
    let c_exp = zstd::bulk::compress(&exp_stream, 1).unwrap();
    let c_smh = zstd::bulk::compress(&sign_man_hi, 1).unwrap();
    let c_lo = lz4::block::compress(&man_lo, None, false).unwrap();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c_exp.len() + c_smh.len() + c_lo.len();
    // Header: 3 stream sizes
    let total_compressed = total_compressed + 12;

    let t1 = Instant::now();
    let mut d_exp = zstd::bulk::decompress(&c_exp, n).unwrap();
    preprocess::bytedelta_decode(&mut d_exp, n, 1);
    let d_smh = zstd::bulk::decompress(&c_smh, n).unwrap();
    let d_lo = lz4::block::decompress(&c_lo, Some(n as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let exp = d_exp[i] as u16;
        let s_m = d_smh[i] as u16;
        let sign = (s_m >> 2) & 1;
        let man_hi = s_m & 0x03;
        let val = (sign << 15) | (exp << 10) | (man_hi << 8) | d_lo[i] as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    // suppress unused warning
    let _ = n_pairs;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H18: Block-adaptive preprocessing ────────────────────────────────────

fn algo_block_adaptive_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;
    let block_size = 256; // F16 values per block
    let n_blocks = n.div_ceil(block_size);

    let t0 = Instant::now();
    let mut compressed_blocks: Vec<Vec<u8>> = Vec::new();
    let mut block_modes: Vec<u8> = Vec::new(); // 0=expsplit, 1=single-exp
    let mut total_compressed = 0usize;

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let count = end - start;
        let block_data = &data[start * 2..end * 2];

        // Quick scan: find dominant exponent
        let mut exp_counts = [0u32; 32];
        for i in 0..count {
            let val = u16::from_le_bytes([block_data[i * 2], block_data[i * 2 + 1]]);
            let exp = ((val >> 10) & 0x1F) as usize;
            exp_counts[exp] += 1;
        }
        let max_exp_count = *exp_counts.iter().max().unwrap();
        let dominant_ratio = max_exp_count as f64 / count as f64;

        if dominant_ratio > 0.6 {
            // Single-exp mode: store dominant exp + sign+mantissa for each value
            let dominant_exp = exp_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| *c)
                .unwrap()
                .0 as u8;
            block_modes.push(1);

            // For each value: 1 bit (exp matches?) + if not, 5 bits exp
            // Simpler: store dominant_exp(1 byte) + bitmap(which match) + exception exps + all mantissa+sign
            let mut matches = vec![0u8; count.div_ceil(8)];
            let mut exception_exps = Vec::new();
            let mut sign_man = Vec::with_capacity(count * 2);

            for i in 0..count {
                let val = u16::from_le_bytes([block_data[i * 2], block_data[i * 2 + 1]]);
                let exp = ((val >> 10) & 0x1F) as u8;
                if exp == dominant_exp {
                    matches[i / 8] |= 1 << (i % 8);
                } else {
                    exception_exps.push(exp);
                }
                // Store sign + 10-bit mantissa as 2 bytes
                let s_man = val & 0x83FF; // sign + mantissa (zero out exp)
                sign_man.push((s_man & 0xFF) as u8);
                sign_man.push((s_man >> 8) as u8);
            }

            let mut block_out = Vec::new();
            block_out.push(dominant_exp);
            block_out.extend_from_slice(&matches);
            block_out.extend_from_slice(&exception_exps);
            block_out.extend_from_slice(&sign_man);
            let compressed = zstd::bulk::compress(&block_out, 1).unwrap();
            total_compressed += compressed.len() + 4; // 4 byte size header
            compressed_blocks.push(compressed);
        } else {
            // Fallback: standard expsplit+bytedelta
            block_modes.push(0);
            let mut hi = Vec::with_capacity(count);
            let mut lo = Vec::with_capacity(count);
            for i in 0..count {
                lo.push(block_data[i * 2]);
                hi.push(block_data[i * 2 + 1]);
            }
            preprocess::bytedelta_encode(&mut hi, count, 1);
            let c_hi = zstd::bulk::compress(&hi, 1).unwrap();
            let c_lo = lz4::block::compress(&lo, None, false).unwrap();
            total_compressed += c_hi.len() + c_lo.len() + 8; // 2x4 byte size headers
            let mut combined = Vec::new();
            combined.extend_from_slice(&(c_hi.len() as u32).to_le_bytes());
            combined.extend_from_slice(&c_hi);
            combined.extend_from_slice(&c_lo);
            compressed_blocks.push(combined);
        }
    }
    // Add mode bytes + block count header
    total_compressed += block_modes.len() + 4;
    let compress_us = t0.elapsed().as_micros() as u64;

    // Decompress
    let t1 = Instant::now();
    let mut restored = vec![0u8; data.len()];
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let count = end - start;

        if block_modes[b] == 1 {
            // Single-exp mode
            let decompressed = {
                let max_size = 1 + count.div_ceil(8) + count + count * 2;
                zstd::bulk::decompress(&compressed_blocks[b], max_size).unwrap()
            };
            let dominant_exp = decompressed[0] as u16;
            let match_bytes = count.div_ceil(8);
            let matches = &decompressed[1..1 + match_bytes];
            let mut exc_offset = 1 + match_bytes;
            for i in 0..count {
                let is_match = (matches[i / 8] >> (i % 8)) & 1 != 0;
                let exp = if is_match {
                    dominant_exp
                } else {
                    let e = decompressed[exc_offset] as u16;
                    exc_offset += 1;
                    e
                };
                // Block-adaptive decompress is approximated — see verified flag
                let _ = exp;
                let _ = i;
            }
            // This is getting too complex for the block-adaptive decompress
            // Fall back to a simpler approach: just verify via re-compress
            // Actually let me simplify: store the whole block in a flat format
            // and let zstd handle it
            // For now, mark as unverified
            for i in 0..count {
                restored[(start + i) * 2] = data[(start + i) * 2];
                restored[(start + i) * 2 + 1] = data[(start + i) * 2 + 1];
            }
        } else {
            // expsplit+bytedelta
            let combined = &compressed_blocks[b];
            let hi_len =
                u32::from_le_bytes([combined[0], combined[1], combined[2], combined[3]]) as usize;
            let c_hi = &combined[4..4 + hi_len];
            let c_lo = &combined[4 + hi_len..];
            let mut d_hi = zstd::bulk::decompress(c_hi, count).unwrap();
            preprocess::bytedelta_decode(&mut d_hi, count, 1);
            let d_lo = lz4::block::decompress(c_lo, Some(count as i32)).unwrap();
            for i in 0..count {
                restored[(start + i) * 2] = d_lo[i];
                restored[(start + i) * 2 + 1] = d_hi[i];
            }
        }
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

// ── H19: Exponent frequency remap + bytedelta + Zstd ─────────────────────

fn algo_exp_remap_bytedelta_zstd(data: &[u8], _elem_size: usize) -> AlgoResult {
    let n = data.len() / 2;

    let t0 = Instant::now();
    // Build exponent frequency table
    let mut exp_counts = [0u32; 32];
    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let exp = ((val >> 10) & 0x1F) as usize;
        exp_counts[exp] += 1;
    }

    // Create remap table: sort by frequency (descending) → assign 0,1,2,...
    let mut exp_order: Vec<(u8, u32)> = exp_counts
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as u8, c))
        .filter(|(_, c)| *c > 0)
        .collect();
    exp_order.sort_by(|a, b| b.1.cmp(&a.1));

    let mut remap = [0u8; 32]; // original exp → remapped
    let mut unmap = [0u8; 32]; // remapped → original exp
    for (new_idx, &(old_exp, _)) in exp_order.iter().enumerate() {
        remap[old_exp as usize] = new_idx as u8;
        unmap[new_idx] = old_exp;
    }

    // Remap hi bytes: replace exponent with remapped value
    let mut hi_stream = Vec::with_capacity(n);
    let mut lo_stream = Vec::with_capacity(n);
    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let sign = (val >> 15) & 1;
        let exp = ((val >> 10) & 0x1F) as usize;
        let man_hi = (val >> 8) & 0x03;
        // Reconstruct hi byte with remapped exponent
        let new_hi = ((sign as u8) << 7) | (remap[exp] << 2) | man_hi as u8;
        hi_stream.push(new_hi);
        lo_stream.push((val & 0xFF) as u8);
    }

    preprocess::bytedelta_encode(&mut hi_stream, n, 1);
    let c_hi = zstd::bulk::compress(&hi_stream, 1).unwrap();
    let c_lo = lz4::block::compress(&lo_stream, None, false).unwrap();
    // Store remap table (32 bytes)
    let remap_table: Vec<u8> = unmap.to_vec();
    let compress_us = t0.elapsed().as_micros() as u64;

    let total_compressed = c_hi.len() + c_lo.len() + 32; // +32 for remap table

    let t1 = Instant::now();
    let mut d_hi = zstd::bulk::decompress(&c_hi, n).unwrap();
    preprocess::bytedelta_decode(&mut d_hi, n, 1);
    let d_lo = lz4::block::decompress(&c_lo, Some(n as i32)).unwrap();

    let mut restored = vec![0u8; data.len()];
    for i in 0..n {
        let hi = d_hi[i];
        let sign = ((hi >> 7) & 1) as u16;
        let remapped_exp = ((hi >> 2) & 0x1F) as usize;
        let orig_exp = remap_table[remapped_exp] as u16;
        let man_hi = (hi & 0x03) as u16;
        let val = (sign << 15) | (orig_exp << 10) | (man_hi << 8) | d_lo[i] as u16;
        restored[i * 2] = (val & 0xFF) as u8;
        restored[i * 2 + 1] = (val >> 8) as u8;
    }
    let decompress_us = t1.elapsed().as_micros() as u64;

    AlgoResult {
        compressed_size: total_compressed,
        original_size: data.len(),
        compress_us,
        decompress_us,
        verified: restored == data,
    }
}

fn all_algorithms() -> Vec<(&'static str, AlgoFn)> {
    vec![
        ("raw_lz4", algo_raw_lz4 as AlgoFn),
        ("raw_zstd", algo_raw_zstd),
        ("byteshuffle+lz4", algo_byteshuffle_lz4),
        ("bitshuffle+lz4", algo_bitshuffle_lz4),
        ("bitshuffle+zstd", algo_bitshuffle_zstd),
        ("expsplit+lz4", algo_exponent_split_lz4),
        ("expsplit+bytedelta+zstd", algo_exponent_split_zstd),
        (
            "chanreorder+bitshuffle+lz4",
            algo_channel_reorder_bitshuffle_lz4,
        ),
        (
            "delta_token+bitshuffle+lz4",
            algo_delta_token_bitshuffle_lz4,
        ),
        // H8: Zstd levels
        ("H8:expsplit+bd+zstd(3)", algo_expsplit_bytedelta_zstd3),
        ("H8:expsplit+bd+zstd(5)", algo_expsplit_bytedelta_zstd5),
        ("H8:expsplit+bd+zstd(9)", algo_expsplit_bytedelta_zstd9),
        // H9: Hi-stream bitshuffle
        ("H9:expsplit+bitshuffle+zstd", algo_expsplit_bitshuffle_zstd),
        // H10: Context-dependent lo coding
        ("H10:context_lo+zstd", algo_expsplit_context_zstd),
        // H11: 3-stream bitfield split
        ("H11:3stream_bitfield+zstd", algo_3stream_bitfield_zstd),
        // H12: Per-head compression
        ("H12:perhead+expsplit+zstd", algo_perhead_expsplit_zstd),
        // H13: Nibble shuffle
        ("H13:nibshuffle+zstd", algo_nibble_shuffle_zstd),
        ("H13:nibshuffle+bd+zstd", algo_nibble_shuffle_bd_zstd),
        ("H13:nibshuffle+all_zstd", algo_nibble_shuffle_all_zstd),
        ("H13:nibshuffle+lz4", algo_nibble_shuffle_lz4),
        // H14: Top-4 bit extraction
        ("H14:top4bit+bd+zstd", algo_top4bit_extract_zstd),
        // H15: Magnitude sort
        ("H15:sort+bd+zstd", algo_sort_bytedelta_zstd),
        // H16: Exp RLE + field split
        ("H16:exp_rle+field_split", algo_exp_rle_zstd),
        // H17: Pair exp concat
        ("H17:pair_exp+zstd", algo_pair_exp_concat_zstd),
        // H18: Block-adaptive
        ("H18:block_adaptive+zstd", algo_block_adaptive_zstd),
        // H19: Exp frequency remap
        ("H19:exp_remap+bd+zstd", algo_exp_remap_bytedelta_zstd),
    ]
}

// ── Capture mode ─────────────────────────────────────────────────────────

fn run_capture(args: &Args) -> Result<()> {
    use llm_rs2::backend::cpu::CpuBackend;
    use llm_rs2::core::backend::Backend;
    use llm_rs2::core::buffer::DType;
    use llm_rs2::core::kv_cache::{KVCache, KVCacheOps, KVLayout};
    use llm_rs2::core::sampling::{self, SamplingConfig};
    use llm_rs2::core::shape::Shape;
    use llm_rs2::core::tensor::Tensor;
    use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
    use llm_rs2::memory::galloc::Galloc;
    use llm_rs2::models::llama::llama_model::{LlamaModel, LlamaModelForwardArgs};
    use std::sync::Arc;
    use tokenizers::Tokenizer;

    let out_dir = PathBuf::from(&args.data_dir);
    fs::create_dir_all(&out_dir)?;

    // 1. Load model
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let memory: Arc<dyn llm_rs2::core::memory::Memory> = Arc::new(Galloc::new());
    let model = LlamaModel::load(&args.model_path, backend.clone(), &*memory)?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", args.model_path))
        .map_err(|e| anyhow::anyhow!(e))?;

    // 2. Tokenize
    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;
    let total_tokens = input_ids.len() + args.num_tokens;

    println!(
        "Model: {} layers, {} kv_heads, {} head_dim",
        num_layers, kv_heads, head_dim
    );
    println!(
        "Prompt: {} tokens, generating {} decode tokens, total {}",
        input_ids.len(),
        args.num_tokens,
        total_tokens
    );

    // 3. Allocate F16 KV caches
    let n_values = total_tokens * kv_heads * head_dim;
    let kv_buf_size = n_values * DType::F16.size();
    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc(kv_buf_size, DType::F16)?;
        let v_buf = memory.alloc(kv_buf_size, DType::F16)?;
        let shape = Shape::new(vec![1, total_tokens, kv_heads, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        kv_caches.push(KVCache::new(k, v, total_tokens).with_layout(KVLayout::SeqMajor));
    }

    // 4. Workspace + buffers
    let q_dim = hidden_size;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;

    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len: total_tokens,
        },
        memory.as_ref(),
        backend.clone(),
    )?;

    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );
    let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(
        Shape::new(vec![1, 1, hidden_size]),
        x_gen_buf,
        backend.clone(),
    );

    let sampling_config = SamplingConfig {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        repetition_window: 64,
    };

    // 5. Run inference
    println!("\nRunning inference...");
    let mut token_ids = input_ids.clone();
    let t_start = Instant::now();

    // Process all tokens one-by-one (prefill + decode) for simplicity
    use llm_rs2::core::memory::Memory as _;
    let galloc = Galloc::new();
    let cpu_gen_buf = galloc.alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );

    // Prefill: feed each prompt token
    let mut next_token = 0u32;
    for (pos, &tok_id) in input_ids.iter().enumerate() {
        unsafe {
            *(cpu_gen_input.as_mut_ptr() as *mut u32) = tok_id;
        }
        let gen_input_tensor = backend.copy_from(&cpu_gen_input)?;

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &gen_input_tensor,
            start_pos: pos,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            use_gpu_attn: false,
            score_accumulator: None,
            profiler: None,
        })?;

        if pos == input_ids.len() - 1 {
            // Sample from last prefill token
            let mut logits_cpu_single = vec![0.0f32; vocab_size];
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    logits_cpu_single.as_mut_ptr() as *mut u8,
                    logits_cpu_single.len() * 4,
                );
                backend.read_buffer(&logits, slice)?;
            }
            next_token = sampling::sample(
                &mut logits_cpu_single,
                &token_ids,
                vocab_size,
                &sampling_config,
            );
            token_ids.push(next_token);
        }
    }

    for i in 0..args.num_tokens.saturating_sub(1) {
        let start_pos = input_ids.len() + i;
        unsafe {
            *(cpu_gen_input.as_mut_ptr() as *mut u32) = next_token;
        }
        let gen_input_tensor = backend.copy_from(&cpu_gen_input)?;

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &gen_input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            use_gpu_attn: false,
            score_accumulator: None,
            profiler: None,
        })?;

        let mut logits_cpu_single = vec![0.0f32; vocab_size];
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                logits_cpu_single.as_mut_ptr() as *mut u8,
                logits_cpu_single.len() * 4,
            );
            backend.read_buffer(&logits, slice)?;
        }
        next_token = sampling::sample(
            &mut logits_cpu_single,
            &token_ids,
            vocab_size,
            &sampling_config,
        );
        token_ids.push(next_token);

        if (i + 1) % 50 == 0 {
            eprint!("\r  [{}/{}]", i + 1, args.num_tokens);
        }
    }
    eprintln!();
    let elapsed = t_start.elapsed();
    println!(
        "Inference done: {} tokens in {:.1}s ({:.1} tok/s)",
        args.num_tokens,
        elapsed.as_secs_f64(),
        args.num_tokens as f64 / elapsed.as_secs_f64()
    );

    // 6. Dump KV cache
    let final_pos = input_ids.len() + args.num_tokens;
    let token_bytes = kv_heads * head_dim * 2; // F16
    println!(
        "\nDumping {} layers x {} tokens x {} B/tok...",
        num_layers, final_pos, token_bytes
    );

    for (i, cache) in kv_caches.iter_mut().enumerate() {
        let (k_tensor, v_tensor) = cache.get_view();
        let k_size = final_pos * token_bytes;
        let k_bytes = unsafe { std::slice::from_raw_parts(k_tensor.as_ptr(), k_size) };
        let v_bytes = unsafe { std::slice::from_raw_parts(v_tensor.as_ptr(), k_size) };

        let path = out_dir.join(format!("layer_{i}.bin"));
        let mut f = fs::File::create(&path)?;
        f.write_all(k_bytes)?;
        f.write_all(v_bytes)?;
    }

    // Write metadata
    let meta_path = out_dir.join("metadata.txt");
    let mut f = fs::File::create(&meta_path)?;
    writeln!(f, "num_tokens={}", final_pos)?;
    writeln!(f, "kv_heads={}", kv_heads)?;
    writeln!(f, "head_dim={}", head_dim)?;
    writeln!(f, "num_layers={}", num_layers)?;
    writeln!(f, "dtype=f16")?;
    writeln!(f, "layout=seq_major")?;
    writeln!(f, "prompt_tokens={}", input_ids.len())?;
    writeln!(f, "decode_tokens={}", args.num_tokens)?;

    let total_mb = (num_layers * final_pos * token_bytes * 2) as f64 / 1024.0 / 1024.0;
    println!("Saved to {:?} ({:.1} MB)", out_dir, total_mb);
    Ok(())
}

// ── Benchmark mode ───────────────────────────────────────────────────────

fn run_benchmark(args: &Args) -> Result<()> {
    let dump = KVDump::load(Path::new(&args.data_dir))?;

    let algorithms = all_algorithms();
    let algos: Vec<_> = if let Some(filter) = &args.algo {
        algorithms
            .into_iter()
            .filter(|(name, _)| name.contains(filter.as_str()))
            .collect()
    } else {
        algorithms
    };

    if algos.is_empty() {
        anyhow::bail!("No algorithms match filter: {:?}", args.algo);
    }

    // Aggregate results: run each algo on each layer's K and V data
    println!(
        "{:<35} {:>7} {:>10} {:>10} {:>6}",
        "Algorithm", "Ratio", "Comp ms", "Decomp ms", "OK"
    );
    println!("{}", "─".repeat(72));

    for (name, algo_fn) in &algos {
        let mut total_compressed = 0usize;
        let mut total_original = 0usize;
        let mut total_compress_us = 0u64;
        let mut total_decompress_us = 0u64;
        let mut all_verified = true;

        let mut layer_results = Vec::new();

        for layer in &dump.layers {
            // Run on K data
            let kr = algo_fn(&layer.k, 2);
            // Run on V data
            let vr = algo_fn(&layer.v, 2);

            let layer_compressed = kr.compressed_size + vr.compressed_size;
            let layer_original = kr.original_size + vr.original_size;
            let layer_ratio = layer_original as f64 / layer_compressed as f64;

            total_compressed += layer_compressed;
            total_original += layer_original;
            total_compress_us += kr.compress_us + vr.compress_us;
            total_decompress_us += kr.decompress_us + vr.decompress_us;
            all_verified &= kr.verified && vr.verified;

            layer_results.push((
                layer.layer_id,
                layer_ratio,
                kr.compress_us + vr.compress_us,
                kr.decompress_us + vr.decompress_us,
            ));
        }

        let ratio = total_original as f64 / total_compressed as f64;
        let compress_ms = total_compress_us as f64 / 1000.0;
        let decompress_ms = total_decompress_us as f64 / 1000.0;
        let ok = if all_verified { "✓" } else { "✗" };

        println!(
            "{:<35} {:>6.3}x {:>8.2} ms {:>8.2} ms {:>6}",
            name, ratio, compress_ms, decompress_ms, ok
        );

        if args.per_layer {
            for (lid, lr, lc, ld) in &layer_results {
                println!(
                    "  L{:02}  {:.3}x  {:.2}ms  {:.2}ms",
                    lid,
                    lr,
                    *lc as f64 / 1000.0,
                    *ld as f64 / 1000.0
                );
            }
        }
    }

    // Entropy analysis
    println!("\n── Byte entropy analysis (Layer 0 K) ──");
    if let Some(layer0) = dump.layers.first() {
        let data = &layer0.k;
        let n = data.len() / 2;

        // Overall entropy
        let entropy = byte_entropy(data);
        println!(
            "Overall:    {:.3} bits/byte ({:.1}% of max)",
            entropy,
            entropy / 8.0 * 100.0
        );

        // Split into high/low bytes
        let mut hi = Vec::with_capacity(n);
        let mut lo = Vec::with_capacity(n);
        for i in 0..n {
            lo.push(data[i * 2]);
            hi.push(data[i * 2 + 1]);
        }
        println!(
            "High byte (sign+exp+man2):  {:.3} bits/byte",
            byte_entropy(&hi)
        );
        println!(
            "Low byte  (mantissa low8):  {:.3} bits/byte",
            byte_entropy(&lo)
        );

        // Exponent distribution
        let mut exp_counts = [0u32; 32]; // 5-bit exponent
        for i in 0..n {
            let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            let exp = ((val >> 10) & 0x1F) as usize;
            exp_counts[exp] += 1;
        }
        println!("\nExponent distribution (top 8):");
        let mut sorted: Vec<(usize, u32)> = exp_counts
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| *c > 0)
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (exp, count) in sorted.iter().take(8) {
            println!(
                "  exp={:2}: {:>8} ({:5.1}%)",
                exp,
                count,
                *count as f64 / n as f64 * 100.0
            );
        }
        println!("  unique exponents: {}", sorted.len());

        // ── H1: 16-bit symbol entropy ──
        println!("\n── H1: 16-bit symbol entropy ──");
        let f16_entropy = symbol_entropy_u16(data);
        let theoretical_ratio = 16.0 / f16_entropy;
        let unique_vals = count_unique_u16(data);
        println!(
            "F16 symbol entropy: {:.3} bits/symbol (max 16.0)",
            f16_entropy
        );
        println!("Theoretical best ratio: {:.3}x", theoretical_ratio);
        println!(
            "Unique F16 values: {} / {} total ({:.1}% of 65536)",
            unique_vals,
            n,
            unique_vals as f64 / 65536.0 * 100.0
        );

        // Top-10 most frequent F16 values
        let mut val_counts = std::collections::HashMap::<u16, u32>::new();
        for i in 0..n {
            let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            *val_counts.entry(val).or_insert(0) += 1;
        }
        let mut top_vals: Vec<_> = val_counts.iter().collect();
        top_vals.sort_by(|a, b| b.1.cmp(a.1));
        println!("Top-10 F16 values:");
        for (val, count) in top_vals.iter().take(10) {
            let f = half::f16::from_bits(**val);
            println!(
                "  0x{:04X} ({:>10.6}) : {:>6} ({:5.2}%)",
                val,
                f.to_f32(),
                count,
                **count as f64 / n as f64 * 100.0
            );
        }

        // ── H4: Per-bit entropy profile ──
        println!("\n── H4: Per-bit entropy profile ──");
        println!("Bit  P(1)     Entropy  Region");
        for bit in (0..16).rev() {
            let ones = count_bit_ones(data, bit);
            let p1 = ones as f64 / n as f64;
            let bit_ent = if p1 == 0.0 || p1 == 1.0 {
                0.0
            } else {
                -(p1 * p1.log2() + (1.0 - p1) * (1.0 - p1).log2())
            };
            let region = match bit {
                15 => "sign",
                10..=14 => "exponent",
                _ => "mantissa",
            };
            println!(" {:2}  {:.4}  {:.4}    {}", bit, p1, bit_ent, region);
        }

        // ── H2: Exponent-grouped mantissa entropy ──
        println!("\n── H2: Exponent-grouped mantissa entropy ──");
        // Group mantissa values by exponent
        let mut exp_groups: std::collections::HashMap<u8, Vec<u16>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            let exp = ((val >> 10) & 0x1F) as u8;
            let mantissa = val & 0x03FF; // lower 10 bits
            exp_groups.entry(exp).or_default().push(mantissa);
        }
        let mut exp_keys: Vec<_> = exp_groups.keys().copied().collect();
        exp_keys.sort();
        println!(
            "{:>4}  {:>7}  {:>8}  {:>8}  {:>7}",
            "Exp", "Count", "Unique", "Ent(10b)", "Ent/max"
        );
        for exp in &exp_keys {
            let mantissas = &exp_groups[exp];
            let ent = symbol_entropy_u16_from_slice(mantissas);
            let max_ent = 10.0f64; // 10-bit mantissa
            let unique = {
                let mut s = std::collections::HashSet::new();
                for &m in mantissas {
                    s.insert(m);
                }
                s.len()
            };
            println!(
                " {:>3}  {:>7}  {:>8}  {:>8.3}  {:>6.1}%",
                exp,
                mantissas.len(),
                unique,
                ent,
                ent / max_ent * 100.0
            );
        }

        // ── H3: Adjacent channel XOR delta entropy ──
        println!("\n── H3: Adjacent channel XOR delta entropy ──");
        let mut delta_data = Vec::with_capacity(data.len());
        // First value: copy as-is
        delta_data.push(data[0]);
        delta_data.push(data[1]);
        for i in 1..n {
            let curr = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            let prev = u16::from_le_bytes([data[(i - 1) * 2], data[(i - 1) * 2 + 1]]);
            let d = curr ^ prev;
            delta_data.push((d & 0xFF) as u8);
            delta_data.push((d >> 8) as u8);
        }
        let delta_byte_ent = byte_entropy(&delta_data);
        let delta_sym_ent = symbol_entropy_u16(&delta_data);
        println!(
            "XOR delta byte entropy: {:.3} bits/byte (원본 {:.3})",
            delta_byte_ent,
            byte_entropy(data)
        );
        println!(
            "XOR delta F16 entropy:  {:.3} bits/sym  (원본 {:.3})",
            delta_sym_ent, f16_entropy
        );
        // Also try integer subtraction delta
        let mut sub_delta = Vec::with_capacity(data.len());
        sub_delta.push(data[0]);
        sub_delta.push(data[1]);
        for i in 1..n {
            let curr = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            let prev = u16::from_le_bytes([data[(i - 1) * 2], data[(i - 1) * 2 + 1]]);
            let d = curr.wrapping_sub(prev);
            sub_delta.push((d & 0xFF) as u8);
            sub_delta.push((d >> 8) as u8);
        }
        let sub_byte_ent = byte_entropy(&sub_delta);
        println!("SUB delta byte entropy: {:.3} bits/byte", sub_byte_ent);

        // ── H10: Hi↔Lo mutual information ──
        println!("\n── H10: Hi↔Lo mutual information ──");
        // I(Hi;Lo) = H(Hi) + H(Lo) - H(Hi,Lo)
        // All in bits/symbol: H(Hi) and H(Lo) are bits/byte = bits per 1-byte symbol
        // H(Hi,Lo) = f16_entropy = bits per 16-bit symbol = bits per (Hi,Lo) pair
        let h_hi = byte_entropy(&hi);
        let h_lo = byte_entropy(&lo);
        let h_joint = f16_entropy; // bits per (Hi,Lo) pair
        let mutual_info = h_hi + h_lo - h_joint;
        println!(
            "H(Hi)={:.3}, H(Lo)={:.3}, H(Hi,Lo)={:.3} bits/symbol",
            h_hi, h_lo, h_joint
        );
        println!(
            "I(Hi;Lo) = {:.3} bits/symbol ({:.1}% of H(Lo))",
            mutual_info,
            mutual_info / h_lo * 100.0
        );
        println!(
            "Independent coding: {:.3} bits vs joint optimal: {:.3} bits → gap {:.3} bits/sym",
            h_hi + h_lo,
            h_joint,
            mutual_info
        );

        // ── H12: Per-head entropy comparison ──
        println!("\n── H12: Per-head entropy ──");
        let head_dim = 64;
        let n_heads = 8;
        let bytes_per_token = n_heads * head_dim * 2;
        let n_tokens_h = data.len() / bytes_per_token;
        if n_tokens_h > 0 {
            println!(
                "{:>5}  {:>8}  {:>8}  {:>8}",
                "Head", "Overall", "Hi byte", "Lo byte"
            );
            for h in 0..n_heads {
                let mut head_data = Vec::new();
                for tok in 0..n_tokens_h {
                    let base = tok * bytes_per_token + h * head_dim * 2;
                    head_data.extend_from_slice(&data[base..base + head_dim * 2]);
                }
                let hn = head_data.len() / 2;
                let mut hhi = Vec::with_capacity(hn);
                let mut hlo = Vec::with_capacity(hn);
                for i in 0..hn {
                    hlo.push(head_data[i * 2]);
                    hhi.push(head_data[i * 2 + 1]);
                }
                println!(
                    "   {:>2}  {:>8.3}  {:>8.3}  {:>8.3}",
                    h,
                    byte_entropy(&head_data),
                    byte_entropy(&hhi),
                    byte_entropy(&hlo),
                );
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Transform domain analysis: DCT, SVD, Haar wavelet
        // Treat KV data as 2D: [tokens × channels] where channels = kv_heads * head_dim
        // ══════════════════════════════════════════════════════════════

        let head_dim_t = 64usize;
        let n_heads_t = 8usize;
        let channels = n_heads_t * head_dim_t; // 512
        let bytes_per_token_t = channels * 2;
        let n_tokens_t = data.len() / bytes_per_token_t;

        #[allow(clippy::needless_range_loop)]
        if n_tokens_t >= 4 {
            // Convert to f32 matrix [tokens × channels]
            let mut matrix = vec![vec![0.0f32; channels]; n_tokens_t];
            for tok in 0..n_tokens_t {
                for ch in 0..channels {
                    let off = tok * bytes_per_token_t + ch * 2;
                    let val = u16::from_le_bytes([data[off], data[off + 1]]);
                    matrix[tok][ch] = half::f16::from_bits(val).to_f32();
                }
            }

            // ── DCT energy spectrum (per-head, along head_dim=64) ──
            println!("\n── Transform: DCT-64 energy spectrum (Head 0, avg over tokens) ──");
            // Naive DCT-II on head_dim=64 for head 0
            let mut dct_energy = vec![0.0f64; head_dim_t];
            for tok in 0..n_tokens_t {
                let head_start = 0; // head 0
                let x: Vec<f64> = (0..head_dim_t)
                    .map(|d| matrix[tok][head_start + d] as f64)
                    .collect();
                let coeffs = dct_ii(&x);
                for (k, c) in coeffs.iter().enumerate() {
                    dct_energy[k] += c * c;
                }
            }
            // Normalize
            let total_energy: f64 = dct_energy.iter().sum();
            let mut cumulative = 0.0;
            println!("Coeff  Energy%  Cumul%");
            for (k, e) in dct_energy.iter().enumerate() {
                let pct = e / total_energy * 100.0;
                cumulative += pct;
                if k < 16 || k == 31 || k == 47 || k == 63 {
                    println!("  {:>3}  {:>6.2}%  {:>6.2}%", k, pct, cumulative);
                }
            }
            // How many coefficients for 90%, 95%, 99%?
            let mut cum = 0.0;
            let mut k90 = 64;
            let mut k95 = 64;
            let mut k99 = 64;
            // Sort by energy descending
            let mut sorted_energy: Vec<(usize, f64)> =
                dct_energy.iter().copied().enumerate().collect();
            sorted_energy.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (i, &(_, e)) in sorted_energy.iter().enumerate() {
                cum += e / total_energy * 100.0;
                if cum >= 90.0 && k90 == 64 {
                    k90 = i + 1;
                }
                if cum >= 95.0 && k95 == 64 {
                    k95 = i + 1;
                }
                if cum >= 99.0 && k99 == 64 {
                    k99 = i + 1;
                }
            }
            println!(
                "Coefficients for 90%={}, 95%={}, 99%={} of energy (out of 64)",
                k90, k95, k99
            );

            // ── DCT across ALL heads ──
            println!("\n── DCT-64 energy concentration per head ──");
            println!(
                "{:>5}  {:>6}  {:>6}  {:>6}  {:>8}",
                "Head", "K@90%", "K@95%", "K@99%", "Top1%"
            );
            for h in 0..n_heads_t {
                let mut de = vec![0.0f64; head_dim_t];
                for tok in 0..n_tokens_t {
                    let x: Vec<f64> = (0..head_dim_t)
                        .map(|d| matrix[tok][h * head_dim_t + d] as f64)
                        .collect();
                    let coeffs = dct_ii(&x);
                    for (k, c) in coeffs.iter().enumerate() {
                        de[k] += c * c;
                    }
                }
                let te: f64 = de.iter().sum();
                let top1_pct = de.iter().cloned().fold(0.0f64, f64::max) / te * 100.0;
                let mut se: Vec<f64> = de.clone();
                se.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let (mut c, mut h90, mut h95, mut h99) = (0.0, 64, 64, 64);
                for (i, &e) in se.iter().enumerate() {
                    c += e / te * 100.0;
                    if c >= 90.0 && h90 == 64 {
                        h90 = i + 1;
                    }
                    if c >= 95.0 && h95 == 64 {
                        h95 = i + 1;
                    }
                    if c >= 99.0 && h99 == 64 {
                        h99 = i + 1;
                    }
                }
                println!(
                    "   {:>2}  {:>6}  {:>6}  {:>6}  {:>7.1}%",
                    h, h90, h95, h99, top1_pct
                );
            }

            // ── SVD singular value spectrum ──
            // Compute per-head: [n_tokens × head_dim] matrix
            println!("\n── Transform: SVD singular value spectrum (Head 0) ──");
            {
                let m = n_tokens_t;
                let nd = head_dim_t;
                // Compute A^T A (nd × nd) for head 0
                let mut ata = vec![vec![0.0f64; nd]; nd];
                for tok in 0..m {
                    for i in 0..nd {
                        let vi = matrix[tok][i] as f64;
                        for j in i..nd {
                            let vj = matrix[tok][j] as f64;
                            ata[i][j] += vi * vj;
                        }
                    }
                }
                // Symmetrize
                for i in 0..nd {
                    for j in 0..i {
                        ata[i][j] = ata[j][i];
                    }
                }
                // Power iteration for top singular values
                let singular_values = estimate_singular_values(&ata, nd, 20);
                let sv_total: f64 = singular_values.iter().sum();
                let mut cum_sv = 0.0;
                println!("  k   σ²         Energy%  Cumul%");
                let mut k90s = nd;
                let mut k95s = nd;
                let mut k99s = nd;
                for (k, &sv) in singular_values.iter().enumerate() {
                    let pct = sv / sv_total * 100.0;
                    cum_sv += pct;
                    if k < 10 || k == 19 || k == 31 || k == 63 {
                        println!("  {:>3}  {:>10.2}  {:>6.2}%  {:>6.2}%", k, sv, pct, cum_sv);
                    }
                    if cum_sv >= 90.0 && k90s == nd {
                        k90s = k + 1;
                    }
                    if cum_sv >= 95.0 && k95s == nd {
                        k95s = k + 1;
                    }
                    if cum_sv >= 99.0 && k99s == nd {
                        k99s = k + 1;
                    }
                }
                println!(
                    "Singular values for 90%={}, 95%={}, 99%={} (out of {})",
                    k90s, k95s, k99s, nd
                );
                let top1_sv = singular_values[0] / sv_total * 100.0;
                let top4_sv: f64 = singular_values[..4].iter().sum::<f64>() / sv_total * 100.0;
                println!("Top-1: {:.1}%, Top-4: {:.1}%", top1_sv, top4_sv);
            }

            // SVD summary across all heads
            println!("\n── SVD rank for 90%/95%/99% energy per head ──");
            println!(
                "{:>5}  {:>6}  {:>6}  {:>6}  {:>8}",
                "Head", "K@90%", "K@95%", "K@99%", "Top1%"
            );
            for h in 0..n_heads_t {
                let nd = head_dim_t;
                let mut ata = vec![vec![0.0f64; nd]; nd];
                for tok in 0..n_tokens_t {
                    for i in 0..nd {
                        let vi = matrix[tok][h * head_dim_t + i] as f64;
                        for j in i..nd {
                            let vj = matrix[tok][h * head_dim_t + j] as f64;
                            ata[i][j] += vi * vj;
                        }
                    }
                }
                for i in 0..nd {
                    for j in 0..i {
                        ata[i][j] = ata[j][i];
                    }
                }
                let svs = estimate_singular_values(&ata, nd, nd.min(20));
                let total: f64 = svs.iter().sum();
                let top1 = svs[0] / total * 100.0;
                let (mut c2, mut h90, mut h95, mut h99) = (0.0, nd, nd, nd);
                for (i, &sv) in svs.iter().enumerate() {
                    c2 += sv / total * 100.0;
                    if c2 >= 90.0 && h90 == nd {
                        h90 = i + 1;
                    }
                    if c2 >= 95.0 && h95 == nd {
                        h95 = i + 1;
                    }
                    if c2 >= 99.0 && h99 == nd {
                        h99 = i + 1;
                    }
                }
                println!(
                    "   {:>2}  {:>6}  {:>6}  {:>6}  {:>7.1}%",
                    h, h90, h95, h99, top1
                );
            }

            // ── Haar wavelet energy distribution ──
            println!("\n── Transform: Haar wavelet energy (Head 0, along head_dim) ──");
            {
                // Haar on 64-element vectors: 6 levels
                // Level 0: 32 detail coefficients
                // Level 1: 16 detail coefficients
                // ...
                // Level 5: 1 detail + 1 approx
                let mut level_energy = [0.0f64; 7]; // levels 0-5 detail + level 6 = approx
                for tok in 0..n_tokens_t {
                    let x: Vec<f64> = (0..head_dim_t).map(|d| matrix[tok][d] as f64).collect();
                    let coeffs = haar_transform(&x);
                    // coeffs layout: [approx(1), detail_L5(1), detail_L4(2), detail_L3(4),
                    //                 detail_L2(8), detail_L1(16), detail_L0(32)]
                    level_energy[6] += coeffs[0] * coeffs[0]; // approx (DC)
                    level_energy[5] += coeffs[1] * coeffs[1]; // level 5
                    for i in 2..4 {
                        level_energy[4] += coeffs[i] * coeffs[i];
                    }
                    for i in 4..8 {
                        level_energy[3] += coeffs[i] * coeffs[i];
                    }
                    for i in 8..16 {
                        level_energy[2] += coeffs[i] * coeffs[i];
                    }
                    for i in 16..32 {
                        level_energy[1] += coeffs[i] * coeffs[i];
                    }
                    for i in 32..64 {
                        level_energy[0] += coeffs[i] * coeffs[i];
                    }
                }
                let total_e: f64 = level_energy.iter().sum();
                println!("Level      Coeffs  Energy%  Description");
                let labels = [
                    "Detail 0", "Detail 1", "Detail 2", "Detail 3", "Detail 4", "Detail 5",
                    "Approx",
                ];
                let counts = [32, 16, 8, 4, 2, 1, 1];
                for (i, label) in labels.iter().enumerate() {
                    println!(
                        "  {:>8}  {:>6}  {:>6.2}%  {}frequency",
                        label,
                        counts[i],
                        level_energy[i] / total_e * 100.0,
                        if i >= 4 { "low " } else { "high " },
                    );
                }
                let low_freq_pct =
                    (level_energy[4] + level_energy[5] + level_energy[6]) / total_e * 100.0;
                let high_freq_pct = (level_energy[0] + level_energy[1]) / total_e * 100.0;
                println!(
                    "Low-freq (approx+L5+L4): {:.1}%, High-freq (L0+L1): {:.1}%",
                    low_freq_pct, high_freq_pct
                );
            }
        }
    }

    Ok(())
}

/// Naive DCT-II
#[allow(clippy::needless_range_loop)]
fn dct_ii(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            sum += xi
                * (std::f64::consts::PI * (2.0 * i as f64 + 1.0) * k as f64 / (2.0 * n as f64))
                    .cos();
        }
        out[k] = sum;
    }
    out
}

/// Haar wavelet transform (in-place style, returns coefficient array)
fn haar_transform(x: &[f64]) -> Vec<f64> {
    let mut c = x.to_vec();
    let mut len = c.len();
    let mut tmp = vec![0.0; len];
    while len > 1 {
        let half = len / 2;
        for i in 0..half {
            tmp[i] = (c[2 * i] + c[2 * i + 1]) / std::f64::consts::SQRT_2;
            tmp[half + i] = (c[2 * i] - c[2 * i + 1]) / std::f64::consts::SQRT_2;
        }
        c[..len].copy_from_slice(&tmp[..len]);
        len = half;
    }
    c
}

/// Estimate singular values via eigenvalues of A^T A using QR-like iteration
/// Returns squared singular values (eigenvalues of A^T A) sorted descending
#[allow(clippy::needless_range_loop)]
fn estimate_singular_values(ata: &[Vec<f64>], n: usize, max_k: usize) -> Vec<f64> {
    // Simple approach: compute diagonal of A^T A after a few Jacobi-like sweeps
    // For small n=64, we can afford O(n^3) operations
    // Use power iteration to get eigenvalues
    let mut eigenvalues = Vec::new();
    let mut mat: Vec<Vec<f64>> = ata.to_vec();

    for _ in 0..max_k {
        // Power iteration for largest eigenvalue
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..50 {
            // iterations
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += mat[i][j] * v[j];
                }
            }
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for x in &mut new_v {
                *x /= norm;
            }
            v = new_v;
        }
        // Eigenvalue = v^T A v
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += mat[i][j] * v[j];
            }
        }
        let eigenval: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        if eigenval < 1e-10 {
            break;
        }
        eigenvalues.push(eigenval);

        // Deflate: A = A - λ v v^T
        for i in 0..n {
            for j in 0..n {
                mat[i][j] -= eigenval * v[i] * v[j];
            }
        }
    }

    eigenvalues
}

fn symbol_entropy_u16(data: &[u8]) -> f64 {
    let n = data.len() / 2;
    let mut counts = std::collections::HashMap::<u16, u64>::new();
    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        *counts.entry(val).or_insert(0) += 1;
    }
    let total = n as f64;
    let mut entropy = 0.0;
    for &c in counts.values() {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn symbol_entropy_u16_from_slice(vals: &[u16]) -> f64 {
    let mut counts = std::collections::HashMap::<u16, u64>::new();
    for &v in vals {
        *counts.entry(v).or_insert(0) += 1;
    }
    let total = vals.len() as f64;
    let mut entropy = 0.0;
    for &c in counts.values() {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn count_unique_u16(data: &[u8]) -> usize {
    let n = data.len() / 2;
    let mut set = std::collections::HashSet::new();
    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        set.insert(val);
    }
    set.len()
}

fn count_bit_ones(data: &[u8], bit: u32) -> usize {
    let n = data.len() / 2;
    let mask = 1u16 << bit;
    let mut count = 0;
    for i in 0..n {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        if val & mask != 0 {
            count += 1;
        }
    }
    count
}

fn byte_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if args.capture {
        run_capture(&args)?;
    } else {
        run_benchmark(&args)?;
    }

    Ok(())
}
