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
    }

    Ok(())
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
