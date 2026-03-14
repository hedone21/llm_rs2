//! ZramStore: KV cache compression in memory using configurable filter+codec pipeline.
//!
//! Inspired by Linux zram swap and Blosc2 filter architecture.
//! Pipeline: [trunc_prec (optional, lossy)] → byte-shuffle → [bytedelta (optional)] → codec (LZ4/Zstd).
//!
//! Uses a residual buffer pattern (from KiviCache): recent tokens stay
//! uncompressed, batch-compressed when the residual fills up.

use super::preprocess;
use super::store::OffloadStore;
use anyhow::{Context, Result};

/// Compression codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZramCodec {
    /// LZ4 block compression (fastest).
    Lz4,
    /// Zstandard compression at a given level (better ratio).
    Zstd(i32),
}

/// Configuration for the ZramStore compression pipeline.
#[derive(Debug, Clone)]
pub struct ZramConfig {
    /// Apply bytedelta filter after shuffle (Blosc2-inspired).
    pub use_bytedelta: bool,
    /// Number of F16/F32 mantissa bits to zero before compression (0 = lossless).
    pub trunc_bits: u32,
    /// Compression codec to use.
    pub codec: ZramCodec,
}

impl Default for ZramConfig {
    fn default() -> Self {
        Self {
            use_bytedelta: false,
            trunc_bits: 0,
            codec: ZramCodec::Lz4,
        }
    }
}

/// KV cache compression using configurable filter+codec pipeline.
pub struct ZramStore {
    /// Compressed K data blocks (each block = residual_cap tokens).
    compressed_k: Vec<CompressedBlock>,
    /// Compressed V data blocks.
    compressed_v: Vec<CompressedBlock>,
    /// Uncompressed residual buffer for K (most recent tokens).
    residual_k: Vec<u8>,
    /// Uncompressed residual buffer for V.
    residual_v: Vec<u8>,
    /// Number of tokens in the residual buffer.
    residual_pos: usize,
    /// Residual buffer capacity in tokens.
    residual_cap: usize,
    /// Bytes per token for one of K or V.
    token_bytes: usize,
    /// DType element size for shuffle (2=F16, 4=F32).
    elem_size: usize,
    /// Total tokens in compressed blocks.
    compressed_tokens: usize,
    /// Scratch buffer for shuffle/compress operations.
    shuffle_buf: Vec<u8>,
    /// Pipeline configuration.
    config: ZramConfig,
}

/// A single compressed block representing some tokens worth of data.
struct CompressedBlock {
    data: Vec<u8>,
    /// Original uncompressed size in bytes.
    original_size: usize,
}

impl ZramStore {
    /// Create a new ZramStore with default config (shuffle + LZ4, backward compatible).
    ///
    /// - `token_bytes`: bytes per token for K (or V), i.e. kv_heads × head_dim × dtype_size
    /// - `elem_size`: DType element size (2 for F16, 4 for F32)
    /// - `residual_cap`: number of tokens to buffer before batch-compressing (e.g. 64)
    pub fn new(token_bytes: usize, elem_size: usize, residual_cap: usize) -> Self {
        Self::with_config(token_bytes, elem_size, residual_cap, ZramConfig::default())
    }

    /// Create a new ZramStore with explicit pipeline configuration.
    pub fn with_config(
        token_bytes: usize,
        elem_size: usize,
        residual_cap: usize,
        config: ZramConfig,
    ) -> Self {
        let res_bytes = residual_cap * token_bytes;
        Self {
            compressed_k: Vec::new(),
            compressed_v: Vec::new(),
            residual_k: vec![0u8; res_bytes],
            residual_v: vec![0u8; res_bytes],
            residual_pos: 0,
            residual_cap,
            token_bytes,
            elem_size,
            compressed_tokens: 0,
            shuffle_buf: vec![0u8; res_bytes],
            config,
        }
    }

    /// Compress a data slice using the configured pipeline.
    ///
    /// Pipeline: [trunc_prec] → shuffle → [bytedelta] → codec
    fn compress_block(&mut self, data: &[u8]) -> Result<CompressedBlock> {
        if data.is_empty() {
            return Ok(CompressedBlock {
                data: Vec::new(),
                original_size: 0,
            });
        }

        // Ensure shuffle buffer is large enough
        if self.shuffle_buf.len() < data.len() {
            self.shuffle_buf.resize(data.len(), 0);
        }

        // Step 1 (optional, lossy): truncate mantissa precision
        if self.config.trunc_bits > 0 {
            // We need a mutable copy for trunc_prec since data is &[u8]
            self.shuffle_buf[..data.len()].copy_from_slice(data);
            match self.elem_size {
                2 => preprocess::trunc_prec_f16(
                    &mut self.shuffle_buf[..data.len()],
                    self.config.trunc_bits,
                ),
                4 => preprocess::trunc_prec_f32(
                    &mut self.shuffle_buf[..data.len()],
                    self.config.trunc_bits,
                ),
                _ => {}
            }
            // Shuffle in-place: need a second buffer
            let mut tmp = vec![0u8; data.len()];
            preprocess::shuffle(&self.shuffle_buf[..data.len()], &mut tmp, self.elem_size);
            self.shuffle_buf[..data.len()].copy_from_slice(&tmp);
        } else {
            // Step 2: byte-shuffle
            preprocess::shuffle(data, &mut self.shuffle_buf[..data.len()], self.elem_size);
        }

        // Step 3 (optional): bytedelta
        if self.config.use_bytedelta {
            let n_elements = data.len() / self.elem_size;
            preprocess::bytedelta_encode(
                &mut self.shuffle_buf[..data.len()],
                n_elements,
                self.elem_size,
            );
        }

        // Step 4: codec compress
        let compressed = match self.config.codec {
            ZramCodec::Lz4 => lz4::block::compress(&self.shuffle_buf[..data.len()], None, false)
                .context("LZ4 compress")?,
            ZramCodec::Zstd(level) => zstd::bulk::compress(&self.shuffle_buf[..data.len()], level)
                .context("Zstd compress")?,
        };

        Ok(CompressedBlock {
            data: compressed,
            original_size: data.len(),
        })
    }

    /// Decompress a block into the destination buffer.
    ///
    /// Pipeline (reverse): codec → [bytedelta_decode] → unshuffle
    fn decompress_block(&self, block: &CompressedBlock, dst: &mut [u8]) -> Result<()> {
        if block.original_size == 0 {
            return Ok(());
        }

        // Step 1: codec decompress
        let mut decompressed = match self.config.codec {
            ZramCodec::Lz4 => lz4::block::decompress(&block.data, Some(block.original_size as i32))
                .context("LZ4 decompress")?,
            ZramCodec::Zstd(_) => zstd::bulk::decompress(&block.data, block.original_size)
                .context("Zstd decompress")?,
        };

        // Step 2 (optional): bytedelta decode
        if self.config.use_bytedelta {
            let n_elements = block.original_size / self.elem_size;
            preprocess::bytedelta_decode(&mut decompressed, n_elements, self.elem_size);
        }

        // Step 3: byte-unshuffle
        preprocess::unshuffle(
            &decompressed,
            &mut dst[..block.original_size],
            self.elem_size,
        );

        Ok(())
    }

    /// Flush residual buffer to compressed storage.
    fn flush_residual(&mut self) -> Result<()> {
        if self.residual_pos == 0 {
            return Ok(());
        }

        let k_size = self.residual_pos * self.token_bytes;
        let v_size = k_size;
        let num_tokens = self.residual_pos;

        // Copy data out to avoid borrow conflicts with compress_block
        let k_data = self.residual_k[..k_size].to_vec();
        let v_data = self.residual_v[..v_size].to_vec();

        let k_block = self.compress_block(&k_data)?;
        let v_block = self.compress_block(&v_data)?;

        self.compressed_tokens += num_tokens;
        self.compressed_k.push(k_block);
        self.compressed_v.push(v_block);
        self.residual_pos = 0;

        Ok(())
    }

    /// Total compressed data size in bytes.
    fn total_compressed_size(&self) -> usize {
        self.compressed_k
            .iter()
            .map(|b| b.data.len())
            .sum::<usize>()
            + self
                .compressed_v
                .iter()
                .map(|b| b.data.len())
                .sum::<usize>()
    }

    /// Compression ratio (uncompressed / compressed). Returns 1.0 if no data.
    pub fn compression_ratio(&self) -> f64 {
        let compressed = self.total_compressed_size();
        if compressed == 0 {
            return 1.0;
        }
        let original = self.compressed_tokens * self.token_bytes * 2; // K + V
        original as f64 / compressed as f64
    }

    /// Returns the current pipeline configuration.
    pub fn config(&self) -> &ZramConfig {
        &self.config
    }
}

impl OffloadStore for ZramStore {
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()> {
        let expected = num_tokens * self.token_bytes;
        anyhow::ensure!(k_data.len() == expected, "K data size mismatch");
        anyhow::ensure!(v_data.len() == expected, "V data size mismatch");

        // Clear existing data
        self.clear();

        // Process in residual_cap-sized chunks
        let mut offset = 0;
        let mut remaining = num_tokens;

        while remaining > 0 {
            let batch = remaining.min(self.residual_cap);
            let byte_len = batch * self.token_bytes;

            let k_slice = &k_data[offset..offset + byte_len];
            let v_slice = &v_data[offset..offset + byte_len];

            let k_slice_copy = k_slice.to_vec();
            let k_block = self.compress_block(&k_slice_copy)?;
            let v_data_copy = v_slice.to_vec();
            let v_block = self.compress_block(&v_data_copy)?;

            self.compressed_k.push(k_block);
            self.compressed_v.push(v_block);
            self.compressed_tokens += batch;

            offset += byte_len;
            remaining -= batch;
        }

        Ok(())
    }

    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize> {
        let total = self.compressed_tokens + self.residual_pos;
        if total == 0 {
            return Ok(0);
        }

        // Decompress compressed blocks
        let mut k_offset = 0;
        for block in &self.compressed_k {
            self.decompress_block(block, &mut k_buf[k_offset..])?;
            k_offset += block.original_size;
        }

        let mut v_offset = 0;
        for block in &self.compressed_v {
            self.decompress_block(block, &mut v_buf[v_offset..])?;
            v_offset += block.original_size;
        }

        // Copy residual data
        if self.residual_pos > 0 {
            let res_bytes = self.residual_pos * self.token_bytes;
            k_buf[k_offset..k_offset + res_bytes].copy_from_slice(&self.residual_k[..res_bytes]);
            v_buf[v_offset..v_offset + res_bytes].copy_from_slice(&self.residual_v[..res_bytes]);
        }

        Ok(total)
    }

    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()> {
        anyhow::ensure!(k_token.len() == self.token_bytes, "K token size mismatch");
        anyhow::ensure!(v_token.len() == self.token_bytes, "V token size mismatch");

        // If residual is full, flush to compressed storage
        if self.residual_pos >= self.residual_cap {
            self.flush_residual()?;
        }

        let offset = self.residual_pos * self.token_bytes;
        self.residual_k[offset..offset + self.token_bytes].copy_from_slice(k_token);
        self.residual_v[offset..offset + self.token_bytes].copy_from_slice(v_token);
        self.residual_pos += 1;

        Ok(())
    }

    fn storage_size(&self) -> usize {
        let compressed = self.total_compressed_size();
        let residual = self.residual_pos * self.token_bytes * 2;
        compressed + residual
    }

    fn stored_tokens(&self) -> usize {
        self.compressed_tokens + self.residual_pos
    }

    fn clear(&mut self) {
        self.compressed_k.clear();
        self.compressed_v.clear();
        self.compressed_tokens = 0;
        self.residual_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate realistic F16-like data (similar exponents, varying mantissa).
    fn make_f16_kv_data(num_tokens: usize, token_bytes: usize, seed: u8) -> Vec<u8> {
        let total = num_tokens * token_bytes;
        let mut data = vec![0u8; total];
        for i in 0..total / 2 {
            // F16: exponent ~15 (0x3C00), varying mantissa
            let val = 0x3C00u16.wrapping_add(((seed as u32 * 7 + i as u32) % 1024) as u16);
            data[i * 2] = (val & 0xFF) as u8;
            data[i * 2 + 1] = (val >> 8) as u8;
        }
        data
    }

    /// Generate realistic F32-like data.
    fn make_f32_kv_data(num_tokens: usize, token_bytes: usize, seed: u8) -> Vec<u8> {
        let total = num_tokens * token_bytes;
        let mut data = vec![0u8; total];
        for i in 0..total / 4 {
            // F32 value near 1.0 (0x3F800000) with small perturbations
            let base = 0x3F800000u32;
            let val = base.wrapping_add((seed as u32 * 13 + i as u32) % 65536);
            let bytes = val.to_le_bytes();
            data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        data
    }

    // ── Legacy API compatibility ──

    #[test]
    fn test_zram_store_roundtrip_f16() {
        let token_bytes = 8 * 64 * 2; // 8 heads × 64 dim × 2 bytes
        let num_tokens = 128;
        let mut store = ZramStore::new(token_bytes, 2, 64);

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 1);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 2);

        store.store(&k_data, &v_data, num_tokens).unwrap();
        assert_eq!(store.stored_tokens(), num_tokens);

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        let loaded = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(loaded, num_tokens);
        assert_eq!(&k_buf, &k_data, "K data mismatch (lossless!)");
        assert_eq!(&v_buf, &v_data, "V data mismatch (lossless!)");
    }

    #[test]
    fn test_zram_store_roundtrip_f32() {
        let token_bytes = 8 * 64 * 4; // 8 heads × 64 dim × 4 bytes
        let num_tokens = 64;
        let mut store = ZramStore::new(token_bytes, 4, 32);

        let k_data = make_f32_kv_data(num_tokens, token_bytes, 3);
        let v_data = make_f32_kv_data(num_tokens, token_bytes, 4);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(&k_buf, &k_data, "F32 K roundtrip failed");
        assert_eq!(&v_buf, &v_data, "F32 V roundtrip failed");
    }

    #[test]
    fn test_zram_store_append_decode() {
        let token_bytes = 8 * 64 * 2;
        let mut store = ZramStore::new(token_bytes, 2, 64);

        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..100u8 {
            let k_tok = make_f16_kv_data(1, token_bytes, i);
            let v_tok = make_f16_kv_data(1, token_bytes, i + 128);
            store.append_token(&k_tok, &v_tok).unwrap();
            all_k.extend_from_slice(&k_tok);
            all_v.extend_from_slice(&v_tok);
        }

        assert_eq!(store.stored_tokens(), 100);

        let mut k_buf = vec![0u8; all_k.len()];
        let mut v_buf = vec![0u8; all_v.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(&k_buf, &all_k, "append decode K mismatch");
        assert_eq!(&v_buf, &all_v, "append decode V mismatch");
    }

    #[test]
    fn test_zram_store_compression_ratio_f16() {
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 256;
        let mut store = ZramStore::new(token_bytes, 2, 64);

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 10);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 20);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let ratio = store.compression_ratio();
        println!("F16 compression ratio: {ratio:.2}x");
        assert!(
            ratio >= 1.5,
            "F16 compression ratio {ratio:.2}x < 1.5x — byte-shuffle ineffective"
        );
    }

    #[test]
    fn test_zram_store_compression_ratio_f32() {
        let token_bytes = 8 * 64 * 4;
        let num_tokens = 256;
        let mut store = ZramStore::new(token_bytes, 4, 64);

        let k_data = make_f32_kv_data(num_tokens, token_bytes, 30);
        let v_data = make_f32_kv_data(num_tokens, token_bytes, 40);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let ratio = store.compression_ratio();
        println!("F32 compression ratio: {ratio:.2}x");
        assert!(
            ratio >= 1.3,
            "F32 compression ratio {ratio:.2}x < 1.3x — byte-shuffle ineffective"
        );
    }

    #[test]
    fn test_zram_store_clear() {
        let token_bytes = 64;
        let mut store = ZramStore::new(token_bytes, 2, 16);

        let k = vec![0xABu8; 32 * token_bytes];
        let v = vec![0xCDu8; 32 * token_bytes];
        store.store(&k, &v, 32).unwrap();

        store.clear();
        assert_eq!(store.stored_tokens(), 0);
        assert_eq!(store.storage_size(), 0);
    }

    #[test]
    fn test_zram_store_empty_guard() {
        let store = ZramStore::new(64, 2, 16);
        let mut k = vec![0u8; 64];
        let mut v = vec![0u8; 64];
        let n = store.load_into(&mut k, &mut v).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_zram_store_residual_flush_boundary() {
        let token_bytes = 32;
        let residual_cap = 8;
        let mut store = ZramStore::new(token_bytes, 2, residual_cap);

        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..(residual_cap + 1) as u8 {
            let k = vec![i; token_bytes];
            let v = vec![i + 100; token_bytes];
            store.append_token(&k, &v).unwrap();
            all_k.extend_from_slice(&k);
            all_v.extend_from_slice(&v);
        }

        assert_eq!(store.compressed_tokens, residual_cap);
        assert_eq!(store.residual_pos, 1);
        assert_eq!(store.stored_tokens(), residual_cap + 1);

        let mut k_buf = vec![0u8; all_k.len()];
        let mut v_buf = vec![0u8; all_v.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(&k_buf, &all_k);
        assert_eq!(&v_buf, &all_v);
    }

    // ── Bytedelta pipeline tests ──

    #[test]
    fn test_zram_bytedelta_lz4_roundtrip_f16() {
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 128;
        let config = ZramConfig {
            use_bytedelta: true,
            trunc_bits: 0,
            codec: ZramCodec::Lz4,
        };
        let mut store = ZramStore::with_config(token_bytes, 2, 64, config);

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 1);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 2);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        let loaded = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(loaded, num_tokens);
        assert_eq!(&k_buf, &k_data, "bytedelta+LZ4 K roundtrip failed");
        assert_eq!(&v_buf, &v_data, "bytedelta+LZ4 V roundtrip failed");
    }

    #[test]
    fn test_zram_bytedelta_zstd_roundtrip_f16() {
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 128;
        let config = ZramConfig {
            use_bytedelta: true,
            trunc_bits: 0,
            codec: ZramCodec::Zstd(1),
        };
        let mut store = ZramStore::with_config(token_bytes, 2, 64, config);

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 5);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 6);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(&k_buf, &k_data, "bytedelta+Zstd K roundtrip failed");
        assert_eq!(&v_buf, &v_data, "bytedelta+Zstd V roundtrip failed");
    }

    #[test]
    fn test_zram_trunc_bytedelta_zstd_roundtrip_f16() {
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 128;
        let config = ZramConfig {
            use_bytedelta: true,
            trunc_bits: 5,
            codec: ZramCodec::Zstd(1),
        };
        let mut store = ZramStore::with_config(token_bytes, 2, 64, config);

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 7);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 8);

        store.store(&k_data, &v_data, num_tokens).unwrap();

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        // Lossy: data should NOT match original exactly
        // But should match the truncated version
        let mut k_expected = k_data.clone();
        let mut v_expected = v_data.clone();
        preprocess::trunc_prec_f16(&mut k_expected, 5);
        preprocess::trunc_prec_f16(&mut v_expected, 5);

        assert_eq!(
            &k_buf, &k_expected,
            "trunc+bytedelta+Zstd K should match truncated"
        );
        assert_eq!(
            &v_buf, &v_expected,
            "trunc+bytedelta+Zstd V should match truncated"
        );
    }

    #[test]
    fn test_zram_bytedelta_append_decode() {
        let token_bytes = 8 * 64 * 2;
        let config = ZramConfig {
            use_bytedelta: true,
            trunc_bits: 0,
            codec: ZramCodec::Lz4,
        };
        let mut store = ZramStore::with_config(token_bytes, 2, 64, config);

        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..100u8 {
            let k_tok = make_f16_kv_data(1, token_bytes, i);
            let v_tok = make_f16_kv_data(1, token_bytes, i + 128);
            store.append_token(&k_tok, &v_tok).unwrap();
            all_k.extend_from_slice(&k_tok);
            all_v.extend_from_slice(&v_tok);
        }

        let mut k_buf = vec![0u8; all_k.len()];
        let mut v_buf = vec![0u8; all_v.len()];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(&k_buf, &all_k, "bytedelta append K mismatch");
        assert_eq!(&v_buf, &all_v, "bytedelta append V mismatch");
    }

    // ── Compression ratio comparison benchmark ──

    fn pipeline_configs() -> Vec<(&'static str, ZramConfig)> {
        vec![
            (
                "shuffle+LZ4",
                ZramConfig {
                    use_bytedelta: false,
                    trunc_bits: 0,
                    codec: ZramCodec::Lz4,
                },
            ),
            (
                "shuffle+bytedelta+LZ4",
                ZramConfig {
                    use_bytedelta: true,
                    trunc_bits: 0,
                    codec: ZramCodec::Lz4,
                },
            ),
            (
                "shuffle+bytedelta+Zstd(1)",
                ZramConfig {
                    use_bytedelta: true,
                    trunc_bits: 0,
                    codec: ZramCodec::Zstd(1),
                },
            ),
            (
                "trunc(3)+shuffle+bytedelta+LZ4",
                ZramConfig {
                    use_bytedelta: true,
                    trunc_bits: 3,
                    codec: ZramCodec::Lz4,
                },
            ),
            (
                "trunc(5)+shuffle+bytedelta+Zstd(1)",
                ZramConfig {
                    use_bytedelta: true,
                    trunc_bits: 5,
                    codec: ZramCodec::Zstd(1),
                },
            ),
        ]
    }

    #[test]
    fn test_compression_ratio_comparison_synthetic() {
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 256;

        let k_data = make_f16_kv_data(num_tokens, token_bytes, 42);
        let v_data = make_f16_kv_data(num_tokens, token_bytes, 43);
        let original_size = num_tokens * token_bytes * 2;

        println!("\n=== Compression Ratio Comparison (synthetic F16) ===");
        println!("{:<40} {:>10} {:>12}", "Pipeline", "Ratio", "Size");
        println!("{}", "-".repeat(64));

        for (name, config) in &pipeline_configs() {
            let mut store = ZramStore::with_config(token_bytes, 2, 64, config.clone());
            store.store(&k_data, &v_data, num_tokens).unwrap();

            let ratio = store.compression_ratio();
            let compressed_size = store.total_compressed_size();
            println!("{:<40} {:>9.2}x {:>10} B", name, ratio, compressed_size);

            // Verify lossless roundtrip
            let mut k_buf = vec![0u8; k_data.len()];
            let mut v_buf = vec![0u8; v_data.len()];
            store.load_into(&mut k_buf, &mut v_buf).unwrap();
            if config.trunc_bits == 0 {
                assert_eq!(&k_buf, &k_data, "{name}: lossless K mismatch");
                assert_eq!(&v_buf, &v_data, "{name}: lossless V mismatch");
            }
        }
        println!("Original size: {original_size} B");
    }

    /// Generate high-entropy F16 data mimicking real KV cache activations.
    /// Uses pseudo-random XORShift to produce near-random mantissa bits
    /// with concentrated exponents (like real model outputs).
    fn make_high_entropy_f16_data(num_tokens: usize, token_bytes: usize, seed: u64) -> Vec<u8> {
        let total = num_tokens * token_bytes;
        let mut data = vec![0u8; total];
        let mut rng = seed;
        for i in 0..total / 2 {
            // XORShift64 PRNG
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Exponent in small range (12-17 → 0x3000-0x4400), mantissa random
            let exp = ((rng >> 32) % 6 + 12) as u16; // exponent 12-17
            let mantissa = (rng & 0x03FF) as u16; // random 10-bit mantissa
            let val = (exp << 10) | mantissa;
            data[i * 2] = (val & 0xFF) as u8;
            data[i * 2 + 1] = (val >> 8) as u8;
        }
        data
    }

    #[test]
    fn test_compression_ratio_comparison_high_entropy() {
        let token_bytes = 8 * 64 * 2; // 8 heads × 64 dim × F16
        let num_tokens = 256;

        let k_data = make_high_entropy_f16_data(num_tokens, token_bytes, 12345);
        let v_data = make_high_entropy_f16_data(num_tokens, token_bytes, 67890);
        let original_size = num_tokens * token_bytes * 2;

        println!(
            "\n=== Compression Ratio Comparison (HIGH-ENTROPY F16, pseudo-random mantissa) ==="
        );
        println!(
            "{:<40} {:>10} {:>14} {:>14}",
            "Pipeline", "Ratio", "Compress ms", "Decompr ms"
        );
        println!("{}", "-".repeat(82));

        for (name, config) in &pipeline_configs() {
            let mut store = ZramStore::with_config(token_bytes, 2, 64, config.clone());

            // Measure compress time
            let t0 = std::time::Instant::now();
            store.store(&k_data, &v_data, num_tokens).unwrap();
            let compress_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ratio = store.compression_ratio();

            // Measure decompress time
            let mut k_buf = vec![0u8; k_data.len()];
            let mut v_buf = vec![0u8; v_data.len()];
            let t1 = std::time::Instant::now();
            store.load_into(&mut k_buf, &mut v_buf).unwrap();
            let decompress_ms = t1.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{:<40} {:>9.2}x {:>12.3} ms {:>12.3} ms",
                name, ratio, compress_ms, decompress_ms
            );

            // Verify roundtrip
            if config.trunc_bits == 0 {
                assert_eq!(&k_buf, &k_data, "{name}: lossless K mismatch");
                assert_eq!(&v_buf, &v_data, "{name}: lossless V mismatch");
            } else {
                // Lossy: verify matches truncated expectation
                let mut k_expected = k_data.clone();
                let mut v_expected = v_data.clone();
                preprocess::trunc_prec_f16(&mut k_expected, config.trunc_bits);
                preprocess::trunc_prec_f16(&mut v_expected, config.trunc_bits);
                assert_eq!(&k_buf, &k_expected, "{name}: lossy K mismatch");
                assert_eq!(&v_buf, &v_expected, "{name}: lossy V mismatch");
            }
        }
        println!(
            "Original size: {original_size} B ({:.1} KB)",
            original_size as f64 / 1024.0
        );
    }

    #[test]
    fn test_compression_ratio_comparison_high_entropy_large() {
        // Simulate a realistic layer: 512 tokens × 8 heads × 64 dim × F16
        let token_bytes = 8 * 64 * 2;
        let num_tokens = 512;

        let k_data = make_high_entropy_f16_data(num_tokens, token_bytes, 11111);
        let v_data = make_high_entropy_f16_data(num_tokens, token_bytes, 22222);
        let original_size = num_tokens * token_bytes * 2;

        println!("\n=== Compression Ratio (HIGH-ENTROPY, 512 tokens, ~1MB) ===");
        println!(
            "{:<40} {:>10} {:>12} {:>14} {:>14}",
            "Pipeline", "Ratio", "Size", "Compress", "Decompress"
        );
        println!("{}", "-".repeat(96));

        for (name, config) in &pipeline_configs() {
            let mut store = ZramStore::with_config(token_bytes, 2, 128, config.clone());

            let t0 = std::time::Instant::now();
            store.store(&k_data, &v_data, num_tokens).unwrap();
            let compress_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ratio = store.compression_ratio();
            let compressed_size = store.total_compressed_size();

            let mut k_buf = vec![0u8; k_data.len()];
            let mut v_buf = vec![0u8; v_data.len()];
            let t1 = std::time::Instant::now();
            store.load_into(&mut k_buf, &mut v_buf).unwrap();
            let decompress_ms = t1.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{:<40} {:>9.2}x {:>10} B {:>12.3} ms {:>12.3} ms",
                name, ratio, compressed_size, compress_ms, decompress_ms
            );
        }
        println!(
            "Original size: {original_size} B ({:.1} KB)",
            original_size as f64 / 1024.0
        );
    }
}
