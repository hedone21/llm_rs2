//! DiskStore: lossless KV cache offload to disk files.
//!
//! Uses a single temporary file per layer with sequential layout:
//! [K data: num_tokens × token_k_bytes][V data: num_tokens × token_v_bytes]

use super::store::OffloadStore;
use anyhow::{Context, Result};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Lossless KV cache storage backed by a temporary file.
pub struct DiskStore {
    /// Path to the temporary file.
    path: PathBuf,
    /// Bytes per token for K (kv_heads × head_dim × dtype_size).
    token_k_bytes: usize,
    /// Bytes per token for V (same as K for standard layouts).
    token_v_bytes: usize,
    /// Number of tokens currently stored.
    num_tokens: usize,
    /// File handle (opened on first store/append).
    file: Option<fs::File>,
}

impl DiskStore {
    pub fn new(dir: &std::path::Path, layer_id: usize, token_kv_bytes: usize) -> Result<Self> {
        fs::create_dir_all(dir).context("create offload dir")?;
        let path = dir.join(format!("kv_layer_{layer_id}.bin"));
        Ok(Self {
            path,
            token_k_bytes: token_kv_bytes,
            token_v_bytes: token_kv_bytes,
            num_tokens: 0,
            file: None,
        })
    }

    fn ensure_file(&mut self) -> Result<&mut fs::File> {
        if self.file.is_none() {
            let f = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&self.path)
                .with_context(|| format!("open offload file: {:?}", self.path))?;
            self.file = Some(f);
        }
        Ok(self.file.as_mut().unwrap())
    }

    // File layout: [all K data][all V data]
}

impl OffloadStore for DiskStore {
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()> {
        let expected_k = num_tokens * self.token_k_bytes;
        let expected_v = num_tokens * self.token_v_bytes;
        anyhow::ensure!(
            k_data.len() == expected_k,
            "K data size mismatch: {} vs {}",
            k_data.len(),
            expected_k
        );
        anyhow::ensure!(
            v_data.len() == expected_v,
            "V data size mismatch: {} vs {}",
            v_data.len(),
            expected_v
        );

        let f = self.ensure_file()?;
        f.seek(SeekFrom::Start(0))?;
        f.set_len(0)?;
        // Write K then V
        f.write_all(k_data)?;
        f.write_all(v_data)?;
        f.sync_data()?;
        self.num_tokens = num_tokens;
        Ok(())
    }

    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize> {
        if self.num_tokens == 0 {
            return Ok(0);
        }
        let f = self
            .file
            .as_ref()
            .context("DiskStore: no file open for load")?;

        let total_k = self.num_tokens * self.token_k_bytes;
        let total_v = self.num_tokens * self.token_v_bytes;
        anyhow::ensure!(k_buf.len() >= total_k, "K buffer too small");
        anyhow::ensure!(v_buf.len() >= total_v, "V buffer too small");

        // Use pread-style: clone file handle for concurrent read
        let mut reader = f.try_clone()?;
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut k_buf[..total_k])?;
        reader.read_exact(&mut v_buf[..total_v])?;

        Ok(self.num_tokens)
    }

    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()> {
        anyhow::ensure!(k_token.len() == self.token_k_bytes, "K token size mismatch");
        anyhow::ensure!(v_token.len() == self.token_v_bytes, "V token size mismatch");

        // Capture fields before mutable borrow of file
        let num_tokens = self.num_tokens;
        let token_k_bytes = self.token_k_bytes;
        let token_v_bytes = self.token_v_bytes;
        {
            let f = self.ensure_file()?;

            if num_tokens == 0 {
                // First token: just write K then V
                f.seek(SeekFrom::Start(0))?;
                f.write_all(k_token)?;
                f.write_all(v_token)?;
                f.sync_data()?;
                // (first token written + synced)
            } else {
                // Read existing V data
                let v_start = (num_tokens * token_k_bytes) as u64;
                let v_size = num_tokens * token_v_bytes;
                let mut v_data = vec![0u8; v_size];
                f.seek(SeekFrom::Start(v_start))?;
                f.read_exact(&mut v_data)?;

                // Append new K token at end of K section
                let k_append_pos = (num_tokens * token_k_bytes) as u64;
                f.seek(SeekFrom::Start(k_append_pos))?;
                f.write_all(k_token)?;

                // Rewrite V section: old V data + new V token
                f.write_all(&v_data)?;
                f.write_all(v_token)?;

                // Batch fsync: don't fsync every token (R-D2 mitigation)
                if (num_tokens + 1).is_multiple_of(64) {
                    f.sync_data()?;
                }
            }
        }

        if num_tokens == 0 {
            self.num_tokens = 1;
        } else {
            self.num_tokens = num_tokens + 1;
        }
        Ok(())
    }

    fn storage_size(&self) -> usize {
        self.num_tokens * (self.token_k_bytes + self.token_v_bytes)
    }

    fn stored_tokens(&self) -> usize {
        self.num_tokens
    }

    fn clear(&mut self) {
        self.num_tokens = 0;
        if let Some(ref mut f) = self.file {
            f.set_len(0).ok();
        }
    }
}

impl Drop for DiskStore {
    fn drop(&mut self) {
        // Clean up temporary file
        if self.file.take().is_some() {
            fs::remove_file(&self.path).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("llm_rs2_test_disk_store");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_disk_store_roundtrip() {
        let dir = test_dir();
        let token_bytes = 64; // 8 heads × 4 dims × 2 bytes (F16-like)
        let mut store = DiskStore::new(&dir, 99, token_bytes).unwrap();

        let num_tokens = 32;
        let k_data: Vec<u8> = (0..num_tokens * token_bytes)
            .map(|i| (i % 256) as u8)
            .collect();
        let v_data: Vec<u8> = (0..num_tokens * token_bytes)
            .map(|i| ((i + 128) % 256) as u8)
            .collect();

        store.store(&k_data, &v_data, num_tokens).unwrap();
        assert_eq!(store.stored_tokens(), num_tokens);

        let mut k_buf = vec![0u8; k_data.len()];
        let mut v_buf = vec![0u8; v_data.len()];
        let loaded = store.load_into(&mut k_buf, &mut v_buf).unwrap();

        assert_eq!(loaded, num_tokens);
        assert_eq!(&k_buf, &k_data, "K data mismatch");
        assert_eq!(&v_buf, &v_data, "V data mismatch");
    }

    #[test]
    fn test_disk_store_append_token() {
        let dir = test_dir();
        let token_bytes = 16;
        let mut store = DiskStore::new(&dir, 100, token_bytes).unwrap();

        // Store 2 initial tokens
        let k_init = vec![1u8; 2 * token_bytes];
        let v_init = vec![2u8; 2 * token_bytes];
        store.store(&k_init, &v_init, 2).unwrap();

        // Append 1 token
        let k_new = vec![3u8; token_bytes];
        let v_new = vec![4u8; token_bytes];
        store.append_token(&k_new, &v_new).unwrap();
        assert_eq!(store.stored_tokens(), 3);

        // Load and verify
        let mut k_buf = vec![0u8; 3 * token_bytes];
        let mut v_buf = vec![0u8; 3 * token_bytes];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        // K: [1,1,..., 1,1,..., 3,3,...]
        assert_eq!(&k_buf[..token_bytes], &[1u8; 16][..]);
        assert_eq!(&k_buf[token_bytes..2 * token_bytes], &[1u8; 16][..]);
        assert_eq!(&k_buf[2 * token_bytes..], &[3u8; 16][..]);

        // V: [2,2,..., 2,2,..., 4,4,...]
        assert_eq!(&v_buf[..token_bytes], &[2u8; 16][..]);
        assert_eq!(&v_buf[token_bytes..2 * token_bytes], &[2u8; 16][..]);
        assert_eq!(&v_buf[2 * token_bytes..], &[4u8; 16][..]);
    }

    #[test]
    fn test_disk_store_append_from_empty() {
        let dir = test_dir();
        let token_bytes = 8;
        let mut store = DiskStore::new(&dir, 101, token_bytes).unwrap();

        for i in 0..5u8 {
            let k = vec![i; token_bytes];
            let v = vec![i + 100; token_bytes];
            store.append_token(&k, &v).unwrap();
        }
        assert_eq!(store.stored_tokens(), 5);

        let mut k_buf = vec![0u8; 5 * token_bytes];
        let mut v_buf = vec![0u8; 5 * token_bytes];
        store.load_into(&mut k_buf, &mut v_buf).unwrap();

        for i in 0..5u8 {
            let start = i as usize * token_bytes;
            assert!(k_buf[start..start + token_bytes].iter().all(|&b| b == i));
            assert!(
                v_buf[start..start + token_bytes]
                    .iter()
                    .all(|&b| b == i + 100)
            );
        }
    }

    #[test]
    fn test_disk_store_clear() {
        let dir = test_dir();
        let mut store = DiskStore::new(&dir, 102, 16).unwrap();

        store.store(&[0u8; 32], &[0u8; 32], 2).unwrap();
        assert_eq!(store.stored_tokens(), 2);

        store.clear();
        assert_eq!(store.stored_tokens(), 0);
        assert_eq!(store.storage_size(), 0);
    }

    #[test]
    fn test_disk_store_cleanup_on_drop() {
        let dir = test_dir();
        let path;
        {
            let mut store = DiskStore::new(&dir, 103, 8).unwrap();
            store.store(&[0u8; 8], &[0u8; 8], 1).unwrap();
            path = store.path.clone();
            assert!(path.exists());
        }
        // File should be removed after drop
        assert!(!path.exists(), "temp file should be cleaned up on drop");
    }

    #[test]
    fn test_disk_store_empty_load() {
        let dir = test_dir();
        let store = DiskStore::new(&dir, 104, 8).unwrap();
        let mut k = vec![0u8; 8];
        let mut v = vec![0u8; 8];
        let n = store.load_into(&mut k, &mut v).unwrap();
        assert_eq!(n, 0);
    }
}
