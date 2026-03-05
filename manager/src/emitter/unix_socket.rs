use crate::emitter::Emitter;
use llm_shared::SystemSignal;
use std::io::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Unix socket signal emitter for Android and cross-platform use.
///
/// Listens on a Unix domain socket and sends length-prefixed JSON
/// signals to connected clients. Wire format matches the engine's
/// `UnixSocketTransport`: `[4-byte BE u32 length][UTF-8 JSON payload]`.
pub struct UnixSocketEmitter {
    socket_path: PathBuf,
    client: Option<UnixStream>,
    listener: UnixListener,
}

impl UnixSocketEmitter {
    /// Create emitter listening on the given Unix socket path.
    pub fn new(socket_path: &Path) -> anyhow::Result<Self> {
        // Remove stale socket
        let _ = std::fs::remove_file(socket_path);

        let listener = UnixListener::bind(socket_path)?;
        listener.set_nonblocking(true)?;
        log::info!("[UnixSocketEmitter] Listening on {}", socket_path.display());

        Ok(Self {
            socket_path: socket_path.to_path_buf(),
            client: None,
            listener,
        })
    }

    /// Wait for a client to connect (blocking with timeout).
    pub fn wait_for_client(&mut self, timeout: Duration, shutdown: &Arc<AtomicBool>) -> bool {
        let start = std::time::Instant::now();
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return false;
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok();
                    log::info!("[UnixSocketEmitter] Client connected");
                    self.client = Some(stream);
                    return true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if start.elapsed() >= timeout {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    log::error!("[UnixSocketEmitter] Accept error: {}", e);
                    return false;
                }
            }
        }
    }

    fn write_signal(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        let stream = self
            .client
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No client connected"))?;

        let json = serde_json::to_vec(signal)?;
        let len = (json.len() as u32).to_be_bytes();
        stream.write_all(&len)?;
        stream.write_all(&json)?;
        stream.flush()?;
        Ok(())
    }
}

impl Emitter for UnixSocketEmitter {
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()> {
        if self.client.is_none() {
            // Accept new client if available (non-blocking)
            if let Ok((stream, _)) = self.listener.accept() {
                stream.set_nonblocking(false).ok();
                log::info!("[UnixSocketEmitter] New client connected");
                self.client = Some(stream);
            } else {
                return Ok(()); // No client, skip
            }
        }

        match self.write_signal(signal) {
            Ok(()) => {
                log::debug!("[UnixSocketEmitter] Sent signal");
                Ok(())
            }
            Err(e) => {
                log::warn!("[UnixSocketEmitter] Client disconnected: {}", e);
                self.client = None;
                Ok(()) // Don't propagate — losing client is not fatal
            }
        }
    }

    fn emit_initial(&mut self, signals: &[SystemSignal]) -> anyhow::Result<()> {
        for signal in signals {
            self.emit(signal)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "UnixSocketEmitter"
    }
}

impl Drop for UnixSocketEmitter {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_shared::Level;
    use std::io::Read;

    #[test]
    fn roundtrip_signal_over_socket() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test.sock");

        let mut emitter = UnixSocketEmitter::new(&sock_path).unwrap();

        // Connect a client
        let mut client = UnixStream::connect(&sock_path).unwrap();

        // Accept client
        let shutdown = Arc::new(AtomicBool::new(false));
        assert!(emitter.wait_for_client(Duration::from_secs(1), &shutdown));

        // Emit a signal
        let signal = SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 500_000_000,
            reclaim_target_bytes: 50_000_000,
        };
        emitter.emit(&signal).unwrap();

        // Read from client
        let mut len_buf = [0u8; 4];
        client.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut json_buf = vec![0u8; len];
        client.read_exact(&mut json_buf).unwrap();

        let received: SystemSignal = serde_json::from_slice(&json_buf).unwrap();
        assert_eq!(received.level(), Level::Warning);
    }

    #[test]
    fn emit_without_client_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test2.sock");

        let mut emitter = UnixSocketEmitter::new(&sock_path).unwrap();

        let signal = SystemSignal::ThermalAlert {
            level: Level::Normal,
            temperature_mc: 45000,
            throttling_active: false,
            throttle_ratio: 1.0,
        };

        // Should not error — just skip
        emitter.emit(&signal).unwrap();
    }
}
