use std::fmt;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc;

use super::signal::SystemSignal;

// ── TransportError ──────────────────────────────────────────

/// Error type for transport operations.
#[derive(Debug)]
pub enum TransportError {
    ConnectionFailed(String),
    Disconnected,
    ParseError(String),
    Io(std::io::Error),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::ConnectionFailed(msg) => write!(f, "connection failed: {}", msg),
            TransportError::Disconnected => write!(f, "disconnected"),
            TransportError::ParseError(msg) => write!(f, "parse error: {}", msg),
            TransportError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        TransportError::Io(e)
    }
}

// ── Transport trait ─────────────────────────────────────────

/// Platform-agnostic transport for receiving system signals.
pub trait Transport: Send + 'static {
    /// Establish connection to the signal source.
    fn connect(&mut self) -> Result<(), TransportError>;

    /// Receive the next signal (blocking).
    fn recv(&mut self) -> Result<SystemSignal, TransportError>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ── SignalListener ──────────────────────────────────────────

/// Generic listener that bridges a Transport to an mpsc channel.
pub struct SignalListener<T: Transport> {
    transport: T,
    tx: mpsc::Sender<SystemSignal>,
}

impl<T: Transport> SignalListener<T> {
    pub fn new(transport: T, tx: mpsc::Sender<SystemSignal>) -> Self {
        Self { transport, tx }
    }

    /// Spawn the listener loop in a dedicated thread.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            self.run();
        })
    }

    fn run(mut self) {
        if let Err(e) = self.transport.connect() {
            log::warn!(
                "{} transport connect failed: {}. Continuing without resilience.",
                self.transport.name(),
                e
            );
            return;
        }
        log::info!("{} transport connected.", self.transport.name());

        loop {
            match self.transport.recv() {
                Ok(signal) => {
                    log::debug!(
                        "Received signal via {}: {:?}",
                        self.transport.name(),
                        signal
                    );
                    if self.tx.send(signal).is_err() {
                        log::info!(
                            "Receiver dropped. Stopping {} listener.",
                            self.transport.name()
                        );
                        break;
                    }
                }
                Err(TransportError::ParseError(msg)) => {
                    log::warn!("{} parse error: {}. Skipping.", self.transport.name(), msg);
                    continue;
                }
                Err(TransportError::Disconnected) => {
                    log::info!("{} disconnected. Stopping listener.", self.transport.name());
                    break;
                }
                Err(e) => {
                    log::warn!("{} error: {}. Stopping listener.", self.transport.name(), e);
                    break;
                }
            }
        }
    }
}

// ── MockTransport ───────────────────────────────────────────

/// Mock transport for testing. Receives signals from a channel or pre-loaded vec.
pub struct MockTransport {
    rx: mpsc::Receiver<SystemSignal>,
}

/// Sender handle paired with MockTransport.
pub struct MockSender {
    tx: mpsc::Sender<SystemSignal>,
}

impl MockSender {
    pub fn send(&self, signal: SystemSignal) -> Result<(), mpsc::SendError<SystemSignal>> {
        self.tx.send(signal)
    }
}

impl MockTransport {
    /// Create a channel-based mock transport.
    pub fn channel() -> (Self, MockSender) {
        let (tx, rx) = mpsc::channel();
        (Self { rx }, MockSender { tx })
    }

    /// Create a mock transport pre-loaded with signals.
    pub fn from_signals(signals: Vec<SystemSignal>) -> Self {
        let (tx, rx) = mpsc::channel();
        for sig in signals {
            tx.send(sig).unwrap();
        }
        drop(tx); // Will produce Disconnected after all signals consumed
        Self { rx }
    }
}

impl Transport for MockTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        Ok(())
    }

    fn recv(&mut self) -> Result<SystemSignal, TransportError> {
        self.rx.recv().map_err(|_| TransportError::Disconnected)
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

// ── UnixSocketTransport ─────────────────────────────────────

/// Maximum message payload size (64KB sanity check).
const MAX_PAYLOAD_SIZE: u32 = 64 * 1024;

/// Unix domain socket transport.
/// Wire format: `[4 bytes BE u32 length][UTF-8 JSON payload]`
#[cfg(unix)]
pub struct UnixSocketTransport {
    path: PathBuf,
    stream: Option<UnixStream>,
}

#[cfg(unix)]
impl UnixSocketTransport {
    pub fn new(path: PathBuf) -> Self {
        Self { path, stream: None }
    }

    /// Create from an already-connected stream (for testing).
    #[cfg(test)]
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            path: PathBuf::new(),
            stream: Some(stream),
        }
    }

    fn read_exact(stream: &mut UnixStream, buf: &mut [u8]) -> Result<(), TransportError> {
        match stream.read_exact(buf) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(TransportError::Disconnected)
            }
            Err(e) => Err(TransportError::Io(e)),
        }
    }
}

#[cfg(unix)]
impl Transport for UnixSocketTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        match UnixStream::connect(&self.path) {
            Ok(stream) => {
                self.stream = Some(stream);
                Ok(())
            }
            Err(e) => Err(TransportError::ConnectionFailed(format!(
                "{}: {}",
                self.path.display(),
                e
            ))),
        }
    }

    fn recv(&mut self) -> Result<SystemSignal, TransportError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;

        // Read 4-byte length header (big-endian)
        let mut len_buf = [0u8; 4];
        Self::read_exact(stream, &mut len_buf)?;
        let len = u32::from_be_bytes(len_buf);

        if len > MAX_PAYLOAD_SIZE {
            return Err(TransportError::ParseError(format!(
                "payload too large: {} bytes (max {})",
                len, MAX_PAYLOAD_SIZE
            )));
        }

        // Read JSON payload
        let mut payload = vec![0u8; len as usize];
        Self::read_exact(stream, &mut payload)?;

        serde_json::from_slice(&payload)
            .map_err(|e| TransportError::ParseError(format!("invalid JSON: {}", e)))
    }

    fn name(&self) -> &str {
        "UnixSocket"
    }
}

/// Helper: write a length-prefixed JSON message to a stream.
#[cfg(all(unix, test))]
pub fn write_signal(stream: &mut UnixStream, signal: &SystemSignal) -> std::io::Result<()> {
    use std::io::Write as _;
    let json = serde_json::to_vec(signal).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    stream.write_all(&len)?;
    stream.write_all(&json)?;
    stream.flush()
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::{EnergyReason, Level};

    fn sample_signal() -> SystemSignal {
        SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 200_000_000,
            reclaim_target_bytes: 50_000_000,
        }
    }

    // ── MockTransport tests ────────────────────────────────

    #[test]
    fn test_mock_from_signals_delivers_all() {
        let signals = vec![
            sample_signal(),
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.5,
            },
        ];
        let mut transport = MockTransport::from_signals(signals);
        assert!(transport.connect().is_ok());

        let s1 = transport.recv().unwrap();
        assert_eq!(s1.level(), Level::Warning);

        let s2 = transport.recv().unwrap();
        assert_eq!(s2.level(), Level::Critical);

        assert!(matches!(
            transport.recv(),
            Err(TransportError::Disconnected)
        ));
    }

    #[test]
    fn test_mock_channel_send_recv() {
        let (mut transport, sender) = MockTransport::channel();
        assert!(transport.connect().is_ok());

        sender.send(sample_signal()).unwrap();
        let sig = transport.recv().unwrap();
        assert_eq!(sig.level(), Level::Warning);
    }

    #[test]
    fn test_mock_channel_disconnect() {
        let (mut transport, sender) = MockTransport::channel();
        assert!(transport.connect().is_ok());
        drop(sender);
        assert!(matches!(
            transport.recv(),
            Err(TransportError::Disconnected)
        ));
    }

    #[test]
    fn test_mock_connect_always_ok() {
        let mut transport = MockTransport::from_signals(vec![]);
        assert!(transport.connect().is_ok());
    }

    // ── UnixSocketTransport tests ──────────────────────────

    #[cfg(unix)]
    mod unix_tests {
        use super::*;
        use std::io::Write as _;
        use std::os::unix::net::UnixListener;

        fn tmp_socket_path() -> PathBuf {
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            std::env::temp_dir().join(format!("llm_rs_test_{}.sock", id))
        }

        #[test]
        fn test_unix_socket_round_trip() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv().unwrap()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            let sig = sample_signal();
            write_signal(&mut server_stream, &sig).unwrap();
            drop(server_stream);

            let received = handle.join().unwrap();
            assert_eq!(received.level(), Level::Warning);
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_multiple_messages() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                let s1 = transport.recv().unwrap();
                let s2 = transport.recv().unwrap();
                let s3 = transport.recv().unwrap();
                (s1, s2, s3)
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            for level in [Level::Normal, Level::Warning, Level::Critical] {
                let sig = SystemSignal::MemoryPressure {
                    level,
                    available_bytes: 1000,
                    reclaim_target_bytes: 500,
                };
                write_signal(&mut server_stream, &sig).unwrap();
            }
            drop(server_stream);

            let (s1, s2, s3) = handle.join().unwrap();
            assert_eq!(s1.level(), Level::Normal);
            assert_eq!(s2.level(), Level::Warning);
            assert_eq!(s3.level(), Level::Critical);
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_parse_error() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            // Write valid length + invalid JSON
            let bad_json = b"not json!";
            let len = (bad_json.len() as u32).to_be_bytes();
            server_stream.write_all(&len).unwrap();
            server_stream.write_all(bad_json).unwrap();
            server_stream.flush().unwrap();
            drop(server_stream);

            let result = handle.join().unwrap();
            assert!(matches!(result, Err(TransportError::ParseError(_))));
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_oversized_rejected() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv()
            });

            let (mut server_stream, _) = listener.accept().unwrap();
            // Write length > 64KB
            let len = (MAX_PAYLOAD_SIZE + 1).to_be_bytes();
            server_stream.write_all(&len).unwrap();
            server_stream.flush().unwrap();
            drop(server_stream);

            let result = handle.join().unwrap();
            assert!(matches!(result, Err(TransportError::ParseError(_))));
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_connection_closed() {
            let path = tmp_socket_path();
            let listener = UnixListener::bind(&path).unwrap();

            let path2 = path.clone();
            let handle = std::thread::spawn(move || {
                let mut transport = UnixSocketTransport::new(path2);
                transport.connect().unwrap();
                transport.recv()
            });

            let (server_stream, _) = listener.accept().unwrap();
            drop(server_stream); // Close immediately

            let result = handle.join().unwrap();
            assert!(matches!(result, Err(TransportError::Disconnected)));
            std::fs::remove_file(&path).ok();
        }

        #[test]
        fn test_unix_socket_connect_fail() {
            let mut transport =
                UnixSocketTransport::new(PathBuf::from("/tmp/nonexistent_llm_test.sock"));
            assert!(matches!(
                transport.connect(),
                Err(TransportError::ConnectionFailed(_))
            ));
        }
    }

    // ── SignalListener tests ───────────────────────────────

    #[test]
    fn test_listener_forwards_to_channel() {
        let signals = vec![
            sample_signal(),
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.5,
            },
            SystemSignal::EnergyConstraint {
                level: Level::Emergency,
                reason: EnergyReason::BatteryCritical,
                power_budget_mw: 0,
            },
        ];
        let transport = MockTransport::from_signals(signals);
        let (tx, rx) = mpsc::channel();
        let listener = SignalListener::new(transport, tx);
        let handle = listener.spawn();

        let s1 = rx.recv().unwrap();
        assert_eq!(s1.level(), Level::Warning);
        let s2 = rx.recv().unwrap();
        assert_eq!(s2.level(), Level::Critical);
        let s3 = rx.recv().unwrap();
        assert_eq!(s3.level(), Level::Emergency);

        handle.join().unwrap();
    }

    #[test]
    fn test_listener_survives_parse_errors() {
        // Custom transport that returns ParseError then Ok then Disconnected
        struct FlakyTransport {
            calls: u32,
        }
        impl Transport for FlakyTransport {
            fn connect(&mut self) -> Result<(), TransportError> {
                Ok(())
            }
            fn recv(&mut self) -> Result<SystemSignal, TransportError> {
                self.calls += 1;
                match self.calls {
                    1 => Err(TransportError::ParseError("bad data".into())),
                    2 => Ok(SystemSignal::MemoryPressure {
                        level: Level::Warning,
                        available_bytes: 1000,
                        reclaim_target_bytes: 500,
                    }),
                    _ => Err(TransportError::Disconnected),
                }
            }
            fn name(&self) -> &str {
                "Flaky"
            }
        }

        let (tx, rx) = mpsc::channel();
        let listener = SignalListener::new(FlakyTransport { calls: 0 }, tx);
        let handle = listener.spawn();

        // Should receive the one good signal (parse error skipped)
        let sig = rx.recv().unwrap();
        assert_eq!(sig.level(), Level::Warning);

        handle.join().unwrap();
    }

    #[test]
    fn test_listener_stops_on_disconnect() {
        let transport = MockTransport::from_signals(vec![sample_signal()]);
        let (tx, rx) = mpsc::channel();
        let listener = SignalListener::new(transport, tx);
        let handle = listener.spawn();

        let _ = rx.recv().unwrap();
        // Thread should terminate after Disconnected
        handle.join().unwrap();
    }

    #[test]
    fn test_listener_stops_when_receiver_dropped() {
        let (transport, sender) = MockTransport::channel();
        let (tx, rx) = mpsc::channel();
        let listener = SignalListener::new(transport, tx);
        let handle = listener.spawn();

        // Drop receiver — next send by listener should fail
        drop(rx);

        // Send a signal — listener should detect send failure and stop
        sender.send(sample_signal()).unwrap();

        handle.join().unwrap();
    }
}
