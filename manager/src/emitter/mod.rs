pub mod dbus;
pub mod unix_socket;

use llm_shared::SystemSignal;

/// Emitter delivers SystemSignals to connected LLM engine clients.
///
/// # Implementations
///
/// - `DbusEmitter` — Emits signals on the Linux System Bus (`org.llm.Manager1`)
/// - `UnixSocketEmitter` — Sends length-prefixed JSON over Unix domain sockets
pub trait Emitter: Send {
    /// Emit a signal to all connected clients.
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()>;

    /// Emit initial state signals to newly connected clients.
    ///
    /// Called when a new LLM client connects, providing it with
    /// the current system state across all signal types.
    fn emit_initial(&mut self, signals: &[SystemSignal]) -> anyhow::Result<()>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}
