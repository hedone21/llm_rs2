pub mod manager;
pub mod signal;
pub mod state;
pub mod strategy;
pub mod transport;

#[cfg(feature = "resilience")]
pub mod dbus_transport;

pub use manager::{InferenceContext, ResilienceManager, execute_action};
pub use signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
pub use state::OperatingMode;
pub use strategy::{ResilienceAction, ResilienceStrategy};
#[cfg(unix)]
pub use transport::UnixSocketTransport;
pub use transport::{MockSender, MockTransport, SignalListener, Transport, TransportError};

#[cfg(feature = "resilience")]
pub use dbus_transport::DbusTransport;
