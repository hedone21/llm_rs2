pub mod executor;
pub mod manager;
pub mod signal;
pub mod state;
pub mod strategy;
pub mod transport;

#[cfg(feature = "resilience")]
pub mod dbus_transport;

pub use executor::{CommandExecutor, EvictPlan, ExecutionPlan, KVSnapshot};
pub use manager::{InferenceContext, ResilienceManager, execute_action};
pub use signal::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineDirective,
    EngineMessage, EngineState, EngineStatus, ManagerMessage, ResourceLevel,
};
pub use signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
pub use state::OperatingMode;
pub use strategy::{ResilienceAction, ResilienceStrategy};
#[cfg(unix)]
pub use transport::UnixSocketTransport;
pub use transport::{
    MessageLoop, MockManagerEnd, MockSender, MockTransport, Transport, TransportError,
};

#[cfg(feature = "resilience")]
pub use dbus_transport::DbusTransport;
