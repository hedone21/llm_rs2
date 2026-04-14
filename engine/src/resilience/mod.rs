pub mod executor;
pub mod gpu_self_meter;
pub mod manager;
pub mod proc_self_meter;
pub mod signal;
pub mod state;
pub mod strategy;
pub mod transport;

#[cfg(feature = "resilience")]
pub mod dbus_transport;

pub use executor::{
    CommandExecutor, EvictMethod, EvictPlan, ExecutionPlan, KVSnapshot, StreamingParams,
};
pub use gpu_self_meter::{GpuSelfMeter, NoOpGpuMeter, OpenClEventGpuMeter};
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
    MessageLoop, MockManagerEnd, MockSender, MockTransport, TcpTransport, Transport, TransportError,
};

#[cfg(feature = "resilience")]
pub use dbus_transport::DbusTransport;
