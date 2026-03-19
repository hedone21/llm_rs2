// Backward compatibility re-exports
pub use crate::models::config::ModelConfig as LlamaConfig;
pub use crate::models::transformer::{
    TransformerModel as LlamaModel,
    TransformerModelForwardArgs as LlamaModelForwardArgs,
};
