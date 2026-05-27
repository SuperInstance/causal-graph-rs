use thiserror::Error;

#[derive(Error, Debug)]
pub enum CausalError {
    #[error("cycle detected: adding {from} -> {to} would create a cycle")]
    CycleDetected { from: String, to: String },
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("edge not found: {from} -> {to}")]
    EdgeNotFound { from: String, to: String },
    #[error("invalid operation: {0}")]
    Invalid(String),
    #[error("data error: {0}")]
    Data(String),
}
