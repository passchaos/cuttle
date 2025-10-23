//! Error handling module
//!
//! Defines various error types that may occur in the inference engine

use thiserror::Error;

/// Error types for Cuttle inference engine
#[derive(Error, Debug)]
pub enum CuttleError {
    /// Tensor operation error
    #[error("Tensor operation error: {0}")]
    TensorError(String),

    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoadError(String),

    /// Model configuration error
    #[error("Model configuration error: {0}")]
    ModelConfigError(String),

    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    /// File I/O error
    #[error("File I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Out of memory error
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Result type for Cuttle inference engine
pub type Result<T> = std::result::Result<T, CuttleError>;

/// Convert from ndarray ShapeError to CuttleError
// No longer needed after removing ndarray dependency

/// Convert from serde_json error to CuttleError
impl From<serde_json::Error> for CuttleError {
    fn from(err: serde_json::Error) -> Self {
        CuttleError::SerializationError(format!("JSON error: {}", err))
    }
}

/// Convert from bincode error to CuttleError
impl From<bincode::Error> for CuttleError {
    fn from(err: bincode::Error) -> Self {
        CuttleError::SerializationError(format!("Bincode error: {}", err))
    }
}
