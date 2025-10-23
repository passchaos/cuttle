//! Cuttle - A CPU-based Large Language Model Inference Engine
//!
//! This crate provides a high-performance inference engine for large language models
//! optimized for CPU execution.

pub mod error;
pub mod inference;
pub mod model;
pub mod tensor;
pub mod tokenizer;
pub mod utils;
pub mod downloader;

pub use error::{CuttleError, Result};
pub use inference::{InferenceConfig, InferenceEngine};
pub use model::{Model, ModelConfig};
pub use tensor::Tensor;
pub use tokenizer::Tokenizer;

/// Version information of the inference engine
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model configuration
pub struct DefaultConfig;

impl DefaultConfig {
    /// Get the default model configuration
    pub fn model_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 151936,
            hidden_size: 1024,
            num_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            intermediate_size: 2816,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: Some(1000000.0),
            use_sliding_window: Some(false),
            sliding_window: None,
            model_type: Some("qwen3".to_string()),
            architectures: Some(vec!["Qwen3ForCausalLM".to_string()]),
        }
    }
}
