//! Model definition module
//!
//! Contains structure definitions and configurations for large language models

use crate::error::{CuttleError, Result};
use crate::tensor::{Tensor, Tensor1D, Tensor2D, Tensor3D};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Model configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden layer dimension
    pub hidden_size: usize,
    /// Number of Transformer layers
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: Option<usize>,
    /// Feedforward network intermediate layer dimension
    pub intermediate_size: usize,
    /// Maximum position embedding length
    pub max_position_embeddings: usize,
    /// RMS normalization epsilon value
    pub rms_norm_eps: f32,
    /// RoPE theta parameter
    #[serde(default = "default_rope_theta")]
    pub rope_theta: Option<f32>,
    /// Whether to use sliding window attention
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// Sliding window size
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Model type
    #[serde(default)]
    pub model_type: Option<String>,
    /// Architecture names
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

fn default_num_key_value_heads() -> Option<usize> {
    None
}

fn default_rope_theta() -> Option<f32> {
    Some(10000.0)
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
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

/// Attention layer structure
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Query weight matrix
    pub q_proj: Tensor2D,
    /// Key weight matrix
    pub k_proj: Tensor2D,
    /// Value weight matrix
    pub v_proj: Tensor2D,
    /// Output projection weight matrix
    pub o_proj: Tensor2D,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each head
    pub head_dim: usize,
}

impl AttentionLayer {
    /// Create new attention layer
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(CuttleError::ModelConfigError(format!(
                "Hidden size {} must be divisible by num_heads {}",
                hidden_size, num_heads
            )));
        }

        let head_dim = hidden_size / num_heads;

        Ok(Self {
            q_proj: Tensor2D::randn([hidden_size, hidden_size])?,
            k_proj: Tensor2D::randn([hidden_size, hidden_size])?,
            v_proj: Tensor2D::randn([hidden_size, hidden_size])?,
            o_proj: Tensor2D::randn([hidden_size, hidden_size])?,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor3D) -> Result<Tensor3D> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        // Calculate Q, K, V
        let q = hidden_states.matmul(&self.q_proj)?;
        let k = hidden_states.matmul(&self.k_proj)?;
        let v = hidden_states.matmul(&self.v_proj)?;

        // Simplified attention mechanism implementation (without reshape)
        // In static dimension design, we directly use 3D tensors for computation
        let attention_output = v; // Simplified implementation

        // Output projection
        attention_output.matmul(&self.o_proj)
    }
}

/// Feed-forward network layer
#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    /// Up projection weights
    pub up_proj: Tensor2D,
    /// Gate projection weights
    pub gate_proj: Tensor2D,
    /// Down projection weights
    pub down_proj: Tensor2D,
}

impl FeedForwardLayer {
    /// Create new feed-forward network layer
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            up_proj: Tensor2D::randn([hidden_size, intermediate_size])?,
            gate_proj: Tensor2D::randn([hidden_size, intermediate_size])?,
            down_proj: Tensor2D::randn([intermediate_size, hidden_size])?,
        })
    }

    /// Forward pass (SwiGLU activation)
    pub fn forward(&self, hidden_states: &Tensor3D) -> Result<Tensor3D> {
        let up_output = hidden_states.matmul(&self.up_proj)?;
        let gate_output = hidden_states.matmul(&self.gate_proj)?.gelu();

        let intermediate = up_output.mul(&gate_output)?;
        intermediate.matmul(&self.down_proj)
    }
}

/// Transformer layer
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    /// Self-attention layer
    pub self_attention: AttentionLayer,
    /// Feed-forward network layer
    pub feed_forward: FeedForwardLayer,
    /// RMS normalization before attention layer
    pub input_layernorm: Tensor1D,
    /// RMS normalization before feed-forward network layer
    pub post_attention_layernorm: Tensor1D,
}

impl TransformerLayer {
    /// Create new Transformer layer
    pub fn new(config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            self_attention: AttentionLayer::new(config.hidden_size, config.num_attention_heads)?,
            feed_forward: FeedForwardLayer::new(config.hidden_size, config.intermediate_size)?,
            input_layernorm: Tensor1D::randn([config.hidden_size])?,
            post_attention_layernorm: Tensor1D::randn([config.hidden_size])?,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor3D, rms_norm_eps: f32) -> Result<Tensor3D> {
        // Attention layer
        let normed_input = hidden_states.rms_norm(rms_norm_eps)?;
        let attention_output = self.self_attention.forward(&normed_input)?;
        let hidden_states = hidden_states.add(&attention_output)?;

        // Feed-forward network layer
        let normed_hidden = hidden_states.rms_norm(rms_norm_eps)?;
        let ff_output = self.feed_forward.forward(&normed_hidden)?;
        hidden_states.add(&ff_output)
    }
}

/// Complete language model
#[derive(Debug, Clone)]
pub struct Model {
    /// Model configuration
    pub config: ModelConfig,
    /// Token embedding layer
    pub embed_tokens: Tensor2D,
    /// List of Transformer layers
    pub layers: Vec<TransformerLayer>,
    /// Final normalization layer
    pub norm: Tensor1D,
    /// Language model head (for generating vocabulary probabilities)
    pub lm_head: Tensor2D,
}

impl Model {
    /// Create new model
    pub fn new(config: ModelConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(&config)?);
        }

        Ok(Self {
            embed_tokens: Tensor2D::randn([config.vocab_size, config.hidden_size])?,
            norm: Tensor1D::randn([config.hidden_size])?,
            lm_head: Tensor2D::randn([config.hidden_size, config.vocab_size])?,
            layers,
            config,
        })
    }

    /// Load model from config file
    pub fn from_config_file<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path).map_err(|e| {
            CuttleError::ModelLoadError(format!("Failed to read config file: {}", e))
        })?;

        let config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| CuttleError::ModelLoadError(format!("Failed to parse config: {}", e)))?;

        Self::new(config)
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &[usize]) -> Result<Tensor3D> {
        if input_ids.is_empty() {
            return Err(CuttleError::InferenceError(
                "Input cannot be empty".to_string(),
            ));
        }

        let seq_len = input_ids.len();
        let batch_size = 1; // Simplified to single batch

        // Token embedding
        let mut hidden_states = self.embed_input_ids(input_ids)?;

        // Pass through all Transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, self.config.rms_norm_eps)?;
        }

        // Final normalization
        hidden_states = hidden_states.rms_norm(self.config.rms_norm_eps)?;

        // Language model head
        hidden_states.matmul(&self.lm_head)
    }

    /// Convert input IDs to embeddings
    fn embed_input_ids(&self, input_ids: &[usize]) -> Result<Tensor3D> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size;

        let mut embeddings = Tensor3D::zeros([1, seq_len, hidden_size])?;
        let embed_data = embeddings.data_mut();

        // Simplified embedding lookup (actual implementation needs more efficient indexing)
        for (i, &token_id) in input_ids.iter().enumerate() {
            if token_id >= self.config.vocab_size {
                return Err(CuttleError::InferenceError(format!(
                    "Token ID {} exceeds vocab size {}",
                    token_id, self.config.vocab_size
                )));
            }

            // Copy corresponding embedding vector from embedding matrix
            let embed_start = token_id * hidden_size;
            let embed_end = embed_start + hidden_size;
            let output_start = i * hidden_size;
            let output_end = output_start + hidden_size;

            if embed_end <= self.embed_tokens.data().len() {
                embed_data[output_start..output_end]
                    .copy_from_slice(&self.embed_tokens.data()[embed_start..embed_end]);
            }
        }

        Ok(embeddings)
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Save model configuration to file
    pub fn save_config<P: AsRef<Path>>(&self, config_path: P) -> Result<()> {
        let config_str = serde_json::to_string_pretty(&self.config).map_err(|e| {
            CuttleError::SerializationError(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(config_path, config_str).map_err(|e| CuttleError::IoError(e))?;

        Ok(())
    }
}
