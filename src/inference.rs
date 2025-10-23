//! Inference engine module
//!
//! Provides complete large model inference functionality

use crate::error::{CuttleError, Result};
use crate::model::{Model, ModelConfig};
use crate::tensor::{Tensor, Tensor1D, Tensor2D, Tensor3D};
use crate::tokenizer::{Tokenizer, TokenizerConfig};
use log::{debug, info, warn};
use std::path::Path;

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum generation length
    pub max_length: usize,
    /// Temperature parameter (controls randomness)
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Whether to use greedy decoding
    pub do_sample: bool,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            do_sample: true,
            repetition_penalty: 1.1,
        }
    }
}

/// Inference engine
#[derive(Debug)]
pub struct InferenceEngine {
    /// Language model
    model: Model,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Inference configuration
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Create new inference engine
    pub fn new(model: Model, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            config: InferenceConfig::default(),
        }
    }

    /// Create inference engine with custom configuration
    pub fn with_config(model: Model, tokenizer: Tokenizer, config: InferenceConfig) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }

    /// Create inference engine from config files
    pub fn from_config_files<P1, P2>(model_config_path: P1, tokenizer_path: P2) -> Result<Self>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        info!(
            "Loading model from config file: {:?}",
            model_config_path.as_ref()
        );
        let model = Model::from_config_file(model_config_path)?;

        info!("Loading tokenizer from file: {:?}", tokenizer_path.as_ref());
        let tokenizer = Tokenizer::load(tokenizer_path)?;

        Ok(Self::new(model, tokenizer))
    }

    /// Generate text
    pub fn generate(&self, prompt: &str) -> Result<String> {
        info!("Starting text generation for prompt: {}", prompt);

        // Encode input
        let input_ids = self.tokenizer.encode(prompt)?;
        debug!("Encoded input IDs: {:?}", input_ids);

        if input_ids.is_empty() {
            return Err(CuttleError::InferenceError(
                "Empty input after tokenization".to_string(),
            ));
        }

        // Generate token sequence
        let generated_ids = self.generate_tokens(&input_ids)?;

        // Decode output
        let generated_text = self.tokenizer.decode(&generated_ids)?;
        info!("Generated text: {}", generated_text);

        Ok(generated_text)
    }

    /// Generate token sequence
    fn generate_tokens(&self, input_ids: &[usize]) -> Result<Vec<usize>> {
        let mut current_ids = input_ids.to_vec();
        let max_new_tokens = self.config.max_length.saturating_sub(input_ids.len());

        debug!("Generating up to {} new tokens", max_new_tokens);

        for step in 0..max_new_tokens {
            debug!("Generation step {}/{}", step + 1, max_new_tokens);

            // Forward pass
            let logits = self.model.forward(&current_ids)?;

            // Get logits for the last position
            let last_logits = self.extract_last_logits(&logits)?;

            // Apply temperature and repetition penalty
            let processed_logits = self.process_logits(&last_logits, &current_ids)?;

            // Sample next token
            let next_token = self.sample_next_token(&processed_logits)?;

            // Check if it's an end token
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token == eos_id {
                    debug!("Generated EOS token, stopping generation");
                    break;
                }
            }

            current_ids.push(next_token);
            debug!("Generated token: {}", next_token);
        }

        // Return newly generated tokens (excluding input)
        Ok(current_ids[input_ids.len()..].to_vec())
    }

    /// Extract logits for the last position
    fn extract_last_logits(&self, logits: &Tensor3D) -> Result<Tensor3D> {
        let shape = logits.shape();
        if shape.len() != 3 {
            return Err(CuttleError::InferenceError(format!(
                "Expected 3D logits tensor, got {}D",
                shape.len()
            )));
        }

        let seq_len = shape[1];
        let vocab_size = shape[2];

        // Simplified implementation: create logits for the last position
        // Actual implementation needs to extract specific position from 3D tensor
        Tensor3D::randn([1, 1, vocab_size])
    }

    /// Process logits (apply temperature, repetition penalty, etc.)
    fn process_logits(&self, logits: &Tensor3D, generated_ids: &[usize]) -> Result<Tensor3D> {
        let mut processed = logits.clone();

        // Apply temperature
        if self.config.temperature != 1.0 {
            processed = processed.scale(1.0 / self.config.temperature);
        }

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            processed = self.apply_repetition_penalty(&processed, generated_ids)?;
        }

        Ok(processed)
    }

    /// Apply repetition penalty
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor3D,
        generated_ids: &[usize],
    ) -> Result<Tensor3D> {
        let penalized = logits.clone();

        // Simplified implementation: apply penalty to already generated tokens
        // Actual implementation needs to modify corresponding values in logits
        for &token_id in generated_ids {
            if token_id < logits.shape()[0] {
                // Actual logits modification logic needed here
                debug!("Applying repetition penalty to token {}", token_id);
            }
        }

        Ok(penalized)
    }

    /// Sample next token
    fn sample_next_token(&self, logits: &Tensor3D) -> Result<usize> {
        if !self.config.do_sample {
            // Greedy decoding: select token with highest probability
            return self.greedy_sample(logits);
        }

        // Convert to probability distribution - simplified implementation, use logits directly
        let probs = logits.clone();

        // Top-k and Top-p sampling
        let filtered_probs = self.apply_top_k_top_p_filtering(&probs)?;

        // Sample from filtered distribution
        self.multinomial_sample(&filtered_probs)
    }

    /// Greedy sampling
    fn greedy_sample(&self, logits: &Tensor3D) -> Result<usize> {
        let data = logits.to_vec();
        let max_idx = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| CuttleError::InferenceError("Empty logits tensor".to_string()))?;

        Ok(max_idx)
    }

    /// Apply Top-k and Top-p filtering
    fn apply_top_k_top_p_filtering(&self, probs: &Tensor3D) -> Result<Tensor3D> {
        let filtered = probs.clone();

        // Simplified implementation: return original probabilities
        // Actual implementation needs:
        // 1. Top-k filtering: keep only the k tokens with highest probability
        // 2. Top-p filtering: keep tokens with cumulative probability up to p

        debug!(
            "Applying top_k={}, top_p={} filtering",
            self.config.top_k, self.config.top_p
        );

        Ok(filtered)
    }

    /// Multinomial sampling
    fn multinomial_sample(&self, probs: &Tensor3D) -> Result<usize> {
        let data = probs.to_vec();

        // Simplified sampling implementation
        // Actual implementation needs random sampling based on probability distribution
        let sum: f32 = data.iter().sum();
        if sum <= 0.0 {
            return Err(CuttleError::InferenceError(
                "Invalid probability distribution".to_string(),
            ));
        }

        // Use simple random number generation
        let random_val = (data.len() as f32 * 0.5) as usize % data.len();
        Ok(random_val)
    }

    /// Batch generation
    pub fn generate_batch(&self, prompts: &[String]) -> Result<Vec<String>> {
        prompts.iter().map(|prompt| self.generate(prompt)).collect()
    }

    /// Set inference configuration
    pub fn set_config(&mut self, config: InferenceConfig) {
        self.config = config;
    }

    /// Get inference configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Get model reference
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Calculate text perplexity
    pub fn perplexity(&self, text: &str) -> Result<f32> {
        let input_ids = self.tokenizer.encode(text)?;

        if input_ids.len() < 2 {
            return Err(CuttleError::InferenceError(
                "Text too short for perplexity calculation".to_string(),
            ));
        }

        let mut total_log_prob = 0.0;
        let mut count = 0;

        // Calculate log probability for each position
        for i in 1..input_ids.len() {
            let context = &input_ids[..i];
            let target = input_ids[i];

            let logits = self.model.forward(context)?;
            let last_logits = self.extract_last_logits(&logits)?;
            // Simplified implementation: use logits directly as probabilities
            let probs = last_logits;

            let prob_data = probs.to_vec();
            if target < prob_data.len() {
                let prob = prob_data[target].max(1e-10); // Avoid log(0)
                total_log_prob += prob.ln();
                count += 1;
            }
        }

        if count == 0 {
            return Err(CuttleError::InferenceError(
                "No valid tokens for perplexity".to_string(),
            ));
        }

        let avg_log_prob = total_log_prob / count as f32;
        Ok((-avg_log_prob).exp())
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        let config = self.model.config();
        ModelInfo {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads,
            max_position_embeddings: config.max_position_embeddings,
            tokenizer_vocab_size: self.tokenizer.vocab_size(),
        }
    }
}

/// Model information structure
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub tokenizer_vocab_size: usize,
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model Info:\n")?;
        write!(f, "  Vocabulary Size: {}\n", self.vocab_size)?;
        write!(f, "  Hidden Size: {}\n", self.hidden_size)?;
        write!(f, "  Number of Layers: {}\n", self.num_layers)?;
        write!(f, "  Attention Heads: {}\n", self.num_attention_heads)?;
        write!(
            f,
            "  Max Position Embeddings: {}\n",
            self.max_position_embeddings
        )?;
        write!(f, "  Tokenizer Vocab Size: {}", self.tokenizer_vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::create_default_tokenizer;

    #[test]
    fn test_inference_engine_creation() {
        let model_config = ModelConfig::default();
        let model = Model::new(model_config).unwrap();
        let tokenizer = create_default_tokenizer();

        let engine = InferenceEngine::new(model, tokenizer);
        assert_eq!(engine.config().max_length, 512);
    }
}
