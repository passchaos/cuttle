//! Tokenizer module
//!
//! Provides text tokenization and encoding functionality

use crate::error::{CuttleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Unknown token
    pub unk_token: String,
    /// Beginning of sequence token
    pub bos_token: String,
    /// End of sequence token
    pub eos_token: String,
    /// Padding token
    pub pad_token: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            unk_token: "<unk>".to_string(),
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: "<pad>".to_string(),
        }
    }
}

/// Simple tokenizer implementation
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Configuration
    config: TokenizerConfig,
    /// Vocabulary: token -> ID
    vocab: HashMap<String, usize>,
    /// Reverse vocabulary: ID -> token
    id_to_token: HashMap<usize, String>,
    /// Special token IDs
    special_tokens: HashMap<String, usize>,
}

impl Tokenizer {
    /// Create a new tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        let mut tokenizer = Self {
            config: config.clone(),
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
        };

        // Add special tokens
        tokenizer.add_special_token(&config.pad_token, 0);
        tokenizer.add_special_token(&config.unk_token, 1);
        tokenizer.add_special_token(&config.bos_token, 2);
        tokenizer.add_special_token(&config.eos_token, 3);

        tokenizer
    }

    /// Add special token
    fn add_special_token(&mut self, token: &str, id: usize) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.special_tokens.insert(token.to_string(), id);
    }

    /// Build basic vocabulary
    pub fn build_vocab(&mut self, texts: &[String]) -> Result<()> {
        let mut word_freq = HashMap::new();

        // Count word frequency
        for text in texts {
            let words = self.simple_tokenize(text);
            for word in words {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency and add to vocabulary
        let mut sorted_words: Vec<_> = word_freq.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending

        let mut current_id = self.special_tokens.len();
        for (word, _freq) in sorted_words {
            if current_id >= self.config.vocab_size {
                break;
            }

            if !self.vocab.contains_key(&word) {
                self.vocab.insert(word.clone(), current_id);
                self.id_to_token.insert(current_id, word);
                current_id += 1;
            }
        }

        Ok(())
    }

    /// Simple text tokenization
    fn simple_tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                // Remove punctuation
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Encode text to token ID sequence
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let words = self.simple_tokenize(text);
        let mut token_ids = Vec::new();

        // Add beginning of sequence token
        if let Some(&bos_id) = self.special_tokens.get(&self.config.bos_token) {
            token_ids.push(bos_id);
        }

        // Encode each word
        for word in words {
            let token_id = self.vocab.get(&word).copied().unwrap_or_else(|| {
                self.special_tokens
                    .get(&self.config.unk_token)
                    .copied()
                    .unwrap_or(1) // Default UNK ID
            });
            token_ids.push(token_id);
        }

        // Add end of sequence token
        if let Some(&eos_id) = self.special_tokens.get(&self.config.eos_token) {
            token_ids.push(eos_id);
        }

        Ok(token_ids)
    }

    /// Decode token ID sequence to text
    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut words = Vec::new();

        for &token_id in token_ids {
            if let Some(token) = self.id_to_token.get(&token_id) {
                // Skip special tokens
                if !self.special_tokens.contains_key(token) {
                    words.push(token.clone());
                }
            } else {
                return Err(CuttleError::TokenizerError(format!(
                    "Unknown token ID: {}",
                    token_id
                )));
            }
        }

        Ok(words.join(" "))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special token ID
    pub fn get_special_token_id(&self, token: &str) -> Option<usize> {
        self.special_tokens.get(token).copied()
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<usize> {
        self.get_special_token_id(&self.config.bos_token)
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<usize> {
        self.get_special_token_id(&self.config.eos_token)
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> Option<usize> {
        self.get_special_token_id(&self.config.pad_token)
    }

    /// Get UNK token ID
    pub fn unk_token_id(&self) -> Option<usize> {
        self.get_special_token_id(&self.config.unk_token)
    }

    /// Save tokenizer to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let tokenizer_data = TokenizerData {
            config: self.config.clone(),
            vocab: self.vocab.clone(),
            special_tokens: self.special_tokens.clone(),
        };

        let serialized = serde_json::to_string_pretty(&tokenizer_data).map_err(|e| {
            CuttleError::SerializationError(format!("Failed to serialize tokenizer: {}", e))
        })?;

        std::fs::write(path, serialized).map_err(|e| CuttleError::IoError(e))?;

        Ok(())
    }

    /// Load tokenizer from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| CuttleError::IoError(e))?;

        // Try to load Hugging Face format tokenizer.json
        if let Ok(hf_tokenizer) = serde_json::from_str::<serde_json::Value>(&content) {
            return Self::from_huggingface_json(&hf_tokenizer);
        }

        // Fallback to custom format
        let tokenizer_data: TokenizerData = serde_json::from_str(&content).map_err(|e| {
            CuttleError::SerializationError(format!("Failed to deserialize tokenizer: {}", e))
        })?;

        let mut id_to_token = HashMap::new();
        for (token, id) in &tokenizer_data.vocab {
            id_to_token.insert(*id, token.clone());
        }

        Ok(Self {
            config: tokenizer_data.config,
            vocab: tokenizer_data.vocab,
            id_to_token,
            special_tokens: tokenizer_data.special_tokens,
        })
    }

    /// Load from Hugging Face tokenizer.json format
    fn from_huggingface_json(json: &serde_json::Value) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Extract vocabulary from model.vocab
        if let Some(model) = json.get("model") {
            if let Some(vocab_obj) = model.get("vocab") {
                if let Some(vocab_map) = vocab_obj.as_object() {
                    for (token, id) in vocab_map {
                        if let Some(id_num) = id.as_u64() {
                            let id_usize = id_num as usize;
                            vocab.insert(token.clone(), id_usize);
                            id_to_token.insert(id_usize, token.clone());
                        }
                    }
                }
            }
        }

        // Set default special tokens
        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            unk_token: "<|endoftext|>".to_string(),
            bos_token: "<|endoftext|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            pad_token: "<|endoftext|>".to_string(),
        };

        // Add special tokens to mapping
        if let Some(unk_id) = vocab.get(&config.unk_token) {
            special_tokens.insert(config.unk_token.clone(), *unk_id);
        }

        Ok(Self {
            config,
            vocab,
            id_to_token,
            special_tokens,
        })
    }

    /// Batch encoding
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<usize>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Batch decoding
    pub fn decode_batch(&self, token_ids_batch: &[Vec<usize>]) -> Result<Vec<String>> {
        token_ids_batch
            .iter()
            .map(|token_ids| self.decode(token_ids))
            .collect()
    }
}

/// Tokenizer data structure for serialization
#[derive(Serialize, Deserialize)]
struct TokenizerData {
    config: TokenizerConfig,
    vocab: HashMap<String, usize>,
    special_tokens: HashMap<String, usize>,
}

/// Create default tokenizer
pub fn create_default_tokenizer() -> Tokenizer {
    Tokenizer::new(TokenizerConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode() {
        let mut tokenizer = create_default_tokenizer();

        // Build simple vocabulary
        let texts = vec!["hello world".to_string(), "this is a test".to_string()];
        tokenizer.build_vocab(&texts).unwrap();

        // Test encoding
        let encoded = tokenizer.encode("hello world").unwrap();
        assert!(!encoded.is_empty());

        // Test decoding
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world");
    }
}
