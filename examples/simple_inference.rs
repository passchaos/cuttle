//! Simple inference example for Qwen3-0.6B
//!
//! This example demonstrates basic text generation using the Cuttle inference engine with Qwen3-0.6B model.

use cuttle::{InferenceConfig, InferenceEngine, Model, Tokenizer, error::Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Qwen3-0.6B Inference Example");
    println!("{}", "=".repeat(50));

    // Use Qwen3-0.6B model files from assets directory
    let model_config_path = PathBuf::from("./assets/qwen3-0.6b/config.json");
    let tokenizer_path = PathBuf::from("./assets/qwen3-0.6b/tokenizer.json");

    // Check if model files exist
    if !model_config_path.exists() {
        eprintln!("Model config file not found: {:?}", model_config_path);
        eprintln!("Please run 'cargo run -- download' to download Qwen3-0.6B model files first.");
        return Ok(());
    }

    if !tokenizer_path.exists() {
        eprintln!("Tokenizer file not found: {:?}", tokenizer_path);
        eprintln!("Please run 'cargo run -- download' to download Qwen3-0.6B model files first.");
        return Ok(());
    }

    println!("📋 Loading Qwen3-0.6B model...");
    let model = Model::from_config_file(&model_config_path)?;
    let tokenizer = Tokenizer::load(&tokenizer_path)?;

    println!("📊 Model information:");
    println!("  - Vocabulary Size: {}", tokenizer.vocab_size());
    println!("  - Model: Qwen3-0.6B");

    // Create inference engine with Qwen3-optimized config
    let inference_config = InferenceConfig {
        max_length: 256,
        temperature: 0.7,
        top_p: 0.8,
        top_k: 40,
        do_sample: true,
        repetition_penalty: 1.05,
    };

    println!("⚙️  Creating inference engine...");
    let engine = InferenceEngine::with_config(model, tokenizer, inference_config);

    // Test prompts for Qwen3-0.6B (supporting both Chinese and English)
    println!("\n✨ Text Generation Examples:");
    println!("{}", "-".repeat(40));

    let test_prompts = vec![
        "你好，请介绍一下自己。",
        "What is the capital of France?",
        "请写一首关于春天的诗。",
        "Explain quantum computing in simple terms.",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n=== Example {} ===", i + 1);
        println!("Prompt: {}", prompt);
        println!("Generated:");

        match engine.generate(prompt) {
            Ok(generated_text) => {
                println!("{}", generated_text);
            }
            Err(e) => {
                println!("Generation failed: {}", e);
            }
        }

        println!("{}", "=".repeat(50));
    }

    println!("\n✅ Qwen3-0.6B inference example completed!");
    println!("💡 Note: Qwen3-0.6B supports both Chinese and English text generation");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_inference() {
        // Create minimal configuration for testing
        let model_config = ModelConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 256,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
        };

        let model = Model::new(model_config).unwrap();
        let mut tokenizer = create_default_tokenizer();

        let texts = vec!["test text".to_string()];
        tokenizer.build_vocab(&texts).unwrap();

        let engine = InferenceEngine::new(model, tokenizer);

        // Test basic functionality
        assert!(engine.tokenizer().vocab_size() > 0);

        // Test encoding and decoding
        let encoded = engine.tokenizer().encode("test").unwrap();
        assert!(!encoded.is_empty());

        let decoded = engine.tokenizer().decode(&encoded).unwrap();
        assert!(!decoded.is_empty());
    }
}
