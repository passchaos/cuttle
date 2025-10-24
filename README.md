# Cuttle 🦀

A CPU-based large language model inference engine implemented in pure Rust, specifically optimized for Qwen3-0.6B model.

## ✨ Features

- 🦀 **Pure Rust Implementation**: No Python dependencies, high-performance CPU inference
- 🤖 **Qwen3-0.6B Support**: Specifically optimized for Qwen3-0.6B model
- 🌐 **Bilingual Support**: Supports both Chinese and English text generation
- 📦 **Auto Download**: Automatic model download functionality
- 💻 **Command Line Interface**: Easy-to-use CLI tool
- 🔧 **Flexible Configuration**: Configurable inference parameters and tokenization system
- 📊 **Performance Monitoring**: Built-in performance analysis and benchmarking

## 🏗️ Architecture

Cuttle adopts a modular design with the following main components:

- **Tensor Module** (`tensor`): High-performance tensor operations using pure Rust
- **Model Module** (`model`): Transformer architecture implementation
- **Tokenizer Module** (`tokenizer`): Text tokenization and encoding
- **Inference Engine** (`inference`): Complete inference pipeline
- **Utils Module** (`utils`): Performance monitoring and utility functions

## 📦 Installation and Build

### System Requirements

- Rust 1.70+
- Memory: 4GB+ recommended
- Storage: ~2GB for model files
- Network: Internet connection required for initial model download

### Build from Source

```bash
# Clone repository
git clone https://github.com/passchaos/cuttle.git
cd cuttle

# Debug build
cargo build

# Release build (recommended for production use)
cargo build --release

# Install command line tool
cargo install --path .
```

## 🚀 Quick Start

### 1. Download Qwen3-0.6B Model

```bash
# Download Qwen3-0.6B model files to assets directory
cargo run -- download

# Force re-download (if files already exist)
cargo run -- download --force
```

### 2. Text Generation

```bash
# Chinese text generation
cargo run -- generate --prompt "你好，请介绍一下自己。"

# English text generation
cargo run -- generate --prompt "Hello, how are you?"

# Interactive mode
cargo run -- generate --interactive

# Custom parameters
cargo run -- generate \
  --prompt "请写一首关于春天的诗。" \
  --max-length 200 \
  --temperature 0.8 \
  --top-p 0.9
```

### 3. View Model Information

```bash
# Display model information
cargo run -- info
```

## 💻 Programming Interface

### Basic Usage

```rust
use cuttle::{
    InferenceEngine, Model, ModelConfig, 
    Tokenizer, InferenceConfig
};

// Create model configuration
let config = ModelConfig::default();
let model = Model::new(config)?;

// Create tokenizer
let mut tokenizer = cuttle::tokenizer::create_default_tokenizer();
let texts = vec!["hello world".to_string()];
tokenizer.build_vocab(&texts)?;

// Create inference engine
let engine = InferenceEngine::new(model, tokenizer);

// Generate text
let response = engine.generate("Hello, how are you?")?;
println!("Generated: {}", response);
```

### Custom Inference Configuration

```rust
let inference_config = InferenceConfig {
    max_length: 512,
    temperature: 0.8,
    top_p: 0.9,
    top_k: 50,
    do_sample: true,
    repetition_penalty: 1.1,
};

let engine = InferenceEngine::with_config(model, tokenizer, inference_config);
```

### Batch Processing

```rust
let prompts = vec![
    "What is AI?".to_string(),
    "Explain machine learning".to_string(),
    "How does deep learning work?".to_string(),
];

let responses = engine.generate_batch(&prompts)?;
for (prompt, response) in prompts.iter().zip(responses.iter()) {
    println!("Q: {}\nA: {}\n", prompt, response);
}
```

### Tensor Operations

```rust
use cuttle::tensor::Tensor;

// Create tensors
let a = Tensor::randn(&[128, 256])?;
let b = Tensor::randn(&[256, 512])?;

// Matrix multiplication
let c = a.matmul(&b)?;

// Activation function
let activated = c.gelu();

// Softmax
let probs = activated.softmax(1)?;
```

## ⚙️ Configuration

### Model Configuration (config.json)

```json
{
  "vocab_size": 32000,
  "hidden_size": 4096,
  "num_layers": 32,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "max_position_embeddings": 2048,
  "rms_norm_eps": 1e-6
}
```

### Configuration Options

- `--max-length`: Maximum generation length (default: 512)
- `--temperature`: Temperature parameter, controls randomness (default: 1.0)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--interactive`: Interactive mode
- `--force`: Force re-download model

## 📊 Performance Benchmarks

Run benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench tensor_operations
cargo bench inference
```

### Performance Optimization Tips

1. **Compilation Optimization**: Use `--release` mode
2. **Pure Rust Implementation**: No external BLAS dependencies required
3. **Parallel Processing**: Utilize Rayon for parallel computation
4. **Memory Management**: Avoid unnecessary memory allocations

## 🧪 Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration

# Run documentation tests
cargo test --doc
```

## 📚 API Documentation

Generate and view API documentation:

```bash
cargo doc --open
```

## 🛠️ Development

### Project Structure

```
cuttle/
├── src/
│   ├── lib.rs          # Library entry point
│   ├── main.rs         # Command line tool
│   ├── model.rs        # Model definition
│   ├── inference.rs    # Inference engine
│   ├── tensor.rs       # Tensor operations
│   ├── tokenizer.rs    # Tokenizer
│   ├── downloader.rs   # Model downloader
│   ├── error.rs        # Error handling
│   └── utils.rs        # Utility functions
├── assets/             # Model file storage directory
│   └── qwen3-0.6b/    # Qwen3-0.6B model files
├── examples/           # Example code
├── benches/           # Performance tests
├── tests/             # Integration tests
├── Cargo.toml         # Project configuration
└── README.md          # Project documentation
```

## 🤖 Qwen3-0.6B Model Configuration

- **Parameters**: 0.6B
- **Vocabulary Size**: 151,936
- **Hidden Dimension**: 1,024
- **Layers**: 28
- **Attention Heads**: 16
- **Key-Value Heads**: 8 (GQA)
- **Supported Languages**: Chinese, English, and other multilingual support

## 📝 Usage Examples

### Chinese Text Generation

```bash
cargo run -- generate --prompt "请写一首关于春天的诗。" --max-length 150
```

### English Text Generation

```bash
cargo run -- generate --prompt "Explain quantum computing in simple terms." --max-length 200
```

### Interactive Dialogue

```bash
cargo run -- generate --interactive
```

### Contributing Guidelines

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

### Code Style

- Use `rustfmt` to format code
- Use `clippy` for code linting
- Write comprehensive documentation and tests

```bash
# Format code
cargo fmt

# Code linting
cargo clippy
```

## 🔧 Troubleshooting

### Common Issues

**Q: Compilation errors**

A: Ensure you have the latest Rust toolchain:

```bash
# Update Rust
rustup update

# Use Rust 2024 edition
rustup toolchain install nightly
```

**Q: Slow inference speed**

A: Check the following optimization options:
- Compile with `--release` mode
- Adjust batch processing size
- Use smaller models for testing
- Enable parallel processing

**Q: High memory usage**

A: Try the following approaches:
- Reduce model size
- Lower batch processing size
- Use smaller sequence lengths

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [rayon](https://github.com/rayon-rs/rayon) - Parallel computing framework
- [serde](https://github.com/serde-rs/serde) - Serialization framework
- [clap](https://github.com/clap-rs/clap) - Command line argument parsing
- [tokio](https://github.com/tokio-rs/tokio) - Asynchronous runtime

## 🔗 Related Links

- [Documentation](https://docs.rs/cuttle)
- [Examples](./examples)
- [Changelog](./CHANGELOG.md)
- [Contributing Guide](./CONTRIBUTING.md)

---

**Cuttle** - Power your AI inference with Rust 🦀✨
