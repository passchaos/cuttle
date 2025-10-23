# Cuttle 🦀

一个基于CPU的大语言模型推理引擎，使用纯Rust实现，专门优化支持Qwen3-0.6B模型。

## ✨ 特性

- 🦀 **纯Rust实现**: 无Python依赖，高性能CPU推理
- 🤖 **Qwen3-0.6B支持**: 专门优化支持Qwen3-0.6B模型
- 🌐 **中英文双语**: 支持中英文双语文本生成
- 📦 **自动下载**: 自动模型下载功能
- 💻 **命令行界面**: 易于使用的CLI工具
- 🔧 **灵活配置**: 可配置的推理参数和分词系统
- 📊 **性能监控**: 内置性能分析和基准测试

## 🏗️ Architecture

Cuttle adopts a modular design with the following main components:

- **Tensor Module** (`tensor`): High-performance tensor operations using pure Rust
- **Model Module** (`model`): Transformer architecture implementation
- **Tokenizer Module** (`tokenizer`): Text tokenization and encoding
- **Inference Engine** (`inference`): Complete inference pipeline
- **Utils Module** (`utils`): Performance monitoring and utility functions

## 📦 安装和构建

### 系统要求

- Rust 1.70+
- 内存: 建议4GB以上
- 存储: 约2GB用于模型文件
- 网络: 首次下载模型需要网络连接

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/passchaos/cuttle.git
cd cuttle

# 调试构建
cargo build

# 发布构建（推荐用于实际使用）
cargo build --release

# 安装命令行工具
cargo install --path .
```

## 🚀 快速开始

### 1. 下载Qwen3-0.6B模型

```bash
# 下载Qwen3-0.6B模型文件到assets目录
cargo run -- download

# 强制重新下载（如果文件已存在）
cargo run -- download --force
```

### 2. 文本生成

```bash
# 中文文本生成
cargo run -- generate --prompt "你好，请介绍一下自己。"

# 英文文本生成
cargo run -- generate --prompt "Hello, how are you?"

# 交互式模式
cargo run -- generate --interactive

# 自定义参数
cargo run -- generate \
  --prompt "请写一首关于春天的诗。" \
  --max-length 200 \
  --temperature 0.8 \
  --top-p 0.9
```

### 3. 查看模型信息

```bash
# 显示模型信息
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

### 配置选项

- `--max-length`: 最大生成长度 (默认: 512)
- `--temperature`: 温度参数，控制随机性 (默认: 1.0)
- `--top-p`: Top-p采样参数 (默认: 0.9)
- `--top-k`: Top-k采样参数 (默认: 50)
- `--interactive`: 交互式模式
- `--force`: 强制重新下载模型

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

### 项目结构

```
cuttle/
├── src/
│   ├── lib.rs          # 库入口
│   ├── main.rs         # 命令行工具
│   ├── model.rs        # 模型定义
│   ├── inference.rs    # 推理引擎
│   ├── tensor.rs       # 张量运算
│   ├── tokenizer.rs    # 分词器
│   ├── downloader.rs   # 模型下载器
│   ├── error.rs        # 错误处理
│   └── utils.rs        # 工具函数
├── assets/             # 模型文件存储目录
│   └── qwen3-0.6b/    # Qwen3-0.6B模型文件
├── examples/           # 示例代码
├── benches/           # 性能测试
├── tests/             # 集成测试
├── Cargo.toml         # 项目配置
└── README.md          # 项目文档
```

## 🤖 Qwen3-0.6B模型配置

- **参数量**: 0.6B
- **词汇表大小**: 151,936
- **隐藏层维度**: 1,024
- **层数**: 28
- **注意力头数**: 16
- **键值头数**: 8 (GQA)
- **支持语言**: 中文、英文等多语言

## 📝 使用示例

### 中文文本生成

```bash
cargo run -- generate --prompt "请写一首关于春天的诗。" --max-length 150
```

### 英文文本生成

```bash
cargo run -- generate --prompt "Explain quantum computing in simple terms." --max-length 200
```

### 交互式对话

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
