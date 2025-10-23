//! 推理引擎性能基准测试

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cuttle::{
    InferenceEngine, Model, ModelConfig, Tokenizer, InferenceConfig,
    tensor::Tensor, tokenizer::create_default_tokenizer,
};

/// 创建测试用的小型模型
fn create_test_model() -> Model {
    let config = ModelConfig {
        vocab_size: 1000,
        hidden_size: 256,
        num_layers: 4,
        num_attention_heads: 8,
        intermediate_size: 1024,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
    };
    
    Model::new(config).expect("Failed to create test model")
}

/// 创建测试用的分词器
fn create_test_tokenizer() -> Tokenizer {
    let mut tokenizer = create_default_tokenizer();
    
    // 构建简单的词汇表
    let test_texts = vec![
        "hello world this is a test".to_string(),
        "machine learning artificial intelligence".to_string(),
        "rust programming language performance".to_string(),
        "neural networks deep learning".to_string(),
    ];
    
    tokenizer.build_vocab(&test_texts).expect("Failed to build vocab");
    tokenizer
}

/// 张量操作基准测试
fn tensor_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    
    // 矩阵乘法基准测试
    for size in [64, 128, 256, 512].iter() {
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            size,
            |b, &size| {
                let tensor_a = Tensor::randn(&[size, size]).unwrap();
                let tensor_b = Tensor::randn(&[size, size]).unwrap();
                
                b.iter(|| {
                    black_box(tensor_a.matmul(&tensor_b).unwrap())
                })
            },
        );
    }
    
    // 激活函数基准测试
    for size in [1024, 4096, 16384].iter() {
        group.bench_with_input(
            BenchmarkId::new("gelu", size),
            size,
            |b, &size| {
                let tensor = Tensor::randn(&[size]).unwrap();
                
                b.iter(|| {
                    black_box(tensor.gelu())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("relu", size),
            size,
            |b, &size| {
                let tensor = Tensor::randn(&[size]).unwrap();
                
                b.iter(|| {
                    black_box(tensor.relu())
                })
            },
        );
    }
    
    // Softmax基准测试
    for size in [1000, 5000, 10000, 32000].iter() {
        group.bench_with_input(
            BenchmarkId::new("softmax", size),
            size,
            |b, &size| {
                let tensor = Tensor::randn(&[size]).unwrap();
                
                b.iter(|| {
                    black_box(tensor.softmax(0).unwrap())
                })
            },
        );
    }
    
    // RMS归一化基准测试
    for size in [256, 512, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("rms_norm", size),
            size,
            |b, &size| {
                let tensor = Tensor::randn(&[32, *size]).unwrap();
                
                b.iter(|| {
                    black_box(tensor.rms_norm(1e-6).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// 分词器基准测试
fn tokenizer_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer");
    let tokenizer = create_test_tokenizer();
    
    let test_texts = vec![
        "Short text",
        "This is a medium length text that contains several words and should test tokenization performance",
        "This is a much longer text that contains many more words and is designed to test the performance of the tokenizer when processing longer sequences of text that might be more representative of real-world usage scenarios where users input substantial amounts of text for processing",
    ];
    
    // 编码基准测试
    for (i, text) in test_texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("encode", i),
            text,
            |b, &text| {
                b.iter(|| {
                    black_box(tokenizer.encode(text).unwrap())
                })
            },
        );
    }
    
    // 解码基准测试
    let token_sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        (1..50).collect::<Vec<_>>(),
    ];
    
    for (i, tokens) in token_sequences.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("decode", i),
            tokens,
            |b, tokens| {
                b.iter(|| {
                    black_box(tokenizer.decode(tokens).unwrap_or_default())
                })
            },
        );
    }
    
    group.finish();
}

/// 模型前向传播基准测试
fn model_forward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_forward");
    let model = create_test_model();
    
    // 不同序列长度的前向传播
    for seq_len in [8, 16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            seq_len,
            |b, &seq_len| {
                let input_ids: Vec<usize> = (0..seq_len).map(|i| i % 1000).collect();
                
                b.iter(|| {
                    black_box(model.forward(&input_ids).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// 端到端推理基准测试
fn end_to_end_inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_inference");
    
    let model = create_test_model();
    let tokenizer = create_test_tokenizer();
    
    let inference_config = InferenceConfig {
        max_length: 50,
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        do_sample: false, // 使用贪婪解码以获得确定性结果
        repetition_penalty: 1.0,
    };
    
    let engine = InferenceEngine::with_config(model, tokenizer, inference_config);
    
    let test_prompts = vec![
        "Hello",
        "The quick brown fox",
        "Machine learning is",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("generate", i),
            prompt,
            |b, &prompt| {
                b.iter(|| {
                    black_box(engine.generate(prompt).unwrap_or_default())
                })
            },
        );
    }
    
    group.finish();
}

/// 内存使用基准测试
fn memory_usage_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // 测试不同大小模型的内存分配
    let configs = vec![
        ("tiny", ModelConfig {
            vocab_size: 500,
            hidden_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 512,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
        }),
        ("small", ModelConfig {
            vocab_size: 1000,
            hidden_size: 256,
            num_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 1024,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
        }),
        ("medium", ModelConfig {
            vocab_size: 2000,
            hidden_size: 512,
            num_layers: 8,
            num_attention_heads: 16,
            intermediate_size: 2048,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
        }),
    ];
    
    for (name, config) in configs {
        group.bench_function(
            BenchmarkId::new("model_creation", name),
            |b| {
                b.iter(|| {
                    black_box(Model::new(config.clone()).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// 并行处理基准测试
fn parallel_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");
    
    let model = create_test_model();
    let tokenizer = create_test_tokenizer();
    let engine = InferenceEngine::new(model, tokenizer);
    
    let batch_prompts = vec![
        "Hello world".to_string(),
        "How are you".to_string(),
        "What is AI".to_string(),
        "Explain machine learning".to_string(),
        "Tell me about Rust".to_string(),
    ];
    
    // 顺序处理
    group.bench_function("sequential_batch", |b| {
        b.iter(|| {
            let results: Vec<_> = batch_prompts.iter()
                .map(|prompt| engine.generate(prompt).unwrap_or_default())
                .collect();
            black_box(results)
        })
    });
    
    // 批量处理
    group.bench_function("batch_processing", |b| {
        b.iter(|| {
            black_box(engine.generate_batch(&batch_prompts).unwrap_or_default())
        })
    });
    
    group.finish();
}

/// 数学工具基准测试
fn math_utils_benchmark(c: &mut Criterion) {
    use cuttle::utils::MathUtils;
    
    let mut group = c.benchmark_group("math_utils");
    
    // Softmax基准测试
    for size in [100, 1000, 10000, 32000].iter() {
        let values: Vec<f32> = (0..*size).map(|i| i as f32 * 0.01).collect();
        
        group.bench_with_input(
            BenchmarkId::new("softmax", size),
            &values,
            |b, values| {
                b.iter(|| {
                    black_box(MathUtils::softmax(values))
                })
            },
        );
    }
    
    // 余弦相似度基准测试
    for size in [128, 256, 512, 1024].iter() {
        let vec_a: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();
        let vec_b: Vec<f32> = (0..*size).map(|i| (i as f32).cos()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", size),
            &(vec_a, vec_b),
            |b, (vec_a, vec_b)| {
                b.iter(|| {
                    black_box(MathUtils::cosine_similarity(vec_a, vec_b).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    tensor_operations_benchmark,
    tokenizer_benchmark,
    model_forward_benchmark,
    end_to_end_inference_benchmark,
    memory_usage_benchmark,
    parallel_processing_benchmark,
    math_utils_benchmark
);

criterion_main!(benches);
