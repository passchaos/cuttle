//! Cuttle - CPU-based Large Language Model Inference Engine
//!
//! Command line tool for running large language model inference

use clap::{Parser, Subcommand};
use cuttle::{
    InferenceConfig, InferenceEngine, Model, ModelConfig, Tokenizer,
    downloader::ModelDownloader,
    error::{CuttleError, Result},
    utils::{ProgressBar, StringUtils, Timer},
};
use log::{error, info, warn};
use std::io::{self, Write};
use std::path::PathBuf;

/// Cuttle CLI arguments
#[derive(Parser)]
#[command(name = "cuttle")]
#[command(about = "A CPU-based large language model inference engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

/// Available commands
#[derive(Subcommand)]
enum Commands {
    /// Initialize new model configuration
    Init {
        /// Output directory
        #[arg(short, long, default_value = "./model")]
        output: PathBuf,

        /// Model type
        #[arg(short, long, default_value = "default")]
        model_type: String,
    },

    /// Download Qwen3-0.6B model files
    Download {
        /// Output directory for model files
        #[arg(short, long, default_value = "./assets/qwen3-0.6b")]
        output: PathBuf,

        /// Force re-download even if files exist
        #[arg(short, long)]
        force: bool,
    },

    /// Run text generation
    Generate {
        /// Model configuration file path
        #[arg(short, long)]
        model_config: Option<PathBuf>,

        /// Tokenizer file path
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Input prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Maximum generation length
        #[arg(long, default_value = "512")]
        max_length: usize,

        /// Temperature parameter
        #[arg(long, default_value = "1.0")]
        temperature: f32,

        /// Top-p parameter
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Top-k parameter
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Interactive mode
        #[arg(short, long)]
        interactive: bool,
    },

    /// Evaluate model performance
    Evaluate {
        /// Model configuration file path
        #[arg(short, long)]
        model_config: Option<PathBuf>,

        /// Tokenizer file path
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Test text file
        #[arg(short, long)]
        test_file: Option<PathBuf>,
    },

    /// Show model information
    Info {
        /// Model configuration file path
        #[arg(short, long)]
        model_config: Option<PathBuf>,

        /// Tokenizer file path
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    init_logger(&cli.log_level, cli.verbose);

    // Execute command
    if let Err(e) = run_command(cli.command) {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Initialize logging system
fn init_logger(level: &str, verbose: bool) {
    let log_level = if verbose { "debug" } else { level };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();
}

/// Execute command
fn run_command(command: Commands) -> Result<()> {
    match command {
        Commands::Init { output, model_type } => init_model(&output, &model_type),
        Commands::Download { output, force } => tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(download_qwen3_model(&output, force)),
        Commands::Generate {
            model_config,
            tokenizer,
            prompt,
            max_length,
            temperature,
            top_p,
            top_k,
            interactive,
        } => {
            let inference_config = InferenceConfig {
                max_length,
                temperature,
                top_p,
                top_k,
                do_sample: true,
                repetition_penalty: 1.1,
            };

            if interactive {
                run_interactive_mode(model_config, tokenizer, inference_config)
            } else {
                run_generation(model_config, tokenizer, prompt, inference_config)
            }
        }
        Commands::Evaluate {
            model_config,
            tokenizer,
            test_file,
        } => run_evaluation(model_config, tokenizer, test_file),
        Commands::Info {
            model_config,
            tokenizer,
        } => show_model_info(model_config, tokenizer),
    }
}

/// Initialize model configuration
fn init_model(output_dir: &PathBuf, model_type: &str) -> Result<()> {
    info!("Initializing model in directory: {:?}", output_dir);

    // Create output directory
    std::fs::create_dir_all(output_dir).map_err(|e| CuttleError::IoError(e))?;

    // Create model configuration
    let config = match model_type {
        "small" => ModelConfig {
            vocab_size: 16000,
            hidden_size: 1024,
            num_layers: 12,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            intermediate_size: 4096,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            use_sliding_window: Some(false),
            sliding_window: None,
            model_type: Some("small".to_string()),
            architectures: Some(vec!["SmallForCausalLM".to_string()]),
        },
        "large" => ModelConfig {
            vocab_size: 50000,
            hidden_size: 8192,
            num_layers: 48,
            num_attention_heads: 64,
            num_key_value_heads: Some(32),
            intermediate_size: 32768,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: Some(10000.0),
            use_sliding_window: Some(false),
            sliding_window: None,
            model_type: Some("large".to_string()),
            architectures: Some(vec!["LargeForCausalLM".to_string()]),
        },
        _ => ModelConfig::default(),
    };

    // Save configuration file
    let config_path = output_dir.join("config.json");
    let config_str = serde_json::to_string_pretty(&config).map_err(|e| {
        CuttleError::SerializationError(format!("Failed to serialize config: {}", e))
    })?;

    std::fs::write(&config_path, config_str).map_err(|e| CuttleError::IoError(e))?;

    info!("Model configuration saved to: {:?}", config_path);

    // Create example tokenizer
    let tokenizer = cuttle::tokenizer::create_default_tokenizer();
    let tokenizer_path = output_dir.join("tokenizer.json");
    tokenizer.save(&tokenizer_path)?;

    info!("Tokenizer saved to: {:?}", tokenizer_path);

    println!("✅ Model initialized successfully!");
    println!("📁 Output directory: {:?}", output_dir);
    println!("📄 Config file: {:?}", config_path);
    println!("🔤 Tokenizer file: {:?}", tokenizer_path);

    Ok(())
}

/// Run text generation
fn run_generation(
    model_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    prompt: Option<String>,
    inference_config: InferenceConfig,
) -> Result<()> {
    let timer = Timer::new("Model Loading");

    // Load inference engine
    let engine = load_inference_engine(model_config_path, tokenizer_path)?;
    timer.stop();

    // Get prompt text
    let prompt_text = match prompt {
        Some(text) => text,
        None => {
            print!("Enter your prompt: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        }
    };

    if prompt_text.is_empty() {
        return Err(CuttleError::InferenceError("Empty prompt".to_string()));
    }

    println!("\n🤖 Generating response...");
    println!("📝 Prompt: {}", StringUtils::truncate(&prompt_text, 100));
    println!(
        "⚙️  Config: max_len={}, temp={:.1}, top_p={:.1}, top_k={}",
        inference_config.max_length,
        inference_config.temperature,
        inference_config.top_p,
        inference_config.top_k
    );

    let generation_timer = Timer::new("Text Generation");

    // Create engine with configuration
    let mut engine_with_config = InferenceEngine::with_config(
        engine.model().clone(),
        engine.tokenizer().clone(),
        inference_config,
    );

    // Generate text
    let generated_text = engine_with_config.generate(&prompt_text)?;
    let elapsed = generation_timer.stop();

    // Display results
    println!("\n✨ Generated text:");
    println!("{}", "─".repeat(50));
    println!("{}", generated_text);
    println!("{}", "─".repeat(50));
    println!("⏱️  Generation time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "📊 Words generated: {}",
        StringUtils::word_count(&generated_text)
    );

    Ok(())
}

/// Run interactive mode
fn run_interactive_mode(
    model_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    inference_config: InferenceConfig,
) -> Result<()> {
    println!("🚀 Starting Cuttle Interactive Mode");
    println!("Type 'quit' or 'exit' to stop, 'help' for commands\n");

    let timer = Timer::new("Model Loading");
    let engine = load_inference_engine(model_config_path, tokenizer_path)?;
    timer.stop();

    let mut engine_with_config = InferenceEngine::with_config(
        engine.model().clone(),
        engine.tokenizer().clone(),
        inference_config,
    );

    loop {
        print!("cuttle> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();

        match input {
            "quit" | "exit" => {
                println!("👋 Goodbye!");
                break;
            }
            "help" => {
                show_interactive_help();
            }
            "info" => {
                let model_info = engine_with_config.model_info();
                println!("{}", model_info);
            }
            "" => continue,
            _ => {
                let timer = Timer::new("Generation");
                match engine_with_config.generate(input) {
                    Ok(response) => {
                        let elapsed = timer.stop();
                        println!("\n🤖: {}", response);
                        println!("⏱️  ({:.2}s)\n", elapsed.as_secs_f64());
                    }
                    Err(e) => {
                        error!("Generation failed: {}", e);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Show interactive mode help
fn show_interactive_help() {
    println!("\n📚 Available commands:");
    println!("  help     - Show this help message");
    println!("  info     - Show model information");
    println!("  quit/exit - Exit the program");
    println!("  <text>   - Generate response for the given text\n");
}

/// Run model evaluation
fn run_evaluation(
    model_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    test_file_path: Option<PathBuf>,
) -> Result<()> {
    println!("📊 Starting model evaluation...");

    let engine = load_inference_engine(model_config_path, tokenizer_path)?;

    // Read test file or use default test texts
    let test_texts = match test_file_path {
        Some(path) => {
            let content = std::fs::read_to_string(&path).map_err(|e| CuttleError::IoError(e))?;
            content.lines().map(|s| s.to_string()).collect()
        }
        None => {
            vec![
                "The quick brown fox jumps over the lazy dog.".to_string(),
                "Artificial intelligence is transforming the world.".to_string(),
                "Machine learning models require large datasets.".to_string(),
            ]
        }
    };

    println!("🧪 Evaluating {} test samples...", test_texts.len());

    let mut total_perplexity = 0.0;
    let mut valid_samples = 0;
    let mut progress = ProgressBar::new(test_texts.len());

    for (i, text) in test_texts.iter().enumerate() {
        progress.update(i + 1);
        print!("\r{}", progress);
        io::stdout().flush().unwrap();

        match engine.perplexity(text) {
            Ok(perplexity) => {
                total_perplexity += perplexity;
                valid_samples += 1;
            }
            Err(e) => {
                warn!("Failed to compute perplexity for sample {}: {}", i, e);
            }
        }
    }

    println!(); // New line

    if valid_samples > 0 {
        let avg_perplexity = total_perplexity / valid_samples as f32;
        println!("\n📈 Evaluation Results:");
        println!("  Valid samples: {}/{}", valid_samples, test_texts.len());
        println!("  Average perplexity: {:.2}", avg_perplexity);
    } else {
        println!("❌ No valid samples for evaluation");
    }

    Ok(())
}

/// Show model information
fn show_model_info(
    model_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
) -> Result<()> {
    let engine = load_inference_engine(model_config_path, tokenizer_path)?;
    let model_info = engine.model_info();

    println!("\n📋 {}", model_info);

    Ok(())
}

/// Load inference engine
fn load_inference_engine(
    model_config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
) -> Result<InferenceEngine> {
    let model_config_path =
        model_config_path.unwrap_or_else(|| PathBuf::from("./model/config.json"));
    let tokenizer_path = tokenizer_path.unwrap_or_else(|| PathBuf::from("./model/tokenizer.json"));

    info!("Loading model config from: {:?}", model_config_path);
    info!("Loading tokenizer from: {:?}", tokenizer_path);

    // Check if files exist
    if !model_config_path.exists() {
        return Err(CuttleError::ModelLoadError(format!(
            "Model config file not found: {:?}",
            model_config_path
        )));
    }

    if !tokenizer_path.exists() {
        return Err(CuttleError::TokenizerError(format!(
            "Tokenizer file not found: {:?}",
            tokenizer_path
        )));
    }

    InferenceEngine::from_config_files(model_config_path, tokenizer_path)
}

/// Download Qwen3-0.6B model
async fn download_qwen3_model(output_dir: &PathBuf, force: bool) -> Result<()> {
    info!("Starting download of Qwen3-0.6B model to: {:?}", output_dir);

    let downloader = ModelDownloader::new();

    // If not forcing download, check if files already exist
    if !force {
        if let Ok(valid) = downloader.verify_qwen3_0_6b(output_dir).await {
            if valid {
                info!("Model files already exist and are complete, skipping download");
                return Ok(());
            }
        }
    }

    // Download model files
    downloader.download_qwen3_0_6b(output_dir).await?;

    // Verify downloaded files
    let valid = downloader.verify_qwen3_0_6b(output_dir).await?;
    if !valid {
        return Err(CuttleError::ModelLoadError(
            "Downloaded model file verification failed".to_string(),
        ));
    }

    info!("Qwen3-0.6B model download completed!");
    Ok(())
}
