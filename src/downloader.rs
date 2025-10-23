//! Model download module
//!
//! Provides functionality to download model files from platforms like Hugging Face

use crate::error::{CuttleError, Result};
use futures_util::StreamExt;
use log::{info, warn};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Model downloader
pub struct ModelDownloader {
    /// HTTP client
    client: reqwest::Client,
    /// Base URL
    base_url: String,
}

impl ModelDownloader {
    /// Create new model downloader
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://huggingface.co".to_string(),
        }
    }

    /// Create downloader with custom base URL
    pub fn with_base_url(base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
        }
    }

    /// Download Qwen3-0.6B model files
    pub async fn download_qwen3_0_6b<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();

        // Ensure output directory exists
        fs::create_dir_all(output_dir).await.map_err(|e| {
            CuttleError::ModelLoadError(format!("Failed to create output directory: {}", e))
        })?;

        info!(
            "Starting download of Qwen3-0.6B model files to: {:?}",
            output_dir
        );

        // List of files to download
        let files_to_download = vec![
            ("config.json", "Qwen/Qwen3-0.6B/resolve/main/config.json"),
            (
                "tokenizer.json",
                "Qwen/Qwen3-0.6B/resolve/main/tokenizer.json",
            ),
            (
                "tokenizer_config.json",
                "Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json",
            ),
            (
                "model.safetensors",
                "Qwen/Qwen3-0.6B/resolve/main/model.safetensors",
            ),
        ];

        for (filename, url_path) in files_to_download {
            let file_path = output_dir.join(filename);

            // Check if file already exists
            if file_path.exists() {
                info!("File {} already exists, skipping download", filename);
                continue;
            }

            let url = format!("{}/{}", self.base_url, url_path);
            info!("Downloading file: {} from {}", filename, url);

            self.download_file(&url, &file_path).await?;
            info!("File {} download completed", filename);
        }

        info!("Qwen3-0.6B model files download completed");
        Ok(())
    }

    /// Download single file
    async fn download_file<P: AsRef<Path>>(&self, url: &str, output_path: P) -> Result<()> {
        let output_path = output_path.as_ref();

        // Send HTTP request
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| CuttleError::NetworkError(format!("Failed to send request: {}", e)))?;

        if !response.status().is_success() {
            return Err(CuttleError::NetworkError(format!(
                "HTTP error {}: {}",
                response.status(),
                response.status().canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Get file size (if available)
        let total_size = response.content_length();
        if let Some(size) = total_size {
            info!("File size: {} bytes", size);
        }

        // Create output file
        let mut file = File::create(output_path)
            .map_err(|e| CuttleError::ModelLoadError(format!("Failed to create file: {}", e)))?;

        // Stream download file
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|e| CuttleError::NetworkError(format!("Failed to read chunk: {}", e)))?;

            file.write_all(&chunk).map_err(|e| {
                CuttleError::ModelLoadError(format!("Failed to write chunk: {}", e))
            })?;

            downloaded += chunk.len() as u64;

            // Show download progress (every 10MB)
            if downloaded % (10 * 1024 * 1024) == 0 {
                if let Some(total) = total_size {
                    let progress = (downloaded as f64 / total as f64) * 100.0;
                    info!(
                        "Download progress: {:.1}% ({}/{})",
                        progress, downloaded, total
                    );
                } else {
                    info!("Downloaded: {} bytes", downloaded);
                }
            }
        }

        file.flush()
            .map_err(|e| CuttleError::ModelLoadError(format!("Failed to flush file: {}", e)))?;

        info!("File download completed, total size: {} bytes", downloaded);
        Ok(())
    }

    /// Verify downloaded model files
    pub async fn verify_qwen3_0_6b<P: AsRef<Path>>(&self, model_dir: P) -> Result<bool> {
        let model_dir = model_dir.as_ref();

        let required_files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ];

        for filename in required_files {
            let file_path = model_dir.join(filename);
            if !file_path.exists() {
                warn!("Missing required file: {}", filename);
                return Ok(false);
            }

            // Check if file is empty
            let metadata = fs::metadata(&file_path).await.map_err(|e| {
                CuttleError::ModelLoadError(format!("Failed to read file metadata: {}", e))
            })?;

            if metadata.len() == 0 {
                warn!("File is empty: {}", filename);
                return Ok(false);
            }
        }

        info!("Model files verification passed");
        Ok(true)
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}
