//! Utility module
//!
//! Provides various helper functions and utilities

use crate::error::{CuttleError, Result};
use log::{debug, info};
use std::path::Path;
use std::time::{Duration, Instant};

/// Performance timer
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Create a new timer
    pub fn new(name: &str) -> Self {
        info!("Starting timer: {}", name);
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed milliseconds
    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    /// Get elapsed seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Stop timer and log result
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        info!(
            "Timer '{}' finished: {:.2}ms",
            self.name,
            elapsed.as_millis()
        );
        elapsed
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Used memory (bytes)
    pub used_bytes: usize,
    /// Peak memory usage (bytes)
    pub peak_bytes: usize,
}

impl MemoryStats {
    /// Create new memory statistics
    pub fn new() -> Self {
        Self {
            used_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Update memory usage
    pub fn update(&mut self, bytes: usize) {
        self.used_bytes = bytes;
        if bytes > self.peak_bytes {
            self.peak_bytes = bytes;
        }
    }

    /// Get used memory (MB)
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get peak memory (MB)
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / 1024.0 / 1024.0
    }
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory: {:.2}MB used, {:.2}MB peak",
            self.used_mb(),
            self.peak_mb()
        )
    }
}

/// File operation utilities
pub struct FileUtils;

impl FileUtils {
    /// Check if file exists
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    /// Get file size
    pub fn file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = std::fs::metadata(path).map_err(|e| CuttleError::IoError(e))?;
        Ok(metadata.len())
    }

    /// Get file size (MB)
    pub fn file_size_mb<P: AsRef<Path>>(path: P) -> Result<f64> {
        let size = Self::file_size(path)?;
        Ok(size as f64 / 1024.0 / 1024.0)
    }

    /// Create directory (if not exists)
    pub fn create_dir_if_not_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| CuttleError::IoError(e))?;
            info!("Created directory: {:?}", path);
        }
        Ok(())
    }

    /// Safely write file (write to temp file first, then rename)
    pub fn safe_write<P: AsRef<Path>>(path: P, content: &str) -> Result<()> {
        let path = path.as_ref();
        let temp_path = path.with_extension("tmp");

        // Write to temp file
        std::fs::write(&temp_path, content).map_err(|e| CuttleError::IoError(e))?;

        // Rename to target file
        std::fs::rename(&temp_path, path).map_err(|e| CuttleError::IoError(e))?;

        debug!("Safely wrote file: {:?}", path);
        Ok(())
    }

    /// Backup file
    pub fn backup_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(()); // File doesn't exist, no need to backup
        }

        let backup_path = path.with_extension("bak");
        std::fs::copy(path, &backup_path).map_err(|e| CuttleError::IoError(e))?;

        info!("Backed up file: {:?} -> {:?}", path, backup_path);
        Ok(())
    }
}

/// Math utilities
pub struct MathUtils;

impl MathUtils {
    /// Calculate softmax
    pub fn softmax(values: &[f32]) -> Vec<f32> {
        if values.is_empty() {
            return vec![];
        }

        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = values.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();

        if sum == 0.0 {
            return vec![1.0 / values.len() as f32; values.len()];
        }

        exp_values.iter().map(|&x| x / sum).collect()
    }

    /// Calculate log softmax
    pub fn log_softmax(values: &[f32]) -> Vec<f32> {
        if values.is_empty() {
            return vec![];
        }

        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let shifted: Vec<f32> = values.iter().map(|&x| x - max_val).collect();
        let log_sum_exp = shifted.iter().map(|&x| x.exp()).sum::<f32>().ln();

        shifted.iter().map(|&x| x - log_sum_exp).collect()
    }

    /// Calculate cosine similarity
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CuttleError::TensorError(
                "Vectors must have the same length".to_string(),
            ));
        }

        if a.is_empty() {
            return Ok(0.0);
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Calculate Euclidean distance
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CuttleError::TensorError(
                "Vectors must have the same length".to_string(),
            ));
        }

        let sum_squared_diff: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum();

        Ok(sum_squared_diff.sqrt())
    }

    /// Normalize vector
    pub fn normalize(values: &[f32]) -> Vec<f32> {
        let norm: f32 = values.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return vec![0.0; values.len()];
        }

        values.iter().map(|&x| x / norm).collect()
    }

    /// Calculate mean
    pub fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = Self::mean(values);
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;

        variance.sqrt()
    }
}

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Truncate string to specified length
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    /// Clean text (remove extra whitespace)
    pub fn clean_text(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Count words in text
    pub fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Count characters in text (excluding spaces)
    pub fn char_count(text: &str) -> usize {
        text.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Format byte size
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Format duration
    pub fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else {
            format!("{}ms", millis)
        }
    }
}

/// Progress bar
#[derive(Debug)]
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    start_time: Instant,
}

impl ProgressBar {
    /// Create new progress bar
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            width: 50,
            start_time: Instant::now(),
        }
    }

    /// Set progress bar width
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current.min(self.total);
    }

    /// Increment progress
    pub fn increment(&mut self) {
        self.update(self.current + 1);
    }

    /// Display progress bar
    pub fn display(&self) -> String {
        let percentage = if self.total == 0 {
            100.0
        } else {
            (self.current as f64 / self.total as f64) * 100.0
        };

        let filled = ((self.current as f64 / self.total as f64) * self.width as f64) as usize;
        let empty = self.width - filled;

        let elapsed = self.start_time.elapsed();
        let eta = if self.current > 0 {
            let rate = self.current as f64 / elapsed.as_secs_f64();
            let remaining = (self.total - self.current) as f64 / rate;
            StringUtils::format_duration(Duration::from_secs_f64(remaining))
        } else {
            "--:--".to_string()
        };

        format!(
            "[{}{}] {:.1}% ({}/{}) ETA: {}",
            "█".repeat(filled),
            "░".repeat(empty),
            percentage,
            self.current,
            self.total,
            eta
        )
    }
}

impl std::fmt::Display for ProgressBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_utils_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = MathUtils::softmax(&values);

        // Check if sum equals 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check if monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_string_utils_format_bytes() {
        assert_eq!(StringUtils::format_bytes(1024), "1.00 KB");
        assert_eq!(StringUtils::format_bytes(1048576), "1.00 MB");
        assert_eq!(StringUtils::format_bytes(500), "500 B");
    }

    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(100);
        pb.update(50);
        let display = pb.display();
        assert!(display.contains("50.0%"));
        assert!(display.contains("(50/100)"));
    }
}
