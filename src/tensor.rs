//! Tensor operations module
//!
//! Provides high-performance tensor computation functionality using pure Rust implementation

use crate::error::{CuttleError, Result};
use rayon::prelude::*;
use std::ops::{Add, Mul};

use vectra::prelude::*;

/// Multi-dimensional tensor structure with static dimensions
pub type Tensor<const D: usize> = Array<D, f32>;

/// Type aliases for common tensor dimensions
pub type Tensor1D = Tensor<1>;
pub type Tensor2D = Tensor<2>;
pub type Tensor3D = Tensor<3>;
pub type Tensor4D = Tensor<4>;

// Specialized implementation: 2D tensor
impl Tensor2D {
    /// RMS normalization
    pub fn rms_norm(&self, eps: f32) -> Result<Tensor2D> {
        let mut result = self.data.clone();
        let (rows, cols) = (self.shape[0], self.shape[1]);

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;

            // Calculate RMS
            let sum_sq: f32 = result[start..end].iter().map(|x| x * x).sum();
            let rms = (sum_sq / cols as f32 + eps).sqrt();

            // Normalize
            for i in start..end {
                result[i] /= rms;
            }
        }

        Tensor2D::new(result, self.shape)
    }
}

// Specialized implementation: 3D tensor
impl Tensor3D {
    /// 3D tensor and 2D matrix multiplication (batch dimension)
    pub fn matmul_2d(&self, other: &Tensor2D) -> Result<Tensor3D> {
        let (batch_size, seq_len, hidden_size) = (self.shape[0], self.shape[1], self.shape[2]);
        let (hidden_size2, output_size) = (other.shape[0], other.shape[1]);

        if hidden_size != hidden_size2 {
            return Err(CuttleError::TensorError(format!(
                "Matrix dimensions don't match: {}x{}x{} and {}x{}",
                batch_size, seq_len, hidden_size, hidden_size2, output_size
            )));
        }

        let mut result = vec![0.0; batch_size * seq_len * output_size];

        {
            // Pure Rust implementation of batch matrix multiplication
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for j in 0..output_size {
                        let mut sum = 0.0;
                        for k in 0..hidden_size {
                            let self_idx = b * seq_len * hidden_size + i * hidden_size + k;
                            let other_idx = k * output_size + j;
                            sum += self.data[self_idx] * other.data[other_idx];
                        }
                        let result_idx = b * seq_len * output_size + i * output_size + j;
                        result[result_idx] = sum;
                    }
                }
            }
        }

        Tensor3D::new(result, [batch_size, seq_len, output_size])
    }

    /// Alias for 3D tensor and 2D matrix multiplication
    pub fn matmul(&self, other: &Tensor2D) -> Result<Tensor3D> {
        self.matmul_2d(other)
    }

    /// Tensor addition
    pub fn add(&self, other: &Tensor3D) -> Result<Tensor3D> {
        if self.shape != other.shape {
            return Err(CuttleError::TensorError(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor3D::new(result, self.shape)
    }

    /// Tensor multiplication (element-wise)
    pub fn mul(&self, other: &Tensor3D) -> Result<Tensor3D> {
        if self.shape != other.shape {
            return Err(CuttleError::TensorError(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor3D::new(result, self.shape)
    }

    /// RMS normalization
    pub fn rms_norm(&self, eps: f32) -> Result<Tensor3D> {
        let mut result = self.data.clone();
        let (batch_size, seq_len, hidden_size) = (self.shape[0], self.shape[1], self.shape[2]);

        for b in 0..batch_size {
            for s in 0..seq_len {
                let start = b * seq_len * hidden_size + s * hidden_size;
                let end = start + hidden_size;

                // Calculate RMS
                let sum_sq: f32 = result[start..end].iter().map(|x| x * x).sum();
                let rms = (sum_sq / hidden_size as f32 + eps).sqrt();

                // Normalize
                for i in start..end {
                    result[i] /= rms;
                }
            }
        }

        Tensor3D::new(result, self.shape)
    }
}

impl<const N: usize> std::fmt::Display for Tensor<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor<{}>({:?}, data=[...{}])",
            N,
            self.shape,
            self.data.len()
        )
    }
}
