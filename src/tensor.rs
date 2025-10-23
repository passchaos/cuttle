//! Tensor operations module
//!
//! Provides high-performance tensor computation functionality using pure Rust implementation

use crate::error::{CuttleError, Result};
use rayon::prelude::*;
use std::ops::{Add, Mul};

/// Multi-dimensional tensor structure with static dimensions
#[derive(Debug, Clone)]
pub struct Tensor<const N: usize> {
    data: Vec<f32>,
    shape: [usize; N],
}

/// Type aliases for common tensor dimensions
pub type Tensor1D = Tensor<1>;
pub type Tensor2D = Tensor<2>;
pub type Tensor3D = Tensor<3>;
pub type Tensor4D = Tensor<4>;

impl<const N: usize> Tensor<N> {
    /// Create a new tensor
    pub fn new(data: Vec<f32>, shape: [usize; N]) -> Result<Self> {
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(CuttleError::TensorError(format!(
                "Data size {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }
        Ok(Self { data, shape })
    }

    /// Create a zero tensor from shape
    pub fn zeros(shape: [usize; N]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0; size];
        Self::new(data, shape)
    }

    /// Create a random tensor from shape
    pub fn randn(shape: [usize; N]) -> Result<Self> {
        use std::f32::consts::PI;
        let size = shape.iter().product::<usize>();
        let mut values = Vec::with_capacity(size);

        // Simple Box-Muller transform to generate normally distributed random numbers
        for i in (0..size).step_by(2) {
            let u1: f32 = (i as f32 + 1.0) / (size as f32 + 1.0);
            let u2: f32 = ((i + 1) as f32 + 1.0) / (size as f32 + 1.0);

            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();

            values.push(z0);
            if i + 1 < size {
                values.push(z1);
            }
        }

        Self::new(values, shape)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    /// Get tensor number of dimensions
    pub fn ndim(&self) -> usize {
        N
    }

    /// Get data reference
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    /// Get mutable data reference
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Self {
        let result: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Self {
            data: result,
            shape: self.shape,
        }
    }

    /// ReLU activation function
    pub fn relu(&self) -> Self {
        let result: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Self {
            data: result,
            shape: self.shape,
        }
    }

    /// GELU activation function
    pub fn gelu(&self) -> Self {
        let result: Vec<f32> = self
            .data
            .iter()
            .map(|x| {
                0.5_f32 * x * (1.0_f32 + (0.7978845608_f32 * (x + 0.044715_f32 * x.powi(3))).tanh())
            })
            .collect();
        Self {
            data: result,
            shape: self.shape,
        }
    }
}

// Specialized implementation: 1D tensor
impl Tensor1D {
    /// Create 1D tensor from vector
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = [data.len()];
        Self { data, shape }
    }
}

// Specialized implementation: 2D tensor
impl Tensor2D {
    /// Matrix multiplication (2D x 2D)
    pub fn matmul(&self, other: &Tensor2D) -> Result<Tensor2D> {
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(CuttleError::TensorError(format!(
                "Matrix dimensions don't match: {}x{} and {}x{}",
                m, k, k2, n
            )));
        }

        let mut result = vec![0.0; m * n];

        {
            // Pure Rust implementation of matrix multiplication
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.data[i * k + l] * other.data[l * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
        }

        Tensor2D::new(result, [m, n])
    }

    /// Tensor addition
    pub fn add(&self, other: &Tensor2D) -> Result<Tensor2D> {
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

        Tensor2D::new(result, self.shape)
    }

    /// Tensor multiplication (element-wise)
    pub fn mul(&self, other: &Tensor2D) -> Result<Tensor2D> {
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

        Tensor2D::new(result, self.shape)
    }

    /// Softmax function (along the last dimension)
    pub fn softmax(&self) -> Result<Tensor2D> {
        let mut result = self.data.clone();
        let (rows, cols) = (self.shape[0], self.shape[1]);

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;

            // Find maximum value for numerical stability
            let max_val = result[start..end]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate exp and sum
            let mut sum = 0.0;
            for i in start..end {
                result[i] = (result[i] - max_val).exp();
                sum += result[i];
            }

            // Normalize
            for i in start..end {
                result[i] /= sum;
            }
        }

        Tensor2D::new(result, self.shape)
    }

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
