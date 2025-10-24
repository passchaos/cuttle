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
