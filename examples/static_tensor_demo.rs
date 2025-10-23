//! 静态维度张量演示示例
//!
//! 展示如何使用编译时确定维度的StaticTensor

use cuttle::tensor::{Tensor, Tensor1D, Tensor2D, Tensor3D};

fn main() {
    println!("=== 静态维度张量演示 ===");

    // 1D张量创建
    println!("\n1. 创建1D静态张量:");
    let data1d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor1d = Tensor1D::new(data1d, [5]).expect("创建1D张量失败");
    println!("维度数: {}", tensor1d.ndim());
    println!("形状: {:?}", tensor1d.shape());

    // 2D张量创建
    println!("\n2. 创建2D静态张量:");
    let data2d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor2d = Tensor2D::new(data2d, [2, 3]).expect("创建2D张量失败");
    println!("维度数: {}", tensor2d.ndim());
    println!("形状: {:?}", tensor2d.shape());

    // 3D张量创建
    println!("\n3. 创建3D静态张量:");
    let data3d = vec![1.0; 24]; // 2x3x4 = 24个元素
    let tensor3d = Tensor3D::new(data3d, [2, 3, 4]).expect("创建3D张量失败");
    println!("维度数: {}", tensor3d.ndim());
    println!("形状: {:?}", tensor3d.shape());

    // 零张量创建
    println!("\n4. 创建零张量:");
    let zeros2d = Tensor2D::zeros([3, 4]).expect("创建零张量失败");
    println!("零张量形状: {:?}", zeros2d.shape());

    // 激活函数演示
    println!("\n5. 激活函数演示:");
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let tensor = Tensor1D::new(data, [5]).expect("创建张量失败");
    println!("原始张量形状: {:?}", tensor.shape());

    let relu_result = tensor.relu();
    println!("ReLU结果数据: {:?}", relu_result.data());

    let scaled = tensor.scale(2.0);
    println!("缩放x2结果数据: {:?}", scaled.data());

    // 矩阵运算
    println!("\n6. 矩阵运算演示:");
    let matrix_a =
        Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).expect("创建矩阵A失败");
    let matrix_b =
        Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).expect("创建矩阵B失败");
    println!("矩阵A形状: {:?}", matrix_a.shape());
    println!("矩阵B形状: {:?}", matrix_b.shape());

    match matrix_a.matmul(&matrix_b) {
        Ok(result) => println!("矩阵乘法结果形状: {:?}", result.shape()),
        Err(e) => println!("矩阵乘法错误: {:?}", e),
    }

    // 编译时类型安全演示
    println!("\n7. 编译时类型安全:");
    let tensor_2d = Tensor2D::zeros([2, 3]).expect("创建2D零张量失败");
    let tensor_3d = Tensor3D::zeros([2, 3, 4]).expect("创建3D零张量失败");
    println!("2D张量维度: {}", tensor_2d.ndim());
    println!("3D张量维度: {}", tensor_3d.ndim());
    println!("Tensor1D、Tensor2D和Tensor3D是不同的类型");
    println!("这提供了编译时的维度检查和类型安全");

    // 错误处理演示
    println!("\n8. 错误处理演示:");
    let small_tensor = Tensor1D::new(vec![1.0, 2.0], [2]).expect("创建小张量失败");
    let large_tensor = Tensor1D::new(vec![1.0, 2.0, 3.0], [3]).expect("创建大张量失败");

    println!("小张量形状: {:?}", small_tensor.shape());
    println!("大张量形状: {:?}", large_tensor.shape());
    println!("静态维度系统在编译时就能捕获维度错误!");

    println!("\n=== 演示完成 ===");
}
