# DX12 CUDA Interop Example

这是一个简单的DX12与CUDA互操作示例，展示了如何在CUDA中生成程序化内容并在DirectX 12中渲染。

## 功能特性

- **CUDA计算**: 使用CUDA kernel生成动画波纹效果
- **DirectX 12渲染**: 将CUDA生成的内容作为纹理在屏幕上显示
- **简化的数据流**: 通过CPU内存在CUDA和D3D12之间传输数据

## 架构说明

由于CUDA D3D12 interop API在不同版本中有变化，这个实现使用了简化的方法：

1. CUDA在GPU内存中生成内容
2. 数据通过CPU内存传输到D3D12纹理
3. D3D12渲染管线将纹理显示在全屏四边形上

## 编译要求

- CUDA SDK
- DirectX 12支持的显卡
- Visual Studio 2019或更高版本

## 未来扩展

- 实现真正的零拷贝GPU-GPU interop
- 添加更复杂的CUDA计算示例
- 支持FP8量化和tensor core操作

## 注意事项

这个示例主要用于演示基本的CUDA-D3D12集成模式。在生产环境中，建议使用真正的共享资源来避免CPU-GPU数据传输的性能开销。