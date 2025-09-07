#pragma once

// Constant buffer structures shared between CPU and GPU
struct TensorConstants
{
    uint32_t input_width;
    uint32_t input_height; 
    uint32_t input_channels;
    uint32_t batch_size;
    
    uint32_t output_width;
    uint32_t output_height;
    uint32_t output_channels;
    uint32_t layer_index;
    
    float quant_scale;
    float quant_zero_point;
    uint32_t quant_format; // 0 = E4M3, 1 = E5M2
    uint32_t activation_type; // 0 = None, 1 = ReLU, 2 = GELU
    
    // Convolution parameters
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t weight_offset;
    
    // Padding to 256 bytes
    uint32_t padding_[44];
};

// CUDA kernel launch parameters
struct CudaLaunchParams
{
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem_size;
};