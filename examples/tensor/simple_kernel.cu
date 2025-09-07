#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// CUDA kernel to generate animated procedural texture
__global__ void simple_kernel(float4* output, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    unsigned int idx = y * width + x;
    
    // Debug: Print first few pixels (reduced output)
    if (idx < 5 && time < 0.1f) {
        printf("Pixel %d: x=%d, y=%d\n", idx, x, y);
    }
    
    // Normalized coordinates [0, 1]
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    
    // Center coordinates [-1, 1]
    float cx = u * 2.0f - 1.0f;
    float cy = v * 2.0f - 1.0f;
    
    // Generate animated pattern
    float dist = sqrtf(cx * cx + cy * cy);
    float angle = atan2f(cy, cx);
    
    // Animated ripple effect
    float ripple = sinf(dist * 10.0f - time * 2.0f) * 0.5f + 0.5f;
    
    // Color based on angle and time
    float r = sinf(angle + time) * 0.5f + 0.5f;
    float g = sinf(angle + time + 2.09f) * 0.5f + 0.5f;  // 2π/3
    float b = sinf(angle + time + 4.18f) * 0.5f + 0.5f;  // 4π/3
    
    // Modulate with ripple
    r *= ripple;
    g *= ripple;
    b *= ripple;
    
    // Add some radial gradient
    float fade = 1.0f - dist;
    fade = fmaxf(fade, 0.0f);
    
    float4 result = make_float4(r * fade, g * fade, b * fade, 1.0f);
    
    // Debug: Print first pixel result and center pixel (reduced output)
    if (idx == 0 && time < 0.1f) {
        printf("First pixel result: R=%.2f, G=%.2f, B=%.2f, A=%.2f\n", result.x, result.y, result.z, result.w);
    }
    
    // Check center pixel (should have color) - less frequent output
    unsigned int center_idx = (height/2) * width + (width/2);
    if (idx == center_idx && fmod(time, 1.0f) < 0.1f) {
        printf("Center pixel (%d,%d) result: R=%.2f, G=%.2f, B=%.2f, A=%.2f\n", 
               width/2, height/2, result.x, result.y, result.z, result.w);
    }
    
    output[idx] = result;
}

// Wrapper function called from C++
extern "C" void launch_simple_kernel(float4* d_data, unsigned int width, unsigned int height, float time)
{
    // Reduce debug prints - only print first few times
    if (time < 0.2f) {
        printf("CUDA kernel launch: width=%d, height=%d, time=%.2f\n", width, height, time);
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    if (time < 0.1f) {
        printf("Grid size: %d x %d, Block size: %d x %d\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    }
    
    simple_kernel<<<gridSize, blockSize>>>(d_data, width, height, time);
    
    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    // Wait for kernel to complete and check for errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    if (time < 0.1f) {
        printf("CUDA kernel completed successfully\n");
    }
}