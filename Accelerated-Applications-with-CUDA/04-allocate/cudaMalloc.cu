#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to add a constant value to every element
__global__ void addOffset(float *data, float offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += offset;
    }
}

int main() {
    const int N = 1024;
    float *data;
    size_t size = N * sizeof(float);

    // Allocate Unified Memory
    cudaMallocManaged(&data, size);

    // Initialize on CPU
    for (int i = 0; i < N; i++) data[i] = 1.0f;

    // Launch with 4 blocks of 256 threads
    addOffset<<<4, 256>>>(data, 5.0f, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Verify result
    printf("Result at index 0: %f (Expected 6.0)\n", data[0]);

    // Free memory
    cudaFree(data);
    return 0;
}