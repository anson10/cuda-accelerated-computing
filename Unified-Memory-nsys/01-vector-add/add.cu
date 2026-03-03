#include <stdio.h>
#include <cuda_runtime.h>

/**
 * GPU Kernel: Computes the sum of two vectors.
 * Uses a grid-stride loop to handle any vector size N.
 */
__global__ void addVectors(float *res, float *a, float *b, int n) {
    // Unique global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of threads in the grid (stride)
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop ensures all elements are processed
    for (int i = tid; i < n; i += stride) {
        res[i] = a[i] + b[i];
    }
}

int main() {
    // Large N (2^25) to ensure nsys captures the kernel
    const int N = 1 << 25; 
    size_t size = N * sizeof(float);

    float *a, *b, *res;

    // Allocate Unified Memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&res, size);

    // Initialize data on the CPU
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        res[i] = 0.0f;
    }

    // --- Execution Configuration ---
    // Using 256 threads per block is a common performance standard
    int threadsPerBlock = 256;
    // Calculate blocks to cover the entire vector
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(res, a, b, N);

    // Check for any immediate launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Launch Error: %s\n", cudaGetErrorString(launchErr));
    }

    // Synchronize: The CPU waits for the GPU to finish
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Sync Error: %s\n", cudaGetErrorString(syncErr));
    }

    // Verify a sample of the results
    if (res[0] == 3.0f && res[N-1] == 3.0f) {
        printf("Success! All values calculated correctly.\n");
    } else {
        printf("FAIL: Calculation error.\n");
    }

    // Free Unified Memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    return 0;
}