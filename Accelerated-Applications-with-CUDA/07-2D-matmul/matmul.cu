#include <stdio.h>
#include <cuda_runtime.h>

#define N 64 

__global__ void matrixMulGPU(float *a, float *b, float *c, int size)
{
    // Calculate global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0;

    if (row < size && col < size)
    {
        // Dot product: multiply row of A by column of B
        for (int i = 0; i < size; ++i)
        {
            val += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = val;
    }
}

// CPU version for verification
void matrixMulCPU(float *a, float *b, float *c, int size)
{
    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            float val = 0;
            for (int i = 0; i < size; ++i)
            {
                val += a[row * size + i] * b[i * size + col];
            }
            c[row * size + col] = val;
        }
    }
}

int main()
{
    size_t size = N * N * sizeof(float);

    float *a, *b, *c_cpu, *c_gpu;

    // Allocate Unified Memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i)
    {
        a[i] = 1.0f; 
        b[i] = 2.0f;
        c_cpu[i] = 0.0f;
        c_gpu[i] = 0.0f;
    }

    // Define dim3 for execution configuration
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                          (N + threads_per_block.y - 1) / threads_per_block.y);

    // Launch GPU Kernel
    matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu, N);

    cudaDeviceSynchronize();

    // Run CPU version for comparison
    matrixMulCPU(a, b, c_cpu, N);

    // Verify results
    bool error = false;
    for (int i = 0; i < N * N; ++i)
    {
        if (c_cpu[i] != c_gpu[i])
        {
            printf("FAIL: Index %d: CPU (%f) != GPU (%f)\n", i, c_cpu[i], c_gpu[i]);
            error = true;
            break;
        }
    }

    if (!error)
    {
        printf("SUCCESS! Matrices multiplied correctly on the GPU.\n");
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);

    return 0;
}