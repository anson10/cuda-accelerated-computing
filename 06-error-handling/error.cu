#include <stdio.h>

/*
 * MACRO: CHECK
 */
#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                  \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                       \
    }                                                                  \
}

/**
 * __global__ kernel: processElements
 * Now includes a printf so you can verify it's actually running.
 */
__global__ void processElements(int *a, int N) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride) 
  {
    a[i] *= 2;
    // Printing only for the first few elements to avoid flooding the terminal
    if(i < 5) {
        printf("GPU processing index %d: value is now %d\n", i, a[i]);
    }
  }
}

int main() 
{
  // Dataset size: ~2 million elements
  int N = 2 << 20;
  size_t size = N * sizeof(int);
  int *a;

  // Allocate Unified Memory
  CHECK(cudaMallocManaged(&a, size));

  // Initialize data on CPU
  for (int i = 0; i < N; ++i) 
  {
    a[i] = i;
  }

  // Hardware-safe Configuration (Max 1024 threads per block)
  size_t threads_per_block = 256; 
  size_t number_of_blocks = 32;

  printf("Launching kernel with %d blocks and %d threads per block...\n", (int)number_of_blocks, (int)threads_per_block);

  // Launch Kernel
  processElements<<<number_of_blocks, threads_per_block>>>(a, N);
  
  // Check for Launch Errors 
  CHECK(cudaGetLastError());
  
  CHECK(cudaDeviceSynchronize());

  printf("Kernel execution completed successfully.\n");

  // Clean up
  CHECK(cudaFree(a));
  
  return 0;
}