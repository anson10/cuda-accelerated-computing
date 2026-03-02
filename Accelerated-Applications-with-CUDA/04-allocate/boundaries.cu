#include <stdio.h>

// Kernel to square numbers
__global__ void squareArray(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensures threads 301-320 don't crash the app
    if (i < n) {
        a[i] = a[i] * a[i];
    }
}

int main() {
    const int N = 300; // Data size
    int *a;
    cudaMallocManaged(&a, N * sizeof(int));

    for (int i = 0; i < N; i++) a[i] = i;

    // We use 32 threads per block. 10 blocks = 320 threads.
    // 320 threads > 300 elements.
    squareArray<<<10, 32>>>(a, N);
    
    cudaDeviceSynchronize();
    printf("a[299] squared: %d\n", a[299]);

    cudaFree(a);
    return 0;
}