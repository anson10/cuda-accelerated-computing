#include <stdio.h>

__global__ void gridStrideCheck(int *data, int n) {
    // Starting position
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Total threads in the grid
    int stride = gridDim.x * blockDim.x;

    // Thread jumps by the 'stride' until the end of array
    for (int i = index; i < n; i += stride) {
        data[i] = 1; // Mark element as processed
    }
}

int main() {
    const int N = 100000;
    int *data;
    cudaMallocManaged(&data, N * sizeof(int));

    // Only launching 1024 threads for 100,000 elements
    gridStrideCheck<<<4, 256>>>(data, N);

    cudaDeviceSynchronize();
    
    int sum = 0;
    for (int i = 0; i < N; i++) sum += data[i];
    printf("Processed %d elements with only 1024 threads.\n", sum);

    cudaFree(data);
    return 0;
}