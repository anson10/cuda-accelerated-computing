#include <stdio.h>

/**
 * DEVICE CODE
 * Defined with __global__ to be a GPU kernel.
 * Returns void because it executes asynchronously.
 */

__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

/**
 * HOST CODE
 * Standard C++ function running on the CPU.
 */
void helloFromCPU() {
    printf("Hello from the CPU.\n");
}

int main() {
    // Launch Kernel: GPU prints first
    // Configuration: 1 Block, 1 Thread
    helloFromGPU<<<1, 1>>>();

    // Synchronize: Ensure GPU finishes before CPU proceeds
    cudaDeviceSynchronize();

    // CPU execution
    helloFromCPU();

    // Launch Kernel again: GPU prints last
    helloFromGPU<<<1, 1>>>();

    // Final Synchronize: Wait for last GPU output before exiting
    cudaDeviceSynchronize();

    return 0;
}