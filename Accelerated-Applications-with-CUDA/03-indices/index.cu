#include <stdio.h>

/**
 * A kernel that demonstrates how threads identify themselves.
 */
__global__ void printIndices()
{
    // Local ID: Thread's position within its own block
    int local_id = threadIdx.x;

    // Block ID: Which group is this thread in?
    int block_id = blockIdx.x;

    // Grid Dimension: How many threads are in one block?
    int threads_per_block = blockDim.x;

    // GLOBAL ID: The unique index across the entire GPU
    int global_id = (block_id * threads_per_block) + local_id;

    printf("Block: %d | Local Thread: %d | GLOBAL ID: %d\n", 
           block_id, local_id, global_id);
}

int main()
{
    // Launching 2 blocks with 4 threads each
    // Total threads = 8
    int blocks = 2;
    int threads_per_block = 4;

    printf("Launching %d blocks with %d threads each...\n\n", blocks, threads_per_block);

    printIndices<<<blocks, threads_per_block>>>();

    // Wait for all threads to finish printing
    cudaDeviceSynchronize();

    printf("\nCalculation complete. Check the Global IDs above for uniqueness.\n");

    return 0;
}