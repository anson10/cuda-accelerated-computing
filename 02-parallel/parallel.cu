#include <stdio.h>

/*
 * __global__ kernel: Each thread will execute this code.
   Even though they all print the same string, they are independent execution units.
 */

__global__ void firstParallel()
{
    printf("Parallel thread reporting\n");
}

int main()
{
    // 5 threads in 1 block (Total = 5)
    printf("Launching 1 Block with 5 Threads\n");
    firstParallel<<<1, 5>>>();
    cudaDeviceSynchronize();

    // 5 blocks with 5 threads each (Total = 25)
    printf("\nLaunching 5 Blocks with 5 Threads each\n");
    firstParallel<<<5, 5>>>();
    cudaDeviceSynchronize();

    printf("\nAll GPU threads have completed.\n");
    return 0;
}