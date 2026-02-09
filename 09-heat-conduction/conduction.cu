#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Macro for 1D indexing from 2D coordinates
#define I2D(ni, i, j) ((j)*(ni)+(i))

// GPU Kernel: Assigns one thread per internal grid point
__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  // Global thread indices for column (i) and row (j)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Perform calculation only for internal points (skipping boundaries)
  if (i > 0 && i < ni - 1 && j > 0 && j < nj - 1) {
    int i00  = I2D(ni, i, j);
    int im10 = I2D(ni, i - 1, j); // West
    int ip10 = I2D(ni, i + 1, j); // East
    int i0m1 = I2D(ni, i, j - 1); // North
    int i0p1 = I2D(ni, i, j + 1); // South

    // Numerical approximation of the Laplacian
    float d2tdx2 = temp_in[im10] - 2.0f * temp_in[i00] + temp_in[ip10];
    float d2tdy2 = temp_in[i0m1] - 2.0f * temp_in[i00] + temp_in[i0p1];

    // Update the output temperature for the next time step
    temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      int i00 = I2D(ni, i, j);
      int im10 = I2D(ni, i - 1, j);
      int ip10 = I2D(ni, i + 1, j);
      int i0m1 = I2D(ni, i, j - 1);
      int i0p1 = I2D(ni, i, j + 1);

      float d2tdx2 = temp_in[im10] - 2.0f * temp_in[i00] + temp_in[ip10];
      float d2tdy2 = temp_in[i0m1] - 2.0f * temp_in[i00] + temp_in[i0p1];

      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  const int ni = 200;
  const int nj = 100;
  const int nstep = 500;
  float tfac = 8.418e-5;
  const int size = ni * nj * sizeof(float);

  float *temp1, *temp2, *temp1_ref, *temp2_ref, *temp_tmp;

  // Use Unified Memory to ensure GPU and CPU see the same data
  cudaMallocManaged(&temp1, size);
  cudaMallocManaged(&temp2, size);
  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);

  // Initialize both versions with identical data
  for (int i = 0; i < ni * nj; ++i) {
    temp1_ref[i] = temp1[i] = (float)rand() / (float)(RAND_MAX / 100.0f);
  }

  // CPU Reference Loop
  for (int s = 0; s < nstep; s++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
    temp_tmp = temp1_ref; temp1_ref = temp2_ref; temp2_ref = temp_tmp;
  }

  // GPU Configuration
  dim3 threads_per_block(16, 16);
  dim3 number_of_blocks((ni + 15) / 16, (nj + 15) / 16);

  // GPU Execution Loop
  for (int s = 0; s < nstep; s++) {
    step_kernel_mod<<<number_of_blocks, threads_per_block>>>(ni, nj, tfac, temp1, temp2);
    
    cudaDeviceSynchronize();

    temp_tmp = temp1; temp1 = temp2; temp2 = temp_tmp;
  }

  float maxError = 0;
  for (int i = 0; i < ni * nj; ++i) {
    float diff = fabsf(temp1[i] - temp1_ref[i]);
    if (diff > maxError) maxError = diff;
  }

  if (maxError > 0.0005f)
    printf("Problem! Max Error: %.5f\n", maxError);
  else
    printf("Success! Max Error: %.5f\n", maxError);

  free(temp1_ref); free(temp2_ref);
  cudaFree(temp1); cudaFree(temp2);

  return 0;
}