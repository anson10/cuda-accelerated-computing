#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

__global__ void visualizeAddition(float *result, float *a, float *b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i == 0 || i == N/2 || i == N - 1) {
        printf("Thread %d (Block %d) says: %0.0f + %0.0f = %0.0f\n", 
               i, blockIdx.x, a[i], b[i], result[i]);
    }
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 1 << 21; 
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
  visualizeAddition<<<1, 1>>>(c, a, b, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}