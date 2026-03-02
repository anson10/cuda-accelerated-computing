# Matrix Multiplication in 2D with CUDA

This program demonstrates how to perform Square Matrix Multiplication on the GPU. Unlike vector addition, matrix multiplication is a 2-dimensional problem that requires mapping threads to a 2D coordinate system (Rows and Columns).

---

## 1. The Mathematical Problem

In matrix multiplication , each element  is calculated by taking the dot product of the -th row of Matrix A and the -th column of Matrix B.

This is computationally expensive (complexity) on a CPU but highly efficient on a GPU because every single  can be calculated independently and simultaneously.

---

## 2. 2D Thread Indexing Logic

To solve a 2D problem, we use CUDA's `dim3` structure to launch a "grid of blocks" and "blocks of threads."

### Index Calculation

Each thread calculates its own global  and  using the following formulas:

* **Column (x-axis):** `int col = blockIdx.x * blockDim.x + threadIdx.x;`
* **Row (y-axis):** `int row = blockIdx.y * blockDim.y + threadIdx.y;`

### Memory Mapping (Row-Major Order)

Computers store 2D matrices as a flat 1D array in memory. To access the element at `(row, col)`, we use the formula:
`index = row * width + col`

---

## 3. The Kernel Operation

The GPU kernel follows these steps for every thread:

1. **Identify Location:** Determine the specific row and column this thread is responsible for.
2. **Boundary Check:** Ensure the thread is within the matrix limits (`row < size && col < size`).
3. **Accumulation:** Use a `for` loop to walk across the row of A and down the column of B, multiplying and summing the values into a local variable.
4. **Write Back:** Save the final sum into the correct position in Matrix C.

```text
Matrix A (Row)      Matrix B (Col)       Matrix C (Result)
[ 1 1 1 1 ]    x    [ 2 ]           =    [ (1*2 + 1*2 + 1*2 + 1*2) ]
                    [ 2 ]                [          = 8            ]
                    [ 2 ]
                    [ 2 ]

```

---

## 4. Execution Configuration

In the host code (CPU), we define how the GPU hardware should be sliced:

* **`threads_per_block(16, 16)`**: We create square tiles of 256 threads.
* **`number_of_blocks`**: We calculate how many 16x16 tiles are needed to cover the entire  matrix.

```cpp
dim3 threads_per_block(16, 16);
dim3 number_of_blocks((N + 15) / 16, (N + 15) / 16);

```

---

## 5. Summary of Workflow

1. **Allocation:** Use `cudaMallocManaged` to create memory accessible by both CPU and GPU.
2. **Initialization:** Fill Matrix A and B with data on the CPU.
3. **GPU Computation:** Launch the `matrixMulGPU` kernel. Thousands of threads calculate the dot products in parallel.
4. **CPU Computation:** The CPU runs a sequential version of the same math to create a "Ground Truth" result.
5. **Verification:** The program compares every single element of the GPU result against the CPU result. If they match perfectly, it prints **SUCCESS**.

---
