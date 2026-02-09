# Vector Addition on GPU with CUDA

This program demonstrates how to accelerate a standard vector addition operation using NVIDIA CUDA. By moving the computation from the CPU (Host) to the GPU (Device), we can perform millions of additions simultaneously.

---

## 1. Theoretical Background

### Parallelism: CPU vs. GPU

A **CPU** consists of a few cores optimized for **sequential** serial processing (doing one complex task at a time). A **GPU** consists of thousands of smaller, simpler cores designed for **parallel** processing (doing many simple tasks at once).

**Vector Addition** is a "data-parallel" problem. Since  does not depend on the result of , we can calculate every element of the array at the exact same time.

### CUDA Memory Model: Unified Memory

In older CUDA versions, developers had to manually copy data between CPU RAM and GPU VRAM. This program uses **Managed Memory** (`cudaMallocManaged`), which creates a pool of memory accessible by both the CPU and GPU. The CUDA driver handles the data migration automatically.

---

## 2. Program Structure & ASCII Visualization

### The Grid-Block-Thread Hierarchy

CUDA organizes threads into a hierarchy to manage the hardware efficiently:

* **Thread**: The smallest unit of execution.
* **Block**: A group of threads that can share data.
* **Grid**: A group of blocks launched for a single kernel.

```text
ARRAY N: [ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ... | N-1 ]
           ^   ^   ^   ^   ^   ^   ^   ^         ^
           |   |   |   |   |   |   |   |         |
THREADS:   T0  T1  T2  T3  T4  T5  T6  T7  ...   TN

[ BLOCK 0 ] [ BLOCK 1 ] [ BLOCK 2 ] ... [ BLOCK M ]
\_________________________________________________/
                      GRID

```

### The Grid-Stride Loop

In our kernel, we use a **Grid-Stride Loop**. This is a robust design pattern that allows our code to process an array of any size, even if it is larger than the number of threads available on the hardware.

```cpp
int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for(int i = index; i < N; i += stride) {
    result[i] = a[i] + b[i];
}

```

**Why use a stride?**
If  but our GPU only supports  threads at once, the `stride` ensures that after a thread finishes its first addition, it "steps" forward to the next available element.

---

## 3. Operation Workflow

| Step | Operation | Responsibility |
| --- | --- | --- |
| **1. Allocation** | `cudaMallocManaged` | Reserve memory on both Host and Device. |
| **2. Initialization** | `initWith` | CPU fills the arrays with initial values. |
| **3. Execution** | `addVectorsInto<<<...>>>` | CPU signals GPU to launch thousands of threads. |
| **4. Synchronization** | `cudaDeviceSynchronize` | CPU waits for the GPU to finish its "work" before reading results. |
| **5. Verification** | `checkElementsAre` | CPU checks the math to ensure . |
| **6. Cleanup** | `cudaFree` | Release the memory. |

---

## 4. Hardware Indexing Formulas

To map a specific thread to a specific data index, we use the following standard CUDA formulas:

1. **Block Offset**: `blockIdx.x * blockDim.x` (Moves the pointer to the start of the current block).
2. **Thread Offset**: `threadIdx.x` (Finds the specific thread within that block).
3. **Global Index**: `index = (blockIdx.x * blockDim.x) + threadIdx.x`.

---

## 5. How to Run and Profile

To compile the program:

```bash
nvcc -o add add.cu -run

```

To profile and verify GPU execution:

```bash
nsys profile --stats=true ./add

```
