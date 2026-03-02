# 2D Heat Conduction Simulation with CUDA

This program implements a parallelized simulation of thermal conductivity across a 2D silver plate. It utilizes the Finite Difference Method to solve the heat equation, accelerating the process by mapping individual grid points to thousands of GPU threads.

---

## 1. Physical and Mathematical Logic

The simulation models how heat spreads through a material over time. The temperature update at any given point  depends on its current temperature and the temperatures of its four immediate neighbors.

### The Stencil Operation

In a discrete 2D grid, we approximate the second derivative of temperature (the Laplacian) by looking at the "stencil" of surrounding points:

* **Center:** 
* **North:**  (Row above)
* **South:**  (Row below)
* **East:**  (Column right)
* **West:**  (Column left)

---

## 2. GPU Acceleration Strategy

### 2D Thread Mapping

The simulation uses a 2D execution configuration (`dim3`) to match the geometry of the silver plate. We map the x-dimension of the grid to columns () and the y-dimension to rows ().

* **Column Index ():** `blockIdx.x * blockDim.x + threadIdx.x`
* **Row Index ():** `blockIdx.y * blockDim.y + threadIdx.y`

### Boundary Protection

Because the stencil calculation requires neighbors, threads assigned to the outermost edges of the grid are instructed to remain idle via an `if` statement. This prevents the kernel from attempting to read memory addresses that do not exist (out-of-bounds access).

---

## 3. Implementation Details

### Unified Memory

The program uses `cudaMallocManaged` to allocate memory that is automatically synchronized between the CPU and GPU. This allows the CPU to initialize the data and verify the final results while the GPU performs the heavy computation.

### The Pointer Swap (Ping-Pong)

To avoid race conditions, the simulation uses two separate arrays: `temp_in` and `temp_out`.

* Threads read from the "input" array and write the updated values to the "output" array.
* At the end of each time step, the pointers are swapped so the new temperatures become the input for the next step.

### Synchronization

`cudaDeviceSynchronize()` is called within the simulation loop. This ensures the GPU has completely finished writing all data for the current time step before the CPU swaps the pointers and launches the next iteration.

---

## 4. Performance & Hardware Mapping

* **Block Size:** A  thread block is used, providing 256 threads per block, which is a standard "sweet spot" for many NVIDIA architectures.
* **Indexing:** The `I2D` macro converts 2D coordinates into 1D linear memory indices using Row-Major Order, ensuring efficient memory access patterns.

---

