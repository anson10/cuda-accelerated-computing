# Module 2: The CUDA Thread Hierarchy

This module explores the scalability of CUDA. Unlike a CPU, which might handle 8 to 16 threads, a GPU can manage thousands of threads simultaneously by organizing them into a structured hierarchy.

## 1. Blocks and Threads
When you launch a kernel using `<<<B, T>>>`, you are defining two levels of parallelism:

* **Grid:** The entire collection of threads launched for a single kernel.
* **Block (B):** A group of threads that can cooperate with each other. Blocks are independent; the GPU can run them in any order.
* **Thread (T):** The smallest unit of execution. Each thread runs the exact same code (the Kernel), but handles different data.



## 2. Scalability: The "Why"
* **Total Threads:** $5 \text{ blocks} \times 5 \text{ threads/block} = 25 \text{ total threads}$.
* **Hardware Mapping:** If your GPU has many Multiprocessors (SMs), it might run all 5 blocks at once. If it's a smaller GPU, it might run 2 blocks, then another 2, then the final 1. 
* **Significance:** This allows the same CUDA code to run on a low-end laptop GPU and a high-end Data Center GPU without modification—it simply scales with the hardware.

## 3. Asynchronous Behavior Revisited
Even with 25 threads, the launch is still **asynchronous**.
* The CPU sends the instruction to the "Grid" and moves on immediately.
* Without `cudaDeviceSynchronize()`, the program would terminate before the GPU "printf" buffers can be flushed to the console.

## 4. Materials Science Context: Domain Decomposition
In Computational Materials Science, we often use this hierarchy for **Domain Decomposition**. 
* Each Block could represent a specific region or "cell" of a material lattice.
* Each Thread within that block calculates the forces for a single atom within that cell.
This structured approach is how simulations like LAMMPS achieve high performance on GPUs.