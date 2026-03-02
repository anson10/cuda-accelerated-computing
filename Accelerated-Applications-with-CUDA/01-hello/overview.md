# Module 1: NVCC and Kernel Execution

This module establishes the foundational syntax for offloading computation from the Host (CPU) to the Device (GPU).

## 1. The NVIDIA CUDA Compiler (NVCC)
The `nvcc` tool is not a single compiler but a compiler driver. It manages a complex multi-stage process to handle heterogeneous code (code containing both CPU and GPU instructions).

### Compilation Workflow
When you run `nvcc -o hello-gpu 01-hello/hello-gpu.cu -run`:
1. **Separation:** `nvcc` parses the file and separates the standard C++ (Host) from the `__global__` kernels (Device).
2. **Host Compilation:** The host code is passed to a general-purpose compiler like `gcc` or `cl`.
3. **Device Compilation:** The device code is compiled into PTX (Parallel Thread Execution), a low-level virtual machine and instruction set.
4. **Binary Generation:** The PTX is further compiled into a device-specific binary (cubin).
5. **Linking:** All parts are linked into a single executable.
6. **Execution:** The `-run` flag immediately executes the binary after a successful build.



## 2. Kernel Syntax & Execution Space
The `__global__` qualifier is an execution space qualifier. 

* **`__global__`**: Defines a function that is callable from the host but executed on the device. It must return `void` because kernels are launched asynchronously.
* **Kernel Launch:** The `<<<...>>>` syntax is the Execution Configuration.
    * The first parameter is the number of Blocks.
    * The second parameter is the number of Threads per Block.
    * In this lab, `<<<1, 1>>>` represents the simplest possible configuration: a single thread in a single block.

## 3. The Asynchronous Nature of CUDA
One of the most critical concepts in GPU programming is that kernel launches are non-blocking.

### Why `cudaDeviceSynchronize()` is Mandatory:
When the CPU reaches a kernel call, it instructs the GPU to start working and then immediately moves to the next instruction in `main()`.
* If `main()` finishes before the GPU, the program terminates, and you will see no output from the GPU.
* `cudaDeviceSynchronize()` forces the Host thread to wait (block) until the Device has finished all previously issued tasks.



## 4. Connection to Computational Materials Science
In materials simulations (like Molecular Dynamics or Monte Carlo methods), the CPU usually handles the setup of the atomic lattice, while the GPU handles the force calculations between thousands of atoms simultaneously. Mastering this "Host-Device" handshake is the first step toward accelerating your simulation research.