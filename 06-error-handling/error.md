# CUDA Error Handling: Technical Specification

In CUDA, error handling is non-trivial because of the **asynchronous execution** between the Host (CPU) and Device (GPU). Standard C++ `try-catch` blocks do not work for GPU code.

## 1. The `cudaError_t` Type

Almost all CUDA API functions return an integer status code of type `cudaError_t`.

* **`cudaSuccess` (0):** The operation completed without issues.
* **Anything else:** An error occurred (e.g., `cudaErrorInvalidConfiguration`, `cudaErrorMemoryAllocation`).

To convert these integers into human-readable messages, use:
`cudaGetErrorString(cudaError_t error)`

---

## 2. Synchronous vs. Asynchronous Errors

| Error Type | Source | Detection Method |
| --- | --- | --- |
| **Synchronous** | Memory allocation, API calls | Checking the return value of the function. |
| **Asynchronous** | Kernel execution, memory access | `cudaGetLastError()` or `cudaDeviceSynchronize()`. |

---

## 3. The `CHECK` Macro Pattern

To avoid writing redundant `if` statements, professional CUDA code uses a pre-processor macro. This captures the **File Name** and **Line Number** where the failure occurred.

```cpp
#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
        printf("Error: %s:%d, code: %d, reason: %s\n",                 \
               __FILE__, __LINE__, error, cudaGetErrorString(error));  \
        exit(1);                                                       \
    }                                                                  \
}

```

---

## 4. Catching Kernel Launch Failures

Kernels (`kernel<<<...>>>`) are "fire and forget" and do not return a value. To catch errors during a launch, you must check the state of the GPU immediately after the launch and after synchronization.

### Step 1: `cudaGetLastError()`

Checks if the **launch configuration** was valid.

* **Catches:** Exceeding 1024 threads per block, passing 0 blocks, or using too much shared memory.

### Step 2: `cudaDeviceSynchronize()`

Forces the CPU to wait and reports errors that happened **during** the kernel's run.

* **Catches:** "Illegal Memory Access" (going out of array bounds) or hardware timeouts.

---

## 5. Implementation Workflow

1. **Wrap** all `cudaMallocManaged` and `cudaFree` calls in the `CHECK` macro.
2. **Launch** your kernel.
3. **Call** `CHECK(cudaGetLastError())` to ensure the GPU accepted the task.
4. **Call** `CHECK(cudaDeviceSynchronize())` to ensure the GPU finished the task correctly.

---

