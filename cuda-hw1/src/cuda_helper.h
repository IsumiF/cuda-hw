#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

#if defined(__JETBRAINS_IDE__)

#define __global__
#define __device__
#define __host__

#define DEVICE_CALL(f, m, n) f

#else

#define DEVICE_CALL(f, m, n) f<<<(m), (n)>>>

#endif

namespace cudaHelper {

namespace internal {

void onFailure(cudaError_t err);

void throwErr(cudaError_t err);

}

inline
void checkErr(cudaError_t err) {
    if (err != cudaSuccess) {
        internal::onFailure(err);
    }
}

class CudaException : std::runtime_error {
public:
    CudaException(const std::string &whatArg, cudaError_t errCode)
            : std::runtime_error(whatArg), errCode(errCode) {}

private:
    cudaError_t errCode;
};

inline
void throwErr(cudaError_t err) {
    if (err != cudaSuccess) {
        internal::throwErr(err);
    }
}

/**
 * Get the unique ID of current thread, among all threads in the grid.
 * @return the unique id
 */
__device__ inline
size_t currentThreadId() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

}
