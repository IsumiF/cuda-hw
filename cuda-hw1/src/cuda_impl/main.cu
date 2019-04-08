/**
 * 姓名：丰泽霖
 * 文件说明：用CUDA计算二维高斯函数在某一定义域上的值
 */

#include "cuda_helper.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>

/**
 * Kernel function to compute gause function in a domain.
 */
__global__
void gause(float *arr, size_t width, size_t height, int shift, float sigma) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t size = width * height;
    while (idx < size) {
        int i = idx % width;
        int j = idx / height;
        float x = i - shift;
        float y = j - shift;
        float value = expf(-(x * x + y * y) / (2 * sigma * sigma)) / (sigma * sqrtf(2 * M_PI));
        arr[idx] = value;

        idx += gridDim.x * blockDim.x;
    }
}

/**
 * Get current time in millis (only used for profiling)
 */
int64_t currentTimeMillis() {
    using namespace std::chrono;
    milliseconds ms = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
    );
    return ms.count();
}

/**
 * Get the time required to execute a function
 */
template<typename F>
int64_t timeElapsed(F f) {
    int64_t start = currentTimeMillis();
    f();
    return currentTimeMillis() - start;
}

/**
 * Div x by y, rounding up.
 */
template<typename T>
T divup(T x, T y) {
    return (x + y - 1) / y;
}

int main() {
    size_t s;
    std::cin >> s;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // int64_t timeElapsed_ = timeElapsed([=]() {
    size_t width = 6 * s + 1;
    size_t height = width;
    std::vector<float> arr(width * height);
    float *arr_d;
    cudaHelper::throwErr(cudaMalloc(&arr_d, arr.size() * sizeof(float)));

    int gridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, gause, 0, deviceProp.maxThreadsPerBlock);

    DEVICE_CALL(gause, gridSize, blockSize)(arr_d, width, height, static_cast<int>(3 * s), s);
    cudaMemcpy(arr.data(), arr_d, arr.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (auto it = arr.cbegin(); it != arr.cend(); ++it) {
        printf("%5.4f ", *it);
    }
    printf("\n");
    cudaFree(arr_d);
    // });
    // printf("Time Elapsed: %ld\n", timeElapsed_);

    return 0;
}
