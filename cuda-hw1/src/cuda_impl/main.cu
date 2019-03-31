#include "cuda_helper.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>

__global__
void gause(float *arr, size_t width, size_t height, int shift, float sigma) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < width) {
        size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
        while (idy < height) {
            int x = static_cast<int>(idx) - shift;
            int y = static_cast<int>(idy) - shift;
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma)) / (sigma * sqrtf(2 * M_PI));
            arr[idx + idy * width] = value;

            idy += blockDim.y * gridDim.y;
        }
        idx += blockDim.x * gridDim.x;
    }
}

int64_t currentTimeMillis() {
    using namespace std::chrono;
    milliseconds ms = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
    );
    return ms.count();
}

int main() {
    size_t s;
    std::cin >> s;

    int64_t currentTime = currentTimeMillis();

    size_t width = 6 * s + 1;
    size_t height = width;
    std::vector<float> arr(width * height);
    float *arr_d;
    cudaHelper::throwErr(cudaMalloc(&arr_d, arr.size() * sizeof(float)));

    dim3 blocks(1, 1);
    dim3 threads(20, 20);
    DEVICE_CALL(gause, blocks, threads)(arr_d, width, height, static_cast<int>(3 * s), s);
    cudaMemcpy(arr.data(), arr_d, arr.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (auto it = arr.cbegin(); it != arr.cend(); ++it) {
        printf("%5.4f ", *it);
    }
    printf("\n");

    printf("%ld\n", currentTimeMillis() - currentTime);
    return 0;
}
