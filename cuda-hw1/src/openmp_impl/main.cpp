#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>

namespace {

inline
float gause2D(int x, int y, float sigma) {
    return static_cast<float>(std::exp(-(x * x + y * y) / (2 * (sigma * sigma))) *
                              (1.0f / (sigma * std::sqrt(2 * M_PI))));
}

void gauseParallel(std::vector<float> &result, int width, int height, int shift) {
    int threadCount = omp_get_max_threads();
    int taskCount = width * height;
#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        while (threadId < taskCount) {
            int x = threadId % width - shift;
            int y = threadId / width - shift;
            result[threadId] = gause2D(x, y, static_cast<float>(shift) / 3);

            threadId += threadCount;
        }
    }
}

}

int main() {
    size_t s;
    std::cin >> s;

    size_t width = 6 * s + 1;
    size_t height = width;
    std::vector<float> arr(width * height);

    gauseParallel(arr, static_cast<int>(width), static_cast<int>(height), static_cast<int>(3 * s));

    for (auto it = arr.cbegin(); it != arr.cend(); ++it) {
        printf("%5.4f ", *it);
    }
    printf("\n");
    return 0;
}
