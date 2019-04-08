/**
 * 姓名：丰泽霖
 * 文件说明：CUDA辅助函数的实现
 */

#include "cuda_helper.h"
#include <cstdlib>
#include <iostream>

namespace cudaHelper {

namespace internal {

void onFailure(cudaError_t err) {
    std::cerr << __FILE__ << " " << __LINE__ << ":"
              << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
}

void throwErr(cudaError_t err) {
    throw CudaException(cudaGetErrorString(err), err);
}

}

}
