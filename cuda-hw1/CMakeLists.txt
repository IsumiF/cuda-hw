cmake_minimum_required(VERSION 3.13)
project(cuda_hw1 LANGUAGES CUDA C CXX)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src/cuda_impl)
add_subdirectory(src/openmp_impl)