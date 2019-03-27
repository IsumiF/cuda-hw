cmake_minimum_required(VERSION 3.13)
project(cuda_hw1 LANGUAGES CUDA C CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_hw1
        src/main.cu
        src/cuda_helper.h
        src/cuda_helper.cpp
        )
target_include_directories(cuda_hw1
        PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
