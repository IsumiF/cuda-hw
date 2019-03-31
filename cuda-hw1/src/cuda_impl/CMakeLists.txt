add_executable(cuda_hw1
        main.cu
        cuda_helper.h
        cuda_helper.cpp
        )
target_include_directories(cuda_hw1
        PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
