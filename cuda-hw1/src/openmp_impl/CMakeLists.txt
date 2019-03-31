
find_package(OpenMP REQUIRED)

add_executable(openmp_hw1
        main.cpp
        )
target_link_libraries(openmp_hw1 PUBLIC OpenMP::OpenMP_CXX)
