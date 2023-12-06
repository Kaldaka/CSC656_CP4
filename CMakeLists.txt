cmake_minimum_required(VERSION 3.26)
project(CSC656_CP4)

set(CMAKE_CXX_STANDARD 14)

add_executable(CSC656_CP4
        vecadd_cpu.cpp
        vecadd_gpu_1t.cu
        vecadd_gpu_256t.cu
        vecadd_gpu_256t_mb.cu
        vecadd_gpu_256t_mb_prefetch.cu
)
