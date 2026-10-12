cmake_minimum_required(VERSION 3.20.0)
project(tensor_core LANGUAGES C CXX CUDA)

# RTX30xx = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# # 开启CUDA编译特性，WMMA需要，并且提升nvcc严格度
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可执行文件：源码hgemm_v1.cu
add_executable(hgemm_v1 hgemm_v1.cu)

# 链接库：cudart 运行时库，cublas 用于tester里的基准GEMM校验
target_link_libraries(hgemm_v1 PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_include_directories(hgemm_v1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)


add_executable(hgemm_v2 hgemm_v2.cu)

target_link_libraries(hgemm_v2 PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_include_directories(hgemm_v2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
