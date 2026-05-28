cmake_minimum_required(VERSION 3.24)
project(cuda_kernel_lab LANGUAGES CXX CUDA)

# Detect the compute capability of the GPU in the build machine at configure
# time rather than hardcoding an architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm src/sgemm/sgemm.cu)
target_link_libraries(sgemm PRIVATE CUDA::cublas)
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
