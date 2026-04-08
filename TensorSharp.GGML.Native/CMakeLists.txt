cmake_minimum_required(VERSION 3.20)
project(GgmlOps LANGUAGES C CXX)
option(TENSORSHARP_GGML_NATIVE_BUILD_TESTS "Build TensorSharp GGML native regression tests" OFF)
option(TENSORSHARP_GGML_NATIVE_ENABLE_CUDA "Enable the ggml-cuda backend when building on Linux" OFF)

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "TENSORSHARP_GGML_NATIVE_ENABLE_CUDA is supported on Linux only.")
endif()

if (APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Threads REQUIRED)

set(GGML_VERSION "0.0.0" CACHE STRING "" FORCE)
set(GGML_VERSION_MAJOR 0 CACHE STRING "" FORCE)
set(GGML_SCHED_MAX_COPIES 4 CACHE STRING "" FORCE)
add_compile_definitions(GGML_VERSION=\"0.0.0\" GGML_COMMIT=\"unknown\")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_METAL_NDEBUG OFF CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_CANN OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN OFF CACHE BOOL "" FORCE)
set(GGML_ZENDNN OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

if (APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_NDEBUG ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FA_ALL_QUANTS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_VMM OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_PEER_COPY OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_DEBUG OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128 CACHE STRING "" FORCE)
    set(GGML_CUDA_COMPRESSION_MODE "speed" CACHE STRING "" FORCE)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src" ggml-build)

add_library(GgmlOps SHARED ggml_ops.cpp)
target_include_directories(GgmlOps PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src"
)
target_link_libraries(GgmlOps PRIVATE ggml)

if (APPLE)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_METAL=1)
endif()

set_target_properties(GgmlOps PROPERTIES
    OUTPUT_NAME GgmlOps
)

if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS)
    add_executable(GgmlOpsGetRowsQuantTest tests/get_rows_quant_test.cpp)
    target_include_directories(GgmlOpsGetRowsQuantTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsGetRowsQuantTest PRIVATE GgmlOps)

    add_executable(GgmlOpsAddmmVectorTest tests/addmm_vector_test.cpp)
    target_include_directories(GgmlOpsAddmmVectorTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsAddmmVectorTest PRIVATE GgmlOps)
endif()
