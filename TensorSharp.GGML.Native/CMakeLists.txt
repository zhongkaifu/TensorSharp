cmake_minimum_required(VERSION 3.20)
project(GgmlOps LANGUAGES C CXX)
option(TENSORSHARP_GGML_NATIVE_BUILD_TESTS "Build TensorSharp GGML native regression tests" OFF)
option(TENSORSHARP_GGML_NATIVE_ENABLE_CUDA "Enable the ggml-cuda backend when building on Windows or Linux" OFF)
option(TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN "Enable the ggml-vulkan backend when building on Windows or Linux" OFF)

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND APPLE)
    message(FATAL_ERROR "TENSORSHARP_GGML_NATIVE_ENABLE_CUDA is supported on Windows and Linux only.")
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN AND APPLE)
    message(FATAL_ERROR "TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN is supported on Windows and Linux only (Metal is the GPU backend on macOS).")
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

if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>)
endif()

find_package(Threads REQUIRED)

set(GGML_SCHED_MAX_COPIES 4 CACHE STRING "" FORCE)

# Upstream ggml-org/ggml derives GGML_VERSION/GGML_COMMIT in its root CMakeLists and
# injects them onto its own targets, so we no longer set them here.

# Do not build ggml's bundled tests/examples when consumed as a subdirectory.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

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

# Turn on ggml's ccache integration only when ccache/sccache is actually on PATH.
# A cached clean rebuild of the CUDA backend drops from minutes to seconds, and
# gating on availability avoids ggml's "ccache not found" warning on machines that
# do not have it. Install ccache (e.g. `apt install ccache`) to opt in.
find_program(TSG_CCACHE_PROGRAM NAMES ccache sccache)
if (TSG_CCACHE_PROGRAM)
    set(GGML_CCACHE ON CACHE BOOL "" FORCE)
else()
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
endif()

if (APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_NDEBUG ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    # GgmlOps itself compiles a CUDA source (the on-device diffusion sampler), so it needs the
    # CUDA language enabled. CMAKE_CUDA_ARCHITECTURES is normally supplied by build-windows.ps1
    # (detected from the local GPU); fall back to a broad default if the caller did not set it.
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FA_ALL_QUANTS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_VMM OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_PEER_COPY OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_DEBUG OFF CACHE BOOL "" FORCE)
    # NCCL is only used for multi-GPU collective communication and defaults to ON
    # upstream, which makes the build emit "Warning: NCCL not found ..." on the
    # common single-GPU / no-NCCL setup. TensorSharp does not depend on NCCL, so
    # disable the probe to keep the build quiet (multi-GPU still works without it).
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128 CACHE STRING "" FORCE)
    set(GGML_CUDA_COMPRESSION_MODE "speed" CACHE STRING "" FORCE)
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
    # ggml-vulkan needs the Vulkan headers + loader import library and the glslc
    # shader compiler (find_package(Vulkan COMPONENTS glslc) inside ggml), plus
    # the SPIRV-Headers CMake package. With a LunarG Vulkan SDK installed
    # (VULKAN_SDK env var) all of these are found automatically; otherwise
    # build-windows.ps1 provisions a portable toolchain under
    # ExternalProjects/vulkan-toolchain (see eng/fetch-vulkan-toolchain.ps1) and
    # passes Vulkan_INCLUDE_DIR / Vulkan_LIBRARY / Vulkan_GLSLC_EXECUTABLE /
    # SPIRV-Headers_DIR explicitly.
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# The binary-dir name is deliberately terse ("g", not "ggml-build"): ggml-vulkan's
# vulkan-shaders-gen ExternalProject nests
#   <bindir>/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build
# under it, and MSBuild's file tracker (.tlog) breaks CMake try-compiles once that
# inner binary dir exceeds ~137 characters ("CL.exe could not be run:
# DirectoryNotFoundException ... cmTC_*.tlog"). Every character saved here buys
# headroom for repos cloned into deeper paths.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml" g)

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN AND TARGET ggml-vulkan)
    # ggml-vulkan.cpp includes <spirv/unified1/spirv.hpp>, but upstream's CMake
    # only *checks* that the SPIRV-Headers package exists — it assumes the header
    # is reachable through the Vulkan SDK's Include directory (LunarG layout) or
    # a system include path (Linux distro packages). With the portable toolchain
    # (bare Vulkan-Headers checkout) neither holds, so link the imported
    # SPIRV-Headers interface target onto ggml-vulkan to add its include dir.
    # Scoped to the one target (not include_directories) so the CUDA/CPU targets'
    # command lines — and therefore their incremental-build state — are untouched.
    find_package(SPIRV-Headers CONFIG QUIET)
    if (TARGET SPIRV-Headers::SPIRV-Headers)
        target_link_libraries(ggml-vulkan PRIVATE SPIRV-Headers::SPIRV-Headers)
    endif()
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
    # Guard against the MSBuild path-length limit described above: the deepest
    # path the vulkan-shaders-gen ExternalProject configure creates is
    #   <bindir>/g/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/CMakeFiles/CMakeScratch/TryCompile-XXXXXX/cmTC_XXXXX.dir/Debug/cmTC_XXXXX.tlog/...
    # which fails opaquely once the inner binary dir passes ~137 characters.
    # 74 = "/g/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build".
    string(LENGTH "${CMAKE_BINARY_DIR}" TSG_BINARY_DIR_LEN)
    math(EXPR TSG_VK_SHADERS_GEN_DIR_LEN "${TSG_BINARY_DIR_LEN} + 74")
    if (CMAKE_GENERATOR MATCHES "Visual Studio" AND TSG_VK_SHADERS_GEN_DIR_LEN GREATER 137)
        message(FATAL_ERROR
            "TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN: the build directory path is too long "
            "(${TSG_BINARY_DIR_LEN} chars). ggml-vulkan's shader-generator sub-build would "
            "exceed the MSBuild file-tracker path limit and fail with a cryptic "
            "'CL.exe could not be run: DirectoryNotFoundException (...cmTC_*.tlog)' error. "
            "Clone the repository into a shorter path (e.g. C:/Works/TensorSharp) or use the Ninja generator.")
    endif()
endif()

set(TSG_GGMLOPS_SOURCES
    ggml_ops_core.cpp
    ggml_ops_matmul.cpp
    ggml_ops_elementwise.cpp
    ggml_ops_norm_attn.cpp
    ggml_ops_training.cpp
    ggml_ops_fused.cpp
    ggml_ops_transformer.cpp
    ggml_ops_diffusion.cpp
    ggml_ops_qwen_image.cpp
    ggml_ops_gated_delta_net.cpp
    ggml_ops_qwen35_recurrent_prefill.cpp
    ggml_ops_mamba2.cpp
    ggml_ops_moe.cpp
    ggml_ops_paged_attention.cpp
)
if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_diffusion_sample.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_mask.cu)
endif()

add_library(GgmlOps SHARED ${TSG_GGMLOPS_SOURCES})

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_VULKAN=1)
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_CUDA=1)
    # Fast intrinsic transcendentals (tanhf/expf) for the diffusion sampler kernel — the accurate
    # library tanhf is ~100x slower on-device and the sampler only needs FP-reduction-level accuracy.
    target_compile_options(GgmlOps PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
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
