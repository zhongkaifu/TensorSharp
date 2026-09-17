cmake_minimum_required(VERSION 3.20)
project(GgmlOps LANGUAGES C CXX)
option(TENSORSHARP_GGML_NATIVE_BUILD_TESTS "Build TensorSharp GGML native regression tests" OFF)
option(TENSORSHARP_GGML_NATIVE_ENABLE_CUDA "Enable the ggml-cuda backend when building on Windows or Linux" OFF)
option(TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN "Enable the ggml-vulkan backend when building on Windows or Linux" OFF)
# Cache marker maintained by build-windows.ps1 / build-linux.sh: ON when the
# ENABLE_VULKAN value came from an explicit user choice (flag or env var) rather
# than auto-detection, so later script runs keep honoring the explicit choice
# while auto-detected defaults are re-evaluated. Not read by the build itself.
option(TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT "The Vulkan on/off choice was made explicitly (build-script cache marker)" OFF)
# Metal is normally forced ON for every Apple build; the iOS *simulator* slice must
# be able to opt out. The simulator's Metal device only advertises the Apple2
# feature set, so ggml-metal's simdgroup matmul / flash-attention kernels would
# all be reported unsupported and every op would bounce to the CPU through split
# buffers - a clean CPU-only build is strictly better there. Everything Metal-only
# in this tree (tsg_metal_*.m/.mm, the MPSGraph convolution, TSG_GGML_USE_METAL)
# hangs off this one switch, which is why native sources must gate Metal code on
# TSG_GGML_USE_METAL rather than __APPLE__: the simulator is the one Apple build
# where the two differ.
option(TENSORSHARP_GGML_NATIVE_ENABLE_METAL "Enable the ggml-metal backend on Apple platforms" ON)
# iOS/iPadOS builds produce a static archive (a bare .dylib is not shippable in an
# app bundle); build-ios.sh merges the slices into GgmlOps.xcframework.
option(TENSORSHARP_GGML_NATIVE_IOS "Build a static GgmlOps for iOS/iPadOS (device or simulator)" OFF)

# This library is 64-bit only, and a 32-bit configure otherwise fails hundreds of
# lines into the compile with errors that name neither the cause nor the word
# "32-bit": ggml's Vulkan backend leaves VK_USE_64_BIT_PTR_DEFINES at 0 on a
# 32-bit target, so vulkan.hpp marks every handle's conversion operator
# `explicit` and ordinary uses of vk::Buffer stop compiling ("no operator found
# which takes a left-hand operand of type 'std::basic_ostream'",
# "'vk::CommandBuffer::copyBuffer': no matching overloaded function"). ggml-cuda
# is likewise unsupported on 32-bit hosts.
#
# This is easy to hit by accident on Windows: the "Developer PowerShell for VS"
# and "Developer Command Prompt" shortcuts default to the x86 toolset, so a
# build started from one used to inherit a 32-bit cl.exe. build-windows.ps1 now
# imports the x64 environment over such a prompt, and this check is the backstop
# for every other entry point (a hand-run cmake, -A Win32, a cross toolchain
# file). See https://github.com/zhongkaifu/TensorSharp/discussions/130
if (CMAKE_SIZEOF_VOID_P AND CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR
        "TensorSharp.GGML.Native requires a 64-bit toolchain, but this one targets "
        "${CMAKE_SIZEOF_VOID_P}-byte pointers (32-bit). On Windows, run build-windows.ps1 from a plain "
        "PowerShell so it can set up the x64 MSVC environment itself, or from an \"x64 Native Tools\" "
        "prompt - the default \"Developer PowerShell for VS\" targets x86. With a Visual Studio "
        "generator, pass -A x64.")
endif()

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

if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_NDEBUG ON CACHE BOOL "" FORCE)
    # Embedded shader source is required on iOS: the non-embed path shells out to
    # `xcrun -sdk macosx metal` at build time, while the embed path .incbin's the
    # .metal source and compiles it at run time via newLibraryWithSource.
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    # An MSVC newer than the host-compiler range the installed CUDA toolkit
    # accepts makes nvcc abort in host_config.h with "unsupported Microsoft
    # Visual Studio version!", which surfaces here as a failed CUDA compiler
    # identification ("No CMAKE_CUDA_COMPILER could be found" / "not able to
    # compile a simple test program"). Pass -allow-unsupported-compiler so nvcc
    # proceeds; the flag must be in CMAKE_CUDA_FLAGS *before*
    # enable_language(CUDA) because CMake uses those flags during compiler
    # identification too.
    #
    # Applied on every MSVC host rather than above a fixed MSVC_VERSION: nvcc's
    # ceiling moves with the toolkit (CUDA 12.6 stops at MSVC 19.49 - the
    # VS 2026 / 14.5x case from discussion #130 - but CUDA 12.4 already stops at
    # 19.39, so VS 2022 17.10+ hits the identical error), and the flag is simply
    # a no-op when the toolset is one nvcc already supports.
    if (MSVC AND NOT "${CMAKE_CUDA_FLAGS}" MATCHES "allow-unsupported-compiler")
        string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
        message(STATUS "MSVC ${MSVC_VERSION}: adding -allow-unsupported-compiler to CMAKE_CUDA_FLAGS so nvcc accepts this host toolset (no-op when it already does)")
    endif()
    # GgmlOps itself compiles a CUDA source (the on-device diffusion sampler), so it needs the
    # CUDA language enabled. CMAKE_CUDA_ARCHITECTURES is normally supplied by build-windows.ps1
    # (detected from the local GPU); fall back to a broad default if the caller did not set it.
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "" FORCE)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
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
    # NCCL powers the cross-GPU AllReduce on the tensor-parallel path
    # (ggml-cuda's comm backend prefers it over the P2P pipeline and the
    # butterfly fallback). Upstream defaults it ON and warns loudly when the
    # library is missing, which is noise on the common single-GPU box — so probe
    # for it here and only enable it when it is actually installed.
    # TENSORSHARP_GGML_NATIVE_ENABLE_NCCL=ON/OFF forces the decision.
    if (DEFINED TENSORSHARP_GGML_NATIVE_ENABLE_NCCL)
        set(TSG_USE_NCCL ${TENSORSHARP_GGML_NATIVE_ENABLE_NCCL})
    else()
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/cmake")
        find_package(NCCL QUIET)
        if (NCCL_FOUND)
            set(TSG_USE_NCCL ON)
            message(STATUS "NCCL found: enabling multi-GPU collective AllReduce for tensor parallelism")
        else()
            set(TSG_USE_NCCL OFF)
            message(STATUS "NCCL not found: tensor-parallel AllReduce will use the P2P/butterfly fallback")
        endif()
    endif()
    set(GGML_CUDA_NCCL ${TSG_USE_NCCL} CACHE BOOL "" FORCE)
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
# Consume upstream sources unchanged. Architecture-specific kernels and
# precision requirements belong to TensorSharp's own native implementation.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml" g)

# Eight CUDA devices each have a companion backend for architecture-specific
# kernels, followed by the host backend: 17 scheduler entries. Upstream's
# default of 16 aborts the first forward pass on an eight-GPU DeepSeek load.
target_compile_definitions(ggml-base PRIVATE GGML_SCHED_MAX_BACKENDS=32)

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
    ggml_ops_attention_alloc.cpp
    ggml_ops_attention_precision.cpp
    ggml_ops_flash_attn_guard.cpp
    ggml_ops_matmul.cpp
    ggml_ops_matmul_precision.cpp
    ggml_ops_q8_precision.cpp
    ggml_ops_elementwise.cpp
    ggml_ops_norm_attn.cpp
    ggml_ops_training.cpp
    ggml_ops_fused.cpp
    ggml_ops_transformer.cpp
    ggml_ops_transformer_prefill.cpp
    ggml_ops_qwen3_decode.cpp
    ggml_ops_qwen3_prefill.cpp
    ggml_ops_gptoss_decode.cpp
    ggml_ops_gptoss_batched.cpp
    ggml_ops_qwen35_batched_arena.cpp
    ggml_ops_qwen4exp.cpp
    ggml_ops_qwen4exp_mtp.cpp
    ggml_ops_qwen4exp_arena.cpp
    ggml_ops_gptoss_prefill.cpp
    ggml_ops_qwen35_decode.cpp
    ggml_ops_qwen35_verify.cpp
    ggml_ops_gemma4_decode.cpp
    ggml_ops_gemma4_batched.cpp
    ggml_ops_gemma4_verify.cpp
    ggml_ops_gemma4_moe.cpp
    ggml_ops_muse_glimmer.cpp
    ggml_ops_muse_glimmer_vision.cpp
    ggml_ops_dflash.cpp
    ggml_ops_diffusion.cpp
    ggml_ops_qwen_image.cpp
    ggml_ops_wan.cpp
    ggml_ops_minimax_h3.cpp
    ggml_ops_gated_delta_net.cpp
    ggml_ops_qwen35_recurrent_prefill.cpp
    ggml_ops_qwen35_gdn_tp.cpp
    ggml_ops_mamba2.cpp
    ggml_ops_moe.cpp
    ggml_ops_deepseek4.cpp
    ggml_ops_deepseek41_tp.cpp
    ggml_ops_deepseek41_vision.cpp
    ggml_ops_embeddings.cpp
    ggml_ops_glm_dsa.cpp
    ggml_ops_dsv4_fused_cpu.cpp
    ggml_ops_paged_attention.cpp
    ggml_ops_paged_kv_pool.cpp
    ggml_ops_tensor_parallel.cpp
)
if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
    # Raises the process-wide default MSL version so ggml's Metal 4 tensor-API probe can compile
    # under the .NET apphost's stale recorded SDK. See the file header for the full story.
    list(APPEND TSG_GGMLOPS_SOURCES tsg_metal_msl_default.m)
    # Wan VAE convolutions on MPSGraph (see tsg_metal_mps_conv.mm).
    list(APPEND TSG_GGMLOPS_SOURCES tsg_metal_mps_conv.mm)
endif()
if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_diffusion_sample.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_mask.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_dsv4_fused.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_matmul_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_attention_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_q8_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_tp_probe.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_host_pin.cu)
    # Wan / Qwen-Image VAE convolutions on cuDNN (see tsg_cuda_cudnn_conv.cu).
    # Optional: the VAE falls back to ggml's im2col+GEMM when cuDNN is absent.
    #
    # Only the HEADERS are needed to build. The library itself is resolved with
    # LoadLibrary/dlopen at run time, so nothing is linked and nothing is copied
    # next to the output (the Windows redistributable is ~1 GB of DLLs). That also
    # means a binary built here still loads on a machine with no cuDNN at all.
    #
    # eng/fetch-cudnn.{ps1,sh} provisions ExternalProjects/cudnn from NVIDIA's public
    # redist channel; a system install, CUDA_PATH or TS_CUDNN_DIR is used ahead of it.
    set(TSG_CUDNN_PROVISIONED "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/cudnn")
    find_path(TSG_CUDNN_INCLUDE_DIR cudnn.h
              HINTS "$ENV{TS_CUDNN_DIR}/include" "$ENV{CUDNN_DIR}/include"
                    "${TSG_CUDNN_PROVISIONED}/include"
                    ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/cuda/include)
    if (TSG_CUDNN_INCLUDE_DIR)
        message(STATUS "TensorSharp: cuDNN headers at ${TSG_CUDNN_INCLUDE_DIR}; VAE convolutions will use cuDNN when its runtime is present")
        list(APPEND TSG_GGMLOPS_SOURCES tsg_cuda_cudnn_conv.cu)
        set(TSG_HAVE_CUDNN ON)
        # Where the runtime should look first. Points at the provisioned copy when
        # that is what supplied the headers; TS_CUDNN_DIR still overrides at run time.
        get_filename_component(TSG_CUDNN_ROOT "${TSG_CUDNN_INCLUDE_DIR}" DIRECTORY)
        set(TSG_CUDNN_RUNTIME_HINT "${TSG_CUDNN_ROOT}")
    else()
        message(STATUS "TensorSharp: cuDNN headers not found; VAE convolutions stay on ggml im2col+GEMM "
                       "(run eng/fetch-cudnn to provision, or set TENSORSHARP_CUDNN=OFF to stop looking)")
    endif()
endif()

if (TENSORSHARP_GGML_NATIVE_IOS)
    add_library(GgmlOps STATIC ${TSG_GGMLOPS_SOURCES})
else()
    add_library(GgmlOps SHARED ${TSG_GGMLOPS_SOURCES})
endif()

if (WIN32)
    # Re-export the statically-linked ggml C API so the managed layer
    # (TensorSharp.Backends.GGML/Interop/GgmlApi.cs) can P/Invoke it directly.
    # On Linux/macOS the symbols are already visible (default ELF/Mach-O
    # visibility); Windows needs the explicit .def export list.
    #
    # A .def entry forces the linker to resolve the symbol, so the CUDA/Vulkan
    # re-exports must be stripped when the corresponding backend is not being
    # built or the link fails with LNK2001 unresolved externals. The managed
    # layer probes these entry points at run time and tolerates their absence,
    # matching the Linux/macOS builds where disabled-backend symbols are simply
    # not present.
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ggml_api_exports.def" TSG_DEF_CONTENT)
    if (NOT TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        string(REGEX REPLACE "[ \t]+ggml_backend_(cuda_[A-Za-z0-9_]+|is_cuda)\r?\n" "" TSG_DEF_CONTENT "${TSG_DEF_CONTENT}")
    endif()
    if (NOT TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
        string(REGEX REPLACE "[ \t]+ggml_backend_(vk_[A-Za-z0-9_]+|is_vk)\r?\n" "" TSG_DEF_CONTENT "${TSG_DEF_CONTENT}")
    endif()
    set(TSG_DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/ggml_api_exports.def")
    # Only rewrite on change so incremental builds don't relink GgmlOps.
    set(TSG_DEF_OLD_CONTENT "")
    if (EXISTS "${TSG_DEF_FILE}")
        file(READ "${TSG_DEF_FILE}" TSG_DEF_OLD_CONTENT)
    endif()
    if (NOT TSG_DEF_CONTENT STREQUAL TSG_DEF_OLD_CONTENT)
        file(WRITE "${TSG_DEF_FILE}" "${TSG_DEF_CONTENT}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ggml_api_exports.def)
    target_sources(GgmlOps PRIVATE "${TSG_DEF_FILE}")
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_VULKAN=1)
endif()

if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_CUDA=1)
    target_link_libraries(GgmlOps PRIVATE CUDA::cublas)
    if (TSG_HAVE_CUDNN)
        target_include_directories(GgmlOps PRIVATE ${TSG_CUDNN_INCLUDE_DIR})
        # Deliberately NOT linked -- resolved with LoadLibrary/dlopen at run time.
        target_compile_definitions(GgmlOps PRIVATE TSG_HAVE_CUDNN=1)
        if (TSG_CUDNN_RUNTIME_HINT)
            target_compile_definitions(GgmlOps PRIVATE
                "TSG_CUDNN_RUNTIME_DIR=\"${TSG_CUDNN_RUNTIME_HINT}\"")
        endif()
    endif()
    # Fast intrinsic transcendentals (tanhf/expf) for the diffusion sampler kernel — the accurate
    # library tanhf is ~100x slower on-device and the sampler only needs FP-reduction-level accuracy.
    # Scope fast math to the existing kernels: precision attention needs the
    # normal expf implementation as well as F32 subnormals and IEEE division.
    foreach(TSG_CUDA_SOURCE IN LISTS TSG_GGMLOPS_SOURCES)
        if (TSG_CUDA_SOURCE MATCHES "\\.cu$" AND
            NOT TSG_CUDA_SOURCE MATCHES "ggml_ops_(matmul|attention|q8)_precision\\.cu$")
            set_property(SOURCE ${TSG_CUDA_SOURCE} APPEND PROPERTY COMPILE_OPTIONS --use_fast_math)
        endif()
    endforeach()
    set_source_files_properties(ggml_ops_matmul_precision.cu ggml_ops_attention_precision.cu
        ggml_ops_q8_precision.cu PROPERTIES
        COMPILE_OPTIONS "--ftz=false;--prec-div=true;--prec-sqrt=true")
endif()
target_include_directories(GgmlOps PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src"
)
target_link_libraries(GgmlOps PRIVATE ggml)

if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_METAL=1)

    # tsg_metal_msl_default.m talks to Metal and the Objective-C runtime directly.
    find_library(TSG_FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(TSG_METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(TSG_MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    find_library(TSG_MPS_FRAMEWORK      MetalPerformanceShaders      REQUIRED)
    target_link_libraries(GgmlOps PRIVATE ${TSG_FOUNDATION_LIBRARY} ${TSG_METAL_FRAMEWORK}
                                          ${TSG_MPSGRAPH_FRAMEWORK} ${TSG_MPS_FRAMEWORK})
endif()

set_target_properties(GgmlOps PROPERTIES
    OUTPUT_NAME GgmlOps
)

if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS)
    set_property(SOURCE ggml_ops_qwen4exp.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    add_executable(GgmlOpsQwen4ExpQsaPlanTest tests/qwen4exp_qsa_plan_test.cpp)
    target_include_directories(GgmlOpsQwen4ExpQsaPlanTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (MSVC)
        target_compile_options(GgmlOpsQwen4ExpQsaPlanTest PRIVATE /O2 /EHsc)
    endif()
    add_test(NAME qwen4exp-qsa-position-plan COMMAND GgmlOpsQwen4ExpQsaPlanTest)
    set_tests_properties(qwen4exp-qsa-position-plan PROPERTIES TIMEOUT 30)
    set_property(SOURCE ggml_ops_glm_dsa.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    set_property(SOURCE ggml_ops_deepseek4.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    set_property(SOURCE ggml_ops_deepseek41_tp.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    enable_testing()
    add_executable(GgmlOpsDsv41EngramIoTest tests/dsv41_engram_io_test.cpp)
    target_link_libraries(GgmlOpsDsv41EngramIoTest PRIVATE Threads::Threads)
    add_test(NAME deepseek41-engram-parallel-io COMMAND GgmlOpsDsv41EngramIoTest)
    add_executable(GgmlOpsDsv41EngramAdviceTest tests/dsv41_engram_advice_test.cpp)
    add_test(NAME deepseek41-engram-mapped-advice COMMAND GgmlOpsDsv41EngramAdviceTest)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(GgmlOpsDsv41EngramIoBench tests/dsv41_engram_io_bench.cpp)
        target_link_libraries(GgmlOpsDsv41EngramIoBench PRIVATE Threads::Threads)
        # Run explicitly with a scratch directory on the filesystem being measured.
    endif()
    add_executable(GgmlOpsDsv41WorkersTest tests/dsv41_workers_test.cpp)
    target_include_directories(GgmlOpsDsv41WorkersTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsDsv41WorkersTest PRIVATE Threads::Threads)
    add_test(NAME deepseek41-tp-worker-lifetime COMMAND GgmlOpsDsv41WorkersTest)
    set_tests_properties(deepseek41-tp-worker-lifetime PROPERTIES TIMEOUT 30)

    add_executable(GgmlOpsDsv41TpTest tests/dsv41_tp_test.cpp ggml_ops_deepseek41_tp.cpp
        ggml_ops_matmul_precision.cpp)
    target_compile_definitions(GgmlOpsDsv41TpTest PRIVATE TSG_GGML_TEST_HOOKS)
    target_include_directories(GgmlOpsDsv41TpTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsDsv41TpTest PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME deepseek41-moe-tensor-parallel COMMAND GgmlOpsDsv41TpTest)
    add_test(NAME deepseek41-tp-failure-recovery COMMAND GgmlOpsDsv41TpTest --failure-only)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND NOT WIN32)
        target_compile_definitions(GgmlOpsDsv41TpTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_sources(GgmlOpsDsv41TpTest PRIVATE ggml_ops_dsv4_fused.cu ggml_ops_matmul_precision.cu
            ggml_ops_attention_precision.cu ggml_ops_q8_precision.cu)
        target_include_directories(GgmlOpsDsv41TpTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsDsv41TpTest PRIVATE ggml-cuda CUDA::cublas)
        add_test(NAME deepseek41-quantized-strip-projections-cuda
                 COMMAND GgmlOpsDsv41TpTest --cuda 1 --quant-strip-only)
        set_tests_properties(deepseek41-quantized-strip-projections-cuda PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 600)
        # Multi-device tests are explicit: a one-GPU CI lane must not claim
        # that its skipped multi-device checks established tensor parallelism.
        foreach(TP_RANKS 2 4 7 8)
            add_test(NAME deepseek41-moe-tensor-parallel-cuda-${TP_RANKS}
                     COMMAND GgmlOpsDsv41TpTest --cuda ${TP_RANKS})
            set_tests_properties(deepseek41-moe-tensor-parallel-cuda-${TP_RANKS}
                                 PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
    endif()
    add_executable(GgmlOpsSchedulerCapacityTest tests/scheduler_capacity_test.cpp)
    target_include_directories(GgmlOpsSchedulerCapacityTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsSchedulerCapacityTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME scheduler-seventeen-backends COMMAND GgmlOpsSchedulerCapacityTest)

    add_executable(GgmlOpsSchedulerAllocTest tests/scheduler_alloc_test.cpp)
    target_include_directories(GgmlOpsSchedulerAllocTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsSchedulerAllocTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME scheduler-allocation-failure-recovery COMMAND GgmlOpsSchedulerAllocTest)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_executable(GgmlOpsCudaSchedulerAllocTest tests/cuda_scheduler_alloc_test.cpp)
        target_include_directories(GgmlOpsCudaSchedulerAllocTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(GgmlOpsCudaSchedulerAllocTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cudart)
        add_test(NAME cuda-scheduler-allocation-failure-recovery COMMAND GgmlOpsCudaSchedulerAllocTest)
        set_tests_properties(cuda-scheduler-allocation-failure-recovery
            PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()

    add_executable(GgmlOpsDsv41RawGatherTest
        tests/dsv41_raw_gather_test.cpp ggml_ops_dsv4_fused_cpu.cpp)
    target_include_directories(GgmlOpsDsv41RawGatherTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsDsv41RawGatherTest PRIVATE ggml-base)
    add_test(NAME deepseek41-compact-raw-gather COMMAND GgmlOpsDsv41RawGatherTest)

    add_executable(GgmlOpsDsv41TruncateTest tests/dsv41_truncate_test.cpp)
    target_include_directories(GgmlOpsDsv41TruncateTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-truncate-guards COMMAND GgmlOpsDsv41TruncateTest)
    add_executable(GgmlOpsDsv41DsparkRingTest tests/dsv41_dspark_ring_test.cpp)
    target_include_directories(GgmlOpsDsv41DsparkRingTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-dspark-ring-guards COMMAND GgmlOpsDsv41DsparkRingTest)
    add_executable(GgmlOpsDsv41RetentionTest tests/dsv41_retention_test.cpp)
    target_include_directories(GgmlOpsDsv41RetentionTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-retention-budget COMMAND GgmlOpsDsv41RetentionTest)
    add_executable(GgmlOpsGemma4MmMaskTest tests/gemma4_mm_mask_test.cpp)
    target_include_directories(GgmlOpsGemma4MmMaskTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME gemma4-multimodal-mask-after-reused-prefix COMMAND GgmlOpsGemma4MmMaskTest)

    add_executable(GgmlOpsDsv41QuantTest
        tests/dsv41_quant_test.cpp ggml_ops_dsv4_fused_cpu.cpp)
    target_include_directories(GgmlOpsDsv41QuantTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsDsv41QuantTest PRIVATE ggml-base)
    add_test(NAME deepseek41-activation-quant-candidates COMMAND GgmlOpsDsv41QuantTest)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND NOT WIN32)
        add_executable(GgmlOpsDsv41SparseFaTest tests/dsv41_sparse_fa_test.cpp)
        target_link_libraries(GgmlOpsDsv41SparseFaTest PRIVATE ggml-cuda ggml-base Threads::Threads)
        add_test(NAME deepseek41-sparse-flash-attention-cuda COMMAND GgmlOpsDsv41SparseFaTest)
        set_tests_properties(deepseek41-sparse-flash-attention-cuda PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(GgmlOpsDsv41QuantCudaTest tests/dsv41_quant_test.cpp)
        target_compile_definitions(GgmlOpsDsv41QuantCudaTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsDsv41QuantCudaTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
        target_link_libraries(GgmlOpsDsv41QuantCudaTest PRIVATE GgmlOps ggml-cuda)
        add_test(NAME deepseek41-activation-quant-candidates-cuda COMMAND GgmlOpsDsv41QuantCudaTest)
        set_tests_properties(deepseek41-activation-quant-candidates-cuda PROPERTIES SKIP_RETURN_CODE 77)
        # Explicit two-GPU transfer/attention measurement; never time this in a
        # concurrent CTest lane or treat it as full-model throughput evidence.
        add_executable(GgmlOpsDsv41CompactGatherBench tests/dsv41_compact_gather_bench.cpp)
        target_compile_definitions(GgmlOpsDsv41CompactGatherBench PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsDsv41CompactGatherBench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
        target_link_libraries(GgmlOpsDsv41CompactGatherBench PRIVATE GgmlOps ggml-cuda ggml-cpu)
    endif()

    # Compile the owned implementation into each precision test so Windows
    # exercises the same code without depending on DLL re-export details.
    add_executable(GgmlOpsMatmulPrecisionCpuTest tests/cuda_matmul_precision_test.cpp
        ggml_ops_matmul_precision.cpp)
    target_include_directories(GgmlOpsMatmulPrecisionCpuTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMatmulPrecisionCpuTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME cpu-explicit-f32-matmul COMMAND GgmlOpsMatmulPrecisionCpuTest)
    add_executable(GgmlOpsAttentionPrecisionCpuTest tests/attention_precision_test.cpp
        ggml_ops_attention_precision.cpp ggml_ops_matmul_precision.cpp)
    target_include_directories(GgmlOpsAttentionPrecisionCpuTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsAttentionPrecisionCpuTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME cpu-explicit-f32-attention COMMAND GgmlOpsAttentionPrecisionCpuTest)
    add_executable(GgmlOpsAttentionExceptionBoundaryTest tests/attention_exception_boundary_test.cpp
        ggml_ops_attention_precision.cpp ggml_ops_matmul_precision.cpp)
    target_include_directories(GgmlOpsAttentionExceptionBoundaryTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsAttentionExceptionBoundaryTest PRIVATE ggml-cpu ggml-base)
    if (MSVC)
        target_compile_options(GgmlOpsAttentionExceptionBoundaryTest PRIVATE /O2 /EHsc)
    endif()
    add_test(NAME attention-exception-boundary COMMAND GgmlOpsAttentionExceptionBoundaryTest)
    set_tests_properties(attention-exception-boundary PROPERTIES TIMEOUT 30)
    add_executable(GgmlOpsQ8PrecisionCpuTest tests/q8_precision_test.cpp ggml_ops_q8_precision.cpp)
    target_include_directories(GgmlOpsQ8PrecisionCpuTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsQ8PrecisionCpuTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME cpu-q8-f32-projection COMMAND GgmlOpsQ8PrecisionCpuTest)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_executable(GgmlOpsCudaMatmulPrecisionTest tests/cuda_matmul_precision_test.cpp
            ggml_ops_matmul_precision.cpp ggml_ops_matmul_precision.cu ggml_ops_dsv4_fused.cu
            ggml_ops_attention_precision.cu ggml_ops_q8_precision.cu)
        target_compile_definitions(GgmlOpsCudaMatmulPrecisionTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsCudaMatmulPrecisionTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsCudaMatmulPrecisionTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)
        add_test(NAME cuda-explicit-f32-matmul COMMAND GgmlOpsCudaMatmulPrecisionTest)
        # Measurement probe (not a test): which CUDA kernels a qwen4exp verify
        # row shares with the one-token decode graph, and at what cost.
        add_executable(GgmlOpsQwen4ExpRowKernelProbe tests/qwen4exp_row_kernel_probe.cpp)
        target_compile_definitions(GgmlOpsQwen4ExpRowKernelProbe PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsQwen4ExpRowKernelProbe PRIVATE ggml-cuda ggml-cpu ggml-base)
        set_tests_properties(cuda-explicit-f32-matmul PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(GgmlOpsCudaAttentionPrecisionTest tests/attention_precision_test.cpp
            ggml_ops_attention_precision.cpp ggml_ops_attention_precision.cu
            ggml_ops_matmul_precision.cpp ggml_ops_matmul_precision.cu ggml_ops_dsv4_fused.cu
            ggml_ops_q8_precision.cu)
        target_compile_definitions(GgmlOpsCudaAttentionPrecisionTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsCudaAttentionPrecisionTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsCudaAttentionPrecisionTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)
        add_test(NAME cuda-explicit-f32-attention COMMAND GgmlOpsCudaAttentionPrecisionTest)
        set_tests_properties(cuda-explicit-f32-attention PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(GgmlOpsCudaQ8PrecisionTest tests/q8_precision_test.cpp
            ggml_ops_q8_precision.cpp ggml_ops_q8_precision.cu ggml_ops_matmul_precision.cu
            ggml_ops_attention_precision.cu ggml_ops_dsv4_fused.cu)
        target_compile_definitions(GgmlOpsCudaQ8PrecisionTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsCudaQ8PrecisionTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsCudaQ8PrecisionTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)
        add_test(NAME cuda-q8-f32-projection COMMAND GgmlOpsCudaQ8PrecisionTest)
        set_tests_properties(cuda-q8-f32-projection PROPERTIES SKIP_RETURN_CODE 77)
        # Explicit performance control: includes transient conversion costs,
        # independent sampled numerical checks and a scratch-reuse guard.
        # Never register benchmarks in the automatic correctness CTest lane.
        add_executable(GgmlOpsQ8ProjectionBench tests/q8_projection_bench.cpp
            ggml_ops_q8_precision.cpp ggml_ops_q8_precision.cu
            ggml_ops_matmul_precision.cpp ggml_ops_matmul_precision.cu
            ggml_ops_attention_precision.cu ggml_ops_dsv4_fused.cu)
        target_compile_definitions(GgmlOpsQ8ProjectionBench PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsQ8ProjectionBench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsQ8ProjectionBench PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)

    endif()

    add_executable(GgmlOpsAttentionAllocTest
        tests/attention_alloc_test.cpp ggml_ops_attention_alloc.cpp)
    target_include_directories(GgmlOpsAttentionAllocTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsAttentionAllocTest PRIVATE ggml)
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsAttentionAllocTest PRIVATE TSG_GGML_USE_METAL=1)
    endif()
    add_test(NAME attention-allocation COMMAND GgmlOpsAttentionAllocTest)

    # Flash attention on shapes the backend has no kernel for (ggml-cuda aborts in
    # fattn.cu): the guard must fall back to explicit attention with the same
    # result and keep supported shapes on the kernel. CPU always, plus the first
    # GPU device the build has (CUDA / Metal).
    add_executable(GgmlOpsFlashAttnGuardTest
        tests/flash_attn_guard_test.cpp ggml_ops_flash_attn_guard.cpp)
    target_include_directories(GgmlOpsFlashAttnGuardTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsFlashAttnGuardTest PRIVATE ggml ggml-cpu ggml-base)
    if (MSVC)
        target_compile_options(GgmlOpsFlashAttnGuardTest PRIVATE /O2 /EHsc)
    endif()
    add_test(NAME flash-attn-unsupported-shape-fallback COMMAND GgmlOpsFlashAttnGuardTest)
    set_tests_properties(flash-attn-unsupported-shape-fallback PROPERTIES TIMEOUT 120)

    add_executable(GgmlOpsGetRowsQuantTest tests/get_rows_quant_test.cpp)
    target_include_directories(GgmlOpsGetRowsQuantTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsGetRowsQuantTest PRIVATE GgmlOps)

    # The standalone MoE kernel's uploaded biases must never share memory with an
    # activation a fused backend kernel writes while still reading them.
    add_executable(GgmlOpsMoeFusedBiasAliasTest tests/moe_fused_bias_alias_test.cpp)
    target_link_libraries(GgmlOpsMoeFusedBiasAliasTest PRIVATE GgmlOps)
    add_test(NAME moe-fused-bias-alias-cpu COMMAND GgmlOpsMoeFusedBiasAliasTest cpu)
    set_tests_properties(moe-fused-bias-alias-cpu PROPERTIES SKIP_RETURN_CODE 77)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME moe-fused-bias-alias-cuda COMMAND GgmlOpsMoeFusedBiasAliasTest cuda)
        set_tests_properties(moe-fused-bias-alias-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        add_test(NAME moe-fused-bias-alias-metal COMMAND GgmlOpsMoeFusedBiasAliasTest metal)
        set_tests_properties(moe-fused-bias-alias-metal PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(GgmlOpsAddmmVectorTest tests/addmm_vector_test.cpp)
    target_include_directories(GgmlOpsAddmmVectorTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsAddmmVectorTest PRIVATE GgmlOps)

    add_executable(GgmlOpsWanDitResidentCacheTest tests/wan_dit_resident_cache_test.cpp)
    target_include_directories(GgmlOpsWanDitResidentCacheTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsWanDitResidentCacheTest PRIVATE GgmlOps)
endif()
