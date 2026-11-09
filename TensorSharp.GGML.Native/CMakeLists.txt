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
# imports the x64 (or, on Windows on Arm, the ARM64) environment over such a
# prompt, and this check is the backstop for every other entry point (a hand-run
# cmake, -A Win32, a cross toolchain file). See
# https://github.com/zhongkaifu/TensorSharp/discussions/130
if (CMAKE_SIZEOF_VOID_P AND CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR
        "TensorSharp.GGML.Native requires a 64-bit toolchain, but this one targets "
        "${CMAKE_SIZEOF_VOID_P}-byte pointers (32-bit). On Windows, run build-windows.ps1 from a plain "
        "PowerShell so it can set up the x64 (or ARM64) MSVC environment itself, or from an "
        "\"x64 Native Tools\" (\"ARM64 Native Tools\") prompt - the default \"Developer PowerShell for VS\" "
        "targets x86. With a Visual Studio generator, pass -A x64 (-A ARM64 on Windows on Arm).")
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

# Use upstream's existing tiled CPU GEMM on new Linux x86 builds. Preserve an
# explicit GGML_LLAMAFILE setting (including an existing OFF cache); no fetched
# source is modified. Apple/Windows/other architectures keep their defaults.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND
    NOT DEFINED GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

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
    # ceiling moves with the toolkit (CUDA 13.4's host_config.h accepts
    # _MSC_VER < 1960; older CUDA 13 releases were not checked against VS 2026's
    # 14.5x toolsets, the case from discussion #130), and the flag is simply a
    # no-op when the toolset is one nvcc already supports.
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
    # TensorSharp builds only with CUDA 13 or newer (CUDA 12 is not supported).
    # Probe the nvcc CMake is about to identify, in CMake's own order:
    # CMAKE_CUDA_COMPILER, else CUDACXX, else CMake's compiler search, which looks
    # at PATH first and at CUDA_PATH last (tsg_cuda_compiler_search.cmake). An old
    # toolkit then fails here with a clear message instead of somewhere inside the
    # ggml-cuda build. The CUDAToolkit_VERSION check after find_package below is the
    # authoritative gate; it also covers Visual Studio generators, whose nvcc comes
    # from the VS CUDA toolset and is not probed here.
    include("${CMAKE_CURRENT_SOURCE_DIR}/tsg_cuda_compiler_search.cmake")
    set(_tsg_nvcc_probe "${CMAKE_CUDA_COMPILER}")
    if (NOT _tsg_nvcc_probe AND NOT "$ENV{CUDACXX}" STREQUAL "")
        set(_tsg_nvcc_probe "$ENV{CUDACXX}")
    endif()
    if (NOT _tsg_nvcc_probe)
        tsg_find_cuda_compiler(_tsg_nvcc_probe "${CMAKE_GENERATOR}")
    endif()
    if (_tsg_nvcc_probe)
        execute_process(COMMAND "${_tsg_nvcc_probe}" --version
                        OUTPUT_VARIABLE _tsg_nvcc_version_text ERROR_QUIET RESULT_VARIABLE _tsg_nvcc_version_result)
        if (_tsg_nvcc_version_result EQUAL 0 AND _tsg_nvcc_version_text MATCHES "release ([0-9]+)\\.([0-9]+),")
            if (CMAKE_MATCH_1 LESS 13)
                message(FATAL_ERROR "TensorSharp requires CUDA 13 or newer; found CUDA ${CMAKE_MATCH_1}.${CMAKE_MATCH_2} (${_tsg_nvcc_probe}). CUDA 12 is not supported: install a CUDA 13 toolkit, or point CMAKE_CUDA_COMPILER / CUDACXX at one.")
            endif()
        endif()
    endif()
    # CUDA 13 compiles nothing below compute capability 7.5 (Turing); say so before CMake
    # hands such an entry to nvcc during compiler identification.
    foreach (_tsg_cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if (_tsg_cuda_arch MATCHES "^(native|all|all-major)$")
            continue()
        endif()
        string(REGEX REPLACE "(-real|-virtual)$" "" _tsg_cuda_arch_number "${_tsg_cuda_arch}")
        string(REGEX REPLACE "[a-z]+$" "" _tsg_cuda_arch_number "${_tsg_cuda_arch_number}")
        if (_tsg_cuda_arch_number MATCHES "^[0-9]+$" AND _tsg_cuda_arch_number LESS 75)
            message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES entry '${_tsg_cuda_arch}' is below 75: CUDA 13 supports compute capability 7.5 (Turing) and newer only.")
        endif()
    endforeach()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if (CUDAToolkit_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "TensorSharp requires CUDA 13 or newer; found CUDA ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER}). CUDA 12 is not supported.")
    endif()
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

if (APPLE AND TARGET ggml-metal)
    # macOS 27 deprecates the eGPU location and Common GPU-family APIs used by
    # upstream's device discovery. Keep ggml unchanged and scope the diagnostic
    # exception to that upstream Objective-C file; TensorSharp sources and the
    # rest of ggml retain their deprecation warnings. Remove when upstream has
    # migrated these calls.
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src/ggml-metal/ggml-metal-device.m"
        TARGET_DIRECTORY ggml-metal APPEND PROPERTY COMPILE_OPTIONS
        "$<$<COMPILE_LANG_AND_ID:OBJC,AppleClang>:-Wno-deprecated-declarations>")
endif()

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
    bonsai_quant.cpp
    ggml_ops_core.cpp
    ggml_ops_cuda_launch_policy.cpp
    ggml_ops_graph_optimize.cpp
    ggml_ops_bonsai.cpp
    ggml_ops_attention_alloc.cpp
    ggml_ops_attention_precision.cpp
    ggml_ops_muse_spark_attention.cpp
    ggml_ops_flash_attn_guard.cpp
    ggml_ops_matmul.cpp
    ggml_ops_matmul_precision.cpp
    ggml_ops_q8_precision.cpp
    ggml_ops_elementwise.cpp
    ggml_ops_gemma4_audio.cpp
    ggml_ops_norm_attn.cpp
    ggml_ops_training.cpp
    ggml_ops_fused.cpp
    ggml_ops_transformer.cpp
    ggml_ops_transformer_prefill.cpp
    ggml_ops_gptoss_decode.cpp
    ggml_ops_gptoss_batched.cpp
    ggml_ops_qwen35_batched_arena.cpp
    ggml_ops_qwen35_batched_verify.cpp
    ggml_ops_qwen4exp.cpp
    ggml_ops_qwen4exp_reduce.cpp
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
    ggml_ops_muse_spark.cpp
    ggml_ops_muse_glimmer_vision.cpp
    ggml_ops_minimax_m3_vision.cpp
    ggml_ops_dflash.cpp
    ggml_ops_diffusion.cpp
    ggml_ops_qwen_image.cpp
    ggml_ops_qwen_image21.cpp
    ggml_ops_wan.cpp
    ggml_ops_minimax_h3.cpp
    ggml_ops_gated_delta_net.cpp
    ggml_ops_qwen35_recurrent_prefill.cpp
    ggml_ops_qwen35_gdn_tp.cpp
    ggml_ops_mamba2.cpp
    ggml_ops_moe.cpp
    ggml_ops_host_moe_decode.cpp
    ggml_ops_host_moe_cache.cpp
    ggml_ops_moe_hot_tier.cpp
    ggml_ops_moe_expert_io.cpp
    ggml_ops_buffer_reuse.cpp
    ggml_ops_deepseek4.cpp
    ggml_ops_deepseek41_tp.cpp
    ggml_ops_deepseek41_vision.cpp
    ggml_ops_embeddings.cpp
    ggml_ops_glm_dsa.cpp
    ggml_ops_muse_spark_exec.cpp
    ggml_ops_muse_spark_tp.cpp
    ggml_ops_minimax_m3_exec.cpp
    ggml_ops_minimax_m3_tp.cpp
    ggml_ops_minimax_m3_tp_backend.cpp
    ggml_ops_dsv4_fused_cpu.cpp
    ggml_ops_minimax_msa_cpu.cpp
    ggml_ops_paged_attention.cpp
    ggml_ops_tensor_parallel.cpp
)
# MiniMax-M3's TS_MINIMAX_M3_MSA=kernel mode exists once TensorSharp's MSA
# kernels (ggml_ops_minimax_msa.*, contract C3) are part of the library.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml_ops_minimax_msa_cpu.cpp")
    set_property(SOURCE ggml_ops_minimax_m3_exec.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_MINIMAX_MSA_KERNELS=1)
endif()
if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
    # Raises the process-wide default MSL version so ggml's Metal 4 tensor-API probe can compile
    # under the .NET apphost's stale recorded SDK. See the file header for the full story.
    list(APPEND TSG_GGMLOPS_SOURCES tsg_metal_msl_default.m)
    # Wan VAE convolutions on MPSGraph (see tsg_metal_mps_conv.mm).
    list(APPEND TSG_GGMLOPS_SOURCES tsg_metal_mps_conv.mm)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_gemma4_audio_metal_precision.mm)
    # This bridge stores Objective-C objects in C++ cache entries and releases
    # them by clearing entries/assigning nil. ARC is required for those ownership
    # operations, including temporary MPSGraphTensorData and grown staging buffers.
    set_source_files_properties(tsg_metal_mps_conv.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_source_files_properties(ggml_gemma4_audio_metal_precision.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    # TensorSharp Metal kernels (tsm/): grouped routed experts for the tiered MoE seams, ported from Strata's
    # MIT kernels (tsm/LICENSE-STRATA). Embedded into the library at build time, compiled at run time.
    set(TSM_KERNEL_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/quant.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/units.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/moe.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_moe_glue.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_gdn_prompt.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_hc_prompt.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_qsa_prompt.metal")
    set(TSM_GGML_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src/ggml-common.h")
    set(TSM_SOURCE_INC "${CMAKE_CURRENT_BINARY_DIR}/tsm_moe_source.inc")
    string(REPLACE ";" "\\;" TSM_KERNEL_FILES_ARG "${TSM_KERNEL_FILES}")
    add_custom_command(OUTPUT "${TSM_SOURCE_INC}"
        COMMAND ${CMAKE_COMMAND} -DOUT=${TSM_SOURCE_INC} -DGGML_COMMON=${TSM_GGML_COMMON}
                "-DFILES=${TSM_KERNEL_FILES_ARG}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        DEPENDS ${TSM_KERNEL_FILES} "${TSM_GGML_COMMON}" "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        COMMENT "Embedding the TensorSharp Metal kernels")
    # The tensor-API kernels: a library of their own (Metal language 4.0), so a device or SDK without the tensor API keeps
    # every kernel above.
    set(TSM_MPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_qsa_attn_mm.metal")
    set(TSM_MPP_INC "${CMAKE_CURRENT_BINARY_DIR}/tsm_mpp_source.inc")
    string(REPLACE ";" "\\;" TSM_MPP_FILES_ARG "${TSM_MPP_FILES}")
    add_custom_command(OUTPUT "${TSM_MPP_INC}"
        COMMAND ${CMAKE_COMMAND} -DOUT=${TSM_MPP_INC} -DVAR=kTsmMppSource -DMPP=1
                "-DFILES=${TSM_MPP_FILES_ARG}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        DEPENDS ${TSM_MPP_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        COMMENT "Embedding the TensorSharp Metal tensor-API kernels")
    # Strata's grouped expert GEMM for prompt chunks (tsm_moe_mm.metal) on the tensor API: a library of its own with
    # ggml-common.h and quant.metal, compiled on first use where the tensor API runs.
    set(TSM_MOEMM_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/quant.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/tsm/kernels/tsm_moe_mm.metal")
    set(TSM_MOEMM_INC "${CMAKE_CURRENT_BINARY_DIR}/tsm_moe_mm_source.inc")
    string(REPLACE ";" "\\;" TSM_MOEMM_FILES_ARG "${TSM_MOEMM_FILES}")
    add_custom_command(OUTPUT "${TSM_MOEMM_INC}"
        COMMAND ${CMAKE_COMMAND} -DOUT=${TSM_MOEMM_INC} -DVAR=kTsmMoeMmSource -DMPP=1 -DGGML_COMMON=${TSM_GGML_COMMON}
                "-DFILES=${TSM_MOEMM_FILES_ARG}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        DEPENDS ${TSM_MOEMM_FILES} "${TSM_GGML_COMMON}" "${CMAKE_CURRENT_SOURCE_DIR}/tsm/tsm_embed.cmake"
        COMMENT "Embedding the TensorSharp Metal prompt expert kernels")
    list(APPEND TSG_GGMLOPS_SOURCES tsm/tsm_moe.mm "${TSM_SOURCE_INC}" "${TSM_MPP_INC}" "${TSM_MOEMM_INC}")
    set_source_files_properties(tsm/tsm_moe.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
        OBJECT_DEPENDS "${TSM_SOURCE_INC};${TSM_MPP_INC};${TSM_MOEMM_INC}")
endif()
if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_diffusion_sample.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_mask.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_dsv4_fused.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_matmul_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_attention_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_q8_precision.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_tp_probe.cu ggml_ops_deepseek41_tp_cuda.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_host_pin.cu)
    list(APPEND TSG_GGMLOPS_SOURCES ggml_ops_minimax_msa.cu)
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

# With NCCL, ggml-cuda's VMM scratch pool shares every chunk with all
# peer-capable GPUs, which CUDA refuses past eight peers per GPU. Every target
# that links the static ggml-cuda routes ggml's cuMemSetAccess through
# TensorSharp's guard (see ggml_ops_cuda_vmm_peer_guard.cpp) instead of a
# patched ggml. GNU ld and lld rewrite the references inside the static
# ggml-cuda archive too; a NO_VMM build never calls it. Windows builds have no
# NCCL, so there the file only supplies the no-op NCCL policy.
function(tsg_add_cuda_vmm_peer_guard target)
    target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ggml_ops_cuda_vmm_peer_guard.cpp")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    if (TSG_USE_NCCL)
        target_compile_definitions(${target} PRIVATE TSG_GGML_CUDA_NCCL=1)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT GGML_CUDA_NO_VMM)
        target_compile_definitions(${target} PRIVATE TSG_CUDA_VMM_PEER_GUARD=1)
        get_target_property(TSG_GUARD_TYPE ${target} TYPE)
        if (TSG_GUARD_TYPE STREQUAL "SHARED_LIBRARY")
            target_compile_definitions(${target} PRIVATE TSG_CUDA_VMM_PEER_GUARD_HIDDEN=1)
        endif()
        target_link_options(${target} PRIVATE "$<HOST_LINK:LINKER:--wrap=cuMemSetAccess>")
        target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
    endif()
endfunction()

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
    target_link_libraries(GgmlOps PRIVATE CUDA::cublas CUDA::cuda_driver)
    tsg_add_cuda_vmm_peer_guard(GgmlOps)
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
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(GgmlOps PRIVATE ggml ${CMAKE_DL_LIBS})

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND GGML_LLAMAFILE)
    set_property(SOURCE ggml_ops_embeddings.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS TSG_EMBEDDING_CPU_TILED_GEMM=1)
endif()

if (APPLE AND GGML_ACCELERATE)
    find_library(TSG_EMBEDDING_ACCELERATE_FRAMEWORK Accelerate)
    if (TSG_EMBEDDING_ACCELERATE_FRAMEWORK)
        target_compile_definitions(GgmlOps PRIVATE TSG_GGML_USE_ACCELERATE=1
            ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
        target_link_libraries(GgmlOps PRIVATE ${TSG_EMBEDDING_ACCELERATE_FRAMEWORK})
    endif()
endif()

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
    # The CUDA 13 pre-check must probe the nvcc CMake itself would pick (PATH before
    # CUDA_PATH). Script-only: fake nvcc files, no toolkit and no GPU needed.
    add_test(NAME cuda-compiler-search-order
        COMMAND ${CMAKE_COMMAND} -DTSG_TEST_WORK_DIR=${CMAKE_BINARY_DIR}/artifacts/cuda-compiler-search
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_compiler_search_test.cmake")

    add_executable(GgmlOpsMiniMaxH3GraphInputsTest ../eng/validation/minimax-h3-graph-inputs.cpp)
    target_include_directories(GgmlOpsMiniMaxH3GraphInputsTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMiniMaxH3GraphInputsTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME minimax-h3-graph-inputs-cpu COMMAND GgmlOpsMiniMaxH3GraphInputsTest cpu)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsMiniMaxH3GraphInputsTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsMiniMaxH3GraphInputsTest PRIVATE ggml-cuda)
        add_test(NAME minimax-h3-graph-inputs-cuda COMMAND GgmlOpsMiniMaxH3GraphInputsTest cuda)
        set_tests_properties(minimax-h3-graph-inputs-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(GgmlOpsMiniMaxH3TokenRowsTest ../eng/validation/minimax-h3-token-rows.cpp)
    target_include_directories(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME minimax-h3-token-rows-cpu COMMAND GgmlOpsMiniMaxH3TokenRowsTest cpu --test)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE ggml-cuda)
        add_test(NAME minimax-h3-token-rows-cuda COMMAND GgmlOpsMiniMaxH3TokenRowsTest cuda --test)
        set_tests_properties(minimax-h3-token-rows-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE TSG_GGML_USE_METAL=1)
        target_link_libraries(GgmlOpsMiniMaxH3TokenRowsTest PRIVATE ggml-metal)
        add_test(NAME minimax-h3-token-rows-metal COMMAND GgmlOpsMiniMaxH3TokenRowsTest metal --test)
        set_tests_properties(minimax-h3-token-rows-metal PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(GgmlOpsMiniMaxH3LoraTest ../eng/validation/minimax-h3-lora.cpp)
    target_include_directories(GgmlOpsMiniMaxH3LoraTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMiniMaxH3LoraTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME minimax-h3-lora-cpu COMMAND GgmlOpsMiniMaxH3LoraTest cpu --test)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsMiniMaxH3LoraTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsMiniMaxH3LoraTest PRIVATE ggml-cuda)
        add_test(NAME minimax-h3-lora-cuda COMMAND GgmlOpsMiniMaxH3LoraTest cuda --test)
        set_tests_properties(minimax-h3-lora-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsMiniMaxH3LoraTest PRIVATE TSG_GGML_USE_METAL=1)
        target_link_libraries(GgmlOpsMiniMaxH3LoraTest PRIVATE ggml-metal)
        add_test(NAME minimax-h3-lora-metal COMMAND GgmlOpsMiniMaxH3LoraTest metal --test)
        set_tests_properties(minimax-h3-lora-metal PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(GgmlOpsMetalFlashKvPadTest ../eng/validation/metal-flash-kv-pad.cpp)
    target_include_directories(GgmlOpsMetalFlashKvPadTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMetalFlashKvPadTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME metal-flash-kv-pad-cpu COMMAND GgmlOpsMetalFlashKvPadTest cpu --test)
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsMetalFlashKvPadTest PRIVATE TSG_GGML_USE_METAL=1)
        target_link_libraries(GgmlOpsMetalFlashKvPadTest PRIVATE ggml-metal)
        add_test(NAME metal-flash-kv-pad-metal COMMAND GgmlOpsMetalFlashKvPadTest metal --test)
        set_tests_properties(metal-flash-kv-pad-metal PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(GgmlOpsGemma4VisionStackTest tests/gemma4_vision_stack_test.cpp)
    target_include_directories(GgmlOpsGemma4VisionStackTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsGemma4VisionStackTest PRIVATE GgmlOps)
    add_test(NAME gemma4-vision-stack-descriptors COMMAND GgmlOpsGemma4VisionStackTest)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME gemma4-vision-stack-cuda-admission COMMAND GgmlOpsGemma4VisionStackTest --require-cuda)
        set_tests_properties(gemma4-vision-stack-cuda-admission PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    add_executable(GgmlOpsHeadNormTest tests/head_norm_test.cpp)
    target_include_directories(GgmlOpsHeadNormTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsHeadNormTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME head-norm-cpu COMMAND GgmlOpsHeadNormTest cpu)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
        target_link_libraries(GgmlOpsHeadNormTest PRIVATE ggml-vulkan)
        target_compile_definitions(GgmlOpsHeadNormTest PRIVATE TSG_GGML_USE_VULKAN=1)
        add_test(NAME head-norm-vulkan COMMAND GgmlOpsHeadNormTest vulkan)
        set_tests_properties(head-norm-vulkan PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()
    add_executable(GgmlOpsEmbeddingCpuRmsTest tests/embedding_cpu_rms_test.cpp)
    target_include_directories(GgmlOpsEmbeddingCpuRmsTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsEmbeddingCpuRmsTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME embedding-cpu-rms COMMAND GgmlOpsEmbeddingCpuRmsTest)
    add_executable(GgmlOpsEmbeddingCpuAttentionTest tests/embedding_cpu_attention_test.cpp)
    target_include_directories(GgmlOpsEmbeddingCpuAttentionTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsEmbeddingCpuAttentionTest PRIVATE ggml-cpu ggml-base ${CMAKE_DL_LIBS})
    add_test(NAME embedding-cpu-attention COMMAND GgmlOpsEmbeddingCpuAttentionTest)
    add_executable(GgmlOpsEmbeddingCpuMatmulTest tests/embedding_cpu_matmul_test.cpp)
    target_include_directories(GgmlOpsEmbeddingCpuMatmulTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(GgmlOpsEmbeddingCpuMatmulTest PRIVATE ggml-cpu ggml-base ${CMAKE_DL_LIBS})
    if (TSG_EMBEDDING_ACCELERATE_FRAMEWORK)
        target_compile_definitions(GgmlOpsEmbeddingCpuMatmulTest PRIVATE TSG_GGML_USE_ACCELERATE=1
            ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
        target_link_libraries(GgmlOpsEmbeddingCpuMatmulTest PRIVATE ${TSG_EMBEDDING_ACCELERATE_FRAMEWORK})
    endif()
    add_test(NAME embedding-cpu-matmul COMMAND GgmlOpsEmbeddingCpuMatmulTest)
    set_property(SOURCE ggml_ops_qwen4exp.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    set_property(SOURCE ggml_ops_qwen4exp_arena.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    set_property(SOURCE ggml_ops_moe_hot_tier.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
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
    add_executable(GgmlOpsGraphOptimizeTest tests/graph_optimize_alloc_test.cpp ggml_ops_graph_optimize.cpp)
    target_include_directories(GgmlOpsGraphOptimizeTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsGraphOptimizeTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME graph-optimizer-allocation-dependencies COMMAND GgmlOpsGraphOptimizeTest)
    add_executable(GgmlOpsBonsaiHadamardTest tests/bonsai_hadamard_test.cpp ggml_ops_bonsai.cpp)
    target_include_directories(GgmlOpsBonsaiHadamardTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsBonsaiHadamardTest PRIVATE ggml ggml-cpu ggml-base)
    add_test(NAME bonsai2-hadamard-cpu COMMAND GgmlOpsBonsaiHadamardTest cpu)
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsBonsaiHadamardTest PRIVATE TSG_GGML_USE_METAL=1)
        target_link_libraries(GgmlOpsBonsaiHadamardTest PRIVATE ggml-metal)
        add_test(NAME bonsai2-hadamard-metal COMMAND GgmlOpsBonsaiHadamardTest metal)
        set_tests_properties(bonsai2-hadamard-metal PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsBonsaiHadamardTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsBonsaiHadamardTest PRIVATE ggml-cuda)
        add_test(NAME bonsai2-hadamard-cuda COMMAND GgmlOpsBonsaiHadamardTest cuda)
        set_tests_properties(bonsai2-hadamard-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    add_executable(GgmlOpsBonsaiQuantTest tests/bonsai_quant_test.cpp bonsai_quant.cpp)
    target_include_directories(GgmlOpsBonsaiQuantTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsBonsaiQuantTest PRIVATE ggml ggml-cpu ggml-base)
    add_test(NAME bonsai2-lossless-quant-cpu COMMAND GgmlOpsBonsaiQuantTest cpu)
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        target_compile_definitions(GgmlOpsBonsaiQuantTest PRIVATE TSG_GGML_USE_METAL=1)
        target_link_libraries(GgmlOpsBonsaiQuantTest PRIVATE ggml-metal)
        add_test(NAME bonsai2-lossless-quant-metal COMMAND GgmlOpsBonsaiQuantTest metal)
        set_tests_properties(bonsai2-lossless-quant-metal PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsBonsaiQuantTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsBonsaiQuantTest PRIVATE ggml-cuda)
        add_test(NAME bonsai2-lossless-quant-cuda COMMAND GgmlOpsBonsaiQuantTest cuda)
        set_tests_properties(bonsai2-lossless-quant-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    add_executable(GgmlOpsDsv41EngramTest tests/dsv41_engram_test.cpp)
    add_test(NAME deepseek41-engram-hash-lookup COMMAND GgmlOpsDsv41EngramTest)
    add_executable(GgmlOpsMuseSparkTpSplitTest tests/muse_spark_tp_split_test.cpp ggml_ops_muse_spark_tp.cpp)
    target_include_directories(GgmlOpsMuseSparkTpSplitTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMuseSparkTpSplitTest PRIVATE ggml-base)
    add_test(NAME muse-spark-tp-split-plan COMMAND GgmlOpsMuseSparkTpSplitTest)
    # MiniMax-M3's tensor-parallel split plan and memory budget on the real
    # checkpoint's shapes (pure planning: no GPU).
    add_executable(GgmlOpsMiniMaxM3TpSplitTest tests/minimax_m3_tp_split_test.cpp ggml_ops_minimax_m3_tp.cpp)
    target_include_directories(GgmlOpsMiniMaxM3TpSplitTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMiniMaxM3TpSplitTest PRIVATE ggml-base)
    add_test(NAME minimax-m3-tp-split-plan COMMAND GgmlOpsMiniMaxM3TpSplitTest)
    add_executable(GgmlOpsDsv41EngramGgufTest tests/dsv41_engram_gguf_test.cpp)
    target_link_libraries(GgmlOpsDsv41EngramGgufTest PRIVATE ggml-base)
    add_test(NAME deepseek41-engram-embedded-gguf COMMAND GgmlOpsDsv41EngramGgufTest)
    add_executable(GgmlOpsDsv41EngramIoTest tests/dsv41_engram_io_test.cpp)
    target_link_libraries(GgmlOpsDsv41EngramIoTest PRIVATE Threads::Threads)
    add_test(NAME deepseek41-engram-parallel-io COMMAND GgmlOpsDsv41EngramIoTest)
    add_executable(GgmlOpsDsv4ExactReadTest tests/dsv4_exact_read_test.cpp)
    target_link_libraries(GgmlOpsDsv4ExactReadTest PRIVATE Threads::Threads)
    add_test(NAME deepseek4-exact-file-reads COMMAND GgmlOpsDsv4ExactReadTest)
    set_tests_properties(deepseek4-exact-file-reads PROPERTIES TIMEOUT 30)
    add_executable(GgmlOpsDsv4FileWarmTest tests/dsv4_file_warm_test.cpp)
    target_link_libraries(GgmlOpsDsv4FileWarmTest PRIVATE Threads::Threads)
    add_test(NAME deepseek41-host-file-warm COMMAND GgmlOpsDsv4FileWarmTest)

    add_executable(GgmlOpsGlmFileStripTest tests/glm_file_strip_test.cpp)
    add_test(NAME glm-checkpoint-strips COMMAND GgmlOpsGlmFileStripTest)
    set_tests_properties(deepseek41-host-file-warm PROPERTIES TIMEOUT 120)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Explicit host-only storage measurement on a real checkpoint shard; it
        # evicts the named range from the page cache. Never part of CTest.
        add_executable(GgmlOpsDsv4FileWarmBench tests/dsv4_file_warm_bench.cpp)
        target_link_libraries(GgmlOpsDsv4FileWarmBench PRIVATE Threads::Threads)
    endif()
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
        ggml_ops_matmul_precision.cpp ggml_ops_dsv4_fused_cpu.cpp)
    target_compile_definitions(GgmlOpsDsv41TpTest PRIVATE TSG_GGML_TEST_HOOKS)
    target_include_directories(GgmlOpsDsv41TpTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsDsv41TpTest PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME deepseek41-moe-tensor-parallel COMMAND GgmlOpsDsv41TpTest)
    add_test(NAME deepseek41-tp-failure-recovery COMMAND GgmlOpsDsv41TpTest --failure-only)
    add_test(NAME deepseek41-distributed-moe-numerical COMMAND GgmlOpsDsv41TpTest --distributed-only)
    set_tests_properties(deepseek41-distributed-moe-numerical PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND NOT WIN32)
        target_compile_definitions(GgmlOpsDsv41TpTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_sources(GgmlOpsDsv41TpTest PRIVATE ggml_ops_dsv4_fused.cu ggml_ops_matmul_precision.cu
            ggml_ops_attention_precision.cu ggml_ops_q8_precision.cu ggml_ops_tp_probe.cu
            ggml_ops_deepseek41_tp_cuda.cu)
        target_include_directories(GgmlOpsDsv41TpTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsDsv41TpTest PRIVATE ggml-cuda CUDA::cublas ${CMAKE_DL_LIBS})
        add_test(NAME deepseek41-quantized-strip-projections-cuda
                 COMMAND GgmlOpsDsv41TpTest --cuda 1 --quant-strip-only)
        set_tests_properties(deepseek41-quantized-strip-projections-cuda PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 600)
        # A rank graph that throws on the device-resident exchange: rank 0,
        # a worker rank, then both. A regression aborts or hangs, hence the
        # timeout.
        add_test(NAME deepseek41-tp-resident-failure-cuda
                 COMMAND GgmlOpsDsv41TpTest --cuda 2 --resident-failure-only)
        set_tests_properties(deepseek41-tp-resident-failure-cuda PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 600)
        # Multi-device tests are explicit: a one-GPU CI lane must not claim
        # that its skipped multi-device checks established tensor parallelism.
        foreach(TP_RANKS 2 3 4 5 6 7 8)
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

    # The production MoE selector must return compact, score-ordered IDs before
    # weight normalization and expert accumulation. No model checkpoint needed.
    add_executable(GgmlOpsOrderedMoeRoutingTest tests/ordered_moe_routing_test.cpp)
    target_include_directories(GgmlOpsOrderedMoeRoutingTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsOrderedMoeRoutingTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME ordered-moe-routing-cpu COMMAND GgmlOpsOrderedMoeRoutingTest cpu)
    set_tests_properties(ordered-moe-routing-cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 60)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsOrderedMoeRoutingTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsOrderedMoeRoutingTest PRIVATE ggml-cuda)
        add_test(NAME ordered-moe-routing-cuda COMMAND GgmlOpsOrderedMoeRoutingTest cuda)
        set_tests_properties(ordered-moe-routing-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()

    add_executable(GgmlOpsGraphAllocCurrentPlanTest tests/graph_alloc_current_plan_test.cpp)
    target_include_directories(GgmlOpsGraphAllocCurrentPlanTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsGraphAllocCurrentPlanTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME current-graph-plan-cpu COMMAND GgmlOpsGraphAllocCurrentPlanTest cpu)
    set_tests_properties(current-graph-plan-cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 60)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsGraphAllocCurrentPlanTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsGraphAllocCurrentPlanTest PRIVATE ggml-cuda)
        add_test(NAME current-graph-plan-cuda COMMAND GgmlOpsGraphAllocCurrentPlanTest cuda)
        set_tests_properties(current-graph-plan-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()

    add_executable(GgmlOpsHostMoeExpertCacheTest
        tests/host_moe_expert_cache_test.cpp ggml_ops_host_moe_cache.cpp)
    target_include_directories(GgmlOpsHostMoeExpertCacheTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsHostMoeExpertCacheTest PRIVATE ggml-cpu ggml-base)
    add_executable(GgmlOpsHostMoeDecodeRowsTest
        tests/host_moe_decode_rows_test.cpp ggml_ops_host_moe_decode.cpp)
    target_include_directories(GgmlOpsHostMoeDecodeRowsTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsHostMoeDecodeRowsTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME host-moe-decode-independent-rows COMMAND GgmlOpsHostMoeDecodeRowsTest)
    set_tests_properties(host-moe-decode-independent-rows PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
    add_test(NAME host-moe-expert-cache-cpu-fallback COMMAND GgmlOpsHostMoeExpertCacheTest --cpu)
    add_test(NAME host-moe-expert-cache-invalid-budget-sign
        COMMAND GgmlOpsHostMoeExpertCacheTest --invalid-budget "+8192")
    add_test(NAME host-moe-expert-cache-invalid-budget-leading-space
        COMMAND GgmlOpsHostMoeExpertCacheTest --invalid-budget " 8192")
    add_test(NAME host-moe-expert-cache-invalid-budget-trailing-space
        COMMAND GgmlOpsHostMoeExpertCacheTest --invalid-budget "8192 ")
    add_test(NAME host-moe-expert-cache-invalid-budget-overflow
        COMMAND GgmlOpsHostMoeExpertCacheTest --invalid-budget "18446744073709551616")
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsHostMoeExpertCacheTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsHostMoeExpertCacheTest PRIVATE ggml-cuda)
        add_test(NAME host-moe-expert-cache-cuda-parity COMMAND GgmlOpsHostMoeExpertCacheTest)
        add_test(NAME host-moe-expert-cache-cuda-iq1-m-parity
            COMMAND GgmlOpsHostMoeExpertCacheTest --quantization iq1_m)
        add_test(NAME host-moe-expert-cache-cuda-iq2-xxs-parity
            COMMAND GgmlOpsHostMoeExpertCacheTest --quantization iq2_xxs)
        add_test(NAME host-moe-expert-cache-cuda-prefetch-parity
            COMMAND GgmlOpsHostMoeExpertCacheTest --prefetch)
        add_test(NAME host-moe-expert-cache-cuda-iq1-m-prefetch-parity
            COMMAND GgmlOpsHostMoeExpertCacheTest --quantization iq1_m --prefetch)
        add_test(NAME host-moe-expert-cache-cuda-iq2-xxs-prefetch-parity
            COMMAND GgmlOpsHostMoeExpertCacheTest --quantization iq2_xxs --prefetch)
        set_tests_properties(host-moe-expert-cache-cuda-parity
            host-moe-expert-cache-cuda-iq1-m-parity host-moe-expert-cache-cuda-iq2-xxs-parity
            host-moe-expert-cache-cuda-prefetch-parity host-moe-expert-cache-cuda-iq1-m-prefetch-parity
            host-moe-expert-cache-cuda-iq2-xxs-prefetch-parity
            PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()

    add_executable(GgmlOpsDsv41RawGatherTest
        tests/dsv41_raw_gather_test.cpp ggml_ops_dsv4_fused_cpu.cpp)
    target_include_directories(GgmlOpsDsv41RawGatherTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsDsv41RawGatherTest PRIVATE ggml-base)
    add_test(NAME deepseek41-compact-raw-gather COMMAND GgmlOpsDsv41RawGatherTest)

    add_executable(GgmlOpsDsv41YocoTest tests/dsv41_yoco_test.cpp)
    target_include_directories(GgmlOpsDsv41YocoTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-yoco-policy COMMAND GgmlOpsDsv41YocoTest)
    add_executable(GgmlOpsDsv41TruncateTest tests/dsv41_truncate_test.cpp)
    target_include_directories(GgmlOpsDsv41TruncateTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-truncate-guards COMMAND GgmlOpsDsv41TruncateTest)
    add_executable(GgmlOpsDsv41DsparkRingTest tests/dsv41_dspark_ring_test.cpp)
    target_include_directories(GgmlOpsDsv41DsparkRingTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-dspark-ring-guards COMMAND GgmlOpsDsv41DsparkRingTest)
    add_executable(GgmlOpsDsv41VisionClipTest tests/dsv41_vision_clip_test.cpp)
    target_include_directories(GgmlOpsDsv41VisionClipTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    set(TSG_DSV41_VISION_CLIP_LIBRARIES GgmlOps ggml-base)
    if (WIN32)
        # The test's GGUF writer needs ggml-base APIs not exported by GgmlOps.
        # Resolve its ggml functions from that static archive before the DLL's
        # import library, or ggml.c collides with already imported ggml exports
        # (LNK2005). Only fixture filenames, never ggml contexts, cross the DLL.
        set(TSG_DSV41_VISION_CLIP_LIBRARIES ggml-base GgmlOps)
    endif()
    target_link_libraries(GgmlOpsDsv41VisionClipTest PRIVATE ${TSG_DSV41_VISION_CLIP_LIBRARIES})
    add_test(NAME deepseek41-vision-clip-mmproj COMMAND GgmlOpsDsv41VisionClipTest)
    add_executable(GgmlOpsDsv41RetentionTest tests/dsv41_retention_test.cpp)
    target_include_directories(GgmlOpsDsv41RetentionTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-retention-budget COMMAND GgmlOpsDsv41RetentionTest)
    add_executable(GgmlOpsGemma4MmMaskTest tests/gemma4_mm_mask_test.cpp)
    target_include_directories(GgmlOpsGemma4MmMaskTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME gemma4-multimodal-mask-after-reused-prefix COMMAND GgmlOpsGemma4MmMaskTest)
    add_executable(GgmlOpsGemma4PleNormTest tests/gemma4_ple_norm_test.cpp)
    target_include_directories(GgmlOpsGemma4PleNormTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsGemma4PleNormTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME gemma4-ple-norm-cpu COMMAND GgmlOpsGemma4PleNormTest cpu)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
        target_link_libraries(GgmlOpsGemma4PleNormTest PRIVATE ggml-vulkan)
        target_compile_definitions(GgmlOpsGemma4PleNormTest PRIVATE TSG_GGML_USE_VULKAN=1)
        add_test(NAME gemma4-ple-norm-vulkan COMMAND GgmlOpsGemma4PleNormTest vulkan)
        set_tests_properties(gemma4-ple-norm-vulkan PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()

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
    add_executable(GgmlOpsAudioCpuProjectionTest tests/embedding_audio_cpu_projection_test.cpp)
    target_include_directories(GgmlOpsAudioCpuProjectionTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsAudioCpuProjectionTest PRIVATE ggml-cpu ggml-base ${CMAKE_DL_LIBS})
    add_test(NAME audio-cpu-projection-policy COMMAND GgmlOpsAudioCpuProjectionTest)
    add_executable(GgmlOpsMatmulPrecisionCpuTest tests/cuda_matmul_precision_test.cpp
        ggml_ops_matmul_precision.cpp)
    target_include_directories(GgmlOpsMatmulPrecisionCpuTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsMatmulPrecisionCpuTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME cpu-explicit-f32-matmul COMMAND GgmlOpsMatmulPrecisionCpuTest)
    add_test(NAME cpu-quantized-ffn-strip COMMAND GgmlOpsMatmulPrecisionCpuTest --quant-strip-only)
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
        target_compile_definitions(GgmlOpsCudaMatmulPrecisionTest PRIVATE TSG_GGML_USE_CUDA=1 TSG_GGML_TEST_HOOKS)
        target_include_directories(GgmlOpsCudaMatmulPrecisionTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsCudaMatmulPrecisionTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)
        add_test(NAME cuda-explicit-f32-matmul COMMAND GgmlOpsCudaMatmulPrecisionTest)
        add_test(NAME cuda-quantized-ffn-strip COMMAND GgmlOpsCudaMatmulPrecisionTest --quant-strip-only --inner 2560)
        add_test(NAME cuda-quantized-ffn-wide-scratch COMMAND GgmlOpsCudaMatmulPrecisionTest --quant-strip-wide-only)
        # Measurement probe (not a test): which CUDA kernels a qwen4exp verify
        # row shares with the one-token decode graph, and at what cost.
        add_executable(GgmlOpsQwen4ExpRowKernelProbe tests/qwen4exp_row_kernel_probe.cpp)
        target_compile_definitions(GgmlOpsQwen4ExpRowKernelProbe PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsQwen4ExpRowKernelProbe PRIVATE ggml-cuda ggml-cpu ggml-base)
        set_tests_properties(cuda-explicit-f32-matmul PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(cuda-quantized-ffn-strip PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        set_tests_properties(cuda-quantized-ffn-wide-scratch PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 600)
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
        add_executable(GgmlOpsQwen4ExpReduceTest tests/qwen4exp_reduce_test.cpp
            ggml_ops_qwen4exp_reduce.cpp ggml_ops_matmul_precision.cu
            ggml_ops_attention_precision.cu ggml_ops_q8_precision.cu ggml_ops_dsv4_fused.cu)
        target_compile_definitions(GgmlOpsQwen4ExpReduceTest PRIVATE TSG_GGML_USE_CUDA=1)
        target_include_directories(GgmlOpsQwen4ExpReduceTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsQwen4ExpReduceTest PRIVATE ggml-cuda ggml-cpu ggml-base CUDA::cublas)
        add_test(NAME qwen4exp-prefill-combine-cuda COMMAND GgmlOpsQwen4ExpReduceTest cuda)
        set_tests_properties(qwen4exp-prefill-combine-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
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
        # MiniMax sparse attention kernels against the CPU reference and a dense
        # HF-semantics reference, plus the fused-backend graph node. CTest runs
        # the reduced matrix; run the binary without arguments for the full one
        # (minutes: 33K-200K-row CPU references) and with `--bench` for
        # real-shape timings, never in a concurrent CTest lane.
        add_executable(GgmlOpsMiniMaxMsaKernelTest tests/minimax_msa_kernel_test.cu
            ggml_ops_minimax_msa.cu ggml_ops_minimax_msa_cpu.cpp ggml_ops_dsv4_fused.cu
            ggml_ops_matmul_precision.cu ggml_ops_attention_precision.cu ggml_ops_q8_precision.cu)
        target_compile_definitions(GgmlOpsMiniMaxMsaKernelTest PRIVATE TSG_GGML_USE_CUDA=1 TSG_MSA_TEST_HOOKS)
        target_include_directories(GgmlOpsMiniMaxMsaKernelTest PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
        target_link_libraries(GgmlOpsMiniMaxMsaKernelTest PRIVATE
            ggml-cuda ggml-cpu ggml-base CUDA::cublas Threads::Threads)
        add_test(NAME minimax-msa-kernels-cuda COMMAND GgmlOpsMiniMaxMsaKernelTest --quick)
        set_tests_properties(minimax-msa-kernels-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900)

    endif()

    add_executable(GgmlOpsQwen4ExpReduceCpuTest tests/qwen4exp_reduce_test.cpp ggml_ops_qwen4exp_reduce.cpp)
    target_include_directories(GgmlOpsQwen4ExpReduceCpuTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsQwen4ExpReduceCpuTest PRIVATE ggml-cpu ggml-base)
    add_test(NAME qwen4exp-prefill-combine-cpu COMMAND GgmlOpsQwen4ExpReduceCpuTest cpu)

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

    add_executable(GgmlOpsDsv4UbatchPlanTest tests/dsv4_ubatch_plan_test.cpp)
    target_include_directories(GgmlOpsDsv4UbatchPlanTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek4-ubatch-plan COMMAND GgmlOpsDsv4UbatchPlanTest)
    set_tests_properties(deepseek4-ubatch-plan PROPERTIES TIMEOUT 60)

    # DeepSeek V4.1: a micro-batch's attention width (and so its attention
    # kernel) must not depend on the forward call that contains it, or a
    # prompt prefilled whole and in chunks builds different caches. Header-only.
    add_executable(GgmlOpsDsv41CallInvarianceTest tests/dsv41_call_invariance_test.cpp)
    target_include_directories(GgmlOpsDsv41CallInvarianceTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME deepseek41-call-invariant-attention-width COMMAND GgmlOpsDsv41CallInvarianceTest)
    set_tests_properties(deepseek41-call-invariant-attention-width PROPERTIES TIMEOUT 60)

    add_executable(GgmlOpsLayerPartitionTest tests/layer_partition_test.cpp)
    target_include_directories(GgmlOpsLayerPartitionTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME distributed-layer-byte-partition COMMAND GgmlOpsLayerPartitionTest)
    set_tests_properties(distributed-layer-byte-partition PROPERTIES TIMEOUT 60)

    # host_read_barrier's drain gate with a sync the test blocks: a caller that
    # arrives while another thread drains waits for that drain. Header-only.
    add_executable(GgmlOpsHostReadGateTest tests/host_read_gate_test.cpp)
    target_include_directories(GgmlOpsHostReadGateTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsHostReadGateTest PRIVATE Threads::Threads)
    add_test(NAME host-read-barrier-drain-gate COMMAND GgmlOpsHostReadGateTest)
    set_tests_properties(host-read-barrier-drain-gate PROPERTIES TIMEOUT 60)
    # Explicit per-width routed-expert layer timing behind the automatic ubatch
    # (see the source header). Never registered as a CTest.
    add_executable(GgmlOpsDsv4MoeWidthBench tests/dsv4_moe_width_bench.cpp)
    target_include_directories(GgmlOpsDsv4MoeWidthBench PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsDsv4MoeWidthBench PRIVATE ggml-cpu ggml-base)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        target_compile_definitions(GgmlOpsDsv4MoeWidthBench PRIVATE TSG_GGML_USE_CUDA=1)
        target_link_libraries(GgmlOpsDsv4MoeWidthBench PRIVATE ggml-cuda)
    endif()

    add_executable(GgmlOpsGetRowsQuantTest tests/get_rows_quant_test.cpp)
    target_include_directories(GgmlOpsGetRowsQuantTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsGetRowsQuantTest PRIVATE GgmlOps)

    # Test-only observability; regular production builds expose no probe ABI.
    target_compile_definitions(GgmlOps PRIVATE TSG_GGML_TEST_DIAGNOSTICS=1)
    add_executable(GgmlOpsPagedAttentionLifecycleTest tests/paged_attention_lifecycle_test.cpp)
    target_link_libraries(GgmlOpsPagedAttentionLifecycleTest PRIVATE GgmlOps)
    add_test(NAME paged-attention-lifecycle-cpu COMMAND GgmlOpsPagedAttentionLifecycleTest cpu)
    set_tests_properties(paged-attention-lifecycle-cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME paged-attention-lifecycle-cuda COMMAND GgmlOpsPagedAttentionLifecycleTest cuda)
        add_test(NAME paged-attention-lifecycle-cuda-multi COMMAND GgmlOpsPagedAttentionLifecycleTest multi)
        set_tests_properties(paged-attention-lifecycle-cuda paged-attention-lifecycle-cuda-multi
            PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
    # Cached buffers a queued download may read are drained before release:
    # session-cache eviction and reuse-allocator growth, on the CPU backend.
    add_executable(GgmlOpsHostReadEvictionTest tests/host_read_eviction_test.cpp)
    target_link_libraries(GgmlOpsHostReadEvictionTest PRIVATE GgmlOps)
    add_test(NAME host-read-barrier-eviction-cpu COMMAND GgmlOpsHostReadEvictionTest)
    set_tests_properties(host-read-barrier-eviction-cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)

    # The standalone MoE kernel's uploaded biases must never share memory with an
    # activation a fused backend kernel writes while still reading them.
    add_executable(GgmlOpsMoeFusedBiasAliasTest tests/moe_fused_bias_alias_test.cpp)
    add_executable(GgmlOpsMuseSparkMoeTest tests/muse_spark_moe_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkMoeTest PRIVATE GgmlOps)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_executable(GgmlOpsAttentionDeviceCacheTest tests/attention_device_cache_test.cpp)
        target_link_libraries(GgmlOpsAttentionDeviceCacheTest PRIVATE GgmlOps)
        add_test(NAME attention-device-cache-cuda COMMAND GgmlOpsAttentionDeviceCacheTest)
        set_tests_properties(attention-device-cache-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_test(NAME muse-spark-moe-cpu COMMAND GgmlOpsMuseSparkMoeTest cpu)
    set_tests_properties(muse-spark-moe-cpu PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(GgmlOpsMuseSparkCpuMoeScratchTest tests/muse_spark_cpu_moe_scratch_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkCpuMoeScratchTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-cpu-moe-scratch-off COMMAND GgmlOpsMuseSparkCpuMoeScratchTest off)
    add_test(NAME muse-spark-cpu-moe-scratch-on COMMAND GgmlOpsMuseSparkCpuMoeScratchTest on)
    add_test(NAME muse-spark-moe-cpu-reuse COMMAND GgmlOpsMuseSparkMoeTest cpu)
    add_test(NAME moe-fused-bias-alias-cpu-reuse COMMAND GgmlOpsMoeFusedBiasAliasTest cpu)
    # Each test is a new process: native startup getenv does not depend on .NET
    # environment propagation. Legacy/non-Spark allocation must stay unchanged.
    set_tests_properties(muse-spark-cpu-moe-scratch-off muse-spark-moe-cpu
        PROPERTIES ENVIRONMENT "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=0;TS_CPU_MOE_THREADS=2")
    set_tests_properties(muse-spark-cpu-moe-scratch-on muse-spark-moe-cpu-reuse moe-fused-bias-alias-cpu-reuse
        PROPERTIES ENVIRONMENT "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1;TS_CPU_MOE_THREADS=2")
    set_tests_properties(muse-spark-cpu-moe-scratch-off muse-spark-cpu-moe-scratch-on
        muse-spark-moe-cpu-reuse moe-fused-bias-alias-cpu-reuse
        PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-moe-cuda COMMAND GgmlOpsMuseSparkMoeTest cuda)
        set_tests_properties(muse-spark-moe-cuda PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME muse-spark-moe-cuda-multi COMMAND GgmlOpsMuseSparkMoeTest cuda-multi)
        set_tests_properties(muse-spark-moe-cuda-multi PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME muse-spark-moe-cuda-reuse COMMAND GgmlOpsMuseSparkMoeTest cuda)
        add_test(NAME muse-spark-moe-cuda-multi-reuse COMMAND GgmlOpsMuseSparkMoeTest cuda-multi)
        set_tests_properties(muse-spark-moe-cuda muse-spark-moe-cuda-multi PROPERTIES
            ENVIRONMENT "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=0;TS_CPU_MOE_THREADS=2")
        set_tests_properties(muse-spark-moe-cuda-reuse muse-spark-moe-cuda-multi-reuse PROPERTIES
            ENVIRONMENT "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1;TS_CPU_MOE_THREADS=2"
            SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()
    target_link_libraries(GgmlOpsMoeFusedBiasAliasTest PRIVATE GgmlOps)
    add_test(NAME moe-fused-bias-alias-cpu COMMAND GgmlOpsMoeFusedBiasAliasTest cpu)
    set_tests_properties(moe-fused-bias-alias-cpu PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(moe-fused-bias-alias-cpu PROPERTIES
        ENVIRONMENT "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=0;TS_CPU_MOE_THREADS=2")
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME moe-fused-bias-alias-cuda COMMAND GgmlOpsMoeFusedBiasAliasTest cuda)
        set_tests_properties(moe-fused-bias-alias-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        add_test(NAME moe-fused-bias-alias-metal COMMAND GgmlOpsMoeFusedBiasAliasTest metal)
        set_tests_properties(moe-fused-bias-alias-metal PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # MiniMax-M3's whole-model executor on a fixture the test writes itself: a
    # sequence's logits do not depend on how it is fed (chunks, decode, rewind,
    # slot copy, batched decode, segments, forced sparse attention below the
    # dense span) or placed (layer split, host experts).
    add_executable(GgmlOpsMiniMaxM3ExecTest tests/minimax_m3_exec_test.cpp)
    target_include_directories(GgmlOpsMiniMaxM3ExecTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    if (WIN32)
        # The synthetic GGUF writer calls APIs not exported by GgmlOps.dll.
        # Keep the static archive first to avoid duplicate imported ggml symbols;
        # only the written fixture filename crosses the DLL boundary.
        target_link_libraries(GgmlOpsMiniMaxM3ExecTest PRIVATE ggml-base GgmlOps)
    else()
        target_link_libraries(GgmlOpsMiniMaxM3ExecTest PRIVATE GgmlOps)
    endif()
    add_test(NAME minimax-m3-exec-cpu COMMAND GgmlOpsMiniMaxM3ExecTest cpu)
    set_tests_properties(minimax-m3-exec-cpu PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 600)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME minimax-m3-exec-cuda COMMAND GgmlOpsMiniMaxM3ExecTest cuda)
        add_test(NAME minimax-m3-exec-cuda-multi COMMAND GgmlOpsMiniMaxM3ExecTest cuda-multi)
        add_test(NAME minimax-m3-exec-cuda-cpumoe COMMAND GgmlOpsMiniMaxM3ExecTest cuda-cpumoe)
        add_test(NAME minimax-m3-exec-cuda-tp COMMAND GgmlOpsMiniMaxM3ExecTest cuda-tp)
        set_tests_properties(minimax-m3-exec-cuda minimax-m3-exec-cuda-multi minimax-m3-exec-cuda-cpumoe minimax-m3-exec-cuda-tp
            PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 600)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        # TS_MINIMAX_M3_MSA=kernel is refused on a Metal GPU (the MSA kernels are CUDA's).
        add_test(NAME minimax-m3-exec-metal COMMAND GgmlOpsMiniMaxM3ExecTest metal)
        set_tests_properties(minimax-m3-exec-metal PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 600)
    endif()
    # The executor's sparse-attention graphs (decode, verification, chunked
    # prompt) against the MSA reference math on well-separated synthetic block
    # scores, through the TSGgml_MiniMaxM3TestAttention hook; also against
    # tsg_msa_attention_cpu when TensorSharp's MSA kernels are in the tree.
    set_property(SOURCE ggml_ops_minimax_m3_exec.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    add_executable(GgmlOpsMiniMaxM3MsaGraphTest tests/minimax_m3_msa_graph_test.cpp)
    target_include_directories(GgmlOpsMiniMaxM3MsaGraphTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src"
    )
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml_ops_minimax_msa_cpu.cpp")
        target_sources(GgmlOpsMiniMaxM3MsaGraphTest PRIVATE ggml_ops_minimax_msa_cpu.cpp)
        target_compile_definitions(GgmlOpsMiniMaxM3MsaGraphTest PRIVATE TSG_MM3_TEST_MSA_REFERENCE=1)
        target_link_libraries(GgmlOpsMiniMaxM3MsaGraphTest PRIVATE ggml-base)
    endif()
    target_link_libraries(GgmlOpsMiniMaxM3MsaGraphTest PRIVATE GgmlOps)
    add_test(NAME minimax-m3-msa-graph-cpu COMMAND GgmlOpsMiniMaxM3MsaGraphTest cpu)
    set_tests_properties(minimax-m3-msa-graph-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME minimax-m3-msa-graph-cuda COMMAND GgmlOpsMiniMaxM3MsaGraphTest cuda)
        set_tests_properties(minimax-m3-msa-graph-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 600)
    endif()
    # MiniMax-M3's DSpark drafter on eng/minimax-m3-fixture.py's small fixture
    # (model.gguf + dspark.gguf in $TS_MM3_FIXTURE_DIR, generated, not checked
    # in; skipped without it): speculative greedy == plain greedy, the drafter
    # context after verify rollbacks, rewinds, slot copies and batched decode.
    add_executable(GgmlOpsMiniMaxM3DsparkTest tests/minimax_m3_dspark_test.cpp)
    target_link_libraries(GgmlOpsMiniMaxM3DsparkTest PRIVATE GgmlOps)
    add_test(NAME minimax-m3-dspark-cpu COMMAND GgmlOpsMiniMaxM3DsparkTest cpu)
    set_tests_properties(minimax-m3-dspark-cpu PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME minimax-m3-dspark-cuda COMMAND GgmlOpsMiniMaxM3DsparkTest cuda)
        set_tests_properties(minimax-m3-dspark-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900)
        # Two GPUs: taps cross GPUs and prompts prefill pipelined (skipped on one GPU).
        add_test(NAME minimax-m3-dspark-cuda-multi COMMAND GgmlOpsMiniMaxM3DsparkTest cuda-multi)
        set_tests_properties(minimax-m3-dspark-cuda-multi PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900)
        # Four GPUs, 19 micro-batches: taps skip pipeline stages (skipped below four GPUs).
        add_test(NAME minimax-m3-dspark-cuda-multi4 COMMAND GgmlOpsMiniMaxM3DsparkTest cuda-multi4)
        set_tests_properties(minimax-m3-dspark-cuda-multi4 PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900)
    endif()

    add_executable(GgmlOpsAddmmVectorTest tests/addmm_vector_test.cpp)
    # One-shot return/exception injection is present only in native test builds.
    set_property(SOURCE ggml_ops_matmul.cpp APPEND PROPERTY COMPILE_DEFINITIONS TSG_GGML_TEST_HOOKS)
    add_executable(GgmlOpsAddmmQuantCacheRecoveryTest tests/addmm_quant_cache_recovery_test.cpp)
    target_link_libraries(GgmlOpsAddmmQuantCacheRecoveryTest PRIVATE GgmlOps)
    add_test(NAME addmm-quant-cache-recovery-cpu COMMAND GgmlOpsAddmmQuantCacheRecoveryTest cpu)
    set_tests_properties(addmm-quant-cache-recovery-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 RUN_SERIAL TRUE)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME addmm-quant-cache-recovery-cuda COMMAND GgmlOpsAddmmQuantCacheRecoveryTest cuda)
        set_tests_properties(addmm-quant-cache-recovery-cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 RUN_SERIAL TRUE)
    endif()
    add_executable(GgmlOpsResidentDeviceCopyTest tests/resident_device_copy_test.cpp)
    add_executable(GgmlOpsCudaLaunchPolicyTest tests/cuda_launch_policy_test.cpp ggml_ops_cuda_launch_policy.cpp)
    target_include_directories(GgmlOpsCudaLaunchPolicyTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach(TSG_PDL_CASE IN ITEMS unset 0 1 unaffected)
        add_test(NAME cuda-launch-policy-${TSG_PDL_CASE} COMMAND GgmlOpsCudaLaunchPolicyTest ${TSG_PDL_CASE})
    endforeach()
    add_executable(GgmlOpsQwen35ArenaScaleTest tests/qwen35_arena_scale_test.cpp)
    target_include_directories(GgmlOpsQwen35ArenaScaleTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsQwen35ArenaScaleTest PRIVATE GgmlOps ggml-base)
    add_executable(GgmlOpsDeviceCopyRangesTest tests/device_copy_ranges_test.cpp)
    target_link_libraries(GgmlOpsDeviceCopyRangesTest PRIVATE GgmlOps)
    foreach(TSG_DEVCOPY_BACKEND IN ITEMS cpu cuda metal)
        add_test(NAME device-copy-ranges-${TSG_DEVCOPY_BACKEND} COMMAND GgmlOpsDeviceCopyRangesTest ${TSG_DEVCOPY_BACKEND})
        set_tests_properties(device-copy-ranges-${TSG_DEVCOPY_BACKEND} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 RUN_SERIAL TRUE)
    endforeach()
    add_executable(GgmlOpsDFlashScaleTest tests/dflash_scale_test.cpp)
    target_link_libraries(GgmlOpsDFlashScaleTest PRIVATE GgmlOps ggml-base)
    add_test(NAME dflash-sidecar-scales-cpu COMMAND GgmlOpsDFlashScaleTest cpu)
    set_tests_properties(dflash-sidecar-scales-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME qwen35-arena-sidecar-scales-cuda COMMAND GgmlOpsQwen35ArenaScaleTest cuda default)
        set_tests_properties(qwen35-arena-sidecar-scales-cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)
        foreach(TSG_PDL_VALUE IN ITEMS 0 1)
            add_test(NAME cuda-launch-policy-device-${TSG_PDL_VALUE} COMMAND GgmlOpsQwen35ArenaScaleTest cuda pdl${TSG_PDL_VALUE})
            set_tests_properties(cuda-launch-policy-device-${TSG_PDL_VALUE} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30 RUN_SERIAL TRUE)
        endforeach()
        add_test(NAME dflash-sidecar-scales-cuda COMMAND GgmlOpsDFlashScaleTest cuda)
        set_tests_properties(dflash-sidecar-scales-cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        add_test(NAME qwen35-arena-sidecar-scales-metal COMMAND GgmlOpsQwen35ArenaScaleTest metal)
        set_tests_properties(qwen35-arena-sidecar-scales-metal PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)
        add_test(NAME dflash-sidecar-scales-metal COMMAND GgmlOpsDFlashScaleTest metal)
        set_tests_properties(dflash-sidecar-scales-metal PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
    add_executable(GgmlOpsMuseSparkProjectionPrecisionTest tests/muse_spark_projection_precision_test.cpp)
    add_executable(GgmlOpsMuseSparkAttentionPrecisionTest tests/muse_spark_attention_precision_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkAttentionPrecisionTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-attention-precision-cpu COMMAND GgmlOpsMuseSparkAttentionPrecisionTest cpu)
    set_tests_properties(muse-spark-attention-precision-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 RUN_SERIAL TRUE)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-attention-precision-cuda COMMAND GgmlOpsMuseSparkAttentionPrecisionTest cuda)
        add_test(NAME muse-spark-attention-precision-cuda-multi COMMAND GgmlOpsMuseSparkAttentionPrecisionTest cuda-multi)
        set_tests_properties(muse-spark-attention-precision-cuda muse-spark-attention-precision-cuda-multi
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240 RUN_SERIAL TRUE)
    endif()
    target_link_libraries(GgmlOpsMuseSparkProjectionPrecisionTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-projection-precision-cpu COMMAND GgmlOpsMuseSparkProjectionPrecisionTest cpu)
    set_tests_properties(muse-spark-projection-precision-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 RUN_SERIAL TRUE)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-projection-precision-cuda COMMAND GgmlOpsMuseSparkProjectionPrecisionTest cuda)
        add_test(NAME muse-spark-projection-precision-cuda-multi COMMAND GgmlOpsMuseSparkProjectionPrecisionTest cuda-multi)
        set_tests_properties(muse-spark-projection-precision-cuda muse-spark-projection-precision-cuda-multi
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 RUN_SERIAL TRUE)
    endif()
    target_link_libraries(GgmlOpsResidentDeviceCopyTest PRIVATE GgmlOps)
    add_test(NAME resident-device-copy-cpu COMMAND GgmlOpsResidentDeviceCopyTest cpu)
    set_tests_properties(resident-device-copy-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(GgmlOpsMuseSparkResidualTest tests/muse_spark_residual_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkResidualTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-residual-cpu COMMAND GgmlOpsMuseSparkResidualTest cpu)
    add_executable(GgmlOpsMuseSparkKv2DUploadTest tests/muse_spark_kv_2d_upload_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkKv2DUploadTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-kv-2d-cpu COMMAND GgmlOpsMuseSparkKv2DUploadTest cpu)
    set_tests_properties(muse-spark-kv-2d-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-kv-2d-cuda COMMAND GgmlOpsMuseSparkKv2DUploadTest cuda)
        add_test(NAME muse-spark-kv-2d-cuda-multi COMMAND GgmlOpsMuseSparkKv2DUploadTest multi)
        set_tests_properties(muse-spark-kv-2d-cuda muse-spark-kv-2d-cuda-multi
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    endif()
    add_executable(GgmlOpsMuseSparkNormGateTest tests/muse_spark_norm_gate_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkNormGateTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-norm-gate-cpu COMMAND GgmlOpsMuseSparkNormGateTest cpu)
    set_tests_properties(muse-spark-norm-gate-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-norm-gate-cuda COMMAND GgmlOpsMuseSparkNormGateTest cuda)
        add_test(NAME muse-spark-norm-gate-cuda-multi COMMAND GgmlOpsMuseSparkNormGateTest multi)
        set_tests_properties(muse-spark-norm-gate-cuda muse-spark-norm-gate-cuda-multi PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
    add_executable(GgmlOpsMuseSparkQKPrepTest tests/muse_spark_qk_prep_test.cpp)
    target_link_libraries(GgmlOpsMuseSparkQKPrepTest PRIVATE GgmlOps)
    add_test(NAME muse-spark-qk-prep-cpu COMMAND GgmlOpsMuseSparkQKPrepTest cpu)
    set_tests_properties(muse-spark-qk-prep-cpu PROPERTIES SKIP_RETURN_CODE 77)
    if (GGML_CUDA)
        add_test(NAME muse-spark-qk-prep-cuda COMMAND GgmlOpsMuseSparkQKPrepTest cuda)
        set_tests_properties(muse-spark-qk-prep-cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    set_tests_properties(muse-spark-residual-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME muse-spark-residual-cuda COMMAND GgmlOpsMuseSparkResidualTest cuda)
        set_tests_properties(muse-spark-residual-cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME resident-device-copy-cuda COMMAND GgmlOpsResidentDeviceCopyTest cuda)
        set_tests_properties(resident-device-copy-cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
    target_include_directories(GgmlOpsAddmmVectorTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsAddmmVectorTest PRIVATE GgmlOps)

    add_executable(GgmlOpsWanDitResidentCacheTest tests/wan_dit_resident_cache_test.cpp)
    target_include_directories(GgmlOpsWanDitResidentCacheTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include"
    )
    target_link_libraries(GgmlOpsWanDitResidentCacheTest PRIVATE GgmlOps)

    # Backend choice is immutable in the DLL. Generate a CPU explicit-attention
    # reference in a separate process, then compare GPU forwards against it.
    add_executable(GgmlOpsQwenImage21Test tests/qwen_image21_test.cpp)
    target_include_directories(GgmlOpsQwenImage21Test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(GgmlOpsQwenImage21Test PRIVATE GgmlOps)
    set(TSG_QWEN21_REFERENCE "${CMAKE_BINARY_DIR}/artifacts/qwen-image21-cpu-reference.bin")
    add_test(NAME qwen-image21-whole-graph-cpu
        COMMAND GgmlOpsQwenImage21Test cpu --write-reference "${TSG_QWEN21_REFERENCE}")
    set_tests_properties(qwen-image21-whole-graph-cpu PROPERTIES
        FIXTURES_SETUP qwen_image21_cpu_reference SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_executable(GgmlOpsQwenImage21VaeShortcutTest tests/qwen_image21_vae_shortcut_test.cpp)
    target_link_libraries(GgmlOpsQwenImage21VaeShortcutTest PRIVATE GgmlOps)
    add_test(NAME qwen-image21-vae-shortcuts-cpu COMMAND GgmlOpsQwenImage21VaeShortcutTest cpu)
    set_tests_properties(qwen-image21-vae-shortcuts-cpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    if (APPLE AND TENSORSHARP_GGML_NATIVE_ENABLE_METAL)
        add_test(NAME qwen-image21-whole-graph-metal
            COMMAND GgmlOpsQwenImage21Test metal --reference "${TSG_QWEN21_REFERENCE}")
        set_tests_properties(qwen-image21-whole-graph-metal PROPERTIES
            FIXTURES_REQUIRED qwen_image21_cpu_reference SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 300)
        add_test(NAME qwen-image21-vae-f32-convolution-metal COMMAND GgmlOpsQwenImage21VaeShortcutTest metal)
        set_tests_properties(qwen-image21-vae-f32-convolution-metal PROPERTIES
            ENVIRONMENT "TS_VAE_MPS_CONV=1;TS_WAN_VAE_MPS_CONV=1"
            SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
        add_test(NAME qwen-image21-vae-f32-convolution-metal-no-mps COMMAND GgmlOpsQwenImage21VaeShortcutTest metal)
        set_tests_properties(qwen-image21-vae-f32-convolution-metal-no-mps PROPERTIES
            ENVIRONMENT "TS_VAE_MPS_CONV=0"
            SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
    endif()
    if (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)
        add_test(NAME qwen-image21-whole-graph-vulkan
            COMMAND GgmlOpsQwenImage21Test vulkan --reference "${TSG_QWEN21_REFERENCE}")
        set_tests_properties(qwen-image21-whole-graph-vulkan PROPERTIES
            FIXTURES_REQUIRED qwen_image21_cpu_reference SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 300)
    endif()
    if (TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
        add_test(NAME qwen-image21-whole-graph-cuda
            COMMAND GgmlOpsQwenImage21Test cuda --reference "${TSG_QWEN21_REFERENCE}")
        set_tests_properties(qwen-image21-whole-graph-cuda PROPERTIES
            FIXTURES_REQUIRED qwen_image21_cpu_reference SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 300)
        add_test(NAME qwen-image21-vae-shortcuts-cuda COMMAND GgmlOpsQwenImage21VaeShortcutTest cuda)
        set_tests_properties(qwen-image21-vae-shortcuts-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 60)
        if (WIN32 AND TSG_HAVE_CUDNN)
            # Isolated optional-runtime failure fixtures. Never replace any real
            # cuDNN installation: the loader sees only these build artifacts.
            #   missing: reports this build's CUDA major, then lacks every cuDNN entry
            #            (an incomplete export table);
            #   cuda12:  exports every entry but reports a CUDA 12.8 build, which the
            #            loader must decline on the CUDA major before calling any of them.
            foreach(fixture IN ITEMS missing cuda12)
                if (fixture STREQUAL "missing")
                    set(fixture_target GgmlOpsMissingCudnnFixture)
                    set(fixture_source tests/cudnn_missing_symbols.cpp)
                else()
                    set(fixture_target GgmlOpsCuda12CudnnFixture)
                    set(fixture_source tests/cudnn_cuda12_build.cpp)
                endif()
                set(fixture_dir "${CMAKE_BINARY_DIR}/artifacts/qwen-image21-${fixture}-cudnn")
                add_library(${fixture_target} SHARED ${fixture_source})
                target_include_directories(${fixture_target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
                # Both fixtures are cudnn64_9: keep their import libraries apart too.
                set_target_properties(${fixture_target} PROPERTIES
                    OUTPUT_NAME cudnn64_9
                    RUNTIME_OUTPUT_DIRECTORY "${fixture_dir}"
                    ARCHIVE_OUTPUT_DIRECTORY "${fixture_dir}"
                    LIBRARY_OUTPUT_DIRECTORY "${fixture_dir}")
                foreach(sibling IN ITEMS cudnn_graph64_9 cudnn_ops64_9 cudnn_heuristic64_9 cudnn_engines_precompiled64_9)
                    add_custom_command(TARGET ${fixture_target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "$<TARGET_FILE:${fixture_target}>"
                            "$<TARGET_FILE_DIR:${fixture_target}>/${sibling}.dll")
                endforeach()
                add_dependencies(GgmlOpsQwenImage21VaeShortcutTest ${fixture_target})
                add_test(NAME qwen-image21-vae-${fixture}-cudnn-cuda
                    COMMAND GgmlOpsQwenImage21VaeShortcutTest cuda --${fixture}-cudnn-runtime)
                set_tests_properties(qwen-image21-vae-${fixture}-cudnn-cuda PROPERTIES
                    ENVIRONMENT "TS_CUDNN_DIR=$<TARGET_FILE_DIR:${fixture_target}>;TS_VAE_CUDNN_CONV=1;TS_WAN_VAE_CUDNN_CONV=1"
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 60)
            endforeach()
        endif()
    endif()
endif()

# Standalone optional NCCL row-gather contract: true F32 bit copies, uneven
# tilings, non-mutating validation failures and communicator lifecycle.
if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS AND TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND NOT WIN32)
    add_executable(GgmlOpsTpCollectiveGatherTest
        tests/tp_collective_gather_test.cpp ggml_ops_tp_probe.cu)
    target_compile_definitions(GgmlOpsTpCollectiveGatherTest PRIVATE TSG_GGML_USE_CUDA=1)
    target_include_directories(GgmlOpsTpCollectiveGatherTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/src")
    target_link_libraries(GgmlOpsTpCollectiveGatherTest PRIVATE
        ggml-cuda ggml-cpu ggml-base CUDA::cudart ${CMAKE_DL_LIBS})
    add_test(NAME cuda-f32-row-gather COMMAND GgmlOpsTpCollectiveGatherTest)
    set_tests_properties(cuda-f32-row-gather PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
endif()

# ggml-cuda's scratch pool on every visible GPU through GgmlOps, i.e. through
# the cuMemSetAccess guard that keeps 10+ peer-capable GPUs working.
if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS AND TENSORSHARP_GGML_NATIVE_ENABLE_CUDA AND
    CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT GGML_CUDA_NO_VMM)
    add_executable(GgmlOpsCudaVmmPeerGuardTest tests/cuda_vmm_peer_guard_test.cpp)
    target_include_directories(GgmlOpsCudaVmmPeerGuardTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml/include")
    target_link_libraries(GgmlOpsCudaVmmPeerGuardTest PRIVATE GgmlOps CUDA::cudart)
    if (TSG_USE_NCCL)
        # ggml then asks to share pool memory, so the wrapper must be what saved it.
        target_compile_definitions(GgmlOpsCudaVmmPeerGuardTest PRIVATE TSG_GGML_CUDA_NCCL=1)
    endif()
    add_test(NAME cuda-vmm-peer-guard COMMAND GgmlOpsCudaVmmPeerGuardTest)
    set_tests_properties(cuda-vmm-peer-guard PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 120)
endif()

# Test executables that link the static ggml-cuda themselves (instead of only
# GgmlOps) need the same guard, or they abort on 10+ peer-capable GPUs; it also
# supplies tsg::tp_cuda_respect_vmm_peer_limit to those compiling the TP probes.
if (TENSORSHARP_GGML_NATIVE_BUILD_TESTS AND TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)
    get_property(TSG_ALL_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(TSG_TARGET IN LISTS TSG_ALL_TARGETS)
        get_target_property(TSG_TARGET_TYPE ${TSG_TARGET} TYPE)
        get_target_property(TSG_TARGET_LIBS ${TSG_TARGET} LINK_LIBRARIES)
        if (TSG_TARGET_TYPE STREQUAL "EXECUTABLE" AND TSG_TARGET_LIBS AND
            ("ggml-cuda" IN_LIST TSG_TARGET_LIBS OR "ggml" IN_LIST TSG_TARGET_LIBS))
            tsg_add_cuda_vmm_peer_guard(${TSG_TARGET})
        endif()
    endforeach()
endif()
