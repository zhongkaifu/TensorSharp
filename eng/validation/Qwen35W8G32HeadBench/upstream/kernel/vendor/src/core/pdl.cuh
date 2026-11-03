#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() { cudaTriggerProgrammaticLaunchCompletion(); }

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() { cudaGridDependencySynchronize(); }

// Entry wait for kernels whose inputs are entirely producer-dependent: block until the producer
// grid's writes are visible before any dependent read. No-op in launches without the
// programmatic-serialization attribute.
__device__ __forceinline__ void sync() { wait_for_dependencies(); }

// Exit publish for producer kernels: announces dependent launches once the calling thread has
// issued its global stores. Only writes issued before a CTA's publish are guaranteed visible to
// a dependent that waited, so publish belongs at kernel end; CTAs that return early without
// publishing still satisfy the launch gate through completion.
__device__ __forceinline__ void publish() { trigger_dependents(); }

} // namespace ninfer::pdl
