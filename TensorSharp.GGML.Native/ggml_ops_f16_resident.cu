// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_f16_resident.h"
#include "ggml-cuda/mmf.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
thread_local char last_error[512]{};
void error(const char* text) noexcept { std::snprintf(last_error, sizeof(last_error), "%s", text); }
bool checked(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return true;
    std::snprintf(last_error, sizeof(last_error), "F16 resident %s: %s", operation, cudaGetErrorString(status));
    return false;
}
bool checked(cublasStatus_t status, const char* operation) noexcept {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::snprintf(last_error, sizeof(last_error), "F16 resident %s: cuBLAS status %d", operation, int(status));
    return false;
}
enum class path { mmvf, mmf, cublas };
struct layout {
    path algorithm;
    std::size_t first, second, total, workspace;
    int padded_rows;
};
std::size_t aligned_product(std::size_t a, std::size_t b) {
    if (b == 0 || a > (std::size_t(INT64_MAX) - 255) / b)
        throw std::overflow_error("F16 resident scratch size overflow.");
    return (a * b + 255) & ~std::size_t(255);
}
layout sizes(int device, int inner, int rows, int columns, int logical_columns, std::int64_t logical_rows) {
    if (device < 0 || inner <= 0 || inner % 64 != 0 || rows <= 0 || rows > INT_MAX - 31 ||
        columns <= 0 || columns > 65535 * 8 || logical_columns < columns || logical_columns > 65535 * 8 ||
        logical_rows < rows || logical_rows > INT_MAX || logical_rows % 32 != 0)
        throw std::invalid_argument("F16 resident compatibility requires K divisible by 64, original rows divisible by 32, and valid bounded dimensions.");
    cudaDeviceProp properties{};
    if (!checked(cudaGetDeviceProperties(&properties, device), "query device")) throw std::runtime_error(last_error);
    const int cc = properties.major * 100 + properties.minor * 10;
    if (cc < 800 || cc > 900 || properties.warpSize != 32 || !ampere_mma_available(cc))
        throw std::invalid_argument("F16 resident compatibility currently supports NVIDIA Ampere/Ada/Hopper only.");
    const char* override_type = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (override_type && override_type[0] != '\0' && std::string(override_type) != "auto")
        throw std::invalid_argument("F16 resident compatibility requires the default GGML_CUDA_CUBLAS_COMPUTE_TYPE.");
    if (logical_columns == 1) return {path::mmvf, 0, 0, 0, 0, rows};
    const int padded = (rows + 31) / 32 * 32;
    if (logical_columns <= 16) {
        // The reused pinned MMA device kernel carries these packed strides and
        // products in int. Reject a representable byte layout whose indexing
        // would nevertheless overflow inside that implementation.
        if (std::int64_t(padded) * (inner / 2) > INT_MAX ||
            std::int64_t(columns) * (inner / 2) > INT_MAX ||
            std::int64_t(padded) * columns > INT_MAX)
            throw std::invalid_argument("F16 resident MMA tile exceeds the pinned kernel's 32-bit indexing range.");
        const auto weight = aligned_product(std::size_t(inner) * 2, padded);
        const auto output = aligned_product(std::size_t(columns) * 4, padded);
        if (weight > std::size_t(INT64_MAX) - output) throw std::overflow_error("F16 resident scratch sum overflow.");
        return {path::mmf, weight, output, weight + output, 0, padded};
    }
    // Match the unchanged upstream cuBLAS context workspace so algorithm choice
    // is not silently changed by a tiny workspace. Handle/context bookkeeping is
    // metadata, while this entire numerical workspace is caller-budgeted payload.
    const std::size_t workspace = cc >= 900 ? 32 * 1024 * 1024 : 4 * 1024 * 1024;
    // cuBLAS' default algorithm depends on both physical M and N. Even a
    // 256-row pad can choose different half-accumulation arithmetic from the
    // original dense matrix. Preserve the complete logical shape, filling
    // inactive rows/columns with zero; charge every byte before allocation.
    // This is deliberately a higher minimum than full-precision row streaming.
    const auto input = aligned_product(std::size_t(inner) * 2, logical_columns);
    const auto output = aligned_product(std::size_t(logical_rows) * 2, logical_columns);
    const auto weight = aligned_product(std::size_t(inner) * 2, std::size_t(logical_rows));
    if (input > std::size_t(INT64_MAX) - output || input + output > std::size_t(INT64_MAX) - weight ||
        input + output + weight > std::size_t(INT64_MAX) - workspace)
        throw std::overflow_error("F16 resident scratch sum overflow.");
    return {path::cublas, input, output, workspace + input + output + weight, workspace, int(logical_rows)};
}
struct state {
    int device, inner, max_rows, max_columns, logical_columns;
    std::int64_t logical_rows;
    cudaStream_t stream;
    void* scratch;
    layout bytes;
    cublasHandle_t blas = nullptr;
    bool ready = false;
};

// Dense, unfused subset of pinned mmvf.cu: F16 default arithmetic uses half2
// input conversion and half2 accumulation, followed by the original F32 reduction.
// One row is independent of output-row tiling. No allocations or upstream host
// launch wrappers with fatal CUDA_CHECK behavior are invoked.
__global__ void resident_mmvf(const half* weights, const float* input, float* output, int inner) {
    const int tid = threadIdx.x;
    const auto* x = reinterpret_cast<const half2*>(weights + std::size_t(blockIdx.x) * inner);
    const auto* y = reinterpret_cast<const float2*>(input);
    half2 sum = make_half2(0.0f, 0.0f);
    for (int pair = tid; pair < inner / 2; pair += blockDim.x) {
        const float2 value = y[pair];
        sum += x[pair] * make_half2(value.x, value.y);
    }
    float value = __low2float(sum) + __high2float(sum);
    value = warp_reduce_sum<32>(value);
    __shared__ float warp_sums[32];
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (tid % 32 == 0) warp_sums[tid / 32] = value;
    __syncthreads();
    if (tid < 32) {
        value = warp_reduce_sum<32>(warp_sums[tid]);
        if (tid == 0) output[blockIdx.x] = value;
    }
}

template<int Columns, int Warps>
void launch_mmf(const half* weights, const float* input, float* output, int inner, int rows, cudaStream_t stream) {
    const int shared_iteration = Warps * 16 * (32 + 4) * 4;
    const int shared_combine = ((Columns + 7) / 8 * 8) * (Warps * 32 + 4) * 4;
    const int shared = std::max(shared_iteration, shared_combine);
    // Reuse unchanged upstream MMA device implementation; packed two-dimensional
    // shapes have channel/sample ratios one and no IDs or fused epilogue.
    mul_mat_f<half2, 32, Columns, Warps, false><<<dim3(rows / 32, 1, 1), dim3(32, Warps, 1), shared, stream>>>(
        reinterpret_cast<const half2*>(weights), input, nullptr, output,
        inner / 2, Columns, 1, inner / 2, inner / 2, rows,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
}
template<int Columns>
void launch_mmf_warps(int warps, const half* weights, const float* input, float* output, int inner, int rows, cudaStream_t stream) {
#define TSG_MMF_WARP(N) case N: launch_mmf<Columns, N>(weights, input, output, inner, rows, stream); break
    switch (warps) {
        TSG_MMF_WARP(1); TSG_MMF_WARP(2); TSG_MMF_WARP(3); TSG_MMF_WARP(4);
        TSG_MMF_WARP(5); TSG_MMF_WARP(6); TSG_MMF_WARP(7); TSG_MMF_WARP(8);
        default: throw std::logic_error("Invalid F16 MMA warp count.");
    }
#undef TSG_MMF_WARP
}
void launch_mmf_columns(int columns, int warps, const half* weights, const float* input, float* output, int inner, int rows, cudaStream_t stream) {
#define TSG_MMF_COLUMN(N) case N: launch_mmf_warps<N>(warps, weights, input, output, inner, rows, stream); break
    switch (columns) {
        TSG_MMF_COLUMN(1); TSG_MMF_COLUMN(2); TSG_MMF_COLUMN(3); TSG_MMF_COLUMN(4);
        TSG_MMF_COLUMN(5); TSG_MMF_COLUMN(6); TSG_MMF_COLUMN(7); TSG_MMF_COLUMN(8);
        TSG_MMF_COLUMN(9); TSG_MMF_COLUMN(10); TSG_MMF_COLUMN(11); TSG_MMF_COLUMN(12);
        TSG_MMF_COLUMN(13); TSG_MMF_COLUMN(14); TSG_MMF_COLUMN(15); TSG_MMF_COLUMN(16);
        default: throw std::logic_error("Invalid F16 MMA token count.");
    }
#undef TSG_MMF_COLUMN
}
__global__ void convert_half(const float* input, half* output, std::size_t count) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x)
        output[i] = __float2half_rn(input[i]);
}
__global__ void compact_half_output(const half* input, float* output, int padded, int rows, int columns) {
    const std::size_t count = std::size_t(rows) * columns;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x)
        output[i] = __half2float(input[i / rows * padded + i % rows]);
}
__global__ void compact_output(const float* input, float* output, int padded, int rows, int columns) {
    const std::size_t count = std::size_t(rows) * columns;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x)
        output[i] = input[i / rows * padded + i % rows];
}
unsigned grid(std::size_t count) { return unsigned(std::min<std::size_t>((count + 255) / 256, 65535)); }
}

std::size_t tsg_f16_resident_scratch_bytes(int device, int inner, int max_rows, int max_columns,
    int logical_columns, std::int64_t logical_rows) {
    return sizes(device, inner, max_rows, max_columns, logical_columns, logical_rows).total;
}
const char* tsg_f16_resident_last_error() noexcept { return last_error; }

int tsg_f16_resident_create(int device, void* stream, void* scratch, std::size_t scratch_bytes,
    int inner, int max_rows, int max_columns, int logical_columns, std::int64_t logical_rows, void** output) noexcept {
    last_error[0] = '\0';
    if (!output) { error("F16 resident requires an ownership output."); return 0; }
    *output = nullptr;
    try {
        const auto bytes = sizes(device, inner, max_rows, max_columns, logical_columns, logical_rows);
        if (scratch_bytes < bytes.total || (bytes.total && (!scratch || std::uintptr_t(scratch) % 256 != 0)))
            throw std::invalid_argument("F16 resident scratch is null, misaligned or smaller than its reserved layout.");
        auto* value = new state{device, inner, max_rows, max_columns, logical_columns, logical_rows,
            static_cast<cudaStream_t>(stream), scratch, bytes};
        *output = value;
        if (!checked(cudaSetDevice(device), "select creation device")) return 0;
        if (bytes.algorithm == path::cublas) {
            if (!checked(cublasCreate(&value->blas), "create handle") ||
                !checked(cublasSetMathMode(value->blas, CUBLAS_TF32_TENSOR_OP_MATH), "set math policy") ||
                !checked(cublasSetStream(value->blas, value->stream), "set borrowed stream") ||
                !checked(cublasSetWorkspace(value->blas, scratch, bytes.workspace), "set reserved workspace")) return 0;
        }
        value->ready = true;
        return 1;
    } catch (const std::exception& ex) { error(ex.what()); return 0; }
    catch (...) { error("Unknown F16 resident creation failure."); return 0; }
}

int tsg_f16_resident_launch(void* handle, const void* weights, const float* input, float* output,
    int rows, int columns) noexcept {
    last_error[0] = '\0';
    auto* value = static_cast<state*>(handle);
    if (!value || !value->ready || !weights || !input || !output || rows <= 0 || rows > value->max_rows ||
        columns <= 0 || columns > value->max_columns) { error("Invalid F16 resident session or tile."); return 0; }
    try {
        if (!checked(cudaSetDevice(value->device), "select projection device")) return 0;
        const int inner = value->inner;
        const auto stream = value->stream;
        if (value->bytes.algorithm == path::mmvf) {
            int block = 32, iterations = (inner + 63) / 64;
            for (int candidate = 64; candidate <= 256; candidate += 32) {
                const int count = int((std::int64_t(inner) + 2 * candidate - 1) / (2 * candidate));
                if (count < iterations) { iterations = count; block = candidate; }
            }
            resident_mmvf<<<rows, block, 0, stream>>>(static_cast<const half*>(weights), input, output, inner);
            return checked(cudaGetLastError(), "launch MMVF") ? 1 : 0;
        }
        if (value->bytes.algorithm == path::mmf) {
            const int padded = (rows + 31) / 32 * 32;
            auto* padded_weights = static_cast<half*>(value->scratch);
            auto* padded_output = reinterpret_cast<float*>(static_cast<char*>(value->scratch) + value->bytes.first);
            if (!checked(cudaMemcpyAsync(padded_weights, weights, std::size_t(inner) * rows * 2, cudaMemcpyDeviceToDevice, stream), "stage padded MMA weights")) return 0;
            if (padded != rows && !checked(cudaMemsetAsync(padded_weights + std::size_t(inner) * rows, 0,
                std::size_t(inner) * (padded - rows) * 2, stream), "clear MMA row padding")) return 0;
            int warps = 1, iterations = (inner / 2 + 63) / 64;
            for (int candidate = 2; candidate <= 8; candidate++) {
                const int count = int((std::int64_t(inner) / 2 + 64 * candidate - 1) / (64 * candidate));
                if (count < iterations) { iterations = count; warps = candidate; }
            }
            launch_mmf_columns(columns, warps, padded_weights, input, padded_output, inner, padded, stream);
            if (!checked(cudaGetLastError(), "launch MMF")) return 0;
            compact_output<<<grid(std::size_t(rows) * columns), 256, 0, stream>>>(padded_output, output, padded, rows, columns);
            return checked(cudaGetLastError(), "compact MMF output") ? 1 : 0;
        }
        auto* half_input = reinterpret_cast<half*>(static_cast<char*>(value->scratch) + value->bytes.workspace);
        auto* half_output = reinterpret_cast<half*>(reinterpret_cast<char*>(half_input) + value->bytes.first);
        auto* padded_weights = reinterpret_cast<half*>(reinterpret_cast<char*>(half_output) + value->bytes.second);
        const int logical_rows = int(value->logical_rows), logical_columns = value->logical_columns;
        // Preserve the complete cuBLAS launch shape even for a short row tile
        // or the final token chunk. Clear every inactive source element: old
        // session input/weights must not affect either output or kernel safety.
        if (!checked(cudaMemcpyAsync(padded_weights, weights, std::size_t(inner) * rows * 2,
                cudaMemcpyDeviceToDevice, stream), "stage original-shape cuBLAS weights")) return 0;
        if (logical_rows > rows && !checked(cudaMemsetAsync(padded_weights + std::size_t(inner) * rows, 0,
                std::size_t(inner) * (logical_rows - rows) * 2, stream), "clear original-shape weight padding")) return 0;
        if (logical_columns > columns && !checked(cudaMemsetAsync(half_input + std::size_t(inner) * columns, 0,
                std::size_t(inner) * (logical_columns - columns) * 2, stream), "clear original-shape input padding")) return 0;
        convert_half<<<grid(std::size_t(inner) * columns), 256, 0, stream>>>(input, half_input, std::size_t(inner) * columns);
        if (!checked(cudaGetLastError(), "convert cuBLAS input")) return 0;
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        if (!checked(cublasGemmEx(value->blas, CUBLAS_OP_T, CUBLAS_OP_N, logical_rows, logical_columns, inner,
            &alpha, padded_weights, CUDA_R_16F, inner, half_input, CUDA_R_16F, inner,
            &beta, half_output, CUDA_R_16F, logical_rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "project cuBLAS F16")) return 0;
        compact_half_output<<<grid(std::size_t(rows) * columns), 256, 0, stream>>>(half_output, output, logical_rows, rows, columns);
        return checked(cudaGetLastError(), "compact cuBLAS output") ? 1 : 0;
    } catch (const std::exception& ex) { error(ex.what()); return 0; }
    catch (...) { error("Unknown F16 resident projection failure."); return 0; }
}

int tsg_f16_resident_destroy(void* handle) noexcept {
    last_error[0] = '\0';
    auto* value = static_cast<state*>(handle);
    if (!value) return 1;
    value->ready = false;
    if (!checked(cudaSetDevice(value->device), "select release device") ||
        !checked(cudaStreamSynchronize(value->stream), "complete work before metadata release")) return 0;
    if (value->blas) {
        if (!checked(cublasDestroy(value->blas), "destroy handle (ownership retained)")) return 0;
        value->blas = nullptr;
    }
    delete value;
    return 1;
}
