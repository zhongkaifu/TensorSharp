// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_internal.h"
#include "ggml_ops_q8_streaming.h"
#include "ggml_ops_q8_precision.h"
#include <climits>
#if defined(TSG_GGML_USE_CUDA)
#include <cuda_runtime_api.h>
#include "ggml-cuda.h"
#endif

namespace {
#if defined(TSG_GGML_TEST_HOOKS)
std::atomic<int> injected_failures{0};
bool take_failure(int bit) {
    const int previous = injected_failures.fetch_and(~bit, std::memory_order_acq_rel);
    return (previous & bit) != 0;
}
#endif
struct layout {
    std::size_t input, weights, output, total, row_bytes;
};
layout sizes(std::int64_t inner, int rows, int columns) {
    if (inner <= 0 || inner > INT_MAX || inner % 32 != 0 || rows <= 0 ||
        columns <= 0 || columns > 65535 * 8)
        throw std::invalid_argument("Q8 streaming requires K divisible by 32, positive rows, and 1..524280 tokens.");
    const auto product = [](std::uint64_t a, std::uint64_t b) {
        if (a > (std::uint64_t(INT64_MAX) - 255) / b)
            throw std::overflow_error("Q8 streaming payload size overflow.");
        return a * b;
    };
    const auto aligned = [](std::uint64_t n) { return (n + 255) & ~std::uint64_t(255); };
    const auto row = product(std::uint64_t(inner / 32), 34);
    const auto input = aligned(product(product(std::uint64_t(inner), columns), 4));
    const auto weights = aligned(product(row, rows));
    const auto output = aligned(product(product(rows, columns), 4));
    if (input > std::uint64_t(INT64_MAX) - weights ||
        input + weights > std::uint64_t(INT64_MAX) - output ||
        input + weights + output > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Q8 streaming payload sum overflow.");
    return {std::size_t(input), std::size_t(weights), std::size_t(output),
        std::size_t(input + weights + output), std::size_t(row)};
}
#if defined(TSG_GGML_USE_CUDA)
struct session {
    int device = -1, inner = 0, max_rows = 0, columns = 0;
    layout bytes{};
    void* allocation = nullptr;
    cudaStream_t stream = nullptr;
    bool ready = false;
};
bool checked(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    tsg::set_last_error(std::string("Q8 streaming ") + operation + ": " + cudaGetErrorString(error));
    return false;
}
#endif
}

TSG_EXPORT std::int64_t TSGgml_Q8StreamingPayloadBytes(std::int64_t inner, int rows, int columns) {
    try { return std::int64_t(sizes(inner, rows, columns).total); }
    catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_Q8StreamingCreate(int rank, std::int64_t inner, int rows, int columns,
        const float* input, std::int64_t capacity, void** handle) {
    if (handle == nullptr) { tsg::set_last_error("Q8 streaming requires an output handle."); return 0; }
    *handle = nullptr;
    try {
        const auto bytes = sizes(inner, rows, columns);
        if (input == nullptr || capacity < 0 || std::uint64_t(capacity) < bytes.total)
            throw std::invalid_argument("Q8 streaming input is null or payload exceeds its reserved capacity.");
#if defined(TSG_GGML_USE_CUDA)
        if (tsg::g_backend_type != tsg::BACKEND_TYPE_CUDA || rank < 0 ||
            rank >= tsg::g_device_count.load(std::memory_order_acquire) ||
            tsg::dev(rank).backend == nullptr || !ggml_backend_is_cuda(tsg::dev(rank).backend))
            throw std::invalid_argument("Q8 streaming requires an initialized CUDA rank.");
        // Legacy singleton initialization leaves DeviceState.device_index=-1.
        // Resolve the actual backend device instead of assuming rank==ordinal
        // or requiring the multi-device initialization path.
        const auto backend_device = ggml_backend_get_device(tsg::dev(rank).backend);
        const auto registry = ggml_backend_cuda_reg();
        int ordinal = -1;
        for (std::size_t index = 0; index < ggml_backend_reg_dev_count(registry); ++index)
            if (ggml_backend_reg_dev_get(registry, index) == backend_device) { ordinal = int(index); break; }
        if (ordinal < 0) throw std::runtime_error("Q8 streaming cannot resolve the initialized CUDA device.");
        // Publish ownership before any CUDA resource is acquired. The caller
        // can retry destruction if either construction or rollback fails.
        auto* value = new session;
        *handle = value;
        value->device = ordinal;
        value->inner = int(inner); value->max_rows = rows; value->columns = columns; value->bytes = bytes;
        if (!checked(cudaSetDevice(value->device), "select device") ||
            !checked(cudaStreamCreateWithFlags(&value->stream, cudaStreamNonBlocking), "create stream") ||
            !checked(cudaMalloc(&value->allocation, bytes.total), "allocate bounded payload")) return 0;
#if defined(TSG_GGML_TEST_HOOKS)
        if (take_failure(1)) { tsg::set_last_error("Injected Q8 streaming failure after allocation."); return 0; }
#endif
        if (
            !checked(cudaMemcpy(value->allocation, input, std::size_t(inner) * columns * sizeof(float),
                cudaMemcpyHostToDevice), "upload input") ||
            !checked(cudaStreamSynchronize(nullptr), "finish input upload")) return 0;
        value->ready = true;
        return 1;
#else
        (void)rank;
        throw std::runtime_error("Q8 streaming requires a CUDA-enabled build.");
#endif
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_Q8StreamingExecute(void* handle, const void* weights, int rows, float* output) {
    try {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
    if (value == nullptr || !value->ready || weights == nullptr || output == nullptr ||
        rows <= 0 || rows > value->max_rows) {
        tsg::set_last_error("Q8 streaming received an invalid session, pointer, or tile row count."); return 0;
    }
    if (!checked(cudaSetDevice(value->device), "select device")) return 0;
    auto* device_weights = static_cast<char*>(value->allocation) + value->bytes.input;
    auto* device_output = reinterpret_cast<float*>(device_weights + value->bytes.weights);
    // Synchronous H2D consumes the caller's host tile before kernel submission;
    // even a later launch/sync failure cannot leave CUDA reading a reused tile.
    if (!checked(cudaMemcpy(device_weights, weights, value->bytes.row_bytes * rows,
            cudaMemcpyHostToDevice), "upload tile") ||
        !checked(cudaStreamSynchronize(nullptr), "finish tile upload")) { value->ready = false; return 0; }
    const auto launched = static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch(device_weights,
        value->allocation, device_output, value->inner, rows, value->columns,
        value->bytes.row_bytes, sizeof(float), std::size_t(value->inner) * sizeof(float), value->stream));
    const auto completed = cudaStreamSynchronize(value->stream);
    if (!checked(launched, "launch projection") || !checked(completed, "finish projection") ||
        !checked(cudaMemcpy(output, device_output, std::size_t(rows) * value->columns * sizeof(float),
            cudaMemcpyDeviceToHost), "download tile")) { value->ready = false; return 0; }
    return 1;
#else
    (void)handle; (void)weights; (void)rows; (void)output;
    tsg::set_last_error("Q8 streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_Q8StreamingDestroy(void* handle) {
    try {
    if (handle == nullptr) return 1;
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
    value->ready = false;
#if defined(TSG_GGML_TEST_HOOKS)
    if (take_failure(2)) { tsg::set_last_error("Injected Q8 streaming release failure (ownership retained)."); return 0; }
#endif
    if (!checked(cudaSetDevice(value->device), "select device for release")) return 0;
    if (value->stream && !checked(cudaStreamSynchronize(value->stream), "finish before release")) return 0;
    if (value->allocation) {
        if (!checked(cudaFree(value->allocation), "release payload")) return 0;
        value->allocation = nullptr;
    }
    if (value->stream) {
        if (!checked(cudaStreamDestroy(value->stream), "release stream")) return 0;
        value->stream = nullptr;
    }
    delete value;
    return 1;
#else
    tsg::set_last_error("Q8 streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

#if defined(TSG_GGML_TEST_HOOKS)
#define TSG_Q8_STREAM_TEST_EXPORT TSG_EXPORT
TSG_Q8_STREAM_TEST_EXPORT void TSGgml_TestQ8StreamingFailNext(int flags) {
    injected_failures.store(flags, std::memory_order_release);
}
#undef TSG_Q8_STREAM_TEST_EXPORT
#endif
