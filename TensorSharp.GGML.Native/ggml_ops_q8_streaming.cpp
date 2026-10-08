// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_internal.h"
#include "ggml_ops_q8_streaming.h"
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_f16_streaming.h"
#include "ggml_ops_q8_resident_streaming.h"
#include "ggml_ops_f16_resident.h"
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
    std::size_t input, weights, output, total, row_bytes, scratch;
};
#if defined(TSG_GGML_USE_CUDA)
int resolve_device(int rank) {
    if (tsg::g_backend_type != tsg::BACKEND_TYPE_CUDA || rank < 0 ||
        rank >= tsg::g_device_count.load(std::memory_order_acquire) ||
        tsg::dev(rank).backend == nullptr || !ggml_backend_is_cuda(tsg::dev(rank).backend))
        throw std::invalid_argument("Weight streaming requires an initialized CUDA rank.");
    const auto backend_device = ggml_backend_get_device(tsg::dev(rank).backend);
    const auto registry = ggml_backend_cuda_reg();
    for (std::size_t index = 0; index < ggml_backend_reg_dev_count(registry); ++index)
        if (ggml_backend_reg_dev_get(registry, index) == backend_device) return int(index);
    throw std::runtime_error("Weight streaming cannot resolve the initialized CUDA device.");
}
#endif
layout sizes(int weight_type, std::int64_t inner, int rows, int columns,
        int rank = 0, int arithmetic = 0, int logical_columns = 0, std::int64_t logical_rows = 0) {
    if ((weight_type != GGML_TYPE_Q8_0 && weight_type != GGML_TYPE_F16) ||
        inner <= 0 || inner > INT_MAX || (weight_type == GGML_TYPE_Q8_0 && inner % 32 != 0) || rows <= 0 ||
        columns <= 0 || columns > 65535 * 8)
        throw std::invalid_argument("Weight streaming requires F16 or Q8_0, positive dimensions, Q8 K divisible by 32, and 1..524280 tokens.");
    if (arithmetic != 0 && arithmetic != 1) throw std::invalid_argument("Unknown weight streaming arithmetic policy.");
    const auto product = [](std::uint64_t a, std::uint64_t b) {
        if (a > (std::uint64_t(INT64_MAX) - 255) / b)
            throw std::overflow_error("Weight streaming payload size overflow.");
        return a * b;
    };
    const auto aligned = [](std::uint64_t n) { return (n + 255) & ~std::uint64_t(255); };
    const auto row = weight_type == GGML_TYPE_Q8_0 ? product(std::uint64_t(inner / 32), 34)
        : product(std::uint64_t(inner), 2);
    const auto input = aligned(product(product(std::uint64_t(inner), columns), 4));
    auto weights = aligned(product(row, rows));
    auto output = aligned(product(product(rows, columns), 4));
    std::uint64_t scratch = 0;
    if (arithmetic == 1) {
        if (logical_columns < columns || logical_rows < rows)
            throw std::invalid_argument("Resident CUDA arithmetic requires the complete logical Linear dimensions.");
#if defined(TSG_GGML_USE_CUDA)
        const int device = resolve_device(rank);
        if (weight_type == GGML_TYPE_Q8_0) {
            const auto compat = tsg_q8_resident_sizes(device, int(inner), rows, columns, logical_columns, logical_rows);
            weights = compat.weight_bytes; output = compat.output_bytes; scratch = compat.scratch_bytes;
        } else {
            scratch = aligned(tsg_f16_resident_scratch_bytes(device, int(inner), rows, columns, logical_columns, logical_rows));
        }
#else
        (void)rank;
        throw std::invalid_argument("Resident CUDA arithmetic requires a CUDA-enabled build.");
#endif
    }
    if (input > std::uint64_t(INT64_MAX) - weights ||
        input + weights > std::uint64_t(INT64_MAX) - output ||
        input + weights + output > std::uint64_t(INT64_MAX) - scratch ||
        input + weights + output + scratch > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Weight streaming payload sum overflow.");
    return {std::size_t(input), std::size_t(weights), std::size_t(output),
        std::size_t(input + weights + output + scratch), std::size_t(row), std::size_t(scratch)};
}
#if defined(TSG_GGML_USE_CUDA)
struct session {
    int device = -1, inner = 0, max_rows = 0, max_columns = 0, columns = 0;
    int weight_type = GGML_TYPE_Q8_0;
    int arithmetic = 0, logical_columns = 0;
    std::int64_t logical_rows = 0;
    layout bytes{};
    void* allocation = nullptr;
    cudaStream_t stream = nullptr;
    void* f16_state = nullptr;
    bool ready = false;
    bool full_shape = false, projected = false;
    int uploaded_rows = 0, uploaded_columns = 0;
};
bool checked(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    tsg::set_last_error(std::string("Weight streaming ") + operation + ": " + cudaGetErrorString(error));
    return false;
}
#endif
void validate_full_shape(int weight_type, int columns) {
    if (!((weight_type == GGML_TYPE_Q8_0 && columns > 8) || (weight_type == GGML_TYPE_F16 && columns > 16)))
        throw std::invalid_argument("Complete resident-shape streaming requires Q8_0 with N>8 or F16 with N>16.");
}

int create_session(int rank, int weight_type, std::int64_t inner, int rows, int columns,
        int arithmetic, int logical_columns, std::int64_t logical_rows,
        const float* input, std::int64_t capacity, void** handle, bool full_shape) {
    const auto bytes = sizes(weight_type, inner, rows, columns, rank, arithmetic, logical_columns, logical_rows);
    if ((!full_shape && input == nullptr) || capacity < 0 || std::uint64_t(capacity) < bytes.total)
        throw std::invalid_argument("Weight streaming input is null or payload exceeds its reserved capacity.");
#if defined(TSG_GGML_USE_CUDA)
    // Both session modes publish ownership before acquiring any CUDA resource.
    const int ordinal = resolve_device(rank);
    auto* value = new session;
    *handle = value;
    value->device = ordinal; value->weight_type = weight_type;
    value->arithmetic = arithmetic; value->logical_columns = logical_columns; value->logical_rows = logical_rows;
    value->inner = int(inner); value->max_rows = rows;
    value->max_columns = columns; value->columns = columns; value->bytes = bytes;
    value->full_shape = full_shape;
    if (!checked(cudaSetDevice(value->device), "select device") ||
        !checked(cudaStreamCreateWithFlags(&value->stream, cudaStreamNonBlocking), "create stream") ||
        !checked(cudaMalloc(&value->allocation, bytes.total), "allocate bounded payload")) return 0;
    if (arithmetic == 1 && weight_type == GGML_TYPE_F16 &&
        !tsg_f16_resident_create(ordinal, value->stream,
            static_cast<char*>(value->allocation) + bytes.input + bytes.weights + bytes.output,
            bytes.scratch, int(inner), rows, columns, logical_columns, logical_rows, &value->f16_state)) {
        tsg::set_last_error(tsg_f16_resident_last_error()); return 0;
    }
#if defined(TSG_GGML_TEST_HOOKS)
    if (take_failure(1)) { tsg::set_last_error("Injected Weight streaming failure after allocation."); return 0; }
#endif
    if (full_shape) {
        // Only padding requires initialization; consecutive host row uploads
        // replace every real weight row before the first projection.
        if (!checked(cudaMemsetAsync(static_cast<char*>(value->allocation) + bytes.input, 0, bytes.weights,
                value->stream), "initialize complete weight arena") ||
            !checked(cudaStreamSynchronize(value->stream), "finish complete weight initialization")) return 0;
    } else if (!checked(cudaMemcpy(value->allocation, input, std::size_t(inner) * columns * sizeof(float),
            cudaMemcpyHostToDevice), "upload input") ||
        !checked(cudaStreamSynchronize(nullptr), "finish input upload")) return 0;
    value->ready = true;
    return 1;
#else
    (void)handle;
    throw std::runtime_error("Weight streaming requires a CUDA-enabled build.");
#endif
}
}

TSG_EXPORT std::int64_t TSGgml_WeightStreamingPayloadBytes(int weight_type, std::int64_t inner, int rows, int columns) {
    try { return std::int64_t(sizes(weight_type, inner, rows, columns).total); }
    catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}
TSG_EXPORT std::int64_t TSGgml_WeightStreamingPayloadBytesEx(int weight_type, std::int64_t inner, int rows, int columns,
        int rank, int arithmetic, int logical_columns, std::int64_t logical_rows) {
    try { return std::int64_t(sizes(weight_type, inner, rows, columns, rank, arithmetic, logical_columns, logical_rows).total); }
    catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_WeightStreamingCreate(int rank, int weight_type, std::int64_t inner, int rows, int columns,
        const float* input, std::int64_t capacity, void** handle) {
    return TSGgml_WeightStreamingCreateEx(rank, weight_type, inner, rows, columns, 0, columns, rows, input, capacity, handle);
}
TSG_EXPORT int TSGgml_WeightStreamingCreateEx(int rank, int weight_type, std::int64_t inner, int rows, int columns,
        int arithmetic, int logical_columns, std::int64_t logical_rows,
        const float* input, std::int64_t capacity, void** handle) {
    if (handle == nullptr) { tsg::set_last_error("Weight streaming requires an output handle."); return 0; }
    *handle = nullptr;
    try {
        return create_session(rank, weight_type, inner, rows, columns, arithmetic, logical_columns,
            logical_rows, input, capacity, handle, false);
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_WeightStreamingUploadInput(void* handle, const float* input, int columns) {
    try {
#if defined(TSG_GGML_USE_CUDA)
        auto* value = static_cast<session*>(handle);
        // Reject arguments before touching the previously uploaded input. A
        // failed CUDA operation, however, can leave partially replaced bytes.
        if (value == nullptr || !value->ready || value->full_shape || input == nullptr ||
            columns <= 0 || columns > value->max_columns) {
            tsg::set_last_error("Weight streaming received an invalid session, input, or token count."); return 0;
        }
        value->ready = false;
        if (!checked(cudaSetDevice(value->device), "select device for input upload") ||
            !checked(cudaMemcpy(value->allocation, input, std::size_t(value->inner) * columns * sizeof(float),
                cudaMemcpyHostToDevice), "upload replacement input") ||
            !checked(cudaStreamSynchronize(nullptr), "finish replacement input upload")) return 0;
#if defined(TSG_GGML_TEST_HOOKS)
        if (take_failure(4)) { tsg::set_last_error("Injected Weight streaming failure after replacement input upload."); return 0; }
#endif
        // The payload layout and allocation remain at their original capacity;
        // only the valid column count changes after the upload has completed.
        value->columns = columns;
        value->ready = true;
        return 1;
#else
        (void)handle; (void)input; (void)columns;
        tsg::set_last_error("Weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT int TSGgml_WeightStreamingExecute(void* handle, const void* weights, int rows, float* output) {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
#endif
    try {
#if defined(TSG_GGML_USE_CUDA)
    if (value == nullptr || !value->ready || value->full_shape || weights == nullptr || output == nullptr ||
        rows <= 0 || rows > value->max_rows) {
        tsg::set_last_error("Weight streaming received an invalid session, pointer, or tile row count."); return 0;
    }
    value->ready = false;
    if (!checked(cudaSetDevice(value->device), "select device")) return 0;
    auto* device_weights = static_cast<char*>(value->allocation) + value->bytes.input;
    auto* device_output = reinterpret_cast<float*>(device_weights + value->bytes.weights);
    auto* scratch = reinterpret_cast<char*>(device_output) + value->bytes.output;
    if (value->arithmetic == 1 && value->weight_type == GGML_TYPE_Q8_0 &&
        !checked(cudaMemsetAsync(device_weights, 0, value->bytes.weights, nullptr), "clear padded weight tile")) return 0;
    // Synchronous H2D consumes the caller's host tile before kernel submission;
    // even a later launch/sync failure cannot leave CUDA reading a reused tile.
    if (!checked(cudaMemcpy(device_weights, weights, value->bytes.row_bytes * rows,
            cudaMemcpyHostToDevice), "upload tile") ||
        !checked(cudaStreamSynchronize(nullptr), "finish tile upload")) { value->ready = false; return 0; }
    cudaError_t launched = cudaSuccess;
    int output_rows = rows;
    if (value->arithmetic == 1 && value->weight_type == GGML_TYPE_Q8_0) {
        output_rows = tsg_q8_resident_output_rows(rows);
        tsg_q8_resident_launch(value->device, device_weights, static_cast<const float*>(value->allocation), device_output,
            scratch, value->bytes.scratch, value->inner, rows, value->columns, value->logical_columns, value->logical_rows, value->stream);
    } else if (value->arithmetic == 1) {
        if (!tsg_f16_resident_launch(value->f16_state, device_weights, static_cast<const float*>(value->allocation),
            device_output, rows, value->columns)) {
            tsg::set_last_error(tsg_f16_resident_last_error()); return 0;
        }
    } else {
    launched = static_cast<cudaError_t>(value->weight_type == GGML_TYPE_Q8_0 ? tsg_matmul_q8_cuda_launch(device_weights,
        value->allocation, device_output, value->inner, rows, value->columns,
        value->bytes.row_bytes, sizeof(float), std::size_t(value->inner) * sizeof(float), value->stream)
        : tsg_matmul_f16_cuda_launch(device_weights, value->allocation, device_output,
            value->inner, rows, value->columns, value->stream));
    }
    const auto completed = cudaStreamSynchronize(value->stream);
    if (!checked(launched, "launch projection") || !checked(completed, "finish projection") ||
        !checked(cudaMemcpy2D(output, std::size_t(rows) * sizeof(float), device_output, std::size_t(output_rows) * sizeof(float),
            std::size_t(rows) * sizeof(float), value->columns, cudaMemcpyDeviceToHost), "download tile")) return 0;
    value->ready = true;
    return 1;
#else
    (void)handle; (void)weights; (void)rows; (void)output;
    tsg::set_last_error("Weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) {
#if defined(TSG_GGML_USE_CUDA)
        if (value) value->ready = false;
#endif
        tsg::set_last_error(error.what()); return 0;
    }
}

TSG_EXPORT std::int64_t TSGgml_ResidentWeightPayloadBytes(int rank, int weight_type,
        std::int64_t inner, int rows, int columns) {
    try {
        validate_full_shape(weight_type, columns);
        return std::int64_t(sizes(weight_type, inner, rows, columns, rank, 1, columns, rows).total);
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}
TSG_EXPORT int TSGgml_ResidentWeightCreate(int rank, int weight_type, std::int64_t inner,
        int rows, int columns, std::int64_t capacity, void** handle) {
    if (!handle) { tsg::set_last_error("Resident weight streaming requires an output handle."); return 0; }
    *handle = nullptr;
    try {
        validate_full_shape(weight_type, columns);
        return create_session(rank, weight_type, inner, rows, columns, 1, columns, rows,
            nullptr, capacity, handle, true);
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}
TSG_EXPORT int TSGgml_ResidentWeightUploadRows(void* handle, const void* rows, int first_row, int row_count) {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
#endif
    try {
#if defined(TSG_GGML_USE_CUDA)
        if (!value || !value->full_shape || !value->ready || !rows || row_count <= 0 || first_row < 0 || first_row > value->max_rows ||
            first_row != value->uploaded_rows || row_count > value->max_rows - first_row) {
            tsg::set_last_error("Resident weight rows must cover the complete matrix once, consecutively from zero."); return 0;
        }
        value->ready = false;
        auto* destination = static_cast<char*>(value->allocation) + value->bytes.input +
            std::size_t(first_row) * value->bytes.row_bytes;
        if (!checked(cudaSetDevice(value->device), "select device for complete weight upload") ||
            !checked(cudaMemcpy(destination, rows, std::size_t(row_count) * value->bytes.row_bytes,
                cudaMemcpyHostToDevice), "upload complete weight rows") ||
            !checked(cudaStreamSynchronize(nullptr), "finish complete weight row upload")) return 0;
#if defined(TSG_GGML_TEST_HOOKS)
        if (take_failure(8)) { tsg::set_last_error("Injected Weight streaming failure after complete weight upload."); return 0; }
#endif
        value->uploaded_rows += row_count;
        value->ready = true;
        return 1;
#else
        (void)handle; (void)rows; (void)first_row; (void)row_count;
        tsg::set_last_error("Resident weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) {
#if defined(TSG_GGML_USE_CUDA)
        if (value) value->ready = false;
#endif
        tsg::set_last_error(error.what()); return 0;
    }
}
TSG_EXPORT int TSGgml_ResidentWeightUploadInput(void* handle, const float* input, int first_token, int token_count) {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
#endif
    try {
#if defined(TSG_GGML_USE_CUDA)
        if (!value || !value->full_shape || !value->ready || !input || token_count <= 0 || first_token < 0 ||
            first_token > value->max_columns || token_count > value->max_columns - first_token ||
            (first_token != 0 && first_token != value->uploaded_columns)) {
            tsg::set_last_error("Resident input tokens must cover the original N consecutively from zero."); return 0;
        }
        value->ready = false; value->projected = false;
        if (first_token == 0) value->uploaded_columns = 0;
        auto* destination = static_cast<float*>(value->allocation) + std::size_t(first_token) * value->inner;
        if (!checked(cudaSetDevice(value->device), "select device for complete input upload") ||
            !checked(cudaMemcpy(destination, input, std::size_t(token_count) * value->inner * sizeof(float),
                cudaMemcpyHostToDevice), "upload complete input tokens") ||
            !checked(cudaStreamSynchronize(nullptr), "finish complete input upload")) return 0;
#if defined(TSG_GGML_TEST_HOOKS)
        if (take_failure(4)) { tsg::set_last_error("Injected Weight streaming failure after replacement input upload."); return 0; }
#endif
        value->uploaded_columns += token_count;
        value->ready = true;
        return 1;
#else
        (void)handle; (void)input; (void)first_token; (void)token_count;
        tsg::set_last_error("Resident weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) {
#if defined(TSG_GGML_USE_CUDA)
        if (value) value->ready = false;
#endif
        tsg::set_last_error(error.what()); return 0;
    }
}
TSG_EXPORT int TSGgml_ResidentWeightProject(void* handle) {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
#endif
    try {
#if defined(TSG_GGML_USE_CUDA)
        if (!value || !value->full_shape || !value->ready || value->uploaded_rows != value->max_rows ||
            value->uploaded_columns != value->max_columns) {
            tsg::set_last_error("Resident projection requires every original weight row and input token."); return 0;
        }
        if (value->projected) return 1;
        value->ready = false;
        if (!checked(cudaSetDevice(value->device), "select device for complete projection")) return 0;
        auto* weights = static_cast<char*>(value->allocation) + value->bytes.input;
        auto* output = reinterpret_cast<float*>(weights + value->bytes.weights);
        auto* scratch = reinterpret_cast<char*>(output) + value->bytes.output;
        // Original rows AND tokens determine stream-K partitions/fixup order.
        // Upload/download chunk sizes never enter the arithmetic dispatch.
        if (value->weight_type == GGML_TYPE_Q8_0) {
            tsg_q8_resident_launch(value->device, weights, static_cast<const float*>(value->allocation), output,
                scratch, value->bytes.scratch, value->inner, value->max_rows, value->max_columns,
                value->max_columns, value->max_rows, value->stream);
        } else if (!tsg_f16_resident_launch(value->f16_state, weights, static_cast<const float*>(value->allocation),
                output, value->max_rows, value->max_columns)) {
            tsg::set_last_error(tsg_f16_resident_last_error()); return 0;
        }
        if (!checked(cudaStreamSynchronize(value->stream), "finish complete projection")) return 0;
        value->projected = true; value->ready = true;
        return 1;
#else
        (void)handle;
        tsg::set_last_error("Resident weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) {
#if defined(TSG_GGML_USE_CUDA)
        if (value) value->ready = false;
#endif
        tsg::set_last_error(error.what()); return 0;
    }
}
TSG_EXPORT int TSGgml_ResidentWeightDownload(void* handle, float* output,
        int first_token, int token_count, int first_row, int row_count) {
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
#endif
    try {
#if defined(TSG_GGML_USE_CUDA)
        if (!value || !value->full_shape || !value->ready || !value->projected || !output ||
            first_token < 0 || first_token > value->max_columns || token_count <= 0 || token_count > value->max_columns - first_token ||
            first_row < 0 || first_row > value->max_rows || row_count <= 0 || row_count > value->max_rows - first_row) {
            tsg::set_last_error("Resident download requires a completed projection and an in-range output rectangle."); return 0;
        }
        value->ready = false;
        const int stride = value->weight_type == GGML_TYPE_Q8_0
            ? tsg_q8_resident_output_rows(value->max_rows) : value->max_rows;
        const auto* source = reinterpret_cast<const float*>(static_cast<const char*>(value->allocation) +
            value->bytes.input + value->bytes.weights) + std::size_t(first_token) * stride + first_row;
        if (!checked(cudaSetDevice(value->device), "select device for complete output download") ||
            !checked(cudaMemcpy2D(output, std::size_t(row_count) * sizeof(float), source, std::size_t(stride) * sizeof(float),
                std::size_t(row_count) * sizeof(float), token_count, cudaMemcpyDeviceToHost), "download complete output rectangle")) return 0;
        value->ready = true;
        return 1;
#else
        (void)handle; (void)output; (void)first_token; (void)token_count; (void)first_row; (void)row_count;
        tsg::set_last_error("Resident weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) {
#if defined(TSG_GGML_USE_CUDA)
        if (value) value->ready = false;
#endif
        tsg::set_last_error(error.what()); return 0;
    }
}

TSG_EXPORT int TSGgml_WeightStreamingDestroy(void* handle) {
    try {
    if (handle == nullptr) return 1;
#if defined(TSG_GGML_USE_CUDA)
    auto* value = static_cast<session*>(handle);
    value->ready = false;
#if defined(TSG_GGML_TEST_HOOKS)
    if (take_failure(2)) { tsg::set_last_error("Injected Weight streaming release failure (ownership retained)."); return 0; }
#endif
    if (!checked(cudaSetDevice(value->device), "select device for release")) return 0;
    if (value->stream && !checked(cudaStreamSynchronize(value->stream), "finish before release")) return 0;
    if (value->f16_state) {
        if (!tsg_f16_resident_destroy(value->f16_state)) { tsg::set_last_error(tsg_f16_resident_last_error()); return 0; }
        value->f16_state = nullptr;
    }
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
    tsg::set_last_error("Weight streaming requires a CUDA-enabled build."); return 0;
#endif
    } catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT std::int64_t TSGgml_Q8StreamingPayloadBytes(std::int64_t inner, int rows, int columns) {
    return TSGgml_WeightStreamingPayloadBytes(GGML_TYPE_Q8_0, inner, rows, columns);
}
TSG_EXPORT int TSGgml_Q8StreamingCreate(int rank, std::int64_t inner, int rows, int columns,
        const float* input, std::int64_t capacity, void** handle) {
    return TSGgml_WeightStreamingCreate(rank, GGML_TYPE_Q8_0, inner, rows, columns, input, capacity, handle);
}
TSG_EXPORT int TSGgml_Q8StreamingUploadInput(void* handle, const float* input, int columns) {
    return TSGgml_WeightStreamingUploadInput(handle, input, columns);
}
TSG_EXPORT int TSGgml_Q8StreamingExecute(void* handle, const void* weights, int rows, float* output) {
    return TSGgml_WeightStreamingExecute(handle, weights, rows, output);
}
TSG_EXPORT int TSGgml_Q8StreamingDestroy(void* handle) {
    return TSGgml_WeightStreamingDestroy(handle);
}

#if defined(TSG_GGML_TEST_HOOKS)
#define TSG_Q8_STREAM_TEST_EXPORT TSG_EXPORT
TSG_Q8_STREAM_TEST_EXPORT void TSGgml_TestQ8StreamingFailNext(int flags) {
    injected_failures.store(flags, std::memory_order_release);
}
#undef TSG_Q8_STREAM_TEST_EXPORT
#endif
