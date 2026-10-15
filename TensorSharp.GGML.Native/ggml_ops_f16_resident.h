// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(TSG_GGML_USE_CUDA)
// Pinned ggml ffa4e8b8 default F16 CUDA arithmetic for dense packed matrices.
// Supported: Ampere/Ada/Hopper, K divisible by 64, original output rows divisible
// by 32. logical_columns selects MMVF/MMF/cuBLAS before token/row tiling.
// The query throws on unsupported hardware/shape/overflow. A successful MMVF
// query returns zero. All payload scratch (including cuBLAS workspace) belongs
// to the caller; metadata borrows that storage and the stream. It never cudaMallocs.
// For logical_columns > 16, preserving cuBLAS arithmetic requires complete
// logical-shape F16 weight/input/output scratch. This is fully included in the
// query and can impose a much larger minimum than the caller's active row tile.
std::size_t tsg_f16_resident_scratch_bytes(int device, int inner, int max_rows,
    int max_columns, int logical_columns, std::int64_t logical_rows);

// Returns 1 on success, 0 on failure. Create publishes ownership before acquiring
// cuBLAS metadata; destroy a nonnull state even when creation fails. Failed destroy
// retains state, scratch and stream ownership: retry before freeing those resources.
int tsg_f16_resident_create(int device, void* stream, void* scratch, std::size_t scratch_bytes,
    int inner, int max_rows, int max_columns, int logical_columns, std::int64_t logical_rows,
    void** state) noexcept;
int tsg_f16_resident_launch(void* state, const void* weights, const float* input,
    float* output, int rows, int columns) noexcept;
int tsg_f16_resident_destroy(void* state) noexcept;
const char* tsg_f16_resident_last_error() noexcept;
#endif
