// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>

// No pointer-key cache, captured graph, or host payload allocation is used.
// Input is packed [N,K] F32; each packed Q8_0/F16 tile is [rows,K]; output is
// packed [N,rows] F32. Every successful execute is synchronous.
// Create fixes the maximum token capacity. UploadInput replaces the active
// input with 1..capacity tokens without reallocating; output uses that active N.
// Calls on a handle must be serialized. Invalid upload arguments leave the old
// input usable; a CUDA upload failure invalidates the session until destruction.
// Failed Create can return an owned non-null handle: Destroy it even on failure.
// Failed Destroy retains ownership and must be retried; never release its budget.
#if defined(TSG_EXPORT)
#define TSG_Q8_STREAM_API TSG_EXPORT
#else
#define TSG_Q8_STREAM_API extern "C"
#endif
// Generic entry points support GGML_TYPE_F16 (1) and GGML_TYPE_Q8_0 (8).
// F16 rows contain K packed halves; Q8_0 requires K divisible by 32.
TSG_Q8_STREAM_API std::int64_t TSGgml_WeightStreamingPayloadBytes(int weight_type, std::int64_t inner, int max_rows, int columns);
TSG_Q8_STREAM_API int TSGgml_WeightStreamingCreate(int rank, int weight_type, std::int64_t inner, int max_rows, int columns,
    const float* input, std::int64_t capacity, void** handle);
// arithmetic: 0=full F32 activation precision, 1=resident CUDA arithmetic.
// Logical dimensions describe the original Linear, before either tiling axis.
TSG_Q8_STREAM_API std::int64_t TSGgml_WeightStreamingPayloadBytesEx(int weight_type, std::int64_t inner,
    int max_rows, int columns, int rank, int arithmetic, int logical_columns, std::int64_t logical_rows);
TSG_Q8_STREAM_API int TSGgml_WeightStreamingCreateEx(int rank, int weight_type, std::int64_t inner,
    int max_rows, int columns, int arithmetic, int logical_columns, std::int64_t logical_rows,
    const float* input, std::int64_t capacity, void** handle);
TSG_Q8_STREAM_API int TSGgml_WeightStreamingUploadInput(void* handle, const float* input, int columns);
TSG_Q8_STREAM_API int TSGgml_WeightStreamingExecute(void* handle, const void* weights, int rows, float* output);
TSG_Q8_STREAM_API int TSGgml_WeightStreamingDestroy(void* handle);
// A temporary complete logical matrix, populated from bounded host tiles.
// Supports Q8_0 with N>8 and F16 with N>16. No persistent weight cache is involved.
// Weight rows must be uploaded once, consecutively from zero. Input uploads
// start at zero and continue consecutively; starting at zero replaces the input.
// Project requires complete weights/input and computes original M/N once.
// Download writes packed [token_count,row_count] F32 after successful Project.
// Handles share the same destroy and recoverable ownership contract above.
TSG_Q8_STREAM_API std::int64_t TSGgml_ResidentWeightPayloadBytes(int rank, int weight_type,
    std::int64_t inner, int rows, int columns);
TSG_Q8_STREAM_API int TSGgml_ResidentWeightCreate(int rank, int weight_type, std::int64_t inner,
    int rows, int columns, std::int64_t capacity, void** handle);
TSG_Q8_STREAM_API int TSGgml_ResidentWeightUploadRows(void* handle, const void* rows, int first_row, int row_count);
TSG_Q8_STREAM_API int TSGgml_ResidentWeightUploadInput(void* handle, const float* input, int first_token, int token_count);
TSG_Q8_STREAM_API int TSGgml_ResidentWeightProject(void* handle);
TSG_Q8_STREAM_API int TSGgml_ResidentWeightDownload(void* handle, float* output,
    int first_token, int token_count, int first_row, int row_count);
// Retained Q8_0 ABI for existing callers.
TSG_Q8_STREAM_API std::int64_t TSGgml_Q8StreamingPayloadBytes(std::int64_t inner, int max_rows, int columns);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingCreate(int rank, std::int64_t inner, int max_rows, int columns,
    const float* input, std::int64_t capacity, void** handle);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingUploadInput(void* handle, const float* input, int columns);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingExecute(void* handle, const void* weights, int rows, float* output);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingDestroy(void* handle);
#undef TSG_Q8_STREAM_API
