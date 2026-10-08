// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>

// No pointer-key cache, captured graph, or host payload allocation is used.
// Input is packed [N,K] F32; each packed Q8_0 tile is [rows,K]; output is
// packed [N,rows] F32. Every successful execute is synchronous.
// Failed Create can return an owned non-null handle: Destroy it even on failure.
// Failed Destroy retains ownership and must be retried; never release its budget.
#if defined(TSG_EXPORT)
#define TSG_Q8_STREAM_API TSG_EXPORT
#else
#define TSG_Q8_STREAM_API extern "C"
#endif
TSG_Q8_STREAM_API std::int64_t TSGgml_Q8StreamingPayloadBytes(std::int64_t inner, int max_rows, int columns);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingCreate(int rank, std::int64_t inner, int max_rows, int columns,
    const float* input, std::int64_t capacity, void** handle);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingExecute(void* handle, const void* weights, int rows, float* output);
TSG_Q8_STREAM_API int TSGgml_Q8StreamingDestroy(void* handle);
#undef TSG_Q8_STREAM_API
