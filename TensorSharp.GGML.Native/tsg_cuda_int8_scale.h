// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Reading a product an int8 GEMM left unscaled (tsg_cuda_q4k_gemm.cu).
//
// The int8 route multiplies two operands that were each divided by a scale and rounded:
// row n of the weight by rowpeak[n] / 127, token t's activations by tokpeak[t] / 127.
// The int32 sums come back as F32 once each is multiplied by both scales. That is a
// pass over the whole product, and for the widest projections it costs as much as the
// narrowing of the next operand. Every product in MiniMax-H3's fused graphs is read by
// exactly one kernel of ours, so the GEMM can leave the int32 sums where the F32 would
// be, and the reader applies the scales as it loads each element: the same two
// multiplications, in the same order and rounding, so the values are those the pass
// would have stored.
//
// The GEMM registers such a product by its device address; the reader asks for it
// (tsg_cuda_int8_pending), reads through tsg_int8_product, and says when it is done
// (tsg_cuda_int8_consumed). A reader that cannot take int32 calls
// tsg_cuda_int8_materialize, which runs the pass in place. ggml_ops_minimax_h3.cpp
// decides per graph which products may be left this way (h3_deferrable).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// The scale of token t's activations times 1 / 127^2, from its peak's F32 bits.
__device__ __forceinline__ float tsg_int8_token_scale(uint32_t tokpeakBits)
{
    return __uint_as_float(tokpeakBits) * (1.0f / (127.0f * 127.0f));
}

// Element i of one token's column of a product: the F32 stored there or, with
// `rowpeak`, the int32 sum times its two scales (`sx` from tsg_int8_token_scale).
// `first` is the row of element 0, for a view that starts inside the column.
__device__ __forceinline__ float tsg_int8_product(const float* column, long long i, const uint32_t* rowpeak,
                                                  long long first, float sx)
{
    if (rowpeak == nullptr) return column[i];
    const float acc = (float) reinterpret_cast<const int32_t*>(column)[i];
    return __fmul_rn(acc, __fmul_rn(__uint_as_float(rowpeak[first + i]), sx));
}

extern "C" {
// Whether the product at `data` holds int32 sums. When it does, *rowpeak has `*rows`
// entries, *tokpeak one per token, and a token's column is `*ld` elements long.
bool tsg_cuda_int8_pending(const void* data, const uint32_t** rowpeak, const uint32_t** tokpeak, int64_t* rows,
                           int64_t* ld);
// The reader of the product at `data` has finished with it (its kernels have run).
void tsg_cuda_int8_consumed(const void* data);
// Scales the product at `data` to F32 in place if it holds int32 sums. False only when
// that pass failed.
bool tsg_cuda_int8_materialize(const void* data);
// The number of products left unscaled and never read: zero after a correct graph.
// Forgets them.
int tsg_cuda_int8_forget_pending(void);
}
