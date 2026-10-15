// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#if defined(TSG_GGML_USE_CUDA)
// Packed F16 [rows,K] x F32 [tokens,K], output packed F32 [tokens,rows].
// Returns CUDA launch status; storage and synchronization belong to the caller.
int tsg_matmul_f16_cuda_launch(const void* weights, const void* input, float* output,
    int inner, int rows, int columns, void* stream);
#endif
