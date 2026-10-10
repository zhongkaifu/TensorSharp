// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>
#include <cstdint>
struct tsg_q8_resident_layout {
    std::size_t weight_bytes, output_bytes, scratch_bytes;
};
#if defined(TSG_GGML_USE_CUDA)
tsg_q8_resident_layout tsg_q8_resident_sizes(int device, int inner, int rows, int columns,
    int logical_columns, std::int64_t logical_rows);
int tsg_q8_resident_output_rows(int rows);
void tsg_q8_resident_launch(int device, const void* weights, const float* input, float* output,
    void* scratch, std::size_t scratch_bytes, int inner, int rows, int columns,
    int logical_columns, std::int64_t logical_rows, void* stream, bool input_quantized = false);
#endif
