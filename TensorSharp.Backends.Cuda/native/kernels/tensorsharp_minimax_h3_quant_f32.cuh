#pragma once

// H3-only bounded conversion for cuBLAS projections. The existing owned
// decoder returns F32 weights; activations never pass through F16 or q8_1.
extern "C" __global__ void ts_h3_weight_quant_f32(
    const uint8_t* weights, float* output, int type, int inner, int count)
{
    TS_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / inner;
    int column = i - row * inner;
    const uint8_t* source = weights + (size_t)row * qrow_bytes(type, inner);
    output[i] = qvalue_at(source, type, column);
}
