#pragma once

// Gemma4 axial RoPE: each head splits into X/Y halves, each with rotate-half
// pairs. Every thread owns both elements of one pair, so the in-place update
// never races another pair. Angle tables contain X-quarter then Y-quarter.
extern "C" __global__ void ts_gemma4_vision_rope2d_f32(
    float* data, const float* cosine, const float* sine, int pairs, int heads, int head_dim)
{
    TS_PDL_ENTRY();
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    int half = head_dim / 2, quarter = head_dim / 4;
    int token = pair / (heads * half);
    int head = (pair / half) % heads;
    int channel = pair % half, axis = channel / quarter, index = channel % quarter;
    size_t base = ((size_t)token * heads + head) * head_dim + axis * half + index;
    int table = token * half + channel;
    float first = data[base], second = data[base + quarter];
    float c = cosine[table], s = sine[table];
    data[base] = first * c - second * s;
    data[base + quarter] = first * s + second * c;
}

extern "C" __global__ void ts_gemma4_vision_pad_f32(
    const float* input, float* output, int count, int input_columns, int output_columns)
{
    TS_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / output_columns, column = i % output_columns;
    output[i] = column < input_columns ? input[(size_t)row * input_columns + column] : 0.0f;
}
