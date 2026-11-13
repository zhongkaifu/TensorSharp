// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// EmbeddingGemma 2 bidirectional GQA. All score and value accumulation is F32.
// Inputs and output are [batch, sequence, heads, head_dim], with grouped K/V
// heads. Optional per-query [start,end) bounds isolate packed input sequences.

template<int D, int GROUP, int QROWS, int KC>
__device__ __forceinline__ void ts_embedding_attention_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const int* __restrict__ bounds,
    float* __restrict__ output,
    int batch, int q_heads, int kv_heads, int seq, int radius, float scale)
{
    constexpr int ROWS = GROUP * QROWS;
    constexpr int COLS = D / 256;
    const int b = blockIdx.z;
    const int kv_head = blockIdx.x;
    const int q0 = blockIdx.y * QROWS;
    if (b >= batch || kv_head >= kv_heads || q0 >= seq || q_heads != GROUP * kv_heads) return;
    const int count = min(QROWS, seq - q0);
    const int first_head = kv_head * GROUP;
    extern __shared__ float embedding_workspace[];
    float* q_s = embedding_workspace;
    float* scores = q_s + ROWS * D;
    float* maxima = scores + ROWS * KC;
    float* sums = maxima + ROWS;
    float* corrections = sums + ROWS;
    for (int i = threadIdx.x; i < ROWS * D; i += blockDim.x) {
        const int row = i / D, qi = row / GROUP, h = row % GROUP, d = i % D;
        q_s[i] = qi < count ? query[((size_t(b) * seq + q0 + qi) * q_heads + first_head + h) * D + d] : 0.0f;
    }
    if (threadIdx.x < ROWS) {
        maxima[threadIdx.x] = -FLT_MAX;
        sums[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    int lo = radius > 0 ? max(0, q0 - radius) : 0;
    int hi = radius > 0 ? min(seq - 1, q0 + count - 1 + radius) : seq - 1;
    if (bounds) {
        lo = max(lo, max(0, bounds[(size_t(b) * seq + q0) * 2]));
        hi = min(hi, min(seq - 1, bounds[(size_t(b) * seq + q0 + count - 1) * 2 + 1] - 1));
    }
    float accumulator[ROWS * COLS];
#pragma unroll
    for (int i = 0; i < ROWS * COLS; ++i) accumulator[i] = 0.0f;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int k0 = lo; k0 <= hi; k0 += KC) {
        const int chunk = min(KC, hi - k0 + 1);
        for (int task = warp; task < chunk * QROWS; task += warps) {
            const int ki = task % chunk, qi = task / chunk, kpos = k0 + ki;
            const float* k = key + ((size_t(b) * seq + kpos) * kv_heads + kv_head) * D;
            const float* q = q_s + qi * GROUP * D;
            float dots[GROUP] = {};
#pragma unroll
            for (int d2 = lane; d2 < D / 2; d2 += 32) {
                const float2 kv = *reinterpret_cast<const float2*>(k + d2 * 2);
#pragma unroll
                for (int h = 0; h < GROUP; ++h) {
                    dots[h] = fmaf(q[h * D + d2 * 2], kv.x, dots[h]);
                    dots[h] = fmaf(q[h * D + d2 * 2 + 1], kv.y, dots[h]);
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
#pragma unroll
                for (int h = 0; h < GROUP; ++h) dots[h] += __shfl_down_sync(0xFFFFFFFF, dots[h], offset);
            if (lane == 0) {
                bool allowed = qi < count && (radius == 0 || abs(kpos - (q0 + qi)) <= radius);
                if (bounds && qi < count) {
                    const size_t index = (size_t(b) * seq + q0 + qi) * 2;
                    allowed = allowed && kpos >= bounds[index] && kpos < bounds[index + 1];
                }
#pragma unroll
                for (int h = 0; h < GROUP; ++h) scores[(qi * GROUP + h) * KC + ki] = allowed ? dots[h] * scale : -FLT_MAX;
            }
        }
        __syncthreads();
        for (int row = warp; row < ROWS; row += warps) {
            constexpr int SLOTS = KC / 32;
            float values[SLOTS], maximum = -FLT_MAX;
#pragma unroll
            for (int i = 0; i < SLOTS; ++i) {
                const int column = lane + i * 32;
                values[i] = column < chunk ? scores[row * KC + column] : -FLT_MAX;
                maximum = fmaxf(maximum, values[i]);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFF, maximum, offset));
            const float updated = fmaxf(maxima[row], maximum);
            float sum = 0;
#pragma unroll
            for (int i = 0; i < SLOTS; ++i) {
                const int column = lane + i * 32;
                const float probability = column < chunk && updated > -FLT_MAX ? expf(values[i] - updated) : 0.0f;
                if (column < chunk) scores[row * KC + column] = probability;
                sum += probability;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
            if (lane == 0) {
                const float correction = expf(maxima[row] - updated);
                corrections[row] = correction;
                maxima[row] = updated;
                sums[row] = sums[row] * correction + sum;
            }
        }
        __syncthreads();
#pragma unroll
        for (int row = 0; row < ROWS; ++row)
#pragma unroll
            for (int column = 0; column < COLS; ++column) accumulator[row * COLS + column] *= corrections[row];
        for (int ki = 0; ki < chunk; ++ki) {
            const float* v = value + ((size_t(b) * seq + k0 + ki) * kv_heads + kv_head) * D;
            float values[COLS];
#pragma unroll
            for (int column = 0; column < COLS; ++column) values[column] = v[threadIdx.x + column * 256];
#pragma unroll
            for (int row = 0; row < ROWS; ++row)
#pragma unroll
                for (int column = 0; column < COLS; ++column)
                    accumulator[row * COLS + column] = fmaf(scores[row * KC + ki], values[column], accumulator[row * COLS + column]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        const int qi = row / GROUP, h = row % GROUP;
        if (qi >= count) continue;
        const float inverse = sums[row] > 0.0f ? 1.0f / sums[row] : 0.0f;
        const size_t index = ((size_t(b) * seq + q0 + qi) * q_heads + first_head + h) * D;
#pragma unroll
        for (int column = 0; column < COLS; ++column)
            output[index + threadIdx.x + column * 256] = accumulator[row * COLS + column] * inverse;
    }
}

extern "C" __global__ __launch_bounds__(256)
void ts_embedding_attention_group2_d256_f32(const float* q, const float* k, const float* v, const int* bounds,
    float* output, int batch, int q_heads, int kv_heads, int seq, int radius, float scale)
{
    TS_PDL_ENTRY();
    ts_embedding_attention_body<256, 2, 8, 128>(q, k, v, bounds, output, batch, q_heads, kv_heads, seq, radius, scale);
}

extern "C" __global__ __launch_bounds__(256)
void ts_embedding_attention_group4_d512_f32(const float* q, const float* k, const float* v, const int* bounds,
    float* output, int batch, int q_heads, int kv_heads, int seq, int radius, float scale)
{
    TS_PDL_ENTRY();
    ts_embedding_attention_body<512, 4, 4, 128>(q, k, v, bounds, output, batch, q_heads, kv_heads, seq, radius, scale);
}
