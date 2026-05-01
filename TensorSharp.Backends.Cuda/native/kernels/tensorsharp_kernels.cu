#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define GGML_Q4_0 2
#define GGML_Q4_1 3
#define GGML_Q5_0 6
#define GGML_Q5_1 7
#define GGML_Q8_0 8
#define GGML_Q8_1 9
#define GGML_Q4_K 12
#define GGML_Q5_K 13
#define GGML_Q6_K 14

__device__ __forceinline__ unsigned int read_u32_unaligned(const uint8_t* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ int qrow_bytes(int type, int cols)
{
    switch (type)
    {
        case GGML_Q4_0: return (cols / 32) * 18;
        case GGML_Q4_1: return (cols / 32) * 20;
        case GGML_Q5_0: return (cols / 32) * 22;
        case GGML_Q5_1: return (cols / 32) * 24;
        case GGML_Q8_0: return (cols / 32) * 34;
        case GGML_Q8_1: return (cols / 32) * 36;
        case GGML_Q4_K: return (cols / 256) * 144;
        case GGML_Q5_K: return (cols / 256) * 176;
        case GGML_Q6_K: return (cols / 256) * 210;
        default: return 0;
    }
}

__device__ __forceinline__ int get_scale_min_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index] & 0x3F;
    return (s[index + 4] & 0x0F) | ((s[index - 4] >> 6) << 4);
}

__device__ __forceinline__ int get_min_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index + 4] & 0x3F;
    return (s[index + 4] >> 4) | ((s[index] >> 6) << 4);
}

__device__ __forceinline__ float qvalue_at(const uint8_t* row, int type, int col)
{
    if (type == GGML_Q4_0)
    {
        const uint8_t* block = row + (col / 32) * 18;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        int lane = col & 31;
        uint8_t packed = block[2 + (lane & 15)];
        int v = lane < 16 ? (packed & 0x0F) - 8 : (packed >> 4) - 8;
        return d * (float)v;
    }

    if (type == GGML_Q4_1)
    {
        const uint8_t* block = row + (col / 32) * 20;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        int lane = col & 31;
        uint8_t packed = block[4 + (lane & 15)];
        int v = lane < 16 ? (packed & 0x0F) : (packed >> 4);
        return d * (float)v + m;
    }

    if (type == GGML_Q5_0)
    {
        const uint8_t* block = row + (col / 32) * 22;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        unsigned int qh = read_u32_unaligned(block + 2);
        int lane = col & 31;
        uint8_t packed = block[6 + (lane & 15)];
        int qhbit = (qh >> lane) & 1;
        int v = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) | (qhbit << 4);
        return d * (float)(v - 16);
    }

    if (type == GGML_Q5_1)
    {
        const uint8_t* block = row + (col / 32) * 24;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        unsigned int qh = read_u32_unaligned(block + 4);
        int lane = col & 31;
        uint8_t packed = block[8 + (lane & 15)];
        int qhbit = (qh >> lane) & 1;
        int v = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) | (qhbit << 4);
        return d * (float)v + m;
    }

    if (type == GGML_Q8_0)
    {
        const uint8_t* block = row + (col / 32) * 34;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        return d * (float)qs[col & 31];
    }

    if (type == GGML_Q8_1)
    {
        const uint8_t* block = row + (col / 32) * 36;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
        return d * (float)qs[col & 31];
    }

    if (type == GGML_Q4_K)
    {
        const uint8_t* block = row + (col / 256) * 144;
        int t = col & 255;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        int pair = t / 64;
        int pos = t & 63;
        int is_odd = pos / 32;
        int j = pos & 31;
        int sub = pair * 2 + is_odd;
        int sc = get_scale_min_k4(scales, sub);
        int m = get_min_k4(scales, sub);
        uint8_t packed = qs[pair * 32 + j];
        int v = is_odd ? (packed >> 4) : (packed & 0x0F);
        return d * (float)sc * (float)v - dmin * (float)m;
    }

    if (type == GGML_Q5_K)
    {
        const uint8_t* block = row + (col / 256) * 176;
        int t = col & 255;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;
        int sub = t / 32;
        int pos = t & 31;
        int sc = get_scale_min_k4(scales, sub);
        int m = get_min_k4(scales, sub);
        const uint8_t* sub_qs = qs + sub * 16;
        const uint8_t* sub_qh = qh + sub * 4;
        int j = pos / 2;
        uint8_t packed = sub_qs[j];
        int bit = (sub_qh[j / 4] >> ((j & 3) * 2 + (pos & 1))) & 1;
        int v = ((pos & 1) ? (packed >> 4) : (packed & 0x0F)) | (bit << 4);
        return d * (float)sc * (float)v - dmin * (float)m;
    }

    if (type == GGML_Q6_K)
    {
        const uint8_t* block = row + (col / 256) * 210;
        int t = col & 255;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));
        int half_idx = t / 128;
        int pos = t & 127;
        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;
        int group = pos / 32;
        int l = pos & 31;
        int q;
        if (group == 0)
            q = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
        else if (group == 1)
            q = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
        else if (group == 2)
            q = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
        else
            q = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;
        int isc = l / 16;
        return d * (float)sc_half[isc + group * 2] * (float)q;
    }

    return 0.0f;
}

__device__ __forceinline__ float block_reduce_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        v = lane < num_warps ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }

    return v;
}

__device__ __forceinline__ float block_reduce_max(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, offset));

    __shared__ float warp_vals[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_vals[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        v = lane < num_warps ? warp_vals[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, offset));
    }

    return v;
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void ts_fill_f32(float* output, int count, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = value;
}

extern "C" __global__ void ts_unary_f32(const float* input, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = input[i];
    float y = x;
    if (op == 0)
        y = fmaxf(x, 0.0f);
    else if (op == 1)
        y = 1.0f / (1.0f + expf(-x));
    else if (op == 2)
        y = silu(x);
    else if (op == 3)
        y = gelu(x);
    else if (op == 4)
        y = tanhf(x);
    output[i] = y;
}

extern "C" __global__ void ts_binary_f32(const float* lhs, const float* rhs, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = lhs[i];
    float y = rhs[i];
    if (op == 0)
        output[i] = x + y;
    else if (op == 1)
        output[i] = x - y;
    else if (op == 2)
        output[i] = x * y;
    else
        output[i] = x / y;
}

extern "C" __global__ void ts_scalar_f32(const float* input, float* output, int count, float value, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = input[i];
    if (op == 0)
        output[i] = x + value;
    else if (op == 1)
        output[i] = x - value;
    else if (op == 2)
        output[i] = x * value;
    else if (op == 3)
        output[i] = x / value;
    else if (op == 4)
        output[i] = value - x;
    else
        output[i] = value / x;
}

extern "C" __global__ void ts_ternary_f32(const float* x, const float* y, const float* z, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float a = x[i];
    float b = y[i];
    float c = z[i];
    if (op == 0)
        output[i] = a + b * c;
    else
        output[i] = a + b / c;
}

extern "C" __global__ void ts_addmul_scalar_f32(const float* x, const float* y, float* output, int count, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = x[i] + y[i] * value;
}

extern "C" __global__ void ts_mulmuladd_f32(const float* x, const float* y, const float* z, const float* w, float* output, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = x[i] * y[i] + z[i] * w[i];
}

extern "C" __global__ void ts_binary_activation_f32(const float* a, const float* b, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = a[i];
    float y = b[i];
    if (op == 0)
        output[i] = silu(x) * y;
    else if (op == 1)
        output[i] = gelu(x) * y;
    else
        output[i] = x * (1.0f / (1.0f + expf(-y)));
}

extern "C" __global__ void ts_silu_mul_split_f32(const float* gate_up, float* output, int rows, int half_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_dim;
    if (idx >= total)
        return;

    int row = idx / half_dim;
    int col = idx - row * half_dim;
    const float* row_ptr = gate_up + (size_t)row * half_dim * 2;
    float gate = row_ptr[col];
    float up = row_ptr[col + half_dim];
    output[idx] = silu(gate) * up;
}

extern "C" __global__ void ts_gelu_mul_split_f32(const float* gate_up, float* output, int rows, int half_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_dim;
    if (idx >= total)
        return;

    int row = idx / half_dim;
    int col = idx - row * half_dim;
    const float* row_ptr = gate_up + (size_t)row * half_dim * 2;
    float gate = row_ptr[col];
    float up = row_ptr[col + half_dim];
    output[idx] = gelu(gate) * up;
}

extern "C" __global__ void ts_rmsnorm_f32(
    const float* input,
    const float* alpha,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float eps)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i] * inv_rms * alpha[i];
        if (beta != 0)
            v += beta[i];
        y[i] = v;
    }
}

extern "C" __global__ void ts_softmax_f32(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    float max_v = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        max_v = fmaxf(max_v, x[i]);

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float e = expf(x[i] - shared_max);
        y[i] = e;
        sum += e;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = 1.0f / sum;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] *= inv_sum;
}

extern "C" __global__ void ts_scaled_dot_product_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    float* output,
    int batch,
    int seq_q,
    int seq_k,
    int heads,
    int key_dim,
    int value_dim,
    float scale,
    int has_mask)
{
    int batch_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (batch_head >= batch * heads || q_pos >= seq_q)
        return;

    int b = batch_head / heads;
    int h = batch_head - b * heads;
    const float* q = query + (((size_t)b * seq_q + q_pos) * heads + h) * key_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int k_pos = threadIdx.x; k_pos < seq_k; k_pos += blockDim.x)
    {
        const float* k = key + (((size_t)b * seq_k + k_pos) * heads + h) * key_dim;
        float dot = 0.0f;
        for (int d = 0; d < key_dim; d++)
            dot += q[d] * k[d];

        float score = dot * scale;
        if (has_mask)
            score += mask[(((size_t)b * heads + h) * seq_q + q_pos) * seq_k + k_pos];
        scores[k_pos] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < seq_k; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (((size_t)b * seq_q + q_pos) * heads + h) * value_dim;
    for (int d = threadIdx.x; d < value_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos < seq_k; k_pos++)
        {
            const float* v = value + (((size_t)b * seq_k + k_pos) * heads + h) * value_dim;
            acc += scores[k_pos] * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (q_head >= num_q_heads || q_pos >= seq_len)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int visible = mask_start + q_pos;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);

    const float* q = query + ((size_t)q_head * seq_len + q_pos) * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int k_pos = threadIdx.x; k_pos < kv_len; k_pos += blockDim.x)
    {
        bool allowed = k_pos <= visible && k_pos >= min_visible;
        float score = -FLT_MAX;
        if (allowed)
        {
            const float* k = key + ((size_t)kv_head * kv_len + k_pos) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q[d] * k[d];
            score = dot * scale;
            max_v = fmaxf(max_v, score);
        }
        scores[k_pos] = score;
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < kv_len; k_pos += blockDim.x)
    {
        float score = scores[k_pos];
        float p = score == -FLT_MAX ? 0.0f : expf(score - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + ((size_t)q_pos * num_q_heads + q_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos < kv_len; k_pos++)
        {
            float p = scores[k_pos];
            if (p != 0.0f)
            {
                const float* v = value + ((size_t)kv_head * kv_len + k_pos) * head_dim;
                acc += p * inv_sum * v[d];
            }
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        int logical_pos = attend_start + t;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const float* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];

        float score = dot * scale;
        scores[t] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        float p = expf(scores[t] - shared_max);
        scores[t] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < attend_len; t++)
        {
            int logical_pos = attend_start + t;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const float* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[t] * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_slice_columns_f32(
    const float* source,
    float* output,
    int rows,
    int source_cols,
    int col_offset,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (idx >= total)
        return;

    int row = idx / width;
    int col = idx - row * width;
    output[idx] = source[(size_t)row * source_cols + col_offset + col];
}

extern "C" __global__ void ts_flat_to_head_first_f32(
    const float* source,
    float* output,
    int seq_len,
    int num_heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    output[idx] = source[((size_t)seq * num_heads + head) * head_dim + d];
}

extern "C" __global__ void ts_split_qkv_head_first_f32(
    const float* source,
    float* output,
    int seq_len,
    int source_cols,
    int col_offset,
    int num_heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    output[idx] = source[(size_t)seq * source_cols + col_offset + head * head_dim + d];
}

extern "C" __global__ void ts_copy_head_first_to_cache_f32(
    const float* source,
    float* cache,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size,
    int circular)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = circular ? ((start_pos + seq) % cache_size) : (start_pos + seq);
    cache[((size_t)head * cache_size + cache_pos) * head_dim + d] = source[idx];
}

extern "C" __global__ void ts_gather_circular_head_first_f32(
    const float* cache,
    float* output,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = (start_pos + seq) % cache_size;
    if (cache_pos < 0)
        cache_pos += cache_size;
    output[idx] = cache[((size_t)head * cache_size + cache_pos) * head_dim + d];
}

extern "C" __global__ void ts_concat_head_first_f32(
    const float* a,
    const float* b,
    float* output,
    int num_heads,
    int len_a,
    int len_b,
    int head_dim)
{
    int total_len = len_a + len_b;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * total_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % total_len;
    int head = tmp / total_len;
    if (seq < len_a)
        output[idx] = a[((size_t)head * len_a + seq) * head_dim + d];
    else
        output[idx] = b[((size_t)head * len_b + (seq - len_a)) * head_dim + d];
}

extern "C" __global__ void ts_neox_rope_head_first_f32(
    float* data,
    const float* cos_table,
    const float* sin_table,
    int num_heads,
    int seq_len,
    int head_dim,
    int rope_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * rope_half;
    if (idx >= total)
        return;

    int j = idx % rope_half;
    int tmp = idx / rope_half;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    size_t base = ((size_t)head * seq_len + seq) * head_dim;
    size_t table = (size_t)seq * rope_half + j;
    float cos_v = cos_table[table];
    float sin_v = sin_table[table];
    float x0 = data[base + j];
    float x1 = data[base + j + rope_half];
    data[base + j] = x0 * cos_v - x1 * sin_v;
    data[base + j + rope_half] = x0 * sin_v + x1 * cos_v;
}

extern "C" __global__ void ts_index_select_f32(
    const float* source,
    const void* indices,
    float* output,
    int rows,
    int cols,
    int source_rows,
    int indices_are_int32,
    int is_add)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int src_idx = indices_are_int32
        ? reinterpret_cast<const int*>(indices)[row]
        : (int)reinterpret_cast<const float*>(indices)[row];
    if (src_idx < 0 || src_idx >= source_rows)
        return;

    const float* src_row = source + (size_t)src_idx * cols;
    float* out_row = output + (size_t)row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        float v = src_row[col];
        out_row[col] = is_add ? out_row[col] + v : v;
    }
}

extern "C" __global__ void ts_add_causal_mask_f32(float* tensor, int rows, int cols, int seq_len, int start_pos, float masked_value)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int q = row % seq_len;
    int visible = start_pos + q;
    float* row_ptr = tensor + (size_t)row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        if (col > visible)
            row_ptr[col] += masked_value;
    }
}

extern "C" __global__ void ts_rope_f32(const float* input, float* output, int rows, int cols, int seq_len, int row_offset)
{
    int pair_count = cols / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * pair_count;
    if (idx >= total)
        return;

    int row = idx / pair_count;
    int pair = idx - row * pair_count;
    int m = (row % seq_len) + row_offset;
    float theta = powf(500000.0f, -2.0f * (float)pair / (float)cols);
    float angle = theta * (float)m;
    float c = cosf(angle);
    float s = sinf(angle);

    const float* src = input + (size_t)row * cols;
    float* dst = output + (size_t)row * cols;
    float left = src[pair * 2];
    float right = src[pair * 2 + 1];
    dst[pair * 2] = left * c - right * s;
    dst[pair * 2 + 1] = right * c + left * s;
}

__device__ __forceinline__ float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float freq_base)
{
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(freq_base));
}

__device__ __forceinline__ void yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float* low, float* high)
{
    if (beta_fast == 0.0f && beta_slow == 0.0f)
    {
        *low = FLT_MAX;
        *high = (float)(n_dims / 2 - 1);
    }
    else
    {
        *low = fmaxf(0.0f, floorf(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base)));
        *high = fminf((float)(n_dims / 2 - 1), ceilf(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base)));
    }
}

__device__ __forceinline__ void yarn_rope(float theta_extrap, float freq_scale, float corr_low, float corr_high, int i0, float ext_factor, float mscale, float* c, float* s)
{
    float theta_interp = freq_scale * theta_extrap;
    float ramp_y = ((float)i0 - corr_low) / fmaxf(0.001f, corr_high - corr_low);
    float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, ramp_y))) * ext_factor;
    float theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

extern "C" __global__ void ts_rope_ex_f32(
    const float* input,
    const void* positions,
    float* output,
    int rows,
    int cols,
    int rope_dim,
    int mode,
    int positions_are_int32,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int add_to_result)
{
    int active_dim = rope_dim < cols ? rope_dim : cols;
    int pair_count = active_dim / 2;
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_count <= 0)
    {
        int total = rows * cols;
        if (global < total && !add_to_result)
            output[global] = input[global];
        return;
    }

    if (!add_to_result && global < rows * cols)
    {
        int col = global % cols;
        if (col >= pair_count * 2)
            output[global] = input[global];
    }

    int idx = global;
    int total = rows * pair_count;
    if (idx >= total)
        return;

    const int GGML_ROPE_TYPE_NEOX = 2;
    bool neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    int row = idx / pair_count;
    int i = idx - row * pair_count;
    int pos = positions_are_int32
        ? reinterpret_cast<const int*>(positions)[row]
        : (int)reinterpret_cast<const float*>(positions)[row];

    float theta_extrap = (float)pos * powf(freq_base, -2.0f * (float)i / (float)active_dim);
    float c;
    float s;
    if (ext_factor != 0.0f)
    {
        float corr_low;
        float corr_high;
        yarn_corr_dims(active_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_low, &corr_high);
        float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        yarn_rope(theta_extrap, freq_scale, corr_low, corr_high, i, ext_factor, mscale, &c, &s);
    }
    else
    {
        float angle = theta_extrap * freq_scale;
        c = cosf(angle);
        s = sinf(angle);
    }

    const float* src = input + (size_t)row * cols;
    float* dst = output + (size_t)row * cols;

    int left_index;
    int right_index;
    if (neox)
    {
        left_index = i;
        right_index = i + pair_count;
    }
    else
    {
        left_index = i * 2;
        right_index = i * 2 + 1;
    }

    float left = src[left_index];
    float right = src[right_index];
    float out_left = left * c - right * s;
    float out_right = right * c + left * s;
    if (add_to_result)
    {
        dst[left_index] += out_left;
        dst[right_index] += out_right;
    }
    else
    {
        dst[left_index] = out_left;
        dst[right_index] = out_right;
    }
}

extern "C" __global__ void ts_quant_matmul_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int type,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = qrow_bytes(type, in_dim);
    const float* x_row = input + (size_t)row * in_dim;

    const uint8_t* w_row0 = weights + (size_t)(out_col0 + 0) * row_bytes;
    const uint8_t* w_row1 = out_col0 + 1 < out_dim ? weights + (size_t)(out_col0 + 1) * row_bytes : 0;
    const uint8_t* w_row2 = out_col0 + 2 < out_dim ? weights + (size_t)(out_col0 + 2) * row_bytes : 0;
    const uint8_t* w_row3 = out_col0 + 3 < out_dim ? weights + (size_t)(out_col0 + 3) * row_bytes : 0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        float x = x_row[k];
        acc0 += qvalue_at(w_row0, type, k) * x;
        if (w_row1 != 0)
            acc1 += qvalue_at(w_row1, type, k) * x;
        if (w_row2 != 0)
            acc2 += qvalue_at(w_row2, type, k) * x;
        if (w_row3 != 0)
            acc3 += qvalue_at(w_row3, type, k) * x;
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && w_row1 != 0)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && w_row2 != 0)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && w_row3 != 0)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

extern "C" __global__ void ts_quant_matmul_q4_0_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = (in_dim / 32) * 18;
    const float* x_row = input + (size_t)row * in_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 18;
        int lane = k & 31;
        int packed_index = lane & 15;
        int high = lane >> 4;
        float x = x_row[k];

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        uint8_t p0 = w0[2 + packed_index];
        int q0 = (high ? (p0 >> 4) : (p0 & 0x0F)) - 8;
        acc0 += d0 * (float)q0 * x;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            uint8_t p1 = w1[2 + packed_index];
            int q1 = (high ? (p1 >> 4) : (p1 & 0x0F)) - 8;
            acc1 += d1 * (float)q1 * x;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            uint8_t p2 = w2[2 + packed_index];
            int q2 = (high ? (p2 >> 4) : (p2 & 0x0F)) - 8;
            acc2 += d2 * (float)q2 * x;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            uint8_t p3 = w3[2 + packed_index];
            int q3 = (high ? (p3 >> 4) : (p3 & 0x0F)) - 8;
            acc3 += d3 * (float)q3 * x;
        }
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

extern "C" __global__ void ts_quant_matmul_q8_0_single_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = (in_dim / 32) * 34;
    const float* x_row = input + (size_t)row * in_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 34;
        int lane = k & 31;
        float x = x_row[k];

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        int8_t q0 = reinterpret_cast<const int8_t*>(w0 + 2)[lane];
        acc0 += d0 * (float)q0 * x;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            int8_t q1 = reinterpret_cast<const int8_t*>(w1 + 2)[lane];
            acc1 += d1 * (float)q1 * x;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            int8_t q2 = reinterpret_cast<const int8_t*>(w2 + 2)[lane];
            acc2 += d2 * (float)q2 * x;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            int8_t q3 = reinterpret_cast<const int8_t*>(w3 + 2)[lane];
            acc3 += d3 * (float)q3 * x;
        }
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

extern "C" __global__ void ts_quant_matmul_q8_0_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;

    int row0 = blockIdx.y * 4;
    if (out_col0 >= out_dim || row0 >= rows)
        return;

    int row_bytes = (in_dim / 32) * 34;
    bool has_r1 = row0 + 1 < rows;
    bool has_r2 = row0 + 2 < rows;
    bool has_r3 = row0 + 3 < rows;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 34;
        int lane = k & 31;
        float x0 = input[(size_t)(row0 + 0) * in_dim + k];
        float x1 = has_r1 ? input[(size_t)(row0 + 1) * in_dim + k] : 0.0f;
        float x2 = has_r2 ? input[(size_t)(row0 + 2) * in_dim + k] : 0.0f;
        float x3 = has_r3 ? input[(size_t)(row0 + 3) * in_dim + k] : 0.0f;

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        int8_t q0 = reinterpret_cast<const int8_t*>(w0 + 2)[lane];
        float wv0 = d0 * (float)q0;
        acc00 += wv0 * x0;
        acc10 += wv0 * x1;
        acc20 += wv0 * x2;
        acc30 += wv0 * x3;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            int8_t q1 = reinterpret_cast<const int8_t*>(w1 + 2)[lane];
            float wv1 = d1 * (float)q1;
            acc01 += wv1 * x0;
            acc11 += wv1 * x1;
            acc21 += wv1 * x2;
            acc31 += wv1 * x3;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            int8_t q2 = reinterpret_cast<const int8_t*>(w2 + 2)[lane];
            float wv2 = d2 * (float)q2;
            acc02 += wv2 * x0;
            acc12 += wv2 * x1;
            acc22 += wv2 * x2;
            acc32 += wv2 * x3;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            int8_t q3 = reinterpret_cast<const int8_t*>(w3 + 2)[lane];
            float wv3 = d3 * (float)q3;
            acc03 += wv3 * x0;
            acc13 += wv3 * x1;
            acc23 += wv3 * x2;
            acc33 += wv3 * x3;
        }
    }

    acc00 = block_reduce_sum(acc00);
    if (threadIdx.x == 0)
        output[(size_t)(row0 + 0) * out_dim + out_col0] = acc00;
    __syncthreads();

    acc01 = block_reduce_sum(acc01);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 1] = acc01;
    __syncthreads();

    acc02 = block_reduce_sum(acc02);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 2] = acc02;
    __syncthreads();

    acc03 = block_reduce_sum(acc03);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 3] = acc03;
    __syncthreads();

    acc10 = block_reduce_sum(acc10);
    if (threadIdx.x == 0 && has_r1)
        output[(size_t)(row0 + 1) * out_dim + out_col0] = acc10;
    __syncthreads();

    acc11 = block_reduce_sum(acc11);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 1] = acc11;
    __syncthreads();

    acc12 = block_reduce_sum(acc12);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 2] = acc12;
    __syncthreads();

    acc13 = block_reduce_sum(acc13);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 3] = acc13;
    __syncthreads();

    acc20 = block_reduce_sum(acc20);
    if (threadIdx.x == 0 && has_r2)
        output[(size_t)(row0 + 2) * out_dim + out_col0] = acc20;
    __syncthreads();

    acc21 = block_reduce_sum(acc21);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 1] = acc21;
    __syncthreads();

    acc22 = block_reduce_sum(acc22);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 2] = acc22;
    __syncthreads();

    acc23 = block_reduce_sum(acc23);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 3] = acc23;
    __syncthreads();

    acc30 = block_reduce_sum(acc30);
    if (threadIdx.x == 0 && has_r3)
        output[(size_t)(row0 + 3) * out_dim + out_col0] = acc30;
    __syncthreads();

    acc31 = block_reduce_sum(acc31);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 1] = acc31;
    __syncthreads();

    acc32 = block_reduce_sum(acc32);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 2] = acc32;
    __syncthreads();

    acc33 = block_reduce_sum(acc33);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 3] = acc33;
}

extern "C" __global__ void ts_quant_get_rows_f32(
    const uint8_t* weights,
    const void* indices,
    float* output,
    int type,
    int cols,
    int rows,
    int indices_are_int32)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int src_row = indices_are_int32
        ? reinterpret_cast<const int*>(indices)[row]
        : (int)reinterpret_cast<const float*>(indices)[row];
    if (src_row < 0)
        return;

    int row_bytes = qrow_bytes(type, cols);
    const uint8_t* w_row = weights + (size_t)src_row * row_bytes;
    float* out_row = output + (size_t)row * cols;

    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        out_row[col] = qvalue_at(w_row, type, col);
}
