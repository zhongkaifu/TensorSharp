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
