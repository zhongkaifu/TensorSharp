// TensorSharp-owned Gemma4 audio F32 kernels; upstream ggml remains unchanged.
extern "C" __global__ void ts_gemma4_audio_attention_f32(
    const float *q, const float *k, const float *v, const float *relative, float *out,
    int sequence, int heads, int head, int past, int future, float cap)
{
    TS_PDL_ENTRY();
    const int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int window = past + future + 1;
    __shared__ float scores[257], ccWarps[4], cpWarps[4];
    for (int slot = 0; slot < window; ++slot) {
        const int key = t + slot - past;
        float cc = 0, cp = 0;
        if (key >= 0 && key < sequence) {
            for (int d = threadIdx.x; d < head; d += blockDim.x) {
                const float query = q[(t * heads + h) * head + d];
                cc = fmaf(query, k[(key * heads + h) * head + d], cc);
                cp = fmaf(query, relative[(slot * heads + h) * head + d], cp);
            }
        }
        for (int shift = 16; shift > 0; shift >>= 1) {
            cc += __shfl_down_sync(0xffffffffu, cc, shift);
            cp += __shfl_down_sync(0xffffffffu, cp, shift);
        }
        if (lane == 0) { ccWarps[warp] = cc; cpWarps[warp] = cp; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float content = 0, position = 0;
            for (int w = 0; w < 4; ++w) { content += ccWarps[w]; position += cpWarps[w]; }
            scores[slot] = key >= 0 && key < sequence ? tanhf((content + position) / cap) * cap : -1.0e9f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float maximum = -INFINITY, denominator = 0;
        for (int slot = 0; slot < window; ++slot) maximum = fmaxf(maximum, scores[slot]);
        for (int slot = 0; slot < window; ++slot) { scores[slot] = expf(scores[slot] - maximum); denominator += scores[slot]; }
        for (int slot = 0; slot < window; ++slot) scores[slot] /= denominator;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float sum = 0;
        for (int slot = 0; slot < window; ++slot) {
            const int key = t + slot - past;
            if (key >= 0 && key < sequence) sum = fmaf(scores[slot], v[(key * heads + h) * head + d], sum);
        }
        out[(t * heads + h) * head + d] = sum;
    }
}

extern "C" __global__ void ts_gemma4_audio_depthwise_f32(
    const float *x, const float *weight, float *out, int sequence, int channels, int kernel)
{
    TS_PDL_ENTRY();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sequence * channels) return;
    const int t = index / channels, channel = index % channels;
    float sum = 0;
    for (int k = 0; k < kernel; ++k) {
        const int source = t - (kernel - 1 - k);
        if (source >= 0) sum = fmaf(x[source * channels + channel], weight[channel * kernel + k], sum);
    }
    out[index] = sum;
}
