// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include "ggml_ops_gptoss_kv.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

using namespace tsg;

// ============================================================================
// GPT-OSS attention layer prefill: full attention block (norm → fused QKV with
// bias → RoPE → KV cache append → causal+SWA mask softmax with sinks → flash /
// fallback attention → output projection with bias) as ONE GGML graph.
//
// Replaces ~10 separate C# → GGML round trips per attention layer (each its
// own ggml_cgraph + Metal command buffer) with a single graph dispatch. This
// is the GPT-OSS member of the fused single-layer prefills.
//
// llama.cpp reference: src/models/openai-moe-iswa.cpp::llm_build_openai_moe_iswa
// (the attention block before the MoE FFN).
//
// Key model-specific points:
//   - Single contiguous KV cache per layer (no rolling window). The sliding
//     window is implemented purely via the attention mask.
//   - Attention sinks: a per-head extra logit added in the softmax denominator
//     (ggml_soft_max_add_sinks). Even-indexed layers (isSwa != 0) use the SWA
//     mask; odd-indexed layers do full causal attention.
//   - QKV (and O) projections have biases applied via ggml_add of a 1-D bias
//     tensor that GGML broadcasts across the seqLen dimension.
//   - RoPE is NeoX-style with yarn scaling (mode=2, beta_fast=32, beta_slow=1).
//   - The MoE FFN is *not* part of this kernel; it runs through the existing
//     fused MoE prefill kernel (TSGgml_MoEFFNPrefillSwiGLUQuantF32).
//
// Hidden state in/out is a flat [seqLen, hiddenSize] F32 buffer. The kernel
// writes the residual (input + attn_out_proj(attn(norm(input)))) back into the
// same buffer, ready for the MoE FFN to consume.
// ============================================================================
// ---------------------------------------------------------------------------
// Persistent device-side KV windows for the GPT-OSS attention kernel.
//
// The kernel used to allocate its KV window per call and upload the whole
// prefix [0, startPos) into it every time. That is O(context) of PCIe traffic
// per layer per token — at ~3.2k tokens and 24 layers roughly 315 MB per decoded
// token, which is why decode sat at single-digit tok/s and got worse as the
// conversation grew.
//
// Instead keep one device buffer per host cache (K and V are separate caches,
// so the host pointer identifies them), append the fresh rows into it, and only
// upload a prefix when the device copy cannot already contain it. The host cache
// stays a correct mirror because the freshly written rows are still downloaded
// after each call — that part is O(seqLen), not O(context).
//
// Staleness: `rows_valid` records how much of the device copy is known good.
// A caller that rewinds (startPos < rows_valid, e.g. a rejected speculative
// draft) or jumps forward (startPos > rows_valid, e.g. a restored KV prefix
// from another sequence) re-uploads what it needs. Anything that rewrites the
// host cache *in place* without moving startPos must call
// TSGgml_GptOssInvalidateKvCache.
// The registry itself lives in namespace tsg_gptoss (declared in
// ggml_ops_gptoss_kv.h) so the whole-model decode graph in
// ggml_ops_gptoss_decode.cpp attaches to the SAME device windows instead of
// keeping a second copy of every cache.
namespace tsg_gptoss
{
namespace
{
    std::mutex g_gptoss_kv_mutex;
    std::map<const void*, KvWindow> g_gptoss_kv;

    void kv_release(KvWindow& w)
    {
        if (w.buffer != nullptr) ggml_backend_buffer_free(w.buffer);
        if (w.ctx != nullptr) ggml_free(w.ctx);
        w = KvWindow{};
    }
} // namespace

std::mutex& kv_mutex() { return g_gptoss_kv_mutex; }

KvWindow* kv_find(const void* host_cache)
{
    if (host_cache == nullptr)
        return nullptr;
    auto it = g_gptoss_kv.find(host_cache);
    return (it == g_gptoss_kv.end()) ? nullptr : &it->second;
}

// Device window for `host_cache`, sized to hold at least `needed_rows`.
// Returns null when the allocation fails (caller falls back to a per-call
// window).
KvWindow* kv_acquire(const void* host_cache, int head_dim, int kv_heads,
                     ggml_type type, std::int64_t needed_rows, std::int64_t cache_rows)
{
    // If this cache currently decodes in the token-batched arena, its newest
    // rows live only there — flush them to the host mirror before any other
    // path attaches to the window. No-op from inside the batched kernel.
    batched_on_external_acquire(host_cache);

    auto it = g_gptoss_kv.find(host_cache);
    if (it != g_gptoss_kv.end())
    {
        KvWindow& w = it->second;
        const bool compatible = w.tensor != nullptr && w.backend == g_backend &&
            w.head_dim == head_dim && w.kv_heads == kv_heads &&
            w.type == static_cast<int>(type) && w.capacity >= needed_rows;
        if (compatible)
            return &w;
        kv_release(w);
        g_gptoss_kv.erase(it);
    }

    // Grow in powers of two so a conversation that lengthens gradually does
    // not reallocate (and re-upload) on every token, capped at the cache the
    // model actually declared.
    std::int64_t capacity = 1024;
    while (capacity < needed_rows)
        capacity *= 2;
    if (capacity > cache_rows)
        capacity = cache_rows;
    if (capacity < needed_rows)
        return nullptr;

    KvWindow w;
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 2;
    params.no_alloc = true;
    w.ctx = ggml_init(params);
    if (w.ctx == nullptr)
        return nullptr;
    w.tensor = ggml_new_tensor_3d(w.ctx, type, head_dim, capacity, kv_heads);
    if (w.tensor == nullptr) { kv_release(w); return nullptr; }
    w.buffer = ggml_backend_alloc_ctx_tensors(w.ctx, g_backend);
    if (w.buffer == nullptr) { kv_release(w); return nullptr; }
    // Zero the whole window. The whole-model decode graph attends over a window
    // padded up to a fixed stride, so it READS rows past the end of the written
    // prefix; recycled VRAM there decodes as F16 NaN, and NaN survives the -inf
    // padding mask (exp(NaN + -inf) = NaN), poisoning the entire attention row.
    ggml_backend_buffer_clear(w.buffer, 0);

    w.backend = g_backend;
    w.capacity = capacity;
    w.rows_valid = 0;
    w.head_dim = head_dim;
    w.kv_heads = kv_heads;
    w.type = static_cast<int>(type);
    if (vram_log_enabled())
        vram_log("gptoss-kv-window", static_cast<std::int64_t>(ggml_nbytes(w.tensor)));

    auto res = g_gptoss_kv.emplace(host_cache, w);
    return &res.first->second;
}

// Caller already holds kv_mutex().
void kv_drop_locked(const void* host_cache)
{
    if (host_cache == nullptr)
        return;
    // Dropping means the host was rewritten behind the kernels' backs (KV
    // snapshot restore, holder reuse, dispose): any batched-arena copy of this
    // cache is stale too. Discard, never flush.
    batched_on_drop(host_cache);
    auto it = g_gptoss_kv.find(host_cache);
    if (it == g_gptoss_kv.end())
        return;
    kv_release(it->second);
    g_gptoss_kv.erase(it);
}

void kv_drop_pair_locked(const void* k_cache, const void* v_cache)
{
    kv_drop_locked(k_cache);
    kv_drop_locked(v_cache);
}

void kv_drop_all_locked()
{
    batched_on_drop_all();
    for (auto& kv : g_gptoss_kv)
        kv_release(kv.second);
    g_gptoss_kv.clear();
}

void kv_download(KvWindow* w, void* host_cache, int cache_rows, std::int64_t rows)
{
    if (w == nullptr || w->tensor == nullptr || host_cache == nullptr || rows <= 0)
        return;
    rows = std::min<std::int64_t>(rows, std::min<std::int64_t>(w->capacity, cache_rows));
    const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(w->type), w->head_dim);
    const std::size_t host_stride = static_cast<std::size_t>(cache_rows) * row_bytes;
    const std::size_t win_stride = static_cast<std::size_t>(w->capacity) * row_bytes;
    const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
    char* host = static_cast<char*>(host_cache);
    for (int h = 0; h < w->kv_heads; h++)
        ggml_backend_tensor_get(w->tensor, host + h * host_stride, h * win_stride, bytes);
}

void kv_upload(KvWindow* w, const void* host_cache, int cache_rows,
               std::int64_t from_row, std::int64_t to_row)
{
    if (w == nullptr || w->tensor == nullptr || host_cache == nullptr || to_row <= from_row)
        return;
    const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(w->type), w->head_dim);
    const std::size_t host_stride = static_cast<std::size_t>(cache_rows) * row_bytes;
    const std::size_t win_stride = static_cast<std::size_t>(w->capacity) * row_bytes;
    const std::size_t offset = static_cast<std::size_t>(from_row) * row_bytes;
    const std::size_t bytes = static_cast<std::size_t>(to_row - from_row) * row_bytes;
    const char* host = static_cast<const char*>(host_cache);
    for (int h = 0; h < w->kv_heads; h++)
        ggml_backend_tensor_set(w->tensor, host + h * host_stride + offset,
                                h * win_stride + offset, bytes);
}

bool kv_acquire_pair(const TSGgmlGptOssLayerDesc& d, std::int64_t needed_rows,
                     KvWindow*& k_win, KvWindow*& v_win)
{
    const ggml_type kvType = static_cast<ggml_type>(d.kv_cache_type);
    KvWindow* existing_k = kv_find(d.k_cache);
    KvWindow* existing_v = kv_find(d.v_cache);
    if (existing_k != nullptr && existing_k->capacity < needed_rows)
        kv_download(existing_k, d.k_cache, d.cache_size, existing_k->rows_valid);
    if (existing_v != nullptr && existing_v->capacity < needed_rows)
        kv_download(existing_v, d.v_cache, d.cache_size, existing_v->rows_valid);

    k_win = kv_acquire(d.k_cache, d.head_dim, d.num_kv_heads, kvType, needed_rows, d.cache_size);
    v_win = (k_win == nullptr)
        ? nullptr
        : kv_acquire(d.v_cache, d.head_dim, d.num_kv_heads, kvType, needed_rows, d.cache_size);
    if (k_win == nullptr || v_win == nullptr)
    {
        // Half a pair is worse than none: drop both so the per-layer path
        // rebuilds from the host mirror rather than trusting a stale
        // rows_valid on the survivor.
        kv_drop_pair_locked(d.k_cache, d.v_cache);
        return false;
    }
    return true;
}
} // namespace tsg_gptoss

using GptOssKvWindow = tsg_gptoss::KvWindow;

// Drop both windows of a K/V pair; the caller holds the registry lock.
static void TSGgml_GptOssInvalidateKvCacheLocked(const void* kCacheData, const void* vCacheData)
{
    tsg_gptoss::kv_drop_pair_locked(kCacheData, vCacheData);
}

// Drop the device-resident copy of a GPT-OSS KV cache. Call this after writing
// the host cache behind the kernel's back (restoring a KV snapshot, swapping in
// another sequence's cache, clearing it); passing null clears every window.
TSG_EXPORT void TSGgml_GptOssInvalidateKvCache(const void* kCacheData, const void* vCacheData)
{
    std::lock_guard<std::mutex> lock(tsg_gptoss::kv_mutex());
    if (kCacheData == nullptr && vCacheData == nullptr)
    {
        tsg_gptoss::kv_drop_all_locked();
        return;
    }
    TSGgml_GptOssInvalidateKvCacheLocked(kCacheData, vCacheData);
}

TSG_EXPORT int TSGgml_GptOssAttentionLayerPrefill(
    float* hidden_data,        // [seqLen * hiddenSize] in/out (residual is added in place)
    int hiddenSize, int seqLen,
    // Attention norm
    void* attnNormW,           // F32 [hiddenSize]
    // Fused QKV (or Q-only when isQkvFused == 0; see kArr/vArr below)
    void* qkvW, int qkvType, std::int64_t qkvNe0, std::int64_t qkvNe1, std::int64_t qkvBytes,
    void* qkvB,                // F32 [qDim+2*kDim] when isQkvFused, else F32 [qDim]; may be null
    int isQkvFused,
    // Optional separate K/V weights+biases (used when isQkvFused == 0)
    void* kW, int kType, std::int64_t kNe0, std::int64_t kNe1, std::int64_t kBytes,
    void* kB,                  // F32 [kDim], may be null
    void* vW, int vType, std::int64_t vNe0, std::int64_t vNe1, std::int64_t vBytes,
    void* vB,                  // F32 [kDim], may be null
    // Output projection
    void* oW, int oType, std::int64_t oNe0, std::int64_t oNe1, std::int64_t oBytes,
    void* oB,                  // F32 [hiddenSize], may be null
    // KV cache (bound zero-copy where supported)
    void* kCacheData, void* vCacheData,
    int numHeads, int kvHeads, int headDim,
    int cacheSize, int startPos,
    // SWA / sinks
    int isSwa,                 // non-zero: apply sliding-window mask in addition to causal
    int slidingWindow,
    float* sinksData,          // F32 [numHeads], may be null (no sinks)
    // RoPE (NeoX yarn-scaled)
    float ropeBase, float ropeFreqScale, int ropeDims,
    int originalContextLength,
    // KV cache element type (0 = F32, 1 = F16)
    int kvCacheType,
    // Numerics
    float eps)
{
    try
    {
        if (!ensure_backend()) return 0;

        const int qDim = numHeads * headDim;
        const int kDim = kvHeads * headDim;
        const int totalSeqLen = startPos + seqLen;
        const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        const ggml_type kvType = static_cast<ggml_type>(kvCacheType);

        // GPT-OSS uses a single contiguous cache per layer (no rolling SWA).
        // Attention reads positions [0, totalSeqLen) and the SWA mask zeros out
        // anything older than (startPos + q_idx - slidingWindow + 1).
        const int kvLen = totalSeqLen;

        // 32 MiB context: same upper bound as Gemma4 prefill (covers concat-free
        // attention path + FFN graph allocations even at long ubatches).
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create context for GPT-OSS attention layer prefill.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // === Tensor declarations (allocated by ggml_backend_alloc_ctx_tensors below) ===

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);

        const int qkvDim = isQkvFused ? (qDim + 2 * kDim) : qDim;
        ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkvType), qkvNe0, qkvNe1);
        ggml_tensor* qkv_b = (qkvB != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkvDim) : nullptr;

        ggml_tensor* k_w = nullptr;
        ggml_tensor* k_b = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* v_b = nullptr;
        if (!isQkvFused)
        {
            k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kType), kNe0, kNe1);
            v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(vType), vNe0, vNe1);
            if (kB != nullptr) k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            if (vB != nullptr) v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
        }

        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(oType), oNe0, oNe1);
        ggml_tensor* o_b = (oB != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize) : nullptr;

        // KV cache "window" tensors: instead of binding the full per-layer
        // host cache (kvHeads * cacheSize * headDim * elemSize bytes — for
        // GPT-OSS this is 256 MiB per K and per V at context_length=131072,
        // which alone consumes ~12 GiB across 24 layers and exceeds Metal's
        // recommendedMaxWorkingSetSize on Apple Silicon, triggering
        // command-buffer OOMs in subsequent kernels), we allocate a per-call
        // window of shape [headDim, kvLen, kvHeads]. We upload the existing
        // prefix [0, startPos) before compute, ggml_cpy the fresh K/V into
        // [startPos, kvLen), then download the fresh slice back to the host
        // cache after compute. This keeps GPU residency for the cache to
        // O(kvLen) rather than O(cacheSize) and matches what llama.cpp's
        // build_attn_mha does internally for non-static caches.
        // Persistent device windows when they can be had; otherwise fall back to
        // the historical per-call window (allocated with the graph below).
        std::unique_lock<std::mutex> kv_lock(tsg_gptoss::kv_mutex());
        GptOssKvWindow* k_win = tsg_gptoss::kv_acquire(kCacheData, headDim, kvHeads, kvType, kvLen, cacheSize);
        GptOssKvWindow* v_win = k_win == nullptr
            ? nullptr
            : tsg_gptoss::kv_acquire(vCacheData, headDim, kvHeads, kvType, kvLen, cacheSize);
        const bool persistent_kv = k_win != nullptr && v_win != nullptr;
        if (!persistent_kv)
        {
            // Half a pair is worse than none: this call would fall back to
            // per-call windows and append only to the host cache, leaving the
            // surviving window's `rows_valid` claiming rows it no longer holds.
            TSGgml_GptOssInvalidateKvCacheLocked(kCacheData, vCacheData);
            kv_lock.unlock();
        }

        // The graph always sees a [headDim, kvLen, kvHeads] view. For the
        // persistent window that is a strided view into a larger allocation (the
        // head stride is the window capacity, not kvLen).
        ggml_tensor* k_cache_t = persistent_kv
            ? ggml_view_3d(ctx, k_win->tensor, headDim, kvLen, kvHeads,
                           k_win->tensor->nb[1], k_win->tensor->nb[2], 0)
            : ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, kvHeads);
        ggml_tensor* v_cache_t = persistent_kv
            ? ggml_view_3d(ctx, v_win->tensor, headDim, kvLen, kvHeads,
                           v_win->tensor->nb[1], v_win->tensor->nb[2], 0)
            : ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, kvHeads);

        ggml_tensor* sinks_t = (sinksData != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, numHeads) : nullptr;

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
        std::vector<int32_t> pos_data(seqLen);
        for (int i = 0; i < seqLen; i++) pos_data[i] = startPos + i;

        // === Build graph ===

        // 1. attention norm (RMSNorm + scale)
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);

        // 2. QKV projection (+ bias). For fused QKV the output is [qDim+2kDim, seqLen]
        // and we slice into Q/K/V views; for the separate path we run three matmuls.
        ggml_tensor* q_heads = nullptr;
        ggml_tensor* k_heads = nullptr;
        ggml_tensor* v_heads = nullptr;

        if (isQkvFused)
        {
            ggml_tensor* qkv_out = ggml_mul_mat(ctx, qkv_w, normed);
            if (qkv_b != nullptr)
                qkv_out = ggml_add(ctx, qkv_out, qkv_b);

            // Strided 2-D views into the fused QKV output, then ggml_cont +
            // reshape to [headDim, n_heads * seqLen] so RoPE sees the standard
            // 4-D [headDim, n_heads, seqLen, 1] layout (heads-fastest within the
            // flattened second dim, matching Gemma4LayerPrefill's QKV split).
            ggml_tensor* q_raw = ggml_view_2d(ctx, qkv_out, qDim, seqLen,
                qkv_out->nb[1], 0);
            ggml_tensor* k_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim) * sizeof(float));
            ggml_tensor* v_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim + kDim) * sizeof(float));

            q_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, q_raw), headDim, numHeads * seqLen);
            k_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, k_raw), headDim, kvHeads * seqLen);
            v_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, v_raw), headDim, kvHeads * seqLen);
        }
        else
        {
            ggml_tensor* q_proj = ggml_mul_mat(ctx, qkv_w, normed);
            if (qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, qkv_b);
            ggml_tensor* k_proj = ggml_mul_mat(ctx, k_w, normed);
            if (k_b != nullptr) k_proj = ggml_add(ctx, k_proj, k_b);
            ggml_tensor* v_proj = ggml_mul_mat(ctx, v_w, normed);
            if (v_b != nullptr) v_proj = ggml_add(ctx, v_proj, v_b);

            q_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, q_proj), headDim, numHeads * seqLen);
            k_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, k_proj), headDim, kvHeads * seqLen);
            v_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, v_proj), headDim, kvHeads * seqLen);
        }

        // 3. RoPE (NeoX yarn-scaled). Q and K share the same position tensor.
        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_heads, headDim, numHeads, seqLen, 1);
        ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_heads, headDim, kvHeads, seqLen, 1);
        ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr,
            ropeDims, /*mode=*/2, originalContextLength, ropeBase, ropeFreqScale,
            /*ext_factor=*/1.0f, /*attn_factor=*/1.0f,
            /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
        ggml_tensor* k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr,
            ropeDims, 2, originalContextLength, ropeBase, ropeFreqScale,
            1.0f, 1.0f, 32.0f, 1.0f);

        // 4. Reshape to attention layout. Q stays as a strided permute view (matches
        // the Gemma4 prefill kernel and llama.cpp's build_attn_mha). Fresh K/V are
        // brought to the tight [headDim, kvHeads, seqLen] layout so they can be
        // ggml_cpy'd into the cache and (for the fallback path) directly attended.
        ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
        ggml_tensor* k_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);
        ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_heads, headDim, kvHeads, seqLen, 1);
        ggml_tensor* v_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);

        // 5. KV cache write: contiguous append at startPos. Uses nb[1] so the offset
        // is correct for any cache dtype (F32/F16); ggml_cpy converts F32 fresh K/V
        // to the cache type as needed.
        std::size_t kv_offset = static_cast<std::size_t>(startPos) * k_cache_t->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_t,
            headDim, seqLen, kvHeads,
            k_cache_t->nb[1], k_cache_t->nb[2], kv_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_t,
            headDim, seqLen, kvHeads,
            v_cache_t->nb[1], v_cache_t->nb[2], kv_offset);
        ggml_tensor* k_cpy = ggml_cpy(ctx, k_fresh, k_dst);
        ggml_tensor* v_cpy = ggml_cpy(ctx, v_fresh, v_dst);

        // 6. Build the attention K/V source in the llama.cpp / Gemma4 tight
        // layout [headDim, kvLen, kvHeads] (heads slowest, positions in the
        // middle). For chunk 1 we use the fresh K/V directly (already in this
        // layout); for chunk 2+ we view the cache prefix [headDim, startPos,
        // kvHeads] (materialising as F32 when the cache is F16) and ggml_concat
        // it with fresh along ne[1] (positions).
        ggml_tensor* k_attn = k_fresh;
        ggml_tensor* v_attn = v_fresh;
        if (startPos > 0)
        {
            ggml_tensor* k_prev = ggml_view_3d(ctx, k_cache_t,
                headDim, startPos, kvHeads,
                k_cache_t->nb[1], k_cache_t->nb[2], 0);
            ggml_tensor* v_prev = ggml_view_3d(ctx, v_cache_t,
                headDim, startPos, kvHeads,
                v_cache_t->nb[1], v_cache_t->nb[2], 0);

            if (kvType != GGML_TYPE_F32)
            {
                ggml_tensor* k_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                ggml_tensor* v_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                k_prev = ggml_cpy(ctx, k_prev, k_prev_f32);
                v_prev = ggml_cpy(ctx, v_prev, v_prev_f32);
            }
            // Concat along ne[1] (positions); both inputs share [headDim, *, kvHeads].
            k_attn = ggml_concat(ctx, k_prev, k_fresh, 1);
            v_attn = ggml_concat(ctx, v_prev, v_fresh, 1);
        }

        // 7. Causal + optional SWA mask. The GPT-OSS C# attention path uses
        // exactly this: cell (q_idx, kv_idx) is unmasked iff kv_idx <= startPos +
        // q_idx (causal) AND, for SWA layers, kv_idx > startPos + q_idx -
        // slidingWindow.
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * seqLen);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            const int win = (isSwa != 0) ? slidingWindow : 0;
            for (int qi = 0; qi < seqLen; qi++)
            {
                int threshold = startPos + qi;
                int winStart = (win > 0) ? std::max(0, threshold - win + 1) : 0;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(qi) * kvLen];
                for (int ki = 0; ki < kvLen; ki++)
                    row[ki] = (ki > threshold || ki < winStart) ? neg_inf : zero_val;
            }
        }

        // 8. Attention. Use ggml_flash_attn_ext (with optional sinks) when the
        // backend supports the op for the current K/V dtype + head_dim — this
        // avoids materialising the [kvLen, seqLen, numHeads] scores tensor
        // (which is multi-tens-of-MiB at long contexts and triggers GPU OOM
        // when several layers' worth are in-flight on Metal). Fall back to the
        // explicit mul_mat → soft_max → mul_mat chain only when flash_attn_ext
        // isn't supported.
        ggml_tensor* attn_flat = nullptr;
        ggml_tensor* fa = flash_attn_ext_guarded(ctx, "GPT-OSS layer prefill", q_attn, k_attn, v_attn, mask_t,
            scale, 0.0f, 0.0f, sinks_t, GGML_PREC_F32);
        attn_flat = ggml_reshape_2d(ctx, fa, qDim, seqLen);

        // 9. Output projection (+ bias) and residual add.
        ggml_tensor* o_out = ggml_mul_mat(ctx, o_w, attn_flat);
        if (o_b != nullptr)
            o_out = ggml_add(ctx, o_out, o_b);
        ggml_tensor* residual = ggml_add(ctx, hidden_t, o_out);

        ggml_tensor* output = ggml_cpy(ctx, residual, hidden_out_t);
        ggml_set_output(output);

        // === Build & bind the graph ===
        const std::size_t graph_size = 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, k_cpy);
        ggml_build_forward_expand(graph, v_cpy);
        ggml_build_forward_expand(graph, output);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        tsg::PendingDeviceCopies pending;  // cached device copies bound below and not yet written: dropped unless committed after the upload
        std::vector<BufferHandle> ephem;

        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cache,
                        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cache && bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs, usage)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        if (needs) { uploads.push_back({t, data, bytes}); pending.add(data); }
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cache, buf)) {
                    if (!cache) ephem.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            uploads.push_back({t, data, bytes});
        };

        bind(qkv_w, qkvW, static_cast<std::size_t>(qkvBytes), true);
        bind(o_w, oW, static_cast<std::size_t>(oBytes), true);
        bind(attn_norm_w, attnNormW, hiddenSize * sizeof(float), true);
        if (qkv_b != nullptr) bind(qkv_b, qkvB, qkvDim * sizeof(float), true);
        if (k_w != nullptr) bind(k_w, kW, static_cast<std::size_t>(kBytes), true);
        if (v_w != nullptr) bind(v_w, vW, static_cast<std::size_t>(vBytes), true);
        if (k_b != nullptr) bind(k_b, kB, kDim * sizeof(float), true);
        if (v_b != nullptr) bind(v_b, vB, kDim * sizeof(float), true);
        if (o_b != nullptr) bind(o_b, oB, hiddenSize * sizeof(float), true);
        if (sinks_t != nullptr) bind(sinks_t, sinksData, numHeads * sizeof(float), true);
        // NOTE: k_cache_t / v_cache_t are now small per-call windows
        // [headDim, kvLen, kvHeads] sized to the active prefix only. They are
        // intentionally NOT bound to the host cache (kCacheData / vCacheData)
        // here — they're allocated by ggml_backend_alloc_ctx_tensors below as
        // GPU-only scratch, then we manually upload the prefix (per head, F32
        // path only when startPos > 0) before compute and download the freshly
        // appended slice back to the host cache after compute. This keeps the
        // KV cache GPU residency to O(kvLen * kvHeads * headDim) per layer
        // instead of O(cacheSize * kvHeads * headDim), which on GPT-OSS at
        // context_length=131072 would otherwise exhaust Metal's working set.

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) {
            set_last_error("Failed to allocate buffer for GPT-OSS attention layer prefill.");
            return 0;
        }

        // Drain any pending async work targeting hidden_data (the previous layer
        // / MoE FFN may have written it via a deferred-sync zero-copy path).
        host_read_barrier();

        for (auto& u : uploads)
            ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        pending.commit();

        ggml_backend_tensor_set(hidden_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, pos_data.data(), 0, seqLen * sizeof(int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        // Bring the device copy of the prefix [0, startPos) up to date. With a
        // persistent window this is normally a no-op — the rows are already
        // there from the calls that produced them — and only a rewind, a jump,
        // or a fresh allocation pays for an upload. The per-call fallback window
        // always starts empty, so it uploads the whole prefix as before.
        //
        // Host cache layout is [headDim, cacheSize, kvHeads] (heads slowest,
        // contiguous within a head); the device window is the same layout with
        // its own row capacity, so the copy is per head.
        const std::size_t elemSize = ggml_type_size(kvType);
        const std::int64_t deviceRows = persistent_kv ? k_win->capacity : kvLen;
        std::int64_t uploadFrom = 0;
        std::int64_t uploadTo = startPos;
        if (persistent_kv)
        {
            const std::int64_t valid = std::min(k_win->rows_valid, v_win->rows_valid);
            uploadFrom = (valid >= startPos) ? startPos : valid;   // >= startPos: nothing to do
            uploadTo = startPos;
        }
        if (uploadTo > uploadFrom)
        {
            const std::size_t hostStrideBytes   = static_cast<std::size_t>(cacheSize)  * headDim * elemSize;
            const std::size_t windowStrideBytes = static_cast<std::size_t>(deviceRows) * headDim * elemSize;
            const std::size_t offsetBytes       = static_cast<std::size_t>(uploadFrom) * headDim * elemSize;
            const std::size_t prefixBytes       = static_cast<std::size_t>(uploadTo - uploadFrom) * headDim * elemSize;
            char* kHost = static_cast<char*>(kCacheData);
            char* vHost = static_cast<char*>(vCacheData);
            ggml_tensor* kStore = persistent_kv ? k_win->tensor : k_cache_t;
            ggml_tensor* vStore = persistent_kv ? v_win->tensor : v_cache_t;
            for (int h = 0; h < kvHeads; h++)
            {
                ggml_backend_tensor_set(kStore, kHost + h * hostStrideBytes + offsetBytes,
                    h * windowStrideBytes + offsetBytes, prefixBytes);
                ggml_backend_tensor_set(vStore, vHost + h * hostStrideBytes + offsetBytes,
                    h * windowStrideBytes + offsetBytes, prefixBytes);
            }
        }

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            set_last_error("Graph compute failed for GPT-OSS attention layer prefill.");
            return 0;
        }

        // Synchronously download hidden_out + the freshly written K/V slice
        // and wait for the GPU to retire all command buffers before returning.
        // We cannot use the async download here because BufferHandle's
        // destructor frees the per-call compute buffer immediately, while
        // pipelined MoE work can exhaust the GPU working set otherwise.
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(hidden_out_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));

        // Download the freshly appended K/V slice [startPos, kvLen) per head
        // back to the host cache. This is O(seqLen), not O(context), and it is
        // what keeps the host cache a valid mirror for the readers that do not
        // go through this kernel (the per-op attention path and the
        // engine's KV snapshot/swap).
        {
            const std::size_t hostStrideBytes   = static_cast<std::size_t>(cacheSize)  * headDim * elemSize;
            const std::size_t windowStrideBytes = static_cast<std::size_t>(deviceRows) * headDim * elemSize;
            const std::size_t freshOffsetBytes  = static_cast<std::size_t>(startPos)   * headDim * elemSize;
            const std::size_t freshBytes        = static_cast<std::size_t>(seqLen)     * headDim * elemSize;
            char* kHost = static_cast<char*>(kCacheData);
            char* vHost = static_cast<char*>(vCacheData);
            ggml_tensor* kStore = persistent_kv ? k_win->tensor : k_cache_t;
            ggml_tensor* vStore = persistent_kv ? v_win->tensor : v_cache_t;
            for (int h = 0; h < kvHeads; h++)
            {
                ggml_backend_tensor_get(kStore,
                    kHost + h * hostStrideBytes + freshOffsetBytes,
                    h * windowStrideBytes + freshOffsetBytes, freshBytes);
                ggml_backend_tensor_get(vStore,
                    vHost + h * hostStrideBytes + freshOffsetBytes,
                    h * windowStrideBytes + freshOffsetBytes, freshBytes);
            }
        }

        if (persistent_kv)
        {
            k_win->rows_valid = kvLen;
            v_win->rows_valid = kvLen;
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in GPT-OSS attention layer prefill."); return 0; }
}
