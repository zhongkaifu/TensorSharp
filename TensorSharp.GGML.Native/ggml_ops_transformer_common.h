// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#pragma once

#include "ggml_ops_internal.h"

#ifdef TSG_GGML_USE_CUDA
// On-device causal-mask fill (ggml_ops_mask.cu): generate the verify kernel's
// [kvLen, N] F16 causal(+windowed) masks straight into their device buffers,
// eliminating the host fill + H2D upload. Bit-identical to the host path.
extern "C" bool tsg_cuda_fill_causal_mask_f16(
    void* mask_dev, int kvLen, int N, int nPast, int window, int validLen);
extern "C" bool tsg_cuda_fill_ring_mask_f16(
    void* mask_dev, int ringRows, int N, int startPos, int window);
extern "C" bool tsg_cuda_sync_stream0(void);
// Highest visible NVIDIA compute capability, ggml's encoding (8.6 -> 860).
// Used by kv_window_needs_cuda_flash_attn_copy to mirror ggml-cuda's
// device-dependent flash-attention kernel choice.
extern "C" int tsg_cuda_max_compute_capability(void);
#endif

// ============================================================================
// Shared pieces of the fused transformer kernels, used by the per-model
// translation units (ggml_ops_transformer.cpp, ggml_ops_qwen35_*.cpp,
// ggml_ops_gemma4_*.cpp, ggml_ops_transformer_prefill.cpp): KV-cache sizing
// and circular-window views, flash-attention padding/mask helpers, async
// input upload, and the C# layer-descriptor structs. Header-only (inline)
// so every TU compiles the exact definitions it had when these lived in a
// single file.
// ============================================================================
namespace tsg
{
    // KV-cache element size in bytes for the given GGML tensor type.
    // F32 = 4, F16 = 2. For block-quantized types (Q8_0) the *element size* is
    // fractional (1.0625 bytes for Q8_0) so callers that need a per-element byte
    // count should NOT use this helper - they should use ggml's per-row stride
    // (`tensor->nb[1]`) directly, which already accounts for block padding.
    // We still expose this helper for the linear types because some byte-offset
    // arithmetic happens before the cache tensor is materialised.
    inline std::size_t kv_cache_elem_size(int kv_cache_type)
    {
        switch (static_cast<ggml_type>(kv_cache_type))
        {
            case GGML_TYPE_F32:  return 4;
            case GGML_TYPE_F16:  return 2;
            // Block-quantized: callers should use nb[1] / row_size instead.
            // Returning 0 here makes any accidental `head_dim * elem_size` use
            // visibly wrong rather than silently miscounting.
            case GGML_TYPE_Q8_0: return 0;
            case GGML_TYPE_Q4_0: return 0;
            default:             return 4;
        }
    }

    inline bool kv_cache_is_block_quantized(int kv_cache_type)
    {
        const ggml_type t = static_cast<ggml_type>(kv_cache_type);
        return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
    }

    // Bytes occupied by a [kv_heads, cache_size, head_dim] cache tensor of the
    // given GGML type. Uses ggml_row_size so block-quantized layouts (Q8_0) are
    // accounted for correctly: a Q8_0 row of 256 elements occupies 8 blocks * 34
    // bytes = 272 bytes (vs. 256 raw bytes if we used a fractional 1.0625 value).
    inline std::size_t kv_cache_bytes(int kv_heads, int cache_size, int head_dim, int kv_cache_type = GGML_TYPE_F32)
    {
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        return static_cast<std::size_t>(kv_heads) *
               static_cast<std::size_t>(cache_size) *
               row_bytes;
    }

    inline constexpr int kFlashAttnKvStride = 256;

    inline bool flash_attn_requires_masked_padding(int head_dim)
    {
        // The custom CUDA kernels added for 512/576-dim attention only support
        // the grouped-query path, which expects a non-null mask and a KV length
        // aligned to FATTN_KQ_STRIDE.
        return head_dim == 512 || head_dim == 576;
    }

    inline int flash_attn_kv_length(int valid_len, int cache_size, int head_dim)
    {
        if (!flash_attn_requires_masked_padding(head_dim))
            return valid_len;

        const int padded = ((valid_len + kFlashAttnKvStride - 1) / kFlashAttnKvStride) * kFlashAttnKvStride;
        return std::min(cache_size, std::max(valid_len, padded));
    }

    inline void fill_flash_attn_mask(std::vector<ggml_fp16_t>& mask, int padded_len, int valid_len)
    {
        mask.assign(static_cast<std::size_t>(padded_len), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const int unclamped_valid = std::max(valid_len, 0);
        const int clamped_valid = std::min(unclamped_valid, padded_len);
        std::fill_n(mask.begin(), clamped_valid, static_cast<ggml_fp16_t>(0));
    }

    // Upload one captured-decode INPUT tensor without a per-copy stream sync.
    // On CUDA the copy is queued on the backend stream (ordered ahead of the graph
    // replay that reads it; the post-compute download syncs the stream), so we drop
    // the redundant cudaStreamSynchronize that the synchronous setter issues per
    // call — meaningful when a decode token refreshes ~2*num_layers small inputs.
    // CUDA pageable host->device async is host-synchronous w.r.t. the source, so a
    // caller's transient buffer (e.g. a per-layer mask vector) is safe to free right
    // after this returns. Non-CUDA backends fall back to the synchronous setter.
    inline void decode_input_set_async(ggml_tensor* tensor, const void* data, std::size_t bytes)
    {
        if (tensor == nullptr || data == nullptr || bytes == 0)
            return;
        if (g_backend_type == BACKEND_TYPE_CUDA)
            ggml_backend_tensor_set_async(g_backend, tensor, data, 0, bytes);
        else
            ggml_backend_tensor_set(tensor, data, 0, bytes);
    }

    // Same, for a partial refresh: `offset` bytes into the tensor. Used where the
    // caller knows only a suffix of an input changed since the last replay (the
    // causal decode mask grows by one entry per token), so the H2D drops from the
    // whole buffer to the delta. The tensor's address is fixed by the persistent
    // graph, so the untouched bytes are exactly what the previous replay wrote.
    inline void decode_input_set_range_async(ggml_tensor* tensor, const void* data,
                                             std::size_t offset, std::size_t bytes)
    {
        if (tensor == nullptr || data == nullptr || bytes == 0)
            return;
        if (g_backend_type == BACKEND_TYPE_CUDA)
            ggml_backend_tensor_set_async(g_backend, tensor, data, offset, bytes);
        else
            ggml_backend_tensor_set(tensor, data, offset, bytes);
    }

    // ------------------------------------------------------------------
    // Softmax-gated top-k MoE routing, in the shape ggml-cuda can fuse
    // ------------------------------------------------------------------
    // ggml-cuda collapses a MoE router's whole gating chain into ONE kernel
    // (`ggml_cuda_op_topk_moe`), but `ggml_cuda_topk_moe_fusion` recognises it
    // by literal node sequence:
    //
    //   SOFT_MAX -> RESHAPE -> ARGSORT -> VIEW -> GET_ROWS
    //            [-> RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE]   (norm)
    //
    // which is exactly what llama.cpp's `build_moe_ffn` emits. Two things kept
    // TensorSharp off it:
    //
    //  * `ggml_top_k` is its own op (GGML_OP_TOP_K), not `argsort + view`, so
    //    the pattern never matched. It is also strictly more work — a full sort
    //    of all n_expert scores per token where the fused kernel does a partial
    //    selection — and it hands back a COMPACT [k, tokens] tensor, whereas the
    //    fused kernel recovers n_expert from the ids' row stride
    //    (`ids->nb[1] / ids->nb[0]`), which only holds for the argsort view.
    //  * the reshape of `probs` has to be emitted between the softmax and the
    //    argsort, which is a consequence of build order, not of the maths.
    //
    // Emitting the chain through this helper and expanding `weights` into the
    // graph as a unit (see the callers) gives the DFS the required order.
    //
    // Fusion additionally requires that nothing outside the chain consumes its
    // intermediates: only the ids and the final weights may have external
    // readers. A model that gathers a per-expert scale off `probs_3d` (Gemma 4's
    // `down_exps_scale`) therefore stays unfused, and correctly so.
    struct MoeTopKRouting
    {
        ggml_tensor* ids = nullptr;        // I32 [n_used, n_tokens] — argsort view
        ggml_tensor* weights_3d = nullptr; // F32 [1, n_used, n_tokens]
        ggml_tensor* weights_2d = nullptr; // F32 [n_used, n_tokens]
        ggml_tensor* probs_3d = nullptr;   // F32 [1, n_expert, n_tokens]
    };

    inline MoeTopKRouting build_topk_moe_routing(
        ggml_context* ctx,
        ggml_tensor* logits,        // [n_expert, n_tokens]
        int n_expert,
        int n_expert_used,
        std::int64_t n_tokens,
        bool norm_topk)
    {
        MoeTopKRouting r;
        ggml_tensor* probs = ggml_soft_max(ctx, logits);                     // SOFT_MAX
        r.probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);     // RESHAPE
        r.ids = ggml_argsort_top_k(ctx, probs, n_expert_used);               // ARGSORT + VIEW
        ggml_tensor* w = ggml_get_rows(ctx, r.probs_3d, r.ids);              // GET_ROWS -> [1, n_used, T]

        ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, n_expert_used, n_tokens);
        if (norm_topk)
        {
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                   // SUM_ROWS
            // The clamp is part of the pattern, and is what keeps the division
            // finite when every selected probability underflows.
            w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);       // CLAMP
            w_2d = ggml_div(ctx, w_2d, w_sum);                               // DIV
        }
        r.weights_2d = w_2d;
        r.weights_3d = ggml_reshape_3d(ctx, w_2d, 1, n_expert_used, n_tokens);
        return r;
    }

    // Does a [head_dim, length, kv_heads] window that starts at row `start_idx`
    // of a `cache_size`-row cache have to be MATERIALISED before ggml-cuda's
    // flash attention may read it?
    //
    // ggml-cuda's flash-attention VEC kernel - the one it selects for a
    // SINGLE-ROW decode (fattn.cu `ggml_cuda_get_best_fattn_kernel`, Q->ne[1] == 1
    // on Ada+ with an unquantized KV cache) - returns wrong results when K/V is a
    // view whose ne[1] is a truncated prefix of a longer axis, i.e. one where
    // nb[2] != ne[1]*nb[1]. That is exactly the shape every caller of
    // view_kv_cache_window produces once the live sequence is shorter than the
    // allocated cache: the KV-head stride jumps over the rows the window skips.
    //
    // Measured on Muse-Glimmer (52-token prompt, 8192-row cache, 256-row padded
    // window, 32 query heads over 2 KV heads, F16 cache): with the strided view
    // the first full-attention layer's flash-attention output came back with
    // sum 195.95 against 78.31 for the materialised window, from BYTE-IDENTICAL
    // K, V, Q and mask (verified by hashing the logical window out of the cache
    // after the graph ran), and the error compounded through the remaining layers
    // into a different token stream - fluent but wrong. Forcing the same layer
    // onto the non-flash soft_max path, or widening the window until it covers
    // the whole cache, both reproduce the materialised result.
    //
    // The two narrowings below are what keep this from taxing every decode:
    //
    //  * `length % kFlashAttnKvStride != 0` - ggml-cuda can only choose the vec
    //    kernel when K->ne[1] is a multiple of FATTN_KQ_STRIDE (fattn.cu,
    //    `can_use_vector_kernel`). A window sized to the exact live length lands
    //    on the MMA/tile kernels instead, and those honour the strides: the same
    //    Muse-Glimmer prefill (52 query rows, MMA, same strided 256-row window)
    //    produced bit-identical hidden states with and without the copy.
    //
    //  * head dims the vec kernel has no instance for - it needs
    //    Q->ne[0] <= 256, a multiple of 64, and not 192 (same predicate). This is
    //    what keeps the 512/576-dim MLA attention, whose KV length
    //    flash_attn_kv_length() ALWAYS pads to the stride, off the copy.
    //
    //  * `fattn_query_rows != 1` - every vec selection for an UNQUANTIZED K/V in
    //    ggml_cuda_get_best_fattn_kernel requires Q->ne[1] == 1, except Volta's
    //    `Q->ne[1] * gqa_ratio_eff <= 2`, which admits 2 rows only when the
    //    gqa_ratio is odd - i.e. MHA - and every multi-row caller here (verify,
    //    prefill, batched) is GQA. Multi-row graphs land on the MMA/tile kernels,
    //    which honour the strides (measured: see above). Callers whose window is
    //    not read by flash attention at all (materialising ggml_cpy gathers, the
    //    draft head's explicit soft_max attention, concat inputs) pass 0.
    //
    //  * F32 caches - launch_fattn converts a non-F16 K/V to F16 BEFORE the
    //    kernel runs (need_f16_K/V for every vec instance), and the conversion
    //    (to_fp16_nc_cuda / the contiguously-allocated fast path) honours the
    //    source strides, so the kernel never sees the truncated view's nb[2].
    //    Only an F16 cache is handed to the vec kernel with its raw strides.
    //
    //  * block-quantized caches - ggml_cuda_cpy has no q8_0->q8_0 / q4_0->q4_0
    //    kernel, so ggml_cont of such a window ABORTS the process. Those caches
    //    keep the raw view. (They also take the vec kernel's quantized path for
    //    Q->ne[1] <= 2; whether that path shares the fault is checked empirically
    //    per model - the F16 default cache is what every shipped config uses.)
    //
    // NOTE ON THE PREDICATE: it has to be "the window is a sub-range of the
    // cache", NOT `!ggml_is_contiguous(view)`. ggml_is_contiguous_n SKIPS
    // dimensions whose ne is 1, so with a SINGLE KV head - which is what every
    // rank holds under --tp 2 - a truncated window reports itself contiguous and
    // the guard would let the bad shape straight through. --tp 2 reproduced the
    // identical wrong token stream until this predicate stopped consulting
    // ggml_is_contiguous.
    //
    //  * `fattn_gqa_ratio` - THE narrowing that matters at long context, and the
    //    reason this predicate takes a parameter no other caller needs. It is
    //    n_query_heads / n_kv_heads of the flash-attention op that reads the
    //    window, or 0 when the caller does not know. ggml-cuda only reaches its
    //    VEC branch for an unquantized K/V when
    //        cc >= GGML_CUDA_CC_ADA_LOVELACE (890)
    //        && Q->ne[1] == 1 && Q->ne[3] == 1
    //        && !(gqa_ratio > 4 && K->ne[1] >= 8192)
    //    (fattn.cu, ggml_cuda_get_best_fattn_kernel). The `!gqa_opt_applies`
    //    escape one line below it cannot fire for a GQA model with a mask and a
    //    stride-aligned window, so with gqa_ratio >= 2 everything outside that
    //    condition lands on BEST_FATTN_KERNEL_MMA_F16 - the kernel the prefill
    //    path already proves stride-correct. Concretely, for Muse-Glimmer
    //    (gqa_ratio 16) the copy was being taken on EVERY decode step at EVERY
    //    context length while the kernel it guards against is only selected
    //    below 8192 KV rows, and never at all on Turing/Ampere. At 124K that is
    //    26 x 63.6 MB of pure-waste copy per token (3.3 GB of traffic) plus the
    //    same again pinned in the persistent graph's buffer.
    //
    //    CALLER CONTRACT: pass a non-zero ratio only when the flash-attention op
    //    that reads this window has a MASK and `max_bias == 0`. Both are part of
    //    `gqa_opt_applies`, and without them ggml can reach the vec kernel through
    //    the `!gqa_opt_applies && Q->ne[1] == 1` branch even on Turing/Ampere.
    //    Callers that pass 0 keep the old unconditional behaviour.
    //
    // Restricted to CUDA: Metal, Vulkan and CPU were never affected and keep
    // their exact graphs.
    inline bool kv_window_needs_cuda_flash_attn_copy(
        int head_dim, int cache_size, int start_idx, int length, int kv_cache_type,
        int fattn_query_rows, int fattn_gqa_ratio = 0)
    {
        if (g_backend_type != BACKEND_TYPE_CUDA)
            return false;
        // Diagnostic kill-switch (TS_KV_FATTN_COPY=0): hand flash attention the
        // raw strided views again. This is how the fault is REPRODUCED for an
        // A/B inside one binary - never set it in production.
        static const bool copy_enabled = [] {
            const char* e = std::getenv("TS_KV_FATTN_COPY");
            return e == nullptr || e[0] != '0';
        }();
        if (!copy_enabled)
            return false;
        if (fattn_query_rows != 1)
            return false;                               // vec needs Q->ne[1] == 1
        if (start_idx == 0 && length >= cache_size)
            return false;                               // window IS the cache
        if (length % kFlashAttnKvStride != 0)
            return false;                               // vec kernel unreachable
        if (head_dim > 256 || head_dim % 64 != 0 || head_dim == 192)
            return false;                               // no vec instance
        if (static_cast<ggml_type>(kv_cache_type) != GGML_TYPE_F16)
            return false;                               // F32 is converted with
                                                        // stride-aware kernels;
                                                        // quantized can't be cont'd
#ifdef TSG_GGML_USE_CUDA
        // TS_KV_FATTN_COPY=force pins the old unconditional copy, so a field
        // failure after an ExternalProjects/ggml bump that moves the kernel
        // selection can be worked around without a rebuild.
        static const bool copy_forced = [] {
            const char* e = std::getenv("TS_KV_FATTN_COPY");
            return e != nullptr && std::strcmp(e, "force") == 0;
        }();
        if (!copy_forced && fattn_gqa_ratio >= 2)
        {
            // gqa_opt_applies holds (gqa_ratio >= 2, a mask is always present on
            // these paths, max_bias 0, length % 256 == 0 checked above, and every
            // nb is 16-aligned because head_dim % 64 == 0 on an F16 cache), so
            // ggml-cuda's only route to the VEC kernel is the Ada+ branch.
            const int cc = tsg_cuda_max_compute_capability();
            if (cc >= 750 && cc < 890)
                return false;                           // Turing/Ampere -> MMA_F16
            if (cc >= 890 && fattn_gqa_ratio > 4 && length >= 8192)
                return false;                           // Ada+ -> MMA_F16 (fattn.cu)
        }
#endif
        return true;
    }

    // `fattn_query_rows`: ne[1] of the Q tensor of the ggml_flash_attn_ext that
    // reads this window DIRECTLY, 0 if no flash-attention op does (the window
    // feeds a cpy/concat/soft_max chain instead). Defaults to 1 - the shape every
    // single-token decode graph has, and the only one ggml-cuda's vec kernel (the
    // one that misreads truncated views; see kv_window_needs_cuda_flash_attn_copy)
    // can be selected for with an F16/F32 cache.
    inline ggml_tensor* view_kv_cache_window(
        ggml_context* ctx,
        ggml_tensor* cache,
        int head_dim,
        int cache_size,
        int kv_heads,
        int start_idx,
        int length,
        int kv_cache_type = GGML_TYPE_F32,
        int fattn_query_rows = 1,
        // n_query_heads / n_kv_heads of the flash-attention op that reads this
        // window; 0 = unknown, which keeps the conservative always-copy
        // behaviour. See kv_window_needs_cuda_flash_attn_copy.
        int fattn_gqa_ratio = 0)
    {
        if (ctx == nullptr || cache == nullptr || head_dim <= 0 || cache_size <= 0 || kv_heads <= 0 || length <= 0)
            return nullptr;

        start_idx %= cache_size;
        if (start_idx < 0)
            start_idx += cache_size;

        // ggml_row_size handles block-quantized types (Q8_0) correctly: a row of
        // 256 Q8_0 elements is 8 blocks * 34 bytes = 272 bytes, not 256/1.0625.
        // For linear types it reduces to head_dim * sizeof(elem).
        const std::size_t nb1 = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        const std::size_t nb2 = static_cast<std::size_t>(cache_size) * nb1;

        if (start_idx + length <= cache_size)
        {
            ggml_tensor* window = ggml_view_3d(
                ctx,
                cache,
                head_dim,
                length,
                kv_heads,
                nb1,
                nb2,
                static_cast<std::size_t>(start_idx) * nb1);
            if (window != nullptr &&
                kv_window_needs_cuda_flash_attn_copy(head_dim, cache_size, start_idx, length, kv_cache_type,
                                                     fattn_query_rows, fattn_gqa_ratio))
            {
                // See kv_window_needs_cuda_flash_attn_copy above. One copy of the
                // window per affected cache per forward; it only fires when the
                // window is a strict sub-range, so a full-cache read costs nothing.
                window = ggml_cont(ctx, window);
            }
            return window;
        }

        const int tail_length = cache_size - start_idx;
        const int head_length = length - tail_length;
        ggml_tensor* tail = ggml_view_3d(
            ctx,
            cache,
            head_dim,
            tail_length,
            kv_heads,
            nb1,
            nb2,
            static_cast<std::size_t>(start_idx) * nb1);
        ggml_tensor* head = ggml_view_3d(ctx, cache, head_dim, head_length, kv_heads, nb1, nb2, 0);
        if (tail == nullptr || head == nullptr)
            return nullptr;

        // GPU concat kernels only implement F32 inputs. Wrapped circular windows may
        // come from F16/Q8_0 KV caches, so materialize both slices as F32 first.
        if (static_cast<ggml_type>(kv_cache_type) != GGML_TYPE_F32)
        {
            ggml_tensor* tail_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, tail_length, kv_heads);
            ggml_tensor* head_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, head_length, kv_heads);
            if (tail_f32 == nullptr || head_f32 == nullptr)
                return nullptr;

            tail = ggml_cpy(ctx, tail, tail_f32);
            head = ggml_cpy(ctx, head, head_f32);
        }

        return ggml_concat(ctx, tail, head, 1);
    }
}

// Per-layer descriptor for the Qwen3.5/3.6 full-model kernels (decode,
// verify, batched decode). Passed by pointer from C#. Layout MUST match the
// Slot order of TSGgmlQwen35LayerDesc::proj_scales (must match
// Qwen35Model.BuildProjScaleTable). Each slot is the optional sidecar
// per-tensor scale (NVFP4 scale2) multiplying that projection's matmul
// output; 1.0f = no scale.
enum {
    TSQ35_SC_QKV = 0, TSQ35_SC_K = 1, TSQ35_SC_V = 2, TSQ35_SC_O = 3,
    TSQ35_SC_GDN_QKV = 4, TSQ35_SC_GDN_GATE = 5, TSQ35_SC_BETA = 6, TSQ35_SC_ALPHA = 7,
    TSQ35_SC_SSM_OUT = 8, TSQ35_SC_GU = 9, TSQ35_SC_FFN_GATE = 10, TSQ35_SC_FFN_UP = 11,
    TSQ35_SC_DOWN = 12, TSQ35_SC_COUNT = 16,
};

// Multiply a projection's matmul output by its optional 1-element sidecar
// scale tensor. Null tensor = no sidecar, returns x unchanged (so graphs
// without scales are node-for-node identical to before). Emitting the scale
// as MUL(mm, 1-elem F32) matches the exact node pattern ggml-cuda fuses into
// its NVFP4 mmvq epilogue at batch-1.
static inline ggml_tensor* q35_scaled(ggml_context* ctx, ggml_tensor* x, ggml_tensor* sc)
{
    return sc != nullptr ? ggml_mul(ctx, x, sc) : x;
}

// mirror struct in GgmlNative.cs: pointers first, then int64 shapes, then
// int32 scalars, so natural alignment is identical on both sides.
struct TSGgmlQwen35LayerDesc
{
    // --- pointers (host memory); 8-byte each, FIRST per interop convention ---
    void* attn_norm_w;       // [hidden] F32 (input norm, both layer kinds)
    void* post_attn_norm_w;  // [hidden] F32 (FFN input norm)
    // attention layer
    void* qkv_w;             // attn_qkv [hidden, 2*qDim + 2*kDim]
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32
    void* o_w;               // attn_output [qDim, hidden]
    void* k_cache;           // device-resident [kv_heads, cache, head_dim]
    void* v_cache;
    // gated-delta-net layer
    void* gdn_qkv_w;         // attn_qkv (recurrent) [hidden, conv_dim]
    void* gdn_gate_w;        // attn_gate / z [hidden, value_dim]
    void* ssm_beta_w;        // [hidden, num_v_heads]
    void* ssm_alpha_w;       // [hidden, num_v_heads]
    void* conv1d_w;          // [conv_kernel, conv_dim] F32
    void* ssm_dt_w;          // [num_v_heads] F32 (dt bias)
    void* ssm_a_w;           // [num_v_heads] F32 (-exp(A_log))
    void* ssm_norm_w;        // [head_v_dim] F32
    void* ssm_out_w;         // [value_dim, hidden]
    void* conv_state_in;     // host [conv_kernel-1, conv_dim] ggml layout (ne0=time)
    void* delta_state_in;    // host [head_k_dim, head_v_dim, num_v_heads]
    void* conv_state_out;    // host, same layout as conv_state_in
    void* delta_state_out;   // host, same layout as delta_state_in
    // dense FFN
    void* gu_w;              // ffn_gate_up [hidden, 2*ff_dense]
    void* down_w;            // ffn_down [ff_dense, hidden]
    // separate attention K/V (when separate_qkv: qkv_w holds Q+gate [hidden, 2*qDim])
    void* k_w;
    void* v_w;
    // MoE FFN (used when is_moe != 0)
    void* gate_inp_w;        // router [hidden, num_experts]
    void* gate_exps;         // stacked [hidden, expert_ff, num_experts]
    void* up_exps;           // stacked [hidden, expert_ff, num_experts]
    void* down_exps;         // stacked [expert_ff, hidden, num_experts]
    void* shexp_gate_w;      // [hidden, shared_ff]
    void* shexp_up_w;        // [hidden, shared_ff]
    void* shexp_down_w;      // [shared_ff, hidden]
    void* shexp_gate_inp_w;  // [hidden] F32 (shared-expert sigmoid gate)
    // Dense FFN with gate and up UNFUSED. A mixed-quant "UD"/dynamic GGUF can
    // store ffn_gate and ffn_up in different types (IQ2_XS vs IQ2_S, ...), which
    // a single fused tensor cannot represent and which no imatrix-free
    // requantization can reconcile. Those layers keep both tensors as they were
    // quantized and the graph runs two matmuls instead of one. Non-null exactly
    // when gu_w is null.
    void* ffn_gate_w;        // ffn_gate [hidden, ff_dense]
    void* ffn_up_w;          // ffn_up   [hidden, ff_dense]
    // Optional host pointer to TSQ35_SC_COUNT floats: per-projection
    // matmul-output scales (NVFP4 per-tensor scale2 sidecars). Null when no
    // projection of this layer carries a scale.
    void* proj_scales;

    // --- int64 weight shapes/bytes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t gdn_qkv_ne0, gdn_qkv_ne1, gdn_qkv_bytes;
    std::int64_t gdn_gate_ne0, gdn_gate_ne1, gdn_gate_bytes;
    std::int64_t ssm_beta_ne0, ssm_beta_ne1, ssm_beta_bytes;
    std::int64_t ssm_alpha_ne0, ssm_alpha_ne1, ssm_alpha_bytes;
    std::int64_t ssm_out_ne0, ssm_out_ne1, ssm_out_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gate_inp_ne0, gate_inp_ne1, gate_inp_bytes;
    std::int64_t gate_exps_bytes, up_exps_bytes, down_exps_bytes;
    std::int64_t shexp_gate_ne0, shexp_gate_ne1, shexp_gate_bytes;
    std::int64_t shexp_up_ne0, shexp_up_ne1, shexp_up_bytes;
    std::int64_t shexp_down_ne0, shexp_down_ne1, shexp_down_bytes;
    std::int64_t ffn_gate_ne0, ffn_gate_ne1, ffn_gate_bytes;
    std::int64_t ffn_up_ne0, ffn_up_ne1, ffn_up_bytes;

    // --- int32 scalars ---
    std::int32_t struct_bytes;
    std::int32_t is_recurrent;
    std::int32_t is_moe;
    std::int32_t qkv_type, o_type;
    std::int32_t gdn_qkv_type, gdn_gate_type, ssm_beta_type, ssm_alpha_type, ssm_out_type;
    std::int32_t gu_type, down_type;
    std::int32_t ff_dense;
    std::int32_t separate_qkv, k_type, v_type;
    std::int32_t gate_inp_type, gate_exps_type, up_exps_type, down_exps_type;
    std::int32_t shexp_gate_type, shexp_up_type, shexp_down_type;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;
    std::int32_t ffn_gate_type, ffn_up_type;
};

// This layer's sidecar scale tensor for slot `s`, CREATED the first time the
// graph actually consumes it (null when the slot carries no scale).
//
// Creating every non-1.0 slot up front instead (the original shape of this
// code) put slots the layer's branch never reads into the CONTEXT but not into
// the GRAPH. The persistent builders allocate with ggml_backend_alloc_ctx_tensors
// and so covered them by accident, but the non-persist builders allocate from
// the graph (gallocr), leaving those tensors with a null buffer while they were
// still queued for upload -> ggml_backend_tensor_set aborts on
// GGML_ASSERT(buf != NULL && "tensor buffer not set"). A GGUF that carries a
// `blk.N.attn_qkv.weight.scale` sidecar hits it on every layer, because the GDN
// in-projection shares that tensor name: BuildProjScaleTable fills both
// TSQ35_SC_QKV and TSQ35_SC_GDN_QKV from it while each layer kind reads only
// one of the two. Creating on demand keeps "non-null" and "in the graph"
// the same predicate, whatever the branch mix.
template <class LayerTensorsT>
static inline ggml_tensor* q35_psc(ggml_context* ctx, LayerTensorsT& t,
                                   const TSGgmlQwen35LayerDesc& d, int s)
{
    if (d.proj_scales == nullptr)
        return nullptr;
    if (static_cast<const float*>(d.proj_scales)[s] == 1.0f)
        return nullptr;
    if (t.psc[s] == nullptr)
        t.psc[s] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    return t.psc[s];
}

// Per-layer descriptor for the GPT-OSS whole-model decode kernel
// (TSGgml_GptOssModelDecode). Passed by pointer from C#; layout MUST match
// GptOssLayerDecodeArgs in GgmlNative.cs — pointers first, then int64, then
// int32, then float, so natural alignment is identical on both sides.
struct TSGgmlGptOssLayerDesc
{
    // --- pointers (host memory) ---
    void* attn_norm_w;       // [hidden] F32
    void* qkv_w;             // fused QKV [hidden, qDim+2*kDim], or Q-only when separate_qkv
    void* qkv_b;             // F32 [qDim+2*kDim] (or [qDim]); may be null
    void* k_w;               // separate K weight (null unless separate_qkv)
    void* k_b;               // F32 [kDim]; may be null
    void* v_w;
    void* v_b;
    void* o_w;               // attn_output [qDim, hidden]
    void* o_b;               // F32 [hidden]; may be null
    void* k_cache;           // HOST cache [kv_heads, cache_size, head_dim] — identifies the device window
    void* v_cache;
    void* sinks;             // F32 [num_heads] attention sinks; may be null
    void* post_attn_norm_w;  // [hidden] F32 (MoE input norm)
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_b;        // [num_experts] F32; may be null
    void* gate_exps;         // stacked [hidden, expert_ff, num_experts]
    void* gate_exps_b;       // [expert_ff, num_experts] F32; may be null
    void* up_exps;           // stacked [hidden, expert_ff, num_experts]
    void* up_exps_b;         // [expert_ff, num_experts] F32; may be null
    void* down_exps;         // stacked [expert_ff, hidden, num_experts]
    void* down_exps_b;       // [hidden, num_experts] F32; may be null

    // --- int64 weight shapes (per-expert ne0/ne1 + TOTAL bytes for stacked) ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t ge_ne0, ge_ne1, ge_bytes;
    std::int64_t ue_ne0, ue_ne1, ue_bytes;
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_size;         // rows in the HOST cache
    std::int32_t is_swa;             // non-zero: sliding-window layer
    std::int32_t sliding_window;
    std::int32_t rope_n_dims;
    std::int32_t orig_ctx_len;       // RoPE yarn original context length
    std::int32_t kv_cache_type;      // GGML_TYPE_F32 / GGML_TYPE_F16
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t separate_qkv;
    std::int32_t qkv_type, k_type, v_type, o_type;
    std::int32_t ge_type, ue_type, de_type;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;

    // --- float scalars ---
    float eps;
    float rope_base;
    float rope_freq_scale;
    float oai_alpha;
    float oai_limit;

    // --- optional F16 prefill-GEMM weight copies (host; may be null) ---
    // Prefill is compute-bound: at 2048-token chunks the quantized MMQ path
    // runs the big GEMMs at roughly half the tensor-core F16 rate, and forcing
    // cuBLAS on the quantized weights loses even more to per-chunk dequant.
    // When the model has VRAM headroom the C# side dequantizes these once
    // (Q8_0 -> F16 is exact) and the PREFILL kernel binds them instead; the
    // decode kernels ignore these fields (decode is bandwidth-bound and wants
    // the small quantized reads). TS_GPTOSS_PREFILL_F16=1.
    void* qkv_w_f16;
    void* k_w_f16;
    void* v_w_f16;
    void* o_w_f16;
    void* gate_exps_f16;
    void* up_exps_f16;
    void* down_exps_f16;
};

// MoE layer descriptor for the Gemma 4 MoE kernels (layer/model decode,
// verify, batched decode).
// Descriptor passed by pointer from C#. Layout MUST match
// Gemma4MoELayerDecodeDesc in GgmlNative.cs. 8-byte fields (pointers + int64)
// are grouped first, then 4-byte (int32 + float), so natural alignment is
// identical on both sides with no implicit padding surprises.
struct TSGgmlGemma4MoELayerDesc
{
    // --- pointers (host memory) ---
    void* hidden;            // [hidden_size] F32, in/out (residual stream)
    void* attn_norm_w;       // [hidden_size] F32
    void* qkv_w;             // fused QKV, or Q-only when separate_qkv
    void* k_w;               // separate K weight (null unless separate_qkv)
    void* v_w;               // separate V weight (null unless separate_qkv)
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32 (null for shared layers)
    void* o_w;               // attn_output weight
    void* post_attn_norm_w;  // [hidden_size] F32
    void* k_cache;           // [kv_heads, cache_size, head_dim] (donor's for shared)
    void* v_cache;
    void* freq_factors;      // [freq_factors_len] F32 (null for local/no-scaling)
    void* ffn_norm_w;        // [hidden_size] F32
    void* gu_w;              // dense fused gate_up weight [hidden, 2*ff_dense]
    void* down_w;            // dense down weight [ff_dense, hidden]
    void* post_ffw_norm_1_w; // [hidden_size] F32
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_scale;    // [hidden] F32 (null if absent)
    void* pre_ffw_norm_2_w;  // [hidden_size] F32 (expert input norm)
    void* gate_up_exps;      // stacked experts [hidden, 2*ff_moe, num_experts]
    void* down_exps;         // stacked experts [ff_moe, hidden, num_experts]
    void* down_exps_scale;   // [num_experts] F32 (null if absent)
    void* post_ffw_norm_2_w; // [hidden_size] F32
    void* post_ffw_norm_w;   // [hidden_size] F32

    // --- int64 weight shapes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gue_ne0, gue_ne1, gue_bytes; // per-expert ne0/ne1 + TOTAL bytes
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars / shapes ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_size;
    std::int32_t is_local;
    std::int32_t is_shared;
    std::int32_t sliding_window;
    std::int32_t position;
    std::int32_t rope_n_dims;
    std::int32_t kv_cache_type;
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t freq_factors_len;
    std::int32_t qkv_type;
    std::int32_t k_type;
    std::int32_t v_type;
    std::int32_t o_type;
    std::int32_t gu_type;
    std::int32_t down_type;
    std::int32_t gue_type;
    std::int32_t de_type;
    std::int32_t separate_qkv;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;

    // --- float scalars ---
    float eps;
    float rope_base;
    float inv_sqrt_hidden;     // 1/sqrt(hidden_size) for the router
    float layer_output_scale;
};
