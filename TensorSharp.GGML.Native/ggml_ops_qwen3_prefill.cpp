// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace tsg;

// Dense Qwen 3 layer descriptor used by the model-wide prompt graph below.
// Keep the field order in lock-step with Qwen3PrefillLayerArgs in
// GgmlNative.Qwen3Prefill.cs: pointer-sized values, int64 shapes, then int32.
// The explicit sizeof field turns an accidental managed/native layout change
// into a clean fallback instead of reading a weight with the wrong metadata.
struct TSGgmlQwen3PrefillLayerDesc
{
    void* attn_norm_w;
    void* qkv_w;             // fused [Q | K | V]
    void* q_norm_w;
    void* k_norm_w;
    void* o_w;
    void* ffn_norm_w;
    void* gu_w;              // fused [gate | up]
    void* down_w;
    void* k_cache;
    void* v_cache;
    void* qkv_bias;          // optional (Qwen 2 variants)

    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;

    std::int32_t struct_bytes;
    std::int32_t qkv_type;
    std::int32_t o_type;
    std::int32_t gu_type;
    std::int32_t down_type;
};

namespace
{
    constexpr const char* kQwen3PrefillKernel = "Qwen3 model prefill";

    struct LayerTensors
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* qkv_w = nullptr;
        ggml_tensor* q_w = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* q_norm_w = nullptr;
        ggml_tensor* k_norm_w = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* gu_w = nullptr;
        ggml_tensor* gate_w = nullptr;
        ggml_tensor* up_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* qkv_bias = nullptr;
        ggml_tensor* k_cache = nullptr;
        ggml_tensor* v_cache = nullptr;
        ggml_tensor* q_root = nullptr;
        ggml_tensor* k_root = nullptr;
        ggml_tensor* v_root = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
    };

    struct HostBinding
    {
        ggml_tensor* tensor;
        const void* data;
        std::size_t bytes;
    };

    // One causal mask is shared by every layer. GGML stores ne0 contiguously,
    // so each query owns one [kv_len] row.
    void fill_causal_mask(std::vector<ggml_fp16_t>& mask, int kv_len,
                          int valid_len, int start_pos, int num_tokens)
    {
        const ggml_fp16_t neg_inf =
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        mask.resize(static_cast<std::size_t>(kv_len) *
                    static_cast<std::size_t>(num_tokens));
        for (int q = 0; q < num_tokens; ++q)
        {
            ggml_fp16_t* row = mask.data() +
                static_cast<std::size_t>(q) * static_cast<std::size_t>(kv_len);
            const int visible = std::min(valid_len, start_pos + q + 1);
            std::fill(row, row + visible, zero);
            std::fill(row + visible, row + kv_len, neg_inf);
        }
    }

    bool valid_weight(const void* data, std::int64_t ne0, std::int64_t ne1,
                      std::int64_t bytes)
    {
        return data != nullptr && ne0 > 0 && ne1 > 0 && bytes > 0;
    }
}

// N prompt tokens through embedding, every dense Qwen 3 transformer layer,
// final RMS norm, and the tied/untied LM head as ONE GGML graph. K/V rows are
// written directly into the generic resident cache buffers shared with the
// persistent decode kernel. With compute_logits == 0, the graph stops after
// the last layer's K/V projection because that is all a discarded refill chunk
// needs; with compute_logits != 0, only the final layer's last query goes through
// attention/FFN/the LM head.
TSG_EXPORT int TSGgml_Qwen3ModelPrefill(
    const TSGgmlQwen3PrefillLayerDesc* layers, int num_layers,
    const std::int32_t* token_ids, int num_tokens, int start_pos,
    float* logits_data, int vocab_size, int compute_logits,
    void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1,
    std::int64_t token_embd_bytes,
    void* output_norm_data,
    void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1,
    std::int64_t lm_head_bytes,
    int hidden_size, int num_heads, int num_kv_heads, int head_dim,
    int intermediate_size, int max_seq_len, int kv_cache_type,
    float eps, float rope_base, float rope_freq_scale, int rope_mode,
    int rope_original_context, float rope_ext_factor, float rope_attn_factor,
    float rope_beta_fast, float rope_beta_slow)
{
    PhaseTimer pt(kQwen3PrefillKernel);
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || token_ids == nullptr ||
            num_tokens <= 0 || start_pos < 0 || hidden_size <= 0 ||
            num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
            intermediate_size <= 0 || max_seq_len <= 0)
        {
            set_last_error("Qwen3 model prefill: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes !=
            static_cast<std::int32_t>(sizeof(TSGgmlQwen3PrefillLayerDesc)))
        {
            set_last_error("Qwen3 model prefill: descriptor size mismatch.");
            return 0;
        }
        if (num_heads % num_kv_heads != 0 ||
            hidden_size != num_heads * head_dim)
        {
            set_last_error("Qwen3 model prefill: invalid GQA dimensions.");
            return 0;
        }
        const int total_seq_len = start_pos + num_tokens;
        if (total_seq_len > max_seq_len)
        {
            set_last_error("Qwen3 model prefill: KV cache capacity exceeded.");
            return 0;
        }
        if (kv_cache_type != GGML_TYPE_F16 && kv_cache_type != GGML_TYPE_F32 &&
            kv_cache_type != GGML_TYPE_Q8_0 && kv_cache_type != GGML_TYPE_Q4_0)
        {
            set_last_error("Qwen3 model prefill: unsupported KV cache type.");
            return 0;
        }
        if (!valid_weight(token_embd_data, token_embd_ne0, token_embd_ne1,
                          token_embd_bytes) ||
            token_embd_ne0 != hidden_size || token_embd_ne1 != vocab_size ||
            output_norm_data == nullptr)
        {
            set_last_error("Qwen3 model prefill: invalid embedding/final norm.");
            return 0;
        }
        if (compute_logits != 0 &&
            (logits_data == nullptr ||
             !valid_weight(lm_head_data, lm_head_ne0, lm_head_ne1, lm_head_bytes) ||
             lm_head_ne0 != hidden_size || lm_head_ne1 != vocab_size))
        {
            set_last_error("Qwen3 model prefill: invalid LM head/logits output.");
            return 0;
        }

        const int q_dim = num_heads * head_dim;
        const int k_dim = num_kv_heads * head_dim;
        const int qkv_dim = q_dim + 2 * k_dim;
        const int attn_kv_len =
            flash_attn_kv_length(total_seq_len, max_seq_len, head_dim);
        const float attn_scale = 1.0f /
            std::sqrt(static_cast<float>(head_dim));
        // A/B knobs for matching llama.cpp's three independent attention
        // projections and two independent FFN input projections while retaining
        // TensorSharp's one fused, device-resident backing allocation.
        const bool split_qkv = [] {
            const char* e = std::getenv("TS_QWEN3_PREFILL_SPLIT_QKV");
            return e != nullptr && e[0] != '0';
        }();
        const bool split_gu = [] {
            const char* e = std::getenv("TS_QWEN3_PREFILL_SPLIT_GU");
            return e != nullptr && e[0] != '0';
        }();

        // A pooled context only stores tensor/graph metadata. Activations are
        // allocated below with gallocr, which packs by lifetime instead of
        // retaining the multi-token intermediates from all layers at once.
        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024))
        {
            set_last_error("Qwen3 model prefill: context allocation failed.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* token_embd = ggml_new_tensor_2d(
            ctx, static_cast<ggml_type>(token_embd_type),
            token_embd_ne0, token_embd_ne1);
        ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_tokens);
        ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_tokens);
        ggml_set_input(ids);
        ggml_set_input(positions);

        std::vector<ggml_fp16_t> mask_data;
        fill_causal_mask(mask_data, attn_kv_len, total_seq_len,
                         start_pos, num_tokens);
        ggml_tensor* attn_mask = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, attn_kv_len, num_tokens, 1, 1);

        std::vector<LayerTensors> lt(static_cast<std::size_t>(num_layers));
        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = layers[l];
            if (d.struct_bytes !=
                    static_cast<std::int32_t>(sizeof(TSGgmlQwen3PrefillLayerDesc)) ||
                !valid_weight(d.qkv_w, d.qkv_ne0, d.qkv_ne1, d.qkv_bytes) ||
                !valid_weight(d.o_w, d.o_ne0, d.o_ne1, d.o_bytes) ||
                !valid_weight(d.gu_w, d.gu_ne0, d.gu_ne1, d.gu_bytes) ||
                !valid_weight(d.down_w, d.down_ne0, d.down_ne1, d.down_bytes) ||
                d.attn_norm_w == nullptr || d.ffn_norm_w == nullptr ||
                d.k_cache == nullptr || d.v_cache == nullptr ||
                d.qkv_ne0 != hidden_size || d.qkv_ne1 != qkv_dim ||
                d.o_ne0 != q_dim || d.o_ne1 != hidden_size ||
                d.gu_ne0 != hidden_size || d.gu_ne1 != 2LL * intermediate_size ||
                d.down_ne0 != intermediate_size || d.down_ne1 != hidden_size)
            {
                set_last_error("Qwen3 model prefill: invalid layer descriptor.");
                return 0;
            }

            LayerTensors& t = lt[static_cast<std::size_t>(l)];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            t.qkv_w = ggml_new_tensor_2d(ctx,
                static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (split_qkv)
            {
                t.q_w = ggml_view_2d(ctx, t.qkv_w, hidden_size, q_dim,
                    t.qkv_w->nb[1], 0);
                t.k_w = ggml_view_2d(ctx, t.qkv_w, hidden_size, k_dim,
                    t.qkv_w->nb[1],
                    static_cast<std::size_t>(q_dim) * t.qkv_w->nb[1]);
                t.v_w = ggml_view_2d(ctx, t.qkv_w, hidden_size, k_dim,
                    t.qkv_w->nb[1],
                    static_cast<std::size_t>(q_dim + k_dim) * t.qkv_w->nb[1]);
            }
            t.q_norm_w = d.q_norm_w != nullptr
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim) : nullptr;
            t.k_norm_w = d.k_norm_w != nullptr
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim) : nullptr;
            t.qkv_bias = d.qkv_bias != nullptr
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkv_dim) : nullptr;
            t.o_w = ggml_new_tensor_2d(ctx,
                static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            t.gu_w = ggml_new_tensor_2d(ctx,
                static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            if (split_gu)
            {
                t.gate_w = ggml_view_2d(ctx, t.gu_w,
                    hidden_size, intermediate_size, t.gu_w->nb[1], 0);
                t.up_w = ggml_view_2d(ctx, t.gu_w,
                    hidden_size, intermediate_size, t.gu_w->nb[1],
                    static_cast<std::size_t>(intermediate_size) * t.gu_w->nb[1]);
            }
            t.down_w = ggml_new_tensor_2d(ctx,
                static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.k_cache = ggml_new_tensor_3d(ctx,
                static_cast<ggml_type>(kv_cache_type),
                head_dim, max_seq_len, num_kv_heads);
            t.v_cache = ggml_new_tensor_3d(ctx,
                static_cast<ggml_type>(kv_cache_type),
                head_dim, max_seq_len, num_kv_heads);
        }

        ggml_tensor* hidden = ggml_get_rows(ctx, token_embd, ids); // [H, N]
        const bool metal_strided_qkv = g_backend_type == BACKEND_TYPE_METAL;

        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = layers[l];
            LayerTensors& t = lt[static_cast<std::size_t>(l)];
            const bool last_layer = l == num_layers - 1;

            ggml_tensor* normed = ggml_mul(
                ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* q3 = nullptr;
            ggml_tensor* k3 = nullptr;
            ggml_tensor* v3 = nullptr;
            if (split_qkv)
            {
                ggml_tensor* q = ggml_mul_mat(ctx, t.q_w, normed);
                ggml_tensor* k = ggml_mul_mat(ctx, t.k_w, normed);
                ggml_tensor* v = ggml_mul_mat(ctx, t.v_w, normed);
                if (t.qkv_bias != nullptr)
                {
                    q = ggml_add(ctx, q, ggml_view_1d(
                        ctx, t.qkv_bias, q_dim, 0));
                    k = ggml_add(ctx, k, ggml_view_1d(
                        ctx, t.qkv_bias, k_dim,
                        static_cast<std::size_t>(q_dim) * sizeof(float)));
                    v = ggml_add(ctx, v, ggml_view_1d(
                        ctx, t.qkv_bias, k_dim,
                        static_cast<std::size_t>(q_dim + k_dim) * sizeof(float)));
                }
                q3 = ggml_reshape_3d(ctx, q, head_dim, num_heads, num_tokens);
                k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, num_tokens);
                v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, num_tokens);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);
                if (t.qkv_bias != nullptr)
                    qkv = ggml_add(ctx, qkv, t.qkv_bias);

                if (metal_strided_qkv)
                {
                    const std::size_t head_stride =
                        static_cast<std::size_t>(head_dim) * sizeof(float);
                    q3 = ggml_view_3d(ctx, qkv, head_dim, num_heads, num_tokens,
                        head_stride, qkv->nb[1], 0);
                    k3 = ggml_view_3d(ctx, qkv, head_dim, num_kv_heads, num_tokens,
                        head_stride, qkv->nb[1],
                        static_cast<std::size_t>(q_dim) * sizeof(float));
                    v3 = ggml_view_3d(ctx, qkv, head_dim, num_kv_heads, num_tokens,
                        head_stride, qkv->nb[1],
                        static_cast<std::size_t>(q_dim + k_dim) * sizeof(float));
                }
                else
                {
                    ggml_tensor* q = ggml_cont(ctx, ggml_view_2d(
                        ctx, qkv, q_dim, num_tokens, qkv->nb[1], 0));
                    ggml_tensor* k = ggml_cont(ctx, ggml_view_2d(
                        ctx, qkv, k_dim, num_tokens, qkv->nb[1],
                        static_cast<std::size_t>(q_dim) * sizeof(float)));
                    ggml_tensor* v = ggml_cont(ctx, ggml_view_2d(
                        ctx, qkv, k_dim, num_tokens, qkv->nb[1],
                        static_cast<std::size_t>(q_dim + k_dim) * sizeof(float)));
                    q3 = ggml_reshape_3d(ctx, q, head_dim, num_heads, num_tokens);
                    k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, num_tokens);
                    v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, num_tokens);
                }
            }

            if (t.q_norm_w != nullptr)
                q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, eps), t.q_norm_w);
            if (t.k_norm_w != nullptr)
                k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), t.k_norm_w);

            ggml_tensor* q_rope = ggml_rope_ext(
                ctx, q3, positions, nullptr, head_dim, rope_mode,
                rope_original_context, rope_base, rope_freq_scale,
                rope_ext_factor, rope_attn_factor,
                rope_beta_fast, rope_beta_slow);
            ggml_tensor* k_rope = ggml_rope_ext(
                ctx, k3, positions, nullptr, head_dim, rope_mode,
                rope_original_context, rope_base, rope_freq_scale,
                rope_ext_factor, rope_attn_factor,
                rope_beta_fast, rope_beta_slow);

            // Preserve llama.cpp's deliberate Q -> V -> K expansion order.
            // These roots are emitted before either cache write below. Besides
            // keeping independent projection branches together, that ordering
            // reduces backend graph splits and leaves the K RoPE adjacent to its
            // store for backends which fuse those operations. Deferring every
            // root until the graph tail used to make DFS discover K-copy first,
            // then V-copy, and only much later the Q/attention critical path.
            t.q_root = q_rope;
            t.v_root = v3;
            t.k_root = k_rope;

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(
                ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));
            ggml_tensor* v_write = ggml_cont(
                ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));

            const std::size_t cache_offset =
                static_cast<std::size_t>(start_pos) * t.k_cache->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(
                ctx, t.k_cache, head_dim, num_tokens, num_kv_heads,
                t.k_cache->nb[1], t.k_cache->nb[2], cache_offset);
            ggml_tensor* v_dst = ggml_view_3d(
                ctx, t.v_cache, head_dim, num_tokens, num_kv_heads,
                t.v_cache->nb[1], t.v_cache->nb[2], cache_offset);
            t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
            t.v_cpy = ggml_cpy(ctx, v_write, v_dst);

            // A refill chunk whose hidden states are discarded only needs this
            // layer's K/V rows. No later layer consumes the final residual.
            if (last_layer && compute_logits == 0)
                break;

            ggml_tensor* k_full = view_kv_cache_window(
                ctx, t.k_cache, head_dim, max_seq_len, num_kv_heads,
                0, attn_kv_len, kv_cache_type, num_tokens);
            ggml_tensor* v_full = view_kv_cache_window(
                ctx, t.v_cache, head_dim, max_seq_len, num_kv_heads,
                0, attn_kv_len, kv_cache_type, num_tokens);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Qwen3 model prefill: failed to create KV views.");
                return 0;
            }

            ggml_tensor* mask = attn_mask;
            ggml_tensor* residual = hidden;
            int query_count = num_tokens;
            if (last_layer && compute_logits != 0 && num_tokens > 1)
            {
                q_attn = ggml_view_3d(
                    ctx, q_attn, head_dim, 1, num_heads,
                    q_attn->nb[1], q_attn->nb[2],
                    static_cast<std::size_t>(num_tokens - 1) * q_attn->nb[1]);
                mask = ggml_view_2d(
                    ctx, attn_mask, attn_kv_len, 1, attn_mask->nb[1],
                    static_cast<std::size_t>(num_tokens - 1) * attn_mask->nb[1]);
                residual = ggml_view_2d(
                    ctx, hidden, hidden_size, 1, hidden->nb[1],
                    static_cast<std::size_t>(num_tokens - 1) * hidden->nb[1]);
                query_count = 1;
            }

            ggml_tensor* attn = flash_attn_ext_guarded(ctx, "Qwen3 model prefill", q_attn, k_full, v_full, mask,
                attn_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn, q_dim, query_count);

            ggml_tensor* projected = ggml_mul_mat(ctx, t.o_w, attn_flat);
            ggml_tensor* residual1 = ggml_add(ctx, residual, projected);
            ggml_tensor* ffn_normed = ggml_mul(
                ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* activated = nullptr;
            if (split_gu)
            {
                ggml_tensor* gate = ggml_mul_mat(ctx, t.gate_w, ffn_normed);
                ggml_tensor* up = ggml_mul_mat(ctx, t.up_w, ffn_normed);
                activated = ggml_swiglu_split(ctx, gate, up);
            }
            else
            {
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);
                // The fused projection is [gate | up]. ggml_swiglu computes
                // silu(first half) * second half without materialising two
                // strided views and their contiguous copies.
                activated = ggml_swiglu(ctx, gu);
            }
            ggml_tensor* down = ggml_mul_mat(ctx, t.down_w, activated);
            hidden = ggml_add(ctx, residual1, down);
        }

        ggml_tensor* logits = nullptr;
        ggml_tensor* output_norm = nullptr;
        ggml_tensor* lm_head = nullptr;
        if (compute_logits != 0)
        {
            output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            const bool tied = token_embd_data == lm_head_data &&
                token_embd_type == lm_head_type &&
                token_embd_ne0 == lm_head_ne0 &&
                token_embd_ne1 == lm_head_ne1 &&
                token_embd_bytes == lm_head_bytes;
            lm_head = tied ? token_embd : ggml_new_tensor_2d(
                ctx, static_cast<ggml_type>(lm_head_type),
                lm_head_ne0, lm_head_ne1);
            ggml_tensor* final_hidden = ggml_mul(
                ctx, ggml_rms_norm(ctx, hidden, eps), output_norm);
            logits = ggml_mul_mat(ctx, lm_head, final_hidden);
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            ggml_set_output(logits);
        }

        const std::size_t graph_size =
            static_cast<std::size_t>(num_layers) * 96 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // Cache writes are explicit roots: the cache view read by attention does
        // not otherwise carry a dependency edge from ggml_cpy. Expand Q, V and K
        // first, in the same order as llama.cpp's build_attn(), so DFS cannot put
        // the cache-copy branches ahead of the attention critical path.
        for (int l = 0; l < num_layers; ++l)
        {
            LayerTensors& t = lt[static_cast<std::size_t>(l)];
            ggml_build_forward_expand(graph, t.q_root);
            ggml_build_forward_expand(graph, t.v_root);
            ggml_build_forward_expand(graph, t.k_root);
            ggml_build_forward_expand(graph, t.k_cpy);
            ggml_build_forward_expand(graph, t.v_cpy);
        }
        if (logits != nullptr)
            ggml_build_forward_expand(graph, logits);

        pt.mark("build");
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        std::vector<HostBinding> uploads;
        std::vector<BufferHandle> ephemeral_buffers;

        auto bind_or_upload = [&](ggml_tensor* tensor, const void* data,
                                  std::size_t bytes, bool cacheable,
                                  enum ggml_backend_buffer_usage usage =
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (tensor == nullptr || data == nullptr || bytes == 0)
                return;
            void* key = const_cast<void*>(data);
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tensor, key, bytes,
                                           needs_upload, usage))
                {
                    if (needs_upload)
                        uploads.push_back({tensor, data, bytes});
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buffer = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, key, bytes,
                                            cacheable, buffer))
                {
                    if (!cacheable)
                        ephemeral_buffers.emplace_back(buffer);
                    if (ggml_backend_tensor_alloc(buffer, tensor, key) ==
                        GGML_STATUS_SUCCESS)
                        return;
                }
            }
            uploads.push_back({tensor, data, bytes});
        };

        bind_or_upload(token_embd, token_embd_data,
            static_cast<std::size_t>(token_embd_bytes), true);
        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = layers[l];
            LayerTensors& t = lt[static_cast<std::size_t>(l)];
            bind_or_upload(t.attn_norm_w, d.attn_norm_w,
                static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_upload(t.qkv_w, d.qkv_w,
                static_cast<std::size_t>(d.qkv_bytes), true);
            bind_or_upload(t.q_norm_w, d.q_norm_w,
                static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_upload(t.k_norm_w, d.k_norm_w,
                static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_upload(t.qkv_bias, d.qkv_bias,
                static_cast<std::size_t>(qkv_dim) * sizeof(float), true);
            bind_or_upload(t.o_w, d.o_w,
                static_cast<std::size_t>(d.o_bytes), true);
            bind_or_upload(t.ffn_norm_w, d.ffn_norm_w,
                static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_upload(t.gu_w, d.gu_w,
                static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_upload(t.down_w, d.down_w,
                static_cast<std::size_t>(d.down_bytes), true);
            bind_or_upload(t.k_cache, d.k_cache,
                kv_cache_bytes(num_kv_heads, max_seq_len, head_dim,
                               kv_cache_type), true,
                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_upload(t.v_cache, d.v_cache,
                kv_cache_bytes(num_kv_heads, max_seq_len, head_dim,
                               kv_cache_type), true,
                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        bind_or_upload(attn_mask, mask_data.data(),
            mask_data.size() * sizeof(ggml_fp16_t), false,
            GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (output_norm != nullptr)
            bind_or_upload(output_norm, output_norm_data,
                static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        if (lm_head != nullptr && lm_head != token_embd)
            bind_or_upload(lm_head, lm_head_data,
                static_cast<std::size_t>(lm_head_bytes), true);

        pt.mark("bind");
        BufferHandle fallback_buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            // Low-memory liveness packing is the normal path. The context
            // allocator fallback preserves correctness on builds without it.
            fallback_buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (fallback_buffer.value == nullptr)
            {
                set_last_error("Qwen3 model prefill: backend allocation failed.");
                return 0;
            }
        }

        pt.mark("alloc");
        host_read_barrier();
        for (const auto& u : uploads)
            ggml_backend_tensor_set(
                u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        ggml_backend_tensor_set(ids, token_ids, 0,
            static_cast<std::size_t>(num_tokens) * sizeof(std::int32_t));
        std::vector<std::int32_t> pos(static_cast<std::size_t>(num_tokens));
        for (int i = 0; i < num_tokens; ++i)
            pos[static_cast<std::size_t>(i)] = start_pos + i;
        ggml_backend_tensor_set(positions, pos.data(), 0,
            pos.size() * sizeof(std::int32_t));

        const ggml_status status =
            graph_compute_profiled(g_backend, graph, kQwen3PrefillKernel);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3 model prefill: graph execution failed.");
            return 0;
        }
        pt.mark("compute");

        if (logits != nullptr)
        {
            finalize_compute_with_download(
                logits, logits_data,
                static_cast<std::size_t>(vocab_size) * sizeof(float));
            // Unlike the managed per-op path, the caller's pinned array must be
            // fully populated before this ABI call returns. This also closes the
            // Metal async lifetime hole that motivated the model-wide graph.
            host_read_barrier();
            pt.mark("download");
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in Qwen3 model prefill.");
        return 0;
    }
}
