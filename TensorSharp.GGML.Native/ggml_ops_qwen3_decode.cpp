// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/work/TensorSharp
//
// Persistent, model-wide Qwen3 token decode. This specializes the generic dense
// transformer decoder for Qwen3's fused QKV / fused gate-up layout and folds the
// final RMSNorm + quantized LM head into the same Metal submission.

#include "ggml_ops_internal.h"
#include "ggml_ops_attention_alloc.h"
#include "ggml_ops_transformer_common.h"

#include <mutex>

using namespace tsg;

namespace
{
    struct TSGgmlQwen3LayerDesc
    {
        void* attn_norm_w;
        void* qkv_w;
        void* q_norm_w;
        void* k_norm_w;
        void* o_w;
        void* ffn_norm_w;
        void* gu_w;
        void* down_w;
        void* k_cache;
        void* v_cache;

        std::int64_t qkv_bytes;
        std::int64_t o_bytes;
        std::int64_t gu_bytes;
        std::int64_t down_bytes;

        std::int32_t qkv_type;
        std::int32_t o_type;
        std::int32_t gu_type;
        std::int32_t down_type;
        std::int32_t struct_bytes;
    };

    constexpr int kQwen3DecodeCacheCount = 8;
    constexpr int kQwen3MetalKvStride = 128;

    inline int qwen3_metal_kv_stride()
    {
        static const int value = [] {
            const char* env = std::getenv("TS_QWEN3_KV_STRIDE");
            if (env == nullptr || env[0] == '\0') return kQwen3MetalKvStride;
            const long parsed = std::strtol(env, nullptr, 10);
            return parsed >= 1 && parsed <= 512
                ? static_cast<int>(parsed)
                : kQwen3MetalKvStride;
        }();
        return value;
    }

    struct Qwen3DecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* token_in = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* pos_in = nullptr;
        ggml_tensor* mask_in = nullptr;
        std::vector<ggml_tensor*> movable_kv_copies;
        const void* model_key = nullptr;
        const void* cache_key = nullptr;
        int num_layers = 0;
        int hidden_size = 0;
        int vocab_size = 0;
        int window = 0;

        void reset()
        {
            if (buffer != nullptr)
            {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            if (ctx != nullptr)
            {
                ggml_free(ctx);
                ctx = nullptr;
            }
            graph = nullptr;
            token_in = logits_out = pos_in = mask_in = nullptr;
            movable_kv_copies.clear();
            model_key = cache_key = nullptr;
            num_layers = hidden_size = vocab_size = window = 0;
            valid = false;
        }
    };

    struct Qwen3DecodeCachePool
    {
        Qwen3DecodeCache entries[kQwen3DecodeCacheCount];
        std::uint64_t used[kQwen3DecodeCacheCount] = {};
        std::uint64_t clock = 0;

        Qwen3DecodeCache* find(const void* model_key, const void* cache_key, int window)
        {
            for (int i = 0; i < kQwen3DecodeCacheCount; ++i)
            {
                if (entries[i].valid && entries[i].model_key == model_key &&
                    entries[i].cache_key == cache_key && entries[i].window == window)
                {
                    used[i] = ++clock;
                    return &entries[i];
                }
            }
            return nullptr;
        }

        Qwen3DecodeCache& claim(const void* model_key, const void* cache_key, int window)
        {
            for (int i = 0; i < kQwen3DecodeCacheCount; ++i)
            {
                if (entries[i].valid && entries[i].model_key == model_key &&
                    entries[i].cache_key == cache_key && entries[i].window == window)
                {
                    entries[i].reset();
                    used[i] = ++clock;
                    return entries[i];
                }
            }
            for (int i = 0; i < kQwen3DecodeCacheCount; ++i)
            {
                if (!entries[i].valid)
                {
                    entries[i].reset();
                    used[i] = ++clock;
                    return entries[i];
                }
            }
            int lru = 0;
            for (int i = 1; i < kQwen3DecodeCacheCount; ++i)
                if (used[i] < used[lru]) lru = i;
            entries[lru].reset();
            used[lru] = ++clock;
            return entries[lru];
        }

        void drop_by_cache(const void* cache_key)
        {
            if (cache_key == nullptr) return;
            for (auto& entry : entries)
                if (entry.valid && entry.cache_key == cache_key)
                    entry.reset();
        }

        void reset_all()
        {
            for (auto& entry : entries) entry.reset();
        }
    };

    Qwen3DecodeCachePool g_qwen3_decode_pools[TSG_MAX_DEVICES];
    // ModelBase.GpuComputeLock is per managed model, while this retained pool and
    // the Metal backend are process-wide. Hold one native lock for the complete
    // decode/replay lifetime so another model's reset/drop cannot free an entry
    // after find()/claim() has returned it. A narrow metadata lock would still
    // leave the graph and buffer exposed during compute and publication.
    std::mutex g_qwen3_decode_mutex;

    inline Qwen3DecodeCachePool& qwen3_decode_pool()
    {
        return g_qwen3_decode_pools[g_active_rank];
    }

    inline bool valid_type(std::int32_t type)
    {
        return type >= 0 && type < GGML_TYPE_COUNT;
    }

}

TSG_EXPORT int TSGgml_Qwen3ModelDecodeLogits(
    const TSGgmlQwen3LayerDesc* desc, int num_layers,
    int token_id,
    void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1,
    std::int64_t token_embd_bytes,
    int hidden_size, int position,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int intermediate_size, int kv_cache_type,
    float eps, float rope_base, float rope_freq_scale,
    int rope_mode, int rope_original_context,
    float rope_ext_factor, float rope_attn_factor,
    float rope_beta_fast, float rope_beta_slow,
    void* logits_data, int vocab_size,
    void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1,
    std::int64_t lm_head_bytes,
    void* final_norm_data)
{
    try
    {
        std::lock_guard<std::mutex> pool_lock(g_qwen3_decode_mutex);
        if (!ensure_backend()) return 0;

        // This path is tuned and validated for Metal. Other GGML backends retain
        // the generic decoder until they have equivalent replay validation.
        if (g_backend_type != BACKEND_TYPE_METAL)
        {
            set_last_error("Qwen3 fused-logits decode currently requires GGML Metal.");
            return 0;
        }
        if (desc == nullptr || num_layers <= 0 || token_id < 0 ||
            token_embd_data == nullptr || !valid_type(token_embd_type) ||
            token_embd_ne0 != hidden_size || token_embd_ne1 != vocab_size ||
            token_id >= token_embd_ne1 || token_embd_bytes <= 0 ||
            logits_data == nullptr || lm_head_data == nullptr ||
            final_norm_data == nullptr || hidden_size <= 0 || position < 0 ||
            num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
            cache_size <= position || intermediate_size <= 0 || vocab_size <= 0 ||
            !valid_type(kv_cache_type) || !valid_type(lm_head_type) ||
            lm_head_ne0 != hidden_size || lm_head_ne1 != vocab_size ||
            lm_head_bytes <= 0)
        {
            set_last_error("Qwen3 fused-logits decode received invalid arguments.");
            return 0;
        }
        if (desc[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen3LayerDesc)))
        {
            set_last_error("Qwen3 decode descriptor size mismatch (managed " +
                std::to_string(desc[0].struct_bytes) + " vs native " +
                std::to_string(sizeof(TSGgmlQwen3LayerDesc)) + ").");
            return 0;
        }

        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const int qkv_dim = q_dim + 2 * kv_dim;
        const int total_seq_len = position + 1;
        const int kv_stride = qwen3_metal_kv_stride();
        const int window = std::min(cache_size,
            ((total_seq_len + kv_stride - 1) / kv_stride) * kv_stride);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const void* model_key = desc[0].attn_norm_w;
        const void* cache_key = desc[0].k_cache;

        // Separate projection nodes let Metal's graph scheduler overlap Q/K/V
        // and gate/up exactly as llama.cpp does. The fused host buffers remain a
        // useful storage optimization; these are zero-copy tensor views.
        static const bool split_qkv_projections = [] {
            const char* env = std::getenv("TS_QWEN3_SPLIT_QKV");
            return env != nullptr && env[0] == '1';
        }();
        static const bool split_gu_projections = [] {
            const char* env = std::getenv("TS_QWEN3_SPLIT_GU");
            return env != nullptr && env[0] == '1';
        }();

        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = desc[l];
            if (d.struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen3LayerDesc)) ||
                d.attn_norm_w == nullptr || d.qkv_w == nullptr ||
                d.q_norm_w == nullptr || d.k_norm_w == nullptr ||
                d.o_w == nullptr || d.ffn_norm_w == nullptr ||
                d.gu_w == nullptr || d.down_w == nullptr ||
                d.k_cache == nullptr || d.v_cache == nullptr ||
                !valid_type(d.qkv_type) || !valid_type(d.o_type) ||
                !valid_type(d.gu_type) || !valid_type(d.down_type) ||
                d.qkv_bytes <= 0 || d.o_bytes <= 0 ||
                d.gu_bytes <= 0 || d.down_bytes <= 0)
            {
                set_last_error("Qwen3 fused-logits decode received an incomplete layer descriptor.");
                return 0;
            }
        }

        static const bool persist_enabled = [] {
            const char* env = std::getenv("TS_QWEN3_FD_PERSIST");
            return env == nullptr || env[0] != '0';
        }();
        const bool persist = persist_enabled;

        // Reuse a graph while its padded attention bucket is unchanged. Metal
        // re-encodes tensor bindings on replay, so each KV CPY's destination view
        // can move to the new absolute row without changing graph topology.
        Qwen3DecodeCache* cached = persist
            ? qwen3_decode_pool().find(model_key, cache_key, window)
            : nullptr;
        if (cached != nullptr && cached->graph != nullptr &&
            cached->num_layers == num_layers &&
            cached->hidden_size == hidden_size &&
            cached->vocab_size == vocab_size && cached->window == window)
        {
            host_read_barrier();
            const std::int32_t token_value = token_id;
            ggml_backend_tensor_set(cached->token_in, &token_value, 0,
                sizeof(token_value));
            const std::int32_t pos_value = position;
            ggml_backend_tensor_set(cached->pos_in, &pos_value, 0, sizeof(pos_value));

            for (ggml_tensor* copy : cached->movable_kv_copies)
            {
                ggml_tensor* dst = copy != nullptr ? copy->src[1] : nullptr;
                ggml_tensor* base = dst != nullptr ? dst->view_src : nullptr;
                if (base == nullptr || base->data == nullptr)
                {
                    set_last_error("Qwen3 decode retained an invalid movable KV view.");
                    cached->reset();
                    return 0;
                }
                const std::size_t offset =
                    static_cast<std::size_t>(position) * base->nb[1];
                dst->view_offs = offset;
                dst->data = static_cast<char*>(base->data) + offset;
                copy->data = dst->data;
            }

            std::vector<ggml_fp16_t> mask_data;
            fill_flash_attn_mask(mask_data, window, total_seq_len);
            ggml_backend_tensor_set(cached->mask_in, mask_data.data(), 0,
                mask_data.size() * sizeof(ggml_fp16_t));

            const bool async_submit =
                g_async_compute_enabled.load(std::memory_order_acquire);
            const ggml_status status = async_submit
                ? ggml_backend_graph_compute_async(g_backend, cached->graph)
                : tsg::compute_graph(g_backend, cached->graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen3 cached decode graph execution failed.");
                cached->reset();
                return 0;
            }
            finalize_compute_with_download(cached->logits_out, logits_data,
                static_cast<std::size_t>(vocab_size) * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        Qwen3DecodeCache* retained = persist
            ? &qwen3_decode_pool().claim(model_key, cache_key, window)
            : nullptr;

        constexpr std::size_t ctx_size = 16 * 1024 * 1024;
        PooledContextHandle pooled;
        ggml_context* ctx = nullptr;
        if (persist)
        {
            const ggml_init_params params = { ctx_size, nullptr, true };
            ctx = ggml_init(params);
        }
        else if (pooled.init(ctx_size))
        {
            ctx = pooled.value;
        }
        if (ctx == nullptr)
        {
            set_last_error("Failed to create Qwen3 decode context.");
            return 0;
        }

        ggml_tensor* token_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, window, 1, 1, 1);
        ggml_set_input(token_in);
        ggml_set_input(pos_in);
        ggml_set_input(mask_in);

        ggml_tensor* token_embd = ggml_new_tensor_2d(ctx,
            static_cast<ggml_type>(token_embd_type),
            token_embd_ne0, token_embd_ne1);
        ggml_tensor* lm_head = ggml_new_tensor_2d(ctx,
            static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        struct LayerTensors
        {
            ggml_tensor* attn_norm = nullptr;
            ggml_tensor* qkv = nullptr;
            ggml_tensor* q_norm = nullptr;
            ggml_tensor* k_norm = nullptr;
            ggml_tensor* o = nullptr;
            ggml_tensor* ffn_norm = nullptr;
            ggml_tensor* gu = nullptr;
            ggml_tensor* down = nullptr;
            ggml_tensor* k_cache = nullptr;
            ggml_tensor* v_cache = nullptr;
            ggml_tensor* k_copy = nullptr;
            ggml_tensor* v_copy = nullptr;
        };
        std::vector<LayerTensors> layers(static_cast<std::size_t>(num_layers));

        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = desc[l];
            auto& t = layers[static_cast<std::size_t>(l)];
            t.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            t.qkv = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type),
                hidden_size, qkv_dim);
            t.q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            t.k_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            t.o = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type),
                q_dim, hidden_size);
            t.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            t.gu = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type),
                hidden_size, 2LL * intermediate_size);
            t.down = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type),
                intermediate_size, hidden_size);
            t.k_cache = ggml_new_tensor_3d(ctx,
                static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
            t.v_cache = ggml_new_tensor_3d(ctx,
                static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
        }

        ggml_tensor* hidden = ggml_reshape_1d(ctx,
            ggml_get_rows(ctx, token_embd, token_in), hidden_size);
        for (int l = 0; l < num_layers; ++l)
        {
            auto& t = layers[static_cast<std::size_t>(l)];

            ggml_tensor* normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, hidden, eps), t.attn_norm);
            ggml_tensor* normed_2d =
                ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (split_qkv_projections)
            {
                const std::size_t row_bytes = t.qkv->nb[1];
                ggml_tensor* q_w = ggml_view_2d(ctx, t.qkv,
                    hidden_size, q_dim, row_bytes, 0);
                ggml_tensor* k_w = ggml_view_2d(ctx, t.qkv,
                    hidden_size, kv_dim, row_bytes,
                    static_cast<std::size_t>(q_dim) * row_bytes);
                ggml_tensor* v_w = ggml_view_2d(ctx, t.qkv,
                    hidden_size, kv_dim, row_bytes,
                    static_cast<std::size_t>(q_dim + kv_dim) * row_bytes);
                q_raw = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, q_w, normed_2d), q_dim);
                k_raw = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, k_w, normed_2d), kv_dim);
                v_raw = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, v_w, normed_2d), kv_dim);
            }
            else
            {
                ggml_tensor* qkv = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, t.qkv, normed_2d), qkv_dim);
                q_raw = ggml_view_1d(ctx, qkv, q_dim, 0);
                k_raw = ggml_view_1d(ctx, qkv, kv_dim,
                    static_cast<std::size_t>(q_dim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv, kv_dim,
                    static_cast<std::size_t>(q_dim + kv_dim) * sizeof(float));
            }

            ggml_tensor* q_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx,
                    ggml_reshape_2d(ctx, q_raw, head_dim, num_heads), eps),
                t.q_norm);
            ggml_tensor* k_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx,
                    ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads), eps),
                t.k_norm);

            ggml_tensor* q_rope = ggml_rope_ext(ctx,
                ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1),
                pos_in, nullptr, head_dim, rope_mode, rope_original_context,
                rope_base, rope_freq_scale, rope_ext_factor, rope_attn_factor,
                rope_beta_fast, rope_beta_slow);
            ggml_tensor* k_rope = ggml_rope_ext(ctx,
                ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1),
                pos_in, nullptr, head_dim, rope_mode, rope_original_context,
                rope_base, rope_freq_scale, rope_ext_factor, rope_attn_factor,
                rope_beta_fast, rope_beta_slow);

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            // For one token [D,H,1] and [D,1,H] have the same physical order.
            // A metadata reshape avoids two CONT copies per layer on Metal.
            ggml_tensor* k_write = ggml_reshape_3d(ctx, k_rope,
                head_dim, 1, num_kv_heads);
            ggml_tensor* v_write = ggml_reshape_3d(ctx,
                ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1),
                head_dim, 1, num_kv_heads);

            const std::size_t kv_offset =
                static_cast<std::size_t>(position) * t.k_cache->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache,
                head_dim, 1, num_kv_heads,
                t.k_cache->nb[1], t.k_cache->nb[2], kv_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache,
                head_dim, 1, num_kv_heads,
                t.v_cache->nb[1], t.v_cache->nb[2], kv_offset);
            t.k_copy = ggml_cpy(ctx, k_write, k_dst);
            t.v_copy = ggml_cpy(ctx, v_write, v_dst);

            ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache,
                head_dim, cache_size, num_kv_heads, 0, window,
                kv_cache_type, 1, num_heads / num_kv_heads);
            ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache,
                head_dim, cache_size, num_kv_heads, 0, window,
                kv_cache_type, 1, num_heads / num_kv_heads);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Failed to create Qwen3 KV cache window.");
                if (persist) ggml_free(ctx);
                return 0;
            }

            ggml_tensor* attn = flash_attn_ext_guarded(ctx, "Qwen3 model decode", q_attn, k_full, v_full, mask_in, attn_scale, 0.0f, 0.0f,
                nullptr, GGML_PREC_F32);
            ggml_tensor* attn_proj = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, t.o,
                    ggml_reshape_2d(ctx, attn, q_dim, 1)),
                hidden_size);
            ggml_tensor* residual = ggml_add(ctx, hidden, attn_proj);

            ggml_tensor* ffn_in = ggml_mul(ctx,
                ggml_rms_norm(ctx, residual, eps), t.ffn_norm);
            ggml_tensor* ffn_in_2d =
                ggml_reshape_2d(ctx, ffn_in, hidden_size, 1);
            ggml_tensor* gate;
            ggml_tensor* up;
            if (split_gu_projections)
            {
                const std::size_t row_bytes = t.gu->nb[1];
                ggml_tensor* gate_w = ggml_view_2d(ctx, t.gu,
                    hidden_size, intermediate_size, row_bytes, 0);
                ggml_tensor* up_w = ggml_view_2d(ctx, t.gu,
                    hidden_size, intermediate_size, row_bytes,
                    static_cast<std::size_t>(intermediate_size) * row_bytes);
                gate = ggml_mul_mat(ctx, gate_w, ffn_in_2d);
                up = ggml_mul_mat(ctx, up_w, ffn_in_2d);
            }
            else
            {
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu, ffn_in_2d);
                const std::size_t gu_stride =
                    static_cast<std::size_t>(2 * intermediate_size) * sizeof(float);
                gate = ggml_view_2d(ctx, gu,
                    intermediate_size, 1, gu_stride, 0);
                up = ggml_view_2d(ctx, gu,
                    intermediate_size, 1, gu_stride,
                    static_cast<std::size_t>(intermediate_size) * sizeof(float));
            }
            ggml_tensor* activated = ggml_swiglu_split(ctx, gate, up);
            ggml_tensor* ffn_out = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, t.down,
                    ggml_reshape_2d(ctx, activated, intermediate_size, 1)),
                hidden_size);
            hidden = ggml_add(ctx, residual, ffn_out);
        }

        ggml_tensor* output_normed = ggml_mul(ctx,
            ggml_rms_norm(ctx, hidden, eps), final_norm);
        ggml_tensor* logits_out = ggml_reshape_1d(ctx,
            ggml_mul_mat(ctx, lm_head,
                ggml_reshape_2d(ctx, output_normed, hidden_size, 1)),
            vocab_size);
        ggml_set_output(logits_out);

        const std::size_t graph_size =
            static_cast<std::size_t>(num_layers) * 96 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; ++l)
        {
            ggml_build_forward_expand(graph, layers[l].k_copy);
            ggml_build_forward_expand(graph, layers[l].v_copy);
        }
        ggml_build_forward_expand(graph, logits_out);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct Upload
        {
            ggml_tensor* tensor;
            const void* data;
            std::size_t bytes;
        };
        std::vector<Upload> uploads;
        std::vector<BufferHandle> ephemeral_buffers;

        auto bind_or_upload = [&](ggml_tensor* tensor, void* data,
                                  std::size_t bytes, bool cacheable,
                                  ggml_backend_buffer_usage usage =
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (tensor == nullptr || data == nullptr || bytes == 0) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tensor, data, bytes,
                                           needs_upload, usage))
                {
                    if (needs_upload) uploads.push_back({ tensor, data, bytes });
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t host_buffer = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes,
                                            cacheable, host_buffer))
                {
                    if (!cacheable) ephemeral_buffers.emplace_back(host_buffer);
                    if (ggml_backend_tensor_alloc(host_buffer, tensor, data) ==
                        GGML_STATUS_SUCCESS)
                        return;
                }
            }
            uploads.push_back({ tensor, data, bytes });
        };

        for (int l = 0; l < num_layers; ++l)
        {
            const auto& d = desc[l];
            auto& t = layers[static_cast<std::size_t>(l)];
            bind_or_upload(t.attn_norm, d.attn_norm_w,
                static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_upload(t.qkv, d.qkv_w,
                static_cast<std::size_t>(d.qkv_bytes), true);
            bind_or_upload(t.q_norm, d.q_norm_w,
                static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_upload(t.k_norm, d.k_norm_w,
                static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_upload(t.o, d.o_w,
                static_cast<std::size_t>(d.o_bytes), true);
            bind_or_upload(t.ffn_norm, d.ffn_norm_w,
                static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_upload(t.gu, d.gu_w,
                static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_upload(t.down, d.down_w,
                static_cast<std::size_t>(d.down_bytes), true);
            bind_or_upload(t.k_cache, d.k_cache,
                kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type),
                true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_upload(t.v_cache, d.v_cache,
                kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type),
                true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        bind_or_upload(token_embd, token_embd_data,
            static_cast<std::size_t>(token_embd_bytes), true);
        bind_or_upload(lm_head, lm_head_data,
            static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_upload(final_norm, final_norm_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float), true);

        static const bool optimize_graph = [] {
            const char* env = std::getenv("TS_QWEN3_METAL_OPT");
            return env == nullptr || env[0] != '0';
        }();
        if (optimize_graph) optimize_graph_for_metal(graph);

        BufferHandle transient_buffer(nullptr);
        ggml_backend_buffer_t persistent_buffer = nullptr;
        if (persist)
        {
            persistent_buffer = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (persistent_buffer == nullptr)
            {
                set_last_error("Failed to allocate persistent Qwen3 decode buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            transient_buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (transient_buffer.value == nullptr)
            {
                set_last_error("Failed to allocate Qwen3 decode buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (const auto& upload : uploads)
            ggml_backend_tensor_set(upload.tensor,
                resolve_upload_source(upload.data), 0, upload.bytes);
        const std::int32_t token_value = token_id;
        ggml_backend_tensor_set(token_in, &token_value, 0, sizeof(token_value));
        const std::int32_t pos_value = position;
        ggml_backend_tensor_set(pos_in, &pos_value, 0, sizeof(pos_value));
        std::vector<ggml_fp16_t> mask_data;
        fill_flash_attn_mask(mask_data, window, total_seq_len);
        ggml_backend_tensor_set(mask_in, mask_data.data(), 0,
            mask_data.size() * sizeof(ggml_fp16_t));

        const bool async_submit =
            g_async_compute_enabled.load(std::memory_order_acquire);
        const ggml_status status = async_submit
            ? ggml_backend_graph_compute_async(g_backend, graph)
            : tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3 decode graph execution failed.");
            if (persist)
            {
                ggml_backend_buffer_free(persistent_buffer);
                ggml_free(ctx);
            }
            return 0;
        }
        finalize_compute_with_download(logits_out, logits_data,
            static_cast<std::size_t>(vocab_size) * sizeof(float));
        host_read_barrier();

        if (persist && retained != nullptr)
        {
            retained->ctx = ctx;
            retained->buffer = persistent_buffer;
            retained->graph = graph;
            retained->token_in = token_in;
            retained->logits_out = logits_out;
            retained->pos_in = pos_in;
            retained->mask_in = mask_in;
            retained->movable_kv_copies.reserve(
                static_cast<std::size_t>(num_layers) * 2);
            for (int l = 0; l < num_layers; ++l)
            {
                retained->movable_kv_copies.push_back(layers[l].k_copy);
                retained->movable_kv_copies.push_back(layers[l].v_copy);
            }
            retained->model_key = model_key;
            retained->cache_key = cache_key;
            retained->num_layers = num_layers;
            retained->hidden_size = hidden_size;
            retained->vocab_size = vocab_size;
            retained->window = window;
            retained->valid = true;
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
        set_last_error("Unknown error in Qwen3 fused-logits decode.");
        return 0;
    }
}

TSG_EXPORT void TSGgml_Qwen3ResetDecodeCache()
{
    std::lock_guard<std::mutex> pool_lock(g_qwen3_decode_mutex);
    for (auto& pool : g_qwen3_decode_pools) pool.reset_all();
}

TSG_EXPORT void TSGgml_Qwen3DropDecodeCache(void* first_k_cache)
{
    std::lock_guard<std::mutex> pool_lock(g_qwen3_decode_mutex);
    for (auto& pool : g_qwen3_decode_pools) pool.drop_by_cache(first_k_cache);
}
