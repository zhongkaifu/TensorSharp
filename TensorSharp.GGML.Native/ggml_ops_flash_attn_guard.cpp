// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// See ggml_ops_flash_attn_guard.h for why this exists and what it guarantees.

#include "ggml_ops_flash_attn_guard.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_set>

namespace
{
    std::atomic<std::uint64_t> g_fallbacks{0};

    // The [S, N, H, B] score matrix of the explicit path is what flash attention
    // exists to avoid; each chunk of query rows is bounded to this many bytes so
    // a long prefill that falls back cannot allocate O(N * S) at once.
    constexpr std::int64_t kScoreBudgetBytes = 256LL * 1024 * 1024;

    ggml_tensor* as_f32_contiguous(ggml_context* ctx, ggml_tensor* t)
    {
        // ggml_cast copies into a fresh contiguous F32 tensor and honours the
        // source strides (the KV windows handed to flash attention are often
        // strided sub-views of a larger cache), dequantizing Q8_0/Q4_0 caches.
        if (t->type != GGML_TYPE_F32)
            return ggml_cast(ctx, t, GGML_TYPE_F32);
        return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    }

    // One range of query rows. q32/k32 are contiguous F32, v_rows is V as
    // [S, Dv, Hk, Bk], mask is contiguous [S, >= N, Hm, Bm] or null.
    ggml_tensor* attention_rows(
        ggml_context* ctx, ggml_tensor* q32, ggml_tensor* k32, ggml_tensor* v_rows, ggml_tensor* mask,
        float scale, float max_bias, float logit_softcap, ggml_tensor* sinks)
    {
        // [S, N, H, B]: K heads/batches broadcast over Q's.
        ggml_tensor* scores = ggml_mul_mat(ctx, k32, q32);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

        float softmax_scale = scale;
        if (logit_softcap != 0.0f)
        {
            // ggml's flash attention: s = softcap * tanh(q.k * scale / softcap).
            scores = ggml_scale(ctx, scores, scale / logit_softcap);
            scores = ggml_tanh(ctx, scores);
            scores = ggml_scale(ctx, scores, logit_softcap);
            softmax_scale = 1.0f;
        }

        // soft_max_ext reads the first N mask rows and broadcasts Hm/Bm the way
        // flash attention does; ALiBi slopes and sinks match as well.
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask, softmax_scale, max_bias);
        if (sinks != nullptr)
            ggml_soft_max_add_sinks(probs, sinks);

        ggml_tensor* out = ggml_mul_mat(ctx, v_rows, probs);          // [Dv, N, H, B]
        return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));    // [Dv, H, N, B]
    }

    void warn_once(const char* site, ggml_backend_t backend,
                   const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
                   const ggml_tensor* mask)
    {
        static std::mutex mutex;
        static std::unordered_set<std::string> reported;
        const std::string key = site != nullptr ? site : "(unnamed)";
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!reported.insert(key).second)
                return;
        }
        std::fprintf(stderr,
            "[TensorSharp] warning: %s has no flash-attention kernel for %s "
            "(K head %lld, V head %lld, KV rows %lld, query rows %lld, heads %lld/%lld, "
            "K/V %s/%s, mask %s); running this attention as explicit F32 "
            "mul_mat + soft_max instead (same math, slower, more memory). "
            "Reported once per call site.\n",
            backend != nullptr ? ggml_backend_name(backend) : "the backend",
            key.c_str(),
            (long long) k->ne[0], (long long) v->ne[0], (long long) k->ne[1], (long long) q->ne[1],
            (long long) q->ne[2], (long long) k->ne[2],
            ggml_type_name(k->type), ggml_type_name(v->type),
            mask != nullptr ? "yes" : "none");
        std::fflush(stderr);
    }
}

ggml_tensor* tsg_attention_explicit(
    ggml_context* ctx,
    ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
    float scale, float max_bias, float logit_softcap, ggml_tensor* sinks,
    std::int64_t score_budget_bytes)
{
    ggml_tensor* q32 = as_f32_contiguous(ctx, q);
    ggml_tensor* k32 = as_f32_contiguous(ctx, k);
    ggml_tensor* v32 = as_f32_contiguous(ctx, v);
    // [S, Dv, Hk, Bk] so the value matmul broadcasts V heads over the scores'.
    ggml_tensor* v_rows = ggml_cont(ctx, ggml_permute(ctx, v32, 1, 0, 2, 3));

    const std::int64_t S = k->ne[1], N = q->ne[1], H = q->ne[2], B = q->ne[3];

    // soft_max_ext needs exactly S mask columns (and a contiguous mask).
    if (mask != nullptr && mask->ne[0] != S)
        mask = ggml_view_4d(ctx, mask, S, mask->ne[1], mask->ne[2], mask->ne[3],
                            mask->nb[1], mask->nb[2], mask->nb[3], 0);

    const std::int64_t row_bytes = std::max<std::int64_t>(1, S * H * B * static_cast<std::int64_t>(sizeof(float)));
    const std::int64_t budget = score_budget_bytes > 0 ? score_budget_bytes : kScoreBudgetBytes;
    const std::int64_t chunk = std::max<std::int64_t>(1, budget / row_bytes);
    if (N <= chunk)
    {
        ggml_tensor* m = (mask != nullptr && !ggml_is_contiguous(mask)) ? ggml_cont(ctx, mask) : mask;
        return attention_rows(ctx, q32, k32, v_rows, m, scale, max_bias, logit_softcap, sinks);
    }

    ggml_tensor* result = nullptr;                                     // [Dv, H, rows so far, B]
    for (std::int64_t start = 0; start < N; start += chunk)
    {
        const std::int64_t len = std::min(chunk, N - start);
        ggml_tensor* q_c = ggml_cont(ctx, ggml_view_4d(ctx, q32, q32->ne[0], len, H, B,
            q32->nb[1], q32->nb[2], q32->nb[3], static_cast<std::size_t>(start) * q32->nb[1]));
        ggml_tensor* m_c = nullptr;
        if (mask != nullptr)
            m_c = ggml_cont(ctx, ggml_view_4d(ctx, mask, S, len, mask->ne[2], mask->ne[3],
                mask->nb[1], mask->nb[2], mask->nb[3], static_cast<std::size_t>(start) * mask->nb[1]));
        ggml_tensor* part = attention_rows(ctx, q_c, k32, v_rows, m_c, scale, max_bias, logit_softcap, sinks);
        result = result == nullptr ? part : ggml_concat(ctx, result, part, 2);
    }
    return result;
}

ggml_tensor* tsg_flash_attn_ext_guarded(
    ggml_context* ctx, ggml_backend_t backend, const char* site,
    ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
    float scale, float max_bias, float logit_softcap,
    ggml_tensor* sinks, ggml_prec prec)
{
    ggml_tensor* fa = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap);
    if (prec != GGML_PREC_DEFAULT)
        ggml_flash_attn_ext_set_prec(fa, prec);
    if (sinks != nullptr)
        ggml_flash_attn_ext_add_sinks(fa, sinks);
    if (backend == nullptr || ggml_backend_supports_op(backend, fa))
        return fa;

    g_fallbacks.fetch_add(1, std::memory_order_relaxed);
    warn_once(site, backend, q, k, v, mask);
    return tsg_attention_explicit(ctx, q, k, v, mask, scale, max_bias, logit_softcap, sinks);
}

std::uint64_t tsg_flash_attn_fallback_count()
{
    return g_fallbacks.load(std::memory_order_relaxed);
}
