// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// DiffusionGemma fused decode layer.
//
// One GGML graph for an entire diffusion-gemma decode layer applied to the
// canvas (C tokens) while reading the cached prompt K/V (P tokens). This is the
// throughput fix for diffusion-gemma on the Metal/CUDA backends: the per-op C#
// path issues hundreds of separate graph dispatches per layer (each with its
// own context/alloc/compute/host-sync), which leaves the GPU idle in the CPU
// gaps (~30% utilisation). Folding the layer into a single graph amortises the
// build/encode/sync the way TSGgml_Gemma4MoELayerDecode / Gemma4LayerPrefill
// already do for the autoregressive Gemma 4 model.
//
// Layer math (matches DiffusionGemmaModel.DecodeCanvas exactly):
//   normed      = rms_norm(hidden) * attn_norm_w
//   q/k/v       = separate projections of normed
//   q_normed    = rms_norm(q_head) * q_norm_w ; rope(NeoX, freq_factors on global)
//   k_normed    = rms_norm(k_head) * k_norm_w ; rope
//   v_normed    = rms_norm(v_head)            (unweighted, no rope)
//                 (when the layer has no V projection, v sources the *raw* K
//                  projection, mirroring the C# `Ops.Copy(v, k)` fallback)
//   k/v_full    = concat(prompt_kv, fresh_kv)            [hd, P+C, kvHeads]
//   attn        = flash_attn(q, k_full, v_full, mask)    rectangular decode mask
//   attnOut     = rms_norm(o_w @ attn) * post_attn_norm_w ; += hidden  (residual1)
//   dense       = down(gelu(gate(ffn_norm(residual1))) * up(...))
//   mlp         = rms_norm(dense) * post_ffw_norm_1_w
//   moe         = sum_k w_k * down_e(geglu(gate_up_e(pre_ffw_norm_2(residual1))))
//                 on-device top-k routing (rms_norm * 1/sqrt(H) * gate_inp_scale)
//   mlp        += rms_norm(moe) * post_ffw_norm_2_w
//   out         = residual1 + rms_norm(mlp) * post_ffw_norm_w
//   out        *= dec_scale
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace tsg;

extern "C" {

struct TSGgmlDiffusionDecodeLayerDesc
{
    // --- pointers (host memory) ---
    void* hidden;            // [hidden_size * C] F32, in/out (canvas residual stream)
    void* attn_norm_w;       // [hidden_size] F32
    void* q_w;  void* k_w;  void* v_w;   // separate projections (v_w null if no V proj)
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32
    void* o_w;               // attn_output weight
    void* post_attn_norm_w;  // [hidden_size] F32
    void* prompt_k;          // cached prompt K [head_dim, P, kv_heads] F32 (head-first)
    void* prompt_v;          // cached prompt V [head_dim, P, kv_heads] F32
    void* freq_factors;      // [freq_factors_len] F32 (null for local / no scaling)
    void* ffn_norm_w;        // [hidden_size] F32
    void* gate_w; void* up_w; void* down_w;   // dense MLP (separate gate/up/down)
    void* post_ffw_norm_1_w; // [hidden_size] F32
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_scale;    // [hidden] F32 (null if absent)
    void* pre_ffw_norm_2_w;  // [hidden_size] F32 (expert input norm)
    void* gate_up_exps;      // stacked experts [hidden, 2*ff_moe, num_experts]
    void* down_exps;         // stacked experts [ff_moe, hidden, num_experts]
    void* down_exps_scale;   // [num_experts] F32 (null if absent)
    void* post_ffw_norm_2_w; // [hidden_size] F32
    void* post_ffw_norm_w;   // [hidden_size] F32

    // --- int64 weight shapes (ne0, ne1, total raw bytes) ---
    std::int64_t q_ne0, q_ne1, q_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t gate_ne0, gate_ne1, gate_bytes;
    std::int64_t up_ne0, up_ne1, up_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gue_ne0, gue_ne1, gue_bytes;   // per-expert ne0/ne1 + TOTAL bytes
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars / shapes ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t canvas_len;         // C
    std::int32_t prompt_len;         // P
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t is_local;
    std::int32_t has_v_proj;
    std::int32_t sliding_window;
    std::int32_t rope_n_dims;
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t freq_factors_len;
    std::int32_t q_type, k_type, v_type, o_type;
    std::int32_t gate_type, up_type, down_type;
    std::int32_t gue_type, de_type;

    // --- float scalars ---
    float eps;
    float rope_base;
    float inv_sqrt_hidden;     // 1/sqrt(hidden_size) for the router
    float dec_scale;           // per-layer decoder output scalar
};

TSG_EXPORT int TSGgml_DiffusionDecodeLayer(const TSGgmlDiffusionDecodeLayerDesc* d)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (d == nullptr)
        {
            set_last_error("Diffusion decode layer: null descriptor.");
            return 0;
        }
        if (d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlDiffusionDecodeLayerDesc)))
        {
            set_last_error("Diffusion decode layer: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int H = d->hidden_size;
        const int C = d->canvas_len;
        const int P = d->prompt_len;
        const int kvLen = P + C;
        const int hd = d->head_dim;
        const int nH = d->num_heads;
        const int kvH = d->num_kv_heads;
        const int qDim = nH * hd;
        const int kDim = kvH * hd;
        const bool isLocal = d->is_local != 0;
        const bool hasV = d->has_v_proj != 0;
        const int nExp = d->num_experts;
        const int nUsed = d->num_experts_used;
        const float eps = d->eps;
        const std::int64_t ffDense = d->gate_ne1;        // gate output dim
        const std::int64_t ffMoe = d->gue_ne1 / 2;       // fused gate_up -> half is ff
        const int klo = isLocal ? std::max(0, P - d->sliding_window + 1) : 0;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Diffusion decode layer: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // ---- input / weight tensors ----
        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, C);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
        ggml_tensor* freq_factors_t = (d->freq_factors != nullptr && d->freq_factors_len > 0)
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_factors_len) : nullptr;

        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* q_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->q_type), d->q_ne0, d->q_ne1);
        ggml_tensor* k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->k_type), d->k_ne0, d->k_ne1);
        ggml_tensor* v_w = hasV ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->v_type), d->v_ne0, d->v_ne1) : nullptr;
        ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->o_type), d->o_ne0, d->o_ne1);
        ggml_tensor* post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* prompt_k_t = (P > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, P, kvH) : nullptr;
        ggml_tensor* prompt_v_t = (P > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, P, kvH) : nullptr;

        ggml_tensor* ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->gate_type), d->gate_ne0, d->gate_ne1);
        ggml_tensor* up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->up_type), d->up_ne0, d->up_ne1);
        ggml_tensor* down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->down_type), d->down_ne0, d->down_ne1);
        ggml_tensor* post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
        ggml_tensor* gate_inp_scale_t = (d->gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        ggml_tensor* pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->gue_type), d->gue_ne0, d->gue_ne1, nExp);
        ggml_tensor* down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->de_type), d->de_ne0, d->de_ne1, nExp);
        ggml_tensor* down_exps_scale_t = (d->down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
        ggml_tensor* post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        // ===================== Attention =====================
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);   // [H, C]

        ggml_tensor* q_raw = ggml_mul_mat(ctx, q_w, normed);   // [qDim, C]
        ggml_tensor* k_raw = ggml_mul_mat(ctx, k_w, normed);   // [kDim, C]
        ggml_tensor* v_raw = hasV ? ggml_mul_mat(ctx, v_w, normed) : k_raw;   // [kDim, C]

        ggml_tensor* q_heads = ggml_reshape_2d(ctx, q_raw, hd, nH * C);
        ggml_tensor* k_heads = ggml_reshape_2d(ctx, k_raw, hd, kvH * C);
        ggml_tensor* v_heads = ggml_reshape_2d(ctx, v_raw, hd, kvH * C);
        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_heads, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_heads, eps), k_norm_w);
        ggml_tensor* v_normed = ggml_rms_norm(ctx, v_heads, eps);   // unweighted

        ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;
        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, hd, nH, C, 1);
        ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, hd, kvH, C, 1);
        ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, rope_ff,
            d->rope_n_dims, GGML_ROPE_TYPE_NEOX, 0, d->rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor* k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, rope_ff,
            d->rope_n_dims, GGML_ROPE_TYPE_NEOX, 0, d->rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Head-first attention layout (matches Gemma4LayerPrefill):
        //   Q stays a strided permute view [hd, C, nH].
        //   K/V become tight [hd, C, kvH] so they concat with the cached prompt K/V.
        ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);                 // [hd, C, nH]
        ggml_tensor* k_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)), hd, C, kvH);      // [hd, C, kvH]
        ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_normed, hd, kvH, C, 1);
        ggml_tensor* v_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)), hd, C, kvH);     // [hd, C, kvH]

        ggml_tensor* k_attn = k_fresh;
        ggml_tensor* v_attn = v_fresh;
        if (P > 0)
        {
            k_attn = ggml_concat(ctx, prompt_k_t, k_fresh, 1);   // [hd, P+C, kvH]
            v_attn = ggml_concat(ctx, prompt_v_t, v_fresh, 1);
        }

        // Rectangular decode mask: every canvas query sees keys [klo, kvLen).
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, C, 1, 1);
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * C);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int qi = 0; qi < C; qi++)
            {
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(qi) * kvLen];
                for (int ki = 0; ki < kvLen; ki++)
                    row[ki] = (ki < klo) ? neg_inf : zero_val;
            }
        }

        ggml_tensor* flash = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_t, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        if (!backend_supports_op(flash))
        {
            // Caller falls back to the per-op C# attention path on this backend.
            set_last_error("Diffusion decode layer: flash_attn unsupported for this head_dim/backend.");
            return 0;
        }
        // flash output is [hd, nH, C] -> [qDim, C]
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, flash, qDim, C);

        ggml_tensor* o_proj = ggml_mul_mat(ctx, o_w, attn_flat);   // [H, C]
        ggml_tensor* attn_out = ggml_mul(ctx, ggml_rms_norm(ctx, o_proj, eps), post_attn_norm_w);
        ggml_tensor* residual1 = ggml_add(ctx, attn_out, hidden_t);   // [H, C]

        // ===================== Dense shared FFN =====================
        ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);
        ggml_tensor* gate = ggml_mul_mat(ctx, gate_w, ffn_normed);   // [ffDense, C]
        ggml_tensor* up = ggml_mul_mat(ctx, up_w, ffn_normed);       // [ffDense, C]
        ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
        ggml_tensor* dense_down = ggml_mul_mat(ctx, down_w, dense_h); // [H, C]
        ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), post_ffw_norm_1_w);

        // ===================== MoE router (in-graph, multi-token) =====================
        ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);
        route_n = ggml_scale(ctx, route_n, d->inv_sqrt_hidden);
        if (gate_inp_scale_t != nullptr)
            route_n = ggml_mul(ctx, route_n, gate_inp_scale_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, gate_inp_w, route_n);   // [nExp, C]
        ggml_tensor* probs = ggml_soft_max(ctx, logits);               // [nExp, C]
        ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);              // [nUsed, C] i32
        ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, C);
        ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);             // [1, nUsed, C]
        ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, C);
        ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                 // [1, C]
        w_2d = ggml_div(ctx, w_2d, w_sum);
        if (down_exps_scale_t != nullptr)
        {
            // Broadcast the per-expert scale [nExp] across the C canvas tokens so the
            // per-token get_rows (indexed by sel [nUsed, C]) lines up: a->ne[2] must equal C.
            ggml_tensor* scale_r = ggml_reshape_3d(ctx, down_exps_scale_t, 1, nExp, 1);
            ggml_tensor* scale_bcast = ggml_repeat_4d(ctx, scale_r, 1, nExp, C, 1);  // [1, nExp, C]
            ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_bcast, sel);           // [1, nUsed, C]
            w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, C));
        }
        ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, C);

        // ===================== MoE experts =====================
        ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), pre_ffw_norm_2_w);  // [H, C]
        ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, C);
        ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_up_exps_t, moe_in_3d, sel);   // [2*ffMoe, nUsed, C]
        ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
        ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);                // [ffMoe, nUsed, C]
        ggml_tensor* moe_down = ggml_mul_mat_id(ctx, down_exps_t, moe_act, sel);        // [H, nUsed, C]
        ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                       // broadcast [H, nUsed, C]

        // aggregate over the nUsed dim -> [H, C]
        ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, C, weighted->nb[2], 0);
        for (int u = 1; u < nUsed; ++u)
        {
            ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, C, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
            moe_out = ggml_add(ctx, moe_out, view_u);
        }
        ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), post_ffw_norm_2_w);
        mlp = ggml_add(ctx, mlp, moe_normed);

        // ===================== Final residual + decoder scalar =====================
        ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), post_ffw_norm_w);
        ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
        if (std::fabs(d->dec_scale - 1.0f) > 1e-9f)
            result = ggml_scale(ctx, result, d->dec_scale);

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, C);
        ggml_tensor* out_cpy = ggml_cpy(ctx, result, hidden_out);
        ggml_set_output(out_cpy);

        // --- build graph ---
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(q_w, d->q_w, static_cast<std::size_t>(d->q_bytes), true);
        bind_or_mark(k_w, d->k_w, static_cast<std::size_t>(d->k_bytes), true);
        if (hasV) bind_or_mark(v_w, d->v_w, static_cast<std::size_t>(d->v_bytes), true);
        bind_or_mark(o_w, d->o_w, static_cast<std::size_t>(d->o_bytes), true);
        bind_or_mark(gate_w, d->gate_w, static_cast<std::size_t>(d->gate_bytes), true);
        bind_or_mark(up_w, d->up_w, static_cast<std::size_t>(d->up_bytes), true);
        bind_or_mark(down_w, d->down_w, static_cast<std::size_t>(d->down_bytes), true);
        bind_or_mark(gate_up_exps_t, d->gate_up_exps, static_cast<std::size_t>(d->gue_bytes), true);
        bind_or_mark(down_exps_t, d->down_exps, static_cast<std::size_t>(d->de_bytes), true);
        bind_or_mark(gate_inp_w, d->gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);

        bind_or_mark(attn_norm_w, d->attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_attn_norm_w, d->post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(ffn_norm_w, d->ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_1_w, d->post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(pre_ffw_norm_2_w, d->pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_2_w, d->post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_w, d->post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(q_norm_w, d->q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
        bind_or_mark(k_norm_w, d->k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
        if (gate_inp_scale_t != nullptr)
            bind_or_mark(gate_inp_scale_t, d->gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
        if (down_exps_scale_t != nullptr)
            bind_or_mark(down_exps_scale_t, d->down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);

        if (P > 0)
        {
            std::size_t pk_bytes = static_cast<std::size_t>(hd) * P * kvH * sizeof(float);
            bind_or_mark(prompt_k_t, d->prompt_k, pk_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(prompt_v_t, d->prompt_v, pk_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        bind_or_mark(mask_t, mask_data.data(), mask_data.size() * sizeof(ggml_fp16_t), false);

        // Reuse a persistent compute buffer across the per-layer calls (as Gemma4MoELayerDecode
        // does) so we don't pay a fresh backend allocation for every layer of every step.
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Diffusion decode layer: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, d->hidden, 0, static_cast<std::size_t>(H) * C * sizeof(float));
        std::vector<std::int32_t> pos_data(C);
        for (int i = 0; i < C; i++) pos_data[i] = P + i;
        ggml_backend_tensor_set(pos_tensor, pos_data.data(), 0, static_cast<std::size_t>(C) * sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, d->freq_factors, 0, static_cast<std::size_t>(d->freq_factors_len) * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Diffusion decode layer: graph execution failed.");
            return 0;
        }

        finalize_compute_with_download(hidden_out, d->hidden, static_cast<std::size_t>(H) * C * sizeof(float));
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
        set_last_error("Unknown error in diffusion decode layer.");
        return 0;
    }
}

// ============================================================================
// Model-wide DiffusionGemma decode: ALL layers + output_norm + lm_head + softcap
// as ONE GGML graph, dispatched/synchronised once per denoising step. This keeps
// the canvas hidden state on-device across all layers (no inter-layer host
// round-trip), which is what actually closes the gap to llama.cpp-class speed -
// the per-layer kernel above is correct but serialises on the host hop between
// layers (mirrors TSGgml_Gemma4MoEModelDecode vs the per-layer Gemma4 decode).
//
// `layers` is one TSGgmlDiffusionDecodeLayerDesc per layer; the per-desc hidden /
// canvas_len / prompt_len are ignored (taken from the shared params). Outputs the
// canvas logits [vocab, C] (already softcapped) to logits_out.
// ============================================================================
TSG_EXPORT int TSGgml_DiffusionModelDecode(
    const TSGgmlDiffusionDecodeLayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int canvas_len, int prompt_len,
    void* output_norm_w_data,
    void* lm_head_w_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    void* logits_out, int vocab, float final_logit_softcap)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Diffusion model decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlDiffusionDecodeLayerDesc)))
        {
            set_last_error("Diffusion model decode: descriptor size mismatch.");
            return 0;
        }

        const int H = hidden_size;
        const int C = canvas_len;
        const int P = prompt_len;
        const int kvLen = P + C;
        const float eps = layers[0].eps;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Diffusion model decode: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, C);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
        // These are gallocr-allocated and written via ggml_backend_tensor_set after graph allocation, so
        // they must be flagged as inputs or gallocr may reuse their memory for compute intermediates.
        ggml_set_input(hidden_t);
        ggml_set_input(pos_tensor);

        void* freq_data = nullptr; int freq_len = 0;
        for (int l = 0; l < num_layers; l++)
            if (layers[l].freq_factors != nullptr && layers[l].freq_factors_len > 0)
            { freq_data = layers[l].freq_factors; freq_len = layers[l].freq_factors_len; break; }
        ggml_tensor* freq_factors_t = freq_data ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_len) : nullptr;

        struct LT {
            ggml_tensor *attn_norm_w, *q_w, *k_w, *v_w, *q_norm_w, *k_norm_w, *o_w, *post_attn_norm_w;
            ggml_tensor *prompt_k_t, *prompt_v_t, *ffn_norm_w, *gate_w, *up_w, *down_w, *post_ffw_norm_1_w;
            ggml_tensor *gate_inp_w, *gate_inp_scale_t, *pre_ffw_norm_2_w, *gate_up_exps_t, *down_exps_t;
            ggml_tensor *down_exps_scale_t, *post_ffw_norm_2_w, *post_ffw_norm_w, *mask_t;
            std::vector<ggml_fp16_t> mask_data;
        };
        std::vector<LT> lt(num_layers);

        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlDiffusionDecodeLayerDesc& d = layers[l];
            LT& t = lt[l];
            const int hd = d.head_dim;
            const int nH = d.num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const bool isLocal = d.is_local != 0;
            const bool hasV = d.has_v_proj != 0;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gate_ne1;
            const std::int64_t ffMoe = d.gue_ne1 / 2;
            const int klo = isLocal ? std::max(0, P - d.sliding_window + 1) : 0;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.q_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.q_type), d.q_ne0, d.q_ne1);
            t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
            t.v_w = hasV ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1) : nullptr;
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.prompt_k_t = (P > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, P, kvH) : nullptr;
            t.prompt_v_t = (P > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, P, kvH) : nullptr;
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_type), d.gate_ne0, d.gate_ne1);
            t.up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.up_type), d.up_ne0, d.up_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
            t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* q_raw = ggml_mul_mat(ctx, t.q_w, normed);
            ggml_tensor* k_raw = ggml_mul_mat(ctx, t.k_w, normed);
            ggml_tensor* v_raw = hasV ? ggml_mul_mat(ctx, t.v_w, normed) : k_raw;
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, q_raw, hd, nH * C), eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, k_raw, hd, kvH * C), eps), t.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, v_raw, hd, kvH * C), eps);
            ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;
            ggml_tensor* q_roped = ggml_rope_ext(ctx, ggml_reshape_4d(ctx, q_normed, hd, nH, C, 1), pos_tensor, rope_ff,
                d.rope_n_dims, GGML_ROPE_TYPE_NEOX, 0, d.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor* k_roped = ggml_rope_ext(ctx, ggml_reshape_4d(ctx, k_normed, hd, kvH, C, 1), pos_tensor, rope_ff,
                d.rope_n_dims, GGML_ROPE_TYPE_NEOX, 0, d.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
            ggml_tensor* k_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)), hd, C, kvH);
            ggml_tensor* v_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v_normed, hd, kvH, C, 1), 0, 2, 1, 3)), hd, C, kvH);
            ggml_tensor* k_attn = k_fresh, *v_attn = v_fresh;
            if (P > 0) { k_attn = ggml_concat(ctx, t.prompt_k_t, k_fresh, 1); v_attn = ggml_concat(ctx, t.prompt_v_t, v_fresh, 1); }

            t.mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, C, 1, 1);
            t.mask_data.resize(static_cast<std::size_t>(kvLen) * C);
            {
                const ggml_fp16_t ni = ggml_fp32_to_fp16(-INFINITY), zv = ggml_fp32_to_fp16(0.0f);
                for (int qi = 0; qi < C; qi++)
                    for (int ki = 0; ki < kvLen; ki++)
                        t.mask_data[static_cast<std::size_t>(qi) * kvLen + ki] = (ki < klo) ? ni : zv;
            }
            ggml_tensor* flash = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, t.mask_t, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
            if (l == 0 && !backend_supports_op(flash))
            {
                set_last_error("Diffusion model decode: flash_attn unsupported for this head_dim/backend.");
                return 0;
            }
            ggml_tensor* o_proj = ggml_mul_mat(ctx, t.o_w, ggml_reshape_2d(ctx, flash, qDim, C));
            ggml_tensor* residual1 = ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, o_proj, eps), t.post_attn_norm_w), hidden);

            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, ggml_mul_mat(ctx, t.gate_w, ffn_normed)), ggml_mul_mat(ctx, t.up_w, ffn_normed));
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_mul_mat(ctx, t.down_w, dense_h), eps), t.post_ffw_norm_1_w);

            ggml_tensor* route_n = ggml_scale(ctx, ggml_rms_norm(ctx, residual1, eps), d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr) route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, t.gate_inp_w, route_n));   // [nExp, C]
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);                                     // [nUsed, C]
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, nExp, C), sel), nUsed, C);
            w_2d = ggml_div(ctx, w_2d, ggml_sum_rows(ctx, w_2d));
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_bcast = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), 1, nExp, C, 1);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, ggml_get_rows(ctx, scale_bcast, sel), nUsed, C));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, C);

            ggml_tensor* moe_in = ggml_reshape_3d(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w), H, 1, C);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in, sel);   // [2*ffMoe, nUsed, C]
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, ggml_geglu_split(ctx, moe_gate, moe_up), sel);   // [H, nUsed, C]
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, C, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
                moe_out = ggml_add(ctx, moe_out, ggml_view_2d(ctx, weighted, H, C, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]));
            mlp = ggml_add(ctx, mlp, ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w));

            ggml_tensor* result = ggml_add(ctx, residual1, ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w));
            if (std::fabs(d.dec_scale - 1.0f) > 1e-9f) result = ggml_scale(ctx, result, d.dec_scale);
            hidden = result;
        }

        // If the caller supplies the lm_head, fold output_norm + lm_head + softcap into the graph and output
        // canvas logits [vocab, C]. Otherwise output the final hidden [H, C] (C# applies the lm_head tail).
        const bool do_lm_head = lm_head_w_data != nullptr && output_norm_w_data != nullptr && logits_out != nullptr && vocab > 0;
        ggml_tensor* output_norm_w = nullptr;
        ggml_tensor* lm_head_w = nullptr;
        ggml_tensor* out_cpy = nullptr;
        ggml_tensor* download_t = nullptr;
        std::size_t download_bytes = 0;
        void* download_dst = nullptr;
        if (do_lm_head)
        {
            output_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            lm_head_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            ggml_tensor* normed_out = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), output_norm_w);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_w, normed_out);   // [vocab, C]
            // In-place softcap (avoids 3 extra 268 MB tensors) + set the matmul output directly as the
            // graph output (no separate copy), keeping the gallocr peak low.
            if (final_logit_softcap > 0.0f)
            {
                logits = ggml_scale_inplace(ctx, logits, 1.0f / final_logit_softcap);
                logits = ggml_tanh_inplace(ctx, logits);
                logits = ggml_scale_inplace(ctx, logits, final_logit_softcap);
            }
            out_cpy = logits;
            download_t = logits; download_bytes = (std::size_t)vocab * C * sizeof(float); download_dst = logits_out;
        }
        else
        {
            ggml_tensor* hidden_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, C);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out_t);
            download_t = hidden_out_t; download_bytes = (std::size_t)H * C * sizeof(float); download_dst = hidden_data;
        }
        ggml_set_output(out_cpy);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * 200 + 512, false);
        ggml_build_forward_expand(graph, out_cpy);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf); if (ggml_backend_tensor_alloc(buf, tt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlDiffusionDecodeLayerDesc& d = layers[l];
            LT& t = lt[l];
            const int hd = d.head_dim, kvH = d.num_kv_heads, nExp = d.num_experts;
            bind_or_mark(t.q_w, d.q_w, (std::size_t)d.q_bytes, true);
            bind_or_mark(t.k_w, d.k_w, (std::size_t)d.k_bytes, true);
            if (d.has_v_proj != 0) bind_or_mark(t.v_w, d.v_w, (std::size_t)d.v_bytes, true);
            bind_or_mark(t.o_w, d.o_w, (std::size_t)d.o_bytes, true);
            bind_or_mark(t.gate_w, d.gate_w, (std::size_t)d.gate_bytes, true);
            bind_or_mark(t.up_w, d.up_w, (std::size_t)d.up_bytes, true);
            bind_or_mark(t.down_w, d.down_w, (std::size_t)d.down_bytes, true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, (std::size_t)d.gue_bytes, true);
            bind_or_mark(t.down_exps_t, d.down_exps, (std::size_t)d.de_bytes, true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, (std::size_t)H * nExp * sizeof(float), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, (std::size_t)H * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, (std::size_t)hd * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, (std::size_t)hd * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr) bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, (std::size_t)H * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr) bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, (std::size_t)nExp * sizeof(float), true);
            if (P > 0)
            {
                std::size_t pk = (std::size_t)hd * P * kvH * sizeof(float);
                bind_or_mark(t.prompt_k_t, d.prompt_k, pk, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.prompt_v_t, d.prompt_v, pk, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            bind_or_mark(t.mask_t, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        if (freq_factors_t != nullptr) bind_or_mark(freq_factors_t, freq_data, (std::size_t)freq_len * sizeof(float), true);
        if (do_lm_head)
        {
            bind_or_mark(output_norm_w, output_norm_w_data, (std::size_t)H * sizeof(float), true);
            bind_or_mark(lm_head_w, lm_head_w_data, (std::size_t)lm_head_bytes, true);
        }

        // Graph-aware allocation: gallocr reuses memory by tensor lifetime (peak, not sum), so the
        // 30-layer x C-token intermediates fit. Weights/inputs are pre-bound above (zero-copy) and skipped.
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
        {
            if (galloc != nullptr) ggml_gallocr_free(galloc);
            set_last_error("Diffusion model decode: graph allocation failed.");
            return 0;
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, (std::size_t)H * C * sizeof(float));
        std::vector<std::int32_t> pos_data(C);
        for (int i = 0; i < C; i++) pos_data[i] = P + i;
        ggml_backend_tensor_set(pos_tensor, pos_data.data(), 0, (std::size_t)C * sizeof(std::int32_t));
        if (freq_factors_t != nullptr) ggml_backend_tensor_set(freq_factors_t, freq_data, 0, (std::size_t)freq_len * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) { ggml_gallocr_free(galloc); set_last_error("Diffusion model decode: graph execution failed."); return 0; }

        finalize_compute_with_download(download_t, download_dst, download_bytes);
        // When the lm_head is folded in (do_lm_head), download_dst is a raw caller-owned host buffer that
        // C# reads directly — drain the queued async blit before returning (same hazard as
        // TSGgml_DiffusionLmHead). The hidden-output path writes a TensorSharp tensor's storage, whose
        // subsequent GetFloatPtr in C# fires the host-read barrier, so it does not need this.
        if (do_lm_head)
            host_read_barrier();
        ggml_gallocr_free(galloc);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in diffusion model decode."); return 0; }
}

// ============================================================================
// Fused DiffusionGemma lm_head tail: output_norm + lm_head matmul + final-logit
// softcap as ONE small GGML graph (separate from the layers, so there is no
// layer-graph interference). Replaces the C# RMSNorm + AddmmQuant + softcap +
// readback chain (3+ dispatches) with a single fused dispatch. Reads the canvas
// hidden [H, C] and writes canvas logits [vocab, C] to logits_out.
// ============================================================================
TSG_EXPORT int TSGgml_DiffusionLmHead(
    void* hidden_data, int hidden_size, int canvas_len,
    void* output_norm_w_data,
    void* lm_head_w_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    void* logits_out, int vocab, float eps, float final_logit_softcap)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (hidden_data == nullptr || output_norm_w_data == nullptr || lm_head_w_data == nullptr || logits_out == nullptr)
        {
            set_last_error("Diffusion lm_head: invalid arguments.");
            return 0;
        }
        const int H = hidden_size;
        const int C = canvas_len;

        PooledContextHandle context;
        if (!context.init(8 * 1024 * 1024))
        {
            set_last_error("Diffusion lm_head: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, C);
        ggml_set_input(hidden_t);
        ggml_tensor* output_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* lm_head_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);

        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), output_norm_w);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_w, normed);   // [vocab, C]
        if (final_logit_softcap > 0.0f)
        {
            // NON-inplace softcap: the inplace variants on this large gallocr-managed tensor produced
            // wrong results (verified: host-side softcap is correct). Use fresh tensors instead.
            logits = ggml_scale(ctx, logits, 1.0f / final_logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, final_logit_softcap);
        }
        ggml_set_output(logits);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tt, void* data, std::size_t bytes, bool cacheable) {
            if (tt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tt, data, bytes, buf, addr, needs_upload, GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                {
                    if (ggml_backend_tensor_alloc(buf, tt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            upload_list.push_back({tt, data, bytes});
        };
        bind_or_mark(output_norm_w, output_norm_w_data, (std::size_t)H * sizeof(float), true);
        bind_or_mark(lm_head_w, lm_head_w_data, (std::size_t)lm_head_bytes, true);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
        {
            if (galloc != nullptr) ggml_gallocr_free(galloc);
            set_last_error("Diffusion lm_head: graph allocation failed.");
            return 0;
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, (std::size_t)H * C * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) { ggml_gallocr_free(galloc); set_last_error("Diffusion lm_head: graph execution failed."); return 0; }

        finalize_compute_with_download(logits, logits_out, (std::size_t)vocab * C * sizeof(float));
        // logits_out is a raw caller-owned host buffer (a pinned C# float[]), NOT a TensorSharp tensor, so
        // no GetFloatPtr/EnsureHostReadable host-read barrier ever fires for it. In async Metal mode
        // finalize_compute_with_download only QUEUES the device->host blit and returns; the C# caller then
        // reads (and host-softcaps) the buffer immediately. Drain the queued work here so the buffer is
        // fully populated before we return — and before gallocr frees the source tensor the blit reads.
        // Without this the host read races the in-flight download and sees stale/partial logits, observed
        // as the diffusion canvas decoding correctly for the first few positions then collapsing into
        // repetition/garbage on the Metal backend.
        host_read_barrier();
        ggml_gallocr_free(galloc);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in diffusion lm_head."); return 0; }
}

} // extern "C"
