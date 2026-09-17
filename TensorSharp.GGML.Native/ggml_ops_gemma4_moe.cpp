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
#include "ggml_ops_attention_alloc.h"
#include "ggml_ops_transformer_common.h"
#include "gemma4_mm_mask.h"
#include <chrono>
#include <cstdio>

using namespace tsg;

// Kernel name for MoE CPU offload diagnostics (see tsg::HostMoeSegment).
static constexpr const char* kG4MoeLayerKernel = "Gemma4 MoE layer decode";
static constexpr const char* kG4MoeVerifyKernel = "Gemma4 MoE model verify";

// ============================================================================
// Fused single-layer MoE decode (seqLen == 1): runs an ENTIRE Gemma 4 MoE
// transformer block as one GGML graph on the device, eliminating the ~18-20
// per-op C#→GGML dispatches the legacy TransformerBlock issues per MoE layer
// (each of which allocates+frees a Metal buffer and synchronises). Handles:
//   attn_norm → QKV (fused or separate/mixed-quant) → QK/V-norm → RoPE →
//   KV-cache write (circular for SWA) → flash_attn → O-proj →
//   post_attn_norm → residual1
//   ┌ dense shared FFN: ffn_norm → gate_up → gelu*up → down → post_ffw_norm_1
//   └ MoE: in-graph router (rms_norm·1/√H·gate_inp_scale → mul_mat → softmax →
//          top_k → gather+renorm → ×down_exps_scale) → mul_mat_id experts
//          (geglu) → weighted sum → post_ffw_norm_2 → add into dense output
//   result = residual1 + post_ffw_norm(mlp); ×layer_output_scale
// The in-graph router mirrors Gemma4Model.MoERoute + TryMoEForwardResidual
// exactly so the device path is numerically equivalent to the per-op path.
// ============================================================================

TSG_EXPORT int TSGgml_Gemma4MoELayerDecode(const TSGgmlGemma4MoELayerDesc* d)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (d == nullptr)
        {
            set_last_error("Gemma4 MoE layer decode: null descriptor.");
            return 0;
        }
        if (d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE layer decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int H = d->hidden_size;
        const int position = d->position;
        const int totalSeqLen = position + 1;
        const int hd = d->head_dim;
        const int nH = d->num_heads;
        const int kvH = d->num_kv_heads;
        const int qDim = nH * hd;
        const int kDim = kvH * hd;
        const int cacheSize = d->cache_size;
        const bool isLocal = d->is_local != 0;
        const bool isShared = d->is_shared != 0;
        const bool separate_qkv = d->separate_qkv != 0;
        const int kvType = d->kv_cache_type;
        const float eps = d->eps;
        const int nExp = d->num_experts;
        const int nUsed = d->num_experts_used;
        const int attendLen = isLocal ? std::min(totalSeqLen, d->sliding_window) : totalSeqLen;
        const std::int64_t ffDense = d->gu_ne1 / 2;
        const std::int64_t ffMoe = d->gue_ne1 / 2;

        const std::size_t ctx_size = 16 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Gemma4 MoE layer decode: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // --- input / weight tensors ---
        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* freq_factors_t = (d->freq_factors != nullptr && d->freq_factors_len > 0)
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_factors_len) : nullptr;

        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->qkv_type), d->qkv_ne0, d->qkv_ne1);
        ggml_tensor* k_w = separate_qkv ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->k_type), d->k_ne0, d->k_ne1) : nullptr;
        ggml_tensor* v_w = separate_qkv ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->v_type), d->v_ne0, d->v_ne1) : nullptr;
        ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* k_norm_w = isShared ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->o_type), d->o_ne0, d->o_ne1);
        ggml_tensor* post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
        ggml_tensor* v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);

        ggml_tensor* ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->gu_type), d->gu_ne0, d->gu_ne1);
        ggml_tensor* down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->down_type), d->down_ne0, d->down_ne1);
        ggml_tensor* post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
        ggml_tensor* gate_inp_scale_t = (d->gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        ggml_tensor* pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        // MoE CPU offload: leave the routed-expert tensors null so the bind pass
        // below never asks for a device copy of them. That omission IS the VRAM
        // saving - the host reads the same bytes from the GGUF mmap.
        const bool cpu_moe = d->cpu_moe != 0 && !host_moe_verify_enabled();
        ggml_tensor* gate_up_exps_t = cpu_moe ? nullptr
            : ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->gue_type), d->gue_ne0, d->gue_ne1, nExp);
        ggml_tensor* down_exps_t = cpu_moe ? nullptr
            : ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->de_type), d->de_ne0, d->de_ne1, nExp);
        ggml_tensor* down_exps_scale_t = (d->down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
        ggml_tensor* post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        // ===================== Attention =====================
        ggml_tensor* hidden = hidden_t;
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), attn_norm_w);
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);

        ggml_tensor* q_rope;
        ggml_tensor* k_full = nullptr;
        ggml_tensor* v_full = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
        ggml_tensor* attn_mask = nullptr;
        std::vector<ggml_fp16_t> attn_mask_data;

        const int rope_dims = d->rope_n_dims;
        ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

        if (!isShared)
        {
            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (separate_qkv)
            {
                q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim);
                k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, k_w, normed_2d), kDim);
                v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, v_w, normed_2d), kDim);
            }
            else
            {
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim + 2 * kDim);
                q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
                k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, hd, nH);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, hd, kvH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), k_norm_w);
            ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, hd, kvH);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, hd, kvH, 1);
            q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, hd, kvH, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
            ggml_tensor* v_write = ggml_cont(ctx, v_perm);

            const int cachePos = isLocal ? (position % cacheSize) : position;
            const int activeStart = swa_decode_window_start(isLocal, totalSeqLen, attendLen, cacheSize);
            const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
            const std::size_t kv_byte_offset = static_cast<std::size_t>(cachePos) * k_cached_t->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, k_cached_t, hd, 1, kvH, k_cached_t->nb[1], k_cached_t->nb[2], kv_byte_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, v_cached_t, hd, 1, kvH, v_cached_t->nb[1], v_cached_t->nb[2], kv_byte_offset);
            k_cpy = ggml_cpy(ctx, k_write, k_dst);
            v_cpy = ggml_cpy(ctx, v_write, v_dst);
            if (flash_attn_requires_masked_padding(hd))
            {
                attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                fill_flash_attn_mask(attn_mask_data, attnKvLen, attendLen);
            }
            k_full = view_kv_cache_window(ctx, k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            v_full = view_kv_cache_window(ctx, v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
        }
        else
        {
            ggml_tensor* q_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim);
            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_flat, hd, nH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);

            const int activeStart = swa_decode_window_start(isLocal, totalSeqLen, attendLen, cacheSize);
            const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
            if (flash_attn_requires_masked_padding(hd))
            {
                attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                fill_flash_attn_mask(attn_mask_data, attnKvLen, attendLen);
            }
            k_full = view_kv_cache_window(ctx, k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            v_full = view_kv_cache_window(ctx, v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
        }

        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Gemma4 MoE layer decode: failed to build KV cache views.");
            return 0;
        }

        ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, attn_mask, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
        ggml_tensor* o_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), H);
        ggml_tensor* post_attn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, o_flat, eps), post_attn_norm_w);
        ggml_tensor* residual1 = ggml_add(ctx, post_attn_normed, hidden);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

        // ===================== Dense shared FFN =====================
        ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);
        ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
        ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, gu_w, ffn_normed_2d), 2 * ffDense);
        ggml_tensor* dense_gate = ggml_view_1d(ctx, gu_flat, ffDense, 0);
        ggml_tensor* dense_up = ggml_view_1d(ctx, gu_flat, ffDense, static_cast<std::size_t>(ffDense) * sizeof(float));
        ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
        ggml_tensor* dense_h_2d = ggml_reshape_2d(ctx, dense_h, ffDense, 1);
        ggml_tensor* dense_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, down_w, dense_h_2d), H);
        ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), post_ffw_norm_1_w);

        // ===================== MoE router (in-graph) =====================
        ggml_tensor* route_n = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps);
        route_n = ggml_scale(ctx, route_n, d->inv_sqrt_hidden);
        if (gate_inp_scale_t != nullptr)
            route_n = ggml_mul(ctx, route_n, gate_inp_scale_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, gate_inp_w, route_n);          // [nExp, 1]
        ggml_tensor* probs = ggml_soft_max(ctx, logits);                       // softmax over nExp
        ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);                      // [nUsed, 1] i32
        ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, 1);
        ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                     // [1, nUsed, 1]
        ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, 1);
        ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                         // [1, 1]
        w_2d = ggml_div(ctx, w_2d, w_sum);                                     // renormalised over selected
        if (down_exps_scale_t != nullptr)
        {
            ggml_tensor* scale_r = ggml_reshape_3d(ctx, down_exps_scale_t, 1, nExp, 1);
            ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_r, sel);         // [1, nUsed, 1]
            w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, 1));
        }
        ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, 1);

        // ===================== MoE experts =====================
        ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps), pre_ffw_norm_2_w);
        std::vector<tsg::HostMoeSegment> host_moe;
        ggml_tensor* moe_out_1d = nullptr;
        if (cpu_moe)
        {
            // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
            // Same seam the whole-model kernel uses, with a single segment: the
            // graph pauses after the router, the host multiplies the selected
            // experts straight out of the GGUF mmap, and the rest of the layer
            // (norms, the dense FFN and the residual tail) stays on the GPU.
            tsg::HostMoeSegment hm;
            hm.layer = 0;
            hm.moe_in = ggml_cont(ctx, ggml_reshape_1d(ctx, moe_in, H));
            hm.sel_ids = ggml_cont(ctx, ggml_reshape_1d(ctx, sel, nUsed));
            hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, nUsed));
            ggml_set_output(hm.moe_in);
            ggml_set_output(hm.sel_ids);
            ggml_set_output(hm.weights);

            moe_out_1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_set_input(moe_out_1d);
            ggml_set_output(moe_out_1d);

            hm.moe_out = moe_out_1d;
            hm.gate_data = d->gate_up_exps; hm.gate_type = d->gue_type;
            hm.gate_ne0 = d->gue_ne0;       hm.gate_ne1 = d->gue_ne1;  hm.gate_bytes = d->gue_bytes;
            hm.down_data = d->down_exps;    hm.down_type = d->de_type;
            hm.down_ne0 = d->de_ne0;        hm.down_ne1 = d->de_ne1;   hm.down_bytes = d->de_bytes;
            hm.activation = 2;              // gelu(gate) * up
            hm.num_experts = nExp;
            hm.n_used = nUsed;
            hm.n_ff = (int) ffMoe;
            hm.seq_len = 1;
            hm.hidden = H;
            host_moe.push_back(hm);
        }
        else
        {
            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_up_exps_t, moe_in_3d, sel);   // [2*ffMoe, nUsed, 1]
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);               // [ffMoe, nUsed, 1]
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, down_exps_t, moe_act, sel);       // [H, nUsed, 1]
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                     // broadcast [H, nUsed, 1]

            // aggregate over the nUsed dim -> [H, 1]
            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);
        }
        ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out_1d, eps), post_ffw_norm_2_w);
        mlp = ggml_add(ctx, mlp, moe_normed);

        // ===================== Final residual + layer scale =====================
        ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), post_ffw_norm_w);
        ggml_tensor* result = ggml_add(ctx, mlp_normed, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
        if (std::fabs(d->layer_output_scale - 1.0f) > 1e-9f)
            result = ggml_scale(ctx, result, d->layer_output_scale);

        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* out_cpy = ggml_cpy(ctx, result, hidden_out);
        ggml_set_output(out_cpy);

        // --- build graph (KV writes first to order them before reads) ---
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 2048, false);
        if (k_cpy != nullptr) ggml_build_forward_expand(graph, k_cpy);
        if (v_cpy != nullptr) ggml_build_forward_expand(graph, v_cpy);
        // The host-MoE boundary tensors are not on out_cpy's dependency chain (the
        // expert matmuls were replaced by an input tensor), so expand them first.
        for (const tsg::HostMoeSegment& hm : host_moe)
        {
            ggml_build_forward_expand(graph, hm.moe_in);
            ggml_build_forward_expand(graph, hm.sel_ids);
            ggml_build_forward_expand(graph, hm.weights);
        }
        ggml_build_forward_expand(graph, out_cpy);

        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kG4MoeLayerKernel))
            return 0;

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
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) upload_list.push_back({t, data, bytes});
                    return;
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

        bind_or_mark(qkv_w, d->qkv_w, static_cast<std::size_t>(d->qkv_bytes), true);
        if (separate_qkv)
        {
            bind_or_mark(k_w, d->k_w, static_cast<std::size_t>(d->k_bytes), true);
            bind_or_mark(v_w, d->v_w, static_cast<std::size_t>(d->v_bytes), true);
        }
        bind_or_mark(o_w, d->o_w, static_cast<std::size_t>(d->o_bytes), true);
        bind_or_mark(gu_w, d->gu_w, static_cast<std::size_t>(d->gu_bytes), true);
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
        if (!isShared)
            bind_or_mark(k_norm_w, d->k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
        if (gate_inp_scale_t != nullptr)
            bind_or_mark(gate_inp_scale_t, d->gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
        if (down_exps_scale_t != nullptr)
            bind_or_mark(down_exps_scale_t, d->down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);

        bind_or_mark(k_cached_t, d->k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cached_t, d->v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Allocate intermediates (reuse persistent compute buffer across tokens).
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx, graph))
        {
            buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 MoE layer decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, d->hidden, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, d->freq_factors, 0, static_cast<std::size_t>(d->freq_factors_len) * sizeof(float));

        // MoE CPU offload pauses the graph after the router so the host can
        // multiply the selected experts; everything else stays one submission.
        ggml_status status = GGML_STATUS_SUCCESS;
        if (!host_moe.empty())
        {
            if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kG4MoeLayerKernel))
                status = GGML_STATUS_FAILED;
        }
        else
        {
            status = tsg::compute_graph(g_backend, graph);
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            if (host_moe.empty())
                set_last_error("Gemma4 MoE layer decode: graph execution failed.");
            return 0;
        }

        finalize_compute_with_download(hidden_out, d->hidden, static_cast<std::size_t>(H) * sizeof(float));
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
        set_last_error("Unknown error in Gemma4 MoE layer decode.");
        return 0;
    }
}

// ============================================================================
// Gemma4 MoE MODEL-WIDE decode: the whole transformer (all layers) as ONE
// GGML graph, dispatched/synchronised once per token instead of once per
// layer. This is the throughput fix for MoE Gemma 4 (e.g. gemma-4-26B-A4B):
// the per-layer TSGgml_Gemma4MoELayerDecode rebuilds a graph + encodes a Metal
// command buffer + synchronises ~30x per token, leaving the GPU idle in the
// inter-layer CPU gaps (~60% utilisation). Folding all layers into one graph
// amortises the build/encode/sync across the model, keeping the GPU saturated.
//
// Each layer's graph is byte-for-byte the same construction as the proven
// single-layer kernel above (attention + dense shared FFN + in-graph MoE
// router/experts), just chained: layer L's output residual feeds layer L+1.
// Adding every layer's KV-cache write to the graph before the final output (the
// same ordering the dense TSGgml_Gemma4ModelDecode relies on) guarantees each
// layer's cache write executes before that layer's attention reads it.
//
// Scope: non-shared (no KV-donor) layers, no PLE. The C# caller only routes the
// all-MoE / no-PLE / no-donor shape here and falls back to the per-layer path
// otherwise, so this kernel rejects (returns 0) anything it doesn't handle.
// The per-layer descriptor array reuses TSGgmlGemma4MoELayerDesc unchanged;
// `hidden`/`position` are taken from the shared params, the per-desc copies are
// ignored.
//
// PERSIST / CUDA-graph capture (g_g4moe): the exact analogue of the dense
// g_g4dc. Without it this kernel rebuilds the whole-model graph and re-binds
// every weight per token, and the position-dependent KV-cache view offset +
// attention window make the graph topology change token-to-token, so
// ggml-cuda's CUDA-graph capture never engages — on WDDM the GPU is starved in
// the per-node scheduling gaps (~16 tok/s, 92% host "Other"). With persist the
// graph is built ONCE with stable tensor addresses (raw ggml ctx +
// alloc_ctx_tensors); the KV write uses ggml_set_rows (write row = an I64
// INPUT) and attention reads a FIXED padded window [0, window) with an F16 mask
// INPUT, so the topology is identical token-to-token and capture engages. The
// window is padded up to a 256-token stride so it only changes (forcing a
// rebuild) every 256 tokens. Dropped + rebuilt when any layer window grows a
// stride, the model instance changes, or the KV-cache buffer is reallocated
// (TSGgml_Gemma4MoEResetDecodeCache, called by C# before any prefill / on KV
// reset). MoE Gemma 4 has no PLE and no KV-donor (shared) layers, so the
// persist bookkeeping is simpler than the dense path's.
// ============================================================================
namespace
{
    constexpr int kG4MoePersistKvStride = 256;
    constexpr const char* kG4MoeDecodeKernel = "Gemma4 MoE model decode";

    struct G4MoEDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        std::vector<ggml_tensor*> kv_index;          // per-layer I64 set_rows write row
        std::vector<ggml_tensor*> attn_mask;         // per-layer F16 padding mask
        std::vector<int> layer_window;               // per-layer padded window length
        const void* sig_disc = nullptr;              // model-instance discriminator (attn_norm[0])
        const void* sig_kcache0 = nullptr;           // first KV buffer ptr (detects realloc/grow)
        int num_layers = 0;
        int hidden_size = 0;
        bool folded = false;                         // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                           // floats to download (vocab when folded, else hidden)
        // Tensor-parallel plan: this rank's graph cut at the three row-parallel
        // projections per layer (attention output, dense FFN down, MoE down).
        tsg::TpRankPlan tp_plan;
        // MoE CPU offload: the layers whose routed experts the host multiplies,
        // and the node index each accelerator segment stops at (one per entry in
        // host_moe, plus a final entry for the tail = graph node count). The
        // REPLAY path needs these too — a cached graph run end-to-end would read
        // whatever stale bytes moe_out still held from the previous token.
        std::vector<tsg::HostMoeSegment> host_moe;
        std::vector<int> host_moe_seg_end;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            host_moe.clear(); host_moe_seg_end.clear();
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = 0;
            folded = false; out_count = 0;
            tp_plan.clear();
        }
    };

    // Per-request pool, identical in purpose to g_g4dc_pool (see its comment):
    // concurrent requests each decode through their own KV holder, so a single
    // shared entry would rebuild on every switch and collapse N>=2 throughput.
    // Keyed by (sig_disc, sig_kcache0) so each in-flight request keeps its own
    // persistent captured graph; ggml-cuda caches a cuda graph per nodes[0].
    constexpr int kG4MoeMaxDecodeCaches = 8;
    struct G4MoEDecodeCachePool
    {
        G4MoEDecodeCache entries[kG4MoeMaxDecodeCaches];
        std::uint64_t used[kG4MoeMaxDecodeCaches] = {};
        std::uint64_t clock = 0;

        G4MoEDecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        G4MoEDecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4MoeMaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    // One pool per rank: under tensor parallelism each GPU builds its own
    // graph over its own shards, and the entries own that rank's ggml
    // context and backend buffer.
    G4MoEDecodeCachePool g_g4moe_pools[tsg::TSG_MAX_DEVICES];
}
static G4MoEDecodeCachePool& g_moe_pool() { return g_g4moe_pools[tsg::g_active_rank]; }
namespace
{
}

TSG_EXPORT int TSGgml_Gemma4MoEModelDecode(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int position,
    // Folded final-norm + lm_head (nullable). When logits_data / lm_head_data /
    // final_norm_data are non-null and vocab_size > 0, the graph appends the
    // output RMSNorm, the lm_head matmul and (when logit_softcap > 0) the tanh
    // logit softcap, writing logits[vocab] to logits_data so the whole token —
    // including the 256K-vocab projection — is one captured replay (no separate
    // per-token lm_head graph_compute disturbing the captured graph). The C#
    // caller then skips its own final-norm/lm_head/softcap dispatches.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap,
    // Tensor parallelism. With tp_degree > 1 every weight and KV pointer in
    // `layers` is THIS rank's shard (attention/dense FFN Megatron-split, experts
    // sliced on the intermediate dimension so `sel` stays global), and the call
    // builds and binds the graph instead of running it — returning a plan cut at
    // the three row-parallel projections per layer.
    int tp_degree, void** tp_plan_out)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Gemma4 MoE model decode: invalid arguments.");
            return 0;
        }

        // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
        if (tp_mode)
            *tp_plan_out = nullptr;
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE model decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_shared != 0)
            {
                set_last_error("Gemma4 MoE model decode: KV-donor (shared) layers unsupported; use per-layer path.");
                return 0;
            }
        }

        const int H = hidden_size;
        const int totalSeqLen = position + 1;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        // Fold final-norm + lm_head into the graph when the caller supplies them.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        const int out_count = fold ? vocab_size : H;

        // ===== Persist / CUDA-graph capture setup (mirrors dense g_g4dc) =====
        static const bool g4moe_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();

        std::vector<int> pwindow(num_layers, 0);          // padded window length per layer
        std::vector<int> pvalid(num_layers, 0);           // unmasked (valid) length per layer
        std::vector<std::int64_t> pwrite(num_layers, 0);  // set_rows write row per layer
        // Persist on CUDA and Vulkan; Metal falls through to the ggml_cpy path
        // (ggml_set_rows segfaults there — null-context buffer in
        // ggml_metal_op_set_rows), which still folds the LM head and keeps the
        // KV cache device-resident. On CUDA persist enables CUDA-graph capture;
        // on Vulkan reusing the built graph removes the per-token rebuild +
        // gallocr work, and it is what makes the tensor-parallel plan possible
        // (the driver executes the graph after this call returns). Vulkan's
        // set_rows / mul_mat_id / argsort cover everything this graph emits —
        // see the dense TSGgml_Gemma4ModelDecode for the full rationale.
        bool can_persist = g4moe_persist &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN);
        {
            auto roundup_stride = [](int v){ return ((v + kG4MoePersistKvStride - 1) / kG4MoePersistKvStride) * kG4MoePersistKvStride; };
            for (int l = 0; l < num_layers; l++)
            {
                const int csz = layers[l].cache_size;
                if (csz <= 0) { can_persist = false; break; }
                const bool isLocal = layers[l].is_local != 0;
                if (isLocal)
                {
                    if (totalSeqLen <= csz) { pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen; }
                    else { pwindow[l] = csz; pvalid[l] = csz; }   // saturated: read whole circular cache flat
                    pwrite[l] = position % csz;
                }
                else
                {
                    pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen;
                    pwrite[l] = position;
                    // A global cache that already overflowed can't be expressed as a
                    // single padded window -> let the legacy per-token path handle it.
                    if (pvalid[l] > pwindow[l]) { can_persist = false; break; }
                }
            }
        }
        // The driver runs the graph after this call returns, so it must outlive
        // the call — which is what persist mode provides.
        if (tp_mode && !can_persist)
        {
            set_last_error("Gemma4 MoE model decode: tensor-parallel mode requires the persistent decode graph.");
            return 0;
        }

        const void* g4moe_sig = layers[0].attn_norm_w;
        const void* g4moe_kc0 = layers[0].k_cache;

        // ---- reuse fast-path: replay THIS request's captured graph ----
        // Per-request lookup (model instance + KV holder); find() already matched
        // sig_disc + sig_kcache0, so only the finer shape fields are checked here.
        G4MoEDecodeCache* dc = can_persist ? g_moe_pool().find(g4moe_sig, g4moe_kc0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->layer_window == pwindow &&
            dc->folded == fold && dc->out_count == out_count)
        {
            host_read_barrier();
            // Async per-token input refresh (stream-ordered ahead of the captured
            // replay below); see the dense TSGgml_Gemma4ModelDecode reuse path for
            // the full rationale. Removes ~N redundant per-copy stream syncs/token.
            decode_input_set_async(dc->hidden_in, hidden_data, static_cast<std::size_t>(H) * sizeof(float));
            std::int32_t pos_val = position;
            decode_input_set_async(dc->pos_tensor, &pos_val, sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                if (dc->kv_index[l] != nullptr)
                    decode_input_set_async(dc->kv_index[l], &pwrite[l], sizeof(std::int64_t));
                if (dc->attn_mask[l] != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    decode_input_set_async(dc->attn_mask[l], md.data(), md.size() * sizeof(ggml_fp16_t));
                }
            }
            if (tp_mode)
            {
                if (!dc->tp_plan.valid())
                {
                    set_last_error("Gemma4 MoE model decode: cached graph has no tensor-parallel plan.");
                    dc->reset();
                    return 0;
                }
                dc->tp_plan.out_tensor = dc->hidden_out;
                dc->tp_plan.out_host = dc->folded ? logits_data : hidden_data;
                dc->tp_plan.out_bytes = static_cast<std::size_t>(dc->out_count) * sizeof(float);
                *tp_plan_out = &dc->tp_plan;
                return 1;
            }

            // MoE CPU offload replays with the same segment cuts the build pass
            // recorded; everything else still goes out as one submission.
            ggml_status st = GGML_STATUS_SUCCESS;
            if (!dc->host_moe.empty())
            {
                if (!host_moe_execute_segments(dc->graph, dc->host_moe, dc->host_moe_seg_end, kG4MoeDecodeKernel))
                    st = GGML_STATUS_FAILED;
            }
            else
            {
                st = tsg::graph_compute_profiled(g_backend, dc->graph, "gemma4 MoE model decode");
            }
            if (st != GGML_STATUS_SUCCESS)
            {
                if (dc->host_moe.empty())
                    set_last_error("Gemma4 MoE model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            void* reuse_out = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, reuse_out, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            return 1;
        }
        // Miss -> claim this request's slot to (re)build into (when persistable).
        G4MoEDecodeCache* g4moe = can_persist ? &g_moe_pool().claim(g4moe_sig, g4moe_kc0) : nullptr;

        // ctx holds only tensor metadata (no_alloc: data is bound externally). Non-
        // persist uses a pooled 32 MB block; persist uses a raw ctx kept alive in
        // g_g4moe for graph reuse + CUDA-graph capture (stable tensor addresses).
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to init persist ggml context.");
                return 0;
            }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Gemma4 MoE model decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        // Per-layer persist inputs (created in the build loop; null in legacy mode).
        std::vector<ggml_tensor*> layer_kv_index(num_layers, nullptr);

        // Tensor-parallel cut points: three per layer, one for each row-parallel
        // projection whose output is only this rank's share of the sum — the
        // attention output, the dense FFN down, and the MoE down (reduced after
        // the routing-weighted expert sum, which is linear, so summing partials
        // then weighting is the same as weighting then summing).
        std::vector<ggml_tensor*> tp_partial;
        std::vector<ggml_tensor*> tp_boundary;
        if (tp_mode)
        {
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 3);
            tp_boundary.reserve(static_cast<std::size_t>(num_layers) * 3);
        }

        // MoE CPU offload: filled per offloaded layer while the graph is built,
        // then turned into segment boundaries after ggml_build_forward_expand
        // has fixed the node order.
        std::vector<tsg::HostMoeSegment> host_moe;

        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        if (can_persist)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_tensor);
        }

        // Single shared rope_freqs tensor (same weight across all global layers).
        ggml_tensor* freq_factors_t = nullptr;
        void* freq_data = nullptr;
        int freq_len = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].freq_factors != nullptr && layers[l].freq_factors_len > 0)
            {
                freq_data = layers[l].freq_factors;
                freq_len = layers[l].freq_factors_len;
                break;
            }
        }
        if (freq_data != nullptr)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_len);

        struct MoeLayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffw_norm_1_w;
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_inp_scale_t;
            ggml_tensor* pre_ffw_norm_2_w;
            ggml_tensor* gate_up_exps_t;
            ggml_tensor* down_exps_t;
            ggml_tensor* down_exps_scale_t;
            ggml_tensor* post_ffw_norm_2_w;
            ggml_tensor* post_ffw_norm_w;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            ggml_tensor* attn_mask;
            std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<MoeLayerTensors> lt(num_layers);

        // --- create per-layer weight / cache tensors ---
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int cacheSize = d.cache_size;
            const int nExp = d.num_experts;
            const bool separate_qkv = d.separate_qkv != 0;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (separate_qkv)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
            }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            // MoE CPU offload: leave the routed-expert tensors null so the bind
            // pass below never asks for a device copy of them. That omission IS
            // the VRAM saving - the host reads the same bytes from the GGUF mmap.
            // TS_HOST_MOE_VERIFY's on-GPU reference chain needs the experts
            // resident, which cannot work under TP for an offloaded layer: the
            // descriptor points at the UNSHARDED stack there (the host computes
            // the layer once) while these would be the rank's Megatron slice.
            if (d.cpu_moe == 0 || (host_moe_verify_enabled() && !tp_mode))
            {
                t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
                t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            }
            else
            {
                t.gate_up_exps_t = nullptr;
                t.down_exps_t = nullptr;
            }
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cpy = nullptr;
            t.v_cpy = nullptr;
            t.attn_mask = nullptr;
        }

        // --- build the chained graph ---
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];

            const int hd = d.head_dim;
            const int nH = num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int cacheSize = d.cache_size;
            const bool isLocal = d.is_local != 0;
            const bool separate_qkv = d.separate_qkv != 0;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const int attendLen = isLocal ? std::min(totalSeqLen, d.sliding_window) : totalSeqLen;
            const std::int64_t ffDense = d.gu_ne1 / 2;
            const std::int64_t ffMoe = d.gue_ne1 / 2;
            const int rope_dims = d.rope_n_dims;
            ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

            // ===== Attention =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);

            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (separate_qkv)
            {
                q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qDim);
                k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.k_w, normed_2d), kDim);
                v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.v_w, normed_2d), kDim);
            }
            else
            {
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qDim + 2 * kDim);
                q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
                k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, hd, nH);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, hd, kvH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), t.k_norm_w);
            ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, hd, kvH);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, hd, kvH, 1);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, hd, kvH, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
            ggml_tensor* v_write = ggml_cont(ctx, v_perm);

            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (can_persist)
            {
                // KV write via set_rows: the write row is an I64 INPUT, so the graph
                // topology is identical token-to-token (capturable). Attention reads
                // a FIXED padded window [0, win) with an F16 mask INPUT zeroing valid
                // positions and -inf'ing the padding.
                const int win = pwindow[l];
                ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                ggml_set_input(kv_idx);
                layer_kv_index[l] = kv_idx;
                t.k_cpy = ggml_set_rows(ctx, t.k_cached_t, k_write, kv_idx);
                t.v_cpy = ggml_set_rows(ctx, t.v_cached_t, v_write, kv_idx);
                t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win, 1, 1, 1);
                ggml_set_input(t.attn_mask);
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, 0, win, kvType);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, 0, win, kvType);
            }
            else
            {
                const int cachePos = isLocal ? (position % cacheSize) : position;
                const int activeStart = swa_decode_window_start(isLocal, totalSeqLen, attendLen, cacheSize);
                const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
                const std::size_t kv_byte_offset = static_cast<std::size_t>(cachePos) * t.k_cached_t->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cached_t, hd, 1, kvH, t.k_cached_t->nb[1], t.k_cached_t->nb[2], kv_byte_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cached_t, hd, 1, kvH, t.v_cached_t->nb[1], t.v_cached_t->nb[2], kv_byte_offset);
                t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                t.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                if (flash_attn_requires_masked_padding(hd))
                {
                    t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                    fill_flash_attn_mask(t.attn_mask_data, attnKvLen, attendLen);
                }
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to build KV cache views.");
                if (can_persist) ggml_free(ctx);
                return 0;
            }

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, t.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
            ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_flat);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, o_mm, H);
            if (tp_mode) { tp_partial.push_back(o_mm); tp_boundary.push_back(o_flat); }
            ggml_tensor* post_attn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, o_flat, eps), t.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, post_attn_normed, hidden);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // ===== Dense shared FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.gu_w, ffn_normed_2d), 2 * ffDense);
            ggml_tensor* dense_gate = ggml_view_1d(ctx, gu_flat, ffDense, 0);
            ggml_tensor* dense_up = ggml_view_1d(ctx, gu_flat, ffDense, static_cast<std::size_t>(ffDense) * sizeof(float));
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
            ggml_tensor* dense_h_2d = ggml_reshape_2d(ctx, dense_h, ffDense, 1);
            ggml_tensor* dense_down_mm = ggml_mul_mat(ctx, t.down_w, dense_h_2d);
            ggml_tensor* dense_down = ggml_reshape_1d(ctx, dense_down_mm, H);
            if (tp_mode) { tp_partial.push_back(dense_down_mm); tp_boundary.push_back(dense_down); }
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);

            // ===== MoE router (in-graph) =====
            ggml_tensor* route_n = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps);
            route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr)
                route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);
            ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);
            ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, 1);
            ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, 1);
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
            w_2d = ggml_div(ctx, w_2d, w_sum);
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_r = ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1);
                ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_r, sel);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, 1));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, 1);

            // ===== MoE experts =====
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps), t.pre_ffw_norm_2_w);
            ggml_tensor* moe_out_1d = nullptr;
            if (d.cpu_moe != 0)
            {
                // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
                // Hand the host everything it needs to reproduce exactly what the
                // mul_mat_id chain below would have computed: the expert-input
                // norm and the router's own top-k ids and final weights (which
                // already fold in down_exps_scale, so the host stays identical to
                // the resident path). Contiguous copies rather than the raw views
                // keep each download a single flat memcpy and stop the allocator
                // recycling their storage before we read it.
                tsg::HostMoeSegment hm;
                hm.layer = l;
                hm.moe_in = ggml_cont(ctx, ggml_reshape_1d(ctx, moe_in, H));
                hm.sel_ids = ggml_cont(ctx, ggml_reshape_1d(ctx, sel, nUsed));
                hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, nUsed));
                ggml_set_output(hm.moe_in);
                ggml_set_output(hm.sel_ids);
                ggml_set_output(hm.weights);

                // The host writes here between segments. Flagged BOTH input and
                // output: ggml-alloc pre-allocates inputs but still frees them
                // after their last consumer, and this one is written from outside
                // the graph, so it has to stay pinned for the whole pass.
                moe_out_1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
                ggml_set_input(moe_out_1d);
                ggml_set_output(moe_out_1d);

                hm.moe_out = moe_out_1d;
                // Gemma 4 stacks gate and up into ONE tensor: up_data == nullptr
                // selects that layout in the host builder, which splits it the
                // same way this graph's two views do.
                hm.gate_data = d.gate_up_exps; hm.gate_type = d.gue_type;
                hm.gate_ne0 = d.gue_ne0;       hm.gate_ne1 = d.gue_ne1;  hm.gate_bytes = d.gue_bytes;
                hm.down_data = d.down_exps;    hm.down_type = d.de_type;
                hm.down_ne0 = d.de_ne0;        hm.down_ne1 = d.de_ne1;   hm.down_bytes = d.de_bytes;
                hm.activation = 2;             // gelu(gate) * up
                hm.num_experts = nExp;
                hm.n_used = nUsed;
                hm.n_ff = ffMoe;
                hm.seq_len = 1;
                hm.hidden = H;
                // Nothing downstream reduces an offloaded layer's MoE output
                // (the tp_partial push below belongs to the resident branch
                // only), so every rank takes the one host result verbatim.
                hm.tp_reduced = 0;

                if (host_moe_verify_enabled() && !tp_mode)
                {
                    ggml_tensor* vin = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
                    ggml_tensor* vgu = ggml_mul_mat_id(ctx, t.gate_up_exps_t, vin, sel);
                    ggml_tensor* vg = ggml_view_3d(ctx, vgu, ffMoe, vgu->ne[1], vgu->ne[2], vgu->nb[1], vgu->nb[2], 0);
                    ggml_tensor* vu = ggml_view_3d(ctx, vgu, ffMoe, vgu->ne[1], vgu->ne[2], vgu->nb[1], vgu->nb[2], static_cast<std::size_t>(ffMoe) * vgu->nb[0]);
                    ggml_tensor* vd = ggml_mul_mat_id(ctx, t.down_exps_t, ggml_geglu_split(ctx, vg, vu), sel);
                    ggml_tensor* vw = ggml_mul(ctx, vd, w_final);
                    ggml_tensor* vsum = ggml_view_2d(ctx, vw, H, 1, vw->nb[2], 0);
                    for (int u = 1; u < nUsed; ++u)
                    {
                        ggml_tensor* vv = ggml_view_2d(ctx, vw, H, 1, vw->nb[2], static_cast<std::size_t>(u) * vw->nb[1]);
                        vsum = ggml_add(ctx, vsum, vv);
                    }
                    hm.verify_gpu = ggml_cont(ctx, ggml_reshape_1d(ctx, vsum, H));
                    ggml_set_output(hm.verify_gpu);
                }

                host_moe.push_back(hm);
            }
            else
            {
                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
                ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);
                ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
                ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);

                ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
                for (int u = 1; u < nUsed; ++u)
                {
                    ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, view_u);
                }
                moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);
                if (tp_mode) { tp_partial.push_back(moe_out); tp_boundary.push_back(moe_out_1d); }
            }
            ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out_1d, eps), t.post_ffw_norm_2_w);
            mlp = ggml_add(ctx, mlp, moe_normed);

            // ===== Final residual + layer scale =====
            ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
            ggml_tensor* result = ggml_add(ctx, mlp_normed, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
            if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f)
                result = ggml_scale(ctx, result, d.layer_output_scale);

            hidden = result;
        }

        // Output: either the bare hidden state, or — when folding — the final
        // RMSNorm * output_norm, the lm_head projection and the tanh logit softcap,
        // so the 256K-vocab logits are part of the captured replay.
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out;
        ggml_tensor* out_cpy;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, H, 1);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_2d);   // [vocab, 1]
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_cpy = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_cpy);

        // KV writes first so they are ordered before the reads (mirrors the dense
        // model-wide decode).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // Expand layer by layer. ggml_build_forward_expand appends each root's
        // not-yet-emitted dependencies in topological order, so walking the
        // layers in order — KV writes first, then the layer's host-MoE boundary
        // if it has one — lays the nodes out as
        //   [... layer l attention ...][layer l norms + router][... layer l+1 ...]
        // which is exactly where the accelerator has to pause for the host. The
        // trailing expand of out_cpy then picks up every remaining node (the
        // dense FFN tail, residual adds, the final norm and the LM head).
        std::size_t next_host_moe = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (lt[l].k_cpy != nullptr)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
            if (next_host_moe < host_moe.size() && host_moe[next_host_moe].layer == l)
            {
                const tsg::HostMoeSegment& hm = host_moe[next_host_moe];
                ggml_build_forward_expand(graph, hm.moe_in);
                ggml_build_forward_expand(graph, hm.sel_ids);
                ggml_build_forward_expand(graph, hm.weights);
                if (hm.verify_gpu != nullptr)
                    ggml_build_forward_expand(graph, hm.verify_gpu);
                ++next_host_moe;
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        // Turn the recorded boundaries into node cut points (see
        // host_moe_build_segment_ends). It fails when the builder and the
        // expander disagree, which would silently feed the host stale values.
        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kG4MoeDecodeKernel))
        {
            if (can_persist) ggml_free(ctx);
            return 0;
        }

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                    {
                        return;
                    }
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int nExp = d.num_experts;
            const int cacheSize = d.cache_size;

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr)
            {
                bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
            }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr)
                bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr)
                bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);
            bind_or_mark(t.k_cached_t, d.k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(t.v_cached_t, d.v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            if (t.attn_mask != nullptr && !t.attn_mask_data.empty())
                bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        if (freq_factors_t != nullptr)
            bind_or_mark(freq_factors_t, freq_data, static_cast<std::size_t>(freq_len) * sizeof(float), true);
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);
        }

        // Non-persist (legacy / TS_GEMMA4_FD_PERSIST=0) keeps the peak-packed gallocr
        // (shared with the MoE verify): the bump allocator's footprint is the SUM of
        // every intermediate (~870 MB on the 26B-A4B), which on top of the ~16 GB
        // resident weights/KV would OOM; gallocr packs by tensor LIFETIME (peak).
        // Persist: stable tensor addresses, with Metal attention workspace reuse, so the
        // built graph + KV buffers keep fixed addresses for CUDA-graph capture; the
        // ctx/graph/buffer are kept alive in g_g4moe. The N=1 padded-window decode's
        // intermediate footprint is small (a few MB), unlike the verify's. Non-
        // persist keeps the peak-packed gallocr (the original behaviour).
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, freq_data, 0, static_cast<std::size_t>(freq_len) * sizeof(float));
        if (can_persist)
        {
            // Per-token inputs for the first build (the reuse path updates these).
            for (int l = 0; l < num_layers; l++)
            {
                if (layer_kv_index[l] != nullptr)
                    ggml_backend_tensor_set(layer_kv_index[l], &pwrite[l], 0, sizeof(std::int64_t));
                if (lt[l].attn_mask != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    ggml_backend_tensor_set(lt[l].attn_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
                }
            }
        }

        void* out_data = fold ? logits_data : hidden_data;

        if (!tp_mode)
        {
            // MoE CPU offload runs the graph in segments, pausing at each
            // offloaded layer for the host expert matmul. Everything else still
            // goes out as one graph submission.
            ggml_status status = GGML_STATUS_SUCCESS;
            if (!host_moe.empty())
            {
                if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kG4MoeDecodeKernel))
                    status = GGML_STATUS_FAILED;
            }
            else
            {
                status = tsg::compute_graph(g_backend, graph);
            }
            if (status != GGML_STATUS_SUCCESS)
            {
                if (host_moe.empty())
                    set_last_error("Gemma4 MoE model decode: graph execution failed.");
                if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
                return 0;
            }

            finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
            // Unconditional drain: `out_data` is the caller's host logits/hidden
            // buffer and on Metal async mode the download above is only QUEUED.
            // (It also has to happen before BufferHandle frees a per-call
            // fallback buffer that an in-flight blit is still reading.)
            host_read_barrier();
        }

        if (can_persist && g4moe != nullptr)
        {
            // Keep the ctx/graph/buffer + input handles alive for capture+replay,
            // in this request's own pool slot.
            g4moe->ctx = ctx;
            g4moe->buffer = persist_buf;
            g4moe->graph = graph;
            g4moe->hidden_in = hidden_t;
            g4moe->hidden_out = hidden_out;
            g4moe->pos_tensor = pos_tensor;
            g4moe->kv_index = layer_kv_index;
            g4moe->attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) g4moe->attn_mask[l] = lt[l].attn_mask;
            g4moe->layer_window = pwindow;
            // MoE CPU offload: the replay path re-runs these segments, so the
            // plan has to live in the cache slot. Without it a cached graph
            // would run end to end and read whatever stale bytes moe_out still
            // held from the previous token.
            g4moe->host_moe = host_moe;
            g4moe->host_moe_seg_end = host_moe_seg_end;
            g4moe->sig_disc = g4moe_sig;
            g4moe->sig_kcache0 = g4moe_kc0;
            g4moe->num_layers = num_layers;
            g4moe->hidden_size = H;
            g4moe->folded = fold;
            g4moe->out_count = out_count;
            g4moe->valid = true;
        }

        if (tp_mode)
        {
            if (g4moe == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: tensor-parallel build produced no cache slot.");
                return 0;
            }
            g4moe->tp_plan.clear();
            g4moe->tp_plan.graph = graph;
            g4moe->tp_plan.ar_tensor = tp_partial;
            // Offloaded layers pause this graph too; tp_plan_segments merges
            // their cuts into the same schedule as the AllReduce ones.
            g4moe->tp_plan.host_moe = host_moe;
            if (!tp_plan_segments(g4moe->tp_plan, tp_boundary))
            {
                g4moe->reset();
                return 0;
            }
            g4moe->tp_plan.out_tensor = hidden_out;
            g4moe->tp_plan.out_host = out_data;
            g4moe->tp_plan.out_bytes = static_cast<std::size_t>(out_count) * sizeof(float);
            *tp_plan_out = &g4moe->tp_plan;
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
        set_last_error("Unknown error in Gemma4 MoE model decode.");
        return 0;
    }
}

// Drop the persistent (CUDA-graph-captured) Gemma4 MoE decode graph. The captured
// graph pins ggml-cuda's compute-pool scratch addresses and the KV-cache device
// buffers; a prefill (which grows the pool) or a KV reset/grow can move those, so
// the C# caller drops the cache before any prefill and on ResetKVCache. The next
// decode rebuilds + re-captures against the current pool state. No-op when persist
// mode is off (the cache is never populated).
TSG_EXPORT void TSGgml_Gemma4MoEResetDecodeCache()
{
    // Every rank's pool: a KV reset invalidates the cached graphs on all GPUs and
    // the caller is not necessarily on the rank that built them.
    for (auto& pool : g_g4moe_pools)
        pool.reset_all();
}

namespace
{
    // Resources a tensor-parallel MoE verify graph borrows between "build" and
    // "execute". Unlike the decode graph this one is not persistent, so the
    // pooled context and any per-call buffer have to be parked until the driver
    // has run every rank's segments. Recycled on that rank's next build.
    struct G4MoEVerifyTpPending
    {
        PooledContextHandle context;
        BufferHandle buffer{ nullptr };
        std::vector<BufferHandle> ephemeral;
        tsg::TpRankPlan plan;

        void reset()
        {
            plan.clear();
            ephemeral.clear();
            buffer = BufferHandle(nullptr);
            context = PooledContextHandle();
        }
    };
    G4MoEVerifyTpPending g_g4moev_tp[tsg::TSG_MAX_DEVICES];
}

// Release every rank's parked tensor-parallel MoE verify graph.
TSG_EXPORT void TSGgml_Gemma4MoEReleaseVerifyTpGraphs()
{
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; ++r)
    {
        if (g_g4moev_tp[r].plan.graph == nullptr && g_g4moev_tp[r].context.value == nullptr)
            continue;
        tsg::ScopedRank rank(r);
        g_g4moev_tp[r].reset();
    }
}

// ============================================================================
// Gemma4 MoE MODEL-WIDE multi-token VERIFY: the whole MoE transformer over N
// tokens as ONE GGML graph. This is the MoE sibling of the dense
// TSGgml_Gemma4ModelVerify and the multi-token sibling of
// TSGgml_Gemma4MoEModelDecode — it is what makes MTP speculative decoding pay
// off on MoE Gemma 4 (e.g. gemma-4-26B-A4B): a K+1 verify batch runs as a single
// dispatch/sync instead of (K+1) fused single-token decodes or, far worse, the
// per-op TransformerBlock fallback (~390 ms/step that made spec net-negative).
//
// Attention is built exactly like the dense verify (manual masked attention:
// mul_mat + ggml_diag_mask_inf(attendLen-N) + soft_max, robust at head_dim 512
// and for SWA windows; wrap-aware circular KV write + windowed read), and the
// FFN is built exactly like the MoE decode (dense shared FFN + in-graph router +
// stacked-expert ggml_mul_mat_id), generalised from 1 to N tokens. Output is the
// per-row layer-stack hidden state [hidden_size, N] (pre output_norm); the C#
// caller owns output_norm + the LM head. Reuses TSGgmlGemma4MoELayerDesc unchanged
// (hidden/position per-desc are ignored; start_pos + num_tokens are shared params).
//
// Scope (enforced by the C# gate in Gemma4Model.NativeGemma4MoEModelVerify):
// all-MoE, non-shared (no KV donor), no PLE, F32/F16 KV cache, and (for global
// layers) start_pos + N <= cache_size. Returns 0 on anything it cannot handle so
// the caller falls back to the per-op verify.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4MoEModelVerify(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int start_pos, int num_tokens,
    // Multimodal bidirectional-span mask, nullable, length N (one byte per
    // CHUNK token, 1 = "soft" image/audio token), honoured at any start_pos;
    // mirrors the dense verify's is_except_arr (see gemma4_mm_mask.h).
    const unsigned char* mm_is_except,
    // Tensor parallelism — same contract as TSGgml_Gemma4MoEModelDecode.
    int tp_degree, void** tp_plan_out)
{
    tsg::PhaseTimer pt("gemma4 MoE model verify");
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Gemma4 MoE model verify: invalid arguments.");
            return 0;
        }

        // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
        if (tp_mode)
        {
            *tp_plan_out = nullptr;
            g_g4moev_tp[tsg::g_active_rank].reset();
        }
        // Cut points: the three row-parallel projections per layer (and per query
        // tile, when the tiled path runs — both ranks tile identically, so the
        // segment structure still matches).
        std::vector<ggml_tensor*> tp_partial;
        // MoE CPU offload: one entry per (offloaded layer, query tile). Pushed in
        // layer_tail below, which the tiled path calls once per tile — push order
        // therefore matches graph dependency order, which is what the segment cuts
        // require.
        std::vector<tsg::HostMoeSegment> host_moe;
        // Per-layer root of the MoE gating chain. Expanded on its own below so
        // the chain's nodes land contiguously in ggml-cuda's fusable order.
        std::vector<ggml_tensor*> routing_root(num_layers, nullptr);
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE model verify: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int N = num_tokens;
        if (N <= 1)
            return 0;
        const int H = hidden_size;
        const int totalSeqLen = start_pos + N;
        // The bidirectional-span mask is indexed by chunk position; both mask
        // builders below map a key to its chunk index at any start_pos
        // (gemma4_mm_mask.h), so an image chunk after a reused prefix stays here.
        const unsigned char* is_except = mm_is_except;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        // The whole layer (attention + dense FFN + experts) is processed in query
        // tiles of moe_attn_tile (see the per-layer loop / get_tile_mask), so the
        // [*, N] FFN/router/expert intermediates are tile-bounded rather than scaling
        // with N — this is what mirrors llama.cpp's n_ubatch and keeps the gallocr
        // peak (~the residual stream + K/V) under the ~1 GB headroom the 26B-A4B
        // leaves on a 16 GB card. (The old per-FFN TS_G4_MOE_FFN_TILE loop is
        // subsumed by the whole-layer tiling; the knob no longer applies.)

        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_shared != 0)
            {
                set_last_error("Gemma4 MoE model verify: KV-donor (shared) layers unsupported; use per-op path.");
                return 0;
            }
            // Global (full-attention) layers use a linear cache that must cover the
            // whole sequence (totalSeqLen <= cache_size), so global overflow always
            // bails. SWA (local) layers handle overflow (totalSeqLen > window) at any
            // start_pos: at start_pos==0 the fresh chunk is the whole history
            // (swaFresh); at start_pos>0 the kernel gathers the previous window from
            // the rolling cache and prepends it to the fresh chunk (swaPrev) — this is
            // what lets a long prompt be processed as bounded ubatches (the C# wrapper
            // sub-chunks; each native call keeps the gallocr peak small). (totalSeqLen
            // <= window is always exact: causal == windowed.)
            const bool isLocal = layers[l].is_local != 0;
            if (totalSeqLen > layers[l].cache_size && !isLocal)
            {
                set_last_error("Gemma4 MoE model verify: global cache too small for sequence; use per-op path.");
                return 0;
            }
        }

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Gemma4 MoE model verify: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

        ggml_tensor* freq_factors_t = nullptr;
        void* freq_data = nullptr;
        int freq_len = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].freq_factors != nullptr && layers[l].freq_factors_len > 0)
            {
                freq_data = layers[l].freq_factors;
                freq_len = layers[l].freq_factors_len;
                break;
            }
        }
        if (freq_data != nullptr)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_len);

        struct MoeLayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffw_norm_1_w;
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_inp_scale_t;
            ggml_tensor* pre_ffw_norm_2_w;
            ggml_tensor* gate_up_exps_t;
            ggml_tensor* down_exps_t;
            ggml_tensor* down_exps_scale_t;
            ggml_tensor* post_ffw_norm_2_w;
            ggml_tensor* post_ffw_norm_w;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;     // primary cache write
            ggml_tensor* k_cpy2; ggml_tensor* v_cpy2;   // wrapped tail (circular SWA)
            ggml_tensor* k_prev; ggml_tensor* v_prev;   // swaPrev: F32 copy of the prev window (read before the cache write)
        };
        std::vector<MoeLayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int cacheSize = d.cache_size;
            const int nExp = d.num_experts;
            const bool separate_qkv = d.separate_qkv != 0;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (separate_qkv)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
            }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            // MoE CPU offload: leave the routed-expert tensors null so the bind
            // pass below never asks for a device copy of them. Prefill is where the
            // whole expert set would otherwise land in VRAM (the decode graph alone
            // touches the same weights), so this is the binding that actually makes
            // --n-cpu-moe save memory.
            if (d.cpu_moe == 0)
            {
                t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
                t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            }
            else
            {
                t.gate_up_exps_t = nullptr;
                t.down_exps_t = nullptr;
            }
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cpy = nullptr; t.v_cpy = nullptr; t.k_cpy2 = nullptr; t.v_cpy2 = nullptr;
            t.k_prev = nullptr; t.v_prev = nullptr;
        }

        // Causal masks for ggml_flash_attn_ext. Attention here mirrors the MoE
        // decode and the prefill paths (flash_attn_ext, not a manual mul_mat +
        // soft_max chain): the manual chain is numerically unreliable for
        // multi-token Q at head_dim 256/512 on Metal (silent F16 accumulator
        // overflow plus barely-exercised soft_max/mul_mat kernels), while
        // flash_attn_ext is the proven path on Metal AND CUDA and is faster.
        // One mask per distinct (kvLen, validLen, window) — at most three: a
        // circular-windowed SWA mask, a swaFresh sliding-window mask (window>0,
        // attends the fresh chunk), and a (flash-padded) global mask — shared
        // across layers. mask[qi][ki] = 0 iff ki < validLen (real, not padding)
        // AND ki <= (validLen-N)+qi (causal) AND (window==0 || ki > threshold-window)
        // (sliding-window low bound). For the circular-window read (window==0) the
        // SWA low-end is already below the cache view so no extra windowing is
        // needed; the swaFresh path (window=W) attends the full fresh chunk so it
        // applies the window low bound explicitly (same as the dense verify).
        struct VerifyMask { int kvLen; int validLen; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<VerifyMask> mask_cache;
        std::vector<std::vector<ggml_fp16_t>> mask_data_store;
        auto get_causal_mask = [&](int kvLen, int validLen, int window) -> ggml_tensor* {
            for (auto& m : mask_cache)
                if (m.kvLen == kvLen && m.validLen == validLen && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvLen, N);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kvLen) * N);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            // Causal band per row (filled analytically — a per-element host loop
            // blocks the GPU at long prefill), plus the chunk's soft keys ahead of
            // a soft query (no window low bound on that clause). The chunk's keys
            // are the last N real keys [validLen-N, validLen).
            for (int qi = 0; qi < N; qi++)
                tsg_gemma4_mask::fill_relative_row(&data[static_cast<std::size_t>(qi) * kvLen],
                    kvLen, qi, N, validLen, window, is_except, zero_val, neg_inf);
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            mask_cache.push_back({kvLen, validLen, window, mt, idx});
            return mt;
        };

        // Tiled attention (mirrors llama.cpp's n_ubatch and the dense verify's
        // TS_G4_SWA_TILE). A single full-N flash builds an O(N^2) F16 mask plus
        // O(N) q_rope / flash-output / score transients that, on the near-full
        // 26B-A4B GPU (~1 GB free with 14 GB experts resident), spill into WDDM
        // shared memory at long prefill (the 4k/8k cliff). Splitting the N queries
        // into tiles bounds the mask to [kLen, qLen<=tile]; for LOCAL (SWA) layers
        // each tile reads only its sliding-window key slice (less compute too),
        // while GLOBAL (full-attention) layers read [0, qe) — bounded mask, same
        // O(N^2) compute as llama. Both default-on; TS_G4_MOE_ATTN_TILED=0 reverts.
        static const bool moe_attn_tiled = []{ const char* e = std::getenv("TS_G4_MOE_ATTN_TILED"); return e == nullptr || e[0] != '0'; }();
        static const int moe_attn_tile = []{ const char* e = std::getenv("TS_G4_MOE_ATTN_TILE"); int v = e ? std::atoi(e) : 0; return (v >= 256) ? v : 1024; }();
        // Unified causal/sliding-window tile mask. gQ is the ABSOLUTE query
        // position (start_pos + tile-local index); ki indexes the key slice that
        // starts at logical position kStart. window<=0 means no sliding-window low
        // bound (global / full causal). Keys form a contiguous unmasked band
        // [lo, hi] per query row — filled analytically (a per-element host loop
        // blocks the GPU at long prefill).
        struct MoeTileMask { int kLen; int qLen; int qStartAbs; int kStart; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<MoeTileMask> tile_mask_cache;
        auto get_tile_mask = [&](int kLen, int qLen, int qStartAbs, int kStart, int window) -> ggml_tensor* {
            for (auto& m : tile_mask_cache)
                if (m.kLen == kLen && m.qLen == qLen && m.qStartAbs == qStartAbs && m.kStart == kStart && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kLen, qLen);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kLen) * qLen);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            // Key ki sits at logical position kStart + ki; a soft query also
            // keeps the chunk's soft keys ahead of it, with no window low bound on
            // that clause (the chunk starts at start_pos).
            for (int qi = 0; qi < qLen; qi++)
                tsg_gemma4_mask::fill_absolute_row(&data[static_cast<std::size_t>(qi) * kLen],
                    kLen, qStartAbs + qi, kStart, window, start_pos, N, is_except, zero_val, neg_inf);
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            tile_mask_cache.push_back({kLen, qLen, qStartAbs, kStart, window, mt, idx});
            return mt;
        };

        ggml_tensor* hidden = current;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];

            const int hd = d.head_dim;
            const int nH = num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int cacheSize = d.cache_size;
            const bool isLocal = d.is_local != 0;
            const bool separate_qkv = d.separate_qkv != 0;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gu_ne1 / 2;
            const std::int64_t ffMoe = d.gue_ne1 / 2;
            const int rope_dims = d.rope_n_dims;
            ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

            // ===== Attention (multi-token, flash) =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);  // [H, N]

            // Tiling decision (see get_tile_mask). swaWindow = local SWA prefill that
            // overflows the circular window: at start_pos==0 attend the FRESH chunk
            // (swaFresh); at start_pos>0 gather the previous window from the rolling
            // cache and prepend it to the fresh chunk (swaPrev). GLOBAL (full-attention)
            // layers always tile when N is large; their Q/attn tensors are [qDim=8192,N]
            // (16 heads x 512) and dominate the spill. When separate_qkv, Q is projected
            // from `normed` per tile so the full-N q_rope is never materialised (tileQ);
            // a fused QKV weight keeps the full q_rope.
            const bool swaFresh = isLocal && start_pos == 0 && totalSeqLen > cacheSize;
            const bool swaPrev  = isLocal && start_pos != 0 && totalSeqLen > cacheSize;
            // Tile whenever it bounds the working set: swaPrev ALWAYS (its extended K/V
            // + the M-token layer tail otherwise peak higher than the start-of-prompt
            // chunk), otherwise once there are >= 3 query tiles.
            // Bidi multimodal spans need forward attention past a query tile's
            // key slice — keep the full-N attention there (mirrors the dense
            // verify's swa_tiled fallback), including swaPrev: a media chunk after a
            // reused prefix attends [prev window ++ chunk] in one flash call.
            const bool moe_use_tiled = moe_attn_tiled && is_except == nullptr
                && (swaPrev || (N > 2 * moe_attn_tile && (swaFresh || !isLocal)));
            const bool tileQ = moe_use_tiled && separate_qkv;

            ggml_tensor* q_lin = nullptr; ggml_tensor* k_lin; ggml_tensor* v_lin;
            if (separate_qkv)
            {
                if (!tileQ) q_lin = ggml_mul_mat(ctx, t.qkv_w, normed);  // Q projected per-tile when tileQ
                k_lin = ggml_mul_mat(ctx, t.k_w, normed);
                v_lin = ggml_mul_mat(ctx, t.v_w, normed);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);  // [qDim+2kDim, N]
                q_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, qDim, N, qkv->nb[1], 0));
                k_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, N, qkv->nb[1], static_cast<std::size_t>(qDim) * sizeof(float)));
                v_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, N, qkv->nb[1], static_cast<std::size_t>(qDim + kDim) * sizeof(float)));
            }

            // K/V projection + norm + RoPE (always full N — needed for the cache write
            // and as the attention key/value source for every query tile).
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_lin, hd, kvH, N);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_lin, hd, kvH, N);
            k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), t.k_norm_w);
            v_3d = ggml_rms_norm(ctx, v_3d, eps);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);  // [hd, kvH, N]

            // Full-N Q only when NOT projecting Q per tile (small N, or fused QKV).
            ggml_tensor* q_rope = nullptr;
            if (!tileQ)
            {
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_lin, hd, nH, N);
                q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), t.q_norm_w);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);  // [hd, nH, N]
            }

            // Write N new K/V (wrap-aware circular write for SWA; linear for global).
            ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));  // [hd, N, kvH]
            ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));     // [hd, N, kvH]

            // swaPrev: gather the previous window [start_pos-prevCount, start_pos) from
            // the rolling circular cache into a MATERIALISED F32 copy. The copy is built
            // into the graph (and ggml_build_forward_expand'd) AHEAD of the cache write
            // below, so on the single execution stream it reads the OLD cache contents
            // before this chunk overwrites them. prevCount = min(W, start_pos).
            const int prevCount = swaPrev ? std::min(cacheSize, start_pos) : 0;
            const int swaBase = start_pos - prevCount;   // logical position of the prev-window start
            if (swaPrev)
            {
                ggml_tensor* kpv = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, swaBase, prevCount, kvType, /*fattn_query_rows=*/0);
                ggml_tensor* vpv = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, swaBase, prevCount, kvType, /*fattn_query_rows=*/0);
                if (kpv == nullptr || vpv == nullptr)
                {
                    set_last_error("Gemma4 MoE model verify: failed to view prev SWA window.");
                    return 0;
                }
                ggml_tensor* kpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, prevCount, kvH);
                ggml_tensor* vpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, prevCount, kvH);
                t.k_prev = ggml_cpy(ctx, kpv, kpf);   // F16/F32 view -> F32 copy (materialised before the write)
                t.v_prev = ggml_cpy(ctx, vpv, vpf);
            }

            // SWA prefill overflowing the circular window: only the LAST W positions are
            // written to the circular cache (for subsequent decode) — writing all N
            // would wrap the W buffer multiple times and corrupt it. Mirrors the dense
            // verify (TSGgml_Gemma4ModelVerify). Applies at any start_pos (swaFresh/swaPrev).
            const int writeOff = (isLocal && N > cacheSize) ? (N - cacheSize) : 0;
            const int writeLen = N - writeOff;
            const int writeStart = start_pos + writeOff;
            const int cacheBase = isLocal ? (writeStart % cacheSize) : writeStart;
            const int n1 = (isLocal && cacheBase + writeLen > cacheSize) ? (cacheSize - cacheBase) : writeLen;
            auto writePart = [&](ggml_tensor* cache, ggml_tensor* src, int srcOff, int dstSlot, int cnt) -> ggml_tensor* {
                ggml_tensor* s = ggml_view_3d(ctx, src, hd, cnt, kvH, src->nb[1], src->nb[2], static_cast<std::size_t>(srcOff) * src->nb[1]);
                ggml_tensor* dd = ggml_view_3d(ctx, cache, hd, cnt, kvH, cache->nb[1], cache->nb[2], static_cast<std::size_t>(dstSlot) * cache->nb[1]);
                return ggml_cpy(ctx, s, dd);
            };
            t.k_cpy = writePart(t.k_cached_t, k_write, writeOff, cacheBase, n1);
            t.v_cpy = writePart(t.v_cached_t, v_write, writeOff, cacheBase, n1);
            if (n1 < writeLen)
            {
                t.k_cpy2 = writePart(t.k_cached_t, k_write, writeOff + n1, 0, writeLen - n1);
                t.v_cpy2 = writePart(t.v_cached_t, v_write, writeOff + n1, 0, writeLen - n1);
            }

            int attendLen;
            int attnKvLen;
            int maskWindow;
            int keyBase;                             // logical position of k_full index 0
            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (swaFresh)
            {
                // Attend the FRESH full chunk (all N positions, post norm/RoPE) with a
                // sliding-window mask. ki == logical position (k_write is in chrono
                // order [0, N)). Local layers (head_dim 256) need no flash KV padding.
                attendLen = N;
                attnKvLen = N;
                maskWindow = cacheSize;              // sliding window W
                keyBase = 0;
                // flash_attn_ext needs K/V in the cache dtype; convert the fresh F32
                // chunk when the cache is F16 (no-op when F32).
                if (kvType == GGML_TYPE_F32)
                {
                    k_full = k_write;
                    v_full = v_write;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, N, kvH);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, N, kvH);
                    k_full = ggml_cpy(ctx, k_write, kf);
                    v_full = ggml_cpy(ctx, v_write, vf);
                }
            }
            else if (swaPrev)
            {
                // Extended K/V = [prev window (prevCount) ++ fresh chunk (N)], covering
                // logical [swaBase, start_pos+N). Attended with a sliding-window mask;
                // keyBase = swaBase so the mask maps a key's index to its logical pos.
                const int bufLen = prevCount + N;
                attendLen = bufLen;
                attnKvLen = bufLen;
                maskWindow = cacheSize;
                keyBase = swaBase;
                ggml_tensor* kext = ggml_concat(ctx, t.k_prev, k_write, 1);   // [hd, prevCount+N, kvH] F32
                ggml_tensor* vext = ggml_concat(ctx, t.v_prev, v_write, 1);
                if (kvType == GGML_TYPE_F32)
                {
                    k_full = kext;
                    v_full = vext;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, bufLen, kvH);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, bufLen, kvH);
                    k_full = ggml_cpy(ctx, kext, kf);
                    v_full = ggml_cpy(ctx, vext, vf);
                }
            }
            else
            {
                attendLen = isLocal ? std::min(totalSeqLen, cacheSize) : totalSeqLen;
                const int activeStart = isLocal ? ((totalSeqLen - attendLen) % cacheSize) : 0;
                // Flash attention reads a (possibly padded) KV window; the padding slots
                // beyond attendLen are masked out. Padding only ever applies to global
                // (linear, activeStart==0) layers — local SWA layers (head_dim 256) need
                // no padding, so the wrap-around window is never combined with padding.
                attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
                maskWindow = 0;                       // window already enforced by the cache view
                keyBase = 0;
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType, N);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType, N);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Gemma4 MoE model verify: failed to build KV cache views.");
                return 0;
            }

            // Post-attention layer tail (dense shared FFN + in-graph-routed experts +
            // final residual/scale) over a residual stream of M tokens. Factored out
            // so the query-tiled path runs it PER TILE (M = qLen) — bounding every
            // [*, M] FFN/router/MoE intermediate — while the small-N path runs it once
            // (M = N). The ops are token-independent, so per-tile output is byte-
            // identical to the full-N output. This is what bounds the gallocr peak to
            // ~the residual stream + K/V (the whole-layer [*, N] intermediates summed
            // to ~2.4 GB at N=8192 and spilled into WDDM shared memory on the 26B).
            auto layer_tail = [&](ggml_tensor* residual1, int M) -> ggml_tensor* {
                // dense shared FFN (fused GeGLU = gelu(gate)*up on the [2*ffDense, M] tensor)
                ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);                 // [2*ffDense, M]
                ggml_tensor* dense_h = ggml_geglu(ctx, gu);                             // [ffDense, M]
                ggml_tensor* dense_down = ggml_mul_mat(ctx, t.down_w, dense_h);         // [H, M]
                if (tp_mode) tp_partial.push_back(dense_down);
                ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);
                // MoE router (in-graph)
                ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);              // [H, M]
                route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
                if (t.gate_inp_scale_t != nullptr)
                    route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);  // [nExp, M]
                // Emitted in ggml-cuda's fusable order (see build_topk_moe_routing);
                // the caller expands `routing_root` per layer so the chain lands
                // contiguously and the fused top-k-MoE kernel matches it.
                tsg::MoeTopKRouting routing =
                    tsg::build_topk_moe_routing(ctx, router_logits, nExp, nUsed, M, /*norm_topk=*/true);
                ggml_tensor* sel = routing.ids;
                ggml_tensor* probs_r = routing.probs_3d;
                ggml_tensor* w_2d = routing.weights_2d;
                ggml_tensor* w_final = routing.weights_3d;
                if (t.down_exps_scale_t != nullptr)
                {
                    // A per-expert output scale gathered off probs_r gives the
                    // chain an outside reader, so this branch is deliberately
                    // unfusable; correctness first.
                    ggml_tensor* scale_b = ggml_repeat(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), probs_r);  // [1, nExp, M]
                    ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_b, sel);          // [1, nUsed, M]
                    w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, M));
                    w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, M);
                }
                routing_root[l] = w_final;
                // MoE experts (M is already tile-bounded — no inner token tiling needed)
                ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w);  // [H, M]
                ggml_tensor* moe_out = nullptr;
                if (d.cpu_moe != 0)
                {
                    // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
                    // Prefill hands the host all M tokens of this tile at once, so
                    // the offloaded side is a genuine GEMM over the tile rather than
                    // M matvecs — the same shape llama.cpp's CPU buffer-type override
                    // computes. Everything else (attention, dense FFN, the router)
                    // stays in the one fused graph.
                    tsg::HostMoeSegment hm;
                    hm.layer = l;
                    hm.moe_in = ggml_cont(ctx, moe_in);
                    // sel is a VIEW of the argsort (see build_topk_moe_routing), so it
                    // must be made contiguous BEFORE reshaping - ggml_reshape asserts
                    // contiguity and would abort every offloaded layer.
                    hm.sel_ids = ggml_reshape_1d(ctx, ggml_cont(ctx, sel), (std::int64_t) nUsed * M);
                    hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, (std::int64_t) nUsed * M));
                    ggml_set_output(hm.moe_in);
                    ggml_set_output(hm.sel_ids);
                    ggml_set_output(hm.weights);

                    // Written by the host between segments; flagged BOTH input and
                    // output so ggml-alloc pre-allocates it and never recycles the
                    // block behind our back.
                    moe_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, M);
                    ggml_set_input(moe_out);
                    ggml_set_output(moe_out);

                    hm.moe_out = moe_out;
                    hm.gate_data = d.gate_up_exps; hm.gate_type = d.gue_type;
                    hm.gate_ne0 = d.gue_ne0;       hm.gate_ne1 = d.gue_ne1;  hm.gate_bytes = d.gue_bytes;
                    hm.down_data = d.down_exps;    hm.down_type = d.de_type;
                    hm.down_ne0 = d.de_ne0;        hm.down_ne1 = d.de_ne1;   hm.down_bytes = d.de_bytes;
                    hm.activation = 2;             // gelu(gate) * up
                    hm.num_experts = nExp;
                    hm.n_used = nUsed;
                    hm.n_ff = (int) ffMoe;
                    hm.seq_len = M;
                    hm.hidden = H;
                    // Nothing downstream reduces it (see the tp_partial guard
                    // below), so every rank takes the one host result verbatim.
                    hm.tp_reduced = 0;
                    host_moe.push_back(hm);
                }
                else
                {
                    ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, M);
                    ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);   // [2*ffMoe, nUsed, M]
                    ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                    ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
                    ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);                  // [ffMoe, nUsed, M]
                    ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);       // [H, nUsed, M]
                    ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                        // [H, nUsed, M]
                    moe_out = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2], 0);
                    for (int u = 1; u < nUsed; ++u)
                    {
                        ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                        moe_out = ggml_add(ctx, moe_out, view_u);
                    }
                }
                // An offloaded layer's moe_out is an INPUT the host writes with
                // the complete (unsharded) result, so there is nothing to reduce
                // — and it is not a graph node the planner could cut on anyway.
                if (tp_mode && d.cpu_moe == 0) tp_partial.push_back(moe_out);
                ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w);
                mlp = ggml_add(ctx, mlp, moe_normed);
                // final residual + layer scale
                ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
                ggml_tensor* result = ggml_add(ctx, mlp_normed, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
                if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f)
                    result = ggml_scale(ctx, result, d.layer_output_scale);
                return result;
            };

            // Attention (Gemma scale = 1.0) fused with the layer tail. The tiled path
            // processes the WHOLE layer (attention + FFN + experts) in query tiles so
            // only the residual stream + K/V are full-N; everything else is tile-
            // bounded. The full-N path (small N / in-window-circular reads) keeps the
            // single flash + one layer_tail call.
            ggml_tensor* result;
            if (moe_use_tiled)
            {
                ggml_tensor* result_acc = nullptr;
                // Fused QKV (no separate Q weight): Q can't be re-projected from
                // `normed` per tile, so keep a full-N q_rope and view it per tile.
                ggml_tensor* q_t_full = (!tileQ) ? ggml_permute(ctx, q_rope, 0, 2, 1, 3) : nullptr;  // [hd, N, nH]
                for (int qs = 0; qs < N; qs += moe_attn_tile)
                {
                    const int qe = (qs + moe_attn_tile < N) ? (qs + moe_attn_tile) : N;
                    const int qLen = qe - qs;

                    // Per-tile Q (project from the normed-column slice) or a view of
                    // the full q_t when QKV is fused.
                    ggml_tensor* q_tile;
                    if (tileQ)
                    {
                        ggml_tensor* normed_tile = ggml_view_2d(ctx, normed, H, qLen,
                            normed->nb[1], static_cast<std::size_t>(qs) * normed->nb[1]);
                        ggml_tensor* q_lin_t = ggml_mul_mat(ctx, t.qkv_w, normed_tile);     // [qDim, qLen]
                        ggml_tensor* q_3d_t = ggml_reshape_3d(ctx, q_lin_t, hd, nH, qLen);
                        q_3d_t = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d_t, eps), t.q_norm_w);
                        ggml_tensor* pos_t = ggml_view_1d(ctx, pos_tensor, qLen,
                            static_cast<std::size_t>(qs) * sizeof(std::int32_t));
                        ggml_tensor* q_rope_t = ggml_rope_ext(ctx, q_3d_t, pos_t, rope_ff,
                            rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);               // [hd, nH, qLen]
                        q_tile = ggml_permute(ctx, q_rope_t, 0, 2, 1, 3);                  // [hd, qLen, nH]
                    }
                    else
                    {
                        q_tile = ggml_view_3d(ctx, q_t_full, hd, qLen, nH,
                            q_t_full->nb[1], q_t_full->nb[2], static_cast<std::size_t>(qs) * q_t_full->nb[1]);
                    }

                    int ks, kLen, window, kStartLogical;
                    if (maskWindow > 0)
                    {
                        // Windowed (swaFresh keyBase==0, or swaPrev keyBase==swaBase).
                        // k_full covers logical [keyBase, ...]; bufQ = index of this
                        // tile's first query's own key. Attend its window slice; the low
                        // bound is enforced by the mask.
                        const int bufQ = (start_pos + qs) - keyBase;
                        ks = (bufQ > maskWindow) ? (bufQ - maskWindow) : 0;
                        kLen = (bufQ + qLen) - ks;
                        if (kLen > attnKvLen) kLen = attnKvLen;
                        window = maskWindow;            // sliding window W
                        kStartLogical = keyBase + ks;   // logical position of the slice start
                    }
                    else
                    {
                        // Global linear cache: attend [0, start_pos+qe). Pad kLen for the
                        // head_dim-512 flash kernel (kvLen % 256 == 0).
                        ks = 0;
                        kLen = flash_attn_kv_length(start_pos + qe, cacheSize, hd);
                        if (kLen > attnKvLen) kLen = attnKvLen;
                        window = 0;                     // pure causal
                        kStartLogical = 0;
                    }
                    ggml_tensor* k_tile = ggml_view_3d(ctx, k_full, hd, kLen, kvH,
                        k_full->nb[1], k_full->nb[2], static_cast<std::size_t>(ks) * k_full->nb[1]);
                    ggml_tensor* v_tile = ggml_view_3d(ctx, v_full, hd, kLen, kvH,
                        v_full->nb[1], v_full->nb[2], static_cast<std::size_t>(ks) * v_full->nb[1]);
                    ggml_tensor* m_tile = get_tile_mask(kLen, qLen, start_pos + qs, kStartLogical, window);
                    ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_tile, k_tile, v_tile, m_tile, 1.0f, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                    if (qs == 0 && !backend_supports_op(fa))
                    {
                        set_last_error("Gemma4 MoE model verify: tiled flash attention unsupported for this shape; use per-op path.");
                        return 0;
                    }
                    // O projection (per tile) → post-attn norm → residual → layer tail.
                    ggml_tensor* fa_flat = ggml_reshape_2d(ctx, fa, qDim, qLen);
                    ggml_tensor* o_tile = ggml_mul_mat(ctx, t.o_w, fa_flat);               // [H, qLen]
                    if (tp_mode) tp_partial.push_back(o_tile);
                    ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_tile, eps), t.post_attn_norm_w);
                    ggml_tensor* hidden_tile = ggml_view_2d(ctx, hidden, H, qLen,
                        hidden->nb[1], static_cast<std::size_t>(qs) * hidden->nb[1]);
                    ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden_tile);        // [H, qLen]
                    ggml_tensor* result_tile = layer_tail(residual1, qLen);                // [H, qLen]
                    result_acc = (result_acc == nullptr) ? result_tile : ggml_concat(ctx, result_acc, result_tile, 1);  // [H, qe]
                }
                result = result_acc;                                                      // [H, N]
            }
            else
            {
                ggml_tensor* q_t = ggml_permute(ctx, q_rope, 0, 2, 1, 3);                  // [hd, N, nH]
                // Windowed reads (swaFresh/swaPrev) use the keyBase-aware tile mask over
                // the whole chunk (handles keyBase != 0 for swaPrev); global / in-window
                // circular reads use the position-relative causal mask.
                ggml_tensor* fa_mask = (maskWindow > 0)
                    ? get_tile_mask(attnKvLen, N, start_pos, keyBase, maskWindow)
                    : get_causal_mask(attnKvLen, attendLen, maskWindow);
                ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_t, k_full, v_full, fa_mask, 1.0f, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
                if (!backend_supports_op(attn_out))
                {
                    // No supported flash kernel for this shape (e.g. an exotic head_dim):
                    // fall back to the per-op verify rather than the numerically-fragile
                    // manual chain.
                    set_last_error("Gemma4 MoE model verify: flash attention unsupported for this shape; use per-op path.");
                    return 0;
                }
                // flash_attn_ext returns [hd, nH, N, 1] — each column holds all heads
                // contiguously, exactly what the O projection wants.
                ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, N);
                ggml_tensor* o_out = ggml_mul_mat(ctx, t.o_w, attn_flat);                  // [H, N]
                if (tp_mode) tp_partial.push_back(o_out);
                ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), t.post_attn_norm_w);
                ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden);                // [H, N]
                result = layer_tail(residual1, N);                                         // [H, N]
            }

            hidden = result;
        }

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_tensor* out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_cpy);

        // Each query tile runs the whole layer (per-tile Q proj/norm/rope + flash +
        // O proj + post-attn + residual + dense FFN + router + experts + concat) ≈ 56
        // nodes; budget for that plus the per-layer full-N K/V projection + the
        // optional swaPrev gather/concat/convert (~128 fixed headroom).
        const int attnTilesPerLayer = moe_attn_tiled ? ((N + moe_attn_tile - 1) / moe_attn_tile) : 1;
        const std::size_t graph_size = static_cast<std::size_t>(num_layers)
            * (128 + static_cast<std::size_t>(attnTilesPerLayer) * 56) + 512;
        pt.mark("build");
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // swaPrev: add the prev-window F32 copies FIRST so they are ordered ahead of
        // the cache writes below — on the single execution stream they then read the
        // old (previous-chunk) cache contents before this chunk overwrites them.
        for (int l = 0; l < num_layers; l++)
        {
            if (lt[l].k_prev != nullptr) ggml_build_forward_expand(graph, lt[l].k_prev);
            if (lt[l].v_prev != nullptr) ggml_build_forward_expand(graph, lt[l].v_prev);
        }
        // Expand layer by layer, KV writes then that layer's host-MoE boundaries.
        //
        // The interleaving is load-bearing for MoE CPU offload. A boundary tensor is
        // a `cont` nothing consumes, so it is only emitted when explicitly expanded;
        // expanding every layer's KV write first would drag the WHOLE model's
        // forward chain into the node list ahead of layer 0's cut, and every
        // offloaded layer would then read a moe_out the host had not written yet.
        // Per layer, the KV write still precedes that layer's attention read (which
        // arrives with the boundary expansion or with out_cpy), which is the
        // ordering the cache aliasing requires.
        std::size_t next_host_moe = 0;
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, lt[l].k_cpy);
            ggml_build_forward_expand(graph, lt[l].v_cpy);
            if (lt[l].k_cpy2 != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy2);
            if (lt[l].v_cpy2 != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy2);
            // The gating chain as one unit: expanding it here emits
            // SOFT_MAX -> RESHAPE -> ARGSORT -> VIEW -> GET_ROWS -> RESHAPE ->
            // SUM_ROWS -> CLAMP -> DIV -> RESHAPE back to back, which is what
            // ggml-cuda's fused top-k-MoE kernel matches. Left to the expert
            // matmuls to pull in, the ids half and the weights half would be
            // emitted at different points and nothing would fuse.
            if (routing_root[l] != nullptr) ggml_build_forward_expand(graph, routing_root[l]);
            // One entry per query tile for this layer, in the order layer_tail
            // pushed them.
            while (next_host_moe < host_moe.size() && host_moe[next_host_moe].layer == l)
            {
                const tsg::HostMoeSegment& hm = host_moe[next_host_moe];
                ggml_build_forward_expand(graph, hm.moe_in);
                ggml_build_forward_expand(graph, hm.sel_ids);
                ggml_build_forward_expand(graph, hm.weights);
                ++next_host_moe;
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kG4MoeVerifyKernel))
            return 0;

        pt.mark("expand");
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int nExp = d.num_experts;
            const int cacheSize = d.cache_size;

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr)
            {
                bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
            }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr)
                bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr)
                bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);
            bind_or_mark(t.k_cached_t, d.k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(t.v_cached_t, d.v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (freq_factors_t != nullptr)
            bind_or_mark(freq_factors_t, freq_data, static_cast<std::size_t>(freq_len) * sizeof(float), true);

        // Graph-aware allocation: gallocr packs the N-token intermediates by tensor
        // LIFETIME (peak, not sum), unlike the linear alloc_ctx_tensors_reuse bump
        // allocator the single-token decode uses. For a K+1 verify over 30 layers
        // the linear sum is hundreds of MB — on the 26B-A4B (model already ~13 GB
        // resident) that exhausts VRAM and starves the draft/weight caches (the
        // draft step then thrashes). gallocr's peak is ~10-20x smaller. The
        // pre-bound weights/KV caches above already own buffers and are skipped.
        // Persistent gallocr (reused across verify calls, grown on demand) instead
        // of a fresh ggml_gallocr_new()/_free() each step. The MoE verify graph's
        // intermediate buffer is ~400 MB; allocating and freeing that on every
        // verify churned Metal's shared (vm_allocate) device memory and fragmented
        // it over hundreds of steps until a contiguous allocation failed
        // (kIOGPUCommandBufferCallbackErrorOutOfMemory). Reusing one gallocr keeps
        // the buffer resident and stable, which removes the fragmentation source
        // (and the per-call ~20 ms allocation).
        pt.mark("bind");
        if (!alloc_graph_reuse_gallocr(graph))
        {
            set_last_error("Gemma4 MoE model verify: graph allocation failed.");
            return 0;
        }

        host_read_barrier();

        pt.mark("alloc");
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));

        std::vector<std::int32_t> pos_vals(N);
        for (int i = 0; i < N; i++) pos_vals[i] = start_pos + i;
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, freq_data, 0, static_cast<std::size_t>(freq_len) * sizeof(float));

        for (auto& m : mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));
        for (auto& m : tile_mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));

        if (tp_mode)
        {
            auto& pending = g_g4moev_tp[tsg::g_active_rank];
            pending.plan.clear();
            pending.plan.graph = graph;
            pending.plan.ar_tensor = tp_partial;
            // Offloaded layers pause this graph too; tp_plan_segments merges
            // their cuts into the same schedule as the AllReduce ones.
            pending.plan.host_moe = host_moe;
            if (!tp_plan_segments(pending.plan, tp_partial))
            {
                pending.reset();
                return 0;
            }
            // Every rank ends with the same replicated hidden state; they share one
            // host buffer, so only rank 0 copies it back.
            if (tsg::g_active_rank == 0)
            {
                pending.plan.out_tensor = hidden_out;
                pending.plan.out_host = hidden_data;
                pending.plan.out_bytes = static_cast<std::size_t>(H) * N * sizeof(float);
            }
            pending.context = std::move(context);
            pending.ephemeral = std::move(ephemeral_bufs);
            *tp_plan_out = &pending.plan;
            clear_last_error();
            return 1;
        }

        // MoE CPU offload runs the graph in segments, pausing at each offloaded
        // layer (and query tile) for the host expert GEMM over its M tokens.
        ggml_status status = GGML_STATUS_SUCCESS;
        if (!host_moe.empty())
        {
            if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kG4MoeVerifyKernel))
                status = GGML_STATUS_FAILED;
        }
        else
        {
            status = tsg::graph_compute_profiled(g_backend, graph, "gemma4 MoE model verify");
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            if (host_moe.empty())
                set_last_error("Gemma4 MoE model verify: graph execution failed.");
            return 0;
        }

        // The persistent gallocr buffer is NOT freed here — it is reused by the
        // next verify. The finalize below queues an async device->host blit of
        // hidden_out (which lives in that persistent buffer); it is drained either
        // by the C# caller's first logits read (host_read_barrier in
        // GetFloatPointer) or by the next verify's leading host_read_barrier
        // before it reuses the buffer, so there is no read-after-write hazard.
        pt.mark("compute");
        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(H) * N * sizeof(float));
        pt.mark("download");
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
        set_last_error("Unknown error in Gemma4 MoE model verify.");
        return 0;
    }
}

