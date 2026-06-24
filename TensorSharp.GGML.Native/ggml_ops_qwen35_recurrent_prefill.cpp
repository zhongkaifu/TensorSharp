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

using namespace tsg;

// ---------------------------------------------------------------------------
// TSGgml_Qwen35RecurrentLayerPrefill
// ---------------------------------------------------------------------------
// Device-resident, single-graph fused prefill for ONE Qwen3.5/Qwen3-Next
// gated-delta-net (recurrent) layer over N prompt tokens. This is the prefill
// sibling of the recurrent layer inside TSGgml_Qwen35ModelVerify (the K=N GDN
// construction): input RMSNorm + packed in-proj (qkv/z/beta/alpha) + ssm_conv
// + SiLU + L2-norm/head-tile + ggml_gated_delta_net(K=1 over N) + gated RMSNorm
// + output projection + residual add — ALL in one ggml graph dispatch.
//
// Why a dedicated kernel: the legacy per-op chunked prefill path downloads the
// in-projection output to the host, runs Conv1D/SiLU/pack on the CPU, re-uploads
// staging buffers, runs the scan, then downloads `gated`. On WDDM every one of
// those per-layer round-trips drains the GPU; on short prompts the GPU never
// ramps its clocks (a smaller prompt runs *slower* per call than a larger one),
// which is the ~1.5 s fixed-overhead floor that dominates short-prompt TTFT.
// Keeping the whole recurrent block on-device removes the round-trips and keeps
// the GPU busy. The dense/MoE FFN and attention layers stay on the C# fused
// paths (already single-dispatch); only the GDN block moves here.
//
// VRAM / weight discipline:
//   * The graph + activation buffer are CACHED per shape (keyed on N + weight
//     quant types + head geometry) and REUSED across ALL recurrent layers and
//     forwards (~one layer's worth of intermediates, not 30×). No per-call
//     context/buffer allocation (the trap that thrashes the ~14/16 GB 35B).
//   * The quantized projection weights are DEVICE-RESIDENT (shared with the
//     per-op path via the same CacheKey-keyed cacheable buffers). Each call
//     re-points the graph's weight leaves to the current layer's resident buffer
//     (try_get_cacheable_tensor_buffer → tensor->{buffer,data}); NO host re-read
//     of the (possibly synthetic / paged-out) weight pointer, NO re-upload.
//   * Per call we only ggml_backend_tensor_set the small F32 inputs (hidden,
//     conv/delta state, norm/conv/dt/a constants) and ggml_backend_tensor_get
//     the outputs.
//
// Returns 1 on success, 0 on any failure (caller falls back to the chunked path).
namespace
{
    struct Q35RecPrefillKey
    {
        int N, hidden, head_k_dim, head_v_dim, num_k_heads, num_v_heads, conv_kernel;
        int qkv_type, gate_type, beta_type, alpha_type, out_type;
        float eps;
    };

    struct Q35RecPrefillKeyEq
    {
        bool operator()(const Q35RecPrefillKey& a, const Q35RecPrefillKey& b) const noexcept
        {
            return a.N == b.N && a.hidden == b.hidden && a.head_k_dim == b.head_k_dim
                && a.head_v_dim == b.head_v_dim && a.num_k_heads == b.num_k_heads
                && a.num_v_heads == b.num_v_heads && a.conv_kernel == b.conv_kernel
                && a.qkv_type == b.qkv_type && a.gate_type == b.gate_type
                && a.beta_type == b.beta_type && a.alpha_type == b.alpha_type
                && a.out_type == b.out_type && a.eps == b.eps;
        }
    };

    struct Q35RecPrefillKeyHash
    {
        std::size_t operator()(const Q35RecPrefillKey& k) const noexcept
        {
            std::size_t h = 1469598103934665603ull;
            auto mix = [&](std::size_t v) { h = (h ^ v) * 1099511628211ull; };
            mix((std::size_t)k.N); mix((std::size_t)k.hidden);
            mix((std::size_t)k.head_k_dim); mix((std::size_t)k.head_v_dim);
            mix((std::size_t)k.num_k_heads); mix((std::size_t)k.num_v_heads);
            mix((std::size_t)k.conv_kernel);
            mix((std::size_t)k.qkv_type); mix((std::size_t)k.gate_type);
            mix((std::size_t)k.beta_type); mix((std::size_t)k.alpha_type); mix((std::size_t)k.out_type);
            std::uint32_t e; std::memcpy(&e, &k.eps, sizeof(e)); mix((std::size_t)e);
            return h;
        }
    };

    struct Q35RecPrefillEntry
    {
        std::unique_ptr<std::uint8_t[]> ctx_buffer;
        ggml_context* ctx = nullptr;
        BufferHandle buffer{nullptr};
        ggml_cgraph* graph = nullptr;

        // F32 input leaves (uploaded per call via tensor_set).
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* conv1d_w = nullptr;
        ggml_tensor* ssm_dt_w = nullptr;
        ggml_tensor* ssm_a_w = nullptr;
        ggml_tensor* ssm_norm_w = nullptr;
        ggml_tensor* conv_state_in = nullptr;
        ggml_tensor* delta_state_in = nullptr;

        // Projection-weight leaves (re-pointed to resident device buffers per call;
        // F32 ones are persist-allocated and uploaded via tensor_set).
        ggml_tensor* gdn_qkv_w = nullptr;
        ggml_tensor* gdn_gate_w = nullptr;
        ggml_tensor* ssm_beta_w = nullptr;
        ggml_tensor* ssm_alpha_w = nullptr;
        ggml_tensor* ssm_out_w = nullptr;

        // Output leaves (downloaded per call).
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* conv_state_out = nullptr;
        ggml_tensor* delta_state_out = nullptr;

        std::mutex compute_mutex;

        ~Q35RecPrefillEntry()
        {
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
        }
    };

    std::mutex g_q35_rec_prefill_mutex;
    std::unordered_map<Q35RecPrefillKey, std::unique_ptr<Q35RecPrefillEntry>,
                       Q35RecPrefillKeyHash, Q35RecPrefillKeyEq> g_q35_rec_prefill_cache;

    void ensure_q35_rec_prefill_cleanup_registered()
    {
        static std::once_flag flag;
        std::call_once(flag, []() {
            std::atexit([]() {
                std::lock_guard<std::mutex> lk(g_q35_rec_prefill_mutex);
                g_q35_rec_prefill_cache.clear();
            });
        });
    }
}

TSG_EXPORT int TSGgml_Qwen35RecurrentLayerPrefill(
    void* hidden_data, int hidden_size, int N,
    void* attn_norm_w_data,
    void* gdn_qkv_w_data, int gdn_qkv_type, std::int64_t gdn_qkv_ne0, std::int64_t gdn_qkv_ne1, std::int64_t gdn_qkv_bytes,
    void* gdn_gate_w_data, int gdn_gate_type, std::int64_t gdn_gate_ne0, std::int64_t gdn_gate_ne1, std::int64_t gdn_gate_bytes,
    void* ssm_beta_w_data, int ssm_beta_type, std::int64_t ssm_beta_ne0, std::int64_t ssm_beta_ne1, std::int64_t ssm_beta_bytes,
    void* ssm_alpha_w_data, int ssm_alpha_type, std::int64_t ssm_alpha_ne0, std::int64_t ssm_alpha_ne1, std::int64_t ssm_alpha_bytes,
    void* ssm_out_w_data, int ssm_out_type, std::int64_t ssm_out_ne0, std::int64_t ssm_out_ne1, std::int64_t ssm_out_bytes,
    void* conv1d_w_data, void* ssm_dt_w_data, void* ssm_a_w_data, void* ssm_norm_w_data,
    void* conv_state_in_data, void* delta_state_in_data,
    void* conv_state_out_data, void* delta_state_out_data,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (N <= 0 || hidden_size <= 0) { set_last_error("RecPrefill: bad N/hidden."); return 0; }
        if (num_k_heads <= 0 || num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0)
        { set_last_error("RecPrefill: bad head counts."); return 0; }

        const int H = hidden_size;
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = num_v_heads / num_k_heads;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        // Re-point a quantized projection-weight leaf to its device-resident
        // cacheable buffer (no host re-read; the per-op path shares the cache).
        // F32 weights (type 0) are persist-allocated and uploaded via tensor_set.
        auto bind_weight = [&](ggml_tensor* t, void* data, int type, std::size_t bytes) -> bool {
            if (type == 0) { ggml_backend_tensor_set(t, data, 0, bytes); return true; }
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
            if (!try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs,
                    GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                return false;
            t->buffer = buf; t->data = addr;
            if (needs) ggml_backend_tensor_set(t, data, 0, bytes);
            return true;
        };

        Q35RecPrefillKey cache_key{N, H, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            conv_kernel, gdn_qkv_type, gdn_gate_type, ssm_beta_type, ssm_alpha_type, ssm_out_type, eps};

        Q35RecPrefillEntry* entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_q35_rec_prefill_mutex);
            auto it = g_q35_rec_prefill_cache.find(cache_key);
            if (it != g_q35_rec_prefill_cache.end()) entry = it->second.get();
        }

        if (entry == nullptr)
        {
            std::size_t ctx_size = 6 * 1024 * 1024;
            auto ne = std::make_unique<Q35RecPrefillEntry>();
            ne->ctx_buffer = std::make_unique<std::uint8_t[]>(ctx_size);
            ggml_init_params params = {};
            params.mem_size = ctx_size;
            params.mem_buffer = ne->ctx_buffer.get();
            params.no_alloc = true;
            ne->ctx = ggml_init(params);
            if (ne->ctx == nullptr) { set_last_error("RecPrefill: ctx init failed."); return 0; }
            ggml_context* ctx = ne->ctx;

            ggml_tensor* hidden_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            ggml_tensor* attn_norm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_tensor* gdn_qkv_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gdn_qkv_type), gdn_qkv_ne0, gdn_qkv_ne1);
            ggml_tensor* gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gdn_gate_type), gdn_gate_ne0, gdn_gate_ne1);
            ggml_tensor* ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ssm_beta_type), ssm_beta_ne0, ssm_beta_ne1);
            ggml_tensor* ssm_alpha_w= ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ssm_alpha_type), ssm_alpha_ne0, ssm_alpha_ne1);
            ggml_tensor* ssm_out_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ssm_out_type), ssm_out_ne0, ssm_out_ne1);
            ggml_tensor* conv1d_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
            ggml_tensor* ssm_dt_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
            ggml_tensor* ssm_a_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
            ggml_tensor* ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
            ggml_tensor* conv_state_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
            ggml_tensor* delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);

            // ---- graph ----
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm); // [H, N]

            ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, gdn_qkv_w, normed);   // [conv_dim, N]
            ggml_tensor* z_all     = ggml_mul_mat(ctx, gdn_gate_w, normed);  // [value_dim, N]
            ggml_tensor* beta_all  = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ssm_beta_w, normed));
            ggml_tensor* alpha_all = ggml_mul_mat(ctx, ssm_alpha_w, normed);
            ggml_tensor* g_all     = ggml_softplus(ctx, ggml_add(ctx, alpha_all, ssm_dt_w));
            g_all = ggml_mul(ctx, g_all, ssm_a_w);

            ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_mixed)); // [N, conv_dim]
            ggml_tensor* conv_input = ggml_concat(ctx, conv_state_in, qkv_T, 0);  // [convDim+N, conv_dim]
            ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, conv1d_w)); // [conv_dim, N]
            ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                conv_input->nb[1], static_cast<std::size_t>(N) * conv_input->nb[0]));

            ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1], 0));
            ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1],
                static_cast<std::size_t>(key_dim) * sizeof(float)));
            ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, value_dim, N, conv_out->nb[1],
                static_cast<std::size_t>(2 * key_dim) * sizeof(float)));

            ggml_tensor* q_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, q_part, head_k_dim, num_k_heads * N), eps);
            ggml_tensor* k_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, k_part, head_k_dim, num_k_heads * N), eps);
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_hn, head_k_dim, num_k_heads, N);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_hn, head_k_dim, num_k_heads, N);
            ggml_tensor* q_tl = q_3d; ggml_tensor* k_tl = k_3d;
            for (int r = 1; r < head_tile; r++) { q_tl = ggml_concat(ctx, q_tl, q_3d, 1); k_tl = ggml_concat(ctx, k_tl, k_3d, 1); }
            ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, N, 1);
            ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, N, 1);
            ggml_tensor* v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, N, 1);
            ggml_tensor* g4 = ggml_reshape_4d(ctx, ggml_cont(ctx, g_all), 1, num_v_heads, N, 1);
            ggml_tensor* beta4 = ggml_reshape_4d(ctx, ggml_cont(ctx, beta_all), 1, num_v_heads, N, 1);
            ggml_tensor* state4 = ggml_reshape_4d(ctx, delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

            ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g4, beta4, state4, 1);
            ggml_tensor* gdn_out = ggml_view_2d(ctx, gdn, value_dim, N, ggml_row_size(gdn->type, value_dim), 0);
            ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                ggml_row_size(gdn->type, head_k_dim),
                ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N));

            ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads * N);
            ggml_tensor* out_n  = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), ssm_norm_w);
            ggml_tensor* out_n_3d = ggml_reshape_3d(ctx, out_n, head_v_dim, num_v_heads, N);
            ggml_tensor* z_3d = ggml_reshape_3d(ctx, z_all, head_v_dim, num_v_heads, N);
            ggml_tensor* gated = ggml_mul(ctx, out_n_3d, ggml_silu(ctx, z_3d));
            ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, N);
            ggml_tensor* block_out = ggml_mul_mat(ctx, ssm_out_w, gated_flat); // [H, N]
            ggml_tensor* hidden_res = ggml_add(ctx, hidden_t, block_out);      // residual add

            ggml_tensor* hidden_out      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            ggml_tensor* conv_state_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
            ggml_tensor* delta_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
            ggml_tensor* hidden_cpy = ggml_cpy(ctx, hidden_res, hidden_out);
            ggml_tensor* conv_cpy   = ggml_cpy(ctx, new_conv, conv_state_out);
            ggml_tensor* delta_cpy  = ggml_cpy(ctx, new_state, delta_state_out);
            ggml_set_output(hidden_cpy); ggml_set_output(conv_cpy); ggml_set_output(delta_cpy);

            ne->graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
            ggml_build_forward_expand(ne->graph, hidden_cpy);
            ggml_build_forward_expand(ne->graph, conv_cpy);
            ggml_build_forward_expand(ne->graph, delta_cpy);

            // Bind ONLY the QUANTIZED projection weights to their resident device
            // buffers BEFORE alloc_ctx_tensors so the bump allocator skips them (they
            // already have a buffer). F32 weights stay unbound -> persist-allocated and
            // uploaded per call. (Calling bind_weight's F32 tensor_set branch here would
            // fault: the tensor isn't allocated until alloc_ctx_tensors below.)
            host_read_barrier();
            auto build_bind_q = [&](ggml_tensor* t, void* data, int type, std::size_t bytes) -> bool {
                if (type == 0) return true; // F32: leave for alloc_ctx_tensors
                return bind_weight(t, data, type, bytes);
            };
            if (!build_bind_q(gdn_qkv_w,  gdn_qkv_w_data,  gdn_qkv_type,  (std::size_t)gdn_qkv_bytes)
             || !build_bind_q(gdn_gate_w, gdn_gate_w_data, gdn_gate_type, (std::size_t)gdn_gate_bytes)
             || !build_bind_q(ssm_beta_w, ssm_beta_w_data, ssm_beta_type, (std::size_t)ssm_beta_bytes)
             || !build_bind_q(ssm_alpha_w,ssm_alpha_w_data,ssm_alpha_type,(std::size_t)ssm_alpha_bytes)
             || !build_bind_q(ssm_out_w,  ssm_out_w_data,  ssm_out_type,  (std::size_t)ssm_out_bytes))
            {
                set_last_error("RecPrefill: weight bind failed."); return 0;
            }

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (!buffer.value) { set_last_error("RecPrefill: buffer alloc failed."); return 0; }
            ne->buffer = std::move(buffer);

            ne->hidden_t = hidden_t; ne->attn_norm_w = attn_norm;
            ne->conv1d_w = conv1d_w; ne->ssm_dt_w = ssm_dt_w; ne->ssm_a_w = ssm_a_w; ne->ssm_norm_w = ssm_norm_w;
            ne->conv_state_in = conv_state_in; ne->delta_state_in = delta_state_in;
            ne->gdn_qkv_w = gdn_qkv_w; ne->gdn_gate_w = gdn_gate_w;
            ne->ssm_beta_w = ssm_beta_w; ne->ssm_alpha_w = ssm_alpha_w; ne->ssm_out_w = ssm_out_w;
            ne->hidden_out = hidden_out; ne->conv_state_out = conv_state_out; ne->delta_state_out = delta_state_out;

            std::lock_guard<std::mutex> lk(g_q35_rec_prefill_mutex);
            // Bound VRAM: every recurrent layer of one prefill forward shares N (the
            // prompt length), so a forward needs exactly ONE cached graph. Evict any
            // cached graphs for a DIFFERENT N before inserting this one — otherwise a
            // server seeing many prompt lengths would accumulate per-N activation
            // buffers (each ~hundreds of MB at long N) and spill the 16 GB-tight 35B.
            for (auto it2 = g_q35_rec_prefill_cache.begin(); it2 != g_q35_rec_prefill_cache.end(); )
            {
                if (it2->first.N != N) it2 = g_q35_rec_prefill_cache.erase(it2);
                else ++it2;
            }
            auto [it, inserted] = g_q35_rec_prefill_cache.emplace(cache_key, std::move(ne));
            entry = it->second.get();
            ensure_q35_rec_prefill_cleanup_registered();
        }

        // ---- bind weights for THIS layer + upload inputs + compute + download ----
        {
            std::lock_guard<std::mutex> lk(entry->compute_mutex);
            host_read_barrier();

            // Re-point quantized weight leaves to this layer's resident buffers (F32
            // weights are uploaded). A wrong bind here would read another layer's
            // weights -> bail so the caller falls back rather than produce garbage.
            if (!bind_weight(entry->gdn_qkv_w,  gdn_qkv_w_data,  gdn_qkv_type,  (std::size_t)gdn_qkv_bytes)
             || !bind_weight(entry->gdn_gate_w, gdn_gate_w_data, gdn_gate_type, (std::size_t)gdn_gate_bytes)
             || !bind_weight(entry->ssm_beta_w, ssm_beta_w_data, ssm_beta_type, (std::size_t)ssm_beta_bytes)
             || !bind_weight(entry->ssm_alpha_w,ssm_alpha_w_data,ssm_alpha_type,(std::size_t)ssm_alpha_bytes)
             || !bind_weight(entry->ssm_out_w,  ssm_out_w_data,  ssm_out_type,  (std::size_t)ssm_out_bytes))
            { set_last_error("RecPrefill: per-call weight bind failed."); return 0; }

            ggml_backend_tensor_set(entry->hidden_t,       hidden_data,        0, ggml_nbytes(entry->hidden_t));
            ggml_backend_tensor_set(entry->attn_norm_w,    attn_norm_w_data,   0, ggml_nbytes(entry->attn_norm_w));
            ggml_backend_tensor_set(entry->conv1d_w,       conv1d_w_data,      0, ggml_nbytes(entry->conv1d_w));
            ggml_backend_tensor_set(entry->ssm_dt_w,       ssm_dt_w_data,      0, ggml_nbytes(entry->ssm_dt_w));
            ggml_backend_tensor_set(entry->ssm_a_w,        ssm_a_w_data,       0, ggml_nbytes(entry->ssm_a_w));
            ggml_backend_tensor_set(entry->ssm_norm_w,     ssm_norm_w_data,    0, ggml_nbytes(entry->ssm_norm_w));
            ggml_backend_tensor_set(entry->conv_state_in,  conv_state_in_data, 0, ggml_nbytes(entry->conv_state_in));
            ggml_backend_tensor_set(entry->delta_state_in, delta_state_in_data,0, ggml_nbytes(entry->delta_state_in));

            ggml_status status = ggml_backend_graph_compute(g_backend, entry->graph);
            if (status != GGML_STATUS_SUCCESS) { set_last_error("RecPrefill: graph compute failed."); return 0; }
            ggml_backend_synchronize(g_backend);

            ggml_backend_tensor_get(entry->hidden_out,      hidden_data,         0, ggml_nbytes(entry->hidden_out));
            ggml_backend_tensor_get(entry->conv_state_out,  conv_state_out_data, 0, ggml_nbytes(entry->conv_state_out));
            ggml_backend_tensor_get(entry->delta_state_out, delta_state_out_data,0, ggml_nbytes(entry->delta_state_out));
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 recurrent layer prefill."); return 0; }
}
