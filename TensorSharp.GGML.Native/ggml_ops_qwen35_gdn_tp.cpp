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
#include "ggml_ops_q8_weight_policy.h"

using namespace tsg;

// ---------------------------------------------------------------------------
// TSGgml_Qwen35GdnLayerTP
// ---------------------------------------------------------------------------
// One rank's share of a Qwen3.5 / Qwen3.6 GatedDeltaNet block, as ONE ggml
// graph: input RMSNorm, the PACKED in-projection, ssm_conv over the recurrent
// conv window, SiLU, q/k L2-norm + head tiling, ggml_gated_delta_net, the gated
// RMSNorm and the SiLU(z) gate. The caller finishes the block with the
// row-parallel ssm_out matmul and the AllReduce.
//
// This is the GGML counterpart of the direct-CUDA `ts_qwen35_gdn_*` packed
// kernel, and it is what lets Qwen3.5 run tensor-parallel on the GGML backends.
// The pieces the bridge already had — the chunked prefill kernel and the batched
// per-token step — both take Q/K/V/Z/beta/alpha already split apart, with the
// conv1d and the gate arithmetic done on the host. Under TP that shape is
// unusable: each rank owns a block-cyclic slice of the V heads, so every one of
// those host steps would run per rank per layer with a device round-trip on
// either side.
//
// Two properties matter for speed:
//
//   * The recurrent state stays ON THE DEVICE. Both the conv window and the
//     delta state are bound through the per-rank cacheable-buffer cache, keyed
//     on the caller's host pointer, so they upload once and are then updated
//     in place by the graph. Downloading the delta state would cost
//     num_v_heads * head_v_dim * head_k_dim floats per layer per token — 1 MB
//     per layer on the 35B, ~60 MB per token per rank round-tripped over PCIe.
//     C# drops the device copy (InvalidateHostBuffer) when it resets the cache.
//
//   * The graph is CACHED per (rank, shape). Ranks run concurrently on their own
//     ggml backends, so the cache is keyed on the rank as well: sharing one
//     entry would hand rank 1 an activation buffer living on rank 0's GPU.
//
// Returns 1 on success, 0 on failure (the caller reports it; there is no
// silently-wrong path).
namespace
{
    struct GdnTpKey
    {
        int rank, N, hidden, head_k_dim, head_v_dim, num_k_heads, num_v_heads;
        int conv_kernel, inproj_type;
        float eps;
    };

    struct GdnTpKeyEq
    {
        bool operator()(const GdnTpKey& a, const GdnTpKey& b) const noexcept
        {
            return a.rank == b.rank && a.N == b.N && a.hidden == b.hidden
                && a.head_k_dim == b.head_k_dim && a.head_v_dim == b.head_v_dim
                && a.num_k_heads == b.num_k_heads && a.num_v_heads == b.num_v_heads
                && a.conv_kernel == b.conv_kernel && a.inproj_type == b.inproj_type
                && a.eps == b.eps;
        }
    };

    struct GdnTpKeyHash
    {
        std::size_t operator()(const GdnTpKey& k) const noexcept
        {
            std::size_t h = 1469598103934665603ull;
            auto mix = [&](std::size_t v) { h = (h ^ v) * 1099511628211ull; };
            mix((std::size_t)k.rank); mix((std::size_t)k.N); mix((std::size_t)k.hidden);
            mix((std::size_t)k.head_k_dim); mix((std::size_t)k.head_v_dim);
            mix((std::size_t)k.num_k_heads); mix((std::size_t)k.num_v_heads);
            mix((std::size_t)k.conv_kernel); mix((std::size_t)k.inproj_type);
            std::uint32_t e; std::memcpy(&e, &k.eps, sizeof(e)); mix((std::size_t)e);
            return h;
        }
    };

    struct GdnTpEntry
    {
        std::unique_ptr<std::uint8_t[]> ctx_buffer;
        ggml_context* ctx = nullptr;
        BufferHandle buffer{nullptr};
        ggml_cgraph* graph = nullptr;

        // Uploaded per call (all small).
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* conv1d_w = nullptr;
        ggml_tensor* dt_bias_t = nullptr;
        ggml_tensor* a_log_t = nullptr;
        ggml_tensor* ssm_norm_t = nullptr;

        // Re-pointed per call at the layer's resident buffers.
        ggml_tensor* inproj_w = nullptr;
        ggml_tensor* conv_state_t = nullptr;
        ggml_tensor* delta_state_t = nullptr;

        // Views OF the two state tensors. A view's data pointer is resolved once,
        // when the graph is allocated, so re-pointing the base alone would leave
        // these addressing the buffers of whichever layer built the graph — every
        // later layer would then read and write layer 0's recurrent state.
        ggml_tensor* delta_state_view = nullptr;  // the 4D state fed to gated_delta_net
        ggml_tensor* conv_state_write = nullptr;  // ggml_cpy dst view
        ggml_tensor* delta_state_write = nullptr; // ggml_cpy dst view

        // Downloaded per call.
        ggml_tensor* gated_out_t = nullptr;

        std::mutex compute_mutex;

        ~GdnTpEntry()
        {
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
        }
    };

    std::mutex g_gdn_tp_mutex;
    std::unordered_map<GdnTpKey, std::unique_ptr<GdnTpEntry>, GdnTpKeyHash, GdnTpKeyEq> g_gdn_tp_cache;

    void ensure_gdn_tp_cleanup_registered()
    {
        static std::once_flag flag;
        std::call_once(flag, []() {
            std::atexit([]() {
                std::lock_guard<std::mutex> lk(g_gdn_tp_mutex);
                g_gdn_tp_cache.clear();
            });
        });
    }
}

TSG_EXPORT int TSGgml_Qwen35GdnLayerTP(
    void* hidden_data, int hidden_size, int N,
    void* attn_norm_w_data,
    void* inproj_w_data, int inproj_type,
    std::int64_t inproj_ne0, std::int64_t inproj_ne1, std::int64_t inproj_bytes,
    void* conv1d_w_data, void* dt_bias_data, void* a_log_data, void* ssm_norm_w_data,
    void* conv_state_data, void* delta_state_data,
    void* gated_out_data,
    int packed_dim, int qkv_dim, int qk_dim, int v_dim,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_kernel, float eps)
{
    try
    {
        if (bonsai_weight_has_transform(inproj_w_data)) {
            set_last_error("Qwen35GdnLayerTP: Bonsai Hadamard transforms are not supported by this tensor-parallel graph.");
            return 0;
        }
        if (!ensure_backend()) return 0;

        if (hidden_data == nullptr || attn_norm_w_data == nullptr || inproj_w_data == nullptr ||
            conv1d_w_data == nullptr || dt_bias_data == nullptr || a_log_data == nullptr ||
            ssm_norm_w_data == nullptr || conv_state_data == nullptr ||
            delta_state_data == nullptr || gated_out_data == nullptr)
        {
            set_last_error("Qwen35GdnLayerTP: null pointer argument.");
            return 0;
        }
        if (N <= 0 || hidden_size <= 0 || conv_kernel <= 1)
        {
            set_last_error("Qwen35GdnLayerTP: bad N / hidden_size / conv_kernel.");
            return 0;
        }
        if (num_k_heads <= 0 || num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0)
        {
            set_last_error("Qwen35GdnLayerTP: V heads must be a multiple of K heads.");
            return 0;
        }
        if (head_k_dim != head_v_dim)
        {
            // ggml_gated_delta_net requires S_k == S_v for the packed state.
            set_last_error("Qwen35GdnLayerTP: head_k_dim must equal head_v_dim.");
            return 0;
        }
        if (qk_dim != head_k_dim * num_k_heads || v_dim != head_v_dim * num_v_heads ||
            qkv_dim != 2 * qk_dim + v_dim ||
            packed_dim != qkv_dim + v_dim + 2 * num_v_heads)
        {
            set_last_error("Qwen35GdnLayerTP: packed layout dimensions are inconsistent.");
            return 0;
        }
        if (inproj_ne0 != hidden_size || inproj_ne1 != packed_dim)
        {
            set_last_error("Qwen35GdnLayerTP: in-projection shard shape does not match the packed layout.");
            return 0;
        }

        const int H = hidden_size;
        const int convDim = conv_kernel - 1;
        const int head_tile = num_v_heads / num_k_heads;
        const int rank = tsg::g_active_rank;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        // Point a leaf at its device-resident buffer. Weights are read-only;
        // the two recurrent-state tensors are read AND written by the graph, so
        // they are bound with COMPUTE usage (a WEIGHTS buffer may be wrapped
        // zero-copy around the host pointer on unified-memory backends, which
        // would silently make the in-place state update a host write).
        // `data` doubles as the cache key. For a preloaded quantized weight that
        // key is a GCHandle, not the weight bytes, so the miss path must resolve
        // it before uploading (see resolve_upload_source) — uploading from the
        // key itself would dereference an address that is not weight memory.
        auto bind_resident = [&](ggml_tensor* t, void* data, std::size_t bytes,
                                 enum ggml_backend_buffer_usage usage) -> bool {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
            if (!try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs, usage))
                return false;
            t->buffer = buf; t->data = addr;
            // Upload only when the buffer was just created — that is what makes
            // the recurrent state persist on the device across calls, and what
            // makes a re-created (invalidated) state pick up the host reset.
            if (needs)
                ggml_backend_tensor_set(t, resolve_upload_source(data), 0, bytes);
            return true;
        };

        const std::size_t conv_state_bytes =
            static_cast<std::size_t>(convDim) * static_cast<std::size_t>(qkv_dim) * sizeof(float);
        const std::size_t delta_state_bytes =
            static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(head_v_dim) *
            static_cast<std::size_t>(head_k_dim) * sizeof(float);

        GdnTpKey cache_key{rank, N, H, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
                           conv_kernel, inproj_type, eps};

        GdnTpEntry* entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_gdn_tp_mutex);
            auto it = g_gdn_tp_cache.find(cache_key);
            if (it != g_gdn_tp_cache.end()) entry = it->second.get();
        }

        if (entry == nullptr)
        {
            const std::size_t ctx_size = 4 * 1024 * 1024;
            auto ne = std::make_unique<GdnTpEntry>();
            ne->ctx_buffer = std::make_unique<std::uint8_t[]>(ctx_size);
            ggml_init_params params = {};
            params.mem_size = ctx_size;
            params.mem_buffer = ne->ctx_buffer.get();
            params.no_alloc = true;
            ne->ctx = ggml_init(params);
            if (ne->ctx == nullptr) { set_last_error("Qwen35GdnLayerTP: ctx init failed."); return 0; }
            ggml_context* ctx = ne->ctx;

            ggml_tensor* hidden_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_tensor* inproj_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(inproj_type),
                                                          inproj_ne0, inproj_ne1);
            ggml_tensor* conv1d_w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, qkv_dim);
            ggml_tensor* dt_bias_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
            ggml_tensor* a_log_t     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
            ggml_tensor* ssm_norm_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
            ggml_tensor* conv_state_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, qkv_dim);
            ggml_tensor* delta_state_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                            head_k_dim, head_v_dim, num_v_heads);
            ggml_tensor* gated_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, v_dim, N);

            if (hidden_t == nullptr || attn_norm_w == nullptr || inproj_w == nullptr ||
                conv1d_w == nullptr || dt_bias_t == nullptr || a_log_t == nullptr ||
                ssm_norm_t == nullptr || conv_state_t == nullptr || delta_state_t == nullptr ||
                gated_out_t == nullptr)
            {
                set_last_error("Qwen35GdnLayerTP: tensor allocation failed.");
                return 0;
            }

            // ---- graph ----
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);

            // Packed in-projection: [Q | K | V | Z | beta | alpha] for this rank's heads.
            ggml_tensor* packed = tsg::weight_mul_mat(ctx, inproj_w, normed, inproj_w_data); // [packed_dim, N]
            const std::size_t f32 = sizeof(float);

            ggml_tensor* qkv_v = ggml_view_2d(ctx, packed, qkv_dim, N, packed->nb[1], 0);
            ggml_tensor* z_v   = ggml_view_2d(ctx, packed, v_dim, N, packed->nb[1],
                                              static_cast<std::size_t>(qkv_dim) * f32);
            ggml_tensor* beta_v = ggml_view_2d(ctx, packed, num_v_heads, N, packed->nb[1],
                                               static_cast<std::size_t>(qkv_dim + v_dim) * f32);
            ggml_tensor* alpha_v = ggml_view_2d(ctx, packed, num_v_heads, N, packed->nb[1],
                                                static_cast<std::size_t>(qkv_dim + v_dim + num_v_heads) * f32);

            ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_cont(ctx, beta_v));
            ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, ggml_cont(ctx, alpha_v), dt_bias_t));
            g_all = ggml_mul(ctx, g_all, a_log_t);

            // Conv window: [convDim + N, qkv_dim] with time along ne0.
            ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, ggml_cont(ctx, qkv_v)));
            ggml_tensor* conv_input = ggml_concat(ctx, conv_state_t, qkv_T, 0);
            ggml_tensor* conv_out = ggml_silu(ctx, tsg::ssm_conv_any(ctx, conv_input, conv1d_w)); // [qkv_dim, N]
            ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, qkv_dim,
                conv_input->nb[1], static_cast<std::size_t>(N) * conv_input->nb[0]));

            ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, qk_dim, N, conv_out->nb[1], 0));
            ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, qk_dim, N, conv_out->nb[1],
                static_cast<std::size_t>(qk_dim) * f32));
            ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, v_dim, N, conv_out->nb[1],
                static_cast<std::size_t>(2 * qk_dim) * f32));

            ggml_tensor* q_hn = build_gdn_l2_norm(ctx, ggml_reshape_2d(ctx, q_part, head_k_dim, num_k_heads * N), eps);
            ggml_tensor* k_hn = build_gdn_l2_norm(ctx, ggml_reshape_2d(ctx, k_part, head_k_dim, num_k_heads * N), eps);
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_hn, head_k_dim, num_k_heads, N);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_hn, head_k_dim, num_k_heads, N);

            // Tile K-heads up to V-heads: local V head lv reads local K head
            // lv % num_k_heads, which is exactly what the block-cyclic TP shard
            // preserves (rank r owns a contiguous K block and every V head that
            // maps into it, in ascending global order).
            ggml_tensor* q_tl = q_3d;
            ggml_tensor* k_tl = k_3d;
            for (int r = 1; r < head_tile; r++)
            {
                q_tl = ggml_concat(ctx, q_tl, q_3d, 1);
                k_tl = ggml_concat(ctx, k_tl, k_3d, 1);
            }

            ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, N, 1);
            ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, N, 1);
            ggml_tensor* v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, N, 1);
            ggml_tensor* g4 = ggml_reshape_4d(ctx, ggml_cont(ctx, g_all), 1, num_v_heads, N, 1);
            ggml_tensor* beta4 = ggml_reshape_4d(ctx, ggml_cont(ctx, beta_all), 1, num_v_heads, N, 1);
            ggml_tensor* state4 = ggml_reshape_4d(ctx, delta_state_t, head_k_dim, head_v_dim, num_v_heads, 1);

            ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g4, beta4, state4, 1);
            if (gdn == nullptr)
            {
                set_last_error("Qwen35GdnLayerTP: ggml_gated_delta_net node creation failed.");
                return 0;
            }

            ggml_tensor* gdn_scores = ggml_view_2d(ctx, gdn, v_dim, N, ggml_row_size(gdn->type, v_dim), 0);
            ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                ggml_row_size(gdn->type, head_k_dim),
                ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                ggml_row_size(gdn->type, v_dim) * static_cast<std::size_t>(N));

            ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_scores), head_v_dim, num_v_heads * N);
            ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), ssm_norm_t);
            ggml_tensor* out_n_3d = ggml_reshape_3d(ctx, out_n, head_v_dim, num_v_heads, N);
            ggml_tensor* z_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, z_v), head_v_dim, num_v_heads, N);
            ggml_tensor* gated = ggml_mul(ctx, out_n_3d, ggml_silu(ctx, z_3d));

            ggml_tensor* gated_cpy = ggml_cpy(ctx, ggml_reshape_2d(ctx, gated, v_dim, N), gated_out_t);
            // Recurrent state is updated IN PLACE: both copies are the last nodes
            // in topological order, after the readers have consumed the old value.
            ggml_tensor* conv_cpy = ggml_cpy(ctx, new_conv, conv_state_t);
            ggml_tensor* delta_cpy = ggml_cpy(ctx, new_state, delta_state_t);
            ggml_set_output(gated_cpy); ggml_set_output(conv_cpy); ggml_set_output(delta_cpy);

            ne->graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
            if (ne->graph == nullptr) { set_last_error("Qwen35GdnLayerTP: graph creation failed."); return 0; }
            ggml_build_forward_expand(ne->graph, gated_cpy);
            ggml_build_forward_expand(ne->graph, conv_cpy);
            ggml_build_forward_expand(ne->graph, delta_cpy);

            // Bind the externally-resident leaves BEFORE alloc_ctx_tensors so the
            // bump allocator skips them: they already own a buffer, and the
            // in-projection shard is far too big to duplicate into the per-shape
            // activation buffer.
            host_read_barrier();
            if (!bind_resident(inproj_w, inproj_w_data, static_cast<std::size_t>(inproj_bytes),
                               GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
             || !bind_resident(conv_state_t, conv_state_data, conv_state_bytes,
                               GGML_BACKEND_BUFFER_USAGE_COMPUTE)
             || !bind_resident(delta_state_t, delta_state_data, delta_state_bytes,
                               GGML_BACKEND_BUFFER_USAGE_COMPUTE))
            {
                set_last_error("Qwen35GdnLayerTP: resident buffer bind failed at graph build.");
                return 0;
            }

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (!buffer.value) { set_last_error("Qwen35GdnLayerTP: activation buffer alloc failed."); return 0; }
            ne->buffer = std::move(buffer);

            ne->hidden_t = hidden_t; ne->attn_norm_w = attn_norm_w;
            ne->conv1d_w = conv1d_w; ne->dt_bias_t = dt_bias_t; ne->a_log_t = a_log_t;
            ne->ssm_norm_t = ssm_norm_t;
            ne->inproj_w = inproj_w;
            ne->conv_state_t = conv_state_t; ne->delta_state_t = delta_state_t;
            ne->delta_state_view = state4;
            ne->conv_state_write = conv_cpy;
            ne->delta_state_write = delta_cpy;
            ne->gated_out_t = gated_out_t;

            std::lock_guard<std::mutex> lk(g_gdn_tp_mutex);
            // One prefill forward drives every recurrent layer at the same N, so
            // a rank needs exactly one cached graph. Evict this rank's entries for
            // other N values rather than accumulating an activation buffer per
            // prompt length (each is hundreds of MB at long N, on a card the 35B
            // already fills).
            for (auto it2 = g_gdn_tp_cache.begin(); it2 != g_gdn_tp_cache.end(); )
            {
                if (it2->first.rank == rank && it2->first.N != N) it2 = g_gdn_tp_cache.erase(it2);
                else ++it2;
            }
            auto [it, inserted] = g_gdn_tp_cache.emplace(cache_key, std::move(ne));
            entry = it->second.get();
            ensure_gdn_tp_cleanup_registered();
        }

        // ---- per-call: bind this layer's weights + state, upload, compute ----
        {
            std::lock_guard<std::mutex> lk(entry->compute_mutex);
            host_read_barrier();

            if (!bind_resident(entry->inproj_w, inproj_w_data, static_cast<std::size_t>(inproj_bytes),
                               GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
             || !bind_resident(entry->conv_state_t, conv_state_data, conv_state_bytes,
                               GGML_BACKEND_BUFFER_USAGE_COMPUTE)
             || !bind_resident(entry->delta_state_t, delta_state_data, delta_state_bytes,
                               GGML_BACKEND_BUFFER_USAGE_COMPUTE))
            {
                set_last_error("Qwen35GdnLayerTP: per-call resident buffer bind failed.");
                return 0;
            }

            // Follow the base tensors with their views (see the field comments):
            // the graph is shared by every recurrent layer, so this is what makes
            // each layer read and write its OWN recurrent state.
            auto repoint_view = [](ggml_tensor* view, const ggml_tensor* base) {
                view->buffer = base->buffer;
                view->data = static_cast<char*>(base->data) + view->view_offs;
            };
            repoint_view(entry->delta_state_view, entry->delta_state_t);
            repoint_view(entry->conv_state_write, entry->conv_state_t);
            repoint_view(entry->delta_state_write, entry->delta_state_t);

            ggml_backend_tensor_set(entry->hidden_t,    hidden_data,      0, ggml_nbytes(entry->hidden_t));
            ggml_backend_tensor_set(entry->attn_norm_w, attn_norm_w_data, 0, ggml_nbytes(entry->attn_norm_w));
            ggml_backend_tensor_set(entry->conv1d_w,    conv1d_w_data,    0, ggml_nbytes(entry->conv1d_w));
            ggml_backend_tensor_set(entry->dt_bias_t,   dt_bias_data,     0, ggml_nbytes(entry->dt_bias_t));
            ggml_backend_tensor_set(entry->a_log_t,     a_log_data,       0, ggml_nbytes(entry->a_log_t));
            ggml_backend_tensor_set(entry->ssm_norm_t,  ssm_norm_w_data,  0, ggml_nbytes(entry->ssm_norm_t));

            ggml_status status = tsg::compute_graph(g_backend, entry->graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen35GdnLayerTP: graph compute failed.");
                return 0;
            }
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(entry->gated_out_t, gated_out_data, 0, ggml_nbytes(entry->gated_out_t));
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen35GdnLayerTP."); return 0; }
}

// ---------------------------------------------------------------------------
// TSGgml_Qwen35GdnDropTpGraphs
// ---------------------------------------------------------------------------
// Release every cached per-rank GDN graph and its activation buffer. Called when
// a TP model is disposed so its VRAM does not outlive it (the entries are keyed
// on rank + shape, not on the model, so nothing else reclaims them).
TSG_EXPORT void TSGgml_Qwen35GdnDropTpGraphs()
{
    std::lock_guard<std::mutex> lk(g_gdn_tp_mutex);
    g_gdn_tp_cache.clear();
}
