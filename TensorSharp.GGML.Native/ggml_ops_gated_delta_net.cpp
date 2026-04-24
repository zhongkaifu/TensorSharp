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
// GatedDeltaNetChunked graph cache
// ---------------------------------------------------------------------------
// The chunked GDN kernel is invoked O(num_recurrent_layers * prefill_chunks)
// times per request, and every call previously rebuilt the full GGML graph,
// allocated a fresh backend buffer, and re-uploaded constants like the mask
// scratch. For typical Qwen3.5 traces this dominated the chunked path latency
// (~150 ms/call out of ~170 ms/call on Metal/CPU).
//
// We hash on the shape tuple (T, H, D, chunk_size, eps). For a given shape the
// graph topology is identical, so we can build it once, retain the backend
// buffer, and on subsequent calls just rebind input data via
// ggml_backend_tensor_set / _get. ssm_norm_w is uploaded per call because it
// is a per-layer constant and our key intentionally ignores layer identity to
// share entries across all recurrent layers of the same shape.
//
// Cache lifetime: entries persist for the process lifetime (cleared on
// backend reset) since the working set is bounded by the number of distinct
// (T, chunk_size) pairs the model sees, which is typically just one or two.
namespace
{
    struct GdnChunkedCacheKey
    {
        int T;
        int H;
        int D;
        int cS;
        float eps;
    };

    struct GdnChunkedCacheKeyEq
    {
        bool operator()(const GdnChunkedCacheKey& a, const GdnChunkedCacheKey& b) const noexcept
        {
            return a.T == b.T && a.H == b.H && a.D == b.D && a.cS == b.cS && a.eps == b.eps;
        }
    };

    struct GdnChunkedCacheKeyHash
    {
        std::size_t operator()(const GdnChunkedCacheKey& k) const noexcept
        {
            std::size_t h = static_cast<std::size_t>(k.T);
            h = h * 1315423911u + static_cast<std::size_t>(k.H);
            h = h * 1315423911u + static_cast<std::size_t>(k.D);
            h = h * 1315423911u + static_cast<std::size_t>(k.cS);
            std::uint32_t e_bits;
            std::memcpy(&e_bits, &k.eps, sizeof(e_bits));
            h = h * 1315423911u + static_cast<std::size_t>(e_bits);
            return h;
        }
    };

    struct GdnChunkedCacheEntry
    {
        // Owns its own metadata buffer so we don't hold a slab from the per-call
        // pool indefinitely. ggml_init with no_alloc=true lays tensor metadata in
        // this buffer; backend storage lives in the BufferHandle below.
        std::unique_ptr<std::uint8_t[]> ctx_buffer;
        ggml_context* ctx = nullptr;
        BufferHandle buffer{nullptr};
        ggml_cgraph* graph = nullptr;

        // Storage tensors (the leaves backing each binding view) - upload targets.
        ggml_tensor* q_storage = nullptr;
        ggml_tensor* k_storage = nullptr;
        ggml_tensor* v_storage = nullptr;
        ggml_tensor* z_storage = nullptr;
        ggml_tensor* alpha_storage = nullptr;
        ggml_tensor* beta_storage = nullptr;
        ggml_tensor* state_storage = nullptr;
        ggml_tensor* ssm_norm_storage = nullptr;

        // Output storage for download.
        ggml_tensor* gated_out_storage = nullptr;

        // Sizes from create_standard_binding (raw_bytes).
        std::size_t q_bytes = 0;
        std::size_t k_bytes = 0;
        std::size_t v_bytes = 0;
        std::size_t z_bytes = 0;
        std::size_t alpha_bytes = 0;
        std::size_t beta_bytes = 0;
        std::size_t state_bytes = 0;
        std::size_t ssm_norm_bytes = 0;
        std::size_t gated_out_bytes = 0;

        // Serialize compute on the same entry so multiple threads using the same
        // shape don't trample each other's uploads/downloads on the shared graph.
        std::mutex compute_mutex;

        ~GdnChunkedCacheEntry()
        {
            if (ctx != nullptr)
            {
                ggml_free(ctx);
                ctx = nullptr;
            }
        }
    };

    std::mutex g_gdn_chunked_cache_mutex;
    std::unordered_map<GdnChunkedCacheKey,
                       std::unique_ptr<GdnChunkedCacheEntry>,
                       GdnChunkedCacheKeyHash,
                       GdnChunkedCacheKeyEq> g_gdn_chunked_cache;

    // Register an atexit handler that frees cached GDN graphs/buffers BEFORE the
    // Metal device singleton's static destructor runs. atexit handlers fire in
    // LIFO order interleaved with C++ static destructors, so registering after
    // the Metal device has been initialized guarantees we tear down GPU
    // resources first and avoid `[rsets->data count] == 0` assertions.
    void ensure_gdn_cache_cleanup_registered()
    {
        static std::once_flag flag;
        std::call_once(flag, []() {
            std::atexit([]() {
                std::lock_guard<std::mutex> lk(g_gdn_chunked_cache_mutex);
                g_gdn_chunked_cache.clear();
            });
        });
    }
}

// ---------------------------------------------------------------------------
// TSGgml_GatedDeltaNetChunkedF32
// ---------------------------------------------------------------------------
// Single fused kernel that performs Qwen3.5/Qwen3-Next chunked GatedDeltaNet
// for one layer. Conv1D, the dt_bias/softplus/mul gate computation and the
// sigmoid on beta are all run on CPU upstream. Their outputs are uploaded
// via the alpha / beta staging slots which lets the fused Metal graph skip
// four trivially-vectorisable ops per layer (~8 ops/call after accounting for
// dependency fencing) and removes two constant-tensor uploads.
//
// Inputs (all C# row-major, F32):
//   q, k, v   : [seqLen, H, D]
//   z         : [seqLen, H, D]
//   alpha     : [seqLen, H]              pre-computed gate = a_log * softplus(alpha_raw + dt_bias)
//   beta      : [seqLen, H]              pre-computed beta_sig = sigmoid(beta_raw)
//   state     : [H, D, D]                in-place updated; D is shared
//                                       (function asserts headKDim == headVDim)
//   gated_out : [seqLen, H, D]           output written via copy
//   dt_bias   : [H]                      UNUSED (kept for ABI stability, ignored)
//   a_log     : [H]                      UNUSED (kept for ABI stability, ignored)
//   ssm_norm_w: [D]
//
// chunk_size is the chunked attention chunk (typ. 64).
// eps is the epsilon used for L2Norm and RMSNorm.
//
// The function:
//   1. L2-normalises Q and K, scales Q by 1/sqrt(D).
//   2. Reads gate (alpha) and beta_sig (beta) directly - no device-side add,
//      softplus, mul or sigmoid.
//   3. Applies beta_sig to v/k -> v_beta, k_beta.
//   4. Pads sequence to a multiple of chunk_size, reshapes into chunks.
//   5. Builds the per-chunk decay/causal/identity mask system, runs the
//      triangular solve and combines pre-computed (k @ q) per-chunk attention
//      with cross-chunk recurrent state propagation. qGExp = q * gExp is
//      pre-computed once over the full chunked layout so the per-chunk loop
//      body contains four mul_mat / two broadcast mul / two add ops only.
//   6. Runs RMSNorm and gates by silu(z).
//   7. Writes the output back to gated_out and the updated recurrent state
//      back to the state tensor.
TSG_EXPORT int TSGgml_GatedDeltaNetChunkedF32(
    TensorView3DDesc q_desc,
    TensorView3DDesc k_desc,
    TensorView3DDesc v_desc,
    TensorView3DDesc z_desc,
    TensorView2DDesc alpha_desc,
    TensorView2DDesc beta_desc,
    TensorView3DDesc state_desc,
    TensorView3DDesc gated_out_desc,
    void* dt_bias_data,
    void* a_log_data,
    void* ssm_norm_w_data,
    int chunk_size,
    float eps)
{
    try
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(q_desc, "q") || !validate_desc(k_desc, "k") ||
            !validate_desc(v_desc, "v") || !validate_desc(z_desc, "z") ||
            !validate_desc(alpha_desc, "alpha") || !validate_desc(beta_desc, "beta") ||
            !validate_desc(state_desc, "state") || !validate_desc(gated_out_desc, "gated_out"))
        {
            return 0;
        }

        if (ssm_norm_w_data == nullptr)
        {
            set_last_error("GatedDeltaNetChunked: ssm_norm_w must be non-null.");
            return 0;
        }
        // dt_bias_data and a_log_data are unused in this kernel - gate is pre-computed on
        // the host. We keep the parameters on the C ABI so the existing C# binding keeps
        // compiling; the values are ignored here.
        (void)dt_bias_data;
        (void)a_log_data;

        // Shape sanity. Q/K/V share [seqLen, H, D]; state is [H, D, D].
        const int T  = q_desc.dim0;
        const int H  = q_desc.dim1;
        const int D  = q_desc.dim2;

        if (k_desc.dim0 != T || k_desc.dim1 != H || k_desc.dim2 != D ||
            v_desc.dim0 != T || v_desc.dim1 != H || v_desc.dim2 != D ||
            z_desc.dim0 != T || z_desc.dim1 != H || z_desc.dim2 != D ||
            gated_out_desc.dim0 != T || gated_out_desc.dim1 != H || gated_out_desc.dim2 != D)
        {
            set_last_error("GatedDeltaNetChunked: Q/K/V/Z/output shape mismatch.");
            return 0;
        }
        if (alpha_desc.dim0 != T || alpha_desc.dim1 != H ||
            beta_desc.dim0  != T || beta_desc.dim1  != H)
        {
            set_last_error("GatedDeltaNetChunked: alpha/beta shape mismatch.");
            return 0;
        }
        if (state_desc.dim0 != H || state_desc.dim1 != D || state_desc.dim2 != D)
        {
            set_last_error("GatedDeltaNetChunked: state must be [H, D, D] (chunked path requires headKDim == headVDim).");
            return 0;
        }
        if (chunk_size <= 0 || (chunk_size & (chunk_size - 1)) != 0)
        {
            set_last_error("GatedDeltaNetChunked: chunk_size must be a positive power of two.");
            return 0;
        }
        if (T <= 0)
        {
            // Nothing to do.
            clear_last_error();
            return 1;
        }

        const int cS = chunk_size;
        const int T_padded = ((T + cS - 1) / cS) * cS;
        const int pad = T_padded - T;
        const int nC = T_padded / cS;

        // ----- Cache lookup -----------------------------------------------------
        GdnChunkedCacheKey cache_key{T, H, D, cS, eps};
        GdnChunkedCacheEntry* entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_gdn_chunked_cache_mutex);
            auto it = g_gdn_chunked_cache.find(cache_key);
            if (it != g_gdn_chunked_cache.end())
                entry = it->second.get();
        }

        // ----- Cache miss: build graph + buffer once for this shape -------------
        if (entry == nullptr)
        {
            // Estimate tensor metadata budget. Each tensor uses ~256-384 bytes of
            // metadata; we create roughly (32 + 10 * nC) tensors after hoisting qGExp
            // out of the per-chunk loop and pre-computing gate/beta_sig on the host.
            // Includes slack for the balanced concat tree with ~2 * nC nodes.
            const std::size_t per_tensor_bytes = 384;
            const std::size_t tensor_count_estimate = 256 + static_cast<std::size_t>(20 * nC);
            std::size_t ctx_size = tensor_count_estimate * per_tensor_bytes;
            if (ctx_size < 4 * 1024 * 1024) ctx_size = 4 * 1024 * 1024;

            auto new_entry = std::make_unique<GdnChunkedCacheEntry>();
            new_entry->ctx_buffer = std::make_unique<std::uint8_t[]>(ctx_size);

            ggml_init_params params = {};
            params.mem_size = ctx_size;
            params.mem_buffer = new_entry->ctx_buffer.get();
            params.no_alloc = true;
            new_entry->ctx = ggml_init(params);
            if (new_entry->ctx == nullptr)
            {
                set_last_error("GatedDeltaNetChunked: cached context init failed.");
                return 0;
            }
            ggml_context* ctx = new_entry->ctx;

            // Bind input/output tensors.
            TensorBinding q_bind  = create_standard_binding(ctx, q_desc);
            TensorBinding k_bind  = create_standard_binding(ctx, k_desc);
            TensorBinding v_bind  = create_standard_binding(ctx, v_desc);
            TensorBinding z_bind  = create_standard_binding(ctx, z_desc);
            TensorBinding a_bind  = create_standard_binding(ctx, alpha_desc);
            TensorBinding b_bind  = create_standard_binding(ctx, beta_desc);
            TensorBinding st_bind = create_standard_binding(ctx, state_desc);
            TensorBinding go_bind = create_standard_binding(ctx, gated_out_desc);

            // Constants (per-layer): ssm_norm_w is [D]. dt_bias / a_log are no longer
            // uploaded - gate is pre-computed on the host.
            ggml_tensor* ssm_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);

            // Scratch tensors for masks. ggml_fill needs a contiguous source tensor.
            ggml_tensor* mask_src    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cS, cS, 1, 1);

            // ----------- Build the graph -----------

            // alpha already holds the pre-computed gate and beta already holds sigmoid(beta_raw),
            // both in [seqLen, H] row-major which GGML binds as (H, T). Reshape to the (1, H, T, 1)
            // 4D form expected by the per-chunk broadcasts below - no add/softplus/mul/sigmoid ops.
            ggml_tensor* alpha = a_bind.tensor;       // (H, T) = gate in place
            ggml_tensor* beta  = b_bind.tensor;       // (H, T) = beta_sig in place

            ggml_tensor* gate_4d = ggml_reshape_4d(ctx, alpha, 1, H, T, 1);
            ggml_tensor* beta_4d = ggml_reshape_4d(ctx, beta,  1, H, T, 1);

            // Q/K/V GGML views: ne0=D, ne1=H, ne2=T, ne3=1.
            ggml_tensor* q = q_bind.tensor;
            ggml_tensor* k = k_bind.tensor;
            ggml_tensor* v = v_bind.tensor;
            ggml_tensor* z = z_bind.tensor;

            // L2 normalize Q, K along D (ne0).
            ggml_tensor* q_norm = ggml_l2_norm(ctx, q, eps);
            ggml_tensor* k_norm = ggml_l2_norm(ctx, k, eps);

            // Scale Q by 1/sqrt(D).
            const float qScale = 1.0f / std::sqrt(static_cast<float>(D));
            ggml_tensor* q_scaled = ggml_scale(ctx, q_norm, qScale);

            // Permute Q/K/V to (D, T, H, 1) to lay tokens contiguous along ne1.
            ggml_tensor* q_p = ggml_cont(ctx, ggml_permute(ctx, q_scaled, 0, 2, 1, 3));
            ggml_tensor* k_p = ggml_cont(ctx, ggml_permute(ctx, k_norm,   0, 2, 1, 3));
            ggml_tensor* v_p = ggml_cont(ctx, ggml_permute(ctx, v,        0, 2, 1, 3));

            // gate/beta from (1, H, T, 1) -> (1, T, H, 1)
            ggml_tensor* gate_p = ggml_cont(ctx, ggml_permute(ctx, gate_4d, 0, 2, 1, 3));
            ggml_tensor* beta_p = ggml_cont(ctx, ggml_permute(ctx, beta_4d, 0, 2, 1, 3));

            // Pad along ne1 (T) by pad zeros.
            if (pad > 0)
            {
                q_p    = ggml_pad(ctx, q_p,    0, pad, 0, 0);
                k_p    = ggml_pad(ctx, k_p,    0, pad, 0, 0);
                v_p    = ggml_pad(ctx, v_p,    0, pad, 0, 0);
                gate_p = ggml_pad(ctx, gate_p, 0, pad, 0, 0);
                beta_p = ggml_pad(ctx, beta_p, 0, pad, 0, 0);
            }

            // v_beta, k_beta = v_p * beta_p, k_p * beta_p (broadcast on D).
            ggml_tensor* v_beta_full = ggml_mul(ctx, v_p, beta_p);
            ggml_tensor* k_beta_full = ggml_mul(ctx, k_p, beta_p);

            // Reshape to chunks: (D, cS, nC, H).
            ggml_tensor* q_chunked      = ggml_reshape_4d(ctx, q_p,         D, cS, nC, H);
            ggml_tensor* k_chunked      = ggml_reshape_4d(ctx, k_p,         D, cS, nC, H);
            ggml_tensor* k_beta_chunked = ggml_reshape_4d(ctx, k_beta_full, D, cS, nC, H);
            ggml_tensor* v_beta_chunked = ggml_reshape_4d(ctx, v_beta_full, D, cS, nC, H);
            ggml_tensor* gate_chunked   = ggml_reshape_4d(ctx, gate_p,      1, cS, nC, H);

            // gate_chunked: (1, cS, nC, H) -> permute (1,0,2,3) -> (cS, 1, nC, H), then cumsum.
            ggml_tensor* gate_cs  = ggml_cont(ctx, ggml_permute(ctx, gate_chunked, 1, 0, 2, 3));
            ggml_tensor* g_cumsum = ggml_cumsum(ctx, gate_cs);  // (cS, 1, nC, H)

            // Build the per-chunk constant masks.
            ggml_tensor* mask_ones   = ggml_fill(ctx, mask_src, 1.0f);                       // (cS, cS, 1, 1)
            ggml_tensor* causal_mask = ggml_tri(ctx, mask_ones, GGML_TRI_TYPE_LOWER);        // strict lower
            ggml_tensor* diag_mask   = ggml_tri(ctx, mask_ones, GGML_TRI_TYPE_LOWER_DIAG);   // lower with diag
            ggml_tensor* identity_mask = ggml_sub(ctx, diag_mask, causal_mask);              // diagonal only

            // Decay mask: exp((cumsum_j - cumsum_i)) where j>=i (lower triangle).
            ggml_tensor* gcsJ = ggml_reshape_4d(ctx, g_cumsum, 1, cS, nC, H);
            ggml_tensor* gcsBroadcast = ggml_repeat_4d(ctx, gcsJ, cS, cS, nC, H);
            ggml_tensor* decay_mask_raw = ggml_sub(ctx, gcsBroadcast, g_cumsum);
            ggml_tensor* decay_mask = ggml_mul(ctx, decay_mask_raw, diag_mask);
            decay_mask = ggml_exp(ctx, decay_mask);
            decay_mask = ggml_mul(ctx, decay_mask, diag_mask);

            // attn_init = -(k @ k_beta^T) * decay_mask  with strict lower mask applied.
            ggml_tensor* k_kbeta   = ggml_mul_mat(ctx, k_chunked, k_beta_chunked); // (cS, cS, nC, H)
            ggml_tensor* k_decay   = ggml_mul(ctx, k_kbeta, decay_mask);
            ggml_tensor* attn_init = ggml_mul(ctx, ggml_neg(ctx, k_decay), causal_mask);

            // Triangular solve: (I - attn_lower) X = attn_init.
            ggml_tensor* attn_lower = ggml_mul(ctx, attn_init, causal_mask);
            ggml_tensor* lhs        = ggml_add(ctx, ggml_neg(ctx, attn_lower), identity_mask);
            ggml_tensor* attn_solved = ggml_solve_tri(ctx, lhs, attn_init, true, true, false);
            ggml_tensor* attn_lower2 = ggml_mul(ctx, attn_solved, causal_mask);
            ggml_tensor* attn        = ggml_add(ctx, attn_lower2, identity_mask);            // (cS, cS, nC, H)

            // v_new = mulmat(v_beta^T, attn) -> (D, cS, nC, H) (D=headVDim under D == K).
            ggml_tensor* vBetaT = ggml_cont(ctx, ggml_permute(ctx, v_beta_chunked, 1, 0, 2, 3));
            ggml_tensor* v_new  = ggml_mul_mat(ctx, vBetaT, attn);                          // (D, cS, nC, H)

            // gExp = exp(g_cumsum_T) where g_cumsum_T is (1, cS, nC, H).
            ggml_tensor* gCumsumT = ggml_cont(ctx, ggml_permute(ctx, g_cumsum, 1, 0, 2, 3));
            ggml_tensor* gExp     = ggml_exp(ctx, gCumsumT);                                 // (1, cS, nC, H)

            // kBetaGExp = k_beta * gExp (broadcast on D).
            ggml_tensor* kBetaGExp  = ggml_mul(ctx, k_beta_chunked, gExp);
            ggml_tensor* kBetaGExpT = ggml_cont(ctx, ggml_permute(ctx, kBetaGExp, 1, 0, 2, 3));
            ggml_tensor* kCumdecay  = ggml_mul_mat(ctx, attn, kBetaGExpT);                  // (cS, D, nC, H)
            kCumdecay = ggml_cont(ctx, ggml_permute(ctx, kCumdecay, 1, 0, 2, 3));            // (D, cS, nC, H)

            // attn_kq = (k @ q) * decay * diag.
            ggml_tensor* attn_kq = ggml_mul_mat(ctx, k_chunked, q_chunked);                  // (cS, cS, nC, H)
            attn_kq = ggml_mul(ctx, attn_kq, decay_mask);
            attn_kq = ggml_mul(ctx, attn_kq, diag_mask);

            // gLast = view of last cumsum slot per chunk: (1, 1, nC, H).
            ggml_tensor* gLast = ggml_view_4d(ctx, g_cumsum, 1, 1, nC, H,
                g_cumsum->nb[1], g_cumsum->nb[2], g_cumsum->nb[3],
                static_cast<std::size_t>(cS - 1) * sizeof(float));
            gLast = ggml_cont(ctx, gLast);                                                   // (1, 1, nC, H)
            ggml_tensor* gLastExp = ggml_exp(ctx, gLast);

            // gDiff = gLast - g_cumsum, exp, reshape to (1, cS, nC, H).
            ggml_tensor* gDiff = ggml_add(ctx, ggml_neg(ctx, g_cumsum), gLast);             // (cS, 1, nC, H)
            ggml_tensor* gDiffExp   = ggml_exp(ctx, gDiff);
            ggml_tensor* gDiffExpRe = ggml_reshape_4d(ctx, gDiffExp, 1, cS, nC, H);

            // keyGDiff = k_chunked * gDiffExpRe (broadcast on D); transpose to (cS, D, nC, H).
            ggml_tensor* keyGDiff  = ggml_mul(ctx, k_chunked, gDiffExpRe);
            ggml_tensor* keyGDiffT = ggml_cont(ctx, ggml_permute(ctx, keyGDiff, 1, 0, 2, 3));

            // vT = transpose v_new for chunked mulmat with attn.
            ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, v_new, 1, 0, 2, 3));         // (cS, D, nC, H)

            // Hoist `q * gExp` out of the per-chunk loop: computing it over the whole
            // (D, cS, nC, H) layout once is one Metal dispatch instead of `nC` tiny ones.
            ggml_tensor* qTimesGExp = ggml_mul(ctx, q_chunked, gExp);                        // (D, cS, nC, H)

            // stateT layout: ne[0]=k_dim, ne[1]=v_dim, ne[2]=1, ne[3]=H (matches Ollama's stateT).
            // state binding from C# [H, V, K] row-major produces a GGML view with
            // ne=(K, V, H, 1) where ne[0]=k_dim (innermost), ne[1]=v_dim. That layout
            // already matches Ollama's stateT semantics, so we only need to reshape
            // (D, D, H, 1) -> (D, D, 1, H) without permuting axes 0 and 1.
            ggml_tensor* state_in = st_bind.tensor;                                          // (D, D, H, 1)
            ggml_tensor* stateT = ggml_reshape_4d(ctx, state_in, D, D, 1, H);                // (D, D, 1, H)

            // Per-chunk recurrence.
            std::vector<ggml_tensor*> chunk_outputs(nC);
            auto chunk_view = [&](ggml_tensor* src, int c) -> ggml_tensor* {
                return ggml_view_4d(ctx, src, src->ne[0], src->ne[1], 1, src->ne[3],
                    src->nb[1], src->nb[2], src->nb[3],
                    static_cast<std::size_t>(c) * src->nb[2]);
            };

            for (int c = 0; c < nC; c++)
            {
                ggml_tensor* vTChunk        = chunk_view(vT,          c);  // (cS, D, 1, H)
                ggml_tensor* qGExpChunk     = chunk_view(qTimesGExp,  c);  // (D, cS, 1, H) - pre-hoisted
                ggml_tensor* kCumdecayChunk = chunk_view(kCumdecay,   c);  // (D, cS, 1, H)
                ggml_tensor* attnChunk      = chunk_view(attn_kq,     c);  // (cS, cS, 1, H)

                // v'_t = mulmat(kCumdecay, stateT) -> (cS, D, 1, H)
                ggml_tensor* vTPrime = ggml_mul_mat(ctx, kCumdecayChunk, stateT);

                // v_t_new = vT - v'_t
                ggml_tensor* vTNew = ggml_sub(ctx, vTChunk, vTPrime);

                // attnInter = mulmat(stateT, qGExp) -> (D, cS, 1, H).
                ggml_tensor* attnInter = ggml_mul_mat(ctx, stateT, qGExpChunk);

                // vAttn = mulmat(vTNew, attnChunk) -> (D, cS, 1, H)
                ggml_tensor* vAttn = ggml_mul_mat(ctx, vTNew, attnChunk);

                chunk_outputs[c] = ggml_add(ctx, attnInter, vAttn);

                // State update.
                ggml_tensor* gExpLastChunk = chunk_view(gLastExp,  c);  // (1, 1, 1, H)
                ggml_tensor* kGDiffChunkT  = chunk_view(keyGDiffT, c);  // (cS, D, 1, H)
                ggml_tensor* kgdMulVNew    = ggml_mul_mat(ctx, kGDiffChunkT, vTNew); // (D, D, 1, H)
                stateT = ggml_mul(ctx, stateT, gExpLastChunk);
                stateT = ggml_add(ctx, stateT, kgdMulVNew);
            }

            // Balanced concat tree along ne2 (chunk axis).
            std::vector<ggml_tensor*> level = std::move(chunk_outputs);
            while (level.size() > 1)
            {
                std::vector<ggml_tensor*> next;
                next.reserve((level.size() + 1) / 2);
                for (std::size_t i = 0; i + 1 < level.size(); i += 2)
                {
                    next.push_back(ggml_concat(ctx, level[i], level[i + 1], 2));
                }
                if (level.size() % 2 == 1)
                {
                    next.push_back(level.back());
                }
                level = std::move(next);
            }
            ggml_tensor* concat_result = level[0];                                           // (D, cS, nC, H)

            // Reshape to (D, T_padded, H, 1) and slice off padding.
            ggml_tensor* core_attn = ggml_reshape_4d(ctx, concat_result, D, cS * nC, H, 1);
            if (pad > 0)
            {
                ggml_tensor* sliced = ggml_view_4d(ctx, core_attn, D, T, H, 1,
                    core_attn->nb[1], core_attn->nb[2], core_attn->nb[3], 0);
                core_attn = ggml_cont(ctx, sliced);
            }

            // RMSNorm + per-D weight.
            ggml_tensor* attn_rms = ggml_rms_norm(ctx, core_attn, eps);
            attn_rms = ggml_mul(ctx, attn_rms, ssm_norm_t);

            // z permute (D, H, T, 1) -> (D, T, H, 1), silu, multiply.
            ggml_tensor* z_p     = ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3));
            ggml_tensor* z_silu  = ggml_silu(ctx, z_p);
            ggml_tensor* gated   = ggml_mul(ctx, attn_rms, z_silu);                          // (D, T, H, 1)

            // Permute back to (D, H, T, 1) and copy into output binding.
            ggml_tensor* gated_out = ggml_cont(ctx, ggml_permute(ctx, gated, 0, 2, 1, 3));
            ggml_tensor* out_cpy   = ggml_cpy(ctx, gated_out, go_bind.tensor);

            // Write the updated state back. stateT (D=k, D=v, 1, H) is already in the same
            // semantic layout as state_in, just reshape (D, D, 1, H) -> (D, D, H, 1) and copy.
            ggml_tensor* stateT_cont = ggml_cont(ctx, stateT);
            ggml_tensor* state_out_4d = ggml_reshape_4d(ctx, stateT_cont, D, D, H, 1);
            ggml_tensor* state_cpy = ggml_cpy(ctx, state_out_4d, state_in);

            ggml_set_output(out_cpy);
            ggml_set_output(state_cpy);

            new_entry->graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);
            ggml_build_forward_expand(new_entry->graph, out_cpy);
            ggml_build_forward_expand(new_entry->graph, state_cpy);

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (!buffer.value)
            {
                set_last_error("GatedDeltaNetChunked: buffer alloc failed.");
                return 0;
            }
            new_entry->buffer = std::move(buffer);

            // Persist storage tensor handles + sizes for the per-call upload/download.
            new_entry->q_storage = q_bind.storage;
            new_entry->k_storage = k_bind.storage;
            new_entry->v_storage = v_bind.storage;
            new_entry->z_storage = z_bind.storage;
            new_entry->alpha_storage = a_bind.storage;
            new_entry->beta_storage  = b_bind.storage;
            new_entry->state_storage = st_bind.storage;
            new_entry->ssm_norm_storage = ssm_norm_t;
            new_entry->gated_out_storage = go_bind.storage;

            new_entry->q_bytes = q_bind.raw_bytes;
            new_entry->k_bytes = k_bind.raw_bytes;
            new_entry->v_bytes = v_bind.raw_bytes;
            new_entry->z_bytes = z_bind.raw_bytes;
            new_entry->alpha_bytes = a_bind.raw_bytes;
            new_entry->beta_bytes  = b_bind.raw_bytes;
            new_entry->state_bytes = st_bind.raw_bytes;
            new_entry->ssm_norm_bytes = static_cast<std::size_t>(D) * sizeof(float);
            new_entry->gated_out_bytes = go_bind.raw_bytes;

            // Publish to the cache. If another thread raced us in here we keep the
            // first published entry (deterministic) and let ours go out of scope.
            std::lock_guard<std::mutex> lk(g_gdn_chunked_cache_mutex);
            auto [it, inserted] = g_gdn_chunked_cache.emplace(cache_key, std::move(new_entry));
            entry = it->second.get();
            // Register the atexit cleanup hook on first cache insertion. By now the
            // Metal device singleton is fully initialized (we just allocated a
            // buffer on it), so our atexit fires before its destructor.
            ensure_gdn_cache_cleanup_registered();
        }

        // ----- Cache hit path: rebind data, run, and copy outputs back ----------
        {
            std::lock_guard<std::mutex> entry_lk(entry->compute_mutex);

            ggml_backend_tensor_set(entry->q_storage,    q_desc.data,     0, entry->q_bytes);
            ggml_backend_tensor_set(entry->k_storage,    k_desc.data,     0, entry->k_bytes);
            ggml_backend_tensor_set(entry->v_storage,    v_desc.data,     0, entry->v_bytes);
            ggml_backend_tensor_set(entry->z_storage,    z_desc.data,     0, entry->z_bytes);
            ggml_backend_tensor_set(entry->alpha_storage, alpha_desc.data, 0, entry->alpha_bytes);
            ggml_backend_tensor_set(entry->beta_storage,  beta_desc.data,  0, entry->beta_bytes);
            ggml_backend_tensor_set(entry->state_storage, state_desc.data, 0, entry->state_bytes);
            ggml_backend_tensor_set(entry->ssm_norm_storage, ssm_norm_w_data, 0, entry->ssm_norm_bytes);

            ggml_status status = ggml_backend_graph_compute(g_backend, entry->graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("GatedDeltaNetChunked: graph compute failed.");
                return 0;
            }
            ggml_backend_synchronize(g_backend);

            ggml_backend_tensor_get(entry->gated_out_storage, gated_out_desc.data, 0, entry->gated_out_bytes);
            ggml_backend_tensor_get(entry->state_storage,     state_desc.data,     0, entry->state_bytes);
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
        set_last_error("Unknown error in GatedDeltaNetChunked.");
        return 0;
    }
}
