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

            // Drain any pending async work so the CPU memcpys below don't race
            // with in-flight GPU writes targeting q/k/v/z/etc. host buffers.
            host_read_barrier();

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

// ---------------------------------------------------------------------------
// TSGgml_GatedDeltaNetBatchedStepF32
// ---------------------------------------------------------------------------
// vLLM-style batched per-token GDN compute. Replaces N separate calls to
// the C# per-token loop (one per active sequence in a batched decode step)
// with one native dispatch that walks every (seq, token) pair in a tight
// loop, indexing into per-sequence state via the descriptors.
//
// API mirrors vLLM's fused_sigmoid_gating_delta_rule_update_cpu:
//   - packed_batched: [num_tokens, packed_dim] row-major, layout per row is
//       [qkv (qkv_dim) | z (z_dim) | beta (num_v_heads) | alpha (num_v_heads)]
//   - seqs[i] gives seq_start (row offset), seq_len, and pointers to that
//     sequence's conv state ring buffer ((conv_kernel-1) * qkv_dim floats)
//     and ssm state (num_v_heads * head_v_dim * head_k_dim floats). The
//     conv_write_idx field is updated in place.
//   - gated_out: [num_tokens, ssm_d_inner] row-major output.
//
// The math is a faithful port of Qwen35Model.GatedDeltaNet.GatedDeltaNetStep
// in C#: ring-buffered Conv1D + SiLU, optional GQA expand (when num_k_heads
// != num_v_heads), per-head L2-norm + Q scale, per-head delta-rule scan
// (state *= exp(gate); delta = (v - state·k) * beta; state += k⊗delta;
// core = state·q), then RMSNorm + ssm_norm * SiLU(z) gating.
//
// Currently scalar; SIMD/parallelism can come later (the math is correct
// regardless of vectorisation). Each sequence is independent so this is
// trivially parallelisable across seqs.
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace
{
    struct GdnBatchedSeqDesc
    {
        int32_t seq_start;
        int32_t seq_len;
        int32_t conv_write_idx;  // in/out
        int32_t pad;
        void*   conv_state;      // float* — [conv_dim * qkv_dim], conv_dim = conv_kernel - 1
        void*   ssm_state;       // float* — [num_v_heads, head_v_dim, head_k_dim]
    };

    static inline float softplus_scalar(float x)
    {
        // Numerically stable softplus matching the C# SoftplusScalar helper.
        if (x > 20.0f) return x;
        if (x < -20.0f) return std::exp(x);
        return std::log1p(std::exp(x));
    }

    static inline float sigmoid_scalar(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static inline float silu_scalar(float x)
    {
        return x * sigmoid_scalar(x);
    }

#if defined(__ARM_NEON)
    static inline float vec_dot(const float* a, const float* b, int n)
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; i <= n - 8; i += 8)
        {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        }
        float32x4_t acc = vaddq_f32(acc0, acc1);
        for (; i <= n - 4; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
        float s = vaddvq_f32(acc);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    static inline void vec_scale(float* a, float s, int n)
    {
        int i = 0;
        float32x4_t vs = vdupq_n_f32(s);
        for (; i <= n - 8; i += 8)
        {
            vst1q_f32(a + i,     vmulq_f32(vld1q_f32(a + i),     vs));
            vst1q_f32(a + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vs));
        }
        for (; i <= n - 4; i += 4)
            vst1q_f32(a + i, vmulq_f32(vld1q_f32(a + i), vs));
        for (; i < n; i++) a[i] *= s;
    }

    static inline void vec_scale_add(float* a, const float* b, float s, int n)
    {
        int i = 0;
        float32x4_t vs = vdupq_n_f32(s);
        for (; i <= n - 8; i += 8)
        {
            vst1q_f32(a + i,     vfmaq_f32(vld1q_f32(a + i),     vld1q_f32(b + i),     vs));
            vst1q_f32(a + i + 4, vfmaq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4), vs));
        }
        for (; i <= n - 4; i += 4)
            vst1q_f32(a + i, vfmaq_f32(vld1q_f32(a + i), vld1q_f32(b + i), vs));
        for (; i < n; i++) a[i] += b[i] * s;
    }

    static inline float vec_sumsq(const float* a, int n)
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; i <= n - 8; i += 8)
        {
            float32x4_t v0 = vld1q_f32(a + i);
            float32x4_t v1 = vld1q_f32(a + i + 4);
            acc0 = vfmaq_f32(acc0, v0, v0);
            acc1 = vfmaq_f32(acc1, v1, v1);
        }
        float32x4_t acc = vaddq_f32(acc0, acc1);
        for (; i <= n - 4; i += 4)
        {
            float32x4_t v = vld1q_f32(a + i);
            acc = vfmaq_f32(acc, v, v);
        }
        float s = vaddvq_f32(acc);
        for (; i < n; i++) s += a[i] * a[i];
        return s;
    }

    // Element-wise multiply-into: dst[i] = a[i] * b[i]
    static inline void vec_mul_into(float* dst, const float* a, const float* b, int n)
    {
        int i = 0;
        for (; i <= n - 8; i += 8)
        {
            vst1q_f32(dst + i,     vmulq_f32(vld1q_f32(a + i),     vld1q_f32(b + i)));
            vst1q_f32(dst + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
        }
        for (; i <= n - 4; i += 4)
            vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        for (; i < n; i++) dst[i] = a[i] * b[i];
    }

    // Element-wise FMA: dst[i] += a[i] * b[i]
    static inline void vec_fma(float* dst, const float* a, const float* b, int n)
    {
        int i = 0;
        for (; i <= n - 8; i += 8)
        {
            vst1q_f32(dst + i,     vfmaq_f32(vld1q_f32(dst + i),     vld1q_f32(a + i),     vld1q_f32(b + i)));
            vst1q_f32(dst + i + 4, vfmaq_f32(vld1q_f32(dst + i + 4), vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
        }
        for (; i <= n - 4; i += 4)
            vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(a + i), vld1q_f32(b + i)));
        for (; i < n; i++) dst[i] += a[i] * b[i];
    }
#else
    static inline float vec_dot(const float* a, const float* b, int n)
    {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }
    static inline void vec_scale(float* a, float s, int n)
    {
        for (int i = 0; i < n; i++) a[i] *= s;
    }
    static inline void vec_scale_add(float* a, const float* b, float s, int n)
    {
        for (int i = 0; i < n; i++) a[i] += b[i] * s;
    }
    static inline float vec_sumsq(const float* a, int n)
    {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += a[i] * a[i];
        return s;
    }
    static inline void vec_mul_into(float* dst, const float* a, const float* b, int n)
    {
        for (int i = 0; i < n; i++) dst[i] = a[i] * b[i];
    }
    static inline void vec_fma(float* dst, const float* a, const float* b, int n)
    {
        for (int i = 0; i < n; i++) dst[i] += a[i] * b[i];
    }
#endif

    static inline void l2_normalize_in_place(float* x, int n)
    {
        float ss = vec_sumsq(x, n);
        float inv = 1.0f / std::sqrt(ss + 1e-12f);
        vec_scale(x, inv, n);
    }
}

TSG_EXPORT int TSGgml_GatedDeltaNetBatchedStepF32(
    int32_t                num_seqs,
    GdnBatchedSeqDesc*     seqs,
    int32_t                num_tokens,
    const float*           packed_batched,
    int32_t                packed_dim,
    int32_t                qkv_dim,
    int32_t                qk_dim,
    int32_t                v_dim,
    int32_t                z_dim,
    int32_t                num_k_heads,
    int32_t                num_v_heads,
    int32_t                head_k_dim,
    int32_t                head_v_dim,
    int32_t                conv_kernel,
    int32_t                ssm_d_inner,
    const float*           conv_wt,
    const float*           dt_bias,
    const float*           a_log,
    const float*           ssm_norm_w,
    float                  eps,
    float*                 gated_out)
{
    try
    {
        if (num_seqs <= 0 || seqs == nullptr ||
            packed_batched == nullptr || gated_out == nullptr ||
            conv_wt == nullptr || dt_bias == nullptr ||
            a_log == nullptr || ssm_norm_w == nullptr)
        {
            set_last_error("GatedDeltaNetBatchedStep: null arg or num_seqs<=0.");
            return 0;
        }

        const int conv_dim = conv_kernel - 1;
        const int state_per_head = head_v_dim * head_k_dim;
        const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_v_dim));

        // Scratch per call. Sized for the largest contiguous step we do; small
        // enough to live on the stack-ish heap. Using std::vector keeps it simple.
        std::vector<float> conv_out(qkv_dim);
        std::vector<float> q_active(num_v_heads * head_k_dim);
        std::vector<float> k_active(num_v_heads * head_k_dim);
        std::vector<float> delta_buf(num_v_heads * head_v_dim);
        std::vector<float> core_buf(num_v_heads * head_v_dim);

        for (int si = 0; si < num_seqs; si++)
        {
            GdnBatchedSeqDesc& sd = seqs[si];
            float* conv_state = static_cast<float*>(sd.conv_state);
            float* ssm_state  = static_cast<float*>(sd.ssm_state);
            int    write_idx  = sd.conv_write_idx;
            const int seq_start = sd.seq_start;
            const int seq_len   = sd.seq_len;
            if (seq_len <= 0) continue;
            if (conv_state == nullptr || ssm_state == nullptr)
            {
                set_last_error("GatedDeltaNetBatchedStep: per-seq state pointer is null.");
                return 0;
            }

            for (int t = 0; t < seq_len; t++)
            {
                const float* row = packed_batched + static_cast<long>(seq_start + t) * packed_dim;
                const float* qkv_in   = row;
                const float* z_ptr    = row + qkv_dim;
                const float* beta_ptr = z_ptr + z_dim;
                const float* alpha_ptr= beta_ptr + num_v_heads;

                // --- Conv1D step ---
                // tap 0..conv_dim-1: from ring state; tap conv_dim: current input.
                if (conv_dim > 0)
                {
                    int slot = write_idx;
                    const float* sp = conv_state + static_cast<long>(slot) * qkv_dim;
                    const float* wp = conv_wt; // tap 0
                    vec_mul_into(conv_out.data(), sp, wp, qkv_dim);
                    for (int ki = 1; ki < conv_dim; ki++)
                    {
                        slot = (write_idx + ki) % conv_dim;
                        sp = conv_state + static_cast<long>(slot) * qkv_dim;
                        wp = conv_wt + static_cast<long>(ki) * qkv_dim;
                        vec_fma(conv_out.data(), sp, wp, qkv_dim);
                    }
                    const float* wp_now = conv_wt + static_cast<long>(conv_dim) * qkv_dim;
                    vec_fma(conv_out.data(), qkv_in, wp_now, qkv_dim);
                }
                else
                {
                    vec_mul_into(conv_out.data(), qkv_in, conv_wt, qkv_dim);
                }
                // SiLU in place. Scalar — fine; qkv_dim is small per token.
                for (int ch = 0; ch < qkv_dim; ch++) conv_out[ch] = silu_scalar(conv_out[ch]);

                // Update ring state with current input (matches C# behavior:
                // GatedDeltaNetStep writes qkvPtr to state[writeIdx] AFTER
                // computing the conv output that already used that input as
                // the "current" tap above).
                if (conv_dim > 0)
                {
                    float* dst = conv_state + static_cast<long>(write_idx) * qkv_dim;
                    std::memcpy(dst, qkv_in, qkv_dim * sizeof(float));
                    write_idx = (write_idx + 1) % conv_dim;
                }

                // Split conv_out into Q (qk_dim), K (qk_dim), V (v_dim).
                const float* q_raw = conv_out.data();
                const float* k_raw = conv_out.data() + qk_dim;
                const float* v_raw = conv_out.data() + 2 * qk_dim;

                // Expand Q/K from KV-heads to V-heads if GQA differs.
                float* q_active_ptr;
                float* k_active_ptr;
                if (num_k_heads == num_v_heads)
                {
                    // No expansion needed; use raw pointers but we still need
                    // mutable arrays for L2-norm + scale. Copy into the active
                    // buffers (cheap; qk_dim is small per token).
                    std::memcpy(q_active.data(), q_raw, qk_dim * sizeof(float));
                    std::memcpy(k_active.data(), k_raw, qk_dim * sizeof(float));
                }
                else
                {
                    for (int h = 0; h < num_v_heads; h++)
                    {
                        int src = h % num_k_heads;
                        std::memcpy(q_active.data() + h * head_k_dim,
                                    q_raw + src * head_k_dim,
                                    head_k_dim * sizeof(float));
                        std::memcpy(k_active.data() + h * head_k_dim,
                                    k_raw + src * head_k_dim,
                                    head_k_dim * sizeof(float));
                    }
                }
                q_active_ptr = q_active.data();
                k_active_ptr = k_active.data();

                // L2 normalize per head, then scale Q by 1/sqrt(head_v_dim).
                for (int h = 0; h < num_v_heads; h++)
                {
                    l2_normalize_in_place(q_active_ptr + h * head_k_dim, head_k_dim);
                    l2_normalize_in_place(k_active_ptr + h * head_k_dim, head_k_dim);
                }
                vec_scale(q_active_ptr, q_scale, num_v_heads * head_k_dim);

                // Per-head delta-rule scan. Heads are fully independent within a
                // single token (each has its own state slice), so we run them in
                // parallel via GCD on macOS — matching the C# Parallel.For path
                // in Qwen35Model.GatedDeltaNet.GatedDeltaNetStep, which is what
                // makes the legacy per-seq loop fast enough to be competitive.
                float* gated_row = gated_out + static_cast<long>(seq_start + t) * ssm_d_inner;
                auto head_body = [&](int h) {
                    float* state_h  = ssm_state + static_cast<long>(h) * state_per_head;
                    const float* qh = q_active_ptr + h * head_k_dim;
                    const float* kh = k_active_ptr + h * head_k_dim;
                    const float* vh = v_raw + h * head_v_dim;
                    const float* zh = z_ptr + h * head_v_dim;
                    float* delta_h  = delta_buf.data() + h * head_v_dim;
                    float* core_h   = core_buf.data() + h * head_v_dim;
                    float* gated_h  = gated_row + h * head_v_dim;

                    float alpha_biased = alpha_ptr[h] + dt_bias[h];
                    float gate_h = softplus_scalar(alpha_biased) * a_log[h];
                    vec_scale(state_h, std::exp(gate_h), state_per_head);

                    float beta_h = sigmoid_scalar(beta_ptr[h]);
                    for (int row = 0; row < head_v_dim; row++)
                    {
                        float kv_mem = vec_dot(state_h + row * head_k_dim, kh, head_k_dim);
                        delta_h[row] = (vh[row] - kv_mem) * beta_h;
                    }
                    for (int row = 0; row < head_v_dim; row++)
                    {
                        float* state_row = state_h + row * head_k_dim;
                        vec_scale_add(state_row, kh, delta_h[row], head_k_dim);
                        core_h[row] = vec_dot(state_row, qh, head_k_dim);
                    }
                    float rms_inv = 1.0f / std::sqrt((vec_sumsq(core_h, head_v_dim) / head_v_dim) + eps);
                    for (int i = 0; i < head_v_dim; i++)
                        gated_h[i] = core_h[i] * rms_inv * ssm_norm_w[i] * silu_scalar(zh[i]);
                };
#if defined(__APPLE__)
                if (num_v_heads >= 4)
                {
                    dispatch_apply(static_cast<size_t>(num_v_heads),
                                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                                   ^(size_t h) { head_body(static_cast<int>(h)); });
                }
                else
                {
                    for (int h = 0; h < num_v_heads; h++) head_body(h);
                }
#else
                for (int h = 0; h < num_v_heads; h++) head_body(h);
#endif
            }

            sd.conv_write_idx = write_idx;
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
        set_last_error("Unknown error in GatedDeltaNetBatchedStep.");
        return 0;
    }
}
