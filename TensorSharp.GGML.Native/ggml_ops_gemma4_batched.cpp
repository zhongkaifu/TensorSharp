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
#include <chrono>
#include <cstdio>

using namespace tsg;

// PERSIST / CUDA-graph capture pool for the token-batched dense decode
// (TSGgml_Gemma4ModelDecodeBatched). Analogue of the single-stream g_g4dc_pool,
// but each entry's identity is the SET of N per-request KV caches (sig_kc, in the
// C#-canonicalised request-id order) + N + per-layer padded window. A recurring
// concurrent request-set replays its captured graph (set_rows KV write + fixed
// padded window + F16 mask inputs => identical topology + stable addresses) so
// ggml-cuda's CUDA-graph capture engages. Dropped on prefill / KV reset
// (TSGgml_Gemma4ResetBatchedDecodeCache) since those move device addresses.
namespace
{
    constexpr int kG4BatchPersistKvStride = 256;

    struct G4BatchedDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_gallocr_t galloc = nullptr;      // dedicated packed allocator (MoE batched: VRAM-frugal capture)
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* ple_ids = nullptr;       // nullable: in-kernel PLE gather (I32 [n_seqs] token ids)
        ggml_tensor* ple_input = nullptr;     // nullable: uploaded PLE (F32 [num_layers*ple_dim, n_seqs])
        int ple_dim = 0;
        bool ple_gather = false;
        std::vector<ggml_tensor*> kv_index;   // [num_layers * n_seqs] I64 set_rows write rows (null for shared layers)
        std::vector<ggml_tensor*> attn_mask;  // [num_layers] F16 [win,1,1,n_seqs] (null for shared layers)
        std::vector<int> layer_window;        // [num_layers]
        const void* sig_disc = nullptr;
        std::vector<const void*> sig_kc;      // [n_seqs] layer-0 K cache ptrs (canonical order)
        int num_layers = 0, hidden_size = 0, n_seqs = 0, vocab = 0;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = pos_tensor = logits_out = ple_ids = ple_input = nullptr;
            ple_dim = 0; ple_gather = false;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = nullptr; sig_kc.clear();
            num_layers = hidden_size = n_seqs = vocab = 0;
        }
    };

    constexpr int kG4BatchedMaxCaches = 4;
    struct G4BatchedDecodeCachePool
    {
        G4BatchedDecodeCache entries[kG4BatchedMaxCaches];
        std::uint64_t used[kG4BatchedMaxCaches] = {};
        std::uint64_t clock = 0;

        G4BatchedDecodeCache* find(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        G4BatchedDecodeCache& claim(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4BatchedMaxCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    G4BatchedDecodeCachePool g_g4batched_pool;
}

// ============================================================================
// TRUE TOKEN-BATCHED dense decode (N concurrent sequences, one token each, in
// ONE ggml graph + ONE compute buffer). This is the llama-parity concurrency
// path: where TSGgml_Gemma4ModelDecode (ggml_ops_gemma4_decode.cpp) decodes ONE token and the engine
// round-robins N serial calls for N concurrent requests (N weight loads ->
// aggregate ~= single-stream), this kernel processes all N decode tokens
// together so every weight is loaded ONCE and applied to N tokens. Decode is
// memory-bandwidth bound, so that amortisation is the win (and one compute
// buffer instead of N fixes the per-request-buffer VRAM blowup).
//
// Each sequence owns its OWN KV cache (the per-request holders the C# engine
// manages); k_cache_arr / v_cache_arr are sized num_layers * n_seqs, indexed
// [layer * n_seqs + seq]. positions[seq] is each sequence's current length.
// Hidden in/out and logits are packed column-major [.., n_seqs].
//
// Scope: DENSE only (no MoE; the MoE sibling is TSGgml_Gemma4MoEModelDecodeBatched),
// requires the folded lm_head. The `Ex` entry adds what the E2B/E4B family needs
// (see TSGgml_Gemma4BatchedDecodeCapabilities):
//   * Per-Layer Embedding (PLE): per-row PLE gathered IN-KERNEL from the resident
//     quantized per_layer_token_embd table via get_rows over the N token ids
//     (+ the hidden projection / norm / combine exactly as the single-token kernel
//     and C#'s ComputePLE), or uploaded as ple_data [n_seqs][num_layers*ple_dim].
//   * KV-donor (shared) layers: kv_source_arr[l] != l runs the Q-only projection
//     and attends the donor layer's per-sequence window (no K/V write), exactly
//     like the single-token kernel.
//   * SWA wrap: a local layer whose sequence exceeds its ring writes at
//     pos % cache_size and reads the whole ring flat (decode softmax is
//     permutation-invariant over keys, so the rotation is harmless). Global
//     (linear) layers must still fit their cache; the caller grows them.
// The C# caller (Gemma4Model.TryForwardBatchedFusedDecode) enforces the rest
// and otherwise falls back to the round-robin per-sequence path.
//
// Attention: each sequence runs the solo-shaped single-row flash_attn_ext over a
// direct view of ITS OWN cache window (see step 5+6 below); projections, PLE and
// the FFN are token-batched over [hidden, n_seqs].
//
// CUDA-graph capture (persist): every per-step input is a ggml INPUT tensor
// written with ggml_backend_tensor_set (hidden, positions, per-(layer,seq) I64
// set_rows rows, per-layer F16 masks, PLE token ids / uploaded PLE) and the graph
// lives in its own no_alloc context + own-slot buffer (never a gallocr slot
// reuse), so a recurring request set replays a stable topology at stable
// addresses.
// ============================================================================
namespace
{
    constexpr int kG4BatchedCapPle = 1;
    constexpr int kG4BatchedCapKvDonor = 2;
    constexpr int kG4BatchedCapSwaWrap = 4;
}

// Capability bitmask of TSGgml_Gemma4ModelDecodeBatchedEx. The managed caller
// probes this (EntryPointNotFound on an older native build => 0) and keeps its
// PLE / KV-donor / SWA-wrap gates for whatever bit is missing.
TSG_EXPORT int TSGgml_Gemma4BatchedDecodeCapabilities()
{
    return kG4BatchedCapPle | kG4BatchedCapKvDonor | kG4BatchedCapSwaWrap;
}

TSG_EXPORT int TSGgml_Gemma4ModelDecodeBatchedEx(
    float* hidden_data, int hidden_size, int num_layers, int n_seqs,
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-(layer,seq) KV caches: k_cache_arr[layer * n_seqs + seq].
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads,
    const int* positions,           // [n_seqs]
    float eps, int sliding_window,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    // Separate K/V projection weights for mixed-quant layers (null => fused qkv).
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Folded final-norm + lm_head (required for this kernel).
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap,
    // ---- Ex additions (all nullable; null / 0 reproduces the v1 kernel) ----
    // KV-donor map: kv_source_arr[l] is the layer whose cache layer l attends
    // (== l for a layer with its own K/V). null => every layer owns its cache.
    const int* kv_source_arr,
    // Uploaded PLE: ple_data is [n_seqs][num_layers * ple_dim] F32 (row s = seq
    // s, C#'s ComputePLE layout), ignored when the in-kernel gather is active.
    const float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr,
    // In-kernel PLE gather: the resident quantized per_layer_token_embd table +
    // the N token ids (one per sequence, canonical order), and optionally the
    // quantized per_layer_model_proj + its F32 norm (token-embedding-only PLE
    // when null). Mirrors TSGgml_Gemma4ModelDecode / ModelVerify.
    const void* ple_token_embd_data, int ple_token_embd_type,
    std::int64_t ple_token_embd_ne0, std::int64_t ple_token_embd_ne1, std::int64_t ple_token_embd_bytes,
    const int* ple_token_ids,
    const void* ple_model_proj_data, int ple_model_proj_type,
    std::int64_t ple_model_proj_ne0, std::int64_t ple_model_proj_ne1, std::int64_t ple_model_proj_bytes,
    const float* ple_model_proj_norm_data)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        {
            set_last_error("Gemma4 batched decode: invalid arguments.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Gemma4 batched decode: folded lm_head required.");
            return 0;
        }

        // PLE mode: in-kernel gather (per-row token ids) beats the uploaded
        // ple_data; neither => no PLE (ple_dim treated as 0 for the graph).
        const bool ple_gather = ple_dim > 0 && ple_token_embd_data != nullptr && ple_token_ids != nullptr;
        const bool ple_upload = !ple_gather && ple_dim > 0 && ple_data != nullptr;
        const bool has_ple = ple_gather || ple_upload;
        if (has_ple && (ple_gate_arr == nullptr || ple_proj_arr == nullptr || ple_post_norm_arr == nullptr))
        {
            set_last_error("Gemma4 batched decode: PLE requested without the per-layer injection weights.");
            return 0;
        }
        const int total_ple_dim = has_ple ? num_layers * ple_dim : 0;

        struct LayerInfo { int hd; int kvHeads; int qDim; int kDim; int cacheSize; bool isLocal; int kvSource; bool isShared; int win; };
        std::vector<LayerInfo> li(num_layers);

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++)
            maxTotal = std::max(maxTotal, positions[s] + 1);

        // Persist (CUDA-graph capture) gate: identical to the single-stream
        // g_g4dc. Capturable when on CUDA and TS_GEMMA4_FD_PERSIST != 0.
        static const bool g4b_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4b_persist && g_backend_type == BACKEND_TYPE_CUDA;

        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.kvSource = kv_source_arr != nullptr ? kv_source_arr[l] : l;
            if (info.kvSource < 0 || info.kvSource >= num_layers)
            {
                set_last_error("Gemma4 batched decode: kv_source out of range.");
                return 0;
            }
            info.isShared = info.kvSource != l;
            // A donor must own its cache (a shared layer whose donor is itself
            // shared is not a shape any Gemma 4 GGUF produces) and must precede
            // its consumer so its window exists when the consumer is built.
            if (info.isShared && (info.kvSource > l || (kv_source_arr != nullptr && kv_source_arr[info.kvSource] != info.kvSource)))
            {
                set_last_error("Gemma4 batched decode: unsupported KV-donor topology.");
                return 0;
            }
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            // Shared layers attend the donor's cache: its size / locality.
            info.cacheSize = cache_size_arr[info.kvSource];
            info.isLocal = is_local_arr[info.kvSource] != 0;
            if (info.isShared && (info.hd != head_dim_arr[info.kvSource] || info.kvHeads != kv_heads_arr[info.kvSource]))
            {
                set_last_error("Gemma4 batched decode: shared layer head shape differs from its donor.");
                return 0;
            }
            if (info.cacheSize <= 0)
            {
                set_last_error("Gemma4 batched decode: empty KV cache.");
                return 0;
            }
            // Local (SWA ring) layers saturate at cache_size: past that the ring
            // is read flat and fully valid. A global (linear) cache must still
            // hold the whole sequence; the caller grows it before calling.
            const int validMax = info.isLocal ? std::min(maxTotal, info.cacheSize) : maxTotal;
            if (validMax > info.cacheSize)
            {
                set_last_error("Gemma4 batched decode: sequence exceeds a global layer's cache (caller must grow it).");
                return 0;
            }
            // Persist: pad the window to a 256-stride (bounded by the cache) so the
            // graph topology is identical token-to-token (only changes every 256
            // tokens), which is what lets ggml-cuda capture engage. Non-persist
            // uses the tight flash_attn length (clamped to the ring).
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(validMax), flash_attn_kv_length(validMax, info.cacheSize, info.hd)))
                : std::min(info.cacheSize, flash_attn_kv_length(validMax, info.cacheSize, info.hd));
        }

        // Per-(layer,seq) KV write row: the ring slot for a local layer, the
        // linear position otherwise.
        auto write_row = [&](int l, int s) -> std::int64_t {
            return li[l].isLocal ? (positions[s] % li[l].cacheSize) : positions[s];
        };

        // Per-seq attention-mask fill: column s zeroes [0, pos_s+1) (clamped to
        // win, which is the whole ring once a local layer has wrapped), -inf
        // elsewhere. Shared by the reuse fast-path and the build path.
        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++)
                    md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        // Pool identity: model instance + the canonical-order set of N layer-0 K
        // caches + N + per-layer window. (C# canonicalises the request order.)
        const void* sig_disc = attn_norm_arr[0];
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];   // layer 0, seq s
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        const std::size_t ple_upload_bytes = ple_upload
            ? static_cast<std::size_t>(n_seqs) * total_ple_dim * sizeof(float) : 0;

        // ---- reuse fast-path: replay this request-set's captured graph ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4batched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->vocab == vocab_size && dc->layer_window == winvec &&
            dc->ple_dim == (has_ple ? ple_dim : 0) && dc->ple_gather == ple_gather)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            if (dc->ple_ids != nullptr)
                ggml_backend_tensor_set(dc->ple_ids, ple_token_ids, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            else if (dc->ple_input != nullptr && ple_upload)
                ggml_backend_tensor_set(dc->ple_input, ple_data, 0, ple_upload_bytes);
            for (int l = 0; l < num_layers; l++)
            {
                if (li[l].isShared) continue;   // no write, donor's mask
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = write_row(l, s);
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md;
                fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (tsg::compute_graph(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 batched decode: replay graph compute failed.");
                dc->reset();
                return 0;
            }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        // ---- build context: persist = raw no_alloc ctx kept alive in the pool
        // (stable addresses for capture); non-persist = pooled (<=32MB). ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 batched decode: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Gemma4 batched decode: failed to create ggml context.");
                return 0;
            }
            ctx = context.value;
        }
        // Persist ctx must not leak on any early return below.
        struct PersistCtxGuard {
            ggml_context* c; bool armed;
            ~PersistCtxGuard() { if (armed && c != nullptr) ggml_free(c); }
        } persist_guard{ can_persist ? ctx : nullptr, can_persist };

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input [total_ple_dim, n_seqs]: gathered in-kernel (per-row token
        // ids are the INPUT) or uploaded (the tensor itself is the INPUT).
        // Mirrors TSGgml_Gemma4ModelVerify's batched PLE over N rows.
        ggml_tensor* ple_input = nullptr;
        ggml_tensor* ple_table_t = nullptr;
        ggml_tensor* ple_ids_t = nullptr;
        ggml_tensor* ple_model_proj_t = nullptr;
        ggml_tensor* ple_model_proj_norm_t = nullptr;
        if (ple_gather)
        {
            ple_table_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_token_embd_type),
                ple_token_embd_ne0, ple_token_embd_ne1);
            ple_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
            if (can_persist) ggml_set_input(ple_ids_t);
            // Token-embedding component: sqrt(ple_dim) * get_rows(table, ids) -> [total_ple_dim, N].
            ggml_tensor* ple_tok = ggml_get_rows(ctx, ple_table_t, ple_ids_t);
            ple_tok = ggml_scale(ctx, ple_tok, sqrtf(static_cast<float>(ple_dim)));
            if (ple_model_proj_data != nullptr && ple_model_proj_norm_data != nullptr)
            {
                // Hidden-projection component: rmsnorm((hidden @ proj)/sqrt(hidden), norm).
                ple_model_proj_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_model_proj_type),
                    ple_model_proj_ne0, ple_model_proj_ne1);
                ple_model_proj_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ple_dim);
                ggml_tensor* proj = ggml_mul_mat(ctx, ple_model_proj_t, current);   // [total_ple_dim, N]
                proj = ggml_scale(ctx, proj, 1.0f / sqrtf(static_cast<float>(hidden_size)));
                // Per-(seq,layer) RMSNorm over ple_dim: view as [ple_dim, num_layers*N].
                ggml_tensor* proj_r = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, static_cast<std::int64_t>(num_layers) * n_seqs);
                proj_r = ggml_mul(ctx, ggml_rms_norm(ctx, proj_r, eps), ple_model_proj_norm_t);
                proj = ggml_reshape_2d(ctx, proj_r, total_ple_dim, n_seqs);
                ple_input = ggml_scale(ctx, ggml_add(ctx, proj, ple_tok), 1.0f / sqrtf(2.0f));
            }
            else
            {
                ple_input = ple_tok;
            }
        }
        else if (ple_upload)
        {
            ple_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_ple_dim, n_seqs);
            if (can_persist) ggml_set_input(ple_input);
        }

        // Per-(layer,seq) I64 set_rows write-row inputs (persist only); kept for
        // the pool entry so the reuse path can refresh them each replay.
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            ggml_tensor* ple_gate_w; ggml_tensor* ple_proj_w; ggml_tensor* ple_post_norm_w;
            std::vector<ggml_tensor*> k_cached;   // per seq (own-cache layers)
            std::vector<ggml_tensor*> v_cached;   // per seq
            std::vector<ggml_tensor*> k_cpy;      // per seq (KV write op)
            std::vector<ggml_tensor*> v_cpy;
            std::vector<ggml_tensor*> k_src;      // per seq: the tensor the window read sees
            std::vector<ggml_tensor*> v_src;
            std::vector<ggml_tensor*> attn_col_cpy;   // per seq attention-column writes
            ggml_tensor* attn_mask;               // own-cache layers only (shared: donor's)
            std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            // Shared layers carry only the Q weight in qkv_w and never run their
            // own K/V projection (or K norm).
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            const bool separate_qkv = (!info.isShared && k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else { lt.k_w = nullptr; lt.v_w = nullptr; }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = info.isShared ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            lt.ple_gate_w = nullptr; lt.ple_proj_w = nullptr; lt.ple_post_norm_w = nullptr;
            if (has_ple && ple_gate_arr[l] != nullptr && ple_proj_arr[l] != nullptr && ple_post_norm_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }

            lt.k_cached.assign(n_seqs, nullptr);
            lt.v_cached.assign(n_seqs, nullptr);
            lt.k_cpy.assign(n_seqs, nullptr);
            lt.v_cpy.assign(n_seqs, nullptr);
            lt.k_src.assign(n_seqs, nullptr);
            lt.v_src.assign(n_seqs, nullptr);
            lt.attn_col_cpy.assign(n_seqs, nullptr);
            lt.attn_mask = nullptr;
            if (!info.isShared)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    lt.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                    lt.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                }
                lt.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
                if (can_persist) ggml_set_input(lt.attn_mask);
            }
        }

        // lm_head + final norm
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // ---- build graph ----
        ggml_tensor* hidden = current;   // [hidden_size, n_seqs]
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];
            int rope_dims = rope_n_dims_arr[l];
            ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
            // Window / mask owner: the layer itself, or its KV donor.
            auto& wl = layers[info.kvSource];
            auto& winfo = li[info.kvSource];

            // 1. attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);   // [H, N]

            // 2. Q(KV) projection -> [qDim, N] / [kDim, N]
            ggml_tensor* q_raw; ggml_tensor* k_raw = nullptr; ggml_tensor* v_raw = nullptr;
            if (info.isShared)
            {
                q_raw = ggml_mul_mat(ctx, lt.qkv_w, normed);   // Q only
            }
            else if (lt.k_w != nullptr)
            {
                q_raw = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim, N]
                k_raw = ggml_mul_mat(ctx, lt.k_w, normed);     // [kDim, N]
                v_raw = ggml_mul_mat(ctx, lt.v_w, normed);     // [kDim, N]
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim+2kDim, N]
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
            }

            // 3. per-head Q norm (+ K norm / V norm for own-cache layers) -> [hd, heads, N]
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            // 4. RoPE (pos[seq] applied per ne2 slice)
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, num_heads, N]
            ggml_tensor* k_rope = nullptr; ggml_tensor* v_normed = nullptr;
            if (!info.isShared)
            {
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvHeads, n_seqs);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvHeads, n_seqs);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);
                v_normed = ggml_rms_norm(ctx, v_3d, eps);
                k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, kvHeads, N]
            }

            // 5+6. Per-seq KV write, then per-seq single-query flash attention
            // over a DIRECT window view. The first version concatenated every
            // sequence's padded window along ne3 into one [hd, win, kvH, N]
            // tensor per layer per step; the chained concats re-copied the
            // accumulated windows (O(N^2) rows per layer per token), which made
            // the step time LINEAR in N (measured 15.9 / 31.7 ms TPOT at
            // N=2/4 on E4B) and pinned aggregate throughput at the
            // single-stream rate. This is the same rework the GPT-OSS batched
            // kernel got: each sequence runs the solo-shaped single-row fattn
            // reading its window in place, and only the tiny [qDim] output
            // column is copied. Projections and the FFN stay token-batched.
            ggml_tensor* q_rope_cont = ggml_cont(ctx, q_rope);   // [hd, nH, N]
            ggml_tensor* attn_2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.qDim, n_seqs);
            for (int s = 0; s < n_seqs; s++)
            {
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (info.isShared)
                {
                    // Donor's window, written earlier in this graph (donor < l).
                    k_src = wl.k_src[s];
                    v_src = wl.v_src[s];
                    if (k_src == nullptr || v_src == nullptr)
                    {
                        set_last_error("Gemma4 batched decode: shared layer has no donor KV.");
                        return 0;
                    }
                }
                else
                {
                    const std::int64_t cachePos = write_row(l, s);   // ring slot (local) or position
                    // slice this seq's new K/V: [hd, kvHeads, 1] -> permute -> [hd, 1, kvHeads]
                    ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvHeads, 1,
                        k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                    ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvHeads, 1,
                        v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                    ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                    ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                    // k_src/v_src is the tensor the window read sees. For persist we
                    // read from the set_rows RESULT (full-cache tensor), which gives a
                    // real graph edge write->read so the topological sort can never
                    // place the window read before this seq's KV write (without the
                    // edge, ggml relied on insertion order, which reordered the last
                    // sequence's read ahead of its write at N>=4 -> stale K/V).
                    if (can_persist)
                    {
                        // set_rows write (row = an I64 INPUT) keeps the graph topology
                        // identical token-to-token so CUDA-graph capture engages.
                        ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                        ggml_set_input(kv_idx);
                        kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                        lt.k_cpy[s] = ggml_set_rows(ctx, lt.k_cached[s], k_write, kv_idx);
                        lt.v_cpy[s] = ggml_set_rows(ctx, lt.v_cached[s], v_write, kv_idx);
                        k_src = lt.k_cpy[s];   // full-cache result of the write
                        v_src = lt.v_cpy[s];
                    }
                    else
                    {
                        ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cached[s], info.hd, 1, info.kvHeads,
                            lt.k_cached[s]->nb[1], lt.k_cached[s]->nb[2],
                            static_cast<std::size_t>(cachePos) * lt.k_cached[s]->nb[1]);
                        ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cached[s], info.hd, 1, info.kvHeads,
                            lt.v_cached[s]->nb[1], lt.v_cached[s]->nb[2],
                            static_cast<std::size_t>(cachePos) * lt.v_cached[s]->nb[1]);
                        lt.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                        lt.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                        k_src = lt.k_cached[s];
                        v_src = lt.v_cached[s];
                    }
                    lt.k_src[s] = k_src;
                    lt.v_src[s] = v_src;
                }

                // Windowed read [0, win) straight off this sequence's cache —
                // the single-row fattn (fattn_query_rows=1) applies the same
                // defensive-copy logic the solo decode kernel uses.
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, winfo.hd, winfo.cacheSize, winfo.kvHeads, 0, winfo.win, kv_cache_type, 1);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, winfo.hd, winfo.cacheSize, winfo.kvHeads, 0, winfo.win, kv_cache_type, 1);
                if (k_win == nullptr || v_win == nullptr)
                {
                    set_last_error("Gemma4 batched decode: failed to build KV window views.");
                    return 0;
                }

                // This sequence's query, [hd, 1, nH] like the solo kernel's.
                ggml_tensor* q_s = ggml_view_3d(ctx, q_rope_cont, info.hd, num_heads, 1,
                    q_rope_cont->nb[1], q_rope_cont->nb[2],
                    static_cast<std::size_t>(s) * q_rope_cont->nb[2]);
                ggml_tensor* q_attn = ggml_permute(ctx, q_s, 0, 2, 1, 3);
                // Column s of the window owner's [win, 1, 1, N] mask input.
                ggml_tensor* mask_s = ggml_view_4d(ctx, wl.attn_mask, winfo.win, 1, 1, 1,
                    wl.attn_mask->nb[1], wl.attn_mask->nb[2], wl.attn_mask->nb[3],
                    static_cast<std::size_t>(s) * wl.attn_mask->nb[3]);
                ggml_tensor* fa = flash_attn_ext_guarded(ctx, "Gemma4 batched decode", q_attn, k_win, v_win, mask_s,
                    1.0f, 0.0f, 0.0f, nullptr, GGML_PREC_F32);

                // Deposit this sequence's attention output into column s of the
                // packed [qDim, N] activation; node ORDER (expanded before the
                // O projection's consumers) guarantees the writes land first.
                ggml_tensor* fa_flat = ggml_reshape_1d(ctx, fa, info.qDim);
                ggml_tensor* col = ggml_view_1d(ctx, attn_2d, info.qDim,
                    static_cast<std::size_t>(s) * attn_2d->nb[1]);
                lt.attn_col_cpy[s] = ggml_cpy(ctx, fa_flat, col);
            }

            // 7. O projection -> [H, N]
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_2d);   // [H, N]

            // 8. post-attn norm + residual
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // 9. FFN
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            std::int64_t inter = gu_ne1_arr[l] / 2;
            ggml_tensor* gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);   // [2*inter, N]
            ggml_tensor* gate = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1], 0);
            ggml_tensor* up = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1],
                static_cast<std::size_t>(inter) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, ggml_cont(ctx, gate)), ggml_cont(ctx, up));   // [inter, N]
            ggml_tensor* down_out = ggml_mul_mat(ctx, lt.down_w, ffn_hidden);   // [H, N]

            // 10. post-FFN norm + residual
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down_out, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, post_ffn, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // 11. PLE injection (mirrors Gemma4ModelDecode / ModelVerify, batched
            // over the N rows). ple_slice is a strided column view of ple_input:
            // ggml_mul only needs contiguous ROWS, so no cont.
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                ggml_tensor* ple_slice = ggml_view_2d(ctx, ple_input, ple_dim, n_seqs,
                    static_cast<std::size_t>(total_ple_dim) * sizeof(float),
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float));               // [ple_dim, N]
                ggml_tensor* ple_gate_proj = ggml_mul_mat(ctx, lt.ple_gate_w, residual2);  // [ple_dim, N]
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);  // [ple_dim, N]
                ggml_tensor* ple_proj = ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated);       // [H, N]
                ggml_tensor* ple_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, ple_normed, residual2);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
            }

            // 12. layer scalar
            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // final norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);   // [vocab, N]
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        // build graph: KV writes first, then output
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (256 + 48 * static_cast<std::size_t>(n_seqs)) + 2048;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (layers[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy[s]);
                if (layers[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy[s]);
                if (layers[l].attn_col_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].attn_col_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        // ---- bind weights ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
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
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            if (lt.k_w != nullptr)
            {
                bind_or_mark(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
                bind_or_mark(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            }
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (lt.k_norm_w != nullptr)
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
            if (info.isShared) continue;   // donor's caches + mask
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(lt.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            // per-seq attention mask (host scratch, not cacheable)
            fill_batched_mask(lt.attn_mask_data, info.win);
            bind_or_mark(lt.attn_mask, lt.attn_mask_data.data(), lt.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        if (ple_gather)
        {
            bind_or_mark(ple_table_t, const_cast<void*>(ple_token_embd_data), static_cast<std::size_t>(ple_token_embd_bytes), true);
            if (ple_model_proj_t != nullptr)
                bind_or_mark(ple_model_proj_t, const_cast<void*>(ple_model_proj_data), static_cast<std::size_t>(ple_model_proj_bytes), true);
            if (ple_model_proj_norm_t != nullptr)
                bind_or_mark(ple_model_proj_norm_t, const_cast<void*>(static_cast<const void*>(ple_model_proj_norm_data)), static_cast<std::size_t>(ple_dim) * sizeof(float), true);
        }

        // Persist: stable addresses for capture; Metal shares completed attention workspaces,
        // kept alive in the pool. Non-persist: reuse the pooled compute buffer.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate persist buffer.");
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx, graph))
        {
            buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0, static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));
        if (ple_ids_t != nullptr)
            ggml_backend_tensor_set(ple_ids_t, ple_token_ids, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        else if (ple_upload && ple_input != nullptr)
            ggml_backend_tensor_set(ple_input, ple_data, 0, ple_upload_bytes);
        if (can_persist)
        {
            for (int l = 0; l < num_layers; l++)
            {
                if (li[l].isShared) continue;
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = write_row(l, s);
                    ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
            }
        }

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 batched decode: graph compute failed.");
            if (can_persist) ggml_backend_buffer_free(persist_buf);
            return 0;
        }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
        // Unconditional: logits_data is the caller's host buffer and on Metal
        // async mode the download above is only QUEUED.
        host_read_barrier();

        if (can_persist)
        {
            G4BatchedDecodeCache& e = g_g4batched_pool.claim(sig_disc, sig_kc, n_seqs);
            persist_guard.armed = false;   // ownership moves to the pool entry
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = layers[l].attn_mask;
            e.layer_window = winvec;
            e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = hidden_size; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.ple_ids = ple_ids_t;
            e.ple_input = ple_upload ? ple_input : nullptr;
            e.ple_dim = has_ple ? ple_dim : 0;
            e.ple_gather = ple_gather;
            e.valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 batched decode."); return 0; }
}

// v1 ABI (kept for older managed builds): no KV-donor map, no PLE, and the
// kernel's own no-wrap regime is what its callers already enforce.
TSG_EXPORT int TSGgml_Gemma4ModelDecodeBatched(
    float* hidden_data, int hidden_size, int num_layers, int n_seqs,
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads,
    const int* positions,
    float eps, int sliding_window,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    return TSGgml_Gemma4ModelDecodeBatchedEx(
        hidden_data, hidden_size, num_layers, n_seqs,
        attn_norm_arr, qkv_arr, q_norm_arr, k_norm_arr, o_arr, post_attn_norm_arr,
        ffn_norm_arr, gu_arr, down_arr, post_ffn_norm_arr,
        k_cache_arr, v_cache_arr,
        head_dim_arr, kv_heads_arr, cache_size_arr, is_local_arr,
        rope_base_arr, layer_scalar_arr,
        qkv_type_arr, qkv_ne0_arr, qkv_ne1_arr, qkv_bytes_arr,
        o_type_arr, o_ne0_arr, o_ne1_arr, o_bytes_arr,
        gu_type_arr, gu_ne0_arr, gu_ne1_arr, gu_bytes_arr,
        down_type_arr, down_ne0_arr, down_ne1_arr, down_bytes_arr,
        num_heads, positions, eps, sliding_window,
        rope_freq_factors, rope_freq_factors_len, rope_n_dims_arr, kv_cache_type,
        k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr,
        v_arr, v_type_arr, v_ne0_arr, v_ne1_arr, v_bytes_arr,
        logits_data, vocab_size,
        lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
        final_norm_data, logit_softcap,
        /*kv_source_arr=*/nullptr,
        /*ple_data=*/nullptr, /*ple_dim=*/0,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr,
        nullptr, 0, 0, 0, 0, nullptr,
        nullptr, 0, 0, 0, 0, nullptr);
}

// Drop all captured token-batched decode graphs. The captured graphs pin
// ggml-cuda's compute-pool scratch + the per-request KV buffers; a prefill (grows
// the pool) or a KV reset/grow can move those, so the C# caller drops the cache
// before any prefill and on ResetKVCache. No-op when persist is off.
TSG_EXPORT void TSGgml_Gemma4ResetBatchedDecodeCache()
{
    g_g4batched_pool.reset_all();
}

namespace { G4BatchedDecodeCachePool g_g4moebatched_pool; }

// ============================================================================
// TRUE TOKEN-BATCHED MoE decode (N concurrent sequences, one token each, one
// ggml graph + one compute buffer). The MoE sibling of
// TSGgml_Gemma4ModelDecodeBatched: identical batched attention (per-request KV,
// flash_attn_ext over ne3=N, set_rows write + fixed padded window for capture)
// but the FFN is the Gemma-4 MoE block (dense shared FFN + in-graph router +
// stacked experts via ggml_mul_mat_id over N tokens — token-parallel, copied
// from the verify kernel). Weights come from the existing per-layer
// TSGgmlGemma4MoELayerDesc array (its k_cache/v_cache/position fields are
// IGNORED — overridden by k_cache_arr[layer*n_seqs+seq] / positions[seq]).
// v1 scope mirrors the dense path: all-MoE, no PLE, no KV-donor, folded lm_head,
// no-wrap regime; CUDA-graph capture via g_g4moebatched_pool.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4MoEModelDecodeBatched(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,
    void** k_cache_arr, void** v_cache_arr,   // [layer * n_seqs + seq]
    const int* positions,                     // [n_seqs]
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        { set_last_error("Gemma4 MoE batched decode: invalid arguments."); return 0; }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        { set_last_error("Gemma4 MoE batched decode: descriptor size mismatch."); return 0; }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold) { set_last_error("Gemma4 MoE batched decode: folded lm_head required."); return 0; }

        const int H = layers[0].hidden_size;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++) maxTotal = std::max(maxTotal, positions[s] + 1);

        static const bool g4mb_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4mb_persist && g_backend_type == BACKEND_TYPE_CUDA;
        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };

        struct LInfo { int hd, kvH, qDim, cacheSize, win; bool isLocal; };
        std::vector<LInfo> li(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l];
            if (d.is_shared != 0) { set_last_error("Gemma4 MoE batched decode: KV-donor layers unsupported."); return 0; }
            auto& info = li[l];
            info.hd = d.head_dim; info.kvH = d.num_kv_heads; info.qDim = num_heads * d.head_dim;
            info.cacheSize = d.cache_size; info.isLocal = d.is_local != 0;
            if (info.cacheSize <= 0 || maxTotal > info.cacheSize)
            { set_last_error("Gemma4 MoE batched decode: sequence exceeds cache window (wrap unsupported)."); return 0; }
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(maxTotal), flash_attn_kv_length(maxTotal, info.cacheSize, info.hd)))
                : flash_attn_kv_length(maxTotal, info.cacheSize, info.hd);
        }

        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++) md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        const void* sig_disc = layers[0].attn_norm_w;
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        // ---- reuse fast-path ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4moebatched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->vocab == vocab_size && dc->layer_window == winvec)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md; fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (tsg::compute_graph(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            { set_last_error("Gemma4 MoE batched decode: replay failed."); dc->reset(); return 0; }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        // ---- build ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 MoE batched decode: ctx init failed."); return 0; }
        }
        else { if (!context.init(ctx_size)) { set_last_error("Gemma4 MoE batched decode: ctx pool failed."); return 0; } ctx = context.value; }

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LT {
            ggml_tensor *attn_norm_w, *qkv_w, *k_w, *v_w, *q_norm_w, *k_norm_w, *o_w, *post_attn_norm_w;
            ggml_tensor *ffn_norm_w, *gu_w, *down_w, *post_ffw_norm_1_w;
            ggml_tensor *gate_inp_w, *gate_inp_scale_t, *pre_ffw_norm_2_w, *gate_up_exps_t, *down_exps_t, *down_exps_scale_t, *post_ffw_norm_2_w, *post_ffw_norm_w;
            ggml_tensor *freq_factors_t;
            std::vector<ggml_tensor*> k_cached, v_cached, k_cpy, v_cpy, attn_col_cpy;
            ggml_tensor* attn_mask; std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LT> lt(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts;
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (d.separate_qkv != 0) { t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1); t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1); }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
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
            t.freq_factors_t = (!info.isLocal && d.freq_factors != nullptr && d.freq_factors_len > 0)
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.freq_factors_len) : nullptr;
            t.k_cached.resize(n_seqs); t.v_cached.resize(n_seqs); t.k_cpy.resize(n_seqs, nullptr); t.v_cpy.resize(n_seqs, nullptr);
            t.attn_col_cpy.resize(n_seqs, nullptr);
            for (int s = 0; s < n_seqs; s++)
            {
                t.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
                t.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
            }
            t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
            if (can_persist) ggml_set_input(t.attn_mask);
        }

        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);


        ggml_tensor* hidden = current;
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts, nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gu_ne1 / 2, ffMoe = d.gue_ne1 / 2;
            ggml_tensor* rope_ff = t.freq_factors_t;

            // ---- attention (batched, per-seq KV) ----
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* q_raw; ggml_tensor* k_raw; ggml_tensor* v_raw;
            if (t.k_w != nullptr) { q_raw = ggml_mul_mat(ctx, t.qkv_w, normed); k_raw = ggml_mul_mat(ctx, t.k_w, normed); v_raw = ggml_mul_mat(ctx, t.v_w, normed); }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim + info.kvH * info.hd) * sizeof(float));
            }
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), t.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_3d, eps);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);

            // Per-seq single-query fattn over direct window views — replaces
            // the concat-along-ne3 chain whose re-copies made the step LINEAR
            // in N (see the dense kernel above for the full story).
            ggml_tensor* q_rope_cont = ggml_cont(ctx, q_rope);   // [hd, nH, N]
            ggml_tensor* attn_2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.qDim, n_seqs);
            for (int s = 0; s < n_seqs; s++)
            {
                const int cachePos = positions[s];
                ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvH, 1, k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvH, 1, v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));
                ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (can_persist)
                {
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                    t.k_cpy[s] = ggml_set_rows(ctx, t.k_cached[s], k_write, kv_idx);
                    t.v_cpy[s] = ggml_set_rows(ctx, t.v_cached[s], v_write, kv_idx);
                    k_src = t.k_cpy[s]; v_src = t.v_cpy[s];
                }
                else
                {
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cached[s], info.hd, 1, info.kvH, t.k_cached[s]->nb[1], t.k_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.k_cached[s]->nb[1]);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cached[s], info.hd, 1, info.kvH, t.v_cached[s]->nb[1], t.v_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.v_cached[s]->nb[1]);
                    t.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                    t.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                    k_src = t.k_cached[s]; v_src = t.v_cached[s];
                }
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType, 1);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType, 1);
                if (k_win == nullptr || v_win == nullptr)
                {
                    set_last_error("Gemma4 MoE batched decode: failed to build KV window views.");
                    return 0;
                }
                ggml_tensor* q_s = ggml_view_3d(ctx, q_rope_cont, info.hd, num_heads, 1,
                    q_rope_cont->nb[1], q_rope_cont->nb[2],
                    static_cast<std::size_t>(s) * q_rope_cont->nb[2]);
                ggml_tensor* q_attn = ggml_permute(ctx, q_s, 0, 2, 1, 3);
                ggml_tensor* mask_s = ggml_view_4d(ctx, t.attn_mask, info.win, 1, 1, 1,
                    t.attn_mask->nb[1], t.attn_mask->nb[2], t.attn_mask->nb[3],
                    static_cast<std::size_t>(s) * t.attn_mask->nb[3]);
                ggml_tensor* fa = flash_attn_ext_guarded(ctx, "Gemma4 MoE batched decode", q_attn, k_win, v_win, mask_s,
                    1.0f, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
                ggml_tensor* fa_flat = ggml_reshape_1d(ctx, fa, info.qDim);
                ggml_tensor* col = ggml_view_1d(ctx, attn_2d, info.qDim,
                    static_cast<std::size_t>(s) * attn_2d->nb[1]);
                t.attn_col_cpy[s] = ggml_cpy(ctx, fa_flat, col);
            }
            ggml_tensor* o_out = ggml_mul_mat(ctx, t.o_w, attn_2d);
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), t.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden);   // [H, N]

            // ---- dense shared FFN (N tokens) ----
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);
            ggml_tensor* dense_gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], 0));
            ggml_tensor* dense_up = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
            ggml_tensor* dense_down = ggml_mul_mat(ctx, t.down_w, dense_h);
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);

            // ---- MoE router (N tokens) ----
            ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);
            route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr) route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);
            ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);
            ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, n_seqs);
            ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, n_seqs);
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
            w_2d = ggml_div(ctx, w_2d, w_sum);
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_b = ggml_repeat(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), probs_r);
                ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_b, sel);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, n_seqs));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, n_seqs);

            // ---- MoE experts (N tokens) ----
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w);
            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, n_seqs);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w);
            mlp = ggml_add(ctx, mlp, moe_normed);

            ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
            ggml_tensor* result = ggml_add(ctx, mlp_normed, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
            if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f) result = ggml_scale(ctx, result, d.layer_output_scale);
            hidden = result;
        }

        // fold final-norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (384 + 48 * static_cast<std::size_t>(n_seqs)) + 2048;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (lt[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy[s]);
                if (lt[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy[s]);
                if (lt[l].attn_col_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].attn_col_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf); if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr) { bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true); bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true); }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * d.num_experts * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr) bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr) bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(d.num_experts) * sizeof(float), true);
            if (t.freq_factors_t != nullptr) bind_or_mark(t.freq_factors_t, d.freq_factors, static_cast<std::size_t>(d.freq_factors_len) * sizeof(float), true);
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(t.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            fill_batched_mask(t.attn_mask_data, info.win);
            bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        // NOTE: a dedicated gallocr (packed, VRAM-frugal) was tried here to fit the
        // 26B on 16GB — it ran FAST (N=2 ~99 t/s) but produced garbage under this
        // fork's CUDA-graph capture (gallocr slot-REUSE is incompatible with the
        // captured graph; own-slot works precisely because it never reuses). So we
        // keep the capture-safe own-slot alloc; the 26B batched is VRAM-bound on 16GB
        // and stays gated off (see TS_BATCHED_FUSED_MOE).
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (persist_buf == nullptr) { set_last_error("Gemma4 MoE batched decode: persist alloc failed."); ggml_free(ctx); return 0; }
        }
        else if (!alloc_ctx_tensors_reuse(ctx, graph))
        {
            buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr) { set_last_error("Gemma4 MoE batched decode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (can_persist)
            for (int l = 0; l < num_layers; l++)
                for (int s = 0; s < n_seqs; s++)
                { std::int64_t row = positions[s]; ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t)); }

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        { set_last_error("Gemma4 MoE batched decode: graph compute failed."); if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); } return 0; }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
        // Unconditional: logits_data is the caller's host buffer and on Metal
        // async mode the download above is only QUEUED.
        host_read_barrier();

        if (can_persist)
        {
            G4BatchedDecodeCache& e = g_g4moebatched_pool.claim(sig_disc, sig_kc, n_seqs);
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = lt[l].attn_mask;
            e.layer_window = winvec; e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = H; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 MoE batched decode."); return 0; }
}

TSG_EXPORT void TSGgml_Gemma4ResetMoEBatchedDecodeCache()
{
    g_g4moebatched_pool.reset_all();
}

