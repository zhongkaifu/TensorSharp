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

// ============================================================================
// Gemma4 full-model decode: ALL dense transformer layers in a single GGML graph.
// Handles Gemma4-specific features: GELU activation, V norm, post-attn/FFN norms,
// layer scalars, different head dims per layer type, sliding window, softcap.
// ============================================================================

namespace
{
    // Persistent decode-graph cache for the dense Gemma 4 trunk — the exact
    // analogue of g_q35dc (see its comment). The whole-model decode graph is
    // built ONCE with stable tensor addresses (raw ggml ctx + alloc_ctx_tensors)
    // and reused across tokens, so ggml-cuda's CUDA-graph capture engages (one
    // captured replay instead of re-launching ~1300 kernels/token, which on WDDM
    // starves the GPU — measured ~25 tok/s vs ~38 for llama.cpp before this).
    //
    // To keep the graph topology byte-identical token-to-token (the requirement
    // for replay, see ggml_cuda_graph_update_required): the KV write uses
    // ggml_set_rows (write row = an I64 INPUT, not a baked view offset) and the
    // attention reads a FIXED padded window [0, window) with an F16 mask INPUT.
    // The window is padded up to a 256-token stride so it only changes (forcing a
    // rebuild) every 256 tokens. Global layers grow their window; local (SWA)
    // layers saturate at the sliding-window size and then read the whole circular
    // cache flat (attention is permutation-invariant over the KV axis, so the
    // circular order is irrelevant). PLE / per-layer-input embeddings are an
    // extra per-token input. Dropped + rebuilt when any layer window grows a
    // stride, the model instance changes, or the KV-cache buffer is reallocated.
    constexpr int kG4PersistKvStride = 256;

    struct G4DecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* ple_input = nullptr;            // nullable (uploaded-PLE mode)
        ggml_tensor* ple_ids = nullptr;              // nullable (in-kernel PLE gather mode: per-token I32 id)
        std::vector<ggml_tensor*> kv_index;          // per-layer I64 write row (null for shared)
        std::vector<ggml_tensor*> attn_mask;         // per-layer F16 padding mask (shared -> donor's)
        std::vector<int> layer_window;               // per-layer padded window length
        const void* sig_disc = nullptr;              // model-instance discriminator (attn_norm[0])
        const void* sig_kcache0 = nullptr;           // first KV buffer ptr (detects realloc/grow)
        int num_layers = 0;
        int hidden_size = 0;
        int ple_dim = 0;
        bool ple_gather = false;                     // in-kernel PLE gather graph vs uploaded ple_data
        bool folded = false;                         // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                           // floats to download (vocab when folded, else hidden)
        // Tensor-parallel execution plan for this rank: the same graph, cut at
        // the two row-parallel projections per layer whose partial sums the
        // ranks must AllReduce. Empty for a single-device decode.
        tsg::TpRankPlan tp_plan;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = ple_input = ple_ids = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = ple_dim = 0;
            ple_gather = false;
            folded = false; out_count = 0;
            tp_plan.clear();
        }
    };

    // TS_GEMMA4_FD_DEBUG=1: report any graph tensor that reached compute without a
    // backend buffer. ggml-vulkan asserts deep inside ggml_vk_tensor_subbuffer for
    // these, which gives no hint which node or which path produced it.
    void g4_debug_check_buffers(ggml_cgraph* graph, const char* where)
    {
        static const bool on = []{ const char* e = std::getenv("TS_GEMMA4_FD_DEBUG"); return e != nullptr && e[0] == '1'; }();
        if (!on || graph == nullptr) return;
        int bad = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            ggml_tensor* n = ggml_graph_node(graph, i);
            if (n == nullptr) continue;
            if (n->buffer == nullptr)
                fprintf(stderr, "[g4-fd %s] node %d '%s' op=%s HAS NULL BUFFER\n",
                        where, i, n->name, ggml_op_name(n->op)), bad++;
            for (int j = 0; j < GGML_MAX_SRC; j++)
            {
                ggml_tensor* srcj = n->src[j];
                if (srcj != nullptr && srcj->buffer == nullptr)
                    fprintf(stderr, "[g4-fd %s] node %d '%s' op=%s src%d '%s' op=%s HAS NULL BUFFER\n",
                            where, i, n->name, ggml_op_name(n->op), j, srcj->name, ggml_op_name(srcj->op)), bad++;
            }
        }
        fprintf(stderr, "[g4-fd %s] nodes=%d null-buffer findings=%d\n", where, ggml_graph_n_nodes(graph), bad);
        fflush(stderr);
    }

    // Concurrent (N>=2) requests each decode through their OWN KV-cache holder
    // (Gemma4Model.BindSequenceCache swaps _kvCacheK), so a single shared cache
    // entry — whose identity key includes sig_kcache0, the holder's layer-0 KV
    // pointer — is busted on EVERY request switch and rebuilt from scratch,
    // collapsing aggregate throughput BELOW the single-stream rate. Keep a small
    // pool keyed by (sig_disc, sig_kcache0) so each in-flight request retains its
    // own persistent, CUDA-graph-captured decode graph. ggml-cuda already caches a
    // separate captured cuda graph per distinct cgraph->nodes[0]
    // (ggml_cuda_graph_get_key), so switching requests is a cheap replay instead
    // of a full graph rebuild + recapture.
    //
    // Address-keying is self-consistent: a captured graph bakes in the KV device
    // addresses, and the per-call inputs (hidden / pos / kv_index / mask) are
    // uploaded fresh each replay, so an entry is only ever matched when a LIVE
    // holder presents its kc0 — a freed-then-reallocated address simply binds the
    // new live holder's buffers. Entries' own ctx/buffer stay alive until reset or
    // LRU eviction; ResetKVCache drops them all.

    constexpr int kG4MaxDecodeCaches = 8;
    struct G4DecodeCachePool
    {
        G4DecodeCache entries[kG4MaxDecodeCaches];
        std::uint64_t used[kG4MaxDecodeCaches] = {};   // LRU clock per slot
        std::uint64_t clock = 0;

        // Live entry matching this identity (model instance + KV holder), or null.
        G4DecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        // A reset slot to (re)build this identity's graph into: reuse its existing
        // slot if present (shape change -> rebuild in place), else an empty slot,
        // else evict the least-recently-used.
        G4DecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4MaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        // Drop the entry for one identity (a call that cannot persist for it).
        void drop(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                    entries[i].reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    // One pool per rank. Under tensor parallelism each GPU builds its own graph
    // over its own weight shards, and the entries hold that rank's ggml context
    // and backend buffer — they are not interchangeable between devices.
    G4DecodeCachePool g_g4dc_pools[tsg::TSG_MAX_DEVICES];
    inline G4DecodeCachePool& g4dc_pool() { return g_g4dc_pools[tsg::g_active_rank]; }
}

TSG_EXPORT int TSGgml_Gemma4ModelDecode(
    float* hidden_data, int hidden_size, int num_layers,
    // Per-layer weight pointers (arrays of size num_layers)
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-layer KV caches
    void** k_cache_arr, void** v_cache_arr,
    // Per-layer metadata (arrays of size num_layers)
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    int* kv_source_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    // Per-layer weight shapes
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    // Global params
    int num_heads, int position,
    float eps, int sliding_window,
    // RoPE freq_factors (nullable, for global layers with proportional RoPE)
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    // PLE data (nullable)
    float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr,
    int kv_cache_type,
    // Separate K/V projection weights for mixed-quantization models (e.g.
    // UD-IQ2_M) where attn_q/attn_k/attn_v carry DIFFERENT ggml types and so
    // cannot be fused into a single attn_qkv tensor. When k_arr[l] != nullptr
    // the layer runs three separate Q/K/V matmuls and qkv_arr[l] then holds
    // the Q weight (with qkv_*_arr[l] describing Q). When k_arr == nullptr or
    // k_arr[l] == nullptr the layer uses the fused attn_qkv weight as before,
    // so existing fully-fused callers are unaffected.
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Folded final-norm + lm_head (nullable). When logits_data / lm_head_data /
    // final_norm_data are non-null and vocab_size > 0, the graph appends the
    // output RMSNorm, the lm_head matmul and (when logit_softcap > 0) the
    // tanh logit softcap, writing logits[vocab] to logits_data so the whole
    // token — including the 262K-vocab projection — is one captured replay and
    // the C# caller skips its separate final-norm/lm_head/softcap dispatches.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap,
    // In-kernel PLE gather (nullable, mirrors TSGgml_Gemma4ModelVerify): when the
    // quantized per_layer_token_embd table (+ optionally the quantized
    // per_layer_model_proj and its F32 norm) is supplied along with the token id,
    // the graph reproduces C#'s ComputePLE on-device — get_rows + hidden
    // projection + RMSNorm + combine — and `ple_data` is ignored. This removes
    // ~5 per-op device dispatches per decode token (measured ~4.8 ms/token on
    // ggml-vulkan, ~1 ms on ggml-cuda).
    const void* ple_token_embd_data, int ple_token_embd_type,
    std::int64_t ple_token_embd_ne0, std::int64_t ple_token_embd_ne1, std::int64_t ple_token_embd_bytes,
    int ple_token_id,
    const void* ple_model_proj_data, int ple_model_proj_type,
    std::int64_t ple_model_proj_ne0, std::int64_t ple_model_proj_ne1, std::int64_t ple_model_proj_bytes,
    const float* ple_model_proj_norm_data,
    // Tensor parallelism. With tp_degree > 1 the caller drives one rank at a
    // time: every weight/KV pointer above is THIS rank's shard, num_heads and
    // kv_heads_arr are this rank's head counts, and the call *builds and binds*
    // the graph instead of running it — it hands back a plan (via tp_plan_out)
    // that names the two per-layer AllReduce points. The caller collects one
    // plan per rank and runs them together through
    // TSGgml_TensorParallelExecutePlans, so the GPUs overlap and the partial
    // sums are reduced in VRAM. tp_degree <= 1 is the ordinary single-device
    // path and ignores both parameters.
    int tp_degree, void** tp_plan_out,
    // Separate gate/up FFN weights, for the same reason as the separate K/V
    // weights above and one more: fusing ffn_gate and ffn_up into ffn_gate_up is
    // a COPY, because a GGUF writes a block's tensors alphabetically and
    // ffn_norm sits between them. On gemma-4-12b UD-Q4_K_XL that copy is 3.1 GB
    // of anonymous memory duplicating bytes already mapped from the file, which
    // on a phone is charged against the jetsam limit and is what killed the app.
    // When gate_arr[l] != nullptr the layer runs two matmuls and gu_arr[l] is
    // ignored; when gate_arr == nullptr or gate_arr[l] == nullptr the fused
    // weight is used exactly as before, so existing callers are unaffected.
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr)
{
    try
    {
        if (!ensure_backend())
            return 0;

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

        const int totalSeqLen = position + 1;

        // In-kernel PLE gather mode (see the parameter block above).
        const bool ple_gather = ple_dim > 0 && ple_token_embd_data != nullptr && ple_token_id >= 0;

        // Fold final-norm + lm_head into the graph when the caller supplies the
        // lm_head weight, output-norm weight and a logits output buffer.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;

        // Compute max head dim for context sizing
        int maxHd = 0;
        for (int l = 0; l < num_layers; l++)
            if (head_dim_arr[l] > maxHd) maxHd = head_dim_arr[l];

        // Prepare per-layer KV cache metadata
        struct LayerInfo {
            int hd;
            int kvHeads;
            int qDim;
            int kDim;
            int cacheSize;
            bool isLocal;
            bool isShared;
            int kvSource;
            int attendLen;
        };
        std::vector<LayerInfo> li(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.kvSource = kv_source_arr[l];
            info.isShared = (info.kvSource != l);

            // For shared layers, use the donor's cache size/local flag
            int kvSrc = info.kvSource;
            info.cacheSize = cache_size_arr[kvSrc];
            info.isLocal = is_local_arr[kvSrc] != 0;
            info.attendLen = info.isLocal ? std::min(totalSeqLen, sliding_window) : totalSeqLen;
        }

        // ============================ persist decode ============================
        // Per-token capturable-graph setup (mirrors Qwen3.5 g_q35dc). Compute each
        // layer's PADDED attention window, its valid (unmasked) length, and the KV
        // write row, then either replay the cached graph or fall through to build
        // a fresh persistent one.
        static const bool g4_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();

        std::vector<int> pwindow(num_layers, 0);          // padded window length per layer
        std::vector<int> pvalid(num_layers, 0);           // unmasked length per layer
        std::vector<std::int64_t> pwrite(num_layers, 0);  // set_rows write row per layer
        // Persist (capturable-graph) decode writes the KV cache with ggml_set_rows
        // (write row = an I64 INPUT). On the Metal backend ggml_set_rows over a
        // gallocr-allocated graph hands a null-context buffer to
        // ggml_metal_op_set_rows -> ggml_metal_buffer_get_id (EXC_BAD_ACCESS at
        // 0x14, SIGSEGV), so Metal falls through to the proven ggml_cpy path
        // below, which STILL folds the LM head and keeps the KV cache
        // device-resident. On CUDA persist enables CUDA-graph capture; on Vulkan
        // there is no capture but reusing the built graph still removes the
        // per-token rebuild + bind + gallocr work (~1 ms/token measured) and lets
        // ggml-vulkan's rope+view+set_rows subgraph fusion apply. Vulkan's
        // set_rows covers every KV dtype used here (F32/F16/BF16/Q8_0/Q4_0).
        // TS_GEMMA4_METAL_PERSIST=1 re-tests the Metal ggml_set_rows crash described
        // above. On the current vendored ggml it no longer reproduces: the graph
        // builds, replays, and decodes byte-identically. It stays OFF anyway,
        // because the thing it was supposed to buy is not there — measured on
        // gemma-4-E4B Q8_0 / M5 Pro, 46.4 tok/s off against 46.6 on, i.e. the
        // per-token rebuild is ~0.4% of a decode step, not the ~1 ms/token it
        // costs on Vulkan. Decode here is bandwidth-bound on the weights (7.6 GB
        // per token at ~350 GB/s), so there is nothing for a cached graph to win.
        // Left as an A/B lever rather than a default: flipping a path that once
        // SIGSEGV'd needs a better reason than noise.
        static const bool s_metalPersistProbe = []{
            const char* e = std::getenv("TS_GEMMA4_METAL_PERSIST");
            return e != nullptr && e[0] == '1';
        }();
        bool can_persist = g4_persist &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
             (s_metalPersistProbe && g_backend_type == BACKEND_TYPE_METAL));
        {
            auto roundup_stride = [](int v){ return ((v + kG4PersistKvStride - 1) / kG4PersistKvStride) * kG4PersistKvStride; };
            for (int l = 0; l < num_layers; l++)
            {
                const int csz = li[l].cacheSize;
                if (csz <= 0) { can_persist = false; break; }
                if (li[l].isLocal)
                {
                    if (totalSeqLen <= csz) { pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen; }
                    else { pwindow[l] = csz; pvalid[l] = csz; }   // saturated: read whole circular cache flat
                }
                else
                {
                    pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen;
                }
                pwrite[l] = li[l].isLocal ? (position % csz) : position;
                // A global cache that already overflowed can't be expressed as a
                // single padded window -> let the legacy per-op path handle it.
                if (!li[l].isShared && pvalid[l] > pwindow[l]) { can_persist = false; break; }
            }
        }
        // The tensor-parallel driver executes the graph *after* this call returns,
        // so the graph, its context and its backend buffer have to outlive the
        // call — which is exactly what persist mode provides. Without it there is
        // nothing to hand back, so the caller falls back to its per-op forward.
        if (tp_mode && !can_persist)
        {
            set_last_error("Gemma4 model decode: tensor-parallel mode requires the persistent decode graph.");
            return 0;
        }

        const void* g4_sig = attn_norm_arr[0];
        const void* g4_kc0 = k_cache_arr != nullptr ? k_cache_arr[0] : nullptr;

        // ---- reuse fast-path: replay THIS request's captured graph ----
        // Look the graph up by (model instance, KV holder). A per-request hit lets
        // concurrent requests each replay their own captured graph instead of one
        // shared entry that rebuilds on every switch. find() already matched
        // sig_disc + sig_kcache0, so only the finer shape fields are checked here.
        G4DecodeCache* dc = (can_persist && g4_persist) ? g4dc_pool().find(g4_sig, g4_kc0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->ple_dim == ple_dim && dc->ple_gather == ple_gather &&
            dc->layer_window == pwindow &&
            dc->folded == fold && dc->out_count == (fold ? vocab_size : hidden_size))
        {
            host_read_barrier();
            // Per-token input refresh. These ~N*2 small host->device copies feed the
            // captured graph's INPUT tensors; they are not graph nodes. Issue them
            // ASYNC on the backend stream so they queue (stream-ordered) ahead of the
            // graph replay below instead of each blocking on its own stream sync —
            // the replay and the trailing finalize_compute_with_download() sync make
            // the data visible at the right points. (CUDA pageable H2D async is
            // host-synchronous w.r.t. the source, so the per-iteration mask buffer is
            // safe to free immediately.) Removes ~N redundant per-copy syncs/token.
            decode_input_set_async(dc->hidden_in, hidden_data, static_cast<std::size_t>(hidden_size) * sizeof(float));
            std::int32_t pos_val = position;
            decode_input_set_async(dc->pos_tensor, &pos_val, sizeof(std::int32_t));
            std::int32_t ple_id_val = ple_token_id;
            if (dc->ple_ids != nullptr)
                decode_input_set_async(dc->ple_ids, &ple_id_val, sizeof(std::int32_t));
            else if (dc->ple_input != nullptr && ple_data != nullptr)
                decode_input_set_async(dc->ple_input, ple_data, static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));
            for (int l = 0; l < num_layers; l++)
            {
                if (dc->kv_index[l] != nullptr)
                    decode_input_set_async(dc->kv_index[l], &pwrite[l], sizeof(std::int64_t));
                if (dc->attn_mask[l] != nullptr && !li[l].isShared)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    decode_input_set_async(dc->attn_mask[l], md.data(), md.size() * sizeof(ggml_fp16_t));
                }
            }
            if (tp_mode)
            {
                // Inputs are staged; the driver runs the segments across ranks.
                if (!dc->tp_plan.valid())
                {
                    set_last_error("Gemma4 model decode: cached graph has no tensor-parallel plan.");
                    dc->reset();
                    return 0;
                }
                dc->tp_plan.out_tensor = dc->hidden_out;
                dc->tp_plan.out_host = dc->folded ? logits_data : hidden_data;
                dc->tp_plan.out_bytes = static_cast<std::size_t>(dc->out_count) * sizeof(float);
                *tp_plan_out = &dc->tp_plan;
                clear_last_error();
                return 1;
            }

            g4_debug_check_buffers(dc->graph, "replay");
            ggml_status st = tsg::graph_compute_profiled(g_backend, dc->graph, "gemma4 model decode");
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            void* out_data = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, out_data, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }
        // Miss -> (re)build. Claim this request's slot (reset in place / evict LRU)
        // for a persistable call; otherwise drop any stale entry for this identity.
        G4DecodeCache* g4dc = nullptr;
        if (g4_persist)
        {
            if (can_persist) g4dc = &g4dc_pool().claim(g4_sig, g4_kc0);
            else             g4dc_pool().drop(g4_sig, g4_kc0);
        }


        // Create GGML context. Persist mode uses a raw no_alloc ctx kept alive in
        // g_g4dc (stable tensor addresses for capture); legacy uses the pool.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 model decode: failed to init persist ggml context."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Failed to create ggml context for Gemma4 model decode.");
                return 0;
            }
            ctx = context.value;
        }

        // Per-layer persist inputs (created in the build loop below; null in legacy).
        std::vector<ggml_tensor*> layer_kv_index(num_layers, nullptr);

        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input: either gathered in-kernel from the resident quantized table
        // (ple_gather; per-token input = the token id) or uploaded as an F32
        // buffer computed by C#'s ComputePLE (legacy path).
        const int total_ple_dim = num_layers * ple_dim;
        ggml_tensor* ple_input = nullptr;
        ggml_tensor* ple_table_t = nullptr;
        ggml_tensor* ple_ids_t = nullptr;
        ggml_tensor* ple_model_proj_t = nullptr;
        ggml_tensor* ple_model_proj_norm_t = nullptr;
        if (ple_gather)
        {
            ple_table_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_token_embd_type),
                ple_token_embd_ne0, ple_token_embd_ne1);
            ple_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            // Token-embedding component: sqrt(ple_dim) * get_rows(table, id).
            ggml_tensor* ple_tok = ggml_get_rows(ctx, ple_table_t, ple_ids_t);
            ple_tok = ggml_scale(ctx, ple_tok, sqrtf(static_cast<float>(ple_dim)));
            ple_tok = ggml_reshape_1d(ctx, ple_tok, total_ple_dim);

            if (ple_model_proj_data != nullptr && ple_model_proj_norm_data != nullptr)
            {
                // Hidden-projection component: rmsnorm((hidden @ proj)/sqrt(hidden), norm).
                ple_model_proj_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_model_proj_type),
                    ple_model_proj_ne0, ple_model_proj_ne1);
                ple_model_proj_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ple_dim);
                ggml_tensor* proj = ggml_mul_mat(ctx, ple_model_proj_t, current);   // [total_ple_dim]
                proj = ggml_scale(ctx, proj, 1.0f / sqrtf(static_cast<float>(hidden_size)));
                // Per-layer RMSNorm over ple_dim: view [total_ple_dim] as
                // [ple_dim, num_layers], norm rows, scale by the norm weight.
                ggml_tensor* proj_r = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, num_layers);
                proj_r = ggml_mul(ctx, ggml_rms_norm(ctx, proj_r, eps), ple_model_proj_norm_t);
                proj = ggml_reshape_1d(ctx, proj_r, total_ple_dim);
                // combined = (proj + tok) / sqrt(2)
                ple_input = ggml_scale(ctx, ggml_add(ctx, proj, ple_tok), 1.0f / sqrtf(2.0f));
            }
            else
            {
                ple_input = ple_tok;
            }
        }
        else if (ple_data != nullptr && ple_dim > 0)
        {
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_layers * ple_dim);
        }

        if (can_persist)
        {
            ggml_set_input(current);
            ggml_set_input(pos_tensor);
            if (ple_gather && ple_ids_t != nullptr) ggml_set_input(ple_ids_t);
            else if (ple_input != nullptr) ggml_set_input(ple_input);
        }

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;   // separate K weight (mixed-quant); null when fused
            ggml_tensor* v_w;   // separate V weight (mixed-quant); null when fused
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* gate_w;   // separate gate weight (unfused FFN); null when fused
            ggml_tensor* up_w;     // separate up weight   (unfused FFN); null when fused
            ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            // PLE
            ggml_tensor* ple_gate_w;
            ggml_tensor* ple_proj_w;
            ggml_tensor* ple_post_norm_w;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            // Mixed-quant layers carry separate K/V weights (qkv_w then holds Q
            // only). Shared layers never run their own K/V projection.
            const bool separate_qkv = (!info.isShared && k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else
            {
                lt.k_w = nullptr;
                lt.v_w = nullptr;
            }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            const bool separate_gate_up = (gate_arr != nullptr && gate_arr[l] != nullptr);
            if (separate_gate_up)
            {
                lt.gu_w = nullptr;
                lt.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type_arr[l]), gate_ne0_arr[l], gate_ne1_arr[l]);
                lt.up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type_arr[l]), up_ne0_arr[l], up_ne1_arr[l]);
            }
            else
            {
                lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
                lt.gate_w = nullptr;
                lt.up_w = nullptr;
            }
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            if (!info.isShared)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            else
            {
                lt.k_cached_t = nullptr;
                lt.v_cached_t = nullptr;
            }

            lt.k_cpy = nullptr;
            lt.v_cpy = nullptr;

            lt.ple_gate_w = nullptr;
            lt.ple_proj_w = nullptr;
            lt.ple_post_norm_w = nullptr;
            // ple_input exists in BOTH PLE modes: uploaded (ple_data) or gathered
            // in-kernel (ple_gather); the per-layer injection weights are needed
            // whenever the graph carries per-layer embeddings at all.
            if ((ple_data != nullptr || ple_gather) && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Link shared layers to donor KV tensors
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            if (info.isShared)
            {
                layers[l].k_cached_t = layers[info.kvSource].k_cached_t;
                layers[l].v_cached_t = layers[info.kvSource].v_cached_t;
            }
        }

        // Build compute graph
        ggml_tensor* hidden = current;

        // Track the active KV tensors produced by each donor layer.
        std::vector<ggml_tensor*> layer_k_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_attn_mask(num_layers, nullptr);
        std::vector<std::vector<ggml_fp16_t>> layer_attn_mask_data(num_layers);

        // Tensor-parallel cut points. `tp_partial` is the row-parallel matmul
        // whose output is only this rank's share of the sum — that buffer is what
        // the collective reduces. `tp_boundary` is the last graph node that may
        // run before the reduction; it is the reshape view of the same data, so
        // reducing `tp_partial` and resuming after `tp_boundary` are consistent.
        std::vector<ggml_tensor*> tp_partial;
        std::vector<ggml_tensor*> tp_boundary;
        if (tp_mode)
        {
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 2);
            tp_boundary.reserve(static_cast<std::size_t>(num_layers) * 2);
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];

            // 1. Attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* q_rope;
            ggml_tensor* k_full;
            ggml_tensor* v_full;

            if (!info.isShared)
            {
                // 2. QKV projection. Mixed-quant layers (lt.k_w != nullptr) run
                // three separate matmuls because Q/K/V carry different ggml
                // types and cannot share one fused weight; otherwise a single
                // fused attn_qkv matmul is sliced into Q/K/V.
                ggml_tensor* q_raw;
                ggml_tensor* k_raw;
                ggml_tensor* v_raw;
                if (lt.k_w != nullptr)
                {
                    q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim);
                    k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.k_w, normed_2d), info.kDim);
                    v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.v_w, normed_2d), info.kDim);
                }
                else
                {
                    ggml_tensor* qkv_flat = ggml_reshape_1d(ctx,
                        ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim + 2 * info.kDim);
                    q_raw = ggml_view_1d(ctx, qkv_flat, info.qDim, 0);
                    k_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                        static_cast<std::size_t>(info.qDim) * sizeof(float));
                    v_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                        static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
                }

                // Per-head Q/K norm
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, info.hd, num_heads);
                ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, info.hd, info.kvHeads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), lt.k_norm_w);

                // V norm (unweighted RMSNorm)
                ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, info.hd, info.kvHeads);
                ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

                // RoPE (use per-layer n_dims and optional freq_factors)
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, info.hd, info.kvHeads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);
                ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, info.hd, info.kvHeads, 1);
                ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
                ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
                ggml_tensor* v_write = ggml_cont(ctx, v_perm);
                if (can_persist)
                {
                    // KV write via set_rows: the write row is an I64 INPUT, so the
                    // graph topology is identical token-to-token (capturable). The
                    // attention reads a FIXED padded window [0, pwindow[l]) with an
                    // F16 mask INPUT zeroing valid positions and -inf'ing padding.
                    const int win = pwindow[l];
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    layer_kv_index[l] = kv_idx;
                    lt.k_cpy = ggml_set_rows(ctx, lt.k_cached_t, k_write, kv_idx);
                    lt.v_cpy = ggml_set_rows(ctx, lt.v_cached_t, v_write, kv_idx);
                    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win, 1, 1, 1);
                    ggml_set_input(mask);
                    layer_attn_mask[l] = mask;
                    k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, 0, win, kv_cache_type);
                    v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, 0, win, kv_cache_type);
                }
                else
                {
                const int cachePos = info.isLocal ? (position % info.cacheSize) : position;
                const int activeStart = swa_decode_window_start(
                    info.isLocal, totalSeqLen, info.attendLen, info.cacheSize);
                const int attnKvLen = flash_attn_kv_length(info.attendLen, info.cacheSize, info.hd);
                const std::size_t kv_byte_offset =
                    static_cast<std::size_t>(cachePos) * lt.k_cached_t->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cached_t,
                    info.hd, 1, info.kvHeads,
                    lt.k_cached_t->nb[1], lt.k_cached_t->nb[2], kv_byte_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cached_t,
                    info.hd, 1, info.kvHeads,
                    lt.v_cached_t->nb[1], lt.v_cached_t->nb[2], kv_byte_offset);
                lt.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                lt.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                if (flash_attn_requires_masked_padding(info.hd))
                {
                    layer_attn_mask[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                    fill_flash_attn_mask(layer_attn_mask_data[l], attnKvLen, info.attendLen);
                }
                k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
                v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
                }
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Failed to create Gemma4 KV cache views.");
                    if (can_persist) ggml_free(ctx);
                    return 0;
                }
                layer_k_full[l] = k_full;
                layer_v_full[l] = v_full;
            }
            else
            {
                // Shared layer: Q-only projection (qkv_w is just Q weight)
                ggml_tensor* q_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim);
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_flat, info.hd, num_heads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                // Use the donor layer's K/V (already computed earlier in the graph)
                int donor = info.kvSource;
                k_full = layer_k_full[donor];
                v_full = layer_v_full[donor];
                layer_attn_mask[l] = layer_attn_mask[donor];
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Shared layer has no KV data available.");
                    return 0;
                }
            }

            layer_k_full[l] = k_full;
            layer_v_full[l] = v_full;

            // Flash attention (scale=1.0 due to QK-Norm, no attention softcap)
            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* attn_out = flash_attn_ext_guarded(ctx, "Gemma4 model decode",
                q_attn, k_full, v_full, layer_attn_mask[l], 1.0f, 0.0f, 0.0f);

            // 8. O projection. Row-parallel under TP: lt.o_w holds this rank's
            // slice of the input dimension, so o_mm is a partial sum and every
            // rank has to add its neighbours' before the (non-linear) norm below.
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, info.qDim, 1);
            ggml_tensor* o_mm = ggml_mul_mat(ctx, lt.o_w, attn_flat);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, o_mm, hidden_size);
            if (tp_mode) { tp_partial.push_back(o_mm); tp_boundary.push_back(o_flat); }

            // 9. Post-attn norm + residual
            ggml_tensor* post_attn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, o_flat, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, post_attn_normed, hidden);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // 10. FFN: norm → gate_up → GELU*up → down → post_ffn_norm
            ggml_tensor* ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, hidden_size, 1);

            ggml_tensor* gate;
            ggml_tensor* up;
            std::int64_t intermediate_size;
            if (lt.gate_w != nullptr)
            {
                // Two matmuls over the mapped weights instead of one over a copy of
                // them. Bit-identical: the fused weight is only ever these two
                // concatenated along the output dimension, so row i of the fused
                // result is row i of whichever half it came from.
                intermediate_size = gate_ne1_arr[l];
                gate = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.gate_w, ffn_normed_2d), intermediate_size);
                up = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.up_w, ffn_normed_2d), intermediate_size);
            }
            else
            {
                intermediate_size = gu_ne1_arr[l] / 2;
                ggml_tensor* gu_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.gu_w, ffn_normed_2d), 2 * intermediate_size);
                gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
                up = ggml_view_1d(ctx, gu_flat, intermediate_size,
                    static_cast<std::size_t>(intermediate_size) * sizeof(float));
            }
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, gate), up);

            ggml_tensor* ffn_2d = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
            ggml_tensor* down_mm = ggml_mul_mat(ctx, lt.down_w, ffn_2d);
            ggml_tensor* down_flat = ggml_reshape_1d(ctx, down_mm, hidden_size);
            if (tp_mode) { tp_partial.push_back(down_mm); tp_boundary.push_back(down_flat); }

            // 11. Post-FFN norm + residual
            ggml_tensor* post_ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, down_flat, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, post_ffn_normed, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // 12. PLE injection (if present)
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                ggml_tensor* ple_slice = ggml_view_1d(ctx, ple_input, ple_dim,
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float));
                ggml_tensor* ple_slice_2d = ggml_reshape_2d(ctx, residual2, hidden_size, 1);
                ggml_tensor* ple_gate_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_gate_w, ple_slice_2d), ple_dim);
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);
                ggml_tensor* ple_gated_2d = ggml_reshape_2d(ctx, ple_gated, ple_dim, 1);
                ggml_tensor* ple_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated_2d), hidden_size);
                ggml_tensor* ple_normed = ggml_mul(ctx,
                    ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, ple_normed, residual2);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
            }

            // 13. Layer scalar
            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // Output: either the bare hidden state, or — when folding — the final
        // RMSNorm * output_norm, the lm_head projection and the tanh logit
        // softcap, so the 262K-vocab logits are part of the captured replay.
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out;
        ggml_tensor* out_hidden;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, hidden_size, 1);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_2d);   // [vocab, 1]
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_hidden = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_hidden);

        // Build graph: add KV cache writes first to ensure they execute before reads
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 128 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].k_cpy != nullptr)
            {
                ggml_build_forward_expand(graph, layers[l].k_cpy);
                ggml_build_forward_expand(graph, layers[l].v_cpy);
            }
        }
        ggml_build_forward_expand(graph, out_hidden);


        // Bind weight data
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
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                {
                    if (needs_upload)
                        upload_list.push_back({t, data, bytes});
                    return;
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
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
            if (lt.gate_w != nullptr)
            {
                bind_or_mark(lt.gate_w, gate_arr[l], static_cast<std::size_t>(gate_bytes_arr[l]), true);
                bind_or_mark(lt.up_w, up_arr[l], static_cast<std::size_t>(up_bytes_arr[l]), true);
            }
            else
            {
                bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            }
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);

            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (!info.isShared)
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);

            if (!info.isShared)
            {
                bind_or_mark(lt.k_cached_t, k_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached_t, v_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (layer_attn_mask[l] != nullptr && !layer_attn_mask_data[l].empty())
                    bind_or_mark(layer_attn_mask[l], layer_attn_mask_data[l].data(), layer_attn_mask_data[l].size() * sizeof(ggml_fp16_t), false);
            }

            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
        }

        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        }

        if (ple_gather)
        {
            bind_or_mark(ple_table_t, const_cast<void*>(ple_token_embd_data), static_cast<std::size_t>(ple_token_embd_bytes), true);
            if (ple_model_proj_t != nullptr)
                bind_or_mark(ple_model_proj_t, const_cast<void*>(ple_model_proj_data), static_cast<std::size_t>(ple_model_proj_bytes), true);
            if (ple_model_proj_norm_t != nullptr)
                bind_or_mark(ple_model_proj_norm_t, const_cast<void*>(static_cast<const void*>(ple_model_proj_norm_data)), static_cast<std::size_t>(ple_dim) * sizeof(float), true);
        }


        // Allocate backend buffer. Reuse a persistent compute buffer across
        // decode steps instead of allocating a fresh one every token (llama.cpp
        // amortizes this via a persistent graph allocator; we mirror that). The
        // host_read_barrier below drains the prior step's GPU work before this
        // graph runs, so reusing the buffer is race-free.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            // Stable addresses for capture/replay. Metal reuses completed
            // attention workspaces; ordinary activations retain unique slots.
            persist_buf = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
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
                set_last_error("Failed to allocate backend buffer for Gemma4 model decode.");
                return 0;
            }
        }


        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        // Upload data
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0,
                static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        if (ple_gather && ple_ids_t != nullptr)
        {
            std::int32_t ple_id_val = ple_token_id;
            ggml_backend_tensor_set(ple_ids_t, &ple_id_val, 0, sizeof(std::int32_t));
        }
        else if (ple_input != nullptr && ple_data != nullptr)
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));

        if (can_persist)
        {
            // Per-layer set_rows write rows + padding masks (the per-token inputs
            // that the reuse fast-path updates before each replay).
            for (int l = 0; l < num_layers; l++)
            {
                if (layer_kv_index[l] != nullptr)
                    ggml_backend_tensor_set(layer_kv_index[l], &pwrite[l], 0, sizeof(std::int64_t));
                if (!li[l].isShared && layer_attn_mask[l] != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    ggml_backend_tensor_set(layer_attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
                }
            }
        }


        const int g4_out_count = fold ? vocab_size : hidden_size;
        void* g4_out_data = fold ? logits_data : hidden_data;

        if (!tp_mode)
        {
            // Execute single graph
            g4_debug_check_buffers(graph, can_persist ? "build-persist" : "build-transient");
            ggml_status status = tsg::compute_graph(g_backend, graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("ggml backend graph execution failed for Gemma4 model decode.");
                if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
                return 0;
            }

            // Download hidden state (async blit on Metal in async mode), or the
            // folded logits[vocab] when the lm_head was folded into the graph.
            finalize_compute_with_download(hidden_out, g4_out_data, static_cast<std::size_t>(g4_out_count) * sizeof(float));
            // Unconditional: g4_out_data is the caller's host buffer and on Metal
            // async mode the download above is only QUEUED, so returning without a
            // drain hands the caller the previous token's bytes.
            host_read_barrier();
        }

        if (can_persist && g4dc != nullptr)
        {
            // Keep the ctx/graph/buffer + input handles alive for capture+replay,
            // in this request's own pool slot.
            g4dc->ctx = ctx;
            g4dc->buffer = persist_buf;
            g4dc->graph = graph;
            g4dc->hidden_in = current;
            g4dc->hidden_out = hidden_out;
            g4dc->pos_tensor = pos_tensor;
            // In gather mode ple_input is a computed node — only the token id is a
            // per-token input, so that's what the REUSE fast-path refreshes.
            g4dc->ple_input = ple_gather ? nullptr : ple_input;
            g4dc->ple_ids = ple_gather ? ple_ids_t : nullptr;
            g4dc->kv_index = layer_kv_index;
            g4dc->attn_mask = layer_attn_mask;
            g4dc->layer_window = pwindow;
            g4dc->sig_disc = g4_sig;
            g4dc->sig_kcache0 = g4_kc0;
            g4dc->num_layers = num_layers;
            g4dc->hidden_size = hidden_size;
            g4dc->ple_dim = ple_dim;
            g4dc->ple_gather = ple_gather;
            g4dc->folded = fold;
            g4dc->out_count = g4_out_count;
            g4dc->valid = true;
        }

        if (tp_mode)
        {
            if (g4dc == nullptr)
            {
                set_last_error("Gemma4 model decode: tensor-parallel build produced no cache slot.");
                return 0;
            }
            g4dc->tp_plan.clear();
            g4dc->tp_plan.graph = graph;
            g4dc->tp_plan.ar_tensor = tp_partial;
            if (!tp_plan_segments(g4dc->tp_plan, tp_boundary))
            {
                g4dc->reset();
                return 0;
            }
            g4dc->tp_plan.out_tensor = hidden_out;
            g4dc->tp_plan.out_host = g4_out_data;
            g4dc->tp_plan.out_bytes = static_cast<std::size_t>(g4_out_count) * sizeof(float);
            *tp_plan_out = &g4dc->tp_plan;
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
        set_last_error("Unknown error in Gemma4 model decode.");
        return 0;
    }
}

// Drop the persistent (CUDA-graph-captured) Gemma4 decode graph. The captured
// graph pins ggml-cuda's compute-pool scratch addresses and the KV-cache device
// buffers; a prefill (which grows the pool) or a KV reset/grow can move those,
// so the C# caller drops the cache before any prefill and on ResetKVCache. The
// next decode rebuilds + re-captures against the current pool state. No-op when
// persist mode is off (the cache is never populated).
TSG_EXPORT void TSGgml_Gemma4ResetDecodeCache()
{
    // Every rank's pool: a KV reset invalidates the cached graphs on all GPUs,
    // and the caller is not necessarily on the rank that built them.
    for (auto& pool : g_g4dc_pools)
        pool.reset_all();
}
