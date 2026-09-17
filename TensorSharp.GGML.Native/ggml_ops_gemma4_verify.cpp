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
#include "ggml_ops_transformer_common.h"
#include <chrono>
#include <cstdio>

using namespace tsg;

namespace
{
    // Resources a tensor-parallel verify graph borrows between "build" and
    // "execute". This kernel allocates from the context pool and (for prefill-
    // sized N) the shared gallocr, so unlike the persistent decode graph it has
    // nothing that naturally outlives the call — and under TP the caller runs the
    // graph only after every rank has built one. Each rank therefore parks its
    // context and any per-call buffer here, and the slot is recycled when that
    // rank builds its next graph. One prefill chunk's worth of scratch is held
    // between calls; the alternative is a use-after-free.
    struct G4VerifyTpPending
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
    G4VerifyTpPending g_g4v_tp[tsg::TSG_MAX_DEVICES];
}

// Release every rank's parked tensor-parallel verify graph. The C# side calls
// this when the model is torn down or the KV cache is reset.
TSG_EXPORT void TSGgml_Gemma4ReleaseVerifyTpGraphs()
{
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; ++r)
    {
        if (g_g4v_tp[r].plan.graph == nullptr && g_g4v_tp[r].context.value == nullptr)
            continue;
        tsg::ScopedRank rank(r);
        g_g4v_tp[r].reset();
    }
}

// ============================================================================
// Fused MULTI-TOKEN verify (seqLen == num_tokens > 1): runs the whole dense
// Gemma 4 transformer over a small batch of tokens [start_pos, start_pos+N) as
// ONE GGML graph — the speculative-decoding verify. The single-token decode
// kernel above (TSGgml_Gemma4ModelDecode) is the only thing fast enough to beat
// the per-op verify, and it is seqLen==1 only; this is its multi-token sibling.
//
// Supports the dense (non-MoE) trunk including per-layer embeddings (PLE) and
// shared-KV (KV-donor) layers — both ported from the decode kernel — so the
// Gemma 4 E-series (e.g. E4B) verifies on this fused path. Still enforced by the
// C# caller's gate: dense only (no MoE), and for GLOBAL layers
// total_seq_len = start_pos + N <= the cache size
// (the SWA window). The last condition means the SWA cache has NOT wrapped, so
// every query's window covers [0, total_seq_len) and attention is PURE CAUSAL —
// no circular-window gymnastics, one ggml_diag_mask_inf(start_pos) mask. The
// caller still owns the post-final-norm + LM head; this returns the per-row
// hidden state [hidden_size, N] (the layer-stack output, pre output_norm).
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4ModelVerify(
    float* hidden_data, int hidden_size, int num_layers, int num_tokens,
    void** attn_norm_arr, void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, void** post_attn_norm_arr,
    void** ffn_norm_arr, void** gu_arr, void** down_arr, void** post_ffn_norm_arr,
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr, int* kv_heads_arr, int* cache_size_arr, int* is_local_arr,
    float* rope_base_arr, float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads, int start_pos,
    float eps,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Shared-KV (KV-donor) map: kv_source_arr[l] == l for a normal layer; a
    // different layer index means layer l reads that donor's K/V (Gemma 4
    // E-series shared_kv_layers). Nullable (treated as the identity map).
    int* kv_source_arr,
    // PLE (per-layer-embedding) data, per token per layer — nullable. Layout is
    // [num_tokens, num_layers * ple_dim] row-major (the C# perLayerInputs tensor).
    float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr,
    // Multimodal bidirectional-span mask, nullable, length N (one byte per token,
    // 1 = "soft" image/audio token). When set (multimodal prefill, start_pos==0)
    // the attention mask is causal PLUS bidirectional within the soft-token spans:
    // a soft-token query may attend forward to a soft-token key. Mirrors the C#
    // per-op ApplyCausalMask exceptPositions path. Null for text / MTP verify.
    const unsigned char* is_except_arr,
    // In-kernel PLE gather. When ple_token_embd_data != nullptr, the per-layer
    // embeddings are gathered INSIDE this graph via ggml_get_rows on the resident
    // quantized per_layer_token_embd table (ple_token_ids = the chunk's N token
    // ids), instead of being computed in C# and uploaded as ple_data. This avoids
    // the device->host->device round-trip of the ~88 MB gathered PLE per chunk
    // (the PLE table is already resident; only the tiny id list is uploaded).
    // When set, ple_data is ignored. ple_token_embd is [ne0 = num_layers*ple_dim,
    // ne1 = vocab]; get_rows(table, ids) -> [num_layers*ple_dim, N], byte-identical
    // layout to the uploaded ple_data.
    const void* ple_token_embd_data, int ple_token_embd_type,
    std::int64_t ple_token_embd_ne0, std::int64_t ple_token_embd_ne1, std::int64_t ple_token_embd_bytes,
    const std::int32_t* ple_token_ids,
    // Hidden-projection PLE component (nullable): ple = (sqrt(ple_dim)*get_rows +
    // rmsnorm((hidden @ per_layer_model_proj)/sqrt(hidden), per_layer_proj_norm)) / sqrt(2).
    // per_layer_model_proj is [ne0=hidden, ne1=num_layers*ple_dim]; per_layer_proj_norm
    // is F32 [ple_dim]. When ple_proj_w_data is null the token-embedding component is
    // used alone (matches ComputePLE's pleProj==null branch).
    const void* ple_proj_w_data, int ple_proj_w_type,
    std::int64_t ple_proj_w_ne0, std::int64_t ple_proj_w_ne1, std::int64_t ple_proj_w_bytes,
    const void* ple_proj_norm_data,
    // Tensor parallelism — same contract as TSGgml_Gemma4ModelDecode: with
    // tp_degree > 1 every pointer above is THIS rank's shard, and the call
    // builds and binds the graph instead of running it, handing back a plan cut
    // at the two row-parallel projections per layer.
    int tp_degree, void** tp_plan_out,
    // Separate gate/up FFN weights — the multi-token sibling of the same option in
    // TSGgml_Gemma4ModelDecode. Fusing ffn_gate and ffn_up is a copy (a GGUF writes
    // a block alphabetically, so ffn_norm sits between them), 3.1 GB of it on
    // gemma-4-12b UD-Q4_K_XL. When gate_arr[l] != nullptr this layer runs two
    // matmuls over the mapped weights and gu_arr[l] is ignored.
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    // Folded output norm + LM head for EVERY row (nullable, the multi-row sibling
    // of the fold in TSGgml_Gemma4ModelDecode). When logits_data / lm_head_data /
    // final_norm_data are non-null and vocab_size > 0, the graph appends the
    // output RMSNorm, the lm_head matmul over all N rows and (when
    // logit_softcap > 0) the tanh softcap, writes logits[N][vocab] to logits_data
    // and returns the POST-NORM rows in hidden_data. The speculative verify is
    // the caller: its per-op tail (norm, a 262K-vocab matmul over 8 rows, softcap
    // and the host round-trips between them) was 13 ms of a 58 ms E4B verify.
    // Ignored in tensor-parallel plan mode (the plan's output is the hidden).
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
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
        {
            *tp_plan_out = nullptr;
            // Recycle this rank's previous parked graph now that the caller has
            // finished with it (execution completed before it asked for another).
            g_g4v_tp[tsg::g_active_rank].reset();
        }

        const int N = num_tokens;
        const int totalSeqLen = start_pos + N;
        const bool fold = !tp_mode && logits_data != nullptr && lm_head_data != nullptr
            && final_norm_data != nullptr && vocab_size > 0;
        if (N <= 1)
            return 0;

        // Opt-in setup attribution without the per-node profiler's loss of
        // graph fusion. The explicit compute drain only runs while profiling,
        // separating execution (including the folded head) from output copies.
        static const bool native_profile_enabled = [] {
            const char* value = std::getenv("TS_GMTP_NATIVE_PROFILE");
            return value != nullptr && value[0] == '1';
        }();
        const bool native_profile = native_profile_enabled && !tp_mode && N <= 16;
        using ProfileClock = std::chrono::steady_clock;
        const auto profile_now = [&] { return native_profile ? ProfileClock::now() : ProfileClock::time_point{}; };
        const auto profile_start = profile_now();

        // Batch (query-count) padding, Vulkan only. ggml-vulkan's quantized
        // GEMMs and flash attention both run measurably faster when the batch
        // dimension is tile-aligned: every weight GEMM here has ne11 == N, and
        // whole-prefill was ~8% slower at N=1985 than at N=2016/2048 on
        // gemma4-12B (the GEMM kernels drop from ~28 to ~24 TFLOPS, flash from
        // its aligned to its unaligned shader). llama.cpp never hits this
        // because its ubatches are 512-token aligned. Pad the query batch to a
        // multiple of 64 with dummy rows: their input embeddings are uploaded
        // as ZEROS (finite through every column-wise op — matmul / norms / GLU
        // touch each column independently), causality already excludes key
        // positions >= N from every real query row (dummy keys sit after the
        // real ones), the KV-cache writes below copy only the first N rows,
        // and only the first N rows of the output hidden are downloaded. The
        // dummy columns cost <= 63 columns of extra GEMM work (< 3% at 2k
        // tokens) and their outputs are discarded. Gated to N > 64 so the MTP
        // speculative verify (N <= 16, latency-critical) keeps its exact
        // shapes. TS_G4_VERIFY_NPAD=0 disables.
        static const bool g4v_npad_enabled = []{ const char* e = std::getenv("TS_G4_VERIFY_NPAD"); return e == nullptr || e[0] != '0'; }();
        const bool pad_batch = g4v_npad_enabled && g_backend_type == BACKEND_TYPE_VULKAN && N > 64;
        const int NQ = pad_batch ? ((N + 63) & ~63) : N;

        // The bidirectional-span mask maps view-index == logical position only at
        // start_pos==0 (nPast==0); the C# gate guarantees that for multimodal.
        const unsigned char* is_except = (start_pos == 0) ? is_except_arr : nullptr;

        struct LayerInfo { int hd; int kvHeads; int qDim; int kDim; int cacheSize; bool isLocal; bool isShared; int kvSource; };
        std::vector<LayerInfo> li(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.kvSource = (kv_source_arr != nullptr) ? kv_source_arr[l] : l;
            info.isShared = (info.kvSource != l);
            // Shared layers borrow the donor's cache size / locality (the donor
            // physically owns the K/V buffer).
            info.cacheSize = cache_size_arr[info.kvSource];
            info.isLocal = is_local_arr[info.kvSource] != 0;
            // Global (full-attention) layers use a linear cache that must cover the
            // whole sequence (the C# caller grows it via EnsureCacheCapacity). SWA
            // (local) layers use a circular window cache and are handled at any
            // length below (windowed read + wrap-aware write).
            if (!info.isLocal && totalSeqLen > info.cacheSize)
                return 0;
        }

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 model verify.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, NQ);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ);

        // Tensor-parallel cut points: the attention output projection and the FFN
        // down projection, whose outputs are this rank's partial sums.
        std::vector<ggml_tensor*> tp_partial;
        if (tp_mode)
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 2);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input: all N tokens' per-layer embeddings, laid out
        // [num_tokens, num_layers * ple_dim] row-major (matches perLayerInputs).
        // In-kernel mode gathers it here via get_rows on the resident table; the
        // uploaded-ple_data mode allocates a plain F32 tensor filled by H2D below.
        const bool ple_in_kernel = ple_token_embd_data != nullptr && ple_token_ids != nullptr && ple_dim > 0;
        const int total_ple_dim = num_layers * ple_dim;
        ggml_tensor* ple_input = nullptr;
        ggml_tensor* ple_table = nullptr;
        ggml_tensor* ple_ids = nullptr;
        ggml_tensor* ple_proj_w = nullptr;
        ggml_tensor* ple_proj_norm_w = nullptr;
        if (ple_in_kernel)
        {
            ple_table = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_token_embd_type),
                ple_token_embd_ne0, ple_token_embd_ne1);
            ple_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ);
            // Token-embedding component: sqrt(ple_dim) * get_rows(table, ids).
            // get_rows(table[ne0=total_ple_dim, ne1=vocab], ids[N]) -> [total_ple_dim, N]
            // F32, same memory layout as the uploaded ple_data.
            ggml_tensor* ple_tok = ggml_get_rows(ctx, ple_table, ple_ids);
            ple_tok = ggml_scale(ctx, ple_tok, sqrtf(static_cast<float>(ple_dim)));

            if (ple_proj_w_data != nullptr)
            {
                // Hidden-projection component: rmsnorm((hidden @ proj)/sqrt(hidden), norm).
                ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_w_type),
                    ple_proj_w_ne0, ple_proj_w_ne1);
                ple_proj_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ple_dim);
                ggml_tensor* proj = ggml_mul_mat(ctx, ple_proj_w, current);   // [total_ple_dim, N]
                proj = ggml_scale(ctx, proj, 1.0f / sqrtf(static_cast<float>(hidden_size)));
                // Per-(token,layer) RMSNorm over ple_dim: view [total_ple_dim, N] as
                // [ple_dim, num_layers*N], norm rows, scale by norm weight, reshape back.
                ggml_tensor* proj_r = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, static_cast<std::int64_t>(num_layers) * NQ);
                proj_r = ggml_mul(ctx, ggml_rms_norm(ctx, proj_r, eps), ple_proj_norm_w);
                proj = ggml_reshape_2d(ctx, proj_r, total_ple_dim, NQ);
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
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                static_cast<std::int64_t>(NQ) * num_layers * ple_dim);
        }
        const bool has_ple = ple_input != nullptr;

        struct LayerTensors {
            ggml_tensor* attn_norm_w; ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w; ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w; ggml_tensor* gu_w; ggml_tensor* down_w; ggml_tensor* post_ffn_norm_w;
            ggml_tensor* gate_w; ggml_tensor* up_w;   // set instead of gu_w when the FFN was not fused
            ggml_tensor* k_cached_t; ggml_tensor* v_cached_t;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;     // primary cache write
            ggml_tensor* k_cpy2; ggml_tensor* v_cpy2;   // wrapped tail (circular SWA write past the buffer end)
            ggml_tensor* k_prev; ggml_tensor* v_prev;   // swaPrev: F32 copy of the prev window (read before the cache write)
            ggml_tensor* ple_gate_w; ggml_tensor* ple_proj_w; ggml_tensor* ple_post_norm_w;
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
            else { lt.k_w = nullptr; lt.v_w = nullptr; }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            if (gate_arr != nullptr && gate_arr[l] != nullptr)
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
            // Shared layers borrow the donor's cache tensors (linked below); they
            // own no K/V buffer of their own.
            if (!info.isShared)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            else { lt.k_cached_t = nullptr; lt.v_cached_t = nullptr; }
            lt.k_cpy = nullptr; lt.v_cpy = nullptr; lt.k_cpy2 = nullptr; lt.v_cpy2 = nullptr;

            lt.ple_gate_w = nullptr; lt.ple_proj_w = nullptr; lt.ple_post_norm_w = nullptr;
            if (has_ple && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Link shared layers to their donor's KV cache tensors (the donor, an
        // earlier layer, writes them in this same graph).
        for (int l = 0; l < num_layers; l++)
        {
            if (li[l].isShared)
            {
                layers[l].k_cached_t = layers[li[l].kvSource].k_cached_t;
                layers[l].v_cached_t = layers[li[l].kvSource].v_cached_t;
            }
        }

        // Additive causal masks for ggml_flash_attn_ext. Attention runs through
        // flash_attn_ext (not a manual mul_mat + soft_max chain): flash is O(seq)
        // memory — it never materializes the [attendLen, N, heads] score tensor that
        // reached multiple GB at multi-thousand-token prefill (forcing a one-time
        // multi-GB gallocr growth on the first long prompt, the dominant cold-start
        // cost) — and is the proven path on Metal AND CUDA (mirrors the MoE verify
        // and the decode path; the manual chain is numerically fragile at head_dim
        // 256/512). One mask per distinct (kvLen, validLen, window), shared across
        // layers. mask[qi][ki] = 0 iff ki < validLen (real, not flash padding) AND
        // ki <= (validLen-N)+qi (causal) AND (window==0 || ki > threshold-window)
        // (sliding-window low bound). The windowed variant (window>0) serves SWA
        // prefill that attends the FRESH K/V of all N tokens (start_pos==0, N >
        // sliding window); window==0 covers the circular-cache read and global.
        // On-device causal-mask generation (CUDA, text prefill only): fill each
        // [kvLen, N] mask straight into its device buffer after allocation instead
        // of the O(N*kvLen) host fill + H2D upload. The bidirectional multimodal
        // case (is_except != nullptr) keeps the host path. dataIdx == -1 in the
        // cache marks a GPU-filled mask (no host data / no upload).
#ifdef TSG_GGML_USE_CUDA
        static const bool gpu_mask_enabled = []{ const char* e = std::getenv("TS_G4_GPU_MASK"); return e == nullptr || e[0] != '0'; }();
        const bool gpu_mask = gpu_mask_enabled && g_backend_type == BACKEND_TYPE_CUDA && is_except == nullptr;
#else
        const bool gpu_mask = false;
#endif
        struct GpuMaskFill { ggml_tensor* tensor; int kvLen; int mN; int nPast; int window; int validLen; };
        std::vector<GpuMaskFill> gpu_mask_fills;

        struct VerifyMask { int kvLen; int validLen; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<VerifyMask> mask_cache;
        std::vector<std::vector<ggml_fp16_t>> mask_data_store;
        auto get_causal_mask = [&](int kvLen, int validLen, int window) -> ggml_tensor* {
            for (auto& m : mask_cache)
                if (m.kvLen == kvLen && m.validLen == validLen && m.window == window) return m.tensor;
            // Mask rows cover the padded query batch (NQ >= N); the extra dummy
            // rows keep a non-empty in-range band (their causal threshold is past
            // validLen, so the band clamps to real keys) — never fully masked, so
            // flash attention's row softmax stays finite. nPast derives from the
            // REAL token count: validLen - N == start_pos for real rows.
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvLen, NQ);
            if (gpu_mask)
            {
                // Defer to a device-side fill after gallocr allocates mt->data.
                gpu_mask_fills.push_back({mt, kvLen, NQ, validLen - N, window, validLen});
                mask_cache.push_back({kvLen, validLen, window, mt, -1});
                return mt;
            }
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kvLen) * NQ);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            const int nPast = validLen - N;
            for (int qi = 0; qi < NQ; qi++)
            {
                const int threshold = nPast + qi;
                const int low = (window > 0) ? (threshold - window + 1) : 0;
                // Bidirectional within soft-token (image/audio) spans: a soft query
                // also keeps soft keys ahead of it. Window low-bound is not applied
                // to the bidi branch (mirrors the C# per-op exceptPositions path).
                const bool q_except = is_except != nullptr && qi < N && is_except[qi] != 0;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kvLen];
                if (!q_except)
                {
                    // Fast path (no multimodal bidi): the unmasked (zero) keys form a
                    // single contiguous band [lo, hi]: lo = sliding-window low bound,
                    // hi = min(causal threshold, last real key). Fill it analytically
                    // instead of a per-element branch + fp16 convert over [0, kvLen)
                    // — this host loop is O(N*kvLen) and at multi-thousand-token
                    // prefill it blocks the GPU (the [N,N] mask reaches 134M entries).
                    const int lo = (low > 0) ? low : 0;
                    const int hi = std::min(threshold, validLen - 1);
                    std::fill(row, row + kvLen, neg_inf);
                    if (hi >= lo && lo < kvLen)
                        std::fill(row + lo, row + std::min(hi + 1, kvLen), zero_val);
                    continue;
                }
                for (int ki = 0; ki < kvLen; ki++)
                {
                    bool causal = (ki < validLen) && (ki <= threshold) && !(window > 0 && ki < low);
                    bool bidi = q_except && ki < N && is_except[ki] != 0;
                    row[ki] = (causal || bidi) ? zero_val : neg_inf;
                }
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            mask_cache.push_back({kvLen, validLen, window, mt, idx});
            return mt;
        };

        // Tiled sliding-window attention (mirrors llama.cpp's iSWA windowed KV):
        // a SWA prefill that overflows the window (start_pos==0, N > window) attends
        // the FRESH chunk's full N keys with a sliding-window mask, but ggml flash
        // only skips the causal-FUTURE region (flash_attn_mask_to_KV_max), not the
        // sliding-window PAST, so a single full-N flash iterates [0, qpos] per query
        // and wastes ~93% of its work at long prefill (local flash was ~10% of
        // prefill GPU time). Instead split the N queries into tiles of TS_G4_SWA_TILE
        // (default 1024, a multiple of the 512 window so each tile's key slice stays
        // FATTN_KQ_STRIDE-aligned -> keeps the fast GQA flash kernel). Each query
        // tile attends only its window slice [tileStart-W, tileEnd) of the fresh K/V.
        // Bidi multimodal spans (is_except) need forward attention past a query tile
        // -> the caller falls back to the full-N flash there.
        static const bool swa_tiled = []{ const char* e = std::getenv("TS_G4_SWA_TILED"); return e == nullptr || e[0] != '0'; }();
        static const int swa_tile = []{ const char* e = std::getenv("TS_G4_SWA_TILE"); int v = e ? std::atoi(e) : 0; return (v >= 256) ? v : 1024; }();

        // Flash-attention KV alignment (Vulkan only). ggml-vulkan's FA dispatch
        // only picks the fast "aligned" shader variant when the KV length is a
        // multiple of the pipeline's block_cols (32/64 for these shapes — see
        // ggml_vk_flash_attn's `aligned`); llama.cpp always satisfies it because
        // its unified KV cache pads n_kv. Our SWA fresh-chunk/extended-buffer
        // reads used the raw length, so any chunk with N % 32 != 0 silently
        // dropped all 40 SWA layers' flash onto the slow unaligned shader
        // (~10% whole-prefill regression measured on gemma4-12B at N=1985 vs
        // N=2016/2048). Pad the flash KV length to a multiple of 64: ggml_pad
        // zero-fills the tail rows (finite) and the causal mask already marks
        // ki >= validLen as -inf, so results are unchanged. CUDA handles
        // unaligned KV without a slow path — keep other backends' graphs
        // unchanged. TS_G4_FLASH_KV_PAD=0 disables.
        static const bool flash_kv_pad_enabled = []{ const char* e = std::getenv("TS_G4_FLASH_KV_PAD"); return e == nullptr || e[0] != '0'; }();
        const bool pad_flash_kv = flash_kv_pad_enabled && g_backend_type == BACKEND_TYPE_VULKAN;
        // 64 covers every FA pipeline's block_cols on the shapes used here.
        auto flash_pad_len = [pad_flash_kv](int len) {
            return pad_flash_kv ? ((len + 63) & ~63) : len;
        };
        // Zero-pad a fresh [hd, len, kvHeads] F32 K/V tensor to padLen rows.
        auto pad_kv_rows = [ctx](ggml_tensor* t, int len, int padLen) {
            return (padLen > len) ? ggml_pad(ctx, t, 0, padLen - len, 0, 0) : t;
        };
        struct TileMask { int kLen; int qLen; int qStart; int kStart; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<TileMask> tile_mask_cache;
        auto get_window_tile_mask = [&](int kLen, int qLen, int qStart, int kStart, int window) -> ggml_tensor* {
            for (auto& m : tile_mask_cache)
                if (m.kLen == kLen && m.qLen == qLen && m.qStart == qStart && m.kStart == kStart && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kLen, qLen);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kLen) * qLen);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int qi = 0; qi < qLen; qi++)
            {
                const int gQ = qStart + qi;
                // keep global key gK in (gQ - window, gQ]  ->  ki in [gQ-window+1-kStart, gQ-kStart]
                int lo = gQ - window + 1 - kStart; if (lo < 0) lo = 0;
                int hi = gQ - kStart; if (hi > kLen - 1) hi = kLen - 1;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kLen];
                std::fill(row, row + kLen, neg_inf);
                if (hi >= lo) std::fill(row + lo, row + hi + 1, zero_val);
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            tile_mask_cache.push_back({kLen, qLen, qStart, kStart, window, mt, idx});
            return mt;
        };

        ggml_tensor* hidden = current;

        // Retain each non-shared layer's FRESH full-chunk K/V (all N positions,
        // post norm/RoPE) so a shared (KV-donor) SWA layer whose sequence exceeds
        // the circular window at start_pos==0 can attend over the donor's whole
        // chunk directly — the W-sized cache only kept the last W positions, which
        // drops keys for the early query rows. Null for shared layers / unused.
        std::vector<ggml_tensor*> layer_k_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_full(num_layers, nullptr);

        // swaPrev (start_pos>0, SWA window wrapped): retain each non-shared SWA
        // layer's gathered previous-window F32 copy so a shared (KV-donor) SWA layer
        // downstream can prepend the SAME prev window the donor used. Mirrors
        // layer_k_full but for the [start_pos-prevCount, start_pos) rolling history.
        std::vector<ggml_tensor*> layer_k_prev(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_prev(num_layers, nullptr);

        // A shared layer reads precisely its donor's prepared attention window.
        // Keep the final representation too: the previous-window gather and
        // fresh rows were already shared, but concat and cache-dtype conversion
        // were repeated for every borrower.
        struct PreparedAttentionKv {
            ggml_tensor* k = nullptr;
            ggml_tensor* v = nullptr;
            int attend_len = 0;
            int padded_len = 0;
            int mask_window = 0;
        };
        std::vector<PreparedAttentionKv> prepared_kv(num_layers);
        // Read the A/B switch per call so numerical tests can compare the
        // original and shared graphs against exactly the same resident weights.
        const char* reuse_shared_kv_env = std::getenv("TS_GMTP_REUSE_KV");
        const bool reuse_shared_kv_enabled = reuse_shared_kv_env == nullptr || reuse_shared_kv_env[0] != '0';

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];

            // attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);  // [hidden, N]

            // Q projection (+ K/V for non-shared layers). Shared (KV-donor) layers
            // project ONLY Q (qkv_w is the Q-only weight) and read the donor's K/V.
            int rope_dims = rope_n_dims_arr[l];
            ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
            // Q/K/V for this layer as [head_dim, heads, NQ].
            //
            // When the three projections were fused into one weight at load time,
            // SPLIT THE RESULT WITH STRIDED VIEWS rather than ggml_cont copies of
            // three 2-D slices. Each token is one row of [qDim + 2*kDim], so within
            // a row the head axis is already dense and only the token axis strides
            // — a legal ggml_view_3d, and the shape ggml's rms_norm/rope want
            // (contiguous ROWS, arbitrary outer stride; Metal and CUDA both gate
            // exactly on ggml_is_contiguous_rows). This is what llama.cpp does with
            // its own fused wqkv (see llama.cpp src/models/mimo2.cpp:138-140).
            //
            // The three conts they replace were the single largest source of copy
            // traffic in this graph: on gemma-4-E4B pp512 they wrote ~155 MB per
            // forward — a full extra round trip through memory for activations that
            // the very next op could have read in place.
            ggml_tensor* q_3d = nullptr;
            ggml_tensor* k_3d = nullptr;
            ggml_tensor* v_3d = nullptr;
            if (info.isShared)
            {
                // Q-only weight; K/V come from the donor layer.
                q_3d = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed), info.hd, num_heads, NQ);
            }
            else if (lt.k_w != nullptr)
            {
                q_3d = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed), info.hd, num_heads, NQ);
                k_3d = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.k_w, normed), info.hd, info.kvHeads, NQ);
                v_3d = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.v_w, normed), info.hd, info.kvHeads, NQ);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, lt.qkv_w, normed);  // [qDim+2kDim, NQ]
                const std::size_t headStride = static_cast<std::size_t>(info.hd) * sizeof(float);
                q_3d = ggml_view_3d(ctx, qkv, info.hd, num_heads, NQ,
                    headStride, qkv->nb[1], 0);
                k_3d = ggml_view_3d(ctx, qkv, info.hd, info.kvHeads, NQ,
                    headStride, qkv->nb[1], static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_3d = ggml_view_3d(ctx, qkv, info.hd, info.kvHeads, NQ,
                    headStride, qkv->nb[1], static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
            }

            // per-head Q norm + RoPE (always; Q is this layer's own)
            q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);  // [hd, num_heads, N]

            lt.k_cpy = nullptr; lt.v_cpy = nullptr; lt.k_cpy2 = nullptr; lt.v_cpy2 = nullptr;
            lt.k_prev = nullptr; lt.v_prev = nullptr;
            // Fresh post-norm/RoPE K/V for this chunk, retained so SWA layers whose
            // sequence exceeds the (circular) window at start_pos==0 can attend over
            // the whole chunk directly instead of the W-sized cache view.
            ggml_tensor* k_write = nullptr;
            ggml_tensor* v_write = nullptr;

            // swaPrev: a SWA (local) layer at start_pos>0 whose window has wrapped
            // (totalSeqLen > W) must attend up to W-1 keys from the PREVIOUS chunk(s).
            // Those live in the rolling cache right now but this chunk's own K/V write
            // (below) will overwrite them, so we gather them into an F32 copy first and
            // prepend to the fresh chunk (mirrors the swaFresh start_pos==0 case and the
            // MoE verify swaPrev). prevCount/swaBase are shared by a KV-donor and the
            // SWA layers that follow it (same cacheSize / start_pos).
            const bool swaPrev = info.isLocal && start_pos != 0 && totalSeqLen > info.cacheSize;
            const int prevCount = swaPrev ? std::min(info.cacheSize, start_pos) : 0;
            const int swaBase = start_pos - prevCount;   // logical position of the prev-window start

            if (!info.isShared)
            {
                // per-head K norm + V norm (unweighted), then RoPE on K.
                k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);
                v_3d = ggml_rms_norm(ctx, v_3d, eps);
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);  // [hd, kvHeads, NQ]

                // Write the K/V into the persistent cache. Global: linear append of
                // all N at start_pos. SWA (circular, size = window): only the LAST
                // min(N, cacheSize) positions survive — when N exceeds the window the
                // earlier positions would be immediately overwritten, so skip them
                // (writeOffsetInChunk) rather than wrapping the buffer multiple times
                // (which would overflow the cache view). The remaining run may still
                // cross the wrap point once, so it splits into up to two cpy ops.
                k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));  // [hd, N, kvHeads]
                v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));     // [hd, N, kvHeads]

                // swaPrev gather: materialise [swaBase, start_pos) from the rolling
                // cache into F32 BEFORE the write below overwrites it (the cpy is built
                // into the graph ahead of the write cpy, so on the single execution
                // stream it reads the OLD cache contents). Retained in layer_k_prev so a
                // following shared SWA layer reuses the donor's prev window.
                if (swaPrev && prevCount > 0)
                {
                    ggml_tensor* kpv = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, swaBase, prevCount, kv_cache_type, /*fattn_query_rows=*/0);
                    ggml_tensor* vpv = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, swaBase, prevCount, kv_cache_type, /*fattn_query_rows=*/0);
                    if (kpv == nullptr || vpv == nullptr)
                    {
                        set_last_error("Gemma4 model verify: failed to view prev SWA window.");
                        return 0;
                    }
                    ggml_tensor* kpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, prevCount, info.kvHeads);
                    ggml_tensor* vpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, prevCount, info.kvHeads);
                    lt.k_prev = ggml_cpy(ctx, kpv, kpf);
                    lt.v_prev = ggml_cpy(ctx, vpv, vpf);
                    layer_k_prev[l] = lt.k_prev;
                    layer_v_prev[l] = lt.v_prev;
                }

                const int writeOffsetInChunk = (info.isLocal && N > info.cacheSize) ? (N - info.cacheSize) : 0;
                const int writeLen = N - writeOffsetInChunk;                       // <= cacheSize for SWA
                const int writeStartLogical = start_pos + writeOffsetInChunk;
                const int cacheBase = info.isLocal ? (writeStartLogical % info.cacheSize) : writeStartLogical;
                const int n1 = (info.isLocal && cacheBase + writeLen > info.cacheSize) ? (info.cacheSize - cacheBase) : writeLen;
                auto writePart = [&](ggml_tensor* cache, ggml_tensor* src, int srcOff, int dstSlot, int cnt) -> ggml_tensor* {
                    ggml_tensor* s = ggml_view_3d(ctx, src, info.hd, cnt, info.kvHeads,
                        src->nb[1], src->nb[2], static_cast<std::size_t>(srcOff) * src->nb[1]);
                    ggml_tensor* d = ggml_view_3d(ctx, cache, info.hd, cnt, info.kvHeads,
                        cache->nb[1], cache->nb[2], static_cast<std::size_t>(dstSlot) * cache->nb[1]);
                    return ggml_cpy(ctx, s, d);
                };
                lt.k_cpy = writePart(lt.k_cached_t, k_write, writeOffsetInChunk, cacheBase, n1);
                lt.v_cpy = writePart(lt.v_cached_t, v_write, writeOffsetInChunk, cacheBase, n1);
                if (n1 < writeLen)
                {
                    lt.k_cpy2 = writePart(lt.k_cached_t, k_write, writeOffsetInChunk + n1, 0, writeLen - n1);
                    lt.v_cpy2 = writePart(lt.v_cached_t, v_write, writeOffsetInChunk + n1, 0, writeLen - n1);
                }
                // Publish the fresh full-chunk K/V so shared SWA layers can attend it.
                layer_k_full[l] = k_write;
                layer_v_full[l] = v_write;
            }

            // SWA prefill that overflows the circular window at start_pos==0: the
            // W-sized cache can't hold all N keys, so attend over the FRESH chunk
            // K/V (k_write/v_write, all N positions) with a sliding-window causal
            // mask. Only valid when start_pos==0 (the fresh chunk IS the whole
            // history) and the layer computed its own K/V (non-shared). The cache
            // still received the last W positions above for subsequent decode.
            const bool swaFresh = info.isLocal && !info.isShared && start_pos == 0
                && totalSeqLen > info.cacheSize && k_write != nullptr;

            // Same overflow case for a SHARED (KV-donor) SWA layer: its donor (a
            // non-shared SWA layer, processed earlier in this graph) retained its
            // FRESH full chunk in layer_k_full/v_full. Attend that directly with the
            // sliding-window mask instead of the donor's W-sized circular cache view
            // (which lost the early positions). This is what lets E-series models
            // (KV-donor layers) run prompts > window through the fused kernel.
            const bool swaFreshShared = info.isLocal && info.isShared && start_pos == 0
                && totalSeqLen > info.cacheSize && layer_k_full[info.kvSource] != nullptr;

            // Read the attention window. SWA overflow (swaFresh/swaFreshShared):
            // attend the FRESH full chunk [0, N) with a sliding-window mask. Else:
            // SWA reads the last min(total, W) circular-cache positions; global reads
            // [0, total) (flash-padded for head_dim 512). Attention runs through
            // ggml_flash_attn_ext below (O(seq) memory, no materialized score tensor).
            int attendLen;
            int attnKvLen;
            int maskWindow;
            ggml_tensor* k_full;
            ggml_tensor* v_full;
            const int donor = info.kvSource;
            const bool same_donor_view = reuse_shared_kv_enabled && g_backend_type == BACKEND_TYPE_METAL
                && info.isShared && donor >= 0 && donor < l
                && prepared_kv[donor].k != nullptr && prepared_kv[donor].v != nullptr
                && lt.k_cached_t == layers[donor].k_cached_t
                && lt.v_cached_t == layers[donor].v_cached_t
                && info.hd == li[donor].hd && info.kvHeads == li[donor].kvHeads
                && info.cacheSize == li[donor].cacheSize && info.isLocal == li[donor].isLocal;
            if (same_donor_view)
            {
                const auto& prepared = prepared_kv[donor];
                k_full = prepared.k;
                v_full = prepared.v;
                attendLen = prepared.attend_len;
                attnKvLen = prepared.padded_len;
                maskWindow = prepared.mask_window;
            }
            else if (swaFresh || swaFreshShared)
            {
                ggml_tensor* k_fresh = swaFresh ? k_write : layer_k_full[info.kvSource];  // [hd, NQ, kvHeads]
                ggml_tensor* v_fresh = swaFresh ? v_write : layer_v_full[info.kvSource];  // [hd, NQ, kvHeads]
                attendLen = N;                       // real keys: the fresh chunk's first N rows
                // Batch padding already leaves the fresh K/V 64-row aligned (the
                // dummy tail rows are finite and causally masked); without it,
                // zero-pad for the FA aligned-shader requirement.
                attnKvLen = (NQ > N) ? NQ : flash_pad_len(N);
                maskWindow = info.cacheSize;         // sliding window W
                k_fresh = pad_kv_rows(k_fresh, NQ, attnKvLen);
                v_fresh = pad_kv_rows(v_fresh, NQ, attnKvLen);
                // flash_attn_ext needs K/V in the cache dtype; convert the fresh F32
                // chunk when the cache is F16 (no-op when F32).
                if (kv_cache_type == GGML_TYPE_F32)
                {
                    k_full = k_fresh;
                    v_full = v_fresh;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    k_full = ggml_cpy(ctx, k_fresh, kf);
                    v_full = ggml_cpy(ctx, v_fresh, vf);
                }
            }
            else if (swaPrev && ((!info.isShared && lt.k_prev != nullptr)
                                 || (info.isShared && layer_k_prev[info.kvSource] != nullptr
                                     && layer_k_full[info.kvSource] != nullptr)))
            {
                // SWA window overflow at start_pos>0: attend the extended K/V
                // [prev window (prevCount) ++ fresh chunk (N)], covering logical
                // [swaBase, start_pos+N) in chronological order. Non-shared layers use
                // their own gathered prev window + fresh K/V; shared (KV-donor) layers
                // reuse the donor's retained prev window + fresh chunk. The buffer is in
                // chronological order, so get_causal_mask(bufLen, bufLen, W) below maps
                // query qi (causal cutoff prevCount+qi, W-wide window) correctly — no
                // explicit keyBase needed (nPast = bufLen - N = prevCount).
                ggml_tensor* k_p = info.isShared ? layer_k_prev[info.kvSource] : lt.k_prev;
                ggml_tensor* v_p = info.isShared ? layer_v_prev[info.kvSource] : lt.v_prev;
                ggml_tensor* k_f = info.isShared ? layer_k_full[info.kvSource] : k_write;
                ggml_tensor* v_f = info.isShared ? layer_v_full[info.kvSource] : v_write;
                // The fresh chunk contributes NQ rows (batch padding): keys are
                // [prevCount real ++ N real ++ NQ-N dummy]; validLen = prevCount+N
                // excludes the finite dummy tail via the causal mask.
                const int bufLen = prevCount + NQ;
                attendLen = prevCount + N;
                attnKvLen = flash_pad_len(bufLen);   // FA-alignment padding (Vulkan); mask covers the tail
                maskWindow = info.cacheSize;     // sliding window W
                ggml_tensor* kext = ggml_concat(ctx, k_p, k_f, 1);   // [hd, prevCount+NQ, kvHeads] F32
                ggml_tensor* vext = ggml_concat(ctx, v_p, v_f, 1);
                kext = pad_kv_rows(kext, bufLen, attnKvLen);
                vext = pad_kv_rows(vext, bufLen, attnKvLen);
                if (kv_cache_type == GGML_TYPE_F32)
                {
                    k_full = kext;
                    v_full = vext;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    k_full = ggml_cpy(ctx, kext, kf);
                    v_full = ggml_cpy(ctx, vext, vf);
                }
            }
            else
            {
                attendLen = info.isLocal ? std::min(totalSeqLen, info.cacheSize) : totalSeqLen;
                const int activeStart = info.isLocal ? ((totalSeqLen - attendLen) % info.cacheSize) : 0;
                // Flash attention reads a (possibly padded) KV window; padding slots
                // beyond attendLen are masked out. head_dim 512/576 requires it
                // (custom CUDA kernels); on Vulkan pad every layer so the FA
                // dispatch stays on the aligned fast shader (never-written cache
                // rows are finite — same contract the 512-dim path relies on).
                // This branch's SWA reads only occur unwrapped (activeStart==0:
                // overflow goes through swaFresh/swaPrev above), so a padded view
                // never crosses the circular boundary.
                attnKvLen = flash_attn_kv_length(attendLen, info.cacheSize, info.hd);
                if (pad_flash_kv)
                    attnKvLen = std::min(info.cacheSize, std::max(attnKvLen, flash_pad_len(attendLen)));
                maskWindow = 0;                 // window already enforced by the cache view
                k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type, N);
                v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type, N);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Failed to create Gemma4 verify KV cache views.");
                return 0;
            }
            prepared_kv[l] = {k_full, v_full, attendLen, attnKvLen, maskWindow};

            // Flash attention (Gemma scale = 1.0). q_t [hd, N, num_heads]; the mask
            // (kvLen=attnKvLen, validLen=attendLen, window=maskWindow) encodes causal
            // + sliding-window + flash-padding + multimodal-bidi semantics.
            ggml_tensor* q_t = ggml_permute(ctx, q_rope, 0, 2, 1, 3);                   // [hd, N, num_heads]
            ggml_tensor* attn_out;
            // SWA window overflow (swaFresh/swaFreshShared) attends the fresh chunk's
            // N keys with a window mask (maskWindow>0). Tile the queries so each tile
            // only reads its window slice instead of the full N (see comment above).
            // Not applied to global / circular-cache reads (maskWindow==0), the rare
            // multimodal-bidi case (handled by the full-N mask), or tiny N.
            // Only tile when there are >= 3 tiles (N > 2*tile): below that the full-N
            // flash's wasted (masked) work is small and the per-tile launch + concat
            // overhead would erase the win.
            // maskWindow > 0 implies the swaFresh/swaFreshShared branch above:
            // k_full/v_full hold the fresh chunk's N rows (+ FA-alignment padding).
            const bool use_tiled = swa_tiled && maskWindow > 0 && is_except == nullptr
                && N > 2 * swa_tile && attendLen == N;
            if (use_tiled)
            {
                ggml_tensor* acc = nullptr;
                for (int qs = 0; qs < NQ; qs += swa_tile)
                {
                    const int qe = (qs + swa_tile < NQ) ? (qs + swa_tile) : NQ;
                    const int qLen = qe - qs;
                    const int ks = (qs > maskWindow) ? (qs - maskWindow) : 0;
                    // Pad each tile's key slice for FA alignment (Vulkan). ks is
                    // 64-aligned (qs is a multiple of swa_tile, maskWindow of the
                    // 512/1024 window), so the padded slice stays within the padded
                    // parent [0, attnKvLen); rows beyond the real qe - ks keys are
                    // causal-future (finite) or zero padding, and the tile mask
                    // marks them -inf either way.
                    const int kLenReal = qe - ks;
                    const int kLen = std::min(flash_pad_len(kLenReal), attnKvLen - ks);
                    ggml_tensor* q_tile = ggml_view_3d(ctx, q_t, info.hd, qLen, num_heads,
                        q_t->nb[1], q_t->nb[2], static_cast<std::size_t>(qs) * q_t->nb[1]);
                    ggml_tensor* k_tile = ggml_view_3d(ctx, k_full, info.hd, kLen, info.kvHeads,
                        k_full->nb[1], k_full->nb[2], static_cast<std::size_t>(ks) * k_full->nb[1]);
                    ggml_tensor* v_tile = ggml_view_3d(ctx, v_full, info.hd, kLen, info.kvHeads,
                        v_full->nb[1], v_full->nb[2], static_cast<std::size_t>(ks) * v_full->nb[1]);
                    ggml_tensor* m_tile = get_window_tile_mask(kLen, qLen, qs, ks, maskWindow);
                    ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_tile, k_tile, v_tile, m_tile, 1.0f, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                    if (!backend_supports_op(fa))
                    {
                        set_last_error("Gemma4 model verify: tiled flash attention unsupported for this shape; use per-op path.");
                        return 0;
                    }
                    acc = (acc == nullptr) ? fa : ggml_concat(ctx, acc, fa, 2);          // along the query (N) dim
                }
                attn_out = acc;                                                          // [hd, num_heads, N]
            }
            else
            {
                ggml_tensor* fa_mask = get_causal_mask(attnKvLen, attendLen, maskWindow);
                attn_out = ggml_flash_attn_ext(ctx, q_t, k_full, v_full, fa_mask, 1.0f, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
                if (!backend_supports_op(attn_out))
                {
                    // No supported flash kernel for this shape (e.g. an exotic head_dim):
                    // fall back to the per-op verify rather than the numerically-fragile
                    // manual chain.
                    set_last_error("Gemma4 model verify: flash attention unsupported for this shape; use per-op path.");
                    return 0;
                }
            }
            // flash_attn_ext returns [hd, num_heads, NQ, 1] — each column holds all
            // heads contiguously, exactly what the O projection wants.
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, info.qDim, NQ);

            // O projection -> post-attn norm -> residual
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_flat);                  // [hidden, N]
            if (tp_mode) tp_partial.push_back(o_out);
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), lt.post_attn_norm_w);
            // Normed term FIRST, residual second. ggml-metal fuses the triple
            // rms_norm -> mul -> add into one kernel (ggml_metal_op_norm), but only
            // when the add's src[0] IS the mul it follows; with the residual in
            // src[0] the fusion is declined and the add costs a whole extra
            // [hidden, N] read+write. Addition is commutative elementwise, so the
            // swap is bit-identical — it is purely which operand ggml sees first.
            ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden);

            // FFN: norm -> gate_up -> gelu*up -> down -> post_ffn norm -> residual
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_hidden_split = nullptr;
            ggml_tensor* gu = nullptr;
            if (lt.gate_w != nullptr)
            {
                // Two matmuls over the mapped weights instead of one over a copy of
                // them. Bit-identical to the geglu below: the fused weight is only
                // ever these two concatenated along the output dimension, and
                // ggml_gelu/ggml_mul is what ggml_geglu computes.
                ggml_tensor* gate = ggml_mul_mat(ctx, lt.gate_w, ffn_normed);           // [ff, N]
                ggml_tensor* up = ggml_mul_mat(ctx, lt.up_w, ffn_normed);               // [ff, N]
                ffn_hidden_split = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
            }
            else
            {
                gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);                            // [2*ff, N]
            }
            // Fused GeGLU: gelu(gate) * up computed directly on the contiguous
            // [2*ff, N] tensor (gate = first half, up = second half -> non-swapped
            // ggml_geglu). Avoids two full [ff, N] ggml_cont materializations plus
            // the separate gelu/mul ops (cpy_scalar F32->F32 was ~14% of prefill
            // GPU time). Bit-identical: ggml_vec_geglu / op_gelu use the same tanh
            // gelu as the old ggml_gelu. Mirrors llama.cpp build_ffn (LLM_FFN_GELU).
            ggml_tensor* ffn_hidden = ffn_hidden_split != nullptr
                ? ffn_hidden_split
                : ggml_geglu(ctx, gu);                                                  // [ff, N]
            ggml_tensor* down = ggml_mul_mat(ctx, lt.down_w, ffn_hidden);               // [hidden, N]
            if (tp_mode) tp_partial.push_back(down);
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, post_ffn, residual1);   // normed first: fuses

            // PLE injection (mirrors Gemma4ModelDecode, batched over the N rows).
            // ple_slice is a strided view of ple_input: column i (row i) at layer l.
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                // Strided view, not a cont: ggml_mul only asks that its operands have
                // contiguous ROWS (ggml.c ggml_is_contiguous_rows -> nb[0] == type
                // size), which a column slice of ple_input satisfies. The cont was a
                // full [ple_dim, NQ] copy per layer for nothing.
                ggml_tensor* ple_slice = ggml_view_2d(ctx, ple_input, ple_dim, NQ,
                    static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float),
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float));                // [ple_dim, NQ]
                ggml_tensor* ple_gate_proj = ggml_mul_mat(ctx, lt.ple_gate_w, residual2);  // [ple_dim, N]
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);  // [ple_dim, N]
                ggml_tensor* ple_proj = ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated);       // [hidden, N]
                ggml_tensor* ple_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, ple_normed, residual2);   // normed first: fuses
            }

            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // hidden holds NQ columns; only the first N (real) columns are downloaded
        // below — they are the contiguous prefix of the buffer.
        // Folded tail: output norm on every row (what the caller downloads as the
        // per-row hidden state) and the LM head + softcap over all NQ columns.
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* out_logits = nullptr;
        if (fold)
        {
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            hidden = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);   // [hidden, NQ]
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, hidden);             // [vocab, NQ]
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, NQ);
            out_logits = ggml_cpy(ctx, logits, logits_out);
            ggml_set_output(out_logits);
        }

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, NQ);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Tiled SWA attention adds ~8 nodes (flash + concat + views + mask) per query
        // tile per local layer; budget for it so the graph never overflows.
        const int swa_tiles = (swa_tiled && NQ > swa_tile) ? ((NQ + swa_tile - 1) / swa_tile) : 1;
        // +16/layer headroom for swaPrev (gather view+cpy, concat, optional dtype cpy).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (208 + static_cast<std::size_t>(swa_tiles) * 8) + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // swaPrev gathers MUST be expanded (and thus scheduled) before the cache
        // writes: the gather reads the rolling window the write then overwrites, so
        // it has to run first on the single execution stream (mirrors the MoE verify).
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].k_prev != nullptr) ggml_build_forward_expand(graph, layers[l].k_prev);
            if (layers[l].v_prev != nullptr) ggml_build_forward_expand(graph, layers[l].v_prev);
        }
        for (int l = 0; l < num_layers; l++)
        {
            // Shared (KV-donor) layers write no K/V of their own.
            if (layers[l].k_cpy != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy);
            if (layers[l].v_cpy != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy);
            if (layers[l].k_cpy2 != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy2);
            if (layers[l].v_cpy2 != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy2);
        }
        ggml_build_forward_expand(graph, out_hidden);
        if (out_logits != nullptr)
            ggml_build_forward_expand(graph, out_logits);

        const auto profile_build_end = profile_now();

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        // g4v-timing attribution: where bind time goes (cache lookups vs
        // tensor_alloc vs fallthrough), and how many tensors were re-uploaded.
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            // 512-byte floor (was 4096): the per-head q/k norm weights are 1-2 KB,
            // and leaving them below the cacheable floor re-uploaded ~96 tiny
            // tensors per prefill call — each ggml_backend_tensor_set on Vulkan is
            // a full submit+wait, so those uploads dominated the setup phase.
            if (cacheable && bytes >= 512)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS) return;
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
            {
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
                // Shared layers reuse the donor's cache tensor (bound when the
                // donor layer is processed); binding it again would double-alloc.
                bind_or_mark(lt.k_cached_t, k_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached_t, v_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
        }

        // In-kernel PLE: bind the per_layer_token_embd table (and the projection +
        // norm weights) resident so the gather / projection read them on-device.
        if (ple_in_kernel && ple_table != nullptr)
        {
            bind_or_mark(ple_table, const_cast<void*>(ple_token_embd_data),
                static_cast<std::size_t>(ple_token_embd_bytes), true);
            if (ple_proj_w != nullptr)
                bind_or_mark(ple_proj_w, const_cast<void*>(ple_proj_w_data),
                    static_cast<std::size_t>(ple_proj_w_bytes), true);
            if (ple_proj_norm_w != nullptr)
                bind_or_mark(ple_proj_norm_w, const_cast<void*>(ple_proj_norm_data),
                    static_cast<std::size_t>(ple_dim) * sizeof(float), true);
        }


        // Allocation strategy. Small N (MTP speculative verify, N<=16) keeps the
        // retained compute buffer (unique activation slots and stable addresses,
        // with completed Metal attention workspaces shared). Large N (prefill routed through this
        // kernel) MUST use the gallocr lifetime-packing allocator: the bump
        // allocator's footprint is the SUM of every layer's N-token intermediates
        // (~31 GB at N=776 over 48 layers → OOM), whereas gallocr packs by tensor
        // lifetime so the peak is one layer's working set (~10-20x smaller). The
        // pre-bound weights / KV caches above already own buffers and are skipped
        // by both allocators.
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        }

        const auto profile_bind_end = profile_now();

        const bool useGallocr = (N > 16);
        BufferHandle buffer(nullptr);
        if (useGallocr)
        {
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model verify (gallocr).");
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx, graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model verify.");
                return 0;
            }
        }


        const auto profile_alloc_end = profile_now();
        host_read_barrier();


        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * N * sizeof(float));
        if (NQ > N)
        {
            // Zero the dummy (batch-padding) embedding rows so every value that
            // flows out of them is finite: rms_norm(0)*w == 0, matmul of a zero
            // column is zero, and the causal mask already keeps them out of
            // every real query's attention.
            const std::vector<float> zero_rows(static_cast<std::size_t>(hidden_size) * (NQ - N), 0.0f);
            ggml_backend_tensor_set(current, zero_rows.data(),
                static_cast<std::size_t>(hidden_size) * N * sizeof(float),
                zero_rows.size() * sizeof(float));
        }

        std::vector<std::int32_t> pos_vals(NQ);
        for (int i = 0; i < NQ; i++) pos_vals[i] = start_pos + i;
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(NQ) * sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0,
                static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        for (auto& m : mask_cache)
            if (m.dataIdx >= 0)   // dataIdx == -1: GPU-filled below, no host upload
                ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                    mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));
        for (auto& m : tile_mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));

#ifdef TSG_GGML_USE_CUDA
        // Generate the deferred causal masks directly in their (now-allocated)
        // device buffers, then block until they complete so the backend-stream
        // graph compute below sees them (bit-identical to the host fill).
        if (!gpu_mask_fills.empty())
        {
            for (auto& g : gpu_mask_fills)
                tsg_cuda_fill_causal_mask_f16(g.tensor->data, g.kvLen, g.mN, g.nPast, g.window, g.validLen);
            tsg_cuda_sync_stream0();
        }
#endif

        if (ple_in_kernel)
        {
            // Upload only the tiny token-id list; get_rows gathers the PLE on-device.
            if (ple_ids != nullptr)
            {
                ggml_backend_tensor_set(ple_ids, ple_token_ids, 0,
                    static_cast<std::size_t>(N) * sizeof(std::int32_t));
                if (NQ > N)
                {
                    // Dummy rows gather a valid (finite) PLE row; their columns
                    // are discarded.
                    const std::vector<std::int32_t> tail_ids(static_cast<std::size_t>(NQ - N), ple_token_ids[N - 1]);
                    ggml_backend_tensor_set(ple_ids, tail_ids.data(),
                        static_cast<std::size_t>(N) * sizeof(std::int32_t),
                        tail_ids.size() * sizeof(std::int32_t));
                }
            }
        }
        else if (ple_input != nullptr)
        {
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(N) * num_layers * ple_dim * sizeof(float));
            if (NQ > N)
            {
                const std::vector<float> zero_ple(static_cast<std::size_t>(NQ - N) * num_layers * ple_dim, 0.0f);
                ggml_backend_tensor_set(ple_input, zero_ple.data(),
                    static_cast<std::size_t>(N) * num_layers * ple_dim * sizeof(float),
                    zero_ple.size() * sizeof(float));
            }
        }


        if (tp_mode)
        {
            // Park the context and any per-call buffer so the graph stays valid
            // until the driver has run every rank's segments.
            auto& pending = g_g4v_tp[tsg::g_active_rank];
            pending.plan.clear();
            pending.plan.graph = graph;
            pending.plan.ar_tensor = tp_partial;
            if (!tp_plan_segments(pending.plan, tp_partial))
            {
                pending.reset();
                return 0;
            }
            // Every rank ends with the same replicated hidden state; only rank 0
            // copies it back, since they all share one host buffer.
            if (tsg::g_active_rank == 0)
            {
                pending.plan.out_tensor = hidden_out;
                pending.plan.out_host = hidden_data;
                pending.plan.out_bytes = static_cast<std::size_t>(hidden_size) * N * sizeof(float);
            }
            pending.context = std::move(context);
            pending.buffer = std::move(buffer);
            pending.ephemeral = std::move(ephemeral_bufs);
            *tp_plan_out = &pending.plan;
            clear_last_error();
            return 1;
        }

        const auto profile_upload_end = profile_now();
        ggml_status status = tsg::graph_compute_profiled(g_backend, graph, "gemma4 model verify");
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 model verify.");
            return 0;
        }

        if (native_profile)
            sync_backend(g_backend);
        const auto profile_compute_end = profile_now();


        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(hidden_size) * N * sizeof(float));
        if (fold)
        {
            // The first N rows of logits_out are a contiguous prefix, like hidden_out.
            finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * N * sizeof(float));
            // The caller samples from logits_data as soon as we return; a queued
            // (async) download must have landed by then.
            host_read_barrier();
        }


        // If this call allocated a per-call backend buffer (the reuse-buffer
        // fallback), drain the queued async download before BufferHandle frees
        // it at scope exit — otherwise the in-flight blit reads a freed buffer
        // and Metal keeps it resident. No-op (and zero cost) on the common path
        // where the persistent reuse buffer is used (buffer.value == nullptr).
        if (buffer.value != nullptr) host_read_barrier();
        if (native_profile)
        {
            host_read_barrier();
            const auto end = ProfileClock::now();
            const auto ms = [](ProfileClock::time_point from, ProfileClock::time_point to) {
                return std::chrono::duration<double, std::milli>(to - from).count();
            };
            std::fprintf(stderr,
                "[gemma4 verify phases] rows=%d pos=%d nodes=%d uploads=%zu fold=%d "
                "ms: build=%.3f bind=%.3f alloc=%.3f upload=%.3f compute=%.3f download=%.3f total=%.3f\n",
                N, start_pos, ggml_graph_n_nodes(graph), upload_list.size(), fold ? 1 : 0,
                ms(profile_start, profile_build_end), ms(profile_build_end, profile_bind_end),
                ms(profile_bind_end, profile_alloc_end), ms(profile_alloc_end, profile_upload_end),
                ms(profile_upload_end, profile_compute_end), ms(profile_compute_end, end),
                ms(profile_start, end));
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
        set_last_error("Unknown error in Gemma4 model verify.");
        return 0;
    }
}

// ============================================================================
// Fused Gemma 4 MTP draft step (the "gemma4-assistant" recurrent draft head):
// runs the whole draft head — backbone-embed + concat(h_prev) + pre-projection,
// num_dlayers Gemma blocks whose attention reads the TARGET's donor KV cache
// (no K/V of its own), output norm, draft LM head, post-projection — as ONE GGML
// graph. Without this, the C# draft alternates device matmuls with host-side
// RoPE/attention, and the per-step device↔host ping-pong makes a 4-layer head
// cost as much as a full 48-layer decode. Single query per call, so attention
// is unmasked (every cached key is in the past). Gated by the C# caller to
// fixed_pos <= donor cache size (the SWA window has not wrapped).
// Outputs: logits[vocab] (draft LM head, no softcap) and h_out[backbone]
// (post-projection — the recurrent input chaining the next draft step).
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4DraftStep(
    int token, const float* h_prev, int fixed_pos,
    int backbone, int draft_hidden, int num_dlayers, int num_heads, int vocab,
    float eps, int kv_cache_type,
    float* rope_freq_factors, int rope_freq_factors_len,
    // singleton weights
    void* tgt_tok_embd, int tte_type, std::int64_t tte_ne0, std::int64_t tte_ne1, std::int64_t tte_bytes,
    void* nextn_pre, int npre_type, std::int64_t npre_ne0, std::int64_t npre_ne1, std::int64_t npre_bytes,
    void* nextn_post, int npost_type, std::int64_t npost_ne0, std::int64_t npost_ne1, std::int64_t npost_bytes,
    void* draft_tok_embd, int dte_type, std::int64_t dte_ne0, std::int64_t dte_ne1, std::int64_t dte_bytes,
    void* output_norm_w,
    // per-layer (size num_dlayers)
    void** attn_norm_arr, void** wq_arr, int* wq_type, std::int64_t* wq_ne0, std::int64_t* wq_ne1, std::int64_t* wq_bytes,
    void** q_norm_arr, void** wo_arr, int* wo_type, std::int64_t* wo_ne0, std::int64_t* wo_ne1, std::int64_t* wo_bytes,
    void** post_attn_norm_arr, void** ffn_norm_arr,
    void** gate_arr, int* gate_type, std::int64_t* gate_ne0, std::int64_t* gate_ne1, std::int64_t* gate_bytes,
    void** up_arr, int* up_type, std::int64_t* up_ne0, std::int64_t* up_ne1, std::int64_t* up_bytes,
    void** down_arr, int* down_type, std::int64_t* down_ne0, std::int64_t* down_ne1, std::int64_t* down_bytes,
    void** post_ffw_norm_arr, float* out_scale_arr,
    int* hd_arr, int* kv_heads_arr, int* is_local_arr, float* rope_base_arr, int* rope_dims_arr,
    void** donor_k_arr, void** donor_v_arr, int* donor_cache_size_arr,
    // outputs
    float* logits_out, float* h_out)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (fixed_pos <= 0)
            return 0;
        // SWA donor caches are circular windows (handled below); a global donor's
        // linear cache must cover fixed_pos (the target's forward grows it).
        for (int l = 0; l < num_dlayers; l++)
            if (is_local_arr[l] == 0 && fixed_pos > donor_cache_size_arr[l])
                return 0;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 draft step.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* tok_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* h_prev_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, backbone);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* freq_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        ggml_tensor* tte_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tte_type), tte_ne0, tte_ne1);
        ggml_tensor* npre_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(npre_type), npre_ne0, npre_ne1);
        ggml_tensor* npost_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(npost_type), npost_ne0, npost_ne1);
        ggml_tensor* dte_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(dte_type), dte_ne0, dte_ne1);
        ggml_tensor* onorm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);

        struct DL {
            ggml_tensor* attn_norm; ggml_tensor* wq; ggml_tensor* q_norm; ggml_tensor* wo;
            ggml_tensor* post_attn_norm; ggml_tensor* ffn_norm; ggml_tensor* gate; ggml_tensor* up;
            ggml_tensor* down; ggml_tensor* post_ffw_norm; ggml_tensor* k_cache; ggml_tensor* v_cache;
            int hd; int kvHeads; int csize;
        };
        std::vector<DL> dl(num_dlayers);
        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            d.hd = hd_arr[l]; d.kvHeads = kv_heads_arr[l]; d.csize = donor_cache_size_arr[l];
            d.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.wq = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(wq_type[l]), wq_ne0[l], wq_ne1[l]);
            d.q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.hd);
            d.wo = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(wo_type[l]), wo_ne0[l], wo_ne1[l]);
            d.post_attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.gate = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type[l]), gate_ne0[l], gate_ne1[l]);
            d.up = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type[l]), up_ne0[l], up_ne1[l]);
            d.down = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type[l]), down_ne0[l], down_ne1[l]);
            d.post_ffw_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.k_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), d.hd, d.csize, d.kvHeads);
            d.v_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), d.hd, d.csize, d.kvHeads);
        }

        // x = target.tok_embd[token] * sqrt(backbone) ; xh = concat(x, h_prev)
        ggml_tensor* x = ggml_get_rows(ctx, tte_w, tok_idx);            // [backbone]
        x = ggml_scale(ctx, x, sqrtf((float) backbone));
        ggml_tensor* x1 = ggml_reshape_1d(ctx, x, backbone);
        ggml_tensor* xh = ggml_concat(ctx, x1, h_prev_t, 0);           // [2*backbone]
        ggml_tensor* cur = ggml_mul_mat(ctx, npre_w, ggml_reshape_2d(ctx, xh, 2 * backbone, 1)); // [draft_hidden,1]
        cur = ggml_reshape_1d(ctx, cur, draft_hidden);

        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), d.attn_norm);
            ggml_tensor* q = ggml_mul_mat(ctx, d.wq, ggml_reshape_2d(ctx, normed, draft_hidden, 1)); // [num_heads*hd,1]
            ggml_tensor* q2 = ggml_reshape_2d(ctx, q, d.hd, num_heads);
            q2 = ggml_mul(ctx, ggml_rms_norm(ctx, q2, eps), d.q_norm);
            ggml_tensor* q3 = ggml_reshape_3d(ctx, q2, d.hd, num_heads, 1);
            ggml_tensor* rff = (is_local_arr[l] != 0) ? nullptr : freq_t;
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q3, pos_t, rff, rope_dims_arr[l], 2, 0, rope_base_arr[l], 1.0f, 0, 1, 0, 0);

            // SWA donor: read the last min(fixed_pos, window) positions (circular,
            // unwrapped by view_kv_cache_window). Global donor: read [0, fixed_pos).
            const int dAttendLen = (is_local_arr[l] != 0) ? std::min(fixed_pos, d.csize) : fixed_pos;
            const int dActiveStart = (is_local_arr[l] != 0) ? ((fixed_pos - dAttendLen) % d.csize) : 0;
            ggml_tensor* k_full = view_kv_cache_window(ctx, d.k_cache, d.hd, d.csize, d.kvHeads, dActiveStart, dAttendLen, kv_cache_type, /*fattn_query_rows=*/0);
            ggml_tensor* v_full = view_kv_cache_window(ctx, d.v_cache, d.hd, d.csize, d.kvHeads, dActiveStart, dAttendLen, kv_cache_type, /*fattn_query_rows=*/0);
            if (k_full == nullptr || v_full == nullptr) { set_last_error("draft donor cache view failed"); return 0; }

            ggml_tensor* q_t = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));  // [hd,1,num_heads]
            ggml_tensor* kq = ggml_mul_mat(ctx, k_full, q_t);                          // [dAttendLen,1,num_heads]
            kq = ggml_soft_max(ctx, kq);                                               // single query: all window keys valid
            ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));   // [fixed_pos,hd,kvHeads]
            ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);                             // [hd,1,num_heads]
            ggml_tensor* attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));     // [hd,num_heads,1]
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn, d.hd * num_heads, 1);

            ggml_tensor* o = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.wo, attn_flat), draft_hidden);
            o = ggml_mul(ctx, ggml_rms_norm(ctx, o, eps), d.post_attn_norm);
            ggml_tensor* attn_out = ggml_add(ctx, o, cur);   // normed first: fuses

            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, attn_out, eps), d.ffn_norm);
            ggml_tensor* fn2 = ggml_reshape_2d(ctx, fn, draft_hidden, 1);
            ggml_tensor* gate = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.gate, fn2), gate_ne1[l]);
            ggml_tensor* up = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.up, fn2), up_ne1[l]);
            ggml_tensor* fh = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
            ggml_tensor* down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.down, ggml_reshape_2d(ctx, fh, gate_ne1[l], 1)), draft_hidden);
            down = ggml_mul(ctx, ggml_rms_norm(ctx, down, eps), d.post_ffw_norm);
            ggml_tensor* res = ggml_add(ctx, down, attn_out);   // normed first: fuses

            float sc = out_scale_arr[l];
            if (std::fabs(sc - 1.0f) > 1e-6f)
                res = ggml_scale(ctx, res, sc);
            cur = res;
        }

        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), onorm_w);
        ggml_tensor* cur2 = ggml_reshape_2d(ctx, cur, draft_hidden, 1);
        ggml_tensor* logits = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, dte_w, cur2), vocab);   // [vocab]
        ggml_tensor* hnext = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, npost_w, cur2), backbone); // [backbone]

        ggml_tensor* logits_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
        ggml_tensor* hnext_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, backbone);
        ggml_tensor* logits_cpy = ggml_cpy(ctx, logits, logits_dst);
        ggml_tensor* hnext_cpy = ggml_cpy(ctx, hnext, hnext_dst);
        ggml_set_output(logits_cpy);
        ggml_set_output(hnext_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_dlayers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, logits_cpy);
        ggml_build_forward_expand(graph, hnext_cpy);

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
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
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

        bind_or_mark(tte_w, tgt_tok_embd, static_cast<std::size_t>(tte_bytes), true);
        bind_or_mark(npre_w, nextn_pre, static_cast<std::size_t>(npre_bytes), true);
        bind_or_mark(npost_w, nextn_post, static_cast<std::size_t>(npost_bytes), true);
        bind_or_mark(dte_w, draft_tok_embd, static_cast<std::size_t>(dte_bytes), true);
        bind_or_mark(onorm_w, output_norm_w, static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            bind_or_mark(d.attn_norm, attn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.wq, wq_arr[l], static_cast<std::size_t>(wq_bytes[l]), true);
            bind_or_mark(d.q_norm, q_norm_arr[l], static_cast<std::size_t>(d.hd) * sizeof(float), true);
            bind_or_mark(d.wo, wo_arr[l], static_cast<std::size_t>(wo_bytes[l]), true);
            bind_or_mark(d.post_attn_norm, post_attn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.ffn_norm, ffn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.gate, gate_arr[l], static_cast<std::size_t>(gate_bytes[l]), true);
            bind_or_mark(d.up, up_arr[l], static_cast<std::size_t>(up_bytes[l]), true);
            bind_or_mark(d.down, down_arr[l], static_cast<std::size_t>(down_bytes[l]), true);
            bind_or_mark(d.post_ffw_norm, post_ffw_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.k_cache, donor_k_arr[l], kv_cache_bytes(d.kvHeads, d.csize, d.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(d.v_cache, donor_v_arr[l], kv_cache_bytes(d.kvHeads, d.csize, d.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx, graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("Failed to allocate backend buffer for Gemma4 draft step."); return 0; }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        std::int32_t tok = token;
        ggml_backend_tensor_set(tok_idx, &tok, 0, sizeof(std::int32_t));
        ggml_backend_tensor_set(h_prev_t, h_prev, 0, static_cast<std::size_t>(backbone) * sizeof(float));
        std::int32_t posv = fixed_pos;
        ggml_backend_tensor_set(pos_t, &posv, 0, sizeof(std::int32_t));
        if (freq_t != nullptr)
            ggml_backend_tensor_set(freq_t, rope_freq_factors, 0, static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 draft step.");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_get(logits_dst, logits_out, 0, static_cast<std::size_t>(vocab) * sizeof(float));
        ggml_backend_tensor_get(hnext_dst, h_out, 0, static_cast<std::size_t>(backbone) * sizeof(float));

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
        set_last_error("Unknown error in Gemma4 draft step.");
        return 0;
    }
}
