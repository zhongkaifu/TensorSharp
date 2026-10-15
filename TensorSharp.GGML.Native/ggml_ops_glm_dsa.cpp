// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// GLM (glm-dsa) whole-model executor for the GGML backends.
//
// Like ggml_ops_deepseek4.cpp — and for the same reason — this is a
// self-contained model runtime rather than a per-op fused path: GLM-5.2 is
// 744B parameters (226 GiB at IQ2_XXS), so its weights have to be placed
// across every visible GPU and every token has to be one graph submission.
// A per-op managed forward would spend more time in host round trips than in
// arithmetic.
//
// The graph is a port of llama.cpp's `src/models/glm-dsa.cpp`:
//   * MLA with the absorption optimisation. The cache holds ONE compressed row
//     per token (kv_lora_rank + n_rot = 576 halfs), and the per-head K/V
//     decompression is folded into the query (wk_b) and the output (wv_b), so
//     64 query heads attend to a single key head.
//   * A DeepSeek-style "lightning indexer": a 32-head / 128-dim scorer over the
//     whole cache whose top-k (2048) selects which cached tokens the real
//     attention may see. GLM computes it on ~1 layer in 4 and shares the
//     selection with the layers in between (glm-dsa.attention.indexer.types).
//   * Sigmoid-gated MoE with a selection bias, weight renormalisation, a x2.5
//     routed scale, and one always-on shared expert. The leading
//     `leading_dense_block_count` layers are dense SwiGLU instead.
//
// Placement, offload and caching follow the DSV4 executor:
//   * layer split across GPUs, sized against each device's free VRAM;
//   * `--n-cpu-moe` moves the leading layers' routed experts to system RAM,
//     served zero-copy from the GGUF mmap (the routed experts are 92% of this
//     checkpoint, so it is the only knob with enough range to matter);
//   * an LRU cache of built+allocated graphs keyed by shape, so decode reuses
//     one graph (and one CUDA-graph capture) token after token.
//
// The Hadamard rotation llama.cpp applies to the indexer keys IS reproduced.
// It is an orthonormal involution applied to both sides of a dot product, so in
// exact arithmetic it cancels — but the indexer key cache is F16, and rotating
// before rounding is what keeps the error even across the 128 dimensions.
// Skipping it changed which tokens the top-k picked at long context and broke
// token-for-token parity with llama.cpp; reproducing it restores it.
// ---------------------------------------------------------------------------

#include "ggml_ops_internal.h"
#include "ggml-backend-impl.h"
#include "dsv4_file_warm.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_host_moe_cache_policy.h"
#include "glm_file_strip.h"
#include "glm_joined_workers.h"
#include "glm_metadata_owner.h"
#include "glm_upload_buffer.h"
#include "glm_graph_budget.h"
#include "glm_batched_execution.h"
#include "ggml_ops_graph_budget.h"

#include "gguf.h"
#include "ggml-impl.h" // ggml_graph_view for segmented TP execution
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#endif
#include "glm_host_mapping.h"

#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tsg_glm
{

/// TS_GLM_BD_DEBUG=1 narrates the batched-decode path — which sequences are in
/// the batch, whether their graph was reused, and how far the step got. Batched
/// decode is the one place where a single step touches N sequence slots at once,
/// so when something goes wrong the first question is always which of them.
static bool bd_debug()
{
    static const bool on = []() { const char * e = getenv("TS_GLM_BD_DEBUG"); return e && atoi(e) != 0; }();
    return on;
}

static void bd_log(const char * fmt, ...)
{
    if (!bd_debug()) return;
    fprintf(stderr, "[glm-bd] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

static constexpr int MAX_GPUS = 8;
/// Sequences one batched decode step will serve. Past this the shared weight
/// read stops paying for the extra per-token attention subgraphs.
static constexpr int MAX_BATCHED_DECODE = 16;

static int64_t pad_to(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// ---------------------------------------------------------------------------
// hyper-parameters
// ---------------------------------------------------------------------------
struct glm_hparams
{
    int32_t n_layer_all = 0;     // block_count, including the NextN/MTP block(s)
    int32_t n_layer = 0;         // trunk blocks the main graph runs
    int32_t n_layer_nextn = 0;
    int32_t n_dense_lead = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_rot = 0;
    int32_t n_embd_head_k = 0;   // n_embd_head_k_mla
    int32_t n_embd_head_v = 0;   // n_embd_head_v_mla
    int32_t n_nope = 0;          // n_embd_head_k - n_rot
    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t n_kv_row = 0;        // kv_lora_rank + n_rot: one cache row
    int32_t n_ff = 0;            // dense feed_forward_length
    int32_t n_ff_exp = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm = false;
    int32_t expert_gating_func = 2;   // 1 softmax, 2 sigmoid
    int32_t indexer_n_head = 0;
    int32_t indexer_head_size = 0;
    int32_t indexer_top_k = 0;
    std::vector<uint8_t> indexer_full;   // per trunk layer

    // --- GLM-5.3-Flash (glm5next) ------------------------------------------
    // The hybrid successor: 34 of 45 trunk layers are KDA linear attention, the
    // MLA+DSA layers are NoPE (n_rot 0), the indexer scores 4-cell pools, and
    // every residual crossing goes through Sinkhorn-constrained hyper-connections.
    bool    g5n = false;
    int32_t kda_head_dim = 0;            // 128
    int32_t kda_n_head = 0;              // n_head on KDA layers (64)
    float   kda_gate_lb = -5.0f;         // multiplicative gate lower bound
    int32_t d_conv = 0;                  // short conv kernel (4)
    int32_t indexer_kpool = 0;           // cells per pool (4); 0 = unpooled (5.2)
    int32_t hc_mult = 0;                 // hyper-connection stream count (4)
    int32_t hc_sinkhorn = 20;
    float   hc_eps = 1e-6f;
    float   swiglu_clamp = 0.0f;         // 0 = no clamp
    float   norm_eps = 0.0f;             // indexer k_norm LayerNorm eps (1e-6; 0 for glm-dsa)
    std::vector<uint8_t> is_recr;        // per trunk layer: KDA (1) vs MLA+DSA (0)
    float   rms_eps = 1e-5f;
    float   rope_freq_base = 10000.0f;
    int32_t n_ctx_train = 0;
    int32_t n_vocab = 0;

    float kq_scale() const { return 1.0f / sqrtf((float) n_embd_head_k); }
};

// One rank's view of a layer. In layer-split mode only rank 0 exists and holds
// the whole layer; under tensor parallelism every rank has its own set, with the
// head-sharded and expert-sharded matrices sliced and everything else replicated.
struct glm_layer_weights
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq_a = nullptr;
    ggml_tensor * q_a_norm = nullptr;
    ggml_tensor * wq_b = nullptr;        // column-parallel: this rank's heads
    ggml_tensor * wkv_a_mqa = nullptr;
    ggml_tensor * kv_a_norm = nullptr;
    ggml_tensor * wk_b = nullptr;        // this rank's heads
    ggml_tensor * wv_b = nullptr;        // this rank's heads
    ggml_tensor * wo = nullptr;          // row-parallel: this rank's head columns

    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;

    ggml_tensor * ffn_gate_inp = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr;   // this rank's experts
    ggml_tensor * ffn_up_exps = nullptr;
    ggml_tensor * ffn_down_exps = nullptr;
    ggml_tensor * ffn_gate_shexp = nullptr;
    ggml_tensor * ffn_up_shexp = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;

    ggml_tensor * idx_attn_q_b = nullptr;
    ggml_tensor * idx_attn_k = nullptr;
    ggml_tensor * idx_k_norm_w = nullptr;
    ggml_tensor * idx_k_norm_b = nullptr;
    ggml_tensor * idx_proj = nullptr;

    // --- glm5next: KDA linear attention ----------------------------------
    ggml_tensor * kda_wq = nullptr;      // [n_embd, d_inner]
    ggml_tensor * kda_wk = nullptr;
    ggml_tensor * kda_wv = nullptr;
    ggml_tensor * kda_wo = nullptr;      // [d_inner, n_embd]
    ggml_tensor * kda_conv_q = nullptr;  // [d_conv, 1, d_inner]
    ggml_tensor * kda_conv_k = nullptr;
    ggml_tensor * kda_conv_v = nullptr;
    ggml_tensor * kda_f_a = nullptr;     // [n_embd, 128]
    ggml_tensor * kda_f_b = nullptr;     // [128, d_inner]
    ggml_tensor * kda_dt_b = nullptr;    // [d_inner]
    ggml_tensor * kda_a = nullptr;       // [n_head] (-exp(A_log))
    ggml_tensor * kda_beta = nullptr;    // [n_embd, n_head]
    ggml_tensor * kda_g_a = nullptr;     // [n_embd, 128]
    ggml_tensor * kda_g_b = nullptr;     // [128, d_inner]
    ggml_tensor * kda_o_norm = nullptr;  // [head_dim]

    // --- glm5next: Sinkhorn hyper-connections ----------------------------
    ggml_tensor * hc_attn_fn = nullptr;    // [hc*n_embd, (2+hc)*hc]
    ggml_tensor * hc_attn_scale = nullptr; // [3]
    ggml_tensor * hc_attn_base = nullptr;  // [(2+hc)*hc]
    ggml_tensor * hc_ffn_fn = nullptr;
    ggml_tensor * hc_ffn_scale = nullptr;
    ggml_tensor * hc_ffn_base = nullptr;

    // --- glm5next: pooled indexer compressor -----------------------------
    ggml_tensor * idx_comp_gate = nullptr; // [n_embd, d_idx]
    ggml_tensor * idx_comp_ape = nullptr;  // [d_idx, kpool]

    // NextN/MTP wiring. Only the trailing draft block carries these; the eh_proj
    // is replicated on every rank (it runs before the head split) and the two
    // optional tensors are absent from GLM-5.2, which shares the trunk's
    // embedding table and LM head.
    ggml_tensor * nextn_eh_proj = nullptr;
    ggml_tensor * nextn_enorm = nullptr;
    ggml_tensor * nextn_hnorm = nullptr;
    ggml_tensor * nextn_head_norm = nullptr;
    ggml_tensor * nextn_embd = nullptr;
    ggml_tensor * nextn_head = nullptr;
};

struct glm_layer
{
    glm_layer_weights w[MAX_GPUS];

    int  device = 0;          // layer-split: the device that hosts this layer
    bool cpu_moe = false;     // routed experts live in host RAM
    bool indexer_full = false;
    bool is_moe = false;
    bool recurrent = false;   // glm5next: KDA linear attention instead of MLA

    // Shard boundaries under tensor parallelism (rank r owns [first[r], first[r+1])).
    int head_first[MAX_GPUS + 1] = {};
    /// Where each rank's slice of the routed experts' hidden dimension starts.
    /// The experts are split row-wise (every rank holds a strip of every
    /// expert), not by expert id: that keeps the router's global top-k ids valid
    /// on every rank, which ggml_mul_mat_id requires — it cannot handle the
    /// duplicate ids an id-space split would have to invent for the experts a
    /// rank does not own.
    int moe_ff_first[MAX_GPUS + 1] = {};
};

/// First index owned by rank `r` when `total` items are spread over `n` ranks;
/// the remainder goes to the leading ranks, and `shard_first(total, n, n)` is
/// `total`, so a count is always `first[r+1] - first[r]`.
static int shard_first(int total, int n, int r)
{
    const int base = total / n;
    const int rem = total % n;
    return r * base + std::min(r, rem);
}

/// Same contiguous tiling, but the remainder is handed out starting at rank
/// `rot` instead of always at rank 0. The routed experts are split in whole
/// quantization blocks, and GLM-5.2 has only eight of them per expert row: at
/// tp=3 a fixed split gives 3/3/2, so one rank carries a third less weight and
/// does a third less work in every one of the 78 layers. Rotating by layer
/// index evens the totals out without breaking contiguity.
static int shard_first_rot(int total, int n, int r, int rot)
{
    const int base = total / n;
    const int rem = total % n;
    int first = base * r;
    for (int k = 0; k < r; k++)
        if (((k - rot) % n + n) % n < rem) first++;
    return first;
}

// Per-sequence caches. One row per token per layer; the indexer cache exists
// only on layers that compute a fresh top-k.
struct glm_slot
{
    // A partial recurrent-state restore cannot be reused until a successful reset.
    bool kda_restore_failed = false;
    int id = 0;
    int64_t n_past = 0;
    // [rank][layer]. Under tensor parallelism every rank keeps its own copy of
    // the (rank-independent) MLA and indexer rows, so attention never reads
    // across the split and no collective is needed to share them.
    std::vector<ggml_tensor *> kv_k[MAX_GPUS];    // [n_kv_row, n_ctx] F16
    std::vector<ggml_tensor *> idx_k[MAX_GPUS];   // [indexer_head_size, n_ctx] F16 (or null)
                                                  // glm5next: [d_idx, 2, n_ctx] key|gate pairs

    // glm5next KDA recurrent state, F32, updated in place by the graph:
    //   conv: [d_conv-1, 3*d_inner]   ssm: [head_dim, head_dim, n_head]
    std::vector<ggml_tensor *> kda_conv[MAX_GPUS];
    std::vector<ggml_tensor *> kda_ssm[MAX_GPUS];
};

struct tensor_source
{
    int shard = -1;
    size_t offset = 0;
    size_t size = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

struct shard_files
{
    std::vector<std::string> paths;
};

struct graph_inputs
{
    ggml_tensor * tokens[MAX_GPUS + 1] = {};   // I32 [nt]
    ggml_tensor * pos[MAX_GPUS + 1] = {};      // I32 [nt]
    ggml_tensor * kq_mask[MAX_GPUS + 1] = {};  // F16/F32 [n_kv, nt]
    ggml_tensor * lid_mask[MAX_GPUS + 1] = {}; // F16 [n_kv, nt] (sparse layers only)
    ggml_tensor * kv_idxs[MAX_GPUS + 1] = {};  // I64 [nt] destination cache rows
    // glm5next pooled indexer (sparse graphs only): the pool->cell map, the
    // per-query pool visibility bias, and the always-attended trailing cells of
    // each query's own (incomplete) pool with their write values (0 for a real
    // tail cell, -inf for an unused lane - a no-op write on the -inf canvas).
    ggml_tensor * pool_cells[MAX_GPUS + 1] = {};   // I32 [kpool * n_pools]
    ggml_tensor * pool_bias[MAX_GPUS + 1] = {};    // F16 (fused) / F32 [n_pools, nt]
    ggml_tensor * trail_cells[MAX_GPUS + 1] = {};  // I32 [kpool, nt]
    ggml_tensor * trail_vals[MAX_GPUS + 1] = {};   // F32 [1, kpool, nt]
    // glm5next vision: embedding rows that replace the token embeddings of
    // image-placeholder positions (embedding device only).
    ggml_tensor * embd_rows = nullptr;             // F32 [n_embd, n_ovr]
    ggml_tensor * embd_idx = nullptr;              // I64 [n_ovr]
    ggml_tensor * out_ids = nullptr;           // I32 [n_out]
    /// NextN/MTP only: the trunk hidden state of the token preceding each row,
    /// F32 [n_embd, nt]. Lives on the embedding device like inp.tokens.
    ggml_tensor * h_in[MAX_GPUS + 1] = {};
};

/// Named intermediates kept as graph outputs when TS_GLM_TRACE names layers,
/// so a cross-implementation (or cross-configuration) divergence can be walked
/// back to the first tensor that differs.
struct trace_entry
{
    std::string name;
    ggml_tensor * t;
    std::string tag;
    int layer = 0, rank = 0;
};

// Validation only. A fresh directory and a final manifest distinguish a
// complete packed capture from files left by a failed download/write. This
// deliberately has no dependency on an OS API or a cryptographic library.
struct trace_dump_state
{
    std::filesystem::path directory;
    uint64_t rows = 512, maximum_bytes = 256ULL << 20, maximum_passes = 1;
    uint64_t next_model = 0, next_pass = 0, reserved_bytes = 0;
    std::map<std::string, int> axes;
    std::mutex mutex;

    static uint64_t number(const char* name, uint64_t fallback, uint64_t maximum)
    {
        const char* value = getenv(name);
        if (!value) return fallback;
        if (!*value) throw std::runtime_error(std::string(name) + " needs a positive decimal integer");
        uint64_t n = 0;
        for (; *value; ++value) {
            if (*value < '0' || *value > '9' || n > (maximum - uint64_t(*value - '0')) / 10)
                throw std::runtime_error(std::string(name) + " exceeds its bounded positive integer range");
            n = n * 10 + uint64_t(*value - '0');
        }
        if (!n || n > maximum) throw std::runtime_error(std::string(name) + " exceeds its bounded positive integer range");
        return n;
    }

    static trace_dump_state* configured()
    {
        static auto state = []() -> std::unique_ptr<trace_dump_state> {
            const char* path = getenv("TS_GLM_TRACE_DUMP_DIR");
            if (!path || !*path) return nullptr;
            const char* trace = getenv("TS_GLM_TRACE");
            if (!trace || !*trace) throw std::runtime_error("TS_GLM_TRACE_DUMP_DIR requires explicit TS_GLM_TRACE layers");
            auto result = std::make_unique<trace_dump_state>();
            result->directory = std::filesystem::u8path(path).lexically_normal();
            if (!result->directory.is_absolute()) throw std::runtime_error("GLM trace dump directory must be absolute and fresh");
            result->rows = number("TS_GLM_TRACE_DUMP_ROWS", 512, 1024);
            result->maximum_bytes = number("TS_GLM_TRACE_DUMP_MAX_BYTES", 256ULL << 20, 1ULL << 30);
            result->maximum_passes = number("TS_GLM_TRACE_DUMP_MAX_PASSES", 1, 64);
            if (const char* axis_spec = getenv("TS_GLM_TRACE_DUMP_TOKEN_AXES")) {
                const std::string spec(axis_spec);
                size_t begin = 0;
                while (begin < spec.size()) {
                    const size_t end = spec.find(',', begin);
                    const std::string item = spec.substr(begin, end == std::string::npos ? end : end - begin);
                    const size_t equal = item.find('=');
                    if (equal == 0 || equal == std::string::npos || equal + 2 != item.size()
                        || item.back() < '0' || item.back() > '3')
                        throw std::runtime_error("GLM trace token axes require unique tag=0..3 comma-separated entries");
                    const std::string tag = item.substr(0, equal);
                    for (char c : tag)
                        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '_')
                            throw std::runtime_error("GLM trace token axis tag contains an unsupported character");
                    if (!result->axes.emplace(tag, item.back() - '0').second)
                        throw std::runtime_error("GLM trace token axis tag is duplicated");
                    if (end == std::string::npos) break;
                    begin = end + 1;
                    if (begin == spec.size()) throw std::runtime_error("GLM trace token axes have an empty trailing entry");
                }
            }
            if (!std::filesystem::create_directory(result->directory))
                throw std::runtime_error("GLM trace dump directory already exists; use a fresh directory");
            return result;
        }();
        return state.get();
    }
};

/// GLM owns its backend instances instead of using the process-wide GGML TP
/// singleton, so its segmented executor owns a communicator for those exact
/// instances as well.  The ABI is the same one used by ggml's meta backend and
/// TensorSharp's fused Gemma/Qwen paths (NCCL on CUDA, with the backend's other
/// transports behind the same entry point).
struct glm_tp_comm
{
    using init_fn_t = void * (*)(ggml_backend_t *, size_t);
    using free_fn_t = void (*)(void *);
    using allreduce_fn_t = bool (*)(void *, ggml_tensor **);

    void * ctx = nullptr;
    free_fn_t free_fn = nullptr;
    allreduce_fn_t allreduce_fn = nullptr;

    ~glm_tp_comm()
    {
        if (ctx && free_fn) free_fn(ctx);
    }

    bool init(ggml_backend_t * backends, int n)
    {
        if (!backends || n < 2 || !backends[0]) return false;
        ggml_backend_dev_t dev = ggml_backend_get_device(backends[0]);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg) return false;
        auto init_fn = reinterpret_cast<init_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
        free_fn = reinterpret_cast<free_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
        allreduce_fn = reinterpret_cast<allreduce_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
        if (!init_fn || !allreduce_fn) return false;
        ctx = init_fn(backends, (size_t) n);
        return ctx != nullptr;
    }

    bool allreduce(ggml_tensor ** tensors) const
    {
        return ctx && allreduce_fn && allreduce_fn(ctx, tensors);
    }
};

/// One token of a batched decode step: which sequence slot it belongs to, where
/// in that slot's cache it lands, and the per-token inputs whose shape depends
/// on that slot's length. Everything else in the step — every projection, the
/// router and the experts — is shared across the batch, which is the point:
/// N concurrent requests read the weights once between them instead of N times.
struct bd_token
{
    int slot_id = 0;
    int64_t p = 0;            // position of this token in its sequence
    int64_t n_kv = 0;         // cache rows the token may attend to (padded)
    bool sparse = false;      // past the indexer top-k, so the selection bites

    ggml_tensor * kv_idx[MAX_GPUS + 1] = {};    // I64 [1]  destination cache row
    ggml_tensor * kq_mask[MAX_GPUS + 1] = {};   // F16 [n_kv, 1]
    ggml_tensor * lid_mask[MAX_GPUS + 1] = {};  // F16 [n_kv, 1]
    // glm5next pooled indexer (sparse tokens only): this token's pool map,
    // pool-visibility bias and always-attended trailing-pool lanes.
    ggml_tensor * pool_cells[MAX_GPUS + 1] = {};  // I32 [n_kv]
    ggml_tensor * pool_bias[MAX_GPUS + 1] = {};   // F16 (fused) / F32 [n_pools, 1]
    ggml_tensor * trail_cells[MAX_GPUS + 1] = {}; // I32 [kpool, 1]
    ggml_tensor * trail_vals[MAX_GPUS + 1] = {};  // F32 [1, kpool, 1]
};

struct graph_build_result
{
    struct selected_expert_seam {
        tsg::HostMoeSegment segment;
        ggml_tensor* boundary = nullptr;
        ggml_backend_t backend = nullptr;
    };
    std::vector<selected_expert_seam> selected_expert_seams;
    std::unordered_map<const ggml_tensor*, size_t> selected_expert_boundaries;
    std::vector<float> selected_expert_output;
    std::vector<int32_t> selected_expert_ids;
    size_t selected_expert_completed = 0;
    bool selected_expert_failed = false;
    std::uint64_t selected_expert_error_epoch = 0;
    int64_t n_ovr = 0;   // glm5next vision-override rows this graph expects

    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    graph_budget allocation_budget; // scheduler must be freed before this member
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    /// Post-final-norm hidden state, one row per token, when want_h. This is
    /// llama.cpp's `h_nextn`: what the trunk hands the draft head, and what the
    /// draft head hands its own next step.
    ggml_tensor * h_nextn = nullptr;
    int64_t nt = 0;
    int64_t n_kv = 0;
    std::vector<int64_t> decode_row_kv;
    int64_t n_out = 0;
    bool sparse = false;
    bool want_logits = true;
    bool want_h = false;
    /// 0 = trunk, 1 = the NextN/MTP draft block. Part of the cache key: the two
    /// graphs have different shapes at the same (nt, n_kv).
    int kind = 0;
    int slot_id = 0;
    /// Non-empty when this is a batched-decode graph; one entry per token.
    std::vector<bd_token> bd;
    std::vector<trace_entry> traces;

    // Fast glm5next TP: one independently allocated graph per rank, cut at the
    // attention and routed-FFN row-parallel outputs. Rank 0 is this object and
    // the other ranks live in tp_peers; keeping them under the same cache entry
    // makes graph-shape eviction atomic across the group.
    bool tp_fused = false;
    int tp_rank = -1;
    ggml_gallocr_t galloc = nullptr;
    tsg::TpRankPlan tp_plan;
    std::vector<ggml_tensor *> tp_boundary;
    std::vector<std::unique_ptr<graph_build_result>> tp_peers;

    ~graph_build_result()
    {
        tp_peers.clear();
        if (galloc) tsg::graph_budget_gallocr_free(galloc);
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct glm_model
{
    glm_hparams hp;
    batched_execution_recorder batched_execution;
    // Additive, quiescent opt-in. No global backend is replaced. Each stream
    // and registered file range remains alive until its compact entries retire.
    bool selected_expert_seams = false;
    int selected_expert_groups = 0;
    size_t selected_expert_minimum_bytes = 0;
    std::vector<tsg::HostMoePhaseGroupMinimum> selected_expert_group_minima;
    int selected_expert_planned_rows = 0;
    struct expert_file_range {
        const void* pointer; int64_t bytes; std::string path; int64_t offset;
    };
    std::vector<expert_file_range> selected_expert_ranges;
    std::vector<const void*> registered_expert_ranges;
    uint64_t selected_expert_graphs = 0, selected_expert_calls = 0;

    ggml_backend_t backends[MAX_GPUS + 1] = {};
    // Owned execution wrappers borrow the original CUDA streams/buffer types.
    // Allocation, device identity, expert-cache and collectives keep backends[].
    ggml_backend_t projection_backends[MAX_GPUS] = {};
    // Capture the experimental policy once at load. A later environment edit
    // must not change arithmetic between already cached graph shapes.
    bool strict_recurrent_requested = false;
    bool strict_recurrent_projections = false;
    int n_gpu = 0;
    int n_backends = 0;
    ggml_threadpool_t cpu_threadpool = nullptr;

    ggml_backend_t sched_backends[MAX_GPUS + 1] = {};
    ggml_backend_buffer_type_t sched_bufts[MAX_GPUS + 1] = {};
    int n_sched_backends = 0;

    ggml_context * w_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t w_buf[MAX_GPUS + 1] = {};
    ggml_context * c_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t c_buf[MAX_GPUS + 1] = {};

    std::vector<void *> mmap_addrs;
    std::vector<std::unique_ptr<tsg_glm_host::readonly_file_mapping>> mmap_owners;
    std::vector<ggml_backend_buffer_t> mmap_bufs;
    size_t mmap_weight_bytes = 0;

    ggml_tensor * tok_embd = nullptr;
    // Rank-local copies used by the segmented TP path. tok_embd_rank[0] aliases
    // tok_embd; the remaining entries are separately loaded onto their rank's
    // GPU so every graph starts from a wholly local residual stream.
    ggml_tensor * tok_embd_rank[MAX_GPUS] = {};
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;

    std::vector<glm_layer> layers;

    /// The trailing NextN/MTP draft block, when it was requested AND the
    /// checkpoint ships it. `mtp_layer` indexes `layers` (== hp.n_layer); the
    /// trunk graph still runs exactly hp.n_layer blocks, so every loop that
    /// walks the trunk is unaffected by its presence.
    bool has_mtp = false;
    int  mtp_layer = -1;
    /// Tokens the draft block has written into its own MLA cache, per slot, is
    /// tracked by the caller through the positions it passes; this is only the
    /// high-water mark used to validate a draft that would read unwritten rows.
    bool mtp_dense_attn = true;

    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;

    /// 1 = layer split (each layer on one device). >1 = tensor parallel: every
    /// rank runs every layer with its own slice of the head- and expert-sharded
    /// matrices, and the two row-parallel outputs per layer are summed across
    /// ranks.
    int tp = 1;

    /// Which tensors the tensor-parallel mode actually shards, as a bitmask
    /// (1 = attention heads, 2 = routed experts). Both by default;
    /// TS_GLM_TP_SHARD exists so either half can be turned off to attribute a
    /// regression or measure what each contributes.
    int tp_shard = 3;
    bool tp_heads() const { return (tp_shard & 1) != 0; }
    /// Number of adjacent attention heads that must stay together so every
    /// row-parallel output-projection slice starts and ends on a quantization
    /// block boundary. GLM-5.3's 128-wide heads and K-quantized output weights
    /// make this 2; GLM-5.2's 256-wide heads leave it at 1.
    int tp_head_group = 1;
    /// Cleared when the routed experts' rows do not split on a quantization
    /// block boundary; the experts then stay whole on every rank.
    bool tp_moe_rows = true;
    bool tp_experts() const { return (tp_shard & 2) != 0 && tp_moe_rows; }

    /// GPU that hosts rank `r`. One rank per GPU in every real configuration;
    /// the wrap only happens under TS_GLM_TP_OVERSUBSCRIBE.
    int rank_device(int r) const { return n_gpu > 0 ? r % n_gpu : 0; }

    // Per-slot cache contexts/buffers, tagged with the slot they belong to so
    // freeing a sequence releases exactly its own VRAM.
    struct slot_ctx { int slot_id; ggml_context * ctx; ggml_backend_buffer_t buf; };
    std::vector<slot_ctx> slot_ctxs;

    std::map<int, std::unique_ptr<glm_slot>> slots;
    glm_slot * active_slot = nullptr;
    int next_slot_id = 0;

    /// glm5next speculative decoding: a device-resident copy of ONE slot's KDA
    /// recurrent state (conv tail + delta-net state, every rank and layer),
    /// taken right before a verify batch and copied back when part of that
    /// batch is rejected. One arena serves every slot - a speculative step
    /// snapshots and restores within the same step, so two slots never need a
    /// live copy at once, and every slot's state tensors have identical shapes
    /// (TSGgml_GlmKdaStateCapture / TSGgml_GlmKdaStateRestore).
    struct kda_snapshot
    {
        int slot_id = -1;
        int64_t n_past = 0;
        bool valid = false;
        std::vector<ggml_tensor *> conv[MAX_GPUS];   // [rank][layer], mirrors glm_slot::kda_conv
        std::vector<ggml_tensor *> ssm[MAX_GPUS];    // [rank][layer], mirrors glm_slot::kda_ssm
        std::vector<ggml_context *> ctxs;            // one per buffer type the mirrors live in
        std::vector<ggml_backend_buffer_t> bufs;

        void release()
        {
            for (auto b : bufs) if (b) tsg::graph_budget_free_buffer(b);
            for (auto c : ctxs) if (c) ggml_free(c);
            bufs.clear();
            ctxs.clear();
            for (int r = 0; r < MAX_GPUS; r++) { conv[r].clear(); ssm[r].clear(); }
            valid = false;
            slot_id = -1;
        }
    } kda_snap;

    bool flash_attn = false;
    bool fused_lid = false;      // ggml_lightning_indexer has a kernel here

    // ggml_dsv4_hc_pre/post have a kernel on this backend (glm5next only).
    // Probed at load; when false the graph builds the equivalent batched
    // mul_mat instead of bouncing the residual stream through the CPU backend.
    bool hc_native = true;

    // The high-throughput glm5next TP executor. Unsupported configurations keep
    // using the existing combined multi-backend scheduler graph.
    bool tp_fused = false;
    std::unique_ptr<glm_tp_comm> tp_comm;
    bool tp_comm_reported = false;
    std::vector<float> tp_host_stage[MAX_GPUS];

    // ggml_backend_sched's "a higher-priority backend wants this op" heuristic.
    // It is a win when a host-resident tensor is small enough to stream, and a
    // hard failure when it is not: with the routed experts offloaded, a
    // prefill-sized batch makes the scheduler try to copy 2.9 GiB of experts PER
    // LAYER onto the accelerator, and it sizes one buffer for all of them at
    // once (28 GiB here) — which simply does not fit next to the weights that
    // are already resident. So it is off whenever any layer's experts live on
    // the host; TS_GLM_OP_OFFLOAD=1 forces it back on.
    bool op_offload = true;

    // Orthonormal Walsh-Hadamard rotation applied to the indexer keys and
    // queries. It cancels out of their dot product in exact arithmetic, but the
    // indexer key CACHE is F16: rotating first spreads the outliers so the
    // rounding is even across the 128 dimensions, and llama.cpp does the same,
    // so reproducing it is what keeps long-context top-k selections identical.
    std::vector<float> hadamard_host;
    ggml_tensor * hadamard[MAX_GPUS + 1] = {};


    std::list<std::unique_ptr<graph_build_result>> graph_cache;
    int graph_cache_cap = 8;
    bool shared_graph_owner = false;

    // Slots the managed side retains for finished conversations' next turns,
    // oldest first (TSGgml_GlmSetReclaimableSlots). A graph that does not fit
    // frees them in that order before failing (glm_reclaim_for_graph); the ids
    // it freed wait in `reclaimed` for TSGgml_GlmTakeReclaimedSlots.
    std::vector<int> reclaimable;
    std::vector<int> reclaimed;

    std::vector<float> logits;
    trace_dump_state* trace_dump = nullptr; // Process diagnostic session; owns no model state.
    uint64_t trace_model_id = 0;

    // glm5next vision: embedding rows queued by the managed side to OVERRIDE the
    // token embeddings of image-placeholder positions in the NEXT prompt
    // forward. `index` is the row's position within that forward call's token
    // array; the chunking wrapper slices per ubatch. Consumed (cleared) by the
    // wrapper after the prompt completes.
    struct embd_override { int64_t index; std::vector<float> rows; };
    std::vector<embd_override> embd_ovr;

    // host scratch, refilled per ubatch
    std::vector<int32_t> h_tokens;
    std::vector<int32_t> h_pos;
    std::vector<int64_t> h_kv_idxs;
    std::vector<uint16_t> h_mask_f16;
    std::vector<float> h_mask_f32;
    std::vector<int32_t> h_out_ids;

    ~glm_model()
    {
        graph_cache.clear();
        for (int i = 0; i < n_backends; ++i)
            tsg::host_moe_expert_cache_release_backend(backends[i]);
        for (const void* pointer : registered_expert_ranges)
            tsg::host_moe_expert_cache_on_drop(pointer);
        registered_expert_ranges.clear();
        slots.clear();
        for (auto & sc : slot_ctxs)
        {
            if (sc.buf) tsg::graph_budget_free_buffer(sc.buf);
            if (sc.ctx) ggml_free(sc.ctx);
        }
        slot_ctxs.clear();
        kda_snap.release();
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (c_buf[i]) tsg::graph_budget_free_buffer(c_buf[i]);
            if (c_ctx[i]) ggml_free(c_ctx[i]);
            if (w_buf[i]) tsg::graph_budget_free_buffer(w_buf[i]);
            if (w_ctx[i]) ggml_free(w_ctx[i]);
        }
        if (!mmap_addrs.empty())
        {
            for (ggml_backend_buffer_t b : mmap_bufs)
                if (b) tsg::graph_budget_free_buffer(b);
            // CPU buffer wrappers borrow the views; release the wrappers first.
            mmap_owners.clear();
        }
        // A backend communicator refers to the backend instances, so release it
        // before the loop below destroys those instances.
        tp_comm.reset();
        for (int i = 0; i < n_gpu; ++i)
            if (projection_backends[i]) ggml_backend_free(projection_backends[i]);
        for (int i = 0; i < n_backends; i++)
            if (backends[i]) ggml_backend_free(backends[i]);
        if (cpu_threadpool) ggml_threadpool_free(cpu_threadpool);
    }
};

static void glm_setenv_default(const char * name, const char * value)
{
    if (getenv(name) != nullptr) return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

/// Initialise the communicator against the backend objects owned by this GLM
/// model. This deliberately does not touch tsg::g_device_states: the managed
/// GLM wrapper keeps a lightweight GGML CPU context alive alongside the native
/// executor, and replacing that process-wide singleton would invalidate it.
static bool init_glm_tp_comm(glm_model & m)
{
    if (m.tp < 2 || m.tp > m.n_gpu) return false;
    if (const char * ar = getenv("GGML_CUDA_ALLREDUCE"))
        if (strcmp(ar, "none") == 0) return false;

    ggml_backend_reg_t reg0 = nullptr;
    for (int r = 0; r < m.tp; r++)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(m.backends[m.rank_device(r)]);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg || (reg0 && reg != reg0)) return false;
        reg0 = reg;
    }

    // Keep decode-size residual reductions in F32 when the CUDA backend selects
    // its internal two-rank transport. That transport may use BF16 for large
    // prefill payloads, where halving transfer bytes is useful; its historical
    // 1-byte threshold unnecessarily rounded every 16-KiB decode reduction.
    // NCCL owns its separate datatype policy inside ggml-cuda.
    glm_setenv_default("GGML_CUDA_AR_BF16_THRESHOLD", "1048576");

#ifdef TSG_GGML_USE_CUDA
    const char * reg_name = reg0 ? ggml_backend_reg_name(reg0) : nullptr;
    if (reg_name && strcmp(reg_name, "CUDA") == 0 && getenv("GGML_CUDA_ALLREDUCE") == nullptr)
    {
        // The native loader takes the first N devices of the requested backend,
        // so its rank order is the CUDA ordinal order seen by this process.
        int devices[MAX_GPUS];
        for (int r = 0; r < m.tp; r++) devices[r] = r;

        if (getenv("NCCL_P2P_DISABLE") == nullptr &&
            tsg::tp_probe_cuda_peer_access(devices, m.tp) == 0)
        {
            glm_setenv_default("NCCL_P2P_DISABLE", "1");
            fprintf(stderr, "[glm] TP peer access is non-functional; NCCL will use shared-memory transport\n");
        }
        if (tsg::tp_probe_cuda_collective(devices, m.tp) == 0)
        {
            // The backend's pinned-host device pipeline is a good two-rank
            // fallback and still avoids 90 explicit tensor downloads/uploads.
            glm_setenv_default("GGML_CUDA_ALLREDUCE", m.tp == 2 ? "internal" : "none");
            fprintf(stderr, m.tp == 2
                ? "[glm] NCCL probe failed; using the CUDA backend's internal two-rank AllReduce\n"
                : "[glm] NCCL probe failed; segmented TP will use host-staged reductions\n");
        }
    }
#endif

    std::unique_ptr<glm_tp_comm> comm(new glm_tp_comm());
    if (!comm->init(m.backends, m.tp)) return false;
    m.tp_comm = std::move(comm);
    return true;
}

// ---------------------------------------------------------------------------
// GGUF helpers
// ---------------------------------------------------------------------------

static bool kv_u32(gguf_context * g, const char * key, int32_t * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(g, id))
    {
        case GGUF_TYPE_UINT32: *out = (int32_t) gguf_get_val_u32(g, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT16: *out = (int32_t) gguf_get_val_u16(g, id); return true;
        case GGUF_TYPE_UINT64: *out = (int32_t) gguf_get_val_u64(g, id); return true;
        default: return false;
    }
}

static bool kv_f32(gguf_context * g, const char * key, float * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    *out = gguf_get_val_f32(g, id);
    return true;
}

static bool kv_bool(gguf_context * g, const char * key, bool * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return false;
    *out = gguf_get_val_bool(g, id);
    return true;
}

static bool kv_arr_u32_as_u8(gguf_context * g, const char * key, std::vector<uint8_t> & out)
{
    const int64_t i = gguf_find_key(g, key);
    if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY) return false;
    const gguf_type et = gguf_get_arr_type(g, i);
    const size_t n = gguf_get_arr_n(g, i);
    out.resize(n);
    const void * data = gguf_get_arr_data(g, i);
    for (size_t j = 0; j < n; j++)
    {
        int64_t v = 0;
        switch (et)
        {
            case GGUF_TYPE_UINT8:  v = ((const uint8_t  *) data)[j]; break;
            case GGUF_TYPE_INT8:   v = ((const int8_t   *) data)[j]; break;
            case GGUF_TYPE_UINT16: v = ((const uint16_t *) data)[j]; break;
            case GGUF_TYPE_INT16:  v = ((const int16_t  *) data)[j]; break;
            case GGUF_TYPE_UINT32: v = ((const uint32_t *) data)[j]; break;
            case GGUF_TYPE_INT32:  v = ((const int32_t  *) data)[j]; break;
            default: return false;
        }
        out[j] = v != 0 ? 1 : 0;
    }
    return true;
}

static bool kv_arr_f32_first(gguf_context * g, const char * key, float * out)
{
    const int64_t i = gguf_find_key(g, key);
    if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY) return false;
    if (gguf_get_arr_type(g, i) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(g, i) == 0) return false;
    *out = ((const float *) gguf_get_arr_data(g, i))[0];
    return true;
}

static bool kv_arr_u8(gguf_context * g, const char * key, std::vector<uint8_t> & out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return false;
    const gguf_type et = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    out.resize(n);
    if (et == GGUF_TYPE_UINT32 || et == GGUF_TYPE_INT32)
    {
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) out[i] = d[i] != 0 ? 1 : 0;
        return true;
    }
    if (et == GGUF_TYPE_BOOL)
    {
        const int8_t * d = (const int8_t *) gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) out[i] = d[i] != 0 ? 1 : 0;
        return true;
    }
    return false;
}

static shard_files resolve_shards(const std::string & first, int split_count)
{
    shard_files res;
    if (split_count <= 1) { res.paths.push_back(first); return res; }

    const std::string marker = "-00001-of-";
    size_t pos = first.find(marker);
    if (pos == std::string::npos) { res.paths.push_back(first); return res; }

    for (int i = 1; i <= split_count; i++)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "-%05d-of-", i);
        std::string p = first;
        p.replace(pos, marker.size(), buf);
        res.paths.push_back(p);
    }
    return res;
}

/// One weight tensor and where its bytes come from. Sliced tensors (tensor
/// parallelism) read a sub-range of the file, which is contiguous for ne1/ne2
/// splits and strided for an ne0 split, so the loader is told both.
struct pending_weight
{
    ggml_tensor * t = nullptr;
    int shard = -1;
    size_t file_off = 0;    // absolute offset of the first byte to read
    size_t bytes = 0;       // total bytes of the (possibly sliced) tensor
    size_t src_row = 0;     // 0 = contiguous; otherwise the file stride per row
    size_t row_len = 0;     // bytes per row when src_row != 0
};

struct weight_loader
{
    glm_model & m;
    std::map<std::string, tensor_source> & sources;
    std::vector<pending_weight> pending;

    weight_loader(glm_model & model, std::map<std::string, tensor_source> & src) : m(model), sources(src) {}

    const tensor_source * find(const char * name, bool required)
    {
        auto it = sources.find(name);
        if (it == sources.end())
        {
            if (required) fprintf(stderr, "[glm] missing tensor: %s\n", name);
            return nullptr;
        }
        return &it->second;
    }

    /// The whole tensor, as stored.
    ggml_tensor * full(int device, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        ggml_set_name(t, name);
        pending.push_back({ t, src->shard, src->offset, src->size, 0, 0 });
        return t;
    }

    /// Rows [first, first+count) of dimension `dim` (1 or 2). Both are contiguous
    /// runs in the file, so this is a plain sub-range.
    ggml_tensor * slice_hi(int device, int dim, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const size_t row = ggml_row_size(src->type, src->ne[0]);
        int64_t ne[4] = { src->ne[0], src->ne[1], src->ne[2], src->ne[3] };
        size_t unit = row;                       // bytes per unit of `dim`
        if (dim == 2) unit = row * src->ne[1];
        ne[dim] = count;

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, ne[0], ne[1], ne[2], ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + (size_t) first * unit, (size_t) count * unit, 0, 0 });
        return t;
    }

    /// Rows [first, first+count) of dimension 1, for every matrix along dim 2.
    /// Each expert's rows are contiguous but the experts are not adjacent, so
    /// this reads one strided run per expert.
    ggml_tensor * slice_mid(int device, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const size_t row = ggml_row_size(src->type, src->ne[0]);
        const size_t src_stride = row * (size_t) src->ne[1];    // one whole expert
        const size_t part = row * (size_t) count;               // this rank's strip of it

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, src->ne[0], count, src->ne[2], src->ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + (size_t) first * row,
                            part * (size_t) (src->ne[2] * src->ne[3]), src_stride, part });
        return t;
    }

    /// Columns [first, first+count) of dimension 0 — the row-parallel split. The
    /// bounds must be block-aligned for the quantization type; they always are
    /// here because the split falls on attention-head boundaries and a head is a
    /// whole number of blocks.
    ggml_tensor * slice_lo(int device, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const int64_t blck = ggml_blck_size(src->type);
        if (first % blck != 0 || count % blck != 0)
        {
            tsg::report_load_refusal( "[glm] %s: a row-parallel split at [%" PRId64 ", %" PRId64 ") is not a multiple of the "
                    "%" PRId64 "-element block of %s\n", name, first, first + count, blck, ggml_type_name(src->type));
            return nullptr;
        }

        const size_t full_row = ggml_row_size(src->type, src->ne[0]);
        const size_t part_row = ggml_row_size(src->type, count);
        const size_t off_row = ggml_row_size(src->type, first);
        const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, count, src->ne[1], src->ne[2], src->ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + off_row, (size_t) rows * part_row, full_row, part_row });
        return t;
    }
};

static bool file_size_of(const char * path, size_t & bytes)
{
    unique_file file(ggml_fopen(path, "rb"));
    return tsg_glm::read_file_size(file.get(), bytes);
}

static bool check_shard_complete(const char * path, gguf_context * g, ggml_context * meta)
{
    size_t fsz = 0;
    if (!file_size_of(path, fsz)) {
        tsg::report_load_refusal("[glm] cannot open or determine the size of checkpoint shard %s\n", path);
        return false;
    }

    const size_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    size_t needed = data_off;
    const char * last = nullptr;
    for (int64_t ti = 0; ti < n_tensors; ti++)
    {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        const size_t offset = gguf_get_tensor_offset(g, ti);
        const size_t bytes = ggml_nbytes(mt);
        const size_t maximum = (std::numeric_limits<size_t>::max)();
        if (offset > maximum - data_off || bytes > maximum - data_off - offset) {
            tsg::report_load_refusal("[glm] invalid byte range for tensor %s in %s\n", name, path);
            return false;
        }
        const size_t end = data_off + offset + bytes;
        if (end > needed) { needed = end; last = name; }
    }
    if (needed <= fsz) return true;

    tsg::report_load_refusal( "[glm] %s is incomplete: the file is %zu bytes but its %" PRId64 " tensors need %zu "
                    "(%.2f GiB missing; %s is the last one). Re-download this file.\n",
            path, fsz, n_tensors, needed, (needed - fsz) / (1024.0 * 1024.0 * 1024.0), last ? last : "?");
    return false;
}

static ggml_backend_buffer_t mmap_shard(glm_model & m, const shard_files & shards, int si)
{
    if (si < 0 || si >= (int) shards.paths.size()) return nullptr;
    if (m.mmap_bufs.size() < shards.paths.size())
    {
        m.mmap_addrs.resize(shards.paths.size(), nullptr);
        m.mmap_owners.resize(shards.paths.size());
        m.mmap_bufs.resize(shards.paths.size(), nullptr);
    }
    if (m.mmap_bufs[si]) return m.mmap_bufs[si];

    auto owner = std::make_unique<tsg_glm_host::readonly_file_mapping>();
    std::string error;
    if (!owner->open(shards.paths[si], error)) {
        tsg::report_load_refusal("[glm] cannot map host weights from %s: %s\n",
                                 shards.paths[si].c_str(), error.c_str());
        return nullptr;
    }
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(owner->data(), owner->size());
    if (!buf) return nullptr;
    // WEIGHTS, not the default ANY: ggml_backend_sched only pins a node to the
    // backend that owns an input when that input's buffer is marked as weights
    // (ggml-backend.cpp, "assign nodes that use weights to the backend of the
    // weights"). Left at ANY, an offloaded layer's mul_mat_id is assigned to the
    // accelerator instead, and the scheduler copies the whole 2.9 GiB expert
    // block over the bus once per token — measured 33 s/token against 0.19 s
    // with the flag set.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.mmap_addrs[si] = owner->data();
    m.mmap_owners[si] = std::move(owner);
    m.mmap_bufs[si] = buf;
    return buf;
}

struct load_job
{
    ggml_tensor * t;
    int shard;
    size_t file_off;     // where the first byte of this chunk lives in the file
    size_t tensor_off;   // where it lands inside the tensor
    size_t len;          // bytes to move
    size_t src_row = 0;  // 0 = one contiguous read; otherwise the file stride per row
    size_t row_len = 0;  // bytes per row when src_row != 0
};

static bool upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads, size_t chunk)
{
    if (jobs.empty()) return true;
    std::sort(jobs.begin(), jobs.end(), [](const load_job & a, const load_job & b)
    {
        if (a.shard != b.shard) return a.shard < b.shard;
        return a.file_off < b.file_off;
    });

    if (n_threads < 1) n_threads = 1;

    upload_plan plan;
    try { plan = plan_upload(jobs, (size_t) n_threads, chunk); }
    catch (const upload_budget_refused & refusal) {
        fprintf(stderr, "[glm] upload host-budget preflight refused: negotiation_refusals=%zu minimum_payload=%zu; no upload started\n",
                refusal.refusals, refusal.minimum_payload);
        return false;
    }
    // Allocate/commit the complete payload before starting any worker. No worker
    // can race another for temporary budget, grow a slice, or retry a half-load.
    std::unique_ptr<upload_arena<>> arena;
    if (plan.accounted) arena = std::make_unique<upload_arena<>>(plan);
    fprintf(stderr, "[glm] upload plan: accounted=%d workers=%zu chunk=%zu staging=%zu slab=%zu aggregate_payload_capacity=%zu negotiation_refusals=%zu lazy=%d\n",
            plan.accounted ? 1 : 0, plan.workers, plan.chunk, plan.staging, plan.slab,
            plan.aggregate, plan.refusals, plan.accounted ? 0 : 1);

    std::atomic<size_t> cursor(0);
    std::atomic<size_t> worker_slot(0);
    std::atomic<bool> failed(false);

    auto work = [&](auto & staging, auto & source_slab, size_t segment_chunk, size_t slab_limit)
    {
        std::vector<unique_file> files(shards.paths.size());
        while (!failed.load(std::memory_order_relaxed))
        {
            size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) break;
            const load_job & j = jobs[i];

            auto & file = files[j.shard];
            if (!file)
            {
                file.reset(ggml_fopen(shards.paths[j.shard].c_str(), "rb"));
                if (!file)
                {
                    fprintf(stderr, "[glm] cannot open %s\n", shards.paths[j.shard].c_str());
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            FILE * f = file.get();
            for (size_t done = 0; done < j.len && !failed.load(std::memory_order_relaxed);)
            {
                const size_t bytes = upload_segment_bytes(j.len - done, segment_chunk, j.src_row, j.row_len);
                if (staging.size() < bytes) staging.resize(bytes);
                const size_t source_offset = upload_add(j.file_off,
                    j.src_row ? upload_multiply(done / j.row_len, j.src_row) : done);
                const bool ok = j.src_row == 0
                    ? read_file_range(f, source_offset, bytes, staging.data())
                    : tsg_glm::read_file_strips(f, source_offset, j.src_row, j.row_len,
                                              bytes, staging.data(), source_slab, slab_limit);
                if (!ok) {
                    fprintf(stderr, "[glm] seek/read failed for %s: %zu bytes at offset %zu of %s\n",
                            j.t->name, bytes, source_offset, shards.paths[j.shard].c_str());
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                ggml_backend_tensor_set(j.t, staging.data(), upload_add(j.tensor_off, done), bytes);
                done += bytes;
            }
        }
    };
    auto worker = [&]() {
        if (arena) {
            const size_t slot = worker_slot.fetch_add(1, std::memory_order_relaxed);
            auto staging = arena->staging(slot);
            auto source_slab = arena->slab(slot);
            work(staging, source_slab, plan.chunk, plan.slab_limit);
        } else {
            // Preserve the ordinary path's lazy allocation and original job/read
            // sizes. The unaccounted plan ticket prevents mid-load host attach.
            upload_buffer<> staging, source_slab;
            work(staging, source_slab, (std::numeric_limits<size_t>::max)(), 1024 * 1024);
        }
    };
    run_joined_workers(plan.workers, failed, worker);
    return !failed.load();
}

static bool ieq(const char * a, const char * b)
{
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b)
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) return false;
    return *a == *b;
}

// Backend registry names are compared case-insensitively: ggml spells them
// "CUDA" / "Metal" / "Vulkan", but which capitalisation a given build uses is
// not something callers should have to know.
static bool backend_matches(const char * reg_name, const char * want)
{
    if (!want || !*want) return true;
    if (!reg_name) return false;
    if (ieq(want, "CUDA"))
        return ieq(reg_name, "CUDA") || ieq(reg_name, "ROCm") || ieq(reg_name, "MUSA");
    // ggml's Metal registry calls itself "MTL"; accept the name users type too.
    if (ieq(want, "Metal"))
        return ieq(reg_name, "Metal") || ieq(reg_name, "MTL");
    return ieq(reg_name, want);
}

// llama.cpp's published GLM-5.2 indexer pattern: layers 0,1,2 are full, then
// every fourth layer from 6 on. GLM-5.0/5.1 (shorter training context) made
// every layer full. Overridden by the metadata array when the file has one.
static void resolve_indexer_types(glm_hparams & hp, gguf_context * g)
{
    hp.indexer_full.assign((size_t) hp.n_layer, 0);
    const bool pre_5_2 = hp.n_ctx_train > 0 && hp.n_ctx_train < 1048576;
    for (int il = 0; il < hp.n_layer; il++)
        hp.indexer_full[il] = pre_5_2 ? 1 : (il < 3 || ((il - 2) % 4 == 0)) ? 1 : 0;

    std::vector<uint8_t> explicit_types;
    if (kv_arr_u8(g, "glm-dsa.attention.indexer.types", explicit_types))
        for (int il = 0; il < hp.n_layer && il < (int) explicit_types.size(); il++)
            hp.indexer_full[il] = explicit_types[il];
}

// Orthonormal Walsh-Hadamard matrix (H == H^T, H*H == I), the Sylvester
// construction ggml_gen_hadamard uses.
static void gen_hadamard(std::vector<float> & out, int n)
{
    out.assign((size_t) n * n, 0.0f);
    out[0] = 1.0f / sqrtf((float) n);
    for (int s = 1; s < n; s *= 2)
    {
        for (int i = 0; i < s; i++)
        {
            for (int j = 0; j < s; j++)
            {
                const float v = out[(size_t) i * n + j];
                out[(size_t) (i + s) * n + (j    )] =  v;
                out[(size_t) (i    ) * n + (j + s)] =  v;
                out[(size_t) (i + s) * n + (j + s)] = -v;
            }
        }
    }
}

static bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

// What a sequence slot beyond the planned ones must leave free besides graph
// room (glm_slot_fits): the loader's 1 GiB driver reserve by default.
// TS_GLM_SLOT_HEADROOM_MB overrides it for those slots only.
static size_t glm_driver_reserve_bytes()
{
    size_t mb = 1024;
    if (const char * e = getenv("TS_GLM_SLOT_HEADROOM_MB"))
    {
        char * end = nullptr;
        const long long v = strtoll(e, &end, 10);
        if (end != e && !*end && v >= 0 && (unsigned long long) v <= SIZE_MAX / (1024 * 1024)) mb = (size_t) v;
    }
    return mb * 1024 * 1024;
}

// Compute-buffer bytes the cached graphs hold on device d, and the largest one.
static void glm_cached_graph_bytes(glm_model & m, int d, size_t & total, size_t & largest)
{
    total = 0;
    largest = 0;
    auto add = [&](size_t n) { total += n; largest = std::max(largest, n); };
    for (const auto & entry : m.graph_cache)
    {
        if (entry->sched) {
            auto* scheduled = entry->allocation_budget.scheduler_backend(m.sched_backends[d]);
            // A rank-local plan may omit this model device entirely. Such a
            // backend owns no scheduler payload; never pass its original
            // pointer into an exact-ID scheduler lookup.
            if (scheduled) add(ggml_backend_sched_get_buffer_size(entry->sched, scheduled));
        }
        if (entry->galloc && m.rank_device(std::max(0, entry->tp_rank)) == d)
            add(ggml_gallocr_get_buffer_size(entry->galloc, 0));
        for (const auto & peer : entry->tp_peers)
            if (peer->galloc && m.rank_device(peer->tp_rank) == d)
                add(ggml_gallocr_get_buffer_size(peer->galloc, 0));
    }
}

// A sequence slot beyond the ones the context was sized for is opportunistic:
// each holds a whole context of cache rows, taken from the memory the compute
// graphs grow into. Admit one only while every device it lands on keeps room
// for another graph as large as the largest cached one, plus the reserve.
// GLM-5.3-Flash on a 6x A40 layer split admitted a third extra slot that left
// device 0 116 MiB short of the new request's own first prefill graph, so the
// request failed instead of waiting. This is admission, not a promise: a
// sequence whose context grows toward n_ctx needs graphs this rule does not
// hold back (TS_GLM_PLAN_SLOTS sizes the context for that many slots).
static bool glm_slot_fits(glm_model & m, const std::vector<ggml_context *> & ctxs)
{
    const size_t reserve = glm_driver_reserve_bytes();
    for (int d = 0; d < m.n_gpu; d++)
    {
        if (!ctxs[(size_t) d] || !ggml_get_first_tensor(ctxs[(size_t) d])) continue;
        ggml_backend_dev_t dev = ggml_backend_get_device(m.backends[d]);
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        if (!total_b) continue;   // unknown memory: the allocation itself decides
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(m.backends[d]);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t need = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctxs[(size_t) d]); t; t = ggml_get_next_tensor(ctxs[(size_t) d], t))
            need = GGML_PAD(need, align) + ggml_backend_buft_get_alloc_size(buft, t);
        size_t graphs = 0, largest = 0;
        glm_cached_graph_bytes(m, d, graphs, largest);
        if (free_b < need || free_b - need < largest + reserve)
        {
            fprintf(stderr, "[glm] no room for another sequence slot on device %d: it needs %.0f MiB and must leave "
                    "%.0f MiB (largest cached graph %.0f + reserve %.0f); %.0f MiB free\n",
                    d, need / 1048576.0, (largest + reserve) / 1048576.0, largest / 1048576.0,
                    reserve / 1048576.0, free_b / 1048576.0);
            return false;
        }
    }
    return true;
}

// admit: apply glm_slot_fits. The load-time primary slot is the one the context
// was sized for, so it is allocated without the check.
#if defined(TSG_GGML_TEST_HOOKS)
static thread_local int glm_test_slot_alloc_stage = 0;
TSG_EXPORT void TSGgml_GlmTestSlotAllocFault(int stage) { glm_test_slot_alloc_stage = stage; }
#endif
static void glm_test_slot_alloc_fault(int stage)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (glm_test_slot_alloc_stage == stage) { glm_test_slot_alloc_stage = 0; throw std::bad_alloc(); }
#else
    (void) stage;
#endif
}
static glm_slot * slot_alloc(glm_model & m, bool admit = false)
{
    if (m.next_slot_id == std::numeric_limits<int>::max()) return nullptr;
    auto slot = std::unique_ptr<glm_slot>(new glm_slot());
    slot->id = m.next_slot_id++;
    // One extra MLA row set for the draft block. It needs no indexer cache: the
    // draft runs dense attention (llama.cpp's graph_mtp does the same), so its
    // lightning-indexer weights are never read.
    const int n_cache_layers = m.hp.n_layer + (m.has_mtp ? 1 : 0);
    for (int r = 0; r < m.tp; r++)
    {
        slot->kv_k[r].assign((size_t) n_cache_layers, nullptr);
        slot->idx_k[r].assign((size_t) n_cache_layers, nullptr);
        slot->kda_conv[r].assign((size_t) n_cache_layers, nullptr);
        slot->kda_ssm[r].assign((size_t) n_cache_layers, nullptr);
    }

    // Cache tensors live on the device that reads them, so attention never
    // crosses the split.
    std::vector<ggml_context *> ctxs((size_t) m.n_gpu + 1, nullptr);
    struct context_owner {
        std::vector<ggml_context *> & contexts;
        ~context_owner() { for (auto * context : contexts) if (context) ggml_free(context); }
    } metadata_owner{ctxs};
    glm_test_slot_alloc_fault(1);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (8 * m.tp * n_cache_layers + 16) * ggml_tensor_overhead(), nullptr, true };
        ctxs[d] = ggml_init(p);
        if (!ctxs[d]) throw std::bad_alloc();
    }

    for (int il = 0; il < n_cache_layers; il++)
    {
        const bool recr = il < m.hp.n_layer && m.layers[il].recurrent;
        for (int r = 0; r < m.tp; r++)
        {
            const int d = m.tp > 1 ? m.rank_device(r) : m.layers[il].device;
            if (recr)
            {
                // KDA linear attention keeps a fixed-size recurrent state per
                // sequence instead of KV rows: the short-conv tail and the
                // per-head delta-net state, both updated in place by the graph.
                // KDA heads are independent, so TP keeps only this rank's heads
                // instead of replicating the full recurrent state on every GPU.
                const int64_t n_head = m.tp > 1 && m.tp_heads()
                    ? m.layers[(size_t) il].head_first[r + 1] - m.layers[(size_t) il].head_first[r]
                    : m.hp.kda_n_head;
                const int64_t d_inner = (int64_t) m.hp.kda_head_dim * n_head;
                slot->kda_conv[r][il] = ggml_new_tensor_2d(ctxs[d], GGML_TYPE_F32,
                        m.hp.d_conv - 1, 3 * d_inner);
                slot->kda_ssm[r][il] = ggml_new_tensor_3d(ctxs[d], GGML_TYPE_F32,
                        m.hp.kda_head_dim, m.hp.kda_head_dim, n_head);
                ggml_format_name(slot->kda_conv[r][il], "slot%d_kconv.%d.%d", slot->id, r, il);
                ggml_format_name(slot->kda_ssm[r][il], "slot%d_kssm.%d.%d", slot->id, r, il);
                continue;
            }
            slot->kv_k[r][il] = ggml_new_tensor_2d(ctxs[d], GGML_TYPE_F16, m.hp.n_kv_row, m.n_ctx);
            ggml_format_name(slot->kv_k[r][il], "slot%d_kv.%d.%d", slot->id, r, il);
            if (il < m.hp.n_layer && m.layers[il].indexer_full)
            {
                // glm5next caches an indexer key AND a compressor gate per cell,
                // packed [key | gate] into one row so a pool's members are
                // gathered once.
                const int64_t idx_row = m.hp.indexer_kpool > 0 ? 2 * m.hp.indexer_head_size
                                                               : m.hp.indexer_head_size;
                slot->idx_k[r][il] = ggml_new_tensor_2d(ctxs[d], GGML_TYPE_F16, idx_row, m.n_ctx);
                ggml_format_name(slot->idx_k[r][il], "slot%d_idx.%d.%d", slot->id, r, il);
            }
        }
    }

    if (admit && !glm_slot_fits(m, ctxs))
        return nullptr;

    // Prepare every owning node and tracker before allocating device payloads.
    // A failed buffer free must retain its owner/ticket, even on this failed
    // construction path. An unpublished slot stays poisoned throughout it.
    m.slot_ctxs.reserve(m.slot_ctxs.size() + ctxs.size());
    const int id = slot->id;
    slot->kda_restore_failed = true;
    auto inserted = m.slots.emplace(id, std::move(slot));
    if (!inserted.second) return nullptr;
    glm_slot * raw = inserted.first->second.get();
    const size_t first_entry = m.slot_ctxs.size();
    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (ggml_get_first_tensor(ctxs[d]) == nullptr) continue;
        m.slot_ctxs.push_back({ id, ctxs[d], nullptr }); // capacity already reserved
        ctxs[d] = nullptr; // the model now owns this metadata, including failures
    }
    bool ok = false;
    try
    {
        glm_test_slot_alloc_fault(2);
        const auto epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
        if (tsg::g_backend_compute_failed.load(std::memory_order_acquire)) throw 42;
        bool first = true;
        size_t entry_index = first_entry;
        for (int device = 0; device <= m.n_gpu; ++device)
        {
            // Nonempty contexts were transferred in ascending device order;
            // empty ones remain in metadata_owner for ordinary destruction.
            if (ctxs[device] != nullptr) continue;
            auto & entry = m.slot_ctxs[entry_index++];
            entry.buf = tsg::graph_budget_alloc_ctx_tensors(entry.ctx, m.backends[device], device);
            if (!entry.buf) throw std::bad_alloc();
            ggml_backend_buffer_clear(entry.buf, 0);
            if (first) { first = false; glm_test_slot_alloc_fault(3); }
        }
        for (int d = 0; d < m.n_backends; ++d)
            if (m.backends[d]) ggml_backend_synchronize(m.backends[d]);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire)) throw 43;
        ok = true;
    }
    catch (...)
    {
        bool drained = true;
        for (int d = 0; d < m.n_backends; ++d)
            try { if (m.backends[d]) ggml_backend_synchronize(m.backends[d]); } catch (...) { drained = false; }
        if (drained)
            for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
            {
                if (it->slot_id != id) { ++it; continue; }
                try
                {
                    if (it->buf) { tsg::graph_budget_free_buffer(it->buf); it->buf = nullptr; }
                    if (it->ctx) { ggml_free(it->ctx); it->ctx = nullptr; }
                    it = m.slot_ctxs.erase(it);
                }
                catch (...) { ++it; } // retain the still-live physical owner
            }
        const bool live = std::any_of(m.slot_ctxs.begin(), m.slot_ctxs.end(), [id](const auto & entry) { return entry.slot_id == id; });
        if (!live) m.slots.erase(id);
        throw;
    }
    if (!ok) return nullptr;
    raw->kda_restore_failed = false;
    return raw;
}

/// Zero a slot's KDA recurrent state (glm5next): a new conversation must not
/// inherit the previous one's conv tail or delta-net state. The MLA rows need
/// no such wipe - they are rewritten before anything reads them.
static void slot_clear_recurrent(glm_model & m, glm_slot & slot)
{
    std::vector<uint8_t> zeros;
    auto wipe = [&](ggml_tensor * t) {
        if (!t) return;
        const size_t nb = ggml_nbytes(t);
        if (zeros.size() < nb) zeros.assign(nb, 0);
        ggml_backend_tensor_set(t, zeros.data(), 0, nb);
    };
    for (int r = 0; r < m.tp; r++)
    {
        for (size_t il = 0; il < slot.kda_conv[r].size(); il++) wipe(slot.kda_conv[r][il]);
        for (size_t il = 0; il < slot.kda_ssm[r].size(); il++) wipe(slot.kda_ssm[r][il]);
    }
}

static void slot_free(glm_model & m, int slot_id)
{
    // Graphs bake this slot's cache addresses into their nodes; drop them first.
    // A batched-decode graph belongs to no single slot but reads N of them, and
    // slot ids are reused, so a stale one could be matched by a later batch and
    // run against caches that no longer exist.
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); )
    {
        bool refs = (*it)->slot_id == slot_id;
        for (const bd_token & B : (*it)->bd)
            if (B.slot_id == slot_id) refs = true;
        it = refs ? m.graph_cache.erase(it) : std::next(it);
    }

    m.slots.erase(slot_id);
    m.reclaimable.erase(std::remove(m.reclaimable.begin(), m.reclaimable.end(), slot_id), m.reclaimable.end());
    // A snapshot of this slot's KDA state describes caches that no longer
    // exist; slot ids are reused, so it must not survive to match a new one.
    if (m.kda_snap.slot_id == slot_id) m.kda_snap.valid = false;
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id != slot_id) { ++it; continue; }
        if (it->buf) tsg::graph_budget_free_buffer(it->buf);
        if (it->ctx) ggml_free(it->ctx);
        it = m.slot_ctxs.erase(it);
    }
}

#if defined(TSG_GGML_TEST_HOOKS)
static thread_local int glm_test_graph_alloc_failures = 0;
#ifndef TSG_TEST_EXPORT
#define TSG_TEST_EXPORT TSG_EXPORT
#endif
// Fixture only: the next `n` graph allocations fail as a full device would.
TSG_TEST_EXPORT void TSGgml_GlmTestFailGraphAllocs(int n)
{
    glm_test_graph_alloc_failures = n;
}
#endif
static bool glm_test_graph_alloc_fault()
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (glm_test_graph_alloc_failures > 0)
    {
        glm_test_graph_alloc_failures--;
        return true;
    }
#endif
    return false;
}

#if defined(TSG_GGML_TEST_HOOKS)
static thread_local int glm_test_batched_dispatch_stage = 0, glm_test_batched_dispatch_kind = 0;
// Stage 1: compute completed but no logits were published. Stage 2: the
// first row's logits/head were copied. Both leave genuine mutated native state.
TSG_TEST_EXPORT void TSGgml_GlmTestBatchedDispatchFault(int stage, int kind)
{
    glm_test_batched_dispatch_stage = stage;
    glm_test_batched_dispatch_kind = kind;
}
#endif
static void glm_test_batched_dispatch_fault(int stage)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (glm_test_batched_dispatch_stage == stage)
    {
        glm_test_batched_dispatch_stage = 0;
        if ((stage == 4 || stage == 5) && glm_test_batched_dispatch_kind == 1) {
            // Reported IDs download or last-seam drain error only, not a real
            // driver failure. The production epoch guards reject publication.
            tsg::g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        if (glm_test_batched_dispatch_kind == 1) throw std::bad_alloc();
        throw 17;
    }
#else
    (void) stage;
#endif
}

// A graph that could not be allocated: free the oldest retained slot it does
// not read, so the caller can build and allocate it again. A retained slot only
// keeps a finished conversation's next turn cheap, and a running request
// outranks it here as it does for a new slot (the managed AllocReclaiming).
// Every slot the admission rule let in may be retained at once, so without
// this a graph that grows past the reserve (a longer context, a larger batch)
// failed the request while those slots held the memory. False when none is left.
static bool glm_reclaim_for_graph(glm_model & m, const int32_t * reads, int n_reads)
{
    for (size_t i = 0; i < m.reclaimable.size(); i++)
    {
        const int id = m.reclaimable[i];
        if (std::find(reads, reads + n_reads, id) != reads + n_reads) continue;
        if (m.active_slot && m.active_slot->id == id) continue;
        if (!m.slots.count(id))
        {
            m.reclaimable.erase(m.reclaimable.begin() + (std::ptrdiff_t) i--);
            continue;
        }
        slot_free(m, id);
        m.reclaimed.push_back(id);
        fprintf(stderr, "[glm] freed retained sequence slot %d for a graph that did not fit\n", id);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// `ctx_is_hard_limit` distinguishes "the user asked for this context" from
/// "this is what the GGUF advertises". The first is honoured or refused; the
/// second is a ceiling the loader may cap to whatever the VRAM actually holds,
/// because a 1M-token advertisement is not a promise that 1M tokens of KV fit
/// beside the weights.
static std::string akey(const char * arch, const char * suffix)
{
    std::string k(arch);
    k += ".";
    k += suffix;
    return k;
}

static glm_model * glm_load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch,
                            int n_threads, int n_cpu_moe_req, const char * backend_name, int tp_req,
                            bool ctx_is_hard_limit, bool load_mtp)
{
    auto t_start = std::chrono::steady_clock::now();
    std::unique_ptr<glm_model> m(new glm_model());
    const char * strict_recurrent = getenv("TS_GGML_STRICT_RECURRENT_PROJECTIONS");
    m->strict_recurrent_requested = strict_recurrent && strcmp(strict_recurrent, "1") == 0;
    m->trace_dump = trace_dump_state::configured();
    if (m->trace_dump) {
        std::lock_guard<std::mutex> guard(m->trace_dump->mutex);
        m->trace_model_id = ++m->trace_dump->next_model;
    }

    // --- backends ---------------------------------------------------------
    // "CPU" is an explicit request for a host-only run (--backend ggml_cpu):
    // without it the device scan would happily pick up Metal or CUDA, which is
    // never what that flag means.
    const bool cpu_only = backend_name && strcmp(backend_name, "CPU") == 0;
    int n_gpu = 0;
    for (int pass = 0; pass < 2 && n_gpu == 0 && !cpu_only; pass++)
    {
        const char * want = pass == 0 ? backend_name : nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && n_gpu < MAX_GPUS; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (!backend_matches(reg ? ggml_backend_reg_name(reg) : nullptr, want)) continue;
            if (n_gpu_req > 0 && n_gpu >= n_gpu_req) break;
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (!be) continue;
            m->backends[n_gpu++] = be;
        }
        if (pass == 0 && n_gpu == 0 && backend_name && *backend_name)
            fprintf(stderr, "[glm] no %s GPU devices found; falling back to any available GPU backend\n", backend_name);
    }
    m->n_gpu = n_gpu;

    // Tensor parallelism uses exactly `tp` devices, each running every layer on
    // its own slice of the head- and expert-sharded matrices. Anything else is
    // the layer split, where each layer lives on one device.
    if (tp_req > 1)
    {
        if (tp_req > MAX_GPUS)
        {
            tsg::report_load_refusal( "[glm] --tp %d exceeds this executor's maximum of %d ranks\n",
                    tp_req, MAX_GPUS);
            return nullptr;
        }
        if (tp_req > n_gpu)
        {
            // More ranks than GPUs is not a production configuration — it buys
            // no parallelism — but it is how the sharding math gets tested on a
            // single-GPU machine, so allow it explicitly rather than only on a
            // multi-GPU host.
            if (getenv("TS_GLM_TP_OVERSUBSCRIBE") == nullptr)
            {
                tsg::report_load_refusal( "[glm] --tp %d needs %d GPUs; only %d are visible "
                        "(TS_GLM_TP_OVERSUBSCRIBE=1 packs several ranks onto one GPU for testing)\n",
                        tp_req, tp_req, n_gpu);
                return nullptr;
            }
            fprintf(stderr, "[glm] --tp %d on %d GPU(s): ranks share devices (testing only, no speedup)\n",
                    tp_req, n_gpu);
        }
        m->tp = tp_req;
        if (const char * e = getenv("TS_GLM_TP_SHARD")) m->tp_shard = atoi(e) & 3;
    }

    // The registry helper enumerates every linked backend, including CUDA.
    // Construct the requested CPU backend directly so host-only inference does
    // not initialize an unrelated accelerator or its vendor callbacks.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    {
        // With --n-cpu-moe (or a GPU-less run) the routed-expert matmuls land
        // here on every token; that is memory-bandwidth bound and wants every
        // core the process is actually allowed to use. --cpu-moe-threads
        // arrives through the native atomic override, never the environment
        // (.NET's SetEnvironmentVariable does not reach getenv on Linux), so it
        // must be consulted explicitly or the flag silently does nothing here.
        // Same precedence as the DSV4 executor: the explicit override applies
        // next, and an inherited native TS_CPU_MOE_THREADS has final priority.
        int cpu_threads = n_threads > 0 ? n_threads : 16;
        if (n_cpu_moe_req != 0 || n_gpu == 0) cpu_threads = tsg::available_cpu_parallelism();
        const int explicit_threads = tsg::host_moe_explicit_thread_count();
        if (explicit_threads > 0) cpu_threads = explicit_threads;
        if (const char * e = getenv("TS_CPU_MOE_THREADS")) { int v = atoi(e); if (v > 0) cpu_threads = v; }
        ggml_backend_cpu_set_n_threads(cpu, cpu_threads);

        // A persistent pool: one token is one CPU graph split per offloaded
        // layer, and spawning/joining a disposable pool per split costs more
        // than the matmuls it parallelises.
        ggml_threadpool_params tpp = ggml_threadpool_params_default(cpu_threads);
        m->cpu_threadpool = ggml_threadpool_new(&tpp);
        if (m->cpu_threadpool) ggml_backend_cpu_set_threadpool(cpu, m->cpu_threadpool);
        fprintf(stderr, "[glm] CPU worker pool: threads=%d, persistent=%s\n",
                cpu_threads, m->cpu_threadpool ? "yes" : "no");
    }
    m->backends[n_gpu] = cpu;
    m->n_backends = n_gpu + 1;

    {
        int nb = 0;
        for (int d = 0; d < n_gpu; d++) m->sched_backends[nb++] = m->backends[d];
        m->sched_backends[nb++] = cpu;
        for (int i = 0; i < nb; i++)
            m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->sched_backends[i]);
        m->n_sched_backends = nb;

        if (const char * e = getenv("TS_GLM_GRAPH_CACHE")) { int v = atoi(e); if (v > 0) m->graph_cache_cap = v; }
    }
    const bool graph_cache_cap_explicit = getenv("TS_GLM_GRAPH_CACHE") != nullptr;

    // --- metadata ---------------------------------------------------------
    tsg_glm::metadata_owner<> root_metadata;
    gguf_init_params ip = { true, &root_metadata.tensors };
    gguf_context * g0 = root_metadata.file = gguf_init_from_file(gguf_path, ip);
    if (!g0) { tsg::report_load_refusal( "[glm] failed to open %s\n", gguf_path); return nullptr; }

    int32_t split_count = 1;
    kv_u32(g0, "split.count", &split_count);
    shard_files shards = resolve_shards(gguf_path, split_count);

    glm_hparams & hp = m->hp;
    // GLM-5.3-Flash ("glm5next") loads through this same executor: it keeps the
    // MLA+DSA+MoE core and adds KDA layers, pooled indexing and Sinkhorn
    // hyper-connections on top.
    std::string architecture_prefix = "glm-dsa";
    {
        const int64_t ai = gguf_find_key(g0, "general.architecture");
        if (ai >= 0 && gguf_get_kv_type(g0, ai) == GGUF_TYPE_STRING)
        {
            const char * architecture = gguf_get_val_str(g0, ai);
            hp.g5n = strcmp(architecture, "glm5next") == 0 || strcmp(architecture, "glm5-next") == 0;
            // The current published spelling also prefixes every metadata key.
            // Own the string beyond gguf_free(g0); never rewrite the checkpoint.
            if (hp.g5n) architecture_prefix = architecture;
        }
    }
    const char * AP = architecture_prefix.c_str();
#ifdef TSG_GGML_USE_CUDA
    if (hp.g5n && m->n_gpu > 0) {
        bool all_cuda = true;
        for (int d = 0; d < m->n_gpu; ++d)
            all_cuda &= ggml_backend_is_cuda(m->backends[d]);
        if (all_cuda) {
            if (tsg_f32_projection_environment_conflicts(getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE"))) {
                tsg::report_load_refusal("[glm] sensitive floating-point gate projections require F32 activations; "
                    "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16/fp16/bf16 overrides that contract. Use auto or f32.\n");
                return nullptr;
            }
            for (int d = 0; d < m->n_gpu; ++d) {
                m->projection_backends[d] = tsg_dsv4_fused_backend_init(m->backends[d]);
                if (!m->projection_backends[d]) {
                    tsg::report_load_refusal("[glm] cannot create the owned F32 sensitive projection executor\n");
                    return nullptr;
                }
                m->sched_backends[d] = m->projection_backends[d];
                m->sched_bufts[d] = ggml_backend_get_default_buffer_type(m->sched_backends[d]);
            }
        }
    }
#endif
    m->strict_recurrent_projections = m->strict_recurrent_requested && hp.g5n
        && m->n_gpu == 1 && m->tp == 1 && m->projection_backends[0] != nullptr;
    if (hp.g5n) {
        fprintf(stderr, "[glm] Experimental strict recurrent projections requested=%d effective=%d "
            "(requires owned CUDA executor, one GPU, tp=1; other backends/TP retain existing arithmetic).\n",
            int(m->strict_recurrent_requested), int(m->strict_recurrent_projections));
    }
    bool ok = true;
    ok &= kv_u32(g0, akey(AP, "block_count").c_str(), &hp.n_layer_all);
    ok &= kv_u32(g0, akey(AP, "embedding_length").c_str(), &hp.n_embd);
    ok &= kv_u32(g0, akey(AP, "attention.head_count").c_str(), &hp.n_head);
    ok &= kv_u32(g0, akey(AP, "rope.dimension_count").c_str(), &hp.n_rot);
    ok &= kv_u32(g0, akey(AP, "attention.q_lora_rank").c_str(), &hp.q_lora_rank);
    ok &= kv_u32(g0, akey(AP, "attention.kv_lora_rank").c_str(), &hp.kv_lora_rank);
    ok &= kv_f32(g0, akey(AP, "attention.layer_norm_rms_epsilon").c_str(), &hp.rms_eps);
    ok &= kv_u32(g0, akey(AP, "expert_count").c_str(), &hp.n_expert);
    ok &= kv_u32(g0, akey(AP, "expert_used_count").c_str(), &hp.n_expert_used);
    ok &= kv_u32(g0, akey(AP, "expert_feed_forward_length").c_str(), &hp.n_ff_exp);
    ok &= kv_u32(g0, akey(AP, "attention.indexer.head_count").c_str(), &hp.indexer_n_head);
    ok &= kv_u32(g0, akey(AP, "attention.indexer.key_length").c_str(), &hp.indexer_head_size);
    ok &= kv_u32(g0, akey(AP, "attention.indexer.top_k").c_str(), &hp.indexer_top_k);

    kv_u32(g0, akey(AP, "feed_forward_length").c_str(), &hp.n_ff);
    kv_u32(g0, akey(AP, "leading_dense_block_count").c_str(), &hp.n_dense_lead);
    kv_u32(g0, akey(AP, "nextn_predict_layers").c_str(), &hp.n_layer_nextn);
    kv_u32(g0, akey(AP, "expert_shared_count").c_str(), &hp.n_expert_shared);
    kv_f32(g0, akey(AP, "expert_weights_scale").c_str(), &hp.expert_weights_scale);
    kv_bool(g0, akey(AP, "expert_weights_norm").c_str(), &hp.expert_weights_norm);
    kv_u32(g0, akey(AP, "expert_gating_func").c_str(), &hp.expert_gating_func);
    kv_f32(g0, akey(AP, "rope.freq_base").c_str(), &hp.rope_freq_base);
    kv_u32(g0, akey(AP, "context_length").c_str(), &hp.n_ctx_train);

    int32_t key_len = 0, val_len = 0;
    kv_u32(g0, akey(AP, "attention.key_length").c_str(), &key_len);
    kv_u32(g0, akey(AP, "attention.value_length").c_str(), &val_len);
    hp.n_embd_head_k = key_len;
    hp.n_embd_head_v = val_len;
    kv_u32(g0, akey(AP, "attention.key_length_mla").c_str(), &hp.n_embd_head_k);
    kv_u32(g0, akey(AP, "attention.value_length_mla").c_str(), &hp.n_embd_head_v);

    int32_t expert_groups = 1, expert_groups_used = 1;
    kv_u32(g0, akey(AP, "expert_group_count").c_str(), &expert_groups);
    kv_u32(g0, akey(AP, "expert_group_used_count").c_str(), &expert_groups_used);

    hp.n_layer = hp.n_layer_all - hp.n_layer_nextn;
    hp.n_nope = hp.n_embd_head_k - hp.n_rot;
    hp.n_kv_row = hp.kv_lora_rank + hp.n_rot;
    if (hp.expert_gating_func == 0) hp.expert_gating_func = 2;

    if (hp.g5n)
    {
        ok &= kv_u32(g0, akey(AP, "kda.head_dim").c_str(), &hp.kda_head_dim);
        ok &= kv_u32(g0, akey(AP, "ssm.conv_kernel").c_str(), &hp.d_conv);
        ok &= kv_u32(g0, akey(AP, "attention.indexer.kpool").c_str(), &hp.indexer_kpool);
        ok &= kv_u32(g0, akey(AP, "hyper_connection.count").c_str(), &hp.hc_mult);
        kv_f32(g0, akey(AP, "kda.gate_lower_bound").c_str(), &hp.kda_gate_lb);
        kv_u32(g0, akey(AP, "hyper_connection.sinkhorn_iterations").c_str(), &hp.hc_sinkhorn);
        kv_f32(g0, akey(AP, "hyper_connection.epsilon").c_str(), &hp.hc_eps);
        kv_arr_f32_first(g0, akey(AP, "swiglu_clamp_exp").c_str(), &hp.swiglu_clamp);
        kv_f32(g0, akey(AP, "attention.layer_norm_epsilon").c_str(), &hp.norm_eps);

        // per-layer type: attention.head_count_kv is 0 on KDA layers, 1 on MLA
        std::vector<uint8_t> kvh;
        if (kv_arr_u32_as_u8(g0, akey(AP, "attention.head_count_kv").c_str(), kvh))
        {
            hp.is_recr.assign((size_t) hp.n_layer, 0);
            for (int il = 0; il < hp.n_layer && il < (int) kvh.size(); il++)
                hp.is_recr[(size_t) il] = kvh[(size_t) il] == 0 ? 1 : 0;
        }
        else
        {
            ok = false;
        }
        hp.kda_n_head = hp.n_head;
        if (hp.kda_head_dim <= 0 || hp.d_conv <= 1 || hp.indexer_kpool <= 0 || hp.hc_mult <= 0)
            ok = false;
        if (hp.indexer_top_k % (hp.indexer_kpool > 0 ? hp.indexer_kpool : 1) != 0)
            ok = false;
    }

    resolve_indexer_types(hp, g0);
    root_metadata.reset();

    if (!ok || hp.n_layer <= 0 || hp.n_nope <= 0 || hp.n_head <= 0)
    {
        tsg::report_load_refusal( "[glm] missing or invalid glm-dsa metadata\n");
        return nullptr;
    }
    if (expert_groups > 1 && expert_groups_used != expert_groups)
    {
        tsg::report_load_refusal( "[glm] %d expert groups (top-%d) are not supported\n", expert_groups, expert_groups_used);
        return nullptr;
    }
    if (hp.g5n)
    {
        // glm5next: the indexer is full on every MLA layer and absent on KDA ones.
        hp.indexer_full.assign((size_t) hp.n_layer, 0);
        for (int il = 0; il < hp.n_layer; il++)
            hp.indexer_full[(size_t) il] = hp.is_recr[(size_t) il] ? 0 : 1;
    }
    else if (hp.indexer_full.empty() || hp.indexer_full[0] == 0)
    {
        tsg::report_load_refusal( "[glm] layer 0 must carry a full DSA indexer\n");
        return nullptr;
    }

    // --- tensor sources across shards -------------------------------------
    std::map<std::string, tensor_source> sources;
    std::vector<size_t> layer_bytes((size_t) hp.n_layer, 0);
    std::vector<size_t> layer_exps_bytes((size_t) hp.n_layer, 0);
    size_t root_bytes = 0;
    // The trailing NextN/MTP block. Counted apart from the trunk so the layer
    // split prices it onto the device that will actually host it instead of
    // discovering it after the pack.
    size_t mtp_bytes = 0;
    size_t mtp_exps_bytes = 0;
    bool   mtp_present = false;

    for (size_t si = 0; si < shards.paths.size(); si++)
    {
        tsg_glm::metadata_owner<> shard_metadata;
        gguf_init_params sp = { true, &shard_metadata.tensors };
        gguf_context * g = shard_metadata.file = gguf_init_from_file(shards.paths[si].c_str(), sp);
        ggml_context * meta = shard_metadata.tensors;
        if (!g) { tsg::report_load_refusal( "[glm] failed to open shard %s\n", shards.paths[si].c_str()); return nullptr; }
        if (!check_shard_complete(shards.paths[si].c_str(), g, meta)) return nullptr;

        const size_t data_off = gguf_get_data_offset(g);
        const int64_t n_tensors = gguf_get_n_tensors(g);
        for (int64_t ti = 0; ti < n_tensors; ti++)
        {
            const char * name = gguf_get_tensor_name(g, ti);
            ggml_tensor * mt = ggml_get_tensor(meta, name);
            if (!mt) continue;
            tensor_source src;
            src.shard = (int) si;
            src.offset = data_off + gguf_get_tensor_offset(g, ti);
            src.size = ggml_nbytes(mt);
            src.type = mt->type;
            for (int d = 0; d < 4; d++) src.ne[d] = mt->ne[d];
            sources[name] = src;

            int bid = -1;
            if (sscanf(name, "blk.%d.", &bid) == 1 && bid >= 0 && bid < hp.n_layer)
            {
                layer_bytes[bid] += src.size;
                if (strstr(name, "_exps.") != nullptr) layer_exps_bytes[bid] += src.size;
            }
            else if (bid < 0)
            {
                root_bytes += src.size;   // token_embd / output / output_norm
            }
            else if (bid == hp.n_layer && hp.n_layer_nextn == 1)
            {
                // The NextN/MTP block. The trunk graph never runs it; the draft
                // head does, and only when speculation was requested.
                mtp_bytes += src.size;
                if (strstr(name, "_exps.") != nullptr) mtp_exps_bytes += src.size;
                if (strstr(name, "nextn.eh_proj") != nullptr) mtp_present = true;
            }
        }
    }

    size_t embd_bytes = 0;
    {
        auto it = sources.find("token_embd.weight");
        if (it == sources.end()) { tsg::report_load_refusal( "[glm] token_embd.weight missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) it->second.ne[1];
        embd_bytes = it->second.size;
    }
    size_t head_bytes = root_bytes > embd_bytes ? root_bytes - embd_bytes : 0;

    // --- NextN/MTP draft block -------------------------------------------
    // Loading it costs a whole extra decoder layer (~3 GiB of GLM-5.2 at
    // IQ2_XXS) that also competes with the KV cache for the VRAM the context is
    // sized against, so it is opt-in: the hosts set TS_SPEC from --spec
    // before the model loads, and the managed side forwards that as
    // `load_mtp`. A checkpoint that declares nextn_predict_layers but ships no
    // MTP tensors (a trunk-only re-quantization) loads normally without one.
    if (load_mtp && hp.g5n)
    {
        // The glm5next NextN block is a full DSA decoder layer wrapped in its
        // own hyper-connections; llama.cpp asserts its graph unimplemented and
        // this executor does not build it yet either.
        fprintf(stderr, "[glm] glm5next NextN/MTP draft heads are not implemented; ngram speculation remains available\n");
    }
    if (load_mtp && !hp.g5n)
    {
        auto has_src = [&](const char * suffix) {
            char nm[256];
            snprintf(nm, sizeof(nm), "blk.%d.%s", hp.n_layer, suffix);
            return sources.find(nm) != sources.end();
        };
        // Everything is decided BEFORE a byte is placed: a decision made later
        // would already have priced (and, worse, uploaded) 3 GiB of a block
        // that then turns out to be unusable.
        const bool wiring = mtp_present && has_src("nextn.enorm.weight") && has_src("nextn.hnorm.weight");
        // A draft block with no head of its own borrows the trunk's, which is
        // column-parallel under tensor parallelism — the draft would read one
        // rank's strip of the vocabulary and produce nonsense.
        const bool borrows_split_head = !has_src("nextn.shared_head_head.weight") && m->tp > 1;

        if (hp.n_layer_nextn > 1)
            fprintf(stderr, "[glm] %d NextN blocks declared; only 1 is supported. Serving standard decode\n",
                    hp.n_layer_nextn);
        else if (hp.n_layer_nextn < 1)
            fprintf(stderr, "[glm] speculation requested but this checkpoint declares no NextN block; "
                            "serving standard decode\n");
        else if (!wiring)
            fprintf(stderr, "[glm] speculation requested but this checkpoint's NextN block is missing its wiring "
                            "(a trunk-only requantization); serving standard decode\n");
        else if (borrows_split_head)
            fprintf(stderr, "[glm] the NextN block has no LM head of its own and the trunk head is column-parallel "
                            "under --tp %d; serving standard decode\n", m->tp);
        else
        {
            m->has_mtp = true;
            m->mtp_layer = hp.n_layer;
        }
    }
    if (!m->has_mtp) { mtp_bytes = 0; mtp_exps_bytes = 0; }
    else if (!graph_cache_cap_explicit)
    {
        // Speculation multiplies the number of live graph SHAPES: the trunk
        // verify is rebuilt for every window length the drafter produces
        // (2..K+1 rows), the draft block adds its own 1-row and catch-up
        // shapes, and both still key on the padded KV window. At the default
        // cap of 8 that thrashes — every step would rebuild and re-allocate a
        // graph it had a moment ago. These graphs are all tiny next to the
        // 1024-row prefill graph the cache already holds.
        int max_draft = 8;
        if (const char * e = getenv("TS_SPEC_DRAFT")) { int v = atoi(e); if (v > 0 && v <= 64) max_draft = v; }
        m->graph_cache_cap = std::min(64, 8 + 2 * (max_draft + 1));
    }
    // The draft block lands beside the LM head (it reads the trunk's post-norm
    // hidden state and writes through the same head), so it is priced with it.
    head_bytes += mtp_bytes;

    m->n_ctx = n_ctx > 0 ? n_ctx : 8192;
    // What the caller asked for, kept so a later measurement can hand slack back
    // without ever exceeding the request.
    const int ctx_requested = m->n_ctx;
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    m->layers.resize((size_t) (hp.n_layer + (m->has_mtp ? 1 : 0)));
    for (int il = 0; il < hp.n_layer; il++)
    {
        m->layers[il].indexer_full = hp.indexer_full[il] != 0;
        m->layers[il].is_moe = il >= hp.n_dense_lead;
        m->layers[il].recurrent = hp.g5n && hp.is_recr[(size_t) il] != 0;
    }
    if (m->has_mtp)
    {
        glm_layer & M = m->layers[(size_t) m->mtp_layer];
        M.indexer_full = false;             // the draft block attends densely
        M.is_moe = m->mtp_layer >= hp.n_dense_lead;
    }

    // A row-parallel output projection is sliced along ne0, so each head shard
    // must begin and end on a whole quantization block. This is automatic for
    // GLM-5.2's 256-wide value heads. GLM-5.3's KDA heads are 128-wide while
    // its K-quantized output matrices use 256-element blocks, so those heads
    // travel in pairs. Compute the requirement from the file instead of baking
    // either model's current dimensions into the executor.
    if (m->tp > 1 && m->tp_heads())
    {
        int group = 1;
        const int n_loaded_layers = hp.n_layer + (m->has_mtp ? 1 : 0);
        for (int il = 0; il < n_loaded_layers; il++)
        {
            char name[256];
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il);
            auto it = sources.find(name);
            if (it == sources.end()) continue; // the ordinary required-weight check reports this later
            const int head_width = m->layers[(size_t) il].recurrent ? hp.kda_head_dim : hp.n_embd_head_v;
            const int block = (int) ggml_blck_size(it->second.type);
            const int need = block / std::gcd(block, head_width);
            group = std::lcm(group, need);
        }
        if (hp.n_head % group != 0 || hp.n_head / group < m->tp)
        {
            tsg::report_load_refusal(
                    "[glm] --tp %d cannot split %d heads in the %d-head groups required by the "
                    "output weights' quantization blocks\n",
                    m->tp, hp.n_head, group);
            return nullptr;
        }
        m->tp_head_group = group;
    }

    // What one layer's caches cost the device that hosts it. Priced into the
    // split so a device packed to the last byte of weights does not fail at
    // slot allocation instead of at load.
    auto layer_cache_bytes = [&](int il) -> size_t
    {
        // A glm5next KDA layer keeps a fixed-size recurrent state instead of
        // per-position rows; an MLA layer's indexer cache holds [key | gate]
        // pairs when the indexer is pooled.
        if (hp.g5n && il < hp.n_layer && hp.is_recr[(size_t) il])
            return (size_t) (hp.d_conv - 1) * 3 * hp.kda_head_dim * hp.kda_n_head * 4
                 + (size_t) hp.kda_head_dim * hp.kda_head_dim * hp.kda_n_head * 4 + 4 * 256;
        size_t b = (size_t) hp.n_kv_row * m->n_ctx * 2;
        if (hp.indexer_full[il])
            b += (size_t) (hp.indexer_kpool > 0 ? 2 : 1) * hp.indexer_head_size * m->n_ctx * 2;
        return b + 4 * 256;
    };

    // --- per-device budget + layer split ----------------------------------
    std::vector<size_t> dev_budget((size_t) std::max(n_gpu, 1), 0);
    std::vector<size_t> dev_free((size_t) std::max(n_gpu, 1), 0);
    {
        size_t reserve_mb = 3072;
        if (const char * e = getenv("TS_GLM_VRAM_RESERVE_MB")) { long v = atol(e); if (v >= 0) reserve_mb = (size_t) v; }
        const size_t reserve = reserve_mb * 1024 * 1024;
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            dev_free[d] = free_b;
            dev_budget[d] = free_b > reserve ? free_b - reserve : 0;
        }
    }

    size_t total_bytes = root_bytes;
    for (auto b : layer_bytes) total_bytes += b;

    // KV bytes one token costs, summed over every layer. GLM-5.2 advertises a
    // 1M context, which at 78 layers is ~93 GiB of cache — a whole card's worth
    // on top of the weights — so the advertised number is a ceiling to fit
    // under, not a promise.
    // Row widths must match slot_alloc exactly: the indexer packs [key | gate]
    // when the cells are pooled, and KDA layers keep a fixed-size recurrent
    // state instead of per-token rows.
    const size_t idx_row_bytes = (hp.indexer_kpool > 0 ? 2 * (size_t) hp.indexer_head_size
                                                       : (size_t) hp.indexer_head_size) * 2;
    const int max_kda_heads_per_rank = hp.g5n && m->tp > 1 && m->tp_heads()
        ? shard_first(hp.kda_n_head / m->tp_head_group, m->tp, 1) * m->tp_head_group
        : hp.kda_n_head;
    const size_t kda_state_bytes = hp.g5n
        ? ((size_t) (hp.d_conv - 1) * 3 * hp.kda_head_dim * max_kda_heads_per_rank
           + (size_t) hp.kda_head_dim * hp.kda_head_dim * max_kda_heads_per_rank) * 4
        : 0;
    size_t kv_bytes_per_token = 0;
    size_t kv_bytes_fixed = 0;
    for (int il = 0; il < hp.n_layer; il++)
    {
        if (hp.g5n && hp.is_recr[(size_t) il] != 0) { kv_bytes_fixed += kda_state_bytes; continue; }
        kv_bytes_per_token += (size_t) hp.n_kv_row * 2;
        if (hp.indexer_full[il]) kv_bytes_per_token += idx_row_bytes;
    }
    if (m->has_mtp) kv_bytes_per_token += (size_t) hp.n_kv_row * 2;

    // What a graph needs on top of the caches. Two things scale with the
    // context: the DSA top-k masks, which are [n_kv, n_ubatch] F16 and live for
    // the whole graph on every full-indexer layer, and nothing else. The rest is
    // fixed per ubatch and dominated by the LM head's [n_vocab, n_ubatch] output.
    // Counted twice over because the graph cache keeps several built graphs
    // alive, each holding its own allocation.
    const int n_full_indexer = (int) std::count(hp.indexer_full.begin(), hp.indexer_full.end(), (uint8_t) 1);
    const size_t graph_bytes_per_token = 2 * (size_t) n_full_indexer * (size_t) m->n_ubatch * 2;
    const size_t graph_bytes_fixed =
        2 * (size_t) m->n_ubatch * ((size_t) hp.n_vocab + 16 * (size_t) hp.n_embd) * 4
        + kv_bytes_fixed;

    /// Largest context whose caches AND graphs still fit in `avail`, in whole
    /// 256-token steps because that is the granularity the graphs are keyed on.
    auto ctx_that_fits = [&](size_t avail) -> int
    {
        const size_t per_token = kv_bytes_per_token + graph_bytes_per_token;
        if (per_token == 0 || avail <= graph_bytes_fixed) return 0;
        size_t tokens = (avail - graph_bytes_fixed) / per_token;
        tokens -= tokens % 256;
        if (tokens > (size_t) ctx_requested) tokens = (size_t) ctx_requested;
        return (int) tokens;
    };

    /// Everything a rank must hold at `n_ctx` besides its weights.
    auto runtime_bytes = [&](int n_ctx_want) -> size_t
    {
        return (size_t) n_ctx_want * (kv_bytes_per_token + graph_bytes_per_token) + graph_bytes_fixed;
    };

    // The pre-load estimate only has to be conservative enough to refuse the
    // hopeless cases; when the devices can be asked directly after the weights
    // land, that measurement decides the context and the estimate stays quiet.
    const bool remeasure_ctx = !ctx_is_hard_limit && n_gpu > 0;

    int n_cpu_moe = 0;
    if (m->tp > 1)
    {
        // Every rank runs every layer, so the budget is per rank: the routed
        // experts are the only weights that shrink with `tp`, and the MLA and
        // indexer caches do not shrink at all — they are rank-independent, so
        // every rank keeps its own full-length copy. Without this the load only
        // discovered it did not fit when the first sequence slot failed to
        // allocate, which is minutes of weight reads after the point of no
        // return.
        auto rank_weight_bytes = [&](int n_cpu) -> size_t
        {
            size_t b = embd_bytes + head_bytes;              // replicated on rank 0's device
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t e = layer_exps_bytes[il];
                b += layer_bytes[il] - e;                    // dense half: replicated (upper bound)
                if (il >= n_cpu) b += e / (size_t) m->tp;    // this rank's expert rows
            }
            return b;
        };
        // Only meaningful with one rank per GPU: a host-only run has no device
        // budget to measure, and TS_GLM_TP_OVERSUBSCRIBE deliberately stacks
        // several ranks on one card, where a per-rank budget says nothing.
        const bool budgeted = n_gpu > 0 && m->tp <= n_gpu;

        const size_t kv_bytes = runtime_bytes(m->n_ctx);

        size_t budget = 0;
        if (budgeted)
        {
            budget = SIZE_MAX;
            for (int r = 0; r < m->tp; r++) budget = std::min(budget, dev_budget[(size_t) m->rank_device(r)]);
        }

        int need_cpu_moe = 0;
        if (budgeted)
            while (need_cpu_moe <= hp.n_layer && rank_weight_bytes(need_cpu_moe) + kv_bytes > budget) need_cpu_moe++;

        n_cpu_moe = n_cpu_moe_req < 0 ? std::min(need_cpu_moe, hp.n_layer)
                                      : std::min(std::max(n_cpu_moe_req, 0), hp.n_layer);

        // Experts can only free so much; past that the context is what does not
        // fit, and shrinking it is the only remedy that keeps the model resident.
        const size_t w = rank_weight_bytes(n_cpu_moe);
        if (budgeted && w + kv_bytes > budget)
        {
            const int fit = w < budget ? ctx_that_fits(budget - w) : 0;
            if (fit < 256 || ctx_is_hard_limit)
            {
                // Only advice that can actually make this load fit. The message
                // used to end "Lower MAX_CONTEXT (0 tokens would fit) or add
                // --n-cpu-moe N" even when the weights alone overflowed a rank
                // and every routed expert was already counted in system RAM.
                std::string advice;
                auto add_advice = [&](const std::string & option)
                {
                    const bool first = advice.empty();
                    advice += first ? " " : "; or ";
                    advice += option;
                    if (first) advice[1] = (char) std::toupper((unsigned char) advice[1]);
                };
                char buf[256];
                if (fit >= 256)
                {
                    snprintf(buf, sizeof(buf), "set MAX_CONTEXT to %d or less", fit);
                    add_advice(buf);
                }
                // Fewest leading layers whose experts, kept in system RAM, make
                // the requested context fit at this rank count.
                int cpu_moe_fix = -1;
                for (int n = n_cpu_moe + 1; n <= hp.n_layer; n++)
                    if (rank_weight_bytes(n) + kv_bytes <= budget) { cpu_moe_fix = n; break; }
                const size_t w_all = rank_weight_bytes(hp.n_layer);
                const int fit_all = w_all < budget ? ctx_that_fits(budget - w_all) : 0;
                if (cpu_moe_fix > 0)
                {
                    snprintf(buf, sizeof(buf), "re-run with --n-cpu-moe %d (keeps the routed experts of the "
                             "first %d layer(s) in system RAM)", cpu_moe_fix, cpu_moe_fix);
                    add_advice(buf);
                }
                else if (n_cpu_moe < hp.n_layer && fit_all >= 256)
                {
                    snprintf(buf, sizeof(buf), "re-run with --cpu-moe and MAX_CONTEXT at most %d", fit_all);
                    add_advice(buf);
                }
                if (advice.empty())
                {
                    snprintf(buf, sizeof(buf), "Neither a shorter context nor --n-cpu-moe can fix this: with every "
                             "routed expert in system RAM a rank still needs %.1f GiB for the weights every rank "
                             "replicates. Run without --tp", w_all / 1073741824.0);
                    advice = std::string(" ") + buf +
                             " (the layer split stores each layer on one GPU instead of on every rank), "
                             "or on GPUs with more memory";
                }
                tsg::report_load_refusal(
                        "[glm] not enough VRAM for --tp %d: %.1f GiB per rank of weights plus %.1f GiB of KV and "
                        "graphs for a %d-token context, against %.1f GiB usable on the smallest rank.%s.\n",
                        m->tp, w / 1073741824.0, kv_bytes / 1073741824.0, m->n_ctx,
                        budget / 1073741824.0, advice.c_str());
                return nullptr;
            }
            if (!remeasure_ctx)
                fprintf(stderr, "[glm] context capped to %d tokens (the GGUF advertises %d): %.1f GiB per rank of "
                        "weights leaves %.1f GiB for the caches and graphs. Set MAX_CONTEXT to ask for a "
                        "different one.\n",
                        fit, m->n_ctx, w / 1073741824.0, (budget - w) / 1073741824.0);
            m->n_ctx = fit;
        }

        for (int il = 0; il < n_cpu_moe && il < hp.n_layer; il++) m->layers[il].cpu_moe = true;
        if (n_cpu_moe > 0)
            fprintf(stderr, "[glm] MoE CPU offload: this rank's experts for layers 0..%d stay in system RAM%s\n",
                    n_cpu_moe - 1, n_cpu_moe_req < 0 ? " (auto: they do not fit alongside the caches otherwise)" : "");
    }
    else if (n_gpu == 0)
    {
        // Pure CPU run: everything on the host backend.
        for (int il = 0; il < hp.n_layer; il++) { m->layers[il].device = 0; m->layers[il].cpu_moe = true; }
        n_cpu_moe = hp.n_layer;
    }
    else
    {
        std::vector<size_t> fixed_bytes((size_t) n_gpu, 0);
        fixed_bytes[0] += embd_bytes;
        fixed_bytes[(size_t) n_gpu - 1] += head_bytes;
        // The draft block's own MLA cache (no indexer cache — it attends densely).
        if (m->has_mtp)
            fixed_bytes[(size_t) n_gpu - 1] += (size_t) hp.n_kv_row * m->n_ctx * 2 + 1024;

        auto layer_cost = [&](int il, int n_cpu) -> size_t
        {
            size_t w = layer_bytes[il];
            if (il < n_cpu) w -= layer_exps_bytes[il];
            return w + layer_cache_bytes(il);
        };

        // Contiguous runs, in pipeline order: fill each device to `frac` of its
        // budget and report whether every layer landed.
        auto pack = [&](double frac, int n_cpu, std::vector<int> * out) -> bool
        {
            int dev = 0;
            size_t used = fixed_bytes[0];
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t cost = layer_cost(il, n_cpu);
                while (used + cost > (size_t) (dev_budget[dev] * frac))
                {
                    if (dev + 1 >= n_gpu) return false;
                    used = fixed_bytes[++dev];
                }
                used += cost;
                if (out) (*out)[il] = dev;
            }
            return true;
        };

        // An advertised context that does not fit is capped, not refused — the
        // same ceiling rule the tensor-parallel path uses. Only when MAX_CONTEXT
        // named the number is it treated as a requirement.
        if (!ctx_is_hard_limit)
        {
            const int want_cpu = n_cpu_moe_req < 0 ? 0 : std::min(n_cpu_moe_req, hp.n_layer);
            if (!pack(1.0, want_cpu, nullptr))
            {
                size_t weights = root_bytes;
                for (int il = 0; il < hp.n_layer; il++)
                {
                    weights += layer_bytes[il];
                    if (il < want_cpu) weights -= layer_exps_bytes[il];
                }
                size_t budget_total = 0;
                for (int d = 0; d < n_gpu; d++) budget_total += dev_budget[d];
                const int fit = weights < budget_total ? ctx_that_fits(budget_total - weights) : 0;
                if (fit >= 256 && fit < m->n_ctx)
                {
                    if (!remeasure_ctx)
                        fprintf(stderr, "[glm] context capped to %d tokens (the GGUF advertises %d) so the weights "
                                "and their caches fit the %d visible GPU(s). Set MAX_CONTEXT to ask for a "
                                "different one.\n", fit, m->n_ctx, n_gpu);
                    m->n_ctx = fit;
                }
            }
        }

        int need_cpu_moe = 0;
        while (need_cpu_moe <= hp.n_layer && !pack(1.0, need_cpu_moe, nullptr)) need_cpu_moe++;

        if (n_cpu_moe_req < 0)
        {
            n_cpu_moe = std::min(need_cpu_moe, hp.n_layer);
        }
        else
        {
            n_cpu_moe = std::min(n_cpu_moe_req, hp.n_layer);
            if (n_cpu_moe < need_cpu_moe && need_cpu_moe <= hp.n_layer)
            {
                size_t free_total = 0;
                for (int d = 0; d < n_gpu; d++) free_total += dev_free[d];
                size_t would_free = 0;
                for (int il = 0; il < need_cpu_moe && il < hp.n_layer; il++) would_free += layer_exps_bytes[il];
                tsg::report_load_refusal(
                        "[glm] not enough VRAM: %.1f GiB of weights plus this context's KV caches against %.1f GiB "
                        "free across %d device(s). Re-run with --n-cpu-moe %d (moves the routed experts of the "
                        "first %d layer(s), %.1f GiB, to system RAM) or --cpu-moe to offload every layer.\n",
                        total_bytes / 1073741824.0, free_total / 1073741824.0, n_gpu,
                        need_cpu_moe, need_cpu_moe, would_free / 1073741824.0);
                return nullptr;
            }
        }

        // Balance the largest fraction-of-budget used, then fall back to a
        // proportional split if even a perfect pack cannot fit.
        std::vector<int> assign((size_t) hp.n_layer, 0);
        double lo = 0.0, hi = 1.0;
        if (pack(1.0, n_cpu_moe, &assign))
        {
            for (int it = 0; it < 40; it++)
            {
                const double mid = 0.5 * (lo + hi);
                std::vector<int> tmp((size_t) hp.n_layer, 0);
                if (pack(mid, n_cpu_moe, &tmp)) { hi = mid; assign = tmp; }
                else lo = mid;
            }
        }
        else
        {
            size_t acc = 0;
            for (int il = 0; il < hp.n_layer; il++)
            {
                int dev = (int) ((acc * n_gpu) / (total_bytes + 1));
                if (dev >= n_gpu) dev = n_gpu - 1;
                assign[il] = dev;
                acc += layer_bytes[il];
            }
        }
        for (int il = 0; il < hp.n_layer; il++) m->layers[il].device = assign[il];
        for (int il = 0; il < n_cpu_moe && il < hp.n_layer; il++) m->layers[il].cpu_moe = true;

        if (n_cpu_moe > 0)
        {
            size_t host_bytes = 0;
            for (int il = 0; il < n_cpu_moe; il++) host_bytes += layer_exps_bytes[il];
            fprintf(stderr, "[glm] MoE CPU offload: routed experts of layers 0..%d (%.1f GiB) stay in system RAM "
                    "and run on the host%s\n", n_cpu_moe - 1, host_bytes / 1073741824.0,
                    n_cpu_moe_req < 0 ? " (auto: the model does not fit the visible VRAM otherwise)" : "");
        }
    }

    // The draft block lives with the LM head: it consumes the trunk's post-norm
    // hidden state and writes through the same head, so putting it anywhere else
    // would add two cross-device copies per draft step. Under tensor parallelism
    // `.device` is unused (each rank runs on its own), and its routed experts
    // follow the trunk's offload policy so `--cpu-moe` still fits the model.
    if (m->has_mtp)
    {
        glm_layer & M = m->layers[(size_t) m->mtp_layer];
        M.device = n_gpu > 0 ? m->layers[(size_t) hp.n_layer - 1].device : 0;
        M.cpu_moe = n_gpu == 0 || m->mtp_layer < n_cpu_moe;
    }

    // --- weight tensors ---------------------------------------------------
    for (int d = 0; d <= n_gpu; d++)
    {
        // ~28 tensors per layer per rank that lands on this device. Sized for the
        // worst case (every rank on one device, which TS_GLM_TP_OVERSUBSCRIBE
        // allows) rather than the expected one-rank-per-GPU.
        const size_t per_dev = (size_t) (32 * (hp.n_layer + (m->has_mtp ? 1 : 0)) * std::max(1, m->tp) + 64);
        ggml_init_params wp = { per_dev * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(wp);
        ggml_init_params cp = { (size_t) 64 * ggml_tensor_overhead(), nullptr, true };
        m->c_ctx[d] = ggml_init(cp);
    }

    // The LM head remains on rank 0. The segmented glm5next TP path gives every
    // rank its own embedding table so the independent graphs can construct the
    // initial residual locally; the scheduler fallback still reads rank 0's
    // copy exactly as before.
    const int dev_first = m->tp > 1 ? 0 : m->layers[0].device;
    const int dev_last = m->tp > 1 ? 0 : m->layers[(size_t) hp.n_layer - 1].device;

    weight_loader WL(*m, sources);

    m->tok_embd = WL.full(dev_first, true, "token_embd.weight");
    m->tok_embd_rank[0] = m->tok_embd;
    const bool load_rank_embeddings = hp.g5n && m->tp > 1 && m->tp <= n_gpu && m->tp_shard == 3 &&
        n_cpu_moe == 0 && !(getenv("TS_GLM_TRACE") && *getenv("TS_GLM_TRACE"));
    if (load_rank_embeddings)
    {
        for (int r = 1; r < m->tp; r++)
            m->tok_embd_rank[r] = WL.full(m->rank_device(r), true, "token_embd.weight");
    }
    m->output_norm = WL.full(dev_last, true, "output_norm.weight");
    m->output = WL.full(dev_last, false, "output.weight");
    if (!m->output)   // tied embeddings
        m->output = WL.full(dev_last, true, "token_embd.weight");
    if (!m->tok_embd || !m->output_norm || !m->output) return nullptr;
    if (load_rank_embeddings)
        for (int r = 0; r < m->tp; r++) if (!m->tok_embd_rank[r]) return nullptr;

    const int n_rank = m->tp;
    // Splitting the routed experts row-wise cuts every down-projection ROW, so a
    // rank's strip has to start and end on a quantization block boundary: the
    // split is counted in blocks, not elements. A model whose expert hidden size
    // is not a whole number of blocks, or one with fewer blocks than ranks, keeps
    // its experts whole and splits only the attention heads — slower, still exact.
    int moe_ff_blck = 1;
    {
        bool & ok = m->tp_moe_rows;
        for (int il = 0; il < hp.n_layer && n_rank > 1 && m->tp_experts(); il++)
        {
            char nm[256];
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            auto it = sources.find(nm);
            if (it == sources.end()) continue;
            const int b = (int) ggml_blck_size(it->second.type);
            moe_ff_blck = std::max(moe_ff_blck, b);
            if (hp.n_ff_exp % moe_ff_blck != 0 || hp.n_ff_exp / moe_ff_blck < n_rank)
            {
                ok = false;
                fprintf(stderr, "[glm] %d expert rows are not %d whole %d-element %s blocks: keeping the experts "
                        "whole and splitting only the attention heads\n",
                        hp.n_ff_exp, n_rank, moe_ff_blck, ggml_type_name(it->second.type));
            }
        }
    }

    // The draft block is an ordinary glm-dsa decoder block, so it loads through
    // exactly the same code — head sharding, expert sharding and CPU offload
    // included. The only differences are that its indexer weights are skipped
    // (it attends densely) and that it additionally carries the NextN wiring.
    const int n_load_layers = hp.n_layer + (m->has_mtp ? 1 : 0);
    for (int il = 0; il < n_load_layers; il++)
    {
        glm_layer & L = m->layers[il];

        for (int r = 0; r <= n_rank; r++)
        {
            const bool sh = n_rank > 1 && m->tp_heads();
            const bool se = n_rank > 1 && m->tp_experts();
            L.head_first[r] = sh
                ? shard_first(hp.n_head / m->tp_head_group, n_rank, r) * m->tp_head_group
                : (r == 0 ? 0 : hp.n_head);
            L.moe_ff_first[r] = se ? shard_first_rot(hp.n_ff_exp / moe_ff_blck, n_rank, r, il) * moe_ff_blck
                                   : (r == 0 ? 0 : hp.n_ff_exp);
        }

        for (int r = 0; r < n_rank; r++)
        {
            glm_layer_weights & W = L.w[r];
            const int d = n_rank > 1 ? m->rank_device(r) : L.device;
            const int64_t h0 = L.head_first[r];
            const int64_t hn = L.head_first[r + 1] - h0;

            // --- replicated on every rank -------------------------------------
            // Norms, the query/KV down-projections, the router and the indexer
            // are a few percent of the layer, and replicating them keeps the KV
            // rows and the top-k selection identical on every rank without a
            // collective.
            W.attn_norm = WL.full(d, true, "blk.%d.attn_norm.weight", il);
            W.ffn_norm  = WL.full(d, true, "blk.%d.ffn_norm.weight", il);

            if (hp.g5n)
            {
                // Every glm5next layer crosses the residual through Sinkhorn
                // hyper-connections, twice.
                W.hc_attn_fn    = WL.full(d, true, "blk.%d.hc_attn_fn.weight", il);
                W.hc_attn_scale = WL.full(d, true, "blk.%d.hc_attn_scale.weight", il);
                W.hc_attn_base  = WL.full(d, true, "blk.%d.hc_attn_base.weight", il);
                W.hc_ffn_fn     = WL.full(d, true, "blk.%d.hc_ffn_fn.weight", il);
                W.hc_ffn_scale  = WL.full(d, true, "blk.%d.hc_ffn_scale.weight", il);
                W.hc_ffn_base   = WL.full(d, true, "blk.%d.hc_ffn_base.weight", il);
            }

            if (L.recurrent)
            {
                // KDA is head-separable through the output projection. q/k/v,
                // the short convolution, decay/gate projections and recurrent
                // state are column-parallel by head; attn_output is row-parallel
                // and its rank partials are reduced before the hyper-connection.
                const int64_t c0 = h0 * hp.kda_head_dim;
                const int64_t cn = hn * hp.kda_head_dim;
                if (n_rank > 1 && m->tp_heads())
                {
                    W.kda_wq     = WL.slice_hi(d, 1, c0, cn, true, "blk.%d.attn_q.weight", il);
                    W.kda_wk     = WL.slice_hi(d, 1, c0, cn, true, "blk.%d.attn_k.weight", il);
                    W.kda_wv     = WL.slice_hi(d, 1, c0, cn, true, "blk.%d.attn_v.weight", il);
                    W.kda_wo     = WL.slice_lo(d, c0, cn, true, "blk.%d.attn_output.weight", il);
                    W.kda_conv_q = WL.slice_hi(d, 2, c0, cn, true, "blk.%d.ssm_conv1d_q.weight", il);
                    W.kda_conv_k = WL.slice_hi(d, 2, c0, cn, true, "blk.%d.ssm_conv1d_k.weight", il);
                    W.kda_conv_v = WL.slice_hi(d, 2, c0, cn, true, "blk.%d.ssm_conv1d_v.weight", il);
                    W.kda_f_a    = WL.full(d, true, "blk.%d.ssm_f_a.weight", il);
                    W.kda_f_b    = WL.slice_hi(d, 1, c0, cn, true, "blk.%d.ssm_f_b.weight", il);
                    W.kda_dt_b   = WL.slice_lo(d, c0, cn, true, "blk.%d.ssm_dt.bias", il);
                    W.kda_a      = WL.slice_lo(d, h0, hn, true, "blk.%d.ssm_a", il);
                    W.kda_beta   = WL.slice_hi(d, 1, h0, hn, true, "blk.%d.ssm_beta.weight", il);
                    W.kda_g_a    = WL.full(d, true, "blk.%d.ssm_g_a.weight", il);
                    W.kda_g_b    = WL.slice_hi(d, 1, c0, cn, true, "blk.%d.ssm_g_b.weight", il);
                    W.kda_o_norm = WL.full(d, true, "blk.%d.ssm_norm.weight", il);
                }
                else
                {
                    W.kda_wq     = WL.full(d, true, "blk.%d.attn_q.weight", il);
                    W.kda_wk     = WL.full(d, true, "blk.%d.attn_k.weight", il);
                    W.kda_wv     = WL.full(d, true, "blk.%d.attn_v.weight", il);
                    W.kda_wo     = WL.full(d, true, "blk.%d.attn_output.weight", il);
                    W.kda_conv_q = WL.full(d, true, "blk.%d.ssm_conv1d_q.weight", il);
                    W.kda_conv_k = WL.full(d, true, "blk.%d.ssm_conv1d_k.weight", il);
                    W.kda_conv_v = WL.full(d, true, "blk.%d.ssm_conv1d_v.weight", il);
                    W.kda_f_a    = WL.full(d, true, "blk.%d.ssm_f_a.weight", il);
                    W.kda_f_b    = WL.full(d, true, "blk.%d.ssm_f_b.weight", il);
                    W.kda_dt_b   = WL.full(d, true, "blk.%d.ssm_dt.bias", il);
                    W.kda_a      = WL.full(d, true, "blk.%d.ssm_a", il);
                    W.kda_beta   = WL.full(d, true, "blk.%d.ssm_beta.weight", il);
                    W.kda_g_a    = WL.full(d, true, "blk.%d.ssm_g_a.weight", il);
                    W.kda_g_b    = WL.full(d, true, "blk.%d.ssm_g_b.weight", il);
                    W.kda_o_norm = WL.full(d, true, "blk.%d.ssm_norm.weight", il);
                }
            }
            else
            {
            W.wq_a      = WL.full(d, true, "blk.%d.attn_q_a.weight", il);
            W.q_a_norm  = WL.full(d, true, "blk.%d.attn_q_a_norm.weight", il);
            W.wkv_a_mqa = WL.full(d, true, "blk.%d.attn_kv_a_mqa.weight", il);
            W.kv_a_norm = WL.full(d, true, "blk.%d.attn_kv_a_norm.weight", il);

            if (L.indexer_full)
            {
                W.idx_attn_q_b = WL.full(d, true, "blk.%d.indexer.attn_q_b.weight", il);
                W.idx_attn_k   = WL.full(d, true, "blk.%d.indexer.attn_k.weight", il);
                W.idx_k_norm_w = WL.full(d, true, "blk.%d.indexer.k_norm.weight", il);
                W.idx_k_norm_b = WL.full(d, true, "blk.%d.indexer.k_norm.bias", il);
                W.idx_proj     = WL.full(d, true, "blk.%d.indexer.proj.weight", il);
                if (hp.indexer_kpool > 0)
                {
                    W.idx_comp_gate = WL.full(d, true, "blk.%d.indexer_compressor_gate.weight", il);
                    W.idx_comp_ape  = WL.full(d, true, "blk.%d.indexer_compressor_ape.weight", il);
                }
            }

            // --- attention, sharded by head ------------------------------------
            if (n_rank > 1 && m->tp_heads())
            {
                W.wq_b = WL.slice_hi(d, 1, h0 * hp.n_embd_head_k, hn * hp.n_embd_head_k, true,
                                     "blk.%d.attn_q_b.weight", il);
                W.wk_b = WL.slice_hi(d, 2, h0, hn, true, "blk.%d.attn_k_b.weight", il);
                W.wv_b = WL.slice_hi(d, 2, h0, hn, true, "blk.%d.attn_v_b.weight", il);
                // Row-parallel: this rank's heads are a contiguous column range
                // of the output projection, and a head is n_embd_head_v elements,
                // so the split is always block-aligned.
                W.wo   = WL.slice_lo(d, h0 * hp.n_embd_head_v, hn * hp.n_embd_head_v, true,
                                     "blk.%d.attn_output.weight", il);
            }
            else
            {
                W.wq_b = WL.full(d, true, "blk.%d.attn_q_b.weight", il);
                W.wk_b = WL.full(d, true, "blk.%d.attn_k_b.weight", il);
                W.wv_b = WL.full(d, true, "blk.%d.attn_v_b.weight", il);
                W.wo   = WL.full(d, true, "blk.%d.attn_output.weight", il);
            }
            }   // !L.recurrent

            // --- FFN ------------------------------------------------------------
            if (!L.is_moe)
            {
                // Three dense layers in the whole model; replicating them costs
                // less than the collective a split would add.
                W.ffn_gate = WL.full(d, true, "blk.%d.ffn_gate.weight", il);
                W.ffn_up   = WL.full(d, true, "blk.%d.ffn_up.weight", il);
                W.ffn_down = WL.full(d, true, "blk.%d.ffn_down.weight", il);
            }
            else
            {
                W.ffn_gate_inp = WL.full(d, true, "blk.%d.ffn_gate_inp.weight", il);
                W.exp_probs_b  = WL.full(d, false, "blk.%d.exp_probs_b.bias", il);

                // Only the routed experts move to the host under --n-cpu-moe: the
                // router, the norms and the always-active shared expert are small
                // and every token needs them, so keeping them on the accelerator
                // saves a second host round trip for nothing.
                const int de = L.cpu_moe ? n_gpu : d;
                const int64_t f0 = L.moe_ff_first[r];
                const int64_t fn = L.moe_ff_first[r + 1] - f0;
                // Host-resident experts are left whole even under tensor
                // parallelism. Splitting them would save no host RAM and no host
                // time — the CPU backend already threads across every core — but
                // a strided strip cannot be served in place from the GGUF
                // mapping, so it would turn a mapped file into a private copy of
                // 200+ GiB of anonymous memory. Rank 0 evaluates these layers.
                if (n_rank > 1 && m->tp_experts() && !L.cpu_moe)
                {
                    // Column-parallel gate/up (a strip of the hidden dimension)
                    // feeding a row-parallel down, exactly as for a dense MLP —
                    // only here every one of the n_expert matrices is split the
                    // same way, so the ranks' outputs sum to the dense result.
                    W.ffn_gate_exps = WL.slice_mid(de, f0, fn, true, "blk.%d.ffn_gate_exps.weight", il);
                    W.ffn_up_exps   = WL.slice_mid(de, f0, fn, true, "blk.%d.ffn_up_exps.weight", il);
                    W.ffn_down_exps = WL.slice_lo(de, f0, fn, true, "blk.%d.ffn_down_exps.weight", il);
                }
                else
                {
                    W.ffn_gate_exps = WL.full(de, true, "blk.%d.ffn_gate_exps.weight", il);
                    W.ffn_up_exps   = WL.full(de, true, "blk.%d.ffn_up_exps.weight", il);
                    W.ffn_down_exps = WL.full(de, true, "blk.%d.ffn_down_exps.weight", il);
                }

                if (hp.n_expert_shared > 0)
                {
                    W.ffn_gate_shexp = WL.full(d, true, "blk.%d.ffn_gate_shexp.weight", il);
                    W.ffn_up_shexp   = WL.full(d, true, "blk.%d.ffn_up_shexp.weight", il);
                    W.ffn_down_shexp = WL.full(d, true, "blk.%d.ffn_down_shexp.weight", il);
                }
            }

            if (il == m->mtp_layer)
            {
                // NextN wiring. eh_proj runs before any split, so it is
                // replicated; embed_tokens and shared_head_head are optional and
                // absent from GLM-5.2, which shares the trunk's table and head.
                W.nextn_eh_proj   = WL.full(d, true,  "blk.%d.nextn.eh_proj.weight", il);
                W.nextn_enorm     = WL.full(d, true,  "blk.%d.nextn.enorm.weight", il);
                W.nextn_hnorm     = WL.full(d, true,  "blk.%d.nextn.hnorm.weight", il);
                W.nextn_head_norm = WL.full(d, false, "blk.%d.nextn.shared_head_norm.weight", il);
                W.nextn_embd      = WL.full(d, false, "blk.%d.nextn.embed_tokens.weight", il);
                W.nextn_head      = WL.full(d, false, "blk.%d.nextn.shared_head_head.weight", il);

                // The viability decision was made before placement (see above);
                // this only catches a source table that disagreed with it.
                if (!W.nextn_eh_proj || !W.nextn_enorm || !W.nextn_hnorm)
                {
                    fprintf(stderr, "[glm] the NextN/MTP block is missing its wiring; serving standard decode\n");
                    m->has_mtp = false;
                    m->mtp_layer = -1;
                }
            }

            bool complete = W.attn_norm && W.ffn_norm;
            if (L.recurrent)
                complete = complete && W.kda_wq && W.kda_wk && W.kda_wv && W.kda_wo
                        && W.kda_conv_q && W.kda_conv_k && W.kda_conv_v
                        && W.kda_f_a && W.kda_f_b && W.kda_dt_b && W.kda_a && W.kda_beta
                        && W.kda_g_a && W.kda_g_b && W.kda_o_norm;
            else
                complete = complete && W.wq_a && W.wq_b && W.wkv_a_mqa && W.wk_b && W.wv_b && W.wo;
            if (hp.g5n && il < hp.n_layer)
                complete = complete && W.hc_attn_fn && W.hc_attn_scale && W.hc_attn_base
                        && W.hc_ffn_fn && W.hc_ffn_scale && W.hc_ffn_base;
            if (!complete)
            {
                tsg::report_load_refusal( "[glm] layer %d rank %d is incomplete\n", il, r);
                return nullptr;
            }
        }
    }

    // Where each weight's bytes live, keyed by tensor pointer: sliced tensors
    // (tensor parallelism) read a sub-range that the GGUF tensor table does not
    // describe, so the loader carries its own list rather than re-deriving it.
    std::map<const ggml_tensor *, const pending_weight *> pending_by_tensor;
    for (const pending_weight & pw : WL.pending) pending_by_tensor[pw.t] = &pw;

    // --- host-resident experts straight from the GGUF mapping -------------
    // A private copy of 200+ GiB of experts is anonymous memory the kernel
    // cannot reclaim: under a container quota that is a silent OOM kill rather
    // than a slow load. File pages are evictable, so the same model loads
    // everywhere and degrades only to page-cache speed.
    if (ggml_get_first_tensor(m->w_ctx[n_gpu]) != nullptr)
    {
        const char * e = getenv("TS_GLM_MOE_MMAP");
        if (!(e && atoi(e) == 0))
        {
            ggml_context * hctx = m->w_ctx[n_gpu];
            std::vector<tsg_dsv4::file_warm_range> mapped_ranges;
            for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
            {
                auto it = pending_by_tensor.find(t);
                if (it == pending_by_tensor.end()) continue;
                const pending_weight & pw = *it->second;
                // Only a contiguous range can be served in place.
                if (pw.src_row != 0 || ggml_nbytes(t) != pw.bytes) continue;
                ggml_backend_buffer_t buf = mmap_shard(*m, shards, pw.shard);
                if (!buf) {
                    tsg::report_load_refusal("[glm] host weight mapping failed; refusing an unbounded private copy\n");
                    return nullptr;
                }
                const size_t mapped_size = m->mmap_owners[pw.shard]->size();
                if (pw.file_off > mapped_size || pw.bytes > mapped_size - pw.file_off) {
                    tsg::report_load_refusal("[glm] host weight %s exceeds mapped shard bounds\n", t->name);
                    return nullptr;
                }
                char * base = (char *) m->mmap_addrs[pw.shard];
                if (ggml_backend_tensor_alloc(buf, t, base + pw.file_off) != GGML_STATUS_SUCCESS) {
                    tsg::report_load_refusal("[glm] cannot bind mapped host weight %s\n", t->name);
                    return nullptr;
                }
                mapped_ranges.push_back({ pw.shard, pw.file_off, ggml_nbytes(t), t->data });
            }
            // TP ranks can hold separate tensor views of the same mapped host
            // experts. Memory allowance applies to unique file pages, not views.
            for (const auto & range : tsg_dsv4::merge_file_warm_ranges(std::move(mapped_ranges)))
                m->mmap_weight_bytes += range.bytes;
            if (m->mmap_weight_bytes > 0)
                fprintf(stderr, "[glm] host experts served from the GGUF mapping (%.1f GiB, no private copy)\n",
                        m->mmap_weight_bytes / 1073741824.0);
        }
    }

    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->w_ctx[d]) != nullptr)
        {
            bool unallocated = false;
            for (ggml_tensor * t = ggml_get_first_tensor(m->w_ctx[d]); t; t = ggml_get_next_tensor(m->w_ctx[d], t))
                if (t->data == nullptr) { unallocated = true; break; }
            if (unallocated)
            {
                m->w_buf[d] = tsg::graph_budget_alloc_ctx_tensors(m->w_ctx[d], m->backends[d], d);
                if (!m->w_buf[d]) { tsg::report_load_refusal( "[glm] weight allocation failed on device %d\n", d); return nullptr; }
                // Weights, and ggml_backend_sched has to be told so: only a
                // WEIGHTS buffer makes it place an op on the device that holds
                // the matrix. With the default usage it picks by its own
                // heuristics, and under tensor parallelism that means a rank's
                // mul_mat_id can be scheduled onto another rank's device and
                // dereference a pointer that device does not own.
                ggml_backend_buffer_set_usage(m->w_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
    }

    // --- upload -----------------------------------------------------------
    size_t uploaded = 0;
    {
        const size_t chunk = upload_option(getenv("TS_GLM_LOAD_CHUNK_MB"), size_t(64) * 1024 * 1024,
            1024 * 1024, (size_t) (std::numeric_limits<int64_t>::max)());
        int load_threads = (int) upload_option(getenv("TS_GLM_LOAD_THREADS"), 16, 1,
            (size_t) (std::numeric_limits<int>::max)());
        unsigned hw = std::thread::hardware_concurrency();
        if (hw > 0 && (unsigned) load_threads > hw) load_threads = (int) hw;

        std::vector<load_job> jobs;
        bool sizes_ok = true;
        for (const pending_weight & pw : WL.pending)
        {
            ggml_tensor * t = pw.t;
            if (t->buffer != nullptr &&
                std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), t->buffer) != m->mmap_bufs.end())
                continue;   // served from the GGUF mapping

            const size_t total = ggml_nbytes(t);
            if (total != pw.bytes)
            {
                fprintf(stderr, "[glm] size mismatch for %s: tensor %zu vs source %zu\n", t->name, total, pw.bytes);
                sizes_ok = false;
                continue;
            }
            if (!total) continue;
            validate_upload_range(pw.file_off, total, pw.src_row, pw.row_len);

            if (pw.src_row == 0)
            {
                for (size_t off = 0; off < total;) {
                    const size_t bytes = std::min(chunk, total - off);
                    jobs.push_back({ t, pw.shard, upload_add(pw.file_off, off), off, bytes, 0, 0 });
                    off += bytes;
                }
            }
            else
            {
                // Strided (row-parallel) slice: chunk by whole source rows.
                const size_t rows_total = total / pw.row_len;
                const size_t rows_per_chunk = std::max<size_t>(1, chunk / pw.row_len);
                for (size_t r0 = 0; r0 < rows_total;)
                {
                    const size_t rows = std::min(rows_per_chunk, rows_total - r0);
                    jobs.push_back({ t, pw.shard, upload_add(pw.file_off, upload_multiply(r0, pw.src_row)),
                                     upload_multiply(r0, pw.row_len), upload_multiply(rows, pw.row_len), pw.src_row, pw.row_len });
                    r0 += rows;
                }
            }
            uploaded = upload_add(uploaded, total);
        }
        if (!sizes_ok) return nullptr;
        if (!upload_parallel(shards, jobs, load_threads, chunk)) return nullptr;

        // Warm the mapped experts with the same parallelism the copy path had —
        // lazy faulting would charge the whole read to the first prompt at
        // single-stream storage speed. Only when they actually fit: warming
        // more than the allowance just evicts itself.
        if (m->mmap_weight_bytes > 0)
        {
            const uint64_t allow = tsg_glm_host::host_available_bytes();
            const size_t headroom = (size_t) 8 * 1024 * 1024 * 1024;
            if (tsg_glm_host::can_prefault(m->mmap_weight_bytes, allow, headroom))
            {
                const auto t_warm = std::chrono::steady_clock::now();
                std::vector<tsg_dsv4::file_warm_range> ranges;
                ggml_context * hctx = m->w_ctx[n_gpu];
                for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
                {
                    if (t->buffer == nullptr ||
                        std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), t->buffer) == m->mmap_bufs.end())
                        continue;
                    const auto pending = pending_by_tensor.find(t);
                    if (pending == pending_by_tensor.end())
                    {
                        fprintf(stderr, "[glm] missing file range for mapped host tensor %s\n", t->name);
                        return nullptr;
                    }
                    const pending_weight & pw = *pending->second;
                    ranges.push_back({ pw.shard, pw.file_off, ggml_nbytes(t), t->data });
                }
#if defined(_WIN32)
                // The Unix pread warmer deliberately has no Windows reader.
                // Only touch the mapped pages when the complete set fits the
                // current available RAM plus headroom; large checkpoints remain
                // demand mapped. This uses the existing OS cache and no copy.
                SYSTEM_INFO system_info = {};
                GetSystemInfo(&system_info);
                const size_t page_size = system_info.dwPageSize ? system_info.dwPageSize : 4096;
                size_t pages = 0;
                for (const auto & range : tsg_dsv4::merge_file_warm_ranges(std::move(ranges)))
                    pages += tsg_glm_host::prefault_pages(range.mapped, (size_t) range.bytes, page_size);
                fprintf(stderr, "[glm] prefaulted %.1f GiB of mmapped host experts in %.1fs "
                                "(Windows mapped pages, %zu touches; OS cache reused)\n",
                        m->mmap_weight_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(), pages);
#else
                // Faulting one mapped page at a time serializes small reads on
                // network filesystems. Reuse the bounded sequential reader;
                // populate only after pread has brought each block into cache.
                tsg_dsv4::file_warm_options options;
                options.threads = load_threads;
                options.populate = true;
                const auto result = tsg_dsv4::warm_file_ranges(shards.paths, ranges, options);
                if (!result.ok)
                {
                    fprintf(stderr, "[glm] prefaulting mapped host experts failed: %s\n", result.error.c_str());
                    return nullptr;
                }
                if (!result.budget_skipped)
                    fprintf(stderr, "[glm] prefaulted %.1f GiB of mmapped host experts in %.1fs "
                                "(pread, %d threads: %.1f GiB read, %.1f GiB already resident)\n",
                        m->mmap_weight_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(),
                        result.threads, result.bytes_read / 1073741824.0, result.bytes_resident / 1073741824.0);
#endif
            }
            else
            {
                fprintf(stderr, "[glm] skipping host-expert prefault: %.1f GiB mapped + 8 GiB headroom "
                                "exceeds %.1f GiB current available memory (0 means unknown)\n",
                        m->mmap_weight_bytes / 1073741824.0, allow / 1073741824.0);
            }
        }
    }

    // --- tensor-parallel expert lookup tables ------------------------------

    // --- indexer rotation constant ----------------------------------------
    if (is_pow2(hp.indexer_head_size))
    {
        gen_hadamard(m->hadamard_host, hp.indexer_head_size);
        std::vector<uint8_t> dev_needs((size_t) n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++)
        {
            if (!hp.indexer_full[il]) continue;
            if (m->tp > 1) { for (int r = 0; r < m->tp; r++) dev_needs[(size_t) m->rank_device(r)] = 1; }
            else dev_needs[(size_t) m->layers[il].device] = 1;
        }

        for (int d = 0; d <= n_gpu; d++)
        {
            if (!dev_needs[(size_t) d]) continue;
            m->hadamard[d] = ggml_new_tensor_2d(m->c_ctx[d], GGML_TYPE_F32,
                                                hp.indexer_head_size, hp.indexer_head_size);
            ggml_format_name(m->hadamard[d], "indexer_rot.%d", d);
        }
    }
    else
    {
        fprintf(stderr, "[glm] indexer key length %d is not a power of two; the Hadamard rotation is skipped "
                "(scores are mathematically the same, but F16 rounding will differ from llama.cpp)\n",
                hp.indexer_head_size);
    }

    // One allocation pass for every constant created above.
    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->c_ctx[d]) == nullptr) continue;
        m->c_buf[d] = tsg::graph_budget_alloc_ctx_tensors(m->c_ctx[d], m->backends[d], d);
        if (!m->c_buf[d]) { tsg::report_load_refusal( "[glm] constant allocation failed on device %d\n", d); return nullptr; }
        // These are per-device read-only constants — the Hadamard basis and, under
        // expert parallelism, the rank's expert mask. Marking the buffer as weights
        // is what makes ggml_backend_sched pin their consumers to this device; with
        // the default usage the scheduler is free to run the consumer elsewhere and
        // read the pointer from the wrong device.
        ggml_backend_buffer_set_usage(m->c_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (m->hadamard[d])
            ggml_backend_tensor_set(m->hadamard[d], m->hadamard_host.data(), 0,
                                    m->hadamard_host.size() * sizeof(float));
    }

    // --- flash attention / fused indexer probes ---------------------------
    if (n_gpu > 0)
    {
        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.n_kv_row, 1, hp.n_head, 1);
            ggml_tensor * k = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.n_kv_row, 256, 1, 1);
            ggml_tensor * v = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.kv_lora_rank, 256, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * fa = ggml_flash_attn_ext(pctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            m->flash_attn = ggml_backend_supports_op(m->backends[0], fa);
            ggml_free(pctx);
        }

        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.indexer_head_size, hp.indexer_n_head, 1, 1);
            // glm5next scores POOLS whose keys come out of ggml_get_rows in
            // F32; glm-dsa scores the F16 cache directly.
            ggml_tensor * k = ggml_new_tensor_4d(pctx, hp.g5n ? GGML_TYPE_F32 : GGML_TYPE_F16,
                                                 hp.indexer_head_size, 1, 256, 1);
            ggml_tensor * w = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.indexer_n_head, 1, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * li = ggml_lightning_indexer(pctx, q, k, w, mask);
            m->fused_lid = ggml_backend_supports_op(m->backends[0], li);
            ggml_free(pctx);
        }
    }

    if (hp.g5n)
    {
        ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
        ggml_context * pctx = ggml_init(pp);
        const int64_t hcn = hp.hc_mult;
        ggml_tensor * x = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hp.n_embd, hcn, 1);
        ggml_tensor * w = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hcn, 1);
        ggml_tensor * c = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hcn, hcn, 1);
        ggml_tensor * xf = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hp.n_embd, 1);
        ggml_tensor * hpre = ggml_dsv4_hc_pre(pctx, x, w);
        ggml_tensor * hpost = ggml_dsv4_hc_post(pctx, xf, x, w, c);
        m->hc_native = ggml_backend_supports_op(m->backends[0], hpre)
                    && ggml_backend_supports_op(m->backends[0], hpost);
        ggml_free(pctx);
        fprintf(stderr, "[glm] hyper-connection ops: %s\n",
                m->hc_native ? "native" : "decomposed (backend has no fused kernel)");
    }

    if (hp.g5n && m->tp > 1)
    {
        const char * why = nullptr;
        if (m->tp > n_gpu) why = "ranks are oversubscribed on the visible GPUs";
        else if (m->tp_shard != 3 || !m->tp_experts()) why = "heads and routed-expert rows are not both sharded";
        else if (n_cpu_moe != 0) why = "CPU MoE offload is active";
        else if (!m->hc_native) why = "the backend lacks native hyper-connection kernels";
        else if (getenv("TS_GLM_TRACE") && *getenv("TS_GLM_TRACE")) why = "tensor tracing is active";
        else
        {
            m->tp_fused = true;
            const bool device_comm = init_glm_tp_comm(*m);
            fprintf(stderr,
                    "[glm] TP executor: segmented rank-local graphs, concurrent submission, AllReduce=%s\n",
                    device_comm ? "backend collective (verified on first use)" : "host staging fallback");
        }
        if (why)
            fprintf(stderr, "[glm] TP executor: combined scheduler fallback (%s)\n", why);
    }

    m->op_offload = (n_cpu_moe == 0);
    if (const char * e = getenv("TS_GLM_OP_OFFLOAD")) m->op_offload = atoi(e) != 0;

    // Preserve exact loader ranges without opening any additional file handles
    // on the default path. Only the opt-in adapter registers these sources.
    if (hp.g5n && m->tp == 1 && n_gpu == 1 && !m->has_mtp) {
        std::vector<std::array<int, 3>> groups;
        bool complete = true;
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& layer = m->layers[il];
            const auto& weights = layer.w[0];
            if (!layer.cpu_moe || !weights.ffn_gate_exps) continue;
            ggml_tensor* parts[3] = {weights.ffn_gate_exps, weights.ffn_up_exps, weights.ffn_down_exps};
            std::array<int, 3> types{};
            for (int p = 0; p < 3; ++p) {
                auto pw = pending_by_tensor.find(parts[p]);
                if (!parts[p] || !parts[p]->data || pw == pending_by_tensor.end()
                    || pw->second->src_row != 0 || ggml_nbytes(parts[p]) != pw->second->bytes
                    || pw->second->file_off > static_cast<size_t>(INT64_MAX)
                    || pw->second->bytes > static_cast<size_t>(INT64_MAX)) { complete = false; break; }
                const auto& range = *pw->second;
                m->selected_expert_ranges.push_back({parts[p]->data, static_cast<int64_t>(range.bytes),
                    shards.paths[range.shard], static_cast<int64_t>(range.file_off)});
                types[p] = parts[p]->type;
            }
            if (!complete) break;
            if (std::find(groups.begin(), groups.end(), types) == groups.end()) groups.push_back(types);
        }
        if (complete && !m->selected_expert_ranges.empty()) m->selected_expert_groups = static_cast<int>(groups.size());
        else m->selected_expert_ranges.clear();
    }

    // The weights are resident now, so stop estimating: ask the devices how much
    // they actually have left and size the caches to that. The pre-load estimate
    // above only has to be conservative enough to refuse the hopeless cases; this
    // is what decides the context the session really gets.
    if (!ctx_is_hard_limit && n_gpu > 0 && (m->tp <= 1 || m->tp <= n_gpu))
    {
        // How many sequence slots the context must leave room for. Each slot is
        // a full-context set of cache rows, so a server running N sequences
        // needs N of everything per-token below. Defaults to the interactive
        // case; a serving config sets TS_GLM_PLAN_SLOTS instead of hand-tuning
        // MAX_CONTEXT.
        size_t plan_slots = 1;
        if (const char * e = getenv("TS_GLM_PLAN_SLOTS")) { long v = atol(e); if (v >= 1 && v <= 256) plan_slots = (size_t) v; }
        const size_t reserve = (size_t) 1024 * 1024 * 1024;   // leave the driver room to breathe
        int fit = 0;
        if (m->tp > 1)
        {
            // Every rank holds a full-length copy of every cache.
            size_t free_min = SIZE_MAX;
            for (int r = 0; r < m->tp; r++)
            {
                size_t free_b = 0, total_b = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(m->backends[m->rank_device(r)]), &free_b, &total_b);
                free_min = std::min(free_min, free_b);
            }
            // The LM-head graph and the DSA masks exist once per rank, not per
            // slot; only the cache rows and KDA state scale with plan_slots.
            const size_t fixed = reserve + graph_bytes_fixed + (plan_slots - 1) * kv_bytes_fixed;
            const size_t per_tok = plan_slots * kv_bytes_per_token + graph_bytes_per_token;
            const size_t avail = free_min > fixed ? free_min - fixed : 0;
            size_t t = per_tok > 0 ? avail / per_tok : 0;
            t -= t % 256;
            fit = (int) std::min((size_t) ctx_requested, t);
        }
        else
        {
            // The caches land where their layers live (slot_alloc), so size the
            // context against each device's actual share of the rows. Assuming
            // an even 1/n_gpu spread over-commits whichever device carries the
            // most cache layers: GLM-5.2 on 3 GPUs sized itself to 342k tokens
            // this way and then failed its very first slot allocation.
            const int n_cache_layers = hp.n_layer + (m->has_mtp ? 1 : 0);
            std::vector<size_t> tok_b((size_t) n_gpu, 0);      // per-slot, per-token
            std::vector<size_t> fix_slot_b((size_t) n_gpu, 0); // per-slot, fixed
            std::vector<size_t> gtok_b((size_t) n_gpu, 0);     // shared graph, per-token
            for (int il = 0; il < n_cache_layers; il++)
            {
                const int d = m->layers[(size_t) il].device;
                if (d < 0 || d >= n_gpu) continue;
                if (il < hp.n_layer && m->layers[(size_t) il].recurrent)
                {
                    fix_slot_b[(size_t) d] += kda_state_bytes;
                    continue;
                }
                tok_b[(size_t) d] += (size_t) hp.n_kv_row * 2;
                if (il < hp.n_layer && hp.indexer_full[(size_t) il])
                {
                    tok_b[(size_t) d] += idx_row_bytes;
                    // this layer's DSA top-k masks: [n_kv, n_ubatch] F16, twice
                    // over for the graph cache, shared across slots
                    gtok_b[(size_t) d] += 2 * (size_t) m->n_ubatch * 2;
                }
            }
            // The LM-head output buffer dominates graph_bytes_fixed and is
            // allocated where the head runs - the last trunk layer's device
            // (the MTP block follows it there). The KDA state share of that
            // constant is per-slot and already charged via fix_slot_b, so
            // subtract it back out.
            const int dev_head = m->layers[(size_t) hp.n_layer - 1].device;
            const size_t graph_fixed_head = graph_bytes_fixed > kv_bytes_fixed
                                          ? graph_bytes_fixed - kv_bytes_fixed : 0;
            fit = ctx_requested;
            for (int d = 0; d < n_gpu; d++)
            {
                const size_t per_tok = plan_slots * tok_b[(size_t) d] + gtok_b[(size_t) d];
                const size_t fixed = reserve + (d == dev_head ? graph_fixed_head : 0)
                                   + plan_slots * fix_slot_b[(size_t) d];
                if (per_tok == 0 && fix_slot_b[(size_t) d] == 0 && d != dev_head)
                    continue;   // no caches, no graphs: this device does not constrain
                size_t free_b = 0, total_b = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
                const size_t avail = free_b > fixed ? free_b - fixed : 0;
                int dev_fit;
                if (per_tok > 0)
                {
                    size_t t = avail / per_tok;
                    t -= t % 256;
                    dev_fit = (int) std::min((size_t) ctx_requested, t);
                }
                else
                {
                    // Only fixed-size state lives here (all-KDA device, or just
                    // the head graph): the context is unconstrained as long as
                    // that state actually fits.
                    dev_fit = avail > 0 || fixed <= free_b ? ctx_requested : 0;
                }
                fit = std::min(fit, dev_fit);
            }
        }
        if (fit < 256)
        {
            // Refuse now, with the reason, rather than letting slot_alloc fail
            // after the whole load: the measurement above mirrors exactly what
            // slot_alloc is about to allocate.
            tsg::report_load_refusal( "[glm] not enough VRAM left after the weights for even a 256-token context%s. "
                    "Lower TS_GLM_PLAN_SLOTS%s, set MAX_CONTEXT explicitly, or add --n-cpu-moe N to move "
                    "expert weights off the GPUs.\n",
                    plan_slots > 1 ? " per slot" : "",
                    plan_slots > 1 ? "" : " (unset)");
            return nullptr;
        }
        if (fit != m->n_ctx)
        {
            if (fit < ctx_requested)
                fprintf(stderr, "[glm] context %d tokens (the GGUF advertises %d)%s: sized to what is free on each "
                        "device after the weights, with the caches and graphs placed where their layers are. Set "
                        "MAX_CONTEXT to ask for a different one.\n", fit, ctx_requested,
                        plan_slots > 1 ? " per slot" : "");
            m->n_ctx = fit;
        }
    }

    m->active_slot = slot_alloc(*m);
    if (!m->active_slot) { tsg::report_load_refusal( "[glm] primary slot allocation failed\n"); return nullptr; }

    m->logits.resize((size_t) hp.n_vocab);

    {
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        // Say whether the NextN block was actually LOADED, not just that the
        // metadata declares one: the two differ whenever speculation was not
        // requested, and "(+1 MTP)" on a run with no draft head is exactly the
        // kind of log line that costs an hour.
        const char * mtp_state = m->has_mtp ? " +1 NextN/MTP draft block"
                               : (hp.n_layer_nextn > 0 ? (hp.g5n ? " (NextN block present; glm5next draft heads are not implemented)"
                                                               : " (NextN block present but not loaded; --spec loads it)")
                                                       : "");
        fprintf(stderr, "[glm] %s: %d trunk layers%s, n_embd=%d, %d heads, MLA(q_lora=%d, kv_lora=%d, "
                "head_k=%d, head_v=%d, rope=%d), %d experts top-%d (ff=%d), dense_lead=%d, indexer %dx%d top-%d on "
                "%d/%d layers, vocab=%d\n",
                hp.g5n ? "glm5next" : "glm-dsa", hp.n_layer, mtp_state, hp.n_embd, hp.n_head, hp.q_lora_rank, hp.kv_lora_rank,
                hp.n_embd_head_k, hp.n_embd_head_v, hp.n_rot, hp.n_expert, hp.n_expert_used, hp.n_ff_exp,
                hp.n_dense_lead, hp.indexer_n_head, hp.indexer_head_size, hp.indexer_top_k,
                (int) std::count(hp.indexer_full.begin(), hp.indexer_full.end(), (uint8_t) 1), hp.n_layer, hp.n_vocab);
        fprintf(stderr, "[glm] loaded %.1f GiB across %d GPU(s) in %.1fs (%.2f GiB/s%s); n_ctx=%d, n_ubatch=%d, "
                "flash_attn=%s, fused_indexer=%s\n",
                uploaded / 1073741824.0, n_gpu, secs, secs > 0 ? uploaded / 1073741824.0 / secs : 0.0,
                m->mmap_weight_bytes > 0 ? ", host experts mmapped" : "",
                m->n_ctx, m->n_ubatch, m->flash_attn ? "on" : "off", m->fused_lid ? "on" : "off");
        if (m->tp > 1)
        {
            // The expert rows are split in whole quantization blocks and the
            // remainder rotates by layer, so quote the per-rank average rather
            // than one layer's slice.
            int rows0 = 0;
            for (int il = 0; il < hp.n_layer; il++)
                if (m->layers[il].is_moe) rows0 += m->layers[il].moe_ff_first[1];
            const int moe_layers = std::max(1, hp.n_layer - hp.n_dense_lead);
            fprintf(stderr, "[glm] tensor parallel across %d rank(s): every layer on every rank, "
                    "heads %d/%d and ~%d/%d expert rows per rank, 2 reductions per layer\n",
                    m->tp, m->layers[0].head_first[1], hp.n_head, rows0 / moe_layers, hp.n_ff_exp);
        }
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            if (m->tp > 1)
            {
                if (d < m->tp)
                    fprintf(stderr, "[glm]   rank %d: all %d layers, %.1f GiB free after load\n",
                            d, hp.n_layer, free_b / 1073741824.0);
                continue;
            }
            int first = -1, last = -1, count = 0;
            for (int il = 0; il < hp.n_layer; il++)
                if (m->layers[il].device == d) { if (first < 0) first = il; last = il; count++; }
            if (count > 0)
                fprintf(stderr, "[glm]   device %d: layers %d..%d (%d), %.1f GiB free after load\n",
                        d, first, last, count, free_b / 1073741824.0);
            else
                fprintf(stderr, "[glm]   device %d: no layers, %.1f GiB free after load\n", d, free_b / 1073741824.0);
        }
    }

    return m.release();
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

static tsg::HostMoeSegment selected_expert_segment(const glm_model& m, int layer, int rows, bool joint)
{
    const auto& hp = m.hp;
    const auto& w = m.layers.at(layer).w[0];
    tsg::HostMoeSegment segment;
    segment.layer = layer; segment.hidden = hp.n_embd; segment.n_ff = static_cast<int>(w.ffn_gate_exps->ne[1]);
    segment.num_experts = hp.n_expert; segment.n_used = hp.n_expert_used;
    segment.seq_len = rows; segment.independent_decode_rows = joint;
    segment.swiglu_clamp = hp.swiglu_clamp;
    segment.pooled_model_experts = hp.n_expert > 0 && hp.n_layer <= 65536 / hp.n_expert
        ? hp.n_layer * hp.n_expert : 0;
    segment.gate_data = w.ffn_gate_exps->data; segment.gate_type = w.ffn_gate_exps->type;
    segment.gate_ne0 = w.ffn_gate_exps->ne[0]; segment.gate_ne1 = w.ffn_gate_exps->ne[1];
    segment.gate_bytes = ggml_nbytes(w.ffn_gate_exps);
    segment.up_data = w.ffn_up_exps->data; segment.up_type = w.ffn_up_exps->type;
    segment.up_ne0 = w.ffn_up_exps->ne[0]; segment.up_ne1 = w.ffn_up_exps->ne[1];
    segment.up_bytes = ggml_nbytes(w.ffn_up_exps);
    segment.down_data = w.ffn_down_exps->data; segment.down_type = w.ffn_down_exps->type;
    segment.down_ne0 = w.ffn_down_exps->ne[0]; segment.down_ne1 = w.ffn_down_exps->ne[1];
    segment.down_bytes = ggml_nbytes(w.ffn_down_exps);
    return segment;
}

struct graph_builder
{
    glm_model & m;
    const glm_hparams & hp;
    ggml_context * ctx;
    ggml_cgraph * gf;
    graph_build_result & res;
    glm_slot & slot;
    int64_t nt;      // tokens in this ubatch
    int64_t p0;      // position of the first token
    int64_t n_kv;    // cache columns the attention may read
    bool sparse;     // the indexer can actually drop something at this length

    graph_builder(glm_model & model, graph_build_result & r, glm_slot & s,
                  int64_t nt_, int64_t p0_, int64_t n_kv_, bool sparse_)
        : m(model), hp(model.hp), ctx(r.ctx), gf(r.gf), res(r), slot(s),
          nt(nt_), p0(p0_), n_kv(n_kv_), sparse(sparse_) {}

    /// Device that runs layer `il` for rank `r`: the rank's device under tensor
    /// parallelism, the layer's home device under a layer split.
    int device_of(int il, int r) const { return m.tp > 1 ? m.rank_device(r) : m.layers[il].device; }

    // A verify window must commit the same arithmetic as single-token decode.
    // On the real IQ1_S glm5next model the first four-row verify changed logits
    // by 0.56 before any rollback, and accepted rows carried that error forward.
    // Keep narrow projections on one-column reductions (broadcast channels for
    // ordinary weights), and routed experts on the one-token expert kernel.
    // Wider prefill graphs keep their throughput kernels.
    bool decode_rows() const { return hp.g5n && nt > 1 && nt <= TSG_PRECISION_DECODE_COLUMNS; }

    ggml_tensor * decode_mul_mat(ggml_context * c, ggml_tensor * w, ggml_tensor * x, bool widened_constant_weight = false)
    {
        if (!decode_rows() || x->ne[1] != nt || x->ne[3] != 1)
            return ggml_mul_mat(c, w, x);
        // Computed hyper-connection matrices use dimension 1 for streams,
        // not tokens. Do not confuse an equal stream count with a verify width.
        const ggml_tensor * base = w;
        while (base->view_src) base = base->view_src;
        if (base->op != GGML_OP_NONE && !widened_constant_weight) return ggml_mul_mat(c, w, x);
        if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
        const bool fp = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16;
        if (x->ne[2] == 1 && w->ne[2] == 1 && w->ne[3] == 1 &&
            ((fp && w->ne[0] % 2 == 0) || ggml_is_quantized(w->type)))
        {
            ggml_tensor * y = ggml_mul_mat(c, w, ggml_reshape_3d(c, x, x->ne[0], 1, nt));
            return ggml_reshape_2d(c, y, y->ne[0], nt);
        }
        ggml_tensor * result = nullptr;
        for (int64_t r = 0; r < nt; ++r)
        {
            ggml_tensor * row = ggml_cont(c, ggml_view_3d(c, x, x->ne[0], 1, x->ne[2],
                x->nb[1], x->nb[2], (size_t) r * x->nb[1]));
            row = ggml_mul_mat(c, w, row);
            result = result ? ggml_concat(c, result, row, 1) : row;
        }
        return result;
    }

    ggml_tensor * decode_mul_mat_id(ggml_context * c, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids)
    {
        if (!decode_rows() || x->ne[2] != nt || x->ne[3] != 1 || ids->ne[1] != nt)
            return ggml_mul_mat_id(c, w, x, ids);
        ggml_tensor * result = nullptr;
        for (int64_t r = 0; r < nt; ++r)
        {
            ggml_tensor * row = ggml_cont(c, ggml_view_3d(c, x, x->ne[0], x->ne[1], 1,
                x->nb[1], x->nb[2], (size_t) r * x->nb[2]));
            ggml_tensor * selected = ggml_view_2d(c, ids, ids->ne[0], 1, ids->nb[1], (size_t) r * ids->nb[1]);
            row = ggml_mul_mat_id(c, w, row, selected);
            result = result ? ggml_concat(c, result, row, 2) : row;
        }
        return result;
    }

    // Precision belongs on the actual matmuls, below the layout-only wrappers.
    static void decode_prec_set_acc(ggml_tensor * t, ggml_prec precision)
    {
        while (t->view_src) t = t->view_src;
        if (t->op == GGML_OP_CONCAT)
        {
            decode_prec_set_acc(t->src[0], precision);
            decode_prec_set_acc(t->src[1], precision);
        }
        else ggml_prec_set_acc(t, precision);
    }

    bool sensitive_projection_executor() const
    {
        // Full-model dispatch is qualified on the owned CUDA executor. Keep
        // other backend combinations on their existing arithmetic, rather than
        // silently introducing CPU custom-op copies or widened graph payloads.
        if (!hp.g5n || m.n_gpu <= 0) return false;
        for (int d = 0; d < m.n_gpu; ++d)
            if (!m.projection_backends[d]) return false;
        return true;
    }

    ggml_tensor * role_mul_mat(ggml_context * c, ggml_tensor * w, ggml_tensor * x, tsg_projection_role role)
    {
        if (!sensitive_projection_executor()) return decode_mul_mat(c, w, x);
        // Keep the original 2D activation columns for the owned kernel:
        // it preserves F32 activation bits with one K-ordered sum per column
        // at N1, true joint decode and wide prefill, without packing scratch.
        if (m.strict_recurrent_projections) {
            if (auto* quant = tsg_projection_quant_f32_if_required(c, w, x, role)) return quant;
        } else if (auto* q8 = tsg_projection_q8_f32_if_required(c, w, x, role)) return q8;
        auto* weight = tsg_projection_weight(c, w, role);
        // Only this explicit cast of a constant checkpoint projection may
        // bypass the computed-HC-matrix exclusion in narrow row construction.
        // Keep N2..N8 on the same single-column reduction as scalar decode.
        const ggml_tensor* original = w;
        while (original->view_src) original = original->view_src;
        auto* result = decode_mul_mat(c, weight, x, weight != w && original->op == GGML_OP_NONE);
        tsg_require_projection_precision(c, result, role, m.n_gpu > 0 && m.projection_backends[0] != nullptr);
        return result;
    }

    ggml_tensor * recurrent_projection(ggml_context * c, ggml_tensor * w, ggml_tensor * x,
                                      tsg_projection_role role)
    {
        // These larger checkpoint projections are a separate, explicit
        // candidate. Preserve existing scalar/joint/TP arithmetic by default.
        return m.strict_recurrent_projections && sensitive_projection_executor()
            ? role_mul_mat(c, w, x, role) : decode_mul_mat(c, w, x);
    }

    /// Sum of one partial per rank. With a single rank this is the identity, so
    /// the layer-split path pays nothing for it.
    ggml_tensor * reduce_ranks(ggml_tensor ** parts, int n)
    {
        ggml_tensor * acc = parts[0];
        for (int r = 1; r < n; r++) acc = ggml_add(ctx, acc, parts[r]);
        return acc;
    }

    // ---- batched decode ---------------------------------------------------
    // One graph, N tokens, N different sequences. Everything that does not touch
    // a cache is shared across the batch; only the cache write, the indexer
    // scoring and the attention itself are built per token, because each token
    // sees a different slot's history of a different length.

    /// The top-k mask for a single token, over that token's own n_kv.
    ggml_tensor * build_topk_mask_1(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k, int64_t nkv)
    {
        const int64_t n_top_k = top_k->ne[0];

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);                // [nkv, 1]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, nkv, 1, base->nb[0], base->nb[1], 0);
        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, 1, 1, top_k->nb[1], top_k->nb[1], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, 1), 0.0f);

        ggml_tensor * unmasked = ggml_set_rows(ctx, all, zeros, idx);
        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, nkv, 1, unmasked->nb[2], 0);
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);
        pin(base, dev);
        pin(zeros, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    /// Indexer for a batch of single tokens. The projections run once over the
    /// whole batch; the cache append and the scoring run per token.
    void build_indexer_bd(int il, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                          bool score_now, std::vector<ggml_tensor *> & top_k)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[0];
        const int dev = device_of(il, 0);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;
        const int64_t N = nt;

        ggml_tensor * k = decode_mul_mat(ctx, LW.idx_attn_k, cur);           // [D, N]
        k = ggml_norm(ctx, k, 0.0f);
        k = ggml_mul(ctx, k, LW.idx_k_norm_w);
        k = ggml_add(ctx, k, LW.idx_k_norm_b);
        k = ggml_reshape_3d(ctx, k, D, 1, N);
        k = rope(k, pos);
        if (m.hadamard[dev]) k = decode_mul_mat(ctx, m.hadamard[dev], k);
        k = ggml_cont(ctx, k);

        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            ggml_tensor * cache = m.slots.at(B.slot_id)->idx_k[0][il];
            ggml_tensor * ki = ggml_cont(ctx, ggml_view_2d(ctx, k, D, 1, k->nb[1], (size_t) i * k->nb[2]));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, ki, B.kv_idx[dev]));
        }

        if (!score_now) return;

        ggml_tensor * q = decode_mul_mat(ctx, LW.idx_attn_q_b, qr);          // [D*H, N]
        q = ggml_reshape_3d(ctx, q, D, H, N);
        q = rope(q, pos);
        if (m.hadamard[dev]) q = decode_mul_mat(ctx, m.hadamard[dev], q);
        q = ggml_cont(ctx, q);

        ggml_tensor * w = decode_mul_mat(ctx, LW.idx_proj, cur);             // [H, N]
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            if (!B.sparse) { top_k[(size_t) i] = nullptr; continue; }

            ggml_tensor * cache = m.slots.at(B.slot_id)->idx_k[0][il];
            ggml_tensor * k_all = ggml_view_3d(ctx, cache, D, 1, B.n_kv, cache->nb[1], cache->nb[1], 0);
            // Materialised rather than passed as offset views: a fused kernel is
            // free to assume its small operands start at their buffer.
            ggml_tensor * qi = ggml_cont(ctx, ggml_view_3d(ctx, q, D, H, 1, q->nb[1], q->nb[2], (size_t) i * q->nb[2]));
            ggml_tensor * wi = ggml_cont(ctx, ggml_view_2d(ctx, w, H, 1, w->nb[1], (size_t) i * w->nb[1]));

            ggml_tensor * score = nullptr;
            if (m.fused_lid)
            {
                score = ggml_lightning_indexer(ctx, qi, k_all, wi, B.lid_mask[dev]);
            }
            else
            {
                ggml_tensor * qp = ggml_permute(ctx, qi, 0, 2, 1, 3);      // [D, 1, H]
                ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);   // [D, n_kv, 1]
                ggml_tensor * kq = decode_mul_mat(ctx, kp, qp);              // [n_kv, 1, H]
                decode_prec_set_acc(kq, GGML_PREC_F32);
                kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));    // [H, 1, n_kv]
                ggml_tensor * sc = ggml_relu(ctx, kq);
                sc = ggml_mul(ctx, sc, wi);
                sc = ggml_sum_rows(ctx, sc);
                sc = ggml_cont(ctx, ggml_permute(ctx, sc, 2, 1, 0, 3));    // [n_kv, 1, 1]
                score = ggml_add(ctx, sc, ggml_cast(ctx, B.lid_mask[dev], GGML_TYPE_F32));
            }
            const int n_top_k = (int) std::min<int64_t>(B.n_kv, hp.indexer_top_k);
            top_k[(size_t) i] = ggml_cont(ctx, ggml_top_k(ctx, score, n_top_k));
        }
    }

    /// Attention for a batch of single tokens: shared projections, per-token
    /// cache write and softmax, then a shared V decompression and output
    /// projection over the re-assembled batch.
    ggml_tensor * build_attention_bd(int il, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                     const std::vector<ggml_tensor *> & masks)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[0];
        const int dev = device_of(il, 0);
        const int64_t n_head = hp.n_head;
        const int64_t N = nt;

        ggml_tensor * q = decode_mul_mat(ctx, LW.wq_b, qr);
        ggml_tensor * kv_pe = decode_mul_mat(ctx, LW.wkv_a_mqa, cur);

        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        if (hp.n_rot == 0)
        {
            // glm5next: nope-only, the latent IS the cache row.
            ggml_tensor * kv_cmpr = rms(kv_pe, LW.kv_a_norm);
            ggml_tensor * q3 = ggml_reshape_3d(ctx, q, hp.n_embd_head_k, n_head, N);
            ggml_tensor * q_nope_p = ggml_permute(ctx, q3, 0, 2, 1, 3);
            ggml_tensor * q_abs = decode_mul_mat(ctx, LW.wk_b, q_nope_p);          // [kv_lora, N, n_head]
            Qcur = ggml_cont(ctx, ggml_permute(ctx, q_abs, 0, 2, 1, 3));         // [kv_lora, n_head, N]
            Kcur = ggml_cont(ctx, ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, N));
        }
        else
        {
        ggml_tensor * q_nope = ggml_view_3d(ctx, q, hp.n_nope, n_head, N,
                                            ggml_row_size(q->type, hp.n_embd_head_k),
                                            ggml_row_size(q->type, hp.n_embd_head_k) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx, q, hp.n_rot, n_head, N,
                                          ggml_row_size(q->type, hp.n_embd_head_k),
                                          ggml_row_size(q->type, hp.n_embd_head_k) * n_head,
                                          ggml_row_size(q->type, hp.n_nope));

        const size_t row = ggml_row_size(kv_pe->type, hp.n_kv_row);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx, kv_pe, hp.kv_lora_rank, N, row, 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx, kv_pe, hp.n_rot, 1, N, row, row,
                                          ggml_row_size(kv_pe->type, hp.kv_lora_rank));

        q_pe = rope(q_pe, pos);
        k_pe = rope(k_pe, pos);
        kv_cmpr = rms(kv_cmpr, LW.kv_a_norm);

        ggml_tensor * q_nope_p = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_abs = decode_mul_mat(ctx, LW.wk_b, q_nope_p);              // [kv_lora, N, n_head]
        q_abs = ggml_permute(ctx, q_abs, 0, 2, 1, 3);                           // [kv_lora, n_head, N]

        Qcur = ggml_cont(ctx, ggml_concat(ctx, q_abs, q_pe, 0));                // [n_kv_row, n_head, N]
        Kcur = ggml_cont(ctx, ggml_concat(ctx,
                ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, N), k_pe, 0)); // [n_kv_row, 1, N]
        }

        ggml_tensor * cat = nullptr;
        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            ggml_tensor * kv = m.slots.at(B.slot_id)->kv_k[0][il];

            ggml_tensor * ki = ggml_cont(ctx, ggml_view_2d(ctx, Kcur, hp.n_kv_row, 1, Kcur->nb[1], (size_t) i * Kcur->nb[2]));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv, ki, B.kv_idx[dev]));

            ggml_tensor * K = ggml_view_3d(ctx, kv, hp.n_kv_row, B.n_kv, 1, kv->nb[1], kv->nb[1] * B.n_kv, 0);
            ggml_tensor * V = ggml_view_3d(ctx, kv, hp.kv_lora_rank, B.n_kv, 1, kv->nb[1], kv->nb[1] * B.n_kv, 0);
            ggml_tensor * Qi = ggml_cont(ctx, ggml_view_3d(ctx, Qcur, hp.n_kv_row, n_head, 1,
                                                           Qcur->nb[1], Qcur->nb[2], (size_t) i * Qcur->nb[2]));

            ggml_tensor * out = nullptr;                                        // [kv_lora, 1, n_head]
            if (m.flash_attn)
            {
                ggml_tensor * qf = ggml_permute(ctx, Qi, 0, 2, 1, 3);           // [n_kv_row, 1, n_head]
                ggml_tensor * fa = tsg_flash_attn_ext_guarded(ctx, m.backends[dev], "GLM-DSA batched decode", qf, K, V, masks[(size_t) i], hp.kq_scale(), 0.0f, 0.0f,
                    nullptr, GGML_PREC_F32);
                out = ggml_permute(ctx, fa, 0, 2, 1, 3);                        // [kv_lora, 1, n_head]
            }
            else
            {
                ggml_tensor * qp = ggml_permute(ctx, Qi, 0, 2, 1, 3);
                ggml_tensor * kq = decode_mul_mat(ctx, K, qp);                    // [n_kv, 1, n_head]
                decode_prec_set_acc(kq, GGML_PREC_F32);
                kq = ggml_soft_max_ext(ctx, kq, masks[(size_t) i], hp.kq_scale(), 0.0f);
                ggml_tensor * vp = ggml_cont(ctx, ggml_transpose(ctx, V));
                out = decode_mul_mat(ctx, vp, kq);                                // [kv_lora, 1, n_head]
                decode_prec_set_acc(out, GGML_PREC_F32);
            }
            out = ggml_cont(ctx, out);
            cat = cat ? ggml_concat(ctx, cat, out, 1) : out;                    // grow along the token axis
        }

        // One V decompression and one output projection for the whole batch.
        ggml_tensor * kqv = decode_mul_mat(ctx, LW.wv_b, cat);                    // [head_v, N, n_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                               // [head_v, n_head, N]
        ggml_tensor * flat = ggml_cont_2d(ctx, kqv, (int64_t) hp.n_embd_head_v * n_head, N);
        return decode_mul_mat(ctx, LW.wo, flat);
    }


    // =====================================================================
    // glm5next batched decode: one graph, N tokens, N sequences. The shared
    // parts (projections, hyper-connections, router, experts, LM head) run
    // once over the batch; the per-token parts (KDA recurrence against each
    // slot's own state, cache writes, pooled scoring, attention) are built per
    // token because each token sees a different slot's history.
    // =====================================================================

    /// KDA for a batch of single tokens: the six projections run over the whole
    /// batch, then each token runs its own conv/delta-net step against its
    /// slot's persistent state.
    ggml_tensor * build_kda_bd(int il, ggml_tensor * cur)
    {
        const glm_layer_weights & LW = m.layers[il].w[0];
        const int64_t hd = hp.kda_head_dim;
        const int64_t H = hp.kda_n_head;
        const int64_t d_inner = hd * H;
        const int64_t dc = hp.d_conv;
        const int64_t N = nt;

        ggml_tensor * qp = recurrent_projection(ctx, LW.kda_wq, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * kp = recurrent_projection(ctx, LW.kda_wk, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * vp = recurrent_projection(ctx, LW.kda_wv, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * qkv = ggml_concat(ctx, ggml_concat(ctx, qp, kp, 0), vp, 0);   // [3*d_inner, N]

        ggml_tensor * conv_w = ggml_concat(ctx,
                ggml_concat(ctx,
                    ggml_reshape_2d(ctx, LW.kda_conv_q, dc, d_inner),
                    ggml_reshape_2d(ctx, LW.kda_conv_k, dc, d_inner), 1),
                ggml_reshape_2d(ctx, LW.kda_conv_v, dc, d_inner), 1);               // [dc, 3*d_inner]

        ggml_tensor * g_pre = role_mul_mat(ctx, LW.kda_f_b,
            role_mul_mat(ctx, LW.kda_f_a, cur, tsg_projection_role::recurrent_decay), tsg_projection_role::recurrent_decay);
        g_pre = ggml_add(ctx, g_pre, LW.kda_dt_b);                                  // [d_inner, N]
        ggml_tensor * beta_all = ggml_sigmoid(ctx, role_mul_mat(ctx, LW.kda_beta, cur, tsg_projection_role::recurrent_update)); // [H, N]
        ggml_tensor * gate_all = role_mul_mat(ctx, LW.kda_g_b,
            role_mul_mat(ctx, LW.kda_g_a, cur, tsg_projection_role::recurrent_output_gate), tsg_projection_role::recurrent_output_gate); // [d_inner, N]

        ggml_tensor * cat = nullptr;
        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            glm_slot & sl = *m.slots.at(B.slot_id);
            ggml_tensor * conv_state = sl.kda_conv[0][(size_t) il];                  // [dc-1, 3*d_inner]
            ggml_tensor * ssm_state = sl.kda_ssm[0][(size_t) il];                    // [hd, hd, H]

            ggml_tensor * col = ggml_cont(ctx, ggml_view_2d(ctx, qkv, 3 * d_inner, 1,
                    qkv->nb[1], (size_t) i * qkv->nb[1]));
            ggml_tensor * col_t = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, col)),
                    1, 3 * d_inner, 1);
            ggml_tensor * conv_in = ggml_concat(ctx,
                    ggml_reshape_3d(ctx, conv_state, dc - 1, 3 * d_inner, 1), col_t, 0);  // [dc, 3*d_inner, 1]
            ggml_tensor * conv_out = ggml_silu(ctx, tsg::ssm_conv_any(ctx, conv_in, conv_w)); // [3*d_inner, 1]

            ggml_tensor * tail = ggml_view_3d(ctx, conv_in, dc - 1, 3 * d_inner, 1,
                    conv_in->nb[1], conv_in->nb[2], conv_in->nb[0]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, conv_state));

            const size_t rs_hd = ggml_row_size(conv_out->type, hd);
            ggml_tensor * qc = ggml_view_3d(ctx, conv_out, hd, H, 1, rs_hd, conv_out->nb[1], 0);
            ggml_tensor * kc = ggml_view_3d(ctx, conv_out, hd, H, 1, rs_hd, conv_out->nb[1],
                    ggml_row_size(conv_out->type, d_inner));
            ggml_tensor * vc = ggml_view_3d(ctx, conv_out, hd, H, 1, rs_hd, conv_out->nb[1],
                    ggml_row_size(conv_out->type, 2 * d_inner));

            qc = ggml_reshape_4d(ctx, ggml_l2_norm(ctx, ggml_cont(ctx, qc), 1e-6f), hd, H, 1, 1);
            kc = ggml_reshape_4d(ctx, ggml_l2_norm(ctx, ggml_cont(ctx, kc), 1e-6f), hd, H, 1, 1);
            vc = ggml_reshape_4d(ctx, ggml_cont(ctx, vc), hd, H, 1, 1);

            ggml_tensor * g = ggml_cont(ctx, ggml_view_2d(ctx, g_pre, d_inner, 1,
                    g_pre->nb[1], (size_t) i * g_pre->nb[1]));
            g = ggml_reshape_3d(ctx, g, hd, H, 1);
            g = ggml_mul(ctx, g, ggml_reshape_3d(ctx, LW.kda_a, 1, H, 1));
            g = ggml_sigmoid(ctx, ggml_scale(ctx, g, -1.0f));
            g = ggml_scale(ctx, g, hp.kda_gate_lb);
            ggml_tensor * g4 = ggml_reshape_4d(ctx, g, hd, H, 1, 1);

            ggml_tensor * b4 = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, beta_all, H, 1,
                    beta_all->nb[1], (size_t) i * beta_all->nb[1])), 1, H, 1, 1);

            ggml_tensor * s4 = ggml_reshape_4d(ctx, ssm_state, hd, hd, H, 1);
            ggml_tensor * gdn = ggml_gated_delta_net(ctx, qc, kc, vc, g4, b4, s4, 1);

            ggml_tensor * core = ggml_view_3d(ctx, gdn, hd, H, 1,
                    ggml_row_size(gdn->type, hd), ggml_row_size(gdn->type, hd * H), 0);
            ggml_tensor * new_state = ggml_view_3d(ctx, gdn, hd, hd, H,
                    ggml_row_size(gdn->type, hd), ggml_row_size(gdn->type, hd * hd),
                    ggml_row_size(gdn->type, hd * H));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

            ggml_tensor * gate = ggml_cont(ctx, ggml_view_2d(ctx, gate_all, d_inner, 1,
                    gate_all->nb[1], (size_t) i * gate_all->nb[1]));
            gate = ggml_reshape_3d(ctx, gate, hd, H, 1);
            ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, core), hp.rms_eps),
                                            LW.kda_o_norm);
            ggml_tensor * gated = ggml_mul(ctx, normed, ggml_sigmoid(ctx, gate));
            ggml_tensor * out_i = ggml_reshape_2d(ctx, ggml_cont(ctx, gated), d_inner, 1);

            cat = cat ? ggml_concat(ctx, cat, out_i, 1) : out_i;                    // [d_inner, N]
        }

        return recurrent_projection(ctx, LW.kda_wo, cat, tsg_projection_role::residual_mixing); // [n_embd, N]
    }

    /// Pooled indexer for a batch of single tokens: shared projections, then a
    /// per-token cache write and (when past the dense limit) per-token pooled
    /// scoring against that token's own slot.
    void build_indexer_g5n_bd(int il, ggml_tensor * cur, ggml_tensor * qr,
                              std::vector<ggml_tensor *> & top_k)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[0];
        const int dev = device_of(il, 0);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;
        const int64_t r = hp.indexer_kpool;
        const int64_t N = nt;

        ggml_tensor * ik = decode_mul_mat(ctx, LW.idx_attn_k, cur);                   // [D, N]
        ik = ggml_norm(ctx, ik, hp.norm_eps);
        ik = ggml_mul(ctx, ik, LW.idx_k_norm_w);
        ik = ggml_add(ctx, ik, LW.idx_k_norm_b);
        ggml_tensor * gate = decode_mul_mat(ctx, LW.idx_comp_gate, cur);              // [D, N]
        ggml_tensor * packed = ggml_concat(ctx,
                ggml_reshape_3d(ctx, ik, D, 1, N),
                ggml_reshape_3d(ctx, gate, D, 1, N), 1);                            // [D, 2, N]
        packed = ggml_cont(ctx, packed);

        ggml_tensor * iq = decode_mul_mat(ctx, LW.idx_attn_q_b, qr);                  // [D*H, N]
        iq = ggml_reshape_3d(ctx, iq, D, H, N);
        ggml_tensor * w = decode_mul_mat(ctx, LW.idx_proj, cur);                      // [H, N]
        decode_prec_set_acc(w, GGML_PREC_F32);
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        ggml_tensor * ape = ggml_cont(ctx, ggml_transpose(ctx, LW.idx_comp_ape));   // [r, D]

        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            glm_slot & sl = *m.slots.at(B.slot_id);
            ggml_tensor * cache = sl.idx_k[0][(size_t) il];                          // [2D, n_ctx]

            ggml_tensor * pk = ggml_cont(ctx, ggml_view_2d(ctx, packed, 2 * D, 1,
                    (size_t) 2 * D * packed->nb[0], (size_t) i * packed->nb[2]));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, pk, B.kv_idx[dev]));

            if (!B.sparse) { top_k[(size_t) i] = nullptr; continue; }

            const int64_t n_pools = B.n_kv / r;
            ggml_tensor * kg = ggml_view_2d(ctx, cache, 2 * D, B.n_kv, cache->nb[1], 0);
            ggml_tensor * members = ggml_get_rows(ctx, kg, B.pool_cells[dev]);       // [2D, n_kv] F32

            const size_t nb_mem = members->nb[1];
            ggml_tensor * mem_k = ggml_view_3d(ctx, members, D, r, n_pools, nb_mem, nb_mem * r, 0);
            ggml_tensor * mem_g = ggml_view_3d(ctx, members, D, r, n_pools, nb_mem, nb_mem * r,
                                               (size_t) D * members->nb[0]);
            ggml_tensor * keys_t = ggml_cont(ctx, ggml_permute(ctx, mem_k, 1, 0, 2, 3));
            ggml_tensor * gate_t = ggml_cont(ctx, ggml_permute(ctx, mem_g, 1, 0, 2, 3));
            gate_t = ggml_add(ctx, gate_t, ggml_reshape_3d(ctx, ape, r, D, 1));
            ggml_tensor * probs = ggml_soft_max(ctx, gate_t);
            ggml_tensor * pool_k = ggml_sum_rows(ctx, ggml_mul(ctx, keys_t, probs)); // [1, D, n_pools]
            pool_k = ggml_reshape_2d(ctx, ggml_cont(ctx, pool_k), D, n_pools);

            ggml_tensor * qi = ggml_cont(ctx, ggml_view_3d(ctx, iq, D, H, 1,
                    iq->nb[1], iq->nb[2], (size_t) i * iq->nb[2]));
            ggml_tensor * wi = ggml_cont(ctx, ggml_view_2d(ctx, w, H, 1,
                    w->nb[1], (size_t) i * w->nb[1]));

            ggml_tensor * score = nullptr;
            if (m.fused_lid)
            {
                ggml_tensor * pool_kf = ggml_reshape_3d(ctx, pool_k, D, 1, n_pools);
                score = ggml_lightning_indexer(ctx, qi, pool_kf, wi, B.pool_bias[dev]);   // [n_pools, 1]
            }
            else
            {
                ggml_tensor * qp2 = ggml_permute(ctx, qi, 0, 2, 1, 3);              // [D, 1, H]
                ggml_tensor * kq = decode_mul_mat(ctx, pool_k, qp2);                  // [n_pools, 1, H]
                kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));             // [H, 1, n_pools]
                ggml_tensor * sc = ggml_relu(ctx, kq);
                sc = ggml_mul(ctx, sc, wi);
                sc = ggml_sum_rows(ctx, sc);                                        // [1, 1, n_pools]
                sc = ggml_cont(ctx, ggml_permute(ctx, sc, 2, 1, 0, 3));             // [n_pools, 1, 1]
                score = ggml_add(ctx, sc, B.pool_bias[dev]);
            }

            const int64_t select_k = std::min<int64_t>(n_pools, hp.indexer_top_k / r);
            ggml_tensor * sel = ggml_cont(ctx, ggml_top_k(ctx, score, (int) select_k));
            ggml_tensor * pc2 = ggml_reshape_2d(ctx, B.pool_cells[dev], r, n_pools);
            ggml_tensor * sel_flat = ggml_reshape_1d(ctx, sel, select_k);
            ggml_tensor * cells = ggml_get_rows(ctx, pc2, sel_flat);                // I32 [r, select_k]
            top_k[(size_t) i] = ggml_reshape_2d(ctx, cells, r * select_k, 1);
        }
    }

    /// Single-token variant of build_topk_mask_g5n.
    ggml_tensor * build_topk_mask_g5n_1(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k,
                                        ggml_tensor * trail_cells, ggml_tensor * trail_vals, int64_t nkv)
    {
        const int64_t n_top_k = top_k->ne[0];
        const int64_t r = trail_cells->ne[0];

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);                     // [nkv, 1]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, nkv, 1, base->nb[0], base->nb[1], 0);

        ggml_tensor * t_idx = ggml_view_3d(ctx, trail_cells, r, 1, 1,
                                           trail_cells->nb[1], trail_cells->nb[1], 0);
        ggml_tensor * with_tail = ggml_set_rows(ctx, all, trail_vals, t_idx);

        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, 1, 1, top_k->nb[1], top_k->nb[1], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, 1), 0.0f);
        ggml_tensor * unmasked = ggml_set_rows(ctx, with_tail, zeros, idx);

        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, nkv, 1, unmasked->nb[2], 0);
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);
        pin(base, dev);
        pin(zeros, dev);
        pin(with_tail, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    /// Whole graph for one glm5next batched decode step.
    void build_batched_g5n()
    {
        graph_inputs & inp = res.inp;
        const int64_t N = nt;
        const int64_t hcm = hp.hc_mult;
        const int64_t r = hp.indexer_kpool;

        bool dev_used[MAX_GPUS + 1] = {};
        bool dev_mla[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++)
        {
            dev_used[m.layers[il].device] = true;
            if (!m.layers[il].recurrent) dev_mla[m.layers[il].device] = true;
        }

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input(GGML_TYPE_I32, N, 0, nb, d);
            if (!dev_mla[d]) continue;
            for (int64_t i = 0; i < N; i++)
            {
                bd_token & B = res.bd[(size_t) i];
                snprintf(nb, sizeof(nb), "bd%lld_kv_idx.%d", (long long) i, d);
                B.kv_idx[d] = new_input(GGML_TYPE_I64, 1, 0, nb, d);
                snprintf(nb, sizeof(nb), "bd%lld_kq_mask.%d", (long long) i, d);
                B.kq_mask[d] = new_input(m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, B.n_kv, 1, nb, d);
                if (B.sparse)
                {
                    snprintf(nb, sizeof(nb), "bd%lld_pool_cells.%d", (long long) i, d);
                    B.pool_cells[d] = new_input(GGML_TYPE_I32, B.n_kv, 0, nb, d);
                    snprintf(nb, sizeof(nb), "bd%lld_pool_bias.%d", (long long) i, d);
                    B.pool_bias[d] = new_input(m.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                               B.n_kv / r, 1, nb, d);
                    snprintf(nb, sizeof(nb), "bd%lld_trail_cells.%d", (long long) i, d);
                    B.trail_cells[d] = new_input(GGML_TYPE_I32, r, 1, nb, d);
                    snprintf(nb, sizeof(nb), "bd%lld_trail_vals.%d", (long long) i, d);
                    B.trail_vals[d] = new_input_3d(GGML_TYPE_F32, 1, r, 1, nb, d);
                }
            }
        }

        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);
        ggml_tensor * inpL = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, hp.n_embd, 1, N),
                                            hp.n_embd, hcm, N, 1);

        std::vector<ggml_tensor *> top_k((size_t) N, nullptr);
        std::vector<ggml_tensor *> masks((size_t) N, nullptr);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];
            const glm_layer_weights & LW = L.w[0];
            const int dev = L.device;

            if (il > 0 && dev != m.layers[il - 1].device)
                pin(inpL, dev);

            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;
            ggml_tensor * cur = build_hc_pre(inpL, LW.hc_attn_fn, LW.hc_attn_scale, LW.hc_attn_base,
                                             &post, &comb);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);
            cur = rms(cur, LW.attn_norm);

            ggml_tensor * attn = nullptr;
            if (L.recurrent)
            {
                attn = build_kda_bd(il, cur);
            }
            else
            {
                ggml_tensor * qr = rms(decode_mul_mat(ctx, LW.wq_a, cur), LW.q_a_norm);
                build_indexer_g5n_bd(il, cur, qr, top_k);
                for (int64_t i = 0; i < N; i++)
                {
                    bd_token & B = res.bd[(size_t) i];
                    masks[(size_t) i] = (B.sparse && top_k[(size_t) i])
                        ? build_topk_mask_g5n_1(dev, B.kq_mask[dev], top_k[(size_t) i],
                                                B.trail_cells[dev], B.trail_vals[dev], B.n_kv)
                        : B.kq_mask[dev];
                }
                attn = build_attention_bd(il, cur, qr, nullptr, masks);
            }

            inpL = build_hc_post(attn, residual, post, comb);

            residual = inpL;
            cur = build_hc_pre(inpL, LW.hc_ffn_fn, LW.hc_ffn_scale, LW.hc_ffn_base, &post, &comb);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);
            cur = rms(cur, LW.ffn_norm);

            ggml_tensor * ffn = nullptr;
            if (!L.is_moe)
            {
                ffn = build_dense_ffn(il, 0, cur);
            }
            else
            {
                ffn = build_moe(il, 0, cur);
                ggml_tensor * sh = build_shexp(il, 0, cur);
                if (sh) ffn = ggml_add(ctx, ffn, sh);
            }
            inpL = build_hc_post(ffn, residual, post, comb);
        }

        // every token's logits
        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hcm * hp.n_embd, N);
        ggml_tensor * x3 = ggml_reshape_3d(ctx, flat, hp.n_embd, hcm, N);
        ggml_tensor * cur = hc_mean(x3);
        cur = rms(cur, m.output_norm);
        cur = decode_mul_mat(ctx, m.output, cur);                                      // [vocab, N]
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
        bd_log("g5n built, %d nodes\n", ggml_graph_n_nodes(gf));
    }

    /// Whole graph for one batched decode step.
    void build_batched()
    {

        graph_inputs & inp = res.inp;
        const int64_t N = nt;

        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input(GGML_TYPE_I32, N, 0, nb, d);
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input(GGML_TYPE_I32, N, 0, nb, d);

            for (int64_t i = 0; i < N; i++)
            {
                bd_token & B = res.bd[(size_t) i];
                snprintf(nb, sizeof(nb), "bd%lld_kv_idx.%d", (long long) i, d);
                B.kv_idx[d] = new_input(GGML_TYPE_I64, 1, 0, nb, d);
                snprintf(nb, sizeof(nb), "bd%lld_kq_mask.%d", (long long) i, d);
                B.kq_mask[d] = new_input(m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, B.n_kv, 1, nb, d);
                if (B.sparse)
                {
                    snprintf(nb, sizeof(nb), "bd%lld_lid_mask.%d", (long long) i, d);
                    B.lid_mask[d] = new_input(GGML_TYPE_F16, B.n_kv, 1, nb, d);
                }
            }
        }

        ggml_tensor * inpL = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);

        std::vector<ggml_tensor *> top_k((size_t) N, nullptr);
        std::vector<ggml_tensor *> masks((size_t) N, nullptr);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];
            const int dev = L.device;

            if (il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            ggml_tensor * residual = inpL;
            ggml_tensor * cur = rms(inpL, L.w[0].attn_norm);
            ggml_tensor * qr = rms(decode_mul_mat(ctx, L.w[0].wq_a, cur), L.w[0].q_a_norm);

            if (L.indexer_full)
            {
                build_indexer_bd(il, cur, qr, inp.pos[dev], true, top_k);
                for (int64_t i = 0; i < N; i++)
                {
                    bd_token & B = res.bd[(size_t) i];
                    masks[(size_t) i] = (B.sparse && top_k[(size_t) i])
                        ? build_topk_mask_1(dev, B.kq_mask[dev], top_k[(size_t) i], B.n_kv)
                        : B.kq_mask[dev];
                }
            }
            else
            {
                // A layer without its own indexer has no key cache either — it
                // attends through the previous full layer's selection.
                for (int64_t i = 0; i < N; i++)
                    if (!masks[(size_t) i]) masks[(size_t) i] = res.bd[(size_t) i].kq_mask[dev];
            }

            trace("attn_norm", il, 0, cur);
            trace("qr", il, 0, qr);
            inpL = ggml_add(ctx, residual, build_attention_bd(il, cur, qr, inp.pos[dev], masks));
            trace("ffn_inp", il, 0, inpL);

            residual = inpL;
            cur = rms(inpL, L.w[0].ffn_norm);
            if (!L.is_moe)
            {
                inpL = ggml_add(ctx, residual, build_dense_ffn(il, 0, cur));
                trace("l_out", il, 0, inpL);
                continue;
            }
            ggml_tensor * ffn = build_moe(il, 0, cur);
            ggml_tensor * shexp = build_shexp(il, 0, cur);
            if (shexp) ffn = ggml_add(ctx, ffn, shexp);
            inpL = ggml_add(ctx, residual, ffn);
            trace("l_out", il, 0, inpL);
        }

        ggml_tensor * cur = rms(inpL, m.output_norm);
        cur = decode_mul_mat(ctx, m.output, cur);                                 // [vocab, N]
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
        bd_log("built, %d nodes\n", ggml_graph_n_nodes(gf));
    }

    /// Layers TS_GLM_TRACE selects (comma-separated, or "all").
    static bool trace_layer(int il)
    {
        static const std::string spec = []() { const char * e = getenv("TS_GLM_TRACE"); return e ? std::string(e) : std::string(); }();
        if (spec.empty()) return false;
        if (spec == "all") return true;
        const std::string needle = std::to_string(il);
        size_t pos = 0;
        while ((pos = spec.find(needle, pos)) != std::string::npos)
        {
            const bool lb = pos == 0 || spec[pos - 1] == ',';
            const size_t end = pos + needle.size();
            const bool rb = end == spec.size() || spec[end] == ',';
            if (lb && rb) return true;
            pos = end;
        }
        return false;
    }

    int trace_il = 0;
    int trace_rank = 0;

    void trace(const char * tag, int il, int rank, ggml_tensor * t)
    {
        if (!t || !trace_layer(il)) return;
        char name[128];
        snprintf(name, sizeof(name), "%s-%d.r%d", tag, il, rank);
        ggml_set_output(t);
        res.traces.push_back({ name, t, tag, il, rank });
        ggml_build_forward_expand(gf, t);
    }

    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w)
    {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    // A pin is honoured by ggml_backend_sched unconditionally, so it is a
    // promise that the target backend can run the node. Leaf inputs have
    // nothing to execute and are always safe to pin.
    void pin(ggml_tensor * t, int dev)
    {
        // A segmented TP graph is allocated directly on its one rank backend;
        // it has no multi-backend scheduler to pin through.
        if (res.tp_fused) return;
        if (dev < 0 || dev >= m.n_gpu) return;
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(m.sched_backends[dev], t)) return;
        ggml_backend_sched_set_tensor_backend(res.sched, t, res.allocation_budget.scheduler_backend(m.sched_backends[dev]));
    }

    ggml_tensor * new_input_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * new_input(ggml_type type, int64_t ne0, int64_t ne1, const char * name, int dev)
    {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1)
                                  : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * pos)
    {
        // glm-dsa is LLAMA_ROPE_TYPE_NORM and carries no YaRN scaling.
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, 0,
                             hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    // ---- DSA lightning indexer -------------------------------------------
    // score[j, t] = sum_h relu(q[t,h] . k[j]) * w[t,h], masked causally; the
    // top-k of that is what the real attention is allowed to see.
    //
    // The keys are cached on EVERY full-indexer layer, including while the
    // context is still shorter than top_k and no selection is needed: a later
    // ubatch scores these tokens, and a chunk that skipped its own keys would
    // score whatever zeros the cache was cleared to. Returns null when only the
    // keys were wanted.
    ggml_tensor * build_indexer(int il, int rank, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                ggml_tensor * lid_mask, ggml_tensor * kv_idxs, bool score_now)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int dev = device_of(il, rank);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;

        ggml_tensor * k = decode_mul_mat(ctx, LW.idx_attn_k, cur);           // [D, nt]
        k = ggml_norm(ctx, k, 0.0f);                                       // f_norm_eps is 0 for glm-dsa
        k = ggml_mul(ctx, k, LW.idx_k_norm_w);
        k = ggml_add(ctx, k, LW.idx_k_norm_b);
        k = ggml_reshape_3d(ctx, k, D, 1, nt);
        k = rope(k, pos);
        if (m.hadamard[dev]) k = decode_mul_mat(ctx, m.hadamard[dev], k);

        // Append this ubatch's indexer keys. The destination rows arrive as a
        // graph INPUT rather than a baked view offset, so one cached graph
        // serves every position (which is the whole point of the cache).
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, slot.idx_k[rank][il], ggml_reshape_2d(ctx, k, D, nt), kv_idxs));

        if (!score_now)
            return nullptr;

        ggml_tensor * q = decode_mul_mat(ctx, LW.idx_attn_q_b, qr);          // [D*H, nt]
        q = ggml_reshape_3d(ctx, q, D, H, nt);
        q = rope(q, pos);
        if (m.hadamard[dev]) q = decode_mul_mat(ctx, m.hadamard[dev], q);

        ggml_tensor * k_all = ggml_view_3d(ctx, slot.idx_k[rank][il], D, 1, n_kv,
                                           slot.idx_k[rank][il]->nb[1], slot.idx_k[rank][il]->nb[1], 0);

        ggml_tensor * w = decode_mul_mat(ctx, LW.idx_proj, cur);             // [H, nt]
        // Pre-scaled so the scale never touches the big [n_kv, nt] score tensor.
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        ggml_tensor * score = nullptr;
        if (m.fused_lid)
        {
            score = ggml_lightning_indexer(ctx, q, k_all, w, lid_mask);    // [n_kv, nt]
        }
        else
        {
            ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);           // [D, nt, H]
            ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);       // [D, n_kv, 1]
            ggml_tensor * kq = decode_mul_mat(ctx, kp, qp);                  // [n_kv, nt, H]
            // F32 accumulation: the cache is F16, and without this ggml would
            // convert the query to F16 and dot in half precision. The fused
            // kernel converts the KEY up to F32 instead, and matching it is what
            // keeps the top-k selection identical at long context.
            decode_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));        // [H, nt, n_kv]
            ggml_tensor * s = ggml_relu(ctx, kq);
            s = ggml_mul(ctx, s, w);                                       // broadcast over n_kv
            s = ggml_sum_rows(ctx, s);                                     // [1, nt, n_kv]
            s = ggml_cont(ctx, ggml_permute(ctx, s, 2, 1, 0, 3));          // [n_kv, nt, 1]
            score = ggml_add(ctx, s, ggml_cast(ctx, lid_mask, GGML_TYPE_F32));
        }
        const int n_top_k = (int) std::min<int64_t>(n_kv, hp.indexer_top_k);
        return ggml_cont(ctx, ggml_top_k(ctx, score, n_top_k));            // I32 [n_top_k, nt]
    }

    /// Causal mask with the DSA selection folded in: start fully masked, unmask
    /// the cells the indexer picked, then add the plain causal mask back so a
    /// picked-but-future cell stays masked.
    ///
    /// <para>Built once per FULL indexer layer and reused by the "shared" layers
    /// that follow it — they attend through the same selection, so rebuilding an
    /// identical [n_kv, n_tokens] mask three more times is pure cost.</para>
    ///
    /// <para>The -inf canvas is marked as a graph output on purpose. ggml_set_rows
    /// writes THROUGH a view into that tensor's buffer, and left as an ordinary
    /// intermediate the graph allocator is free to hand the same bytes to another
    /// node while the view is still live — which showed up as an all-masked row
    /// and a NaN softmax as soon as several tensor-parallel ranks put enough
    /// nodes between the write and the read. Marking it an output takes it out of
    /// the reuse pool; hoisting keeps the number of such buffers at one per full
    /// indexer layer.</para>
    ggml_tensor * build_topk_mask(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k)
    {
        const int64_t n_top_k = top_k->ne[0];
        trace("topk", trace_il, trace_rank, top_k);

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);               // [n_kv, nt]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, n_kv, nt, base->nb[0], base->nb[1], 0);   // [1, n_kv, nt]

        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, nt, 1,
                                         top_k->nb[1], top_k->nb[2], 0);        // [n_top_k, nt, 1]

        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, nt), 0.0f);

        ggml_tensor * unmasked = ggml_set_rows(ctx, all, zeros, idx);           // [1, n_kv, nt]
        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, n_kv, nt, unmasked->nb[2], 0);   // [n_kv, nt]
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);

        // ggml_set_rows writes THROUGH a view into `base`, so the write and the
        // buffer have to belong to the same backend. Left to the scheduler's own
        // heuristics the fill and the write can land on different devices, and
        // then the store goes to a buffer the writing backend does not own —
        // which corrupts whatever else lives there. Pinning the whole chain to
        // the device that consumes the mask keeps the aliasing local.
        pin(base, dev);
        pin(zeros, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    // ---- MLA attention ----------------------------------------------------
    ggml_tensor * build_attention(int il, int rank, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                  ggml_tensor * mask, ggml_tensor * kv_idxs)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        // Under tensor parallelism this rank owns a contiguous run of heads; the
        // key/value cache row is head-independent, so it is computed (identically)
        // and stored by every rank instead of being shared through a collective.
        const int64_t n_head = (m.tp > 1 && m.tp_heads()) ? (L.head_first[rank + 1] - L.head_first[rank])
                                                          : hp.n_head;

        ggml_tensor * q = decode_mul_mat(ctx, LW.wq_b, qr);                       // [n_head*head_k, nt]
        ggml_tensor * kv_pe = decode_mul_mat(ctx, LW.wkv_a_mqa, cur);              // [kv_lora + rope, nt]

        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        if (hp.n_rot == 0)
        {
            // glm5next MLA is nope-only: no rope half anywhere, so the latent IS
            // the whole cache row and the absorbed query needs no concat - but
            // it does need the cont deepseek-style graphs get from theirs.
            ggml_tensor * kv_cmpr = rms(kv_pe, LW.kv_a_norm);
            ggml_tensor * q3 = ggml_reshape_3d(ctx, q, hp.n_embd_head_k, n_head, nt);
            ggml_tensor * q_nope_p = ggml_permute(ctx, q3, 0, 2, 1, 3);          // [head_k, nt, n_head]
            ggml_tensor * q_abs = decode_mul_mat(ctx, LW.wk_b, q_nope_p);          // [kv_lora, nt, n_head]
            Qcur = ggml_cont(ctx, ggml_permute(ctx, q_abs, 0, 2, 1, 3));         // [kv_lora, n_head, nt]
            Kcur = ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, nt);
        }
        else
        {
        ggml_tensor * q_nope = ggml_view_3d(ctx, q, hp.n_nope, n_head, nt,
                                            ggml_row_size(q->type, hp.n_embd_head_k),
                                            ggml_row_size(q->type, hp.n_embd_head_k) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx, q, hp.n_rot, n_head, nt,
                                          ggml_row_size(q->type, hp.n_embd_head_k),
                                          ggml_row_size(q->type, hp.n_embd_head_k) * n_head,
                                          ggml_row_size(q->type, hp.n_nope));

        const size_t row = ggml_row_size(kv_pe->type, hp.n_kv_row);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx, kv_pe, hp.kv_lora_rank, nt, row, 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx, kv_pe, hp.n_rot, 1, nt, row, row,
                                          ggml_row_size(kv_pe->type, hp.kv_lora_rank));

        q_pe = rope(q_pe, pos);
        k_pe = rope(k_pe, pos);
        kv_cmpr = rms(kv_cmpr, LW.kv_a_norm);

        // q_absorbed[h] = wk_b[h]^T q_nope[h] : the per-head K decompression,
        // folded into the query so 64 heads share one 576-wide key head.
        ggml_tensor * q_nope_p = ggml_permute(ctx, q_nope, 0, 2, 1, 3);         // [nope, nt, n_head]
        ggml_tensor * q_abs = decode_mul_mat(ctx, LW.wk_b, q_nope_p);              // [kv_lora, nt, n_head]
        q_abs = ggml_permute(ctx, q_abs, 0, 2, 1, 3);                           // [kv_lora, n_head, nt]

        // rope goes last so an in-place context shift stays possible
        Qcur = ggml_concat(ctx, q_abs, q_pe, 0);                                // [n_kv_row, n_head, nt]

        Kcur = ggml_concat(ctx, ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, nt),
                           k_pe, 0);                                            // [n_kv_row, 1, nt]
        }

        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, slot.kv_k[rank][il], ggml_reshape_2d(ctx, Kcur, hp.n_kv_row, nt), kv_idxs));

        ggml_tensor * kv = slot.kv_k[rank][il];
        ggml_tensor * K = ggml_view_3d(ctx, kv, hp.n_kv_row, n_kv, 1, kv->nb[1], kv->nb[1] * n_kv, 0);
        // V is the compressed half of the same rows; wv_b decompresses the result.
        ggml_tensor * V = ggml_view_3d(ctx, kv, hp.kv_lora_rank, n_kv, 1, kv->nb[1], kv->nb[1] * n_kv, 0);

        ggml_tensor * cur_attn = nullptr;
        if (decode_rows())
        {
            // Every row sees the same padded key extent as an ordinary decode
            // at its own position. The extents participate in the graph key.
            for (int64_t r = 0; r < nt; ++r)
            {
                const int64_t nk = std::min<int64_t>(pad_to(p0 + r + 1, 256), m.n_ctx);
                ggml_tensor * kr = ggml_view_3d(ctx, kv, hp.n_kv_row, nk, 1, kv->nb[1], kv->nb[1] * nk, 0);
                ggml_tensor * vr = ggml_view_3d(ctx, kv, hp.kv_lora_rank, nk, 1, kv->nb[1], kv->nb[1] * nk, 0);
                ggml_tensor * qr = ggml_cont(ctx, ggml_view_3d(ctx, Qcur, hp.n_kv_row, n_head, 1,
                    Qcur->nb[1], Qcur->nb[2], (size_t) r * Qcur->nb[2]));
                ggml_tensor * mr = ggml_cont(ctx, ggml_view_2d(ctx, mask, nk, 1, mask->nb[1], (size_t) r * mask->nb[1]));
                ggml_tensor * out;
                if (m.flash_attn)
                {
                    ggml_tensor * qf = ggml_permute(ctx, qr, 0, 2, 1, 3);
                    out = tsg_flash_attn_ext_guarded(ctx, m.backends[device_of(il, rank)], "GLM decode-row verify",
                        qf, kr, vr, mr, hp.kq_scale(), 0.0f, 0.0f, nullptr, GGML_PREC_F32);
                    out = ggml_permute(ctx, out, 0, 2, 1, 3);
                }
                else
                {
                    ggml_tensor * qp = ggml_permute(ctx, qr, 0, 2, 1, 3);
                    ggml_tensor * kq = ggml_mul_mat(ctx, kr, qp);
                    ggml_prec_set_acc(kq, GGML_PREC_F32);
                    kq = ggml_soft_max_ext(ctx, kq, mr, hp.kq_scale(), 0.0f);
                    ggml_tensor * vp = ggml_cont(ctx, ggml_transpose(ctx, vr));
                    out = ggml_mul_mat(ctx, vp, kq);
                    ggml_prec_set_acc(out, GGML_PREC_F32);
                }
                out = ggml_mul_mat(ctx, LW.wv_b, out);
                out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
                out = ggml_reshape_2d(ctx, out, (int64_t) hp.n_embd_head_v * n_head, 1);
                cur_attn = cur_attn ? ggml_concat(ctx, cur_attn, out, 1) : out;
            }
        }
        else if (m.flash_attn)
        {
            ggml_tensor * qf = ggml_permute(ctx, Qcur, 0, 2, 1, 3);             // [n_kv_row, nt, n_head]
            ggml_tensor * fa = tsg_flash_attn_ext_guarded(ctx, m.backends[device_of(il, rank)], "GLM-DSA forward", qf, K, V, mask, hp.kq_scale(), 0.0f, 0.0f,
                nullptr, GGML_PREC_F32);
            // [kv_lora, n_head, nt] -> [kv_lora, nt, n_head] so wv_b's per-head
            // matmul runs as a matrix-matrix product with nt in dimension 1.
            fa = ggml_permute(ctx, fa, 0, 2, 1, 3);
            fa = decode_mul_mat(ctx, LW.wv_b, fa);                                 // [head_v, nt, n_head]
            fa = ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3));             // [head_v, n_head, nt]
            cur_attn = ggml_reshape_2d(ctx, fa, (int64_t) hp.n_embd_head_v * n_head, nt);
        }
        else
        {
            // K and V already come out of the cache in the layout mul_mat wants
            // ([n_embd, n_kv, n_head_kv]); only the query needs permuting.
            ggml_tensor * qp = ggml_permute(ctx, Qcur, 0, 2, 1, 3);             // [n_kv_row, nt, n_head]
            ggml_tensor * kq = decode_mul_mat(ctx, K, qp);                        // [n_kv, nt, n_head]
            decode_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, mask, hp.kq_scale(), 0.0f);

            ggml_tensor * vp = ggml_cont(ctx, ggml_transpose(ctx, V));          // [n_kv, kv_lora, 1]
            ggml_tensor * kqv = decode_mul_mat(ctx, vp, kq);                      // [kv_lora, nt, n_head]
            // V is F16; without this the F32 softmax weights would be rounded to
            // half before the context product.
            decode_prec_set_acc(kqv, GGML_PREC_F32);
            kqv = decode_mul_mat(ctx, LW.wv_b, kqv);                               // [head_v, nt, n_head]
            kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                           // [head_v, n_head, nt]
            cur_attn = ggml_cont_2d(ctx, kqv, (int64_t) hp.n_embd_head_v * n_head, nt);
        }

        return decode_mul_mat(ctx, LW.wo, cur_attn);
    }

    // ---- FFN --------------------------------------------------------------

    /// silu(gate) * up, with glm5next's clamp: up into [-L, L], gate into
    /// (-inf, L], BEFORE the activation - the reference routes the dense
    /// layers, the shared expert and the routed experts all through it.
    ggml_tensor * swiglu(ggml_tensor * gate, ggml_tensor * up)
    {
        if (hp.swiglu_clamp > 0.0f)
        {
            up = ggml_clamp(ctx, up, -hp.swiglu_clamp, hp.swiglu_clamp);
            gate = ggml_clamp(ctx, gate, -INFINITY, hp.swiglu_clamp);
        }
        return ggml_mul(ctx, ggml_silu(ctx, gate), up);
    }

    ggml_tensor * build_dense_ffn(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer_weights & LW = m.layers[il].w[rank];
        ggml_tensor * gate = decode_mul_mat(ctx, LW.ffn_gate, cur);
        ggml_tensor * up = decode_mul_mat(ctx, LW.ffn_up, cur);
        ggml_tensor * h = swiglu(gate, up);
        return decode_mul_mat(ctx, LW.ffn_down, h);
    }

    ggml_tensor * build_shexp(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer_weights & LW = m.layers[il].w[rank];
        if (!LW.ffn_gate_shexp) return nullptr;
        ggml_tensor * gate = decode_mul_mat(ctx, LW.ffn_gate_shexp, cur);
        ggml_tensor * up = decode_mul_mat(ctx, LW.ffn_up_shexp, cur);
        ggml_tensor * h = swiglu(gate, up);
        return decode_mul_mat(ctx, LW.ffn_down_shexp, h);
    }

    /// Routed-expert FFN for one rank. Returns this rank's PARTIAL sum; the
    /// caller reduces across ranks and adds the (replicated) shared expert.
    ggml_tensor * build_moe(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int64_t n_expert = hp.n_expert;
        const int64_t n_used = hp.n_expert_used;

        // Routing is replicated: every rank sees the same logits and picks the
        // same global top-k, so no collective is needed to agree on it.
        // A hard top-k cut can amplify input narrowing or TF32 rounding. The
        // shared role policy preserves the qualified F32 activation paths for
        // GLM5Next; other GLM architectures and unsupported weight formats keep
        // their existing backend arithmetic and accumulator tag.
        const auto router_type = LW.ffn_gate_inp->type;
        const bool sensitive_router = sensitive_projection_executor() && (router_type == GGML_TYPE_F32
            || router_type == GGML_TYPE_F16 || router_type == GGML_TYPE_BF16 || router_type == GGML_TYPE_Q8_0);
        ggml_tensor * logits = sensitive_router
            ? role_mul_mat(ctx, LW.ffn_gate_inp, cur, tsg_projection_role::routing_logits)
            : decode_mul_mat(ctx, LW.ffn_gate_inp, cur);    // [n_expert, nt]
        if (!sensitive_router) decode_prec_set_acc(logits, GGML_PREC_F32);

        ggml_tensor * probs = hp.expert_gating_func == 2
            ? ggml_sigmoid(ctx, logits)
            : ggml_soft_max(ctx, logits);

        // The routing bias steers SELECTION only; the weights come from the
        // unbiased probabilities.
        ggml_tensor * selection = LW.exp_probs_b ? ggml_add(ctx, probs, LW.exp_probs_b) : probs;
        ggml_tensor * selected = ggml_argsort_top_k(ctx, selection, (int) n_used);   // I32 [n_used, nt]

        // The router runs identically on every rank — same logits, same global
        // top-k, same weights — because the split is inside each expert, not
        // across them. Backends fuse this whole chain (sigmoid, bias, argsort,
        // gather, normalise) into one kernel, so nothing here may reach in for
        // an intermediate: a fused kernel never materialises them.
        ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, nt);
        ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);      // [1, n_used, nt]

        if (hp.expert_weights_norm)
        {
            weights = ggml_reshape_2d(ctx, weights, n_used, nt);
            ggml_tensor * sum = ggml_sum_rows(ctx, weights);
            sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, nt);
        }
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.expert_weights_scale);

        // Selected seams return before the ordinary expert graph below. Keep
        // their routing tensors visible to the same optional trace diagnostic.
        trace("router_logits", il, rank, logits);
        trace("moe_sel", il, rank, selected);
        trace("moe_probs", il, rank, probs);
        trace("moe_w", il, rank, weights);
        ggml_build_forward_expand(gf, weights);

        if (m.selected_expert_seams && L.cpu_moe && rank == 0 && res.kind == 0) {
            // The scheduler callback must observe materialized routing, never a
            // fused router's unmaterialized intermediate. Retain all producers,
            // then create a distinct terminal boundary node before the external
            // input leaf is consumed. Its CUDA assignment is verified below.
            ggml_set_output(cur); ggml_set_output(selected); ggml_set_output(weights);
            // argsort_top_k is a view into a full expert-count row. Its next
            // token row starts at num_experts*sizeof(I32), not K*sizeof(I32).
            // Pack only the selected K IDs on-device before the callback reads
            // K*N contiguous entries; otherwise batched rows read unrelated
            // unselected IDs from the first row without violating range checks.
            auto* packed_ids = ggml_cont(ctx, selected);
            ggml_set_output(packed_ids);
            ggml_build_forward_expand(gf, packed_ids);
            auto* packed_input = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx, cur);
            ggml_set_output(packed_input);
            ggml_build_forward_expand(gf, packed_input);
            auto* boundary = ggml_cont(ctx, weights);
            ggml_set_output(boundary);
            ggml_build_forward_expand(gf, boundary);
            auto* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, nt);
            ggml_set_input(output); ggml_set_output(output);
            auto segment = selected_expert_segment(m, il, static_cast<int>(nt), !res.bd.empty());
            segment.moe_in = packed_input; segment.sel_ids = packed_ids;
            segment.weights = boundary; segment.moe_out = output;
            res.selected_expert_boundaries.emplace(boundary, res.selected_expert_seams.size());
            res.selected_expert_seams.push_back({segment, boundary, m.backends[0]});
            return output;
        }

        ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, nt);

        ggml_tensor * up = decode_mul_mat_id(ctx, LW.ffn_up_exps, cur3, selected);    // [n_ff_exp, n_used, nt]
        ggml_tensor * gate = decode_mul_mat_id(ctx, LW.ffn_gate_exps, cur3, selected);
        ggml_tensor * h = swiglu(gate, up);
        ggml_tensor * experts = decode_mul_mat_id(ctx, LW.ffn_down_exps, h, selected); // [n_embd, n_used, nt]
        if (hp.g5n && nt <= TSG_PRECISION_DECODE_COLUMNS)
            ggml_set_output(experts); // keep the weighted-reduction fusion's input from aliasing its output

        trace("moe_up", il, rank, up);
        trace("moe_gate", il, rank, gate);
        trace("moe_down", il, rank, experts);

        experts = ggml_mul(ctx, experts, weights);

        ggml_tensor * moe_out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, hp.n_embd, nt, experts->nb[2], e * experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        return moe_out;
    }


    // =====================================================================
    // glm5next (GLM-5.3-Flash): KDA linear attention, pooled DSA indexing and
    // Sinkhorn hyper-connections around every residual crossing. The MLA
    // attention, the MoE and the graph plumbing are shared with glm-dsa above.
    // =====================================================================

    // ---- hyper-connections ----
    // The residual stream is [n_embd, hc, nt]. Each crossing computes, from an
    // RMS-normed flattening of the stream, hc pre-weights (which stream mix the
    // sublayer consumes), hc post-weights (how the sublayer output is written
    // back per stream) and an [hc, hc] Sinkhorn-normalized mixing matrix.
    std::map<ggml_tensor *, ggml_tensor *> hc_t_cache;

    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base)
    {
        return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
    }

    ggml_tensor * view_row_1d(ggml_tensor * t, int64_t ne0, int64_t i0)
    {
        return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
    }

    ggml_tensor * view_row_2d(ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0)
    {
        return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
    }

    /// The residual stream with the stream axis moved into dim 0, for the
    /// decomposed mul_mat forms. Memoized: hc_post takes the same `residual`
    /// its hc_pre took as `x`, so a layer pays for the transpose once.
    ggml_tensor * hc_streams_first(ggml_tensor * x)
    {
        auto it = hc_t_cache.find(x);
        if (it != hc_t_cache.end()) return it->second;
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [hc, n_embd, nt]
        hc_t_cache[x] = t;
        return t;
    }

    ggml_tensor * build_hc_pre_op(ggml_tensor * x, ggml_tensor * w)
    {
        if (m.hc_native)
            return ggml_dsv4_hc_pre(ctx, x, w);
        const int64_t hcm = x->ne[1];
        const int64_t n = x->ne[2];
        ggml_tensor * xt = hc_streams_first(x);                       // [hc, n_embd, n]
        ggml_tensor * w3 = ggml_reshape_3d(ctx, w, hcm, 1, n);        // [hc, 1, n]
        ggml_tensor * out = decode_mul_mat(ctx, xt, w3);                // [n_embd, 1, n]
        return ggml_reshape_2d(ctx, out, x->ne[0], n);
    }

    ggml_tensor * build_hc_pre(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
                               ggml_tensor ** post, ggml_tensor ** comb)
    {
        const int64_t hcm = hp.hc_mult;
        const int64_t hc_dim = hcm * hp.n_embd;
        const int64_t n = x->ne[2];

        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, n);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, hp.rms_eps);
        ggml_tensor * mixes = role_mul_mat(ctx, fn, flat_norm, tsg_projection_role::residual_mixing); // [(2+hc)*hc, n]

        ggml_tensor * scale_pre = view_row_1d(hc_scale, 1, 0);
        ggml_tensor * scale_post = view_row_1d(hc_scale, 1, 1);
        ggml_tensor * base_pre = view_row_1d(hc_base, hcm, 0);
        ggml_tensor * base_post = view_row_1d(hc_base, hcm, hcm);

        ggml_tensor * pre = view_row_2d(mixes, hcm, n, 0);
        pre = hc_affine(pre, scale_pre, base_pre);
        pre = ggml_sigmoid(ctx, pre);
        pre = ggml_scale_bias(ctx, pre, 1.0f, hp.hc_eps);

        *post = view_row_2d(mixes, hcm, n, hcm);
        *post = hc_affine(*post, scale_post, base_post);
        *post = ggml_sigmoid(ctx, *post);
        *post = ggml_scale(ctx, *post, 2.0f);

        *comb = ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.hc_eps, hp.hc_sinkhorn);

        return build_hc_pre_op(x, pre);
    }

    ggml_tensor * build_hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb)
    {
        if (m.hc_native)
            return ggml_dsv4_hc_post(ctx, x, residual, post, comb);

        const int64_t n_embd = x->ne[0];
        const int64_t n = x->ne[1];
        const int64_t hcm = residual->ne[1];

        // rank-1 term: an outer product per token, expressed as a mul_mat whose
        // contracted dimension is 1.
        ggml_tensor * x1 = ggml_reshape_3d(ctx, x, 1, n_embd, n);
        ggml_tensor * p1 = ggml_reshape_3d(ctx, post, 1, hcm, n);
        ggml_tensor * outer = decode_mul_mat(ctx, x1, p1);              // [n_embd, hc, n]

        ggml_tensor * rt = hc_streams_first(residual);                // [hc(src), n_embd, n]
        ggml_tensor * ct = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));   // [hc(src), hc(dst), n]
        ggml_tensor * mixed = decode_mul_mat(ctx, rt, ct);              // [n_embd, hc(dst), n]

        return ggml_add(ctx, outer, mixed);
    }

    /// Mean over the hyper-connection streams: [n_embd, hc, n] -> [n_embd, n].
    /// glm5next's head is this unweighted mean, not DSV4's learned gated one.
    ggml_tensor * hc_mean(ggml_tensor * x)
    {
        const int64_t hcm = x->ne[1];
        const int64_t n = x->ne[2];
        ggml_tensor * acc = nullptr;
        for (int64_t c = 0; c < hcm; c++)
        {
            ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, x, hp.n_embd, n, x->nb[2], (size_t) c * x->nb[1]));
            acc = acc ? ggml_add(ctx, acc, v) : v;
        }
        return ggml_scale(ctx, acc, 1.0f / (float) hcm);
    }

    // ---- KDA linear attention ----
    // Short conv over concatenated q/k/v with a persistent (d_conv-1)-column
    // tail, l2-normed q/k, a per-CHANNEL decay gate bounded below
    // multiplicatively, and the fused gated-delta-net recurrence whose state
    // lives in the slot and is committed in-graph. f, g and beta read the layer
    // input, not the convolved q/k/v.
    ggml_tensor * build_kda(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int64_t hd = hp.kda_head_dim;
        const int64_t H = m.tp > 1 && m.tp_heads()
            ? L.head_first[rank + 1] - L.head_first[rank]
            : hp.kda_n_head;
        const int64_t d_inner = hd * H;
        const int64_t dc = hp.d_conv;

        ggml_tensor * qp = recurrent_projection(ctx, LW.kda_wq, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * kp = recurrent_projection(ctx, LW.kda_wk, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * vp = recurrent_projection(ctx, LW.kda_wv, cur, tsg_projection_role::recurrent_update);
        ggml_tensor * qkv = ggml_concat(ctx, ggml_concat(ctx, qp, kp, 0), vp, 0);   // [3*d_inner, nt]
        trace("kda_qkv", il, rank, qkv);

        // stored separately in the file, stacked back into one kernel
        ggml_tensor * conv_w = ggml_concat(ctx,
                ggml_concat(ctx,
                    ggml_reshape_2d(ctx, LW.kda_conv_q, dc, d_inner),
                    ggml_reshape_2d(ctx, LW.kda_conv_k, dc, d_inner), 1),
                ggml_reshape_2d(ctx, LW.kda_conv_v, dc, d_inner), 1);               // [dc, 3*d_inner]

        ggml_tensor * conv_state = slot.kda_conv[rank][(size_t) il];                 // [dc-1, 3*d_inner]
        ggml_tensor * qkv_t = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, qkv)), nt, 3 * d_inner, 1);
        ggml_tensor * conv_in = ggml_concat(ctx, ggml_reshape_3d(ctx, conv_state, dc - 1, 3 * d_inner, 1),
                                            qkv_t, 0);                               // [dc-1+nt, 3*d_inner, 1]
        // SiLU on the conv output, not on the projections
        ggml_tensor * conv_out = ggml_silu(ctx, tsg::ssm_conv_any(ctx, conv_in, conv_w));   // [3*d_inner, nt]
        trace("kda_conv", il, rank, conv_out);

        // keep the last dc-1 columns for the next ubatch. The concat above has
        // materialized conv_in, so overwriting the state here cannot disturb it.
        ggml_tensor * tail = ggml_view_3d(ctx, conv_in, dc - 1, 3 * d_inner, 1,
                conv_in->nb[1], conv_in->nb[2], (size_t) nt * conv_in->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, conv_state));

        const size_t rs_hd = ggml_row_size(conv_out->type, hd);
        ggml_tensor * qc = ggml_view_3d(ctx, conv_out, hd, H, nt, rs_hd, conv_out->nb[1], 0);
        ggml_tensor * kc = ggml_view_3d(ctx, conv_out, hd, H, nt, rs_hd, conv_out->nb[1],
                ggml_row_size(conv_out->type, d_inner));
        ggml_tensor * vc = ggml_view_3d(ctx, conv_out, hd, H, nt, rs_hd, conv_out->nb[1],
                ggml_row_size(conv_out->type, 2 * d_inner));

        // 1e-6 is the reference's own constant, not the model's norm eps. The
        // 1/sqrt(hd) query scale is applied inside the fused op.
        qc = ggml_l2_norm(ctx, ggml_cont(ctx, qc), 1e-6f);
        kc = ggml_l2_norm(ctx, ggml_cont(ctx, kc), 1e-6f);
        qc = ggml_reshape_4d(ctx, qc, hd, H, nt, 1);
        kc = ggml_reshape_4d(ctx, kc, hd, H, nt, 1);
        vc = ggml_reshape_4d(ctx, ggml_cont(ctx, vc), hd, H, nt, 1);

        // forget gate: g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)),
        // per channel; ssm_a holds -exp(A_log), so exp(A_log)*y == -(y * ssm_a).
        ggml_tensor * g = role_mul_mat(ctx, LW.kda_f_b,
            role_mul_mat(ctx, LW.kda_f_a, cur, tsg_projection_role::recurrent_decay), tsg_projection_role::recurrent_decay);
        g = ggml_add(ctx, g, LW.kda_dt_b);
        g = ggml_reshape_3d(ctx, g, hd, H, nt);
        g = ggml_mul(ctx, g, ggml_reshape_3d(ctx, LW.kda_a, 1, H, 1));
        g = ggml_sigmoid(ctx, ggml_scale(ctx, g, -1.0f));
        g = ggml_scale(ctx, g, hp.kda_gate_lb);
        ggml_tensor * g4 = ggml_reshape_4d(ctx, g, hd, H, nt, 1);
        trace("kda_gate", il, rank, g4);

        ggml_tensor * beta = ggml_sigmoid(ctx, role_mul_mat(ctx, LW.kda_beta, cur, tsg_projection_role::recurrent_update));
        ggml_tensor * b4 = ggml_reshape_4d(ctx, beta, 1, H, nt, 1);

        ggml_tensor * state = slot.kda_ssm[rank][(size_t) il];                       // [hd, hd, H]
        ggml_tensor * s4 = ggml_reshape_4d(ctx, state, hd, hd, H, 1);

        ggml_tensor * gdn = ggml_gated_delta_net(ctx, qc, kc, vc, g4, b4, s4, 1);

        const int64_t attn_elems = hd * H * nt;
        ggml_tensor * core = ggml_view_3d(ctx, gdn, hd, H, nt,
                ggml_row_size(gdn->type, hd), ggml_row_size(gdn->type, hd * H), 0);
        ggml_tensor * new_state = ggml_view_3d(ctx, gdn, hd, hd, H,
                ggml_row_size(gdn->type, hd), ggml_row_size(gdn->type, hd * hd),
                ggml_row_size(gdn->type, attn_elems));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, state));
        trace("kda_scan_out", il, rank, core);

        // low-rank output gate; RMS over hd with one weight shared by every
        // head, then a plain sigmoid gate (not FusedRMSNormGated's SiLU).
        ggml_tensor * gate = role_mul_mat(ctx, LW.kda_g_b,
            role_mul_mat(ctx, LW.kda_g_a, cur, tsg_projection_role::recurrent_output_gate), tsg_projection_role::recurrent_output_gate);
        gate = ggml_reshape_3d(ctx, gate, hd, H, nt);
        ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, core), hp.rms_eps), LW.kda_o_norm);
        ggml_tensor * gated = ggml_mul(ctx, normed, ggml_sigmoid(ctx, gate));
        trace("kda_normed", il, rank, gated);

        ggml_tensor * out2 = ggml_reshape_2d(ctx, ggml_cont(ctx, gated), d_inner, nt);
        ggml_tensor * proj = recurrent_projection(ctx, LW.kda_wo, out2, tsg_projection_role::residual_mixing);
        trace("kda_out", il, rank, proj);
        return proj;
    }

    // ---- pooled lightning indexer ----
    // Caches this layer's indexer key AND compressor gate ([key | gate] in one
    // cell row - stored unconditionally, exactly like glm-dsa's keys), then,
    // when scoring, compresses each kpool-cell pool with a softmax over the
    // cached gates (plus a per-slot additive position embedding), scores the
    // POOLS, takes the top-k over pools and expands the winners back to their
    // member cells. Cell-level top-k is NOT equivalent: ReLU drives most pool
    // scores to exactly 0, tie groups span pools, and an unordered top-k then
    // truncates a pool mid-way.
    //
    // No rope (the tower is nope-only) and no Hadamard: the rotation exists to
    // help fp8 kernels, scoring in F32 here it would only cost accuracy.
    ggml_tensor * build_indexer_g5n(int il, int rank, ggml_tensor * cur, ggml_tensor * qr, bool score_now)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int dev = device_of(il, rank);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;
        const int64_t r = hp.indexer_kpool;

        // a genuine LayerNorm: weight AND bias, at eps 1e-6 (from the GGUF).
        ggml_tensor * ik = decode_mul_mat(ctx, LW.idx_attn_k, cur);          // [D, nt]
        ik = ggml_norm(ctx, ik, hp.norm_eps);
        ik = ggml_mul(ctx, ik, LW.idx_k_norm_w);
        ik = ggml_add(ctx, ik, LW.idx_k_norm_b);
        trace("indexer_k", il, rank, ik);

        // the pooling gate is a SECOND, INDEPENDENT projection of the hidden
        // state; it must be cached beside the key because a pool is only built
        // once its members have left the batch.
        ggml_tensor * gate = decode_mul_mat(ctx, LW.idx_comp_gate, cur);     // [D, nt]

        ggml_tensor * packed = ggml_concat(ctx,
                ggml_reshape_3d(ctx, ik, D, 1, nt),
                ggml_reshape_3d(ctx, gate, D, 1, nt), 1);                  // [D, 2, nt]
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, slot.idx_k[rank][il], ggml_reshape_2d(ctx, packed, 2 * D, nt),
                          res.inp.kv_idxs[dev]));

        if (!score_now)
            return nullptr;

        ggml_tensor * kbuf = slot.idx_k[rank][il];                          // [2D, n_ctx] F16
        const int64_t n_pools = n_kv / r;

        // gather each pool's members: one row per cell, key and gate adjacent,
        // so the members are fetched once. ggml_get_rows yields F32, so the
        // compression runs in F32 even though the cache is F16.
        ggml_tensor * kg = ggml_view_2d(ctx, kbuf, 2 * D, n_kv, kbuf->nb[1], 0);
        ggml_tensor * members = ggml_get_rows(ctx, kg, res.inp.pool_cells[dev]);   // [2D, r*n_pools]

        const size_t nb_mem = members->nb[1];
        ggml_tensor * mem_k = ggml_view_3d(ctx, members, D, r, n_pools, nb_mem, nb_mem * r, 0);
        ggml_tensor * mem_g = ggml_view_3d(ctx, members, D, r, n_pools, nb_mem, nb_mem * r,
                                           (size_t) D * members->nb[0]);

        // D independent r-way softmaxes over the SLOT axis; ape is added
        // pre-softmax and is indexed by logical slot (position % kpool), which
        // pool_cells' position order matches by construction.
        ggml_tensor * keys_t = ggml_cont(ctx, ggml_permute(ctx, mem_k, 1, 0, 2, 3));   // [r, D, n_pools]
        ggml_tensor * gate_t = ggml_cont(ctx, ggml_permute(ctx, mem_g, 1, 0, 2, 3));
        ggml_tensor * ape = ggml_cont(ctx, ggml_transpose(ctx, LW.idx_comp_ape));      // [r, D]
        gate_t = ggml_add(ctx, gate_t, ggml_reshape_3d(ctx, ape, r, D, 1));
        ggml_tensor * probs = ggml_soft_max(ctx, gate_t);

        // per-channel weighted average over the pool members -> [D, n_pools]
        ggml_tensor * pool_k = ggml_sum_rows(ctx, ggml_mul(ctx, keys_t, probs));       // [1, D, n_pools]
        pool_k = ggml_reshape_2d(ctx, ggml_cont(ctx, pool_k), D, n_pools);
        trace("indexer_pool_k", il, rank, pool_k);

        ggml_tensor * q = decode_mul_mat(ctx, LW.idx_attn_q_b, qr);           // [D*H, nt]
        q = ggml_reshape_3d(ctx, q, D, H, nt);

        // sign-unconstrained head weights, both scale constants folded in on
        // the small tensor. F32 is not cosmetic: a bf16 head gate moves logits
        // by ~1e-2, enough to swap near-tied pools under a hard top-k cut.
        ggml_tensor * w = decode_mul_mat(ctx, LW.idx_proj, cur);              // [H, nt]
        decode_prec_set_acc(w, GGML_PREC_F32);
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        ggml_tensor * score = nullptr;
        if (m.fused_lid)
        {
            // pool_k stays F32 so the kernel takes its F32 vector path; the
            // pool-visibility bias rides in as the mask.
            ggml_tensor * pool_kf = ggml_reshape_3d(ctx, pool_k, D, 1, n_pools);
            score = ggml_lightning_indexer(ctx, q, pool_kf, w, res.inp.pool_bias[dev]);   // [n_pools, nt]
        }
        else
        {
            ggml_tensor * qp2 = ggml_permute(ctx, q, 0, 2, 1, 3);           // [D, nt, H]
            ggml_tensor * kq = decode_mul_mat(ctx, pool_k, qp2);              // [n_pools, nt, H]
            // the ReLU sits BETWEEN the per-head dot product and the head
            // weighting; the weights are sign-free, so moving it is a
            // different function.
            kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));         // [H, nt, n_pools]
            ggml_tensor * sc = ggml_relu(ctx, kq);
            sc = ggml_mul(ctx, sc, w);
            sc = ggml_sum_rows(ctx, sc);                                    // [1, nt, n_pools]
            sc = ggml_cont(ctx, ggml_permute(ctx, sc, 2, 1, 0, 3));         // [n_pools, nt, 1]
            score = ggml_add(ctx, sc, res.inp.pool_bias[dev]);
        }
        trace("indexer_pool_score", il, rank, score);

        // top-k over POOLS, then expand each winner to its member cells.
        const int64_t select_k = std::min<int64_t>(n_pools, hp.indexer_top_k / r);
        ggml_tensor * sel = ggml_cont(ctx, ggml_top_k(ctx, score, (int) select_k));    // I32 [select_k, nt]
        ggml_tensor * pc2 = ggml_reshape_2d(ctx, res.inp.pool_cells[dev], r, n_pools);
        ggml_tensor * sel_flat = ggml_reshape_1d(ctx, sel, select_k * nt);
        ggml_tensor * cells = ggml_get_rows(ctx, pc2, sel_flat);            // I32 [r, select_k*nt]
        ggml_tensor * out = ggml_reshape_2d(ctx, cells, r * select_k, nt);
        trace("indexer_top_k", il, rank, out);
        return out;
    }

    /// build_topk_mask plus the always-attended trailing cells of each query's
    /// own incomplete pool: -inf canvas, write the trail lane values (0 for a
    /// real tail cell, -inf for an unused lane - a no-op on the canvas), then
    /// unmask the selected pools' cells, then add the causal mask back so a
    /// selected-but-future cell stays masked. The trail write runs FIRST so a
    /// pool scatter landing on the same cell wins.
    ggml_tensor * build_topk_mask_g5n(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k,
                                      ggml_tensor * trail_cells, ggml_tensor * trail_vals)
    {
        const int64_t n_top_k = top_k->ne[0];
        const int64_t r = trail_cells->ne[0];
        trace("topk", trace_il, trace_rank, top_k);

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);               // [n_kv, nt]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, n_kv, nt, base->nb[0], base->nb[1], 0);

        ggml_tensor * t_idx = ggml_view_3d(ctx, trail_cells, r, nt, 1,
                                           trail_cells->nb[1], trail_cells->nb[2], 0);
        ggml_tensor * with_tail = ggml_set_rows(ctx, all, trail_vals, t_idx);   // [1, n_kv, nt]

        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, nt, 1,
                                         top_k->nb[1], top_k->nb[2], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, nt), 0.0f);
        ggml_tensor * unmasked = ggml_set_rows(ctx, with_tail, zeros, idx);

        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, n_kv, nt, unmasked->nb[2], 0);
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);

        pin(base, dev);
        pin(zeros, dev);
        pin(with_tail, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    // ---- the glm5next trunk ----

    /// One wholly rank-local copy of the glm5next trunk. Every nonlinear and
    /// hyper-connection operation is replicated; only the row-parallel
    /// attention projection and routed-expert down projection stop the graph.
    /// The segmented driver AllReduces those F32 partials in place before it
    /// resumes the following hyper-connection segment on every rank.
    void build_g5n_rank(int rank)
    {
        graph_inputs & inp = res.inp;
        const int dev = m.rank_device(rank);
        const int64_t hcm = hp.hc_mult;
        const int64_t pool = hp.indexer_kpool;
        const int64_t n_pools = n_kv / pool;

        bool has_mla = false;
        for (int il = 0; il < hp.n_layer; il++) has_mla = has_mla || !m.layers[il].recurrent;
        if (has_mla)
        {
            const ggml_type mask_type = m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
            char nb[64];
            snprintf(nb, sizeof(nb), "kq_mask.%d", dev);
            inp.kq_mask[dev] = new_input(mask_type, n_kv, nt, nb, dev);
            snprintf(nb, sizeof(nb), "kv_idxs.%d", dev);
            inp.kv_idxs[dev] = new_input(GGML_TYPE_I64, nt, 0, nb, dev);
            if (sparse)
            {
                snprintf(nb, sizeof(nb), "pool_cells.%d", dev);
                inp.pool_cells[dev] = new_input(GGML_TYPE_I32, pool * n_pools, 0, nb, dev);
                snprintf(nb, sizeof(nb), "pool_bias.%d", dev);
                inp.pool_bias[dev] = new_input(m.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                               n_pools, nt, nb, dev);
                snprintf(nb, sizeof(nb), "trail_cells.%d", dev);
                inp.trail_cells[dev] = new_input(GGML_TYPE_I32, pool, nt, nb, dev);
                snprintf(nb, sizeof(nb), "trail_vals.%d", dev);
                inp.trail_vals[dev] = new_input_3d(GGML_TYPE_F32, 1, pool, nt, nb, dev);
            }
        }

        {
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", dev);
            inp.tokens[dev] = new_input(GGML_TYPE_I32, nt, 0, nb, dev);
            snprintf(nb, sizeof(nb), "inp_pos.%d", dev);
            inp.pos[dev] = new_input(GGML_TYPE_I32, nt, 0, nb, dev);
        }
        if (rank == 0 && res.want_logits)
            inp.out_ids = new_input(GGML_TYPE_I32, res.n_out, 0, "inp_out_ids", dev);

        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd_rank[rank], inp.tokens[dev]);
        if (res.n_ovr > 0)
        {
            inp.embd_rows = new_input(GGML_TYPE_F32, hp.n_embd, res.n_ovr, "embd_rows", dev);
            inp.embd_idx = new_input(GGML_TYPE_I64, res.n_ovr, 0, "embd_idx", dev);
            emb = ggml_set_rows(ctx, emb, inp.embd_rows, inp.embd_idx);
        }
        ggml_tensor * inpL = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, hp.n_embd, 1, nt),
                                            hp.n_embd, hcm, nt, 1);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];
            const glm_layer_weights & W = L.w[rank];

            // ---- attention crossing -------------------------------------
            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;
            ggml_tensor * hc_cur = build_hc_pre(inpL, W.hc_attn_fn, W.hc_attn_scale, W.hc_attn_base,
                                                &post, &comb);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            ggml_tensor * cur = rms(hc_cur, W.attn_norm);
            ggml_tensor * partial = nullptr;
            if (L.recurrent)
            {
                partial = build_kda(il, rank, cur);
            }
            else
            {
                ggml_tensor * qr = rms(decode_mul_mat(ctx, W.wq_a, cur), W.q_a_norm);
                ggml_tensor * mask = inp.kq_mask[dev];
                if (L.indexer_full)
                {
                    ggml_tensor * tk = build_indexer_g5n(il, rank, cur, qr, sparse);
                    if (sparse && tk)
                        mask = build_topk_mask_g5n(dev, inp.kq_mask[dev], tk,
                                                   inp.trail_cells[dev], inp.trail_vals[dev]);
                }
                partial = build_attention(il, rank, cur, qr, inp.pos[dev], mask, inp.kv_idxs[dev]);
            }
            res.tp_plan.ar_tensor.push_back(partial);
            res.tp_boundary.push_back(partial);
            inpL = build_hc_post(partial, residual, post, comb);

            // ---- FFN crossing -------------------------------------------
            residual = inpL;
            hc_cur = build_hc_pre(inpL, W.hc_ffn_fn, W.hc_ffn_scale, W.hc_ffn_base, &post, &comb);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            ggml_tensor * ffn = nullptr;
            cur = rms(hc_cur, W.ffn_norm);
            if (!L.is_moe)
            {
                // Dense layers are deliberately replicated. They see the same
                // residual on every rank after the preceding collective.
                ffn = build_dense_ffn(il, rank, cur);
            }
            else
            {
                ffn = build_moe(il, rank, cur);
                // Cut after the routed partial itself. Expanding it now keeps
                // the replicated shared expert in the following segment, so
                // every rank evaluates the established arithmetic order:
                // (routed rank 0 + routed rank 1 + ...) + shared.
                ggml_build_forward_expand(gf, ffn);
                res.tp_plan.ar_tensor.push_back(ffn);
                res.tp_boundary.push_back(ffn);
                ggml_tensor * shared = build_shexp(il, rank, cur);
                if (shared) ffn = ggml_add(ctx, ffn, shared);
            }
            inpL = build_hc_post(ffn, residual, post, comb);
        }

        if (rank != 0 || !res.want_logits)
        {
            // Non-root ranks still need their final HC post to finish so their
            // replicated residual is ready for a cached graph's next replay.
            ggml_build_forward_expand(gf, inpL);
            return;
        }

        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hcm * hp.n_embd, nt);
        ggml_tensor * selr = ggml_get_rows(ctx, flat, inp.out_ids);
        ggml_tensor * x3 = ggml_reshape_3d(ctx, selr, hp.n_embd, hcm, res.n_out);
        ggml_tensor * out = hc_mean(x3);
        out = rms(out, m.output_norm);
        out = decode_mul_mat(ctx, m.output, out);
        ggml_set_output(out);
        ggml_set_name(out, "logits");
        res.logits = out;
        ggml_build_forward_expand(gf, out);
    }

    void build_g5n()
    {
        if (res.tp_fused)
        {
            build_g5n_rank(res.tp_rank);
            return;
        }
        graph_inputs & inp = res.inp;
        const int64_t hcm = hp.hc_mult;
        const int64_t r = hp.indexer_kpool;
        const int64_t n_pools = n_kv / r;

        std::vector<uint8_t> used_dev((size_t) m.n_gpu + 1, 0);
        if (m.tp > 1)
            for (int rank = 0; rank < m.tp; rank++) used_dev[(size_t) m.rank_device(rank)] = 1;
        else
            for (int il = 0; il < hp.n_layer; il++) used_dev[(size_t) m.layers[il].device] = 1;

        // Only MLA layers read the mask, the cache indices and the pool inputs;
        // a KDA-only device gets none (an input no node reads is never
        // allocated, and writing to it would fault).
        const ggml_type mask_type = m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        std::vector<uint8_t> dev_mla((size_t) m.n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++)
        {
            if (m.layers[il].recurrent) continue;
            if (m.tp > 1 && m.tp_heads())
                for (int rank = 0; rank < m.tp; rank++) dev_mla[(size_t) m.rank_device(rank)] = 1;
            else
                dev_mla[(size_t) (m.tp > 1 ? m.rank_device(0) : m.layers[il].device)] = 1;
        }

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used_dev[(size_t) d] || !dev_mla[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "kq_mask.%d", d);
            inp.kq_mask[d] = new_input(mask_type, n_kv, nt, nb, d);
            snprintf(nb, sizeof(nb), "kv_idxs.%d", d);
            inp.kv_idxs[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
            if (sparse)
            {
                snprintf(nb, sizeof(nb), "pool_cells.%d", d);
                inp.pool_cells[d] = new_input(GGML_TYPE_I32, r * n_pools, 0, nb, d);
                snprintf(nb, sizeof(nb), "pool_bias.%d", d);
                inp.pool_bias[d] = new_input(m.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32, n_pools, nt, nb, d);
                snprintf(nb, sizeof(nb), "trail_cells.%d", d);
                inp.trail_cells[d] = new_input(GGML_TYPE_I32, r, nt, nb, d);
                snprintf(nb, sizeof(nb), "trail_vals.%d", d);
                inp.trail_vals[d] = new_input_3d(GGML_TYPE_F32, 1, r, nt, nb, d);
            }
        }

        const int dev_embd = m.tp > 1 ? m.rank_device(0) : m.layers[0].device;
        {
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", dev_embd);
            inp.tokens[dev_embd] = new_input(GGML_TYPE_I32, nt, 0, nb, dev_embd);
        }
        const int dev_last = m.tp > 1 ? m.rank_device(0) : m.layers[(size_t) hp.n_layer - 1].device;
        if (res.want_logits)
            inp.out_ids = new_input(GGML_TYPE_I32, res.n_out, 0, "inp_out_ids", dev_last);

        // hc_mult exact copies of the embedding: no scaling, no one-hot.
        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[dev_embd]);
        if (res.n_ovr > 0)
        {
            // Vision rows: the projected image embeddings replace the token
            // embeddings of the placeholder positions before the stream fans
            // out. get_rows leaves `emb` contiguous, so set_rows may write
            // straight through it.
            inp.embd_rows = new_input(GGML_TYPE_F32, hp.n_embd, res.n_ovr, "embd_rows", dev_embd);
            inp.embd_idx = new_input(GGML_TYPE_I64, res.n_ovr, 0, "embd_idx", dev_embd);
            emb = ggml_set_rows(ctx, emb, inp.embd_rows, inp.embd_idx);
        }
        ggml_tensor * inpL = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, hp.n_embd, 1, nt),
                                            hp.n_embd, hcm, nt, 1);

        ggml_tensor * part[MAX_GPUS] = {};
        ggml_tensor * shexp[MAX_GPUS] = {};
        const int n_attn = m.tp > 1 && m.tp_heads() ? m.tp : 1;
        const int n_moe = m.tp > 1 && m.tp_experts() ? m.tp : 1;

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];
            const glm_layer_weights & LW = L.w[0];
            const int dev = device_of(il, 0);

            if (m.tp == 1 && il > 0 && dev != m.layers[il - 1].device)
                pin(inpL, dev);

            // ---- attention crossing ----
            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;
            ggml_tensor * hc_cur = build_hc_pre(inpL, LW.hc_attn_fn, LW.hc_attn_scale, LW.hc_attn_base,
                                                &post, &comb);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);
            trace("hc_attn_pre", il, 0, hc_cur);

            for (int rank = 0; rank < n_attn; rank++)
            {
                const glm_layer_weights & RW = L.w[rank];
                const int rank_dev = device_of(il, rank);
                ggml_tensor * cur = rms(hc_cur, RW.attn_norm);
                trace("attn_norm", il, rank, cur);

                if (L.recurrent)
                {
                    part[rank] = build_kda(il, rank, cur);
                }
                else
                {
                    ggml_tensor * qr = rms(decode_mul_mat(ctx, RW.wq_a, cur), RW.q_a_norm);
                    trace("qr", il, rank, qr);
                    ggml_tensor * mask = inp.kq_mask[rank_dev];
                    if (L.indexer_full)
                    {
                        ggml_tensor * tk = build_indexer_g5n(il, rank, cur, qr, sparse);
                        trace_il = il; trace_rank = rank;
                        if (sparse && tk)
                            mask = build_topk_mask_g5n(rank_dev, inp.kq_mask[rank_dev], tk,
                                                       inp.trail_cells[rank_dev], inp.trail_vals[rank_dev]);
                    }
                    part[rank] = build_attention(il, rank, cur, qr, inp.pos[rank_dev], mask,
                                                 inp.kv_idxs[rank_dev]);
                }
                trace("attn_out", il, rank, part[rank]);
            }

            // Both MLA and KDA end in a row-parallel output projection. Reduce
            // the partial hidden vectors before the nonlinear hyper-connection.
            ggml_tensor * attn = reduce_ranks(part, n_attn);
            inpL = build_hc_post(attn, residual, post, comb);
            trace("hc_attn_post", il, 0, inpL);

            // ---- FFN crossing ----
            residual = inpL;
            hc_cur = build_hc_pre(inpL, LW.hc_ffn_fn, LW.hc_ffn_scale, LW.hc_ffn_base, &post, &comb);
            // expand before the sublayer so op offload cannot pull the mHC
            // state onto the expert weights' backend
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            ggml_tensor * ffn = nullptr;
            if (!L.is_moe)
            {
                ggml_tensor * cur = rms(hc_cur, LW.ffn_norm);
                trace("ffn_norm", il, 0, cur);
                ffn = build_dense_ffn(il, 0, cur);
            }
            else
            {
                // CPU-offloaded experts stay whole and run once. Otherwise each
                // rank evaluates its row slice; the shared expert is replicated
                // and is therefore added exactly once after the reduction.
                const int n_moe_l = L.cpu_moe ? 1 : n_moe;
                for (int rank = 0; rank < n_moe_l; rank++)
                {
                    ggml_tensor * cur = rms(hc_cur, L.w[rank].ffn_norm);
                    part[rank] = build_moe(il, rank, cur);
                    shexp[rank] = rank == 0 ? build_shexp(il, 0, cur) : nullptr;
                    trace("ffn_norm", il, rank, cur);
                    trace("moe_out", il, rank, part[rank]);
                }
                ffn = reduce_ranks(part, n_moe_l);
                if (shexp[0]) ffn = ggml_add(ctx, ffn, shexp[0]);
            }
            trace("ffn_out", il, 0, ffn);

            inpL = build_hc_post(ffn, residual, post, comb);
            trace("l_out", il, 0, inpL);
        }

        if (!res.want_logits && !res.want_h)
        {
            // A non-final prefill chunk only has to leave its caches and
            // recurrent state behind.
            ggml_build_forward_expand(gf, inpL);
            return;
        }

        if (res.want_h)
        {
            // Speculation wants one post-final-norm hidden state per token
            // (llama.cpp's h_nextn), so the stream mean and the norm run over
            // EVERY row and the LM head selects from the normed rows. Both are
            // row-wise, so the rows the head reads are bit-identical to the
            // select-first branch below; only rows nobody reads are extra, and
            // at a speculative window that is a handful.
            ggml_tensor * x3all = ggml_reshape_3d(ctx, inpL, hp.n_embd, hcm, nt);
            ggml_tensor * hn = rms(hc_mean(x3all), m.output_norm);
            ggml_set_output(hn);
            ggml_set_name(hn, "h_nextn");
            res.h_nextn = hn;
            ggml_build_forward_expand(gf, hn);
            if (!res.want_logits)
                return;
            ggml_tensor * hsel = ggml_get_rows(ctx, hn, inp.out_ids);
            hsel = decode_mul_mat(ctx, m.output, hsel);
            ggml_set_output(hsel);
            ggml_set_name(hsel, "logits");
            res.logits = hsel;
            ggml_build_forward_expand(gf, hsel);
            return;
        }

        // select the output rows BEFORE the mean: one token's streams are one
        // contiguous row of the flattened stream tensor.
        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hcm * hp.n_embd, nt);
        ggml_tensor * selr = ggml_get_rows(ctx, flat, inp.out_ids);
        ggml_tensor * x3 = ggml_reshape_3d(ctx, selr, hp.n_embd, hcm, res.n_out);

        // unweighted mean over the streams, then the ordinary norm + head.
        ggml_tensor * cur = hc_mean(x3);
        cur = rms(cur, m.output_norm);
        cur = decode_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
    }

    void build()
    {
        if (hp.g5n) { build_g5n(); return; }

        graph_inputs & inp = res.inp;

        // Per-token inputs are duplicated per participating device so the
        // scheduler never inserts a synchronized cross-backend input copy
        // inside the per-token compute.
        std::vector<uint8_t> used_dev((size_t) m.n_gpu + 1, 0);
        if (m.tp > 1)
            for (int r = 0; r < m.tp; r++) used_dev[(size_t) m.rank_device(r)] = 1;
        else
            for (int il = 0; il < hp.n_layer; il++) used_dev[(size_t) m.layers[il].device] = 1;

        // Only the token ids are single-device (the embedding lookup); the
        // positions, masks and cache indices are consumed by every layer, so
        // each device gets its own copy and the scheduler never has to insert a
        // synchronized cross-backend input copy inside the per-token compute.
        //
        // An input no node reads is never allocated by the graph allocator, so
        // creating one for a device that turns out not to need it would leave a
        // buffer-less tensor for the input fill to write into. Only devices that
        // actually host a layer get inputs, and the "has a full indexer" test
        // gates the indexer mask the same way.
        const ggml_type mask_type = m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        std::vector<uint8_t> dev_has_indexer((size_t) m.n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++)
        {
            if (!m.layers[il].indexer_full) continue;
            if (m.tp > 1) { for (int r = 0; r < m.tp; r++) dev_has_indexer[(size_t) m.rank_device(r)] = 1; }
            else dev_has_indexer[(size_t) m.layers[il].device] = 1;
        }

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used_dev[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input(GGML_TYPE_I32, nt, 0, nb, d);
            snprintf(nb, sizeof(nb), "kq_mask.%d", d);
            inp.kq_mask[d] = new_input(mask_type, n_kv, nt, nb, d);
            snprintf(nb, sizeof(nb), "kv_idxs.%d", d);
            inp.kv_idxs[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
            if (sparse && dev_has_indexer[(size_t) d])
            {
                snprintf(nb, sizeof(nb), "lid_mask.%d", d);
                inp.lid_mask[d] = new_input(GGML_TYPE_F16, n_kv, nt, nb, d);
            }
        }
        {
            const int dev_embd = m.tp > 1 ? 0 : m.layers[0].device;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", dev_embd);
            inp.tokens[dev_embd] = new_input(GGML_TYPE_I32, nt, 0, nb, dev_embd);
        }
        const int dev_last = m.tp > 1 ? 0 : m.layers[(size_t) hp.n_layer - 1].device;
        // Only when the LM head runs: an input no node reads is never allocated,
        // and writing to it would fault.
        if (res.want_logits)
            inp.out_ids = new_input(GGML_TYPE_I32, res.n_out, 0, "inp_out_ids", dev_last);

        ggml_tensor * inpL = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.tp > 1 ? 0 : m.layers[0].device]);

        // `top_k` deliberately survives a device (or rank) change. A "shared"
        // indexer layer must see the selection of the last FULL layer wherever
        // that landed, exactly as llama.cpp does — dropping it would silently run
        // the layers after a boundary with dense attention, which is a different
        // model.
        ggml_tensor * top_k[MAX_GPUS] = {};
        // The attention mask that goes with the current selection, per rank.
        ggml_tensor * topk_mask[MAX_GPUS] = {};
        ggml_tensor * part[MAX_GPUS] = {};
        ggml_tensor * shexp[MAX_GPUS] = {};
        // A half that is not sharded is computed once, on rank 0: every rank
        // holds the same weights there, so summing N identical partials would
        // multiply the result by N.
        const int n_attn = (m.tp > 1 && m.tp_heads()) ? m.tp : 1;
        const int n_moe = (m.tp > 1 && m.tp_experts()) ? m.tp : 1;

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];

            if (m.tp == 1 && il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            // ---- attention: every rank runs its own heads ----------------------
            ggml_tensor * residual = inpL;
            for (int r = 0; r < n_attn; r++)
            {
                const int dev = device_of(il, r);
                ggml_tensor * cur = rms(inpL, L.w[r].attn_norm);
                ggml_tensor * qr = rms(decode_mul_mat(ctx, L.w[r].wq_a, cur), L.w[r].q_a_norm);

                trace("attn_norm", il, r, cur);
                trace("qr", il, r, qr);

                if (L.indexer_full)
                {
                    top_k[r] = build_indexer(il, r, cur, qr, inp.pos[dev], inp.lid_mask[dev],
                                             inp.kv_idxs[dev], sparse);
                    trace_il = il; trace_rank = r;
                    topk_mask[r] = (sparse && top_k[r]) ? build_topk_mask(dev, inp.kq_mask[dev], top_k[r]) : nullptr;
                }

                ggml_tensor * mask = (sparse && topk_mask[r]) ? topk_mask[r] : inp.kq_mask[dev];
                part[r] = build_attention(il, r, cur, qr, inp.pos[dev], mask, inp.kv_idxs[dev]);
                trace("attn_out", il, r, part[r]);
            }
            // Row-parallel output projection: the ranks' partials sum to the
            // dense result.
            inpL = ggml_add(ctx, residual, reduce_ranks(part, n_attn));
            trace("ffn_inp", il, 0, inpL);

            // ---- FFN ------------------------------------------------------------
            residual = inpL;
            if (!L.is_moe)
            {
                // Dense layers are replicated, so rank 0's result IS the result.
                ggml_tensor * cur = rms(inpL, L.w[0].ffn_norm);
                inpL = ggml_add(ctx, residual, build_dense_ffn(il, 0, cur));
                continue;
            }

            // A layer whose experts stayed on the host also stayed whole, so
            // rank 0's result is the whole result.
            const int n_moe_l = L.cpu_moe ? 1 : n_moe;
            for (int r = 0; r < n_moe_l; r++)
            {
                ggml_tensor * cur = rms(inpL, L.w[r].ffn_norm);
                part[r] = build_moe(il, r, cur);
                shexp[r] = r == 0 ? build_shexp(il, 0, cur) : nullptr;
                trace("ffn_norm", il, r, cur);
                trace("moe_out", il, r, part[r]);
            }
            // The shared expert is replicated, so it is added ONCE, after the
            // routed partials have been reduced.
            ggml_tensor * ffn = reduce_ranks(part, n_moe_l);
            if (shexp[0]) ffn = ggml_add(ctx, ffn, shexp[0]);
            inpL = ggml_add(ctx, residual, ffn);
            trace("l_out", il, 0, inpL);
        }

        if (!res.want_logits && !res.want_h)
        {
            // A non-final prefill chunk only has to leave its KV behind. Running
            // the 154880-row LM head anyway would re-read the whole output matrix
            // once per chunk for a result nobody reads.
            ggml_build_forward_expand(gf, inpL);
            return;
        }

        if (!res.want_logits)
        {
            // A speculative prefill chunk: no logits, but the draft head still
            // needs this chunk's hidden states, so the final norm runs and the
            // LM head does not. (Without this branch the early return above
            // would hand back a graph with no h_nextn at all.)
            ggml_tensor * hn_only = rms(inpL, m.output_norm);
            ggml_set_output(hn_only);
            ggml_set_name(hn_only, "h_nextn");
            res.h_nextn = hn_only;
            ggml_build_forward_expand(gf, hn_only);
            return;
        }

        if (!res.want_h)
        {
            ggml_tensor * cur = ggml_get_rows(ctx, inpL, inp.out_ids);
            cur = rms(cur, m.output_norm);
            cur = decode_mul_mat(ctx, m.output, cur);
            ggml_set_output(cur);
            ggml_set_name(cur, "logits");
            res.logits = cur;
            ggml_build_forward_expand(gf, cur);
            return;
        }

        // Speculation needs one hidden state per token, so the final norm runs
        // over every row and the LM head selects from the NORMED rows instead of
        // the other way round. RMS norm is row-wise, so the rows the head reads
        // are bit-identical to the branch above — only the rows nobody reads are
        // extra, and at a speculative window that is a handful.
        ggml_tensor * hn = rms(inpL, m.output_norm);
        ggml_set_output(hn);
        ggml_set_name(hn, "h_nextn");
        res.h_nextn = hn;
        ggml_build_forward_expand(gf, hn);

        ggml_tensor * cur = ggml_get_rows(ctx, hn, inp.out_ids);
        cur = decode_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
    }

    // ---- NextN/MTP draft block ------------------------------------------
    //
    //   h_mtp = shared_head_norm( block( eh_proj([enorm(embed(t)) ; hnorm(h)]) ) )
    //
    // `block` is an ordinary glm-dsa decoder block reusing the trunk builders,
    // with one deliberate difference: the attention is DENSE. The block ships
    // lightning-indexer weights, but llama.cpp's graph_mtp builds the plain MLA
    // attention input and never reads them, and it has no indexer key cache to
    // score against — so running the indexer here would be a different model,
    // not a faster one.
    void build_mtp()
    {
        graph_inputs & inp = res.inp;
        const int il = m.mtp_layer;
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & W0 = L.w[0];
        const ggml_type mask_type = m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

        std::vector<uint8_t> used_dev((size_t) m.n_gpu + 1, 0);
        if (m.tp > 1) { for (int r = 0; r < m.tp; r++) used_dev[(size_t) m.rank_device(r)] = 1; }
        else used_dev[(size_t) L.device] = 1;

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used_dev[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "mtp_pos.%d", d);
            inp.pos[d] = new_input(GGML_TYPE_I32, nt, 0, nb, d);
            snprintf(nb, sizeof(nb), "mtp_kq_mask.%d", d);
            inp.kq_mask[d] = new_input(mask_type, n_kv, nt, nb, d);
            snprintf(nb, sizeof(nb), "mtp_kv_idxs.%d", d);
            inp.kv_idxs[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
        }

        const int dev0 = m.tp > 1 ? 0 : L.device;
        inp.tokens[dev0] = new_input(GGML_TYPE_I32, nt, 0, "mtp_tokens", dev0);
        inp.h_in[dev0] = new_input(GGML_TYPE_F32, hp.n_embd, nt, "mtp_h_in", dev0);
        if (res.want_logits)
            inp.out_ids = new_input(GGML_TYPE_I32, res.n_out, 0, "mtp_out_ids", dev0);

        // The NextN wiring is replicated on every rank, so rank 0's result IS
        // the result and the block's input is computed once (like inpL).
        ggml_tensor * embd_w = W0.nextn_embd ? W0.nextn_embd : m.tok_embd;
        ggml_tensor * tok = ggml_get_rows(ctx, embd_w, inp.tokens[dev0]);
        ggml_tensor * e_norm = rms(tok, W0.nextn_enorm);
        ggml_tensor * h_norm = rms(inp.h_in[dev0], W0.nextn_hnorm);
        ggml_tensor * cat = ggml_concat(ctx, e_norm, h_norm, 0);        // [2*n_embd, nt]
        ggml_tensor * inpSA = decode_mul_mat(ctx, W0.nextn_eh_proj, cat); // [n_embd, nt]

        // ---- attention: every rank runs its own heads ---------------------
        ggml_tensor * part[MAX_GPUS] = {};
        const int n_attn = (m.tp > 1 && m.tp_heads()) ? m.tp : 1;
        for (int r = 0; r < n_attn; r++)
        {
            const int dev = device_of(il, r);
            ggml_tensor * cur = rms(inpSA, L.w[r].attn_norm);
            ggml_tensor * qr = rms(decode_mul_mat(ctx, L.w[r].wq_a, cur), L.w[r].q_a_norm);
            part[r] = build_attention(il, r, cur, qr, inp.pos[dev], inp.kq_mask[dev], inp.kv_idxs[dev]);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx, inpSA, reduce_ranks(part, n_attn));

        // ---- FFN ----------------------------------------------------------
        ggml_tensor * out = nullptr;
        if (!L.is_moe)
        {
            ggml_tensor * cur = rms(ffn_inp, L.w[0].ffn_norm);
            out = ggml_add(ctx, ffn_inp, build_dense_ffn(il, 0, cur));
        }
        else
        {
            ggml_tensor * mpart[MAX_GPUS] = {};
            ggml_tensor * shexp0 = nullptr;
            const int n_moe_l = L.cpu_moe ? 1 : ((m.tp > 1 && m.tp_experts()) ? m.tp : 1);
            for (int r = 0; r < n_moe_l; r++)
            {
                ggml_tensor * cur = rms(ffn_inp, L.w[r].ffn_norm);
                mpart[r] = build_moe(il, r, cur);
                if (r == 0) shexp0 = build_shexp(il, 0, cur);
            }
            ggml_tensor * ffn = reduce_ranks(mpart, n_moe_l);
            if (shexp0) ffn = ggml_add(ctx, ffn, shexp0);
            out = ggml_add(ctx, ffn_inp, ffn);
        }

        // shared_head_norm both seeds the next draft step and feeds the LM head.
        ggml_tensor * head_norm_w = W0.nextn_head_norm ? W0.nextn_head_norm : m.output_norm;
        ggml_tensor * hn = rms(out, head_norm_w);
        ggml_set_output(hn);
        ggml_set_name(hn, "mtp_h_nextn");
        res.h_nextn = hn;
        ggml_build_forward_expand(gf, hn);

        if (res.want_logits)
        {
            ggml_tensor * head_w = W0.nextn_head ? W0.nextn_head : m.output;
            ggml_tensor * sel = ggml_get_rows(ctx, hn, inp.out_ids);
            ggml_tensor * logits = decode_mul_mat(ctx, head_w, sel);
            ggml_set_output(logits);
            ggml_set_name(logits, "logits");
            res.logits = logits;
            ggml_build_forward_expand(gf, logits);
        }
    }
};

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

static bool glm_selected_expert_callback(ggml_tensor* tensor, bool ask, void* data)
{
    auto& graph = *static_cast<graph_build_result*>(data);
    const auto found = graph.selected_expert_boundaries.find(tensor);
    if (ask) return graph.selected_expert_failed || found != graph.selected_expert_boundaries.end();
    if (graph.selected_expert_failed || found == graph.selected_expert_boundaries.end()) return false;
    // The public scheduler has completed and synchronized this producer range.
    // It never resets its allocator before continuing the dependent nodes.
    try {
        if (found->second != graph.selected_expert_completed) throw std::runtime_error("GLM expert boundaries executed out of order");
        const auto& seam = graph.selected_expert_seams[found->second];
        if (seam.segment.n_used < 1 || seam.segment.n_used > 64
            || !tsg::host_moe_cache_prefill_fits(seam.segment.seq_len, seam.segment.n_used, true)
            || (seam.segment.independent_decode_rows && seam.segment.seq_len > tsg::kHostMoeCacheMaxDecodeRows))
            throw std::runtime_error("GLM selected-expert callback exceeds its complete row/route bound");
        const auto routes = static_cast<size_t>(seam.segment.n_used) * seam.segment.seq_len;
        graph.selected_expert_ids.resize(routes);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != graph.selected_expert_error_epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("GLM selected-expert producer failed before IDs download");
        const auto errors_before_ids = tsg::g_ggml_error_count.load(std::memory_order_acquire);
        ggml_backend_tensor_get(seam.segment.sel_ids, graph.selected_expert_ids.data(), 0, routes * sizeof(int32_t));
        glm_test_batched_dispatch_fault(4);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != errors_before_ids
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("GLM selected expert IDs download failed before cache dispatch");
        const int cached = tsg::host_moe_cached_experts(seam.segment, nullptr, graph.selected_expert_ids.data(),
            nullptr, nullptr,
            !seam.segment.independent_decode_rows
                && seam.segment.seq_len > TSG_PRECISION_DECODE_COLUMNS
                ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", seam.backend);
        // A quota or capability refusal after router dispatch is not a safe
        // place to silently change expert arithmetic. Fence this request and
        // retain its state until explicit reset instead of mixing CPU/GPU rows.
        if (cached != 1) throw std::runtime_error(std::string("GLM selected-expert CUDA cache declined the dispatched route union: ") + tsg::g_last_error);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != graph.selected_expert_error_epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("GLM selected-expert helper failed before final drain");
        // Even the device bridge can leave a final copy queued. Drain the same
        // actual model stream before a consumer or cache eviction observes it.
        tsg::sync_backend(seam.backend);
        if (found->second + 1 == graph.selected_expert_seams.size()) glm_test_batched_dispatch_fault(5);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != graph.selected_expert_error_epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("GLM selected-expert final drain failed before seam completion");
        glm_test_batched_dispatch_fault(3);
        ++graph.selected_expert_completed;
        return true;
    } catch (const std::exception& error) {
        tsg::set_last_error(std::string("GLM selected-expert callback failed: ") + error.what());
    } catch (...) {
        tsg::set_last_error("GLM selected-expert callback failed with a non-standard exception");
    }
    // Upstream's callback false only stops its current split. The caller checks
    // this latch before publishing any logits or heads and fences every slot.
    graph.selected_expert_failed = true;
    return false;
}

static bool glm_prepare_selected_expert_graph(glm_model& model, graph_build_result& graph)
{
    if (graph.selected_expert_seams.empty()) return true;
    auto* backend = graph.allocation_budget.scheduler_backend(model.sched_backends[0]);
    if (!backend || model.n_gpu != 1 || model.tp != 1) return false;
    // An externally produced input must never be copied by an earlier CPU split
    // before its callback. Require all consumers/producers on the exact same
    // CUDA alias and reject unsupported nodes before dispatch, without fallback
    // to an unqualified schedule or resetting a live allocator.
    for (int i = 0; i < ggml_graph_n_nodes(graph.gf); ++i) {
        auto* node = graph.gf->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            tsg::set_last_error(std::string("GLM selected-expert CUDA graph lacks operation ") + ggml_op_name(node->op));
            return false;
        }
        ggml_backend_sched_set_tensor_backend(graph.sched, node, backend);
    }
    for (const auto& seam : graph.selected_expert_seams) {
        if (seam.segment.n_used < 1 || seam.segment.n_used > 64
            || !tsg::host_moe_cache_prefill_fits(seam.segment.seq_len, seam.segment.n_used, true)
            || (seam.segment.independent_decode_rows && seam.segment.seq_len > tsg::kHostMoeCacheMaxDecodeRows)) {
            tsg::set_last_error("GLM selected-expert graph exceeds its complete row/route bound");
            return false;
        }
        ggml_backend_sched_set_tensor_backend(graph.sched, seam.segment.moe_out, backend);
        ggml_backend_sched_set_tensor_backend(graph.sched, seam.segment.moe_in, backend);
        ggml_backend_sched_set_tensor_backend(graph.sched, seam.segment.sel_ids, backend);
        ggml_backend_sched_set_tensor_backend(graph.sched, seam.boundary, backend);
    }
    ggml_backend_sched_set_eval_callback(graph.sched, glm_selected_expert_callback, &graph);
    return true;
}

static bool glm_check_selected_expert_assignment(glm_model& model, graph_build_result& graph)
{
    auto* backend = graph.allocation_budget.scheduler_backend(model.sched_backends[0]);
    for (const auto& seam : graph.selected_expert_seams)
        for (auto* tensor : {seam.segment.moe_in, seam.segment.sel_ids, seam.boundary, seam.segment.moe_out})
            if (ggml_backend_sched_get_tensor_backend(graph.sched, tensor) != backend) return false;
    // The gate above deliberately restricts this prototype to one accelerator
    // scheduler span. Subgraph FFN dispatches still contain multiple kernels.
    return graph.selected_expert_seams.empty() || ggml_backend_sched_get_n_splits(graph.sched) == 1;
}

static void glm_begin_selected_expert_graph(graph_build_result& graph)
{
    graph.selected_expert_completed = 0; graph.selected_expert_failed = false;
    if (!graph.selected_expert_seams.empty())
        graph.selected_expert_error_epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
}

static bool glm_finish_selected_expert_graph(glm_model& model, const graph_build_result& graph)
{
    if (graph.selected_expert_failed || graph.selected_expert_completed != graph.selected_expert_seams.size()) return false;
    if (!graph.selected_expert_seams.empty()) {
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != graph.selected_expert_error_epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire)) {
            tsg::set_last_error("GLM selected-expert graph completion failed before logits publication");
            return false;
        }
        ++model.selected_expert_graphs;
        model.selected_expert_calls += graph.selected_expert_completed;
    }
    return true;
}

// Cache columns the attention reads. Padded so decode reuses one graph shape
// for a whole 256-token stretch instead of rebuilding every token.
static int64_t plan_n_kv(const glm_model & m, int64_t p_end)
{
    return std::min<int64_t>(pad_to(p_end, 256), m.n_ctx);
}

/// An input the built graph turned out not to need. The per-device inputs are
/// created for every device a layer maps to, but a device can end up hosting
/// only part of a layer — expert-parallel-only sharding gives a rank the MoE
/// and no attention — and then the scheduler never allocates the inputs that
/// part would have consumed. Writing to one of those asserts inside ggml, so
/// every host-side fill goes through this check.
static inline bool live(const ggml_tensor * t)
{
    return t != nullptr && t->buffer != nullptr;
}

static void set_input_i32(ggml_tensor * t, const int32_t * v, size_t n)
{
    if (live(t)) ggml_backend_tensor_set(t, v, 0, n * sizeof(int32_t));
}

static std::string trace_json_string(const std::string& value)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"';
    return out.str();
}

static uint64_t trace_checked_add(uint64_t a, uint64_t b)
{
    if (a > std::numeric_limits<uint64_t>::max() - b) throw std::runtime_error("GLM trace byte offset overflow");
    return a + b;
}

static uint64_t trace_checked_mul(uint64_t a, uint64_t b)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b) throw std::runtime_error("GLM trace element count overflow");
    return a * b;
}

static void dump_traces(glm_model& model, const graph_build_result* gr, int64_t p0,
                        const int32_t* slot_ids, const int32_t* positions, int sequence_count)
{
    auto* state = model.trace_dump;
    if (!state || gr->traces.empty()) return;
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->next_pass >= state->maximum_passes) return;
    if (gr->nt <= 0 || gr->traces.size() > 4096)
        throw std::runtime_error("GLM trace capture exceeds its positive token or4096-tensor metadata bound");
    if (sequence_count < 0 || (sequence_count > 0 && (!slot_ids || !positions || sequence_count != gr->nt))
        || (sequence_count == 0 && p0 < 0))
        throw std::runtime_error("GLM trace capture lacks actual sequence positions");

    struct plan {
        const trace_entry* entry;
        std::array<int64_t, 4> shape;
        std::vector<int> candidates;
        int axis;
        bool explicit_axis;
        uint64_t source_bytes, elements, bytes;
    };
    std::vector<plan> plans;
    uint64_t total_bytes = 0;
    for (const auto& entry : gr->traces) {
        const auto* tensor = entry.t;
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32)
            throw std::runtime_error("GLM trace dump supports only complete F32/I32 logical values: " + entry.name);
        plan item{}; item.entry = &entry;
        item.source_bytes = ggml_nbytes(tensor);
        for (int d = 0; d < 4; ++d) {
            if (tensor->ne[d] <= 0) throw std::runtime_error("GLM trace tensor has a nonpositive dimension");
            item.shape[d] = tensor->ne[d];
            if (tensor->ne[d] == gr->nt) item.candidates.push_back(d);
        }
        const auto explicit_axis = state->axes.find(entry.tag);
        item.explicit_axis = explicit_axis != state->axes.end();
        if (item.explicit_axis) item.axis = explicit_axis->second;
        else if (item.candidates.size() == 1) item.axis = item.candidates[0];
        else throw std::runtime_error("GLM trace token axis is missing or ambiguous for " + entry.name
            + "; provide explicit TS_GLM_TRACE_DUMP_TOKEN_AXES=tag=0..3");
        if (tensor->ne[item.axis] != gr->nt)
            throw std::runtime_error("GLM trace explicit token axis does not match actual pass width: " + entry.name);
        item.shape[item.axis] = std::min<int64_t>(gr->nt, static_cast<int64_t>(state->rows));
        item.elements = 1;
        uint64_t last_byte = sizeof(float);
        for (int d = 0; d < 4; ++d) {
            item.elements = trace_checked_mul(item.elements, static_cast<uint64_t>(item.shape[d]));
            last_byte = trace_checked_add(last_byte, trace_checked_mul(static_cast<uint64_t>(item.shape[d] - 1), tensor->nb[d]));
        }
        if (last_byte > item.source_bytes) throw std::runtime_error("GLM trace slice exceeds its source stride envelope");
        item.bytes = trace_checked_mul(item.elements, sizeof(float));
        total_bytes = trace_checked_add(total_bytes, item.bytes);
        if (total_bytes > state->maximum_bytes - state->reserved_bytes)
            throw std::runtime_error("GLM trace capture exceeds its configured process-wide byte bound");
        plans.push_back(std::move(item));
    }

    // Reserve the complete pass before creating any filesystem payload. Failed
    // writes keep their incomplete files for diagnosis, so their reserved bytes
    // must remain charged across checked resets and subsequent pass attempts.
    // This conservative lifetime quota also covers failures before the first
    // payload opens; never refund bytes whose cleanup has not been established.
    state->reserved_bytes += total_bytes;
    const uint64_t pass = state->next_pass++;
    std::ostringstream prefix;
    prefix << "pass-" << std::setw(6) << std::setfill('0') << pass;
    const auto marker = state->directory / (prefix.str() + ".incomplete");
    const auto manifest = state->directory / (prefix.str() + ".json");
    auto open_new = [](const std::filesystem::path& path) {
        if (std::filesystem::exists(path)) throw std::runtime_error("GLM trace output exists; refusing overwrite");
        std::ofstream stream;
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.open(path, std::ios::binary | std::ios::out);
        return stream;
    };
    { auto pending = open_new(marker); pending << "Incomplete GLM validation capture; no logits publication is certified.\n"; pending.close(); }
    const auto error_epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
    auto check_download = [&]() {
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != error_epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("GLM trace backend download failed before logits publication");
    };
    check_download();
    std::array<unsigned char, 64 * 1024> transfer;
    const uint32_t endian_probe = 1;
    const bool little_endian = *reinterpret_cast<const unsigned char*>(&endian_probe) == 1;
    std::ostringstream json;
    json << "{\"schema\":\"tensorsharp.glm-trace.v1\",\"pass_id\":" << pass
        << ",\"model_id\":" << model.trace_model_id << ",\"kind\":\""
        << (sequence_count > 0 ? "batched" : gr->kind == 1 ? "mtp" : "trunk") << "\",\"p0\":";
    if (sequence_count > 0) json << "null"; else json << p0;
    json << ",\"nt\":" << gr->nt << ",\"n_kv\":" << gr->n_kv << ",\"slot_id\":" << gr->slot_id
        << ",\"sequences\":[";
    if (sequence_count == 0)
        json << "{\"slot_id\":" << gr->slot_id << ",\"position\":" << p0 << '}';
    for (int i = 0; i < sequence_count; ++i) {
        if (i) json << ',';
        json << "{\"slot_id\":" << slot_ids[i] << ",\"position\":" << positions[i] << '}';
    }
    json << "],\"rows_limit\":" << state->rows << ",\"max_bytes\":" << state->maximum_bytes
        << ",\"max_passes\":" << state->maximum_passes << ",\"total_bytes\":" << total_bytes
        << ",\"endian\":\"" << (little_endian ? "little" : "big") << "\",\"token_axis_overrides\":{";
    bool first_axis = true;
    for (const auto& axis : state->axes) {
        if (!first_axis) json << ',';
        first_axis = false; json << trace_json_string(axis.first) << ':' << axis.second;
    }
    json << "},\"scope\":\"Diagnostic retained-output graph; fusion and liveness may differ. "
        "FNV1a64 is noncryptographic. No original-graph performance, model/input identity, semantic quality or physical-hard-cap acceptance.\",\"traces\":[";
    for (size_t ordinal = 0; ordinal < plans.size(); ++ordinal) {
        const auto& item = plans[ordinal]; const auto& entry = *item.entry; const auto* tensor = entry.t;
        std::ostringstream filename;
        filename << prefix.str() << ".tensor-" << std::setw(4) << std::setfill('0') << ordinal << ".bin";
        const auto final_path = state->directory / filename.str();
        const auto temporary = state->directory / (filename.str() + ".tmp");
        auto stream = open_new(temporary);
        uint64_t hash = 14695981039346656037ULL, written = 0;
        const graph_build_result::selected_expert_seam* independent_seam = nullptr;
        bool check_ids = false, check_weights = false;
        if (!gr->selected_expert_seams.empty()) {
            const auto& last = gr->selected_expert_seams.back();
            if (entry.layer == last.segment.layer && entry.rank == 0) {
                check_ids = entry.tag == "moe_sel";
                check_weights = entry.tag == "moe_w";
                if (check_ids || check_weights) independent_seam = &last;
            }
        }
        if (check_ids && (tensor->type != GGML_TYPE_I32 || item.axis != 1
                || tensor->ne[0] != independent_seam->segment.n_used || tensor->ne[2] != 1 || tensor->ne[3] != 1
                || gr->selected_expert_ids.size() != size_t(tensor->ne[0]) * size_t(gr->nt)))
            throw std::runtime_error("GLM trace I32 independent packed route snapshot has a different shape");
        if (check_weights && (tensor->type != GGML_TYPE_F32 || item.axis != 2
                || tensor->ne[0] != 1 || tensor->ne[1] != independent_seam->segment.n_used || tensor->ne[3] != 1
                || independent_seam->boundary->type != GGML_TYPE_F32 || !ggml_is_contiguous(independent_seam->boundary)
                || ggml_nelements(independent_seam->boundary) != ggml_nelements(tensor)))
            throw std::runtime_error("GLM trace F32 independent packed weight boundary has a different shape");
        // Use the existing64KiB transfer allocation's second half for the
        // independent cont-boundary download; no additional full tensor copy.
        const size_t transfer_capacity = check_weights ? transfer.size() / 2 : transfer.size();
        auto emit = [&](size_t count) {
            if (check_ids) {
                const uint64_t snapshot_bytes = trace_checked_mul(gr->selected_expert_ids.size(), sizeof(int32_t));
                if (trace_checked_add(written, count) > snapshot_bytes
                    || std::memcmp(transfer.data(), reinterpret_cast<const unsigned char*>(gr->selected_expert_ids.data()) + written, count) != 0)
                    throw std::runtime_error("GLM trace strided I32 values differ from the independent packed callback IDs");
            }
            if (check_weights) {
                if (trace_checked_add(written, count) > ggml_nbytes(independent_seam->boundary))
                    throw std::runtime_error("GLM trace packed weight snapshot exceeds its boundary");
                ggml_backend_tensor_get(independent_seam->boundary, transfer.data() + transfer_capacity,
                    static_cast<size_t>(written), count);
                check_download();
                if (std::memcmp(transfer.data(), transfer.data() + transfer_capacity, count) != 0)
                    throw std::runtime_error("GLM trace logical F32 weights differ from the independent contiguous seam boundary");
            }
            for (size_t i = 0; i < count; ++i) hash = (hash ^ transfer[i]) * 1099511628211ULL;
            stream.write(reinterpret_cast<const char*>(transfer.data()), static_cast<std::streamsize>(count));
            written = trace_checked_add(written, count);
        };
        // Preserve the original four-dimensional strides; top-k is often a
        // narrow view with an experts-wide row stride. Pack logical elements
        // in ne0-fastest order instead of copying padding as model values.
        for (int64_t i3 = 0; i3 < item.shape[3]; ++i3)
        for (int64_t i2 = 0; i2 < item.shape[2]; ++i2)
        for (int64_t i1 = 0; i1 < item.shape[1]; ++i1) {
            const uint64_t base = trace_checked_add(trace_checked_mul(i1, tensor->nb[1]),
                trace_checked_add(trace_checked_mul(i2, tensor->nb[2]), trace_checked_mul(i3, tensor->nb[3])));
            int64_t done = 0;
            while (done < item.shape[0]) {
                const size_t count = static_cast<size_t>(std::min<int64_t>(item.shape[0] - done, transfer_capacity / sizeof(float)));
                const size_t bytes = count * sizeof(float);
                if (tensor->nb[0] == sizeof(float)) {
                    const uint64_t offset = trace_checked_add(base, trace_checked_mul(done, tensor->nb[0]));
                    if (trace_checked_add(offset, bytes) > item.source_bytes) throw std::runtime_error("GLM trace row read exceeds its source");
                    ggml_backend_tensor_get(tensor, transfer.data(), static_cast<size_t>(offset), bytes);
                    check_download();
                } else {
                    for (size_t k = 0; k < count; ++k) {
                        const uint64_t offset = trace_checked_add(base, trace_checked_mul(static_cast<uint64_t>(done) + k, tensor->nb[0]));
                        if (trace_checked_add(offset, sizeof(float)) > item.source_bytes) throw std::runtime_error("GLM trace strided element exceeds its source");
                        ggml_backend_tensor_get(tensor, transfer.data() + k * sizeof(float), static_cast<size_t>(offset), sizeof(float));
                        check_download();
                    }
                }
                emit(bytes); done += static_cast<int64_t>(count);
            }
        }
        if (written != item.bytes) throw std::runtime_error("GLM trace packed byte count mismatch");
        stream.close();
        if (std::filesystem::exists(final_path)) throw std::runtime_error("GLM trace binary already exists");
        std::filesystem::rename(temporary, final_path);
        if (ordinal) json << ',';
        json << "{\"ordinal\":" << ordinal << ",\"name\":" << trace_json_string(entry.name)
            << ",\"tag\":" << trace_json_string(entry.tag) << ",\"layer\":" << entry.layer << ",\"rank\":" << entry.rank
            << ",\"type\":\"" << (tensor->type == GGML_TYPE_F32 ? "f32" : "i32") << "\",\"type_code\":" << int(tensor->type)
            << ",\"source_bytes\":" << item.source_bytes << ",\"ne\":[";
        for (int d = 0; d < 4; ++d) { if (d) json << ','; json << tensor->ne[d]; }
        json << "],\"nb\":[";
        for (int d = 0; d < 4; ++d) { if (d) json << ','; json << tensor->nb[d]; }
        json << "],\"candidate_token_axes\":[";
        for (size_t i = 0; i < item.candidates.size(); ++i) { if (i) json << ','; json << item.candidates[i]; }
        json << "],\"token_axis\":" << item.axis << ",\"token_axis_source\":\"" << (item.explicit_axis ? "override" : "unique-extent")
            << "\",\"shape\":[";
        for (int d = 0; d < 4; ++d) { if (d) json << ','; json << item.shape[d]; }
        std::ostringstream digest; digest << std::hex << std::setw(16) << std::setfill('0') << hash;
        json << "],\"slice_start\":0,\"token_count\":" << item.shape[item.axis] << ",\"logical_elements\":" << item.elements
            << ",\"bytes\":" << item.bytes << ",\"file\":" << trace_json_string(filename.str())
            << ",\"content_fnv1a64\":\"" << digest.str() << "\",\"snapshot_checked_elements\":"
            << (independent_seam ? item.elements : 0) << ",\"snapshot_source\":";
        if (!independent_seam) json << "null";
        else {
            const auto* snapshot = check_ids ? independent_seam->segment.sel_ids : independent_seam->boundary;
            json << "{\"kind\":\"" << (check_ids ? "selected_ids_host_snapshot" : "selected_weights_cont_boundary")
                << "\",\"layer\":" << independent_seam->segment.layer << ",\"rank\":0,\"tensor_name\":"
                << trace_json_string(ggml_get_name(snapshot)) << ",\"op\":" << trace_json_string(ggml_op_name(snapshot->op))
                << ",\"ne\":[";
            for (int d = 0; d < 4; ++d) { if (d) json << ','; json << snapshot->ne[d]; }
            json << "],\"nb\":[";
            for (int d = 0; d < 4; ++d) { if (d) json << ','; json << snapshot->nb[d]; }
            json << "],\"source_bytes\":" << ggml_nbytes(snapshot) << '}';
        }
        json << '}';
        if (json.tellp() > std::streampos(4 * 1024 * 1024)) throw std::runtime_error("GLM trace metadata exceeds4MiB");
    }
    json << "]}\n";
    check_download();
    const auto temporary_manifest = state->directory / (prefix.str() + ".json.tmp");
    { auto stream = open_new(temporary_manifest); stream << json.str(); stream.close(); }
    if (std::filesystem::exists(manifest)) throw std::runtime_error("GLM trace manifest already exists");
    std::filesystem::rename(temporary_manifest, manifest);
    if (!std::filesystem::remove(marker)) throw std::runtime_error("GLM trace incomplete marker could not be retired");
    fprintf(stderr, "[glm-trace-dump] pass=%" PRIu64 " model=%" PRIu64 " nt=%" PRId64 " p0=%" PRId64
        " tensors=%zu logical_bytes=%" PRIu64 " manifest=%s (diagnostic graph only)\n",
        pass, model.trace_model_id, gr->nt, p0, plans.size(), total_bytes, manifest.u8string().c_str());
}

/// Print what TS_GLM_TRACE selected, once the graph has run.
static void print_traces(glm_model& model, const graph_build_result * gr, int64_t p0,
                         const int32_t* slot_ids = nullptr, const int32_t* positions = nullptr, int sequence_count = 0)
{
    if (gr->traces.empty()) return;
    if (model.trace_dump) {
        dump_traces(model, gr, p0, slot_ids, positions, sequence_count);
        return; // The bounded dump replaces the old full-span summary download.
    }
    {
        std::vector<float> buf;
        for (const trace_entry & e : gr->traces)
        {
            // A top-k view can have expert-wide strides despite its narrow
            // logical shape. The legacy contiguous summary must not interpret
            // padding as model values. Bounded dumps pack those strides exactly.
            if (!ggml_is_contiguous(e.t)) {
                fprintf(stderr, "[glm-trace] %-20s noncontiguous logical tensor; summary unavailable, use TS_GLM_TRACE_DUMP_DIR\n",
                    e.name.c_str());
                continue;
            }
            const int64_t n = ggml_nelements(e.t);
            buf.resize((size_t) n);
            if (e.t->type == GGML_TYPE_I32)
            {
                std::vector<int32_t> ib((size_t) n);
                ggml_backend_tensor_get(e.t, ib.data(), 0, (size_t) n * sizeof(int32_t));
                int32_t lo = ib.empty() ? 0 : ib[0], hi = lo;
                long long isum = 0;
                for (int64_t i = 0; i < n; i++) { lo = std::min(lo, ib[(size_t) i]); hi = std::max(hi, ib[(size_t) i]); isum += ib[(size_t) i]; }
                fprintf(stderr, "[glm-trace] %-20s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] i32 min=%d max=%d sum=%lld\n",
                        e.name.c_str(), e.t->ne[0], e.t->ne[1], e.t->ne[2], lo, hi, isum);
                continue;
            }
            ggml_backend_tensor_get(e.t, buf.data(), 0, (size_t) n * sizeof(float));
            double sum = 0, asum = 0;
            bool nan = false;
            for (int64_t i = 0; i < n; i++)
            {
                sum += buf[(size_t) i];
                asum += fabs((double) buf[(size_t) i]);
                if (std::isnan(buf[(size_t) i])) nan = true;
            }
            fprintf(stderr, "[glm-trace] %-20s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] asum=%.6f sum=%.6f%s\n",
                    e.name.c_str(), e.t->ne[0], e.t->ne[1], e.t->ne[2], asum, sum, nan ? "  <-- NaN" : "");
        }
    }
}

// Diagnostic filesystem/bounds/download failures are ordinary refusals, not
// exceptions crossing the C/PInvoke boundary. The caller's dispatch guard
// fences the touched state before it can publish logits or advance heads.
static bool try_print_traces(glm_model& model, const graph_build_result* gr, int64_t p0,
    const int32_t* slot_ids = nullptr, const int32_t* positions = nullptr, int sequence_count = 0) noexcept
{
    try {
        print_traces(model, gr, p0, slot_ids, positions, sequence_count);
        return true;
    } catch (const std::exception& error) {
        try { tsg::set_last_error(std::string("GLM trace capture failed before publication: ") + error.what()); }
        catch (...) { std::fputs("GLM trace capture failed; insufficient memory to store native error text.\n", stderr); }
    } catch (...) {
        try { tsg::set_last_error("GLM trace capture failed before publication with a non-standard exception"); }
        catch (...) { std::fputs("GLM trace capture failed; native error text could not be stored.\n", stderr); }
    }
    return false;
}

/// A batched-decode graph is keyed on the exact batch shape: how many tokens,
/// which slot each belongs to, how much of that slot's cache it sees (padded, so
/// the key is stable for 256 steps), and whether it is past the indexer top-k.
static graph_build_result * acquire_batched_graph(glm_model & m, int n, const int32_t * slot_ids,
                                                  const int32_t * positions, bool * out_reused)
{
    std::vector<bd_token> want((size_t) n);
    for (int i = 0; i < n; i++)
    {
        bd_token & B = want[(size_t) i];
        B.slot_id = slot_ids[i];
        B.p = positions[i];
        B.n_kv = plan_n_kv(m, B.p + 1);
        const int64_t n_select = m.hp.indexer_kpool > 0
            ? (int64_t) m.hp.indexer_top_k + m.hp.indexer_kpool - 1
            : (int64_t) m.hp.indexer_top_k;
        B.sparse = (B.p + 1) > n_select;
        if (B.p + 1 > m.n_ctx) return nullptr;
    }

    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        graph_build_result * e = it->get();
        if ((int) e->bd.size() != n || e->nt != n) continue;
        bool same = true;
        for (int i = 0; i < n && same; i++)
            same = e->bd[(size_t) i].slot_id == want[(size_t) i].slot_id
                && e->bd[(size_t) i].n_kv == want[(size_t) i].n_kv
                && e->bd[(size_t) i].sparse == want[(size_t) i].sparse;
        if (!same) continue;
        auto entry = std::move(*it);
        m.graph_cache.erase(it);
        graph_build_result * raw = entry.get();
        m.graph_cache.push_front(std::move(entry));
        for (int i = 0; i < n; i++) raw->bd[(size_t) i].p = want[(size_t) i].p;
        if (out_reused) *out_reused = true;
        return raw;
    }

    size_t nodes_per_layer = 256;
    if (const char * e = getenv("TS_GLM_NODES_PER_LAYER")) { int v = atoi(e); if (v > 0) nodes_per_layer = (size_t) v; }
    // The per-token part of a layer (cache write, scoring, softmax) is built n
    // times over, so the budget grows with the batch.
    const size_t n_nodes = (size_t) m.hp.n_layer * nodes_per_layer * (size_t) std::max(1, n) + 1024;
    ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
                            nullptr, true };

    for (;;)
    {
        auto entry = std::unique_ptr<graph_build_result>(new graph_build_result());
        entry->nt = n;
        entry->n_kv = 0;
        entry->n_out = n;
        entry->want_logits = true;
        entry->slot_id = -1;
        entry->bd = want;
        entry->ctx = ggml_init(gp);
        if (!entry->ctx) return nullptr;
        entry->gf = ggml_new_graph_custom(entry->ctx, n_nodes, false);
        auto budget_types = entry->allocation_budget.prepare(m.sched_bufts, m.sched_backends, m.n_sched_backends, m.n_gpu,
            m.shared_graph_owner || !entry->bd.empty() ? 5 : 2);
        if (!budget_types) return nullptr;
        entry->sched = ggml_backend_sched_new(entry->allocation_budget.scheduler_backends(), budget_types, m.n_sched_backends,
                                              n_nodes, false, m.op_offload);
        if (!entry->sched) return nullptr;

        glm_slot & any = *m.slots.at(slot_ids[0]);
        graph_builder gb(m, *entry, any, n, 0, 0, false);
        if (m.hp.g5n) gb.build_batched_g5n();
        else          gb.build_batched();
        if (!glm_prepare_selected_expert_graph(m, *entry)) return nullptr;

        if (!glm_test_graph_alloc_fault() && ggml_backend_sched_alloc_graph(entry->sched, entry->gf))
        {
            if (!glm_check_selected_expert_assignment(m, *entry)) return nullptr;
            graph_build_result * raw = entry.get();
            m.graph_cache.push_front(std::move(entry));
            while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
            if (out_reused) *out_reused = false;
            return raw;
        }
        entry.reset();   // its compute buffers go back before a slot is freed
        if (!glm_reclaim_for_graph(m, slot_ids, n))
        {
            fprintf(stderr, "[glm] failed to allocate a batched graph for n=%d\n", n);
            return nullptr;
        }
    }
}

/// One decode step for n sequences at once. Every sequence contributes exactly
/// one token; the weights are read once for the whole batch, which is the only
/// reason to do this rather than n separate forwards.
static bool forward_batched_decode(glm_model & m, int n, const int32_t * slot_ids, const int32_t * tokens,
                                   const int32_t * positions, float * logits_out)
{
    if (bd_debug())
    {
        fprintf(stderr, "[glm-bd] enter n=%d", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " (slot %d tok %d pos %d)", slot_ids[i], tokens[i], positions[i]);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    for (int i = 0; i < n; i++)
    {
        auto it = m.slots.find(slot_ids[i]);
        if (it == m.slots.end() || it->second->n_past != positions[i]) return false;
        for (int j = 0; j < i; j++) if (slot_ids[j] == slot_ids[i]) return false;
    }

    bd_log("checks passed\n");
    bool reused = false;
    graph_build_result * gr = acquire_batched_graph(m, n, slot_ids, positions, &reused);
    if (!gr) { bd_log("no graph\n"); return false; }
    glm_begin_selected_expert_graph(*gr);
    bd_log("graph %s\n", reused ? "reused" : "built");

    const bool f16_mask = m.flash_attn;
    std::vector<ggml_fp16_t> m16;
    std::vector<float> m32;
    std::vector<int64_t> idx(1);
    std::vector<int32_t> v32((size_t) n);

    // Each input is set on its own liveness. The token ids are only read on the
    // device that embeds them, but EVERY device with a RoPE layer reads its own
    // positions: skipping a device whose token input was pruned left its
    // positions uninitialised, so on a multi-GPU layer split every layer past
    // the first device rotated q/k by garbage and concurrent GLM-5.2 streams
    // degenerated (glm5next is NoPE, which is why it never showed).
    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (live(gr->inp.tokens[d]))
        {
            for (int i = 0; i < n; i++) v32[(size_t) i] = tokens[i];
            set_input_i32(gr->inp.tokens[d], v32.data(), (size_t) n);
        }
        if (live(gr->inp.pos[d]))
        {
            for (int i = 0; i < n; i++) v32[(size_t) i] = positions[i];
            set_input_i32(gr->inp.pos[d], v32.data(), (size_t) n);
        }
    }

    for (int i = 0; i < n; i++)
    {
        bd_token & B = gr->bd[(size_t) i];
        const int64_t p = positions[i];
        idx[0] = p;
        // Everything at or before this token's position is visible; the padding
        // past it is masked out.
        if (f16_mask)
        {
            m16.assign((size_t) B.n_kv, (ggml_fp16_t) 0xFC00);
            for (int64_t j = 0; j <= p && j < B.n_kv; j++) m16[(size_t) j] = 0;
        }
        else
        {
            m32.assign((size_t) B.n_kv, -INFINITY);
            for (int64_t j = 0; j <= p && j < B.n_kv; j++) m32[(size_t) j] = 0.0f;
        }
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (live(B.kv_idx[d]))
                ggml_backend_tensor_set(B.kv_idx[d], idx.data(), 0, sizeof(int64_t));
            if (live(B.kq_mask[d]))
            {
                if (f16_mask) ggml_backend_tensor_set(B.kq_mask[d], m16.data(), 0, m16.size() * 2);
                else          ggml_backend_tensor_set(B.kq_mask[d], m32.data(), 0, m32.size() * 4);
            }
            if (live(B.lid_mask[d]))
            {
                // The indexer scores in F16 and never sees a padded row.
                if (!f16_mask)
                {
                    m16.assign((size_t) B.n_kv, (ggml_fp16_t) 0xFC00);
                    for (int64_t j = 0; j <= p && j < B.n_kv; j++) m16[(size_t) j] = 0;
                }
                ggml_backend_tensor_set(B.lid_mask[d], m16.data(), 0, (size_t) B.n_kv * 2);
            }
            if (live(B.pool_cells[d]))
            {
                const int64_t r = m.hp.indexer_kpool;
                const int64_t n_pools = B.n_kv / r;
                std::vector<int32_t> pc((size_t) B.n_kv);
                for (int64_t j = 0; j < B.n_kv; j++) pc[(size_t) j] = (int32_t) j;
                ggml_backend_tensor_set(B.pool_cells[d], pc.data(), 0, pc.size() * sizeof(int32_t));

                const int64_t bo_vis = std::min<int64_t>((p + 1) / r, n_pools);
                if (B.pool_bias[d] && B.pool_bias[d]->buffer)
                {
                    if (B.pool_bias[d]->type == GGML_TYPE_F16)
                    {
                        std::vector<uint16_t> pb((size_t) n_pools, 0xFC00);
                        for (int64_t b = 0; b < bo_vis; b++) pb[(size_t) b] = 0;
                        ggml_backend_tensor_set(B.pool_bias[d], pb.data(), 0, pb.size() * 2);
                    }
                    else
                    {
                        std::vector<float> pb((size_t) n_pools, -INFINITY);
                        for (int64_t b = 0; b < bo_vis; b++) pb[(size_t) b] = 0.0f;
                        ggml_backend_tensor_set(B.pool_bias[d], pb.data(), 0, pb.size() * 4);
                    }
                }
                const int64_t ts = (p + 1) / r * r;
                const int64_t n_tail = p + 1 - ts;
                std::vector<int32_t> tc((size_t) r);
                std::vector<float> tv((size_t) r);
                for (int64_t j = 0; j < r; j++)
                {
                    tc[(size_t) j] = (int32_t) std::min<int64_t>(ts + j, B.n_kv - 1);
                    tv[(size_t) j] = j < n_tail ? 0.0f : -INFINITY;
                }
                if (live(B.trail_cells[d]))
                    ggml_backend_tensor_set(B.trail_cells[d], tc.data(), 0, tc.size() * sizeof(int32_t));
                if (live(B.trail_vals[d]))
                    ggml_backend_tensor_set(B.trail_vals[d], tv.data(), 0, tv.size() * sizeof(float));
            }
        }
    }

    bd_log("inputs set\n");
    // Inputs and allocation refusals above are retryable. Once dispatch starts,
    // an error may leave both KV and recurrent state partially advanced. Keep
    // every participating slot fenced until an explicit successful reset.
    struct failed_dispatch_guard {
        glm_model& model;
        graph_build_result* graph;
        ~failed_dispatch_guard() noexcept {
            if (!model.batched_execution.failure_needs_reset()) return;
            try { ggml_backend_sched_synchronize(graph->sched); } catch (...) {}
            for (int i = 0; i < model.batched_execution.attempted_rows(); ++i) {
                auto it = model.slots.find(model.batched_execution.attempted_slot(i));
                if (it != model.slots.end()) it->second->kda_restore_failed = true;
            }
            model.kda_snap.valid = false;
        }
    } dispatch_guard{m, gr};
    if (!m.batched_execution.begin_dispatch(n, slot_ids)) return false;
    if (ggml_backend_sched_graph_compute_async(gr->sched, gr->gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_sched_synchronize(gr->sched);
    if (!glm_finish_selected_expert_graph(m, *gr)) return false;
    glm_test_batched_dispatch_fault(1);
    bd_log("computed\n");
    if (!try_print_traces(m, gr, -1, slot_ids, positions, n)) return false;

    const int64_t vocab = m.hp.n_vocab;
    for (int i = 0; i < n; i++)
    {
        ggml_backend_tensor_get(gr->logits, logits_out + (size_t) i * vocab,
                                (size_t) i * vocab * sizeof(float), (size_t) vocab * sizeof(float));
        m.slots.at(slot_ids[i])->n_past = positions[i] + 1;
        if (i == 0) glm_test_batched_dispatch_fault(2);
    }
    std::int32_t published[MAX_BATCHED_DECODE]{};
    for (int i = 0; i < n; ++i) published[i] = static_cast<std::int32_t>(m.slots.at(slot_ids[i])->n_past);
    // Publish only after backend synchronization, complete logits download,
    // and every slot-head update. A scheduler graph can contain many dispatches.
    if (!m.batched_execution.record_success(n, ggml_graph_n_nodes(gr->gf),
        ggml_backend_sched_get_n_splits(gr->sched), reused, slot_ids, positions, published)) return false;
    bd_log("done\n");
    return true;
}

static graph_build_result * acquire_graph(glm_model & m, glm_slot & slot, int64_t nt, int64_t p0,
                                          int64_t n_out, bool want_logits, bool * out_reused,
                                          bool want_h = false, int kind = 0, int64_t n_ovr = 0)
{
    const int64_t n_kv = plan_n_kv(m, p0 + nt);
    std::vector<int64_t> decode_row_kv;
    if (m.hp.g5n && nt > 1 && nt <= TSG_PRECISION_DECODE_COLUMNS)
        for (int64_t r = 0; r < nt; ++r) decode_row_kv.push_back(plan_n_kv(m, p0 + r + 1));
    // The draft block attends densely, so the indexer's sparsity never applies
    // to it and must not enter its cache key.
    // glm5next selects whole pools plus the query's own trailing pool; below
    // top_k + kpool - 1 resident positions the selection cannot drop anything.
    const int64_t n_select = m.hp.indexer_kpool > 0
        ? (int64_t) m.hp.indexer_top_k + m.hp.indexer_kpool - 1
        : (int64_t) m.hp.indexer_top_k;
    const bool sparse = kind == 0 && (p0 + nt) > n_select;

    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        graph_build_result * e = it->get();
        if (e->nt == nt && e->n_kv == n_kv && e->n_out == n_out && e->want_logits == want_logits &&
            e->sparse == sparse && e->slot_id == slot.id && e->want_h == want_h && e->kind == kind &&
            e->n_ovr == n_ovr && e->decode_row_kv == decode_row_kv)
        {
            auto entry = std::move(*it);
            m.graph_cache.erase(it);
            graph_build_result * raw = entry.get();
            m.graph_cache.push_front(std::move(entry));
            if (out_reused) *out_reused = true;
            return raw;
        }
    }

    auto make_entry = [&]()
    {
        auto p = std::unique_ptr<graph_build_result>(new graph_build_result());
        p->nt = nt;
        p->n_kv = n_kv;
        p->decode_row_kv = decode_row_kv;
        p->n_out = n_out;
        p->sparse = sparse;
        p->want_logits = want_logits;
        p->want_h = want_h;
        p->kind = kind;
        p->slot_id = slot.id;
        p->n_ovr = n_ovr;
        return p;
    };
    auto entry = make_entry();

    // Node budget. A GLM layer costs ~110 nodes per rank (MoE alone is ~40, the
    // DSA indexer another ~20 on a full layer), so 256 per layer per rank leaves
    // room for the scheduler's own copies without making the hash set
    // pointlessly large.
    size_t nodes_per_layer = 256;
    if (m.hp.g5n && nt > 1 && nt <= TSG_PRECISION_DECODE_COLUMNS)
        nodes_per_layer *= (size_t) nt;
    if (const char * e = getenv("TS_GLM_NODES_PER_LAYER")) { int v = atoi(e); if (v > 0) nodes_per_layer = (size_t) v; }
    // The draft block is one layer; give it the same per-layer budget plus the
    // fixed 1024 so its (much smaller) graph never has to share the trunk's.
    const size_t graph_layers = kind == 0 ? (size_t) m.hp.n_layer : 1;

    // The fast path is intentionally narrow. Other graph kinds/configurations
    // continue below through the mature multi-backend scheduler implementation.
    const bool use_fused_tp = m.tp_fused && kind == 0 && !want_h;
    if (use_fused_tp)
    {
        const size_t rank_nodes = graph_layers * nodes_per_layer + 1024;
        bool rank_out_of_memory = false;

        auto build_rank = [&](graph_build_result & rg, int rank) -> bool
        {
            rg.nt = nt;
            rg.n_kv = n_kv;
            rg.n_out = n_out;
            rg.sparse = sparse;
            rg.want_logits = rank == 0 && want_logits;
            rg.want_h = false;
            rg.kind = kind;
            rg.slot_id = slot.id;
            rg.n_ovr = n_ovr;
            rg.tp_fused = true;
            rg.tp_rank = rank;

            ggml_init_params rp = {
                ggml_tensor_overhead() * rank_nodes + ggml_graph_overhead_custom(rank_nodes, false),
                nullptr, true
            };
            rg.ctx = ggml_init(rp);
            if (!rg.ctx) return false;
            rg.gf = ggml_new_graph_custom(rg.ctx, rank_nodes, false);
            if (!rg.gf) return false;

            graph_builder gb(m, rg, slot, nt, p0, n_kv, sparse);
            gb.build();

            rg.tp_plan.graph = rg.gf;
            if (!tsg::tp_plan_segments(rg.tp_plan, rg.tp_boundary))
            {
                fprintf(stderr, "[glm] failed to segment rank %d TP graph\n", rank);
                return false;
            }

            ggml_backend_t backend = m.sched_backends[m.rank_device(rank)];
            const int nn = ggml_graph_n_nodes(rg.gf);
            for (int i = 0; i < nn; i++)
            {
                if (!ggml_backend_supports_op(backend, rg.gf->nodes[i]))
                {
                    fprintf(stderr, "[glm] rank %d backend cannot execute TP graph node %s (%s)\n",
                            rank, rg.gf->nodes[i]->name, ggml_op_name(rg.gf->nodes[i]->op));
                    return false;
                }
            }

            rg.galloc = tsg::graph_budget_gallocr_new(ggml_backend_get_default_buffer_type(backend), m.rank_device(rank));
            if (!rg.galloc || glm_test_graph_alloc_fault() || !tsg::graph_budget_gallocr_alloc_graph(rg.galloc, rg.gf))
            {
                fprintf(stderr, "[glm] failed to allocate rank %d TP graph for nt=%" PRId64
                                " n_kv=%" PRId64 "\n", rank, nt, n_kv);
                rank_out_of_memory = rg.galloc != nullptr;
                return false;
            }
            return true;
        };

        for (;;)
        {
            rank_out_of_memory = false;
            bool built = build_rank(*entry, 0);
            for (int rank = 1; built && rank < m.tp; rank++)
            {
                auto peer = std::unique_ptr<graph_build_result>(new graph_build_result());
                built = build_rank(*peer, rank);
                entry->tp_peers.push_back(std::move(peer));
            }
            if (built)
            {
                const size_t n_seg = entry->tp_plan.seg_end.size();
                for (const auto & peer : entry->tp_peers)
                    if (peer->tp_plan.seg_end.size() != n_seg) built = false;
            }

            if (built)
            {
                graph_build_result * raw = entry.get();
                m.graph_cache.push_front(std::move(entry));
                while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
                if (out_reused) *out_reused = false;
                return raw;
            }
            // Memory, not a capability: free a retained slot and build the
            // ranks again. With none left, fall through as before.
            if (!rank_out_of_memory) break;
            entry = make_entry();   // the partial ranks' buffers go back first
            const int32_t reads[1] = { slot.id };
            if (!glm_reclaim_for_graph(m, reads, 1)) break;
        }

        // A backend capability can differ from the load-time probes. Fall back
        // once for the model instead of failing the request or rebuilding the
        // same unusable rank graphs for every cache shape.
        fprintf(stderr, "[glm] segmented TP graph is unavailable; switching to combined scheduler fallback\n");
        m.tp_fused = false;
        m.tp_comm.reset();
        m.graph_cache.clear();
        entry = make_entry();
    }

    const size_t n_nodes = graph_layers * nodes_per_layer * (size_t) std::max(1, m.tp) + 1024;
    ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
                            nullptr, true };
    for (;;)
    {
        if (!entry) entry = make_entry();
        entry->ctx = ggml_init(gp);
        if (!entry->ctx) return nullptr;
        entry->gf = ggml_new_graph_custom(entry->ctx, n_nodes, false);

        auto budget_types = entry->allocation_budget.prepare(m.sched_bufts, m.sched_backends, m.n_sched_backends, m.n_gpu,
            m.shared_graph_owner || !entry->bd.empty() ? 5 : 2);
        if (!budget_types) return nullptr;
        entry->sched = ggml_backend_sched_new(entry->allocation_budget.scheduler_backends(), budget_types, m.n_sched_backends,
                                              n_nodes, false, m.op_offload);
        if (!entry->sched) return nullptr;

        graph_builder gb(m, *entry, slot, nt, p0, n_kv, sparse);
        if (kind == 0) gb.build();
        else           gb.build_mtp();
        if (!glm_prepare_selected_expert_graph(m, *entry)) return nullptr;

        // Allocate once, here, and never reset: the allocation (and, on CUDA, the
        // captured graph) is what makes a cached entry worth keeping. Note this
        // must not be preceded by ggml_backend_sched_reserve, which resets the
        // scheduler and would drop the per-device pins build() just set.
        if (!glm_test_graph_alloc_fault() && ggml_backend_sched_alloc_graph(entry->sched, entry->gf))
        {
            if (!glm_check_selected_expert_assignment(m, *entry)) return nullptr;
            graph_build_result * raw = entry.get();
            m.graph_cache.push_front(std::move(entry));
            while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
            if (out_reused) *out_reused = false;
            return raw;
        }
        entry.reset();   // its compute buffers go back before a slot is freed
        const int32_t reads[1] = { slot.id };
        if (!glm_reclaim_for_graph(m, reads, 1))
        {
            fprintf(stderr, "[glm] failed to allocate a graph for nt=%" PRId64 " n_kv=%" PRId64 "\n", nt, n_kv);
            return nullptr;
        }
    }
}

static graph_build_result * fused_rank_graph(graph_build_result & root, int rank)
{
    return rank == 0 ? &root : root.tp_peers[(size_t) rank - 1].get();
}

static bool glm_tp_host_reduce(glm_model & m, ggml_tensor ** tensors)
{
    const int ranks = m.tp;
    const int64_t count = ggml_nelements(tensors[0]);
    if (count <= 0) return true;
    const size_t bytes = (size_t) count * sizeof(float);
    std::vector<float *> ptrs((size_t) ranks, nullptr);
    for (int r = 0; r < ranks; r++)
    {
        m.tp_host_stage[r].resize((size_t) count);
        ptrs[(size_t) r] = m.tp_host_stage[r].data();
    }

    auto download = [&](int rank)
    {
        ggml_backend_t backend = m.backends[m.rank_device(rank)];
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensors[rank], ptrs[(size_t) rank], 0, bytes);
    };
    std::vector<std::thread> workers;
    workers.reserve((size_t) ranks - 1);
    for (int r = 1; r < ranks; r++) workers.emplace_back(download, r);
    download(0);
    for (auto & worker : workers) worker.join();

    tsg::tp_host_allreduce_mt(ptrs.data(), ranks, count);

    workers.clear();
    auto upload = [&](int rank)
    {
        ggml_backend_tensor_set(tensors[rank], ptrs[(size_t) rank], 0, bytes);
    };
    for (int r = 1; r < ranks; r++) workers.emplace_back(upload, r);
    upload(0);
    for (auto & worker : workers) worker.join();
    return true;
}

/// Submit each rank's next local segment before waiting on any of them. The
/// backend collective consumes the resulting partial tensors in place; after
/// it completes, every rank resumes from the same reduced residual stream.
static bool execute_fused_tp(glm_model & m, graph_build_result & root)
{
    if (!root.tp_fused || m.tp < 2 || root.tp_peers.size() + 1 != (size_t) m.tp) return false;
    const size_t n_seg = root.tp_plan.seg_end.size();
    std::vector<int> begin((size_t) m.tp, 0);
    std::vector<ggml_tensor *> partial((size_t) m.tp, nullptr);

    for (size_t s = 0; s < n_seg; s++)
    {
        for (int rank = 0; rank < m.tp; rank++)
        {
            graph_build_result * rg = fused_rank_graph(root, rank);
            if (!rg || rg->tp_plan.seg_end.size() != n_seg) return false;
            const int end = rg->tp_plan.seg_end[s];
            const int start = begin[(size_t) rank];
            begin[(size_t) rank] = end;
            if (end <= start) continue;
            ggml_cgraph view = ggml_graph_view(rg->gf, start, end);
            ggml_backend_t backend = m.sched_backends[m.rank_device(rank)];
            if (ggml_backend_graph_compute_async(backend, &view) != GGML_STATUS_SUCCESS)
            {
                fprintf(stderr, "[glm] rank %d TP segment %zu compute failed\n", rank, s);
                return false;
            }
        }

        if (s + 1 == n_seg) continue;
        for (int rank = 0; rank < m.tp; rank++)
        {
            graph_build_result * rg = fused_rank_graph(root, rank);
            if (s >= rg->tp_plan.ar_tensor.size()) return false;
            partial[(size_t) rank] = rg->tp_plan.ar_tensor[s];
            if (!partial[(size_t) rank] || partial[(size_t) rank]->type != GGML_TYPE_F32 ||
                ggml_nelements(partial[(size_t) rank]) != ggml_nelements(partial[0]))
            {
                fprintf(stderr, "[glm] invalid rank %d TP reduction tensor at segment %zu\n", rank, s);
                return false;
            }
        }

        bool device = m.tp_comm && m.tp_comm->allreduce(partial.data());
        if (!device)
        {
            // A communicator can decline an unsupported payload. Stop probing
            // it after the first refusal and use the correct host path for the
            // remainder of this model's lifetime.
            m.tp_comm.reset();
            if (!glm_tp_host_reduce(m, partial.data())) return false;
        }
        if (!m.tp_comm_reported)
        {
            fprintf(stderr, "[glm] segmented TP reductions active via %s\n",
                    device ? "backend device collective" : "host staging");
            m.tp_comm_reported = true;
        }
    }

    for (int rank = 0; rank < m.tp; rank++)
        ggml_backend_synchronize(m.backends[m.rank_device(rank)]);
    return true;
}

/// One trunk ubatch.
///
/// `h_out`, when non-null, additionally receives the post-final-norm hidden
/// state of EVERY row (nt * n_embd floats) — llama.cpp's `h_nextn`, the input
/// the NextN draft head consumes. `all_logits_rows` runs the LM head on every
/// row instead of only the last, which is what makes one verify pass over a
/// speculative window cost one trunk forward instead of K+1 of them.
static bool forward_ubatch(glm_model & m, const int32_t * tokens, int64_t nt, bool want_logits, float * logits_out,
                           float * h_out = nullptr, bool all_logits_rows = false,
                           const float * ovr_rows = nullptr, const int64_t * ovr_idx = nullptr, int64_t n_ovr = 0)
{
    glm_slot & slot = *m.active_slot;
    const int64_t p0 = slot.n_past;
    if (p0 + nt > m.n_ctx)
    {
        fprintf(stderr, "[glm] context overflow: %" PRId64 " + %" PRId64 " > %d\n", p0, nt, m.n_ctx);
        return false;
    }

    const bool want_h = h_out != nullptr;
    const int64_t n_out = (want_logits && all_logits_rows) ? nt : 1;
    bool reused = false;
    graph_build_result * gr = acquire_graph(m, slot, nt, p0, n_out, want_logits, &reused, want_h, /*kind=*/0, n_ovr);
    if (!gr) return false;
    glm_begin_selected_expert_graph(*gr);

    const int64_t n_kv = gr->n_kv;
    std::vector<graph_build_result *> exec_graphs;
    exec_graphs.push_back(gr);
    if (gr->tp_fused)
        for (auto & peer : gr->tp_peers) exec_graphs.push_back(peer.get());

    m.h_tokens.assign(tokens, tokens + nt);
    m.h_pos.resize((size_t) nt);
    m.h_kv_idxs.resize((size_t) nt);
    for (int64_t i = 0; i < nt; i++)
    {
        m.h_pos[(size_t) i] = (int32_t) (p0 + i);
        m.h_kv_idxs[(size_t) i] = p0 + i;
    }
    m.h_out_ids.resize((size_t) n_out);
    for (int64_t i = 0; i < n_out; i++)
        m.h_out_ids[(size_t) i] = (int32_t) (n_out == 1 ? nt - 1 : i);

    // Causal mask over the padded cache window.
    const bool f16_mask = m.flash_attn;
    if (f16_mask) m.h_mask_f16.assign((size_t) (n_kv * nt), 0);
    else          m.h_mask_f32.assign((size_t) (n_kv * nt), 0.0f);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t last = p0 + t;
        for (int64_t j = 0; j < n_kv; j++)
        {
            const bool visible = j <= last;
            if (f16_mask) m.h_mask_f16[(size_t) (t * n_kv + j)] = visible ? 0 : 0xFC00 /* -inf */;
            else          m.h_mask_f32[(size_t) (t * n_kv + j)] = visible ? 0.0f : -INFINITY;
        }
    }

    for (graph_build_result * xgr : exec_graphs)
    for (int d = 0; d <= m.n_gpu; d++)
    {
        set_input_i32(xgr->inp.tokens[d], m.h_tokens.data(), (size_t) nt);
        set_input_i32(xgr->inp.pos[d], m.h_pos.data(), (size_t) nt);
        if (xgr->inp.kv_idxs[d])
            if (live(xgr->inp.kv_idxs[d]))
                ggml_backend_tensor_set(xgr->inp.kv_idxs[d], m.h_kv_idxs.data(), 0, (size_t) nt * sizeof(int64_t));
        if (xgr->inp.kq_mask[d])
        {
            if (live(xgr->inp.kq_mask[d]))
            {
                if (f16_mask) ggml_backend_tensor_set(xgr->inp.kq_mask[d], m.h_mask_f16.data(), 0, m.h_mask_f16.size() * 2);
                else          ggml_backend_tensor_set(xgr->inp.kq_mask[d], m.h_mask_f32.data(), 0, m.h_mask_f32.size() * 4);
            }
        }
        if (xgr->inp.lid_mask[d])
        {
            // The indexer always wants an F16 mask, whatever the attention uses.
            if (!f16_mask)
            {
                m.h_mask_f16.resize((size_t) (n_kv * nt));
                for (size_t i = 0; i < m.h_mask_f16.size(); i++)
                    m.h_mask_f16[i] = m.h_mask_f32[i] == 0.0f ? 0 : 0xFC00;
            }
            if (live(xgr->inp.lid_mask[d]))
                ggml_backend_tensor_set(xgr->inp.lid_mask[d], m.h_mask_f16.data(), 0, m.h_mask_f16.size() * 2);
        }
    }
    for (graph_build_result * xgr : exec_graphs)
        set_input_i32(xgr->inp.out_ids, m.h_out_ids.data(), (size_t) n_out);

    if (n_ovr > 0)
    {
        for (graph_build_result * xgr : exec_graphs)
        {
            if (live(xgr->inp.embd_rows))
                ggml_backend_tensor_set(xgr->inp.embd_rows, ovr_rows, 0,
                                        (size_t) n_ovr * m.hp.n_embd * sizeof(float));
            if (live(xgr->inp.embd_idx))
                ggml_backend_tensor_set(xgr->inp.embd_idx, ovr_idx, 0, (size_t) n_ovr * sizeof(int64_t));
        }
    }

    // glm5next pooled-indexer inputs. Pools are position-aligned windows of
    // kpool cells over the padded cache window (256 % kpool == 0, so the map is
    // the identity); a pool is scoreable for a query only when its LAST member
    // is visible, which also drops the query's own trailing pool - those cells
    // ride in through trail_cells/trail_vals instead, unconditionally.
    if (m.hp.g5n && gr->sparse)
    {
        const int64_t r = m.hp.indexer_kpool;
        const int64_t n_pools = n_kv / r;
        std::vector<int32_t> pcells((size_t) (r * n_pools));
        for (int64_t j = 0; j < r * n_pools; j++) pcells[(size_t) j] = (int32_t) j;

        std::vector<float> pbias_f32;
        std::vector<uint16_t> pbias_f16;
        bool want_f16 = false, want_f32 = false;
        for (graph_build_result * xgr : exec_graphs)
            for (int d = 0; d <= m.n_gpu; d++)
                if (live(xgr->inp.pool_bias[d]))
                    (xgr->inp.pool_bias[d]->type == GGML_TYPE_F16 ? want_f16 : want_f32) = true;
        if (want_f16) pbias_f16.assign((size_t) (n_pools * nt), 0xFC00);
        if (want_f32) pbias_f32.assign((size_t) (n_pools * nt), -INFINITY);
        for (int64_t t = 0; t < nt; t++)
        {
            const int64_t bo_vis = std::min<int64_t>((p0 + t + 1) / r, n_pools);
            if (want_f16) for (int64_t b = 0; b < bo_vis; b++) pbias_f16[(size_t) (t * n_pools + b)] = 0;
            if (want_f32) for (int64_t b = 0; b < bo_vis; b++) pbias_f32[(size_t) (t * n_pools + b)] = 0.0f;
        }

        std::vector<int32_t> tcells((size_t) (r * nt));
        std::vector<float> tvals((size_t) (r * nt));
        for (int64_t t = 0; t < nt; t++)
        {
            const int64_t q = p0 + t;
            const int64_t ts = (q + 1) / r * r;
            const int64_t n_tail = q + 1 - ts;
            for (int64_t j = 0; j < r; j++)
            {
                tcells[(size_t) (t * r + j)] = (int32_t) std::min<int64_t>(ts + j, n_kv - 1);
                tvals[(size_t) (t * r + j)] = j < n_tail ? 0.0f : -INFINITY;
            }
        }

        for (graph_build_result * xgr : exec_graphs)
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (live(xgr->inp.pool_cells[d]))
                ggml_backend_tensor_set(xgr->inp.pool_cells[d], pcells.data(), 0, pcells.size() * sizeof(int32_t));
            if (live(xgr->inp.pool_bias[d]))
            {
                if (xgr->inp.pool_bias[d]->type == GGML_TYPE_F16)
                    ggml_backend_tensor_set(xgr->inp.pool_bias[d], pbias_f16.data(), 0, pbias_f16.size() * 2);
                else
                    ggml_backend_tensor_set(xgr->inp.pool_bias[d], pbias_f32.data(), 0, pbias_f32.size() * 4);
            }
            if (live(xgr->inp.trail_cells[d]))
                ggml_backend_tensor_set(xgr->inp.trail_cells[d], tcells.data(), 0, tcells.size() * sizeof(int32_t));
            if (live(xgr->inp.trail_vals[d]))
                ggml_backend_tensor_set(xgr->inp.trail_vals[d], tvals.data(), 0, tvals.size() * sizeof(float));
        }
    }

    // _async, not the synchronous entry point: the graph is already allocated
    // above (which the synchronous one asserts against), and the explicit
    // synchronize below is where the logits read waits.
    struct selected_failure_guard {
        glm_model& model; glm_slot& slot; graph_build_result& graph;
        bool armed; bool complete = false;
        ~selected_failure_guard() noexcept {
            if (!armed || complete) return;
            try { ggml_backend_sched_synchronize(graph.sched); } catch (...) {}
            slot.kda_restore_failed = true; model.kda_snap.valid = false;
        }
    } selected_guard{m, slot, *gr, !gr->selected_expert_seams.empty() || m.trace_dump != nullptr || !gr->traces.empty()};
    if (gr->tp_fused)
    {
        if (!execute_fused_tp(m, *gr))
        {
            fprintf(stderr, "[glm] segmented TP graph compute failed\n");
            return false;
        }
    }
    else if (ggml_backend_sched_graph_compute_async(gr->sched, gr->gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[glm] graph compute failed\n");
        return false;
    }
    if (!gr->tp_fused) ggml_backend_sched_synchronize(gr->sched);
    if (!glm_finish_selected_expert_graph(m, *gr)) return false;

    if (!try_print_traces(m, gr, p0)) return false;

    if (want_logits && logits_out && gr->logits)
        ggml_backend_tensor_get(gr->logits, logits_out, 0, (size_t) n_out * m.hp.n_vocab * sizeof(float));
    if (h_out && gr->h_nextn)
        ggml_backend_tensor_get(gr->h_nextn, h_out, 0, (size_t) nt * m.hp.n_embd * sizeof(float));
    slot.n_past += nt;
    selected_guard.complete = true;
    return true;
}

/// One NextN/MTP draft-block pass over `nt` tokens at `start_pos`.
///
/// `h_in` is nt rows of n_embd: row k is the hidden state of the token PRECEDING
/// tokens[k] — the trunk's for a catch-up replay, the draft block's own for a
/// chained draft step. Writes this window's MLA rows into the draft block's own
/// cache (never the trunk's), fills `h_out` with one row per token, and, when
/// `want_logits`, the LAST row's logits.
///
/// The draft block's cache needs no rollback: a catch-up always rewrites from
/// the verified position forward, so speculative rows are overwritten before
/// anything reads them.
static bool mtp_forward(glm_model & m, const int32_t * tokens, int64_t nt, const float * h_in,
                        int64_t start_pos, bool want_logits, float * logits_out, float * h_out)
{
    if (!m.has_mtp)
    {
        fprintf(stderr, "[glm] MTP draft step requested but no NextN block is loaded\n");
        return false;
    }
    glm_slot & slot = *m.active_slot;
    if (start_pos < 0 || start_pos + nt > m.n_ctx)
    {
        fprintf(stderr, "[glm] MTP window [%" PRId64 ", %" PRId64 ") is outside the context (%d)\n",
                start_pos, start_pos + nt, m.n_ctx);
        return false;
    }

    bool reused = false;
    graph_build_result * gr = acquire_graph(m, slot, nt, start_pos, /*n_out=*/1, want_logits, &reused,
                                            /*want_h=*/true, /*kind=*/1);
    if (!gr) return false;

    const int64_t n_kv = gr->n_kv;

    m.h_tokens.assign(tokens, tokens + nt);
    m.h_pos.resize((size_t) nt);
    m.h_kv_idxs.resize((size_t) nt);
    for (int64_t i = 0; i < nt; i++)
    {
        m.h_pos[(size_t) i] = (int32_t) (start_pos + i);
        m.h_kv_idxs[(size_t) i] = start_pos + i;
    }
    m.h_out_ids.assign(1, (int32_t) (nt - 1));

    const bool f16_mask = m.flash_attn;
    if (f16_mask) m.h_mask_f16.assign((size_t) (n_kv * nt), 0);
    else          m.h_mask_f32.assign((size_t) (n_kv * nt), 0.0f);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t last = start_pos + t;
        for (int64_t j = 0; j < n_kv; j++)
        {
            const bool visible = j <= last;
            if (f16_mask) m.h_mask_f16[(size_t) (t * n_kv + j)] = visible ? 0 : 0xFC00 /* -inf */;
            else          m.h_mask_f32[(size_t) (t * n_kv + j)] = visible ? 0.0f : -INFINITY;
        }
    }

    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (live(gr->inp.tokens[d])) set_input_i32(gr->inp.tokens[d], m.h_tokens.data(), (size_t) nt);
        if (live(gr->inp.pos[d])) set_input_i32(gr->inp.pos[d], m.h_pos.data(), (size_t) nt);
        if (live(gr->inp.kv_idxs[d]))
            ggml_backend_tensor_set(gr->inp.kv_idxs[d], m.h_kv_idxs.data(), 0, (size_t) nt * sizeof(int64_t));
        if (live(gr->inp.kq_mask[d]))
        {
            if (f16_mask) ggml_backend_tensor_set(gr->inp.kq_mask[d], m.h_mask_f16.data(), 0, m.h_mask_f16.size() * 2);
            else          ggml_backend_tensor_set(gr->inp.kq_mask[d], m.h_mask_f32.data(), 0, m.h_mask_f32.size() * 4);
        }
        if (live(gr->inp.h_in[d]))
            ggml_backend_tensor_set(gr->inp.h_in[d], h_in, 0, (size_t) nt * m.hp.n_embd * sizeof(float));
    }
    if (want_logits) set_input_i32(gr->inp.out_ids, m.h_out_ids.data(), 1);

    if (ggml_backend_sched_graph_compute_async(gr->sched, gr->gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[glm] MTP graph compute failed\n");
        return false;
    }
    ggml_backend_sched_synchronize(gr->sched);

    if (want_logits && logits_out && gr->logits)
        ggml_backend_tensor_get(gr->logits, logits_out, 0, (size_t) m.hp.n_vocab * sizeof(float));
    if (h_out && gr->h_nextn)
        ggml_backend_tensor_get(gr->h_nextn, h_out, 0, (size_t) nt * m.hp.n_embd * sizeof(float));
    return true;
}

} // namespace tsg_glm

// ---------------------------------------------------------------------------
// exported API
// ---------------------------------------------------------------------------

using namespace tsg_glm;

// Version 1 accounts native GLM weights/constants/slots and cached schedulers.
TSG_EXPORT int TSGgml_GlmUnifiedBudgetVersion() { return 2; }
TSG_EXPORT int TSGgml_GlmSelectedExpertAdapterVersion() { return 7; }
TSG_EXPORT int TSGgml_GlmTrimMappedResidency(void* handle)
{
    auto* m = static_cast<glm_model*>(handle);
    if (!m) return 0;
    try {
        for (int d = 0; d < m->n_backends; ++d) ggml_backend_synchronize(m->backends[d]);
        bool success = true;
        for (auto& mapping : m->mmap_owners) if (mapping) success = mapping->trim_residency() && success;
        return success ? 1 : 0;
    } catch (...) { return 0; }
}
TSG_EXPORT int TSGgml_GlmSelectedExpertGroupCount(void* handle)
{
    const auto* m = static_cast<const glm_model*>(handle);
    return m ? m->selected_expert_groups : 0;
}
TSG_EXPORT int TSGgml_GlmSelectedExpertMinimumBudget(void* handle, int rows, int64_t* bytes)
{
    auto* m = static_cast<glm_model*>(handle);
    if (bytes) *bytes = 0;
    if (!m || !bytes || rows < 1 || rows > 3 || !m->hp.g5n || m->tp != 1 || m->n_gpu != 1
        || m->has_mtp || m->selected_expert_groups <= 0 || m->selected_expert_seams || !m->graph_cache.empty()) return 0;
    if (m->hp.n_expert_used < 1 || m->hp.n_expert_used > 64
        || !tsg::host_moe_cache_prefill_fits(m->n_ubatch, m->hp.n_expert_used, true)) return 0;
    for (const auto& slot : m->slots) if (slot.second->n_past != 0) return 0;
    try {
        std::vector<tsg::HostMoePhaseGroupMinimum> groups;
        groups.reserve(static_cast<size_t>(m->selected_expert_groups));
        for (int il = 0; il < m->hp.n_layer; ++il) {
            const auto& layer = m->layers[il];
            if (!layer.cpu_moe || !layer.w[0].ffn_gate_exps) continue;
            const auto segment = selected_expert_segment(*m, il, rows, rows > 1);
            if (segment.n_ff != m->hp.n_ff_exp)
                throw std::runtime_error("GLM pooled groups require their declared uniform expert geometry");
            const auto key = tsg::host_moe_expert_group_key(segment);
            if (std::any_of(groups.begin(), groups.end(), [&](const auto& group) { return group.key == key; })) continue;
            size_t narrow = 0, wide = 0;
            if (!tsg::host_moe_expert_cache_minimum_phase_plan(segment, rows, m->n_ubatch, m->backends[0], narrow, wide))
                throw std::runtime_error("GLM compact expert layout or exact-width graph cannot be planned");
            groups.push_back({key, narrow, wide});
        }
        // Only one serialized helper is wide at a time. Its whole union and
        // the other groups' narrow floors define the measured phase minimum.
        // Actual inactive owners are physically evicted before any credit reuse.
        tsg::HostMoePhaseQuotaPlan plan(std::move(groups));
        if (plan.size() != static_cast<size_t>(m->selected_expert_groups))
            throw std::runtime_error("GLM group identities disagree with the loaded trunk");
        const size_t total = plan.minimum();
        auto published = plan.groups(); // Any metadata allocation precedes model publication.
        m->selected_expert_group_minima.swap(published);
        m->selected_expert_minimum_bytes = total; m->selected_expert_planned_rows = rows;
        *bytes = static_cast<int64_t>(total);
        return 1;
    } catch (const std::exception& error) {
        tsg::set_last_error(std::string("GLM minimum expert planning failed: ") + error.what());
    } catch (...) { tsg::set_last_error("GLM minimum expert planning failed with a non-standard exception"); }
    return 0;
}
TSG_EXPORT int TSGgml_GlmSelectedExpertExecutionStats(void* handle, uint64_t* graphs, uint64_t* calls)
{
    const auto* m = static_cast<const glm_model*>(handle);
    if (!m || !graphs || !calls) return 0;
    *graphs = m->selected_expert_graphs; *calls = m->selected_expert_calls;
    return 1;
}
TSG_EXPORT int TSGgml_GlmConfigureSelectedExpertSeams(void* handle, int enabled)
{
    auto* m = static_cast<glm_model*>(handle);
    if (!m || (enabled != 0 && enabled != 1) || !m->hp.g5n || m->tp != 1 || m->n_gpu != 1
        || m->has_mtp || m->selected_expert_groups <= 0 || m->selected_expert_ranges.empty()) return 0;
    if (enabled && (m->hp.n_expert_used < 1 || m->hp.n_expert_used > 64
        || !tsg::host_moe_cache_prefill_fits(m->n_ubatch, m->hp.n_expert_used, true))) return 0;
    if (enabled && !tsg::host_moe_expert_cache_scope_matches_plan(m->selected_expert_minimum_bytes,
        m->selected_expert_groups, m->n_ubatch, m->selected_expert_planned_rows)) {
        tsg::set_last_error("GLM expert scope cannot protect the measured admitted union and graph variants before dispatch");
        return 0;
    }
    if (m->selected_expert_seams == (enabled != 0)) return 1;
    if (!m->graph_cache.empty()) return 0;
    for (const auto& slot : m->slots) if (slot.second->n_past != 0) return 0;
    try {
        for (int d = 0; d < m->n_backends; ++d) ggml_backend_synchronize(m->backends[d]);
        if (enabled) {
            int64_t used, budget, hits, misses, calls;
            tsg::host_moe_expert_cache_stats(&used, &budget, &hits, &misses, &calls);
            if (budget <= 0) return 0;
            m->registered_expert_ranges.reserve(m->selected_expert_ranges.size());
            for (const auto& range : m->selected_expert_ranges) {
                if (!tsg::register_host_file_source(range.pointer, range.bytes, range.path.c_str(), range.offset))
                    throw std::runtime_error("GLM expert source already registered or invalid");
                m->registered_expert_ranges.push_back(range.pointer);
            }
            if (!tsg::host_moe_expert_cache_install_phase_plan(m->selected_expert_group_minima,
                m->n_ubatch, m->selected_expert_planned_rows, m->backends[0]))
                throw std::runtime_error("GLM measured group floors cannot be installed in this empty exclusive scope");
        } else {
            tsg::host_moe_expert_cache_release_backend(m->backends[0]);
            for (const void* pointer : m->registered_expert_ranges) tsg::host_moe_expert_cache_on_drop(pointer);
            m->registered_expert_ranges.clear();
        }
        m->selected_expert_seams = enabled != 0;
        return 1;
    } catch (const std::exception& error) {
        tsg::set_last_error(std::string("GLM selected-expert configuration failed: ") + error.what());
    } catch (...) { tsg::set_last_error("GLM selected-expert configuration failed with a non-standard exception"); }
    for (const void* pointer : m->registered_expert_ranges) tsg::host_moe_expert_cache_on_drop(pointer);
    m->registered_expert_ranges.clear();
    return 0;
}

TSG_EXPORT void * TSGgml_GlmLoadModel(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch,
                                      int n_threads, int n_cpu_moe, const char * backend_name, int tp,
                                      int ctx_is_hard_limit, int load_mtp)
{
    // A stale error from an earlier op must not be reported as this load's reason.
    tsg::clear_last_error();
    try
    {
        return glm_load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, n_cpu_moe, backend_name, tp,
                        ctx_is_hard_limit != 0, load_mtp != 0);
    }
    catch (const std::exception & e)
    {
        tsg::report_load_refusal("[glm] load failed: %s\n", e.what());
        return nullptr;
    }
    catch (...)
    {
        tsg::report_load_refusal("[glm] load failed: unknown native exception\n");
        return nullptr;
    }
}

TSG_EXPORT int TSGgml_GlmVocabSize(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return m ? m->hp.n_vocab : 0;
}

TSG_EXPORT int TSGgml_GlmCtxSize(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return m ? m->n_ctx : 0;
}

// Quiescent owner calls only: slots and weights remain live. Releasing the
// scheduler first frees every original compute buffer, then its budget tickets.
TSG_EXPORT int TSGgml_GlmConfigureGraphCache(void * handle, int capacity, int trim)
{
    glm_model * m = (glm_model *) handle;
    if (!m || capacity < 1 || capacity > 32) return 0;
    try {
        for (int d = 0; d < m->n_backends; ++d) ggml_backend_synchronize(m->backends[d]);
        if (trim == 1) m->graph_cache.clear();
        else if (trim == 2) {
            for (auto it = m->graph_cache.begin(); it != m->graph_cache.end(); )
                it = (*it)->bd.empty() && (*it)->nt > 1 ? m->graph_cache.erase(it) : std::next(it);
        }
        else if (trim != 0) return 0;
        m->graph_cache_cap = capacity;
        while ((int) m->graph_cache.size() > capacity) m->graph_cache.pop_back();
        return 1;
    } catch (...) { return 0; }
}

// One single-rank GLM model has one bounded graph list, regardless of how many
// requests its graphs reference. Route every scheduler's payload to one stable
// session owner rather than multiplying solo/PP/joint graph peaks per request.
// Existing graphs must be physically released before their ownership changes.
TSG_EXPORT int TSGgml_GlmConfigureSharedGraphOwner(void * handle, int enabled)
{
    auto * m = static_cast<glm_model *>(handle);
    if (!m || m->tp != 1 || m->n_gpu > 1 || (enabled != 0 && enabled != 1)) return 0;
    if (m->shared_graph_owner == (enabled != 0)) return 1;
    if (!m->graph_cache.empty()) return 0;
    try {
        for (int d = 0; d < m->n_backends; ++d) ggml_backend_synchronize(m->backends[d]);
        m->shared_graph_owner = enabled != 0;
        return 1;
    } catch (...) { return 0; }
}

TSG_EXPORT int TSGgml_GlmNPast(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return (m && m->active_slot) ? (int) m->active_slot->n_past : 0;
}

// Evaluate n_tokens at the active slot's current position; logits_out receives
// the last token's row when it is non-null.
TSG_EXPORT int TSGgml_GlmForward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out)
{
    glm_model * m = (glm_model *) handle;
    if (m) m->batched_execution.begin_attempt();
    if (!m || !m->active_slot || m->active_slot->kda_restore_failed || n_tokens <= 0) return 0;

    const int ub = m->n_ubatch > 0 ? m->n_ubatch : 512;
    int done = 0;
    // Per-ubatch slice of the queued vision-override rows. Row indices are
    // relative to THIS call's token array; inside a ubatch they become
    // row-in-ubatch offsets for the graph's set_rows.
    std::vector<float> ovr_rows;
    std::vector<int64_t> ovr_idx;
    while (done < n_tokens)
    {
        const int take = std::min(ub, n_tokens - done);
        const bool last = (done + take) == n_tokens;
        ovr_rows.clear();
        ovr_idx.clear();
        if (!m->embd_ovr.empty())
        {
            const int64_t ne = m->hp.n_embd;
            for (const auto & span : m->embd_ovr)
            {
                const int64_t rows = (int64_t) (span.rows.size() / (size_t) ne);
                for (int64_t r = 0; r < rows; r++)
                {
                    const int64_t gi = span.index + r;
                    if (gi < done || gi >= done + take) continue;
                    ovr_idx.push_back(gi - done);
                    ovr_rows.insert(ovr_rows.end(),
                                    span.rows.begin() + (size_t) (r * ne),
                                    span.rows.begin() + (size_t) ((r + 1) * ne));
                }
            }
        }
        if (!forward_ubatch(*m, tokens + done, take, last && logits_out != nullptr, logits_out,
                            nullptr, false,
                            ovr_rows.empty() ? nullptr : ovr_rows.data(),
                            ovr_idx.empty() ? nullptr : ovr_idx.data(),
                            (int64_t) ovr_idx.size()))
        {
            m->embd_ovr.clear();
            return 0;
        }
        done += take;
    }
    m->embd_ovr.clear();
    return 1;
}

/// Queue projected vision-embedding rows to override the token embeddings of
/// image-placeholder positions in the NEXT TSGgml_GlmForward call. `index` is
/// the first placeholder's position within that call's token array. Cleared
/// after the forward (successful or not).
TSG_EXPORT int TSGgml_GlmQueueVisionRows(void * handle, const float * rows, int n_rows, int index)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !rows || n_rows <= 0 || index < 0) return 0;
    if (!m->hp.g5n)
    {
        fprintf(stderr, "[glm] vision embedding rows are only supported for glm5next\n");
        return 0;
    }
    glm_model::embd_override span;
    span.index = index;
    span.rows.assign(rows, rows + (size_t) n_rows * m->hp.n_embd);
    m->embd_ovr.push_back(std::move(span));
    return 1;
}

/// Drop queued vision rows (a cancelled or re-planned prompt).
TSG_EXPORT void TSGgml_GlmClearVisionRows(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (m) m->embd_ovr.clear();
}

/// One decode step for `n` sequences at once (one token each). Declines — by
/// returning 0 before dispatch when it cannot serve the batch. A failure after
/// dispatch fences all rows and sets LastBatchedFailureNeedsReset; callers must
/// not retry that failure on the per-sequence path.
TSG_EXPORT int TSGgml_GlmForwardBatchedDecode(void * handle, int n, const int32_t * slot_ids,
                                              const int32_t * tokens, const int32_t * positions,
                                              float * logits_out)
{
    glm_model * m = (glm_model *) handle;
    if (m) m->batched_execution.begin_attempt();
    if (!m || n < 2 || !slot_ids || !tokens || !positions || !logits_out) return 0;
    if (n > MAX_BATCHED_DECODE) return 0;
    for (int i = 0; i < n; ++i)
    {
        if (tokens[i] < 0 || tokens[i] >= m->hp.n_vocab || positions[i] < 0 || positions[i] >= m->n_ctx) return 0;
        auto it = m->slots.find(slot_ids[i]);
        if (it == m->slots.end() || it->second->kda_restore_failed) return 0;
    }
    // Tensor parallelism splits the batch's work across ranks a second time; the
    // per-sequence path already handles that case correctly, so the batched
    // graph stays single-rank rather than duplicating the reduction plumbing.
    if (m->tp > 1) return 0;
    try
    {
        return forward_batched_decode(*m, n, slot_ids, tokens, positions, logits_out) ? 1 : 0;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[glm] batched decode failed: %s\n", e.what());
        return 0;
    }
    catch (...)
    {
        fprintf(stderr, "[glm] batched decode failed: non-standard exception\n");
        return 0;
    }
}

// Read immediately after a declined batch under the same executor lock. A
// started failure must never be treated as a retryable serial fallback.
TSG_EXPORT int TSGgml_GlmLastBatchedFailureNeedsReset(void * handle)
{
    auto * m = static_cast<glm_model *>(handle);
    return m && m->batched_execution.failure_needs_reset() ? 1 : 0;
}

// Quiescent/executor-lock query. This is one logical graph's completed native
// execution; backend_splits records scheduler dispatch spans, not GPU launches.
TSG_EXPORT int TSGgml_GlmLastBatchedExecution(void * handle, std::uint64_t * id,
    int * rows, int * nodes, int * backend_splits, int * reused,
    std::int32_t * slots, std::int32_t * before, std::int32_t * after, int capacity)
{
    auto * m = static_cast<glm_model *>(handle);
    if (!m || !id || !rows || !nodes || !backend_splits || !reused || !slots || !before || !after) return 0;
    const auto & value = m->batched_execution.last();
    if (capacity < value.rows) return 0;
    *id = value.id; *rows = value.rows; *nodes = value.nodes;
    *backend_splits = value.backend_splits; *reused = value.graph_reused ? 1 : 0;
    for (int i = 0; i < value.rows; ++i) {
        slots[i] = value.slots[i]; before[i] = value.before[i]; after[i] = value.after[i];
    }
    return value.id != 0 ? 1 : 0;
}

TSG_EXPORT int TSGgml_GlmResetChecked(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot) return 0;
    auto & slot = *m->active_slot;
    slot.kda_restore_failed = true;
    if (m->kda_snap.slot_id == slot.id) m->kda_snap.valid = false;
    try
    {
        if (m->hp.g5n) slot_clear_recurrent(*m, slot);
        slot.n_past = 0;
        slot.kda_restore_failed = false;
        return 1;
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[glm] reset failed; slot remains unusable: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[glm] reset failed with an unknown exception; slot remains unusable\n");
    }
    return 0;
}

TSG_EXPORT void TSGgml_GlmReset(void * handle)
{
    TSGgml_GlmResetChecked(handle);
}

TSG_EXPORT int TSGgml_GlmRewind(void * handle, int n_past)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || m->active_slot->kda_restore_failed || n_past < 0 || n_past > m->active_slot->n_past) return 0;
    // The KDA recurrence cannot be rewound to an earlier position: a cached
    // prefix is reusable only when the new prompt EXTENDS it (no rewind), and
    // anything else must restart from zero with a cleared state.
    if (m->hp.g5n && n_past != m->active_slot->n_past && n_past != 0) return 0;
    if (m->hp.g5n && n_past == 0)
    {
        return TSGgml_GlmResetChecked(handle);
    }
    m->active_slot->n_past = n_past;
    return 1;
}

TSG_EXPORT int TSGgml_GlmSlotAlloc(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return -1;
    try
    {
        tsg::g_last_error.reserve(256);
        glm_slot * s = slot_alloc(*m, /*admit=*/ true);
        if (s) return s->id;
    }
    catch (...) { }
    try { tsg::g_last_error.assign("GLM sequence slot allocation refused or failed; any surviving physical owner remains charged to the model"); }
    catch (...) { }
    return -1;
}

// Exact physical payload, rather than geometry forecasts used for admission.
// Failed cleanup owners remain visible here until their actual buffer release.
TSG_EXPORT int64_t TSGgml_GlmSlotBytes(void * handle, int slot_id)
{
    try
    {
        auto * m = static_cast<glm_model *>(handle);
        if (!m || m->slots.find(slot_id) == m->slots.end()) return 0;
        uint64_t bytes = 0;
        for (const auto & entry : m->slot_ctxs)
            if (entry.slot_id == slot_id && entry.buf)
            {
                const uint64_t count = ggml_backend_buffer_get_size(entry.buf);
                if (count > uint64_t(std::numeric_limits<int64_t>::max()) - bytes) return 0;
                bytes += count;
            }
        return int64_t(bytes);
    }
    catch (...) { return 0; }
}

// Additive exact-state copy interface. The caller owns and has already admitted
// both complete slots; no allocation, rewind or active-slot switch occurs here.
TSG_EXPORT int TSGgml_GlmSlotCopyVersion()
{
    return 1;
}

#if defined(TSG_GGML_TEST_HOOKS)
static thread_local int glm_test_slot_copy_stage = 0, glm_test_slot_copy_kind = 0;
// Reported backend errors / recoverable exceptions, never a fatal driver fault.
TSG_TEST_EXPORT void TSGgml_GlmTestSlotCopyFault(int stage, int kind)
{
    glm_test_slot_copy_stage = stage;
    glm_test_slot_copy_kind = kind;
}
#endif
static void glm_test_slot_copy_fault(int stage)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (glm_test_slot_copy_stage == stage)
    {
        glm_test_slot_copy_stage = 0;
        if (glm_test_slot_copy_kind == 1)
            tsg::g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
        else throw 38;
    }
#else
    (void) stage;
#endif
}

static bool glm_slot_copy_layout(const ggml_tensor * source, const ggml_tensor * destination)
{
    if (!source || !destination) return source == destination;
    if (source == destination || !source->data || !destination->data || source->data == destination->data
        || !source->buffer || !destination->buffer || source->buffer == destination->buffer
        || !ggml_is_contiguous(source) || !ggml_is_contiguous(destination)
        || !ggml_are_same_layout(source, destination) || ggml_nbytes(source) == 0
        || ggml_backend_buffer_get_type(source->buffer) != ggml_backend_buffer_get_type(destination->buffer))
        return false;
    const bool host = ggml_backend_buffer_is_host(source->buffer);
    if (host != ggml_backend_buffer_is_host(destination->buffer)) return false;
    // Bypass ggml_backend_tensor_copy's potentially giant, untracked malloc
    // fallback. Device buffers must expose their own direct tensor-copy hook.
    return host || destination->buffer->iface.cpy_tensor != nullptr;
}

static void glm_slot_copy_sync(glm_model & m)
{
    for (int d = 0; d < m.n_backends; ++d)
        if (m.backends[d]) ggml_backend_synchronize(m.backends[d]);
}

TSG_EXPORT int TSGgml_GlmSlotCopyInto(void * handle, int source_id, int destination_id, int expected_n_past)
{
    auto * m = static_cast<glm_model *>(handle);
    glm_slot * destination = nullptr;
    bool destructive = false;
    auto refuse = [](const char * message) noexcept {
        // Capacity is prepared before any tensor copy, so fixed diagnostics do
        // not allocate after destructive mutation. Failure reporting itself may
        // never let an exception escape this C ABI.
        try { tsg::g_last_error.assign(message); } catch (...) { }
        return 0;
    };
    try
    {
        tsg::g_last_error.reserve(256);
        if (!m || source_id == destination_id || expected_n_past < 0 || expected_n_past > m->n_ctx)
            return refuse("GLM slot copy requires distinct existing slots and an exact valid source head");
        // Direct backend copy currently depends on the active CUDA device.
        // Only CPU and a single GPU/rank are qualified for this interface.
        if (m->tp != 1 || m->n_gpu > 1)
            return refuse("GLM exact slot copy supports only CPU or one GPU with tensor parallelism disabled");
        auto source_it = m->slots.find(source_id), destination_it = m->slots.find(destination_id);
        if (source_it == m->slots.end() || destination_it == m->slots.end())
            return refuse("GLM slot copy source or destination does not exist");
        const auto & source = *source_it->second;
        destination = destination_it->second.get();
        if (source.kda_restore_failed || destination->kda_restore_failed || source.n_past != expected_n_past)
            return refuse("GLM slot copy requires healthy slots and the exact source head; reset required after a failed copy");
        const size_t layers = size_t(m->hp.n_layer + (m->has_mtp ? 1 : 0));
        if (m->tp < 1 || m->tp > MAX_GPUS || m->n_backends < 1)
            return refuse("GLM slot copy backend geometry is invalid");
        for (int rank = 0; rank < m->tp; ++rank)
        {
            const std::vector<ggml_tensor *> * sources[] = { &source.kv_k[rank], &source.idx_k[rank], &source.kda_conv[rank], &source.kda_ssm[rank] };
            const std::vector<ggml_tensor *> * targets[] = { &destination->kv_k[rank], &destination->idx_k[rank], &destination->kda_conv[rank], &destination->kda_ssm[rank] };
            for (int family = 0; family < 4; ++family)
            {
                if (sources[family]->size() != layers || targets[family]->size() != layers)
                    return refuse("GLM slot copy cache families do not contain the complete trunk/draft layout");
                for (size_t layer = 0; layer < layers; ++layer)
                    if (!glm_slot_copy_layout((*sources[family])[layer], (*targets[family])[layer]))
                        return refuse("GLM slot copy layout or direct backend copy is unsupported");
            }
        }
        const auto epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
        auto healthy = [&] {
            return tsg::g_ggml_error_count.load(std::memory_order_acquire) == epoch
                && !tsg::g_backend_compute_failed.load(std::memory_order_acquire);
        };
        if (!healthy()) return refuse("GLM slot copy refused a failed backend");
        glm_slot_copy_sync(*m);
        glm_test_slot_copy_fault(1);
        if (!healthy()) return refuse("GLM slot copy source fence reported an error before mutation");
        // Every layout/metadata/diagnostic capacity check precedes this point.
        // The old destination head remains unpublished until all copies drain.
        destination->kda_restore_failed = true;
        destructive = true;
        if (m->kda_snap.slot_id == destination_id) m->kda_snap.valid = false;
        bool copied = false;
        for (int rank = 0; rank < m->tp; ++rank)
        {
            const std::vector<ggml_tensor *> * sources[] = { &source.kv_k[rank], &source.idx_k[rank], &source.kda_conv[rank], &source.kda_ssm[rank] };
            const std::vector<ggml_tensor *> * targets[] = { &destination->kv_k[rank], &destination->idx_k[rank], &destination->kda_conv[rank], &destination->kda_ssm[rank] };
            for (int family = 0; family < 4; ++family)
                for (size_t layer = 0; layer < layers; ++layer)
                {
                    auto * src = (*sources[family])[layer], * dst = (*targets[family])[layer];
                    if (!src) continue;
                    if (ggml_backend_buffer_is_host(src->buffer)) std::memcpy(dst->data, src->data, ggml_nbytes(src));
                    else if (!ggml_backend_buffer_copy_tensor(src, dst))
                        throw 39; // Never use the upstream allocating fallback.
                    if (!copied) { copied = true; glm_test_slot_copy_fault(2); }
                    if (!healthy()) throw 40;
                }
        }
        glm_slot_copy_sync(*m);
        glm_test_slot_copy_fault(3);
        if (!healthy()) throw 41;
        destination->n_past = source.n_past;
        destination->kda_restore_failed = false;
        tsg::g_last_error.clear();
        return 1;
    }
    catch (...)
    {
        // Drain each backend even if a previous drain threw. The destination
        // stays poisoned after any potentially partial copy; its owner must
        // reset or free it. Source bytes, head and selection never change.
        if (m && destructive)
            for (int d = 0; d < m->n_backends; ++d)
                try { if (m->backends[d]) ggml_backend_synchronize(m->backends[d]); } catch (...) { }
        if (destination && destructive) destination->kda_restore_failed = true;
        return refuse(destructive
            ? "GLM exact slot copy failed; destination requires checked reset before reuse"
            : "GLM exact slot copy failed before destination mutation");
    }
}

#if defined(TSG_GGML_TEST_HOOKS)
// Independent readback of every cache byte; fixed 8KiB test-only host scratch,
// never production copy staging. No digest substitutes for byte equality.
TSG_TEST_EXPORT int TSGgml_GlmTestSlotCompare(void * handle, int source_id, int destination_id,
    int64_t * compared_bytes, int * compared_tensors)
{
    try
    {
        auto * m = static_cast<glm_model *>(handle);
        if (!m || !compared_bytes || !compared_tensors || source_id == destination_id) return 0;
        auto a = m->slots.find(source_id), b = m->slots.find(destination_id);
        if (a == m->slots.end() || b == m->slots.end() || a->second->n_past != b->second->n_past
            || a->second->kda_restore_failed || b->second->kda_restore_failed) return 0;
        const auto epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
        glm_slot_copy_sync(*m);
        unsigned char left[4096], right[4096];
        int64_t bytes = 0;
        int tensors = 0;
        for (int rank = 0; rank < m->tp; ++rank)
        {
            const std::vector<ggml_tensor *> * sources[] = { &a->second->kv_k[rank], &a->second->idx_k[rank], &a->second->kda_conv[rank], &a->second->kda_ssm[rank] };
            const std::vector<ggml_tensor *> * targets[] = { &b->second->kv_k[rank], &b->second->idx_k[rank], &b->second->kda_conv[rank], &b->second->kda_ssm[rank] };
            for (int family = 0; family < 4; ++family)
            {
                if (sources[family]->size() != targets[family]->size()) return 0;
                for (size_t layer = 0; layer < sources[family]->size(); ++layer)
                {
                    auto * src = (*sources[family])[layer], * dst = (*targets[family])[layer];
                    if (!glm_slot_copy_layout(src, dst)) return 0;
                    if (!src) continue;
                    const size_t count = ggml_nbytes(src);
                    if (uint64_t(count) > uint64_t(std::numeric_limits<int64_t>::max() - bytes)) return 0;
                    for (size_t offset = 0; offset < count; )
                    {
                        const size_t take = std::min(sizeof(left), count - offset);
                        ggml_backend_tensor_get(src, left, offset, take);
                        ggml_backend_tensor_get(dst, right, offset, take);
                        if (std::memcmp(left, right, take) != 0) return 0;
                        offset += take;
                    }
                    bytes += int64_t(count); ++tensors;
                }
            }
        }
        glm_slot_copy_sync(*m);
        if (tsg::g_ggml_error_count.load(std::memory_order_acquire) != epoch
            || tsg::g_backend_compute_failed.load(std::memory_order_acquire)) return 0;
        *compared_bytes = bytes; *compared_tensors = tensors;
        return tensors > 0 ? 1 : 0;
    }
    catch (...) { return 0; }
}
#endif

TSG_EXPORT int TSGgml_GlmSetActiveSlot(void * handle, int slot_id)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return 0;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    m->active_slot = it->second.get();
    return 1;
}

TSG_EXPORT int TSGgml_GlmSlotFree(void * handle, int slot_id)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return 0;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    if (m->active_slot == it->second.get())
        m->active_slot = nullptr;
    slot_free(*m, slot_id);
    if (!m->active_slot && !m->slots.empty())
        m->active_slot = m->slots.begin()->second.get();
    return 1;
}

/// The slots the managed side retains for finished conversations, oldest
/// first, replacing the previous list: what a graph that does not fit may free
/// (glm_reclaim_for_graph). The active slot and the slots a graph reads are
/// never freed.
TSG_EXPORT int TSGgml_GlmSetReclaimableSlots(void * handle, const int32_t * slot_ids, int n)
{
    glm_model * m = (glm_model *) handle;
    if (!m || n < 0 || (n > 0 && !slot_ids)) return 0;
    m->reclaimable.assign(slot_ids, slot_ids + n);
    return 1;
}

/// Hand over the ids of retained slots freed for a graph since the last call,
/// at most `cap` of them (the rest stay queued); returns how many were written.
TSG_EXPORT int TSGgml_GlmTakeReclaimedSlots(void * handle, int32_t * out, int cap)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !out || cap <= 0) return 0;
    const int n = std::min(cap, (int) m->reclaimed.size());
    std::copy(m->reclaimed.begin(), m->reclaimed.begin() + n, out);
    m->reclaimed.erase(m->reclaimed.begin(), m->reclaimed.begin() + n);
    return n;
}

// ---------------------------------------------------------------------------
// NextN/MTP speculative decoding
// ---------------------------------------------------------------------------

TSG_EXPORT int TSGgml_GlmHasMtp(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return (m && m->has_mtp) ? 1 : 0;
}

TSG_EXPORT int TSGgml_GlmHiddenSize(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return m ? m->hp.n_embd : 0;
}

/// Trunk forward that also reads back the post-final-norm hidden state of every
/// row (`h_out`, n_tokens * n_embd floats). With `all_logits_rows` the LM head
/// runs on every row and `logits_out` receives n_tokens * n_vocab floats —
/// that is the speculative VERIFY pass, one trunk forward for a whole window.
///
/// A prompt longer than one ubatch is chunked exactly as TSGgml_GlmForward
/// chunks it, and each chunk's hidden states land at their own offset in
/// `h_out`, so the caller always gets one contiguous row per token.
TSG_EXPORT int TSGgml_GlmSpecForward(void * handle, const int32_t * tokens, int n_tokens,
                                     float * h_out, float * logits_out, int all_logits_rows)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || m->active_slot->kda_restore_failed || n_tokens <= 0) return 0;

    int ub = m->n_ubatch > 0 ? m->n_ubatch : 512;
    // A fresh recurrent snapshot identifies a verification (or the accepted
    // prefix replay after restore). Explicit --spec-draft values may exceed
    // the decode arithmetic width. Split those windows into bounded graphs,
    // retaining every hidden/logit row and the original rollback snapshot.
    // Prefill has no snapshot at its current position and keeps its ubatch.
    if (m->hp.g5n && m->kda_snap.valid && m->kda_snap.slot_id == m->active_slot->id &&
        m->kda_snap.n_past == m->active_slot->n_past)
        ub = std::min(ub, (int) TSG_PRECISION_DECODE_COLUMNS);
    const int64_t n_embd = m->hp.n_embd;
    const int64_t n_vocab = m->hp.n_vocab;
    int done = 0;
    while (done < n_tokens)
    {
        const int take = std::min(ub, n_tokens - done);
        const bool last = (done + take) == n_tokens;
        // Only the final chunk needs logits when the caller wants one row; with
        // all-rows logits every chunk contributes its own block.
        const bool want_logits = logits_out != nullptr && (all_logits_rows != 0 || last);
        float * h_chunk = h_out ? h_out + (int64_t) done * n_embd : nullptr;
        float * l_chunk = (want_logits && all_logits_rows != 0) ? logits_out + (int64_t) done * n_vocab
                                                                : logits_out;
        if (!forward_ubatch(*m, tokens + done, take, want_logits, l_chunk, h_chunk, all_logits_rows != 0))
            return 0;
        done += take;
    }
    return 1;
}

/// One NextN draft step: consume `token` at `pos` together with the hidden state
/// of the token before it, and predict the token after it. `h_out` receives the
/// draft block's own hidden state so the next step in the window chains from it.
TSG_EXPORT int TSGgml_GlmMtpDraftStep(void * handle, int token, const float * h_prev, int pos,
                                      float * logits_out, float * h_out)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || !h_prev || pos < 0) return 0;
    const int32_t tok = token;
    return mtp_forward(*m, &tok, 1, h_prev, pos, /*want_logits=*/true, logits_out, h_out) ? 1 : 0;
}

/// Replay verified trunk tokens through the draft block so its KV cache tracks
/// the real context. No logits: the drafts for this window are already drawn.
TSG_EXPORT int TSGgml_GlmMtpCatchUp(void * handle, const int32_t * tokens, int n_tokens,
                                    const float * h_rows, int start_pos)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || !tokens || !h_rows || n_tokens <= 0 || start_pos < 0) return 0;

    // A prompt-sized catch-up is chunked like any other pass; the draft block is
    // one layer, but a 1M-token prompt would still build a graph over the whole
    // window otherwise.
    const int ub = m->n_ubatch > 0 ? m->n_ubatch : 512;
    const int64_t n_embd = m->hp.n_embd;
    int done = 0;
    while (done < n_tokens)
    {
        const int take = std::min(ub, n_tokens - done);
        if (!mtp_forward(*m, tokens + done, take, h_rows + (int64_t) done * n_embd, start_pos + done,
                         /*want_logits=*/false, nullptr, nullptr))
            return 0;
        done += take;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// glm5next: KDA recurrent-state snapshot / restore for speculative decoding
// ---------------------------------------------------------------------------
//
// A verify batch over a speculative window advances every KDA layer's conv
// tail and delta-net state by the whole window, and unlike an MLA row that
// state cannot be rewound by position: after a partial rejection the only way
// back to "the accepted prefix, and nothing else" is a copy of the state from
// before the verify. The managed side takes that copy (capture) right before
// each verify, and on a partial rejection restores it, rewinds to the captured
// position and re-forwards the accepted prefix - the same contract Qwen 3.5's
// GatedDeltaNet and Qwen 3.8's trunk honour (SpecVerifyPersistsAcceptedKv=false).
//
// The copies are device-to-device on the device that owns each layer's state;
// nothing crosses the host. On GLM-5.3-Flash that is ~150 MB per capture.

#if defined(TSG_GGML_TEST_HOOKS)
static thread_local int glm_test_kda_stage = 0, glm_test_kda_kind = 0;
// Keep fixture-only entry points separate from the production iOS export list.
#ifndef TSG_TEST_EXPORT
#define TSG_TEST_EXPORT TSG_EXPORT
#endif
TSG_TEST_EXPORT void TSGgml_GlmTestKdaSnapshotFault(int stage, int kind)
{
    glm_test_kda_stage = stage;
    glm_test_kda_kind = kind;
}
#endif
static void glm_test_kda_fault(int stage)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (glm_test_kda_stage == stage)
    {
        glm_test_kda_stage = 0;
        if (glm_test_kda_kind == 1) throw std::bad_alloc();
        throw 17;
    }
#else
    (void) stage;
#endif
}

/// Make sure the snapshot arena mirrors the active slot's state tensors (same
/// shapes, same buffer types). All slots share one arena because their state
/// tensors are allocated identically; a shape mismatch (defensive) rebuilds it.
static bool kda_snapshot_ensure(glm_model & m, glm_slot & slot)
{
    glm_model::kda_snapshot & snap = m.kda_snap;
    bool fits = !snap.ctxs.empty();
    for (int r = 0; r < m.tp && fits; r++)
    {
        if (snap.conv[r].size() != slot.kda_conv[r].size() || snap.ssm[r].size() != slot.kda_ssm[r].size())
        { fits = false; break; }
        for (size_t il = 0; il < slot.kda_conv[r].size() && fits; il++)
        {
            ggml_tensor * a = slot.kda_conv[r][il], * b = snap.conv[r][il];
            ggml_tensor * c = slot.kda_ssm[r][il],  * d = snap.ssm[r][il];
            if ((a == nullptr) != (b == nullptr) || (c == nullptr) != (d == nullptr)) { fits = false; break; }
            if (a && (!ggml_are_same_shape(a, b) || a->type != b->type
                      || ggml_backend_buffer_get_type(a->buffer) != ggml_backend_buffer_get_type(b->buffer)))
            { fits = false; break; }
            if (c && (!ggml_are_same_shape(c, d) || c->type != d->type
                      || ggml_backend_buffer_get_type(c->buffer) != ggml_backend_buffer_get_type(d->buffer)))
            { fits = false; break; }
        }
    }
    if (fits) return true;
    snap.release();

    // Group the mirrors by buffer type (one context + one buffer each), so a
    // layer-split model gets one arena per device and TP one per rank.
    std::vector<ggml_backend_buffer_type_t> bufts;
    std::vector<size_t> counts;
    auto group_of = [&](ggml_tensor * t) -> size_t
    {
        ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(t->buffer);
        for (size_t i = 0; i < bufts.size(); i++) if (bufts[i] == bt) return i;
        bufts.push_back(bt);
        counts.push_back(0);
        return bufts.size() - 1;
    };
    for (int r = 0; r < m.tp; r++)
    {
        for (ggml_tensor * t : slot.kda_conv[r]) if (t) counts[group_of(t)]++;
        for (ggml_tensor * t : slot.kda_ssm[r])  if (t) counts[group_of(t)]++;
    }
    if (bufts.empty())
        return true;                         // nothing recurrent to mirror

    snap.ctxs.assign(bufts.size(), nullptr);
    snap.bufs.assign(bufts.size(), nullptr);
    for (size_t g = 0; g < bufts.size(); g++)
    {
        ggml_init_params ip = { (counts[g] + 4) * ggml_tensor_overhead(), nullptr, true };
        snap.ctxs[g] = ggml_init(ip);
        if (!snap.ctxs[g]) { snap.release(); return false; }
    }
    for (int r = 0; r < m.tp; r++)
    {
        snap.conv[r].assign(slot.kda_conv[r].size(), nullptr);
        snap.ssm[r].assign(slot.kda_ssm[r].size(), nullptr);
        for (size_t il = 0; il < slot.kda_conv[r].size(); il++)
        {
            if (ggml_tensor * t = slot.kda_conv[r][il])
            {
                snap.conv[r][il] = ggml_dup_tensor(snap.ctxs[group_of(t)], t);
                ggml_format_name(snap.conv[r][il], "snap_kconv.%d.%d", r, (int) il);
            }
            if (ggml_tensor * t = slot.kda_ssm[r][il])
            {
                snap.ssm[r][il] = ggml_dup_tensor(snap.ctxs[group_of(t)], t);
                ggml_format_name(snap.ssm[r][il], "snap_kssm.%d.%d", r, (int) il);
            }
        }
    }
    for (size_t g = 0; g < bufts.size(); g++)
    {
        snap.bufs[g] = ggml_backend_alloc_ctx_tensors_from_buft(snap.ctxs[g], bufts[g]);
        if (!snap.bufs[g])
        {
            fprintf(stderr, "[glm] KDA snapshot arena allocation failed (%s)\n", ggml_backend_buft_name(bufts[g]));
            snap.release();
            return false;
        }
    }
    return true;
}

/// Wait for every backend so a device-to-device state copy neither races the
/// graph that produced the state nor the graph that will consume it.
static void kda_snapshot_sync(glm_model & m)
{
    for (int i = 0; i < m.n_backends; i++)
        if (m.backends[i]) ggml_backend_synchronize(m.backends[i]);
}

/// Copy the active slot's KDA recurrent state into the snapshot arena and
/// record the slot's position. Returns 1 on success. On glm-dsa proper (no
/// recurrent state) only the position is recorded, so a restore is a rewind.
static int glm_kda_state_capture(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot) return 0;
    glm_slot & slot = *m->active_slot;
    glm_model::kda_snapshot & snap = m->kda_snap;
    snap.valid = false;
    glm_test_kda_fault(1);
    if (m->hp.g5n)
    {
        if (!kda_snapshot_ensure(*m, slot)) return 0;
        kda_snapshot_sync(*m);
        for (int r = 0; r < m->tp; r++)
        {
            for (size_t il = 0; il < slot.kda_conv[r].size(); il++)
            {
                if (slot.kda_conv[r][il]) ggml_backend_tensor_copy(slot.kda_conv[r][il], snap.conv[r][il]);
                if (slot.kda_ssm[r][il])  ggml_backend_tensor_copy(slot.kda_ssm[r][il],  snap.ssm[r][il]);
            }
        }
        kda_snapshot_sync(*m);
    }
    snap.slot_id = slot.id;
    snap.n_past = slot.n_past;
    snap.valid = true;
    return 1;
}

/// Copy the snapshot back into the active slot and rewind the slot to the
/// captured position. Returns that position (>= 0), or -1 when there is no
/// usable snapshot for the active slot (never taken, taken of another slot,
/// invalidated by a reset/free, or the slot is already behind it).
static int glm_kda_state_restore(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot) return -1;
    glm_slot & slot = *m->active_slot;
    glm_model::kda_snapshot & snap = m->kda_snap;
    if (!snap.valid || snap.slot_id != slot.id || snap.n_past > slot.n_past)
    {
        fprintf(stderr, "[glm] KDA state restore refused: no snapshot for slot %d at or before position %" PRId64 "\n",
                slot.id, slot.n_past);
        return -1;
    }
    if (m->hp.g5n)
    {
        // The arena was built against THIS slot's shapes (or an identical
        // slot's). A mismatch means the slot was reallocated underneath the
        // snapshot; a rebuilt arena would hold nothing, so refuse rather than
        // restore zeros.
        for (int r = 0; r < m->tp; r++)
        {
            if (snap.conv[r].size() != slot.kda_conv[r].size() ||
                snap.ssm[r].size() != slot.kda_ssm[r].size() ||
                slot.kda_ssm[r].size() != slot.kda_conv[r].size()) return -1;
            for (size_t il = 0; il < slot.kda_conv[r].size(); il++)
            {
                ggml_tensor * a = slot.kda_conv[r][il], * b = snap.conv[r][il];
                ggml_tensor * c = slot.kda_ssm[r][il],  * d = snap.ssm[r][il];
                if ((a == nullptr) != (b == nullptr) || (c == nullptr) != (d == nullptr)) return -1;
                if (a && !ggml_are_same_shape(a, b)) return -1;
                if (c && !ggml_are_same_shape(c, d)) return -1;
            }
        }
        kda_snapshot_sync(*m);
        for (int r = 0; r < m->tp; r++)
        {
            for (size_t il = 0; il < slot.kda_conv[r].size(); il++)
            {
                if (slot.kda_conv[r][il])
                {
                    ggml_backend_tensor_copy(snap.conv[r][il], slot.kda_conv[r][il]);
                    glm_test_kda_fault(2);
                }
                if (slot.kda_ssm[r][il])  ggml_backend_tensor_copy(snap.ssm[r][il],  slot.kda_ssm[r][il]);
            }
        }
        kda_snapshot_sync(*m);
    }
    slot.n_past = snap.n_past;
    return (int) snap.n_past;
}

// C ABI callers must receive a failure status even when arena allocation or
// a backend state copy throws. A failed capture leaves live state untouched;
// a failed restore may have copied only some layers and therefore poisons only
// that slot until ResetChecked succeeds.
TSG_EXPORT int TSGgml_GlmKdaStateCapture(void * handle)
{
    auto * m = (glm_model *) handle;
    if (!m || !m->active_slot || m->active_slot->kda_restore_failed) return 0;
    try { return glm_kda_state_capture(handle); }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[glm] KDA capture failed: %s\n", error.what());
    }
    catch (...) { fprintf(stderr, "[glm] KDA capture failed with an unknown exception\n"); }
    m->kda_snap.valid = false;
    return 0;
}

TSG_EXPORT int TSGgml_GlmKdaStateRestore(void * handle)
{
    auto * m = (glm_model *) handle;
    if (!m || !m->active_slot || m->active_slot->kda_restore_failed) return -1;
    try { return glm_kda_state_restore(handle); }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[glm] KDA restore failed; reset required: %s\n", error.what());
    }
    catch (...) { fprintf(stderr, "[glm] KDA restore failed with an unknown exception; reset required\n"); }
    m->kda_snap.valid = false;
    m->active_slot->kda_restore_failed = true;
    return -1;
}

TSG_EXPORT void TSGgml_GlmFree(void * handle)
{
    delete (glm_model *) handle;
}
