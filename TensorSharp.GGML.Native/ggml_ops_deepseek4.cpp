// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model executor for the GGML backend.
//
// Unlike the per-op fused paths used by the other architectures, this file is
// a self-contained model runtime: it loads the (possibly split) GGUF itself,
// places the per-layer weights across every visible GPU (layer split, so a
// model larger than one GPU's VRAM is hosted across all of them), owns the
// DSV4 KV caches (raw sliding-window ring + CSA/HCA compressed K caches +
// compressor state rings + lightning-indexer cache), and executes prefill /
// decode ubatches through ggml_backend_sched.
//
// The graph is a port of llama.cpp's `src/models/deepseek4.cpp` restricted to
// a single sequence / single stream:
//   * 4-stream hyper-connections (fused ggml_dsv4_hc_pre/comb/post ops).
//   * LoRA-factored Q, single shared 512-dim K(=V) head, per-head q RMS norm,
//     attention sinks, RoPE -> attention -> inverse RoPE on the rope slice,
//     grouped LoRA output projection.
//   * Per-layer compress ratio 0 (raw SWA-128 attention only), 4 (CSA:
//     overlap-compressed rows + lightning-indexer top-k selection), or 128
//     (HCA: block-compressed rows, all visible).
//   * MoE: sqrt(softplus) routing in F32, bias for selection, first
//     `hash_layer_count` layers route by token id through a tid2eid LUT,
//     swiglu clamp, weight normalization, x1.5 routed scale + shared expert.
//
// The Hadamard rotation llama.cpp applies to quantized KV caches is skipped
// on purpose: caches here are F16/F32 and the rotation is an orthonormal
// involution applied consistently to q/k/v and inverted on output, so
// skipping it everywhere is mathematically identical.
// ---------------------------------------------------------------------------

#include "ggml_ops_internal.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_scheduler_alloc.h"
#include "dsv41_engram.h"
#include "dsv41_raw_gather.h"
#include "dsv41_truncate.h"
#include "dsv41_dspark.h"
#include "ggml_ops_precision_policy.h"
#include "dsv41_retention.h"
#include "dsv41_engram_io.h"
#include "dsv41_engram_advice.h"
#include "dsv4_file_warm.h"
#include "ggml_ops_deepseek41_vision.h"
#include "ggml_ops_deepseek41_tp.h"
#if defined(TSG_GGML_TEST_HOOKS)
#include "ggml-impl.h"
#endif

#include "gguf.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <filesystem>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tsg_dsv4
{

// The token-batched decode entry point caps a step at this many sequences; the
// per-slot attention forks are its only O(n) graph cost.
static constexpr int DSV4_MAX_BATCHED_SLOTS = 16;
static constexpr int64_t CSA_RATIO = 4;
static constexpr int64_t HCA_RATIO = 128;
static constexpr int     MAX_GPUS  = 8;

static int64_t pad64(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------
struct dsv4_hparams
{
    bool v41 = false;
    std::vector<int32_t> kv_sources, index_sources;
    int candidate_source = -1, candidate_topk = 0, candidate_block = 0;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_vocab = 0;
    int32_t n_embd_head = 0;      // attention.key_length (512)
    int32_t n_rot = 0;            // rope.dimension_count (64)
    int32_t q_lora_rank = 0;
    int32_t o_groups = 0;
    int32_t o_lora_rank = 0;
    int32_t n_swa = 0;            // sliding window (raw attention)
    float   rms_eps = 1e-6f;

    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    int32_t n_ff_exp = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm = false;
    int32_t hash_layer_count = 0;
    std::vector<float> swiglu_clamp_exp;
    std::vector<float> swiglu_clamp_shexp;

    int32_t indexer_n_head = 0;
    int32_t indexer_head_size = 0;
    int32_t indexer_top_k = 0;

    std::vector<int32_t> compress_ratios;
    float compress_rope_base = 10000.0f;

    int32_t hc_mult = 0;
    int32_t hc_sinkhorn_iters = 0;
    float   hc_eps = 1e-6f;

    float rope_freq_base = 10000.0f;
    float yarn_freq_scale = 1.0f;   // 1/factor
    float yarn_ext_factor = 0.0f;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    int32_t n_ctx_orig = 0;
};

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor)
{
    if (ext_factor == 0.0f) return 1.0f;
    return 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
struct dsv4_layer
{
    int kv_source = -1, index_source = -1, engram_index = -1;
    ggml_tensor * indexer_k = nullptr;
    ggml_tensor * indexer_k_norm = nullptr;
    ggml_tensor * engram_embd = nullptr;
    ggml_tensor * engram_k = nullptr;
    ggml_tensor * engram_q = nullptr;
    ggml_tensor * engram_wkv = nullptr;
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_sinks = nullptr;
    ggml_tensor * wq_a = nullptr;
    ggml_tensor * attn_q_a_norm = nullptr;
    ggml_tensor * wq_b = nullptr;
    ggml_tensor * wkv = nullptr;
    ggml_tensor * attn_kv_norm = nullptr;
    ggml_tensor * wo_a = nullptr;
    ggml_tensor * wo_b = nullptr;

    ggml_tensor * hc_attn_fn = nullptr;
    ggml_tensor * hc_attn_base = nullptr;
    ggml_tensor * hc_attn_scale = nullptr;
    ggml_tensor * hc_ffn_fn = nullptr;
    ggml_tensor * hc_ffn_base = nullptr;
    ggml_tensor * hc_ffn_scale = nullptr;

    ggml_tensor * attn_comp_wkv = nullptr;
    ggml_tensor * attn_comp_wgate = nullptr;
    ggml_tensor * attn_comp_ape = nullptr;
    ggml_tensor * attn_comp_norm = nullptr;

    ggml_tensor * indexer_proj = nullptr;
    ggml_tensor * indexer_attn_q_b = nullptr;
    ggml_tensor * indexer_comp_wkv = nullptr;
    ggml_tensor * indexer_comp_wgate = nullptr;
    ggml_tensor * indexer_comp_ape = nullptr;
    ggml_tensor * indexer_comp_norm = nullptr;

    ggml_tensor * ffn_gate_inp = nullptr;
    ggml_tensor * ffn_gate_tid2eid = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr;
    ggml_tensor * ffn_down_exps = nullptr;
    ggml_tensor * ffn_up_exps = nullptr;
    ggml_tensor * ffn_gate_shexp = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;
    ggml_tensor * ffn_up_shexp = nullptr;

    int device = 0;
    // MoE CPU offload (--n-cpu-moe): the routed experts of this layer live in
    // system RAM and their mul_mat_id chain runs on the ggml CPU backend.
    // Everything else in the layer — attention, norms, router, shared expert —
    // stays on `device`. 137 of this checkpoint's 151 GiB are routed experts,
    // so a handful of offloaded layers is the difference between "does not fit
    // in the visible VRAM" and "runs".
    bool cpu_moe = false;
};

// Per-layer KV caches / compressor state for ONE sequence slot
// (device-resident, persist across ubatches).
struct dsv4_slot_layer
{
    ggml_tensor * raw_k = nullptr;        // F16 [n_embd_head, ring_raw]
    ggml_tensor * csa_k = nullptr;        // F16 [n_embd_head, n_csa_rows]   (ratio 4)
    ggml_tensor * lid_k = nullptr;        // F16 [idx_head_size, n_csa_rows] (ratio 4)
    ggml_tensor * hca_k = nullptr;        // F16 [n_embd_head, n_hca_rows]   (ratio 128)
    ggml_tensor * comp_state_kv = nullptr;    // F32 [coff*head, state_size]
    ggml_tensor * comp_state_score = nullptr; // F32 [coff*head, state_size]
    ggml_tensor * lid_state_kv = nullptr;     // F32 [2*idx_head, 8]
    ggml_tensor * lid_state_score = nullptr;  // F32 [2*idx_head, 8]

    // Rewind checkpoint (V4.1 only, see dsv4_slot::cp_n_past). Shadow copies of
    // the two caches whose addressing is MODULAR, so a later position overwrites
    // an earlier one's row: the raw sliding-window ring and the compressor state
    // ring. The compressed caches (csa_k / lid_k / hca_k) need no shadow - they
    // are addressed by absolute block index pos/ratio and only ever grow, so a
    // truncation makes their tail invisible rather than wrong.
    ggml_tensor * raw_k_cp = nullptr;          // F16 [n_embd_head, ring_raw]
    ggml_tensor * comp_state_kv_cp = nullptr;  // F32, same shape as comp_state_kv
    ggml_tensor * comp_state_score_cp = nullptr;
};

// One independent sequence context: its own caches + position, sharing the
// model weights. The server's continuous-batching engine binds one slot per
// in-flight request; the CLI / single-stream path only ever uses slot 0.
// DSpark speculative decoding: the checkpoint's `mtp.*` support module, loaded
// from a separate drafter GGUF. Three DSV4 blocks (compress_ratio 0) that read
// the trunk's hidden states, a Markov head that conditions each block position
// on the token before it, and a confidence head predicting per-position
// acceptance. The stage weights are appended to dsv4_model::layers so the
// existing MoE / hyper-connection builders work on them unchanged.
struct dsv4_dspark
{
    bool loaded = false;
    int32_t block_size = 0;
    int32_t markov_rank = 0;
    int32_t noise_token = 0;
    int32_t n_stages = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    std::vector<int32_t> target_layers;   // V4 outputs; V4.1 attention-input stream means
    int layer_base = 0;                   // index of stage 0 in dsv4_model::layers
    int dev = 0;                          // hosting device (the output head's)

    ggml_tensor * main_norm = nullptr;
    ggml_tensor * main_proj = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * hc_head_fn = nullptr;
    ggml_tensor * hc_head_scale = nullptr;
    ggml_tensor * hc_head_base = nullptr;
    ggml_tensor * markov_w1 = nullptr;
    ggml_tensor * markov_w2 = nullptr;
    ggml_tensor * conf_proj = nullptr;
};

struct dsv4_slot
{
    std::vector<int32_t> engram_history;
    // A failed execution may have written only part of its cache topology,
    // including V4 pipelined/speculative forwards. The historical field name
    // is retained, but both architectures require a complete reset for reuse.
    bool v41_failed = false;
    int id = 0;
    int32_t n_past = 0;
    // Rewind checkpoint: the modular ring state as it stood at the end of the
    // most recent MULTI-TOKEN forward, i.e. at a prompt boundary. -1 when there
    // is none (a fresh or reset slot, or a build without the shadow tensors).
    //
    // Why a boundary rather than "wherever we are": the raw ring holds only
    // ring_raw = pad64(n_swa + n_ubatch, 256) positions, so generating a turn's
    // answer overwrites the rows of the tokens the NEXT turn wants to rewind to.
    // A chat template that re-renders a past assistant turn differently (V4.1
    // drops past reasoning in ordinary chat) diverges from the cache exactly one
    // token after the previous turn's <|Assistant|>, which is one position
    // before this checkpoint - so keeping the boundary state is what makes that
    // rewind possible at all. Decode steps deliberately do not move it.
    int32_t cp_n_past = -1;
    int32_t spec_begin = -1;
    int32_t spec_end = -1;
    std::vector<dsv4_slot_layer> layers;
    // DSpark: one SWA ring per drafter stage, keyed by trunk position and fed
    // from the trunk's own hidden states (see build_dspark_ring_update).
    std::vector<ggml_tensor *> ds_k;
    std::vector<ggml_tensor *> ds_k_cp;
    ggml_context * ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t buf[MAX_GPUS + 1] = {};

    ~dsv4_slot()
    {
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (buf[i]) ggml_backend_buffer_free(buf[i]);
            if (ctx[i]) ggml_free(ctx[i]);
        }
    }
};

struct dsv41_slot_write_guard
{
    dsv4_slot * slot;
    bool started = false;
    bool complete = false;
    ~dsv41_slot_write_guard() { if (slot && started && !complete) slot->v41_failed = true; }
};

// The token-batched decode path mutates every participating slot's Engram
// history before the graph runs, so a failure anywhere latches all of them:
// their histories have advanced past caches that were never written.
struct dsv41_batched_write_guard
{
    std::vector<dsv4_slot *> slots;
    bool complete = false;
    ~dsv41_batched_write_guard()
    {
        if (complete) return;
        for (auto * slot : slots) if (slot) slot->v41_failed = true;
    }
};

#if defined(TSG_GGML_TEST_HOOKS)
// Faults occur after real mutation, so regression tests exercise recovery from
// partial history/cache updates. Release builds do not contain these hooks.
static void dsv41_test_fail(const char * stage, int64_t position)
{
    const char * configured = std::getenv("TS_DSV41_TEST_FAIL_STAGE");
    const char * minimum = std::getenv("TS_DSV41_TEST_FAIL_POSITION");
    if (configured && std::strcmp(configured, stage) == 0 &&
        position >= (minimum ? std::atoll(minimum) : 0))
    {
        const char * kind = std::getenv("TS_DSV41_TEST_FAIL_KIND");
        if (kind && std::strcmp(kind, "unknown") == 0) throw 41;
        if (kind && std::strcmp(kind, "bad_alloc") == 0) throw std::bad_alloc();
        throw std::runtime_error(std::string("Injected V4.1 execution failure after ") + stage);
    }
}

// Used only by the model-free C ABI regression below. Interception is before
// any model tensors are accessed, never in a production or real-fixture run.
static thread_local int dsv4_test_boundary_failure = 0;
static thread_local int dsv4_test_boundary_visits = 0;
static thread_local int dsv4_test_boundary_stage = 0;
static bool dsv4_test_boundary_fault(int stage = 0)
{
    if (!dsv4_test_boundary_failure || stage != dsv4_test_boundary_stage) return false;
    ++dsv4_test_boundary_visits;
    if (dsv4_test_boundary_failure == 1) throw std::bad_alloc();
    if (dsv4_test_boundary_failure == 2) throw 41;
    return true;
}
#endif

// gguf tensor location within the (possibly split) model
struct tensor_source
{
    int shard = -1;
    size_t offset = 0; // absolute file offset of the tensor data
    size_t size = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

// ---------------------------------------------------------------------------
// Compression plans (single sequence port of dsv4_build_comp_plan)
// ---------------------------------------------------------------------------
struct comp_plan
{
    std::vector<int32_t> state_pos;
    std::vector<int32_t> state_persist_src_idxs;
    std::vector<int64_t> state_persist_dst_idxs;
    std::vector<int32_t> state_read_idxs;
    std::vector<int64_t> state_write_idxs;
    std::vector<int32_t> state_write_pos;
    std::vector<int32_t> n_visible;
    int64_t n_kv = 0;
};

struct plan_inputs
{
    ggml_tensor * state_pos = nullptr;      // I32 [nt]
    ggml_tensor * read_idxs = nullptr;      // I32 [...]
    ggml_tensor * write_idxs = nullptr;     // I64 [...]  (unfused)
    ggml_tensor * write_pos = nullptr;      // I32 [...]  (unfused)
    ggml_tensor * persist_src = nullptr;    // I32 [...]  (unfused)
    ggml_tensor * persist_dst = nullptr;    // I64 [...]  (unfused)
    // fused: single I32 input [read_idxs | (cache_row,rope_pos)*n_blocks | (src_col,dst_row)*np]
    ggml_tensor * comp_meta = nullptr;
    ggml_tensor * kq_mask = nullptr;        // F16 [n_kv, nt]
    int64_t n_kv = 0;
};

// Per-token inputs are duplicated per participating device (indexed by device
// id) so no scheduler-driven cross-backend input copies — each a synchronized
// host round trip — happen inside the per-token compute.
struct graph_inputs
{
    // Host-lookup path: F32 [hash_columns * head_dim, nt] staged rows.
    std::vector<ggml_tensor *> engram;
    // Device-table path: I32 [hash_columns * nt] row ids consumed by get_rows.
    std::vector<ggml_tensor *> engram_ids;
    ggml_tensor * engram_rows = nullptr;
    ggml_tensor * image_embeddings = nullptr; // F32 [hidden, number of image positions]
    ggml_tensor * image_indices = nullptr;    // I64 [number of image positions]
    ggml_tensor * image_types[MAX_GPUS + 1] = {}; // I32 [nt], 0 text / 1 image (including delimiters)
    ggml_tensor * tokens[MAX_GPUS + 1] = {};    // I32 [nt]
    ggml_tensor * pos[MAX_GPUS + 1] = {};       // I32 [nt]
    ggml_tensor * raw_idxs[MAX_GPUS + 1] = {};  // I64 [nt]
    ggml_tensor * raw_read_idxs[MAX_GPUS + 1] = {}; // V4.1 optional I32 [padded SWA] compact decode read
    ggml_tensor * raw_mask[MAX_GPUS + 1] = {};  // F16 [ring, nt]
    // decode gather: F16 [(raw ring or padded SWA) + top_k, 1], top-k tail zero
    ggml_tensor * gather_mask[MAX_GPUS + 1] = {};
    ggml_tensor * out_ids = nullptr;            // I32 [1] or [nt] (last device)
    // DSpark draft graph inputs (drafter device)
    ggml_tensor * ds_tokens = nullptr;          // I32 [block]  [anchor, noise...]
    ggml_tensor * ds_pos = nullptr;             // I32 [block]
    ggml_tensor * ds_idxs = nullptr;            // I64 [block]  ring slots of the block
    ggml_tensor * ds_mask = nullptr;            // F16 [ring + block, block]
    plan_inputs csa[MAX_GPUS + 1];
    plan_inputs hca[MAX_GPUS + 1];
    plan_inputs lid[MAX_GPUS + 1];
};

// Per-slot piece of a batched multi-slot decode graph: one decode token per
// slot, each against its own caches at its own position.
struct bd_slot_state
{
    int slot_id = 0;
    int64_t p0 = 0;          // this slot's decode position
    bool skip_topk = false;  // per-slot indexer-skip decision
    comp_plan plan_csa;
    comp_plan plan_hca;
    comp_plan plan_lid;
    // per-device inputs (created only on devices hosting matching layers)
    ggml_tensor * raw_idxs[MAX_GPUS + 1] = {};    // I64 [1]
    ggml_tensor * raw_mask[MAX_GPUS + 1] = {};    // F16 [ring, 1]
    ggml_tensor * gather_mask[MAX_GPUS + 1] = {}; // F16 [ring+top_k, 1] (!skip)
    ggml_tensor * csa_mask[MAX_GPUS + 1] = {};    // F16 [n_kv_csa, 1] (skip)
    ggml_tensor * lid_mask[MAX_GPUS + 1] = {};    // F16 [n_kv_lid, 1] (!skip)
    ggml_tensor * hca_mask[MAX_GPUS + 1] = {};    // F16 [n_kv_hca, 1]
    ggml_tensor * csa_meta[MAX_GPUS + 1] = {};    // I32 comp meta
    ggml_tensor * lid_meta[MAX_GPUS + 1] = {};
    ggml_tensor * hca_meta[MAX_GPUS + 1] = {};
    // V4.1 batched decode: this slot's compressor index inputs. V4.1 keeps the
    // unfused compressor (explicit get_rows/set_rows), so each slot needs its
    // own index tensors rather than the packed meta the V4 fused path uses.
    // Their read/persist indices address the SHARED batched projection, so the
    // fill offsets every current-ubatch index by this slot's column.
    plan_inputs v41_csa[MAX_GPUS + 1];
    plan_inputs v41_hca[MAX_GPUS + 1];
    plan_inputs v41_lid[MAX_GPUS + 1];
    ggml_tensor * raw_read_idxs[MAX_GPUS + 1] = {};  // I32 [padded SWA] compact raw gather
    bool use_gather = false;         // this slot's sparse-selection decision
    bool compact_raw = false;        // ... and whether its raw prefix is compacted
};

struct graph_build_result
{
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;   // owned; keeps this graph's allocation alive
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ds_toks = nullptr;    // I32 [block] drafted ids
    ggml_tensor * ds_conf = nullptr;    // F32 [block] acceptance probabilities
    bool all_logits = false;            // logits for every row (speculative verify)
    int image_tokens = 0;               // compact embedding input changes graph shape
    bool draft = false;                 // this entry is the drafter's graph
    comp_plan plan_csa;
    comp_plan plan_hca;
    comp_plan plan_lid;
    int64_t nt = 0;
    int64_t p0 = 0;
    // sequence slot whose cache tensors this graph was built against; entries
    // are purged when their slot is freed (the baked addresses die with it)
    int slot_id = 0;
    // batched multi-slot decode: per-slot plans/inputs (empty for normal
    // single-slot graphs). Purge-on-free checks these slot ids too.
    std::vector<bd_slot_state> bd;
    // shape signature this graph was built for (plan array sizes + n_kv pads)
    uint64_t sig = 0;
    // descriptors referenced (by pointer) from the fused GGML_OP_CUSTOM nodes
    std::deque<tsg_dsv4_fused_desc> descs;
    // per-device completion events of this entry's last pipelined use; input
    // refills wait on these instead of a full pipeline drain
    ggml_backend_event_t use_events[MAX_GPUS] = {};
    // this entry's own compute buffers, per device, recorded once it is
    // allocated. The cache is trimmed against these, not against a count.
    size_t buffer_bytes[MAX_GPUS] = {};

    ~graph_build_result()
    {
        for (auto & ev : use_events)
            if (ev) ggml_backend_event_free(ev);
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct dsv41_vision_attachment
{
    std::shared_ptr<tsg_dsv41_vision::encoder> encoder;
    std::vector<ggml_tensor *> router_bias; // [experts, 2]: text bias, then visual bias
    ggml_context * ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t buf[MAX_GPUS + 1] = {};
    ~dsv41_vision_attachment()
    {
        for (int d = 0; d <= MAX_GPUS; ++d)
        {
            if (buf[d]) ggml_backend_buffer_free(buf[d]);
            if (ctx[d]) ggml_free(ctx[d]);
        }
    }
};

struct dsv4_model
{
    dsv4_hparams hp;
    tsg_dsv41::engram_data engram;
    std::unique_ptr<tsg_dsv41::engram_io_pool> engram_io;
    // Engram tables live on the GPU that owns their layer and are gathered by
    // ggml get_rows, instead of being read and dequantized on the host. Set at
    // load once the placement is known to fit; see dsv4_load.
    bool engram_on_device = false;
    // Host-resident tables are read a few scattered rows per token, so every
    // row that is not already page cache is a storage round trip. Warming the
    // mapping turns that into a RAM read. It costs minutes on a network
    // filesystem, so the automatic form runs AFTER load on its own thread and
    // the model stays usable (slower) while it proceeds.
    std::thread engram_warm_thread;
    std::atomic<bool> engram_warm_stop{false};
    std::unique_ptr<tsg_dsv41_tp::executor> moe_tp;
    std::unique_ptr<dsv41_vision_attachment> vision;

    // backends: [0..n_gpu) GPU, [n_gpu] CPU
    ggml_backend_t backends[MAX_GPUS + 1] = {};
    int n_gpu = 0;
    int n_backends = 0;

    // fused-op backends (one per GPU, wrapping that GPU's CUDA backend)
    ggml_backend_t ts_backends[MAX_GPUS] = {};
    // What ggml_backend_sched is given for each device: the wrapper for a GPU
    // when fused ops are available, the CUDA backend otherwise, and the CPU
    // backend at index n_gpu -- the same indexing as backends[], because
    // graph_builder::pin() is called with n_gpu for host-resident routed
    // experts. Use this, not backends[], to place graph tensors.
    ggml_backend_t dev_backends[MAX_GPUS + 1] = {};
    bool fused = false;

    // Persistent worker pool for the CPU backend — see dsv4_load. Shared by
    // every host split of every graph, so --n-cpu-moe pays for its threads once
    // instead of once per offloaded layer per token.
    ggml_threadpool_t cpu_threadpool = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
    int test_cpu_pool_threads = 0;
#endif

    // scheduler backend list: [cuda 0..n_gpu) , (fused ts 0..n_gpu) , cpu]
    ggml_backend_t sched_backends[2 * MAX_GPUS + 1] = {};
    ggml_backend_buffer_type_t sched_bufts[2 * MAX_GPUS + 1] = {};
    int n_sched_backends = 0;

    // weight + cache contexts/buffers per device (+1 = cpu, unused normally)
    ggml_context * w_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t w_buf[MAX_GPUS + 1] = {};
    ggml_context * c_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t c_buf[MAX_GPUS + 1] = {};

    // GGUF shard mmaps backing the host-resident (cpu_moe) expert tensors.
    // File-backed pages are page cache the kernel can evict and re-read, so a
    // model whose offloaded experts outweigh what the host allows this process
    // (cgroup memory limit on containers — `free` lies there) loads and runs
    // at page-cache speed instead of being OOM-killed mid-copy. Indexed by
    // shard; null when that shard is not mapped.
    std::vector<void *> mmap_addrs;
    std::vector<size_t> mmap_sizes;
    std::vector<ggml_backend_buffer_t> mmap_bufs;
    size_t mmap_weight_bytes = 0; // tensor bytes served from the mappings

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * hc_head_fn = nullptr;
    ggml_tensor * hc_head_base = nullptr;
    ggml_tensor * hc_head_scale = nullptr;

    std::vector<dsv4_layer> layers;
    dsv4_dspark ds;

    // Extra rows in every compressor state ring, so a speculative verify's
    // REJECTED tail cannot alias a row the next pass still reads (a rejected
    // write at position q and a live read at position p are less than
    // window+max_draft apart, so widening the modulus by max_draft decouples
    // them and rollback needs no restore at all).
    int64_t state_extra = 0;

    // geometry
    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;
    int64_t ring_raw = 0;
    int64_t n_csa_rows = 0;
    int64_t n_hca_rows = 0;

    // Whether every slot carries the rewind-checkpoint shadow rings, so
    // TSGgml_Dsv4Truncate can rewind past the raw ring's own span. V4.1 only,
    // and TS_DSV41_REWIND_CHECKPOINT=0 turns it off (the truncate then only
    // reaches back as far as the live ring, and refuses beyond it).
    bool rewind_cp = false;
    // Positions a truncation may rewind past, counted from the reference state
    // it rewinds from. The raw ring holds ring_raw positions and a query at the
    // new head still reads n_swa - 1 of them, so this is the slack between the
    // two. Zero when the architecture has no raw window to preserve.
    int64_t rewind_span = 0;
    // The truncation target must be a multiple of this, so no compression block
    // straddles it and the compressor state ring is never read for a position
    // the rewind dropped. lcm of the per-layer compression ratios.
    int32_t truncate_align = 1;

    // sequence slots (per-request caches); slot 0 is the primary/single-stream
    // slot created at load. All Forward/Reset/NPast calls act on active_slot.
    std::map<int, std::unique_ptr<dsv4_slot>> slots;
    dsv4_slot * active_slot = nullptr;
    int next_slot_id = 0;

    // final position of the current TSGgml_Dsv4Forward call; keeps every
    // chunk of one prefill on one graph shape (bucketing + indexer skip)
    int64_t pos_end_hint = 0;

    // flash attention (F16 masks, fused kernel). Probed at load; TS_DSV4_FA=0 forces off.
    bool flash_attn = false;

    // ggml_dsv4_hc_pre/post have a kernel on this backend. Probed at load; when
    // false the graph builds the equivalent batched mul_mat instead of letting
    // the scheduler bounce the residual stream through the CPU backend.
    bool hc_native = true;

    // RoPE cos/sin tables for the fused table-driven kernels, generated by
    // running ggml's own rope over (1,0) unit pairs so YaRN semantics and the
    // attn_factor scaling are baked in exactly. [0]=raw layers, [1]=compress
    // layers. Device copies live in the cache contexts, which Reset() clears —
    // the kept host copies are re-uploaded afterwards.
    std::vector<float> rope_tab_host[2];
    ggml_tensor * rope_tab_dev[2][MAX_GPUS + 1] = {};

    // LRU cache of built+allocated graphs keyed by shape signature. Each entry
    // owns its scheduler (and therefore its allocation), so alternating shapes
    // (decode phases, prefill chunk parity) reuse their graphs without
    // re-building, re-allocating, or thrashing ggml-cuda's captured graphs.
    std::list<std::unique_ptr<graph_build_result>> graph_cache;
    int graph_cache_cap = 12;

    // scratch for logits
    std::vector<float> logits;

    bool gather_env = true; // TS_DSV4_GATHER=0 disables decode index-gather
    bool compact_raw_gather = false; // V4.1 opt-in; compact raw rows before cross-GPU gather

    ~dsv4_model()
    {
        engram_warm_stop.store(true, std::memory_order_relaxed);
        if (engram_warm_thread.joinable()) engram_warm_thread.join();
        graph_cache.clear();
        engram_io.reset();
        moe_tp.reset();
        vision.reset();
        slots.clear();   // slot buffers must go before the backends below
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (c_buf[i]) ggml_backend_buffer_free(c_buf[i]);
            if (c_ctx[i]) ggml_free(c_ctx[i]);
            if (w_buf[i]) ggml_backend_buffer_free(w_buf[i]);
            if (w_ctx[i]) ggml_free(w_ctx[i]);
        }
        if (!mmap_addrs.empty())
        {
            // Expert ranges inside the mappings may be cudaHostRegister'ed
            // (host_pin_range at load); munmap of still-registered pages
            // leaves them pinned in the driver forever, so unregister first.
            tsg::host_pin_release_all();
            for (ggml_backend_buffer_t b : mmap_bufs)
                if (b) ggml_backend_buffer_free(b); // from_ptr: frees the wrapper only
#if !defined(_WIN32)
            for (size_t i = 0; i < mmap_addrs.size(); i++)
                if (mmap_addrs[i]) munmap(mmap_addrs[i], mmap_sizes[i]);
#endif
        }
        for (int i = 0; i < MAX_GPUS; i++)
            if (ts_backends[i]) ggml_backend_free(ts_backends[i]);
        for (int i = 0; i < n_backends; i++)
            if (backends[i]) ggml_backend_free(backends[i]);
        // After the CPU backend, which holds a borrowed pointer to it.
        if (cpu_threadpool) ggml_threadpool_free(cpu_threadpool);
    }
};

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

// A graph owns its scheduler arena and may capture addresses from multiple
// slots. Scheduler destruction itself does not drain submitted work, so wait
// before releasing an arena used by a V4 asynchronous prefill microbatch.
static void dsv4_drop_slot_graphs(dsv4_model & model, int slot_id)
{
    for (auto it = model.graph_cache.begin(); it != model.graph_cache.end(); )
    {
        bool hit = (*it)->slot_id == slot_id;
        for (const auto & slot : (*it)->bd)
            if (slot.slot_id == slot_id) { hit = true; break; }
        if (hit)
        {
            if ((*it)->sched) ggml_backend_sched_synchronize((*it)->sched);
            it = model.graph_cache.erase(it);
        }
        else ++it;
    }
}

static bool gguf_get_f32_key(gguf_context * ctx, const char * key, float * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    *out = gguf_get_val_f32(ctx, id);
    return true;
}

static bool gguf_get_u32_key(gguf_context * ctx, const char * key, int32_t * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(ctx, id))
    {
        case GGUF_TYPE_UINT32: *out = (int32_t) gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(ctx, id); return true;
        case GGUF_TYPE_UINT64: *out = (int32_t) gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT64:  *out = (int32_t) gguf_get_val_i64(ctx, id); return true;
        case GGUF_TYPE_UINT16: *out = (int32_t) gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_INT16:  *out = (int32_t) gguf_get_val_i16(ctx, id); return true;
        case GGUF_TYPE_UINT8:  *out = (int32_t) gguf_get_val_u8(ctx, id); return true;
        case GGUF_TYPE_INT8:   *out = (int32_t) gguf_get_val_i8(ctx, id); return true;
        default: return false;
    }
}

static bool gguf_get_bool_key(gguf_context * ctx, const char * key, bool * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    *out = gguf_get_val_bool(ctx, id);
    return true;
}

static bool gguf_get_arr_i32(gguf_context * ctx, const char * key, std::vector<int32_t> & out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return false;
    size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    const void * data = gguf_get_arr_data(ctx, id);
    switch (gguf_get_arr_type(ctx, id))
    {
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT32:
            memcpy(out.data(), data, n * sizeof(int32_t));
            return true;
        default:
            return false;
    }
}

static bool gguf_get_arr_f32(gguf_context * ctx, const char * key, std::vector<float> & out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return false;
    size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    const void * data = gguf_get_arr_data(ctx, id);
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_FLOAT32) return false;
    memcpy(out.data(), data, n * sizeof(float));
    return true;
}

struct shard_files
{
    std::vector<std::string> paths;
};

// Derive split shard paths from the first shard: "...-00001-of-000NN.gguf"
static shard_files resolve_shards(const std::string & first, int split_count)
{
    shard_files res;
    if (split_count <= 1)
    {
        res.paths.push_back(first);
        return res;
    }

    const std::string marker = "-00001-of-";
    size_t pos = first.find(marker);
    if (pos == std::string::npos)
    {
        res.paths.push_back(first);
        return res;
    }

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

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static ggml_tensor * dsv4_create_weight(
    dsv4_model & m,
    std::map<std::string, tensor_source> & sources,
    int device,
    const char * fmt, ...)
{
    char name[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(name, sizeof(name), fmt, args);
    va_end(args);

    auto it = sources.find(name);
    if (it == sources.end())
    {
        fprintf(stderr, "[dsv4] missing tensor: %s\n", name);
        return nullptr;
    }
    const tensor_source & src = it->second;
    ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src.type, src.ne[0], src.ne[1], src.ne[2], src.ne[3]);
    ggml_set_name(t, name);
    return t;
}

static size_t dsv4_file_size(const char * path)
{
    FILE * f = fopen(path, "rb");
    if (!f) return (size_t) -1;
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    long long sz = _ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    off_t sz = ftello(f);
#endif
    fclose(f);
    return sz < 0 ? (size_t) -1 : (size_t) sz;
}

// A truncated shard - an interrupted download or copy is the usual cause -
// otherwise surfaces as a "short read" tens of seconds into loading, long
// after the split has committed every GPU's weight buffer, which reads as a
// bug in the loader rather than a bad file. The header says exactly how many
// bytes of tensor data the shard should hold, so check it against the file
// before allocating anything.
static bool dsv4_check_shard_complete(const char * path, gguf_context * g, ggml_context * meta)
{
    const size_t fsz = dsv4_file_size(path);
    if (fsz == (size_t) -1) return true; // unreadable size: let the read path report it

    const size_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    size_t needed = data_off;
    const char * last = nullptr;
    for (int64_t ti = 0; ti < n_tensors; ti++)
    {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        const size_t end = data_off + gguf_get_tensor_offset(g, ti) + ggml_nbytes(mt);
        if (end > needed) { needed = end; last = name; }
    }
    if (needed <= fsz) return true;

    fprintf(stderr, "[dsv4] %s is incomplete: the file is %zu bytes but its %" PRId64 " tensors need %zu "
                    "(%.2f GiB missing; %s is the last one). Re-download this file.\n",
            path, fsz, n_tensors, needed, (needed - fsz) / (1024.0 * 1024.0 * 1024.0),
            last ? last : "?");
    return false;
}

// Memory this process is actually allowed to keep resident: the cgroup limit
// when the process runs under one (containers — /proc/meminfo shows the HOST
// there and is off by an order of magnitude), MemTotal otherwise. 0 = unknown.
static size_t dsv4_host_mem_allowance()
{
    size_t limit = 0;
#ifdef __linux__
    if (FILE * f = fopen("/sys/fs/cgroup/memory.max", "r"))   // cgroup v2
    {
        char v[64] = { 0 };
        if (fscanf(f, "%63s", v) == 1 && strcmp(v, "max") != 0)
            limit = (size_t) atoll(v);
        fclose(f);
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r"))   // cgroup v1
        {
            long long v = 0;
            if (fscanf(f, "%lld", &v) == 1 && v > 0 && v < (1ll << 62))
                limit = (size_t) v;
            fclose(f);
        }
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/proc/meminfo", "r"))
        {
            long long kb = 0;
            if (fscanf(f, "MemTotal: %lld kB", &kb) == 1 && kb > 0)
                limit = (size_t) kb * 1024ull;
            fclose(f);
        }
    }
#endif
    return limit;
}

// How much memory the host can still hand out without evicting something it is
// using, in the same units as dsv4_host_mem_allowance(). Reclaimable page cache
// counts, which is what makes this the right test for "can this mapping stay
// resident": the Engram pages a warm pass faults in ARE page cache. Returns 0
// when the platform does not report it, meaning "unknown, do not block on it".
static size_t dsv4_host_mem_available()
{
#ifdef __linux__
    if (FILE * f = fopen("/proc/meminfo", "r"))
    {
        char line[256];
        while (fgets(line, sizeof(line), f))
        {
            long long kb = 0;
            if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1 && kb > 0)
            {
                fclose(f);
                return (size_t) kb * 1024ull;
            }
        }
        fclose(f);
    }
#endif
    return 0;
}

// Map shard `si` read-only and wrap the mapping in a CPU-backend buffer so
// host-resident weights can point straight into the file. Returns the buffer
// (cached in m.mmap_bufs) or null; every failure is non-fatal — the caller
// falls back to the allocate+copy path.
static ggml_backend_buffer_t dsv4_mmap_shard(dsv4_model & m, const shard_files & shards, int si)
{
#if defined(_WIN32)
    (void) m; (void) shards; (void) si;
    return nullptr;
#else
    if (si < 0 || si >= (int) shards.paths.size()) return nullptr;
    if (m.mmap_bufs.size() < shards.paths.size())
    {
        m.mmap_addrs.resize(shards.paths.size(), nullptr);
        m.mmap_sizes.resize(shards.paths.size(), 0);
        m.mmap_bufs.resize(shards.paths.size(), nullptr);
    }
    if (m.mmap_bufs[si]) return m.mmap_bufs[si];

    const char * path = shards.paths[si].c_str();
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
    {
        close(fd);
        return nullptr;
    }
    void * addr = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); // the mapping keeps its own reference
    if (addr == MAP_FAILED) return nullptr;

    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(addr, (size_t) st.st_size);
    if (!buf)
    {
        munmap(addr, (size_t) st.st_size);
        return nullptr;
    }
    // WEIGHTS, not the default ANY: ggml_backend_sched only pins a node to the
    // backend that owns an input when that input's buffer is marked as weights
    // (ggml-backend.cpp, "assign nodes that use weights to the backend of the
    // weights"). Left at ANY, an offloaded layer's mul_mat_id is assigned to the
    // accelerator instead, and the scheduler copies the whole 2.9 GiB expert
    // block over the bus once per token — measured 33 s/token against 0.19 s
    // with the flag set.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.mmap_addrs[si] = addr;
    m.mmap_sizes[si] = (size_t) st.st_size;
    m.mmap_bufs[si] = buf;
    return buf;
#endif
}

// One chunk of one weight tensor: read [file_off, file_off+len) from its
// shard and land it at tensor_off inside the (already allocated) tensor.
struct load_job
{
    ggml_tensor * t;
    int shard;
    size_t file_off;
    size_t tensor_off;
    size_t len;
};

// Stream every job through a pool of reader threads. Model files commonly sit
// on slow or network-backed storage where one reader stream is the whole load
// time (a single thread also serializes disk reads against H2D copies).
// Each worker owns its FILE* per shard and its staging buffer, and
// ggml_backend_tensor_set copies on cudaStreamPerThread, so reads and uploads
// to all GPUs proceed concurrently. Jobs are handed out in file order per
// shard to keep the concurrent streams roughly sequential for readahead.
//
// PAGE CACHE (TS_DSV4_LOAD_DROP_CACHE). The loader never reads an uploaded
// chunk's bytes again, but it does read the host-mapped weights next (the expert
// prefault, the Engram warm) and serves them from the page cache afterwards. When
// the upload plus those mapped bytes cannot fit the host allowance (the cgroup
// limit: page cache is charged to it), every later read competes with reclaim.
// The seven-A40 lane is that case: 263.0 GiB uploaded, then 48.2 GiB of experts and
// 103 GiB of Engram tables read through the mapping, 414 GiB into a 326.9 GiB
// cgroup. Its load logged the expert prefault at 0.37 GiB/s and the Engram warm
// at 0.33 GiB/s, while the same page walks measured 0.62-0.74 GiB/s on that VM
// with the cgroup about half full. So by default each consumed chunk's page
// cache is dropped exactly when upload + mapped + 8 GiB exceeds the allowance,
// and kept otherwise (and whenever the allowance is unknown): an unconditional
// drop would make every reload of a GPU-resident checkpoint cold. Dropping costs
// 5.9-7.3 ms per resident 64 MiB chunk on that mount (POSIX_FADV_DONTNEED,
// measured with GgmlOpsDsv4FileWarmBench --drop-cost), ~25-30 s of thread time
// for 263 GiB. It cannot speed the upload itself, which already runs at the
// storage rate; the benefit expected is on the later stages, and whether it
// outweighs the drop cost is settled by comparing a cold load against =0.
// TS_DSV4_LOAD_DROP_CACHE=0 never drops, =1 always drops. Cache the loader does
// NOT own is left alone.
static bool dsv4_upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads,
                                 size_t mmap_weight_bytes)
{
    std::sort(jobs.begin(), jobs.end(), [](const load_job & a, const load_job & b)
    {
        if (a.shard != b.shard) return a.shard < b.shard;
        return a.file_off < b.file_off;
    });

    if (n_threads > (int) jobs.size()) n_threads = (int) jobs.size();
    if (n_threads < 1) n_threads = 1;

    std::atomic<bool> failed(false);
    // See PAGE CACHE above: automatic unless TS_DSV4_LOAD_DROP_CACHE is set.
    const bool drop_cache = [&]()
    {
        uint64_t upload_bytes = 0;
        for (const load_job & j : jobs) upload_bytes += j.len;
        const uint64_t allowance = dsv4_host_mem_allowance();
        const tsg_dsv4::drop_cache_decision d = tsg_dsv4::decide_drop_cache(
            getenv("TS_DSV4_LOAD_DROP_CACHE"), upload_bytes, mmap_weight_bytes, allowance);
        if (upload_bytes > 0)
            fprintf(stderr, "[dsv4] load page cache: %s\n",
                    tsg_dsv4::describe_drop_cache(d, upload_bytes, mmap_weight_bytes, allowance).c_str());
        return d.drop;
    }();

    // Each thread walks ONE CONTIGUOUS RUN of the sorted job list instead of
    // taking every n_threads'th job from a shared cursor.
    //
    // Why it matters: jobs are sorted by (shard, file_off), so a shared cursor
    // makes every descriptor read a chunk and then jump n_threads * chunk bytes -
    // 1 GiB at the defaults. Readahead is per-descriptor, so none of the streams
    // is sequential. Measured on the MooseFS mount, 16 threads, 64 MiB reads,
    // same bytes: strided 1.16-1.28 GiB/s, contiguous 2.16-2.17 GiB/s. The API is
    // not the difference (fread and pread came out within noise of each other in
    // both orders); the order is. TensorSharp.Runtime/GgufReader.cs:330 records
    // the same finding for the managed prefault ("~3x slower on MooseFS").
    //
    // Ranges are split by BYTES, not by job count, because a tensor's last chunk
    // is a partial one and counting jobs would hand some threads more data than
    // others. A thread that finishes early steals from the BACK of the furthest
    // behind range, so the victim keeps reading forwards.
    bool contiguous = true;
    if (const char * e = getenv("TS_DSV4_LOAD_CONTIGUOUS")) contiguous = atoi(e) != 0;

    // `end` is atomic because a stealer shrinks another range's end while that
    // range's owner is reading it to decide whether its own claim is in bounds.
    struct range { std::atomic<size_t> next; std::atomic<size_t> end; };
    std::vector<range> ranges((size_t) n_threads);
    {
        size_t total = 0;
        for (const load_job & j : jobs) total += j.len;
        size_t at = 0, acc = 0;
        for (int k = 0; k < n_threads; k++)
        {
            const size_t target = (size_t) ((double) total * (k + 1) / n_threads);
            const size_t begin = at;
            while (at < jobs.size() && (acc < target || k == n_threads - 1))
            {
                acc += jobs[at].len;
                at++;
                if (k == n_threads - 1 ? at == jobs.size() : acc >= target) break;
            }
            ranges[(size_t) k].next.store(begin, std::memory_order_relaxed);
            ranges[(size_t) k].end.store(at, std::memory_order_relaxed);
        }
        // Any tail left by rounding belongs to the last range.
        ranges[(size_t) n_threads - 1].end.store(jobs.size(), std::memory_order_relaxed);
    }
    std::atomic<size_t> cursor(0);

    // Hand out the next job for thread k: its own range first, then, once that is
    // exhausted, the BACK of whichever range has the most left - so the thread it
    // steals from keeps reading forwards. Stealing is rare (ranges are equal by
    // bytes) and is serialized on one mutex rather than raced with atomics.
    std::mutex steal_mu;
    auto claim = [&](int k, size_t & out) -> bool
    {
        const size_t own = ranges[(size_t) k].next.fetch_add(1, std::memory_order_relaxed);
        if (own < ranges[(size_t) k].end.load(std::memory_order_acquire)) { out = own; return true; }

        std::lock_guard<std::mutex> lock(steal_mu);
        int victim = -1;
        size_t most = 0;
        for (int v = 0; v < n_threads; v++)
        {
            const size_t next = ranges[(size_t) v].next.load(std::memory_order_relaxed);
            const size_t end = ranges[(size_t) v].end.load(std::memory_order_relaxed);
            const size_t left = end > next ? end - next : 0;
            if (left > most) { most = left; victim = v; }
        }
        // Require at least TWO jobs left, so the index taken here (end-1) is always
        // strictly ahead of the index the owner will claim next. That is what makes
        // a steal and an owner's fetch_add unable to name the same job.
        if (victim < 0 || most < 2) return false;
        const size_t take = ranges[(size_t) victim].end.load(std::memory_order_relaxed) - 1;
        ranges[(size_t) victim].end.store(take, std::memory_order_release);
        out = take;
        return true;
    };
    // Where a slow load actually goes. Summed over threads, so the totals exceed
    // the wall clock by roughly the thread count when both stages are busy; the
    // RATIO between them is what says whether to tune the reader or the uploader.
    std::atomic<uint64_t> read_ns(0), set_ns(0), read_bytes(0);
    const bool perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e && atoi(e) > 0; }();

    auto worker = [&](int thread_index)
    {
        std::vector<FILE *> files(shards.paths.size(), nullptr);
        std::vector<uint8_t> staging;
        while (!failed.load(std::memory_order_relaxed))
        {
            size_t i;
            if (contiguous)
            {
                if (!claim(thread_index, i)) break;
            }
            else
            {
                i = cursor.fetch_add(1, std::memory_order_relaxed);
                if (i >= jobs.size()) break;
            }
            const load_job & j = jobs[i];

            FILE *& f = files[j.shard];
            if (!f)
            {
                f = fopen(shards.paths[j.shard].c_str(), "rb");
                if (!f)
                {
                    fprintf(stderr, "[dsv4] cannot open %s\n", shards.paths[j.shard].c_str());
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            if (staging.size() < j.len) staging.resize(j.len);

#if defined(_WIN32)
            _fseeki64(f, (long long) j.file_off, SEEK_SET);
#else
            fseeko(f, (off_t) j.file_off, SEEK_SET);
#endif
            auto t_read0 = std::chrono::steady_clock::now();
            const size_t got = fread(staging.data(), 1, j.len, f);
            auto t_read1 = std::chrono::steady_clock::now();
            if (got != j.len)
            {
                // Not a truncated file (dsv4_check_shard_complete already ruled
                // that out), so name the exact read that failed.
                fprintf(stderr, "[dsv4] short read for %s: %zu bytes at offset %zu of %s\n",
                        j.t->name, j.len, j.file_off, shards.paths[j.shard].c_str());
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            ggml_backend_tensor_set(j.t, staging.data(), j.tensor_off, j.len);
            // After the device has it, this range is dead weight in the page
            // cache. Dropping it keeps the reader threads out of reclaim.
            // Only Linux can drop one range: macOS has no POSIX_FADV_DONTNEED
            // (the header does not even declare it, so the old !_WIN32 guard
            // did not compile there) and offers whole-descriptor F_NOCACHE
            // instead. Anywhere else the request has no effect and says so.
            if (drop_cache)
            {
#if defined(__linux__)
                posix_fadvise(fileno(f), (off_t) j.file_off, (off_t) j.len, POSIX_FADV_DONTNEED);
#elif defined(__APPLE__)
                // Idempotent and per-descriptor: from here this handle's reads
                // bypass the page cache entirely.
                fcntl(fileno(f), F_NOCACHE, 1);
#else
                static std::once_flag warned;
                std::call_once(warned, []() {
                    fprintf(stderr, "[dsv4] TS_DSV4_LOAD_DROP_CACHE has no effect on this platform; "
                                    "the checkpoint's pages stay in the page cache\n");
                });
#endif
            }
            if (perf)
            {
                auto t_set1 = std::chrono::steady_clock::now();
                read_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_read1 - t_read0).count(), std::memory_order_relaxed);
                set_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_set1 - t_read1).count(), std::memory_order_relaxed);
                read_bytes.fetch_add(j.len, std::memory_order_relaxed);
            }
        }
        for (FILE * f : files) if (f) fclose(f);
    };

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) pool.emplace_back(worker, i);
    for (auto & th : pool) th.join();
    if (perf && read_bytes.load() > 0)
    {
        const double gib = read_bytes.load() / 1073741824.0;
        const double rs = read_ns.load() / 1e9, ss = set_ns.load() / 1e9;
        fprintf(stderr, "[dsv4] load split over %d %s thread(s): file reads %.1fs (%.2f GiB/s per thread), "
                "host->device %.1fs (%.2f GiB/s per thread), %.1f%% of thread time in reads\n",
                n_threads, contiguous ? "contiguous" : "interleaved",
                rs, rs > 0 ? gib / rs : 0.0, ss, ss > 0 ? gib / ss : 0.0,
                100.0 * rs / std::max(1e-9, rs + ss));
    }
    return !failed.load();
}

// TS_DSV4_LOAD_THREADS: reader threads for every whole-file pass of a load
// (the weight upload, the host-expert prefault and the Engram warm).
static int dsv4_load_thread_count()
{
    int load_threads = 16;
    if (const char * e = getenv("TS_DSV4_LOAD_THREADS")) { int v = atoi(e); if (v > 0) load_threads = v; }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0 && load_threads > (int) hw) load_threads = (int) hw;
    return load_threads;
}

// TS_DSV4_WARM_PREAD: whether the load-time warm passes read with pread (default)
// or walk the mapping page by page (=0). See dsv4_file_warm.h.
static bool dsv4_warm_pread()
{
    const char * value = getenv("TS_DSV4_WARM_PREAD");
    bool invalid = false;
    const bool pread_warm = tsg_dsv4::resolve_warm_pread(value, &invalid);
    if (invalid)
    {
        static std::once_flag warned;
        std::call_once(warned, [value]() {
            fprintf(stderr, "[dsv4] TS_DSV4_WARM_PREAD=%s is not 0 or 1; warming with pread (the default)\n", value);
        });
    }
    return pread_warm;
}

// Where a host tensor served from a shard mapping lives in its file: the shard
// (= mapping) index, the file offset and the mapped address. False when the
// tensor is not in one of the mappings (a private copy, e.g. on Windows).
static bool dsv4_mapped_file_range(const dsv4_model & m, const ggml_tensor * t, tsg_dsv4::file_warm_range & out)
{
    if (t == nullptr || t->buffer == nullptr || t->data == nullptr) return false;
    const auto found = std::find(m.mmap_bufs.begin(), m.mmap_bufs.end(), t->buffer);
    if (found == m.mmap_bufs.end()) return false;
    const size_t si = (size_t) (found - m.mmap_bufs.begin());
    if (si >= m.mmap_addrs.size() || m.mmap_addrs[si] == nullptr) return false;
    out.file = (int) si;
    out.offset = (uint64_t) ((const char *) t->data - (const char *) m.mmap_addrs[si]);
    out.bytes = ggml_nbytes(t);
    out.mapped = t->data;
    return true;
}

// Load stage: fault the mmapped host-resident weights (the cpu_moe experts) in
// before the model serves. Returns false only on a load error.
static bool dsv4_prefault_host_experts(dsv4_model & loaded, const shard_files & shards, int load_threads)
{
    dsv4_model * const m = &loaded;
    const dsv4_hparams & hp = m->hp;
    const int n_gpu = m->n_gpu;
    // Warm the mmapped experts with the same parallelism the copy path had.
    // Lazy faulting would make the first prompt pay for the whole read at
    // single-stream storage speed, mid-generation. Only when they actually
    // fit: warming 137 GiB into an 87 GiB allowance just evicts itself (and
    // everything else) for nothing, so there lazy-on-demand IS the plan.
    //
    // The bytes are read with pread in 64 MiB blocks, one contiguous run of the
    // expert ranges per thread (TS_DSV4_LOAD_THREADS), skipping blocks that
    // mincore already reports resident. The page-touch walk this replaced
    // faults 4 KiB at a time, and on a network filesystem every fault is a
    // synchronous read capped at the mount's readahead: 129.7 s for 48.2 GiB on
    // the seven-A40 lane. MADV_WILLNEED, POSIX_FADV_WILLNEED and readahead(2)
    // are capped the same way there (0.20% of a range resident); do not retry
    // them. TS_DSV4_WARM_PREAD=0 restores the walk exactly.
    if (m->mmap_weight_bytes > 0)
    {
        const size_t allow = dsv4_host_mem_allowance();
        const size_t headroom = (size_t) 8 * 1024 * 1024 * 1024;
        if (allow == 0 || m->mmap_weight_bytes + headroom <= allow)
        {
            const auto t_warm = std::chrono::steady_clock::now();
            std::vector<std::pair<const volatile char *, size_t>> ranges;
            std::vector<tsg_dsv4::file_warm_range> file_ranges;
            size_t warm_bytes = 0;
            ggml_context * hctx = m->w_ctx[n_gpu];
            for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
            {
                tsg_dsv4::file_warm_range range;
                if (!dsv4_mapped_file_range(*m, t, range)) continue;
                // Engram reads only 24 rows per token; prefaulting its entire
                // hundred-billion-element table wastes I/O and evicts useful pages.
                if (hp.v41 && strstr(t->name, ".engram_embd.") != nullptr) continue;
                ranges.emplace_back((const volatile char *) t->data, ggml_nbytes(t));
                file_ranges.push_back(range);
                warm_bytes += ggml_nbytes(t);
            }
            if (!dsv4_warm_pread())
            {
                tsg_dsv4::touch_prefault_spans(ranges, load_threads);
                fprintf(stderr, "[dsv4] prefaulted %.1f GiB of mmapped host experts in %.1fs (page-touch walk, TS_DSV4_WARM_PREAD=0)\n",
                        warm_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count());
                return true;
            }
            tsg_dsv4::file_warm_options options;
            options.threads = load_threads;
            // pread fills the page cache but not this process's page tables,
            // which the walk also populated. The first prefill reads the experts
            // densely, so each cached block is also mapped here (one read per
            // page, minor faults only: ~0.005 s/GiB at 16 threads, measured).
            options.populate = true;
            const tsg_dsv4::file_warm_result r = tsg_dsv4::warm_file_ranges(shards.paths, file_ranges, options);
            if (!r.ok)
            {
                fprintf(stderr, "[dsv4] prefaulting the mmapped host experts failed: %s\n", r.error.c_str());
                return false;
            }
            fprintf(stderr, "[dsv4] prefaulted %.1f GiB of mmapped host experts in %.1fs "
                            "(pread, %d threads: %.1f GiB read, %.1f GiB already resident)\n",
                    warm_bytes / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(),
                    r.threads, r.bytes_read / 1073741824.0, r.bytes_resident / 1073741824.0);
        }
    }
    return true;
}

// Load stage: optionally page-lock the offloaded experts.
static void dsv4_pin_host_experts(dsv4_model & loaded, int n_cpu_moe)
{
    dsv4_model * const m = &loaded;
    const dsv4_hparams & hp = m->hp;
    // Page-locking the offloaded experts buys nothing here, so it is off unless
    // TS_HOST_MOE_PIN=1 asks for it. build_moe_host assigns EVERY node of an
    // offloaded layer's routed experts to the CPU backend (mul_mat_id up, gate
    // and down, clamp, swiglu, mul and the expert adds), and
    // ggml_backend_sched never overrides a user assignment, so its op_offload
    // rule never streams the expert weights to a GPU: only [n_embd, n_tokens]
    // activations cross the bus. A pinned expert is therefore never a DMA
    // source. Registering them cost 20.4 s for 48.2 GiB on the seven-A40 lane
    // (--n-cpu-moe 6) and made those pages unevictable inside the cgroup the
    // page cache lives in. ggml_ops_moe.cpp, which really streams offloaded
    // experts for the other MoE architectures, keeps pinning by default.
    if (n_cpu_moe > 0 && !tsg_dsv4::dsv4_host_expert_pin_requested(getenv("TS_HOST_MOE_PIN")))
    {
        fprintf(stderr, "[dsv4] host experts of %d offloaded layer(s) run on the CPU backend and stay pageable "
                        "(TS_HOST_MOE_PIN=1 registers them with the GPU driver)\n", n_cpu_moe);
        return;
    }
    // Not when the mmapped experts outweigh the host allowance: registering
    // faults the pages in at storage speed (minutes on a network FS) and
    // every pinned page is one the kernel can no longer evict, which is
    // exactly the headroom an over-committed page cache lives on. Failures
    // (no budget, driver refusal) leave the pages pageable; see host_pin_range.
    const bool experts_over_allowance = m->mmap_weight_bytes > 0 &&
        [&]{ const size_t a = dsv4_host_mem_allowance();
             return a > 0 && m->mmap_weight_bytes + (size_t) 8 * 1024 * 1024 * 1024 > a; }();
    if (n_cpu_moe > 0 && !experts_over_allowance)
    {
        const auto t_pin = std::chrono::steady_clock::now();
        std::size_t pinned = 0;
        for (int il = 0; il < hp.n_layer; il++)
        {
            const dsv4_layer & L = m->layers[il];
            if (!L.cpu_moe) continue;
            for (ggml_tensor * t : { L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps })
            {
                if (t == nullptr || t->data == nullptr) continue;
                if (tsg::host_pin_range(t->data, ggml_nbytes(t)))
                    pinned += ggml_nbytes(t);
            }
        }
        if (pinned > 0)
        {
            fprintf(stderr, "[dsv4] page-locked %.1f GiB of host experts in %.1fs (TS_HOST_MOE_PIN=1; they compute on the CPU backend, "
                            "so this speeds up no transfer)\n",
                    pinned / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pin).count());
        }
    }
}

// Load stage (V4.1): warm the host-mapped Engram tables, synchronously or on a
// background thread, and apply the sparse-read mapping advice once warming is
// done. Returns false only on a load error.
static bool dsv4_warm_engram(dsv4_model & loaded, const shard_files & shards,
                             bool engram_random_advice, bool engram_random_override)
{
    dsv4_model * const m = &loaded;
    // Sparse-read advice turns off the kernel's readahead, which is exactly
    // what a whole-table warm pass depends on, so this runs only once the
    // warming (synchronous or background) is finished. Never alter another
    // host tensor's entire shard, or an allocation that is not one of our
    // mmaps.
    dsv4_model * const model_ptr = m;
    auto apply_engram_advice = [model_ptr, engram_random_advice, engram_random_override]()
    {
        if (!engram_random_advice)
        {
            if (!model_ptr->engram_on_device)
                fprintf(stderr, "[dsv41] Engram mmap advice: default (%s)\n",
                    engram_random_override ? "override=0" : "automatic: one I/O thread");
            return;
        }
        size_t accepted = 0, unsupported = 0, skipped = 0, failed = 0;
        for (const auto & layout : model_ptr->engram.layers)
        {
            const auto * table = model_ptr->layers[layout.id].engram_embd;
            const auto found = std::find(model_ptr->mmap_bufs.begin(), model_ptr->mmap_bufs.end(), table->buffer);
            if (!table->buffer || found == model_ptr->mmap_bufs.end()) { ++skipped; continue; }
            const size_t index = size_t(found - model_ptr->mmap_bufs.begin());
            const auto result = tsg_dsv41::advise_engram_random(model_ptr->mmap_addrs[index], model_ptr->mmap_sizes[index],
                table->data, ggml_nbytes(table));
            if (result.status == tsg_dsv41::mapped_advice_status::applied) ++accepted;
            else if (result.status == tsg_dsv41::mapped_advice_status::unsupported) ++unsupported;
            else if (result.status == tsg_dsv41::mapped_advice_status::empty) ++skipped;
            else
            {
                ++failed;
                fprintf(stderr, "[dsv41] Engram random advice was not applied to layer %d (error %d); continuing\n",
                    layout.id, result.error);
            }
        }
        fprintf(stderr, "[dsv41] Engram mmap advice: RANDOM (%s; accepted=%zu, unsupported=%zu, skipped=%zu, failed=%zu)\n",
            engram_random_override ? "override=1" : "automatic: parallel I/O", accepted, unsupported, skipped, failed);
    };
    bool advice_deferred = false;

    // A host-resident Engram table is read 24 scattered rows per token per
    // table. Each row that is not page cache is one storage round trip, and
    // on a network filesystem that is ~1 ms: input preparation for a
    // 1024-token prefill chunk measured 1.44 s cold against 0.04-0.09 s
    // warm on the eight-A40 VM's Q4_K_M checkpoint (prefill 201-255 ->
    // 363-381 tok/s, decode 23-24 -> 29 tok/s).
    //
    // So warming is the default whenever the pages can actually STAY
    // resident. It is minutes of I/O, so the automatic form runs on its own
    // thread after the model is serving rather than holding up load;
    // TS_DSV41_ENGRAM_WARM=1 keeps the documented synchronous behaviour and
    // =0 turns warming off.
    const char * warm_opt = m->engram_on_device ? nullptr : getenv("TS_DSV41_ENGRAM_WARM");
    const int warm_mode = m->engram_on_device ? 0 : (warm_opt ? atoi(warm_opt) : -1);
    if (warm_mode != 0)
    {
        size_t bytes = 0;
        for (const auto & layout : m->engram.layers) bytes += ggml_nbytes(m->layers[layout.id].engram_embd);
        const size_t headroom = (size_t) 8 * 1024 * 1024 * 1024;
        const size_t allowance = dsv4_host_mem_allowance();
        const size_t resident = std::max(bytes, m->mmap_weight_bytes);
        const char * refused = nullptr;
        if (allowance && (resident > allowance || allowance - resident < headroom))
            refused = "host allowance lacks 8 GiB headroom";
        else if (warm_mode < 0)
        {
            // Automatic only: an explicit =1 is the operator's call. Warming
            // pages that the host will immediately evict costs the I/O and
            // buys nothing, so require room for the whole table set now.
            const size_t available = dsv4_host_mem_available();
            if (available && available < bytes + headroom)
                refused = "free host memory would not keep the tables cached";
        }

        // TS_DSV4_WARM_PREAD=0 keeps the page-touch walks (engram_io_pool::warm,
        // 8 MiB chunks from a shared cursor) for both the synchronous and the
        // background warm. A table that is not one of the mappings is a private
        // copy already in memory, so the pread form has nothing to read for it.
        const bool use_pread = refused == nullptr && dsv4_warm_pread();
        const int pread_threads = dsv4_load_thread_count();
        std::vector<tsg_dsv4::file_warm_range> table_ranges;
        if (use_pread)
        {
            for (const auto & layout : m->engram.layers)
            {
                tsg_dsv4::file_warm_range range;
                if (dsv4_mapped_file_range(*m, m->layers[layout.id].engram_embd, range)) table_ranges.push_back(range);
            }
        }

        if (refused)
        {
            fprintf(stderr, "[dsv41] Engram warming skipped: %s\n", refused);
        }
        else if (warm_mode > 0 && use_pread)
        {
            // Same pread helper as the expert prefault: a whole table is one
            // contiguous range of its shard, split into one run per thread.
            // 311 s for the Q4_K_M tables with the page-touch walk below on the
            // seven-A40 lane, where every fault was a 128 KiB synchronous read.
            const auto start = std::chrono::steady_clock::now();
            fprintf(stderr, "[dsv41] warming %.2f GiB of Engram pages with %d pread threads...\n",
                bytes / 1073741824.0, pread_threads);
            tsg_dsv4::file_warm_options options;
            options.threads = pread_threads;
            const tsg_dsv4::file_warm_result r = tsg_dsv4::warm_file_ranges(shards.paths, table_ranges, options);
            if (!r.ok)
            {
                fprintf(stderr, "[dsv41] warming the Engram tables failed: %s\n", r.error.c_str());
                return false;
            }
            fprintf(stderr, "[dsv41] warmed %.2f GiB of Engram pages in %.2fs (%d pread threads: %.2f GiB read, "
                "%.2f GiB already resident)\n",
                bytes / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                r.threads, r.bytes_read / 1073741824.0, r.bytes_resident / 1073741824.0);
        }
        else if (warm_mode > 0)
        {
            const auto start = std::chrono::steady_clock::now();
            fprintf(stderr, "[dsv41] warming %.2f GiB of Engram pages with %u I/O threads...\n",
                bytes / 1073741824.0, m->engram_io->threads());
            for (const auto & layout : m->engram.layers)
            {
                auto * table = m->layers[layout.id].engram_embd;
                m->engram_io->warm(table->data, ggml_nbytes(table));
            }
            fprintf(stderr, "[dsv41] warmed %.2f GiB of Engram pages in %.2fs (%u I/O threads)\n",
                bytes / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                m->engram_io->threads());
        }
        else
        {
            // Its own pool, not m->engram_io: that one is what per-token
            // lookups run on, and a warming submission holds it for the
            // length of a slice.
            std::vector<std::pair<void *, size_t>> ranges;
            for (const auto & layout : m->engram.layers)
            {
                auto * table = m->layers[layout.id].engram_embd;
                ranges.emplace_back(table->data, ggml_nbytes(table));
            }
            const unsigned warm_threads = use_pread ? (unsigned) pread_threads
                                                    : std::min<unsigned>(16u, m->engram_io->threads());
            fprintf(stderr, "[dsv41] warming %.2f GiB of Engram pages in the background with %u %s threads; "
                "requests run at page-cache speed once it finishes (TS_DSV41_ENGRAM_WARM=0 disables)\n",
                bytes / 1073741824.0, warm_threads, use_pread ? "pread" : "I/O");
            dsv4_model * model = model_ptr;
            advice_deferred = true;
            std::vector<std::string> paths = shards.paths;
            m->engram_warm_thread = std::thread([model, ranges, paths, table_ranges, use_pread, bytes, warm_threads,
                                                 apply_engram_advice]() {
                try
                {
                    const auto start = std::chrono::steady_clock::now();
                    if (use_pread)
                    {
                        // The stop flag is checked before every 64 MiB block, so
                        // a model freed mid-warm waits at most one block.
                        tsg_dsv4::file_warm_options options;
                        options.threads = (int) warm_threads;
                        options.stop = &model->engram_warm_stop;
                        const tsg_dsv4::file_warm_result r = tsg_dsv4::warm_file_ranges(paths, table_ranges, options);
                        if (r.stopped) return;
                        if (!r.ok)
                            fprintf(stderr, "[dsv41] background Engram warming stopped: %s\n", r.error.c_str());
                        else
                            fprintf(stderr, "[dsv41] warmed %.2f GiB of Engram pages in %.2fs (background, %d pread threads: "
                                "%.2f GiB read, %.2f GiB already resident)\n",
                                bytes / 1073741824.0,
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                                r.threads, r.bytes_read / 1073741824.0, r.bytes_resident / 1073741824.0);
                        apply_engram_advice();
                        return;
                    }
                    tsg_dsv41::engram_io_pool pool(warm_threads);
                    // Slices keep the stop flag responsive: a model freed
                    // mid-warm waits at most one slice, not one table.
                    constexpr size_t slice = (size_t) 1024 * 1024 * 1024;
                    for (const auto & range : ranges)
                    {
                        for (size_t off = 0; off < range.second; off += slice)
                        {
                            if (model->engram_warm_stop.load(std::memory_order_relaxed)) return;
                            pool.warm((const char *) range.first + off, std::min(slice, range.second - off));
                        }
                    }
                    fprintf(stderr, "[dsv41] warmed %.2f GiB of Engram pages in %.2fs (background, %u I/O threads)\n",
                        bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                        warm_threads);
                }
                catch (const std::exception & e)
                {
                    fprintf(stderr, "[dsv41] background Engram warming stopped: %s\n", e.what());
                }
                // Readahead is no longer wanted: from here the table is read
                // a few scattered rows at a time.
                apply_engram_advice();
            });
        }
    }
    if (!advice_deferred) apply_engram_advice();
    return true;
}

// The overlap compressors read a synthetic "before the first block" source
// from the state rings' extra row: kv stays zero (buffer clear), the score
// row must be -inf so the per-element softmax gives those slots zero weight.
static void dsv4_init_slot_state_rows(dsv4_slot & slot)
{
    for (auto & C : slot.layers)
    {
        auto fill_neg_inf_row = [&](ggml_tensor * t)
        {
            if (!t) return;
            std::vector<float> row((size_t) t->ne[0], -INFINITY);
            ggml_backend_tensor_set(t, row.data(), (size_t) (t->ne[1] - 1) * t->nb[1], row.size() * sizeof(float));
        };
        fill_neg_inf_row(C.comp_state_score);
        fill_neg_inf_row(C.lid_state_score);
    }
}

// Clear a slot's caches back to the fresh state (position 0).
static void dsv4_reset_slot(dsv4_slot & slot)
{
    slot.n_past = 0;
    slot.spec_begin = slot.spec_end = -1;
    slot.engram_history.clear();
    for (int d = 0; d <= MAX_GPUS; d++)
        if (slot.buf[d]) ggml_backend_buffer_clear(slot.buf[d], 0);
#if defined(TSG_GGML_TEST_HOOKS)
    dsv4_test_boundary_fault(1);
#endif
    dsv4_init_slot_state_rows(slot);
    slot.v41_failed = false;
    // The shadow rings live in the same buffers and were just zeroed, so the
    // checkpoint they described is gone with them.
    slot.cp_n_past = -1;
}

// Copy the modular ring state into (save) or out of (!save) the slot's shadow
// tensors. Both sides live in the same per-device buffer, so this is a
// device-local copy; the compressed caches are deliberately not touched.
static void dsv4_copy_rewind_rings(dsv4_slot & slot, bool save)
{
    for (auto & C : slot.layers)
    {
        auto move = [&](ggml_tensor * live, ggml_tensor * shadow)
        {
            if (!live || !shadow) return;
            ggml_backend_tensor_copy(save ? live : shadow, save ? shadow : live);
        };
        move(C.raw_k, C.raw_k_cp);
#if defined(TSG_GGML_TEST_HOOKS)
        if (!save) dsv4_test_boundary_fault(2);
#endif
        move(C.comp_state_kv, C.comp_state_kv_cp);
        move(C.comp_state_score, C.comp_state_score_cp);
    }
    for (size_t i = 0; i < slot.ds_k_cp.size(); ++i)
        ggml_backend_tensor_copy(save ? slot.ds_k[i] : slot.ds_k_cp[i],
                                 save ? slot.ds_k_cp[i] : slot.ds_k[i]);
}

// Record the slot's state as the rewind checkpoint. Called at the end of every
// multi-token forward (see dsv4_slot::cp_n_past); a no-op without the shadows.
static void dsv4_checkpoint_slot(const dsv4_model & m, dsv4_slot & slot)
{
    if (!m.rewind_cp || slot.v41_failed) return;
    dsv4_copy_rewind_rings(slot, /*save*/ true);
#if defined(TSG_GGML_TEST_HOOKS)
    dsv41_test_fail("checkpoint", slot.n_past);
#endif
    slot.cp_n_past = slot.n_past;
    // engram_history needs no shadow: it is indexed by absolute position and
    // hash_tokens only ever writes from the current head forward, so its first
    // `target` entries are already what a rewind to `target` wants.
    //
    // What decides whether a later rewind can USE this - the alignment and depth
    // guards, and why each is exactly where it is - lives in dsv41_truncate.h so
    // it can be tested without a model.
}

// Allocate a new sequence slot: per-layer caches + compressor state rings on
// each layer's device. Weights and rope tables are shared across slots.
// Returns nullptr on allocation failure (e.g. VRAM exhausted).
static dsv4_slot * dsv4_slot_alloc(dsv4_model & m)
{
    const dsv4_hparams & hp = m.hp;
    auto slot = std::make_unique<dsv4_slot>();
    slot->id = m.next_slot_id++;
    slot->layers.resize(hp.n_layer);

    for (int d = 0; d <= m.n_gpu; d++)
    {
        // 10 cache tensors per layer, plus the 3 rewind-checkpoint shadows.
        ggml_init_params cp = { (size_t) (hp.n_layer * 13 + m.ds.n_stages * 2 + 32) * ggml_tensor_overhead(), nullptr, true };
        slot->ctx[d] = ggml_init(cp);
        if (!slot->ctx[d]) return nullptr;
    }

    if (m.ds.loaded)
    {
        slot->ds_k.assign(m.ds.n_stages, nullptr);
        if (m.rewind_cp) slot->ds_k_cp.assign(m.ds.n_stages, nullptr);
        for (int s = 0; s < m.ds.n_stages; s++)
        {
            slot->ds_k[s] = ggml_new_tensor_2d(slot->ctx[m.ds.dev], GGML_TYPE_F16,
                                               hp.n_embd_head, m.ring_raw);
            ggml_format_name(slot->ds_k[s], "cache_ds_k.%d.%d", slot->id, s);
            if (m.rewind_cp)
            {
                slot->ds_k_cp[s] = ggml_new_tensor_2d(slot->ctx[m.ds.dev], GGML_TYPE_F16,
                                                      hp.n_embd_head, m.ring_raw);
                ggml_format_name(slot->ds_k_cp[s], "cp_ds_k.%d.%d", slot->id, s);
            }
        }
    }

    for (int il = 0; il < hp.n_layer; il++)
    {
        const int d = m.layers[il].device;
        const int ratio = hp.compress_ratios[il];
        dsv4_slot_layer & C = slot->layers[il];
        ggml_context * ctx = slot->ctx[d];

        const int64_t head = hp.n_embd_head;
        C.raw_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.ring_raw);
        ggml_format_name(C.raw_k, "cache_raw_k.%d.%d", slot->id, il);
        // State rings carry one extra row (index state_size) that stays
        // zero (kv) / -inf (score): the overlap compressor's synthetic
        // "before the first block" source. It is never written by persists
        // (dst = pos % state_size < state_size), so no per-eval zero-row
        // append is needed in the graph.
        if (hp.v41)
        {
            if (m.layers[il].kv_source == il)
            {
                const int64_t rows = ratio == 2 ? m.n_csa_rows : m.n_hca_rows;
                C.csa_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, rows);
                C.lid_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hp.indexer_head_size, rows);
                if (ratio > 1)
                {
                    C.comp_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, ratio + m.state_extra + 1);
                    C.comp_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, ratio + m.state_extra + 1);
                }
            }
            if (m.rewind_cp)
            {
                C.raw_k_cp = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.ring_raw);
                ggml_format_name(C.raw_k_cp, "cp_raw_k.%d.%d", slot->id, il);
                if (C.comp_state_kv)
                {
                    C.comp_state_kv_cp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                        C.comp_state_kv->ne[0], C.comp_state_kv->ne[1]);
                    C.comp_state_score_cp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                        C.comp_state_score->ne[0], C.comp_state_score->ne[1]);
                    ggml_format_name(C.comp_state_kv_cp, "cp_csa_kv.%d.%d", slot->id, il);
                    ggml_format_name(C.comp_state_score_cp, "cp_csa_score.%d.%d", slot->id, il);
                }
            }
        }
        else if (ratio == CSA_RATIO)
        {
            C.csa_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.n_csa_rows);
            ggml_format_name(C.csa_k, "cache_csa_k.%d.%d", slot->id, il);
            C.lid_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hp.indexer_head_size, m.n_csa_rows);
            ggml_format_name(C.lid_k, "cache_lid_k.%d.%d", slot->id, il);
            C.comp_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head, 2 * CSA_RATIO + m.state_extra + 1);
            C.comp_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head, 2 * CSA_RATIO + m.state_extra + 1);
            C.lid_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * hp.indexer_head_size, 2 * CSA_RATIO + m.state_extra + 1);
            C.lid_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * hp.indexer_head_size, 2 * CSA_RATIO + m.state_extra + 1);
            ggml_format_name(C.comp_state_kv, "state_csa_kv.%d.%d", slot->id, il);
            ggml_format_name(C.comp_state_score, "state_csa_score.%d.%d", slot->id, il);
            ggml_format_name(C.lid_state_kv, "state_lid_kv.%d.%d", slot->id, il);
            ggml_format_name(C.lid_state_score, "state_lid_score.%d.%d", slot->id, il);
        }
        else if (ratio == HCA_RATIO)
        {
            C.hca_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.n_hca_rows);
            ggml_format_name(C.hca_k, "cache_hca_k.%d.%d", slot->id, il);
            C.comp_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, HCA_RATIO + m.state_extra + 1);
            C.comp_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, HCA_RATIO + m.state_extra + 1);
            ggml_format_name(C.comp_state_kv, "state_hca_kv.%d.%d", slot->id, il);
            ggml_format_name(C.comp_state_score, "state_hca_score.%d.%d", slot->id, il);
        }
    }

    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (ggml_get_first_tensor(slot->ctx[d]) == nullptr) continue;
        slot->buf[d] = ggml_backend_alloc_ctx_tensors(slot->ctx[d], m.backends[d]);
        if (!slot->buf[d])
        {
            fprintf(stderr, "[dsv4] slot %d cache alloc failed on device %d\n", slot->id, d);
            return nullptr;
        }
        ggml_backend_buffer_clear(slot->buf[d], 0);
    }
    dsv4_init_slot_state_rows(*slot);

    dsv4_slot * raw = slot.get();
    m.slots[slot->id] = std::move(slot);
    return raw;
}

// Build the [n_rot, n_ctx] cos/sin table by running ggml's own rope over
// (1,0) unit pairs on the CPU backend: out[pos, 2i+0] = cos_i(pos),
// out[pos, 2i+1] = sin_i(pos), with YaRN correction and attn_factor scaling
// exactly as the graph-level ggml_rope_ext computes them.
static bool dsv4_build_rope_table_host(dsv4_model & m, bool comp, std::vector<float> & out)
{
    const dsv4_hparams & hp = m.hp;
    const int64_t n_rot = hp.n_rot;
    const int64_t n_pos = m.n_ctx;

    const float freq_base   = comp ? hp.compress_rope_base : hp.rope_freq_base;
    const float freq_scale  = comp ? hp.yarn_freq_scale : 1.0f;
    const float ext_factor  = comp ? hp.yarn_ext_factor : 0.0f;
    const float attn_factor = dsv4_rope_attn_factor(freq_scale, ext_factor);
    const float beta_fast   = comp ? hp.yarn_beta_fast : 0.0f;
    const float beta_slow   = comp ? hp.yarn_beta_slow : 0.0f;
    const int   n_ctx_orig  = comp ? hp.n_ctx_orig : 0;

    const size_t need = (size_t) (n_rot * n_pos) * sizeof(float) * 2
        + (size_t) n_pos * sizeof(int32_t)
        + 16 * ggml_tensor_overhead() + ggml_graph_overhead() + (1u << 20);
    ggml_init_params ip = { need, nullptr, false };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;

    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_rot, 1, n_pos);
    float * xd = (float *) x->data;
    for (int64_t p = 0; p < n_pos; p++)
        for (int64_t i = 0; i < n_rot; i += 2)
        {
            xd[p * n_rot + i] = 1.0f;
            xd[p * n_rot + i + 1] = 0.0f;
        }

    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    int32_t * pd = (int32_t *) pos->data;
    for (int64_t p = 0; p < n_pos; p++) pd[p] = (int32_t) p;

    ggml_tensor * r = ggml_rope_ext(c, x, pos, nullptr, (int) n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_cgraph * g = ggml_new_graph(c);
    ggml_build_forward_expand(g, r);
    if (tsg::compute_graph(m.backends[m.n_gpu], g) != GGML_STATUS_SUCCESS)
    {
        ggml_free(c);
        return false;
    }

    const float * rd = (const float *) r->data;
    out.assign(rd, rd + (size_t) (n_rot * n_pos));
    ggml_free(c);
    return true;
}

static void dsv4_upload_rope_tables(dsv4_model & m)
{
    for (int k = 0; k < 2; k++)
    {
        if (m.rope_tab_host[k].empty()) continue;
        for (int d = 0; d < m.n_gpu; d++)
            if (m.rope_tab_dev[k][d])
                ggml_backend_tensor_set(m.rope_tab_dev[k][d], m.rope_tab_host[k].data(), 0,
                                        m.rope_tab_host[k].size() * sizeof(float));
    }
}

// Reads the drafter GGUF's metadata and registers its tensors as an extra
// shard, so the normal create+upload path moves them. Returns false only on a
// malformed/mismatched file; a null path just leaves DSpark disabled.
static bool dsv4_scan_dspark(dsv4_model & m, const char * dspark_path,
                             shard_files & shards, std::map<std::string, tensor_source> & sources,
                             size_t * out_bytes)
{
    if (out_bytes) *out_bytes = 0;
    if (!dspark_path || !*dspark_path) return true;

    ggml_context * meta = nullptr;
    gguf_init_params sp = { true, &meta };
    gguf_context * g = gguf_init_from_file(dspark_path, sp);
    if (!g)
    {
        fprintf(stderr, "[dsv4] failed to open DSpark drafter %s\n", dspark_path);
        return false;
    }
    if (!dsv4_check_shard_complete(dspark_path, g, meta))
    {
        gguf_free(g); if (meta) ggml_free(meta);
        return false;
    }

    // Published drafters carry the same weights under three naming schemes
    // (the ds4 builder's mtp.* plus two dspark.* variants that differ in the
    // metadata prefix), so try every spelling for both keys and tensors.
    auto kv_u32 = [&](std::initializer_list<const char *> keys, int32_t * out) -> bool
    {
        for (const char * k : keys)
            if (gguf_get_u32_key(g, k, out)) return true;
        return false;
    };
    auto kv_arr = [&](std::initializer_list<const char *> keys, std::vector<int32_t> & out) -> bool
    {
        for (const char * k : keys)
            if (gguf_get_arr_i32(g, k, out)) return true;
        return false;
    };

    dsv4_dspark & ds = m.ds;
    bool ok = true;
    ok &= kv_u32({ "dspark.block_size", "deepseek4.dspark.block_size" }, &ds.block_size);
    ok &= kv_u32({ "dspark.markov_rank", "deepseek4.dspark.markov_rank" }, &ds.markov_rank);
    ok &= kv_u32({ "dspark.noise_token_id", "deepseek4.dspark.noise_token_id" }, &ds.noise_token);
    ok &= kv_u32({ "dspark.n_layers", "dspark.stage_count", "dspark.layer_count",
                   "deepseek4.dspark.n_layers", "deepseek4.dspark.layer_count" }, &ds.n_stages);
    ok &= kv_arr({ "dspark.target_layer_ids", "dspark.target_layers",
                   "deepseek4.dspark.target_layer_ids", "deepseek4.dspark.target_layers" }, ds.target_layers);
    ds.n_expert = m.hp.n_expert;
    ds.n_expert_used = m.hp.n_expert_used;
    if (m.hp.v41)
    {
        ok &= kv_u32({ "dspark.expert_count" }, &ds.n_expert);
        ok &= kv_u32({ "dspark.expert_used_count" }, &ds.n_expert_used);
    }
    else
    {
        kv_u32({ "dspark.expert_count" }, &ds.n_expert);
        kv_u32({ "dspark.expert_used_count" }, &ds.n_expert_used);
    }
    if (!ok || ds.block_size <= 0 || ds.block_size > 64 || ds.n_stages <= 0 || ds.n_stages > 16 ||
        ds.markov_rank <= 0 || ds.target_layers.empty() || ds.noise_token < 0 || ds.noise_token >= m.hp.n_vocab ||
        ds.n_expert <= 0 || ds.n_expert_used <= 0 || ds.n_expert_used > ds.n_expert ||
        !std::is_sorted(ds.target_layers.begin(), ds.target_layers.end()) ||
        std::adjacent_find(ds.target_layers.begin(), ds.target_layers.end()) != ds.target_layers.end())
    {
        fprintf(stderr, "[dsv4] %s is missing dspark.* metadata\n", dspark_path);
        gguf_free(g); if (meta) ggml_free(meta);
        return false;
    }
    for (int32_t tl : ds.target_layers)
    {
        if (tl < 0 || tl >= m.hp.n_layer)
        {
            fprintf(stderr, "[dsv4] DSpark target layer %d out of range\n", (int) tl);
            gguf_free(g); if (meta) ggml_free(meta);
            return false;
        }
    }

    const char * arch = nullptr;
    {
        const int64_t ai = gguf_find_key(g, "general.architecture");
        if (ai >= 0 && gguf_get_kv_type(g, ai) == GGUF_TYPE_STRING)
            arch = gguf_get_val_str(g, ai);
    }
    const bool correct_arch = m.hp.v41
        ? arch && strcmp(arch, "deepseek41-dspark") == 0
        : !arch || strcmp(arch, "deepseek4-dspark") == 0 || strcmp(arch, "deepseek_v4_flash_dspark_draft") == 0;
    if (!correct_arch)
    {
        fprintf(stderr, "[dsv4] %s has drafter architecture '%s', incompatible with target %s\n",
                dspark_path, arch ? arch : "missing", m.hp.v41 ? "deepseek41" : "deepseek4");
        gguf_free(g); if (meta) ggml_free(meta);
        return false;
    }

    const int si = (int) shards.paths.size();
    shards.paths.push_back(dspark_path);

    const size_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t ti = 0; ti < n_tensors; ti++)
    {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        tensor_source src;
        src.shard = si;
        src.offset = data_off + gguf_get_tensor_offset(g, ti);
        src.size = ggml_nbytes(mt);
        src.type = mt->type;
        for (int d = 0; d < 4; d++) src.ne[d] = mt->ne[d];
        sources[name] = src;
        if (out_bytes) *out_bytes += src.size;
    }

    gguf_free(g);
    if (meta) ggml_free(meta);
    ds.loaded = true;
    return true;
}

// True when `reg` is the backend family the caller asked for. GGML_CUDA_NAME
// is "CUDA" / "ROCm" / "MUSA" depending on how ggml-cuda was built, so the
// managed side's "CUDA" hint has to accept all three.
static bool dsv4_backend_matches(const char * reg_name, const char * want)
{
    if (!reg_name || !want || !*want) return true;
    auto ieq = [](const char * a, const char * b)
    {
#ifdef _WIN32
        return _stricmp(a, b) == 0;
#else
        return strcasecmp(a, b) == 0;
#endif
    };
    if (ieq(reg_name, want)) return true;
    if (ieq(want, "CUDA"))
        return ieq(reg_name, "ROCm") || ieq(reg_name, "MUSA") || ieq(reg_name, "HIP");
    return false;
}

static dsv4_model * dsv4_load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch, int n_threads,
                              const char * dspark_path, int n_cpu_moe_req, const char * backend_name)
{
    auto t_start = std::chrono::steady_clock::now();

    std::unique_ptr<dsv4_model> m(new dsv4_model());
    bool engram_random_advice = false, engram_random_override = false;

    // --- backends ---
    // Only devices belonging to the backend the caller selected. GgmlOps links
    // every backend it was built with, so on a CUDA box `--backend ggml_vulkan`
    // used to silently run on the CUDA devices (they enumerate first) and the
    // Vulkan path was never exercised at all.
    int n_gpu = 0;
    const bool cpu_only = backend_name && (strcmp(backend_name, "CPU") == 0 || strcmp(backend_name, "cpu") == 0);
    if (cpu_only)
    {
        // One logical device reuses the scheduler and placement code while
        // explicitly selected CPU inference stays off accelerators.
        m->backends[0] = ggml_backend_cpu_init();
        if (!m->backends[0]) throw std::runtime_error("Failed to initialize the DeepSeek CPU backend");
        ggml_backend_cpu_set_n_threads(m->backends[0], n_threads > 0 ? n_threads : 16);
        n_gpu = 1;
    }
    for (int pass = 0; !cpu_only && pass < 2 && n_gpu == 0; pass++)
    {
        // pass 0 honors the hint; pass 1 (only reached when nothing matched)
        // takes any GPU rather than refusing to run.
        const char * want = pass == 0 ? backend_name : nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && n_gpu < MAX_GPUS; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (!dsv4_backend_matches(reg ? ggml_backend_reg_name(reg) : nullptr, want)) continue;
            if (n_gpu_req > 0 && n_gpu >= n_gpu_req) break;
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (!be) continue;
            m->backends[n_gpu++] = be;
        }
        if (pass == 0 && n_gpu == 0 && backend_name && *backend_name)
            fprintf(stderr, "[dsv4] no %s GPU devices found; falling back to whatever GPU backend is available\n",
                    backend_name);
    }
    if (n_gpu == 0)
    {
        fprintf(stderr, "[dsv4] no GPU backend available; refusing CPU-only run for a %s\n", "250B model");
        return nullptr;
    }

    // Report the compute devices before metadata scans and weight uploads can
    // take minutes. The CPU scheduler pool below is auxiliary to these devices.
    fprintf(stderr, "[dsv4] compute devices initialized: %d %s\n",
            n_gpu, cpu_only ? "CPU device(s)" : "GPU(s)");
    for (int d = 0; d < n_gpu; ++d)
    {
        ggml_backend_t backend = m->backends[d];
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        fprintf(stderr, "[dsv4]   compute device %d: backend=%s, device=%s (%s)\n",
                d, ggml_backend_name(backend),
                device ? ggml_backend_dev_name(device) : "unknown",
                device ? ggml_backend_dev_description(device) : "unknown");
    }

    m->n_gpu = n_gpu;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    {
        // The CPU backend normally only builds rope tables at load time, so the
        // caller's modest thread count is right. With --n-cpu-moe it also runs
        // the routed-expert matmuls on every token, which is memory-bandwidth
        // bound and wants every core the process may actually use — note
        // available_cpu_parallelism, NOT hardware_concurrency: sizing this pool
        // to 96 hardware threads under a 24-CPU cgroup quota measured 25x
        // slower than sizing it to the quota. The explicit CLI atomic override
        // applies next; inherited native TS_CPU_MOE_THREADS has final priority.
        //
        // Deliberately NOT host_moe_default_thread_count(): that rule caps at
        // 64 because the seam architectures read ~40 MB of expert rows per
        // offloaded layer per token, so the pool runs out of work before it
        // runs out of cores. DSV4 reads ~260 MB (6 of 256 experts, n_ff 2048,
        // n_embd 7168) and keeps scaling past that point.
        int cpu_threads = n_threads > 0 ? n_threads : 16;
        if (n_cpu_moe_req != 0)
            cpu_threads = tsg::available_cpu_parallelism();
        const int explicit_threads = tsg::host_moe_explicit_thread_count();
        if (explicit_threads > 0) cpu_threads = explicit_threads;
        if (const char * e = getenv("TS_CPU_MOE_THREADS")) { int v = atoi(e); if (v > 0) cpu_threads = v; }
        ggml_backend_cpu_set_n_threads(cpu, cpu_threads);

        // A persistent thread pool, because a DSV4 token is not one CPU graph
        // but one per offloaded layer: without it ggml_graph_compute spawns and
        // joins a disposable pool per scheduler split, and at 13 offloaded
        // layers x n_threads that thread churn cost more than the matmuls it
        // was parallelizing (measured 4.3 s/token, ~330 ms of it per layer).
        ggml_threadpool_params tpp = ggml_threadpool_params_default(cpu_threads);
        m->cpu_threadpool = ggml_threadpool_new(&tpp);
        if (m->cpu_threadpool) ggml_backend_cpu_set_threadpool(cpu, m->cpu_threadpool);
        fprintf(stderr, "[dsv4] auxiliary CPU worker pool: threads=%d, persistent=%s\n",
                cpu_threads, m->cpu_threadpool ? "yes" : "no");
#if defined(TSG_GGML_TEST_HOOKS)
        if (m->cpu_threadpool) m->test_cpu_pool_threads = tpp.n_threads;
#endif
    }
    m->backends[n_gpu] = cpu;
    m->n_backends = n_gpu + 1;

    // --- fused-op backends (kernel-count is the decode wall; the fused ops
    // collapse the small-op chains, injected via GGML_OP_CUSTOM nodes) ---
    //
    // Each fused backend WRAPS its GPU's CUDA backend and takes its place in the
    // scheduler, so a device's whole subgraph stays in one split. Registering
    // both would split the graph at every alternation between them: a V4.1 layer
    // alternates ~14 times, which cost 565 splits and 564 blocking host
    // synchronizations per decode token.
    {
        const char * fe = getenv("TS_DSV4_FUSED");
        bool want_fused = !cpu_only && !(fe && atoi(fe) == 0);
#ifdef TSG_GGML_USE_CUDA
        // V4.1 precision is required even when optional elementwise fusion
        // is disabled. Keep the TensorSharp CUDA executor for those matmuls.
        if (want_fused || (!cpu_only && m->hp.v41))
        {
            bool all_ok = true;
            for (int d = 0; d < n_gpu; d++)
            {
                m->ts_backends[d] = tsg_dsv4_fused_backend_init(m->backends[d]);
                if (!m->ts_backends[d]) { all_ok = false; break; }
            }
            m->fused = all_ok && want_fused;
        }
#else
        (void) want_fused;
#endif
        // The scheduler backend for a device: the wrapper when fused ops are
        // available, otherwise the CUDA backend itself. Weights and buffers are
        // unaffected -- both report the same buffer type. Index n_gpu is the CPU
        // backend, which host-resident routed experts are pinned to.
        for (int d = 0; d < n_gpu; d++)
            m->dev_backends[d] = m->ts_backends[d] ? m->ts_backends[d] : m->backends[d];
        m->dev_backends[n_gpu] = cpu;
        int nb = 0;
        for (int d = 0; d < n_gpu; d++) m->sched_backends[nb++] = m->dev_backends[d];
        m->sched_backends[nb++] = cpu;
        for (int i = 0; i < nb; i++)
            m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->sched_backends[i]);
        m->n_sched_backends = nb;

        const char * gc = getenv("TS_DSV4_GRAPH_CACHE");
        if (gc && atoi(gc) > 0) m->graph_cache_cap = atoi(gc);

        const char * ge = getenv("TS_DSV4_GATHER");
        m->gather_env = !(ge && atoi(ge) == 0);
        const char * compact_raw = getenv("TS_DSV41_COMPACT_RAW_GATHER");
        m->compact_raw_gather = compact_raw && strcmp(compact_raw, "1") == 0;
    }

    // --- read gguf metadata (all shards) ---
    ggml_context * meta_ctx0 = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, &meta_ctx0 };
    gguf_context * g0 = gguf_init_from_file(gguf_path, ip);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata_owner(g0, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata_tensors_owner(meta_ctx0, ggml_free);
    if (!g0)
    {
        fprintf(stderr, "[dsv4] failed to open %s\n", gguf_path);
        return nullptr;
    }

    int32_t split_count = 1;
    gguf_get_u32_key(g0, "split.count", &split_count);
    shard_files shards = resolve_shards(gguf_path, split_count);

    dsv4_hparams & hp = m->hp;
    const int64_t arch_id = gguf_find_key(g0, "general.architecture");
    const std::string arch = arch_id >= 0 ? gguf_get_val_str(g0, arch_id) : "";
    if (arch != "deepseek4" && arch != "deepseek41")
        throw std::runtime_error("DeepSeek executor requires deepseek4 or deepseek41 architecture");
    hp.v41 = arch == "deepseek41";
    if (hp.v41 && !cpu_only)
    {
        // V4.1's fused kernels are CUDA-only. A non-CUDA GPU backend can still
        // run the model -- its architecture-specific ops fall to the CPU
        // backend's scalar implementations, which are the same ones the CPU
        // oracle is checked against -- but at a host round trip per occurrence.
        // That is a portability path, not a serving one, so it is opt-in: the
        // failure this rejection originally closed was a SILENT fallback onto
        // whichever GPU enumerated first, not an explicit request.
        const char * allow = getenv("TS_DSV41_ALLOW_NON_CUDA_GPU");
        if (allow && strcmp(allow, "0") != 0 && strcmp(allow, "1") != 0)
            throw std::runtime_error("TS_DSV41_ALLOW_NON_CUDA_GPU must be 0 or 1");
        const bool allowed = allow && strcmp(allow, "1") == 0;
        for (int d = 0; d < n_gpu; ++d)
        {
            ggml_backend_dev_t device = ggml_backend_get_device(m->backends[d]);
            ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
            const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
            if (name && dsv4_backend_matches(name, "CUDA")) continue;
            if (!allowed)
                throw std::runtime_error("DeepSeek V4.1 requires CUDA-family GPU devices; selected device " +
                    std::to_string(d) + " uses " + (name ? name : "an unknown backend") +
                    ". Alternate GPU fallback is not supported for V4.1. Set "
                    "TS_DSV41_ALLOW_NON_CUDA_GPU=1 to run it anyway, with this architecture's ops on the CPU "
                    "backend (correct, but a host round trip per occurrence).");
            fprintf(stderr,
                "[dsv41] device %d uses the %s backend: V4.1's architecture-specific ops have no kernels there "
                "and will run on the CPU backend, one host round trip each. Correctness path, not a serving "
                "path.\n", d, name ? name : "selected");
        }
    }
    if (!hp.v41 && !cpu_only)
    {
        // DeepSeek V4's architecture-specific ops (the 4-stream hyper-connections
        // and the lightning indexer) ship kernels for CPU and CUDA only.
        // Elsewhere ggml_backend_sched routes them to the CPU backend, which is
        // correct but costs a host round trip per layer. Worth saying once, so a
        // slow Vulkan run does not read as a mystery.
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(m->backends[0]));
        const char * rn = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (!dsv4_backend_matches(rn, "CUDA"))
        {
            fprintf(stderr,
                    "[dsv4] note: the %s backend has no kernels for this architecture's hyper-connection and "
                    "lightning-indexer ops; they run on the CPU backend (correct, but a host round trip per "
                    "layer). --backend ggml_cuda / cuda keep everything on the GPU.\n",
                    rn ? rn : "selected");
        }
    }
    int tp_ranks = 0;
    if (const char * value = getenv("TS_DSV41_TP"))
    {
        char * end = nullptr;
        const long ranks = strtol(value, &end, 10);
        if (!*value || *end || ranks < 0 || ranks > MAX_GPUS)
            throw std::runtime_error("TS_DSV41_TP must be 0 or the number of participating GPUs (2..8)");
        tp_ranks = (int) ranks;
        if (tp_ranks && (!hp.v41 || cpu_only || tp_ranks < 2 || tp_ranks != n_gpu))
            throw std::runtime_error("TS_DSV41_TP requires deepseek41 and must equal its selected GPU count (2..8)");
    }
    auto key = [&](const char * suffix) { return arch + "." + suffix; };
    bool ok = true;
    ok &= gguf_get_u32_key(g0, key("block_count").c_str(), &hp.n_layer);
    ok &= gguf_get_u32_key(g0, key("embedding_length").c_str(), &hp.n_embd);
    ok &= gguf_get_u32_key(g0, key("attention.head_count").c_str(), &hp.n_head);
    ok &= gguf_get_u32_key(g0, key("attention.key_length").c_str(), &hp.n_embd_head);
    ok &= gguf_get_u32_key(g0, key("rope.dimension_count").c_str(), &hp.n_rot);
    ok &= gguf_get_u32_key(g0, key("attention.q_lora_rank").c_str(), &hp.q_lora_rank);
    ok &= gguf_get_u32_key(g0, key("attention.output_group_count").c_str(), &hp.o_groups);
    ok &= gguf_get_u32_key(g0, key("attention.output_lora_rank").c_str(), &hp.o_lora_rank);
    ok &= gguf_get_u32_key(g0, key("attention.sliding_window").c_str(), &hp.n_swa);
    ok &= gguf_get_f32_key(g0, key("attention.layer_norm_rms_epsilon").c_str(), &hp.rms_eps);
    ok &= gguf_get_u32_key(g0, key("expert_count").c_str(), &hp.n_expert);
    ok &= gguf_get_u32_key(g0, key("expert_used_count").c_str(), &hp.n_expert_used);
    ok &= gguf_get_u32_key(g0, key("expert_shared_count").c_str(), &hp.n_expert_shared);
    ok &= gguf_get_u32_key(g0, key("expert_feed_forward_length").c_str(), &hp.n_ff_exp);
    ok &= gguf_get_f32_key(g0, key("expert_weights_scale").c_str(), &hp.expert_weights_scale);
    ok &= gguf_get_bool_key(g0, key("expert_weights_norm").c_str(), &hp.expert_weights_norm);
    ok &= gguf_get_u32_key(g0, key("hash_layer_count").c_str(), &hp.hash_layer_count);
    ok &= gguf_get_u32_key(g0, key("attention.indexer.head_count").c_str(), &hp.indexer_n_head);
    ok &= gguf_get_u32_key(g0, key("attention.indexer.key_length").c_str(), &hp.indexer_head_size);
    ok &= gguf_get_u32_key(g0, key("attention.indexer.top_k").c_str(), &hp.indexer_top_k);
    ok &= gguf_get_arr_i32(g0, key("attention.compress_ratios").c_str(), hp.compress_ratios);
    ok &= gguf_get_f32_key(g0, key("attention.compress_rope_freq_base").c_str(), &hp.compress_rope_base);
    ok &= gguf_get_u32_key(g0, key("hyper_connection.count").c_str(), &hp.hc_mult);
    ok &= gguf_get_u32_key(g0, key("hyper_connection.sinkhorn_iterations").c_str(), &hp.hc_sinkhorn_iters);
    ok &= gguf_get_f32_key(g0, key("hyper_connection.epsilon").c_str(), &hp.hc_eps);
    gguf_get_arr_f32(g0, key("swiglu_clamp_exp").c_str(), hp.swiglu_clamp_exp);
    if (!gguf_get_arr_f32(g0, key("swiglu_clamp_shexp").c_str(), hp.swiglu_clamp_shexp))
        hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
    gguf_get_f32_key(g0, key("rope.freq_base").c_str(), &hp.rope_freq_base);

    float yarn_factor = 0.0f;
    if (gguf_get_f32_key(g0, key("rope.scaling.factor").c_str(), &yarn_factor) && yarn_factor > 0.0f)
    {
        hp.yarn_freq_scale = 1.0f / yarn_factor;
        hp.yarn_ext_factor = 1.0f;
    }
    gguf_get_u32_key(g0, key("rope.scaling.original_context_length").c_str(), &hp.n_ctx_orig);
    gguf_get_f32_key(g0, key("rope.scaling.yarn_beta_fast").c_str(), &hp.yarn_beta_fast);
    gguf_get_f32_key(g0, key("rope.scaling.yarn_beta_slow").c_str(), &hp.yarn_beta_slow);

    if (!ok || hp.n_layer <= 0 || (int) hp.compress_ratios.size() < hp.n_layer)
    {
        fprintf(stderr, "[dsv4] missing/invalid deepseek4 metadata\n");
        return nullptr;
    }
    if (hp.v41)
    {
        const int64_t tid = gguf_find_key(g0, "tokenizer.ggml.tokens");
        if (tid < 0) throw std::runtime_error("V4.1 tokenizer metadata is missing");
        const uint32_t vocab = (uint32_t) gguf_get_arr_n(g0, tid);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (uint32_t i = 0; i < vocab; i++)
            hash = tsg_dsv41::engram_data::fingerprint_token(hash, gguf_get_arr_str(g0, tid, i));
        std::string path(gguf_path);
        const auto slash = path.find_last_of("/\\");
        path = (slash == std::string::npos ? "" : path.substr(0, slash + 1)) + "deepseek41.engram.bin";
        m->engram = tsg_dsv41::engram_data::load(path, vocab, hash);
        hp.kv_sources = m->engram.kv_source_layer_ids;
        hp.index_sources = m->engram.index_source_layer_ids;
        hp.candidate_source = m->engram.candidate_source_layer_id;
        hp.candidate_topk = m->engram.candidate_topk_blocks;
        hp.candidate_block = m->engram.candidate_block_size;
        if (hp.kv_sources.back() >= hp.n_layer || hp.index_sources.back() >= hp.n_layer)
            throw std::runtime_error("V4.1 shared-cache source exceeds layer count");
        if (m->engram.layers.back().id >= hp.n_layer)
            throw std::runtime_error("V4.1 Engram layer exceeds layer count");
        if (hp.candidate_source >= 0 &&
            std::find(hp.index_sources.begin(), hp.index_sources.end(), hp.candidate_source) == hp.index_sources.end())
            throw std::runtime_error("V4.1 candidate source must own an indexer");
        if (hp.swiglu_clamp_exp.empty()) hp.swiglu_clamp_exp.assign(hp.n_layer, 10.0f);
        if (hp.swiglu_clamp_shexp.empty()) hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
    }
    metadata_owner.reset();
    metadata_tensors_owner.reset();

    // --- gather tensor sources across shards ---
    std::map<std::string, tensor_source> sources;
    std::vector<size_t> layer_bytes(hp.n_layer, 0);
    // Routed-expert bytes only (the "_exps." tensors), the slice --n-cpu-moe
    // can move off the accelerator. Tracked separately from layer_bytes so the
    // split can price a layer with and without them.
    std::vector<size_t> layer_exps_bytes(hp.n_layer, 0);
    // V4.1 Engram table bytes only. Priced into the split when the tables are
    // placed on their layer's GPU, and excluded when they stay host mappings.
    std::vector<size_t> layer_engram_bytes(hp.n_layer, 0);
    size_t root_bytes = 0;

    for (size_t si = 0; si < shards.paths.size(); si++)
    {
        ggml_context * meta = nullptr;
        gguf_init_params sp = { true, &meta };
        gguf_context * g = gguf_init_from_file(shards.paths[si].c_str(), sp);
        if (!g)
        {
            fprintf(stderr, "[dsv4] failed to open shard %s\n", shards.paths[si].c_str());
            return nullptr;
        }
        if (!dsv4_check_shard_complete(shards.paths[si].c_str(), g, meta))
        {
            gguf_free(g);
            ggml_free(meta);
            return nullptr;
        }
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
                // Engram tables are priced separately: whether they weigh on a
                // device depends on the placement chosen below.
                if (hp.v41 && strstr(name, ".engram_embd.") != nullptr)
                    layer_engram_bytes[bid] += src.size;
                else
                    layer_bytes[bid] += src.size;
                if (strstr(name, "_exps.") != nullptr)
                    layer_exps_bytes[bid] += src.size;
            }
            else
                root_bytes += src.size;
        }
        gguf_free(g);
        ggml_free(meta);
    }

    // --- vocab size from tok_embd ---
    // Its size is also where the split learns how the root bytes really land:
    // the embedding table on the first device, the output head and the whole
    // drafter on the last one (see the W(dev_first/dev_last) placements below).
    size_t embd_bytes = 0;
    {
        auto it = sources.find("token_embd.weight");
        if (it == sources.end()) { fprintf(stderr, "[dsv4] token_embd missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) it->second.ne[1];
        embd_bytes = it->second.size;
    }
    // Draft metadata includes a vocabulary-bounded noise token. Validate it
    // only after the target embedding has established the real vocabulary.
    size_t dspark_bytes = 0;
    if (!dsv4_scan_dspark(*m, dspark_path, shards, sources, &dspark_bytes))
        return nullptr;
    // The drafter lands entirely on the output-head device, so it has to weigh
    // on the split: counted as root bytes it pushes whole layers onto the
    // earlier devices instead of overflowing the last one.
    root_bytes += dspark_bytes;
    const size_t head_bytes = root_bytes - embd_bytes;

    // --- geometry ---
    m->n_ctx = n_ctx > 0 ? n_ctx : 16384;
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    if (m->ds.loaded && m->ds.block_size >= m->n_ubatch)
        throw std::runtime_error("DSpark requires ubatch >= draft block size + 1");
    // The owned CUDA precision paths compute a verify batch exactly like
    // single-token decode only up to TSG_PRECISION_DECODE_COLUMNS rows. A
    // wider drafter still works, but its accepted rows may be committed with
    // last-bit differences that the cache quantization can turn into a
    // full step, so the rewind/greedy parity fixtures would no longer hold.
    if (m->ds.loaded && n_gpu > 0 && (int64_t) m->ds.block_size + 1 > TSG_PRECISION_DECODE_COLUMNS)
        fprintf(stderr, "[dsv4] warning: DSpark verify batches of %d rows exceed the decode-class width %lld; "
                        "speculative verify and single-token decode will not commit bit-identical cache rows on GPU\n",
                m->ds.block_size + 1, (long long) TSG_PRECISION_DECODE_COLUMNS);
    m->ring_raw = pad64(hp.n_swa + m->n_ubatch, 256);
    // +1 so the masked scratch row (last row) used by non-boundary CSA/LID
    // decode steps never collides with a real compressed row.
    m->n_csa_rows = pad64(m->n_ctx / (hp.v41 ? 2 : CSA_RATIO) + 1, 256);
    m->n_hca_rows = pad64(m->n_ctx / (hp.v41 ? 1 : HCA_RATIO) + 1, 256);

    // --- truncation (partial KV reuse) capability -------------------------
    // V4.1 only. V4's compressor OVERLAPS blocks (build_comp_plan's `overlap`
    // is !v41), so a block boundary still reads the PREVIOUS block's rows out
    // of the state ring and aligning the target to the ratio is not enough;
    // that case is left refusing rather than half-proved. V4.1 compresses
    // disjoint blocks, so a target aligned to the widest ratio reads nothing
    // the rewind dropped.
    if (hp.v41)
    {
        int32_t align = 1;
        for (int il = 0; il < hp.n_layer; il++)
        {
            const int32_t r = std::max(1, hp.compress_ratios[il]);
            align = (int32_t) std::lcm((int64_t) align, (int64_t) r);
        }
        m->truncate_align = align;
        // A query at the new head reads the raw window (new_head - n_swa,
        // new_head]; the ring holds ring_raw consecutive positions, so this is
        // how far the head may move back before one of those rows has been
        // overwritten by an abandoned position.
        m->rewind_span = std::max<int64_t>(0, m->ring_raw - hp.n_swa + 1);
        m->rewind_cp = true;
        if (const char * e = getenv("TS_DSV41_REWIND_CHECKPOINT")) m->rewind_cp = atoi(e) != 0;
    }

    // --- what a layer costs its device beyond its weights -----------------
    // The KV caches and compressor state rings are allocated on the layer's
    // own device right after the weights, so the split has to price them:
    // a device packed to the last byte of weights fails at slot allocation
    // instead of at load, which reads as a runtime crash rather than a
    // capacity problem.
    const int64_t state_extra_est = m->ds.loaded ? m->ds.block_size : 0;
    auto layer_cache_bytes = [&](int il) -> size_t
    {
        const int64_t head = hp.n_embd_head;
        const int64_t idx  = hp.indexer_head_size;
        size_t b = (size_t) (head * m->ring_raw * 2);                    // raw_k F16
        const int ratio = hp.compress_ratios[il];
        if (hp.v41)
        {
            if (std::find(hp.kv_sources.begin(), hp.kv_sources.end(), il) != hp.kv_sources.end())
            {
                const int64_t rows = ratio == 2 ? m->n_csa_rows : m->n_hca_rows;
                b += (head + idx) * rows * 2;
                if (ratio > 1) b += 2 * head * (ratio + state_extra_est + 1) * 4;
            }
            if (m->rewind_cp)
            {
                // Rewind-checkpoint shadows: the raw ring, and the compressor
                // state ring where the layer owns one.
                b += (size_t) (head * m->ring_raw * 2);
                if (ratio > 1
                    && std::find(hp.kv_sources.begin(), hp.kv_sources.end(), il) != hp.kv_sources.end())
                {
                    b += (size_t) (2 * head * (ratio + state_extra_est + 1) * 4);
                }
            }
        }
        else if (ratio == CSA_RATIO)
        {
            const int64_t st = 2 * CSA_RATIO + state_extra_est + 1;
            b += (size_t) (head * m->n_csa_rows * 2);                    // csa_k F16
            b += (size_t) (idx * m->n_csa_rows * 2);                     // lid_k F16
            b += (size_t) (2 * (2 * head) * st * 4);                     // comp_state kv+score F32
            b += (size_t) (2 * (2 * idx) * st * 4);                      // lid_state  kv+score F32
        }
        else if (ratio == HCA_RATIO)
        {
            const int64_t st = HCA_RATIO + state_extra_est + 1;
            b += (size_t) (head * m->n_hca_rows * 2);                    // hca_k F16
            b += (size_t) (2 * head * st * 4);                           // comp_state kv+score F32
        }
        return b + 13 * 256;   // ggml buffer alignment padding per tensor
    };

    // --- per-device VRAM budget -------------------------------------------
    // Free VRAM now, minus what the graph itself needs at run time (the
    // scheduler's compute buffers, which are sized from the largest ubatch
    // graph and cannot be known before the model exists). Everything else the
    // split accounts for exactly.
    std::vector<size_t> dev_budget((size_t) n_gpu, 0);
    std::vector<size_t> dev_free((size_t) n_gpu, 0);
    {
        // The scheduler's compute buffers are sized from the largest ubatch graph
        // and cannot be known before the model exists, so the split holds back a
        // reserve. A flat 2 GiB was too small for this architecture once the
        // weights nearly fill the cards: the lightning indexer's top-k runs an
        // argsort over every visible compressed row, and CUB takes its workspace
        // from the CUDA VMM pool at RUN time, not from any layer's budget. A
        // 1024-token ubatch over a 64k context is ~768 MiB for that one transient
        // alone, and a DeepSeek V4.1 Q4_K_M prefill of a 28k-token prompt aborted
        // in argsort_f32_i32_cuda_cub with the flat reserve.
        //
        // So price the transients that scale: the indexer's scores and its sort
        // workspace, and the hidden activations of one ubatch across the streams.
        // TS_DSV4_VRAM_RESERVE_MB still overrides, including downward.
        const int64_t comp_rows = m->n_ctx / (hp.v41 ? 1 : CSA_RATIO) + 1;
        // The factor on the indexer term used to be 4, because the reserve also
        // had to cover a graph cache capped by ENTRY COUNT: twelve prefill
        // graphs of a few hundred MiB each is more than any headroom.
        // dsv4_trim_graph_cache now bounds that cache by bytes, so the reserve
        // covers one graph's compute buffers plus the run-time transients that
        // never pass through a graph buffer (ggml-cuda takes CUB's segmented
        // argsort workspace straight from the VMM pool).
        //
        // Measured on the eight-A40 VM, Q4_K_M, ubatch 1024, 64k context: the
        // largest graph's compute buffers were 1,336 MiB on a device (1.7x the
        // indexer term), and a 57,424-token prefill peaked with 1,522 MiB still
        // free on the tightest device against a 3,072 MiB reserve. 1.25 lands
        // just above that, and holding back more costs routed-expert offload --
        // 5,240 MiB forced three CPU-MoE layers where 3,174 MiB needs one, worth
        // 350 -> 480 prefill tok/s. TS_DSV4_VRAM_RESERVE_MB still overrides,
        // including upward for a rig that wants more margin.
        const double idx_mb = (double) m->n_ubatch * comp_rows * 4.0 * 3.0 / (1024.0 * 1024.0);
        const double act_mb = (double) m->n_ubatch * hp.n_embd * 4.0 * (hp.hc_mult + 2) / (1024.0 * 1024.0);
        size_t reserve_mb = (size_t) std::max(2048.0, 1.25 * idx_mb + act_mb + 2048.0);
        if (const char * e = getenv("TS_DSV4_VRAM_RESERVE_MB")) { long v = atol(e); if (v >= 0) reserve_mb = (size_t) v; }
        fprintf(stderr, "[dsv4] VRAM reserve: %zu MiB per device (indexer %.0f x1.25 + activations %.0f + 2048 headroom)\n",
                reserve_mb, idx_mb, act_mb);
        // Per-device residents the split does not attribute to any layer.
        size_t per_dev_fixed = 0;
        if (m->fused) per_dev_fixed += (size_t) (2 * hp.n_rot * m->n_ctx * 4);   // rope cos/sin tables
        const size_t reserve = reserve_mb * 1024 * 1024 + per_dev_fixed;
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            dev_free[d] = free_b;
            dev_budget[d] = free_b > reserve ? free_b - reserve : 0;
        }
    }

    // --- routed-expert CPU offload + layer -> device split -----------------
    // Within a device the split minimizes the LARGEST *fraction* of budget
    // used, counting each device's fixed residents (embedding table on the
    // first, output head + drafter on the last). Spreading those fixed bytes
    // evenly instead -- what a plain byte-proportional split does -- hands the
    // first device a share of layers it should never have had: with a drafter
    // loaded that was ~1.7 GB of surplus weights on device 0, enough that a
    // long prompt's prefill compute buffer no longer fit and ggml's allocator
    // faulted. Proportional-to-budget (rather than equal bytes) matters as soon
    // as the devices differ: one A6000 with a desktop attached to it has ~3 GiB
    // less to give than its siblings.
    //
    // Across devices: when even a perfect split does not fit, the leading
    // layers' routed experts move to system RAM (--n-cpu-moe). 91% of this
    // checkpoint's bytes are routed experts, so this is the only knob with
    // enough range to matter -- and the fewest possible layers are offloaded,
    // since each one costs a host matmul on every token.
    size_t total_bytes = root_bytes;
    for (auto b : layer_bytes) total_bytes += b;
    m->layers.resize(hp.n_layer);
    int n_cpu_moe = 0;
    {
        // A tensor-parallel rank owns a strip from every resident routed MoE.
        // Price those strips on each rank before placing attention and shared
        // experts. Offloaded layers contribute no accelerator strips.
        std::vector<std::vector<size_t>> tp_bytes(hp.n_layer, std::vector<size_t>(n_gpu));
        if (tp_ranks)
        {
            for (int il = 0; il < hp.n_layer; ++il)
            {
                const std::string prefix = "blk." + std::to_string(il) + ".";
                const auto & gate = sources.at(prefix + "ffn_gate_exps.weight");
                const auto & up = sources.at(prefix + "ffn_up_exps.weight");
                const auto & down = sources.at(prefix + "ffn_down_exps.weight");
                const auto strips = tsg_dsv41_tp::split_weights(down.ne[0], down.type, tp_ranks, il);
                for (int d = 0; d < tp_ranks; ++d)
                    tp_bytes[il][d] = strips[d].count *
                        (ggml_row_size(gate.type, gate.ne[0]) * gate.ne[2] +
                         ggml_row_size(up.type, up.ne[0]) * up.ne[2]) +
                        ggml_row_size(down.type, strips[d].count) * down.ne[1] * down.ne[2];
            }
        }
        std::vector<size_t> fixed_bytes((size_t) n_gpu, 0);
        fixed_bytes[0] += embd_bytes;
        fixed_bytes[(size_t) n_gpu - 1] += head_bytes;
        if (m->ds.loaded)
            fixed_bytes[(size_t) n_gpu - 1] += (size_t) m->ds.n_stages *
                ((size_t) hp.n_embd_head * m->ring_raw * sizeof(ggml_fp16_t) + 256) * (m->rewind_cp ? 2 : 1);

        // V4.1: prefer placing each Engram table on the GPU that owns its
        // layer. The lookup then becomes a device get_rows over the quantized
        // table instead of host page faults plus a CPU dequantize and an
        // upload. A table is tens of GiB, so it is only taken when it costs no
        // routed-expert offload the run was not already going to pay; the
        // packer prices the tables through layer_cost while `engram_device` is
        // set, and the decision is made just after the offload search below.
        bool engram_device = false;
        bool engram_device_forced = false;
        {
            // Validate the option on every path, including CPU-only, so a typo
            // is never silently accepted -- but only V4.1 on an accelerator has
            // tables to place.
            const char * option = getenv("TS_DSV41_ENGRAM_DEVICE");
            bool want = !option || strcmp(option, "1") == 0;
            if (option && strcmp(option, "0") != 0 && strcmp(option, "1") != 0)
                throw std::runtime_error("TS_DSV41_ENGRAM_DEVICE must be 0 or 1 (unset selects automatically)");
            if (hp.v41 && !cpu_only)
            {
                // The gather is a plain ggml get_rows over the quantized table.
                // A backend that has no get_rows kernel for that type would put
                // the node on the CPU and copy tens of GiB per token, so check
                // before choosing the placement rather than discovering it at
                // the first decode.
                bool gatherable = true;
                for (int il = 0; il < hp.n_layer && gatherable; ++il)
                {
                    const auto it = sources.find("blk." + std::to_string(il) + ".engram_embd.weight");
                    if (it == sources.end()) continue;
                    ggml_init_params probe_params = { ggml_tensor_overhead() * 8, nullptr, true };
                    ggml_context * probe = ggml_init(probe_params);
                    if (!probe) { gatherable = false; break; }
                    ggml_tensor * table = ggml_new_tensor_2d(probe, it->second.type, it->second.ne[0], it->second.ne[1]);
                    ggml_tensor * ids = ggml_new_tensor_1d(probe, GGML_TYPE_I32, 24);
                    ggml_tensor * rows = ggml_get_rows(probe, table, ids);
                    for (int d = 0; d < n_gpu && gatherable; ++d)
                        gatherable = ggml_backend_supports_op(m->backends[d], rows);
                    ggml_free(probe);
                }
                if (!gatherable)
                {
                    if (option && strcmp(option, "1") == 0)
                        throw std::runtime_error("TS_DSV41_ENGRAM_DEVICE=1 cannot be honoured: the selected devices "
                            "have no get_rows kernel for this checkpoint's Engram table quantization.");
                    if (want)
                        fprintf(stderr, "[dsv41] Engram tables stay host mappings: the selected devices have no "
                            "get_rows kernel for their quantization\n");
                    want = false;
                }
                engram_device = want;
                engram_device_forced = option && strcmp(option, "1") == 0;
            }
            else if (option && strcmp(option, "1") == 0)
                throw std::runtime_error("TS_DSV41_ENGRAM_DEVICE=1 requires DeepSeek V4.1 on GPU devices; this run "
                    + std::string(hp.v41 ? "selected the CPU device" : "is not V4.1") + ".");
        }
        auto layer_cost = [&](int il, int n_cpu) -> size_t
        {
            size_t w = layer_bytes[il];
            if (il < n_cpu || tp_ranks) w -= layer_exps_bytes[il];
            if (engram_device) w += layer_engram_bytes[il];
            return w + layer_cache_bytes(il);
        };

        // Layers stay in pipeline order, so every device takes one contiguous
        // run: fill each device up to `frac` of its budget, and report whether
        // all of them fit.
        auto pack = [&](double frac, int n_cpu, std::vector<int> * out) -> bool
        {
            auto fixed = fixed_bytes;
            if (tp_ranks)
                for (int il = n_cpu; il < hp.n_layer; ++il)
                    for (int d = 0; d < tp_ranks; ++d) fixed[d] += tp_bytes[il][d];
            if (tp_ranks)
                for (int d = 0; d < n_gpu; ++d)
                    if (fixed[d] > (size_t) (dev_budget[d] * frac)) return false;
            int dev = 0;
            size_t used = fixed[0];
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t cost = layer_cost(il, n_cpu);
                while (used + cost > (size_t) (dev_budget[dev] * frac))
                {
                    if (dev + 1 >= n_gpu) return false;
                    used = fixed[++dev];
                }
                used += cost;
                if (out) (*out)[il] = dev;
            }
            return true;
        };

        // How many leading layers have to give up their experts.
        int need_cpu_moe = 0;
        while (need_cpu_moe <= hp.n_layer && !pack(1.0, need_cpu_moe, nullptr)) need_cpu_moe++;
        if (engram_device)
        {
            // `need_cpu_moe` above was priced WITH the tables on GPUs. Price the
            // same model without them, and keep the tables on GPUs only when
            // two things hold: they cost no routed-expert offload beyond what
            // this run was going to pay anyway, and they still leave a little of
            // every device's budget unspent.
            //
            // The first matters because paying for device tables with host
            // expert matmuls on every token is a bad trade -- though an operator
            // who already asked for --n-cpu-moe should not be refused a
            // placement that fits inside it. The second matters because the
            // packer prices ONE sequence slot's caches, and 60 GiB of tables
            // would otherwise be allowed to consume exactly the headroom the
            // next concurrent sequence needs.
            constexpr double engram_device_margin = 0.95;
            const int with_tables = need_cpu_moe;
            engram_device = false;
            int without_tables = 0;
            while (without_tables <= hp.n_layer && !pack(1.0, without_tables, nullptr)) without_tables++;
            const int already_paying = n_cpu_moe_req >= 0 ? std::max(n_cpu_moe_req, without_tables) : without_tables;

            const char * refused = nullptr;
            if (with_tables > hp.n_layer)
                refused = "they do not fit these devices even with every routed expert on the host";
            else if (with_tables > already_paying)
                refused = "they would cost routed-expert offload this run was not already paying";
            else
            {
                // pack() prices the tables only while engram_device is set, so
                // set it before asking whether the margin holds.
                engram_device = true;
                if (!pack(engram_device_margin, std::max(with_tables, n_cpu_moe_req < 0 ? 0 : n_cpu_moe_req), nullptr))
                {
                    engram_device = false;
                    refused = "they would leave no headroom for a second concurrent sequence";
                }
            }

            if (engram_device)
                need_cpu_moe = with_tables;
            else
            {
                need_cpu_moe = without_tables;
                if (engram_device_forced)
                    throw std::runtime_error(std::string("TS_DSV41_ENGRAM_DEVICE=1 does not fit on these devices: ") +
                        refused + ". With host tables this model needs --n-cpu-moe " +
                        std::to_string(without_tables) + ". Unset the option to choose automatically, or free VRAM.");
                fprintf(stderr, "[dsv41] Engram tables stay host mappings: %s\n", refused);
            }
        }
        m->engram_on_device = engram_device;

        if (n_cpu_moe_req < 0)
        {
            n_cpu_moe = std::min(need_cpu_moe, hp.n_layer);   // opt-in auto
        }
        else
        {
            n_cpu_moe = std::min(n_cpu_moe_req, hp.n_layer);  // operator's choice
            if (n_cpu_moe < need_cpu_moe && need_cpu_moe <= hp.n_layer)
            {
                // Decline instead of loading into a certain out-of-memory
                // abort. Naming WHICH number would work is the whole value
                // here: the operator cannot derive it from the model size,
                // because what has to fit is the weights PLUS this context's
                // KV caches.
                size_t free_total = 0;
                for (int d = 0; d < n_gpu; d++) free_total += dev_free[d];
                size_t would_free = 0;
                for (int il = 0; il < need_cpu_moe && il < hp.n_layer; il++) would_free += layer_exps_bytes[il];
                fprintf(stderr,
                        "[dsv4] not enough VRAM: %.1f GiB of weights plus this context's KV caches against "
                        "%.1f GiB free across %d device(s)%s. Re-run with --n-cpu-moe %d (moves the routed "
                        "experts of the first %d layer(s), %.1f GiB, to system RAM) or --cpu-moe to offload "
                        "every layer.\n",
                        total_bytes / 1073741824.0, free_total / 1073741824.0, n_gpu,
                        n_cpu_moe > 0 ? " at the requested offload" : "",
                        need_cpu_moe, need_cpu_moe, would_free / 1073741824.0);
                return nullptr;
            }
        }

        if (need_cpu_moe > hp.n_layer)
        {
            size_t free_total = 0;
            for (int d = 0; d < n_gpu; d++) free_total += dev_free[d];
            fprintf(stderr, "[dsv4] model does not fit: %.1f GiB of weights (%.1f GiB of them routed experts) "
                    "against %.1f GiB free across %d device(s), even with every expert on the host. "
                    "Free VRAM, add devices, or use a smaller quantization.\n",
                    total_bytes / 1073741824.0,
                    std::accumulate(layer_exps_bytes.begin(), layer_exps_bytes.end(), (size_t) 0) / 1073741824.0,
                    free_total / 1073741824.0, n_gpu);
            return nullptr;
        }

        // Balance: smallest peak budget fraction that still fits.
        double lo = 0.0, hi = 1.0;
        for (int it = 0; it < 40; it++)
        {
            const double mid = 0.5 * (lo + hi);
            if (pack(mid, n_cpu_moe, nullptr)) hi = mid; else lo = mid;
        }

        std::vector<int> devs((size_t) hp.n_layer, 0);
        if (pack(hi, n_cpu_moe, &devs))
        {
            for (int il = 0; il < hp.n_layer; il++)
                m->layers[il].device = devs[il];
        }
        else
        {
            // Fewer layers than devices, or a fixed resident so large that a
            // device cannot take any layer: the balanced split would leave the
            // output head stranded on a device the pipeline never reaches, so
            // fall back to spreading by cumulative bytes.
            size_t acc = 0;
            for (int il = 0; il < hp.n_layer; il++)
            {
                int dev = (int) ((acc * n_gpu) / (total_bytes + 1));
                if (dev >= n_gpu) dev = n_gpu - 1;
                m->layers[il].device = dev;
                acc += layer_bytes[il];
            }
        }

        for (int il = 0; il < n_cpu_moe && il < hp.n_layer; il++)
            m->layers[il].cpu_moe = true;

        if (cpu_only)
            fprintf(stderr, "[dsv4] routed-expert placement: all %d layer(s) on the explicitly selected CPU device\n",
                    hp.n_layer);
        else
            fprintf(stderr, "[dsv4] routed-expert CPU offload: %d of %d layer(s); %d layer(s) on GPUs\n",
                    n_cpu_moe, hp.n_layer, hp.n_layer - n_cpu_moe);
        if (n_cpu_moe > 0)
        {
            size_t host_bytes = 0;
            for (int il = 0; il < n_cpu_moe; il++) host_bytes += layer_exps_bytes[il];
            fprintf(stderr, "[dsv4] MoE CPU offload: routed experts of layers 0..%d (%.1f GiB) stay in system RAM "
                    "and run on the host%s\n", n_cpu_moe - 1, host_bytes / 1073741824.0,
                    n_cpu_moe_req < 0 ? " (auto: the model does not fit the visible VRAM otherwise)" : "");
        }
    }

    // --- create weight contexts ---
    for (int d = 0; d <= n_gpu; d++)
    {
        ggml_init_params wp = { (size_t) 4096 * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(wp);
        ggml_init_params cp = { (size_t) 4096 * ggml_tensor_overhead(), nullptr, true };
        m->c_ctx[d] = ggml_init(cp);
    }

    const int dev_first = m->layers[0].device;
    const int dev_last = m->layers[hp.n_layer - 1].device;

    if (tp_ranks)
    {
        std::vector<ggml_backend_dev_t> devices;
        for (int d = 0; d < tp_ranks; ++d) devices.push_back(ggml_backend_get_device(m->backends[d]));
        m->moe_tp = std::make_unique<tsg_dsv41_tp::executor>(devices, hp.n_expert_used);
        fprintf(stderr, "[dsv41] routed-MoE tensor parallelism: %d ranks, sharded gate/up/down weights; "
                "attention and shared experts use layer placement; host-staged F32 reduction\n", tp_ranks);
    }

    auto tp_source = [&](int il, const char * suffix)
    {
        const auto & src = sources.at("blk." + std::to_string(il) + "." + suffix);
        tsg_dsv41_tp::source result;
        result.path = shards.paths.at(src.shard);
        result.offset = src.offset;
        result.type = src.type;
        std::copy_n(src.ne, 4, result.ne.begin());
        return result;
    };

    auto W = [&](int device, const char * fmt, auto... args) -> ggml_tensor *
    {
        return dsv4_create_weight(*m, sources, device, fmt, args...);
    };

    if (m->ds.loaded)
    {
        // The drafter runs where the output head lives. Target feature rows
        // are reduced on their producing devices and gathered by its graph;
        // their much larger trunk layers need not share this device.
        m->ds.dev = dev_last;
        m->ds.layer_base = hp.n_layer;
        m->layers.resize(hp.n_layer + m->ds.n_stages);
        for (int s = 0; s < m->ds.n_stages; s++)
        {
            m->layers[hp.n_layer + s].device = dev_last;
            hp.compress_ratios.push_back(0);
            if (!hp.swiglu_clamp_exp.empty())
                hp.swiglu_clamp_exp.push_back(hp.swiglu_clamp_exp.back());
            if (!hp.swiglu_clamp_shexp.empty())
                hp.swiglu_clamp_shexp.push_back(hp.swiglu_clamp_shexp.back());
        }
        m->state_extra = m->ds.block_size;
    }

    m->tok_embd = W(dev_first, "token_embd.weight");
    m->output_norm = W(dev_last, "output_norm.weight");
    m->output = W(dev_last, "output.weight");
    if (!hp.v41)
    {
        m->hc_head_fn = W(dev_last, "output_hc_fn.weight");
        m->hc_head_base = W(dev_last, "output_hc_base.weight");
        m->hc_head_scale = W(dev_last, "output_hc_scale.weight");
    }
    if (!m->tok_embd || !m->output_norm || !m->output ||
        (!hp.v41 && (!m->hc_head_fn || !m->hc_head_base || !m->hc_head_scale)))
        return nullptr;

    for (int il = 0; il < hp.n_layer; il++)
    {
        dsv4_layer & L = m->layers[il];
        const int d = L.device;
        const int ratio = hp.compress_ratios[il];

        L.attn_norm = W(d, "blk.%d.attn_norm.weight", il);
        L.attn_sinks = W(d, "blk.%d.attn_sinks.weight", il);
        L.wq_a = W(d, "blk.%d.attn_q_a.weight", il);
        L.attn_q_a_norm = W(d, "blk.%d.attn_q_a_norm.weight", il);
        L.wq_b = W(d, "blk.%d.attn_q_b.weight", il);
        L.wkv = W(d, "blk.%d.attn_kv.weight", il);
        L.attn_kv_norm = W(d, "blk.%d.attn_kv_a_norm.weight", il);
        L.wo_a = W(d, "blk.%d.attn_output_a.weight", il);
        L.wo_b = W(d, "blk.%d.attn_output_b.weight", il);

        L.hc_attn_fn = W(d, "blk.%d.hc_attn_fn.weight", il);
        L.hc_attn_base = W(d, "blk.%d.hc_attn_base.weight", il);
        L.hc_attn_scale = W(d, "blk.%d.hc_attn_scale.weight", il);
        L.hc_ffn_fn = W(d, "blk.%d.hc_ffn_fn.weight", il);
        L.hc_ffn_base = W(d, "blk.%d.hc_ffn_base.weight", il);
        L.hc_ffn_scale = W(d, "blk.%d.hc_ffn_scale.weight", il);

        if (hp.v41)
        {
            auto has = [&](const char * suffix) {
                return sources.count("blk." + std::to_string(il) + "." + suffix) != 0;
            };
            if (has("attn_compressor_kv.weight"))
            {
                L.attn_comp_wkv = W(d, "blk.%d.attn_compressor_kv.weight", il);
                if (ratio > 1) L.attn_comp_wgate = W(d, "blk.%d.attn_compressor_gate.weight", il);
                L.attn_comp_norm = W(d, "blk.%d.attn_compressor_norm.weight", il);
                L.indexer_k = W(d, "blk.%d.indexer.attn_k.weight", il);
                L.indexer_k_norm = W(d, "blk.%d.indexer.k_norm.weight", il);
            }
            if (has("indexer.attn_q_b.weight"))
            {
                L.indexer_proj = W(d, "blk.%d.indexer.proj.weight", il);
                L.indexer_attn_q_b = W(d, "blk.%d.indexer.attn_q_b.weight", il);
            }
            if (has("engram_embd.weight"))
            {
                // On the layer's own device the graph gathers rows with
                // get_rows; otherwise the table stays a host mapping (context
                // n_gpu) that the executor reads and dequantizes per token.
                L.engram_embd = W(m->engram_on_device ? d : n_gpu, "blk.%d.engram_embd.weight", il);
                L.engram_k = W(d, "blk.%d.engram_k.weight", il);
                L.engram_q = W(d, "blk.%d.engram_q.weight", il);
                L.engram_wkv = W(d, "blk.%d.engram_wkv.weight", il);
            }
        }
        else if (ratio != 0)
        {
            L.attn_comp_wkv = W(d, "blk.%d.attn_compressor_kv.weight", il);
            L.attn_comp_wgate = W(d, "blk.%d.attn_compressor_gate.weight", il);
            L.attn_comp_ape = W(d, "blk.%d.attn_compressor_ape.weight", il);
            L.attn_comp_norm = W(d, "blk.%d.attn_compressor_norm.weight", il);
            if (ratio == CSA_RATIO)
            {
                L.indexer_proj = W(d, "blk.%d.indexer.proj.weight", il);
                L.indexer_attn_q_b = W(d, "blk.%d.indexer.attn_q_b.weight", il);
                L.indexer_comp_wkv = W(d, "blk.%d.indexer_compressor_kv.weight", il);
                L.indexer_comp_wgate = W(d, "blk.%d.indexer_compressor_gate.weight", il);
                L.indexer_comp_ape = W(d, "blk.%d.indexer_compressor_ape.weight", il);
                L.indexer_comp_norm = W(d, "blk.%d.indexer_compressor_norm.weight", il);
            }
        }

        L.ffn_gate_inp = W(d, "blk.%d.ffn_gate_inp.weight", il);
        if (il < hp.hash_layer_count)
            L.ffn_gate_tid2eid = W(d, "blk.%d.ffn_gate_tid2eid.weight", il);
        else
            L.ffn_exp_probs_b = W(d, "blk.%d.exp_probs_b.bias", il);
        L.ffn_norm = W(d, "blk.%d.ffn_norm.weight", il);
        // Offloaded layers put ONLY the routed experts on the host context
        // (index n_gpu): the router, the norms and the always-active shared
        // expert are small and every token needs them, so keeping them on the
        // accelerator costs nothing and saves a second host round trip.
        const int de = L.cpu_moe ? n_gpu : d;
        if (m->moe_tp && !L.cpu_moe)
        {
            const float clamp = hp.swiglu_clamp_exp.empty() ? 0.0f : hp.swiglu_clamp_exp[il];
            m->moe_tp->add_layer(il, tp_source(il, "ffn_gate_exps.weight"),
                tp_source(il, "ffn_up_exps.weight"), tp_source(il, "ffn_down_exps.weight"), clamp);
        }
        else
        {
            L.ffn_gate_exps = W(de, "blk.%d.ffn_gate_exps.weight", il);
            L.ffn_down_exps = W(de, "blk.%d.ffn_down_exps.weight", il);
            L.ffn_up_exps = W(de, "blk.%d.ffn_up_exps.weight", il);
        }
        L.ffn_gate_shexp = W(d, "blk.%d.ffn_gate_shexp.weight", il);
        L.ffn_down_shexp = W(d, "blk.%d.ffn_down_shexp.weight", il);
        L.ffn_up_shexp = W(d, "blk.%d.ffn_up_shexp.weight", il);

        if (!L.attn_norm || !L.attn_sinks || !L.wq_a || !L.wq_b || !L.wkv || !L.wo_a || !L.wo_b ||
            !L.ffn_gate_inp || !L.ffn_norm ||
            ((!m->moe_tp || !m->moe_tp->has_layer(il)) &&
             (!L.ffn_gate_exps || !L.ffn_down_exps || !L.ffn_up_exps)))
        {
            fprintf(stderr, "[dsv4] layer %d incomplete\n", il);
            return nullptr;
        }
    }

    if (m->ds.loaded)
    {
        dsv4_dspark & ds = m->ds;
        // First spelling present in the drafter file wins (see dsv4_scan_dspark).
        auto pick = [&](const std::string & a, const std::string & b) -> std::string
        {
            return sources.count(a) ? a : b;
        };
        auto DSW = [&](int d, const std::string & a, const std::string & b) -> ggml_tensor *
        {
            return dsv4_create_weight(*m, sources, d, "%s", pick(a, b).c_str());
        };
        auto SW = [&](int d, int st, const char * suffix) -> ggml_tensor *
        {
            const std::string n = std::to_string(st);
            return DSW(d, "mtp." + n + "." + suffix, "dspark." + n + "." + suffix);
        };
        auto shape = [](ggml_tensor * tensor, std::initializer_list<int64_t> dims)
        {
            if (!tensor) return false;
            int axis = 0;
            for (int64_t n : dims) if (axis >= 4 || tensor->ne[axis++] != n) return false;
            while (axis < 4) if (tensor->ne[axis++] != 1) return false;
            return true;
        };
        for (int st = 0; st < ds.n_stages; st++)
        {
            dsv4_layer & L = m->layers[hp.n_layer + st];
            const int d = ds.dev;
            L.attn_norm      = SW(d, st, "attn_norm.weight");
            L.attn_q_a_norm  = SW(d, st, "attn_q_a_norm.weight");
            L.attn_kv_norm   = SW(d, st, "attn_kv_a_norm.weight");
            L.attn_sinks     = SW(d, st, "attn_sinks.weight");
            L.wq_a           = SW(d, st, "attn_q_a.weight");
            L.wq_b           = SW(d, st, "attn_q_b.weight");
            L.wkv            = SW(d, st, "attn_kv.weight");
            L.wo_a           = SW(d, st, "attn_output_a.weight");
            L.wo_b           = SW(d, st, "attn_output_b.weight");
            L.hc_attn_fn     = SW(d, st, "hc_attn_fn.weight");
            L.hc_attn_scale  = SW(d, st, "hc_attn_scale.weight");
            L.hc_attn_base   = SW(d, st, "hc_attn_base.weight");
            L.hc_ffn_fn      = SW(d, st, "hc_ffn_fn.weight");
            L.hc_ffn_scale   = SW(d, st, "hc_ffn_scale.weight");
            L.hc_ffn_base    = SW(d, st, "hc_ffn_base.weight");
            L.ffn_norm       = SW(d, st, "ffn_norm.weight");
            L.ffn_gate_inp   = SW(d, st, "ffn_gate_inp.weight");
            L.ffn_exp_probs_b = SW(d, st, "exp_probs_b.bias");
            L.ffn_gate_exps  = SW(d, st, "ffn_gate_exps.weight");
            L.ffn_down_exps  = SW(d, st, "ffn_down_exps.weight");
            L.ffn_up_exps    = SW(d, st, "ffn_up_exps.weight");
            L.ffn_gate_shexp = SW(d, st, "ffn_gate_shexp.weight");
            L.ffn_down_shexp = SW(d, st, "ffn_down_shexp.weight");
            L.ffn_up_shexp   = SW(d, st, "ffn_up_shexp.weight");
            const int64_t hc = hp.hc_mult, mix = hc * (hc + 2), dim = hp.n_embd, ff = hp.n_ff_exp;
            if (!shape(L.attn_norm, {dim}) || !shape(L.ffn_norm, {dim}) ||
                !shape(L.attn_q_a_norm, {hp.q_lora_rank}) || !shape(L.attn_kv_norm, {hp.n_embd_head}) ||
                !shape(L.attn_sinks, {hp.n_head}) ||
                !shape(L.wq_a, {dim, hp.q_lora_rank}) || !shape(L.wq_b, {hp.q_lora_rank, (int64_t) hp.n_embd_head * hp.n_head}) ||
                !shape(L.wkv, {dim, hp.n_embd_head}) ||
                (!shape(L.wo_a, {(int64_t) hp.n_embd_head * hp.n_head / hp.o_groups, (int64_t) hp.o_lora_rank * hp.o_groups}) &&
                 !shape(L.wo_a, {(int64_t) hp.n_embd_head * hp.n_head / hp.o_groups, hp.o_lora_rank, hp.o_groups})) ||
                !shape(L.wo_b, {(int64_t) hp.o_lora_rank * hp.o_groups, dim}) ||
                !shape(L.hc_attn_fn, {dim * hc, mix}) || !shape(L.hc_ffn_fn, {dim * hc, mix}) ||
                !shape(L.hc_attn_base, {mix}) || !shape(L.hc_ffn_base, {mix}) ||
                !shape(L.hc_attn_scale, {3}) || !shape(L.hc_ffn_scale, {3}) ||
                !shape(L.ffn_gate_inp, {dim, ds.n_expert}) || !shape(L.ffn_exp_probs_b, {ds.n_expert}) ||
                !shape(L.ffn_gate_exps, {dim, ff, ds.n_expert}) || !shape(L.ffn_up_exps, {dim, ff, ds.n_expert}) ||
                !shape(L.ffn_down_exps, {ff, dim, ds.n_expert}) ||
                !shape(L.ffn_gate_shexp, {dim, ff * hp.n_expert_shared}) ||
                !shape(L.ffn_up_shexp, {dim, ff * hp.n_expert_shared}) ||
                !shape(L.ffn_down_shexp, {ff * hp.n_expert_shared, dim}))
            {
                fprintf(stderr, "[dsv4] DSpark stage %d has missing/incompatible tensor dimensions\n", st);
                return nullptr;
            }
        }
        const std::string last = std::to_string(ds.n_stages - 1);
        ds.main_norm     = DSW(ds.dev, "mtp.0.main_norm.weight", "dspark.main_norm.weight");
        ds.main_proj     = DSW(ds.dev, "mtp.0.main_proj.weight", "dspark.main_proj.weight");
        ds.norm          = DSW(ds.dev, "mtp." + last + ".norm.weight", "dspark.norm.weight");
        if (!hp.v41)
        {
            ds.hc_head_fn    = DSW(ds.dev, "mtp." + last + ".hc_head_fn.weight", "dspark.hc_head_fn.weight");
            ds.hc_head_scale = DSW(ds.dev, "mtp." + last + ".hc_head_scale.weight", "dspark.hc_head_scale.weight");
            ds.hc_head_base  = DSW(ds.dev, "mtp." + last + ".hc_head_base.weight", "dspark.hc_head_base.weight");
        }
        ds.markov_w1     = DSW(ds.dev, "mtp." + last + ".markov_head.markov_w1.weight", "dspark.markov_w1.weight");
        ds.markov_w2     = DSW(ds.dev, "mtp." + last + ".markov_head.markov_w2.weight", "dspark.markov_w2.weight");
        ds.conf_proj     = DSW(ds.dev, "mtp." + last + ".confidence_head.proj.weight",
                               sources.count("dspark.conf_proj.weight") ? "dspark.conf_proj.weight"
                                                                       : "dspark.confidence_head.weight");
        if (!shape(ds.main_norm, {hp.n_embd}) ||
            !shape(ds.main_proj, {(int64_t) hp.n_embd * (int64_t) ds.target_layers.size(), hp.n_embd}) ||
            !shape(ds.norm, {hp.n_embd}) ||
            (!hp.v41 && (!ds.hc_head_fn || !ds.hc_head_scale || !ds.hc_head_base)) || !ds.markov_w1 ||
            !shape(ds.markov_w1, {ds.markov_rank, hp.n_vocab}) ||
            !shape(ds.markov_w2, {ds.markov_rank, hp.n_vocab}) ||
            !shape(ds.conf_proj, {(int64_t) hp.n_embd + ds.markov_rank, 1}))
        {
            fprintf(stderr, "[dsv4] DSpark heads are incomplete\n");
            return nullptr;
        }
        fprintf(stderr, "[dsv4] DSpark drafter on device %d: %d stage(s), block_size=%d, markov_rank=%d, "
                "target_layers=[%d..%d], noise_token=%d\n",
                ds.dev, ds.n_stages, ds.block_size, ds.markov_rank,
                (int) ds.target_layers.front(), (int) ds.target_layers.back(), ds.noise_token);
    }

    if (hp.v41)
    {
        unsigned io_threads = std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
        if (const char * value = getenv("TS_DSV41_ENGRAM_THREADS"))
        {
            char * end = nullptr;
            const long requested = strtol(value, &end, 10);
            if (end == value || *end || requested < 1 || requested > 32)
                throw std::runtime_error("TS_DSV41_ENGRAM_THREADS must be in [1, 32]");
            io_threads = (unsigned) requested;
        }
        const char * random_option = getenv("TS_DSV41_ENGRAM_RANDOM");
        // Resolve first, unconditionally: the parser is what rejects a bad
        // value, and short-circuiting it would silently accept typos on the
        // now-default device path.
        const bool random_wanted = tsg_dsv41::resolve_engram_random(random_option, io_threads);
        engram_random_advice = random_wanted && !m->engram_on_device;
        engram_random_override = random_option != nullptr;
        if (m->engram_on_device)
        {
            // Nothing reads the tables from host memory, so there is no I/O
            // pool, no whole-table warming and no mapping advice to apply.
            fprintf(stderr, "[dsv41] Engram lookup: GPU-resident tables, gathered in-graph (no host reads)\n");
            if (random_option || getenv("TS_DSV41_ENGRAM_THREADS") || getenv("TS_DSV41_ENGRAM_WARM"))
                fprintf(stderr, "[dsv41] note: TS_DSV41_ENGRAM_THREADS/WARM/RANDOM only affect host-mapped "
                    "tables; set TS_DSV41_ENGRAM_DEVICE=0 to use that path\n");
        }
        else
        {
            m->engram_io.reset(new tsg_dsv41::engram_io_pool(io_threads));
            fprintf(stderr, "[dsv41] Engram sparse prefill/decode reads: %u persistent I/O threads\n", io_threads);
        }
        int kv_source = -1, index_source = -1;
        for (int il = 0; il < hp.n_layer; ++il)
        {
            auto & L = m->layers[il];
            const int ratio = hp.compress_ratios[il];
            if (ratio < 0 || ratio > 2) throw std::runtime_error("Invalid V4.1 compression ratio");
            const bool owns_kv = std::find(hp.kv_sources.begin(), hp.kv_sources.end(), il) != hp.kv_sources.end();
            const bool owns_index = std::find(hp.index_sources.begin(), hp.index_sources.end(), il) != hp.index_sources.end();
            if (owns_kv) kv_source = il;
            if (owns_index) index_source = il;
            if (ratio && (kv_source < 0 || index_source < 0 || hp.compress_ratios[kv_source] != ratio || hp.compress_ratios[index_source] != ratio))
                throw std::runtime_error("V4.1 cache-sharing topology does not match compression ratios");
            L.kv_source = ratio ? kv_source : -1;
            L.index_source = ratio ? index_source : -1;
            if (owns_kv != (L.attn_comp_wkv != nullptr) || owns_index != (L.indexer_attn_q_b != nullptr) ||
                (owns_kv && (!L.attn_comp_norm || !L.indexer_k || !L.indexer_k_norm || (ratio > 1 && !L.attn_comp_wgate))))
                throw std::runtime_error("V4.1 cache-sharing metadata does not match checkpoint tensors");
            for (size_t e = 0; e < m->engram.layers.size(); ++e)
                if (m->engram.layers[e].id == il) L.engram_index = (int) e;
            if ((L.engram_index >= 0) != (L.engram_embd != nullptr))
                throw std::runtime_error("V4.1 Engram layer list does not match checkpoint tensors");
            if (L.engram_embd && (L.engram_embd->ne[0] != m->engram.head_dim ||
                (uint64_t) L.engram_embd->ne[1] != m->engram.layers[L.engram_index].rows ||
                !L.engram_k || !L.engram_q || !L.engram_wkv ||
                L.engram_k->ne[0] != hp.n_embd || L.engram_k->ne[1] != hp.hc_mult ||
                L.engram_q->ne[0] != hp.n_embd || L.engram_q->ne[1] != hp.hc_mult ||
                L.engram_wkv->ne[0] != m->engram.hash_columns() * m->engram.head_dim ||
                L.engram_wkv->ne[1] != (hp.hc_mult + 1) * hp.n_embd))
                throw std::runtime_error("V4.1 Engram tensor dimensions do not match prepared metadata");
        }
    }

    // rope cos/sin tables for the fused table-driven kernels (per device)
    if (m->fused)
    {
        for (int d = 0; d < n_gpu; d++)
        {
            for (int k = 0; k < 2; k++)
            {
                m->rope_tab_dev[k][d] = ggml_new_tensor_2d(m->c_ctx[d], GGML_TYPE_F32, hp.n_rot, m->n_ctx);
                ggml_format_name(m->rope_tab_dev[k][d], "rope_tab_%s.%d", k == 0 ? "raw" : "comp", d);
            }
        }
    }

    // --- allocate + upload ---

    // Host-resident weights (the cpu_moe experts) are served straight from the
    // GGUF mmap instead of a private copy. The copy path allocates their full
    // size as ANONYMOUS memory, which the kernel cannot reclaim: a 137 GiB
    // expert set under a container memory quota is not a slow load, it is a
    // silent SIGKILL from the OOM killer with no error output at all. File
    // pages are evictable, so the same model loads everywhere and degrades to
    // page-cache/storage speed only when the host genuinely lacks the memory.
    // TS_DSV4_MOE_MMAP=0 restores the copy path (tensors the mapping cannot
    // serve fall back to it automatically, e.g. on Windows).
    {
        const char * e = getenv("TS_DSV4_MOE_MMAP");
        const bool want_mmap = !(e && atoi(e) == 0);
        if (want_mmap)
        {
            ggml_context * hctx = m->w_ctx[n_gpu];
            for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
            {
                auto it = sources.find(t->name);
                if (it == sources.end()) continue;
                const tensor_source & src = it->second;
                if (ggml_nbytes(t) != src.size) continue; // size mismatch reported by the copy path
                ggml_backend_buffer_t buf = dsv4_mmap_shard(*m, shards, src.shard);
                if (!buf) continue;
                char * base = (char *) m->mmap_addrs[src.shard];
                if (ggml_backend_tensor_alloc(buf, t, base + src.offset) != GGML_STATUS_SUCCESS)
                    continue;
                m->mmap_weight_bytes += ggml_nbytes(t);
            }
            if (m->mmap_weight_bytes > 0)
            {
                const size_t allow = dsv4_host_mem_allowance();
                fprintf(stderr, "[dsv4] host weights are served from the GGUF mapping (%.1f GiB, no private copy)\n",
                        m->mmap_weight_bytes / 1073741824.0);
                if (allow > 0 && m->mmap_weight_bytes + (size_t) 8 * 1024 * 1024 * 1024 > allow)
                {
                    fprintf(stderr,
                            "[dsv4] note: %.1f GiB of host experts against a %.1f GiB process memory allowance "
                            "(cgroup limit; `free` shows the host, not this container). The kernel will evict and "
                            "re-read expert pages from storage on demand, so decode speed is bound by the model "
                            "file's storage, not by compute.\n",
                            m->mmap_weight_bytes / 1073741824.0, allow / 1073741824.0);
                }
            }
        }
    }

    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->w_ctx[d]) != nullptr)
        {
            // Tensors the mmap already placed keep their file backing;
            // ggml_backend_alloc_ctx_tensors only allocates the rest. All
            // mapped = nothing left to allocate = a null buffer that means
            // "empty", not failure.
            bool unallocated = false;
            for (ggml_tensor * t = ggml_get_first_tensor(m->w_ctx[d]); t; t = ggml_get_next_tensor(m->w_ctx[d], t))
                if (t->data == nullptr) { unallocated = true; break; }
            if (unallocated)
            {
                m->w_buf[d] = ggml_backend_alloc_ctx_tensors(m->w_ctx[d], m->backends[d]);
                if (!m->w_buf[d]) { fprintf(stderr, "[dsv4] weight alloc failed on device %d\n", d); return nullptr; }
            }
        }
        if (ggml_get_first_tensor(m->c_ctx[d]) != nullptr)
        {
            m->c_buf[d] = ggml_backend_alloc_ctx_tensors(m->c_ctx[d], m->backends[d]);
            if (!m->c_buf[d]) { fprintf(stderr, "[dsv4] cache alloc failed on device %d\n", d); return nullptr; }
            ggml_backend_buffer_clear(m->c_buf[d], 0);
        }
    }

    {
        // Chunk size balances two pressures: big reads amortize network-FS
        // round trips, small chunks keep all reader threads busy at the tail.
        size_t chunk = (size_t) 64 * 1024 * 1024;
        if (const char * e = getenv("TS_DSV4_LOAD_CHUNK_MB")) { long v = atol(e); if (v > 0) chunk = (size_t) v * 1024 * 1024; }

        int load_threads = dsv4_load_thread_count();

        std::vector<load_job> jobs;
        size_t uploaded = 0;
        if (m->moe_tp)
            for (int d = 0; d < tp_ranks; ++d) uploaded += m->moe_tp->rank_weight_bytes(d);
        bool sizes_ok = true;
        for (int d = 0; d <= n_gpu; d++)
        {
            ggml_context * ctx = m->w_ctx[d];
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
            {
                auto it = sources.find(t->name);
                if (it == sources.end()) continue; // cache tensors etc.
                if (t->buffer != nullptr &&
                    std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), t->buffer) != m->mmap_bufs.end())
                    continue; // served from the GGUF mapping, nothing to copy
                const tensor_source & src = it->second;
                const size_t total = ggml_nbytes(t);
                if (total != src.size)
                {
                    fprintf(stderr, "[dsv4] size mismatch for %s: tensor %zu vs file %zu\n", t->name, total, src.size);
                    sizes_ok = false;
                    continue;
                }
                for (size_t off = 0; off < total; off += chunk)
                    jobs.push_back({ t, src.shard, src.offset + off, off, std::min(chunk, total - off) });
                uploaded += total;
            }
        }
        if (!sizes_ok) return nullptr;
        if (!dsv4_upload_parallel(shards, jobs, load_threads, m->mmap_weight_bytes)) return nullptr;

        if (!dsv4_prefault_host_experts(*m, shards, load_threads)) return nullptr;

        dsv4_pin_host_experts(*m, n_cpu_moe);

        auto t_end = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "[dsv4] loaded %.1f GiB across %d %s in %.1fs (%.2f GiB/s, %d load threads%s; n_ctx=%d, ring=%" PRId64 ", csa_rows=%" PRId64 ", hca_rows=%" PRId64 ")\n",
                uploaded / (1024.0 * 1024.0 * 1024.0), n_gpu, cpu_only ? "CPU device(s)" : "GPU(s)", secs,
                secs > 0 ? uploaded / (1024.0 * 1024.0 * 1024.0) / secs : 0.0,
                load_threads,
                m->mmap_weight_bytes > 0 ? "; host weights mmapped" : "",
                m->n_ctx, m->ring_raw, m->n_csa_rows, m->n_hca_rows);

        // What the split actually did, and what is left for the compute
        // buffers — the number to look at before changing
        // TS_DSV4_VRAM_RESERVE_MB (too little and a long prompt fails to
        // allocate its graph; too much and layers spill to the host for
        // nothing).
        for (int d = 0; d < n_gpu; d++)
        {
            int first = -1, last = -1, count = 0;
            for (int il = 0; il < hp.n_layer; il++)
                if (m->layers[il].device == d) { if (first < 0) first = il; last = il; count++; }
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            if (m->moe_tp)
                fprintf(stderr, "[dsv41]   rank %d: %.3f GiB of routed-expert tensor shards\n",
                        d, m->moe_tp->rank_weight_bytes(d) / 1073741824.0);
            if (count > 0)
                fprintf(stderr, "[dsv4]   device %d: layers %d..%d (%d), %.1f GiB free after load\n",
                        d, first, last, count, free_b / 1073741824.0);
            else
                fprintf(stderr, "[dsv4]   device %d: no layers, %.1f GiB free after load\n",
                        d, free_b / 1073741824.0);
        }
    }

    if (hp.v41 && !dsv4_warm_engram(*m, shards, engram_random_advice, engram_random_override))
        return nullptr;

    // primary sequence slot (slot 0) — the CLI / single-stream cache
    m->active_slot = dsv4_slot_alloc(*m);
    if (!m->active_slot)
    {
        fprintf(stderr, "[dsv4] primary slot allocation failed\n");
        return nullptr;
    }

    if (m->fused)
    {
        if (!dsv4_build_rope_table_host(*m, false, m->rope_tab_host[0]) ||
            !dsv4_build_rope_table_host(*m, true, m->rope_tab_host[1]))
        {
            fprintf(stderr, "[dsv4] rope table build failed; disabling fused ops\n");
            m->fused = false;
        }
        else
        {
            dsv4_upload_rope_tables(*m);
        }
        fprintf(stderr, "[dsv4] fused ops: %s\n", m->fused ? "on" : "off");
    }

    // --- hyper-connection op probe ---
    // ggml ships DSV4_HC_PRE/POST kernels for CPU and CUDA only. Where the
    // accelerator has none, ggml_backend_sched runs them on the CPU backend:
    // correct, but each one is a scheduler split, and the tensors crossing it
    // are the residual stream itself ([n_embd, hc, n_tokens] = 67 MiB at a
    // 1024-token prefill chunk, four times per layer). That was most of the
    // ~2x Vulkan deficit. Both ops are exactly a batched mul_mat over the
    // stream axis, so where the fused op is missing the graph builds the
    // equivalent out of primitives every backend has and the whole layer stays
    // on the accelerator. TS_DSV4_HC_NATIVE=0 forces the decomposition (A/B).
    {
        ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
        ggml_context * pctx = ggml_init(pp);
        const int64_t hc = hp.hc_mult;
        ggml_tensor * x = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hp.n_embd, hc, 1);
        ggml_tensor * w = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hc, 1);
        ggml_tensor * c = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hc, hc, 1);
        ggml_tensor * xf = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hp.n_embd, 1);
        ggml_tensor * pre = ggml_dsv4_hc_pre(pctx, x, w);
        ggml_tensor * post = ggml_dsv4_hc_post(pctx, xf, x, w, c);
        m->hc_native = ggml_backend_supports_op(m->backends[0], pre)
                    && ggml_backend_supports_op(m->backends[0], post);
        ggml_free(pctx);
        // Symmetric override so the two paths can be A/B'd on one backend:
        // 0 forces the decomposition, 1 forces the fused op (which, where the
        // backend has no kernel, means the scheduler's CPU fallback).
        if (const char * e = getenv("TS_DSV4_HC_NATIVE")) m->hc_native = atoi(e) != 0;
        fprintf(stderr, "[dsv4] hyper-connection ops: %s\n",
                m->hc_native ? "native" : "decomposed (backend has no fused kernel)");
    }

    // --- flash attention probe ---
    {
        const char * fa_env = getenv("TS_DSV4_FA");
        const bool want_fa = !(fa_env && atoi(fa_env) == 0);
        if (want_fa)
        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.n_embd_head, 1, hp.n_head, 1);
            ggml_tensor * k = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.n_embd_head, 256, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * fa = ggml_flash_attn_ext(pctx, q, k, k, mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            m->flash_attn = ggml_backend_supports_op(m->backends[0], fa);
            ggml_free(pctx);
        }
        fprintf(stderr, "[dsv4] flash attention: %s\n", m->flash_attn ? "on" : "off");
        if (hp.v41 && m->n_gpu > 0 && m->ts_backends[0])
        {
            const char * sparse = getenv("TS_DSV41_SPARSE_FA");
            const bool sparse_prefill = sparse && atoi(sparse) != 0;
            fprintf(stderr, "[dsv4] V4.1 CUDA precision: TensorSharp F32 matmul; attention=%s\n",
                m->flash_attn ? (sparse_prefill
                    ? "TensorSharp F32 (streaming decode; sparse prefill for queries>4, keys>=8192; tiled otherwise)"
                    : "TensorSharp F32 (streaming decode, tiled prefill)") :
                    "TensorSharp F32 decomposed (FA=0 or unavailable)");
        }
    }

    m->logits.resize(hp.n_vocab);
    return m.release();
}

// Decode-time index-gather sparse attention: past the indexer-skip horizon
// (pos_end > top_k * ratio) the decode token sees n_visible >= top_k
// compressed rows, so every top-k index is a valid row and CSA attention can
// read a compact [raw ring | gathered top-k] K with an all-visible tail mask
// instead of masked-dense over every compressed row. Attention cost then
// stops growing with context; only the indexer scores/top-k stay O(n).
static bool dsv4_use_gather(const dsv4_model & m, int64_t nt, int64_t pos_end)
{
    return m.fused && m.gather_env && nt == 1 &&
           pos_end > (int64_t) m.hp.indexer_top_k * CSA_RATIO;
}

static bool dsv41_use_gather(const dsv4_model & m, int64_t nt, int64_t p0)
{
    // Use the actual query position, never the prefill's future-position hint:
    // Both ratios must have more than top_k visible rows: at equality the
    // ratio-2 source can skip top-k and publish only the complete dense mask.
    // In particular p0=1024 has exactly512 rows for the released checkpoint.
    // Candidate pruning pins the newest (possibly partial) block, so reserve
    // only one visible row from that block when proving sufficient candidates.
    const auto & hp = m.hp;
    const int64_t candidate_min = (int64_t) (hp.candidate_topk - 1) * hp.candidate_block + 1;
    return hp.v41 && m.gather_env && nt == 1 && hp.indexer_top_k > 0 &&
        (p0 + 1) / 2 > (int64_t) hp.indexer_top_k &&
        (hp.candidate_source < 0 || candidate_min >= hp.indexer_top_k);
}

static bool dsv41_use_compact_raw_gather(const dsv4_model & m, int64_t nt, int64_t p0)
{
    // The released checkpoint's sparse-selection threshold is later than its
    // complete raw window. Small numerical fixtures can cross these separately.
    return m.compact_raw_gather && dsv41_use_gather(m, nt, p0) &&
        m.hp.n_swa > 0 && p0 >= m.hp.n_swa - 1;
}

static int64_t dsv41_compact_raw_rows(const dsv4_model & m)
{
    // CUDA head-512 Flash Attention requires its full K width to be a multiple
    // of 256. Pad the raw prefix with masked duplicate IDs to keep both gather
    // stages on the existing kernels. Only n_swa unique rows remain visible.
    return pad64((int64_t) m.hp.n_swa + m.hp.indexer_top_k, 256) - m.hp.indexer_top_k;
}

static comp_plan build_comp_plan(
    int64_t p0, int64_t nt,
    int64_t ratio, bool overlap,
    int64_t state_size, int64_t kv_rows,
    int64_t hint_rows = 0)
{
    comp_plan plan;
    plan.n_visible.resize(nt);

    struct persist_row { int64_t dst; int32_t src; int64_t pos; };
    std::vector<persist_row> persist;

    std::vector<int32_t> prev_reads, cur_reads;

    // Source row space: [0, state_size) persistent ring, state_size = the
    // ring's built-in zero/-inf row, [state_size+1, ...) current-ubatch scratch.
    auto src_idx = [&](int64_t pos) -> int32_t
    {
        if (pos < 0) return (int32_t) state_size;                       // built-in zero/-inf row
        if (pos >= p0) return (int32_t) (state_size + 1 + (pos - p0));  // current-ubatch scratch
        return (int32_t) (pos % state_size);                            // persistent ring
    };

    for (int64_t i = 0; i < nt; i++)
    {
        const int64_t pos = p0 + i;
        plan.state_pos.push_back((int32_t) (pos % ratio));
        const int64_t n_visible = (pos + 1) / ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        const int64_t state_idx = pos % state_size;
        auto it = std::find_if(persist.begin(), persist.end(), [&](const persist_row & r) { return r.dst == state_idx; });
        if (it == persist.end()) persist.push_back({ state_idx, (int32_t) i, pos });
        else if (pos > it->pos) { it->src = (int32_t) i; it->pos = pos; }

        if ((pos + 1) % ratio != 0) continue;

        const int64_t source_start = pos + 1 - ratio;
        plan.state_write_idxs.push_back(pos / ratio);
        plan.state_write_pos.push_back((int32_t) source_start);

        if (overlap)
        {
            const int64_t prev_start = source_start - ratio;
            for (int64_t j = 0; j < ratio; j++) prev_reads.push_back(src_idx(prev_start + j));
            for (int64_t j = 0; j < ratio; j++) cur_reads.push_back(src_idx(source_start + j));
        }
        else
        {
            for (int64_t j = 0; j < ratio; j++) plan.state_read_idxs.push_back(src_idx(source_start + j));
        }
    }

    // CSA-rate plans keep the graph shape stable on non-boundary steps by
    // compressing into a masked scratch row (last cache row).
    if (ratio == CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty())
    {
        const int32_t source_idx = src_idx(p0);
        plan.state_write_idxs.push_back(kv_rows - 1);
        plan.state_write_pos.push_back(0);
        if (overlap)
        {
            for (int64_t j = 0; j < ratio; j++) { prev_reads.push_back(source_idx); cur_reads.push_back(source_idx); }
        }
        else
        {
            for (int64_t j = 0; j < ratio; j++) plan.state_read_idxs.push_back(source_idx);
        }
    }

    if (overlap)
    {
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), prev_reads.begin(), prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), cur_reads.begin(), cur_reads.end());
    }

    // Power-of-two n_kv buckets (min 256): the graph shape then only changes
    // at bucket doublings, so the graph cache and the pipelined prefill reuse
    // entries across long position stretches instead of re-building (and
    // device-synchronizing on fresh arena allocations) every 1024 positions.
    // hint_rows (the whole forward call's final row count, capped by the
    // caller) makes every chunk of one prefill share a single bucket.
    {
        int64_t bucket = 256;
        const int64_t needed = std::max<int64_t>(std::max(plan.n_kv, hint_rows), 1);
        while (bucket < needed) bucket <<= 1;
        plan.n_kv = std::min(bucket, kv_rows);
    }

    std::sort(persist.begin(), persist.end(), [](const persist_row & a, const persist_row & b) { return a.dst < b.dst; });
    for (const auto & r : persist)
    {
        plan.state_persist_src_idxs.push_back(r.src);
        plan.state_persist_dst_idxs.push_back(r.dst);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

struct graph_builder
{
    dsv4_model & m;
    const dsv4_hparams & hp;
    ggml_context * ctx;
    ggml_cgraph * gf;
    graph_build_result & res;
    int64_t nt;
    int64_t p0;

    graph_builder(dsv4_model & model, graph_build_result & r, int64_t nt_, int64_t p0_)
        : m(model), hp(model.hp), ctx(r.ctx), gf(r.gf), res(r), nt(nt_), p0(p0_) {}

    // Memoized [n_embd, hc, n] -> [hc, n_embd, n] transposes for the decomposed
    // hyper-connection path (see hc_streams_first). Graph-scoped, like the
    // builder itself.
    std::map<ggml_tensor *, ggml_tensor *> hc_t_cache;

    void trace_v41(ggml_tensor * t, int il, const char * stage)
    {
        if (!hp.v41 || !getenv("TS_DSV41_TRACE_DIR")) return;
        ggml_format_name(t, "v41.%02d.%s", il, stage);
        ggml_build_forward_expand(gf, t);
    }

    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w)
    {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    // Inputs are pinned to the device that consumes them so the scheduler
    // never inserts per-token synchronized cross-backend input copies.
    //
    // A pin is honored by ggml_backend_sched UNCONDITIONALLY -- pass 1 of
    // ggml_backend_sched_split_graph documents "do not overwrite user
    // assignments" and never consults ggml_backend_supports_op. So a pin is a
    // promise that the target backend can run the node, and breaking it is
    // silent: ggml-vulkan's build_graph returns false for an op it has no
    // kernel for, leaving the destination buffer at whatever it held (zero).
    //
    // DeepSeek V4's own ops (the hyper-connections and the lightning indexer)
    // ship kernels for CPU and CUDA only. Pinning a layer-boundary
    // DSV4_HC_POST to a Vulkan device therefore handed it an op it silently
    // skipped: the residual stream arrived on every device after the first as
    // all zeros, and the model emitted noise. Unpinned, the scheduler routes
    // those ops to the CPU backend that does have the kernel -- which is what
    // it already did for the non-boundary ones.
    //
    // Leaf inputs (op NONE) are plain buffers with nothing to execute, so they
    // are always pinned; supports_op is not meaningful for them.
    void pin(ggml_tensor * t, int dev)
    {
        ggml_backend_t backend = m.dev_backends[dev];
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(backend, t))
            return;
        ggml_backend_sched_set_tensor_backend(res.sched, t, backend);
    }

    void precise_matmul(ggml_tensor * t)
    {
        auto * backend = ggml_backend_sched_get_tensor_backend(res.sched, t);
        if (t->src[0]->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F32 &&
            (m.n_gpu == 0 || (backend && ggml_backend_is_cpu(backend))))
        {
            // Native CPU F32/F32 matmul already preserves both sources. Keep
            // its optimized batching and established reduction order for
            // CPU-only execution and explicitly offloaded expert projections.
            ggml_prec_set_acc(t, GGML_PREC_F32);
            ggml_prec_set_src(t, GGML_PREC_F32, 1);
            return;
        }
        tsg_matmul_require_f32(ctx, t);
        // A projection may already have a layer-placement hint from before
        // the final precision pass. Route unsupported custom nodes to the
        // CPU fallback instead of leaving them pinned to Metal or Vulkan.
        if (backend && !ggml_backend_supports_op(backend, t))
            ggml_backend_sched_set_tensor_backend(res.sched, t, m.dev_backends[m.n_gpu]);
    }

    ggml_tensor * new_input_i32(int64_t n, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * new_input_i64(int64_t n, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    // Exact [n_kv, nt] — ggml_lightning_indexer asserts mask ne1 == nt, and the
    // non-flash soft_max path accepts unpadded masks.
    ggml_tensor * new_input_mask(int64_t n_kv, ggml_type type, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n_kv, nt);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    // Emit a fused GGML_OP_CUSTOM node handled by the TensorSharp fused-op
    // backend of `dev` (CUDA) or by the CPU reference fallback.
    ggml_tensor * fused_node(int kind, ggml_type type,
                             int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                             std::initializer_list<ggml_tensor *> args, int dev,
                             int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0,
                             float f0 = 0.0f, float f1 = 0.0f)
    {
        res.descs.emplace_back();
        tsg_dsv4_fused_desc & d = res.descs.back();
        d.kind = kind;
        d.i0 = i0; d.i1 = i1; d.i2 = i2; d.i3 = i3;
        d.f0 = f0; d.f1 = f1;

        ggml_tensor * a[GGML_MAX_SRC];
        int n = 0;
        for (ggml_tensor * t : args) a[n++] = t;

        ggml_tensor * out = ggml_custom_4d(ctx, type, ne0, ne1, ne2, ne3, a, n, tsg_dsv4_fused_cpu, 1, &d);
        // Keep the node on its layer's device; its backend runs it there.
        if (m.fused) pin(out, dev);
        return out;
    }

    // rope parameters for a layer
    void rope_params(int il, float & freq_base, float & freq_scale, float & ext_factor,
                     float & attn_factor, float & beta_fast, float & beta_slow, int & n_ctx_orig) const
    {
        const bool comp = hp.compress_ratios[il] != 0;
        freq_base = comp ? hp.compress_rope_base : hp.rope_freq_base;
        freq_scale = comp ? hp.yarn_freq_scale : 1.0f;
        ext_factor = comp ? hp.yarn_ext_factor : 0.0f;
        attn_factor = dsv4_rope_attn_factor(freq_scale, ext_factor);
        beta_fast = comp ? hp.yarn_beta_fast : 0.0f;
        beta_slow = comp ? hp.yarn_beta_slow : 0.0f;
        n_ctx_orig = comp ? hp.n_ctx_orig : 0;
    }

    ggml_tensor * rope_compress(ggml_tensor * x, ggml_tensor * pos)
    {
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig,
                             hp.compress_rope_base, hp.yarn_freq_scale, hp.yarn_ext_factor,
                             dsv4_rope_attn_factor(hp.yarn_freq_scale, hp.yarn_ext_factor),
                             hp.yarn_beta_fast, hp.yarn_beta_slow);
    }

    // ---- hyper connections ----
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

    ggml_tensor * build_hc_pre(ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
                               ggml_tensor ** post, ggml_tensor ** comb, int dev,
                               ggml_tensor ** next_pre = nullptr, ggml_tensor * previous_pre = nullptr)
    {
        const int64_t hc = hp.hc_mult;
        const int64_t hc_dim = hc * hp.n_embd;
        const int64_t n = x->ne[2];

        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, n);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, hp.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);

        ggml_tensor * pre;
        if (m.fused)
        {
            ggml_tensor * gates = fused_node(TSG_DSV4_FUSED_HC_GATES, GGML_TYPE_F32, 2 * hc, n, 1, 1,
                    { mixes, hc_scale, hc_base }, dev, 0, 0, 0, 0, hp.hc_eps);
            pre = view_row_2d(gates, hc, n, 0);
            *post = view_row_2d(gates, hc, n, hc);
        }
        else
        {
            ggml_tensor * scale_pre = view_row_1d(hc_scale, 1, 0);
            ggml_tensor * scale_post = view_row_1d(hc_scale, 1, 1);
            ggml_tensor * base_pre = view_row_1d(hc_base, hc, 0);
            ggml_tensor * base_post = view_row_1d(hc_base, hc, hc);

            pre = view_row_2d(mixes, hc, n, 0);
            pre = hc_affine(pre, scale_pre, base_pre);
            pre = ggml_sigmoid(ctx, pre);
            pre = ggml_scale_bias(ctx, pre, 1.0f, hp.hc_eps);

            *post = view_row_2d(mixes, hc, n, hc);
            *post = hc_affine(*post, scale_post, base_post);
            *post = ggml_sigmoid(ctx, *post);
            *post = ggml_scale(ctx, *post, 2.0f);
        }

        *comb = ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.hc_eps, hp.hc_sinkhorn_iters);

        if (next_pre)
        {
            *next_pre = pre;
            return previous_pre ? build_hc_pre_op(x, previous_pre) : hc_mean(x);
        }
        return build_hc_pre_op(x, pre);
    }

    // ---- hyper-connection pre/post ----
    //
    // Both ops reduce over the hyper-connection stream axis, which makes each of
    // them a batched mul_mat once the stream axis is moved into position 0
    // (ggml_mul_mat contracts dim 0 and batches over dims 2/3):
    //
    //   hc_pre : dst[i, t]       = sum_h  x[i, h, t] * w[h, t]
    //          = mul_mat( x^T[h, i, t] , w[h, 1, t] )       -> [n_embd, 1, t]
    //   hc_post: dst[i, d, t]    = x[i, t]*post[d, t]
    //                            + sum_s residual[i, s, t] * comb[d, s, t]
    //          = mul_mat( x[1, i, t], post[1, d, t] )       (rank-1 term)
    //          + mul_mat( residual^T[s, i, t], comb^T[s, d, t] )
    //
    // Used only where the backend has no fused kernel. The transpose of the
    // residual stream is the one real cost, and it is memoized: the tensor
    // hc_post takes as `residual` is the same one the preceding hc_pre took as
    // `x`, so a layer pays for it once rather than twice.
    ggml_tensor * hc_streams_first(ggml_tensor * x)
    {
        auto it = hc_t_cache.find(x);
        if (it != hc_t_cache.end())
            return it->second;
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [hc, n_embd, n]
        hc_t_cache[x] = t;
        return t;
    }

    ggml_tensor * build_hc_pre_op(ggml_tensor * x, ggml_tensor * w)
    {
        if (m.hc_native)
            return ggml_dsv4_hc_pre(ctx, x, w);

        const int64_t hc = x->ne[1];
        const int64_t n = x->ne[2];
        ggml_tensor * xt = hc_streams_first(x);                       // [hc, n_embd, n]
        ggml_tensor * w3 = ggml_reshape_3d(ctx, w, hc, 1, n);         // [hc, 1, n]
        ggml_tensor * out = ggml_mul_mat(ctx, xt, w3);                // [n_embd, 1, n]
        return ggml_reshape_2d(ctx, out, x->ne[0], n);
    }

    ggml_tensor * build_hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb)
    {
        if (m.hc_native)
            return ggml_dsv4_hc_post(ctx, x, residual, post, comb);

        const int64_t n_embd = x->ne[0];
        const int64_t n = x->ne[1];
        const int64_t hc = residual->ne[1];

        // rank-1 term: an outer product per token, expressed as a mul_mat whose
        // contracted dimension is 1 (cheaper than materializing a repeat).
        ggml_tensor * x1 = ggml_reshape_3d(ctx, x, 1, n_embd, n);     // [1, n_embd, n]
        ggml_tensor * p1 = ggml_reshape_3d(ctx, post, 1, hc, n);      // [1, hc, n]
        ggml_tensor * outer = ggml_mul_mat(ctx, x1, p1);              // [n_embd, hc, n]

        // stream-mixing term
        ggml_tensor * rt = hc_streams_first(residual);                // [hc(src), n_embd, n]
        ggml_tensor * ct = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));   // [hc(src), hc(dst), n]
        ggml_tensor * mixed = ggml_mul_mat(ctx, rt, ct);              // [n_embd, hc(dst), n]

        return ggml_add(ctx, outer, mixed);
    }

    ggml_tensor * build_hc_head(ggml_tensor * x)
    {
        return build_hc_head_w(x, m.hc_head_fn, m.hc_head_scale, m.hc_head_base);
    }

    ggml_tensor * build_hc_head_w(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base)
    {
        const int64_t hc = hp.hc_mult;
        const int64_t hc_dim = hc * hp.n_embd;
        const int64_t n = x->ne[2];

        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, n);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, hp.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(ctx, fn, flat_norm);

        ggml_tensor * pre = hc_affine(mixes, scale, base);
        pre = ggml_sigmoid(ctx, pre);
        pre = ggml_scale_bias(ctx, pre, 1.0f, hp.hc_eps);

        return build_hc_pre_op(x, pre);
    }

    // ---- DSpark drafter ----

    // Mean over the hyper-connection streams: [n_embd, hc, n] -> [n_embd, n].
    // This is what the drafter's main_proj consumes from each target layer.
    ggml_tensor * hc_mean(ggml_tensor * x)
    {
        const int64_t hc = hp.hc_mult;
        const int64_t n = x->ne[2];
        ggml_tensor * acc = nullptr;
        for (int64_t c = 0; c < hc; c++)
        {
            ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, x, hp.n_embd, n, x->nb[2], c * x->nb[1]));
            acc = acc ? ggml_add(ctx, acc, v) : v;
        }
        return ggml_scale(ctx, acc, 1.0f / (float) hc);
    }

    // Commit one key row per COMMITTED position into every drafter stage's SWA
    // ring, straight from the trunk's own hidden states. Doing it inside the
    // trunk graph is what keeps prefill free of host round trips: the drafter
    // never needs a catch-up pass of its own.
    void build_dspark_ring_update(const std::vector<ggml_tensor *> & feats,
                                  ggml_tensor * inp_pos, ggml_tensor * raw_idxs)
    {
        const dsv4_dspark & ds = m.ds;
        const int64_t head = hp.n_embd_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;

        ggml_tensor * feat = feats[0];
        for (size_t i = 1; i < feats.size(); i++)
        {
            feat = ggml_concat(ctx, feat, feats[i], 0);          // [n_target*n_embd, nt]
            pin(feat, ds.dev);
        }

        ggml_tensor * mx = ggml_mul_mat(ctx, ds.main_proj, feat); // [n_embd, nt]
        pin(mx, ds.dev);
        mx = rms(mx, ds.main_norm);
        trace_v41(feat, 0, "dspark_features");
        trace_v41(mx, 0, "dspark_main_normalized");

        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int n_ctx_orig;
        rope_params(ds.layer_base, freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow, n_ctx_orig);

        for (int st = 0; st < ds.n_stages; st++)
        {
            const dsv4_layer & L = m.layers[ds.layer_base + st];
            ggml_tensor * kv = ggml_mul_mat(ctx, L.wkv, mx);
            kv = rms(kv, L.attn_kv_norm);
            kv = ggml_reshape_3d(ctx, kv, head, 1, nt);

            ggml_tensor * kv_nope = ggml_view_3d(ctx, kv, n_nope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx, kv, n_rope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head),
                    ggml_row_size(kv->type, n_nope));
            kv_pe = ggml_rope_ext(ctx, kv_pe, inp_pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL,
                                  n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                                  beta_fast, beta_slow);
            kv = ggml_concat(ctx, kv_nope, kv_pe, 0);

            trace_v41(kv, st, "dspark_committed_prequant");
            if (hp.v41) kv = v41_quant(kv, 0, ds.dev);
            trace_v41(kv, st, "dspark_committed_quantized");
            ggml_tensor * kv2d = ggml_reshape_2d(ctx, kv, head, nt);
            ggml_build_forward_expand(gf,
                ggml_set_rows(ctx, m.active_slot->ds_k[st], kv2d, raw_idxs));
        }
    }

    // Block attention for one drafter stage: queries at the block's positions
    // over [committed ring | the block itself], the block part NON-causal (the
    // mask input carries both halves).
    ggml_tensor * build_dspark_attention(int st, ggml_tensor * cur, ggml_tensor * inp_pos)
    {
        const dsv4_layer & L = m.layers[m.ds.layer_base + st];
        const int64_t head = hp.n_embd_head;
        const int64_t n_head = hp.n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_groups = hp.o_groups;
        const int64_t o_lora = hp.o_lora_rank;
        const int64_t o_group_dim = (n_head / n_groups) * head;

        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int n_ctx_orig;
        rope_params(m.ds.layer_base + st, freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow, n_ctx_orig);
        auto rope_l = [&](ggml_tensor * x, ggml_tensor * pos)
        {
            return ggml_rope_ext(ctx, x, pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        };

        ggml_tensor * qr = ggml_mul_mat(ctx, L.wq_a, cur);
        qr = rms(qr, L.attn_q_a_norm);
        ggml_tensor * q = ggml_mul_mat(ctx, L.wq_b, qr);
        q = ggml_reshape_3d(ctx, q, head, n_head, nt);
        if (!hp.v41) q = ggml_rms_norm(ctx, q, hp.rms_eps);
        {
            ggml_tensor * q_nope = ggml_view_3d(ctx, q, n_nope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx, q, n_rope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head,
                    ggml_row_size(q->type, n_nope));
            q_pe = rope_l(q_pe, inp_pos);
            q = ggml_concat(ctx, q_nope, q_pe, 0);
        }

        ggml_tensor * kv = ggml_mul_mat(ctx, L.wkv, cur);
        kv = rms(kv, L.attn_kv_norm);
        kv = ggml_reshape_3d(ctx, kv, head, 1, nt);
        {
            ggml_tensor * kv_nope = ggml_view_3d(ctx, kv, n_nope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx, kv, n_rope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head),
                    ggml_row_size(kv->type, n_nope));
            kv_pe = rope_l(kv_pe, inp_pos);
            kv = ggml_concat(ctx, kv_nope, kv_pe, 0);
        }

        // [ring | block] keys. The block's own keys stay in the graph (they are
        // speculative, not committed), so they are concatenated rather than
        // written to the ring.
        if (hp.v41) kv = v41_quant(kv, 0, m.ds.dev);
        ggml_tensor * ring = ggml_view_2d(ctx, m.active_slot->ds_k[st], head, m.ring_raw,
                                          m.active_slot->ds_k[st]->nb[1], 0);
        ring = ggml_reshape_3d(ctx, ring, head, 1, m.ring_raw);
        ggml_tensor * blk = ggml_cast(ctx, ggml_reshape_2d(ctx, kv, head, nt), GGML_TYPE_F16);
        blk = ggml_reshape_3d(ctx, blk, head, 1, nt);
        ggml_tensor * k_all = ggml_concat(ctx, ring, blk, 2);
        if (hp.v41 && ggml_backend_is_cpu(m.backends[m.ds.dev])) k_all = ggml_cast(ctx, k_all, GGML_TYPE_F32);

        const float kq_scale = 1.0f / sqrtf((float) head);
        ggml_tensor * out = attn_mha(q, k_all, res.inp.ds_mask, L.attn_sinks, kq_scale, m.ds.dev);

        out = ggml_reshape_3d(ctx, out, head, n_head, nt);
        ggml_tensor * out_nope = ggml_view_3d(ctx, out, n_nope, n_head, nt,
                ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head, 0);
        ggml_tensor * out_pe = ggml_view_3d(ctx, out, n_rope, n_head, nt,
                ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head,
                ggml_row_size(out->type, n_nope));
        out_pe = ggml_rope_ext_back(ctx, out_pe, inp_pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL,
                                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                                    beta_fast, beta_slow);
        out = ggml_concat(ctx, out_nope, out_pe, 0);

        out = ggml_reshape_3d(ctx, out, o_group_dim, n_groups, nt);
        ggml_tensor * grouped = ggml_permute(ctx, out, 0, 2, 1, 3);

        ggml_tensor * oa = ggml_mul_mat(ctx,
                ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], o_lora, n_groups), grouped);
        oa = ggml_permute(ctx, oa, 0, 2, 1, 3);
        oa = ggml_cont_2d(ctx, oa, o_lora * n_groups, nt);
        return ggml_mul_mat(ctx, L.wo_b, oa);
    }

    // The whole drafter: [anchor, noise x (block-1)] -> block_size proposals,
    // each conditioned on the one before it through the Markov head, plus the
    // confidence head's per-position acceptance estimate.
    void build_dspark_draft()
    {
        const dsv4_dspark & ds = m.ds;
        const int64_t hc = hp.hc_mult;
        const int64_t rank = ds.markov_rank;
        const int dev = ds.dev;

        res.inp.ds_tokens = new_input_i32(nt, "inp_ds_tokens", dev);
        res.inp.ds_pos = new_input_i32(nt, "inp_ds_pos", dev);
        res.inp.ds_mask = new_input_mask(m.ring_raw + nt, GGML_TYPE_F16, "inp_ds_mask", dev);

        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, res.inp.ds_tokens);
        ggml_tensor * x = ggml_reshape_3d(ctx, emb, hp.n_embd, 1, nt);
        x = ggml_repeat_4d(ctx, x, hp.n_embd, hc, nt, 1);
        pin(x, dev);
        ggml_tensor * delayed_pre = nullptr;

        for (int st = 0; st < ds.n_stages; st++)
        {
            const dsv4_layer & L = m.layers[ds.layer_base + st];
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;

            ggml_tensor * residual = x;
            ggml_tensor * attn_pre = nullptr;
            ggml_tensor * cur = build_hc_pre(x, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &post, &comb, dev,
                hp.v41 ? &attn_pre : nullptr, delayed_pre);
            cur = rms(cur, L.attn_norm);
            cur = build_dspark_attention(st, cur, res.inp.ds_pos);
            x = build_hc_post(cur, residual, post, comb);

            residual = x;
            cur = build_hc_pre(x, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &post, &comb, dev,
                hp.v41 ? &delayed_pre : nullptr, attn_pre);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);
            cur = rms(cur, L.ffn_norm);
            ggml_tensor * shexp = build_shexp(ds.layer_base + st, cur);
            cur = build_moe(ds.layer_base + st, cur, nullptr, shexp);
            x = build_hc_post(cur, residual, post, comb);
        }

        // head: hc_head -> the drafter's norm -> the TRUNK's LM head
        ggml_tensor * h = hp.v41 ? build_hc_pre_op(x, delayed_pre)
            : build_hc_head_w(x, ds.hc_head_fn, ds.hc_head_scale, ds.hc_head_base); // [n_embd, 1, nt]
        h = ggml_reshape_2d(ctx, h, hp.n_embd, nt);
        ggml_tensor * base = ggml_mul_mat(ctx, m.output, rms(h, ds.norm));   // [n_vocab, nt]
        ggml_mul_mat_set_prec(base, GGML_PREC_F32);

        // Markov chain: position i is biased by W2 . W1[prev(i)], and its argmax
        // is prev(i+1). prev(0) is the anchor, i.e. block token 0.
        ggml_tensor * prev = ggml_view_1d(ctx, res.inp.ds_tokens, 1, 0);
        ggml_tensor * toks = nullptr;
        ggml_tensor * conf = nullptr;
        for (int64_t i = 0; i < nt; i++)
        {
            ggml_tensor * w1 = ggml_get_rows(ctx, ds.markov_w1, prev);        // [rank, 1]
            ggml_tensor * bias = ggml_mul_mat(ctx, ds.markov_w2, w1);         // [n_vocab, 1]
            ggml_mul_mat_set_prec(bias, GGML_PREC_F32);
            ggml_tensor * col = ggml_view_2d(ctx, base, hp.n_vocab, 1, base->nb[1], i * base->nb[1]);
            col = ggml_add(ctx, col, bias);
            ggml_tensor * tok = ggml_argmax(ctx, col);                        // I32 [1]
            toks = toks ? ggml_concat(ctx, toks, tok, 0) : tok;

            // conf(i) = sigmoid(proj . [h(i); W1[prev(i)]])
            ggml_tensor * hi = ggml_view_2d(ctx, h, hp.n_embd, 1, h->nb[1], i * h->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx, ggml_cont(ctx, hi), ggml_reshape_2d(ctx, w1, rank, 1), 0);
            ggml_tensor * c = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ds.conf_proj, feat));   // [1, 1]
            conf = conf ? ggml_concat(ctx, conf, c, 0) : c;

            prev = tok;
        }

        ggml_set_output(toks);
        ggml_set_name(toks, "ds_toks");
        ggml_set_output(conf);
        ggml_set_name(conf, "ds_conf");
        res.ds_toks = toks;
        res.ds_conf = conf;
        ggml_build_forward_expand(gf, toks);
        ggml_build_forward_expand(gf, conf);
        if (hp.v41)
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
            {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
                    node->src[0]->type == GGML_TYPE_F32) precise_matmul(node);
            }
    }

    // ---- compression ----

    // Overlap (ratio 4) compression from [persistent ring | scratch | zero] rows.
    ggml_tensor * build_overlap_compressed(
        ggml_tensor * kv_state,      // [2*head, state+nt]
        ggml_tensor * score_state,   // [2*head, state+nt]
        ggml_tensor * read_idxs,     // I32 [2*ratio*n_blocks]
        ggml_tensor * comp_pos,      // I32 [n_blocks]
        ggml_tensor * norm_w,
        int64_t ratio, int64_t head)
    {
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_blocks = comp_pos->ne[0];
        const int64_t n_read = ratio * n_blocks;

        // The zero/-inf "before the first block" source row is baked into the
        // persistent state ring (row state_size), so no per-eval append here.
        ggml_tensor * kv_rows = ggml_get_rows(ctx, kv_state, read_idxs);
        ggml_tensor * score_rows = ggml_get_rows(ctx, score_state, read_idxs);

        ggml_tensor * kv_prev = ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, head, n_read, kv_rows->nb[1], 0));
        kv_prev = ggml_reshape_3d(ctx, kv_prev, head, ratio, n_blocks);
        ggml_tensor * score_prev = ggml_cont(ctx, ggml_view_2d(ctx, score_rows, head, n_read, score_rows->nb[1], 0));
        score_prev = ggml_reshape_3d(ctx, score_prev, head, ratio, n_blocks);

        ggml_tensor * kv_cur = ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, head, n_read, kv_rows->nb[1],
                n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, head)));
        kv_cur = ggml_reshape_3d(ctx, kv_cur, head, ratio, n_blocks);
        ggml_tensor * score_cur = ggml_cont(ctx, ggml_view_2d(ctx, score_rows, head, n_read, score_rows->nb[1],
                n_read * score_rows->nb[1] + ggml_row_size(score_rows->type, head)));
        score_cur = ggml_reshape_3d(ctx, score_cur, head, ratio, n_blocks);

        ggml_tensor * values = ggml_concat(ctx, kv_prev, kv_cur, 1);
        ggml_tensor * scores = ggml_concat(ctx, score_prev, score_cur, 1);

        values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
        scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx, scores);
        ggml_tensor * comp = ggml_mul(ctx, values, weights);
        comp = ggml_sum_rows(ctx, comp);
        comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));

        comp = rms(comp, norm_w);

        ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, n_nope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head), 0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, n_rope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head),
                ggml_row_size(comp->type, n_nope));

        comp_pe = rope_compress(comp_pe, comp_pos);
        return ggml_concat(ctx, comp_nope, comp_pe, 0);
    }

    // Non-overlap (ratio 128) compression.
    ggml_tensor * build_hca_compressed(
        ggml_tensor * kv_state, ggml_tensor * score_state,
        ggml_tensor * read_idxs, ggml_tensor * comp_pos,
        ggml_tensor * norm_w, int64_t head)
    {
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_blocks = comp_pos->ne[0];

        ggml_tensor * kv = ggml_get_rows(ctx, kv_state, read_idxs);
        kv = ggml_reshape_3d(ctx, kv, head, HCA_RATIO, n_blocks);
        ggml_tensor * score = ggml_get_rows(ctx, score_state, read_idxs);
        score = ggml_reshape_3d(ctx, score, head, HCA_RATIO, n_blocks);

        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
        ggml_tensor * scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx, scores);
        ggml_tensor * comp = ggml_mul(ctx, values, weights);
        comp = ggml_sum_rows(ctx, comp);
        comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));

        comp = rms(comp, norm_w);

        ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, n_nope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head), 0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, n_rope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head),
                ggml_row_size(comp->type, n_nope));

        comp_pe = rope_compress(comp_pe, comp_pos);
        return ggml_concat(ctx, comp_nope, comp_pe, 0);
    }

    // Update a compressor state ring + write completed compressed rows into `cache`.
    void run_compressor(
        int il,
        ggml_tensor * cur,               // [n_embd, nt]
        ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm_w,
        ggml_tensor * state_kv, ggml_tensor * state_score,
        ggml_tensor * cache, int64_t head, int64_t ratio, bool overlap,
        const plan_inputs & pi, const comp_plan & plan, int dev)
    {
        ggml_tensor * st_kv = ggml_mul_mat(ctx, wkv, cur);        // [coff*head, nt]
        ggml_tensor * st_score = ggml_mul_mat(ctx, wgate, cur);
        ggml_tensor * ape_rows = ggml_get_rows(ctx, ape, pi.state_pos);
        st_score = ggml_add(ctx, st_score, ape_rows);

        if (m.fused)
        {
            // one fused node: window softmax-compress + RMS + table RoPE + F16
            // cache commit, then state-ring persist (mutates state/cache in
            // place; the marker result orders it before the attention reads)
            const int n_blocks = (int) plan.state_write_idxs.size();
            const int np = (int) plan.state_persist_src_idxs.size();
            ggml_tensor * marker = fused_node(TSG_DSV4_FUSED_COMPRESS, GGML_TYPE_F32, 1, 1, 1, 1,
                    { st_kv, st_score, state_kv, state_score, norm_w, m.rope_tab_dev[1][dev],
                      pi.comp_meta, cache },
                    dev,
                    n_blocks, (int) ratio, overlap ? 2 : 1, hp.n_rot,
                    hp.rms_eps, (float) np);
            ggml_build_forward_expand(gf, marker);
            return;
        }

        ggml_tensor * source_kv = ggml_concat(ctx, state_kv, st_kv, 1);
        ggml_tensor * source_score = ggml_concat(ctx, state_score, st_score, 1);

        ggml_tensor * comp = nullptr;
        if (pi.write_idxs != nullptr)
        {
            if (overlap)
                comp = build_overlap_compressed(source_kv, source_score, pi.read_idxs, pi.write_pos, norm_w, ratio, head);
            else
                comp = build_hca_compressed(source_kv, source_score, pi.read_idxs, pi.write_pos, norm_w, head);

            // write compressed rows into the cache (F16 set_rows casts)
            ggml_tensor * comp2d = ggml_reshape_2d(ctx, comp, head, comp->ne[2]);
            ggml_tensor * wr = ggml_set_rows(ctx, cache, comp2d, pi.write_idxs);
            ggml_build_forward_expand(gf, wr);
        }

        // persist ring update — expanded after the compression nodes, and each
        // device's split executes its nodes strictly in graph order on one
        // stream, so the ring writes land after the compression reads without
        // needing llama.cpp's zero-dependency ordering trick
        GGML_UNUSED(comp);
        ggml_tensor * persist_kv = ggml_get_rows(ctx, st_kv, pi.persist_src);
        ggml_tensor * persist_score = ggml_get_rows(ctx, st_score, pi.persist_src);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state_kv, persist_kv, pi.persist_dst));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state_score, persist_score, pi.persist_dst));
    }

    // ---- attention ----
    ggml_tensor * attn_mha(ggml_tensor * q, ggml_tensor * k, ggml_tensor * kq_mask, ggml_tensor * sinks, float kq_scale,
                           int device = -1)
    {
        // q [head, n_head, nt], k [head, 1, n_kv] contiguous F16. K doubles as V.
        const int64_t head = k->ne[0];
        const int64_t n_kv = k->ne[2];

        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);   // [head, nt, n_head]
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);   // [head, n_kv, 1]

        ggml_tensor * cur;
        if (m.flash_attn)
        {
            const int attention_device = device >= 0 ? device : 0;
            const bool owned_precision = hp.v41 && attention_device < m.n_gpu &&
                m.ts_backends[attention_device] != nullptr;
            if (owned_precision)
            {
                // ggml's CUDA FA narrows Q and softmax weights to F16 even
                // with F32 accumulation. V4.1 quantization/routing can amplify
                // those lost bits, so keep the complete attention path F32.
                const char * sparse = getenv("TS_DSV41_SPARSE_FA");
                const int capacity = sparse && atoi(sparse) != 0 && q->ne[2] > 4 && n_kv >= 8192
                    ? hp.n_swa + hp.indexer_top_k : 0;
                cur = tsg_attention_f32_on_backend(ctx, res.sched, m.dev_backends[attention_device],
                    qp, kp, kp, kq_mask, sinks, kq_scale, capacity);
            }
            else
            {
                cur = ggml_flash_attn_ext(ctx, qp, kp, kp, kq_mask, kq_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_add_sinks(cur, sinks);
                ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                // V4.1 exposes at most the sliding window plus the selected
                // compressed rows per query. CUDA can compact this existing mask
                // and attend to its finite entries without duplicating K/V for
                // every prefill token. Keep an explicit A/B switch until qualified.
                const char * sparse_fa = hp.v41 ? getenv("TS_DSV41_SPARSE_FA") : nullptr;
                // Dense tiles reuse K/V across prefill queries more efficiently
                // at short contexts. Preserve that path below the measured
                // crossover; upstream also checks device/kernel eligibility.
                if (sparse_fa && atoi(sparse_fa) != 0 && (q->ne[2] == 1 || n_kv >= 16384))
                    ggml_flash_attn_ext_set_n_kv_max(cur, hp.n_swa + hp.indexer_top_k);
                if (device >= 0) pin(cur, device);
            }
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        else
        {
            ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);   // [n_kv, nt, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            if (hp.v41) precise_matmul(kq);
            if (device >= 0) pin(kq, device);

            kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);
            ggml_soft_max_add_sinks(kq, sinks);
            if (device >= 0) pin(kq, device);

            // v^T [n_kv, head] from the contiguous pre-permute K
            ggml_tensor * v = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, k, head, n_kv)));
            if (device >= 0) pin(v, device);
            ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);  // [head, nt, n_head]
            if (hp.v41)
            {
                precise_matmul(kqv);
            }
            if (device >= 0) pin(kqv, device);

            ggml_tensor * curp = ggml_permute(ctx, kqv, 0, 2, 1, 3);   // [head, n_head, nt]
            cur = ggml_cont_2d(ctx, curp, curp->ne[0] * curp->ne[1], curp->ne[2]);
        }
        ggml_build_forward_expand(gf, cur);
        return cur;
    }

    ggml_tensor * build_lid_top_k(int il, ggml_tensor * qr, ggml_tensor * cur, ggml_tensor * inp_pos, int dev)
    {
        const dsv4_layer & L = m.layers[il];
        const dsv4_slot_layer & C = m.active_slot->layers[il];
        const int64_t idx_head = hp.indexer_head_size;
        const int64_t n_idx_head = hp.indexer_n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = idx_head - n_rope;

        ggml_tensor * iq = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr);
        iq = ggml_reshape_3d(ctx, iq, idx_head, n_idx_head, nt);

        ggml_tensor * iq_nope = ggml_view_3d(ctx, iq, n_nope, n_idx_head, nt,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head, 0);
        ggml_tensor * iq_pe = ggml_view_3d(ctx, iq, n_rope, n_idx_head, nt,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head,
                ggml_row_size(iq->type, n_nope));
        iq_pe = rope_compress(iq_pe, inp_pos);
        iq = ggml_concat(ctx, iq_nope, iq_pe, 0);   // [idx_head, n_idx_head, nt]

        ggml_tensor * iw = ggml_mul_mat(ctx, L.indexer_proj, cur);   // [n_idx_head, nt]
        iw = ggml_scale(ctx, iw, 1.0f / sqrtf((float) (idx_head * n_idx_head)));

        const int64_t n_lid = res.plan_lid.n_kv;
        ggml_tensor * ik = ggml_view_2d(ctx, C.lid_k, idx_head, n_lid, C.lid_k->nb[1], 0);
        ik = ggml_reshape_3d(ctx, ik, idx_head, 1, n_lid);

        // fused lightning indexer: q [idx_head, n_idx_head, nt], k [idx_head, 1, n_lid],
        // weights [n_idx_head, nt], mask f16 [n_lid, nt] -> [n_lid, nt]
        ggml_tensor * score = ggml_lightning_indexer(ctx, iq, ik, iw, res.inp.lid[dev].kq_mask);

        const int64_t k = std::min<int64_t>(hp.indexer_top_k, score->ne[0]);
        ggml_tensor * top_k = ggml_cont(ctx, ggml_top_k(ctx, score, (int) k));
        return top_k;
    }

    ggml_tensor * build_top_k_mask(ggml_tensor * kq_mask, ggml_tensor * top_k)
    {
        // port of llama.cpp build_top_k_mask, n_stream == 1
        ggml_tensor * mask_all = ggml_fill(ctx, kq_mask, -INFINITY);
        mask_all = ggml_view_4d(ctx, mask_all, 1, mask_all->ne[0], mask_all->ne[1], 1,
                mask_all->nb[0], mask_all->nb[1], mask_all->nb[2], 0);

        ggml_tensor * top_k_3d = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], 1, 1,
                top_k->nb[1], top_k->nb[2], top_k->nb[3], 0);

        ggml_tensor * zeros = ggml_new_tensor_4d(ctx, kq_mask->type, 1, top_k_3d->ne[0], top_k_3d->ne[1], 1);
        zeros = ggml_fill(ctx, zeros, 0.0f);

        ggml_tensor * masked = ggml_set_rows(ctx, mask_all, zeros, top_k_3d);
        masked = ggml_view_4d(ctx, masked, masked->ne[1], masked->ne[2], 1, 1,
                masked->nb[2], masked->nb[3], masked->nb[3], 0);

        return ggml_add(ctx, masked, kq_mask);
    }

    ggml_tensor * build_attention(int il, ggml_tensor * cur, ggml_tensor * inp_pos)
    {
        const dsv4_layer & L = m.layers[il];
        const dsv4_slot_layer & C = m.active_slot->layers[il];
        const int dev = L.device;
        const int64_t head = hp.n_embd_head;
        const int64_t n_head = hp.n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_groups = hp.o_groups;
        const int64_t o_lora = hp.o_lora_rank;
        const int64_t o_group_dim = (n_head / n_groups) * head;
        const int32_t ratio = hp.compress_ratios[il];
        const bool comp_layer = ratio != 0;
        ggml_tensor * rope_tab = m.rope_tab_dev[comp_layer ? 1 : 0][dev];

        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int n_ctx_orig;
        rope_params(il, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_ctx_orig);

        auto rope_l = [&](ggml_tensor * x, ggml_tensor * pos)
        {
            return ggml_rope_ext(ctx, x, pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        };

        ggml_tensor * qr = ggml_mul_mat(ctx, L.wq_a, cur);
        qr = rms(qr, L.attn_q_a_norm);

        ggml_tensor * q = nullptr;
        if (m.fused)
        {
            // Emit the compressor scratch matmuls first so attn_prep and the
            // compress nodes form one contiguous fused-backend split (fewer
            // scheduler splits, better per-segment CUDA-graph reuse).
            auto comp_scratch = [&](ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape,
                                    const plan_inputs & pi, ggml_tensor ** st_kv, ggml_tensor ** st_score)
            {
                *st_kv = ggml_mul_mat(ctx, wkv, cur);        // [coff*head, nt]
                ggml_tensor * sc = ggml_mul_mat(ctx, wgate, cur);
                ggml_tensor * ape_rows = ggml_get_rows(ctx, ape, pi.state_pos);
                *st_score = ggml_add(ctx, sc, ape_rows);
            };
            auto emit_compress = [&](ggml_tensor * st_kv, ggml_tensor * st_score,
                                     ggml_tensor * state_kv, ggml_tensor * state_score,
                                     ggml_tensor * norm_w, ggml_tensor * cache,
                                     int64_t cratio, bool overlap,
                                     const plan_inputs & pi, const comp_plan & plan)
            {
                const int n_blocks = (int) plan.state_write_idxs.size();
                const int np = (int) plan.state_persist_src_idxs.size();
                ggml_tensor * marker = fused_node(TSG_DSV4_FUSED_COMPRESS, GGML_TYPE_F32, 1, 1, 1, 1,
                        { st_kv, st_score, state_kv, state_score, norm_w, m.rope_tab_dev[1][dev],
                          pi.comp_meta, cache },
                        dev,
                        n_blocks, (int) cratio, overlap ? 2 : 1, hp.n_rot,
                        hp.rms_eps, (float) np);
                ggml_build_forward_expand(gf, marker);
            };

            ggml_tensor * q_raw = ggml_mul_mat(ctx, L.wq_b, qr);    // [head*n_head, nt]
            ggml_tensor * kv_raw = ggml_mul_mat(ctx, L.wkv, cur);   // [head, nt]

            ggml_tensor * cs_kv = nullptr, * cs_score = nullptr;
            ggml_tensor * ls_kv = nullptr, * ls_score = nullptr;
            if (ratio == CSA_RATIO)
            {
                comp_scratch(L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, res.inp.csa[dev], &cs_kv, &cs_score);
                comp_scratch(L.indexer_comp_wkv, L.indexer_comp_wgate, L.indexer_comp_ape, res.inp.lid[dev], &ls_kv, &ls_score);
            }
            else if (ratio == HCA_RATIO)
            {
                comp_scratch(L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, res.inp.hca[dev], &cs_kv, &cs_score);
            }

            // per-head q RMS + RoPE, kv RMS(w) + RoPE + F16 SWA-ring commit
            q = fused_node(TSG_DSV4_FUSED_ATTN_PREP, GGML_TYPE_F32, head, n_head, nt, 1,
                    { q_raw, kv_raw, L.attn_kv_norm, rope_tab, inp_pos, C.raw_k, res.inp.raw_idxs[dev] },
                    dev, (int) n_rope, 0, 0, 0, hp.rms_eps);
            ggml_build_forward_expand(gf, q);

            if (ratio == CSA_RATIO)
            {
                emit_compress(cs_kv, cs_score, C.comp_state_kv, C.comp_state_score, L.attn_comp_norm,
                              C.csa_k, CSA_RATIO, true, res.inp.csa[dev], res.plan_csa);
                emit_compress(ls_kv, ls_score, C.lid_state_kv, C.lid_state_score, L.indexer_comp_norm,
                              C.lid_k, CSA_RATIO, true, res.inp.lid[dev], res.plan_lid);
            }
            else if (ratio == HCA_RATIO)
            {
                emit_compress(cs_kv, cs_score, C.comp_state_kv, C.comp_state_score, L.attn_comp_norm,
                              C.hca_k, HCA_RATIO, false, res.inp.hca[dev], res.plan_hca);
            }
        }
        else
        {
            q = ggml_mul_mat(ctx, L.wq_b, qr);
            q = ggml_reshape_3d(ctx, q, head, n_head, nt);
            q = ggml_rms_norm(ctx, q, hp.rms_eps);

            ggml_tensor * q_nope = ggml_view_3d(ctx, q, n_nope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx, q, n_rope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head,
                    ggml_row_size(q->type, n_nope));
            q_pe = rope_l(q_pe, inp_pos);
            q = ggml_concat(ctx, q_nope, q_pe, 0);   // [head, n_head, nt]

            ggml_tensor * kv = ggml_mul_mat(ctx, L.wkv, cur);
            kv = rms(kv, L.attn_kv_norm);
            kv = ggml_reshape_3d(ctx, kv, head, 1, nt);

            ggml_tensor * kv_nope = ggml_view_3d(ctx, kv, n_nope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx, kv, n_rope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head),
                    ggml_row_size(kv->type, n_nope));
            kv_pe = rope_l(kv_pe, inp_pos);
            kv = ggml_concat(ctx, kv_nope, kv_pe, 0);   // [head, 1, nt]

            // write raw K into the SWA ring
            ggml_tensor * kv2d = ggml_reshape_2d(ctx, kv, head, nt);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.raw_k, kv2d, res.inp.raw_idxs[dev]));
            ggml_build_forward_expand(gf, q);

            // compressor state updates + compressed row commits
            if (ratio == CSA_RATIO)
            {
                run_compressor(il, cur, L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, L.attn_comp_norm,
                               C.comp_state_kv, C.comp_state_score, C.csa_k, head, CSA_RATIO, true,
                               res.inp.csa[dev], res.plan_csa, dev);
                run_compressor(il, cur, L.indexer_comp_wkv, L.indexer_comp_wgate, L.indexer_comp_ape, L.indexer_comp_norm,
                               C.lid_state_kv, C.lid_state_score, C.lid_k, hp.indexer_head_size, CSA_RATIO, true,
                               res.inp.lid[dev], res.plan_lid, dev);
            }
            else if (ratio == HCA_RATIO)
            {
                run_compressor(il, cur, L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, L.attn_comp_norm,
                               C.comp_state_kv, C.comp_state_score, C.hca_k, head, HCA_RATIO, false,
                               res.inp.hca[dev], res.plan_hca, dev);
            }
        }

        // attention source: raw ring (+ compressed rows)
        ggml_tensor * raw_k = ggml_view_2d(ctx, C.raw_k, head, m.ring_raw, C.raw_k->nb[1], 0);
        raw_k = ggml_reshape_3d(ctx, raw_k, head, 1, m.ring_raw);

        ggml_tensor * out = nullptr;
        const float kq_scale = 1.0f / sqrtf((float) head);

        if (ratio == CSA_RATIO && res.inp.gather_mask[dev])
        {
            // Decode index-gather: compact [raw ring | top-k rows] K. Every
            // top-k index is a valid compressed row here (see
            // dsv4_use_gather), so the gathered tail needs no masking and the
            // softmax over the gathered subset equals the masked-dense
            // result. The gather runs after this layer's compressor commit
            // (same stream, graph order), so a just-completed block is
            // selectable exactly like in the masked-dense path.
            ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
            ggml_tensor * k_sel = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16,
                    head, 1, m.ring_raw + top_k->ne[0], 1,
                    { C.raw_k, C.csa_k, top_k }, dev, (int) m.ring_raw);
            out = attn_mha(q, k_sel, res.inp.gather_mask[dev], L.attn_sinks, kq_scale);
        }
        else if (ratio == CSA_RATIO)
        {
            ggml_tensor * csa_k = ggml_view_2d(ctx, C.csa_k, head, res.plan_csa.n_kv, C.csa_k->nb[1], 0);
            csa_k = ggml_reshape_3d(ctx, csa_k, head, 1, res.plan_csa.n_kv);

            ggml_tensor * k_all = ggml_concat(ctx, raw_k, csa_k, 2);

            // While every visible compressed row is within the top-k budget
            // (pos_end/ratio <= top_k), the indexer cannot exclude anything —
            // the plain visibility mask is exact and the whole indexer query
            // chain (projections, RoPE, lightning indexer, argsort) is dead
            // weight. The LID cache is still maintained by its compressor.
            // Decided on the whole forward call's final position (must match
            // dsv4_acquire_graph's signature bit).
            const bool skip_topk = m.fused &&
                std::max(m.pos_end_hint, p0 + nt) <= (int64_t) hp.indexer_top_k * CSA_RATIO;

            ggml_tensor * kq_mask;
            if (skip_topk)
            {
                kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.csa[dev].kq_mask, 0);
            }
            else if (m.fused)
            {
                ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
                ggml_tensor * base = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.csa[dev].kq_mask, 0);
                kq_mask = fused_node(TSG_DSV4_FUSED_TOPK_MASK, GGML_TYPE_F16,
                        m.ring_raw + res.plan_csa.n_kv, nt, 1, 1,
                        { base, top_k }, dev, (int) m.ring_raw);
            }
            else
            {
                ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
                ggml_tensor * csa_mask = build_top_k_mask(res.inp.csa[dev].kq_mask, top_k);
                kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], csa_mask, 0);
            }

            out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
        }
        else if (ratio == HCA_RATIO)
        {
            ggml_tensor * hca_k = ggml_view_2d(ctx, C.hca_k, head, res.plan_hca.n_kv, C.hca_k->nb[1], 0);
            hca_k = ggml_reshape_3d(ctx, hca_k, head, 1, res.plan_hca.n_kv);

            ggml_tensor * k_all = ggml_concat(ctx, raw_k, hca_k, 2);
            ggml_tensor * kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.hca[dev].kq_mask, 0);

            out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
        }
        else
        {
            out = attn_mha(q, raw_k, res.inp.raw_mask[dev], L.attn_sinks, kq_scale);
        }

        ggml_tensor * grouped = nullptr;
        if (m.fused)
        {
            // inverse table-RoPE on the attention output + regroup to the
            // grouped-LoRA layout [o_group_dim, nt, n_groups]
            grouped = fused_node(TSG_DSV4_FUSED_ATTN_FINISH, GGML_TYPE_F32, o_group_dim, nt, n_groups, 1,
                    { out, rope_tab, inp_pos }, dev, (int) n_rope, (int) n_groups, (int) head);
        }
        else
        {
            // inverse RoPE on the rope slice of the attention output
            out = ggml_reshape_3d(ctx, out, head, n_head, nt);
            ggml_tensor * out_nope = ggml_view_3d(ctx, out, n_nope, n_head, nt,
                    ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head, 0);
            ggml_tensor * out_pe = ggml_view_3d(ctx, out, n_rope, n_head, nt,
                    ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head,
                    ggml_row_size(out->type, n_nope));
            out_pe = ggml_rope_ext_back(ctx, out_pe, inp_pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            out = ggml_concat(ctx, out_nope, out_pe, 0);

            out = ggml_reshape_3d(ctx, out, o_group_dim, n_groups, nt);
            grouped = ggml_permute(ctx, out, 0, 2, 1, 3);   // [o_group_dim, nt, n_groups]
        }

        // grouped LoRA output projection
        ggml_tensor * oa = ggml_mul_mat(ctx,
                ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], o_lora, n_groups), grouped);   // [o_lora, nt, n_groups]
        oa = ggml_permute(ctx, oa, 0, 2, 1, 3);     // [o_lora, n_groups, nt]
        oa = ggml_cont_2d(ctx, oa, o_lora * n_groups, nt);

        return ggml_mul_mat(ctx, L.wo_b, oa);
    }

    // ---- MoE ----

    ggml_tensor * swiglu_clamped(ggml_tensor * gate, ggml_tensor * up, float clamp_limit, int dev)
    {
        if (m.fused)
        {
            const float limit = clamp_limit > 1e-6f ? clamp_limit : INFINITY;
            return fused_node(TSG_DSV4_FUSED_SWIGLU_CLAMP, GGML_TYPE_F32,
                    gate->ne[0], gate->ne[1], gate->ne[2], gate->ne[3],
                    { gate, up }, dev, 0, 0, 0, 0, limit);
        }
        if (clamp_limit > 1e-6f)
        {
            up = ggml_clamp(ctx, up, -clamp_limit, clamp_limit);
            gate = ggml_clamp(ctx, gate, -INFINITY, clamp_limit);
        }
        return ggml_swiglu_split(ctx, gate, up);
    }

    // Routed experts of a --n-cpu-moe layer: the stacked expert blocks live in
    // a host buffer, so the whole mul_mat_id chain runs on the ggml CPU backend
    // and only [n_embd, nt] activations cross the bus in each direction.
    //
    // Every node is pinned explicitly. Leaving the placement to the scheduler
    // would invite its "offload big batches" rule (op_offload is on) to pull a
    // mul_mat_id over host weights up to the GPU whenever nt >= 32 — copying
    // the 3.2 GiB of expert blocks into the compute buffer per prefill chunk,
    // which is exactly the VRAM this offload exists to free.
    //
    // The chain is the non-fused arithmetic (the fused custom ops are CUDA
    // nodes), so an offloaded layer computes bit-for-bit what TS_DSV4_FUSED=0
    // computes for a resident one.
    ggml_tensor * build_moe_host(int il, ggml_tensor * cur3, ggml_tensor * weights, ggml_tensor * selected,
                                 ggml_tensor * shexp, float clamp_limit, int dev)
    {
        const dsv4_layer & L = m.layers[il];
        const int64_t n_used = hp.n_expert_used;
        const int cpu_dev = m.n_gpu;
        auto on_cpu = [&](ggml_tensor * t) -> ggml_tensor * { pin(t, cpu_dev); return t; };

        ggml_tensor * up = on_cpu(ggml_mul_mat_id(ctx, L.ffn_up_exps, cur3, selected));   // [n_ff, n_used, nt]
        ggml_tensor * gate = on_cpu(ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, selected));
        if (clamp_limit > 1e-6f)
        {
            up = on_cpu(ggml_clamp(ctx, up, -clamp_limit, clamp_limit));
            gate = on_cpu(ggml_clamp(ctx, gate, -INFINITY, clamp_limit));
        }
        ggml_tensor * h = on_cpu(ggml_swiglu_split(ctx, gate, up));

        ggml_tensor * experts = on_cpu(ggml_mul_mat_id(ctx, L.ffn_down_exps, h, selected)); // [n_embd, n_used, nt]
        experts = on_cpu(ggml_mul(ctx, experts, weights));

        ggml_tensor * moe_out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, hp.n_embd, nt, experts->nb[2], e * experts->nb[1]);
            moe_out = moe_out ? on_cpu(ggml_add(ctx, moe_out, v)) : v;
        }
        if (n_used == 1) moe_out = on_cpu(ggml_cont(ctx, moe_out));

        // The shared expert stayed on the accelerator, so the sum returns there
        // and the rest of the layer never notices the detour.
        ggml_tensor * out = ggml_add(ctx, moe_out, shexp);
        pin(out, dev);
        return out;
    }

    // shexp: the shared expert's FFN output (added to the routed experts)
    ggml_tensor * build_moe(int il, ggml_tensor * cur, ggml_tensor * inp_tokens, ggml_tensor * shexp)
    {
        const dsv4_layer & L = m.layers[il];
        const int dev = L.device;
        const bool is_draft = m.ds.loaded && il >= m.ds.layer_base;
        const int64_t n_expert = is_draft ? m.ds.n_expert : hp.n_expert;
        const int64_t n_used = is_draft ? m.ds.n_expert_used : hp.n_expert_used;
        const float clamp_limit = hp.swiglu_clamp_exp.empty() ? 0.0f : hp.swiglu_clamp_exp[il];

        ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);   // [n_expert, nt]
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);

        ggml_tensor * selected = nullptr;
        ggml_tensor * weights = nullptr;

        if (m.fused && (!res.image_tokens || is_draft))
        {
            if (il < hp.hash_layer_count)
                selected = ggml_get_rows(ctx, L.ffn_gate_tid2eid, inp_tokens);   // I32 [n_used, nt]
            else
                selected = fused_node(TSG_DSV4_FUSED_MOE_TOPK, GGML_TYPE_I32, n_used, nt, 1, 1,
                        { logits, L.ffn_exp_probs_b }, dev);

            const float scale = (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
                ? hp.expert_weights_scale : 1.0f;
            weights = fused_node(TSG_DSV4_FUSED_MOE_WEIGHTS, GGML_TYPE_F32, 1, n_used, nt, 1,
                    { logits, selected }, dev, hp.expert_weights_norm && (!is_draft || n_used > 1) ? 1 : 0, 0, 0, 0, scale);
        }
        else
        {
            ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));

            if (il < hp.hash_layer_count)
            {
                selected = ggml_get_rows(ctx, L.ffn_gate_tid2eid, inp_tokens);   // I32 [n_used, nt]
            }
            else
            {
                // The trained visual bias changes top-k selection only. The
                // selected experts' normalized weights still use unbiased scores.
                ggml_tensor * bias = res.image_tokens && !is_draft
                    ? ggml_get_rows(ctx, m.vision->router_bias[il], res.inp.image_types[dev])
                    : L.ffn_exp_probs_b;
                ggml_tensor * selection = ggml_add(ctx, probs, bias);
                selected = ggml_argsort_top_k(ctx, selection, (int) n_used);
            }

            ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, nt);
            weights = ggml_get_rows(ctx, probs3, selected);       // [1, n_used, nt]

            if (hp.expert_weights_norm && (!is_draft || n_used > 1))
            {
                weights = ggml_reshape_2d(ctx, weights, n_used, nt);
                ggml_tensor * sum = ggml_sum_rows(ctx, weights);
                sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
                weights = ggml_div(ctx, weights, sum);
                weights = ggml_reshape_3d(ctx, weights, 1, n_used, nt);
            }
            if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
                weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
        }

        ggml_build_forward_expand(gf, weights);

        ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, nt);

        if (L.cpu_moe)
            return build_moe_host(il, cur3, weights, selected, shexp, clamp_limit, dev);

        if (m.moe_tp && m.moe_tp->has_layer(il))
        {
            ggml_tensor * routed = m.moe_tp->build(ctx, il, cur, weights, selected);
            pin(routed, m.n_gpu);
            ggml_tensor * out = ggml_add(ctx, routed, shexp);
            pin(out, dev);
            return out;
        }

        ggml_tensor * up = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur3, selected);     // [n_ff, n_used, nt]
        ggml_tensor * gate = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, selected);

        ggml_tensor * h = swiglu_clamped(gate, up, clamp_limit, dev);

        ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, h, selected); // [n_embd, n_used, nt]

        if (m.fused)
        {
            // weighted expert sum + shared-expert add in one launch
            return fused_node(TSG_DSV4_FUSED_EXPERT_REDUCE, GGML_TYPE_F32, hp.n_embd, nt, 1, 1,
                    { experts, weights, shexp }, dev);
        }

        experts = ggml_mul(ctx, experts, weights);

        // sum over the used experts
        ggml_tensor * moe_out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, hp.n_embd, nt, experts->nb[2], e * experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        return ggml_add(ctx, moe_out, shexp);
    }

    ggml_tensor * build_shexp(int il, ggml_tensor * cur)
    {
        const dsv4_layer & L = m.layers[il];
        const float clamp_limit = hp.swiglu_clamp_shexp.empty() ? 0.0f : hp.swiglu_clamp_shexp[il];

        ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
        ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
        bool pin_shared = hp.v41;
#if defined(TSG_GGML_TEST_HOOKS)
        // Same-binary regression control for the former scheduler placement.
        const char * legacy = getenv("TS_DSV41_TEST_LEGACY_SHARED_PLACEMENT");
        if (legacy && strcmp(legacy, "1") == 0) pin_shared = false;
#endif
        // A preceding CPU-routed branch and the CUSTOM-only fused SwiGLU
        // backend can otherwise expand CPU placement into these unassigned
        // projections. Shared experts belong to the layer device even when
        // routed experts use CPU offload or a separate TP executor.
        if (pin_shared) { pin(up, L.device); pin(gate, L.device); }
        ggml_tensor * h = swiglu_clamped(gate, up, clamp_limit, L.device);
        ggml_tensor * down = ggml_mul_mat(ctx, L.ffn_down_shexp, h);
        if (pin_shared) pin(down, L.device);
        return down;
    }

#include "ggml_ops_deepseek41.inc"

    // ---- batched multi-slot decode ----
    //
    // One decode token for each of N sequence slots in a single graph: the
    // weight-heavy dense parts (hyper-connections, q/kv/compressor
    // projections, MoE + shared expert, attention epilogue, lm head) run
    // batched over the N tokens so every weight matrix is read once per
    // step, while the cache-touching parts (ring commit, compressors,
    // indexer, top-k gather, attention) fork out per slot on column views —
    // each token reads/writes only its own slot's caches at its own
    // position.

    // [ne0, 1] view of column i of a contiguous 2-D activation
    ggml_tensor * col_view(ggml_tensor * t, int64_t i)
    {
        return ggml_view_2d(ctx, t, t->ne[0], 1, t->nb[1], i * t->nb[1]);
    }

    ggml_tensor * new_input_mask1(int64_t n_kv, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    // per-slot lightning-indexer top-k (single token at pos_i against
    // lid_cache; the parametrized twin of build_lid_top_k)
    ggml_tensor * bd_lid_top_k(int il, ggml_tensor * lid_cache, ggml_tensor * qr_i,
                               ggml_tensor * cur_i, ggml_tensor * pos_i,
                               ggml_tensor * lid_mask, int64_t n_lid)
    {
        const dsv4_layer & L = m.layers[il];
        const int64_t idx_head = hp.indexer_head_size;
        const int64_t n_idx_head = hp.indexer_n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = idx_head - n_rope;

        ggml_tensor * iq = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr_i);
        iq = ggml_reshape_3d(ctx, iq, idx_head, n_idx_head, 1);

        ggml_tensor * iq_nope = ggml_view_3d(ctx, iq, n_nope, n_idx_head, 1,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head, 0);
        ggml_tensor * iq_pe = ggml_view_3d(ctx, iq, n_rope, n_idx_head, 1,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head,
                ggml_row_size(iq->type, n_nope));
        iq_pe = rope_compress(iq_pe, pos_i);
        iq = ggml_concat(ctx, iq_nope, iq_pe, 0);

        ggml_tensor * iw = ggml_mul_mat(ctx, L.indexer_proj, cur_i);
        iw = ggml_scale(ctx, iw, 1.0f / sqrtf((float) (idx_head * n_idx_head)));

        ggml_tensor * ik = ggml_view_2d(ctx, lid_cache, idx_head, n_lid, lid_cache->nb[1], 0);
        ik = ggml_reshape_3d(ctx, ik, idx_head, 1, n_lid);

        ggml_tensor * score = ggml_lightning_indexer(ctx, iq, ik, iw, lid_mask);
        const int64_t k = std::min<int64_t>(hp.indexer_top_k, score->ne[0]);
        return ggml_cont(ctx, ggml_top_k(ctx, score, (int) k));
    }

    ggml_tensor * build_attention_batched(int il, ggml_tensor * cur, ggml_tensor * inp_pos)
    {
        const dsv4_layer & L = m.layers[il];
        const int dev = L.device;
        const int64_t head = hp.n_embd_head;
        const int64_t n_head = hp.n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_groups = hp.o_groups;
        const int64_t o_lora = hp.o_lora_rank;
        const int64_t o_group_dim = (n_head / n_groups) * head;
        const int32_t ratio = hp.compress_ratios[il];
        ggml_tensor * rope_tab = m.rope_tab_dev[ratio != 0 ? 1 : 0][dev];
        const int64_t N = nt;
        const float kq_scale = 1.0f / sqrtf((float) head);

        // ---- dense prologue, batched over the N tokens ----
        ggml_tensor * qr = ggml_mul_mat(ctx, L.wq_a, cur);
        qr = rms(qr, L.attn_q_a_norm);
        ggml_tensor * q_raw = ggml_mul_mat(ctx, L.wq_b, qr);    // [head*n_head, N]
        ggml_tensor * kv_raw = ggml_mul_mat(ctx, L.wkv, cur);   // [head, N]

        ggml_tensor * cs_kv = nullptr, * cs_score = nullptr;
        ggml_tensor * ls_kv = nullptr, * ls_score = nullptr;
        if (ratio == CSA_RATIO)
        {
            cs_kv = ggml_mul_mat(ctx, L.attn_comp_wkv, cur);
            cs_score = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_comp_wgate, cur),
                    ggml_get_rows(ctx, L.attn_comp_ape, res.inp.csa[dev].state_pos));
            ls_kv = ggml_mul_mat(ctx, L.indexer_comp_wkv, cur);
            ls_score = ggml_add(ctx, ggml_mul_mat(ctx, L.indexer_comp_wgate, cur),
                    ggml_get_rows(ctx, L.indexer_comp_ape, res.inp.csa[dev].state_pos));
        }
        else if (ratio == HCA_RATIO)
        {
            cs_kv = ggml_mul_mat(ctx, L.attn_comp_wkv, cur);
            cs_score = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_comp_wgate, cur),
                    ggml_get_rows(ctx, L.attn_comp_ape, res.inp.hca[dev].state_pos));
        }

        // ---- per-slot cache ops + attention ----
        ggml_tensor * attn_cat = nullptr;
        for (int64_t i = 0; i < N; i++)
        {
            bd_slot_state & B = res.bd[i];
            const dsv4_slot_layer & C = m.slots.at(B.slot_id)->layers[il];
            ggml_tensor * pos_i = ggml_view_1d(ctx, inp_pos, 1, i * sizeof(int32_t));

            ggml_tensor * q = fused_node(TSG_DSV4_FUSED_ATTN_PREP, GGML_TYPE_F32, head, n_head, 1, 1,
                    { col_view(q_raw, i), col_view(kv_raw, i), L.attn_kv_norm, rope_tab, pos_i,
                      C.raw_k, B.raw_idxs[dev] },
                    dev, (int) n_rope, 0, 0, 0, hp.rms_eps);
            ggml_build_forward_expand(gf, q);

            auto emit_compress = [&](ggml_tensor * st_kv, ggml_tensor * st_score,
                                     ggml_tensor * state_kv, ggml_tensor * state_score,
                                     ggml_tensor * norm_w, ggml_tensor * cache,
                                     int64_t cratio, bool overlap,
                                     ggml_tensor * meta, const comp_plan & plan)
            {
                const int n_blocks = (int) plan.state_write_idxs.size();
                const int np = (int) plan.state_persist_src_idxs.size();
                ggml_tensor * marker = fused_node(TSG_DSV4_FUSED_COMPRESS, GGML_TYPE_F32, 1, 1, 1, 1,
                        { st_kv, st_score, state_kv, state_score, norm_w, m.rope_tab_dev[1][dev],
                          meta, cache },
                        dev, n_blocks, (int) cratio, overlap ? 2 : 1, hp.n_rot,
                        hp.rms_eps, (float) np);
                ggml_build_forward_expand(gf, marker);
            };
            if (ratio == CSA_RATIO)
            {
                emit_compress(col_view(cs_kv, i), col_view(cs_score, i),
                              C.comp_state_kv, C.comp_state_score, L.attn_comp_norm, C.csa_k,
                              CSA_RATIO, true, B.csa_meta[dev], B.plan_csa);
                emit_compress(col_view(ls_kv, i), col_view(ls_score, i),
                              C.lid_state_kv, C.lid_state_score, L.indexer_comp_norm, C.lid_k,
                              CSA_RATIO, true, B.lid_meta[dev], B.plan_lid);
            }
            else if (ratio == HCA_RATIO)
            {
                emit_compress(col_view(cs_kv, i), col_view(cs_score, i),
                              C.comp_state_kv, C.comp_state_score, L.attn_comp_norm, C.hca_k,
                              HCA_RATIO, false, B.hca_meta[dev], B.plan_hca);
            }

            ggml_tensor * raw_k = ggml_view_2d(ctx, C.raw_k, head, m.ring_raw, C.raw_k->nb[1], 0);
            raw_k = ggml_reshape_3d(ctx, raw_k, head, 1, m.ring_raw);

            ggml_tensor * out;
            if (ratio == CSA_RATIO && !B.skip_topk)
            {
                ggml_tensor * top_k = bd_lid_top_k(il, C.lid_k, col_view(qr, i), col_view(cur, i),
                                                   pos_i, B.lid_mask[dev], B.plan_lid.n_kv);
                ggml_tensor * k_sel = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16,
                        head, 1, m.ring_raw + top_k->ne[0], 1,
                        { C.raw_k, C.csa_k, top_k }, dev, (int) m.ring_raw);
                out = attn_mha(q, k_sel, B.gather_mask[dev], L.attn_sinks, kq_scale);
            }
            else if (ratio == CSA_RATIO)
            {
                ggml_tensor * csa_k = ggml_view_2d(ctx, C.csa_k, head, B.plan_csa.n_kv, C.csa_k->nb[1], 0);
                csa_k = ggml_reshape_3d(ctx, csa_k, head, 1, B.plan_csa.n_kv);
                ggml_tensor * k_all = ggml_concat(ctx, raw_k, csa_k, 2);
                ggml_tensor * kq_mask = ggml_concat(ctx, B.raw_mask[dev], B.csa_mask[dev], 0);
                out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
            }
            else if (ratio == HCA_RATIO)
            {
                ggml_tensor * hca_k = ggml_view_2d(ctx, C.hca_k, head, B.plan_hca.n_kv, C.hca_k->nb[1], 0);
                hca_k = ggml_reshape_3d(ctx, hca_k, head, 1, B.plan_hca.n_kv);
                ggml_tensor * k_all = ggml_concat(ctx, raw_k, hca_k, 2);
                ggml_tensor * kq_mask = ggml_concat(ctx, B.raw_mask[dev], B.hca_mask[dev], 0);
                out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
            }
            else
            {
                out = attn_mha(q, raw_k, B.raw_mask[dev], L.attn_sinks, kq_scale);
            }
            attn_cat = attn_cat ? ggml_concat(ctx, attn_cat, out, 1) : out;   // [head*n_head, N]
        }

        // ---- batched epilogue: inverse rope + grouped LoRA out (weights once) ----
        ggml_tensor * grouped = fused_node(TSG_DSV4_FUSED_ATTN_FINISH, GGML_TYPE_F32, o_group_dim, N, n_groups, 1,
                { attn_cat, rope_tab, inp_pos }, dev, (int) n_rope, (int) n_groups, (int) head);
        ggml_tensor * oa = ggml_mul_mat(ctx,
                ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], o_lora, n_groups), grouped);
        oa = ggml_permute(ctx, oa, 0, 2, 1, 3);
        oa = ggml_cont_2d(ctx, oa, o_lora * n_groups, N);
        return ggml_mul_mat(ctx, L.wo_b, oa);
    }

    static size_t bd_meta_size(const comp_plan & plan)
    {
        return plan.state_read_idxs.size()
            + 2 * plan.state_write_idxs.size()
            + 2 * plan.state_persist_src_idxs.size();
    }

    // V4.1 token-batched decode: the same layer stack as build(), with one
    // decode token per sequence slot. Everything except the per-slot attention
    // fork is shared, including the Engram lookup (its staged rows are already
    // one column per token, so a slot is just a column) and the whole MoE.
    void build_batched_v41()
    {
        graph_inputs & inp = res.inp;
        const int64_t N = nt;

        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;
        const int dev_last = m.layers[hp.n_layer - 1].device;

        auto make_slot_plan_inputs = [&](plan_inputs & pi, const comp_plan & plan,
                                         const char * tag, size_t slot, int d)
        {
            char nb[128];
            pi.n_kv = plan.n_kv;
            snprintf(nb, sizeof(nb), "bd%zu_%s_persist_src.%d", slot, tag, d);
            pi.persist_src = new_input_i32(plan.state_persist_src_idxs.size(), nb, d);
            snprintf(nb, sizeof(nb), "bd%zu_%s_persist_dst.%d", slot, tag, d);
            pi.persist_dst = new_input_i64(plan.state_persist_dst_idxs.size(), nb, d);
            if (!plan.state_write_idxs.empty())
            {
                snprintf(nb, sizeof(nb), "bd%zu_%s_read_idxs.%d", slot, tag, d);
                pi.read_idxs = new_input_i32(plan.state_read_idxs.size(), nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_%s_write_idxs.%d", slot, tag, d);
                pi.write_idxs = new_input_i64(plan.state_write_idxs.size(), nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_%s_write_pos.%d", slot, tag, d);
                pi.write_pos = new_input_i32(plan.state_write_pos.size(), nb, d);
            }
            snprintf(nb, sizeof(nb), "bd%zu_%s_mask.%d", slot, tag, d);
            pi.kq_mask = new_input_mask1(plan.n_kv, nb, d);
        };

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[96];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input_i32(N, nb, d);
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input_i32(N, nb, d);

            for (size_t i = 0; i < res.bd.size(); i++)
            {
                bd_slot_state & B = res.bd[i];
                snprintf(nb, sizeof(nb), "bd%zu_raw_idxs.%d", i, d);
                B.raw_idxs[d] = new_input_i64(1, nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_raw_mask.%d", i, d);
                B.raw_mask[d] = new_input_mask1(m.ring_raw, nb, d);
                if (B.use_gather)
                {
                    snprintf(nb, sizeof(nb), "bd%zu_gather_mask.%d", i, d);
                    B.gather_mask[d] = new_input_mask1(
                        (B.compact_raw ? dsv41_compact_raw_rows(m) : m.ring_raw) + hp.indexer_top_k, nb, d);
                    if (B.compact_raw)
                    {
                        snprintf(nb, sizeof(nb), "bd%zu_raw_read_idxs.%d", i, d);
                        B.raw_read_idxs[d] = new_input_i32(dsv41_compact_raw_rows(m), nb, d);
                    }
                }
                make_slot_plan_inputs(B.v41_csa[d], B.plan_csa, "csa", i, d);
                make_slot_plan_inputs(B.v41_hca[d], B.plan_hca, "hca", i, d);
                make_slot_plan_inputs(B.v41_lid[d], B.plan_lid, "lid", i, d);
            }
        }
        inp.out_ids = new_input_i32(N, "inp_out_ids", dev_last);

        inp.engram.resize(m.engram.layers.size());
        inp.engram_ids.resize(m.engram.layers.size());
        inp.engram_rows = new_input_i32(hp.hc_mult, "engram_stream_rows", m.layers[m.engram.layers[0].id].device);
        for (size_t e = 0; e < m.engram.layers.size(); ++e)
        {
            const int dev = m.layers[m.engram.layers[e].id].device;
            if (m.engram_on_device)
            {
                auto * ids = new_input_i32(m.engram.hash_columns() * N, "engram_ids", dev);
                ggml_format_name(ids, "engram_ids.%d", (int) e);
                inp.engram_ids[e] = ids;
                continue;
            }
            auto * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.engram.hash_columns() * m.engram.head_dim, N);
            ggml_set_input(rows);
            ggml_format_name(rows, "engram_lookup.%d", (int) e);
            pin(rows, dev);
            inp.engram[e] = rows;
        }

        const int64_t hc = hp.hc_mult;
        ggml_tensor * delayed_pre = nullptr;
        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);
        ggml_tensor * inpL = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, hp.n_embd, 1, N), hp.n_embd, hc, N, 1);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const dsv4_layer & L = m.layers[il];
            const int dev = L.device;

            if (il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            inpL = build_engram(il, inpL);
            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;

            ggml_tensor * attn_pre = nullptr;
            ggml_tensor * cur = build_hc_pre(inpL, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base,
                &post, &comb, dev, &attn_pre, delayed_pre);
            cur = rms(cur, L.attn_norm);
            cur = build_attention_v41_batched(il, cur, inp.pos[dev]);
            inpL = build_hc_post(cur, residual, post, comb);

            residual = inpL;
            cur = build_hc_pre(inpL, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base,
                &post, &comb, dev, &delayed_pre, attn_pre);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            cur = rms(cur, L.ffn_norm);
            ggml_tensor * shexp = build_shexp(il, cur);
            cur = build_moe(il, cur, inp.tokens[dev], shexp);
            inpL = build_hc_post(cur, residual, post, comb);
        }

        ggml_tensor * flat = ggml_get_rows(ctx, ggml_reshape_2d(ctx, inpL, hp.n_embd * hc, N), inp.out_ids);
        inpL = ggml_reshape_3d(ctx, flat, hp.n_embd, hc, N);
        ggml_tensor * cur = build_hc_pre_op(inpL, ggml_get_rows(ctx, ggml_cont(ctx, delayed_pre), inp.out_ids));
        cur = rms(cur, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);

        // Same F32 projection policy as build(): TF32 truncation can move cache
        // quantization and sparse routing across bin boundaries.
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
        {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
                node->src[0]->type == GGML_TYPE_F32)
            {
                precise_matmul(node);
            }
        }
    }

    void build_batched()
    {
        graph_inputs & inp = res.inp;
        const int64_t N = nt;

        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;
        const int dev_last = m.layers[hp.n_layer - 1].device;

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[96];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input_i32(N, nb, d);
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input_i32(N, nb, d);
            // batched APE row selectors (pos % ratio), shared across slots;
            // csa's doubles for the lid compressor (same ratio)
            snprintf(nb, sizeof(nb), "inp_csa_state_pos.%d", d);
            inp.csa[d].state_pos = new_input_i32(N, nb, d);
            snprintf(nb, sizeof(nb), "inp_hca_state_pos.%d", d);
            inp.hca[d].state_pos = new_input_i32(N, nb, d);

            for (size_t i = 0; i < res.bd.size(); i++)
            {
                bd_slot_state & B = res.bd[i];
                snprintf(nb, sizeof(nb), "bd%zu_raw_idxs.%d", i, d);
                B.raw_idxs[d] = new_input_i64(1, nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_raw_mask.%d", i, d);
                B.raw_mask[d] = new_input_mask1(m.ring_raw, nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_csa_meta.%d", i, d);
                B.csa_meta[d] = new_input_i32(std::max<size_t>(bd_meta_size(B.plan_csa), 1), nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_lid_meta.%d", i, d);
                B.lid_meta[d] = new_input_i32(std::max<size_t>(bd_meta_size(B.plan_lid), 1), nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_hca_meta.%d", i, d);
                B.hca_meta[d] = new_input_i32(std::max<size_t>(bd_meta_size(B.plan_hca), 1), nb, d);
                snprintf(nb, sizeof(nb), "bd%zu_hca_mask.%d", i, d);
                B.hca_mask[d] = new_input_mask1(B.plan_hca.n_kv, nb, d);
                if (B.skip_topk)
                {
                    snprintf(nb, sizeof(nb), "bd%zu_csa_mask.%d", i, d);
                    B.csa_mask[d] = new_input_mask1(B.plan_csa.n_kv, nb, d);
                }
                else
                {
                    snprintf(nb, sizeof(nb), "bd%zu_lid_mask.%d", i, d);
                    B.lid_mask[d] = new_input_mask1(B.plan_lid.n_kv, nb, d);
                    snprintf(nb, sizeof(nb), "bd%zu_gather_mask.%d", i, d);
                    B.gather_mask[d] = new_input_mask1(m.ring_raw + hp.indexer_top_k, nb, d);
                }
            }
        }
        inp.out_ids = new_input_i32(N, "inp_out_ids", dev_last);

        const int64_t hc = hp.hc_mult;
        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);
        ggml_tensor * inpL = ggml_reshape_3d(ctx, emb, hp.n_embd, 1, N);
        inpL = ggml_repeat_4d(ctx, inpL, hp.n_embd, hc, N, 1);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const dsv4_layer & L = m.layers[il];
            const int dev = L.device;

            if (il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;

            ggml_tensor * cur = build_hc_pre(inpL, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &post, &comb, dev);
            cur = rms(cur, L.attn_norm);
            cur = build_attention_batched(il, cur, inp.pos[dev]);
            inpL = build_hc_post(cur, residual, post, comb);

            residual = inpL;
            cur = build_hc_pre(inpL, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &post, &comb, dev);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            cur = rms(cur, L.ffn_norm);

            ggml_tensor * shexp = build_shexp(il, cur);
            cur = build_moe(il, cur, inp.tokens[dev], shexp);

            inpL = build_hc_post(cur, residual, post, comb);
        }

        // logits for every slot's token
        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hp.n_embd * hc, N);
        flat = ggml_get_rows(ctx, flat, inp.out_ids);
        inpL = ggml_reshape_3d(ctx, flat, hp.n_embd, hc, N);

        ggml_tensor * cur = build_hc_head(inpL);
        cur = rms(cur, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // ---- whole graph ----
    void build()
    {
        graph_inputs & inp = res.inp;

        // devices that host at least one layer
        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;
        const int dev_last = m.layers[hp.n_layer - 1].device;

        auto make_plan_inputs = [&](plan_inputs & pi, const comp_plan & plan, ggml_type mask_type, const char * tag, int d)
        {
            char nb[128];
            pi.n_kv = plan.n_kv;
            snprintf(nb, sizeof(nb), "inp_%s_state_pos.%d", tag, d);
            pi.state_pos = new_input_i32(plan.state_pos.size(), nb, d);
            if (m.fused && !hp.v41)
            {
                const size_t n_meta = plan.state_read_idxs.size()
                    + 2 * plan.state_write_idxs.size()
                    + 2 * plan.state_persist_src_idxs.size();
                snprintf(nb, sizeof(nb), "inp_%s_comp_meta.%d", tag, d);
                pi.comp_meta = new_input_i32(std::max<size_t>(n_meta, 1), nb, d);
            }
            else
            {
                snprintf(nb, sizeof(nb), "inp_%s_persist_src.%d", tag, d);
                pi.persist_src = new_input_i32(plan.state_persist_src_idxs.size(), nb, d);
                snprintf(nb, sizeof(nb), "inp_%s_persist_dst.%d", tag, d);
                pi.persist_dst = new_input_i64(plan.state_persist_dst_idxs.size(), nb, d);
                if (!plan.state_write_idxs.empty())
                {
                    snprintf(nb, sizeof(nb), "inp_%s_read_idxs.%d", tag, d);
                    pi.read_idxs = new_input_i32(plan.state_read_idxs.size(), nb, d);
                    snprintf(nb, sizeof(nb), "inp_%s_write_idxs.%d", tag, d);
                    pi.write_idxs = new_input_i64(plan.state_write_idxs.size(), nb, d);
                    snprintf(nb, sizeof(nb), "inp_%s_write_pos.%d", tag, d);
                    pi.write_pos = new_input_i32(plan.state_write_pos.size(), nb, d);
                }
            }
            snprintf(nb, sizeof(nb), "inp_%s_mask.%d", tag, d);
            pi.kq_mask = new_input_mask(plan.n_kv, mask_type, nb, d);
        };

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input_i32(nt, nb, d);
            if (res.image_tokens)
            {
                snprintf(nb, sizeof(nb), "inp_image_types.%d", d);
                inp.image_types[d] = new_input_i32(nt, nb, d);
            }
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input_i32(nt, nb, d);
            snprintf(nb, sizeof(nb), "inp_raw_idxs.%d", d);
            inp.raw_idxs[d] = new_input_i64(nt, nb, d);
            snprintf(nb, sizeof(nb), "inp_raw_mask.%d", d);
            inp.raw_mask[d] = new_input_mask(m.ring_raw, GGML_TYPE_F16, nb, d);
            if (hp.v41 ? dsv41_use_gather(m, nt, p0)
                       : dsv4_use_gather(m, nt, std::max(m.pos_end_hint, p0 + nt)))
            {
                snprintf(nb, sizeof(nb), "inp_gather_mask.%d", d);
                const bool compact = dsv41_use_compact_raw_gather(m, nt, p0);
                inp.gather_mask[d] = new_input_mask((compact ? dsv41_compact_raw_rows(m) : m.ring_raw) + hp.indexer_top_k, GGML_TYPE_F16, nb, d);
                if (compact)
                {
                    snprintf(nb, sizeof(nb), "inp_raw_read_idxs.%d", d);
                    inp.raw_read_idxs[d] = new_input_i32(dsv41_compact_raw_rows(m), nb, d);
                }
            }
            make_plan_inputs(inp.csa[d], res.plan_csa, GGML_TYPE_F16, "csa", d);
            make_plan_inputs(inp.hca[d], res.plan_hca, GGML_TYPE_F16, "hca", d);
            make_plan_inputs(inp.lid[d], res.plan_lid, GGML_TYPE_F16, "lid", d);
        }
        inp.out_ids = new_input_i32(res.all_logits ? nt : 1, "inp_out_ids", dev_last);

        if (hp.v41)
        {
            inp.engram.resize(m.engram.layers.size());
            inp.engram_ids.resize(m.engram.layers.size());
            inp.engram_rows = new_input_i32(hp.hc_mult, "engram_stream_rows", m.layers[m.engram.layers[0].id].device);
            for (size_t e = 0; e < m.engram.layers.size(); ++e)
            {
                const int dev = m.layers[m.engram.layers[e].id].device;
                if (m.engram_on_device)
                {
                    // One id per (token, hash column); build_engram gathers the
                    // quantized rows on this device. 4 bytes per row crosses the
                    // link instead of head_dim F32 values.
                    auto * ids = new_input_i32(m.engram.hash_columns() * nt, "engram_ids", dev);
                    ggml_format_name(ids, "engram_ids.%d", (int) e);
                    inp.engram_ids[e] = ids;
                    continue;
                }
                auto * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.engram.hash_columns() * m.engram.head_dim, nt);
                ggml_set_input(rows);
                ggml_format_name(rows, "engram_lookup.%d", (int) e);
                pin(rows, dev);
                inp.engram[e] = rows;
            }
        }
        const int64_t hc = hp.hc_mult;
        ggml_tensor * delayed_pre = nullptr;

        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);   // [n_embd, nt]
        if (res.image_tokens)
        {
            const int dev = m.layers[0].device;
            inp.image_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, res.image_tokens);
            ggml_set_input(inp.image_embeddings);
            ggml_set_name(inp.image_embeddings, "inp_image_embeddings");
            pin(inp.image_embeddings, dev);
            inp.image_indices = new_input_i64(res.image_tokens, "inp_image_indices", dev);
            emb = ggml_set_rows(ctx, emb, inp.image_embeddings, inp.image_indices);
            pin(emb, dev);
        }
        ggml_tensor * inpL = ggml_reshape_3d(ctx, emb, hp.n_embd, 1, nt);
        inpL = ggml_repeat_4d(ctx, inpL, hp.n_embd, hc, nt, 1);

        std::vector<ggml_tensor *> ds_feats;

        for (int il = 0; il < hp.n_layer; il++)
        {
            const dsv4_layer & L = m.layers[il];
            const int dev = L.device;

            // At a device boundary, pin the incoming activations to the new
            // device. Without this the scheduler ping-pongs the boundary
            // layer's elementwise ops between GPUs (x-derived sources vote for
            // the old device, weight views for the new one), costing ~6 extra
            // graph splits per eval.
            if (il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            if (hp.v41) inpL = build_engram(il, inpL);
            trace_v41(inpL, il, "engram_output");
            if (hp.v41 && m.ds.loaded &&
                std::find(m.ds.target_layers.begin(), m.ds.target_layers.end(), il) != m.ds.target_layers.end())
            {
                auto * feature = hc_mean(inpL);
                pin(feature, dev);
                ds_feats.push_back(feature);
            }
            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;

            ggml_tensor * attn_pre = nullptr;
            ggml_tensor * cur = build_hc_pre(inpL, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &post, &comb, dev,
                hp.v41 ? &attn_pre : nullptr, delayed_pre);
            cur = rms(cur, L.attn_norm);
            cur = hp.v41 ? build_attention_v41(il, cur, inp.pos[dev]) : build_attention(il, cur, inp.pos[dev]);
            inpL = build_hc_post(cur, residual, post, comb);

            residual = inpL;
            cur = build_hc_pre(inpL, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &post, &comb, dev,
                hp.v41 ? &delayed_pre : nullptr, attn_pre);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            cur = rms(cur, L.ffn_norm);

            ggml_tensor * shexp = build_shexp(il, cur);
            cur = build_moe(il, cur, inp.tokens[dev], shexp);
            trace_v41(cur, il, "ffn_output");

            inpL = build_hc_post(cur, residual, post, comb);
            trace_v41(inpL, il, "output");

            if (!hp.v41 && m.ds.loaded &&
                std::find(m.ds.target_layers.begin(), m.ds.target_layers.end(), il) != m.ds.target_layers.end())
            {
                auto * feature = hc_mean(inpL);
                pin(feature, dev);
                ds_feats.push_back(feature);
            }
        }

        if (m.ds.loaded && ds_feats.size() == m.ds.target_layers.size())
            build_dspark_ring_update(ds_feats, inp.pos[m.ds.dev], inp.raw_idxs[m.ds.dev]);

        // gather output row(s)
        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hp.n_embd * hc, nt);
        flat = ggml_get_rows(ctx, flat, inp.out_ids);
        const int64_t n_out = res.all_logits ? nt : 1;
        inpL = ggml_reshape_3d(ctx, flat, hp.n_embd, hc, n_out);

        ggml_tensor * cur = hp.v41
            ? build_hc_pre_op(inpL, ggml_get_rows(ctx, ggml_cont(ctx, delayed_pre), inp.out_ids))
            : build_hc_head(inpL);
        cur = rms(cur, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        trace_v41(cur, hp.n_layer, "logits");
        res.logits = cur;

        ggml_build_forward_expand(gf, cur);
        if (hp.v41)
        {
            // TF32 truncation can move cache quantization and sparse routing
            // across bin boundaries. Preserve F32 projections; quantized
            // weights continue using the optimized quantized matmul kernels.
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
            {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
                    node->src[0]->type == GGML_TYPE_F32)
                {
                    precise_matmul(node);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// Unreferenced per-device input tensors (e.g. tokens on a device with no hash
// MoE layer) never enter the graph and stay unallocated — skip those.
static void set_i32(ggml_tensor * t, const std::vector<int32_t> & v)
{
    if (!t || !t->buffer || v.empty()) return;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
}

static void set_i64(ggml_tensor * t, const std::vector<int64_t> & v)
{
    if (!t || !t->buffer || v.empty()) return;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int64_t));
}

static uint64_t dsv4_plan_sig(const comp_plan & p)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix((uint64_t) p.state_pos.size());
    mix((uint64_t) p.state_persist_src_idxs.size());
    mix((uint64_t) p.state_read_idxs.size());
    mix((uint64_t) p.state_write_idxs.size());
    mix((uint64_t) p.n_kv);
    return h;
}

// ---------------------------------------------------------------------------
// Cross-backend node diff (TS_DSV4_NODE_DUMP=<path>)
// ---------------------------------------------------------------------------
// Writes one line per executed node — index, name, op, shape, backend, and a
// value summary — so two runs on different backends can be diffed to find the
// FIRST op whose output disagrees. The non-fused graph (TS_DSV4_FUSED=0 on
// CUDA, which is what Vulkan builds anyway) is node-for-node identical across
// backends, so the line numbers line up.
//
// Reads every node back to the host, so it is orders of magnitude slower than a
// real run and is only ever enabled by hand.
namespace
{
    FILE *  g_node_dump = nullptr;
    int     g_node_dump_idx = 0;
    int     g_node_dump_limit = 0;

    bool dsv4_node_dump_cb(ggml_tensor * t, bool ask, void * user_data)
    {
        ggml_backend_sched_t sched = (ggml_backend_sched_t) user_data;
        if (ask) return true;
        if (!g_node_dump) return true;

        const int idx = g_node_dump_idx++;
        if (g_node_dump_limit > 0 && idx >= g_node_dump_limit) return true;

        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, t);
        const int64_t n = ggml_nelements(t);
        double sum = 0.0, amax = 0.0;
        int64_t nan_count = 0;

        // Summarize as F32 where the type allows it; anything else is reported
        // by shape alone (its consumers show up as F32 further down).
        if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_I32)
        {
            std::vector<uint8_t> host(ggml_nbytes(t));
            ggml_backend_tensor_get(t, host.data(), 0, host.size());
            for (int64_t i = 0; i < n; i++)
            {
                double v = 0.0;
                if (t->type == GGML_TYPE_F32)      v = ((const float *) host.data())[i];
                else if (t->type == GGML_TYPE_F16) v = ggml_fp16_to_fp32(((const ggml_fp16_t *) host.data())[i]);
                else                               v = ((const int32_t *) host.data())[i];
                if (std::isnan(v)) { nan_count++; continue; }
                if (std::isinf(v)) continue;   // masks are legitimately -inf
                sum += v;
                amax = std::max(amax, std::fabs(v));
            }
        }

        fprintf(g_node_dump, "%6d %-22s %-18s [%5" PRId64 ",%5" PRId64 ",%5" PRId64 "] %-10s sum=%+.6e amax=%.6e nan=%" PRId64 "\n",
                idx, t->name, ggml_op_name(t->op), t->ne[0], t->ne[1], t->ne[2],
                be ? ggml_backend_name(be) : "?", sum, amax, nan_count);
        return true;
    }

    // Install on a freshly built graph's scheduler; no-op unless the env is set.
    void dsv4_node_dump_attach(ggml_backend_sched_t sched)
    {
        static const char * path = getenv("TS_DSV4_NODE_DUMP");
        if (!path || !*path) return;
        if (!g_node_dump)
        {
            g_node_dump = fopen(path, "w");
            if (const char * e = getenv("TS_DSV4_NODE_DUMP_LIMIT")) g_node_dump_limit = atoi(e);
            if (!g_node_dump) { fprintf(stderr, "[dsv4] cannot open %s for node dump\n", path); return; }
        }
        ggml_backend_sched_set_eval_callback(sched, dsv4_node_dump_cb, sched);
    }
}

// Keep the graph cache inside the VRAM it is allowed to use.
//
// The cache is capped by entry COUNT, which is blind to what an entry costs:
// a decode graph's compute buffers are small, a 1024-token prefill graph's are
// hundreds of MiB per device, and concurrent sequences sitting at different
// positions produce many distinct shapes. Four concurrent 10.8k-token prefills
// on the eight-A40 box filled the twelve slots and ran device 1 out of memory.
//
// Explicit checked reservation below lets an allocation failure reject the
// graph safely. Trimming first also avoids unnecessary failures and allocation
// churn when older entries can release enough memory for the next shape.
//
// The rule: before building a new entry, free least-recently-used entries
// until every device has room for another entry as large as the largest one
// cached, plus a floor for the run-time transients that do not come out of a
// graph buffer at all (ggml-cuda takes the indexer's CUB sort workspace
// straight from the VMM pool). TS_DSV4_GRAPH_CACHE_HEADROOM_MB overrides the
// floor; 0 restores the pure count cap.
static void dsv4_trim_graph_cache(dsv4_model & m)
{
    static const size_t floor_bytes = []() -> size_t {
        if (const char * e = getenv("TS_DSV4_GRAPH_CACHE_HEADROOM_MB"))
        {
            const long v = atol(e);
            if (v >= 0) return (size_t) v * 1024 * 1024;
        }
        return (size_t) 1024 * 1024 * 1024;
    }();
    if (!floor_bytes || m.graph_cache.empty()) return;

    size_t want[MAX_GPUS] = {};
    for (int d = 0; d < m.n_gpu; d++)
    {
        size_t largest = 0;
        for (const auto & entry : m.graph_cache) largest = std::max(largest, entry->buffer_bytes[d]);
        want[d] = largest + floor_bytes;
    }
    auto tight = [&]() -> bool
    {
        for (int d = 0; d < m.n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m.backends[d]), &free_b, &total_b);
            if (free_b < want[d]) return true;
        }
        return false;
    };
    if (!tight()) return;

    // An entry's buffers may still be backing kernels this process launched
    // (pipelined prefill submits without waiting), so drain before freeing.
    for (int d = 0; d < m.n_gpu; d++) ggml_backend_synchronize(m.backends[d]);
    int dropped = 0;
    while (m.graph_cache.size() > 1 && tight())
    {
        m.graph_cache.pop_back();
        dropped++;
    }
    if (dropped)
        fprintf(stderr, "[dsv4] graph cache trimmed: freed %d least-recently-used entr%s to keep "
                "device memory for the next graph (%zu cached)\n",
                dropped, dropped == 1 ? "y" : "ies", m.graph_cache.size());
}

// Build (or fetch from the LRU cache) the graph entry for a (nt, p0,
// pipeline-parity) shape. Each entry owns its scheduler/allocation, so
// alternating shapes never rebuild, realloc, or thrash the per-split CUDA
// graphs captured by ggml-cuda. Fresh entries device-synchronize on their
// arena allocation — TSGgml_Dsv4Forward pre-acquires the first two prefill
// chunks so this happens while the devices are idle.
static graph_build_result * dsv4_acquire_graph(dsv4_model & m, int64_t nt, int64_t p0, bool pipeline, bool * out_reuse,
                                              bool all_logits = false, int image_tokens = 0)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (dsv4_test_boundary_fault()) return nullptr;
#endif
    const dsv4_hparams & hp = m.hp;
    static const int perf_build = []() { const char * e = getenv("TS_DSV4_PERF"); return e ? atoi(e) : 0; }();

    // Uniform shape across a forward call: bucket by the call's final
    // position (capped so very long prefills don't over-pad early attention)
    // and decide the indexer skip by the final position, so every chunk of a
    // prefill shares one graph-shape pair and the pipeline never stalls on a
    // mid-flight allocation.
    const int64_t pos_end = std::max(m.pos_end_hint, p0 + nt);
    const int64_t hint = std::min<int64_t>(pos_end, 8192);

    const int cr = hp.v41 ? 2 : CSA_RATIO, hr = hp.v41 ? 1 : HCA_RATIO;
    comp_plan plan_csa = build_comp_plan(p0, nt, cr, !hp.v41, (hp.v41 ? cr : 2 * cr) + m.state_extra, m.n_csa_rows, hint / cr);
    comp_plan plan_hca = build_comp_plan(p0, nt, hr, false, hr + m.state_extra, m.n_hca_rows, hint / hr);
    comp_plan plan_lid = plan_csa;

    uint64_t sig = 14695981039346656037ull ^ (uint64_t) nt;
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_csa);
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_hca);
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_lid);
    // graphs bake the active slot's cache tensor addresses
    sig = sig * 1099511628211ull ^ (uint64_t) (m.active_slot->id + 1);
    // the short-context indexer skip changes the graph shape
    if (!hp.v41 && m.fused && pos_end <= (int64_t) hp.indexer_top_k * CSA_RATIO)
        sig ^= 0x9e3779b97f4a7c15ull;
    // Dense and gathered decode can share all compressor-plan dimensions at
    // the visibility threshold, but have different attention/input shapes.
    if (dsv41_use_gather(m, nt, p0))
        sig ^= 0xbefca6a34d210597ull;
    if (dsv41_use_compact_raw_gather(m, nt, p0))
        sig ^= 0xe41362d99b3ca810ull;
    // pipelined chunks alternate between two entries so in-flight inputs stay private
    if (pipeline && ((p0 / std::max<int64_t>(nt, 1)) & 1))
        sig ^= 0x517cc1b727220a95ull;
    // per-row logits change the head's shape
    if (all_logits)
        sig ^= 0xd6e8feb86659fd93ull;
    if (image_tokens)
        sig = (sig ^ 0x7adb9c312d4508e1ull) * 1099511628211ull ^ (uint64_t) image_tokens;

    graph_build_result * res_p = nullptr;
    bool reuse = false;
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        if ((*it)->sig == sig)
        {
            if (it != m.graph_cache.begin())
                m.graph_cache.splice(m.graph_cache.begin(), m.graph_cache, it);
            res_p = m.graph_cache.front().get();
            reuse = true;
            break;
        }
    }

    if (!reuse)
    {
        // A failed build must never become a reusable cache entry after
        // the caller resets its sequence. Keep ownership local until complete.
        auto pending = std::make_unique<graph_build_result>();
        graph_build_result & r = *pending;
        dsv4_trim_graph_cache(m);
        const size_t meta_size = (size_t) 96 * 1024 * 1024;
        ggml_init_params gp = { meta_size, nullptr, true };
        r.ctx = ggml_init(gp);
        r.gf = ggml_new_graph_custom(r.ctx, 32768, false);
        r.nt = nt;
        r.sig = sig;
        r.slot_id = m.active_slot->id;
        r.plan_csa = plan_csa;
        r.plan_hca = plan_hca;
        r.plan_lid = plan_lid;
        r.all_logits = all_logits;
        r.image_tokens = image_tokens;
        r.sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, 32768, false, true);
        if (!r.sched)
        {
            fprintf(stderr, "[dsv4] sched creation failed\n");
            return nullptr;
        }

        graph_builder gb(m, r, nt, p0);
        gb.build();
#if defined(TSG_GGML_TEST_HOOKS)
        if (hp.v41) dsv41_test_fail("graph", p0);
#endif

        if (!tsg_scheduler_alloc_graph(r.sched, r.gf))
        {
            fprintf(stderr, "[dsv4] sched_alloc_graph failed (nt=%" PRId64 ")\n", nt);
            return nullptr;
        }
        dsv4_node_dump_attach(r.sched);
        if (perf_build >= 3)
        {
            // What the node count of a decode/prefill graph is actually made of.
            // Views and reshapes cost nothing at run time; everything else is a
            // kernel, and a kernel between two fused nodes also ends an ordinary
            // run (see the submission counters).
            std::map<std::string, int> hist;
            int real = 0;
            for (int i = 0; i < ggml_graph_n_nodes(r.gf); i++)
            {
                ggml_tensor * node = ggml_graph_node(r.gf, i);
                const bool noop = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                    node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
                if (!noop) real++;
                hist[ggml_op_name(node->op)]++;
            }
            fprintf(stderr, "[dsv4] graph nt=%" PRId64 " p0=%" PRId64 ": %d nodes, %d computing (%.1f per layer)\n",
                nt, p0, ggml_graph_n_nodes(r.gf), real, (double) real / hp.n_layer);
            std::string line;
            for (const auto & entry : hist)
                line += entry.first + "=" + std::to_string(entry.second) + " ";
            fprintf(stderr, "[dsv4]   ops: %s\n", line.c_str());
        }
        if (hp.v41 && getenv("TS_DSV41_TRACE_DIR"))
        {
            std::filesystem::create_directories(getenv("TS_DSV41_TRACE_DIR"));
            ggml_backend_sched_set_eval_callback(r.sched,
                [](ggml_tensor * t, bool ask, void * data) -> bool {
                    const char * trace_dir = getenv("TS_DSV41_TRACE_DIR");
                    if (!trace_dir || !*trace_dir) return !ask;
                    if (strncmp(t->name, "v41.", 4) != 0) return !ask;
                    if (ask) return true;
                    auto & result = *(graph_build_result *) data;
                    char name[256];
                    snprintf(name, sizeof(name), "/p%06lld_%s.%s", (long long) result.p0, t->name,
                        t->type == GGML_TYPE_I32 ? "i32" : "f32");
                    const std::string path = std::string(trace_dir) + name;
                    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32))
                        return true;
                    std::vector<uint8_t> bytes(ggml_nbytes(t));
                    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
                    FILE * f = fopen(path.c_str(), "wb");
                    if (f) { fwrite(bytes.data(), 1, bytes.size(), f); fclose(f); }
                    return true;
                }, &r);
        }
        for (int d = 0; d < m.n_gpu; d++)
            r.buffer_bytes[d] = ggml_backend_sched_get_buffer_size(r.sched, m.dev_backends[d]);
        if (perf_build >= 2)
        {
            // What this shape actually costs each device, against what the
            // loader held back for it (TS_DSV4_VRAM_RESERVE_MB).
            std::string line;
            for (int d = 0; d < m.n_gpu; d++)
            {
                size_t free_b = 0, total_b = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(m.backends[d]), &free_b, &total_b);
                char buf[96];
                snprintf(buf, sizeof(buf), "%d:%.0f/%.0f ", d, r.buffer_bytes[d] / 1048576.0, free_b / 1048576.0);
                line += buf;
            }
            fprintf(stderr, "[dsv4] graph nt=%" PRId64 " p0=%" PRId64 " compute buffer MiB / free MiB: %s\n",
                nt, p0, line.c_str());
        }
        res_p = &r;
        m.graph_cache.emplace_front(std::move(pending));

        while ((int) m.graph_cache.size() > m.graph_cache_cap)
            m.graph_cache.pop_back();
    }

    res_p->p0 = p0;
    res_p->plan_csa = std::move(plan_csa);
    res_p->plan_hca = std::move(plan_hca);
    res_p->plan_lid = std::move(plan_lid);
    if (out_reuse) *out_reuse = reuse;
    return res_p;
}

// Stage the Engram rows for `nt` tokens whose hashes are already laid out
// [table][token][hash column] -- the layout hash_tokens() produces, and the one
// the token-batched path assembles from its per-slot calls. Shared by the
// single-sequence and token-batched decode paths.
static void dsv41_stage_engram(dsv4_model & m, graph_build_result & res, const int32_t * hashes,
                               int64_t nt, const uint8_t * image_mask, int image_tokens)
{
    const dsv4_hparams & hp = m.hp;
    std::vector<int32_t> streams(hp.hc_mult);
    std::iota(streams.begin(), streams.end(), 0);
    set_i32(res.inp.engram_rows, streams);
    const size_t n_hash = m.engram.hash_columns();
    if (nt <= 0 || n_hash == 0 || m.engram.head_dim == 0 ||
        (size_t) nt > SIZE_MAX / n_hash || (size_t) nt * n_hash > SIZE_MAX / m.engram.head_dim / sizeof(float))
        throw std::runtime_error("V4.1 Engram staging dimensions overflow");
    const size_t n_rows = (size_t) nt * n_hash;
    if (m.engram_on_device)
    {
        // Hand the validated row ids to the graph; the gather and the
        // dequantize happen on the device that owns the table.
        for (size_t e = 0; e < m.engram.layers.size(); ++e)
        {
            const auto * table = m.layers[m.engram.layers[e].id].engram_embd;
            const int32_t * ids = hashes + e * n_rows;
            for (size_t i = 0; i < n_rows; ++i)
                if (ids[i] < 0 || (uint64_t) ids[i] >= (uint64_t) table->ne[1])
                    throw std::runtime_error("DeepSeek V4.1 Engram lookup is out of bounds");
            ggml_backend_tensor_set(res.inp.engram_ids[e], ids, 0, n_rows * sizeof(int32_t));
        }
        return;
    }

    const size_t row_values = n_rows * m.engram.head_dim;
    const size_t table_bytes = row_values * sizeof(float);
    // Stage both published Engram tables together, without amplifying
    // large custom batches/configurations beyond 64 MiB. A single table
    // larger than the budget retains the prior one-table memory bound.
    constexpr size_t staging_budget = (size_t) 64 * 1024 * 1024;
    const size_t group_size = std::max<size_t>(1, std::min(m.engram.layers.size(), staging_budget / table_bytes));
    std::vector<std::vector<float>> rows;
    rows.reserve(group_size);
    for (size_t e = 0; e < group_size; ++e) rows.emplace_back(row_values);
    std::vector<std::function<void(size_t)>> reads;
    reads.reserve(m.engram.layers.size());
    for (size_t e = 0; e < m.engram.layers.size(); ++e)
    {
        auto * table = m.layers[m.engram.layers[e].id].engram_embd;
        if (!table->data || !ggml_backend_buffer_is_host(table->buffer))
            throw std::runtime_error("V4.1 Engram lookup requires a host-resident table");
        const auto * traits = ggml_get_type_traits(table->type);
        reads.emplace_back(m.engram.prepare_lookup(e, table->data, table->ne[1], table->nb[1],
            hashes + e * n_rows, nt, rows[e % group_size].data(),
            [type = table->type, to_float = traits->to_float](const void * src, float * dst, size_t n) {
                if (type == GGML_TYPE_F32) memcpy(dst, src, n * sizeof(float));
                else if (to_float) to_float(src, dst, n);
                else throw std::runtime_error("Unsupported V4.1 Engram quantization");
            }));
    }
    // Every table/hash is validated before the first read. Interleave
    // tables so a slow tail in layer 1 cannot hold up all layer 14 reads.
    for (size_t first = 0; first < reads.size(); first += group_size)
    {
        const size_t count = std::min(group_size, reads.size() - first);
        m.engram_io->run(count * n_rows, [&](size_t task) {
            const size_t table = first + task % count, row = task / count;
            // Visual positions break hashing history and never touch the
            // table, including when a zero row occupies the shared group.
            if (image_tokens && image_mask[row / n_hash])
                std::fill_n(rows[table % group_size].data() + row * m.engram.head_dim, m.engram.head_dim, 0.0f);
            else reads[table](row);
        });
        // run() drains all workers before uploads or staging-buffer reuse.
        for (size_t e = first; e < first + count; ++e)
            ggml_backend_tensor_set(res.inp.engram[e], rows[e % group_size].data(), 0, table_bytes);
    }
}

static bool dsv4_forward_ubatch(dsv4_model & m, const int32_t * tokens, int64_t nt, int64_t p0, bool want_logits, float * logits_out,
                                bool all_logits = false, const uint8_t * image_mask = nullptr,
                                const float * image_embeddings = nullptr, int image_tokens = 0)
{
    const dsv4_hparams & hp = m.hp;
    if (m.active_slot->v41_failed) return false;
#if defined(TSG_GGML_TEST_HOOKS)
    if (m.moe_tp) m.moe_tp->test_set_position(p0);
#endif

    static const int perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e ? atoi(e) : 0; }();
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto t0 = now();

    // Non-final prefill chunks are submitted asynchronously so the layer-split
    // devices pipeline: chunk i+1's first-device layers overlap chunk i's
    // second-device layers. Adjacent chunks are forced onto alternating cache
    // entries (chunk-parity in the signature) so each entry's input tensors
    // are private to in-flight work; reuse waits on the entry's use_events.
    const bool pipeline = !hp.v41 && !want_logits && m.n_gpu > 1;

    bool reuse = false;
    graph_build_result * res_p = dsv4_acquire_graph(m, nt, p0, pipeline, &reuse, all_logits, image_tokens);
    if (!res_p) return false;
    graph_build_result & res = *res_p;

    auto t_build = now();
    auto t_alloc = now();

    // Before refilling a reused entry's inputs, wait until its previous
    // pipelined use has finished reading them (near-free at steady state:
    // the entry was last used two chunks ago).
    if (reuse)
    {
        for (int d = 0; d < m.n_gpu; d++)
            if (res.use_events[d])
                ggml_backend_event_synchronize(res.use_events[d]);
    }

    // ---- fill inputs (per participating device) ----
    {
        std::vector<int32_t> v32(nt);
        std::vector<int64_t> ridx(nt);
        std::vector<int32_t> out_ids;
        if (all_logits)
        {
            out_ids.resize(nt);
            for (int64_t i = 0; i < nt; i++) out_ids[i] = (int32_t) i;
        }
        else
        {
            out_ids.assign(1, (int32_t) (nt - 1));
        }
        std::vector<int32_t> meta;

        const ggml_fp16_t NEG_INF16 = ggml_fp32_to_fp16(-INFINITY);
        const ggml_fp16_t ZERO16 = ggml_fp32_to_fp16(0.0f);

        // raw SWA mask [ring, nt] (F16)
        const int64_t p_last = p0 + nt - 1;
        std::vector<ggml_fp16_t> mask((size_t) m.ring_raw * nt, NEG_INF16);
        for (int64_t i = 0; i < nt; i++)
        {
            const int64_t p = p0 + i;
            for (int64_t s = 0; s < m.ring_raw; s++)
            {
                // token currently held by slot s (all ubatch tokens are written before attention)
                int64_t t = p_last - ((p_last - s) % m.ring_raw + m.ring_raw) % m.ring_raw;
                if (t < 0) continue;
                if (t <= p && t > p - hp.n_swa)
                    mask[(size_t) i * m.ring_raw + s] = ZERO16;
            }
        }

        auto plan_mask = [&](const comp_plan & plan, std::vector<ggml_fp16_t> & mk)
        {
            const int64_t n_kv = plan.n_kv;
            mk.assign((size_t) n_kv * nt, NEG_INF16);
            for (int64_t i = 0; i < nt; i++)
                for (int64_t r = 0; r < std::min<int64_t>(plan.n_visible[i], n_kv); r++)
                    mk[(size_t) i * n_kv + r] = ZERO16;
        };
        std::vector<ggml_fp16_t> mk_csa, mk_hca, mk_lid;
        plan_mask(res.plan_csa, mk_csa);
        plan_mask(res.plan_hca, mk_hca);
        plan_mask(res.plan_lid, mk_lid);

        // gather-mode attention mask: [raw ring mask | top-k rows, all visible]
        std::vector<ggml_fp16_t> gmask;
        bool any_gather = false;
        for (int d = 0; d <= m.n_gpu && !any_gather; d++) any_gather = res.inp.gather_mask[d] != nullptr;
        if (any_gather)
        {
            if (dsv41_use_compact_raw_gather(m, nt, p0))
            {
                const int64_t raw_rows = dsv41_compact_raw_rows(m);
                gmask.assign((size_t) (raw_rows + hp.indexer_top_k), ZERO16);
                std::fill(gmask.begin() + hp.n_swa, gmask.begin() + raw_rows, NEG_INF16);
            }
            else
            {
                gmask.assign(mask.begin(), mask.end());
                gmask.resize((size_t) (m.ring_raw + hp.indexer_top_k) * nt, ZERO16);
            }
        }
        std::vector<int32_t> raw_read_ids;
        if (dsv41_use_compact_raw_gather(m, nt, p0))
        {
            dsv41_raw_window_ids(p0, m.ring_raw, hp.n_swa, raw_read_ids);
            raw_read_ids.resize((size_t) dsv41_compact_raw_rows(m), raw_read_ids.front());
        }

        auto set_f16 = [](ggml_tensor * t, const std::vector<ggml_fp16_t> & v)
        {
            if (!t || !t->buffer || v.empty()) return;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
        };

        auto fill_plan = [&](plan_inputs & pi, const comp_plan & plan, const std::vector<ggml_fp16_t> & mk)
        {
            set_i32(pi.state_pos, plan.state_pos);
            if (m.fused && !hp.v41)
            {
                meta.clear();
                meta.insert(meta.end(), plan.state_read_idxs.begin(), plan.state_read_idxs.end());
                for (size_t i = 0; i < plan.state_write_idxs.size(); i++)
                {
                    meta.push_back((int32_t) plan.state_write_idxs[i]);
                    meta.push_back(plan.state_write_pos[i]);
                }
                for (size_t i = 0; i < plan.state_persist_src_idxs.size(); i++)
                {
                    meta.push_back(plan.state_persist_src_idxs[i]);
                    meta.push_back((int32_t) plan.state_persist_dst_idxs[i]);
                }
                set_i32(pi.comp_meta, meta);
            }
            else
            {
                set_i32(pi.persist_src, plan.state_persist_src_idxs);
                set_i64(pi.persist_dst, plan.state_persist_dst_idxs);
                if (pi.write_idxs)
                {
                    set_i32(pi.read_idxs, plan.state_read_idxs);
                    set_i64(pi.write_idxs, plan.state_write_idxs);
                    set_i32(pi.write_pos, plan.state_write_pos);
                }
            }
            set_f16(pi.kq_mask, mk);
        };

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!res.inp.pos[d]) continue;
            for (int64_t i = 0; i < nt; i++) v32[i] = tokens[i];
            set_i32(res.inp.tokens[d], v32);
            if (image_tokens)
            {
                for (int64_t i = 0; i < nt; ++i) v32[i] = image_mask[i];
                set_i32(res.inp.image_types[d], v32);
            }
            for (int64_t i = 0; i < nt; i++) v32[i] = (int32_t) (p0 + i);
            set_i32(res.inp.pos[d], v32);
            for (int64_t i = 0; i < nt; i++) ridx[i] = (p0 + i) % m.ring_raw;
            set_i64(res.inp.raw_idxs[d], ridx);
            set_i32(res.inp.raw_read_idxs[d], raw_read_ids);
            set_f16(res.inp.raw_mask[d], mask);
            set_f16(res.inp.gather_mask[d], gmask);
            fill_plan(res.inp.csa[d], res.plan_csa, mk_csa);
            fill_plan(res.inp.hca[d], res.plan_hca, mk_hca);
            fill_plan(res.inp.lid[d], res.plan_lid, mk_lid);
        }
        set_i32(res.inp.out_ids, out_ids);
        if (image_tokens)
        {
            std::vector<int64_t> indices;
            indices.reserve(image_tokens);
            for (int64_t i = 0; i < nt; ++i) if (image_mask[i]) indices.push_back(i);
            set_i64(res.inp.image_indices, indices);
            ggml_backend_tensor_set(res.inp.image_embeddings, image_embeddings, 0,
                (size_t) image_tokens * hp.n_embd * sizeof(float));
        }
    }

    dsv41_slot_write_guard writes{m.active_slot};
    if (hp.v41)
    {
        std::vector<int32_t> hash_tokens;
        if (image_tokens)
        {
            hash_tokens.assign(tokens, tokens + nt);
            for (int64_t i = 0; i < nt; ++i) if (image_mask[i]) hash_tokens[i] = -1;
        }
        writes.started = true;
        auto hashes = m.engram.hash_tokens(image_tokens ? hash_tokens.data() : tokens,
            (size_t) nt, (size_t) p0, m.active_slot->engram_history);
#if defined(TSG_GGML_TEST_HOOKS)
        dsv41_test_fail("engram", p0);
#endif
        dsv41_stage_engram(m, res, hashes.data(), nt, image_mask, image_tokens);
    }
    auto t_inputs = now();

    // Both architectures can have partially written cache tensors once a
    // graph is submitted, including a non-final asynchronous V4 microbatch.
    writes.started = true;

#ifdef TSG_GGML_USE_CUDA
    if (perf >= 3) tsg_dsv4_fused_counters_reset();
#endif
    if (pipeline)
    {
        if (ggml_backend_sched_graph_compute_async(res.sched, res.gf) != GGML_STATUS_SUCCESS)
        {
            fprintf(stderr, "[dsv4] graph compute failed\n");
            return false;
        }
        for (int d = 0; d < m.n_gpu; d++)
        {
            if (!res.use_events[d])
                res.use_events[d] = ggml_backend_event_new(ggml_backend_get_device(m.backends[d]));
            if (res.use_events[d])
                ggml_backend_event_record(res.use_events[d], m.backends[d]);
        }
    }
    else if (ggml_backend_sched_graph_compute(res.sched, res.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[dsv4] graph compute failed\n");
        return false;
    }

    if (m.moe_tp)
    {
        const auto error = m.moe_tp->error();
        if (!error.empty())
        {
            fprintf(stderr, "[dsv41] tensor-parallel compute failed: %s\n", error.c_str());
            return false;
        }
    }
#if defined(TSG_GGML_TEST_HOOKS)
    if (hp.v41) dsv41_test_fail("compute", p0);
#endif
    auto t_compute = now();

    if (want_logits && logits_out)
        ggml_backend_tensor_get(res.logits, logits_out, 0,
                                (size_t) (all_logits ? nt : 1) * hp.n_vocab * sizeof(float));

    if (perf >= 2)
    {
        fprintf(stderr, "[dsv4] ubatch nt=%" PRId64 " p0=%" PRId64 ": build %.2fms alloc %.2fms inputs %.2fms compute %.2fms (nodes %d, splits %d%s)\n",
                nt, p0, ms(t0, t_build), ms(t_build, t_alloc), ms(t_alloc, t_inputs), ms(t_inputs, t_compute),
                ggml_graph_n_nodes(res.gf), ggml_backend_sched_get_n_splits(res.sched), reuse ? ", reused" : "");
    }
#ifdef TSG_GGML_USE_CUDA
    if (perf >= 3)
    {
        const auto counters = tsg_dsv4_fused_counters_read();
        fprintf(stderr, "[dsv4]   submission: %llu device subgraph(s), %llu node(s), %llu ordinary run(s), "
                "%llu fused launch(es), %.2fms on the submitting thread\n",
                counters.calls, counters.nodes, counters.views, counters.fused, counters.submit_ms);
    }
#endif

    if (perf >= 3)
    {
        // report backend transitions in execution order (split boundaries)
        ggml_backend_t prev = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(res.gf); i++)
        {
            ggml_tensor * node = ggml_graph_node(res.gf, i);
            ggml_backend_t be = ggml_backend_sched_get_tensor_backend(res.sched, node);
            if (be != prev)
            {
                fprintf(stderr, "[dsv4]   node %5d %-16s op=%-14s -> %s\n",
                        i, node->name, ggml_op_name(node->op), be ? ggml_backend_name(be) : "?");
                prev = be;
            }
        }
    }

    writes.complete = true;
    return true;
}

// Build (or fetch) the batched multi-slot decode graph for the given slot/
// position vector. Cached alongside the normal graphs; the sig covers N, the
// slot ids, every slot's plan shapes and its indexer-skip bit, so steady
// concurrent decode reuses one entry (and its captured CUDA graphs) until a
// slot crosses a bucket or block boundary.
static graph_build_result * dsv4_acquire_batched_graph(
    dsv4_model & m, int n, const int32_t * slot_ids, const int32_t * positions, bool * out_reuse)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (dsv4_test_boundary_fault()) return nullptr;
#endif
    const dsv4_hparams & hp = m.hp;

    // V4.1 keeps the unfused compressor and its own ratios; the plan sizes and
    // the per-slot sparse-selection decisions are part of the graph shape, so
    // they go in the signature alongside the slot ids.
    const int64_t cr = hp.v41 ? 2 : CSA_RATIO, hr = hp.v41 ? 1 : HCA_RATIO;
    std::vector<bd_slot_state> bd((size_t) n);
    uint64_t sig = 0xBD5EEDB47C8ED0DEull ^ (uint64_t) n;
    for (int i = 0; i < n; i++)
    {
        bd_slot_state & B = bd[i];
        B.slot_id = slot_ids[i];
        B.p0 = positions[i];
        const int64_t pos_end = B.p0 + 1;
        const int64_t hint = std::min<int64_t>(pos_end, 8192);
        B.plan_csa = build_comp_plan(B.p0, 1, cr, !hp.v41, (hp.v41 ? cr : 2 * cr) + m.state_extra, m.n_csa_rows, hint / cr);
        B.plan_hca = build_comp_plan(B.p0, 1, hr, false, hr + m.state_extra, m.n_hca_rows, hint / hr);
        B.plan_lid = B.plan_csa;
        if (!hp.v41)
            B.plan_lid = build_comp_plan(B.p0, 1, CSA_RATIO, true, 2 * CSA_RATIO + m.state_extra, m.n_csa_rows, hint / CSA_RATIO);
        B.skip_topk = pos_end <= (int64_t) hp.indexer_top_k * (hp.v41 ? cr : CSA_RATIO);
        B.use_gather = hp.v41 && dsv41_use_gather(m, 1, B.p0);
        B.compact_raw = hp.v41 && dsv41_use_compact_raw_gather(m, 1, B.p0);
        sig = sig * 1099511628211ull ^ (uint64_t) (B.slot_id + 1);
        sig = sig * 1099511628211ull ^ dsv4_plan_sig(B.plan_csa);
        sig = sig * 1099511628211ull ^ dsv4_plan_sig(B.plan_hca);
        sig = sig * 1099511628211ull ^ dsv4_plan_sig(B.plan_lid);
        if (B.skip_topk) sig ^= 0x9e3779b97f4a7c15ull;
        if (B.use_gather) sig = sig * 1099511628211ull ^ 0xbefca6a34d210597ull;
        if (B.compact_raw) sig = sig * 1099511628211ull ^ 0x2545f4914f6cdd1dull;
    }

    graph_build_result * res_p = nullptr;
    bool reuse = false;
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        if ((*it)->sig == sig && (*it)->bd.size() == (size_t) n)
        {
            if (it != m.graph_cache.begin())
                m.graph_cache.splice(m.graph_cache.begin(), m.graph_cache, it);
            res_p = m.graph_cache.front().get();
            reuse = true;
            break;
        }
    }

    if (!reuse)
    {
        dsv4_trim_graph_cache(m);
        auto pending = std::make_unique<graph_build_result>();
        graph_build_result & r = *pending;
        // The per-slot attention forks are the only O(n) node cost, roughly
        // ~500 nodes/slot including the attn_cat concat chain; wide spans need
        // a bigger node table and metadata arena.
        const size_t graph_nodes = n <= 8 ? 32768 : 65536;
        const size_t meta_size = (size_t) (n <= 8 ? 96 : 176) * 1024 * 1024;
        ggml_init_params gp = { meta_size, nullptr, true };
        r.ctx = ggml_init(gp);
        r.gf = ggml_new_graph_custom(r.ctx, graph_nodes, false);
        r.nt = n;
        r.sig = sig;
        r.slot_id = -1;   // multi-slot; purge-on-free checks r.bd
        r.bd = std::move(bd);
        r.sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, (int) graph_nodes, false, true);
        if (!r.sched)
        {
            fprintf(stderr, "[dsv4] batched sched creation failed\n");
            return nullptr;
        }

        // Publish only a complete graph. Allocation/build exceptions also
        // destroy the local pending entry before the caller can retry it.
        try
        {
            graph_builder gb(m, r, n, 0);
            if (hp.v41) gb.build_batched_v41(); else gb.build_batched();
        }
        catch (const std::exception & error)
        {
            fprintf(stderr, "[dsv4] batched graph build failed (n=%d): %s\n", n, error.what());
            return nullptr;
        }

        if (!tsg_scheduler_alloc_graph(r.sched, r.gf))
        {
            fprintf(stderr, "[dsv4] batched sched_alloc_graph failed (n=%d)\n", n);
            return nullptr;
        }
        for (int d = 0; d < m.n_gpu; d++)
            r.buffer_bytes[d] = ggml_backend_sched_get_buffer_size(r.sched, m.dev_backends[d]);
        res_p = &r;

        m.graph_cache.emplace_front(std::move(pending));

        while ((int) m.graph_cache.size() > m.graph_cache_cap)
            m.graph_cache.pop_back();
    }
    else
    {
        // refresh per-slot plans/positions; input tensor pointers stay
        for (int i = 0; i < n; i++)
        {
            bd_slot_state & dst = res_p->bd[i];
            dst.p0 = bd[i].p0;
            dst.plan_csa = std::move(bd[i].plan_csa);
            dst.plan_hca = std::move(bd[i].plan_hca);
            dst.plan_lid = std::move(bd[i].plan_lid);
        }
    }

    if (out_reuse) *out_reuse = reuse;
    return res_p;
}

static bool dsv4_forward_batched_decode(
    dsv4_model & m, int n, const int32_t * slot_ids, const int32_t * tokens,
    const int32_t * positions, float * logits_out)
{
    const dsv4_hparams & hp = m.hp;

    static const int perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e ? atoi(e) : 0; }();
    auto t0 = std::chrono::steady_clock::now();

    bool reuse = false;
    graph_build_result * res_p = dsv4_acquire_batched_graph(m, n, slot_ids, positions, &reuse);
    if (!res_p) return false;
    graph_build_result & res = *res_p;

    const ggml_fp16_t NEG_INF16 = ggml_fp32_to_fp16(-INFINITY);
    const ggml_fp16_t ZERO16 = ggml_fp32_to_fp16(0.0f);

    // ---- V4.1: its own input layout (unfused compressor, per-slot sparse
    // selection, and one Engram staging column per slot) ----
    dsv41_batched_write_guard writes;
    if (hp.v41)
    {
        auto set_f16 = [](ggml_tensor * t, const std::vector<ggml_fp16_t> & v)
        {
            if (!t || !t->buffer || v.empty()) return;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
        };
        std::vector<int32_t> v32(n);
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!res.inp.pos[d]) continue;
            for (int i = 0; i < n; i++) v32[i] = tokens[i];
            set_i32(res.inp.tokens[d], v32);
            for (int i = 0; i < n; i++) v32[i] = positions[i];
            set_i32(res.inp.pos[d], v32);
        }
        {
            std::vector<int32_t> out_ids(n);
            for (int i = 0; i < n; i++) out_ids[i] = i;
            set_i32(res.inp.out_ids, out_ids);
        }

        std::vector<int64_t> ridx(1);
        std::vector<ggml_fp16_t> rmask, gmask, pmask;
        std::vector<int32_t> raw_read_ids;
        auto fill_slot_plan = [&](plan_inputs & pi, const comp_plan & plan)
        {
            set_i32(pi.persist_src, plan.state_persist_src_idxs);
            set_i64(pi.persist_dst, plan.state_persist_dst_idxs);
            if (pi.write_idxs)
            {
                set_i32(pi.read_idxs, plan.state_read_idxs);
                set_i64(pi.write_idxs, plan.state_write_idxs);
                set_i32(pi.write_pos, plan.state_write_pos);
            }
            pmask.assign((size_t) plan.n_kv, NEG_INF16);
            for (int64_t r = 0; r < std::min<int64_t>(plan.n_visible[0], plan.n_kv); r++)
                pmask[(size_t) r] = ZERO16;
            set_f16(pi.kq_mask, pmask);
        };

        for (int i = 0; i < n; i++)
        {
            bd_slot_state & B = res.bd[i];
            const int64_t p = B.p0;

            rmask.assign((size_t) m.ring_raw, NEG_INF16);
            for (int64_t s = 0; s < m.ring_raw; s++)
            {
                const int64_t t = p - ((p - s) % m.ring_raw + m.ring_raw) % m.ring_raw;
                if (t < 0) continue;
                if (t <= p && t > p - hp.n_swa) rmask[(size_t) s] = ZERO16;
            }
            gmask.clear();
            raw_read_ids.clear();
            if (B.use_gather)
            {
                if (B.compact_raw)
                {
                    const int64_t raw_rows = dsv41_compact_raw_rows(m);
                    gmask.assign((size_t) (raw_rows + hp.indexer_top_k), ZERO16);
                    std::fill(gmask.begin() + hp.n_swa, gmask.begin() + raw_rows, NEG_INF16);
                    dsv41_raw_window_ids(p, m.ring_raw, hp.n_swa, raw_read_ids);
                    raw_read_ids.resize((size_t) raw_rows, raw_read_ids.front());
                }
                else
                {
                    gmask.assign(rmask.begin(), rmask.end());
                    gmask.resize((size_t) (m.ring_raw + hp.indexer_top_k), ZERO16);
                }
            }
            ridx[0] = p % m.ring_raw;
            for (int d = 0; d <= m.n_gpu; d++)
            {
                if (!res.inp.pos[d]) continue;
                set_i64(B.raw_idxs[d], ridx);
                set_f16(B.raw_mask[d], rmask);
                set_f16(B.gather_mask[d], gmask);
                set_i32(B.raw_read_idxs[d], raw_read_ids);
                fill_slot_plan(B.v41_csa[d], B.plan_csa);
                fill_slot_plan(B.v41_hca[d], B.plan_hca);
                fill_slot_plan(B.v41_lid[d], B.plan_lid);
            }
        }

        // Engram: hash each slot's token against ITS history, then lay the
        // results out [table][slot column][hash column] -- the same layout one
        // sequence's nt tokens produce, so the staging path is unchanged.
        const size_t n_hash = m.engram.hash_columns();
        const size_t tables = m.engram.layers.size();
        std::vector<int32_t> hashes(tables * (size_t) n * n_hash);
        for (int i = 0; i < n; i++)
        {
            dsv4_slot * slot = m.slots.at(slot_ids[i]).get();
            // Each slot's history is mutated here; a failure after the first
            // leaves those slots inconsistent, so they are all latched failed.
            writes.slots.push_back(slot);
            const auto one = m.engram.hash_tokens(&tokens[i], 1, (size_t) positions[i], slot->engram_history);
            for (size_t e = 0; e < tables; ++e)
                std::copy(one.begin() + e * n_hash, one.begin() + (e + 1) * n_hash,
                          hashes.begin() + (e * (size_t) n + i) * n_hash);
        }
        dsv41_stage_engram(m, res, hashes.data(), n, nullptr, 0);
    }

    // ---- shared batched inputs ----
    if (!hp.v41)
    {
        std::vector<int32_t> v32(n);
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!res.inp.pos[d]) continue;
            for (int i = 0; i < n; i++) v32[i] = tokens[i];
            set_i32(res.inp.tokens[d], v32);
            for (int i = 0; i < n; i++) v32[i] = positions[i];
            set_i32(res.inp.pos[d], v32);
            for (int i = 0; i < n; i++) v32[i] = (int32_t) (positions[i] % CSA_RATIO);
            set_i32(res.inp.csa[d].state_pos, v32);
            for (int i = 0; i < n; i++) v32[i] = (int32_t) (positions[i] % HCA_RATIO);
            set_i32(res.inp.hca[d].state_pos, v32);
        }
        for (int i = 0; i < n; i++) v32[i] = i;
        set_i32(res.inp.out_ids, v32);
    }

    // ---- per-slot inputs ----
    if (!hp.v41)
    {
        std::vector<ggml_fp16_t> mask;
        std::vector<int32_t> meta;
        std::vector<int64_t> ridx(1);

        auto set_f16 = [](ggml_tensor * t, const std::vector<ggml_fp16_t> & v)
        {
            if (!t || !t->buffer || v.empty()) return;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
        };
        auto fill_meta = [&](const comp_plan & plan)
        {
            meta.clear();
            meta.insert(meta.end(), plan.state_read_idxs.begin(), plan.state_read_idxs.end());
            for (size_t k = 0; k < plan.state_write_idxs.size(); k++)
            {
                meta.push_back((int32_t) plan.state_write_idxs[k]);
                meta.push_back(plan.state_write_pos[k]);
            }
            for (size_t k = 0; k < plan.state_persist_src_idxs.size(); k++)
            {
                meta.push_back(plan.state_persist_src_idxs[k]);
                meta.push_back((int32_t) plan.state_persist_dst_idxs[k]);
            }
        };
        auto plan_mask1 = [&](const comp_plan & plan)
        {
            mask.assign((size_t) plan.n_kv, NEG_INF16);
            for (int64_t r = 0; r < std::min<int64_t>(plan.n_visible[0], plan.n_kv); r++)
                mask[(size_t) r] = ZERO16;
        };

        for (int i = 0; i < n; i++)
        {
            bd_slot_state & B = res.bd[i];
            const int64_t p = B.p0;

            // raw SWA ring mask for this slot's single token
            std::vector<ggml_fp16_t> rmask((size_t) m.ring_raw, NEG_INF16);
            for (int64_t s = 0; s < m.ring_raw; s++)
            {
                int64_t t = p - ((p - s) % m.ring_raw + m.ring_raw) % m.ring_raw;
                if (t < 0) continue;
                if (t <= p && t > p - hp.n_swa)
                    rmask[(size_t) s] = ZERO16;
            }
            std::vector<ggml_fp16_t> gmask;
            if (!B.skip_topk)
            {
                gmask.assign(rmask.begin(), rmask.end());
                gmask.resize((size_t) (m.ring_raw + hp.indexer_top_k), ZERO16);
            }

            ridx[0] = p % m.ring_raw;
            for (int d = 0; d <= m.n_gpu; d++)
            {
                set_i64(B.raw_idxs[d], ridx);
                set_f16(B.raw_mask[d], rmask);
                set_f16(B.gather_mask[d], gmask);
                fill_meta(B.plan_csa);
                if (!meta.empty()) set_i32(B.csa_meta[d], meta);
                fill_meta(B.plan_lid);
                if (!meta.empty()) set_i32(B.lid_meta[d], meta);
                fill_meta(B.plan_hca);
                if (!meta.empty()) set_i32(B.hca_meta[d], meta);
                if (B.csa_mask[d]) { plan_mask1(B.plan_csa); set_f16(B.csa_mask[d], mask); }
                if (B.lid_mask[d]) { plan_mask1(B.plan_lid); set_f16(B.lid_mask[d], mask); }
                if (B.hca_mask[d]) { plan_mask1(B.plan_hca); set_f16(B.hca_mask[d], mask); }
            }
        }
    }

    if (!hp.v41)
    {
        // Input staging has not mutated V4 caches. Arm all slots only before
        // compute; a graph-allocation decline remains safe for serial fallback.
        writes.slots.reserve(n);
        for (int i = 0; i < n; ++i) writes.slots.push_back(m.slots.at(slot_ids[i]).get());
    }
    if (ggml_backend_sched_graph_compute(res.sched, res.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[dsv4] batched graph compute failed\n");
        return false;
    }

    ggml_backend_tensor_get(res.logits, logits_out, 0, (size_t) n * hp.n_vocab * sizeof(float));
    writes.complete = true;

    if (perf >= 2)
    {
        auto t1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[dsv4] batched decode n=%d: %.2fms (nodes %d, splits %d%s)\n",
                n, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                ggml_graph_n_nodes(res.gf), ggml_backend_sched_get_n_splits(res.sched),
                reuse ? ", reused" : "");
    }
    return true;
}

// One DSpark draft at `position` (the position of `anchor_token`, which is not
// yet in the trunk cache). The drafter's key ring was committed by the trunk's
// own forward passes, so this needs no state from the host beyond the anchor.
static int dsv4_dspark_draft(dsv4_model & m, int32_t anchor_token, int64_t position,
                             int32_t * toks_out, float * conf_out)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (dsv4_test_boundary_fault()) return 0;
#endif
    const dsv4_dspark & ds = m.ds;
    if (!ds.loaded || position <= 0) return 0;

    const int64_t B = ds.block_size;

    // The drafter's graph shape never changes, so it gets one cache entry of
    // its own (keyed like the trunk's, plus a draft bit).
    uint64_t sig = 14695981039346656037ull ^ 0x44535041524bull;
    sig = sig * 1099511628211ull ^ (uint64_t) (m.active_slot->id + 1);

    graph_build_result * res_p = nullptr;
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        if ((*it)->sig == sig)
        {
            if (it != m.graph_cache.begin())
                m.graph_cache.splice(m.graph_cache.begin(), m.graph_cache, it);
            res_p = m.graph_cache.front().get();
            break;
        }
    }

    if (!res_p)
    {
        auto pending = std::make_unique<graph_build_result>();
        graph_build_result & r = *pending;
        ggml_init_params gp = { (size_t) 32 * 1024 * 1024, nullptr, true };
        r.ctx = ggml_init(gp);
        r.gf = ggml_new_graph_custom(r.ctx, 8192, false);
        r.nt = B;
        r.sig = sig;
        r.draft = true;
        r.slot_id = m.active_slot->id;
        r.sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, 8192, false, true);
        if (!r.sched)
        {
            return 0;
        }
        graph_builder gb(m, r, B, position);
        gb.build_dspark_draft();
        if (!tsg_scheduler_alloc_graph(r.sched, r.gf))
        {
            fprintf(stderr, "[dsv4] DSpark draft graph alloc failed\n");
            return 0;
        }
        res_p = &r;
        m.graph_cache.emplace_front(std::move(pending));
    }
    graph_build_result & res = *res_p;

    // inputs: [anchor, noise...] at positions [position .. position+B-1], and a
    // mask exposing the committed window (ending at position-1) plus the whole
    // block (non-causal inside the block).
    {
        std::vector<int32_t> ids((size_t) B, ds.noise_token);
        ids[0] = anchor_token;
        std::vector<int32_t> pos((size_t) B);
        for (int64_t i = 0; i < B; i++) pos[i] = (int32_t) (position + i);
        set_i32(res.inp.ds_tokens, ids);
        set_i32(res.inp.ds_pos, pos);

        const ggml_fp16_t NEG_INF16 = ggml_fp32_to_fp16(-INFINITY);
        const ggml_fp16_t ZERO16 = ggml_fp32_to_fp16(0.0f);
        const int64_t n_kv = m.ring_raw + B;
        std::vector<ggml_fp16_t> mask((size_t) n_kv * B, NEG_INF16);
        const int64_t p_last = position - 1;                   // last committed position
        for (int64_t i = 0; i < B; i++)
        {
            for (int64_t sIdx = 0; sIdx < m.ring_raw; sIdx++)
            {
                int64_t t = p_last - ((p_last - sIdx) % m.ring_raw + m.ring_raw) % m.ring_raw;
                if (t < 0) continue;
                if (t <= p_last && t > p_last - m.hp.n_swa)
                    mask[(size_t) i * n_kv + sIdx] = ZERO16;
            }
            for (int64_t b = 0; b < B; b++)
                mask[(size_t) i * n_kv + m.ring_raw + b] = ZERO16;
        }
        ggml_backend_tensor_set(res.inp.ds_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(res.sched, res.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[dsv4] DSpark draft compute failed\n");
        return 0;
    }

    ggml_backend_tensor_get(res.ds_toks, toks_out, 0, (size_t) B * sizeof(int32_t));
    ggml_backend_tensor_get(res.ds_conf, conf_out, 0, (size_t) B * sizeof(float));
    return (int) B;
}

} // namespace tsg_dsv4

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

// n_cpu_moe: routed-expert CPU offload policy — -1 auto (offload the fewest
// leading layers that make the model fit the visible VRAM), 0 none, N the
// first N layers, INT_MAX every layer (--cpu-moe).
// backend_name: ggml backend registry name to take GPU devices from ("CUDA",
// "Vulkan", ...); null/empty takes any GPU. GgmlOps links every backend it was
// built with, so without it the caller's --backend choice is not honored.
TSG_EXPORT void * TSGgml_Dsv4LoadModelDspark(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
                                             const char * dspark_path, int n_cpu_moe, const char * backend_name)
{
    try
    {
        return tsg_dsv4::dsv4_load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, dspark_path, n_cpu_moe, backend_name);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] DSpark load failed: %s\n", e.what());
        return nullptr;
    }
}

TSG_EXPORT void * TSGgml_Dsv4LoadModel(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
                                       int n_cpu_moe, const char * backend_name)
{
    try
    {
        return tsg_dsv4::dsv4_load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, nullptr, n_cpu_moe, backend_name);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] load failed: %s\n", e.what());
        return nullptr;
    }
}

TSG_EXPORT int TSGgml_Dsv4VocabSize(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->hp.n_vocab;
}

TSG_EXPORT int TSGgml_Dsv4CtxSize(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->n_ctx;
}

TSG_EXPORT int TSGgml_Dsv4NPast(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->active_slot->n_past;
}

#if defined(TSG_GGML_TEST_HOOKS)
// Observe the width used to successfully allocate the CPU worker pool without
// exposing a production tuning ABI or duplicating the loader's resolver.
// Keep test exports distinct from the production ABI manifest's TSG_EXPORT scan.
#define TSG_TEST_EXPORT TSG_EXPORT
TSG_TEST_EXPORT int TSGgml_Dsv4TestCpuThreads(void * handle)
{
    auto * m = static_cast<tsg_dsv4::dsv4_model *>(handle);
    return m && m->cpu_threadpool ? m->test_cpu_pool_threads : 0;
}
// Inspect the actual allocated scheduler graph, not the builder's intent.
// Return 1 only when the selected projection uses the layer's backend object.
TSG_TEST_EXPORT int TSGgml_Dsv4TestSharedPlacement(void * handle, int layer, int projection,
                                                char * description, int capacity)
{
    auto * m = static_cast<tsg_dsv4::dsv4_model *>(handle);
    if (!m || layer < 0 || layer >= m->hp.n_layer || projection < 0 || projection > 2 ||
        !description || capacity <= 0 || m->graph_cache.empty()) return -1;
    const auto & L = m->layers[layer];
    ggml_tensor * weight = projection == 0 ? L.ffn_gate_shexp : projection == 1 ? L.ffn_up_shexp : L.ffn_down_shexp;
    auto & graph = *m->graph_cache.front();
    for (int i = 0; i < ggml_graph_n_nodes(graph.gf); ++i)
    {
        ggml_tensor * node = ggml_graph_node(graph.gf, i);
        // Scheduler copies can replace src[0], so its tensor name is the
        // stable identity after allocation. Copies include the original name.
        if (!node->src[0] ||
            (node->src[0] != weight && !strstr(node->src[0]->name, weight->name))) continue;
        bool projection_op = node->op == GGML_OP_MUL_MAT;
        if (node->op == GGML_OP_CUSTOM)
        {
            ggml_custom_op_params params;
            memcpy(&params, node->op_params, sizeof(params));
            const auto * descriptor = static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
            projection_op = descriptor && descriptor->magic == TSG_DSV4_FUSED_MAGIC
                && descriptor->kind == TSG_MATMUL_F32;
        }
        if (!projection_op) continue;
        auto actual = ggml_backend_sched_get_tensor_backend(graph.sched, node);
        // The scheduler's backend for a device, which is the fused wrapper when
        // fused ops are available and the CUDA backend otherwise.
        auto expected = m->dev_backends[L.device];
        snprintf(description, (size_t) capacity, "%s -> %s", actual ? ggml_backend_name(actual) : "unassigned",
                 ggml_backend_name(expected));
        return actual == expected ? 1 : 0;
    }
    snprintf(description, (size_t) capacity, "projection not found");
    return -2;
}
// Numerical fixture seam: expose the committed drafter window in logical
// position order. Read-only and absent from production builds.
TSG_TEST_EXPORT int TSGgml_Dsv4TestLayerDevice(void * handle, int layer)
{
    auto * m = static_cast<tsg_dsv4::dsv4_model *>(handle);
    if (!m || layer < 0 || layer >= m->hp.n_layer) return -1;
    return m->layers[layer].device;
}

TSG_TEST_EXPORT int TSGgml_Dsv4TestDsparkReadRing(void * handle, int stage, int count, float * output)
{
    auto * m = static_cast<tsg_dsv4::dsv4_model *>(handle);
    if (!m || !m->active_slot || !m->ds.loaded || !output || stage < 0 || stage >= m->ds.n_stages ||
        count <= 0 || count > m->ring_raw || count > m->active_slot->n_past) return 0;
    try
    {
        auto * ring = m->active_slot->ds_k[stage];
        const int64_t width = m->hp.n_embd_head;
        std::vector<ggml_fp16_t> row((size_t) width);
        for (int i = 0; i < count; ++i)
        {
            const int64_t position = m->active_slot->n_past - count + i;
            ggml_backend_tensor_get(ring, row.data(), (size_t) (position % m->ring_raw) * ring->nb[1], row.size() * sizeof(ggml_fp16_t));
            for (int64_t j = 0; j < width; ++j) output[i * width + j] = ggml_fp16_to_fp32(row[(size_t) j]);
        }
        return 1;
    }
    catch (...) { return 0; }
}
#undef TSG_TEST_EXPORT
#endif

// Feeds `n_tokens` tokens at positions [n_past, n_past + n_tokens) of the
// ACTIVE slot and writes the last token's logits into logits_out (size
// n_vocab). Returns 0 on success, negative on failure.
TSG_EXPORT int TSGgml_Dsv4Forward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out)
{
    if (!handle || !tokens || n_tokens <= 0) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    tsg_dsv4::dsv4_slot * slot = m->active_slot;

    if (slot->v41_failed)
    {
        fprintf(stderr, "[dsv4] sequence execution previously failed; reset or destroy the slot before reuse\n");
        return -4;
    }
    if (m->hp.v41)
    {
        // Validate the whole request before a first microbatch can mutate the
        // sequence, including invalid IDs in a later microbatch.
        for (int i = 0; i < n_tokens; ++i)
            if (tokens[i] < 0 || tokens[i] >= m->hp.n_vocab) return -1;
    }

    if ((int64_t) slot->n_past + n_tokens > m->n_ctx)
    {
        fprintf(stderr, "[dsv4] context overflow: n_past=%d + %d > n_ctx=%d\n", slot->n_past, n_tokens, m->n_ctx);
        return -2;
    }

    const bool perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e && atoi(e) > 0; }();
    auto t0 = std::chrono::steady_clock::now();

    try
    {
        slot->spec_begin = slot->spec_end = -1;
        m->pos_end_hint = (int64_t) slot->n_past + n_tokens;
        if (m->moe_tp) m->moe_tp->begin_forward();

        // Pre-acquisition allocates graph metadata and device arenas. It must
        // share the C ABI error boundary with compute and checkpointing.
        if (!m->hp.v41 && n_tokens > m->n_ubatch)
        {
            const int nt2 = std::min(m->n_ubatch, n_tokens - m->n_ubatch);
            const bool last2 = (2 * m->n_ubatch >= n_tokens);
            if (!tsg_dsv4::dsv4_acquire_graph(*m, m->n_ubatch, slot->n_past, m->n_gpu > 1, nullptr) ||
                !tsg_dsv4::dsv4_acquire_graph(*m, nt2, slot->n_past + m->n_ubatch, !last2 && m->n_gpu > 1, nullptr))
            {
                slot->v41_failed = true;
                return -3;
            }
            if (!last2)
            {
                const int last_nt = n_tokens - m->n_ubatch * ((n_tokens - 1) / m->n_ubatch);
                if (!tsg_dsv4::dsv4_acquire_graph(*m, last_nt,
                        (int64_t) slot->n_past + (n_tokens - last_nt), false, nullptr))
                {
                    slot->v41_failed = true;
                    return -3;
                }
            }
        }
        int done = 0;
        while (done < n_tokens)
        {
            const int nt = std::min(m->n_ubatch, n_tokens - done);
            const bool last = (done + nt == n_tokens);
            if (!tsg_dsv4::dsv4_forward_ubatch(*m, tokens + done, nt, slot->n_past, last, last ? logits_out : nullptr))
            {
                slot->v41_failed = true;
                return -3;
            }
            slot->n_past += nt;
            done += nt;
        }

        // Only multi-token calls publish prompt-boundary rewind checkpoints.
        // A copy failure can leave a partial shadow, so it also requires reset.
        if (n_tokens > 1) tsg_dsv4::dsv4_checkpoint_slot(*m, *slot);
    }
    catch (const std::exception & error)
    {
        slot->v41_failed = true;
        fprintf(stderr, "[dsv4] forward failed: %s\n", error.what());
        return -3;
    }
    catch (...)
    {
        slot->v41_failed = true;
        fprintf(stderr, "[dsv4] forward failed with an unknown execution exception\n");
        return -3;
    }

    if (perf)
    {
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "[dsv4] forward %d tokens in %.3fs (%.1f tok/s)\n", n_tokens, s, n_tokens / s);
    }
    return 0;
}

#include "ggml_ops_deepseek41_vision_text.inc"

// Resets the ACTIVE slot's caches to position 0. Weights, rope tables and
// other slots are untouched.
TSG_EXPORT void TSGgml_Dsv4Reset(void * handle)
{
    if (!handle) return;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    // Initializing compressor sentinel rows uses host scratch. If that fails
    // under memory pressure, the void reset ABI cannot claim a usable cache.
    const bool failed = m->active_slot->v41_failed;
    m->active_slot->v41_failed = true;
    try
    {
        // Failed requests must release their captured arenas before retrying
        // under memory pressure. Normal reset keeps successful graph reuse.
        if (failed) tsg_dsv4::dsv4_drop_slot_graphs(*m, m->active_slot->id);
        tsg_dsv4::dsv4_reset_slot(*m->active_slot);
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[dsv4] reset failed; slot remains unusable: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] reset failed with an unknown exception; slot remains unusable\n");
    }
}

// Slot inspection never selects a different slot or mutates continuation state.
// The legacy void Reset ABI remains available; managed ownership transfers use
// this checked form for every architecture, including V4 with a DSpark drafter.
TSG_EXPORT int TSGgml_Dsv4ResetChecked(void * handle)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->active_slot) return 0;
    TSGgml_Dsv4Reset(handle);
    return !m->active_slot->v41_failed && m->active_slot->n_past == 0;
}

// Slot inspection never selects a different slot or mutates continuation state.
// Status: 1 = available V4.1 slot, 0 = unavailable; health is a separate output.
TSG_EXPORT int TSGgml_Dsv4SlotStatus(void * handle, int slot_id,
    int * head, int * checkpoint, int * healthy)
{
    if (!handle || !head || !checkpoint || !healthy) return 0;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m->hp.v41 || m->ds.loaded) return 0;
    auto found = m->slots.find(slot_id);
    if (found == m->slots.end()) return 0;
    const auto & slot = *found->second;
    *head = slot.n_past;
    *checkpoint = m->rewind_cp ? slot.cp_n_past : -1;
    *healthy = !slot.v41_failed;
    return 1;
}

TSG_EXPORT int TSGgml_Dsv4SlotCanReuse(void * handle, int slot_id, int cached_head, int target)
{
    int head = 0, checkpoint = -1, healthy = 0;
    if (!TSGgml_Dsv4SlotStatus(handle, slot_id, &head, &checkpoint, &healthy) ||
        !healthy || head != cached_head) return 0;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    return dsv41_plan_truncate(target, head, checkpoint, m->rewind_span, m->truncate_align)
        != dsv41_truncate_route::refuse;
}

TSG_EXPORT int TSGgml_Dsv4SlotReleaseGraphs(void * handle, int slot_id)
{
    if (!handle) return 0;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m->hp.v41 || m->slots.find(slot_id) == m->slots.end()) return 0;
    try { tsg_dsv4::dsv4_drop_slot_graphs(*m, slot_id); return 1; }
    catch (...) { return 0; }
}

// Admission uses the actual full-context slot buffers, not prefix length. All
// slots in this model have the same geometry. Charging every graph once is
// conservative when some belong to active requests. This never trims a graph.
TSG_EXPORT int TSGgml_Dsv4SlotCanRetain(void * handle, int slot_id,
    int retained_count, uint64_t budget_per_device)
{
    int head = 0, checkpoint = -1, healthy = 0;
    if (retained_count < 0 || !TSGgml_Dsv4SlotStatus(handle, slot_id, &head, &checkpoint, &healthy) ||
        !healthy || head <= 0) return 0;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    const auto & slot = *m->slots.find(slot_id)->second;
    uint64_t reserve = 1024ULL * 1024 * 1024;
    if (const char * e = getenv("TS_DSV4_GRAPH_CACHE_HEADROOM_MB"))
    {
        char * end = nullptr;
        const long long mb = strtoll(e, &end, 10);
        if (end != e && !*end && mb >= 0 && (uint64_t) mb <= UINT64_MAX / (1024 * 1024))
            reserve = (uint64_t) mb * 1024 * 1024;
    }
    try
    {
        for (int d = 0; d <= m->n_gpu; ++d)
        {
            uint64_t bytes = slot.buf[d] ? ggml_backend_buffer_get_size(slot.buf[d]) : 0;
            if (d == m->n_gpu)
            {
                uint64_t history = (uint64_t) m->n_ctx;
                for (const auto & entry : m->slots)
                    history = std::max(history, (uint64_t) entry.second->engram_history.capacity());
                if (history > (UINT64_MAX - bytes) / sizeof(int32_t)) return 0;
                bytes += history * sizeof(int32_t);
            }
            uint64_t graphs = 0, largest = 0;
            for (const auto & graph : m->graph_cache)
            {
                const uint64_t n = graph->sched
                    ? ggml_backend_sched_get_buffer_size(graph->sched, m->dev_backends[d]) : 0;
                if (n > UINT64_MAX - graphs) return 0;
                graphs += n;
                largest = std::max(largest, n);
            }
            size_t free_bytes = 0, total_bytes = 0;
            // n_gpu is also one for the explicitly selected CPU executor.
            // Classify the actual device, not its position in that array.
            ggml_backend_dev_t device = ggml_backend_get_device(m->dev_backends[d]);
            if (!device) return 0;
            const bool accelerator = ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU;
            if (accelerator)
            {
                ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
                if (!total_bytes) return 0; // Unknown accelerator headroom must not authorize retention.
            }
            const bool fits = dsv41_retention_fits(bytes, graphs, largest, (uint64_t) retained_count,
                budget_per_device, accelerator, free_bytes, reserve);
            fprintf(stderr, "[dsv41 retain-admission] slot=%d device=%d accelerator=%d "
                "cache_bytes=%llu all_graph_bytes=%llu largest_graph_bytes=%llu retained=%d "
                "budget_bytes=%llu free_bytes=%llu reserve_bytes=%llu accepted=%d\n",
                slot_id, d, accelerator, (unsigned long long) bytes,
                (unsigned long long) graphs, (unsigned long long) largest, retained_count,
                (unsigned long long) budget_per_device, (unsigned long long) free_bytes,
                (unsigned long long) reserve, fits);
            if (!fits) return 0;
        }
        return 1;
    }
    catch (...) { return 0; }
}

// Allocate a new sequence slot (own caches, shared weights). Returns the slot
// id, or -1 on allocation failure. The new slot is NOT made active.
TSG_EXPORT int TSGgml_Dsv4SlotAlloc(void * handle)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    try
    {
        tsg_dsv4::dsv4_slot * slot = tsg_dsv4::dsv4_slot_alloc(*m);
        return slot ? slot->id : -1;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] slot alloc failed: %s\n", e.what());
        return -1;
    }
}

// Select which slot Forward/Reset/NPast operate on. Returns 0 on success,
// -1 for an unknown slot id.
TSG_EXPORT int TSGgml_Dsv4SetActiveSlot(void * handle, int slot_id)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return -1;
    m->active_slot = it->second.get();
    return 0;
}

// Free a slot's caches and every cached graph built against them (captured
// graphs bake the slot's device addresses). The active slot cannot be freed;
// callers switch to another slot first.
TSG_EXPORT int TSGgml_Dsv4SlotFree(void * handle, int slot_id)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return -1;
    if (it->second.get() == m->active_slot)
    {
        fprintf(stderr, "[dsv4] refusing to free the active slot %d\n", slot_id);
        return -1;
    }
    try
    {
        tsg_dsv4::dsv4_drop_slot_graphs(*m, slot_id);
        m->slots.erase(it);
        return 0;
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[dsv4] slot free failed: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] slot free failed with an unknown exception\n");
    }
    return -1;
}

// TRUE token-batched decode: one token for each of `n` distinct slots in a
// single fused graph (dense weights and MoE experts loaded once per step).
// slot_ids/tokens/positions are parallel arrays; positions[i] must equal slot
// i's current n_past. On success writes logits_out as n rows of n_vocab
// floats and advances every slot by one position. Returns 0 on success,
// negative when this step can't be batched (caller falls back to per-slot
// forwards).
TSG_EXPORT int TSGgml_Dsv4ForwardBatchedDecode(
    void * handle, int n, const int32_t * slot_ids, const int32_t * tokens,
    const int32_t * positions, float * logits_out)
{
    if (!handle || n < 2 || n > tsg_dsv4::DSV4_MAX_BATCHED_SLOTS ||
        !slot_ids || !tokens || !positions || !logits_out) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    // The multi-slot graph has no DSpark feature-ring writes. Preserve each
    // request's target/draft state by taking the per-slot path when attached.
    if (!m->fused || m->ds.loaded) return -2;

    tsg_dsv4::dsv4_slot * slots[tsg_dsv4::DSV4_MAX_BATCHED_SLOTS];
    for (int i = 0; i < n; i++)
    {
        auto it = m->slots.find(slot_ids[i]);
        if (it == m->slots.end()) return -2;
        slots[i] = it->second.get();
        if (slots[i]->n_past != positions[i]) return -2;
        if (positions[i] + 1 > m->n_ctx) return -2;
        // A slot whose caches are inconsistent must be reset before it
        // can decode again; batching it would spread the failure.
        if (slots[i]->v41_failed) return -2;
        for (int j = 0; j < i; j++)
            if (slot_ids[j] == slot_ids[i]) return -2;
    }

    try
    {
        if (!tsg_dsv4::dsv4_forward_batched_decode(*m, n, slot_ids, tokens, positions, logits_out))
            return -3;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] batched decode failed: %s\n", e.what());
        return -3;
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] batched decode failed with an unknown execution exception\n");
        return -3;
    }

    for (int i = 0; i < n; i++)
        slots[i]->n_past += 1;
    return 0;
}

TSG_EXPORT int TSGgml_Dsv4DsparkBlockSize(void * handle)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    return (m && m->ds.loaded) ? m->ds.block_size : 0;
}

// Trunk forward with per-row logits (the speculative verify). Behaves exactly
// like TSGgml_Dsv4Forward otherwise, including the drafter ring commit.
TSG_EXPORT int TSGgml_Dsv4ForwardSpec(void * handle, const int32_t * tokens, int n_tokens, float * logits_out)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->active_slot || !tokens || !logits_out || n_tokens <= 0 ||
        (m->hp.v41 && (!m->ds.loaded || n_tokens > m->ds.block_size + 1))) return 0;
    if (n_tokens > m->n_ubatch)
    {
        fprintf(stderr, "[dsv4] spec forward needs a single micro-batch (%d > %d)\n", n_tokens, m->n_ubatch);
        return 0;
    }
    tsg_dsv4::dsv4_slot & slot = *m->active_slot;
    if (slot.v41_failed || (int64_t) slot.n_past + n_tokens > m->n_ctx) return 0;
    if (m->hp.v41)
        for (int i = 0; i < n_tokens; ++i)
            if (tokens[i] < 0 || tokens[i] >= m->hp.n_vocab) return 0;
    try
    {
        slot.spec_begin = slot.spec_end = -1;
        m->pos_end_hint = (int64_t) slot.n_past + n_tokens;
        if (m->moe_tp) m->moe_tp->begin_forward();
        if (tsg_dsv4::dsv4_forward_ubatch(*m, tokens, n_tokens, slot.n_past, true, logits_out, /*all_logits*/ true))
        {
            slot.spec_begin = slot.n_past;
            slot.n_past += n_tokens;
            slot.spec_end = slot.n_past;
            return 1;
        }
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[dsv4] speculative forward failed: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] speculative forward failed with an unknown execution exception\n");
    }
    slot.v41_failed = true;
    return 0;
}

TSG_EXPORT int TSGgml_Dsv4DsparkDraft(void * handle, int anchor_token, int32_t * toks_out, float * conf_out)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->active_slot || !m->ds.loaded || !toks_out || !conf_out) return 0;
    if (anchor_token < 0 || anchor_token >= m->hp.n_vocab) return 0;
    if (m->active_slot->v41_failed) return 0;
    try
    {
        return tsg_dsv4::dsv4_dspark_draft(*m, anchor_token, m->active_slot->n_past, toks_out, conf_out);
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[dsv4] DSpark draft failed: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] DSpark draft failed with an unknown execution exception\n");
    }
    // A failed draft does not commit trunk KV; its local pending graph is
    // discarded on construction failure. A normal zero return is retryable.
    return 0;
}

// Drop the KV of rejected speculative tokens. Nothing is restored: the rings
// are sized so a rejected tail cannot alias a row a later pass reads, and the
// compressed rows it wrote are recomputed before they become visible. V4.1
// additionally restricts this to the last successful verify's retained prefix;
// every compressor ring has state_extra == DSpark's block size. A
// conversational rewind (any depth, from any position) is TSGgml_Dsv4Truncate.
TSG_EXPORT int TSGgml_Dsv4Rewind(void * handle, int n_past)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->active_slot || m->active_slot->v41_failed || n_past < 0 || n_past > m->active_slot->n_past) return 0;
    auto & slot = *m->active_slot;
    if (m->hp.v41 && (!m->ds.loaded || !dsv41_dspark_can_rewind(
        slot.spec_begin, slot.spec_end, slot.n_past, n_past, m->ds.block_size, m->rewind_span))) return 0;
    slot.n_past = n_past;
    if (m->hp.v41)
    {
        if ((int64_t) slot.engram_history.size() > n_past) slot.engram_history.resize((size_t) n_past);
        if (slot.cp_n_past > n_past) slot.cp_n_past = -1;
    }
    return 1;
}

// The multiple a TSGgml_Dsv4Truncate target must be, or 0 when this model
// cannot truncate at all. The caller aligns its own reuse length DOWN to this
// rather than losing the whole prefix to a refusal over one token.
TSG_EXPORT int TSGgml_Dsv4TruncateAlign(void * handle)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->hp.v41) return 0;
    return m->truncate_align;
}

// Move the ACTIVE slot's head back to `n_past`, so the next forward appends
// there and the K/V of the first `n_past` positions is reused instead of
// re-prefilled. This is the conversational counterpart of Rewind: it serves any
// depth the slot can still honour, not just a speculative block.
//
// Returns 1 when the slot now holds exactly `n_past` positions and a forward
// from there is identical to a fresh prefill of the same tokens; 0 when it
// cannot. A normal eligibility refusal mutates nothing. An execution exception
// while restoring/resetting instead latches the slot unusable and logs the
// failure; both outcomes require the caller to Reset and re-prefill. The raw ring is
// only ring_raw positions deep, so a rewind past the checkpoint's reach has no
// correct answer (the dropped positions' K rows are gone, and recomputing one
// needs its own equally-gone window).
TSG_EXPORT int TSGgml_Dsv4Truncate(void * handle, int n_past)
{
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    if (!m || !m->hp.v41) return 0;
    tsg_dsv4::dsv4_slot * slot = m->active_slot;
    if (!slot || slot->v41_failed) return 0;

    const dsv41_truncate_route route = dsv41_plan_truncate(
        n_past, slot->n_past, m->rewind_cp ? slot->cp_n_past : -1,
        m->rewind_span, m->truncate_align);

    if (route == dsv41_truncate_route::refuse) return 0;
    if (route == dsv41_truncate_route::none) return 1;

    // Reset/restore may already have changed buffers when an allocation or
    // backend exception is raised. Contain it at this C ABI as well as Reset.
    tsg_dsv4::dsv41_slot_write_guard writes{slot};
    try
    {
        writes.started = true;
        if (route == dsv41_truncate_route::reset)
        {
            tsg_dsv4::dsv4_reset_slot(*slot);
            writes.complete = true;
            return 1;
        }
        if (route == dsv41_truncate_route::checkpoint)
            tsg_dsv4::dsv4_copy_rewind_rings(*slot, /*save*/ false);
        slot->spec_begin = slot->spec_end = -1;
        slot->n_past = n_past;
        // Engram history is indexed by absolute position; shrinking discards
        // exactly the tail and restores the hasher's expected next position.
        if ((int32_t) slot->engram_history.size() > n_past)
            slot->engram_history.resize((size_t) n_past);
        // A shadow beyond the new head contains tokens on the abandoned
        // branch. Subsequent single-token decoding does not refresh it, so it
        // must not later restore old rings alongside the new Engram history.
        if (slot->cp_n_past > n_past) slot->cp_n_past = -1;
        writes.complete = true;
        return 1;
    }
    catch (const std::exception & error)
    {
        fprintf(stderr, "[dsv4] truncate failed; reset required: %s\n", error.what());
    }
    catch (...)
    {
        fprintf(stderr, "[dsv4] truncate failed with an unknown exception; reset required\n");
    }
    return 0;
}

TSG_EXPORT void TSGgml_Dsv4Free(void * handle)
{
    if (!handle) return;
    delete (tsg_dsv4::dsv4_model *) handle;
}

#if defined(TSG_GGML_TEST_HOOKS)
// Model-free regression of the actual C entry points. A scoped acquisition
// fault intercepts before tensor access, so no model, backend or GPU allocation
// is needed. Real partial-compute/checkpoint recovery is tested separately by
// eng/tests/dsv41-failure-state.py against its independent model oracle.
#define TSG_TEST_EXPORT TSG_EXPORT
static void dsv4_test_cached_graphs(tsg_dsv4::dsv4_model & model)
{
    for (int id : {0, 1, -1})
    {
        auto graph = std::make_unique<tsg_dsv4::graph_build_result>();
        graph->slot_id = id;
        if (id == -1)
        {
            graph->bd.resize(2);
            graph->bd[0].slot_id = 0;
            graph->bd[1].slot_id = 1;
        }
        model.graph_cache.push_back(std::move(graph));
    }
}
static int dsv4_test_cached_owner_mask(const tsg_dsv4::dsv4_model & model)
{
    int mask = 0;
    for (const auto & graph : model.graph_cache)
        mask |= graph->slot_id == 0 ? 1 : graph->slot_id == 1 ? 2 : 4;
    return mask;
}
TSG_TEST_EXPORT int TSGgml_Dsv4TestExecutionBoundary(int v41, int api, int failure, int * observed, int capacity)
{
    if (!observed || capacity < (api == 4 ? 14 : 11) || v41 < 0 || v41 > 1 || api < 0 || api > 4 ||
        failure < 1 || failure > 3 || (v41 && (api == 1 || api == 3))) return -1;
    using namespace tsg_dsv4;
    struct intercept_scope
    {
        explicit intercept_scope(int kind)
        {
            dsv4_test_boundary_failure = kind;
            dsv4_test_boundary_visits = 0;
            dsv4_test_boundary_stage = 0;
        }
        ~intercept_scope() { dsv4_test_boundary_failure = 0; dsv4_test_boundary_stage = 0; }
    } intercept(failure);
    try
    {
        dsv4_model model;
        model.hp.v41 = v41 != 0;
        model.hp.n_vocab = 32;
        model.n_ctx = 128;
        model.n_ubatch = 3;
        model.fused = true;
        // Ordinary batched decode must reach the injected execution fault.
        // A loaded draft head deliberately refuses that route before compute;
        // mode 4 checks that separate refusal without weakening either gate.
        model.ds.loaded = api == 3 || api == 4;
        for (int id = 0; id < 2; ++id)
        {
            auto slot = std::make_unique<dsv4_slot>();
            slot->id = id;
            slot->n_past = 5;
            model.slots.emplace(id, std::move(slot));
        }
        model.active_slot = model.slots.at(0).get();
        dsv4_test_cached_graphs(model);
        auto * original_active = model.active_slot;
        auto * original_peer = model.slots.at(1).get();
        if (api == 4)
        {
            original_active->engram_history.assign(5, 11);
            original_peer->engram_history.assign(5, 22);
        }
        const auto original_history = original_active->engram_history;
        const auto original_peer_history = original_peer->engram_history;
        const bool original_head_loaded = model.ds.loaded;
        std::vector<const graph_build_result *> original_graphs;
        for (const auto & graph : model.graph_cache) original_graphs.push_back(graph.get());
        const int32_t tokens[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        const int32_t ids[2] = {0, 1}, positions[2] = {5, 5};
        float scratch[64] = {};
        int32_t drafted[8] = {};
        if (api == 4)
        {
            std::fill(std::begin(scratch), std::end(scratch), 12345.0f);
            std::fill(std::begin(drafted), std::end(drafted), 12345);
        }
        auto invoke = [&]()
        {
            switch (api)
            {
                case 0: return TSGgml_Dsv4Forward(&model, tokens, 8, scratch);
                case 1: return TSGgml_Dsv4ForwardSpec(&model, tokens, 2, scratch);
                case 2:
                case 4: return TSGgml_Dsv4ForwardBatchedDecode(&model, 2, ids, tokens, positions, scratch);
                default: return TSGgml_Dsv4DsparkDraft(&model, 1, drafted, scratch);
            }
        };
        observed[0] = invoke();
        observed[1] = model.active_slot->v41_failed;
        observed[2] = model.active_slot->n_past;
        observed[3] = model.slots.at(1)->v41_failed;
        observed[4] = model.slots.at(1)->n_past;
        observed[5] = invoke();
        observed[6] = dsv4_test_boundary_visits;
        if (api == 4)
        {
            // Capture refusal invariants before Reset can hide any mutation.
            observed[11] = model.ds.loaded == original_head_loaded &&
                !original_active->v41_failed && original_active->n_past == 5 &&
                !original_peer->v41_failed && original_peer->n_past == 5 &&
                original_active->engram_history == original_history &&
                original_peer->engram_history == original_peer_history;
            bool graphs_unchanged = model.graph_cache.size() == original_graphs.size();
            if (graphs_unchanged)
            {
                size_t i = 0;
                for (const auto & graph : model.graph_cache)
                {
                    graphs_unchanged = graphs_unchanged && graph.get() == original_graphs[i];
                    ++i;
                }
            }
            observed[12] = model.active_slot == original_active && model.slots.size() == 2 &&
                model.slots.at(0).get() == original_active && model.slots.at(1).get() == original_peer &&
                graphs_unchanged && dsv4_test_cached_owner_mask(model) == 7 &&
                original_graphs[0]->slot_id == 0 && original_graphs[1]->slot_id == 1 &&
                original_graphs[2]->slot_id == -1 && original_graphs[2]->bd.size() == 2 &&
                original_graphs[2]->bd[0].slot_id == 0 && original_graphs[2]->bd[1].slot_id == 1;
            observed[13] = std::all_of(std::begin(scratch), std::end(scratch),
                    [](float value) { return value == 12345.0f; }) &&
                std::all_of(std::begin(drafted), std::end(drafted),
                    [](int32_t value) { return value == 12345; });
        }
        TSGgml_Dsv4Reset(&model);
        observed[7] = model.active_slot->v41_failed;
        observed[8] = model.active_slot->n_past;
        observed[9] = (int) model.graph_cache.size();
        observed[10] = dsv4_test_cached_owner_mask(model);
        return 0;
    }
    catch (...)
    {
        // Fixture construction can fail, but an exception escaping the actual
        // entry point is never reported as a successful boundary regression.
        return -2;
    }
}

TSG_TEST_EXPORT int TSGgml_Dsv4TestResetTruncateBoundary(int api, int failure, int * observed, int capacity)
{
    if (!observed || capacity < 16 || api < 0 || api > 5 || failure < 1 || failure > 2) return -1;
    using namespace tsg_dsv4;
    struct intercept_scope
    {
        ~intercept_scope() { dsv4_test_boundary_failure = 0; dsv4_test_boundary_stage = 0; }
    } intercept;
    try
    {
        dsv4_model model;
        model.hp.v41 = api < 4;
        model.ds.loaded = api == 5;
        model.hp.n_vocab = 32;
        model.n_ctx = 128;
        model.n_ubatch = 3;
        model.rewind_cp = true;
        model.rewind_span = 0;
        model.backends[0] = ggml_backend_cpu_init();
        if (!model.backends[0]) return -2;
        model.n_backends = 1;
        auto owner = std::make_unique<dsv4_slot>();
        auto * slot = owner.get();
        model.slots.emplace(0, std::move(owner));
        model.active_slot = slot;
        dsv4_test_cached_graphs(model);
        slot->n_past = 20;
        slot->cp_n_past = 10;
        slot->engram_history.assign(20, 1);
        slot->layers.resize(1);
        slot->ctx[0] = ggml_init({64 * 1024, nullptr, true});
        if (!slot->ctx[0]) return -2;
        auto & layer = slot->layers[0];
        layer.raw_k = ggml_new_tensor_1d(slot->ctx[0], GGML_TYPE_F32, 4);
        layer.raw_k_cp = ggml_new_tensor_1d(slot->ctx[0], GGML_TYPE_F32, 4);
        slot->buf[0] = ggml_backend_alloc_ctx_tensors(slot->ctx[0], model.backends[0]);
        if (!slot->buf[0]) return -2;
        const float live[4] = {1, 2, 3, 4}, shadow[4] = {77, 78, 79, 80};
        ggml_backend_tensor_set(layer.raw_k, live, 0, sizeof(live));
        ggml_backend_tensor_set(layer.raw_k_cp, shadow, 0, sizeof(shadow));
        auto first = [](ggml_tensor * tensor)
        {
            float value = 0;
            ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
            return (int) value;
        };
        // An ordinary out-of-range refusal must still leave all state intact.
        observed[12] = TSGgml_Dsv4Truncate(&model, 21);
        observed[13] = !slot->v41_failed && slot->n_past == 20 && slot->cp_n_past == 10 &&
            slot->engram_history.size() == 20 && first(layer.raw_k) == 1 && first(layer.raw_k_cp) == 77;
        dsv4_test_boundary_failure = failure;
        dsv4_test_boundary_stage = api == 2 ? 2 : 1;
        dsv4_test_boundary_visits = 0;
        if (api == 0) { TSGgml_Dsv4Reset(&model); observed[0] = 0; }
        else if (api >= 3) observed[0] = TSGgml_Dsv4ResetChecked(&model);
        else observed[0] = TSGgml_Dsv4Truncate(&model, api == 1 ? 0 : 10);
        observed[1] = slot->v41_failed;
        observed[2] = slot->n_past;
        observed[3] = first(layer.raw_k);
        observed[4] = first(layer.raw_k_cp);
        const int32_t token = 1;
        float logits[32] = {};
        observed[5] = TSGgml_Dsv4Forward(&model, &token, 1, logits);
        observed[6] = TSGgml_Dsv4Truncate(&model, 0);
        observed[7] = dsv4_test_boundary_visits;
        dsv4_test_boundary_failure = 0;
        if (!TSGgml_Dsv4ResetChecked(&model)) return -3;
        observed[8] = slot->v41_failed;
        observed[9] = slot->n_past;
        observed[10] = first(layer.raw_k);
        observed[11] = first(layer.raw_k_cp);
        observed[14] = (int) model.graph_cache.size();
        observed[15] = dsv4_test_cached_owner_mask(model);
        return 0;
    }
    catch (...)
    {
        return -2;
    }
}
#undef TSG_TEST_EXPORT
#endif
