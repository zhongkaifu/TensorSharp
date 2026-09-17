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
#include "ggml_ops_gptoss_kv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace tsg;

// ============================================================================
// TRUE TOKEN-BATCHED GPT-OSS decode: N concurrent sequences, one token each,
// in ONE ggml graph (one compute buffer, every weight loaded once).
//
// This is the llama-parity concurrency path, and its attention is built the
// way llama.cpp's multi-stream KV cache and vLLM's decode step are built:
//
//   * Every sequence's K/V lives in one persistent per-layer ARENA tensor
//     [hd, cap, kvH, n_slots] (head-plane-major, matching the tsg_gptoss
//     window/host layout), so the whole batch's KV write is ONE ggml_set_rows
//     per K and per V per layer (row index = slot*kvH*cap + head*cap + pos),
//     and the whole batch's attention is ONE ggml_flash_attn_ext with
//     ne3 = n_slots. A first version instead viewed each sequence's private
//     KV window and ran per-sequence fattn: correct, but the graph grew to
//     ~8400 nodes at N=16 (per-seq CONT/VIEW/SET_ROWS/fattn) and almost half
//     of every step went to bookkeeping kernels.
//
//   * The graph is SLOT-STABLE, like vLLM's captured decode graphs: it depends
//     only on (model, n_slots bucket, cap) — never on WHICH sequences occupy
//     the slots. hidden/positions/kv-row-indices/masks are graph inputs
//     refreshed per step, so request churn keeps replaying the same captured
//     CUDA graph instead of rebuilding + recapturing per composition. A
//     sequence entering the batch is assigned a free slot and its existing
//     rows are copied into its arena slice once; padded (unoccupied) columns
//     write to a dedicated scratch slice and their outputs are discarded.
//
//   * Causality and each layer's sliding window live entirely in the mask
//     ([cap, 1, 1, n_slots], one per distinct (swa, window) geometry, shared
//     by all layers of that geometry). The CUDA fattn kernel derives KV_max
//     from the mask and skips fully-masked tail blocks, so mask-bounded
//     uniform caps do not cost extra attention compute.
//
// COHERENCE with the solo kernels: the per-request tsg_gptoss device windows
// and host mirrors remain the authoritative interchange format. While a
// sequence decodes in a slot, its newest rows exist ONLY in the arena; the
// slot records [clean, len) as the not-yet-flushed range. Any other consumer
// of that cache — solo prefill/decode (kv_acquire), host sync, KV
// snapshot/growth — first hits one of the batched_on_* hooks below, which
// flushes the dirty rows straight into the HOST mirror and retires the slot.
// The sequence simply re-joins (one copy-in) the next time it decodes
// batched. Window rows_valid never exceeds the flushed prefix of a slot, so
// window content is never stale; the flush only appends beyond it on the
// host, exactly where kv_upload resumes from.
//
// Lifecycle: unlike the solo decode pool, this pool does NOT die on every
// prefill — nothing in it references KV windows or the shared gallocr pool.
// It is dropped on backend teardown / device-copy release (explicit calls in
// ggml_ops_core.cpp) and via TSGgml_GptOssResetBatchedDecodeCache.
//
// v1 scope: no MoE CPU offload (decline), F32/F16 KV, folded lm_head, NO-WRAP
// (positions[s] + 1 <= cache rows). The legacy per-seq-window path remains as
// the fallback for backends without persist support (or TS_GPTOSS_BATCHED_ARENA=0).
// ============================================================================
namespace
{
    constexpr int kGobMaxEntries = 4;
    constexpr const char* kGptOssBatchKernel = "GPT-OSS batched decode";

    int gob_slot_bucket(int n)
    {
        int b = 2;
        while (b < n) b *= 2;
        return b;
    }

    std::int64_t gob_cap_round(std::int64_t rows)
    {
        const std::int64_t q = 1024;
        return ((rows + q - 1) / q) * q;
    }

    struct GobSlot
    {
        bool active = false;
        const void* key = nullptr;              // layer-0 K host cache pointer
        std::vector<const void*> k_hosts;       // per layer host cache pointers
        std::vector<const void*> v_hosts;
        int cache_rows = 0;
        std::int64_t len = 0;                   // arena rows [0, len) valid
        std::int64_t clean = 0;                 // rows [clean, len) not yet in the host mirror
        std::uint64_t last_used = 0;
    };

    struct GobEntry
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;       // [H, n_slots]
        ggml_tensor* pos_in = nullptr;          // I32 [n_slots]
        ggml_tensor* idx_in = nullptr;          // I64 [kvH * n_slots] set_rows targets
        ggml_tensor* logits_out = nullptr;      // [vocab, n_slots]
        ggml_tensor* sampled_out = nullptr;     // I32 [n_slots] device argmax (greedy fast path)
        bool has_argmax = false;
        struct MaskEnt { int swa = 0; int window = 0; ggml_tensor* t = nullptr; };
        std::vector<MaskEnt> masks;             // one per distinct (swa, window)
        std::vector<ggml_tensor*> k_arena;      // per layer [hd, (n_slots+1)*kvH*cap]
        std::vector<ggml_tensor*> v_arena;
        const void* sig_disc = nullptr;
        int num_layers = 0, H = 0, vocab = 0, n_slots = 0;
        int hd = 0, kvH = 0;
        std::int64_t cap = 0;
        std::int64_t rows_per_slot = 0;         // kvH * cap
        ggml_type kv_type = GGML_TYPE_F32;
        std::vector<GobSlot> slots;

        // reused per-step host staging
        std::vector<float> hidden_stage;
        std::vector<std::int32_t> pos_stage;
        std::vector<std::int64_t> idx_stage;
        std::vector<ggml_fp16_t> mask_stage;
        std::vector<float> logits_stage;
        std::vector<std::int32_t> sampled_stage;

        void release_graph()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            hidden_in = pos_in = idx_in = logits_out = nullptr;
            sampled_out = nullptr;
            has_argmax = false;
            masks.clear(); k_arena.clear(); v_arena.clear();
            valid = false;
        }
    };

    struct GobPool
    {
        GobEntry entries[kGobMaxEntries];
        std::uint64_t used[kGobMaxEntries] = {};
        std::uint64_t clock = 0;
    };
    GobPool g_gob_pools[tsg::TSG_MAX_DEVICES];
    GobPool& gob_pool() { return g_gob_pools[tsg::g_active_rank]; }

    // host cache pointer (any layer, K or V) -> (entry index, slot index)
    using GobRegistry = std::unordered_map<const void*, std::pair<int, int>>;
    GobRegistry g_gob_registries[tsg::TSG_MAX_DEVICES];
    GobRegistry& gob_registry() { return g_gob_registries[tsg::g_active_rank]; }

    // Set while the batched kernel (or a flush it triggers) runs, so the
    // batched_on_* hooks fired by kv_acquire / kv_drop calls we make ourselves
    // do not recurse or retire the slot being serviced.
    thread_local bool g_gob_guard = false;
    struct GobGuard
    {
        bool prev;
        GobGuard() : prev(g_gob_guard) { g_gob_guard = true; }
        ~GobGuard() { g_gob_guard = prev; }
    };

    void gob_unregister_slot(int entry_idx, GobSlot& sl)
    {
        GobRegistry& reg = gob_registry();
        for (const void* p : sl.k_hosts) reg.erase(p);
        for (const void* p : sl.v_hosts) reg.erase(p);
        (void)entry_idx;
        sl = GobSlot{};
    }

    // Flush a slot's dirty arena rows [clean, len) into its HOST mirrors and
    // retire the slot. Host layout is head-plane-major with cache_rows pitch —
    // identical shape to the arena slice, so this is one tensor_get per
    // (layer, K/V, head). Windows are left alone: their rows_valid never
    // exceeds `clean`, so their content stays a valid prefix and kv_upload
    // resumes exactly at the rows written here. kv_mutex must be held.
    void gob_flush_and_drop_slot(GobEntry& e, int entry_idx, int slot_idx)
    {
        GobSlot& sl = e.slots[slot_idx];
        if (!sl.active) return;
        GobGuard guard;
        if (sl.len > sl.clean && g_backend != nullptr && e.valid)
        {
            const std::size_t row_bytes = ggml_row_size(e.kv_type, e.hd);
            const std::size_t bytes = static_cast<std::size_t>(sl.len - sl.clean) * row_bytes;
            for (int l = 0; l < e.num_layers; l++)
            {
                for (int which = 0; which < 2; which++)
                {
                    ggml_tensor* arena = (which == 0) ? e.k_arena[l] : e.v_arena[l];
                    const void* host = (which == 0) ? sl.k_hosts[l] : sl.v_hosts[l];
                    if (arena == nullptr || host == nullptr) continue;
                    for (int h = 0; h < e.kvH; h++)
                    {
                        const std::int64_t arow = static_cast<std::int64_t>(slot_idx) * e.rows_per_slot +
                                                  static_cast<std::int64_t>(h) * e.cap + sl.clean;
                        char* dst = const_cast<char*>(static_cast<const char*>(host)) +
                            (static_cast<std::int64_t>(h) * sl.cache_rows + sl.clean) * row_bytes;
                        ggml_backend_tensor_get(arena, dst, static_cast<std::size_t>(arow) * row_bytes, bytes);
                    }
                }
            }
        }
        gob_unregister_slot(entry_idx, sl);
    }

    void gob_flush_entry(GobEntry& e, int entry_idx)
    {
        for (int s = 0; s < static_cast<int>(e.slots.size()); s++)
            gob_flush_and_drop_slot(e, entry_idx, s);
    }

    // Per-geometry mask fill over [cap, n_slots]: column s zero on [lo, pos_s],
    // -inf elsewhere (span fills — this runs every step). Slots without a
    // sequence in this call (pos < 0) attend row 0 only: reads are harmless
    // (their outputs are discarded) and a fully-masked column would softmax
    // over nothing.
    void gob_fill_arena_mask(std::vector<ggml_fp16_t>& mask, std::int64_t cap, int n_slots,
                             const std::int64_t* pos_by_slot, bool swa, int sliding_window)
    {
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        mask.resize(static_cast<std::size_t>(cap) * n_slots);
        for (int s = 0; s < n_slots; s++)
        {
            ggml_fp16_t* col = &mask[static_cast<std::size_t>(s) * cap];
            const std::int64_t pos = pos_by_slot[s];
            if (pos < 0)
            {
                col[0] = zero_val;
                std::fill(col + 1, col + cap, neg_inf);
                continue;
            }
            const std::int64_t lo = (swa && sliding_window > 0)
                ? std::max<std::int64_t>(0, pos - sliding_window + 1) : 0;
            const std::int64_t hi = std::min<std::int64_t>(pos, cap - 1);
            std::fill(col, col + lo, neg_inf);
            std::fill(col + lo, col + hi + 1, zero_val);
            std::fill(col + hi + 1, col + cap, neg_inf);
        }
    }
}

// ---------------------------------------------------------------------------
// Coherence hooks, called (with kv_mutex held) from the tsg_gptoss KV window
// registry and the host-sync entry point. See the header comment.
// ---------------------------------------------------------------------------
namespace tsg_gptoss
{
    void batched_on_external_acquire(const void* host_cache)
    {
        if (g_gob_guard || host_cache == nullptr) return;
        GobRegistry& reg = gob_registry();
        auto it = reg.find(host_cache);
        if (it == reg.end()) return;
        const int ei = it->second.first;
        const int si = it->second.second;
        gob_flush_and_drop_slot(gob_pool().entries[ei], ei, si);
    }

    void batched_on_drop(const void* host_cache)
    {
        if (g_gob_guard || host_cache == nullptr) return;
        GobRegistry& reg = gob_registry();
        auto it = reg.find(host_cache);
        if (it == reg.end()) return;
        // The host cache was rewritten behind the kernel's back (snapshot
        // restore / holder reuse): the arena rows are stale — drop, no flush.
        GobEntry& e = gob_pool().entries[it->second.first];
        gob_unregister_slot(it->second.first, e.slots[it->second.second]);
    }

    void batched_on_drop_all()
    {
        if (g_gob_guard) return;
        for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
        {
            for (auto& e : g_gob_pools[r].entries)
                for (auto& sl : e.slots)
                    sl = GobSlot{};
            g_gob_registries[r].clear();
        }
    }
}

// Drop all batched decode state. Dirty slots are flushed to their host
// mirrors first (when the backend is still alive), so no generated KV is
// lost. NOT chained into the solo TSGgml_GptOssResetDecodeCache anymore: the
// solo pool dies on every prefill ("prefill moves the compute pool"), but
// nothing here references the shared compute pool or the KV windows, and
// surviving prefills is what keeps request churn on the captured graph.
TSG_EXPORT void TSGgml_GptOssResetBatchedDecodeCache()
{
    std::lock_guard<std::mutex> lock(tsg_gptoss::kv_mutex());
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
    {
        for (int i = 0; i < kGobMaxEntries; i++)
        {
            GobEntry& e = g_gob_pools[r].entries[i];
            if (r == tsg::g_active_rank)
                gob_flush_entry(e, i);
            else
                for (auto& sl : e.slots) sl = GobSlot{};
            e.release_graph();
            e.slots.clear();
        }
        g_gob_registries[r].clear();
    }
}

// ---------------------------------------------------------------------------
// Legacy fallback: per-sequence window views + per-sequence fattn, no
// persistent state. Used when the backend has no persist support or the
// arena path is disabled. One-shot graph per call.
// ---------------------------------------------------------------------------
static int gob_decode_batched_legacy(
    const TSGgmlGptOssLayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,
    void** k_cache_arr, void** v_cache_arr,
    const int* cache_sizes,
    const int* positions,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    std::int32_t* sampled_data)
{
    const int H = layers[0].hidden_size;
    const int kvType = layers[0].kv_cache_type;

    int maxTotal = 0;
    for (int s = 0; s < n_seqs; s++)
        maxTotal = std::max(maxTotal, positions[s] + 1);

    std::unique_lock<std::mutex> kv_lock(tsg_gptoss::kv_mutex());
    GobGuard guard;   // our own kv_acquires must not retire arena slots

    std::vector<tsg_gptoss::KvWindow*> k_wins(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);
    std::vector<tsg_gptoss::KvWindow*> v_wins(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);
    for (int l = 0; l < num_layers; l++)
    {
        for (int s = 0; s < n_seqs; s++)
        {
            TSGgmlGptOssLayerDesc d = layers[l];
            d.k_cache = k_cache_arr[static_cast<std::size_t>(l) * n_seqs + s];
            d.v_cache = v_cache_arr[static_cast<std::size_t>(l) * n_seqs + s];
            d.cache_size = cache_sizes[s];
            const std::int64_t needed = std::min<std::int64_t>(maxTotal, cache_sizes[s]);
            tsg_gptoss::KvWindow* kw = nullptr;
            tsg_gptoss::KvWindow* vw = nullptr;
            if (!kv_acquire_pair(d, needed, kw, vw))
            {
                set_last_error("GPT-OSS batched decode: failed to acquire a device KV window.");
                return 0;
            }
            k_wins[static_cast<std::size_t>(l) * n_seqs + s] = kw;
            v_wins[static_cast<std::size_t>(l) * n_seqs + s] = vw;
        }
    }

    std::vector<int> win(num_layers, 0);
    for (int l = 0; l < num_layers; l++)
    {
        std::int64_t cap = std::numeric_limits<std::int64_t>::max();
        for (int s = 0; s < n_seqs; s++)
        {
            cap = std::min(cap, k_wins[static_cast<std::size_t>(l) * n_seqs + s]->capacity);
            cap = std::min(cap, v_wins[static_cast<std::size_t>(l) * n_seqs + s]->capacity);
        }
        win[l] = static_cast<int>(std::min<std::int64_t>(maxTotal, cap));
        if (win[l] < maxTotal)
        {
            set_last_error("GPT-OSS batched decode: KV window smaller than the longest sequence.");
            return 0;
        }
    }

    for (int l = 0; l < num_layers; l++)
    {
        for (int s = 0; s < n_seqs; s++)
        {
            const std::size_t i = static_cast<std::size_t>(l) * n_seqs + s;
            const std::int64_t kvalid = std::min<std::int64_t>(k_wins[i]->rows_valid, positions[s]);
            const std::int64_t vvalid = std::min<std::int64_t>(v_wins[i]->rows_valid, positions[s]);
            kv_upload(k_wins[i], k_cache_arr[i], cache_sizes[s], kvalid, positions[s]);
            kv_upload(v_wins[i], v_cache_arr[i], cache_sizes[s], vvalid, positions[s]);
        }
    }

    const std::size_t ctx_size = 64 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("GPT-OSS batched decode: failed to acquire ggml context.");
        return 0;
    }
    ggml_context* ctx = context.value;

    struct LayerTensors
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* qkv_w = nullptr;
        ggml_tensor* qkv_b = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* k_b = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* v_b = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* o_b = nullptr;
        ggml_tensor* sinks = nullptr;
        ggml_tensor* post_attn_norm_w = nullptr;
        ggml_tensor* gate_inp_w = nullptr;
        ggml_tensor* gate_inp_b = nullptr;
        ggml_tensor* gate_exps = nullptr;
        ggml_tensor* gate_exps_b = nullptr;
        ggml_tensor* up_exps = nullptr;
        ggml_tensor* up_exps_b = nullptr;
        ggml_tensor* down_exps = nullptr;
        ggml_tensor* down_exps_b = nullptr;
        std::vector<ggml_tensor*> k_cache;
        std::vector<ggml_tensor*> v_cache;
        std::vector<ggml_tensor*> k_cpy;
        std::vector<ggml_tensor*> v_cpy;
        std::vector<ggml_tensor*> attn_col_cpy;
        ggml_tensor* attn_mask = nullptr;
        std::vector<ggml_fp16_t> mask_data;
    };
    std::vector<LayerTensors> lt(num_layers);

    ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_seqs);
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);

    for (int l = 0; l < num_layers; l++)
    {
        const TSGgmlGptOssLayerDesc& d = layers[l];
        LayerTensors& t = lt[l];
        const int hd = d.head_dim;
        const int kvH = d.num_kv_heads;
        const int qDim = d.num_heads * hd;
        const int kDim = kvH * hd;
        const int nExp = d.num_experts;

        t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
        const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);
        if (d.qkv_b != nullptr) t.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkvDim);
        if (d.separate_qkv != 0)
        {
            t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
            t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
            if (d.k_b != nullptr) t.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            if (d.v_b != nullptr) t.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
        }
        t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
        if (d.o_b != nullptr) t.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        if (d.sinks != nullptr) t.sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.num_heads);
        t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
        if (d.gate_inp_b != nullptr) t.gate_inp_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp);
        t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ge_type), d.ge_ne0, d.ge_ne1, nExp);
        t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ue_type), d.ue_ne0, d.ue_ne1, nExp);
        t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
        if (d.gate_exps_b != nullptr) t.gate_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ge_ne1, nExp);
        if (d.up_exps_b != nullptr) t.up_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ue_ne1, nExp);
        if (d.down_exps_b != nullptr) t.down_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.de_ne1, nExp);

        t.k_cache.resize(n_seqs);
        t.v_cache.resize(n_seqs);
        t.k_cpy.resize(n_seqs, nullptr);
        t.v_cpy.resize(n_seqs, nullptr);
        t.attn_col_cpy.resize(n_seqs, nullptr);
        for (int s = 0; s < n_seqs; s++)
        {
            const std::size_t i = static_cast<std::size_t>(l) * n_seqs + s;
            t.k_cache[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, k_wins[i]->capacity, kvH);
            t.v_cache[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, v_wins[i]->capacity, kvH);
        }
        t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win[l], 1, 1, n_seqs);
    }

    ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
    ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

    ggml_tensor* hidden = hidden_t;
    for (int l = 0; l < num_layers; l++)
    {
        const TSGgmlGptOssLayerDesc& d = layers[l];
        LayerTensors& t = lt[l];
        const int hd = d.head_dim;
        const int nH = d.num_heads;
        const int kvH = d.num_kv_heads;
        const int qDim = nH * hd;
        const int kDim = kvH * hd;
        const int nExp = d.num_experts;
        const int nUsed = d.num_experts_used;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, d.eps), t.attn_norm_w);

        ggml_tensor* q_raw;
        ggml_tensor* k_raw;
        ggml_tensor* v_raw;
        if (d.separate_qkv != 0)
        {
            ggml_tensor* q_proj = ggml_mul_mat(ctx, t.qkv_w, normed);
            if (t.qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, t.qkv_b);
            ggml_tensor* k_proj = ggml_mul_mat(ctx, t.k_w, normed);
            if (t.k_b != nullptr) k_proj = ggml_add(ctx, k_proj, t.k_b);
            ggml_tensor* v_proj = ggml_mul_mat(ctx, t.v_w, normed);
            if (t.v_b != nullptr) v_proj = ggml_add(ctx, v_proj, t.v_b);
            q_raw = q_proj;
            k_raw = k_proj;
            v_raw = v_proj;
        }
        else
        {
            ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);
            if (t.qkv_b != nullptr) qkv = ggml_add(ctx, qkv, t.qkv_b);
            q_raw = ggml_view_2d(ctx, qkv, qDim, n_seqs, qkv->nb[1], 0);
            k_raw = ggml_view_2d(ctx, qkv, kDim, n_seqs, qkv->nb[1],
                static_cast<std::size_t>(qDim) * sizeof(float));
            v_raw = ggml_view_2d(ctx, qkv, kDim, n_seqs, qkv->nb[1],
                static_cast<std::size_t>(qDim + kDim) * sizeof(float));
        }

        ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), hd, nH, n_seqs);
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), hd, kvH, n_seqs);
        ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), hd, kvH, n_seqs);

        ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
            d.rope_n_dims, 2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
            1.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
            d.rope_n_dims, 2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
            1.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor* q_rope_cont = ggml_cont(ctx, q_rope);
        ggml_tensor* attn_2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qDim, n_seqs);
        for (int s = 0; s < n_seqs; s++)
        {
            const std::size_t i = static_cast<std::size_t>(l) * n_seqs + s;
            ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, hd, kvH, 1,
                k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
            ggml_tensor* v_s = ggml_view_3d(ctx, v_3d, hd, kvH, 1,
                v_3d->nb[1], v_3d->nb[2], static_cast<std::size_t>(s) * v_3d->nb[2]);
            ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));
            ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));

            const std::size_t kv_offset = static_cast<std::size_t>(positions[s]) * t.k_cache[s]->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache[s], hd, 1, kvH,
                t.k_cache[s]->nb[1], t.k_cache[s]->nb[2], kv_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache[s], hd, 1, kvH,
                t.v_cache[s]->nb[1], t.v_cache[s]->nb[2], kv_offset);
            t.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
            t.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);

            ggml_tensor* k_win_v = view_kv_cache_window(ctx, t.k_cache[s], hd,
                static_cast<int>(k_wins[i]->capacity), kvH, 0, win[l], kvType, 1);
            ggml_tensor* v_win_v = view_kv_cache_window(ctx, t.v_cache[s], hd,
                static_cast<int>(v_wins[i]->capacity), kvH, 0, win[l], kvType, 1);
            if (k_win_v == nullptr || v_win_v == nullptr)
            {
                set_last_error("GPT-OSS batched decode: failed to build KV window views.");
                return 0;
            }

            ggml_tensor* q_s = ggml_view_3d(ctx, q_rope_cont, hd, nH, 1,
                q_rope_cont->nb[1], q_rope_cont->nb[2],
                static_cast<std::size_t>(s) * q_rope_cont->nb[2]);
            ggml_tensor* q_attn = ggml_permute(ctx, q_s, 0, 2, 1, 3);
            ggml_tensor* mask_s = ggml_view_4d(ctx, t.attn_mask, win[l], 1, 1, 1,
                t.attn_mask->nb[1], t.attn_mask->nb[2], t.attn_mask->nb[3],
                static_cast<std::size_t>(s) * t.attn_mask->nb[3]);
            ggml_tensor* fa = flash_attn_ext_guarded(ctx, "GPT-OSS batched decode", q_attn, k_win_v, v_win_v, mask_s,
                scale, 0.0f, 0.0f, t.sinks, GGML_PREC_F32);

            ggml_tensor* fa_flat = ggml_reshape_1d(ctx, fa, qDim);
            ggml_tensor* col = ggml_view_1d(ctx, attn_2d, qDim,
                static_cast<std::size_t>(s) * attn_2d->nb[1]);
            t.attn_col_cpy[s] = ggml_cpy(ctx, fa_flat, col);
        }

        ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_2d);
        if (t.o_b != nullptr) o_mm = ggml_add(ctx, o_mm, t.o_b);
        ggml_tensor* ffn_inp = ggml_add(ctx, hidden, o_mm);

        ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, d.eps), t.post_attn_norm_w);

        ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, moe_in);
        if (t.gate_inp_b != nullptr)
            router_logits = ggml_add(ctx, router_logits, t.gate_inp_b);
        ggml_tensor* sel = ggml_top_k(ctx, router_logits, nUsed);
        ggml_tensor* logits_r = ggml_reshape_3d(ctx, router_logits, 1, nExp, n_seqs);
        ggml_tensor* w = ggml_get_rows(ctx, logits_r, sel);
        ggml_tensor* w_soft = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, nUsed, n_seqs));
        ggml_tensor* w_final = ggml_reshape_3d(ctx, w_soft, 1, nUsed, n_seqs);

        ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, n_seqs);
        ggml_tensor* gate = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);
        if (t.gate_exps_b != nullptr) gate = ggml_add_id(ctx, gate, t.gate_exps_b, sel);
        ggml_tensor* up = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
        if (t.up_exps_b != nullptr) up = ggml_add_id(ctx, up, t.up_exps_b, sel);
        ggml_tensor* act = ggml_swiglu_oai(ctx, gate, up, d.oai_alpha, d.oai_limit);
        ggml_tensor* down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);
        if (t.down_exps_b != nullptr) down = ggml_add_id(ctx, down, t.down_exps_b, sel);
        ggml_tensor* weighted = ggml_mul(ctx, down, w_final);

        ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], 0);
        for (int u = 1; u < nUsed; ++u)
        {
            ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2],
                static_cast<std::size_t>(u) * weighted->nb[1]);
            moe_out = ggml_add(ctx, moe_out, view_u);
        }
        hidden = ggml_add(ctx, ffn_inp, moe_out);
    }

    ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, layers[0].eps), final_norm_t);
    ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);
    ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
    ggml_tensor* out_cpy = ggml_cpy(ctx, logits, logits_out);
    ggml_set_output(out_cpy);

    const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (256 + 48 * static_cast<std::size_t>(n_seqs)) + 2048;
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
    for (int l = 0; l < num_layers; l++)
        for (int s = 0; s < n_seqs; s++)
        {
            if (lt[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy[s]);
            if (lt[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy[s]);
            if (lt[l].attn_col_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].attn_col_cpy[s]);
        }
    ggml_build_forward_expand(graph, out_cpy);

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

    bool bind_failed = false;
    for (int l = 0; l < num_layers && !bind_failed; l++)
    {
        const TSGgmlGptOssLayerDesc& d = layers[l];
        LayerTensors& t = lt[l];
        const int hd = d.head_dim;
        const int nExp = d.num_experts;
        const int qDim = d.num_heads * hd;
        const int kDim = d.num_kv_heads * hd;
        const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);

        bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
        bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
        bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
        bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
        bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.ge_bytes), true);
        bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.ue_bytes), true);
        bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
        bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
        bind_or_mark(t.gate_inp_b, d.gate_inp_b, static_cast<std::size_t>(nExp) * sizeof(float), true);
        bind_or_mark(t.qkv_b, d.qkv_b, static_cast<std::size_t>(qkvDim) * sizeof(float), true);
        bind_or_mark(t.k_b, d.k_b, static_cast<std::size_t>(kDim) * sizeof(float), true);
        bind_or_mark(t.v_b, d.v_b, static_cast<std::size_t>(kDim) * sizeof(float), true);
        bind_or_mark(t.o_b, d.o_b, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(t.sinks, d.sinks, static_cast<std::size_t>(d.num_heads) * sizeof(float), true);
        bind_or_mark(t.gate_exps_b, d.gate_exps_b, static_cast<std::size_t>(d.ge_ne1) * nExp * sizeof(float), true);
        bind_or_mark(t.up_exps_b, d.up_exps_b, static_cast<std::size_t>(d.ue_ne1) * nExp * sizeof(float), true);
        bind_or_mark(t.down_exps_b, d.down_exps_b, static_cast<std::size_t>(d.de_ne1) * nExp * sizeof(float), true);

        for (int s = 0; s < n_seqs && !bind_failed; s++)
        {
            const std::size_t i = static_cast<std::size_t>(l) * n_seqs + s;
            if (ggml_backend_tensor_alloc(k_wins[i]->buffer, t.k_cache[s], k_wins[i]->tensor->data) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(v_wins[i]->buffer, t.v_cache[s], v_wins[i]->tensor->data) != GGML_STATUS_SUCCESS)
            {
                bind_failed = true;
            }
        }

        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            t.mask_data.assign(static_cast<std::size_t>(win[l]) * n_seqs, neg_inf);
            for (int s = 0; s < n_seqs; s++)
            {
                const int pos = positions[s];
                const int lo = (d.is_swa != 0 && d.sliding_window > 0) ? std::max(0, pos - d.sliding_window + 1) : 0;
                const int hi = std::min(pos, win[l] - 1);
                for (int k = lo; k <= hi; k++)
                    t.mask_data[static_cast<std::size_t>(s) * win[l] + k] = zero_val;
            }
            bind_or_mark(t.attn_mask, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
        }
    }
    if (bind_failed)
    {
        set_last_error("GPT-OSS batched decode: failed to bind a KV window.");
        return 0;
    }
    bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
    bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("GPT-OSS batched decode: failed to allocate backend buffer.");
            return 0;
        }
    }

    host_read_barrier();
    for (auto& u : upload_list)
        ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

    ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
    ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));

    ggml_status status = tsg::graph_compute_profiled(g_backend, graph, kGptOssBatchKernel);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("GPT-OSS batched decode: graph execution failed.");
        return 0;
    }

    finalize_compute_with_download(logits_out, logits_data,
                                   static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
    host_read_barrier();

    if (sampled_data != nullptr)
    {
        // Host argmax (first max wins) so the greedy fast path behaves the
        // same on backends without the in-graph argmax.
        const float* lg = static_cast<const float*>(logits_data);
        for (int s = 0; s < n_seqs; s++)
        {
            const float* row = lg + static_cast<std::size_t>(s) * vocab_size;
            int best = 0;
            for (int v = 1; v < vocab_size; v++)
                if (row[v] > row[best]) best = v;
            sampled_data[s] = best;
        }
    }

    for (int l = 0; l < num_layers; l++)
        for (int s = 0; s < n_seqs; s++)
        {
            const std::size_t i = static_cast<std::size_t>(l) * n_seqs + s;
            k_wins[i]->rows_valid = std::max<std::int64_t>(k_wins[i]->rows_valid, positions[s] + 1);
            v_wins[i]->rows_valid = std::max<std::int64_t>(v_wins[i]->rows_valid, positions[s] + 1);
        }

    clear_last_error();
    return 1;
}

// ---------------------------------------------------------------------------
// Arena path
// ---------------------------------------------------------------------------
static int gob_decode_batched_arena(
    const TSGgmlGptOssLayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,
    void** k_cache_arr, void** v_cache_arr,
    const int* cache_sizes,
    const int* positions,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    std::int32_t* sampled_data, int want_logits,
    bool& fell_through)
{
    fell_through = false;

    const int H = layers[0].hidden_size;
    const ggml_type kvType = static_cast<ggml_type>(layers[0].kv_cache_type);
    const int hd = layers[0].head_dim;
    const int kvH = layers[0].num_kv_heads;
    // The arena assumes one uniform KV geometry across layers (true for
    // GPT-OSS); anything else takes the legacy path.
    for (int l = 1; l < num_layers; l++)
    {
        if (layers[l].head_dim != hd || layers[l].num_kv_heads != kvH ||
            layers[l].kv_cache_type != layers[0].kv_cache_type)
        {
            fell_through = true;
            return 0;
        }
    }

    std::int64_t maxTotal = 0;
    for (int s = 0; s < n_seqs; s++)
        maxTotal = std::max<std::int64_t>(maxTotal, positions[s] + 1);

    const int n_slots = gob_slot_bucket(n_seqs);
    const void* sig_disc = layers[0].attn_norm_w;

    std::unique_lock<std::mutex> kv_lock(tsg_gptoss::kv_mutex());

    // ---- entry lookup ----
    GobPool& pool = gob_pool();
    int entry_idx = -1;
    for (int i = 0; i < kGobMaxEntries; i++)
    {
        GobEntry& e = pool.entries[i];
        if (e.valid && e.sig_disc == sig_disc && e.n_slots == n_slots &&
            e.num_layers == num_layers && e.H == H && e.vocab == vocab_size &&
            e.hd == hd && e.kvH == kvH && e.kv_type == kvType)
        {
            entry_idx = i;
            break;
        }
    }

    const bool needs_build = (entry_idx < 0) ||
        (pool.entries[entry_idx].cap < maxTotal);

    if (needs_build)
    {
        if (entry_idx < 0)
        {
            // Claim a pool slot: free first, else LRU (flushed before reuse).
            for (int i = 0; i < kGobMaxEntries; i++)
                if (!pool.entries[i].valid) { entry_idx = i; break; }
            if (entry_idx < 0)
            {
                entry_idx = 0;
                for (int i = 1; i < kGobMaxEntries; i++)
                    if (pool.used[i] < pool.used[entry_idx]) entry_idx = i;
            }
        }
        GobEntry& e = pool.entries[entry_idx];
        // Growing an existing entry doubles at minimum: a long generation
        // otherwise rebuilds (flush + re-copy every slot) each 1024 rows.
        const std::int64_t prev_cap = e.valid ? e.cap : 0;
        gob_flush_entry(e, entry_idx);
        e.release_graph();

        e.sig_disc = sig_disc;
        e.num_layers = num_layers;
        e.H = H;
        e.vocab = vocab_size;
        e.n_slots = n_slots;
        e.hd = hd;
        e.kvH = kvH;
        e.kv_type = kvType;
        e.cap = std::max(gob_cap_round(maxTotal), prev_cap * 2);
        e.rows_per_slot = static_cast<std::int64_t>(kvH) * e.cap;
        e.slots.assign(n_slots, GobSlot{});

        // ---- build the slot-stable graph ----
        const std::size_t ctx_size = 96 * 1024 * 1024;
        ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr)
        {
            set_last_error("GPT-OSS batched decode: failed to init arena ggml context.");
            fell_through = true;
            return 0;
        }
        e.ctx = ctx;

        auto abort_build = [&](const char* msg) {
            (void)msg;
            e.release_graph();
            fell_through = true;
            return 0;
        };

        // rows: slots [0, n_slots), plus one scratch slice for padded columns
        const std::int64_t arena_rows = e.rows_per_slot * (n_slots + 1);

        e.hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_slots);
        e.pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_slots);
        e.idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, static_cast<std::int64_t>(kvH) * n_slots);
        ggml_set_input(e.hidden_in);
        ggml_set_input(e.pos_in);
        ggml_set_input(e.idx_in);

        e.k_arena.resize(num_layers, nullptr);
        e.v_arena.resize(num_layers, nullptr);
        for (int l = 0; l < num_layers; l++)
        {
            e.k_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
            e.v_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
        }

        // one mask per distinct (swa, window) geometry
        auto mask_for = [&](const TSGgmlGptOssLayerDesc& d) -> ggml_tensor* {
            const int swa = (d.is_swa != 0) ? 1 : 0;
            const int wnd = swa ? d.sliding_window : 0;
            for (auto& m : e.masks)
                if (m.swa == swa && m.window == wnd) return m.t;
            ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, e.cap, 1, 1, n_slots);
            ggml_set_input(t);
            e.masks.push_back({swa, wnd, t});
            return t;
        };

        struct LayerW
        {
            ggml_tensor* attn_norm_w = nullptr;
            ggml_tensor* qkv_w = nullptr;
            ggml_tensor* qkv_b = nullptr;
            ggml_tensor* k_w = nullptr;
            ggml_tensor* k_b = nullptr;
            ggml_tensor* v_w = nullptr;
            ggml_tensor* v_b = nullptr;
            ggml_tensor* o_w = nullptr;
            ggml_tensor* o_b = nullptr;
            ggml_tensor* sinks = nullptr;
            ggml_tensor* post_attn_norm_w = nullptr;
            ggml_tensor* gate_inp_w = nullptr;
            ggml_tensor* gate_inp_b = nullptr;
            ggml_tensor* gate_exps = nullptr;
            ggml_tensor* gate_exps_b = nullptr;
            ggml_tensor* up_exps = nullptr;
            ggml_tensor* up_exps_b = nullptr;
            ggml_tensor* down_exps = nullptr;
            ggml_tensor* down_exps_b = nullptr;
        };
        std::vector<LayerW> lw(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerW& t = lw[l];
            const int qDim = d.num_heads * hd;
            const int kDim = kvH * hd;
            const int nExp = d.num_experts;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);
            if (d.qkv_b != nullptr) t.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkvDim);
            if (d.separate_qkv != 0)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                if (d.k_b != nullptr) t.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
                if (d.v_b != nullptr) t.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            }
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            if (d.o_b != nullptr) t.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.sinks != nullptr) t.sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.num_heads);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            if (d.gate_inp_b != nullptr) t.gate_inp_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp);
            t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ge_type), d.ge_ne0, d.ge_ne1, nExp);
            t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ue_type), d.ue_ne0, d.ue_ne1, nExp);
            t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            if (d.gate_exps_b != nullptr) t.gate_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ge_ne1, nExp);
            if (d.up_exps_b != nullptr) t.up_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ue_ne1, nExp);
            if (d.down_exps_b != nullptr) t.down_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.de_ne1, nExp);
        }

        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        const std::size_t row_bytes = ggml_row_size(kvType, hd);

        ggml_tensor* hidden = e.hidden_in;   // [H, n_slots]
        bool op_unsupported = false;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerW& t = lw[l];
            const int nH = d.num_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, d.eps), t.attn_norm_w);   // [H, N]

            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (d.separate_qkv != 0)
            {
                ggml_tensor* q_proj = ggml_mul_mat(ctx, t.qkv_w, normed);
                if (t.qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, t.qkv_b);
                ggml_tensor* k_proj = ggml_mul_mat(ctx, t.k_w, normed);
                if (t.k_b != nullptr) k_proj = ggml_add(ctx, k_proj, t.k_b);
                ggml_tensor* v_proj = ggml_mul_mat(ctx, t.v_w, normed);
                if (t.v_b != nullptr) v_proj = ggml_add(ctx, v_proj, t.v_b);
                q_raw = q_proj;
                k_raw = k_proj;
                v_raw = v_proj;
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);   // [qDim+2kDim, N]
                if (t.qkv_b != nullptr) qkv = ggml_add(ctx, qkv, t.qkv_b);
                q_raw = ggml_view_2d(ctx, qkv, qDim, n_slots, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, kDim, n_slots, qkv->nb[1],
                    static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, kDim, n_slots, qkv->nb[1],
                    static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), hd, nH, n_slots);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), hd, kvH, n_slots);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), hd, kvH, n_slots);

            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, e.pos_in, nullptr,
                d.rope_n_dims, /*mode=*/2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                1.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, e.pos_in, nullptr,
                d.rope_n_dims, 2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                1.0f, 1.0f, 32.0f, 1.0f);

            // ONE scatter per K/V per layer for the whole batch: the arena is
            // [hd, (n_slots+1)*kvH*cap] rows and idx_in carries
            // slot*kvH*cap + head*cap + pos for every (seq, head) — padded
            // columns aim at the scratch slice. Reading the attention view off
            // the set_rows RESULT makes the write→read edge a real src edge.
            ggml_tensor* k_rows = ggml_reshape_2d(ctx, k_rope, hd, static_cast<std::int64_t>(kvH) * n_slots);
            ggml_tensor* v_rows = ggml_reshape_2d(ctx, v_3d, hd, static_cast<std::int64_t>(kvH) * n_slots);
            ggml_tensor* k_set = ggml_set_rows(ctx, e.k_arena[l], k_rows, e.idx_in);
            ggml_tensor* v_set = ggml_set_rows(ctx, e.v_arena[l], v_rows, e.idx_in);
            if (l == 0 && !backend_supports_op(k_set))
                op_unsupported = true;

            ggml_tensor* k_view = ggml_view_4d(ctx, k_set, hd, e.cap, kvH, n_slots,
                row_bytes,
                static_cast<std::size_t>(e.cap) * row_bytes,
                static_cast<std::size_t>(e.rows_per_slot) * row_bytes,
                0);
            ggml_tensor* v_view = ggml_view_4d(ctx, v_set, hd, e.cap, kvH, n_slots,
                row_bytes,
                static_cast<std::size_t>(e.cap) * row_bytes,
                static_cast<std::size_t>(e.rows_per_slot) * row_bytes,
                0);

            ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_rope, hd, 1, nH, n_slots);
            ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_4d, k_view, v_view, mask_for(d), scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            if (t.sinks != nullptr)
                ggml_flash_attn_ext_add_sinks(fa, t.sinks);
            // Every layer: sliding-window and full layers read different lengths.
            if (!backend_supports_op(fa))
                op_unsupported = true;
            if (op_unsupported)
                return abort_build("unsupported op");

            ggml_tensor* attn_2d = ggml_reshape_2d(ctx, fa, qDim, n_slots);   // [qDim, N]

            ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_2d);   // [H, N]
            if (t.o_b != nullptr) o_mm = ggml_add(ctx, o_mm, t.o_b);
            ggml_tensor* ffn_inp = ggml_add(ctx, hidden, o_mm);

            // ===== MoE FFN (identical to the legacy path) =====
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, d.eps), t.post_attn_norm_w);

            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, moe_in);
            if (t.gate_inp_b != nullptr)
                router_logits = ggml_add(ctx, router_logits, t.gate_inp_b);
            ggml_tensor* sel = ggml_top_k(ctx, router_logits, nUsed);
            ggml_tensor* logits_r = ggml_reshape_3d(ctx, router_logits, 1, nExp, n_slots);
            ggml_tensor* w = ggml_get_rows(ctx, logits_r, sel);
            ggml_tensor* w_soft = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, nUsed, n_slots));
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_soft, 1, nUsed, n_slots);

            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, n_slots);
            ggml_tensor* gate = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);
            if (t.gate_exps_b != nullptr) gate = ggml_add_id(ctx, gate, t.gate_exps_b, sel);
            ggml_tensor* up = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
            if (t.up_exps_b != nullptr) up = ggml_add_id(ctx, up, t.up_exps_b, sel);
            ggml_tensor* act = ggml_swiglu_oai(ctx, gate, up, d.oai_alpha, d.oai_limit);
            ggml_tensor* down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);
            if (t.down_exps_b != nullptr) down = ggml_add_id(ctx, down, t.down_exps_b, sel);
            ggml_tensor* weighted = ggml_mul(ctx, down, w_final);

            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_slots, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, n_slots, weighted->nb[2],
                    static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            hidden = ggml_add(ctx, ffn_inp, moe_out);
        }

        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, layers[0].eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);
        e.logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_slots);
        ggml_tensor* out_cpy = ggml_cpy(ctx, logits, e.logits_out);
        ggml_set_output(out_cpy);

        // Device-side greedy sampling: one argmax column-reduce turns the
        // 25 MB/step logits download into 4 bytes/sequence when the sampler is
        // plain greedy (vLLM samples on-device for the same reason). The
        // logits path stays in the graph for steps that still want them.
        ggml_tensor* amax = ggml_argmax(ctx, logits);
        e.has_argmax = backend_supports_op(amax);
        if (e.has_argmax)
        {
            ggml_set_output(amax);
            e.sampled_out = amax;
        }

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 192 + 2048;
        e.graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(e.graph, out_cpy);
        if (e.has_argmax)
            ggml_build_forward_expand(e.graph, amax);

        // ---- bind weights ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;

        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes) {
            if (tgt == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload,
                                           GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                {
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerW& t = lw[l];
            const int nExp = d.num_experts;
            const int qDim = d.num_heads * hd;
            const int kDim = kvH * hd;
            const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes));
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes));
            bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes));
            bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes));
            bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.ge_bytes));
            bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.ue_bytes));
            bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.de_bytes));
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float));
            bind_or_mark(t.gate_inp_b, d.gate_inp_b, static_cast<std::size_t>(nExp) * sizeof(float));
            bind_or_mark(t.qkv_b, d.qkv_b, static_cast<std::size_t>(qkvDim) * sizeof(float));
            bind_or_mark(t.k_b, d.k_b, static_cast<std::size_t>(kDim) * sizeof(float));
            bind_or_mark(t.v_b, d.v_b, static_cast<std::size_t>(kDim) * sizeof(float));
            bind_or_mark(t.o_b, d.o_b, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(t.sinks, d.sinks, static_cast<std::size_t>(d.num_heads) * sizeof(float));
            bind_or_mark(t.gate_exps_b, d.gate_exps_b, static_cast<std::size_t>(d.ge_ne1) * nExp * sizeof(float));
            bind_or_mark(t.up_exps_b, d.up_exps_b, static_cast<std::size_t>(d.ue_ne1) * nExp * sizeof(float));
            bind_or_mark(t.down_exps_b, d.down_exps_b, static_cast<std::size_t>(d.de_ne1) * nExp * sizeof(float));
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes));
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float));

        // Everything unbound (inputs, arena, intermediates, logits) lands in
        // the entry's own buffer — nothing references the shared gallocr pool,
        // which is what lets this graph survive prefills.
        e.buffer = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, e.graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (e.buffer == nullptr)
            return abort_build("alloc failed");
        // Zero it all: fattn reads (masked) arena rows that were never
        // written; recycled VRAM decodes as NaN and NaN survives the -inf
        // mask. Same hazard the KV windows document.
        ggml_backend_buffer_clear(e.buffer, 0);

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        e.hidden_stage.assign(static_cast<std::size_t>(H) * n_slots, 0.0f);
        e.pos_stage.assign(n_slots, 0);
        e.idx_stage.assign(static_cast<std::size_t>(kvH) * n_slots, 0);
        e.logits_stage.assign(static_cast<std::size_t>(vocab_size) * n_slots, 0.0f);
        e.sampled_stage.assign(n_slots, 0);
        e.valid = true;
    }

    GobEntry& e = pool.entries[entry_idx];
    pool.used[entry_idx] = ++pool.clock;

    // ---- slot assignment + joins ----
    GobGuard guard;
    const std::size_t row_bytes = ggml_row_size(kvType, hd);
    std::vector<int> slot_of(n_seqs, -1);
    for (int i = 0; i < n_seqs; i++)
    {
        const void* key = k_cache_arr[i];   // layer-0 K host pointer
        for (int s = 0; s < e.n_slots; s++)
            if (e.slots[s].active && e.slots[s].key == key) { slot_of[i] = s; break; }
    }
    // A sequence crossing a concurrency bucket may hold a DIRTY slot in a
    // different bucket's entry; flush that mapping before joining here or the
    // join below seeds from host bytes that lack the arena-only rows.
    {
        GobRegistry& reg = gob_registry();
        for (int i = 0; i < n_seqs; i++)
        {
            if (slot_of[i] >= 0) continue;
            auto it = reg.find(k_cache_arr[i]);
            if (it != reg.end())
                gob_flush_and_drop_slot(
                    gob_pool().entries[it->second.first], it->second.first, it->second.second);
        }
    }
    for (int i = 0; i < n_seqs; i++)
    {
        if (slot_of[i] >= 0) continue;
        int s = -1;
        for (int j = 0; j < e.n_slots; j++)
            if (!e.slots[j].active) { s = j; break; }
        if (s < 0)
        {
            // evict the LRU slot not used by this call
            std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
            for (int j = 0; j < e.n_slots; j++)
            {
                bool in_call = false;
                for (int k = 0; k < n_seqs; k++)
                    if (slot_of[k] == j) { in_call = true; break; }
                if (!in_call && e.slots[j].last_used < best) { best = e.slots[j].last_used; s = j; }
            }
            if (s < 0)
            {
                set_last_error("GPT-OSS batched decode: no free arena slot.");
                fell_through = true;
                return 0;
            }
            gob_flush_and_drop_slot(e, entry_idx, s);
        }
        slot_of[i] = s;
        GobSlot& sl = e.slots[s];
        sl.active = true;
        sl.key = k_cache_arr[i];
        sl.cache_rows = cache_sizes[i];
        sl.len = 0;
        sl.clean = 0;
        sl.k_hosts.resize(num_layers);
        sl.v_hosts.resize(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            sl.k_hosts[l] = k_cache_arr[static_cast<std::size_t>(l) * n_seqs + i];
            sl.v_hosts[l] = v_cache_arr[static_cast<std::size_t>(l) * n_seqs + i];
        }
        GobRegistry& reg = gob_registry();
        for (int l = 0; l < num_layers; l++)
        {
            reg[sl.k_hosts[l]] = {entry_idx, s};
            reg[sl.v_hosts[l]] = {entry_idx, s};
        }
    }

    // Copy-ins: a slot whose arena rows do not cover [0, pos) pulls the newest
    // rows through the host mirror (window → host → arena). A slot that is
    // AHEAD (rollback) keeps its prefix — the rows are append-only history.
    for (int i = 0; i < n_seqs; i++)
    {
        const int s = slot_of[i];
        GobSlot& sl = e.slots[s];
        const std::int64_t pos = positions[i];
        if (sl.len != pos)
        {
            // (Re)joining at `pos`: from here on, rows >= pos are written to
            // the ARENA only. A window whose rows_valid exceeds pos holds
            // tail rows from a previous generation of this holder; clamp so
            // no later sync can resurrect them over arena-flushed data.
            for (int l = 0; l < num_layers; l++)
            {
                tsg_gptoss::KvWindow* kw = tsg_gptoss::kv_find(sl.k_hosts[l]);
                tsg_gptoss::KvWindow* vw = tsg_gptoss::kv_find(sl.v_hosts[l]);
                if (kw != nullptr) kw->rows_valid = std::min(kw->rows_valid, pos);
                if (vw != nullptr) vw->rows_valid = std::min(vw->rows_valid, pos);
            }
        }
        if (sl.len > pos)
        {
            sl.len = pos;
            sl.clean = std::min(sl.clean, pos);
        }
        if (sl.len < pos)
        {
            const std::int64_t from = sl.len;
            for (int l = 0; l < num_layers; l++)
            {
                for (int which = 0; which < 2; which++)
                {
                    const void* host = (which == 0) ? sl.k_hosts[l] : sl.v_hosts[l];
                    ggml_tensor* arena = (which == 0) ? e.k_arena[l] : e.v_arena[l];
                    tsg_gptoss::KvWindow* w = tsg_gptoss::kv_find(host);
                    if (w != nullptr && w->rows_valid > from)
                        kv_download(w, const_cast<void*>(host), sl.cache_rows,
                                    std::min<std::int64_t>(w->rows_valid, pos));
                    const std::size_t bytes = static_cast<std::size_t>(pos - from) * row_bytes;
                    for (int h = 0; h < kvH; h++)
                    {
                        const std::int64_t arow = static_cast<std::int64_t>(s) * e.rows_per_slot +
                                                  static_cast<std::int64_t>(h) * e.cap + from;
                        const char* src = static_cast<const char*>(host) +
                            (static_cast<std::int64_t>(h) * sl.cache_rows + from) * row_bytes;
                        ggml_backend_tensor_set(arena, src, static_cast<std::size_t>(arow) * row_bytes, bytes);
                    }
                }
            }
            sl.len = pos;
            sl.clean = pos;
        }
    }

    // ---- per-step inputs (slot order) ----
    std::vector<std::int64_t> pos_by_slot(e.n_slots, -1);
    std::fill(e.hidden_stage.begin(), e.hidden_stage.end(), 0.0f);
    std::fill(e.pos_stage.begin(), e.pos_stage.end(), 0);
    for (int s = 0; s < e.n_slots; s++)
        for (int h = 0; h < kvH; h++)
            e.idx_stage[static_cast<std::size_t>(s) * kvH + h] =
                static_cast<std::int64_t>(e.n_slots) * e.rows_per_slot +
                static_cast<std::int64_t>(h) * e.cap + s;   // scratch slice
    const float* hidden_src = static_cast<const float*>(hidden_data);
    for (int i = 0; i < n_seqs; i++)
    {
        const int s = slot_of[i];
        pos_by_slot[s] = positions[i];
        e.pos_stage[s] = positions[i];
        std::memcpy(&e.hidden_stage[static_cast<std::size_t>(s) * H],
                    hidden_src + static_cast<std::size_t>(i) * H,
                    static_cast<std::size_t>(H) * sizeof(float));
        for (int h = 0; h < kvH; h++)
            e.idx_stage[static_cast<std::size_t>(s) * kvH + h] =
                static_cast<std::int64_t>(s) * e.rows_per_slot +
                static_cast<std::int64_t>(h) * e.cap + positions[i];
    }

    host_read_barrier();
    decode_input_set_async(e.hidden_in, e.hidden_stage.data(),
                           e.hidden_stage.size() * sizeof(float));
    decode_input_set_async(e.pos_in, e.pos_stage.data(),
                           e.pos_stage.size() * sizeof(std::int32_t));
    decode_input_set_async(e.idx_in, e.idx_stage.data(),
                           e.idx_stage.size() * sizeof(std::int64_t));
    for (auto& m : e.masks)
    {
        gob_fill_arena_mask(e.mask_stage, e.cap, e.n_slots, pos_by_slot.data(),
                            m.swa != 0, m.window);
        decode_input_set_async(m.t, e.mask_stage.data(),
                               e.mask_stage.size() * sizeof(ggml_fp16_t));
    }

    ggml_status st = tsg::graph_compute_profiled(g_backend, e.graph, kGptOssBatchKernel);
    if (st != GGML_STATUS_SUCCESS)
    {
        set_last_error("GPT-OSS batched decode: arena graph execution failed.");
        gob_flush_entry(e, entry_idx);
        e.release_graph();
        e.slots.clear();
        return 0;
    }

    const bool need_logits = (want_logits != 0) ||
        (sampled_data != nullptr && !e.has_argmax);
    if (need_logits)
        finalize_compute_with_download(e.logits_out, e.logits_stage.data(),
                                       e.logits_stage.size() * sizeof(float));
    if (sampled_data != nullptr && e.has_argmax)
        finalize_compute_with_download(e.sampled_out, e.sampled_stage.data(),
                                       e.sampled_stage.size() * sizeof(std::int32_t));
    host_read_barrier();

    if (want_logits != 0 && logits_data != nullptr)
    {
        float* logits_dst = static_cast<float*>(logits_data);
        for (int i = 0; i < n_seqs; i++)
        {
            const int s = slot_of[i];
            std::memcpy(logits_dst + static_cast<std::size_t>(i) * vocab_size,
                        &e.logits_stage[static_cast<std::size_t>(s) * vocab_size],
                        static_cast<std::size_t>(vocab_size) * sizeof(float));
        }
    }
    if (sampled_data != nullptr)
    {
        for (int i = 0; i < n_seqs; i++)
        {
            const int s = slot_of[i];
            if (e.has_argmax)
            {
                sampled_data[i] = e.sampled_stage[s];
            }
            else
            {
                const float* row = &e.logits_stage[static_cast<std::size_t>(s) * vocab_size];
                int best = 0;
                for (int v = 1; v < vocab_size; v++)
                    if (row[v] > row[best]) best = v;
                sampled_data[i] = best;
            }
        }
    }

    for (int i = 0; i < n_seqs; i++)
    {
        GobSlot& sl = e.slots[slot_of[i]];
        sl.len = positions[i] + 1;    // this step's row now lives in the arena
        sl.last_used = ++pool.clock;
    }

    clear_last_error();
    return 1;
}

TSG_EXPORT int TSGgml_GptOssModelDecodeBatched(
    const TSGgmlGptOssLayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,                       // [hidden, n_seqs] F32, column s = seq s
    void** k_cache_arr, void** v_cache_arr,  // [layer * n_seqs + seq] HOST cache pointers
    const int* cache_sizes,                  // [n_seqs] rows in each holder's host cache
    const int* positions,                    // [n_seqs] current length of each sequence
    void* logits_data, int vocab_size,       // out [vocab, n_seqs] (see want_logits)
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    // Greedy fast path: when non-null receives argmax(logits) per sequence.
    // want_logits == 0 additionally skips the [vocab, n] download+scatter on
    // the arena path (the caller must then pass sampled_data). The legacy and
    // no-argmax fallbacks still fill both correctly.
    std::int32_t* sampled_data, int want_logits)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || n_seqs < 2 || hidden_data == nullptr ||
            k_cache_arr == nullptr || v_cache_arr == nullptr || cache_sizes == nullptr || positions == nullptr)
        {
            set_last_error("GPT-OSS batched decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGptOssLayerDesc)))
        {
            set_last_error("GPT-OSS batched decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }
        if (lm_head_data == nullptr || final_norm_data == nullptr || vocab_size <= 0 ||
            (logits_data == nullptr && sampled_data == nullptr) ||
            (want_logits != 0 && logits_data == nullptr))
        {
            set_last_error("GPT-OSS batched decode: folded lm_head required.");
            return 0;
        }

        const int kvType = layers[0].kv_cache_type;
        if (kvType != GGML_TYPE_F32 && kvType != GGML_TYPE_F16)
        {
            set_last_error("GPT-OSS batched decode: only F32/F16 KV caches are supported.");
            return 0;
        }
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].cpu_moe != 0)
            {
                set_last_error("GPT-OSS batched decode: MoE CPU offload not supported (v1).");
                return 0;
            }
        }
        for (int s = 0; s < n_seqs; s++)
        {
            if (positions[s] + 1 > cache_sizes[s])
            {
                set_last_error("GPT-OSS batched decode: sequence exceeds its cache rows (no-wrap v1).");
                return 0;
            }
        }

        static const bool gob_arena = []{
            const char* e = std::getenv("TS_GPTOSS_BATCHED_ARENA");
            return e == nullptr || e[0] != '0';
        }();
        static const bool gob_persist = []{
            const char* e = std::getenv("TS_GPTOSS_FD_PERSIST");
            return e == nullptr || e[0] != '0';
        }();
        const bool can_persist = gob_persist && gob_arena &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN);

        if (can_persist)
        {
            bool fell_through = false;
            int rc = gob_decode_batched_arena(layers, num_layers, n_seqs, hidden_data,
                k_cache_arr, v_cache_arr, cache_sizes, positions, logits_data, vocab_size,
                lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
                final_norm_data, sampled_data, want_logits, fell_through);
            if (!fell_through)
                return rc;
        }

        if (logits_data == nullptr)
        {
            set_last_error("GPT-OSS batched decode: legacy path requires a logits buffer.");
            return 0;
        }
        return gob_decode_batched_legacy(layers, num_layers, n_seqs, hidden_data,
            k_cache_arr, v_cache_arr, cache_sizes, positions, logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data, sampled_data);
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in GPT-OSS batched decode."); return 0; }
}
