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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace tsg;

// Defined in ggml_ops_qwen35_decode.cpp / ggml_ops_qwen35_verify.cpp: drop the
// persistent solo decode graphs keyed on a holder's first-attention K pointer,
// and the verify graph cache. The arena flush frees the resident cacheable
// copies those captured graphs bake in; without these drops a later solo
// replay dereferences freed device memory (the documented persistent-graph
// use-after-free class).
extern void tsg_q35_drop_decode_graphs_for_kv(const void* k_cache0);
extern "C" void TSGgml_Qwen35ResetVerifyCacheForHostPointer(const void* host_ptr);

// ============================================================================
// Qwen3.5/3.8 SLOT-STABLE ARENA token-batched decode: N concurrent sequences,
// one token each, in ONE ggml graph — the port of the GPT-OSS arena design
// (ggml_ops_gptoss_batched.cpp) to the hybrid GDN + full-attention family.
//
// Without this, N>=2 requests round-robin N solo whole-model graphs per step:
// N full weight sweeps per token, so a 27B Q8 model serves ~35 tok/s aggregate
// no matter the concurrency while vLLM batches to 650+. The existing
// TSGgml_Qwen35ModelDecodeBatched (BatchedPaged path) is not the answer: it
// re-uploads and downloads ~150 MB of GDN conv/delta state through the host
// EVERY step (which also blocks CUDA-graph capture) and its paged pools keep a
// host mirror that nothing coherently maintains.
//
// Design (mirrors gptoss arena; per-family differences noted):
//   * ATTENTION layers (every 4th on Qwen3.8): per-layer persistent K/V arena
//     [head_dim, cap, kv_heads, n_slots] (head-plane-major, IDENTICAL to
//     this family's host cache layout [kv_heads, cap, head_dim]), followed by
//     one discard row per (slot, head) for padded SET_ROWS writes. ONE
//     ggml_set_rows per K/V per layer (row = slot*kvH*cap + head*cap + pos),
//     a shared causal mask [cap, 1, 1, n_slots], and CUDA per-slot flash_attn_ext
//     nodes in the SAME graph (Metal retains its batched flash-attention).
//     All QKV/output projections and FFNs read their weights once for the batch.
//   * GDN layers (the other 3/4): per-slot recurrent-state ARENAS —
//     conv_arena [convDim, conv_dim, n_slots] and delta_arena
//     [head_k, head_v, v_heads, n_slots] — updated IN-GRAPH by the same
//     per-slot ssm_conv + ggml_gated_delta_net + cpy-back chain the solo
//     kernel uses. The state never crosses the bus between steps; that is
//     exactly the piece whose absence forced the old batched path's per-step
//     host round trip. Projections/FFN/MoE/head run batched over all N.
//   * CUDA attention uses each holder's solo decode window. The graph also
//     keys on these windows in stable-slot order, rebuilding at a window
//     boundary or when a replacement holder needs a different window.
//     In-graph argmax serves the greedy fast path.
//
// COHERENCE: while a sequence occupies a slot, its newest KV rows and GDN
// state exist ONLY in the arenas. The rest of the engine reads this family's
// caches through the resident cacheable-buffer copies keyed on the HOST
// pointers (solo decode/verify bind them with USAGE_COMPUTE), with lazy
// host-mirror syncs on top. The contract here:
//   * JOIN seeds arena slices from the resident device copy when one exists
//     (it is the truth after solo decode), else from the host bytes (the truth
//     after prefill / a fresh holder). The managed caller guarantees the GDN
//     conv scratch is in ggml [time,channel] layout for host-authoritative
//     holders (it converts the ring once before the first join).
//   * FLUSH (slot retirement) writes arena KV rows [clean,len) and the full
//     conv/delta state back to the HOST bytes, then invalidates the resident
//     copies for those pointers, so the next solo bind re-uploads the current
//     state. Flush fires from the on_external_touch hook wired into the solo
//     decode and verify kernel entries, from LRU slot eviction, and from the
//     pool reset. on_drop (InvalidateHostBuffer chain, holder dispose)
//     discards without flushing — the host was rewritten behind us.
//
// HOLDER-BOUND MODE (CUDA default; TS_QAB_BOUND=0 disables): attention and
// recurrence already run per slot on CUDA, so slot s binds its holder's own
// resident K/V, conv and delta copies (the solo decode's keys, sizes and usage)
// instead of arena planes. One graph per (slot order, capacities, windows, keys);
// replay first checks every key still maps to the device copy the graph baked.
// There is no join, flush or seed: a step leaves each holder exactly as a solo
// fused decode step would, and no KV is duplicated.
//
// Gates (decline -> engine round-robins, correctness never at risk):
// CUDA or Metal backend, no TP, fused-graph KV cache types, no MoE CPU offload,
// folded lm_head, uniform attention geometry across layers, no-wrap positions.
// ============================================================================
namespace
{
    constexpr int kQabMaxEntries = 4;
    constexpr const char* kQabKernel = "Qwen3.5 arena batched decode";

    int qab_slot_bucket(int n)
    {
        int b = 2;
        while (b < n) b *= 2;
        return b;
    }

    std::int64_t qab_cap_round(std::int64_t rows)
    {
        const std::int64_t q = g_backend_type == BACKEND_TYPE_CUDA ? 256 : 1024;
        return ((rows + q - 1) / q) * q;
    }

    struct QabSlot
    {
        bool active = false;
        const void* key = nullptr;              // first attention layer's K host pointer
        std::vector<const void*> k_hosts;       // per attention layer
        std::vector<const void*> v_hosts;
        std::vector<const void*> conv_hosts;    // per GDN layer (scratch, ggml layout)
        std::vector<const void*> delta_hosts;   // per GDN layer
        int cache_rows = 0;
        std::int64_t len = 0;                   // arena KV rows [0, len) valid
        std::int64_t clean = 0;                 // KV rows [clean, len) not yet in host
        bool state_seeded = false;              // conv/delta arena slice holds this sequence's state
        bool state_dirty = false;               // GDN arena state newer than host
        std::uint64_t last_used = 0;
    };

    struct QabEntry
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* token_in = nullptr;        // I32 [n_slots]
        ggml_tensor* hidden_in = nullptr;       // optional F32 [H, n_slots], host-gathered embeddings
        ggml_tensor* pos_in = nullptr;          // I32 [n_slots]
        ggml_tensor* idx_in = nullptr;          // I64 [kvH * n_slots] set_rows targets
        ggml_tensor* attn_mask = nullptr;       // F16 [cap, 1, 1, n_slots]
        ggml_tensor* logits_out = nullptr;      // [vocab, n_slots]
        ggml_tensor* sampled_out = nullptr;     // I32 [n_slots]
        bool has_argmax = false;
        std::vector<ggml_tensor*> k_arena;      // per layer (null on GDN layers)
        std::vector<ggml_tensor*> v_arena;
        std::vector<ggml_tensor*> conv_arena;   // per layer (null on attention layers)
        std::vector<ggml_tensor*> delta_arena;
        const void* sig_disc = nullptr;
        int num_layers = 0, H = 0, vocab = 0, n_slots = 0;
        int hd = 0, kvH = 0;
        int conv_dim = 0, convDim = 0, head_k = 0, head_v = 0, v_heads = 0;
        std::int64_t cap = 0;
        std::int64_t rows_per_slot = 0;         // kvH * cap
        ggml_type kv_type = GGML_TYPE_F32;
        bool hidden_input = false;
        std::vector<QabSlot> slots;
        std::vector<int> attn_windows;         // CUDA: effective KV lengths in stable-slot order

        // Holder-bound mode (CUDA): per-(layer, slot) tensors bound to each holder's
        // own resident device copies instead of arena-owned planes. Indexed
        // [layer * n_slots + slot]; null where the layer kind does not apply.
        bool bound = false;
        std::vector<std::uintptr_t> bound_sig;  // slot count, rows, windows and every bound host key
        std::vector<int> bound_cache_rows;      // per slot
        std::int64_t mask_len = 0;              // attention mask rows (max window when bound)
        std::vector<ggml_tensor*> k_bound, v_bound, conv_bound, delta_bound;
        struct BoundKey { const void* host; ggml_backend_buffer_t buffer; void* addr; std::size_t bytes; };
        std::vector<BoundKey> bound_keys;       // replay is valid while every key keeps this device copy

        std::vector<std::int32_t> tok_stage;
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
            token_in = hidden_in = pos_in = idx_in = attn_mask = logits_out = sampled_out = nullptr;
            has_argmax = false;
            k_arena.clear(); v_arena.clear(); conv_arena.clear(); delta_arena.clear();
            k_bound.clear(); v_bound.clear(); conv_bound.clear(); delta_bound.clear();
            bound_keys.clear();
            bound_sig.clear();
            bound_cache_rows.clear();
            bound = false;
            mask_len = 0;
            valid = false;
        }

        // A bound graph bakes the device addresses of its holders' resident copies.
        // Any free or replacement of one (demotion, reseed, resize) invalidates it.
        bool bound_bindings_current() const
        {
            for (const BoundKey& k : bound_keys)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (!try_peek_cached_device_copy(k.host, k.bytes, buf, addr) ||
                    buf != k.buffer || addr != k.addr)
                    return false;
            }
            return true;
        }
    };

    // Holder-bound arena decode: CUDA attention already runs per slot, so each
    // slot can read and write its holder's own resident K/V and recurrent-state
    // copies (the buffers the solo decode and verify graphs bind). That removes
    // the arena's duplicate KV planes, the join copy through host memory, the
    // flush on retirement and the full re-seed at every window rebuild; the
    // holder's state after a step is exactly the solo decode's. Env
    // TS_QAB_BOUND=0 restores the arena-owned layout.
    bool qab_bound_enabled()
    {
        static const bool enabled = [] {
            const char* v = std::getenv("TS_QAB_BOUND");
            return v == nullptr || std::strcmp(v, "0") != 0;
        }();
        return enabled && g_backend_type == BACKEND_TYPE_CUDA;
    }

    struct QabPool
    {
        QabEntry entries[kQabMaxEntries];
        std::uint64_t used[kQabMaxEntries] = {};
        std::uint64_t clock = 0;
    };
    QabPool g_qab_pools[tsg::TSG_MAX_DEVICES];
    QabPool& qab_pool() { return g_qab_pools[tsg::g_active_rank]; }

    // host pointer (any registered cache/state pointer of a slot) -> (entry, slot)
    using QabRegistry = std::unordered_map<const void*, std::pair<int, int>>;
    QabRegistry g_qab_registries[tsg::TSG_MAX_DEVICES];
    QabRegistry& qab_registry() { return g_qab_registries[tsg::g_active_rank]; }

    std::mutex& qab_mutex()
    {
        static std::mutex m;
        return m;
    }

    thread_local bool g_qab_guard = false;
    struct QabGuard
    {
        bool prev;
        QabGuard() : prev(g_qab_guard) { g_qab_guard = true; }
        ~QabGuard() { g_qab_guard = prev; }
    };

    void qab_unregister_slot(QabSlot& sl)
    {
        QabRegistry& reg = qab_registry();
        for (const void* p : sl.k_hosts) reg.erase(p);
        for (const void* p : sl.v_hosts) reg.erase(p);
        for (const void* p : sl.conv_hosts) reg.erase(p);
        for (const void* p : sl.delta_hosts) reg.erase(p);
        sl = QabSlot{};
    }

    // Flush a slot's arena-held truth back to the HOST bytes and retire it:
    //   * KV rows [clean, len) per attention layer (host layout matches the
    //     arena head-plane-major layout with the holder's own cache_rows pitch),
    //   * the full GDN conv scratch + delta state per GDN layer when dirty.
    // Then invalidate the resident cacheable-buffer copies of every pointer we
    // rewrote, so the next solo/verify bind re-uploads the current bytes
    // (their device copies predate the arena's rows). qab_mutex held.
    void qab_flush_and_drop_slot(QabEntry& e, int slot_idx)
    {
        QabSlot& sl = e.slots[slot_idx];
        if (!sl.active) return;
        QabGuard guard;
        bool freed_any = false;
        if (g_backend != nullptr && e.valid)
        {
            const std::size_t row_bytes = ggml_row_size(e.kv_type, e.hd);
            if (sl.len > sl.clean)
            {
                const std::size_t bytes = static_cast<std::size_t>(sl.len - sl.clean) * row_bytes;
                for (int l = 0, al = 0; l < e.num_layers; l++)
                {
                    if (e.k_arena[l] == nullptr) continue;
                    for (int which = 0; which < 2; which++)
                    {
                        ggml_tensor* arena = (which == 0) ? e.k_arena[l] : e.v_arena[l];
                        const void* host = (which == 0) ? sl.k_hosts[al] : sl.v_hosts[al];
                        if (arena == nullptr || host == nullptr) continue;
                        for (int h = 0; h < e.kvH; h++)
                        {
                            const std::int64_t arow = static_cast<std::int64_t>(slot_idx) * e.rows_per_slot +
                                                      static_cast<std::int64_t>(h) * e.cap + sl.clean;
                            char* dst = const_cast<char*>(static_cast<const char*>(host)) +
                                (static_cast<std::int64_t>(h) * sl.cache_rows + sl.clean) * row_bytes;
                            ggml_backend_tensor_get(arena, dst, static_cast<std::size_t>(arow) * row_bytes, bytes);
                        }
                        invalidate_cached_buffer(const_cast<void*>(host));
                        freed_any = true;
                    }
                    (void)al; al++;
                }
            }
            if (sl.state_dirty)
            {
                const std::size_t convBytes = static_cast<std::size_t>(e.convDim) * e.conv_dim * sizeof(float);
                const std::size_t deltaBytes = static_cast<std::size_t>(e.head_k) * e.head_v * e.v_heads * sizeof(float);
                for (int l = 0, gl = 0; l < e.num_layers; l++)
                {
                    if (e.conv_arena[l] == nullptr) continue;
                    const void* conv_host = sl.conv_hosts[gl];
                    const void* delta_host = sl.delta_hosts[gl];
                    if (conv_host != nullptr)
                    {
                        ggml_backend_tensor_get(e.conv_arena[l],
                            const_cast<void*>(conv_host),
                            static_cast<std::size_t>(slot_idx) * convBytes, convBytes);
                        invalidate_cached_buffer(const_cast<void*>(conv_host));
                        freed_any = true;
                    }
                    if (delta_host != nullptr)
                    {
                        ggml_backend_tensor_get(e.delta_arena[l],
                            const_cast<void*>(delta_host),
                            static_cast<std::size_t>(slot_idx) * deltaBytes, deltaBytes);
                        invalidate_cached_buffer(const_cast<void*>(delta_host));
                        freed_any = true;
                    }
                    gl++;
                }
            }
        }
        if (freed_any)
        {
            // Captured solo decode / verify graphs bake the just-freed resident
            // buffers into their nodes; drop them so the next solo step rebuilds
            // against the re-uploaded copies instead of replaying freed memory.
            if (!sl.k_hosts.empty())
                tsg_q35_drop_decode_graphs_for_kv(sl.k_hosts[0]);
            // Retire only graphs that actually pin this slot's model buffers.
            // A process-global reset can destroy another live Qwen35 model's
            // deferred snapshot between verify and commit.
            const void* verify_host = !sl.k_hosts.empty() ? sl.k_hosts[0]
                : (!sl.conv_hosts.empty() ? sl.conv_hosts[0]
                    : (!sl.delta_hosts.empty() ? sl.delta_hosts[0] : nullptr));
            TSGgml_Qwen35ResetVerifyCacheForHostPointer(verify_host);
        }
        qab_unregister_slot(sl);
    }

    void qab_flush_entry(QabEntry& e)
    {
        for (int s = 0; s < static_cast<int>(e.slots.size()); s++)
            qab_flush_and_drop_slot(e, s);
    }

    // Causal mask fill over [cap, n_slots]: column s zero on [0, pos_s], -inf
    // beyond (span fills). Padded/absent columns attend row 0 only.
    void qab_fill_mask(std::vector<ggml_fp16_t>& mask, std::int64_t cap, int n_slots,
                       const std::int64_t* pos_by_slot)
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
            const std::int64_t hi = std::min<std::int64_t>(pos, cap - 1);
            std::fill(col, col + hi + 1, zero_val);
            std::fill(col + hi + 1, col + cap, neg_inf);
        }
    }

    // Read `bytes` at byte `offset` of `host`'s cache bytes into `dst`, from
    // the device-resident cacheable copy when a CURRENT one exists (post solo
    // decode the resident copy is the truth and the host mirror lags), else
    // from the host bytes themselves (post prefill / fresh holder). The probe
    // never allocates or evicts. A resident read is also MIRRORED back into
    // the host bytes so that, from the join onward, the host mirror is current
    // up to the joined position — the flush later appends only [clean, len).
    // `force_host` skips the resident copy (managed says host is newer).
    bool qab_read_source(const void* host, std::size_t offset, std::size_t bytes,
                         std::size_t total_bytes, void* dst, bool force_host)
    {
        if (host == nullptr) return false;
        if (!force_host)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* base = nullptr;
            if (try_peek_cached_device_copy(host, total_bytes, buf, base))
            {
                ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
                ggml_context* tmp = ggml_init(ip);
                if (tmp != nullptr)
                {
                    ggml_tensor* t = ggml_new_tensor_1d(tmp, GGML_TYPE_I8,
                        static_cast<std::int64_t>(total_bytes));
                    bool ok = t != nullptr &&
                        ggml_backend_tensor_alloc(buf, t, base) == GGML_STATUS_SUCCESS;
                    if (ok)
                        ggml_backend_tensor_get(t, dst, offset, bytes);
                    ggml_free(tmp);
                    if (ok)
                    {
                        std::memcpy(const_cast<char*>(static_cast<const char*>(host)) + offset, dst, bytes);
                        return true;
                    }
                }
            }
        }
        std::memcpy(dst, static_cast<const char*>(host) + offset, bytes);
        return true;
    }
}

// ---------------------------------------------------------------------------
// Coherence hooks (see the header comment). All take qab_mutex internally so
// call sites in other translation units stay one-liners.
// ---------------------------------------------------------------------------
namespace tsg_q35arena
{
    void on_external_touch(const void* host_ptr)
    {
        if (g_qab_guard || host_ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(qab_mutex());
        QabRegistry& reg = qab_registry();
        auto it = reg.find(host_ptr);
        if (it == reg.end()) return;
        qab_flush_and_drop_slot(qab_pool().entries[it->second.first], it->second.second);
    }

    void on_drop(const void* host_ptr)
    {
        if (g_qab_guard || host_ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(qab_mutex());
        QabRegistry& reg = qab_registry();
        auto it = reg.find(host_ptr);
        if (it == reg.end()) return;
        qab_unregister_slot(qab_pool().entries[it->second.first].slots[it->second.second]);
    }

    void on_drop_all()
    {
        if (g_qab_guard) return;
        std::lock_guard<std::mutex> lock(qab_mutex());
        for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
        {
            for (auto& e : g_qab_pools[r].entries)
                for (auto& sl : e.slots)
                    sl = QabSlot{};
            g_qab_registries[r].clear();
        }
    }
}

// Flush-and-retire the arena slot (if any) registered for `host_ptr` — the
// managed-callable form of on_external_touch, for paths that read or replace a
// holder's caches/state outside the hooked native kernels (cache growth, host
// syncs, snapshot extraction, residency release).
TSG_EXPORT void TSGgml_Qwen35ArenaFlushHostPointer(void* host_ptr)
{
    tsg_q35arena::on_external_touch(host_ptr);
}

// Retire one arena slot without copying its device-resident state back to the
// host. Completed requests deliberately use this path before their holder is
// pooled: the host buffers will be reset for a different request, so flushing
// the old request over those buffers later would corrupt the new sequence.
// Unlike TSGgml_InvalidateHostBuffer this does not evict the normal resident
// copy or reset persistent graphs belonging to unrelated active holders.
TSG_EXPORT void TSGgml_Qwen35ArenaDiscardHostPointer(void* host_ptr)
{
    tsg_q35arena::on_drop(host_ptr);
}

// 1 when batched decode binds each holder's own resident cache copies (see
// qab_bound_enabled): a successful step then leaves holders exactly as a solo
// fused decode does, with no arena slot to flush.
TSG_EXPORT int TSGgml_Qwen35ArenaUsesHolderBuffers()
{
    if (!ensure_backend())
        return 0;
    return qab_bound_enabled() ? 1 : 0;
}

// Drop all arena state. Dirty slots flush to their host bytes first on the
// active rank. Like the GPT-OSS arena pool, this is NOT chained into the solo
// TSGgml_Qwen35ResetDecodeCache (the solo pool churns per holder swap and
// state event; this pool's survival across those is the point) — teardown
// paths in ggml_ops_core.cpp and the managed dispose call it explicitly.
TSG_EXPORT void TSGgml_Qwen35ArenaResetBatchedDecodeCache()
{
    std::lock_guard<std::mutex> lock(qab_mutex());
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
    {
        for (int i = 0; i < kQabMaxEntries; i++)
        {
            QabEntry& e = g_qab_pools[r].entries[i];
            if (r == tsg::g_active_rank)
                qab_flush_entry(e);
            else
                for (auto& sl : e.slots) sl = QabSlot{};
            e.release_graph();
            e.slots.clear();
        }
        g_qab_registries[r].clear();
    }
}

// ============================================================================
// The batched arena step.
//   layers: slot-invariant weight/geometry descriptors (kv/state pointer
//           fields in them are IGNORED — the arrays below carry per-slot ones).
//   k/v_cache_arr:   [attn_layer_index * n_seqs + s] host K/V cache pointers.
//   conv/delta_arr:  [gdn_layer_index * n_seqs + s] host GDN state pointers
//                    (conv scratch in ggml [time, channel] layout).
//   gdn_host_auth:   [n_seqs] non-zero when the holder's GDN state is
//                    host-authoritative (fresh/post-prefill; managed already
//                    converted the ring into the scratch layout). Zero means
//                    the resident device copy is the truth (post solo decode).
//   token_ids/positions: [n_seqs]; embedding happens in-graph (get_rows).
//                    positions are KV indices (the write row and the mask length).
//   rope_positions:  [n_seqs] RoPE positions, or null for positions. They differ
//                    for a Qwen-VL sequence after an image, whose span holds H*W
//                    rows but only max(H, W) M-RoPE positions.
//   logits_data [vocab, n_seqs] filled when want_logits; sampled_data [n_seqs]
//   filled when non-null (in-graph argmax, first-max ties).
// Returns 1 on success, 0 when declining before graph execution (safe for the
// managed serial fallback), and -1 when graph execution failed after recurrent
// state may have been partially advanced (the affected requests must fail).
// ============================================================================
static int qab_decode_batched(
    const TSGgmlQwen35LayerDesc* layers, int num_layers, int n_seqs,
    const std::int32_t* token_ids, const std::int32_t* positions,
    const std::int32_t* rope_positions,
    void** k_cache_arr, void** v_cache_arr,
    void** conv_state_arr, void** delta_state_arr,
    const std::int32_t* gdn_host_auth,
    const std::int32_t* cache_sizes,
    int num_heads, int num_kv_heads, int head_dim,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    const void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1, std::int64_t token_embd_bytes,
    std::int32_t* sampled_data, int want_logits, const float* embedding_rows)
{
    bool compute_started = false;
    try
    {
        if (!ensure_backend())
            return 0;
        if (g_backend_type != BACKEND_TYPE_CUDA && g_backend_type != BACKEND_TYPE_METAL)
        {
            set_last_error("Qwen3.5 arena batched decode: requires CUDA or Metal.");
            return 0;
        }
        if (layers == nullptr || num_layers <= 0 || n_seqs < 2 ||
            token_ids == nullptr || positions == nullptr ||
            k_cache_arr == nullptr || v_cache_arr == nullptr ||
            conv_state_arr == nullptr || delta_state_arr == nullptr ||
            gdn_host_auth == nullptr || cache_sizes == nullptr)
        {
            set_last_error("Qwen3.5 arena batched decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 arena batched decode: descriptor size mismatch.");
            return 0;
        }
        const bool hidden_input = embedding_rows != nullptr;
        if (lm_head_data == nullptr || final_norm_data == nullptr || vocab_size <= 0 ||
            (!hidden_input && token_embd_data == nullptr) ||
            token_embd_ne0 <= 0 || token_embd_ne0 > std::numeric_limits<int>::max() ||
            token_embd_ne1 <= 0 ||
            (logits_data == nullptr && sampled_data == nullptr) ||
            (want_logits != 0 && logits_data == nullptr))
        {
            set_last_error("Qwen3.5 arena batched decode: folded lm_head + token embedding required.");
            return 0;
        }
        // F32/F16 plus the two block-quantized cache types the solo fused graphs
        // already run. Nothing in this file is dtype-specific: the arenas are created
        // with static_cast<ggml_type>(kv_cache_type) and every join, flush and view
        // strides by ggml_row_size(kv_type, head_dim), so a quantized row is simply a
        // shorter row (hd=256 at Q8_0 is 8 blocks = 272 bytes) and every offset stays
        // a whole multiple of it. What actually decides the answer is whether the
        // BACKEND has kernels for the arena's set_rows and 4-D flash_attn shapes at
        // this type - and that is checked below on the built nodes, where a miss
        // aborts the build and the caller falls back to round-robin serving.
        if (kv_cache_type != GGML_TYPE_F32 && kv_cache_type != GGML_TYPE_F16 &&
            kv_cache_type != GGML_TYPE_Q8_0 && kv_cache_type != GGML_TYPE_Q4_0)
        {
            set_last_error("Qwen3.5 arena batched decode: F32/F16/Q8_0/Q4_0 KV cache only.");
            return 0;
        }
        if (head_k_dim != head_v_dim || conv_kernel <= 1)
        {
            set_last_error("Qwen3.5 arena batched decode: unsupported GDN geometry.");
            return 0;
        }
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_moe != 0 && layers[l].proj_scales != nullptr)
            {
                set_last_error("Qwen3.5 arena batched decode: MoE sidecar scales are unsupported.");
                return 0;
            }
            if (layers[l].is_moe != 0 && layers[l].cpu_moe != 0)
            {
                set_last_error("Qwen3.5 arena batched decode: MoE CPU offload not supported (v1).");
                return 0;
            }
        }
        if (!hidden_input)
        {
            if (token_embd_type < 0 || token_embd_type >= GGML_TYPE_COUNT)
            {
                set_last_error("Qwen3.5 arena batched decode: bad token embedding type.");
                return 0;
            }
            const ggml_type emb_type = static_cast<ggml_type>(token_embd_type);
            const std::int64_t bs = ggml_blck_size(emb_type);
            if (bs <= 0 || token_embd_ne0 % bs != 0)
            {
                set_last_error("Qwen3.5 arena batched decode: bad token embedding type.");
                return 0;
            }
        }

        std::int64_t maxTotal = 0;
        int attn_layers = 0, gdn_layers = 0;
        for (int l = 0; l < num_layers; l++)
            (layers[l].is_recurrent != 0 ? gdn_layers : attn_layers)++;
        if (attn_layers == 0)
        {
            set_last_error("Qwen3.5 arena batched decode: no attention layers.");
            return 0;
        }
        for (int s = 0; s < n_seqs; s++)
        {
            if (token_ids[s] < 0 || token_ids[s] >= token_embd_ne1)
            {
                set_last_error("Qwen3.5 arena batched decode: token id out of range.");
                return 0;
            }
            if (positions[s] < 0 || cache_sizes[s] <= 0 ||
                static_cast<std::int64_t>(positions[s]) + 1 > cache_sizes[s])
            {
                set_last_error("Qwen3.5 arena batched decode: sequence exceeds its cache rows (no-wrap).");
                return 0;
            }
            maxTotal = std::max<std::int64_t>(maxTotal, static_cast<std::int64_t>(positions[s]) + 1);
        }

        const int H = static_cast<int>(token_embd_ne0);
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        const int hd = head_dim;
        const int kvH = num_kv_heads;
        const int nH = num_heads;
        const int qDim = nH * hd;
        const int qFullDim = qDim * 2;                 // Q + per-head gate interleaved
        const int kDim = kvH * hd;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const std::size_t convBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        const bool bound = qab_bound_enabled();
        const int n_slots_req = bound ? n_seqs : qab_slot_bucket(n_seqs);
        const void* sig_disc = layers[0].attn_norm_w;

        std::lock_guard<std::mutex> qlock(qab_mutex());
        QabGuard guard;

        // ---- entry lookup / build ----
        QabPool& pool = qab_pool();
        // A model uses only one batch-width bucket at a time. Keeping its old
        // 2-lane arena while allocating a 4-lane arena duplicates every KV and
        // recurrent-state buffer and can exhaust the device before allocation
        // falls back. Publish dirty slots to their holders first, including
        // temporarily absent sequences that may return on the next batch.
        // Other model instances retain their own independently keyed entries.
        for (auto& previous : pool.entries)
        {
            if (!bound && previous.valid && previous.sig_disc == sig_disc && previous.n_slots != n_slots_req)
            {
                qab_flush_entry(previous);
                previous.release_graph();
                previous.slots.clear();
            }
        }
        // Bound entries are keyed by everything their graph bakes in: the slot
        // order, each holder's capacity and attention window, and every host key.
        std::vector<std::uintptr_t> bound_sig;
        std::vector<int> bound_windows;
        if (bound)
        {
            bound_windows.resize(n_seqs);
            bound_sig.reserve(1 + static_cast<std::size_t>(n_seqs) * (2 + 2 * attn_layers + 2 * gdn_layers));
            bound_sig.push_back(static_cast<std::uintptr_t>(n_seqs));
            for (int i = 0; i < n_seqs; i++)
            {
                bound_windows[i] = static_cast<int>(std::min<std::int64_t>(cache_sizes[i],
                    qab_cap_round(static_cast<std::int64_t>(positions[i]) + 1)));
                bound_sig.push_back(static_cast<std::uintptr_t>(cache_sizes[i]));
                bound_sig.push_back(static_cast<std::uintptr_t>(bound_windows[i]));
            }
            for (std::size_t j = 0; j < static_cast<std::size_t>(attn_layers) * n_seqs; j++)
            {
                bound_sig.push_back(reinterpret_cast<std::uintptr_t>(k_cache_arr[j]));
                bound_sig.push_back(reinterpret_cast<std::uintptr_t>(v_cache_arr[j]));
            }
            for (std::size_t j = 0; j < static_cast<std::size_t>(gdn_layers) * n_seqs; j++)
            {
                bound_sig.push_back(reinterpret_cast<std::uintptr_t>(conv_state_arr[j]));
                bound_sig.push_back(reinterpret_cast<std::uintptr_t>(delta_state_arr[j]));
            }
        }
        int entry_idx = -1;
        for (int i = 0; i < kQabMaxEntries; i++)
        {
            QabEntry& e = pool.entries[i];
            if (e.valid && e.bound == bound && (!bound || e.bound_sig == bound_sig) &&
                e.sig_disc == sig_disc && e.n_slots == n_slots_req &&
                e.num_layers == num_layers && e.H == H && e.vocab == vocab_size &&
                e.hd == hd && e.kvH == kvH && e.kv_type == kvType &&
                e.hidden_input == hidden_input)
            {
                entry_idx = i;
                break;
            }
        }
        if (bound && entry_idx >= 0 && !pool.entries[entry_idx].bound_bindings_current())
        {
            pool.entries[entry_idx].release_graph();
            entry_idx = -1;
        }
        const bool cuda_windows = !bound && g_backend_type == BACKEND_TYPE_CUDA;
        std::vector<int> windows_by_seq;
        std::vector<int> desired_windows;
        bool windows_changed = false;
        if (cuda_windows)
        {
            // Match the whole-model solo decode policy: captured decode rounds
            // to 256 rows, capped by the holder's physical allocation.
            windows_by_seq.resize(n_seqs);
            for (int i = 0; i < n_seqs; i++)
                windows_by_seq[i] = static_cast<int>(std::min<std::int64_t>(cache_sizes[i],
                    qab_cap_round(static_cast<std::int64_t>(positions[i]) + 1)));

            // Predict the assignment below without retiring or seeding state:
            // keep resident callers, then place newcomers in the first free
            // slots. Absent callers will be retired and count as free here.
            // Unused slots retain their windows to avoid rebuilds on a shrink.
            desired_windows.assign(n_slots_req, 256);
            std::vector<int> planned_slots(n_seqs, -1);
            std::vector<bool> occupied(n_slots_req, false);
            if (entry_idx >= 0)
            {
                const QabEntry& e = pool.entries[entry_idx];
                if (e.attn_windows.size() == desired_windows.size())
                    desired_windows = e.attn_windows;
                for (int i = 0; i < n_seqs; i++)
                    for (int s = 0; s < n_slots_req; s++)
                        if (e.slots[s].active && e.slots[s].key == k_cache_arr[i])
                        {
                            planned_slots[i] = s;
                            occupied[s] = true;
                            break;
                        }
            }
            for (int i = 0; i < n_seqs; i++)
            {
                if (planned_slots[i] < 0)
                    for (int s = 0; s < n_slots_req; s++)
                        if (!occupied[s])
                        {
                            planned_slots[i] = s;
                            occupied[s] = true;
                            break;
                        }
                desired_windows[planned_slots[i]] = windows_by_seq[i];
            }
            windows_changed = entry_idx >= 0 &&
                pool.entries[entry_idx].attn_windows != desired_windows;
        }
        const bool needs_build = (entry_idx < 0) ||
            (!bound && pool.entries[entry_idx].cap < maxTotal) || windows_changed;

        if (needs_build)
        {
            if (bound)
            {
                // Windows only grow while the same holders decode together, so a
                // graph for the same slot keys at older windows will not replay.
                for (auto& stale : pool.entries)
                    if (stale.valid && stale.bound && stale.sig_disc == sig_disc &&
                        stale.n_slots == n_slots_req && stale.bound_sig.size() == bound_sig.size() &&
                        std::equal(bound_sig.begin() + 1 + 2 * n_seqs, bound_sig.end(),
                                   stale.bound_sig.begin() + 1 + 2 * n_seqs))
                        stale.release_graph();
            }
            if (entry_idx < 0)
            {
                for (int i = 0; i < kQabMaxEntries; i++)
                    if (!pool.entries[i].valid) { entry_idx = i; break; }
                if (entry_idx < 0)
                {
                    entry_idx = 0;
                    for (int i = 1; i < kQabMaxEntries; i++)
                        if (pool.used[i] < pool.used[entry_idx]) entry_idx = i;
                }
            }
            QabEntry& e = pool.entries[entry_idx];
            const std::int64_t prev_cap = e.valid ? e.cap : 0;
            const std::vector<int> prev_windows = e.valid ? e.attn_windows : std::vector<int>{};
            qab_flush_entry(e);
            e.release_graph();

            e.sig_disc = sig_disc;
            e.num_layers = num_layers;
            e.H = H;
            e.vocab = vocab_size;
            e.n_slots = n_slots_req;
            e.hd = hd;
            e.kvH = kvH;
            e.kv_type = kvType;
            e.hidden_input = hidden_input;
            e.conv_dim = conv_dim; e.convDim = convDim;
            e.head_k = head_k_dim; e.head_v = head_v_dim; e.v_heads = num_v_heads;
            if (bound)
            {
                e.bound = true;
                e.bound_sig = bound_sig;
                e.bound_cache_rows.assign(cache_sizes, cache_sizes + n_seqs);
                e.attn_windows = bound_windows;
                e.mask_len = *std::max_element(bound_windows.begin(), bound_windows.end());
                e.cap = 0;
                e.rows_per_slot = 0;
                e.slots.clear();
            }
            else
            {
            // Window-only rebuilds preserve physical allocation. Grow the arena
            // geometrically only when its existing capacity is insufficient.
            e.cap = std::max(qab_cap_round(maxTotal),
                cuda_windows && maxTotal <= prev_cap ? prev_cap : prev_cap * 2);
            e.rows_per_slot = static_cast<std::int64_t>(kvH) * e.cap;
            e.mask_len = e.cap;
            e.slots.assign(e.n_slots, QabSlot{});
            e.attn_windows.clear();
            }
            if (cuda_windows)
            {
                if (e.cap > std::numeric_limits<int>::max())
                {
                    set_last_error("Qwen3.5 arena batched decode: KV arena capacity exceeds supported window indexing.");
                    return 0;
                }
                e.attn_windows.assign(e.n_slots, 256);
                for (int s = n_seqs; s < e.n_slots && s < static_cast<int>(prev_windows.size()); s++)
                    e.attn_windows[s] = prev_windows[s];
                // Full rebuilds flush every old slot; the fresh assignment
                // below places caller i in slot i, independently of the old map.
                for (int i = 0; i < n_seqs; i++)
                    e.attn_windows[i] = windows_by_seq[i];
            }
            const int n_slots = e.n_slots;

            const std::size_t ctx_size = 128 * 1024 * 1024;
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ggml_context* ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("Qwen3.5 arena batched decode: failed to init ggml context.");
                return 0;
            }
            e.ctx = ctx;
            auto abort_build = [&](const char* msg) -> int {
                set_last_error(std::string("Qwen3.5 arena batched decode: ") + msg);
                e.release_graph();
                return 0;
            };

            // Attention reads only the real slot planes. Padded SET_ROWS
            // columns need one unique discard row per (slot, head), not a full
            // context-sized extra slot. Distinct rows avoid concurrent writes.
            const std::int64_t discard_rows = static_cast<std::int64_t>(kvH) * n_slots;
            const std::int64_t arena_rows = e.rows_per_slot * n_slots + discard_rows;

            if (hidden_input)
            {
                e.hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_slots);
                ggml_set_input(e.hidden_in);
            }
            else
            {
                e.token_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_slots);
                ggml_set_input(e.token_in);
            }
            e.pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_slots);
            e.idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, static_cast<std::int64_t>(kvH) * n_slots);
            e.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, e.mask_len, 1, 1, n_slots);
            ggml_set_input(e.pos_in);
            ggml_set_input(e.idx_in);
            ggml_set_input(e.attn_mask);

            e.k_arena.assign(num_layers, nullptr);
            e.v_arena.assign(num_layers, nullptr);
            e.conv_arena.assign(num_layers, nullptr);
            e.delta_arena.assign(num_layers, nullptr);
            if (bound)
            {
                // Shapes and byte sizes match the solo decode's bindings of the same
                // host keys, so both graphs resolve to one resident copy per key.
                const std::size_t per = static_cast<std::size_t>(num_layers) * n_slots;
                e.k_bound.assign(per, nullptr);
                e.v_bound.assign(per, nullptr);
                e.conv_bound.assign(per, nullptr);
                e.delta_bound.assign(per, nullptr);
                for (int l = 0; l < num_layers; l++)
                    for (int s = 0; s < n_slots; s++)
                    {
                        const std::size_t ix = static_cast<std::size_t>(l) * n_slots + s;
                        if (layers[l].is_recurrent == 0)
                        {
                            const std::int64_t rows = static_cast<std::int64_t>(kvH) * e.bound_cache_rows[s];
                            e.k_bound[ix] = ggml_new_tensor_2d(ctx, kvType, hd, rows);
                            e.v_bound[ix] = ggml_new_tensor_2d(ctx, kvType, hd, rows);
                        }
                        else
                        {
                            e.conv_bound[ix] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                            e.delta_bound[ix] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads, 1);
                        }
                    }
            }
            for (int l = 0; l < num_layers && !bound; l++)
            {
                if (layers[l].is_recurrent == 0)
                {
                    e.k_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
                    e.v_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
                }
                else
                {
                    e.conv_arena[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, convDim, conv_dim, n_slots);
                    e.delta_arena[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads, n_slots);
                }
            }

            struct LW
            {
                ggml_tensor* attn_norm_w = nullptr;
                ggml_tensor* post_attn_norm_w = nullptr;
                ggml_tensor* qkv_w = nullptr;
                ggml_tensor* k_w = nullptr;
                ggml_tensor* v_w = nullptr;
                ggml_tensor* q_norm_w = nullptr;
                ggml_tensor* k_norm_w = nullptr;
                ggml_tensor* o_w = nullptr;
                ggml_tensor* gdn_qkv_w = nullptr;
                ggml_tensor* gdn_gate_w = nullptr;
                ggml_tensor* ssm_beta_w = nullptr;
                ggml_tensor* ssm_alpha_w = nullptr;
                ggml_tensor* conv1d_w = nullptr;
                ggml_tensor* ssm_dt_w = nullptr;
                ggml_tensor* ssm_a_w = nullptr;
                ggml_tensor* ssm_norm_w = nullptr;
                ggml_tensor* ssm_out_w = nullptr;
                ggml_tensor* gu_w = nullptr;
                ggml_tensor* ffn_gate_w = nullptr;
                ggml_tensor* ffn_up_w = nullptr;
                ggml_tensor* down_w = nullptr;
                ggml_tensor* gate_inp_w = nullptr;
                ggml_tensor* gate_exps = nullptr;
                ggml_tensor* up_exps = nullptr;
                ggml_tensor* down_exps = nullptr;
                ggml_tensor* shexp_gate_w = nullptr;
                ggml_tensor* shexp_up_w = nullptr;
                ggml_tensor* shexp_down_w = nullptr;
                ggml_tensor* shexp_gate_inp_w = nullptr;
                ggml_tensor* psc[TSQ35_SC_COUNT] = {};
            };
            std::vector<LW> lw(num_layers);
            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                LW& t = lw[l];
                t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
                t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
                if (d.is_recurrent == 0)
                {
                    t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                    if (d.separate_qkv != 0)
                    {
                        t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                        t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                    }
                    t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                    t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                    t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                }
                else
                {
                    t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                    if (d.gdn_gate_w != nullptr)
                    {
                        t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                        t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                        t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                    }
                    t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                    t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                    t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                    t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                    t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                }
                if (d.is_moe == 0)
                {
                    if (d.gu_w != nullptr)
                        t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                    else
                    {
                        t.ffn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_gate_type), d.ffn_gate_ne0, d.ffn_gate_ne1);
                        t.ffn_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_up_type), d.ffn_up_ne0, d.ffn_up_ne1);
                    }
                    t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
                }
                else
                {
                    t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                    t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), H, expert_ff, num_experts);
                    t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), H, expert_ff, num_experts);
                    t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, H, num_experts);
                    t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                    t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                    t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                    t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
                }
            }

            ggml_tensor* token_embd_t = hidden_input ? nullptr :
                ggml_new_tensor_2d(ctx, static_cast<ggml_type>(token_embd_type), token_embd_ne0, token_embd_ne1);
            ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

            const std::size_t row_bytes = ggml_row_size(kvType, hd);

            tsg::BonsaiGraphScope bonsai_scope;

            // ---- graph ----
            // Unsupported CUDA embedding types (including Q2_K) are gathered
            // once per row by the caller. Only H*N floats cross the bus; every
            // transformer projection and the LM head still runs in this graph.
            ggml_tensor* hidden = e.hidden_in;
            if (!hidden_input)
            {
                ggml_tensor* rows = ggml_get_rows(ctx, token_embd_t, e.token_in);
                if (!backend_supports_op(rows))
                    return abort_build("token embedding get_rows unsupported; use host-gathered embedding input.");
                hidden = tsg::bonsai_transform(ctx, rows, token_embd_data, true);
            }
            std::vector<ggml_tensor*> state_writes;
            state_writes.reserve(static_cast<std::size_t>(gdn_layers) * n_slots * 2 + attn_layers * 2);
            bool op_unsupported = false;

            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                LW& t = lw[l];
                ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);   // [H, N]
                ggml_tensor* block_out;

                if (d.is_recurrent == 0)
                {
                    // ===== Full attention, batched over N (solo recipe widened) =====
                    ggml_tensor* qg_part;
                    ggml_tensor* k_raw;
                    ggml_tensor* v_raw;
                    if (d.separate_qkv != 0)
                    {
                        qg_part = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.qkv_w, normed, d.qkv_w), q35_psc(ctx, t, d, TSQ35_SC_QKV));                       // [qFullDim, N]
                        k_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.k_w, normed, d.k_w), q35_psc(ctx, t, d, TSQ35_SC_K));                           // [kDim, N]
                        v_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.v_w, normed, d.v_w), q35_psc(ctx, t, d, TSQ35_SC_V));
                    }
                    else
                    {
                        ggml_tensor* qkv = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.qkv_w, normed, d.qkv_w), q35_psc(ctx, t, d, TSQ35_SC_QKV));              // [qFullDim+2kDim, N]
                        qg_part = ggml_view_2d(ctx, qkv, qFullDim, n_slots, qkv->nb[1], 0);
                        k_raw = ggml_view_2d(ctx, qkv, kDim, n_slots, qkv->nb[1],
                            static_cast<std::size_t>(qFullDim) * sizeof(float));
                        v_raw = ggml_view_2d(ctx, qkv, kDim, n_slots, qkv->nb[1],
                            static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                    }

                    // Deinterleave the per-head [Q | gate] pairs, batched: view the
                    // packed [hd, 2, nH, N] and slice.
                    ggml_tensor* qg_cont = ggml_cont(ctx, qg_part);                          // [qFullDim, N]
                    ggml_tensor* qg_4d = ggml_reshape_4d(ctx, qg_cont, hd, 2, nH, n_slots);
                    ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d, hd, nH, n_slots,
                        qg_4d->nb[2], qg_4d->nb[3], 0);
                    ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d, hd, nH, n_slots,
                        qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);

                    ggml_tensor* q_2d = ggml_cont(ctx, q_view);                              // [hd, nH, N]
                    ggml_tensor* k_2d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), hd, kvH, n_slots);
                    ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), t.q_norm_w);
                    ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), t.k_norm_w);

                    // Solo decode reshapes these normalized tensors before RoPE,
                    // preventing CUDA's RMS_NORM + MUL + ROPE fusion. The arena's
                    // direct edge otherwise permits a different fused computation;
                    // its small differences compound through recurrent decode.
                    // Preserve the same boundary here. RMS_NORM + MUL can still
                    // fuse, and every heavy projection remains token-batched.
                    if (g_backend_type == BACKEND_TYPE_CUDA)
                    {
                        ggml_set_output(q_normed);
                        ggml_set_output(k_normed);
                    }

                    ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, e.pos_in, nullptr,
                        rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [hd, nH, N]
                    ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, e.pos_in, nullptr,
                        rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [hd, kvH, N]

                    // ONE scatter per K/V for the whole batch (idx carries
                    // slot*kvH*cap + head*cap + pos; padded columns aim at the
                    // scratch slice). Attention reads views of the set_rows
                    // RESULT so the write->read edge is a real src edge.
                    ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), hd, kvH, n_slots);
                    ggml_tensor* k_set = nullptr;
                    ggml_tensor* v_set = nullptr;
                    std::vector<ggml_tensor*> k_sets, v_sets;
                    if (bound)
                    {
                        // One scatter per slot into that holder's own K/V copy
                        // (idx carries head*rows + pos for the slot's kvH rows).
                        for (int s = 0; s < n_slots; s++)
                        {
                            const std::size_t ix = static_cast<std::size_t>(l) * n_slots + s;
                            ggml_tensor* idx_s = ggml_view_1d(ctx, e.idx_in, kvH,
                                static_cast<std::size_t>(s) * kvH * sizeof(std::int64_t));
                            ggml_tensor* k_rows_s = ggml_view_2d(ctx, k_rope, hd, kvH, k_rope->nb[1],
                                static_cast<std::size_t>(s) * k_rope->nb[2]);
                            ggml_tensor* v_rows_s = ggml_view_2d(ctx, v_3d, hd, kvH, v_3d->nb[1],
                                static_cast<std::size_t>(s) * v_3d->nb[2]);
                            ggml_tensor* ks = ggml_set_rows(ctx, e.k_bound[ix], k_rows_s, idx_s);
                            ggml_tensor* vs = ggml_set_rows(ctx, e.v_bound[ix], v_rows_s, idx_s);
                            if (!op_unsupported && !backend_supports_op(ks))
                                op_unsupported = true;
                            state_writes.push_back(ks);
                            state_writes.push_back(vs);
                            k_sets.push_back(ks);
                            v_sets.push_back(vs);
                        }
                    }
                    else
                    {
                        ggml_tensor* k_rows = ggml_reshape_2d(ctx, k_rope, hd, static_cast<std::int64_t>(kvH) * n_slots);
                        ggml_tensor* v_rows = ggml_reshape_2d(ctx, v_3d, hd, static_cast<std::int64_t>(kvH) * n_slots);
                        k_set = ggml_set_rows(ctx, e.k_arena[l], k_rows, e.idx_in);
                        v_set = ggml_set_rows(ctx, e.v_arena[l], v_rows, e.idx_in);
                        if (!op_unsupported && !backend_supports_op(k_set))
                            op_unsupported = true;
                        state_writes.push_back(k_set);
                        state_writes.push_back(v_set);
                    }

                    if (op_unsupported)
                        return abort_build("set_rows unsupported for the arena shapes.");

                    ggml_tensor* attn_2d = nullptr;
                    if (g_backend_type == BACKEND_TYPE_CUDA)
                    {
                        // Use the solo sequence-axis geometry. CUDA's multi-sequence
                        // attention changes the reduction topology; the recurrent
                        // model can amplify the resulting numerical differences.
                        // Attention has no shared weights to amortize: isolate these
                        // nodes while projections/FFNs remain token-batched. Views
                        // read SET_ROWS results to preserve every write/read edge.
                        for (int s = 0; s < n_slots; s++)
                        {
                            // Bound slots read their holder's copy at its own pitch.
                            const std::int64_t slot_cap = bound ? e.bound_cache_rows[s] : e.cap;
                            const std::size_t slot_offset = bound ? 0 :
                                static_cast<std::size_t>(s) * e.rows_per_slot * row_bytes;
                            ggml_tensor* k_src = bound ? k_sets[s] : k_set;
                            ggml_tensor* v_src = bound ? v_sets[s] : v_set;
                            ggml_tensor* k_slot = ggml_view_3d(ctx, k_src, hd, slot_cap, kvH,
                                row_bytes, static_cast<std::size_t>(slot_cap) * row_bytes, slot_offset);
                            ggml_tensor* v_slot = ggml_view_3d(ctx, v_src, hd, slot_cap, kvH,
                                row_bytes, static_cast<std::size_t>(slot_cap) * row_bytes, slot_offset);
                            const int window = e.attn_windows[s];
                            // Keep physical head pitch while exposing the same
                            // effective length as solo. The shared helper also
                            // protects affected CUDA F16 truncated-window reads.
                            ggml_tensor* k_view = view_kv_cache_window(ctx, k_slot, hd,
                                static_cast<int>(slot_cap), kvH, 0, window, kv_cache_type, 1, nH / kvH);
                            ggml_tensor* v_view = view_kv_cache_window(ctx, v_slot, hd,
                                static_cast<int>(slot_cap), kvH, 0, window, kv_cache_type, 1, nH / kvH);
                            if (k_view == nullptr || v_view == nullptr)
                                return abort_build("failed to create per-slot KV attention windows.");
                            ggml_tensor* q_slot = ggml_view_3d(ctx, q_rope, hd, 1, nH,
                                q_rope->nb[2], q_rope->nb[1],
                                static_cast<std::size_t>(s) * q_rope->nb[2]);
                            ggml_tensor* mask_slot = ggml_view_2d(ctx, e.attn_mask, window, 1,
                                e.attn_mask->nb[1], static_cast<std::size_t>(s) * e.attn_mask->nb[3]);
                            ggml_tensor* fa = flash_attn_ext_guarded(ctx,
                                "Qwen3.5 arena slot decode", q_slot, k_view, v_view, mask_slot,
                                attn_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
                            ggml_tensor* attn_col = ggml_reshape_3d(ctx, fa, hd, nH, 1);
                            attn_2d = attn_2d == nullptr
                                ? attn_col
                                : ggml_concat(ctx, attn_2d, attn_col, 2);
                        }
                    }
                    else
                    {
                        // Preserve Metal's established batched attention graph.
                        ggml_tensor* k_view = ggml_view_4d(ctx, k_set, hd, e.cap, kvH, n_slots,
                            row_bytes, static_cast<std::size_t>(e.cap) * row_bytes,
                            static_cast<std::size_t>(e.rows_per_slot) * row_bytes, 0);
                        ggml_tensor* v_view = ggml_view_4d(ctx, v_set, hd, e.cap, kvH, n_slots,
                            row_bytes, static_cast<std::size_t>(e.cap) * row_bytes,
                            static_cast<std::size_t>(e.rows_per_slot) * row_bytes, 0);
                        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_rope, hd, 1, nH, n_slots);
                        ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_4d, k_view, v_view,
                            e.attn_mask, attn_scale, 0.0f, 0.0f);
                        ggml_prec_set_acc(fa, GGML_PREC_F32);
                        if (!backend_supports_op(fa))
                            return abort_build("flash_attn unsupported for the arena shapes.");
                        attn_2d = ggml_reshape_3d(ctx, fa, hd, nH, n_slots);
                    }

                    ggml_tensor* gate_cont = ggml_cont(ctx, gate_view);
                    ggml_tensor* attn_gated = ggml_mul(ctx, attn_2d, ggml_sigmoid(ctx, gate_cont));
                    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_gated), qDim, n_slots);
                    block_out = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.o_w, attn_flat, d.o_w), q35_psc(ctx, t, d, TSQ35_SC_O));                          // [H, N]
                }
                else
                {
                    // ===== Gated Delta Net: batched projections, per-slot recurrence =====
                    ggml_tensor* qkv_mixed;   // [conv_dim, N]
                    ggml_tensor* z;           // [value_dim, N]
                    ggml_tensor* beta_raw;    // [num_v_heads, N]
                    ggml_tensor* alpha_raw;
                    if (t.gdn_gate_w == nullptr)
                    {
                        const std::int64_t packed_dim = d.gdn_qkv_ne1;
                        ggml_tensor* packed = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_qkv_w, normed, d.gdn_qkv_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_QKV));         // [packed_dim, N]
                        qkv_mixed = ggml_view_2d(ctx, packed, conv_dim, n_slots, packed->nb[1], 0);
                        z = ggml_view_2d(ctx, packed, value_dim, n_slots, packed->nb[1],
                            static_cast<std::size_t>(conv_dim) * sizeof(float));
                        beta_raw = ggml_view_2d(ctx, packed, num_v_heads, n_slots, packed->nb[1],
                            static_cast<std::size_t>(conv_dim + value_dim) * sizeof(float));
                        alpha_raw = ggml_view_2d(ctx, packed, num_v_heads, n_slots, packed->nb[1],
                            static_cast<std::size_t>(conv_dim + value_dim + num_v_heads) * sizeof(float));
                        (void)packed_dim;
                    }
                    else
                    {
                        qkv_mixed = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_qkv_w, normed, d.gdn_qkv_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_QKV));
                        z = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_gate_w, normed, d.gdn_gate_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_GATE));
                        beta_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_beta_w, normed, d.ssm_beta_w), q35_psc(ctx, t, d, TSQ35_SC_BETA));
                        alpha_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_alpha_w, normed, d.ssm_alpha_w), q35_psc(ctx, t, d, TSQ35_SC_ALPHA));
                    }
                    ggml_tensor* qkv_cont = ggml_cont(ctx, qkv_mixed);                        // [conv_dim, N]
                    ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_cont(ctx, beta_raw));      // [num_v_heads, N]
                    ggml_tensor* alpha_cont = ggml_cont(ctx, alpha_raw);
                    ggml_tensor* g_all = ggml_softplus(ctx,
                        ggml_add(ctx, alpha_cont, t.ssm_dt_w));                               // [num_v_heads, N]
                    g_all = ggml_mul(ctx, g_all, t.ssm_a_w);

                    // Per-slot recurrent chain over the persistent state slices.
                    // Metal's concurrent graph scheduler needs an explicit
                    // producer -> consumer edge for every output column. A set
                    // of independent cpy graph outputs can otherwise race the
                    // consumer of gdn_batch. Keep CUDA's established copy graph
                    // unchanged and use concat to express that dependency only
                    // on Metal.
                    const bool dependency_batch = g_backend_type == BACKEND_TYPE_METAL;
                    ggml_tensor* gdn_batch = dependency_batch
                        ? nullptr
                        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, value_dim, n_slots);
                    for (int s = 0; s < n_slots; s++)
                    {
                        const std::size_t bix = static_cast<std::size_t>(l) * n_slots + s;
                        ggml_tensor* conv_slice = bound ? e.conv_bound[bix] :
                            ggml_view_2d(ctx, e.conv_arena[l], convDim, conv_dim,
                                e.conv_arena[l]->nb[1], static_cast<std::size_t>(s) * e.conv_arena[l]->nb[2]);
                        ggml_tensor* qkv_col = ggml_view_2d(ctx, qkv_cont, conv_dim, 1,
                            qkv_cont->nb[1], static_cast<std::size_t>(s) * qkv_cont->nb[1]);
                        ggml_tensor* conv_input = ggml_concat(ctx, conv_slice,
                            ggml_transpose(ctx, qkv_col), 0);                                  // [convDim+1, conv_dim]
                        ggml_tensor* conv_out = ggml_silu(ctx, tsg::ssm_conv_any(ctx, conv_input, t.conv1d_w)); // [conv_dim, 1]
                        ggml_tensor* conv_out_1d = ggml_reshape_1d(ctx, conv_out, conv_dim);

                        ggml_tensor* new_conv = ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                            conv_input->nb[1], conv_input->nb[0]);
                        state_writes.push_back(ggml_cpy(ctx, new_conv, conv_slice));

                        ggml_tensor* q_c = ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                            static_cast<std::size_t>(head_k_dim) * sizeof(float), 0);
                        ggml_tensor* k_c = ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                            static_cast<std::size_t>(head_k_dim) * sizeof(float),
                            static_cast<std::size_t>(key_dim) * sizeof(float));
                        ggml_tensor* v_c = ggml_view_2d(ctx, conv_out_1d, head_v_dim, num_v_heads,
                            static_cast<std::size_t>(head_v_dim) * sizeof(float),
                            static_cast<std::size_t>(2 * key_dim) * sizeof(float));
                        q_c = build_gdn_l2_norm(ctx, q_c, eps);
                        k_c = build_gdn_l2_norm(ctx, k_c, eps);

                        ggml_tensor* q4 = ggml_reshape_4d(ctx, q_c, head_k_dim, num_k_heads, 1, 1);
                        ggml_tensor* k4 = ggml_reshape_4d(ctx, k_c, head_k_dim, num_k_heads, 1, 1);
                        ggml_tensor* v4 = ggml_reshape_4d(ctx, v_c, head_v_dim, num_v_heads, 1, 1);
                        ggml_tensor* beta_s = ggml_view_4d(ctx, beta_all, 1, num_v_heads, 1, 1,
                            beta_all->nb[0], beta_all->nb[1], beta_all->nb[1],
                            static_cast<std::size_t>(s) * beta_all->nb[1]);
                        ggml_tensor* g_s = ggml_view_4d(ctx, g_all, 1, num_v_heads, 1, 1,
                            g_all->nb[0], g_all->nb[1], g_all->nb[1],
                            static_cast<std::size_t>(s) * g_all->nb[1]);
                        ggml_tensor* state4 = bound ? e.delta_bound[bix] :
                            ggml_view_4d(ctx, e.delta_arena[l],
                                head_k_dim, head_v_dim, num_v_heads, 1,
                                e.delta_arena[l]->nb[1], e.delta_arena[l]->nb[2], e.delta_arena[l]->nb[3],
                                static_cast<std::size_t>(s) * e.delta_arena[l]->nb[3]);

                        ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g_s, beta_s, state4, 1);
                        if (l == 0 || (s == 0 && !op_unsupported))
                        {
                            if (!backend_supports_op(gdn))
                                return abort_build("gated_delta_net unsupported for the arena shapes.");
                        }
                        ggml_tensor* new_state = ggml_view_4d(ctx, gdn,
                            head_k_dim, head_v_dim, num_v_heads, 1,
                            ggml_row_size(gdn->type, head_k_dim),
                            ggml_row_size(gdn->type, static_cast<std::int64_t>(head_k_dim) * head_v_dim),
                            ggml_row_size(gdn->type, static_cast<std::int64_t>(head_k_dim) * head_v_dim * num_v_heads),
                            ggml_row_size(gdn->type, static_cast<std::int64_t>(head_v_dim) * num_v_heads));
                        state_writes.push_back(ggml_cpy(ctx, new_state, state4));

                        ggml_tensor* gdn_out = ggml_view_1d(ctx, gdn, value_dim, 0);
                        if (dependency_batch)
                        {
                            ggml_tensor* gdn_col = ggml_reshape_2d(ctx, gdn_out, value_dim, 1);
                            gdn_batch = gdn_batch == nullptr
                                ? gdn_col
                                : ggml_concat(ctx, gdn_batch, gdn_col, 1);
                        }
                        else
                        {
                            ggml_tensor* out_col = ggml_view_1d(ctx, gdn_batch, value_dim,
                                static_cast<std::size_t>(s) * gdn_batch->nb[1]);
                            state_writes.push_back(ggml_cpy(ctx, gdn_out, out_col));
                        }
                    }

                    // Batched gated RMSNorm + output projection over the collected
                    // [value_dim, N] core outputs.
                    ggml_tensor* out_3d = ggml_reshape_3d(ctx, gdn_batch, head_v_dim, num_v_heads, n_slots);
                    ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_3d, eps), t.ssm_norm_w);
                    ggml_tensor* z_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, z), head_v_dim, num_v_heads, n_slots);
                    ggml_tensor* gated = ggml_mul(ctx, out_n, ggml_silu(ctx, z_3d));
                    ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, n_slots);
                    block_out = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_out_w, gated_flat, d.ssm_out_w), q35_psc(ctx, t, d, TSQ35_SC_SSM_OUT));                    // [H, N]
                }

                ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out);

                // ===== FFN (batched over N; solo recipe with the token axis restored) =====
                ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w);
                ggml_tensor* ffn_down;
                if (d.is_moe == 0)
                {
                    ggml_tensor* act;
                    if (t.gu_w != nullptr)
                        act = ggml_swiglu(ctx, q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gu_w, ffn_normed, d.gu_w), q35_psc(ctx, t, d, TSQ35_SC_GU)));
                    else
                    {
                        ggml_tensor* g2 = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ffn_gate_w, ffn_normed, d.ffn_gate_w), q35_psc(ctx, t, d, TSQ35_SC_FFN_GATE));
                        ggml_tensor* u2 = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ffn_up_w, ffn_normed, d.ffn_up_w), q35_psc(ctx, t, d, TSQ35_SC_FFN_UP));
                        act = ggml_mul(ctx, ggml_silu(ctx, g2), u2);
                    }
                    ffn_down = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.down_w, act, d.down_w), q35_psc(ctx, t, d, TSQ35_SC_DOWN));                               // [H, N]
                }
                else
                {
                    ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed);  // [nExp, N]
                    ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                    ggml_tensor* sel = tsg::ordered_moe_top_k(ctx, probs, num_experts_used);    // [nUsed, N]
                    ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, n_slots);
                    ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                          // [1, nUsed, N]
                    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, n_slots);
                    if (norm_topk != 0)
                    {
                        ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
                        w_2d = ggml_div(ctx, w_2d, w_sum);
                    }
                    if (expert_weights_scale != 1.0f)
                        w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                    ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, n_slots);

                    ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, n_slots);
                    ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);
                    ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                    ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                    ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);       // [H, nUsed, N]
                    ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
                    ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_slots, weighted->nb[2], 0);
                    for (int u = 1; u < num_experts_used; ++u)
                    {
                        ggml_tensor* vu = ggml_view_2d(ctx, weighted, H, n_slots, weighted->nb[2],
                            static_cast<std::size_t>(u) * weighted->nb[1]);
                        moe_out = ggml_add(ctx, moe_out, vu);
                    }

                    ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed);
                    ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                    ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                    ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act);          // [H, N]
                    ggml_tensor* sh_gate = ggml_sigmoid(ctx,
                        ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, N]
                    ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                    ffn_down = ggml_add(ctx, moe_out, sh_out);
                }

                hidden = ggml_add(ctx, residual1, ffn_down);
            }

            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* logits = tsg::bonsai_mul_mat(ctx, lm_head_t, fn, lm_head_data);                            // [vocab, N]
            e.logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_slots);
            ggml_tensor* out_cpy = ggml_cpy(ctx, logits, e.logits_out);
            ggml_set_output(out_cpy);

            ggml_tensor* amax = ggml_argmax(ctx, logits);
            e.has_argmax = backend_supports_op(amax);
            if (e.has_argmax)
            {
                ggml_set_output(amax);
                e.sampled_out = amax;
            }

            const std::size_t graph_size =
                static_cast<std::size_t>(num_layers) * (192 + 40 * static_cast<std::size_t>(n_slots)) + 8192;
            e.graph = ggml_new_graph_custom(ctx, graph_size, false);
            // The recurrent-state copies and KV scatters have no consumer edge to
            // the logits; expand them explicitly (in build order: writes precede
            // the next layer's reads within each slot's chain by construction).
            for (ggml_tensor* wnode : state_writes)
            {
                ggml_set_output(wnode);
                ggml_build_forward_expand(e.graph, wnode);
            }
            ggml_build_forward_expand(e.graph, out_cpy);
            if (e.has_argmax)
                ggml_build_forward_expand(e.graph, amax);

            // ---- bind weights ----
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
            std::vector<HostBinding> upload_list;
            auto bind_or_mark = [&](ggml_tensor* tgt, const void* data, std::size_t bytes) {
                if (tgt == nullptr || data == nullptr) return;
                if (bytes >= 4096)
                {
                    bool needs_upload = false;
                    if (try_bind_cached_tensor(g_backend, dev, tgt, const_cast<void*>(data), bytes, needs_upload,
                                               GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                    {
                        if (needs_upload) upload_list.push_back({tgt, const_cast<void*>(data), bytes});
                        return;
                    }
                    ggml_backend_buffer_t buf = nullptr;
                    if (try_get_host_ptr_buffer(g_backend, dev, const_cast<void*>(data), bytes, true, buf))
                    {
                        if (ggml_backend_tensor_alloc(buf, tgt, const_cast<void*>(data)) == GGML_STATUS_SUCCESS)
                            return;
                    }
                }
                upload_list.push_back({tgt, const_cast<void*>(data), bytes});
            };

            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                LW& t = lw[l];
                // Only consumed scale slots exist, matching the solo and verify graphs.
                for (int slot = 0; slot < TSQ35_SC_COUNT; ++slot)
                    if (t.psc[slot] != nullptr)
                        bind_or_mark(t.psc[slot], static_cast<float*>(d.proj_scales) + slot, sizeof(float));
                bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
                bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
                if (d.is_recurrent == 0)
                {
                    bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes));
                    if (d.separate_qkv != 0)
                    {
                        bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes));
                        bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes));
                    }
                    bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes));
                    bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float));
                    bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float));
                }
                else
                {
                    bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes));
                    bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes));
                    bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes));
                    bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes));
                    bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float));
                    bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float));
                    bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float));
                    bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float));
                    bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes));
                }
                if (d.is_moe == 0)
                {
                    if (t.gu_w != nullptr)
                        bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes));
                    else
                    {
                        bind_or_mark(t.ffn_gate_w, d.ffn_gate_w, static_cast<std::size_t>(d.ffn_gate_bytes));
                        bind_or_mark(t.ffn_up_w, d.ffn_up_w, static_cast<std::size_t>(d.ffn_up_bytes));
                    }
                    bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes));
                }
                else
                {
                    bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes));
                    bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes));
                    bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes));
                    bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes));
                    bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes));
                    bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes));
                    bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes));
                    bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float));
                }
            }
            bind_or_mark(lm_head_t, lm_head_data, static_cast<std::size_t>(lm_head_bytes));
            bind_or_mark(final_norm_t, final_norm_data, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(token_embd_t, token_embd_data, static_cast<std::size_t>(token_embd_bytes));

            if (bound)
            {
                // Bind each slot to its holder's resident copies with the solo
                // decode's usage and sizes. A copy created here has never been
                // written: fill it from the host bytes (the truth whenever no copy
                // exists) before anything can fail, so the cache never publishes
                // an unwritten entry.
                auto bind_holder = [&](ggml_tensor* tgt, const void* host, std::size_t bytes) -> bool {
                    bool needs_upload = false;
                    if (!try_bind_cached_tensor(g_backend, dev, tgt, const_cast<void*>(host), bytes, needs_upload,
                                                GGML_BACKEND_BUFFER_USAGE_COMPUTE))
                        return false;
                    if (needs_upload)
                        ggml_backend_tensor_set(tgt, host, 0, bytes);
                    e.bound_keys.push_back({host, tgt->buffer, tgt->data, bytes});
                    return true;
                };
                host_read_barrier();
                for (int l = 0, al = 0, gl = 0; l < num_layers; l++)
                {
                    const bool attn = layers[l].is_recurrent == 0;
                    for (int s = 0; s < n_slots; s++)
                    {
                        const std::size_t ix = static_cast<std::size_t>(l) * n_slots + s;
                        bool ok;
                        if (attn)
                        {
                            const std::size_t kv_bytes = kv_cache_bytes(kvH, e.bound_cache_rows[s], hd, kv_cache_type);
                            ok = bind_holder(e.k_bound[ix], k_cache_arr[static_cast<std::size_t>(al) * n_seqs + s], kv_bytes) &&
                                 bind_holder(e.v_bound[ix], v_cache_arr[static_cast<std::size_t>(al) * n_seqs + s], kv_bytes);
                        }
                        else
                        {
                            ok = bind_holder(e.conv_bound[ix], conv_state_arr[static_cast<std::size_t>(gl) * n_seqs + s], convBytes) &&
                                 bind_holder(e.delta_bound[ix], delta_state_arr[static_cast<std::size_t>(gl) * n_seqs + s], deltaBytes);
                        }
                        if (!ok)
                            return abort_build("failed to bind a holder's resident cache copy.");
                    }
                    (attn ? al : gl)++;
                }
            }

            // Everything unbound (inputs, arenas, intermediates, logits) lands in
            // the entry's own buffer — nothing references the shared gallocr
            // pool, which is what lets this graph survive prefills and holder
            // churn. Zero it: fattn reads masked-but-unwritten arena rows, and
            // recycled VRAM decodes as NaN which survives the -inf mask.
            e.buffer = (g_backend_type == BACKEND_TYPE_METAL
                ? alloc_ctx_tensors_with_attention_reuse(ctx, e.graph, g_backend)
                : ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (e.buffer == nullptr)
                return abort_build("failed to allocate the arena backend buffer.");
            ggml_backend_buffer_clear(e.buffer, 0);

            host_read_barrier();
            for (auto& u : upload_list)
                ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

            e.tok_stage.assign(n_slots, 0);
            e.hidden_stage.assign(hidden_input ? static_cast<std::size_t>(H) * n_slots : 0, 0.0f);
            e.pos_stage.assign(n_slots, 0);
            e.idx_stage.assign(static_cast<std::size_t>(kvH) * n_slots, 0);
            e.logits_stage.assign(static_cast<std::size_t>(vocab_size) * n_slots, 0.0f);
            e.sampled_stage.assign(n_slots, 0);
            e.valid = true;
        }

        QabEntry& e = pool.entries[entry_idx];
        pool.used[entry_idx] = ++pool.clock;
        const int n_slots = e.n_slots;
        const std::size_t row_bytes = ggml_row_size(kvType, hd);

        // ---- slot assignment + joins ----
        std::vector<int> slot_of(n_seqs, -1);
        if (bound)
        {
            // Slot i is sequence i; the holders' copies already hold their state.
            for (int i = 0; i < n_seqs; i++)
                slot_of[i] = i;
        }
        for (int i = 0; i < n_seqs && !bound; i++)
        {
            const void* key = k_cache_arr[i];   // first attention layer, seq i
            for (int s = 0; s < n_slots; s++)
                if (e.slots[s].active && e.slots[s].key == key) { slot_of[i] = s; break; }
        }
        // Arena-owned slots retire absent sequences and join newcomers by
        // copying their state in; bound slots have nothing to move.
        if (!bound)
        {
            // The graph advances EVERY slot's GDN recurrent state each step (there
            // is no per-slot skip in a slot-stable graph), so an ACTIVE slot whose
            // sequence is not in this call would be phantom-advanced by a pad
            // token. Flush and retire such slots now; their sequences simply
            // rejoin later. (KV padded lanes are already scratch-redirected; the
            // recurrent state has no such escape.)
            for (int s = 0; s < n_slots; s++)
            {
                if (!e.slots[s].active) continue;
                bool in_call = false;
                for (int i = 0; i < n_seqs; i++)
                    if (slot_of[i] == s) { in_call = true; break; }
                if (!in_call)
                    qab_flush_and_drop_slot(e, s);
            }
            // A sequence may hold a DIRTY slot in a different bucket's entry (the
            // concurrency crossed a power of two). Flush that mapping first so the
            // join below reads current host bytes instead of pre-arena state.
            {
                QabRegistry& reg = qab_registry();
                for (int i = 0; i < n_seqs; i++)
                {
                    if (slot_of[i] >= 0) continue;
                    auto it = reg.find(k_cache_arr[i]);
                    if (it != reg.end())
                    {
                        // slot_of[i] < 0 means no matching slot in THIS entry, so
                        // any registry hit is a stale mapping elsewhere.
                        qab_flush_and_drop_slot(
                            pool.entries[it->second.first], it->second.second);
                    }
                }
            }
            for (int i = 0; i < n_seqs; i++)
            {
                if (slot_of[i] >= 0) continue;
                int s = -1;
                for (int j = 0; j < n_slots; j++)
                    if (!e.slots[j].active) { s = j; break; }
                if (s < 0)
                {
                    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
                    for (int j = 0; j < n_slots; j++)
                    {
                        bool in_call = false;
                        for (int k = 0; k < n_seqs; k++)
                            if (slot_of[k] == j) { in_call = true; break; }
                        if (!in_call && e.slots[j].last_used < best) { best = e.slots[j].last_used; s = j; }
                    }
                    if (s < 0)
                    {
                        set_last_error("Qwen3.5 arena batched decode: no free arena slot.");
                        return 0;
                    }
                    qab_flush_and_drop_slot(e, s);
                }
                slot_of[i] = s;
                QabSlot& sl = e.slots[s];
                sl.active = true;
                sl.key = k_cache_arr[i];
                sl.cache_rows = cache_sizes[i];
                sl.len = 0;
                sl.clean = 0;
                sl.state_seeded = false;
                sl.state_dirty = false;
                sl.k_hosts.resize(attn_layers);
                sl.v_hosts.resize(attn_layers);
                sl.conv_hosts.resize(gdn_layers);
                sl.delta_hosts.resize(gdn_layers);
                for (int l = 0, al = 0, gl = 0; l < num_layers; l++)
                {
                    if (layers[l].is_recurrent == 0)
                    {
                        sl.k_hosts[al] = k_cache_arr[static_cast<std::size_t>(al) * n_seqs + i];
                        sl.v_hosts[al] = v_cache_arr[static_cast<std::size_t>(al) * n_seqs + i];
                        al++;
                    }
                    else
                    {
                        sl.conv_hosts[gl] = conv_state_arr[static_cast<std::size_t>(gl) * n_seqs + i];
                        sl.delta_hosts[gl] = delta_state_arr[static_cast<std::size_t>(gl) * n_seqs + i];
                        gl++;
                    }
                }
                QabRegistry& reg = qab_registry();
                for (const void* p : sl.k_hosts) reg[p] = {entry_idx, s};
                for (const void* p : sl.v_hosts) reg[p] = {entry_idx, s};
                for (const void* p : sl.conv_hosts) reg[p] = {entry_idx, s};
                for (const void* p : sl.delta_hosts) reg[p] = {entry_idx, s};
            }

            // Joins: extend KV when a slot's valid length lags this call's position,
            // and independently seed the complete GDN state when a slot is newly
            // assigned or rolled back (including position zero). KV comes from the
            // resident device copy when one is current (post solo decode), else from
            // the host bytes (post prefill). GDN state uses the same source rule,
            // except a host-authoritative holder forces the host source.
            std::vector<char> bounce;
            for (int i = 0; i < n_seqs; i++)
            {
                const int s = slot_of[i];
                QabSlot& sl = e.slots[s];
                const std::int64_t pos = positions[i];
                if (sl.len > pos)
                {
                    // A position rollback cannot reuse the arena: KV rows could keep
                    // their prefix, but the GDN recurrent state at `pos` is NOT a
                    // prefix of the state at `len` (recurrence has no rewind — the
                    // glm5next/KDA rule applies here too). Rejoin from scratch off
                    // the resident/host source, which the engine has already rolled
                    // back through its own solo paths.
                    sl.len = 0;
                    sl.clean = 0;
                    sl.state_seeded = false;
                    sl.state_dirty = false;
                }
                const bool needs_kv_join = sl.len < pos;
                const bool needs_state_seed = !sl.state_seeded || needs_kv_join;
                if (needs_kv_join)
                {
                    const std::int64_t from = sl.len;
                    const std::size_t bytes = static_cast<std::size_t>(pos - from) * row_bytes;
                    for (int l = 0, al = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent != 0) continue;
                        const void* kh = sl.k_hosts[al];
                        const void* vh = sl.v_hosts[al];
                        const std::size_t cache_total = static_cast<std::size_t>(sl.cache_rows) * kvH * row_bytes;
                        bounce.resize(bytes);
                        for (int which = 0; which < 2; which++)
                        {
                            const void* host = (which == 0) ? kh : vh;
                            ggml_tensor* arena = (which == 0) ? e.k_arena[l] : e.v_arena[l];
                            for (int h = 0; h < kvH; h++)
                            {
                                const std::size_t src_off =
                                    (static_cast<std::size_t>(h) * sl.cache_rows + static_cast<std::size_t>(from)) * row_bytes;
                                if (!qab_read_source(host, src_off, bytes, cache_total, bounce.data(), false))
                                {
                                    set_last_error("Qwen3.5 arena batched decode: KV join source read failed.");
                                    return 0;
                                }
                                const std::int64_t arow = static_cast<std::int64_t>(s) * e.rows_per_slot +
                                                          static_cast<std::int64_t>(h) * e.cap + from;
                                ggml_backend_tensor_set(arena, bounce.data(),
                                    static_cast<std::size_t>(arow) * row_bytes, bytes);
                            }
                        }
                        al++;
                    }
                    sl.len = pos;
                    sl.clean = pos;
                }
                // Recurrent state is not a KV row prefix. A newly assigned (or
                // rolled-back) slot must seed its complete conv/delta state even at
                // position zero, where there are no KV rows to join. Otherwise the
                // first decode reuses the previous occupant's arena state.
                if (needs_state_seed)
                {
                    const bool force_host = gdn_host_auth[i] != 0;
                    bounce.resize(std::max(convBytes, deltaBytes));
                    for (int l = 0, gl = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        const void* ch = sl.conv_hosts[gl];
                        const void* dh = sl.delta_hosts[gl];
                        if (!qab_read_source(ch, 0, convBytes, convBytes, bounce.data(), force_host))
                        {
                            set_last_error("Qwen3.5 arena batched decode: conv state join read failed.");
                            return 0;
                        }
                        ggml_backend_tensor_set(e.conv_arena[l], bounce.data(),
                            static_cast<std::size_t>(s) * convBytes, convBytes);
                        if (!qab_read_source(dh, 0, deltaBytes, deltaBytes, bounce.data(), force_host))
                        {
                            set_last_error("Qwen3.5 arena batched decode: delta state join read failed.");
                            return 0;
                        }
                        ggml_backend_tensor_set(e.delta_arena[l], bounce.data(),
                            static_cast<std::size_t>(s) * deltaBytes, deltaBytes);
                        gl++;
                    }
                    sl.state_seeded = true;
                }
            }

        }

        // ---- per-step inputs (slot order) ----
        std::vector<std::int64_t> pos_by_slot(n_slots, -1);
        std::fill(e.tok_stage.begin(), e.tok_stage.end(), 0);
        std::fill(e.hidden_stage.begin(), e.hidden_stage.end(), 0.0f);
        std::fill(e.pos_stage.begin(), e.pos_stage.end(), 0);
        for (int s = 0; s < n_slots; s++)
            for (int h = 0; h < kvH; h++)
                e.idx_stage[static_cast<std::size_t>(s) * kvH + h] =
                    static_cast<std::int64_t>(n_slots) * e.rows_per_slot +
                    static_cast<std::int64_t>(s) * kvH + h;   // unique discard row
        for (int i = 0; i < n_seqs; i++)
        {
            const int s = slot_of[i];
            pos_by_slot[s] = positions[i];
            e.tok_stage[s] = token_ids[i];
            if (hidden_input)
                std::memcpy(e.hidden_stage.data() + static_cast<std::size_t>(s) * H,
                    embedding_rows + static_cast<std::size_t>(i) * H,
                    static_cast<std::size_t>(H) * sizeof(float));
            e.pos_stage[s] = rope_positions != nullptr ? rope_positions[i] : positions[i];
            for (int h = 0; h < kvH; h++)
                e.idx_stage[static_cast<std::size_t>(s) * kvH + h] = bound
                    ? static_cast<std::int64_t>(h) * e.bound_cache_rows[s] + positions[i]
                    : static_cast<std::int64_t>(s) * e.rows_per_slot +
                      static_cast<std::int64_t>(h) * e.cap + positions[i];
        }

        host_read_barrier();
        if (hidden_input)
            decode_input_set_async(e.hidden_in, e.hidden_stage.data(), e.hidden_stage.size() * sizeof(float));
        else
            decode_input_set_async(e.token_in, e.tok_stage.data(), e.tok_stage.size() * sizeof(std::int32_t));
        decode_input_set_async(e.pos_in, e.pos_stage.data(), e.pos_stage.size() * sizeof(std::int32_t));
        decode_input_set_async(e.idx_in, e.idx_stage.data(), e.idx_stage.size() * sizeof(std::int64_t));
        qab_fill_mask(e.mask_stage, e.mask_len, n_slots, pos_by_slot.data());
        decode_input_set_async(e.attn_mask, e.mask_stage.data(), e.mask_stage.size() * sizeof(ggml_fp16_t));

        compute_started = true;
        ggml_status st = tsg::graph_compute_profiled(g_backend, e.graph, kQabKernel);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 arena batched decode: graph execution failed.");
            // A partially executed graph may have advanced SOME layers' GDN
            // state for this call's slots and not others — that mixed state is
            // unrecoverable. Dropping without a state flush avoids overwriting
            // host memory with that mixed state, but the host may also predate
            // earlier successful arena steps. Return the distinct fail-closed
            // status -1: managed code must fail the affected sequences instead
            // of re-serving this token through a stale serial recurrent state.
            // Their pre-step KV rows [clean,len) remain coherent and are flushed.
            for (int i = 0; i < n_seqs && !bound; i++)
            {
                QabSlot& sl = e.slots[slot_of[i]];
                sl.state_dirty = false;
            }
            qab_flush_entry(e);
            e.release_graph();
            e.slots.clear();
            return -1;
        }

        const bool need_logits = (want_logits != 0) || (sampled_data != nullptr && !e.has_argmax);
        if (need_logits)
            finalize_compute_with_download(e.logits_out, e.logits_stage.data(),
                                           e.logits_stage.size() * sizeof(float));
        if (sampled_data != nullptr && e.has_argmax)
            finalize_compute_with_download(e.sampled_out, e.sampled_stage.data(),
                                           e.sampled_stage.size() * sizeof(std::int32_t));
        host_read_barrier();

        if (want_logits != 0 && logits_data != nullptr)
        {
            float* dst = static_cast<float*>(logits_data);
            for (int i = 0; i < n_seqs; i++)
            {
                const int s = slot_of[i];
                std::memcpy(dst + static_cast<std::size_t>(i) * vocab_size,
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

        for (int i = 0; i < n_seqs && !bound; i++)
        {
            QabSlot& sl = e.slots[slot_of[i]];
            sl.len = positions[i] + 1;
            sl.state_dirty = true;
            sl.last_used = ++pool.clock;
        }

        clear_last_error();
        return 1;
    }
    // Download/backend exceptions after submission have the same unsafe
    // recurrent-state contract as a failed graph status: never retry serially.
    catch (const std::exception& ex) { set_last_error(ex.what()); return compute_started ? -1 : 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 arena batched decode."); return compute_started ? -1 : 0; }
}

// Preserve the original ABI for libraries/callers using device GET_ROWS. The
// sibling adds host-gathered embeddings without changing any cache/state ABI.
TSG_EXPORT int TSGgml_Qwen35ArenaDecodeBatched(
    const TSGgmlQwen35LayerDesc* layers, int num_layers, int n_seqs,
    const std::int32_t* token_ids, const std::int32_t* positions,
    const std::int32_t* rope_positions,
    void** k_cache_arr, void** v_cache_arr,
    void** conv_state_arr, void** delta_state_arr,
    const std::int32_t* gdn_host_auth, const std::int32_t* cache_sizes,
    int num_heads, int num_kv_heads, int head_dim,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    const void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1, std::int64_t token_embd_bytes,
    std::int32_t* sampled_data, int want_logits)
{
    return qab_decode_batched(layers, num_layers, n_seqs, token_ids, positions, rope_positions,
        k_cache_arr, v_cache_arr, conv_state_arr, delta_state_arr, gdn_host_auth, cache_sizes,
        num_heads, num_kv_heads, head_dim, rope_n_dims, rope_mode, kv_cache_type,
        conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
        eps, rope_base, rope_freq_scale,
        num_experts, num_experts_used, expert_ff, shared_ff, norm_topk, expert_weights_scale,
        logits_data, vocab_size, lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
        final_norm_data, token_embd_data, token_embd_type, token_embd_ne0, token_embd_ne1, token_embd_bytes,
        sampled_data, want_logits, nullptr);
}

// embedding_rows is F32 [n_seqs, hidden_size] in CALLER sequence order, not
// arena-slot order. The stable staging buffer owns the asynchronous upload.
// token_embd_ne0/ne1 retain the hidden/vocabulary dimensions; its pointer,
// storage type, and byte count are unused by this entry point.
TSG_EXPORT int TSGgml_Qwen35ArenaDecodeBatchedHidden(
    const TSGgmlQwen35LayerDesc* layers, int num_layers, int n_seqs,
    const std::int32_t* token_ids, const std::int32_t* positions,
    const std::int32_t* rope_positions,
    void** k_cache_arr, void** v_cache_arr,
    void** conv_state_arr, void** delta_state_arr,
    const std::int32_t* gdn_host_auth, const std::int32_t* cache_sizes,
    int num_heads, int num_kv_heads, int head_dim,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    const void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1, std::int64_t token_embd_bytes,
    std::int32_t* sampled_data, int want_logits, const float* embedding_rows)
{
    if (embedding_rows == nullptr)
    {
        set_last_error("Qwen3.5 arena batched decode: host embedding rows missing.");
        return 0;
    }
    return qab_decode_batched(layers, num_layers, n_seqs, token_ids, positions, rope_positions,
        k_cache_arr, v_cache_arr, conv_state_arr, delta_state_arr, gdn_host_auth, cache_sizes,
        num_heads, num_kv_heads, head_dim, rope_n_dims, rope_mode, kv_cache_type,
        conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
        eps, rope_base, rope_freq_scale,
        num_experts, num_experts_used, expert_ff, shared_ff, norm_topk, expert_weights_scale,
        logits_data, vocab_size, lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
        final_norm_data, token_embd_data, token_embd_type, token_embd_ne0, token_embd_ne1, token_embd_bytes,
        sampled_data, want_logits, embedding_rows);
}
