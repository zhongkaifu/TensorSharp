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
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_transformer_common.h"
#include "ggml_ops_qwen4exp_qsa.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tsg;

// ============================================================================
// qwen4exp (Qwen3.8-Flash-Next) SLOT-STABLE ARENA token-batched decode: N
// concurrent sequences, one token each, in ONE ggml graph — the port of the
// Qwen3.5 arena design (ggml_ops_qwen35_batched_arena.cpp) to the hybrid
// GDN + full-attention + hyper-connections + PLE family.
//
// Without this, N>=2 requests round-robin N solo token-span graphs per step:
// N full weight sweeps per token, so aggregate throughput is flat in the
// concurrency. The layer math is NOT re-derived here: every half-layer is
// composed from the SAME q4e_nodes_{ffn,gdn,attn,ple,head} builders the solo
// span uses (ggml_ops_qwen4exp.cpp), so solo and arena share one source of
// truth. Per-family differences from the qwen35 arena:
//   * The residual is hyper-connections wide: [hc_dim = hc * n_embd, n_slots].
//     The graph starts from an in-graph get_rows(token_embd) broadcast to the
//     hc streams (the managed BroadcastToStreams, on the device).
//   * ATTENTION layers: per-layer persistent K/V arena
//     [head_dim, cap, kv_heads, n_slots+1] (head-plane-major, IDENTICAL to
//     this family's host cache layout). Each lane writes and reads its own
//     slice through explicit graph dependencies, with the same padded KV
//     window its solo decode reads. All lanes execute in the shared graph.
//   * GDN layers run per-slot with T=1 through q4e_nodes_gdn against
//     conv/ssm ARENA slices ([hist, conv_dim, n_slots+1] /
//     [head_v, head_v, v_heads, n_slots+1]); the per-slot residual columns are
//     reassembled into the batched residual for the (batched) FFN half.
//   * The PLE layer consumes the host-gathered rows input [n_embd, n_slots]
//     (the n-gram hash + table gather stays managed, PEEKED — the caller
//     commits only after success) and a per-slot history arena
//     [hc_dim, hist, n_slots+1] updated in-graph.
//   * Dual positions: kv_idx / the mask use the CACHE position, RoPE uses the
//     (possibly image-compacted) ROPE position — two per-slot inputs.
//
// COHERENCE: while a sequence occupies a slot, its newest KV rows exist ONLY
// in the arenas and its GDN/PLE recurrent truth ONLY in the arena slices. KV
// flushes to the HOST bytes (then the resident cacheable copies are
// invalidated and the holder's captured solo graphs dropped —
// q4e_drop_holder_graphs — because they bake those buffers). The recurrent
// state flushes DEVICE-TO-DEVICE into the per-sequence seq-state buffers
// (g_q4e_seq_state entries keyed on the holder's host seed pointers): that map
// is device-authoritative from the first forward, so no host trip, no
// invalidation, and `ready` stays as it is. JOIN is the mirror image: KV from
// the resident device copy when one exists (the truth after solo decode), else
// from the host bytes; GDN/PLE device-to-device from the seq-state buffer, and
// a MISSING map entry declines the whole call (prefill-through-span creates
// them; fabricating recurrent state here would be wrong by construction).
//
// Eligibility: one CPU, Metal or CUDA device, F16 KV, folded head descriptors
// and no-wrap positions. All lanes retain the solo flash-attention geometry.
// CPU exercises the same cache ownership path. Managed code gates speculative
// sessions and tensor/layer splits. Failures after compute starts invalidate
// affected holders rather than retrying against partly advanced state.
// ============================================================================
namespace
{
    constexpr int kQ4abMaxEntries = 4;
    constexpr const char* kQ4abKernel = "qwen4exp arena batched decode";

    int q4ab_slot_bucket(int n)
    {
        // A padded recurrent lane still evaluates every hybrid layer and its
        // host experts. Match the live width instead of decoding dummy tokens.
        return n;
    }

    std::int64_t q4ab_cap_round(std::int64_t rows)
    {
        const std::int64_t q = 256;
        return ((rows + q - 1) / q) * q;
    }

    struct Q4abSlot
    {
        bool active = false;
        // A global weight/cache eviction must retire its graph even when a
        // state flush fails. Keep this holder's existing registration as a
        // fence until reset/free; its discarded arena truth cannot be retried.
        bool state_failed = false;
        const void* key = nullptr;              // first attention layer's K host pointer
        std::vector<const void*> k_hosts;       // per attention layer
        std::vector<const void*> v_hosts;
        std::vector<const void*> conv_keys;     // per GDN layer: seq-state map key (+ !ready host seed)
        std::vector<const void*> ssm_hosts;     // per GDN layer: !ready host seed only
        std::vector<const void*> qsa_keys;      // per layer, null without QSA
        const void* ple_key = nullptr;          // seq-state map key (+ !ready host seed)
        // Holder descriptor bases, for q4e_drop_holder_graphs at flush.
        const void* attn_desc = nullptr;
        const void* gdn_desc = nullptr;
        const void* ple_desc = nullptr;
        int cache_rows = 0;                     // host K/V pitch in rows per head plane
        std::int64_t len = 0;                   // arena KV rows [0, len) valid
        std::int64_t clean = 0;                 // KV rows [clean, len) not yet in host
        bool state_dirty = false;               // arena GDN/PLE state newer than the seq-state bufs
        std::uint64_t last_used = 0;
    };

    struct Q4abEntry
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_backend_t precise_backend = nullptr;
        ggml_tensor* token_in = nullptr;        // I32 [n_slots]
        ggml_tensor* embedding_in = nullptr;    // optional host-gathered F32 [H, n_slots]
        ggml_tensor* local_idx_in = nullptr;    // I64 [n_slots], holder-relative row
        ggml_tensor* rope_pos_in = nullptr;     // I32 [n_slots] (RoPE position, not cache row)
        ggml_tensor* idx_in = nullptr;          // I64 [kvH * n_slots] set_rows targets (cache rows)
        ggml_tensor* attn_mask = nullptr;       // F16 [cap, 1, 1, n_slots]
        ggml_tensor* ple_emb_in = nullptr;      // F32 [n_embd, n_slots], null without PLE
        ggml_tensor* logits_out = nullptr;      // [vocab, n_slots]
        ggml_tensor* sampled_out = nullptr;     // I32 [n_slots]
        bool has_argmax = false;
        std::vector<ggml_tensor*> k_arena;      // per layer, 2D [hd, (n_slots+1)*kvH*cap]; null on GDN layers
        std::vector<ggml_tensor*> v_arena;
        std::vector<ggml_tensor*> conv_arena;   // per layer [hist, conv_dim, n_slots+1]; null on attention layers
        std::vector<ggml_tensor*> ssm_arena;    // per layer [head_v, head_v, v_heads, n_slots+1]
        ggml_tensor* ple_arena = nullptr;       // [hc_dim, ple_hist, n_slots+1], null without PLE state
        std::vector<ggml_tensor*> qsa_arena;
        std::vector<TSGgmlQwen4ExpQsaArgs> qsa_args;
        std::vector<std::vector<Q4eQsaInputs>> qsa_inputs;
        std::vector<tsg::HostMoeSegment> host_moe;
        std::vector<int> host_moe_seg_end;
        std::vector<ggml_tensor*> trace_res;
        const void* sig_disc = nullptr;
        int num_layers = 0, H = 0, vocab = 0, n_slots = 0;
        int hd = 0, kvH = 0, hc = 0;
        int conv_dim = 0, gdn_hist = 0, head_v = 0, v_heads = 0;
        int ple_hist = 0, hc_dim = 0;
        std::int64_t cap = 0;
        std::int64_t rows_per_slot = 0;         // kvH * cap
        ggml_type kv_type = GGML_TYPE_F16;
        std::vector<Q4abSlot> slots;
        std::vector<int> attention_windows;

        std::vector<std::int32_t> tok_stage;
        std::vector<std::int32_t> pos_stage;
        std::vector<std::int64_t> idx_stage;
        std::vector<std::int64_t> local_idx_stage;
        std::vector<float> embedding_stage;
        std::vector<ggml_fp16_t> mask_stage;
        std::vector<float> ple_stage;
        std::vector<float> logits_stage;
        std::vector<std::int32_t> sampled_stage;

        void release_graph()
        {
            if (precise_backend != nullptr)
            {
                ggml_backend_synchronize(precise_backend);
                ggml_backend_free(precise_backend);
                precise_backend = nullptr;
            }
            if (alloc != nullptr) { tsg::graph_budget_gallocr_free(alloc); alloc = nullptr; }
            buffer = nullptr;
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            token_in = embedding_in = local_idx_in = rope_pos_in = idx_in = attn_mask = ple_emb_in = nullptr;
            logits_out = sampled_out = ple_arena = nullptr;
            has_argmax = false;
            k_arena.clear(); v_arena.clear(); conv_arena.clear(); ssm_arena.clear();
            qsa_arena.clear(); qsa_args.clear(); qsa_inputs.clear();
            host_moe.clear(); host_moe_seg_end.clear();
            trace_res.clear();
            valid = false;
        }

        ggml_backend_t execution_backend() const
        {
            return precise_backend != nullptr ? precise_backend : g_backend;
        }
    };

    struct Q4abPool
    {
        Q4abEntry entries[kQ4abMaxEntries];
        std::uint64_t used[kQ4abMaxEntries] = {};
        std::uint64_t clock = 0;
    };
    Q4abPool g_q4ab_pools[tsg::TSG_MAX_DEVICES];
    Q4abPool& q4ab_pool() { return g_q4ab_pools[tsg::g_active_rank]; }

    bool q4ab_has_failed_slot(const Q4abEntry& e)
    {
        for (const auto& sl : e.slots)
            if (sl.active && sl.state_failed) return true;
        return false;
    }

    // host pointer (any registered cache/state pointer of a slot) -> (entry, slot)
    using Q4abRegistry = std::unordered_map<const void*, std::pair<int, int>>;
    Q4abRegistry g_q4ab_registries[tsg::TSG_MAX_DEVICES];
    Q4abRegistry& q4ab_registry() { return g_q4ab_registries[tsg::g_active_rank]; }

    std::mutex& q4ab_mutex()
    {
        static std::mutex m;
        return m;
    }

    thread_local bool g_q4ab_guard = false;
    struct Q4abGuard
    {
        bool prev;
        Q4abGuard() : prev(g_q4ab_guard) { g_q4ab_guard = true; }
        ~Q4abGuard() { g_q4ab_guard = prev; }
    };

    void q4ab_unregister_slot(Q4abSlot& sl)
    {
        Q4abRegistry& reg = q4ab_registry();
        for (const void* p : sl.k_hosts) reg.erase(p);
        for (const void* p : sl.v_hosts) reg.erase(p);
        for (const void* p : sl.conv_keys) reg.erase(p);
        for (const void* p : sl.ssm_hosts) reg.erase(p);
        for (const void* p : sl.qsa_keys) if (p) reg.erase(p);
        if (sl.ple_key != nullptr) reg.erase(sl.ple_key);
        sl = Q4abSlot{};
    }

    // Device-to-device blob copy between two backend buffers on the active
    // rank. ggml_backend_tensor_copy on same-buft CUDA buffers is a device
    // memcpy; the CALLER drains the compute stream first (the copy rides the
    // legacy stream — the documented span-replay race).
#ifdef TSG_GGML_TEST_HOOKS
    std::atomic<int> g_q4ab_copy_fault{0};
#endif

    bool q4ab_dev_copy(ggml_backend_buffer_t src_buf, void* src_addr,
                       ggml_backend_buffer_t dst_buf, void* dst_addr, std::size_t bytes)
    {
#ifdef TSG_GGML_TEST_HOOKS
        int remaining = g_q4ab_copy_fault.load(std::memory_order_relaxed);
        while (remaining > 0 && !g_q4ab_copy_fault.compare_exchange_weak(
            remaining, remaining - 1, std::memory_order_relaxed)) { }
        if (remaining == 1) return false;
#endif
        if (src_buf == nullptr || dst_buf == nullptr || bytes == 0) return false;
        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true };
        ggml_context* tmp = ggml_init(ip);
        if (tmp == nullptr) return false;
        ggml_tensor* s = ggml_new_tensor_1d(tmp, GGML_TYPE_I8, static_cast<std::int64_t>(bytes));
        ggml_tensor* d = ggml_new_tensor_1d(tmp, GGML_TYPE_I8, static_cast<std::int64_t>(bytes));
        bool ok = s != nullptr && d != nullptr &&
            ggml_backend_tensor_alloc(src_buf, s, src_addr) == GGML_STATUS_SUCCESS &&
            ggml_backend_tensor_alloc(dst_buf, d, dst_addr) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_copy(s, d);
        ggml_free(tmp);
        return ok;
    }

    struct Q4abFlushFailure : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    // Flush a slot's arena-held truth and retire it:
    //   * KV rows [clean, len) per attention layer to the HOST bytes (host
    //     layout matches the arena head-plane-major layout at the holder's own
    //     cache_rows pitch), then invalidate the resident cacheable copies of
    //     the rewritten pointers and drop the holder's captured solo graphs
    //     (they bake the just-freed resident buffers).
    //   * GDN conv/ssm + PLE history DEVICE-TO-DEVICE into the seq-state
    //     buffers when dirty — the seq-state map is the solo path's state
    //     truth, so no host write, no invalidation, `ready` untouched. A slot
    //     that joined a !ready entry made it ready at join time, so the
    //     invariant "ready => buffer holds the current state" survives.
    // q4ab_mutex held; the active rank owns the entry.
    void q4ab_flush_and_drop_slot(Q4abEntry& e, int slot_idx)
    {
        Q4abSlot& sl = e.slots[slot_idx];
        if (!sl.active) return;
        if (sl.state_failed)
            throw Q4abFlushFailure("qwen4exp arena: state lost during global cache reset; reset this conversation before reuse");
        Q4abGuard guard;
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
                    al++;
                }
            }
            if (sl.state_dirty)
            {
                const std::size_t convBytes = static_cast<std::size_t>(e.gdn_hist) * e.conv_dim * sizeof(float);
                const std::size_t ssmBytes = static_cast<std::size_t>(e.head_v) * e.head_v * e.v_heads * sizeof(float);
                const std::size_t ssm_off = (convBytes + 255) & ~static_cast<std::size_t>(255);
                const std::size_t pleBytes = static_cast<std::size_t>(e.ple_hist) * e.hc_dim * sizeof(float);
                // The copies ride the legacy stream; drain compute first.
                ggml_backend_synchronize(g_backend);
                for (int l = 0, gl = 0; l < e.num_layers; l++)
                {
                    if (e.conv_arena[l] == nullptr) continue;
                    Q4eSeqStateEntry* st = q4e_seq_state_find(sl.conv_keys[gl]);
                    if (st == nullptr || st->buf == nullptr || st->bytes < ssm_off + ssmBytes)
                        throw Q4abFlushFailure("qwen4exp arena: GDN flush state missing");
                    char* base = static_cast<char*>(ggml_backend_buffer_get_base(st->buf));
                    if (!q4ab_dev_copy(e.buffer,
                            static_cast<char*>(e.conv_arena[l]->data) + static_cast<std::size_t>(slot_idx) * convBytes,
                            st->buf, base, convBytes)
                        || !q4ab_dev_copy(e.buffer,
                            static_cast<char*>(e.ssm_arena[l]->data) + static_cast<std::size_t>(slot_idx) * ssmBytes,
                            st->buf, base + ssm_off, ssmBytes))
                        throw Q4abFlushFailure("qwen4exp arena: GDN flush copy failed");
                    gl++;
                }
                if (e.ple_arena != nullptr && sl.ple_key != nullptr && pleBytes > 0)
                {
                    Q4eSeqStateEntry* st = q4e_seq_state_find(sl.ple_key);
                    if (st == nullptr || st->buf == nullptr || st->bytes < pleBytes)
                        throw Q4abFlushFailure("qwen4exp arena: PLE flush state missing");
                    if (!q4ab_dev_copy(e.buffer,
                            static_cast<char*>(e.ple_arena->data) + static_cast<std::size_t>(slot_idx) * pleBytes,
                            st->buf, ggml_backend_buffer_get_base(st->buf), pleBytes))
                        throw Q4abFlushFailure("qwen4exp arena: PLE flush copy failed");
                }
                for (int l = 0; l < e.num_layers; ++l)
                {
                    if (!e.qsa_arena[l]) continue;
                    const auto& a = e.qsa_args[l];
                    const size_t row = ggml_row_size((ggml_type)a.cache_type, a.head_dim);
                    auto* st = q4e_seq_state_find(sl.qsa_keys[l]);
                    if (!st || !st->buf || st->bytes < (size_t)sl.len * row)
                        throw Q4abFlushFailure("qwen4exp arena: QSA flush state missing");
                    if (!q4ab_dev_copy(e.buffer,
                        (char*)e.qsa_arena[l]->data + (size_t)slot_idx * e.cap * row,
                        st->buf, ggml_backend_buffer_get_base(st->buf), (size_t)sl.len * row))
                        throw Q4abFlushFailure("qwen4exp arena: QSA flush copy failed");
                }
            }
        }
        if (freed_any)
        {
            // Captured solo span / per-layer graphs bake the just-freed
            // resident KV buffers into their nodes; drop them so the next solo
            // step rebuilds against the re-uploaded copies. refresh_bindings'
            // 1/32 sampling is NOT protection against this.
            q4e_drop_holder_graphs(sl.attn_desc, sl.gdn_desc, sl.ple_desc);
        }
        q4ab_unregister_slot(sl);
    }

    void q4ab_flush_entry(Q4abEntry& e)
    {
        for (int s = 0; s < static_cast<int>(e.slots.size()); s++)
            q4ab_flush_and_drop_slot(e, s);
    }

    // Causal mask fill over [cap, n_slots]: column s zero on [0, pos_s], -inf
    // beyond. Padded/absent columns attend row 0 only.
    void q4ab_fill_mask(std::vector<ggml_fp16_t>& mask, std::int64_t cap, int n_slots,
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
    // from the host bytes themselves (post prefill / a fresh holder). The
    // probe never allocates or evicts. A resident read is also MIRRORED back
    // into the host bytes so that, from the join onward, the host mirror is
    // current up to the joined position — the flush later appends only
    // [clean, len).
    bool q4ab_read_source(const void* host, std::size_t offset, std::size_t bytes,
                          std::size_t total_bytes, void* dst)
    {
        if (host == nullptr) return false;
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
                // A resident copy is authoritative. Allocation failure while
                // reading it cannot fall back to the stale host seed.
                return false;
            }
        }
        std::memcpy(dst, static_cast<const char*>(host) + offset, bytes);
        return true;
    }
}

// ---------------------------------------------------------------------------
// Coherence hooks (see the header comment). All take q4ab_mutex internally so
// call sites in other translation units stay one-liners. The pools are
// per-rank and the caller's active rank is not necessarily the slot's, so
// touch/drop sweep every initialized device.
// ---------------------------------------------------------------------------
namespace tsg_q4earena
{
    void on_external_touch(const void* host_ptr)
    {
        if (g_q4ab_guard || host_ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(q4ab_mutex());
        const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
        for (int r = 0; r < ndev && r < tsg::TSG_MAX_DEVICES; r++)
        {
            auto it = g_q4ab_registries[r].find(host_ptr);
            if (it == g_q4ab_registries[r].end()) continue;
            tsg::ScopedRank rank(r);
            q4ab_flush_and_drop_slot(g_q4ab_pools[r].entries[it->second.first], it->second.second);
        }
    }

    void on_drop(const void* host_ptr)
    {
        if (g_q4ab_guard || host_ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(q4ab_mutex());
        for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
        {
            auto it = g_q4ab_registries[r].find(host_ptr);
            if (it == g_q4ab_registries[r].end()) continue;
            tsg::ScopedRank rank(r);
            q4ab_unregister_slot(g_q4ab_pools[r].entries[it->second.first].slots[it->second.second]);
        }
    }

    void on_drop_all()
    {
        if (g_q4ab_guard) return;
        std::lock_guard<std::mutex> lock(q4ab_mutex());
        for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
        {
            tsg::ScopedRank rank(r);
            for (auto& e : g_q4ab_pools[r].entries)
                for (auto& sl : e.slots)
                    // A process-global seq-state release does not identify
                    // which managed holders survive. Only a holder-specific
                    // reset/free may forget its lost-state fence.
                    if (!sl.state_failed) q4ab_unregister_slot(sl);
        }
    }
}

// Flush-and-retire the arena slot (if any) registered for `host_ptr` — the
// managed-callable form of on_external_touch, for paths that read or replace a
// holder's caches/state outside the hooked native kernels (cache growth, host
// syncs, state invalidation, holder disposal, residency release).
#ifdef TSG_GGML_TEST_HOOKS
// Fail exactly the nth subsequent arena state copy. Tests reset this hook
// after each scenario; production builds contain neither the export nor branch.
// Keep test exports distinct from the production ABI manifest's TSG_EXPORT scan.
#ifndef TSG_TEST_EXPORT
#define TSG_TEST_EXPORT TSG_EXPORT
#endif
TSG_TEST_EXPORT int TSGgml_Qwen4ExpArenaTestCopyFault(int nth)
{
    if (nth < 0) return 0;
    g_q4ab_copy_fault.store(nth, std::memory_order_relaxed);
    return 1;
}
#undef TSG_TEST_EXPORT
#endif

TSG_EXPORT int TSGgml_Qwen4ExpArenaFlushHostPointerStatus(void* host_ptr)
{
    try
    {
        tsg_q4earena::on_external_touch(host_ptr);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error flushing qwen4exp arena state."); return 0; }
}

// Preserve the older ABI while preventing C++ exceptions from crossing it.
// New callers use the status form and fence the affected holder on failure.
TSG_EXPORT void TSGgml_Qwen4ExpArenaFlushHostPointer(void* host_ptr)
{
    (void)TSGgml_Qwen4ExpArenaFlushHostPointerStatus(host_ptr);
}

// Drop all arena state. Dirty slots flush first (each on its own rank). Like
// the qwen35/gptoss arena pools, this is deliberately NOT chained into
// TSGgml_Qwen4ExpResetFfnCache — the span cache churns per holder swap and
// this pool's survival across those swaps is the point. Teardown paths in
// ggml_ops_core.cpp and the managed dispose call it explicitly.
TSG_EXPORT void TSGgml_Qwen4ExpArenaResetBatchedDecodeCache()
{
    try
    {
        std::lock_guard<std::mutex> lock(q4ab_mutex());
        const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
        bool failed = false;
        for (int r = 0; r < tsg::TSG_MAX_DEVICES; r++)
        {
            tsg::ScopedRank rank(r);
            for (int i = 0; i < kQ4abMaxEntries; i++)
            {
                Q4abEntry& e = g_q4ab_pools[r].entries[i];
                // A failed slot must not abort retirement: the caller will
                // free the resident weights after this void ABI returns.
                // Flush other holders independently and preserve the failed
                // registration without allocating a separate poison map.
                for (int s = 0; s < static_cast<int>(e.slots.size()); ++s)
                {
                    Q4abSlot& sl = e.slots[s];
                    if (!sl.active) continue;
                    if (sl.state_failed) { failed = true; continue; }
                    try
                    {
                        if (r < ndev)
                            q4ab_flush_and_drop_slot(e, s);
                        else if (sl.state_dirty || sl.len > sl.clean)
                            throw Q4abFlushFailure("qwen4exp arena: backend unavailable during state flush");
                        else
                            q4ab_unregister_slot(sl);
                    }
                    catch (...)
                    {
                        sl.state_failed = true;
                        failed = true;
                        // KV copies may already have been invalidated before
                        // the failing recurrent-state copy. Retire captured
                        // solo graphs as well as the shared arena graph.
                        q4e_drop_holder_graphs(sl.attn_desc, sl.gdn_desc, sl.ple_desc);
                    }
                }
                e.release_graph();
                if (!q4ab_has_failed_slot(e)) e.slots.clear();
            }
        }
        if (failed)
            set_last_error("qwen4exp arena: state flush failed during global cache reset; reset affected conversations before reuse");
        else
            clear_last_error();
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); }
    catch (...) { set_last_error("Unknown error resetting qwen4exp arena state."); }
}

// ============================================================================
// The batched arena step. See the ABI notes: the ffn/gdn/attn/head/ple
// descriptor bases come from ONE warmed holder and only their WEIGHT fields
// are read — per-sequence cache/state comes from the flat arrays
// ([attn_layer * n + s] / [gdn_layer * n + s] / [s]). Outputs are in the
// caller's sequence order.
// ============================================================================
TSG_EXPORT int TSGgml_Qwen4ExpArenaDecodeBatched(
    const TSGgmlQwen4ExpFfnArgs* ffn,
    const TSGgmlQwen4ExpGdnArgs* gdn,
    const TSGgmlQwen4ExpAttnArgs* attn,
    const TSGgmlQwen4ExpHeadArgs* head,
    const TSGgmlQwen4ExpPleArgs* ple,
    const unsigned char* kinds, int n_layers, int ple_layer,
    int n_seqs,
    const std::int32_t* token_ids,
    const std::int32_t* cache_positions,
    const std::int32_t* rope_positions,
    const std::int32_t* cache_rows,
    void** k_cache_arr, void** v_cache_arr,
    void** conv_state_arr, void** ssm_state_arr, void** ple_state_arr,
    void** attn_desc_arr, void** gdn_desc_arr, void** ple_desc_arr,
    const TSGgmlQwen4ExpQsaArgs* const* qsa_arr,
    const std::int32_t* const* qsa_positions_arr, const std::int32_t* qsa_position_counts,
    int n_embd, int hc, int hc_low_rank,
    int head_dim, int n_head, int n_head_kv, int n_rot,
    float rope_base, float rope_freq_scale, float attn_scale,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh,
    float eps, int kv_cache_type,
    const void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1, std::int64_t token_embd_bytes,
    const float* embedding_rows, const float* ple_emb,
    float* logits_data, std::int32_t* sampled_data, int want_logits,
    int device)
{
    bool compute_started = false;
    try
    {
        tsg::ScopedRank q4ab_rank(q4e_resolve_device(device));
        if (!ensure_backend())
            return 0;
        if (g_backend_type != BACKEND_TYPE_CUDA && g_backend_type != BACKEND_TYPE_METAL &&
            !ggml_backend_is_cpu(g_backend))
        {
            set_last_error("qwen4exp arena batched decode: CUDA, Metal or CPU required.");
            return 0;
        }
        if (ffn == nullptr || gdn == nullptr || attn == nullptr || head == nullptr ||
            kinds == nullptr || n_layers <= 0 || n_seqs < 2 ||
            token_ids == nullptr || cache_positions == nullptr ||
            rope_positions == nullptr || cache_rows == nullptr ||
            k_cache_arr == nullptr || v_cache_arr == nullptr ||
            conv_state_arr == nullptr || ssm_state_arr == nullptr ||
            attn_desc_arr == nullptr || gdn_desc_arr == nullptr)
        {
            set_last_error("qwen4exp arena batched decode: invalid arguments.");
            return 0;
        }
        const bool has_ple = ple != nullptr;
        const bool trace = std::getenv("TS_Q4E_SPAN_TRACE") && std::getenv("TS_Q4E_SPAN_TRACE")[0] == '1';
        if (has_ple != (ple_layer >= 0 && ple_layer < n_layers) ||
            (has_ple && (ple_emb == nullptr || ple_state_arr == nullptr || ple_desc_arr == nullptr)))
        {
            set_last_error("qwen4exp arena batched decode: inconsistent PLE arguments.");
            return 0;
        }
        if (head->head == nullptr || head->hc_norm == nullptr || head->vocab <= 0 ||
            token_embd_data == nullptr || token_embd_ne0 <= 0 || token_embd_ne1 <= 0 ||
            (logits_data == nullptr && sampled_data == nullptr) ||
            (want_logits != 0 && logits_data == nullptr))
        {
            set_last_error("qwen4exp arena batched decode: head + token embedding required.");
            return 0;
        }
        // Lanes use their solo padded attention windows through flash. Flash
        // eligibility is a hard gate rather than a fork (this also pins F16 KV).
        const bool use_flash = q4e_flash_attn_ok(kv_cache_type, head_dim);
        if (kv_cache_type != GGML_TYPE_F16 || !use_flash)
        {
            set_last_error("qwen4exp arena batched decode: F16 KV + flash-capable head required (v1).");
            return 0;
        }
        if (head_v_dim <= 0 || n_v_heads <= 0 || d_conv <= 1 || hc <= 0 || n_embd <= 0)
        {
            set_last_error("qwen4exp arena batched decode: unsupported GDN geometry.");
            return 0;
        }
        {
            const ggml_type emb_type = static_cast<ggml_type>(token_embd_type);
            const std::int64_t bs = ggml_blck_size(emb_type);
            if (token_embd_type < 0 || token_embd_type >= GGML_TYPE_COUNT ||
                bs <= 0 || token_embd_ne0 % bs != 0)
            {
                set_last_error("qwen4exp arena batched decode: bad token embedding type.");
                return 0;
            }
        }

        int attn_layers = 0, gdn_layers = 0;
        for (int l = 0; l < n_layers; l++)
            (kinds[l] != 0 ? gdn_layers : attn_layers)++;
        if (attn_layers == 0)
        {
            set_last_error("qwen4exp arena batched decode: no attention layers.");
            return 0;
        }

        std::int64_t maxTotal = 0;
        for (int s = 0; s < n_seqs; s++)
        {
            if (token_ids[s] < 0 || token_ids[s] >= token_embd_ne1)
            {
                set_last_error("qwen4exp arena batched decode: token id out of range.");
                return 0;
            }
            if (cache_positions[s] < 0 || cache_positions[s] >= cache_rows[s] || rope_positions[s] < 0)
            {
                set_last_error("qwen4exp arena batched decode: sequence exceeds its cache rows (no-wrap).");
                return 0;
            }
            maxTotal = std::max<std::int64_t>(maxTotal, cache_positions[s] + 1);
            for (int t = 0; t < s; ++t)
                if (k_cache_arr[s] == k_cache_arr[t])
                    throw std::invalid_argument("qwen4exp arena: duplicate sequence cache");
            if (qsa_arr && (!qsa_arr[s] || !qsa_positions_arr || !qsa_positions_arr[s] ||
                !qsa_position_counts || qsa_position_counts[s] != cache_positions[s] + 1))
                throw std::invalid_argument("qwen4exp arena: invalid QSA position history");
            if (qsa_arr) for (int l = 0; l < n_layers; ++l)
            {
                const auto& a = qsa_arr[s][l];
                if (!a.ratio) continue;
                if (kinds[l] || a.ratio < 1 || a.ratio > 64 || a.top_k <= 0 || a.top_k > INT32_MAX - a.ratio
                    || a.head_dim < n_rot || a.head_dim > 4096 || a.heads <= 0 || a.heads > 1024
                    || !a.k_proj || !a.q_proj || !a.k_norm || !a.q_norm || !a.cache
                    || (a.cache_type != GGML_TYPE_F16 && a.cache_type != GGML_TYPE_F32)
                    || a.cache_bytes != (int64_t)ggml_row_size((ggml_type)a.cache_type, a.head_dim) * cache_rows[s])
                    throw std::invalid_argument("qwen4exp arena: invalid QSA descriptor/storage");
                auto matrix_valid = [&](int type, int64_t bytes, int64_t rows) {
                    if (type < 0 || type >= GGML_TYPE_COUNT || bytes <= 0) return false;
                    const auto* tr = ggml_get_type_traits((ggml_type)type);
                    return tr->type_size && tr->blck_size && n_embd % tr->blck_size == 0
                        && bytes == (int64_t)(n_embd / tr->blck_size) * tr->type_size * rows;
                };
                if (!matrix_valid(a.k_type, a.k_bytes, a.head_dim) ||
                    !matrix_valid(a.q_type, a.q_bytes, (int64_t)a.head_dim * a.heads))
                    throw std::invalid_argument("qwen4exp arena: invalid QSA matrix storage");
                int64_t sections = 0;
                for (int section : a.rope_sections) { if (section < 0) throw std::invalid_argument("qwen4exp arena: negative QSA rotary section"); sections += section; }
                if (sections <= 0 || sections > n_rot || n_rot <= 0 || n_rot % 2)
                    throw std::invalid_argument("qwen4exp arena: invalid QSA rotary sections");
            }
        }

        const int H = n_embd;
        const int hc_dim = hc * n_embd;
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        const int hd = head_dim;
        const int kvH = n_head_kv;
        const int key_dim = head_k_dim * n_k_heads;
        const int value_dim = head_v_dim * n_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int gdn_hist = d_conv - 1;
        const std::size_t convBytes = static_cast<std::size_t>(gdn_hist) * conv_dim * sizeof(float);
        const std::size_t ssmBytes = static_cast<std::size_t>(head_v_dim) * head_v_dim * n_v_heads * sizeof(float);
        const std::size_t ssm_off = (convBytes + 255) & ~static_cast<std::size_t>(255);
        const int ple_hist = has_ple ? (ple->kern - 1) * ple->dil : 0;
        const std::size_t pleBytes = static_cast<std::size_t>(ple_hist) * hc_dim * sizeof(float);
        const int vocab_size = head->vocab;
        const int n_slots_req = q4ab_slot_bucket(n_seqs);
        const void* sig_disc = ffn[0].hc_norm;

        std::lock_guard<std::mutex> qlock(q4ab_mutex());
        Q4abGuard guard;

        // ---- entry lookup / build ----
        Q4abPool& pool = q4ab_pool();
        for (int i = 0; i < n_seqs; ++i)
        {
            const auto it = q4ab_registry().find(k_cache_arr[i]);
            if (it != q4ab_registry().end() &&
                pool.entries[it->second.first].slots[it->second.second].state_failed)
                throw Q4abFlushFailure("qwen4exp arena: state lost during global cache reset; reset this conversation before reuse");
        }
        auto window_for = [&](int i) {
            const int64_t rows = (int64_t)cache_positions[i] + 1;
            return (int)std::min<int64_t>(cache_rows[i], use_flash ? q4ab_cap_round(rows) : rows);
        };
        auto matching_windows = [&](const Q4abEntry& e) {
            std::vector<int> assigned(n_seqs, -1);
            std::vector<bool> used(e.n_slots, false);
            for (int i = 0; i < n_seqs; ++i)
                for (int s = 0; s < e.n_slots; ++s)
                    if (e.slots[s].active && e.slots[s].key == k_cache_arr[i]) { assigned[i] = s; used[s] = true; break; }
            for (int i = 0; i < n_seqs; ++i) if (assigned[i] < 0)
                for (int s = 0; s < e.n_slots; ++s) if (!used[s]) { assigned[i] = s; used[s] = true; break; }
            for (int i = 0; i < n_seqs; ++i)
                if (assigned[i] < 0 || e.attention_windows[assigned[i]] != window_for(i)) return false;
            return true;
        };
        int entry_idx = -1;
        for (int i = 0; i < kQ4abMaxEntries; i++)
        {
            Q4abEntry& e = pool.entries[i];
            if (e.valid && e.sig_disc == sig_disc && e.n_slots == n_slots_req &&
                e.num_layers == n_layers && e.H == H && e.vocab == vocab_size &&
                e.hd == hd && e.kvH == kvH && e.kv_type == kvType && e.hc == hc &&
                e.conv_dim == conv_dim && e.gdn_hist == gdn_hist &&
                e.head_v == head_v_dim && e.v_heads == n_v_heads &&
                e.ple_hist == ple_hist)
            {
                bool same_qsa = true;
                for (int l = 0; l < n_layers; ++l)
                    same_qsa &= e.qsa_args[l].ratio == (qsa_arr ? qsa_arr[0][l].ratio : 0);
                if (!same_qsa || (e.embedding_in != nullptr) != (embedding_rows != nullptr)) continue;
                entry_idx = i;
                break;
            }
        }
        const bool needs_build = (entry_idx < 0) || (pool.entries[entry_idx].cap < maxTotal)
            || !matching_windows(pool.entries[entry_idx]);

        if (needs_build)
        {
            if (g_backend_type == BACKEND_TYPE_METAL)
            {
                // These recurrent arenas can retain hundreds of MiB per
                // geometry. On unified memory that competes with the mapped
                // host experts. Keep the matching graph reusable, but retire
                // other empty graphs before allocating a new geometry. Live
                // slots and lost-state fences must remain registered.
                for (int i = 0; i < kQ4abMaxEntries; ++i)
                {
                    auto& unused = pool.entries[i];
                    if (i != entry_idx && unused.valid &&
                        std::none_of(unused.slots.begin(), unused.slots.end(),
                            [](const Q4abSlot& slot) { return slot.active; }))
                    {
                        unused.release_graph();
                        unused.slots.clear();
                    }
                }
            }
            if (entry_idx < 0)
            {
                for (int i = 0; i < kQ4abMaxEntries; i++)
                    if (!pool.entries[i].valid && !q4ab_has_failed_slot(pool.entries[i])) { entry_idx = i; break; }
                if (entry_idx < 0)
                {
                    for (int i = 0; i < kQ4abMaxEntries; i++)
                        if (!q4ab_has_failed_slot(pool.entries[i]) &&
                            (entry_idx < 0 || pool.used[i] < pool.used[entry_idx])) entry_idx = i;
                }
                if (entry_idx < 0)
                {
                    set_last_error("qwen4exp arena: all entries fenced after failed state flush; reset affected conversations");
                    return 0;
                }
            }
            Q4abEntry& e = pool.entries[entry_idx];
            const std::int64_t prev_cap = e.valid ? e.cap : 0;
            q4ab_flush_entry(e);
            e.release_graph();

            e.sig_disc = sig_disc;
            e.num_layers = n_layers;
            e.H = H;
            e.vocab = vocab_size;
            e.n_slots = n_slots_req;
            e.hd = hd;
            e.kvH = kvH;
            e.hc = hc;
            e.hc_dim = hc_dim;
            e.kv_type = kvType;
            e.conv_dim = conv_dim; e.gdn_hist = gdn_hist;
            e.head_v = head_v_dim; e.v_heads = n_v_heads;
            e.ple_hist = ple_hist;
            e.cap = std::max(q4ab_cap_round(maxTotal), prev_cap);
            e.rows_per_slot = static_cast<std::int64_t>(kvH) * e.cap;
            e.slots.assign(e.n_slots, Q4abSlot{});
            e.attention_windows.assign(e.n_slots, std::min<int64_t>(256, e.cap));
            for (int i = 0; i < n_seqs; ++i) e.attention_windows[i] = window_for(i);
            const int n_slots = e.n_slots;
            struct IndependentRows {
                explicit IndependentRows(int rows) { q4e_independent_decode_rows(rows); }
                ~IndependentRows() { q4e_independent_decode_rows(0); }
            } independent_rows(n_slots);

            // The GDN half runs per slot (its recurrence is per-sequence), so
            // node count scales with n_slots; everything else is batched.
            const std::size_t graph_size =
                static_cast<std::size_t>(n_layers) * (192 + 160 * static_cast<std::size_t>(n_slots)) + 32768;
            ggml_init_params ip = {
                ggml_tensor_overhead() * (graph_size + 8192) + ggml_graph_overhead_custom(graph_size, false),
                nullptr, /*no_alloc=*/true };
            ggml_context* ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("qwen4exp arena batched decode: failed to init ggml context.");
                return 0;
            }
            e.ctx = ctx;
            auto abort_build = [&](const char* msg) -> int {
                set_last_error(std::string("qwen4exp arena batched decode: ") + msg);
                e.release_graph();
                return 0;
            };

            const std::int64_t arena_rows = e.rows_per_slot * (n_slots + 1);

            e.token_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_slots);
            e.rope_pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_slots);
            e.idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, static_cast<std::int64_t>(kvH) * n_slots);
            e.local_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_slots);
            e.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, e.cap, 1, 1, n_slots);
            ggml_set_input(e.token_in);
            ggml_set_input(e.rope_pos_in);
            ggml_set_input(e.idx_in);
            ggml_set_input(e.local_idx_in);
            ggml_set_input(e.attn_mask);
            if (has_ple)
            {
                e.ple_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_slots);
                ggml_set_input(e.ple_emb_in);
            }

            e.k_arena.assign(n_layers, nullptr);
            e.v_arena.assign(n_layers, nullptr);
            e.conv_arena.assign(n_layers, nullptr);
            e.ssm_arena.assign(n_layers, nullptr);
            e.qsa_arena.assign(n_layers, nullptr);
            e.qsa_args.assign(n_layers, TSGgmlQwen4ExpQsaArgs{});
            e.qsa_inputs.resize(n_layers);
            for (int l = 0; l < n_layers; l++)
            {
                if (kinds[l] == 0)
                {
                    e.k_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
                    e.v_arena[l] = ggml_new_tensor_2d(ctx, kvType, hd, arena_rows);
                    if (qsa_arr && qsa_arr[0][l].ratio > 0)
                    {
                        e.qsa_args[l] = qsa_arr[0][l];
                        const auto& a = e.qsa_args[l];
                        e.qsa_arena[l] = ggml_new_tensor_3d(ctx, (ggml_type)a.cache_type,
                            a.head_dim, e.cap, n_slots);
                        ggml_set_input(e.qsa_arena[l]);
                        e.qsa_inputs[l].resize(n_slots);
                        for (auto& in : e.qsa_inputs[l])
                        {
                            in.ratio = a.ratio;
                            const int slot = (int)(&in - e.qsa_inputs[l].data());
                            const int64_t padded = e.attention_windows[slot];
                            const int64_t blocks = (padded + a.ratio - 1) / a.ratio;
                            in.cell_blocks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, padded);
                            in.block_cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a.ratio * blocks);
                            in.block_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * blocks);
                            in.query_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
                            in.bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks, 1);
                            for (auto* t : {in.cell_blocks, in.block_cells, in.block_positions, in.query_positions, in.bias})
                                ggml_set_input(t);
                        }
                    }
                }
                else
                {
                    e.conv_arena[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gdn_hist, conv_dim, n_slots + 1);
                    e.ssm_arena[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads, n_slots + 1);
                }
            }
            if (has_ple && ple_hist > 0)
                e.ple_arena = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc_dim, ple_hist, n_slots + 1);
            // These leaves carry state between executions, so graph allocation
            // must never recycle their storage for a later layer's scratch.
            for (auto* tensors : {&e.k_arena, &e.v_arena, &e.conv_arena, &e.ssm_arena, &e.qsa_arena})
                for (auto* t : *tensors) if (t) { ggml_set_input(t); ggml_set_output(t); }
            if (e.ple_arena) { ggml_set_input(e.ple_arena); ggml_set_output(e.ple_arena); }

            ggml_tensor* token_embd_t = embedding_rows ? nullptr : ggml_new_tensor_2d(ctx,
                static_cast<ggml_type>(token_embd_type), token_embd_ne0, token_embd_ne1);

            // ---- graph ----
            e.graph = ggml_new_graph_custom(ctx, graph_size, false);
            Q4eBinder binder{ggml_backend_get_device(g_backend)};

            ggml_tensor* emb = embedding_rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_slots)
                : ggml_get_rows(ctx, token_embd_t, e.token_in);
            if (embedding_rows) { e.embedding_in = emb; ggml_set_input(emb); }
            if (!embedding_rows && !backend_supports_op(emb))
                return abort_build("get_rows unsupported for the token embedding type.");
            // The wide residual starts as hc identical copies of the embedding
            // (the managed BroadcastToStreams, in-graph).
            ggml_tensor* res = ggml_reshape_2d(ctx,
                ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, n_embd, 1, n_slots),
                               n_embd, hc, n_slots, 1),
                hc_dim, n_slots);                                                  // [hc_dim, N]

            std::vector<ggml_tensor*> state_writes;
            state_writes.reserve(static_cast<std::size_t>(gdn_layers) * n_slots * 3 +
                                 static_cast<std::size_t>(attn_layers) * 2 + 4);

            for (int il = 0; il < n_layers; il++)
            {
                if (has_ple && il == ple_layer)
                {
                    ggml_tensor* hist_view = nullptr;
                    int ple_streams = 1;
                    int ple_T = n_slots;
                    if (e.ple_arena != nullptr)
                    {
                        // Per-slot histories: real slots only (lane n_slots is
                        // never read; the buffer clear below keeps it finite).
                        hist_view = ggml_view_3d(ctx, e.ple_arena, hc_dim, ple_hist, n_slots,
                                e.ple_arena->nb[1], e.ple_arena->nb[2], 0);
                        ple_streams = n_slots;
                        ple_T = 1;
                    }
                    res = q4e_nodes_ple(ctx, e.graph, binder, ple, res, e.ple_emb_in,
                            hist_view, n_embd, hc, ple_T, ple_streams, eps, &state_writes);
                }

                if (kinds[il] != 0)
                {
                    // ===== GDN half: per-slot recurrence against arena slices =====
                    ggml_tensor* res_next = nullptr;
                    for (int s = 0; s < n_slots; s++)
                    {
                        ggml_tensor* conv_slice = ggml_view_3d(ctx, e.conv_arena[il],
                                gdn_hist, conv_dim, 1,
                                e.conv_arena[il]->nb[1], e.conv_arena[il]->nb[2],
                                static_cast<std::size_t>(s) * e.conv_arena[il]->nb[2]);
                        ggml_tensor* ssm_slice = ggml_view_3d(ctx, e.ssm_arena[il],
                                head_v_dim, head_v_dim, n_v_heads,
                                e.ssm_arena[il]->nb[1], e.ssm_arena[il]->nb[2],
                                static_cast<std::size_t>(s) * e.ssm_arena[il]->nb[3]);
                        ggml_tensor* res_col = ggml_view_2d(ctx, res, hc_dim, 1,
                                res->nb[1], static_cast<std::size_t>(s) * res->nb[1]);
                        Q4eGdnWriteback wb{};
                        ggml_tensor* out_col = q4e_nodes_gdn(ctx, binder, &gdn[il], res_col,
                                conv_slice, ssm_slice,
                                n_embd, hc, hc_low_rank, /*T=*/1,
                                head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv,
                                eps, &wb, nullptr);
                        // CONCAT gives every consumer an explicit edge to every
                        // producer. Writing columns into a producer-less tensor
                        // lets Metal schedule the next layer before they exist.
                        res_next = res_next ? ggml_concat(ctx, res_next, out_col, 1) : out_col;
                        ggml_build_forward_expand(e.graph, out_col);
                        ggml_build_forward_expand(e.graph, ggml_cpy(ctx, wb.tail, conv_slice));
                        ggml_build_forward_expand(e.graph, ggml_cpy(ctx, wb.new_state, ssm_slice));
                    }
                    res = res_next;
                }
                else
                {
                    ggml_tensor* next = nullptr;
                    const size_t rb = ggml_row_size(kvType, hd);
                    for (int s = 0; s < n_slots; ++s)
                    {
                        auto* col = ggml_cont(ctx, ggml_view_2d(ctx, res, hc_dim, 1, res->nb[1], (size_t)s * res->nb[1]));
                        const int padded = e.attention_windows[s];
                        auto* mask = ggml_view_2d(ctx, e.attn_mask, padded, 1, e.attn_mask->nb[1], (size_t)s * e.attn_mask->nb[3]);
                        auto* pos = ggml_view_1d(ctx, e.rope_pos_in, 1, (size_t)s * sizeof(int32_t));
                        auto* idx = ggml_view_1d(ctx, e.local_idx_in, 1, (size_t)s * sizeof(int64_t));
                        auto* k = ggml_view_3d(ctx, e.k_arena[il], hd, e.cap, kvH, rb, e.cap * rb, (size_t)s * e.rows_per_slot * rb);
                        auto* v = ggml_view_3d(ctx, e.v_arena[il], hd, e.cap, kvH, rb, e.cap * rb, (size_t)s * e.rows_per_slot * rb);
                        Q4eQsaGraph qsa{};
                        if (e.qsa_arena[il])
                        {
                            const auto& a = e.qsa_args[il];
                            auto* raw = e.qsa_arena[il];
                            qsa.args = &a;
                            qsa.inputs = &e.qsa_inputs[il][s];
                            qsa.cache = ggml_view_2d(ctx, raw, a.head_dim, e.cap, raw->nb[1], (size_t)s * raw->nb[2]);
                        }
                        auto* out = q4e_nodes_attn(ctx, e.graph, binder, &attn[il], col, mask, pos, idx,
                            n_embd, hc, hc_low_rank, 1, head_dim, n_head, kvH, (int)e.cap, padded,
                            n_rot, rope_base, rope_freq_scale, attn_scale, eps, use_flash,
                            nullptr, nullptr, nullptr, nullptr, k, v, qsa.args ? &qsa : nullptr);
                        next = next ? ggml_concat(ctx, next, out, 1) : out;
                    }
                    res = next;
                }

                if (trace) { ggml_set_output(res); e.trace_res.push_back(res); }
                // ===== FFN half, batched =====
                res = q4e_nodes_ffn(ctx, binder, &ffn[il], res,
                        n_embd, hc, hc_low_rank, n_slots,
                        n_expert, n_expert_used, n_ff, n_ff_sh, eps,
                        e.graph, &e.host_moe, il);
                if (ffn[il].cpu_moe) e.host_moe.back().independent_decode_rows = true;
                ggml_build_forward_expand(e.graph, res);
                if (trace) { ggml_set_output(res); e.trace_res.push_back(res); }
            }

            // Every slot's token is "last": the head mixer runs with T = n_slots.
            ggml_tensor* logits = q4e_nodes_head(ctx, binder, head, res,
                    n_embd, hc, hc_low_rank, n_slots, eps);
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

            // The state writes have no consumer edge to the logits; expand them
            // explicitly, in build order: each layer's writes precede the next
            // layer's reads within each slot's chain by construction.
            for (ggml_tensor* wnode : state_writes)
            {
                ggml_set_output(wnode);
                ggml_build_forward_expand(e.graph, wnode);
            }
            ggml_build_forward_expand(e.graph, out_cpy);
            if (e.has_argmax)
                ggml_build_forward_expand(e.graph, amax);

#ifdef TSG_GGML_USE_CUDA
            if (ggml_backend_is_cuda(g_backend))
            {
                // QSA starts using owned F32 matmuls when the attention window
                // reaches its sparse selector. Select the same execution
                // backend as the solo span from the completed graph, so dense
                // short-context arenas keep upstream's uninterrupted CUDA graph.
                for (int i = 0; i < ggml_graph_n_nodes(e.graph); ++i)
                {
                    if (ggml_graph_node(e.graph, i)->op != GGML_OP_CUSTOM) continue;
                    e.precise_backend = tsg_dsv4_fused_backend_init(g_backend);
                    if (e.precise_backend == nullptr)
                        return abort_build("precise backend creation failed.");
                    break;
                }
            }
#endif

            if (token_embd_t) binder.add(token_embd_t, const_cast<void*>(token_embd_data),
                static_cast<std::size_t>(token_embd_bytes));
            if (!host_moe_build_segment_ends(e.graph, e.host_moe, e.host_moe_seg_end, kQ4abKernel))
                return abort_build("host expert graph boundaries invalid.");
            SuppressGraphReorder keep_order(!e.host_moe.empty());

            // Everything unbound (inputs, arenas, intermediates, logits) lands
            // in the entry's OWN buffer — nothing touches the shared gallocr
            // pool, which is what lets this graph survive prefills and holder
            // churn (and structurally sidesteps the gallocr leaf-free class).
            // Zero it: fattn reads masked-but-unwritten arena rows, and
            // recycled VRAM decodes as NaN which survives the -inf mask.
            // Allocate only reachable graph tensors and reuse intermediates.
            // alloc_ctx_tensors also allocates the unused stacked-expert tensor
            // descriptors of CPU-offloaded layers, wiring tens of GB on Metal.
            optimize_graph_for_metal(ctx, e.graph);
            e.alloc = tsg::graph_budget_gallocr_new(ggml_backend_get_default_buffer_type(g_backend), tsg::g_active_rank);
            if (!e.alloc || !tsg::graph_budget_gallocr_alloc_graph(e.alloc, e.graph))
                return abort_build("failed to allocate the arena backend buffer.");
            for (auto* tensors : {&e.k_arena, &e.v_arena, &e.conv_arena, &e.ssm_arena, &e.qsa_arena})
                for (auto* t : *tensors)
                    if (t && t->buffer) {
                        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                        e.buffer = t->buffer;
                    }
            if (e.ple_arena && e.ple_arena->buffer)
                ggml_backend_tensor_memset(e.ple_arena, 0, 0, ggml_nbytes(e.ple_arena));

            host_read_barrier();
            binder.flush();

#ifdef TSG_GGML_USE_CUDA
            if (ggml_backend_is_cuda(g_backend))
            {
                // Direct graph execution does not consult supports_op. Reject
                // an unsupported graph before joining slots or advancing any
                // recurrent state, preserving the caller's solo fallback.
                for (int i = 0; i < ggml_graph_n_nodes(e.graph); ++i)
                {
                    const auto* node = ggml_graph_node(e.graph, i);
                    if (ggml_backend_supports_op(e.execution_backend(), node)) continue;
                    const auto message = std::string("backend does not support ") +
                        node->name + " (" + ggml_op_name(node->op) + ").";
                    return abort_build(message.c_str());
                }
            }
#endif

            e.tok_stage.assign(n_slots, 0);
            e.pos_stage.assign(n_slots, 0);
            e.idx_stage.assign(static_cast<std::size_t>(kvH) * n_slots, 0);
            e.local_idx_stage.assign(n_slots, 0);
            e.embedding_stage.assign(embedding_rows ? (size_t)H * n_slots : 0, 0.f);
            e.ple_stage.assign(has_ple ? static_cast<std::size_t>(n_embd) * n_slots : 0, 0.0f);
            e.logits_stage.assign(static_cast<std::size_t>(vocab_size) * n_slots, 0.0f);
            e.sampled_stage.assign(n_slots, 0);
            e.valid = true;
        }

        Q4abEntry& e = pool.entries[entry_idx];
        pool.used[entry_idx] = ++pool.clock;
        const int n_slots = e.n_slots;
        const std::size_t row_bytes = ggml_row_size(kvType, hd);

        // ---- slot assignment ----
        std::vector<int> slot_of(n_seqs, -1);
        for (int i = 0; i < n_seqs; i++)
        {
            const void* key = k_cache_arr[i];   // first attention layer, seq i
            for (int s = 0; s < n_slots; s++)
                if (e.slots[s].active && e.slots[s].key == key) { slot_of[i] = s; break; }
        }
        // The graph advances EVERY slot's recurrent state each step (there is
        // no per-slot skip in a slot-stable graph), so an ACTIVE slot whose
        // sequence is not in this call would be phantom-advanced by a pad
        // token. Flush and retire such slots now; their sequences simply
        // rejoin later. (KV padded lanes are scratch-redirected; the GDN/PLE
        // state has no such escape.)
        for (int s = 0; s < n_slots; s++)
        {
            if (!e.slots[s].active) continue;
            bool in_call = false;
            for (int i = 0; i < n_seqs; i++)
                if (slot_of[i] == s) { in_call = true; break; }
            if (!in_call)
                q4ab_flush_and_drop_slot(e, s);
        }
        // A sequence may hold a DIRTY slot in another width's entry.
        // Flush that mapping first so the
        // join below reads current truth instead of pre-arena state.
        {
            Q4abRegistry& reg = q4ab_registry();
            for (int i = 0; i < n_seqs; i++)
            {
                if (slot_of[i] >= 0) continue;
                auto it = reg.find(k_cache_arr[i]);
                if (it != reg.end())
                {
                    // slot_of[i] < 0 means no matching slot in THIS entry, so
                    // any registry hit is a stale mapping elsewhere.
                    q4ab_flush_and_drop_slot(
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
                    set_last_error("qwen4exp arena batched decode: no free arena slot.");
                    return 0;
                }
                q4ab_flush_and_drop_slot(e, s);
            }
            slot_of[i] = s;
            Q4abSlot& sl = e.slots[s];
            sl.active = true;
            sl.key = k_cache_arr[i];
            sl.cache_rows = cache_rows[i];
            sl.len = 0;
            sl.clean = 0;
            sl.state_dirty = false;
            sl.attn_desc = attn_desc_arr[i];
            sl.gdn_desc = gdn_desc_arr[i];
            sl.ple_desc = has_ple ? ple_desc_arr[i] : nullptr;
            sl.ple_key = has_ple ? ple_state_arr[i] : nullptr;
            sl.k_hosts.resize(attn_layers);
            sl.v_hosts.resize(attn_layers);
            sl.conv_keys.resize(gdn_layers);
            sl.ssm_hosts.resize(gdn_layers);
            sl.qsa_keys.assign(n_layers, nullptr);
            for (int l = 0, al = 0, gl = 0; l < n_layers; l++)
            {
                if (kinds[l] == 0)
                {
                    if (e.qsa_arena[l]) sl.qsa_keys[l] = qsa_arr[i][l].cache;
                    sl.k_hosts[al] = k_cache_arr[static_cast<std::size_t>(al) * n_seqs + i];
                    sl.v_hosts[al] = v_cache_arr[static_cast<std::size_t>(al) * n_seqs + i];
                    al++;
                }
                else
                {
                    sl.conv_keys[gl] = conv_state_arr[static_cast<std::size_t>(gl) * n_seqs + i];
                    sl.ssm_hosts[gl] = ssm_state_arr[static_cast<std::size_t>(gl) * n_seqs + i];
                    gl++;
                }
            }
            Q4abRegistry& reg = q4ab_registry();
            for (const void* p : sl.k_hosts) reg[p] = {entry_idx, s};
            for (const void* p : sl.v_hosts) reg[p] = {entry_idx, s};
            for (const void* p : sl.conv_keys) reg[p] = {entry_idx, s};
            for (const void* p : sl.ssm_hosts) reg[p] = {entry_idx, s};
            for (const void* p : sl.qsa_keys) if (p) reg[p] = {entry_idx, s};
            if (sl.ple_key != nullptr) reg[sl.ple_key] = {entry_idx, s};
        }

        // ---- joins ----
        // KV seeds from the resident device copy when one is current (post
        // solo decode), else from the host bytes (post prefill; the read is
        // mirrored into the host so the flush appends only [clean, len)).
        // GDN/PLE state joins DEVICE-TO-DEVICE from the seq-state buffers; a
        // missing entry declines the call (the sequence never prefilled
        // through the span, so no device truth exists to join from). A !ready
        // entry (managed re-armed the seed after a reset) seeds BOTH the
        // buffer and the arena slice from the host seed and flips it ready, so
        // "ready => buffer holds current state" keeps holding at flush time.
        std::vector<char> bounce;
        bool synced = false;
        for (int i = 0; i < n_seqs; i++)
        {
            const int s = slot_of[i];
            Q4abSlot& sl = e.slots[s];
            const std::int64_t pos = cache_positions[i];
            if (sl.len > pos)
            {
                // A position rollback cannot reuse the arena: the recurrent
                // state at `pos` is NOT a prefix of the state at `len`
                // (recurrence has no rewind — and this family refuses KV
                // truncation, so a live rollback only follows a full reset,
                // which retires the slot through the flush hooks first).
                // Rejoin from scratch off the current sources.
                sl.len = 0;
                sl.clean = 0;
                sl.state_dirty = false;
            }
            if (sl.len >= pos)
                continue;
            const std::int64_t from = sl.len;
            const std::size_t bytes = static_cast<std::size_t>(pos - from) * row_bytes;
            for (int l = 0, al = 0; l < e.num_layers; l++)
            {
                if (e.k_arena[l] == nullptr) continue;
                const std::size_t cache_total = static_cast<std::size_t>(sl.cache_rows) * kvH * row_bytes;
                bounce.resize(bytes);
                for (int which = 0; which < 2; which++)
                {
                    const void* host = (which == 0) ? sl.k_hosts[al] : sl.v_hosts[al];
                    ggml_tensor* arena = (which == 0) ? e.k_arena[l] : e.v_arena[l];
                    for (int h = 0; h < kvH; h++)
                    {
                        const std::size_t src_off =
                            (static_cast<std::size_t>(h) * sl.cache_rows + static_cast<std::size_t>(from)) * row_bytes;
                        if (!q4ab_read_source(host, src_off, bytes, cache_total, bounce.data()))
                        {
                            set_last_error("qwen4exp arena batched decode: KV join source read failed.");
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
            for (int l = 0, gl = 0; l < e.num_layers; l++)
            {
                if (e.conv_arena[l] == nullptr) continue;
                // find, never create: a missing entry means this sequence never
                // prefilled through the span, and fabricating recurrent state
                // here would be wrong by construction — decline instead.
                Q4eSeqStateEntry* st = q4e_seq_state_find(sl.conv_keys[gl]);
                if (st == nullptr || st->buf == nullptr || st->bytes < ssm_off + ssmBytes)
                {
                    set_last_error("qwen4exp arena batched decode: recurrent state not resident (prefill must run through the span).");
                    return 0;
                }
                char* base = static_cast<char*>(ggml_backend_buffer_get_base(st->buf));
                if (!synced) { ggml_backend_synchronize(g_backend); synced = true; }
                if (st->ready)
                {
                    if (!q4ab_dev_copy(st->buf, base,
                        e.buffer, static_cast<char*>(e.conv_arena[l]->data) + static_cast<std::size_t>(s) * convBytes,
                        convBytes)
                        || !q4ab_dev_copy(st->buf, base + ssm_off,
                        e.buffer, static_cast<char*>(e.ssm_arena[l]->data) + static_cast<std::size_t>(s) * ssmBytes,
                        ssmBytes))
                    {
                        set_last_error("qwen4exp arena batched decode: GDN join copy failed.");
                        return 0;
                    }
                }
                else
                {
                    // Seed buffer AND arena slice from the host seeds, then
                    // mark ready: from here the device copies are the truth.
                    ggml_init_params tip = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true };
                    ggml_context* tmp = ggml_init(tip);
                    if (tmp == nullptr)
                    {
                        set_last_error("qwen4exp arena batched decode: state seed context failed.");
                        return 0;
                    }
                    ggml_tensor* ct = ggml_new_tensor_1d(tmp, GGML_TYPE_I8, static_cast<std::int64_t>(convBytes));
                    ggml_tensor* dt = ggml_new_tensor_1d(tmp, GGML_TYPE_I8, static_cast<std::int64_t>(ssmBytes));
                    bool ok = ct != nullptr && dt != nullptr &&
                        ggml_backend_tensor_alloc(st->buf, ct, base) == GGML_STATUS_SUCCESS &&
                        ggml_backend_tensor_alloc(st->buf, dt, base + ssm_off) == GGML_STATUS_SUCCESS;
                    if (ok)
                    {
                        ggml_backend_tensor_set(ct, sl.conv_keys[gl], 0, convBytes);
                        ggml_backend_tensor_set(dt, sl.ssm_hosts[gl], 0, ssmBytes);
                    }
                    ggml_free(tmp);
                    if (!ok)
                    {
                        set_last_error("qwen4exp arena batched decode: state seed upload failed.");
                        return 0;
                    }
                    ggml_backend_tensor_set(e.conv_arena[l], sl.conv_keys[gl],
                        static_cast<std::size_t>(s) * convBytes, convBytes);
                    ggml_backend_tensor_set(e.ssm_arena[l], sl.ssm_hosts[gl],
                        static_cast<std::size_t>(s) * ssmBytes, ssmBytes);
                    st->ready = true;
                }
                gl++;
            }
            if (e.ple_arena != nullptr)
            {
                Q4eSeqStateEntry* st = q4e_seq_state_find(sl.ple_key);
                if (st == nullptr || st->buf == nullptr || st->bytes < pleBytes)
                {
                    set_last_error("qwen4exp arena batched decode: PLE state not resident (prefill must run through the span).");
                    return 0;
                }
                if (!synced) { ggml_backend_synchronize(g_backend); synced = true; }
                if (st->ready)
                {
                    if (!q4ab_dev_copy(st->buf, ggml_backend_buffer_get_base(st->buf),
                        e.buffer, static_cast<char*>(e.ple_arena->data) + static_cast<std::size_t>(s) * pleBytes,
                        pleBytes))
                    {
                        set_last_error("qwen4exp arena batched decode: PLE join copy failed.");
                        return 0;
                    }
                }
                else
                {
                    ggml_init_params tip = { ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
                    ggml_context* tmp = ggml_init(tip);
                    ggml_tensor* pt = tmp != nullptr
                        ? ggml_new_tensor_1d(tmp, GGML_TYPE_I8, static_cast<std::int64_t>(pleBytes)) : nullptr;
                    bool ok = pt != nullptr &&
                        ggml_backend_tensor_alloc(st->buf, pt,
                            ggml_backend_buffer_get_base(st->buf)) == GGML_STATUS_SUCCESS;
                    if (ok)
                        ggml_backend_tensor_set(pt, sl.ple_key, 0, pleBytes);
                    if (tmp != nullptr) ggml_free(tmp);
                    if (!ok)
                    {
                        set_last_error("qwen4exp arena batched decode: PLE seed upload failed.");
                        return 0;
                    }
                    ggml_backend_tensor_set(e.ple_arena, sl.ple_key,
                        static_cast<std::size_t>(s) * pleBytes, pleBytes);
                    st->ready = true;
                }
            }
            for (int l = 0; l < n_layers; ++l)
            {
                if (!e.qsa_arena[l]) continue;
                const auto& a = qsa_arr[i][l];
                const size_t rb = ggml_row_size((ggml_type)a.cache_type, a.head_dim);
                auto* st = q4e_seq_state_find(a.cache);
                if (!st || !st->buf || st->bytes < (size_t)pos * rb)
                    throw std::runtime_error("qwen4exp arena: QSA state not resident");
                if (!synced) { ggml_backend_synchronize(g_backend); synced = true; }
                if (st->ready)
                {
                    if (!q4ab_dev_copy(st->buf, ggml_backend_buffer_get_base(st->buf), e.buffer,
                        (char*)e.qsa_arena[l]->data + (size_t)s * e.cap * rb, (size_t)pos * rb))
                        throw std::runtime_error("qwen4exp arena: QSA join copy failed");
                }
                else
                {
                    ggml_init_params ip{ggml_tensor_overhead() * 2, nullptr, true};
                    std::unique_ptr<ggml_context, decltype(&ggml_free)> tmp(ggml_init(ip), ggml_free);
                    if (!tmp) throw std::bad_alloc();
                    auto* raw = ggml_new_tensor_1d(tmp.get(), GGML_TYPE_I8, a.cache_bytes);
                    if (ggml_backend_tensor_alloc(st->buf, raw, ggml_backend_buffer_get_base(st->buf)) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("qwen4exp arena: QSA seed bind failed");
                    ggml_backend_tensor_set(raw, a.cache, 0, a.cache_bytes);
                    ggml_backend_tensor_set(e.qsa_arena[l], a.cache, (size_t)s * e.cap * rb, (size_t)pos * rb);
                    st->ready = true;
                }
            }
            sl.len = pos;
            sl.clean = pos;
        }

        // ---- per-step inputs (slot order) ----
        std::vector<std::int64_t> pos_by_slot(n_slots, -1);
        std::fill(e.tok_stage.begin(), e.tok_stage.end(), 0);
        std::fill(e.pos_stage.begin(), e.pos_stage.end(), 0);
        std::fill(e.local_idx_stage.begin(), e.local_idx_stage.end(), 0);
        std::fill(e.embedding_stage.begin(), e.embedding_stage.end(), 0.f);
        if (!e.ple_stage.empty())
            std::fill(e.ple_stage.begin(), e.ple_stage.end(), 0.0f);
        for (int s = 0; s < n_slots; s++)
            for (int h = 0; h < kvH; h++)
                e.idx_stage[static_cast<std::size_t>(s) * kvH + h] =
                    static_cast<std::int64_t>(n_slots) * e.rows_per_slot +
                    static_cast<std::int64_t>(h) * e.cap + s;   // scratch slice
        for (int i = 0; i < n_seqs; i++)
        {
            const int s = slot_of[i];
            pos_by_slot[s] = cache_positions[i];
            e.tok_stage[s] = token_ids[i];
            e.pos_stage[s] = rope_positions[i];
            e.local_idx_stage[s] = cache_positions[i];
            if (e.embedding_in)
                std::memcpy(e.embedding_stage.data() + (size_t)s * H, embedding_rows + (size_t)i * H, (size_t)H * sizeof(float));
            for (int h = 0; h < kvH; h++)
                e.idx_stage[static_cast<std::size_t>(s) * kvH + h] =
                    static_cast<std::int64_t>(s) * e.rows_per_slot +
                    static_cast<std::int64_t>(h) * e.cap + cache_positions[i];
            if (!e.ple_stage.empty())
                std::memcpy(&e.ple_stage[static_cast<std::size_t>(s) * n_embd],
                            ple_emb + static_cast<std::size_t>(i) * n_embd,
                            static_cast<std::size_t>(n_embd) * sizeof(float));
        }

        host_read_barrier();

        if (e.token_in->buffer)
            decode_input_set_async(e.token_in, e.tok_stage.data(), e.tok_stage.size() * sizeof(std::int32_t));
        decode_input_set_async(e.rope_pos_in, e.pos_stage.data(), e.pos_stage.size() * sizeof(std::int32_t));
        if (e.idx_in->buffer)
            decode_input_set_async(e.idx_in, e.idx_stage.data(), e.idx_stage.size() * sizeof(std::int64_t));
        if (e.local_idx_in->buffer)
            decode_input_set_async(e.local_idx_in, e.local_idx_stage.data(), e.local_idx_stage.size() * sizeof(std::int64_t));
        if (e.embedding_in)
            decode_input_set_async(e.embedding_in, e.embedding_stage.data(), e.embedding_stage.size() * sizeof(float));
        q4ab_fill_mask(e.mask_stage, e.cap, n_slots, pos_by_slot.data());
        decode_input_set_async(e.attn_mask, e.mask_stage.data(), e.mask_stage.size() * sizeof(ggml_fp16_t));
        if (e.ple_emb_in != nullptr)
            decode_input_set_async(e.ple_emb_in, e.ple_stage.data(), e.ple_stage.size() * sizeof(float));

        std::map<std::pair<int, int>, Q4eQsaPlan> plans;
        for (int l = 0; l < n_layers; ++l)
        {
            if (!e.qsa_arena[l]) continue;
            for (int s = 0; s < n_slots; ++s)
            {
                const auto& in = e.qsa_inputs[l][s];
                // Dense short contexts only append raw keys; the sparse
                // selector has no graph inputs and needs no history plan.
                if (!in.cell_blocks->buffer) continue;
                int caller = -1;
                for (int i = 0; i < n_seqs; ++i) if (slot_of[i] == s) { caller = i; break; }
                const int32_t dummy[3] = {0, 0, 0};
                const auto key = std::make_pair(caller, e.qsa_args[l].ratio);
                auto found = plans.find(key);
                if (found == plans.end()) found = plans.emplace(key,
                    q4e_qsa_plan(caller >= 0 ? qsa_positions_arr[caller] : dummy,
                        caller >= 0 ? qsa_position_counts[caller] : 1, e.attention_windows[s],
                        caller >= 0 ? cache_positions[caller] : 0, 1, key.second)).first;
                const auto& plan = found->second;
                auto put = [](ggml_tensor* t, const auto& values) {
                    if (t && t->buffer)
                        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(values[0]));
                };
                put(in.cell_blocks, plan.cell_blocks); put(in.block_cells, plan.block_cells);
                put(in.block_positions, plan.block_positions); put(in.query_positions, plan.query_positions); put(in.bias, plan.bias);
            }
        }

        compute_started = true;
        const auto execution_backend = e.execution_backend();
        const bool succeeded = e.host_moe.empty()
            ? tsg::graph_compute_profiled(execution_backend, e.graph, kQ4abKernel) == GGML_STATUS_SUCCESS
            : host_moe_execute_segments(e.graph, e.host_moe, e.host_moe_seg_end, kQ4abKernel, execution_backend);
        if (!succeeded)
        {
            set_last_error("qwen4exp arena batched decode: graph execution failed.");
            // A partially executed graph may have advanced SOME layers'
            // GDN/PLE state for this call's slots and not others — that mixed
            // state is unrecoverable, so those slots drop WITHOUT a state
            // flush. Managed code fails these holders and never retries them;
            // partially advanced recurrent state cannot be served safely.
            // Only the previously committed KV rows [clean, len) are flushed.
            for (int i = 0; i < n_seqs; i++)
                e.slots[slot_of[i]].state_dirty = false;
            q4ab_flush_entry(e);
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

        if (trace) for (size_t i = 0; i < e.trace_res.size(); ++i)
        {
            auto* t = e.trace_res[i];
            std::vector<float> values((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
            for (int caller = 0; caller < n_seqs; ++caller)
            {
                const float* row = values.data() + (size_t)slot_of[caller] * hc_dim;
                double norm = 0;
                uint64_t hash = 1469598103934665603ULL;
                for (int c = 0; c < hc_dim; ++c) { norm += (double)row[c] * row[c]; uint32_t bits; std::memcpy(&bits, row + c, sizeof(bits)); hash = (hash ^ bits) * 1099511628211ULL; }
                fprintf(stderr, "[q4e-arena-trace] row=%d pos=%d l=%zu half=%zu l2=%.9e hash=%016llx head=%.9e %.9e\n",
                    caller, cache_positions[caller], i / 2, i % 2, std::sqrt(norm), (unsigned long long)hash, row[0], row[1]);
            }
        }

        if (want_logits != 0 && logits_data != nullptr)
        {
            for (int i = 0; i < n_seqs; i++)
            {
                const int s = slot_of[i];
                std::memcpy(logits_data + static_cast<std::size_t>(i) * vocab_size,
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
            Q4abSlot& sl = e.slots[slot_of[i]];
            sl.len = cache_positions[i] + 1;
            sl.state_dirty = true;
            sl.last_used = ++pool.clock;
        }

        clear_last_error();
        return 1;
    }
    // A failed flush can leave solo state only partially copied, even before
    // this call computes. Do not let callers retry the batch as solo decode.
    catch (const Q4abFlushFailure& ex) { set_last_error(ex.what()); return -1; }
    catch (const std::exception& ex) { set_last_error(ex.what()); return compute_started ? -1 : 0; }
    catch (...) { set_last_error("Unknown error in qwen4exp arena batched decode."); return compute_started ? -1 : 0; }
}
