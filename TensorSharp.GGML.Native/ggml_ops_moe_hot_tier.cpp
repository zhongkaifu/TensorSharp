// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ============================================================================
// Priority-ranked accelerator tier for host-offloaded routed experts, and the
// routing counters that rank them.
//
// A layer whose experts run on the host (MoE CPU offload) can keep a compact
// device copy of its most frequently routed experts. At the host seam the
// routes split: a route whose expert is in the tier is computed by the
// accelerator, inside the same token graph, from the compact copy; every other
// route runs on the host as before. The two partial sums meet in the graph.
// This is the "fixed region" of an expert cache whose members are chosen by a
// calibrated routing priority (NInfer's Flash Next expert cache uses the same
// idea on CUDA); on unified memory it also decides which experts occupy RAM:
// a whole wired layer pins all of its experts, used or not, while a tier pins
// only the ones that are routed most.
//
// The tier is keyed by the mapped address of the layer's gate weight, which is
// what every host seam already carries, so a model opts in by registering its
// tiers after loading and by building the accelerator half with
// host_moe_build_hot_part. The tier never changes routing or route weights.
//
// Routing counters (also keyed by the gate address) accumulate every routed
// id a host seam sees. A model reads them back to persist a routing profile,
// which is what the next load ranks its tier members by.
// ============================================================================

#include "ggml_ops_internal.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tsg
{
    namespace
    {
        std::mutex g_tier_mutex;
        std::vector<std::unique_ptr<MoeHotTier>> g_tiers;
        std::size_t g_tier_bytes = 0;

        std::mutex g_count_mutex;
        std::unordered_map<const void*, std::vector<std::uint64_t>> g_counts;

        void free_tier(MoeHotTier& tier)
        {
            if (tier.buffer != nullptr)
            {
                // A graph replayed after this point would read freed memory; the
                // owners drop their persistent graphs through the same
                // invalidation hook that calls this.
                if (tier.backend != nullptr) sync_backend(tier.backend);
                ggml_backend_buffer_free(tier.buffer);
                tier.buffer = nullptr;
            }
        }

        // Clean file-backed pages of a copied expert are now a duplicate of the
        // tier; let the VM reclaim them first so RAM goes to experts that are
        // still read from the mapping.
        void release_source_pages(const void* begin, std::size_t bytes)
        {
#if defined(__APPLE__)
            static const long page = sysconf(_SC_PAGESIZE);
            if (page <= 0 || begin == nullptr || bytes == 0) return;
            auto start = reinterpret_cast<std::uintptr_t>(begin);
            auto end = start + bytes;
            // Only whole pages inside this expert: a neighbour may share an edge page.
            start = (start + page - 1) / page * page;
            end = end / page * page;
            if (end > start)
                madvise(reinterpret_cast<void*>(start), end - start, MADV_DONTNEED);
#else
            (void)begin; (void)bytes;
#endif
        }
    }

    MoeHotTier* moe_hot_tier_find(const void* gate_source)
    {
        if (gate_source == nullptr || g_backend == nullptr) return nullptr;
        std::lock_guard<std::mutex> lock(g_tier_mutex);
        for (const auto& tier : g_tiers)
            if (tier->source[0] == gate_source && tier->backend == g_backend)
                return tier.get();
        return nullptr;
    }

    void moe_hot_tier_bind(const MoeHotTier& tier, int part, ggml_tensor* tensor)
    {
        if (tensor == nullptr || part < 0 || part > 2 || tier.buffer == nullptr)
            throw std::invalid_argument("MoE hot tier: invalid binding.");
        auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(tier.buffer));
        if (ggml_nbytes(tensor) != tier.stride[part] * static_cast<std::size_t>(tier.count))
            throw std::invalid_argument("MoE hot tier: compact tensor shape does not match its tier.");
        if (ggml_backend_tensor_alloc(tier.buffer, tensor, base + tier.offset[part]) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MoE hot tier: could not bind the compact expert tensor.");
    }

    ggml_tensor* host_moe_build_hot_part(ggml_context* ctx, HostMoeSegment& hm, ggml_tensor* moe_in,
        MoeHotTier& tier, MulMatIdFn mul_mat_id, int overlap_routes, ggml_cgraph* graph, HostMoeSegment* wait_out)
    {
        const int T = hm.seq_len, n_used = hm.n_used;
        hm.hot_tier = &tier;
        auto pinned_input = [&](ggml_tensor* t) {
            // Written by the host at the seam, read by a later segment: inputs that
            // must also stay pinned for the whole pass (see moe_out).
            ggml_set_input(t);
            ggml_set_output(t);
            return t;
        };
        hm.hot_slot = pinned_input(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T));
        hm.hot_weight = pinned_input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, T));
        ggml_tensor* w[3] = {
            ggml_new_tensor_3d(ctx, static_cast<ggml_type>(tier.type[0]), hm.hidden, hm.n_ff, tier.count),
            ggml_new_tensor_3d(ctx, static_cast<ggml_type>(tier.type[1]), hm.hidden, hm.n_ff, tier.count),
            ggml_new_tensor_3d(ctx, static_cast<ggml_type>(tier.type[2]), hm.n_ff, hm.hidden, tier.count),
        };
        for (int p = 0; p < 3; ++p) moe_hot_tier_bind(tier, p, w[p]);
        if (mul_mat_id == nullptr) mul_mat_id = ggml_mul_mat_id;

        ggml_tensor* x = ggml_reshape_3d(ctx, moe_in, hm.hidden, 1, T);
        auto routes = [&](ggml_tensor* slot, ggml_tensor* weight, int n) {
            ggml_tensor* gate = mul_mat_id(ctx, w[0], x, slot);
            ggml_tensor* up = mul_mat_id(ctx, w[1], x, slot);
            ggml_tensor* activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* down = mul_mat_id(ctx, w[2], activation, slot);   // [hidden, n, T]
            down = ggml_mul(ctx, down, weight);
            ggml_tensor* sum = ggml_view_2d(ctx, down, hm.hidden, T, down->nb[2], 0);
            for (int k = 1; k < n; ++k)
                sum = ggml_add(ctx, sum, ggml_view_2d(ctx, down, hm.hidden, T, down->nb[2],
                    static_cast<std::size_t>(k) * down->nb[1]));
            return sum;
        };
        ggml_tensor* sum = routes(hm.hot_slot, hm.hot_weight, n_used);

        // Verify-width overlap: the routes above cover what the tier holds when the
        // seam runs; up to overlap_routes window misses a row are read in the
        // background meanwhile and computed below, after a second cut that only
        // waits for those reads.
        const bool overlap = overlap_routes > 0 && graph != nullptr && wait_out != nullptr
            && tier.count > tier.fixed && tier.cpu_visible && tier.count >= n_used + overlap_routes;
        if (!overlap) return sum;
        ggml_build_forward_expand(graph, sum);       // the first group precedes the cut
        *wait_out = HostMoeSegment{};
        wait_out->io_wait = true;
        wait_out->layer = hm.layer;
        wait_out->seq_len = T;
        wait_out->moe_in = wait_out->sel_ids = wait_out->weights = sum;
        hm.hot_b = overlap_routes;
        hm.hot_slot_b = pinned_input(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, overlap_routes, T));
        hm.hot_weight_b = pinned_input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, overlap_routes, T));
        return ggml_add(ctx, sum, routes(hm.hot_slot_b, hm.hot_weight_b, overlap_routes));
    }

    bool host_moe_split_hot_routes(const HostMoeSegment& hm, const std::int32_t* ids, const float* weights,
        std::vector<std::int32_t>& cold_ids, std::vector<float>& cold_weights, std::vector<std::uint8_t>& cold_pad,
        int& cold_used)
    {
        MoeHotTier& tier = *hm.hot_tier;
        const int T = hm.seq_len, n_used = hm.n_used;
        const std::size_t routes = static_cast<std::size_t>(T) * n_used;
        for (std::size_t i = 0; i < routes; ++i)
            if (ids[i] < 0 || ids[i] >= tier.num_experts) return false;

        // Window: give every routed expert that has no slot one, evicting the
        // least recently used window slot this step does not use, and read all
        // of this step's misses together straight into their slots. Routes left
        // without a slot (a step wider than the window) run on the host.
        const std::uint64_t step = ++tier.step;
        for (std::size_t i = 0; i < routes; ++i)
        {
            const int slot = tier.slot_of[ids[i]];
            if (slot >= 0 && tier.step_used[slot] != step)
            {
                tier.step_used[slot] = step;
                tier.last_use[slot] = ++tier.clock;
                ++tier.hits;
            }
        }
        // With an overlap group (hm.hot_b), a miss whose every row still has room
        // in it is read in the background instead: the accelerator computes the
        // routes it already holds meanwhile, and these after the io_wait cut.
        static thread_local std::vector<int> s_deferred;      // slots being read in the background
        static thread_local std::vector<int> s_row_b;         // deferred routes a row has
        s_deferred.clear();
        s_row_b.assign(T, 0);
        auto is_deferred = [&](int slot) {
            return std::find(s_deferred.begin(), s_deferred.end(), slot) != s_deferred.end();
        };
        if (tier.count > tier.fixed && tier.cpu_visible)
        {
            static thread_local std::vector<const void*> s_src, a_src;
            static thread_local std::vector<std::size_t> s_len, a_len;
            static thread_local std::vector<std::uint8_t*> s_dst, a_dst;
            static thread_local std::vector<int> s_loaded;
            s_src.clear(); s_len.clear(); s_dst.clear(); s_loaded.clear();
            a_src.clear(); a_len.clear(); a_dst.clear();
            auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(tier.buffer));
            for (std::size_t i = 0; i < routes; ++i)
            {
                const int e = ids[i];
                if (tier.slot_of[e] >= 0) continue;
                int victim = -1;
                for (int s2 = tier.fixed; s2 < tier.count; ++s2)
                    if (tier.step_used[s2] != step && (victim < 0 || tier.last_use[s2] < tier.last_use[victim]))
                        victim = s2;
                if (victim < 0) { ++tier.overflow; break; }
                if (tier.expert_of[victim] >= 0) tier.slot_of[tier.expert_of[victim]] = -1;
                tier.expert_of[victim] = e;
                tier.slot_of[e] = victim;
                tier.step_used[victim] = step;
                tier.last_use[victim] = ++tier.clock;
                ++tier.misses;
                bool defer = hm.hot_b > 0;
                for (int r = 0; r < T && defer; ++r)
                    for (int k = 0; k < n_used; ++k)
                        if (ids[static_cast<std::size_t>(r) * n_used + k] == e && s_row_b[r] >= hm.hot_b) { defer = false; break; }
                if (defer)
                {
                    for (int r = 0; r < T; ++r)
                        for (int k = 0; k < n_used; ++k)
                            if (ids[static_cast<std::size_t>(r) * n_used + k] == e) ++s_row_b[r];
                    s_deferred.push_back(victim);
                    ++tier.deferred;
                }
                else
                    s_loaded.push_back(victim);
                for (int p = 0; p < 3; ++p)
                {
                    const void* src = static_cast<const std::uint8_t*>(tier.source[p]) + static_cast<std::size_t>(e) * tier.stride[p];
                    std::uint8_t* dst = base + tier.offset[p] + static_cast<std::size_t>(victim) * tier.stride[p];
                    (defer ? a_src : s_src).push_back(src);
                    (defer ? a_len : s_len).push_back(tier.stride[p]);
                    (defer ? a_dst : s_dst).push_back(dst);
                }
            }
            if (!s_src.empty() && !host_expert_io_read_slices(s_src.data(), s_len.data(), s_dst.data(),
                    static_cast<int>(s_src.size())))
            {
                // A failed read leaves those slots undefined: empty them, and
                // their routes run on the host from the mapping instead.
                for (int v : s_loaded)
                {
                    tier.slot_of[tier.expert_of[v]] = -1;
                    tier.expert_of[v] = -1;
                    tier.last_use[v] = 0;
                }
            }
            // The background batch starts once the reads the first group needs are in.
            if (!a_src.empty())
                host_expert_io_read_slices_async(a_src.data(), a_len.data(), a_dst.data(), static_cast<int>(a_src.size()));
        }

        static thread_local std::vector<std::int32_t> s_slots, s_slots_b;
        static thread_local std::vector<float> s_hot_w, s_hot_w_b;
        static thread_local std::vector<int> s_cold_count;
        s_slots.assign(routes, 0);
        s_hot_w.assign(routes, 0.0f);
        s_cold_count.assign(T, 0);
        const std::size_t routes_b = static_cast<std::size_t>(T) * hm.hot_b;
        s_slots_b.assign(routes_b, 0);
        s_hot_w_b.assign(routes_b, 0.0f);
        int first_cold = -1;
        cold_used = 0;
        for (int r = 0; r < T; ++r)
        {
            const std::int32_t* row = ids + static_cast<std::size_t>(r) * n_used;
            std::int32_t* row_slots = s_slots.data() + static_cast<std::size_t>(r) * n_used;
            std::int32_t* row_b = s_slots_b.data() + static_cast<std::size_t>(r) * hm.hot_b;
            int nb = 0;
            // A route that the tier does not hold (or whose slot is still being
            // read) still needs a valid slot for the accelerator's mul_mat_id, with
            // weight 0. It takes a slot no other route of this row uses (ggml-cuda's
            // batched MUL_MAT_ID breaks on an expert repeated within a token) and no
            // slot a background read is writing (0 x a half-written NaN is NaN).
            int next_free = 0;
            for (int k = 0; k < n_used; ++k)
            {
                const std::size_t i = static_cast<std::size_t>(r) * n_used + k;
                const int slot = tier.slot_of[row[k]];
                if (slot >= 0 && !is_deferred(slot)) { row_slots[k] = slot; s_hot_w[i] = weights[i]; continue; }
                if (slot >= 0)
                {
                    row_b[nb] = slot;
                    s_hot_w_b[static_cast<std::size_t>(r) * hm.hot_b + nb] = weights[i];
                    ++nb;
                }
                for (;;)
                {
                    // The graph hook only binds tiers with >= n_used slots, so a free
                    // one always exists; the wrap is a guard, never a duplicate path.
                    if (next_free >= tier.count) { next_free = 0; break; }
                    bool taken = is_deferred(next_free);
                    for (int j = 0; j < n_used && !taken; ++j)
                        taken = tier.slot_of[row[j]] == next_free;
                    for (int j = 0; j < k && !taken; ++j)
                        taken = row_slots[j] == next_free;
                    if (!taken) break;
                    ++next_free;
                }
                row_slots[k] = next_free++;
                if (slot >= 0) continue;          // deferred: computed in the second group
                ++s_cold_count[r];
                if (first_cold < 0) first_cold = row[k];
            }
            // The second group's unused entries: weight 0 on distinct slots of this
            // row's first group (just read there), every slot whole by then.
            for (int k = 0; k < n_used && nb < hm.hot_b; ++k)
            {
                const int slot = row_slots[k];
                if (s_hot_w[static_cast<std::size_t>(r) * n_used + k] == 0.0f) continue;
                bool dup = false;
                for (int j = 0; j < nb && !dup; ++j) dup = row_b[j] == slot;
                if (!dup) row_b[nb++] = slot;
            }
            for (int cand = 0; nb < hm.hot_b && cand < tier.count; ++cand)
            {
                bool dup = false;
                for (int j = 0; j < nb && !dup; ++j) dup = row_b[j] == cand;
                if (!dup) row_b[nb++] = cand;
            }
            cold_used = std::max(cold_used, s_cold_count[r]);
        }
        ggml_backend_tensor_set(hm.hot_slot, s_slots.data(), 0, routes * sizeof(std::int32_t));
        ggml_backend_tensor_set(hm.hot_weight, s_hot_w.data(), 0, routes * sizeof(float));
        if (hm.hot_b > 0)
        {
            ggml_backend_tensor_set(hm.hot_slot_b, s_slots_b.data(), 0, routes_b * sizeof(std::int32_t));
            ggml_backend_tensor_set(hm.hot_weight_b, s_hot_w_b.data(), 0, routes_b * sizeof(float));
        }

        cold_ids.clear();
        cold_weights.clear();
        cold_pad.clear();
        if (cold_used == 0) return true;
        // Rows share one route count on the host kernels: pad a short row with a
        // cold expert it already reads (or the batch's first cold expert), weight 0.
        cold_ids.resize(static_cast<std::size_t>(T) * cold_used);
        cold_weights.assign(static_cast<std::size_t>(T) * cold_used, 0.0f);
        cold_pad.assign(static_cast<std::size_t>(T) * cold_used, 0);
        for (int r = 0; r < T; ++r)
        {
            const std::int32_t* row = ids + static_cast<std::size_t>(r) * n_used;
            int n = 0, pad = first_cold;
            for (int k = 0; k < n_used; ++k)
            {
                const int e = row[k];
                if (tier.slot_of[e] >= 0) continue;
                if (n == 0) pad = e;
                cold_ids[static_cast<std::size_t>(r) * cold_used + n] = e;
                cold_weights[static_cast<std::size_t>(r) * cold_used + n] = weights[static_cast<std::size_t>(r) * n_used + k];
                ++n;
            }
            for (; n < cold_used; ++n)
            {
                cold_ids[static_cast<std::size_t>(r) * cold_used + n] = pad;
                cold_pad[static_cast<std::size_t>(r) * cold_used + n] = 1;
            }
        }
        return true;
    }

    void moe_hot_tier_release()
    {
        std::lock_guard<std::mutex> lock(g_tier_mutex);
        for (auto& tier : g_tiers) free_tier(*tier);
        g_tiers.clear();
        g_tier_bytes = 0;
    }

    // Set while a model whose routing profile does not yet cover every layer runs a
    // plan that keeps some MoE layers on the accelerator: those layers have no host
    // seam, so their graphs hand their router ids to the counters instead.
    static std::atomic<bool> g_record_device_routes{false};
    bool moe_record_device_routes() { return g_record_device_routes.load(std::memory_order_relaxed); }
    void moe_set_record_device_routes(bool on) { g_record_device_routes.store(on, std::memory_order_relaxed); }

    void moe_routing_count(const void* gate_source, int num_experts, const std::int32_t* ids, std::size_t count)
    {
        if (gate_source == nullptr || ids == nullptr || num_experts <= 0 || count == 0) return;
        std::lock_guard<std::mutex> lock(g_count_mutex);
        auto& counts = g_counts[gate_source];
        if (counts.size() != static_cast<std::size_t>(num_experts)) counts.assign(num_experts, 0);
        for (std::size_t i = 0; i < count; ++i)
            if (ids[i] >= 0 && ids[i] < num_experts) ++counts[ids[i]];
    }

    void moe_hot_tier_on_drop(const void* ptr)
    {
        if (ptr == nullptr) return;
        {
            std::lock_guard<std::mutex> lock(g_tier_mutex);
            for (auto it = g_tiers.begin(); it != g_tiers.end();)
            {
                auto& tier = **it;
                if (tier.source[0] == ptr || tier.source[1] == ptr || tier.source[2] == ptr)
                {
                    g_tier_bytes -= ggml_backend_buffer_get_size(tier.buffer);
                    free_tier(tier);
                    it = g_tiers.erase(it);
                }
                else ++it;
            }
        }
        std::lock_guard<std::mutex> lock(g_count_mutex);
        g_counts.erase(ptr);
    }
}

// Give one host-offloaded layer accelerator expert slots: `count` fixed slots
// holding `experts` (most-routed first, no repeats) for the tier's lifetime,
// plus `window` slots the seams fill on demand. Returns 1 on success; 0 leaves
// the layer without a tier (every route keeps running on the host). Replaces
// an existing tier for the same gate address; call before graphs that use the
// layer are built, or drop them (TSGgml_InvalidateHostBuffer) first.
TSG_EXPORT int TSGgml_MoeHotTierRegister(
    void* gate, void* up, void* down,
    int gate_type, int up_type, int down_type,
    long long gate_bytes, long long up_bytes, long long down_bytes,
    int num_experts, const int* experts, int count, int window, int release_source)
{
    using namespace tsg;
    try
    {
        if (g_backend == nullptr || gate == nullptr || up == nullptr || down == nullptr
            || num_experts <= 0 || count < 0 || window < 0 || count + window <= 0
            || count + window > num_experts || (count > 0 && experts == nullptr)
            || gate_bytes <= 0 || up_bytes <= 0 || down_bytes <= 0
            || gate_bytes % num_experts || up_bytes % num_experts || down_bytes % num_experts)
        {
            set_last_error("MoE hot tier: invalid registration.");
            return 0;
        }
        moe_hot_tier_on_drop(gate);   // replace, and keep the counters keyed below fresh
        auto tier = std::make_unique<MoeHotTier>();
        tier->backend = g_backend;
        const int slots = count + window;
        void* source[3] = {gate, up, down};
        const int types[3] = {gate_type, up_type, down_type};
        const long long bytes[3] = {gate_bytes, up_bytes, down_bytes};
        std::size_t total = 0;
        for (int p = 0; p < 3; ++p)
        {
            if (types[p] < 0 || types[p] >= GGML_TYPE_COUNT)
            {
                set_last_error("MoE hot tier: unknown weight type.");
                return 0;
            }
            tier->source[p] = source[p];
            tier->type[p] = types[p];
            tier->stride[p] = static_cast<std::size_t>(bytes[p]) / num_experts;
            total = (total + 255) / 256 * 256;
            tier->offset[p] = total;
            // Backends may allocate more than the raw bytes (ggml-cuda pads the end of
            // a quantized matrix for its MMQ kernels): leave room after every part.
            constexpr std::size_t kPartSlack = std::size_t(1) << 20;
            total += tier->stride[p] * static_cast<std::size_t>(slots) + kPartSlack;
        }
        tier->num_experts = num_experts;
        tier->count = slots;
        tier->fixed = count;
        tier->slot_of.assign(num_experts, -1);
        tier->expert_of.assign(slots, -1);
        tier->last_use.assign(slots, 0);
        tier->step_used.assign(slots, 0);
        for (int s = 0; s < count; ++s)
        {
            const int e = experts[s];
            if (e < 0 || e >= num_experts || tier->slot_of[e] >= 0)
            {
                set_last_error("MoE hot tier: expert list has an out-of-range or repeated id.");
                return 0;
            }
            tier->slot_of[e] = s;
            tier->expert_of[s] = e;
        }
        tier->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(g_backend), total);
        if (tier->buffer == nullptr)
        {
            set_last_error("MoE hot tier: the accelerator could not allocate the compact expert buffer.");
            return 0;
        }
        ggml_backend_buffer_set_usage(tier->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // The window is filled from the host between segments, so it needs slots
        // the CPU can write in place: a host buffer, or Metal shared (unified)
        // memory. Elsewhere only the fixed slots are used.
        {
            const char* name = ggml_backend_buft_name(ggml_backend_buffer_get_type(tier->buffer));
            tier->cpu_visible = ggml_backend_buffer_is_host(tier->buffer)
                || (name != nullptr && std::strncmp(name, "MTL", 3) == 0 && std::strstr(name, "_Private") == nullptr
                    && std::strstr(name, "_Mapped") == nullptr);
            if (!tier->cpu_visible) tier->count = tier->fixed;
        }
        if (tier->count == 0)
        {
            set_last_error("MoE hot tier: no fixed experts and no host-writable window.");
            return 0;
        }

        auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(tier->buffer));
        if (tier->cpu_visible)
        {
            // Read the fixed experts straight into their slots (explicit reads
            // when the mapping is registered, otherwise copied from it).
            std::vector<const void*> src;
            std::vector<std::size_t> len;
            std::vector<std::uint8_t*> dst;
            for (int p = 0; p < 3; ++p)
                for (int s = 0; s < count; ++s)
                {
                    src.push_back(static_cast<const std::uint8_t*>(source[p])
                        + static_cast<std::size_t>(experts[s]) * tier->stride[p]);
                    len.push_back(tier->stride[p]);
                    dst.push_back(base + tier->offset[p] + static_cast<std::size_t>(s) * tier->stride[p]);
                }
            if (!src.empty() && !host_expert_io_read_slices(src.data(), len.data(), dst.data(), static_cast<int>(src.size())))
            {
                set_last_error("MoE hot tier: reading the fixed experts failed.");
                ggml_backend_buffer_free(tier->buffer);
                return 0;
            }
            if (release_source)
                for (std::size_t i = 0; i < src.size(); ++i) release_source_pages(src[i], len[i]);
        }
        else
        {
            // Fault the selected experts in on many threads first: a page fault per
            // 16 KB on one copying thread runs far below the SSD's rate.
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            for (int p = 0; p < 3; ++p)
            {
                ranges.clear();
                for (int s = 0; s < count; ++s)
                    ranges.emplace_back(static_cast<std::size_t>(experts[s]) * tier->stride[p], tier->stride[p]);
                std::sort(ranges.begin(), ranges.end());
                prefetch_mapped_ranges(static_cast<const std::uint8_t*>(source[p]), ranges);
                ggml_tensor view{};
                view.type = GGML_TYPE_I8;
                view.ne[0] = static_cast<std::int64_t>(tier->stride[p]) * slots;
                view.ne[1] = view.ne[2] = view.ne[3] = 1;
                view.nb[0] = 1; view.nb[1] = view.nb[2] = view.nb[3] = static_cast<std::size_t>(view.ne[0]);
                view.buffer = tier->buffer;
                view.data = base + tier->offset[p];
                for (int s = 0; s < count; ++s)
                {
                    const auto* src = static_cast<const std::uint8_t*>(source[p])
                        + static_cast<std::size_t>(experts[s]) * tier->stride[p];
                    ggml_backend_tensor_set(&view, src, static_cast<std::size_t>(s) * tier->stride[p], tier->stride[p]);
                    if (release_source) release_source_pages(src, tier->stride[p]);
                }
            }
        }
        std::lock_guard<std::mutex> lock(g_tier_mutex);
        g_tier_bytes += ggml_backend_buffer_get_size(tier->buffer);
        g_tiers.push_back(std::move(tier));
        return 1;
    }
    catch (const std::exception& e)
    {
        set_last_error(std::string("MoE hot tier: ") + e.what());
        return 0;
    }
}

// (fixed hits + window hits, window misses, routes left to the host) over every tier.
// hits, window reads, overflow steps, window reads done in the background.
TSG_EXPORT void TSGgml_MoeHotTierStats(long long* stats4)
{
    if (stats4 == nullptr) return;
    std::lock_guard<std::mutex> lock(tsg::g_tier_mutex);
    stats4[0] = stats4[1] = stats4[2] = stats4[3] = 0;
    for (const auto& t : tsg::g_tiers)
    {
        stats4[0] += static_cast<long long>(t->hits);
        stats4[1] += static_cast<long long>(t->misses);
        stats4[2] += static_cast<long long>(t->overflow);
        stats4[3] += static_cast<long long>(t->deferred);
    }
}

// Free the tier of the layer whose gate stack is mapped at `gate` (null: every
// tier). The caller must already have dropped the graphs that bind it.
TSG_EXPORT void TSGgml_MoeHotTierRelease(const void* gate)
{
    if (gate == nullptr) { tsg::moe_hot_tier_release(); return; }
    std::lock_guard<std::mutex> lock(tsg::g_tier_mutex);
    for (auto it = tsg::g_tiers.begin(); it != tsg::g_tiers.end();)
    {
        if ((*it)->source[0] == gate)
        {
            tsg::g_tier_bytes -= ggml_backend_buffer_get_size((*it)->buffer);
            tsg::free_tier(**it);
            it = tsg::g_tiers.erase(it);
        }
        else ++it;
    }
}

TSG_EXPORT long long TSGgml_MoeHotTierTotalBytes()
{
    std::lock_guard<std::mutex> lock(tsg::g_tier_mutex);
    return static_cast<long long>(tsg::g_tier_bytes);
}

// Routed-id counts the host seams saw for the layer whose gate weight is
// mapped at `gate`. Writes min(capacity, experts) counts; returns the number
// of experts counted (0 when the layer never reached a host seam).
TSG_EXPORT int TSGgml_MoeRoutingCounts(const void* gate, unsigned long long* counts, int capacity)
{
    std::lock_guard<std::mutex> lock(tsg::g_count_mutex);
    auto it = tsg::g_counts.find(gate);
    if (it == tsg::g_counts.end()) return 0;
    const int n = static_cast<int>(it->second.size());
    if (counts != nullptr)
        for (int i = 0; i < std::min(n, capacity); ++i) counts[i] = it->second[i];
    return n;
}

TSG_EXPORT void TSGgml_MoeRoutingRecordDevice(int on)
{
    tsg::moe_set_record_device_routes(on != 0);
}

TSG_EXPORT void TSGgml_MoeRoutingReset(const void* gate)
{
    std::lock_guard<std::mutex> lock(tsg::g_count_mutex);
    if (gate == nullptr) tsg::g_counts.clear();
    else tsg::g_counts.erase(gate);
}
